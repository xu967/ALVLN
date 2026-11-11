#!/usr/bin/env python3
"""--action-latency（W1-0「动作派发时刻固定到仿真时间」）的测试。不起 Gazebo。

背景：--window-from observation 下动作窗口从观测位姿戳起算，执行器截止时刻 = 观测戳 + hold；
观测到派发之间的延迟（相机帧龄 + 策略段的墙钟耗时 × RTF）每步不同，动作实际生效时长 = hold − 延迟，
K=1 柱阵 0.871 s、K=3 长课程 0.77–0.79 s（2026-11-10 长课程第二轮）。给 --action-latency L 之后：
策略出动作后等位姿话题的仿真戳 ≥ 观测戳 + L 再派发（生效时长固定为 hold − L）；动作就绪时仿真戳已超过
观测戳 + L + 0.02 s 的步记「迟到」。每步行记 dispatch_lag_s；manifest 记 action_latency_s 与 late_dispatch_steps。

覆盖：
  collect_one：动作先于 L 就绪（派发戳 = 观测戳 + L，截止基准仍是观测戳）、晚于 L 就绪（立即派发并记迟到）、
    迟到阈值的边界（严格大于 0.02 s）、不给开关（派发戳 = 最新位姿戳，行只多 dispatch_lag_s，不多 late_dispatch）、
    影子教师标签的计算位置（默认在派发前；学生 + β=0 + 开关 → 派发后；β>0 / 教师策略不动）、等待超时。
  collect.main：manifest 键集合（默认与旧版一字不差，开关才多两个键）、行里的 late_dispatch、参数校验、每集预热。
  eval_closed_loop：--action-latency 透传（没给不传，旧命令行逐字不变）、per_episode 的 dispatch_lag_s、
    迟到集进 failed 并单列、协议键、参数校验。
  tools/preprocess 的口径键、teacher.warm_free_cells 分别在 tools/test_preprocess_split.py、sim/tools/test_astar_cache.py；
  scene_gen --rtf 在 sim/tools/test_scene_rtf.py。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_action_latency.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campus
import collect
import eval_closed_loop as ec
import split_manifest as sm
import teacher
import test_collect as tc          # 只借替身与 fake_row
import test_eval_provenance as tep  # 以模块名引用：import 类名会让 unittest 把对方的用例再跑一遍

FWD = collect.FORWARD_SLOW
OBS_S = 10.005                     # RGB 戳 10.0 s 之后第一帧位姿（E1 位姿 200 Hz，5 ms 一帧）
OBS_NS = 10_005_000_000
PILLARS = [{"x": 0.8, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]   # 正前方 0.8 m 一根柱：A* 标签是转向

# 旧代码（2026-11-10 main 4498177）默认路径写出的 manifest 键集合；不给 --action-latency 时必须一字不差。
MANIFEST_KEYS_TEACHER = {
    "actions_ref", "appearance_version", "camera", "camera_profile", "camera_source", "dynamics",
    "dynamics_experimental", "dynamics_profile", "episode_id", "error", "experimental_policy", "family",
    "geometry_gid", "geometry_version", "goal_line_x", "goal_radius", "hold_s", "instruction", "map_id",
    "num_steps", "oracle_ref", "performance", "planner", "relation", "render_backend_requested",
    "requested_steps", "resource_sha256", "rl_checkpoint", "scene_id", "scene_meta_sha256", "schema", "seed",
    "split", "split_policy", "stop_radius", "stopped_on_collision", "stopped_on_goal", "success_criterion",
    "target_color", "task_version", "teacher", "teacher_tiebreak", "teacher_version", "template_sha256",
    "validity", "vehicle", "version", "waypoint", "window_from", "world_sha256"}
MANIFEST_KEYS_STUDENT = MANIFEST_KEYS_TEACHER | {
    "beta", "dagger_round", "executed_by", "n_expert_executed", "student_ckpt", "student_ckpt_sha256"}


# ---------------------------------------------------------------- collect_one 的替身

class LatencyPoseSub(tc.ClockPoseSub):
    """200 Hz 位姿的假订阅：latest_after(stamp) 把假时钟推进到第一帧 > stamp 并返回它，同时记下每次请求。"""
    PERIOD_NS = 5_000_000

    def __init__(self, initial_s):
        super().__init__(tc.pose_msg(initial_s))
        self.waits = []

    def latest_after(self, stamp_ns, timeout=0.0):
        self.waits.append(stamp_ns)
        now = collect._pose_stamp(self.current)
        if now <= stamp_ns:
            n = (stamp_ns - now) // self.PERIOD_NS + 1
            self.current = tc.pose_msg((now + n * self.PERIOD_NS) / 1e9)
        return self.current


class RecordingSitl(tc.FakeSitl):
    """cf_sitl 执行器替身：记下 (动作, 执行器截止基准, 派发那一刻的位姿戳) 和调用顺序。"""

    def __init__(self, pose, log):
        super().__init__()
        self.pose, self.log, self.calls = pose, log, []

    def dispatch(self, action, t_ns):
        self.calls.append((action, t_ns, collect._pose_stamp(self.pose.latest())))
        self.log.append("dispatch")


def run_step(ready_s, latency=0.16, policy="student", beta=0.0, tiebreak="legacy", rng_value=0.0,
             window_from="observation", plan_raises=None, wait_returns_none=False):
    """真 collect_one + cf_sitl 替身。观测戳 10.005 s；策略（学生推理 / 教师 A*）结束时假时钟停在 ready_s。

    返回 (row, err, sil, pose, 调用顺序, 影子标签函数收到的参数)。latency=None 即不给开关。
    """
    w, h = 4, 6
    pose = LatencyPoseSub(OBS_S)
    if wait_returns_none:
        pose.latest_after = lambda *a, **k: None
    subs = {"rgb": tc.FakeSub(after=tc.image_msg(w, h, 10)),
            "depth": tc.FakeSub(frames=[(1, tc._fresh_depth(w, h))]),
            "pose": pose, "contact": tc.FakeSub()}
    log, plan_args = [], []
    sil = RecordingSitl(pose, log)
    real_plan = collect.plan_action_global_tiebreak

    def spy_plan(*a, **kw):
        log.append("label")
        plan_args.append((a, kw))
        if plan_raises is not None:
            raise plan_raises
        if policy == "teacher" and "dispatch" not in log:
            pose.current = tc.pose_msg(ready_s)     # 教师：A* 就是动作，它跑完时钟才到 ready_s
        return real_plan(*a, **kw)

    class Student:
        def act(self, msg, prev_action=None):
            log.append("act")
            pose.current = tc.pose_msg(ready_s)     # 推理（含护栏）结束
            return FWD, 1.0, [0.0, 1.0, 0.0, 0.0]

    def advance(_secs):
        pose.current = tc.pose_msg(OBS_S + 1.0 + 0.01)   # hold 1.0 s 走满

    extra = {} if latency is None else {"action_latency_s": latency}
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance), \
            patch.object(collect, "stop_executor") as stopped, \
            patch.object(collect, "plan_action_global_tiebreak", side_effect=spy_plan):
        try:
            row, err = collect.collect_one(
                d, 0, 1.0, policy, subs, tc.FakePub(), pillars=PILLARS, sil=sil, window_from=window_from,
                student=Student() if policy == "student" else None, tiebreak=tiebreak, beta=beta,
                rng=tc.FixedRng(rng_value), **extra)
        except Exception as e:  # noqa: BLE001 - 影子标签失败的用例要看异常
            row, err = None, e
    return row, err, sil, pose, log, plan_args, stopped


class ActionLatencyStep(unittest.TestCase):
    def test_ready_before_L_waits_until_obs_plus_L(self):
        row, err, sil, pose, log, _, _ = run_step(OBS_S + 0.06)
        self.assertIsNone(err)
        self.assertEqual(sil.calls[0][2], OBS_NS + 160_000_000, "派发那一刻的位姿戳 = 观测戳 + L")
        self.assertEqual(sil.calls[0][1], OBS_NS, "执行器截止基准仍是观测戳（截止 = 观测戳 + hold）")
        self.assertEqual(row["action_start_ns"], OBS_NS)
        self.assertAlmostEqual(row["dispatch_lag_s"], 0.16, places=9)
        self.assertIs(row["late_dispatch"], False)
        self.assertEqual(pose.waits, [OBS_NS + 160_000_000 - 1], "只等一次：位姿戳严格大于 目标戳 − 1 ns")

    def test_ready_after_L_dispatches_at_once_and_is_late(self):
        row, err, sil, pose, _, _, _ = run_step(OBS_S + 0.30)
        self.assertIsNone(err)
        self.assertEqual(pose.waits, [], "已经过了 L，不再等")
        self.assertEqual(sil.calls[0][2], OBS_NS + 300_000_000)
        self.assertAlmostEqual(row["dispatch_lag_s"], 0.30, places=9)
        self.assertIs(row["late_dispatch"], True)

    def test_late_threshold_is_strictly_more_than_20ms_over_L(self):
        on_time, err, _, pose, _, _, _ = run_step(OBS_S + 0.18)      # L + 0.02 s 整：不算迟到，也不等
        self.assertIsNone(err)
        self.assertIs(on_time["late_dispatch"], False)
        self.assertEqual(pose.waits, [])
        self.assertAlmostEqual(on_time["dispatch_lag_s"], 0.18, places=9)
        late, err, _, _, _, _, _ = run_step(OBS_S + 0.185)           # 多 5 ms：迟到
        self.assertIsNone(err)
        self.assertIs(late["late_dispatch"], True)

    def test_ready_between_L_and_threshold_not_late(self):
        row, err, sil, pose, _, _, _ = run_step(OBS_S + 0.17)
        self.assertIsNone(err)
        self.assertIs(row["late_dispatch"], False)
        self.assertEqual(sil.calls[0][2], OBS_NS + 170_000_000)

    def test_no_flag_dispatches_at_latest_pose_and_adds_only_dispatch_lag(self):
        row, err, sil, pose, _, _, _ = run_step(OBS_S + 0.06, latency=None)
        self.assertIsNone(err)
        self.assertEqual(pose.waits, [], "没给开关不等")
        self.assertEqual(sil.calls[0][2], OBS_NS + 60_000_000, "旧行为：派发在最新位姿戳")
        self.assertAlmostEqual(row["dispatch_lag_s"], 0.06, places=9)
        self.assertIsNone(row["late_dispatch"])
        keys = set(collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0}))
        self.assertEqual(keys, tc.ROW_KEYS_DEFAULT | {"executed_by", "probs"})   # 学生行本来就带 executed_by；W1-3 起另带 probs

    def test_flag_row_adds_late_dispatch_after_dispatch_lag(self):
        row, _, _, _, _, _, _ = run_step(OBS_S + 0.06)
        ar = collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})
        self.assertEqual(set(ar), tc.ROW_KEYS_DEFAULT | {"executed_by", "late_dispatch", "probs"})   # probs：W1-3，学生行才有
        self.assertIs(ar["late_dispatch"], False)
        keys = list(ar)
        self.assertEqual(keys[-2:], ["dispatch_lag_s", "late_dispatch"], "新键排在最后：旧键的行内顺序不变")

    def test_dispatch_window_semantics_rejected(self):
        # run_step 把异常收进 err：window_from=dispatch 的截止是派发戳 + hold，与「截止仍是观测戳 + hold」矛盾
        _, err, *_ = run_step(OBS_S + 0.06, window_from="dispatch")
        self.assertIsInstance(err, ValueError)

    def test_wait_timeout_fails_the_step(self):
        row, err, *_ = run_step(OBS_S + 0.06, wait_returns_none=True)
        self.assertIsNone(row)
        self.assertIn("派发", str(err))

    # ---- 影子教师标签的计算位置
    def test_default_student_label_is_computed_before_dispatch(self):
        row, err, _, _, log, _, _ = run_step(OBS_S + 0.06, latency=None)
        self.assertIsNone(err)
        self.assertEqual(log, ["act", "label", "dispatch"])

    def test_flag_student_beta0_label_after_dispatch_same_call_and_same_row(self):
        base, _, _, _, log0, args0, _ = run_step(OBS_S + 0.06, latency=None, tiebreak="visible")
        row, err, _, _, log, args, _ = run_step(OBS_S + 0.06, tiebreak="visible")
        self.assertIsNone(err)
        self.assertEqual(log, ["act", "dispatch", "label"], "学生 + β=0 + 开关：标签挪到派发之后")
        self.assertEqual(args, args0, "同一个函数、同一组位姿/参数")
        for k in ("action_expert", "action_executed", "executed_by", "tie_margin_steps"):
            self.assertEqual(row[k], base[k], k)
        self.assertIn(row["action_expert"], (collect.YAW_LEFT, collect.YAW_RIGHT))
        self.assertEqual(row["action_executed"], FWD)       # 执行的仍是学生动作
        self.assertEqual(row["executed_by"], "student")
        self.assertIsInstance(row["tie_margin_steps"], float)

    def test_flag_student_beta_positive_label_stays_before_dispatch(self):
        row, err, _, _, log, _, _ = run_step(OBS_S + 0.06, beta=1.0, rng_value=0.3)
        self.assertIsNone(err)
        self.assertEqual(log, ["act", "label", "dispatch"])
        self.assertEqual(row["executed_by"], "expert")
        self.assertEqual(row["action_executed"], row["action_expert"])

    def test_flag_teacher_label_is_the_action_and_stays_before_dispatch(self):
        row, err, sil, _, log, _, _ = run_step(OBS_S + 0.06, policy="teacher")
        self.assertIsNone(err)
        self.assertEqual(log, ["label", "dispatch"])
        self.assertEqual(sil.calls[0][2], OBS_NS + 160_000_000)
        self.assertEqual(row["action_executed"], row["action_expert"])

    def test_deferred_label_failure_stops_the_executor_and_propagates(self):
        row, err, sil, _, log, _, stopped = run_step(OBS_S + 0.06, plan_raises=RuntimeError("boom"))
        self.assertIsNone(row)
        self.assertIsInstance(err, RuntimeError)
        self.assertEqual(log, ["act", "dispatch", "label"])
        stopped.assert_called_once()


# ---------------------------------------------------------------- collect.main

def late_rows(flags, lag=0.16):
    return [tc.fake_row(i, 1.0 + 0.25 * i, dispatch_lag_s=(0.30 if late else lag), late_dispatch=late)
            for i, late in enumerate(flags)]


class ActionLatencyMain(unittest.TestCase):
    def test_default_manifest_and_rows_are_unchanged_except_dispatch_lag(self):
        rows = [tc.fake_row(i, 1.0 + 0.25 * i, dispatch_lag_s=0.129, late_dispatch=None) for i in range(3)]
        man, acts, code, _ = tc.run_main([], rows)
        self.assertIsNone(code)
        self.assertEqual(set(man), MANIFEST_KEYS_TEACHER, "不给开关：manifest 键集合与旧版一字不差")
        self.assertEqual({frozenset(a) for a in acts}, {frozenset(tc.ROW_KEYS_DEFAULT)})
        self.assertEqual([a["dispatch_lag_s"] for a in acts], [0.129] * 3)
        man_s, acts_s, code_s, _ = tc.run_main(["--policy", "student", "--student-ckpt", "CKPT"],
                                               [tc.fake_row(i, 1.0, dispatch_lag_s=0.129, executed_by="student")
                                                for i in range(2)], student_cls=tc.FakeStudentPolicy)
        self.assertIsNone(code_s)
        self.assertEqual(set(man_s), MANIFEST_KEYS_STUDENT)

    def test_flag_records_manifest_keys_and_row_flags(self):
        man, acts, code, _ = tc.run_main(["--action-latency", "0.16"], late_rows([False, True, False, True]))
        self.assertIsNone(code)
        self.assertEqual(set(man), MANIFEST_KEYS_TEACHER | {"action_latency_s", "late_dispatch_steps"})
        self.assertEqual(man["action_latency_s"], 0.16)
        self.assertEqual(man["late_dispatch_steps"], 2)
        self.assertEqual([a["late_dispatch"] for a in acts], [False, True, False, True])
        self.assertEqual([a["dispatch_lag_s"] for a in acts], [0.16, 0.30, 0.16, 0.30])
        self.assertEqual(man["hold_s"], 1.0)
        self.assertEqual(man["window_from"], "observation")

    def test_flag_passes_latency_to_every_collect_one_call(self):
        seen = []

        def fake(out_dir, step_idx, *args, **kw):
            seen.append(kw.get("action_latency_s", "缺"))
            return tc.fake_row(step_idx, 1.0 + 0.25 * step_idx, dispatch_lag_s=0.16, late_dispatch=False), None

        self.assertEqual(self.run_loop(["--action-latency", "0.16"], fake, steps=3)[0], 0)
        self.assertEqual(seen, [0.16] * 3)
        seen.clear()
        self.assertEqual(self.run_loop([], fake, steps=2)[0], 0)
        self.assertEqual(seen, [None, None], "不给开关传 None")

    def test_invalid_values_rejected(self):
        rows = late_rows([False])
        for bad in ("0", "-0.1", "nan", "inf", "1.0", "1.5"):
            self.assertEqual(tc.run_main(["--action-latency", bad], rows)[2], 2, bad)
        self.assertEqual(tc.run_main(["--action-latency", "0.16", "--window-from", "dispatch"], rows)[2], 2)
        self.assertEqual(tc.run_main(["--action-latency", "0.16", "--hold", "0.1"], rows)[2], 2, "L ≥ hold")
        self.assertIsNone(tc.run_main(["--action-latency", "0.16"], rows)[2])

    # ---- 每集预热
    def run_loop(self, extra_argv, fake_collect_one, steps, policy_argv=(), student_cls=None):
        """真 collect.main + 自己的 collect_one 替身；返回 (退出码, 事件日志)。"""
        log = []
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st, \
                contextlib.redirect_stderr(io.StringIO()):
            out, meta, ckpt = (os.path.join(d, n) for n in ("ep", "scene.json", "ckpt.bin"))
            Path(meta).write_text(json.dumps(tc.SCENE))
            Path(ckpt).write_bytes(b"x")
            argv = ["collect.py", "--out", out, "--scene-meta", meta, "--split", sm.split_for_scene(tc.SCENE),
                    "--steps", str(steps)] + list(policy_argv) + [a.replace("CKPT", ckpt) for a in extra_argv]
            subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}

            def step(*a, **kw):
                log.append("step")
                return fake_collect_one(*a, **kw)

            st.enter_context(patch.object(sys, "argv", argv))
            st.enter_context(patch.object(collect, "open_subs", return_value=subs))
            st.enter_context(patch.object(collect, "wait_ready", return_value=None))
            st.enter_context(patch.object(collect, "read_camera_info", return_value=None))
            st.enter_context(patch.object(collect.gz_io, "open_twist_publisher", return_value=tc.MainPub()))
            st.enter_context(patch.object(collect, "take_off",
                                          side_effect=lambda *a, **k: log.append("takeoff") or 0))
            st.enter_context(patch.object(collect, "stop_executor"))
            st.enter_context(patch.object(collect, "close_subs"))
            st.enter_context(patch.object(collect, "collect_one", side_effect=step))
            st.enter_context(patch.object(teacher, "warm_free_cells",
                                          side_effect=lambda *a, **k: log.append(("warm", a, k)) or 0))
            if student_cls is not None:
                st.enter_context(patch("student.StudentPolicy", student_cls))
            code = 0
            try:
                collect.main()
            except SystemExit as e:
                code = e.code
        return code, log

    @staticmethod
    def ok_row(out_dir, step_idx, *a, **kw):
        return tc.fake_row(step_idx, 1.0 + 0.25 * step_idx, dispatch_lag_s=0.16, late_dispatch=False), None

    def test_warm_up_runs_once_before_takeoff_only_with_flag(self):
        code, log = self.run_loop(["--action-latency", "0.16"], self.ok_row, steps=2)
        self.assertEqual(code, 0)
        self.assertEqual([e if isinstance(e, str) else e[0] for e in log], ["warm", "takeoff", "step", "step"])
        warm = log[0]
        self.assertEqual(warm[1], (tc.SCENE["pillars"],), "整张栅格按本集 scene-meta 的柱子填")
        code, log = self.run_loop([], self.ok_row, steps=2)
        self.assertEqual(code, 0)
        self.assertEqual(log, ["takeoff", "step", "step"], "不给开关：不预热")

    def test_warm_up_also_for_student_shadow_teacher_but_not_for_forward(self):
        code, log = self.run_loop(["--policy", "student", "--student-ckpt", "CKPT", "--action-latency", "0.16"],
                                  self.ok_row, steps=1, student_cls=tc.FakeStudentPolicy)
        self.assertEqual(code, 0)
        self.assertEqual([e if isinstance(e, str) else e[0] for e in log], ["warm", "takeoff", "step"])
        code, log = self.run_loop(["--policy", "forward", "--action-latency", "0.16"], self.ok_row, steps=1)
        self.assertEqual(code, 0)
        self.assertEqual(log, ["takeoff", "step"])


# ---------------------------------------------------------------- eval_closed_loop

class EvalHarness:
    """完整 ec.main() + 假 scene_gen / 假 collect 子进程（沿用 test_eval_provenance 的做法），记下两条命令行。"""

    @staticmethod
    def run(extra_argv, actions, manifest=None, collect_rc=0, expect_exit=None):
        cmds = {}

        def generate(cmd, **_kw):
            cmds["scene_gen"] = list(cmd)
            world = Path(cmd[cmd.index("--out") + 1])
            world.write_text("<sdf/>")
            world.with_suffix(".json").write_text(json.dumps(tep.EvalProvenanceTests.scene_meta()))
            return Mock(returncode=0)

        def collect_proc(cmd, **_kw):
            cmds["collect"] = list(cmd)
            ep = Path(cmd[cmd.index("--out") + 1])
            (ep / "actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in actions))
            (ep / "oracle.jsonl").write_text("".join(json.dumps({"pose_gt": {"x": 2.0, "y": 1.0}}) + "\n"
                                                      for _ in actions))
            (ep / "episode_manifest.json").write_text(json.dumps(manifest or {}))
            proc = Mock(returncode=collect_rc)
            proc.communicate.return_value = ("", "")
            return proc

        errbuf = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "eval"
            argv = ["eval_closed_loop.py", "--policy", "forward", "--task", "campus", "--window-from", "observation",
                    "--split", campus.split(16), "--seeds", "16", "--steps", "1", "--out", str(out)] + extra_argv
            with patch.object(ec.sys, "argv", argv), \
                 patch.object(ec.subprocess, "run", side_effect=generate), \
                 patch.object(ec.subprocess, "Popen", side_effect=collect_proc), \
                 patch.object(ec.cp, "_start_sim", return_value=Mock()), \
                 patch.object(ec.cp, "wait_ready"), patch.object(ec.cp, "_register"), \
                 patch.object(ec.cp, "stop_group"), patch.object(ec.cp, "cleanup_all"), \
                 patch.object(ec.cp, "XDG_RUNTIME_DIR", str(Path(d) / "xdg")), \
                 patch.object(ec.time, "sleep"), patch.object(ec.signal, "signal"), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errbuf):
                if expect_exit is not None:
                    try:
                        ec.main()
                    except SystemExit as e:
                        cmds["stderr"] = errbuf.getvalue()
                        return e.code, cmds
                    return None, cmds
                ec.main()
            return json.loads((out / "eval.json").read_text()), cmds


def eval_rows(lags, late_ids=(), flagged=True):
    rows = []
    for i, lag in enumerate(lags):
        r = {"action_executed": 0, "collision": False, "infer_ms": None, "dispatch_lag_s": lag}
        if flagged:
            r["late_dispatch"] = i in late_ids
        rows.append(r)
    return rows


class EvalPassThrough(unittest.TestCase):
    def test_default_command_lines_have_no_new_flags_and_report_has_no_new_top_keys(self):
        report, cmds = EvalHarness.run([], eval_rows([0.129, 0.131, 0.140], flagged=False))
        self.assertNotIn("--action-latency", cmds["collect"])
        for key in ("action_latency_s", "late_dispatch_episodes"):
            self.assertNotIn(key, report)
        ep = report["per_episode"][0]
        self.assertNotIn("late_dispatch_steps", ep)
        self.assertAlmostEqual(ep["dispatch_lag_s_median"], 0.131, places=9)
        self.assertAlmostEqual(ep["dispatch_lag_s_max"], 0.140, places=9)

    def test_flag_is_passed_through_and_recorded(self):
        report, cmds = EvalHarness.run(["--action-latency", "0.16"],
                                       eval_rows([0.16, 0.16, 0.165]), {"action_latency_s": 0.16,
                                                                        "late_dispatch_steps": 0})
        c = cmds["collect"]
        self.assertEqual(c[c.index("--action-latency") + 1], "0.16")
        self.assertEqual(report["action_latency_s"], 0.16)
        self.assertEqual(report["late_dispatch_episodes"], [])
        ep = report["per_episode"][0]
        self.assertEqual(ep["late_dispatch_steps"], 0)
        self.assertAlmostEqual(ep["dispatch_lag_s_median"], 0.16, places=9)
        self.assertAlmostEqual(ep["dispatch_lag_s_max"], 0.165, places=9)
        self.assertEqual(report["failed_count"], 0)

    def test_late_episode_is_infrastructure_failure_listed_separately(self):
        report, _ = EvalHarness.run(["--action-latency", "0.16"], eval_rows([0.16, 0.30, 0.16, 0.31], late_ids=(1, 3)),
                                    {"action_latency_s": 0.16, "late_dispatch_steps": 2})
        self.assertEqual(report["per_episode"], [], "迟到集不进完成集")
        self.assertEqual(report["failed_count"], 1)
        failure = report["failed"][0]
        self.assertEqual(failure["seed"], 16)
        self.assertIn("late_dispatch", failure["error"])
        self.assertEqual(failure["late_dispatch_steps"], 2)
        self.assertEqual(failure["late_step_ids"], [1, 3])
        self.assertAlmostEqual(failure["dispatch_lag_s_max"], 0.31, places=9)
        self.assertAlmostEqual(failure["dispatch_lag_s_median"], 0.23, places=9)
        self.assertEqual(report["late_dispatch_episodes"], [{
            "seed": 16, "late_dispatch_steps": 2, "late_step_ids": [1, 3],
            "dispatch_lag_s_median": failure["dispatch_lag_s_median"], "dispatch_lag_s_max": 0.31}])
        self.assertEqual(report["requested_success_rate"], 0.0, "迟到集占请求名额、不算成功")
        self.assertEqual(report["attempted_seeds"], [16])

    def test_protocol_key_distinguishes_latency_runs(self):
        self.assertIn("action_latency_s", ec.PROTOCOL_KEYS)
        got = ec.protocol_mismatches({"action_latency_s": 0.16}, {"action_latency_s": None})
        self.assertEqual(got, [("action_latency_s", 0.16, None)])
        self.assertEqual(ec.protocol_mismatches({"action_latency_s": 0.16}, {"action_latency_s": 0.16}), [])
        self.assertEqual(ec.protocol_mismatches({"hold_s": 1.0}, {"hold_s": 1.0, "action_latency_s": None}), [],
                         "旧协议快照没有这个键：不判 OOD")

    def test_invalid_values_rejected(self):
        for flag, bad in (("--action-latency", "0"), ("--action-latency", "-0.1"), ("--action-latency", "nan"),
                          ("--action-latency", "1.0")):
            code, cmds = EvalHarness.run([flag, bad], eval_rows([0.16]), expect_exit=True)
            self.assertEqual(code, 2, f"{flag} {bad}")
            self.assertIn(flag + " 必须", cmds["stderr"], f"{flag} {bad}：要是我们的校验报的，不是「不认识的参数」")
        code, cmds = EvalHarness.run(["--action-latency", "0.16", "--window-from", "dispatch"], eval_rows([0.16]),
                                     expect_exit=True)
        self.assertEqual(code, 2)
        self.assertIn("--action-latency 只能配 --window-from observation", cmds["stderr"])
        code, cmds = EvalHarness.run(["--action-latency", "0.16", "--hold", "0.1"], eval_rows([0.16]),
                                     expect_exit=True)
        self.assertEqual(code, 2)
        self.assertIn("--action-latency 必须小于 --hold", cmds["stderr"])


if __name__ == "__main__":
    unittest.main()
