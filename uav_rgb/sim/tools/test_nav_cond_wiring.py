#!/usr/bin/env python3
"""航向码 --student-cond 的接线测试（任务卡 W1-9）：collect_one 传给学生的值 = nav_cond.heading_code(net_yaw)；
collect.main 主循环的 net_yaw 与 manifest / actions.jsonl 留痕；collect_parallel 透传；eval_closed_loop 的透传、
参数校验、一致性闸门与 eval.json 记录。默认（prev-action）时命令行、manifest、行字段、eval.json 都不留痕迹。

不起 Gazebo：collect_one、gz、StudentPolicy 全部替身，替身沿用 test_collect.py / test_eval_sim09v2.py / test_heading_guard.py 的。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_nav_cond_wiring.py
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect
import collect_parallel as parallel
import nav_cond
import split_manifest as sm
import test_collect as tc          # 只借替身与 fake_row
import test_eval_sim09v2 as t09    # 以模块名引用：import 类名会让 unittest 把对方的用例再跑一遍

LEFT, RIGHT, FORWARD = collect.YAW_LEFT, collect.YAW_RIGHT, collect.FORWARD_SLOW
STUDENT = ["--policy", "student", "--student-ckpt", "CKPT"]


class CollectOneTests(unittest.TestCase):
    """真 collect_one + 假学生：学生收到的 prev_action 是什么。"""

    def test_heading_code_replaces_prev_executed_with_nav_cond_value(self):
        for n in range(-4, 5):
            row, err, seen = tc.shadow_teacher_step(beta=0.0, prev_executed=LEFT, net_yaw=n, student_cond="heading-code")
            self.assertIsNone(err)
            self.assertEqual(seen, [nav_cond.heading_code(n)], f"net_yaw={n}")
            self.assertEqual(row["cond_code"], nav_cond.heading_code(n))

    def test_first_step_sends_code_zero_even_without_prev_executed(self):
        _row, _err, seen = tc.shadow_teacher_step(beta=0.0, prev_executed=None, net_yaw=0, student_cond="heading-code")
        self.assertEqual(seen, [0], "首帧 n=0 → 码 0（默认路径此时传 None，由 student.act 取自己上一步的输出）")

    def test_default_is_bit_identical_to_before(self):
        # 默认：学生收到的还是 prev_executed（net_yaw 再大也不碰），行里没有 cond_code
        for kw in ({}, {"student_cond": "prev-action"}):
            row, err, seen = tc.shadow_teacher_step(beta=0.0, prev_executed=RIGHT, net_yaw=5, **kw)
            self.assertIsNone(err)
            self.assertEqual(seen, [RIGHT], kw)
            self.assertIsNone(row.get("cond_code"), kw)
        _row, _err, seen = tc.shadow_teacher_step(beta=0.0)
        self.assertEqual(seen, [None])

    def test_heading_code_is_independent_of_beta_mixing(self):
        # β=1：执行教师动作，学生仍然收到码（码只看已执行的转向数）
        _row, err, seen = tc.shadow_teacher_step(beta=1.0, rng_value=0.3, prev_executed=3, net_yaw=2,
                                                 student_cond="heading-code")
        self.assertIsNone(err)
        self.assertEqual(seen, [2])

    def test_action_constants_match_collect(self):
        self.assertEqual((nav_cond.FORWARD_SLOW, nav_cond.YAW_LEFT, nav_cond.YAW_RIGHT), (FORWARD, LEFT, RIGHT))

    def test_unknown_student_cond_is_a_value_error(self):
        with self.assertRaises(ValueError):
            tc.shadow_teacher_step(beta=0.0, student_cond="bogus")

    def test_action_row_writes_cond_code_only_when_present_and_keeps_zero(self):
        base = tc.fake_row(0, 1.0) | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0}
        self.assertNotIn("cond_code", collect.action_row(base))
        self.assertNotIn("cond_code", collect.action_row(base | {"cond_code": None}))
        self.assertEqual(collect.action_row(base | {"cond_code": 0})["cond_code"], 0)
        self.assertEqual(collect.action_row(base | {"cond_code": 3})["cond_code"], 3)
        self.assertEqual(list(collect.action_row(base | {"cond_code": 2}))[-1], "cond_code",
                         "新键排在最后：旧键与 W1-0 的 dispatch_lag_s 的行内顺序不动")


def run_loop(extra_argv, steps, student_action):
    """真 collect.main() 主循环 + 假 collect_one：记下主循环传进来的 net_yaw 与 student_cond，按开关给 cond_code 留痕。

    返回 (net_yaw 序列, student_cond 序列, actions.jsonl 行, manifest 或 None, stderr 文本)。
    """
    seen = []

    def fake_collect_one(out_dir, step_idx, *args, **kw):
        seen.append((kw.get("net_yaw"), kw.get("student_cond")))
        code = nav_cond.heading_code(kw["net_yaw"]) if kw.get("student_cond") == "heading-code" else None
        return tc.fake_row(step_idx, 1.0 + 0.25 * step_idx, action_executed=student_action,
                           executed_by="student", cond_code=code), None

    err = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st, contextlib.redirect_stderr(err):
        out, meta, ckpt = (os.path.join(d, n) for n in ("ep", "scene.json", "ckpt.bin"))
        Path(meta).write_text(json.dumps(tc.SCENE))
        Path(ckpt).write_bytes(b"not-a-real-checkpoint")
        argv = ["collect.py", "--out", out, "--scene-meta", meta, "--split", sm.split_for_scene(tc.SCENE),
                "--steps", str(steps)] + [a.replace("CKPT", ckpt) for a in extra_argv]
        subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}
        st.enter_context(patch.object(sys, "argv", argv))
        st.enter_context(patch.object(collect, "open_subs", return_value=subs))
        st.enter_context(patch.object(collect, "wait_ready", return_value=None))
        st.enter_context(patch.object(collect, "read_camera_info", return_value=None))
        st.enter_context(patch.object(collect.gz_io, "open_twist_publisher", return_value=tc.MainPub()))
        st.enter_context(patch.object(collect, "take_off", return_value=0))
        st.enter_context(patch.object(collect, "stop_executor"))
        st.enter_context(patch.object(collect, "close_subs"))
        st.enter_context(patch.object(collect, "collect_one", side_effect=fake_collect_one))
        st.enter_context(patch("student.StudentPolicy", tc.FakeStudentPolicy))
        with contextlib.suppress(SystemExit):
            collect.main()
        man_path = Path(out, "episode_manifest.json")
        man = json.loads(man_path.read_text()) if man_path.is_file() else None
        rows = ([json.loads(line) for line in Path(out, "actions.jsonl").read_text().splitlines() if line.strip()]
                if man is not None else [])
    return [s[0] for s in seen], [s[1] for s in seen], rows, man, err.getvalue()


class CollectMainLoopTests(unittest.TestCase):
    def test_main_loop_passes_net_yaw_that_matches_the_executed_chain(self):
        # 学生一直左转：执行前的 n 依次 0 1 2 3 4；主循环的 net_yaw 必须等于 nav_cond 对已执行动作的累计
        nets, conds, rows, man, _ = run_loop(STUDENT + ["--student-cond", "heading-code"], 5, LEFT)
        self.assertEqual(nets, [0, 1, 2, 3, 4])
        self.assertEqual(conds, ["heading-code"] * 5)
        executed = [r["action_executed"] for r in rows]
        self.assertEqual([r["cond_code"] for r in rows], nav_cond.heading_codes(executed))
        self.assertEqual([r["cond_code"] for r in rows], [0, 0, 2, 2, 2])
        self.assertEqual(man["student_cond"], "heading-code")

    def test_right_turns_give_code_three(self):
        _nets, _conds, rows, _man, _ = run_loop(STUDENT + ["--student-cond", "heading-code"], 4, RIGHT)
        self.assertEqual([r["cond_code"] for r in rows], [0, 0, 3, 3])

    def test_default_leaves_no_trace_in_manifest_or_rows(self):
        _nets, conds, rows, man, _ = run_loop(STUDENT, 3, LEFT)
        self.assertEqual(conds, ["prev-action"] * 3)
        self.assertNotIn("student_cond", man)
        self.assertTrue(all("cond_code" not in r for r in rows))
        _nets, _conds, rows, man, _ = run_loop(STUDENT + ["--student-cond", "prev-action"], 3, LEFT)
        self.assertNotIn("student_cond", man, "显式给默认值也不写键：manifest 与不给时逐位相同")

    def test_rejected_for_non_student_policy(self):
        _n, _c, _r, man, err = run_loop(["--student-cond", "heading-code"], 2, LEFT)    # 默认策略是教师
        self.assertIsNone(man)
        self.assertIn("--student-cond 只能和 --policy student 一起用", err)

    def test_unknown_value_is_rejected(self):
        _n, _c, _r, man, err = run_loop(STUDENT + ["--student-cond", "bogus"], 2, LEFT)
        self.assertIsNone(man)
        self.assertIn("--student-cond", err)


def args_ns(**over):
    base = dict(steps=40, hold=1.0, policy="teacher", teacher_planner="astar", rl_checkpoint="",
                window_from="observation", dynamics="cf_sitl", save_depth=False, vehicle="e1_mini",
                student_ckpt=None, infer_bin=None, beta=0.0, dagger_round=0,
                stop_at_goal=False, tiebreak="legacy")
    base.update(over)
    return SimpleNamespace(**base)


class CollectParallelTests(unittest.TestCase):
    """DAgger 流水线的采集走 collect_parallel：它不转发就等于流水线的透传是断的。"""

    @staticmethod
    def task():
        return parallel.Task(0, 0, 42, 42, "train", "world.sdf", "scene.json")

    def test_student_cond_is_forwarded_only_when_not_default(self):
        cmd = parallel.collect_command(self.task(), args_ns(policy="student", student_ckpt="/c.bin",
                                                            student_cond="heading-code"))
        self.assertEqual(cmd[cmd.index("--student-cond") + 1], "heading-code")
        for kw in ({}, {"student_cond": "prev-action"}):
            cmd = parallel.collect_command(self.task(), args_ns(policy="student", student_ckpt="/c.bin", **kw))
            self.assertNotIn("--student-cond", cmd, kw)
        self.assertNotIn("--student-cond", parallel.collect_command(self.task(), args_ns()))

    def test_default_command_line_is_byte_for_byte_the_old_one(self):
        self.assertEqual(parallel.collect_command(self.task(), args_ns())[2:],
                         parallel.collect_command(self.task(), args_ns(student_cond="prev-action"))[2:])


def write_dataset(path, meta):
    body = json.dumps(meta).encode("utf-8")
    head = struct.pack("<4sHHIHHBBBBQQQI", b"UAVR", 1, 48, 0, 120, 160, 3, 1, 1, 0, 48, 48, 48, len(body))
    Path(path).write_bytes(head + body)


GUARD = "--student-cond"


class EvalPassthroughTests(unittest.TestCase):
    run_eval = t09.Sim09V2Tests.run_eval      # 复用那边整套 gz / scene_gen / collect 替身（只借方法，不继承）
    tearDown = t09.Sim09V2Tests.tearDown

    def make_ckpt(self, trained=None, prev_dim=4, with_config=True):
        """临时 ckpt + 推理程序；trained: None（数据集 meta 没有 cond 键）或 cond 标签。返回 (ckpt, binary, probe)。"""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        run = Path(tmp.name, "run")
        run.mkdir()
        ds = Path(tmp.name, "ds", "dataset.uavr")
        ds.parent.mkdir()
        write_dataset(ds, {"schema": "uavrgb.dataset.v1", **({"cond": trained} if trained else {})})
        if with_config:
            (run / "config.json").write_text(json.dumps({
                "schema": "uavrgb.train_config.v1", "data": str(ds),
                "dataset_sha256": hashlib.sha256(ds.read_bytes()).hexdigest(), "prev_action": bool(prev_dim)}))
        ckpt, binary = run / "ckpt_best.bin", Path(tmp.name, "uavrgb_infer")
        ckpt.write_bytes(b"ckpt")
        binary.write_bytes(b"binary")
        return ckpt, binary, Mock(text_dim=0, prev_dim=prev_dim)

    def run_student(self, ckpt, binary, probe, *extra):
        with patch("student.StudentPolicy", return_value=probe):
            return self.run_eval([t09.test_seeds(1)[0]], extra_argv=(
                "--policy", "student", "--ckpt", str(ckpt), "--infer-bin", str(binary), *extra))

    def assert_rejected(self, run, fragment):
        # 直接截 parser.error 的消息：旧代码里 --student-cond 是未知参数，同样退出 2，只有消息能区分
        with patch.object(argparse.ArgumentParser, "error", side_effect=SystemExit(2)) as error:
            with self.assertRaises(SystemExit):
                run()
        self.assertIn(fragment, error.call_args[0][0])

    def test_heading_code_is_passed_to_collect_and_recorded_with_the_gate_result(self):
        ckpt, binary, probe = self.make_ckpt(trained="heading_code.v1")
        rc, report, cmds, _w, _out = self.run_student(ckpt, binary, probe, GUARD, "heading-code")
        self.assertEqual(rc, 0)
        self.assertEqual(cmds[0][cmds[0].index(GUARD) + 1], "heading-code")
        self.assertEqual(report["student_cond"], "heading-code")
        self.assertEqual(report["student_cond_check"]["status"], "ok")
        self.assertEqual(report["student_cond_check"]["trained"], "heading-code")
        self.assertEqual(report["metric_version"], "sim09.v2", "条件输入改的是策略不是判据，metric_version 不动")

    def test_heading_code_model_with_default_flag_is_rejected(self):
        ckpt, binary, probe = self.make_ckpt(trained="heading_code.v1")
        self.assert_rejected(lambda: self.run_student(ckpt, binary, probe), "heading-code")

    def test_condition_flag_on_a_model_without_condition_input_is_rejected(self):
        ckpt, binary, probe = self.make_ckpt(trained=None, prev_dim=0)
        self.assert_rejected(lambda: self.run_student(ckpt, binary, probe, GUARD, "heading-code"), "不会生效")

    def test_prev_action_model_with_heading_code_flag_is_rejected(self):
        ckpt, binary, probe = self.make_ckpt(trained=None, prev_dim=4)
        self.assert_rejected(lambda: self.run_student(ckpt, binary, probe, GUARD, "heading-code"), "prev-action")

    def test_prev_action_model_with_default_flag_runs_and_leaves_no_trace(self):
        ckpt, binary, probe = self.make_ckpt(trained=None, prev_dim=4)
        rc, report, cmds, _w, _out = self.run_student(ckpt, binary, probe)
        self.assertEqual(rc, 0)
        self.assertNotIn(GUARD, cmds[0])
        self.assertNotIn("student_cond", report)
        self.assertNotIn("student_cond_check", report)

    def test_unobtainable_training_cond_warns_and_is_written_to_eval_json(self):
        ckpt, binary, probe = self.make_ckpt(trained="heading_code.v1", with_config=False)
        with patch.object(sys.modules["eval_closed_loop"], "print", create=True) as shown:
            rc, report, cmds, _w, _out = self.run_student(ckpt, binary, probe, GUARD, "heading-code")
        self.assertEqual(rc, 0, "拿不到训练条件类型只警告，不拦")
        self.assertEqual(report["student_cond"], "heading-code")
        self.assertEqual(report["student_cond_check"]["status"], "unverified")
        self.assertIsNone(report["student_cond_check"]["trained"])
        self.assertTrue(any("警告" in " ".join(map(str, c.args)) and "条件" in " ".join(map(str, c.args))
                            for c in shown.call_args_list), "要打印警告")
        self.assertEqual(cmds[0][cmds[0].index(GUARD) + 1], "heading-code")

    def test_default_student_with_a_mock_probe_is_unchanged(self):
        # 其它测试的替身只给 text_dim（prev_dim 是个 Mock）：不能因为闸门读不出 int 就报错或留痕
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt, binary = Path(tmp.name, "ckpt.bin"), Path(tmp.name, "uavrgb_infer")
        ckpt.write_bytes(b"ckpt")
        binary.write_bytes(b"binary")
        rc, report, cmds, _w, _out = self.run_student(ckpt, binary, Mock(text_dim=0))
        self.assertEqual(rc, 0)
        self.assertNotIn(GUARD, cmds[0])
        self.assertNotIn("student_cond", report)
        self.assertNotIn("student_cond_check", report)

    def test_default_forward_command_line_and_report_unchanged(self):
        _rc, report, cmds, _w, _out = self.run_eval([t09.test_seeds(1)[0]])
        self.assertNotIn(GUARD, cmds[0])
        self.assertNotIn("student_cond", report)

    def test_rejected_for_non_student_policy(self):
        self.assert_rejected(lambda: self.run_eval([t09.test_seeds(1)[0]], extra_argv=(GUARD, "heading-code")),
                             "--student-cond 只能和 --policy student 一起用")


class CollectGateAndWaypointTests(unittest.TestCase):
    """W1-13b I1-3 & I2-2(2)：collect.py 构造学生后调 check_student_cond 闸门；
    collect.py / collect_parallel.py / eval_closed_loop.py 三处入口在带 waypoint 任务上拒绝 --student-cond heading-code。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def _make_trained_ckpt(self, tag, cond_tag=None, prev_dim=4):
        root = Path(self._tmp.name, tag)
        root.mkdir()
        ds = root / "dataset.uavr"
        write_dataset(ds, {"schema": "uavrgb.dataset.v1", **({"cond": cond_tag} if cond_tag else {})})
        ckpt = root / "ckpt_best.bin"
        ckpt.write_bytes(b"ckpt")
        (root / "config.json").write_text(json.dumps({
            "schema": "uavrgb.train_config.v1", "data": str(ds),
            "dataset_sha256": hashlib.sha256(ds.read_bytes()).hexdigest(),
            "prev_action": bool(prev_dim),
        }))
        closed = []

        class _Stu:
            def __init__(self, _c, _b, text_ids=None):
                self.prev_dim = prev_dim
                self.text_dim = 0

            def close(self):
                closed.append(True)

        return ckpt, _Stu, closed

    def _run_collect(self, ckpt, stu_cls, extra_argv, scene=None):
        scene = tc.SCENE if scene is None else scene
        d = Path(self._tmp.name, f"c_{len(os.listdir(self._tmp.name))}")
        d.mkdir()
        out, meta = d / "ep", d / "scene.json"
        meta.write_text(json.dumps(scene))
        argv = ["collect.py", "--out", str(out), "--scene-meta", str(meta),
                "--split", sm.split_for_scene(scene), "--steps", "2",
                "--policy", "student", "--student-ckpt", str(ckpt)] + list(extra_argv)
        subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}
        steps_called = []

        def fake_step(_out, step_idx, *_a, **_kw):
            steps_called.append(step_idx)
            return tc.fake_row(step_idx, 1.0 + 0.25 * step_idx, executed_by="student"), None

        err = io.StringIO()
        code = 0
        with patch.object(sys, "argv", argv), \
             patch.object(collect, "open_subs", return_value=subs), \
             patch.object(collect, "wait_ready", return_value=None), \
             patch.object(collect, "read_camera_info", return_value=None), \
             patch.object(collect.gz_io, "open_twist_publisher", return_value=tc.MainPub()), \
             patch.object(collect, "take_off", return_value=0), \
             patch.object(collect, "stop_executor"), \
             patch.object(collect, "close_subs"), \
             patch.object(collect, "collect_one", side_effect=fake_step), \
             patch("student.StudentPolicy", stu_cls), \
             contextlib.redirect_stderr(err):
            try:
                collect.main()
            except SystemExit as e:
                code = e.code
        man_p = out / "episode_manifest.json"
        man = json.loads(man_p.read_text()) if man_p.is_file() else None
        return code, err.getvalue(), steps_called, man

    def test_collect_rejects_pure_vision_ckpt_with_heading_code_and_closes_student(self):
        ckpt, stu_cls, closed = self._make_trained_ckpt("pv0", cond_tag=None, prev_dim=0)
        code, stderr, steps, man = self._run_collect(ckpt, stu_cls, ["--student-cond", "heading-code"])
        self.assertEqual(code, 2)
        self.assertEqual(steps, [], "闸门拒绝后一步也不能采")
        self.assertEqual(closed, [True], "已启动的学生推理进程必须收尾关闭")
        self.assertIn("不会生效", stderr)
        if man is not None:
            self.assertIn("不会生效", man["error"])
            self.assertFalse(man["validity"]["episode_executed"])

    def test_collect_rejects_heading_code_ckpt_when_flag_omitted(self):
        ckpt, stu_cls, closed = self._make_trained_ckpt("hc", cond_tag=nav_cond.DATASET_COND_TAG, prev_dim=4)
        code, stderr, steps, _ = self._run_collect(ckpt, stu_cls, [])
        self.assertEqual(code, 2)
        self.assertEqual(steps, [])
        self.assertEqual(closed, [True])
        self.assertIn("heading-code", stderr)
        self.assertIn("不一致", stderr)

    def test_collect_rejects_prev_action_ckpt_with_heading_code_flag(self):
        ckpt, stu_cls, closed = self._make_trained_ckpt("pa", cond_tag=None, prev_dim=4)
        code, stderr, steps, _ = self._run_collect(ckpt, stu_cls, ["--student-cond", "heading-code"])
        self.assertEqual(code, 2)
        self.assertEqual(steps, [])
        self.assertEqual(closed, [True])
        self.assertIn("prev-action", stderr)
        self.assertIn("不一致", stderr)

    def test_collect_accepts_matching_ckpts_and_keeps_default_manifest_unchanged(self):
        import test_action_latency as tal
        ckpt_pa, stu_pa, _ = self._make_trained_ckpt("ok_pa", cond_tag=None, prev_dim=4)
        code_pa, _, steps_pa, man_pa = self._run_collect(ckpt_pa, stu_pa, [])
        self.assertEqual((code_pa, steps_pa), (0, [0, 1]))
        self.assertEqual(set(man_pa), tal.MANIFEST_KEYS_STUDENT, "匹配默认 prev-action 时 manifest 键集逐位不变")

        ckpt_hc, stu_hc, _ = self._make_trained_ckpt("ok_hc", cond_tag=nav_cond.DATASET_COND_TAG, prev_dim=4)
        code_hc, _, steps_hc, man_hc = self._run_collect(ckpt_hc, stu_hc, ["--student-cond", "heading-code"])
        self.assertEqual((code_hc, steps_hc, man_hc["student_cond"]), (0, [0, 1], "heading-code"))

    def test_all_three_entrypoints_reject_heading_code_on_waypoint_tasks(self):
        import eval_closed_loop as ec
        ckpt_hc, stu_hc, _ = self._make_trained_ckpt("wp_hc", cond_tag=nav_cond.DATASET_COND_TAG, prev_dim=4)
        infer = Path(self._tmp.name, "uavrgb_infer")
        infer.write_bytes(b"bin")

        # 1) collect.py：scene_meta 含 waypoint
        wp_scene = dict(tc.SCENE, waypoint={"x": 3.0, "y": 1.0}, stop_radius=0.5,
                        instruction="go left", target_color="red", relation="left")
        code_c, err_c, steps_c, man_c = self._run_collect(ckpt_hc, stu_hc, ["--student-cond", "heading-code"],
                                                          scene=wp_scene)
        self.assertEqual(code_c, 2)
        self.assertEqual(steps_c, [])
        self.assertIsNone(man_c, "在参数/场景校验阶段直接拒绝，不进主循环")
        self.assertIn("waypoint", err_c)

        # 2) collect_parallel.py：l1 / campus / campus_assets
        for task_argv in (["--task", "l1", "--spawn-mode", "random"],
                          ["--task", "campus"],
                          ["--task", "campus_assets"]):
            err_p = io.StringIO()
            out_p = Path(self._tmp.name, f"p_{len(os.listdir(self._tmp.name))}")
            argv_p = ["collect_parallel.py", "--policy", "student", "--student-ckpt", str(ckpt_hc),
                      "--student-cond", "heading-code", "--out", str(out_p)] + task_argv
            with patch.object(sys, "argv", argv_p), contextlib.redirect_stderr(err_p):
                with self.assertRaises(SystemExit) as cm:
                    parallel.main()
            self.assertEqual(cm.exception.code, 2, task_argv)
            self.assertIn("waypoint", err_p.getvalue(), task_argv)

        # 3) eval_closed_loop.py：l1 / campus / campus_assets
        for task_name in ("l1", "campus", "campus_assets"):
            err_e = io.StringIO()
            out_e = Path(self._tmp.name, f"e_{task_name}")
            argv_e = ["eval_closed_loop.py", "--policy", "student", "--ckpt", str(ckpt_hc),
                      "--infer-bin", str(infer), "--student-cond", "heading-code",
                      "--task", task_name, "--seeds", "16", "--steps", "1", "--out", str(out_e)]
            with patch.object(ec.sys, "argv", argv_e), \
                 patch("student.StudentPolicy", return_value=Mock(text_dim=0, prev_dim=4)), \
                 contextlib.redirect_stderr(err_e):
                with self.assertRaises(SystemExit) as cm:
                    ec.main()
            self.assertEqual(cm.exception.code, 2, task_name)
            self.assertIn("waypoint", err_e.getvalue(), task_name)


if __name__ == "__main__":
    unittest.main()
