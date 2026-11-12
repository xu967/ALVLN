#!/usr/bin/env python3
"""W1-13 第一部分「步间记账」的测试：A1 safe_gap_s、A3 obs_gap_s 与停顿步、A5 plugin_ack_sim_t。不起 Gazebo。

背景（审核 A 的发现 1）：--action-latency L 只固定了「观测到派发」，没有管「上一动作截止到下一动作派发」。
执行器截止 = 观测戳 + hold，下一次派发 = 下一观测戳 + L，相邻观测相隔 frame_dt（hold=1.0、相机 10 Hz 时恒为 0.9 s：
动作窗口结束时到手的最新一帧是 0.1 s 前拍的），两步之间有 L − (hold − frame_dt) 的空档，执行器回 SAFE
（速度参考 0 加位置保持，主动刹停）。L=0.20 时空档恒为 0.100 s。这段空档不落盘，迟到判据也看不到。

覆盖：
  A1  collect_one 的 safe_gap_s（本步派发戳减上一步执行器截止，首步 None，可为负）与 until_ns（本步截止，主循环带给下一步）；
      action_row 落盘（首步为 null、until_ns 不落盘）；collect.main 把上一步的 until_ns 带给下一步；
      summarize_episode 的 safe_gap_s_median / safe_gap_s_max（旧数据没有这个字段就不多键）。
  A3  collect_one 的 obs_gap_s（本观测戳减上一观测戳，首步 None）与停顿判据 is_stall_gap（观测间隔 > 名义步周期 0.9 s + 0.2 s）；
      action_row 落盘；collect.main 的 manifest stall_steps（只在给了 --action-latency 时写）；
      summarize_episode 的 obs_gap_s_max / stall_steps / stall_step_ids；
      scan_episodes 与完整 ec.main()：带 --action-latency 时含停顿步的集与含迟到步的集同等处理（进 failed[]，占请求名额，不计成功，
      顶层 stall_episodes 单列），不带开关只记不判。
  A5  cf_sitl 执行器的 ack_sim_t()（插件侧实际生效的仿真时刻）：在 state 缓冲里找第一帧「重建出的截止 == 本步发出的截止」，
      连着两步都是 FORWARD 时靠截止区分（动作字段不变），float32 舍入不影响匹配，缓冲够存 3 s；collect_one 在窗口走完后读它，
      行 plugin_ack_sim_t 只在执行器有 ack_sim_t 时写键（没确认到帧是 null），action_row 落盘。只记录，不改迟到判据。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_step_gaps.py
"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_sitl_exec as ce
import collect
import eval_closed_loop as ec
import split_manifest as sm
import teacher
import test_action_latency as tal   # 以模块名引用：import 类名会让 unittest 把对方的用例再跑一遍
import test_collect as tc           # 只借替身与 fake_row
import test_eval_summary as tes     # 只借 gap_rows / gap_summary

W, H = 4, 6
NS = 10**9
PILLARS = [{"x": 0.8, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]


def ns(sec):
    """秒 → 整数 ns（避免浮点误差）。"""
    return int(round(sec * NS))


def stamped(msg, ts_ns):
    msg["header"] = {"stamp": {"sec": ts_ns // NS, "nsec": ts_ns % NS}}
    return msg


class StepWorld:
    """一步的假世界：RGB 帧戳 rgb_s，之后第一帧 200 Hz 位姿（+5 ms）是观测位姿；策略结束时假时钟停在 ready_s（None = 不耗时）。

    返回 (row, err, sil)。hold 窗口走满由 time.sleep 的替身一步跳到 观测戳 + hold + 0.01。
    """

    def __init__(self, rgb_s, ready_s=None, hold=1.0, latency=0.2, policy="forward", action=collect.FORWARD_SLOW,
                 sil_factory=None, window_from="observation", **collect_kw):
        self.rgb_ts = ns(rgb_s)
        self.obs_s = rgb_s + 0.005
        self.ready_s = ready_s
        self.hold, self.latency, self.policy, self.action = hold, latency, policy, action
        self.sil_factory = sil_factory or tal.RecordingSitl
        self.window_from = window_from
        self.collect_kw = collect_kw

    def run(self, step_idx=0, prev_rgb_ts=0, prev_until_ns=None):
        pose = tal.LatencyPoseSub(self.obs_s)
        subs = {"rgb": tc.FakeSub(after=stamped(tc.image_msg(W, H, 0), self.rgb_ts)),
                "depth": tc.FakeSub(frames=[(1, stamped(tc._fresh_depth(W, H), self.rgb_ts))]),
                "pose": pose, "contact": tc.FakeSub()}
        sil = self.sil_factory(pose, [])
        world = self

        class Student:
            def act(self, msg, prev_action=None):
                if world.ready_s is not None:
                    pose.current = tc.pose_msg(world.ready_s)
                return world.action, 1.0, [0.0, 1.0, 0.0, 0.0]

        def advance(_secs):
            pose.current = tc.pose_msg(self.obs_s + self.hold + 0.01)

        extra = {} if self.latency is None else {"action_latency_s": self.latency}
        with tempfile.TemporaryDirectory() as d, \
                patch.object(collect, "write_rgb_png", return_value=(True, None)), \
                patch.object(collect.time, "sleep", side_effect=advance), \
                patch.object(collect, "stop_executor"):
            row, err = collect.collect_one(
                d, step_idx, self.hold, self.policy, subs, tc.FakePub(), pillars=PILLARS, sil=sil,
                window_from=self.window_from, student=Student() if self.policy == "student" else None,
                prev_rgb_ts=prev_rgb_ts, prev_until_ns=prev_until_ns, **extra, **self.collect_kw)
        return row, err, sil


def drive(rgb_stamps_s, **kw):
    """依次跑若干步，像主循环那样把上一步的 rgb_ts / until_ns 带给下一步。返回行列表（任一步失败就抛）。"""
    rows, prev_rgb, prev_until = [], 0, None
    for k, rs in enumerate(rgb_stamps_s):
        row, err, _ = StepWorld(rs, **kw).run(k, prev_rgb_ts=prev_rgb, prev_until_ns=prev_until)
        if err is not None:
            raise AssertionError(f"step {k}: {err}")
        rows.append(row)
        prev_rgb, prev_until = row["rgb_ts"], row.get("until_ns")
    return rows


def line(row):
    """collect_one 的行经 action_row 写进 actions.jsonl 的样子。"""
    return collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})


# ---------------------------------------------------------------- A1：safe_gap_s

class SafeGapInCollectOne(unittest.TestCase):
    def test_first_step_has_null_gap(self):
        row, err, _ = StepWorld(10.0).run()
        self.assertIsNone(err)
        self.assertIn("safe_gap_s", row, "首步也要有这个键，值为 None（落盘是 null）")
        self.assertIsNone(row["safe_gap_s"])

    def test_gap_is_dispatch_stamp_minus_previous_deadline(self):
        # 上一步观测位姿 9.105 → 截止 = 9.105 + hold 1.0 = 10.105；本步观测位姿 10.005，L=0.2 → 派发戳 10.205；空档 0.100 s
        # （审核 A 的发现 1：L=0.20、frame_dt=0.9、hold=1.0 时恒为 0.100 s）
        row, err, sil = StepWorld(10.0).run(prev_rgb_ts=ns(9.1), prev_until_ns=ns(10.105))
        self.assertIsNone(err)
        self.assertEqual(sil.calls[0][2], ns(10.205), "派发戳 = 观测戳 + L")
        self.assertEqual(row["safe_gap_s"], (ns(10.205) - ns(10.105)) / NS)
        self.assertAlmostEqual(row["safe_gap_s"], 0.100, places=9)

    def test_gap_is_negative_when_next_action_arrives_before_the_deadline(self):
        row, err, _ = StepWorld(10.0).run(prev_rgb_ts=ns(9.1), prev_until_ns=ns(10.300))
        self.assertIsNone(err)
        self.assertAlmostEqual(row["safe_gap_s"], -0.095, places=9)

    def test_gap_is_recorded_without_the_latency_flag_too(self):
        # 旧协议：派发在最新位姿戳（10.005，没有策略耗时），上一步截止 10.105 → 本步在上一步截止之前 0.1 s 派发
        row, err, sil = StepWorld(10.0, latency=None).run(prev_rgb_ts=ns(9.1), prev_until_ns=ns(10.105))
        self.assertIsNone(err)
        self.assertEqual(sil.calls[0][2], ns(10.005))
        self.assertAlmostEqual(row["safe_gap_s"], -0.100, places=9)

    def test_row_carries_this_steps_deadline(self):
        # deadline 协议：执行器截止 = 动作窗口起点（观测戳）+ hold；主循环把它带给下一步
        row, err, _ = StepWorld(10.0).run()
        self.assertIsNone(err)
        self.assertEqual(row["until_ns"], row["action_start_ns"] + ns(1.0))
        self.assertEqual(row["until_ns"], ns(11.005))
        row_h, _, _ = StepWorld(10.0, hold=0.5, latency=0.2).run()
        self.assertEqual(row_h["until_ns"], ns(10.505), "跟着 hold 走")

    def test_two_steps_in_a_row_reproduce_the_constant_gap(self):
        # 连续三步，RGB 戳 10.0 / 10.9 / 11.8（frame_dt 恒 0.9），L=0.2，hold=1.0：第 1、2 步的空档都是 0.100 s
        rows = drive([10.0, 10.9, 11.8])
        self.assertEqual([r["safe_gap_s"] for r in rows][0], None)
        self.assertEqual([round(r["safe_gap_s"], 9) for r in rows[1:]], [0.1, 0.1])


class SafeGapInActionRow(unittest.TestCase):
    def test_default_row_has_safe_gap_even_when_null(self):
        row, err, _ = StepWorld(10.0).run()
        ar = line(row)
        self.assertIn("safe_gap_s", ar)
        self.assertIsNone(ar["safe_gap_s"])
        self.assertEqual(json.loads(json.dumps(ar))["safe_gap_s"], None, "落盘是 null")
        row2, _, _ = StepWorld(10.0).run(prev_rgb_ts=ns(9.1), prev_until_ns=ns(10.105))
        self.assertAlmostEqual(line(row2)["safe_gap_s"], 0.1, places=9)

    def test_until_ns_is_internal_and_not_persisted(self):
        row, _, _ = StepWorld(10.0).run()
        self.assertNotIn("until_ns", line(row))

    def test_new_key_sits_before_the_old_tail_keys(self):
        # 旧键的行内顺序不动：dispatch_lag_s（与带开关时的 late_dispatch）仍在最后
        row, _, _ = StepWorld(10.0, latency=None).run()          # 不给开关：late_dispatch 为 None，不落盘
        keys = list(line(row))
        self.assertEqual(keys[-1], "dispatch_lag_s")
        self.assertLess(keys.index("safe_gap_s"), keys.index("dispatch_lag_s"))
        row_l, _, _ = StepWorld(10.0).run()                      # 给了 --action-latency：多 late_dispatch，仍排在最后
        keys_l = list(line(row_l))
        self.assertEqual(keys_l[-2:], ["dispatch_lag_s", "late_dispatch"])

    def test_default_row_key_set_is_the_old_keys_plus_the_new_one(self):
        # 真 collect_one 的默认行（教师 / 盲前进）经 action_row：键集合 = 旧默认键 + safe_gap_s（A3 再加 obs_gap_s），until_ns 不在内
        row, _, _ = StepWorld(10.0, latency=None).run()
        self.assertEqual(set(line(row)), tc.ROW_KEYS_DEFAULT)
        self.assertEqual(tc.ROW_KEYS_DEFAULT - {"safe_gap_s", "obs_gap_s"}, tc.ROW_KEYS_LEGACY | {"dispatch_lag_s"})
        row_l, _, _ = StepWorld(10.0).run()
        self.assertEqual(set(line(row_l)), tc.ROW_KEYS_DEFAULT | {"late_dispatch"})


# ---------------------------------------------------------------- A1：主循环把上一步的截止带给下一步

def run_loop(extra_argv, rows_or_factory, steps):
    """真 collect.main() 主循环 + 假 collect_one：返回 (退出码, collect_one 收到的 kwargs 列表, manifest 或 None, actions.jsonl 行)。"""
    seen = []
    made = iter(rows_or_factory) if isinstance(rows_or_factory, (list, tuple)) else None
    with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st, \
            contextlib.redirect_stderr(io.StringIO()):
        out, meta, ckpt = (os.path.join(d, n) for n in ("ep", "scene.json", "ckpt.bin"))
        Path(meta).write_text(json.dumps(tc.SCENE))
        Path(ckpt).write_bytes(b"x")
        argv = ["collect.py", "--out", out, "--scene-meta", meta, "--split", sm.split_for_scene(tc.SCENE),
                "--steps", str(steps)] + [a.replace("CKPT", ckpt) for a in extra_argv]
        subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}

        def fake(out_dir, step_idx, *a, **kw):
            seen.append(dict(kw, _prev_rgb_ts=a[5] if len(a) > 5 else kw.get("prev_rgb_ts")))
            row = next(made) if made is not None else rows_or_factory(step_idx)
            return row, None

        st.enter_context(patch.object(sys, "argv", argv))
        st.enter_context(patch.object(collect, "open_subs", return_value=subs))
        st.enter_context(patch.object(collect, "wait_ready", return_value=None))
        st.enter_context(patch.object(collect, "read_camera_info", return_value=None))
        st.enter_context(patch.object(collect.gz_io, "open_twist_publisher", return_value=tc.MainPub()))
        st.enter_context(patch.object(collect, "take_off", return_value=0))
        st.enter_context(patch.object(collect, "stop_executor"))
        st.enter_context(patch.object(collect, "close_subs"))
        st.enter_context(patch.object(collect, "collect_one", side_effect=fake))
        st.enter_context(patch.object(teacher, "warm_free_cells", return_value=0))
        code = 0
        try:
            collect.main()
        except SystemExit as e:
            code = e.code
        man_path = os.path.join(out, "episode_manifest.json")
        manifest = json.loads(Path(man_path).read_text()) if os.path.isfile(man_path) else None
        acts = ([json.loads(x) for x in Path(out, "actions.jsonl").read_text().splitlines() if x.strip()]
                if os.path.isfile(os.path.join(out, "actions.jsonl")) else [])
    return code, seen, manifest, acts


def fake_rows(n, **per_step):
    """n 步假行；per_step 的键给每步的 callable(step) 或常量。until_ns 缺省 = 观测戳 + 1 s（与真 collect_one 一致）。"""
    rows = []
    for i in range(n):
        r = tc.fake_row(i, 1.0 + 0.25 * i)
        r["until_ns"] = r["action_start_ns"] + NS
        for k, v in per_step.items():
            r[k] = v(i) if callable(v) else v
        rows.append(r)
    return rows


class MainLoopCarriesTheDeadline(unittest.TestCase):
    def test_first_call_gets_none_then_the_previous_rows_until_ns(self):
        rows = fake_rows(3)
        code, seen, _, _ = run_loop([], rows, steps=3)
        self.assertEqual(code, 0)
        self.assertEqual([s.get("prev_until_ns", "缺") for s in seen],
                         [None, rows[0]["until_ns"], rows[1]["until_ns"]])

    def test_carried_without_the_latency_flag_and_with_it(self):
        for argv in ([], ["--action-latency", "0.2"]):
            rows = fake_rows(2)
            code, seen, _, _ = run_loop(argv, rows, steps=2)
            self.assertEqual(code, 0, argv)
            self.assertEqual(seen[1].get("prev_until_ns", "缺"), rows[0]["until_ns"], argv)

    def test_persisted_rows_have_the_gap_and_not_the_deadline(self):
        rows = fake_rows(3, safe_gap_s=lambda i: None if i == 0 else 0.1)
        code, _, man, acts = run_loop(["--action-latency", "0.2"], rows, steps=3)
        self.assertEqual(code, 0)
        self.assertEqual([a["safe_gap_s"] for a in acts], [None, 0.1, 0.1])
        self.assertTrue(all("until_ns" not in a for a in acts))


# ---------------------------------------------------------------- A1：summarize_episode 的 safe_gap_s 统计

def safe_run(gaps, **kw):
    """一条飞过线的无碰撞轨迹，第 i 步的行带 safe_gap_s = gaps[i]（None 允许）。"""
    rows = [{"collision": False, "action_executed": 1, "rgb_ts": (10 + i) * NS, "safe_gap_s": g}
            for i, g in enumerate(gaps)]
    return tes.gap_summary(rows, **kw)


class SafeGapInSummary(unittest.TestCase):
    def test_median_and_max_over_the_non_null_gaps(self):
        s = safe_run([None, 0.10, 0.12, 0.08, 0.30])
        self.assertAlmostEqual(s["safe_gap_s_median"], 0.11, places=12)   # 0.08 0.10 0.12 0.30 的中位数
        self.assertEqual(s["safe_gap_s_max"], 0.30)

    def test_negative_gaps_are_part_of_the_distribution(self):
        s = safe_run([None, -0.1, -0.1, 0.0])
        self.assertEqual(s["safe_gap_s_median"], -0.1)
        self.assertEqual(s["safe_gap_s_max"], 0.0)

    def test_no_new_keys_when_rows_have_no_gap_values(self):
        for gaps in ([], [None], [None, None, None]):
            self.assertFalse({"safe_gap_s_median", "safe_gap_s_max"} & set(safe_run(gaps)), gaps)
        rows = tes.gap_rows(1.0, 1.0, 1.0)                    # 旧数据：行里根本没有这个键
        self.assertFalse({"safe_gap_s_median", "safe_gap_s_max"} & set(tes.gap_summary(rows)))

    def test_only_these_two_keys_are_added_and_old_values_do_not_move(self):
        with_gap = safe_run([None, 0.1, 0.1, 0.1])
        without = tes.gap_summary([{"collision": False, "action_executed": 1, "rgb_ts": (10 + i) * NS}
                                   for i in range(4)])
        self.assertEqual(set(with_gap) - set(without), {"safe_gap_s_median", "safe_gap_s_max"})
        self.assertEqual({k: with_gap[k] for k in without}, without)


# ---------------------------------------------------------------- A3：obs_gap_s 与停顿步

class StallPredicate(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(collect.NOMINAL_STEP_PERIOD_S, 0.9)    # 实测：round-2 学习者臂 4,972/4,972 个相邻步对都是 0.9 s
        self.assertEqual(collect.STALL_MARGIN_S, 0.2)

    def test_stall_is_strictly_more_than_nominal_plus_margin(self):
        for g in (None, 0.8, 0.9, 1.0, 1.1, 1.1 + 1e-9):
            self.assertFalse(collect.is_stall_gap(g), g)
        for g in (1.101, 1.2, 1.6, 2.6, 3.6, 5.5):
            self.assertTrue(collect.is_stall_gap(g), g)

    def test_threshold_equals_review_a_rule_on_the_safe_gap(self):
        # 审核 A 用「SAFE 空档 > 0.3 s」判停顿；L=0.2、hold=1.0 时 safe_gap = obs_gap − 0.8，两条阈值是同一条
        for obs_gap in (1.0, 1.1, 1.2, 1.6):
            self.assertEqual(collect.is_stall_gap(obs_gap), round(obs_gap - 0.8, 9) > 0.3)


class ObsGapInCollectOne(unittest.TestCase):
    def test_first_step_has_null_gap(self):
        row, err, _ = StepWorld(10.0).run()
        self.assertIsNone(err)
        self.assertIn("obs_gap_s", row)
        self.assertIsNone(row["obs_gap_s"])

    def test_gap_is_the_rgb_stamp_difference(self):
        row, err, _ = StepWorld(10.0).run(prev_rgb_ts=ns(9.1))
        self.assertIsNone(err)
        self.assertEqual(row["obs_gap_s"], (ns(10.0) - ns(9.1)) / NS)
        self.assertAlmostEqual(row["obs_gap_s"], 0.9, places=9)
        row2, _, _ = StepWorld(10.0).run(prev_rgb_ts=ns(8.4))     # 停顿：1.6 s
        self.assertAlmostEqual(row2["obs_gap_s"], 1.6, places=9)

    def test_gap_is_recorded_without_the_latency_flag_too(self):
        row, err, _ = StepWorld(10.0, latency=None).run(prev_rgb_ts=ns(9.1))
        self.assertIsNone(err)
        self.assertAlmostEqual(row["obs_gap_s"], 0.9, places=9)

    def test_constant_frame_dt_over_a_run(self):
        rows = drive([10.0, 10.9, 11.8, 12.7])
        self.assertEqual([r["obs_gap_s"] for r in rows][0], None)
        self.assertEqual([round(r["obs_gap_s"], 9) for r in rows[1:]], [0.9, 0.9, 0.9])
        self.assertFalse(any(collect.is_stall_gap(r["obs_gap_s"]) for r in rows))

    def test_a_stall_shows_in_both_gaps(self):
        # 第 2 步前停了 0.7 s：RGB 戳 10.9 → 12.5（frame_dt 1.6），L=0.2：obs_gap 1.6、safe_gap = 12.705 − 11.905 = 0.8
        rows = drive([10.0, 10.9, 12.5])
        self.assertAlmostEqual(rows[2]["obs_gap_s"], 1.6, places=9)
        self.assertAlmostEqual(rows[2]["safe_gap_s"], 0.8, places=9)
        self.assertTrue(collect.is_stall_gap(rows[2]["obs_gap_s"]))


class ObsGapInActionRow(unittest.TestCase):
    def test_default_row_has_obs_gap_even_when_null(self):
        row, _, _ = StepWorld(10.0, latency=None).run()
        ar = line(row)
        self.assertIn("obs_gap_s", ar)
        self.assertIsNone(ar["obs_gap_s"])
        row2, _, _ = StepWorld(10.0, latency=None).run(prev_rgb_ts=ns(9.1))
        self.assertAlmostEqual(line(row2)["obs_gap_s"], 0.9, places=9)

    def test_key_order_old_tail_keys_stay_last(self):
        row, _, _ = StepWorld(10.0, latency=None).run()
        keys = list(line(row))
        self.assertEqual(keys[-1], "dispatch_lag_s")
        self.assertEqual(keys[keys.index("safe_gap_s") + 1], "obs_gap_s")
        self.assertLess(keys.index("obs_gap_s"), keys.index("dispatch_lag_s"))
        row_l, _, _ = StepWorld(10.0).run()
        self.assertEqual(list(line(row_l))[-2:], ["dispatch_lag_s", "late_dispatch"])

    def test_default_key_set_is_the_old_keys_plus_the_two_new_ones(self):
        row, _, _ = StepWorld(10.0, latency=None).run()
        self.assertEqual(set(line(row)), tc.ROW_KEYS_DEFAULT)
        self.assertEqual(tc.ROW_KEYS_DEFAULT, tc.ROW_KEYS_LEGACY | {"dispatch_lag_s", "safe_gap_s", "obs_gap_s"})


class StallStepsInManifest(unittest.TestCase):
    OBS = [None, 0.9, 1.6, 0.9, 2.6, 1.1]      # 1.6 与 2.6 是停顿；1.1 刚好在阈值上，不算

    def test_count_only_with_the_latency_flag(self):
        rows = fake_rows(len(self.OBS), obs_gap_s=lambda i: self.OBS[i])
        code, _, man, acts = run_loop(["--action-latency", "0.2"], rows, steps=len(rows))
        self.assertEqual(code, 0)
        self.assertEqual(man["stall_steps"], 2)
        self.assertEqual(set(man), tal.MANIFEST_KEYS_TEACHER | {"action_latency_s", "late_dispatch_steps", "stall_steps"})
        self.assertEqual([a["obs_gap_s"] for a in acts], self.OBS)

    def test_default_manifest_key_set_is_unchanged(self):
        rows = fake_rows(len(self.OBS), obs_gap_s=lambda i: self.OBS[i])
        code, _, man, acts = run_loop([], rows, steps=len(rows))
        self.assertEqual(code, 0)
        self.assertEqual(set(man), tal.MANIFEST_KEYS_TEACHER, "不给开关：manifest 键集合与旧版一字不差")
        self.assertEqual([a["obs_gap_s"] for a in acts], self.OBS, "行里照常记")


# ---------------------------------------------------------------- A3：summarize_episode

def obs_run(gaps, **kw):
    """一条飞过线的无碰撞轨迹，第 i 步的行带 obs_gap_s = gaps[i]（首项一般 None）与对应的 rgb_ts。"""
    ts, rows = 10 * NS, []
    for i, g in enumerate(gaps):
        ts = ts if g is None else ts + int(round(g * NS))
        rows.append({"step": i, "collision": False, "action_executed": 1, "rgb_ts": ts, "obs_gap_s": g})
    return tes.gap_summary(rows, **kw)


class StallInSummary(unittest.TestCase):
    def test_max_count_and_ids(self):
        s = obs_run([None, 0.9, 1.6, 0.9, 2.6, 1.1])
        self.assertEqual(s["obs_gap_s_max"], 2.6)
        self.assertEqual(s["stall_steps"], 2)
        self.assertEqual(s["stall_step_ids"], [2, 4])

    def test_no_stall_writes_zero_and_no_id_list(self):
        s = obs_run([None, 0.9, 0.9, 1.0])
        self.assertEqual(s["stall_steps"], 0)
        self.assertNotIn("stall_step_ids", s)
        self.assertEqual(s["obs_gap_s_max"], 1.0)

    def test_equals_w1_3_max_frame_gap_on_data_with_both(self):
        s = obs_run([None, 0.9, 1.6, 0.9], hold=1.0)
        self.assertEqual(s["obs_gap_s_max"], s["max_frame_gap_s"])

    def test_no_new_keys_without_the_field(self):
        rows = tes.gap_rows(1.0, 2.6, 1.0)                    # 旧数据：有 rgb_ts，没有 obs_gap_s
        s = tes.gap_summary(rows, hold=1.0)
        self.assertFalse({"obs_gap_s_max", "stall_steps", "stall_step_ids"} & set(s))
        self.assertIn("max_frame_gap_s", s, "W1-3 的键照旧")
        self.assertFalse({"obs_gap_s_max", "stall_steps"} & set(obs_run([])), "空集")
        self.assertFalse({"obs_gap_s_max", "stall_steps"} & set(obs_run([None])), "只有首步（None）")

    def test_only_these_keys_are_added_and_old_values_do_not_move(self):
        with_gap = obs_run([None, 0.9, 1.6, 0.9])
        without = tes.gap_summary(tes.gap_rows(0.9, 1.6, 0.9))      # 同样的 rgb_ts，行里没有 obs_gap_s
        self.assertEqual(set(with_gap) - set(without), {"obs_gap_s_max", "stall_steps", "stall_step_ids"})
        self.assertEqual({k: with_gap[k] for k in without}, without)


# ---------------------------------------------------------------- A3：scan_episodes 与完整 eval

def scan_summary(seed, stall=None, late=None, ok=True):
    s = {"seed": seed, "collect_rc": 0 if ok else 1, "steps": 10 if ok else 0, "dedup_key": f"k{seed}", "success": True,
         "collision": False, "max_x": 6.0, "infer_ms": [], "geometry_gid": f"g{seed}", "stable_termination": None,
         "stderr_tail": [], "performance": None, "render_backend_requested": None}
    if stall is not None:
        s.update({"stall_steps": len(stall), "obs_gap_s_max": 1.6, "safe_gap_s_max": 0.8})
        if stall:
            s["stall_step_ids"] = list(stall)
    if late is not None:
        s.update({"late_dispatch_steps": len(late), "late_step_ids": list(late),
                  "dispatch_lag_s_median": 0.3, "dispatch_lag_s_max": 0.31})
    return s


QUIET = lambda *a, **k: None  # noqa: E731


class StallInScan(unittest.TestCase):
    def run_scan(self, summaries, **kw):
        return ec.scan_episodes(list(range(len(summaries))), len(summaries), lambda seed, _seen: (summaries[seed], None),
                                log=QUIET, **kw)

    def test_with_the_flag_a_stall_episode_is_an_infrastructure_failure(self):
        res, fails, _, _, attempted = self.run_scan([scan_summary(0, stall=[]), scan_summary(1, stall=[2, 4])],
                                                    fail_on_stall=True)
        self.assertEqual([r["seed"] for r in res], [0])
        self.assertEqual(attempted, [0, 1], "占请求名额")
        self.assertEqual(len(fails), 1)
        f = fails[0]
        self.assertEqual(f["seed"], 1)
        self.assertTrue(f["error"].startswith("stall:"), f["error"])
        self.assertEqual((f["stall_steps"], f["stall_step_ids"], f["obs_gap_s_max"], f["safe_gap_s_max"]), (2, [2, 4], 1.6, 0.8))
        self.assertNotIn("late_dispatch_steps", f)

    def test_without_the_flag_it_is_only_recorded(self):
        res, fails, _, _, _ = self.run_scan([scan_summary(0, stall=[]), scan_summary(1, stall=[2, 4])])
        self.assertEqual([r["seed"] for r in res], [0, 1])
        self.assertEqual(fails, [])
        self.assertEqual(res[1]["stall_steps"], 2)

    def test_late_only_entry_is_unchanged(self):
        # 只有迟到步：failed[] 条目的键、键序与值与 W1-0 一字不差
        _, fails, _, _, _ = self.run_scan([scan_summary(0, late=[1, 3])], fail_on_stall=True)
        self.assertEqual(list(fails[0]), ["seed", "rc", "error", "late_dispatch_steps", "late_step_ids",
                                          "dispatch_lag_s_median", "dispatch_lag_s_max", "stderr_tail", "performance",
                                          "render_backend_requested"])
        self.assertTrue(fails[0]["error"].startswith("late_dispatch: 2 步迟到"))

    def test_late_and_stall_in_one_episode_is_one_failure_with_both(self):
        _, fails, _, _, attempted = self.run_scan([scan_summary(0, stall=[2], late=[1])], fail_on_stall=True)
        self.assertEqual(attempted, [0])
        self.assertEqual(len(fails), 1)
        f = fails[0]
        self.assertTrue(f["error"].startswith("late_dispatch:") and "stall:" in f["error"], f["error"])
        self.assertEqual((f["late_dispatch_steps"], f["stall_steps"]), (1, 1))

    def test_zero_stall_steps_is_not_a_failure(self):
        res, fails, _, _, _ = self.run_scan([scan_summary(0, stall=[])], fail_on_stall=True)
        self.assertEqual((len(res), fails), (1, []))

    def test_collect_failure_keeps_its_own_entry(self):
        _, fails, _, _, _ = self.run_scan([scan_summary(0, stall=[2], ok=False)], fail_on_stall=True)
        self.assertNotIn("error", fails[0])
        self.assertEqual(fails[0]["rc"], 1)


def stall_actions(obs_gaps, late_ids=()):
    """eval_rows 的每步加 obs_gap_s（与 safe_gap_s = obs_gap − 0.8）；late_ids 里的步带 late_dispatch。"""
    rows = tal.eval_rows([0.2] * len(obs_gaps), late_ids=late_ids, flagged=True)
    for i, (r, g) in enumerate(zip(rows, obs_gaps)):
        r["obs_gap_s"] = g
        r["safe_gap_s"] = None if g is None else round(g - 0.8, 9)
        r["rgb_ts"] = (10 + i) * NS
    return rows


class StallInEval(unittest.TestCase):
    def test_with_the_flag_stall_episode_goes_to_failed_and_is_listed(self):
        report, _ = tal.EvalHarness.run(["--action-latency", "0.2"], stall_actions([None, 0.9, 1.6, 0.9]),
                                        {"action_latency_s": 0.2, "late_dispatch_steps": 0, "stall_steps": 1})
        self.assertEqual(report["per_episode"], [], "停顿集不进完成集")
        self.assertEqual(report["failed_count"], 1)
        f = report["failed"][0]
        self.assertTrue(f["error"].startswith("stall:"), f["error"])
        self.assertEqual((f["stall_steps"], f["stall_step_ids"]), (1, [2]))
        self.assertAlmostEqual(f["obs_gap_s_max"], 1.6, places=9)
        self.assertAlmostEqual(f["safe_gap_s_max"], 0.8, places=9)
        self.assertEqual(report["stall_episodes"], [{"seed": f["seed"], "stall_steps": 1, "stall_step_ids": [2],
                                                     "obs_gap_s_max": f["obs_gap_s_max"], "safe_gap_s_max": f["safe_gap_s_max"]}])
        self.assertEqual(report["late_dispatch_episodes"], [])
        self.assertEqual(report["requested_success_rate"], 0.0, "占请求名额、不算成功")

    def test_clean_episode_with_the_flag_is_kept_and_carries_the_new_keys(self):
        report, _ = tal.EvalHarness.run(["--action-latency", "0.2"], stall_actions([None, 0.9, 0.9, 0.9]),
                                        {"action_latency_s": 0.2, "late_dispatch_steps": 0, "stall_steps": 0})
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(report["stall_episodes"], [])
        ep = report["per_episode"][0]
        self.assertEqual((ep["stall_steps"], ep["obs_gap_s_max"]), (0, 0.9))
        self.assertAlmostEqual(ep["safe_gap_s_median"], 0.1, places=9)
        self.assertAlmostEqual(ep["safe_gap_s_max"], 0.1, places=9)

    def test_without_the_flag_it_is_only_recorded(self):
        report, cmds = tal.EvalHarness.run([], stall_actions([None, 0.9, 1.6, 0.9]))
        self.assertEqual(report["failed_count"], 0)
        self.assertNotIn("stall_episodes", report)
        self.assertNotIn("--action-latency", cmds["collect"])
        ep = report["per_episode"][0]
        self.assertEqual((ep["stall_steps"], ep["stall_step_ids"]), (1, [2]))

    def test_late_and_stall_both_listed(self):
        report, _ = tal.EvalHarness.run(["--action-latency", "0.2"], stall_actions([None, 0.9, 1.6, 0.9], late_ids=(1,)),
                                        {"action_latency_s": 0.2, "late_dispatch_steps": 1, "stall_steps": 1})
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual([e["seed"] for e in report["late_dispatch_episodes"]], [report["failed"][0]["seed"]])
        self.assertEqual([e["seed"] for e in report["stall_episodes"]], [report["failed"][0]["seed"]])

    def test_old_data_without_the_field_adds_no_top_key(self):
        report, _ = tal.EvalHarness.run(["--action-latency", "0.2"], tal.eval_rows([0.2, 0.2, 0.2]),
                                        {"action_latency_s": 0.2, "late_dispatch_steps": 0})
        self.assertEqual(report["failed_count"], 0)
        ep = report["per_episode"][0]
        self.assertFalse({"stall_steps", "obs_gap_s_max", "safe_gap_s_max"} & set(ep))


# ---------------------------------------------------------------- A5：plugin_ack_sim_t

def f32(x):
    """固件里 action_until_s 是 float（executor.h），state[5] 是它转成 double 的值。"""
    return struct.unpack("f", struct.pack("f", x))[0]


def state_frame(sim_t, base_s, action, until_sim):
    """一帧 /e1_mini/state（Double_V，40 个值）：[0]=SITL 秒 = sim_t − base_s，[4]=动作（不在 ACTING 为 -1），
    [5]=截止（SITL 秒，float32），[32]=gz 仿真秒。"""
    data = [0.0] * ce.STATE_FIELDS
    data[ce.IDX_T] = sim_t - base_s
    data[ce.IDX_ACTION] = float(action)
    data[ce.IDX_UNTIL] = f32(until_sim - base_s)
    data[ce.IDX_SIM_T] = sim_t
    return {"data": data}


class RingSub:
    """TopicSub 的替身：只给 frames / latest / healthy / close。"""

    def __init__(self, frames):
        self._frames = list(enumerate(frames, 1))

    def frames(self):
        return list(self._frames)

    def latest(self):
        return self._frames[-1][1] if self._frames else None

    def healthy(self):
        return True

    def close(self):
        pass


class RecPub:
    def __init__(self):
        self.sent = []

    def action(self, action, until_s):
        self.sent.append((action, until_s))

    def cmd(self, text):
        pass

    def close(self):
        pass


def executor(frames, hold=1.0):
    ex = ce.CfSitlExecutor(ce.CfSitlParams("x.json", b"{}", {}, lib_version=31), hold_s=hold)
    ex._sub, ex._pub = RingSub(frames), RecPub()
    return ex


def grid(t0, t1, step=0.005):
    """[t0, t1] 上的 200 Hz 网格（整数 ms 算，避免浮点累加误差）。"""
    return [round(k * step, 6) for k in range(int(round(t0 / step)), int(round(t1 / step)) + 1)]


class AckSimTime(unittest.TestCase):
    BASE = 3.0

    def frames(self, ack_at, new_until, new_action=1, old_action=1, old_until=11.105, lo=10.2, hi=10.3):
        """lo..hi 的 200 Hz 帧：ack_at 之前是上一条动作（截止 old_until），之后是本步动作（截止 new_until）。"""
        return [state_frame(t, self.BASE, new_action, new_until) if t >= ack_at - 1e-9
                else state_frame(t, self.BASE, old_action, old_until) for t in grid(lo, hi)]

    def test_none_before_any_dispatch(self):
        ex = executor(self.frames(10.21, 11.005))
        self.assertIsNone(ex.ack_sim_t())

    def test_none_without_a_state_subscription(self):
        ex = executor([])
        ex.dispatch(1, ns(10.005))
        ex._sub = None
        self.assertIsNone(ex.ack_sim_t())

    def test_first_frame_that_carries_the_sent_deadline(self):
        ex = executor(self.frames(10.21, 11.005))
        ex.dispatch(1, ns(10.005))                   # 截止 = 10.005 + hold 1.0 = 11.005
        self.assertEqual(ex._pub.sent, [(1, 11.005)])
        self.assertEqual(ex.ack_sim_t(), 10.21)

    def test_forward_after_forward_is_found_by_the_deadline_not_the_action_field(self):
        # 上一步的 FORWARD 到 11.105 才截止，本步的 FORWARD 在 10.21 生效：动作字段一直是 1，只有截止在变（下一步派发紧接上一步的情形）
        frames = self.frames(10.21, 11.905, new_action=1, old_action=1, old_until=11.105)
        self.assertEqual({f["data"][ce.IDX_ACTION] for f in frames}, {1.0})
        ex = executor(frames)
        ex.dispatch(1, ns(10.905))
        self.assertEqual(ex.ack_sim_t(), 10.21)
        self.assertNotEqual(ex.ack_sim_t(), frames[0]["data"][ce.IDX_SIM_T], "不能把第一帧 FORWARD 当成确认")

    def test_float32_rounding_of_the_deadline_is_tolerated(self):
        base = 88.123456789
        until_sim = 100.4049999
        self.assertNotEqual(f32(until_sim - base), until_sim - base, "要真的有舍入误差才算数")
        frames = [state_frame(t, base, 1, until_sim) if t >= 99.5 else state_frame(t, base, 1, 99.9) for t in grid(99.4, 99.6)]
        ex = executor(frames)
        ex.dispatch(1, int(until_sim * NS) - ns(1.0))
        self.assertEqual(ex.ack_sim_t(), 99.5)

    def test_each_step_finds_its_own_frame(self):
        frames = self.frames(10.21, 11.005) + self.frames(11.11, 11.905, old_until=11.005, lo=11.0, hi=11.2)
        ex = executor(frames)
        ex.dispatch(1, ns(10.005))
        self.assertEqual(ex.ack_sim_t(), 10.21)
        ex.dispatch(2, ns(10.905))
        self.assertEqual(ex.ack_sim_t(), 11.11)

    def test_none_when_no_frame_carries_the_deadline(self):
        ex = executor(self.frames(10.21, 11.005))
        ex.dispatch(1, ns(12.005))                   # 本步的截止 13.005 没有任何一帧带过（动作还没到插件，或帧已被缓冲淘汰）
        self.assertIsNone(ex.ack_sim_t())

    def test_a_late_action_that_never_acts_is_still_acked_by_its_deadline(self):
        # 到达时窗口已过的动作插件照常传给固件，固件立即视为过期（不进 ACTING，state[4] = -1），但截止字段已更新
        frames = self.frames(10.21, 11.005, new_action=-1)
        ex = executor(frames)
        ex.dispatch(1, ns(10.005))
        self.assertEqual(ex.ack_sim_t(), 10.21)

    def test_short_or_broken_frames_are_skipped(self):
        frames = [{"data": [0.0] * (ce.STATE_FIELDS - 1)}, {"data": None}, {}] + self.frames(10.21, 11.005)
        ex = executor(frames)
        ex.dispatch(1, ns(10.005))
        self.assertEqual(ex.ack_sim_t(), 10.21)

    def test_the_state_subscription_keeps_three_seconds_of_frames(self):
        # 主线程每 20 ms 轮询一次（负载高时更慢）才去读，state 是 200 Hz：缓冲要比一步窗口（1 s = 200 帧）大得多
        made = {}

        def fake_sub(topic, **kw):
            made.update(kw, topic=topic)
            return RingSub([])

        ex = executor([])
        with patch.object(ce.gz_io, "TopicSub", side_effect=fake_sub), patch.object(ce, "STATE_WAIT_S", 0.03), \
                patch.object(ce, "no_state_reason", return_value="测试"):
            with self.assertRaises(ce.WaitTimeout):
                ex.start()
        self.assertEqual(made["topic"], ce.STATE_TOPIC)
        self.assertGreaterEqual(made["buf"] * 0.005, 3.0)
        self.assertIs(made["stamp_fn"], ce.state_stamp_ns)


class AckSitl(tal.RecordingSitl):
    """带 ack_sim_t 的 cf_sitl 执行器替身；记下被读那一刻的位姿戳（要在 hold 窗口走完之后）。"""
    ack = 11.234

    def ack_sim_t(self):
        self.pose_at_ack = collect._pose_stamp(self.pose.latest())
        return self.ack


class AckNullSitl(AckSitl):
    ack = None


class AckInCollectOne(unittest.TestCase):
    def test_row_carries_the_executors_ack(self):
        row, err, sil = StepWorld(10.0, sil_factory=AckSitl).run()
        self.assertIsNone(err)
        self.assertEqual(row["plugin_ack_sim_t"], 11.234)
        self.assertEqual(line(row)["plugin_ack_sim_t"], 11.234)

    def test_ack_is_read_after_the_window_has_run(self):
        row, _, sil = StepWorld(10.0, sil_factory=AckSitl).run()
        self.assertGreaterEqual(sil.pose_at_ack, row["until_ns"], "窗口走完（位姿戳过了截止）之后才去读，确认帧一定已在缓冲里")

    def test_unconfirmed_ack_is_a_null_not_a_missing_key(self):
        row, _, _ = StepWorld(10.0, sil_factory=AckNullSitl).run()
        self.assertIn("plugin_ack_sim_t", row)
        self.assertIsNone(row["plugin_ack_sim_t"])
        ar = line(row)
        self.assertIn("plugin_ack_sim_t", ar)
        self.assertIsNone(json.loads(json.dumps(ar))["plugin_ack_sim_t"])

    def test_no_key_when_the_executor_cannot_tell(self):
        row, _, _ = StepWorld(10.0, latency=None).run()          # RecordingSitl / x3_velocity：没有 ack_sim_t
        self.assertNotIn("plugin_ack_sim_t", row)
        self.assertNotIn("plugin_ack_sim_t", line(row))

    def test_key_position_keeps_the_old_tail_keys_last(self):
        row, _, _ = StepWorld(10.0, latency=None, sil_factory=AckSitl).run()
        keys = list(line(row))
        self.assertEqual(keys[keys.index("obs_gap_s") + 1], "plugin_ack_sim_t")
        self.assertEqual(keys[-1], "dispatch_lag_s")
        self.assertEqual(set(line(row)), tc.ROW_KEYS_DEFAULT | {"plugin_ack_sim_t"})
        row_l, _, _ = StepWorld(10.0, sil_factory=AckSitl).run()
        self.assertEqual(list(line(row_l))[-2:], ["dispatch_lag_s", "late_dispatch"])

    def test_late_criterion_is_not_touched(self):
        # 先量后定：这一版只记录。确认时刻再晚也不影响 late_dispatch（派发时刻没变）
        class VeryLate(AckSitl):
            ack = 99.0
        row, _, _ = StepWorld(10.0, sil_factory=VeryLate).run()
        self.assertIs(row["late_dispatch"], False)


# ---------------------------------------------------------------- A1/A3 补：没有新键的行不凭空多键

class LegacyRowsKeepTheirKeySet(unittest.TestCase):
    """action_row 只在行里有键时才写新键：真 collect_one 的行一定有（首步值为 None，落盘 null），
    旧代码构造的行、别的测试里的合成行没有，就不能凭空多出 safe_gap_s / obs_gap_s（test_save_depth.py 钉的就是这一条）。"""

    LEGACY = {"step": 3, "timestamp_ns": 10, "action_expert": 1, "action_executed": 1, "collision": False, "rgb_ok": True,
              "frame": "rgb_00003.png", "src": None, "rgb_ts": 10, "depth_ts": 9, "depth_rgb_gap_ns": 1,
              "pose_before_ts": 11, "pose_rgb_gap_ns": 1, "dyaw_deg": 0.0, "displacement_m": 0.1, "infer_ms": None,
              "action_start_ns": 12, "pose_before": {"x": 0.0, "y": 0.0, "z": 2.0}, "yaw_before": 0.0}

    def test_row_without_the_new_keys_is_written_without_them(self):
        out = collect.action_row(dict(self.LEGACY))
        self.assertEqual(set(out) & {"safe_gap_s", "obs_gap_s", "plugin_ack_sim_t"}, set())
        self.assertEqual(set(out), set(tc.ROW_KEYS_LEGACY))

    def test_real_first_step_row_has_them_as_null(self):
        row, _, _ = StepWorld(10.0, latency=None).run()
        out = line(row)
        self.assertIn("safe_gap_s", out)
        self.assertIn("obs_gap_s", out)
        self.assertEqual((out["safe_gap_s"], out["obs_gap_s"]), (None, None))

    def test_present_none_is_kept_and_present_value_is_written(self):
        out = collect.action_row(dict(self.LEGACY, safe_gap_s=None, obs_gap_s=0.9))
        self.assertIsNone(out["safe_gap_s"])
        self.assertEqual(out["obs_gap_s"], 0.9)
        self.assertEqual(list(out)[-2:], ["safe_gap_s", "obs_gap_s"])


# ---------------------------------------------------------------- A3 补：名义步周期随 hold 计算（hold != 1.0 时不误判、不漏判）

class StallPredicateHoldScaling(unittest.TestCase):
    def test_nominal_period_tracks_hold_minus_camera_period(self):
        self.assertAlmostEqual(collect.nominal_step_period_s(None), 0.9, places=9)
        self.assertAlmostEqual(collect.nominal_step_period_s(1.0), 0.9, places=9)
        self.assertAlmostEqual(collect.nominal_step_period_s(0.5), 0.4, places=9)
        self.assertAlmostEqual(collect.nominal_step_period_s(2.0), 1.9, places=9)

    def test_hold_two_seconds_does_not_false_positive_on_normal_steps(self):
        # hold=2.0 s，名义间隔 1.9 s，阈值 2.1 s：1.9 s 与 2.1 s 不算停顿，2.2 s 才算
        self.assertFalse(collect.is_stall_gap(1.9, hold=2.0))
        self.assertFalse(collect.is_stall_gap(2.1, hold=2.0))
        self.assertTrue(collect.is_stall_gap(2.2, hold=2.0))
        rows = fake_rows(4, obs_gap_s=lambda i: None if i == 0 else 1.9)
        code, _, man, _ = run_loop(["--hold", "2.0", "--action-latency", "0.2"], rows, steps=4)
        self.assertEqual(code, 0)
        self.assertEqual(man["stall_steps"], 0)
        report, _ = tal.EvalHarness.run(["--hold", "2.0", "--action-latency", "0.2"],
                                        stall_actions([None, 1.9, 1.9, 1.9]),
                                        {"action_latency_s": 0.2, "late_dispatch_steps": 0, "stall_steps": 0})
        self.assertEqual(report["failed_count"], 0)
        self.assertEqual(report["per_episode"][0]["stall_steps"], 0)

    def test_hold_half_second_catches_sub_second_stall(self):
        # hold=0.5 s，名义间隔 0.4 s，阈值 0.6 s：0.8 s 已多停 0.4 s，必须检出停顿并进 failed[]
        self.assertFalse(collect.is_stall_gap(0.4, hold=0.5))
        self.assertFalse(collect.is_stall_gap(0.6, hold=0.5))
        self.assertTrue(collect.is_stall_gap(0.8, hold=0.5))
        rows = fake_rows(4, obs_gap_s=lambda i: [None, 0.4, 0.8, 0.4][i])
        code, _, man, _ = run_loop(["--hold", "0.5", "--action-latency", "0.2"], rows, steps=4)
        self.assertEqual(code, 0)
        self.assertEqual(man["stall_steps"], 1)
        report, _ = tal.EvalHarness.run(["--hold", "0.5", "--action-latency", "0.2"],
                                        stall_actions([None, 0.4, 0.8, 0.4]),
                                        {"action_latency_s": 0.2, "late_dispatch_steps": 0, "stall_steps": 1})
        self.assertEqual(report["failed_count"], 1)
        self.assertIn("名义步周期 0.4 s", report["failed"][0]["error"])
        self.assertEqual(report["stall_episodes"][0]["stall_step_ids"], [2])


# ---------------------------------------------------------------- A1/A3/A5 补：默认落盘产物除新增记账键外逐字节不变

class DefaultBitwiseInvariance(unittest.TestCase):
    """文件落盘级回归：不带 --action-latency 时，episode_manifest.json 与 oracle.jsonl 逐字节不变；
    actions.jsonl 与 eval.json 仅多本卡第一部分的记账键，剔除这些新键后重新序列化的字节串与旧基线逐字节一致。"""

    NEW_ROW_KEYS = ("safe_gap_s", "obs_gap_s", "plugin_ack_sim_t")
    NEW_EP_KEYS = ("safe_gap_s_median", "safe_gap_s_max", "obs_gap_s_max", "stall_steps", "stall_step_ids")

    @staticmethod
    def _collect_bytes(rows):
        with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st, \
                contextlib.redirect_stderr(io.StringIO()):
            out, meta = os.path.join(d, "ep"), os.path.join(d, "scene.json")
            Path(meta).write_text(json.dumps(tc.SCENE))
            argv = ["collect.py", "--out", out, "--scene-meta", meta,
                    "--split", sm.split_for_scene(tc.SCENE), "--steps", str(len(rows))]
            subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}
            it = iter(rows)
            st.enter_context(patch.object(sys, "argv", argv))
            st.enter_context(patch.object(collect, "open_subs", return_value=subs))
            st.enter_context(patch.object(collect, "wait_ready", return_value=None))
            st.enter_context(patch.object(collect, "read_camera_info", return_value=None))
            st.enter_context(patch.object(collect.gz_io, "open_twist_publisher", return_value=tc.MainPub()))
            st.enter_context(patch.object(collect, "take_off", return_value=0))
            st.enter_context(patch.object(collect, "stop_executor"))
            st.enter_context(patch.object(collect, "close_subs"))
            st.enter_context(patch.object(collect, "collect_one", side_effect=lambda *_a, **_kw: (next(it), None)))
            st.enter_context(patch.object(teacher, "warm_free_cells", return_value=0))
            collect.main()
            return (Path(out, "episode_manifest.json").read_bytes(),
                    Path(out, "oracle.jsonl").read_bytes(),
                    Path(out, "actions.jsonl").read_text().splitlines())

    def test_default_collect_files_are_byte_identical_except_new_row_keys(self):
        expected_legacy_order = [
            "step", "timestamp_ns", "action_expert", "action_executed", "collision", "rgb_ok", "frame",
            "src", "rgb_ts", "depth_ts", "depth_rgb_gap_ns", "pose_ts", "pose_rgb_gap_ns",
            "dyaw_deg", "displacement_m", "infer_ms", "action_start_ns", "pose_before", "yaw_before",
            "dispatch_lag_s",
        ]
        expected_oracle_order = ["step", "timestamp_ns", "pose_gt", "action_start_ns", "action_end_ns", "speed_mps"]
        old_rows = []
        new_rows = []
        for i, g in enumerate((None, 0.9, 1.6)):   # 含一步停顿（1.6 s）：若默认 manifest 偷记 stall_steps 会立刻不一致
            base = tc.fake_row(i, 1.0 + 0.25 * i, dispatch_lag_s=0.129)
            for k in self.NEW_ROW_KEYS:
                base.pop(k, None)
            old_rows.append(dict(base))
            new_rows.append(dict(base,
                                 safe_gap_s=None if g is None else round(g - 0.8, 9),
                                 obs_gap_s=g,
                                 plugin_ack_sim_t=10.2 + 0.9 * i))
        man_old, ora_old, acts_old = self._collect_bytes(old_rows)
        man_new, ora_new, acts_new = self._collect_bytes(new_rows)
        self.assertEqual(man_new, man_old, "默认 episode_manifest.json 必须逐字节不变")
        self.assertEqual(set(json.loads(man_new)), tal.MANIFEST_KEYS_TEACHER)
        self.assertEqual(ora_new, ora_old, "默认 oracle.jsonl 必须逐字节不变")
        for ora_line in ora_new.decode().splitlines():
            self.assertEqual(list(json.loads(ora_line)), expected_oracle_order)
        self.assertEqual(len(acts_new), len(acts_old))
        for old_line, new_line in zip(acts_old, acts_new):
            obj = json.loads(new_line)
            self.assertEqual(set(obj) - set(json.loads(old_line)), set(self.NEW_ROW_KEYS))
            for k in self.NEW_ROW_KEYS:
                obj.pop(k)
            self.assertEqual(list(obj), expected_legacy_order)
            self.assertEqual(json.dumps(obj), old_line, "剔除新增键后旧键序与旧值必须逐字节一致")

    def test_default_eval_json_is_byte_identical_except_new_episode_keys(self):
        old_acts = []
        new_acts = []
        for i, g in enumerate((None, 0.9, 1.6, 0.9)):
            base = {"step": i, "action_executed": 1, "collision": False, "infer_ms": None,
                    "rgb_ts": (10 + i) * NS, "dispatch_lag_s": 0.13}
            old_acts.append(dict(base))
            new_acts.append(dict(base, obs_gap_s=g, safe_gap_s=None if g is None else round(g - 0.8, 9),
                                 plugin_ack_sim_t=10.13 + i))
        with patch.object(ec.time, "monotonic", return_value=100.0):
            rep_old, _ = tal.EvalHarness.run([], old_acts)
            rep_new, _ = tal.EvalHarness.run([], new_acts)
        self.assertEqual(list(rep_new), list(rep_old), "默认 eval.json 顶层键名与顺序必须不变")
        added = set(rep_new["per_episode"][0]) - set(rep_old["per_episode"][0])
        self.assertEqual(added, set(self.NEW_EP_KEYS))
        stripped = json.loads(json.dumps(rep_new))
        for k in self.NEW_EP_KEYS:
            stripped["per_episode"][0].pop(k, None)
        self.assertEqual(json.dumps(stripped, ensure_ascii=False, indent=2),
                         json.dumps(rep_old, ensure_ascii=False, indent=2),
                         "剔除 per_episode 新增记账键后 eval.json 必须逐字节一致")


if __name__ == "__main__":
    unittest.main()
