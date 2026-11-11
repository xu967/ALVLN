#!/usr/bin/env python3
"""--rule-version v1 的接线测试（计划 D6）：collect_one 的 v1 分支（记忆、条目、新增落盘字段）、collect.main 主循环维护记忆列表、
manifest 键、参数校验；eval_closed_loop 的透传、校验与 eval.json 记录。v0（默认）的 actions 行、manifest、命令行逐位不变。

规则本身的行为在 test_rule_v1.py；v0 的接线在 test_rule_oracle_wiring.py。不起 Gazebo，替身沿用那份的。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_rule_v1_wiring.py
"""
import argparse
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect
import rule_policy
import split_manifest as sm
import test_collect as tc
import test_eval_sim09v2 as t09
import test_rule_oracle_wiring as t0     # 只借 pose_with_yaw / AHEAD 等助手；不 import 它的测试类（会让 unittest 重复跑）

LEFT, RIGHT, FORWARD, SAFE = collect.YAW_LEFT, collect.YAW_RIGHT, collect.FORWARD_SLOW, collect.SAFE_NEUTRAL
RULE_V1_ARGS = ["--policy", "rule-oracle", "--rule-version", "v1"]
AHEAD = t0.AHEAD      # 柱面离机体 0.6 m，正前方


def v1_step(pillars=AHEAD, x=0.0, y=0.0, yaw_deg=0.0, step_idx=0, rule_mem=None, **kw):
    """真 collect_one + rule-oracle v1 分支。返回 (row, err)。"""
    w, h = 4, 6
    pose = tc.ClockPoseSub(t0.pose_with_yaw(10.2, x, y, yaw_deg))
    subs = {"rgb": tc.FakeSub(after=tc.image_msg(w, h, 10)), "depth": tc.FakeSub(frames=[(1, tc._fresh_depth(w, h))]),
            "pose": pose, "contact": tc.FakeSub()}

    def advance(_secs):
        pose.current = t0.pose_with_yaw(11.2, x + 0.25, y, yaw_deg)

    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        return collect.collect_one(d, step_idx, 1.0, "rule-oracle", subs, tc.FakePub(), pillars=pillars,
                                   rule_version="v1", rule_mem=[] if rule_mem is None else rule_mem, **kw)


class CollectOneV1Tests(unittest.TestCase):
    def test_blocked_ahead_row(self):
        mem = []
        row, err = v1_step(rule_mem=mem)
        self.assertIsNone(err)
        self.assertEqual((row["action_executed"], row["action_expert"]), (LEFT, LEFT), "n=0 无 last_turn 平局 → V1-left")
        self.assertEqual((row["rule_entry"], row["executed_by"]), ("V1-left", "rule_oracle"))
        self.assertEqual(row["rule_clear"], [False, False, False])
        self.assertGreater(row["rule_mem_n"], 3)
        self.assertEqual(row["rule_mem_n"], len(mem), "记忆列表原地更新，点数 = 本步换到机体系的点数")
        self.assertTrue(all(t == 0 for _x, _y, t in mem))
        self.assertEqual(len(row["rule_cols"]), 20)

    def test_open_field_forward_and_back(self):
        row, _ = v1_step(pillars=[])
        self.assertEqual((row["action_executed"], row["rule_entry"], row["rule_mem_n"]), (FORWARD, "V1-fwd", 0))
        self.assertEqual(row["rule_clear"], [True, True, True])
        row, _ = v1_step(pillars=[], net_yaw=2)
        self.assertEqual((row["action_executed"], row["rule_entry"]), (RIGHT, "V1-back"))
        row, _ = v1_step(pillars=[], net_yaw=-1, last_turn=RIGHT)
        self.assertEqual((row["action_executed"], row["rule_entry"]), (LEFT, "V1-back"))

    def test_memory_carries_over_and_uses_the_true_pose(self):
        mem = []
        v1_step(rule_mem=mem, step_idx=0)                    # 在原点看见 0.6 m 外的柱面
        first = len(mem)
        row, _ = v1_step(pillars=[], x=0.5, rule_mem=mem, step_idx=3)   # 前进到 x=0.5，这一步看不见任何柱子（pillars 空）
        self.assertEqual(row["rule_mem_n"], first, "新一步没有新点，旧点全在（年龄 3 ≤ 8）")
        # 柱面在世界系 x≈0.6，机体在 x=0.5：点现在在前方 0.1 m 处，已进 d_look 通道 → 挡住
        self.assertEqual(row["rule_clear"][1], False)
        self.assertNotEqual(row["action_executed"], FORWARD, "记忆里的柱子挡着当前航向，不许前进")

    def test_memory_expires(self):
        mem = []
        v1_step(rule_mem=mem, step_idx=0)
        row, _ = v1_step(pillars=[], x=0.5, rule_mem=mem, step_idx=8)
        self.assertGreater(row["rule_mem_n"], 0, "年龄 8 = mem_steps，还在")
        row, _ = v1_step(pillars=[], x=0.5, rule_mem=mem, step_idx=9)
        self.assertEqual(row["rule_mem_n"], 0, "年龄 9 > mem_steps：过期")
        self.assertEqual((row["action_executed"], row["rule_entry"]), (FORWARD, "V1-fwd"))

    def test_81_deg_point_count(self):
        mem = []
        row, _ = v1_step(pillars=[{"x": 0.8, "y": 0.0, "z": 1.5, "size": [0.4, 8.0, 3.0]}], rule_mem=mem, rule_hfov_deg=81.0)
        self.assertEqual(len(row["rule_cols"]), 27)
        self.assertEqual(row["rule_mem_n"], 27, "一堵宽墙：27 列全有距离 → 27 个点")

    def test_needs_memory_list(self):
        w, h = 4, 6
        subs = {"rgb": tc.FakeSub(after=tc.image_msg(w, h, 10)), "depth": tc.FakeSub(frames=[(1, tc._fresh_depth(w, h))]),
                "pose": tc.ClockPoseSub(t0.pose_with_yaw(10.2, 0.0, 0.0, 0.0)), "contact": tc.FakeSub()}
        with tempfile.TemporaryDirectory() as d, patch.object(collect, "write_rgb_png", return_value=(True, None)):
            row, err = collect.collect_one(d, 0, 1.0, "rule-oracle", subs, tc.FakePub(), pillars=AHEAD, rule_version="v1")
        self.assertIsNone(row)
        self.assertIn("rule_mem", err)

    def test_v0_row_is_unchanged(self):
        row, _ = t0.rule_step()
        self.assertIsNone(row["rule_mem_n"])
        self.assertIsNone(row["rule_clear"])
        ar = collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})
        self.assertEqual(set(ar), tc.ROW_KEYS_DEFAULT | {"executed_by", "rule_cols", "rule_entry"}, "v0 落盘字段集与 v0 实验一致")
        self.assertEqual(row["rule_entry"], "R2-left")

    def test_v1_action_row_has_the_two_new_fields(self):
        row, _ = v1_step()
        ar = collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})
        self.assertEqual(set(ar), tc.ROW_KEYS_DEFAULT | {"executed_by", "rule_cols", "rule_entry", "rule_mem_n", "rule_clear"})
        self.assertEqual(ar["rule_clear"], [False, False, False])


def run_loop_v1(extra_argv, actions, with_scene=True):
    """真 collect.main() 主循环 + 假 collect_one：记下每步传进来的 rule_version、记忆列表（对象身份与进入时的长度）。"""
    seen = []

    def fake_collect_one(out_dir, step_idx, *args, **kw):
        mem = kw.get("rule_mem")
        seen.append({"rule_version": kw.get("rule_version"), "mem_id": None if mem is None else id(mem),
                     "mem_len_in": None if mem is None else len(mem), "net_yaw": kw.get("net_yaw")})
        extra = {}
        if args[1] == "rule-oracle":
            extra = {"executed_by": "rule_oracle", "rule_cols": [4.0] * 20,
                     "rule_entry": "V1-fwd" if kw.get("rule_version") == "v1" else "R3"}
            if kw.get("rule_version") == "v1":
                if mem is not None:
                    mem.append((1.0, 0.0, step_idx))          # 假装本步并入一个点：主循环必须把同一个列表一直传下去
                extra |= {"rule_mem_n": 1, "rule_clear": [True, True, False]}
        return tc.fake_row(step_idx, 1.0 + 0.25 * step_idx, action_executed=actions[step_idx], **extra), None

    err = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.ExitStack() as st, contextlib.redirect_stderr(err):
        out, meta = os.path.join(d, "ep"), os.path.join(d, "scene.json")
        Path(meta).write_text(json.dumps(tc.SCENE))
        argv = ["collect.py", "--out", out, "--split", sm.split_for_scene(tc.SCENE), "--steps", str(len(actions))]
        argv += ["--scene-meta", meta] if with_scene else []
        subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}
        st.enter_context(patch.object(sys, "argv", argv + list(extra_argv)))
        st.enter_context(patch.object(collect, "open_subs", return_value=subs))
        st.enter_context(patch.object(collect, "wait_ready", return_value=None))
        st.enter_context(patch.object(collect, "read_camera_info", return_value=None))
        st.enter_context(patch.object(collect.gz_io, "open_twist_publisher", return_value=tc.MainPub()))
        st.enter_context(patch.object(collect, "take_off", return_value=0))
        st.enter_context(patch.object(collect, "stop_executor"))
        st.enter_context(patch.object(collect, "close_subs"))
        st.enter_context(patch.object(collect, "collect_one", side_effect=fake_collect_one))
        with contextlib.suppress(SystemExit):
            collect.main()
        man_path = Path(out, "episode_manifest.json")
        man = json.loads(man_path.read_text()) if man_path.is_file() else None
        rows = ([json.loads(line) for line in Path(out, "actions.jsonl").read_text().splitlines() if line.strip()]
                if man is not None else [])
    return seen, man, err.getvalue(), rows


class CollectLoopV1Tests(unittest.TestCase):
    def test_loop_passes_one_memory_list_per_episode(self):
        seen, man, err, rows = run_loop_v1(RULE_V1_ARGS, [FORWARD, LEFT, LEFT, FORWARD])
        self.assertEqual({s["rule_version"] for s in seen}, {"v1"})
        self.assertEqual(len({s["mem_id"] for s in seen}), 1, "同一集的每一步传同一个记忆列表")
        self.assertEqual([s["mem_len_in"] for s in seen], [0, 1, 2, 3], "第 0 步进入时为空（每集清零），之后逐步累积")
        self.assertEqual([s["net_yaw"] for s in seen], [0, 0, 1, 2])
        self.assertEqual(rows[0]["rule_mem_n"], 1)
        self.assertEqual(rows[0]["rule_clear"], [True, True, False])
        self.assertEqual(rows[0]["rule_entry"], "V1-fwd")
        self.assertIn("mem=1", err)
        self.assertIn("clear=110", err)

    def test_manifest_records_version_and_params(self):
        _seen, man, _err, _rows = run_loop_v1(RULE_V1_ARGS + ["--rule-hfov-deg", "81"], [FORWARD, FORWARD])
        self.assertEqual((man["teacher"], man["planner"]), ("rule_oracle", "rule_v1"))
        self.assertEqual(man["rule_oracle"], {"version": "v1", "hfov_deg": 81.0, "params": rule_policy.RULE_V1})

    def test_v0_default_is_bit_identical(self):
        seen, man, _err, rows = run_loop_v1(["--policy", "rule-oracle"], [FORWARD, LEFT])
        self.assertEqual({s["rule_version"] for s in seen}, {"v0"})
        self.assertEqual((man["teacher"], man["planner"]), ("rule_oracle", "rule_v0"))
        self.assertEqual(man["rule_oracle"], {"hfov_deg": 60.0, "params": rule_policy.RULE_V0}, "v0 的 manifest 不加 version 键")
        self.assertTrue(all("rule_mem_n" not in r and "rule_clear" not in r for r in rows))
        seen, man, _err, _rows = run_loop_v1(["--policy", "rule-oracle", "--rule-version", "v0"], [FORWARD])
        self.assertEqual(man["rule_oracle"], {"hfov_deg": 60.0, "params": rule_policy.RULE_V0})

    def test_other_policies_leave_no_trace(self):
        _seen, man, _err, rows = run_loop_v1([], [FORWARD, LEFT])
        self.assertNotIn("rule_oracle", man)
        self.assertTrue(all(set(r) == tc.ROW_KEYS_DEFAULT for r in rows))

    def test_version_validation(self):
        _seen, man, err, _rows = run_loop_v1(["--rule-version", "v1"], [FORWARD])          # 默认策略是教师
        self.assertIsNone(man)
        self.assertIn("--rule-version 只能和 --policy rule-oracle 一起用", err)
        _seen, man, err, _rows = run_loop_v1(["--policy", "rule-oracle", "--rule-version", "v2"], [FORWARD])
        self.assertIsNone(man)
        self.assertIn("--rule-version", err)
        _seen, man, _err, _rows = run_loop_v1(["--rule-version", "v0"], [FORWARD])         # 等于默认值：不算用了它
        self.assertIsNotNone(man)


class EvalV1PassthroughTests(unittest.TestCase):
    run_eval = t09.Sim09V2Tests.run_eval
    tearDown = t09.Sim09V2Tests.tearDown

    def run_rule(self, *extra, **kw):
        return self.run_eval([t09.test_seeds(1)[0]], extra_argv=("--policy", "rule-oracle", *extra), **kw)

    def test_v1_command_and_report(self):
        rc, report, cmds, _w, _out = self.run_rule("--rule-version", "v1", "--rule-hfov-deg", "81")
        self.assertEqual(rc, 0)
        cmd = cmds[0]
        self.assertEqual(cmd[cmd.index("--rule-version") + 1], "v1")
        self.assertEqual(cmd[cmd.index("--rule-hfov-deg") + 1], "81.0")
        self.assertEqual(report["rule_version"], "v1")
        self.assertEqual(report["rule_params"], rule_policy.RULE_V1)
        self.assertEqual(report["rule_hfov_deg"], 81.0)
        self.assertEqual(report["metric_version"], "sim09.v2", "规则层改的是策略不是判据，metric_version 不动")

    def test_v0_command_line_and_report_unchanged(self):
        _rc, report, cmds, _w, _out = self.run_rule()
        self.assertNotIn("--rule-version", cmds[0], "v0 的命令行逐字不变")
        self.assertNotIn("rule_version", report)
        self.assertEqual(report["rule_params"], rule_policy.RULE_V0)
        _rc, report, cmds, _w, _out = self.run_rule("--rule-version", "v0")
        self.assertNotIn("--rule-version", cmds[0])
        self.assertNotIn("rule_version", report)

    def test_other_policies_unchanged(self):
        _rc, report, cmds, _w, _out = self.run_eval([t09.test_seeds(1)[0]])
        self.assertNotIn("--rule-version", cmds[0])
        self.assertNotIn("rule_version", report)

    def assert_rejected(self, run, text):
        with patch.object(argparse.ArgumentParser, "error", side_effect=SystemExit(2)) as error:
            with self.assertRaises(SystemExit):
                run()
        self.tearDown()
        self.assertIn(text, error.call_args[0][0])

    def test_rejected_for_other_policy(self):
        self.assert_rejected(lambda: self.run_eval([t09.test_seeds(1)[0]], extra_argv=("--rule-version", "v1")),
                             "只能和 --policy rule-oracle 一起用")

    def test_course_base_goes_through(self):
        # course_base 的柱子列表由 scene_course 生成，collect 只认 --scene-meta 里的 pillars；这里验命令行与报告不挑任务
        rc, report, cmds, _w, _out = self.run_rule("--rule-version", "v1", "--blocks", "3", task="course_base")
        self.assertEqual(rc, 0)
        self.assertEqual(cmds[0][cmds[0].index("--rule-version") + 1], "v1")
        self.assertEqual(report["blocks"], 3)
        self.assertEqual(report["rule_version"], "v1")


if __name__ == "__main__":
    unittest.main()
