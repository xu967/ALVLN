#!/usr/bin/env python3
"""--policy rule-oracle 的接线测试（计划 D3）：collect_one 的分支与落盘字段、collect.main 主循环维护 net_yaw / last_turn、
manifest 键、参数校验；eval_closed_loop 的透传、校验与 eval.json 记录。其它策略的命令行、manifest、actions 行不留痕迹。

规则本身的行为在 test_rule_policy.py。不起 Gazebo：collect_one、gz、StudentPolicy 全部替身，替身沿用
test_collect.py / test_eval_sim09v2.py 的（风格同 test_heading_guard.py）。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_rule_oracle_wiring.py
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
import test_collect as tc          # 只借替身与 fake_row
import test_eval_sim09v2 as t09    # 以模块名引用：import 类名会让 unittest 把对方的用例再跑一遍

LEFT, RIGHT, FORWARD, SAFE = collect.YAW_LEFT, collect.YAW_RIGHT, collect.FORWARD_SLOW, collect.SAFE_NEUTRAL
RULE = ["--policy", "rule-oracle"]
HFOV_ERROR = "--rule-hfov-deg"
AHEAD = [{"x": 0.8, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]     # 柱面离机体 0.6 m，正前方：blocked


def pose_with_yaw(sec, x, y, yaw_deg):
    whole = int(sec)
    half = math.radians(yaw_deg) / 2
    return {"pose": [{"position": {"x": x, "y": y, "z": 2.1},
                      "orientation": {"x": 0.0, "y": 0.0, "z": math.sin(half), "w": math.cos(half)},
                      "header": {"stamp": {"sec": whole, "nsec": int(round((sec - whole) * 1e9))}}}]}


def rule_step(pillars=AHEAD, x=0.0, y=0.0, yaw_deg=0.0, **kw):
    """真 collect_one + rule-oracle 分支：位姿与柱子真值 → 列距离 → 规则。返回 (row, err)。"""
    w, h = 4, 6
    pose = tc.ClockPoseSub(pose_with_yaw(10.2, x, y, yaw_deg))
    subs = {"rgb": tc.FakeSub(after=tc.image_msg(w, h, 10)), "depth": tc.FakeSub(frames=[(1, tc._fresh_depth(w, h))]),
            "pose": pose, "contact": tc.FakeSub()}

    def advance(_secs):
        pose.current = pose_with_yaw(11.2, x + 0.25, y, yaw_deg)

    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        return collect.collect_one(d, 0, 1.0, "rule-oracle", subs, tc.FakePub(), pillars=pillars, **kw)


class CollectOneTests(unittest.TestCase):
    def test_blocked_ahead_row_has_columns_entry_and_executor(self):
        row, err = rule_step()
        self.assertIsNone(err)
        self.assertEqual((row["action_executed"], row["action_expert"]), (LEFT, LEFT), "n=0 无 last_turn 平局 → 固定左")
        self.assertEqual(row["executed_by"], "rule_oracle")
        self.assertEqual(row["rule_entry"], "R2-left")
        cols = row["rule_cols"]
        self.assertEqual(len(cols), 20)
        self.assertTrue(all(round(v, 2) == v for v in cols), "列距离保留 2 位小数")
        self.assertAlmostEqual(cols[9], 0.6 - 0.045, delta=0.006)     # 柱面 0.6 m 减相机前移 0.045 m，落盘取 2 位小数

    def test_uses_pose_before_and_yaw_before_not_the_world_frame(self):
        # 机头朝 +y（yaw 90°），柱子在 +y 方向 0.8 m：相对机体仍是正前方 blocked；同一根柱在 yaw=0 时在机体左侧、不挡
        row, _ = rule_step([{"x": 0.0, "y": 0.8, "z": 1.5, "size": [0.4, 0.4, 3.0]}], yaw_deg=90.0)
        self.assertEqual(row["rule_entry"], "R2-left")
        row, _ = rule_step([{"x": 0.0, "y": 0.8, "z": 1.5, "size": [0.4, 0.4, 3.0]}], yaw_deg=0.0)
        self.assertEqual((row["action_executed"], row["rule_entry"]), (FORWARD, "R3"))

    def test_net_yaw_and_last_turn_are_inputs(self):
        row, _ = rule_step(last_turn=RIGHT)
        self.assertEqual((row["action_executed"], row["rule_entry"]), (RIGHT, "R2-last"))
        row, _ = rule_step(pillars=[], net_yaw=2)
        self.assertEqual((row["action_executed"], row["rule_entry"]), (RIGHT, "R1"))
        row, _ = rule_step(pillars=[], net_yaw=-2)
        self.assertEqual(row["action_executed"], LEFT)

    def test_hfov_changes_column_count(self):
        row, _ = rule_step(rule_hfov_deg=81.0)
        self.assertEqual(len(row["rule_cols"]), 27)

    def test_action_row_writes_rule_fields_only_for_this_policy(self):
        row, _ = rule_step()
        ar = collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})
        self.assertEqual(set(ar), tc.ROW_KEYS_LEGACY | {"executed_by", "rule_cols", "rule_entry"})
        self.assertEqual(ar["rule_cols"], row["rule_cols"])
        plain = collect.action_row(tc.fake_row(0, 1.0) | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})
        self.assertEqual(set(plain), tc.ROW_KEYS_LEGACY, "其它策略的行字段集不变")

    def test_missing_pose_or_pillars_fail_the_step(self):
        row, err = rule_step(pillars=None)
        self.assertIsNone(row)
        self.assertIn("rule-oracle", err)
        w, h = 4, 6
        subs = {"rgb": tc.FakeSub(after=tc.image_msg(w, h, 10)), "depth": tc.FakeSub(frames=[(1, tc._fresh_depth(w, h))]),
                "pose": tc.ClockPoseSub({"pose": [{"orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                                                   "header": {"stamp": {"sec": 10, "nsec": 200000000}}}]}),
                "contact": tc.FakeSub()}
        with tempfile.TemporaryDirectory() as d, patch.object(collect, "write_rgb_png", return_value=(True, None)):
            row, err = collect.collect_one(d, 0, 1.0, "rule-oracle", subs, tc.FakePub(), pillars=AHEAD)
        self.assertIsNone(row)
        self.assertIn("rule-oracle", err)


def run_loop(extra_argv, actions, with_scene=True):
    """真 collect.main() 主循环 + 假 collect_one：按 actions 依次执行，并记下主循环每步传进来的 net_yaw / last_turn / rule_hfov_deg。

    返回 (记录, manifest 或 None, stderr 文本)。
    """
    seen = []

    def fake_collect_one(out_dir, step_idx, *args, **kw):
        seen.append({k: kw.get(k) for k in ("net_yaw", "last_turn", "rule_hfov_deg")})
        extra = ({"executed_by": "rule_oracle", "rule_cols": [4.0] * 20, "rule_entry": "R3"}
                 if args[1] == "rule-oracle" else {})
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


class CollectLoopWiringTests(unittest.TestCase):
    def test_loop_maintains_net_yaw_and_last_turn_from_executed_actions(self):
        seen, man, _err, rows = run_loop(RULE, [FORWARD, LEFT, LEFT, FORWARD, SAFE, RIGHT, FORWARD])
        self.assertEqual([s["net_yaw"] for s in seen], [0, 0, 1, 2, 2, 2, 1])
        self.assertEqual([s["last_turn"] for s in seen], [None, None, LEFT, LEFT, LEFT, LEFT, RIGHT],
                         "last_turn = 最近一次执行的转向方向；前进与 SAFE 不改它")
        self.assertTrue(all(s["rule_hfov_deg"] == 60.0 for s in seen))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["rule_entry"], "R3")

    def test_manifest_records_rule_oracle(self):
        _seen, man, _err, _rows = run_loop(RULE + ["--rule-hfov-deg", "81"], [FORWARD, FORWARD])
        self.assertEqual((man["teacher"], man["planner"], man["teacher_version"]), ("rule_oracle", "rule_v0", None))
        self.assertEqual(man["rule_oracle"], {"hfov_deg": 81.0, "params": rule_policy.RULE_V0})
        self.assertEqual(man["experimental_policy"], False)

    def test_other_policies_leave_no_trace(self):
        seen, man, _err, rows = run_loop([], [FORWARD, LEFT])
        self.assertNotIn("rule_oracle", man)
        self.assertEqual((man["teacher"], man["planner"]), ("plan_action", "astar"))
        self.assertTrue(all(set(r) == tc.ROW_KEYS_LEGACY for r in rows))
        seen, man, _err, _rows = run_loop(["--policy", "forward"], [FORWARD, FORWARD])
        self.assertNotIn("rule_oracle", man)

    def test_needs_scene_meta(self):
        _seen, man, err, _rows = run_loop(RULE, [FORWARD], with_scene=False)
        self.assertIsNone(man)
        self.assertIn("--policy rule-oracle 需要 --scene-meta", err)

    def test_hfov_validation(self):
        for bad in ("0", "-60", "nan", "inf", "181", "5"):
            _seen, man, err, _rows = run_loop(RULE + ["--rule-hfov-deg", bad], [FORWARD])
            self.assertIsNone(man, bad)
            self.assertIn(HFOV_ERROR, err)
        _seen, man, err, _rows = run_loop(["--rule-hfov-deg", "81"], [FORWARD])       # 默认策略是教师
        self.assertIsNone(man)
        self.assertIn("只能和 --policy rule-oracle 一起用", err)
        _seen, man, _err, _rows = run_loop(["--rule-hfov-deg", "60"], [FORWARD])     # 等于默认值：不算用了它
        self.assertIsNotNone(man)


class EvalPassthroughTests(unittest.TestCase):
    run_eval = t09.Sim09V2Tests.run_eval      # 复用那边整套 gz / scene_gen / collect 替身（只借方法，不继承）
    tearDown = t09.Sim09V2Tests.tearDown

    def run_rule(self, *extra):
        return self.run_eval([t09.test_seeds(1)[0]], extra_argv=("--policy", "rule-oracle", *extra))

    def test_rule_oracle_command_and_report(self):
        rc, report, cmds, _w, _out = self.run_rule("--rule-hfov-deg", "81")
        self.assertEqual(rc, 0)
        cmd = cmds[0]
        self.assertEqual(cmd[cmd.index("--policy") + 1], "rule-oracle")
        self.assertEqual(cmd[cmd.index("--rule-hfov-deg") + 1], "81.0")
        for flag in ("--student-ckpt", "--infer-bin", "--heading-guard"):
            self.assertNotIn(flag, cmd)
        self.assertEqual(report["policy"], "rule-oracle")
        self.assertEqual(report["rule_hfov_deg"], 81.0)
        self.assertEqual(report["rule_params"], rule_policy.RULE_V0)
        self.assertEqual(report["metric_version"], "sim09.v2", "规则层改的是策略不是判据，metric_version 不动")
        self.assertEqual(report["per_episode"][0]["policy"], "rule-oracle")

    def test_default_hfov_is_passed_explicitly_for_rule_oracle(self):
        _rc, report, cmds, _w, _out = self.run_rule()
        self.assertEqual(cmds[0][cmds[0].index("--rule-hfov-deg") + 1], "60.0")
        self.assertEqual(report["rule_hfov_deg"], 60.0)

    def test_other_policies_command_line_and_report_unchanged(self):
        _rc, report, cmds, _w, _out = self.run_eval([t09.test_seeds(1)[0]])      # forward
        self.assertNotIn("--rule-hfov-deg", cmds[0])
        self.assertNotIn("rule_hfov_deg", report)
        self.assertNotIn("rule_params", report)

    def assert_rejected(self, run, text):
        # 直接截 parser.error 的消息：旧代码里 --rule-hfov-deg 是未知参数，同样退出 2，只有消息能区分
        with patch.object(argparse.ArgumentParser, "error", side_effect=SystemExit(2)) as error:
            with self.assertRaises(SystemExit):
                run()
        self.tearDown()                      # 循环里多次调用时，上一次的临时目录不留给 GC
        self.assertIn(text, error.call_args[0][0])

    def test_rejected_for_other_policy_and_bad_values(self):
        self.assert_rejected(lambda: self.run_eval([t09.test_seeds(1)[0]], extra_argv=("--rule-hfov-deg", "81")),
                             "只能和 --policy rule-oracle 一起用")
        for bad in ("0", "-5", "nan", "181", "5"):
            self.assert_rejected(lambda: self.run_rule("--rule-hfov-deg", bad), HFOV_ERROR)

    def test_rejected_with_ckpt_less_flags_it_does_not_use(self):
        self.assert_rejected(lambda: self.run_rule("--heading-guard", "2"), "--heading-guard")


if __name__ == "__main__":
    unittest.main()
