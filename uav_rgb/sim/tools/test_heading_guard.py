#!/usr/bin/env python3
"""航向护栏 --heading-guard 的接线测试（计划 C1）：collect.main 主循环累计净转向、manifest 键、参数校验；
eval_closed_loop 的透传、校验与 eval.json 记录。默认 0 时命令行、manifest、eval.json 都不留痕迹。

test_collect.py 已覆盖 heading_guard_action 与 collect_one 的改写，这里补它们上面的两层接线。
不起 Gazebo：collect_one、gz、StudentPolicy 全部替身，替身沿用 test_collect.py / test_eval_sim09v2.py 的。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_heading_guard.py
"""
import argparse
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
import collect
import split_manifest as sm
import test_collect as tc          # 只借替身与 fake_row
import test_eval_sim09v2 as t09    # 以模块名引用：import 类名会让 unittest 把对方的用例再跑一遍

LEFT, RIGHT, FORWARD = collect.YAW_LEFT, collect.YAW_RIGHT, collect.FORWARD_SLOW


def run_loop(extra_argv, steps, student_action):
    """真 collect.main() 主循环 + 假 collect_one：假的用真 heading_guard_action 改写动作，并记下主循环传进来的 net_yaw。

    返回 (每步收到的 net_yaw, 每步执行的动作, manifest 或 None, stderr 文本)；extra_argv 里的 CKPT 换成临时权重路径。
    """
    nets = []

    def fake_collect_one(out_dir, step_idx, *args, **kw):
        nets.append(kw.get("net_yaw"))
        # 护栏关着时不碰 heading_guard_action：默认 0 的用例要在没有这个函数的旧代码上也能跑，才算「前后不变」的锁
        guard = collect.heading_guard_action(kw["net_yaw"], kw["heading_guard"]) if kw.get("heading_guard") else None
        return tc.fake_row(step_idx, 1.0 + 0.25 * step_idx,
                           action_executed=student_action if guard is None else guard,
                           executed_by="student" if guard is None else "heading_guard"), None

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
        acts = ([json.loads(line)["action_executed"] for line in Path(out, "actions.jsonl").read_text().splitlines()
                 if line.strip()] if man is not None else [])
    return nets, acts, man, err.getvalue()


STUDENT = ["--policy", "student", "--student-ckpt", "CKPT"]
GUARD_ERROR = "--heading-guard 必须是非负整数"


class CollectLoopWiringTests(unittest.TestCase):
    def test_left_drift_is_turned_back_and_net_yaw_counts_executed_actions(self):
        nets, acts, man, _ = run_loop(STUDENT + ["--heading-guard", "2"], 6, LEFT)
        self.assertEqual(acts, [LEFT, LEFT, RIGHT, LEFT, RIGHT, LEFT])
        self.assertEqual(nets, [0, 1, 2, 1, 2, 1], "累计的是已执行动作：被护栏改写成右转的步也要减 1")
        self.assertEqual(man["heading_guard"], 2)

    def test_right_drift_is_turned_back(self):
        nets, acts, _man, _ = run_loop(STUDENT + ["--heading-guard", "2"], 6, RIGHT)
        self.assertEqual(acts, [RIGHT, RIGHT, LEFT, RIGHT, LEFT, RIGHT])
        self.assertEqual(nets, [0, -1, -2, -1, -2, -1])

    def test_forward_steps_do_not_move_net_yaw(self):
        nets, acts, _man, _ = run_loop(STUDENT + ["--heading-guard", "2"], 4, FORWARD)
        self.assertEqual((nets, acts), ([0, 0, 0, 0], [FORWARD] * 4))

    def test_default_off_does_not_rewrite_and_leaves_no_manifest_key(self):
        _nets, acts, man, _ = run_loop(STUDENT, 4, LEFT)
        self.assertEqual(acts, [LEFT] * 4, "默认 0：学生一直左转也不干预")
        self.assertNotIn("heading_guard", man, "默认 manifest 逐位不变：不写 heading_guard 键")

    def test_rejected_for_non_student_policy_and_negative_value(self):
        _n, _a, man, err = run_loop(["--heading-guard", "2"], 2, LEFT)          # 默认策略是教师
        self.assertIsNone(man)
        self.assertIn(GUARD_ERROR, err)
        _n, _a, man, err = run_loop(STUDENT + ["--heading-guard", "-1"], 2, LEFT)
        self.assertIsNone(man)
        self.assertIn(GUARD_ERROR, err)


class EvalPassthroughTests(unittest.TestCase):
    run_eval = t09.Sim09V2Tests.run_eval      # 复用那边整套 gz / scene_gen / collect 替身（只借方法，不继承）
    tearDown = t09.Sim09V2Tests.tearDown

    def run_student(self, *extra):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        ckpt, binary = Path(tmp.name, "ckpt.bin"), Path(tmp.name, "uavrgb_infer")
        ckpt.write_bytes(b"ckpt")
        binary.write_bytes(b"binary")
        with patch("student.StudentPolicy", return_value=Mock(text_dim=0)):
            return self.run_eval([t09.test_seeds(1)[0]], extra_argv=(
                "--policy", "student", "--ckpt", str(ckpt), "--infer-bin", str(binary), *extra))

    def test_guard_is_passed_to_collect_and_recorded_in_eval_json(self):
        rc, report, cmds, _w, _out = self.run_student("--heading-guard", "2")
        self.assertEqual(rc, 0)
        self.assertEqual(cmds[0][cmds[0].index("--heading-guard") + 1], "2")
        self.assertEqual(report["heading_guard"], 2)
        self.assertEqual(report["metric_version"], "sim09.v2", "护栏改的是策略不是判据，metric_version 不动")

    def test_default_off_student_command_line_and_report_unchanged(self):
        _rc, report, cmds, _w, _out = self.run_student()
        self.assertNotIn("--heading-guard", cmds[0], "0 不传：旧命令行逐字不变")
        self.assertNotIn("heading_guard", report)

    def test_default_off_forward_command_line_and_report_unchanged(self):
        _rc, report, cmds, _w, _out = self.run_eval([t09.test_seeds(1)[0]])
        self.assertNotIn("--heading-guard", cmds[0])
        self.assertNotIn("heading_guard", report)

    def assert_rejected(self, run):
        # 直接截 parser.error 的消息：旧代码里 --heading-guard 是未知参数，同样退出 2，只有消息能区分
        with patch.object(argparse.ArgumentParser, "error", side_effect=SystemExit(2)) as error:
            with self.assertRaises(SystemExit):
                run()
        self.assertIn(GUARD_ERROR, error.call_args[0][0])

    def test_rejected_for_non_student_policy(self):
        self.assert_rejected(lambda: self.run_eval([t09.test_seeds(1)[0]], extra_argv=("--heading-guard", "2")))

    def test_rejected_for_negative_value(self):
        self.assert_rejected(lambda: self.run_student("--heading-guard", "-1"))


if __name__ == "__main__":
    unittest.main()
