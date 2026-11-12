#!/usr/bin/env python3
"""不启动 Gazebo，验证并行采集的跨运行分区隔离和子进程分区传递。"""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import collect_parallel as parallel


def task():
    return parallel.Task(0, 0, 42, 42, "train", "world.sdf", "scene.json")


class PartitionTests(unittest.TestCase):
    def test_independent_runs_and_episodes_do_not_share_partition(self):
        tasks = [task() for _ in range(8)]
        self.assertEqual(len({t.partition for t in tasks}), len(tasks))
        for item in tasks:
            self.assertTrue(item.partition.startswith("uavrgb_"))

    def test_episode_passes_one_partition_to_sim_readiness_and_collector(self):
        item = task()
        proc, pump = Mock(), Mock()
        proc.wait.return_value = 1
        args = SimpleNamespace(steps=1, hold=0.5, policy="teacher",
                               sim_timeout=1, episode_timeout=1)
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(parallel, "_start_sim", return_value=proc) as start, \
                 patch.object(parallel, "wait_ready") as ready, \
                 patch.object(parallel, "_popen_collect", return_value=(proc, pump)) as collect, \
                 patch.object(parallel, "stop_group"), \
                 patch.object(parallel.time, "sleep"):
                parallel._run_episode_once(item, args, directory, directory)
        self.assertEqual(start.call_args.args[1], item.partition)
        self.assertEqual(ready.call_args.args[1], item.partition)
        self.assertEqual(collect.call_args.args[2]["GZ_PARTITION"], item.partition)
        self.assertFalse(item.ok)  # 缺 manifest 的失败不能被隔离测试伪装成成功


def args_ns(**over):
    base = dict(steps=40, hold=1.0, policy="teacher", teacher_planner="astar", rl_checkpoint="",
                window_from="observation", dynamics="cf_sitl", save_depth=False, vehicle="e1_mini",
                student_ckpt=None, infer_bin=None, beta=0.0, dagger_round=0,
                stop_at_goal=False, tiebreak="legacy")
    base.update(over)
    return SimpleNamespace(**base)


class CommandTests(unittest.TestCase):
    """2026-11-07 DEV-DATA 开关的透传：默认命令行逐字不变，开了才追加对应 collect.py 参数。"""

    def test_default_command_has_no_switch_flags(self):
        cmd = parallel.collect_command(task(), args_ns())
        self.assertEqual(cmd[2:], ["--steps", "40", "--hold", "1.0", "--out", task().staging,
                                   "--scene-id", "42", "--seed", "42", "--episode-id", "42",
                                   "--split", "train", "--scene-meta", "scene.json",
                                   "--window-from", "observation", "--dynamics", "cf_sitl",
                                   "--vehicle", "e1_mini"])
        for flag in ("--stop-at-goal", "--tiebreak", "--beta", "--dagger-round", "--student-ckpt", "--policy"):
            self.assertNotIn(flag, cmd)

    def test_switches_pass_through(self):
        cmd = parallel.collect_command(task(), args_ns(stop_at_goal=True, tiebreak="visible"))
        self.assertIn("--stop-at-goal", cmd)
        self.assertEqual(cmd[cmd.index("--tiebreak") + 1], "visible")
        cmd = parallel.collect_command(task(), args_ns(policy="student", student_ckpt="/c.bin",
                                                       infer_bin="/b/uavrgb_infer", beta=0.5, dagger_round=2))
        for flag, val in (("--policy", "student"), ("--student-ckpt", "/c.bin"),
                          ("--infer-bin", "/b/uavrgb_infer"), ("--beta", "0.5"), ("--dagger-round", "2")):
            self.assertEqual(cmd[cmd.index(flag) + 1], val)
        self.assertNotIn("--stop-at-goal", cmd)
        self.assertNotIn("--tiebreak", cmd)

    def test_costmap_pass_through(self):
        # --teacher-planner costmap：默认 tol/前瞻不传（collect.py 默认值相同），非默认才追加
        cmd = parallel.collect_command(task(), args_ns(teacher_planner="costmap"))
        self.assertEqual(cmd[cmd.index("--teacher-planner") + 1], "costmap")
        self.assertNotIn("--costmap-tol", cmd)
        self.assertNotIn("--costmap-lookahead-steps", cmd)
        cmd = parallel.collect_command(task(), args_ns(teacher_planner="costmap", costmap_tol=0.0,
                                                       costmap_lookahead_steps=2))
        self.assertEqual(cmd[cmd.index("--costmap-tol") + 1], "0.0")
        self.assertEqual(cmd[cmd.index("--costmap-lookahead-steps") + 1], "2")
        cmd = parallel.collect_command(task(), args_ns(costmap_tol=0.0))   # astar 下不透传
        self.assertNotIn("--costmap-tol", cmd)

    def test_action_latency_and_rtf_pass_through_and_default_invariance(self):
        import contextlib
        import io
        import json
        import os
        import sys
        from pathlib import Path

        # 1) 默认 collect_command 与 scene_gen_extra 逐字不变
        default_cmd = parallel.collect_command(task(), args_ns())
        self.assertNotIn("--action-latency", default_cmd)
        self.assertEqual(parallel.scene_gen_extra(args_ns()), [])
        # 2) 给了才追加
        cmd_lat = parallel.collect_command(task(), args_ns(action_latency=0.20))
        self.assertEqual(cmd_lat[:-2], default_cmd)
        self.assertEqual(cmd_lat[-2:], ["--action-latency", "0.2"])
        self.assertEqual(parallel.scene_gen_extra(args_ns(rtf=0.5)), ["--rtf", "0.5"])

        # 3) 完整 main() CLI 透传、manifest 记录与默认逐字不变
        def run_main(extra_argv, expect_exit=False):
            seen = {"scene_gen": [], "collect": []}
            pillars = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]

            def fake_run(cmd, **_kw):
                if parallel.SCENE_GEN in cmd:
                    seen["scene_gen"].append(list(cmd))
                    w = Path(cmd[cmd.index("--out") + 1])
                    w.write_text("<sdf/>")
                    w.with_suffix(".json").write_text(json.dumps({"pillars": pillars, "pillar_count": 1}))
                return Mock(returncode=0, stdout="", stderr="")

            def fake_popen_collect(item, args_obj, _env, _log):
                seen["collect"].append(parallel.collect_command(item, args_obj))
                os.makedirs(item.staging, exist_ok=True)
                Path(item.staging, "episode_manifest.json").write_text(json.dumps({
                    "num_steps": args_obj.steps,
                    "validity": {"episode_executed": True, "camera_verified": True, "physics_verified": True},
                }))
                proc, pump = Mock(), Mock()
                proc.wait.return_value = 0
                return proc, pump

            err = io.StringIO()
            with tempfile.TemporaryDirectory() as d:
                out = os.path.join(d, "out")
                argv = ["collect_parallel.py", "--instances", "1", "--episodes-per-instance", "1",
                        "--seeds", "2", "--steps", "5", "--out", out] + extra_argv
                with patch.object(sys, "argv", argv), \
                     patch.object(parallel.subprocess, "run", side_effect=fake_run), \
                     patch.object(parallel, "_start_sim", return_value=Mock(pid=99)), \
                     patch.object(parallel, "wait_ready"), \
                     patch.object(parallel, "_popen_collect", side_effect=fake_popen_collect), \
                     patch.object(parallel, "stop_group"), \
                     patch.object(parallel, "cleanup_all"), \
                     patch.object(parallel, "_sweep_world_ruby", return_value=[]), \
                     patch.object(parallel, "_ruby_pids", return_value=[]), \
                     patch.object(parallel, "XDG_RUNTIME_DIR", os.path.join(d, "xdg")), \
                     patch.object(parallel.time, "sleep"), \
                     patch.object(parallel.signal, "signal"), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                    if expect_exit:
                        with self.assertRaises(SystemExit) as cm:
                            parallel.main()
                        return cm.exception.code, err.getvalue(), seen, None
                    rc = parallel.main()
                man = json.loads(Path(out, "collection_manifest.json").read_text())
                return rc, err.getvalue(), seen, man

        rc0, _, seen0, man0 = run_main([])
        self.assertEqual(rc0, 0)
        self.assertNotIn("--rtf", seen0["scene_gen"][0])
        self.assertNotIn("--action-latency", seen0["collect"][0])
        self.assertNotIn("action_latency_s", man0)
        self.assertNotIn("rtf", man0)

        rc1, _, seen1, man1 = run_main(["--action-latency", "0.20", "--rtf", "0.5"])
        self.assertEqual(rc1, 0)
        self.assertEqual(seen1["scene_gen"][0][-2:], ["--rtf", "0.5"])
        self.assertEqual(seen1["collect"][0][-2:], ["--action-latency", "0.2"])
        self.assertEqual(man1["action_latency_s"], 0.2)
        self.assertEqual(man1["rtf"], 0.5)
        # 去掉末尾两个新参数后，两条子进程命令行与默认运行逐项一致（除临时目录路径外）
        self.assertEqual([x for x in seen1["collect"][0][:-2] if not x.startswith("/")],
                         [x for x in seen0["collect"][0] if not x.startswith("/")])

        for bad_argv, msg in (
            (["--action-latency", "0"], "--action-latency 必须"),
            (["--action-latency", "-0.1"], "--action-latency 必须"),
            (["--action-latency", "nan"], "--action-latency 必须"),
            (["--action-latency", "1.0"], "--action-latency 必须小于 --hold"),
            (["--action-latency", "0.2", "--window-from", "dispatch"], "--window-from observation"),
            (["--rtf", "0"], "--rtf 必须"),
            (["--rtf", "-0.5"], "--rtf 必须"),
            (["--rtf", "inf"], "--rtf 必须"),
        ):
            code, stderr, _, _ = run_main(bad_argv, expect_exit=True)
            self.assertEqual(code, 2, bad_argv)
            self.assertIn(msg, stderr, bad_argv)


class StopTests(unittest.TestCase):
    """AUDIT Q1/Q2：停止位置位后 worker 不重试、不起下一集，否则 Ctrl-C/SIGTERM 之后还会起新 gz。"""

    def tearDown(self):
        parallel._STOP.clear()
        parallel._PROC_GROUPS.clear()

    def test_stop_skips_retry_and_remaining_episodes(self):
        calls = []

        def once(item, *_):
            calls.append(item.scene_id)
            item.ok = False
            parallel._STOP.set()   # 第一集跑到一半收到 Ctrl-C / SIGTERM，被 cleanup_all 杀成 rc=-15

        with patch.object(parallel, "_run_episode_once", side_effect=once):
            results = parallel.run_instance(SimpleNamespace(steps=1), [task(), task()], "", "")
        self.assertEqual(calls, [42])          # 不进 attempt 2，第二集不开始
        self.assertEqual([r.status for r in results], ["failed"])

    def test_register_after_stop_kills_late_group(self):
        parallel._STOP.set()
        proc = Mock(pid=4242)
        with patch.object(parallel, "stop_group") as stop:
            parallel._register(proc)   # cleanup_all 扫完之后才起来的进程组，登记时就地收掉
        self.assertIs(stop.call_args.args[0], proc)


if __name__ == "__main__":
    unittest.main()
