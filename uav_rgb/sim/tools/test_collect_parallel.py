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
