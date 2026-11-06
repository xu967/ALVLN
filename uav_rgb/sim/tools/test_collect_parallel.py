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
