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


if __name__ == "__main__":
    unittest.main()
