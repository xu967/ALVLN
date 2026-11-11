#!/usr/bin/env python3
"""W1-3「per_episode 记 max_frame_gap_s / frame_gap_flag」的接线测试。不起 Gazebo。

summarize_episode 自己的取值与 hold 的来源在 test_eval_summary.py。这里走完整 ec.main()（假 scene_gen 出真 meta、
假 collect 把指定轨迹写成集目录；夹具是 test_eval_blocks.run_eval）一路到 eval.json，锁三件事：
  - run_episode 传给 summarize_episode 的是 --hold：同样的帧间隔，--hold 不同，frame_gap_flag 不同
    （gz 采集的 actions.jsonl 行没有 sim_dt，flag 能不能出来全靠这条接线）；
  - actions 行没有 rgb_ts（旧数据）的集，per_episode 不多键；
  - 新键只进 per_episode：eval.json 顶层键集合不变（ec.aggregate 没动），per_episode 的旧键一个不少。
依赖：test_eval_blocks.py（W1-4）的 run_eval / traj / ramp 与旧键集合；W1-3 分支基于 W1-4 分支。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_eval_frame_gap.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_eval_blocks import OLD_EPISODE_KEYS_BASE, OLD_TOP_KEYS_BASE, ramp, run_eval, traj  # noqa: E402

GAP_KEYS = {"max_frame_gap_s", "frame_gap_flag"}


def stamped(pair, gaps_s):
    """给 traj() 的每行 actions 补 rgb_ts：从 10 s 起，相邻两行相隔 gaps_s[i] 秒（整数 ns；gaps_s 比行数少 1）。"""
    oracle, actions = pair
    ts, rows = 10 * 10**9, []
    for i, a in enumerate(actions):
        rows.append({**a, "rgb_ts": ts})
        if i < len(gaps_s):
            ts += int(round(gaps_s[i] * 1e9))
    return oracle, rows


def trajectories():
    """base 柱阵三集：1 号中间有一次 2.4 s 的间隔，113 号全程 1.0 s，122 号的行没有 rgb_ts（旧数据）。"""
    n = len(ramp(6.3)) - 1
    spike = [1.0] * n
    spike[10] = 2.4
    return {1: stamped(traj(ramp(6.3)), spike),
            113: stamped(traj(ramp(6.3)), [1.0] * n),
            122: traj(ramp(6.3))}


def by_seed(report):
    return {e["seed"]: e for e in report["per_episode"]}


class FrameGapReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_eval("base", trajectories())        # 默认 --hold 1.0
        cls.eps = by_seed(cls.report)

    def test_gap_and_flag_reach_per_episode_with_the_default_hold(self):
        self.assertEqual(self.eps[1]["max_frame_gap_s"], 2.4)
        self.assertIs(self.eps[1]["frame_gap_flag"], True)
        self.assertEqual(self.eps[113]["max_frame_gap_s"], 1.0)
        self.assertNotIn("frame_gap_flag", self.eps[113], "间隔 1.0 s 不打旗，键不出现")

    def test_episode_without_rgb_ts_gets_no_new_keys(self):
        self.assertEqual(set(self.eps[122]), OLD_EPISODE_KEYS_BASE)

    def test_only_per_episode_gains_keys(self):
        self.assertEqual(set(self.report), OLD_TOP_KEYS_BASE | {"collisions_per_100m"},
                         "顶层键集合与 W1-4 之后一致：aggregate 没动，也没有新的汇总键")
        self.assertEqual(set(self.eps[1]), OLD_EPISODE_KEYS_BASE | GAP_KEYS)
        self.assertEqual(set(self.eps[113]), OLD_EPISODE_KEYS_BASE | {"max_frame_gap_s"})

    def test_old_values_are_the_same_as_without_the_timestamps(self):
        """同一批轨迹去掉 rgb_ts 再评一遍：除新键外，每集每个键的值逐项相等。"""
        bare = {s: (o, [{k: v for k, v in a.items() if k != "rgb_ts"} for a in acts])
                for s, (o, acts) in trajectories().items()}
        eps_bare = by_seed(run_eval("base", bare))
        for seed, e in self.eps.items():
            stripped = {k: v for k, v in e.items() if k not in GAP_KEYS | {"wall_s"}}   # wall_s 是真实耗时，两次评测必然不同
            bare_ep = {k: v for k, v in eps_bare[seed].items() if k != "wall_s"}
            self.assertEqual(stripped, bare_ep, f"seed {seed}")


class HoldComesFromTheCommandLine(unittest.TestCase):
    """gz 采集的 actions 行没有 sim_dt：run_episode 必须把 --hold 传下去，否则 frame_gap_flag 永远出不来。"""

    def test_flag_follows_hold(self):
        n = len(ramp(6.3)) - 1
        spike = [1.0] * n
        spike[10] = 2.4
        pair = {1: stamped(traj(ramp(6.3)), spike), 113: stamped(traj(ramp(6.3)), [1.0] * n)}
        # --hold 2.0：阈值 2.4 s，2.4 s 的间隔不算大于
        eps = by_seed(run_eval("base", pair, extra_argv=["--hold", "2.0"]))
        self.assertEqual(eps[1]["max_frame_gap_s"], 2.4)
        self.assertNotIn("frame_gap_flag", eps[1])
        # --hold 0.5：阈值 0.6 s，全程 1.0 s 的那集也算断帧
        eps = by_seed(run_eval("base", pair, extra_argv=["--hold", "0.5"]))
        self.assertIs(eps[113]["frame_gap_flag"], True)
        self.assertIs(eps[1]["frame_gap_flag"], True)


if __name__ == "__main__":
    unittest.main()
