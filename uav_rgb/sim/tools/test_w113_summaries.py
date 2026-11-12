#!/usr/bin/env python3
"""W1-13 实验入口目录里两个汇总脚本的离线测试（不起 Gazebo、不读真实实验输出）。

  summarize_ack.py  冒烟判定 --smoke-check 与 plugin_ack_sim_t 分布汇总（任务卡第一部分第 3 条「先量后定」的量法）
  summarize_cal.py  新协议标定门（后续提交追加）

运行（仓库根）：.venv-tools/bin/python sim/tools/test_w113_summaries.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(os.path.dirname(os.path.dirname(HERE)), "experiments", "2026-11-12-w1-13-newprotocol")
sys.path.insert(0, EXP)
import summarize_ack as sa  # noqa: E402

NS = 10**9
L = 0.20


def row(step, obs_s=10.005, lag=0.2, ack=None, safe_gap=0.1, obs_gap=0.9, with_ack_key=True, **extra):
    """一行 actions.jsonl：观测位姿戳 obs_s，派发 = 观测 + lag，确认 ack（仿真秒，None = 没确认到）。"""
    r = {"step": step, "pose_ts": int(round(obs_s * NS)), "dispatch_lag_s": lag,
         "safe_gap_s": None if step == 0 else safe_gap, "obs_gap_s": None if step == 0 else obs_gap}
    if with_ack_key:
        r["plugin_ack_sim_t"] = ack
    r.update(extra)
    return r


def write_arm(root, name, episodes, eval_json=None):
    """episodes: {seed: [行...]}。"""
    arm = os.path.join(root, name)
    os.makedirs(arm, exist_ok=True)
    for seed, rows in episodes.items():
        d = os.path.join(arm, "episodes", f"seed_{seed}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
    with open(os.path.join(arm, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(eval_json if eval_json is not None else {"episodes_done": len(episodes), "failed_count": 0, "partial": False}, f)
    return arm


def good_episode(n=6, delay_s=0.005):
    """每步确认 = 观测 + L + delay_s；观测戳每步 +0.9 s。"""
    rows = []
    for i in range(n):
        obs = 10.005 + 0.9 * i
        rows.append(row(i, obs_s=obs, ack=obs + L + delay_s))
    return rows


class Percentile(unittest.TestCase):
    def test_linear_interpolation_like_numpy(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        self.assertEqual(sa.percentile(xs, 0), 0.0)
        self.assertEqual(sa.percentile(xs, 50), 2.0)
        self.assertEqual(sa.percentile(xs, 100), 4.0)
        self.assertAlmostEqual(sa.percentile(xs, 90), 3.6)
        self.assertAlmostEqual(sa.percentile(list(range(101)), 99), 99.0)
        self.assertIsNone(sa.percentile([], 50))

    def test_describe(self):
        d = sa.describe([0.001, 0.002, 0.003])
        self.assertEqual((d["n"], d["min"], d["median"], d["max"]), (3, 0.001, 0.002, 0.003))
        self.assertEqual(sa.describe([]), {"n": 0})


class StepDeltas(unittest.TestCase):
    def test_three_differences(self):
        d_ack, d_py, d_tr = sa.step_deltas(row(3, obs_s=10.005, lag=0.203, ack=10.215), L)
        self.assertAlmostEqual(d_ack, 0.010, places=9)     # 10.215 − (10.005 + 0.2)
        self.assertAlmostEqual(d_py, 0.003, places=9)      # 派发位姿戳 10.208 − 10.205
        self.assertAlmostEqual(d_tr, 0.007, places=9)      # 10.215 − 10.208

    def test_missing_pieces_give_none(self):
        self.assertEqual(sa.step_deltas(row(3, ack=None), L)[0::2], (None, None))
        self.assertIsNotNone(sa.step_deltas(row(3, ack=None), L)[1])      # Python 侧的差不依赖确认
        self.assertEqual(sa.step_deltas({"step": 1}, L), (None, None, None))


class SmokeCheck(unittest.TestCase):
    def check(self, episodes, eval_json=None):
        with tempfile.TemporaryDirectory() as d:
            arm = write_arm(d, "smoke", episodes, eval_json)
            return sa.smoke_check(arm, L)

    def test_good_arm_passes(self):
        ok, text = self.check({2: good_episode(), 3: good_episode()})
        self.assertTrue(ok, text)
        self.assertIn("2 集 12 步", text)

    def test_missing_key_fails_and_names_it(self):
        eps = {2: good_episode()}
        eps[2][3] = row(3, ack=None, with_ack_key=False)
        ok, text = self.check(eps)
        self.assertFalse(ok)
        self.assertIn("plugin_ack_sim_t", text)

    def test_first_step_must_be_null_and_the_rest_not(self):
        eps = {2: good_episode()}
        eps[2][0]["safe_gap_s"] = 0.1
        eps[2][2]["obs_gap_s"] = None
        ok, text = self.check(eps)
        self.assertFalse(ok)
        self.assertIn("safe_gap_s", text)
        self.assertIn("obs_gap_s", text)

    def test_too_few_acks_fails(self):
        eps = {2: good_episode(10)}
        for i in (1, 2, 3):                         # 3/10 没确认到：70% < 95%
            eps[2][i]["plugin_ack_sim_t"] = None
        ok, text = self.check(eps)
        self.assertFalse(ok)
        self.assertIn("7/10", text)

    def test_ack_before_the_dispatch_is_impossible(self):
        eps = {2: good_episode(6)}
        eps[2][4]["plugin_ack_sim_t"] = eps[2][4]["pose_ts"] / NS + 0.2 - 0.05   # 比派发戳早 50 ms
        ok, text = self.check(eps)
        self.assertFalse(ok)
        self.assertIn("早于", text)

    def test_one_period_of_slack_is_allowed(self):
        eps = {2: good_episode(6)}
        eps[2][4]["plugin_ack_sim_t"] = eps[2][4]["pose_ts"] / NS + 0.2 - 0.005   # 一个 state 周期以内
        ok, text = self.check(eps)
        self.assertTrue(ok, text)

    def test_partial_or_missing_eval_fails(self):
        ok, text = self.check({2: good_episode()}, {"partial": True, "episodes_done": 1, "failed_count": 0})
        self.assertFalse(ok)
        self.assertIn("partial", text)
        with tempfile.TemporaryDirectory() as d:
            ok, text = sa.smoke_check(os.path.join(d, "nope"), L)
        self.assertFalse(ok)
        self.assertIn("eval.json", text)


class Summary(unittest.TestCase):
    def build(self, idle_delay_s, load_delay_s):
        d = tempfile.mkdtemp(prefix="w113-sum-")
        write_arm(d, "smoke", {2: good_episode(), 3: good_episode()})
        write_arm(d, "idle", {s: good_episode(40, idle_delay_s) for s in (2, 3, 4, 6, 7)})
        write_arm(d, "load", {s: good_episode(40, load_delay_s) for s in (2, 3, 4, 6, 7)})
        with open(os.path.join(d, "idle.cg"), "w") as f:
            f.write("start 1000.0 0 0 0\nend 1100.0 600000000 2 5000\n")
        return d

    def test_both_conditions_fast_means_record_only(self):
        d = self.build(0.005, 0.005)
        text = sa.summarize(d, L)
        self.assertIn("只记录，不改迟到判据", text)
        self.assertIn("冒烟判定：**PASS**", text)
        self.assertIn("平均 6.00 核", text)                   # 600e6 µs / 100 s
        self.assertTrue(os.path.isfile(os.path.join(d, "summary.md")))

    def test_slow_loaded_condition_switches_the_late_criterion(self):
        d = self.build(0.005, 0.030)
        text = sa.summarize(d, L)
        self.assertIn("本轮不改迟到判据", text)
        self.assertIn("由主会话定", text)
        self.assertIn("load：p99 = 30.0 ms（最大 30.0 ms），≥ 10 ms", text)
        self.assertIn("idle：p99 = 5.0 ms（最大 5.0 ms），< 10 ms", text)
        self.assertIn("p50（中位）", text)

    def test_partial_reruns_are_listed_in_summary(self):
        d = self.build(0.005, 0.005)
        write_arm(d, "load.partial-023000", {2: good_episode(10), 3: good_episode(10)},
                  {"episodes_done": 2, "failed_count": 0, "partial": True})
        text = sa.summarize(d, L)
        self.assertIn("load.partial-023000", text)
        self.assertIn("2 集", text)

    def test_measurement_check_validates_both_arms(self):
        d = self.build(0.005, 0.025)
        ok, msg = sa.measurement_check(d, L, min_episodes=5)
        self.assertTrue(ok, msg)
        d_short = tempfile.mkdtemp(prefix="w113-mchk-")
        write_arm(d_short, "idle", {2: good_episode(), 3: good_episode()})
        write_arm(d_short, "load", {s: good_episode() for s in (2, 3, 4, 6, 7)})
        ok2, msg2 = sa.measurement_check(d_short, L, min_episodes=5)
        self.assertFalse(ok2)
        self.assertIn("idle", msg2)

    def test_exactly_ten_ms_is_not_below_the_limit(self):
        d = self.build(0.005, 0.0100000001)
        self.assertIn("≥ 10 ms", sa.summarize(d, L))

    def test_missing_arms_are_marked_not_finished(self):
        d = tempfile.mkdtemp(prefix="w113-sum-")
        write_arm(d, "idle", {2: good_episode()})
        text = sa.summarize(d, L)
        self.assertIn("未完成", text)
        self.assertIn("两个条件的数据不全，不下结论", text)


class Cgroup(unittest.TestCase):
    def test_read_cg(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.cg")
            with open(p, "w") as f:
                f.write("start 100.0 1000000 1 10\nend 200.0 801000000 4 9010\n")
            cores, dn, dh, wall = sa.read_cg(p)
            self.assertAlmostEqual(cores, 8.0)
            self.assertEqual((dn, dh, wall), (3, 9000, 100.0))
            self.assertIsNone(sa.read_cg(os.path.join(d, "none.cg")))


if __name__ == "__main__":
    unittest.main()
