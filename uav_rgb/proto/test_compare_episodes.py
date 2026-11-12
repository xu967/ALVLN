#!/usr/bin/env python3
"""proto/compare_episodes.py 的测试：合成 actions.jsonl 与 eval.json，不需要仿真。"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import compare_episodes as ce  # noqa: E402


def row(i, act, lag=0.2, late=False, ms=1.0, x=0.0, y=0.0, yaw=0.0, by="student", expert=1):
    return {"step": i, "timestamp_ns": 1_000_000_000 * (i + 15), "action_expert": expert, "action_executed": act,
            "executed_by": by, "dispatch_lag_s": lag, "late_dispatch": late, "infer_ms": ms, "collision": False,
            "pose_before": {"x": x, "y": y, "z": 2.0}, "yaw_before": yaw}


def write_arm(root, name, episodes, success=True):
    """episodes：{seed: [行]}；写 <root>/<name>/episodes/seed_<s>/actions.jsonl 与 eval.json。"""
    arm = os.path.join(root, name)
    per = []
    for seed, rows in episodes.items():
        d = os.path.join(arm, "episodes", f"seed_{seed}")
        os.makedirs(d)
        with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        per.append({"seed": seed, "success": success, "collision": False, "steps": len(rows), "spl": 0.9})
    with open(os.path.join(arm, "eval.json"), "w", encoding="utf-8") as f:
        json.dump({"per_episode": per}, f)
    return arm


class TestCompare(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.root = self.td.name

    def tearDown(self):
        self.td.cleanup()

    def base(self, n=6, acts=None):
        acts = acts or [1, 1, 3, 1, 1, 2]
        return [row(i, acts[i], x=0.1 * i) for i in range(n)]

    def test_identical(self):
        ref = write_arm(self.root, "ref", {1: self.base()})
        arm = write_arm(self.root, "arm", {1: self.base()})
        r = ce.run(ref, {"a": arm}, [1])
        v = r["arms"]["a"]["per_seed"][1]
        self.assertTrue(v["identical"])
        self.assertIsNone(v["first_divergent_step"])
        self.assertTrue(v["timestamps_equal_in_prefix"])
        self.assertTrue(r["arms"]["a"]["all_identical"])

    def test_action_divergence_reports_both_sides(self):
        ref_rows = self.base()
        arm_rows = self.base(acts=[1, 1, 3, 2, 1, 2])
        arm_rows[3] = row(3, 2, lag=0.231, late=True, ms=4.5, x=0.31, yaw=0.02)
        arm_rows[2] = row(2, 3, x=0.2, yaw=0.01)    # 分歧之前位姿已有微小差
        ref = write_arm(self.root, "ref", {1: ref_rows})
        arm = write_arm(self.root, "arm", {1: arm_rows})
        v = ce.run(ref, {"a": arm}, [1])["arms"]["a"]["per_seed"][1]
        self.assertFalse(v["identical"])
        self.assertEqual((v["first_divergent_step"], v["divergence_reason"]), (3, "action"))
        self.assertEqual((v["ref"]["at"]["action_executed"], v["arm"]["at"]["action_executed"]), (1, 2))
        self.assertAlmostEqual(v["arm"]["at"]["dispatch_lag_s"], 0.231)
        self.assertTrue(v["arm"]["at"]["late_dispatch"])
        self.assertEqual(v["arm"]["before"]["step"], 2)
        self.assertAlmostEqual(v["pose_gap_prefix_max"]["dyaw_rad"], 0.01)
        self.assertAlmostEqual(v["pose_gap_at_divergence"]["dx"], abs(0.3 - 0.31), places=9)

    def test_length_divergence(self):
        ref = write_arm(self.root, "ref", {1: self.base()})
        arm = write_arm(self.root, "arm", {1: self.base()[:4]})
        v = ce.run(ref, {"a": arm}, [1])["arms"]["a"]["per_seed"][1]
        self.assertEqual((v["identical"], v["first_divergent_step"], v["divergence_reason"]), (False, 4, "length"))
        self.assertIsNone(v["arm"]["at"])

    def test_retry_directory_wins_and_first_attempt_is_listed(self):
        ref = write_arm(self.root, "ref", {1: self.base()})
        first = [dict(r) for r in self.base()]
        first[2] = row(2, 3, late=True, lag=0.3)
        arm = write_arm(self.root, "arm", {1: first})
        write_arm(self.root, "arm-retry1", {1: self.base()})
        r = ce.run(ref, {"a": arm}, [1])["arms"]["a"]
        self.assertTrue(r["used"][1]["from_retry"])
        self.assertTrue(r["per_seed"][1]["identical"])
        self.assertEqual(r["per_seed"][1]["first_attempt_late_steps"], 1)

    def test_missing_seed_and_markdown(self):
        ref = write_arm(self.root, "ref", {1: self.base(), 2: self.base()})
        arm = write_arm(self.root, "arm", {1: self.base(acts=[1, 1, 3, 3, 1, 2])})
        res = ce.run(ref, {"a": arm}, [1, 2])
        self.assertTrue(res["arms"]["a"]["per_seed"][2]["missing"])
        self.assertFalse(res["arms"]["a"]["all_identical"])
        md = ce.render_md(res, [1, 2])
        self.assertIn("| 1 |", md)
        self.assertIn("缺", md)

    def test_stats(self):
        s = ce.arm_stats({1: [row(0, 1, ms=1.0), row(1, 1, ms=3.0, lag=0.25, late=True)]})
        self.assertEqual((s["steps"], s["late_dispatch_steps"]), (2, 1))
        self.assertAlmostEqual(s["infer_ms"]["p50"], 2.0)
        self.assertAlmostEqual(s["dispatch_lag_s"]["max"], 0.25)

    def test_yaw_wrap(self):
        self.assertAlmostEqual(ce.yaw_diff(3.1, -3.1), 2 * 3.14159265358979 - 6.2, places=6)


if __name__ == "__main__":
    unittest.main()
