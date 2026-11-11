#!/usr/bin/env python3
"""select_params.py 的单测：排序规则（含三层并列）、残缺数据拒选、非候选臂不参与。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_select_params.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_params as sp  # noqa: E402


def make_dev(root, results, scenes=("sA", "sB"), routes=None, k=1, extra_arms=True, incomplete=None, lost=None):
    """results = {臂名: (r_safe, d_look, [每条路线的 (success_stop, collide_first, timeout)] 按场景 × 路线展开的顺序)}。"""
    routes = routes or {"sA": [0, 1], "sB": [2, 3]}
    arms = [{"name": n, "base": "rule-occ", "ckpt": None, "guard": "route", "rule_params": {"r_safe": rs, "d_look": dl}}
            for n, (rs, dl, _) in results.items()]
    if extra_arms:
        arms.append({"name": "forward+route@0.5", "base": "forward", "ckpt": None, "guard": "route", "hold": 0.5, "step_mult": 2})
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"scenes": list(scenes), "routes": routes, "k": k, "arms": arms}, f)
    for n, (_, _, outs) in results.items():
        it = iter(outs)
        for s in scenes:
            for r in routes[s]:
                d = os.path.join(root, n, s, f"r{r}")
                os.makedirs(os.path.join(d, "ep_00000"))
                succ, cf, to = next(it)
                is_lost = lost == (n, s, r)          # 这一集失败：没有 metrics.json，done.json 记失败
                if not is_lost:
                    with open(os.path.join(d, "ep_00000", "metrics.json"), "w", encoding="utf-8") as f:
                        json.dump({"success_stop": succ, "collide_first": cf, "timeout": to}, f)
                if incomplete != (n, s, r):
                    with open(os.path.join(d, "done.json"), "w", encoding="utf-8") as f:
                        json.dump({"failures": [{"episode": 0}] if is_lost else [], "episodes": [] if is_lost else [{"episode": 0}]}, f)
    # 测速臂有结果但不是候选
    d = os.path.join(root, "forward+route@0.5", "sA", "r0", "ep_00000")
    if extra_arms:
        os.makedirs(d)
        with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"success_stop": True, "collide_first": False, "timeout": False}, f)


def run(root, *extra):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = sp.main(["--dev", root, *extra])
    return rc, out.getvalue(), err.getvalue()


S, F = (True, False, False), (False, False, False)        # 到点 / 没到点也没撞
CF = (False, True, False)                                  # 没到点就撞了


class TestSelectParams(unittest.TestCase):
    def test_highest_sr_wins(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"a": (0.35, 1.2, [S, S, F, F]), "b": (0.20, 0.8, [S, S, S, F]), "c": (0.25, 1.2, [S, F, F, F])})
            rc, out, _ = run(d)
            self.assertEqual(rc, 0)
            self.assertTrue(out.strip().splitlines()[-1] == "R_SAFE=0.20 D_LOOK=0.8")        # SR 0.75 最高，r_safe 小也选它
            sel = json.load(open(os.path.join(d, "dev_selection.json"), encoding="utf-8"))
            self.assertEqual([c["arm"] for c in sel["ranked"]], ["b", "a", "c"])
            self.assertEqual(sel["chosen"]["arm"], "b")

    def test_tie_on_sr_takes_larger_r_safe(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"lo": (0.20, 1.2, [S, S, F, F]), "hi": (0.35, 0.8, [S, F, S, F]), "mid": (0.25, 1.2, [S, S, F, F])})
            rc, out, _ = run(d)
            self.assertEqual(out.strip().splitlines()[-1], "R_SAFE=0.35 D_LOOK=0.8")

    def test_tie_on_sr_and_r_safe_takes_fewer_collisions_first(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"d08": (0.25, 0.8, [S, S, F, CF]), "d12": (0.25, 1.2, [S, S, CF, CF])})      # SR 同 0.5，撞得少的是 d_look 小的 d08
            rc, out, _ = run(d)
            self.assertEqual(out.strip().splitlines()[-1], "R_SAFE=0.25 D_LOOK=0.8")
            self.assertEqual(json.load(open(os.path.join(d, "dev_selection.json"), encoding="utf-8"))["chosen"]["arm"], "d08")

    def test_full_tie_takes_larger_d_look(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"d08": (0.25, 0.8, [S, S, F, F]), "d12": (0.25, 1.2, [S, F, S, F])})
            rc, out, _ = run(d)
            self.assertEqual(out.strip().splitlines()[-1], "R_SAFE=0.25 D_LOOK=1.2")

    def test_incomplete_candidate_is_refused_unless_allowed(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"a": (0.35, 1.2, [S, S, S, S]), "b": (0.20, 0.8, [S, F, F, F])}, incomplete=("a", "sB", 3))
            rc, out, err = run(d)
            self.assertEqual(rc, 3)
            self.assertIn("a", err)
            self.assertFalse(os.path.exists(os.path.join(d, "dev_selection.json")))
            rc, out, _ = run(d, "--allow-partial")
            self.assertEqual(rc, 0)
            self.assertTrue(json.load(open(os.path.join(d, "dev_selection.json"), encoding="utf-8"))["partial_allowed"])

    def test_speed_probe_arm_is_not_a_candidate(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"a": (0.35, 1.2, [F, F, F, F])})
            cands = sp.collect(d)
            self.assertEqual([c["arm"] for c in cands], ["a"])

    def test_no_candidates_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {})
            rc, _, err = run(d)
            self.assertEqual(rc, 2)

    def test_sr_denominator_counts_failed_episodes_as_failures(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"a": (0.35, 1.2, [S, S, S, S])}, lost=("a", "sB", 3))
            (c,) = sp.collect(d)
            self.assertEqual((c["n_req"], c["n_done"], c["k_stop"], c["sr_stop"], c["complete"]), (4, 3, 3, 0.75, False))

    def test_metrics_without_done_json_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            make_dev(d, {"a": (0.35, 1.2, [S, S, S, S])}, incomplete=("a", "sA", 0))
            (c,) = sp.collect(d)
            self.assertEqual((c["n_req"], c["k_stop"], c["complete"]), (4, 4, False))


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
