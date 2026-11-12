"""recheck_sweep.py 的单元测试：对账与篡改检出、P6 重跑与停顿步/位移统计、失败构成符号翻转与二项波动基准。

运行：cd experiments/2026-11-12-recipe-stability && <解释器> -m unittest test_recheck_sweep
"""
import json
import math
import os
import tempfile
import unittest

import recheck_sweep as rs
import summarize_sweep as ss

SEEDS = [30012 + 10 * i for i in range(20)]


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_episode_actions(arm_dir, seed, actions_seq, gaps_s=None, lag_s=0.20, disp_m=0.22):
    ep_dir = os.path.join(arm_dir, "episodes", f"seed_{seed}")
    os.makedirs(ep_dir, exist_ok=True)
    ts_ns = 10_000_000_000
    rows = []
    for i, a in enumerate(actions_seq):
        if i > 0:
            gap = gaps_s[i - 1] if (gaps_s and i - 1 < len(gaps_s)) else 0.90
            ts_ns += int(round(gap * 1e9))
        rows.append({
            "step": i,
            "rgb_ts": ts_ns,
            "action_executed": a,
            "collision": False,
            "displacement_m": disp_m + 0.01 * min(i, 4),
            "dispatch_lag_s": lag_s,
            "late_dispatch": lag_s > 0.22,
        })
    with open(os.path.join(ep_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _make_arm(cl_dir, name, outcomes_by_seed, late_seeds=(), cg_throttled_usec=6_000_000):
    arm_dir = os.path.join(cl_dir, name)
    os.makedirs(arm_dir, exist_ok=True)
    per_ep = []
    failed = []
    late_list = []
    for sd in SEEDS:
        if sd in late_seeds:
            failed.append({"seed": sd, "rc": 0, "late_dispatch_steps": 1, "late_step_ids": [2]})
            late_list.append({"seed": sd, "late_dispatch_steps": 1, "late_step_ids": [2], "dispatch_lag_s_max": 0.35})
            continue
        oc = outcomes_by_seed[sd]
        seq = [1, 1, 1, 1, 2, 3]
        _write_episode_actions(arm_dir, sd, seq)
        per_ep.append({
            "seed": sd,
            "geometry_gid": f"gid_{sd:05d}",
            "success": oc == "success",
            "collision": oc == "collision",
            "corridor_violation": oc == "corridor",
            "steps": len(seq),
            "actions": {"1": 4, "2": 1, "3": 1},
        })
    ev = {
        "schema": "uavrgb.eval.v1",
        "episodes_requested": 20,
        "episodes_done": len(per_ep),
        "failed_count": len(failed),
        "partial": False,
        "metric_version": rs.EXPECTED_METRIC,
        "action_latency_s": rs.EXPECTED_LATENCY_S,
        "corridor_half_width": rs.EXPECTED_CORRIDOR_M,
        "split": rs.EXPECTED_SPLIT,
        "seeds": SEEDS,
        "wall_s": 100.0,
        "ckpt": f"/tmp/{name}.bin",
        "ckpt_sha256": "11" * 32,
        "late_dispatch_episodes": late_list,
        "failed": failed,
        "per_episode": per_ep,
    }
    with open(os.path.join(arm_dir, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f)
    with open(os.path.join(cl_dir, f"{name}.cg"), "w", encoding="utf-8") as f:
        f.write(f"start 1000.0 0 10 0\nend 1100.0 900000000 60 {cg_throttled_usec}\n")


class TestRecheckSweep(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.cl = os.path.join(self.td.name, "closed-loop")
        os.makedirs(self.cl, exist_ok=True)
        with open(os.path.join(self.cl, "config.env"), "w", encoding="utf-8") as f:
            f.write("SEEDS=" + ",".join(map(str, SEEDS)) + "\nL_STAR=0.20\nCODE_HEAD=abc123\nEXTRA_ARM=R2e\n")

    def tearDown(self):
        self.td.cleanup()

    def _populate_full_sweep(self):
        for s in (0, 1, 2):
            oc_m = {sd: ("success" if i < 12 else ("collision" if i == 12 else "corridor")) for i, sd in enumerate(SEEDS)}
            oc_0 = {sd: ("success" if i < 13 else ("collision" if i < 16 else "corridor")) for i, sd in enumerate(SEEDS)}
            oc_p = {sd: ("success" if i < 13 else ("collision" if i < 19 else "corridor")) for i, sd in enumerate(SEEDS)}
            oc_x = {sd: ("success" if i < 14 else "corridor") for i, sd in enumerate(SEEDS)}
            _make_arm(self.cl, f"lvl-m0.5-s{s}", oc_m, cg_throttled_usec=500_000_000)
            _make_arm(self.cl, f"lvl-0-s{s}", oc_0, cg_throttled_usec=600_000_000)
            _make_arm(self.cl, f"lvl-p0.5-s{s}", oc_p, cg_throttled_usec=700_000_000)
            _make_arm(self.cl, f"x-R2e-s{s}", oc_x, cg_throttled_usec=800_000_000)
        ss.main([self.cl, "--note", "unit-test"])

    def test_reconcile_clean_and_detect_tamper(self):
        self._populate_full_sweep()
        post_dir = os.path.join(self.cl, "post")
        rc = rs.main(["--closed-loop-dir", self.cl, "--out-dir", post_dir])
        self.assertEqual(rc, 0)
        out = _read_json(os.path.join(post_dir, "recheck.json"))
        self.assertTrue(out["reconciliation"]["all_match"])
        self.assertTrue(out["all_protocols_ok"])
        self.assertTrue(out["gid_consistent_across_12_arms"])
        self.assertTrue(out["timing_totals"]["all_steps_match_eval_json"])
        for c in out["paired_comparisons"].values():
            self.assertEqual(c["p_cluster_abs_diff"], 0.0)
            self.assertEqual(c["p_mcnemar_abs_diff"], 0.0)

        # 篡改 summary.json 的一格成功数，必须被检出并返回非 0
        sj_path = os.path.join(self.cl, "summary.json")
        sj = _read_json(sj_path)
        sj["levels"]["0"]["total"] += 1
        with open(sj_path, "w", encoding="utf-8") as f:
            json.dump(sj, f)
        rc_bad = rs.main(["--closed-loop-dir", self.cl, "--out-dir", post_dir])
        self.assertEqual(rc_bad, 2)
        out_bad = _read_json(os.path.join(post_dir, "recheck.json"))
        self.assertFalse(out_bad["reconciliation"]["all_match"])
        items = [m["item"] for m in out_bad["reconciliation"]["mismatches"]]
        self.assertIn("level.0.total", items)

    def test_p6_retry_and_stall_detection_and_cruise_disp(self):
        oc = {sd: "success" for sd in SEEDS}
        _make_arm(self.cl, "lvl-0-s0", oc, late_seeds=(SEEDS[0],), cg_throttled_usec=450_000_000)
        retry_dir = os.path.join(self.cl, "lvl-0-s0-retry1")
        os.makedirs(retry_dir, exist_ok=True)
        _write_episode_actions(retry_dir, SEEDS[0], [1, 1, 1, 1, 1], gaps_s=[0.90, 1.25, 0.90, 0.90], disp_m=0.20)
        retry_ev = {
            "episodes_requested": 1,
            "episodes_done": 1,
            "failed_count": 0,
            "partial": False,
            "per_episode": [{
                "seed": SEEDS[0],
                "geometry_gid": f"gid_{SEEDS[0]:05d}",
                "success": True,
                "collision": False,
                "corridor_violation": False,
                "steps": 5,
                "actions": {"1": 5},
            }],
        }
        with open(os.path.join(retry_dir, "eval.json"), "w", encoding="utf-8") as f:
            json.dump(retry_ev, f)

        arm_rec = rs.load_arm_p6(self.cl, "lvl-0-s0", SEEDS)
        self.assertTrue(arm_rec["reran"])
        self.assertEqual(arm_rec["outcomes"]["success"], 20)
        self.assertEqual(arm_rec["episodes"][SEEDS[0]]["source"], "retry1")

        t = rs.scan_arm_timing(arm_rec, self.cl)
        self.assertAlmostEqual(t["throttled_pct_wall"], 450.0)
        self.assertAlmostEqual(t["throttled_pct_quota12"], 450.0 / 12.0)
        self.assertEqual(t["stall_steps_gt_1p1s"], 1)
        self.assertEqual(t["stall_episodes"], [SEEDS[0]])
        self.assertAlmostEqual(t["obs_gap_max_s"], 1.25)
        self.assertGreater(t["cruise_disp_n"], 0)
        self.assertLess(t["cruise_disp_n"], t["steady_disp_n"])

    def test_failure_posthoc_and_binomial_reference(self):
        self._populate_full_sweep()
        out = rs.run_recheck(self.cl)
        # m0.5 撞柱 3 (i=12 处 +3) vs p0.5 撞柱 18 (i=13..18 共 6 处各 -3): 7 个非零几何 [+3, -3*6]
        # 2^7=128 种符号中 |S| >= 15 有 7 同号(2) + 6 同 1 异(14) = 16 种，p = 16/128 = 0.125
        col_cmp = out["failure_posthoc"]["collision"]["m0.5_vs_p0.5"]
        self.assertEqual((col_cmp["count_a"], col_cmp["count_b"], col_cmp["diff_episodes"]), (3, 18, -15))
        self.assertEqual((col_cmp["n_nonzero_geoms"], col_cmp["geoms_pos"], col_cmp["geoms_neg"]), (7, 1, 6))
        self.assertAlmostEqual(col_cmp["p_cluster"], 16.0 / 128.0)

        b = rs.binomial_sampling_reference(n=20, p=0.67, k_seeds=3)
        self.assertAlmostEqual(b["sd_count"], math.sqrt(20 * 0.67 * 0.33), places=12)
        self.assertTrue(0.0 < b["prob_range_ge_8"] < 0.10)


if __name__ == "__main__":
    unittest.main()
