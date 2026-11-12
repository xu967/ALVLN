"""summarize_sweep.py 的测试：统计函数对暴力枚举，预注册判定的边界，合成输出目录上的合并 / 重跑 / 迟到 / 未完成逻辑，CLI。

运行：cd experiments/2026-11-12-recipe-stability && <解释器> -m unittest test_sweep
"""
import itertools
import json
import os
import random
import tempfile
import unittest

import summarize_sweep as ss

SEEDS = [30012 + 10 * i for i in range(20)]        # 合成的 20 个几何 seed（只要求互不相同）


def row(seed, success=True, collision=False, corridor=False, actions=None, steps=30):
    return {"seed": seed, "geometry_gid": f"{seed:016x}", "success": success, "collision": collision, "corridor_violation": corridor,
            "steps": steps, "actions": actions or {"1": steps - 2, "2": 1, "3": 1}}


def write_eval(out, name, rows, late=(), aborted=(), partial=False, requested=20, cg=None, wall=1000.0):
    """写 <out>/<name>/eval.json：rows 是干净的集；late = 迟到的 seed；aborted = 采集中止的 seed。"""
    d = os.path.join(out, name)
    os.makedirs(d, exist_ok=True)
    failed = [{"seed": s, "rc": 0, "error": "late_dispatch: 1 步迟到", "late_dispatch_steps": 1, "late_step_ids": [5]} for s in late]
    failed += [{"seed": s, "rc": 2, "stderr_tail": ["x"]} for s in aborted]
    e = {"schema": "uavrgb.eval.v1", "episodes_requested": requested, "episodes_done": len(rows), "failed_count": len(failed), "partial": partial,
         "action_latency_s": 0.2, "metric_version": "sim09.v2.coast1s.corridor3m", "split": "val", "ckpt": "/x/ck.bin", "ckpt_sha256": "ab" * 32, "wall_s": wall,
         "late_dispatch_episodes": [{"seed": s, "late_dispatch_steps": 1, "late_step_ids": [5], "dispatch_lag_s_median": 0.2, "dispatch_lag_s_max": 0.35} for s in late],
         "failed": failed, "skipped": [], "per_episode": rows}
    with open(os.path.join(d, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(e, f)
    if cg:
        with open(os.path.join(out, name + ".cg"), "w", encoding="utf-8") as f:
            f.write(cg)


def write_cfg(out, extra=""):
    with open(os.path.join(out, "config.env"), "w", encoding="utf-8") as f:
        f.write("SEEDS=" + ",".join(map(str, SEEDS)) + f"\nL_STAR=0.20\nCODE_HEAD=deadbeef\nEXTRA_ARM={extra}\n")


def full_arm(out, name, ok_seeds, **kw):
    """20 集全部干净，ok_seeds 里的成功；其余随机归为撞柱 / 越界 / 未到线（按 seed 取模，保证可复现）。"""
    rows = []
    for s in SEEDS:
        if s in ok_seeds:
            rows.append(row(s))
        else:
            k = s % 3
            rows.append(row(s, success=False, collision=(k == 0), corridor=(k == 1)))
    write_eval(out, name, rows, **kw)


class TestStats(unittest.TestCase):
    def test_signflip_matches_bruteforce(self):
        rng = random.Random(1)
        for _ in range(40):
            d = [rng.choice([-3, -2, -1, 0, 0, 1, 2, 3]) for _ in range(rng.randint(1, 9))]
            nz = [x for x in d if x]
            obs = abs(sum(d))
            if not nz:
                self.assertEqual(ss.signflip_exact_p(d), 1.0)
                continue
            hit = sum(1 for signs in itertools.product((1, -1), repeat=len(nz)) if abs(sum(s * v for s, v in zip(signs, nz))) >= obs)
            self.assertAlmostEqual(ss.signflip_exact_p(d), hit / 2 ** len(nz), places=12)

    def test_signflip_known_values(self):
        self.assertAlmostEqual(ss.signflip_exact_p([1, 1, 1, 1, 1]), 2 / 32)
        self.assertEqual(ss.signflip_exact_p([0, 0, 0]), 1.0)
        self.assertEqual(ss.signflip_exact_p([3]), 1.0)                     # 单个非零几何：|T| 永远等于观测，p = 1
        self.assertAlmostEqual(ss.signflip_exact_p([2, 1, -1]), 6 / 8)      # 观测 |2+1−1|=2；{±2±1±1} 的 8 个和是 ±4、±2、±2、0、0，|和| ≥ 2 的占 6/8
        self.assertAlmostEqual(ss.signflip_exact_p([1] * 20), 2 / 2 ** 20)

    def test_signflip_symmetric_in_sign(self):
        d = [2, -1, 3, 0, 1]
        self.assertEqual(ss.signflip_exact_p(d), ss.signflip_exact_p([-x for x in d]))

    def test_mcnemar(self):
        self.assertEqual(ss.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(ss.mcnemar_exact(5, 0), 2 / 32)
        self.assertEqual(ss.mcnemar_exact(3, 3), 1.0)
        self.assertAlmostEqual(ss.mcnemar_exact(0, 5), ss.mcnemar_exact(5, 0))

    def test_bootstrap(self):
        self.assertEqual(ss.bootstrap_ci([1] * 20), (20, 20))
        self.assertEqual(ss.bootstrap_ci([0] * 20), (0, 0))
        a, b = ss.bootstrap_ci([1, -1] * 10), ss.bootstrap_ci([1, -1] * 10)
        self.assertEqual(a, b)                                              # 固定随机种子：同输入同输出
        self.assertLessEqual(a[0], 0)
        self.assertGreaterEqual(a[1], 0)
        lo, hi = ss.bootstrap_ci([2, 1, 1, 0, 0, 0, 1, -1, 2, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0])
        self.assertLess(lo, hi)
        self.assertGreater(lo, -5)

    def test_wilson(self):
        lo, hi = ss.wilson(45, 60)
        self.assertTrue(0.62 < lo < 0.65 and 0.82 < hi < 0.86, (lo, hi))
        self.assertTrue(all(x != x for x in ss.wilson(0, 0)))


class TestVerdicts(unittest.TestCase):
    def test_spread_boundaries(self):
        v = lambda a, b, c: ss.verdict_spread({"m0.5": a, "0": b, "p0.5": c})[0]    # noqa: E731
        self.assertEqual(v(40, 48, 44), "sensitive")        # 极差 8：含边界
        self.assertEqual(v(41, 48, 44), "between")          # 极差 7
        self.assertEqual(v(44, 48, 46), "between")          # 极差 4：含边界，不算「不敏感」
        self.assertEqual(v(45, 48, 46), "insensitive")      # 极差 3
        self.assertEqual(v(46, 46, 46), "insensitive")
        self.assertEqual(v(40, None, 44), "incomplete")
        self.assertEqual(ss.verdict_spread({"m0.5": 1, "0": 2})[0], "incomplete")

    def test_extra_rule(self):
        def cnt(per_seed):
            return {"per_seed": dict(zip(ss.TRAIN_SEEDS, per_seed)), "range": max(per_seed) - min(per_seed), "total": sum(per_seed)}
        self.assertTrue(ss.verdict_extra(cnt([15, 16, 18]), 52)[0])     # 极差 3，总数 49 ≥ 52−3
        self.assertFalse(ss.verdict_extra(cnt([14, 16, 18]), 50)[0])    # 极差 4
        self.assertFalse(ss.verdict_extra(cnt([15, 15, 18]), 52)[0])    # 总数 48 < 49
        self.assertTrue(ss.verdict_extra(cnt([16, 16, 16]), 51)[0])     # 总数 48 = 51−3，含边界


class TestMerge(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.out = self.td.name

    def tearDown(self):
        self.td.cleanup()

    def test_late_retry_clean_is_used(self):
        rows = [row(s) for s in SEEDS if s not in (SEEDS[3], SEEDS[7])]
        write_eval(self.out, "a", rows, late=[SEEDS[3], SEEDS[7]])
        write_eval(self.out, "a-retry1", [row(SEEDS[3], success=True), row(SEEDS[7], success=False, collision=True)], requested=2)
        m = ss.merged_arm(self.out, "a", SEEDS)
        self.assertEqual(m["status"], "complete")
        self.assertEqual(m["eps"][SEEDS[3]]["state"], "retry")
        self.assertTrue(ss.success_of(m["eps"][SEEDS[3]]))
        self.assertEqual(m["eps"][SEEDS[7]]["state"], "retry")
        self.assertFalse(ss.success_of(m["eps"][SEEDS[7]]))
        self.assertEqual(ss.outcome_of(m["eps"][SEEDS[7]]), "collision")
        self.assertTrue(m["reran"])

    def test_late_retry_still_late_or_missing(self):
        rows = [row(s) for s in SEEDS if s not in (SEEDS[1], SEEDS[2])]
        write_eval(self.out, "b", rows, late=[SEEDS[1], SEEDS[2]])
        write_eval(self.out, "b-retry1", [row(SEEDS[1])], late=[SEEDS[2]], requested=2)
        m = ss.merged_arm(self.out, "b", SEEDS)
        self.assertEqual(m["eps"][SEEDS[1]]["state"], "retry")
        self.assertEqual(m["eps"][SEEDS[2]]["state"], "late")
        self.assertFalse(ss.success_of(m["eps"][SEEDS[2]]))
        self.assertEqual(ss.outcome_of(m["eps"][SEEDS[2]]), "late")
        write_eval(self.out, "c", rows, late=[SEEDS[1], SEEDS[2]])           # 没有重跑目录
        m = ss.merged_arm(self.out, "c", SEEDS)
        self.assertEqual(m["eps"][SEEDS[1]]["state"], "late")
        self.assertFalse(m["reran"])

    def test_aborted_not_retried(self):
        rows = [row(s) for s in SEEDS if s != SEEDS[5]]
        write_eval(self.out, "d", rows, aborted=[SEEDS[5]])
        write_eval(self.out, "d-retry1", [row(SEEDS[5])], requested=1)      # 即使有重跑目录，中止的不换
        m = ss.merged_arm(self.out, "d", SEEDS)
        self.assertEqual(m["eps"][SEEDS[5]]["state"], "aborted")
        self.assertFalse(ss.success_of(m["eps"][SEEDS[5]]))

    def test_status(self):
        write_eval(self.out, "p", [row(s) for s in SEEDS[:5]], partial=True)
        self.assertEqual(ss.merged_arm(self.out, "p", SEEDS)["status"], "partial")
        write_eval(self.out, "q", [row(s) for s in SEEDS[:5]])
        self.assertEqual(ss.merged_arm(self.out, "q", SEEDS)["status"], "incomplete")
        self.assertEqual(ss.merged_arm(self.out, "none-yet", SEEDS)["status"], "none")

    def test_late_seeds_cli(self):
        write_eval(self.out, "e", [row(s) for s in SEEDS if s != SEEDS[9]], late=[SEEDS[9]])
        self.assertEqual(ss.late_seeds(os.path.join(self.out, "e")), [SEEDS[9]])
        self.assertEqual(ss.late_seeds(os.path.join(self.out, "nothing")), [])

    def test_read_cg(self):
        with open(os.path.join(self.out, "z.cg"), "w", encoding="utf-8") as f:
            f.write("start 1000.0 5000000 10 2000000\nend 1100.0 15000000 13 2500000\n")
        cg = ss.read_cg(self.out, "z")
        self.assertAlmostEqual(cg["avg_cores"], 0.1)
        self.assertEqual(cg["throttled"], 3)
        self.assertAlmostEqual(cg["throttled_s"], 0.5)
        with open(os.path.join(self.out, "y.cg"), "w", encoding="utf-8") as f:
            f.write("start 1000.0 5000000 10 2000000\n")
        self.assertIsNone(ss.read_cg(self.out, "y"))                        # 缺 end（臂还在跑）


class TestBuild(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.out = self.td.name

    def tearDown(self):
        self.td.cleanup()

    def make_levels(self, ok):
        """ok: {档 key: {训练种子: 成功的几何 seed 集合}}"""
        for k, per in ok.items():
            for s, okset in per.items():
                full_arm(self.out, ss.lvl_arm(k, s), okset)

    def test_full_summary_numbers(self):
        write_cfg(self.out)
        base = set(SEEDS[:14])
        self.make_levels({"m0.5": {s: set(SEEDS[:10]) for s in ss.TRAIN_SEEDS},      # 每种子 10 → 30
                          "0": {0: base, 1: base, 2: base},                           # 每种子 14 → 42
                          "p0.5": {s: set(SEEDS[:12]) for s in ss.TRAIN_SEEDS}})      # 每种子 12 → 36
        text, res = ss.build(self.out, "测试")
        self.assertEqual(res["levels"]["m0.5"]["total"], 30)
        self.assertEqual(res["levels"]["0"]["total"], 42)
        self.assertEqual(res["levels"]["p0.5"]["total"], 36)
        self.assertEqual(res["verdicts"]["spread"]["code"], "sensitive")             # 42 − 30 = 12 ≥ 8
        r = res["tests"]["m0.5_vs_0"]
        self.assertEqual(r["diff"], -12)
        self.assertEqual(r["geoms_nonzero"], 4)
        self.assertEqual(r["geoms_neg"], 4)
        self.assertAlmostEqual(r["p_cluster"], 2 / 2 ** 4 * 1.0)                      # 4 个几何都 −3：P(|T|=12) = 2/16
        self.assertEqual((r["mcnemar_b"], r["mcnemar_c"]), (0, 12))
        self.assertAlmostEqual(r["p_mcnemar"], 2 / 2 ** 12)
        self.assertFalse(r["agree"])                                                  # 聚类 p = 0.125，逐集 p = 0.0005：结论不一致
        self.assertIn("不一致", text)
        self.assertIn("**42**", text)
        self.assertNotIn("配方臂 R1", text)

    def test_outcome_counts_and_action_shares(self):
        write_cfg(self.out)
        for k, _ in ss.LEVELS:
            for s in ss.TRAIN_SEEDS:
                full_arm(self.out, ss.lvl_arm(k, s), set(SEEDS[:10]))
        _, res = ss.build(self.out)
        c = res["levels"]["0"]
        total_fail = sum(c["outcomes"][k] for k in ("collision", "corridor", "not_reached"))
        self.assertEqual(c["outcomes"]["success"], 30)
        self.assertEqual(total_fail, 30)
        self.assertEqual(c["outcomes"]["late"] + c["outcomes"]["aborted"], 0)
        sh = c["action_shares"]
        self.assertAlmostEqual(sum(sh.values()), 1.0)
        self.assertAlmostEqual(sh["F"], 28 / 30)                                      # 每集 28 步 F、1 步 L、1 步 R
        self.assertAlmostEqual(sh["L"], 1 / 30)

    def test_incomplete_levels_not_judged(self):
        write_cfg(self.out)
        for s in ss.TRAIN_SEEDS:
            full_arm(self.out, ss.lvl_arm("0", s), set(SEEDS[:10]))
        full_arm(self.out, ss.lvl_arm("m0.5", 0), set(SEEDS[:10]))                   # 只有 1 个种子
        text, res = ss.build(self.out)
        self.assertEqual(res["verdicts"]["spread"]["code"], "incomplete")
        self.assertIn("未完成", text)
        self.assertNotIn("m0.5_vs_0", res["tests"])

    def test_extra_arm_pass_and_fail(self):
        write_cfg(self.out, extra="R1")
        z = set(SEEDS[:15])
        for k in ("m0.5", "p0.5", "0"):
            for s in ss.TRAIN_SEEDS:
                full_arm(self.out, ss.lvl_arm(k, s), z)
        for s, n in zip(ss.TRAIN_SEEDS, (15, 14, 13)):                                # 极差 2，总数 42 ≥ 45 − 3
            full_arm(self.out, ss.extra_arm("R1", s), set(SEEDS[:n]))
        text, res = ss.build(self.out)
        self.assertTrue(res["verdicts"]["extra"]["pass"])
        self.assertEqual(res["extra"]["range"], 2)
        self.assertIn("配方臂 R1", text)
        self.assertNotIn("替代", text)                                                # 卡里的臂，不带替代标注
        for s, n in zip(ss.TRAIN_SEEDS, (15, 14, 10)):                                # 极差 5
            full_arm(self.out, ss.extra_arm("R1", s), set(SEEDS[:n]))
        _, res = ss.build(self.out)
        self.assertFalse(res["verdicts"]["extra"]["pass"])
        self.assertIn("extra_vs_0", res["tests"])

    def test_substitute_extra_arm_is_labelled(self):
        write_cfg(self.out, extra="R2e")
        z = set(SEEDS[:15])
        for k in ("m0.5", "p0.5", "0"):
            for s in ss.TRAIN_SEEDS:
                full_arm(self.out, ss.lvl_arm(k, s), z)
        for s in ss.TRAIN_SEEDS:
            full_arm(self.out, ss.extra_arm("R2e", s), z)
        text, res = ss.build(self.out)
        self.assertEqual(res["extra"]["tag"], "R2e")
        self.assertTrue(res["verdicts"]["extra"]["pass"])
        self.assertIn("配方臂 R2e（替代）", text)
        self.assertIn("替代臂：卡里的 R2、R3 被训练器拒绝零权重", text)
        self.assertIn("主会话可以不采用", text)
        self.assertIn("**判定二（配方臂 R2e（替代））**", text)

    def test_late_and_retry_in_summary(self):
        write_cfg(self.out)
        z = set(SEEDS[:12])
        for k in ("m0.5", "p0.5"):
            for s in ss.TRAIN_SEEDS:
                full_arm(self.out, ss.lvl_arm(k, s), z)
        for s in (1, 2):
            full_arm(self.out, ss.lvl_arm("0", s), z)
        rows = [row(sd) if sd in z else row(sd, success=False, collision=True) for sd in SEEDS if sd != SEEDS[0]]
        write_eval(self.out, ss.lvl_arm("0", 0), rows, late=[SEEDS[0]])               # 种子 0 在 SEEDS[0] 上迟到
        write_eval(self.out, ss.lvl_arm("0", 0) + "-retry1", [row(SEEDS[0])], requested=1)
        text, res = ss.build(self.out)
        self.assertEqual(res["levels"]["0"]["total"], 36)                             # 3×12，迟到那集重跑后成功（SEEDS[0] 在 z 里）
        self.assertIn("重评", text)
        self.assertIn("重跑后干净，采用重跑结果", text)
        self.assertEqual(res["levels"]["0"]["outcomes"]["late"], 0)

    def test_cli_writes_files(self):
        write_cfg(self.out)
        for k, _ in ss.LEVELS:
            for s in ss.TRAIN_SEEDS:
                full_arm(self.out, ss.lvl_arm(k, s), set(SEEDS[:10]), cg="start 1.0 0 0 0\nend 11.0 10000000 2 1000000\n")
        self.assertEqual(ss.main([self.out, "--note", "进行中"]), 0)
        with open(os.path.join(self.out, "summary.md"), encoding="utf-8") as f:
            md = f.read()
        self.assertIn("⚠", md)                                                        # 限流增量 2 > 0
        self.assertIn("进行中", md)
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["verdicts"]["spread"]["code"], "insensitive")


if __name__ == "__main__":
    unittest.main()
