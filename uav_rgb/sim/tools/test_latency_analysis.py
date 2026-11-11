#!/usr/bin/env python3
"""experiments/2026-11-10-action-latency/analyze_round1.py 与 profile_late_steps.py 的单测（合成数据，不起 gz）。

背景：第一轮标定的门没过（L=0.16 与 0.20），K=3 的稳态前飞位移比 K=1 低 2.6%–3.3% 而逐步生效时长只差 0.1%–0.2%。
主会话的猜测是「步段 / 电量不同」；离线归因要回答差来自哪里，需要把稳态前进步按「连续前进的第几步」分层、
对 K=1 的构成做直接标准化，还要把 A* 的出堆数、是否退回局部规划数出来。这些是新的统计逻辑，先写这里的测试。

覆盖：
  RunPosition        fwd_samples 的连续前进位置 r（含第 0 步计数、碰撞步只计数不作样本）、select 的 r 范围；
  Decomposition      std_rel：只有构成不同时标准化后差为 0，层内真有差时只剩层内的差；decompose 的口径一路收窄；
  Stats              mann_whitney / spearman / median_iqr 对手算值；
  CountingAndClasses count_call 在真教师上计出堆数、识别 A* 无路退回局部规划、调用后还原模块属性；classify 的边界；
                     late_rate_table；pick_controls 只从出现过迟到步的集里抽；verify_scene 的哈希核对。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_latency_analysis.py
"""
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# LATENCY_EXP_DIR：指向一份被故意改坏的脚本副本，用来确认这些测试真的会红（变异检查）；缺省是仓库里的脚本
EXP = os.path.abspath(os.environ.get("LATENCY_EXP_DIR") or os.path.join(HERE, "..", "..", "experiments", "2026-11-10-action-latency"))
sys.path.insert(0, EXP)


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


an = load("latency_analyze_round1", "analyze_round1.py")
pf = load("latency_profile_late_steps", "profile_late_steps.py")
import teacher  # noqa: E402

F, Y, S = 1, 2, 0   # FORWARD_SLOW / YAW_LEFT / SAFE_NEUTRAL


def steps(spec):
    """spec = [(动作, 位移)] → actions.jsonl 行。"""
    return [{"step": i, "action_executed": a, "collision": False, "displacement_m": d, "dyaw_deg": 0.0, "dispatch_lag_s": 0.16}
            for i, (a, d) in enumerate(spec)]


def samples_of(spec):
    return an.fwd_samples(steps(spec))


class RunPosition(unittest.TestCase):
    def test_run_position_counts_from_step_zero(self):
        """第 0 步不作样本，但计入连续前进：F F Y F F F F → 步 1 的 r=2（第 0 步也是前进），转向后的第一个前进 r=1，之后 2、3、4。"""
        got = samples_of([(F, 0.1), (F, 0.3), (Y, 0.0), (F, 0.08), (F, 0.3), (F, 0.25), (F, 0.23)])
        self.assertEqual([(i, p, r) for i, p, _d, r in got], [(1, F, 2), (3, Y, 1), (4, F, 2), (5, F, 3), (6, F, 4)])

    def test_collision_step_is_not_a_sample_but_still_counts_in_the_run(self):
        """碰撞的前进步没有可比的位移，不作样本；它仍是一个 FORWARD，后面的步 r 接着数。"""
        acts = steps([(F, 0.1), (F, 0.3), (F, 0.25), (F, 0.23)])
        acts[2]["collision"] = True
        got = an.fwd_samples(acts)
        self.assertEqual([(i, r) for i, _p, _d, r in got], [(1, 2), (3, 4)])

    def test_select_run_range(self):
        got = samples_of([(F, 0.1), (F, 0.3), (F, 0.25), (F, 0.23), (F, 0.22)])
        self.assertEqual(an.select(got, run_lo=2, run_hi=2), [0.3])
        self.assertEqual(an.select(got, run_lo=4), [0.23, 0.22])
        self.assertEqual(an.select(got, lo=3, run_lo=1), [0.23, 0.22])
        self.assertEqual(an.select(got, prev="fwd"), [0.3, 0.25, 0.23, 0.22])


def eps_from(seq_by_seed):
    """{seed: [(动作, 位移)]} → {seed: fwd_samples}"""
    return {s: samples_of(spec) for s, spec in seq_by_seed.items()}


class Decomposition(unittest.TestCase):
    def two_arms(self, k3_scale_r4=1.0):
        """K=1：每集 2 块 [转向, r=1, r=2, r=3, r≥4 × 2]；K=3：每集 2 块 [转向, r=1, r=2, r=3, r≥4 × 8]（巡航占比大得多）。
        层内位移相同（K=3 的 r≥4 可乘 k3_scale_r4）。r≥2 的构成：K=1 = 2/8、2/8、4/8，K=3 = 2/20、2/20、16/20。"""
        def ep(n4, scale):
            spec = [(F, 0.12)]
            for _ in range(2):
                spec += [(Y, 0.0), (F, 0.08), (F, 0.30), (F, 0.25)] + [(F, 0.23 * scale)] * n4
            return spec
        k1 = eps_from({s: ep(2, 1.0) for s in range(6)})
        k3 = eps_from({s: ep(8, k3_scale_r4) for s in range(6)})
        return k1, k3

    def test_pure_composition_effect_vanishes_after_standardization(self):
        """层内位移完全相同，只是 K=3 的巡航（r≥4）占比大：原始口径 K=3 更低，按 K=1 构成加权后差为 0，巡航口径差为 0。"""
        k1, k3 = self.two_arms()
        raw = an.boot_rel(k1, k3, prev="fwd")
        self.assertLess(raw[0], -0.01)                  # 构成不同就能造出可观的负差
        std = an.std_rel(k1, k3, lo=0, hi=10 ** 6)
        self.assertAlmostEqual(std[0], 0.0, places=9)
        cruise = an.boot_rel(k1, k3, prev="fwd", run_lo=an.CRUISE_MIN_RUN)
        self.assertAlmostEqual(cruise[0], 0.0, places=9)
        self.assertAlmostEqual(std[4][2], 4 / 8, places=9)     # K=1 的 r≥4 占比
        self.assertAlmostEqual(std[5][2], 16 / 20, places=9)   # K=3 的 r≥4 占比

    def test_real_within_stratum_difference_is_kept(self):
        """K=3 的巡航步真低 2%：标准化后的差 = 各层按 K=1 构成加权的差，巡航口径差 = -2%。"""
        k1, k3 = self.two_arms(k3_scale_r4=0.98)
        std = an.std_rel(k1, k3, lo=0, hi=10 ** 6)
        w = [2 / 8, 2 / 8, 4 / 8]
        mean_a = w[0] * 0.30 + w[1] * 0.25 + w[2] * 0.23
        mean_b = w[0] * 0.30 + w[1] * 0.25 + w[2] * 0.23 * 0.98
        self.assertAlmostEqual(std[0], mean_b / mean_a - 1.0, places=9)
        cruise = an.boot_rel(k1, k3, prev="fwd", run_lo=an.CRUISE_MIN_RUN)
        self.assertAlmostEqual(cruise[0], -0.02, places=9)

    def test_decompose_narrows_step_by_step(self):
        """decompose 给出 门口径 / 共同步段 / 标准化 / 巡航 四个数；这批数据里门口径与共同步段都包含全部步，所以相等。"""
        k1, k3 = self.two_arms()
        dec = an.decompose(k1, k3)
        self.assertEqual(set(dec), {"all", "common", "cruise", "cruise_all", "std"})
        self.assertLess(dec["all"][0], dec["std"][0])   # 标准化把负差拉回到 0
        self.assertAlmostEqual(dec["cruise_all"][0], 0.0, places=9)

    def test_std_rel_without_common_seeds_is_nan(self):
        k1 = eps_from({1: [(F, 0.1), (F, 0.3), (F, 0.25)]})
        k3 = eps_from({2: [(F, 0.1), (F, 0.3), (F, 0.25)]})
        std = an.std_rel(k1, k3)
        self.assertTrue(std[0] != std[0])
        self.assertEqual(std[3], 0)

    def test_stratum_missing_in_one_arm_is_dropped_from_both_weights(self):
        """K=3 没有 r=3 的样本：该层从两边的加权里去掉，不拿 0 去平均。"""
        k1 = eps_from({s: [(F, 0.12), (F, 0.30), (F, 0.25), (F, 0.23)] for s in range(3)})
        k3 = eps_from({s: [(F, 0.12), (F, 0.30)] for s in range(3)})
        std = an.std_rel(k1, k3, lo=0, hi=10 ** 6)
        self.assertAlmostEqual(std[0], 0.0, places=9)   # 只剩 r=2 一层，两边都是 0.30


class Stats(unittest.TestCase):
    def test_mann_whitney_known_values(self):
        u, z, p = pf.mann_whitney([1, 2, 3], [4, 5, 6])
        self.assertEqual(u, 0.0)
        self.assertAlmostEqual(z, -4.5 / (5.25 ** 0.5), places=9)
        self.assertAlmostEqual(p, 0.0495, places=3)
        u, z, p = pf.mann_whitney([4, 5, 6], [1, 2, 3])
        self.assertEqual(u, 9.0)
        self.assertAlmostEqual(p, 0.0495, places=3)

    def test_mann_whitney_identical_samples_have_p_one(self):
        _u, z, p = pf.mann_whitney([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(z, 0.0, places=9)
        self.assertAlmostEqual(p, 1.0, places=9)

    def test_mann_whitney_empty_is_nan(self):
        self.assertTrue(pf.mann_whitney([], [1.0])[2] != pf.mann_whitney([], [1.0])[2])

    def test_spearman(self):
        self.assertAlmostEqual(pf.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0, places=9)
        self.assertAlmostEqual(pf.spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0, places=9)
        self.assertAlmostEqual(pf.spearman([1, 2, 3, 4], [1, 3, 2, 4]), 0.8, places=9)
        r = pf.spearman([1, 2, 3], [5, 5, 5])
        self.assertTrue(r != r)                         # 一列不变：没有相关可言
        self.assertTrue(pf.spearman([1, 2], [1, 2]) != pf.spearman([1, 2], [1, 2]))   # 少于 3 个点

    def test_median_iqr(self):
        m, lo, hi = pf.median_iqr([1, 2, 3, 4, 5])
        self.assertEqual((m, lo, hi), (3, 2, 4))
        self.assertTrue(pf.median_iqr([])[0] != pf.median_iqr([])[0])


def tiny_scene():
    """两根柱子的小场景，足够 A* 走通；教师按 e1_mini / base 档。"""
    teacher.set_vehicle("e1_mini", "base")
    return [{"name": "p0", "x": 2.0, "y": 0.5, "z": 1.5, "size": [0.4, 0.4, 3.0]},
            {"name": "p1", "x": 4.0, "y": -0.5, "z": 1.5, "size": [0.4, 0.4, 3.0]}]


class CountingAndClasses(unittest.TestCase):
    def test_count_call_counts_expansions_and_restores_module(self):
        pillars = tiny_scene()
        orig = (teacher.heapq, teacher.astar_path, teacher.plan_action)
        (action, tie), counts = pf.count_call(teacher, teacher.plan_action_global_tiebreak, -0.5, 0.0, 0.0, pillars, "legacy")
        self.assertIsNone(tie)                          # legacy：不跑平局推演，margin 恒为 None
        self.assertEqual(counts["astar_calls"], 1)      # legacy 一步只调一次 A*
        self.assertEqual(counts["fallback_calls"], 0)
        self.assertGreater(counts["pops"], 10)
        self.assertGreaterEqual(counts["pushes"], counts["pops"])
        self.assertIsNotNone(counts["path_cells"])
        self.assertEqual((teacher.heapq, teacher.astar_path, teacher.plan_action), orig)   # 调用后三样都还原
        # 计数不改结果：与直接调用逐位相同
        self.assertEqual((action, tie), teacher.plan_action_global_tiebreak(-0.5, 0.0, 0.0, pillars, "legacy"))

    def test_count_call_detects_astar_failure_fallback(self):
        """起点在柱子正中：邻格全在膨胀区里，A* 第一次出堆就没有路 → 退回局部 plan_action。"""
        pillars = tiny_scene()
        _res, counts = pf.count_call(teacher, teacher.plan_action_global_tiebreak, 2.0, 0.5, 0.0, pillars, "legacy")
        self.assertEqual(counts["astar_calls"], 1)
        self.assertEqual(counts["fallback_calls"], 1)
        self.assertEqual(counts["pops"], 1)
        self.assertIsNone(counts["path_cells"])

    def test_count_call_restores_module_when_fn_raises(self):
        orig = (teacher.heapq, teacher.astar_path, teacher.plan_action)

        def boom():
            raise RuntimeError("x")
        with self.assertRaises(RuntimeError):
            pf.count_call(teacher, boom)
        self.assertEqual((teacher.heapq, teacher.astar_path, teacher.plan_action), orig)

    def test_classify_boundaries(self):
        base = {"astar_calls": 1, "pushes": 0, "path_cells": 5}
        self.assertEqual(pf.classify({**base, "fallback_calls": 1, "pops": 1}), "F")
        self.assertEqual(pf.classify({**base, "fallback_calls": 1, "pops": 5000}), "F")      # 退回局部规划优先
        self.assertEqual(pf.classify({**base, "fallback_calls": 0, "pops": pf.POPS_HEAVY}), "H")
        self.assertEqual(pf.classify({**base, "fallback_calls": 0, "pops": pf.POPS_HEAVY - 1}), "N")
        self.assertEqual(pf.classify({**base, "fallback_calls": 0, "pops": 10}, pops_heavy=5), "H")

    def test_late_rate_table(self):
        rows = [{"arm": "a", "late": True, "klass": "F"}, {"arm": "a", "late": False, "klass": "F"},
                {"arm": "a", "late": True, "klass": "H"}, {"arm": "a", "late": False, "klass": "N"},
                {"arm": "b", "late": False, "klass": "N"}, {"arm": "a", "late": False}]   # 没分类的行不计
        t = pf.late_rate_table(rows, ["a", "b"])
        self.assertEqual(t["a"], {"F": [1, 2], "H": [1, 1], "N": [0, 1]})
        self.assertEqual(t["b"], {"F": [0, 0], "H": [0, 0], "N": [0, 1]})

    def test_pick_controls_only_from_episodes_that_have_late_steps(self):
        rows = []
        for seed in (1, 2, 3):
            for step in range(10):
                rows.append({"arm": "k3", "seed": seed, "step": step, "action": Y if step % 3 == 0 else F,
                             "late": (seed == 1 and step in (1, 4))})
        rows.append({"arm": "other", "seed": 1, "step": 0, "action": F, "late": False})
        got = pf.pick_controls(rows, "k3", seed=7, n=5, early_max=3)
        self.assertEqual(len(got["A"]), 5)
        for g in got.values():
            for r in g:
                self.assertEqual((r["arm"], r["seed"], r["late"]), ("k3", 1, False))   # 只来自出现过迟到步的集、且不迟到
        self.assertEqual(sorted(r["step"] for r in got["B"]), [0, 2, 3])               # 步号 ≤ 3 的不迟到步
        self.assertTrue(all(r["action"] == Y for r in got["C"]))
        again = pf.pick_controls(rows, "k3", seed=7, n=5, early_max=3)
        self.assertEqual([(r["seed"], r["step"]) for r in got["A"]], [(r["seed"], r["step"]) for r in again["A"]])   # 同种子同抽样

    def test_pick_controls_c_is_empty_without_late_turn(self):
        rows = [{"arm": "k3", "seed": 1, "step": i, "action": Y if i % 2 else F, "late": i == 2} for i in range(8)]
        self.assertEqual(pf.pick_controls(rows, "k3")["C"], [])     # 迟到的都是 FORWARD：没有可比的 YAW 迟到步

    def test_verify_scene_checks_both_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            meta_path = os.path.join(tmp, "scene.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"world_sha256": "abc", "pillars": []}, f)
            with open(meta_path, "rb") as f:
                sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(pf.verify_scene(meta_path, "abc", sha), (True, True))
            self.assertEqual(pf.verify_scene(meta_path, "xyz", sha), (False, True))
            self.assertEqual(pf.verify_scene(meta_path, "abc", "0" * 64), (True, False))
            self.assertEqual(pf.verify_scene(meta_path, None, None), (False, False))   # manifest 没记哈希：算核对不过


if __name__ == "__main__":
    unittest.main(verbosity=2)
