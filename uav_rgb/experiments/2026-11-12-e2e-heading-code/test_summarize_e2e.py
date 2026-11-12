#!/usr/bin/env python3
"""summarize_e2e.py 的回归（卡 E2E-1）：配对计数与三个预注册判定的边界、结局分类、航向码统计、cond_code 一致性、
假评测桩（stub_eval_e2e.py）生成的 12 个臂上的端到端汇总（含迟到集重跑的三种结局、缺臂时不判）。纯 CPU，临时目录。

运行（仓库根）：.venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_summarize_e2e.py
"""
import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e2e_arms as ea  # noqa: E402
import summarize_e2e as se  # noqa: E402

STUB = os.path.join(HERE, "stub_eval_e2e.py")
SEEDS = ",".join(str(s) for s in se.FROZEN_SEEDS)


def run_stub(out, arm, seed, env=None, seeds=SEEDS, episodes=20, retry=False):
    name = f"{ea.NAME[arm]}-s{seed}" + ("-retry1" if retry else "")
    cmd = [sys.executable, STUB, "--policy", "student", "--ckpt", ea.ckpt_path(out, arm, seed), "--infer-bin", "x"]
    if ea.IS_CODE[arm]:
        cmd += ["--student-cond", "heading-code"]
    cmd += ["--task", "base", "--steps", "80", "--split", "test", "--seeds", seeds, "--corridor-half-width", "3.0", "--action-latency", "0.20",
            "--episodes", str(episodes), "--sim-timeout", "120.0", "--episode-timeout", "1500", "--dynamics", "cf_sitl", "--vehicle", "e1_mini",
            "--hold", "1.0", "--window-from", "observation", "--out", os.path.join(out, name)]
    full_env = {**os.environ, "STUB_SLEEP": "0", **(env or {})}
    return subprocess.run(cmd, env=full_env, capture_output=True, text=True)


def make_world(out, env=None):
    for a in ea.ARMS:
        for s in ea.SEEDS_TRAIN:
            run_stub(out, a, s, env)


def eval_json(out, name):
    with open(os.path.join(out, name, "eval.json"), encoding="utf-8") as f:
        return json.load(f)


class PairAndVerdictTests(unittest.TestCase):
    def test_pair_counts(self):
        x = {1: True, 2: True, 3: False, 4: False, 5: True}
        y = {1: True, 2: False, 3: True, 4: False, 6: True}      # 键 5、6 不是共同键
        pc = se.pair_counts(x, y)
        self.assertEqual((pc["n"], pc["b"], pc["c"], pc["both"], pc["neither"]), (4, 1, 1, 1, 1))
        self.assertEqual((pc["succ_x"], pc["succ_y"]), (2, 2))
        self.assertAlmostEqual(pc["p"], 1.0)

    def test_pair_counts_p_is_exact_mcnemar(self):
        x = {i: True for i in range(15)}
        y = {i: False for i in range(15)}
        pc = se.pair_counts(x, y)
        self.assertEqual((pc["b"], pc["c"]), (15, 0))
        self.assertAlmostEqual(pc["p"], 2 / 2 ** 15)

    def test_m1_boundaries(self):
        self.assertEqual(se.verdict_m1(15, 0.0099), se.PRED)
        self.assertEqual(se.verdict_m1(15, 0.01), se.MID)         # p < 0.01 才算，等于不算
        self.assertEqual(se.verdict_m1(30, 0.5), se.MID)          # 差够、p 不够
        self.assertEqual(se.verdict_m1(14, 1e-6), se.MID)
        self.assertEqual(se.verdict_m1(6, 1e-6), se.MID)          # 多不到 6 集才证伪，6 不算证伪
        self.assertEqual(se.verdict_m1(5, 1e-6), se.FALS)
        self.assertEqual(se.verdict_m1(0, 1.0), se.FALS)
        self.assertEqual(se.verdict_m1(-7, 1.0), se.FALS)

    def test_m2_prediction_and_falsification(self):
        self.assertEqual(se.verdict_m2(10, 5, 40, 40)[0], se.PRED)       # 恰为一半
        self.assertEqual(se.verdict_m2(10, 5, 40, 41)[0], se.PRED)
        self.assertEqual(se.verdict_m2(10, 6, 40, 45)[0], se.MID)        # 6 > 一半，且少 4 集，不证伪
        self.assertEqual(se.verdict_m2(10, 5, 40, 39)[0], se.MID)        # 越界够少但成功数更低：预测不成立，也没被证伪
        self.assertEqual(se.verdict_m2(10, 8, 40, 45)[0], se.FALS)       # 少 2 集 < 3
        self.assertEqual(se.verdict_m2(10, 7, 40, 45)[0], se.MID)        # 少恰好 3 集不算证伪
        self.assertEqual(se.verdict_m2(10, 12, 40, 45)[0], se.FALS)      # 越界更多

    def test_m2_conflict_when_a0_corridor_small(self):
        for a0, a1 in ((4, 2), (3, 1), (2, 0), (1, 0), (0, 0)):
            v, pred, fals = se.verdict_m2(a0, a1, 40, 40)
            self.assertTrue(pred and fals, (a0, a1))
            self.assertEqual(v, se.CONFLICT, (a0, a1))
        # 5 → 2：少 3 集，不证伪，预测成立
        v, pred, fals = se.verdict_m2(5, 2, 40, 40)
        self.assertEqual((v, pred, fals), (se.PRED, True, False))

    def test_m3_boundaries(self):
        self.assertEqual(se.verdict_m3(6), se.PRED)
        self.assertEqual(se.verdict_m3(5), se.MID)
        self.assertEqual(se.verdict_m3(1), se.MID)
        self.assertEqual(se.verdict_m3(0), se.FALS)
        self.assertEqual(se.verdict_m3(-4), se.FALS)

    def test_w22(self):
        self.assertTrue(se.w22_status(48, 3))
        self.assertFalse(se.w22_status(47, 3))
        self.assertFalse(se.w22_status(48, 4))
        self.assertTrue(se.w22_status(60, 0))


class ClassifyTests(unittest.TestCase):
    @staticmethod
    def traj(xs, ys=None, collide_at=None, src=None):
        n = len(xs)
        ys = ys or [0.0] * n
        actions = [{"collision": i == collide_at, **({"src": src} if (src and i == collide_at) else {})} for i in range(n)]
        oracle = [{"pose_gt": {"x": xs[i], "y": ys[i], "z": 2.0}} for i in range(n)]
        return actions, oracle

    def test_success(self):
        a, o = self.traj([1, 3, 5, 6.2])
        self.assertEqual(se.classify_row(True, False, a, o, 6.0), ("success", True))

    def test_collision_first(self):
        a, o = self.traj([1, 3, 3.5], ys=[0, 0, 3.4], collide_at=2)           # 同一步既碰撞又越界 → 撞柱
        self.assertEqual(se.classify_row(False, False, a, o, 6.0), ("collision", True))

    def test_corridor(self):
        a, o = self.traj([1, 2, 3, 4], ys=[0, 1, 3.2, 3.5])
        self.assertEqual(se.classify_row(False, False, a, o, 6.0), ("corridor", True))

    def test_not_reached_and_aborted(self):
        a, o = self.traj([1, 1.5, 2.0])
        self.assertEqual(se.classify_row(False, False, a, o, 6.0), ("not_reached", True))
        self.assertEqual(se.classify_row(False, True, a, o, 6.0), ("aborted", True))

    def test_stuck_is_not_collision(self):
        a, o = self.traj([1, 1.5, 2.0], collide_at=1, src="stuck")
        self.assertEqual(se.classify_row(False, False, a, o, 6.0)[0], "not_reached")

    def test_inconsistent_flags(self):
        a, o = self.traj([1, 3, 5, 6.2])
        self.assertEqual(se.classify_row(False, False, a, o, 6.0), ("other", False))          # eval 说失败，轨迹判据说通过
        a2, o2 = self.traj([1, 2, 3], collide_at=1)
        self.assertEqual(se.classify_row(True, False, a2, o2, 6.0), ("success", False))       # eval 说成功，轨迹判据说撞柱


class CodeStatsTests(unittest.TestCase):
    def test_hand_computed(self):
        # 执行动作 [L, L, F, R, R, R]：执行前的 n = 0,1,2,2,1,0 → 码 0,0,2,2,0,0
        st = se.code_stats([[2, 2, 1, 3, 3, 3]])
        self.assertEqual(st["steps"], 6)
        self.assertEqual(st["action_counts"], [0, 1, 2, 3])
        self.assertEqual((st["code_steps"], st["corrective"]), (2, 1))       # 码 2 的两步里第二步取了 YAW_RIGHT
        self.assertEqual((st["code0_steps"], st["turn_in_code0"]), (4, 4))
        self.assertEqual((st["episodes"], st["episodes_with_code"], st["max_abs_n"]), (1, 1, 2))

    def test_right_excess_is_corrected_by_left(self):
        # [R, R, F, L]：n = 0,-1,-2,-2 → 码 0,0,3,3；第 4 步（码 3）取 YAW_LEFT = 回转
        st = se.code_stats([[3, 3, 1, 2]])
        self.assertEqual((st["code_steps"], st["corrective"]), (2, 1))

    def test_merge(self):
        a = se.code_stats([[2, 2, 1, 3]])
        b = se.code_stats([[1, 1]])
        m = se.merge_code_stats([a, b])
        self.assertEqual((m["steps"], m["episodes"], m["episodes_with_code"]), (6, 2, 1))
        self.assertEqual(m["action_counts"], [0, 3, 2, 1])

    def test_cond_consistency(self):
        executed = [2, 2, 1, 3, 3]
        codes = se.nav_cond.heading_codes(executed)
        rows = [{"action_executed": a, "cond_code": c} for a, c in zip(executed, codes)]
        self.assertEqual(se.cond_consistency(rows), (5, 0, 0))
        rows[3]["cond_code"] = 0 if rows[3]["cond_code"] else 2
        self.assertEqual(se.cond_consistency(rows), (5, 1, 0))
        del rows[2]["cond_code"]
        self.assertEqual(se.cond_consistency(rows), (4, 1, 1))
        self.assertEqual(se.cond_consistency([{"action_executed": a} for a in executed]), (0, 0, 5))


class EndToEndWithStub(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = self._tmp.name

    def test_complete_world(self):
        make_world(self.out)
        text, v, data = se.build(self.out)
        self.assertTrue(v["complete"])
        # 独立计数：直接读各臂 eval.json 的 success 与 collision / corridor_violation 标记
        for a in ea.ARMS:
            succ = fail_col = fail_cor = fail_nr = 0
            for s in ea.SEEDS_TRAIN:
                for x in eval_json(self.out, f"{ea.NAME[a]}-s{s}")["per_episode"]:
                    if x["success"]:
                        succ += 1
                    elif x["collision"]:
                        fail_col += 1
                    elif x["corridor_violation"]:
                        fail_cor += 1
                    else:
                        fail_nr += 1
            self.assertEqual(v["success"][a], succ, a)
            c = v["classes"][a]
            self.assertEqual((c["success"], c["collision"], c["corridor"], c["not_reached"]), (succ, fail_col, fail_cor, fail_nr), a)
            self.assertEqual(c["n"], 60)
            self.assertEqual(v["flags"][a]["corridor_flag"], fail_cor)
        # 判定与纯函数一致
        m1, m3 = v["m1"], v["m3"]
        self.assertEqual(m1["verdict"], se.verdict_m1(m1["diff"], m1["p"]))
        self.assertEqual(m3["verdict"], se.verdict_m3(m3["diff"]))
        self.assertEqual(v["m2"]["verdict"], se.verdict_m2(v["classes"]["a0"]["corridor"], v["classes"]["a1"]["corridor"], v["m2"]["a0_success"], v["m2"]["a1_success"])[0])
        self.assertEqual(m1["n"], 60)
        for key in ("M1", "M2", "M3"):
            self.assertTrue(v["pairs"][key]["full"])
        for needle in ("预注册的三个比较", "2×2", "M1 A2 对 A3", "M2 A1 对 A0", "M3 A2 对 A0", "只报告（不设判决）", "采集行 cond_code"):
            self.assertIn(needle, text)
        # cond_code：A1、A2 的采集行都带码且与重算相同，A0、A3 没有码
        for a in ea.ARMS:
            w = bad = 0
            for s in ea.SEEDS_TRAIN:
                for rec in data[(a, s)]["eps"].values():
                    w, bad = w + rec["cond"][0], bad + rec["cond"][1]
            if ea.IS_CODE[a]:
                self.assertGreater(w, 0)
            else:
                self.assertEqual(w, 0)
            self.assertEqual(bad, 0)
        # 写文件
        se.write_outputs(self.out)
        with open(os.path.join(self.out, "episodes.csv"), encoding="utf-8") as f:
            rows = list(csv.reader(f))
        self.assertEqual(len(rows), 1 + 240)
        with open(os.path.join(self.out, "verdicts.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["m1"]["verdict"], m1["verdict"])

    def test_prediction_outcome_under_designed_rates(self):
        # A2 对 A3 差距大、A0 越界多而 A1 越界少：三个判定都应落在预期的一侧
        make_world(self.out, {"STUB_P": "0.50,0.80,0.95,0.20", "STUB_FAIL_MIX": "0.1,0.8,0.1"})
        v = se.build(self.out)[1]
        self.assertGreaterEqual(v["m1"]["diff"], 15)
        self.assertEqual(v["m1"]["verdict"], se.PRED)
        self.assertEqual(v["m3"]["verdict"], se.PRED)

    def test_missing_arm_is_not_judged(self):
        make_world(self.out)
        import shutil
        shutil.rmtree(os.path.join(self.out, f"{ea.NAME['a3']}-s2"))
        text, v, _ = se.build(self.out)
        self.assertFalse(v["complete"])
        for k in ("m1", "m2", "m3"):
            self.assertEqual(v[k]["verdict"], se.UNDONE)
        self.assertIn("尚未全部评完", text)
        self.assertLess(v["pairs"]["M1"]["n"], 60)

    def test_late_dispatch_retry_variants(self):
        late = {"a1-code-demo-s0": [113, 172], "a3-img-dagger-s1": [1]}
        env = {"STUB_LATE": json.dumps(late)}
        make_world(self.out, env)
        self.assertEqual(subprocess.run([sys.executable, os.path.join(HERE, "summarize_e2e.py"), "--late-seeds", os.path.join(self.out, "a1-code-demo-s0")],
                                        capture_output=True, text=True).stdout.strip(), "113,172")
        # 没有重跑：臂不完整，含迟到步的集按没成功算
        _, v, data = se.build(self.out)
        self.assertFalse(v["complete"])
        self.assertEqual(data[("a1", 0)]["eps"][113]["retry"], "missing")
        self.assertFalse(data[("a1", 0)]["eps"][113]["success"])
        self.assertEqual(data[("a1", 0)]["eps"][113]["klass"], "late")
        # 重跑干净：以重跑结果为准
        run_stub(self.out, "a1", 0, env, seeds="113,172", episodes=2, retry=True)
        run_stub(self.out, "a3", 1, env, seeds="1", episodes=1, retry=True)
        _, v, data = se.build(self.out)
        self.assertTrue(v["complete"])
        rec = data[("a1", 0)]["eps"][113]
        self.assertEqual((rec["final"], rec["retry"], rec["first_late"]), ("retry", "ok", True))
        retry_ok = {x["seed"]: x["success"] for x in eval_json(self.out, "a1-code-demo-s0-retry1")["per_episode"]}
        self.assertEqual(rec["success"], retry_ok[113])
        self.assertFalse(rec["first_success"])                                   # 首次记录里迟到集按没成功
        self.assertEqual(v["classes"]["a1"]["late"], 0)
        # 重跑仍迟到：按没成功算，归「迟到」
        import shutil
        shutil.rmtree(os.path.join(self.out, "a1-code-demo-s0-retry1"))
        run_stub(self.out, "a1", 0, {**env, "STUB_RETRY_LATE": "1"}, seeds="113,172", episodes=2, retry=True)
        _, v, data = se.build(self.out)
        rec = data[("a1", 0)]["eps"][113]
        self.assertEqual((rec["final"], rec["retry"], rec["success"], rec["klass"]), ("none", "late", False, "late"))
        self.assertEqual(v["classes"]["a1"]["late"], 2)
        self.assertTrue(v["complete"])

    def test_cond_flag_when_image_arm_has_cond_code(self):
        make_world(self.out)
        # 把 A0 种子 0 的一集采集行改成带 cond_code：汇总应标 ⚠
        path = os.path.join(self.out, "a0-img-demo-s0", "episodes", "seed_1", "actions.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        for r in rows:
            r["cond_code"] = 0
        with open(path, "w", encoding="utf-8") as f:
            f.write("".join(json.dumps(r) + "\n" for r in rows))
        text, _v, _d = se.build(self.out)
        self.assertIn("图像臂不该有 cond_code", text)

    def test_guard_reference_matches_card(self):
        ref = se.guard_reference()
        if ref is None:
            self.skipTest("第二轮的三个 K=1 臂不在本机")
        self.assertEqual((ref["success"], ref["n"]), (53, 60))                  # 卡里写的参照 53/60


class CgroupTests(unittest.TestCase):
    """公共规则 §4（2026-11-12 01:40 起）：汇总里每个臂给一行：平均核数、被限流时长占墙钟的比例、迟到步数。"""

    def test_cg_row_hand_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            se.r2.OUT = tmp
            with open(os.path.join(tmp, "a1-code-demo-s0.cg"), "w", encoding="utf-8") as f:
                f.write("start 100.0 1000000 10 500000\nend 200.0 401000000 14 2500000\n")      # 墙钟 100 s，CPU 400 s → 4.0 核；限流 +4 次、+2.0 s → 2.00%
            self.assertEqual(se.cg_row("a1-code-demo-s0", os.path.join(tmp, "a1-code-demo-s0")), "| a1-code-demo-s0 | 100 | 4.0 | 4 ⚠ | 2.00 | 2.00% | - |")
            with open(os.path.join(tmp, "a0-img-demo-s0.cg"), "w", encoding="utf-8") as f:
                f.write("start 0.0 0 7 100\nend 50.0 150000000 7 100\n")                          # 3.0 核，没有新增限流
            self.assertEqual(se.cg_row("a0", os.path.join(tmp, "a0-img-demo-s0")), "| a0 | 50 | 3.0 | 0 | 0.00 | 0.00% | - |")
            self.assertEqual(se.cg_row("a2", os.path.join(tmp, "a2-code-dagger-s0")), "| a2 | - | - | - | - | - | - |")      # 没有 .cg
            with open(os.path.join(tmp, "a3-img-dagger-s0.cg"), "w", encoding="utf-8") as f:
                f.write("start 0.0 0 0 0\n")                                                       # 臂还在跑：只有 start
            self.assertEqual(se.cg_row("a3", os.path.join(tmp, "a3-img-dagger-s0")), "| a3 | - | - | - | - | - | - |")

    def test_summary_has_cgroup_table_and_start_fields(self):
        with tempfile.TemporaryDirectory() as out:
            make_world(out)
            for a in ea.ARMS:
                for s in ea.SEEDS_TRAIN:
                    with open(os.path.join(out, f"{ea.NAME[a]}-s{s}.cg"), "w", encoding="utf-8") as f:
                        f.write("start 0.0 0 0 0\nend 100.0 500000000 3 1000000\n")               # 5.0 核，限流 +3 次、1.0 s → 1.00%
            with open(os.path.join(out, "start.txt"), "w", encoding="utf-8") as f:
                f.write("started_at=2026-11-12 02:00:00\nidle_cores=9.50\nidle_max_cores=8\nidle_limit_s=600\nidle_forced=1\ngz_count=0\n")
            text, _v, _d = se.build(out)
            self.assertIn("被限流时长占墙钟", text)
            self.assertIn("| a0-img-demo-s0 | 100 | 5.0 | 3 ⚠ | 1.00 | 1.00% |", text)
            self.assertIn("idle_forced=1", text)
            self.assertIn("idle_limit_s=600", text)
            self.assertIn("idle_max_cores=8", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
