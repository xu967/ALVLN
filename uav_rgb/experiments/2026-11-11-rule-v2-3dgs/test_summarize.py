#!/usr/bin/env python3
"""summarize.py 的单测：预注册判决的边界（Δ = 0.10 / 0.05、碰撞率 0.10 / 0.15、撞在前 0.08、p = 0.05、室外 0.93 与 b − c = 5、0.80）、
配对计数的方向、路线层面的符号检验；再用桩驱动造一份完整的输出目录，把汇总的数字与「直接读 metrics.json / rule.jsonl 重算」的数字逐项对拍
（失败形态、V2-stuck 收尾列表、出生即碰撞、转向步占比、K1 配对、数据不全的标记）。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_summarize.py
"""
import contextlib
import glob
import io
import json
import os
import sys
import tempfile
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_config as bc  # noqa: E402
import dry_run_driver as drd  # noqa: E402
import summarize as sm  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def read_lines(path):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def run_main(args):
    """跑 summarize.main，吞掉它打印的整份 summary.md（测试输出里不要这堵墙）。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return sm.main(args)


def pc(n, d, p, ka=None):
    """配对结果：n 对，到点数之差 d（ka − kb），p。"""
    ka = (n // 2 + d) if ka is None else ka
    return {"a": "a", "b": "b", "pairs": n, "ka": ka, "kb": ka - d, "b_count": max(d, 0), "c_count": max(-d, 0), "p": p}


def main_arm(n, k_coll, k_cf):
    return {"n_req": n, "k_coll": k_coll, "k_cf": k_cf}


class TestPrimaryVerdict(unittest.TestCase):
    def test_exactly_on_every_predicted_boundary_is_predicted(self):
        j = sm.judge_primary(pc(100, 10, 0.04), main_arm(100, 10, 8))        # Δ = 0.10、碰撞 0.10、撞在前 0.08：都是「不低于 / 不高于」，边界算满足
        self.assertEqual(j["verdict"], "达到预测", j)
        self.assertTrue(j["delta_ge_pred"] and j["coll_le_pred"] and j["cf_le_pred"] and j["p_lt_alpha"])

    def test_each_predicted_condition_alone_breaks_the_prediction(self):
        cases = {"delta 0.09": (pc(100, 9, 0.01), main_arm(100, 5, 2)), "p = 0.05": (pc(100, 12, 0.05), main_arm(100, 5, 2)),
                 "coll 0.11": (pc(100, 12, 0.01), main_arm(100, 11, 2)), "cf 0.09": (pc(100, 12, 0.01), main_arm(100, 5, 9))}
        for name, (c, m) in cases.items():
            j = sm.judge_primary(c, m)
            self.assertEqual(j["verdict"], "介于预测与证伪之间", (name, j))

    def test_falsification_thresholds_are_strict(self):
        self.assertEqual(sm.judge_primary(pc(100, 5, 0.3), main_arm(100, 5, 2))["verdict"], "介于预测与证伪之间", "Δ 恰好 0.05：不是「低于 0.05」")
        self.assertEqual(sm.judge_primary(pc(100, 4, 0.3), main_arm(100, 5, 2))["verdict"], "证伪", "Δ 0.04 < 0.05")
        self.assertEqual(sm.judge_primary(pc(100, 12, 0.01), main_arm(100, 15, 2))["verdict"], "介于预测与证伪之间", "碰撞恰好 0.15：不是「高于 0.15」")
        self.assertEqual(sm.judge_primary(pc(100, 12, 0.01), main_arm(100, 16, 2))["verdict"], "证伪", "碰撞 0.16 > 0.15")

    def test_falsification_wins_over_prediction(self):
        j = sm.judge_primary(pc(100, 15, 0.001), main_arm(100, 16, 2))
        self.assertTrue(j["delta_ge_pred"] and j["p_lt_alpha"], "Δ 与 p 都达到预测的条件，碰撞率却高于 0.15")
        self.assertTrue(j["falsified"])
        self.assertFalse(j["predicted"])
        self.assertEqual(j["verdict"], "证伪")

    def test_negative_delta_and_no_data(self):
        self.assertEqual(sm.judge_primary(pc(100, -3, 0.5), main_arm(100, 0, 0))["verdict"], "证伪")
        self.assertEqual(sm.judge_primary(pc(0, 0, 1.0), main_arm(0, 0, 0))["verdict"], "无数据")

    def test_exact_fractions_not_floats(self):
        j = sm.judge_primary(pc(30, 3, 0.01), main_arm(30, 3, 2))             # 3/30 = 0.1、3/30 碰撞 = 0.1、2/30 撞在前 < 0.08
        self.assertEqual(j["verdict"], "达到预测")
        self.assertEqual(Fraction(3, 30), sm.PRE["delta_pred"])
        j2 = sm.judge_primary(pc(25, 3, 0.01), main_arm(25, 2, 2))            # 撞在前 2/25 = 0.08 恰好
        self.assertTrue(j2["cf_le_pred"])
        j3 = sm.judge_primary(pc(125, 13, 0.01), main_arm(125, 12, 10))       # 10/125 = 0.08 恰好；13/125 = 0.104 ≥ 0.10
        self.assertTrue(j3["cf_le_pred"] and j3["delta_ge_pred"] and j3["verdict"] == "达到预测")


class TestSecondaryAndOutdoor(unittest.TestCase):
    def test_secondary_reports_080_boundary(self):
        self.assertTrue(sm.judge_secondary(pc(120, 10, 0.1), {"n_req": 120, "k_stop": 96})["sr_ge_080"])
        self.assertFalse(sm.judge_secondary(pc(120, 10, 0.1), {"n_req": 120, "k_stop": 95})["sr_ge_080"])
        self.assertEqual(sm.judge_secondary(pc(120, 10, 0.1), {"n_req": 120, "k_stop": 96})["verdict"], "报告", "次终点只报告，不下判决")

    def test_outdoor_noninferiority_boundaries(self):
        def run(k_stop, b, c, k_coll, n=120):
            return sm.judge_outdoor({"n_req": n, "k_stop": k_stop, "k_coll": k_coll}, {"pairs": n, "b_count": b, "c_count": c, "p": 1.0}, 0)
        self.assertEqual(run(112, 6, 1, 12)["verdict"], "不劣", "SR 112/120 = 0.9333 ≥ 0.93，b − c = 5，碰撞 12/120 = 0.10：全在边界内")
        self.assertEqual(run(111, 6, 1, 12)["verdict"], "不满足不劣", "SR 0.925 < 0.93")
        self.assertEqual(run(112, 7, 1, 12)["verdict"], "不满足不劣", "b − c = 6 > 5")
        self.assertEqual(run(112, 6, 1, 13)["verdict"], "不满足不劣", "碰撞 13/120 > 0.10")
        self.assertTrue(run(112, 6, 1, 12)["sr_ge"] and run(112, 6, 1, 12)["bc_le"] and run(112, 6, 1, 12)["coll_le"])
        self.assertEqual(run(100, 0, 0, 0, n=100)["sr"], 1.0)
        self.assertTrue(sm.judge_outdoor({"n_req": 100, "k_stop": 93, "k_coll": 0}, {"pairs": 100, "b_count": 0, "c_count": 0, "p": 1.0}, 0)["sr_ge"], "93/100 = 0.93 恰好")

    def test_outdoor_missing_k1_keys_is_not_judged(self):
        j = sm.judge_outdoor({"n_req": 120, "k_stop": 120, "k_coll": 0}, {"pairs": 100, "b_count": 0, "c_count": 0, "p": 1.0}, 20)
        self.assertEqual(j["verdict"], "K1 缺键，不判决")
        self.assertEqual(sm.judge_outdoor({"n_req": 0, "k_stop": 0, "k_coll": 0}, {"pairs": 0, "b_count": 0, "c_count": 0, "p": 1.0}, 0)["verdict"], "无数据")


def ep(success, **kw):
    return {"success_stop": success, **kw}


class TestPairing(unittest.TestCase):
    def test_counts_and_direction(self):
        idx = {}
        for i, (a, b) in enumerate([(1, 0), (1, 0), (1, 1), (0, 1), (0, 0), (1, 0)]):
            idx[("A", "s", i // 3, i)] = ep(bool(a))
            idx[("B", "s", i // 3, i)] = ep(bool(b))
        idx[("A", "s", 9, 99)] = ep(True)             # B 没有的键：不进配对
        p = sm.paired2(idx, "A", "B")
        self.assertEqual((p["pairs"], p["ka"], p["kb"], p["b_count"], p["c_count"]), (6, 4, 2, 3, 1), "b = A 成 B 败 3、c = A 败 B 成 1")
        self.assertAlmostEqual(p["p"], 0.625)         # 精确二项：2·P(X ≤ 1 | n = 4) = 2·5/16
        q = sm.paired2(idx, "B", "A")
        self.assertEqual((q["b_count"], q["c_count"]), (1, 3), "换臂后 b / c 对调")

    def test_route_level_sign_counts_each_route_once(self):
        idx = {}
        for route, (sa, sb) in enumerate([(3, 1), (2, 2), (0, 3), (3, 0)]):
            for k in range(3):
                idx[("A", "s", route, k)] = ep(k < sa)
                idx[("B", "s", route, k)] = ep(k < sb)
        rs = sm.route_sign(idx, "A", "B")
        self.assertEqual((rs["routes"], rs["win"], rs["loss"], rs["tie"]), (4, 2, 1, 1))


# ---------------------------------------------------------------- 用桩驱动造一份完整的输出目录，把汇总与直接重算的数字对拍
TABLE = """v2a|0.25:1.2|lexi
v2b|0.25:1.2,0.25:0.8|lexi
v2c|0.25:1.2,0.25:0.8|near"""
SELECTED = {"variant": "v2c", "depth_variant": "v2c", "gain_over_v1": 7, "gate_bypassed": False}


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_out(tmp):
    labels = ["<6 m", "6-10 m", ">=10 m"]
    routes = {"params": {"bucket_labels": labels},
              "scenes": {s: [{"route": r, "name": f"{s}-r{r}", "bucket": labels[r % 3], "astar_path_m": 4.0 + r} for r in range(12)] for s in ("sA", "sB")}}
    for n in ("dev", "holdout", "formal40"):
        jdump(routes, os.path.join(tmp, f"routes_{n}.json"))
    e = {"REPO": "/repo", "SCENES_DIR": "/scenes", "HOLD": "1.0", "STEP_FRAC": "0.5", "STEP_PAD": "15", "GUARD_DEG": "45.0", "DROP_TRACE": "true", "CF_SITL_PROC": "/cf",
         "YAW_STEP": "31.7", "V1_R_SAFE": "0.25", "V1_D_LOOK": "1.2", "VARIANTS": TABLE, "INDOOR_SCENES": "sA sB", "OUTDOOR_SCENES": "o1 o2", "ROUTES_OUTDOOR": "0 1 2",
         "ROUTES_DEV": os.path.join(tmp, "routes_dev.json"), "ROUTES_HOLDOUT": os.path.join(tmp, "routes_holdout.json"), "ROUTES_FORMAL40": os.path.join(tmp, "routes_formal40.json"),
         "K_DEV": "1", "K_HOLDOUT": "3", "K_FORMAL40": "3", "K_OUTDOOR": "5", "SEED0_DEV": "9000", "SEED0_HOLDOUT": "5000", "SEED0_FORMAL40": "8000", "SEED0_OUTDOOR": "7000",
         "JITTER_INDOOR": "0.2", "JITTER_OUTDOOR": "0.5", "LOOKAHEAD_INDOOR": "1.5", "LOOKAHEAD_OUTDOOR": "3.0", "STEP_CAP_INDOOR": "100"}
    out, k1 = os.path.join(tmp, "out"), os.path.join(tmp, "k1")
    jdump(SELECTED, os.path.join(out, "selected.json"))
    for cls in ("holdout", "formal40", "outdoor"):
        cfg = bc.build(cls, e, SELECTED)
        jdump(cfg, os.path.join(out, cls, "config.json"))
        for s in cfg["scenes"]:
            drd.run_scene(os.path.join(out, cls), cfg, s)
        if cls == "outdoor":                                                   # 假 K1：只有 v1 臂，同键，结局只由 (场景, 路线, seed) 决定
            kcfg = {**cfg, "arms": [bc.v1_arm(e)], "class": "k1"}
            jdump(kcfg, os.path.join(k1, "formal", "outdoor", "config.json"))
            for s in cfg["scenes"]:
                drd.run_scene(os.path.join(k1, "formal", "outdoor"), kcfg, s)
    return out, k1


def raw_episodes(root, cls, arm):
    rows = []
    for p in sorted(glob.glob(os.path.join(root, cls, arm, "*", "r*", "ep_*", "metrics.json"))):
        m = jload(p)
        m["_dir"] = os.path.dirname(p)
        rows.append(m)
    return rows


def last_entry(m):
    return json.loads(read_lines(os.path.join(m["_dir"], "rule.jsonl"))[-1])["entry"]


def plant_early_collisions(out):
    """把留出主臂里三个失败集改成早期撞：collision_step = 0（出生即撞）、1、4（前 4 步内碰撞的口径是步号 <= 3），供两个口径的核对用。"""
    eps = [m for m in raw_episodes(out, "holdout", "rule2-occ+route") if not m["success_stop"]][:3]
    assert len(eps) == 3, "桩的主臂至少要有 3 个失败集"
    for m, step in zip(eps, (0, 1, 4)):
        path = os.path.join(m["_dir"], "metrics.json")
        d = jload(path)
        d.update({"collision": True, "collide_first": True, "collision_step": step})
        jdump(d, path)


class TestSummarizeOnStubData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out, cls.k1 = make_out(cls._tmp.name)
        plant_early_collisions(cls.out)
        cls.rc = run_main([cls.out, "--k1-root", cls.k1])
        cls.summary = jload(os.path.join(cls.out, "summary.json"))
        cls.md = read_text(os.path.join(cls.out, "summary.md"))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_outputs_exist_and_data_is_complete(self):
        self.assertEqual(self.rc, 0)
        for n in ("summary.md", "summary.json", "verdict.txt"):
            self.assertTrue(os.path.exists(os.path.join(self.out, n)), n)
        self.assertTrue(self.summary["complete"])
        self.assertNotIn("数据不全", self.md)

    def test_primary_numbers_match_a_direct_recount(self):
        main, v1 = raw_episodes(self.out, "holdout", "rule2-occ+route"), raw_episodes(self.out, "holdout", "rule-occ+route")
        key = lambda m: (m["scene"], m["route"], m["seed"])  # noqa: E731
        by1 = {key(m): m for m in v1}
        ka = sum(m["success_stop"] for m in main)
        kb = sum(by1[key(m)]["success_stop"] for m in main)
        b = sum(m["success_stop"] and not by1[key(m)]["success_stop"] for m in main)
        c = sum((not m["success_stop"]) and by1[key(m)]["success_stop"] for m in main)
        j = self.summary["verdict"]["primary"]
        self.assertEqual((len(main), j["pairs"], j["delta_num"]), (72, 72, ka - kb))
        self.assertGreater(ka - kb, 0, "桩的 v2 比 v1 好")
        self.assertEqual(c, 0, "桩里 v2 的成功集包含 v1 的成功集")
        coll, cf = sum(m["collision"] for m in main), sum(m["collide_first"] for m in main)
        self.assertAlmostEqual(j["coll"], coll / 72)
        self.assertAlmostEqual(j["cf"], cf / 72)
        want = "证伪" if (Fraction(ka - kb, 72) < Fraction(1, 20) or Fraction(coll, 72) > Fraction(3, 20)) else (
            "达到预测" if (Fraction(ka - kb, 72) >= Fraction(1, 10) and j["p"] < 0.05 and Fraction(coll, 72) <= Fraction(1, 10) and Fraction(cf, 72) <= Fraction(2, 25)) else "介于预测与证伪之间")
        self.assertEqual(j["verdict"], want)
        self.assertIn(f"**判决：{want}**", self.md)
        self.assertIn(f"主终点{want}", read_text(os.path.join(self.out, "verdict.txt")))

    def test_secondary_and_outdoor_use_the_right_pairing(self):
        sec = self.summary["verdict"]["secondary"]
        main = raw_episodes(self.out, "formal40", "rule2-occ+route")
        self.assertEqual(sec["pairs"], 72)
        self.assertEqual(sec["sr_ge_080"], Fraction(sum(m["success_stop"] for m in main), len(main)) >= Fraction(4, 5))
        out_main = raw_episodes(self.out, "outdoor", "rule2-occ+route")
        k1 = {}
        for p in sorted(glob.glob(os.path.join(self.k1, "formal", "outdoor", "rule-occ+route", "*", "r*", "ep_*", "metrics.json"))):
            m = jload(p)
            k1[(m["scene"], m["route"], m["seed"])] = m
        b = sum(k1[(m["scene"], m["route"], m["seed"])]["success_stop"] and not m["success_stop"] for m in out_main)
        c = sum((not k1[(m["scene"], m["route"], m["seed"])]["success_stop"]) and m["success_stop"] for m in out_main)
        o = self.summary["verdict"]["outdoor"]
        self.assertEqual((o["b_v1_only"], o["c_v2_only"], o["pairs"]), (b, c, 30))
        self.assertEqual(len(out_main), 30)

    def test_stuck_endings_spawn_collisions_and_turn_share(self):
        res = self.summary["classes"]["holdout"]["facts"]["rule2-occ+route"]
        main = raw_episodes(self.out, "holdout", "rule2-occ+route")
        want_stuck = sorted((m["scene"], m["route"], m["seed"]) for m in main if not m["success_stop"] and last_entry(m) == "V2-stuck")
        self.assertEqual(sorted((x["scene"], x["route"], x["seed"]) for x in res["stuck_ending"]), want_stuck)
        self.assertGreater(len(want_stuck), 0, "桩里有以 V2-stuck 收尾的失败集，列表不能是空的")
        for x in res["stuck_ending"]:
            self.assertAlmostEqual(x["min_clear_last20"], 0.15, places=6, msg="桩给 V2-stuck 的末 20 步间隙是 0.15")
            self.assertEqual(len(x["last_pose"]), 3)
        self.assertEqual(res["success_ending_stuck"], sum(1 for m in main if m["success_stop"] and last_entry(m) == "V2-stuck"))
        want_spawn = sorted([m["scene"], m["route"], m["seed"]] for m in main if m["collision_step"] == 0)
        self.assertEqual(res["spawn_collision"], want_spawn)
        self.assertEqual(len(want_spawn), 1, "夹具里种了一个出生即撞（step 0）；step 1 与 step 4 的不算")
        first4 = sum(1 for m in main if m["collision_step"] is not None and m["collision_step"] <= 3)
        self.assertEqual(res["collision_first4"], first4)
        self.assertGreaterEqual(first4, 2, "step 0 与 step 1 的算前 4 步内碰撞，step 4 的不算")
        self.assertEqual(sum(1 for m in main if m["collision_step"] == 4), 1, "step 4 的那一集在夹具里确实存在")
        turn = sum(sum(1 for ln in read_lines(os.path.join(m["_dir"], "actions.jsonl")) if json.loads(ln)["action_executed"] in (2, 3)) for m in main)
        acts = sum(m["steps"] for m in main)
        self.assertAlmostEqual(res["turn_share_pooled"], turn / acts)
        self.assertAlmostEqual(res["steps_mean"], acts / len(main))
        self.assertEqual(self.summary["classes"]["holdout"]["facts"]["rule-occ+route"]["stuck_ending"], [], "v1 臂没有 V2-stuck 条目")

    def test_entry_shares_include_v2_stuck_and_levels(self):
        e = self.summary["classes"]["holdout"]["entries"]["rule2-occ+route"]
        main = raw_episodes(self.out, "holdout", "rule2-occ+route")
        self.assertEqual(e["n_steps"], sum(m["steps"] for m in main))
        self.assertEqual(e["whole"].get("V2-stuck", 0), sum(m["rule"]["entries"].get("V2-stuck", 0) for m in main))
        self.assertGreater(e["whole"].get("V2-fwd1", 0), 0, "两级的变体 v2c：第 1 级的前进也有条目")
        self.assertIn("V2-stuck", self.md)
        self.assertIn("V2-fwd1", self.md)

    def test_buckets_are_listed_with_the_long_one_last(self):
        i_short, i_long = self.md.index("| <6 m（"), self.md.index("| >=10 m（")
        self.assertLess(i_short, i_long)
        self.assertIn("出生即碰撞", self.md)

    def test_incomplete_data_is_flagged_and_not_final(self):
        victim = sorted(glob.glob(os.path.join(self.out, "holdout", "rule2-occ+route", "*", "r*", "done.json")))[0]
        saved = jload(victim)
        os.remove(victim)
        try:
            self.assertEqual(run_main([self.out, "--k1-root", self.k1]), 0)
            s = jload(os.path.join(self.out, "summary.json"))
            self.assertFalse(s["complete"])
            self.assertEqual(len(s["classes"]["holdout"]["missing_routes"]), 1)
            self.assertTrue(read_text(os.path.join(self.out, "verdict.txt")).startswith("数据不全；"))
        finally:
            jdump(saved, victim)
            run_main([self.out, "--k1-root", self.k1])

    def test_missing_k1_root_gives_no_outdoor_verdict(self):
        run_main([self.out])
        s = jload(os.path.join(self.out, "summary.json"))
        self.assertEqual(s["verdict"]["outdoor"]["verdict"], "K1 缺键，不判决")
        run_main([self.out, "--k1-root", self.k1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
