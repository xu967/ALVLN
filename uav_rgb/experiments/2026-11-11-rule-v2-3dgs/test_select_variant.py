#!/usr/bin/env python3
"""select_variant.py 与 variants.py 的单测：变体表解析与校验、自动选取规则（撞在前出局、并列取靠前、关口）、完整性检查、
selected.json 的内容与冻结（拒绝改写）。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_select_variant.py
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_variant as sv  # noqa: E402
import variants as vr  # noqa: E402

TABLE = """v2a|0.25:1.2|lexi
v2b|0.25:1.2,0.25:0.8|lexi
v2c|0.25:1.2,0.25:0.8|near
v2d|0.25:1.2,0.25:0.8,0.20:0.55*|near
v2e|0.15:1.2|lexi
v2f|0.15:1.2,0.15:0.8,0.12:0.55*|near"""
NAMES = ["v2a", "v2b", "v2c", "v2d", "v2e", "v2f"]
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP


def stats(k_stop, k_cf=0, n=40):
    return {"n": n, "k_stop": k_stop, "k_collide_first": k_cf, "k_collision": k_cf, "k_timeout": n - k_stop - k_cf, "by_origin": {}}


def table_of(v1, **variants):
    t = {"v1": stats(v1)}
    for n in NAMES:
        t[n] = stats(*variants.get(n, (0, 0)))
    return t


KW = dict(max_collide_first=3, tie_slack=1, gate_min_gain=3)


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class TestVariants(unittest.TestCase):
    def test_parse_card_table(self):
        vs = vr.parse_variants(TABLE)
        self.assertEqual([v["name"] for v in vs], NAMES)
        self.assertEqual(vs[3]["levels"], [[0.25, 1.2, False], [0.25, 0.8, False], [0.20, 0.55, True]])
        self.assertEqual((vs[3]["order"], vs[2]["order"], vs[0]["order"]), ("near", "near", "lexi"))
        self.assertEqual(vs[5]["levels"], [[0.15, 1.2, False], [0.15, 0.8, False], [0.12, 0.55, True]])
        self.assertEqual(vr.rule_params_of(vs[1]), {"levels": [[0.25, 1.2, False], [0.25, 0.8, False]], "order": "lexi"})

    def test_blank_lines_ignored(self):
        self.assertEqual(len(vr.parse_variants("\n" + TABLE + "\n\n")), 6)

    def test_rejects_relaxing_levels_and_bad_rows(self):
        for bad in ("x|0.25:0.8,0.25:1.2|lexi", "x|0.25:1.2,0.30:0.8|lexi", "x|0.25:1.2|first", "x|0.25|lexi", "x|0.25:1.2", "x|a:b|lexi"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                vr.parse_variants(bad)

    def test_rejects_duplicates_and_v1_and_empty(self):
        for bad in ("a|0.25:1.2|lexi\na|0.25:1.2|near", "v1|0.25:1.2|lexi", "\n"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                vr.parse_variants(bad)

    def test_depth_map(self):
        vs = vr.parse_variants(TABLE)
        m = vr.parse_depth_map("v2e=v2a v2f=v2d", vs)
        self.assertEqual([m[n] for n in NAMES], ["v2a", "v2b", "v2c", "v2d", "v2a", "v2d"])
        with self.assertRaises(ValueError):
            vr.parse_depth_map("v2e=v9", vs)


class TestChoose(unittest.TestCase):
    def pick(self, v1, **variants):
        return sv.choose(table_of(v1, **variants), NAMES, **KW)

    def test_highest_k_stop_wins_when_clear(self):
        d = self.pick(20, v2a=(24, 0), v2b=(26, 1), v2c=(30, 2), v2d=(27, 0))
        self.assertEqual(d["selected"], "v2c")
        self.assertEqual((d["best"], d["gain_over_v1"], d["gate_passed"]), (30, 10, True))

    def test_within_one_episode_counts_as_tie_and_earlier_wins(self):
        d = self.pick(20, v2a=(29, 0), v2b=(30, 0), v2c=(30, 0))
        self.assertEqual(d["tied"], ["v2a", "v2b", "v2c"])
        self.assertEqual(d["selected"], "v2a")
        d = self.pick(20, v2a=(28, 0), v2b=(30, 0))
        self.assertEqual(d["tied"], ["v2b"], "相差 2 集不算并列")
        self.assertEqual(d["selected"], "v2b")

    def test_tie_is_judged_against_the_best_not_chained(self):
        d = self.pick(20, v2a=(28, 0), v2b=(29, 0), v2c=(30, 0))
        self.assertEqual(d["tied"], ["v2b", "v2c"])
        self.assertEqual(d["selected"], "v2b")

    def test_collide_first_above_three_is_out_exactly_at_three_is_kept(self):
        d = self.pick(20, v2a=(30, 4), v2b=(28, 3), v2c=(25, 0))
        self.assertEqual(d["excluded"], ["v2a"])
        self.assertNotIn("v2b", d["excluded"])
        self.assertEqual(d["selected"], "v2b", "撞在前恰好 3 集仍有资格；4 集出局")

    def test_excluded_arm_does_not_count_for_the_best(self):
        d = self.pick(20, v2a=(35, 5), v2b=(26, 0))
        self.assertEqual((d["best"], d["selected"]), (26, "v2b"))

    def test_all_excluded_selects_nothing(self):
        d = sv.choose({"v1": stats(20), **{n: stats(30, 9) for n in NAMES}}, NAMES, **KW)
        self.assertIsNone(d["selected"])
        self.assertFalse(d["gate_passed"])

    def test_gate_needs_at_least_three_more_than_v1(self):
        self.assertTrue(self.pick(20, v2a=(23, 0))["gate_passed"])
        self.assertFalse(self.pick(20, v2a=(22, 0))["gate_passed"])
        self.assertFalse(self.pick(25, v2a=(24, 0))["gate_passed"])
        self.assertEqual(self.pick(25, v2a=(24, 0))["gain_over_v1"], -1)

    def test_v1_never_selected(self):
        d = self.pick(39, v2a=(10, 0))
        self.assertEqual(d["selected"], "v2a")
        self.assertFalse(d["gate_passed"])


def write_arm(dev, arm, scene_routes, k, k_stop, k_cf=0, origin_split=None, fail=None, skip_done=None):
    """给 arm 写一整套结果：前 k_stop 集到点，接着 k_cf 集撞在前，其余超时。origin_split 不用。"""
    i = 0
    for s, rs in scene_routes.items():
        for r in rs:
            d = os.path.join(dev, arm, s, f"r{r}")
            eps = []
            for kk in range(k):
                os.makedirs(os.path.join(d, "ep_%05d" % kk), exist_ok=True)
                ok = i < k_stop
                cf = (not ok) and i < k_stop + k_cf
                m = {"success_stop": ok, "collide_first": cf, "collision": cf, "timeout": (not ok) and not cf}
                jdump(m, os.path.join(d, "ep_%05d" % kk, "metrics.json"))
                eps.append({"episode": kk})
                i += 1
            if (s, r) != skip_done:
                jdump({"episodes": eps, "failures": [{"x": 1}] if fail == (s, r) else []}, os.path.join(d, "done.json"))


class TestMain(unittest.TestCase):
    ROUTES = {"sA": list(range(8)), "sB": list(range(8))}              # 16 路线 × 1 集 = 16 集每臂；origin：每景路线 0-1 是 k1，其余 new

    def build(self, tmp, results, **kw):
        dev = os.path.join(tmp, "dev", "indoor")
        routes_file = os.path.join(tmp, "routes_dev.json")
        jdump({"scenes": {s: [{"route": r, "origin": "k1" if r < 2 else "new"} for r in rs] for s, rs in self.ROUTES.items()}}, routes_file)
        arms = [{"name": "v1"}] + [{"name": n} for n in NAMES]
        os.makedirs(dev)
        jdump({"arms": arms, "scenes": list(self.ROUTES), "routes": self.ROUTES, "k": 1, "hold": 1.0, "seed0": 9000, "routes_selected": routes_file},
              os.path.join(dev, "config.json"))
        for arm, (k_stop, k_cf) in results.items():
            write_arm(dev, arm, self.ROUTES, 1, k_stop, k_cf, **(kw if arm == "v2a" else {}))
        return dev

    def run_main(self, tmp, dev, extra=()):
        out = os.path.join(tmp, "out")
        argv = ["--dev", dev, "--out-dir", out, "--variants", TABLE, "--depth-map", "v2e=v2a v2f=v2d", *extra]
        o, e = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
            rc = sv.main(argv)
        return rc, out, o.getvalue(), e.getvalue()

    RESULTS = {"v1": (4, 0), "v2a": (5, 0), "v2b": (7, 1), "v2c": (8, 0), "v2d": (9, 4), "v2e": (3, 0), "v2f": (6, 0)}

    def test_passes_and_writes_selected_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, stdout, _ = self.run_main(tmp, self.build(tmp, self.RESULTS))
            self.assertEqual(rc, 0)
            sel = jload(os.path.join(out, "selected.json"))
            self.assertEqual(sel["variant"], "v2b", "v2d 到点最多（9）但撞在前 4 集出局；其余最优 8（v2c）；与最优相差不超过 1 的并列：v2b（7）、v2c（8），取靠前的 v2b")
            self.assertEqual((sel["order"], sel["levels"]), ("lexi", [[0.25, 1.2, False], [0.25, 0.8, False]]))
            self.assertEqual((sel["depth_variant"], sel["gain_over_v1"], sel["gate_bypassed"]), ("v2b", 3, False))
            self.assertIn("v2b", stdout)
            dsel = jload(os.path.join(out, "dev_selection.json"))
            self.assertEqual(dsel["decision"]["excluded"], ["v2d"])
            self.assertTrue(os.path.exists(os.path.join(out, "dev_selection.md")))

    def test_depth_variant_maps_for_the_zero_point_one_five_variants(self):
        res = {**self.RESULTS, "v2e": (9, 0), "v2c": (4, 0), "v2b": (4, 0)}
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, _ = self.run_main(tmp, self.build(tmp, res))
            sel = jload(os.path.join(out, "selected.json"))
            self.assertEqual((sel["variant"], sel["depth_variant"], sel["depth_levels"]), ("v2e", "v2a", [[0.25, 1.2, False]]))

    def test_gate_fail_returns_20_and_writes_no_selected_json(self):
        res = {**self.RESULTS, "v1": (6, 0)}
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, err = self.run_main(tmp, self.build(tmp, res))
            self.assertEqual(rc, 20)
            self.assertIn("没有比 v1 好", err)
            self.assertFalse(os.path.exists(os.path.join(out, "selected.json")))
            self.assertTrue(os.path.exists(os.path.join(out, "dev_selection.md")))

    def test_no_gate_flag_writes_selected_anyway(self):
        res = {**self.RESULTS, "v1": (6, 0)}
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, _ = self.run_main(tmp, self.build(tmp, res), extra=["--no-gate"])
            self.assertEqual(rc, 0)
            self.assertTrue(jload(os.path.join(out, "selected.json"))["gate_bypassed"])

    def test_incomplete_dev_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, err = self.run_main(tmp, self.build(tmp, self.RESULTS, skip_done=("sB", 4)))
            self.assertEqual(rc, 2)
            self.assertIn("不完整", err)
            self.assertFalse(os.path.exists(os.path.join(out, "selected.json")))

    def test_failed_episode_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, _, _, err = self.run_main(tmp, self.build(tmp, self.RESULTS, fail=("sA", 1)))
            self.assertEqual(rc, 2)

    def test_arms_with_different_episode_counts_are_refused(self):
        """load_arm 对每个臂读同一批 (场景, 路线, k)，集数按构造相同；main 里的这道检查是防御：以后 load_arm 的口径变了，各臂集数不同时不选。"""
        real = sv.load_arm

        def uneven(dev_dir, name, *args):
            eps = real(dev_dir, name, *args)
            return eps[:-1] if name == "v2c" else eps
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(sv, "load_arm", uneven):
            rc, out, _, err = self.run_main(tmp, self.build(tmp, self.RESULTS))
            self.assertEqual(rc, 2)
            self.assertIn("各臂集数不同", err)
            self.assertFalse(os.path.exists(os.path.join(out, "selected.json")))

    def test_existing_selected_json_is_never_overwritten_by_a_different_choice(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = self.build(tmp, self.RESULTS)
            rc, out, _, _ = self.run_main(tmp, dev)
            self.assertEqual(rc, 0)
            path = os.path.join(out, "selected.json")
            before = jload(path)
            rc, _, _, _ = self.run_main(tmp, dev)                        # 同一结果再跑：幂等
            self.assertEqual(rc, 0)
            self.assertEqual(jload(path), before)
            for arm, (ks, kc) in {**self.RESULTS, "v2e": (10, 0)}.items():
                write_arm(dev, arm, self.ROUTES, 1, ks, kc)
            rc, _, _, err = self.run_main(tmp, dev)
            self.assertEqual(rc, 2)
            self.assertIn("拒绝覆盖", err)
            self.assertEqual(jload(path), before)

    def test_arm_list_must_match_the_variant_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = self.build(tmp, self.RESULTS)
            cfg = jload(os.path.join(dev, "config.json"))
            cfg["arms"] = cfg["arms"][:-1]
            jdump(cfg, os.path.join(dev, "config.json"))
            rc, _, _, err = self.run_main(tmp, dev)
            self.assertEqual(rc, 2)

    def test_rate_reading_is_recorded_but_never_decides(self):
        # 16 集：对照读法的上限 = floor(3·16/40) = 1 集；字面读法（3 集）选 v2b（7 集，与最优 8 相差 1，靠前），对照读法下 v2b 撞在前 2 集出局，选 v2c
        res = {**self.RESULTS, "v2b": (7, 2)}
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, stdout, _ = self.run_main(tmp, self.build(tmp, res))
            self.assertEqual(rc, 0)
            self.assertEqual(jload(os.path.join(out, "selected.json"))["variant"], "v2b", "选取只按字面读法")
            alt = jload(os.path.join(out, "dev_selection.json"))["decision_rate_reading"]
            self.assertEqual((alt["max_collide_first"], alt["selected"]), (1, "v2c"))
            self.assertIn("v2b", alt["excluded"])
            self.assertIn("对照读法", stdout)
            self.assertIn("与字面读法不同", stdout)

    def test_origin_breakdown_in_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, stdout, _ = self.run_main(tmp, self.build(tmp, self.RESULTS))
            dsel = jload(os.path.join(out, "dev_selection.json"))
            bo = dsel["arms"]["v1"]["by_origin"]
            self.assertEqual((bo["k1"]["n"], bo["new"]["n"]), (4, 12))
            self.assertEqual(bo["k1"]["k_stop"] + bo["new"]["k_stop"], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
