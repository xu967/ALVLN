#!/usr/bin/env python3
"""build_config.py 的单测：四个类别的臂表（名字、顺序、base、rule_version、rule_params）、种子与 K、路线取法、selected.json 的使用、缺项报错。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_build_config.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_config as bc  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

TABLE = """v2a|0.25:1.2|lexi
v2b|0.25:1.2,0.25:0.8|lexi
v2c|0.25:1.2,0.25:0.8|near
v2d|0.25:1.2,0.25:0.8,0.20:0.55*|near
v2e|0.15:1.2|lexi
v2f|0.15:1.2,0.15:0.8,0.12:0.55*|near"""


def jdump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


class Env:
    def __init__(self, tmp):
        self.tmp = tmp
        routes = {"scenes": {"sA": [{"route": 3}, {"route": 7}], "sB": [{"route": 1}]}}
        for n in ("ROUTES_DEV", "ROUTES_HOLDOUT", "ROUTES_FORMAL40"):
            jdump(routes, os.path.join(tmp, n.lower() + ".json"))
        self.e = {"REPO": "/repo", "SCENES_DIR": "/scenes", "HOLD": "1.0", "STEP_FRAC": "0.5", "STEP_PAD": "15", "GUARD_DEG": "45.0",
                  "DROP_TRACE": "true", "CF_SITL_PROC": "/cf", "YAW_STEP": "31.7", "V1_R_SAFE": "0.25", "V1_D_LOOK": "1.2", "VARIANTS": TABLE,
                  "INDOOR_SCENES": "sA sB", "OUTDOOR_SCENES": "o1 o2", "ROUTES_OUTDOOR": "0 1 2 3 4 5",
                  "ROUTES_DEV": os.path.join(tmp, "routes_dev.json"), "ROUTES_HOLDOUT": os.path.join(tmp, "routes_holdout.json"),
                  "ROUTES_FORMAL40": os.path.join(tmp, "routes_formal40.json"),
                  "K_DEV": "1", "K_HOLDOUT": "2", "K_FORMAL40": "3", "K_OUTDOOR": "5",       # 四个互不相同：类别与 K 对错位会被看出来
                  "SEED0_DEV": "9000", "SEED0_HOLDOUT": "5000", "SEED0_FORMAL40": "8000", "SEED0_OUTDOOR": "7000",
                  "JITTER_INDOOR": "0.2", "JITTER_OUTDOOR": "0.5", "LOOKAHEAD_INDOOR": "1.5", "LOOKAHEAD_OUTDOOR": "3.0", "STEP_CAP_INDOOR": "100"}


SELECTED = {"variant": "v2c", "depth_variant": "v2c"}
SELECTED_E = {"variant": "v2e", "depth_variant": "v2a"}


class TestArms(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.e = Env(self._tmp.name).e

    def tearDown(self):
        self._tmp.cleanup()

    def names(self, cls, sel=None):
        return [a["name"] for a in bc.build(cls, self.e, sel)["arms"]]

    def test_arm_lists_and_order(self):
        self.assertEqual(self.names("dev"), ["v1", "v2a", "v2b", "v2c", "v2d", "v2e", "v2f"])
        self.assertEqual(self.names("holdout", SELECTED), ["forward+route", "rule-occ+route", "rule2-occ+route", "rule2-depth+route"])
        self.assertEqual(self.names("formal40", SELECTED), ["rule-occ+route", "rule2-occ+route", "rule2-depth+route"])
        self.assertEqual(self.names("outdoor", SELECTED), ["rule2-occ+route", "rule2-depth+route"])

    def test_v1_arm_is_k1s_frozen_arm_without_new_keys(self):
        for cls in ("dev", "holdout", "formal40"):
            arms = {a["name"]: a for a in bc.build(cls, self.e, SELECTED if cls != "dev" else None)["arms"]}
            a = arms["v1" if cls == "dev" else "rule-occ+route"]
            self.assertEqual({k: v for k, v in a.items() if k != "name"},
                             {"base": "rule-occ", "ckpt": None, "guard": "route", "rule_params": {"r_safe": 0.25, "d_look": 1.2}})
            self.assertNotIn("rule_version", a)

    def test_dev_variants_carry_the_card_table(self):
        arms = {a["name"]: a for a in bc.build("dev", self.e)["arms"]}
        self.assertEqual(arms["v2a"]["rule_params"], {"levels": [[0.25, 1.2, False]], "order": "lexi"})
        self.assertEqual(arms["v2d"]["rule_params"], {"levels": [[0.25, 1.2, False], [0.25, 0.8, False], [0.20, 0.55, True]], "order": "near"})
        self.assertEqual(arms["v2f"]["rule_params"], {"levels": [[0.15, 1.2, False], [0.15, 0.8, False], [0.12, 0.55, True]], "order": "near"})
        for n in ("v2a", "v2b", "v2c", "v2d", "v2e", "v2f"):
            self.assertEqual((arms[n]["base"], arms[n]["rule_version"], arms[n]["guard"]), ("rule-occ", "v2", "route"))

    def test_main_and_depth_arms_use_the_selection(self):
        arms = {a["name"]: a for a in bc.build("holdout", self.e, SELECTED_E)["arms"]}
        self.assertEqual(arms["rule2-occ+route"]["rule_params"], {"levels": [[0.15, 1.2, False]], "order": "lexi"})
        self.assertEqual(arms["rule2-depth+route"]["rule_params"], {"levels": [[0.25, 1.2, False]], "order": "lexi"}, "depth 臂用同结构的 0.25 一档")
        self.assertEqual((arms["rule2-occ+route"]["base"], arms["rule2-depth+route"]["base"]), ("rule-occ", "rule-depth"))
        self.assertEqual(arms["forward+route"], {"name": "forward+route", "base": "forward", "ckpt": None, "guard": "route"})

    def test_global_rule_params_and_frozen_block(self):
        cfg = bc.build("holdout", self.e, SELECTED)
        self.assertEqual(cfg["rule_params"], {"yaw_step_deg": 31.7})
        self.assertEqual(cfg["selected"], SELECTED)
        self.assertEqual(cfg["frozen"], {"yaw_step_deg": 31.7, "v1_r_safe": 0.25, "v1_d_look": 1.2})

    def test_k_seed_and_routes(self):
        dev, hold, f40, out = (bc.build("dev", self.e), bc.build("holdout", self.e, SELECTED), bc.build("formal40", self.e, SELECTED),
                               bc.build("outdoor", self.e, SELECTED))
        self.assertEqual([(c["k"], c["seed0"]) for c in (dev, hold, f40, out)], [(1, 9000), (2, 5000), (3, 8000), (5, 7000)])
        self.assertEqual(dev["routes"], {"sA": [3, 7], "sB": [1]})
        self.assertEqual(out["routes"], [0, 1, 2, 3, 4, 5])
        self.assertEqual(out["scenes"], ["o1", "o2"])
        self.assertEqual((dev["jitter"], out["jitter"], dev["route_lookahead_m"], out["route_lookahead_m"]), (0.2, 0.5, 1.5, 3.0))
        self.assertEqual((dev["hold"], dev["step_cap"], dev["step_frac"], dev["step_pad"], dev["guard_threshold_deg"]), (1.0, 100, 0.5, 15, 45.0))
        self.assertNotIn("step_cap", out)

    def test_routes_limit_for_smoke(self):
        self.e["ROUTES_LIMIT"] = "1"
        self.assertEqual(bc.build("dev", self.e)["routes"], {"sA": [3], "sB": [1]})
        self.assertEqual(bc.build("outdoor", self.e, SELECTED)["routes"], [0])

    def test_non_dev_needs_selection_and_missing_env_is_an_error(self):
        with self.assertRaises(ValueError):
            bc.build("holdout", self.e, None)
        with self.assertRaises(ValueError):
            bc.build("nope", self.e)
        del self.e["YAW_STEP"]
        with self.assertRaises(ValueError):
            bc.build("dev", self.e)

    def test_selected_variant_unknown_to_table_is_an_error(self):
        with self.assertRaises(KeyError):
            bc.build("holdout", self.e, {"variant": "v9", "depth_variant": "v2a"})

    def test_only_keeps_named_arms_in_table_order(self):
        """F0 用：正式 40 条先只跑 v1 臂（只改 arms，其余键逐项相同），之后换完整三臂版。"""
        full = bc.build("formal40", self.e, SELECTED)
        f0 = bc.build("formal40", self.e, SELECTED, only=["rule-occ+route"])
        self.assertEqual([a["name"] for a in f0["arms"]], ["rule-occ+route"])
        self.assertEqual(f0["arms"][0], full["arms"][0])
        self.assertEqual({k: v for k, v in f0.items() if k not in ("arms", "only")}, {k: v for k, v in full.items() if k != "arms"})
        self.assertEqual(f0["only"], ["rule-occ+route"])
        two = bc.build("formal40", self.e, SELECTED, only=["rule2-depth+route", "rule-occ+route"])
        self.assertEqual([a["name"] for a in two["arms"]], ["rule-occ+route", "rule2-depth+route"], "顺序仍按臂表，不按 only 的写法")
        self.assertNotIn("only", full)

    def test_only_with_unknown_or_empty_names_is_an_error(self):
        with self.assertRaises(ValueError):
            bc.build("formal40", self.e, SELECTED, only=["rule-occ+route", "nope"])
        with self.assertRaises(ValueError):
            bc.build("formal40", self.e, SELECTED, only=[])

    def test_cli_only_flag(self):
        out = os.path.join(self._tmp.name, "f0.json")
        sel = os.path.join(self._tmp.name, "selected.json")
        jdump(SELECTED, sel)
        old = dict(os.environ)
        os.environ.update(self.e)
        try:
            self.assertEqual(bc.main(["formal40", out, "--selected", sel, "--only", "rule-occ+route"]), 0)
        finally:
            os.environ.clear()
            os.environ.update(old)
        with open(out, encoding="utf-8") as f:
            self.assertEqual([a["name"] for a in json.load(f)["arms"]], ["rule-occ+route"])

    def test_cli_writes_json(self):
        out = os.path.join(self._tmp.name, "sub", "config.json")
        old = dict(os.environ)
        os.environ.update(self.e)
        try:
            self.assertEqual(bc.main(["dev", out]), 0)
        finally:
            os.environ.clear()
            os.environ.update(old)
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f)["class"], "dev")


if __name__ == "__main__":
    unittest.main(verbosity=2)
