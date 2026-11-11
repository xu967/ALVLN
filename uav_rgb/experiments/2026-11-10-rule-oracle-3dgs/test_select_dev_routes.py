#!/usr/bin/env python3
"""select_dev_routes.py 的单测：排除试点已选、沿用试点的步数预算过滤与分位规则、不足时全取。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_select_dev_routes.py
"""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_dev_routes as sd  # noqa: E402

KW = dict(per_scene=4, step_frac=0.5, step_pad=15, step_cap=100, len_edges=(5.0, 10.0))


def route(i, length, max_steps=60):
    return {"name": f"r{i}", "max_steps": max_steps, "sage": {"astar_path_m": length, "sage_path_m": length, "min_clearance_m": 0.4, "split": "x"}}


class TestSelectDev(unittest.TestCase):
    def test_pilot_routes_are_never_selected(self):
        routes = [route(i, 2.0 + i) for i in range(20)]
        pilot = [{"route": i} for i in (0, 5, 10, 15, 19)]
        res = sd.select_dev({"s": routes}, {"s": pilot}, **KW)
        got = [r["route"] for r in res["scenes"]["s"]]
        self.assertEqual(len(got), 4)
        self.assertFalse(set(got) & {0, 5, 10, 15, 19})
        self.assertEqual(res["counts"]["s"], {"total": 20, "pool": 20, "pilot_excluded": 5, "rest": 15, "selected": 4})
        self.assertEqual(res["excluded_pilot"]["s"], [0, 5, 10, 15, 19])

    def test_quantile_picks_over_the_rest_sorted_by_astar_length(self):
        # 剩下 7 条，长度 1..7：floor(i·6/3 + 0.5) = 0, 2, 4, 6 -> 长度 1, 3, 5, 7（含最短与最长）
        routes = [route(i, float(i + 1)) for i in range(7)]
        res = sd.select_dev({"s": routes}, {"s": []}, **KW)
        self.assertEqual([r["astar_path_m"] for r in res["scenes"]["s"]], [1.0, 3.0, 5.0, 7.0])

    def test_budget_filter_is_the_pilots(self):
        # max_steps 170 -> ceil(85) + 15 = 100，不超封顶；max_steps 400 -> 215 > 100，需满电档，不进池
        routes = [route(0, 3.0, 170), route(1, 4.0, 400), route(2, 5.0, 60), route(3, 6.0, 60), route(4, 7.0, 60)]
        res = sd.select_dev({"s": routes}, {"s": []}, **KW)
        self.assertEqual([r["route"] for r in res["need_full_battery"]["s"]], [1])
        self.assertNotIn(1, [r["route"] for r in res["scenes"]["s"]])
        self.assertEqual(res["counts"]["s"]["pool"], 4)

    def test_fewer_than_n_left_takes_all_of_them_sorted(self):
        routes = [route(0, 9.0), route(1, 3.0), route(2, 6.0), route(3, 1.0)]
        res = sd.select_dev({"s": routes}, {"s": [{"route": 3}]}, **KW)
        self.assertEqual([r["route"] for r in res["scenes"]["s"]], [1, 2, 0])
        self.assertEqual(res["counts"]["s"], {"total": 4, "pool": 4, "pilot_excluded": 1, "rest": 3, "selected": 3})

    def test_scene_without_routes_and_unknown_pilot_scene(self):
        res = sd.select_dev({"s": [], "t": [route(0, 2.0)]}, {"t": [{"route": 0}], "zzz": [{"route": 9}]}, **KW)
        self.assertEqual(res["scenes"], {"s": [], "t": []})
        self.assertEqual(res["counts"]["t"]["rest"], 0)

    def test_buckets_are_filled(self):
        routes = [route(0, 2.0), route(1, 7.0), route(2, 12.0)]
        res = sd.select_dev({"s": routes}, {"s": []}, **KW)
        self.assertEqual([r["bucket"] for r in res["scenes"]["s"]], ["<5 m", "5–10 m", ">=10 m"])

    def test_main_writes_the_file_and_records_the_pilot_sha(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "scenes", "s"))
            with open(os.path.join(d, "scenes", "s", "scene.json"), "w", encoding="utf-8") as f:
                json.dump({"routes": [route(i, 2.0 + i) for i in range(10)]}, f)
            with open(os.path.join(d, "pilot.json"), "w", encoding="utf-8") as f:
                json.dump({"scenes": {"s": [{"route": 0}, {"route": 9}]}}, f)
            out = os.path.join(d, "dev.json")
            rc = sd.main(["--scenes-dir", os.path.join(d, "scenes"), "--scenes", "s", "--pilot-routes", os.path.join(d, "pilot.json"),
                          "--out", out, "--per-scene", "4", "--step-frac", "0.5", "--step-pad", "15", "--step-cap", "100"])
            self.assertEqual(rc, 0)
            with open(out, encoding="utf-8") as f:
                res = json.load(f)
            self.assertEqual(len(res["scenes"]["s"]), 4)
            self.assertEqual(len(res["params"]["pilot_routes_sha256"]), 64)
            self.assertFalse({r["route"] for r in res["scenes"]["s"]} & {0, 9})
            self.assertFalse(os.path.exists(out + ".tmp"))


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
