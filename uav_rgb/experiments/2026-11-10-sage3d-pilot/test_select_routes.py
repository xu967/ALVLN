#!/usr/bin/env python3
"""select_routes.py 单测：只用标准库，不需要 numpy / torch / GPU。

  PYTHONDONTWRITEBYTECODE=1 python test_select_routes.py
没设 TMPDIR 时临时目录落数据盘（根分区只剩 2 GB，不往 /tmp 写）。
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
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

import select_routes as sr  # noqa: E402

KW = dict(per_scene=8, step_frac=0.5, step_pad=15, step_cap=100, len_edges=(5.0, 10.0))


def route(i, astar, max_steps=60, split="train"):
    """合成一条 scene.json 的 routes[i]（字段同数据接口）。"""
    return {"name": f"tt_{i}", "spawn": {"x": 0, "y": 0, "z": 1.2, "yaw_deg": 0},
            "waypoint": {"x": 1, "y": 1, "z": 1.2}, "max_steps": max_steps, "note": "",
            "sage": {"trajectory_id": i, "split": split, "sage_path_m": astar + 0.5, "n_points": 10,
                     "astar_path_m": astar, "turn_deg": 30.0, "min_clearance_m": 0.3,
                     "agl_model_unverified": False, "tof_terrain_m": None, "instructions": []}}


class TestPureFunctions(unittest.TestCase):
    def test_steps_h1_formula(self):
        self.assertEqual(sr.steps_h1(60, 0.5, 15), 45)
        self.assertEqual(sr.steps_h1(61, 0.5, 15), 46)     # ceil(30.5) = 31
        self.assertEqual(sr.steps_h1(0, 0.5, 15), 15)

    def test_bucket_edges_are_left_closed(self):
        e = (5.0, 10.0)
        self.assertEqual([sr.bucket_of(v, e) for v in (0.0, 4.99, 5.0, 9.99, 10.0, 25.0)],
                         ["<5 m", "<5 m", "5–10 m", "5–10 m", ">=10 m", ">=10 m"])
        self.assertEqual(sr.bucket_labels(e), ["<5 m", "5–10 m", ">=10 m"])

    def test_pick_indices(self):
        self.assertEqual(sr.pick_indices(5, 8), [0, 1, 2, 3, 4])             # n <= m 全取
        self.assertEqual(sr.pick_indices(8, 8), list(range(8)))
        self.assertEqual(sr.pick_indices(20, 8), [0, 3, 5, 8, 11, 14, 16, 19])   # round(i*19/7)
        self.assertEqual(sr.pick_indices(9, 8), [0, 1, 2, 3, 5, 6, 7, 8])
        self.assertEqual(sr.pick_indices(10, 1), [0])                         # 冒烟：m=1 取最短
        self.assertEqual(sr.pick_indices(10, 3), [0, 5, 9])                   # round(4.5) 向上取 5
        self.assertEqual(sr.pick_indices(0, 8), [])

    def test_pick_indices_always_distinct_and_cover_ends(self):
        for n in range(9, 200):
            idx = sr.pick_indices(n, 8)
            self.assertEqual(len(idx), 8, n)
            self.assertEqual(len(set(idx)), 8, n)
            self.assertEqual((idx[0], idx[-1]), (0, n - 1), n)


class TestSelectScene(unittest.TestCase):
    def test_n_le_8_takes_all_sorted_by_astar(self):
        routes = [route(0, 7.0), route(1, 3.0), route(2, 12.0)]
        out = sr.select_scene("s", routes, **KW)
        self.assertEqual([r["route"] for r in out["selected"]], [1, 0, 2])   # 按 A* 长度升序，route 是原下标
        self.assertEqual([r["bucket"] for r in out["selected"]], ["<5 m", "5–10 m", ">=10 m"])
        self.assertEqual(out["need_full_battery"], [])

    def test_row_schema(self):
        out = sr.select_scene("s", [route(4, 6.0, max_steps=80, split="test_Scene_Unseen")], **KW)
        self.assertEqual(out["selected"], [{
            "route": 0, "name": "tt_4", "split": "test_Scene_Unseen", "astar_path_m": 6.0, "sage_path_m": 6.5,
            "min_clearance_m": 0.3, "max_steps": 80, "steps_h1": 55, "bucket": "5–10 m"}])

    def test_quantile_pick_when_more_than_8(self):
        lengths = [float(v) for v in range(1, 21)]          # 20 条，A* 长度 1..20
        routes = [route(i, lengths[(i * 7) % 20]) for i in range(20)]   # 打乱输入顺序（7 与 20 互素）
        out = sr.select_scene("s", routes, **KW)
        self.assertEqual([r["astar_path_m"] for r in out["selected"]], [1.0, 4.0, 6.0, 9.0, 12.0, 15.0, 17.0, 20.0])
        self.assertEqual(len(out["selected"]), 8)

    def test_tie_broken_by_route_index(self):
        routes = [route(i, 5.0) for i in range(12)]
        out = sr.select_scene("s", routes, **KW)
        self.assertEqual([r["route"] for r in out["selected"]], [0, 2, 3, 5, 6, 8, 9, 11])

    def test_full_battery_boundary(self):
        # steps = ceil(max_steps*0.5)+15：max_steps=170 -> 100（留），171 -> 101（剔）
        routes = [route(0, 5.0, max_steps=170), route(1, 6.0, max_steps=171)]
        out = sr.select_scene("s", routes, **KW)
        self.assertEqual([r["route"] for r in out["selected"]], [0])
        self.assertEqual(out["selected"][0]["steps_h1"], 100)
        self.assertEqual(out["need_full_battery"], [{"route": 1, "name": "tt_1", "max_steps": 171,
                                                     "steps_uncapped": 101, "astar_path_m": 6.0}])

    def test_filter_before_stratifying(self):
        # 12 条里最长的 4 条需满电档：剔除后池里恰好 8 条，必须全取（先分层再剔除只会剩 5–6 条）
        routes = [route(i, float(i + 1), max_steps=(300 if i >= 8 else 60)) for i in range(12)]
        out = sr.select_scene("s", routes, **KW)
        self.assertEqual([r["route"] for r in out["selected"]], list(range(8)))
        self.assertEqual([r["route"] for r in out["need_full_battery"]], [8, 9, 10, 11])
        self.assertEqual((out["total"], out["pool"]), (12, 8))

    def test_per_scene_override_for_smoke(self):
        routes = [route(i, float(i + 1)) for i in range(10)]
        kw = dict(KW, per_scene=1)
        self.assertEqual([r["route"] for r in sr.select_scene("s", routes, **kw)["selected"]], [0])
        kw = dict(KW, per_scene=3)
        self.assertEqual([r["route"] for r in sr.select_scene("s", routes, **kw)["selected"]], [0, 5, 9])

    def test_all_filtered_gives_empty_selection(self):
        out = sr.select_scene("s", [route(0, 5.0, max_steps=999), route(1, 6.0, max_steps=999)], **KW)
        self.assertEqual(out["selected"], [])
        self.assertEqual(len(out["need_full_battery"]), 2)

    def test_missing_or_bad_astar_raises_with_context(self):
        bad = route(3, 5.0)
        del bad["sage"]["astar_path_m"]
        with self.assertRaisesRegex(ValueError, r"scene_x.*routes\[3\]"):
            sr.select_scene("scene_x", [route(0, 1.0), route(1, 2.0), route(2, 3.0), bad], **KW)
        nan = route(0, float("nan"))
        with self.assertRaisesRegex(ValueError, r"astar_path_m"):
            sr.select_scene("scene_x", [nan], **KW)

    def test_missing_max_steps_raises(self):
        r = route(0, 5.0)
        del r["max_steps"]
        with self.assertRaisesRegex(ValueError, r"max_steps"):
            sr.select_scene("scene_x", [r], **KW)


class TestCli(unittest.TestCase):
    def test_main_writes_routes_selected_json(self):
        with tempfile.TemporaryDirectory() as d:
            scenes = {"sage3d_1": [route(i, float(i + 1)) for i in range(12)],            # 12 条 -> 取 8
                      "sage3d_2": [route(0, 3.0), route(1, 20.0, max_steps=400)],          # 1 条留，1 条需满电档
                      "sage3d_3": [route(0, 3.0, max_steps=400)]}                          # 全部需满电档
            for name, rs in scenes.items():
                os.makedirs(os.path.join(d, name))
                with open(os.path.join(d, name, "scene.json"), "w", encoding="utf-8") as f:
                    json.dump({"schema": "uavrgb.3dgs_scene.v1", "routes": rs}, f)
            out = os.path.join(d, "routes_selected.json")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                rc = sr.main(["--scenes-dir", d, "--scenes", "sage3d_1 sage3d_2 sage3d_3", "--out", out,
                              "--per-scene", "8", "--step-frac", "0.5", "--step-pad", "15", "--step-cap", "100"])
            self.assertEqual(rc, 0)
            self.assertIn("需满电档 sage3d_2 r1（tt_1）：未封顶 215 步", printed.getvalue())
            with open(out, encoding="utf-8") as f:
                j = json.load(f)
            self.assertEqual(list(j["scenes"]), ["sage3d_1", "sage3d_2", "sage3d_3"])
            self.assertEqual(len(j["scenes"]["sage3d_1"]), 8)
            self.assertEqual([r["route"] for r in j["scenes"]["sage3d_2"]], [0])
            self.assertEqual(j["scenes"]["sage3d_3"], [])
            self.assertEqual(j["n_filtered"], {"sage3d_1": 0, "sage3d_2": 1, "sage3d_3": 1})
            self.assertEqual(j["need_full_battery"]["sage3d_2"][0]["steps_uncapped"], 215)
            self.assertEqual(j["need_full_battery"]["sage3d_1"], [])
            self.assertEqual(j["counts"]["sage3d_1"], {"total": 12, "pool": 12, "selected": 8})
            self.assertEqual(j["params"]["bucket_labels"], ["<5 m", "5–10 m", ">=10 m"])
            self.assertEqual(j["params"]["per_scene"], 8)

    def test_main_missing_scene_json_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                sr.main(["--scenes-dir", d, "--scenes", "nope", "--out", os.path.join(d, "o.json"),
                         "--per-scene", "8", "--step-frac", "0.5", "--step-pad", "15", "--step-cap", "100"])
            self.assertFalse(os.path.exists(os.path.join(d, "o.json")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
