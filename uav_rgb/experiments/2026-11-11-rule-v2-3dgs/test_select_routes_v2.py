#!/usr/bin/env python3
"""select_routes_v2.py 的单测：三批路线互不相交、沿用 K1 的合格条件与分位取法、不足时有多少取多少、复现 K1 开发路线的自检、
held-out 少于 24 条的停止码。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_select_routes_v2.py
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
import select_routes_v2 as s2  # noqa: E402

sd = s2.sd
KW = dict(step_frac=0.5, step_pad=15, step_cap=100, len_edges=(5.0, 10.0))
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP


def route(i, length, max_steps=60):
    return {"name": f"r{i}", "max_steps": max_steps, "sage": {"astar_path_m": length, "sage_path_m": length, "min_clearance_m": 0.4, "split": "x"}}


def scene(n, offset=1.0, bad=()):
    """n 条路线，A* 长度 offset + i；bad 里的下标步数预算超封顶（不合格）。"""
    return [route(i, offset + i, 400 if i in bad else 60) for i in range(n)]


def pilot_for(scene_routes, per_scene=8):
    """像试点那样先取 per_scene 条（sd.select_dev 的取法，排除名单为空）；返回 {场景: [路线 dict]}。"""
    return sd.select_dev(scene_routes, {}, per_scene=per_scene, **KW)["scenes"]


def k1_dev_for(scene_routes, pilot, per_scene=4):
    return sd.select_dev(scene_routes, pilot, per_scene=per_scene, **KW)["scenes"]


def run_select(scene_routes, dev_new=4, holdout=8, per_scene_pilot=8):
    pilot = pilot_for(scene_routes, per_scene_pilot)
    k1 = k1_dev_for(scene_routes, pilot)
    res = s2.select_v2(scene_routes, pilot, k1, dev_new_per_scene=dev_new, holdout_per_scene=holdout, **KW)
    return pilot, k1, res


def ids(routes):
    return [r["route"] for r in routes]



def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

class TestSelect(unittest.TestCase):
    def setUp(self):
        self.scenes = {"a": scene(60), "b": scene(45, 2.0), "c": scene(30, 0.5)}

    def test_three_sets_are_disjoint_and_complete_in_size(self):
        pilot, k1, res = run_select(self.scenes)
        dev, hold = res["dev"], res["holdout"]
        self.assertEqual(s2.disjoint_problems(pilot, dev, hold), [])
        for s in self.scenes:
            self.assertEqual(len(pilot[s]), 8)
            self.assertEqual(len(dev["scenes"][s]), 8, "K1 的 4 条 + 新的 4 条")
            self.assertEqual(len(hold["scenes"][s]), 8)
            self.assertEqual(len(set(ids(pilot[s]) + ids(dev["scenes"][s]) + ids(hold["scenes"][s]))), 24)

    def test_k1_dev_routes_are_kept_and_tagged(self):
        pilot, k1, res = run_select(self.scenes)
        for s in self.scenes:
            got = {r["route"]: r["origin"] for r in res["dev"]["scenes"][s]}
            self.assertEqual({r for r, o in got.items() if o == "k1"}, set(ids(k1[s])))
            self.assertEqual(sum(o == "new" for o in got.values()), 4)
            lens = [r["astar_path_m"] for r in res["dev"]["scenes"][s]]
            self.assertEqual(lens, sorted(lens), "dev 路线按 A* 长度升序")

    def test_k1_selection_is_reproduced_by_the_check(self):
        _, _, res = run_select(self.scenes)
        self.assertEqual(res["checks"]["k1_dev_reproduced"], {"a": True, "b": True, "c": True})

    def test_check_flags_a_different_k1_selection(self):
        pilot = pilot_for(self.scenes)
        k1 = k1_dev_for(self.scenes, pilot)
        k1["b"] = list(reversed(k1["b"]))
        res = s2.select_v2(self.scenes, pilot, k1, dev_new_per_scene=4, holdout_per_scene=8, **KW)
        self.assertEqual(res["checks"]["k1_dev_reproduced"], {"a": True, "b": False, "c": True})

    def test_new_routes_pass_k1s_qualification_filter(self):
        scenes = {"a": scene(60, bad=set(range(0, 60, 3)))}                  # 每 3 条有 1 条要满电档
        pilot, k1, res = run_select(scenes)
        bad = set(range(0, 60, 3))
        for key in ("dev", "holdout"):
            self.assertFalse(set(ids(res[key]["scenes"]["a"])) & bad, key)
        self.assertFalse(set(ids(pilot["a"])) & bad)

    def test_dev_new_uses_quantiles_of_the_rest_and_holdout_of_what_is_left(self):
        pilot, k1, res = run_select({"a": scene(60)})
        taken = set(ids(pilot["a"])) | set(ids(k1["a"]))
        rest = [r for r in sd.select_dev({"a": scene(60)}, {}, per_scene=60, **KW)["scenes"]["a"] if r["route"] not in taken]
        want_new = [rest[j]["route"] for j in sd.sr.pick_indices(len(rest), 4)]
        self.assertEqual([r["route"] for r in res["dev"]["scenes"]["a"] if r["origin"] == "new"], sorted(
            want_new, key=lambda i: (scene(60)[i]["sage"]["astar_path_m"], i)))
        rest2 = [r for r in rest if r["route"] not in set(want_new)]
        self.assertEqual(sorted(ids(res["holdout"]["scenes"]["a"])), sorted(rest2[j]["route"] for j in sd.sr.pick_indices(len(rest2), 8)))

    def test_exhausted_scene_takes_what_is_left_then_holdout_is_empty(self):
        scenes = {"a": scene(14)}                                           # 14：试点 8、K1 开发 4、剩 2
        pilot, k1, res = run_select(scenes)
        self.assertEqual((len(pilot["a"]), len(k1["a"])), (8, 4))
        self.assertEqual(len(res["dev"]["scenes"]["a"]), 6, "K1 的 4 条 + 只剩的 2 条")
        self.assertEqual(res["dev"]["counts"]["a"], {"k1": 4, "new": 2, "total": 6, "rest_before_new": 2})
        self.assertEqual(res["holdout"]["scenes"]["a"], [])
        self.assertEqual(s2.disjoint_problems(pilot, res["dev"], res["holdout"]), [])

    def test_holdout_takes_fewer_than_requested_when_short(self):
        scenes = {"a": scene(30)}                                           # 30 - 8 - 4 = 18 -> dev-new 4 -> 14 -> holdout 8 够
        _, _, res = run_select(scenes)
        self.assertEqual(len(res["holdout"]["scenes"]["a"]), 8)
        scenes = {"a": scene(22)}                                           # 22 - 12 = 10 -> dev-new 4 -> 6 -> holdout 6
        _, _, res = run_select(scenes)
        self.assertEqual(len(res["holdout"]["scenes"]["a"]), 6)

    def test_disjoint_problems_detects_overlap(self):
        pilot = {"a": [{"route": 1}, {"route": 2}]}
        dev = {"scenes": {"a": [{"route": 2}, {"route": 3}]}}
        hold = {"scenes": {"a": [{"route": 3}, {"route": 1}]}}
        self.assertEqual(sorted(s2.disjoint_problems(pilot, dev, hold)),
                         sorted([("a", "formal40", "dev", 2), ("a", "formal40", "holdout", 1), ("a", "dev", "holdout", 3)]))

    def test_counts_table_totals(self):
        pilot, k1, res = run_select(self.scenes)
        table = s2.counts_table(pilot, res["dev"], res["holdout"])
        self.assertIn("| 合计 | 24 | 24 | 24 |", table)


class TestMain(unittest.TestCase):
    def make_inputs(self, tmp, n_routes):
        scenes_dir = os.path.join(tmp, "scenes")
        scene_routes = {f"s{i}": scene(n) for i, n in enumerate(n_routes)}
        for s, routes in scene_routes.items():
            os.makedirs(os.path.join(scenes_dir, s))
            with open(os.path.join(scenes_dir, s, "scene.json"), "w", encoding="utf-8") as f:
                json.dump({"routes": routes}, f)
        pilot = pilot_for(scene_routes)
        k1 = k1_dev_for(scene_routes, pilot)
        files = {}
        for name, scenes in (("pilot.json", pilot), ("k1dev.json", k1)):
            files[name] = os.path.join(tmp, name)
            with open(files[name], "w", encoding="utf-8") as f:
                json.dump({"scenes": scenes}, f)
        return scenes_dir, " ".join(scene_routes), files

    def run_main(self, tmp, n_routes, min_holdout=24):
        scenes_dir, names, files = self.make_inputs(tmp, n_routes)
        out = os.path.join(tmp, "out")
        argv = ["--scenes-dir", scenes_dir, "--scenes", names, "--pilot-routes", files["pilot.json"], "--k1-dev-routes", files["k1dev.json"],
                "--out-dir", out, "--dev-new-per-scene", "4", "--holdout-per-scene", "8", "--min-holdout", str(min_holdout),
                "--step-frac", "0.5", "--step-pad", "15", "--step-cap", "100"]
        buf, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
            rc = s2.main(argv)
        return rc, out, files, err.getvalue()

    def test_writes_three_route_files_and_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, files, _ = self.run_main(tmp, [60, 45, 30])
            self.assertEqual(rc, 0)
            self.assertEqual(sorted(os.listdir(out)), ["counts.md", "routes_dev.json", "routes_formal40.json", "routes_holdout.json"])
            with open(files["pilot.json"], encoding="utf-8") as f1, open(os.path.join(out, "routes_formal40.json"), encoding="utf-8") as f2:
                self.assertEqual(f1.read(), f2.read(), "formal40 = 试点文件原样")
            dev, hold = jload(os.path.join(out, "routes_dev.json")), jload(os.path.join(out, "routes_holdout.json"))
            self.assertEqual((sum(map(len, dev["scenes"].values())), sum(map(len, hold["scenes"].values()))), (24, 24))
            self.assertIn("bucket_labels", dev["params"])
            self.assertIn("bucket_labels", hold["params"])
            self.assertTrue(all("bucket" in r for rs in hold["scenes"].values() for r in rs))

    def test_holdout_below_minimum_returns_3_but_still_writes_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, out, _, err = self.run_main(tmp, [22, 22, 22], min_holdout=24)           # 每场景 holdout 6，合计 18 < 24
            self.assertEqual(rc, 3)
            self.assertIn("少于 24", err)
            self.assertTrue(os.path.exists(os.path.join(out, "routes_holdout.json")))

    def test_holdout_exactly_at_minimum_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, _, _, _ = self.run_main(tmp, [22, 22, 22, 22], min_holdout=24)           # 4 × 6 = 24
            self.assertEqual(rc, 0)

    def test_overlap_between_the_three_sets_returns_2_and_writes_nothing(self):
        """select_v2 按构造给出互不相交的三批；main 写文件之前仍要再查一遍（K1 开发文件与试点文件若有重叠，也在这里被拦下）。"""
        real = s2.select_v2

        def overlapping(*args, **kw):
            res = real(*args, **kw)
            s = next(iter(res["holdout"]["scenes"]))
            res["holdout"]["scenes"][s].append(dict(res["dev"]["scenes"][s][0]))        # 留出里混进一条 dev 路线
            return res
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(s2, "select_v2", overlapping):
            rc, out, _, err = self.run_main(tmp, [60, 45, 30])
            self.assertEqual(rc, 2)
            self.assertIn("三批路线有重叠", err)
            self.assertFalse(os.path.exists(os.path.join(out, "routes_dev.json")), "重叠时不写路线文件")
            self.assertFalse(os.path.exists(os.path.join(out, "routes_holdout.json")))

    def test_k1_mismatch_returns_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            scenes_dir, names, files = self.make_inputs(tmp, [60, 45])
            k1 = jload(files["k1dev.json"])
            k1["scenes"]["s0"] = k1["scenes"]["s0"][::-1]
            with open(files["k1dev.json"], "w", encoding="utf-8") as f:
                json.dump(k1, f)
            argv = ["--scenes-dir", scenes_dir, "--scenes", names, "--pilot-routes", files["pilot.json"], "--k1-dev-routes", files["k1dev.json"],
                    "--out-dir", os.path.join(tmp, "out"), "--dev-new-per-scene", "4", "--holdout-per-scene", "8", "--min-holdout", "1",
                    "--step-frac", "0.5", "--step-pad", "15", "--step-cap", "100"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(s2.main(argv), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
