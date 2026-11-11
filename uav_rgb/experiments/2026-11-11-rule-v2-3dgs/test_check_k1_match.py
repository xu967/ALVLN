#!/usr/bin/env python3
"""check_k1_match.py 的单测：逐集一致 / 各字段不一致 / 出生位姿不一致 / 数据不全 / K1 缺键 的退出码与内容。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_check_k1_match.py
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
import check_k1_match as ck  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

ROUTES = {"sA": [0, 3], "sB": [1]}
K = 3


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def metrics(scene, route, k, **over):
    m = {"scene": scene, "route": route, "episode": k, "seed": 8000 + 100 * route + k, "success_stop": k != 1, "collision": k == 2,
         "collide_first": k == 2, "timeout": False, "steps": 40 + k, "reach_step": 30 + k if k != 1 else None, "collision_step": 20 if k == 2 else None,
         "start_pose": {"x": 1.0 + route, "y": 2.0, "yaw_deg": 10.0 * k}}
    m.update(over)
    return m


def write_arm(cdir, arm, routes=ROUTES, skip_done=None, patch=None):
    for s, rs in routes.items():
        for r in rs:
            d = os.path.join(cdir, arm, s, f"r{r}")
            for k in range(K):
                m = metrics(s, r, k)
                if patch and (s, r, k) in patch:
                    m.update(patch[(s, r, k)])
                jdump(m, os.path.join(d, f"ep_{k:05d}", "metrics.json"))
            if (s, r) != skip_done:
                jdump({"episodes": [{} for _ in range(K)], "failures": []}, os.path.join(d, "done.json"))


def run_check(new, k1):
    o, e = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
        rc = ck.main(["--new", new, "--k1", k1, "--out-json", os.path.join(new, "k1_match.json"), "--out-md", os.path.join(new, "k1_match.md")])
    return rc, o.getvalue(), e.getvalue()


class TestMatch(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.new, self.k1 = os.path.join(self.tmp, "formal40"), os.path.join(self.tmp, "k1")
        jdump({"scenes": list(ROUTES), "routes": ROUTES, "k": K, "arms": [{"name": "rule-occ+route"}]}, os.path.join(self.new, "config.json"))
        write_arm(self.k1, "rule-occ+route")

    def tearDown(self):
        self._tmp.cleanup()

    def test_identical_is_rc0_and_writes_reports(self):
        write_arm(self.new, "rule-occ+route")
        rc, out, _ = run_check(self.new, self.k1)
        self.assertEqual(rc, 0, out)
        with open(os.path.join(self.new, "k1_match.json"), encoding="utf-8") as f:
            res = json.load(f)
        self.assertEqual((res["matched"], res["equal"], res["n_expected"], res["ok"]), (9, 9, 9, True))
        with open(os.path.join(self.new, "k1_match.md"), encoding="utf-8") as f:
            self.assertIn("一致", f.read())

    def test_each_compared_field_triggers_rc1(self):
        for field, val in (("success_stop", True), ("collision", True), ("steps", 99), ("reach_step", 5), ("collision_step", 7), ("timeout", True),
                           ("collide_first", True)):
            with self.subTest(field=field):
                write_arm(self.new, "rule-occ+route", patch={("sA", 3, 1): {field: val if field != "success_stop" else True}})
                rc, out, _ = run_check(self.new, self.k1)
                self.assertEqual(rc, 1, out)
                self.assertIn(field, out)
                self.assertIn("sA", out)

    def test_start_pose_difference_triggers_rc1(self):
        write_arm(self.new, "rule-occ+route", patch={("sB", 1, 0): {"start_pose": {"x": 2.0000001 + 1e-3, "y": 2.0, "yaw_deg": 0.0}}})
        rc, out, _ = run_check(self.new, self.k1)
        self.assertEqual(rc, 1)
        self.assertIn("start_pose", out)

    def test_pose_within_tolerance_is_equal(self):
        write_arm(self.new, "rule-occ+route", patch={("sB", 1, 0): {"start_pose": {"x": 2.0 + 5e-7, "y": 2.0, "yaw_deg": 0.0}}})
        self.assertEqual(run_check(self.new, self.k1)[0], 0)

    def test_missing_route_is_rc2(self):
        write_arm(self.new, "rule-occ+route", skip_done=("sA", 3))
        rc, out, _ = run_check(self.new, self.k1)
        self.assertEqual(rc, 2, out)

    def test_missing_route_is_incomplete_even_when_extra_episodes_make_up_the_count(self):
        """完整性要同时看缺的路线与集数：sA r3 没有 done.json，sB r1 多出 3 集（K1 里也有），总集数正好等于期望的 9，仍然不完整。"""
        write_arm(self.new, "rule-occ+route", skip_done=("sA", 3))
        for root in (self.new, self.k1):
            for k in (3, 4, 5):
                jdump(metrics("sB", 1, k), os.path.join(root, "rule-occ+route", "sB", "r1", f"ep_{k:05d}", "metrics.json"))
        rc, out, _ = run_check(self.new, self.k1)
        self.assertEqual(rc, 2, out)
        with open(os.path.join(self.new, "k1_match.json"), encoding="utf-8") as f:
            res = json.load(f)
        self.assertEqual((res["n_new"], res["n_expected"], res["missing_routes"], res["complete"], res["ok"]), (9, 9, [["sA", 3]], False, False))

    def test_k1_missing_a_key_is_rc2(self):
        import shutil
        write_arm(self.new, "rule-occ+route")
        shutil.rmtree(os.path.join(self.k1, "rule-occ+route", "sB", "r1", "ep_00002"))
        rc, out, _ = run_check(self.new, self.k1)
        self.assertEqual(rc, 2, out)

    def test_mismatch_wins_over_incomplete(self):
        write_arm(self.new, "rule-occ+route", skip_done=("sA", 3), patch={("sA", 0, 0): {"steps": 1}})
        self.assertEqual(run_check(self.new, self.k1)[0], 1)

    def test_extra_keys_in_new_are_rc2(self):
        write_arm(self.new, "rule-occ+route")
        write_arm(self.k1, "rule-occ+route", routes={"sA": [0], "sB": [1]})            # K1 少了 sA r3
        import shutil
        shutil.rmtree(os.path.join(self.k1, "rule-occ+route", "sA", "r3"))
        self.assertEqual(run_check(self.new, self.k1)[0], 2)

    def test_no_config_is_rc2(self):
        os.remove(os.path.join(self.new, "config.json"))
        self.assertEqual(run_check(self.new, self.k1)[0], 2)

    def test_list_style_routes_in_config(self):
        jdump({"scenes": ["sA"], "routes": [0, 3], "k": K, "arms": []}, os.path.join(self.new, "config.json"))
        write_arm(self.new, "rule-occ+route", routes={"sA": [0, 3]})
        self.assertEqual(run_check(self.new, self.k1)[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
