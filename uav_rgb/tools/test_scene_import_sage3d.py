#!/usr/bin/env python3
"""tools/scene_import_sage3d.py 的离线测试：纯函数 + 合成场景，不下载、不碰 GPU、不依赖 msgpack/torch。

跑法（与 tools/check.sh 同口径）：.venv-tools/bin/python tools/test_scene_import_sage3d.py
覆盖：地面（z 直方图）、坐标系复核（合成 L 形走廊）、VLN 轨迹解析（起终点/朝向/指令字段）、路线过滤三条规则 + 追加的一条、
route_check 的余量与地形参数透传、真实 Occupancy/OccPlanner 上的小房间、scene.json 构造（nav_to_world、ground_z、飞行高度一致性）、
飞行高度写进 cf_sitl 起飞指令且 dynamics_profile 不随之变（假 cf_sitl_proc）、下载的 429 退避。
"""
import json
import math
import os
import stat
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
import importlib  # noqa: E402

import scene_import_sage3d as imp  # noqa: E402

collision = importlib.import_module("sim.3dgs.collision")
planner_mod = importlib.import_module("sim.3dgs.planner")
dyn_mod = importlib.import_module("sim.3dgs.dynamics")
scene_mod = importlib.import_module("sim.3dgs.scene")


def wall_gaussians(segments, z=1.2, step=0.1):
    """沿线段撒小高斯（σ 0.05 m、竖直拉长），返回 means/scales/opacities。"""
    pts = []
    for (x0, y0), (x1, y1) in segments:
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / step) + 1)
        pts += [(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, z) for t in np.linspace(0, 1, n)]
    means = np.array(pts, dtype=np.float64)
    scales = np.tile([0.05, 0.05, 0.4], (len(means), 1))
    return means, scales, np.full(len(means), 0.9)


class TestGround(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def _z(self, floor_n=3000, ceil_n=1500, floor_z=-0.08, ceil_z=2.7):
        return np.concatenate([self.rng.normal(floor_z, 0.01, floor_n), self.rng.uniform(0.1, 2.6, 4000),
                               self.rng.normal(ceil_z, 0.01, ceil_n)])

    def test_floor_and_ceiling_peaks(self):
        z = self._z()
        r = imp.ground_from_z_hist(z, np.full(len(z), 0.9))
        self.assertAlmostEqual(r["ground_z"], -0.08, delta=0.02)
        self.assertAlmostEqual(r["ceiling_z"], 2.7, delta=0.06)
        self.assertGreater(r["floor_frac"], 0.2)

    def test_denser_ceiling_does_not_win(self):
        z = self._z(floor_n=2000, ceil_n=5000, ceil_z=2.8)
        r = imp.ground_from_z_hist(z, np.full(len(z), 0.9))
        self.assertAlmostEqual(r["ground_z"], -0.08, delta=0.02)

    def test_transparent_floaters_ignored(self):
        z = self._z()
        op = np.full(len(z), 0.9)
        z2 = np.concatenate([z, np.full(8000, -3.0)])
        op2 = np.concatenate([op, np.full(8000, 0.05)])
        r = imp.ground_from_z_hist(z2, op2)
        self.assertAlmostEqual(r["ground_z"], -0.08, delta=0.02)

    def test_too_few_points_raises(self):
        with self.assertRaises(ValueError):
            imp.ground_from_z_hist(np.zeros(10), np.ones(10))


class TestFrame(unittest.TestCase):
    """合成场景：24 m × 24 m 实心障碍场里挖一条 L 形走廊（世界系）。SAGE 轨迹系 = 世界系绕 z 转 180°。"""

    def test_raster_dilation_does_not_wrap_at_scene_edges(self):
        mask = imp._raster(np.array([[0.0, 0.0]]), 0.0, 0.0, 5, 5, 1.0, 1)
        self.assertTrue(mask[0, 0])
        self.assertTrue(mask[0, 1] and mask[1, 0])
        self.assertFalse(mask[4].any())
        self.assertFalse(mask[:, 4].any())

    @classmethod
    def setUpClass(cls):
        xs = np.arange(-12.0, 12.01, 0.1)
        gx, gy = np.meshgrid(xs, xs)
        pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        leg1 = (pts[:, 0] >= -1.0) & (pts[:, 0] <= 8.8) & (np.abs(pts[:, 1]) < 0.8)
        leg2 = (pts[:, 0] >= 7.2) & (pts[:, 0] <= 8.8) & (pts[:, 1] >= -0.8) & (pts[:, 1] <= 6.5)
        solid = pts[~(leg1 | leg2)]
        cls.means = np.concatenate([solid, np.full((len(solid), 1), 1.2)], axis=1)
        cls.op = np.full(len(cls.means), 0.9)
        cls.smax = np.full(len(cls.means), 0.05)
        t = np.arange(0.0, 8.0, 0.1)
        s = np.arange(0.0, 6.0, 0.1)
        cls.world = np.concatenate([np.stack([t, np.zeros_like(t)], 1), np.stack([np.full_like(s, 8.0), s], 1)])

    def _check(self, traj):
        return imp.frame_check(self.means, self.op, self.smax, traj, z_lo=1.1, z_hi=1.3)

    def test_true_frame_is_rotation_180(self):
        res = self._check(-self.world)           # SAGE 系 = Rz(180°) 世界系
        self.assertEqual(res["best_deg"], 180)
        self.assertLess(res["rotations"]["180"]["hit_frac"], 0.02)
        for k in ("0", "90", "270"):
            self.assertGreater(res["rotations"][k]["hit_frac"], 0.4, k)
        ok, why = imp.frame_verdict(res)
        self.assertTrue(ok, why)

    def test_unrotated_trajectory_fails(self):
        res = self._check(self.world)            # 轨迹已经在世界系：180° 假设不成立（最优是 0°）
        self.assertEqual(res["best_deg"], 0)
        ok, why = imp.frame_verdict(res)
        self.assertFalse(ok)
        self.assertIn("180", why)

    def test_verdict_rules(self):
        def rows(h180, others, shift=0.0):
            r = {str(d): {"hit_frac": h, "best_shift_m": 0.0, "best_shift_hit_frac": h} for d, h in zip((0, 90, 270), others)}
            r["180"] = {"hit_frac": h180, "best_shift_m": shift, "best_shift_hit_frac": h180}
            best = min(r, key=lambda k: r[k]["hit_frac"])
            return {"rotations": r, "best_deg": int(best)}
        self.assertTrue(imp.frame_verdict(rows(0.11, [0.24, 0.30, 0.27]))[0])             # bench 841150 的量级
        self.assertFalse(imp.frame_verdict(rows(0.20, [0.24, 0.30, 0.27]))[0])            # 与第二名差距不够
        self.assertFalse(imp.frame_verdict(rows(0.11, [0.24, 0.30, 0.27], shift=0.8))[0])  # 平移假设（零平移）不成立
        self.assertFalse(imp.frame_verdict(rows(0.30, [0.24, 0.30, 0.27]))[0])            # 180° 不是最小


def vln_doc(trajs, split_scene="0001"):
    samples = []
    for tid, pts, instr in trajs:
        samples.append({"trajectory_id": tid,
                        "instructions": [{"instruction_type": t, "start": a, "end": b, "generated_instruction": s,
                                          "scene_id": split_scene} for t, a, b, s in instr],
                        "points": [{"position": [float(x), float(y), 0.5], "rotation": [0.1, 0.2, 0.3, 0.9]} for x, y in pts]})
    return {"scenes": [{"scene_id": split_scene, "samples": samples}]}


def line(a, b, step=0.05):
    n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / step) + 1
    return [(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t) for t in np.linspace(0, 1, n)]


class TestVln(unittest.TestCase):
    def test_candidate_fields(self):
        pts = line((1.0, 2.0), (4.0, 2.0)) + line((4.0, 2.05), (4.0, 5.0))[1:]
        doc = vln_doc([(7, pts, [("Add_Object", "sofa_1", "bed_2", "Bring a pillow."),
                                 ("Scenario_Driven", "sofa_1", "bed_2", "I want to nap.")])])
        c = imp.routes_from_vln(doc, "test_Trajectory_Unseen")
        self.assertEqual(len(c), 1)
        c = c[0]
        self.assertEqual(c["trajectory_id"], 7)
        self.assertEqual(c["split"], "test_Trajectory_Unseen")
        self.assertEqual(c["n_points"], len(pts))
        self.assertAlmostEqual(c["start"][0], 1.0)
        self.assertAlmostEqual(c["goal"][1], 5.0, places=1)
        self.assertAlmostEqual(c["sage_path_m"], 6.0, delta=0.1)
        self.assertAlmostEqual(c["yaw_deg"], 0.0, delta=1.0)          # 朝路径上离起点 0.6 m 的点，第一段沿 +x
        self.assertEqual([i["instruction_type"] for i in c["instructions"]], ["Add_Object", "Scenario_Driven"])
        self.assertEqual(set(c["instructions"][0]), {"instruction_type", "start", "end", "text"})
        self.assertEqual(c["instructions"][1]["text"], "I want to nap.")
        self.assertEqual((c["instructions"][0]["start"], c["instructions"][0]["end"]), ("sofa_1", "bed_2"))

    def test_yaw_from_geometry_not_quaternion(self):
        c = imp.routes_from_vln(vln_doc([(1, line((0, 0), (0, 3)), [])]), "train")[0]
        self.assertAlmostEqual(c["yaw_deg"], 90.0, delta=1.0)

    def test_degenerate_skipped(self):
        doc = vln_doc([(1, [(0, 0)], []), (2, [(1, 1)] * 5, []), (3, line((0, 0), (3, 0)), [])])
        self.assertEqual([c["trajectory_id"] for c in imp.routes_from_vln(doc, "train")], [3])

    def test_route_name_and_order(self):
        cands = [{"trajectory_id": 12, "split": "train"}, {"trajectory_id": 3, "split": "test_Scene_Unseen"},
                 {"trajectory_id": 5, "split": "test_Instruction_Unseen"}, {"trajectory_id": 2, "split": "train"}]
        order = imp.order_candidates(cands)
        self.assertEqual([c["trajectory_id"] for c in order], [5, 3, 2, 12])     # 测试划分在前，组内按轨迹号
        self.assertEqual(imp.route_name(order[0]), "ti_5")
        self.assertEqual(imp.route_name(order[1]), "ts_3")
        self.assertEqual(imp.route_name(order[2]), "tr_2")


def chk(**kw):
    base = {"feasible": True, "spawn_free": True, "waypoint_free": True, "path_m": 5.0, "clear_ok": True,
            "min_clearance_m": 0.5, "turn_deg": 20.0, "steps_estimate": 60}
    base.update(kw)
    return base


class TestJudge(unittest.TestCase):
    cand = {"sage_path_m": 4.0, "start": (0.0, 0.0), "goal": (3.0, 0.0)}

    def test_pass(self):
        ok, why = imp.judge_route(chk(), self.cand)
        self.assertTrue(ok)
        self.assertEqual(why, [])

    def test_no_path(self):
        ok, why = imp.judge_route(chk(feasible=False, reason="A* 无路径", path_m=None, clear_ok=None), self.cand)
        self.assertFalse(ok)
        self.assertIn("A* 无路径", why)

    def test_blocked_endpoints(self):
        ok, why = imp.judge_route(chk(feasible=False, spawn_free=False), self.cand)
        self.assertFalse(ok)
        self.assertIn("起点被占据", why)
        ok, why = imp.judge_route(chk(feasible=False, waypoint_free=False), self.cand)
        self.assertIn("终点被占据", why)

    def test_clearance(self):
        ok, why = imp.judge_route(chk(clear_ok=False, min_clearance_m=0.15), self.cand)
        self.assertFalse(ok)
        self.assertTrue(any("净空" in w for w in why))

    def test_astar_over_sage_boundary(self):
        self.assertTrue(imp.judge_route(chk(path_m=5.2), self.cand)[0])           # 5.2 / 4.0 = 1.3 整，不超
        ok, why = imp.judge_route(chk(path_m=5.21), self.cand)
        self.assertFalse(ok)
        self.assertTrue(any("A* 长度" in w for w in why))

    def test_goal_too_close(self):
        near = dict(self.cand, goal=(1.9, 0.0), sage_path_m=2.0)
        ok, why = imp.judge_route(chk(path_m=2.0), near)
        self.assertFalse(ok)
        self.assertTrue(any("直线" in w for w in why))
        self.assertTrue(imp.judge_route(chk(path_m=2.1), dict(near, goal=(2.0, 0.0)))[0])    # 恰好 2.0 m 放行

    def test_multiple_reasons_listed(self):
        ok, why = imp.judge_route(chk(clear_ok=False, path_m=9.0), self.cand)
        self.assertGreaterEqual(len(why), 2)

    def test_every_reason_has_a_label(self):
        ok, why = imp.judge_route(chk(feasible=False, reason="A* 无路径", spawn_free=False, waypoint_free=False, clear_ok=False,
                                      path_m=9.0), dict(self.cand, goal=(0.5, 0.0)))
        self.assertEqual(len(why), 6)
        self.assertEqual(sorted({imp.reason_label(w) for w in why}),
                         sorted(label for _, label in imp.REASON_LABELS))
        with self.assertRaises(ValueError):
            imp.reason_label("没登记过的原因")


class TestEvaluate(unittest.TestCase):
    def test_margin_and_terrain_passed_through(self):
        seen = {}

        def spy(planner, spawn, waypoint, **kw):
            seen.update(kw, spawn=spawn, waypoint=waypoint)
            return chk()

        real = planner_mod.route_check
        planner_mod.route_check = spy
        try:
            cand = {"start": (1.0, 2.0), "goal": (5.0, 2.0), "yaw_deg": 30.0, "sage_path_m": 4.2, "trajectory_id": 4,
                    "split": "train"}
            res = imp.evaluate_route(object(), cand, z=1.15, terrain_occ="T")
        finally:
            planner_mod.route_check = real
        self.assertEqual(seen["margin_m"], imp.ROUTE_CLEAR_MARGIN_M)
        self.assertEqual(seen["terrain_occ"], "T")
        self.assertEqual(seen["route_name"], "tr_4")
        self.assertEqual(seen["spawn"], {"x": 1.0, "y": 2.0, "z": 1.15, "yaw_deg": 30.0})
        self.assertEqual(seen["waypoint"], {"x": 5.0, "y": 2.0, "z": 1.15})
        self.assertTrue(res["ok"])
        self.assertEqual(res["reasons"], [])

    def test_real_room(self):
        """6 m × 6 m 封闭房间（墙 = 一圈小高斯，z=1.2）：房内两点可达且净空够；房外到房内无路径。"""
        walls = [((0, 0), (6, 0)), ((6, 0), (6, 6)), ((6, 6), (0, 6)), ((0, 6), (0, 0))]
        means, scales, op = wall_gaussians(walls)
        occ = collision.Occupancy(means, scales, None, op, z_range=(1.0, 1.4), voxel_m=0.1)
        pl = planner_mod.OccPlanner(occ)
        inside = {"start": (1.0, 3.0), "goal": (5.0, 3.0), "yaw_deg": 0.0, "sage_path_m": 4.0, "trajectory_id": 1,
                  "split": "train"}
        r = imp.evaluate_route(pl, inside, z=1.2)
        self.assertTrue(r["ok"], r["reasons"])
        self.assertAlmostEqual(r["chk"]["path_m"], 4.0, delta=0.3)
        outside = dict(inside, start=(-0.5, 3.0))
        r2 = imp.evaluate_route(pl, outside, z=1.2)
        self.assertFalse(r2["ok"])
        self.assertIn("A* 无路径", r2["reasons"])


def make_scene(ground_z=-0.05, n_routes=2):
    routes = []
    for i in range(n_routes):
        routes.append({"name": f"tt_{i}", "spawn": {"x": float(i), "y": 1.0, "z": ground_z + imp.FLY_H_M, "yaw_deg": 10.0},
                       "waypoint": {"x": 5.0, "y": 1.0, "z": ground_z + imp.FLY_H_M}, "max_steps": 40 + i,
                       "note": "n", "sage": {"trajectory_id": i}})
    return imp.build_scene("841150", ground_z, routes, extra_notes=["frame ok"])


class TestScene(unittest.TestCase):
    def test_rz_180_exact(self):
        self.assertEqual(imp.rz_matrix(180.0), [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        m = np.array(imp.rz_matrix(90.0))
        np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_scene_fields(self):
        s = make_scene()
        self.assertEqual(s["schema"], "uavrgb.3dgs_scene.v1")
        self.assertEqual(s["name"], "sage3d_841150")
        self.assertEqual(s["ply"], "scene.ply")
        self.assertEqual(s["nav_to_world"], imp.rz_matrix(180.0))
        self.assertAlmostEqual(s["ground_z"], -0.05)
        self.assertEqual(s["camera"]["width"], 320)
        self.assertEqual(s["camera"]["height"], 240)
        self.assertEqual(s["camera"]["fov_deg"], 60.0)
        self.assertEqual(s["collision"]["source"], "gaussians")
        self.assertEqual(s["collision"]["envelope"], "e1_mini")
        self.assertEqual(s["collision"]["voxel_m"], 0.1)
        self.assertIn("CC-BY-NC-4.0", s["license"])
        self.assertEqual(s["step_s"], 0.5)
        self.assertEqual(s["fly_height_m"], imp.FLY_H_M)
        self.assertEqual(s["spawn"], s["routes"][0]["spawn"])
        self.assertEqual(s["waypoint"], s["routes"][0]["waypoint"])
        self.assertEqual(s["max_steps"], s["routes"][0]["max_steps"])
        self.assertTrue(any("frame ok" in n for n in s["notes"]))

    def test_flight_height_consistency(self):
        """z_slice 的中心、spawn.z、渲染高度同为 ground_z + 飞行高度：改配置区的 FLY_H_M 一处全跟着变。"""
        for g in (-0.05, 0.0, 0.31):
            s = make_scene(g)
            lo, hi = s["collision"]["z_slice"]
            self.assertAlmostEqual((lo + hi) / 2, g + imp.FLY_H_M, places=9)
            self.assertAlmostEqual(hi - lo, 2 * imp.Z_SLICE_HALF_M, places=9)
            self.assertAlmostEqual(s["spawn"]["z"] - s["ground_z"], imp.FLY_H_M, places=9)
        self.assertGreater(imp.FLY_H_M, 0.0)
        self.assertLess(imp.FLY_H_M, 2.5)           # E1 ToF 量程

    def test_scene_json_roundtrip_with_loader(self):
        with tempfile.TemporaryDirectory() as d:
            s = make_scene()
            open(os.path.join(d, "scene.ply"), "wb").close()
            scene_mod.save_scene(s, os.path.join(d, "scene.json"))
            loaded = scene_mod.load_scene(os.path.join(d, "scene.json"))
            applied = scene_mod.apply_route(loaded, 1)
            self.assertEqual(applied["route"]["name"], "tt_1")
            self.assertEqual(applied["max_steps"], 41)
            self.assertEqual(applied["spawn"]["x"], 1.0)


class TestFlightHeightIsPerScene(unittest.TestCase):
    """起飞高度 = spawn.z − ground_z（dynamics.CfSitlDynamics），跟着 scene.json 走；dynamics_profile 只取决于 e1_mini.json 的值与库版本。"""

    def _fake_proc(self, d):
        path = os.path.join(d, "fake_cf_sitl_proc")
        with open(path, "w") as f:
            f.write("#!/usr/bin/env python3\nimport json, sys\n"
                    "for line in sys.stdin:\n"
                    "    req = json.loads(line)\n"
                    "    print(json.dumps({'ok': True, 'state': {'lib_version': 31}}), flush=True)\n"
                    "    if req.get('cmd') == 'quit':\n        break\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
        return path

    def test_takeoff_height_and_profile(self):
        with tempfile.TemporaryDirectory() as d:
            fake = self._fake_proc(d)
            seen = {}
            for fly in (1.2, 2.0):
                scene = {"ground_z": -0.05, "spawn": {"x": 0, "y": 0, "z": -0.05 + fly, "yaw_deg": 0}}
                params = dyn_mod.CfSitlParams.from_json(None, lib_version=31)
                dyn = dyn_mod.CfSitlDynamics(scene, params=params, proc=fake)
                try:
                    seen[fly] = (dyn.takeoff_height, dyn.params.profile(), dyn.params.manifest()["profile"])
                finally:
                    dyn.close()
            self.assertAlmostEqual(seen[1.2][0], 1.2, places=9)
            self.assertAlmostEqual(seen[2.0][0], 2.0, places=9)
            self.assertEqual(seen[1.2][1:], seen[2.0][1:])
            self.assertRegex(seen[1.2][1], r"^cf_sitl_v1:[0-9a-f]{8}:lib31$")

    def test_scene_from_importer_gives_takeoff_1_2(self):
        with tempfile.TemporaryDirectory() as d:
            fake = self._fake_proc(d)
            s = make_scene(-0.05)
            dyn = dyn_mod.CfSitlDynamics(s, params=dyn_mod.CfSitlParams.from_json(None, lib_version=31), proc=fake)
            try:
                self.assertAlmostEqual(dyn.takeoff_height, imp.FLY_H_M, places=9)
            finally:
                dyn.close()


class TestSpread(unittest.TestCase):
    def test_spread_indices(self):
        self.assertEqual(imp.spread_indices(10, 4), [0, 3, 6, 9])
        self.assertEqual(imp.spread_indices(3, 4), [0, 1, 2])
        self.assertEqual(imp.spread_indices(1, 4), [0])
        self.assertEqual(imp.spread_indices(0, 4), [])
        self.assertEqual(len(set(imp.spread_indices(25, 8))), 8)


class TestDist(unittest.TestCase):
    def test_dist_summary(self):
        d = imp.dist([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual((d["n"], d["min"], d["p50"], d["max"]), (5, 1.0, 3.0, 5.0))
        self.assertEqual(imp.dist([])["n"], 0)


class FakeProc:
    def __init__(self, code):
        self.stdout = code


class TestDownload(unittest.TestCase):
    def _runner(self, script, size):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd)
            code, nbytes = script[min(len(calls) - 1, len(script) - 1)]
            dst = cmd[cmd.index("-o") + 1]
            with open(dst, "wb") as f:
                f.write(b"x" * nbytes)
            return FakeProc(code)
        return run, calls

    def test_backoff_on_429_then_ok(self):
        with tempfile.TemporaryDirectory() as d:
            run, calls = self._runner([("429", 10), ("429", 10), ("200", 100)], 100)
            sleeps = []
            imp.curl_download("https://example.invalid/a", os.path.join(d, "a.bin"), 100, run=run, sleep=sleeps.append)
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleeps, [imp.BACKOFF_S, 2 * imp.BACKOFF_S])
            self.assertEqual(os.path.getsize(os.path.join(d, "a.bin")), 100)
            self.assertIn("-L", calls[0])

    def test_size_mismatch_retries_then_raises(self):
        with tempfile.TemporaryDirectory() as d:
            run, calls = self._runner([("200", 50)], 100)
            with self.assertRaises(RuntimeError):
                imp.curl_download("https://example.invalid/a", os.path.join(d, "a.bin"), 100, run=run, sleep=lambda s: None)
            self.assertEqual(len(calls), imp.DOWNLOAD_TRIES)
            self.assertFalse(os.path.exists(os.path.join(d, "a.bin")))

    def test_existing_file_with_right_size_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.bin")
            open(p, "wb").write(b"x" * 100)
            run, calls = self._runner([("200", 100)], 100)
            imp.curl_download("https://example.invalid/a", p, 100, run=run, sleep=lambda s: None)
            self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main(verbosity=1)
