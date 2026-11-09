#!/usr/bin/env python3
"""scene_import_wanderland 的纯函数测试（不下载、不渲染）：usda 网格解析、网格 PLY 写出 ->
collision.read_ply_vertices 回读、nav 轴推导（y-down 世界系）、局部地面高度。

运行：$DATA_ROOT/uav_rgb/3dgs-venv/bin/python -m unittest tools.test_scene_import_wanderland
"""
import importlib
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
imp = importlib.import_module("scene_import_wanderland")
col = importlib.import_module("sim.3dgs.collision")

USDA = """#usda 1.0
def Xform "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"])
{
    float3 xformOp:rotateXYZ = (-90, 0, 180)
    float3 xformOp:scale = (1, 1, 1)
    double3 xformOp:translate = (0, 0, 0)
    def Mesh "mesh"
    {
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
        point3f[] points = [(5, -1, 1.5), (5, 1, 1.5), (5, 1, 2.5), (5, -1, 2.5)]
    }
}
"""


class TestUsda(unittest.TestCase):
    def test_parse_points_faces_xform(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "scene.usda")
            with open(p, "w", encoding="utf-8") as f:
                f.write(USDA)
            pts, faces, xf = imp.usda_mesh_to_arrays(p)
        self.assertEqual(pts.shape, (4, 3))
        self.assertTrue(np.allclose(pts[2], [5, 1, 2.5]))
        self.assertTrue(np.array_equal(faces, [[0, 1, 2], [0, 2, 3]]))
        self.assertEqual(xf["rotateXYZ"], [-90.0, 0.0, 180.0])

    def test_rejects_quads(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "scene.usda")
            with open(p, "w", encoding="utf-8") as f:
                f.write(USDA.replace("[3, 3]", "[4, 2]"))
            with self.assertRaises(ValueError):
                imp.usda_mesh_to_arrays(p)

    def test_write_ply_roundtrip(self):
        pts = np.array([[5, -1, 1.5], [5, 1, 1.5], [5, 1, 2.5], [5, -1, 2.5]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "m.ply")
            imp.write_mesh_ply(p, pts, faces)
            r = col.read_ply_vertices(p)
        self.assertTrue(np.allclose(r["xyz"], pts))
        self.assertTrue(np.array_equal(r["faces"], faces))


class TestAlign(unittest.TestCase):
    def test_nav_axes_from_y_down_world(self):
        # 世界 up = -y；相机沿世界 -z 走 50 m（x 几乎不变）：nav x 应是 -z，nav z = -y，nav y = z×x = +x
        centres = np.stack([np.random.default_rng(0).normal(0, 0.3, 50), np.full(50, -0.2), -np.linspace(0, 50, 50)], 1)
        m = imp.nav_to_world_from_up([0, -1, 0], centres)
        R = m[:3, :3]
        self.assertTrue(np.allclose(R[:, 2], [0, -1, 0]))          # nav z 在世界系
        self.assertTrue(np.allclose(R[:, 0], [0, 0, -1]))          # nav x：轨迹方向
        self.assertTrue(np.allclose(R[:, 1], np.cross(R[:, 2], R[:, 0])))
        self.assertAlmostEqual(np.linalg.det(R), 1.0)              # 右手、纯旋转
        # nav 系里相机应从 x≈0 走到 x≈50，z 为常数
        nav = (np.linalg.inv(m)[:3, :3] @ centres.T).T
        self.assertAlmostEqual(nav[-1, 0] - nav[0, 0], 50.0, places=6)
        self.assertTrue(np.allclose(nav[:, 2], 0.2))

    def test_local_ground_height_percentile(self):
        rng = np.random.default_rng(1)
        n = 2000
        xy = rng.uniform(-5, 5, (n, 2))
        h = np.where(rng.uniform(size=n) < 0.8, rng.normal(0.0, 0.02, n), rng.uniform(0.5, 3.0, n))  # 地面 + 高处杂物
        g = imp.local_ground_height(h, xy, np.array([[0.0, 0.0], [3.0, -3.0], [40.0, 40.0]]), radius=1.5, pct=10)
        self.assertLess(abs(g[0]), 0.05)
        self.assertLess(abs(g[1]), 0.05)
        self.assertTrue(np.isnan(g[2]))                            # 远处没有点 -> NaN，调用方必须处理


class TestProbe(unittest.TestCase):
    def test_step_probe_delegates_to_scene_probe_quality_and_adds_ground_depth(self):
        """step_probe 调 tools/scene_probe_quality.run（判据/阈值单源，R2-3DGS #4），只加俯视高度复核，
        结果并进同一份 OUT/probe_quality/routes_quality.json；mock 渲染器 + 合成走廊，不需要 GPU。"""
        import json
        scene_mod = importlib.import_module("sim.3dgs.scene")
        renderer_mod = importlib.import_module("sim.3dgs.renderer")
        pq = importlib.import_module("scene_probe_quality")
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(os.path.join(tmp, "scene"), seed=0)
            real = renderer_mod.make_renderer
            renderer_mod.make_renderer = lambda scene, mock=False, *a, **k: real(scene, True, *a, **k)
            saved = imp.SCENE_JSON, imp.OUT
            imp.SCENE_JSON, imp.OUT = os.path.join(tmp, "scene", "scene.json"), os.path.join(tmp, "out")
            try:
                with self.assertRaises(SystemExit):           # mock 渐变帧 lapvar≈0.2 全判低质：门槛不过要停，但 JSON 先落盘
                    imp.step_probe()
            finally:
                renderer_mod.make_renderer = real
                imp.SCENE_JSON, imp.OUT = saved
            with open(os.path.join(tmp, "out", "probe_quality", "routes_quality.json"), encoding="utf-8") as f:
                rep = json.load(f)
            self.assertEqual(rep["schema"], "uavrgb.3dgs_routes_quality.v1")
            self.assertEqual(rep["lapvar_min_threshold"], pq.FRAME_LAPVAR_MIN)
            r0 = rep["routes"][0]
            self.assertEqual((r0["pass"], r0["low_frac"], r0["n_low"]), (False, 1.0, r0["n_frames"]))
            self.assertGreaterEqual(r0["n_frames"], 2)
            g = r0["ground_depth_3dgs_m"]                       # step_probe 独有：俯视 -90° 中央 1/3 期望深度中位
            self.assertEqual((g["min"], g["med"], g["max"]), (5.0, 5.0, 5.0))   # mock 深度常量 5 m
            self.assertGreaterEqual(g["n"], 1)
            self.assertFalse(os.path.exists(os.path.join(tmp, "out", "routes_quality.json")))   # 旧独立实现的产物不再写


if __name__ == "__main__":
    unittest.main()
