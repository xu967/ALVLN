#!/usr/bin/env python3
"""渲染器测试：mock 尺寸/深度、viewmat/K、PLY 读写；真实 gsplat 用环境变量开启。"""
import importlib
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
scene_mod = importlib.import_module("sim.3dgs.scene")
renderer_mod = importlib.import_module("sim.3dgs.renderer")


class TestMockRenderer(unittest.TestCase):
    def test_shapes_and_timing(self):
        r = renderer_mod.MockRenderer(width=320, height=240, fov_deg=60)
        rgb, depth, ms = r.render_pose(1.0, 2.0, 2.0, 30.0)
        self.assertEqual(rgb.shape, (240, 320, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertEqual(depth.shape, (240, 320))
        self.assertEqual(depth.dtype, np.float32)
        self.assertTrue(np.all(depth == 5.0))
        self.assertGreaterEqual(ms, 0.0)

    def test_160x120(self):
        r = renderer_mod.MockRenderer(width=160, height=120)
        rgb, depth, _ = r.render_pose(0, 0, 2, 0)
        self.assertEqual(rgb.shape, (120, 160, 3))
        self.assertEqual(depth.shape, (120, 160))


class TestGeometry(unittest.TestCase):
    def test_viewmat_forward(self):
        # 相机在 (0,0,2) 朝 +x：x+1 的点应在相机前方 +z（gsplat/COLMAP 约定）。
        v = scene_mod.viewmat_from_pose(0, 0, 2, 0)
        p = np.array([1.0, 0.0, 2.0, 1.0])
        q = v @ p
        np.testing.assert_allclose(q[:3], [0.0, 0.0, 1.0], atol=1e-9)

    def test_yaw_left_turns_camera(self):
        v = scene_mod.viewmat_from_pose(0, 0, 2, 90)
        q = v @ np.array([0.0, 1.0, 2.0, 1.0])   # +y 方向、与相机同高
        np.testing.assert_allclose(q[:3], [0.0, 0.0, 1.0], atol=1e-9)

    def test_intrinsics(self):
        k = scene_mod.intrinsics(320, 240, 60)
        self.assertAlmostEqual(k[0, 0], 160 / np.tan(np.radians(30)), places=6)
        self.assertEqual((k[0, 2], k[1, 2]), (160.0, 120.0))

    def test_ply_roundtrip(self):
        rng = np.random.default_rng(0)
        means = rng.normal(size=(32, 3)).astype(np.float32)
        colors = rng.uniform(0, 1, size=(32, 3)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.ply")
            scene_mod.write_gaussian_ply(path, means, colors)
            g = scene_mod.read_gaussian_ply(path)
        np.testing.assert_allclose(g["means"], means, atol=1e-5)
        np.testing.assert_allclose(g["colors"], colors, atol=2e-3)
        self.assertTrue(np.all(g["opacities"] > 0.5))


@unittest.skipUnless(os.environ.get("RUN_3DGS_REAL") == "1",
                     "设 RUN_3DGS_REAL=1 才跑真实 gsplat（首次会 JIT 编译）")
class TestGsplatReal(unittest.TestCase):
    def test_two_resolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=0, length=12.0)
            scene = scene_mod.load_scene(os.path.join(tmp, "scene.json"))
            r = renderer_mod.GsplatRenderer(scene, device="cuda")
            for w, h in ((320, 240), (160, 120)):
                rgb, depth, ms = r.render_pose(1.0, 0.0, 2.0, 0.0, width=w, height=h)
                self.assertEqual(rgb.shape, (h, w, 3))
                self.assertEqual(depth.shape, (h, w))
                self.assertTrue(np.isfinite(depth).all())
                self.assertGreater((depth > 0).mean(), 0.2)
                print(f"{w}x{h}: {ms:.2f} ms")
            print(r.stats())


if __name__ == "__main__":
    unittest.main()
