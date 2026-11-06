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
        # write_gaussian_ply 补零的 f_rest 读回来是 sh_degree=3、高阶系数全 0、DC 与 colors 一致
        self.assertEqual(g["sh_degree"], 3)
        self.assertEqual(g["sh"].shape, (32, 16, 3))
        self.assertTrue(np.all(g["sh"][:, 1:] == 0))
        np.testing.assert_allclose(g["sh"][:, 0] * scene_mod.C0 + 0.5, colors, atol=2e-3)

    def test_render_options_and_scale_mask(self):
        self.assertEqual(scene_mod.render_options({}), scene_mod.RENDER_DEFAULTS)
        opts = scene_mod.render_options({"render": {"sh_degree": 0, "bogus": 1}})
        self.assertEqual(opts["sh_degree"], 0)
        self.assertNotIn("bogus", opts)
        self.assertEqual(opts["max_scale_m"], scene_mod.RENDER_DEFAULTS["max_scale_m"])
        scales = np.array([[0.1, 0.1, 0.1], [0.1, 5.0, 0.1], [2.9, 0.2, 0.2]], np.float32)
        np.testing.assert_array_equal(scene_mod.scale_keep_mask(scales, 3.0), [True, False, True])
        np.testing.assert_array_equal(scene_mod.scale_keep_mask(scales, None), [True, True, True])


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

    def test_sh_supersample_and_scale_filter(self):
        """新默认（sh3/ss2/max_scale）与旧路径（sh0/ss1/不过滤）的关系可验证：
        f_rest 全 0 时 sh3 与 sh0 同图；ss2 = 2 倍图 2x2 盒滤波；尺度过滤把巨型高斯去掉。"""
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=0, length=12.0)
            scene = scene_mod.load_scene(os.path.join(tmp, "scene.json"))
            # 加一个 1σ=50 m 的巨型高斯盖住整个走廊（模拟 Callinstrasse 的"天空壳"）。
            # 中心要在相机前方（gsplat 按中心深度排序/剔除，放在相机后面会被整个剔掉）。
            g = scene_mod.read_gaussian_ply(scene["ply_path"])
            means = np.vstack([g["means"], [[2.0, 0.0, 2.0]]]).astype(np.float32)
            colors = np.vstack([g["colors"], [[1.0, 1.0, 1.0]]]).astype(np.float32)
            scales = np.vstack([g["scales"], [[50.0, 50.0, 50.0]]]).astype(np.float32)
            quats = np.vstack([g["quats"], [[1.0, 0.0, 0.0, 0.0]]]).astype(np.float32)
            opac = np.concatenate([g["opacities"], [0.6]]).astype(np.float32)
            scene_mod.write_gaussian_ply(scene["ply_path"], means, colors, scales, quats, opac)
            pose = (1.0, 0.0, 2.0, 0.0)
            old = renderer_mod.GsplatRenderer(scene, sh_degree=0, max_scale_m=0, supersample=1)
            sh3 = renderer_mod.GsplatRenderer(scene, sh_degree=3, max_scale_m=0, supersample=1)
            ss2 = renderer_mod.GsplatRenderer(scene, sh_degree=0, max_scale_m=0, supersample=2)
            filt = renderer_mod.GsplatRenderer(scene, sh_degree=0, max_scale_m=3.0, supersample=1)
            new = renderer_mod.GsplatRenderer(scene)
            self.assertEqual((old.sh_degree, sh3.sh_degree, new.sh_degree), (0, 3, 3))
            self.assertEqual((old.n_removed_by_scale, filt.n_removed_by_scale, new.n_removed_by_scale),
                             (0, 1, 1))
            self.assertEqual(new.supersample, scene_mod.RENDER_DEFAULTS["supersample"])
            rgb_old, d_old, _ = old.render_pose(*pose)
            rgb_sh3, _, _ = sh3.render_pose(*pose)
            rgb_ss2, d_ss2, _ = ss2.render_pose(*pose)
            rgb_filt, _, _ = filt.render_pose(*pose)
            rgb_new, d_new, _ = new.render_pose(*pose)
            big, _, _ = old.render_pose(*pose, width=640, height=480)
            # f_rest 全 0：sh3 与 sh0 同图（允许量化 1 级）
            self.assertLessEqual(np.abs(rgb_sh3.astype(int) - rgb_old.astype(int)).max(), 1)
            # ss2 ≈ 2 倍渲染后 2x2 平均（gsplat 的像素覆盖在两种分辨率下有细微差异，看均值）
            pooled = big.reshape(240, 2, 320, 2, 3).mean(axis=(1, 3))
            self.assertLess(np.abs(rgb_ss2.astype(float) - pooled).mean(), 3.0)
            self.assertEqual(rgb_ss2.shape, (240, 320, 3))
            self.assertEqual(d_ss2.shape, (240, 320))
            # 巨型高斯是半透明白壳：去掉后画面更暗、对比更高
            self.assertLess(rgb_filt.mean(), rgb_old.mean() - 5)
            self.assertGreater(rgb_filt.std(), rgb_old.std())
            self.assertEqual(rgb_new.shape, (240, 320, 3))
            self.assertTrue(np.isfinite(d_new).all())
            self.assertGreater((d_new > 0).mean(), 0.2)
            print("old", rgb_old.mean(), "filt", rgb_filt.mean(), "new", rgb_new.mean(), new.stats())


if __name__ == "__main__":
    unittest.main()
