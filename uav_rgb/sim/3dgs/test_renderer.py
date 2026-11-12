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

    def test_read_gaussian_ply_rejects_non_float_property(self):
        """D10：以前只收 property float、其它类型静默跳过，二进制表会整体错位读成垃圾；现在直接报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "rgb.ply")
            header = ["ply", "format ascii 1.0", "element vertex 1", "property float x", "property float y",
                      "property float z", "property uchar red", "property float f_dc_0", "property float f_dc_1",
                      "property float f_dc_2", "property float opacity", "end_header"]
            with open(path, "w", encoding="ascii") as f:
                f.write("\n".join(header) + "\n0 0 0 255 0 0 0 1\n")
            with self.assertRaises(ValueError) as cm:
                scene_mod.read_gaussian_ply(path)
            self.assertIn("uchar", str(cm.exception))

    def test_render_options_and_scale_mask(self):
        self.assertEqual(scene_mod.render_options({}), scene_mod.RENDER_DEFAULTS)
        opts = scene_mod.render_options({"render": {"sh_degree": 0, "bogus": 1}})
        self.assertEqual(opts["sh_degree"], 0)
        self.assertNotIn("bogus", opts)
        self.assertEqual(opts["max_scale_m"], scene_mod.RENDER_DEFAULTS["max_scale_m"])
        scales = np.array([[0.1, 0.1, 0.1], [0.1, 5.0, 0.1], [2.9, 0.2, 0.2]], np.float32)
        np.testing.assert_array_equal(scene_mod.scale_keep_mask(scales, 3.0), [True, False, True])
        np.testing.assert_array_equal(scene_mod.scale_keep_mask(scales, None), [True, True, True])


class TestClipPlanes(unittest.TestCase):
    """D10：gsplat 默认近/远裁剪 0.01 / 1e10，Gazebo 学生相机是 0.05 / 50 m（uav_rgb_e1.sdf），
    贴脸 1–5 cm 的高斯只会出现在 3DGS 画面里。renderer 固定传 Gazebo 的值并写进 stats()。"""

    def test_render_view_passes_gazebo_clip_planes(self):
        self.assertEqual((renderer_mod.NEAR_PLANE_M, renderer_mod.FAR_PLANE_M), (0.05, 50.0))
        try:
            import torch   # noqa: F401
        except ImportError:
            self.skipTest("无 torch（3dgs-venv）")
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=0, length=12.0)
            scene = scene_mod.load_scene(os.path.join(tmp, "scene.json"))
            r = renderer_mod.GsplatRenderer(scene, device="cpu")      # 不碰 GPU：rasterization 换成记参桩
            seen = {}

            def fake_rasterization(means, quats, scales, opacities, colors, vm, ks, w, h, **kw):
                seen.update(kw)
                return r.torch.zeros(1, h, w, 4), r.torch.zeros(1, h, w, 1), {}

            r.rasterization = fake_rasterization
            r.render_pose(1.0, 0.0, 2.0, 0.0)
            self.assertEqual((seen["near_plane"], seen["far_plane"]), (0.05, 50.0))
            st = r.stats()
            self.assertEqual((st["near_plane_m"], st["far_plane_m"]), (0.05, 50.0))


class TestDepthOpacityMin(unittest.TestCase):
    """B5：GsplatRenderer 可选 depth_opacity_min（缺省 None 逐位不变；给定门限时 RGB 仍用全部高斯，深度只用 opacity >= 门限的高斯另渲一遍）。"""

    def test_default_none_calls_rasterization_once_and_with_threshold_renders_depth_from_filtered_subset(self):
        try:
            import torch
        except ImportError:
            self.skipTest("无 torch（3dgs-venv）")
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=0, length=12.0)
            sjson = os.path.join(tmp, "scene.json")
            scene = scene_mod.load_scene(sjson)
            g = scene_mod.read_gaussian_ply(scene["ply_path"])
            # 造两批高斯：半透明雾 op=0.10（近距 z=1.0 m）、实体墙 op=0.80（远距 z=3.5 m）
            opac = np.where(np.arange(len(g["opacities"])) % 2 == 0, 0.10, 0.80).astype(np.float32)
            scene_mod.write_gaussian_ply(scene["ply_path"], g["means"], g["colors"], g["scales"], g["quats"], opac)
            scene = scene_mod.load_scene(sjson)

            calls = []

            def fake_rast(means, quats, scales, opacities, colors, vm, ks, w, h, **kw):
                calls.append({"n": int(len(opacities)), "min_op": float(opacities.min().item()) if len(opacities) else None,
                              "mode": kw.get("render_mode")})
                # 含 op<0.3 的雾时返回近深度 1.0 m、RGB=0.2；只剩 op>=0.3 的实体时返回真实深度 3.5 m、RGB=0.9
                has_fog = len(opacities) > 0 and float(opacities.min().item()) < 0.30
                z_val = 1.0 if has_fog else 3.5
                c_val = 0.2 if has_fog else 0.9
                mode = kw.get("render_mode", "RGB+ED")
                if mode == "ED":
                    out = torch.full((1, h, w, 1), z_val, dtype=torch.float32)
                else:
                    out = torch.zeros((1, h, w, 4), dtype=torch.float32)
                    out[..., :3] = c_val * 0.8
                    out[..., 3] = z_val
                alp = torch.full((1, h, w, 1), 0.8, dtype=torch.float32)
                return out, alp, {}

            r_def = renderer_mod.GsplatRenderer(scene, device="cpu", supersample=1)
            r_def.rasterization = fake_rast
            rgb0, dep0, _ = r_def.render_pose(1.0, 0.0, 2.0, 0.0)
            self.assertEqual(len(calls), 1, "缺省 depth_opacity_min=None 只光栅化一次")
            self.assertNotIn("depth_opacity_min", r_def.stats())

            calls.clear()
            r_flt = renderer_mod.GsplatRenderer(scene, device="cpu", supersample=1, depth_opacity_min=0.30)
            r_flt.rasterization = fake_rast
            rgb1, dep1, _ = r_flt.render_pose(1.0, 0.0, 2.0, 0.0)
            self.assertEqual(len(calls), 2, "给 depth_opacity_min=0.30 时 RGB 全量 + 深度过滤子集各一次")
            self.assertGreaterEqual(calls[1]["min_op"], 0.30 - 1e-5)
            self.assertLess(calls[1]["n"], calls[0]["n"])
            np.testing.assert_array_equal(rgb0, rgb1, err_msg="RGB 画面与缺省逐像素完全相同")
            self.assertAlmostEqual(float(dep0[0, 0]), 1.0, places=5)
            self.assertAlmostEqual(float(dep1[0, 0]), 3.5, places=5)
            self.assertEqual(r_flt.stats()["depth_opacity_min"], 0.30)

            # set_depth_opacity_min 可在同实例上切换回 None（复用同一份 GPU 基础张量）
            calls.clear()
            r_flt.set_depth_opacity_min(None)
            rgb2, dep2, _ = r_flt.render_pose(1.0, 0.0, 2.0, 0.0)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(dep0, dep2)

            for bad in (-0.1, 1.1, float("nan"), True):
                with self.assertRaises(ValueError):
                    renderer_mod.GsplatRenderer(scene, device="cpu", depth_opacity_min=bad)


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
