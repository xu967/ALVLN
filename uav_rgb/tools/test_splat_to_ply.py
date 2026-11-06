#!/usr/bin/env python3
"""splat_to_ply 回归：字节布局/四元数顺序、旋转同时作用到 means 与 quats（协方差一致）、
只允许行列式 +1、--flip-height 已删；真实 gsplat 渲染等价用 RUN_3DGS_REAL=1 开启。

  .venv-tools/bin/python tools/test_splat_to_ply.py
  RUN_3DGS_REAL=1 TORCH_EXTENSIONS_DIR=... $DATA_ROOT/uav_rgb/3dgs-venv/bin/python tools/test_splat_to_ply.py
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim", "3dgs"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import scene as scene_mod  # noqa: E402
import splat_to_ply as s2p  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP


def quat_to_rotmat(q):
    """[N,4] (w,x,y,z) -> [N,3,3]，与 gsplat/INRIA 同约定。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], axis=1)


def covariances(quats, scales):
    R = quat_to_rotmat(quats)
    S2 = np.asarray(scales, dtype=np.float64) ** 2
    return np.einsum("nij,nj,nkj->nik", R, S2, R)


def random_gaussians(rng, n):
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return {"means": rng.uniform(-2, 2, size=(n, 3)).astype(np.float32),
            "scales": np.exp(rng.uniform(np.log(0.02), np.log(0.4), size=(n, 3))).astype(np.float32),
            "colors": rng.uniform(0, 1, size=(n, 3)).astype(np.float32),
            "opacities": rng.uniform(0.3, 1.0, size=n).astype(np.float32),
            "quats": q.astype(np.float32)}


def write_splat(path, g):
    with open(path, "wb") as f:
        for i in range(len(g["means"])):
            rgba = [int(round(v * 255)) for v in g["colors"][i]] + [int(round(g["opacities"][i] * 255))]
            rot = [int(np.clip(round(v * 128 + 128), 0, 255)) for v in g["quats"][i]]
            f.write(struct.pack("<6f4B4B", *g["means"][i].tolist(), *g["scales"][i].tolist(),
                                *rgba, *rot))


class TestBytesAndQuatOrder(unittest.TestCase):
    def test_roundtrip_no_rotation_keeps_wxyz(self):
        """rot 字节 (w,x,y,z) 原样进 PLY rot_0..3：绕 z 90° 的 (√½,0,0,√½) 读回来还是它。"""
        c = np.sqrt(0.5)
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.splat")
            with open(src, "wb") as f:
                f.write(struct.pack("<6f4B4B", 1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
                                    10, 20, 30, 200, int(c * 128 + 128), 128, 128, int(c * 128 + 128)))
                f.write(struct.pack("<6f4B4B", -1.0, 0.0, 0.5, 0.02, 0.03, 0.04,
                                    255, 0, 128, 100, 255, 128, 128, 128))
            dst = os.path.join(d, "a.ply")
            s2p.convert(src, dst, up="z", scale=1.0)
            g = scene_mod.read_gaussian_ply(dst)
        self.assertEqual(g["means"].shape, (2, 3))
        np.testing.assert_allclose(g["means"][0], [1, 2, 3], atol=1e-5)
        np.testing.assert_allclose(g["scales"][0], [0.1, 0.2, 0.3], atol=1e-5)
        np.testing.assert_allclose(g["colors"][0], [10 / 255, 20 / 255, 30 / 255], atol=1e-5)
        self.assertAlmostEqual(float(g["opacities"][0]), 200 / 255, places=4)
        np.testing.assert_allclose(g["quats"][0], [c, 0, 0, c], atol=2e-3)
        np.testing.assert_allclose(g["quats"][1], [1, 0, 0, 0], atol=1e-6)
        # 一个绕 z 90° 的高斯：它的最长轴（scale_1=0.2 对应局部 y）在世界里应落到 -x
        R = quat_to_rotmat(g["quats"][:1])[0]
        np.testing.assert_allclose(R @ [0, 1, 0], [-1, 0, 0], atol=5e-3)


class TestRotationIsProperAndShared(unittest.TestCase):
    def test_up_rotations_are_proper(self):
        for up in ("y", "-y", "z"):
            R = s2p.rotation_for_up(up)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)
        np.testing.assert_allclose(s2p.rotation_for_up("y") @ [0, 1, 0], [0, 0, 1])     # y-up -> z-up
        np.testing.assert_allclose(s2p.rotation_for_up("-y") @ [0, -1, 0], [0, 0, 1])   # y-down -> z-up
        # --up -y 就是 --up y 之后再绕 x 转 180°（正常旋转），不是翻 z 的镜像
        flip_x_180 = np.diag([1.0, -1.0, -1.0])
        np.testing.assert_allclose(flip_x_180 @ s2p.rotation_for_up("y"), s2p.rotation_for_up("-y"), atol=1e-12)
        old_flip = np.diag([1.0, 1.0, -1.0]) @ s2p.rotation_for_up("y")
        self.assertAlmostEqual(np.linalg.det(old_flip), -1.0, places=12)
        with self.assertRaises(ValueError):
            s2p.rotation_for_up("x")

    def test_rotmat_quat_roundtrip(self):
        rng = np.random.default_rng(1)
        for up in ("y", "-y", "z"):
            R = s2p.rotation_for_up(up)
            np.testing.assert_allclose(quat_to_rotmat(s2p.rotmat_to_quat(R)[None])[0], R, atol=1e-12)
        q = rng.normal(size=(50, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        for qi in q[:10]:
            Ri = quat_to_rotmat(qi[None])[0]
            back = s2p.rotmat_to_quat(Ri)
            self.assertAlmostEqual(abs(float(np.dot(back, qi))), 1.0, places=9)   # ±q 同一旋转
        # Hamilton 积 <-> 矩阵积
        p = q[0]
        Rp = quat_to_rotmat(p[None])[0]
        np.testing.assert_allclose(quat_to_rotmat(s2p.quat_multiply(p, q[1:]))[0],
                                   Rp @ quat_to_rotmat(q[1:2])[0], atol=1e-12)

    def test_covariance_follows_rotation_and_scale(self):
        """Σ' = s² R Σ Rᵀ、μ' = s R μ + t：旋转同时作用到 means 与 quats，scales 乘尺度。"""
        rng = np.random.default_rng(0)
        g = random_gaussians(rng, 300)
        cov_src = covariances(g["quats"], g["scales"])
        for up, scale, z_off in (("y", 1.0, 0.0), ("-y", 2.0, 0.516), ("z", 0.5, -1.0)):
            out = s2p.transform_gaussians(g, up=up, scale=scale, z_offset=z_off)
            R = s2p.rotation_for_up(up)
            want_means = (g["means"].astype(np.float64) @ R.T) * scale + np.array([0, 0, z_off])
            np.testing.assert_allclose(out["means"], want_means, atol=1e-5)
            want_cov = scale ** 2 * np.einsum("ij,njk,lk->nil", R, cov_src, R)
            np.testing.assert_allclose(covariances(out["quats"], out["scales"]), want_cov,
                                       rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(np.linalg.norm(out["quats"], axis=1), 1.0, atol=1e-6)
            np.testing.assert_allclose(out["scales"], g["scales"] * scale, rtol=1e-6)
        # 反例：旧实现只转中心不转四元数，协方差对不上（证明这条检查有牙）
        R = s2p.rotation_for_up("y")
        want_cov = np.einsum("ij,njk,lk->nil", R, cov_src, R)
        self.assertGreater(np.abs(covariances(g["quats"], g["scales"]) - want_cov).max(), 1e-2)

    def test_full_file_roundtrip_matches_transform(self):
        rng = np.random.default_rng(2)
        g = random_gaussians(rng, 64)
        with tempfile.TemporaryDirectory() as d:
            src, dst = os.path.join(d, "r.splat"), os.path.join(d, "r.ply")
            write_splat(src, g)
            read = s2p.read_splat(src)
            s2p.convert(src, dst, up="-y", scale=2.0, z_offset=0.5)
            ply = scene_mod.read_gaussian_ply(dst)
        want = s2p.transform_gaussians(read, up="-y", scale=2.0, z_offset=0.5)
        np.testing.assert_allclose(ply["means"], want["means"], atol=1e-5)
        np.testing.assert_allclose(ply["scales"], want["scales"], rtol=1e-4)
        np.testing.assert_allclose(covariances(ply["quats"], ply["scales"]),
                                   covariances(want["quats"], want["scales"]), rtol=1e-3, atol=1e-6)
        np.testing.assert_allclose(ply["opacities"], read["opacities"], atol=2e-3)

    def test_cli_rejects_flip_height(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "a.splat")
            write_splat(src, random_gaussians(np.random.default_rng(3), 2))
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "splat_to_ply.py"),
                                src, os.path.join(d, "a.ply"), "--flip-height"],
                               capture_output=True, text=True, timeout=60)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--up -y", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(d, "a.ply")))
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "splat_to_ply.py"),
                                src, os.path.join(d, "b.ply"), "--up=-y", "--scale", "2"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertTrue(os.path.exists(os.path.join(d, "b.ply")))
            # 别名 ydown 等价于 -y（argparse 会把 "-y" 当选项，别名更省事）
            r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "splat_to_ply.py"),
                                src, os.path.join(d, "c.ply"), "--up", "ydown", "--scale", "2"],
                               capture_output=True, text=True, timeout=60)
            self.assertEqual(r.returncode, 0, r.stderr)
            b = scene_mod.read_gaussian_ply(os.path.join(d, "b.ply"))
            c = scene_mod.read_gaussian_ply(os.path.join(d, "c.ply"))
            np.testing.assert_allclose(b["means"], c["means"])
            np.testing.assert_allclose(b["quats"], c["quats"])


@unittest.skipUnless(os.environ.get("RUN_3DGS_REAL") == "1",
                     "设 RUN_3DGS_REAL=1 才跑真实 gsplat 渲染等价（需要 3dgs-venv + GPU）")
class TestRenderEquivalence(unittest.TestCase):
    """转换前后同一物理相机渲染一致；只转中心不转四元数（旧实现）则明显不一致。"""

    def _render(self, g, viewmat, W=96, H=72):
        import torch
        from gsplat import rasterization
        dev = torch.device("cuda")
        K = torch.tensor([[80.0, 0, W / 2], [0, 80.0, H / 2], [0, 0, 1]], device=dev)[None]
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float().to(dev)  # noqa: E731
        with torch.no_grad():
            colors, alphas, _ = rasterization(
                t(g["means"]), t(g["quats"]), t(g["scales"]), t(g["opacities"]), t(g["colors"]),
                t(viewmat)[None], K, W, H, render_mode="RGB+ED", packed=True)
        return colors[0, ..., :3].cpu().numpy(), colors[0, ..., 3].cpu().numpy(), alphas[0, ..., 0].cpu().numpy()

    def test_same_image_after_transform(self):
        rng = np.random.default_rng(5)
        g = random_gaussians(rng, 400)
        # 源系：相机在原点、按 OpenCV 约定看 +z；高斯放在相机前 2..6 m
        g["means"] = np.stack([rng.uniform(-1.5, 1.5, 400), rng.uniform(-1.2, 1.2, 400),
                               rng.uniform(2.0, 6.0, 400)], 1).astype(np.float32)
        g["scales"] = np.exp(rng.uniform(np.log(0.05), np.log(0.6), size=(400, 3))).astype(np.float32)
        g["scales"][:, 0] *= 0.15   # 扁平高斯：朝向错了画面立刻不一样
        rgb_src, depth_src, alpha_src = self._render(g, np.eye(4))
        self.assertGreater(alpha_src.mean(), 0.3)
        for up, scale, z_off in (("y", 1.0, 0.0), ("-y", 2.0, 0.516)):
            out = s2p.transform_gaussians(g, up=up, scale=scale, z_offset=z_off)
            R, tvec = out["rotation"], np.array([0.0, 0.0, z_off])
            # p_nav = s R p_src + t  =>  s·p_src = Rᵀ p_nav − Rᵀ t；源相机（viewmat=I）在 nav 里的 viewmat：
            view = np.eye(4)
            view[:3, :3] = R.T
            view[:3, 3] = -R.T @ tvec
            rgb_nav, depth_nav, alpha_nav = self._render(out, view)
            self.assertLess(np.abs(rgb_nav - rgb_src).max(), 2e-3, (up, scale))
            self.assertLess(np.abs(alpha_nav - alpha_src).max(), 2e-3, (up, scale))
            m = alpha_src > 0.5
            np.testing.assert_allclose(depth_nav[m], depth_src[m] * scale, rtol=2e-3, atol=1e-3)
            # 反例：中心转了、四元数不转（旧 splat_to_ply 的做法）
            bad = dict(out, quats=g["quats"])
            rgb_bad, _, _ = self._render(bad, view)
            self.assertGreater(np.abs(rgb_bad - rgb_src).mean(), 1e-2, (up, scale))


if __name__ == "__main__":
    unittest.main()
