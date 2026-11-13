#!/usr/bin/env python3
"""碰撞代理 v2 测试（REVIEW #3 / AUDIT #2、#3）：纯 numpy，不需要 torch/scipy。

负例清单：
  * 薄墙穿越：起点和终点都在空域、线段中段穿墙 -> sweep 判碰（v1 只查终点漏掉）；
  * 幽灵墙：同几何、不透明度 0.05 -> 不判碰；
  * nav_to_world 非单位阵：世界系平移/旋转后的高斯，用导航系坐标查询结果不变；
  * 伸进高度范围的大高斯：中心在范围外、1σ 伸进来 -> 判碰（v1 只看中心漏掉）；
  * 膨胀形状：机体半径 0.1 m 的圆盘，不依赖 scipy。
"""
import importlib
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
col = importlib.import_module("sim.3dgs.collision")
scene_mod = importlib.import_module("sim.3dgs.scene")

E1 = dict(body_radius_m=0.102, body_half_height_m=0.028)


def thin_wall(x=5.0, y_half=1.0, z=2.0, n=11, opacity=0.95, thick=0.01, half_w=0.3):
    """x=常数 的薄墙：n 个扁高斯（x 向 1σ=thick，y/z 向 1σ=half_w）。"""
    ys = np.linspace(-y_half, y_half, n)
    means = np.array([(x, y, z) for y in ys], dtype=np.float64)
    scales = np.tile([thick, half_w, half_w], (n, 1))
    return means, scales, np.tile([1.0, 0, 0, 0], (n, 1)), np.full(n, opacity)


class TestSweepThinWall(unittest.TestCase):
    def setUp(self):
        m, s, q, o = thin_wall()
        self.occ = col.Occupancy(m, s, q, o, z_range=(1.65, 2.35), voxel_m=0.1, **E1)

    def test_endpoints_free_but_segment_crosses(self):
        p0, p1 = (4.6, 0.0, 2.0), (5.3, 0.0, 2.0)       # 0.7 m：终速一步能跨过的距离
        self.assertFalse(self.occ.collides(*p0))
        self.assertFalse(self.occ.collides(*p1))
        hit, pt = self.occ.sweep(p0, p1)
        self.assertTrue(hit)
        self.assertLess(abs(pt[0] - 4.9), 0.11, pt)      # 首个碰撞点贴着墙前的膨胀层

    def test_segment_parallel_to_wall_is_free(self):
        self.assertEqual(self.occ.sweep((4.5, -1.0, 2.0), (4.5, 1.0, 2.0)), (False, None))

    def test_wall_ends_and_body_radius(self):
        self.assertTrue(self.occ.collides(5.0, 1.25, 2.0))    # 1.5σ=0.45 + 机体 0.1
        self.assertFalse(self.occ.collides(5.0, 1.8, 2.0))
        self.assertAlmostEqual(self.occ.clearance(4.0, 0.0, 2.0), 1.0, delta=0.11)

    def test_ground(self):
        self.assertTrue(self.occ.collides(0.0, 0.0, 0.02))
        self.assertFalse(self.occ.collides(0.0, 0.0, 0.5))    # 高度范围外只查地面
        hit, pt = self.occ.sweep((0.0, 0.0, 2.0), (0.0, 0.0, 0.0))
        self.assertTrue(hit)
        self.assertLessEqual(pt[2], 0.028 + 0.051)


class TestGhostWall(unittest.TestCase):
    def test_low_opacity_wall_not_solid(self):
        m, s, q, o = thin_wall(opacity=0.05)
        occ = col.Occupancy(m, s, q, o, z_range=(1.65, 2.35), voxel_m=0.1, **E1)
        self.assertEqual(occ.n_kept, 0)
        self.assertFalse(occ.collides(5.0, 0.0, 2.0))
        self.assertEqual(occ.sweep((4.6, 0, 2), (5.3, 0, 2)), (False, None))

    def test_threshold_is_parameter(self):
        m, s, q, o = thin_wall(opacity=0.05)
        occ = col.Occupancy(m, s, q, o, opacity_min=0.01, z_range=(1.65, 2.35),
                            voxel_m=0.1, **E1)
        self.assertTrue(occ.collides(5.0, 0.0, 2.0))


class TestNavToWorld(unittest.TestCase):
    """同一物理场景换 nav_to_world 表示：导航系查询结果必须不变（AUDIT #2）。"""

    def _probe(self, occ):
        pts = [(4.6, 0.0, 2.0), (5.0, 0.0, 2.0), (5.3, 0.0, 2.0), (5.0, 1.25, 2.0),
               (5.0, 1.6, 2.0), (3.0, 0.5, 2.0)]
        return ([occ.collides(*p) for p in pts],
                occ.sweep((4.6, 0, 2), (5.3, 0, 2))[0],
                occ.sweep((4.5, -1, 2), (4.5, 1, 2))[0])

    def test_translation_and_rotation_consistent(self):
        m, s, q, o = thin_wall()
        ref = col.Occupancy(m, s, q, o, z_range=(1.65, 2.35), voxel_m=0.1, **E1)
        want = self._probe(ref)
        self.assertTrue(want[0][1] and want[1] and not want[0][0])
        yaw = math.radians(37.0)
        c, sn = math.cos(yaw), math.sin(yaw)
        r = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1]])
        t = np.array([10.0, -3.0, 0.7])
        m_w = m @ r.T + t                                  # 世界系高斯中心
        # 高斯朝向也跟着转：q_w = q_R ⊗ q（这里 q=单位，所以直接用 q_R）
        q_r = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        q_w = np.tile(q_r, (len(m), 1))
        n2w = np.eye(4)
        n2w[:3, :3], n2w[:3, 3] = r, t
        occ = col.Occupancy(m_w, s, q_w, o, nav_to_world=n2w, z_range=(1.65, 2.35),
                            voxel_m=0.1, **E1)
        self.assertEqual(self._probe(occ), want)
        # 不转高斯直接用世界坐标建（v1 的做法）会错位
        wrong = col.Occupancy(m_w, s, q_w, o, z_range=(1.65, 2.35), voxel_m=0.1, **E1)
        self.assertNotEqual(self._probe(wrong), want)

    def test_rejects_non_rigid(self):
        m, s, q, o = thin_wall()
        projective = np.eye(4)
        projective[3, 0] = 0.1
        for bad in (np.diag([2.0, 1.0, 1.0, 1.0]), projective):
            with self.subTest(transform=bad.tolist()), self.assertRaisesRegex(ValueError, "nav_to_world"):
                col.Occupancy(m, s, q, o, nav_to_world=bad, **E1)


class TestEllipsoidExtent(unittest.TestCase):
    def test_big_gaussian_reaching_into_z_range(self):
        # 中心 z=1.0（范围外），竖向 1.5σ=0.9 -> 伸到 1.9，必须占据。v1 只看中心漏掉。
        means = np.array([[5.0, 0.0, 1.0]])
        scales = np.array([[0.2, 0.2, 0.6]])
        occ = col.Occupancy(means, scales, None, np.array([0.9]),
                            z_range=(1.65, 2.35), voxel_m=0.1, **E1)
        self.assertEqual(occ.n_in_range, 1)
        self.assertTrue(occ.collides(5.0, 0.0, 1.8))
        self.assertFalse(occ.collides(5.0, 0.0, 2.2))      # 1.9 之上没占据

    def test_rotated_ellipsoid(self):
        # 长轴沿 x 的椭球绕 z 转 90° -> 长轴沿 y；用 quats 表达。
        means = np.array([[5.0, 0.0, 2.0]])
        scales = np.array([[1.0, 0.05, 0.05]])
        q = np.array([[math.cos(math.pi / 4), 0, 0, math.sin(math.pi / 4)]])
        occ = col.Occupancy(means, scales, q, np.array([0.9]),
                            z_range=(1.65, 2.35), voxel_m=0.1, **E1)
        self.assertTrue(occ.collides(5.0, 0.8, 2.0))
        self.assertFalse(occ.collides(5.8, 0.0, 2.0))

    def test_max_scale_filter(self):
        means = np.array([[5.0, 0.0, 2.0]])
        scales = np.array([[10.0, 10.0, 10.0]])            # 巨型"天空"高斯
        occ = col.Occupancy(means, scales, None, np.array([0.9]), max_scale_m=3.0,
                            z_range=(1.65, 2.35), voxel_m=0.1, **E1)
        self.assertEqual(occ.n_kept, 0)
        self.assertFalse(occ.collides(5.0, 0.0, 2.0))


class TestDilationShape(unittest.TestCase):
    def test_disk_not_square(self):
        # 单个体素占据，机体半径 0.25 m / 体素 0.1 -> 半径 2 格的圆盘：对角 (2,2) 不在内。
        # 查询点取体素中心（origin = 0.05-0.5，中心 = -0.45+(i+0.5)*0.1），避开格边界。
        occ = col.Occupancy(np.array([[0.05, 0.05, 2.0]]), None, None, None,
                            z_range=(1.95, 2.05), voxel_m=0.1, body_radius_m=0.25,
                            body_half_height_m=0.03, pad_m=0.5)
        self.assertEqual(occ.dilation_cells, (2, 0))
        self.assertTrue(occ.collides(0.30, 0.10, 2.0))      # 轴向 2 格
        self.assertFalse(occ.collides(0.30, 0.30, 2.0))     # 对角 (2,2)：方形会判碰
        self.assertTrue(occ.collides(0.20, 0.20, 2.0))      # (1,1) 在圆内
        self.assertFalse(occ.collides(0.40, 0.10, 2.0))     # 3 格外

    def test_empty(self):
        occ = col.Occupancy(np.zeros((0, 3)))
        self.assertFalse(occ.collides(0, 0, 2.0))
        self.assertEqual(occ.sweep((0, 0, 2), (1, 0, 2)), (False, None))
        self.assertAlmostEqual(occ.clearance(0, 0, 2.0), 2.0)


class TestEnvelopeAndScene(unittest.TestCase):
    def test_disabled_scale_filter_keeps_visible_wall_solid(self):
        m, s, q, o = thin_wall()
        gaussians = {"means": m, "scales": s, "quats": q, "opacities": o}
        for source in ("render", "collision"):
            for limit in (0, None):
                with self.subTest(source=source, max_scale_m=limit):
                    scene = {source: {"max_scale_m": limit}}
                    occ = col.from_scene(scene, gaussians=gaussians)
                    self.assertEqual(occ.n_kept, len(m))
                    self.assertTrue(occ.sweep((4.6, 0, 2), (5.3, 0, 2))[0])
                    self.assertIsNone(occ.summary()["max_scale_m"])

    def test_rejects_invalid_scale_limits(self):
        m, s, q, o = thin_wall()
        for limit in (-1, float("nan"), float("inf")):
            with self.subTest(max_scale_m=limit), self.assertRaisesRegex(ValueError, "max_scale_m"):
                col.Occupancy(m, s, q, o, max_scale_m=limit)

    def test_e1_mini_envelope_from_params(self):
        env = col.load_body_envelope()
        self.assertAlmostEqual(env["radius_m"], 0.102, places=3)
        self.assertAlmostEqual(env["half_height_m"], 0.028, places=3)

    def test_envelope_file_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "e1.json")
            with open(p, "w") as f:
                json.dump({"collision_envelope": {"value": {
                    "radius_from_center": 0.2, "z_min": -0.05, "z_max": 0.03}}}, f)
            env = col.load_body_envelope(p)
            self.assertEqual((env["radius_m"], env["half_height_m"]), (0.2, 0.05))
        self.assertEqual(col.resolve_envelope("x3")["radius_m"], 0.3)
        with self.assertRaises(ValueError):
            col.resolve_envelope({"radius_m": 0.1})

    def test_scene_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=1, length=10.0)
            scene = scene_mod.load_scene(os.path.join(tmp, "scene.json"))
            occ = col.from_scene(scene)
            self.assertGreater(occ.n_in_range, 100)
            self.assertEqual(occ.summary()["proxy_version"], col.PROXY_VERSION)
            self.assertAlmostEqual(occ.summary()["body_radius_m"], 0.102, places=3)
            sp = scene["spawn"]
            self.assertFalse(occ.collides(sp["x"], sp["y"], sp["z"]))
            self.assertFalse(occ.collides(3.0, 0.0, 2.0))
            self.assertTrue(occ.collides(3.0, 3.9, 2.0))          # 墙
            self.assertTrue(occ.collides(6.6, 1.2, 2.0))          # 立柱表面（x=6.55）
            # 注意：盒子只有表面有高斯，正中心 (7.0,1.2) 可能是空的；真实 3DGS 同样如此，
            # 所以必须靠 sweep 挡在表面，而不是靠"内部占据"。
            # 立柱半宽 0.45（x∈[6.55,7.45]）：两端在柱外 1 m，线段穿过柱子中心
            self.assertFalse(occ.collides(5.5, 1.2, 2.0))
            self.assertFalse(occ.collides(8.5, 1.2, 2.0))
            hit, pt = occ.sweep((5.5, 1.2, 2.0), (8.5, 1.2, 2.0))
            self.assertTrue(hit)
            self.assertLess(pt[0], 6.6, pt)
            # 旧 X3 包络更大：同一位置更容易判碰
            occ_x3 = col.from_scene(scene, envelope="x3")
            self.assertGreaterEqual(occ_x3.grid.sum(), occ.grid.sum())


def _write_ply(path, xyz, faces=None, fmt="binary_little_endian", rgb=False):
    """写测试用 PLY：double xyz（Open3D/MetaCam 风格）+ 可选 uchar rgb + 可选三角面。"""
    n = len(xyz)
    head = [f"ply", f"format {fmt} 1.0", f"element vertex {n}",
            "property double x", "property double y", "property double z"]
    if rgb:
        head += ["property uchar red", "property uchar green", "property uchar blue"]
    if faces is not None:
        head += [f"element face {len(faces)}", "property list uchar int vertex_indices"]
    head += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(head) + "\n").encode("ascii"))
        if fmt == "ascii":
            for i, p in enumerate(xyz):
                row = " ".join(f"{v:.6f}" for v in p) + (" 10 20 30" if rgb else "")
                f.write((row + "\n").encode("ascii"))
            for fc in (faces if faces is not None else []):
                f.write(("3 " + " ".join(str(int(v)) for v in fc) + "\n").encode("ascii"))
        else:
            e = "<" if fmt == "binary_little_endian" else ">"
            vt = [("x", e + "f8"), ("y", e + "f8"), ("z", e + "f8")] + ([("r", "u1"), ("g", "u1"), ("b", "u1")] if rgb else [])
            arr = np.zeros(n, dtype=np.dtype(vt))
            arr["x"], arr["y"], arr["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            f.write(arr.tobytes())
            if faces is not None:
                ft = np.dtype([("n", "u1"), ("i", e + "i4", (3,))])
                fa = np.zeros(len(faces), dtype=ft)
                fa["n"], fa["i"] = 3, np.asarray(faces)
                f.write(fa.tobytes())


class TestTruthFromPoints(unittest.TestCase):
    """v3：米制点云 / 网格 -> 同一 Occupancy 接口。"""

    def wall_points(self, n_per_cell=5, noise=0):
        # x=5 的墙：y∈[-1,1] 每 0.1 m 一簇 n_per_cell 个点（同 xy，z 在 2.0±0.03 同一层内），
        # 这样每簇落在同一个体素里，min_points 的计数是确定的；再撒 noise 个孤立噪点
        ys = np.repeat(np.arange(-1.0, 1.0, 0.1) + 0.05, n_per_cell)
        zs = 2.0 + np.tile(np.linspace(-0.03, 0.03, n_per_cell), len(ys) // n_per_cell)
        pts = np.stack([np.full(len(ys), 5.0), ys, zs], 1)
        if noise:
            # 孤立噪点放在 x∈[1,3] 的空域，彼此相距 > 0.3 m（一格一个）
            nx = np.linspace(1.0, 3.0, noise)
            pts = np.vstack([pts, np.stack([nx, np.zeros(noise), np.full(noise, 2.0)], 1)])
        return pts

    def test_wall_blocks_and_version(self):
        occ = col.from_points(self.wall_points(), voxel_m=0.1, **E1)
        self.assertEqual(occ.summary()["proxy_version"], col.TRUTH_VERSION["points"])
        self.assertEqual(occ.summary()["source"], "points")
        self.assertFalse(occ.collides(4.6, 0.0, 2.0))
        hit, pt = occ.sweep((4.6, 0.0, 2.0), (5.4, 0.0, 2.0))
        self.assertTrue(hit)
        self.assertLess(abs(pt[0] - 4.9), 0.11, pt)
        self.assertEqual(occ.sweep((4.5, -1.0, 2.0), (4.5, 1.0, 2.0)), (False, None))

    def test_min_points_filters_isolated_noise(self):
        pts = self.wall_points(noise=5)
        noisy = col.from_points(pts, voxel_m=0.1, min_points=1, **E1)
        clean = col.from_points(pts, voxel_m=0.1, min_points=3, **E1)
        self.assertTrue(noisy.collides(2.0, 0.0, 2.0))         # 单点就算占据：噪点成了障碍
        self.assertFalse(clean.collides(2.0, 0.0, 2.0))        # 每格 <3 点的孤立噪点被滤掉
        self.assertTrue(clean.collides(5.0, 0.0, 2.0))         # 墙（每格 5 点）保留
        self.assertEqual(noisy.raw.sum() - clean.raw.sum(), 5)

    def test_min_points_only_for_pure_points(self):
        m, s, q, o = thin_wall()
        with self.assertRaises(ValueError):
            col.Occupancy(m, s, q, o, min_points=3, **E1)
        with self.assertRaises(ValueError):
            col.from_points(self.wall_points(), min_points=0, **E1)

    def test_points_outside_z_slice_ignored(self):
        pts = self.wall_points()
        pts[:, 2] = 1.0                                        # 全部压到 1 m：飞行层 1.65–2.35 外
        occ = col.from_points(pts, voxel_m=0.1, **E1)
        self.assertEqual(occ.n_in_range, 0)
        self.assertFalse(occ.collides(5.0, 0.0, 2.0))

    def test_nav_to_world_applied(self):
        # 世界系 y-down（Wanderland/COLMAP）：nav z=up 对应世界 -y；墙放在世界系，查询用导航系
        R = np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float64)   # 列 = nav 轴在世界系
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [0.0, 1.0, 0.0]
        nav_pts = self.wall_points()
        world_pts = nav_pts @ R.T + T[:3, 3]
        occ = col.from_points(world_pts, nav_to_world=T, voxel_m=0.1, **E1)
        ref = col.from_points(nav_pts, voxel_m=0.1, **E1)
        self.assertEqual(occ.n_in_range, ref.n_in_range)
        self.assertTrue(occ.collides(5.0, 0.0, 2.0))
        self.assertFalse(occ.collides(4.0, 0.0, 2.0))

    def test_mesh_surface_sampling_and_from_mesh(self):
        # 两个三角形拼成 x=5、y∈[-1,1]、z∈[1.5,2.5] 的方形墙；只有 4 个顶点，靠撒点填满
        xyz = np.array([[5, -1, 1.5], [5, 1, 1.5], [5, 1, 2.5], [5, -1, 2.5]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        pts = col.sample_mesh_surface(xyz, faces, spacing=0.05)
        self.assertGreater(len(pts), 2.0 * 1.0 / 0.05 ** 2 * 0.8)             # ≈ 面积/间距²
        self.assertTrue(np.allclose(pts[:, 0], 5.0))
        occ = col.from_mesh(xyz, faces, voxel_m=0.1, **E1)
        self.assertEqual(occ.summary()["proxy_version"], col.TRUTH_VERSION["mesh"])
        self.assertEqual(occ.min_points, 1)
        # 墙在 y∈[-1,1] 每 0.1 m 格都占据（顶点稀疏时 from_points 直接用顶点会漏）
        for y in np.arange(-0.95, 1.0, 0.1):
            self.assertTrue(occ.collides(5.0, y, 2.0), y)
        self.assertFalse(occ.collides(5.0, 1.5, 2.0))
        sparse = col.from_points(xyz, voxel_m=0.1, min_points=1, **E1)
        self.assertFalse(sparse.collides(5.0, 0.0, 2.0))      # 对照：只用 4 个顶点中间是空的

    def test_mesh_sampling_deterministic_no_holes(self):
        """D3：随机重心撒点每格期望 4 点但按泊松分布，e^-4 ≈ 2% 的 raw 格一个点都没有
        （R2 实测 20×1 m 墙 voxel 0.1 漏 27/1400、0.2 漏 15/400）；改成确定性格点后 0 漏。"""
        # 20×1 m 墙 x=5.03，y/z 端点都不落在体素边界上，墙穿过的 raw 格集合可以按几何算出来
        y0, y1, z0, z1 = -9.97, 10.01, 1.5, 2.5
        xyz = np.array([[5.03, y0, z0], [5.03, y1, z0], [5.03, y1, z1], [5.03, y0, z1]], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        for voxel in (0.1, 0.2):
            occ = col.from_mesh(xyz, faces, voxel_m=voxel, **E1)
            ix = occ._cell([5.03, 0.0, 2.0])[0]
            iy0, iy1 = occ._cell([5.03, y0, 2.0])[1], occ._cell([5.03, y1, 2.0])[1]
            expected = np.zeros(occ.shape, dtype=bool)
            expected[ix, iy0:iy1 + 1, :] = True                 # z∈[1.5,2.5] 盖满整个 z_slice
            self.assertEqual(int(expected.sum()), (iy1 - iy0 + 1) * occ.shape[2])
            missing = int((expected & ~occ.raw).sum())
            self.assertEqual(missing, 0, f"voxel {voxel}: 墙穿过的 {int(expected.sum())} 格里漏 {missing} 格")
            self.assertTrue(np.array_equal(occ.raw, expected), f"voxel {voxel}: 撒点落到墙外的格")
        # 确定性：没有随机数，同输入两次逐位相同；格点含三顶点、都在三角形平面内且在三角形里
        tri = np.array([[0.0, 0.0, 1.0], [0.7, 0.1, 1.2], [0.2, 0.9, 1.5]])
        p1 = col.sample_mesh_surface(tri, [[0, 1, 2]], spacing=0.1)
        p2 = col.sample_mesh_surface(tri, [[0, 1, 2]], spacing=0.1)
        self.assertTrue(np.array_equal(p1, p2))
        self.assertNotIn("seed", col.sample_mesh_surface.__code__.co_varnames[:col.sample_mesh_surface.__code__.co_argcount])
        a, b, c = tri
        nrm = np.cross(b - a, c - a)
        self.assertTrue(np.allclose((p1 - a) @ nrm, 0.0, atol=1e-9))
        m = np.linalg.lstsq(np.stack([b - a, c - a], 1), (p1 - a).T, rcond=None)[0]   # 重心坐标 (u, v)
        self.assertTrue((m >= -1e-9).all() and (m.sum(axis=0) <= 1 + 1e-9).all())
        for v in tri:
            self.assertTrue((np.linalg.norm(p1 - v, axis=1) < 1e-12).any())
        # 相邻点间距 <= spacing：每个点到最近另一点的距离不超过 spacing
        d = np.linalg.norm(p1[:, None, :] - p1[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        self.assertLessEqual(float(d.min(axis=1).max()), 0.1 + 1e-9)

    def test_body_envelope_single_source_no_draft_fallback(self):
        """D10：机体包络只认 sim/dynamics/e1_mini.json；草案回退（E1_MINI_DRAFT）已删，找不到就报错。"""
        self.assertFalse(hasattr(col, "E1_MINI_DRAFT"))
        env = col.load_body_envelope()
        self.assertEqual(env["source"], col.E1_MINI_PARAMS)
        self.assertAlmostEqual(env["radius_m"], 0.102)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as cm:
                col.load_body_envelope(os.path.join(tmp, "nope.json"))
        self.assertNotIn("草案", str(cm.exception))
        self.assertNotIn("草案", col.load_body_envelope.__doc__ or "")

    def test_read_ply_vertices_formats(self):
        xyz = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64) + 0.25
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        with tempfile.TemporaryDirectory() as tmp:
            for fmt in ("binary_little_endian", "binary_big_endian", "ascii"):
                p = os.path.join(tmp, f"{fmt}.ply")
                _write_ply(p, xyz, faces, fmt=fmt, rgb=True)
                r = col.read_ply_vertices(p)
                self.assertEqual(r["n_vertices"], 4, fmt)
                self.assertTrue(np.allclose(r["xyz"], xyz), fmt)
                self.assertTrue(np.array_equal(r["faces"], faces), fmt)
            p = os.path.join(tmp, "pts.ply")
            _write_ply(p, np.tile(xyz, (25, 1)), None, rgb=False)
            r = col.read_ply_vertices(p, stride=4)
            self.assertEqual(len(r["xyz"]), 25)
            self.assertIsNone(r["faces"])
            with self.assertRaises(ValueError):
                col.read_ply_vertices(os.path.join(tmp, "ascii.ply"), stride=2)   # 带面不能抽稀

    def test_from_scene_source_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=1, length=10.0)
            scene = scene_mod.load_scene(os.path.join(tmp, "scene.json"))
            g_occ = col.from_scene(scene)                        # 默认高斯代理
            # 用高斯中心本身当「点云真值」：每个中心一个点，min_points=1
            g = scene_mod.read_gaussian_ply(scene["ply_path"])
            _write_ply(os.path.join(tmp, "truth.ply"), g["means"], None)
            scene["collision"].update({"source": "points", "geometry": "truth.ply", "min_points": 1})
            p_occ = col.from_scene(scene)
            self.assertEqual(p_occ.source, "points")
            self.assertEqual(p_occ.summary()["proxy_version"], col.TRUTH_VERSION["points"])
            self.assertEqual(p_occ.summary()["geometry"], "truth.ply")
            self.assertTrue(p_occ.collides(3.0, 3.9, 2.0))      # 墙
            self.assertFalse(p_occ.collides(3.0, 0.0, 2.0))
            # 同一 scene 强制回到高斯代理，结果与默认相同
            back = col.from_scene(scene, source="gaussians")
            self.assertEqual(back.summary()["proxy_version"], col.PROXY_VERSION)
            self.assertEqual(back.raw.sum(), g_occ.raw.sum())
            scene["collision"]["source"] = "lidar"
            with self.assertRaises(ValueError):
                col.from_scene(scene)
            scene["collision"]["source"] = "mesh"
            with self.assertRaises(ValueError):                 # 点云文件没有面，不能当网格
                col.from_scene(scene)


if __name__ == "__main__":
    unittest.main()
