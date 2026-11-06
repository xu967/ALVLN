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
        bad = np.diag([2.0, 1.0, 1.0, 1.0])
        with self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
