#!/usr/bin/env python3
"""collision_model（任务卡 S1b，审核 B）的单测：纯 Python + numpy，不需要 torch / GPU。

collision_model 选碰撞口径：v1 = 改动前的现状（缺省，逐位不变）；v2 = 审核 B 的发现 1、2、3 都生效。
本文件随各条发现逐个追加用例（一条发现一个提交，用例与代码同一个提交）：
  v1 逐位锁定   五组合成占据（体素 0.05 / 0.1 / 0.2 m，E1 与 X3 机体包络，带余量）在改动前的代码（main ba54628）上算的栅格、
                查询、扫掠、净空、规划硬栅格 / 软余量摘要，改动后缺省参数与显式 v1 都要逐位相同；另有内嵌的改动前 _dilate 逐字拷贝作对照
  B1  膨胀核    v2 的 xy 结构元素 = 格中心到原始占据体素（方块）的欧氏距离不超过 机体半径 + 余量；0.1 m 体素下斜向格要膨胀（v1 是十字，斜向为 0）
  B2  高度口径  v2 的栅格查询取 z ± (机体半高 + 高度余量) 覆盖到的层的并，同一 xy 高 1 cm 与低 1 cm 的结果相同（v1 只查点所在那一层，层边界上跳层）
  B4  规划硬栅格 v2 的 OccPlanner.hard = 巡航高度处同一个层带的并（v1 是全部层的并）；规则臂的列距离与教师的前向检查走同一个 _grid_hit，随之统一
  B7  出生检查  v2 的候选出生点按同一个层带判空，spawn_two_layer 不再需要（v2 下给了就报错），v1 保留
  B3  逐段扫掠  Occupancy.sweep_path 沿折线逐段扫；run_episode 在 v2 且动力学带 trace（cf_sitl）时每步沿 [起点, 窗口内 10 个 trace 点] 扫，
                v1 与没有 trace 的 cf_blackbox 仍是起点到终点一条弦
  B6  blackbox 高度  cf_blackbox 的 z 固定 2.0；占据栅格的高度范围（z_slice）不含 2.0（室内 1.0–1.4）时 run_episode 报错（两种口径都报），
                不再静默地永远不撞
  key  collision_model  口径开关本身：metric_version 加后缀、manifest / run.json 记口径、命令行 --collision-model（v2 与 --spawn-two-layer 互斥）

用 3dgs-venv 或 .venv-tools 跑（要 numpy）：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_collision_model.py
"""
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
col = importlib.import_module("sim.3dgs.collision")
planner = importlib.import_module("sim.3dgs.planner")

E1 = dict(body_radius_m=0.102, body_half_height_m=0.028)


def mk(means, scales=None, quats=None, opac=None, model=None, **kw):
    """model=None：不传 collision_model（改动前的调用写法，缺省口径）；给了就按关键字传。"""
    if model is not None:
        kw["collision_model"] = model
    return col.Occupancy(means, scales, quats, opac, **kw)


# ----------------------------------------------------------------------------- v1 金标
def _sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _synth(seed, n=900, extent=6.0):
    rng = np.random.default_rng(seed)
    means = np.stack([rng.uniform(-extent, extent, n), rng.uniform(-extent, extent, n), rng.uniform(0.9, 2.3, n)], 1)
    return means, rng.uniform(0.01, 0.25, (n, 3)), rng.normal(size=(n, 4)), rng.uniform(0.0, 1.0, n)


GOLDEN_CASES = {
    "e1_vox010_z1.0-1.4": dict(seed=1, kw=dict(z_range=(1.0, 1.4), voxel_m=0.1, **E1)),
    "e1_vox020_z1.8-2.2": dict(seed=2, kw=dict(z_range=(1.8, 2.2), voxel_m=0.2, **E1)),
    "x3_vox010_z1.7-2.3": dict(seed=3, kw=dict(z_range=(1.7, 2.3), voxel_m=0.1, body_radius_m=0.3, body_half_height_m=0.3)),
    "e1_vox005_z1.9-2.1": dict(seed=4, kw=dict(z_range=(1.9, 2.1), voxel_m=0.05, **E1)),
    "e1_vox010_margin": dict(seed=5, kw=dict(z_range=(1.0, 1.4), voxel_m=0.1, inflate_m=0.05, **E1)),
}
# 在改动前的代码（main ba54628，未加 collision_model）上算出；$D/tmp/phase12-cards/s1b/golden_v1.py，连跑两次一致。
GOLDEN_V1 = {
    "e1_vox005_z1.9-2.1": {"clearance": "47729e23e60f0fb9", "dilation_cells": [2, 1], "grid": "801a3fc0b8d9348a",
                           "hard": "fa6447d0cdd36add", "hits": "354c6142e846f17c", "n_cells": 58674, "n_hits": 330,
                           "raw": "c8afdbef457771b3", "soft": "1c3c12f7b21f4038", "sweep_hit": "25242ad9714a67f8",
                           "sweep_pt": "4e4a6fc8387f79ac"},
    "e1_vox010_margin": {"clearance": "31aa3031da189456", "dilation_cells": [2, 0], "grid": "100059e6fc4459bd",
                         "hard": "0be0668d608bbef2", "hits": "d7782fcb2bfdfae5", "n_cells": 18078, "n_hits": 563,
                         "raw": "156c3f49593b1e37", "soft": "9b16a03773604054", "sweep_hit": "2f180281d29a1228",
                         "sweep_pt": "a4efec422229b4ad"},
    "e1_vox010_z1.0-1.4": {"clearance": "ef2c9a17f29505ae", "dilation_cells": [1, 0], "grid": "36bea3b65ae28f7d",
                           "hard": "a91402798b07fbe0", "hits": "170a44812f66b4f3", "n_cells": 11155, "n_hits": 359,
                           "raw": "0b97a42d80c16ce8", "soft": "5fbf121e78190b38", "sweep_hit": "f6127d3312e160f3",
                           "sweep_pt": "ff26ad8e9c969a04"},
    "e1_vox020_z1.8-2.2": {"clearance": "475b8bcd00284704", "dilation_cells": [1, 0], "grid": "fc4f7c58711be5e0",
                           "hard": "f10fe56bb5ddabfb", "hits": "3e4cb382729aeed0", "n_cells": 2397, "n_hits": 651,
                           "raw": "03ff8cc7cc6116a7", "soft": "d2eaa4e2697fadbc", "sweep_hit": "d55a19ded1842268",
                           "sweep_pt": "ac73b0795b9f5b87"},
    "x3_vox010_z1.7-2.3": {"clearance": "2603e6cf4f236773", "dilation_cells": [3, 3], "grid": "047114b3da2c782a",
                           "hard": "237b9e7cb3a36f3c", "hits": "8c921813c7339b49", "n_cells": 65199, "n_hits": 1731,
                           "raw": "9f252f88ea5989d2", "soft": "1906e5108b15c187", "sweep_hit": "a4c8a5337ed2a58b",
                           "sweep_pt": "8f8ff9fadaa54124"},
}


def digest(name, model=None):
    c = GOLDEN_CASES[name]
    means, scales, quats, opac = _synth(c["seed"])
    occ = mk(means, scales, quats, opac, model=model, **c["kw"])
    rng = np.random.default_rng(100 + c["seed"])
    zlo, zhi = c["kw"]["z_range"]
    q = np.stack([rng.uniform(-6, 6, 4000), rng.uniform(-6, 6, 4000), rng.uniform(zlo - 0.2, zhi + 0.2, 4000)], 1)
    hits = np.array([occ.collides(*p) for p in q])
    sw = []
    for i in range(300):
        p0, p1 = q[i], q[i] + rng.uniform(-0.8, 0.8, 3) * np.array([1, 1, 0.1])
        sw.append(occ.sweep(p0, p1))
    sw_h = np.array([s[0] for s in sw])
    sw_p = np.array([(s[1] if s[1] is not None else [np.nan] * 3) for s in sw])
    clr = np.array([occ.clearance(*p) for p in q[:200]])
    out = {"grid": _sha(occ.grid), "raw": _sha(occ.raw), "hits": _sha(hits), "sweep_hit": _sha(sw_h), "sweep_pt": _sha(sw_p),
           "clearance": _sha(clr), "n_cells": int(occ.grid.sum()), "n_hits": int(hits.sum()),
           "dilation_cells": list(occ.dilation_cells)}
    pl = planner.OccPlanner(occ)
    out["hard"], out["soft"] = _sha(pl.hard), _sha(pl.soft)
    return out


def old_dilate(occ_raw, voxel, radius, inflate, half_h):
    """改动前 Occupancy._dilate 的逐字拷贝（圆盘 xy + z 层，体素数四舍五入），只作对照。"""
    r_cells = int(round((radius + inflate) / voxel))
    h_cells = int(round(half_h / voxel))
    out = occ_raw.copy()
    nx, ny, nz = occ_raw.shape
    for di in range(-r_cells, r_cells + 1):
        for dj in range(-r_cells, r_cells + 1):
            if di * di + dj * dj > r_cells * r_cells:
                continue
            for dk in range(-h_cells, h_cells + 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                src = occ_raw[max(0, -di):nx - max(0, di), max(0, -dj):ny - max(0, dj), max(0, -dk):nz - max(0, dk)]
                out[max(0, di):nx - max(0, -di), max(0, dj):ny - max(0, -dj), max(0, dk):nz - max(0, -dk)] |= src
    return out


class TestV1Locked(unittest.TestCase):
    """v1 = 改动前的现状：缺省不传 collision_model 与显式 v1 都与改动前的摘要逐位相同。"""

    def test_default_matches_golden(self):
        for name in GOLDEN_CASES:
            with self.subTest(case=name):
                self.assertEqual(digest(name), GOLDEN_V1[name])

    def test_explicit_v1_matches_golden(self):
        for name in GOLDEN_CASES:
            with self.subTest(case=name):
                self.assertEqual(digest(name, model="v1"), GOLDEN_V1[name])

    def test_v1_dilate_equals_old_reference_on_random_params(self):
        rng = np.random.default_rng(7)
        for trial in range(12):
            voxel = float(rng.choice([0.05, 0.1, 0.2]))
            radius = float(rng.uniform(0.04, 0.35))
            half_h = float(rng.uniform(0.02, 0.35))
            inflate = float(rng.choice([0.0, 0.03, 0.08]))
            pts = np.stack([rng.uniform(-2, 2, 60), rng.uniform(-2, 2, 60), rng.uniform(1.0, 1.6, 60)], 1)
            occ = mk(pts, z_range=(1.0, 1.6), voxel_m=voxel, body_radius_m=radius, body_half_height_m=half_h,
                     inflate_m=inflate, max_scale_m=None, model="v1")
            ref = old_dilate(occ.raw, voxel, radius, inflate, half_h)
            with self.subTest(trial=trial, voxel=voxel, radius=radius, half_h=half_h, inflate=inflate):
                self.assertTrue(np.array_equal(occ.grid, ref))

    def test_unknown_model_rejected(self):
        with self.assertRaises(ValueError):
            mk(np.array([[0.05, 0.05, 1.15]]), z_range=(1.0, 1.4), voxel_m=0.1, model="v3", **E1)

    def test_summary_records_model(self):
        pts = np.array([[0.05, 0.05, 1.15]])
        s1 = mk(pts, z_range=(1.0, 1.4), voxel_m=0.1, **E1).summary()
        s2 = mk(pts, z_range=(1.0, 1.4), voxel_m=0.1, model="v2", **E1).summary()
        self.assertEqual(s1["collision_model"], "v1")
        self.assertEqual(s1["proxy_version"], col.PROXY_VERSION, "v1 的 proxy_version 串不变")
        self.assertEqual(s2["collision_model"], "v2")
        self.assertEqual(s2["proxy_version"], col.PROXY_VERSION + "+coll.v2")


# ----------------------------------------------------------------------------- B1 膨胀核
def single_voxel(voxel, model, inflate=0.0, radius=0.102, z_range=(1.0, 1.4)):
    """栅格中央放一个占据体素：点放在体素中心，返回 (occ, 体素的格下标 (i, j))。"""
    pts = np.array([[0.5 * voxel, 0.5 * voxel, z_range[0] + 0.5 * (z_range[1] - z_range[0]) - 0.05]])
    occ = mk(pts, z_range=z_range, voxel_m=voxel, body_radius_m=radius, body_half_height_m=0.028,
             inflate_m=inflate, model=model)
    c = occ._cell(pts)[0]
    return occ, (int(c[0]), int(c[1]))


def expected_offsets(voxel, reach_m):
    """规格：偏移 (di, dj) 保留 <=> 该格中心到原始体素方块的欧氏距离 <= reach_m。暴力枚举，不复用被测代码的公式。"""
    out = set()
    for di in range(-8, 9):
        for dj in range(-8, 9):
            cx, cy = di * voxel, dj * voxel                    # 格中心相对原始体素中心
            dx = max(abs(cx) - 0.5 * voxel, 0.0)
            dy = max(abs(cy) - 0.5 * voxel, 0.0)
            if math.hypot(dx, dy) <= reach_m + 1e-9:
                out.add((di, dj))
    return out


class TestB1Kernel(unittest.TestCase):
    def offsets_of(self, occ, ij):
        layer = occ.grid[:, :, :].any(axis=2)
        ii, jj = np.nonzero(layer)
        return {(int(i) - ij[0], int(j) - ij[1]) for i, j in zip(ii, jj)}

    def test_diagonal_at_010_voxel(self):
        """0.1 m 体素、机体半径 0.102 m：斜向邻格中心离原始体素边角 0.0707 m，在机体半径内，v2 要判碰；v1 的十字核判不碰（缺陷，锁定）。"""
        v1, (i, j) = single_voxel(0.1, "v1")
        v2, (i2, j2) = single_voxel(0.1, "v2")
        self.assertEqual((i, j), (i2, j2))
        z = 1.15
        centre = lambda occ, di, dj: (occ.origin[0] + (i + di + 0.5) * 0.1, occ.origin[1] + (j + dj + 0.5) * 0.1, z)  # noqa: E731
        self.assertTrue(v1.collides(*centre(v1, 1, 0)) and v2.collides(*centre(v2, 1, 0)), "轴向邻格两种口径都判碰")
        self.assertFalse(v1.collides(*centre(v1, 1, 1)), "v1 十字核：斜向邻格不判碰")
        self.assertTrue(v2.collides(*centre(v2, 1, 1)), "v2：斜向邻格判碰")
        self.assertFalse(v2.collides(*centre(v2, 2, 0)), "再远一格（0.15 m 到方块）不判碰")
        self.assertFalse(v2.collides(*centre(v2, 2, 1)))
        self.assertFalse(v2.collides(*centre(v2, 2, 2)))
        # 数值例：斜向邻格中心到方块边角的距离
        self.assertAlmostEqual(math.hypot(0.05, 0.05), 0.0707, places=3)
        self.assertLess(math.hypot(0.05, 0.05), 0.102)

    def test_v2_kernel_matches_euclid_definition(self):
        for voxel, inflate, radius in [(0.1, 0.0, 0.102), (0.1, 0.05, 0.102), (0.05, 0.0, 0.102), (0.2, 0.0, 0.102),
                                       (0.1, 0.0, 0.3), (0.1, 0.0, 0.05), (0.2, 0.1, 0.102), (0.1, 0.0, 0.15)]:
            occ, ij = single_voxel(voxel, "v2", inflate=inflate, radius=radius)
            with self.subTest(voxel=voxel, inflate=inflate, radius=radius):
                self.assertEqual(self.offsets_of(occ, ij), expected_offsets(voxel, radius + inflate))

    def test_e1_at_010_is_3x3_square(self):
        occ, ij = single_voxel(0.1, "v2")
        self.assertEqual(self.offsets_of(occ, ij), {(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)})
        self.assertEqual(occ.dilation_cells[0], 1)

    def test_v2_equals_v1_at_020_voxel(self):
        """0.2 m 体素：v1 的圆盘与 v2 的欧氏核都是十字（斜向邻格中心到方块边角 0.141 m 大于 0.102 m），栅格逐位相同。"""
        a, ij = single_voxel(0.2, "v1")
        b, _ = single_voxel(0.2, "v2")
        self.assertTrue(np.array_equal(a.grid, b.grid))
        self.assertEqual(self.offsets_of(b, ij), {(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)})

    def test_v2_is_superset_of_v1(self):
        means, scales, quats, opac = _synth(9)
        kw = dict(z_range=(1.0, 1.4), voxel_m=0.1, **E1)
        a = mk(means, scales, quats, opac, model="v1", **kw)
        b = mk(means, scales, quats, opac, model="v2", **kw)
        self.assertTrue(np.array_equal(a.raw, b.raw), "原始占据与口径无关")
        self.assertTrue((b.grid | a.grid == b.grid).all(), "v2 的膨胀栅格包含 v1 的")
        self.assertGreater(int(b.grid.sum()), int(a.grid.sum()))

    def test_cell_centres_follow_exact_distance(self):
        """随机原始占据：每个格中心判不判碰 = 到最近原始占据体素方块的水平距离 <= 机体半径（同层，z 层不变）。"""
        rng = np.random.default_rng(3)
        pts = np.stack([rng.uniform(-1.5, 1.5, 25), rng.uniform(-1.5, 1.5, 25), np.full(25, 1.15)], 1)
        occ = mk(pts, z_range=(1.0, 1.4), voxel_m=0.1, model="v2", **E1)
        v = occ.voxel
        ii, jj = np.nonzero(occ.raw[:, :, 1])
        bx0, by0 = occ.origin[0] + ii * v, occ.origin[1] + jj * v
        n_hit = 0
        for i in range(0, occ.shape[0], 1):
            for j in range(0, occ.shape[1], 1):
                x, y = occ.origin[0] + (i + 0.5) * v, occ.origin[1] + (j + 0.5) * v
                dx = np.maximum(np.maximum(bx0 - x, 0.0), x - (bx0 + v))
                dy = np.maximum(np.maximum(by0 - y, 0.0), y - (by0 + v))
                want = bool((np.hypot(dx, dy) <= 0.102 + 1e-9).any())
                got = bool(occ.grid[i, j, 1])
                if want != got:
                    self.fail(f"格 ({i},{j}) 中心 ({x:.3f},{y:.3f})：精确距离判 {want}，栅格 {got}")
                n_hit += want
        self.assertGreater(n_hit, 25)


# ----------------------------------------------------------------------------- B2 高度口径
def wall(layer_z, model, margin=None, voxel=0.1, z_range=(1.0, 1.4), env=None):
    """x = 1.0 处沿 y 的一堵点墙，只在高度 layer_z 这一层；返回占据。"""
    ys = np.linspace(-1.0, 1.0, 21)
    pts = np.stack([np.full(21, 1.0), ys, np.full(21, layer_z)], 1)
    kw = dict(z_range=z_range, voxel_m=voxel, **(env or E1))
    if margin is not None:
        kw["height_margin_m"] = margin
    return mk(pts, model=model, **kw)



class TestB2HeightBand(unittest.TestCase):
    def test_v1_flips_across_layer_boundary(self):
        """v1 的缺陷（锁定）：墙只在层 2（z 1.2–1.3），同一 xy 高 1 cm 判碰、低 1 cm 不判。"""
        occ = wall(1.25, "v1")
        self.assertFalse(occ.collides(1.0, 0.0, 1.199))
        self.assertTrue(occ.collides(1.0, 0.0, 1.201))

    def test_v2_same_answer_across_layer_boundary(self):
        """稳态飞行 z（出生高度 1.2 上下 −0.0231 … +0.009，起飞过冲到 +0.046）内，墙在层 1 或层 2 都判碰，不随跳层变化。"""
        for wall_z in (1.15, 1.25):
            occ = wall(wall_z, "v2")
            for z in (1.1769, 1.185, 1.199, 1.2, 1.201, 1.209, 1.2457):
                with self.subTest(wall_z=wall_z, z=z):
                    self.assertTrue(occ.collides(1.0, 0.0, z))

    def test_v2_band_edges_are_body_half_height_plus_margin(self):
        """层带 = z ± (0.028 + 余量)：离带 1 mm 以外的层不算，带内 1 mm 的层算。"""
        h = 0.028 + col.HEIGHT_MARGIN_M
        top = wall(1.35, "v2")                          # 层 3：[1.3, 1.4)
        self.assertFalse(top.collides(1.0, 0.0, 1.3 - h - 1e-3))
        self.assertTrue(top.collides(1.0, 0.0, 1.3 - h + 1e-3))
        bottom = wall(1.05, "v2")                       # 层 0：[1.0, 1.1)
        self.assertFalse(bottom.collides(1.0, 0.0, 1.1 + h + 1e-3))
        self.assertTrue(bottom.collides(1.0, 0.0, 1.1 + h - 1e-3))

    def test_steady_flight_band_stays_in_layers_1_and_2(self):
        """余量 0.025 m 的取值依据：稳态 z 相对出生高度 1.2 的 p1 / p99 为 −0.0231 / +0.009，逐步极值 −0.0342 / +0.018；
        层带只落在层 1 与层 2，层 0、层 3 的障碍不算。"""
        for wall_z in (1.05, 1.35):
            occ = wall(wall_z, "v2")
            for z in (1.1658, 1.1769, 1.19, 1.2, 1.209, 1.218):
                with self.subTest(wall_z=wall_z, z=z):
                    self.assertFalse(occ.collides(1.0, 0.0, z))

    def test_height_margin_parameter(self):
        occ0 = wall(1.35, "v2", margin=0.0)             # 带 ±0.028
        self.assertFalse(occ0.collides(1.0, 0.0, 1.27))
        self.assertTrue(occ0.collides(1.0, 0.0, 1.273))
        occ1 = wall(1.35, "v2", margin=0.1)             # 带 ±0.128
        self.assertTrue(occ1.collides(1.0, 0.0, 1.2))

    def test_height_margin_validated(self):
        with self.assertRaises(ValueError):
            wall(1.25, "v2", margin=-0.01)

    def test_z_outside_range_uses_band_intersection(self):
        """z 在栅格高度范围外：v1 只查地面；v2 看带与栅格最上 / 最下层是否相交。"""
        top_wall = wall(1.35, "v2")
        self.assertTrue(top_wall.collides(1.0, 0.0, 1.42))       # 带 [1.367, 1.473] 与层 3 相交
        self.assertFalse(top_wall.collides(1.0, 0.0, 1.46))      # 带 [1.407, ...] 在栅格上方
        self.assertFalse(wall(1.35, "v1").collides(1.0, 0.0, 1.42))

    def test_layer_span_values(self):
        occ = wall(1.25, "v2")                                       # 4 层，带半宽 0.053
        k0, k1 = occ.layer_span(np.array([1.19, 1.15, 1.25, 3.0, -1.0, np.nan, np.inf]))
        self.assertEqual(list(zip(k0[:3].tolist(), k1[:3].tolist())), [(1, 2), (0, 2), (1, 3)])
        self.assertTrue((k0[3:] > k1[3:]).all(), "带在栅格上 / 下方、z 不是有限数：空")
        k0s, k1s = occ.layer_span(1.19)                                # 标量
        self.assertEqual((int(k0s), int(k1s)), (1, 2))

    def test_v2_grid_has_no_z_dilation(self):
        """v2 的 z 向膨胀由查询时的层带承担：X3 包络（半高 0.3）下栅格不再沿 z 复制，同样的障碍两种写法等价。"""
        x3 = dict(body_radius_m=0.3, body_half_height_m=0.3)
        occ = wall(1.35, "v2", env=x3)
        self.assertEqual(occ.dilation_cells[1], 0)
        self.assertFalse(occ.grid[:, :, 0].any() or occ.grid[:, :, 1].any() or occ.grid[:, :, 2].any(), "栅格只在原始层上")
        self.assertTrue(occ.collides(1.0, 0.0, 1.1), "带 ±0.325 够到层 3")

    def test_vectorised_equals_scalar(self):
        means, scales, quats, opac = _synth(11)
        occ = mk(means, scales, quats, opac, model="v2", z_range=(1.0, 1.4), voxel_m=0.1, **E1)
        rng = np.random.default_rng(5)
        q = np.stack([rng.uniform(-6, 6, 600), rng.uniform(-6, 6, 600), rng.uniform(0.8, 1.6, 600)], 1)
        vec = occ._grid_hit(q)
        one = np.array([occ._grid_hit(q[i:i + 1])[0] for i in range(len(q))])
        self.assertTrue(np.array_equal(vec, one))
        self.assertGreater(int(vec.sum()), 20)

    def test_v2_summary_has_height_fields(self):
        s = wall(1.25, "v2").summary()
        self.assertEqual(s["height_margin_m"], col.HEIGHT_MARGIN_M)
        self.assertAlmostEqual(s["height_band_half_m"], 0.028 + col.HEIGHT_MARGIN_M)
        self.assertNotIn("height_margin_m", wall(1.25, "v1").summary(), "v1 的 summary 不多记高度字段")

    def test_cruise_z_from_scene(self):
        import tempfile
        scene_mod = importlib.import_module("sim.3dgs.scene")
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=1, length=10.0)
            scene = scene_mod.load_scene(os.path.join(tmp, "scene.json"))
            occ = col.from_scene(scene, collision_model="v2")
            self.assertEqual(occ.collision_model, "v2")
            self.assertEqual(occ.cruise_z, float(scene["spawn"]["z"]))
            self.assertEqual(col.from_scene(scene).collision_model, "v1")


# ----------------------------------------------------------------------------- B4 规划硬栅格、规则臂列距离、教师前向检查
class StraightPlanner:
    """OccAstarTeacher 的替身规划器：路径 = 起点到 waypoint 的直线，瞄准点 = waypoint；occ 是真的 Occupancy。"""

    def __init__(self, occ):
        self.occ = occ

    def plan(self, start, goal):
        return [start, goal]

    def aim_point(self, path, x, y, lookahead):
        return path[-1]


class State:
    def __init__(self, x, y, yaw_deg, z):
        self.x, self.y, self.z, self.yaw = x, y, z, math.radians(yaw_deg)
        self.vx = self.vy = 0.0


class TestB4Consumers(unittest.TestCase):
    def test_planner_hard_v1_is_union_of_all_layers(self):
        """v1（锁定）：硬栅格 = 各层并集，层 0 与层 3 的障碍也算。"""
        for wall_z in (1.05, 1.15, 1.25, 1.35):
            occ = wall(wall_z, "v1")
            hard = planner.OccPlanner(occ).hard
            i, j = occ._cell(np.array([[1.0, 0.0, 1.2]]))[0][:2]
            with self.subTest(wall_z=wall_z):
                self.assertTrue(hard[i, j])

    def test_planner_hard_v2_is_band_union_at_cruise_height(self):
        for wall_z, want in ((1.05, False), (1.15, True), (1.25, True), (1.35, False)):
            occ = wall(wall_z, "v2")
            hard = planner.OccPlanner(occ, z=1.2).hard
            i, j = occ._cell(np.array([[1.0, 0.0, 1.2]]))[0][:2]
            with self.subTest(wall_z=wall_z):
                self.assertEqual(bool(hard[i, j]), want)

    def test_planner_v2_uses_cruise_z_of_occupancy(self):
        occ = wall(1.25, "v2")
        with self.assertRaises(ValueError):
            planner.OccPlanner(occ)              # 没有巡航高度（occ.cruise_z 为 None）也没传 z：不猜
        occ.cruise_z = 1.2
        i, j = occ._cell(np.array([[1.0, 0.0, 1.2]]))[0][:2]
        self.assertTrue(planner.OccPlanner(occ).hard[i, j])

    def test_planner_v1_ignores_z_argument(self):
        occ = wall(1.05, "v1")
        i, j = occ._cell(np.array([[1.0, 0.0, 1.2]]))[0][:2]
        self.assertTrue(planner.OccPlanner(occ, z=1.2).hard[i, j])

    def test_rule_columns_follow_unified_band(self):
        ro = importlib.import_module("sim.3dgs.rule_oracle")
        for model, expect_equal in (("v1", False), ("v2", True)):
            occ = wall(1.25, model)                                  # 墙只在层 2
            lo = ro.occ_columns(occ, 0.0, 0.0, 1.19, 0.0)             # 动作前 z 在层 1
            hi = ro.occ_columns(occ, 0.0, 0.0, 1.21, 0.0)             # 动作前 z 在层 2
            mid = len(lo) // 2
            with self.subTest(model=model):
                self.assertLess(hi[mid], ro.MAX_RANGE_M, "层 2 里的墙在 z=1.21 时看得见")
                self.assertEqual(bool(abs(lo[mid] - hi[mid]) < 1e-9), expect_equal)
                if not expect_equal:
                    self.assertEqual(lo[mid], ro.MAX_RANGE_M, "v1：z=1.19 单层，墙在层 2 看不见")

    def test_teacher_forward_check_follows_unified_band(self):
        """教师 0.4 m 前向检查（occ.sweep）：墙在层 2、机体 z=1.19（层 1）。v1 放行（FORWARD），v2 先转（turn_blocked）。"""
        out = {}
        for model in ("v1", "v2"):
            occ_w = mk(np.stack([np.full(41, 0.45), np.linspace(-2, 2, 41), np.full(41, 1.25)], 1),
                       model=model, z_range=(1.0, 1.4), voxel_m=0.1, **E1)
            t = planner.OccAstarTeacher(StraightPlanner(occ_w), {"x": 5.0, "y": 0.0}, hold_s=1.0)
            action, _ = t(None, None, State(0.0, 0.0, 0.0, 1.19))
            out[model] = (action, t.last_branch)
        self.assertEqual(out["v1"], (planner.FORWARD_SLOW, "forward"))
        self.assertEqual(out["v2"][1], "turn_blocked")


# ----------------------------------------------------------------------------- B7 出生检查
class TestB7Spawn(unittest.TestCase):
    def setUp(self):
        self.collect = importlib.import_module("sim.3dgs.collect_3dgs")
        self.scene = {"spawn": {"x": 0.0, "y": 0.0, "z": 1.2, "yaw_deg": 0.0}}

    def filled_box(self, layer_z, model):
        """x、y 在 ±1 内、只在高度 layer_z 这一层铺满点：抖动 0.3 m 的候选全落在里面。"""
        g = np.arange(-1.0, 1.001, 0.05)
        xx, yy = np.meshgrid(g, g, indexing="ij")
        pts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, layer_z)], 1)
        return mk(pts, model=model, z_range=(1.0, 1.4), voxel_m=0.1, **E1)

    def spawn_layer(self, occ):
        return int(np.floor((1.2 - occ.origin[2]) / occ.voxel))

    def test_v1_only_checks_the_layer_of_spawn_height(self):
        """v1（锁定）：出生点只查 scene.spawn.z 所在的那一层；障碍在另一层时抖动候选照样通过。"""
        probe = self.filled_box(1.25, "v1")
        other_z = 1.25 if self.spawn_layer(probe) != 2 else 1.15      # 取不含出生高度的那一层
        occ = self.filled_box(other_z, "v1")
        x, y, _ = self.collect._spawn_for_episode(self.scene, occ, 0.3, np.random.default_rng(0))
        self.assertTrue(abs(x) <= 0.3 and abs(y) <= 0.3)

    def test_v2_rejects_candidates_blocked_in_either_neighbouring_layer(self):
        for wall_z in (1.15, 1.25):
            occ = self.filled_box(wall_z, "v2")
            with self.subTest(wall_z=wall_z):
                with self.assertRaises(RuntimeError):
                    self.collect._spawn_for_episode(self.scene, occ, 0.3, np.random.default_rng(0))

    def test_v2_free_space_spawn_sequence_unchanged(self):
        """没有障碍时 v2 与 v1 的出生位姿序列逐位相同（只多拒绝被挡的候选）。"""
        far = np.array([[8.0, 8.0, 1.15], [8.5, 8.0, 1.15]])
        a = mk(far, model="v1", z_range=(1.0, 1.4), voxel_m=0.1, **E1)
        b = mk(far, model="v2", z_range=(1.0, 1.4), voxel_m=0.1, **E1)
        ra, rb = np.random.default_rng(3), np.random.default_rng(3)
        for _ in range(5):
            self.assertEqual(self.collect._spawn_for_episode(self.scene, a, 0.3, ra),
                             self.collect._spawn_for_episode(self.scene, b, 0.3, rb))

    def test_v2_rejects_spawn_two_layer(self):
        occ = mk(np.array([[8.0, 8.0, 1.15]]), model="v2", z_range=(1.0, 1.4), voxel_m=0.1, **E1)
        with self.assertRaises(ValueError):
            self.collect._spawn_for_episode(self.scene, occ, 0.3, np.random.default_rng(0), two_layer=True)
        occ1 = mk(np.array([[8.0, 8.0, 1.15]]), model="v1", z_range=(1.0, 1.4), voxel_m=0.1, **E1)
        self.collect._spawn_for_episode(self.scene, occ1, 0.3, np.random.default_rng(0), two_layer=True)   # v1 保留


# ----------------------------------------------------------------------------- B3 逐段扫掠
class TestB3SweepPath(unittest.TestCase):
    def block(self, model="v2"):
        """一个原始占据体素（0.1 m 体素），v2 的 3×3 核把它膨胀成约 0.3 m 见方。"""
        return mk(np.array([[0.55, 0.35, 1.25]]), model=model, z_range=(1.0, 1.4), voxel_m=0.1, **E1)

    def test_polyline_catches_obstacle_off_the_chord(self):
        occ = self.block()
        p0, p1 = (0.0, 0.0, 1.25), (1.0, 0.0, 1.25)
        self.assertFalse(occ.sweep(p0, p1)[0], "弦从膨胀后的障碍边 0.15 m 以外过")
        hit, pt = occ.sweep_path([p0, (0.55, 0.35, 1.25), p1])
        self.assertTrue(hit)
        self.assertLess(math.hypot(pt[0] - 0.55, pt[1] - 0.35), 0.3, pt)

    def test_equals_first_hit_of_per_segment_sweeps(self):
        means, scales, quats, opac = _synth(11)
        occ = mk(means, scales, quats, opac, model="v2", z_range=(1.0, 1.4), voxel_m=0.1, **E1)
        rng = np.random.default_rng(8)
        n_hit = 0
        for _ in range(80):
            start = np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), 1.2])
            pts = start + np.cumsum(rng.uniform(-0.4, 0.4, (6, 3)) * np.array([1.0, 1.0, 0.05]), axis=0)
            want = (False, None)
            for a, b in zip(pts[:-1], pts[1:]):
                h, pt = occ.sweep(a, b)
                if h:
                    want = (True, pt)
                    break
            self.assertEqual(occ.sweep_path(pts), want)
            n_hit += want[0]
        self.assertTrue(5 < n_hit < 75, n_hit)

    def test_single_point_and_empty(self):
        occ = self.block()
        self.assertEqual(occ.sweep_path([]), (False, None))
        self.assertEqual(occ.sweep_path([(0.55, 0.35, 1.25)]), (True, [0.55, 0.35, 1.25]))
        self.assertEqual(occ.sweep_path([(3.0, 3.0, 1.25)]), (False, None))

    def test_ground_hit_on_path(self):
        hit, pt = self.block().sweep_path([(3.0, 3.0, 1.25), (3.0, 3.0, 0.5), (3.0, 3.0, 0.02)])
        self.assertTrue(hit)
        self.assertLessEqual(pt[2], 0.028 + 1e-9)


class TestB3StepCollision(unittest.TestCase):
    """collect_3dgs._step_collision：v2 且有 trace 时沿折线，其余沿弦。"""

    def setUp(self):
        self.collect = importlib.import_module("sim.3dgs.collect_3dgs")
        self.p0, self.p1 = (0.0, 0.0, 1.25), (1.0, 0.0, 1.25)
        self.arc = [{"step": 0, "pos": [0.1 * k, 0.35 * math.sin(math.pi * k / 10), 1.25]} for k in range(1, 11)]
        self.arc[-1]["pos"] = list(self.p1)

    def occ(self, model):
        return mk(np.array([[0.55, 0.35, 1.25]]), model=model, z_range=(1.0, 1.4), voxel_m=0.1, **E1)

    def test_v2_with_trace_sweeps_the_polyline(self):
        hit, pt = self.collect._step_collision(self.occ("v2"), self.p0, self.p1, self.arc, 0)
        self.assertTrue(hit)
        self.assertGreater(pt[1], 0.1, pt)

    def test_v1_keeps_the_chord_even_with_trace(self):
        occ = self.occ("v1")
        self.assertEqual(self.collect._step_collision(occ, self.p0, self.p1, self.arc, 0), occ.sweep(self.p0, self.p1))
        self.assertFalse(self.collect._step_collision(occ, self.p0, self.p1, self.arc, 0)[0])

    def test_no_trace_keeps_the_chord_under_v2(self):
        occ = self.occ("v2")
        for trace in (None, [], self.arc[:3]):
            with self.subTest(trace=None if trace is None else len(trace)):
                self.assertEqual(self.collect._step_collision(occ, self.p0, self.p1, trace, len(trace or [])),
                                 occ.sweep(self.p0, self.p1))

    def test_only_this_steps_trace_points_are_used(self):
        occ = self.occ("v2")
        earlier = [{"step": 0, "pos": [0.55, 0.35, 1.25]}] * 5            # 上一步的点穿过障碍，本步不该看
        this_step = [{"step": 1, "pos": [0.1 * k, 0.0, 1.25]} for k in range(1, 11)]
        hit, _ = self.collect._step_collision(occ, self.p0, self.p1, earlier + this_step, len(earlier))
        self.assertFalse(hit)

    def test_end_point_is_always_swept(self):
        occ = self.occ("v2")
        short = [{"step": 0, "pos": [0.1 * k, 0.0, 1.25]} for k in range(1, 6)]     # trace 末点（0.5, 0）不是终点（0.55, 0.35）
        hit, _ = self.collect._step_collision(occ, (0.0, 0.0, 1.25), (0.55, 0.35, 1.25), short, 0)
        self.assertTrue(hit)


class StubSitl:
    """CfSitlDynamics 的替身（不起 SITL 子进程）：每步按 script 走一组 trace 点，末点 = 终点；接口够 run_episode 用。"""
    script = []

    def __init__(self, scene, params, dynamics_path=None, seed=0):
        self.params = types.SimpleNamespace(camera_mount={"xyz": [0.0, 0.0, 0.0], "pitch_deg": 0.0})
        self.trace, self.t, self.i = [], 0.0, 0
        self.x = self.y = self.yaw = 0.0
        self.z = 2.0

    def reset(self, x, y, yaw_deg, episode=0):
        self.x, self.y, self.yaw = x, y, math.radians(yaw_deg)
        return {"x": x, "y": y, "yaw_deg": yaw_deg}

    def camera_viewmat(self, nav_to_world=None):
        return np.eye(4)

    def step(self, action, dt):
        pts = self.script[self.i]
        for p in pts:
            self.t += dt / len(pts)
            self.trace.append({"step": self.i, "t": self.t, "pos": list(p)})
        self.x, self.y, self.z = pts[-1]
        self.i += 1
        return {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "vx": 0.0, "vy": 0.0, "vz": 0.0,
                "yaw": self.yaw, "yaw_deg": math.degrees(self.yaw)}

    def profile(self):
        return {"stub": True}

    def manifest(self):
        return {"stub": True}

    def close(self):
        pass


class ForwardPolicy:
    name = "forward"

    def __call__(self, rgb, depth, state):
        return planner.FORWARD_SLOW, None


class TestB3RunEpisode(unittest.TestCase):
    """真的跑 run_episode（mock 渲染器 + 替身 SITL）：一步弧线从柱旁绕过，弦干净、折线擦到柱子。"""

    ARC = [(5.5 + 0.3 * k, 0.95 * math.sin(math.pi * k / 10), 2.0) for k in range(1, 11)]

    def run_model(self, tmp, model):
        collect = importlib.import_module("sim.3dgs.collect_3dgs")
        scene_mod = importlib.import_module("sim.3dgs.scene")
        scene_dir = os.path.join(tmp, "scene")
        if not os.path.exists(scene_dir):
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
        scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
        scene["spawn"] = {"x": 5.5, "y": 0.0, "z": 2.0, "yaw_deg": 0.0}
        scene["waypoint"] = {"x": 11.0, "y": 0.0, "z": 2.0}
        renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(scene, True)
        occ = col.from_scene(scene, collision_model=model)
        params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
        identity = collect.scene_identity(scene)
        StubSitl.script = [self.ARC]
        old = collect.CfSitlDynamics
        collect.CfSitlDynamics = StubSitl
        try:
            out = os.path.join(tmp, model)
            m, err = collect.run_episode(scene, out, renderer, occ, params, ForwardPolicy(), 1, 1.0,
                                         split=identity["split"], identity=identity, dynamics="cf_sitl")
        finally:
            collect.CfSitlDynamics = old
        self.assertIsNone(err)
        with open(os.path.join(out, "actions.jsonl"), encoding="utf-8") as f:
            return m, [json.loads(line) for line in f]

    def test_v1_chord_misses_v2_polyline_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows1 = self.run_model(tmp, "v1")
            _, rows2 = self.run_model(tmp, "v2")
        self.assertFalse(rows1[0]["collision"], rows1[0]["collision_point"])
        self.assertTrue(rows2[0]["collision"])
        self.assertEqual(rows2[0]["src"], "voxel_proxy")
        x, y, _ = rows2[0]["collision_point"]
        self.assertTrue(6.0 < x < 8.0 and y > 0.5, (x, y))


# ----------------------------------------------------------------------------- B6 cf_blackbox 的固定高度
class TestB6BlackboxHeight(unittest.TestCase):
    """cf_blackbox 的 z 固定 2.0。z_slice 不含 2.0（室内 1.0–1.4）时占据只查得到地面，永远撞不到墙：以前静默跑，现在报错。"""

    def run_blackbox(self, tmp, z_slice, model):
        collect = importlib.import_module("sim.3dgs.collect_3dgs")
        scene_mod = importlib.import_module("sim.3dgs.scene")
        scene_dir = os.path.join(tmp, "scene")
        if not os.path.exists(scene_dir):
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
        scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
        if z_slice is not None:
            scene.setdefault("collision", {})["z_slice"] = z_slice
        renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(scene, True)
        occ = col.from_scene(scene, collision_model=model)
        params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
        identity = collect.scene_identity(scene)
        return collect.run_episode(scene, os.path.join(tmp, "out_" + model), renderer, occ, params, ForwardPolicy(), 2, 1.0,
                                   split=identity["split"], identity=identity, dynamics="cf_blackbox")

    def test_indoor_slice_raises_under_both_models(self):
        for model in ("v1", "v2"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError) as cm:
                    self.run_blackbox(tmp, [1.0, 1.4], model)
                self.assertIn("cf_blackbox", str(cm.exception))
                self.assertIn("z_slice", str(cm.exception))

    def test_slice_containing_fixed_height_runs(self):
        for model in ("v1", "v2"):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as tmp:
                m, err = self.run_blackbox(tmp, None, model)        # 缺省 z_slice (1.65, 2.35) 含 2.0
                self.assertIsNone(err)
                self.assertEqual(m["num_steps"], 2)


# ----------------------------------------------------------------------------- collision_model 键本身
class TestCollisionModelKey(unittest.TestCase):
    def test_metric_version_suffix(self):
        base = "3dgs.pilot.v2+stop.v1"
        self.assertEqual(col.metric_version(base), base)
        self.assertEqual(col.metric_version(base, "v1"), base)
        self.assertEqual(col.metric_version(base, "v2"), base + "+coll.v2")
        with self.assertRaises(ValueError):
            col.metric_version(base, "v3")

    def test_manifest_records_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            m1, _ = TestB3RunEpisode().run_model(tmp, "v1")
            m2, _ = TestB3RunEpisode().run_model(tmp, "v2")
        self.assertEqual((m1["collision_model"], m2["collision_model"]), ("v1", "v2"))
        self.assertEqual(m1["collision_proxy"]["proxy_version"], col.PROXY_VERSION)
        self.assertEqual(m2["collision_proxy"]["proxy_version"], col.PROXY_VERSION + "+coll.v2")


class TestCollisionModelCli(unittest.TestCase):
    def cli(self, tmp, out, *extra):
        scene_mod = importlib.import_module("sim.3dgs.scene")
        scene_dir = os.path.join(tmp, "scene")
        if not os.path.exists(scene_dir):
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
        return subprocess.run([sys.executable, "-m", "sim.3dgs.collect_3dgs", "--scene", os.path.join(scene_dir, "scene.json"),
                               "--out", os.path.join(tmp, out), "--mock-renderer", "--dynamics", "cf_blackbox", "--policy", "forward",
                               "--steps", "2", "--hold", "1.0", *extra], cwd=ROOT, capture_output=True, text=True, timeout=180)

    @staticmethod
    def load(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def test_default_is_v1_and_v2_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.cli(tmp, "dflt")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual(self.load(os.path.join(tmp, "dflt", "run.json"))["collision_model"], "v1")
            m = self.load(os.path.join(tmp, "dflt", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["collision_model"], m["collision_proxy"]["proxy_version"]), ("v1", col.PROXY_VERSION))
            r = self.cli(tmp, "v2", "--collision-model", "v2")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertEqual(self.load(os.path.join(tmp, "v2", "run.json"))["collision_model"], "v2")
            m = self.load(os.path.join(tmp, "v2", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["collision_model"], m["collision_proxy"]["proxy_version"]), ("v2", col.PROXY_VERSION + "+coll.v2"))

    def test_v2_with_spawn_two_layer_and_bad_value_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.cli(tmp, "both", "--collision-model", "v2", "--spawn-jitter", "0.1", "--spawn-two-layer")
            self.assertEqual(r.returncode, 2, r.stderr[-1000:])
            self.assertIn("层带", r.stderr)                                                # 报的是出生检查已统一的那条消息，不是「不认识的参数」
            r = self.cli(tmp, "bad", "--collision-model", "v3")
            self.assertEqual(r.returncode, 2, r.stderr[-1000:])
            r = self.cli(tmp, "v1both", "--spawn-jitter", "0.1", "--spawn-two-layer")      # v1 仍可叠用
            self.assertEqual(r.returncode, 0, r.stderr[-1000:])


if __name__ == "__main__":
    unittest.main()
