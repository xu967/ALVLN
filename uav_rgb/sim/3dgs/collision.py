#!/usr/bin/env python3
"""碰撞代理：3DGS 点云 -> 2 m 高度切片体素占据 + 地面平面。

重要限制：**3DGS 不提供可靠碰撞几何**。这里把高斯 mean（或 COLMAP 稀疏点）
在飞行高度附近的薄片投影成 2D 占据栅格并膨胀，只是近似代理；薄树叶、玻璃、
视角外背面都可能漏检，不能当成真实碰撞真值。真实几何应来自点云/网格/场景真值。
"""
import math

import numpy as np

try:
    from scipy import ndimage as _ndimage
    from scipy.spatial import cKDTree as _KDTree
except ImportError:          # 纯 numpy 兜底：小场景单元测试可跑
    _ndimage = None
    _KDTree = None


class Occupancy:
    """切片体素占据 + 地面；collides() 用膨胀后的栅格，clearance() 用原始点。"""

    def __init__(self, points, z_slice=(1.65, 2.35), voxel_m=0.1,
                 inflate_m=0.35, drone_radius_m=0.3, ground_z=0.0, pad_m=1.0):
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("points 必须是 Nx3")
        if not (z_slice[0] < z_slice[1]) or voxel_m <= 0 or inflate_m < 0:
            raise ValueError("z_slice/voxel/inflate 参数无效")
        self.z_slice = tuple(float(v) for v in z_slice)
        self.voxel = float(voxel_m)
        self.inflate = float(inflate_m)
        self.radius = float(drone_radius_m)
        self.ground_z = float(ground_z)
        self.n_points = len(pts)
        self.n_sliced = 0
        self._tree = None
        if len(pts) == 0:
            self._empty()
            return
        z = pts[:, 2]
        keep = np.isfinite(pts).all(axis=1) & (z >= self.z_slice[0]) & (z <= self.z_slice[1])
        sliced = pts[keep]
        self.n_sliced = len(sliced)
        if len(sliced) == 0:
            self._empty()
            return
        if _KDTree is not None:
            self._tree = _KDTree(sliced[:, :2])
        else:
            self._tree = None
        self._points_xy = sliced[:, :2]
        lo = sliced[:, :2].min(axis=0) - pad_m
        hi = sliced[:, :2].max(axis=0) + pad_m
        self.origin = lo
        shape = np.ceil((hi - lo) / self.voxel).astype(int) + 1
        if shape[0] * shape[1] > 40_000_000:
            raise ValueError(f"占据栅格过大 {shape}；提高 voxel_m 或裁剪点云")
        occ = np.zeros(shape, dtype=bool)
        ij = np.floor((sliced[:, :2] - lo) / self.voxel).astype(int)
        occ[ij[:, 0], ij[:, 1]] = True
        cells = int(math.ceil(self.inflate / self.voxel))
        if cells > 0:
            if _ndimage is not None:
                occ = _ndimage.binary_dilation(occ, iterations=cells)
            else:                       # 无 scipy：方形结构元素迭代膨胀
                st = np.ones((3, 3), bool)
                for _ in range(cells):
                    p = np.pad(occ, 1)
                    occ = sum(p[1 + di:1 + di + occ.shape[0],
                                1 + dj:1 + dj + occ.shape[1]]
                              for di in (-1, 0, 1) for dj in (-1, 0, 1)) > 0
        else:
            occ = occ.copy()
        self.grid = occ

    def _empty(self):
        self.grid = np.zeros((0, 0), dtype=bool)
        self.origin = np.zeros(2)
        self._points_xy = None

    def collides(self, x, y, z=2.0):
        """位于膨胀占据栅格内，或离地低于 radius。z 在切片外只检查地面。"""
        if z - self.radius <= self.ground_z:
            return True
        if self.grid.size == 0 or not (self.z_slice[0] <= z <= self.z_slice[1]):
            return False
        ix, iy = np.floor((np.array([x, y], dtype=float) - self.origin) / self.voxel).astype(int)
        if ix < 0 or iy < 0 or ix >= self.grid.shape[0] or iy >= self.grid.shape[1]:
            return False
        return bool(self.grid[ix, iy])

    def clearance(self, x, y, z=2.0):
        """到最近切片点的水平距离（米，未减机体半径）与到地高度取小。"""
        ground = float(z - self.ground_z)
        if self._tree is not None:
            d, _ = self._tree.query([x, y], k=1)
        elif getattr(self, "_points_xy", None) is not None and len(self._points_xy):
            d = float(np.min(np.linalg.norm(self._points_xy - [x, y], axis=1)))
        else:
            return ground
        return float(min(ground, d))

    def summary(self):
        return {"n_points": self.n_points, "n_sliced": self.n_sliced,
                "z_slice": list(self.z_slice), "voxel_m": self.voxel,
                "inflate_m": self.inflate, "drone_radius_m": self.radius,
                "ground_z": self.ground_z,
                "note": "approximate proxy: 3DGS means are not collision geometry"}


def from_scene(scene):
    """按 scene.json 的 collision 段从 PLY 建占据。"""
    from . import scene as scene_mod
    g = scene_mod.read_gaussian_ply(scene["ply_path"])
    c = scene.get("collision", {})
    return Occupancy(g["means"], z_slice=c.get("z_slice", (1.65, 2.35)),
                     voxel_m=c.get("voxel_m", 0.1),
                     inflate_m=c.get("inflate_m", 0.35),
                     drone_radius_m=c.get("drone_radius_m", 0.3),
                     ground_z=scene.get("ground_z", 0.0))


def _selftest():
    # 一个 2 m 高的方块（顶面/侧面点到 2 m）+ 地面点；检查切片、膨胀、地面。
    pts = []
    for x in np.linspace(4, 5, 11):
        for y in np.linspace(-0.5, 0.5, 11):
            pts.append((x, y, 2.0))
    pts += [(0, 10, 0.0), (30, 0, 2.0)]
    occ = Occupancy(np.array(pts), z_slice=(1.65, 2.35), voxel_m=0.1,
                    inflate_m=0.3, drone_radius_m=0.3)
    assert occ.collides(4.5, 0, 2.0)
    assert occ.collides(4.5, 0.7, 2.0)        # 膨胀 0.3 覆盖 0.5+0.2
    assert not occ.collides(4.5, 1.2, 2.0)
    assert occ.collides(4.5, 0, 0.2)          # 地面
    assert occ.clearance(4.5, 0, 2.0) < 0.01
    assert occ.clearance(4.5, 2.0, 2.0) > 1.0
    print("collision selftest PASS", occ.summary())


if __name__ == "__main__":
    _selftest()
