#!/usr/bin/env python3
"""碰撞代理 v2：3DGS 高斯 -> 导航坐标系 3D 体素占据 + 地面平面，沿轨迹扫掠。

与 v1（REVIEW #3 / AUDIT #2、#3）的差别：
  (a) 高斯先按 scene.nav_to_world 的逆变换到**导航系**再建占据，和渲染端同一坐标；
  (b) 只用不透明度 >= opacity_min 的高斯（Callinstrasse 2 m 薄片里 58% 的高斯
      不透明度 < 0.1，是看不见的幽灵墙），再去掉 smax > max_scale_m 的巨型"天空"高斯；
  (c) 每个高斯按 sigma_k 倍标准差的椭球落到体素，不只用中心；
  (d) 膨胀用固定的圆柱结构元素（xy 圆盘 + z 层），纯 numpy，不再依赖 scipy；
  (e) sweep() 沿线段按 <= 半体素步距采样，终点在墙另一侧也判碰。

仍是**近似代理**：高斯椭球不是真实表面，玻璃/薄叶/背面都可能漏；不能当 Gazebo
contact 或真实碰撞真值。
"""
import json
import math
import os
import sys

import numpy as np

from . import scene as scene_mod

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
# 参数单源：SITL lib agent 写出的 e1_mini.json；还没有时退回硬件 agent 的草案（同键名）。
# TODO(e1_mini.json 落地后删掉草案回退)。
E1_MINI_PARAMS = os.path.join(_REPO, "sim", "dynamics", "e1_mini.json")
E1_MINI_DRAFT = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"),
                             "uav_rgb", "tmp", "research-hw", "e1_mini_params_draft.json")
# 旧 Gazebo X3 的机体尺寸（转子半径 0.1 + 机臂，外径约 0.71 m -> 以前代理用 0.3 m），
# 只给旧数据复现：scene.json collision.envelope = {"radius_m":0.3,"half_height_m":0.3}。
X3_ENVELOPE = {"radius_m": 0.3, "half_height_m": 0.3, "source": "legacy x3 proxy"}

# opacity_min=0.3：单个高斯中心处的像素 alpha 就是 opacity，0.3 以下单独看是半透明雾。
# Callinstrasse z∈[1.8,2.2] 的 58030 个高斯里不透明度是双峰的：<0.05 占 55%、
# 0.95–1.0 占 24%，中间每 0.05 一档只有 1–2%；0.3 落在谷底（<0.3 累计 65.6%，
# 0.25–0.30 一档 738 个），阈值在 0.2–0.5 之间挪对结果影响 <4%。
DEFAULT_OPACITY_MIN = 0.3
DEFAULT_MAX_SCALE_M = scene_mod.RENDER_DEFAULTS["max_scale_m"]   # 与渲染端同值
# sigma_k=1.5：高斯在 1.5σ 处的 alpha 是 exp(-1.125)=0.32·opacity，和 opacity_min 用同一个
# "看得见就算实体"的 0.3 口径；1σ（0.61）会让相邻高斯之间留缝，3σ（0.01）把薄墙画成 3 倍厚。
DEFAULT_SIGMA_K = 1.5
PROXY_VERSION = "3dgs.occ.v2"


def load_body_envelope(path=None):
    """读 E1 MINI collision_envelope -> {"radius_m","half_height_m","source"}。

    草案键：value.radius_from_center / value.z_min / value.z_max（米）。
    """
    candidates = [path] if path else [E1_MINI_PARAMS, E1_MINI_DRAFT]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            with open(cand, "r", encoding="utf-8") as f:
                doc = json.load(f)
            env = doc.get("collision_envelope")
            val = env.get("value") if isinstance(env, dict) else None
            if not isinstance(val, dict):
                raise ValueError(f"{cand}: 缺 collision_envelope.value")
            radius = float(val["radius_from_center"])
            half_h = max(abs(float(val["z_min"])), abs(float(val["z_max"])))
            if not (radius > 0 and half_h > 0):
                raise ValueError(f"{cand}: collision_envelope 数值无效 {val}")
            return {"radius_m": radius, "half_height_m": half_h, "source": cand}
    raise FileNotFoundError(
        f"找不到 E1 MINI 参数：{E1_MINI_PARAMS} 或草案 {E1_MINI_DRAFT}；"
        "也可在 scene.json collision.envelope 里显式给 radius_m/half_height_m")


def quat_to_rotmat(q):
    """[N,4] (w,x,y,z) -> [N,3,3]；与 gsplat / INRIA PLY 的 rot_0..3 顺序一致。"""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((len(q), 3, 3), dtype=np.float64)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def _rigid_parts(nav_to_world):
    """4x4 -> (R, t)，只接受刚体变换（渲染端 viewmat_from_pose 也只按刚体用它）。"""
    m = np.asarray(nav_to_world, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("nav_to_world 必须是有限 4x4")
    r = m[:3, :3]
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-6) or np.linalg.det(r) < 0:
        raise ValueError("nav_to_world 的旋转块必须正交且 det=+1（不支持缩放/镜像）")
    return r, m[:3, 3]


class Occupancy:
    """导航系 3D 体素占据 + 地面；collides()/sweep() 查膨胀栅格，clearance() 查原始栅格。"""

    def __init__(self, means, scales=None, quats=None, opacities=None, *,
                 nav_to_world=None, z_range=(1.65, 2.35), voxel_m=0.1,
                 body_radius_m=0.102, body_half_height_m=0.028, inflate_m=0.0,
                 opacity_min=DEFAULT_OPACITY_MIN, max_scale_m=DEFAULT_MAX_SCALE_M,
                 sigma_k=DEFAULT_SIGMA_K, ground_z=0.0, pad_m=1.0):
        pts = np.asarray(means, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("means 必须是 Nx3")
        if not (z_range[0] < z_range[1]) or voxel_m <= 0 or inflate_m < 0:
            raise ValueError("z_range/voxel/inflate 参数无效")
        if not (body_radius_m > 0 and body_half_height_m > 0 and sigma_k > 0):
            raise ValueError("机体包络与 sigma_k 必须为正")
        self.z_range = (float(z_range[0]), float(z_range[1]))
        self.voxel = float(voxel_m)
        self.body_radius = float(body_radius_m)
        self.body_half_height = float(body_half_height_m)
        self.inflate = float(inflate_m)
        self.opacity_min = float(opacity_min)
        self.max_scale = None if max_scale_m is None else float(max_scale_m)
        self.sigma_k = float(sigma_k)
        self.ground_z = float(ground_z)
        self.n_points = len(pts)
        self.n_kept = 0
        self.n_in_range = 0
        self.n_occupied = 0
        pts = pts[:, :3]

        # (a) 世界 -> 导航：m_nav = R^T (m_world - t)，高斯朝向 R_nav = R^T R_g。
        r_m, t_m = _rigid_parts(np.eye(4) if nav_to_world is None else nav_to_world)
        pts = (pts - t_m) @ r_m
        keep = np.isfinite(pts).all(axis=1)
        if opacities is not None:
            op = np.asarray(opacities, dtype=np.float64).reshape(-1)
            keep &= op >= self.opacity_min                       # (b) 幽灵墙
        if scales is not None:
            sc = np.asarray(scales, dtype=np.float64)
            keep &= np.isfinite(sc).all(axis=1)
            if self.max_scale is not None:
                keep &= sc.max(axis=1) <= self.max_scale         # (b) 巨型天空高斯
        self.n_kept = int(keep.sum())
        pts = pts[keep]
        if scales is None:
            half = np.zeros_like(pts)                            # 纯点：只占中心体素
            rot = None
            sc = None
        else:
            # shortcut: 薄高斯（墙面/地面 σ 常只有 1 cm）按最小半厚 0.6 体素处理，否则
            # 体素中心采样会漏掉整面墙；代价是薄墙最多被画成 1 体素厚。
            sc = np.maximum(sc[keep], 0.6 * self.voxel / self.sigma_k)
            q = (np.tile([1.0, 0, 0, 0], (len(pts), 1)) if quats is None
                 else np.asarray(quats, dtype=np.float64)[keep])
            rot = np.einsum("ji,njk->nik", r_m, quat_to_rotmat(q))  # R_m^T R_g
            # 轴对齐半宽 e_i = k*sqrt(Σ_ii)，Σ = R diag(s²) R^T。
            half = self.sigma_k * np.sqrt(np.einsum("nij,nj->ni", rot ** 2, sc ** 2))
        # 只留 z 方向伸进栅格高度范围的高斯。
        zlo, zhi = self.z_range
        in_range = (pts[:, 2] + half[:, 2] >= zlo) & (pts[:, 2] - half[:, 2] <= zhi)
        self.n_in_range = int(in_range.sum())
        if self.n_in_range == 0:
            self._empty()
            return
        pts, half = pts[in_range], half[in_range]
        if rot is not None:
            rot, sc = rot[in_range], sc[in_range]
        lo = (pts - half)[:, :2].min(axis=0) - pad_m
        hi = (pts + half)[:, :2].max(axis=0) + pad_m
        self.origin = np.array([lo[0], lo[1], zlo])
        nxy = np.ceil((hi - lo) / self.voxel).astype(int) + 1
        nz = max(1, int(math.ceil((zhi - zlo) / self.voxel - 1e-9)))
        self.shape = (int(nxy[0]), int(nxy[1]), nz)
        if self.shape[0] * self.shape[1] * self.shape[2] > 40_000_000:
            raise ValueError(f"占据栅格过大 {self.shape}；提高 voxel_m、缩小 z_range 或裁剪场景")
        occ = np.zeros(self.shape, dtype=bool)
        self._rasterize(occ, pts, half, rot, sc)
        self.raw = occ
        self.n_occupied = int(occ.sum())
        ii, jj = np.nonzero(occ.any(axis=2))
        self._occ_xy = self.origin[:2] + (np.stack([ii, jj], 1) + 0.5) * self.voxel
        self.grid = self._dilate(occ)

    # -- 建栅格 ------------------------------------------------------------
    def _cell(self, p):
        return np.floor((np.asarray(p, dtype=np.float64) - self.origin) / self.voxel).astype(int)

    def _mark(self, occ, cells):
        shape = np.array(self.shape)
        ok = (cells >= 0).all(axis=1) & (cells < shape).all(axis=1)
        cells = cells[ok]
        occ[cells[:, 0], cells[:, 1], cells[:, 2]] = True

    def _rasterize(self, occ, pts, half, rot, sc):
        """(c) 椭球落体素：对每个高斯轴对齐包围盒内的体素中心做马氏距离 <= k 测试。

        均值所在体素总是占据（再小的高斯也至少占一格）。
        """
        c0 = self._cell(pts)
        self._mark(occ, c0)
        if rot is None:
            return
        ncell = np.ceil(half / self.voxel - 1e-9).astype(int)    # 每轴半宽体素数
        nmax = ncell.max(axis=1)
        inv_s = 1.0 / sc
        k2 = self.sigma_k ** 2
        nz = self.shape[2]
        # 小高斯（绝大多数）：按 nmax 分桶，整桶用同一张偏移表做向量化测试。
        n_small = 6
        for n in range(1, n_small + 1):
            idx = np.nonzero(nmax == n)[0]
            if len(idx) == 0:
                continue
            rng = np.arange(-n, n + 1)
            rz = np.arange(-min(n, nz - 1), min(n, nz - 1) + 1)   # z 层数有限，偏移表裁掉
            off = np.stack(np.meshgrid(rng, rng, rz, indexing="ij"), -1).reshape(-1, 3)
            chunk = max(1, 4_000_000 // len(off))
            for s in range(0, len(idx), chunk):
                sel = idx[s:s + chunk]
                cells = c0[sel][:, None, :] + off[None, :, :]            # [G,P,3]
                centers = self.origin + (cells + 0.5) * self.voxel
                d = centers - pts[sel][:, None, :]
                local = np.einsum("nji,npj->npi", rot[sel], d) * inv_s[sel][:, None, :]
                self._mark(occ, cells[(local ** 2).sum(-1) <= k2])
        # 大高斯（少数：墙面、地面大片）：逐个按自己的包围盒（裁到栅格）测试。
        for g in np.nonzero(nmax > n_small)[0]:
            lo = np.maximum(c0[g] - ncell[g], 0)
            hi = np.minimum(c0[g] + ncell[g], np.array(self.shape) - 1)
            if (hi < lo).any():
                continue
            ax = [np.arange(lo[i], hi[i] + 1) for i in range(3)]
            cells = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
            centers = self.origin + (cells + 0.5) * self.voxel
            local = (centers - pts[g]) @ rot[g] * inv_s[g]
            self._mark(occ, cells[(local ** 2).sum(-1) <= k2])

    def _dilate(self, occ):
        """(d) 固定圆柱结构元素：xy 半径 = 机体半径+余量，z 半高 = 机体半高；纯 numpy。

        体素数按四舍五入：占据体素本身已含 ±半格，再 ceil 会把 E1 的 0.102 m 在
        0.1 m 体素下翻倍成 0.2 m。
        """
        r_cells = int(round((self.body_radius + self.inflate) / self.voxel))
        h_cells = int(round(self.body_half_height / self.voxel))
        self.dilation_cells = (r_cells, h_cells)
        out = occ.copy()
        nx, ny, nz = occ.shape
        for di in range(-r_cells, r_cells + 1):
            for dj in range(-r_cells, r_cells + 1):
                if di * di + dj * dj > r_cells * r_cells:
                    continue
                for dk in range(-h_cells, h_cells + 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    src = occ[max(0, -di):nx - max(0, di),
                              max(0, -dj):ny - max(0, dj),
                              max(0, -dk):nz - max(0, dk)]
                    out[max(0, di):nx - max(0, -di),
                        max(0, dj):ny - max(0, -dj),
                        max(0, dk):nz - max(0, -dk)] |= src
        return out

    def _empty(self):
        self.shape = (0, 0, 0)
        self.origin = np.zeros(3)
        self.raw = self.grid = np.zeros((0, 0, 0), dtype=bool)
        self._occ_xy = np.zeros((0, 2))
        self.dilation_cells = (0, 0)

    # -- 查询 ---------------------------------------------------------------
    def _grid_hit(self, p):
        """p:[M,3] -> [M] bool（膨胀栅格；栅格外/高度范围外为 False）。"""
        p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
        hit = np.zeros(len(p), dtype=bool)
        if self.grid.size == 0:
            return hit
        inz = (p[:, 2] >= self.z_range[0]) & (p[:, 2] < self.z_range[1])
        c = self._cell(p)
        ok = inz & (c >= 0).all(axis=1) & (c < np.array(self.shape)).all(axis=1)
        hit[ok] = self.grid[c[ok, 0], c[ok, 1], c[ok, 2]]
        return hit

    def _ground_hit(self, z):
        return np.asarray(z, dtype=np.float64) - self.body_half_height <= self.ground_z

    def collides(self, x, y, z=2.0):
        """机体包络（圆柱）碰到占据体素，或离地低于机体半高。z 在高度范围外只查地面。"""
        if self._ground_hit(z):
            return True
        return bool(self._grid_hit([[x, y, z]])[0])

    def sweep(self, p0, p1):
        """(e) 沿线段 p0->p1 按 <= 半体素步距采样；返回 (是否碰, 首个碰撞点或 None)。

        shortcut: 动作窗口内的真实轨迹是弧线（偏航+前进过渡），这里按直线段近似；
        步长 hold 内位移 <= 0.7 m、偏航 <= 17°，弦与弧偏差 < 1 cm，比体素小。
        """
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)
        n = int(math.ceil(np.linalg.norm(p1 - p0) / (0.5 * self.voxel))) + 1
        ts = np.linspace(0.0, 1.0, max(2, n))
        pts = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        hit = self._ground_hit(pts[:, 2]) | self._grid_hit(pts)
        if not hit.any():
            return False, None
        return True, pts[int(np.argmax(hit))].tolist()

    def clearance(self, x, y, z=2.0):
        """到最近**原始**占据体素中心的水平距离（米，未减机体半径）与离地高度取小。"""
        ground = float(z - self.ground_z)
        if len(self._occ_xy) == 0:
            return ground
        d = float(np.min(np.hypot(self._occ_xy[:, 0] - x, self._occ_xy[:, 1] - y)))
        return float(min(ground, d))

    def summary(self):
        return {"proxy_version": PROXY_VERSION,
                "n_points": self.n_points, "n_kept": self.n_kept,
                "n_in_z_range": self.n_in_range, "n_occupied_cells": self.n_occupied,
                "z_range": list(self.z_range), "voxel_m": self.voxel,
                "body_radius_m": self.body_radius,
                "body_half_height_m": self.body_half_height,
                "envelope_source": getattr(self, "envelope_source", None),
                "margin_m": self.inflate, "dilation_cells": list(self.dilation_cells),
                "opacity_min": self.opacity_min, "max_scale_m": self.max_scale,
                "sigma_k": self.sigma_k, "ground_z": self.ground_z,
                "note": "approximate proxy: opacity-filtered Gaussian ellipsoids in nav frame, "
                        "swept along segments; not collision ground truth"}


def resolve_envelope(spec):
    """"e1_mini" | "x3" | {"radius_m","half_height_m"} -> 机体包络 dict。"""
    if spec in (None, "e1_mini"):
        return load_body_envelope()
    if spec == "x3":
        return dict(X3_ENVELOPE)
    if isinstance(spec, dict) and "radius_m" in spec and "half_height_m" in spec:
        return {"radius_m": float(spec["radius_m"]),
                "half_height_m": float(spec["half_height_m"]),
                "source": spec.get("source", "scene.json collision.envelope")}
    raise ValueError(f"collision.envelope 无效: {spec!r}")


def from_scene(scene, envelope=None, gaussians=None):
    """按 scene.json 的 collision 段建占据；机体包络默认 E1 MINI（e1_mini.json）。

    collision 段键：z_slice（栅格高度范围）、voxel_m、margin_m（机体半径外的余量，默认 0）、
    opacity_min、sigma_k、envelope（"e1_mini" | "x3" | {"radius_m","half_height_m"}）、
    max_scale_m（默认跟 render.max_scale_m 同值）。
    v1 的 inflate_m / drone_radius_m 把机体尺寸混在膨胀里，v2 一律忽略并提示。
    gaussians 可传 read_gaussian_ply 的结果，避免渲染端已读过的 511 MB PLY 再读一遍。
    """
    from . import scene as scene_mod
    c = dict(scene.get("collision", {}))
    legacy = {k: c[k] for k in ("inflate_m", "drone_radius_m") if k in c}
    if legacy:
        print(f"collision: 忽略 v1 键 {legacy}；v2 机体包络来自 collision.envelope"
              "（默认 E1 MINI），余量用 margin_m", file=sys.stderr)
    env = resolve_envelope(envelope if envelope is not None else c.get("envelope"))
    g = gaussians or scene_mod.read_gaussian_ply(scene["ply_path"])
    occ = Occupancy(g["means"], g["scales"], g["quats"], g["opacities"],
                    nav_to_world=scene.get("_nav_to_world"),
                    z_range=c.get("z_slice", (1.65, 2.35)),
                    voxel_m=c.get("voxel_m", 0.1),
                    body_radius_m=env["radius_m"], body_half_height_m=env["half_height_m"],
                    inflate_m=c.get("margin_m", 0.0),
                    opacity_min=c.get("opacity_min", DEFAULT_OPACITY_MIN),
                    max_scale_m=c.get("max_scale_m",
                                      scene_mod.render_options(scene)["max_scale_m"]),
                    sigma_k=c.get("sigma_k", DEFAULT_SIGMA_K),
                    ground_z=scene.get("ground_z", 0.0))
    occ.envelope_source = env.get("source")
    return occ


def _selftest():
    # 一堵 x=5 的薄墙（扁高斯，厚 1 cm、宽 0.6 m）+ 同位置一堵不透明度 0.05 的幽灵墙。
    ys = np.linspace(-1.0, 1.0, 11)
    means = np.array([(5.0, y, 2.0) for y in ys])
    scales = np.tile([0.01, 0.3, 0.3], (len(ys), 1))
    occ = Occupancy(means, scales, None, np.full(len(ys), 0.95),
                    z_range=(1.65, 2.35), voxel_m=0.1,
                    body_radius_m=0.102, body_half_height_m=0.028)
    assert occ.collides(5.0, 0.0, 2.0)
    assert occ.collides(5.0, 1.25, 2.0)          # 墙端 1σ=0.3 + 机体 0.1
    assert not occ.collides(5.0, 1.6, 2.0)
    assert not occ.collides(4.6, 0.0, 2.0) and not occ.collides(5.3, 0.0, 2.0)
    hit, pt = occ.sweep((4.6, 0.0, 2.0), (5.3, 0.0, 2.0))   # 终点在墙另一侧
    assert hit and abs(pt[0] - 4.9) < 0.11, pt
    assert occ.collides(5.0, 0.0, 0.02)                     # 地面
    ghost = Occupancy(means, scales, None, np.full(len(ys), 0.05),
                      z_range=(1.65, 2.35), voxel_m=0.1,
                      body_radius_m=0.102, body_half_height_m=0.028)
    assert not ghost.collides(5.0, 0.0, 2.0) and not ghost.sweep((4.6, 0, 2), (5.3, 0, 2))[0]
    print("collision selftest PASS", occ.summary())


if __name__ == "__main__":
    _selftest()
