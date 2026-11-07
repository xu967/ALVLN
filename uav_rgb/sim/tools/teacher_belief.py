#!/usr/bin/env python3
"""teacher_belief：信念边缘化的运动原语软标签教师（P1）。

学生只看一帧单目图，教师却知道整张地图——直接蒸馏「全知」硬标签，学生学不到
「看不见的地方要留余地」。这里按 A2D（Warrington 2021, Asymmetric DAgger）的做法把
专家在学生的观测后验上边缘化：π*(a|o) = Σ_s p(s|o) π_E(a|s)，近似成 QMDP（Littman 1995）：
  1. 可见集合：从学生位姿按 60° 水平视场发射射线，被遮挡/视场外 = 未知；
  2. 采 K 个世界假设：已知障碍固定，未知区域按场景先验重采（柱阵密度 / 森林档位 /
     3DGS 局部占据率），同时采执行器延迟与悬停漂移；
  3. 每个假设跑 teacher_kino.search_costs 得四个首原语的 cost_to_go（秒），
     QMDP 平均 Q̄(a) = Σ_k w_k Q_k(a)（可切 CVaR 看尾部）；
  4. p(a) ∝ exp(−Q̄(a)/α)，α 默认 = 一步 step_s（软最优：Levine 2018，温度 = 代价尺度）；
     全部假设全部动作都不可行 → SAFE 独热 + fail_closed。
输出每步 {p[4], q_bar[4], k_feasible, fail_closed, visible_frac}；CLI 把整集标成
export_soft_targets 同款 npz（键 {episode_id}/{step}/logits = log p），由 tools/join_soft_targets.py
拼成 UAVSOFT1。

纯 numpy（teacher_kino 是 stdlib）。两类场景：
  - Gazebo 柱阵/森林：障碍 = 方柱/圆柱列表，净空用 teacher_kino 同一套方框距离表（0.05 m 格），
    所以全可见、K=1 时与 teacher_kino.plan_action 逐位一致（test_teacher_belief 用例 a）；
  - 3DGS：障碍 = sim/3dgs/collision.Occupancy 的 2D 占据（0.2 m 体素），净空 = 到最近占据格中心
    距离 − FACE_OFFSET_3DGS，让 teacher_kino 的硬阈值与碰撞代理的 4 邻域膨胀同口径。

用法（标注 3DGS 采集集）：
  python3 sim/tools/teacher_belief.py --occ-npz occ2d.npz --out soft.npz --stats stats.json \\
      [--k 8] [--seed 0] [--alpha-steps 1.0] [--aggregate qmdp|cvar] [--workers 16] EP_DIR...
  occ2d.npz 由 collision.from_scene 的 raw.any(z) 落盘（tmp/agent4-teacher-p1/build_occ_cache.py）。
自检：python3 sim/tools/test_teacher_belief.py
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_kino as tk  # noqa: E402
from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT  # noqa: E402

TEACHER_VERSION = "uavrgb.teacher_belief.v1"
ACTIONS4 = (SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT)   # 输出向量的列序 = 动作编号 0..3

# ---------------------------------------------------------------- 配置区
K_DEFAULT = 8            # 假设数：8 个假设 × 25 ms 一次搜索 ≈ 0.2 s/步；16 以上只让熵更平滑
FOV_DEG = 60.0           # 相机水平视场：uav_rgb.sdf / uav_rgb_e1.sdf / scene.json camera 都是 60°
VIS_RANGE_M = 8.0        # 射线最远距离：3DGS 远处是雾、Gazebo 深度相机也只到几米；8 m 已超视野 2.4 m 三倍
N_RAYS = 121             # 0.5° 一条：8 m 处相邻射线间距 7 cm < 最细栅格 0.1 m，不漏格
SELF_FREE_M = 0.5        # 机体周围当作已知自由：飞到这里没撞，且先验障碍进了 CLEAR_SOFT 带会让所有假设的
                         # 首步都「已越界」，软标签退化成常数；取软带宽 0.5 m
WINDOW_M = 4.0           # 3DGS 假设栅格半宽：视野 16 步×0.15 m=2.4 m + 停止检查 + 目标圆，窗外当自由
VISIBLE_FRAC_R_M = 3.0   # visible_frac 统计半径：前方 3 m 半圆内被看到的格比例（60° 无遮挡上限 1/3）
GEO_CELL_3DGS_M = 0.2    # 3DGS 已知世界 Dijkstra 栅格 = 体素尺寸；Gazebo 用 teacher_kino 的 0.05
FACE_OFFSET_3DGS_M = 0.3 - tk.CLEAR_PLAN   # 碰撞代理：机体中心格与占据格 4 邻接即撞 ⇔ 格心距 < 0.3 m；
                                           # 减掉它后 teacher_kino 的 CLEAR_PLAN 阈值正好落在 0.3 m
BLOB_R_3DGS_M = 0.4      # 3DGS 未知区先验障碍半径：Callinstrasse 树篱/树干截面 0.5–1 m，取半径 0.4
PRIOR_LOCAL_R_M = 10.0   # 3DGS 局部占据率统计半径：10 m 内真实占据格比例（场景先验，不是当前帧信息）
LATENCY_GAZEBO_S = (0.10, 0.20)  # 执行器延迟采样区间（任务书给定；名义 camera_latency 0.1）
LATENCY_3DGS_S = (0.0, 0.10)     # 3DGS dispatch 语义没有过期悬停，只剩固件处理延迟；名义 0
DRIFT_MAX_MPS = 0.03     # 悬停漂移幅度：pilot trace 悬停段 |vel| 0.02–0.03 m/s
ALPHA_STEPS = 1.0        # 温度 α = ALPHA_STEPS × step_s（一步前飞的代价）
C_COLL_STEPS = 2.0 * tk.HORIZON_STEPS  # inf 的有限替身（步数）：撞 = 比最远视野还贵一倍，可参与加权平均
CVAR_Q = 0.5             # --aggregate cvar 时取代价最差的这一比例做平均
OUTSIDE_CLEAR_M = 10.0   # 假设栅格之外视为自由（搜索到不了那么远）
# Gazebo 柱阵先验：scene_gen 8–9 根 0.4 m 方柱落在 x∈[1.5,5.0]×y∈[-1.7,1.7]，可通行缝 ≥ MIN_GAP 0.9
PRIOR_PILLARS = {"kind": "box", "half": 0.2, "density": 8.5 / (3.5 * 3.4),
                 "region": (1.3, 5.2, -1.9, 1.9), "min_gap": 0.9}
# 森林先验：scene_forest 三档树干数 / (4 m × 3 m)，半径 U(0.05,0.15)，缝 = DENSITY_GAP_M
PRIOR_FOREST = {tier: {"kind": "disc", "radius": (0.05, 0.15), "density": n / 12.0,
                       "region": (1.5, 5.5, -1.5, 1.5), "min_gap": gap}
                for tier, n, gap in (("sparse", 16.0, 0.45), ("medium", 22.5, 0.35), ("dense", 29.5, 0.28))}
MODEL_GAZEBO = tk.NOMINAL                      # hold 1.0、窗口首 0.1 s 过期悬停
MODEL_3DGS = tk.Model(step_s=0.5, latency_s=0.0)  # cf_sitl dispatch：until_s = t_obs + hold
# -----------------------------------------------------------------------


class Scene2D:
    """真实世界的 2D 占据栅格（只给可见性与先验用；规划净空另算）。grid[iy, ix]，(x0, y0) 是格 [0,0] 中心。"""

    def __init__(self, occ, x0, y0, cell):
        self.occ = np.asarray(occ, dtype=bool)
        self.ny, self.nx = self.occ.shape
        self.x0, self.y0, self.cell = float(x0), float(y0), float(cell)

    def cells(self, xy):
        """[N,2] 世界坐标 -> [N,2] (ix, iy)（可能越界，调用方自己裁）。"""
        return np.rint((np.asarray(xy, dtype=np.float64) - (self.x0, self.y0)) / self.cell).astype(int)

    def inside(self, c):
        return (c[..., 0] >= 0) & (c[..., 0] < self.nx) & (c[..., 1] >= 0) & (c[..., 1] < self.ny)

    def center(self, ix, iy):
        return self.x0 + ix * self.cell, self.y0 + iy * self.cell

    @classmethod
    def from_occ_npz(cls, path):
        """build_occ_cache.py 的输出：occ (nx,ny) bool、origin_xy（格 [0,0] 的角）、voxel。"""
        with np.load(path, allow_pickle=False) as z:
            occ, origin, voxel = z["occ"], z["origin_xy"], float(z["voxel"])
        return cls(occ.T, origin[0] + 0.5 * voxel, origin[1] + 0.5 * voxel, voxel)

    @classmethod
    def from_obstacles(cls, boxes, discs, x0, y0, nx, ny, cell):
        """Gazebo：方柱 (x, y, hx, hy) + 圆柱 (x, y, r) 栅格化（表面距离 ≤ 0 的格为占据）。"""
        dist = np.minimum(box_dist_table(boxes, x0, y0, nx, ny, cell), disc_dist_table(discs, x0, y0, nx, ny, cell))
        return cls(dist <= 0.0, x0, y0, cell)


def box_dist_table(boxes, x0, y0, nx, ny, cell):
    """teacher_kino._box_dist 的向量化版：[ny, nx] 到最近方柱表面的距离（柱内为负），同一组格点同一公式。"""
    xs = x0 + np.arange(nx) * cell
    ys = y0 + np.arange(ny) * cell
    gx, gy = np.meshgrid(xs, ys)              # [ny, nx]
    best = np.full((ny, nx), np.inf)
    for bx, by, hx, hy in boxes:
        dx, dy = np.abs(gx - bx) - hx, np.abs(gy - by) - hy
        outside = (dx > 0.0) | (dy > 0.0)
        d = np.where(outside, np.hypot(np.maximum(dx, 0.0), np.maximum(dy, 0.0)), np.maximum(dx, dy))
        best = np.minimum(best, d)
    return best


def disc_dist_table(discs, x0, y0, nx, ny, cell):
    """[ny, nx] 到最近圆柱表面的距离（森林树干）；discs = [(x, y, r)]。"""
    xs = x0 + np.arange(nx) * cell
    ys = y0 + np.arange(ny) * cell
    gx, gy = np.meshgrid(xs, ys)
    best = np.full((ny, nx), np.inf)
    for cx, cy, r in discs:
        best = np.minimum(best, np.hypot(gx - cx, gy - cy) - r)
    return best


def occ_dist_table(occ, cell, face_offset):
    """[ny, nx] 到最近占据格中心的距离 − face_offset（纯 numpy 暴力 EDT；窗口 ≤ 80×80 格够快）。"""
    ny, nx = occ.shape
    iy, ix = np.nonzero(occ)
    if len(ix) == 0:
        return np.full((ny, nx), OUTSIDE_CLEAR_M)
    gy, gx = np.mgrid[0:ny, 0:nx]
    pts = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)   # [P,2]
    occ_pts = np.stack([ix, iy], 1).astype(np.float64)                # [M,2]
    best = np.full(len(pts), np.inf)
    chunk = max(1, 4_000_000 // max(1, len(occ_pts)))
    for s in range(0, len(pts), chunk):
        d = np.hypot(pts[s:s + chunk, None, 0] - occ_pts[None, :, 0],
                     pts[s:s + chunk, None, 1] - occ_pts[None, :, 1])
        best[s:s + chunk] = d.min(axis=1)
    return (best * cell - face_offset).reshape(ny, nx)


class HypField:
    """一个世界假设的净空查表 + 共享的已知世界测地线；鸭子类型兼容 teacher_kino 的 field。"""

    def __init__(self, dist, x0, y0, cell, geo):
        self.ny, self.nx = dist.shape
        self.dist = dist.ravel().tolist()      # list 索引比 numpy 标量快 3 倍，搜索里每步查几十次
        self.x0, self.y0, self.cell, self.geo = float(x0), float(y0), float(cell), geo

    def clearance(self, x, y):
        i = int(round((x - self.x0) / self.cell))
        j = int(round((y - self.y0) / self.cell))
        if 0 <= i < self.nx and 0 <= j < self.ny:
            return self.dist[j * self.nx + i]
        return OUTSIDE_CLEAR_M

    def geodesic(self, x, y):
        return self.geo.geodesic(x, y)


def known_geodesic(occ_known, x0, y0, cell, goal, clear):
    """已知世界（只有看见的障碍）的目标根 Dijkstra：对所有假设都是下界，一步算一次，K 个假设共用。

    直接用 teacher_kino._Field，启发公式与 v1 相同；粗格上把阻塞阈值放宽半个对角格，保持下界。
    """
    ny, nx = occ_known.shape
    dist = occ_dist_table(occ_known, cell, 0.5 * cell) if occ_known.any() else np.full((ny, nx), OUTSIDE_CLEAR_M)
    thresh = max(0.0, clear - 0.5 * cell * math.sqrt(2.0))
    return tk._Field(dist.ravel().tolist(), x0, y0, nx, ny, goal, thresh, cell)


# ---------------------------------------------------------------- 可见性
def ray_cast(scene, x, y, yaw, fov_deg=FOV_DEG, range_m=VIS_RANGE_M, n_rays=N_RAYS):
    """从 (x, y) 朝 yaw ± fov/2 发射 n_rays 条射线，返回 (seen_free, seen_occ) 两张 scene 栅格布尔图。

    射线按半格步进；碰到第一格占据就停（该格记为看见的障碍）。栅格外视为自由并继续（不记）。
    """
    seen_free = np.zeros_like(scene.occ)
    seen_occ = np.zeros_like(scene.occ)
    angles = yaw + np.radians(np.linspace(-fov_deg / 2.0, fov_deg / 2.0, n_rays))
    rs = np.arange(0.0, range_m + 1e-9, 0.5 * scene.cell)
    px = x + rs[None, :] * np.cos(angles)[:, None]           # [R, M]
    py = y + rs[None, :] * np.sin(angles)[:, None]
    c = scene.cells(np.stack([px, py], -1))                   # [R, M, 2]
    inside = scene.inside(c)
    cx = np.clip(c[..., 0], 0, scene.nx - 1)
    cy = np.clip(c[..., 1], 0, scene.ny - 1)
    hit = inside & scene.occ[cy, cx]                          # [R, M]
    any_hit = hit.any(axis=1)
    first = np.where(any_hit, hit.argmax(axis=1), hit.shape[1])
    before = np.arange(hit.shape[1])[None, :] < first[:, None]
    sel = before & inside
    seen_free[cy[sel], cx[sel]] = True
    if any_hit.any():
        rows = np.nonzero(any_hit)[0]
        seen_occ[cy[rows, first[rows]], cx[rows, first[rows]]] = True
    # 机体周围（真实为空的格）算已知自由：飞到这里没撞。
    cc = scene.cells([[x, y]])[0]
    r = int(math.ceil(SELF_FREE_M / scene.cell))
    ys = slice(max(0, cc[1] - r), min(scene.ny, cc[1] + r + 1))
    xs = slice(max(0, cc[0] - r), min(scene.nx, cc[0] + r + 1))
    gy, gx = np.mgrid[ys, xs]
    near = np.hypot((gx - cc[0]) * scene.cell, (gy - cc[1]) * scene.cell) <= SELF_FREE_M
    sub_free = seen_free[ys, xs]
    sub_free |= near & ~scene.occ[ys, xs]
    seen_free &= ~seen_occ
    return seen_free, seen_occ


def visible_frac(scene, seen_free, seen_occ, x, y, yaw, radius=VISIBLE_FRAC_R_M):
    """前方 radius 半圆内被看到（自由或障碍）的格比例。"""
    cc = scene.cells([[x, y]])[0]
    r = int(math.ceil(radius / scene.cell))
    ys = slice(max(0, cc[1] - r), min(scene.ny, cc[1] + r + 1))
    xs = slice(max(0, cc[0] - r), min(scene.nx, cc[0] + r + 1))
    gy, gx = np.mgrid[ys, xs]
    dx, dy = (gx - cc[0]) * scene.cell, (gy - cc[1]) * scene.cell
    ahead = (np.hypot(dx, dy) <= radius) & (dx * math.cos(yaw) + dy * math.sin(yaw) >= 0.0)
    n = int(ahead.sum())
    if n == 0:
        return 0.0
    seen = (seen_free | seen_occ)[ys, xs]
    return float((ahead & seen).sum() / n)


# ---------------------------------------------------------------- 世界假设
class Hypothesis:
    """一个采样出的世界：障碍列表（Gazebo）或占据图（3DGS）+ 执行模型。"""

    def __init__(self, model, boxes=(), discs=(), occ=None):
        self.model, self.boxes, self.discs, self.occ = model, tuple(boxes), tuple(discs), occ


def sample_model(rng, base, latency_range):
    """执行器延迟 ~ U(latency_range)，悬停漂移方向均匀、幅度 ~ U(0, DRIFT_MAX)；延迟不超过一步。"""
    lat = min(float(rng.uniform(*latency_range)), 0.9 * base.step_s)
    ang, mag = rng.uniform(0.0, 2.0 * math.pi), rng.uniform(0.0, DRIFT_MAX_MPS)
    return tk.Model(base.step_s, lat, mag * math.cos(ang), mag * math.sin(ang))


def _gap_box(ax, ay, bx, by, size=0.4):
    """两方柱可通行缝（scene_gen._pillar_gap 同式）。"""
    return max(max(0.0, abs(ax - bx) - size), max(0.0, abs(ay - by) - size))


def sample_obstacles_gazebo(rng, scene, seen_free, known, prior):
    """在未知区按先验密度撒障碍：中心落在先验区域 ∩ 未知格，脚印不压已知自由格，且与已有障碍留 min_gap。

    返回新增障碍列表：box -> (x, y, hx, hy)；disc -> (x, y, r)。
    """
    x_lo, x_hi, y_lo, y_hi = prior["region"]
    ix0, ix1 = scene.cells([[x_lo, y_lo], [x_hi, y_hi]])
    ix0 = np.maximum(ix0, 0)
    ix1 = np.minimum(ix1, [scene.nx - 1, scene.ny - 1])
    sub_free = seen_free[ix0[1]:ix1[1] + 1, ix0[0]:ix1[0] + 1]
    sub_occ = scene.occ[ix0[1]:ix1[1] + 1, ix0[0]:ix1[0] + 1]
    unknown = ~(sub_free | sub_occ)
    cand = np.argwhere(unknown)                               # [(iy, ix)] 子窗坐标
    if len(cand) == 0:
        return []
    area = len(cand) * scene.cell ** 2
    n = int(rng.poisson(prior["density"] * area))
    out = []
    picks = cand[rng.integers(0, len(cand), size=max(0, n))]
    for iy, ix in picks:
        cx, cy = scene.center(ix + ix0[0], iy + ix0[1])
        cx += float(rng.uniform(-0.5, 0.5)) * scene.cell
        cy += float(rng.uniform(-0.5, 0.5)) * scene.cell
        if prior["kind"] == "box":
            h = prior["half"]
            foot = (h, h)
        else:
            h = float(rng.uniform(*prior["radius"]))
            foot = (h, h)
        # 脚印（加半格余量）不能压到已知自由格
        c0 = scene.cells([[cx - foot[0], cy - foot[1]]])[0]
        c1 = scene.cells([[cx + foot[0], cy + foot[1]]])[0]
        c0 = np.maximum(c0, 0)
        c1 = np.minimum(c1, [scene.nx - 1, scene.ny - 1])
        if seen_free[c0[1]:c1[1] + 1, c0[0]:c1[0] + 1].any():
            continue
        ok = True
        for o in list(known) + out:
            if prior["kind"] == "box":
                if _gap_box(cx, cy, o[0], o[1]) < prior["min_gap"]:
                    ok = False
                    break
            elif math.hypot(cx - o[0], cy - o[1]) - h - o[2] < prior["min_gap"]:
                ok = False
                break
        if ok:
            out.append((cx, cy, h, h) if prior["kind"] == "box" else (cx, cy, h))
    return out


def sample_occ_3dgs(rng, scene, seen_free, seen_occ, win, rho):
    """3DGS 假设占据图（窗口内）：已知障碍 + 未知区按局部占据率 rho 撒半径 BLOB_R 的圆斑。

    win = (ix0, iy0, nx, ny) 窗口在 scene 栅格里的位置；返回 [ny, nx] bool。
    """
    ix0, iy0, nx, ny = win
    occ = seen_occ[iy0:iy0 + ny, ix0:ix0 + nx].copy()
    free = seen_free[iy0:iy0 + ny, ix0:ix0 + nx]
    unknown = ~(occ | free)
    n_unknown = int(unknown.sum())
    if n_unknown == 0 or rho <= 0.0:
        return occ
    lam = rho / (math.pi * BLOB_R_3DGS_M ** 2)                # 圆斑数密度，使期望占据率 ≈ rho
    n = int(rng.poisson(lam * n_unknown * scene.cell ** 2))
    if n == 0:
        return occ
    cand = np.argwhere(unknown)
    picks = cand[rng.integers(0, len(cand), size=n)]
    gy, gx = np.mgrid[0:ny, 0:nx]
    r_cells = BLOB_R_3DGS_M / scene.cell
    for iy, ix in picks:
        blob = np.hypot(gx - ix, gy - iy) <= r_cells
        if (blob & free).any():                               # 圆斑压到看见的自由格就丢
            continue
        occ |= blob
    return occ


# ---------------------------------------------------------------- 一步标签
def aggregate(q, mode="qmdp", cvar_q=CVAR_Q):
    """[K,4] 有限代价 -> [4]：qmdp = 均匀加权平均；cvar = 每个动作取最差 cvar_q 比例的平均。"""
    if mode == "qmdp":
        return q.mean(axis=0)
    m = max(1, int(math.ceil(cvar_q * q.shape[0])))
    return np.sort(q, axis=0)[-m:].mean(axis=0)


def softmax_costs(q_bar, alpha):
    z = -(q_bar - q_bar.min()) / alpha
    p = np.exp(z)
    return p / p.sum()


class BeliefTeacher:
    """场景 + 先验 + 模型打包；label(state) 出一步软标签。

    scene：Scene2D 真实占据（只用于可见性/先验/自检）；
    boxes/discs：Gazebo 的方柱 (x, y, hx, hy) / 圆柱 (x, y, r) 全集（可见性决定哪些算「看见」）；3DGS 不传；
    prior：PRIOR_PILLARS / PRIOR_FOREST[tier] / {"kind": "occ"}（3DGS）。
    """

    def __init__(self, scene, prior, model=MODEL_GAZEBO, boxes=(), discs=(), latency_range=None,
                 k=K_DEFAULT, alpha_steps=ALPHA_STEPS, mode="qmdp", cvar_q=CVAR_Q,
                 fov_deg=FOV_DEG, range_m=VIS_RANGE_M):
        self.scene, self.prior, self.model = scene, prior, model
        self.boxes, self.discs = tuple(boxes), tuple(discs)
        self.fov_deg, self.range_m = float(fov_deg), float(range_m)   # 测试用 360° 造「全可见」
        self.latency_range = latency_range if latency_range is not None else (
            LATENCY_3DGS_S if prior.get("kind") == "occ" else LATENCY_GAZEBO_S)
        self.k, self.alpha, self.mode, self.cvar_q = int(k), alpha_steps * model.step_s, mode, cvar_q
        if self.k < 1:
            raise ValueError("K 至少为 1")

    # -- Gazebo：假设 = 方柱/圆柱列表，净空表用 teacher_kino 同一套格点 ------------
    def _seen(self, ob, seen_occ):
        """障碍脚印内有任一格被射线击中就算看见。ob = (x, y, hx, hy) 或 (x, y, r)。"""
        hx, hy = (ob[2], ob[3]) if len(ob) == 4 else (ob[2], ob[2])
        c0 = np.maximum(self.scene.cells([[ob[0] - hx, ob[1] - hy]])[0], 0)
        c1 = np.minimum(self.scene.cells([[ob[0] + hx, ob[1] + hy]])[0], [self.scene.nx - 1, self.scene.ny - 1])
        return bool(seen_occ[c0[1]:c1[1] + 1, c0[0]:c1[0] + 1].any())

    def _fields_gazebo(self, rng, x, y, yaw, goal, clear, seen_free, seen_occ):
        known_boxes = [b for b in self.boxes if self._seen(b, seen_occ)]
        known_discs = [d for d in self.discs if self._seen(d, seen_occ)]
        # 格点范围按真实障碍全集定（与 teacher_kino._Field.extent 同），全可见时距离表与 v1 逐值相同
        all_ob = list(self.boxes) + [(d[0], d[1], d[2], d[2]) for d in self.discs] or [(x, y, 0.0, 0.0)]
        x0, y0, nx, ny = tk._Field.extent(all_ob, goal)
        cell = tk.GRID
        base = np.minimum(box_dist_table(known_boxes, x0, y0, nx, ny, cell),
                          disc_dist_table(known_discs, x0, y0, nx, ny, cell))
        geo = tk._Field(base.ravel().tolist(), x0, y0, nx, ny, goal, clear, cell, tuple(known_boxes))
        known = known_boxes if self.prior["kind"] == "box" else known_discs
        fields = []
        for _ in range(self.k):
            if self.k == 1:
                fields.append((HypField(base, x0, y0, cell, geo), self.model))
                continue
            extra = sample_obstacles_gazebo(rng, self.scene, seen_free, known, self.prior)
            table = box_dist_table if self.prior["kind"] == "box" else disc_dist_table
            dist = np.minimum(base, table(extra, x0, y0, nx, ny, cell)) if extra else base
            fields.append((HypField(dist, x0, y0, cell, geo), sample_model(rng, self.model, self.latency_range)))
        return fields, len(known_boxes) + len(known_discs)

    # -- 3DGS：假设 = 窗口占据图 ---------------------------------------------------
    def _fields_3dgs(self, rng, x, y, yaw, goal, clear, seen_free, seen_occ):
        sc = self.scene
        half = int(math.ceil(WINDOW_M / sc.cell))
        cc = sc.cells([[x, y]])[0]
        ix0, iy0 = max(0, cc[0] - half), max(0, cc[1] - half)
        ix1, iy1 = min(sc.nx - 1, cc[0] + half), min(sc.ny - 1, cc[1] + half)
        win = (ix0, iy0, ix1 - ix0 + 1, iy1 - iy0 + 1)
        wx0, wy0 = sc.center(ix0, iy0)
        # 已知世界测地线：窗口 ∪ 目标的包围盒，粗格 = 体素
        gx0 = min(wx0, goal[0]) - 1.0
        gy0 = min(wy0, goal[1]) - 1.0
        gx1 = max(sc.center(ix1, iy1)[0], goal[0]) + 1.0
        gy1 = max(sc.center(ix1, iy1)[1], goal[1]) + 1.0
        gnx, gny = int(math.ceil((gx1 - gx0) / sc.cell)) + 1, int(math.ceil((gy1 - gy0) / sc.cell)) + 1
        gc0 = sc.cells([[gx0, gy0]])[0]
        occ_known = np.zeros((gny, gnx), dtype=bool)
        sy0, sx0 = max(0, gc0[1]), max(0, gc0[0])
        sy1, sx1 = min(sc.ny, gc0[1] + gny), min(sc.nx, gc0[0] + gnx)
        if sy1 > sy0 and sx1 > sx0:
            occ_known[sy0 - gc0[1]:sy1 - gc0[1], sx0 - gc0[0]:sx1 - gc0[0]] = seen_occ[sy0:sy1, sx0:sx1]
        geo = known_geodesic(occ_known, *sc.center(gc0[0], gc0[1]), sc.cell, goal, clear)
        # 局部占据率（场景先验）
        pr = int(math.ceil(PRIOR_LOCAL_R_M / sc.cell))
        sub = sc.occ[max(0, cc[1] - pr):cc[1] + pr + 1, max(0, cc[0] - pr):cc[0] + pr + 1]
        rho = float(sub.mean()) if sub.size else 0.0
        fields = []
        for _ in range(self.k):
            if self.k == 1:
                occ = seen_occ[iy0:iy0 + win[3], ix0:ix0 + win[2]]
                model = self.model
            else:
                occ = sample_occ_3dgs(rng, sc, seen_free, seen_occ, win, rho)
                model = sample_model(rng, self.model, self.latency_range)
            dist = occ_dist_table(occ, sc.cell, FACE_OFFSET_3DGS_M)
            fields.append((HypField(dist, wx0, wy0, sc.cell, geo), model))
        return fields, int(seen_occ.sum())

    def label(self, x, y, yaw_deg, goal, goal_r, stop_mode=True, vx=0.0, vy=0.0, yaw_rate=0.0,
              prev_action=None, seed=0):
        """一步软标签。返回 dict(p, q_bar, q, k_feasible, fail_closed, visible_frac, n_known)。

        goal/goal_r/stop_mode 同 teacher_kino（停靠模式：悬停停住的位置进圈才算到）。
        """
        yaw = math.radians(yaw_deg)
        if stop_mode and math.hypot(x - goal[0], y - goal[1]) <= goal_r:
            p = np.zeros(4)
            p[SAFE_NEUTRAL] = 1.0
            q = np.full(4, math.inf)
            q[SAFE_NEUTRAL] = 0.0
            return {"p": p, "q_bar": q, "q": q[None, :], "k_feasible": self.k, "fail_closed": False,
                    "visible_frac": 1.0, "n_known": 0, "at_goal": True}
        rng = np.random.default_rng(seed)
        seen_free, seen_occ = ray_cast(self.scene, x, y, yaw, self.fov_deg, self.range_m)
        vis = visible_frac(self.scene, seen_free, seen_occ, x, y, yaw)
        if self.prior.get("kind") == "occ":
            fields, n_known = self._fields_3dgs(rng, x, y, yaw, goal, tk.CLEAR_PLAN, seen_free, seen_occ)
        else:
            fields, n_known = self._fields_gazebo(rng, x, y, yaw, goal, tk.CLEAR_PLAN, seen_free, seen_occ)
        start = (float(x), float(y), yaw, float(vx), float(vy), float(yaw_rate))
        q = np.full((self.k, 4), math.inf)
        for i, (field, model) in enumerate(fields):
            clear_first = tk.CLEAR_PLAN if field.clearance(x, y) >= tk.CLEAR_PLAN else tk.ENVELOPE_R + 0.01
            costs = tk.search_costs(start, field, goal, goal_r, stop_mode, clear_first, prev_action, model)
            for a in ACTIONS4:
                q[i, a] = costs[a][0]
        finite_rows = np.isfinite(q).any(axis=1)
        k_feasible = int(finite_rows.sum())
        if k_feasible == 0:
            p = np.zeros(4)
            p[SAFE_NEUTRAL] = 1.0
            return {"p": p, "q_bar": np.full(4, math.inf), "q": q, "k_feasible": 0, "fail_closed": True,
                    "visible_frac": vis, "n_known": n_known, "at_goal": False}
        c_coll = C_COLL_STEPS * self.model.step_s
        q_fin = np.where(np.isfinite(q), q, c_coll)
        q_bar = aggregate(q_fin, self.mode, self.cvar_q)
        return {"p": softmax_costs(q_bar, self.alpha), "q_bar": q_bar, "q": q, "k_feasible": k_feasible,
                "fail_closed": False, "visible_frac": vis, "n_known": n_known, "at_goal": False}


# ---------------------------------------------------------------- 整集标注（3DGS）
def episode_states(ep_dir):
    """从 actions.jsonl 还原每步教师输入：位姿、上一步位移估的速度/偏航率、上一步执行动作、硬标签。"""
    man = json.load(open(os.path.join(ep_dir, "episode_manifest.json"), encoding="utf-8"))
    rows = [json.loads(l) for l in open(os.path.join(ep_dir, "actions.jsonl"), encoding="utf-8") if l.strip()]
    rows.sort(key=lambda r: int(r["step"]))
    hold = float(man["hold_s"])
    out = []
    prev = None
    for r in rows:
        x, y = float(r["pose_before"]["x"]), float(r["pose_before"]["y"])
        yaw = float(r["yaw_before"])
        vx = vy = yaw_rate = 0.0
        prev_action = None
        if prev is not None:
            vx = (x - prev["x"]) / hold
            vy = (y - prev["y"]) / hold
            yaw_rate = tk._wrap(math.radians(yaw - prev["yaw"])) / hold
            prev_action = int(prev["action"])
        out.append({"step": int(r["step"]), "x": x, "y": y, "yaw_deg": yaw, "vx": vx, "vy": vy,
                    "yaw_rate": yaw_rate, "prev_action": prev_action,
                    "hard": int(r["action_expert"]), "collision": bool(r.get("collision", False))})
        prev = {"x": x, "y": y, "yaw": yaw, "action": r["action_executed"]}
    return man, out


def label_episode(args_tuple):
    """多进程入口：标一整集，返回 (episode_id, [每步 dict])。"""
    ep_dir, occ_npz, k, seed, alpha_steps, mode = args_tuple
    scene = Scene2D.from_occ_npz(occ_npz)
    man, states = episode_states(ep_dir)
    teacher = BeliefTeacher(scene, {"kind": "occ"}, model=tk.Model(step_s=float(man["hold_s"]), latency_s=0.0),
                            k=k, alpha_steps=alpha_steps, mode=mode)
    wp = man["waypoint"]
    goal, goal_r = (float(wp["x"]), float(wp["y"])), float(man.get("stop_radius") or 0.5)
    eid = int(man["episode_id"])
    out = []
    for s in states:
        step_seed = (seed * 1_000_003 + eid * 1009 + s["step"]) % (2 ** 32)
        t0 = time.perf_counter()
        r = teacher.label(s["x"], s["y"], s["yaw_deg"], goal, goal_r, True, s["vx"], s["vy"], s["yaw_rate"],
                          s["prev_action"], seed=step_seed)
        ent = float(-(r["p"] * np.log(np.maximum(r["p"], 1e-12))).sum())
        out.append({"step": s["step"], "p": r["p"].tolist(),
                    "q_bar": [float(v) if math.isfinite(v) else None for v in r["q_bar"]],
                    "k_feasible": r["k_feasible"], "fail_closed": bool(r["fail_closed"]),
                    "visible_frac": r["visible_frac"], "n_known": r["n_known"], "hard": s["hard"],
                    "argmax": int(np.argmax(r["p"])), "entropy": ent, "collision": s["collision"],
                    "at_goal": bool(r.get("at_goal")), "ms": 1000 * (time.perf_counter() - t0)})
    return eid, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("episodes", nargs="+", help="collect_3dgs episode 目录（含 episode_manifest.json / actions.jsonl）")
    p.add_argument("--occ-npz", required=True, help="场景 2D 占据缓存（build_occ_cache.py）")
    p.add_argument("--out", required=True, help="输出 npz（键 {episode_id}/{step}/logits = log p）")
    p.add_argument("--stats", required=True, help="逐步统计 JSON（p、q_bar、k_feasible、fail_closed、visible_frac、硬标签）")
    p.add_argument("--k", type=int, default=K_DEFAULT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha-steps", type=float, default=ALPHA_STEPS)
    p.add_argument("--aggregate", choices=("qmdp", "cvar"), default="qmdp")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    jobs = [(d, args.occ_npz, args.k, args.seed, args.alpha_steps, args.aggregate) for d in args.episodes]
    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp
        with mp.Pool(min(args.workers, len(jobs))) as pool:
            results = pool.map(label_episode, jobs)
    else:
        results = [label_episode(j) for j in jobs]
    arrays, stats = {}, {}
    for eid, rows in results:
        stats[str(eid)] = rows
        for r in rows:
            arrays[f"{eid}/{r['step']}/logits"] = np.log(np.maximum(np.asarray(r["p"], dtype=np.float64), 1e-12)).astype(np.float32)
    arrays["teacher_version"] = np.array(TEACHER_VERSION)
    arrays["k"] = np.array(args.k)
    arrays["alpha_steps"] = np.array(args.alpha_steps)
    arrays["aggregate"] = np.array(args.aggregate)
    arrays["seed"] = np.array(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    all_rows = [r for rows in stats.values() for r in rows]
    n = len(all_rows)
    summary = {
        "teacher_version": TEACHER_VERSION, "k": args.k, "alpha_steps": args.alpha_steps,
        "aggregate": args.aggregate, "seed": args.seed, "episodes": len(stats), "steps": n,
        "argmax_eq_hard": sum(r["argmax"] == r["hard"] for r in all_rows) / max(1, n),
        "mean_entropy_nats": float(np.mean([r["entropy"] for r in all_rows])) if n else None,
        "fail_closed_frac": sum(r["fail_closed"] for r in all_rows) / max(1, n),
        "at_goal_frac": sum(r["at_goal"] for r in all_rows) / max(1, n),
        "visible_frac_quantiles": {q: float(np.quantile([r["visible_frac"] for r in all_rows], q / 100))
                                   for q in (10, 50, 90)} if n else None,
        "k_feasible_mean": float(np.mean([r["k_feasible"] for r in all_rows])) if n else None,
        "p_mean_by_hard": {str(h): np.mean([r["p"] for r in all_rows if r["hard"] == h], axis=0).tolist()
                           for h in range(4) if any(r["hard"] == h for r in all_rows)},
        "ms_per_step": float(np.mean([r["ms"] for r in all_rows])) if n else None,
        "wall_s": time.time() - t0,
    }
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "episodes": stats}, f, ensure_ascii=False)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
