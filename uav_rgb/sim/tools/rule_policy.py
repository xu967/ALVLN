#!/usr/bin/env python3
"""规则层上限（计划 D3）：特权「完美列距离」+ 固定规则 v0 出动作。纯 stdlib，不依赖 Gazebo。

可判定性分解：图像决定得了的量（各列到最近障碍的距离）由 column_distances 用仿真真值解析算出，
图像决定不了的量（航向）由机体自己数（net_yaw），两者之间只有 rule_v0 这一层固定规则。
真值只进这条 oracle 策略，不进学生输入；用它的地方名字里必须带 oracle（collect --policy rule-oracle）。

坐标与角度约定同 teacher.py：世界 x 前、y 左；yaw 度，0 = +x，正 = 左转（YAW_LEFT → yaw 增大）；
方位角正 = 左，列从左到右排。柱高 3 m 盖住巡航高度，按水平截面二维求交；|y| = 3 m 的走廊边界只是判据，不是实体。
"""
import math

from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT

COL_DEG = 3.0            # 列宽：60° 视场 20 列、81° 视场 27 列（计划 §8 规则 v0 的 θ_j = (j − 9.5) × 3°）
N_RAYS = 5               # 每列在 3° 扇区内取的射线数（含两条边缘）：5 条相邻间隔 0.75°，2 m 外柱面上约 2.6 cm

# 规则 v0 参数：起跑前定死，只许在 train 分区的几何上调（计划 §8）
RULE_V0 = {
    "r_safe": 0.35,      # 正前方通道半宽：机体半径 0.102（e1_mini.json collision_envelope）+ 余量 0.25
    "d_look": 1.2,       # 通道长度：两步前进 0.5 m + 刹停 0.13 m + 余量
    "n_max": 2,          # |净转向步数| 达到它就朝回转：与 R1 实验的 --heading-guard 2 同值（越界 23 → 0），2 步约 54°
    "side_near": 0.5,    # 回转一侧最外 2 列近于它就不转：r_safe 0.35 + 一步偏航沿新航向漂 0.11（teacher.TIE_YAW_DRIFT_M），取整
    "tie_frac": 0.10,    # 左右两半 Σmin(d, sum_clip) 相差不到较大者的 10% 算平局，沿用上次转向，免得左右摆
    "sum_clip": 4.0,     # = column_distances 的 max_range：远处全空的列不再加分
}


def _hit(ox, oy, ux, uy, p):
    """射线 (ox, oy) + t·(ux, uy)，t ≥ 0 与一根柱的水平截面的最近交点距离；不相交 None，起点在柱内 0。"""
    r = p.get("radius")
    if r is not None:                                             # 圆柱（forest_e1 树干）
        dx, dy = ox - p["x"], oy - p["y"]
        b = ux * dx + uy * dy
        disc = b * b - (dx * dx + dy * dy - float(r) ** 2)
        if disc < 0.0 or -b + math.sqrt(disc) < 0.0:
            return None
        return max(-b - math.sqrt(disc), 0.0)
    hx, hy = (v / 2 for v in p.get("size", (0.4, 0.4))[:2])      # 方柱（轴对齐）：slab 法
    lo, hi = -math.inf, math.inf
    for o, u, c, h in ((ox, ux, p["x"], hx), (oy, uy, p["y"], hy)):
        if abs(u) < 1e-12:
            if abs(o - c) > h:
                return None
            continue
        a, b = (c - h - o) / u, (c + h - o) / u
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return None if lo > hi or hi < 0.0 else max(lo, 0.0)


def column_distances(x, y, yaw_deg, pillars, hfov_deg=60.0, col_deg=COL_DEG, max_range=4.0,
                     cam_forward=0.045, n_rays=N_RAYS):
    """从左到右各列到最近柱子的水平距离（m，沿射线量，超过 max_range 记 max_range）。

    (x, y, yaw_deg) 是机体中心位姿；相机在机体前 cam_forward 处（e1_mini.json camera_mount）、朝向与机头一致。
    列数 = round(hfov / col)；第 j 列中心方位 = (n/2 − 0.5 − j) × col_deg（正 = 左），扇区内均匀取 n_rays 条射线含两条边缘。
    """
    if n_rays < 5:
        raise ValueError("每列至少 5 条射线（含两条边缘）")
    n_cols = round(hfov_deg / col_deg)
    yaw = math.radians(yaw_deg)
    ox, oy = x + cam_forward * math.cos(yaw), y + cam_forward * math.sin(yaw)
    out = []
    for j in range(n_cols):
        phi = (n_cols / 2 - 0.5 - j) * col_deg
        best = max_range
        for k in range(n_rays):
            th = math.radians(yaw_deg + phi + (k / (n_rays - 1) - 0.5) * col_deg)   # 世界系射线角 = 机头角 + 方位
            ux, uy = math.cos(th), math.sin(th)
            for p in pillars:
                t = _hit(ox, oy, ux, uy, p)
                if t is not None and t < best:
                    best = t
        out.append(best)
    return out


def rule_v0_entry(d, n, last_turn, params=RULE_V0, col_deg=COL_DEG):
    """规则 v0：(动作, 触发的规则条目)。d 左到右列距离；n = 已执行 YAW_LEFT 数减 YAW_RIGHT 数；last_turn 上次转向方向或 None。

    条目：R1 朝回转 / R1-fwd、R1-safe 回转侧贴近障碍时的例外（不 blocked 前进、blocked 停）/
    R2-sum 向较开阔一侧 / R2-last 平局沿用上次 / R2-n 平局朝让 n 回零 / R2-left 平局且 n=0 固定左 / R3 前进。
    shortcut: 奇数列时正中间一列不计入左右两半（只影响 81° 视场）；无短期记忆，视锥外的柱子看不见，D3 失败后才做 D6。
    """
    k = len(d)
    blocked = False
    for j, dj in enumerate(d):                                    # 正前方半宽 r_safe、长 d_look 的通道里有障碍
        phi = math.radians((k / 2 - 0.5 - j) * col_deg)
        if dj * abs(math.sin(phi)) < params["r_safe"] and dj * math.cos(phi) < params["d_look"]:
            blocked = True
            break
    if abs(n) >= params["n_max"]:
        back = YAW_RIGHT if n > 0 else YAW_LEFT
        if min(d[-2:] if back == YAW_RIGHT else d[:2]) >= params["side_near"]:
            return back, "R1"
        return (SAFE_NEUTRAL, "R1-safe") if blocked else (FORWARD_SLOW, "R1-fwd")
    if not blocked:
        return FORWARD_SLOW, "R3"
    h = k // 2
    left, right = (sum(min(v, params["sum_clip"]) for v in half) for half in (d[:h], d[k - h:]))
    if abs(left - right) >= params["tie_frac"] * max(left, right):
        return (YAW_LEFT if left > right else YAW_RIGHT), "R2-sum"
    if last_turn in (YAW_LEFT, YAW_RIGHT):
        return last_turn, "R2-last"
    if n != 0:
        return (YAW_RIGHT if n > 0 else YAW_LEFT), "R2-n"
    return YAW_LEFT, "R2-left"


def rule_v0(d, n, last_turn, params=RULE_V0, col_deg=COL_DEG):
    """规则 v0 的动作（四个枚举之一）；要知道触发了哪一条用 rule_v0_entry。"""
    return rule_v0_entry(d, n, last_turn, params, col_deg)[0]
