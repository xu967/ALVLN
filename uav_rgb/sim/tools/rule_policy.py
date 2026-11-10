#!/usr/bin/env python3
"""规则层上限（计划 D3）：特权「完美列距离」+ 固定规则 v0 出动作；v1（计划 D6）在其上加短期障碍记忆。纯 stdlib，不依赖 Gazebo。

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

# 规则 v1 参数（主会话 2026-11-10 20:40 设计，起跑前冻结）：v0 在 test 上 26/40（rep0、rep1）判负后加短期障碍记忆，参数不在 test seed 上调
RULE_V1 = {
    "r_safe": 0.35,        # 候选航向通道半宽：同 v0
    "d_look": 1.2,         # 候选航向通道长度：同 v0
    "yaw_step_deg": 26.0,  # 一步 YAW 的转角：308 个转向步实测 26.12° ± 1.09°（heading-guard 数据，两个方向合并）
    "mem_steps": 8,        # 记忆保留步数：绕过一根 0.4 m 的柱约 6–7 步，再留 1–2 步余量
    "n_cap": 3,            # |净转向步数| 的上限（主会话设计值）：3 步约 78°，到顶不再朝同一方向转，改转回
    "tie_frac": 0.10,      # 左右两半 Σmin(d, sum_clip) 相差不到较大者的 10% 算平局：同 v0
    "sum_clip": 4.0,       # 同 v0
}
V1_ENTRIES = ("V1-back", "V1-fwd", "V1-more", "V1-side", "V1-last", "V1-left", "V1-cap")   # rule_v1_entry 的全部条目


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


# ---------------------------------------------------------------- 规则 v1：短期障碍记忆 + 候选航向选择
def obstacle_points(d, hfov_deg=60.0, col_deg=COL_DEG, max_range=4.0, cam_forward=0.045):
    """列距离 → 机体系障碍点 [(x, y)]（x 前、y 左，原点 = 机体中心）。

    每个 d_j < max_range 的列出一个点：取该列最近距离、列中心方位 φ_j = (k/2 − 0.5 − j)·col_deg（正 = 左），
    点 = (d_j·cos φ_j + cam_forward, d_j·sin φ_j)。点的起算位置是相机，换到机体中心要加 cam_forward。
    列数必须等于 hfov_deg / col_deg（60° → 20、81° → 27），对不上说明视场与列距离不是一对，抛 ValueError。
    """
    k = len(d)
    if k != round(hfov_deg / col_deg):
        raise ValueError(f"列数 {k} 与视场 {hfov_deg}° / 列宽 {col_deg}° 对不上（应为 {round(hfov_deg / col_deg)}）")
    pts = []
    for j, dj in enumerate(d):
        if dj < max_range:
            phi = math.radians((k / 2 - 0.5 - j) * col_deg)
            pts.append((dj * math.cos(phi) + cam_forward, dj * math.sin(phi)))
    return pts


def points_to_world(points, x, y, yaw_deg):
    """机体系点 → 世界系点：机体位姿 (x, y, yaw_deg)（yaw 0 = +x，正 = 左转）。"""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [(x + px * c - py * s, y + px * s + py * c) for px, py in points]


def points_to_body(points_world, x, y, yaw_deg):
    """世界系点 → 机体系点（points_to_world 的逆）。"""
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return [((X - x) * c + (Y - y) * s, -(X - x) * s + (Y - y) * c) for X, Y in points_world]


def update_memory(memory, step, new_world_points, mem_steps):
    """记忆更新（纯函数）：memory = [(X, Y, 产生步号)]（世界系）。丢弃年龄 step − 产生步号 > mem_steps 的点，并入本步新点（年龄 0）。"""
    kept = [m for m in memory if step - m[2] <= mem_steps]
    return kept + [(X, Y, step) for X, Y in new_world_points]


def corridor_clear(points_body, theta_deg, params=RULE_V1):
    """朝相对当前航向 θ（正 = 左）的通道是否畅通：没有点满足 0 < u < d_look 且 |v| < r_safe，u = x·cosθ + y·sinθ，v = −x·sinθ + y·cosθ。"""
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    for x, y in points_body:
        if 0.0 < x * c + y * s < params["d_look"] and abs(-x * s + y * c) < params["r_safe"]:
            return False
    return True


def rule_v1_entry(points_body, d, n, last_turn, params=RULE_V1):
    """规则 v1：(动作, 触发的条目)。points_body = 记忆里全部点（含本步新点）换到当前机体系；d = 本步各列距离（只在 V1-side 里用）；
    n = 已执行 YAW_LEFT 数减 YAW_RIGHT 数；last_turn 上次转向方向或 None。

    决策顺序（α = yaw_step_deg）：
      1. n ≠ 0 且「转回一步」的通道畅通（n > 0 朝右 θ = −α，n < 0 朝左 θ = +α）→ 转回，V1-back；
      2. 当前航向通道畅通 → FORWARD_SLOW，V1-fwd；
      3. 否则转开：n ≠ 0 且 |n| < n_cap → 沿 n 的符号再转一步 V1-more；n = 0 → 左右两半 Σmin(d, sum_clip) 取较大一侧 V1-side，
         相差不到 tie_frac 沿用 last_turn（V1-last），没有则向左（V1-left）；|n| ≥ n_cap → 转回 V1-cap。
    永不输出 SAFE_NEUTRAL；FORWARD_SLOW 只在 clear(0) 时输出，所以它同时是护盾。
    shortcut: 奇数列时正中间一列不计入左右两半（只影响 81° 视场）；记忆里的点只有水平截面，没有「已转过的角度」之外的航向信息。
    """
    alpha = params["yaw_step_deg"]
    if n != 0:
        back_turn, back_theta = (YAW_RIGHT, -alpha) if n > 0 else (YAW_LEFT, alpha)
        if corridor_clear(points_body, back_theta, params):
            return back_turn, "V1-back"
    if corridor_clear(points_body, 0.0, params):
        return FORWARD_SLOW, "V1-fwd"
    if n != 0:
        if abs(n) < params["n_cap"]:
            return (YAW_LEFT if n > 0 else YAW_RIGHT), "V1-more"
        return (YAW_RIGHT if n > 0 else YAW_LEFT), "V1-cap"
    k = len(d)
    h = k // 2
    left, right = (sum(min(v, params["sum_clip"]) for v in half) for half in (d[:h], d[k - h:]))
    if abs(left - right) >= params["tie_frac"] * max(left, right):
        return (YAW_LEFT if left > right else YAW_RIGHT), "V1-side"
    if last_turn in (YAW_LEFT, YAW_RIGHT):
        return last_turn, "V1-last"
    return YAW_LEFT, "V1-left"


def rule_v1_step(memory, step, x, y, yaw_deg, d, n, last_turn, hfov_deg=60.0, params=RULE_V1):
    """v1 的一步（collect.py 与离线模拟共用，纯函数）：本步列距离 → 世界系点并入记忆 → 全部记忆点换回当前机体系 → 规则。

    返回 (动作, 条目, 新记忆, info)；info = {"mem_n": 记忆点数, "clear": [clear(+α), clear(0), clear(−α)]}（左转一步 / 保持 / 右转一步，供事后分析）。
    x, y, yaw_deg 是真值位姿（perfect odometry）：记忆在世界系，不依赖 n。
    """
    new_world = points_to_world(obstacle_points(d, hfov_deg), x, y, yaw_deg)
    memory = update_memory(memory, step, new_world, params["mem_steps"])
    body = points_to_body([(X, Y) for X, Y, _t in memory], x, y, yaw_deg)
    action, entry = rule_v1_entry(body, d, n, last_turn, params)
    a = params["yaw_step_deg"]
    clear = [corridor_clear(body, th, params) for th in (a, 0.0, -a)]
    return action, entry, memory, {"mem_n": len(body), "clear": clear}
