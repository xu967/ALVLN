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


# ---------------------------------------------------------------- 规则 v2（任务卡 K3）：两侧候选 + 分级通道 + 禁忌航向
# 上面 v0 / v1 的常量与函数一行不改，下面全是新增（对已有行的 diff 为 0）。
# 为什么要 v2：K1 在 3DGS 室内量到 v1 的失败以 V1-more ↔ V1-cap 两步循环收尾（docs/reviews/2026-11-10-phase12/k1.md §6、§16）。
#   v1 每步只在三个航向里选（朝回一步 / 保持 / 沿 n 的符号再转开一步），V1-more 不看要转去的航向通不通，V1-cap 在 |n| 到顶时无条件转回
#   刚判过是堵的航向；视场 60° 而一步转约 26–32°，转一步后的航向有一部分在视场外，没看过的方向没有障碍点，按畅通算；通道只有一种尺寸。
# v2 的三处改动：两侧候选（c 段：往两侧找第一个通道畅通的航向）、分级通道（levels：长通道不通时退到更短更窄的）、
#   禁忌航向（在同一位置面对过并判为堵的世界航向，有效期内不再作候选）。
# 记号：α = yaw_step_deg；n = 机头偏离导引方向的整数步数（同 v1，正 = 偏左）；e = 同一个量的浮点值（3DGS 里是 −err/α，不给就取 n）；
#   m = 相对当前航向再转的步数（正 = 左转），第 m 个候选航向相对机头 m·α，它的 n 是 n + m。
# 状态由调用方按集清零、逐步传入传出：memory（同 v1 的世界系障碍点）与 tabu（禁忌表，每条 (世界航向 yaw_deg, x, y, 步号, lb)）。
RULE_V2 = {
    "yaw_step_deg": 26.0,    # 同 RULE_V1（Gazebo 默认）；3DGS 由实验入口覆盖（hold 1.0 实测中位 31.7°）
    "n_cap": 3,              # 同 RULE_V1：候选航向的 |n + m| 上限；已经偏出更多时放宽到 |n|
    "tie_frac": 0.10,        # 同 RULE_V1：左右两半 Σmin(d, sum_clip) 相差不到较大者的 10% 算平局
    "sum_clip": 4.0,         # 同 RULE_V1
    "mem_steps": 24,         # 记忆保留步数（任务卡定值；v1 是 8）
    "levels": ((0.35, 1.2, False),),   # 每级 (r_safe, d_look, 只许静止起步)；只有一级，数值同 v1 的 r_safe / d_look
    "order": "lexi",         # lexi 逐级 | near 先前进
    "tabu_radius": 0.30,     # 禁忌条目的有效半径（m）
    "tabu_steps": 24,        # 禁忌条目的有效步龄（步）
    "m_max": 6,              # 两侧候选最多再转几步（另受 |n + m| <= max(n_cap, |n|) 限制）
}
V2_ORDERS = ("lexi", "near")
V2_MAX_LEVELS = 3          # 任务卡的变体表最多 3 级；V2_ENTRIES 只列到这里，更多级的条目名按同一规则生成
V2_ENTRIES = tuple(f"V2-{kind}{lv}" for lv in range(V2_MAX_LEVELS) for kind in ("back", "fwd", "seek")) + ("V2-stuck",)   # rule_v2_entry 的全部条目


def check_v2_params(params):
    """校验 v2 参数并返回规范化的 levels（元组的元组：(r_safe, d_look, 只许静止起步)）。不合格抛 ValueError。

    levels 非空，每级三项，r_safe 与 d_look 是有限正数；逐级不放宽（后一级的 r_safe 与 d_look 都不大于前一级）：
    于是某一级畅通则更靠后的级也畅通，某一级被堵则更靠前的级也被堵，禁忌条目的 lb（被堵的最靠后的级）才有意义。
    order 只认 lexi / near；n_cap、m_max 是正整数；tabu_radius、tabu_steps 是非负有限数。JSON 里读来的列表与元组都接受。
    """
    raw = params["levels"]
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"levels 至少要有一级，得到 {raw!r}")
    levels = []
    for lv in raw:
        if not isinstance(lv, (list, tuple)) or len(lv) != 3:
            raise ValueError(f"levels 的每一级是 (r_safe, d_look, 只许静止起步)，得到 {lv!r}")
        r, d_look, still = float(lv[0]), float(lv[1]), bool(lv[2])
        if not (math.isfinite(r) and math.isfinite(d_look) and r > 0.0 and d_look > 0.0):
            raise ValueError(f"levels 的 r_safe 与 d_look 必须是有限正数，得到 {lv!r}")
        levels.append((r, d_look, still))
    for i in range(1, len(levels)):
        if levels[i][0] > levels[i - 1][0] or levels[i][1] > levels[i - 1][1]:
            raise ValueError(f"levels 必须逐级不放宽（r_safe 与 d_look 都不增大）：第 {i - 1} 级 {levels[i - 1][:2]}，第 {i} 级 {levels[i][:2]}")
    if params["order"] not in V2_ORDERS:
        raise ValueError(f"order 必须是 {V2_ORDERS}，得到 {params['order']!r}")
    for key in ("n_cap", "m_max"):
        v = params[key]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ValueError(f"{key} 必须是正整数，得到 {v!r}")
    for key in ("tabu_radius", "tabu_steps"):
        v = params[key]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
            raise ValueError(f"{key} 必须是非负有限数，得到 {v!r}")
    return tuple(levels)


def _wrap180(deg):
    return (deg + 180.0) % 360.0 - 180.0


def v2_tabu_hit(tabu, psi_deg, level, step, x, y, params):
    """世界航向 psi_deg 在第 level 级是否被禁：表里有一条同时满足 航向差 <= α/2、位置差 <= tabu_radius、步龄 <= tabu_steps、level <= lb。"""
    half = params["yaw_step_deg"] / 2.0
    for yaw_e, xe, ye, step_e, lb in tabu:
        if (level <= lb and step - step_e <= params["tabu_steps"] and abs(_wrap180(psi_deg - yaw_e)) <= half
                and math.hypot(x - xe, y - ye) <= params["tabu_radius"]):
            return True
    return False


def rule_v2_entry(points_body, d, n, last_turn, tabu, step, pose, prev_action=None, e=None, params=RULE_V2):
    """规则 v2：(动作, 条目, info)。points_body = 记忆里全部点（含本步新点）换到当前机体系；d = 本步各列距离（只用来给两侧候选排序）；
    n、last_turn 同 v1；tabu = 已经含本步条目的禁忌表；pose = (x, y, yaw_deg) 真值位姿（候选航向与禁忌条目在世界系比）；
    prev_action = 上一步执行的动作（None = 第 0 步，按静止算）；e = n 的浮点值（None 取 n）。

    ok(m, ℓ) = 记忆点在第 ℓ 级通道（半宽 r_safe、长 d_look）里对航向 m·α 畅通，且该航向在第 ℓ 级没有被禁；m = 0 不查禁忌表（它是本步直接看到的）。
    三个基本动作：
      a(ℓ)  n ≠ 0 且 ok(朝回一步, ℓ) → 朝回转一步，V2-back<ℓ>；
      b(ℓ)  ok(0, ℓ) → FORWARD_SLOW，V2-fwd<ℓ>；
      c(ℓ)  m ≠ 0、|m| <= m_max、|n + m| <= max(n_cap, |n|) 且 ok(m, ℓ) 的候选里，按（|m| 小、|e + m| 小、v1 的左右两半求和偏向的一侧、
            last_turn 一侧、左）依次比较取最优，朝它转一步，V2-seek<ℓ>。
    order = lexi：ℓ = 0, 1, … 逐级做 a(ℓ)、b(ℓ)、c(ℓ)，第一个命中的返回（短通道只在长通道上两侧都找不到路时才用）；
    order = near：a(0)、b(0)、b(1)、b(2)、…、c(0)、a(1)、c(1)、a(2)、c(2)、…（当前航向在长通道不通、短通道通时先往前走，不去绕路）。
    「只许静止起步」的级只在 prev_action 不是 FORWARD_SLOW 时可用（转向后第一个前进步位移约 0.14 m，连续前进约 0.30 m，w1-11 §7.4），不可用的级整级跳过。
    全都没有命中：沿 last_turn 再转一步（没有则左转），V2-stuck。永不输出 SAFE_NEUTRAL；FORWARD_SLOW 只在某个可用级的 ok(0, ℓ) 为真时输出。

    info：level（命中的级，V2-stuck 为 None）、m（选中的 m，前进 0、转回 ∓1、V2-stuck 为 None）、e、usable（各级是否可用）、
    ok（各级 × m = −n_cap … n_cap 的 ok 位图，ok[ℓ][m + n_cap]）、clear（第 0 级几何通道 [左转一步, 保持, 右转一步]，同 v1 的次序，不含禁忌）。
    shortcut: 候选只是相对当前航向的整数步，记忆里的点没有高度；何时升级：K3 的失败集里出现「候选航向之间的缝里有路」时改成连续航向扫描。
    """
    levels = check_v2_params(params)
    alpha, n_cap, m_max = float(params["yaw_step_deg"]), params["n_cap"], params["m_max"]
    e = float(n) if e is None else float(e)
    x, y, yaw_deg = pose
    n_levels = len(levels)
    usable = [not (still and prev_action == FORWARD_SLOW) for _r, _d, still in levels]
    geo = {}

    def clear(m, lv):                                   # 几何：记忆点在第 lv 级通道里是否畅通
        if (m, lv) not in geo:
            geo[(m, lv)] = corridor_clear(points_body, m * alpha, {"r_safe": levels[lv][0], "d_look": levels[lv][1]})
        return geo[(m, lv)]

    def ok(m, lv):
        return clear(m, lv) and (m == 0 or not v2_tabu_hit(tabu, yaw_deg + m * alpha, lv, step, x, y, params))

    k = len(d)
    left, right = (sum(min(v, params["sum_clip"]) for v in half) for half in (d[:k // 2], d[k - k // 2:]))
    side = 0                                            # v1 的 V1-side：左右两半求和偏向的一侧（+1 左 / −1 右），平局为 0
    if left != right and abs(left - right) >= params["tie_frac"] * max(left, right):
        side = 1 if left > right else -1
    last = {YAW_LEFT: 1, YAW_RIGHT: -1}.get(last_turn, 0)

    def seek(lv):
        cap = max(n_cap, abs(n))
        cands = [m for m in range(-m_max, m_max + 1) if m != 0 and abs(n + m) <= cap and ok(m, lv)]
        if not cands:
            return None
        return min(cands, key=lambda m: (abs(m), abs(e + m), 0 if side == 0 or (m > 0) == (side > 0) else 1,
                                         0 if last == 0 or (m > 0) == (last > 0) else 1, 0 if m > 0 else 1))

    def info(level, m):
        return {"level": level, "m": m, "e": e, "usable": [int(u) for u in usable],
                "ok": [[int(ok(mm, lv)) for mm in range(-n_cap, n_cap + 1)] for lv in range(n_levels)],
                "clear": [int(clear(1, 0)), int(clear(0, 0)), int(clear(-1, 0))]}

    if params["order"] == "lexi":
        seq = [(kind, lv) for lv in range(n_levels) for kind in "abc"]
    else:
        seq = [("a", 0)] + [("b", lv) for lv in range(n_levels)] + [("c", 0)]
        for lv in range(1, n_levels):
            seq += [("a", lv), ("c", lv)]
    back_m = 0 if n == 0 else (-1 if n > 0 else 1)
    for kind, lv in seq:
        if not usable[lv]:
            continue
        if kind == "a":
            if n != 0 and ok(back_m, lv):
                return (YAW_RIGHT if back_m < 0 else YAW_LEFT), f"V2-back{lv}", info(lv, back_m)
        elif kind == "b":
            if ok(0, lv):
                return FORWARD_SLOW, f"V2-fwd{lv}", info(lv, 0)
        else:
            m = seek(lv)
            if m is not None:
                return (YAW_LEFT if m > 0 else YAW_RIGHT), f"V2-seek{lv}", info(lv, m)
    return (last_turn if last_turn in (YAW_LEFT, YAW_RIGHT) else YAW_LEFT), "V2-stuck", info(None, None)


def rule_v2_step(memory, tabu, step, x, y, yaw_deg, d, n, last_turn, e=None, prev_action=None, hfov_deg=60.0, params=RULE_V2):
    """v2 的一步（3DGS 的 RuleOraclePolicy 与离线模拟共用，纯函数）：本步列距离 → 世界系点并入记忆 → 记忆点换回当前机体系 →
    清超龄的禁忌条目，当前航向在第 0 级被堵就记一条 → 规则。

    禁忌条目 = (yaw_deg, x, y, step, lb)，lb = 当前航向被堵的最靠后的一级（按几何算，不看该级本步是否可用）；按世界航向记，不按 n 记：
    导引目标点会随位置跳，n 的编号不稳（w1-11 §6）。返回 (动作, 条目, 新记忆, 新禁忌表, info)；info 在 rule_v2_entry 的基础上加 mem_n、tabu_n。
    x, y, yaw_deg 是真值位姿（perfect odometry）：记忆与禁忌都在世界系，不依赖 n。
    """
    levels = check_v2_params(params)
    new_world = points_to_world(obstacle_points(d, hfov_deg), x, y, yaw_deg)
    memory = update_memory(memory, step, new_world, params["mem_steps"])
    body = points_to_body([(X, Y) for X, Y, _t in memory], x, y, yaw_deg)
    tabu = [t for t in tabu if step - t[3] <= params["tabu_steps"]]
    blocked = [not corridor_clear(body, 0.0, {"r_safe": r, "d_look": dl}) for r, dl, _s in levels]
    if blocked[0]:
        tabu.append((yaw_deg, x, y, step, max(i for i, b in enumerate(blocked) if b)))
    action, entry, info = rule_v2_entry(body, d, n, last_turn, tabu, step, (x, y, yaw_deg), prev_action, e, params)
    info["mem_n"], info["tabu_n"] = len(body), len(tabu)
    return action, entry, memory, tabu, info
