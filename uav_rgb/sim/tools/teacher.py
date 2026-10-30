#!/usr/bin/env python3
"""TeacherPlanner：专家避障策略（训练时特权，学生不可用）。

输出：动作枚举值（与 src/include/uavrgb/action.hpp 对齐）：
  0=SAFE_NEUTRAL, 1=FORWARD_SLOW, 2=YAW_LEFT, 3=YAW_RIGHT。

三个教师：
  - expert_action：深度反应式（三竖带），用于没有柱子真值的对照集；
  - plan_action：局部射线 carrot planner，safe 出生集用它保留真 SAFE 标签；
  - plan_action_global：A* 全局路径 + 纯追踪，普通采集集默认用它。
    T1 实测局部规划在门柱前因 34°/步的偏航过冲而摆振，24 集里 15 集擦柱；
    A* 先找一条净空路径再跟，才能稳定穿过。

纯 stdlib，不依赖 numpy；合成深度/柱阵即可单测。
"""
import heapq
import math

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")

# 教师实现版本：改动作选择（A* 参数/前向安全门/局部容差）必须 bump。
# collect 写进 manifest，audit 只对同版本数据做 label_recompute，避免用新算法
# 复算旧标签造成假 FAIL。
TEACHER_VERSION = "uavrgb.teacher.v4"

# 只取竖直方向的中间带（正前方走廊），比例范围 [0.4, 0.6)。
BAND_LO, BAND_HI = 0.4, 0.6

# carrot planner 把机器拉回走廊中线的增益；collect 与 audit 都走这个默认值，
# 改它必须两边一起变（否则 label_recompute 会把新数据全判不一致）。
# T1 实测 25 会在门柱前反复回中线、把已选的净空方向拉掉；普通集改用 A* 后
# 局部规划只剩 safe 出生集使用，保留 0 让 SAFE 位姿稳定。
Y_GAIN = 0.0


def _strip_min(depth, w, y0, y1, x0, x1):
    """行 [y0,y1) × 列 [x0,x1) 的最小有效深度和有效样本数。

    无效读数（<=0、NaN）跳过。返回 (inf, 0) 表示该区域一个有效样本都没有——
    调用方必须把它当成「不知道」，不能当成「空旷」。
    """
    best = math.inf
    n_valid = 0
    for y in range(y0, y1):
        base = y * w
        for x in range(x0, x1):
            d = depth[base + x]
            if d > 0.0:  # NaN/0/负数都是无效读数
                n_valid += 1
                if d < best:
                    best = d
    return best, n_valid


def expert_action(depth, w, h, turn_dist=1.5, emergency=0.8, prev_action=None,
                  switch_margin=0.75):
    """depth: 行主序深度（米），w*h 长度。返回 0..3 动作。

    prev_action/switch_margin 是**滞回**：两侧都可转时保持上次的转向方向，
    只有另一侧比当前侧空出 switch_margin 米以上才反向。没有滞回时，无人机一边转
    一边看深度，「哪侧更空」会在临界点来回翻转，实测出现左-右-左摆振（z 也从
    1.78 掉到 1.23），会污染训练标签。

    深度失效时 fail closed：中心带无有效样本 -> SAFE_NEUTRAL；某侧无有效样本
    -> 该侧视为堵。小尺寸深度（h<5 / w<3）和短 buffer 直接抛 ValueError。
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"invalid depth size {w}x{h}")
    if len(depth) < w * h:
        raise ValueError(f"depth buffer too short: {len(depth)} < {w * h}")
    # h<5 时中间带 [0.4h,0.6h) 可能为空或只采一行，专家会看错方向。
    if h < 5:
        raise ValueError(f"depth height too small: h={h} (need h>=5)")
    if w < 3:
        raise ValueError(f"depth width too small: w={w} (need w>=3)")

    y0, y1 = int(h * BAND_LO), int(h * BAND_HI)
    xl = w // 3
    xr = w * 2 // 3
    left, left_n = _strip_min(depth, w, y0, y1, 0, xl)
    center, center_n = _strip_min(depth, w, y0, y1, xl, xr)
    right, right_n = _strip_min(depth, w, y0, y1, xr, w)

    if center_n == 0:
        # 深度相机没出图/全 0/全 NaN：不能假设前方空旷。
        return SAFE_NEUTRAL
    if center >= turn_dist:
        return FORWARD_SLOW

    if left_n == 0:
        left = 0.0
    if right_n == 0:
        right = 0.0
    # 只有三面都进了紧急距离才算真被围死。旧版要求「某一整侧 >= min_clear 才敢转」，
    # 一个 1m 宽的地标在 1.88m 处就会同时占满三条带，于是永远 SAFE_NEUTRAL（实测卡死）。
    if center < emergency and left < emergency and right < emergency:
        return SAFE_NEUTRAL
    left_ok = left >= emergency
    right_ok = right >= emergency
    if left_ok and right_ok:
        # 滞回：默认保持上次方向，除非另一侧明显更空。
        if prev_action == YAW_LEFT:
            return YAW_RIGHT if right > left + switch_margin else YAW_LEFT
        if prev_action == YAW_RIGHT:
            return YAW_LEFT if left > right + switch_margin else YAW_RIGHT
        return YAW_LEFT if left >= right else YAW_RIGHT
    if left_ok:
        return YAW_LEFT
    if right_ok:
        return YAW_RIGHT
    return SAFE_NEUTRAL


def _body_hits_box(px, py, yaw_deg, pillar, half):
    """0.47 m 方形机身（含 1 cm 余量）与柱方框是否重叠（2D 分离轴）。

    机身随 yaw 旋转。按轴向包围盒会把 45° 机身的半宽当成 0.245 m，实际方角
    伸到 0.332 m，角部漏判；膨胀到外接圆又会在斜向误报、把脱困路径挡死。
    两条方框的分离轴只有 4 条：世界 x/y 与机身局部 x/y。
    """
    a = 0.235 + 0.01
    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    dx, dy = px - pillar["x"], py - pillar["y"]
    # 世界 x/y 轴：机身投影半宽 = a*(|cos|+|sin|)
    body_w = a * (abs(cy) + abs(sy))
    if abs(dx) > body_w + half or abs(dy) > body_w + half:
        return False
    # 机身局部 x/y 轴：柱框投影半宽 = half*(|cos|+|sin|)（轴同为正交旋转）
    for ux, uy in ((cy, sy), (-sy, cy)):
        if abs(dx * ux + dy * uy) > a + half * (abs(ux) + abs(uy)):
            return False
    return True


def ray_clear(x, y, th_deg, length, pillars, clearance, half=0.2, samples=33,
              skip_clearance_m=0.0, yaw_deg=0.0):
    """从 (x,y) 沿 th_deg 走 length 米，线段是否离所有柱子的**方框表面** ≥ clearance。

    不能用外接圆：0.4×0.4 方的外接圆半径 0.283 m，而车道缝隙只有 1.2 m，
    按外接圆算会把大部分窄缝判成堵死。

    采样精度：点到方框的距离是 1-Lipschitz，相邻采样点之间真实距离最多再降半个
    间隔。samples=33 时 1.6/32 = 0.05 m 间隔 → 最坏低算 25 mm（实测统计约 14 mm）。
    原先这里写"误差 < 1 mm"是错的。

    skip_clearance_m：前这么多米只查**真实碰撞盒**（0.47 m 机身方框 + 1 cm 余量），
    不查 clearance。用于 plan_action 的脱困回退：无人机已经漂进 0.36 m 安全余量时，
    从当前点出发的每条射线都会在 t=0 被判堵，旧逻辑会永远 SAFE 卡死（T1 实测
    8/8 场景卡在 x≈1.1–1.45）。真实碰撞仍然禁止，只是允许先离开余量。
    碰撞盒按 yaw 旋转的方框做分离轴判定（`_body_hits_box`）；外接圆或轴向包围盒
    都会在角部漏判/误判。
    """
    # shortcut: 脱困段只查机身方框，不查四个桨盘（外接半径 0.356 m）；
    # safe 集实测 0 碰撞，若出现擦桨再补逐盘圆判据。
    c, s = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
    for i in range(samples):
        t = length * i / (samples - 1)
        px, py = x + t * c, y + t * s
        for p in pillars:
            if t < skip_clearance_m:
                if _body_hits_box(px, py, yaw_deg, p, half):
                    return False
                continue
            dx = max(0.0, abs(px - p["x"]) - half)
            dy = max(0.0, abs(py - p["y"]) - half)
            if dx * dx + dy * dy < clearance * clearance:
                return False
    return True


def plan_action(x, y, yaw_deg, pillars, lookahead=1.6, clearance=0.36,
                y_gain=Y_GAIN, y_max=30.0, tol=15.0, turn_weight=0.25, grid=5):
    """GT carrot planner：挑一条净空视线，代价 = 偏离目标角 + 0.25×当前转角。

    深度反应式专家解不了这条走廊（60 步 × 8 场景实测只有 1 个穿过，其余卡在
    门柱前左右摆）：它没有目标方向，转开之后只能直飞，航向一偏就斜着飞出
    ±1.7 m 柱区。这里改用柱子真值（特权信息，只决定标签、不进学生输入）。

    目标角 base = clamp(-y_gain*y, ±y_max) 把机器拉回走廊中线。**不能**简单地
    「从 base 向外试、取第一条空视线」：净空锥很窄时它会从 +43° 一步跳到 -47°，
    机器跟着左右各转 100° 以上、净位移为零（实测 4/8 场景这样卡死）。所以对
    -85°..85° 每 5° 求一次代价取最小；turn_weight 只用来在同分时偏向小转角，
    不影响「回中线」的方向偏好。全都堵就退回 base。
    """
    base = max(-y_max, min(y_max, -y_gain * y))

    def best_ray(skip_m):
        aim, best = base, None
        for th in range(-85, 86, grid):
            if not ray_clear(x, y, th, lookahead, pillars, clearance,
                             skip_clearance_m=skip_m, yaw_deg=yaw_deg):
                continue
            cost = (abs(th - base)
                    + turn_weight * abs((yaw_deg - th + 180.0) % 360.0 - 180.0))
            if best is None or cost < best:
                aim, best = th, cost
        return aim, best

    aim, best = best_ray(0.0)
    if best is None:
        # 脱困：无人机可能已经漂进 0.36 m 安全余量（T1 实测 8/8 场景因此卡死）。
        # 允许前 0.2 m 只查真实碰撞盒，离开余量后再要求 full clearance。
        aim, best = best_ray(0.2)
    # 所有方向都堵时不能默认 base 可走：base 自己可能正对着柱子。
    if best is None and not ray_clear(x, y, base, lookahead, pillars, clearance):
        return SAFE_NEUTRAL
    err = (yaw_deg - aim + 180.0) % 360.0 - 180.0
    if abs(err) > tol:
        return YAW_LEFT if err < 0 else YAW_RIGHT
    # 局部回退同样不能盲目前进：0.4 m 内会进 0.36 m 余量就先转。
    if not ray_clear(x, y, yaw_deg, 0.4, pillars, clearance):
        return YAW_LEFT if err >= 0 else YAW_RIGHT
    return FORWARD_SLOW


def _point_blocked(x, y, pillars, clearance):
    """点到柱方框的距离 < clearance 即不可走（用于 A* 栅格）。"""
    for p in pillars:
        dx = max(0.0, abs(x - p["x"]) - 0.2)
        dy = max(0.0, abs(y - p["y"]) - 0.2)
        if dx * dx + dy * dy < clearance * clearance:
            return True
    return False


def astar_path(pillars, start, goal, clearance=0.5, step=0.1):
    """柱区 2D 栅格 A*，返回 [(x,y), ...] 或 None。

    局部射线规划在门柱前会左右摆（34°/步的偏航一次就跨过 10° 容差），
    T1 实测 24 集里 15 集擦柱。A* 从全局找一条净空路径，代价函数只是距离；
    障碍按柱方框 + clearance 膨胀，8 邻接且禁止斜穿角点。
    """
    # 边界由柱子派生，避免改 PILLAR_X_RANGE 后 A* 静默退回局部规划。
    x0 = min(-1.5, min(float(p["x"]) for p in pillars) - 1.5)
    x1 = max(7.0, max(float(p["x"]) for p in pillars) + 3.0)
    y1 = max(2.8, max(abs(float(p["y"])) for p in pillars) + 1.5)
    y0 = -y1
    nx = int(round((x1 - x0) / step)) + 1
    ny = int(round((y1 - y0) / step)) + 1

    def idx(px, py):
        return int(round((px - x0) / step)), int(round((py - y0) / step))

    sx, sy = idx(*start)
    gx, gy = idx(*goal)
    if not (0 <= sx < nx and 0 <= sy < ny and 0 <= gx < nx and 0 <= gy < ny):
        return None
    free = {}

    def is_free(i, j):
        k = (i, j)
        if k not in free:
            free[k] = not _point_blocked(x0 + i * step, y0 + j * step, pillars, clearance)
        return free[k]

    if not is_free(gx, gy):
        return None
    # 起点已经在 0.5 m 规划余量内时不能让 A* 直接失败退回 0.36 m 的局部规划：
    # 强制把起点当自由格，路径先离开余量；纯追踪的前向 0.36 m 检查兜底。
    free[(sx, sy)] = True
    open_heap = [(0.0, 0, sx, sy)]
    gscore = {(sx, sy): 0.0}
    parent = {}
    moves = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
             (1, 1, 1.41421356), (1, -1, 1.41421356),
             (-1, 1, 1.41421356), (-1, -1, 1.41421356))
    while open_heap:
        _f, _g, i, j = heapq.heappop(open_heap)
        if (i, j) == (gx, gy):
            path = []
            cur = (i, j)
            while cur in parent:
                path.append((x0 + cur[0] * step, y0 + cur[1] * step))
                cur = parent[cur]
            path.append((x0 + sx * step, y0 + sy * step))
            return path[::-1]
        gi = gscore[(i, j)]
        for di, dj, w in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny) or not is_free(ni, nj):
                continue
            if di and dj and not (is_free(i + di, j) and is_free(i, j + dj)):
                continue
            ng = gi + w * step
            if ng < gscore.get((ni, nj), math.inf):
                gscore[(ni, nj)] = ng
                parent[(ni, nj)] = (i, j)
                h = math.hypot((ni - gx) * step, (nj - gy) * step)
                heapq.heappush(open_heap, (ng + h, ng, ni, nj))
    return None


def _path_carrot(path, x, y, lookahead):
    """离当前点最近的路径点之后，沿路径累计 lookahead 米的那一点。"""
    best = min(range(len(path)),
               key=lambda k: (path[k][0] - x) ** 2 + (path[k][1] - y) ** 2)
    acc = 0.0
    px, py = path[best]
    for qx, qy in path[best + 1:]:
        acc += math.hypot(qx - px, qy - py)
        px, py = qx, qy
        if acc >= lookahead:
            break
    return px, py


def plan_action_global(x, y, yaw_deg, pillars, clearance=0.5, step=0.1,
                       lookahead=0.7, tol=25.0, waypoint=None):
    """A* 全局路径 + 纯追踪；失败时退回局部 plan_action。

    tol=25°：偏航一步约 34°，容差小于半步会左右过冲（T1 门柱振荡的根因）。
    路径目标比 T1 验收终点（最右柱 x+1.0）再远 1 m：目标正好落在验收线上时，
    飞控会在终点前转头/减速，max_x 常停在 5.97 而判失败；多飞 1 m 才真正越过。
    waypoint 非空时先到彩色目标柱旁，再去走廊终点（L1 route-choice）。
    """
    if not pillars:
        return plan_action(x, y, yaw_deg, [], tol=tol)
    goal = (max(float(p["x"]) for p in pillars) + 2.0, 0.0)
    if waypoint is None:
        path = astar_path(pillars, (x, y), goal, clearance=clearance, step=step)
    else:
        try:
            wp = (float(waypoint["x"]), float(waypoint["y"]))
        except (KeyError, TypeError, ValueError):
            return plan_action(x, y, yaw_deg, pillars, tol=tol)
        first = astar_path(pillars, (x, y), wp, clearance=clearance, step=step)
        rest = (astar_path(pillars, wp, goal, clearance=clearance, step=step)
                if first is not None else None)
        path = (first + rest[1:]) if first is not None and rest is not None else None
    if path is None:
        return plan_action(x, y, yaw_deg, pillars, tol=tol)
    tx, ty = _path_carrot(path, x, y, lookahead)
    err = (math.degrees(math.atan2(ty - y, tx - x)) - yaw_deg + 180.0) % 360.0 - 180.0
    if err > tol:
        return YAW_LEFT
    if err < -tol:
        return YAW_RIGHT
    # 航向已对准也不能盲目前进：机体有惯性，A* 路径的 0.5 m 余量会被一步
    # 0.27 m 的漂移吃掉。前向 0.4 m 若进 0.36 m 机身余量，先转再走。
    if not ray_clear(x, y, yaw_deg, 0.4, pillars, 0.36):
        return YAW_LEFT if err >= 0 else YAW_RIGHT
    return FORWARD_SLOW


def _make(w, h, left_d, center_d, right_d):
    """构造 w*h 深度图：左/中/右三条竖带各填一个深度值（所有行一致）。"""
    depth = []
    for _ in range(h):
        depth += [left_d] * (w // 3) + [center_d] * (w // 3) + [right_d] * (w - 2 * (w // 3))
    return depth


def _make_ground(w, h, band_d, ground_d):
    """构造 w*h 深度图：中间带 [0.4h,0.6h) 填 band_d，其余行填 ground_d（模拟地面）。"""
    y0, y1 = int(h * BAND_LO), int(h * BAND_HI)
    depth = []
    for y in range(h):
        d = band_d if y0 <= y < y1 else ground_d
        depth += [d] * w
    return depth


if __name__ == "__main__":
    w, h = 9, 10
    cases = [
        # (left, center, right, 期望动作)
        (5.0, 5.0, 5.0, FORWARD_SLOW, "全空 -> 前进"),
        (5.0, 0.3, 3.0, YAW_LEFT, "前方挡，左更空 -> 左转"),
        (3.0, 0.3, 5.0, YAW_RIGHT, "前方挡，右更空 -> 右转"),
        (0.3, 0.3, 0.3, SAFE_NEUTRAL, "三面堵死 -> 中立"),
        (0.3, 3.0, 0.3, FORWARD_SLOW, "仅前方通 -> 前进"),
    ]
    ok = True
    for left_d, center_d, right_d, want, name in cases:
        d = _make(w, h, left_d, center_d, right_d)
        got = expert_action(d, w, h)
        status = "ok" if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {status}  {name}: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 地面排除：底/顶行都是 0.15（地面级近距），但中间带全 5.0 -> 应前进，不误判挡。
    d = _make_ground(w, h, band_d=5.0, ground_d=0.15)
    got = expert_action(d, w, h)
    want = FORWARD_SLOW
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  底边地面排除（中间带空）: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 滞回：两侧都能转、右略优于左，但上次是左转 -> 保持左转（不来回摆）。
    d = _make(w, h, 3.0, 0.3, 3.5)
    got = expert_action(d, w, h, prev_action=YAW_LEFT)
    want = YAW_LEFT
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  滞回-小幅差异保持原向: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 滞回：右侧明显更空（超出 switch_margin）-> 必须转向右。
    d = _make(w, h, 2.1, 0.3, 5.0)
    got = expert_action(d, w, h, prev_action=YAW_LEFT)
    want = YAW_RIGHT
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  滞回-明显更空则换向: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 滞回不能压过安全：上次左转，但左侧已被堵 -> 必须转右。
    d = _make(w, h, 1.0, 0.3, 5.0)
    got = expert_action(d, w, h, prev_action=YAW_LEFT)
    want = YAW_RIGHT
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  滞回不覆盖安全: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 回归：一个 1m 地标在 1.88m 处会填满三条带。旧版在此 SAFE_NEUTRAL 卡死（实测 7/10 步）。
    d = _make(w, h, 1.88, 1.88, 1.88)
    got = expert_action(d, w, h)
    want = FORWARD_SLOW
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  三带都 1.88m 不卡死 -> 前进: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 回归：再近一点（1.2m，已进转向距离但没到紧急距离）必须转向而不是停住。
    d = _make(w, h, 1.2, 1.2, 1.2)
    got = expert_action(d, w, h)
    want = YAW_LEFT   # 三侧相同 -> 确定性地先左转
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  三带都 1.2m -> 转向而非停住: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # GT carrot planner：正前方有柱子时必须选出一条让开的视线，而不是硬顶。
    gate = [{"x": 1.5, "y": 0.0}]
    ok = ok and not ray_clear(0.0, 0.0, 0.0, 1.6, gate, 0.36)
    ok = ok and ray_clear(0.0, 0.0, 0.0, 1.6, [], 0.36)
    got = plan_action(0.0, 0.0, 0.0, gate)
    ok = ok and got in (YAW_LEFT, YAW_RIGHT)   # 正中门柱左右对称，任一侧都对
    print(f"  {'ok' if got in (YAW_LEFT, YAW_RIGHT) else 'FAIL'}  门柱正前方 -> 让开: got {ACTIONS[got]}")
    plan_cases = [
        (dict(x=0.0, y=0.0, yaw_deg=0.0, pillars=[]), FORWARD_SLOW, "无柱子 -> 前进"),
        (dict(x=0.0, y=0.0, yaw_deg=40.0, pillars=[]), YAW_RIGHT, "航向偏左 40° -> 右转"),
        # 回中线依赖 y_gain；显式给 25，避免调全局 Y_GAIN 时这条用例失真。
        (dict(x=0.0, y=-1.2, yaw_deg=0.0, pillars=[], y_gain=25.0),
         YAW_LEFT, "偏在右侧 -> 左转回中线"),
    ]
    for kw, want, name in plan_cases:
        got = plan_action(**kw)
        ok = ok and got == want
        print(f"  {'ok' if got == want else 'FAIL'}  {name}: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 所有射线都堵、且 base 自身也被堵：必须 SAFE_NEUTRAL，不能默认前进撞柱。
    orig_ray_clear = ray_clear
    ray_clear = lambda *a, **k: False
    try:
        got = plan_action(0.0, 0.0, 0.0, [])
    finally:
        ray_clear = orig_ray_clear
    want = SAFE_NEUTRAL
    ok = ok and got == want
    print(f"  {'ok' if got == want else 'FAIL'}  全堵且 base 堵 -> 中立: "
          f"got {ACTIONS[got]}, want {ACTIONS[want]}")

    # T1 回归：无人机漂进 0.36 m 安全余量（距柱面 0.344 m，但 AABB 未碰撞）时，
    # 旧逻辑在 t=0 就把所有射线判堵，永远 SAFE；脱困回退必须给出转向/前进。
    gate = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    got = plan_action(1.1, 0.48, 0.0, gate)
    ok = ok and got != SAFE_NEUTRAL
    print(f"  {'ok' if got != SAFE_NEUTRAL else 'FAIL'}  已进安全余量仍能脱困: "
          f"got {ACTIONS[got]}, want != SAFE_NEUTRAL")

    # 脱困碰撞盒：轴向方框的角部（旧圆漏判）必须判堵；45° 机身方角伸到 0.346 m，
    # 正前方 0.5 m 的柱子也必须判堵（轴向 0.245 会漏）。
    corner = [{"x": 0.4, "y": 0.4, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    hit = not ray_clear(0.0, 0.0, 0.0, 0.2, corner, 0.36,
                        skip_clearance_m=0.2, yaw_deg=0.0)
    ok = ok and hit
    print(f"  {'ok' if hit else 'FAIL'}  角部碰撞盒：旧圆漏判的 (0.4,0.4) 必须判堵")
    # length=0：只判当前位姿（射线继续前进 0.2 m 后 yaw=0 也会真实撞上，测不出航向差）。
    front = [{"x": 0.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    clear0 = ray_clear(0.0, 0.0, 0.0, 0.0, front, 0.36,
                       skip_clearance_m=0.2, yaw_deg=0.0)
    hit45 = not ray_clear(0.0, 0.0, 0.0, 0.0, front, 0.36,
                          skip_clearance_m=0.2, yaw_deg=45.0)
    ok = ok and clear0 and hit45
    print(f"  {'ok' if clear0 and hit45 else 'FAIL'}  45° 机身角：yaw=0 判空、"
          f"yaw=45 判堵（轴向包围盒会漏）")

    # 深度失效必须 fail closed（M5）。
    extra = [
        ("全 0 深度 -> SAFE_NEUTRAL", [0.0] * (w * h), SAFE_NEUTRAL),
        ("全 NaN 深度 -> SAFE_NEUTRAL", [float("nan")] * (w * h), SAFE_NEUTRAL),
        ("全负深度 -> SAFE_NEUTRAL", [-1.0] * (w * h), SAFE_NEUTRAL),
        ("中心带全 0、两侧 0.3 -> SAFE_NEUTRAL",
         _make(w, h, 0.3, 0.0, 0.3), SAFE_NEUTRAL),
        ("左侧无有效样本视为堵 -> YAW_RIGHT",
         _make(w, h, 0.0, 1.0, 5.0), YAW_RIGHT),
    ]
    for name, depth, want in extra:
        got = expert_action(depth, w, h)
        ok = ok and got == want
        print(f"  {'ok' if got == want else 'FAIL'}  {name}: got {ACTIONS[got]}, want {ACTIONS[want]}")

    # 小尺寸/短 buffer 必须拒绝，不能静默给出错误动作（S1）。
    bad = [
        ("h=1", _make(w, 1, 0.3, 5.0, 0.3), w, 1),
        ("h=2", _make(w, 2, 0.3, 5.0, 0.3), w, 2),
        ("h=3", _make(w, 3, 0.3, 5.0, 0.3), w, 3),
        ("w=2", _make(2, h, 0.3, 5.0, 0.3), 2, h),
        ("w=0", [], 0, h),
        ("短 buffer", [1.0] * (w * h - 1), w, h),
    ]
    for name, depth, ww, hh in bad:
        try:
            expert_action(depth, ww, hh)
        except ValueError:
            print(f"  ok  {name}: ValueError")
        else:
            ok = False
            print(f"  FAIL  {name}: 未抛 ValueError")

    # T1 回归：局部规划在门柱前会摆振；A* 必须能找到绕行路径并给出非 SAFE 动作。
    gate = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    path = astar_path(gate, (0.0, 0.0), (2.5, 0.0), clearance=0.5)
    ok = ok and path is not None
    got = plan_action_global(0.0, 0.0, 0.0, gate)
    ok = ok and got != SAFE_NEUTRAL
    print(f"  {'ok' if path is not None and got != SAFE_NEUTRAL else 'FAIL'}  "
          f"A* 绕门柱: path={len(path) if path else None}, action={ACTIONS[got]}")

    print("teacher: " + ("全部通过" if ok else "有失败"))
    raise SystemExit(0 if ok else 1)
