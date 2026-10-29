#!/usr/bin/env python3
"""TeacherPlanner：专家避障策略（训练时特权，学生不可用）。

输入：深度图 depth（行主序 float 列表，单位米；<=0 表示无读数/太近）。
输出：动作枚举值（与 src/include/uavrgb/action.hpp 对齐）：
  0=SAFE_NEUTRAL, 1=FORWARD_SLOW, 2=YAW_LEFT, 3=YAW_RIGHT。

策略：把深度图分成左/中/右三条竖带，只在**中间水平带**（对应无人机正前方
  飞行走廊，行 [0.4h, 0.6h)）内取各竖带最小深度；排除底边地面与顶部天空，
  否则地面像素（极近）会让专家误判「前方永远被挡」。前方空 -> FORWARD_SLOW；
  前方挡 -> 朝更空的一侧 YAW；三侧都堵死 -> SAFE_NEUTRAL。

纯 stdlib，不依赖 numpy；合成深度即可单测。
"""
import math

SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = 0, 1, 2, 3
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")

# 只取竖直方向的中间带（正前方走廊），比例范围 [0.4, 0.6)。
BAND_LO, BAND_HI = 0.4, 0.6


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


def ray_clear(x, y, th_deg, length, pillars, clearance, half=0.2, samples=33):
    """从 (x,y) 沿 th_deg 走 length 米，线段是否离所有柱子的**方框表面** ≥ clearance。

    不能用外接圆：0.4×0.4 方的外接圆半径 0.283 m，而车道缝隙只有 0.902 m，
    按外接圆算会把大部分窄缝判成堵死。

    采样精度：点到方框的距离是 1-Lipschitz，相邻采样点之间真实距离最多再降半个
    间隔。samples=33 时 1.6/32 = 0.05 m 间隔 → 最坏低算 25 mm（实测统计约 14 mm）。
    原先这里写"误差 < 1 mm"是错的。

    余量（已知不足，故意保留）：机体真实外接半径 = hypot(0.13,0.22)+0.1 = 0.3555 m
    （SDF 四个旋翼中心 ±0.13/±0.22、桨盘半径 0.1）。clearance=0.36 在最不利的
    角点方向只剩 4.5 mm，比上面的采样误差还小。
    为什么没有放大：实测把 clearance 提到 0.40（采样同时加密到 65）后，
    8 场景 × 80 步的穿越率从 5/8 掉到 2/8，max|y| 从 0.69 涨到 1.86——0.902 m
    的缝里根本不存在 0.40 净空的直线，planner 只能退回 base 并失去回中线。
    真正的解法是**放宽车道间距**（0.902 m 缝 vs 0.711 m 机体外接直径），
    不是继续调这个余量。见 docs/experiments/2026-10-29-teacher-traverse-ablations.md。
    """
    c, s = math.cos(math.radians(th_deg)), math.sin(math.radians(th_deg))
    for i in range(samples):
        t = length * i / (samples - 1)
        px, py = x + t * c, y + t * s
        for p in pillars:
            dx = max(0.0, abs(px - p["x"]) - half)
            dy = max(0.0, abs(py - p["y"]) - half)
            if dx * dx + dy * dy < clearance * clearance:
                return False
    return True


def plan_action(x, y, yaw_deg, pillars, lookahead=1.6, clearance=0.36,
                y_gain=25.0, y_max=30.0, tol=10.0, turn_weight=0.25, grid=5):
    """GT carrot planner：挑一条净空视线，代价 = 偏离目标角 + 0.25×当前转角。

    深度反应式专家解不了这条走廊（60 步 × 8 场景实测只有 1 个穿过，其余卡在
    门柱前左右摆）：它没有目标方向，转开之后只能直飞，航向一偏就斜着飞出
    ±1.5 m 柱区。这里改用柱子真值（特权信息，只决定标签、不进学生输入）。

    目标角 base = clamp(-y_gain*y, ±y_max) 把机器拉回走廊中线。**不能**简单地
    「从 base 向外试、取第一条空视线」：净空锥很窄时它会从 +43° 一步跳到 -47°，
    机器跟着左右各转 100° 以上、净位移为零（实测 4/8 场景这样卡死）。所以对
    -85°..85° 每 5° 求一次代价取最小；turn_weight 只用来在同分时偏向小转角，
    不影响「回中线」的方向偏好。全都堵就退回 base。
    """
    base = max(-y_max, min(y_max, -y_gain * y))
    aim, best = base, None
    for th in range(-85, 86, grid):
        if not ray_clear(x, y, th, lookahead, pillars, clearance):
            continue
        cost = (abs(th - base)
                + turn_weight * abs((yaw_deg - th + 180.0) % 360.0 - 180.0))
        if best is None or cost < best:
            aim, best = th, cost
    # 所有方向都堵时不能默认 base 可走：base 自己可能正对着柱子。
    if best is None and not ray_clear(x, y, base, lookahead, pillars, clearance):
        return SAFE_NEUTRAL
    err = (yaw_deg - aim + 180.0) % 360.0 - 180.0
    if abs(err) > tol:
        return YAW_LEFT if err < 0 else YAW_RIGHT
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
        (dict(x=0.0, y=-1.2, yaw_deg=0.0, pillars=[]), YAW_LEFT, "偏在右侧 -> 左转回中线"),
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

    print("teacher: " + ("全部通过" if ok else "有失败"))
    raise SystemExit(0 if ok else 1)
