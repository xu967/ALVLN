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
