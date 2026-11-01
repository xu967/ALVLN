#!/usr/bin/env python3
"""L1 左右关系 landmark 分配（纯函数，不依赖 Gazebo）。

每集从出生点附近的柱子里选 **两根同色** 柱，分居出生时刻机头轴线的左右两侧；
指令用关系词（left/right）指定目标。同色是关键：图像里没有任何颜色线索能把
两者分开，模型必须读指令才知道往哪边飞。
"""
import math
import random

from instruction import make_instruction

COLORS = ("red", "blue", "yellow", "green")
# 实际场景每 x 只有一根柱（蛇形）；彩色柱必须落在出生视野内的前排。
VISIBLE_X_MAX = 3.0
# 左右判定的最小横向偏移：贴着机头轴线的柱子在图像里和中心分不开。
LATERAL_MIN = 0.3
# 两柱横向跨距上限。相机水平视场 60°（sim/worlds/uav_rgb.sdf），出生点距彩色柱
# 约 2 m，单侧偏移要 ≤1.2 m 才能让两根同时留在 ±30° 内。跨 3.2 m 的（±1.6 两条
# 侧车道）在 60° 下必然有一根出画，「左右」就没法从画面里判定了。
LATERAL_SPAN_MAX = 2.4
# 出画判定：柱心相对光轴的水平偏角必须在半视场减去余量之内。60° 视场下实测
# 500 个 seed 里有 15 个（3%）的最外侧柱心超过 30°，柱子只剩一条边在画幅里。
HALF_FOV_DEG = 30.0
FOV_MARGIN_DEG = 2.0
# 相机相对机身原点的前向偏移（SDF 的 <pose>0.2 0 0</pose>）。
CAMERA_FORWARD_M = 0.2
# SDF 材质 RGB；灰柱仍是 scene_gen 的 0.6。
COLOR_RGB = {
    "red": (0.80, 0.05, 0.05),
    "blue": (0.05, 0.10, 0.80),
    "yellow": (0.90, 0.80, 0.05),
    "green": (0.05, 0.70, 0.10),
}


def lateral(p, spawn):
    """柱相对出生时刻机身左向的偏移（正 = 左）。

    出生时刻机头朝 +x，机身上轴朝 +z，所以左是 +y；spawn 的 yaw 一并折算，
    random 出生模式带 ±5°，3 m 外会偏 0.26 m，比车道间距小但没必要留这个误差。
    """
    yaw = math.radians(spawn.get("yaw_deg") or 0.0)
    dx, dy = p["x"] - spawn["x"], p["y"] - spawn["y"]
    return dy * math.cos(yaw) - dx * math.sin(yaw)


def h_angle(p, spawn):
    """柱心相对相机光轴的水平偏角（度，左正右负）。相机在机身 +0.2 m 处。"""
    yaw = math.radians(spawn.get("yaw_deg") or 0.0)
    cx = spawn["x"] + CAMERA_FORWARD_M * math.cos(yaw)
    cy = spawn["y"] + CAMERA_FORWARD_M * math.sin(yaw)
    dx, dy = p["x"] - cx, p["y"] - cy
    return math.degrees(math.atan2(
        -dx * math.sin(yaw) + dy * math.cos(yaw),
        dx * math.cos(yaw) + dy * math.sin(yaw)))


def waypoint_for(target, pillars, spawn, standoff=0.8, y_range=(-3.0, 3.0)):
    """目标柱旁的通路点：放在柱子**远离出生航线**的一侧。

    standoff=0.8 = 柱半宽 0.2 + A* 余量 0.5 + 0.1 栅格余量：
    0.7 时 waypoint 的 0.1 m 栅格单元会落进 0.5 m 余量内，A* 直接失败
    （L1 pilot 7/52 实测），0.8 后可达。

    2026-11-01 v4：原来朝走廊中心偏。柱子只在 y≈0 / ±1.6 三条车道，出生点在
    y=±(0.72–0.88)，于是两根候选柱的 waypoint 都落在 ±0.8——正好是出生点自己的
    航线，274 个 test 场景没有一个需要按关系词横移。改成朝外偏后横向位移 1.6 m。
    """
    sign = 1.0 if lateral(target, spawn) > 0 else -1.0
    wy = round(target["y"] + sign * standoff, 3)
    if not y_range[0] <= wy <= y_range[1]:
        raise ValueError(f"waypoint y={wy} 超出走廊 {y_range}")
    for p in pillars:
        if p is target:
            continue
        if (abs(target["x"] - p["x"]) < 0.2 + 0.4 and
                abs(wy - p["y"]) < 0.2 + 0.4):
            raise ValueError(
                f"{target['name']} 外侧 waypoint ({target['x']}, {wy}) 被 {p['name']} 挡住")
    return {"x": target["x"], "y": wy}


def assign_landmarks(pillars, seed, spawn=None):
    """就地给两根柱染成同一颜色，返回 (color, relation, instruction, waypoint)。

    2026-11-01 场景升级：原来是「目标色 + 干扰色」两根异色柱，图像里目标色唯一，
    模型完全可以忽略文本（实测 L≈I≈S，换词只把动作概率改 2e-4）；同色后图像只
    给出「有两个候选」，选哪个必须读指令。

    relation 由 seed 奇偶决定，保证 left/right 平衡；left = 经过时目标柱在机身左侧。
    waypoint 在柱子外侧，所以停止点落在柱子的另一侧：柱子留在左边，停止点就在它的
    右边。两根柱必须跨在机头轴线两侧，否则「左右」无从判定——直接报错。
    """
    if len(pillars) < 2:
        raise ValueError("pillars 少于 2 根，无法放左右 landmark")
    spawn = spawn or {"x": -0.5, "y": 0.0, "yaw_deg": 0.0}
    rng = random.Random(seed ^ 0x1A5D)

    cand = [p for p in pillars if p["x"] <= VISIBLE_X_MAX]
    if len(cand) < 2:
        cand = list(pillars)
    pairs = []
    for i, a in enumerate(cand):
        for b in cand[i + 1:]:
            left, right = (a, b) if lateral(a, spawn) > lateral(b, spawn) else (b, a)
            lat_l, lat_r = lateral(left, spawn), lateral(right, spawn)
            if (lat_l >= LATERAL_MIN and lat_r <= -LATERAL_MIN
                    and lat_l - lat_r <= LATERAL_SPAN_MAX):
                pairs.append((left, right))
    if not pairs:
        raise ValueError(
            f"出生点 y={spawn['y']:.2f} 处找不到分居左右、跨距 ≤{LATERAL_SPAN_MAX} m "
            "的两根柱；L1 关系任务必须用 --spawn-mode random")

    # 优先选两根都留在视场里的柱对；一组都留不下时退回全部候选（不提高失败率）。
    limit = HALF_FOV_DEG - FOV_MARGIN_DEG
    visible = [pr for pr in pairs
               if max(abs(h_angle(pr[0], spawn)), abs(h_angle(pr[1], spawn))) <= limit]
    left, right = rng.choice(visible or pairs)
    relation = "left" if seed % 2 == 0 else "right"
    # v4：waypoint 在柱子外侧，目标柱在停止点的内侧——"pass it on your left"
    # 对应「目标 = 右边那根柱」，否则指令字面和几何相反。
    target = right if relation == "left" else left
    color = rng.choice(COLORS)
    left["color"] = right["color"] = color
    return (color, relation, make_instruction(color, relation),
            waypoint_for(target, pillars, spawn))
