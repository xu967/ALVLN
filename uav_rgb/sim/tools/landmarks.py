#!/usr/bin/env python3
"""L1 彩色 landmark 分配（纯函数，不依赖 Gazebo）。

每集从同一排柱里选 2 根，分别染成目标色/干扰色；指令只提目标色。
目标靠近/远离出生点按 seed 奇偶交替，避免「目标总是最近」的位置捷径。
"""
import random

from instruction import make_instruction

COLORS = ("red", "blue", "yellow", "green")
# 实际场景每 x 只有一根柱（蛇形）；彩色柱必须落在出生视野内的前排。
VISIBLE_X_MAX = 3.0
# SDF 材质 RGB；灰柱仍是 scene_gen 的 0.6。
COLOR_RGB = {
    "red": (0.80, 0.05, 0.05),
    "blue": (0.05, 0.10, 0.80),
    "yellow": (0.90, 0.80, 0.05),
    "green": (0.05, 0.70, 0.10),
}


def _dist2(p, spawn):
    return (p["x"] - spawn["x"]) ** 2 + (p["y"] - spawn["y"]) ** 2


def waypoint_for(target, pillars, standoff=0.7, y_range=(-3.0, 3.0)):
    """目标柱旁的通路点：朝走廊中心偏 standoff，避开其它柱。

    standoff=0.7 = 柱半宽 0.2 + A* 余量 0.5，正好落在 A* 的自由一侧。
    """
    sign = -1.0 if target["y"] > 0 else 1.0
    for _ in range(2):
        wy = round(target["y"] + sign * standoff, 3)
        if y_range[0] <= wy <= y_range[1]:
            clear = True
            for p in pillars:
                if p is target:
                    continue
                if (abs(target["x"] - p["x"]) < 0.2 + 0.4 and
                        abs(wy - p["y"]) < 0.2 + 0.4):
                    clear = False
                    break
            if clear:
                return {"x": target["x"], "y": wy}
        sign = -sign
    raise ValueError("找不到目标柱旁的 waypoint")


def assign_landmarks(pillars, seed, spawn=None):
    """就地给两根柱加 color，返回 (target_color, instruction, waypoint)。

    spawn 默认 (-0.5, 0)；scene_gen 应传真实出生位姿。
    从前排视野带（x<=VISIBLE_X_MAX）选两根 y 不同的柱，让它们同时可见；
    没有可用对再退回任意两根。
    """
    if len(pillars) < 2:
        raise ValueError("pillars 少于 2 根，无法放目标/干扰")
    spawn = spawn or {"x": -0.5, "y": 0.0}
    rng = random.Random(seed ^ 0x1A5D)

    cand = [p for p in pillars if p["x"] <= VISIBLE_X_MAX]
    if len(cand) < 2:
        cand = pillars
    pairs = [(a, b) for i, a in enumerate(cand) for b in cand[i + 1:]
             if abs(a["y"] - b["y"]) > 0.5]
    if not pairs:
        pairs = [(a, b) for i, a in enumerate(cand) for b in cand[i + 1:]]
    pair = rng.choice(pairs)

    # 近/远按出生点距离排序；seed 奇偶决定目标在近端还是远端。
    ordered = sorted(pair, key=lambda p: _dist2(p, spawn))
    near, far = ordered[0], ordered[-1]
    target = near if seed % 2 == 0 else far
    distractor = far if target is near else near

    colors = list(COLORS)
    rng.shuffle(colors)
    target["color"], distractor["color"] = colors[0], colors[1]
    wp = waypoint_for(target, pillars)
    return target["color"], make_instruction(target["color"]), wp
