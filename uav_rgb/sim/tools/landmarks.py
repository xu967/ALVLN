#!/usr/bin/env python3
"""L1 彩色 landmark 分配（纯函数，不依赖 Gazebo）。

每集从同一排柱里选 2 根，分别染成目标色/干扰色；指令只提目标色。
目标靠近/远离出生点按 seed 奇偶交替，避免「目标总是最近」的位置捷径。
"""
import random

from instruction import make_instruction

COLORS = ("red", "blue", "yellow", "green")
# SDF 材质 RGB；灰柱仍是 scene_gen 的 0.6。
COLOR_RGB = {
    "red": (0.80, 0.05, 0.05),
    "blue": (0.05, 0.10, 0.80),
    "yellow": (0.90, 0.80, 0.05),
    "green": (0.05, 0.70, 0.10),
}


def _dist2(p, spawn):
    return (p["x"] - spawn["x"]) ** 2 + (p["y"] - spawn["y"]) ** 2


def assign_landmarks(pillars, seed, spawn=None):
    """就地给两根柱加 color，返回 (target_color, instruction)。

    spawn 默认 (-0.5, 0)；scene_gen 应传真实出生位姿。
    优先选同一 x 的一排（同时可见），没有同排就任选两根。
    """
    if len(pillars) < 2:
        raise ValueError("pillars 少于 2 根，无法放目标/干扰")
    spawn = spawn or {"x": -0.5, "y": 0.0}
    rng = random.Random(seed ^ 0x1A5D)

    by_x = {}
    for p in pillars:
        by_x.setdefault(round(p["x"], 3), []).append(p)
    rows = [ps for ps in by_x.values() if len(ps) >= 2]
    if rows:
        pair = rng.choice(rows)[:2]
    else:
        pair = rng.sample(pillars, 2)

    # 近/远按出生点距离排序；seed 奇偶决定目标在近端还是远端。
    ordered = sorted(pair, key=lambda p: _dist2(p, spawn))
    near, far = ordered[0], ordered[-1]
    target = near if seed % 2 == 0 else far
    distractor = far if target is near else near

    colors = list(COLORS)
    rng.shuffle(colors)
    target["color"], distractor["color"] = colors[0], colors[1]
    return target["color"], make_instruction(target["color"])
