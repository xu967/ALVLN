#!/usr/bin/env python3
"""landmarks.py 自检：两色、确定性、目标近/远平衡、同排优先、指令一致。"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landmarks as lm
from instruction import parse_target_color


def make_pillars():
    ps = []
    for xi in range(3):
        for yi in range(-1, 2):
            ps.append({"name": f"p{xi}_{yi}", "x": 1.5 + xi, "y": yi * 1.6,
                       "z": 1.5, "size": [0.4, 0.4, 3.0]})
    return ps


def main():
    fails = []

    def check(ok, msg):
        if not ok:
            fails.append(msg)

    spawn = {"x": -0.5, "y": 0.0}
    near_targets = far_targets = 0
    for seed in range(20):
        ps = make_pillars()
        target, instr = lm.assign_landmarks(ps, seed, spawn)
        colored = [p for p in ps if "color" in p]
        check(len(colored) == 2, f"seed {seed}: 彩色柱数量 {len(colored)}")
        check(len({p["color"] for p in colored}) == 2,
              f"seed {seed}: 两柱颜色相同")
        check(target in lm.COLORS, f"seed {seed}: 目标色非法 {target}")
        check(parse_target_color(instr) == target,
              f"seed {seed}: 指令目标色不一致 {instr}")
        check(colored[0]["x"] == colored[1]["x"],
              f"seed {seed}: 未优先选同排")
        near = min(colored, key=lambda p: (p["x"] - spawn["x"]) ** 2 +
                   (p["y"] - spawn["y"]) ** 2)
        if near["color"] == target:
            near_targets += 1
        else:
            far_targets += 1

    check(near_targets > 0 and far_targets > 0,
          f"目标近/远不均衡: near={near_targets} far={far_targets}")

    ps1, ps2 = make_pillars(), make_pillars()
    t1, i1 = lm.assign_landmarks(ps1, 42)
    t2, i2 = lm.assign_landmarks(ps2, 42)
    check((t1, i1) == (t2, i2), "同 seed 不确定")
    check([p["color"] for p in ps1 if "color" in p] ==
          [p["color"] for p in ps2 if "color" in p], "同 seed 颜色不一致")

    try:
        lm.assign_landmarks(copy.deepcopy(make_pillars()[:1]), 0)
        fails.append("单柱未抛 ValueError")
    except ValueError:
        pass

    if fails:
        for f in fails:
            print("FAIL:", f)
        return 1
    print("test_landmarks: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
