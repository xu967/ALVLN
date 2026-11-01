#!/usr/bin/env python3
"""landmarks.py 自检：同色柱对、分居左右、关系词一致、确定性、左右平衡。"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import landmarks as lm  # noqa: E402
from instruction import parse_target_color  # noqa: E402


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

    # 出生在门柱与侧柱之间的缝里（random 模式的 y）；default 的 y=0 只能选到
    # 跨 3.2 m 的两根侧柱，超出 60° 视场，见文末断言。
    spawn = {"x": -0.5, "y": 0.8, "yaw_deg": 0.0}
    left_n = right_n = 0
    for seed in range(20):
        ps = make_pillars()
        color, relation, instr, wp = lm.assign_landmarks(ps, seed, spawn)
        colored = [p for p in ps if "color" in p]
        check(len(colored) == 2, f"seed {seed}: 彩色柱数量 {len(colored)}")
        check(len({p["color"] for p in colored}) == 1,
              f"seed {seed}: 两柱颜色不同 {[p['color'] for p in colored]}")
        check(color in lm.COLORS, f"seed {seed}: 目标色非法 {color}")
        check(parse_target_color(instr) == color,
              f"seed {seed}: 指令目标色不一致 {instr}")
        check(relation in instr.split(), f"seed {seed}: 指令缺关系词 {instr}")
        check(all(p["x"] <= lm.VISIBLE_X_MAX for p in colored),
              f"seed {seed}: 彩色柱不在前排 {colored}")
        lat = sorted(lm.lateral(p, spawn) for p in colored)
        check(lat[0] <= -lm.LATERAL_MIN and lat[1] >= lm.LATERAL_MIN,
              f"seed {seed}: 两柱未分居左右 lateral={lat}")
        target = min(colored, key=lambda p: lm.lateral(p, spawn))
        if relation == "right":
            target = max(colored, key=lambda p: lm.lateral(p, spawn))
        check(wp["x"] == target["x"], f"seed {seed}: waypoint 不在目标柱")
        check(abs(wp["y"]) <= 3.0, f"seed {seed}: waypoint 越界 {wp}")
        # v4 回归守卫：waypoint 必须在柱子外侧，离出生航线的横移要够大，
        # 否则两个候选点又会落在同一条直线上（闭环测不出关系词）。
        check(lm.lateral(wp, spawn) * lm.lateral(target, spawn) > 0,
              f"seed {seed}: waypoint 不在目标柱外侧 {wp}")
        check(abs(lm.lateral(wp, spawn)) > 0.9,
              f"seed {seed}: waypoint 横移不足 {lm.lateral(wp, spawn):.2f} m")
        check(all(abs(wp["x"] - p["x"]) >= 0.6 or abs(wp["y"] - p["y"]) >= 0.6
                  for p in colored), f"seed {seed}: waypoint 落在彩色柱内 {wp}")
        if relation == "left":
            left_n += 1
        else:
            right_n += 1

    check(left_n > 0 and right_n > 0,
          f"左右不均衡: left={left_n} right={right_n}")

    ps1, ps2 = make_pillars(), make_pillars()
    r1 = lm.assign_landmarks(ps1, 42, spawn)
    r2 = lm.assign_landmarks(ps2, 42, spawn)
    check(r1 == r2, "同 seed 不确定")
    check([p["color"] for p in ps1 if "color" in p] ==
          [p["color"] for p in ps2 if "color" in p], "同 seed 颜色不一致")

    try:
        lm.assign_landmarks(copy.deepcopy(make_pillars()[:1]), 0, spawn)
        fails.append("单柱未抛 ValueError")
    except ValueError:
        pass

    # default 出生（y=0）选不出跨距合格的同色柱对：必须报错，不能静默产出一个
    # 「两根柱有一根在视场外」的伪关系任务。
    try:
        lm.assign_landmarks(make_pillars(), 0, {"x": -0.5, "y": 0.0})
        fails.append("y=0 出生应抛 ValueError（视场装不下左右两根柱）")
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
