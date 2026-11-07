#!/usr/bin/env python3
"""teacher_belief 回归测试（纯 CPU，约 1 分钟）。

  a. 全可见（360° 视场）、K=1：argmax == teacher_kino.plan_action，且 q_bar 与 plan_costs 逐值相同；
  b. 「柱子藏在拐角后」：遮挡柱右前方，真实世界后面没柱（硬标签 FORWARD）；K=8 时 FORWARD 概率
     低于全可见软标签、SAFE+YAW 概率上升（A2D「卡车后面的孩子」：看不见的地方要留余地）；
  c. 温度单调：α 越大熵越大，argmax 不变；
  d. 确定性：同 seed 逐位相同，不同 seed 的假设不同；
  e. 3DGS 占据图通道：合成走廊能标、概率和为 1；被围死 → fail_closed + SAFE 独热。
用法：python3 sim/tools/test_teacher_belief.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen  # noqa: E402
import teacher_kino as tk  # noqa: E402
import teacher_belief as tb  # noqa: E402
from teacher import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'ok' if cond else 'FAIL'}  {name}{(': ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def entropy(p):
    return float(-(p * np.log(np.maximum(p, 1e-12))).sum())


def gazebo_scene(boxes, goal):
    x0, y0, nx, ny = tk._Field.extent(boxes, goal)
    return tb.Scene2D.from_obstacles(boxes, (), x0, y0, nx, ny, tk.GRID)


def test_a_equivalence():
    pillars, _, _ = scene_gen.sample_pillars(180)
    boxes = tk._boxes(pillars)
    goal = (max(b[0] for b in boxes) + tk.GOAL_AHEAD_M, 0.0)
    scene = gazebo_scene(boxes, goal)
    teacher = tb.BeliefTeacher(scene, tb.PRIOR_PILLARS, boxes=boxes, k=1, fov_deg=360.0, range_m=12.0)
    rng = np.random.default_rng(7)
    n = mism = 0
    while n < 15:
        x, y, yaw = rng.uniform(-0.5, 5.5), rng.uniform(-1.5, 1.5), rng.uniform(-60.0, 60.0)
        if tk._box_dist(x, y, boxes) < 0.3:
            continue
        v = rng.uniform(0.0, 0.3)
        vx, vy = v * math.cos(math.radians(yaw)), v * math.sin(math.radians(yaw))
        prev = int(rng.integers(0, 4))
        r = teacher.label(x, y, yaw, goal, tk.GOAL_RADIUS, False, vx, vy, 0.0, prev, seed=n)
        ref = tk.plan_costs(x, y, yaw, pillars, vx=vx, vy=vy, prev_action=prev)
        a_ref = tk.plan_action(x, y, yaw, pillars, vx=vx, vy=vy, prev_action=prev)
        q = r["q_bar"]
        same_q = all((math.isinf(q[a]) and math.isinf(ref[a])) or abs(q[a] - ref[a]) < 1e-9 for a in ref)
        if int(np.argmax(r["p"])) != a_ref or not same_q:
            mism += 1
        n += 1
    check("a. 全可见 K=1：argmax == plan_action 且 q_bar == plan_costs（15 个状态）", mism == 0, f"不一致 {mism}")


def test_b_hidden_pillar():
    occ = (1.0, -0.5)                                   # 遮挡柱：右前方 1 m
    boxes = ((occ[0], occ[1], 0.2, 0.2),)
    goal = (6.0, 0.0)
    scene = gazebo_scene(boxes, goal)
    pillars = [{"x": occ[0], "y": occ[1], "size": [0.4, 0.4, 3.0]}]
    hard = tk.plan_action(0.0, 0.0, 0.0, pillars, waypoint={"x": goal[0], "y": goal[1]})
    t1 = tb.BeliefTeacher(scene, tb.PRIOR_PILLARS, boxes=boxes, k=1)
    t8 = tb.BeliefTeacher(scene, tb.PRIOR_PILLARS, boxes=boxes, k=8)
    r1 = t1.label(0.0, 0.0, 0.0, goal, tk.GOAL_RADIUS, False, seed=1)
    r8 = t8.label(0.0, 0.0, 0.0, goal, tk.GOAL_RADIUS, False, seed=1)
    p1, p8 = r1["p"], r8["p"]
    check("b. 真实世界硬标签 = FORWARD", hard == FORWARD_SLOW, f"{hard}")
    check("b. 全可见 K=1 的 argmax 也是 FORWARD", int(np.argmax(p1)) == FORWARD_SLOW, f"p1={np.round(p1, 3)}")
    check("b. K=8 FORWARD 概率比全可见软标签低 ≥0.05", p8[FORWARD_SLOW] <= p1[FORWARD_SLOW] - 0.05,
          f"p1(F)={p1[FORWARD_SLOW]:.3f} p8(F)={p8[FORWARD_SLOW]:.3f}")
    others = [SAFE_NEUTRAL, YAW_LEFT, YAW_RIGHT]
    check("b. K=8 SAFE+YAW 概率上升", p8[others].sum() > p1[others].sum(),
          f"{p1[others].sum():.3f} -> {p8[others].sum():.3f}")
    check("b. 遮挡下 visible_frac < 1/3（60° 视场无遮挡上限）", r8["visible_frac"] < 1.0 / 3.0,
          f"{r8['visible_frac']:.3f}")
    check("b. 8 个假设都可行（先验柱不堵死起点）", r8["k_feasible"] == 8 and not r8["fail_closed"])
    return scene, boxes, goal


def test_c_temperature(scene, boxes, goal):
    ents, argmaxes = [], []
    for alpha in (0.5, 1.0, 2.0, 4.0):
        t = tb.BeliefTeacher(scene, tb.PRIOR_PILLARS, boxes=boxes, k=8, alpha_steps=alpha)
        r = t.label(0.0, 0.0, 0.0, goal, tk.GOAL_RADIUS, False, seed=1)
        ents.append(entropy(r["p"]))
        argmaxes.append(int(np.argmax(r["p"])))
    check("c. 温度单调：α 0.5→4 熵递增", all(a < b for a, b in zip(ents, ents[1:])),
          " ".join(f"{e:.3f}" for e in ents))
    check("c. argmax 不随温度变", len(set(argmaxes)) == 1, f"{argmaxes}")


def test_d_determinism(scene, boxes, goal):
    t = tb.BeliefTeacher(scene, tb.PRIOR_PILLARS, boxes=boxes, k=8)
    r1 = t.label(0.2, 0.1, 5.0, goal, tk.GOAL_RADIUS, False, 0.2, 0.0, 0.0, FORWARD_SLOW, seed=42)
    r2 = t.label(0.2, 0.1, 5.0, goal, tk.GOAL_RADIUS, False, 0.2, 0.0, 0.0, FORWARD_SLOW, seed=42)
    r3 = t.label(0.2, 0.1, 5.0, goal, tk.GOAL_RADIUS, False, 0.2, 0.0, 0.0, FORWARD_SLOW, seed=43)
    check("d. 同 seed 逐位相同", np.array_equal(r1["p"], r2["p"]) and np.array_equal(r1["q"], r2["q"]))
    check("d. 不同 seed 假设代价不同", not np.array_equal(r1["q"], r3["q"]))


def test_e_3dgs_occupancy():
    # 合成走廊：y=±1.5 两堵墙，x∈[-2,10]；0.2 m 格；目标 (6, 0)
    cell = 0.2
    nx, ny = 70, 30
    x0, y0 = -3.0, -3.0
    occ = np.zeros((ny, nx), dtype=bool)
    xs = x0 + np.arange(nx) * cell
    ys = y0 + np.arange(ny) * cell
    gx, gy = np.meshgrid(xs, ys)
    occ |= (np.abs(np.abs(gy) - 1.5) < 0.1) & (gx > -2.0) & (gx < 10.0)
    scene = tb.Scene2D(occ, x0, y0, cell)
    t = tb.BeliefTeacher(scene, {"kind": "occ"}, model=tb.MODEL_3DGS, k=4)
    r = t.label(0.0, 0.0, 0.0, (6.0, 0.0), 0.5, True, seed=3)
    check("e. 3DGS 走廊：概率和为 1、未 fail_closed", abs(r["p"].sum() - 1.0) < 1e-9 and not r["fail_closed"],
          f"p={np.round(r['p'], 3)} q={np.round(r['q_bar'], 2)} kf={r['k_feasible']}")
    check("e. 3DGS 走廊：argmax = FORWARD", int(np.argmax(r["p"])) == FORWARD_SLOW)
    # 被围死：紧邻 8 格全占据（格心距 0.2 m，净空 0.2−0.118=0.08 < 包络 0.112）→ 连悬停都撞，全 inf
    boxed = occ.copy()
    cc = scene.cells([[0.0, 0.0]])[0]
    boxed[cc[1] - 1:cc[1] + 2, cc[0] - 1:cc[0] + 2] = True
    boxed[cc[1], cc[0]] = False
    t2 = tb.BeliefTeacher(tb.Scene2D(boxed, x0, y0, cell), {"kind": "occ"}, model=tb.MODEL_3DGS, k=4)
    r2 = t2.label(0.0, 0.0, 0.0, (6.0, 0.0), 0.5, True, seed=3)
    check("e. 被围死 → fail_closed + SAFE 独热", r2["fail_closed"] and r2["p"][SAFE_NEUTRAL] == 1.0,
          f"p={r2['p']} kf={r2['k_feasible']}")
    # 停靠圈内 → SAFE 独热、at_goal
    r3 = t.label(5.8, 0.0, 0.0, (6.0, 0.0), 0.5, True, seed=3)
    check("e. 停靠圈内 → SAFE 独热", r3["p"][SAFE_NEUTRAL] == 1.0 and r3.get("at_goal") is True)


def main():
    test_a_equivalence()
    scene, boxes, goal = test_b_hidden_pillar()
    test_c_temperature(scene, boxes, goal)
    test_d_determinism(scene, boxes, goal)
    test_e_3dgs_occupancy()
    print("test_teacher_belief: " + ("全部通过" if not FAILS else f"失败 {len(FAILS)}: {FAILS}"))
    raise SystemExit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
