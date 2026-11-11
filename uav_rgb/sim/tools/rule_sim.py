#!/usr/bin/env python3
"""规则层的极简运动学模拟（不起 Gazebo）：前进一步 0.25 m、转向一步 26°、SAFE_NEUTRAL 原地不动。

用途：规则 v1 / v2 的离线检查（单柱绕行、train 分区几何过线数）。每步调用的规则代码与 collect.py 是同一份
（column_distances → rule_v0_entry / rule_v1_step / rule_v2_step），所以这里的行为差异只来自运动学简化，不来自规则实现。
简化：转向步不位移、前进步无侧漂、位姿无延迟；Gazebo 里每步 1.0 s 的实际位移 0.2–0.3 m、转角 26.1° ± 1.1°。
不计入判据的只有 |y| ≤ 3 m 的走廊（success 按 eval 口径包含走廊，另给 out_of_corridor）。

用法（仓库根）：.venv-tools/bin/python sim/tools/rule_sim.py [--rule v0|v1|v2] [--count 20] [--steps 80]   # train 分区几何的离线检查
纯 stdlib。
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_policy as rp
from rule_policy import SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT

START_X = -0.5            # scene_gen.SPAWN_X
FORWARD_M = 0.25          # 一步 FORWARD_SLOW 的位移（rule60 的 oracle.jsonl 里首批前进步实测 0.2–0.3 m）
TURN_DEG = 26.0           # 一步 YAW 的转角（heading-guard 数据 308 步 26.12° ± 1.09°）
BODY_RADIUS_M = 0.102     # e1_mini.json collision_envelope
GOAL_MARGIN_M = 1.0       # 同 collect.GOAL_MARGIN_M：成功线 = 最远柱 x + 1.0
CORRIDOR_HALF_M = 3.0     # 走廊半宽（判据 sim09.v2.coast1s.corridor3m）
PATH_SAMPLE_M = 0.05      # 前进段上取样查碰撞的间隔


def gap_to_pillar(p, x, y):
    """点 (x, y) 到一根柱的水平截面（方柱或圆柱）表面的距离；在柱内为 0。"""
    if "radius" in p:
        return max(0.0, math.hypot(x - p["x"], y - p["y"]) - p["radius"])
    sx, sy = p.get("size", (0.4, 0.4))[:2]
    dx = max(abs(x - p["x"]) - sx / 2, 0.0)
    dy = max(abs(y - p["y"]) - sy / 2, 0.0)
    return math.hypot(dx, dy)


def min_gap(pillars, x, y):
    return min((gap_to_pillar(p, x, y) for p in pillars), default=float("inf"))


def max_reversal_run(actions):
    """相邻两步都是转向且方向相反（L→R 或 R→L）算一次换向；返回最长的连续换向次数（中间夹前进 / SAFE 就断开）。"""
    run = best = 0
    for a, b in zip(actions, actions[1:]):
        if {a, b} == {YAW_LEFT, YAW_RIGHT}:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def simulate(pillars, rule="v1", hfov_deg=60.0, steps=80, stop_at_goal=True, start=(START_X, 0.0, 0.0),
             v1_params=rp.RULE_V1, v0_params=rp.RULE_V0, v2_params=rp.RULE_V2):
    """跑一集。rule: "v0" | "v1" | "v2" | "forward"（一直前进）| "left"（一直左转），后两个只给模拟自身的测试用。

    返回 dict：success（过线且不撞且不出走廊）、collided、out_of_corridor、steps（执行的步数）、goal_x、n_final、min_gap、
    traj（起点 + 每步之后的 (x, y, yaw_deg)）、actions、entries、mem_n（仅 v1 / v2）、clear（仅 v1 / v2，每步 [左, 保持, 右]）、cols（每步 d）。
    stop_at_goal=False 时过线后继续跑满 steps（检查绕过后航向是否回正）。
    """
    goal_x = max(float(p["x"]) for p in pillars) + GOAL_MARGIN_M
    x, y, yaw = start
    n, last_turn, memory = 0, None, []
    tabu, prev_action = [], None                      # 仅 v2：禁忌表与上一步执行的动作
    traj, actions, entries, mem_n, clears, cols = [(x, y, yaw)], [], [], [], [], []
    gap_min = min_gap(pillars, x, y)
    collided = outside = reached = False
    for step in range(steps):
        d = rp.column_distances(x, y, yaw, pillars, hfov_deg=hfov_deg)
        if rule == "v1":
            action, entry, memory, info = rp.rule_v1_step(memory, step, x, y, yaw, d, n, last_turn, hfov_deg, v1_params)
            mem_n.append(info["mem_n"])
            clears.append(info["clear"])
        elif rule == "v2":
            action, entry, memory, tabu, info = rp.rule_v2_step(memory, tabu, step, x, y, yaw, d, n, last_turn, prev_action=prev_action,
                                                                hfov_deg=hfov_deg, params=v2_params)
            mem_n.append(info["mem_n"])
            clears.append(info["clear"])
        elif rule == "v0":
            action, entry = rp.rule_v0_entry(d, n, last_turn, v0_params)
        elif rule == "forward":
            action, entry = FORWARD_SLOW, "forward"
        elif rule == "left":
            action, entry = YAW_LEFT, "left"
        else:
            raise ValueError(rule)
        cols.append(d)
        if action == FORWARD_SLOW:
            m = max(1, round(FORWARD_M / PATH_SAMPLE_M))
            c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            for i in range(1, m + 1):
                gap_min = min(gap_min, min_gap(pillars, x + c * FORWARD_M * i / m, y + s * FORWARD_M * i / m))
            x, y = x + c * FORWARD_M, y + s * FORWARD_M
        elif action == YAW_LEFT:
            yaw += TURN_DEG
            n += 1
            last_turn = YAW_LEFT
        elif action == YAW_RIGHT:
            yaw -= TURN_DEG
            n -= 1
            last_turn = YAW_RIGHT
        elif action != SAFE_NEUTRAL:
            raise ValueError(action)
        prev_action = action
        actions.append(action)
        entries.append(entry)
        traj.append((x, y, yaw))
        collided = collided or gap_min < BODY_RADIUS_M
        outside = outside or abs(y) > CORRIDOR_HALF_M
        if x >= goal_x:
            reached = True
            if stop_at_goal:
                break
    return {"success": reached and not collided and not outside, "collided": collided, "out_of_corridor": outside,
            "reached": reached, "steps": len(actions), "goal_x": goal_x, "n_final": n, "min_gap": gap_min,
            "traj": traj, "actions": actions, "entries": entries, "mem_n": mem_n, "clear": clears, "cols": cols}


def train_geometries(count=20, task="base", first_seed=1):
    """train 分区的前 count 个互不相同的 base 几何：[(seed, pillars)]（seed 从 first_seed 起升序，按 split_manifest 的几何哈希判分区）。"""
    import scene_gen
    import split_manifest as sm
    out, gids, seed = [], set(), first_seed
    while len(out) < count:
        pillars = scene_gen.sample_pillars(seed)[0]
        gid = sm.geometry_gid(pillars)
        if gid not in gids and sm.split_for_gid(gid) == "train":
            gids.add(gid)
            out.append((seed, pillars))
        seed += 1
    return out


def main():
    ap = argparse.ArgumentParser(description="train 分区几何上的离线（运动学）检查；只用 train 几何，不碰 val / test")
    ap.add_argument("--rule", choices=("v0", "v1", "v2"), default="v1")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--hfov-deg", type=float, default=60.0)
    ap.add_argument("--verbose", action="store_true", help="每个几何打印条目序列")
    args = ap.parse_args()
    geoms = train_geometries(args.count)
    ok = coll = 0
    for seed, pillars in geoms:
        r = simulate(pillars, args.rule, args.hfov_deg, args.steps)
        ok += r["success"]
        coll += r["collided"]
        tag = "过线" if r["success"] else ("撞柱" if r["collided"] else ("出走廊" if r["out_of_corridor"] else "未到线"))
        runs = max_reversal_run(r["actions"])
        print(f"seed {seed:>3}  {tag}  steps={r['steps']:>2}  max_x={max(t[0] for t in r['traj']):5.2f}  n_final={r['n_final']:>2}  "
              f"min_gap={r['min_gap']:.3f}  最长换向 {runs}")
        if args.verbose or not r["success"]:
            print("   条目:", " ".join(r["entries"]))
    print(f"规则 {args.rule}，train 分区 {len(geoms)} 个几何（seed {[s for s, _ in geoms]}）：过线 {ok}/{len(geoms)}，撞柱 {coll}/{len(geoms)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
