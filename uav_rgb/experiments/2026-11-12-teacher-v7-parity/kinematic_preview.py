#!/usr/bin/env python3
"""不起 Gazebo 的闭环预演：用 teacher._rollout 的运动学原语（FORWARD 约 0.25 m；YAW ±27° 并沿新航向平移 0.11 m）加高斯噪声，
让教师（v5 与 v7 规则）在 20 个冻结 seed 的柱阵里自己飞，比较成功、撞柱、出口区 YAW 步占比。

这只是预演：没有 cf_sitl 的惯性、滑行与姿态噪声，数字不能替代 run.sh 的 Gazebo 对拍，也不进验收。用途是在排队前看 B2、B3 有没有明显异常
（摆头、绕圈、擦柱、出口区仍大量转向），以及 P2 预测（出口区 YAW < 5%）大概落在哪一侧。

口径与 summarize.py 一致：出口区 = 步的 pose_before.x > 最远柱 x；YAW = 动作 ∈ {YAW_LEFT, YAW_RIGHT}；占比按步池化；到 x ≥ 最远柱 x + 1.0 即止；
成功 = 到线、没撞柱、|y| ≤ 3.0（走廊）。出生点 x=SPAWN_X、y=0、航向 0（scene_gen.make_spawn default 模式）。

用法：python kinematic_preview.py --out <结果.json> [--reps 10] [--versions v5,v7]
"""
import argparse
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import scene_gen  # noqa: E402
import teacher as T  # noqa: E402

SEEDS = [1, 113, 122, 153, 172, 176, 180, 230, 250, 536, 555, 628, 683, 798, 929, 990, 992, 997, 1113, 1654]
VERSIONS = {"v5": "uavrgb.teacher.v5", "v7": T.TEACHER_VERSION}
YAW = (T.YAW_LEFT, T.YAW_RIGHT)
STEPS = 80
CORRIDOR = 3.0
SIG_FWD_M, SIG_YAW_DEG, SIG_DRIFT_M = 0.03, 4.0, 0.02      # 噪声（粗估，未按 cf_sitl 标定）


def step_pose(rng, x, y, yaw, action):
    nx, ny, nyaw = T._rollout(x, y, yaw, action)
    if action == T.FORWARD_SLOW:
        d = rng.gauss(0.0, SIG_FWD_M)
        return nx + d * math.cos(math.radians(yaw)), ny + d * math.sin(math.radians(yaw)), nyaw
    if action in YAW:
        nyaw += rng.gauss(0.0, SIG_YAW_DEG)
        d = rng.gauss(0.0, SIG_DRIFT_M)
        return nx + d * math.cos(math.radians(nyaw)), ny + d * math.sin(math.radians(nyaw)), nyaw
    return nx, ny, nyaw


def episode(seed, rep, version):
    pillars, _clr, _att = scene_gen.sample_pillars(seed)
    px_max = max(p["x"] for p in pillars)
    goal_x = px_max + 1.0
    rng = random.Random(seed * 1000 + rep)
    T.set_vehicle("e1_mini", "base", version=VERSIONS[version])
    x, y, yaw = scene_gen.SPAWN_X, 0.0, 0.0
    rows, collision, reached, violation = [], False, False, False
    for _ in range(STEPS):
        a = T.plan_action_global(x, y, yaw, pillars)
        rows.append((x, y, yaw, a))
        x, y, yaw = step_pose(rng, x, y, yaw, a)
        if any(T._body_hits_box(x, y, yaw, p, 0.2) for p in pillars):
            collision = True
            break
        if abs(y) > CORRIDOR:
            violation = True
        if x >= goal_x:
            reached = True
            break
    exit_rows = [r for r in rows if r[0] > px_max]
    return {"seed": seed, "rep": rep, "steps": len(rows), "collision": collision, "reached": reached, "violation": violation,
            "success": reached and not collision and not violation,
            "exit_n": len(exit_rows), "exit_yaw": sum(r[3] in YAW for r in exit_rows),
            "exit_yaw_gt25": sum(1 for r in exit_rows if r[3] in YAW and abs((r[2] + 180) % 360 - 180) > 25.0),
            "flips": sum(1 for i in range(1, len(rows)) if rows[i][3] in YAW and rows[i - 1][3] in YAW and rows[i][3] != rows[i - 1][3])}


def wilson(k, n, z=1.96):
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def p_pass(eps, trials=20000, thr=0.05):
    """对拍的预测通过概率：每个 seed 随机取一个重复（真实对拍每个 seed 只跑一集），合并 20 集后出口区 YAW 占比 < thr 的比例。"""
    by_seed = {}
    for e in eps:
        by_seed.setdefault(e["seed"], []).append(e)
    rng = random.Random(20261112)
    ok = 0
    for _ in range(trials):
        pick = [rng.choice(v) for v in by_seed.values()]
        n = sum(e["exit_n"] for e in pick)
        ok += n > 0 and sum(e["exit_yaw"] for e in pick) / n < thr
    return round(ok / trials, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=10, help="每个 seed 的重复次数（噪声种子不同）")
    ap.add_argument("--versions", default="v5,v7")
    args = ap.parse_args()
    res = {"seeds": SEEDS, "reps": args.reps, "noise": {"fwd_m": SIG_FWD_M, "yaw_deg": SIG_YAW_DEG, "drift_m": SIG_DRIFT_M}, "arms": {}}
    for v in args.versions.split(","):
        eps = [episode(s, r, v) for s in SEEDS for r in range(args.reps)]
        n = len(eps)
        en, ey = sum(e["exit_n"] for e in eps), sum(e["exit_yaw"] for e in eps)
        res["arms"][v] = {
            "teacher_version": VERSIONS[v], "episodes": n, "success": sum(e["success"] for e in eps),
            "collision": sum(e["collision"] for e in eps), "violation": sum(e["violation"] for e in eps),
            "not_reached": sum(not e["reached"] and not e["collision"] for e in eps),
            "mean_steps": round(sum(e["steps"] for e in eps) / n, 2),
            "exit_steps": en, "exit_yaw": ey, "exit_yaw_share": round(ey / en, 4) if en else None, "exit_yaw_ci95": wilson(ey, en),
            "exit_yaw_with_heading_off_gt25": sum(e["exit_yaw_gt25"] for e in eps),
            "episodes_with_exit_yaw": sum(e["exit_yaw"] > 0 for e in eps), "adjacent_LR_flips": sum(e["flips"] for e in eps),
            "failed_seeds": sorted({e["seed"] for e in eps if not e["success"]}),
            "p_exit_yaw_share_lt_5pct_for_one_run_of_20_seeds": p_pass(eps),
            "per_episode": eps,
        }
        print(v, json.dumps({k: x for k, x in res["arms"][v].items() if k != "per_episode"}, ensure_ascii=False), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
