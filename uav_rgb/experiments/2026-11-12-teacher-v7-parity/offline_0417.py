#!/usr/bin/env python3
"""W1-7 离线证据：同一批状态（0417 的 300 集 12000 步），教师 v5 规则与 v7 规则（B2 过柱阵后目标取线、B3 A* 无路降余量重试）的标签对比。

只读数据，不改数据。每个状态（pose_before、yaw_before、scene.json 的柱子）分别用
  set_vehicle("e1_mini", "base", version=v5)  与  set_vehicle("e1_mini", "base", version=v7)
调 plan_action_global，再对照 actions.jsonl 里记录的 action_expert（记录的就是 v5 标签）。
这是「同状态重标」，不是新轨迹：v7 的闭环轨迹会不同，闭环数字看 run.sh 的对拍臂。

输出（JSON + 终端文字）：
  1. 记录标签与 v5 重算是否逐步相同（预期 0 个不同：旧数据按旧规则复算不变）；
  2. 区域里 YAW 步占比，记录标签、v5 重算、v7 重算：exit = x > 最远柱 x（任务卡口径，含过线后）；exit_to_line = 最远柱 x < x ≤ 最远柱 x+1.0
     （闭环评测能到的出口区）；exit_back_to_line = 最远柱后表面（x+0.2）< x ≤ 成功线（B2 生效且在线前），再按机头偏离 +x 的角度分
     ≤10° / 10–25° / >25° 三档；post_line = 过成功线之后（路径退化，闭环里不会出现）；
  3. v5 → v7 标签变化的总数、按区域与 (旧, 新) 组合；
  4. A* 标准余量无路的步（path_none）数，v7 重试后仍无路的步数；这些步上 SAFE 标签数 v5 与 v7；
  5. 相邻 L↔R 换向次数（口径同 data-algo 的 a10_teacher_flipflop.py：本步与上一步都非碰撞，都是 YAW，且方向不同），记录序列与 v7 重标序列；
  6. path_none 步上单次 plan_action_global 的耗时，v5 对 v7（B3 最多多跑 2 次 A*）。

用法：python offline_0417.py --data <collection 根目录> --out <结果.json> [--workers 3] [--limit N] [--no-timing]
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from multiprocessing import Pool

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import teacher as T  # noqa: E402

V5 = "uavrgb.teacher.v5"
V7 = T.TEACHER_VERSION
YAW = (T.YAW_LEFT, T.YAW_RIGHT)
NAMES = {T.SAFE_NEUTRAL: "SAFE", T.FORWARD_SLOW: "FWD", T.YAW_LEFT: "L", T.YAW_RIGHT: "R"}
BACK = 0.2          # 柱后表面 = x + 0.2（0.4 m 方柱）


def ep_dirs(root):
    return sorted(d for s in ("train", "val", "test") for d in glob.glob(os.path.join(root, s, "*")) if os.path.isdir(d))


ZONES = ("all", "exit", "exit_to_line", "exit_back_to_line", "exit_back_to_line_yaw_le10", "exit_back_to_line_yaw_10_25",
         "exit_back_to_line_yaw_gt25", "post_line")


def zones(x, yaw, px_max):
    """状态所在区域（可同时属于多个）。闭环评测在 x ≥ 最远柱 x + 1.0（成功线）即止，所以闭环里能到的出口区是 exit_to_line；
    0417 的集在过线后还继续飞到教师目标 x = 最远柱 x + 2.0 附近（post_line），那一段路径只剩一两格，纯追踪误差退化，闭环里不会出现。"""
    z = []
    if x > px_max:
        z.append("exit")                       # 任务卡口径：x > 最远柱 x（含 post_line）
        if x <= px_max + 1.0:
            z.append("exit_to_line")           # 闭环评测能到的出口区：最远柱 x < x ≤ 最远柱 x + 1.0
        else:
            z.append("post_line")              # 过成功线之后
    if px_max + BACK < x <= px_max + 1.0:
        z.append("exit_back_to_line")          # B2 生效（越过最远柱后表面）且还在成功线之前
        # 再按机头偏离 +x 的角度分档：旧轨迹里出口区的航向偏差是旧规则「朝 y=0 转」造成的，
        # v7 的目标在正前方，机头偏离超过容差（25°）的状态仍要转回来；看 B2 本身的效果要看机头已对准的那档。
        a = abs((yaw + 180.0) % 360.0 - 180.0)
        z.append("exit_back_to_line_yaw_le10" if a <= 10.0 else
                 "exit_back_to_line_yaw_10_25" if a <= 25.0 else "exit_back_to_line_yaw_gt25")
    return z


def work(args):
    ep_dir, timing = args
    scene = json.load(open(os.path.join(ep_dir, "scene.json"), encoding="utf-8"))
    pillars = scene["pillars"]
    px_max = max(float(p["x"]) for p in pillars)
    rows = [json.loads(line) for line in open(os.path.join(ep_dir, "actions.jsonl"), encoding="utf-8") if line.strip()]
    steps = []
    for r in rows:
        pb = r.get("pose_before")
        if not isinstance(pb, dict) or r.get("yaw_before") is None:
            continue
        steps.append({"x": float(pb["x"]), "y": float(pb["y"]), "yaw": float(r["yaw_before"]),
                      "rec": r["action_expert"], "collision": bool(r.get("collision"))})
    out = {"split": os.path.basename(os.path.dirname(ep_dir)), "ep": os.path.basename(ep_dir), "px_max": px_max, "steps": steps}
    T.set_vehicle("e1_mini", "base", version=V5)
    for s in steps:
        s["l5"] = T.plan_action_global(s["x"], s["y"], s["yaw"], pillars)
        s["none5"] = T._global_path(s["x"], s["y"], pillars, T.ASTAR_CLEARANCE_M, 0.1) is None
    T.set_vehicle("e1_mini", "base", version=V7)
    for s in steps:
        s["l7"] = T.plan_action_global(s["x"], s["y"], s["yaw"], pillars)
        s["none7"] = T._global_path(s["x"], s["y"], pillars, T.ASTAR_CLEARANCE_M, 0.1) is None
    if timing:
        # path_none 步（v5 口径）的单次耗时：先各调一次让 _FREE_CACHE 就位，再计第二次的 CPU 时间
        for ver, key in ((V5, "t5"), (V7, "t7")):
            T.set_vehicle("e1_mini", "base", version=ver)
            for s in steps:
                if s["none5"]:
                    T.plan_action_global(s["x"], s["y"], s["yaw"], pillars)
                    t0 = time.process_time()
                    T.plan_action_global(s["x"], s["y"], s["yaw"], pillars)
                    s[key] = (time.process_time() - t0) * 1000.0
    return out


def flips(seq):
    """a10 口径：本步与上一步都非碰撞、都是 YAW、方向不同；seq = [(label, collision)]。"""
    n = 0
    for i, (lab, col) in enumerate(seq):
        if col or i == 0:
            continue
        pl, pc = seq[i - 1]
        if lab in YAW and pl in YAW and pl != lab and not pc:
            n += 1
    return n


def share(k, n):
    return None if not n else round(k / n, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 集（冒烟用；0 = 全部）")
    args = ap.parse_args()
    dirs = ep_dirs(args.data)
    if not dirs:
        sys.exit(f"没有 episode：{args.data}")
    if args.limit:
        dirs = dirs[:args.limit]
    with Pool(args.workers) as pool:
        eps = pool.map(work, [(d, not args.no_timing) for d in dirs], chunksize=4)

    n = sum(len(e["steps"]) for e in eps)
    res = {"data": args.data, "episodes": len(eps), "steps": n, "v5": V5, "v7": V7}
    res["recorded_vs_v5_mismatch"] = sum(s["rec"] != s["l5"] for e in eps for s in e["steps"])
    changed = Counter()
    zstat = {z: {"n": 0, "yaw_rec": 0, "yaw_v5": 0, "yaw_v7": 0, "safe_v5": 0, "safe_v7": 0, "changed": 0} for z in ZONES}
    for e in eps:
        for s in e["steps"]:
            zs = ["all"] + zones(s["x"], s["yaw"], e["px_max"])
            ch = s["l5"] != s["l7"]
            if ch:
                changed[(NAMES[s["l5"]], NAMES[s["l7"]])] += 1
            for z in zs:
                st = zstat[z]
                st["n"] += 1
                st["yaw_rec"] += s["rec"] in YAW
                st["yaw_v5"] += s["l5"] in YAW
                st["yaw_v7"] += s["l7"] in YAW
                st["safe_v5"] += s["l5"] == T.SAFE_NEUTRAL
                st["safe_v7"] += s["l7"] == T.SAFE_NEUTRAL
                st["changed"] += ch
    for z, st in zstat.items():
        st["yaw_share_rec"] = share(st["yaw_rec"], st["n"])
        st["yaw_share_v5"] = share(st["yaw_v5"], st["n"])
        st["yaw_share_v7"] = share(st["yaw_v7"], st["n"])
    res["zones"] = zstat
    res["changed_total"] = sum(changed.values())
    res["changed_by_pair"] = {f"{a}->{b}": v for (a, b), v in sorted(changed.items())}
    none5 = [(e, s) for e in eps for s in e["steps"] if s["none5"]]
    res["path_none"] = {
        "v5_steps": len(none5),
        "v7_steps_still_none_after_retry": sum(s["none7"] for _, s in none5),
        "safe_labels_on_v5_none_steps": {"v5": sum(s["l5"] == T.SAFE_NEUTRAL for _, s in none5),
                                         "v7": sum(s["l7"] == T.SAFE_NEUTRAL for _, s in none5)},
        "label_dist_on_v5_none_steps": {"v5": dict(Counter(NAMES[s["l5"]] for _, s in none5)),
                                        "v7": dict(Counter(NAMES[s["l7"]] for _, s in none5))},
        "none7_total_steps": sum(s["none7"] for e in eps for s in e["steps"]),
    }
    res["flips"] = {
        "recorded_labels": sum(flips([(s["rec"], s["collision"]) for s in e["steps"]]) for e in eps),
        "v7_relabeled": sum(flips([(s["l7"], s["collision"]) for s in e["steps"]]) for e in eps),
        "episodes_with_flip_recorded": sum(1 for e in eps if flips([(s["rec"], s["collision"]) for s in e["steps"]]) > 0),
        "episodes_with_flip_v7": sum(1 for e in eps if flips([(s["l7"], s["collision"]) for s in e["steps"]]) > 0),
    }
    if not args.no_timing:
        t5 = [s["t5"] for _, s in none5 if "t5" in s]
        t7 = [s["t7"] for _, s in none5 if "t7" in s]

        def q(v, p):
            v = sorted(v)
            return round(v[min(len(v) - 1, int(math.ceil(p * len(v))) - 1)], 2) if v else None
        res["timing_ms_on_v5_none_steps"] = {
            "n": len(t5),
            "v5": {"median": round(statistics.median(t5), 2) if t5 else None, "p95": q(t5, 0.95), "max": round(max(t5), 2) if t5 else None},
            "v7": {"median": round(statistics.median(t7), 2) if t7 else None, "p95": q(t7, 0.95), "max": round(max(t7), 2) if t7 else None},
            "note": "单进程 CPU 时间，_FREE_CACHE 已就位的第二次调用；v7 在标准余量无路后最多多跑 2 次 A*"}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
