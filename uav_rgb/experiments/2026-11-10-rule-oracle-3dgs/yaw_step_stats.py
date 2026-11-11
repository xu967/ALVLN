#!/usr/bin/env python3
"""3DGS 动力学下 YAW 动作一步的实际转角分布（给规则 v1 的 yaw_step_deg 定值）。只读已有实验输出里的 actions.jsonl，只用标准库。

用法：yaw_step_stats.py <实验输出目录> [<实验输出目录> ...] [--hold 1.0] [--out stats.json]
取每集每个 action_executed ∈ {YAW_LEFT 2, YAW_RIGHT 3} 的行，转角 = |dyaw_deg|（run_episode 写的整个 hold 窗口内的机头方位变化，
已折到 [-180, 180)）；sim_dt != --hold 的行不算（hold 0.5 的数据放在另一个目录里另量）。
按「上一步动作」分三种上下文：起转（上一步不是转向或本集第 0 步）/ 同向续转 / 反向转，因为机体有偏航惯性，续转比起转大。
汇总口径：合并两个实验里所有臂所有转向步的中位数，作为 yaw_step_deg（任务卡：「3DGS 动力学在该 hold 下实测的转角步中位数」）；
另列分臂、分方向、分上下文，看这个中位数对哪种转向有代表性。
"""
import argparse
import glob
import json
import os
import statistics
import sys

YAW_LEFT, YAW_RIGHT = 2, 3


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def describe(xs):
    return {"n": len(xs), "mean": round(statistics.fmean(xs), 2) if xs else None, "p10": None if not xs else round(pct(xs, 0.1), 2),
            "p50": None if not xs else round(pct(xs, 0.5), 2), "p90": None if not xs else round(pct(xs, 0.9), 2)}


def collect(exp_dirs, hold):
    rows = []                                   # (实验, 臂, 方向, 上下文, 转角)
    for exp in exp_dirs:
        name = os.path.basename(os.path.normpath(exp))
        for ap in sorted(glob.glob(os.path.join(exp, "*", "*", "r*", "ep_*", "actions.jsonl"))):
            arm = ap[len(exp.rstrip("/")) + 1:].split(os.sep)[0]
            prev = None
            with open(ap, encoding="utf-8") as f:
                for line in f:
                    a = json.loads(line)
                    act = a["action_executed"]
                    if act in (YAW_LEFT, YAW_RIGHT) and abs(a.get("sim_dt", hold) - hold) < 1e-9:
                        ctx = "起转" if prev not in (YAW_LEFT, YAW_RIGHT) else ("同向续转" if prev == act else "反向转")
                        rows.append((name, arm, "左" if act == YAW_LEFT else "右", ctx, abs(a["dyaw_deg"])))
                    prev = act
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--hold", type=float, default=1.0)
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = collect(args.exps, args.hold)
    if not rows:
        sys.exit("没有找到 hold=%g 的转向步" % args.hold)
    res = {"hold": args.hold, "dirs": args.exps, "pooled": describe([r[4] for r in rows]),
           "by_exp_arm": {}, "by_direction": {}, "by_context": {}}
    for key in sorted({(r[0], r[1]) for r in rows}):
        res["by_exp_arm"][f"{key[0]} / {key[1]}"] = describe([r[4] for r in rows if (r[0], r[1]) == key])
    for d in ("左", "右"):
        res["by_direction"][d] = describe([r[4] for r in rows if r[2] == d])
    for c in ("起转", "同向续转", "反向转"):
        res["by_context"][c] = describe([r[4] for r in rows if r[3] == c])
    for e in sorted({r[0] for r in rows}):
        res.setdefault("by_exp", {})[e] = describe([r[4] for r in rows if r[0] == e])
    res["yaw_step_deg_median"] = res["pooled"]["p50"]
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
