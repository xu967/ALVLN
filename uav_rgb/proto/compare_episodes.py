#!/usr/bin/env python3
"""卡 E2E-P0 第二关（闭环部分）的对照：同一份 checkpoint、同一批冻结 seed，各臂的逐步动作序列与参照臂比。
只读 eval_closed_loop.py 的输出目录（<臂>/eval.json 与 <臂>/episodes/seed_<n>/actions.jsonl），不碰仿真。

参照臂：W1-0 第二轮的 acc-L0.20-k1-s1（C++ 推理，同权重 cal_swa_s1_k10.bin，同新协议）。
比较的是 action_executed 逐步序列。有分歧时给出第一处分歧的步号，以及该步与前一步两边的
dispatch_lag_s、late_dispatch、infer_ms、pose_before、yaw_before、action_expert（教师影子标签）、executed_by。
同时报分歧之前的位姿最大差：动作都一样而位姿已有差，说明分歧的根在仿真本身，不在推理端。
臂目录若有同名 -retry1 兄弟目录（R6：含迟到步的集重跑一次），该 seed 以重跑为准，首次记录另列。

用法：
  python proto/compare_episodes.py --ref <R2 臂目录> --arm wrap=<目录> [--arm cpp=<目录> ...] --seeds 1,113,122 \
      --out compare.json --md compare.md
"""
import argparse
import json
import math
import os
import sys

import numpy as np

STEP_KEYS = ("step", "timestamp_ns", "action_executed", "action_expert", "executed_by", "dispatch_lag_s",
             "late_dispatch", "infer_ms", "collision")


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_path(arm_dir, seed):
    return os.path.join(arm_dir, "episodes", f"seed_{seed}", "actions.jsonl")


def load_eval(arm_dir):
    path = os.path.join(arm_dir, "eval.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def per_episode_outcome(ev, seed):
    if not ev:
        return None
    for e in ev.get("per_episode", []):
        if e.get("seed") == seed:
            return {k: e.get(k) for k in ("success", "collision", "steps", "spl", "timed_out", "stuck", "corridor_violation")}
    return None


def resolve(arm_dir, seed):
    """返回 (用的目录, 是否来自重跑, 首次记录的目录或 None)。"""
    retry = arm_dir.rstrip("/") + "-retry1"
    if os.path.isfile(episode_path(retry, seed)):
        return retry, True, (arm_dir if os.path.isfile(episode_path(arm_dir, seed)) else None)
    return arm_dir, False, None


def yaw_diff(a, b):
    d = (a - b + math.pi) % (2 * math.pi) - math.pi
    return abs(d)


def step_view(r):
    out = {k: r.get(k) for k in STEP_KEYS}
    out["pose_before"] = r.get("pose_before")
    out["yaw_before"] = r.get("yaw_before")
    return out


def pose_gap(a, b):
    pa, pb = a.get("pose_before"), b.get("pose_before")
    if not pa or not pb:
        return None
    return {"dx": abs(pa["x"] - pb["x"]), "dy": abs(pa["y"] - pb["y"]), "dz": abs(pa["z"] - pb["z"]),
            "dyaw_rad": yaw_diff(a["yaw_before"], b["yaw_before"]) if a.get("yaw_before") is not None and b.get("yaw_before") is not None else None}


def compare_pair(ref, arm):
    """ref、arm：actions.jsonl 的行列表。"""
    n = min(len(ref), len(arm))
    ea = [r["action_executed"] for r in ref]
    eb = [r["action_executed"] for r in arm]
    first, reason = None, None
    for i in range(n):
        if ea[i] != eb[i]:
            first, reason = i, "action"
            break
    if first is None and len(ref) != len(arm):
        first, reason = n, "length"
    prefix = n if first is None else min(first, n)
    gaps = [pose_gap(ref[i], arm[i]) for i in range(prefix)]
    gaps = [g for g in gaps if g]
    res = {
        "identical": first is None, "steps_ref": len(ref), "steps_arm": len(arm),
        "first_divergent_step": None if first is None else first, "divergence_reason": reason,
        "timestamps_equal_in_prefix": all(ref[i]["timestamp_ns"] == arm[i]["timestamp_ns"] for i in range(prefix)),
        "prefix_steps": prefix,
        "pose_gap_prefix_max": ({k: max((g[k] or 0.0) for g in gaps) for k in ("dx", "dy", "dz", "dyaw_rad")} if gaps else None),
    }
    if first is not None:
        def side(rows, i):
            return {"at": step_view(rows[i]) if i < len(rows) else None,
                    "before": step_view(rows[i - 1]) if 0 < i <= len(rows) else None}
        res["ref"], res["arm"] = side(ref, first), side(arm, first)
        res["pose_gap_at_divergence"] = (pose_gap(ref[first], arm[first]) if first < len(ref) and first < len(arm) else None)
    return res


def lat_stats(a):
    if not len(a):
        return None
    a = np.asarray(a, dtype=np.float64)
    return {"n": int(len(a)), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "max": float(a.max()), "mean": float(a.mean())}


def arm_stats(rows_by_seed):
    infer, lag, late = [], [], 0
    for rows in rows_by_seed.values():
        for r in rows:
            if r.get("infer_ms") is not None:
                infer.append(r["infer_ms"])
            if r.get("dispatch_lag_s") is not None:
                lag.append(r["dispatch_lag_s"])
            late += 1 if r.get("late_dispatch") else 0
    return {"steps": int(sum(len(v) for v in rows_by_seed.values())), "infer_ms": lat_stats(infer),
            "dispatch_lag_s": lat_stats(lag), "late_dispatch_steps": int(late)}


def run(ref_dir, arms, seeds):
    ref_rows = {s: read_rows(episode_path(ref_dir, s)) for s in seeds}
    ref_eval = load_eval(ref_dir)
    out = {"ref": {"dir": ref_dir, "stats": arm_stats(ref_rows),
                   "outcomes": {s: per_episode_outcome(ref_eval, s) for s in seeds}}, "arms": {}}
    for name, arm_dir in arms.items():
        rows, per_seed, used, outcomes = {}, {}, {}, {}
        for s in seeds:
            d, retried, first_dir = resolve(arm_dir, s)
            if not os.path.isfile(episode_path(d, s)):
                per_seed[s] = {"missing": True}
                continue
            rows[s] = read_rows(episode_path(d, s))
            used[s] = {"dir": d, "from_retry": retried}
            per_seed[s] = compare_pair(ref_rows[s], rows[s])
            if retried and first_dir:
                per_seed[s]["first_attempt_late_steps"] = sum(1 for r in read_rows(episode_path(first_dir, s)) if r.get("late_dispatch"))
            outcomes[s] = per_episode_outcome(load_eval(d), s)
        out["arms"][name] = {"dir": arm_dir, "used": used, "stats": arm_stats(rows), "per_seed": per_seed, "outcomes": outcomes,
                             "all_identical": bool(per_seed) and all(v.get("identical") for v in per_seed.values())}
    return out


def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_md(res, seeds):
    L = ["# 闭环动作序列对照（逐步 action_executed）", "", f"参照臂 {res['ref']['dir']}", ""]
    for name, a in res["arms"].items():
        L += [f"## 臂 {name}（{a['dir']}）", "", "| seed | 步数 参照/本臂 | 序列一致 | 第一处分歧 | 分歧步动作 参照/本臂 | dispatch_lag_s 参照/本臂 | late 参照/本臂 | infer_ms 参照/本臂 | 分歧步位姿差（m, rad） | 结局 参照/本臂 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in seeds:
            v = a["per_seed"].get(s) or {"missing": True}
            if v.get("missing"):
                L.append(f"| {s} | - | 缺 | - | - | - | - | - | - | - |")
                continue
            ro, ao = res["ref"]["outcomes"].get(s), a["outcomes"].get(s)
            out = f"{'成功' if ro and ro['success'] else '未成功'} / {'成功' if ao and ao['success'] else '未成功'}"
            if v["identical"]:
                L.append(f"| {s} | {v['steps_ref']}/{v['steps_arm']} | 是 | - | - | - | - | - | 前缀最大 {fmt(max((v['pose_gap_prefix_max'] or {}).get('dx', 0.0), (v['pose_gap_prefix_max'] or {}).get('dy', 0.0)), 4)} | {out} |")
                continue
            r, b = v["ref"]["at"], v["arm"]["at"]
            g = v.get("pose_gap_at_divergence") or {}
            gap = f"{fmt(max(g.get('dx', 0.0), g.get('dy', 0.0)), 4)}, {fmt(g.get('dyaw_rad'), 4)}" if g else "-"
            L.append(f"| {s} | {v['steps_ref']}/{v['steps_arm']} | 否（{v['divergence_reason']}） | {v['first_divergent_step']} | "
                     f"{fmt(r and r['action_executed'])}/{fmt(b and b['action_executed'])} | {fmt(r and r['dispatch_lag_s'])}/{fmt(b and b['dispatch_lag_s'])} | "
                     f"{fmt(r and r['late_dispatch'])}/{fmt(b and b['late_dispatch'])} | {fmt(r and r['infer_ms'])}/{fmt(b and b['infer_ms'])} | {gap} | {out} |")
        L.append("")
        st, rs = a["stats"], res["ref"]["stats"]
        L += ["| 统计（该 3 集全部步） | 参照 | 本臂 |", "|---|---|---|",
              f"| 步数 | {rs['steps']} | {st['steps']} |",
              f"| infer_ms p50 / p95 / max | {fmt(rs['infer_ms']['p50'])} / {fmt(rs['infer_ms']['p95'])} / {fmt(rs['infer_ms']['max'])} | "
              f"{fmt(st['infer_ms'] and st['infer_ms']['p50'])} / {fmt(st['infer_ms'] and st['infer_ms']['p95'])} / {fmt(st['infer_ms'] and st['infer_ms']['max'])} |",
              f"| dispatch_lag_s 最大 | {fmt(rs['dispatch_lag_s']['max'], 4)} | {fmt(st['dispatch_lag_s'] and st['dispatch_lag_s']['max'], 4)} |",
              f"| 迟到步 | {rs['late_dispatch_steps']} | {st['late_dispatch_steps']} |", ""]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--arm", action="append", required=True, help="名字=目录，可重复")
    ap.add_argument("--seeds", required=True, type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--out", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)
    arms = {}
    for a in args.arm:
        name, _, d = a.partition("=")
        if not name or not d:
            raise SystemExit(f"--arm 要写成 名字=目录：{a!r}")
        arms[name] = d
    res = run(args.ref, arms, args.seeds)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
            f.write("\n")
    md = render_md(res, args.seeds)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
