#!/usr/bin/env python3
"""演练用的评测桩：接 run_closed_loop.sh 里 eval_closed_loop.py 的命令行（用 RS_EVAL 指过来），不起 Gazebo，按权重文件内容与 seed 确定性地编一份 eval.json。

只用于演练 run_closed_loop.sh 的流程（锁、冻结、排队、迟到集重跑、汇总），产出的数字没有任何含义，演练目录里的 summary 不是结果。
行为：成功概率随权重文件内容的哈希在 0.45–0.85 间变；首次运行（--out 不以 -retry1 结尾）里按哈希让约 1/12 的集「迟到」，重跑时不再迟到；
环境变量 STUB_FAIL_ARM=<臂目录名里含的子串> 让匹配的臂整体失败（rc=3、不写 eval.json），用来演练 DONE rc=11；STUB_SLEEP=<秒> 每集睡多久（缺省 0.05）。
"""
import argparse
import hashlib
import json
import os
import sys
import time


def h(*parts):
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy")
    ap.add_argument("--ckpt")
    ap.add_argument("--infer-bin")
    ap.add_argument("--task")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--split")
    ap.add_argument("--seeds")
    ap.add_argument("--corridor-half-width", type=float)
    ap.add_argument("--action-latency", type=float)
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--sim-timeout", type=float)
    ap.add_argument("--episode-timeout", type=float)
    ap.add_argument("--dynamics")
    ap.add_argument("--vehicle")
    ap.add_argument("--hold", type=float)
    ap.add_argument("--window-from")
    ap.add_argument("--heading-guard", type=int, default=0)
    ap.add_argument("--rtf", type=float)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.heading_guard or a.rtf is not None:
        print("stub: 新协议不带 --heading-guard / --rtf", file=sys.stderr)
        return 2
    if a.split != "val" or a.task != "base" or a.action_latency is None:
        print(f"stub: 参数不是第二部分的口径（split={a.split} task={a.task} action_latency={a.action_latency}）", file=sys.stderr)
        return 2
    bad = os.environ.get("STUB_FAIL_ARM")
    if bad and bad in a.out:
        print("stub: 整臂失败", file=sys.stderr)
        return 3
    os.makedirs(a.out, exist_ok=True)
    with open(a.ckpt, "rb") as f:
        ck = hashlib.sha256(f.read()).hexdigest()
    p_ok = 0.45 + 0.40 * (h(ck) % 1000) / 1000.0
    retry = a.out.rstrip("/").endswith("-retry1")
    seeds = [int(x) for x in a.seeds.split(",") if x]
    rows, failed = [], []
    for s in seeds[: a.episodes]:
        time.sleep(float(os.environ.get("STUB_SLEEP", "0.05")))
        if not retry and h("late", ck, s) % 12 == 0:
            failed.append({"seed": s, "rc": 0, "error": "late_dispatch: 1 步迟到", "late_dispatch_steps": 1, "late_step_ids": [3],
                           "dispatch_lag_s_median": 0.2, "dispatch_lag_s_max": 0.31})
            continue
        ok = (h("ok", ck, s) % 1000) / 1000.0 < p_ok
        k = h("kind", ck, s) % 3
        fwd = 60 + h("fwd", ck, s) % 15
        rows.append({"seed": s, "geometry_gid": f"{h('gid', s):016x}", "success": ok, "collision": (not ok) and k == 0,
                     "corridor_violation": (not ok) and k == 1, "steps": 80 if (not ok and k == 2) else 30,
                     "actions": {"1": fwd, "2": 5, "3": 4}, "collect_rc": 0, "late_dispatch_steps": 0, "dedup_key": f"{h('gid', s):016x}"})
    done, nfail = len(rows), len(failed)
    rep = {"schema": "uavrgb.eval.v1", "policy": a.policy, "task": a.task, "split": a.split, "episodes_requested": a.episodes,
           "episodes_done": done, "failed_count": nfail, "skipped_count": 0, "partial": False, "action_latency_s": a.action_latency,
           "metric_version": "sim09.v2.coast1s.corridor3m", "ckpt": os.path.abspath(a.ckpt), "ckpt_sha256": ck, "wall_s": 1.0,
           "late_dispatch_episodes": [{k: f.get(k) for k in ("seed", "late_dispatch_steps", "late_step_ids", "dispatch_lag_s_median", "dispatch_lag_s_max")} for f in failed],
           "requested_success_rate": sum(r["success"] for r in rows) / a.episodes, "failed": failed, "skipped": [], "per_episode": rows}
    with open(os.path.join(a.out, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f)
    print(f"requested: SR={rep['requested_success_rate']} ({sum(r['success'] for r in rows)}/{a.episodes}); failed={nfail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
