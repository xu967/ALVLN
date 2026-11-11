#!/usr/bin/env python3
"""演练用的假评测入口（不起 Gazebo）：认 eval_closed_loop.py 的命令行，按真实格式写 eval.json 与 episodes/seed_N，
用来演练 run.sh 的流程（阶段 1 → 标定门 → L 阶梯 → 阶段 2 → summarize.py → DONE）。run.sh 经 EVAL=本文件 接入，见 rehearse.sh。

假数据的规则（环境变量可改）：
  每步观测戳到派发 = --action-latency L（没给则 0.13 s）；窗口末 = 观测戳 + 1.005 s，所以每步生效 1.005 − lag。
  STUB_K3_LATE_AT（默认 0.16）：逗号分隔的 L 列表；K=3（--task course_base）在这些 L 上第 0 步就绪时刻 = L + 0.035 s，超过 L + 0.02 就记迟到。
  STUB_K3_FWD_SCALE（默认 1.0）：K=3 稳态前飞位移相对 K=1 的比例（试 2% 门用）。
  成功判定：学生臂 K=1 成功率约 0.8，K=3 第 1 块通过率约 STUB_K3_PASS（默认 0.75）；由 (seed, 训练种子) 的哈希决定，可复现。
"""
import argparse
import hashlib
import json
import os
import sys


def parse_seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition(":")
        out += list(range(int(a), int(b))) if b else [int(a)]
    return out


def h(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def main():
    p = argparse.ArgumentParser()
    for name in ("--policy", "--task", "--ckpt", "--infer-bin", "--split", "--seeds", "--density", "--dynamics", "--vehicle",
                 "--window-from", "--out"):
        p.add_argument(name)
    for name in ("--steps", "--blocks", "--episodes", "--heading-guard"):
        p.add_argument(name, type=int, default=0)
    for name in ("--action-latency", "--corridor-half-width", "--sim-timeout", "--episode-timeout", "--hold"):
        p.add_argument(name, type=float, default=None)
    a = p.parse_args()
    k3 = a.task == "course_base"
    L = a.action_latency
    late_at = {float(x) for x in os.environ.get("STUB_K3_LATE_AT", "0.16").split(",") if x}
    fwd_scale = float(os.environ.get("STUB_K3_FWD_SCALE", "1.0"))
    pass3 = float(os.environ.get("STUB_K3_PASS", "0.75"))
    seeds = parse_seeds(a.seeds)[:a.episodes]
    steps_n = 80 if k3 else (20 if a.task == "forest_e1" else 30)
    per_episode, failed, late_eps = [], [], []
    train_seed = int(os.path.basename(a.ckpt or "x_s0").split("_s")[-1][0]) if a.ckpt else -1
    for seed in seeds:
        ep = os.path.join(a.out, "episodes", f"seed_{seed}")
        os.makedirs(ep, exist_ok=True)
        actions, oracle, late_ids = [], [], []
        for i in range(steps_n):
            yaw_step = i % 5 == 2
            lag = 0.13 if L is None else L
            if L is not None and k3 and i == 0 and any(abs(L - x) < 1e-9 for x in late_at):
                lag = L + 0.035        # 比阈值 L + 0.02 s 晚 15 ms；L=0.16 时 = 0.195 s，与冒烟里 K=3 第 0 步一致
            late = None if L is None else lag > L + 0.02
            if late:
                late_ids.append(i)
            start = 10_000_000_000 + i * 1_000_000_000
            row = {"step": i, "action_executed": 2 if yaw_step else 1, "action_expert": 1, "collision": False, "infer_ms": 1.0,
                   "displacement_m": 0.04 if yaw_step else 0.25 * (fwd_scale if k3 else 1.0), "dyaw_deg": 26.0 if yaw_step else 0.1,
                   "dispatch_lag_s": lag, "src": None, "action_start_ns": start}
            if late is not None:
                row["late_dispatch"] = late
            actions.append(row)
            oracle.append({"step": i, "action_start_ns": start, "action_end_ns": start + 1_005_000_000,
                           "pose_gt": {"x": 0.2 * (i + 1), "y": 0.0, "z": 2.0}, "speed_mps": 0.3})
        with open(os.path.join(ep, "actions.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in actions))
        with open(os.path.join(ep, "oracle.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in oracle))
        sim_s = sum(o["action_end_ns"] - o["action_start_ns"] for o in oracle) / 1e9 - sum(r["dispatch_lag_s"] for r in actions)
        with open(os.path.join(ep, "episode_manifest.json"), "w") as f:
            json.dump({"goal_line_x": 16.0 if k3 else 6.0, "performance": {"steps": steps_n, "action_sim_s": sim_s, "action_rtf": 0.8},
                       **({"action_latency_s": L, "late_dispatch_steps": len(late_ids)} if L is not None else {})}, f)
        lags = [r["dispatch_lag_s"] for r in actions]
        rate = pass3 if k3 else 0.8
        ok = h(seed, train_seed, "k3" if k3 else "k1") < rate
        entry = {"seed": seed, "success": ok, "steps": steps_n, "episode_dir": os.path.join("episodes", f"seed_{seed}"),
                 "pillars": [{}] * (26 if k3 else 8), "goal_x": 16.0 if k3 else 6.0,
                 "dispatch_lag_s_median": sorted(lags)[len(lags) // 2], "dispatch_lag_s_max": max(lags),
                 **({"block_goal_x": [6.0, 11.0, 16.0]} if k3 else {})}
        if L is not None:
            entry["late_dispatch_steps"] = len(late_ids)
        if late_ids:
            failed.append({"seed": seed, "rc": 0, "error": "late_dispatch: stub", "late_dispatch_steps": len(late_ids),
                           "late_step_ids": late_ids})
            late_eps.append({"seed": seed, "late_dispatch_steps": len(late_ids), "late_step_ids": late_ids,
                             "dispatch_lag_s_median": entry["dispatch_lag_s_median"], "dispatch_lag_s_max": entry["dispatch_lag_s_max"]})
        else:
            per_episode.append(entry)
    report = {"episodes_requested": a.episodes, "episodes_done": len(per_episode), "failed_count": len(failed), "failed": failed,
              "partial": False, "skipped_count": 0, "per_episode": per_episode, "corridor_half_width": a.corridor_half_width or 3.0,
              "metric_version": "sim09.v2.coast1s.corridor3m", "blocks": a.blocks or None,
              **({"action_latency_s": L, "late_dispatch_episodes": late_eps} if L is not None else {})}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(report, f)
    print(f"requested: stub done={len(per_episode)} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
