#!/usr/bin/env python3
"""演练用的假评测入口（卡 E2E-1，不起 Gazebo）：认 eval_closed_loop.py 的命令行（含 --student-cond），按真实格式写 eval.json 与 episodes/seed_N，
用来演练 run.sh 的每个分支（起跑条件、种子优先的顺序、迟到集重跑、限流标记、续跑）与 summarize_e2e.py。run.sh 经 EVAL=本文件 接入。

臂由 --ckpt 的文件名定：cal_swa_<臂名>_s<种子>_k10.bin（A1–A3）或 cal_swa_s<种子>_k10.bin（A0）。每个臂的成功率与失败构成由环境变量定，可复现（seed、臂、训练种子的哈希）。
环境变量：
  STUB_P            四个臂的成功率，逗号分隔，顺序 a0,a1,a2,a3；缺省 0.70,0.80,0.90,0.50。
  STUB_FAIL_MIX     失败集的构成（撞柱,越界,未到线），缺省 0.4,0.4,0.2。
  STUB_LATE         JSON：{"<评测臂目录名>": [迟到的 seed…]}，这些 seed 首次运行第 0 步迟到（基础设施失败）。
  STUB_RETRY_LATE   1 = 名字以 -retry1 结尾的臂里这些 seed 仍迟到。
  STUB_SLEEP        每臂写文件前睡几秒（缺省 0.3），让并行臂的区间重叠。
  STUB_THROTTLE     1 = 每个臂结束时 $CPU_STAT_FILE 的 nr_throttled 加 5；usage_usec 总是加 2.5 s。
  STUB_ORDER_LOG    非空时把「臂目录名 开始时刻」追加到这个文件，演练脚本据此核对种子优先的顺序。
  CPU_STAT_FILE     run.sh 导出的同名变量；两条车道同时写，用 flock 互斥。
每个臂目录里写 stub_args.txt（收到的命令行），演练脚本据此核对 --student-cond 只在 A1、A2 上出现、任何臂都没有 --heading-guard。
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "sim", "tools"))
import nav_cond  # noqa: E402

HOLD_NS = 1_005_000_000
ARM_OF = {"a0-img-demo": "a0", "a1-code-demo": "a1", "a2-code-dagger": "a2", "a3-img-dagger": "a3"}


def h(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def parse_seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition(":")
        out += list(range(int(a), int(b))) if b else [int(a)]
    return out


def arm_of_ckpt(path):
    """cal_swa_<臂名>_s<种子>_k10.bin → (臂, 种子)；cal_swa_s<种子>_k10.bin → ("a0", 种子)。"""
    base = os.path.basename(path or "")
    stem = base[len("cal_swa_"):-len("_k10.bin")] if base.startswith("cal_swa_") and base.endswith("_k10.bin") else ""
    name, _, seed = stem.rpartition("_s")
    if not name:           # cal_swa_s0_k10.bin：stem = "s0"
        return "a0", int(stem[1:]) if stem.startswith("s") and stem[1:].isdigit() else -1
    return ARM_OF.get(name, "?"), int(seed)


def bump_cpu_stat(throttle):
    path = os.environ.get("CPU_STAT_FILE")
    if not path or not os.path.exists(path):
        return
    with open(path + ".lk", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path) as f:
            kv = dict(line.split() for line in f if line.strip())
        kv["usage_usec"] = str(int(kv.get("usage_usec", 0)) + 2_500_000)
        if throttle:
            kv["nr_throttled"] = str(int(kv.get("nr_throttled", 0)) + 5)
            kv["throttled_usec"] = str(int(kv.get("throttled_usec", 0)) + 1_000_000)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join(f"{k} {v}\n" for k, v in kv.items()))
        os.replace(tmp, path)


def make_episode(seed, arm, train_seed, ok, kind, steps_n, code_arm):
    """假轨迹：成功 = x 到 6.5；撞柱 = 第 steps_n//2 步碰撞；越界 = y 在中途越过 3 m；未到线 = x 停在 2.0。动作由哈希定，码臂在码 ≠ 0 时多回转。"""
    executed = []
    for i in range(steps_n):
        codes = nav_cond.heading_codes(executed + [1])
        code = codes[-1]
        r = h(seed, arm, train_seed, i)
        if code_arm and code == 2 and r < 0.8:
            a = 3
        elif code_arm and code == 3 and r < 0.8:
            a = 2
        elif r < 0.12:
            a = 2
        elif r < 0.22:
            a = 3
        else:
            a = 1
        executed.append(a)
    return executed


def main():
    p = argparse.ArgumentParser()
    for name in ("--policy", "--task", "--ckpt", "--infer-bin", "--split", "--seeds", "--dynamics", "--vehicle", "--window-from", "--out", "--student-cond"):
        p.add_argument(name)
    for name in ("--steps", "--episodes", "--heading-guard", "--blocks"):
        p.add_argument(name, type=int, default=0)
    for name in ("--action-latency", "--corridor-half-width", "--sim-timeout", "--episode-timeout", "--hold", "--rtf"):
        p.add_argument(name, type=float, default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "stub_args.txt"), "w") as f:
        f.write(" ".join(sys.argv[1:]) + "\n")
    name_dir = os.path.basename(os.path.normpath(a.out))
    order_log = os.environ.get("STUB_ORDER_LOG")
    if order_log:
        with open(order_log, "a") as f:
            f.write(f"{name_dir} {time.time():.3f}\n")
    time.sleep(float(os.environ.get("STUB_SLEEP", "0.3")))

    arm, train_seed = arm_of_ckpt(a.ckpt)
    probs = dict(zip(("a0", "a1", "a2", "a3"), (float(x) for x in os.environ.get("STUB_P", "0.70,0.80,0.90,0.50").split(","))))
    mix = [float(x) for x in os.environ.get("STUB_FAIL_MIX", "0.4,0.4,0.2").split(",")]
    late_cfg = json.loads(os.environ.get("STUB_LATE", "{}"))
    is_retry = name_dir.endswith("-retry1")
    late_seeds = {int(x) for x in late_cfg.get(name_dir.replace("-retry1", ""), [])}
    code_arm = a.student_cond == "heading-code"
    L = a.action_latency
    seeds = parse_seeds(a.seeds)[:a.episodes]
    steps_n = 30
    per_episode, failed, late_eps = [], [], []
    for seed in seeds:
        ep = os.path.join(a.out, "episodes", f"seed_{seed}")
        os.makedirs(ep, exist_ok=True)
        late_here = L is not None and seed in late_seeds and (not is_retry or os.environ.get("STUB_RETRY_LATE") == "1")
        ok = h(seed, arm, train_seed, "ok") < probs.get(arm, 0.5)
        r = h(seed, arm, train_seed, "kind")
        kind = "collision" if r < mix[0] else ("corridor" if r < mix[0] + mix[1] else "not_reached")
        executed = make_episode(seed, arm, train_seed, ok, kind, steps_n, code_arm)
        codes = nav_cond.heading_codes(executed)
        actions, oracle, late_ids = [], [], []
        for i in range(steps_n):
            lag = 0.2 if L is None else L
            if late_here and i == 0:
                lag = L + 0.035
            late = None if L is None else lag > L + 0.02 + 1e-9
            if late:
                late_ids.append(i)
            start = 10_000_000_000 + i * 1_000_000_000
            collide = (not ok) and kind == "collision" and i == steps_n // 2
            row = {"step": i, "action_executed": executed[i], "action_expert": 1, "collision": collide, "infer_ms": 1.0,
                   "displacement_m": 0.04 if executed[i] in (2, 3) else 0.23, "dyaw_deg": 26.0 if executed[i] in (2, 3) else 0.1,
                   "dispatch_lag_s": lag, "src": None, "action_start_ns": start, **({} if late is None else {"late_dispatch": late})}
            if code_arm:
                row["cond_code"] = codes[i]
            actions.append(row)
            frac = (i + 1) / steps_n
            x = 6.5 * frac if ok else (2.0 * frac if kind == "not_reached" else 3.0 * frac)
            y = 0.0
            if (not ok) and kind == "corridor" and i >= steps_n // 2:
                y = 3.5
            oracle.append({"step": i, "action_start_ns": start, "action_end_ns": start + HOLD_NS, "pose_gt": {"x": x, "y": y, "z": 2.0}, "speed_mps": 0.3})
            if collide:
                break
        with open(os.path.join(ep, "actions.jsonl"), "w") as f:
            f.write("".join(json.dumps(r_) + "\n" for r_ in actions))
        with open(os.path.join(ep, "oracle.jsonl"), "w") as f:
            f.write("".join(json.dumps(r_) + "\n" for r_ in oracle))
        sim_s = sum(o["action_end_ns"] - o["action_start_ns"] for o in oracle) / 1e9 - sum(r_["dispatch_lag_s"] for r_ in actions)
        with open(os.path.join(ep, "episode_manifest.json"), "w") as f:
            json.dump({"goal_line_x": 6.0, "performance": {"steps": len(actions), "action_sim_s": sim_s, "action_rtf": 0.8},
                       **({"action_latency_s": L, "late_dispatch_steps": len(late_ids)} if L is not None else {})}, f)
        lags = [r_["dispatch_lag_s"] for r_ in actions]
        entry = {"seed": seed, "success": bool(ok), "steps": len(actions), "episode_dir": os.path.join("episodes", f"seed_{seed}"), "pillars": 8, "goal_x": 6.0,
                 "collision": (not ok) and kind == "collision", "corridor_violation": (not ok) and kind == "corridor",
                 "dispatch_lag_s_median": sorted(lags)[len(lags) // 2], "dispatch_lag_s_max": max(lags)}
        if L is not None:
            entry["late_dispatch_steps"] = len(late_ids)
        if late_ids:
            failed.append({"seed": seed, "rc": 0, "error": "late_dispatch: stub", "late_dispatch_steps": len(late_ids), "late_step_ids": late_ids})
            late_eps.append({"seed": seed, "late_dispatch_steps": len(late_ids), "late_step_ids": late_ids,
                             "dispatch_lag_s_median": entry["dispatch_lag_s_median"], "dispatch_lag_s_max": entry["dispatch_lag_s_max"]})
        else:
            per_episode.append(entry)
    report = {"episodes_requested": a.episodes, "episodes_done": len(per_episode), "failed_count": len(failed), "failed": failed,
              "partial": False, "skipped_count": 0, "per_episode": per_episode, "corridor_half_width": a.corridor_half_width or 3.0,
              "metric_version": "sim09.v2.coast1s.corridor3m", "blocks": None, "heading_guard": a.heading_guard or None,
              **({"action_latency_s": L, "late_dispatch_episodes": late_eps} if L is not None else {}),
              **({"student_cond": "heading-code", "student_cond_check": {"status": "ok"}} if code_arm else {})}
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(report, f)
    bump_cpu_stat(os.environ.get("STUB_THROTTLE") == "1")
    print(f"requested: stub done={len(per_episode)} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
