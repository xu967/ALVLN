#!/usr/bin/env python3
"""演练用的假评测入口（第二轮，不起 Gazebo）：认 eval_closed_loop.py 的命令行（含 --rtf），按真实格式写 eval.json 与 episodes/seed_N，
用来演练 run_r2.sh 的每个分支（阶梯 A/B/C、只因位移停、迟到集重跑、限流标记、续跑…）。run_r2.sh 经 EVAL=本文件 接入，见 rehearse_r2.py。
第一轮的 stub_eval.py 不动；本文件是第二轮的版本。

每步的假数据：
  观测戳到派发 lag = --action-latency L（没给则 0.13 s）；窗口末 = 观测戳 + 1.005 s，所以每步生效 1.005 − lag。
  动作序列：每 RUN_LEN 步一次 YAW_LEFT，其余 FORWARD_SLOW；前飞位移由「连续前进第几步 r」定：r=1 0.08，r=2 0.40，r=3 0.28，r≥4 0.23 m（巡航）；YAW 步转角 26°。
  RUN_LEN：K=1 与森林 10；K=3 缺省 10，STUB_K3_RUN_LEN=20 让 K=3 的直线更长（层内位移不变，只改构成，用来演练两种位移口径）。
环境变量：
  STUB_K3_LATE_AT  逗号分隔的配置记号（L 取两位小数，带 --rtf 时接 rtf<值>：0.20、0.20rtf0.5、0.24）：K=3 教师臂在这些配置上第 0 步迟到（就绪 = L + 0.035 s）。缺省 0.20。
  STUB_K1_LATE_AT  同上，K=1 教师臂。缺省空。
  STUB_K3_FWD_AT   K=3 教师臂在这些配置上巡航位移乘 STUB_K3_FWD_SCALE（缺省 0.95）。缺省空。
  STUB_STUDENT_LATE_SEEDS  逗号分隔的 seed：学生臂里这些 seed 的第 0 步迟到；STUB_STUDENT_LATE_ARM 非空时只在这个臂名里生效。
  STUB_RETRY_LATE=1  名字以 -retry1 结尾的臂里这些 seed 仍迟到。
  STUB_FOREST_LATE  森林密度名（如 dense）：该档第 0 步迟到。
  STUB_K3_PASS  学生 K=3 第 1 块通过率（缺省 0.8）；K=1 成功率 0.85；由 (seed, 训练种子, K) 的哈希定，可复现。
  STUB_SLEEP  每臂写文件前睡几秒（缺省 0.3），让并行臂的区间重叠。
  STUB_THROTTLE=1  每个臂结束时 $CPU_STAT_FILE 的 nr_throttled 加 5、throttled_usec 加 1 s（模拟被限流）；usage_usec 总是加 2.5 s。
  CPU_STAT_FILE  run_r2.sh 导出的同名变量；两条车道同时写，用 flock 互斥。
每个臂目录里写 stub_args.txt（收到的命令行），演练脚本据此核对「学生臂与森林臂没有 --rtf、B 组 K=3 有 --rtf 0.5」。
"""
import argparse
import fcntl
import hashlib
import json
import os
import sys
import time

STRATA = {1: 0.08, 2: 0.40, 3: 0.28}
CRUISE = 0.23
HOLD_NS = 1_005_000_000


def parse_seeds(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition(":")
        out += list(range(int(a), int(b))) if b else [int(a)]
    return out


def h(*xs):
    return int(hashlib.sha256("|".join(map(str, xs)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def csv_env(name, default=""):
    return {x for x in os.environ.get(name, default).split(",") if x}


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


def main():
    p = argparse.ArgumentParser()
    for name in ("--policy", "--task", "--ckpt", "--infer-bin", "--split", "--seeds", "--density", "--dynamics", "--vehicle", "--window-from", "--out"):
        p.add_argument(name)
    for name in ("--steps", "--blocks", "--episodes", "--heading-guard"):
        p.add_argument(name, type=int, default=0)
    for name in ("--action-latency", "--corridor-half-width", "--sim-timeout", "--episode-timeout", "--hold", "--rtf"):
        p.add_argument(name, type=float, default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "stub_args.txt"), "w") as f:
        f.write(" ".join(sys.argv[1:]) + "\n")
    time.sleep(float(os.environ.get("STUB_SLEEP", "0.3")))

    k3 = a.task == "course_base"
    forest = a.task == "forest_e1"
    student = a.policy == "student"
    L = a.action_latency
    token = "none" if L is None else f"{L:.2f}" + (f"rtf{a.rtf:g}" if a.rtf else "")
    arm = os.path.basename(os.path.normpath(a.out))
    k3_late = token in csv_env("STUB_K3_LATE_AT", "0.20")
    k1_late = token in csv_env("STUB_K1_LATE_AT")
    k3_fwd = token in csv_env("STUB_K3_FWD_AT")
    fwd_scale = float(os.environ.get("STUB_K3_FWD_SCALE", "0.95"))
    run_len = int(os.environ.get("STUB_K3_RUN_LEN", "10")) if k3 else 10
    pass3 = float(os.environ.get("STUB_K3_PASS", "0.8"))
    late_seeds = {int(x) for x in csv_env("STUB_STUDENT_LATE_SEEDS")}
    arm_filter = os.environ.get("STUB_STUDENT_LATE_ARM", "")
    is_retry = arm.endswith("-retry1")
    forest_late = os.environ.get("STUB_FOREST_LATE", "")
    seeds = parse_seeds(a.seeds)[:a.episodes]
    steps_n = 80 if k3 else (20 if forest else 30)
    train_seed = int(os.path.basename(a.ckpt or "x_s0").split("_s")[-1][0]) if a.ckpt else -1
    cruise = CRUISE * (fwd_scale if (k3 and not student and k3_fwd) else 1.0)
    per_episode, failed, late_eps = [], [], []
    for seed in seeds:
        ep = os.path.join(a.out, "episodes", f"seed_{seed}")
        os.makedirs(ep, exist_ok=True)
        late_here = False
        if L is not None:
            if student:
                late_here = seed in late_seeds and (not arm_filter or arm_filter == arm.replace("-retry1", "")) and (not is_retry or os.environ.get("STUB_RETRY_LATE") == "1")
            elif forest:
                late_here = forest_late == a.density
            else:
                late_here = (k3 and k3_late) or ((not k3) and k1_late)
        ok = (h(seed, train_seed, "k3") < pass3) if k3 else (h(seed, train_seed, "k1") < 0.85)
        x_end = 6.5 if ok else 2.0
        actions, oracle, late_ids = [], [], []
        for i in range(steps_n):
            j = i % run_len
            yaw = j == 0
            lag = 0.13 if L is None else L
            if L is not None and i == 0 and late_here:
                lag = L + 0.035          # 比阈值 L + 0.02 s 晚 15 ms
            late = None if L is None else lag > L + 0.02 + 1e-9
            if late:
                late_ids.append(i)
            start = 10_000_000_000 + i * 1_000_000_000
            actions.append({"step": i, "action_executed": 2 if yaw else 1, "action_expert": 1, "collision": False, "infer_ms": 1.0,
                            "displacement_m": 0.04 if yaw else STRATA.get(j, cruise), "dyaw_deg": 26.0 if yaw else 0.1,
                            "dispatch_lag_s": lag, "src": None, "action_start_ns": start, **({} if late is None else {"late_dispatch": late})})
            oracle.append({"step": i, "action_start_ns": start, "action_end_ns": start + HOLD_NS,
                           "pose_gt": {"x": x_end * (i + 1) / steps_n, "y": 0.0, "z": 2.0}, "speed_mps": 0.3})
        with open(os.path.join(ep, "actions.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in actions))
        with open(os.path.join(ep, "oracle.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in oracle))
        sim_s = sum(o["action_end_ns"] - o["action_start_ns"] for o in oracle) / 1e9 - sum(r["dispatch_lag_s"] for r in actions)
        with open(os.path.join(ep, "episode_manifest.json"), "w") as f:
            json.dump({"goal_line_x": 16.0 if k3 else 6.0, "performance": {"steps": steps_n, "action_sim_s": sim_s, "action_rtf": 0.8},
                       **({"action_latency_s": L, "late_dispatch_steps": len(late_ids)} if L is not None else {})}, f)
        lags = [r["dispatch_lag_s"] for r in actions]
        entry = {"seed": seed, "success": False if k3 else bool(ok), "steps": steps_n, "episode_dir": os.path.join("episodes", f"seed_{seed}"),
                 "pillars": 26 if k3 else 8, "goal_x": 16.0 if k3 else 6.0,
                 "dispatch_lag_s_median": sorted(lags)[len(lags) // 2], "dispatch_lag_s_max": max(lags),
                 **({"block_goal_x": [6.0, 11.0, 16.0]} if k3 else {})}
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
              "metric_version": "sim09.v2.coast1s.corridor3m", "blocks": a.blocks or None,
              **({"action_latency_s": L, "late_dispatch_episodes": late_eps} if L is not None else {}),
              **({"rtf": a.rtf} if a.rtf is not None else {})}
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(report, f)
    bump_cpu_stat(os.environ.get("STUB_THROTTLE") == "1")
    print(f"requested: stub done={len(per_episode)} failed={len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
