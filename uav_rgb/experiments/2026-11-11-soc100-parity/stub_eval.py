#!/usr/bin/env python3
"""演练用的假评测入口（不起 Gazebo）：认 eval_closed_loop.py 的命令行（含 --cf-sitl-params），按真实格式写 eval.json 与 episodes/seed_N，
用来演练 run.sh 与 summarize_parity.py 的每个分支。run.sh 经 EVAL=本文件 接入，见 rehearse.py。
第二轮的 stub_eval_r2.py 不动；本文件按它改：加 --cf-sitl-params，逐集 manifest 写 dynamics_profile / params_sha256，第 0 步写 timestamp_ns，
K=3 按档位生成逐块结果，eval.json 写第二轮 eval.json 里有的协议键。没给 --cf-sitl-params 的臂 = 对照（50% 电量），给了 = 满电。

每步的假数据（与第二轮的桩一致）：
  观测戳到派发 lag = --action-latency L（没给则 0.13 s）；每步生效 1.005 − lag。
  动作序列：每 10 步一次 YAW_LEFT，其余 FORWARD_SLOW；前飞位移由连续前进第几步 r 定：r=1 0.08，r=2 0.40，r=3 0.28，r≥4 0.23 m（巡航）；YAW 步转角 26°。
结果的生成：K=1 成功率 0.85；K=3 通过第 1/2/3 块的概率 0.8/0.6/0.4；都由 (seed, 训练种子, K) 的哈希定，可复现；对照与满电缺省逐集相同。
环境变量（满电臂指有 --cf-sitl-params 的臂）：
  STUB_FULL_DISP_SCALE   满电臂 FORWARD 步的位移乘它（缺省 1.0）：0.97 过 P1，0.93 不过但不证伪，0.88 证伪
  STUB_FULL_YAW_SCALE    满电臂 YAW 步的转角乘它（缺省 1.0）
  STUB_FULL_FLIP_SEEDS   逗号分隔的 seed：K=1 满电臂里这些 seed 的结果取反（造 McNemar 的不一致对）
  STUB_FULL_K3_SCALE     满电臂 K=3 三个通过概率乘它（缺省 1.0）
  STUB_FULL_DROP_SEEDS   逗号分隔的 seed：K=3 满电臂里这些 seed 的末 10 步高度 1.0 m（掉高）
  STUB_FULL_PROFILE_BAD=1  满电臂的 manifest 写对照的指纹（模拟插件没读副本时 Python 侧仍是副本的反例：这里反过来写错指纹，造 INVALID）
  STUB_FULL_STEP0_S      满电臂首步观测时刻比对照多几秒（缺省 2.5）
  STUB_STUDENT_LATE_SEEDS  逗号分隔的 seed：学生臂里这些 seed 的第 0 步迟到；STUB_STUDENT_LATE_ARM 非空时只在这个臂名里生效（臂名不含 -retry1）
  STUB_RETRY_LATE=1      名字以 -retry1 结尾的臂里这些 seed 仍迟到
  STUB_SLEEP             每臂写文件前睡几秒（缺省 0.2）
  STUB_FAIL_ARM          臂名（不含 -retry1）：这个臂的评测进程写完 partial 的 eval.json 后以 rc=9 退出（演练臂没跑完）
  CPU_STAT_FILE          run.sh 导出的同名变量；每个臂结束 usage_usec 加 2.5 s，两条车道同时写用 flock 互斥。
每个臂目录里写 stub_args.txt（收到的命令行），演练脚本据此核对满电臂都带 --cf-sitl-params、对照没有、学生臂没有 --rtf。
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
CONTROL_PROFILE = "cf_sitl_v1:1bc8fedf:lib31"
FULL_PROFILE = "cf_sitl_v1:06e5f017:lib31"


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


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def bump_cpu_stat():
    path = os.environ.get("CPU_STAT_FILE")
    if not path or not os.path.exists(path):
        return
    with open(path + ".lk", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        with open(path) as f:
            kv = dict(line.split() for line in f if line.strip())
        kv["usage_usec"] = str(int(kv.get("usage_usec", 0)) + 2_500_000)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            f.write("".join(f"{k} {v}\n" for k, v in kv.items()))
        os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser()
    for name in ("--policy", "--task", "--ckpt", "--infer-bin", "--split", "--seeds", "--dynamics", "--vehicle", "--window-from", "--out", "--cf-sitl-params"):
        p.add_argument(name)
    for name in ("--steps", "--blocks", "--episodes", "--heading-guard"):
        p.add_argument(name, type=int, default=0)
    for name in ("--action-latency", "--corridor-half-width", "--sim-timeout", "--episode-timeout", "--hold", "--rtf"):
        p.add_argument(name, type=float, default=None)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "stub_args.txt"), "w") as f:
        f.write(" ".join(sys.argv[1:]) + "\n")
    time.sleep(float(os.environ.get("STUB_SLEEP", "0.2")))

    k3 = a.task == "course_base"
    full = a.cf_sitl_params is not None
    L = a.action_latency
    arm = os.path.basename(os.path.normpath(a.out))
    base_arm = arm[:-len("-retry1")] if arm.endswith("-retry1") else arm
    is_retry = arm.endswith("-retry1")
    late_seeds = {int(x) for x in csv_env("STUB_STUDENT_LATE_SEEDS")}
    arm_filter = os.environ.get("STUB_STUDENT_LATE_ARM", "")
    disp_scale = float(os.environ.get("STUB_FULL_DISP_SCALE", "1.0")) if full else 1.0
    yaw_scale = float(os.environ.get("STUB_FULL_YAW_SCALE", "1.0")) if full else 1.0
    k3_scale = float(os.environ.get("STUB_FULL_K3_SCALE", "1.0")) if full else 1.0
    flip = {int(x) for x in csv_env("STUB_FULL_FLIP_SEEDS")} if full else set()
    drops = {int(x) for x in csv_env("STUB_FULL_DROP_SEEDS")} if full else set()
    step0_extra = float(os.environ.get("STUB_FULL_STEP0_S", "2.5")) if full else 0.0
    if full:
        profile = CONTROL_PROFILE if os.environ.get("STUB_FULL_PROFILE_BAD") == "1" else FULL_PROFILE
        params_sha = hashlib.sha256(open(a.cf_sitl_params, "rb").read()).hexdigest() if os.path.exists(a.cf_sitl_params) else "0" * 64
    else:
        profile, params_sha = CONTROL_PROFILE, "683e7400" + "0" * 56
    seeds = parse_seeds(a.seeds)[:a.episodes]
    steps_n = 80 if k3 else 30
    train_seed = int(os.path.basename(a.ckpt or "x_s0").split("_s")[-1][0]) if a.ckpt else -1
    per_episode, failed, late_eps = [], [], []
    for seed in seeds:
        ep = os.path.join(a.out, "episodes", f"seed_{seed}")
        os.makedirs(ep, exist_ok=True)
        late_here = (L is not None and seed in late_seeds and (not arm_filter or arm_filter == base_arm)
                     and (not is_retry or os.environ.get("STUB_RETRY_LATE") == "1"))
        if k3:
            u = h(seed, train_seed, "k3")
            tiers = sum(1 for p_ in (0.8, 0.6, 0.4) if u < p_ * k3_scale)
            x_end = {0: 2.0, 1: 6.5, 2: 11.5, 3: 16.5}[tiers]
            ok = tiers == 3
        else:
            ok = h(seed, train_seed, "k1") < 0.85
            if seed in flip:
                ok = not ok
            x_end = 6.5 if ok else 2.0
        actions, oracle, late_ids = [], [], []
        t0_ns = int((13.0 + 1.5 * h(seed, "t0") + step0_extra) * 1e9)
        for i in range(steps_n):
            j = i % 10
            yaw = j == 0
            lag = 0.13 if L is None else L
            if L is not None and i == 0 and late_here:
                lag = L + 0.035
            late = None if L is None else lag > L + 0.02 + 1e-9
            if late:
                late_ids.append(i)
            start = 10_000_000_000 + i * 1_000_000_000
            disp = 0.04 if yaw else STRATA.get(j, CRUISE) * disp_scale
            actions.append({"step": i, "action_executed": 2 if yaw else 1, "action_expert": 1, "collision": False, "infer_ms": 1.0,
                            "displacement_m": disp, "dyaw_deg": 26.0 * yaw_scale if yaw else 0.1,
                            "dispatch_lag_s": lag, "src": None, "action_start_ns": start, "timestamp_ns": t0_ns + i * HOLD_NS,
                            **({} if late is None else {"late_dispatch": late})})
            z = 1.0 if (k3 and seed in drops and i >= steps_n - 10) else 2.0
            oracle.append({"step": i, "action_start_ns": start, "action_end_ns": start + HOLD_NS,
                           "pose_gt": {"x": x_end * (i + 1) / steps_n, "y": 0.0, "z": z}, "speed_mps": 0.3})
        with open(os.path.join(ep, "actions.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in actions))
        with open(os.path.join(ep, "oracle.jsonl"), "w") as f:
            f.write("".join(json.dumps(r) + "\n" for r in oracle))
        sim_s = sum(o["action_end_ns"] - o["action_start_ns"] for o in oracle) / 1e9 - sum(r["dispatch_lag_s"] for r in actions)
        with open(os.path.join(ep, "episode_manifest.json"), "w") as f:
            json.dump({"goal_line_x": 16.0 if k3 else 6.0, "performance": {"steps": steps_n, "action_sim_s": sim_s, "action_rtf": 0.8},
                       "dynamics_profile": profile, "params_sha256": params_sha, "takeoff_failsafe_retries": 0,
                       **({"action_latency_s": L, "late_dispatch_steps": len(late_ids)} if L is not None else {})}, f)
        lags = [r["dispatch_lag_s"] for r in actions]
        entry = {"seed": seed, "success": bool(ok), "steps": steps_n, "episode_dir": os.path.join("episodes", f"seed_{seed}"),
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
    fail_arm = os.environ.get("STUB_FAIL_ARM") == base_arm and not is_retry
    report = {"episodes_requested": a.episodes, "episodes_done": len(per_episode) // (2 if fail_arm else 1), "failed_count": len(failed), "failed": failed,
              "partial": bool(fail_arm), "skipped_count": 0, "per_episode": per_episode[:len(per_episode) // (2 if fail_arm else 1)],
              "corridor_half_width": a.corridor_half_width or 3.0,
              "metric_version": "sim09.v2.coast1s.corridor3m", "blocks": a.blocks or None, "policy": a.policy, "task": a.task,
              "heading_guard": a.heading_guard, "hold_s": a.hold, "window_from": a.window_from, "action_latency_s": L, "dynamics": a.dynamics,
              "vehicle": a.vehicle, "template_sha256": sha_text("template"), "split": a.split, "steps": a.steps,
              # 真入口：seeds = 评完的集（迟到 / 失败的集不在里面），requested_seeds = 请求的全集（2026-11-12 02:15:48 写出的第一个真实臂 soc100-L0.20-k1-s1 的 eval.json：seeds 19 个，requested_seeds 20 个）
              "seeds": [e["seed"] for e in per_episode], "requested_seeds": parse_seeds(a.seeds),
              "infer_bin_sha256": sha_text(a.infer_bin or ""), "ckpt_sha256": sha_text(a.ckpt or ""), "camera_rate_hz": None, "aux_cameras": True,
              **({"late_dispatch_episodes": late_eps} if L is not None else {}),
              **({"rtf": a.rtf} if a.rtf is not None else {}),
              **({"dynamics_profile": profile, "cf_sitl_params": {"path": a.cf_sitl_params, "sha256": params_sha, "values_sha256": FULL_PROFILE.split(":")[1]}} if full else {})}
    with open(os.path.join(a.out, "eval.json"), "w") as f:
        json.dump(report, f)
    bump_cpu_stat()
    print(f"requested: stub done={len(per_episode)} failed={len(failed)}")
    if fail_arm:
        return 9
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
