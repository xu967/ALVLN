#!/usr/bin/env python3
"""eval_closed_loop.py 的桩（只用于 run_smoke_ack.sh 的演练，不起 Gazebo）：按命令行写出 eval.json 与 episodes/seed_N/actions.jsonl。

环境变量：STUB_DELAY_MS（确认比 观测戳 + L 晚多少毫秒，缺省 5）、STUB_DELAY_LOAD_MS（臂名 load 用，缺省同上）、
STUB_BREAK_SMOKE=1（冒烟臂的行缺 plugin_ack_sim_t 键）、STUB_SLEEP_S（每个臂先睡这么久，演练信号与占核进程用）、STUB_SLEEP_LOAD_S（只对臂名 load 用，缺省同上）。
"""
import json
import os
import sys
import time

NS = 10**9


def arg(name, default=None):
    argv = sys.argv
    return argv[argv.index(name) + 1] if name in argv else default


def main():
    out = arg("--out")
    seeds = [int(x) for x in arg("--seeds", "0").split(",")]
    n = int(arg("--episodes", "1"))
    latency = float(arg("--action-latency", "0.2"))
    arm = os.path.basename(out.rstrip("/"))
    time.sleep(float(os.environ.get("STUB_SLEEP_LOAD_S" if arm == "load" else "STUB_SLEEP_S",
                                    os.environ.get("STUB_SLEEP_S", "0"))))
    delay = float(os.environ.get("STUB_DELAY_LOAD_MS" if arm == "load" else "STUB_DELAY_MS",
                                 os.environ.get("STUB_DELAY_MS", "5"))) / 1e3
    broken = arm == "smoke" and os.environ.get("STUB_BREAK_SMOKE") == "1"
    for seed in seeds[:n]:
        d = os.path.join(out, "episodes", f"seed_{seed}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "actions.jsonl"), "w") as f:
            for i in range(12):
                obs = 10.005 + 0.9 * i
                row = {"step": i, "pose_ts": int(round(obs * NS)), "dispatch_lag_s": latency,
                       "safe_gap_s": None if i == 0 else 0.0, "obs_gap_s": None if i == 0 else 0.9}
                if not broken:
                    row["plugin_ack_sim_t"] = obs + latency + delay
                f.write(json.dumps(row) + "\n")
    with open(os.path.join(out, "eval.json"), "w") as f:
        json.dump({"episodes_done": min(n, len(seeds)), "failed_count": 0, "partial": False,
                   "late_dispatch_episodes": [], "stall_episodes": []}, f)
    print(f"requested: {n} episodes (stub {arm})")


if __name__ == "__main__":
    main()
