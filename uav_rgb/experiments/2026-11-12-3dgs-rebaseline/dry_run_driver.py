#!/usr/bin/env python3
"""3dgs-rebaseline 干跑桩驱动：不碰 GPU、不加载 PLY、不起 cf_sitl，验证 run.sh 与 summarize.py 全链路。

参数接口与 experiments/2026-11-10-bearing-guard-3dgs/driver.py 相同：`python dry_run_driver.py <SUITE_DIR> <SCENE>`。
行为：
  - 按 <SUITE_DIR>/config.json 的 arms、routes、k、seed0、collision_model 生成每集的 metrics.json、trace.jsonl、actions.jsonl 与每条路线的 done.json；
  - 已完成（done.json 无 failures 且集数 >= k）的路线自动跳过；半成品目录改名为 `r<N>.incomplete-<时刻>` 再重做（与真驱动一致）；
  - 支持故障注入环境变量：
    * `STUB_FAIL_SUITE=<套件名>` + `STUB_FAIL_SCENE=<场景名>`：指定任务非零退出（测试 E-6：DONE 的 rc 反映臂/任务失败）；
    * `STUB_DROP_TRACE=1`：故意不写 trace.jsonl（测试 summarize 检出缺 trace）；
    * `STUB_DIFF_SPAWN=1`：在 v2 套件的第 0 集改写 start_pose（测试 summarize 拆分同起点/异起点）。
"""
import hashlib
import importlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

coll_mod = importlib.import_module("sim.3dgs.collision")
METRIC_BASE = "3dgs.pilot.v2+stop.v1"


def u01(tag):
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big") / (2 ** 64)


def finished(d, k):
    try:
        with open(os.path.join(d, "done.json"), encoding="utf-8") as f:
            j = json.load(f)
    except (OSError, ValueError):
        return False
    return not j.get("failures") and len(j.get("episodes", [])) >= k


def main(argv=None):
    argv = argv or sys.argv
    out_dir, scene = os.path.abspath(argv[1]), argv[2]
    with open(os.path.join(out_dir, "config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    suite = cfg.get("suite", os.path.basename(out_dir))
    cm = cfg.get("collision_model", "v1")
    mv = coll_mod.metric_version(METRIC_BASE, cm)

    if os.environ.get("STUB_FAIL_SUITE") == suite and os.environ.get("STUB_FAIL_SCENE") == scene:
        print(f"stub injected failure: {suite} {scene}", file=sys.stderr, flush=True)
        return 1

    routes = cfg["routes"][scene] if isinstance(cfg["routes"], dict) else cfg["routes"]
    if not routes:
        return 0

    time.sleep(float(os.environ.get("STUB_SLEEP_S", "0.05")))
    k_ep = int(cfg["k"])
    seed0 = int(cfg["seed0"])

    for arm in cfg["arms"]:
        g = arm["name"]
        for r in routes:
            rdir = os.path.join(out_dir, g, scene, f"r{r}")
            if os.path.isdir(rdir) and finished(rdir, k_ep):
                continue
            if os.path.isdir(rdir):
                shutil.move(rdir, f"{rdir}.incomplete-{time.strftime('%H%M%S')}")
            os.makedirs(rdir, exist_ok=True)
            eps = []
            for k in range(k_ep):
                seed = seed0 + 100 * int(r) + k
                ep_dir = os.path.join(rdir, f"ep_{k:05d}")
                os.makedirs(ep_dir, exist_ok=True)
                u = u01(f"{g}|{scene}|{r}|{seed}")
                thr = 0.78 if cm == "v1" else (0.76 if "rule" in g else 0.70)
                succ = u < thr
                cf = (not succ) and (u01(f"cf|{g}|{scene}|{r}|{seed}") < 0.4)
                to = (not succ) and (not cf)
                steps = 25
                dx = 0.05 if (cm == "v2" and k == 0 and os.environ.get("STUB_DIFF_SPAWN") == "1") else 0.0
                pose = {"x": round(1.0 + 0.1 * r + dx, 4), "y": 2.0, "yaw_deg": 0.0}
                m = {
                    "success_stop": succ,
                    "success_v2": succ,
                    "collision": cf,
                    "collide_first": cf,
                    "timeout": to,
                    "steps": steps,
                    "reach_step": 18 if succ else None,
                    "collision_step": 12 if cf else None,
                    "spl": 0.8 if succ else 0.0,
                    "spl_stop": 0.8 if succ else 0.0,
                    "final_wp_dist_m": 0.3 if succ else 2.5,
                    "min_wp_dist_m": 0.3 if succ else 2.0,
                    "start_pose": pose,
                    "arm": g,
                    "scene": scene,
                    "route": int(r),
                    "episode": k,
                    "seed": seed,
                    "stopped_on_goal": succ and arm["base"] == "teacher",
                    "wall_s": 0.05,
                    "guard": None,
                    "collision_model": cm,
                    "metric_version": mv,
                }
                if "depth_opacity_min" in arm:
                    m["depth_opacity_min"] = float(arm["depth_opacity_min"])
                with open(os.path.join(ep_dir, "metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(m, f, ensure_ascii=False, indent=1)
                with open(os.path.join(ep_dir, "scene.json"), "w", encoding="utf-8") as f:
                    json.dump({"spawn": {"x": 0.0, "y": 0.0, "z": 1.2},
                               "collision": {"z_slice": [1.0, 1.4], "voxel_m": 0.1, "envelope": "e1_mini"}}, f)
                with open(os.path.join(ep_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
                    f.write(json.dumps({"step": 0, "action_executed": 1, "collision": False,
                                        "pose_before": {"x": pose["x"], "y": pose["y"], "z": 1.2},
                                        "pose_gt": {"x": pose["x"] + 0.1, "y": pose["y"], "z": 1.2},
                                        "yaw_before": pose["yaw_deg"]}) + "\n")
                if not cfg.get("drop_trace") and os.environ.get("STUB_DROP_TRACE") != "1":
                    with open(os.path.join(ep_dir, "trace.jsonl"), "w", encoding="utf-8") as f:
                        f.write(json.dumps({"step": 0, "t": 0.1, "pos": [pose["x"], pose["y"], 1.2]}) + "\n")
                eps.append({
                    "episode": k, "seed": seed, "steps": steps,
                    "success_stop": succ, "collision": cf, "timeout": to, "wall_s": 0.05,
                })
            done = {
                "arm": g, "scene": scene, "route": int(r), "steps": 25, "hold": cfg["hold"],
                "ckpt": None, "guard": arm["guard"], "dynamics_profile": "stub",
                "episodes": eps, "failures": [], "wall_s": 0.1,
                "collision_model": cm, "metric_version": mv,
            }
            if "split_from_identity" in cfg:
                done["split"] = "train"
                done["split_policy"] = "diag-train-scene"
            if "depth_opacity_min" in arm:
                done["depth_opacity_min"] = float(arm["depth_opacity_min"])
            with open(os.path.join(rdir, "done.json"), "w", encoding="utf-8") as f:
                json.dump(done, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
