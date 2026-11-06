#!/usr/bin/env python3
"""Callinstrasse 教师闭环 demo（E1 MINI 固件在环 + 3DGS）：采集 3 集 -> 三视角视频 -> 报告。

运行（守护进程，从仓库根；系统 python3 即可，渲染子进程自己用 3dgs-venv）：
  OUT=/root/rivermind-data/uav_rgb/demo-e1-$(date +%Y%m%d)/3dgs-callinstrasse-teacher; mkdir -p $OUT
  setsid nohup python3 tools/demo_3dgs_callinstrasse.py > $OUT/run.log 2>&1 < /dev/null & echo $! > $OUT/run.pid
结束落 $OUT/DONE（退出码 + 时间）。任一步失败即停。改集数/步数/hold 只改下面配置区；
DEMO_OUT / DEMO_STEPS 环境变量可覆盖；cf_sitl_proc 路径用 CF_SITL_PROC 覆盖（见 sim/3dgs/dynamics.py）。
2026-11-07 实测：40 步（20 s）3 集无碰撞但差 0.16–0.35 m 未进 2 m 到达圈（固件 0.3 m/s）；48 步 3/3 到达。
"""
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------- 配置（改这里）
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本文件在 tools/）
PY = os.environ.get("UAV_RGB_3DGS_PYTHON", "/root/rivermind-data/uav_rgb/3dgs-venv/bin/python")   # torch + gsplat 1.5.3（数据盘 venv）
SCENE = os.environ.get("DEMO_SCENE", "/root/rivermind-data/uav_rgb/3dgs-scenes/callinstrasse/scene.json")   # LUH Callinstrasse（MIT）：spawn (-14,0,2) -> waypoint (-6,0,2)
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
OUT = os.environ.get("DEMO_OUT", os.path.join(DATA_ROOT, "uav_rgb", "demo-e1-" + time.strftime("%Y%m%d"), "3dgs-callinstrasse-teacher"))
EPISODES = 3        # 任务要求 seed 0,1,2
SEED0 = 0
STEPS = int(os.environ.get("DEMO_STEPS", 48))   # 8 m 航程 / 0.3 m/s ≈ 27 s → 48 步（24 s）刚够进 2 m 到达圈；40 步实测差 0.16–0.35 m 到不了
HOLD_S = 0.5        # 与 scene.step_s / 训练数据窗口一致
STOP_ON_COLLISION = True   # demo 口径：代理碰撞即停（真机撞了就是结束）
# 视频超参数（帧率 / 跟随距离 / 顶视高度）在 sim/3dgs/video.py 顶部配置区，这里不重复。
ENV = dict(os.environ, TORCH_EXTENSIONS_DIR="/root/rivermind-data/uav_rgb/torch-cache/extensions-sm89-20261106",
           TMPDIR=os.path.join(DATA_ROOT, "uav_rgb", "tmp"))
ENV.pop("TORCH_CUDA_ARCH_LIST", None)     # gsplat JIT 按本机 sm_89 编；带着旧变量会重编/编错架构


def run(cmd, what):
    print(f"\n### {what}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO, env=ENV)
    print(f"### {what} rc={rc} wall={time.time() - t0:.1f}s", flush=True)
    if rc != 0:
        raise SystemExit(f"{what} 失败 rc={rc}")


def main():
    os.makedirs(OUT, exist_ok=True)
    collect_dir = os.path.join(OUT, "collect")
    # 1. 采集：教师闭环，cf_sitl 后端
    run([PY, "-m", "sim.3dgs.collect_3dgs", "--scene", SCENE, "--out", collect_dir,
         "--policy", "teacher", "--dynamics", "cf_sitl", "--episodes", str(EPISODES),
         "--seed", str(SEED0), "--steps", str(STEPS), "--hold", str(HOLD_S)]
        + (["--stop-on-collision"] if STOP_ON_COLLISION else []), "collect")
    eps = [os.path.join(collect_dir, "ep_%05d" % i) for i in range(EPISODES)]
    # 2. 三视角视频（每集 views.webm + views_key_*.png）
    run([PY, "-m", "sim.3dgs.video", "--scene", SCENE, "--ep", *eps], "video")
    # 3. 报告：到达/碰撞/步数/动作分布（口径 = eval_3dgs.summarize_3dgs_episode）
    sys.path.insert(0, REPO)
    import importlib
    eval_mod = importlib.import_module("sim.3dgs.eval_3dgs")
    scene_mod = importlib.import_module("sim.3dgs.scene")
    scene = scene_mod.load_scene(SCENE)
    rows = []
    for ep in eps:
        man = json.load(open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8"))
        acts = eval_mod._read_jsonl(os.path.join(ep, "actions.jsonl"))
        orc = eval_mod._read_jsonl(os.path.join(ep, "oracle.jsonl"))
        s = eval_mod.summarize_3dgs_episode(acts, orc, scene)
        dist = {}
        for a in acts:
            k = int(a["action_executed"])
            dist[k] = dist.get(k, 0) + 1
        last = orc[-1]["pose_gt"] if orc else None
        zs = [a["pose_gt"]["z"] for a in acts]
        rows.append({"seed": man["seed"], "dir": ep, "steps": s["steps"], "reached": s["success"],
                     "collision": s["collision"], "waypoint_min_dist_m": s["waypoint_min_dist"],
                     "final_waypoint_dist_m": s["final_waypoint_dist"], "final_pose": last,
                     "z_min_max": [min(zs), max(zs)] if zs else None,
                     "actions": {eval_mod.collect_3dgs.ACTIONS[k]: v for k, v in sorted(dist.items())},
                     "takeoff_s": man["dynamics_params"]["takeoff_s"],
                     "stopped_on_collision": man["stopped_on_collision"], "error": man["error"],
                     "render_ms_p50": man["performance"]["render_ms"]["p50"],
                     "video": os.path.join(ep, "views.webm"),
                     "keyframes": sorted(f for f in os.listdir(ep) if f.startswith("views_key_"))})
    report = {"scene": SCENE, "policy": "teacher", "dynamics": "cf_sitl", "steps_max": STEPS,
              "hold_s": HOLD_S, "reach_radius_m": scene.get("reach_radius_m"),
              "window": "until_s = t_obs + hold（渲染/决策/下发同刻）", "episodes": rows}
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    lines = ["| seed | steps | reached | collision | min dist (m) | final dist (m) | z range (m) | actions | takeoff (s) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append("| %d | %d | %s | %s | %.2f | %.2f | %.2f–%.2f | %s | %.2f |" % (
            r["seed"], r["steps"], r["reached"], r["collision"], r["waypoint_min_dist_m"],
            r["final_waypoint_dist_m"], r["z_min_max"][0], r["z_min_max"][1],
            " ".join(f"{k}:{v}" for k, v in r["actions"].items()), r["takeoff_s"]))
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except BaseException as e:  # noqa: BLE001 - DONE 里要有失败原因
        rc = 1
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        with open(os.path.join(OUT, "DONE"), "w") as f:
            f.write(f"rc={rc} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
