#!/usr/bin/env python3
"""Callinstrasse 多路线 demo + E1 3DGS 数据集 pilot（A* 教师 + cf_sitl 固件在环 + gsplat）。

流程：路线离线检查 -> 每条路线采 N 集 -> 三视角视频 -> report -> pilot 数据集（路线轮换 + 出生扰动）
-> 健康检查 -> report。任一步失败即停，结束落 $OUT/DONE（rc + 时间）。

运行（守护进程，从仓库根；3dgs-venv 的 python）：
  OUT=/root/rivermind-data/uav_rgb/demo-e1-$(date +%Y%m%d)/3dgs-callinstrasse-routes; mkdir -p $OUT
  setsid nohup /root/rivermind-data/uav_rgb/3dgs-venv/bin/python tools/demo_3dgs_routes.py \
      > $OUT/run.log 2>&1 < /dev/null & echo $! > $OUT/run.pid
只跑 pilot：DEMO_PHASES=pilot；只跑路线 demo：DEMO_PHASES=routes。改路线/集数/seed 只改下面配置区。
pilot 可断点续跑：PILOT_OUT 里已完整采完的 route_*/ 目录（run.json 无 failures、集数够）直接复用，
半成品挪到 PILOT_OUT/incomplete/ 后重采；quarantine/ 里的失控集保留并计入报告。
路线定义在 scene.json 的 routes[]（spawn/waypoint/max_steps），由 sim/3dgs/planner.py --scene 离线验证。
2026-11-07 实测：每集采集 10–15 s（渲染 4 ms/帧，SITL 1 ms/100 步，A* 20–40 ms/步），
三视角视频约 80 ms/帧（10 fps trace）；6 路线 × 2 集 + 视频约 15 min，20 集 pilot 约 8 min。
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------- 配置（改这里）
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 仓库根（本文件在 tools/）
PY = os.environ.get("UAV_RGB_3DGS_PYTHON", "/root/rivermind-data/uav_rgb/3dgs-venv/bin/python")   # torch + gsplat 1.5.3
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
SCENE = os.environ.get("DEMO_SCENE", os.path.join(DATA_ROOT, "uav_rgb", "3dgs-scenes", "callinstrasse", "scene.json"))
OUT = os.environ.get("DEMO_OUT", os.path.join(DATA_ROOT, "uav_rgb", "demo-e1-" + time.strftime("%Y%m%d"),
                                               "3dgs-callinstrasse-routes"))
PILOT_OUT = os.environ.get("PILOT_OUT", os.path.join(DATA_ROOT, "uav_rgb", "datasets", "e1-3dgs-callinstrasse-pilot"))
PHASES = tuple(os.environ.get("DEMO_PHASES", "routes,pilot").split(","))   # routes = 每路线采集+视频+报告；pilot = 数据集
ROUTES = [0, 1, 2, 3, 4, 5]     # scene.routes 索引；0 是原直线（向后兼容），1–5 带转弯/避障（见 scene.json routes[].note）
EPISODES_PER_ROUTE = 2          # 任务要求每路线 >= 2 集
SEED0 = 0                       # 路线 r 第 k 集 seed = SEED0 + 100*r + k（cf_sitl 噪声种子 = seed；同 seed 可复现）
HOLD_S = 0.5                    # 与 scene.step_s / Gazebo 训练数据窗口一致
POLICY = "astar"                # 占据栅格 A* + 纯追踪 + goal-stop（sim/3dgs/planner.py）；深度教师在开阔路线全程 FORWARD
DYNAMICS = "cf_sitl"            # E1 MINI 固件在环（cf_sitl_proc，路径见 sim/3dgs/dynamics.py）
STOP_ON_COLLISION = True        # 代理碰撞即停（真机撞了就是结束）
STEPS = None                    # None = 用 scene.routes[r].max_steps（按航程算，见 planner.route_check）；整数则全路线统一
PILOT_EPISODES = 20             # pilot 规模：6 路线轮换 -> 路线 0,1 各 4 集、其余各 3 集
PILOT_SEED0 = 1000              # 与 routes 阶段的 seed 不重叠；路线 r 第 k 集 seed = PILOT_SEED0 + 100*r + k
PILOT_SPAWN_JITTER_M = 0.5      # 出生位置 ±0.5 m、偏航 ±10°（collect_3dgs._spawn_for_episode，自动避开占据格）
PILOT_RETRY_MAX = 3             # cf_sitl 偶发执行器失控（collect_3dgs 标 dynamics_anomaly）：该集隔离到 quarantine/，
                                # 同路线 seed+10*n 补采，最多补 3 轮（route 5 实测 3/6 集在 t>35 s 倒飞，其余路线 0/28）
PILOT_SPLIT = os.environ.get("PILOT_SPLIT", "train")   # pilot 是训练数据：整个 Callinstrasse 一个 geometry_gid，split 由哈希定，
                                # 所以显式 --split train。与哈希一致时 manifest 仍是 geometry_gid（当前 nav_to_world 下的 gid
                                # 86b0f5cb953fbb03 恰好落 train）；不一致时标 diag-manual-split，preprocess 要 --allow-diagnostic-split。
                                # 3DGS 测试集走闭环 eval_3dgs 的留出路线，不靠 split 字段。设 PILOT_SPLIT= 空串则按哈希（旧行为）
# 视频超参数（帧率 / 跟随距离 / 顶视范围）在 sim/3dgs/video.py 顶部配置区，这里不重复。
ENV = dict(os.environ, TORCH_EXTENSIONS_DIR=os.path.join(DATA_ROOT, "uav_rgb", "torch-cache", "extensions-sm89-20261106"),
           TMPDIR=os.path.join(DATA_ROOT, "uav_rgb", "tmp"))
ENV.pop("TORCH_CUDA_ARCH_LIST", None)     # gsplat JIT 按本机 sm_89 编；带着旧变量会重编/编错架构
ACTIONS = ("SAFE", "FORWARD", "YAW_L", "YAW_R")


def run(cmd, what, capture=False):
    print(f"\n### {what}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    if capture:
        res = subprocess.run(cmd, cwd=REPO, env=ENV, text=True, capture_output=True)
        print(res.stderr[-3000:], flush=True)
        rc, out = res.returncode, res.stdout
    else:
        rc, out = subprocess.call(cmd, cwd=REPO, env=ENV), None
    print(f"### {what} rc={rc} wall={time.time() - t0:.1f}s", flush=True)
    if rc != 0:
        raise SystemExit(f"{what} 失败 rc={rc}")
    return out


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_row(ep, reach_radius):
    """一集的摘要：步数、到达（轨迹任一点到 waypoint <= reach_radius）、碰撞、动作分布、最小净空。"""
    man = json.load(open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8"))
    acts = read_jsonl(os.path.join(ep, "actions.jsonl"))
    orc = read_jsonl(os.path.join(ep, "oracle.jsonl"))
    wp = man["waypoint"]
    dists = [math.hypot(a["pose_gt"]["x"] - wp["x"], a["pose_gt"]["y"] - wp["y"]) for a in acts]
    if acts:
        p0 = acts[0]["pose_before"]
        dists.insert(0, math.hypot(p0["x"] - wp["x"], p0["y"] - wp["y"]))
    dist = [0, 0, 0, 0]
    for a in acts:
        dist[int(a["action_executed"])] += 1
    n_rgb = len([f for f in os.listdir(ep) if f.startswith("rgb_") and f.endswith(".png")])
    return {"dir": ep, "route": man.get("route"), "seed": man["seed"], "steps": len(acts),
            "requested_steps": man["requested_steps"], "reached": bool(dists and min(dists) <= reach_radius),
            "collision": any(a["collision"] for a in acts), "stopped_on_collision": man["stopped_on_collision"],
            "stopped_on_goal": man.get("stopped_on_goal"), "min_dist_m": round(min(dists), 2) if dists else None,
            "final_dist_m": round(dists[-1], 2) if dists else None,
            "min_clearance_m": round(min(a["clearance_m"] for a in acts), 2) if acts else None,
            "actions": dict(zip(ACTIONS, dist)), "n_rgb": n_rgb, "n_oracle": len(orc),
            "spawn_actual": acts[0]["pose_before"] if acts else None, "error": man["error"],
            "anomaly": man.get("dynamics_anomaly"),
            # A* fail-closed 输出 SAFE 的步数（旧 manifest / 非 A* 教师为 None）
            "planner_failures": man.get("planner_failures"),
            # 低质量帧（雾/糊）数：旧 manifest 没有 frame_quality 时为 None
            "n_low_quality": (man.get("frame_quality") or {}).get("n_low"),
            "validity": man["validity"], "teacher": man["teacher"], "teacher_version": man.get("teacher_version"),
            "render_ms_p50": man["performance"]["render_ms"]["p50"], "takeoff_s": man["dynamics_params"].get("takeoff_s")}


def actions_str(d):
    return " ".join(f"{k}:{v}" for k, v in d.items())


def anomaly_str(x):
    a = x.get("anomaly")
    return f"{a['kind']}@{a['step']}" if a else "-"


def collect(out_dir, route, episodes, seed, jitter=0.0, steps=None, split=None):
    cmd = [PY, "-m", "sim.3dgs.collect_3dgs", "--scene", SCENE, "--out", out_dir, "--route", str(route),
           "--policy", POLICY, "--dynamics", DYNAMICS, "--episodes", str(episodes), "--seed", str(seed),
           "--hold", str(HOLD_S)]
    cmd += ["--steps", str(steps)] if steps else []
    cmd += ["--spawn-jitter", str(jitter)] if jitter > 0 else []
    cmd += ["--stop-on-collision"] if STOP_ON_COLLISION else []
    cmd += ["--split", split] if split else []
    run(cmd, f"collect route {route}")
    return [os.path.join(out_dir, "ep_%05d" % i) for i in range(episodes)]


def complete_episodes(out_dir, episodes):
    """断点续跑：out_dir 是 collect() 完整跑完的产物时返回其中还在的 ep 目录（失控集可能已挪到 quarantine/），
    否则 None。完整 = run.json 存在、failures 为空、episodes 数等于要求数、每个还在的 ep 有无 error 的 manifest。"""
    run_json = os.path.join(out_dir, "run.json")
    if not os.path.isfile(run_json):
        return None
    info = json.load(open(run_json, encoding="utf-8"))
    if info.get("failures") or len(info.get("episodes", [])) != episodes:
        return None
    eps = []
    for e in info["episodes"]:
        ep = os.path.join(out_dir, e["dir"])
        if not os.path.isdir(ep):
            continue                        # 已隔离到 quarantine/
        man = os.path.join(ep, "episode_manifest.json")
        if not os.path.isfile(man) or json.load(open(man, encoding="utf-8")).get("error"):
            return None
        eps.append(ep)
    return eps


def collect_or_reuse(out_dir, route, episodes, seed, jitter, steps, split):
    """已完整采过就复用；半成品挪到 PILOT_OUT/incomplete/ 再重采（不删任何目录）。"""
    eps = complete_episodes(out_dir, episodes)
    if eps is not None:
        print(f"### reuse {out_dir}: {len(eps)} 集已存在，跳过采集", flush=True)
        return eps
    if os.path.isdir(out_dir):
        dst = os.path.join(PILOT_OUT, "incomplete", f"{os.path.basename(out_dir)}-{time.strftime('%H%M%S')}")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(out_dir, dst)
        print(f"### {out_dir} 不完整，挪到 {dst} 后重采", flush=True)
    return collect(out_dir, route, episodes, seed, jitter=jitter, steps=steps, split=split)


def phase_routes(scene, checks):
    reach = float(scene.get("reach_radius_m", 1.5))
    routes_meta = {c["route"]: c for c in checks}
    all_eps = {}
    # 1. 每条路线采 EPISODES_PER_ROUTE 集
    for r in ROUTES:
        all_eps[r] = collect(os.path.join(OUT, "collect", f"route_{r}"), r, EPISODES_PER_ROUTE, SEED0 + 100 * r, steps=STEPS)
    # 2. 三视角视频（一个进程渲染全部集：每集 views.webm + views_key_*.png）
    eps_flat = [e for r in ROUTES for e in all_eps[r]]
    run([PY, "-m", "sim.3dgs.video", "--scene", SCENE, "--ep", *eps_flat], "video")
    # 3. 报告 + 每路线挑 1 集视频汇总到 OUT/videos/
    vid_dir = os.path.join(OUT, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    rows, picks = [], {}
    for r in ROUTES:
        for ep in all_eps[r]:
            row = episode_row(ep, reach)
            row["video"] = os.path.join(ep, "views.webm")
            row["keyframes"] = sorted(os.path.join(ep, f) for f in os.listdir(ep) if f.startswith("views_key_"))
            rows.append(row)
        # 挑：到达且无碰撞/失控里步数最多的（转弯最多）；都没到就挑第一集
        good = [x for x in rows if x["route"]["index"] == r and x["reached"] and not x["collision"] and not x["anomaly"]]
        pick = max(good, key=lambda x: x["steps"]) if good else [x for x in rows if x["route"]["index"] == r][0]
        name = routes_meta[r]["name"]
        dst = os.path.join(vid_dir, f"route{r}_{name}_seed{pick['seed']}.webm")
        shutil.copy2(pick["video"], dst)
        for k, kf in enumerate(pick["keyframes"]):
            shutil.copy2(kf, os.path.join(vid_dir, f"route{r}_{name}_seed{pick['seed']}_key{k:02d}.png"))
        picks[r] = {"seed": pick["seed"], "video": dst, "src": pick["video"]}
    totals = [sum(x["actions"][a] for x in rows) for a in ACTIONS]
    report = {"scene": SCENE, "policy": POLICY, "dynamics": DYNAMICS, "hold_s": HOLD_S, "reach_radius_m": reach,
              "stop_radius_m": 0.5, "routes": checks, "episodes": rows, "picked_videos": picks,
              "action_totals": dict(zip(ACTIONS, totals)),
              "all_actions_present": all(t > 0 for t in totals),
              "window": "until_s = t_obs + hold（渲染/决策/下发同刻）"}
    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    lines = [f"# Callinstrasse 多路线教师闭环（{POLICY} + {DYNAMICS}，hold {HOLD_S}s，到达圈 {reach} m，goal-stop 0.5 m）", "",
             "## 路线（离线检查：sim/3dgs/planner.py）", "",
             "| route | name | spawn (x,y,yaw°) | waypoint | 航程 m | 转向° | 最小净空 m | 步数预算 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in checks:
        sp = c["spawn"]
        lines.append(f"| {c['route']} | {c['name']} | ({sp['x']}, {sp['y']}, {sp['yaw_deg']}) | ({c['waypoint']['x']}, {c['waypoint']['y']}) "
                     f"| {c['path_m']} | {c['turn_deg']} | {c['min_clearance_m']} | {c['max_steps']} |")
    lines += ["", "## 每集结果", "",
              "| route | seed | steps | 到达 | 碰撞 | 失控 | goal-stop | min dist m | 最小净空 m | 低质帧 | 动作分布 | 视频 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in rows:
        lines.append(f"| {x['route']['index']} {x['route']['name']} | {x['seed']} | {x['steps']}/{x['requested_steps']} | {x['reached']} | {x['collision']} "
                     f"| {anomaly_str(x)} | {x['stopped_on_goal']} | {x['min_dist_m']} | {x['min_clearance_m']} | {x['n_low_quality']} | {actions_str(x['actions'])} | {x['video']} |")
    lines += ["", f"动作合计：{actions_str(report['action_totals'])}（四类都出现：{report['all_actions_present']}）", "",
              "## 每路线挑选的视频（已复制到 videos/）", ""]
    for r, pk in picks.items():
        lines.append(f"- route {r} seed {pk['seed']}: {pk['video']}")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def health_check(rows):
    """pilot 健康检查：碰撞即停、四类动作占比、位姿/图像配对、validity、步数。"""
    problems = []
    for x in rows:
        tag = f"{os.path.basename(os.path.dirname(x['dir']))}/{os.path.basename(x['dir'])}"
        if x["error"]:
            problems.append(f"{tag}: error={x['error']}")
        if not (x["n_rgb"] == x["steps"] == x["n_oracle"]):
            problems.append(f"{tag}: rgb {x['n_rgb']} / actions {x['steps']} / oracle {x['n_oracle']} 不配对")
        if x["collision"] and not x["stopped_on_collision"]:
            problems.append(f"{tag}: 有碰撞但没有即停")
        if not all(x["validity"].values()):
            problems.append(f"{tag}: validity={x['validity']}")
        if x["anomaly"]:
            problems.append(f"{tag}: dynamics_anomaly={x['anomaly']}")
        if x["planner_failures"]:
            # fail-closed 的 SAFE 标签在画面里没有依据，这样的集不进训练集
            problems.append(f"{tag}: A* 规划失败 {x['planner_failures']} 步（fail-closed SAFE）")
        acts = read_jsonl(os.path.join(x["dir"], "actions.jsonl"))
        ts = [a["timestamp_ns"] for a in acts]
        if any(b <= a for a, b in zip(ts, ts[1:])):
            problems.append(f"{tag}: timestamp_ns 非严格递增")
        for a in acts:
            pg = a["pose_gt"]
            if not all(math.isfinite(float(pg[k])) for k in ("x", "y", "z", "yaw_deg")):
                problems.append(f"{tag}: step {a['step']} pose 非有限")
                break
            if not os.path.isfile(os.path.join(x["dir"], a["frame"])):
                problems.append(f"{tag}: 缺帧 {a['frame']}")
                break
    totals = [sum(x["actions"][a] for x in rows) for a in ACTIONS]
    n = sum(totals)
    n_low = sum(x["n_low_quality"] or 0 for x in rows)     # 雾/糊帧只统计不判失败：丢不丢由 preprocess 决定
    return {"episodes": len(rows), "frames": n, "reached": sum(x["reached"] for x in rows),
            "collisions": sum(x["collision"] for x in rows),
            "stopped_on_goal": sum(bool(x["stopped_on_goal"]) for x in rows),
            "action_totals": dict(zip(ACTIONS, totals)),
            "action_fractions": {a: round(t / n, 3) if n else None for a, t in zip(ACTIONS, totals)},
            "all_actions_present": all(t > 0 for t in totals),
            "min_clearance_m": min(x["min_clearance_m"] for x in rows if x["min_clearance_m"] is not None),
            "low_quality_frames": n_low, "low_quality_frac": round(n_low / n, 4) if n else None,
            "problems": problems, "ok": not problems}


def phase_pilot(scene, checks):
    reach = float(scene.get("reach_radius_m", 1.5))
    os.makedirs(PILOT_OUT, exist_ok=True)
    per_route = {r: 0 for r in ROUTES}
    for i in range(PILOT_EPISODES):          # 轮换：第 i 集用 ROUTES[i % n]
        per_route[ROUTES[i % len(ROUTES)]] += 1
    eps = []
    for r in ROUTES:
        if per_route[r]:
            eps += collect_or_reuse(os.path.join(PILOT_OUT, f"route_{r}"), r, per_route[r], PILOT_SEED0 + 100 * r,
                                    PILOT_SPAWN_JITTER_M, STEPS, PILOT_SPLIT)
    # 上次运行留下的补采集（route_r_retry_seed*/）也复用，不然同 seed 重采会撞目录
    for name in sorted(os.listdir(PILOT_OUT)):
        if "_retry_seed" in name:
            eps += complete_episodes(os.path.join(PILOT_OUT, name), 1) or []
    # 失控集隔离 + 补采：标了 dynamics_anomaly 的集（FORWARD 下倒飞，标签与运动相反）移到 quarantine/ 留证据，
    # 不进 episodes.txt；同路线换 seed 补一集，直到没有异常集或用完 PILOT_RETRY_MAX 轮
    qdir = os.path.join(PILOT_OUT, "quarantine")
    for attempt in range(1, PILOT_RETRY_MAX + 1):
        bad = [ep for ep in eps if json.load(open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8")).get("dynamics_anomaly")]
        if not bad:
            break
        os.makedirs(qdir, exist_ok=True)
        for ep in bad:
            man = json.load(open(os.path.join(ep, "episode_manifest.json"), encoding="utf-8"))
            r, seed = man["route"]["index"], man["seed"]
            shutil.move(ep, os.path.join(qdir, f"route_{r}_seed{seed}"))
            eps.remove(ep)
            new_seed = seed + 10 * attempt       # k<4、attempt<=3：与同路线其它集和别的路线都不撞
            # shortcut: 上次运行在隔离之后、补采之前被打断的集不会再补（quarantine 里有、eps 里没有），
            # 集数会少 1–2；要补齐就删掉 pilot_manifest 后手动 collect 到 route_r_retry_seed<seed+10>
            eps += [e for e in collect_or_reuse(os.path.join(PILOT_OUT, f"route_{r}_retry_seed{new_seed}"), r, 1,
                                                new_seed, PILOT_SPAWN_JITTER_M, STEPS, PILOT_SPLIT) if e not in eps]
    quarantined = []                         # 从目录重建，续跑时上次隔离的也在
    for name in sorted(os.listdir(qdir)) if os.path.isdir(qdir) else []:
        man = json.load(open(os.path.join(qdir, name, "episode_manifest.json"), encoding="utf-8"))
        quarantined.append({"route": man["route"]["index"], "seed": man["seed"], "dir": os.path.join(qdir, name),
                            "anomaly": man["dynamics_anomaly"]})
    rows = [episode_row(ep, reach) for ep in eps]
    health = health_check(rows)
    health["quarantined"] = quarantined
    man0 = json.load(open(os.path.join(eps[0], "episode_manifest.json"), encoding="utf-8"))
    meta = {"schema": "uavrgb.3dgs_pilot.v1", "scene": SCENE, "routes": ROUTES, "episodes_per_route": per_route,
            "policy": POLICY, "dynamics": DYNAMICS, "hold_s": HOLD_S, "spawn_jitter_m": PILOT_SPAWN_JITTER_M,
            "seed_rule": f"seed = {PILOT_SEED0} + 100*route + k", "stop_on_collision": STOP_ON_COLLISION,
            "pilot_split": PILOT_SPLIT or None,
            "split": man0["split"], "split_policy": man0["split_policy"], "geometry_gid": man0["geometry_gid"],
            "geometry_version": man0["geometry_version"], "teacher_version": man0.get("teacher_version"),
            "dynamics_profile": man0["dynamics_profile"], "episodes": rows, "health": health}
    with open(os.path.join(PILOT_OUT, "pilot_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(PILOT_OUT, "episodes.txt"), "w", encoding="utf-8") as f:   # 给 tools/preprocess.py 当参数
        f.write("\n".join(eps) + "\n")
    lines = [f"# E1 3DGS Callinstrasse pilot（{PILOT_EPISODES} 集，路线轮换 {ROUTES}，出生扰动 ±{PILOT_SPAWN_JITTER_M} m / ±10°，"
             f"seed = {PILOT_SEED0} + 100*route + k）", "",
             f"- 采集：{POLICY} 教师 + {DYNAMICS}，hold {HOLD_S}s，碰撞即停 {STOP_ON_COLLISION}；步数上限 = scene.routes[r].max_steps，goal-stop 连续 3 步 SAFE 提前结束",
             f"- split={man0['split']}（split_policy={man0['split_policy']}，geometry_gid={man0['geometry_gid']}）：整个 Callinstrasse 是一个几何组；"
             + (f"PILOT_SPLIT={PILOT_SPLIT} 与哈希所得不同，记为诊断划分，preprocess 要 --allow-diagnostic-split" if man0["split_policy"] != "geometry_gid"
                else (f"PILOT_SPLIT={PILOT_SPLIT} 与哈希所得一致，preprocess 直接打包" if PILOT_SPLIT
                      else "按哈希划分（未设 PILOT_SPLIT）；若要改 split 设 PILOT_SPLIT=train，manifest 会标 diag-manual-split")),
             f"- 格式：navlab.episode.v1（episode_manifest.json / actions.jsonl / oracle.jsonl / rgb_%05d.png / trace.jsonl），与 sim/tools/collect.py 同口径；差异见 worklog", "",
             "## 健康检查", "",
             f"- 集数 {health['episodes']}，帧数 {health['frames']}，到达 {health['reached']}/{health['episodes']}，碰撞集 {health['collisions']}，goal-stop {health['stopped_on_goal']}",
             f"- 动作合计 {actions_str(health['action_totals'])}；占比 {actions_str(health['action_fractions'])}；四类都出现：{health['all_actions_present']}",
             f"- 最小净空 {health['min_clearance_m']} m；位姿/图像配对、timestamp 递增、validity、碰撞即停、无动力学异常：{'全部通过' if health['ok'] else '有问题'}",
             f"- 低质量帧（lapvar < {man0.get('frame_quality', {}).get('lapvar_min')}，雾/糊，只标记不丢）：{health['low_quality_frames']} / {health['frames']}"
             f"（{health['low_quality_frac']}）；每帧 actions.jsonl frame_quality",
             f"- 失控隔离（quarantine/，不在 episodes.txt）：{len(quarantined)} 集"
             + ("".join(f"；route {q['route']} seed {q['seed']} {q['anomaly']['kind']}@step {q['anomaly']['step']}" for q in quarantined))]
    lines += [f"  - {p}" for p in health["problems"]]
    lines += ["", "## 每集", "", "| route | seed | steps | 到达 | 碰撞 | goal-stop | min dist m | 最小净空 m | 低质帧 | 动作分布 | 实际出生 (x,y) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in rows:
        sp = x["spawn_actual"] or {}
        lines.append(f"| {x['route']['index']} {x['route']['name']} | {x['seed']} | {x['steps']}/{x['requested_steps']} | {x['reached']} | {x['collision']} "
                     f"| {x['stopped_on_goal']} | {x['min_dist_m']} | {x['min_clearance_m']} | {x['n_low_quality']} | {actions_str(x['actions'])} | ({sp.get('x', 0):.2f}, {sp.get('y', 0):.2f}) |")
    with open(os.path.join(PILOT_OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if not health["ok"]:
        raise SystemExit("pilot 健康检查有问题（见 report.md）")


def main():
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, REPO)
    import importlib
    scene = importlib.import_module("sim.3dgs.scene").load_scene(SCENE)
    # 0. 路线离线检查（A* 可达 + 净空 + 步数预算），不通过就停
    out = run([PY, "-m", "sim.3dgs.planner", "--scene", SCENE, "--route", *map(str, ROUTES), "--json"], "route-check", capture=True)
    checks = json.loads(out.strip().splitlines()[-1])
    with open(os.path.join(OUT, "routes_check.json"), "w", encoding="utf-8") as f:
        json.dump(checks, f, ensure_ascii=False, indent=2)
    if "routes" in PHASES:
        phase_routes(scene, checks)
    if "pilot" in PHASES:
        phase_pilot(scene, checks)


if __name__ == "__main__":
    rc = 0
    try:
        main()
    except BaseException as e:  # noqa: BLE001 - DONE 里要有失败原因
        rc = 1
        print(f"FAILED: {type(e).__name__}: {e}", flush=True)
        raise
    finally:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "DONE"), "w") as f:
            f.write(f"rc={rc} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
