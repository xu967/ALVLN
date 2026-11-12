#!/usr/bin/env python3
"""测量整集高度包络与起飞水平漂移（卡 S1b 第二轮 J-1：先量后定）。

定稿碰撞口径（collision_model="v2"）：
  - 碰撞判定不再有高度余量：圆柱 [z − 半高, z + 半高] 与体素方块相交即碰；
  - 按名义出生高度做的两件事（规划硬栅格 planning_grid、出生检查 spawn_ok）用整集高度包络取层：
    [名义高度 − Z_ENVELOPE_DOWN_M − 半高, 名义高度 + Z_ENVELOPE_UP_M + 半高]，
    出生检查另要求候选点到这些层内原始体素方块的水平距离 >= 机体半径 + inflate + SPAWN_DRIFT_MARGIN_M。

本脚本在已跑完的 W1-11 / W1-12 / K1 / K3 共 16 个来源上测量：
  1. 全部步（含第 0 步动作前位姿 pose_before、步末 pose_gt，以及若有 trace.jsonl 的全部 10 Hz 轨迹点）的 z − spawn.z 最小与最大值；
  2. 每集起飞水平漂移：第 0 步动作前位姿 (pose_before.x, pose_before.y) 相对起飞前候选出生点 (cx, cy) 的欧氏距离最大值；
  3. 各极值向外取到 ROUND_UP_TO_M（5 mm）的整数倍，定出 Z_ENVELOPE_DOWN_M、Z_ENVELOPE_UP_M、SPAWN_DRIFT_MARGIN_M。

只读，不用 GPU：
  <3dgs-venv python> experiments/2026-11-12-3dgs-rebaseline/measure_z_margin.py [输出 json 路径]
"""
import importlib
import json
import math
import os
import sys

import numpy as np

# ------------------------------------------------------------------ 配置区
DATA = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
SCENES_DIR = os.path.join(DATA, "3dgs-scenes")
ROUND_UP_TO_M = 0.005       # 向外取到 5 mm 的整数倍
# (标签, 根目录, 根目录与 r<N> 之间的层数, 出生抖动半径 m)：W1-11 / W1-12 / K1 / K3 全部已完成正式臂
SOURCES = []
for _s, _jit in (("indoor", 0.2), ("outdoor", 0.5)):
    SOURCES += [
        (f"teacher-tol/{_s}", f"{DATA}/exp-teacher-tol-3dgs-20261110/{_s}/teacher-tol", 1, _jit),
        (f"swept/{_s}", f"{DATA}/exp-teacher-clearance-3dgs-20261111/{_s}/swept", 1, _jit),
        (f"k1-forward+route/{_s}", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/{_s}/forward+route", 1, _jit),
        (f"k1-rule-occ+route/{_s}", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/{_s}/rule-occ+route", 1, _jit),
    ]
SOURCES += [
    ("k3-forward+route/holdout", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/forward+route", 1, 0.2),
    ("k3-rule-occ+route/holdout", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule-occ+route", 1, 0.2),
    ("k3-rule2-occ+route/holdout", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule2-occ+route", 1, 0.2),
    ("k3-rule2-depth+route/holdout", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule2-depth+route", 1, 0.2),
    ("k3-rule2-occ+route/formal40", f"{DATA}/exp-rule-v2-3dgs-20261111/formal40/rule2-occ+route", 1, 0.2),
    ("k3-rule2-depth+route/formal40", f"{DATA}/exp-rule-v2-3dgs-20261111/formal40/rule2-depth+route", 1, 0.2),
    ("k3-rule2-occ+route/outdoor", f"{DATA}/exp-rule-v2-3dgs-20261111/outdoor/rule2-occ+route", 1, 0.5),
    ("k3-rule2-depth+route/outdoor", f"{DATA}/exp-rule-v2-3dgs-20261111/outdoor/rule2-depth+route", 1, 0.5),
]
# ------------------------------------------------------------------ 流程

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
if os.path.join(REPO, "sim", "tools") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "sim", "tools"))

route_dirs = importlib.import_module("sim.3dgs.route_dirs")
scene_mod = importlib.import_module("sim.3dgs.scene")
coll_mod = importlib.import_module("sim.3dgs.collision")
c3_mod = importlib.import_module("sim.3dgs.collect_3dgs")


def ceil_to(x, step=ROUND_UP_TO_M):
    return round(math.ceil(max(0.0, float(x)) / step - 1e-9) * step, 6)


def scan_source(root, depth=1, jitter=0.0, occ_cache=None, scene_cache=None):
    """扫描单个来源：返回集数、步数、trace 点数，以及 z − spawn.z 的分布与起飞水平漂移分布。"""
    dz, drifts = [], []
    n_ep = n_steps = n_trace = 0
    for ap in route_dirs.episode_files(root, "actions.jsonl", depth):
        ed = os.path.dirname(ap)
        sp_path = os.path.join(ed, "scene.json")
        if not os.path.isfile(sp_path):
            continue
        with open(sp_path, encoding="utf-8") as f:
            ep_sc = json.load(f)
        spawn = ep_sc.get("spawn") or {}
        if "z" not in spawn:
            continue
        spawn_z = float(spawn["z"])
        cx, cy = float(spawn.get("x", 0.0)), float(spawn.get("y", 0.0))
        if jitter > 0 and occ_cache is not None and scene_cache is not None:
            mp = os.path.join(ed, "metrics.json")
            if os.path.isfile(mp):
                with open(mp, encoding="utf-8") as f:
                    m = json.load(f)
                sname, r_idx, seed = m["scene"], int(m["route"]), int(m["seed"])
                if sname not in scene_cache:
                    sc0 = scene_mod.load_scene(os.path.join(SCENES_DIR, sname, "scene.json"))
                    scene_cache[sname] = sc0
                    occ_cache[sname] = coll_mod.from_scene(sc0, collision_model="v1")
                sc_r = dict(scene_cache[sname])
                scene_mod.apply_route(sc_r, r_idx)
                spawn_z = float(sc_r["spawn"]["z"])
                cx, cy, _ = c3_mod._spawn_for_episode(sc_r, occ_cache[sname], jitter, np.random.default_rng(seed), two_layer=False)
        n_ep += 1
        with open(ap, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                a = json.loads(line)
                n_steps += 1
                pb, pg = a["pose_before"], a["pose_gt"]
                if idx == 0:
                    drifts.append(math.hypot(float(pb["x"]) - cx, float(pb["y"]) - cy))
                dz.append(float(pb["z"]) - spawn_z)
                dz.append(float(pg["z"]) - spawn_z)
        tp = os.path.join(ed, "trace.jsonl")
        if os.path.isfile(tp):
            with open(tp, encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    n_trace += 1
                    dz.append(float(t["pos"][2]) - spawn_z)
    if not n_ep:
        return None
    arr = np.asarray(dz, dtype=np.float64)
    darr = np.asarray(drifts, dtype=np.float64)
    return {
        "episodes": n_ep,
        "steps": n_steps,
        "trace_points": n_trace,
        "dz_min": float(arr.min()),
        "dz_p1": float(np.percentile(arr, 1)),
        "dz_p50": float(np.percentile(arr, 50)),
        "dz_p99": float(np.percentile(arr, 99)),
        "dz_max": float(arr.max()),
        "drift_p50": float(np.median(darr)),
        "drift_p99": float(np.percentile(darr, 99)),
        "drift_max": float(darr.max()),
    }


def pick_constants(per_source, round_up_to=ROUND_UP_TO_M):
    raw_down = max(max(0.0, -s["dz_min"]) for s in per_source.values())
    raw_up = max(max(0.0, s["dz_max"]) for s in per_source.values())
    raw_drift = max(max(0.0, s["drift_max"]) for s in per_source.values())
    return {
        "raw_down_m": raw_down,
        "raw_up_m": raw_up,
        "raw_drift_m": raw_drift,
        "z_envelope_down_m": ceil_to(raw_down, round_up_to),
        "z_envelope_up_m": ceil_to(raw_up, round_up_to),
        "spawn_drift_margin_m": ceil_to(raw_drift, round_up_to),
    }


def main(argv):
    out_path = argv[1] if len(argv) > 1 else None
    per_source, missing = {}, []
    occ_cache, scene_cache = {}, {}
    for label, root, depth, jit in SOURCES:
        res = scan_source(root, depth=depth, jitter=jit, occ_cache=occ_cache, scene_cache=scene_cache)
        if res is None:
            missing.append(label)
            continue
        per_source[label] = res
    if not per_source:
        raise SystemExit("没有读到任何来源")
    picked = pick_constants(per_source)
    print("整集高度包络（全部步 pose_before + pose_gt + trace.jsonl）与起飞水平漂移（米）：")
    print(f"{'来源':32s} {'集':>4s} {'步':>6s} {'trace点':>8s} {'dz_min':>9s} {'dz_p1':>8s} {'dz_p99':>8s} {'dz_max':>9s} {'drift_p50':>9s} {'drift_max':>9s}")
    for label, s in per_source.items():
        print(f"{label:32s} {s['episodes']:4d} {s['steps']:6d} {s['trace_points']:8d} "
              f"{s['dz_min']:+9.4f} {s['dz_p1']:+8.4f} {s['dz_p99']:+8.4f} {s['dz_max']:+9.4f} "
              f"{s['drift_p50']:9.4f} {s['drift_max']:9.4f}")
    if missing:
        print("没读到：", ", ".join(missing))
    print(f"极值：dz_min = -{picked['raw_down_m']:.6f} m, dz_max = +{picked['raw_up_m']:.6f} m, drift_max = {picked['raw_drift_m']:.6f} m")
    print(f"向外取整到 {ROUND_UP_TO_M*1000:g} mm：Z_ENVELOPE_DOWN_M = {picked['z_envelope_down_m']:g} m, "
          f"Z_ENVELOPE_UP_M = {picked['z_envelope_up_m']:g} m, SPAWN_DRIFT_MARGIN_M = {picked['spawn_drift_margin_m']:g} m")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"round_up_to_m": ROUND_UP_TO_M, "per_source": per_source, "missing": missing, **picked},
                      f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
