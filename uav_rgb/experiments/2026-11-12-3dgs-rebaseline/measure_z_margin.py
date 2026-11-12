#!/usr/bin/env python3
"""量高度余量（任务卡 S1b 第 2 条：先量后定）：已有 3DGS 实验轨迹里，稳态飞行的 z 相对出生高度（scene.json 的 spawn.z）的分布。

统一高度口径函数取 z ± (机体半高 + 高度余量) 覆盖到的层取并（sim/3dgs/collision.py 的 HEIGHT_MARGIN_M）。余量要盖住稳态飞行 z 偏离出生高度的幅度，
所以按每个来源的稳态 z − spawn.z 的 p1 与 p99 取偏离幅度 max(|p1|, |p99|)，各来源再取最大，向上取到 ROUND_UP_TO_M 的整数倍。

只读，不渲染不用 GPU：
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
STEADY_FROM_STEP = 3        # 稳态 = 第 3 步起（步号从 0 数）：起飞过冲只占前几步，第 3 步动作前位姿位于上层的集室内 6/117、室外 0/119（审核 B 发现 2，复核脚本 step_layers2.py）
PCT = (1, 99)               # 任务卡的口径：稳态 z 的 p1 与 p99
ROUND_UP_TO_M = 0.005       # 向上取到 5 mm 的整数倍，常数不带一长串小数
# (标签, 根目录, 根目录与 r<N> 之间的层数)：K1 / W1-11 / W1-12 / K3 已跑完的 3DGS 臂（cf_sitl，hold 1.0），室内室外各一份
SOURCES = []
for _suite in ("indoor", "outdoor"):
    SOURCES += [
        (f"teacher-tol/{_suite}", f"{DATA}/exp-teacher-tol-3dgs-20261110/{_suite}/teacher-tol", 1),
        (f"swept/{_suite}", f"{DATA}/exp-teacher-clearance-3dgs-20261111/{_suite}/swept", 1),
        (f"k1-forward+route/{_suite}", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/{_suite}/forward+route", 1),
        (f"k1-rule-occ+route/{_suite}", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/{_suite}/rule-occ+route", 1),
    ]
SOURCES += [
    ("k3-rule2-occ+route/holdout", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule2-occ+route", 1),
    ("k3-rule2-depth+route/holdout", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule2-depth+route", 1),
    ("k3-rule2-occ+route/outdoor", f"{DATA}/exp-rule-v2-3dgs-20261111/outdoor/rule2-occ+route", 1),
    ("k3-rule2-depth+route/outdoor", f"{DATA}/exp-rule-v2-3dgs-20261111/outdoor/rule2-depth+route", 1),
]
# ------------------------------------------------------------------ 以下是流程

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
route_dirs = importlib.import_module("sim.3dgs.route_dirs")     # E-12：排除 r<N>.incomplete-<时刻>


def steady_dz(root, depth, steady_from=STEADY_FROM_STEP):
    """一个来源的稳态 z − spawn.z 序列（每步取 pose_gt.z，即该步动作窗口末的高度）；返回 (np.array, 集数)。"""
    dz, n_ep = [], 0
    for ap in route_dirs.episode_files(root, "actions.jsonl", depth):
        sp = os.path.join(os.path.dirname(ap), "scene.json")
        if not os.path.exists(sp):
            continue
        with open(sp, encoding="utf-8") as f:
            spawn_z = float(json.load(f)["spawn"]["z"])
        n_ep += 1
        with open(ap, encoding="utf-8") as f:
            for line in f:
                a = json.loads(line)
                if a["step"] >= steady_from:
                    dz.append(a["pose_gt"]["z"] - spawn_z)
    return np.asarray(dz, dtype=float), n_ep


def stats(dz):
    return {"n": int(len(dz)), "min": float(dz.min()), "p1": float(np.percentile(dz, PCT[0])), "p50": float(np.percentile(dz, 50)),
            "p99": float(np.percentile(dz, PCT[1])), "max": float(dz.max())}


def ceil_to(x, step):
    return round(math.ceil(x / step - 1e-9) * step, 6)


def pick_margin(per_source, round_up_to=ROUND_UP_TO_M):
    """各来源 max(|p1|, |p99|) 的最大值，向上取整到 round_up_to。"""
    need = max(max(abs(s["p1"]), abs(s["p99"])) for s in per_source.values())
    return ceil_to(need, round_up_to), need


def main(argv):
    out_path = argv[1] if len(argv) > 1 else None
    per_source, missing = {}, []
    for label, root, depth in SOURCES:
        dz, n_ep = steady_dz(root, depth)
        if not len(dz):
            missing.append(label)
            continue
        per_source[label] = {"episodes": n_ep, **stats(dz)}
    if not per_source:
        raise SystemExit("没有读到任何来源")
    margin, need = pick_margin(per_source)
    print(f"稳态 = 第 {STEADY_FROM_STEP} 步起；z − spawn.z（米），分位 p{PCT[0]} / p50 / p{PCT[1]}")
    print(f"{'来源':34s} {'集':>4s} {'步':>6s} {'min':>8s} {'p1':>8s} {'p50':>8s} {'p99':>8s} {'max':>8s}")
    for label, s in per_source.items():
        print(f"{label:34s} {s['episodes']:4d} {s['n']:6d} {s['min']:+8.4f} {s['p1']:+8.4f} {s['p50']:+8.4f} {s['p99']:+8.4f} {s['max']:+8.4f}")
    if missing:
        print("没读到（目录不存在或没有集）：", ", ".join(missing))
    print(f"各来源 max(|p1|, |p99|) 的最大值 {need:.4f} m，向上取到 {ROUND_UP_TO_M * 1000:g} mm 的整数倍：高度余量 = {margin:g} m")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"steady_from_step": STEADY_FROM_STEP, "percentiles": list(PCT), "round_up_to_m": ROUND_UP_TO_M,
                       "per_source": per_source, "missing": missing, "need_m": need, "height_margin_m": margin},
                      f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
