#!/usr/bin/env python3
"""碰撞代理 v2（高斯椭球占据）vs 真值几何（LiDAR 点云 / 网格）对照：覆盖率、误报率、对路线可行性的影响。

口径沿用 tmp/phase3-20261107/reports/3DGS_20261107.md 的「可见表面 0.2 m 内被代理覆盖 24%」：
  覆盖率(d) = 真值占据格里，d 米内有代理占据格的比例（漏检的反面，决定机体会不会穿过真实障碍）
  误报率(d) = 代理占据格里，d 米内没有真值占据格的比例（幽灵墙，决定教师会不会被雾围死）
两者都在飞行层（scene.collision.z_slice）的 2D 投影上算，并只统计相机轨迹 NEAR_M 内的格（3DGS 只在拍摄
轨迹附近有意义；远处两边都是垃圾）。路线影响：各路线在两种占据上 route_check，再把代理 A* 路径拿到真值
上 sweep（代理没看见的真实障碍 = 穿墙）、真值路径拿到代理上 sweep（代理多出来的幽灵墙 = 误挡）。
可选 --sweep 扫 opacity_min × sigma_k，给 collision.py 默认值提供依据。

运行（3dgs-venv，仓库根）：
  python tools/collision_truth_compare.py --scene $DATA_ROOT/uav_rgb/3dgs-scenes/wanderland_1MlUucug/scene.json \
      --out $T/agent4-scene-wanderland/compare [--sweep]
"""
import argparse
import importlib
import json
import math
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

NEAR_M = 6.0                    # 只统计离相机轨迹 6 m 内的格：人行道宽 3–5 m，再远是街对面/店内
DISTS_M = (0.0, 0.2, 0.4)       # 覆盖/误报的距离档（0 = 同一格）
SWEEP_OPACITY = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7)
SWEEP_SIGMA = (1.0, 1.5, 2.0, 3.0)


def occupied_xy(occ):
    """原始占据（未膨胀）的 2D 投影格中心 [M,2]。"""
    if occ.raw.size == 0:
        return np.zeros((0, 2))
    ii, jj = np.nonzero(occ.raw.any(axis=2))
    return occ.origin[:2] + (np.stack([ii, jj], 1) + 0.5) * occ.voxel


def nearest_dist(a, b):
    """a 中每点到 b 最近点的距离（b 空则 inf）；用 scipy cKDTree（3dgs-venv 有）。"""
    if len(b) == 0:
        return np.full(len(a), np.inf)
    from scipy.spatial import cKDTree
    d, _ = cKDTree(b).query(a)
    return d


def compare_grids(truth, proxy, cams_xy):
    t_xy, p_xy = occupied_xy(truth), occupied_xy(proxy)
    near_t = nearest_dist(t_xy, cams_xy) <= NEAR_M
    near_p = nearest_dist(p_xy, cams_xy) <= NEAR_M
    d_tp = nearest_dist(t_xy[near_t], p_xy)          # 真值格 -> 最近代理格
    d_pt = nearest_dist(p_xy[near_p], t_xy)          # 代理格 -> 最近真值格
    tol = 0.5 * truth.voxel + 1e-6                   # 同一格：中心距 < 半格
    out = {"n_truth_cells": int(len(t_xy)), "n_proxy_cells": int(len(p_xy)),
           "n_truth_near": int(near_t.sum()), "n_proxy_near": int(near_p.sum()), "near_m": NEAR_M}
    for d in DISTS_M:
        thr = tol if d == 0 else d + tol
        out[f"coverage@{d}"] = round(float(np.mean(d_tp <= thr)), 4) if len(d_tp) else None
        out[f"false_alarm@{d}"] = round(float(np.mean(d_pt > thr)), 4) if len(d_pt) else None
    return out, (t_xy, p_xy, near_t, near_p, d_tp, d_pt)


def route_impact(scene, truth, proxy, planner_mod, scene_mod):
    pl_t, pl_p = planner_mod.OccPlanner(truth), planner_mod.OccPlanner(proxy)
    rows = []
    for i, _r in enumerate(scene.get("routes") or [None]):
        sc = scene_mod.apply_route(dict(scene), i) if scene.get("routes") else scene
        z = float(sc["spawn"].get("z", 2.0))
        ct = planner_mod.route_check(pl_t, sc["spawn"], sc["waypoint"])
        cp = planner_mod.route_check(pl_p, sc["spawn"], sc["waypoint"])
        row = {"route": i, "name": (sc.get("route") or {}).get("name"),
               "truth": {k: ct.get(k) for k in ("feasible", "clear_ok", "path_m", "min_clearance_m")},
               "proxy": {k: cp.get(k) for k in ("feasible", "clear_ok", "path_m", "min_clearance_m")}}
        for name, path, other in (("proxy_path_on_truth", cp.get("path"), truth), ("truth_path_on_proxy", ct.get("path"), proxy)):
            if not path:
                row[name] = None
                continue
            hits = 0
            first = None
            for (ax, ay), (bx, by) in zip(path, path[1:]):
                hit, pt = other.sweep((ax, ay, z), (bx, by, z))
                if hit:
                    hits += 1
                    first = first or [round(v, 2) for v in pt]
            row[name] = {"segments": len(path) - 1, "hit_segments": hits, "first_hit": first,
                         "min_clearance_m": round(min(other.clearance(px, py, z) for px, py in path), 3)}
        rows.append(row)
    return rows


def draw_overlay(path, truth, proxy, cams_xy, parts, px_per_m=20, margin_m=6.0):
    """黑 = 两边都有；蓝 = 只有真值（代理漏）；红 = 只有代理（幽灵）；浅灰 = 轨迹 NEAR_M 外。"""
    from PIL import Image, ImageDraw
    t_xy, p_xy, near_t, near_p, d_tp, d_pt = parts
    x0, x1 = cams_xy[:, 0].min() - margin_m, cams_xy[:, 0].max() + margin_m
    y0, y1 = cams_xy[:, 1].min() - margin_m, cams_xy[:, 1].max() + margin_m
    w, h = int((x1 - x0) * px_per_m) + 1, int((y1 - y0) * px_per_m) + 1
    im = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(im)
    v = truth.voxel * px_per_m

    def box(xy, color):
        a, b = (xy[0] - truth.voxel / 2 - x0) * px_per_m, (y1 - xy[1] - truth.voxel / 2) * px_per_m
        d.rectangle([a, b, a + v, b + v], fill=color)

    thr = 0.2 + 0.5 * truth.voxel
    for xy in t_xy[~near_t]:
        box(xy, (215, 215, 215))
    for xy in p_xy[~near_p]:
        box(xy, (235, 215, 215))
    for xy, dd in zip(t_xy[near_t], d_tp):
        box(xy, (0, 0, 0) if dd <= thr else (0, 90, 255))
    for xy, dd in zip(p_xy[near_p], d_pt):
        if dd > thr:
            box(xy, (255, 0, 0))
    d.line([((x - x0) * px_per_m, (y1 - y) * px_per_m) for x, y in cams_xy], fill=(0, 170, 0), width=1)
    for gx in np.arange(math.ceil(x0 / 2) * 2, x1, 2):
        a = (gx - x0) * px_per_m
        d.line([(a, 0), (a, h)], fill=(200, 200, 255) if gx % 10 else (120, 120, 255))
        d.text((a + 2, 2), f"{gx:.0f}", fill=(0, 0, 160))
    for gy in np.arange(math.ceil(y0 / 2) * 2, y1, 2):
        b = (y1 - gy) * px_per_m
        d.line([(0, b), (w, b)], fill=(200, 200, 255) if gy % 10 else (120, 120, 255))
        d.text((2, b + 2), f"{gy:.0f}", fill=(0, 0, 160))
    d.text((6, h - 14), "black=both  blue=truth only (proxy miss)  red=proxy only (ghost)  grey=outside NEAR_M", fill=(0, 0, 0))
    im.save(path)


def camera_xy_nav(scene, align_json=None):
    """相机轨迹 nav 系 xy：优先 align.json（import 步写的 cameras_nav），否则用 scene.routes 的 spawn/waypoint 连线。"""
    if align_json and os.path.isfile(align_json):
        c = np.array(json.load(open(align_json, encoding="utf-8"))["cameras_nav"])
        return c[:, :2]
    pts = []
    for r in scene.get("routes") or [scene]:
        s, w = r["spawn"], r["waypoint"]
        for t in np.linspace(0, 1, 20):
            pts.append([s["x"] + t * (w["x"] - s["x"]), s["y"] + t * (w["y"] - s["y"])])
    return np.array(pts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--truth-source", choices=("points", "mesh"), default="points")
    p.add_argument("--align-json", default=None, help="scene_import 的 align.json（相机轨迹 nav 系）")
    p.add_argument("--sweep", action="store_true", help="扫 opacity_min × sigma_k")
    args = p.parse_args(argv)
    scene_mod = importlib.import_module("sim.3dgs.scene")
    collision = importlib.import_module("sim.3dgs.collision")
    planner_mod = importlib.import_module("sim.3dgs.planner")
    os.makedirs(args.out, exist_ok=True)
    scene = scene_mod.load_scene(args.scene)
    if not scene.get("collision", {}).get("geometry"):
        raise SystemExit(f"{args.scene}: collision.geometry 缺失，没有真值几何可对照")
    cams_xy = camera_xy_nav(scene, args.align_json or os.path.join(os.path.dirname(args.out), "import", "align.json"))
    t0 = time.time()
    truth = collision.from_scene(scene, source=args.truth_source)
    t_truth = time.time() - t0
    g = scene_mod.read_gaussian_ply(scene["ply_path"])
    t0 = time.time()
    proxy = collision.from_scene(scene, source="gaussians", gaussians=g)
    t_proxy = time.time() - t0
    print(f"truth({args.truth_source}) {truth.summary()['n_occupied_cells']} cells {t_truth:.1f}s; "
          f"proxy(v2 defaults) {proxy.summary()['n_occupied_cells']} cells {t_proxy:.1f}s", flush=True)
    stats, parts = compare_grids(truth, proxy, cams_xy)
    print("grid:", json.dumps(stats), flush=True)
    routes = route_impact(scene, truth, proxy, planner_mod, scene_mod)
    for r in routes:
        print(f"route {r['route']} {r['name']}: truth feasible={r['truth']['feasible']} clear={r['truth']['min_clearance_m']} | "
              f"proxy feasible={r['proxy']['feasible']} clear={r['proxy']['min_clearance_m']} | "
              f"proxy path on truth hits {r['proxy_path_on_truth'] and r['proxy_path_on_truth']['hit_segments']} / "
              f"truth path on proxy hits {r['truth_path_on_proxy'] and r['truth_path_on_proxy']['hit_segments']}", flush=True)
    draw_overlay(os.path.join(args.out, "overlay_truth_vs_proxy.png"), truth, proxy, cams_xy, parts)
    result = {"scene": scene["name"], "truth": truth.summary(), "proxy": proxy.summary(), "grid": stats, "routes": routes,
              "build_s": {"truth": round(t_truth, 1), "proxy": round(t_proxy, 1)}}
    if args.sweep:
        rows = []
        c = dict(scene.get("collision", {}))
        for om in SWEEP_OPACITY:
            for sk in SWEEP_SIGMA:
                sc = dict(scene, collision=dict(c, opacity_min=om, sigma_k=sk))
                occ = collision.from_scene(sc, source="gaussians", gaussians=g)
                st, _ = compare_grids(truth, occ, cams_xy)
                pl = planner_mod.OccPlanner(occ)
                feas = 0
                for i in range(len(scene.get("routes") or [None])):
                    s2 = scene_mod.apply_route(dict(scene), i) if scene.get("routes") else scene
                    ck = planner_mod.route_check(pl, s2["spawn"], s2["waypoint"])
                    feas += int(bool(ck.get("feasible") and ck.get("clear_ok")))
                row = {"opacity_min": om, "sigma_k": sk, "n_proxy_near": st["n_proxy_near"],
                       "coverage@0.2": st["coverage@0.2"], "false_alarm@0.2": st["false_alarm@0.2"],
                       "coverage@0.4": st["coverage@0.4"], "routes_ok": feas}
                rows.append(row)
                print("sweep", json.dumps(row), flush=True)
        result["sweep"] = rows
    with open(os.path.join(args.out, "compare.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    print("wrote", os.path.join(args.out, "compare.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
