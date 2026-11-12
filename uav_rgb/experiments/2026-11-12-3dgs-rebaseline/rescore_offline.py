#!/usr/bin/env python3
"""离线重判已录轨迹（卡 S1b 第二轮第 7 条验收）：用仓库入库的定稿代码（sim.3dgs.collision.Occupancy）对 W1-11、W1-12、K1、K3 已录轨迹重判。

对每个来源同时报 4 列：
  1. `v1`：原口径（Occupancy(collision_model="v1").sweep(p0, p1)），验证与 metrics.json 原记录 100% 一致；
  2. `r1_v2`：第一轮未定稿 v2（格中心膨胀栅格 3×3 方核 / 0.2m 十字核 + 半宽 0.053 m 层带 + 有 trace 时扫折线）；
  3. `exact_band053`：定稿精确欧氏距离查询 + 审核 J 当时对照用的旧层带半宽 0.053 m（body_half_height 临时传 0.053）+ 有 trace 时扫折线，用于与审核 J 的表逐字对拍（如 W1-12 swept 室内 98/120，碰撞 23）；
  4. `final_v2`：定稿 v2（Occupancy(collision_model="v2") 原样调用，圆柱半高 0.028 m 无高度余量 + 精确欧氏距离 + 有 trace 时扫折线），同时统计规划包络外步数（应当为 0）。

只读，纯 CPU：
  CUDA_VISIBLE_DEVICES="" <3dgs-venv python> experiments/2026-11-12-3dgs-rebaseline/rescore_offline.py [输出 json 路径]
"""
import importlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

DATA = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
SCENES_DIR = os.path.join(DATA, "3dgs-scenes")

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

rd = importlib.import_module("sim.3dgs.route_dirs")
scene_mod = importlib.import_module("sim.3dgs.scene")
coll_mod = importlib.import_module("sim.3dgs.collision")

TARGETS = [
    ("W1-11 indoor teacher-tol", f"{DATA}/exp-teacher-tol-3dgs-20261110/indoor/teacher-tol", True),
    ("W1-12 indoor swept", f"{DATA}/exp-teacher-clearance-3dgs-20261111/indoor/swept", True),
    ("K1 indoor forward+route", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/indoor/forward+route", False),
    ("K1 indoor forward+route@0.5", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/indoor/forward+route@0.5", False),
    ("K1 indoor rule-occ(v1)", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/indoor/rule-occ+route", False),
    ("K1 indoor rule-occ(v1)@0.5", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/indoor/rule-occ+route@0.5", False),
    ("K1 indoor rule-occ+goal", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/indoor/rule-occ+goal", False),
    ("K1 indoor rule-depth(v1)", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/indoor/rule-depth+route", False),
    ("K3 formal40 rule-occ(v1)", f"{DATA}/exp-rule-v2-3dgs-20261111/formal40/rule-occ+route", False),
    ("K3 formal40 rule2-occ(v2f)", f"{DATA}/exp-rule-v2-3dgs-20261111/formal40/rule2-occ+route", False),
    ("K3 formal40 rule2-depth(v2d)", f"{DATA}/exp-rule-v2-3dgs-20261111/formal40/rule2-depth+route", False),
    ("K3 holdout forward+route", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/forward+route", False),
    ("K3 holdout rule-occ(v1)", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule-occ+route", False),
    ("K3 holdout rule2-occ(v2f)", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule2-occ+route", False),
    ("K3 holdout rule2-depth(v2d)", f"{DATA}/exp-rule-v2-3dgs-20261111/holdout/rule2-depth+route", False),
    # 室外（4 个场景：garden / bicycle 0.1 m，callinstrasse / wanderland_1MlUucug 0.2 m）
    ("W1-11 outdoor teacher-tol", f"{DATA}/exp-teacher-tol-3dgs-20261110/outdoor/teacher-tol", True),
    ("W1-12 outdoor swept", f"{DATA}/exp-teacher-clearance-3dgs-20261111/outdoor/swept", True),
    ("K1 outdoor forward+route", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/outdoor/forward+route", False),
    ("K1 outdoor rule-occ(v1)", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/outdoor/rule-occ+route", False),
    ("K1 outdoor rule-occ+goal", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/outdoor/rule-occ+goal", False),
    ("K1 outdoor rule-depth(v1)", f"{DATA}/exp-rule-oracle-3dgs-20261110/formal/outdoor/rule-depth+route", False),
    ("K3 outdoor rule2-occ(v2f)", f"{DATA}/exp-rule-v2-3dgs-20261111/outdoor/rule2-occ+route", False),
    ("K3 outdoor rule2-depth(v2d)", f"{DATA}/exp-rule-v2-3dgs-20261111/outdoor/rule2-depth+route", False),
]


def _r1_v2_sweep_hit(occ_v2, points):
    """第一轮未定稿 v2 的判定（只作对照列）：格中心膨胀栅格 occ_v2.grid + 半宽 0.053 m 层带。"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    samples = np.concatenate([occ_v2._segment_samples(a, b) for a, b in zip(pts[:-1], pts[1:])])
    if occ_v2._ground_hit(samples[:, 2]).any():
        return True
    if occ_v2.grid.size == 0:
        return False
    c = occ_v2._cell(samples)
    zlo = float(occ_v2.origin[2])
    k0 = np.maximum(np.floor((samples[:, 2] - 0.053 - zlo) / occ_v2.voxel), 0).astype(int)
    k1 = np.minimum(np.floor((samples[:, 2] + 0.053 - zlo) / occ_v2.voxel), occ_v2.shape[2] - 1).astype(int)
    idx = np.nonzero((c[:, 0] >= 0) & (c[:, 0] < occ_v2.shape[0]) & (c[:, 1] >= 0) & (c[:, 1] < occ_v2.shape[1]) & (k0 <= k1))[0]
    if not len(idx):
        return False
    ci, cj, a, b = c[idx, 0], c[idx, 1], k0[idx], k1[idx]
    for d in range(int((b - a).max()) + 1):
        lv = a + d <= b
        if occ_v2.grid[ci[lv], cj[lv], (a + d)[lv]].any():
            return True
    return False


def rescore_episode(ep_dir, occ_pack, use_trace):
    """对单集返回各口径下的 (success_stop, collided) 与规划包络越界步数。"""
    with open(os.path.join(ep_dir, "metrics.json"), encoding="utf-8") as f:
        m = json.load(f)
    orig_succ = bool(m["success_stop"])
    orig_coll = bool(m["collision"])
    reach = m["reach_step"]
    with open(os.path.join(ep_dir, "actions.jsonl"), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    tr_by_step = defaultdict(list)
    if use_trace:
        tp = os.path.join(ep_dir, "trace.jsonl")
        if os.path.isfile(tp):
            with open(tp, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        t = json.loads(line)
                        tr_by_step[int(t["step"])].append(tuple(t["pos"]))

    occ_v1, occ_b053, occ_v2 = occ_pack["v1"], occ_pack["b053"], occ_pack["v2"]
    with open(os.path.join(ep_dir, "scene.json"), encoding="utf-8") as f:
        ep_sc = json.load(f)
    pk0, pk1 = (int(x) for x in occ_v2.envelope_layer_span(float(ep_sc["spawn"]["z"])))

    fh = {"v1": None, "r1_v2": None, "exact_band053": None, "final_v2": None}
    outside_env_steps = 0

    for a in rows:
        st = int(a["step"])
        p0 = (float(a["pose_before"]["x"]), float(a["pose_before"]["y"]), float(a["pose_before"]["z"]))
        p1 = (float(a["pose_gt"]["x"]), float(a["pose_gt"]["y"]), float(a["pose_gt"]["z"]))
        pts_tr = [p0] + tr_by_step[st] if (use_trace and tr_by_step[st]) else [p0, p1]
        if pts_tr[-1] != p1:
            pts_tr.append(p1)

        # 检查本步所有高度的连续判定层是否落在规划包络层 [pk0, pk1] 内
        zs = np.array([pt[2] for pt in pts_tr], dtype=np.float64)
        sk0, sk1 = occ_v2.layer_span(zs)
        valid = sk0 <= sk1
        if valid.any() and ((sk0[valid] < pk0).any() or (sk1[valid] > pk1).any()):
            outside_env_steps += 1

        if fh["v1"] is None and occ_v1.sweep(p0, p1)[0]:
            fh["v1"] = st
        if fh["r1_v2"] is None and _r1_v2_sweep_hit(occ_v2, pts_tr):
            fh["r1_v2"] = st
        if fh["exact_band053"] is None and occ_b053.sweep_path(pts_tr)[0]:
            fh["exact_band053"] = st
        if fh["final_v2"] is None and occ_v2.sweep_path(pts_tr)[0]:
            fh["final_v2"] = st

    out = {"orig": (orig_succ, orig_coll), "outside_env_steps": outside_env_steps}
    for k, h_step in fh.items():
        succ = orig_succ and not (h_step is not None and reach is not None and h_step <= reach)
        out[k] = (bool(succ), h_step is not None)
    return out


def build_scene_packs(scene_names):
    packs = {}
    for sname in scene_names:
        sc = scene_mod.load_scene(os.path.join(SCENES_DIR, sname, "scene.json"))
        g = (scene_mod.read_gaussian_ply(sc["ply_path"])
             if sc.get("collision", {}).get("source", "gaussians") == "gaussians" else None)
        o_v1 = coll_mod.from_scene(sc, gaussians=g, collision_model="v1")
        o_v2 = coll_mod.from_scene(sc, gaussians=g, collision_model="v2")
        env = coll_mod.resolve_envelope(sc.get("collision", {}).get("envelope"))
        # 构造半高 0.053 m 的 Occupancy(collision_model="v2") 复现审核 J 的 exact_band(0.053) 列
        o_b053 = coll_mod.from_scene(
            sc, envelope={"radius_m": env["radius_m"], "half_height_m": 0.053},
            gaussians=g, collision_model="v2",
        )
        del g
        packs[sname] = {"v1": o_v1, "b053": o_b053, "v2": o_v2}
    return packs


def main(argv=None):
    argv = argv or sys.argv
    out_json = argv[1] if len(argv) > 1 else None
    t0 = time.time()
    all_scenes = ["sage3d_841150", "sage3d_840865", "sage3d_841012", "sage3d_840139", "sage3d_839992",
                  "garden", "wanderland_1MlUucug", "bicycle", "callinstrasse"]
    print(f"加载 9 个场景占据栅格...", flush=True)
    packs = build_scene_packs(all_scenes)
    print(f"占据加载完成 ({time.time() - t0:.1f} s)", flush=True)

    results = {}
    print(f"\n{'来源':28s} {'n':>4s} {'原记录(到点/碰)':>14s} {'v1重判':>10s} {'第一轮v2(方核053)':>18s} {'审核J对照(精确053)':>19s} {'定稿v2(精确028)':>17s} {'v2新碰':>6s} {'包络外步':>8s}")
    for label, root, has_tr in TARGETS:
        ep_files = rd.episode_files(root, "actions.jsonl", depth=1)
        modes = ("orig", "v1", "r1_v2", "exact_band053", "final_v2")
        succ = {m: 0 for m in modes}
        coll = {m: 0 for m in modes}
        v2_new_coll = 0
        out_env_steps = 0
        v1_diff_orig = 0
        per_scene = defaultdict(lambda: {m: [0, 0, 0] for m in modes})  # [n, succ, coll]
        for ap in ep_files:
            ed = os.path.dirname(ap)
            sname = os.path.basename(os.path.dirname(os.path.dirname(ed)))
            r = rescore_episode(ed, packs[sname], has_tr)
            if r["v1"] != r["orig"]:
                v1_diff_orig += 1
            if r["final_v2"][1] and not r["v1"][1]:
                v2_new_coll += 1
            out_env_steps += r["outside_env_steps"]
            for m in modes:
                succ[m] += int(r[m][0])
                coll[m] += int(r[m][1])
                ps = per_scene[sname][m]
                ps[0] += 1
                ps[1] += int(r[m][0])
                ps[2] += int(r[m][1])
        n = len(ep_files)
        results[label] = {
            "n": n, "has_trace": has_tr, "v1_diff_orig": v1_diff_orig,
            "v2_new_coll": v2_new_coll, "outside_env_steps": out_env_steps,
            "modes": {m: {"k_stop": succ[m], "k_coll": coll[m]} for m in modes},
            "per_scene": {s: {m: {"n": v[0], "k_stop": v[1], "k_coll": v[2]} for m, v in md.items()} for s, md in per_scene.items()},
        }
        print(f"{label:28s} {n:4d} {succ['orig']:6d}/{coll['orig']:<6d} {succ['v1']:4d}/{coll['v1']:<4d} "
              f"{succ['r1_v2']:8d}/{coll['r1_v2']:<8d} {succ['exact_band053']:9d}/{coll['exact_band053']:<8d} "
              f"{succ['final_v2']:7d}/{coll['final_v2']:<8d} {v2_new_coll:6d} {out_env_steps:8d}", flush=True)

    # 校验审核 J 对照列核心数字（任务卡第 7 条硬验收：W1-12 swept 室内 98/120，碰撞 23）
    sw_in = results["W1-12 indoor swept"]["modes"]["exact_band053"]
    if (sw_in["k_stop"], sw_in["k_coll"]) != (98, 23):
        print(f"ERROR: W1-12 indoor swept exact_band053 = {sw_in}，与审核 J 的 (98, 23) 不一致！", file=sys.stderr)
        return 1

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"wall_s": round(time.time() - t0, 2), "targets": results}, f, ensure_ascii=False, indent=2)
    print(f"\n全部完成 ({time.time() - t0:.1f} s)：v1 与原记录差异集数 = {sum(x['v1_diff_orig'] for x in results.values())}，"
          f"包络外步数总和 = {sum(x['outside_env_steps'] for x in results.values())}，"
          f"W1-12 swept 对照列 = {sw_in['k_stop']}/120 (碰撞 {sw_in['k_coll']}) 与审核 J 完全一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
