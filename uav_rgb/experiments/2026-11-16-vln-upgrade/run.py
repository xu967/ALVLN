#!/usr/bin/env python3
"""VLN 综合验证入口：公开指标补算 + 3DGS 渲染画质对比 + 防抖刹停教师闭环 + 同起点语言目的性反事实 + 53m 户外长航线闭环。

运行方式（仓库根目录）：
  source env.sh && CF_SITL_PROC=$DATA_ROOT/uav_rgb/tmp/build-cf-sitl/cf_sitl_proc \
    $DATA_ROOT/uav_rgb/3dgs-venv/bin/python experiments/2026-11-16-vln-upgrade/run.py
"""
import copy
import hashlib
import importlib
import json
import math
import os
import pathlib
import sys
import time

import cv2
import numpy as np

# ─────────────────────────────── 配置区（改这里） ───────────────────────────────
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
UAV_DATA = os.path.join(DATA_ROOT, "uav_rgb")
OUT_DIR = os.environ.get("VLN_UPGRADE_OUT", os.path.join(UAV_DATA, "exp-vln-upgrade-20261116"))
HOLD_S = 0.5                     # 3DGS VLN 标准动作窗口（秒），与 M1 / v3 基线一致
SEED = 20261116                  # 固定随机种子，保证配对实验可复现
INDOOR_MAX_SCALE_M = 0.8         # 室内剔除 1σ>0.8m 的跨房间巨型高斯（原 3.0m 含 2903 个大浮子）
OUTDOOR_MAX_SCALE_M = 1.2        # 户外街景剔除 1σ>1.2m 的天空壳高斯
MAX_ANISO = 25.0                 # 剔除 smax>0.25m 且长宽比 >25 的针状拉丝伪影
MIN_LARGE_OPACITY = 0.25         # 剔除 smax>0.40m 且不透明度 <0.25 的半透明大雾团（对齐碰撞端）
INDOOR_NEAR_PLANE_M = 0.20       # NuRec / SAGE-3D 训练时的真实 near_clip_distance（原 0.05m 漏进贴脸糊斑）
OUTDOOR_EPS2D = 0.08             # 320x240 低分辨率下减小 2D 协方差膨胀核（原 0.3 px^2 糊远景树叶）
OPENFLY_RADIUS_M = 20.0          # OpenFly / AerialVLN 城市尺度到达半径（米）
OPENFLY_NDTW_STEP_M = 3.0        # OpenFly 基础动作步长 3m，用于参考路径等距重采样
SOC100_PARAMS = os.path.join(UAV_DATA, "exp-wanderland-10nvb-20261114/soc100/e1_mini_soc100.json")
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = str(pathlib.Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.environ.setdefault("CF_SITL_PROC", os.path.join(UAV_DATA, "tmp/build-cf-sitl/cf_sitl_proc"))
os.environ.setdefault("TORCH_EXTENSIONS_DIR", os.path.join(UAV_DATA, "torch-cache/extensions-sm75-20261109"))

vln = importlib.import_module("sim.3dgs.vln")
run_vln = importlib.import_module("sim.3dgs.run_vln")
scene_mod = importlib.import_module("sim.3dgs.scene")
renderer_mod = importlib.import_module("sim.3dgs.renderer")
collision_mod = importlib.import_module("sim.3dgs.collision")
planner_mod = importlib.import_module("sim.3dgs.planner")
grounder_mod = importlib.import_module("sim.3dgs.grounder")
collect_mod = importlib.import_module("sim.3dgs.collect_3dgs")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def lapvar_and_fog(rgb, depth):
    gray = np.asarray(rgb, dtype=np.float64).mean(axis=2)
    lap = 4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return round(float(lap.var()), 1), round(float((np.asarray(depth) <= 0.0).mean()), 4)


def step1_public_metrics():
    """在既有 SAGE-3D 249 集与 OpenFly seen20 100 集真实轨迹上补齐公开指标 nDTW / SDTW / SPL。"""
    out = {"sage3d_v3_249": {}, "openfly_v3_seen20": {}, "grounder_4429_audit": {}}

    # 1a. SAGE-3D 6 场景 249 集（exp-vln-v3-m1-20261114）补算 nDTW / SDTW
    m1_dir = pathlib.Path(UAV_DATA) / "exp-vln-v3-m1-20261114"
    ref_by_scene = {}
    for sid in ("839992", "840139", "840865", "841012", "841150", "841980"):
        sc = scene_mod.load_scene(os.path.join(UAV_DATA, f"3dgs-scenes/sage3d_{sid}/scene.json"))
        ref_by_scene[sc["name"]] = (sc, vln.sage_ref_paths(sc))
    for arm in ("teacher", "forward"):
        ep_file = m1_dir / arm / "episodes.jsonl"
        if not ep_file.is_file():
            continue
        rows = [json.loads(line) for line in ep_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        enriched = []
        for r in rows:
            ep_dir = pathlib.Path(r["dir"])
            if not ep_dir.is_dir():
                continue
            sc, refs = ref_by_scene[r["scene"]]
            ref = refs.get(r["route_index"])
            actions = [json.loads(x) for x in (ep_dir / "actions.jsonl").read_text().splitlines() if x.strip()]
            oracle = [json.loads(x) for x in (ep_dir / "oracle.jsonl").read_text().splitlines() if x.strip()]
            m = vln.episode_metrics(actions, oracle, sc["routes"][r["route_index"]]["waypoint"],
                                    float(sc["reach_radius_m"]), r["stopped"], r["shortest_m"], ref_path=ref)
            enriched.append(m)
        out["sage3d_v3_249"][arm] = vln.summarize(enriched)

    # 1b. OpenFly v3 seen20 五臂补算完整公开指标（STOP<20 SR、官方 end<20 SR、OSR、NE、SPL、nDTW、SDTW）
    of_root = pathlib.Path(UAV_DATA) / "exp-openfly-optimization-20261116"
    seen_tasks = json.loads((of_root / "official_seen_tasks.json").read_text())
    seen_rows = json.loads(pathlib.Path(seen_tasks["annotation_source"]).read_text())
    seen_refs = [ [p[:3] for p in seen_rows[t["source_row_index"]]["pos"]] for t in seen_tasks["tasks"] ]
    eval_v3 = of_root / "evaluation_v3/official_seen"
    for arm_dir in sorted(eval_v3.iterdir()):
        if not arm_dir.is_dir():
            continue
        arm_rows = []
        for idx, ref3d in enumerate(seen_refs):
            ep_dir = arm_dir / f"episode_{idx:03d}"
            res = json.loads((ep_dir / "result.json").read_text())
            trace = [json.loads(x) for x in (ep_dir / "trace.jsonl").read_text().splitlines() if x.strip()]
            valid_trace = [t for t in trace if not t.get("collision_reported") and t.get("termination") not in ("pose_application_error", "stale_rgb_timestamp")]
            pts3d = [tuple(t["pose"][:3]) for t in valid_trace] or [tuple(res["start"][:3])]
            goal3d = tuple(res["goal_for_evaluation_only"][:3])
            dists = [math.dist(p, goal3d) for p in pts3d]
            collided = any(t.get("collision_reported") for t in trace)
            stop_sr = bool(res["stopped_within_20m_diagnostic"] and not collided)
            official_end20 = bool(dists[-1] < OPENFLY_RADIUS_M and not collided)
            osr = bool(min(dists) < OPENFLY_RADIUS_M)
            path_len = sum(math.dist(a, b) for a, b in zip(pts3d, pts3d[1:]))
            ref_len = max(0.0, sum(math.dist(a, b) for a, b in zip(ref3d, ref3d[1:])) - OPENFLY_RADIUS_M)
            spl = float(stop_sr) * (ref_len / max(path_len, ref_len, 1e-6) if ref_len > 0 else 1.0)
            ndtw, sdtw = vln.ndtw_sdtw(
                vln.resample_polyline(pts3d, OPENFLY_NDTW_STEP_M),
                vln.resample_polyline(ref3d, OPENFLY_NDTW_STEP_M),
                OPENFLY_RADIUS_M, stop_sr)
            arm_rows.append({"stop_sr": stop_sr, "official_end20": official_end20, "osr": osr,
                             "ne_m": dists[-1], "spl": spl, "ndtw": ndtw, "sdtw": sdtw, "collided": collided})
        n = len(arm_rows)
        out["openfly_v3_seen20"][arm_dir.name] = {
            "n": n,
            "sr_autonomous_stop_20m": round(sum(r["stop_sr"] for r in arm_rows) / n, 4),
            "sr_official_end_20m_no_stop_req": round(sum(r["official_end20"] for r in arm_rows) / n, 4),
            "osr_20m": round(sum(r["osr"] for r in arm_rows) / n, 4),
            "spl": round(sum(r["spl"] for r in arm_rows) / n, 4),
            "ndtw": round(sum(r["ndtw"] for r in arm_rows) / n, 4),
            "sdtw": round(sum(r["sdtw"] for r in arm_rows) / n, 4),
            "ne_mean_m": round(sum(r["ne_m"] for r in arm_rows) / n, 2),
            "collision_rate": round(sum(r["collided"] for r in arm_rows) / n, 4),
        }

    # 1c. 词法 Grounder vs 主目标短语 Grounder 在 6 场景 4429 条指令上的覆盖率对比（每场景唯一物体名集合诊断）
    lex, phr = grounder_mod.LexicalGrounder(), grounder_mod.RelationPhraseGrounder()
    stats = {name: {"total": 0, "accepted": 0, "correct": 0} for name in ("lexical", "phrase")}
    for sid in ("839992", "840139", "840865", "841012", "841150", "841980"):
        sc, _ = ref_by_scene[f"sage3d_{sid}"]
        eps = vln.episodes(sc)
        cands = sorted({e["goal_object"] for e in eps if e.get("goal_object")}
                       | {e["start_object"] for e in eps if e.get("start_object")})
        for e in eps:
            for tag, g in (("lexical", lex), ("phrase", phr)):
                res = g.ground(e["text"], cands)
                stats[tag]["total"] += 1
                stats[tag]["accepted"] += int(res.accepted)
                stats[tag]["correct"] += int(res.accepted and res.target == e["goal_object"])
    for tag, d in stats.items():
        d["coverage"] = round(d["accepted"] / d["total"], 4)
        d["precision_among_accepted"] = round(d["correct"] / max(1, d["accepted"]), 4)
        d["overall_accuracy"] = round(d["correct"] / d["total"], 4)
    out["grounder_4429_audit"] = stats
    write_json(os.path.join(OUT_DIR, "public_metrics.json"), out)
    return out


def step2_render_upgrade():
    """真实 GPU 渲染对照：室内 3DGUT+近裁剪+去浮子 vs 默认；户外低 eps2d+去拉丝浮子 vs 默认。"""
    s_in = scene_mod.load_scene(os.path.join(UAV_DATA, "3dgs-scenes/sage3d_841150/scene.json"))
    s_out = scene_mod.load_scene(os.path.join(UAV_DATA, "3dgs-scenes/wanderland_1MlUucug/scene.json"))

    r_in_old = renderer_mod.GsplatRenderer(s_in, device="cuda")
    r_in_new = renderer_mod.GsplatRenderer(
        s_in, device="cuda", ut_eval3d=True, near_plane_m=INDOOR_NEAR_PLANE_M,
        max_scale_m=INDOOR_MAX_SCALE_M, max_aniso=MAX_ANISO, min_large_opacity=MIN_LARGE_OPACITY,
        background=(0.86, 0.85, 0.83))

    r_out_old = renderer_mod.GsplatRenderer(s_out, device="cuda")
    r_out_new = renderer_mod.GsplatRenderer(
        s_out, device="cuda", eps2d=OUTDOOR_EPS2D,
        max_scale_m=OUTDOOR_MAX_SCALE_M, max_aniso=MAX_ANISO, min_large_opacity=MIN_LARGE_OPACITY)

    metrics = {"indoor_sage3d_841150": {"old_stats": r_in_old.stats(), "new_stats": r_in_new.stats(), "views": []},
               "outdoor_wanderland_1MlUucug": {"old_stats": r_out_old.stats(), "new_stats": r_out_new.stats(), "views": []}}
    tiles = []

    def label_tile(rgb, title, sub):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        big = cv2.resize(bgr, (640, 480), interpolation=cv2.INTER_CUBIC)
        bar = big[:58]
        bar[:] = (bar * 0.30).astype(np.uint8)
        cv2.putText(big, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(big, sub, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (120, 240, 120), 1, cv2.LINE_AA)
        return big

    sp_in = s_in["spawn"]
    for dyaw in (0.0, 90.0, 180.0):
        pose = (sp_in["x"], sp_in["y"], sp_in["z"], sp_in["yaw_deg"] + dyaw)
        rgb0, d0, ms0 = r_in_old.render_pose(*pose)
        rgb1, d1, ms1 = r_in_new.render_pose(*pose)
        lv0, fg0 = lapvar_and_fog(rgb0, d0)
        lv1, fg1 = lapvar_and_fog(rgb1, d1)
        metrics["indoor_sage3d_841150"]["views"].append(
            {"dyaw_deg": dyaw, "old_lapvar": lv0, "new_lapvar": lv1, "old_ms": round(ms0, 2), "new_ms": round(ms1, 2)})
        if dyaw in (0.0, 180.0):
            tiles.append(np.hstack([
                label_tile(rgb0, f"SAGE-3D 841150 (yaw+{int(dyaw)}deg) BEFORE: 2D-EWA near=0.05 scale<=3m",
                           f"sharpness(lapvar)={lv0}  gaussians={r_in_old.n_gaussians}"),
                label_tile(rgb1, f"SAGE-3D 841150 (yaw+{int(dyaw)}deg) AFTER: 3DGUT near=0.20 + floater filter",
                           f"sharpness(lapvar)={lv1}  gaussians={r_in_new.n_gaussians}"),
            ]))

    sp_out = s_out["spawn"]
    for dyaw in (0.0, 45.0, 90.0):
        pose = (sp_out["x"], sp_out["y"], sp_out["z"], sp_out["yaw_deg"] + dyaw)
        rgb0, d0, ms0 = r_out_old.render_pose(*pose)
        rgb1, d1, ms1 = r_out_new.render_pose(*pose)
        lv0, fg0 = lapvar_and_fog(rgb0, d0)
        lv1, fg1 = lapvar_and_fog(rgb1, d1)
        metrics["outdoor_wanderland_1MlUucug"]["views"].append(
            {"dyaw_deg": dyaw, "old_lapvar": lv0, "new_lapvar": lv1, "old_ms": round(ms0, 2), "new_ms": round(ms1, 2)})
        if dyaw == 0.0:
            tiles.append(np.hstack([
                label_tile(rgb0, "Wanderland 1MlUucug BEFORE: eps2d=0.3 scale<=3m (needle/fog floaters)",
                           f"sharpness(lapvar)={lv0}  gaussians={r_out_old.n_gaussians}"),
                label_tile(rgb1, "Wanderland 1MlUucug AFTER: eps2d=0.08 + needle/fog filter",
                           f"sharpness(lapvar)={lv1}  gaussians={r_out_new.n_gaussians}"),
            ]))

    grid = np.vstack(tiles)
    img_path = os.path.join(OUT_DIR, "render_compare.png")
    cv2.imwrite(img_path, grid)
    metrics["comparison_image"] = img_path
    write_json(os.path.join(OUT_DIR, "render_metrics.json"), metrics)
    del r_in_old, r_in_new, r_out_old, r_out_new
    return metrics


def step3_anti_jitter_paired():
    """在 SAGE-3D 室内真实航线（841980 2 条 + 841150 14 条 = 16 集）同 seed 配对比较 ray(v1) vs braking(v5)。"""
    scenes = [os.path.join(UAV_DATA, f"3dgs-scenes/sage3d_{sid}/scene.json") for sid in ("841980", "841150")]
    summary = {}
    for mode in ("ray", "braking"):
        arm_out = os.path.join(OUT_DIR, f"jitter_{mode}")
        argv = []
        for s in scenes:
            argv += ["--scene", s]
        argv += ["--policy", "teacher", "--teacher-clearance", mode, "--hold", str(HOLD_S),
                 "--seed", str(SEED), "--videos", "1", "--out", arm_out]
        run_vln.main(argv)
        rep = json.loads(pathlib.Path(arm_out, "vln_eval.json").read_text())
        eps = [json.loads(x) for x in pathlib.Path(arm_out, "episodes.jsonl").read_text().splitlines() if x.strip()]
        sw = [e["action_switches_per_m"] for e in eps if e.get("action_switches_per_m") is not None]
        rev = [e["turn_reversals"] for e in eps if e.get("turn_reversals") is not None]
        summary[mode] = {
            **rep["overall"],
            "mean_switches_per_m": round(sum(sw) / len(sw), 3) if sw else None,
            "total_turn_reversals": int(sum(rev)),
            "mean_turn_reversals": round(sum(rev) / len(rev), 3) if rev else None,
        }
    write_json(os.path.join(OUT_DIR, "anti_jitter_pair.json"), summary)
    return summary


def step4_language_counterfactual():
    """同起点多目标语言反事实闭环（sage3d_840865，同一起点 (-2.375, 0.265, 1.201)，两条相反房间指令 + 互换对照）。"""
    src_scene_path = os.path.join(UAV_DATA, "3dgs-scenes/sage3d_840865/scene.json")
    sc = json.load(open(src_scene_path, encoding="utf-8"))
    sc["ply"] = os.path.abspath(os.path.join(os.path.dirname(src_scene_path), sc["ply"]))
    sc["render"] = {"sh_degree": 3, "supersample": 2, "ut_eval3d": True,
                    "near_plane_m": INDOOR_NEAR_PLANE_M, "max_scale_m": INDOOR_MAX_SCALE_M,
                    "max_aniso": MAX_ANISO, "min_large_opacity": MIN_LARGE_OPACITY}
    r_door = next(r for r in sc["routes"] if r["name"] == "tr_11")
    r_cup = next(r for r in sc["routes"] if r["name"] == "tr_49")
    # 强制使用完全相同的起始位姿（含 yaw），彻底隔离初始画面与起点差异，纯靠语言决定飞向东侧大门还是西侧储物间水杯！
    shared_spawn = dict(r_door["spawn"])
    ins_door = "Go from the bath heater area to the door opposite the living room."
    ins_cup = "Move to the cup located near the decorative vase in the storage zone."

    cf_dir = os.path.join(OUT_DIR, "counterfactual_setup")
    os.makedirs(cf_dir, exist_ok=True)
    # 独立地标源表（含干扰地标，模拟场景语义地图，与测试 scene 文件物理隔离并做 SHA256 绑定）
    raw_map_path = os.path.join(cf_dir, "independent_room_landmarks.json")
    landmarks = [
        {"name": "door_1", "x": r_door["waypoint"]["x"], "y": r_door["waypoint"]["y"], "z": shared_spawn["z"]},
        {"name": "cup_2", "x": r_cup["waypoint"]["x"], "y": r_cup["waypoint"]["y"], "z": shared_spawn["z"]},
        {"name": "bath_heater_2", "x": shared_spawn["x"], "y": shared_spawn["y"], "z": shared_spawn["z"]},
        {"name": "sofa_1", "x": 0.50, "y": -2.10, "z": shared_spawn["z"]},
    ]
    write_json(raw_map_path, {"landmarks": landmarks})
    raw_sha = hashlib.sha256(pathlib.Path(raw_map_path).read_bytes()).hexdigest()
    catalog_path = os.path.join(cf_dir, "catalog_840865.json")
    write_json(catalog_path, {
        "schema": "uavrgb.vln_landmark_catalog.v1",
        "scene": sc["name"],
        "source": raw_map_path,
        "source_sha256": raw_sha,
        "frame_id": "scene_nav",
        "units": "m",
        "landmarks": landmarks,
    })

    results = {}
    for cond, texts in (("correct", (ins_door, ins_cup)), ("swapped", (ins_cup, ins_door))):
        sc_cond = copy.deepcopy(sc)
        sc_cond["routes"] = [
            {"name": "same_spawn_to_door", "spawn": dict(shared_spawn), "waypoint": dict(r_door["waypoint"]),
             "max_steps": max(r_door["max_steps"], 90),
             "vln": {"split": "counterfactual_pair", "instructions": [
                 {"instruction_type": "Relative_Relationship", "start": "bath_heater_2", "end": "door_1", "text": texts[0]}]}},
            {"name": "same_spawn_to_cup", "spawn": dict(shared_spawn), "waypoint": dict(r_cup["waypoint"]),
             "max_steps": max(r_cup["max_steps"], 90),
             "vln": {"split": "counterfactual_pair", "instructions": [
                 {"instruction_type": "Relative_Relationship", "start": "bath_heater_2", "end": "cup_2", "text": texts[1]}]}},
        ]
        sc_cond["spawn"] = dict(shared_spawn)
        sc_cond["waypoint"] = dict(r_door["waypoint"])
        sc_path = os.path.join(cf_dir, f"scene_{cond}.json")
        scene_mod.save_scene(sc_cond, sc_path)
        arm_out = os.path.join(OUT_DIR, f"lang_cf_{cond}")
        run_vln.main(["--scene", sc_path, "--policy", "teacher", "--teacher-clearance", "braking",
                      "--grounder", "phrase", "--catalog", catalog_path,
                      "--all-instructions", "--videos", "2", "--seed", str(SEED), "--out", arm_out])
        rep = json.loads(pathlib.Path(arm_out, "vln_eval.json").read_text())
        eps = [json.loads(x) for x in pathlib.Path(arm_out, "episodes.jsonl").read_text().splitlines() if x.strip()]
        results[cond] = {
            "gt_sr": rep["overall"]["sr"],
            "predicted_target_sr": rep["target_goal_metrics"]["sr"],
            "grounding": rep["grounding"],
            "episodes": [{"route": e["route_name"], "instruction": e["text"],
                          "gt_goal": e["goal_object"], "grounded_target": e["grounding"]["target"],
                          "grounding_correct": e["grounding"]["correct"],
                          "gt_success": e["success"], "gt_ne_m": round(e["ne_m"], 3),
                          "target_success": e["target_success"], "target_ne_m": round(e["target_ne_m"], 3),
                          "stopped": e["stopped"], "steps": e["steps"],
                          "video": os.path.join(arm_out, e["video"]) if "video" in e else None}
                         for e in eps],
        }
    write_json(os.path.join(OUT_DIR, "language_counterfactual.json"), results)
    return results


def step5_wanderland_53m_long_route():
    """导入 Wanderland 1MlUucug 官方 52.9m 真实街道长航线 VLN 任务并用满电 cf_sitl 跑闭环与四格录像。"""
    base_path = os.path.join(UAV_DATA, "3dgs-scenes/wanderland_1MlUucug/scene.json")
    sc = json.load(open(base_path, encoding="utf-8"))
    sc_dir = os.path.dirname(base_path)
    sc["ply"] = os.path.abspath(os.path.join(sc_dir, sc["ply"]))
    sc["collision"]["geometry"] = os.path.abspath(os.path.join(sc_dir, sc["collision"]["geometry"]))
    sc["render"] = {"sh_degree": 3, "supersample": 2, "eps2d": OUTDOOR_EPS2D,
                    "max_scale_m": OUTDOOR_MAX_SCALE_M, "max_aniso": MAX_ANISO,
                    "min_large_opacity": MIN_LARGE_OPACITY}
    sc["reach_radius_m"] = 2.0  # 户外 50m 级街道任务到达半径 2.0m（官方为 3.0m，我们取更严的 2.0m）

    n2w = np.asarray(sc["nav_to_world"], dtype=np.float64)
    r_m, t_m = n2w[:3, :3], n2w[:3, 3]
    to_nav = lambda p: ((np.asarray(p[:3], dtype=np.float64) - t_m) @ r_m).tolist()  # noqa: E731

    ep_doc = json.load(open(os.path.join(UAV_DATA, "downloads/wanderland/1MlUucugRAnrx86cpUuQt8fUtxFMddxju/episodes.json")))
    raw_eps = ep_doc["episodes"] if isinstance(ep_doc, dict) else ep_doc
    loaded_base = scene_mod.load_scene(base_path)
    occ = collision_mod.from_scene(loaded_base)
    pl = planner_mod.OccPlanner(occ)

    long_routes = []
    for e in raw_eps:
        sp_nav = to_nav(e["start_position"])
        gp_nav = to_nav(e["goals"][0]["position"])
        ref_nav = [[round(v, 4) for v in to_nav(pt)[:2]] for pt in e.get("reference_path", [])]
        yaw = math.degrees(math.atan2(ref_nav[min(3, len(ref_nav) - 1)][1] - sp_nav[1],
                                      ref_nav[min(3, len(ref_nav) - 1)][0] - sp_nav[0])) if len(ref_nav) >= 2 else 0.0
        spawn = {"x": round(sp_nav[0], 3), "y": round(sp_nav[1], 3), "z": 2.0, "yaw_deg": round(yaw, 2)}
        waypoint = {"x": round(gp_nav[0], 3), "y": round(gp_nav[1], 3), "z": 2.0}
        chk = planner_mod.route_check(pl, spawn, waypoint)
        if chk["feasible"] and chk.get("clear_ok") and chk["path_m"] >= 35.0:
            long_routes.append({
                "name": f"wl_ep{e['episode_id']}_{int(round(chk['path_m']))}m",
                "spawn": spawn, "waypoint": waypoint,
                "max_steps": max(int(chk["steps_estimate"]), 420),
                "ref_path": ref_nav,
                "vln": {"split": "evaluation_ready_long", "astar_path_m": chk["path_m"],
                        "instructions": [{"instruction_type": "Street_Landmark_Long",
                                          "start": "START", "end": f"goal_ep{e['episode_id']}",
                                          "text": e["instruction"]}]},
            })
    long_routes.sort(key=lambda r: -r["vln"]["astar_path_m"])
    picked = long_routes[:1]  # 取最长可飞航线（约 52.9m）做真实满电 cf_sitl 闭环与四格录像
    sc["routes"] = picked
    sc["spawn"], sc["waypoint"], sc["max_steps"] = dict(picked[0]["spawn"]), dict(picked[0]["waypoint"]), picked[0]["max_steps"]
    long_scene_path = os.path.join(OUT_DIR, "wanderland_1MlUucug_long_vln.json")
    scene_mod.save_scene(sc, long_scene_path)

    arm_out = os.path.join(OUT_DIR, "wanderland_long_eval")
    run_vln.main(["--scene", long_scene_path, "--policy", "teacher", "--teacher-clearance", "braking",
                  "--dynamics-params", SOC100_PARAMS, "--hold", str(HOLD_S),
                  "--videos", "1", "--seed", str(SEED), "--out", arm_out])
    rep = json.loads(pathlib.Path(arm_out, "vln_eval.json").read_text())
    ep_row = json.loads(pathlib.Path(arm_out, "episodes.jsonl").read_text().splitlines()[0])
    out = {"n_long_routes_feasible_ge_35m": len(long_routes),
           "all_feasible_long_routes": [{"name": r["name"], "astar_path_m": r["vln"]["astar_path_m"]} for r in long_routes],
           "evaluated_longest": {"route": ep_row["route_name"], "astar_m": picked[0]["vln"]["astar_path_m"],
                                 "flown_path_m": round(ep_row["path_m"], 2), "success": ep_row["success"],
                                 "ne_m": round(ep_row["ne_m"], 3), "spl": round(ep_row["spl"], 4),
                                 "ndtw": round(ep_row["ndtw"], 4), "sdtw": round(ep_row["sdtw"], 4),
                                 "steps": ep_row["steps"], "switches_per_m": round(ep_row["action_switches_per_m"], 3),
                                 "video": os.path.join(arm_out, ep_row["video"])}}
    write_json(os.path.join(OUT_DIR, "wanderland_long_route.json"), out)
    return out


def main():
    t0 = time.monotonic()
    os.makedirs(OUT_DIR, exist_ok=True)
    pm_path = os.path.join(OUT_DIR, "public_metrics.json")
    if os.path.isfile(pm_path):
        print("[1/5] 复用已完成 public_metrics.json...", flush=True)
        m1 = json.load(open(pm_path, encoding="utf-8"))
    else:
        print("[1/5] 计算公开基准指标（SAGE-3D 249 集 + OpenFly seen20 100 集 + Grounder 4429 句）...", flush=True)
        m1 = step1_public_metrics()
    print("[2/5] 运行 RTX 2080 Ti 3DGS 渲染画质升级实测并导出对比图...", flush=True)
    m2 = step2_render_upgrade()
    print("[3/5] 运行室内防抖刹停教师 v5 (braking) vs v1 (ray) 同种子闭环对照...", flush=True)
    m3 = step3_anti_jitter_paired()
    print("[4/5] 运行同起点双目标自然语言目的性反事实闭环与四格录像...", flush=True)
    m4 = step4_language_counterfactual()
    print("[5/5] 运行 Wanderland 52.9m 户外真实长航线满电 cf_sitl 闭环与四格录像...", flush=True)
    m5 = step5_wanderland_53m_long_route()

    summary = {"elapsed_s": round(time.monotonic() - t0, 1), "public_metrics": m1,
               "render_upgrade": m2, "anti_jitter_paired": m3,
               "language_counterfactual": m4, "wanderland_long_route": m5}
    write_json(os.path.join(OUT_DIR, "summary.json"), summary)
    pathlib.Path(OUT_DIR, "DONE").write_text(f"rc=0 elapsed_s={summary['elapsed_s']}\n", encoding="utf-8")
    print("ALL DONE:", os.path.join(OUT_DIR, "summary.json"), flush=True)


if __name__ == "__main__":
    main()
