#!/usr/bin/env python3
"""生成 3dgs-rebaseline 的 driver config.json（任务卡 S1b）。

所有超参数单源写在同目录 run.sh 顶部配置区并经环境变量传入，本文件只把它们摆成 bearing-guard-3dgs/driver.py 读的字典结构。
6 个套件（3 个路线集 × 2 个碰撞口径 collision_model ∈ {v1, v2}）：
  indoor40_v1 / indoor40_v2   K1 / W1-11 / W1-12 的室内正式 40 条路线 × 3 集（seed = 8000 + 100·路线 + k，±0.2 m，封顶 100 步）
  holdout_v1  / holdout_v2    K3 的室内留出 32 条路线 × 3 集（seed = 5000 + 100·路线 + k，±0.2 m，封顶 100 步）
  outdoor_v1  / outdoor_v2    4 个室外场景 × 6 条路线 × 5 集（seed = 7000 + 100·路线 + k，±0.5 m）
6 个臂（室内外全部同一套臂表，每个套件 6 个臂，v1 与 v2 逐集配对）：
  teacher-tol        占据 A* 教师（容差随 hold 推导 = 18.9076°，前向检查 ray；与 W1-11 teacher-tol 同参）
  teacher-swept      占据 A* 教师（容差同上，teacher_clearance = "swept"；与 W1-12 臂 A swept 同参，不带 spawn_two_layer）
  forward+route      盲前进 + oracle 路线方位护栏（45°；与 K1 / K3 同参）
  rule-occ+route     规则 v1（occ 列距离，r_safe 0.25、d_look 1.2；与 K1 / K3 同参）
  rule2-occ+route    规则 v2 选中变体 v2f（occ 列距离；与 K3 主臂同参）
  rule2-depth+route  规则 v2 对应 0.25 档变体 v2d（渲染深度列距离；与 K3 depth 臂同参）
轨迹保留：drop_trace 恒为 false（每集保留 trace.jsonl，任务卡 S1b 硬性要求）。
场景划分键预留（卡 S1a 审核 D1）：SPLIT_FROM_IDENTITY=true 时在 config 顶层写入 "split_from_identity": true，
  需等主会话把卡 S1a 与本卡合并进 main 后再生效；当前分支未含 S1a，缺省留空（不写该键，与原实验一致）。
用法：python build_config.py <套件名> <输出 config.json>
"""
import argparse
import json
import os
import sys

SUITES = ("indoor40_v1", "indoor40_v2", "holdout_v1", "holdout_v2", "outdoor_v1", "outdoor_v2")
REQUIRED = (
    "REPO", "SCENES_DIR", "CF_SITL_PROC", "HOLD", "STEP_FRAC", "STEP_PAD",
    "GUARD_DEG", "TEACHER_TOL_DEG", "YAW_STEP", "V1_R_SAFE", "V1_D_LOOK",
    "K3_SELECTED", "ROUTES_FORMAL40", "ROUTES_HOLDOUT",
    "INDOOR_SCENES", "OUTDOOR_SCENES", "ROUTES_OUTDOOR",
    "K_INDOOR40", "K_HOLDOUT", "K_OUTDOOR",
    "SEED0_INDOOR40", "SEED0_HOLDOUT", "SEED0_OUTDOOR",
    "JITTER_INDOOR", "JITTER_OUTDOOR", "LOOKAHEAD_INDOOR", "LOOKAHEAD_OUTDOOR", "STEP_CAP_INDOOR",
)


def make_arms(e, selected, collision_model):
    occ_levels = selected["levels"]
    occ_order = selected["order"]
    dep_levels = selected["depth_levels"]
    dep_order = selected["depth_order"]
    arms = [
        {"name": "teacher-tol", "base": "teacher", "ckpt": None, "guard": None},
        {"name": "teacher-swept", "base": "teacher", "ckpt": None, "guard": None, "teacher_clearance": "swept"},
        {"name": "forward+route", "base": "forward", "ckpt": None, "guard": "route"},
        {"name": "rule-occ+route", "base": "rule-occ", "ckpt": None, "guard": "route",
         "rule_params": {"r_safe": float(e["V1_R_SAFE"]), "d_look": float(e["V1_D_LOOK"])}},
        {"name": "rule2-occ+route", "base": "rule-occ", "ckpt": None, "guard": "route",
         "rule_version": "v2", "rule_params": {"levels": occ_levels, "order": occ_order}},
        {"name": "rule2-depth+route", "base": "rule-depth", "ckpt": None, "guard": "route",
         "rule_version": "v2", "rule_params": {"levels": dep_levels, "order": dep_order}},
    ]
    if collision_model == "v2":
        # B5 诊断臂（只在 v2 下跑）：规则 v2（深度）带 depth_opacity_min = 0.30（与占据门限对齐），验证 garden 虚警消除效果
        arms.append({
            "name": "rule2-depth-op030+route", "base": "rule-depth", "ckpt": None, "guard": "route",
            "rule_version": "v2", "rule_params": {"levels": dep_levels, "order": dep_order},
            "depth_opacity_min": 0.30,
        })
    return arms


def build(suite, e):
    if suite not in SUITES:
        raise ValueError(f"套件必须是 {SUITES} 之一，得到 {suite!r}")
    missing = [k for k in REQUIRED if k not in e or e[k] == ""]
    if missing:
        raise ValueError(f"缺环境变量 {missing}")
    base_cls, cm = suite.rsplit("_", 1)
    with open(e["K3_SELECTED"], encoding="utf-8") as f:
        selected = json.load(f)
    lim = int(e["ROUTES_LIMIT"]) if e.get("ROUTES_LIMIT") else None
    drop_trace = e.get("DROP_TRACE", "false").lower() == "true"
    if drop_trace:
        raise ValueError("重定基线要求保留每集 trace.jsonl：DROP_TRACE 必须为 false")

    cfg = {
        "repo": e["REPO"],
        "scenes_dir": e["SCENES_DIR"],
        "hold": float(e["HOLD"]),
        "step_frac": float(e["STEP_FRAC"]),
        "step_pad": int(e["STEP_PAD"]),
        "guard_threshold_deg": float(e["GUARD_DEG"]),
        "teacher_tol_deg": float(e["TEACHER_TOL_DEG"]),
        "drop_trace": False,
        "collision_model": cm,
        "infer_bin": "unused (no student arm)",
        "cf_sitl_proc": e["CF_SITL_PROC"],
        "suite": suite,
        "class": base_cls,
    }
    # 卡 S1a（审核 D1 / J-4）的 driver config 键 split_from_identity：
    # S1a 已合入 main，缺省为 true（按几何哈希写真实 split 与 diag-train-scene）；显式传 "none" 时不写该键。
    split_flag = e.get("SPLIT_FROM_IDENTITY", "true").strip().lower()
    if split_flag in ("", "true", "1"):
        cfg["split_from_identity"] = True
    elif split_flag in ("false", "0"):
        cfg["split_from_identity"] = False
    elif split_flag != "none":
        raise ValueError(f"SPLIT_FROM_IDENTITY 只能为 true、false 或 none，得到 {e.get('SPLIT_FROM_IDENTITY')!r}")

    if base_cls == "outdoor":
        scenes = e["OUTDOOR_SCENES"].split()
        routes = [int(x) for x in e["ROUTES_OUTDOOR"].split()][:lim]
        cfg.update({
            "scenes": scenes,
            "routes": routes,
            "k": int(e["K_OUTDOOR"]),
            "seed0": int(e["SEED0_OUTDOOR"]),
            "jitter": float(e["JITTER_OUTDOOR"]),
            "route_lookahead_m": float(e["LOOKAHEAD_OUTDOOR"]),
            "title": f"室外四场景重定基线（collision_model={cm}）",
        })
    else:
        routes_file = e["ROUTES_FORMAL40" if base_cls == "indoor40" else "ROUTES_HOLDOUT"]
        with open(routes_file, encoding="utf-8") as f:
            sel = json.load(f)
        scenes = e["INDOOR_SCENES"].split()
        k_key = "K_INDOOR40" if base_cls == "indoor40" else "K_HOLDOUT"
        s_key = "SEED0_INDOOR40" if base_cls == "indoor40" else "SEED0_HOLDOUT"
        label = "室内正式 40 条" if base_cls == "indoor40" else "室内留出 32 条"
        cfg.update({
            "scenes": scenes,
            "routes": {s: [r["route"] for r in sel["scenes"].get(s, [])][:lim] for s in scenes},
            "routes_selected": routes_file,
            "k": int(e[k_key]),
            "seed0": int(e[s_key]),
            "jitter": float(e["JITTER_INDOOR"]),
            "step_cap": int(e["STEP_CAP_INDOOR"]),
            "route_lookahead_m": float(e["LOOKAHEAD_INDOOR"]),
            "title": f"{label}重定基线（collision_model={cm}）",
        })

    arms = make_arms(e, selected, cm)
    if e.get("ONLY_ARMS"):
        want = [x.strip() for x in e["ONLY_ARMS"].split(",") if x.strip()]
        bad = [x for x in want if x not in {a["name"] for a in arms}]
        if bad or not want:
            raise ValueError(f"ONLY_ARMS 含未知臂 {bad or want!r}")
        arms = [a for a in arms if a["name"] in want]
        cfg["only"] = [a["name"] for a in arms]
    cfg["arms"] = arms
    cfg["rule_params"] = {"yaw_step_deg": float(e["YAW_STEP"])}
    cfg["selected"] = {"variant": selected["variant"], "depth_variant": selected["depth_variant"]}
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("suite", choices=SUITES)
    ap.add_argument("out")
    args = ap.parse_args(argv)
    cfg = build(args.suite, dict(os.environ))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
