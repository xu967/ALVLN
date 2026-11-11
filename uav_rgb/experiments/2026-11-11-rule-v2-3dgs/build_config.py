#!/usr/bin/env python3
"""写 driver 的 config.json（run.sh 配置区是参数的唯一来源，值经环境变量传进来；这里只负责把它们摆成 driver 读的结构）。

四个类别（每类一个目录，driver 对每个场景起一个进程，跑完该类的全部臂与路线）：
  dev       室内开发路线（K1 的 20 条 + 新选的），1 集，seed 9000 + 100·路线；臂 = v1（K1 冻结参数）+ 变体表里的全部 v2 变体
  holdout   室内 held-out，3 集，seed 5000 + 100·路线 + k；臂 = forward+route、rule-occ+route（v1）、rule2-occ+route（主臂）、rule2-depth+route
  formal40  室内正式 40 条（K1 的同一批），3 集，seed 8000 + 100·路线 + k；臂 = rule-occ+route（v1 重跑）、rule2-occ+route、rule2-depth+route
  outdoor   室外 4 场景 × 6 路线，5 集，seed 7000 + 100·路线 + k；臂 = rule2-occ+route、rule2-depth+route（v1 取 K1 的结果按键配对）
formal 类别读 selected.json（select_variant.py 写）：主臂用选中变体，depth 臂用 DEPTH_MAP 映射后的变体。
臂的顺序即优先级（driver 按这个顺序跑）。v1 臂不带 rule_version 键，与 K1 的 rule-occ+route 臂的配置逐项相同。
用法：python build_config.py <类别> <输出 config.json> [--selected <selected.json>] [--only <臂名>[,<臂名>]]    （环境变量见 run.sh 的 export 行）
--only 只给 F0 用：正式 40 条先只跑 v1 臂（rule-occ+route）与 K1 对拍，对拍过了再换完整三臂版（driver 按 done.json 跳过已评完的 v1 臂）。
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import variants as vr  # noqa: E402

CLASSES = ("dev", "holdout", "formal40", "outdoor")
FWD = {"base": "forward", "ckpt": None}
REQUIRED = ("REPO", "SCENES_DIR", "HOLD", "STEP_FRAC", "STEP_PAD", "GUARD_DEG", "DROP_TRACE", "CF_SITL_PROC", "YAW_STEP", "V1_R_SAFE", "V1_D_LOOK")


def v1_arm(e, name="rule-occ+route"):
    return {"name": name, "base": "rule-occ", "ckpt": None, "guard": "route",
            "rule_params": {"r_safe": float(e["V1_R_SAFE"]), "d_look": float(e["V1_D_LOOK"])}}


def v2_arm(name, base, variant):
    return {"name": name, "base": base, "ckpt": None, "guard": "route", "rule_version": "v2", "rule_params": vr.rule_params_of(variant)}


def arms_for(cls, e, selected):
    if cls == "dev":
        out = [v1_arm(e, "v1")]
        out += [v2_arm(v["name"], "rule-occ", v) for v in vr.parse_variants(e["VARIANTS"])]
        return out
    variants = vr.parse_variants(e["VARIANTS"])
    main = vr.by_name(variants, selected["variant"])
    depth = vr.by_name(variants, selected["depth_variant"])
    main_arm, depth_arm = v2_arm("rule2-occ+route", "rule-occ", main), v2_arm("rule2-depth+route", "rule-depth", depth)
    if cls == "holdout":
        return [{"name": "forward+route", **FWD, "guard": "route"}, v1_arm(e), main_arm, depth_arm]
    if cls == "formal40":
        return [v1_arm(e), main_arm, depth_arm]
    return [main_arm, depth_arm]


def build(cls, e, selected=None, only=None):
    if cls not in CLASSES:
        raise ValueError(f"类别只能是 {CLASSES}，得到 {cls!r}")
    missing = [k for k in REQUIRED if k not in e or e[k] == ""]
    if missing:
        raise ValueError(f"缺环境变量 {missing}")
    if cls != "dev" and not selected:
        raise ValueError(f"{cls} 需要 selected.json（先跑完 dev 与 select_variant.py）")
    lim = int(e["ROUTES_LIMIT"]) if e.get("ROUTES_LIMIT") else None
    cfg = {"repo": e["REPO"], "scenes_dir": e["SCENES_DIR"], "hold": float(e["HOLD"]), "step_frac": float(e["STEP_FRAC"]),
           "step_pad": int(e["STEP_PAD"]), "guard_threshold_deg": float(e["GUARD_DEG"]), "drop_trace": e["DROP_TRACE"] == "true",
           "infer_bin": "unused (no student arm)", "cf_sitl_proc": e["CF_SITL_PROC"], "class": cls}
    if cls == "outdoor":
        scenes = e["OUTDOOR_SCENES"].split()
        cfg.update({"scenes": scenes, "routes": [int(x) for x in e["ROUTES_OUTDOOR"].split()][:lim], "k": int(e["K_OUTDOOR"]),
                    "seed0": int(e["SEED0_OUTDOOR"]), "jitter": float(e["JITTER_OUTDOOR"]), "route_lookahead_m": float(e["LOOKAHEAD_OUTDOOR"]),
                    "title": "室外（3DGS 四场景）规则 v2"})
    else:
        routes_file = e[{"dev": "ROUTES_DEV", "holdout": "ROUTES_HOLDOUT", "formal40": "ROUTES_FORMAL40"}[cls]]
        with open(routes_file, encoding="utf-8") as f:
            sel = json.load(f)
        scenes = e["INDOOR_SCENES"].split()
        k = {"dev": "K_DEV", "holdout": "K_HOLDOUT", "formal40": "K_FORMAL40"}[cls]
        seed0 = {"dev": "SEED0_DEV", "holdout": "SEED0_HOLDOUT", "formal40": "SEED0_FORMAL40"}[cls]
        cfg.update({"scenes": scenes, "routes": {s: [r["route"] for r in sel["scenes"].get(s, [])][:lim] for s in scenes},
                    "routes_selected": routes_file, "k": int(e[k]), "seed0": int(e[seed0]), "jitter": float(e["JITTER_INDOOR"]),
                    "step_cap": int(e["STEP_CAP_INDOOR"]), "route_lookahead_m": float(e["LOOKAHEAD_INDOOR"]),
                    "title": {"dev": "室内开发路线（变体选择）", "holdout": "室内 held-out（主比较）", "formal40": "室内正式 40 条"}[cls]})
    cfg["arms"] = arms_for(cls, e, selected)
    if only is not None:            # F0：先只跑 v1 臂做对拍；顺序仍按臂表，其余键与完整版逐项相同（只多一个 only 键作记录）
        bad = [n for n in only if n not in {a["name"] for a in cfg["arms"]}]
        if bad or not only:
            raise ValueError(f"--only 的臂名 {bad or only!r} 不在 {cls} 的臂表 {[a['name'] for a in cfg['arms']]} 里")
        cfg["arms"] = [a for a in cfg["arms"] if a["name"] in only]
        cfg["only"] = [a["name"] for a in cfg["arms"]]
    cfg["rule_params"] = {"yaw_step_deg": float(e["YAW_STEP"])}
    cfg["frozen"] = {"yaw_step_deg": float(e["YAW_STEP"]), "v1_r_safe": float(e["V1_R_SAFE"]), "v1_d_look": float(e["V1_D_LOOK"])}
    if selected:
        cfg["selected"] = {k: selected[k] for k in ("variant", "depth_variant") if k in selected}
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cls", choices=CLASSES)
    p.add_argument("out")
    p.add_argument("--selected")
    p.add_argument("--only", help="逗号分隔的臂名：只留这些臂（F0 先只跑 v1 臂做对拍）")
    a = p.parse_args(argv)
    selected = None
    if a.selected:
        with open(a.selected, encoding="utf-8") as f:
            selected = json.load(f)
    cfg = build(a.cls, dict(os.environ), selected, only=a.only.split(",") if a.only else None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
