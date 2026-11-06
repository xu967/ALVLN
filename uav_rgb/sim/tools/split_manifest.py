#!/usr/bin/env python3
"""几何组 ID 与冻结 split 清单（PARALLEL_TRAINING_PLAN §19.1）。

几何组 = 场景里障碍的 shape/尺寸/pose 规范化后取 sha256。scene_id / seed /
文件名 / 注释这些不改变物理布局的字段一律排除，所以「同几何、不同 seed」必然
落进同一组，也必然落进同一 split。

split = sha256(SPLIT_SALT + gid) 前 8 位 % 100：<80 train，<90 val，其余 test。
同一 gid 永远得到同一 split，与 seed 范围、生成顺序、已采多少数据都无关。

改 scene_gen 的几何语义（柱尺寸、车道间距、采样模式）会让所有 gid 变，
必须同步 bump SPLIT_SALT 并开新数据版本；已有数据不原地重标。
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scene_gen as sg  # noqa: E402

SPLITS = ("train", "val", "test")
# 数据里可审计的划分策略。scene_id = 旧版只按场景隔离；其余见 split_for_scene。
SPLIT_POLICIES = ("scene_id", "geometry_gid", "campus_map_v1", "campus_assets_map_v1")
# 已冻结、但重打包仍要能按记录复算的校园版本；新增版本时只加当前版本，不删旧值。
FROZEN_CAMPUS_VERSIONS = ("campus.v1",)
# asset_campus.split_key 写死的 salt（split_key 内部不接受参数）。
ASSET_SPLIT_SALT = "asset-campus-split-v1"
# v3：T1 把车道缝从 0.902 m 放宽到 1.2 m（LANE_SEP 1.302→1.6），几何语义变了，
#     必须换 salt；旧 v2 数据不能与 v3 混用，见 docs/experiments/2026-10-30-t1-lane-geometry.md。
# v2：v1 把 -0.0 格式化成 "-0.000"，同几何不同符号会得到不同 gid（seed 337/3651）。
# 单一来源：SPLIT_SALT 与 scene.json 的 geometry_version 必须是同一个字符串。
SPLIT_SALT = sg.GEOMETRY_VERSION
# 采集默认步数（collect.py / collect_parallel.py 共用，改一处即可）：
# v3 几何 + A* 每步约 0.27 m，T2 实测 29–38 步过线（中位 30），40 步留余量；
# 80 步会让约一半帧在柱区外原地转向。
DEFAULT_STEPS = 40
LAYOUT_PATTERN = "lane3_serpentine"   # sample_pillars 的布局模式；改模式要同步改这里
GEOMETRY_PRECISION = 3                # scene_gen 的 x/y 输出精度（round 到 3 位）
SCHEMA = "navlab.split_manifest.v1"
GEN_PATH = os.path.join(sg.SIM_DIR, "tools", "scene_gen.py")


def _fmt(v):
    # +0.0 把 -0.0 归一成 0.0：否则 "%.3f" % -0.0 是 "-0.000"，同几何会算出不同 gid。
    return ("%." + str(GEOMETRY_PRECISION) + "f") % (v + 0.0)


def canonical_geometry(pillars):
    """按坐标排序、只留 x/y/z/尺寸；name 与输入顺序都不参与。"""
    boxes = []
    for p in pillars:
        sx, sy, sz = p["size"]
        boxes.append([_fmt(p["x"]), _fmt(p["y"]), _fmt(p["z"]),
                      _fmt(sx), _fmt(sy), _fmt(sz)])
    boxes.sort()
    return boxes


def geometry_gid(pillars):
    payload = json.dumps(canonical_geometry(pillars), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def family_id(template_sha):
    """场景族：模板 + 布局常量决定，与 seed 无关。

    当前生成器只有一族，所以 split 之间做不到族留出；报告里必须写明
    family_holdout=false —— 几何不重复不等于泛化。
    """
    sig = json.dumps({
        "pattern": LAYOUT_PATTERN,
        "template_sha256": template_sha,
        "lane_sep": sg.LANE_SEP,
        "pillar_h": sg.PILLAR_HEIGHT,
        "corridor_y": list(sg.Y_RANGE),
        "x_range": list(sg.PILLAR_X_RANGE),
        "y_range": list(sg.PILLAR_Y_RANGE),
        "count_range": list(sg.PILLAR_COUNT_RANGE),
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]


def split_for_gid(gid, salt=None):
    h = hashlib.sha256(((salt or SPLIT_SALT) + ":" + gid).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 80:
        return "train"
    return "val" if bucket < 90 else "test"


def is_diagnostic_scheme(name):
    """诊断性划分（map-hash:* / diag*）默认拒绝：它们无视地图/拓扑族归属。"""
    return bool(name) and (name.startswith("map-hash:") or name.startswith("diag"))


def split_salt_for(meta):
    """该 manifest 实际使用的 salt；按记录的 split_salt/版本分派，不回落当前值。

    历史约定 SPLIT_SALT == GEOMETRY_VERSION，所以 v3/v4 等冻结柱阵数据能用
    geometry_version 原样复算。campus 按固定地图族归属，没有 gid salt。
    """
    if meta.get("split_salt"):
        return meta["split_salt"]
    policy = meta.get("split_policy")
    if policy == "campus_assets_map_v1":
        return ASSET_SPLIT_SALT
    if policy == "campus_map_v1":
        return None
    gver = meta.get("geometry_version")
    if not gver:
        return SPLIT_SALT
    # forest_e1 同旧柱阵约定「版本串 = salt」（scene_forest.GEOMETRY_VERSION_FOREST）；
    # scene.json 显式带 split_salt，episode manifest 不带，靠这里复算。
    if gver.startswith("uavrgb.geom.") or gver.startswith("uavrgb.forest_e1."):
        return gver
    raise ValueError("无法复算 split_salt：未知 geometry_version %r" % (gver,))


PROVENANCE_KEYS = ("appearance_version", "map_id", "family", "split_policy",
                   "world_sha256", "template_sha256", "resource_sha256",
                   "camera_profile", "dynamics_profile", "success_criterion", "goal_radius")


def split_for_scene(meta):
    """按 manifest 记录的策略/版本复算 split；冻结数据不回落当前 salt。"""
    import campus
    import asset_campus
    policy = meta.get("split_policy")
    gver = meta.get("geometry_version") or ""
    map_id = meta.get("map_id") or ""
    # 身份自洽：map_id 声称资产校园时，geometry_version 必须同名。
    if map_id.startswith(asset_campus.VERSION + "/") and gver != asset_campus.VERSION:
        raise ValueError("asset campus map_id with mismatched geometry_version")
    if policy == "campus_assets_map_v1" or gver.startswith("campus.assets."):
        if gver != asset_campus.VERSION:
            raise ValueError("asset campus map_id with mismatched geometry_version")
        expected_key = asset_campus.map_key(meta["seed"])
        if map_id != expected_key:
            raise ValueError("asset campus map_id 与 seed 不符")
        return asset_campus.split_key(expected_key)
    if policy == "campus_map_v1" or gver.startswith("campus."):
        if gver not in (campus.VERSION,) + FROZEN_CAMPUS_VERSIONS:
            raise ValueError("未知 campus geometry_version %r" % (gver,))
        prefix = gver + "/"
        if not map_id.startswith(prefix):
            raise ValueError("campus map_id 与 geometry_version 不符")
        family = map_id[len(prefix):]
        if family not in campus.FAMILIES:
            raise ValueError("未知 campus 地图族 %r" % (family,))
        if meta.get("seed") is not None:
            want = campus.FAMILIES[(int(meta["seed"]) // 8) % len(campus.FAMILIES)]
            if family != want:
                raise ValueError("campus map_id 与 seed 不符")
        # 三族→split 本身是冻结合同；campus.split_map 只认当前版本前缀，借用它的表。
        return campus.split_map("%s/%s" % (campus.VERSION, family))
    gid = meta.get("geometry_gid") or geometry_gid(meta["pillars"])
    return split_for_gid(gid, split_salt_for(meta))


def split_for_seed(seed, task="base", density=None):
    if task == "campus":
        import campus
        return campus.split(seed)
    if task == "campus_assets":
        import asset_campus
        return asset_campus.split_key(asset_campus.map_key(seed))
    if task == "forest_e1":
        import scene_forest   # 几何随 density 变，salt 是森林族自己的版本串
        trunks = scene_forest.sample_trunks(seed, density)[0]
        return split_for_gid(geometry_gid(trunks), scene_forest.GEOMETRY_VERSION_FOREST)
    if task not in ("base", "l1"):
        raise ValueError(f"unknown task {task!r}")
    return split_for_gid(geometry_gid(sg.sample_pillars(seed)[0]))


def _file_sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_manifest(seeds, template_path=None):
    tsha = _file_sha(template_path or sg.DEFAULT_TEMPLATE)
    fam = family_id(tsha)
    groups = {}
    for seed in seeds:
        pillars, _, _ = sg.sample_pillars(seed)
        gid = geometry_gid(pillars)
        g = groups.get(gid)
        if g is None:
            g = groups[gid] = {"gid": gid, "family": fam,
                               "pillar_count": len(pillars), "seeds": []}
        g["seeds"].append(seed)
    for g in groups.values():
        g["seeds"].sort()
        g["split"] = split_for_gid(g["gid"])
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": "sim/tools/scene_gen.py",
        "generator_sha256": _file_sha(GEN_PATH),
        "template_sha256": tsha,
        "layout_pattern": LAYOUT_PATTERN,
        "split_salt": SPLIT_SALT,
        "geometry_precision": GEOMETRY_PRECISION,
        "policy": "sha256(salt:gid) %% 100 -> <80 train / <90 val / else test",
        "groups": sorted(groups.values(), key=lambda g: (g["split"], g["gid"])),
    }


def summarize(man):
    per = {s: {"groups": 0, "seeds": 0, "pillars": []} for s in SPLITS}
    fams = set()
    for g in man["groups"]:
        d = per[g["split"]]
        d["groups"] += 1
        d["seeds"] += len(g["seeds"])
        d["pillars"].append(g["pillar_count"])
        fams.add(g["family"])
    out = {"seeds": sum(d["seeds"] for d in per.values()),
           "groups": len(man["groups"]),
           "families": len(fams),
           "per_split": {}}
    for s in SPLITS:
        p = per[s]["pillars"]
        out["per_split"][s] = {
            "groups": per[s]["groups"],
            "seeds": per[s]["seeds"],
            "pillars_min": min(p) if p else 0,
            "pillars_max": max(p) if p else 0,
        }
    out["duplicate_seeds"] = out["seeds"] - out["groups"]
    # 只有一族时任何 split 都是同模板分布内，族留出无从谈起。
    out["family_holdout"] = len(fams) > 1
    return out


def group_id_for_task(seed, task, density=None):
    """按任务返回几何组身份；base/l1 用精确几何，校园用 map 身份，forest_e1 用该 density 下的树干几何。

    与 split_for_seed 同一套任务表：未知任务 fail-closed。旧版对任何非校园任务都回落到柱阵几何，
    forest_e1 的集会被按同 seed 的柱阵 gid 查重——查的是另一片几何，森林集跨 split 查不出来。
    """
    if task == "campus_assets":
        import asset_campus
        return asset_campus.map_key(seed)
    if task == "campus":
        import campus
        return campus.map_id(seed)
    if task == "forest_e1":
        import scene_forest
        if density is None:
            raise ValueError("forest_e1 的几何随 density 变，必须给 density")
        return geometry_gid(scene_forest.sample_trunks(seed, density)[0])
    if task not in ("base", "l1"):
        raise ValueError(f"unknown task {task!r}")
    return geometry_gid(sg.sample_pillars(seed)[0])


def cross_split_groups(pairs, task="base", density=None):
    """pairs = [(seed, split)]；返回跨 split 的几何组 {gid: [splits]}。"""
    seen = {}
    for seed, split in pairs:
        seen.setdefault(group_id_for_task(seed, task, density), set()).add(split)
    return {gid: sorted(s) for gid, s in seen.items() if len(s) > 1}


def pairs_from_collection(path):
    """读 collection_manifest.json 的 splits 字段。"""
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    return [(seed, split) for split, seeds in man["splits"].items() for seed in seeds]


def task_density_from_collection(path):
    """collection_manifest.json 里 collect_parallel 记下的 (task, density)；旧清单没有 task 时按 base。"""
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    return man.get("task") or "base", man.get("density")


def parse_seeds(spec):
    a, _, b = spec.partition(":")
    lo = int(a)
    return list(range(lo, int(b) if b else lo + 1))


def main(argv=None):
    p = argparse.ArgumentParser(description="几何组 ID 与冻结 split 清单")
    p.add_argument("--seeds", default="", help="闭开区间 a:b，例如 1000:2000")
    p.add_argument("--base-seed", type=int, default=None, help="配合 --count 使用")
    p.add_argument("--count", type=int, default=None, help="配合 --base-seed 使用")
    p.add_argument("--out", default="", help="写出 split_manifest.json")
    p.add_argument("--verify", default="", help="检查 collection_manifest.json 是否跨 split 重复")
    p.add_argument("--verify-task", choices=("base", "l1", "campus", "campus_assets", "forest_e1"),
                   default=None, help="--verify 使用的几何组身份函数；缺省取清单记录的 task（旧清单按 base）")
    p.add_argument("--verify-density", choices=("sparse", "medium", "dense"), default=None,
                   help="forest_e1 的密度档；缺省取清单记录的 density")
    args = p.parse_args(argv)

    if args.verify:
        pairs = pairs_from_collection(args.verify)
        rec_task, rec_density = task_density_from_collection(args.verify)
        task = args.verify_task or rec_task
        density = args.verify_density or rec_density
        if task != rec_task:
            print("警告: --verify-task %s 与清单记录的 task %s 不同" % (task, rec_task))
        bad = cross_split_groups(pairs, task, density)
        print("seed 数 %d（task=%s%s），跨 split 的几何组 %d"
              % (len(pairs), task, f" density={density}" if density else "", len(bad)))
        for gid, sp in bad.items():
            seeds = [s for s, sp2 in pairs if sp2 in sp and
                     group_id_for_task(s, task, density) == gid]
            print("  %s  splits=%s  seeds=%s" % (gid, ",".join(sp), seeds))
        return 1 if bad else 0

    if args.seeds:
        seeds = parse_seeds(args.seeds)
    elif args.base_seed is not None and args.count:
        seeds = list(range(args.base_seed, args.base_seed + args.count))
    else:
        p.error("需要 --seeds 或 --base-seed + --count（或用 --verify）")

    man = build_manifest(seeds)
    stats = summarize(man)
    print(json.dumps(stats, ensure_ascii=False, indent=1))
    if not stats["family_holdout"]:
        print("警告: 当前只有 %d 个场景族，split 之间不存在族留出；"
              "几何不重复不等于泛化。" % stats["families"], file=sys.stderr)
    if args.out:
        tmp = args.out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(man, f, ensure_ascii=False, indent=1)
            f.write("\n")
        os.replace(tmp, args.out)
        print("已写 %s（%d 组）" % (args.out, len(man["groups"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
