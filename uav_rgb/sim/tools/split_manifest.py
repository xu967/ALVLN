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
# v2：v1 把 -0.0 格式化成 "-0.000"，同几何不同符号会得到不同 gid（seed 337/3651）。
SPLIT_SALT = "uavrgb.geom.v2"
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


def split_for_gid(gid):
    h = hashlib.sha256((SPLIT_SALT + ":" + gid).encode("utf-8")).hexdigest()
    bucket = int(h[:8], 16) % 100
    if bucket < 80:
        return "train"
    return "val" if bucket < 90 else "test"


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


def cross_split_groups(pairs):
    """pairs = [(seed, split)]；返回跨 split 的几何组 {gid: [splits]}。"""
    seen = {}
    for seed, split in pairs:
        pillars, _, _ = sg.sample_pillars(seed)
        seen.setdefault(geometry_gid(pillars), set()).add(split)
    return {gid: sorted(s) for gid, s in seen.items() if len(s) > 1}


def pairs_from_collection(path):
    """读 collection_manifest.json 的 splits 字段。"""
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    return [(seed, split) for split, seeds in man["splits"].items() for seed in seeds]


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
    args = p.parse_args(argv)

    if args.verify:
        pairs = pairs_from_collection(args.verify)
        bad = cross_split_groups(pairs)
        print("seed 数 %d，跨 split 的几何组 %d" % (len(pairs), len(bad)))
        for gid, sp in bad.items():
            seeds = [s for s, sp2 in pairs if sp2 in sp and
                     geometry_gid(sg.sample_pillars(s)[0]) == gid]
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
