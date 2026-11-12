#!/usr/bin/env python3
"""挑第二部分闭环用的 20 个 val 分区几何（任务卡 E2E-R1 第二部分）。离线、只读，写一份带出处的 JSON，清单再抄进 run_closed_loop.sh 的配置区。

为什么按几何组 ID（gid）排除，而不是只按 seed 区间：base 任务 seed 0–49999 一共只有 804 个不同几何（train 662 / val 71 / test 71，2026-11-12 实测），
71 个 val 几何的首次出现 seed 全部不超过 2632，所以「一段没被别的实验用过的 seed 区间」不能保证几何没被用过：高位 seed 只是在重复低位 seed 的几何。
这里两层都做：① 候选 seed 取 [SEED_START, SEED_MAX)，并核对这段 seed 没有出现在任何已有清单或评测里；② 排除的单位是 gid，排除集见下。

排除集（任一来源用过的几何都不用，gid 按当前几何版本 v5 与该来源的任务复算）：
  1. 训练数据集的全部 300 集（$D/tmp/agent6-distill/trunc/col/{train,val,test}/<seed>）。校准 δ 与选 ckpt 都用过它的 val 分区，所以必须是这 14 个 val 几何之外的几何；
  2. $D 下（深度 <= MAXDEPTH）所有 task=base 或没记任务、几何版本是 v5 或没记、分区方案没写旧盐的 collection_manifest.json 里出现过的 seed；
  3. $D 下所有 task=base、geometry_version=v5 的 eval.json 里评过的 geometry_gid；
  4. tools/eval_seeds.py 的全部清单（它们是 test 分区，排除只是显式留痕）。
不排除的：旧几何版本（v1/v3，盐不同，同一个 seed 是另一个几何）、l1 / forest / campus / course 任务的清单（几何空间不同）。第一版把所有清单一律按 base-v5 复算 gid，
结果是 71 个 val 几何全被排除（旧版本 600 集的清单每个覆盖 41 个），那是把「同 seed 不同几何」当成了「同几何」，不是真实的重叠，所以改成上面的按版本与任务过滤。
选法：seed 升序，分区 = val、gid 不在排除集、gid 之间两两不同，取前 COUNT 个。两个来源是动态的：别的卡随后新采或新评的几何不在排除集里，
JSON 里记了扫描时刻与各来源的文件数，报告里要写明。

用法：
  select_val_geoms.py --out <json> [--data-root <D>]     扫描、挑选、写 JSON（带出处）并打印摘要；挑完用 scene_gen 真生成一遍核对 gid 与分区
  select_val_geoms.py --check <json>                      现算一遍与 JSON 比对（排除集会随别的卡新采新评而变大，清单不一致不代表冻结的清单有错）
  select_val_geoms.py --frozen-check <seed 列表,逗号分隔>   不扫描：只核对冻结清单本身（分区都是 val、gid 两两不同、与 tools/eval_seeds.py 全部清单不相交），入口脚本起跑时用
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
sys.path.insert(0, os.path.join(REPO, "tools"))
import split_manifest as sm  # noqa: E402

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
SEED_START = 30000      # 候选 seed 的起点：低位 seed 是别的实验的惯用区（评测 0–2600、采集 3000–4200），30000 起没有任何已有清单用过（脚本核对）
SEED_MAX = 50000        # 扫描上限（不含）：本脚本核对过的 seed 空间是 0–49999；上限处凑不够 COUNT 就报错退出
COUNT = 20              # 任务卡：20 个 val 几何
TASK = "base"           # 任务卡：--task base
SPLIT = "val"           # 任务卡：--split val
MAXDEPTH = 6            # 在数据根下找 collection_manifest.json / eval.json 的最大深度
DATASET_COL = "tmp/agent6-distill/trunc/col"   # 相对数据根：训练数据集 300 集的目录（train/val/test 子目录下以 seed 命名）
# ──────────────────────────────────────────────────────────────────────────


def find_files(root, name, maxdepth):
    """root 下深度 <= maxdepth 的所有名为 name 的文件；跳过冻结代码副本 code/、演练目录 rehearsal/、各集的 episodes/ 子树。"""
    out = []
    base_depth = root.rstrip("/").count("/")
    for dp, dns, fns in os.walk(root):
        depth = dp.count("/") - base_depth
        dns[:] = [d for d in dns if d not in ("code", "rehearsal", "episodes", ".git", "__pycache__")] if depth < maxdepth else []
        if name in fns:
            out.append(os.path.join(dp, name))
    return sorted(out)


OLD_SALTS = ("geom.v1", "geom.v2", "geom.v3", "geom.v4")   # 分区方案文字里出现这些 = 旧几何版本，同 seed 是另一个几何


def manifest_counts(man):
    """该清单的 seed 能否按「base 任务、当前几何版本」复算出它真实用过的几何：是 → True。"""
    if man.get("task") not in (None, "base"):
        return False
    if man.get("geometry_version") not in (None, sm.SPLIT_SALT):
        return False
    scheme = str(man.get("split_scheme") or man.get("split_policy") or "")
    return not any(x in scheme for x in OLD_SALTS) and "scene_id" not in scheme


def manifest_seeds(path):
    """collection_manifest.json 里的全部 seed（splits 下各列表）与是否计入排除集；读不了的返回 None。"""
    try:
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
        return sorted({int(s) for seeds in man["splits"].values() for s in seeds}), manifest_counts(man)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def eval_seeds_and_gids(path):
    """eval.json 里评过的 seed、geometry_gid（只在 task=base 且 geometry_version=当前版本时取，否则 gid 空）与 split；读不了的返回 None。"""
    try:
        with open(path, encoding="utf-8") as f:
            e = json.load(f)
        pe = e.get("per_episode") or []
        seeds = {int(x["seed"]) for x in pe if "seed" in x} | {int(s) for s in (e.get("seeds") or []) if isinstance(s, int)}
        counts = e.get("task") == "base" and e.get("geometry_version") == sm.SPLIT_SALT
        gids = {x["geometry_gid"] for x in pe if x.get("geometry_gid")} if counts else set()
        return sorted(seeds), sorted(gids), e.get("split")
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def frozen_eval_seed_lists():
    import eval_seeds as es
    out = {}
    for name in dir(es):
        v = getattr(es, name)
        if name.isupper() and isinstance(v, tuple) and v and all(isinstance(s, int) for s in v):
            out[name] = list(v)
    return out


def gid_of(seed):
    return sm.group_id_for_task(seed, TASK, None)


def build_exclusions(data_root):
    """返回 (排除 gid 集, 排除 seed 集, 出处 dict)。"""
    gids, seeds, src = set(), set(), {}
    col = os.path.join(data_root, DATASET_COL)
    ds = sorted(int(os.path.basename(p)) for sp in ("train", "val", "test") for p in glob.glob(os.path.join(col, sp, "*")) if os.path.basename(p).isdigit())
    ds_gids = {gid_of(s) for s in ds}
    gids |= ds_gids
    seeds |= set(ds)
    src["dataset"] = {"dir": col, "episodes": len(ds), "distinct_gids": len(ds_gids), "gids": sorted(ds_gids),
                      "val_gids": sorted(g for g in ds_gids if sm.split_for_gid(g) == "val")}
    mf, n_ok, n_used, mseeds, all_mseeds = find_files(data_root, "collection_manifest.json", MAXDEPTH), 0, 0, set(), set()
    for p in mf:
        got = manifest_seeds(p)
        if got is not None:
            n_ok += 1
            all_mseeds |= set(got[0])
            if got[1]:
                n_used += 1
                mseeds |= set(got[0])
    mg = {gid_of(s) for s in mseeds}
    gids |= mg
    seeds |= all_mseeds       # seed 区间核对用全部清单，不分版本
    src["collection_manifests"] = {"files": len(mf), "readable": n_ok, "counted_as_base_v5": n_used, "seeds_counted": len(mseeds), "distinct_gids": len(mg)}
    ev, n_ok, n_used, eseeds, egids, splits = find_files(data_root, "eval.json", MAXDEPTH), 0, 0, set(), set(), {}
    for p in ev:
        got = eval_seeds_and_gids(p)
        if got is not None:
            n_ok += 1
            eseeds |= set(got[0])
            if got[1]:
                n_used += 1
                egids |= set(got[1])
                splits[got[2]] = splits.get(got[2], 0) + 1
    gids |= egids
    seeds |= eseeds
    src["eval_json"] = {"files": len(ev), "readable": n_ok, "counted_as_base_v5": n_used, "distinct_gids": len(egids), "by_split": {str(k): v for k, v in splits.items()}}
    fz = frozen_eval_seed_lists()
    fseeds = {s for v in fz.values() for s in v}
    fg = {gid_of(s) for s in fseeds}
    gids |= fg
    seeds |= fseeds
    src["tools_eval_seeds"] = {"lists": sorted(fz), "seeds": len(fseeds), "distinct_gids": len(fg)}
    return gids, seeds, src


def verify_with_scene_gen(seeds, gids):
    """对每个 seed 真跑 scene_gen.py（评测入口生成世界走的同一条命令，--vehicle e1_mini --task base --appearance v1），
    从生成出的 meta 里重算 gid 与分区，和清单里的逐个核对。返回 [{seed, gid, gid_scene_gen, split_scene_gen, ok}]。"""
    out = []
    with tempfile.TemporaryDirectory(prefix="geomchk-", dir=os.environ.get("TMPDIR")) as td:
        for seed, gid in zip(seeds, gids):
            world = os.path.join(td, f"scene_{seed}.sdf")
            r = subprocess.run([sys.executable, os.path.join(REPO, "sim", "tools", "scene_gen.py"), "--seed", str(seed), "--scene-id", str(seed),
                                "--out", world, "--task", TASK, "--appearance", "v1", "--vehicle", "e1_mini"], capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                out.append({"seed": seed, "gid": gid, "gid_scene_gen": None, "split_scene_gen": None, "ok": False, "error": r.stderr.strip()[-200:]})
                continue
            with open(os.path.join(td, f"scene_{seed}.json"), encoding="utf-8") as f:
                meta = json.load(f)
            g, sp = sm.geometry_gid(meta["pillars"]), sm.split_for_scene(meta)
            out.append({"seed": seed, "gid": gid, "gid_scene_gen": g, "split_scene_gen": sp, "ok": g == gid and sp == SPLIT})
    return out


def frozen_check(seeds):
    """冻结清单自身的核对（不依赖别的卡此刻的数据）。返回 (是否全过, 记录 dict)。"""
    gids = [gid_of(x) for x in seeds]
    bad_split = [x for x in seeds if sm.split_for_seed(x, TASK, None) != SPLIT]
    dup = sorted({g for g in gids if gids.count(g) > 1})
    lists = frozen_eval_seed_lists()
    in_lists = {name: sorted(set(v) & set(seeds)) for name, v in lists.items() if set(v) & set(seeds)}
    gid_in_lists = {name: sorted({gid_of(y) for y in v} & set(gids)) for name, v in lists.items() if {gid_of(y) for y in v} & set(gids)}
    rec = {"n": len(seeds), "distinct_gids": len(set(gids)), "not_val": bad_split, "duplicate_gids": dup, "seeds_in_eval_seed_lists": in_lists,
           "gids_in_eval_seed_lists": gid_in_lists, "sha256": hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()}
    return not (bad_split or dup or in_lists or gid_in_lists or len(seeds) != len(set(seeds))), rec


def pick(exclude_gids, seed_start=SEED_START, seed_max=SEED_MAX, count=COUNT, split=SPLIT):
    """seed 升序，分区 = split、gid 不在 exclude_gids、gid 两两不同，取前 count 个。返回 (seeds, gids)；凑不够抛 SystemExit。"""
    seeds, gids, seen = [], [], set(exclude_gids)
    for seed in range(seed_start, seed_max):
        if sm.split_for_seed(seed, TASK, None) != split:
            continue
        g = gid_of(seed)
        if g in seen:
            continue
        seen.add(g)
        seeds.append(seed)
        gids.append(g)
        if len(seeds) >= count:
            return seeds, gids
    raise SystemExit(f"[{seed_start}, {seed_max}) 里只凑到 {len(seeds)}/{count} 个不在排除集的 {split} 几何")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="/root/rivermind-data/uav_rgb")
    ap.add_argument("--out")
    ap.add_argument("--check", help="读已写出的 JSON，现算一遍清单并与之比对，同时再跑一遍 scene_gen 核对")
    ap.add_argument("--no-verify", action="store_true", help="跳过 scene_gen 核对（默认要核对）")
    ap.add_argument("--frozen-check", metavar="SEEDS", help="只核对这份冻结清单（逗号分隔的 seed）本身，不扫描")
    args = ap.parse_args(argv)
    if args.frozen_check:
        ok, rec = frozen_check([int(x) for x in args.frozen_check.split(",") if x.strip()])
        print(json.dumps({"ok": ok, **rec}, ensure_ascii=False))
        return 0 if ok else 1
    t0 = time.time()
    gids_ex, seeds_ex, src = build_exclusions(args.data_root)
    used_in_range = sorted(s for s in seeds_ex if SEED_START <= s < SEED_MAX)
    seeds, gids = pick(gids_ex)
    val_all = {gid_of(s) for s in range(0, 50000) if sm.split_for_seed(s, TASK, None) == SPLIT}
    res = {"schema": "e2e-r1.val_geoms.v1", "scanned_at": time.strftime("%F %T"), "task": TASK, "split": SPLIT, "count": COUNT,
           "seed_range": [SEED_START, SEED_MAX], "geometry_version": sm.SPLIT_SALT, "seeds": seeds, "gids": gids,
           "sources": src, "excluded_gids": len(gids_ex), "val_gids_total_in_0_49999": len(val_all),
           "val_gids_excluded": len(val_all & gids_ex), "val_gids_remaining": len(val_all - gids_ex),
           "seeds_used_elsewhere_in_range": used_in_range, "overlap_with_dataset_gids": sorted(set(gids) & set(src["dataset"]["gids"])),
           "elapsed_s": round(time.time() - t0, 1)}
    res["seeds_sha256"] = hashlib.sha256(",".join(map(str, seeds)).encode()).hexdigest()
    if not args.no_verify:
        chk = verify_with_scene_gen(seeds, gids)
        res["scene_gen_check"] = {"ok": all(c["ok"] for c in chk), "n": len(chk), "rows": chk}
        if not res["scene_gen_check"]["ok"]:
            print("scene_gen 核对不过：", [c for c in chk if not c["ok"]], file=sys.stderr)
            return 2
    if args.check:
        with open(args.check, encoding="utf-8") as f:
            prev = json.load(f)
        same = prev["seeds"] == seeds
        print(f"check: 清单{'一致' if same else '不一致'}；记录 {prev['seeds']}，现算 {seeds}（排除集 {prev['excluded_gids']} → {len(gids_ex)} 个 gid；别的卡后来新采新评会使它变大）")
        return 0 if same else 1
    print(json.dumps({k: v for k, v in res.items() if k not in ("sources", "scene_gen_check")}, ensure_ascii=False, indent=1))
    if "scene_gen_check" in res:
        print(f"scene_gen 核对：{res['scene_gen_check']['n']} 个 seed，gid 与分区全部一致" if res["scene_gen_check"]["ok"] else "scene_gen 核对不过")
    print("sources:", json.dumps(src, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
