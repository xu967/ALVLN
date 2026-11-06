#!/usr/bin/env python3
"""挑一组 test 评测 seed：几何两两不同、都在 test 分区、不与训练集合里的任何几何组重合。

背景：2026-11-07 E1 流水线用的 20 个「冻结 test seed」按 gid 去重后只剩 11 个独立几何
（eval_closed_loop 会跳过重复 gid：`跳过: 已评测 <gid>`），四组对照实际 n=11。这是 TASKS.md D-GEODUP
一直没做的检查。本脚本按几何组 ID 挑，输出能直接贴进 e1_pipeline.CFG["eval_seeds"] 的逗号串。

运行（仓库根）：
  .venv-tools/bin/python tools/pick_test_seeds.py                          # 默认配置，打印 20 个 seed
  .venv-tools/bin/python tools/pick_test_seeds.py --exclude-collection <collection_manifest.json> ...
  .venv-tools/bin/python tools/pick_test_seeds.py --self-check
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import split_manifest as sm  # noqa: E402

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
COUNT = 20            # 要多少个独立几何；EVAL_STANDARD 建议 ≥20 个 test 场景
SEED_START = 0        # 从这个 seed 起向上扫；采集段 3000–3399 / 4100+ 不在这个范围也没关系，按 gid 排除
SEED_MAX = 50_000     # 扫描上限，防写错 split 时死循环
TASK = "base"         # 柱阵 base 任务；forest_e1 要配 --density
SPLIT = "test"
EXCLUDE_COLLECTIONS = (   # 训练集合的 collection_manifest.json：其中出现过的几何组一律排除（哪怕它落 test）
    "/root/rivermind-data/uav_rgb/e1-pipeline-20261107-0417/collection/collection_manifest.json",
)
# ──────────────────────────────────────────────────────────────────────────


def gids_in_collection(path, task, density):
    """集合里所有 seed 的几何组 ID（按 split_manifest 的任务身份函数复算，不信任清单里的 split）。"""
    with open(path, encoding="utf-8") as f:
        man = json.load(f)
    seeds = [s for seeds in man["splits"].values() for s in seeds]
    return {sm.group_id_for_task(s, task, density) for s in seeds}


def pick(count, task, split, exclude_gids, density=None, seed_start=SEED_START, seed_max=SEED_MAX):
    """返回 (seeds, gids)：按 seed 升序，几何组两两不同、分区为 split、不在 exclude_gids。"""
    seeds, gids = [], []
    seen = set(exclude_gids)
    for seed in range(seed_start, seed_max):
        if sm.split_for_seed(seed, task, density) != split:
            continue
        gid = sm.group_id_for_task(seed, task, density)
        if gid in seen:
            continue
        seen.add(gid)
        seeds.append(seed)
        gids.append(gid)
        if len(seeds) >= count:
            return seeds, gids
    raise SystemExit(f"扫到 seed {seed_max} 只凑到 {len(seeds)}/{count} 个独立 {split} 几何")


def self_check():
    seeds, gids = pick(5, "base", "test", set())
    assert len(seeds) == 5 and len(set(gids)) == 5, (seeds, gids)
    assert all(sm.split_for_seed(s, "base") == "test" for s in seeds)
    # 排除前 3 个几何后再挑，前 3 个 seed 不该再出现
    seeds2, _ = pick(5, "base", "test", set(gids[:3]))
    assert not set(seeds[:3]) & set(seeds2), (seeds, seeds2)
    print("pick_test_seeds self-check PASS", seeds)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--count", type=int, default=COUNT)
    p.add_argument("--task", default=TASK)
    p.add_argument("--density", default=None, choices=(None, "sparse", "medium", "dense"))
    p.add_argument("--split", default=SPLIT, choices=("train", "val", "test"))
    p.add_argument("--exclude-collection", action="append", default=None,
                   help="collection_manifest.json；可重复；缺省用配置区 EXCLUDE_COLLECTIONS")
    p.add_argument("--self-check", action="store_true")
    args = p.parse_args()
    if args.self_check:
        self_check()
        return
    excludes = args.exclude_collection if args.exclude_collection is not None else list(EXCLUDE_COLLECTIONS)
    exclude_gids = set()
    for path in excludes:
        if os.path.isfile(path):
            exclude_gids |= gids_in_collection(path, args.task, args.density)
        else:
            print(f"警告：排除清单不存在，跳过：{path}", file=sys.stderr)
    seeds, gids = pick(args.count, args.task, args.split, exclude_gids, args.density)
    print(json.dumps({"task": args.task, "density": args.density, "split": args.split, "count": len(seeds),
                      "excluded_gids": len(exclude_gids), "seeds": seeds, "gids": gids}, ensure_ascii=False, indent=1))
    print("eval_seeds=" + ",".join(map(str, seeds)))


if __name__ == "__main__":
    main()
