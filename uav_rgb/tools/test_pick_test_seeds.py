#!/usr/bin/env python3
"""评测 seed 清单的几何独立性（D-GEODUP）：e1_pipeline 四组对照里有两个学生（旧 X3 / 新 E1），评测几何必须对两个
学生的训练集合都是没见过的。2026-11-07 第二轮的 20 个 seed 只排除了 E1 集合，10/20 的几何在旧 X3 学生的训练数据
（t5-full-1155 train split）里出现过。

运行（仓库根）：.venv-tools/bin/python tools/test_pick_test_seeds.py
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import e1_pipeline  # noqa: E402
import pick_test_seeds as pts  # noqa: E402
import split_manifest as sm  # noqa: E402

X3_TRAIN_MANIFEST = "/root/rivermind-data/uav_rgb/t5-full-1155/collection_manifest.json"   # 旧 X3 学生（T6）训练数据


def main():
    # 1. 默认排除清单必须含旧 X3 学生的训练集合，不只是 E1 集合。
    assert X3_TRAIN_MANIFEST in pts.EXCLUDE_COLLECTIONS, pts.EXCLUDE_COLLECTIONS
    seeds = [int(s) for s in e1_pipeline.CFG["eval_seeds"].split(",")]
    gids = [sm.group_id_for_task(s, "base") for s in seeds]
    # 2. 几何两两不同、v5 下都是 test。
    assert len(set(gids)) == len(seeds), (len(set(gids)), len(seeds))
    assert all(sm.split_for_seed(s, "base") == "test" for s in seeds), seeds
    # 3. 与排除清单里每个集合的几何（不分 split）都不相交；清单文件不在本机时跳过并提示。
    for path in pts.EXCLUDE_COLLECTIONS:
        if not os.path.isfile(path):
            print(f"跳过（本机没有）: {path}")
            continue
        hit = sorted(s for s, g in zip(seeds, gids) if g in pts.gids_in_collection(path, "base", None))
        assert not hit, f"e1_pipeline.CFG['eval_seeds'] 里 {len(hit)} 个 seed 的几何出现在 {path}: {hit}"
    print(f"test_pick_test_seeds: PASS ({len(seeds)} seeds, {len(set(gids))} gids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
