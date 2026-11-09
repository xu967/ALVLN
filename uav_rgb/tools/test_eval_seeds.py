#!/usr/bin/env python3
"""tools/eval_seeds.py 是冻结评测 seed 的唯一来源（FIXLIST_R2 A3）。

1. 两个清单各 20 个 seed、两两几何不同、按各自任务都落 test 分区；
2. e1_pipeline / compare_datasets / compare_teachers 都从它读；
3. 仓库 tools/ 与 sim/tools/ 的 .py 里不再有第二份字面量（dagger_pipeline:312、compare_teachers:33 之类）。

运行（仓库根）：.venv-tools/bin/python tools/test_eval_seeds.py
"""
import glob
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import split_manifest as sm  # noqa: E402
import eval_seeds as es  # noqa: E402

FROZEN_BASE = (1, 113, 122, 153, 172, 176, 180, 230, 250, 536, 555, 628, 683, 798, 929, 990, 992, 997, 1113, 1654)
FROZEN_FOREST = (7, 8, 19, 22, 23, 39, 53, 64, 71, 86, 90, 102, 110, 134, 135, 136, 142, 145, 155, 161)
OLD_LISTS = ("180,206,250,485,503,555,586,708,715,721,799,855,932,992,1113,1116,1123,1145,1246,1282",
             "180,250,485,555,586,708,721,992,1113,1116,1145,1654,1873,2087,2506")


def test_frozen_lists():
    assert es.EVAL_SEEDS_BASE_20 == FROZEN_BASE, es.EVAL_SEEDS_BASE_20
    assert es.FOREST_SPARSE_SEEDS_20 == FROZEN_FOREST, es.FOREST_SPARSE_SEEDS_20
    assert es.csv(es.EVAL_SEEDS_BASE_20) == ",".join(map(str, FROZEN_BASE))
    import scene_forest
    # gid 算一次、split 由 gid + 该任务的 salt 推（与 split_for_seed 同一公式）；森林几何采样 0.8 s/seed，不重复算。
    for seeds, task, density, salt in ((FROZEN_BASE, "base", None, sm.SPLIT_SALT),
                                       (FROZEN_FOREST, "forest_e1", "sparse", scene_forest.GEOMETRY_VERSION_FOREST)):
        gids = [sm.group_id_for_task(s, task, density) for s in seeds]
        assert len(set(gids)) == 20, f"{task}: 几何组重复"
        assert all(sm.split_for_gid(g, salt) == "test" for g in gids), f"{task}: 有 seed 不在 test"
    print("ok 两个清单 20 独立几何、全 test")


def test_consumers_read_from_single_source():
    import e1_pipeline as ep
    import compare_datasets as cd
    assert ep.CFG["eval_seeds"] == es.csv(es.EVAL_SEEDS_BASE_20), ep.CFG["eval_seeds"]
    assert cd.CFG["gazebo"]["seeds"] == es.csv(es.EVAL_SEEDS_BASE_20), cd.CFG["gazebo"]["seeds"]
    os.environ.pop("UAV_RGB_COMPARE_SEEDS", None)
    import compare_teachers as ct
    assert ct.SEEDS == list(es.EVAL_SEEDS_BASE_20), ct.SEEDS
    print("ok e1_pipeline / compare_datasets / compare_teachers 读 eval_seeds")


def test_no_second_literal_in_code():
    base_csv = es.csv(es.EVAL_SEEDS_BASE_20)
    forest_csv = es.csv(es.FOREST_SPARSE_SEEDS_20)
    files = glob.glob(os.path.join(REPO, "tools", "*.py")) + glob.glob(os.path.join(REPO, "sim", "tools", "*.py"))
    offenders = []
    for path in files:
        name = os.path.basename(path)
        if name in ("eval_seeds.py", "test_eval_seeds.py"):
            continue
        text = open(path, encoding="utf-8").read()
        for literal in (base_csv, forest_csv) + OLD_LISTS:
            if literal in text and not name.startswith("test_"):
                offenders.append((name, literal[:20] + "…"))
    assert not offenders, f"代码里还有 seed 字面量，应从 tools/eval_seeds.py 读：{offenders}"
    print("ok tools/ 与 sim/tools/ 非测试代码无第二份字面量")


FROZEN_BASE_60 = (1, 37, 58, 113, 122, 127, 132, 153, 172, 176, 180, 190, 230, 250, 256, 259, 270, 312, 313, 327,
                  361, 365, 379, 395, 422, 485, 536, 555, 586, 592, 617, 628, 683, 708, 713, 721, 771, 785, 798, 847,
                  872, 913, 917, 929, 990, 992, 997, 1018, 1091, 1113, 1116, 1145, 1334, 1353, 1437, 1527, 1564, 1654,
                  1667, 1711)
FROZEN_BASE_X3_40 = (1, 113, 122, 132, 153, 172, 176, 180, 230, 250, 365, 379, 485, 536, 555, 586, 628, 683, 708, 721,
                     785, 798, 929, 990, 992, 997, 1018, 1113, 1116, 1145, 1564, 1654, 1711, 1765, 1873, 2087, 2190,
                     2263, 2305, 2506)


def test_base_60_and_x3_40():
    """M3 n=60（用户 2026-11-09 定的 E1 口径）：60 个两两不同的 test 几何且含冻结 20；X3 那张 = 71 个 test 几何去掉
    旧 X3 学生 train/val 碰过的 31 个，也含冻结 20。t5 清单在数据盘，缺了只跳过最后一条。"""
    assert es.EVAL_SEEDS_BASE_60 == FROZEN_BASE_60, es.EVAL_SEEDS_BASE_60
    assert es.EVAL_SEEDS_BASE_X3_40 == FROZEN_BASE_X3_40, es.EVAL_SEEDS_BASE_X3_40
    for seeds in (FROZEN_BASE_60, FROZEN_BASE_X3_40):
        gids = [sm.group_id_for_task(s, "base") for s in seeds]
        assert len(set(gids)) == len(seeds), "几何组重复"
        assert all(sm.split_for_gid(g, sm.SPLIT_SALT) == "test" for g in gids), "有 seed 不在 test"
        assert set(FROZEN_BASE) <= set(seeds), "冻结 20 应是子集"
    t5 = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "t5-full-1155",
                      "collection_manifest.json")
    if not os.path.isfile(t5):
        print(f"skip X3 train/val 复核：缺 {t5}")
    else:
        import json
        splits = json.load(open(t5, encoding="utf-8"))["splits"]
        x3_tv = {sm.group_id_for_task(s, "base") for k in ("train", "val") for s in splits.get(k, [])}
        hit = [s for s in FROZEN_BASE_X3_40 if sm.group_id_for_task(s, "base") in x3_tv]
        assert not hit, f"X3 清单里有旧 X3 学生训练过的几何：{hit}"
    print("ok base 60 / X3 40：几何两两不同、全 test、含冻结 20；X3 清单不含 t5 train/val 几何")


def main():
    test_base_60_and_x3_40()
    test_frozen_lists()
    test_consumers_read_from_single_source()
    test_no_second_literal_in_code()
    print("test_eval_seeds: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
