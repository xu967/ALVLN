#!/usr/bin/env python3
"""冻结评测 seed 的唯一来源（FIXLIST_R2 A3，2026-11-09）。别把这些数字复制到别的文件：从这里 import。

两张 20 个的清单由 tools/pick_test_seeds.py 挑出（按几何组 ID 去重、按 split_manifest 复算分区）；n=60 与旧 X3
的 40 个清单的取法写在各自上方。tools/test_eval_seeds.py 锁定：每张两两几何不同、按各自任务都是 test，60 / 40
两张都含冻结 20。

用法：
  from eval_seeds import EVAL_SEEDS_BASE_20, EVAL_SEEDS_BASE_60, EVAL_SEEDS_BASE_X3_40, FOREST_SPARSE_SEEDS_20, csv
  "--seeds", csv(EVAL_SEEDS_BASE_20)
"""

# base 柱阵（lib v5 几何），2026-11-07 第三轮起的四/五组对照与 stats 表都用它（e1-pipeline-20261107-0417-eval3 起）。
# pick_test_seeds.py 默认配置：COUNT=20、SEED_START=0、TASK=base、SPLIT=test，排除清单（EXCLUDE_COLLECTIONS）
#   - e1-pipeline-20261107-0417/collection/collection_manifest.json（新 E1 学生的 300 集训练集合）
#   - t5-full-1155/collection_manifest.json（旧 X3 学生 T6 ckpt 的 600 集训练数据，v3 盐分 split）
# 即与两个被评测学生的训练几何都零重叠（tools/test_pick_test_seeds.py 复算）。
# 更早的 180,206,…,1282（v3/v5 双 test）按 gid 去重只剩 11 个独立几何；0417-eval2 的 1,58,113,…,536 只排除了
# E1 集合，10/20 的几何在 X3 学生的 train 里——两张旧清单都不再用。
EVAL_SEEDS_BASE_20 = (1, 113, 122, 153, 172, 176, 180, 230, 250, 536,
                      555, 628, 683, 798, 929, 990, 992, 997, 1113, 1654)

# M3 的 n=60（用户 2026-11-09 22:55 定的 E1 口径）。base 任务 seed 0–49999 一共只有 71 个 test 分区几何；E1 学生
# （含 DAgger 各轮）按几何组分区打包，test 分区几何从没进过训练，所以 71 个对 E1 臂都是保留几何。取法：冻结 20 个
# 全留，再按 seed 升序补 40 个几何不同的 test seed。现行严格排除（三个采集集碰过的全排）只剩 24 个，凑不出 60。
EVAL_SEEDS_BASE_60 = (1, 37, 58, 113, 122, 127, 132, 153, 172, 176, 180, 190, 230, 250, 256, 259, 270, 312, 313, 327,
                      361, 365, 379, 395, 422, 485, 536, 555, 586, 592, 617, 628, 683, 708, 713, 721, 771, 785, 798,
                      847, 872, 913, 917, 929, 990, 992, 997, 1018, 1091, 1113, 1116, 1145, 1334, 1353, 1437, 1527,
                      1564, 1654, 1667, 1711)
# 旧 X3 学生（T6 ckpt）只在它没见过的几何上评：71 个 test 几何去掉 t5-full-1155 的 train/val 碰过的 31 个，剩 40 个
# （冻结 20 个全在里面；与上面 60 个重合 33 个，每个几何取同一 seed）。
EVAL_SEEDS_BASE_X3_40 = (1, 113, 122, 132, 153, 172, 176, 180, 230, 250, 365, 379, 485, 536, 555, 586, 628, 683, 708,
                         721, 785, 798, 929, 990, 992, 997, 1018, 1113, 1116, 1145, 1564, 1654, 1711, 1765, 1873, 2087,
                         2190, 2263, 2305, 2506)

# forest_e1 sparse（随机圆柱森林，最窄缝 0.45 m），2026-11-07 森林跨场景评测 xeval-forest-sparse-20261107-1600 用它。
# pick_test_seeds.py --task forest_e1 --density sparse（同一排除清单；森林几何按该 density 的树干几何查重）。
FOREST_SPARSE_SEEDS_20 = (7, 8, 19, 22, 23, 39, 53, 64, 71, 86,
                          90, 102, 110, 134, 135, 136, 142, 145, 155, 161)


def csv(seeds):
    """eval_closed_loop --seeds 要的逗号串。"""
    return ",".join(str(s) for s in seeds)


if __name__ == "__main__":
    print("base:", csv(EVAL_SEEDS_BASE_20))
    print("base 60:", csv(EVAL_SEEDS_BASE_60))
    print("base X3 40:", csv(EVAL_SEEDS_BASE_X3_40))
    print("forest_e1 sparse:", csv(FOREST_SPARSE_SEEDS_20))
