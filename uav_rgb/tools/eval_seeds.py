#!/usr/bin/env python3
"""冻结评测 seed 的唯一来源（FIXLIST_R2 A3，2026-11-09）。别把这些数字复制到别的文件：从这里 import。

两张清单都由 tools/pick_test_seeds.py 挑出（按几何组 ID 去重、按 split_manifest 复算分区），
tools/test_eval_seeds.py 锁定：20 个 seed 两两几何不同、按各自任务都是 test。

用法：
  from eval_seeds import EVAL_SEEDS_BASE_20, FOREST_SPARSE_SEEDS_20, csv
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

# forest_e1 sparse（随机圆柱森林，最窄缝 0.45 m），2026-11-07 森林跨场景评测 xeval-forest-sparse-20261107-1600 用它。
# pick_test_seeds.py --task forest_e1 --density sparse（同一排除清单；森林几何按该 density 的树干几何查重）。
FOREST_SPARSE_SEEDS_20 = (7, 8, 19, 22, 23, 39, 53, 64, 71, 86,
                          90, 102, 110, 134, 135, 136, 142, 145, 155, 161)


def csv(seeds):
    """eval_closed_loop --seeds 要的逗号串。"""
    return ",".join(str(s) for s in seeds)


if __name__ == "__main__":
    print("base:", csv(EVAL_SEEDS_BASE_20))
    print("forest_e1 sparse:", csv(FOREST_SPARSE_SEEDS_20))
