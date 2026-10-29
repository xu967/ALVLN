# 几何组 ID 与冻结 split 清单（PARALLEL_TRAINING_PLAN §19.1）

- 时间：2026-10-29 18:20
- 范围：`sim/tools/split_manifest.py`（新）、`sim/tools/test_split_manifest.py`（新）、
  `sim/tools/collect_parallel.py`（split 策略接线）、`docs/results.md`
- 结论：完成。几何组去重与 split 分配已落地并经真实采集冒烟验证

## 做了什么

- 新增 `sim/tools/split_manifest.py`（约 180 行）：
  - `geometry_gid(pillars)`：x/y/z/size 按 3 位小数格式化（等于 `scene_gen` 的输出精度）、
    按坐标排序、取 sha256 前 16 位。name / 顺序 / scene_id / seed 全部不参与。
  - `family_id(template_sha)`：模板 sha256 + 布局常量 + `LAYOUT_PATTERN` 取 hash。
    与 seed 无关，所以当前全部场景同族。
  - `split_for_gid(gid)`：`sha256("uavrgb.geom.v1:" + gid)` 前 8 位 % 100，
    `<80` train、`<90` val、其余 test。同 gid 永远同 split，与 seed 范围无关。
  - `build_manifest(seeds)`：产出 `navlab.split_manifest.v1`，含每组 gid/family/split/seeds。
  - `cross_split_groups(pairs)` + `--verify <collection_manifest.json>`：审计已有数据。
  - 只有 1 个场景族时向 stderr 打警告，`summarize()` 给出 `family_holdout=false`。
- 新增 `sim/tools/test_split_manifest.py`：8 项 unittest，含「已知重复对 1023/1069 同组」
  和 HANDOFF 数字回归（2000 seed → 241 组跨 split、1124 种几何）。
- 改 `sim/tools/collect_parallel.py`：
  - 删掉 `split_for(scene_id)`（旧 `scene_id % 10` 策略的唯一实现，全仓库只有一处调用）。
  - `Task.split` 初值改成空串，在世界生成、读到 `scene_meta["pillars"]` 之后再填。
  - `SPLIT_POLICY` 与模块 docstring 改成新策略文本。
- `docs/results.md` 追加一节，记录三组真实泄漏 seed 和新旧策略对比。

## 没做什么

- **没有重标已有数据**（`/tmp/ds200` 的 200 条仍带旧 split）。
  - 原因：PARALLEL_TRAINING_PLAN §6 明确「现有数据不原地重标；新质量规范生成新数据版本」。
    要换 split 只能重新采集，不是改几个 json 字段。
- **没有做场景族留出**（`family_holdout=false`）。
  - 原因：`scene_gen` 目前只有一种蛇形车道布局，族只有一个，无从留出。需要先加第二种
    布局族，那是生成器的改动，不在本次范围。
- **没有改 `dataset.cpp` / `tools/preprocess.py`**。
  - 原因：读代码发现 split 本来就从每个 episode 的 `manifest.json` 流进 UAVR meta，
    加载器只做「同 scene_id 不许跨 split」的校验，不重新推导策略。所以只需改采集端的
    一处分配即可，加载端零改动。
- **没有把 split manifest 落盘到采集输出目录**。
  - 原因：下一次采集的 seed 范围还没定；`build_manifest` 按需生成更简单，避免留一份
    会过期的清单。
- **没有验证 `--drop-collision` 之后的样本级分布**。
  - 原因：本次只处理 scene 级划分；样本级类别分布是另一个指标，EVAL_STANDARD §5 已记录。
- **没有跑 200 条规模的重新采集**。
  - 原因：需要约 13 分钟采集 + 后续训练，属于下一格任务；本次只做 3 条冒烟证明接线。
- **没有做族的自动发现**（从几何反推布局族）。
  - 原因：当前只有一族，自动发现没有可验证的输入；等第二种布局族落地再做。
- **没有跑 `bash tools/check.sh` 全量**。
  - 原因：本次只动 Python 工具，check.sh 覆盖的是 C++ 构建与 CTest，已单独跑过 16/16。

## 待确认

- 下一次正式采集的 `--base-seed` 范围。几何组策略下 split 由 gid 决定，不再与
  `base-seed` 的偏移有关；但 val/test 的**组数**仍取决于总量（200 seed → 约 17/18 组）。
- 是否要在扩量前先加第二种布局族。如果不加，采到 2000 条仍是一个模板，
  test 上的数字依旧只能证明「同模板内泛化」。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python3 sim/tools/test_split_manifest.py` | **8/8 通过**，0.640 s |
| `python3 sim/tools/test_collect_parallel.py` | 2/2 通过 |
| `python3 sim/tools/test_collect.py` | 13 项通过 |
| `python3 sim/tools/test_save_frame.py` | PASS |
| `python3 -m py_compile sim/tools/collect_parallel.py` | 通过 |
| `split_manifest.py --verify /tmp/ds200/collection_manifest.json` | 报 3 组跨 split，**exit=1**（预期） |
| `split_manifest.py --verify /tmp/split_smoke/collection_manifest.json` | 0 组，**exit=0** |
| 真实采集冒烟（1 实例 × 3 episode × 3 步，base-seed=1000） | 3/3 episode、9/9 步；split = train/val/train，与目录、episode manifest 三处一致 |
| `cmake --build src/build` | 通过（本次未改 C++，属于复测） |
| `ctest --test-dir src/build` | **16/16 通过**，94.99 s（文档里写的 13 项已过期） |

冒烟采集原始输出：

```
train: scene_id=[1000, 1002]
val:   scene_id=[1001]
seed 1000 4bfdeeb8b919440e train
seed 1001 c724ab209b796e35 val
seed 1002 e789ad9d01dced06 train
```

## 风险与遗留

- **`SPLIT_SALT` 是唯一让旧清单失效的开关。** 改 `scene_gen` 的柱子尺寸、车道间距或
  采样模式会让所有 gid 变，必须同步 bump `SPLIT_SALT` 并开新数据版本；不 bump 会出现
  「同物理场景、不同 split」的静默泄漏。
- **几何组去重只解决精确重复，不解决模板捷径。** 两个几何差 1 mm 的 seed 仍是不同的组，
  仍可能一个在 train 一个在 test。`family_holdout=false` 的警告必须在每次报告里带上。
- **`--verify` 的 seed 用的是 `collection_manifest.json` 的 `splits` 字段**，那里存的是
  scene_id。当前 `seed == scene_id`（`collect_parallel` 里两者同源），但若以后解耦，
  verify 会算错，需要改成读每个 episode 的 manifest。
- **旧数据（`/tmp/ds200`）的解释力进一步下降。** 它在几何层面有 3 组泄漏，
  加上本来就只有 1 个场景族，test 的 0.9733 不能作为任何泛化结论。
- `test_split_manifest.py` 里写死了 241 和 1124 两个数字。改 `scene_gen` 会失败，这是故意的
  —— 失败即提醒同步 bump salt。
