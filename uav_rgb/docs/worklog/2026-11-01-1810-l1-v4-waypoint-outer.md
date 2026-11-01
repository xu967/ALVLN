# L1 v4：waypoint 改到柱子外侧，重采重训

- 时间：2026-11-01 17:50 起
- 范围：`sim/tools/landmarks.py`、`make_counterfactual.py`、`scene_gen.py`、
  `eval_closed_loop.py`、`test_landmarks.py`；数据在 `$DATA_ROOT/uav_rgb/l1-v4-*`
- 结论：进行中。代码已改并全绿，500 集采集中。

## 背景

见 `docs/worklog/2026-11-01-1739-code-review-waypoint-degeneracy.md`：
v3 的 274 个 test 场景**没有一个**要求按关系词横移（两个候选 waypoint 都落在
出生航线上，横移中位 0.09 m），闭环 SR 只由「敢不敢飞 + 会不会停」决定。
用户 2026-11-01 批准改任务并重采重训。

## 做了什么

- `landmarks.waypoint_for`：偏置方向从「朝走廊中心」改成「朝目标柱外侧」，
  新增 `spawn` 入参；删掉两轮 `sign = -sign` 的静默兜底，超界或撞柱直接抛
  `ValueError`（以前会静默产出退化且语义反了的场景）。
- `landmarks.assign_landmarks`：relation 与目标柱的映射翻反
  （`target = right if relation == "left"`），保证「pass the pillar on your left」
  字面为真；`waypoint_for` 调用补 `spawn`。
- `make_counterfactual.alt_target`：反事实柱的选择条件跟着翻（`!=` → `==`），
  `waypoint_for` 补 `scene["spawn"]`。它是 `waypoint_for` 的第二个调用方，
  不跟着改会把「同一边」当成反事实。
- `scene_gen.build_scene`：`task_version` `l1.v3` → `l1.v4`（几何版本保持
  `uavrgb.geom.v5`，柱子和出生点没变，split 不变，新旧数据能对同一批场景比较）。
- `test_landmarks.py`：断言翻反，并新增两条回归守卫——waypoint 必须在目标柱外侧、
  横移 >0.9 m。这两条就是本次 bug 的直接守卫。
- `eval_closed_loop.py`：`--text-mode` 的 help 文案从「shuffle=换一个目标色」
  改成「shuffle=把 left/right 反过来」（v3 起就不换色了）。
- `eval_closed_loop.py`：`--max-seeds` 的 help 里 `约 10%` 的 `%` 会被
  `argparse` 当格式符，`--help` 直接抛 `ValueError: unsupported format
  character`。改成 `10%%`。
- `docs/HANDOFF.md`：补 l1.v4 的任务语义说明。
- 冒烟：2 集 `collect_parallel`，16/15 步，`wp_final` 0.29/0.35，出生 y=±0.83/-0.75
  → 结束 y=-0.64/0.86（横移 1.5 m），确认 teacher 能飞出真的横移。
- 起 500 集采集：`l1-v4-relation-500`，与 v3 同参数同 seed 基（900000），
  5 实例 × 100 集 × 60 步上限。

## 采集与验收结果

采集 499/500（`scene_900055` 因 `publisher helper 发布失败: 超时/已退出`
中断，infra 问题，与 v3 排除的 3 集同性质），8966 步，train 404 / val 47 /
test 48。`audit_dataset.py` **17/17 PASS**。

| 指标 | v3（497 集） | v4（499 集） |
| --- | --- | --- |
| 实际横向位移 \|Δy\| p50 | 0.13 m | **1.58 m**（min 1.26 / p90 1.75） |
| waypoint 横移 \|Δy'\| p50 | 0.09 m | 1.61 m（min 1.23） |
| waypoint 在目标柱外侧 0.80 m | 无保证 | 417/417（采集中途子集） |
| FORWARD_SLOW 占比 | 0.714 | 0.660 |
| YAW_L + YAW_R 占比 | 0.078 | **0.174** |
| 步数 p50 / max（采集上限 60） | 14 / 28 | 17 / 28 |
| 反事实标签变化率（train+val） | 0.5993 | **0.9262** |

反事实标签变化率是最直接的判据：v3 有 40% 的帧换指令后 teacher 动作不变
（那张图本来就无法区分），v4 只剩 7.4%。

顺带修掉的三件事：`class_share`（YAW 两类此前各 4%）、`time_pairing`
（max_gap 100 ms → 36 ms）、`safe_collision`（v3 有 1 个 SAFE 步带碰撞）。

## 没做什么

- **没改 `GEOMETRY_VERSION`**（仍是 `uavrgb.geom.v5`）。柱子位置、出生点、split
  都没变，只有目标点和关系词映射变了，用 `task_version` 区分足够；
  `preprocess.py:240` 会按 `task_version` 拦混包。
  - 代价：gid 不变，所以 v3 和 v4 的 train/val/test 是同一批场景——这是有意的，
    方便直接对比，但也意味着 v3 的 test 场景在 v4 里仍是 test（如果担心场景被
    「看过」，v3 的模型并没有在新任务上训练过，不构成泄漏）。
- **没重跑 PO/EO/AGV/MP**：仍未实现（`EVAL_STANDARD §5.2`）。
- **没修 `carry_resume_best`**：跨数据集续训冻结 `ckpt_best.bin` 的坑还在
  `src/trainer.cpp:40-70`。本轮训练不 resume，碰不到。
- **没给 `eval_closed_loop.py` 的 spawn 修复补回归测试**（上一轮改的）。
- 没提交、没推送：用户未授权。

## 待确认

- 无（改动方案已由用户 2026-11-01 批准）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `test_landmarks.py` | PASS（含新增两条外侧/横移守卫） |
| `test_instruction.py` | PASS |
| `scene_gen.py --self-check --check-seeds 300` | 300/300 PASS |
| `bash tools/check.sh`（全量，含 C++ 构建） | PASS，ctest 21/21 |
| 274 个 test 场景几何复算 | 274/274 两边都必须横移；最小横移 1.15 m；两候选点最小间距 3.23 m；0 失败 |
| 2 集冒烟采集 | 2/2，16/15 步，`wp_final` 0.29/0.35，`task_version=l1.v4` |
| 500 集采集 | 499/500（1 集 infra 超时 `scene_900055`），8966 步，train 404 / val 47 / test 48 |
| `audit_dataset.py` | **17/17 PASS**（v3 的 `time_pairing` / `class_share` / `safe_collision` 三个 FAIL 全消） |
| 打包 / 反事实集 / 训练 / 离线 / 闭环 | 训练中 |

## 风险与遗留

- **新的成功半径关系**：横移 1.6 m vs 成功半径 0.7 m（stop_radius 0.5 + 0.2）。
  错边现在是 3.2 m 外，不会误判；但「飞对边但停歪 0.8 m」仍会算失败——
  这和 v3 口径一致，没有放宽。
- **episode 会变长**：v3 是 p50 14 步 / max 28 步；新任务 teacher 实测 15–18 步，
  采集的 60 步上限和闭环评测的 `--steps 40` 都还够，但学生在闭环里可能更接近上限，
  看结果再说。
- **v3 的全部实验数字作废**（任务定义变了），`docs/experiments/` 里的 v3 记录保留
  作为历史，不能和新数字并列比较。
