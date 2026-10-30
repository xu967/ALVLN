# L1 语言条件任务设计（2026-10-31）

- 目标：在现有 Gazebo 柱阵里加「目标颜色 + 干扰色」，让指令决定走哪，
  用同一套 4 动作接口做最小语言条件闭环。
- 前置：T5b（dispatch 语义）采完/训练完，基线合同确定后再动 scene_gen/collect。
- 本文只设计，未改代码。

## 1. 两个变体，先做小的

| 变体 | 成功判据 | teacher | 改动量 | 结论 |
| --- | --- | --- | --- | --- |
| **A route-choice** | 保留 `max_x ≥ goal_x`（现有走廊终点） | A\* 加一个彩色 waypoint | 小 | **先做** |
| B goal-stop | 距目标柱 ≤1.5 m 且无碰撞 | goal-aware A\* + 新 evaluator | 大 | A 跑通后再做 |

A 的语言必要性：同一画面里两根彩色柱，指令只指定其中一根；
image-only 基线不知道选哪根，文本打乱会走错。这已经能回答 RQ0。

## 2. 场景改造（scene_gen.py）

- `_pillar_xml(p)`：按 `p["color"]` 设材质；默认灰 `0.6 0.6 0.6 1`；
  红 `0.8 0.05 0.05`、蓝 `0.05 0.1 0.8`、黄 `0.9 0.8 0.05`、绿 `0.05 0.7 0.1`。
- `assign_landmarks(pillars, seed)`：
  - 从非出生区选 2 根柱（目标/干扰），颜色不同；
  - 目标色由 seed 决定，**目标不总是离出生点最近**（否则模型可用位置捷径）；
  - 写 `pillar["color"]`、`meta["target_color"]`、`meta["instruction"]`
    （用 `instruction.make_instruction(target_color)`）、`meta["task_version"]="l1.v1"`。
- `geometry_gid` 仍只按布局（x/y/z/size），不把 color 纳入；L1 用新数据目录
  `l1-*`，不与 t5/t5b 混包。`geometry_version` 不 bump（布局未变），
  用 `task_version` 区分任务。
- 可见性检查：至少 2 根彩色柱落在相机前向 FOV 内；先做静态坐标检查 +
  50 集 pilot 人工抽检 20 帧，不建渲染 oracle。

## 3. 数据与教师

- `collect.py`：从 `--scene-meta` 读 `instruction`/`target_color`，写入
  `episode_manifest.json`；`StudentPolicy(text_ids=tokenize(instruction))`。
- teacher：route-choice 模式下 A\* 先到目标柱附近的 waypoint，再到走廊终点；
  `label_recompute` 用同一 waypoint 复算。若 waypoint 不可达，该集不采。
- `preprocess.py`：已支持 `instruction` 侧车（L0），无需改格式。
- `audit_dataset.py`：
  - `instruction_coverage`（已实现）；
  - 新增 `target_color`：指令里的唯一颜色 == `meta.target_color`，
    且该颜色柱存在；
  - 新增 `landmark_balance`：目标不总是最近柱；两色在 split 内都有。
- 注意：L1 数据的 `geometry_gid` 与 T5 相同布局会落同 split；这是有意的
  （同布局、不同颜色任务），但**不能与 T5 数据混训**，用 `task_version` 和
  独立目录隔离。

## 4. 评估

- 冻结 test seed + 指令集；同一批场景跑：
  - 语言版；
  - image-only（文本置空）；
  - text shuffled（随机换一条指令）；
  - 同图换指令反事实对（同一场景，目标色互换）。
- 临时阈值（正式 test 前冻结，只准 pilot 后改一次）：
  - 语言版 SR ≥ image-only + 10 pp；
  - shuffled ≤ 语言版 − 10 pp；
  - 反事实一致率 ≥ 60%。
- 先 2–3 个 seed 冒烟，再 50 集 pilot；不达标先改任务设计（目标太显眼/
  总是最近/干扰不可见），不要加模型。

## 5. 实施顺序（T5b 结束后）

1. `scene_gen.py` 颜色 + `assign_landmarks` + 自测；
2. `collect.py` 写 instruction + `StudentPolicy` token；
3. `audit_dataset.py` target_color/landmark_balance + 负例；
4. 50 集 pilot → audit；
5. L2 文本编码/融合（栈按 D-STACK 决定）。

## 6. 风险

- 颜色受光照/材质影响，RGB 里可能偏色；pilot 先看帧。
- 目标柱可能被灰柱遮挡；静态 FOV 检查 + 抽检。
- teacher waypoint 可能不可达；不可达集直接不采，不做假标签。
- 若目标总是最近/最显眼，语言增益是假的；audit 的 landmark_balance 必须过。
