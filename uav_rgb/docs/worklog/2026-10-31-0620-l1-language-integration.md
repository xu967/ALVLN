# L1 语言任务集成（scene_gen / teacher / collect / audit）

- 时间：2026-10-31 06:00–06:30
- 范围：`sim/tools/scene_gen.py`、`landmarks.py`、`teacher.py`、`collect.py`、
  `collect_parallel.py`、`eval_closed_loop.py`、`audit_dataset.py`、测试
- 结论：L1 链路打通并通过 2 集冒烟；52 集 pilot 待跑。

## 做了什么

- `landmarks.py`：修正为从**前排视野带（x≤3.0）**选两根 y 不同的柱
  （实际场景每 x 只有一根柱，原先「同排」假设不成立）；
  提供 `COLOR_RGB` 与 `waypoint_for`。
- `scene_gen.py`：`--task l1`（默认 base 不变）：给两根柱上色，写
  `task_version=l1.v1` / `target_color` / `instruction` / `waypoint`；
  base 场景的 meta 和 SDF 不变。
- `teacher.py`：`plan_action_global(..., waypoint=None)`；有 waypoint 时
  `A*(start→wp) + A*(wp→end)` 拼接；`TEACHER_VERSION` bump 到 v4。
- `collect.py`：从 scene meta 读 instruction/target_color/waypoint/task_version，
  写进 manifest；teacher 把 waypoint 传给 `plan_action_global`；student 用
  `instruction.tokenize` 生成 `text_ids` 传给 `StudentPolicy`。
- `collect_parallel.py` / `eval_closed_loop.py`：新增 `--task {base,l1}`，
  透传给 scene_gen；`collect_parallel` 同时透传 `--window-from`。
- `audit_dataset.py`：新增 `target_color` / `landmark_balance` /
  `teacher_waypoint`（safe 出生集不纳入 waypoint 检查）；facts 记录
  target_color/waypoint/scene_spawn/waypoint_min_dist；label_recompute 传 waypoint。
- 测试：`test_instruction`、`test_landmarks`、`test_audit_dataset`、
  `test_collect`（21 项）、`teacher.py` 自检、`scene_gen --self-check` 全过。

## 没做什么

- **没有跑 52 集 pilot**：2 集冒烟已过，pilot 下一步。
- **没有训练语言模型**：L2 文本编码/融合未开始。
- **没有改 base 场景语义**：`--task base` 默认路径不变，T5b 复评不受影响。
- **safe 出生集不验证 waypoint**：它走局部 plan_action，是 SAFE 数据来源，
  与 route-choice 指令无关；audit 明确跳过。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `scene_gen --self-check --check-seeds 20` | PASS |
| `scene_gen --seed 42 --task l1` | target green / instruction / waypoint；前排两色 |
| 2 集 L1 冒烟（observation 语义） | 2/2 集，80/80 步 |
| `instruction_coverage` | 2/2 PASS |
| `target_color` | 0/2 不一致 PASS |
| `landmark_balance` | 目标最近 1 / 非最近 1 PASS |
| `teacher_waypoint` | 0 集 >1.0 m PASS |
| `label_recompute` | 0/80 PASS |
| `class_share` | FAIL（2 集无 SAFE，样本太小，预期） |

## 风险与遗留

- 彩色柱是否在 RGB 帧里清晰可见，需 52 集 pilot 抽帧确认。
- safe 出生集不跟 waypoint，语言增益主要看非 safe 集；分析时要分列。
- waypoint 距离阈值 1.0 m 是初值，pilot 后按实测调整并写进 EVAL_STANDARD。

## 更新 2026-10-31 07:30：120 集 L1 pilot 与 waypoint 评测口径

- 120 集 / 4800 步，train 89 / val 10 / test 21；1.883 步/s，42.5 min。
- audit **15/15 PASS**：instruction 120/120、target 0 不一致、balance 60/60、
  teacher_waypoint 0 失败、safe_collision 0/960、class_share 全过、label 0/4800。
- 52 集的问题与修复：waypoint 0.7→0.8（7/52 A* fallback）、120 集解决 val 缺 SAFE。
- **评测口径修正**：route-choice success 现在要求 `waypoint_min_dist ≤1.0 m`，
  否则 image-only 可忽略指令冲线；`test_eval_summary` 加两条断言。
- 实验记录：`experiments/2026-10-31-l1-pilot-120.md`。
- 下一步：preprocess → image-only baseline → L2（D-STACK 待拍板）。
