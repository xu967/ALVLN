# T1：车缝 1.2 m + A* 教师

- 时间：2026-10-30 04:00
- 范围：`sim/tools/scene_gen.py`、`split_manifest.py`、`teacher.py`、`collect.py`、
  `collect_parallel.py`、`audit_dataset.py`、`tools/preprocess.py`、`sim/tools/` 测试
- 结论：**完成**。几何 v3（缝 1.2 m）+ A* 教师；20 个 test 场景、20 个唯一几何，
  40 步/集，**20/20 成功、0 碰撞、穿越率 100%**（验收线 ≥90%）。
  关键发现：只放宽几何不够；局部射线教师在 34°/步偏航下会在门柱前摆振
  （24 集 15 集擦柱），必须换 A* 全局路径 + 25° 纯追踪容差。

## 做了什么

- `scene_gen.py`：v3 几何 `LANE_SEP=1.6`（缝 1.2 m）、`PILLAR_Y_RANGE=±1.7`、
  `LANE_CENTER_ABS_MAX=0.1`；新增 `GEOMETRY_VERSION`、出生位姿
  `default/random/safe`、`SPAWN_X=-0.5`；meta 写 `geometry_version/spawn/lane_gap_m`。
- `split_manifest.py`：`SPLIT_SALT = scene_gen.GEOMETRY_VERSION`，单一来源 v3。
- `teacher.py`：
  - `ray_clear` 增加 `skip_clearance_m`，`plan_action` 在正常射线全堵时用 0.2 m
    脱困回退（修复「漂进 0.36 m 余量后永远 SAFE」）；
  - 局部 `tol=15°`、`Y_GAIN=0`（避免门柱前回中线摆振）；
  - 新增 `astar_path` + `plan_action_global`（0.1 m 栅格、0.5 m 膨胀、0.7 m carrot、
    `tol=25°`、路径目标 = 最右柱 x+2.0，比验收线多 1 m）；
  - 自测补 A* 绕门柱、脱困回归。
- `collect.py`：普通集用 `plan_action_global`，safe 集用局部 `plan_action`；
  manifest 写 `geometry_version/planner`；`take_off` 默认 timeout 30s→**120s**
  （8 实例 RTF 0.2 时爬升要 ~60s 墙钟）。
- `collect_parallel.py`：`--split/--seeds/--spawn-mode/--safe-spawn-fraction`；
  manifest 记 `unique_geometry`；`SPLIT_POLICY` 改 v3。
- `audit_dataset.py`：跳过 `failed/`；新增 `safe_collision`、`geometry_version`；
  `class_share` 带 per-split 计数；`--exclude-collision-episodes`。
- `tools/preprocess.py`：拒绝 `geometry_version` 不匹配、`stopped_on_collision`、
  validity 非全 true；新增 `--exclude-collision-episodes`；meta 写每 split 类别计数、
  `teacher/geometry_version`；`make_synthetic_episode.py` 同步 fixture。
- `sim/tools/traverse_rate.py`（新）：穿越率口径；`test_spawn.py`（新）：safe spawn
  自检；`test_split_manifest.py` 更新 v3 指纹（291 跨 split / 759 几何）。
- `eval_closed_loop.py`：`--policy student|forward`、`--seeds`、起 sim 前按 gid 去重、
  report 写 `policy/seeds`（T7 用；本任务只做代码，不跑闭环）。

## 没做什么

- **没有保留 80 步/集**：A* 直线段实测 0.27 m/步，40 步已走 ~10.8 m；当前
  cgroup 只有 12 核且共享，80 步墙钟翻倍。T1 最终用 40 步；T2/T3 同。
  - 原因：CPU 配额 + 旧 80 步是按 0.105 m/步估的，已过时。
- **没有在同一个 run 里跑 8 实例**：cgroup 12 核下 8 实例 RTF 0.1，
  21/24 集起飞超时；最终用 4 实例。T5 大规模采集需要按当时负载选 4–6。
- **没有重跑 v2 的 8 场景消融**：旧数字只作历史对照，不是 v3 同口径。
- **没有把 A* 接进闭环评测**：闭环评的是学生；教师只在采集侧。
- **没有给 A* 做多线程/缓存优化**：6 ms/步，80 步 0.5 s，相对 Gazebo 可忽略。
- **没有验证 safe 出生集在 A* 下的行为**：safe 集刻意走局部规划，保留真 SAFE。

## 待确认

- 40 步/集是否作为 T5 的正式参数？当前按「A* 0.27 m/步、40 步够穿场」处理；
  若用户坚持 80 步，采集时间翻倍，但口径不变。
- A* 路径目标 `最右柱 x+2.0` 是否要改成可配置？当前硬编码 +2.0，验收线 +1.0。
- `transition_lag` 50% 阈值：T2 数据出来后再定（见 HANDOFF §九）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `scene_gen --self-check --check-seeds 100` | PASS，车道缝 1.20 m，8–9 根 |
| `python3 sim/tools/teacher.py` | PASS（含 A* 绕门柱、脱困回归） |
| `python3 sim/tools/test_spawn.py` | PASS（20 seeds × 3 modes，safe 全部 plan_action=SAFE） |
| `test_split_manifest.py` | 9 项 OK（v3 指纹 291/759） |
| `test_audit_dataset.py` | PASS |
| `test_collect.py` / `test_collect_parallel.py` | PASS |
| T1 单集 scene 48（A*） | 无碰撞、max_x≥6.0 |
| T1 20 test 场景 × 40 步 | **20/20 成功，0 碰撞，100%**；20 个唯一 gid；1641 s / 4 实例 |
| `collection_manifest.json` | `episodes_ok=20 / episodes_failed=0` |

## 风险与遗留

- A* 的路径目标/容差是手调参数；换几何或换 `hold` 要重测。
- `plan_action_global` 每步重跑 A*；T5 500+ 集时 Python 开销需实测（预计 <1 min/集）。
- 40 步/集的类别分布与 80 步不同（空尾更少），T2/T3 的 `class_share` 需重新验。
- 起飞 timeout 120s 只解决慢爬升；若 RTF 再低，仍可能失败，需按负载降实例数。
