# 第十阶段修复：SR 横向边界、可视化三条、DAgger 8 集、保护罩外形

- 时间：2026-11-10 15:35
- 范围：`uav_rgb` sim/tools 与 sim/worlds；`exp-dagger-gz-20261109/full-1109`
- 结论：完成。四项用户拍板的改动全部落地，Python 检查全过；DAgger 已续跑并越过 round_3。

## 做了什么

### 1. SR 加横向边界（用户批准）

- `sim/tools/eval_closed_loop.py`：新增 `--corridor-half-width`（默认 None，旧口径逐位不变）。
  给了值就要求 oracle `pose_gt` 全程 `|y| <= 值`，否则 `success=False`；
  per_episode 新增 `y_max_m` / `corridor_violation`，aggregate 新增 `corridor_violations`，
  eval.json 顶层记录 `corridor_half_width`，`metric_version` 追加 `.corridor<N>m`。
- 新增 `report_metric_version(args)`，让 eval.json 顶层与 per_episode 用同一套后缀规则
  （原来顶层自己拼字符串，加了 corridor 后两处会不一致，测试先红后绿）。
- 不影响 waypoint 任务（l1/campus）：那里的 waypoint 本身在柱阵外侧。

### 2. DAgger round 3 排除 8 集并续跑（用户批准）

- 8 集（7137–7141、7143、7144、7199）移到 `round_3/collect.stalled-20261110/`，
  同目录 `EVIDENCE.json` 记录每集被拉长的步与窗口；旧 FAIL 的 `audit.json` 一并移到那里。
- 重跑审计：`action_window` cv 23.78% → 0.6%（PASS）；只剩 `class_share` / `safe_collision`，
  两者都在 `CFG.audit_ignore` 里，按既有规则放行。
- 续跑 `full-1109`：round_3 训练完成（686 s），进入 more_demos 采集，之后是 eval20 / baselines / eval60 / summary。

### 3. 可视化三条（用户批准）

- `sim/tools/record_views.py`
  - `_yaws` 把 `yaw_before`（度）转弧度再交给 `draw_top_icon`；实测朝向误差从 7°–139° 降到 0°。
  - 速度改用 oracle 的 `speed_mps`，没有时用所在轨迹段差分（`_speed_at`）。
    实测零速帧从 28–30% 降到 1/272 与 1/428。
  - TOP 图标与面板 pos 用 `_pos_at` 在相邻轨迹点间插值，不再落后一步（原中位 0.39 s、学生最大 2.08 s）。
- `sim/tools/test_record_views.py`：更新 yaw 断言（度->弧度）、新增速度与插值两项自检。

### 4. 保护罩按官方 STL 改外形（用户批准）

- `sim/tools/e1_body.py`：四根方条换成 4 个 r=42 mm 外弧圆柱 + 4 个 r=16 mm 凹口圆角圆柱，
  安装高度从 z=15 mm 改为 26 mm（collision_envelope.z_max 之下 4 mm），新增 `GUARD_Z_M` 与
  `GUARD_OUTLINE`（72 点四叶草外沿，供顶视图标用）。`WIDTH_M` 改为 0.168（原 0.172）。
- `sim/worlds/uav_rgb_e1.sdf`：同步 8 个 visual，注释写清 STL 实测与安装高度。
  **模板 sha 从 21c8bc9a 变为 2e554bcd**，旧模板与 0417/DAgger 数据仍按旧 sha 存档，不混包。
- `sim/worlds/templates/uav_rgb_e1.2e554bcd.sdf`：新归档副本（test_e1_template 要求）。
- 碰撞体不改（单圆柱 r=0.102，凹口方向偏保守最多 44 mm）；在 SDF 注释里写明原因：
  改它要换 dynamics_profile，已有数据不能混。

## 没做什么

- 没改碰撞体（见上）。
- 没改 `altitude_assumption` 字段名（会换 profile，需要单独一次口径切换）。原因：超出本轮四项。
- 没改 SITL 的 RC 杆量偏移开关。原因：同上。
- 没跑 C++/CUDA 构建、cf_sitl 自检、Gazebo 实跑。原因：本轮 Python 侧改动不触及 C++；
  DAgger 正在占用 gz 配额。
- 没重跑 eval4 全臂。原因：需要用户决定是否按新口径重算历史表。
- 没提交、没推送。原因：仓库有并发会话，等用户确认。

## 待确认

- `--corridor-half-width` 的取值：建议 3.0（`scene_gen.Y_RANGE`）。M3 的 `e1_pipeline` /
  `dagger_gz_pipeline` 是否默认带上，需要改它们的配置区。
- 保护罩换模板后，新采集的数据与旧数据（21c8bc9a）不能混包；是否重跑 demo。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `sim/tools/test_eval_sim09v2.py` | 11 项 OK（新增 3 项 corridor：默认关/绕行判失败/直穿仍成功） |
| `sim/tools/test_record_views.py` | PASS（新增速度、插值两项） |
| `sim/tools/test_e1_body.py` | PASS（23 部件、外沿 101.4/58 mm、chase 视角 86 px） |
| `sim/tools/test_e1_template.py` | PASS（模板 sha 2e554bcd） |
| `bash tools/check.sh --python-only` | PASS（有 SKIP），52 项 unittest OK |
| demo 零速帧复算 | 教师 77/272 → 1/272；学生 130/429 → 1/428 |
| TOP 图标朝向 | -24.9°/0°/90°/-90°/180° 五个样例误差 0.0–1.7° |
| DAgger round_3 审计 | action_window cv 23.78% → 0.6%，仅剩 audit_ignore 两项 |

## 风险与遗留

- 模板 sha 变了：任何按 `template_sha256` 做一致性校验的流水线会把新采集与旧数据判成两代。
- `WIDTH_M` 0.172 → 0.168，3DGS FOLLOW 的机体像素宽与 demo 字幕里的 0.17 m 会变。
- DAgger 续跑仍在进行（more_demos 采集 → eval20 → eval60），出数前不要动 `exp-dagger-gz-20261109/`。
