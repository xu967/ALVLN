# T1：车道几何放宽 + 教师穿越率

- 时间：2026-10-30
- 环境：GPU 模式（RTX 2080 Ti / sm_75），Gazebo Harmonic 8.15，DISPLAY=:20
- 结论：**几何 v3 落地**（车道缝 0.902 → 1.2 m，`SPLIT_SALT` v3），但只改几何
  不够——局部射线教师在门柱前因 34°/步的偏航过冲摆振，24 个 test 场景只有 9 个
  无碰撞穿过。改用 **A\* 全局路径 + 纯追踪** 后重测（数字见下表）。
  旧的「调 planner 余量」路线仍被证伪；真正缺的是全局路径与 25° 转向容差。

## 1. 几何改动

| 参数 | v2 | v3（本次） | 说明 |
| --- | ---: | ---: | --- |
| `LANE_SEP` | 1.302 | **1.6** | 车道中心距 |
| 横向车道缝 | 0.902 | **1.2** | `LANE_SEP − PILLAR_SIZE` |
| 机体外接直径 | 0.711 | 0.711 | 旋翼中心 + 桨盘 |
| 单侧余量 | 0.0955 | **0.2445** | `(缝 − 机体)/2` |
| `PILLAR_Y_RANGE` | ±1.5 | **±1.7** | 三条车道装得下 |
| `LANE_CENTER_ABS_MAX` | 0.19 | **0.1** | `0.1+1.6 ≤ 1.7` |
| 走廊 `Y_RANGE` | ±3.0 | ±3.0 | 边界净空 ≥1.1 m |
| `PILLAR_COUNT_RANGE` | 8–9 | 8–9 | 3×3 上限不变 |
| `SPLIT_SALT` | `uavrgb.geom.v2` | **`uavrgb.geom.v3`** | 几何语义变了必须换 |
| 出生点 | (0,0,0.15) | **(−0.5,0,0.15)** | 给门柱前多 0.5 m 转向空间 |

- 同车道柱 i / i+3 的 x 间距仍是 1.3125 m（纵向缝 0.9125 m ≥ `MIN_GAP=0.9`）；
  `clearance.min_pair_gap_m` 报的是这个数，**不是**被修的横向车道缝 1.2 m。
- 旧 v2 数据不能与 v3 混用：`scene.json` 写 `geometry_version`，collect 写进
  episode manifest，preprocess/audit 会拒绝跨版本混包。

## 2. 第一次实测：局部教师集体卡死

命令（test split；最终验收用 4 实例 × 5 集 × 40 步）：

```bash
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 5 --steps 40 --hold 1.0 \
  --base-seed 0 --split test --policy teacher --keep-worlds --out /tmp/t1
```

> 步数从 80 降到 40：旧 80 是按「每步 0.105 m」估的；A* 直线段实测
> 0.27 m/步，40 步已走 ~10.8 m，足够从 x=-0.5 穿到 goal 6.0。当前 cgroup
> 只有 12 核且和其它会话共享，80 步/集的墙钟成本翻倍，40 步不影响验收口径。

结果：**8/8 集在 x≈1.07–1.45 m 处返回 SAFE_NEUTRAL 卡死**，只有 1 集走到 x=3.7
后擦柱。根因不是几何，而是 `ray_clear` 从 t=0 开始检查：无人机漂进 0.36 m
安全余量后，当前点本身就在某根柱子的 clearance 圆内，于是**每条射线的第一个
采样点都被判堵**，`best is None` → SAFE → 永远停住。

修复：`ray_clear(..., skip_clearance_m=0.2)` 的前 0.2 m 只查真实碰撞盒
（0.47 m 机身 + 1 cm），之后仍要求 full clearance；`plan_action` 在正常射线
全堵时用这个回退脱困。SAFE 出生集仍用原语义（真被围死才 SAFE）。

## 3. 第二次实测：脱困后 15/24 门柱擦碰

- 24 个 test 场景，穿越判据 `无 collision 且 max_x ≥ 最右柱 x + 1.0`。
- 结果：**9/24 成功（37.5%），15 集在门柱附近 contact**，典型位置
  x≈1.05 m、y≈±0.4 m、step 11–18；碰撞后部分集继续飞到 x=10–19，但已经失败。
- 根因有两个：
  1. `plan_action` 的 `tol=10°` 小于偏航一步的 30–34°：航向差 15° 时它继续
     转，下一步又反向转过头，门柱前左右摆；
  2. `y_gain=25` 在门柱前把已选的净空方向拉回中线，放大了摆振。
- 只把 `tol` 放到 20–25°、`y_gain=0`，仍有 15 集擦柱：局部射线看不到
  「先横移再穿缝」的全局路径，A* 才能。

## 4. 第三次实测：A* 全局路径

- 新增 `teacher.astar_path`（0.1 m 栅格、柱方框 + 0.5 m 膨胀、8 邻接禁斜穿角点）
  与 `plan_action_global`（路径前 0.7 m carrot，`tol=25°`）。
- 普通集（`spawn.mode != safe`）用 A*；safe 出生集仍用局部 `plan_action`
  （保留真 SAFE 标签）。manifest 记录 `planner=astar|plan_action`，
  audit 按这个字段复算。
- 单集 scene 48 实测：**80 步无碰撞，max_x≥6.0，成功**；动作分布
  38 FWD / 29 LEFT / 13 RIGHT。

### 运行可靠性：起飞超时

A* 全量重跑时发现两个环境陷阱，都会让 8 实例批量失败：

1. **并发 CTest/构建**：`test_overfit` 占 GPU/CPU 时，8 实例里 21/24 集起飞
   超时（z 只到 1.4–1.8 m）。采集窗口必须独占。
2. **起飞 timeout=30s 太短**：8 实例软件 GL 下 RTF 可低到 0.2，实测爬升约
   0.15 m/s（指令 0.3 m/s），从 0.15 到 2.1 m 要 ~13 s 仿真 ≈ 60 s 墙钟。
   `collect.take_off` 默认改为 **120s**；它只在真失败时多等，不影响正常集。

### 最终 24 集结果

（本小节由最终 A* 全量 run 填写；命令同上，`--out` 指向 `t1-astar4-full-*`。）

| 项 | 结果 |
| --- | --- |
| 集数 | 20（全部 test split，20 个唯一 geometry_gid） |
| 步数 | 800（40 步/集） |
| 成功（无碰撞且 max_x≥6.0） | **20/20** |
| 穿越率 | **100%** |
| collision | 0 |
| 验收线 | ≥90% → **通过** |
| 墙钟 | 1641 s（4 实例 × 5 集；0.488 步/s，采集窗口内没有并发 GPU/编译任务） |

失败集：无。`collection_manifest.json` 记录 `episodes_ok=20 / episodes_failed=0`；
每个 episode 的 `max_x` 在 6.05–8.4 m（goal 6.0）。

## 4b. 2026-10-30 05:xx 复跑：用当前 teacher 重验证

审核发现上表的 20 集是**中间版本教师**的产物（manifest `teacher_version=null`），
当前 working tree 的 teacher 复算旧数据有 86/800 不一致，版本门禁又会把它降级成
SKIP——「20/20」没有验收当前代码。于是用同 20 个 test seed 复跑：

```bash
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 sim/tools/collect_parallel.py \
  --instances 4 --episodes-per-instance 5 --steps 40 --hold 1.0 \
  --seeds 23,25,36,45,48,49,50,54,57,84,88,99,114,116,129,137,147,171,173,180 \
  --split test --spawn-mode default --policy teacher \
  --out "$DATA_ROOT/uav_rgb/t1-v3e-105435"
```

| 项 | 结果 |
| --- | --- |
| 成功 / 集数 | **20 / 20**（0 碰撞，max_x 全部过线） |
| `traverse_rate --min-rate 0.9` | 100%，test n=20 |
| `audit label_recompute` | **0/800 不一致**（当前 teacher v3，0 跳过） |
| `audit class_share` | FAIL：SAFE 0/800（本批只做穿越验收，四类数据见 T2） |
| 吞吐 | 525 s，1.52 步/s（4 实例） |

数据：`$DATA_ROOT/uav_rgb/t1-v3e-105435`。T1 的验收口径是**穿越率**，不是四类
数据闸门；类别覆盖由 T2 的 52 集负责。

## 5. 改了哪些文件

- `sim/tools/scene_gen.py`：`LANE_SEP=1.6`、`PILLAR_Y_RANGE=±1.7`、
  `LANE_CENTER_ABS_MAX=0.1`、`GEOMETRY_VERSION=v3`、出生位姿
  `default/random/safe`、`SPAWN_X=-0.5`、meta 写 `geometry_version/spawn/lane_gap_m`。
- `sim/tools/split_manifest.py`：`SPLIT_SALT = scene_gen.GEOMETRY_VERSION`（单一来源）。
- `sim/tools/teacher.py`：脱困回退、`Y_GAIN=0`、局部 `tol=15°`、A* + 纯追踪。
- `sim/tools/collect.py`：普通集用 `plan_action_global`，safe 集用局部；
  manifest 写 `geometry_version/planner`。
- `sim/tools/collect_parallel.py`：`--split/--seeds/--spawn-mode/--safe-spawn-fraction`，
  manifest 记 `unique_geometry`。
- `sim/tools/audit_dataset.py`：跳过 `failed/`、`safe_collision`、`geometry_version`、
  per-split 类别计数、`--exclude-collision-episodes`。
- `sim/tools/traverse_rate.py`（新）：T1/T5 的穿越率口径。
- `sim/tools/test_spawn.py`（新）：safe spawn 必须 plan_action=SAFE 且不撞柱。

## 6. 没做

- 没有做 8–12 根柱的任务原始要求：3.5×3.0 m 密布区 + 全对缝 ≥0.9 m 最多 9 根，
  这是 v2 就有的约束，T1 没改。
- 没有重跑 v2 的 8 场景消融（旧数字仍可引用，但不是 v3 同口径）。
- 没有验证 A* 在 500+ 集长采下的稳定性（T5 做）。
- 没有把 A* 接进 `eval_closed_loop.py`：闭环评测用的是学生策略，教师只在采集侧。
