# 3DGS 教师航向容差随转角步推导：验收实验（2026-11-10，任务卡 W1-11）

不训练，不起 Gazebo。`sim/3dgs/planner.py` 的 `OccAstarTeacher` 在 hold 1.0 下容差 12°（死区宽 24°）小于一个转角步（实测 26.3–34.7°），
误差落在 (12°, 步长 − 12°) 时转一步过冲到另一侧，再转回来，左右来回转。改动：`yaw_tol_deg(hold) = max(12°, 0.55 × 名义转角步)`，
名义转角步 = `YAW_RATE_RPS`（0.6 rad/s，抄自 `sim/dynamics/e1_mini.json`）× hold；hold 0.5 仍是 12°（与旧教师逐步相同），hold 1.0 为 18.91°。
本目录是验收：同路线同 seed 重跑教师，与旧教师逐集配对。预注册（三个主比较 M1–M3）全文在 `run.sh` 文件头。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 守护进程入口：阶段 0 确定性回归 → 阶段 1 室外 → 阶段 2 室内 → 阶段 3 分析；预注册、配置区、flock / `run.pid` / `run.log` / `DONE` |
| `measure_turn_step.py` | 「先量再定」：旧实验每集 `actions.jsonl` 里每个转向步的转角（按上一步动作分直飞后 / 悬停后 / 同向连转 / 反向），与容差的关系 |
| `analyze.py` | `acceptance`：新旧教师逐集配对（精确 McNemar）、超时、含交替段的超时、碰撞、SPL、室内按路线长度分桶、验收判决、`--replay` 回放归因；`regress`：两个臂目录逐步逐字段精确比对 |
| `test_analyze.py` | 两个脚本的单测（24 项，标准库，合成数据） |

驱动与汇总共用 `../2026-11-10-bearing-guard-3dgs/{driver.py,summarize.py}`（只读使用，没有改）：新容差通过 `config.json` 已有的 `teacher_tol_deg` 键传入，
数值由 `planner.yaw_tol_deg(HOLD)` 算出，不手写。

## 设计

| 项 | 室外 | 室内 |
| --- | --- | --- |
| 场景 | garden、wanderland_1MlUucug、bicycle、callinstrasse | sage3d_841150、840865、841012、840139、839992 |
| 路线 / 集数 | 路线 0–5 × 每路线 5 集 = 120 | 每场景 8 条（共 40）× 每路线 3 集 = 120；步数封顶 100 |
| seed / 出生扰动 | 7000 + 100·路线 + k，±0.5 m | 8000 + 100·路线下标 + k，±0.2 m |
| 旧教师臂（容差 12°） | `exp-bearing-guard-3dgs-20261110/teacher`：100/120 = 0.83 | `exp-sage3d-pilot-20261110/teacher`：76/120 = 0.63 |
| 验收阈值（任务卡） | 新教师到点成功 >= 0.92 | >= 0.80 |

路线、集数、seed、出生扰动、步数预算逐项从旧实验的 `config.json` 复制（`run.sh` 的配置区不重设它们），同路线同 seed 才能逐集配对；
旧教师臂是旧输出目录的只读符号链接，本实验只写新臂 `teacher-tol`。hold 1.0、`stop_on_collision`、cf_sitl lib31。

## 结果（2026-11-10 22:38:09 起、22:59:32 止，守护进程 PGID 1839205，HEAD `901d039`）

`$OUT/DONE`：`rc=0 2026-11-10 22:59:32 回归=逐步一致 outdoor=达标(119/120) indoor=达标(101/120)`。数字来源 `$OUT/analysis/{acceptance.md,acceptance.json,turn_step.md,regress.log}`。

| 项（新教师 / 旧教师） | 室外 | 室内 |
| --- | --- | --- |
| 到点成功 k/n [Wilson 95%] | 119/120 = 0.992 [0.954, 0.999] / 100/120 = 0.833 [0.757, 0.889] | 101/120 = 0.842 [0.766, 0.896] / 76/120 = 0.633 [0.544, 0.714] |
| 超时 / 其中含交替段 | 0 / 0 对 19 / 19 | 0 / 0 对 29 / 25 |
| 含交替段的集（全部） | 0 对 49 | 1 对 49 |
| 碰撞集（任一步 / 撞在前） | 1 / 1 对 1 / 1 | 21 / 19 对 18 / 15 |
| SPL(stop)、平均步数 | 0.952、44.7 对 0.733、58.5 | 0.792、31.8 对 0.518、45.1 |
| 逐集配对 b / c / Δ / 精确 McNemar p | 19 / 0 / +0.158 / 3.81e-06 | 28 / 3 / +0.208 / 4.65e-06 |
| 预注册 | M1（>= 0.92，b > c 且 p <= 0.05）达标；M3（超时 <= 4）达标 | M2（>= 0.80，b > c 且 p <= 0.05）达标；M3（超时 <= 7）达标 |

室内按 A* 路线长度分桶（到点 / 请求集数，超时，碰撞集），新教师对旧教师：< 5 m 41/42、0、3 对 38/42、2、5；5–10 m 42/51、0、9 对 28/51、16、7；>= 10 m 18/27、0、9 对 10/27、11、6。
确定性回归（新代码 + 显式 `tol_deg=12.0` 重跑旧教师）：室外 garden 30/30 集、室内 sage3d_841150 24/24 集与旧 `actions.jsonl` 逐步逐字段相同。
室内没到点的 19 集全是撞在前（超时 0）：3 集在步 0（新旧臂是同 3 集），15 集在转向与前进的切换步，与航向容差无关（18.0° 诊断臂与本臂同为 101/120，逐集结局有 8 集互换）。仅剩的 1 个含交替段的集（`sage3d_839992` r93 ep1，7 步，最终到点）不是容差极限环：路径逐步不变，瞄准点随毫米级位移在两点间翻。

GPU：验收跑在 `GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70`（2026-11-10），该卡 11-11 起不在容器里（现为 `GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`），`run.sh` 配置区的 `GPU=` 是冻结的实验记录没改，复现时把那一行换成当前卡的 UUID。

## 怎么跑

```bash
cd experiments/2026-11-10-teacher-tol-3dgs
# 正式：守护进程（flock 互斥；写 run.pid / run.log / DONE）。要求 sim/ 与本目录已提交（inputs.txt 才对得上），GPU 已用显存 <= 19000 MiB
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 冒烟：2 个场景 × 每场景 1 条路线 × 1 集（回归也只跑这 2 个场景）
OUT=<草稿目录>/smoke ONLY_SCENES="garden sage3d_841150" ROUTES_LIMIT=1 K_LIMIT=1 ALLOW_DIRTY=1 setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度 / 停（只按 PGID）
tail -f $OUT/run.log; grep -v '^step ' $OUT/driver-outdoor-garden.log | tail
kill -- -$(cat $OUT/run.pid)
# 只重跑分析（随时可跑；--replay 加载场景占据，要 3dgs-venv 的 python，不占 GPU）
<3dgs-venv python> analyze.py acceptance $OUT --replay
# 单测（标准库）
PYTHONDONTWRITEBYTECODE=1 python test_analyze.py
```

续跑：同一条命令重起，已完成的路线跳过，半成品改名为 `*.incomplete-<时刻>`。

## 产物（`$OUT` = `$D/exp-teacher-tol-3dgs-20261110`）

- `outdoor/`、`indoor/`：各自的 `config.json`、新臂 `teacher-tol/<场景>/r<N>/{done.json, ep_*/{metrics.json, actions.jsonl, oracle.jsonl, episode_manifest.json, rgb_00000.png}}`、
  旧臂符号链接 `teacher`、`summary.md`（bearing-guard 的汇总，只在完整跑时生成）。
- `regress-outdoor/`、`regress-indoor/`：阶段 0 的重跑（臂目录也叫 `teacher-tol`，但容差是显式 12°）。
- `analysis/`：`acceptance.{md,json}`、`turn_step.{md,json}`、`regress.log`、`regress-*.json`。
- 顶层：`run.pid`、`run.log`、`DONE`（`rc=… 时刻 回归=… 判决 …`）、`inputs.txt`、`gpu.txt`、`run.sh.snapshot`、`driver-<阶段>-<场景>.log`。

## 新增的记录字段（`sim/3dgs/collect_3dgs.py`）

- `episode_manifest.json` 的 `teacher_yaw_tol_deg`：占据 A* 教师的航向容差（度）；只有带容差的策略（教师、DAgger 的专家）写，forward / 学生 / 深度教师不写。
  旧集没有这个键，等于 12°。
- `teacher_version`：容差 = 12° 仍是 `uavrgb.teacher_occ.v1`；不是 12°（按 hold 推导或显式给的）是 `uavrgb.teacher_occ.v2`。

## 已知限制

- `driver.py` 构造教师时不传 `hold_s`（它是别的分支在改的文件）；本实验用 `teacher_tol_deg` 键绕过。合并后在 `driver.py` 加一行 `hold_s=CFG["hold"]`，
  不给 `teacher_tol_deg` 键时才会按 hold 推导。
- `--replay` 的回放用 `episode_manifest.json` 记录的容差；旧代码产出的集没有这个键，回放时用 config 里该臂的值（新臂 `teacher_tol_deg`，旧臂 12°）。
- 3DGS 的占据 A* 标签没有离线重标工具（Gazebo 的 `tools/relabel_costmap.py` 只处理 Gazebo 数据），所以「旧数据的标签重算按记录值」目前只有记录约定，没有重算代码。
  分派规则：`tol = manifest.get("teacher_yaw_tol_deg", 12.0)`，再 `OccAstarTeacher(planner, waypoint, tol_deg=tol)`。回放核对（草稿 `replay_all.py`，不进仓库）：按记录值回放，新臂 120 集 3811 步、旧臂 120 集 5414 步，动作与记录的 `action_expert` 不一致 0 步；错配（新臂强行 12°、旧臂强行 18.9076°，各取每条路线第 0 集）分别有 271/1285 步、715/1849 步的标签不同，所以不能不分派。
