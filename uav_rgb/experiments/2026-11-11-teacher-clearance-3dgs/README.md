# 3DGS 教师前向净空改扫掠与出生两层判空：验收实验（2026-11-11，任务卡 W1-12）

不训练，不起 Gazebo。W1-11 的新教师（航向容差 18.9°）在室内 120 集里 19 集没到点，全部撞在前：3 集出生即撞，15 集撞在转向与前进互相切换的那一步，
1 集撞在连续前进时，超时 0。教师的前向检查是沿机头一条 0.4 m 的零宽线段（`clearance="ray"`），看不到转向后首个前进步的位移偏向（偏机头中位 25.5°）
和前进后转弯步的前滑（中位 0.066 m）。本卡做三件事：

1. `OccAstarTeacher(clearance="swept")`：机头线段通过之后，再沿本步前进的预测位移延长 `reserve`，对硬栅格做一次 `occ.sweep`。被否决时 `|err|` 不小于容差一半就转
   （`turn_swept`），否则出 SAFE_NEUTRAL 等一步（`wait_swept`），连续最多等 2 步。教师版本 `uavrgb.teacher_occ.v3`。缺省 `clearance="ray"` 与改动前逐步相同。
2. 出生两层判空（`collect_3dgs --spawn-two-layer`，缺省关）：抖动出的候选点在出生高度与巡航高度两层都不碰撞，且 `occ.clearance` 不低于机体半径加 0.05 m。
3. 导引指针（`guard.OracleRouteGuide(nearest_in_lookahead=...)`，缺省关）：本实验不用，只做 K1 两个停滞集的离线重放（`replay_guide.py`）。

本目录是验收：同路线同 seed 重跑教师，臂 A 与 W1-11 的新教师臂逐集配对。预注册全文在 `run.sh` 文件头（任务卡原文，不改）。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 守护进程入口：预注册、配置区、3DGS 池的锁 `gs3d.lock`、回归 → 臂 A → 臂 B → 分析；`run.pid` / `run.log` / `DONE` |
| `measure_disp.py` | 第 1 部分第 1 步「量」：每步实际位移（机体系）的两个预测器 P1（按上一步类的上下文表）与 P2（动作前速度的线性函数），检验半上比误差，给 reserve |
| `counterfactual.py` | 第 1 部分第 2 步「判」：在 W1-11 新教师臂的 240 集上离线重算扫掠否决，报命中数与误拒率，对任务卡的门槛 |
| `replay_guide.py` | 第 3 部分：K1 两个「目标点停滞」集的真实位姿序列，重放 `OracleRouteGuide.target`，核对重放与记录一致，再用 `nearest_in_lookahead=True` 重放同一序列 |
| `analyze.py` | `acceptance`：臂 A 对 W1-11 逐集配对（精确 McNemar）、预注册判决（`Fraction` 比边界）、臂 B 只报告、步分支占比、教师标签分布、撞在前的集逐集归类、超时、分场景与路线长度分桶；`--replay` 逐集回放教师；`regress`：两个臂目录逐步逐字段精确比对 |
| `test_analyze.py` | `analyze.py` 的单测（41 项，标准库，合成数据与桩） |
| `test_driver_clearance.py` | bearing-guard `driver.py` 两个新键的单测（15 项，桩掉场景与仿真） |

驱动与汇总共用 `../2026-11-10-bearing-guard-3dgs/driver.py`。本卡对它只加了两个臂上的可选键（`teacher_clearance`、`spawn_two_layer`），缺省时行为与原版逐位相同。

## 设计

| 项 | 室外 | 室内 |
| --- | --- | --- |
| 场景 | garden、wanderland_1MlUucug、bicycle、callinstrasse | sage3d_841150、840865、841012、840139、839992 |
| 路线 / 集数 | 路线 0–5 × 每路线 5 集 = 120 | 每场景 8 条（共 40）× 每路线 3 集 = 120；步数封顶 100 |
| seed / 出生扰动 | 7000 + 100·路线 + k，±0.5 m | 8000 + 100·路线下标 + k，±0.2 m |
| 对照 | W1-11 新教师臂 `teacher-tol`：119/120 | W1-11 新教师臂 `teacher-tol`：101/120 |

路线、集数、seed、出生扰动、步数预算逐项从 W1-11 的 `config.json` 复制，同路线同 seed 才能逐集配对。W1-11 新教师臂是旧输出目录的只读符号链接。
hold 1.0、容差 `planner.yaw_tol_deg(1.0)` = 18.9076°（`run.sh` 核对它等于 W1-11 config 里的 `teacher_tol_deg`）、`stop_on_collision`、cf_sitl lib31。

| 臂 | 配置 | 用途 |
| --- | --- | --- |
| A `swept` | `teacher_clearance=swept`，出生开关关 | 主臂，对 W1-11 逐集配对，预注册判决只看它 |
| B `swept-spawn` | `teacher_clearance=swept`，`spawn_two_layer=true` | 只报告；出生开关会让被拒的候选点重抽，起点不同的集不是同起点配对 |
| 回归 `ray` | 显式 `teacher_clearance=ray` | 室内 `sage3d_841150` 8 条路线 × 3 集 = 24 集、室外 `garden` 6 条路线 × 5 集 = 30 集，与 W1-11 新教师臂的 `actions.jsonl` 逐步逐字段比对（浮点按 `==`） |

## 第 1 部分：先量，再离线反事实（11-11 夜，输出在草稿目录，数字可由 `measure_disp.py` / `counterfactual.py` 复算）

数据：W1-11 新教师臂 240 集 9174 步；路线下标偶数 120 集拟合，奇数 120 集检验。

| 项 | 数值 |
| --- | --- |
| 检验半前进步位移预测误差 p90（P1 / P2） | 0.080 m / 0.041 m；相差不小于 0.01 m，按任务卡取 P2（速度外推） |
| 前进后转弯步位移模（1187 步） | 中位 0.066 m，p90 0.132 m |
| reserve | 0.132 + 0.041 = 0.1731 m |
| 反事实命中（16 个非出生点的室内撞在前集里，碰撞步及之前 2 步内有前进步被否决） | 13/16（门槛：低于 10 停，预期不少于 12）；P1 交叉拟合旁证 16/16 |
| 反事实误拒（全程无碰撞的集里被否决的前进步占比） | 0.11%（7/6391），室内 0.33%，室外 0；P1 交叉拟合旁证 0.25%（门槛：高于 10% 停，预期不高于 5%） |

门槛通过，才改的代码。P2 的 6 个系数（`SWEPT_PRED_A`、`SWEPT_PRED_B`）用全部 240 集重新拟合后写进 `planner.py`。

## 怎么跑

```bash
cd experiments/2026-11-11-teacher-clearance-3dgs
# 正式：守护进程。要求 sim/、bearing-guard 驱动目录、本目录都已提交；阻塞式拿 3DGS 池的锁（前面可能排着别的卡的 3DGS 实验）；拿到锁后 GPU 已用显存 <= 19000 MiB
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 冒烟（单车道、3 分钟左右、不拿锁；产出不是结果）
SMOKE=1 OUT=<草稿目录>/smoke ONLY_SCENES="sage3d_841150" ROUTES_LIMIT=1 K_LIMIT=1 ALLOW_DIRTY=1 LANES=1 setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度 / 停（只按 PGID）
tail -f $OUT/run.log; grep -v '^step ' $OUT/driver-indoor-swept-sage3d_841150.log | tail
kill -- -$(cat $OUT/run.pid)
# 只重跑分析（随时可跑，离线；--replay 加载场景占据，要 3dgs-venv 的 python，不占 GPU；--occ-cache 缓存占据栅格）
<3dgs-venv python> analyze.py acceptance $OUT --replay [--occ-cache <目录>]
# 单测（标准库）
PYTHONDONTWRITEBYTECODE=1 <3dgs-venv python> test_analyze.py
PYTHONDONTWRITEBYTECODE=1 <3dgs-venv python> test_driver_clearance.py
```

续跑：同一条命令重起，已完成的（臂、场景、路线）跳过，半成品改名为 `*.incomplete-<时刻>`。

## 产物（`$OUT` = `$D/exp-teacher-clearance-3dgs-20261111`）

- `outdoor/`、`indoor/`：各自的 `config.json`、臂 A `swept/<场景>/r<N>/…`、臂 B `swept-spawn/<场景>/r<N>/…`、W1-11 新教师臂的符号链接 `teacher-tol`。
  每集 `ep_*/` 有 `{metrics.json, actions.jsonl, oracle.jsonl, trace.jsonl, episode_manifest.json, scene.json, rgb_00000.png}`；`trace.jsonl` 保留，回放要用它的速度。
- `regress-outdoor/`、`regress-indoor/`：回归臂 `ray/…`。
- `analysis/`：`acceptance.{md,json}`（`json` 里 `verdict_a` 是臂 A 的预注册判决）、`regress.log`、`regress-*.json`。
- 顶层：`run.pid`、`run.log`、`DONE`（`rc=… 时刻 回归=… 臂A判决=… cgroup起点=[…] 终点=[…]`）、`inputs.txt`、`gpu.txt`、`gpu_peak_mib.txt`、`run.sh.snapshot`、`driver-<suite>-<臂>-<场景>.log`。

## 新增的记录字段

- `episode_manifest.json` 的 `teacher_clearance`（`ray` / `swept`）：占据 A* 教师带 `clearance` 属性时写；forward / 深度教师 / 旧产物没有。
  `teacher_version`：`clearance="swept"` 是 `uavrgb.teacher_occ.v3`；`ray` 仍按容差记 v1 / v2。
- `episode_manifest.json` 的 `spawn_two_layer`（布尔，总写）：出生判空是否用两层检查。旧 manifest 没有这个键，等于 `false`。
- `driver.py` 的 `done.json`：臂上给了 `teacher_clearance` / `spawn_two_layer` 才在末尾多记同名键，旧臂的键不变。
- 分析不依赖这些字段回放：回放按 `config.json` 里臂的角色（臂 A / B 用 swept，W1-11 臂用 ray）构造教师。

## 已知限制

- `clearance="swept"` 的位移系数只在 hold 1.0 下量过，别的 hold 构造时报错（`OccAstarTeacher`、`driver.py`、`collect_3dgs` 三处都校验）。
- 位移系数用 W1-11 的 240 集拟合，样本内；离线反事实用交叉拟合（主口径）与全量拟合两个口径都过了门槛。
- 回放把每集第 0 步的速度取 0（没有上一步的 trace）。实验里第 0 步的速度来自仿真状态，是否逐位为 0 未核实；第 0 步的预测因此用 `b`（常数项）。
  回放核对（复算动作与记录的 `action_expert` 不一致的步数）会把这种差异报出来。
- 出生巡航高度取 `spawn.z + 0.05 m`（`SPAWN_CRUISE_DZ_M`）是一个选择，不是测得的唯一值，待主会话确认。
- `nearest_in_lookahead=True` 在窗口内取最近顶点，薄墙两侧的路径若在前视距离内相邻，指针可能跳到墙的另一侧；本卡只在 K1 的两个停滞集上做了离线重放，没有闭环验证。
- 合并：K3 的分支 `p12-k3`（提交 `5058ae4`）在同一个 `driver.py` 的教师构造处也传了 `hold_s=hold`，与本卡的 `**teacher_kw`（内含 `hold_s`）重复，会报 `TypeError: got multiple values for keyword argument 'hold_s'`。
  合并时把 `teacher_kw` 里的 `hold_s` 去掉（只留 `clearance`）。`git merge-tree --write-tree p12-w1-12 p12-k3`（只读试合并，没动任何工作树）在 `driver.py` 里有 4 处冲突：文件头说明、臂校验循环、`teacher_kw` / `spawn_kw` / `rule_kw` 的定义、教师构造；
  前三处两边的内容都保留，第四处按上面的办法合。`run_episode(**spawn_kw)` 与 `done.json` 追加键两处自动合并。其余文件没有冲突。

## 结果（2026-11-12 00:41:35 起、01:02:41 止，守护进程 PGID 379033，HEAD `07ddf18`，3 条车道，拿 3DGS 池的锁时无人排在前面）

`$OUT/DONE`：`rc=0 2026-11-12 01:02:41 回归=逐步一致 臂A判决=介于预测与证伪之间`。数字来源 `$OUT/analysis/{acceptance.md,acceptance.json,regress.log}`；完整表与归因见 `docs/experiments/2026-11-12-teacher-clearance-3dgs.md`。

| 项（臂 A swept / W1-11 新教师 / 臂 B swept-spawn） | 室外 | 室内 |
| --- | --- | --- |
| 到点成功 k/n | 119/120 / 119/120 / 119/120 | 112/120 = 0.933 / 101/120 = 0.842 / 114/120 = 0.950 |
| 撞在前 | 1 / 1 / 1（都是同一个出生即撞的集） | 8 / 19 / 6 |
| 超时 | 0 / 0 / 0 | 0 / 0 / 0 |
| 每集平均步数 | 44.7 / 44.7 / 44.7 | 34.6 / 31.8 / 35.1 |
| 逐集配对 b / c（对 W1-11） | 0 / 0（臂 A）、0 / 0（臂 B） | 11 / 0，p = 0.000977（臂 A）；13 / 0，p = 0.000244（臂 B） |

预注册（臂 A）：室内到点 >= 0.90 满足（0.933）；室内撞在前 <= 0.06 **未满足**（8/120 = 0.067，差 1 集；<= 7 集才满足）；室内超时 <= 0.03 满足（0）；室外到点 >= 0.97 满足（0.992）；
室内配对 b − c >= 8 满足（+11）。证伪条件（室内到点 < 0.87 或超时 > 0.05）都没命中。机械判决：介于预测与证伪之间。
回归臂 `ray`：室外 garden 30/30 集、室内 sage3d_841150 24/24 集与 W1-11 的 `actions.jsonl` 逐步逐字段相同。
回放核对：臂 A、臂 B、W1-11 臂的回放动作与记录的 `action_expert` 不一致 0 步（室外 5363 / 5364 / 5363 步，室内 4148 / 4217 / 3811 步）。
分支占比（室内臂 A，全部步 / 导航阶段）：`wait_swept` 0.7%（29/4148）/ 0.8%，`turn_swept` 0.1%（6/4148）/ 0.1%；室外都是 0。SAFE_NEUTRAL 标签占比：室内 9.7%（401/4148）对 W1-11 的 8.8%（336/3811），
导航阶段 0.8% 对 0.0%；其余悬停是到点后的 stop 步。仍然撞在前的 8 集（臂 A，室内）：出生即撞 3、转向步 3、悬停步 1、前进步且 swept 检查通过 1（逐集表见实验文档）。
