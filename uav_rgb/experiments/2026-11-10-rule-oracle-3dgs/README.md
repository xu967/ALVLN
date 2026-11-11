# 2026-11-10 3DGS 上的规则层上限（rule-oracle-3dgs，任务卡 K1）

计划 `docs/briefs/2026-11-10-phase12-task-cards.md` 的卡 K1。零训练：每步 20 列（3°，60° 视场，上限 4.0 m）的可通行距离，来自占据真值（`occ`）或渲染深度（`depth`），
喂给没有改动的 `sim/tools/rule_policy.rule_v1_step`，再加路线方位的航向计数 `n = round(−err / yaw_step_deg)`（err 为正 = 目标在左，目标在左 n < 0）。
不起 Gazebo，不训练，不含教师臂。

## 问什么

盲飞加路线护栏（`forward+route`，11-10 两个实验）室内 0.60、10 m 以上 0.30、碰撞率 0.43，室外 0.89。感知完美（距离取真值）时，「列距离 + 固定规则 v1 + 路线方向」这套分解能到多高。

预注册（`run.sh` 文件头，任务卡原文，起跑前定）：主臂 `rule-occ+route` 室内到点成功不低于 0.80，10 m 以上路线不低于 0.60，碰撞率不高于 0.15；室外不低于 0.84，碰撞率不高于 0.10。
证伪：主臂室内低于 0.70。主比较三个（室内，逐集配对精确 McNemar）：`rule-occ+route` 对 `forward+route`，`rule-depth+route` 对 `rule-occ+route`，`rule-occ+goal` 对 `rule-occ+route`。

## 设计

- 策略 `sim/3dgs/rule_oracle.py`（测试 `sim/3dgs/test_rule_oracle.py`）：两路列距离每步都算并落日志，决策只用主来源。`occ` = 相机位置沿 20 列 × 5 射线在碰撞同一个占据栅格里步进；`depth` = 渲染 z 深度反投影到机体系，留机体包络上下各 0.10 m 内的点按方位分列取最近。
- 臂（`driver.py` 新增 base `rule-occ` / `rule-depth`、guard `route` / `goal`、臂级 `hold`、`step_mult`、`rule_params`、全局 `drop_trace`；不给新键时旧臂逐位不变）：`forward+route`（参照，重跑用于配对）、`rule-occ+route`（主臂）、`rule-depth+route`、`rule-occ+goal`；仅室内再加 `rule-occ+route@0.5` 与 `forward+route@0.5`（hold 0.5 s，步数预算 ×2）。
- 室内 = sage3d-pilot 的 5 个场景、同 40 条路线 × 3 集 = 120 集，seed = 8000 + 100·路线 + k，出生扰动 ±0.2 m / ±10°，路线前视 1.5 m；室外 = garden、wanderland_1MlUucug、bicycle、callinstrasse 各 6 条路线 × 5 集 = 120 集，seed = 7000 + 100·路线 + k，扰动 ±0.5 m，前视 3.0 m。hold 1.0 s，护栏阈值 45°。
- 参数冻结：`r_safe` 0.25、`d_look` 1.2 在 dev 阶段选定（室内 5 场景里没进试点 40 条的合格路线每场景 4 条，共 20 集每组，6 组候选，14/20 = 0.70 最高，次高三组 13/20；并列规则见 `run.sh` 头）；`yaw_step_deg` hold 1.0 = 31.7°（22,988 个转向步的中位数 31.68°），hold 0.5 = 17.13°（590 步）；其余规则参数不动；室外用同一组。
- 口径：SR(stop) = 首次进入 waypoint 半径早于首次碰撞；碰撞率 = 整集任一步碰撞（含到点之后，规则臂到点后不停，与旧实验同口径）；撞在前 = 没到点就撞；超时 = 既没到点也没撞。

## 结果（正式，2026-11-11 19:42:58 起跑，20:05:20 结束，rc=0；表来自 `RESULT.md`）

室内（120 集每臂）：

| 臂 | SR(stop) [Wilson 95%] | 碰撞率 | 撞在前 | 超时率 | <5 m | 5–10 m | >=10 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `forward+route` | 72/120 = 0.60 [0.51, 0.68] | 0.43 | 0.37 | 0.03 | 36/42 | 28/51 | 8/27 = 0.30 |
| `rule-occ+route`（主臂） | 80/120 = 0.67 [0.58, 0.74] | 0.03 | 0.03 | 0.30 | 34/42 | 29/51 | 17/27 = 0.63 |
| `rule-depth+route` | 84/120 = 0.70 [0.61, 0.77] | 0.04 | 0.03 | 0.28 | 34/42 | 35/51 | 15/27 = 0.56 |
| `rule-occ+goal` | 70/120 = 0.58 [0.49, 0.67] | 0.03 | 0.03 | 0.39 | 35/42 | 29/51 | 6/27 = 0.22 |
| `rule-occ+route@0.5` | 65/120 = 0.54 [0.45, 0.63] | 0.03 | 0.03 | 0.43 | 33/42 | 22/51 | 10/27 = 0.37 |
| `forward+route@0.5` | 68/120 = 0.57 [0.48, 0.65] | 0.51 | 0.40 | 0.03 | 36/42 | 27/51 | 5/27 = 0.19 |

室外（120 集每臂）：

| 臂 | SR(stop) [Wilson 95%] | 碰撞率 | 撞在前 | 超时率 |
| --- | --- | --- | --- | --- |
| `forward+route` | 107/120 = 0.89 [0.82, 0.94] | 0.10 | 0.10 | 0.01 |
| `rule-occ+route`（主臂） | 117/120 = 0.97 [0.93, 0.99] | 0.03 | 0.01 | 0.02 |
| `rule-depth+route` | 99/120 = 0.82 [0.75, 0.88] | 0.04 | 0.04 | 0.13 |
| `rule-occ+goal` | 113/120 = 0.94 [0.88, 0.97] | 0.04 | 0.02 | 0.04 |

## 预注册判决（点估计对阈值，同时列 Wilson 95% 区间）

| 项 | 实测 | 阈值 | 判决 | 区间是否跨阈值 |
| --- | --- | --- | --- | --- |
| 室内到点成功 | 80/120 = 0.67 [0.58, 0.74] | ≥ 0.80 | 未达成 | 否 |
| 室内 10 m 以上路线到点成功 | 17/27 = 0.63 [0.44, 0.78] | ≥ 0.60 | 达成 | 是 |
| 室内碰撞率 | 4/120 = 0.03 [0.01, 0.08] | ≤ 0.15 | 达成 | 否 |
| 证伪：室内主臂低于 0.70 | 80/120 = 0.67 [0.58, 0.74] | < 0.70 | **证伪成立** | 是 |
| 室外到点成功 | 117/120 = 0.97 [0.93, 0.99] | ≥ 0.84 | 达成 | 否 |
| 室外碰撞率 | 4/120 = 0.03 [0.01, 0.08] | ≤ 0.10 | 达成 | 否 |

证伪成立是按点估计对 0.70 记的；这条口径是执行者起跑前补写的，任务卡没写。点估计 0.667 与 0.70 差 4 集（84/120 才是 0.70），区间 [0.58, 0.74] 跨过 0.70。预注册写的证伪含义是「列距离加这条规则在室内不够，要么动作粒度太粗（看 hold 0.5 的臂），要么需要更多几何」：
hold 0.5 的臂 `rule-occ+route@0.5` 是 0.54，比 hold 1.0 的 0.67 低（x3：b = 1、c = 16，p = 0.0003），粒度变细没有帮助；「需要更多几何」本实验没有对应的臂，未检验（见下节：失败集的主要形态在规则的决策逻辑里，`occ` 与 `depth` 两个来源室内成绩接近）。

主比较（室内，逐集配对精确 McNemar；b = A 成 B 败，c = A 败 B 成）：

| 编号 | A 对 B | b | c | Δ（SR A − SR B） | p |
| --- | --- | --- | --- | --- | --- |
| M1 | `rule-occ+route` 对 `forward+route` | 19 | 11 | +0.067 | 0.2005 |
| M2 | `rule-depth+route` 对 `rule-occ+route` | 10 | 6 | +0.033 | 0.4545 |
| M3 | `rule-occ+goal` 对 `rule-occ+route` | 5 | 15 | −0.083 | 0.0414 |

三个主比较没有写多重比较校正；M3 的 p 在逐个比较下低于 0.05，按三个比较做 Bonferroni（0.0167）则不过线。室外三个同名比较不在预注册里（`RESULT.md` 标「补充」）：M1 b = 10、c = 0、p = 0.0020；M2 b = 1、c = 19；M3 b = 0、c = 4、p = 0.125。

## 失败形态（只读分析，没有改参数 / 规则；脚本与输出见下）

- dev（`dev_failures.md`、`dev_failure_split.md`）：选定参数的 6 个没到点集里，5 个以 `V1-more` ↔ `V1-cap` 的两步循环收尾（末 20 步严格交替，n 在 ±2 与 ±3 之间摆，位置包围盒 4 个不超过 0.07 m），1 个（841150 r13）是导引目标点不前进：目标点从第 1 步到第 61 步不变，飞机停在它 0.15 m 内，周围空旷（末 20 步 15 步三通道全畅通），n 在 ±5、±6 之间跳。
- 正式室内 `rule-occ+route` 的 40 个失败集（`formal_failures.md`）：撞在前 4（3 个在步 0，出生间隙 0.11–0.13 m，飞机出生时已贴在碰撞包络上；1 个在第 57 步 V1-back 转向时），超时 36（23 个末 20 步是两步循环，1 个目标点停滞，9 个有 ≥ 6 步交替但没以循环收尾，3 个其他）。三通道全堵（clear 全 0）的步占比：失败集 55.4%（1259/2274），到点集到点之前 0.4%（8/2119）。
- 出生即碰撞（collision_step = 0）的集：室内 3 个（hold 1.0 的 4 个臂是同一批，两个 hold 0.5 的臂 2 个），室外 1 个（4 个臂同一个）；室内 hold 1.0 的臂到点上限 117/120。
- 室外 `rule-depth+route` 的 21 个失败集：garden 15（15/30 到点；r0、r4 为 0/5，r1 为 1/5，r2 为 4/5）、wanderland 5、bicycle 1。garden 失败集到 nav_end 前，正前方 6 列里 depth 比 occ 近 ≥ 0.3 m 的占比 0.162，garden 到点集 0.014（两个占比来自一次性读取脚本，输出在草稿目录 `tmp/phase12-cards/k1/depth_vs_occ_front.txt`，脚本没有入库）：depth 读到了占据栅格里没有的东西，是什么未核实。wanderland r4 有两个撞在前集（ep1、ep4）在第 10 步前进时撞上，前一步 depth 正前方 6 列最小 4.0 m（上限），occ 为 1.01 m 与 0.35 m。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 守护进程入口：`PHASE=dev`（开发路线选参 + hold 0.5 转角步量测）与 `formal`；PGID 自成一组、实例锁与机器锁、体积上限 1500 MB、显存采样、续跑、DONE |
| `select_dev_routes.py`、`select_params.py` | 室内开发路线选取；开发路线上按写死的规则选 `r_safe` / `d_look` |
| `yaw_step_stats.py` | 从 `actions.jsonl` 量 YAW 一步的转角（起转 / 同向续转 / 反向转） |
| `depth_geometry_check.py` | gsplat 上的合成场景几何对拍（渲染深度是 z 深度、轴向约定、相机前向偏移、三路列距离） |
| `summarize.py` | 汇总 → `summary.md`：预注册判决、分臂 / 分场景 / 分桶、McNemar、depth 与 occ 一致性、失败分类、规则条目占比、与旧实验同键对拍 |
| `dev_failure_analysis.py`、`formal_failure_analysis.py` | 失败形态分析（dev 与正式） |
| `test_*.py` | `test_driver_rule.py`（16）、`test_select_dev_routes.py`（7）、`test_select_params.py`（9）、`test_summarize.py`（28）、`test_failure_analysis.py`（32）；规则层 `sim/3dgs/test_rule_oracle.py`（32） |
| `RESULT.md` | 正式 `summary.md` 的副本 |
| `dev_failures.md`、`dev_failure_split.md`、`formal_failures.md` | 失败形态分析输出 |

## 运行

```bash
cd /root/claude-code-backup-20261028/code/wt/p12-k1     # 或任何含本目录与 sim/3dgs 的检出；不需要环境变量
setsid nohup bash experiments/2026-11-10-rule-oracle-3dgs/run.sh > /dev/null 2>&1 < /dev/null &
tail -f /root/rivermind-data/uav_rgb/exp-rule-oracle-3dgs-20261110/formal/run.log
kill -- -$(cat /root/rivermind-data/uav_rgb/exp-rule-oracle-3dgs-20261110/formal/run.pid)    # 停：只按进程组
# 冒烟：SMOKE=1 INDOOR_SCENES=sage3d_841150 OUTDOOR_SCENES=garden ROUTES_LIMIT=1 K_INDOOR=1 K_OUTDOOR=1 OUT_ROOT=<草稿目录>/smoke setsid nohup bash .../run.sh ...
# 只重做汇总：$R/.venv-tools/bin/python experiments/2026-11-10-rule-oracle-3dgs/summarize.py <OUT>/formal --old-indoor $D/exp-sage3d-pilot-20261110 --old-outdoor $D/exp-bearing-guard-3dgs-20261110
# 失败形态：python experiments/2026-11-10-rule-oracle-3dgs/formal_failure_analysis.py <OUT>/formal
```

GPU 用环境里唯一的那张卡（`run.sh` 配置区的 `GPU=` 行，2026-11-11 起 `GPU-b544b42e-…`）；formal 起跑时排机器锁 `tmp/phase12-cards/machine.lock`，持有到脚本退出。

输出：`/root/rivermind-data/uav_rgb/exp-rule-oracle-3dgs-20261110/{dev,formal}/`（`formal/indoor`、`formal/outdoor` 下每臂 / 场景 / 路线 / 集一个目录：`metrics.json`、`rule.jsonl`、`actions.jsonl`、`oracle.jsonl`、`episode_manifest.json`、`rgb_00000.png`、`scene.json`，没有 `trace.jsonl`（`drop_trace`）；`summary.md`、`summary.json`、`inputs.txt`、`run.log`、`gpu_peak_mib.txt`、`DONE`），
分析输出另存 `.../exp-rule-oracle-3dgs-20261110/analysis/`。正式目录 455 MB，整个输出根 520 MB。
