# 3DGS 路线方位护栏 × {盲前进, 学生}（2026-11-10，bearing3dgs）

不训练，只在评测端给策略加一层 oracle 方位护栏，把 11-09 跨域实验里混在一起的两个因素拆开：

- 11-09（`docs/experiments/2026-11-09-xeval-3dgs.md`）：Gazebo 柱阵训的学生在 4 个 3DGS 真实场景、6 条带转弯路线上到点成功率 0.24–0.32，与盲前进 0.31 没有差别，教师 A* 0.83。
- 因素一：学生不知道目标在哪（柱阵是直线任务，观测里没有航向和目标）。因素二：学生没见过真实外观。
- 本实验给策略一个粗方向（路线方位护栏），看 {盲前进, 学生} × {无护栏, 有护栏} 四格的到点成功率。结果决定下一步长程路线任务先补「目标条件」还是先补「真实外观上的感知」。

## 护栏（`sim/3dgs/guard.py`）

- 规则：每步先让内层策略（学生或盲前进）出动作，再按当前真值位姿算「目标方位 − 机头方位」；|误差| ≥ 阈值（45°）就把动作改写成朝目标转（`YAW_LEFT` / `YAW_RIGHT`），否则保留内层动作。
- 目标点：`route` = 占据 A* 全局路径上「离当前位置最近点再往前 3.0 m」的点（路径只在每集第一次调用时从起飞后的实际位姿规划一次，之后不重规划；剩余路径不足 3 m 时取 waypoint）；`goal` = 直接取 waypoint。
- 它用了真值位姿和占据图，是特权信息，不是学生输入。类名带 `Oracle`（`OracleRouteGuide`、`OracleGuardedPolicy`），报告里带 `+route` / `+goal` 的臂不能当成学生自己的能力。
- 参数起跑前定死，没有在这批路线上调：阈值 45°（hold 1.0 单步转角约 31.6°，容许偏 1 步，偏 2 步就纠正），前视 3.0 m。

## 设计

| 项 | 值 |
| --- | --- |
| 场景 × 路线 | garden、wanderland_1MlUucug、bicycle、callinstrasse，每场景 6 条路线（`scene.json` 的 `routes`） |
| 集数 / seed | 每路线 5 集；seed = 7000 + 100·路线 + k；出生扰动 ±0.5 m / ±10°，与 11-09 相同，同 seed 各臂出生位姿相同，可逐集配对 |
| 动作窗口 / 步数 | hold 1.0 s；每路线步数 = ceil(max_steps × 0.5) + 15（电池约 100 s 的限制）；`stop_on_collision=True` |
| 动力学 | cf_sitl，lib31（`cf_sitl_v1:1bc8fedf:lib31`） |
| 推理 | `bin-snapshots/main-97ae04b/uavrgb_infer`；学生权重 `exp-swa-calib-20261110/ckpt/cal_swa_s{0,1,2}_k10.bin` |
| 臂（前 4 个是预注册主臂） | `forward`、`forward+route`、`calswa-s1`、`calswa-s1+route`；次臂 `calswa-s1+goal`、`calswa-s0+route`、`calswa-s2+route`、`teacher`（lib31 下重跑的上界） |
| 并行 | 两条车道，每个驱动进程一个场景（显存各 2–3 GB）；先把四个场景的主臂跑完，再补次臂 |

## 预注册（全文在 `run.sh` 文件头）

主比较三个，逐集配对，指标 SR(stop)，精确 McNemar：

1. P1 `forward+route` 对 `forward`：预测提高（b > c），合计至少高 0.15；证伪是不高。
2. P2 `calswa-s1+route` 对 `calswa-s1`：预测提高（b > c）；证伪是 b ≤ c。
3. P3 `calswa-s1+route` 对 `forward+route`：预测没有可检出的差别（p > 0.05）；证伪是学生加护栏显著好于盲飞加护栏（p < 0.05 且 b > c）。

## 指标

- SR(stop)：首次进入 waypoint 半径早于首次碰撞（与 11-09 主列同义；学生与盲前进到点后不停）。SR(v2)：`3dgs.pilot.v2`，过点后再撞也算失败。
- 碰撞率（整集任一步碰撞，与 11-09 同）、撞在前（没到点就撞）、超时率（既没到点也没撞）、SPL(stop)（成功换成 SR(stop)）。
- 护栏改写的步占比：护栏把内层动作改成别的动作的步数 / 总步数，分「到点或撞之前」和「整集」两个口径；到点后学生与盲前进会在 waypoint 附近绕圈，整集口径被绕圈的步数抬高。
- 分母 = 已写 `done.json` 的路线数 × 5（失败集算失败，单列）；Wilson 95% 区间；`metric_version = 3dgs.pilot.v2+stop.v1`。

## 怎么跑

```bash
cd experiments/2026-11-10-bearing-guard-3dgs
# 正式实验：守护进程（flock 互斥；写 run.pid / run.log / DONE；配置区在 run.sh 顶部，环境变量可覆盖）
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 冒烟：1 个场景 2 条路线 1 集，所有臂一次加载跑完
OUT=<草稿目录>/smoke LANE_A=garden LANE_B= ROUTES="4 1" K=1 N_MAIN=8 setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度与汇总（汇总只读每集 metrics.json，不加载场景，随时可跑）
tail -f $OUT/run.log; grep -v '^step ' $OUT/driver-garden.log | tail
<3dgs-venv python> summarize.py $OUT        # 写 $OUT/summary.md 与 summary.json
# 停（只按 PGID）
kill -- -$(cat $OUT/run.pid)
```

续跑：同一条命令重起，已完成的路线（`done.json` 集数够且无失败）跳过，半成品改名为 `*.incomplete-<时刻>` 后重跑。

## 目录

- `run.sh`：配置区、预注册、守护进程入口。
- `driver.py`：每个场景加载一次，逐集调用 `collect_3dgs.run_episode`；每集落 `metrics.json`、`guard.jsonl`（逐步方位误差、是否改写），每路线落 `done.json`。逐帧 PNG 每集约 4 MB，960 集会占 4 GB，超出 3 GB 配额，所以每集只留首帧 `rgb_00000.png`。
- `summarize.py`：表格与配对检验，按预注册规则机械给出 P1–P3 的判决，结论以人工判读为准。
- 输出：`/root/rivermind-data/uav_rgb/exp-bearing-guard-3dgs-20261110/<臂>/<场景>/r<N>/ep_*`。

## 2026-11-11 起的教师容差

driver 构造教师时传 `hold_s`（臂的生效 hold，没有臂级 hold 时就是 `CFG["hold"]`），教师的航向容差按它推导；显式的 `teacher_tol_deg` 仍然优先。
不带 `teacher_tol_deg` 的旧配置在 hold 1.0 下重跑会得到 18.9° 的教师，复现旧结果要显式写 `"teacher_tol_deg": 12.0`。

## 已知限制

- 前视点按顶点找最近点，适用于 0.1 m 一格的 A* 栅格路径；稀疏折线要改线段投影（`shortcut:` 注释在 `guard.py`）。
- 护栏只在 |误差| ≥ 阈值时改写成转向，不管前方有没有障碍；碰撞留给内层策略。
- 护栏用真值位姿，真机上对应的是上层给的粗方向加一个航向估计，这里没有评估航向估计的误差。
