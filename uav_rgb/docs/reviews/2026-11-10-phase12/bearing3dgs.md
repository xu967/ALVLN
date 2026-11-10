# bearing3dgs 报告：3DGS 路线方位护栏 × {盲前进, 学生} 判别实验

- 任务开始 2026-11-10 19:22（任务书 `tmp/phase12-20261110/BRIEF_COMMON.md`）；代码与冒烟在 19:40 前完成，正式实验 19:40:44 起跑，19:57:58 跑完（`OUT/DONE`：`rc=0`）。
- 代码工作树 `/root/claude-code-backup-20261028/code/wt/p12-bearing3dgs`，分支 `p12-bearing3dgs`（基于 main `e125b3c`），两个提交：`0fe27cc`（feat(3dgs)）、`99e3154`（feat(experiments)）。工作树干净，没推送，没改主仓。
- 草稿目录（下文记作 `B`）`/root/rivermind-data/uav_rgb/tmp/phase12-20261110/bearing3dgs/`；实验输出（下文记作 `OUT`）`/root/rivermind-data/uav_rgb/exp-bearing-guard-3dgs-20261110/`。
- 正式实验 8 个臂 × 120 集全部跑完，失败集 0；数字全部来自 `OUT/summary.md` 与 `OUT/summary.json`（`summarize.py` 自动汇总，19:57:57），个别探索性数字注明脚本。

## 1. 代码改动清单

| 文件 | 行数 | 提交 | 内容 |
| --- | --- | --- | --- |
| `sim/3dgs/guard.py` | 119（新） | `0fe27cc` | `bearing_error_deg`、`bearing_guard_action`、`OracleRouteGuide`、`OracleGuardedPolicy`；任务书里的名字 `RouteGuide`、`GuardedPolicy` 是同名别名 |
| `sim/3dgs/test_guard.py` | 363（新） | `0fe27cc` | 24 项 unittest |
| `experiments/2026-11-10-bearing-guard-3dgs/run.sh` | 131（新） | `99e3154` | 守护进程入口（flock、run.pid、run.log、DONE、配置区），预注册写在文件头 |
| `experiments/2026-11-10-bearing-guard-3dgs/driver.py` | 192（新） | `99e3154` | 基于 11-09 的 `xeval_3dgs_driver.py`，每场景加载一次，臂 = 底座 + 护栏 |
| `experiments/2026-11-10-bearing-guard-3dgs/summarize.py` | 213（新） | `99e3154` | 只读每集 `metrics.json`；Wilson 区间、逐集配对精确 McNemar、P1–P3 机械判决 |
| `experiments/2026-11-10-bearing-guard-3dgs/README.md` | 71（新） | `99e3154` | 问题、设计、预注册摘要、怎么跑、已知限制 |

`git diff --stat e125b3c..HEAD`：6 个文件、1089 行新增，没有改任何已有文件。`run_episode`、`collect_3dgs.py`、`planner.py`、`eval_3dgs.py` 都没动：`collect_3dgs.py:326` 以 `policy(rgb, depth, dyn)` 调用策略，`dyn` 是动力学对象（`.x/.y/.yaw`，yaw 为弧度，`:332` 取 `dyn.yaw`），包装层直接用它拿真值位姿。

护栏接口：

- `bearing_error_deg(x, y, yaw_deg, target_xy)`：目标方位减机头方位，折到 [-180, 180)，正 = 目标在左。
- `bearing_guard_action(err_deg, threshold_deg)`：`|err| >= 阈值` 返回朝目标转的动作（正 -> `YAW_LEFT`，负 -> `YAW_RIGHT`），否则 `None`；阈值 <= 0 关闭。
- `OracleRouteGuide(plan_fn, waypoint_xy, lookahead_m)`：占据 A* 路径在该集第一次调用时从起飞后的实际位姿规划一次，之后不重规划；最近点索引只增不减；前视点 = 最近点沿路径再往前 `lookahead_m`，剩余不足时取 waypoint；规划失败时一直指向 waypoint，`plan_failed=True`。
- `OracleGuardedPolicy(inner, target_fn, threshold_deg)`：先让内层出动作，再改写；`name` 沿用内层（`run_episode` 靠它写 manifest 的 `executed_by`，学生集不写教师标签）；故意不带 `stop_radius`、`prev`、`n_plan_fail`、`n_expert_exec`、`planner_name`，否则 `run_episode`（`collect_3dgs.py:287-299`）会把它当成 goal-stop 教师；每步一行 trace（`step/inner/action/err_deg/target/triggered/rewritten`）。
- 类名带 `Oracle`：护栏用真值位姿和占据图，是特权信息，带 `+route`、`+goal` 的臂不能当成学生自己的能力。

### 测试证据

| 项 | 结果 | 来源 |
| --- | --- | --- |
| `test_guard.py` 红 | `ModuleNotFoundError: No module named 'sim.3dgs.guard'`（实现之前） | `B/test_guard_red.log` |
| `test_guard.py` 绿 | `Ran 24 tests in 2.294s`，`OK` | `B/test_guard_green.log` |
| 变异测试（提交版本，在 `sim/` 的拷贝上改，工作树不动） | 9 个变异全被抓到（各至少 1 项失败）；还原后 24 项 `OK`；工作树 `git status` 0 行 | `B/test_guard_mutations_committed.log`，脚本 `B/mutate_check.sh` |
| `sim/3dgs/test_*.py` 全部 | 7 个文件、110 项（collect_eval 28、collision 26、dynamics 6、dynamics_sitl 11、guard 24、renderer 11、video 4），7 个 rc=0 | `B/test-3dgs-all.log` |
| `tools/check.sh --python-only` | rc=0，输出 `PASS（有 SKIP）：所选检查完成，但未检查：C++/CUDA 构建、CTest、GPU 测试、cf_sitl`；基于 HEAD `99e3154` | `B/check-python-only-2.log`、`B/check-python-only-2.DONE` |

9 个变异：误差号取反（10 项失败）、最近点取全路径全局最近（2 项）、最近点取从 k 往前的全局最小（1 项）、阈值 <= 0 不关闭（1 项）、边界 `>=` 改 `>`（1 项）、少一次 `math.degrees`（2 项）、每步重新规划（2 项）、包装后 `name` 不沿用内层（2 项）、内层动作等于护栏动作时也记成改写（1 项）。

## 2. 坐标与角度约定的核实

约定：ENU 导航系，yaw 0 = +x，逆时针为正，`YAW_LEFT` 使 yaw 增大；误差 = 目标方位 - 机头方位，正 = 目标在左 -> `YAW_LEFT`。四路核对：

1. 读代码：`planner.py:223-227`，`OccAstarTeacher` 用 `err = _wrap(degrees(atan2(ty - y, tx - x)) - yaw_deg)`，`err > tol` 出 `YAW_LEFT`、`err < -tol` 出 `YAW_RIGHT`；`guard.py` 的公式相同，直接复用同一个 `_wrap`。
2. 单测对账教师：`test_sign_matches_planner_teacher` 用桩规划器在 400 个随机位姿上比对护栏与 `OccAstarTeacher` 的转向号，全部一致。
3. 单测对账动力学：`test_sign_matches_cf_blackbox_dynamics` 把护栏接到 `CfBlackbox` 上闭环，`YAW_LEFT` 后 yaw 增大、误差收敛。
4. 真 cf_sitl 冒烟（`B/smoke2/forward+route/garden/r4/ep_00000`）：起点 yaw 97.8°，误差 +116.07，护栏连出 3 个 `YAW_LEFT`，每步起点 yaw 依次为 97.8 -> 129.6 -> 164.1 -> -161.6（即 198.4°，逆时针），误差依次 +116.07 -> +83.69 -> +49.25 -> +14.86，之后护栏不再触发。

变异 1（误差号取反）会让 10 项测试失败，其中包括上面第 2、3 两路对账。

单步转角：任务书写约 31.6°，11-09 文档写 hold 1.0 每步约 27°（`docs/experiments/2026-11-09-xeval-3dgs.md` 第 5 行）。正式实验里按 `actions.jsonl` 的 `dyaw_deg` 统计（`OUT/<臂>/*/r*/ep_*/`，19:58）：`forward+route` 的 `YAW_LEFT` 1259 步均值 +33.2°（p10 31.4、p50 34.2、p90 34.5），`YAW_RIGHT` 1525 步均值 -33.2°；`teacher` 的 `YAW_LEFT` 1296 步均值 +29.1°（p50 27.6），`YAW_RIGHT` 1301 步均值 -29.3°。同一个 hold 1.0 下，教师臂的单步转角比护栏臂小，与 11-09 的 27° 接近；差别和动作历史有关（教师左右交替多，护栏连续同向转多）只是推测，没核实。对阈值 45° 没有影响：偏 1 步（约 29–34°）不触发，偏 2 步（约 58–68°）触发。

## 3. 冒烟结果

两次冒烟，garden 路线 4 与 1，每路线 1 集，8 个臂各跑一遍，都 rc=0（`B/smoke/DONE` 19:36:37，`B/smoke2/DONE` 19:38:59；两次都是在未提交的工作树上跑的，基于 `e125b3c`）。第二次是为了加「到点前」口径的改写占比（见下）而重跑。

| 臂 | SR(stop) | 碰撞 | 到点前改写步占比 / 整集 | 来源 |
| --- | --- | --- | --- | --- |
| `forward` | 0/2 | 2/2 | 无护栏 | `B/smoke2/summary.md` |
| `forward+route` | 2/2 | 0/2 | 0.11 / 0.37 | 同上 |
| `calswa-s1` | 0/2 | 2/2 | 无护栏 | 同上 |
| `calswa-s1+route` | 2/2 | 0/2 | 0.09 / 0.37 | 同上 |
| `calswa-s1+goal` | 0/2 | 2/2 | 0.03 / 0.03 | 同上 |
| `calswa-s0+route` | 2/2 | 0/2 | 0.21 / 0.33 | 同上 |
| `calswa-s2+route` | 2/2 | 0/2 | 0.13 / 0.36 | 同上 |
| `teacher` | 2/2 | 0/2 | 无护栏 | 同上 |

检查项与结果：

- 护栏方向收敛到 waypoint：第 2 节第 4 路的轨迹。`forward+route` 在 r4、r1 两条路线上都到点（到点步 38、41），到 waypoint 的距离（每 3 步采样）一路下降到 0.1–0.3 m（`B/smoke_look.log`）。
- 改写占比：整集口径 0.33–0.37，到点前口径 0.09–0.21。差别来自学生与盲前进到点后不停，在 waypoint 附近绕圈，护栏在绕圈时反复触发（例如 `forward+route` r4：整集 32/76 步，到点前 6/39 步）。所以 driver 与 summarize 同时给两个口径，并给每步 trace 加了 `triggered` 字段。
- 无护栏的 `calswa-s1` 在两条路线上的动作全是 `FORWARD_SLOW`，与盲前进的动作序列和碰撞步（r4 第 21 步、r1 第 23 步）完全相同（`B/smoke_look.log`）。
- `calswa-s1+goal`（只给 waypoint 方位，不给路径）两集都撞：r1 整集 24 步 |误差| 都小于 45°，护栏一次没触发，直线撞上障碍；r4 触发 1 次后在第 6 步撞（`B/smoke_look.log`）。
- 同 seed 各臂起飞后首步位姿最大差 0.00（位置与航向都是），逐集配对成立（`B/smoke2/summary.md` 第 5 节）。
- 两次冒烟的每集 `metrics.json` 去掉 `wall_s` 后只有 `guard` 字段不同（第二次多了 `_pre` 统计），其余逐字段一致：同 seed 的结果可复现。
- 单集耗时约 1–3 s（`metrics.json` 的 `wall_s`，例如 `forward` garden r4 为 1.529 s），两次冒烟各约 70 s（场景加载 36 s，见 `B/smoke2/driver-garden.log`）。
- 教师在 hold 1.0 s 下的动作串里多处左右交替（`B/smoke_look.log`，r4 的 `…FFLRFFLRFLRFF…`），11-09 文档第 5、55 行把它记作 FIXLIST D9；本实验不处理。

## 4. 正式实验

起跑 2026-11-10 19:40:44，`run.log` 首行记 `WT=99e3154`，`inputs.txt` 记 `git HEAD 99e31548… 未提交改动 0 个文件`（起跑时工作树干净，是提交版本）。输入文件的 sha256 在 `OUT/inputs.txt`（`uavrgb_infer`、`cf_sitl_proc`、gsplat `.so`、`e1_mini.json`、`guard.py`、`driver.py`、`summarize.py`、三个学生权重、四个 `scene.json`）。

| 项 | 值 |
| --- | --- |
| 进程组（PGID） | 1289064（`OUT/run.pid`，run.sh 自己的 pid）；跑完后该进程组下 0 个进程（`ps` 19:58 核对） |
| 并行 | 2 个 3DGS 驱动进程：车道 A = garden -> wanderland_1MlUucug，车道 B = bicycle -> callinstrasse；每个车道先跑阶段 1（4 个主臂）再跑阶段 2（4 个次臂），所以每个场景加载两次 |
| GPU | 只用 `GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70`（总 22528 MiB）；19:49 的 `nvidia-smi` 里 GPU 1 上的 4 个进程全属 PGID 1289064（2 个驱动 8010 + 4198 MiB，2 个 `uavrgb_infer` 各 164 MiB），共 12539 MiB；GPU 0 的占用来自别的项目，没碰 |
| 臂 | 主臂 `forward`、`forward+route`、`calswa-s1`、`calswa-s1+route`；次臂 `calswa-s1+goal`、`calswa-s0+route`、`calswa-s2+route`、`teacher` |
| 规模 | 4 场景 × 6 路线 × 5 集 × 8 臂 = 960 集，每个臂 120 集，失败集 0 |
| 时间线 | 19:40:44 起跑；19:48:39 阶段 1（4 个主臂，480 集）完成；19:55:44 车道 A 结束；19:57:57 车道 B 结束；19:57:58 `DONE` 写 `rc=0`；总用时 17 分 14 秒（`OUT/run.log`） |
| 磁盘 | 每集只留首帧 PNG；`OUT` 共 404 MB（`du -sh`，19:58）；数据盘剩 15 GB；本任务新增在 3 GB 配额内 |

命令：

```bash
D=/root/rivermind-data/uav_rgb; OUT=$D/exp-bearing-guard-3dgs-20261110
WT=/root/claude-code-backup-20261028/code/wt/p12-bearing3dgs
# 起跑（已执行过一次；同一条命令重起即续跑：done.json 集数够且无失败的路线跳过，半成品改名 *.incomplete-<时刻> 后重跑）
cd $WT/experiments/2026-11-10-bearing-guard-3dgs && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：车道日志、每臂已完成的路线数（满 24 = 4 场景 × 6 路线）
tail -f $OUT/run.log
for a in forward forward+route calswa-s1 calswa-s1+route calswa-s1+goal calswa-s0+route calswa-s2+route teacher; do echo $a $(ls $OUT/$a/*/r*/done.json 2>/dev/null | wc -l)/24; done
# 驱动逐步日志
grep -v '^step ' $OUT/driver-garden.log | tail
# 结束标记与汇总（run.sh 末尾自动汇总，写 $OUT/summary.md、summary.json、summarize.log；DONE 内容 rc=<n> <时刻>）
cat $OUT/DONE
# 手动汇总（只读每集 metrics.json，不加载场景；会覆盖 $OUT/summary.md）
$D/3dgs-venv/bin/python $WT/experiments/2026-11-10-bearing-guard-3dgs/summarize.py $OUT
# 停（只按 PGID，不用 pkill）
kill -- -$(cat $OUT/run.pid)
```

## 5. 实验数字（正式实验已跑完）

统计口径：`metric_version = 3dgs.pilot.v2+stop.v1`；SR(stop) = 首次进入 waypoint 半径早于首次碰撞；分母 = 请求集数（失败集算失败，本批失败集 0）；同 seed 各臂出生位姿相同（最大差 0.00），逐集配对；护栏阈值 45°，前视 3.0 m。

### 5.1 四个主臂（四场景合计，每臂 n = 120）

| 臂 | SR(stop) [Wilson 95%] | SR(v2) | 碰撞率 | 撞在前 | 超时率 | SPL(stop) | 护栏改写步占比（到点或撞前 / 整集） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `forward` | 37/120 = 0.31 [0.23, 0.40] | 10/120 | 0.81 | 0.58 | 0.11 | 0.30 | 无护栏 |
| `forward+route` | 107/120 = 0.89 [0.82, 0.94] | 107/120 | 0.10 | 0.10 | 0.01 | 0.85 | 0.05 / 0.36 |
| `calswa-s1` | 30/120 = 0.25 [0.18, 0.33] | 9/120 | 0.78 | 0.61 | 0.14 | 0.24 | 无护栏 |
| `calswa-s1+route` | 105/120 = 0.88 [0.80, 0.92] | 105/120 | 0.10 | 0.10 | 0.03 | 0.81 | 0.07 / 0.32 |

2×2 读法：行 = 底座（盲前进、学生），列 = 无护栏 / 路线护栏。盲前进 0.31 -> 0.89，学生 0.25 -> 0.88。

### 5.2 预注册的三个配对检验（逐集配对，精确 McNemar；b = A 成 B 败，c = A 败 B 成）

| 编号 | A vs B | 配对数 | b | c | Δ = SR(A) - SR(B) | p | 脚本按预注册规则给的判决 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | `forward+route` vs `forward` | 120 | 70 | 0 | +0.583 | 1.7e-21 | 支持预测（b>c 且高 ≥ 0.15） |
| P2 | `calswa-s1+route` vs `calswa-s1` | 120 | 75 | 0 | +0.625 | 5.3e-23 | 支持预测（b>c） |
| P3 | `calswa-s1+route` vs `forward+route` | 120 | 1 | 3 | -0.017 | 0.625 | 支持预测（p>0.05，无可检出差别） |

p 值取自 `OUT/summary.json` 的 `pairs[].p`。P3 的 120 对里只有 4 对结果不同（b + c = 4）；「p > 0.05」是对零差别的预测，没有显著不等于等价，这里同时给出 4 对不同这个事实。

### 5.3 分场景 SR(stop)（n = 30 / 格，括号内是碰撞率）

| 臂 | garden | wanderland_1MlUucug | bicycle | callinstrasse |
| --- | --- | --- | --- | --- |
| `forward` | 10/30（1.00） | 12/30（0.83） | 7/30（0.70） | 8/30（0.70） |
| `forward+route` | 28/30（0.07） | 21/30（0.27） | 29/30（0.03） | 29/30（0.03） |
| `calswa-s1` | 7/30（1.00） | 7/30（0.90） | 7/30（0.73） | 9/30（0.50） |
| `calswa-s1+route` | 28/30（0.07） | 21/30（0.27） | 28/30（0.03） | 28/30（0.03） |
| `calswa-s1+goal` | 19/30（0.37） | 19/30（0.37） | 23/30（0.13） | 25/30（0.10） |
| `calswa-s0+route` | 27/30（0.07） | 18/30（0.30） | 27/30（0.03） | 29/30（0.00） |
| `calswa-s2+route` | 28/30（0.07） | 21/30（0.27） | 29/30（0.03） | 26/30（0.03） |
| `teacher` | 24/30（0.00） | 24/30（0.03） | 25/30（0.00） | 27/30（0.00） |

### 5.4 四个次臂（四场景合计，每臂 n = 120；补充比较不进判决）

| 臂 | SR(stop) [Wilson 95%] | SR(v2) | 碰撞率 | 超时率 | SPL(stop) | 护栏改写步占比（到点或撞前 / 整集） |
| --- | --- | --- | --- | --- | --- | --- |
| `calswa-s1+goal`（只给 waypoint 方位） | 86/120 = 0.72 [0.63, 0.79] | 86/120 | 0.24 | 0.04 | 0.65 | 0.07 / 0.30 |
| `calswa-s0+route` | 101/120 = 0.84 [0.77, 0.90] | 101/120 | 0.10 | 0.06 | 0.77 | 0.10 / 0.30 |
| `calswa-s2+route` | 104/120 = 0.87 [0.79, 0.92] | 104/120 | 0.10 | 0.03 | 0.81 | 0.07 / 0.31 |
| `teacher`（lib31 下重跑的上界） | 100/120 = 0.83 [0.76, 0.89] | 100/120 | 0.01 | 0.16 | 0.73 | 无护栏 |

| 编号 | A vs B | 配对数 | b | c | Δ | p |
| --- | --- | --- | --- | --- | --- | --- |
| x1 | `calswa-s1` vs `forward` | 120 | 3 | 10 | -0.058 | 0.092 |
| x2 | `calswa-s1+goal` vs `calswa-s1` | 120 | 56 | 0 | +0.467 | < 0.001 |
| x3 | `calswa-s1+goal` vs `calswa-s1+route` | 120 | 1 | 20 | -0.158 | < 0.001 |
| x4 | `calswa-s0+route` vs `forward+route` | 120 | 1 | 7 | -0.050 | 0.070 |
| x5 | `calswa-s2+route` vs `forward+route` | 120 | 1 | 4 | -0.025 | 0.375 |
| x6 | `teacher` vs `calswa-s1+route` | 120 | 12 | 17 | -0.042 | 0.458 |
| x7 | `teacher` vs `forward+route` | 120 | 9 | 16 | -0.058 | 0.230 |

x1–x7 不在预注册里，没做多重比较校正，p 值只当描述用。

### 5.5 护栏臂没到点的集集中在哪里

- `forward+route` 的 13 个失败集：wanderland_1MlUucug r4 5 集（0/5）、r5 3 集（2/5），另 5 集各 1 个，在 wanderland r0、garden r2、garden r4、bicycle r5、callinstrasse r2。`calswa-s1+route` 的 15 个失败集：wanderland r4 5 集、r5 3 集（与盲前进加护栏相同），另 7 集各 1 个，在 wanderland r0、garden r2、garden r4、bicycle r4、bicycle r5、callinstrasse r1、callinstrasse r3（`OUT/summary.md` 第 3 节）。
- wanderland r4（`r4_face_tree_turn`，出生正对 3 m 外的树丛）：两个主护栏臂的 5 集都在第 3–6 步撞，两个臂的撞击步完全相同（6、3、5、5、4，`OUT/<臂>/wanderland_1MlUucug/r4/ep_*/metrics.json`）；`calswa-s0+route`、`calswa-s2+route` 在 r4 也是 0/5，而 `teacher` 是 5/5（`OUT/summary.md` 第 3 节）。
- 逐步看 `forward+route` 的 ep_00000：占据代理给出出生点净空 1.21 m，护栏连出两次 `YAW_RIGHT`（误差 -109.5 -> -77.4），之后误差 -43.3 小于阈值，护栏放行 `FORWARD_SLOW`，第 6 步在净空 0.08 m 处撞上，那一步护栏误差是 -15.8°（`guard.jsonl`、`actions.jsonl` 的 `clearance_m`）。护栏只看航向误差，不看前方障碍，教师 A* 有前向检查（`planner.py:228-233`）；这几集是护栏设计的盲区，不是路径规划失败（所有护栏臂的规划失败集数为 0，`OUT/summary.md` 第 5 节）。
- 无护栏的两个臂在 wanderland r4 也是 0/5。`calswa-s1` 的 5 集里 4 集动作全是前进，1 集（ep_00003）有 2 次 `YAW_RIGHT`，5 集都撞（`OUT/calswa-s1/wanderland_1MlUucug/r4/ep_*/actions.jsonl`）。
- wanderland r5 的 ep_00003 在第 0 步就碰撞，8 个臂全部如此，包括教师（`OUT/<臂>/wanderland_1MlUucug/r5/ep_00003/metrics.json` 的 `collision_step = 0`，19:59 统计）。这一集没有任何臂能过，SR 上限是 119/120 = 0.99；原因（出生位姿落在占据里还是首步动作的窗口内撞）没查（未核实）。

### 5.6 探索性分析（预注册之外，不进判决；`B/student_actions.py`，输出 `B/student_actions_final.md`，19:58）

学生与盲前进「自己的」输出分布（到点或撞之前的步；护栏臂取 `guard.jsonl` 的 `inner`，即护栏改写之前的动作）：

| 臂 | 集数 | 步数 | FORWARD_SLOW | YAW_LEFT | YAW_RIGHT | SAFE_NEUTRAL | 含非前进输出的集占比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `forward` | 120 | 4032 | 1.000 | 0 | 0 | 0 | 0/120 |
| `calswa-s1` | 120 | 4418 | 0.961 | 0.008 | 0.018 | 0.012 | 53/120 = 0.44 |
| `calswa-s1+route` | 120 | 4188 | 0.935 | 0.011 | 0.038 | 0.016 | 59/120 = 0.49 |
| `calswa-s1+goal` | 120 | 3920 | 0.918 | 0.013 | 0.052 | 0.017 | 54/120 = 0.45 |
| `calswa-s0+route` | 120 | 4491 | 0.891 | 0.016 | 0.080 | 0.013 | 82/120 = 0.68 |
| `calswa-s2+route` | 120 | 4229 | 0.917 | 0.014 | 0.048 | 0.021 | 58/120 = 0.48 |

逐集配对，两臂实际执行的动作序列完全相同的集：`calswa-s1` vs `forward` 62/120，`calswa-s1+route` vs `forward+route` 57/120，`calswa-s2+route` vs `forward+route` 57/120，`calswa-s0+route` vs `forward+route` 29/120。学生 89–96% 的输出是前进；非前进输出最多的 s0（YAW_RIGHT 占 8.0%）在路线护栏下 SR 也最低（0.84）。

盲前进与 11-09 对账：本实验 `forward` 臂 120 集的步数与碰撞标志和 11-09 盲前进组（`xeval-3dgs-20261109/forward/*/r*/done.json`）逐集全部相同；SR(stop) 合计都是 37/120；SR(v2) 合计不同（本实验 10/120，11-09 为 8/120）。11-09 的动力学 profile 是 `cf_sitl_v1:683e7400`，本实验是 `cf_sitl_v1:1bc8fedf:lib31`；SR(v2) 差 2 集的原因没查（未核实）。教师臂本实验 100/120 = 0.83，与 11-09 的教师 100/120 = 0.83 一致（`docs/experiments/2026-11-09-xeval-3dgs.md` 第 2 节）；逐集是否相同没对。

### 5.7 读法（我的判读，以主会话为准）

1. 目标信息是最大的单项因素。学生无护栏 0.25；只给 waypoint 方位（`+goal`）0.72（x2：b=56、c=0）；再给绕障路径（`+route`）0.88（x3：`+goal` 比 `+route` 低 0.158，b=1、c=20）。盲前进同样从 0.31 升到 0.89（P1）。
2. 在这批路线上，学生自己的视觉输出没有表现出增益：无护栏时学生 0.25 低于盲前进 0.31（x1，p=0.092，不显著）；加路线护栏后三个种子的学生（0.88、0.84、0.87）都不高于盲前进（0.89），差 -0.017（P3）、-0.050（x4，p=0.070）、-0.025（x5，p=0.375），均不显著；学生 89–96% 的输出是前进；s1、s2 在路线护栏下各有 57/120 集与盲前进加护栏的动作序列逐步相同，s0 为 29/120（5.6 节）。
3. 这个读法有两处没拆干净。其一，`+route` 的目标点来自占据 A* 路径，带着绕障信息，`+route` 测的是「目标方位 + 绕障路径」的合并效果；护栏臂已到 0.88–0.89，学生最多只能再多出 0.11，P3 检出小差别的功效有限。其二，`+goal` 没有盲前进对照（`forward+goal` 不在任务书的臂清单里），0.72 里多少来自护栏、多少来自学生，现有数据分不开；学生在 `+goal` 下 91.8% 的输出仍是前进（5.6 节），推测盲前进加 `+goal` 会落在接近的位置，但没跑，未核实。
4. 对「下一步长程路线任务先补目标条件还是先补真实外观感知」：现有数据支持目标/路线条件排在前面（没有它，学生与盲前进无异；有了它，盲前进也能到 0.89，教师是 0.83，x7 的 p=0.230，没有可检出的差别）。真实外观上的避障感知有没有价值，这个 2×2 回答不了，需要补 `forward+goal` 对照，或换护栏臂拿不满分的更难路线（出生点近障碍、窄通道，例如 wanderland r4：教师 5/5，所有护栏臂 0/5）再测。

## 6. 风险与需要主会话拍板的事

1. 是否补跑 `forward+goal`（盲前进 + waypoint 方位）：与已有的 `calswa-s1+goal` 配对，才能回答「只给目标方位时学生视觉是否有用」。复用同一套脚本，加一个臂定义即可；耗时按本次次臂的用时估计约 5 分钟，未实测。这是看过结果之后加的臂，属于事后决定，需要你确认再跑。
2. P3 的「无差别」受天花板限制：护栏臂 0.88–0.89，且 wanderland r4、r5 的 8 集两个护栏臂结果相同（护栏盲区，没有前向障碍检查）。这 8 集对 P3 的 b、c 没有贡献，但把天花板压在了 0.89。
3. 护栏的盲区（无前向障碍检查）是设计取舍，不是 bug；要做更强的 oracle 护栏需要新的预注册，不能在这批路线上事后调参。
4. 每个 (场景, 路线) 的 5 集只差出生扰动（±0.5 m / ±10°），集间不是独立样本，McNemar 把 120 对当独立对；P1、P2 的 c 都是 0、b 分别是 70 和 75（p 在 1e-21 量级），P3 只有 4 对不同，结论方向不受影响，但 p 值的绝对大小不要照字面用。
5. 护栏用真值位姿，真机上要靠上层给的粗方向加航向估计，这里没评估估计误差，数字是上界。
6. 学生权重是 `exp-swa-calib-20261110` 的 `cal_swa_s{0,1,2}_k10.bin`，与 11-09 的三个学生（硬标签 BC、成功线截断、sgvis）不是同一批；本文的「学生」指前者，主臂只用 s1，s0、s2 是同配方的另两个种子。
7. 11-09 文档写 hold 1.0 每步转约 27°，本实验护栏臂实测 33–34°、教师臂 29°（第 2 节）；原因没查，对本实验的判决没有影响。
8. README 里写的单步转角约 31.6° 取自任务书，与实测不符；README 里「每集约 4 MB、960 集约 4 GB」是按 11-09 总量折的估计，本实验实际 `OUT` 共 404 MB。两处文字起跑后没改（再提交会让 `run.log`、`inputs.txt` 记的 HEAD `99e3154` 对不上历史），不影响结果，需要的话在合并时顺手改。

## 做了什么

- 读了 `collect_3dgs.py`、`planner.py`、`eval_3dgs.py`、`dynamics.py`、`sim/tools/student.py` 与 11-09 的 `xeval_3dgs_driver.py`、`summarize.py`、`run_xeval_3dgs.sh`、`docs/experiments/2026-11-09-xeval-3dgs.md`；确认策略接口 `(rgb, depth, state)` 的 `state` 就是动力学对象，护栏做成包装层，没改 `run_episode`。
- 写 `sim/3dgs/guard.py`（119 行）与 `sim/3dgs/test_guard.py`（363 行、24 项），先红后绿，9 个变异全部被抓到。
- 写 `experiments/2026-11-10-bearing-guard-3dgs/` 四个文件（run.sh、driver.py、summarize.py、README.md，共 607 行），预注册写进 `run.sh` 文件头。
- 在 garden 的路线 4 与 1 上冒烟两次（8 个臂各 1 集）；第二次为了「到点前」改写占比口径重跑。
- 跑 `sim/3dgs/test_*.py` 全部 7 个文件（110 项）与 `tools/check.sh --python-only`，都 rc=0。
- 提交两次：`0fe27cc`（feat(3dgs)，guard.py + test_guard.py）、`99e3154`（feat(experiments)，四个入口文件）；逐文件 `git add`，提交后工作树干净（实验期间也核对过）。
- 19:40:44 起跑正式实验（两条车道，PGID 1289064），19:57:58 跑完 `rc=0`，960 集、失败集 0，`summarize.py` 自动汇总。
- 预注册之外加了三处只读分析，标为探索性：`B/student_actions.py`（学生自己的输出分布、与盲前进动作序列是否逐步相同）、单步转角统计、盲前进与 11-09 的逐集对账；它们不进 P1–P3 判决。

## 没做什么

| 没做的事 | 原因 |
| --- | --- |
| 没改 `run_episode`、`collect_3dgs.py`、`planner.py`、`eval_3dgs.py` | 策略的 `state` 已带真值位姿，包装层够用；任务书要求没必要不改 |
| 没训练，没做目标条件化，没做真实外观感知 | 本任务只做评测端护栏；做哪个由实验结果和主会话定 |
| 没调护栏参数（阈值 45°、前视 3.0 m），没加前向障碍检查 | 预注册要求起跑前定死，不在这批路线上调；加检查需要新的预注册 |
| 没加 `forward+goal` 臂 | 不在任务书的臂清单里；结果出来后再加属于事后决定，需要主会话拍板（第 6 节第 1 条） |
| 没起 Gazebo | 任务书规定本任务不起 |
| 没跑 C++/CUDA 构建、CTest、GPU 测试、cf_sitl 相关检查，没跑全量 `check.sh` | 只改了 Python 与实验脚本；`--python-only` 通过，全量检查的结果未核实 |
| 没改主仓库，没改 `progress.md`，没推送 | 主仓只读，进度汇总由主会话做 |
| 没写 `docs/experiments/` 下的实验记录 | 任务书的交付清单没有这一项，且结论要主会话判读后再落 |
| 没评估航向估计误差 | 护栏用真值位姿（oracle），真机上对应的是上层粗方向加航向估计，这里不覆盖；README 的已知限制已写 |
| 没修 README 里「单步转角约 31.6°」「每集约 4 MB、960 集约 4 GB」两处估计，没去重 `inputs.txt` 里 s1 权重的三行重复 | 起跑后再提交会让 `run.log`、`inputs.txt` 记的 HEAD `99e3154` 对不上历史；都不影响结果 |
| 没对 P1–P3 之外的比较做多重比较校正 | 预注册只有三个主比较；x1–x7 不进判决，只当描述 |
| 没查 SR(v2) 与 11-09 差 2 集的原因、单步转角 27° / 31.6° / 33° 对不上的原因 | 不影响判决，且超出任务书范围；都标了「未核实」 |
| 没逐集对账教师臂与 11-09 教师组 | 只对了合计（都是 100/120）；逐集对账不在任务书里 |

## 验证到哪一步

| 项 | 证据 | 程度 |
| --- | --- | --- |
| 误差号约定（yaw、左转、YAW_LEFT） | 读代码 + 随机 400 位姿对账教师 + 闭环对账 `CfBlackbox` + 真 cf_sitl 冒烟轨迹（第 2 节） | 已验证 |
| 路线前视点逻辑（规划一次、最近点只增不减、末段取 waypoint、规划失败退回 waypoint） | 单测 + 9 个变异全抓到 + 真 `OccPlanner` 绕墙用例；正式实验 960 集里护栏臂规划失败集数 0 | 已验证 |
| `run_episode` 契约（`name` 沿用内层、不带教师属性） | `test_run_episode_contract` + 真 `run_episode`（mock 渲染 + `CfBlackbox`）闭环用例 + 真 3DGS 冒烟与正式实验 | 已验证 |
| 阈值 <= 0 时与不包装逐位一致 | mock 渲染 + `CfBlackbox` 下的逐位对比用例 | 在 mock 渲染下已验证；真 3DGS 渲染下没有阈值 0 的臂，未核实 |
| `sim/3dgs/test_*.py` 全部 | 7 个文件、110 项、rc=0 | 已验证 |
| `check.sh --python-only` | rc=0（`PASS（有 SKIP）`） | 已验证；C++/CUDA/GPU/cf_sitl 未检查 |
| 冒烟：8 臂 × 2 路线 × 1 集 | 两次 rc=0；同 seed 出生位姿一致；两次结果逐字段一致（除 `guard` 字段） | 已验证 |
| 正式实验 | 960 集、失败集 0、`DONE rc=0`；同 seed 各臂出生位姿最大差 0.00；盲前进臂与 11-09 盲前进组 120 集步数与碰撞标志逐集相同 | 已验证 |
| P1–P3 判决 | `summarize.py` 在正式数据上按预注册规则机械给出：三个都「支持预测」 | 数字已核对；解读（第 5.7 节）以主会话为准 |
| 「真实外观感知有没有价值」 | 现有 2×2 + `+goal` 臂只给出「学生没有显示出增益」，缺 `forward+goal` 对照，且护栏臂近天花板 | 未拆清，见第 5.7 节第 3 条与第 6 节第 1 条 |
