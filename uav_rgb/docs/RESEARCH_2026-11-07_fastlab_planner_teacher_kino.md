# FAST-A：FAST Lab 规划侧代码精读 → E1 动力学感知教师（teacher_kino）

- 日期：2026-11-07 · 作者：FAST-A agent（QUEUE Q2）
- 工作树：`/root/rivermind-data/uav_rgb/tmp/wt-fastlab-20261106`（分支 `fastlab-20261106`，起点 9d764d9；未推送）
- 前置：`reports/B_academic.md`（文献层面不重复）；本报告只讲**代码怎么做的 → 我们对应在哪 → 改了什么 → 预期 → 怎么验**。
- 对比数据：`/root/rivermind-data/uav_rgb/tmp/agent-fastlab-a/compare/{summary.md,summary.json,traj_*.png}`，0.6 m/s 变体在 `compare-fast06/`。

## 0. 结论（先看这个）

1. **E1 以 0.3 m/s 飞当前柱阵，旧的全局 A* 教师（`teacher.plan_action_global`，X3 常量）并没有坏**：20 个 test seed、三组被控对象常数（名义 / τ×1.5 延迟×2 / τ×2.5 延迟×3），corridor 与 l1_stop 都是 100% 到达、0 碰撞。原因是 E1 滑行只有 0.118 m、一窗只走 0.27 m，而柱阵缝 1.2 m、X3 的 0.5 m 规划余量仍留得下路；「假设每步能停住」在这个速度下近似成立。**任务陈述里"旧教师不适配 E1"的风险，在离线几何上不成立**；Gazebo/SITL 里若有失败，根因更可能在偏航跟踪/延迟/感知，不在路径几何。
2. **坏的是另外两个教师**：`teacher.plan_action`（局部版）在 τ×1.5 的被控对象上 20 个 seed 只到 2 个，左右换向 911 次（纯航向追踪 + 1 s 窗口 + 响应滞后 = 极限环）；`teacher_predictive`（cf_stab 模型）原样跑 0 到达——它的 `DistanceField`（0.25 m 格 + 0.38 m 膨胀）把第一排柱前 x∈[1,2] 整条带标成不可达(60)，所有 FORWARD 滚动得 −100 分，停在 SAFE 死锁。
3. **新教师 `teacher_kino` 和旧全局教师在 0.3 m/s 下持平**（corridor 22.0 vs 22.1 步；l1_stop 14.1 vs 15.2 步，少 7%），但多了旧教师做不到的三件事：带速度状态（必撞状态 fail-closed 返回 SAFE，旧教师看不见速度）；一窗原语按实测响应积分（启发/代价都用 0.277 m、27.7° 的真实一窗量）；参数全部来自 `e1_mini.json`（换载具只换 JSON）。把执行速度提到 0.6 m/s 时差距才显出来（§4.3）。
4. 建议：**采集管线暂不切教师**（旧全局教师在 E1 上够用，且标签口径已冻结）；`teacher_kino` 作为 (a) SITL/Gazebo 上 E1 真实轨迹的离线对照器（同接口，喂实测 vx/vy），(b) 若后续 E1 提速或缝变窄再切换。切换前要做的事见 §6。

## 1. 读了哪些代码（file:line 可直接打开）

克隆目录 `/root/rivermind-data/uav_rgb/tmp/research-acad/`。任务单说的 `ego-planner/.../kinodynamic_astar.cpp` **不存在**：ego-planner 只有 ESDF-free 的 `path_searching/src/dyn_a_star.cpp`；运动原语 kinodynamic A* 在 HKUST `Fast-Planner`（ZJU-FAST-Lab 组织下无此仓库，已另克隆 `HKUST-Aerial-Robotics_Fast-Planner`）和 visPlanner 的 `tracking_astar.hpp`（同一份代码加跟踪目标）。

| 仓库 | 文件 | 看点 |
|---|---|---|
| Fast-Planner | `fast_planner/path_searching/src/kinodynamic_astar.cpp:41-270` search()；`:141-167` 输入×时长离散；`:215-222` 原语上 `check_num_` 个采样点查占据；`:234-235` g=(‖u‖²+w_time)τ；`:237-262` 同父节点剪枝；`:87-125` reach_horizon / near_end + 一发 BVP（computeShotTraj）；`:358-393` 庞特里亚金闭式启发 + tie_breaker 1.0001 | 整个 teacher_kino 的骨架 |
| visPlanner | `src/planner/plan_manage/include/plan_manage/tracking_astar.hpp:364-460` | 同上，终止条件换成「跟上被跟踪轨迹的时间」 |
| ego-planner | `bspline_opt/src/bspline_optimizer.cpp:37-200` initControlPoints（A* 路径给每段碰撞控制点定 base point / 方向）；`:342-380` calcDistanceCostRebound（distance<clearance 时三次→二次惩罚，`:347` demarcation=clearance）；`:454-470` calcFeasibilityCost；`:869-960` rebound_optimize（失败→重新 initControlPoints 再优化）| 软余量带 + "几何路径给方向"的思路 |
| ego-planner | `plan_manage/src/ego_replan_fsm.cpp:377-410` checkCollisionCallback（只查前 2/3 轨迹 `:388-391`；距碰撞 <emergency_time_ 0.8 s → EMERGENCY_STOP `:403-406`）；`:504-535` getLocalTarget（planning_horizon 截局部目标）；`:306-311` 重规划阈值 no_replan/replan；`plan_manage/src/planner_manager.cpp:274-283` EmergencyStop = 6 个相同控制点 | 应急停、滚动视野 |
| EGO-Planner-v2 | `swarm-playground/.../plan_manage/src/traj_server.cpp:75-120` calculate_yaw（偏航率 2π/s、偏航加速度 5π/s² 限幅跟踪）；`planner_manager.cpp:325-361` getLocalTarget 末端 `(global_end−local_target) < v²/(2a)` 时局部目标速度置零 | 偏航限幅、刹车距离规则 |
| TGK-Planner | `mapping/occ_grid/src/pos_checker.cpp:25-97` getCheckPos（沿速度方向伸长的检查点集）；`:216-236` checkPolyTraj 返回 remain_safe_time；`decision/state_machine/src/fsm.cpp:262-305` REPLAN_TRAJ 从「指令落地时刻的未来状态」起规划（`getReplanStateFromPath(t_during_traj+dt)`）；`:338-341` 剩余安全时间 < e_stop_time_margin_ 才应急停；`planning/kino_plan/src/krrtplanner.cpp:114-149` 起点在障碍→失败、终点在障碍→20 个方向各挪 0.2 m 找替代；`:189-194` 先试一发 BVP | 未来状态起规划、终点挪位 |
| kino_sampling_with_regional_opti | 同 TGK 结构（kRRT* + 区域优化），没有新的可借点，只对照了 README/结构 | — |
| EVA-planner | `path_searching/src/high_mpcc_optimizer.cpp:395-433`：距障 <dist_1 时惩罚 (dist−dist_1)²·(‖v‖−v_min)²，再乘 `f_beta = 2/(1+exp(K·β))`（β=速度与障碍方向夹角，冲着障碍飞罚更重 `:412-413`）；`launch/planner_exp.xml:137-139` dist_0 0.5 / dist_1 1.0 / vel_min 0.1；`plan_manage/src/adaptive_replan_fsm.cpp:42` 20 Hz 安全检查 | 速度方向相关的软惩罚 |
| Assistive-Racing | `plan_assist/src/risk_monitor.cpp:71-120` SafetyCheck：从当前状态按 risk_acc_max 生成**紧急停轨迹**并查碰撞，撞=当前状态不可接受；`:124-158` VelvecCheck 速度与前向夹角余弦阈值；`plan_manage/config/planner_params.yaml:85-86` 0.60/0.80 双阈值滞回 | 必撞状态剪枝 |
| GPA-Teleoperation | `assisted_planner/src/planner/plan_manage/src/ego_replan_fsm.cpp:121-175` intentionCallback（意图向量→机体系局部目标 `:163-167`）；`traj_opt/src/poly_traj_optimizer.cpp:1727-1750` safeVelocityGradCostV（速度方向与机头夹角余弦 <0.85 即 ~32° 外罚）| 机头-速度耦合约束（我们的 FORWARD 天然沿机头） |

## 2. 借鉴清单（每条：他们怎么做 → 我们对应 → 最小改动 → 预期 → 怎么验）

B1 **运动原语 kinodynamic A***
- 他们：`kinodynamic_astar.cpp:141-262` 离散输入×时长 → stateTransit → 栅格剪枝 → g=(‖u‖²+w_time)τ。
- 我们：`sim/tools/teacher.py:230-310 astar_path` 2D 格点 A*，无速度；`:311-380 plan_action_global` 纯追踪 lookahead 0.7 m、tol 25°。
- 改动：`sim/tools/teacher_kino.py _search()`：节点=(x,y,yaw,vx,vy,yaw_rate)，4 个动作=4 条原语，每条 = 0.1 s 悬停（上一条过期）+ 0.9 s 动作，按一阶响应积分；剪枝格 0.1 m × 10°；g=每步 1 s + 软余量 + 换动作罚；视野 8 步；展开上限 1500 → 取启发最小节点（Fast-Planner run-out 同一处理）。
- 预期：动作序列直接可执行，不需要"几何路径→航向→动作"的两次近似。
- 验：`test_teacher_kino.py` 门柱闭环 x≈−0.05 即开始转、经柱 |y|=0.72、min 0.48 m；20 seed 闭环全到 x≥5.5。

B2 **逐点碰撞检查 + 载具包络**
- 他们：`kinodynamic_astar.cpp:215-222` 原语上 check_num_ 个点查膨胀占据；TGK `pos_checker.cpp:25-97` 检查点沿速度方向伸长。
- 我们：`teacher.py:137 ray_clear` 只查正前 0.4 m 射线，半宽 X3 0.245/余量 0.36。
- 改动：`advance()` 每 0.05 s（1.5 cm）查一次 0.05 m 距离栅格，阈值 = `e1_mini.json collision_envelope 0.102` + 硬余量 0.08；速度方向伸长不用做（原语本身就是沿速度积分的点列）。
- 预期：换载具只改 JSON；碰撞判据与 compare 的 E1 包络一致。
- 验：`test_teacher_kino.py`「e1_mini.json 包络半径 0.102」「0.45 m 外 0.3 m/s 逼近 → 非 FORWARD」。

B3 **必撞状态剪枝（紧急停轨迹）**
- 他们：Assistive-Racing `risk_monitor.cpp:71-120` 从当前状态生成急停轨迹，撞则当前状态不可接受；ego `ego_replan_fsm.cpp:403-406` <0.8 s 撞 → EMERGENCY_STOP。
- 我们：没有；旧教师假设每步都能停。
- 改动：`_search()` 每个子节点再 `advance(end, SAFE, 1.0 s)`，停不住就丢弃；根节点停不住 → `plan_action` 返回 SAFE（fail closed）。
- 预期：教师从不把学生领进「悬停也撞」的状态；标签里 SAFE 只在真必要时出现。
- 验：「必撞状态 → SAFE」「停靠：闭环停在圈里 v=0.000」。

B4 **软余量带（距离代价）**
- 他们：ego `bspline_optimizer.cpp:342-380` distance<clearance 二次罚；EVA `high_mpcc_optimizer.cpp:402-413` 还乘速度方向因子。
- 我们：旧教师只有硬 0.5 m。
- 改动：`g += W_SOFT·max(0, 0.5 − min_clear)²`，W_SOFT=10（代价单位秒：0.3 m 处每步 +0.4 s）。EVA 的方向因子没加——FORWARD 速度总沿机头，方向信息已在原语里。
- 调参事实：0.35/4.0 时门柱场景拖到 0.3 m 才转（1 s/步的时间代价压过软罚），改 0.5/10 后 x≈−0.05 就转。
- 验：门柱 min 0.48；20 seed 均最小净空 0.43–0.47（旧全局 0.43–0.48，持平）。

B5 **几何最短路做启发（A* 引导）**
- 他们：ego `initControlPoints:67-79` 用 A* 路径给优化器方向；Fast-Planner 用庞特里亚金闭式解。
- 我们：旧教师就是几何 A* 本身。
- 改动：目标根 Dijkstra 距离场（0.05 m 格，8 邻域禁斜穿角），h = d/0.277 m + |Δyaw|/27.7° 步（用一窗实际位移/转角而不是 0.3 m/s、0.6 rad/s，因为窗口前 0.1 s 悬停且有 τ）。闭式庞特里亚金不适用：我们的输入是 4 个离散动作不是连续加速度。
- 预期：不走死胡同，热缓存单步 10–40 ms。
- 验：「热缓存单步 <0.5 s」实测 max 0.038 s；建场 0.6 s/场景一次。

B6 **换动作惩罚 / 偏航限幅**
- 他们：EGO-v2 `traj_server.cpp:77-100` 限偏航率/加速度；Assistive `VelvecCheck` 双阈值滞回。
- 我们：`teacher.py:316` 注释已记录 tol<半步会左右过冲；局部版没有任何抑制。
- 改动：`W_SWITCH=0.15 s` 换动作罚；加上 B1 的多步前瞻，振荡自然消失。
- 验：compare 左右换向次数：kino 0 / 旧全局 0–1 / 旧局部 142（名义）、911（τ×1.5）。

B7 **停靠按「悬停后停点」判定 + 刹车距离规则**
- 他们：EGO-v2 `planner_manager.cpp:359-361` 距终点 < v²/(2a) 时目标速度置零；TGK `fsm.cpp:262-305` 从指令落地时的未来状态起规划。
- 我们：`plan_action_global` 进半径才 SAFE，滑行由载具吃掉。
- 改动：stop 模式下到达判定用节点的 `stop_state`（悬停 1 s 后的位置），提前一步刹；「上一条过期→悬停 0.1 s→新动作」直接写进原语，等价于 TGK 的未来状态起规划。
- 验：「停靠：闭环停在圈里且基本停住 end=(1.72,0.00) v=0.000」；l1_stop 20 seed 三组 plant 100% 停进 0.5 m 圈（判据 SAFE 且 |v|<0.05 m/s）。

B8 **终点贴障碍挪位**
- 他们：TGK `krrtplanner.cpp:126-149` 终点 20 方向各挪 0.2 m；ego `dyn_a_star.cpp:91-118`。
- 改动：Dijkstra 根若在不自由格，挪到曼哈顿最近的自由格（`_Field._dijkstra`）。l1 waypoint 离柱面 0.6 m，当前不会触发；保留是防 waypoint 规则改。

没借的：ESDF-free rebound（我们没有连续轨迹优化）；B 样条/MINCO 轨迹类（动作离散、hold 1 s，无需连续参数化）；EVA 的速度方向因子（见 B4）；GPA 的 32° 机头-速度约束（原语天然满足）。

## 3. 原型：`sim/tools/teacher_kino.py`

- 接口：`plan_action(x, y, yaw_deg, pillars, waypoint=None, stop_at_waypoint=False, stop_radius=0.5, vx=0, vy=0, yaw_rate=0, prev_action=None) -> 0..3`，前 7 个参数与 `plan_action_global` 同名同义，可直接替换 `collect.py:779`；不传速度就按静止（旧口径）。
- 状态是 (x, y, yaw, vx, vy, yaw_rate) 而不是任务单写的 (x, y, yaw, v)：滑行动量在世界系、不随机头转（`collect.py CfStab` 同一约定），标量 v 表示不了「边转边滑」。
- 参数：`e1_mini.json` 的 `collision_envelope.radius_from_center=0.102`、`executor.forward_speed_mps=0.3`、`yaw_rate_rps=0.6`、`camera_latency=0.10`；三个时间常数 τ_up 0.313 / τ_down 0.393 / τ_yaw 0.091 由 SITL 实测折算（0.72 s 到 90% → τ=0.72/ln10；滑行 0.118 m ÷ 0.3 m/s；0.21 s 到 90%），标了 `shortcut:` 等 JSON 加字段后改读文件。
- 配置区在文件顶部（STEP_S、τ、CLEAR_HARD/SOFT、W_SOFT/W_SWITCH、GRID、PRUNE、HORIZON、MAX_EXPAND…），每项注释写了为什么是这个数。
- 自检：`python sim/tools/teacher_kino.py`（8 项冒烟）；单测 `sim/tools/test_teacher_kino.py`（70 项，18 s，`tools/check.sh` 的 `test_*.py` 循环会自动跑）。

## 4. 新旧教师离线对比（`sim/tools/compare_teachers.py`）

被控对象：独立的欧拉积分器（dt 0.01，与教师内部 0.05 s 梯形积分不同），世界系速度一阶跟踪「机体前向设定点」，偏航率一阶跟踪，指令延迟期间执行上一条已过期=悬停。碰撞：中心到柱面 < 0.102 m。三组常数：nominal (0.313, 0.393, 0.091, 0.10 s)；sluggish τ×1.5 延迟 0.20；heavy τ×2.5 延迟 0.30（滑行 0.29 m，接近 cf_stab 的 0.34）。旧教师全部用 X3 常量原样跑（不调参，就是"直接换载具"）。

### 4.1 corridor（出生 (−0.5,0) → (6,0) 半径 1.5 m，到圈即到达，上限 60 步）

| plant | teacher | 到达 | 碰撞 | 超时 | 到达均步 | 均最小净空 m | 左右换向 | SAFE/FWD/L/R | ms/步 |
|---|---|---|---|---|---|---|---|---|---|
| nominal | astar_x3（旧全局） | 1.00 | 0 | 0 | 22.1 | 0.48 | 0 | .00/.91/.05/.05 | 14 |
| nominal | local_x3（旧局部） | 0.85 | 0 | 0.15 | 26.1 | 0.42 | 142 | .00/.58/.21/.21 | 14 |
| nominal | predictive_cfstab | 0.00 | 0 | 1.00 | – | 0.66 | 7 | .90/.07/.02/.01 | 3 |
| nominal | **kino_e1** | 1.00 | 0 | 0 | 22.0 | 0.45 | 0 | .00/.91/.05/.05 | 35 |
| sluggish | astar_x3 | 1.00 | 0 | 0 | 25.2 | 0.43 | 0 | .00/.88/.06/.06 | 14 |
| sluggish | local_x3 | 0.10 | 0 | 0.90 | 26.0 | 0.44 | 911 | .00/.14/.43/.43 | 13 |
| sluggish | predictive_cfstab | 0.00 | 0 | 1.00 | – | 0.72 | 0 | .83/.12/.02/.03 | 3 |
| sluggish | **kino_e1** | 1.00 | 0 | 0 | 24.3 | 0.47 | 0 | .00/.92/.04/.04 | 34 |
| heavy | astar_x3 | 1.00 | 0 | 0 | 28.9 | 0.45 | 0 | .00/.89/.06/.06 | 14 |
| heavy | local_x3 | 1.00 | 0 | 0 | 27.0 | 0.44 | 0 | .00/.93/.04/.04 | 15 |
| heavy | predictive_cfstab | 0.00 | 0 | 1.00 | – | 0.49 | 1 | .56/.23/.10/.11 | 3 |
| heavy | **kino_e1** | 1.00 | 0 | 0 | 27.4 | 0.43 | 0 | .00/.93/.04/.04 | 32 |

### 4.2 l1_stop（l1.v4 随机出生 + 柱外侧 waypoint，须 SAFE 且 |v|<0.05 m/s 停进 0.5 m 圈，上限 40 步；local_x3 不支持 waypoint 不跑）

| plant | teacher | 到达 | 碰撞 | 超时 | 到达均步 | 均最小净空 m | 左右换向 | SAFE/FWD/L/R | ms/步 |
|---|---|---|---|---|---|---|---|---|---|
| nominal | astar_x3 | 1.00 | 0 | 0 | 15.2 | 0.50 | 0 | .07/.75/.09/.10 | 5 |
| nominal | predictive_cfstab | 0.00 | 0 | 1.00 | – | 0.54 | 4 | .62/.22/.07/.08 | 3 |
| nominal | **kino_e1** | 1.00 | 0 | 0 | 14.1 | 0.38 | 0 | .07/.82/.06/.05 | 45 |
| sluggish | astar_x3 | 1.00 | 0 | 0 | 17.2 | 0.50 | 0 | .10/.72/.08/.09 | 5 |
| sluggish | **kino_e1** | 1.00 | 0 | 0 | 16.9 | 0.37 | 0 | .12/.76/.06/.06 | 39 |
| heavy | astar_x3 | 1.00 | 0 | 0 | 19.4 | 0.47 | 1 | .09/.73/.09/.09 | 5 |
| heavy | **kino_e1** | 1.00 | 0 | 0 | 19.2 | 0.35 | 0 | .10/.76/.06/.07 | 36 |

l1_stop 里 kino 的最小净空更低（0.35–0.38 vs 0.47–0.50）是几何：waypoint 离柱面 0.6 m，0.5 m 停靠圈的近侧离柱面只有 0.1 m，kino 走直线最短停进圈的近侧（min 0.23 m，仍在硬余量 0.18 m 外）；旧教师因为 0.5 m 余量绕远一点。要更保守把 `CLEAR_SOFT` 调大即可。

### 4.3 0.6 m/s 变体（corridor，`UAV_RGB_E1_PARAMS` 指到 forward_speed 0.6 的 JSON 副本，教师和被控对象同时换）

| plant | teacher | 到达 | 碰撞 | 超时 | 到达均步 | 均最小净空 m | 左右换向 | SAFE/FWD/L/R |
|---|---|---|---|---|---|---|---|---|
| nominal | astar_x3 | 1.00 | 0 | 0 | 14.0 | 0.48 | 0 | .00/.79/.10/.10 |
| nominal | local_x3 | 0.75 | 0 | 0.25 | 12.1 | 0.44 | 269 | .00/.34/.33/.32 |
| nominal | predictive_cfstab | 0.00 | **0.25** | 0.75 | – | 0.41 | 3 | .72/.12/.10/.06 |
| nominal | **kino_e1** | 1.00 | 0 | 0 | 12.3 | 0.47 | 0 | .00/.84/.08/.08 |
| sluggish | astar_x3 | 1.00 | 0 | 0 | 14.9 | 0.44 | 0 | .00/.81/.10/.10 |
| sluggish | **kino_e1** | 1.00 | 0 | 0 | 14.0 | 0.53 | 0 | .00/.86/.07/.07 |
| heavy | astar_x3 | **0.25** | 0 | 0.75 | 19.4 | 0.37 | 288 | .28/.18/.27/.27 |
| heavy | local_x3 | 1.00 | 0 | 0 | 15.0 | 0.44 | 0 | .00/.87/.07/.07 |
| heavy | **kino_e1** | 1.00 | 0 | 0 | 16.5 | 0.39 | 0 | .00/.79/.10/.10 |

0.6 m/s + 滑行 0.59 m（heavy）时旧全局教师只剩 25% 到达、288 次左右换向、28% 的步在 SAFE——它的 0.4 m 前向射线和「能停住」假设在这里失效；kino 在三组 plant 下仍 100%/0 碰撞。这就是动力学感知教师的收益出现的工况：**速度 ≥0.6 m/s 或滑行距离接近缝宽的一半**。E1 当前 0.3 m/s 不在这个区间。

### 4.4 读数要点
- 20 个 test seed 的柱阵只有约 6 种不同布局（`sample_pillars` 的车道/间距约束很紧），每种都留有一条直线走廊，所以 corridor 对任何能转弯的规划器都容易；这是「旧全局教师没坏」的场景原因。
- predictive_cfstab 的 0 到达是它自己的距离场问题（§0.2），不是对比脚本接错：已单独验证它在出生点返回 FORWARD，到 x≈0.9 后因前方整条带 60 分而死锁。
- 规划耗时：kino 35–45 ms/步（纯 Python，含每场景一次 0.6 s 建场摊销），collect 一窗 1 s 仿真时间，够用。

## 5. 没做什么 + 原因

- 没改 `collect.py` 接新教师、没动主仓库、没推送：任务单只要原型 + 离线对比；且 §0 的结论是当前不必切。
- 没加 EVA 的速度方向惩罚、GPA 的机头-速度约束：原语天然满足（FORWARD 沿机头，YAW 不平移）。
- 没把 τ 写进 `e1_mini.json`：那是 SITL-core 的文件/schema，跨 agent 改动需主会话决定；代码里标了 `shortcut:`。
- 没跑 Gazebo/SITL（任务约束）。被控对象是「一阶响应 + 延迟」的近似，没有高度、风、位姿噪声。
- 没验证 kino_sampling_with_regional_opti 的实现细节：结构与 TGK 相同，当前任务无新借点。
- 没把 compare 的 PNG 入库（数据盘保存；仓库只放脚本）。

## 6. 接真 Gazebo/SITL 还要改哪里

1. `collect.py:779` 改调 `teacher_kino.plan_action(..., vx, vy, yaw_rate, prev_action)`：vx/vy 要从 `pose_before` 相邻两帧差分或 odom 速度拿；现在 `pose_before` 只有位姿。
2. hold 若不是 1.0 s，改 `teacher_kino.STEP_S`（或从 `--hold` 传入）；`camera_latency` 若实测变了只改 JSON。
3. τ_up/τ_down/τ_yaw：SITL-core 在 `e1_mini.json` 加 `executor_response`（或类似字段）后，`_load_params` 改读文件并删掉 `shortcut:`；同时 `compare_teachers.PLANTS` 的 nominal 行一起换。
4. 偏航跟踪：原语假设 YAW 时水平速度设定点为 0（原地转）。E1 的 executor 若在 YAW 时保留前向速度，`advance()` 里 `v_cmd` 的分支要改成和 executor 一致。
5. 标签口径：`teacher_version` 在 `collect.py:1529` 处分支，新增 `"uavrgb.teacher_kino.v1"`；preprocess 的 split/几何隔离门禁不受影响（几何 seed 不变）。
6. 验收：先在 SITL 上用 `plan_sequence` 回放 kino 的预测轨迹 vs 实测轨迹，看一窗位移/转角误差是否 <10%；误差大先修 τ，再谈切换。
