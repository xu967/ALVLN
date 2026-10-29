# 参考规划器研究：EGO-Planner / EGO-Planner-v2 / GCOPTER / Implicit-SDF-Planner

- 时间：2026-10-29 00:38
- 范围：只读 `/root/claude-code-backup-20261028/code/idea/repos/` 下四个仓库；对照 `sim/tools/teacher.py`、`sim/tools/collect.py`、`sim/worlds/uav_rgb.sdf`、`sim/tools/scene_gen.py`
- 结论：完成（只读研究与建议；未改任何代码，未运行/编译这些仓库）
- 许可证前提：`ego-planner`、`EGO-Planner-v2`、`Implicit-SDF-Planner` 均为 **GPL-3.0**，只能读思想、不能复制代码；`GCOPTER` 为 **MIT**。下文每条建议都标注「算法思想（可自由实现）」或「具体实现（GPL/MIT，是否可抄）」。

## 0. 结论先行：最值得做的 3 条

| # | 建议 | 改 `teacher.py` 哪一段 | 预计代码量 | 许可证 |
|---|------|----------------------|-----------|--------|
| 1 | 三条竖带 → **列级深度剖面 + 连续可通行区间（1D 自由走廊）** | 替换 `_strip_min`（24–40 行）与决策体（65–98 行） | 约 60–75 行（含新 helper） | 算法思想，自行实现；不抄 GPL 代码 |
| 2 | **机体宽度间隙判据 + 两级安全距离**（硬 0.8 m / 软 1.5 m；0.9 m 缝判据） | 常量区（20–21 行）+ 决策体里的区间过滤 | 约 15–20 行 | 算法思想 + 公开参数（数字是事实） |
| 3 | **最小转向的可达间隙评分**（饱和间隙收益 + 转向惩罚 + 滞回） | 决策体（65–98 行）内部 | 约 25–35 行 | 通用 receding-horizon 思想 |

三条合并落地时共用同一个决策体，净增约 **110–130 行**；`teacher.py` 会从约 140 行涨到 250–280 行，仍在「几百行」范围内。

---

## 1. 四个仓库的定位与算法拆解

### 1.1 `ego-planner`（GPL-3.0）—— ESDF-free 的梯度局部规划器

**一句话定位**：不建 ESDF，用「占据栅格 + A* 初始路径 + B 样条优化 + 回弹方向（rebound）」做局部避障；README 第 36–38 行明确写了 *ESDF-free*，代码里也 grep 不到 `esdf|edt`。

**传感器 → 决策链路**（关键文件/函数）：

1. **深度图 → 占据栅格**：`plan_env/src/grid_map.cpp:201 GridMap::projectDepthImage()` 用 `cx,cy,fx,fy` 把深度像素投到相机系再转到世界系；先做近/远距与容差过滤（`depth_filter_mindist=0.2`、`maxdist=5.0`、`tolerance=0.15`、`margin=1`、`skip_pixel=2`）。
2. **射线投射建图**：`grid_map.cpp:327 GridMap::raycastProcess()` 从相机向每个点投射，按概率 hit/miss 更新（`p_hit=0.65`、`p_miss=0.35`、钳位 `[0.12,0.90]`、`p_occ=0.80`），只更新局部范围（`local_update_range=5.5 m`）；`clearAndInflateLocalMap()` 按 `ceil(0.099/0.1)=1` 个体素在 x/y（z 方向 1 层）膨胀。查询用 `getInflateOccupancy()`。
3. **全局/初始路径**：`path_searching/src/dyn_a_star.cpp` 的 `AStarPathSearcher::AstarSearch()` 在膨胀栅格上做 3D A*。
4. **局部目标怎么选**：`plan_manage/src/ego_replan_fsm.cpp:504 EGOReplanFSM::getLocalTarget()` 沿全局路径向前走，取**第一个距当前位置 ≥ planning_horizon 的点**作为局部目标（launch 里 `planning_horizon=7.5 m`，代码默认 5.0 m），并把该点路径速度作为终端速度；接近终点时终端速度置零。本质是「固定前视距离的滚动目标」。
5. **碰撞代价怎么定义**：`bspline_opt/src/bspline_optimizer.cpp:343 BsplineOptimizer::calcDistanceCostRebound()`。
   - 对每个控制点，先由 `check_collision_and_rebound()`（同文件 657 行）找出它在 A* 路径上的交点，再沿「交点 → 控制点」方向按栅格分辨率采样，找到第一个占据点后回退一个分辨率，得到障碍表面附近的**基点** `base_point`；`direction` 是「控制点 → 交点」的单位向量。
   - 距离 `dist = (p - base_point)·direction`，误差 `dist_err = clearance - dist`；`dist_err < 0` 不罚；`0 ≤ dist_err < clearance` 用三次 `dist_err³`；再远用二次多项式续接，保证 C1。
   - 参数：`optimization/dist0 = 0.5 m`（clearance）、`lambda_collision = 0.5`、栅格 `resolution=0.1`。注意 `obstacles_inflation=0.099` 只是 1 个体素的地图膨胀，**真正的安全间距由优化器的 `dist0=0.5 m` 提供**，不是靠地图膨胀。
   - 不确定点：这个快照里 `direction` 的符号约定我没有完全核清（`dist` 的几何正负与论文里的 `d_ij` 需要再对一次）；但「取最近自由方向、按 clearance 做分段惩罚」的思想是清楚的，我们只借思想。
6. **失败回退**：`plan_manage/include/plan_manage/ego_replan_fsm.h:31` 的状态机 `INIT / WAIT_TARGET / GEN_NEW_TRAJ / REPLAN_TRAJ / EXEC_TRAJ / EMERGENCY_STOP`；`ego_replan_fsm.cpp:403` 若当前轨迹在 `emergency_time_`（代码默认 1.0 s，注释写 0.8 s）内会撞，直接 `EMERGENCY_STOP`，否则重规划；`flag_escape_emergency_` 防止重复触发。

**依赖**：ROS1/catkin、Eigen、PCL、OpenCV/cv_bridge、可选 CUDA（`uav_simulator/local_sensing`）。我们只能接受 Eigen，ROS/PCL 都不行。

---

### 1.2 `EGO-Planner-v2`（GPL-3.0）——  swarm/formation 版 EGO

**一句话定位**：把 EGO 的 B 样条换成 MINCO/GCOPTER 风格的多项式轨迹优化，并加入集群间 clearance 与编队代价；感知与建图仍是占据栅格，**同样没有 ESDF**。

**对本项目有用的部分**：

- `traj_opt/src/poly_traj_optimizer.cpp:1424 PolyTrajOptimizer::obstacleGradCostP()` 用了**两级障碍间距**：
  - 硬层 `obs_clearance_ = 0.1 m`，违反时罚 `dist_err³`；
  - 软层 `obs_clearance_soft_ = 0.5 m`，违反时罚一个平滑 barrier `r²(√(1+dist_err_soft²/r²)-1)`（`r=0.05`），梯度连续；
  - 集群间距 `swarm_clearance_ = 0.5 m`（`plan_manage/launch/advanced_param.xml:128–130`）。
- 这层「硬安全 + 软舒适」的双阈值，比单一 `turn_dist` 更适合我们：硬阈值决定「绝对不能进」，软阈值只影响「更愿意往哪走」。
- 其余（MINCO、编队、drone_detect、swarm_bridge）对本项目不适用。

**依赖**：ROS1、Eigen、PCL、OpenCV；GPL-3.0。

---

### 1.3 `GCOPTER`（MIT）—— 安全飞行走廊（SFC）+ 多项式轨迹优化

**一句话定位**：点云体素地图 → RRT* 全局路径 → **FIRI 最大体积凸多面体走廊** → MINCO 轨迹优化（含动力学约束）；核心是「先把自由空间包成凸走廊，再让轨迹待在走廊里」。

**算法链路**：

1. **体素地图与膨胀**：`gcopter/include/gcopter/voxel_map.hpp` 的 `VoxelMap` 从点云 `setOccupied()`；`global_planning.cpp:137` 调 `dilate(ceil(DilateRadius / VoxelWidth))`，配置 `DilateRadius=0.5 m`、`VoxelWidth=0.25 m` → 膨胀 2 个体素。`query()` 对**地图外返回 true（当作占据）**，是明确的 fail-closed。
2. **全局路径**：`sfc_gen::planPath()`（`sfc_gen.hpp:46`）用 OMPL 的 Informed RRT*，在膨胀后的地图上找无碰路径。
3. **走廊生成**：`sfc_gen::convexCover()`（`sfc_gen.hpp:116`）沿路径每 `progress=7.0 m` 切一段，先以路径段为中心、半宽 `range=3.0 m` 造一个初始盒子，收集盒内障碍表面点，再调 `firi::firi()`（`firi.hpp:267`，默认 4 次迭代）把盒子「膨胀」成最大体积凸多面体；相邻多面体若重叠不足会补一个过渡多面体；最后 `shortCut()` 合并冗余走廊。**走廊内的点全部是膨胀后的自由空间**，所以轨迹只要待在走廊里就无碰。
4. **碰撞怎么进优化**：`gcopter.hpp:410–414` 对每个走廊半空间算 `outerNormal·pos + d`，违反量做惩罚；它**没有**用「到障碍表面的距离」做代价，安全完全由走廊的构造保证。
5. **安全距离/间隙的具体数值**：`DilateRadius=0.5 m` 是唯一的安全半径参数；走廊 `range=3.0 m`、`progress=7.0 m` 是走廊尺寸，不是安全距离。

**依赖**：Eigen、OMPL（仅全局 RRT* 例子用）、ROS1（示例节点）、仓库自带的 L-BFGS/SDLP 头文件。MIT 意味着代码在法律上可抄，但 OMPL 和 3D 轨迹优化对我们没有价值，**不建议抄任何代码**。

---

### 1.4 `Implicit-SDF-Planner`（GPL-3.0）—— 真正带 SDF 梯度的那一个

**一句话定位**：用局部占据地图建 **ESDF（实际查询是正距离/无符号距离）**，前端 A* + 扫掠体（swept volume）做可通行检查，后端用 `safety_hor - SDF` 的 hinge 代价把轨迹推离障碍。

**ESDF 怎么建**（`map_manager/src/Gridmap3D.cpp`）：

- `GridMap3D::generateESDF3d()`（361 行）：把 3D 占据栅格做**平方欧氏距离变换**。先对每个 (x,y) 沿 z 做一次 1D DT，再对每个 (x,z) 沿 y 做一次，最后对每个 (y,z) 沿 x 做一次；每一步用 `fillESDF()`（495 行）的 Felzenszwalb 抛物线包络算法，复杂度 O(N)。正向 DT 从占据格出发，得到「到最近占据格」的正距离；负向 DT 从自由格出发，代码也算了，但**查询时没用**。
- `getGridSDFValue()`（535 行）只返回 `grid_esdf`，即**正距离**；障碍内部距离为 0。所谓 signed 的负半支只写进了未使用的 `grid_map_buffer_all`。所以它的梯度只在障碍外部有效，进入障碍后梯度为 0。这是一个实现上的重要事实，别把它当成完整 signed SDF。
- 查询：`GridMap3D.h:155 getSDFValueWithGrad()` 做三线性插值，并解析求插值函数对 x/y/z 的偏导，得到梯度；梯度指向「距离增大」的方向，即远离障碍。
- 分辨率：配置里 `occupancy_resolution: 1.0`（`plan_manager/config/yourdemo.yaml:55`），这个 demo 是 mesh 缩放的，**单位不一定是米**；不要把这个 1.0 当成推荐分辨率。

**梯度怎么用**：`planner_algorithm/include/planner_algorithm/back_end_optimizer.hpp` 的 `grad_cost_p()` / `grad_cost_p_sw()`：

- 对机体 AABB 内的障碍点查 SDF 和梯度；
- 代价 `smoothedL1(safety_hor - sdf_value, 0.01)`，即 `sdf < safety_hor` 时产生 hinge 惩罚，`safety_hor=0.6 m`（多处配置为 0.6 或 0.866）；
- 梯度沿 SDF 梯度方向反推回轨迹参数，把轨迹推离障碍。这就是「用带符号距离的梯度做避障」的标准做法。

**可通行性/局部目标**：

- 前端 `AstarPathSearcher::AstarGetSucc()`（`front_end_Astar.hpp:197`）26 邻域扩展，条件 = 在地图内 && `!isIndexOccupiedFlate(vi, 0)` && 扫掠体核 `checkKernelValue()`；`isIndexOccupiedFlate` 的越界处理是「越界当占据」（`Gridmap3D.cpp:213` 起）。
- 它没有 EGO 那种「沿全局路径取前视点」的局部目标；目标由 plan_manager 给，属于全局目标。

**依赖**：ROS1、Eigen、PCL、CUDA（`local_sensing/depth_render.cu`）、自定义 swept-volume/LMBM；GPL-3.0。

---

### 1.5 四个仓库的依赖/许可证对照

| 仓库 | 许可证 | 核心依赖 | 我们能接受的部分 |
|------|--------|---------|-----------------|
| ego-planner | GPL-3.0 | ROS1/catkin、Eigen、PCL、OpenCV/cv_bridge、可选 CUDA | Eigen；其余都不行。只取思想 |
| EGO-Planner-v2 | GPL-3.0 | 同上 + swarm 相关 | Eigen；只取思想 |
| GCOPTER | MIT | Eigen、OMPL、ROS1 示例、自带 L-BFGS/SDLP | Eigen；OMPL/ROS 不行。代码法律上可抄但没必要 |
| Implicit-SDF-Planner | GPL-3.0 | ROS1、Eigen、PCL、CUDA、自定义 swept volume | Eigen；只取思想 |

---

## 2. 可直接借鉴的思想（按对本项目的价值排序）

### R1（最高价值）：把「三条竖带」换成「列级自由区间」

- **思想**：GCOPTER 的 SFC 把自由空间表示成「一串凸区域」；Implicit-SDF 的前端把「可通行」定义成逐格查询；EGO 的栅格也是逐体素。降维到 1D 就是：**每一列是一个方向，连续的自由列构成一个可通行区间**，决策只在区间层面做，而不是在三条 20° 宽的带里取最小值。
- **为什么能治缺陷 1**：当前三条带各 20°（320 列里的约 107 列）。一个 1 m 宽的障碍在 1.88 m 处张角约 29.8°（约 159 列），会同时盖满三条带，于是三带最小深度都变成 1.88 m，只能靠阈值绕过。列级剖面下，它只封住中间约 159 列，两侧各剩约 15°，是**两个独立的自由区间**，不会再被平均成「三面都堵」。
- **我们怎么用**：在中间竖直带 `[0.4h, 0.6h)` 内，对每一列取最小有效深度，得到 `prof[x]`；`prof[x] >= HARD` 的连续列是自由区间；每个区间计算 `width_deg`、`center_deg`、左右边界深度 `d_edge`。决策只在这些区间上做。
- **改哪里**：`teacher.py` 的 `_strip_min`（24–40 行）换成 `_column_profile`；决策体（65–98 行）改为区间选择。保留 55–63 行的入参校验与 fail-closed 语义。
- **预计行数**：60–75 行（含 `_free_runs`、区间属性、决策重写）。
- **许可证**：**算法思想（可自由实现）**。不要抄 GCOPTER 的 `convexCover`/`firi`，也不要抄 Implicit-SDF 的 `GridMap3D`；我们只是借用「自由区间/走廊」这个概念。
- **依赖**：无，纯 stdlib。

### R2：机体宽度间隙判据 + 两级安全距离

- **思想**：GCOPTER 用 `DilateRadius=0.5 m` 把障碍按机体半径膨胀；EGO-v2 用「硬 0.1 m / 软 0.5 m」两级 clearance；Implicit-SDF 用 `safety_hor=0.6 m` 的 hinge。共同点是：**安全距离是一个与机体尺寸挂钩的参数，而不是一个拍脑袋的深度阈值**。
- **我们怎么用**：
  - 机体半宽 `HALF_WIDTH=0.32 m`（`uav_rgb.sdf` 注释：旋翼尖半径 0.32 m > 机身半宽 0.235 m；整机旋翼跨度 0.64 m）。
  - 场景生成器保证任意两柱至少 0.9 m 缝（`scene_gen.py:MIN_GAP=0.9`，并注明旋翼跨度约 0.64 m），所以每侧余量 `GAP_MARGIN=0.13 m` 与生成器一致；若深度估计噪声大，可先用 0.15–0.20 m 保守一点。
  - 间隙判据：对区间取左右边界列的深度 `d_edge = min(prof[a-1], prof[b+1])`（边界列是 blocked，深度 < HARD；若区间贴 FOV 边缘，则该侧视为 FAR），物理缝宽近似 `gap_m ≈ 2 * d_edge * sin(width_deg/2)`；要求 `gap_m ≥ 2*(HALF_WIDTH + GAP_MARGIN)`。
  - 等价的角度写法：`width_deg ≥ 2 * atan2(HALF_WIDTH + GAP_MARGIN, d_edge)`（度）。
  - 两级阈值：`HARD=0.8 m`（沿用当前 `emergency`，不可进入；相机到障碍表面 0.8 m，对 0.32 m 旋翼尖仍有约 0.5–0.7 m 余量，取决于相机 0.2 m 前置偏移，和四个仓库的 0.5–0.6 m 表面间距同量级）；`SOFT=1.5 m`（沿用当前 `turn_dist`，舒适前进距离，不满足也不禁止前进，只影响评分）。
- **为什么能治缺陷 3 的一部分**：把「缝宽 vs 机体宽度」显式化后，0.9 m 的缝会被判为可通行、0.5 m 的缝会被判为不可通行；不再出现「三条带都大于某阈值就往前冲」的盲目性。
- **改哪里**：常量区（20–21 行）加 `HFOV_DEG/FAR/HALF_WIDTH/GAP_MARGIN`；决策体里加 `run_gap_m()` 和 `run_body_ok()`。
- **预计行数**：15–20 行。
- **许可证**：**算法思想 + 公开参数**。参数值是配置事实，不构成代码复制；实现自己写。
- **依赖**：无。

### R3：最小转向的可达间隙评分（治「转向后飞走」和「偏航量化」）

- **思想**：EGO 的滚动局部目标 + 重规划，本质是「在每个决策周期选一个可达的局部目标」；我们把它降到 1D 航向空间：**候选航向 = 各个自由区间的中心（以及正前方 0°），评分 = 饱和后的间隙收益 − 转向惩罚，选最大者**。
- **关键公式**：`score = min(d_min, SOFT) - TURN_PENALTY * |center_deg|`。
  - `min(d_min, SOFT)` 把「3 m 空」和「10 m 空」视为一样好，避免为了更空的一侧一路飞走（缺陷 2）。
  - `TURN_PENALTY` 按一个偏航步长标定：当前执行器 `SPEED_YAW=0.6 rad/s`、`hold=0.5 s` → 每步 17.2°；取 `TURN_PENALTY≈0.02 m/deg`，则一次 17° 转向约等价于 0.34 m 的间隙收益。
  - 候选航向只取「区间中心」，天然是当前一步/几步可达的方向；不需要连续转弯半径模型。**注意**：本项目的 `YAW_LEFT/RIGHT` 是原地偏航（`collect.py:make_twist` 里 yaw 动作的线速度为 0），所以不存在「最小转弯半径」；真正的运动学约束是 17.2°/步的偏航量化和 0.15 m/步的前进量。R3 用转向惩罚和区间中心把这两者都考虑进去了。
- **贴障穿行的产生机制**：当前方区间的 `d_min` 在 `[HARD, SOFT)`（例如 1.0 m）而侧向没有明显更好的区间时，`score(forward)=1.0` 会大于「转向去 3 m 外」的 `1.5-0.34=1.16` 吗？不一定；把 `TURN_PENALTY` 调到 0.03 时，17° 转向成本 0.52，侧向得分 0.98 < 1.0，于是**选择前进穿过窄缝**，而不是一路转向最空侧。这正是缺陷 2 想要的困难样本。参数需要实测微调。
- **改哪里**：决策体（65–98 行）内部；候选集合 + 评分 + 选择。`prev_action` 用于滞回：同向转向保持，反向需要超过 `switch_margin` 才切换。
- **预计行数**：25–35 行。
- **许可证**：**通用思想（receding-horizon / sampling-based local planning）**，不来自某个仓库的具体实现。
- **依赖**：无。若要把 `hold` 的变化也纳入，需要 `collect.py` 把 `yaw_step_deg = degrees(SPEED_YAW * hold)` 传给 teacher（一行改动，可选）。

### R4：用「带符号的横向偏移/梯度」代替左右硬比较

- **思想**：EGO 的回弹方向是「从控制点指向最近自由方向」；Implicit-SDF 的 SDF 梯度是「距离增大方向」。在 1D 深度剖面里，这就是 `d(depth)/d(bearing)` 的符号：朝深度增大的一侧转。
- **我们怎么用**：如果不做 R1/R3，最小改法是把 `left/right` 两个最小深度比较换成「自由区间中心的带符号方位角」`steer_deg`；`steer_deg > deadband` → 左转，`< -deadband` → 右转，否则前进。deadband 取 8–10°（约半个偏航步长）。
- **价值**：R1/R3 已经隐含了这个信息（区间中心就是带符号偏移），所以 R4 主要作为**平替方案**：如果只想改 20 行，不想引入区间，就先做 R4。
- **改哪里**：65–98 行的左右判断。
- **预计行数**：10–15 行。
- **许可证**：**算法思想**。
- **依赖**：无。

### R5：失败回退与未知区 fail-closed

- **思想**：EGO 的 `EMERGENCY_STOP` + `emergency_time`（1.0 s）；GCOPTER 的 `VoxelMap::query()` 对地图外返回 true（占据）；Implicit-SDF 的 `isIndexOccupied` 越界返回 true。
- **我们怎么用**（大部分已具备，补三点）：
  1. 列剖面里**任何无效读数（0/NaN/负）都当 blocked**，不要当空旷；整列无有效样本 = blocked。
  2. 没有自由区间 → `SAFE_NEUTRAL`；有区间但全部不满足机体宽度 → 也 `SAFE_NEUTRAL`（或朝最宽区间原地转，二选一，建议先停）。
  3. 可选的时间到碰撞：`ttc = forward_depth / SPEED_FWD`；`ttc < 1.0 s` 时 `SAFE_NEUTRAL`。当前 0.8 m / 0.3 m/s = 2.7 s，已经很保守，TTC 规则不是必需。
- **改哪里**：区间扫描和最终 fallback 分支。
- **预计行数**：10–15 行。
- **许可证**：**算法思想**。
- **依赖**：无。

### R6：参数外置 + 可视化调试

- **思想**：EGO 用 launch XML 把 `dist0/planning_horizon/...` 全部外置；GCOPTER 用 YAML + `Config` 结构体；Implicit-SDF 用 YAML + `Config`；三个仓库都有可视化（EGO 的 `PlanningVisualization`、GCOPTER 的 `misc/visualizer.hpp`、Implicit-SDF 的 `debug_assistant`）。
- **我们怎么用**（不要上配置框架，保持 stdlib）：
  - 在 `teacher.py` 顶部放一个 `PARAMS = {...}` 字典（HARD/SOFT/HALF_WIDTH/GAP_MARGIN/TURN_PENALTY/DEADBAND），`expert_action` 用 `**PARAMS` 的默认值；允许调用方传 `params=` 覆盖。约 15 行。
  - 给 `teacher.py` 加一个可选的 `debug` 输出（返回 `prof/runs/chosen` 的 dict，或单独写一个 `sim/tools/teacher_debug.py` 把剖面画成 PGM/SVG）。约 40–60 行，非必需。
- **改哪里**：文件顶部 + 可选新工具。
- **预计行数**：15 行（参数）；40–60 行（可视化，可选）。
- **许可证**：**工程做法**，无代码复制。
- **依赖**：无。

---

## 3. 明确不适用的部分（以及为什么）

| 部分 | 为什么不用 |
|------|-----------|
| **3D ESDF + Felzenszwalb EDT + 三线性梯度**（Implicit-SDF） | 需要 3D 占据/点云地图和 O(N) 地图更新；我们的输入是一张 320×240 深度图，输出是 4 个离散动作，每 0.5 s 建一次 3D ESDF 纯属浪费。1D 深度剖面已经保留了「每个方位到障碍的距离」这一决策所需信息。若以后有 C++ 局部地图，2D EDT 可以用来补深度空洞，但不是现在。 |
| **B 样条 / MINCO / L-BFGS / 动力学可行性**（EGO、GCOPTER） | 我们没有连续轨迹输出，动作是离散的；速度/加速度约束由执行器和飞控处理，teacher 不需要也不应该建模。 |
| **A* / RRT* / SFC / FIRI**（EGO、GCOPTER） | 需要全局地图、目标和 3D 碰撞检查；teacher 是纯反应式、无地图、无全局目标。走廊的「自由区间」思想可以降维借用（R1），但算法本身不用。 |
| **Swarm / formation / drone_detect**（EGO-v2） | 单机场景，没有机间避让。 |
| **扫掠体姿态核 / 四元数梯度**（Implicit-SDF） | 没有姿态控制输出；机体相对航向固定。R2 的机体半宽就是它在 2D 的降维。 |
| **ROS1 / PCL / CUDA / OMPL 依赖** | 项目禁用 ROS 2，本机也没有 ROS；ROS1 同样不应引入。PCL/CUDA/OMPL 都太重。teacher 保持纯 stdlib。 |
| **3D 局部栅格 + 射线投射建图**（EGO GridMap） | 深度图本身就是距离像；投影到 3D 栅格再查回来会丢掉角度分辨率且更贵。唯一潜在价值是「转头后记住已离开 FOV 的障碍」，但那需要 odometry/偏航积分进 teacher，属于更大的改动，本次不做。 |

---

## 4. 对 `teacher.py` 的具体改法

### 4.1 现状定位

- 20–21 行：`BAND_LO/BAND_HI`，只取中间竖直带，这个选择本身是对的：相机光轴水平、巡航高度 2.1 m、障碍是 3 m 高的柱子，会挡住水平飞行走廊；宽一点会引入地面/天空噪声。
- 24–40 行：`_strip_min`，把宽度三等分，每份取最小深度 —— **缺陷 1 的根因**。
- 65–70 行：三带最小值；75–76 行：前方 ≥1.5 m 前进；84–85 行：三面 <0.8 m 才停；86–98 行：左右可转性与滞回。
- 102–215 行：自测；其中若干 case 的期望会随新逻辑变化（见 4.5）。

### 4.2 目标算法（伪代码，非 GPL 代码）

```text
常量（建议初值）:
  HFOV_DEG = 60.0          # uav_rgb.sdf: horizontal_fov 1.0471975512
  FAR      = 20.0          # depth clip far
  HARD     = 0.8           # 沿用 emergency；低于它视为 blocked
  SOFT     = 1.5           # 沿用 turn_dist；舒适前进距离
  HALF_WIDTH = 0.32        # 旋翼尖半径（uav_rgb.sdf 注释）
  GAP_MARGIN = 0.13        # 与 scene_gen MIN_GAP=0.9 / 旋翼跨度 0.64 对齐
  TURN_PENALTY = 0.02      # m/deg；17.2° 一步 ≈ 0.34 m
  DEADBAND_DEG = 9.0       # 约半个偏航步长
  SWITCH_MARGIN = 0.15     # 评分滞回，m

def expert_action(depth, w, h, prev_action=None, ...):
    # 1) 列剖面：中间带每列最小有效深度；无效/无样本 -> 0（blocked）
    prof = [0.0] * w
    for x in range(w):
        best = inf; found = False
        for y in range(int(0.4*h), int(0.6*h)):
            d = depth[y*w + x]
            if is_valid(d):
                found = True
                best = min(best, d)
        prof[x] = min(best, FAR) if found else 0.0   # 无有效样本 -> blocked

    # 2) 连续自由区间：prof[x] >= HARD
    runs = [(a,b) for maximal a..b where all prof[x] >= HARD]

    # 3) 区间属性（角度统一转弧度）
    for r in runs:
        r.width_deg  = bearing(b+1) - bearing(a)      # bearing(x) 用像素->角度
        r.center_deg = 0.5 * (bearing(a) + bearing(b+1))
        r.d_min      = min(prof[a..b])                 # 区间内最近深度
        r.edge_depth = min( prof[a-1] if a>0 else FAR,
                            prof[b+1] if b+1<w else FAR )
        r.gap_m      = 2 * r.edge_depth * sin(radians(r.width_deg)/2)
        r.body_ok    = r.gap_m >= 2*(HALF_WIDTH + GAP_MARGIN)

    # 4) 正前方是否可走：0° 所在的区间，且 0° 到两侧边界的角度都够
    fwd = the run whose [left,right] contains 0
    need_deg = degrees(atan2(HALF_WIDTH + GAP_MARGIN, fwd.edge_depth))
    fwd_ok = fwd.body_ok and min(0 - fwd.left_deg, fwd.right_deg - 0) >= need_deg
    # 第一阶段保守版：fwd_ok = fwd_ok and fwd.d_min >= SOFT

    # 5) 候选航向与评分
    candidates = []
    if fwd_ok: candidates.append( (0.0, min(fwd.d_min, SOFT), is_forward=True) )
    for r in runs:
        if r contains 0: continue                 # 正前方只由上面的候选代表
        if r.body_ok or r.touches_fov_edge:      # 贴边区间允许转过去看
            candidates.append( (r.center_deg,
                                min(r.d_min, SOFT) - TURN_PENALTY*abs(r.center_deg),
                                is_forward=False) )
    if candidates empty: return SAFE_NEUTRAL

    best = argmax score, 但若 prev_action 是左/右转且同侧候选仍在，
           同侧需要被对侧超过 SWITCH_MARGIN 才切换（滞回）
    # 注意：滞回只在同符号候选之间比较；不要用滞回把不安全的航向强留下来

    # 6) 动作映射
    if best.is_forward: return FORWARD_SLOW
    if abs(best.heading) <= DEADBAND_DEG:
        return FORWARD_SLOW if 0 within best.run else (YAW_LEFT if best.heading >= 0 else YAW_RIGHT)
    return YAW_LEFT if best.heading > 0 else YAW_RIGHT
```

几个容易踩的细节：

- **`fwd.d_min` 用区间内的最小值**；0° 到两侧边界的角度用 `need_deg` 检查，保证不是「缝够宽但正前方偏在缝边缘」。
- **贴 FOV 边缘的区间**：`edge_depth` 在贴边那一侧取 `FAR`，`gap_m` 会很大、`body_ok` 为真；这是有意的——转头是原地动作，先转过去看比直接判定「无路」安全。但**贴边区间不能作为 `fwd_ok`**（正前方 0° 不会在贴边区间里，除非整幅图全空）。
- **无效列**：`prof[x]=0` 会把区间切断，这是 fail-closed；不要用「跳过无效样本」的方式把两个区间连起来。
- **航向角**：`bearing(x)` 建议用 `atan((x+0.5 - w/2) / (w/2) * tan(HFOV/2))`；用线性近似 `(x+0.5-w/2)*HFOV/w` 在 60° FOV 内误差约 1°，也可以接受。
- **滞回**：`prev_action` 只用于「同向保持/反向需超过 margin」，不改变候选集合；这样不会因为滞回把不安全的方向强留下来。
- **参数来源**：`HARD/SOFT` 继续用现有的 `emergency/turn_dist` 形参名，避免破坏 `collect.py` 的调用；新增的参数给默认值即可。

### 4.3 参数建议与依据

| 参数 | 建议值 | 依据 |
|------|--------|------|
| `HALF_WIDTH` | 0.32 m | `uav_rgb.sdf`：旋翼尖半径 0.32 m > 机身半宽 0.235 m；整机跨度 0.64 m |
| `GAP_MARGIN` | 0.13 m（保守可用 0.15–0.20） | `scene_gen.py`：`MIN_GAP=0.9 m`，注释明确按旋翼跨度 0.64 m 留缝；0.9 = 2×(0.32+0.13) |
| `HARD` | 0.8 m | 沿用当前 `emergency`；相机到障碍表面 0.8 m 对 0.32 m 旋翼尖仍有约 0.5–0.7 m 余量（相机前置 0.2 m），与 EGO `dist0=0.5`、GCOPTER `DilateRadius=0.5`、Implicit-SDF `safety_hor=0.6` 同量级 |
| `SOFT` | 1.5 m | 沿用当前 `turn_dist`；只影响评分，不禁止前进 |
| `TURN_PENALTY` | 0.02 m/deg | `SPEED_YAW=0.6 rad/s × hold=0.5 s = 17.2°/步`；17.2° 转向 ≈ 0.34 m 间隙收益 |
| `DEADBAND_DEG` | 9° | 约半个偏航步长，避免「差 1° 就转」 |
| `FAR` | 20 m | `uav_rgb.sdf` 深度相机 `clip/far=20` |

对比参考值（注意单位和场景不同，不要直接照搬）：

- EGO-Planner：`resolution=0.1`、`obstacles_inflation=0.099`（1 体素）、`dist0=0.5 m`、深度过滤 `[0.2, 5.0]`、射线 `[0.1, 4.5]`、`planning_horizon=7.5 m`、`emergency_time=1.0 s`。
- EGO-Planner-v2：硬 `0.1 m` / 软 `0.5 m` / 集群 `0.5 m`。
- GCOPTER：`DilateRadius=0.5 m`、`VoxelWidth=0.25 m`（膨胀 2 体素）、走廊 `progress=7.0 m`、`range=3.0 m`。
- Implicit-SDF：`safety_hor=0.6 m`（部分配置 0.866）、`dilateRadius=0.5`、`voxelWidth=0.15`；`occupancy_resolution=1.0` 是 demo 尺度，**不要当作米**。

### 4.4 分阶段落地建议

1. **第一阶段（只做 R1 + R2）**：列剖面 + 区间 + 机体宽度过滤；保留当前的 `center>=turn_dist → FORWARD` 语义（即区间包含 0° 且 `d_min>=SOFT` 才前进，否则转向）。这一阶段行为变化最小，主要治缺陷 1，先验证不再出现「三带全堵」的假阳性。
2. **第二阶段（加 R3）**：把「`d_min>=SOFT` 才前进」改成「评分选最小转向的可达区间」，允许 `HARD ≤ d_min < SOFT` 的前方区间前进，产生贴障穿行样本。这一步会改变 `teacher.py` 里 `(1.2,1.2,1.2)` 的期望（从 YAW_LEFT 变 FORWARD），需要有意识地更新测试。
3. **第三阶段（可选）**：参数外置 + 可视化 + TTC 紧急停止。

### 4.5 测试改动

`teacher.py` 的 `__main__` 自测需要跟着改：

- **保持不变**：`(5,5,5)→FORWARD`、`(5,0.3,3)→YAW_LEFT`、`(3,0.3,5)→YAW_RIGHT`、`(0.3,0.3,0.3)→SAFE_NEUTRAL`、`(0.3,3,0.3)→FORWARD`、全 0/NaN/负→`SAFE_NEUTRAL`、小尺寸/短 buffer→`ValueError`。
- **期望改变**：`(1.2,1.2,1.2)`：第一阶段若保留 `d_min>=SOFT` 仍是 YAW_LEFT；第二阶段变成 FORWARD（1.2 m 全空、正前方区间够宽）。这是「贴障穿行」的预期行为，不是 bug。
- **新增**：
  - 一条窄障碍在 -10°（只封住约 20 列）、其余 5 m → 应 FORWARD（旧逻辑会因为左带最小值 0.5 而左转，可能朝障碍转）。
  - 两个柱的缝边缘在 ±13°、深度 2 m（缝宽约 2×2×sin13°≈0.90 m）→ 应 FORWARD。
  - 缝边缘在 ±7°、深度 2 m（缝宽约 0.49 m）→ 不应 FORWARD；应 YAW 或 SAFE_NEUTRAL。
  - 正前方 0.7 m 一条窄障碍、两侧 5 m → 应转向，不应前进。
  - 整幅图只有最左 10 列自由、其余 0.5 m → 贴边区间，应 YAW_LEFT（先转过去看）。
  - 区间内部夹一列无效读数 → 应把区间切断，不能连起来。

### 4.6 已知局限

- **没有姿态输入**：中间带假设相机光轴水平；实际有俯仰/滚转时地平线会移动，中间带可能看偏。`teacher.py` 拿不到 IMU/姿态，这是接口限制。
- **没有 FOV 记忆**：偏航后原障碍移出视野，teacher 不记得它；下一步可能朝它转回去。四个仓库用局部地图解决，我们没做。
- **深度是相机前方 0.2 m 处测的**，不是旋翼尖；`HARD=0.8` 已按旋翼尖余量折算，但侧向间隙仍靠机体宽度判据间接保证。
- **间隙宽度是从深度图近似估计的**：用边界列深度和角度算弦长，受柱面朝向、视差、FOV 边缘影响；`GAP_MARGIN` 需要实测校准。
- **`hold` 可变**：`collect.py --hold` 改了就改变偏航步长（0.6 rad/s × hold），`TURN_PENALTY` 需要跟着改；建议后续把 `yaw_step_deg` 传进 teacher。

---

## 5. 值得学习的工程做法

| 做法 | 他们怎么做 | 我们怎么用 |
|------|-----------|-----------|
| 参数外置 | EGO 用 launch XML；GCOPTER/Implicit-SDF 用 YAML + Config 结构体 | `teacher.py` 顶部一个 `PARAMS` dict；不要引入 YAML 解析器 |
| 可视化调试 | EGO `PlanningVisualization` 发布轨迹/走廊/回弹向量；GCOPTER `visualizer.hpp`；Implicit-SDF `debug_assistant` | 可选：把 `prof/runs/chosen` 写成 PGM/SVG，或加 `--debug` 打印每步的区间和评分 |
| 失败回退 | EGO 的 `EMERGENCY_STOP` + `emergency_time`；GCOPTER/Implicit-SDF 的越界当占据 | 无效列当 blocked；无区间/无机身可通行区间→`SAFE_NEUTRAL`；深度全失效→`SAFE_NEUTRAL`（已有） |
| 边界处理 | 地图 `isInMap`、索引钳位、越界 fail-closed | 保留 `w/h/短 buffer` 校验；列索引钳位；FOV 边缘区间特殊处理 |
| 滞回/状态 | EGO 的重规划状态机与 `flag_escape_emergency_` | 继续用 `prev_action`，但把滞回建立在「评分差」而不是「左右最小深度差」上 |
| 确定性 | 本项目的 `scene_gen.py` 按 seed 生成；teacher 无随机 | 保持 teacher 无随机，方便复现标签 |

---

## 6. 明确没读的部分

- `ego-planner`：`uav_simulator/*`（含 CUDA 局部感知、so3_control、mockamap）、`path_searching/dyn_a_star.cpp` 的函数体、`uniform_bspline.cpp`、`gradient_descent_optimizer.cpp`、`traj_server.cpp`、`ego_replan_fsm.cpp` 除状态机/紧急停止/局部目标外的部分、可视化实现。
- `EGO-Planner-v2`：`poly_traj_optimizer.cpp` 除 `obstacleGradCostP` 外的绝大部分、`swarm_bridge`、`drone_detect`、`main_ws/tracking_ws/interlaced_flight_ws`（只 grep 了参数）、教程 PDF。
- `GCOPTER`：`minco.hpp`、`gcopter.hpp` 的优化器主体、`trajectory.hpp`、`flatness.hpp`、`lbfgs.hpp`、`sdlp.hpp`、`root_finder.hpp`、`quickhull.hpp`、`map_gen/*`；只读了 `sfc_gen.hpp`、`firi.hpp` 入口段、`voxel_map.hpp`、`global_planning.cpp`、配置。
- `Implicit-SDF-Planner`：`mid_end.cpp`、`back_end_optimizer.cpp` 除 SDF 代价外的部分、`swept_volume` 内部、`PCSmap_manager` 的渲染、`depth_render.cu`、前端 A* 的搜索主体、`plan_manager.cpp`。
- 本项目：`collect_parallel.py`、`test_collect.py`、`save_frame.py` 只看了调用关系；没有运行任何仿真、没有跑 `teacher.py` 自测、没有编译任何东西。

不确定的点：

- EGO-Planner 快照里回弹方向的符号约定（`dist` 的几何正负）没有完全核清；不影响借用思想。
- Implicit-SDF 的配置单位：`occupancy_resolution=1.0` 和 `box_x/box_y/box_z` 是 mesh demo 的尺度，不能直接当米；`safety_hor=0.6` 大概率是米，但没有独立验证。
- EGO-Planner-v2 的参数在四个 workspace 里不完全一致；我读的是 `formation_ws` 的 `advanced_param.xml`。
- 间隙宽度从深度图的估计误差没有实测数据；`GAP_MARGIN` 需要拿真实 episode 调。

---

## 7. 过程记录

### 做了什么

- 只读检查了四个仓库的目录、许可证、依赖、关键源文件与配置。
- 梳理了 EGO/EGO-v2 的占据栅格 + A* + B 样条/回弹链路，GCOPTER 的 SDF-free 走廊链路，Implicit-SDF 的 EDT/梯度/hinge 代价链路。
- 对照了 `teacher.py`、`collect.py` 的动作映射、`uav_rgb.sdf` 的相机/机体参数、`scene_gen.py` 的 0.9 m 缝约束。
- 写了本报告，给出 R1–R6 的落地位置、行数和许可证标注。

### 没做什么

- 没有修改 `teacher.py`、`collect.py` 或任何源码/配置。
- 没有运行、编译、安装四个仓库中的任何一个。
- 没有运行 `teacher.py` 的现有自测，也没有跑仿真采集。
- 没有读第 6 节列出的文件/模块。
- 没有给出从 GPL 仓库复制的任何代码；报告里的伪代码是自写的。

### 待确认

- 是否按 R1→R2→R3 分阶段改 `teacher.py`；第一阶段是否保留 `d_min>=SOFT` 的前进条件。
- `GAP_MARGIN` 初值取 0.13 还是更保守的 0.15–0.20。
- 是否允许第二阶段产生贴障穿行样本（会改变现有 `(1.2,1.2,1.2)` 测试期望）。
- 是否要把 `hold`/`yaw_step_deg` 从 `collect.py` 传入 teacher。

### 验证到哪一步

| 项 | 结果 |
|----|------|
| 四个仓库许可证 | 已核对：三个 GPL-3.0，GCOPTER MIT |
| EGO 是否 ESDF-free | 已核对：README 明示 + 代码 grep 无 `esdf/edt` |
| 关键函数/参数路径 | 已按文件与行号核对 |
| teacher.py 自测 | 未跑（只读研究，不改代码） |
| 仿真/采集 | 未跑 |
| 代码编译 | 未做 |

### 风险与遗留

- 第二阶段「允许 0.8–1.5 m 的前方区间前进」会增加早期采集的碰撞率；建议先用 `GAP_MARGIN=0.15–0.20`，看 `collision` 统计再放宽。
- 贴 FOV 边缘的区间被当作「可转过去看」，如果边缘外正好是墙，无人机会先原地转、下一帧才发现；原地转不前进，风险可控，但会多消耗步数。
- 没有 FOV 记忆，偏航后可能忘记刚看到的障碍；这是四个仓库用局部地图解决、我们暂时不做的部分。
- 间隙宽度是深度图近似值，柱面/视差会让 0.9 m 的缝估成 0.8–1.0 m；`GAP_MARGIN` 是安全垫，不能省。
- 若 `hold` 改成非 0.5 s，`TURN_PENALTY` 和 `DEADBAND` 需要重标定。
