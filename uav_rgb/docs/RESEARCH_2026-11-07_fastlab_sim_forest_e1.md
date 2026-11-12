# FAST-B：FAST Lab 仿真/控制侧代码精读 → `forest_e1` 场景族 + 执行器建议

- 日期：2026-11-07 · 作者：FAST-B agent（QUEUE Q3）
- 工作树：`/root/rivermind-data/uav_rgb/tmp/wt-fastlab-20261106`（分支 `fastlab-20261106`；FAST-A 的三个文件没碰、没提交）
- 前置：`reports/B_academic.md`（文献层不重复）。本报告只讲**他们代码怎么做 → 我们对应在哪 → 最小改法 → 预期 → 怎么验**。
- 产物：`sim/tools/scene_forest.py`、`sim/tools/test_scene_forest.py`；统计与俯视图在 `reports/fastlab_b_forest/`（源文件 `/root/rivermind-data/uav_rgb/tmp/agent-fastlab-b/forest_stats/`）。

## 0. 结论（先看这个）

1. **`forest_e1` 场景族可用**：随机圆柱森林（半径 0.05–0.15 m），三档最窄缝 0.45 / 0.35 / 0.28 m，每个场景都带一条按 E1 半径 + 2.5 cm 膨胀后仍连通的 A* 走廊，且出生航线 y=0 上至少一根柱（否则盲直飞就能过——sparse 档不加这条时 20 个 seed 有 3 个这样）。90 场景自检 PASS，同 seed 字节级一致；版本串 `uavrgb.forest_e1.v1` 独立，旧柱阵 `SPLIT_SALT`/gid/split 一行没动。
2. **FAST Lab 的 map_generator 能借的是"撒点 + 间距拒绝重采"这一招和参数口径**（机体直径→膨胀→栅格分辨率），借不了的是它们**不保证可达**（靠起点放在障碍区外、规划器自己绕）、**不固定 seed**（`random_device`）、**间距按柱心不按柱面**；这三点我们都反着做了，理由见 §2 B1–B3。
3. **教师侧的真实风险不在场景而在方框近似**：`teacher.py` / `teacher_kino.py` 都把障碍当外接方框，圆柱角部多出 0.41r（r=0.15 时 6 cm）。按 `teacher_kino` 的 0.182 m 判据（即只走 ≥0.364 m 的缝）走廊在 sparse/medium/dense 存在 20/20、20/20、18/20，余量本来就只有几厘米，方框化再吃掉 ≤6 cm 会把其中一部分判成堵。**建议 FAST-A 的 `_boxes()` 认 `radius` 字段**（§4 R1），不然 dense 档对教师接近死局。
4. **px4ctrl 对我们执行器最值得抄的三件事**（§5，只写建议不改代码）：(a) 起飞前拒绝条件（当前 `cf_exec_arm_takeoff` 在 TAKEOFF/READY/ACTING 任何状态都接受，会把高度参考拉回去）；(b) FAILSAFE 应"缓降→怠速"而不是永远开环悬停油门（`executor.c:150-153`）；(c) 悬停油门只在**悬停/巡航态**学，起飞段和地效段冻结（px4ctrl `PX4CtrlFSM.cpp:302-307` 只在 AUTO_HOVER/CMD_CTRL 跑 RLS；我们 `executor.c:83` 在 TAKEOFF 也积分）。RLS 版油门估计需要机体加速度，`cf_exec_input` 目前没有 `acc`，是 SITL-core 的接口改动，列为可选。
5. **没做的**：没起 Gazebo 验 SDF 载入（任务禁止）；没接进 `scene_gen.py --task` / `collect.py` / `audit_dataset.py`（不在我的文件域，改法写在 §4 给对应 agent）；OPNet 只记录思路、e2e_opt 是项目网页无代码、FastSim/Auto-Param-Adaption 只有 README。

## 1. 读了哪些代码（file:line 可直接打开）

克隆目录 `/root/rivermind-data/uav_rgb/tmp/research-acad/`（代理 127.0.0.1:7897 不通，`env -u http_proxy -u https_proxy -u all_proxy git clone --depth 1` 直连）。

| 仓库 | 文件 | 看点 |
|---|---|---|
| EGO-Planner-v2 | `swarm-playground/main_ws/src/uav_simulator/map_generator/src/random_forest_sensing.cpp:165-297` RandomMapGenerateCylinder；`:183-200` 均匀撒点 + 与已放柱**柱心距** `< _min_dist` 则 `i--` 重采；`:204-205` 柱心吸附到栅格中心；`:210-222` 圆柱体素化（widNum×widNum 列，按 `(temp−center).norm() > radius` 裁圆）；`:442-461` 参数（`map/obs_num` 30、`ObstacleShape/lower_rad` 0.3、`min_distance` 1.0）；`:469` `_obs_num = min(_obs_num, x_size·10)` 防饱和死循环；`:25,:474-477` `random_device` 取 seed、只打印不接收 | 随机森林生成的全部逻辑 |
| EGO-Planner-v2 | `planner/plan_manage/launch/launch_gen.py:6-10`（map 20×20×3、`drone_diameter` 0.2、`average_obstacle_spacing` 2.5）；`:79-80` 柱数 = 面积 /（π·(spacing/2)²）；`:96-107` 柱半径 0.5–0.7、`min_distance` 0.8；`:121-123` 障碍区比地图区每边小 2 m（起点落在无障碍带）；`:314-316` `inflation = d/2`、`resolution = (inflation+0.01)/3`（≥0.1）、`swarm_clearance = d/2·1.5` | 机体尺寸→膨胀→分辨率的口径 |
| EGO-Planner-v2 | `uav_simulator/mockamap/src/maps.cpp:15-20` `default_random_engine eng(info.seed)`（**有 seed 参数**）；`:102-176` perlin 3D；`:181-318` 迷宫；`:611-614` `road_width` 等参数 | 可复现地图的做法 |
| EGO-Planner-v2 | `uav_simulator/so3_quadrotor_simulator/src/quadrotor_simulator_so3.cpp:219-221` 从参数读 `init_state_x/y/z`，`:227,:231` 直接 `setStatePos`；`Quadrotor.cpp:17,:22,:29-31` 质量/惯量/电机常数；无地面、无碰撞、按 wall-clock 跑 | 对照：它们的仿真不管碰撞，碰撞判定全在规划器 |
| EGO-Planner-v2 | `uav_simulator/local_sensing/src/depth_render.cu:2-43` 点云 splat 深度渲染（每点投影后画边长 `r = 0.0573·fx/dist` px 的方块，`atomicMin` 取最近，深度 int mm）；`pcl_render_node.cpp:286-293` ≥500 m 置 0 作无效、`:326-334` 相机内参/`sensing_horizon`/`sensing_rate` 参数 | 对照：无噪声、无 FOV 外遮挡模型，只是几何 z-buffer |
| Fast-Drone-250 | `src/realflight_modules/px4ctrl/src/PX4CtrlFSM.cpp:49-146` MANUAL 下进 HOVER/TAKEOFF 的拒绝条件（`:53` 无 odom、`:58` 提前发 cmd、`:65` odom 速度 >0.3、`:90` 非静止、`:95` land_detector 说没落地、`:98-102` RC 杆不在中位）；`:70,:118` 进入时 `resetThrustMapping`；`:150,:194,:248` odom 丢失 → MANUAL；`:201` cmd 超时 → AUTO_HOVER；`:223-241` 起飞：`MOTORS_SPEEDUP_TIME` 3 s 怠速、到高后 `DELAY_TRIGGER_TIME` 2 s 才接 cmd；`:246-296` AUTO_LAND；`:302-307` 只在 HOVER/CMD_CTRL 调 `estimateThrustModel`；`:345-349` 未解锁/地面时 `u.thrust = 0.04`；`:351-394` land_detector（速度 + 时间双条件）；`:423-461` 起飞加速度曲线 `exp((Δt−3)·6)·7−7`；`:504-527` 各消息超时（`msg_timeout` 0.5 s）；`PX4CtrlFSM.h:24-25` 两个常量 | 执行器 FSM 的参照 |
| Fast-Drone-250 | `px4ctrl/src/controller.cpp:77-80` 油门入队（≤100 条）；`:101-143` `estimateThrustModel`：取 35–45 ms 前的油门与当前 IMU z 加速度配对，RLS 更新 `thr2acc_`（`controller.h:74` 遗忘因子 0.998）；`:146-150` `resetThrustMapping`：`thr2acc = g / hover_percentage`，`P = 1e6`；`config/ctrl_param_fpv.yaml` `hover_percentage` 0.30、`msg_timeout` 0.5、起飞速度 0.3 m/s | 悬停油门在线估计 |
| Fast-Drone-250 | `px4ctrl/src/input.cpp:113-117` `check_centered` 四次都查 `ch[0]`（复制粘贴 bug） | 别照抄 |
| Ground-effect-controller | README §3.2 油门→转速二次模型 `n = c2 t² + c1 t + c0`（单电机台架标定）；§3.3 转速环：前馈 `n_esc⁻¹` + 对合推力加速度误差做 PI；§4.1 地效系数 g1–g5、`k_T` 台架 4.0083e-8 vs 真机悬停 4.2958e-8（差 7%） | 标定法；只有 README，无代码 |
| FastSim / Auto-Param-Adaption | 仓库只有 README（FastSim「coming soon」，Auto-Param-Adaption 只有标题 + LICENSE） | 无代码可读 |
| OPNet | `krrt-planner/opnet/src/torch_dense/train_nosurf.py`、`models/*.onnx`、`mapping/occ_grid/src/occ_map.cpp`（TensorRT 推理嵌在建图里）：输入局部体素占据（80×80×40 @5 cm），SG-NN 风格稀疏→稠密补全预测被遮挡占据 | 只记录思路（§6） |
| e2e_opt | 仓库是项目网页（JavaScript/bulma），无代码 | — |

## 2. 借鉴清单（每条：他们怎么做 → 我们对应 → 最小改动 → 预期 → 怎么验）

B1 **撒点 + 间距拒绝重采（RSA）**
- 他们：`random_forest_sensing.cpp:183-200` 均匀撒柱心，和已放柱**中心距** `< _min_dist` 就重采；`:469` 柱数封顶防死循环。
- 我们：`scene_gen.py` 是固定车道柱阵（1.2 m 缝按 X3 定）。
- 改动：`scene_forest.py:170-204 sample_trunks`：同样撒点重采，但拒绝条件改成**柱面缝** `dist − r1 − r2 < gap`（机体能不能过看的是缝不是心距）；候选预算 n×200 代替 `:469` 的封顶；整组不达标就换一组柱数重来（≤100 轮）。
- 预期：三档缝宽是**硬契约**（任意两柱面 ≥ 0.45/0.35/0.28），而不是统计平均。
- 验：`--self-check` 90 场景 min_pair_gap ≥ 档位值全过；`test_scene_forest.py` 同条断言 + 未知档位报错。

B2 **保证可达（他们没有，我们必须有）**
- 他们：不检查连通性，靠 `launch_gen.py:121-123` 把障碍区每边缩 2 m、起点放空地，规划器自己绕；`so3_quadrotor_simulator` 也不判碰撞。
- 我们：`teacher.astar_path` 把障碍当方框、步长 0.1 m，对 0.28 m 缝太粗，不能直接拿来判走廊。
- 改动：`scene_forest.py:110-167 corridor()`：2.5 cm 栅格、8 邻接禁斜穿角的 A*，障碍按圆 + clearance（E1 半径 0.102 + 0.025 = 0.127 m）精确膨胀，走廊带 y∈[−1.5,1.5] 外视为墙（否则绕场外一定更短）；没路就重采。顺带算路径长、最小净空、瓶颈缝宽（路径点两侧最近柱面之和）。
- 预期：每个落盘场景都有一条 E1 几何上能过的路；dense 档瓶颈 ≥0.28。
- 验：自检「走廊最小净空 ≥ 0.127」「瓶颈 ≥ 档位缝」全过；单测横墙负例（缝 = 2·clearance − 2 cm）判 None、开一口判有路。

B3 **显式 seed、seed 混入档位**
- 他们：`random_forest_sensing.cpp:474` `random_device` 取 seed 只打印；`mockamap/maps.cpp:18` 才接 `info.seed`。
- 我们：`scene_gen.py` 已经是 `random.Random(seed)`。
- 改动：`random.Random(f"forest_e1:{density}:{seed}")`——同 seed 不同档位**不共享骨架布局**（否则 train 的 sparse 和 test 的 dense 可能是同一组柱加几根）。
- 预期：12 个 (档位, seed) 组合 gid 两两不同；同 seed 字节级一致。
- 验：单测 `len(gids) == 12`；自检 `w1 == w2`。

B4 **机体尺寸 → 膨胀 → 栅格分辨率的口径**
- 他们：`launch_gen.py:314-316` `inflation = d/2`、`resolution = (inflation + 0.01)/3`、`swarm_clearance = 0.75d`。
- 我们：`CORRIDOR_MARGIN_M = 0.025`、`GRID_M = 0.025`。
- 改动：取值逻辑照它们：dense 档自由带只有 0.28 − 2×0.127 = 0.026 m，栅格必须 ≤ 它（他们的 1/3 规则给 0.037，对我们太粗，故取 0.025）。
- 预期：dense 档不会因为栅格量化把能过的缝判堵。
- 验：自检 dense 20 seed 全部 1 次采样即过（如果栅格太粗会频繁重采）。

B5 **柱数按饱和密度定，不按面积公式硬算**
- 他们：`launch_gen.py:79-80` 柱数 = 面积/(π(spacing/2)²)，再 `/2`（`:123`），本质是经验系数。
- 我们：先测 RSA 饱和（4×3 m²、r 均值 0.10：25 / 34 / 45 根，10 seed 均值），`DENSITY_COUNT` 取其 55–73%。
- 预期：接受率高（实测 100 seed/档 重采 ≤2 次），又不至于稀到没难度。
- 验：`forest_stats.json` 的 `attempts`：sparse 91/9（1 次/2 次）、medium 97/3、dense 100/0。

B6 **起点处"盲飞必撞"的难度保证（自己加的）**
- 他们：无。
- 我们：sparse 档不加约束时 20 seed 有 3 个 y=0 直线上无柱，整集一次避障决策都没有。
- 改动：`validate_trunks` 要求 `|y| < r + E1 半径` 的柱 ≥1，否则重采。
- 预期：每个场景至少一次转向；三档实测直线上柱数均值 2.4 / 3.2 / 4.3。
- 验：自检「出生航线上没有柱」为 0 失败；单测断言 `trunks_on_spawn_line ≥ 1`。

B7 **和旧工具链的兼容：同时给 radius 和外接方框 size**
- 他们：体素化之后形状信息就没了（`:210-222`）。
- 我们：`split_manifest.geometry_gid` 读 `size`；`teacher._boxes`/`teacher_kino._boxes` 读 `size`。
- 改动：每根柱 `{"x","y","z","radius","shape":"cylinder","size":[2r,2r,3.0]}`。旧代码把圆当方框只会偏保守（角部多 0.41r ≤ 6 cm），不会漏判碰撞。
- 验：单测 `teacher._point_blocked(柱心) == True`、`split_for_scene(meta) == meta["split"]`。

B8 **不借的**：`so3_quadrotor_simulator` 的无碰撞、wall-clock 仿真（我们 SITL 定步长、有碰撞）；`depth_render.cu` 的点云 splat 深度（我们场景是解析圆柱，要无 Gazebo 深度直接射线求交更准）；`mockamap` perlin/迷宫（E1 任务是穿林，不是迷宫）。

## 3. `forest_e1` 场景族

### 3.1 设计（全部常量在 `scene_forest.py:38-60` 配置区，每个带理由）

| 项 | 值 | 理由 |
|---|---|---|
| 版本 / salt | `uavrgb.forest_e1.v1` | 独立于 `sg.GEOMETRY_VERSION`；meta 显式写 `split_salt`，因为 `split_manifest.split_salt_for` 只认 `uavrgb.geom.*` 前缀 |
| 三档最窄缝 | 0.45 / 0.35 / 0.28 m | 任务口径 2.5×/2×/1.5× ≈0.18 m 外径；按 `e1_mini.json` 外接圆 0.204 m 是 2.2×/1.7×/1.4×，dense 单侧余量 3.8 cm 是刻意的上限 |
| 柱数范围 | 14–18 / 20–25 / 26–33 | RSA 饱和 25/34/45 的 55–73%（B5） |
| 柱半径 / 高 | 0.05–0.15 m / 3.0 m | 任务给定；高 3 m 同旧柱阵，目标高 2.0 m + 起飞超调 0.21 m 仍在柱内 |
| 森林区 | x∈[1.5,5.5]，y∈[−1.5,1.5] | 起点同旧柱阵；4 m 以 0.3 m/s 约 13 s；顶视相机 (2.5,0,6) 不用动 |
| 出生 / 目标 | (−0.5,0,0.15) / (6.0,0) | 同旧 default；穿出森林即算到 |
| 走廊 clearance / 栅格 | 0.127 / 0.025 m | B4 |
| 载具 pose | 正则匹配 `x3|e1_mini` 第一条 pose | 工作树 `scene_gen.py` 只认 x3，主仓库 `--vehicle` 改动未提交；模板是 x3 时 stderr 警告 |
| 外观 | 复用 `scene_gen._apply_appearance`（v1 默认） | 外观不进 gid；单测验证 legacy/v1 几何一致 |

### 3.2 20 seed × 3 档几何统计（`reports/fastlab_b_forest/forest_stats.{md,json}`）

| 档 | 缝契约 | 柱数 | 最窄缝 min/mean | 直线上柱数 | 密度 根/m² | 面积占比 | A* 路径长 (直线 6.5) | 路径/直线 | 走廊瓶颈 min/mean | 重采 max |
|---|---|---|---|---|---|---|---|---|---|---|
| sparse | ≥0.45 | 15.9 [14–18] | 0.451 / 0.464 | 2.35 [1–5] | 1.33 | 4.2 % | 6.68 [6.54–6.75] | 1.028 | 0.453 / 0.528 | 2 |
| medium | ≥0.35 | 22.7 [20–25] | 0.350 / 0.361 | 3.20 [1–5] | 1.89 | 5.3 % | 6.74 [6.58–6.87] | 1.037 | 0.351 / 0.413 | 2 |
| dense | ≥0.28 | 29.2 [26–33] | 0.280 / 0.286 | 4.35 [3–6] | 2.43 | 7.5 % | 6.80 [6.67–6.99] | 1.047 | 0.282 / 0.320 | 1 |

- 走廊最小净空 min 0.127 m（= clearance，说明 A* 确实贴着膨胀边走，瓶颈值是真实的最窄处）。
- split（100 seed/档，80/10/10 哈希）：sparse 78/11/11、medium 80/12/8、dense 86/3/11（train/val/test）；小样本抖动，不是偏置。
- 路径只比直线长 3–5%：E1 在这些缝里基本是"小幅绕"，不是迷宫——对 4 动作（前/左转/右转/停）的策略是合适的难度；要更难就加密度档，不要加宽森林。
- 盲直飞探针（临时脚本，未入库）：三档 20/20 都会撞；按 `teacher_kino` 的 0.182 m 判据走廊存在 20/20、20/20、18/20。

### 3.3 俯视 PNG（seed 0，每档一张）

- `reports/fastlab_b_forest/forest_sparse_seed0.png`
- `reports/fastlab_b_forest/forest_medium_seed0.png`
- `reports/fastlab_b_forest/forest_dense_seed0.png`

（棕圆 = 柱，细圈 = +clearance 膨胀边，蓝线 = A* 走廊，绿圆 = E1 外接圆出生点，红星 = 目标。）

### 3.4 用法

```
python3 sim/tools/scene_forest.py --seed 42 --density dense --out <dir>/forest_42.sdf   # 同名 .json 是 meta
python3 sim/tools/scene_forest.py --self-check [--check-seeds 30]
python3 sim/tools/scene_forest.py --stats-out <dir> [--stats-seeds 20]                  # 要 matplotlib
python3 sim/tools/test_scene_forest.py
```

模板默认 `sim/worlds/uav_rgb_e1.sdf`（GZ agent 的文件，主仓库未提交；工作树里我复制了一份未加入 git），不存在时报错提示 `--template`。

## 4. 给其它文件/agent 的接入请求（我的文件域外，没改）

R1 **教师认 `radius`（FAST-A / teacher 维护者）**：`teacher.py _boxes()`、`teacher_kino.py _boxes()` 读 `size` 当方框。请加一分支：有 `radius` 字段时用圆距 `hypot(dx,dy) − r`。不改则对教师偏保守 ≤6 cm：dense 档按 0.182 m 判据能走的缝 ≥0.364 m，而档内缝宽分布就在 0.28–0.40 m 一带，6 cm 的误差会把相当一部分可走缝判堵（18/20 存在的走廊会再掉一截）。验：`compare_teachers.py` 换 `scene_forest` 场景跑 20 seed，看到达率；改前改后各跑一次。

R2 **`scene_gen.py --task forest_e1` 分发（scene_gen 维护者）**：`collect.py` / `collect_parallel.py` / `eval_closed_loop.py` 都是 subprocess 调 `scene_gen.py`。最小改法 3 行：`main()` 开头 `if args.task == "forest_e1": import scene_forest; return scene_forest.main()`（再透传 `--density`）。不动旧布局/版本。验：`collect.py --task forest_e1 --density medium` 的 meta `geometry_version` 为 `uavrgb.forest_e1.v1`。

R3 **`audit_dataset.py` 森林分支**：现在要求 `geometry_version == SPLIT_SALT` 且几何常量匹配 `GEOM_CONST`，森林数据会直接 FAIL。加 `geometry_version.startswith("uavrgb.forest_e1.")` 分支，检查项换成 `clearance.min_gap_required_m` ∈ {0.45,0.35,0.28}、`min_pair_gap_m ≥ min_gap_required_m`、`corridor.min_clearance_m ≥ 0.127`、`split == split_for_scene(meta)`。

R4 **`collect.py` label_recompute 口径**：若标签由方框教师复算，森林数据的标签会比几何真相保守；R1 做完前，森林集只用于闭环评估，不混入训练标签。

R5 **E1 模板 XML（GZ agent）**：`sim/worlds/uav_rgb_e1.sdf` 第 14、19 行注释里有 `--camera-rate`、`--no-aux-cameras`，XML 注释内不允许 `--`，expat/`xml.etree` 解析失败（Gazebo 的 tinyxml2 宽容所以能跑）。改成 `—camera-rate` 或 `camera-rate 选项` 即可；我的单测对这个模板只解析生成的柱段并打 note。

## 5. 给 `sim/cf_sitl/exec/` 的建议（只建议，不改；对照 px4ctrl）

现状（主仓库 `sim/cf_sitl/exec/executor.c`，160 行）：DISARMED→TAKEOFF→READY→ACTING→FAILSAFE；高度环 = 悬停杆前馈 + PD + 积分自适应（`:77-85`）；ToF 失效 >0.5 s → FAILSAFE（`:112-114`）；FAILSAFE 永远输出 `hover_stick`（`:150-153`）。

S1 **起飞/动作前置条件（px4ctrl `PX4CtrlFSM.cpp:76-115`）**
- 现状：`cf_exec_arm_takeoff`（`:29-35`）只拒 FAILSAFE，在 TAKEOFF/READY/ACTING 再调用会把 `height_ref_m` 重置、状态拉回 TAKEOFF。
- 建议：只在 DISARMED 接受，其它状态返回错误码（纯 C 可改成 `int` 返回）；`cf_exec_set_action` 在非 READY/ACTING 时忽略并计数（px4ctrl `:58-62` 对"提前发 cmd"直接拒）。
- 预期：宿主误调不再扰动飞行；可观测的拒绝计数便于查 bug。
- 验：`test_cf_sitl_exec`（SITL agent 的）加「TAKEOFF 中重复 arm 不改变 height_ref」用例。

S2 **ToF 超时判定要有"离地"门槛（px4ctrl land_detector `:351-394` 的思路反过来用）**
- 现状：`height_valid` 一旦置 1 永不清零（`:52-53`），起飞初段 ToF 在最小量程 0.04 m 附近时有时无，可能 0.5 s 内没新样本直接 FAILSAFE。
- 建议：FAILSAFE 判定加 `height_est_m > tof_range_min + 0.05`；离地前 ToF 无效只当"高度 0"。
- 预期：消除起飞第 1 秒的假 FAILSAFE。
- 验：SITL 把 ToF 最小量程改 0.10 m 重跑 smoke，不应 FAILSAFE。

S3 **FAILSAFE 改"缓降→怠速"（px4ctrl AUTO_LAND `:246-296` + `:345-349` idle 0.04 + land_detector）**
- 现状：`:150-153` 开环 `hover_stick` 悬停不再闭环，电池掉压后会慢慢爬升或下沉，永不落地。
- 建议：FAILSAFE 内按 `takeoff_climb_rate` 的一半开环降油门（`hover_stick − k·t`，k 使 3 s 内降到 `stick_deadband`），到 `stick_deadband` 后保持怠速；若光流仍有效可继续水平闭环。
- 预期：失效后 3–5 s 内落地，不再悬在空中等 reset。
- 验：SITL 注入 ToF 失效，看落地时间与落地速度（<0.5 m/s）。

S4 **悬停油门自适应的学习时机（px4ctrl `:302-307` 只在 HOVER/CMD_CTRL 调 RLS）**
- 现状：`height_loop` `:83` 在 TAKEOFF 也积分；起飞头几百毫秒在地效里（65 mm 桨，地效在离地约 1–2 倍桨径即 0.07–0.13 m 内显著；出生 z=0.15，ToF 最小量程 0.04），这段学到的 `hover_stick` 偏低，爬出地效后掉高。Ground-effect README §1/§4.1 用 g1–g5 五个系数单独建模地效，说明他们把它当一等误差源，不是小量。
- 建议：`state == TAKEOFF && height_est_m < 0.30` 时冻结积分；或干脆 READY 之后才开积分（px4ctrl 的做法）。
- 预期：READY 后悬停误差不变，爬升段超调减小。
- 验：smoke 对比起飞超调（现 0.21 m）。

S5 **RLS 版悬停油门估计（可选，需 SITL-core 加接口）（px4ctrl `controller.cpp:101-143`）**
- 做法：把每 tick 油门杆入队，取 ~80–100 ms 前的杆（E1 电机 τ_up 60 ms / τ_down 120 ms，比 px4ctrl 的 35–45 ms 慢）与当前机体 z 加速度配对，一维 RLS 估 `thr2acc`，遗忘 0.998；`hover_stick = g / thr2acc`。
- 前提：`cf_exec_input` 没有加速度（`executor.h:33-42`），要 SITL-core 加 `acc_body[3]`；真机 ESP32 有 MPU6500，可行。
- 预期：电池从 4.2 V 掉到 3.5 V 悬停杆从 0.734 变到 0.922（`e1_mini.json hover_duty_estimate`），RLS 比 3 s 时间常数的积分跟得快，且不吃高度误差。
- 验：SITL 电池模型全程放电，比较两种自适应的高度误差。
- 不建议照抄的：`input.cpp:113-117` 四次查 `ch[0]` 的 bug；px4ctrl 的"odom 丢失 → MANUAL"对无遥控的 E1 没有对应物，对应物就是 S3 的缓降。

S6 **起飞前怠速/前馈斜坡（`:225-228` 3 s 怠速、`:423-441` 加速度曲线）**
- 现状：解锁后下一 tick 直接 `height_loop` 输出 ≈0.91 杆。
- 建议：解锁后 0.3 s 内把 `hover_stick` 前馈从 `stick_deadband` 线性拉到标称值（不用 3 s，E1 电机 τ 60 ms），之后再开 PD。
- 预期：减小起跳瞬间的俯仰扰动（四电机启动不同步）。
- 验：smoke 起飞 1 s 内倾角峰值。

S7 **动作超时语义**：我们的 `until_s` 过期回 SAFE（`:140-142`）与 px4ctrl `:201` cmd 超时回 HOVER 等价，不用改。

## 6. 只记录不实施：OPNet 对 3DGS 碰撞的启发

OPNet 把局部体素占据（80×80×40 @5 cm）送进稀疏→稠密补全网络，预测被遮挡区域的占据，规划器在"预测占据"上做碰撞检查，减少探索性绕路。我们 3DGS 场景的碰撞目前靠高斯中心/椭球近似，遮挡背面本来就是空洞；若将来要在 3DGS 里做无 Gazebo 的碰撞标签，思路是：(1) 从 3DGS 渲染深度→体素占据；(2) 用 `forest_e1` 这类解析场景生成"部分观测→完整占据"对训练小补全网络；(3) 碰撞查询用预测占据。本期不做（无 GPU、3DGS 碰撞不在 Q3 范围）。

## 7. 没做什么 + 原因

- 没起 Gazebo 载入生成的 SDF：任务禁止；用 `xml.etree` 验了 X3 模板输出整篇合法、E1 模板输出的柱段合法（E1 模板自身注释问题见 R5）。
- 没改 `scene_gen.py`、`collect.py`、`audit_dataset.py`、`teacher*.py`、`sim/cf_sitl/exec/`：不在文件域，改法在 §4/§5。
- 没提交 `sim/worlds/uav_rgb_e1.sdf`（GZ agent 的未提交文件，只复制到工作树跑测试）。
- 没做 3D 变化（柱倾斜、高度随机、悬挂障碍）：4 动作策略在固定高度飞，任务只要圆柱森林；要做就 bump 到 `forest_e1.v2`。
- 没测 `teacher_kino` 在森林上的到达率：它是 FAST-A 的文件，且 R1 没做前结果只反映方框近似的保守，不反映场景难度。
- 没把 `--stats-out` 加进 `tools/check.sh`：要 matplotlib，且 8 s；`test_scene_forest.py` 已被 check.sh 的 `sim/tools/test_*.py` 循环自动覆盖。
- OPNet/e2e_opt 未克隆完整（OPNet 25 MB、e2e_opt 213 MB，直连超时），用 GitHub API 读了文件树、README 与训练脚本头部；e2e_opt 只有网页。

## 8. 文件 / 提交

- 新增 `sim/tools/scene_forest.py`（生成器 + `--self-check` + `--stats-out`）、`sim/tools/test_scene_forest.py`、`docs/worklog/2026-11-07-0115-fastlab-forest-e1.md`。
- 分支 `fastlab-20261106` 提交 `3becc98`（只含上面三个文件；`sim/worlds/uav_rgb_e1.sdf` 未提交）。
- 临时产物（数据盘，不入库）：`/root/rivermind-data/uav_rgb/tmp/agent-fastlab-b/{forest_stats/,sample/,calib_rsa.py,probe_difficulty.py,PROGRESS.md}`。
