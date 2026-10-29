# 参考真机无人机栈调研：Fast-Drone-250 / Elastic-Tracker / Fast-Racing

- 时间：2026-10-29 00:37
- 范围：只读调研 `/root/claude-code-backup-20261028/code/idea/repos/` 下三个仓库；本文只写结论，不改任何代码
- 结论：完成（静态阅读）；未运行、未编译任何被调研仓库

## 0. 最值得做的 3 条（先给结论）

| # | 做什么 | 解决我们什么问题 | 落在哪个文件 | 工作量 | 许可证风险 |
|---|---|---|---|---|---|
| 1 | 采集 I/O 常驻化：每个话题一个常驻 `gz topic -e`，内存里留 2–3 帧；按 sim-time 配对深度/RGB/位姿；动作 hold 改为等 `/clock` 前进 | 1.7 s/步；168–308 ms 不同帧；wall-clock 动作时长 | `sim/tools/collect.py`（`echo_json`/`collect_one`/`take_off`），必要时 `sim/worlds/uav_rgb.sdf` | 1–1.5 人日 | 仅思想；来源 FD/ET，代码不复制 |
| 2 | 把相机参数固化成 `config/camera.yaml`：从 SDF 的 hfov 推 fx/fy/cx/cy，用 `/uav/camera_info` 和已知点投影校验；manifest 不再写 null | 无内参标定 | `config/camera.yaml`（新建）、`sim/tools/collect.py` 的 manifest 段 | 0.5 人日 | 仅思想；来源 FD/ET |
| 3 | 写一份安全监督器契约：新鲜度谓词 + 优先级 + 停止/悬停/降落三级降级 + ESP32 指令超时；先在仿真里实现可测试版本 | 深度失效只返回 `SAFE_NEUTRAL` 不够；未来部署的失联/失效处理 | `docs/` 新安全规范；`sim/tools/collect.py`/`teacher.py` 记录 safety_state；ESP32 固件后续实现 | 0.5 人日（文档+仿真钩子），ESP32 1–2 人日（硬件未确认） | 仅思想；来源 FD/ET，代码不复制 |

> 如果只做一件：先做 #1。它同时解决“慢”和“不同帧”，并让后续标定/安全字段有地方落。
> 如果 1.7 s 仍降不到目标：需要 C++ `gz-transport` 小桥接（订阅+发布+`/clock`），不是再堆并行实例。

## 1. 三个仓库一句话定位

- **Fast-Drone-250**：浙大 FAST-Lab 的 250mm 真机全栈课程仓库（PX4/MAVROS + VINS-Fusion VIO + EGO-Planner + px4ctrl + 仿真/硬件文档）。**仓库根目录没有 LICENSE**，readme 明确写“严禁商用与转载”，按“保留所有权利”对待；子包另有 GPL-3.0（px4ctrl、VINS-Fusion）、Apache-2.0（realsense-ros）、BSD（drone_detect、DecompROS），但不能覆盖顶层缺失和 readme 限制。
- **Elastic-Tracker**：GPL-3.0，面向空中跟踪的“深度+里程计建图 + 目标 EKF + 弹性轨迹规划”真机/仿真栈，安全侧有心跳和重规划失败悬停。
- **Fast-Racing**：GPL-3.0，面向无人机竞速的 SE(3) 轨迹规划基线，CPU/GPU 双版本，基于 AirSim/UE4；**Ctrl 的 FSM 主体已被整体注释掉**，当前代码没有可用的失效状态机。

## 2. 六个问题的结论

### Q1 传感器数据管线：帧率、丢帧、时间戳、同步

**他们怎么做（证据）：**

- Fast-Drone-250 的仿真传感器节点 `src/uav_simulator/local_sensing/src/pcl_render_node.cpp` 用 30 Hz 定时器渲染一次，然后把深度、彩色深度、相机位姿**全部打上同一个 `last_odom_stamp`** 再发布（`:271-317`，定时器 `:365-367`）。下游因此可以按 header stamp 对齐。
- 真机侧 `src/planner/plan_env/src/grid_map.cpp:102-121` 用 `message_filters::ApproximateTime` 同步“深度图 + 位姿/里程计”，队列 100；`grid_map.cpp:655-660` 用 `odom_depth_timeout: 1.0 s` 检测“深度或里程计丢失”。
- Elastic `src/mapping/src/mapping_nodelet.cpp:21-23, 301-302` 同样用 `ApproximateTime` 同步 depth+odom，队列 100；回调入口用 `std::atomic_flag callback_lock_`，**上一帧没处理完就直接丢新帧**（`:52-56`），这是很实用的背压策略。
- 目标 EKF `src/detection/target_ekf/src/target_ekf_node.cpp` 用 `ApproximateTime` 同步检测框+里程计；预测定时器发现 `update_dt >= 2.0 s` 就停止预测并告警。
- VINS 配置 `config/fast_drone_250.yaml`：相机 30 Hz，但跟踪结果 `freq: 10`；`max_solver_time: 0.04`、`max_num_iterations: 8` 给优化器设上限；`estimate_td: 1`、`td: -0.05` 在线估计相机-IMU 时间偏移。
- realsense 驱动给每个流挂 `FrequencyDiagnostics`（`base_realsense_node.cpp:981-994, 2404`），按期望频率统计实际发布频率。

**结论/可抄的思想：**

1. **统一时钟 + 同一次采样打同一个 stamp**，比“四次独立快照再事后对齐”根本。
2. **按 header stamp 配对，而不是按读到的先后顺序**。
3. **慢消费者丢旧帧（latest-wins）**，不要排长队；队列 100 对 30 Hz 传感器是 3 秒积压，不适合我们。
4. **显式记录/监控实际频率和 stale 时间**，不要静默继续。
5. VINS 的 `td` 只适用于相机-IMU 融合；我们没有 IMU 参与策略，**不抄在线 td**，但可以抄“把固定偏移写进配置并验证”的组织方式。

### Q2 安全与失效处理：失联、传感器失效、状态异常

**他们怎么做（证据）：**

- px4ctrl 的 `src/realflight_modules/px4ctrl/src/PX4CtrlFSM.cpp:506-526` 对 rc/cmd/odom/imu/bat 分别做“最后接收时间 < 0.5 s”的新鲜度判断；阈值在 `config/ctrl_param_fpv.yaml` 的 `msg_timeout`。
- 状态迁移（`PX4CtrlFSM.cpp:145-290`）：
  - `AUTO_HOVER`/`CMD_CTRL` 中 RC 丢失或 odom 丢失 → 回 `MANUAL_CTRL` 并退出 OFFBOARD；
  - `CMD_CTRL` 中指令流丢失但 RC 还在 → 回 `AUTO_HOVER`（位置保持）；
  - `AUTO_LAND` 按固定速度下降，PX4 报 landed 后 disarm；
  - 起飞/切悬停前检查 odom 新鲜、速度阈值（>3 m/s 拒绝悬停、>0.1 m/s 拒绝起飞）、land detector、RC 开关居中。
- 规划器 `src/planner/plan_manage/src/ego_replan_fsm.cpp:686-742`：深度/odom 超时 → `EMERGENCY_STOP`（把轨迹压成当前点，相当于急停），并关闭自动逃逸；轨迹前方 `emergency_time_`（默认 1.0 s）内发现碰撞 → 急停，否则重规划。`planner_manager.cpp:342` 的 `EmergencyStop` 就是把 6 个控制点全设成当前位置。
- Elastic：规划节点重规划失败 → hover；`traj_server.cpp:123-126` 心跳超过 0.5 s → 发布“当前点 + 零速度”的悬停指令；目标 EKF 对速度 >4 m/s 的更新直接拒绝（`target_ekf_node.cpp` 的 `checkValid`/`update`）。
- 电池：px4ctrl 只做电压低通和日志/告警（`input.cpp:255-295`），`low_voltage` 参数没有接入自动降落。**不要把这段当安全范例。**
- Fast-Racing：`CtrlFSM.cpp` 的 switch 整体被注释，只留下 `process_cmd_control`，没有超时/失效处理。**不能作为安全参考。**

**结论/可抄的思想：**

- 他们的优先级是**隐式**的：飞手 RC/硬件开关 > 机载自主；自主内部“新鲜度检查 > 状态机 > 控制指令”；安全定时器（20 Hz）高于规划器主循环。没有一个显式优先级表，但行为是这个顺序。
- **机载电脑进程崩溃/掉电本身没有被 px4ctrl 检测**；这依赖 PX4 自己的 OFFBOARD 失联 failsafe（不在本仓库、未验证）。我们的 ESP32 必须自己实现“指令超时”，不能假设上位机还活着。
- 失效动作分三级：**停止/保持 → 悬停 → 降落**；不是单一“返回中立”。
- 我们的 `SAFE_NEUTRAL` 不够，原因：
  1. 没有 stale 检测和超时升级；
  2. 没有区分“深度失效”和“位姿失效”；
  3. 没有链路失联/电池/策略输出异常；
  4. `SAFE_NEUTRAL` 在真机上只是“零速度指令”，是否真悬停取决于飞控的高度/位置环；ESP32 必须把它定义成“零速度 + 高度保持”，并在指令超时后自己接管；
  5. 仿真采集器在深度读取失败时是**终止 episode**（`collect_one` 返回 None），不是发中立；这对数据质量是对的，但不能当成飞行安全策略。

**建议的优先级（草案，阈值要在仿真里调）：**

| 优先级 | 事件 | 动作 |
|---|---|---|
| 0 | 飞手 RC / 硬件急停 / ESP32 kill | 立即接管，最高 |
| 1 | 上位机指令链路 stale > 200 ms | ESP32 零速度 + 高度保持；> 2 s 自动降落 |
| 2 | 位姿/odom stale > 300 ms | 保持最后有效位置；> 1 s 降落（无法导航） |
| 3 | 深度/RGB stale > 300–500 ms | 立即停止平移和偏航；> 2 s 降落 |
| 4 | 策略输出 NaN/越界/连续低置信 | 单帧 `SAFE_NEUTRAL`；连续 3 帧降落 |
| 5 | 电池低 | 降落 |
| 6 | 碰撞 | 停止；无法后退，先偏航脱离；重复碰撞降落 |

> 不要照抄 px4ctrl 的 0.5 s：它是 100 Hz 控制环 + 30 Hz 传感器的粗粒度值。我们的深度 30 Hz 一帧 33 ms，3 帧约 100 ms 就该报警；具体值用仿真注入延迟来标定。

### Q3 相机标定：内参怎么标、流程和文件怎么组织

**他们怎么做（证据）：**

- Fast-Drone-250 真机流程（`readme.md` 第十章）：启动 realsense，`rostopic echo /camera/infra1/camera_info`，把 K 里的 fx/fy/cx/cy 填进 `VINS-Fusion/config/left.yaml` 和 `right.yaml`；格式是 `model_type: PINHOLE`、`image_width/height`、`distortion_parameters`（k1/k2/p1/p2）、`projection_parameters`（fx/fy/cx/cy）。
- 离线棋盘格标定工具在 VINS 的 `camera_models`：`camera_models/camera_calib_example/readme.txt` 给出 `Calibrations -w 12 -h 8 -s 80 -i calibrationdata --camera-model pinhole`；`camera_models/src/intrinsic_calib.cc:234` 输出 `<cameraName>_camera_calib.yaml`。流程是“拍 12×8、80 mm 棋盘格 → 离线标定 → 把内参抄进 VINS 配置”。
- 外参：`fast_drone_250.yaml` 里 `body_T_cam0/1` 是 4×4 矩阵；`estimate_extrinsic: 1` 先给初值再在线优化；readme 的“外参精确自标定”是拿起飞机缓慢走动，把 `vins_output/extrinsic_parameter.txt` 回填配置，反复直到里程计误差收敛。
- Elastic 的参数组织：`mapping/config/camera.yaml` 与 `D435.yaml` 分开，字段包括 `camera_rate`、`camera_range`、`cam_width/height`、`cam_fx/fy/cx/cy`、`depth_scaling_factor`、`cam2body_R`、`cam2body_p`；launch 里用 `<rosparam command="load" file="...">` 选择。`target_ekf/config/camera.yaml` 是同一套字段的另一种相机。
- 深度尺度：realsense 的 16UC1 是毫米，所以 `depth_scaling_factor: 1000.0`；我们的仿真深度是 `R_FLOAT32` 米，**对应值必须是 1.0，不能照抄 1000**。

**结论/可抄的思想：**

- **内参离线固定、外参可在线估计但必须持久化**；配置文件按“相机型号/场景”分文件，字段显式写单位。
- **先信驱动发布的 `camera_info`，再信公式**；两者不一致时以实测/标定值为准，并记录来源。
- 我们的仿真相机是理想针孔：SDF `sim/worlds/uav_rgb.sdf:160-176` 给 `horizontal_fov=1.0471975512`（60°）、320×240，因此
  `fx = fy = 320 / (2*tan(60°/2)) ≈ 277.128`，`cx=160`，`cy=120`，畸变全 0。
  `sim/README.md` 的话题表里列了 `/uav/camera_info`（gz.msgs.CameraInfo），**现场用 `gz topic -l` 确认它确实存在**；存在就直接读 K，不存在就用上面的公式并在 manifest 里标 `camera_source: sdf_hfov`。
- 校验不要只看数字：在世界里放一个已知点（或用地标），把 3D 点投影到像素，和图像里实际像素比，误差 <1 px 才算过；这一步同时验证 `cx/cy` 的像素中心约定。
- 外参：仿真里相机 `pose 0.2 0 0` 相对 `base_link`，但 Gazebo 相机坐标系与视觉光学坐标系（z 前、x 右、y 下）的约定要实测确认；**不要直接抄 Fast-Drone-250 的 `cam2body_R`**，他们的相机安装方式不同。我们当前的策略只用 RGB，不需要外参；等要做深度投影/真机部署时再固化。

### Q4 数据记录：飞行日志/采集格式

**他们怎么做（证据）：**

- Fast-Drone-250：`shfiles/record.sh` 用 `rosbag record --tcpnodelay` 显式列话题（规划器状态、VINS 里程计、`/camera/infra1/image_rect_raw`、`/position_cmd` 等），没有自定义 schema；调试靠 PlotJuggler + PX4 日志（readme 第八章）。这是“原始流 + 工具回放”，不是面向训练的数据格式。
- Elastic：`src/quadrotor_msgs/msg/ReplanState.msg` 把一次重规划的全部上下文打成一个消息：`state`、`iniState`、`target`、`occmap`、`replan_stamp`；`play_bag_node.cpp` 在急停状态把该消息二进制写到 `debug/replan_state.bin`，用于离线复现。这是“记录决策状态和输入，而不只是输出”的好例子。
- Fast-Racing：只有赛道二进制/`settings.json`，没有飞行日志 schema。

**结论/可抄的思想：**

- 我们的 `actions.jsonl` + `oracle.jsonl` 分离已经比他们的 rosbag 更适合训练；缺的是：
  1. **原始深度没有落盘**，教师标签无法离线重算（换阈值/换策略就得重采）；
  2. 没有 run 级元数据（world 哈希、gz 版本、配置哈希、标定版本、时钟源）；
  3. 没有每步的配对残差、丢帧数、safety 状态。
- 建议字段组织（不改变冻结的 `EpisodeStep`，只加 manifest/旁路文件）：
  - `episode_manifest.json`：`schema`、`episode_id/scene_id/seed/split/version`、`world_sha256`、`gz_version`、`camera{width,height,fx,fy,cx,cy,distortion,depth_scale,source,calib_id}`、`config_sha256`、`clock_source`；
  - `actions.jsonl`：现有字段 + `depth_rgb_gap_ns`、`rgb_pose_gap_ns`、`pairing_rule`、`dropped_frames`、`safety_state`；
  - `sensors/` 或 `depth_%05d.bin`：原始深度（float32 或 float16 + scale），只给教师/离线分析，不进学生输入；
  - `oracle.jsonl`：保持只放真值，不和学生输入混。
- 记录粒度：**原始传感器流和派生标签分开写**；标签可以重算，原始流不能。

### Q5 嵌入式部署：量化、内存、延迟预算

**他们有什么（证据）：**

- **三个仓库都没有 TensorRT / ONNX / INT8 / 量化 / 内存预算 / NPU 的工程实现**。Fast-Drone-250 的 `drone_detect` 是深度图几何检测，不是神经网络；没有 YOLO/TensorRT 代码。Elastic 仓库内也没有实现；唯一痕迹是 `src/detection/target_ekf/launch/target_ekf.launch` 把 `~yolo` remap 到 `/yolov5trt/bboxes_pub`，说明它预期一个外部的 YOLOv5-TensorRT 检测节点，但该节点不在仓库里，**无法读到量化/内存/延迟的具体做法**。Fast-Racing 只有轨迹优化的 CPU/GPU 双版本。
- Fast-Racing：`src/plan_manage/CMakeLists.txt:10` 默认 `set(ENABLE_CUDA false)`，README §5 说 CPU 版优先兼容，改 true 用 CUDA 并行优化；显式 `-gencode arch=compute_70,code=sm_70`，链接 cublas。这是“同一算法两套后端 + 显式架构”的组织方式，不是量化经验。
- Fast-Drone-250：机载电脑是 NUC（readme 第五章），Q&A 明确说“NX 的 CPU 算力很差，跑 CPU 版 VINS 一定跑不动，要换 GPU 版 VINS”（readme Q&A）；`shfiles/sys.sh` 把 CPU governor 设为 performance，`checkcpu.sh` 看频率；readme 建议不要在机载电脑上开 rviz。
- 延迟预算：VINS 30 Hz 相机、跟踪结果 10 Hz、`max_solver_time 0.04`、`max_num_iterations 8`；规划器 100 Hz 主循环 / 20 Hz 地图；px4ctrl 100 Hz；realsense 30 Hz。各阶段频率解耦，没有“全链路同一帧率”。

**结论/可抄的思想：**

- 没有现成量化经验可抄；**不要假设 LicheePi NPU 支持某种格式/FPS**（CLAUDE.md 已写明未确认）。
- 可抄的是“**每阶段显式预算 + 超时降级 + 双后端对拍**”：
  - 给相机、预处理、推理、动作量化、UART、ESP32 控制环各写 p50/p95 预算；推理超过预算就丢帧并输出 `SAFE_NEUTRAL`，同时计数；
  - CPU/CUDA 后端已有 parity 测试（`src/tests/`），NPU 后端到位后接同一套对拍，不要另起一套；
  - 机载端关 GUI、设 performance governor、绑核/绑线程，这些是 Fast-Drone-250 的实战经验；
  - 先在本机 2080 Ti 测 p50/p95，再按算力比例给 LicheePi 留 3–5 倍余量，不要拿 2080 Ti 的数字直接当部署预算。

### Q6 工程结构：CMake、参数、调试工具

**他们怎么做（证据）：**

- CMake：catkin/catkin_simple + `cmake_utils`，每个 package 一个 `CMakeLists.txt`；`Release`、`-O3 -Wall -g`；`find_package(catkin COMPONENTS ...)`；`catkin_package(INCLUDE_DIRS LIBRARIES CATKIN_DEPENDS ...)`；CUDA 用 `ENABLE_CUDA` 开关 + 显式 gencode。`px4ctrl/CMakeLists.txt`、`plan_manage/CMakeLists.txt` 是典型。
- 参数：YAML 用 `<rosparam command="load" file="...">` 加载；launch 用 `<arg>` + `<include>` 组合实验参数；`advanced_param_exp.xml` 把 fsm/grid_map/optimization 的参数集中在一处；相机、控制器各有独立 yaml（`camera.yaml`、`ctrl_param_fpv.yaml`、`D435.yaml`）。
- 调试：`record.sh` 录 rosbag；PlotJuggler 看曲线；rviz 看地图/轨迹；`rostopic hz/echo` 查频率；`sys.sh` 调 governor；`checkcpu.sh` 看 CPU 频率。
- 反例：`shfiles/rspx4.sh` 用 `sleep 10` 等 realsense/mavros 启动，顺序和时序都很脆；我们不要抄固定 sleep。

**我们怎么用：**

- 不引入 ROS/catkin；保留现有 `src/CMakeLists.txt` + CTest 的组织。
- 新增 `config/sim.yaml`（话题名、相机话题、配对容差、超时、hold、是否存深度）和 `config/camera.yaml`（内参/畸变/深度尺度/外参/来源）；`collect.py` 加 `--config`，默认读这两个文件。
- 新增 `sim/tools/run_collect.sh` 作为唯一入口（source env、检查 `gz topic -l`、启动采集），替代散落的命令行。
- 新增 `tools/inspect_episode.py`（纯 stdlib）：读 manifest/actions/oracle，打印每步配对残差、丢帧、safety 状态、动作分布；比开 rviz 更适合我们的 headless 环境。
- 固定 sleep 改成“轮询话题就绪”（`collect_parallel.py` 已有 `_topic_list` 的思路，复用它）。

## 3. 值得借鉴的做法（按价值排序）

> 所有条目都只取“思想/结构”，不复制代码、YAML 文本或注释。许可证风险列写的是“若复制代码”的风险；本报告的建议本身是独立实现。
> GPL-3.0 是 copyleft：把 Elastic/Fast-Racing 的代码并入或链接进我们的仓库，会要求整体按 GPL 分发；Fast-Drone-250 顶层无 LICENSE 且 readme 禁止商用/转载，连代码片段和注释都不要复制。

| # | 做法 | 我们怎么用 | 工作量 | 来源 | 许可证风险 |
|---|---|---|---|---|---|
| A1 | 采集 I/O 常驻化 + 环形缓冲 + 按 stamp 配对 + latest-wins | 重写 `collect.py` 的读取层；每话题一个常驻 `gz topic -e` 线程；步骤只认“新 RGB 帧”，深度/位姿取最近 stamp；容差外重试/丢弃并计数 | 1–1.5 人日 | FD `pcl_render_node.cpp`、`grid_map.cpp`；ET `mapping_nodelet.cpp` | FD 最高（无顶层 LICENSE + 禁止转载）；ET 高（GPL-3.0）；本建议仅思想 |
| A2 | 所有等待/超时用 sim-time，不用 wall-clock | `collect.py` 的 `hold` 改为等 `/clock`（或 pose stamp）前进 `hold` 秒；动作窗口用 sim 时间记录；wall 时间只作诊断 | 0.5 人日（含在 A1） | FD px4ctrl/EGO 全部用 `ros::Time`/timer；VINS td 是反例参考 | 同上；仅思想 |
| A3 | 相机参数“一相机一 yaml + 来源/单位字段”，内参离线固定、外参持久化 | 新建 `config/camera.yaml`；manifest 填 fx/fy/cx/cy、depth_scale、source、calib_id；仿真先用 SDF hfov，再用 `/uav/camera_info` 校验 | 0.5 人日 | FD `left.yaml`/`fast_drone_250.yaml`、`intrinsic_calib.cc`；ET `camera.yaml` | FD 最高；ET 高；仅思想 |
| A4 | 显式安全状态机 + 每个输入一个新鲜度谓词 + 明确的降级动作 | 写 `docs/` 安全规范；在 `sim/tools/collect.py` 的 actions/manifest 里加 `safety_state` 和 stale 注入测试；ESP32 固件实现指令超时/零速/降落 | 0.5 人日文档+仿真；ESP32 1–2 人日 | FD `PX4CtrlFSM.cpp`、`ego_replan_fsm.cpp`；ET `traj_server.cpp` | FD 最高；ET 高；仅思想 |
| A5 | 失效三级降级：停止/保持 → 悬停 → 降落；不是单一 neutral | 在 `docs/` 安全规范和 `sim/tools/teacher.py`/`collect.py` 里定义 `SAFE_NEUTRAL` 的语义为“零速度 + 高度保持”；stale 超时后升级到悬停/降落；采集器深度失效仍终止 episode（数据质量优先） | 0.5 人日（与 A4 合并） | FD px4ctrl AUTO_HOVER/AUTO_LAND；EGO EMERGENCY_STOP；ET 重规划失败 hover | 同上 |
| A6 | 记录“原始传感器流 + 决策状态 + 可重放”，不只记输出 | 在 `sim/tools/collect.py` 保存原始深度；`episode_manifest.json` 加 world/config/calib 哈希；`actions.jsonl` 加 pairing residual/dropped/safety_state；新增 `tools/inspect_episode.py` | 0.5–1 人日 | FD rosbag 话题清单；ET `ReplanState.msg` + `play_bag_node.cpp` | FD 最高；ET 高；仅思想 |
| A7 | 传感器有效性门控：超时停预测、异常测量拒绝 | 在 `sim/tools/collect.py` 里，深度/位姿配对残差超阈值就拒绝本步；EKF 的“速度 >4 m/s 拒绝更新”思想用于未来策略输出合理性检查（`src/` 推理接口） | 0.5 人日 | ET `target_ekf_node.cpp`；FD `grid_map.cpp` timeout | ET 高；FD 最高；仅思想 |
| A8 | 每阶段显式频率/延迟预算 + 频率监控 | 新增 `docs/` 延迟预算表；`sim/tools/collect.py` 记录每步读/算/发耗时；部署端（ESP32/推理进程）加 deadline/drop 计数 | 0.5 人日文档；代码后续 | FD VINS 配置、planner timer、realsense FrequencyDiagnostics | FD 最高；仅思想 |
| A9 | CPU/GPU 双后端 + 显式架构 + 对拍 | 保留 `src/CMakeLists.txt` + `src/tests/` 的 CPU/CUDA parity；NPU 后端到位后接同一套测试；CMake 继续强制 sm_75 | 现有；NPU 3–5 人日（未知） | FR `plan_manage/CMakeLists.txt`、README §5 | FR 高（GPL-3.0）；仅思想 |
| A10 | 按 package 组织 CMake + YAML 参数 + 启动脚本 + 离线检查工具 | 保留 `src/CMakeLists.txt`；新增 `config/*.yaml`、`sim/tools/run_collect.sh`、`tools/inspect_episode.py`；不引入 catkin | 0.5 人日 | FD `px4ctrl/CMakeLists.txt`、`advanced_param_exp.xml`、`record.sh` | FD 最高；仅思想 |

## 4. 不适用的（以及为什么）

- **ROS/catkin/message_filters/rosbag/launch/rviz/MAVROS/PX4 offboard**：我们明确不用 ROS；这些是他们的运行时基础设施。只取“同步/状态机/参数组织”的思想。
- **VINS-Fusion 整套 VIO**：单目+IMU、双目、特征跟踪、在线 td/外参、回环；我们仿真有真值位姿，策略是单目 RGB 端到端四动作，不需要 VIO。真机未来是否需要 VIO 取决于 ESP32/定位方案，目前未确认。
- **EGO-Planner / Elastic 的 MINCO/SE(3) 轨迹优化、占据栅格建图**：我们输出四个离散高层动作，不做连续轨迹规划；这些代码量大、GPL、和我们的接口不匹配。
- **Fast-Racing 的 AirSim/UE4 仿真与 AirLib**：和 Gazebo Harmonic 不兼容，体量大；CUDA 代码是轨迹优化，不是 CNN，不能当部署参考。
- **Fast-Racing 的 CtrlFSM**：主体被注释，没有实际失效处理；不能当安全范例。
- **px4ctrl 的 0.5 s 超时值和电池处理**：0.5 s 对 30 Hz 深度太粗；电池只打日志、不自动降落；数值不能照抄。
- **VINS 的在线外参自标定流程**：需要 IMU/VIO 激励；我们没有这套传感器和融合，暂不适用。
- **他们的 rosbag/PlotJuggler/rviz 工具链**：我们不能用 ROS；用 JSONL + 纯 stdlib 检查工具替代。
- **照抄 YAML 数值**：D435 的 640×480、fx≈384、`depth_scaling_factor=1000` 都不适用于我们的 320×240 仿真相机（fx≈277、深度单位米）。
- **固定 `sleep 10` 的启动脚本**：脆弱；用话题就绪轮询。
- **`Fast-Drone-250` 的任何代码/文本**：顶层无 LICENSE + readme 禁止商用/转载，风险最高；只读思想，不复制。

## 5. 三个痛点的直接建议

### P1 采样慢（1.7 s/步）

**根因（现有 worklog 已测）：** 每步 spawn 多次 `gz topic -e`（深度、RGB、动作前位姿、动作后位姿、接触监听），每次进程启动 + 发现 publisher 有固定开销；`gz topic -p` 发布 twist 还有约 0.8 s 的固定延迟。真正的动作窗口只有 0.5 s。

**改法（按投入分两级）：**

- **Tier 1（0.5–1 人日，纯 stdlib）：**
  - 在 `collect.py` 启动时起常驻订阅：`/uav/camera`、`/uav/depth`、`/model/x3/pose`（或 `/model/x3/odometry`）、`/clock`、`/uav/contact`；每个一个线程，逐行解析 JSON，保留最近 2–3 帧。
  - 每步：等一帧新 RGB → 取 stamp 最接近的 depth/pose（容差如 1/30 s）→ 教师 → 发 twist → 等 `/clock` 前进 `hold` → 发零速 → 取动作后 pose。
  - 接触监听常驻，动作窗口内看是否收到消息，不再每步 `-n 1`。
  - 保留 `collect_parallel.py` 作为吞吐兜底，但它不解决单实例的 1.7 s。
  - **预期**：读侧 spawn 开销消失，单步大概降到 ~0.9–1.0 s；twist 的 0.8 s 还在。不要承诺 ~0.1 s。
- **Tier 2（2–3 人日，C++ 小桥接）：**
  - 写一个 `sim/tools/gz_bridge`（C++，链接 Gazebo transport，不碰 ROS）：订阅相机/深度/位姿/clock/contact，发布 twist；对 Python 暴露行协议（stdin 动作、stdout 观测）。
  - 这样发布也没有进程启动延迟，目标 <0.1 s/步。
  - 已验证：本机 Python 没有 `gz.transport13`/`gz.msgs13` 模块（`ModuleNotFoundError`），所以纯 Python 不能直接 gz-transport；要么 apt 安装绑定（未验证可用性），要么写 C++ 桥。

**注意：** 常驻 `gz topic -e` 是否按行输出 JSON、是否缓冲，未实测；如果不行，直接上 Tier 2。

### P2 时间戳不对齐（168–308 ms、四次独立快照）

**改法：**

1. **统一采样时刻**：每步以“新 RGB 帧”为锚点，深度/位姿取 stamp 最接近的帧；记录 `rgb_pose_gap_ns`、`depth_pose_gap_ns`、`pairing_rule`；超过 1 帧（33 ms）就重试或丢弃，不要静默使用。
2. **动作前位姿必须是锚点附近的位姿**：现在的 `pose_before` 是 RGB 之后另读一次，因果链上可能已经晚了几百 ms；改成“与 RGB 最近的 pose”。
3. **同帧保证（仿真层）**：把 SDF 里分开的 `camera` + `depth_camera`（`sim/worlds/uav_rgb.sdf:160-201`）改成单个 `rgbd_camera` 传感器；一次渲染同时出 color/depth/camera_info，stamp 天然相同。**需现场确认 Harmonic 8.15 的话题命名**（`gz topic -l`），确认后再改采集器。工作量 0.5 人日 + 验证。
4. **动作时长用 sim-time**：发 twist 后记录当时的 `/clock`（或 pose stamp），等 sim 时间前进 `hold` 再发零速；记录 `sim_dt` 和 `wall_dt`，wall 只作诊断。这样低 RTF/卡顿时动作窗口不会漂。
5. **不要照抄 message_filters 的队列 100**：30 Hz 下队列 100 是 3 秒积压；我们用 2–3 帧 + latest-wins，旧帧直接丢并计数（Elastic 的 atomic lock 思想）。

### P3 无标定（fx/fy/cx/cy 全是 null）

**改法：**

1. **仿真先落地**：新建 `config/camera.yaml`：
   - `width: 320, height: 240`
   - `model: pinhole`, `fx: 277.128, fy: 277.128, cx: 160, cy: 120`
   - `distortion: {k1: 0, k2: 0, p1: 0, p2: 0}`
   - `depth_scale: 1.0`（仿真深度是 float32 米；真机 D435 才是 1000）
   - `source: sdf_hfov`（若读到 camera_info 则 `source: camera_info`）
   - `cam2body`: 仿真先写 SDF 的 `pose 0.2 0 0`，但**坐标系约定待实测**；策略暂时不用。
2. **校验**：`gz topic -e -t /uav/camera_info -n 1 --json-output` 读 K（README 话题表列了它）；和公式值比，误差应 <1 px。再放一个已知世界点，投影像素和图像实际像素比，误差 <1 px。
3. **manifest 不再写 null**：在 `collect.py` 的 manifest 段填 `camera{...}`，加 `camera_source`、`calib_id`、`depth_scale`。
4. **真机以后**：12×8、80 mm 棋盘格，拍 30–50 张，OpenCV `calibrateCamera`（C++ OpenCV 4.6 已装；`.venv-tools` 没有 cv2）；输出 YAML 用 VINS 的 `PINHOLE` 结构（`image_width/height`、`distortion_parameters`、`projection_parameters`）；外参用 CAD/实测 4×4，投影校验后固化。工作量：仿真 0.5 人日；真机 1 人日。
5. **不要**把 VINS 的在线 td/外参自标定搬过来：我们没有 IMU/VIO 激励，暂时不适用。

## 6. 做了什么 / 没做什么 / 验证到哪一步 / 风险

### 做了什么

- 静态阅读了三个仓库的目录结构、许可证、readme、关键源码和配置（详见第 2 节证据路径）。
- 核对了 Fast-Drone-250 顶层无 LICENSE、readme 禁止商用/转载；Elastic/Fast-Racing 为 GPL-3.0；子包另有 GPL/BSD/Apache。
- 核对了我们自己的 `sim/tools/collect.py`、`teacher.py`、`sim/worlds/uav_rgb.sdf`、`sim/README.md`、`config/`、`src/include/uavrgb/episode.hpp`，把建议落到具体文件和函数。
- 检查了本机 Python 的 `gz.transport13`/`gz.msgs13` 可用性：**不存在**（`ModuleNotFoundError`）。
- 写了本文。

### 没做什么（含原因）

- **没有运行/编译任何被调研仓库**：任务要求只读，且这些是外部下载的不可信代码；原因：安全 + 范围限制。
- **没有动态验证 `/uav/camera_info`、常驻 `gz topic -e` 的行分帧、`rgbd_camera` 话题名**：需要起 Gazebo 实测；原因：本次是静态调研，未获准运行仿真。文档里已标“待确认”。
- **没有复测 1.7 s/步、168–308 ms**：引用现有 worklog/`docs/results.md` 的实测值，未重新跑；原因：同上。
- **没有读 VINS-Fusion 的 estimator/loop_fusion/global_fusion 全部源码**：只读了配置、标定工具入口和 readme；原因：与我们的端到端四动作策略关系弱，时间有限。
- **没有读 EGO-Planner 的优化器/建图内部、Elastic 的 traj_opt/prediction、Fast-Racing 的 SE(3) 优化器和 AirLib**：只读了安全/同步/CMake/参数相关片段；原因：这些算法不适用于我们的接口，且 GPL 代码不能复制。
- **没有读 `3rd_party.zip`、PX4 `.px4` 固件、PDF、xlsx、图片**：二进制/非代码资料；原因：与本次问题无关。
- **没有读 realsense-ros 的全部驱动内部**：只 grep 了频率诊断、深度尺度、camera_info 使用；原因：我们不用 realsense。
- **没有评估 LicheePi NPU / ESP32 的具体能力**：仓库里没有这些硬件资料；原因：硬件型号/工具链未确认，不能假设。
- **没有读同目录下的 `ego-planner`、`ego-planner-swarm`、`EGO-Planner-v2`、`GCOPTER`、`Implicit-SDF-Planner`、`Swarm-Formation`**：不在本次指定的三个仓库范围内；原因：范围限制。

### 验证到哪一步

| 项 | 结果 |
|---|---|
| 三个仓库静态阅读 | 完成（关键文件见第 2 节） |
| 许可证核对 | 完成：FD 顶层无 LICENSE + readme 禁止转载；ET/FR GPL-3.0；FD 子包 GPL/BSD/Apache |
| 运行/编译被调研仓库 | 未跑（任务要求只读 + 不可信代码） |
| 本机 Python `gz.transport13`/`gz.msgs13` | 已验证不存在（`ModuleNotFoundError`） |
| `/uav/camera_info` 是否实际发布 | 未验证（`sim/README.md` 话题表列出，SDF 未见显式配置） |
| 常驻 `gz topic -e` 是否按行输出 JSON | 未验证 |
| `rgbd_camera` 在 Harmonic 8.15 的话题名/stamp | 未验证 |
| 1.7 s/步、168–308 ms | 引用现有 worklog，未复测 |
| 相机 hfov→内参公式 | 数学推导，未做投影实测 |
| LicheePi NPU / ESP32 | 未验证（无资料） |

### 风险与遗留

- **静态结论**：所有性能/行为判断来自读代码和现有 worklog，没有动态复现；上线前必须用仿真注入延迟、丢帧、传感器失效来验证。
- **常驻子进程的坑**：`gz topic -e` 可能缓冲/不按行输出；线程退出、pipe 阻塞、gz server 重启都要有 watchdog；否则采集器会挂死。
- **同帧不保证**：即使常驻订阅，两个独立 sensor 的 stamp 仍可能差一个渲染 tick；`rgbd_camera` 是更彻底的修法，但改 SDF 会动话题名，需同步改采集器和 README。
- **安全阈值是草案**：200/300/500 ms、2 s 降落等都要用仿真标定；不要直接写死到 ESP32。
- **许可证**：Fast-Drone-250 风险最高（无顶层 LICENSE + 明确禁止商用/转载）；Elastic/Fast-Racing 是 GPL-3.0。任何代码/配置文本都不要复制，保持 clean-room；如果未来要开源我们的代码，建议请人确认。
- **标定**：hfov 公式假设理想针孔和像素中心约定；`/uav/camera_info` 若给出不同值，以它为准；外参坐标系未实测。
- **硬件未知**：LicheePi NPU 的算子支持、量化格式、内存、ESP32 的失控保护/指令超时能力都未确认；不要把本报告的预算当承诺。

## 7. 明确没读的部分（一句话清单）

- VINS-Fusion：estimator/loop_fusion/global_fusion 的算法源码（只读配置 + 标定入口）。
- EGO-Planner：bspline_opt/path_searching/plan_env 的优化与建图内部（只读 grid_map 同步/超时、FSM 安全段、EmergencyStop）。
- Elastic-Tracker：traj_opt/prediction/visualization 的算法源码（只读 mapping 同步、target EKF、traj_server 心跳、ReplanState）。
- Fast-Racing：SE(3) 优化器、AirLib、MavLinkCom、airsim_ros_pkgs 内部（只读 README、CtrlFSM、Ctrl 配置、plan_manage CMake、cuda_computer 头）。
- realsense-ros：驱动内部（只读频率诊断、深度尺度、camera_info 相关 grep）。
- Fast-Drone-250：`3rd_party.zip`、PX4 固件、PDF/xlsx/图片、大部分 launch 和 CMake。
- 同目录其他仓库：ego-planner、ego-planner-swarm、EGO-Planner-v2、GCOPTER、Implicit-SDF-Planner、Swarm-Formation。
- 硬件：LicheePi NPU、ESP32 飞控、真实相机型号与安装方式。
