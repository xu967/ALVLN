# 2026-11-04 仿真真实性 / 可视化 / 训练评测分路 只读审计

## 做了什么

- 只读审计 `uav_rgb/` 主线（忽略 `ALVLN/uav_rgb` 镜像），并行 9 个 subagent：
  动力学闭环、可视化/UI、场景资产、真机 CF-Drone 对照、Git/文档、训练分路、验证/评测分路、
  浙大高飞团队 GitHub、中科大团队与高保真仿真栈。
- 检查本机环境：Gazebo Harmonic 8.15、RTX 2080 Ti、12 核 cgroup、磁盘剩余约 741 MB。
- 抽取现有 `cfstab-video-20261104` mp4 帧，核对 CHASE/TOP/ONBOARD 实际画面。
- 未启动 Gazebo、未训练、未改代码、未推送 GitHub。

## 结论（按严重度）

### P0-1 当前仿真不是用户的真机

- 仿真载具是 Gazebo 官方 X3：1.5 kg、0.47 m 机身、桨尖约 0.32 m
  （`sim/worlds/uav_rgb.sdf:176-188,271-289`）。
- 真机是琛光 E1 系列 118 mm 微型四轴，ESP32 + MPU9250，1 kHz IMU；
  只有 STAB/ACRO/RAW/AUTO，STAB 为 ±30° 姿态杆 + 0~1 油门；
  ALTHOLD 是 stub，没有速度环/位置环/定高（`CF-Drone/control.ino:13,42-53,154-167,327-331`）。
- 仿真用 `MulticopterVelocityControl` 跟踪机体系速度，并用 Gazebo 真值位姿做 P 定高
  （`sim/worlds/uav_rgb.sdf:468-483`；`sim/tools/collect.py:98-101,129-136,178`）。
- `cf_stab` 是 Python 侧速度曲线（2° 倾角、阻力 1/s、0.15 s 延迟，全部占位），
  底层仍是 X3 完美速度环，不是 CF 飞控（`sim/tools/collect.py:139-217`）。
- 实测后果：x3 teacher 0.94 / student 0.59；cf_stab teacher 0.65 / student 0.06；
  cf_stab 下 GT 教师自己撞 6/17（`docs/experiments/2026-11-04-cf-stab-dynamics.md:55-84`）。
- 最深问题：真机没有速度/位置闭环，仿真假设了一个完美速度环；4 动作的
  `FORWARD_SLOW=0.3 m/s` 在真机上没有执行器。必须补“动作→姿态/油门”的
  伴飞控制器与状态估计（光流/ToF/外测），或改成开环姿态脉冲并重训。

### P0-2 可视化会把结论看错

- `classify_episode` 先判 collision 再判 false_forward，而评测里
  `false_forward=True` 必然 `collision=True`，所以“假前进”永远 0%、筛选永远空
  （`ui/studio.py:418-430`；`sim/tools/eval_closed_loop.py:192-195`；
  测试用了不可能的输入 `ui/test_studio.py:169`）。
- SR 分母是完成集；UI 只显示 `success/done`，隐藏 requested/completion/failed/skipped
  （`sim/tools/eval_closed_loop.py:234-249,622-624`；`ui/studio.py:1426-1434`）。
  实测 x3-student 17/20 完成，SR 0.588 vs requested 0.50。
- UI 把 `success_rate` 叫“闭环导航 SR”，但它是 reach-ever 且不要求 STOP
  （`ui/studio.py:824`；`sim/tools/eval_closed_loop.py:217-220`）。
- 视频把 `contact or stuck` 的第一步标成 COLLISION，stuck 也写成 COLLISION；
  goal 线使用 `max(pillar.x)+margin`，与 L1/campus 的 waypoint 判据不一致；
  goal/waypoint 用最终高度投影，campus 里常看不到（`sim/tools/record_views.py:177-183,235-239`）。
- CHASE 相机刚性挂在 base_link，后 2.5 m、上 1.2 m、下俯 25.8°，随无人机 roll/pitch 倾斜；
  不是稳定云台，也不是世界固定追尾（`sim/worlds/uav_rgb.sdf:218-234`）。
- 顶视相机 SDF 已是 6 m（`sim/worlds/uav_rgb.sdf:85`），README 仍写 18 m
  （`ui/README_STUDIO.md:57`），根目录截图也是旧的 18 m 版本。
- ONBOARD 面板写 “model input”，实际显示 320×240 原始图，模型吃 160×120
  （`sim/tools/record_views.py:202`；`tools/preprocess.py:58-62`）。
- 断开/停更后画面和 2D 地图冻结，状态仍可能显示 OK（`ui/studio.py:1747-1794,1864-1869`）。

### P0-3 环境不是真实校园

- base/l1 默认是 100×100 m 灰平面 + 8~9 根 0.4×0.4×3 m 方柱 + 3 个固定纯色方块；
  方块有 collision 但不在教师几何里，是已知捷径
  （`sim/tools/scene_gen.py:57-77,150-179,235-249`；`sim/worlds/uav_rgb.sdf:62-141`）。
- campus 是三套轴对齐 box 布局 + 128/256 px 程序化 albedo；没有 normal/roughness map、
  天空/HDRI/雾/AO/动态物体（`sim/tools/campus.py:19-25,50-95`；`sim/tools/make_campus_textures.py:12-28`）。
- campus_assets 的 mesh 是 96/84 面原创 demo，不是真实 CC0；树冠视觉 1.4 m、碰撞仍是 0.4 m box；
  `asset_catalog.json:3` 明写 OFFLINE DEMO（`sim/tools/build_demo_assets.py:1-7`）。
- 固定地标、颜色/左右捷径、三拓扑与 split 1:1 绑定、campus_assets 按 map hash 打散，
  都会让“泛化”结论不可归因（`sim/tools/campus.py:28-41`；`docs/RESEARCH_BACKLOG_2026-11-04.md:38-50`）。
- 磁盘只剩约 741 MB，真实资产/重采前必须先清 `$DATA_ROOT`（22 GB uav_rgb + 121 GB 其他数据）。

### P0-4 训练/验证分路

- 好的一面：按 episode→scene_id 隔离，C++ 硬拒同 scene 跨 split
  （`src/dataset.cpp:136-139,423`）；class weight 只用 train（`src/class_weights.cpp:51-54`）；
  test 只在训练末尾评一次且加载 best（`src/trainer.cpp:306-323`）；无归一化统计泄漏；
  当前 `.uavr` 实测 scene/gid 跨 split 交集 0。
- 坏的一面：
  1. `split_policy` 永远写 `scene_id`，诊断性 `map-hash:diag1` 也能被训练；
     run 产物不记录 split scheme / dataset SHA（`tools/preprocess.py:326,328`；
     `src/dataset.cpp:423`）。
  2. 冻结数据重打包时 split 复算不匹配：t5 35.4%、campus 35.2% 与原 manifest 不同
     （`tools/preprocess.py:125-139`；`sim/tools/split_manifest.py:105-111`）。
  3. 蒸馏间接 val 泄漏：教师用同一 val 选 best，再导出软标签；教师 ckpt 哈希未记录
     （`teacher/train.py:183-187`；`teacher/export_soft_targets.py:44-52`）。
  4. resume 不存 lr/seed/batch/drop_collision/soft_targets 等，漏传会静默换实验；
     best 分数跨 regime 继承（`src/checkpoint.hpp:21`；`src/trainer.cpp:114-138`；
     `src/class_weights.cpp:171-175`）。
  5. A/B 非逐位确定；梯度裁剪范数 double atomicAdd、shuffle/打包顺序不绑定
     （`src/nn_cuda.cu:129-134,323-340`；`src/trainer.cpp:179,188`）。

### P0-5 评测口径与统计

- `success_rate` 同时承载 route_choice / goal_reach / goal_reach_ever /
  visible_goal_reach_v1，旧产物缺 `success_criterion`；0.574 是事后手算，
  不在 `eval.json`（`sim/tools/eval_closed_loop.py:217-230,613-619`；`docs/HANDOFF.md:162-166`）。
- 主指标 goal_reach_ever 下盲前进不是 0：2/41=4.9%；“forward 0.000”只成立于末位姿口径。
- L1 训练教师目标 0.5 m 停，评测按 1.5 m 到达且不要求停
  （`sim/tools/collect.py:1099`；`sim/tools/eval_closed_loop.py:214-220`）。
- 失败/跳过不进 SR 分母，补扫 seed；`requested_seeds` 默认写整段扫描范围。
- 反事实集复用 base 帧：16184 = 8092 + 8092；val 1688 = 844+844；
  best 在 cf 混合 val 上选，test 在 base test 上报（`docs/HANDOFF.md:368-373`）。
- L1 47 集只覆盖 36 个 gid；campus 20 集只覆盖 3 张图；无 CI/配对/McNemar 代码；
  20 集配对 MDE 约 30 pp。

### 文档 / Git

- `uav_rgb` 主线 HEAD `1e0d6fc`（17:46），无 remote；GitHub `ALVLN` HEAD `3e18f63`
  （01:46），比主线落后 7 个提交、约 6.3k 行；缺 campus v2、cf_stab、资产、rl_teacher、
  Studio、record_views 等。ALVLN 本地与 `origin/main` 一致。
- `TinyAerialVLN.md` 的 SIM-01/02/03/05/06/07 标“已修”，但主线代码仍是旧逻辑，
  `test_sim_review_regressions.py` 从未存在；worklog 1210 已承认“补丁没合入”。
- README/文档过期：ALVLN 落后量、顶视 18 m、三视角“同步”、GUI/headless、
  inline test 用末轮模型等。

## 外部参考：浙大高飞团队（已核实）

- `ZJU-FAST-Lab/ego-planner-swarm`（GPL-3.0）自带 `uav_simulator`：
  `so3_quadrotor_simulator` 是 ROS1 6DOF ODE，含质量/惯量、推力/反扭矩、
  电机一阶滞后、二次阻力；`so3_control` 是位置/速度 → SO3 姿态 → 电机混控。
  **不是 Gazebo，不能直接接；可借鉴公式与架构，代码需重写。**
- `ZJU-FAST-Lab/Fast-Drone-250`：真机平台（1.2 kg、30°、100 Hz、PX4/MAVROS），
  但其 `single_run_in_sim.launch` 用 `poscmd_2_odom` 假无人机，不是动力学仿真；
  顶层无 license 且 README 禁止商用/转载，只能参考参数。
- `ZJU-FAST-Lab/Ground-effect-controller`：给出真实平台的 kT/kI/J_R/阻力/惯量/
  控制增益，可作为系统辨识结构参考（GPL-3.0）。
- 该组织没有 aerial VLN 仓库；`FastSim` 只有 README，未发布。
- 结论：最接近我们需求的是“gz 物理 + Python SO3 内环 + motor_speed”这一层，
  不是搬 ZJU 的 ROS1 仿真器。高飞团队的做法是“仿真做规划、真机做验证”，
  不是靠照片级仿真。

## 外部参考：中科大（已核实）

- 没有找到“中科大官方 + 无人机 VLN”的 GitHub 仓库。最接近的是
  `USTC-AIR-Lab/PE-Planner`（Gazebo classic + PX4 SITL，动态环境规划），
  以及 `SI-LIO`、`RAFT-VINS`；均无 VLN，且 PE-Planner/SI-LIO 未检测到 LICENSE。
- 中科大在 VLN 的公开代表是 AdaNav（室内/地面，非无人机）；NavFoM 代码未发布；
  USTC FLICAR 是高空作业车数据集，但它的标定/时间同步清单可迁移。
- 可借鉴：PE-Planner 的“扰动只加在仿真层、规划器不可见”+ 批量重复评测；
  RAFT-VINS 的前端/后端解耦与真实数据回放；FLICAR 的相机-IMU 标定检查表。
- 结论：不能把“中科大团队”当成一个现成的无人机 VLN 仿真栈；更多是方法参考。

## 外部参考：仿真栈（已核实）

- 只换世界/外观的最低成本路径：Gazebo Harmonic + Gazebo Fuel/PBR 模型；
  逐模型核对 license，不要批量下载。
- PX4-gazebo-models 的 x500/x500_depth（BSD-3-Clause，整仓约 64 MB）可只取
  SDF/网格，不装 PX4 SITL/ROS 2；但 x500 是 2.0 kg，仍不是 118 mm 真机，
  只能当“更接近真实四旋翼”的参考，不是数字孪生。
- 不建议：完整 PX4 SITL + ROS 2（GB 级、控制入口变了）、AirSim/Colosseum
  （UE5、几十 GB、Colosseum 已归档）、Flightmare（Unity）、RotorS（Gazebo classic）、
  ZJU Fast-Drone-250（无 license 且禁止商用）。
- 3DGS/NeRF 只能离线生成纹理/网格，不提供碰撞，也不能直接接 gz 传感器。
- 结论：**视觉环境替换和载具动力学是两条线**。只换世界可以直接做；
  要让载具像 118 mm 真机，仍应走 `cf_sil_v1`，不要用 x500 代替。

## 最小改造顺序（建议）

1. **载具/内环**：保留 Gazebo Harmonic；保留 `MulticopterMotorModel`，
   移除 `MulticopterVelocityControl`；用 CF-Drone 的串级 PID/混控/限幅生成
   `motor_speed`；参数来自台架/外测；新增 `--dynamics cf_sil_v1`，不要动
   collect/preprocess/train/eval 接口。
2. **教师/数据**：先修 `plan_action` 的延迟+滑行预测；cf_stab teacher ≥0.9 后再采数据；
   不要把碰撞写进标签。
3. **指标/可视化**：先修 false_forward 归类、SR 分母/口径标签、视频 stuck/goal；
   再修 stale/顶视/ONBOARD 标注。
4. **环境**：真实 CC0 资产 + PBR albedo + 天空/光照；保留 box collision；
   去固定地标；几何族扩展；split 正交化；光度域增强放 preprocess。
5. **统计**：固定 seed 清单、requested/completion/failed、CI/配对、test 评估锁。

## 没做什么

- 没启动 Gazebo 实跑新的动力学；没训练/重采；没下载真实资产；没改代码；
  没验证真机参数；没推送 GitHub。
- 浙大/中科大/GitHub 调研由 subagent 进行，结论以最终回复为准。

## 风险

- 真机质量、惯量、桨/电机、推力曲线、控制延迟均未实测；`cf_stab` 参数是占位值。
- 真实资产许可/可得性、gz-sim8 对 normal/roughness map 支持有限。
- 旧数据/旧 checkpoint/旧 SR 不可与新协议混报。

## 实施记录（同日 21:xx，Phase 2）

用户拍板：四项全做；真机路线选 C（先只做仿真研究）；有完整参数/日志待放入
`$DATA_ROOT/uav_rgb/real-drone/`；允许清盘、重采/重训、改代码和推送 GitHub。

已完成并验证：

- 代码提交（uav_rgb）：
  - `c0f77c2` eval/UI：sim09 指标、false-forward、视频 contact/stuck、STALE。
  - `5884789` sim：cf_sil 骨架、默认 v1 外观、可移植资产。
  - `03ee62e` data：split 可审计、冻结重打包、诊断 split 门禁。
  - `4314961` trainer：checkpoint v5、严格 resume、蒸馏来源、视频回归。
  - `6241c3f` docs：审计/交付文档。
- 测试：
  - `bash tools/check.sh` 全量 PASS；CTest 25/25；Python 全套 PASS。
  - `cf_sil` 真实 Gazebo 冒烟：起飞 z=2.13 m，3/3 步采集成功；
    `/tmp/cf_sil_smoke3`。
  - `cf_sil` eval + 录像冒烟：`/tmp/cf_sil_eval/videos/seed_1.mp4`。
  - `teacher/train.py --record-video` 命令回归通过；
    真实 teacher-nn 1 集视频保存到
    `$DATA_ROOT/uav_rgb/teacher-video-check-20261104/videos/seed_1.mp4`
    （该集 1 步 stuck，只证明保存链路，不是质量 demo）。
- 关键修复：cf_sil 需要从 world 删除 `MulticopterVelocityControl`，否则其
  Actuators component 会覆盖 transport 电机指令；vm 无 gz 时用
  `/usr/bin/python3` helper 发布。
- GitHub：`ALVLN` 镜像提交 `5a673d5` 已 push；`origin/main` 一致。

未完成/待用户：

- 真机 `.ulg` 与 BOM 尚未放入 `real-drone/`，cf_sil 仍 `calibrated=false`，
  固件 PID 直接搬到 250 Hz 未在 Gazebo 调参。
- 真实 CC0 资产未下载；磁盘盘点完成但未删除，数据盘 88G 可用、根分区约
  659M；是否删 730M 安全项/3.5G pubdata/977M cudnn9 需用户确认。
- 未重采/重训正式数据；旧 SR 与新 metric_version 不可混报。
- checkpoint v5 旧二进制不兼容；`src/dataset.cpp`/`src/trainer.cpp` 超过
  280 行文件约束，需后续拆分。
