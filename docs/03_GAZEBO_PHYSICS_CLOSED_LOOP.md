# 03｜Gazebo Harmonic 四旋翼动力学 + 前视 RGB + 导航策略闭环实现蓝图

## 0. 本节的问题定义

V7.1 的 `scene_tool.py` 仅输出静态柱阵/地面 SDF；没有飞行器的 dynamic link、4 电机插件、相机 sensor、控制桥。**当前 V8 `tools/scene_generator.py` 也只生成静态场景候选**，GZ0/GZ1 等物理实验还未执行。下面是明确的逐模块实施工单，不是已运行的事实。

## 1. 选择执行顺序：不要从 CF `.ino` 开始

**第一条真实物理基线**优先选 Gazebo Harmonic（gz-sim 8）与 PX4 官方 `gz_x500` 或官方 X3 四旋翼例子。原因：它们有成熟的动力学/控制示例，能先消除“是否因为飞控姿态不可用导致坠机”这个变量。不能因为使用 PX4 就声称 CF-Drone 控制律已经验证。

**第二条独立 CF-Drone SIL**：对 CF-Drone 自有 PID/混控公式做 CPU 数值 golden，对比 Flix 开源项目可能已有 Gazebo Classic/Arduino shim 实现；先做源码 diff/许可证核实，再决定抽离数学核还是重用上游。**此阶段不修改 ESP32 固件**。不要复制 X3 的质量、转动惯量、旋向和电机系数到 118mm 微型 CF。

## 2. 一条合格的真实闭环架构

```text
Gazebo world + semantic landmarks + physical obstacles
                   |
          dynamic quadrotor
  4 motors -> thrust/torque -> 6-DoF body state
                   |
        forward-facing RGB sensor
                   |
    CameraBridge (timestamp, intrinsics, encoding)
                   |
 StudentAdapter(I_t, instruction, prior actions, memory)
                   |
    PolicyIntent (bounded command + TTL + action ID)
                   |
     ControllerAdapter (PX4 SITL at first)
                   |
         physically executed command
                   |
          next camera observation
```

数据契约：学生输入 `Observation` 仅 RGB+指令+过去动作/状态；教师/监督侧可读取深度、位姿、物体地图与速度。Gazebo 发布的 Ground Truth 永远通过带权限的 `OracleChannel`，不得进入 student buffer。

## 3. Gazebo GZ0：验证原生动力学（不接视觉网络）

**目标**：`gz sim` 真运行，四电机指令改变物理姿态及位置。

工单：
1. 按官方指南安装 Harmonic 与 PX4。固定版本与 commit，不使用 `gazebo11` 插件文件名加载到 gz-sim8。
2. 使用官方 `quadcopter.sdf` 或 `make px4_sitl gz_x500` 起步，记录是采用官方 X3 还是 x500（两者是不同机型）。
3. 不改模型控制器，执行手动 takeoff/hold/land（由安全的 SITL 指令），收集真实 `pose`、`twist` 和电机命令；保留 stdout/stderr。
4. 核对 `/clock` 仿真时间、机体重置、地面接触碰撞。引擎无法工作则标记 `GZ_RENDERER_BLOCKED`，不生成“已飞行”结果。

**静态预检不足**：四个 `MulticopterMotorModel` 需要检查 `actuator_number ∈ {0,1,2,3}`、唯一性、转向、`jointName`/`linkName` 指向真实链接、机体为 dynamic、SDF include 递归展开。静态检查是必要条件；电机真正旋转且能产生运动是动态必要条件。

## 4. GZ1：安装并验证机载 RGB 相机

相机应安装在机体前向位置，明确机体坐标约定（FLU 或 FRD）、Gazebo ENU 世界系和 camera optical (right-down-forward) 之间变换。记录 `fx,fy,cx,cy,width,height,FOV,distortion`，初始 320×192 或 384×224；实际分辨率由效率和标定决定，不应预先声称与 Aerial 官方 224² 同协议。

验收：红/蓝/黄三个地标以已知位置出现在相机 RGB 对应区域；相机随机体 yaw 转动而移动；无 GUI headless 仍能收到图像；重复 reset 后 frame counter 和 timestamp 正确复位；RGB 的时间戳不晚于当前 sim time，超过可接受 age 的图像必须拒绝而不是重复投喂。

**必须留真图证据**：PNG/JPEG 首帧与最后帧、相机配置文件、RGB topic 名称、观测 header 时间戳、相机原始消息 hash、对应真实姿态轨迹。只有 SDF 的 sensor 标签没有实际消息，判为 `NOT_VERIFIED`。

## 5. GZ2：人工策略与动作闭环

创建 `NavIntent`（此处只是 Gazebo 连续轨的候选接口，不是官方 8 动作）：

```cpp
struct NavIntent {
  uint64_t seq;
  int64_t observation_sim_time_ns;
  float body_vx_mps;   // +forward; frame=FLU (must be documented)
  float body_vy_mps;   // +left
  float yaw_rate_radps;// +CCW
  float vz_mps;        // +up, only if controller has verified vertical velocity hold
  float horizon_sec;   // bounded: start with <=0.25s candidate, not a guarantee
  bool request_terminate_episode; // Task STOP, not motor-off
};
```

起步用 PX4 接受的安全高层 setpoint 适配器，再测试手工意图：短前进/小角度 yaw、接近不同地标、任务 STOP。飞行器的实际运动必须从 `/pose` 得到，不可把“已发送命令”当“已执行命令”。如果只给姿态角且没有速度控制，需单独记录实际运动，不可暗自假设 body_vx 与推力正比。

把控制时间分成：物理积分（候选 1ms）、飞控内部循环（由 SITL 实际周期决定）、相机（候选 20–30 Hz）、导航策略（候选 5–10 Hz）。全部使用仿真时间调度，状态对齐与动作完成 ACK 独立记录；RL 不能依赖 wall-clock sleep 代替 sim-clock。

## 6. GZ3：语言与 RGB 真正闭环

在 `Episode.reset(seed,task_id)` 时初始化无人机物理状态、相机、指令缓存、学生 GRU/ECPT 状态、历史动作缓存。每策略 step：取得满足最大 age 限制的真实 RGB→合法模型输入→输出意图→控制器限幅→执行短时窗口→读取真实位姿/碰撞→取得新的 RGB；被迫降级、超时、相机丢帧、碰撞等以明确终止原因完成当前 episode。

```
reset(scene,seed) -> verified_observation
repeat:
  rgb = capture_after(last_execution, max_age)
  command = policy(rgb,instruction,history)
  command = validate_and_bound(command)
  applied = controller.execute(command,ttl)
  transition = simulator.advance_until(applied_or_timeout)
  log(sim_time,frame_hash,command,applied,collision,goal_distance_private)
until success | collision | timeout | invalid_obs | safety_takeover
```

学生训练过程中只能利用 `rgb` 和合法过去状态；`goal_distance_private` 必须写到 evaluator/oracle 文件，不能回填 student memory。

## 7. 物理参数与 sim2real

X3/x500 与 CF 的 118mm 机体不能用同一物理参数描述。建立 `VehicleProfile` 包含质量、Ixx/Iyy/Izz、臂长、桨旋向、k_f/k_m、最大转速/加速度、deadzone、电机一阶延迟、相机安装与质心位置。X3 仅是 Gazebo 链路验证机体；迁移到 CF 需要单独质量/推力/供电/IMU/定高/速度控制能力测量。不能用未经验证的 `u_normalized -> omega` 公式直接接 Gazebo 电机插件。

真实姿态在飞行时可能存在漂移；不应在物理轨里永久把仿真真值姿态交给控制器并宣称真实 CF 一样稳定。先使用成熟 SITL 闭环作为基线，后续注入姿态噪声、速度延迟与机体参数偏差进行稳健性对照。

## 8. 必须有的错误码和阻断条件

- `GZ_SIM_NOT_INSTALLED`、`GZ_PLUGIN_ABI_MISMATCH`、`GZ_MOTOR_COUNT_MISMATCH`、`GZ_ACTUATOR_ID_DUPLICATE`；
- `CAMERA_TIMEOUT`、`CAMERA_FRAME_STALE`、`CAMERA_ENCODING_UNSUPPORTED`、`SIM_CLOCK_REGRESSION`；
- `ACTION_PROTOCOL_MISMATCH`、`CONTROL_ACK_TIMEOUT`、`CONTROLLER_NOT_STABILIZED`；
- `SCENE_UNSOLVABLE`、`EPISODE_RESET_UNVERIFIED`、`COLLISION_ORACLE_UNAVAILABLE`；
- `CF_SIL_ONLY`（CF 数学核测试不能升级为已验证固件/飞行器）。

## 9. 关键代码模块边界（后续实现，不在 V8 假装已存在）

`simulation/world_runner.cpp` 维护 Gazebo world/reset/clock；`rgb_bridge.cpp` 维护图像订阅/时间戳/颜色布局；`intent_adapter.cpp` 维护动作边界/坐标系；`control_bridge.cpp` 调 SITL setpoint；`episode_recorder.cpp` 持久化真实数据；`oracle_planner.cpp` 仅供训练生成和评测。每个模块只有一个数据责任，拒绝 3000 行大类和跨模块共享隐藏全局状态。

## 10. 最小可判 PASS 的证据

`GZ0`: raw motor/pose trace；`GZ1`: 真实 RGB 双帧与时间戳；`GZ2`: 固定策略完成“图像→意图→下一帧”；`GZ3`: 带语言和重置的真实完整 episode + 成功/失败原因。**Gazebo 安装成功、XML 语法正确、UI 显示 green 都不能替代这些证据。**
