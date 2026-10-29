# TinyAerialVLN V7.0：基于 CF-Drone 的 Gazebo RGB 物理闭环、自研训练与板端部署研究方案

> 版本：V7.0 / 2026-10-28  
> 文档类型：**待实施的研究与工程设计规范**，不是“已经跑通”的实验报告。  
> 本版核心升级：**真正阅读 `CF-Drone.zip` 源码，以源码存在的能力而非假想飞控能力定义闭环；补全 Gazebo 动力学、前视 RGB、控制桥、同步与训练数据闭环；扩展代码规范、一键入口、可视化与错误处理。**  
> 已检查材料：`CF-Drone.zip`（Arduino/ESP32 源码）、V6.1 研究 MD、V6.1 骨架中的 `run.sh` / `scene_tool.py` / 生成的 `forest_42.sdf`、用户之前提供的高飞团队项目参考与开源文档。  
> **真实性限制：本文编写环境没有安装 `gz`、没有接入真实 CF-Drone 板与 LicheeRV Nano，因此没有实际运行 Gazebo/ESP32/SG2002；本版提供严格可执行的集成顺序、接口与验收标准，不把设计示例写成已验证成品。**

---

## 0. Executive Decision：最重要的五个决定

1. **主研究方向不变：Aerial Vision-and-Language Navigation（空中视觉语言导航）。** 学生输入前视单目 RGB、完整自然语言指令、历史动作和在线压缩记忆；**不把 IMU、深度、真实位置、目标坐标作为学生网络输入**。模拟器的这些特权数据仅可用于教师、动力学、训练标签、安全评估。
2. **Gazebo 的第一优先级是“实际生成下一帧图像的物理闭环”，不是更漂亮的场景。** 先官方多旋翼物理例程，再新增 RGB，再导航/专家，再引入 CF-Drone 控制逻辑。
3. **CF-Drone 是姿态飞控底座，不是现成可执行导航的定高/定速飞控。** 实测前不得声称它具有可用的 ALTHOLD、位置或速度闭环；因此基准导航动作与真实控制命令必须经独立适配。
4. **沿用分层软件体系：少量 C++/CUDA 自研核心 + 官方 Gazebo 基础设施 + SG2002 厂商编译器；不以 PyTorch 作为学生主训练路径。** 不为追求“全自研”重写 Gazebo 碰撞、GPU 图像渲染或 NPU 编译器。
5. **保持“先验收、后增加复杂度”的开发纪律。** ROS2、PX4、3DGS、大型多模态教师、全面 HIL 都是可选分支，不能成为第一条闭环的前置条件。

### 0.1 研究问题与工程目标分别是什么

**研究问题 RQ**：在单目 RGB、长指令、极低边缘算力的约束下，怎样通过视觉证据与指令进度协同的紧凑状态，改善长时导航动作与终止（STOP）判断？可以进一步比较蒸馏带来的独立收益。

**工程目标 EG**：用一套可复现的连续动力学仿真生成图像—语言—动作数据，建立自研学生训练、Gazebo 在线评估、SG2002 模型编译、荔枝派 UART 发送、ESP32 硬件台架安全验证的完整链条。

二者的证据绝对不能混用：AerialVLN-S 离散 benchmark 指标不代表真实飞行；Gazebo 真值教师不是 RGB-only 学生；仿真定高不代表 CF-Drone 实机已经定高。

---

## 1. CF-Drone 源码审计：我们**实际拥有**哪些能力

### 1.1 本次读取到的文件与关键位置（来源：用户上传 `CF-Drone.zip`）

| 文件与范围 | 源码事实 | 工程结论 |
|---|---|---|
| `CF-Drone.ino:25–66` | `setup()` 初始化参数、电机、Wi-Fi/WebRC（按宏）、IMU、RC；`loop()` 执行 `readIMU→step→readRC→estimate→control→sendMotors→handleInput`，Wi-Fi 时额外执行 MAVLink。 | 主流程是 **Arduino 实机硬件循环**，不是 Gazebo 可直接加载的系统插件。 |
| `control.ino:13–53` | 注释称针对 **118 mm 轴距微型四轴**；使用滚转/俯仰角速度 PID、姿态角外环、偏航控制；`RAW/ACRO/STAB/ALTHOLD/AUTO` 枚举。 | 已有可借鉴的低层闭环控制数学形式，但需要单独验证定时、限幅和实际机型物理参数。 |
| `control.ino:43` | `ALTHOLD_HOVER_THRUST 0.5f` 注释明确为 **stub，待气压计实现**。 | “有 ALTHOLD 枚举”**不代表**“已有真实定高控制”。 |
| `control.ino:327–330` | WebRC 选择定高会回退 `STAB` 并显示“定高模式暂不支持”。 | 本版明确标注**定高尚未实现**。 |
| `control.ino:109–180` | 遥控输入经 `controlRoll/Pitch/Yaw/Throttle` 解释；STAB 把前两个输入映射为目标姿态，油门映射为归一化推力；ACRO 是目标角速度。 | `Forward 5 m` / `Turn 15°` 不存在现成物理命令映射；倾斜指令会引起加速度，不等于定速。 |
| `control.ino:183–235` | `attitudeTarget → ratesTarget → torqueTarget → motors[4]`，电机混控顺序在源码中定义。 | 可以抽离为可测试 **CF Control Core**，必须用数值黄金用例验证电机布局、偏航符号与混控方向。 |
| `estimate.ino:36–63` | 陀螺积分姿态，落地时有加速度修正；`applyAcc()` 在 `!landed` 时直接 `return`。 | 现有估计是**姿态**而非三维位置/速度；飞行中校正机制有实现局限，须做长时漂移和姿态稳定测试。 |
| `imu.ino` | IMU 初始化、读数、校准与安装方向变换；错误状态 `imuOK` 参与解锁约束。 | 导航学生可以不看 IMU，但 ESP32 保留 IMU 是维持姿态稳定的必要内部功能。 |
| `rc.ino:131–169` | SBUS/CRSF/ELRS 使用 `BOARD_RC_SERIAL` 与参数化引脚；默认 CRSF 420000 baud。 | 不能盲目把 LicheeRV 串口接到 RC 已占用的 UART。 |
| `board_config.h` | ESP32-C3、S3、普通 ESP32 的 RC 串口与引脚不同；C3 默认 `WIFI_ENABLED=0`。 | **必须先确认实际 ESP32 具体型号、PCB、UART 空闲引脚、电压域**。 |
| `CF-Drone.ino:26`、`cli.ino:239–...` | `Serial.begin(115200)`，主串口承担启动日志与文本控制台命令。 | **不能将 `Serial` 调试文本接口直接当可靠二进制飞行指令协议**。 |
| `mavlink.ino:70–105` | MAVLink 接收通过 `receiveWiFi()`、输出经 `sendWiFi()`，支持 `MANUAL_CONTROL`。 | 这里的 MAVLink **不是已完成的 UART MAVLink 接口**。 |
| `mavlink.ino:173–207` | AUTO 下有 `SET_ATTITUDE_TARGET`、`SET_ACTUATOR_CONTROL_TARGET`；姿态目标路径会根据非零推力改变 `armed`，后者可直接写入电机数组。 | **高风险接口**：不得将模型输出直接接入这两个未经飞控安全审计的入口。应新建受限导航入口。 |
| `safety.ino:38–116` | 有 RC 失联、WebRC 失联和 `descend()`；RC 失联阈值默认约 1s，WebRC 阈值 8s。 | 有基础失效保护；但并不等于“独立的导航命令 TTL/序列号/ACK/人工接管已经具备”。 |
| `safety.ino:60–85` | `descend()` 逐渐降低推力，并在低于 0 时上锁。 | 属于**推力渐降**，不是由高度估计闭环控制的安全降落；在不同载荷和高度下有危险。 |
| `motors.ino:9–83` | 4 个归一化电机命令，支持 PWM 等输出。 | Gazebo 电机模型要求**角速度 rad/s**或插件约定命令，必须经过电机标定和单位映射。 |
| `control.h` | `ControlState` 和 `MODE_AUTO` 等接口定义。 | 一份头文件存在**不等于**跨 MCU/模拟器的通用接口已实现。 |
| `LICENSE` 前约 100 行 | 标注 GPLv3 主许可和自定义附加条款，包括商业限制、品牌署名。 | 集成/修改/对外分发前须核对许可兼容性和作者授权；存在法律解释风险，不能默认为无限制可商用或自由改许可。 |

### 1.2 关键结论：CF-Drone 当前并不拥有这些能力

- 未发现可投入使用的连续速度闭环控制器（`vx, vy, vz, yaw_rate`）。
- 未发现运行时可用的飞行中高度反馈闭环与稳定悬停证明。
- 未发现导航侧“接收动作—确认接收—执行完毕”的独立 UART 协议。
- 未发现 Gazebo、SDF、SITL 接口、仿真 IMU 依赖注入或模拟电机输出适配。
- 没有真实机体质量、惯量、推力曲线、偏航力矩系数、桨叶旋向、动力响应时间的测量记录。

**结论不是“飞控不好”，而是：它当前处在自稳飞控与调试遥控功能阶段；导航所需的运动控制和仿真接口，需要通过可验证的增量方式补齐。**

### 1.3 许可证与来源边界（重要）

`CF-Drone/LICENSE` 包含商业限制和强制品牌保留的附加条款，并自行称与 GPLv3 同时适用。额外商业限制与 GPLv3 的兼容性可能需要专业法律核对。本项目建议：

1. 暂将源码只作为用户已提供的**研究参考与独立飞控集成基线**；保留原始文件与版权说明，不删除品牌；不将源码无差别复制进自研 TinyVLN 仓库。
2. 优先**独立实现通用的姿态 PID 数学接口与 Gazebo 适配**，用测试对拍；涉及原项目的实质移植或分发时，再完成许可审查。
3. 对外论文如使用修改版本，准确注明 CF-Drone 来源、许可状态、版本和具体改动；不能将原项目工作表述为我们从零实现的飞控。
4. 这属于工程风险提示，不是法律意见。

---

## 2. V6.1 “Gazebo 尚未闭环”的根因复核

本次实际查看 `TinyAerialVLN_V6_1_工程骨架/artifacts/scenes/forest_42.sdf`：

- 已有：`world`、重力、物理步长、太阳光、地面、圆柱障碍、目标可视标记。
- **缺失**：`<model>` 动态四旋翼、质量/惯量/转子连接、相机 `<sensor type="camera">`、`gz-sim-sensors-system`、电机推力插件、动作输入桥、观测消费方、接触/碰撞事件、episode reset 和数据录制。
- 因而目前 `./run.sh scene` 的真实成果仅是**可复现静态几何 SDF**，不是完整导航训练环境。

**V7 对这个缺口采用最短可验证路径，而不是重写整个 Gazebo。**

### 2.1 三条可选路径比较

| 路线 | 动力学与控制 | 优点 | 不足 | 本版地位 |
|---|---|---|---|---|
| A. Gazebo 官方 X3 电机模型 + C++ 简易控制器 | Gazebo 的 `MulticopterMotorModel` 负责推力与力矩，薄控制桥发送 4 转子速度 | 最容易证明图像、动力学、动作闭环 | 不是 CF-Drone 原固件 | **必须先完成** |
| B. X3/自定义 118mm 机体 + CF-ControlCore SIL | Gazebo 电机物理 + 从 CF-Drone 抽出的姿态/角速率数学核心 | 接近实际代码的控制行为，可以对拍与故障注入 | 需要传感器模拟、参数标定、电机映射 | **V7 主线** |
| C. ESP32 真板 HIL + Gazebo | Gazebo 合成 IMU→真实 ESP32→采集电机输出→Gazebo | 覆盖 Arduino 固件时序及真实外围通信 | IMU 驱动注入、硬件引脚和双向时间同步复杂 | **安全台架后选做** |

另一个参考是 PX4 官方 `gz_x500`、`gz_x500_depth`，它能验证 Gazebo 飞行动力学和摄像头基础设施，但使用的是 PX4 而非 CF-Drone；**不能把 PX4 闭环成绩写成已验证 CF-Drone**。参见 https://docs.px4.io/v1.18/en/sim_gazebo_gz/ 。

### 2.2 为什么不先移植 `.ino` 整个程序

因为 `.ino` 同时包含 Arduino `Serial`、SPI/IMU 硬件驱动、PWM/LEDC、电池 ADC、Wi-Fi、WebRC、Flash 参数等硬件依赖。未经抽离直接编进 Gazebo，容易出现：同名符号、循环调用顺序失真、模拟时间和 `millis()` 不一致、串口回调阻塞、重复电机模型、状态估计时钟错误。

正确方式是通过窄接口提取**物理无关控制逻辑**，再提供 `HardwareIO`、`SimIO` 两种适配；基础阶段可只做控制器 SIL，不必从第一天起重构完整固件。

---

## 3. 最终应运行的系统架构（数据和权限边界）

```text
                  ┌────────────── 训练/评测主机：Ubuntu 24.04 ──────────────┐
自然语言指令 ────> │ Text Encoder/Cache                                 │
Gazebo Camera ──> │ RGB Image → 学生模型 → VLN 动作/STOP/风险             │
                 │                      │                                │
                 │            ActionPolicyAdapter（动作转换层）           │
                 │                      │                                │
                 │              Navigation Safety Gate                  │
                 │                      │                                │
                 │                Flight Setpoint                       │
                 │                      v                                │
                 │     CF-ControlCore-SIL / 基础仿真控制器（2选1）        │
                 │                      │                                │
                 │             4×归一化电机输出                          │
                 │                      │ 校准变换                       │
                 │           gz.msgs.Actuators: ω[4] rad/s               │
                 │                      │                                │
                 │      Gazebo MulticopterMotorModel + Physics          │
                 │                      │                                │
                 │            机体运动 + 碰撞 + 新 RGB                    │
                 │                      └────── loop ──────────────────┘│
                 │                                                      │
                 │ 真值位姿/深度/地图 ──> Expert/Labels/Evaluator ONLY     │
                 └──────────────────────────────────────────────────────┘

真实硬件阶段：
前视相机 → SG2002 TinyVLN → C++安全状态机 → UART协议 → ESP32 自稳飞控
                             （不复用 Gazebo 真值、教师或后门接口）
```

### 3.1 模块职责（严禁越界）

| 模块 | 唯一职责 | 可见数据 | 禁止事项 |
|---|---|---|---|
| `SceneBuilder` | 场景模型、碰撞体、语义对象、任务生成 | 随机种子和场景参数 | 在训练时偷偷提供学生障碍物答案 |
| `WorldRunner` | 控制 Gazebo 世界状态、reset、step、超时 | 全部仿真状态 | 修改模型内部权重或选训练动作 |
| `RgbBridge` | 相机帧、时间戳、格式转换 | RGB + CameraInfo | 读取 GT 深度作为网络特征 |
| `ControlCore` | 目标姿态/角速率/推力到电机 | 允许的飞控内部 IMU/估计状态 | 访问语言或未来参考路线 |
| `MotorBridge` | 4 电机归一化量→4 转子速度 | motor command + motor calibration | 自己暗中修改目标轨迹 |
| `Expert` | 根据完整几何与动力学给出教学信号 | 特权 GT | 在线学生模式直接注入动作 |
| `StudentPolicy` | RGB+语言+记忆→导航意图 | 仅合法 observation | 导航时读 pose、depth、collision GT |
| `SafetyGate` | 输入过期、TTL、人工优先、许可状态 | 观测健康与执行状态 | 声称 neutral=悬停、STOP=断电 |
| `Evaluator` | 计算 NAV/SR/碰撞/轨迹与时延指标 | 所有评测真值 | 将 GT 反馈给学生 |

---

## 4. Gazebo Harmonic：从静态障碍到物理飞行的逐步解决方案

### 4.1 环境固定和官方示例优先

目标环境是 Ubuntu 24.04 + Gazebo Harmonic（gz-sim 8 系列），首先只要求 `gz sim` 和 Gazebo Transport C++，**不安装 ROS 1，不默认要求 ROS 2**。Gazebo Harmonic 官方支持 Ubuntu 24.04，安装文档：https://gazebosim.org/docs/harmonic/install_ubuntu/ 。

```bash
# Ubuntu 24.04，遵循官方仓库安装办法；按系统情况检查已有 NVIDIA 驱动
sudo apt-get update
sudo apt-get install -y curl lsb-release gnupg git cmake ninja-build g++ pkg-config
sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
  -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
sudo apt-get update
sudo apt-get install -y gz-harmonic

gz sim --version
gz sim -v 4 shapes.sdf
```

说明：实际目标主机此前为“空白 Ubuntu”；这些命令**尚未在用户机器执行**。官方包可能更新，应锁定 `apt-cache policy` 得到的具体版本和 `dpkg -l` 报表。CUDA、SG2002 TPU-MLIR 和旧 AirSim 不能混进这个最小仿真栈。

#### 若 RTX 5090 上 Ogre2 渲染失败

先核对 `nvidia-smi`、Vulkan 图形驱动、`gz sim -v 4` 输出及 GUI/Headless 情况；分别测试 GUI、`gz sim -s -r` server-only，必要时诊断软件渲染（仅功能测试，不用来冒充真实性能）；不要贸然降级 5090 主机驱动。`headless`、物理 server 和 RGB 渲染并不是同一回事：server 可以运行但相机渲染链仍可能不工作。

### 4.2 用官方 X3 电机案例先证明“物理动力学确实在跑”

Gazebo 官方示例 `gz-sim/examples/worlds/quadcopter.sdf` 使用 `MulticopterMotorModel`，实际命令主题为 `/X3/gazebo/command/motor_speed`，消息是 `gz.msgs.Actuators`，字段 `velocity` 是四个转子的**角速度**，示例参考 `700 rad/s`；这是**示例机体的测试值，不能用于真实 CF-Drone 电机**。

官方源码：https://github.com/gazebosim/gz-sim/blob/gz-sim8/examples/worlds/quadcopter.sdf （以 Gazebo 8 安装包中的例程为准，若某版本文件路径不同则从安装包或对应 tag 获取）。

建议的官方“无视觉、先物理”命令：

```bash
# 实际路径由安装版本决定，示意：
gz sim -v 4 -r quadcopter.sdf

# 在另一终端（仅用于官方 X3 仿真例子，不连实体电机）：
gz topic -t /X3/gazebo/command/motor_speed \
  -m gz.msgs.Actuators -p 'velocity: [700, 700, 700, 700]'

# 由停转测试检验模型受到电机命令影响：
gz topic -t /X3/gazebo/command/motor_speed \
  -m gz.msgs.Actuators -p 'velocity: [0, 0, 0, 0]'
```

**Gate GZ0 的真实验收：** 物理引擎运行、机体 pose 随电机指令改变、停转导致动力消失、世界可 reset、没有重复转子插件。这里暂时只检查官方示例本身，不导入我们生成的森林世界。

### 4.3 在无人机机体上加前视 RGB 相机（不是悬浮在世界里的独立相机）

**推荐第一版相机定义**：

- `sensor type="camera"`，固定安装在四旋翼 base link 前部；记录 `base_link → camera_optical` 固定变换。
- RGB `640×360` 或 `640×480`，`30 Hz`，水平 FOV 先按实际镜头未知情况下的标定候选值配置；学生输入可以是 `320×192 / 384×224 / 224×224`，但必须记录 letterbox 方法。
- 使用 Gazebo `gz::sim::systems::Sensors` / `<render_engine>ogre2</render_engine>`；相机接口采用 `gz.msgs.Image`。
- `RGB8 / BGR8`、stride、镜像、时间戳、首帧曝光/自动曝光和像素是否有有效纹理要在数据合同中实测。
- 同一场景目标地标必须具有不同的材质/颜色/空间拓扑，不能只靠可见 goal_marker 获得答案。**训练时可视化目标标记必须与学生所见相机分离**。

示意传感器 SDF 片段（**待插入实际四旋翼 link 的设计片段，不是独立可运行 SDF**）：

```xml
<sensor name="front_rgb" type="camera">
  <always_on>true</always_on>
  <update_rate>30</update_rate>
  <topic>/tinyvln/front_rgb/image</topic>
  <camera>
    <horizontal_fov>1.5708</horizontal_fov>
    <image>
      <width>640</width>
      <height>360</height>
      <format>R8G8B8</format>
    </image>
    <clip><near>0.05</near><far>100.0</far></clip>
  </camera>
</sensor>
```

**安装方向必须做真实像素测试**：假设机头正对 X 正方向，在正前方放一个红色立柱，左右两侧放颜色不同的地标；观测验证中间红色、左侧/右侧没有镜像与旋转错误；不能仅凭欧拉角猜相机光轴。

### 4.3.1 将官方 X3 与相机绑定到同一机体的具体操作步骤

这是 V7 的关键实施细节：**不能简单在 world 中额外创建一个静态摄像头，声称它是机载相机**。推荐开发者按如下顺序修改模型：

1. 先运行官方 `quadcopter.sdf`，利用 `gz topic -l` 与模型 pose 确认 X3 实际加载。官方示例通过 Gazebo Fuel 引用 `X3 UAV/4`，需记录首次下载缓存、资源许可证及资源哈希；必要时将合法缓存固定在 `third_party/gazebo_models` 的受控版本，不能依赖未锁版本的在线资源作为论文复现条件。
2. 获取 X3 模型 SDF，查出其真实机身 link 名称、碰撞体中心、四转子关节和四转子旋向。把原始 `include` 展开为自己维护的**小范围派生模型**或经验证的 SDF 合并形式；具体语法以 Gazebo 8 + 当前模型 SDF 为准，不能假定外部 `<include>` 后能直接修改内部 `link`。
3. 将 `<sensor type="camera">` 放在机身刚性 link 上，并添加相机固定外参（机头偏移 + 光学轴旋转）。如果需要独立 camera link，应新增固定关节，而非创建世界固定相机。
4. 使用 `<visual>` 和 `<collision>` 验证相机安装没有改变机体惯量/碰撞体，除非明确模拟相机质量。把机体绕 yaw 旋转 90° 后，RGB 画面中的目标方位应随之旋转。
5. 加载传感系统插件，执行 `gz topic -l` 找到图像主题，再订阅它而不是猜测默认主题字符串。设备采集要保存 `encoding/step/width/height/header-stamp`。
6. 将生成的 `forest_42.sdf` 中静态障碍嵌入世界，保留新的动态四旋翼 model。起飞后记录 `pose(t)` 和 `frame(t)`，确认位姿变化和图像变化由同一物理过程驱动。

用于定位主题和检查帧的调试命令（实际主题以 `gz topic -l` 为准）：

```bash
gz topic -l
gz topic -i -t /tinyvln/front_rgb/image
# 二进制图像不要无节制打印到终端；建议由 C++ subscriber 计数并抽帧保存 PNG。
```

**不要在同一机体同时启用互相独立、争抢同一四转子的 `MulticopterVelocityControl` 和自研 `CFControlCore→MulticopterMotorModel` 命令链**。开发初期的基准控制器与 CF 控制器通过配置二选一，记录 `controller_profile_id`。

### 4.4 World SDF 的系统插件应如何布置

在 V6.1 静态 SDF 基础上，新增/核对四类插件：

- `gz-sim-physics-system`：基础重力、惯量、接触求解。
- `gz-sim-user-commands-system`：创建/复位等用户服务（按实际功能需求启用）。
- `gz-sim-scene-broadcaster-system`：GUI 场景广播。
- `gz-sim-sensors-system`：RGB 相机渲染。
- 可选 `gz-sim-contact-system`：接触检测与评测，不进入学生观察。

官方教程示例有完整配置：https://gazebosim.org/docs/harmonic/sensors/ 。

注意：**SDF 文件只写 `<sensor>`、不加载传感器系统插件，不保证相机真的更新。** 插入自定义世界 `<plugin>` 后也应显式检查所需默认系统是否仍被加载。

### 4.5 四个电机：不要误把 PWM/归一化推力当 rad/s

当前 CF `motors[4]` 是归一化命令（0–1），而 Gazebo `MulticopterMotorModel` 接收转子角速度 `ω_i`，随后根据 `motorConstant`、`momentConstant` 等产生推力/力矩。因此需要独立的 `MotorBridge`：

```text
CF MotorMixer: u0,u1,u2,u3 (normalized)
     ↓ 有限值检查 + 0..1 限幅 + 电机索引/旋向检查
MotorCalibration: u → ω_cmd (rad/s)
     ↓ 转子一阶动力学/插件处理，避免重复建模
Gazebo Transport: gz.msgs.Actuators{velocity:[ω0,ω1,ω2,ω3]}
     ↓ MulticopterMotorModel + Physics
Base link pose, velocity, contacts → Next RGB / CF SIM-IMU
```

**对理想化推力模型**，若 `F_i ≈ k_f·ω_i²`，可以写 `ω_i=sqrt(F_i/k_f)`；**但这不能证明实际 PWM 的 u 与推力线性相关**。对真实 CF-Drone 微型有刷电机/电调，需要测量/拟合 `u→ω` 或 `u→F`，并记录电池电压影响。未标定阶段只能用自洽的模拟曲线做 SIL，不能称为实机动力学复现。

明确标定参数：质量 `m`、三轴惯量 `Ixx/Iyy/Izz`、机臂长度、机体重心、重力方向、每个转子旋向、`k_f`、`k_m`、推力死区与最大转速、上升下降响应时间常数、电机输出顺序。任何数值都是“待测量/来自官方 X3”的二选一，不得把 X3 的 `m/kf` 写进真实 118mm CF 配置。

### 4.6 CF-Drone 电机顺序的强制对拍

`motors.ino` 注释声明引脚顺序为 `RL, RR, FR, FL`，而 `control.ino:224–227` 使用 `MOTOR_FRONT_LEFT/FRONT_RIGHT/REAR_LEFT/REAR_RIGHT` 进行混控。Gazebo 示例 X3 的四电机则分别绑定 `actuator_number=0..3`。

**这不是两个顺序天然相等**。必做测试：

| 输入扰动 | 物理期望 | 不通过时可能原因 |
|---|---|---|
| 4 电机等推力 | 机身近似垂直上升 | 重力、质量、推力模型错误 |
| 单独增加 roll 相关电机组合 | 机身按预期方向滚转 | motor index、旋向、坐标约定颠倒 |
| 单独增加 pitch 组合 | 按预期前后俯仰 | 前后电机索引或 FLU/FRD 颠倒 |
| 单独增加 yaw 组合 | 按预期偏航方向旋转 | CW/CCW、k_m、偏航符号错误 |
| 零推力 | 自然受重力下降 | 仿真偷偷把机体设为 static/kinematic |
| 长时间固定推力 | 位置/速度随动力学积分变化 | `SetPose` 取代物理求解 |

若“仅按 CF 混控算出的四电机值注入 Gazebo”无法稳定，**不能直接修改视觉导航网络去掩盖飞控错误**。

### 4.7 传感器模拟：内部允许 IMU，学生只能 RGB

SIL 的 `SimIMU` 必须来自物理真值的**含重力/不含重力约定正确的 specific force** 与体轴角速度，再加标定噪声/偏置/时间戳。特别注意：不能把世界系加速度直接塞给陀螺/加速度计驱动；`gravity`、ENU/FLU 与 CF 内部坐标变换至少做静止、俯仰 90°、自由落体三个用例。

`SimIMU` 只提供给 `ControlCore` 的姿态估计；学生 batch 中不得出现 IMU。仿真时可将真实姿态直接给一条**“飞控估计器上界诊断轨”**，但不能把它当成 CF-ControlCore 的真实表现。

### 4.8 物理、传感和策略多速率同步

建议初始时钟（仅工程配置候选）：

| 组件 | 建议频率 | 时钟来源 | 记录 |
|---|---:|---|---|
| Gazebo physics | `dt=0.002 s` / 500 Hz | `sim_time` | 全局时间基准 |
| SimIMU + CF 内环 | 200–500 Hz（按稳定性与算力实测） | 同一 sim clock | 真正执行 tick 次数 |
| RGB camera | 30 Hz | 渲染时间戳 | 图像 capture time |
| TinyVLN policy | 2–10 Hz（离散高层意图） | 最新可用图像时间 | perception start/end |
| UART/HIL 指令 | 由实际 ESP32 控制协议决定 | 独立单调时钟+映射 | send/ACK/EXEC_DONE |
| Dashboard | 1–2 Hz 刷新 | 指标日志时钟 | 不得阻塞实时控制 |

必须把 `capture_ts`、`policy_start_ts`、`policy_end_ts`、`command_send_ts`、`control_apply_ts`、`physics_ts`、`collision_ts` 写入 episode，监测 **age-of-information**。`world.reset` 后清空相机缓存、模型 GRU 状态、STOP 状态、上一动作与所有待确认命令。不同 episode 的前一帧不得作为后一 episode 的输入。

**锁步优先**：数据生成/评估优先以 `sim_time` 固定步长推进，等待本步策略或控制结果后再推进，避免不同机器帧率导致的隐式动作时间差；实现上可采用 `gz::sim::Server` 步进控制或 Gazebo world control 服务，但应锁定 simulator ABI 并查验接口实际存在。实时 HIL 则必须采用 wall-clock 并额外测定延迟。

### 4.9 物理碰撞不能靠图片颜色判断

- 机体应有包络碰撞体，最少包括中心机身与旋翼保护包络；记录其几何半径/尺寸。
- 所有生成障碍必须同时有 `<collision>` 与 `<visual>`；二者坐标与尺寸一致。
- 使用接触传感器/仿真碰撞事件判断真实碰撞，同时记录最小间隙、是否进入目标区域。
- **相机看不到碰撞仍可能碰撞**（如旋翼擦到侧面障碍），所以学生可以 RGB-only，但评测必须以完整机体碰撞几何为准。
- reset 必须重置电机转速、速度、姿态、接触状态和碰撞计数，而不只是 teleport 一个位置。

---

## 5. CF-Drone 控制器的仿真抽离方案：从“有姿态 PID”走向“可验证 SIL”

### 5.1 不动/少动原固件的分层策略

**层 A：`CFMath`**——姿态四元数、PID、混控、饱和限制与角速率目标计算。这些是基本数学模块，可独立实现并与原固件用固定输入对拍，优先测试而非直接复制整份 `.ino`。

**层 B：`CFControlCore`**——只持有控制状态、运行 `step(dt, measured_imu, setpoint, health)`，输出归一化 4 电机、故障状态；没有 Arduino `Serial/WiFi/PWM/SPI` 头文件。

**层 C：`SimIO`**——从 Gazebo 接收 IMU/状态，调用控制核心，发 Gazebo 电机消息；绝不包含语言模型。

**层 D：`Esp32IO`**——实机仍运行原 CF-Drone 或经合法授权的派生固件；从新增的安全导航 UART 通道接收高层指令，并经过安全仲裁再使用已有姿态控制器。

### 5.2 候选跨平台接口（精确类型属于 V7 设计，尚未实现）

```cpp
struct ImuSample {
  uint64_t sim_time_ns;             // 单调仿真时钟
  float gyro_radps[3];             // 体轴角速度
  float specific_force_mps2[3];    // 体轴比力，约定与测试固定
  bool valid;
};
struct NavSetpoint {
  uint32_t seq;
  uint64_t issued_ns;
  uint32_t ttl_ms;
  enum Mode { DISABLED, NEUTRAL, YAW_INTENT, FORWARD_INTENT, RESTRICTED_ATTITUDE } mode;
  float value_1;                   // 每种 mode 的单位在协议表定义
  float value_2;
};
struct MotorOutput {
  float motor_norm[4];             // 从 CFD 控制核心输出 0..1
  bool motors_permitted;           // 仅仿真或硬件安全仲裁确认
};
class ControlCore {
 public:
  virtual void reset() = 0;
  virtual MotorOutput step(float dt_s, const ImuSample&, const NavSetpoint&) = 0;
};
```

这里没有引入学生的深度/GT 状态输入，也没有让网络直接输出电机 PWM。

### 5.3 最大工程难点：缺少定高/速度闭环怎么办？

**必须把“能闭环飞起来”和“真正仿真 CF-Drone”分成两个不同 gate。**

- **GZ-P0：独立仿真基准控制器**。先对 X3 使用一个已知可工作的姿态/速度控制器，完成起飞、稳定高度、前视 RGB 和障碍回避的物理闭环。此结果只证明 **Gazebo/Navigation pipeline** 正确。
- **GZ-P1：CF姿态控制核心 SIL**。相同动力学模型下，验证 roll/pitch/yaw 响应、电机映射、安全限幅；由于缺少定高，先在**无自由飞行要求**的姿态扰动/支架模型中测试。
- **GZ-P2：独立实现或补充可验证的高度闭环**。应先明确真实传感器计划（如仅使用飞控内部可用传感器实现定高，不向视觉学生供给），并在模拟器中验证 `height setpoint → throttle/vertical acceleration`。如果真实硬件没有高度反馈，纯模拟真值定高控制只能作为**研究上界**，不可声称可迁移。
- **GZ-P3：添加速度/机体响应界限**。导航命令只能在已验证的控制域内执行，加入推力/倾角/转向速率/命令持续时间限制，考虑刹停距离与延迟。

**未完成 GZ-P2 前，不允许把视觉学生的 FORWARD/STOP 直接连接真实飞行。** 若最终坚持不增加任何用于姿态之外的高度测量，应将实体演示限定为手动保持高度下的导航建议或受约束平台，不把它包装成完全自主飞行。

### 5.4 仿真和实机的目标空间有两套，不要偷换

| 任务层次 | 输入 | 输出 | 评测解释 |
|---|---|---|---|
| AerialVLN 离散评测 | RGB+language+memory | 官方 8 动作（含 5m 位移和 15° 转向） | 纯 benchmark，不含连续动力学 |
| TinyVLN 物理导航 | RGB+language+memory | 有界运动意图（方向、持续时间、是否请求终止） | 必须经过控制适配才能执行 |
| ESP32 控制器 | 自己的飞控状态/IMU + 合法指令 | 姿态/电机闭环 | 无语言处理职责 |

**推荐保留独立双 head 或 adapter**：网络核心共享语言—视觉—记忆，AerialVLN 评测输出官方动作 head，Gazebo 连续任务输出限定的 `NavIntent`，两个任务不共用同一“动作 ID → 物理位移”硬编码。

---

## 6. 数据与任务：小平台启动，不伪称无人机能力

### 6.1 三层数据，不再混成一个 benchmark

| 层级 | 数据来源 | 是否有逐帧 RGB | 是否连续物理 | 使用目的 |
|---|---|---|---|---|
| D0 | BabyAI/MiniGrid | 可渲染简化观测（非真实无人机视觉） | 否 | 测 C++ tokenizer、指令绑定、记忆/STOP、序列重置 |
| D1 | MiniWorld / 简化三维 | 在线第一视角 RGB | 导航动作抽象，非可信四旋翼 | 快速形成 RGB+language 小基准 |
| D2 | **V7 Gazebo-PhysVLN** | 是，前视 RGB | 是，四旋翼动力学 | 主工程闭环与连续导航数据 |
| D3 | AerialVLN-S | 官方仿真器在线 RGB，下载标注不等于下载帧 | 官方主要为离散位姿步进 | 对照已有空中 VLN 论文 |

不要用 MiniGrid 的导航成功率证明无人机避障，不要用 AerialVLN 的 5m 位置跳转证明真实安全。

### 6.2 V7-Gazebo 任务定义（自建训练集）

一条任务记录包含：

```yaml
scene_id: forest_train_0042
world_seed: 42
render_seed: 4201
camera_calib_id: front_cam_v1
vehicle_profile_id: x3_sil_v1  # 或 cf_118mm_identified_v1，不能混用
instruction: "Pass the blue pillar, turn right at the yellow gate, stop near the red marker."
start_pose_gt: [x, y, z, yaw]  # 教师和评测使用；学生不可见
subgoals_gt:
  - {object_id: blue_pillar_1, event: pass}
  - {object_id: yellow_gate_2, event: turn_right}
  - {object_id: red_marker_3, event: stop}
student_observation: [rgb, instruction_tokens, action_history, recurrent_state]
teacher_privileged: [map, state, collision, planned_trajectory]
```

完整 episode 还要记录每次图像、真实控制 apply time、motor velocity、物理 pose/velocity、专家建议、碰撞、最小间隙、终止原因和模型版本。

**语言要由场景对象和可执行参考路径双向约束生成**。不能简单随机给颜色、地点和方向词，否则会出现“说要经过蓝柱，地图里没有蓝柱”或“蓝柱在右边却标注从左绕行”的错误标签。

### 6.3 从高飞团队公开代码借鉴什么

- `uav_simulator/mockamap`、`random_forest_sensing`：随机几何与森林障碍分布；但这些是几何/点云层，**不是现成的 RGB 材质场景**。
- `so3_quadrotor_simulator`：四旋翼 6-DoF、四电机、惯量与积分器的数学参考；不是把其 ROS1 依赖拷进 Gazebo 的理由。
- GCOPTER/MINCO：生成平滑动力学可行的专家参考轨迹。若仅需小任务，可先使用 A* 网格/可行通道 + 限速轨迹跟踪教师；不以 GCOPTER 成功编译为 GZ0 前提。
- 真正要自己写的是**场景语义对象与相机渲染对应、任务合成、控制桥、同步和数据质量控制**。

参考：
- https://github.com/ZJU-FAST-Lab/Fast-Drone-250
- https://github.com/ZJU-FAST-Lab/GCOPTER
- https://github.com/gazebosim/gz-sim

### 6.4 场景四档难度

| 难度 | 组成 | 专家动作与评估重点 | 失败分型 |
|---|---|---|---|
| S0 | 空地、单个颜色标记 | 前进+目标终止、正确 camera/command mapping | 模型不会停、镜像、GT 泄漏 |
| S1 | 两个地标、单个柱体 | 区分 landmark + 指令顺序 | 语言忽略、错误转向 |
| S2 | 森林稀疏柱阵、拐弯、多个相似目标 | 轨迹可通行、反应式避障 | 错误地标、视觉遮挡 |
| S3 | 密集几何、光照/纹理扰动、未见目标组合 | 泛化与控制安全 | 控制饱和、低纹理、误停止 |

**必须自动检查可解性**：场景目标存在、参考路径能穿过飞行包络、障碍 `collision` 与 `visual` 一致、速度与曲率在专家控制器范围内、起始点没有碰撞。生成失败应最多有限次重试，记录 `SCENE_UNSOLVABLE`，不得静默删除困难样本而污染统计。

### 6.5 不同场景 split 和小数据规模

建议首先用 `16/64` 条完成程序过拟合与复现，再采集 `100–300` 条 smoke episode，之后按独立几何实例产生 `500 / 2,000` 条训练 episode。正式 train/val/test 不能随机切“帧”，要按**场景实例、地标组合、参考路径组**隔离；报告每种难度实际覆盖数，场景数量和评测数量是否足够由功效分析决定。

每个真实画面要标记 `scene_hash`、`episode_id`、`step_id`、`camera_intrinsics`、`image_encoding`、`policy_version`、`expert_version`、`randomization_seed`。数据集大小只是初始探索量，不是保证论文成绩的充分条件。

---

## 7. 教师、学生、蒸馏：先保证教师物理可执行

### 7.1 三类教师及适用证据

1. `T_geo`：拥有物理位置与完整障碍的专家，生成**可执行有界运动意图**、可通行性/风险、阶段终止；用于 Gazebo 内连续动力学监督。
2. `T_vlm`：大型视觉语言教师，读合法或声明的特权模态并给分布监督；必须验证它在**在线闭环**上优于学生，不能仅靠动作训练准确率声称更好。
3. `T_aerial`：AerialVLN 参考轨迹与 LAG 式纠偏指导，仅用于其官方离散评测合同，不直接推到 CF 的电机上。

**先执行 `T_geo → BC student`，再比较 `T_geo + T_vlm` 是否带来额外收益。** 如果大模型教师没有优势，删除昂贵教师，不影响主方法。

### 7.2 教师和学生的分布失配

同一个 GT 状态可能有多个可行动作，专家轨迹只是一个偏好，不是唯一路径。对风险头可短程 rollout 候选动作，记录碰撞/接近障碍的频率或风险分位数，优于简单将“专家没选择”全部标成危险。对于 RGB 中不可观察的隐蔽障碍，不能强迫学生回归几何教师的确定标签；可对不可观测样本记录不确定性、降权或进行多模态风险校准。

### 7.3 最小学生网络与损失

- RGB：单张或双帧、固定形状的低算力 CNN；第一版先取一帧，不引入无法编译的 GridSample/动态 shape。
- 语言：完整指令编码，避免 `L=48` 造成截断；词表、P95/P99 token 分布从 train 统计。
- 跨模态：保留少量空间 token，用当前指令相关特征对照全局池化基线。
- 记忆：小 GRU 或可手写的显式 gated state；每 episode reset。
- 头：动作、独立 STOP、可选风险/进度；训练期可有辅助头，板端是否保留由 ABI 和收益决定。

```text
L = λ_bc * CE(action_logits, expert_action)
  + λ_stop * WeightedBCE(stop_head, stop_label)
  + λ_prog * ProgressAuxLoss(estimated_progress, weak_event_label)
  + λ_kd * TemperatureKLD(student_action_distribution, teacher_distribution)
  + λ_risk * RiskCalibrationLoss(student_action_risk, teacher_rollout_risk)
```

先令 `λ_prog=λ_kd=λ_risk=0`，只跑通 BC+STOP；后续每次仅增加一个变量，所有复杂损失都要有同容量、同训练数据和闭环配对对照。进度标签缺失时不允许偷用 GT 子路径索引作为学生推理输入。

### 7.4 两种 STOP 绝对不同

`AerialVLN STOP`：benchmark 终止当前导航 episode，计算与 GT 终点的误差。

`Real STOP_REQUEST`：模型认为目标完成，**请求控制系统终止自主任务**；无人机如何保持高度、减速、降落或切人工由已验证飞控能力决定。不能把 STOP 直接映射成 `motors=0`、`armed=false` 或“保证悬停”。

---

## 8. 自研 C++/CUDA 的实现方案：尽量不依赖 PyTorch，但不重复造轮子

### 8.1 实现原则

学生训练器采用**固定计算图 + 显式反向函数**，而不是先开发任意图 Autograd 框架。核心 TensorView/Allocator、Embedding、Conv/DWConv/Linear、GRU、Attention Lite、Softmax-CE、AdamW 和优化步由 C++/CUDA 开发；cuBLAS/cuDNN 可按需作为 kernel backend。网络结构固定后导出 ONNX，再交 SG2002 兼容工具链。

教师若来自公开预训练权重，可以在隔离环境离线生成 logits/特征，但生产学生的 `tinyvln_train` 不导入 PyTorch。数字正确性测试允许 NumPy 或隔离 PyTorch golden vector，且**必须测试语义、shape、mask、坐标和动作协议**，不能只检查两种实现彼此一致。

### 8.2 开发先后顺序与检查点

| 内核 | 前向 | 反向/训练 | 验证 |
|---|---|---|---|
| TensorView/stride | 内存、shape、dtype | 梯度缓冲 | 越界、随机张量 checksum |
| Linear / Embedding | GEMM、padding | 梯度和 sparse embedding | 与独立 CPU/NumPy reference 对拍 |
| Conv/DWConv | NCHW/stride/pad | 反向卷积 | 逐层 forward、FP64 gradcheck |
| Softmax / loss | mask、极值数值稳定 | dlogits | NaN/Inf、极端分布用例 |
| GRU/compact memory | step、reset、mask | BPTT 截断 | 分段与整序列一致性 |
| Optimizer/checkpoint | AdamW、lr scheduler | 恢复 optimizer state | 同 seed 中断/恢复一致性 |
| Export | 固定 ONNX shape | 不参与训练 | FP32 logits parity、INT8/BN fold |

**学生与场景接口要分离**：`src/model` 不认识 Gazebo/MiniWorld；数据加载器读统一 episode；训练器永远不解析 SDF；模拟器不会调用 CUDA backward。

### 8.3 先写什么、不写什么

优先做静态数据读取、基础模型可过拟合、训练日志、官方数值参考和 `RGB → action` 端到端回放；**先不做**通用自动微分、强化学习 PPO、分布式训练、混合 GPU 多 stream 内存池、自由形状 Tensor 编译器、完整光流/SLAM、大规模多 GPU 集群。

---

## 9. LicheeRV Nano → ESP32：真正的通信适配原则

### 9.1 当前固件没有独立 NAV UART；不能直接复用 CLI

目前 `Serial` 用于启动日志和文本 CLI，RC 走另一组芯片特定串口，MAVLink 通过 Wi-Fi 函数发送/接收，ESP32-C3 默认还关闭 Wi-Fi。V7 必须设计**独立 UART 接收模块**，先进行 GPIO/电压、波特率、串口冲突排查；不要假设某个 `Serial2` 在所有 ESP32/C3/S3 上都存在并空闲。

### 9.2 建议的协议字段和安全语义（草案，不代表固件已支持）

```text
NAV_FRAME_V1:
  MAGIC(2B), VERSION(1B), TYPE(1B), LENGTH(2B)
  SEQ(uint32), SENT_MONOTONIC_MS(uint32), TTL_MS(uint16)
  MODE(uint8), VALUE1(int16 scaled), VALUE2(int16 scaled)
  FLAGS(uint16), CRC16(uint16)

ACK:             只表示“CRC通过、时序有效、命令已被接收并仲裁许可”
EXECUTED:        表示已在规定控制周期应用（不是航点必然到达）
REJECTED:        带原因码（not armed/invalid mode/no heartbeat/manual takeover）
STOP_REQUEST:    请求结束导航自主任务，不直接断电
```

为避免 UART 时钟同步误差，TTL 判断首选“接收端到达后的本地 expiry time”，并用 `seq` 检测乱序/重放；若需要发送端年龄判断，必须先定义双方时钟映射和上限。一次命令超时后不能无限重复保持上一次前进指令。

### 9.3 飞控侧唯一安全仲裁权

人工遥控优先级最高；不允许视觉导航直接解锁、直接给 PWM、绕过 IMU 检查、电池保护和当前模式确认。导航超时、断链和不合法意图应进入**飞控侧已证明可执行的受控退化**；由于目前没有定高，不能写成“安全悬停”。

必须先在**无螺旋桨台架**完成：错误 CRC、乱序/旧 seq、过期 TTL、连续同一包、极端数值、错误模式、串口断开、Linux 停机、人工接管、飞控上锁情况下的命令拒绝；每项记录实际 `mode/armed/motors` 状态。

### 9.4 真机“允许启动飞行”门禁

在真实控制层面具备受控定高或其他经验证的飞行稳定方案、遥控随时接管、安全区/围栏、日志可靠、端到端延迟满足对应速度的刹停要求之前，**只允许录制视频/台架/系留/受控平台演示**；不得宣称已支持无外部状态反馈的开放场景自主飞行。

---

## 10. 代码文件夹 V7 重新规划：加入 `sim/`，但不破坏 V6 的上限

V6.1 的代码限制继续有效：`code/src` 默认 ≤24 个 `.cc/.cpp`、`code/cuda` ≤6 个 `.cu`、公共头 ≤10 个，业务源文件默认 ≤280 有效行、单函数 ≤70 有效行。**Gazebo 适配文件不应挤进模型代码目录**。新增一个 `sim/` 作为独立可审计系统，并设预算；所有数量是上限不是写满目标。

```text
TinyAerialVLN/
├── README.md
├── V7_RESEARCH_PLAN.md                # 本文
├── CODE_RULES.md                      # 全局代码硬规则
├── run.sh                             # 唯一一键入口，子命令路由
├── CMakeLists.txt                     # 仅必要目标，不变成庞大超工程
├── config/
│   ├── experiment.yaml                # 训练/数据/模型选项
│   └── deployment.yaml                # sim/board/transport/ttl profile
├── code/                               # 自研学生训练/导出/板端 C++
│   ├── include/tinyvln/               # ≤10 公共头
│   ├── src/{core,ops,model,data,train,export,board}/
│   ├── cuda/                           # ≤6 CUDA 源文件
│   └── tests/
├── sim/                                # V7 新增：与模型独立
│   ├── worlds/                         # 仅最小世界模板、场景组合声明
│   ├── models/                         # 自研/合法引用的 UAV/Camera SDF
│   ├── include/                        # 模拟器边界头（独立预算）
│   ├── src/
│   │   ├── world_runner.cpp            # episode reset/step/health
│   │   ├── rgb_bridge.cpp              # gz.msgs.Image 接收/时间戳
│   │   ├── motor_bridge.cpp            # 4×u→4×omega
│   │   ├── cf_control_core.cpp         # 独立控制器 SIL
│   │   ├── task_generator.cpp          # 场景对象↔语言↔目标一致性
│   │   └── expert_planner.cpp          # 限制轨迹/教师意图
│   └── tests/                           # 能力/坐标/动力学/碰撞/reset 对拍
├── adapters/{babyai,miniworld,aerialvln}/
├── tools/{launcher.py,scene_tool.py,dashboard.py}/
├── assets/dashboard.html
├── manifests/                           # 版本、hash、split
├── docs/{ACTION_PROTOCOL,EPISODE_SCHEMA,CF_INTEGRATION,SIM_VALIDATION}.md
├── artifacts/{scenes,runs,recordings,reports}/   # gitignored
└── third_party/                         # pin ref + LICENSE + patch，不拷入 code/
```

### 10.1 新增硬上限与职责

- `sim/src/` **≤6 个核心 `.cpp`**（上述是预算名单）；`sim/include/` **≤6 个 `.hpp`**；单个源文件建议 ≤300 有效行。需要额外的 `gazebo plugin` 时优先替换某文件或用 ADR 说明为何确需新增，不得为每个场景复制同一桥接器。
- `sim/worlds/` 只保留小量**手写模板**，大规模随机 SDF 一律生成到 `artifacts/scenes/`，绝不提交 1000 个世界文件。
- 仿真器外部资源禁止复制进 `code/`，Gazebo 确切版本、Fuel 资产版本与 Hash 进 manifest；从高飞团队借鉴的内容要标出处/许可/适配点。
- **三个既定主要可执行目标不变**：`tinyvln_train`、`tinyvln_eval_bridge`、`tinyvln_board`；Gazebo system plugin 可作为共享库由 `gz sim` 加载，仿真原子工具经 `run.sh sim` 调度，不无序新增一批顶层训练可执行程序。
- **所有 `sim/` 模块不得依赖 `code/src/train` 或 `cuda`**；学生网络只经版本化 `PolicyABI` 连接，避免编译依赖循环。
- 正式固件单独版本/仓库维护；`cf_control_core.cpp` 首先是模拟器复刻/独立参考，不承诺直接覆盖 CF-Drone `.ino`。

### 10.2 代码审查强制检查

每次 PR 必须回答：

1. 引用了哪个真实源码行或哪份 Gazebo API？该断言是否已经能测试？
2. 输出文件受哪个 ABI/Schema 约束？是否改了 action ID、单位、坐标系？
3. 本次是否能演示 **previous RGB → action → physical next RGB**？不能则只能报告局部 gate。
4. 是否增加了学生不可见 GT/Oracle 输入？必须由测试拦截。
5. 是否新增了 Gazebo/PX4/ROS 依赖？为什么旧路径不够？
6. 新代码、激活内存、日志、平均与 P95 延迟增加多少？
7. 出错后是否明确上报错误码并自动进入正确安全状态？
8. 有无许可信息与第三方 git commit/hash？
9. 是否遵守预算、避免并行重复实现？

---

## 11. 一键启动：将 V6.1 `run.sh` 扩展为统一入口

`run.sh` 仍然是唯一入口；V7 需要新增子命令，**以下命令是待实现 CLI 合同，不代表 V6.1 现有骨架已经支持**：

```bash
./run.sh doctor                         # GPU/Gazebo/CUDA/固件/板卡/端口能力自检
./run.sh scene --preset forest --seed 42 --count 24
./run.sh sim smoke                     # 官方 X3 动力学无视觉验证
./run.sh sim camera                    # 测前视 RGB + 相机正确安装
./run.sh sim manual --scene forest_42  # 手动/脚本控制 + 动态物理
./run.sh sim cf-sil --scene forest_42  # CF 姿态控制器软件在环
./run.sh sim expert --scene forest_42  # 语言任务+特权专家
./run.sh sim record --split train      # 训练轨迹生成
./run.sh train --run-id bc_baseline_001
./run.sh eval --run-id bc_baseline_001
./run.sh dashboard --port 8765
./run.sh board doctor                  # 需要真板连接才判 PASS
./run.sh uart dry-run                  # 无桨/模拟串口协议回放
```

### 11.1 交互菜单建议

```text
TinyAerialVLN V7
[0] 环境和能力自检             [1] 官方 X3 动力学 Smoke
[2] RGB 相机预览              [3] CF-ControlCore SIL
[4] 生成训练场景及语言任务    [5] 专家闭环与数据录制
[6] 学生训练 + 实时进度       [7] 验证/闭环评测
[8] 打开本地 Dashboard        [9] SG2002 / UART 台架自检
[q] 退出
```

每个入口运行前应检查依赖、输入文件、ABI、端口占用、数据 hash。某阶段未实现时必须提示 `FEATURE_NOT_IMPLEMENTED` 并返回非 0；`sim camera` 无法输出有效 RGB 时不能打印“仿真成功”。

### 11.2 Doctor 的最低输出

```text
[OK] OS: Ubuntu 24.04
[OK] Gazebo: Harmonic / gz-sim 8.x
[OK] GPU: RTX 5090 driver=... render_backend=ogre2
[OK] Scene: forest_42.sdf XML/schema valid
[OK] Vehicle: dynamic link + inertia + 4 rotors
[OK] Camera: /tinyvln/front_rgb/image 640x360 30Hz
[OK] Motor: /X3/gazebo/command/motor_speed, 4 fields rad/s
[OK] Sync: image timestamp advances with simTime
[BLOCKED] CF-SIL: no compiled controller core
[BLOCKED] Board: LicheeRV Nano not connected
```

上面是**期望报告格式示例**，不能预填成真实状态。Doctor 检查必须调用系统 API/实际订阅或运行测试，禁止根据文件名存在就标绿。

---

## 12. 训练可视化和仿真可视化：记录真实指标，不只是进度条

### 12.1 终端进度条

训练必须显示实际已完成的 `global_step/total_steps`，并以移动平均打印 `loss_action/loss_stop`、学习率、梯度范数、吞吐、GPU 显存、设备错误、最近验证分数；**进度 100% ≠ 验收通过**。训练 NaN/Inf 必须终止并保存错误上下文。

```text
[TRAIN][run=bc_001] epoch=4/20 step=4812/26000 [######----] 18.5%
  loss=0.833 action=0.681 stop=0.152 acc=0.72 stop_f1=0.43
  lr=2e-4 imgs/s=155 gpu_mem=... last_ckpt=step_4000
```

以上数据为**显示格式占位**，不能视为实测。

### 12.2 训练 Dashboard 四栏

- **训练**：step/loss/grad/learning rate/throughput/checkpoint/status。
- **导航**：SR、OSR、NE、SDTW、STOP 误差、失败分类、每 scene 指标；离线训练指标与在线 SR 明确区分。
- **仿真**：实时 RGB、俯视世界 pose（**只给开发者，非学生输入**）、语言任务、当前 stage、专家/学生动作、碰撞、轨迹、物理时钟与 RTF。
- **板端**：camera→preprocess→NPU→CPU→UART 的 P50/P95/P99；帧龄、串口 ACK、命令超时、RAM/CPU/NPU 状态。

沿用 V6.1 的原生 HTML+Python 标准库只读 JSONL 看板；新增实时图像可用固定速率 JPG snapshot+JSON metadata，由服务器仅从 `artifacts/` 读取。**网页端不能直接向 ESP32 发送飞行指令**。

### 12.3 JSONL 指标事件合同

```json
{"kind":"train","run_id":"bc_001","step":1200,"loss_total":0.73,"lr":0.0002,"timestamp_utc":"..."}
{"kind":"sim_step","episode_id":"s1_042","step":34,"sim_time_ns":1700000000,"camera_ts_ns":1666000000,"policy_action":"turn_right","collision":false}
{"kind":"eval_episode","episode_id":"s1_042","success":false,"collision":true,"stop_reason":"collision","path_length_m":12.4}
{"kind":"hardware","seq":125,"camera_to_uart_ms":62.3,"ack_ok":true,"ttl_expired":false}
```

这些是**schema 示例，非真实结果**。统一处理字段缺失、落盘失败、断点恢复与实验覆盖；敏感令牌与本机用户名不得进入公开日志。

---

## 13. 报错与恢复：按“哪个层能安全处理”分级

| Code | 何时出现 | 诊断内容 | 处理与下一步 |
|---|---|---|---|
| `GZ_NOT_INSTALLED` | 找不到 `gz` | PATH、OS、包版本 | 停止 sim，不假启动 |
| `GZ_RENDER_INIT_FAILED` | Ogre2/Vulkan 无效 | render 日志、驱动、GPU | 切 GUI/server 排障，暂停 RGB 实验 |
| `GZ_MODEL_NOT_DYNAMIC` | UAV 缺质量惯量或 static | 模型名、惯量、plugin | 拒绝标记物理闭环通过 |
| `GZ_MOTOR_TOPIC_MISSING` | 无 actuator 主题 | 期望/实际 topics、model namespace | 检查 motor 插件与 URI |
| `GZ_MOTOR_LAYOUT_MISMATCH` | yaw/roll/pitch 方向不符 | u[4] 与角加速度日志 | 检查索引/旋向/坐标 |
| `CAMERA_NO_FRAMES` | 一直没 RGB | plugin、topic、render/backend | 停止数据采集 |
| `CAMERA_FRAME_STALE` | 图像时间过期 | capture/current ts、阈值 | 丢弃样本，不沿用旧动作 |
| `SIM_IMU_FRAME_ERROR` | 静止/重力方向不符 | body accel, gyro, gravity | 不运行 CF-SIL |
| `PHYSICS_NAN` | 状态无限/NaN | inertia, dt, motor cmd | 截断 episode，保存 dump |
| `RESET_STATE_LEAK` | reset 后残存图像/GRU/速度 | previous/current episode IDs | 阻断评测并修 reset |
| `SCENE_UNSOLVABLE` | 目标无碰撞自由路径 | scene hash, seed, clearance | 有限重采样并记录 |
| `TEACHER_ACTION_INFEASIBLE` | 专家轨迹超姿态/速度限制 | control limit, segment | 修专家或标记失败，不当标签 |
| `ACTION_PROTOCOL_MISMATCH` | 官方 8 动作误当 UART | source/target ABI version | 拒绝执行 |
| `FC_ALT_HOLD_UNAVAILABLE` | 请求需要定高的真实动作 | mode/available sensors | 禁止实机自主控制 |
| `UART_PORT_CONFLICT` | 与 CRSF/CLI 引脚重叠 | board type、GPIO、baud | 阻止 UART 上线 |
| `UART_CRC_ERROR` | 串口帧校验错 | seq, CRC, length | 拒绝命令，飞控侧超时兜底 |
| `UART_ACK_TIMEOUT` | 未收到收包确认 | seq, timeout | 上游停止发布新动作 |
| `UART_EXECUTE_TIMEOUT` | 已 ACK 却未应用 | seq, last applied | 飞控故障/接管 |
| `TRAIN_NONFINITE_METRIC` | Loss/Grad 不可数 | sample, step, layer | 停训并保留最后可靠 CKPT |
| `ONNX_CVIMODEL_INCOMPATIBLE` | SG2002 算子编译不支持 | op, shape, SDK version | 改算子，不伪造板端支持 |

日志建议统一 `severity/code/module/run_id/scene_id/episode_id/sim_time/monotonic_time/hint/cause`。对外 `INFO` 不超过低频统计，详细调试写 trace 文件，不得把高频文件 IO 放到电机控制周期。

---

## 14. 分阶段实施门禁：先技术证据，后科研实验，最后实机

| Gate | 实现范围 | 可客观验证的交付物 | PASS 标准 | 不通过时 |
|---|---|---|---|---|
| **GZ0** | 官方 X3 Gazebo 动力学 | 录屏+电机指令/pose log | 改 4 电机命令可改变速度/高度，reset 正确 | 不加入模型 |
| **GZ1** | X3 固定前视 RGB | 真实 image+CameraInfo/log | 帧连续、方向正确、机体移动时画面变化 | 修传感器/变换 |
| **GZ2** | V6.1 森林 SDF + X3 | 带地标的可碰撞 world | 障碍存在、碰撞体有效、飞行不穿模 | 修 SceneBuilder |
| **GZ3** | 控制器动作闭环 | Scripted Policy 控制下的 episode | 稳定完成起飞、限定动作、目标停止、下一帧 | 检查控制/定高 |
| **CF0** | CFMath 黄金对拍 | 输入/输出/角速率日志 | roll/pitch/yaw/motor mixing 方向一致 | 修 SIL 适配 |
| **CF1** | CF-Core SIL | 航姿控制响应曲线 | 受控扰动不发散、限制与 dt 测试通过 | 不进入真实飞行 |
| **CF2** | 高度能力/受控速度 | 真实或声明的模拟控制 profile | 能执行有界高层意图且没有未声明 GT | 仅保留基准控制器结果 |
| **DATA0** | 语义任务+专家 | 至少若干可重放完整 episode | 指令/地标/专家一致、训练输入无 GT | 修任务/专家 |
| **TRAIN0** | 自研最小 CNN+文本+GRU | 数值测试+过拟合+恢复日志 | 梯度、loss、文本干预通过 | 不加 ECPT/KD |
| **TRAIN1** | 闭环学生 BC/STOP | 未见场景评估 + 失败分型 | 与基线可比且可复现 | 诊断视觉/语言/控制 |
| **METHOD** | ECPT / 进度证据机制 | 等容量配对消融 | 语言阶段/STOP/导航有独立可辨收益 | 删无效模块 |
| **KD** | 教师蒸馏 | 教师优势+收益消融 | KD 有独立净收益，无 GT 泄漏 | 删除 KD |
| **EDGE** | SG2002 INT8 + camera→UART | 板端 P50/P95/P99、峰值内存 | 设备真实实测达给定预算 | 缩模型或改算子 |
| **HIL** | ESP32 无桨+真实串口 | 接管/断链/TTL/CRC 记录 | 所有失效模式安全行为明确 | 不飞 |
| **FLIGHT** | 受控实机试验 | 法规与安全验收报告 | 飞控能力、人工接管、安全场地全部到位 | 禁止自由飞行 |

### 14.1 不能直接把不同层的 PASS 当作下一级 PASS

- 只有 GZ1 不等于 GZ3：看到摄像头画面不意味着飞行受控。
- 只有 CF1 不等于 CF2：姿态 PID 对拍不等于定高和定速。
- 只有 DATA0 不等于 TRAIN1：离线模仿正确不等于在线闭环稳定。
- 只有 AerialVLN val_unseen SR 变好不等于 FLIGHT：官方基准没有完整物理执行时延与惯性约束。
- 只有 `cvimodel` 编译成功不等于 EDGE：必须有真实图像采集 + 运行 + 串口全链路延迟。

---

## 15. 论文实验设计：哪些证据能支撑创新

### 15.1 研究 baselines

| 模型 | 作用 | 使用条件 |
|---|---|---|
| RGB-only Seq2Seq/CMA-lite | 空中 VLN 最基础参考 | 同 RGB、指令、动作与场景 |
| TinyVLN-BC | 自研学生基线 | 无 STOP 辅助/无进度 |
| TinyVLN-STOP | 增加独立终止校准 | 同参数/数据与阈值调优规则 |
| TinyVLN-GRU+Progress | 普通弱监督进度 | 对照创新新颖性 |
| TinyVLN-ECPT | 提出的证据校准记忆 | 同容量、相同监督来源 |
| TinyVLN-ECPT+KD | 蒸馏消融 | 教师真实闭环先合格 |
| RGB-only Student INT8 | 板端部署实测 | 帧率/内存/P95 与 FP32 决策差 |
| Privileged Oracle | 专家上界 | 不与 RGB-only 当公平同模态比较 |

### 15.2 评测指标不能只看动作分类

在 Gazebo-PhysVLN：无碰撞任务成功率、STOP 准确性、失败分类、最小间隙、控制饱和、碰撞次数、动作平滑、物理轨迹长度、到目标误差、速度、路径效率、紧急退化计数、FPS/RTF、帧龄。

在 AerialVLN-S：官方允许的 SR、OSR、NE、SDTW 等为主；同协议 RGB-only 基线重跑；官方原论文 RGB+Depth 作为不同设定参考；高分辨率二次渲染另列 `Protocol-HD`，不能混入官方表。

小数据不能得到稳定的论文结论；对 `val_unseen` 做 episode 级配对结果，按路径和场景聚类处理不独立样本，报告至少多个训练种子、均值和置信区间，不把逐帧预测当成独立 episode。

### 15.3 语言模型真的有效吗？

务必加入：

- 同图像换不同指令，动作应随目标变化。
- 指令完全乱序、去掉方向词/颜色地标词：导航性能应出现可解释变化；否则模型可能只会避障。
- 按“先 A、后 B、最后 STOP”的阶段划分分析终止错误、进度指针回退、错误地标触发。
- 消融摄像头视场、轻度曝光漂移、图像模糊和指令同义改写，区分视觉/语言鲁棒性。
- 同物理参数与同控制器对比，避免不同动作执行器导致的虚假“模型增益”。

---

## 16. 关键假设、弃用条件、外部资源与风险表

### 16.1 可证伪假设

**H-physics**：使用真实 `MulticopterMotorModel` 和碰撞世界，比位置瞬移更能暴露动作连续性与时延问题。测试：同一策略，Kinematic vs Dynamic 的闭环碰撞/成功差。

**H-progress**：证据校准指令进度在多阶段导航的错停/漏停上比等容量 GRU 有稳定收益。测试：配对多种子、多场景的停止误差与 SR/NE。

**H-KD**：可信教师的风险与终止知识蒸馏，对未知场景有额外收益。测试：与 BC+同数据、同容量对照；负收益删除。

**H-Sim2Real**：相机、控制延迟和物理参数扰动训练可缩小离线视频与受控实机感知/动作差异。没有实体数据不作此主张。

### 16.2 立刻终止或降级的边界

- 若 Gazebo RGB 渲染在 5090 上不可用：先以官方 demo 和 Ogre2 排障，不把失败推给深度学习模型；可在仿真 host 隔离渲染，并保持同一数据合同。
- 若 CF-ControlCore 不稳定：允许独立基准控制器继续训练/研究，但论文中必须声明**CF 实机控制未验证**。
- 若定高和安全超时未实现：不接真实自主控制，只做受控台架。
- 若 ECPT 不优于基线：删去该方法分支或重新定位研究问题，不通过追加 ROI 和 VLM 掩盖失败。
- 若自研 CUDA 底层耗尽开发资源：退化到专用静态图 + cuDNN/cuBLAS backend，保留自己实现的损失、训练调度和部署，不为了造通用框架牺牲实验。

### 16.3 公开参考与用途

1. Gazebo Harmonic 官方安装： https://gazebosim.org/docs/harmonic/install_ubuntu/
2. Gazebo 8 官方 X3 四旋翼电机例程： https://github.com/gazebosim/gz-sim/blob/gz-sim8/examples/worlds/quadcopter.sdf
3. Gazebo Harmonic 传感器教程： https://gazebosim.org/docs/harmonic/sensors/
4. Gazebo MulticopterMotorModel API（8 系列）： https://gazebosim.org/api/sim/8/classgz_1_1sim_1_1systems_1_1MulticopterMotorModel.html
5. PX4 Gazebo SITL 对照例程： https://docs.px4.io/v1.18/en/sim_gazebo_gz/
6. Fast-Drone-250： https://github.com/ZJU-FAST-Lab/Fast-Drone-250
7. EGO-Swarm： https://github.com/ZJU-FAST-Lab/ego-planner-swarm
8. GCOPTER： https://github.com/ZJU-FAST-Lab/GCOPTER
9. Sipeed LicheeRV Nano 官方 Wiki： https://wiki.sipeed.com/hardware/zh/lichee/RV_Nano/1_intro.html
10. SOPHGO TPU-MLIR： https://github.com/sophgo/tpu-mlir
11. SG200X TPU SDK： https://github.com/milkv-duo/tpu-sdk-sg200x
12. 原始 `CF-Drone.zip`：用户本次已上传的第三方 ESP32 飞控源码快照，其 README/许可不自动证明在用户硬件上的实际运行状态。

---

## 17. V7 第一批实际执行任务与交付清单（按依赖而非盲目并行）

1. **CF-Drone Hardware Matrix**：查明实机具体 ESP32 型号、实际串口 GPIO、是否有额外的高度/速度传感器；记录源代码 git commit/hash 和许可。此步骤不需要改模型。
2. **Gazebo Official Demo PASS**：跑官方 X3 电机例程并保存 pose/motor log；验证物理引擎不是 `set_pose` 瞬移。
3. **Gazebo Camera PASS**：把前视相机挂到四旋翼 link，保存移动时的 RGB 与 timestamp；验证左右地标方位。
4. **V6.1 World Compose PASS**：将 `forest_42.sdf` 障碍组合到 X3 动态 world，验证真实碰撞与安全间隙。
5. **Manual Dynamic Loop PASS**：写一个最小 C++ `WorldRunner` 和 `MotorBridge`；scripted action 能改变下一帧图像，不使用学生网络。
6. **CF SIL PASS**：单独抽出 CF 姿态控制相关数学核心，做静态黄金对拍、扰动响应、dt/失效注入；这不是将 Arduino 库生拉硬套到 Linux。
7. **Language Expert PASS**：场景语义 → 长指令 → 专家动力学轨迹 → RGB/动作 episode 数据合同，验证所有教师标签执行可行。
8. **自研训练 PASS**：静态 C++/CUDA 学生完成小数据过拟合、渐进任务和语言交换干预；dashboard 记录真实训练事件。
9. **主方法实验**：在闭环基础上测试 ECPT、独立 STOP 和可选 KD；统计未见场景、碰撞与失败类型。
10. **SG2002/UART 台架**：板卡编译、延迟与协议故障注入；不跳过高度/安全控制前提，不直接自由飞行。

### 17.1 首个最小成功演示应是什么样

> 一条短英文指令：“Fly toward the blue gate, then turn right and stop near the red marker.”

可见 RGB 摄像头固定在**真正受力的四旋翼机体**上。无人机不是被脚本直接设置位置，而是通过电机/动力学改变位置；策略每步读取更新后的 RGB，并依据语言阶段决定动作。记录的 pose、碰撞与参考路径仅用于教师和评分。演示中分别显示 RGB、指令/当前动作、物理状态轨迹、碰撞与耗时，并保存可重复的 episode trace。

**只有这条链从输入图像到下一帧图像真实闭合，才算本次“Gazebo 的 RGB 相机与四旋翼动力学闭环问题已解决”。**

---

## 18. 最终判断

V7 不是把 CF-Drone 和 Gazebo 简单“拼起来”。它实际建立三层不可混淆的证据：

**一，CF-Drone 源码证据**——已有姿态与电机控制数学核心，但没有能直接支撑高层自主导航的定高/定速/UART 安全协议。

**二，Gazebo 物理证据**——必须有真实电机力、机体惯量、相机与环境共同作用，不能用位姿跳转冒充无人机飞行。

**三，学术研究证据**——只有完成前两层基础接口，才能评价视觉—语言—进度模型、STOP 校准和知识蒸馏是否让真正的在线导航变得更好。

项目继续坚持“小数据快速验证、核心算法 C++/CUDA 自研、LicheeRV Nano 部署、ESP32 飞控分工”的方向；但**拒绝把未具备的飞控能力、未运行的 Gazebo 闭环或未测得的板端 FPS 描述成既成事实**。
