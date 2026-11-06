# 真机对齐仿真：学术/开源调研与硬件参数（2026-11-06）

- 时间：2026-11-06
- 范围：浙大 FAST Lab、中科大、微型四轴固件在环生态、空中 VLN 基准、3DGS 训练视觉策略；琛光 E1 MINI 三处来源（oshwhub 项目页、琛光文档、GitHub CF-Drone）的硬件参数。
- 来源：第一阶段调研 agent A（硬件）与 B（学术）的只读报告，原文在 `$DATA_ROOT/uav_rgb/tmp/phase2-20261106/reports/`。克隆的仓库在 `$DATA_ROOT/uav_rgb/tmp/research-acad/`（1.2 GB，只读参考）。
- 用途：解释本轮 `sim/cf_sitl/` 的设计取舍；列出上真机必须实测的参数。

## 1. 结论

1. FAST Lab 和各空中 VLN 基准都没有接近真机的微型机仿真。EGO-Planner-v2 / Fast-Drone-250 默认仿真把规划指令直接当里程计（`fake_drone/poscmd_2_odom.cpp`），和本仓库 X3 用真值跟踪速度是同一类问题。
2. 能借的是两块：Crazyflie 生态的固件在环（CrazySim、safe-control-gym 的 pycffirmware 锁步）和推力/延迟辨识脚本（Bitcraze `tools/system_id/`）；FAST 的真机标定做法（px4ctrl 的悬停油门在线估计）。
3. 没有任何公开参数对应 E1 MINI。`sim/dynamics/e1_mini.json` 里 `status=assumed` 的项只能当初值，上真机后按 §4 替换。

## 2. 调研摘要

### 2.1 浙大 FAST Lab（高飞组）

| 仓库 | 读了什么 | 可借 | 不能直接用 |
| --- | --- | --- | --- |
| `ZJU-FAST-Lab/ego-planner` `src/uav_simulator/`（仓库 GPL-3.0；so3_quadrotor_simulator / so3_control 标 BSD，维护者 Kartik Mohta，UPenn 代码；local_sensing GPLv3） | `Quadrotor.cpp`、`SO3Control.cpp`、`depth_render.cu`、`mockamap` | 电机一阶滞后（τ=1/30 s）+ 刚体 ODE 的写法；随机柱阵/迷宫地图生成 | ROS1；0.98 kg；SO3 环用真值按墙钟跑，不锁步；IMU 话题发的是真值姿态和运动加速度（不是比力），无噪声；CMake 写死 sm_75 |
| `ZJU-FAST-Lab/Fast-Drone-250`（无 LICENSE，readme「严禁商用与转载」；px4ctrl GPLv3） | `px4ctrl/controller.cpp`、`PX4CtrlFSM.cpp`、`ctrl_param_fpv.yaml`、VINS 配置 | **`estimateThrustModel`：遗忘因子 0.998 的 RLS，把 35–45 ms 前的油门指令和当前 z 加速度配对，估「油门→加速度」增益，初值 g/hover_percent**；`hover_percent` 靠自动起飞调（起不来调大、冲过 1 m 调小）；IMU 提到 200 Hz；VINS 外参在线自标定 | 和 PX4 推力接口、VINS 里程计绑死；1.1–1.2 kg、4S |
| 近年学习类论文（Flying on Point Clouds 2503.00496、Precise Aggressive Maneuvers 2604.05828、Safety-Shielded RL 2602.08653、VLA-AN 2512.15258 用 3DGS 造数据、FastSim「coming soon」） | arXiv 全文 + 组织下 70 个仓库 | 无代码可借 | — |
| `Ground-effect-controller`（2506.19424，GPL-3.0） | README | 单电机台架 + 悬停反推推力系数 kT 的方法；5 参数地效模型 | 1.6 kg 机，数值不可迁移；固件在加密网盘 |

### 2.2 中国科大

| 仓库 | 内容 | 可借 | 坑 |
| --- | --- | --- | --- |
| `qinliangql/D-NAV`（RSS 2026，MIT；LICENSE 署名 Zhefan Xu，疑改自 CMU NavRL） | Isaac Sim + OmniDrones Hummingbird；策略输出速度交给 LeePositionController 20 Hz；LiDAR 阈值判碰撞 | 无（依赖 PyTorch/Isaac/LiDAR，接口是速度控制） | — |
| `ustc-arg/Safe-Robot-Learning-Competition`（IROS 2022 Crazyflie sim2real 竞赛，`beta-iros-competition` 分支，MIT） | `firmware_wrapper.py`、`level3.yaml`、`param_identification/` | **pycffirmware 与 PyBullet 500 Hz 锁步；动作/传感延迟按固件 tick 计；level3 随机化质量、惯量、输入噪声、外力** | 与上游 utiasDSL 逐字相同：陀螺用欧拉角差分代替；状态直接喂真值；加计/陀螺低通截止频率写反；无电机延迟 |

### 2.3 微型机固件在环

| 工作 | 结构 | 本轮取舍 |
| --- | --- | --- |
| CrazySim（ICRA 2024，GPL-3.0；`llanesc/crazyflie-firmware` crazysim 分支 + `crazyflie-simulation` MIT） | gz-sim8 插件；真固件独立进程，Gazebo 经 UDP 发 1 kHz 无噪声 IMU，固件传感任务阻塞等 IMU 包定节拍；电机用 `multicopter-motor-model`（τ 升 0.0125 s / 降 0.025 s） | 借「按 IMU 节拍跑固件」；**不借**异步 UDP（PreUpdate 只发最新电机指令，不等固件算完，不是锁步）和 SDF/PWM2OMEGA 两处不一致的 motorConstant |
| Bitcraze `tools/system_id/` | 倒装台架随机 PWM 采样，拟合 F=c00+c10·pwm+c01·V+c20·pwm²+c11·pwm·V；`plot_motor_delay.py` 一阶 λ=16 s⁻¹ | 上真机测推力曲线用这套方法（§4） |
| okalachev/flix `gazebo/`（无 LICENSE） | Gazebo Classic 11 插件里直接编固件；65 g | 借「每个物理步调一次固件」；**不借**偏航改真值、推力∝占空比、电压写死 4.2 V；另发现 maxTorque 代码值是注释的 10 倍、力臂 0.0354 与 SDF 桨位 0.0424 不一致 |
| crazyswarm2 `np.py` / gym-pybullet-drones / RotorPy（MIT） | 无阻力无延迟 / 地效 + 线性转子阻力 / 电机 τ + IMU 偏置随机游走 | 借 RotorPy 的 IMU 偏置 + 随机游走结构 |

### 2.4 空中 VLN 基准（AerialVLN、TravelUAV、CityNav、OpenFly、UAV-ON）

载具要么瞬移（`simSetVehiclePose(ignore_collision=True)`）要么交给 AirSim SimpleFlight 1 m/s；成功半径 20 m；OpenFly 动作表写 15°/5 m、位姿更新实际 30°/3 m，且无碰撞判定。只借 SR / OSR / NE / SPL / nDTW 的定义，米级瞬移和 20 m 阈值与 118 mm 机体不在一个尺度。

### 2.5 3DGS 训视觉策略

- SOUS VIDE + FiGS（Stanford MSL，GPL-3.0）：推力 + 角速度指令，角速度瞬时到位，无转动动力学和电机延迟；acados 100 Hz；splatfacto 渲染；无碰撞；采数据时质量 ±0.5 kg、kt ±2.1 随机。
- GRaD-Nav（NVIDIA 非商用）：PD 角速度环 + 二次阻力；碰撞 = 到点云最近距离 <0.2 m，但只看前向 120° 锥。
- 两者都依赖 PyTorch + nerfstudio，只借思路：碰撞几何与渲染分开、全向、沿轨迹扫掠、只用按不透明度筛过的高斯（本轮 REVIEW #3 修法）。

### 2.6 对比表

| 工作 | 仿真器 | 动力学真实度 | 传感器模型 | 动作空间 | 与本项目相关度 |
| --- | --- | --- | --- | --- | --- |
| FAST uav_simulator | ROS1 自写 ODE | 中，SO3 用真值 | 点云深度，无噪声 | SO3 指令 | 中 |
| Fast-Drone-250 | 真机 PX4 | 真机 | D435 + VINS | 位置 | 高（标定法） |
| D-NAV | Isaac | 中 | LiDAR | 速度 | 低 |
| 竞赛队 / safe-control-gym | PyBullet + CF 固件 | 中，锁步 | 真值 | 轨迹 | 高（SITL 写法） |
| CrazySim | gz-sim8 + CF 固件 | 中高 | 无噪声 IMU | cflib | 高 |
| flix gazebo | Gazebo 11 + 上游固件 | 中，线性推力 | IMU 白噪声 | 摇杆量 | 高 |
| gym-pybullet / RotorPy | Python | 中 | 无 / IMU 偏置 | 转速 | 中（参数） |
| 5 个 VLN 基准 | AirSim/UE/3DGS/GTA | 无 | RGB-D | 米级离散 | 低（只借指标） |
| SOUS VIDE / FiGS | acados + 3DGS | 低 | RGB + 状态噪声 | 推力 + 角速度 | 中 |
| GRaD-Nav | 可微仿真 + 3DGS | 低到中 | RGB/深度 | 推力 + 角速度 | 中（碰撞判定法） |

## 3. 借鉴落地到本轮

| 借什么 | 来源 | 本轮落到哪 |
| --- | --- | --- |
| 每个物理步调一次固件、按 IMU 节拍锁步 | flix、CrazySim | `sim/cf_sitl/`：固件 `loop()` 每 1 ms 一次，`micros()` 打桩成仿真时钟；Gazebo 插件在 `PreUpdate` 里调 |
| 推力 F(duty, V) 多项式 + 一阶 τ、电压掉压 | Bitcraze system_id | `e1_mini.json` 的 `thrust_model`、`motor_tau_up/down`、`battery_*`；实测后替换 |
| IMU 白噪声 + 偏置 + 随机游走；不要用欧拉角差分代替角速度；两个低通别写反 | flix、RotorPy、safe-control-gym 的反例 | `sim/cf_sitl/` 传感器模型 |
| 悬停油门在线估计（RLS，配对 35–45 ms 延迟） | px4ctrl | 本轮执行器用积分自适应前馈；RLS 留给固件移植阶段 |
| 碰撞几何与渲染分开，全向、沿轨迹扫掠、按不透明度筛 | GRaD-Nav 的改进版 | `sim/3dgs/collision.py`（REVIEW #3） |

## 4. 硬件参数：读到的、假设的、要你测的

单源文件：`sim/dynamics/e1_mini.json`（顶层 `calibrated=false`；每项 `status`：`doc`=三处来源原文、`firmware`=CF-Drone 源码、`assumed`=三处都没有，取一个数并写区间）。

### 4.1 三处来源读到的（`doc`）

| 参数 | 值 | 出处 |
| --- | --- | --- |
| 裸机质量 | 约 80 g 含标准电池 | 琛光 specs |
| 轴距 | 约 118 mm；保护罩 STL 电机座圈圆心 (±42, ±42) mm，对角 118.8 mm | 琛光 specs + 下载页 STL |
| 电机 / 桨 | 8520 空心杯；桨 65–75 mm（BOM 原文「都可以」） | oshwhub BOM |
| 电池 | 1S，3.7–4.2 V，约 500 mAh，≥25C，5–10 min | 琛光 quick-start / faq、BOM |
| IMU | MPU6500，无磁力计，yaw 为虚拟值 | 琛光 quick-start |
| 定点模块 | VL53L1X ToF + PMW3901 光流（+ BMP388、QMC5883L）；开源 e7f507f 不驱动它们，只有官方闭源 bin 驱动；PMW3901 最小工作距离 0.08 m | 琛光 extended-module、官方 bin 字符串 |
| 固件控制参数 | 角速率 PID 0.06/0.1/0.001、角度 P 6/6/3、倾角上限 30°、偏航 300°/s、油门映射 0.10–0.90、偏航杆死区 0.1、1 kHz、RC 丢失 1 s 降落 | CF-Drone `control.ino`、`safety.ino` |

### 4.2 假设的（`assumed`，上真机后替换）

整机质量 102 g（80 + LicheeRV Nano W 5 + 相机 6 + ToF/光流 3 + 线/升压 4 + 保护罩 3.8；区间 92–110）；桨径取 65 mm；单电机满推力 38 gf@3.7 V；推力–占空比指数 1.5；惯量 7.5e-5 / 8.9e-5 / 1.49e-4 kg·m²；电机 τ 升 0.06 s / 降 0.12 s；反扭矩/推力比 0.006 m；线性阻力 0.04 N·s/m；电池内阻 0.06 Ω；陀螺零偏残差 0.1 °/s、飞行噪声 2 °/s；ToF 量程 2.5 m / 30 Hz / σ 1 cm；光流 100 Hz / σ 0.05 rad/s；相机按 Sipeed GC4653 估 H81°×V51°、720p@30、前 45 mm 水平安装、延迟 0.1 s。

**悬停余量警告**（按以上假设算，`e1_mini.json` 的 `hover_check`）：102 g 时悬停占空比 满电 0.734 / 50% 电量 0.824 / 开路 3.5 V 时 0.922。后两档超过 0.80 建议值，第三档超过固件油门上限 0.90，飞机会在固件低压降落之前就悬停不住。解法：实测满推力 ≥39.7 gf；或换 75 mm 桨（估计 50% 电量时约 0.74）；或减载荷（裸机 80 g 为 0.609/0.679/0.753）。**上真机第一件事是称重 + 台架测推力曲线。**

### 4.3 上真机要测的（按优先级，不阻塞本轮）

| 参数 | 怎么测 | 用在哪 |
| --- | --- | --- |
| 整机质量（含 LicheePi、相机、传感器、电池、线） | 0.1 g 秤 | 刚体、推重比 |
| 占空比→推力曲线（满电 / 半电 / 低电各一次，同时记电压） | 倒扣固定在秤上扫 0.1–0.9（Bitcraze system_id 方法；固件 `mfr/mfl/mrr/mrl` 只能 30%，扫占空比要用 MAVLink `SET_ACTUATOR_CONTROL_TARGET`，只在 AUTO 模式生效） | 电机/电池模型 |
| 悬停占空比 | 飞一次后 CLI `log dump` 看 `thrustTarget` | 校推力曲线、执行器前馈 |
| 打杆阶跃响应 | 同一份 `log dump`：rates、ratesTarget、attitude、attitudeTarget（100 Hz） | SITL 验收金标准（G1） |
| 电机中心位置、桨径、外包络 | 卡尺 | 力矩臂、碰撞体 |
| 三轴转动惯量 | 双线摆 | 刚体 |
| 电机时间常数 | 占空比阶跃 + 慢动作/麦克风测转速 | 电机模型 |
| 陀螺静置漂移 | 静置 60 s 记录 | 偏置模型（姿态漂移主因） |
| 控制环实际频率 | CLI `time`，WiFi 开/关各一次 | SITL 步长 |
| 飞控存的参数 | CLI `p` 全量导出（NVS 覆盖代码默认值） | SITL 参数 |
| 相机型号、内参、畸变、分辨率、帧率、曝光、端到端延迟、安装位姿 | 棋盘格标定（本机 cv2 可用）；LED 闪灯对拍测延迟 | 相机模型（本轮未做） |
| ToF / 光流型号与实测噪声 | 台架 | 传感器模型 |
| 线性阻力 | 松杆滑行，上方手机拍视频 + 地面刻度 | 刚体 |

把这些放进 `$DATA_ROOT/uav_rgb/real-drone/`（格式见该目录 README），SITL 标定脚本从那里读。

## 5. 许可（事实记录）

本项目为个人学习用途。CF-Drone 为 GPLv3 + 附加条款 v1.1，附加条款第三条对个人学习、教育研究用途不作限制；`sim/cf_sitl/` 自带 GPLv3 LICENSE，固件源码构建时从 `code/CF-Drone` 目录引用，不复制进仓库，保留品牌字符串。上面调研到的第三方仓库只读参考，没有复制代码。
