# 单目 RGB 无人机视觉决策：Ubuntu 从零配置与训练平台方案 V1.0

> 版本：2026-10-28  
> 目的：先在全新 Ubuntu 主机上安装**能工作的原生 C++/CUDA 研发基础和公开无人机仿真器**，再建立可训练的单目 RGB 动作决策基线，最后部署到荔枝派，通过 UART 向已有 ESP32 飞控发送高层动作。  
> **本手册只做环境规划与安装说明，不提供模型实现，也不声称安装完即获得会自主避障的模型。**

## 0. 最终选定的任务与边界

### 0.1 第一版到底要做什么

**输入**：一台前视 RGB 相机的当前图像（推荐后续增加短时帧历史），可选任务方向/目标约束。**不把 IMU 接进第一版学习模型。**  
**输出**：离散安全动作建议，例如 `FORWARD_SLOW`、`YAW_LEFT`、`YAW_RIGHT`、`SAFE_NEUTRAL`；模型**不输出 PWM、电机转速或未经保护的姿态控制量**。  
**执行**：荔枝派负责摄像头、图像预处理、模型推理、动作置信度与节流、UART；ESP32 负责既有姿态闭环/飞控和失效保护。  
**训练**：Ubuntu + RTX 5090 上用公开仿真器构造「图像 → 专家动作」轨迹；用自己实现的 C++/CUDA 小型 CNN / 时序网络进行监督学习；先验证闭环，再考虑深度优化。 

⚠️ **任务限制**：只有一张无目标信息的 RGB 图像，最多能学“当前局部该如何避障/探索”；无法推断未知的终点在哪里，也没有可靠的运动速度/尺度信息。若最终要「飞往指定地点」，后续必须添加可从外部指定的目标/方向，或通过视觉目标标记建立目标条件。**不能把“反应式避障”写成“完整自主导航”。**

⚠️ **飞行安全**：ESP32 是否能稳定悬停取决于其飞控和实际传感器；没有位置估计时，`SAFE_NEUTRAL` 只能表示一条**预先验证的中立/失效处理指令**，绝不默认真实悬停。初期仅做离线推理、仿真和断开螺旋桨的串口联调。

### 0.2 关于“加入 IMU 会不会更好”

- **第一阶段：不用 IMU 作为模型输入**，尊重当前设想，最快建立可部署、可评测的 RGB-only 基线。
- **第二阶段（建议）：RGB 多帧**：使用最近 3–5 帧（按时间顺序）推断相对运动；无需新增传感器，但受相机速度、帧率、遮挡影响。
- **第三阶段（消融实验）：RGB + IMU**：若 ESP32 已能提供 IMU 测量，先作为*可选*辅助输入进行对比，不强制用于主模型。IMU 能提供转动/惯性线索，但不是障碍距离传感器。
- 如果用户限制任何板级通信不得上送 IMU，也完全可以不做第三阶段，只需明确所达到能力的边界。

### 0.3 代码“自己造”指什么

计划**不使用 PyTorch 训练新模型**。自研范围是：`Tensor` 张量与内存布局、CUDA 计算核（或 CUDA/cuBLAS/cuDNN 调度）、算子前后向、优化器、数据加载、训练检查点、模型权重导出、C++ 推理器、动作决策器。 

允许使用基础设施：**CUDA Runtime、cuBLAS（矩阵库）、cuDNN（可选神经网络原语）、OpenCV（摄像头/图像处理）、Eigen、CMake、Gazebo**。复用底层库不等于使用 PyTorch。第一版不推荐从零编写 GPU 驱动或仿真引擎。

**纠正一个可能的误解**：高飞团队的部分轨迹规划、求解器确实是自研 C++，但不能据此推断其所有视觉强化学习模型“完全不使用 PyTorch”。要判断具体项目，应分别查代码依赖；**不应把不用 PyTorch 当作论文创新本身**。

---

## 1. 主环境选择与数据集结论

### 1.1 明确采用的配置

| 层级 | 选择 | 原因 |
|---|---|---|
| 主机 OS | **Ubuntu 24.04 LTS Desktop x86_64** | 为 RTX 5090、CUDA 13.x、Gazebo Harmonic 提供较新的维护基础 |
| GPU | RTX 5090，CUDA Compute Capability 12.0 | 适合自己开发 CUDA kernel；要核对实际驱动与显存 |
| C++ | 系统 GCC/G++，CMake、Ninja、C++17 | 后续 C++/CUDA 自研训练与推理 |
| CUDA Toolkit | **13.1，作为固定起点** | 保留版本可复现；不是声称它是唯一或最新可用版本 |
| 视觉/矩阵基础库 | OpenCV C++、Eigen、cuBLAS；cuDNN 可选 | 不引入 PyTorch |
| **主要仿真** | **Gazebo Harmonic** | Ubuntu 24.04 官方支持；有物理系统、相机和碰撞几何，可开发 C++ 插件 |
| ROS 2 | **先不安装**；后期需要时用 Jazzy + ros_gz | 第一版只验证 Gazebo、C++ 数据通道；避免空白机一开始过度复杂 |
| 第一版数据 | **Gazebo 自建可交互轨迹** | 图片必须和专家动作、状态、碰撞结果同源同时间戳 |
| 外部基准 | FlightBench（后续，隔离旧环境）、AerialVLN-S（旧 VLN 基准） | 对照别人的场景与指标，但动作协议不同 |
| 板端平台 | **待确认荔枝派准确型号** | LicheePi 4A/Zero/RV Nano 的 CPU/NPU、系统和工具链完全不同 |

**不采用的初始主线**：

- **FlightBench 直接安装到主机**：其官方主要测试环境是 Ubuntu 20.04、ROS Noetic、Gazebo 11、GCC 9、Python 3.8，且学习型 baseline 自带训练框架；在空白 Ubuntu 24.04/RTX 5090 上把它作为第一项安装，兼容性成本过高。以后用独立容器/专门旧环境进行基准测试。
- **OpenFly 全量安装**：用户上传的仓库 README 推荐 Ubuntu 22.04、ROS 2 Humble、Python 3.10，服务于视觉语言导航和数据生成，不直接等于 ESP32 离散控制动作监督。先保留 ZIP，不在主机混装旧依赖。
- **AerialVLN 八动作直接发送给飞控**：原工程含约 5m 离散位姿移动、STOP 与偏航动作，属于 benchmark 合同，不是可供 ESP32 安全执行的控制协议。
- **Isaac Sim/3DGS 第一日安装**：图形与系统负担更高；等自研训练器、Gazebo 最小闭环通过再考虑。

### 1.2 用公开数据集时怎么用

| 来源 | 本项目中的角色 | 不能直接做的事 |
|---|---|---|
| **FlightBench / Flightmare** | 借鉴单机视觉导航场景、专家策略、轨迹数据生成和评测方案 | 不能把原始控制标签无转换地当作 ESP32 命令 |
| **Mid-Air** | 单目视觉预训练；部分数据提供 RGB、深度/相机姿态等多模态监督 | 没有本项目的 UART 四动作专家标签，不是端到端动作数据 |
| **AerialVLN-S** | 沿用历史语言导航评测；可研究离散视觉动作，但要保留独立动作协议 | 不能把仿真位姿跳变当作真实低速飞行动作 |
| **OpenFly** | 后期外观泛化/场景多样性与长程语义导航 | 不是直接可下载即用于四分类避障的训练集 |
| **Gazebo 本项目数据** | **第一版模型最重要的训练来源** | 需要自己定义环境、专家、安全动作、评测，光安装软件不会自动出现数据 |

**首批数据目标（规划值，非已下载数据）**：先生成 200 个小 episode 检查协议；随后 2,000 个 episode 建立基线。按**场景实例/地图**划分 train/val/test，不允许在同一地图随机拆帧造成泄漏。每条保存 `rgb/timestamp/action_expert/action_executed/collision/episode_id/scene_id/seed`；额外真值位置、障碍几何只供专家和评测，不进入学生模型在线输入。危险接近、退出死角、正常前进、向左避障、向右避障要有足够样本，不能全是“向前”。

### 1.3 初版模型及动作定义（待 ESP32 协议确认）

- 输入：从实际相机分辨率统一预处理到 **128×128 RGB**；第一次从单帧起步，随后做 4 帧对比。先不使用预训练大 ViT。
- 模型：自研轻量 CNN（Depthwise/普通卷积按实测选），输出 **4 个动作 logit + 可选风险置信度**；初期控制参数量小于约 1M 为工程目标，不是硬约束。
- 候选动作：`0 SAFE_NEUTRAL`（安全中立，具体行为由 ESP32 定义）、`1 FORWARD_SLOW`、`2 YAW_LEFT`、`3 YAW_RIGHT`。
- 动作语义要包含**时长、速度上限、超时、状态确认**；例如 100–200ms 的短时高层指令，不是一次 5 米的跳跃。速度/角速率需在模拟与 ESP32 飞控接口一致后才确定。
- 若一个场景左右均不可通行，应选择安全中立，而不是强行给左右动作。低置信度时回退安全模式。
- 对于**没有明确目标的图像避障**，左右同样安全时不存在唯一正确标签：需要定义巡航方向或专家规则，避免同一画面互相矛盾的监督。

---

## 2. 空白 Ubuntu 安装流程（按顺序执行）

> 下述命令针对**全新 Ubuntu 24.04 Desktop，x86_64，有桌面图形界面，主机配 NVIDIA RTX 5090**。**如果已经是 22.04/20.04，请不要照搬 CUDA/系统仓库步骤**；先核对实际版本。命令在你自己的 Ubuntu 终端执行，本报告没有远程修改你的机器。

### 步骤 0：检查系统与 GPU，不安装任何依赖

```bash
cat /etc/os-release
uname -m
uname -r
lspci | grep -Ei 'nvidia|vga|3d'
free -h
df -h
```

**通过标准**：看到 `VERSION_ID="24.04"`、`x86_64`，并能在 `lspci` 中看到 NVIDIA GPU。若版本/架构不同，先停在这里。

### 步骤 1：基础编译与调试工具

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y \
  build-essential cmake ninja-build git git-lfs \
  curl wget ca-certificates gnupg lsb-release software-properties-common \
  pkg-config unzip zip tar htop tree jq \
  python3 python3-venv python3-pip python3-dev \
  gdb clang-format clang-tidy \
  libopencv-dev libeigen3-dev libyaml-cpp-dev \
  libfmt-dev libspdlog-dev libserialport-dev \
  mesa-utils pciutils usbutils
```

检查：

```bash
gcc --version
g++ --version
cmake --version
ninja --version
pkg-config --modversion opencv4
```

### 步骤 2：安装 NVIDIA 显卡驱动（CUDA 前必须成功）

```bash
sudo apt install -y ubuntu-drivers-common
ubuntu-drivers devices
```

如果你使用的是全新系统，执行 Ubuntu 检测到的推荐驱动安装：

```bash
sudo ubuntu-drivers install
sudo reboot
```

重启后：

```bash
nvidia-smi
```

**必须先看到 RTX 5090 及有效驱动版本**。对于本手册采用的 CUDA 13.1，NVIDIA 对应发布分支为 **R590（匹配版本至少 590.44.01）**；虽然 CUDA 13.x 存在 R580 起的次版本兼容模式，但自研新架构 CUDA kernel 建议使用与 Toolkit 匹配或更新且经过支持的驱动。**若 `ubuntu-drivers` 安装后驱动仍低于 590，不要假设 CUDA 13.1 全特性可以正常编译和运行：先核对 NVIDIA 官方兼容性说明并升级到合适驱动分支。** `nvidia-smi` 中的 `CUDA Version` 是驱动可支持的 CUDA API 上限，**不代表 nvcc 已安装**。如显示 `No devices were found`、驱动未加载、Secure Boot 拒绝模块等，先解决驱动，**不要继续安装 Toolkit**。不要同时使用 Ubuntu apt 驱动和 NVIDIA `.run` 驱动安装器。

### 步骤 3：安装 CUDA Toolkit 13.1（只安装 Toolkit，不强行覆盖驱动）

NVIDIA 官方仓库示例，Ubuntu 24.04 x86_64：

```bash
cd /tmp
wget -O cuda-keyring_1.1-1_all.deb \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-13-1
```

暂时仅对当前终端配置 PATH：

```bash
export CUDA_HOME=/usr/local/cuda-13.1
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version
nvidia-smi
```

长期使用：检查 CUDA 路径存在后，把上述 3 行环境变量加入 `~/.bashrc`，或者由项目自己的 `env.sh` 加载；**不要盲目追加重复路径**。

5090 的 CUDA Compute Capability 为 **12.0**；以后编译本机 kernel 时在 CMake/CUDA 目标中显式指定 `sm_120`（CMake `CUDA_ARCHITECTURES 120`），避免沿用 4090 的 `sm_89` 编译配置。

**检查建议**：先完成 `nvcc --version`，随后用 CUDA Samples 的设备查询/简单 kernel 测试验证能启动内核；不能仅靠 `nvidia-smi` 宣称编译链正常。`cuBLAS` 随 Toolkit 使用；**cuDNN 不作为第一日强制项**，待确认算子设计后按 NVIDIA 官方版本兼容表安装。

### 步骤 4：创建项目目录（不安装 PyTorch）

```bash
mkdir -p ~/uav_rgb/{src,third_party,config,data,experiments,models,logs,docs,build}
mkdir -p ~/uav_rgb/data/{raw,episodes,train,val,test}
cd ~/uav_rgb
```

建议采用如下逻辑职责（这里只建立目录，不写模型）：

```text
~/uav_rgb/
├── src/                 # 后续自研 C++/CUDA：tensor、ops、train、infer、policy
├── third_party/         # 明确版本的第三方库（不放自己训练框架）
├── config/              # 数据/模型/仿真/串口协议配置
├── data/
│   ├── raw/             # 公开数据原始版本
│   ├── episodes/        # 自建仿真 episode
│   ├── train/ val/ test/ # 严格分场景划分
├── models/              # checkpoint/权重导出
├── experiments/         # 实验配置与指标
├── logs/                # 时间戳、FPS、碰撞、UART日志
├── docs/                # 设计与验收记录
└── build/               # CMake 编译输出
```

如果未来用 Python 处理数据元信息，可以建立**只用于数据分析**的虚拟环境；不依赖 PyTorch：

```bash
python3 -m venv ~/uav_rgb/.venv-tools
source ~/uav_rgb/.venv-tools/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pillow pyyaml tqdm matplotlib
python -c "import numpy, PIL, yaml; print('data tools ok')"
deactivate
```

### 步骤 5：安装 Gazebo Harmonic（当前推荐的公开仿真主线）

**先独立安装 Gazebo，不急着安装 ROS**。Gazebo 官方支持 Ubuntu 24.04 + Harmonic。

```bash
sudo apt update
sudo apt install -y curl lsb-release gnupg
sudo curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
  -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

sudo apt update
sudo apt install -y gz-harmonic
```

基本验证：

```bash
gz sim --version
gz sim -v 4 shapes.sdf
```

**通过标准**：Gazebo GUI 能打开 `shapes.sdf` 场景，终端无致命渲染错误。若是远程 SSH 无图形桌面，需改用 headless server/虚拟显示；`shapes.sdf` 本身不是 UAV 训练场景。

下一次任务才继续搭建：**多旋翼动力学 + 前视 RGB 摄像头 + 可控的高层动作接口 + 碰撞真值 + 轨迹自动采集**。Gazebo 官方 `gz-sim` 源码提供 `examples/worlds/quadcopter.sdf` 多旋翼示例，但还需要正确增加相机传感器/控制插件/监督标签，不能把示例开起来就当训练环境完工。

### 步骤 6：按需安装 ROS 2（不是第一版阻塞项）

如果之后明确需要 ROS 的 topic/可视化/日志记录，再在 **Ubuntu 24.04** 上安装 **ROS 2 Jazzy** 和 `ros-jazzy-ros-gz`。ROS / Gazebo 官方推荐的配对是 **Jazzy + Harmonic**，不是 Humble + Harmonic。先看：

- ROS 2 Jazzy Ubuntu: https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
- Gazebo 与 ROS 配对: https://gazebosim.org/docs/harmonic/ros_installation/

**本 V1 不自动执行 ROS 安装**。如要装，按官方 `ros2-apt-source` 安装仓库，再安装 `ros-jazzy-ros-base`（或 `ros-jazzy-desktop`）和 `ros-jazzy-ros-gz`，避免直接添加旧版 ROS 1 / Noetic 源。特别注意：若已从 OSRF 安装 Harmonic，要检查 apt 是否意外切换/重复安装 vendor 包，以 `apt-cache policy` 和实际版本为准。

### 步骤 7：板端连接与系统准备（暂时只做检查，不刷机）

先确认荔枝派的**准确型号**、SoC、RAM、Linux 发行版、能否接摄像头、UART 引脚及是否有可用 NPU/NN SDK。**荔枝派不是一种统一的平台**。例如 LicheePi 4A 是 TH1520/RISC-V，官方资料列出最高 4 TOPS INT8 NPU，但这**不代表你的那块板**也是 4A，更不保证自研 CUDA 权重能直接执行在该 NPU 上。

如果实际为 LicheePi 4A：目标部署路线优先 `C++ 推理器 → ARM/RISC-V CPU 验证（按实际架构） → NPU SDK 支持时再移植`；主机 CUDA **不能直接交叉编译成荔枝派可用 GPU kernel**；如果确定是 LicheePi 4A，其 CPU 属于 RISC-V，而不是 ARM。若是无 NPU 的荔枝派，网络规模、量化和 FPS 约束将大幅收紧。

UART 联调原则：

- **先 USB-UART → PC 日志或 ESP32 的安全测试固件；拆掉螺旋桨**。
- 确认两侧信号电平、共地、TX↔RX、波特率、隔离供电、超时保护；**未经确认绝不直接接 5V TTL**。
- 串口数据帧应包含帧头、版本、序号、动作、持续时间、可选置信度、CRC 和响应/错误码。
- ESP32 必须限制动作持续时间、最大速度、失联处理、解锁状态；**没有新合法指令就不能继续无限执行上一条向前命令**。
- 第一版禁止通过视觉模型直接控制螺旋桨占空比。模型只能发送高层动作请求，ESP32 保留最终否决权。

---

## 3. 从“安装完”到“真正能训练”：还缺哪些组件

安装成功不等于训练成功。进入代码阶段前必须设计以下 6 个接口：

1. **SimObservation**：`uint8 RGB[H,W,3] + timestamp + episode_id`。学生策略只读取 RGB 及可选历史，不读取仿真位置/深度真值。
2. **TeacherPlanner**：从 Gazebo 真值障碍与导航目标生成安全行动；教师可用全局信息，但绝不能通过学生输入泄漏。
3. **ActionAdapter**：将专家连续速度/偏航行为量化为 4 类短动作，定义不能犹豫时的 `SAFE_NEUTRAL`、目标偏好与转向策略。须与 ESP32 实际支持的动作一致。
4. **DatasetWriter**：同步图像、教师动作、执行动作、碰撞、种子及版本；一条 episode 保持时间顺序，按场景做隔离。
5. **NativeTrainer**：C++/CUDA，支持小 CNN forward/backward、交叉熵、Adam、训练/验证、多随机种子、权重保存。不能只有一个能跑的 forward 就称完整自研训练框架。
6. **SimulatorEval**：在未知场景中由模型自己闭环执行，记录碰撞、无碰撞成功率、近碰撞、未响应、动作抖动、平均单帧推理耗时。

### 3.1 数据准备与训练阶段的最小实验

| 阶段 | 任务 | 通过条件（工程门槛，之后可调整） |
|---|---|---|
| E0 | Gazebo shapes 场景正常渲染；CUDA kernel smoke test | 无 GPU/驱动/渲染错误 |
| E1 | 开始/停止无人机模型；订阅 RGB；采集第一个 episode | 图像时间戳、动作、碰撞记录一致 |
| E2 | 专家控制单架无人机低速穿越简单障碍 | 专家可重复完成，失败可复盘 |
| E3 | 采集约 200 episodes；检查四动作分布 | 没有明显无效标注和数据泄漏 |
| E4 | 自研 CUDA 小 CNN 训练；与 CPU 参考数值比较 | 损失下降，前向/反向数值与梯度检查通过 |
| E5 | 模型在测试场景闭环运行 | 可报告成功率、碰撞率，不靠离线分类准确率代替 |
| E6 | 荔枝派离线相机推理 + UART 沙盒 | 实测端到端延迟、丢帧、超时与故障恢复 |

**核心评测**：`action_accuracy`（只辅助分析）、`collision_free_completion_rate`、`collision_rate`、`false_forward_rate`（危险时却前进）、`safe_neutral_rate`、`time_to_collision_warning`、`inference_p50/p95_ms`、`UART_timeout_rate`。非平衡数据必须报告每类混淆矩阵；不要只看 accuracy。

### 3.2 暂时不做的事项

- 不马上训练 7B VLM、PPO、DA、3DGS，不引入全部旧 AirVLN 模型；否则难判断最小闭环的错误。
- 不立刻同时支持 AirSim、Unreal、Unity、FlightBench、OpenFly、Isaac；先将 Gazebo 跑稳。
- 不马上开发自研 ROS、C++ 编译器或 GPU 驱动；把工程投入放在**视觉策略、时序结构、可靠动作决策、训练与部署**上。
- 不在没有可信位置估计/周围环境信息时声称“绝对安全自主飞行”；只能报告受控测试内的成功率和已验证保护。

---

## 4. 安装核验清单（达到这些条件再进入代码阶段）

- [ ] `cat /etc/os-release` 为 Ubuntu 24.04，`uname -m` 为 `x86_64`。
- [ ] `nvidia-smi` 正常识别 RTX 5090。
- [ ] `nvcc --version` 显示可用 CUDA 13.1。
- [ ] OpenCV C++、CMake、Ninja、Eigen 的安装检查完成。
- [ ] `gz sim --version` 可执行。
- [ ] `gz sim -v 4 shapes.sdf` 可以显示 GUI 或无 GUI 下正常启动服务。
- [ ] 项目目录与数据分区已建立。
- [ ] 没有把旧版 ROS 1/Noetic、ROS 2 Humble、多个 CUDA 工具链混装进主机。
- [ ] 已确认荔枝派准确型号及 ESP32 可接受的**高层动作协议**。

出现失败时，请先保存下列输出，用于后续针对性排障：

```bash
cat /etc/os-release
uname -a
lspci | grep -Ei 'nvidia|vga|3d'
nvidia-smi
nvcc --version
gcc --version
cmake --version
gz sim --version
```

若 `nvcc` 尚未安装，提示找不到命令属正常的阶段性情况，不必执行后续 CUDA 编译测试。

---

## 5. 官方参考与核验来源

**公开仓库 / 安装文档（2026-10-28 核对）**：

1. NVIDIA CUDA Linux 安装：<https://docs.nvidia.com/cuda/cuda-installation-guide-linux/>；固定版本 13.1：<https://docs.nvidia.com/cuda/archive/13.1.0/cuda-installation-guide-linux/index.html>
2. RTX 5090 Compute Capability：<https://developer.nvidia.com/cuda/gpus>；CUDA 13.1 驱动分支与兼容：<https://docs.nvidia.com/cuda/archive/13.1.0/cuda-toolkit-release-notes/index.html>
3. Gazebo Harmonic Ubuntu 安装：<https://gazebosim.org/docs/harmonic/install_ubuntu/>
4. Gazebo + ROS 版本兼容：<https://gazebosim.org/docs/harmonic/ros_installation/>
5. Gazebo Quadrotor 示例源码：<https://github.com/gazebosim/gz-sim/blob/main/examples/worlds/quadcopter.sdf>
6. FlightBench 官方平台/旧版本约束：<https://thu-uav.github.io/FlightBench/>、<https://github.com/thu-uav/FlightBench>
7. Flightmare：<https://github.com/uzh-rpg/flightmare>
8. OpenFly Platform：<https://github.com/SHAILAB-IPEC/OpenFly-Platform>
9. Sipeed LicheePi 4A 硬件：<https://wiki.sipeed.com/hardware/en/lichee/th1520/lpi4a/1_intro.html>
10. ROS 2 Jazzy 安装：<https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html>

**本地材料直接核对**：

- `web_code_review_20261027.zip / START_HERE.md`：AerialVLN-S 八动作及 STOP、仿真环境与既有评测合同；`source_snapshot/requirements*.txt` 显示老工程仍依赖 `torch`、`torchvision`，因此新 C++/CUDA 项目必须**独立建立**，不能宣称老工程已变成无 PyTorch。
- `OpenFly-Platform.zip / OpenFly-Platform/README.md`：推荐 Ubuntu 22.04、Python 3.10、ROS 2 Humble；并含 AirSim/UE/3DGS 场景说明，不等于已提供通用四动作无人机控制训练数据。
- `Flying_in_Clutter_Monocular_RGB_3DGS_DA.md`：所述 Actor 输入包括 RGB **及飞行状态向量**；不能把论文简单理解为“单张图片，无状态输入即可安全飞行”。
- `Swarm_of_Micro_Flying_Robots_in_the_Wild.md`：其高效时空轨迹优化与飞控是分层系统，不等于纯视觉动作分类模型。

---

## 6. 下一轮需要确认的两项硬件事实（不影响今天开始装主环境）

1. **荔枝派准确型号**：LicheePi 4A、LicheeRV Nano、LicheePi Zero 或其他？决定最终可用 CPU/NPU、量化/算子与部署方式。
2. **ESP32 现在可以接收什么指令**：速度/偏航角速度？目标姿态？仅 PWM/油门？有没有超时自动归中及解锁保护？这个决定我们模型的输出类别如何映射成飞控动作。

在这两个信息明确前，可以完成 Ubuntu + NVIDIA + CUDA + Gazebo 环境；**不能提前承诺模型在任何荔枝派上达到多少 FPS，或 ESP32 能够在无状态估计下安全悬停。**
