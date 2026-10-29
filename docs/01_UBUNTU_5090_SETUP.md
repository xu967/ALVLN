# 01｜Ubuntu 24.04 + RTX 5090 服务器配置与验收

## 1. 目标环境（建议而不是当前机器已验证）

- 主机：Ubuntu 24.04 LTS x86_64；gcc/g++、CMake、Python 3.12、Git；RTX 5090；Gazebo Harmonic。若已有 ROS 2 发行版，注意其 Gazebo 依赖是否冲突。
- 显卡：先安装**支持 Blackwell 的 NVIDIA 驱动**，以 `nvidia-smi` 显示实际 GPU 和驱动为依据；CUDA 版本按驱动兼容矩阵选择，不只根据 `nvcc --version` 判断。
- C++/CUDA：C++17，CUDA Toolkit，后续模型算子可使用 cuBLAS/cuDNN；即使坚持“不用 PyTorch 主训练”，也可在 `verification/` 隔离使用 NumPy/少量参考框架生成 golden tensor。
- Gazebo：Harmonic 是 Gazebo Sim 8（gz-sim 8.x），Ubuntu 22.04/24.04 官方支持。官方安装教程：https://gazebosim.org/docs/harmonic/install_ubuntu/ 。
- PX4：可用官方 `make px4_sitl gz_x500` 做 baseline；此命令并不自动保证装有我们需要的 RGB 镜头。官方：https://docs.px4.io/v1.18/en/sim_gazebo_gz/ 。

## 2. 第一轮服务器命令

```bash
uname -a
lsb_release -a
nvidia-smi
which nvcc && nvcc --version
which cmake && cmake --version
which g++ && g++ --version
which gz && gz sim --help
./run.sh doctor
./run.sh test
./run.sh cpp-test
./run.sh ui
```

若 `nvidia-smi` 找不到 RTX 5090、驱动 API 出错、`gz sim` 不可用，记录为相应门禁的 BLOCKED；不要通过改文档绕过。

## 3. 安装原则

1. 先备份现有驱动与 CUDA；不要运行来源不明的一键驱动覆盖脚本。
2. `gz-harmonic` 从 Gazebo 官方仓库安装，避免将 Gazebo Classic 与 Harmonic 插件混装。审查 `ldd` 与 `GZ_SIM_SYSTEM_PLUGIN_PATH`。
3. 确认 headless/EGL 渲染是否能输出真实 RGB；如果无显示器，先验证 renderer 在当前 RTX 5090/驱动上是否工作。
4. **不直接安装历史 `code.zip` 内项目的全部 ROS/Python 依赖**。旧 ROS1、ROS2、AirSim、PX4 的版本可能冲突；在不同容器/venv 里隔离。
5. 为 Gazebo、官方 AirSim 场景、训练缓存、模型 checkpoint 分设磁盘目录；历史下载包不可直接解压覆盖最终 `cpp/` 或 `server/`。

## 4. GPU 与 CUDA 最小探针

- GPU：`nvidia-smi -L` 和 30 分钟压力/温度/显存检查（在合适的生产窗口做，不需要本包自动触发）。
- CUDA：编译一个简单向量加法测试，`cudaGetDeviceProperties` 核实设备名称、计算能力；随后再开启 FP32/FP16/Tensor Core 基准。
- 自研训练：只有 CPU 数学 oracle、有限差分和 CPU/CUDA parity 通过，才开始长时间训练。**不要用随机 loss 下降代替梯度正确性证明**。

## 5. 远程访问 UI

```bash
# 服务器：只监听本机
./run.sh ui
# 本地电脑：SSH 认证端口转发
ssh -L 8765:127.0.0.1:8765 user@server
```

**不要使用 `0.0.0.0` 直接向公网开放管理 API**：已实现 API 允许受注册表约束的进程启动，但它并不包含面向互联网的用户鉴权/配额/CSRF 完整防护。

## 6. 外部基准双探针（限时）

- `PROBE_AERIAL`: 记录官方场景/设置下载来源和 hash，headless 运行 1 scene 与 10 episodes，图像尺寸、帧率、reset 一致性、日志。UE4 不兼容时明确 BLOCKED，**不假称 Replay=SR**。
- `PROBE_GAZEBO`: 实际启动 `gz_x500` 或官方 X3 示例，验证 4 转子→姿态/位置→前视 RGB topics 的真实时序，写 trace。只有这项通过才称“物理闭环环境存在”。

## 7. 注意何时可以继续

如果无法复现 10 条 Aerial episode，但 Gazebo RGB/physics 可用，转到小场景 Phys-VLN，保留独立标准评测任务稍后再做。如果两个都不可用，暂停宣称闭环研究，保留纯离线/数值链。
