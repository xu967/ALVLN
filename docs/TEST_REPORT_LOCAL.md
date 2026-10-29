# V8 本地验证报告（更新至 2026-10-29）

## 执行位置与可信边界

**环境：当前隔离 Linux 容器**（不是用户指定的 Ubuntu 24.04 RTX 5090 服务器；无 `nvcc`、`nvidia-smi`、`gz`、SG2002 及 ESP32）。命令均通过当前源文件真实运行；无后台任务、无实体机试飞。

## 实际验证过的操作

| 验证 | 结果 | 说明 |
|---|---|---|
| `./run.sh test` | 36 tests, OK（2026-10-29） | 包括真实受控子进程启动/停止及暂停恢复、stopped 生命周期、CSV 异常与 worker 收尾、任务参数校验、审核 ZIP 跨文件静止快照及一致性、异常恢复、Host/Origin 拒绝、场景确定性与哈希 |
| `./run.sh cpp-test` | 1/1 CTest pass | CPU FP64 Linear/Softmax-CE 中心有限差分；未实现 CNN/GRU 的 CUDA backward |
| `./run.sh scene --seed 42 --difficulty S1` | 生成成功 | SDF+JSON 静态语义候选、A* 轨迹、SHA256。不是四旋翼 physics episode |
| 真实 HTTP 服务 smoke | `GET /`, `/app.js`, `/api/v1/runs`, `/api/v1/simulation/capabilities?task=aerialvln_s_v1` 全 200 | `runs=[]`, 仿真 `ready=false`，不会虚构模型/真实仿真 |
| `./run.sh doctor` | 返回真实设备工具状态 | 当前容器 g++, CMake、Python 可用；nvcc/gz/NVIDIA GPU 缺失 |

## 当前未验证的核心系统

- Gazebo Harmonic `MulticopterMotorModel` 或 PX4 `gz_x500` 在目标 RTX5090 上运行；
- Gazebo 前视 RGB sensor 的实际 Topic 数据以及对应真实 motor→pose 闭环；
- 自研 C++/CUDA TinyVLN 的 CNN/GRU/ECPT 训练结果；
- AirSim/UE AerialVLN-S 官方 10 episodes closed-loop；
- 真正 Unity LANI 自动仿真端到端；
- SG2002 NPU 的 ONNX→cvimodel 转换与真实设备时延；
- CF-Drone 的 UART/HIL 安全改造、实体飞行。

## 需要其他 AI 审核的具体点

本包的 `server/` 初版面向**单用户 SSH port-forward 的本机研究服务**，并没有公网认证。`config/registry.json` 默认空，既是安全限制，也是训练/仿真尚未接通的诚实标志。受控子进程测试仅在测试目录内运行人工编写的合规小程序，用来证明管理器能保存真实 CSV/事件/ZIP，不能据此宣称无人机训练成功。
