# uav_rgb：单目 RGB 四动作避障

本项目已实现自研 C++/CUDA CNN 训练器、Gazebo 专家采集、数据验收闸门、常驻推理进程和学生闭环评测。

当前交付模型是单帧 RGB + 有限词表 FiLM 的四动作策略；GRU 为实验原型，尚未验证真机部署。
最新状态统一见 [HANDOFF](docs/HANDOFF.md)，评测判据见 [EVAL_STANDARD](docs/EVAL_STANDARD.md)。
校园 pilot 用 `--task campus`，使用可见彩柱前安全区域、独立任务版本和拓扑留出；操作见
[校园基线说明](docs/CAMPUS_BASELINE.md)。旧柱阵 `base/l1` 保留独立历史口径。

## 环境与一键检查

在本目录执行：

```bash
source env.sh
bash tools/setup_env.sh              # 新建 .venv-tools（numpy + Pillow + opencv-python-headless，tools/requirements.txt），并跑一次 doctor
# cf_sitl 段需要固件源码；不设 CF_DRONE_DIR 时 check.sh 把 SKIP 写到 stderr，末行 PASS 会列出未检查的段（见下）
export CF_DRONE_DIR=<CF-Drone 目录>
"$UAV_RGB_PYTHON" tools/doctor.py --require-gpu --require-sim
bash tools/check.sh                  # 构建 + CTest + Python 自检 + cf_sitl（CTest/自检）
```

固件源码 CF-Drone（e7f507f）不在本目录：ALVLN 仓库里 `references/snapshots/CF-Drone.zip`
解包即是（与 e7f507f 只差 CRLF），或 `git clone https://github.com/songge8/CF-Drone`
后 `git checkout e7f507f`。

本次实际设备：Ubuntu 24.04、RTX 4080 SUPER（sm_89，见 `CLAUDE.md`；CMake 仍编 sm_75，靠 PTX JIT 运行）、
CUDA 12.8、Gazebo 8.15.0。早期记录的 RTX 2080 Ti / sm_75 是 2026-11-05 前的历史环境。
保留已有驱动、CUDA 与 Gazebo；不使用 PyTorch、TensorRT、cuDNN、cuBLAS、ROS 2。
`doctor` 检查命令与导入，`check` 才执行数值/训练测试；两者都不代替闭环验收。
无卡时可显式执行 `bash tools/check.sh --python-only`，它会打印跳过 C++/CUDA 的说明。

默认新构建目录为 `build/check`；可用 `UAV_RGB_BUILD_DIR` 覆盖。
最近一次从零克隆复现（2026-11-07，GitHub 镜像 6d7c258）：`check.sh` 全程 4 min
（构建 `-j4` 约 2 min），**CTest 30/30**、cf_sitl CTest 7/7、`cf_sitl_selftest` 33/33。
（接手开发先读 [`docs/HANDOFF.md`](docs/HANDOFF.md)；任务与验收见 [`docs/TASKS.md`](docs/TASKS.md)。）
默认构建与推理统一使用 `build/check`；`UAV_RGB_BUILD_DIR` 可同时覆盖两者。旧 `src/build` 不再是默认推理入口，保留用于历史复现。
`UAV_RGB_BUILD_JOBS` 默认 8。Python 工具用 `UAV_RGB_PYTHON`，Gazebo helper 用
`UAVRGB_GZ_PYTHON`（默认系统 `/usr/bin/python3`），避免混用 conda 与系统绑定。

## 数据与训练

所有新数据、权重和运行产物放在 `$DATA_ROOT/uav_rgb/`，默认
`DATA_ROOT=/root/rivermind-data`，不能放进 `code/`。短程采集示例：

```bash
source env.sh
# 渲染后端默认 gpu（NVIDIA EGL）。本容器的 EGL 库是占位文件，默认后端会在
# Ogre2RenderEngine::CreateRenderSystem 段错误（2026-11-07 实测 rc=-11、0/1 集），
# 必须显式走软件 GL：UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg
# 每次必须用新的输出目录：collect_parallel 和 uavrgb_train 都会拒绝复用非空目录
# （uavrgb_train 的 RunOwner 只认 --resume），所以这里按时间戳建一个。
mkdir -p /tmp/xdg && chmod 700 /tmp/xdg
export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg
RUN="$DATA_ROOT/uav_rgb/smoke/$(date +%Y%m%d-%H%M%S)"
"$UAV_RGB_PYTHON" sim/tools/collect_parallel.py \
  --instances 1 --episodes-per-instance 1 --steps 5 --hold 0.5 \
  --base-seed 42 --out "$RUN/collection"       # 实测 1 集 5 步 27 s
"$UAV_RGB_PYTHON" tools/preprocess.py --drop-collision \
  --out "$RUN/preprocessed" "$RUN/collection/train/42"
./build/check/uavrgb_train \
  --data "$RUN/preprocessed/dataset.uavr" \
  --epochs 1 --batch 5 --lr 0.001 --class-weights auto --drop-collision \
  --out "$RUN/train"
DISPLAY=:20 /usr/bin/python3 ui/monitor.py --run-dir "$RUN/train"
```

数据集带 `instructions.jsonl`（`--task l1` 或下面的合成夹具）时，`uavrgb_train` 会拒绝缺省
`--text-dim 0`：纯视觉基线显式加 `--vision-only`，文本分支给 `--text-dim 16`。
不开 Gazebo 的最小链（check.sh 用的同一套夹具，3 集 15 步，全程几秒；2026-11-07 从零克隆实测 2–7 s）：

```bash
"$UAV_RGB_PYTHON" tools/make_synthetic_episode.py --out "$RUN/synth"
"$UAV_RGB_PYTHON" tools/preprocess.py --out "$RUN/ds" "$RUN"/synth/ep{1,2,3}
./build/check/uavrgb_train --data "$RUN/ds/dataset.uavr" --epochs 1 --batch 5 --lr 0.001 \
  --class-weights auto --vision-only --out "$RUN/train-synth"
./build/check/uavrgb_infer --ckpt "$RUN/train-synth/ckpt_best.bin" < /dev/null
# 先打印一行结构头 {"in_h":120,"in_w":160,"in_c":3,"num_classes":4,...}，之后每收一帧
# 160×120×3 HWC uint8 就回一行 {"action":..,"probs":[..],"infer_ms":..}（协议见 src/include/uavrgb/infer.hpp）
```

只有 train 时会明确输出 `DEV_SPLIT_EMPTY`，JSONL 的评测 `split=train`。
`results.csv` 仍保留既有 `dev_*` 列名，因此必须结合事件判断评测数据来源。
正式实验应提供按实际场景隔离的 train/val/test，并覆盖全部四类动作。
旧 `ui/monitor.py` 保留用于兼容历史 DEMO/目录监控；新实验请使用 `ui/launch_studio.sh`，它会启动真实 C++/CUDA trainer，不会默认调用 DEMO。

仿真采集与闭环曾在 RTX 2080 Ti 上接入 EGL 渲染（2026-11-03，历史）；本容器只能走软件 GL（见上）。
CUDA 可用与渲染可用分别验证。
启动方式见 [仿真说明](sim/README.md)，实测与优化顺序见
[GPU 仿真实验](docs/experiments/2026-11-03-gpu-rendering.md)。

## 桌面端工作台（新增）

在 Ubuntu 桌面会话运行：

```bash
bash ui/launch_studio.sh
# 可选：安装 Ubuntu 应用菜单 / 桌面快捷方式
bash ui/install_desktop_entry.sh
```

它会打开原生 Tkinter `UAV Navigation Studio`，不是网页 DEMO。支持真实 C++/CUDA 训练启动与监控、训练/测试指标曲线、原始日志、实验自动归档、Gazebo 三个实时渲染视角（第三人称 chase、原始第一人称 RGB、固定顶视相机）及辅助二维轨迹地图、当前仿真学生策略演示和独立闭环评测。详细说明见 [`ui/README_STUDIO.md`](ui/README_STUDIO.md)。

**三视角说明**：模板 SDF 新增 `/uav/chase_camera` 虚拟追踪相机；原 `/uav/camera` 保持学生唯一 RGB 输入（原始 320×240，模型输入 160×120）；`/uav/top_camera` 是固定在世界坐标 `(2.5, 0, 6 m)` 的渲染顶视，80° HFOV 在地面覆盖约 10.1×7.6 m。三路是独立相机流，不是同步帧；二维轨迹地图另由 `/model/x3/pose` 与 SDF 碰撞几何绘制。旧 world 文件需要按新模板重新生成，才包含 chase/top cameras。

## 能力边界

已有数值核心、CNN/FiLM、数据读写与审计、常驻推理及闭环评测。
校园pilot目前是三族低模、定高四动作任务；GRU、开放指令、真机部署与充分跨地图泛化尚未交付。
`DeviceTrainState` / `BatchReader` 是未接入的实验基础设施，保留测试作为后续接入依据。

## 交接与证据

- [并行训练与正确性改造方案](docs/archive/PARALLEL_TRAINING_PLAN.md)（已归档，未实施）
- [本轮总审查与过程记录](docs/worklog/2026-10-29-1145-project-review.md)
- [核心修复与阅读清单](docs/worklog/2026-10-29-1205-core-review.md)
- [仿真实测与缺陷复现](docs/worklog/2026-10-29-1152-sim-review.md)
- [UI、设计合同与飞控审查](docs/worklog/2026-10-29-1205-integration-review.md)
- [历史交接及本轮更新](docs/HANDOFF.md)

本轮原始采集、stdout、环境 JSON、SHA256、训练和恢复产物位于
`/root/rivermind-data/uav_rgb/review-sim-20261029-115357/`。

## 2026-11-04 可选实验路线：写实 Gazebo + T1/T2 Teacher

详见 [`docs/realism_rl_v1/README.md`](docs/realism_rl_v1/README.md)。
新增 `campus_assets`、`--teacher-planner predictive`、可选 PPO Teacher 训练及测试。
旧默认场景与旧 Teacher 不受影响；PPO 无附带已训练 checkpoint。
本路线未运行 Gazebo 闭环/真机验证、未训练 PPO；上文 25/25 CTest 与 EGL 渲染结论属历史基线，
不适用于新实验路线，也不能据离线单测宣称导航性能提升。
