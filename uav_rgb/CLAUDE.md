# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`uav_rgb` 已实现单目 RGB 四动作 CNN、CPU/CUDA 数值算子、Adam、UAVR 数据读取、checkpoint/resume 和 Gazebo 专家采集。2026-10-29 本轮 13/13 CTest 通过，真实仿真 5 帧已完成预处理、GPU 训练和恢复；单场景冒烟不代表泛化或可飞。GRU/语言导航、学生闭环评测、板端部署尚未实现。当前入口见 `README.md`，完整证据见 `docs/worklog/2026-10-29-1145-project-review.md`。设计规格仍位于 `../TinyAerialVLN_Final_V8/references/reviews/UBUNTU_BASE_GUIDE_V1.md`。文档与代码注释保持中文。

Design boundaries that must not be violated:
- The vision model outputs only four discrete high-level actions: `SAFE_NEUTRAL`, `FORWARD_SLOW`, `YAW_LEFT`, `YAW_RIGHT`. It must never output PWM, motor speeds, or unprotected attitude commands; the ESP32 flight controller keeps final veto.
- Training/inference are to be self-written C++/CUDA (tensor, ops, trainer, inferencer). No PyTorch, TensorRT, or other training frameworks are part of the plan.
- The LicheePi model and ESP32 action protocol are still unconfirmed. Do not assume they support a given input format or FPS.

## Environment constraints (actual machine, not the spec's assumptions)

The machine is a single **NVIDIA GeForce RTX 2080 Ti (sm_75, 22 GiB / 21.49 GiB usable, 68 SM)**, not the RTX 5090 described in the spec. (Earlier records said "Tesla T4"; the current device node is `/dev/nvidia0`.)

- CUDA Toolkit **12.8** at `/usr/local/cuda-12.8`; driver 580.119.02. Do not install CUDA 13.x (driver caps at CUDA 13.0, and Turing/sm_75 gains nothing) and never compile for sm_120.
- Installed and usable: OpenCV 4.6 (C++), Eigen 3.4, Gazebo Harmonic 8.15 (`gz sim`), CMake 3.28 + Ninja, GCC 13.3, ncu/nsys/cuobjdump/nvdisasm.
- Intentionally **not installed**: cuDNN, PyTorch, TensorRT, ROS 2. Do not install them.
- `apt` installs are fine on this machine (the earlier `apt-gazebo` ownership note is obsolete — no such process exists, no dpkg lock held). Still do not run `apt upgrade` / driver / kernel changes.
- GUI rendering goes through software GL on display `:20`; no NVIDIA EGL in the container.
- Python (`.venv-tools`, Python 3.12.3) is for data tooling only: numpy, pillow, pyyaml, tqdm, matplotlib.

Every shell must first source the environment (it is idempotent). `env.sh` now derives `UAV_RGB_ROOT` from its own location, so source it wherever the repo actually sits (on this machine: `/root/claude-code-backup-20261028/code/uav_rgb/env.sh`, not the older `~/code/uav_rgb`):

```bash
source "$UAV_RGB_ROOT/env.sh"            # after first sourcing from the repo's actual path
source "$UAV_RGB_ROOT/.venv-tools/bin/activate"   # only for Python data tooling
```

## Commands

Build and run the standalone CUDA smoke test (core CTest and Python checks also exist):

```bash
source "$UAV_RGB_ROOT/env.sh"
cd "$UAV_RGB_ROOT/experiments/cuda_smoke"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cuda_smoke          # prints device info; must end with RESULT: PASS
```

Verify artifacts target only sm_75:

```bash
cuobjdump -lelf build/cuda_smoke   # expect only cuda_smoke.1.sm_75.cubin
cuobjdump -lptx build/cuda_smoke   # expect only cuda_smoke.1.sm_75.ptx
```

Gazebo is available via `gz sim`; use the software-GL display for GUI runs, or headless for automation.

Build and test the core library (`src/`, C++ + CUDA kernels):

```bash
source "$UAV_RGB_ROOT/env.sh"
cmake -S "$UAV_RGB_ROOT/src" -B "$UAV_RGB_ROOT/src/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$UAV_RGB_ROOT/src/build"
ctest --test-dir "$UAV_RGB_ROOT/src/build" --output-on-failure   # 当前 13 项，以 ctest -N 为准
```

Train a model end-to-end (collect → preprocess → train):

```bash
# 并行采集（自动生成随机场景、按 GZ_PARTITION 隔离实例；纯 headless 相机会 segfault，必须 DISPLAY=:20）
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 "$UAV_RGB_ROOT/sim/tools/collect_parallel.py" \
  --instances 6 --episodes-per-instance 1 --steps 20 --hold 1.0 --base-seed 0 --out /tmp/dataset

# 打包成 UAVR 数据集（默认 160×120，按 scene_id 划分 split）
python3 "$UAV_RGB_ROOT/tools/preprocess.py" --out /tmp/ds /tmp/dataset/train/*/

# 训练（CNN → 4 logits；写 results.csv + metrics.jsonl，监控台可直接读）
"$UAV_RGB_ROOT/src/build/uavrgb_train" --data /tmp/ds/dataset.uavr --epochs 30 --batch 128 --lr 1e-3 --out /tmp/run1

python3 "$UAV_RGB_ROOT/sim/tools/teacher.py"          # 专家策略 22 项自测
python3 "$UAV_RGB_ROOT/sim/tools/scene_gen.py" --seed 42 --out /tmp/w.sdf   # 按 seed 生成场景
```

**split 已改为按几何组**（`sim/tools/split_manifest.py`，2026-10-29）：同几何永远同 split，跨 split 交集为 0；旧 `scene_id % 10` 策略在 seed 0..1999 上实测 241 组跨 split，已废弃。旧 FORWARD ≈89% 是旧场景统计，当前密场景的正式分布尚未验收。传感器停更/陈旧位姿处理、学生闭环评测仍缺。详见 `README.md` 和本轮审查记录。

**进程管理**：`pkill -f 'gz sim'` 会杀掉自己的 shell（命令行自匹配）。用 `setsid` + `kill -- -PGID`；残留 `ruby` 按 `comm` 找 PID。

Run the training-monitor UI (native Tkinter desktop app, reads `results.csv` / `metrics.jsonl` directly — no HTTP server, no browser):

```bash
python3 "$UAV_RGB_ROOT/ui/monitor.py" --run-dir "$UAV_RGB_ROOT/ui/demo"   # 原生窗口，5 s 自动刷新
DISPLAY=:20 python3 "$UAV_RGB_ROOT/ui/test_monitor.py"   # 解析/tail/绘图 + GUI 冒烟；无显示时跳过 GUI
```

`ui/launch_ui.sh`（及 XFCE 桌面的「UAV RGB 训练监控」快捷方式）启动该应用，并自动选择带 tkinter 的 Python（本机 `/usr/bin/python3` 已装 `python3-tk`，conda Python 也可用）。透传 `--run-dir`/`--refresh` 可指向真实 run-dir；界面内也可「选择目录」「暂停」。`metrics.jsonl` 按追加偏移增量读取，文件截断/替换时自动重载；图表用 Tkinter Canvas 手绘，不依赖 matplotlib，纯标准库。

同一窗口带本地训练控制：填运行目录 + `epochs/steps_per_epoch/batch_size/lr/seed` → 原子写 `<run_dir>/train_config.json` → 用 argv（不经 shell）启动 `<trainer> --config <run_dir>/train_config.json` → 实时看曲线、指标事件和 `train.log` → SIGINT→SIGTERM→SIGKILL 停止；运行目录必须是空目录/新目录，启动失败会回滚本次产物。默认命令是纯模拟的 `ui/trainer_demo.py`（DEMO 徽标，不代表真实训练）；原生 C++/CUDA trainer 仍需自己实现同一合同：读 `uavrgb.train.v1` 配置、写 `results.csv` + `metrics.jsonl`、SIGINT 退 130。自检：`python3 ui/test_trainer_demo.py`；GUI 侧 `DISPLAY=:20 python3 ui/test_monitor.py`（无显示自动跳过 GUI）。

## CUDA build rules for this machine

In every new CUDA CMake project, force the architecture **before** `project()`; otherwise CMake 3.28 silently uses nvcc's default (52):

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
project(... LANGUAGES CXX CUDA)
```

Raw nvcc uses `-gencode arch=compute_75,code=sm_75` (optionally add `code=compute_75`). Do not use `-arch=native`. Turing lacks sm_80+ features (TMA, `cp.async.bulk`, FP8, wgmma); BF16 tensor-core support must not be assumed. The current project forbids PyTorch.

## Layout and intended pipeline

Directory roles from the spec:

- `src/` — current C++/CUDA tensor, operators, CNN, trainer and tests; closed-loop policy integration remains pending
- `third_party/` — pinned third-party libraries only, not the training framework
- `config/` — data/model/simulation/UART protocol configs
- `data/`, `models/`, `logs/` — historical placeholders/demo; all new datasets, weights and runs belong under `$DATA_ROOT/uav_rgb/`, outside `code/`
- `experiments/` — each experiment is a self-contained CMake project (see `experiments/cuda_smoke/` as the template: source + `CMakeLists.txt` + local `build/`)
- `docs/` — design and acceptance records

Data splits must be by scene/map, never by random frames from the same scene. Episodes synchronize `rgb/timestamp/action_expert/action_executed/collision/episode_id/scene_id/seed`; simulation ground truth (position, obstacle geometry) must never leak into the student model's input.

The planned pipeline (spec §3) has six interfaces: `SimObservation` (RGB + timestamp + episode_id), `TeacherPlanner` (uses ground truth to label safe actions), `ActionAdapter` (quantizes expert motion into the four actions), `DatasetWriter`, `NativeTrainer` (CUDA CNN forward/backward, cross-entropy, Adam, checkpoints), and `SimulatorEval` (closed-loop: collision-free completion rate, collision rate, false-forward rate, inference p50/p95). Evaluation is closed-loop; offline action accuracy alone is not sufficient.

## 本轮维护入口

**报任何数字前先读 `docs/EVAL_STANDARD.md`**：离地用 scikit-learn 宏指标（`acc` 单独无效），
闭环用 AvoidBench(ICRA 2023) 的 SR/PO/EO/AGV/MP 定义。只报准确率、或声称跑过 AvoidBench 套件，都是违规。

推荐 `bash tools/check.sh`，默认新目录 `build/check`，避免使用搬迁前的 CMake cache。
`bash tools/setup_env.sh` 修复 venv 迁移路径并准备最小 numpy/Pillow 依赖。
`source env.sh` 后使用 `"$UAV_RGB_PYTHON"` 跑数据工具；Gazebo helper 使用系统 Python。
仍需遵守交接中的 UI/collect.py 并发维护边界，任务结束写 `docs/worklog/`。
