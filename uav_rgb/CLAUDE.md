# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project state

`uav_rgb` 已实现单目 RGB 四动作 CNN、CPU/CUDA 数值算子、Adam、UAVR 数据读取、checkpoint/resume（含按 dev 宏 F1 保存的 `ckpt_best.bin`）、eval-only 复评、Gazebo 专家采集、数据验收闸门（11 条不变量，含标签复算）、常驻推理进程与学生闭环评测。**2026-10-30：T1–T7 全部完成**——T5 采集 599 集 / 23960 步（audit 11/11 PASS）；T6 best 权重 test `acc 0.9186 / macro_f1 0.8589 / bal_acc 0.8497`，四类 recall `[1.000, 0.952, 0.700, 0.747]`；T7 旧版（observation 语义）`SR 0.90`；v1 当前环境复测（observation）`SR 0.90`（18/20）；T7-R（dispatch 语义）`SR 0.55`（11/20），A/B 证明 R01 窗口语义是主因（seed 49/57 observation 2/2 vs dispatch 0/2）。v1 与 dispatch 数字禁止混报，以 `docs/HANDOFF.md §4.1` 为准。GRU/语言导航、板端部署尚未实现。**接手先读 [`docs/HANDOFF.md`](docs/HANDOFF.md)**，任务与验收以 [`docs/TASKS.md`](docs/TASKS.md) 为准，实验数字见 `docs/experiments/`。设计规格仍位于 `../ALVLN/references/reviews/UBUNTU_BASE_GUIDE_V1.md`。文档与代码注释保持中文。

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
ctest --test-dir "$UAV_RGB_ROOT/src/build" --output-on-failure   # 当前 20 项，以 ctest -N 为准
```

Train a model end-to-end (collect → preprocess → train):

```bash
# 并行采集（自动生成随机场景、按 GZ_PARTITION 隔离实例；纯 headless 相机会 segfault，必须 DISPLAY=:20）
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg python3 "$UAV_RGB_ROOT/sim/tools/collect_parallel.py" \
  --instances 4 --episodes-per-instance 1 --steps 40 --hold 1.0 --base-seed 0 --out /tmp/dataset

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

同一窗口带本地训练控制：填运行目录 + `epochs/steps_per_epoch/batch_size/lr/seed` → 原子写 `<run_dir>.monitor/train_config.json` → 用 argv（不经 shell）启动 `<trainer> --config <run_dir>.monitor/train_config.json` → 实时看曲线、指标事件和 `train.log` → SIGINT→SIGTERM→SIGKILL 停止；运行目录必须是空目录/新目录，启动失败会回滚本次产物。控制台自己的文件（配置/命令/日志）都在 `<run_dir>.monitor/`，run 目录只归 trainer（`RunOwner` 要求空目录）。默认命令是纯模拟的 `ui/trainer_demo.py`（DEMO 徽标，不代表真实训练）；跑真实训练把命令改成 `python3 ui/trainer_native.py --data <dataset.uavr>`——它把 `uavrgb.train.v1` 转成 `uavrgb_train --data/--epochs/--batch/--lr/--seed/--out` 并 `execv`，额外参数（`--resume` 等）原样透传；resume 目前只能命令行跑（界面强制空目录），跑完可用「选择目录」看曲线。自检：`python3 ui/test_trainer_demo.py`；GUI 侧 `DISPLAY=:20 python3 ui/test_monitor.py`（无显示自动跳过 GUI）。

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

**Datasets live in `/root/rivermind-data/uav_rgb/` (`$DATA_ROOT/uav_rgb`)** — every collected episode, packaged `.uavr`, checkpoint, run artifact, audit/eval JSON. Never write datasets into this repository (`code/`); the in-repo `data/`, `models/`, `logs/` are legacy placeholders. This is enforced as a hard constraint in `docs/ENGINEERING_RULES.md §二.6`.

Data splits must be by scene/map, never by random frames from the same scene. Episodes synchronize `rgb/timestamp/action_expert/action_executed/collision/episode_id/scene_id/seed`; simulation ground truth (position, obstacle geometry) must never leak into the student model's input.

The planned pipeline (spec §3) has six interfaces: `SimObservation` (RGB + timestamp + episode_id), `TeacherPlanner` (uses ground truth to label safe actions), `ActionAdapter` (quantizes expert motion into the four actions), `DatasetWriter`, `NativeTrainer` (CUDA CNN forward/backward, cross-entropy, Adam, checkpoints), and `SimulatorEval` (closed-loop: collision-free completion rate, collision rate, false-forward rate, inference p50/p95). Evaluation is closed-loop; offline action accuracy alone is not sufficient.

## 本轮维护入口

**动手前先读 [`docs/ENGINEERING_RULES.md`](docs/ENGINEERING_RULES.md)**：它是执行细则
（动手前声明范围 → 最小改动 → 全量验证 → 留痕 → git → 清理 → 汇报），
并列出「已知未修项」，避免重复发现。本文档管环境与边界，`EVAL_STANDARD.md` 管口径。

**Git**：本地仓库在 `.git`；远端是 `git@github.com:xu967/ALVLN.git` 的 `uav_rgb/` 子目录。
推送在 `../ALVLN`（完整克隆）里做：替换 `uav_rgb/` → commit → push。重新克隆要 12 分钟，别删。
本仓库有并发会话，提交前逐文件确认归属，不要 `git add -A`。

**报任何数字前先读 `docs/EVAL_STANDARD.md`**：离地用 scikit-learn 宏指标（`acc` 单独无效），
闭环用 AvoidBench(ICRA 2023) 的 SR/PO/EO/AGV/MP 定义。只报准确率、或声称跑过 AvoidBench 套件，都是违规。

推荐 `bash tools/check.sh`，默认新目录 `build/check`，避免使用搬迁前的 CMake cache。
`bash tools/setup_env.sh` 修复 venv 迁移路径并准备最小 numpy/Pillow 依赖。
`source env.sh` 后使用 `"$UAV_RGB_PYTHON"` 跑数据工具；Gazebo helper 使用系统 Python。
仍需遵守交接中的 UI/collect.py 并发维护边界，任务结束写 `docs/worklog/`。
