# uav_rgb：单目 RGB 四动作避障

本项目已实现自研 C++/CUDA CNN 训练器、Gazebo 专家采集和本地监控。
2026-10-29 本轮实测通过 **真实仿真 5 帧 → UAVR → GPU 训练 → checkpoint → 恢复训练**。
这只是链路冒烟：单场景、两个动作类别、没有独立验证集；尚未完成学生闭环避障验收。

## 环境与一键检查

在本目录执行：

```bash
source env.sh
bash tools/setup_env.sh              # 修复备份迁移的 venv 路径，安装最小数据依赖
"$UAV_RGB_PYTHON" tools/doctor.py --require-gpu --require-sim
bash tools/check.sh                  # 构建 + CTest + Python 自检
```

本次实际设备：Ubuntu 24.04、RTX 2080 Ti / sm_75、CUDA 12.8、Gazebo 8.15.0。
保留已有驱动、CUDA 与 Gazebo；不使用 PyTorch、TensorRT、cuDNN、cuBLAS、ROS 2。
`doctor` 检查命令与导入，`check` 才执行数值/训练测试；两者都不代替闭环验收。
无卡时可显式执行 `bash tools/check.sh --python-only`，它会打印跳过 C++/CUDA 的说明。

默认新构建目录为 `build/check`；可用 `UAV_RGB_BUILD_DIR` 覆盖。
最近一次全量验证（2026-10-29，全新构建目录）**18/18 CTest 通过**，编译产物仅 sm_75。
注意 `build/review-core` 是更早的 13 项构建，不要拿它的 `ctest -N` 当准。
`UAV_RGB_BUILD_JOBS` 默认 8。Python 工具用 `UAV_RGB_PYTHON`，Gazebo helper 用
`UAVRGB_GZ_PYTHON`（默认系统 `/usr/bin/python3`），避免混用 conda 与系统绑定。

## 数据与训练

所有新数据、权重和运行产物放在 `$DATA_ROOT/uav_rgb/`，默认
`DATA_ROOT=/root/rivermind-data`，不能放进 `code/`。短程采集示例：

```bash
source env.sh
# 默认 :20 的软件 GL 显示需已可用。
# 每次必须用新的输出目录：collect_parallel 和 uavrgb_train 都会拒绝复用非空目录
# （uavrgb_train 的 RunOwner 只认 --resume），所以这里按时间戳建一个。
RUN="$DATA_ROOT/uav_rgb/smoke/$(date +%Y%m%d-%H%M%S)"
"$UAV_RGB_PYTHON" sim/tools/collect_parallel.py \
  --instances 1 --episodes-per-instance 1 --steps 5 --hold 0.5 \
  --base-seed 42 --out "$RUN/collection"
"$UAV_RGB_PYTHON" tools/preprocess.py --drop-collision \
  --out "$RUN/preprocessed" "$RUN/collection/train/42"
./build/check/uavrgb_train \
  --data "$RUN/preprocessed/dataset.uavr" \
  --epochs 1 --batch 5 --lr 0.001 --class-weights auto --drop-collision \
  --out "$RUN/train"
DISPLAY=:20 /usr/bin/python3 ui/monitor.py --run-dir "$RUN/train"
```

只有 train 时会明确输出 `DEV_SPLIT_EMPTY`，JSONL 的评测 `split=train`。
`results.csv` 仍保留既有 `dev_*` 列名，因此必须结合事件判断评测数据来源。
正式实验应提供按实际场景隔离的 train/val/test，并覆盖全部四类动作。
UI 的默认启动命令是 DEMO；原生训练器目前用上面的 CLI 启动，UI 可以监视真实输出。

## 当前完成程度

| 模块 | 实现与证据 | 尚缺 |
| --- | --- | --- |
| 数值核心与 CNN | CPU/CUDA 算子、gradcheck、Adam、100,388 参数 CNN、18 项 CTest、best-checkpoint（dev 宏 F1） | 设备常驻优化、更完整异常边界 |
| 数据读写 | 320×240 → 160×120、NCHW、碰撞过滤、读端 scene_id/区间校验 | 几何去重、数据质量门禁 |
| 仿真采集 | 常驻订阅/发布、随机世界、独立 UUID 分区、真实 5 步 | 传感器停更故障处理、严格 sim-time 步进 |
| 真实数据训练 | 5 帧 GPU 训练和恢复成功，权重/指标/配置已保存 | 有规模的场景隔离训练与泛化验证 |
| 学生闭环评测 | 尚无实现 | 模型接入 SimulatorEval、成功率/碰撞率/时延 |
| 硬件部署 | 四动作边界已定义 | 荔枝派型号、NPU 合同、ESP32 协议与安全联调 |

正式采集的首要阻塞：2000 个 seed 只有 1124 种实际柱布局，241 组跨 split；
seed 4/train 与 1618/val 几何相同。scene_id 检查不能消除此类泄漏。
必须先按规范化几何分组划分或去重，再做正式数据集；本轮没有更改既有划分合同。

## 交接与证据

- [并行训练与正确性改造方案](docs/PARALLEL_TRAINING_PLAN.md)（拟实施，含模块分工与验收门禁）
- [本轮总审查与过程记录](docs/worklog/2026-10-29-1145-project-review.md)
- [核心修复与阅读清单](docs/worklog/2026-10-29-1205-core-review.md)
- [仿真实测与缺陷复现](docs/worklog/2026-10-29-1152-sim-review.md)
- [UI、设计合同与飞控审查](docs/worklog/2026-10-29-1205-integration-review.md)
- [历史交接及本轮更新](docs/HANDOFF.md)

本轮原始采集、stdout、环境 JSON、SHA256、训练和恢复产物位于
`/root/rivermind-data/uav_rgb/review-sim-20261029-115357/`。
