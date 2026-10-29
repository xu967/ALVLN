# TinyAerialVLN V7 — CODE_RULES（含 Gazebo 与 CF-Drone 仿真扩展）

> 继承 V6.1 的全部禁止事项和 code/ 预算；本文件是未来实施合并审查的工程契约。V7 尚未完成模型/Gazebo 物理闭环或固件改造。

## V7 新增不可违反的核心规则

1. 模型 `code/` 与物理仿真 `sim/` 独立；`code/src` 维持 ≤24 `.cc/.cpp`、`code/cuda` ≤6 `.cu`、公共模型头文件 ≤10。
2. 新增 `sim/src` ≤6 `.cpp`（WorldRunner/RgbBridge/MotorBridge/CFControlCore/TaskGenerator/ExpertPlanner）；`sim/include` ≤6 `.hpp`；单仿真业务文件 ≤300 有效行。不是必须创建所有文件。
3. 仿真生成的场景、图像帧、动态日志和 checkpoint 只能进入 `artifacts/`，严禁提交到 `code/` 或 Git；`sim/worlds` 只留少数版本化模板。
4. 高飞团队、CF-Drone、Gazebo/X3 模型都必须记录 source URL、commit、许可证、引用范围、数学复现范围；不复制完整第三方工程。CF-Drone 的附加商业条款与 GPLv3 兼容性需要另行核对。
5. 不允许 `SetPose` 位姿跳转代替物理动力学，除 episode 初始化/reset 和作为明确命名的 Kinematic baseline。
6. 所有电机顺序、FLU/FRD、ENU/NED、推力单位、旋向和相机光轴必须单测；CF 归一化 0..1 不可直接写入 `gz.msgs.Actuators.velocity`（rad/s）。
7. 学生在线输入仅 RGB+完整文本指令+动作历史+RNN 状态；Gazebo 位姿、IMU、深度、地图、碰撞仅可用于飞控内部、专家及评测器，不得泄漏给 `StudentPolicy`。
8. 在 CF 未实现安全定高与导航超时前，Gazebo 独立控制器的成功不能被描述成 CF-Drone 真实自主飞行成功。
9. 一键启动 `run.sh` 未实现某功能时必须非零退出并打印 `FEATURE_NOT_IMPLEMENTED`，不得制造假成绩或假图表。
10. 新增任何 .cpp/.cu 前先明确测试合同、输入输出、回滚方案；一次 PR 更改代码文件默认 ≤8、新增有效行 ≤500；例外必须 ADR。

## 每阶段新增权限

| Stage | 可以创建/修改 | 禁止提前创建 | Gate |
|---|---|---|---|
| GZ0 | `sim/worlds`, `sim/tests`, `tools/sim_preflight.py` | 学生模型、固件控制代码 | 官方 X3 motor→pose 通过 |
| GZ1 | `sim/models`, `sim/src/rgb_bridge.cpp` | 进度模型、蒸馏 | 相机挂在动态 link，frame+pose 同步 |
| GZ2 | `sim/src/world_runner.cpp`, scene adapter | CF 控制器强制迁移 | 静态森林动态 world+真实碰撞 |
| GZ3 | `sim/src/motor_bridge.cpp` | 实机电机直接控制 | 动作→电机→物理→RGB 闭环 |
| CF0/CF1 | `sim/src/cf_control_core.cpp` | 直接改真实 Arduino 飞控 | PID/mixer/坐标 golden + 稳定性 |
| DATA0 | `sim/src/task_generator.cpp`,`expert_planner.cpp` | PPO/大模型 ROI | 语言/目标/物理轨迹一致 |
| TRAIN | 依照 V6.1 分阶段解锁 `code/` | 广泛重写飞控和通用框架 | 数值 gradcheck/闭环评测/板端探针 |
| HIL | 独立版本的合法授权固件仓库、UART contract | 现场自主飞行 | 无桨台架、人工接管、安全接口 |

## V7 错误日志必备字段

`severity/code/module/run_id/scene_id/episode_id/sim_time_ns/monotonic_time_ns/message/hint/cause`。严重错误必须保存最近有限时间的状态缓冲。`CAMERA_NO_FRAMES`、`GZ_MOTOR_LAYOUT_MISMATCH`、`SIM_IMU_FRAME_ERROR`、`RESET_STATE_LEAK`、`UART_ACK_TIMEOUT` 等必须有独立故障注入测试。

## 工程源依赖图

```text
sim/worlds,models → Gazebo plugin/transport ↔ sim/src/WorldRunner+RgbBridge+MotorBridge
CF ControlCore SIL → MotorBridge → gz Actuators → Gazebo Physics → RGB
StudentPolicy code/model → PolicyABI → sim/EvalBridge (不直接依赖 Gazebo)
Train/DataReader ← Versioned EpisodeDataset ← Scene/Expert recorder
BoardRuntime → SG2002 SDK + NAV UART (不认识 Gazebo pose)
```

---

## V6.1 基础限制（继续有效）

## 1. 一句话纪律

`code/` **只实现 C++/CUDA 学生训练 + ONNX 导出 + 板端 C++ 推理 + 必要数值测试**。禁止把公开模拟器源码、数据集、预训练权重、几十个实验备份或 Python/PyTorch 学生训练器混入 `code/`。

## 2. 强制目录范围

```text
code/
├── include/tinyvln/       # ≤10 个对外头文件
├── src/
│   ├── core/              # TensorView, Arena, Seed, Checkpoint
│   ├── ops/               # Conv/DW/Linear, loss, GRU, optimizer
│   ├── model/             # Vision, Text, Fusion, Memory, Heads, ECPT
│   ├── data/              # EpisodeReader, tokenizer
│   ├── train/             # 唯一学生训练入口
│   ├── export/            # 固定模型 ONNX 导出
│   └── board/             # SG2002 运行时桥与 UART
├── cuda/                  # CUDA ops/dispatch，≤6 个 .cu
└── tests/                 # gradcheck, golden, protocol, parity
```

项目根目录其他合法目录：`config/` 两个 YAML，`adapters/` 轻量仿真桥（BabyAI、MiniWorld、可选 AirVLN），`tools/` 数据导入/校验，`docs/` 最多 5 份长期合同，`manifests/` 数据哈希，`artifacts/` 本地生成，`third_party/` 指向第三方源码，不复制其主体。

## 2.1 首版 24 个预批准源文件（名单外禁止直接新增）

```text
core:   tensor.cc runtime.cc checkpoint.cc                   (3)
ops:    linear.cc conv.cc depthwise.cc loss.cc gru.cc
        optimizer.cc attention.cc                             (7)
model:  visual.cc text.cc fusion.cc recurrent.cc ecpt.cc
        tinyvln.cc                                            (6)
data:   episode_reader.cc tokenizer.cc                         (2)
train:  train_main.cc eval_bridge.cc                           (2)
export: onnx_export.cc                                        (1)
board:  infer_runtime.cc uart_bridge.cc safety_fsm.cc          (3)
总计:  24 个计划中的 .cc 源文件（不要求一次创建）
```

CUDA 初始候选 4 个 `.cu`，公共头文件候选 10 个，详见 V6 研究方案 §7.1.1。允许删减或替换，新增文件必须通过 ADR 审查。不要因规划里有这些名字就提前生成空实现。

## 3. 数字限制

| 约束 | 默认硬限制 | 放宽条件 |
|---|---:|---|
| `code/src/` 内 `.cc/.cpp` 数量 | ≤24 | ADR + 移除/替换旧职责 |
| `code/cuda/` 内 `.cu` 数量 | ≤6 | CUDA probe 实测证明需要 |
| `code/include/tinyvln/` 公共头文件 | ≤10 | API 评审 |
| C++ 业务源文件有效行 | ≤280 | 拆分设计 + ADR |
| 单个 CUDA 文件有效行 | ≤360 | 特殊内核评审 |
| 单个 C++ 函数有效行 | ≤70 | 算法不可拆性说明 |
| PR 变更代码文件数 | ≤8 | ADR |
| PR 新增有效行 | ≤500 | ADR |
| 根目录实验 YAML | 2：`experiment.yaml`/`deployment.yaml` | 禁止任意新增不同版本配置 |
| 顶层可执行目标 | 3：`tinyvln_train` / `tinyvln_eval_bridge` / `tinyvln_board` | 证明必须新增才审议 |

**有效行**：排除空白和纯注释，实际应由后续 CI 工具统计并保留报表。阈值是人为项目管理目标，不能代替代码正确性检查。

## 4. 严禁事项

- 禁止 `model_v2.py`, `train_final.py`, `model_best.cpp`, `backup/`, `temp/`, `copy/`, `new_model_3/` 等平行实现。
- 禁止学生训练主路径加载 `torch`, `torchvision`, `transformers`；它们仅可在隔离 `oracle_tests/` 或冻结教师离线工具中出现，且不得向训练进程提供未声明的在线信息。
- 禁止整套 MiniWorld/AirVLN/Unity/ROS 源码进入 `code/`；仅保留 adapter，第三方组件固定许可证/版本。
- 禁止 `data/`、视频、帧缓存、`.pt/.pth/.onnx/.cvimodel`、大模型权重、GPU build 目录直接放在 `code/` 或提交进 Git。
- 禁止任意平台 action ID 复用：BabyAI、MiniWorld、AerialVLN-S、ESP32 各有独立协议；转换需显式测试。
- 禁止训练/运行时从 GT 位姿、深度、目标坐标、参考路径读取学生在线输入。
- 禁止硬编码串口设备、个人数据路径、训练集根目录、显卡型号、板卡 PCB 版本。
- 禁止在没有同容量/同数据/同观察合同对照时添加“创新点”模块。
- 禁止旧框架大重构：只能修有证据的数值/动作合同错误；不为代码好看重写已经正确的部分。

## 5. 模块依赖准则

```text
core <- ops <- model <- train
              ^         ^
              |         |
            data -------+
adapters -> versioned EpisodeSchema -> data
board -> exported-model schema + platform SDK
```

`core` 不认识 model；`model` 不认识 MiniWorld/AirVLN；`train` 不调用板端 UART；`board` 不连接 GT Teacher；`adapters` 不修改模型；`docs` 只描述真实代码合同。

## 6. 实现次序（按通过门禁解锁目录）

| 里程碑 | 本轮允许更改的模块 | 本轮不允许更改的模块 | 验收 |
|---|---|---|---|
| G0/G1 数据 smoke | adapters、src/data、manifest | CUDA、ECPT、board policy | 30 条 D1 RGB+指令+动作可复现 |
| G2 板卡 Conv probe | src/export、src/board 最小图 | 完整模型、KD、ROI | 真实 SG2002 编译与运行 |
| G3 手写训练 | core、ops、cuda、model 最小集、train | 复杂 ECPT、ROI、PPO | 64 episode overfit + gradient parity |
| G4 基线闭环 | model、train、eval bridge | KD/ROI/全面重构 | 语言 shuffle 实验、SR/STOP 日志 |
| G5 主方法 | model/ECPT、相应 tests | 无关硬件优化 | 等容量、等种子、配对消融 |
| G6 蒸馏（可选） | train/loss、离线 teacher tools | 板端新增大教师 | T1 先证明有真实闭环优势 |
| G7 部署 | export、board、tests | 修改官方评测合同 | BF16/INT8 parity 和 P95 报告 |

## 7. 每次 PR 必须随附的证据

**研究相关**：这个改动解决哪个失败案例或假设？对照是什么？训练种子/场景 split 是否不变？新增参数/时延/内存成本是多少？

**数学相关**：数值 golden、FP64 有限差分梯度、CPU/CUDA forward 与 backward、mask/reset，以及至少一种故意注入错误的阴性对照。

**训练相关**：16–64 episode overfit、恢复训练一致性、NaN/Inf 检测、训练数据 hash、文本干预与 STOP 诊断。

**部署相关**：静态 ONNX 输入输出形状、cvimodel 编译记录、SG2002 实际运行，FP32/INT8 logits、STOP、episode 决策一致性，camera→UART P50/P95/P99，峰值内存与温度。

**安全相关**：UART CRC/seq/TTL、ACK 与 EXEC_DONE 区分、ESP32 超时降级与人工接管优先，不得擅自声称停机就能悬停。

未完成测试可以标记 BLOCKED，但**不得把未测试写成 PASS**。

## 8. 团队/Agent 并行开发的锁区边界（尚未启动任何 Agent）

| 角色 | 独占区域 | 交付合同 |
|---|---|---|
| Numerical | `code/src/core, code/src/ops, code/cuda` | kernel、gradient、数值对照 |
| Data/Sim | `adapters, code/src/data, manifests` | episode schema、action、split、replay |
| Model/Train | `code/src/model, code/src/train` | 可训练学生、loss、消融 |
| Edge | `code/src/export, code/src/board` | cvimodel、板端 runtime、UART |

共享头文件每次变更需 ADR 和版本号；并行任务不在同一个 `.cpp` 文件上同时修改。每位开发者/Agent 默认 **最多一个未合并功能分支**。

## 9. `code/` 的 Definition of Done

代码量没有增长也可以完成一个高质量阶段。只有在“真实数据合同 → 数值正确 → 实验可复现 → 资源测量 → 设备风险可控”的证据链完整时，才准增加下一层模块。**任何方案修改优先删除无收益分支，不靠不断追加代码弥补训练失败。**


## 10. V6.1 新增的一键入口、场景与监控规则

- 根目录只允许唯一 `run.sh`；不能自动 sudo、自动下载大型数据、自动运行实体飞控。
- `tools/` 初始白名单：`launcher.py`、`scene_tool.py`、`dashboard.py`；总量 ≤4，超出须 ADR。必须使用 Python 标准库，不引入新的后台依赖。
- `assets/` 首版只保留 `dashboard.html`，使用原生 Canvas 读取真实 JSONL；不能构造假训练指标。
- `code/include/tinyvln/training_monitor.hpp` 是可复用真实训练进度/日志部件，计入公共头文件 ≤10 的预算。
- 生成的 SDF、manifest、运行日志、权重都位于 `artifacts/`；`code/` 不得复制场景素材或模型权重。
- 数据层必须隔离 `SceneSpec`（可含真值）和 `EpisodeObservation`（仅学生合法观察）。不能将仿真真值泄漏到训练输入。
- 错误统一为机器可读 code + 用户可读 message/hint + 非零退出码；`FATAL` 不得重试掩盖。
- 看板仅绑定 `127.0.0.1`，只读训练指标；没有指标时显示空状态。
- `scene_tool` 只能声称“静态障碍物场景生成”，未经合法 RGB 观测、物理飞行器、碰撞和指令一致性验收，不可声称“已经完成无人机训练仿真”。
- 高飞团队的源码只在 `third_party` 外部版本库/隔离容器中引用；借鉴代码记录来源、commit、许可证与对应数学/几何测试。
- 在项目计划中清楚区分：可运行的基础设施、仅有接口的占位、尚未开始的模型训练与真实飞行。

---

## V7.1 增补条款（2026-10-28）：ESP32 冻结 + UI 与证据完整性

**优先级：以下条款覆盖本文件中与 V7.1 UI/ESP32 相关的旧表述。**

1. **CF-Drone 原固件可以在未来版本修改，但本轮 V7.1 禁止修改任何 `.ino`、飞控 PID 参数、电机混控、ESP32 引脚定义、刷机配置和真实 UART 控制固件。** 原归档是审阅资料，不嵌入本 ZIP、不提交更改；单独的 CF-FW 变更需源码授权、版本追踪、独立备份、控制契约和无桨台架验收。
2. 现有 `assets/dashboard.html` 仍为唯一 UI 源文件，`tools/dashboard.py` 仍为唯一 HTTP 服务。禁止添加 React/Vue/Node 等新构建栈；允许业务上确有需要时经 ADR 将 HTML 拆为静态 CSS/JS，但不得多套 UI 并行。
3. 研究看板只读且只绑定 `127.0.0.1`；无 POST、无网页启动飞控、无远程串口、无训练 shell 执行；未来 UI 控制必须专门安全设计。
4. 状态名称不得用 `ready` 直接代表功能通过，必须给出证据等级；静态 SDF、Gazebo CLI、训练器二进制文件各自都不是 GZ/CF/TRAIN 通过证明。
5. 前端仅显示真实指标：`metrics.jsonl` 为空不能构造占位数；模拟示例只允许出现在标记为示例的文档中。
6. 训练图、进度条、STOP、SR 都必须和真实事件 schema 一致；损坏日志、部分写入、非法 run ID 必须有告警/错误码。
7. `README.md` 与 `docs/V7_1_UI_SYSTEM_DESIGN.md` 必须说明实验事实、已做测试、未做实验。外部 AI 审核包必须包含按状态可复现的测试命令和未完成清单。
8. UI 端不能被当作科学结论证据。论文方法创新和闭环 SR 必须靠真实 episode 记录、固定 split/seed、源代码与运行环境复核。
9. V7 原有 `code/src` 24 `.cpp`、`code/cuda` 6 `.cu`、`sim/src` 6 `.cpp` 的上限原封不动；本轮新增 UI、测试与文档不占模型/仿真 cpp 额度。
10. 本次工具修改仅 `tools/dashboard.py` / `tools/launcher.py`；本次 UI 修改仅 `assets/dashboard.html`；新增测试仅 `code/tests/test_dashboard_v7_1.py`。不以 UI 改动为名铺设全新模块。
