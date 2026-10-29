# UI、设计合同与飞控边界复核

- 时间：2026-10-29 12:05 CST
- 范围：uav_rgb/ui、非仿真历史记录、TinyAerialVLN 设计合同及 frontend 文本、CF-Drone 全部可见源码及说明文本
- 结论：完成本分工的 UI 审查和验证，12:27 CST 补齐 CF 外围与 Tiny 前端阅读；整个目录的第三方源码审计未完成，不能宣称整个工作区全文审核。
- 规范：已读取 `/root/claude-code-backup-20261028/claude-skills/skills/personal-skill/SKILL.md`。

## 做了什么

完整阅读了 `ui/monitor.py`（994 行）、`trainer_demo.py`（233 行）、`test_monitor.py`（446 行）、`test_trainer_demo.py`（220 行）、`launch_ui.sh`，以及 `ui/demo/results.csv`、`metrics.jsonl`。逐项对照 `docs/监控系统交接.md`，并读取当前 `src/train_main.cpp` 验证命令契约。

完整阅读的 uav_rgb 文档包括 `docs/HANDOFF.md`、`监控系统交接.md`、`results.md`、`环境说明.md`、`安装核验.md`、`cuda_smoke.md`，以及以下历史 worklog（文件均位于 `docs/worklog/`）：

- `2026-10-28-1923-uav-rgb-env-verify.md`
- `2026-10-28-2019-code-review-fixes.md`
- `2026-10-28-2040-ui-desktop-shortcut.md`
- `2026-10-28-2314-e4-cuda-conv.md`
- `2026-10-28-2325-e5-numeric-kernels.md`
- `2026-10-28-2326-preprocess-eventschema.md`
- `2026-10-28-2328-resolution-decision.md`
- `2026-10-28-2330-monitor-local-optimize.md`
- `2026-10-28-2334-audit-cpp.md`
- `2026-10-28-2334-audit-cuda.md`
- `2026-10-28-2337-gpu-bench.md`
- `2026-10-28-2343-fix-audit-mustfix.md`
- `2026-10-29-0038-ref-engineering-study.md`
- `2026-10-29-0051-cnn-trainer.md`
- `2026-10-29-1000-class-weights-collision-filter.md`

阅读 `TinyAerialVLN_Final_V8/START_HERE.md` 和 `docs/00` 至 `14`、`TEST_REPORT_LOCAL.md`，用于辨别长期 VLN 设计与当前四动作 RGB 避障实现。阅读 `idea/INDEX.md`，没有把其对上游的摘要当作本轮完整源码审计。

首轮完整阅读 CF-Drone 关键文件：`CF-Drone.ino`、`board_config.h`、`control.h`、`control.ino`、`estimate.ino`、`safety.ino`、`mavlink.ino`、`motors.ino`、`pid.h`、`lpf.h`、`imu.ino`、`battery.ino`、`time.ino`。其余文件已在下方补读节列明。未修改飞控。

## 发现与当前程度

| 项 | 证据 | 结论 |
| --- | --- | --- |
| 真实 trainer 启动按钮不兼容 | `ui/monitor.py:293` 固定追加 `--config`，`src/train_main.cpp` 只接受 CLI 字段；本轮实测退出 2 | P1；监控可读真实指标，不能直接通过按钮启动现有二进制 |
| GUI 配置没有数据集字段 | `ui/monitor.py:479` 仅 epochs/steps_per_epoch/batch_size/lr/seed | 仅给 C++ 加 `--config` 仍不足以完成真实训练集成；需要明确传入 UAVR 数据集 |
| 训练输出默认仍进源码树 | `ui/monitor.py:597` 返回 `REPO_ROOT/logs/runs/...-demo` | 与根 README 的 DATA_ROOT 约定不一致；未来接真实训练前应对齐 |
| 交接首页过期 | `docs/监控系统交接.md` 首页称“不启动训练”，后部已描述训练控制；`trainer_demo.py` 顶部仍称真实 trainer 未实现 | 当前已有进程控制和 stdout 面板；只有默认 trainer 是模拟，真实 C++ trainer 已存在 |
| 日志上限不等于运行中轮转 | `prepare_train_log` 只在启动前调用，且启动要求空目录 | 8 MiB 是启动前轮转条件；长时间运行的 train.log 仍可持续增长 |
| 设计任务不等于已实现任务 | V8 docs/04 和 docs/05 设计语言编码、GRU、STOP、ECPT；uav_rgb 当前合同是 RGB→四动作 | 当前可称 RGB 避障训练原型，不能称完整 VLN/ECPT 研究完成 |
| CF 定高未实现 | `control.ino:43` 明写 stub，模式 3 无控制分支；Web 按钮回退 STAB | 四动作不能直接映射为经验证的实机速度/悬停能力 |
| 控制入口未统一仲裁 | `mavlink.ino:192,206,241` 可写 armed，未复用 SBUS/Web 的 IMU/低压检查；`:245` 接受模式 3 | `source_observed`；需要独立固件/HIL 工作，不应接通自主输出 |
| 自动 setpoint 的失联检测不完整 | `mavlink.ino` 的 SET_ATTITUDE/ACTUATOR 不刷新 controlTime，`safety.ino:50` 对 controlTime==0 直接返回 | `source_observed`；没有独立导航 TTL/ACK 协议，不能把收到命令当作执行完成 |
| 默认姿态估计主要依赖陀螺积分 | `estimate.ino` 的 acc 修正只在 landed，levelWeight/levelBiasGain 默认 0 | 没有据此验证长期飞行姿态稳定性 |

上表中的飞控行为是源码审查结论，没有执行实体攻击、串口写入、刷机或电机操作。`mavlink.ino:169` 还使用消息 `m.count` 作 `strlcpy` 的目标大小，未在该处限到本地数组大小；这需要后续独立固件审计及边界测试。

## 验证到哪一步

所有命令在 code 根目录运行。

| 命令 | 本轮真实输出 |
| --- | --- |
| `nvidia-smi -L` | `GPU 0: NVIDIA GeForce RTX 2080 Ti`，当前为 GPU 模式 |
| `env -u DISPLAY PYTHONDONTWRITEBYTECODE=1 python3 uav_rgb/ui/test_monitor.py` | `test_monitor: 56 项通过，1 项跳过`，exit 0；跳过的是 GUI |
| `PYTHONDONTWRITEBYTECODE=1 python3 uav_rgb/ui/test_trainer_demo.py` | `test_trainer_demo: 89 项通过`，exit 0；含真实 demo 子进程和 SIGINT 测试 |
| 用 `monitor.build_train_argv` 构造已有 `src/build/uavrgb_train --config /tmp/nonexistent-uavrgb-review-config.json` 并执行 | exit 2，`未知参数: --config`，随后打印 CLI 用法；在打开配置之前已拒绝 |

本轮没有用旧 LastTest.log 代替测试结果。UI 测试中的 loss 全部是标明 DEMO 的模拟值，不是模型训练证据。

## 没做什么

- 没有修改 `ui/`：主任务分工要求只读，交接说明另一个会话维护；本轮只输出本文。
- 本分工没有重跑 GUI：本分工测试显式移除 DISPLAY；主任务随后本轮实测 GUI 85 项通过，详见下方补读节。没有把历史通过当作本轮通过。
- 没有重跑核心 CUDA、Gazebo 采集、真实训练或闭环评测：由同组其他分工核验，本文不替其下结论。
- 没有重复通读 sim 源码及仿真相关 worklog，包括 `ref-drone-stack-study`、`ref-planner-study`、`e1-e4-parallel`、`audit-python-sim`：已明确分配给 sim 审查分工。
- 没有编译 CF 固件或进行硬件验证：下方补读已覆盖外围文本，但阅读不能替代固件编译、台架/HIL 和实飞验证。
- 没有逐个阅读 idea 上游 README、第三方仓库全部源码、论文 PDF、ZIP 内容或缓存/构建产物：本轮只读索引和既有研究记录，主任务的 uav_rgb 优先；第三方约 1 GB 不能据目录枚举声称通读。
- 没有修改全局环境、安装依赖、处理旧残留进程、公开发布、连接实体硬件：本分工无需这些操作。

## 待确认

- 实体荔枝派型号/PCB/系统、ESP32 型号/PCB/UART 接线与飞控实际能力：仓库文档未提供可独立验证的实机记录。
- 后续真实训练 UI 需约定数据集输入及 steps_per_epoch 的含义；不能默默把模拟配置字段当原生 trainer 已支持。

## 风险与遗留

下一阶段优先真实数据链路和 SimulatorEval 的可重放指标。UI 测试通过只能证明控制与显示能力，合成过拟合也只能证明数值训练链路；二者都不足以证明避障泛化、VLN 语言能力、NPU 部署或实体自主飞行。

## 12:27 CST 补读完成与同组实测更新

为覆盖用户要求的现有主项目内容，逐段补读了 CF-Drone 的 `rc.ino`、`parameters.ino`、`cli.ino`、`web_rc.ino`、`web_rc_html.h`（666 行）、`wifi.ino`、`log.ino`、`led.ino`、`util.h`、`vector.h`、`quaternion.h`、`drivers.cpp`，以及 `drivers/IMU.h`、`invensense_imu.h`、`invensense_imu.cpp`、`logger.h`、`MPU9250.h`、`MPU9250.cpp`、`SBUS.h`、`SBUS.cpp`。同时完整读取 `README.MD` 和 `LICENSE`（770 行）。这些加上首轮关键文件，覆盖 `rg --files CF-Drone` 列出的全部 35 个文本文件；不等于逐配置编译或全量安全认证。

完整读取 TinyAerialVLN_Final_V8 的 `frontend/app.js`（117 行，含长行）、`index.html`（69 行）、`styles.css`（28 行），及 `README.md`、`API_LOGGING_CONTRACT.md`、`SIMULATION_ORCHESTRATION.md`。七张 `preview_*.png` 属历史预览图片，未作为当前运行证据审阅；Tiny 的 server/tools/tests/cpp/scripts/config 已交由 sim 分工补读，没有重复归属本分工。

补读后的主要源码观察：

- CF `rc.ino:161` 收到 SBUS 帧后使用通道并刷新控制时间，没有检查驱动已解析的 failsafe/lost_frame 标志；`rc.ino:178` 起的通道索引只检查非负，没有限制小于 16。参数设置缺乏对应通道范围约束。
- CF Web 控制 JSON 缺少必填字段校验，缺失 throttle 的默认原始值 0 会映射到约 50% 油门；时间戳没有形成导航命令新鲜度校验。Web console 可进入 `doCommand`，CLI arm 与校准/电机测试路径没有统一复用所有解锁保护。
- `drivers/IMU.h:130` 起 `waitForData` 在中断分支不检查 `read()` 成功与否；轮询分支没有超时。默认 MPU 配置用定时器模拟采样中断，不能等同于已验证的硬件 DRDY 链路。
- Tiny 前端连接、SSE、单步和仿真提交已有代码，且新增 `autoConnectLocal`；三份前端文档仍描述“后端未实现”，存在版本说明滞后。不能据这些旧合同否定当前 V8 的 server 实现；实际能力应以同组 sim 分工对当前 server 的核对为准。
- Tiny 的 infer/eval checkpoint 下拉框在当前 app.js 没有填充流程；仿真 capability 填充独立下拉框。断开 API 没有关停既有 EventSource/仿真轮询；这些是后续集成问题，本轮按只读分工记录，没有扩修原型。

上述补充均为 `source_observed`，没有发出硬件命令。CF 许可文件包含附加条款，本分工没有复制其实现进 uav_rgb。

主任务在本轮另行实际执行 `DISPLAY=:20 .venv-tools/bin/python ui/test_monitor.py`，报告 **85 项通过**；这里明确标作同组当前实测，而非本分工无 DISPLAY 的 56 通过/1 跳过，也不是历史测试记录。此次补读只修改本文，没有修改 UI、Tiny 前端或 CF 固件源码。
