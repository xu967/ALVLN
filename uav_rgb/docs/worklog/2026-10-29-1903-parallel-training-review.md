# 并行训练首批实现审核

- 时间：2026-10-29 19:03，Asia/Shanghai。
- 范围：`PARALLEL_TRAINING_IMPLEMENTATION.md`、`2026-10-29-1734-parallel-training-first-batch.md` 及其对应源码、测试、验收证据。
- 结论：审核发现的两项P2已按用户后续授权修复，新增回归验证修复前失败、修复后通过；完整17项CTest及两项设备memcheck通过。完整并行训练和GPU加速仍待后续实现及验收。

## 做了什么

- 按 personal-skill、ponytail、ponytail-review 规范审核。主审核负责独立构建及 GPU 验证，三个 agent 分别只读核对设备状态、reader/run 锁、文档与证据。
- 阅读首批新增源文件、公共头、测试、CMake，以及关联的 host 模型、数据集和 checkpoint 合同；核对方案第38节。
- 核对归档中的68项源码、24项交付物、4个候选二进制哈希，均与当前文件一致。原始16项CTest及memcheck日志与原实验记录吻合。
- 在独立 `build/resident-review` 目录完成 Release 构建，重新执行三个新增测试。
- 实证 reader 的符号链接路径错误，以及状态快照异常排空时遗漏 FAILED 标记的问题。复现源码、程序和数据均在临时目录中使用，结束自动清理。
- 结果与复现方法单独记录在 [审核实验记录](../experiments/2026-10-29-parallel-training-review.md)；配置/构建原始日志与审核源码哈希放在 `/root/rivermind-data/uav_rgb/parallel_training/20261029-1903-review/`。

## 审核发现

### 1. P2：BatchReader 对合法路径做词法折叠，可能静默读取另一份数据集

- 位置：`src/batch_reader.cpp:58`。
- 触发：`base/link -> base/real/inner`，输入 `base/link/../data.uavr`。Linux 与旧 `load_uavr` 指向 `base/real/data.uavr`，新 reader 的 `lexically_normal()` 却得到 `base/data.uavr`。
- 实证：两份合法文件分别有3条和2条样本，旧 loader 返回3，新 reader 返回2；没有报错。文件身份校验始终检查错误路径，无法发现此问题。
- 最小修复：使用 `std::filesystem::absolute(path).string()`，去掉 `lexically_normal()`，保留 Linux 的符号链接/父目录解析顺序。若改用 canonical，需另行明确运行中原始符号链接改指向的检测合同。
- 回归：同一符号链接加 `..` 路径分别交给旧 loader 与新 reader，核对元信息、标签、split 和像素一致。
- 不修复的影响：采用合法目录别名的数据任务可能使用错误数据集，训练结果与用户选择的数据不符。

### 2. P2：异常排空忽略 CUDA 失败，Context 仍允许继续提交

- 位置：`src/device/state.cpp:17`，特别是18、19行。
- 触发：快照已提交 params D2H，接着因 m 容量不足产生 host 异常；catch 内同步排空再次遇到 CUDA 错误。
- 实证：对收尾 `cudaStreamSynchronize` 返回值注入 `cudaErrorLaunchFailure`，得到 `context_failed=0 activate_accepted=1`。已有测试只覆盖排空成功的容量异常，没有覆盖两种错误同时出现。
- 最小修复：保存首次异常；检查收尾 `cudaSetDevice`、`cudaStreamSynchronize` 的结果，发生 CUDA 错误时经 Context 的错误入口标记 FAILED，并保留首次错误及收尾错误诊断。最后重抛首次异常；不能为保留首次异常而丢失失效状态。纯 host 错误且排空成功时仍保持 Context 可用。
- 回归：扩展已有中途失败用例，断言注入确实命中、首次容量异常保留、Context FAILED、后续 activate/finish 被拒绝；同时保留当前排空成功的对照用例。
- 不修复的影响：调用方只能看到可恢复的 host 参数错误，错误地继续使用实际已经发生 CUDA 失败的上下文，违反实现记录中的生命周期合同。

## 没做什么

- 未修生产代码或改写原始验收记录：用户请求的是审核，保留审阅对象和历史证据。
- 未重跑原有完整16项CTest和memcheck：已核对其日志与当前源码/二进制哈希，本轮独立重跑与新增模块相关的3项测试，并做针对性复现；历史通过和本轮复跑分开记录。
- 未制造真实非法kernel、测试跨卡/NFS或执行训练性能基准：本批没有设备训练计算链，CUDA错误使用返回值注入；这些验证不属于本次基础模块审核结论。
- 未接入 Trainer、修改 checkpoint、增加双槽流水线：属于记录明确披露的后续工单，本轮不把尚未交付功能列为新增bug。
- 未修改UI、仿真与数据采集代码，未提交Git：与本次审核无关。

## 待确认

- 无需用户补充信息即可修复上述两项；本轮仅交付审核，尚未执行修复。

## 验证到哪一步

| 项目 | 结果 |
| --- | --- |
| 硬件检测 | GPU模式，实际列出RTX2080Ti |
| 独立Release配置/构建 | 通过 |
| 新增3项CTest | 通过；不代表未覆盖分支正确 |
| symlink/.. CPU复现 | 成功触发新旧loader读取不同文件 |
| CUDA异常排空故障注入 | 成功触发错误后未标记FAILED |
| 原始验收证据与当前文件 | 哈希和原始日志核对通过 |

## 风险与遗留

- 修复上述两项并加入能复现问题的回归后，再继续设备计算链。当前普通路径测试通过，不能替代异常路径和合法路径别名的验证。
- 当前交付仍是基础设施：实际Trainer没有使用DeviceTrainState、BatchReader或RunOwner；GPU利用率改善尚无本批实证。

## 更新 2026-10-29 19:38：修复两项审核问题

用户在审核后明确要求修改。本节记录后续修复；上文“未修改生产代码”等内容是19:03审核阶段的历史范围。

### 做了什么

- `src/batch_reader.cpp`：移除 `lexically_normal()`，使用absolute路径保留Linux对符号链接后接父目录的解析顺序，同时保留后续stat对原始链接改指向的检测。
- `src/tests/test_batch_reader.cpp`：添加不同合法文件的符号链接回归，比对元信息、标签、train/val/test下标、像素，并验证链接改指向被拒绝。
- `src/device/state.cpp`：增加局部收尾检查，两个CUDA收尾调用均通过Context检查并在失败时标记FAILED、写stderr诊断；即使set失败仍尝试同步，最终重抛首次异常，保持其类型和信息。
- `src/tests/test_device_transfer_errors.cpp`：新增独立异常路径测试，避免原测试文件超过行数限制；覆盖set失败、sync失败、两者同时失败、host异常但收尾成功。测试仅注入返回值，真实调用先执行完毕。
- `src/CMakeLists.txt`：注册新测试及两项GNU ld wrap，仅测试程序使用故障注入。
- 并行分工：agent独立修复reader并做CPU红绿验证，主agent修复设备收尾并串行验证GPU；另一agent只读交叉审核设备改动通过。两条新增回归均已证明旧代码会失败。
- 修复涉及的C++文件均未超过280行；核对历史源码hash，本次仅修改上述四份已有源/构建文件及新增一份测试，其他已有源码未变。

### 没做什么

- 未实现resident训练、双槽或性能优化：本次是两项已确认错误的定点修复，GPU利用率仍需后续训练工单验证。
- 未修改其他会话的UI/仿真/数据代码，未修改旧checkpoint或RunOwner，未创建Git仓库及提交：不属于本次修复。
- 未验证真实CUDA非法访存/驱动损坏后的资源恢复：注入测试验证错误处理，不模拟损坏驱动；不据此承诺故障恢复。
- 未覆盖多GPU、NFS与stat无法识别的原地数据篡改：这些既有限制未改变。
- 未覆盖、重写历史实验日志或历史哈希：修复版本的源码hash和日志单独保存，原首批验收证据不再代表修复后的源码。

### 验证到哪一步

- 重新检测为GPU模式；设备回归修复前失败、修复后通过。
- reader独立CPU回归修复前三项新增断言失败、修复后全部通过。
- 原设备状态测试与新增异常测试均通过memcheck，无内存错误或泄漏。
- 两项修复冻结后完成Release构建与完整17项CTest，全部通过；已更新实现记录，修复后源码和二进制hash另行归档。
- 命令、实际输出和最终集成结果统一记录在 [审核验证结果](../experiments/2026-10-29-parallel-training-review.md)；证据目录为 `/root/rivermind-data/uav_rgb/parallel_training/20261029-1938-fixes/`。

### 待确认与遗留

- 无需用户决策的遗留项。实际训练接线与性能验收继续按原方案后续工单实施。
