# 并行训练首张工单落地与集成

- 时间：2026-10-29 17:34开始
- 范围：`uav_rgb/docs/PARALLEL_TRAINING_PLAN.md` 第38节首张内存/状态工单，以及C角色独立reader/run锁。
- 结论：完成本批工单；完整并行训练方案仍在分阶段实施。

## 做了什么

- 按计划组织1名集成人员和3名agent：A实现设备资源/状态，B实现测试并准备kernel迁移风险清单，C实现reader/run锁并交叉审查设备模块。共享头、CMake由集成人员统一维护，GPU验证串行。
- 开工保存58份源码/合同文件SHA256、只读源码快照、编译器/CUDA/GPU信息及原13项CTest清单；重新构建基线并运行5项相关回归。证据位于 `/root/rivermind-data/uav_rgb/parallel_training/20261029-1734/`。
- 新增 `src/include/uavrgb/device/{buffer,context,state}.hpp`、`src/device/{buffer,context,state,state_validation}.cpp`，实现checked资源分配、显式stream/event、八段参数布局、严格Adam校验、候选上传后swap及独立host快照。
- 新增 `src/{batch_reader,run_owner}.cpp`及对应公共头，支持独立pread、顺序读取、文件身份检查、本地目录锁和非空目录保护。
- 新增 `src/tests/test_device_state.cpp`及helper、`test_batch_reader.cpp`、`test_run_owner.cpp`，在 `src/CMakeLists.txt` 注册新库和测试；新C++实现/测试文件均不超过280行。
- 交叉审查修复两处边界：类型view基址对齐检查；Event同步错误通过所传Context标记FAILED。补入错误注入及已提交D2H后容量异常的收尾测试。
- 非法device先检查编号范围，不故意调用越界CUDA API；memcheck没有关闭API错误报告或其他校验。
- 用哈希核对原有C++/CUDA源文件：除CMake接线外均与开工基线一致。发现另一会话修改了 `CLAUDE.md` 的几何split说明，予以保留，未算作本批成果。
- 更新方案状态与 [实现记录](../PARALLEL_TRAINING_IMPLEMENTATION.md)，将运行结果统一落在 [实验记录](../experiments/2026-10-29-parallel-training-first-batch.md)。
- 在同一证据目录交付 `candidate.patch` 和 `deliverables-sha256.json`，覆盖本批源文件/构建/文档，便于没有Git时审阅；补丁不包含另一会话的CLAUDE改动。

## 没做什么

- 未接resident训练循环、forward/backward/loss/Adam kernel或新CLI选项；原因：第38节明确首张工单只实现内存/状态上传与导出。
- 未实现workspace、db/dW归约、双pinned槽预取或copy/compute重叠，也未宣称加速；原因：需先完成下一张设备链及整步工单，再按计划单独验收P1/P2/P3。
- 未将RunOwner/BatchReader接入现有Trainer；原因：C角色本批交付独立模块，避免在设备整步尚不存在时提前暴露新训练路径。旧CLI仍缺少本轮run锁保护。
- 未修改旧checkpoint、实现v3、sampler精确恢复、异步文件writer或fsync事务；原因：本批只导出独立内存快照，旧加载部分污染和保存持久性风险不在本批修复范围。
- 未生成完整P0性能基线、固定batch性能序列或A/B加速报告；原因：本批尚无候选训练后端，仅记录源码/构建/功能基线。原有训练回归确实运行，但不作为性能实验。
- 未验证真实CUDA非法访存后的恢复、跨卡行为、NFS锁、磁盘写满和文件提交失败；原因：对应kernel/文件writer未实现，本机只对单卡状态搬运与CPU模块验证。事件错误是API返回值注入。
- 未实现reader内容SHA256校验；原因：本批按替换/截断检测合同提供stat指纹，时间戳不变的原地修改仍可能漏检，已保留首次失败证据。
- 未修改/测试UI、collect.py、仿真质量线或导航泛化；原因：有独立维护会话，本批保持文件分工；几何split外部改动未合并为本批验收。
- 未建立Git仓库、分支、worktree或提交；原因：当前目录没有Git，按方案采用只读基线、哈希清单和模块文件所有权，不替用户初始化版本库。

## 待确认

- 无阻塞本批的待确认项。下一工单依照计划第30节第2项继续设备Conv1→ReLU→Pool链。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `nvidia-smi -L` | GPU模式，实际设备信息已保存 |
| 基线独立Release构建及5项相关CTest | 通过，退出0 |
| 首次候选构建 | 失败：`Context::activate() const` 与冻结头的非const声明不同步；统一后重建通过，原日志保留 |
| reader首次CPU故障测试 | 失败：同tick原地修改未被stat识别；明确接口限制并改为确定性mtime变化用例，原输出保留 |
| 首批3个新增目标 | 通过，后续因审查修复重建并纳入完整回归 |
| 最终Release构建、`ctest -N`、串行完整CTest | 通过；数量和原始输出见实验记录 |
| `compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 test_device_state` | 通过，未关闭检查；错误和泄漏明细见实验记录 |
| `cuobjdump -lelf test_device_state` | 仅sm_75 |
| baseline源码哈希对比 | 原有C++/CUDA仅CMake变更；外部CLAUDE变更保留 |

## 风险与遗留

- 完整P0/P1退出门禁尚未通过，当前没有resident训练后端；本批通过不代表全方案完成。
- 裸设备指针和异步buffer复制依赖调用方遵守finish/event生命周期；FAILED上下文必须停止使用，析构不是成功证明。
- reader的文件不可变性依赖普通本地文件使用约束；stat指纹不能替代正式数据内容hash。
- 旧checkpoint与旧CLI的既有风险继续存在，接入新模块时必须完成事务和run所有权门禁。
- 无Git条件下后续合并仍须检查当前文件哈希，不得覆盖其他会话成果。

## 更新 2026-10-29 19:38：审核问题修复

- 后续审核确认reader对符号链接加 `..` 的解析错误，以及设备状态异常收尾遗漏FAILED标记；用户授权后两项均已修复。
- 增加reader路径回归与独立设备异常收尾测试，均验证修复前失败、修复后通过；修复版完整17项CTest通过，两项设备memcheck零错误、零泄漏。
- 本文前面的首批16项测试及历史hash仍对应首批版本；修复版详情见 [审核与修复过程](2026-10-29-1903-parallel-training-review.md)及 [验证结果](../experiments/2026-10-29-parallel-training-review.md)。
- 本轮未新增resident训练或性能优化，原有后续工单范围不变。
