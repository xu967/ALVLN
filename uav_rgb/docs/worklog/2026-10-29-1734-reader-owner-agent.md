# 并行训练首批：reader 与 run 所有权

- 时间：2026-10-29 17:54 CST
- 范围：`PARALLEL_TRAINING_PLAN.md` 第16/17/29/38节中的独立 CPU 模块
- 结论：完成本 agent 负责的独立模块；训练接线由集成人员另行推进

## 做了什么

- 新增 `src/include/uavrgb/batch_reader.hpp`、`src/batch_reader.cpp`：独占 `O_CLOEXEC` fd、复用 uint8 scratch，按给定下标通过 pread 读取，保持旧路径逐 bit 相同的 NCHW FP32 归一化。
- 构造时继续调用 `load_uavr` 校验头、标签、scene split 和可选碰撞过滤；校验前后核验 fd/path 身份，之后不使用共享 ifstream。
- 新增 `read_exact_at`：检查 off_t 范围、处理 EINTR/短读/EOF，拒绝空目标；batch 先整体检查容量和下标再写入。
- 每批前后比较 dev/ino/size/mtime/ctime，检测路径替换、截断及能反映到元信息的修改。
- 新增 `src/include/uavrgb/run_owner.hpp`、`src/run_owner.cpp`：canonical 路径、非阻塞排他 flock、`O_CLOEXEC`、异常/退出自动释放，以及非 resume 非空目录拒绝。
- 新增正式测试 `src/tests/test_batch_reader.cpp`、`src/tests/test_run_owner.cpp`；测试临时目录由 RAII 清理，编译产物放在 DATA_ROOT。

## 没做什么

- 没修改 `trainer.cpp`、CLI、CMake、checkpoint 或旧 dataset 实现；原因：这些属于集成负责人或后续工单，避免共享文件冲突。
- 没实现 worker、双槽、copy stream、快照 writer；原因：本轮只交独立 reader/run 锁，设备执行器尚在首批状态工单。
- 没验证 resume 的 checkpoint 来源、配置和数据合同；原因：`RunOwner(..., true)` 只允许持锁后进入已有目录，验证归训练控制层。
- 没实现内容 SHA256 或防对抗修改检查；原因：当前 `fingerprint()` 明确是 stat 身份/元信息指纹，正式 run manifest 的内容 hash 属于后续接线。
- 没运行 GPU 任务或性能基准；原因：GPU 验证由集成负责人串行调度，本模块测试只用 CPU。
- 没验证 NFS/跨机器锁语义；原因：首个支持范围为 Linux 本地文件系统。

## 待确认

- 无阻塞决策；CMake 中 `test_batch_reader` 需链接 `-Wl,--wrap=pread`，已通知集成负责人。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `g++ -std=c++17 -O2 -Wall -Wextra` 独立编译两个模块及测试 | 通过，无编译警告 |
| `test_batch_reader` 首跑 | 失败：同 tick 写入未改变 stat 元信息，已保留结果记录 |
| `test_batch_reader` 确定性 mtime 故障注入及 move 所有权修订后 | 通过 |
| `test_run_owner` 双进程、异常、退出、exec 与目录边界 | 通过 |
| 新增 cpp 行数检查 | 全部不超过280行 |
| 统一 CMake/CTest 回归 | 由集成人员执行，本 agent 未运行 |

实际命令、首跑失败输出、末次输出及二进制 hash 见
`docs/experiments/2026-10-29-parallel-training-reader-owner.md`。

## 风险与遗留

- stat 无法保证发现同一文件系统 tick 或伪造/不变时间戳的同长度原地写入；首跑已实际暴露此边界，接口注释和结果文档明确说明。
- 同一 reader 的 scratch 不支持并发使用，后续每个 worker 必须独占实例。
- 读取失败可能已写部分目标；调用方只能在整批成功返回后发布 READY。
- run 锁文件保留，不删除锁文件，也不以文件存在判锁；外部主动删除/替换持锁中的锁文件不在本地协作合同内。
