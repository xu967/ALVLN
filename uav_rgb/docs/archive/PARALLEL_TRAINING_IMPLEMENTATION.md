# 并行训练实现状态

- 更新：2026-10-29
- 对应规格：[PARALLEL_TRAINING_PLAN.md](PARALLEL_TRAINING_PLAN.md) 第38节首张工单。
- 当前结论：内存与状态上传/导出已验收；reader/run锁已独立验收。后续审核发现的路径解析与CUDA异常收尾问题已修复，修复版17项CTest通过。
- [过程记录](worklog/2026-10-29-1734-parallel-training-first-batch.md)；[实验结果](experiments/2026-10-29-parallel-training-first-batch.md)。

## 本批实现

| 模块 | 入口 | 已支持的行为 |
| --- | --- | --- |
| 设备/锁页内存 | `device/buffer.hpp` | move-only、容量与对齐校验、checked乘加、256字节arena布局、显式stream复制 |
| CUDA上下文/事件 | `device/context.hpp` | 固定device、non-blocking stream、依赖事件、显式finish、同步错误后FAILED |
| 参数布局/训练状态 | `device/state.hpp` | 按cnn_params的八段顺序计算布局、严格shape和完整Adam校验、上传p/m/v、梯度清零 |
| 独立快照 | `DeviceTrainState::snapshot` | 显式下载到独立CnnModel/TrainState，保留epoch/step/优化器超参数/类别权重 |
| 独立读取器 | `batch_reader.hpp` | 独占fd、pread读满、固定indices顺序、既有归一化、替换/截断/元信息变化检测 |
| 输出目录所有权 | `run_owner.hpp` | canonical路径、Linux非阻塞flock、非空目录保护、O_CLOEXEC、异常退出释放 |

设备实现单独构建为 `uavrgb_device`，依赖已有host模型类型和CUDA runtime。reader和run锁注册到
`uavrgb_core`。原有训练器尚未调用这些模块；当前没有新增可运行的resident训练CLI。

## 状态接口与生命周期

```cpp
#include "uavrgb/device/state.hpp"

auto model = uavrgb::make_cnn(120, 160, 73);
auto host_state = uavrgb::make_train_state(model, 0.002f);
uavrgb::device::Context context(0);
uavrgb::device::DeviceTrainState device_state(context, model, host_state);
auto snapshot = device_state.snapshot(context);
```

`import` 完整校验host输入，再分配候选存储、打包p/m/v并上传；`finish`成功后才swap。
非法输入不会部分覆盖活跃状态。`snapshot`返回独立host对象，修改快照不会影响设备状态。
它仅生成内存快照，不写checkpoint文件。

120×160四类模型布局为100388参数，八段offset保持既有合同。布局由实际shape计算，另有小尺寸回归。
Adam八段必须有相同t/lr/beta1/beta2/eps，p/m/v必须有限，v非负；host的int step范围继续生效。
类别权重本批作为快照元信息保存，损失kernel尚未接入。

`copy_from_host`和`copy_to_host`只提交到指定stream。调用者必须在finish/event完成前保持两端存储
有效，不得移动或销毁仍在使用的buffer。事件使用 `event.synchronize(context)`，错误会令该context
进入FAILED并拒绝继续提交。析构只负责不抛异常地回收资源，不能替代显式错误检查。

## 可复跑验收

从 `code/` 执行：

```bash
source uav_rgb/env.sh
nvidia-smi -L
cmake -S "$UAV_RGB_ROOT/src" -B "$UAV_RGB_ROOT/build/resident-check" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$UAV_RGB_ROOT/build/resident-check" --parallel 8
ctest --test-dir "$UAV_RGB_ROOT/build/resident-check" -N
ctest --test-dir "$UAV_RGB_ROOT/build/resident-check" --output-on-failure -j 1
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$UAV_RGB_ROOT/build/resident-check/test_device_state"
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$UAV_RGB_ROOT/build/resident-check/test_device_transfer_errors"
```

首批版本通过16项CTest；审核修复版在独立 `build/resident-review` 中通过17项CTest和两项memcheck。
上面的命令会重建指定目录；历史 `resident-check` 二进制未在本轮改写。测试包含逐bit状态往返、负零、所有权move、非法导入不污染、
跨stream事件、事件错误注入、D2H中途失败收尾、pread故障、同目录双进程竞争与exec锁释放。
原始日志、源码和二进制hash见实验记录；没有把这些功能测试当作吞吐基准。

## 审核修复更新（2026-10-29 19:38起）

- reader保留符号链接后接 `..` 的Linux路径解析顺序，避免静默读到另一份合法数据；运行中链接改指向仍可检出。
- 状态传输异常收尾检查设备切换与stream同步错误，标记Context FAILED并输出诊断，同时保留首次异常；host异常且排空成功时Context仍可使用。
- 新增正式回归覆盖上述路径；两项回归均验证了修复前失败、修复后通过。完整17项CTest通过，状态往返及异常收尾两个memcheck均零错误、零泄漏。
- [修复过程](worklog/2026-10-29-1903-parallel-training-review.md)、[修复实验结果](experiments/2026-10-29-parallel-training-review.md)。首次验收的hash和日志保持原样，修复版本证据位于 `/root/rivermind-data/uav_rgb/parallel_training/20261029-1938-fixes/`。

## 尚未实施的门禁

- resident forward/backward、CE/裁剪/Adam整步、trainer/eval/CLI接线：按第30节后续步骤实施。
- workspace、device view/算子接口、db/dW并行归约、双槽预取/copy重叠：当前没有训练执行器，尚未开始性能验收。
- v3 checkpoint、sampler精确恢复、唯一临时文件/fsync提交协议：本批只有内存快照，旧checkpoint事务风险仍待修复。
- RunOwner接入真实trainer：当前只能保证使用该模块的调用方互斥，旧CLI尚未获得该保护。
- 数据内容hash、传感器有效性和sim-time门禁：本批reader只有stat身份/元信息指纹，不能保证检出时间戳不变的原地修改。
- 真实kernel异步崩溃、多卡、NFS、导航泛化和性能加速：未验证；单卡功能验证不代表这些门禁通过。

下一张训练线工单是第30节第2项：Conv1→ReLU→Pool设备链，与既有输出/argmax逐层对拍，
然后用时间线证明层间没有H2D/D2H。总体P0/P1退出条件仍未通过，不切换默认backend。
