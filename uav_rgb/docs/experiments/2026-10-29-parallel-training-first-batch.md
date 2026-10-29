# 并行训练首张工单：基线与验收结果

- 记录时间：2026-10-29T18:51+08:00
- 对应：[实现状态](../PARALLEL_TRAINING_IMPLEMENTATION.md)、[过程记录](../worklog/2026-10-29-1734-parallel-training-first-batch.md)。
- 范围：方案第38节内存/状态工单，以及独立reader/run锁；完整训练性能门禁尚未验收。
- 原始证据目录：`/root/rivermind-data/uav_rgb/parallel_training/20261029-1734/`。

## 环境与固定输入

实际 `nvidia-smi -L` 检测到 NVIDIA GeForce RTX 2080 Ti；GPU模式。CUDA Toolkit 12.8.93、
驱动580.119.02、GCC13.3.0、CMake3.28.3；Release、C++17、FP32，CUDA仅sm_75，未加入fast-math。
开工GPU存在selkies Python进程，本次不测吞吐，不把共享机器上的测试用时解释为加速。
环境原文位于 `baseline/environment.txt`。

状态用例：120×160四类CNN、100388参数、seed73、八段不同值且段内逐元素唯一，含负零；
Adam step17、epoch3、类别权重[0.5,1,2,4]，另测小尺寸fresh step0及空权重。
p/m/v往返与shape/offset要求精确比较，不使用浮点容差；原始snapshot不可被返回副本修改。
reader用例：现场生成7张3×8×16 UAVR，seed987，indices=[6,0,5,2,2,1,4]，
与原load_images逐bit比较；故障注入处理EINTR、连续17字节短读和EOF，临时数据由正式测试清理。

## 命令与退出码

每个日志保留完整实际命令和输出。全部GPU命令串行运行，CPU agent只独立执行CPU测试。

| 原始日志 | 退出码 |
| --- | --- |
| `baseline-configure.txt` | 0 |
| `baseline-build.txt` | 0 |
| `baseline-ctest.txt` | 0 |
| `candidate-configure.txt` | 0 |
| `candidate-build.txt` | 1 |
| `candidate-build-fixed.txt` | 0 |
| `candidate-ctest-list.txt` | 0 |
| `candidate-first-batch.txt` | 0 |
| `candidate-build-final.txt` | 0 |
| `candidate-ctest-final.txt` | 0 |
| `candidate-memcheck-final.txt` | 0 |
| `candidate-cubin-final.txt` | 0 |

所有构建从 `source uav_rgb/env.sh` 后执行。原始基线源码由只读tar快照解包到证据目录的
`baseline-source/`，单独构建到 `build/resident-baseline/`；候选构建位于 `build/resident-check/`。
`baseline/environment.txt` 中git status退出128，因为目录没有Git；未声称存在原始git commit。

## 基线回归

`baseline-ctest.txt` 原文：

```text
$ ctest --test-dir /root/claude-code-backup-20261028/code/uav_rgb/build/resident-baseline --output-on-failure -j 1 -R ^(test_tensor|test_nn_cuda|test_trainer|test_dataset_filter|test_dataset_split)$
Internal ctest changing into directory: /root/claude-code-backup-20261028/code/uav_rgb/build/resident-baseline
Test project /root/claude-code-backup-20261028/code/uav_rgb/build/resident-baseline
    Start  2: test_tensor
1/5 Test  #2: test_tensor ......................   Passed    0.00 sec
    Start  7: test_nn_cuda
2/5 Test  #7: test_nn_cuda .....................   Passed    0.37 sec
    Start  8: test_trainer
3/5 Test  #8: test_trainer .....................   Passed    8.49 sec
    Start 11: test_dataset_filter
4/5 Test #11: test_dataset_filter ..............   Passed    0.01 sec
    Start 12: test_dataset_split
5/5 Test #12: test_dataset_split ...............   Passed    0.00 sec

100% tests passed, 0 tests failed out of 5

Total Test time (real) =   8.88 sec

exit_code=0
```

基线发现13项CTest，本次先运行5项相关基线回归；没有把原始13项全量重测描述为已完成。
最终候选随后实际运行全量16项，其中包含原有训练器、过拟合及加权训练回归。

## 首次失败与修正

- 首次候选构建退出1：冻结头先改为非const，编译时实现尚未同步。
  原错误为 `error: no declaration matches ‘void uavrgb::device::Context::activate() const’`。
  待所有者统一声明/实现并冻结后重建通过，失败原日志 `candidate-build.txt` 保留。
- reader首次CPU用例退出1：同tick的同长度原地写入没有改变stat指纹。没有放宽生产检查，
  改成utimensat确定性推进mtime的元信息变化用例，并明确stat无法替代内容hash。
  命令、失败原文及CPU二进制hash见 [reader/锁结果](2026-10-29-parallel-training-reader-owner.md)。
- 初版新增3项测试通过后，交叉审查发现Event同步报错只抛异常而不标记Context FAILED。
  接口改为synchronize(Context&)，补入CUDA API返回值注入回归，并重新构建做最终全量验收。
- 哈希核对曾发现另一会话修改CLAUDE.md的几何split说明；原有C++/CUDA源码核对仍通过，
  外部文档改动予以保留，未作为本批实施内容。

## 最终完整CTest

```text
time=2026-10-29T18:24:11+08:00
$ ctest --test-dir /root/claude-code-backup-20261028/code/uav_rgb/build/resident-check --output-on-failure -j 1
Internal ctest changing into directory: /root/claude-code-backup-20261028/code/uav_rgb/build/resident-check
Test project /root/claude-code-backup-20261028/code/uav_rgb/build/resident-check
      Start  1: test_action
 1/16 Test  #1: test_action ......................   Passed    0.00 sec
      Start  2: test_tensor
 2/16 Test  #2: test_tensor ......................   Passed    0.00 sec
      Start  3: test_gradcheck
 3/16 Test  #3: test_gradcheck ...................   Passed    0.00 sec
      Start  4: test_linear_cuda
 4/16 Test  #4: test_linear_cuda .................   Passed    0.37 sec
      Start  5: test_conv_cuda
 5/16 Test  #5: test_conv_cuda ...................   Passed    0.38 sec
      Start  6: test_softmax_cuda
 6/16 Test  #6: test_softmax_cuda ................   Passed    0.36 sec
      Start  7: test_nn_cuda
 7/16 Test  #7: test_nn_cuda .....................   Passed    0.34 sec
      Start  8: test_trainer
 8/16 Test  #8: test_trainer .....................   Passed    8.20 sec
      Start  9: test_overfit
 9/16 Test  #9: test_overfit .....................   Passed   67.06 sec
      Start 10: test_weighted_loss
10/16 Test #10: test_weighted_loss ...............   Passed    0.30 sec
      Start 11: test_dataset_filter
11/16 Test #11: test_dataset_filter ..............   Passed    0.01 sec
      Start 12: test_dataset_split
12/16 Test #12: test_dataset_split ...............   Passed    0.00 sec
      Start 13: test_class_weights_train
13/16 Test #13: test_class_weights_train .........   Passed   12.28 sec
      Start 14: test_device_state
14/16 Test #14: test_device_state ................   Passed    0.54 sec
      Start 15: test_batch_reader
15/16 Test #15: test_batch_reader ................   Passed    0.01 sec
      Start 16: test_run_owner
16/16 Test #16: test_run_owner ...................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 16

Total Test time (real) =  89.87 sec

exit_code=0
```

## CUDA内存与泄漏检查

以下为生产状态上传/快照路径的真实GPU运行，未关闭API错误报告：

```text
time=2026-10-29T18:25:41+08:00
$ compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 /root/claude-code-backup-20261028/code/uav_rgb/build/resident-check/test_device_state
========= COMPUTE-SANITIZER
layout: 100388 parameters / 8 segments / checked overflow PASS
buffer/context/event: move / bounds / stream dependency PASS
state: p/m/v bitwise roundtrip / independent snapshot / rejected imports PASS
test_device_state: PASS
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors

exit_code=0
```

事件错误通过GNU ld wrap单次返回cudaErrorLaunchFailure，验证Context进入FAILED；没有制造真实
非法kernel或声称从已损坏的CUDA上下文恢复。已提交params D2H后再触发m容量错误，覆盖异常排空
和锁页staging生命周期；该路径也包含于本次memcheck。

## 架构核对

```text
time=2026-10-29T18:25:42+08:00
$ cuobjdump -lelf /root/claude-code-backup-20261028/code/uav_rgb/build/resident-check/test_device_state
ELF file    1: test_device_state.1.sm_75.cubin
ELF file    2: test_device_state.2.sm_75.cubin
ELF file    3: test_device_state.3.sm_75.cubin

exit_code=0
```

## 可追溯hash

开工清单 `baseline/sha256.json` 和只读源码快照 `baseline/source.tar.gz` 保留当前用户成果；
验收版本完整源码清单为 `candidate-source-sha256.json`，二进制清单为 `candidate-binary-sha256.json`。
原有C++/CUDA源文件与开工hash一致，唯一修改的原有构建文件是src/CMakeLists.txt；新增15份源/头/测试。

| 候选二进制 | SHA256 |
| --- | --- |
| `test_device_state` | `0e66593ee32a7e172e3bb17c6cc854f2dafa482cdcd7409fc009098f391fe223` |
| `test_batch_reader` | `80ac8c7370ae5883735e87ea4175978d2383b031a869da5ec8fd42b62cf52648` |
| `test_run_owner` | `f4d4c04e4a38b004397198d71e257c93bd8db5fc5260744735795c6d78358764` |
| `uavrgb_train` | `a97a39ca7f4b1b12d2836d07ce7fed830ac8dce9370e7180da78f76044ca5177` |

## 未运行与可得结论

- P1/P2/P3训练A/B、吞吐、p50/p95、稳态分配/拷贝统计、Nsight重叠：NOT_RUN，设备训练图尚未实现。
- compute-sanitizer synccheck：NOT_RUN，本批没有新增带线程同步的计算kernel；此处memcheck仅验收状态搬运。
- v3文件提交/磁盘失败、sampler精确恢复、真实CUDA上下文损坏恢复：NOT_RUN，对应执行器/codec不在本批。
- 跨GPU、NFS、仿真传感器故障、sim-time动作、可信数据泛化：NOT_RUN；单卡状态功能测试不证明这些能力。

可以确认本批内存、状态、reader和锁的已测功能通过；不能据此宣称resident训练可用、达到2倍加速，
或全方案P0/P1退出条件通过。总体实现状态另行持续维护。
