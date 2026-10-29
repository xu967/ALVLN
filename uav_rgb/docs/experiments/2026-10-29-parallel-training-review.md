# 并行训练首批实现：审核验证结果

- 时间：2026-10-29 19:03，Asia/Shanghai。
- 环境：Linux，GPU模式，`nvidia-smi -L` 列出 NVIDIA GeForce RTX 2080 Ti；CUDA12.8，Release构建。
- 对应：[审核过程与发现](../worklog/2026-10-29-1903-parallel-training-review.md)。
- 构建证据：`/root/rivermind-data/uav_rgb/parallel_training/20261029-1903-review/`。

## 1. 独立构建与新增测试

实际命令（从code目录执行）：

```bash
source uav_rgb/env.sh
nvidia-smi -L
cmake -S "$UAV_RGB_ROOT/src" -B "$UAV_RGB_ROOT/build/resident-review" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$UAV_RGB_ROOT/build/resident-review" --parallel 8
ctest --test-dir "$UAV_RGB_ROOT/build/resident-review" --output-on-failure -j 1 -R '^(test_device_state|test_batch_reader|test_run_owner)$'
```

配置、构建均退出0。CTest实际输出：

```text
    Start 14: test_device_state
1/3 Test #14: test_device_state ................   Passed    0.56 sec
    Start 15: test_batch_reader
2/3 Test #15: test_batch_reader ................   Passed    0.01 sec
    Start 16: test_run_owner
3/3 Test #16: test_run_owner ...................   Passed    0.01 sec

100% tests passed, 0 tests failed out of 3
Total Test time (real) =   0.58 sec
```

这是本轮复跑结果。原始16/16及memcheck通过属于此前验收，本轮只做证据核对，没有重新执行这两项完整验证。

## 2. reader路径语义复现

CPU审核agent以生产 `BatchReader` 和 `load_uavr` 独立编译最小探针，用项目合成数据生成器生成两份合法UAVR：

```text
base/link -> base/real/inner
base/data.uavr: N=2
base/real/data.uavr: N=3
input=base/link/../data.uavr
```

实际编译退出0，探针报告：

```text
POSIX/load_uavr N=3; BatchReader N=2
probe_exit=3
```

退出3是探针专门用于标记“复现到错误”的失败状态；不是生产代码主动拒绝输入。原因是 `lexically_normal()` 在符号链接解析之前消除了 `link/..`。正式测试缺少此路径分支。临时探针、数据和程序已清理。

## 3. CUDA异常排空遗漏FAILED复现

主审核在真实GPU上执行状态上传和D2H，使用GNU ld wrap替换收尾同步返回值。为避免真实损坏GPU上下文，wrapper先执行真实同步，再单次返回 `cudaErrorLaunchFailure`；此实验不证明真实非法访存后的资源恢复。

临时探针核心代码：

```cpp
bool injected = false, consumed = false;
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t);
extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  const auto real = __real_cudaStreamSynchronize(stream);
  if (injected) {
    injected = false;
    consumed = true;
    return cudaErrorLaunchFailure;
  }
  return real;
}
// Context context; make_cnn(8,16,73); make_train_state(model,0.001f);
// DeviceTrainState device(context,model,state);
device.adam_m() = DeviceBuffer(sizeof(float), context.device());
injected = true;
try { auto snapshot = device.snapshot(context); }
catch (const std::exception& e) { /* 记录首次异常 */ }
// 记录 consumed、context.failed()，并尝试context.activate()。
```

实际通过Python subprocess参数数组运行g++；以下等价命令用于说明编译参数，`probe.cpp`是本次已清理的临时文件，不是仓库新目标：

```bash
g++ -std=c++17 -I"$UAV_RGB_ROOT/src/include" -I/usr/local/cuda-12.8/include probe.cpp \
  "$UAV_RGB_ROOT/build/resident-review/libuavrgb_device.a" \
  "$UAV_RGB_ROOT/build/resident-review/libuavrgb_core.a" \
  -L/usr/local/cuda-12.8/lib64 -Wl,-rpath,/usr/local/cuda-12.8/lib64 \
  -lcudart -Wl,--wrap=cudaStreamSynchronize -o probe
```

实际输出：

```text
compile_exit=0
snapshot_exception=device copy: 空指针或容量不足
injection_consumed=1 context_failed=0 activate_accepted=1
probe_exit=3
```

预期应为 `context_failed=1 activate_accepted=0`。探针退出3表示复现了错误。外层Python命令退出0仅表示脚本成功完成并记录子进程结果，不是探针通过。

## 4. 归档一致性

证据agent逐项核对原验收目录 `20261029-1734`：

- `candidate-source-sha256.json`：68项全部匹配当前源码。
- `deliverables-sha256.json`：24项全部匹配当前交付物。
- 候选4个二进制：全部匹配记录的SHA256。
- 原始CTest和memcheck日志存在，分别为16/16通过、0 errors及0 bytes leaked，与实验文档一致。

本次审核没有修改生产源文件，因此这些历史测试有明确对应版本；但它们没有覆盖本轮两项复现的触发条件。

## 更新 2026-10-29 19:38：审核问题修复验证

用户已授权修复两项问题。本节记录修复版本，前述19:03结果及原始验收hash属于修复前版本，不覆盖历史证据。

- 环境：重新执行 `nvidia-smi -L`，实际列出RTX2080Ti；继续使用CUDA12.8、Release、sm_75。
- 原始日志目录：`/root/rivermind-data/uav_rgb/parallel_training/20261029-1938-fixes/`。
- 构建目录：`build/resident-review`，所有GPU命令串行执行。

### CUDA回归：先失败，再修复

先新增正式 `test_device_transfer_errors` 并注册CTest，保持旧生产实现执行：

```bash
cmake --build "$UAV_RGB_ROOT/build/resident-review" --parallel 8 --target test_device_transfer_errors
ctest --test-dir "$UAV_RGB_ROOT/build/resident-review" --output-on-failure -j 1 -R '^test_device_transfer_errors$'
```

构建退出0，CTest退出8；原始 `red-ctest.log`：

```text
test_device_transfer_errors: FAIL: 收尾FAILED状态错误
0% tests passed, 1 tests failed out of 1
```

修复 `state.cpp` 后重新构建 `test_device_state`、`test_device_transfer_errors`，执行：

```bash
ctest --test-dir "$UAV_RGB_ROOT/build/resident-review" --output-on-failure -j 1 -R '^test_device_(state|transfer_errors)$'
```

退出0，2/2通过。新回归覆盖同步失败、设备切换失败、两者同时失败、仅host异常四种情况；保留首次异常类型/消息，核对排空及FAILED状态，失败后拒绝activate/finish。

### 修复后的CUDA内存检查

对以下两个可执行文件分别执行同一命令：

```bash
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$UAV_RGB_ROOT/build/resident-review/test_device_state"
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 1 "$UAV_RGB_ROOT/build/resident-review/test_device_transfer_errors"
```

均退出0，均为 `LEAK SUMMARY: 0 bytes leaked in 0 allocations`、`ERROR SUMMARY: 0 errors`。错误回归的实际输出包含：

```text
device transfer cleanup: cudaStreamSynchronize: unspecified launch failure
device transfer cleanup: cudaSetDevice: unknown error
device transfer cleanup: cudaSetDevice: unknown error
device transfer cleanup: cudaStreamSynchronize: unspecified launch failure
test_device_transfer_errors: PASS (sync/set/both/host-only)
```

这四条stderr是预期注入的错误诊断，不是memcheck检出的错误。wrapper先完成真实CUDA调用再注入返回值，不测试真实驱动损坏后的恢复。正式测试源码保存在 `src/tests/`，不留一次性探针。

### reader回归：先失败，再修复

agent使用生产reader/dataset/tensor源码独立CPU构建正式测试。以下命令中 `test_batch_reader` 位于自动清理的临时目录；完整参数及输出保存在 `reader-red.log`、`reader-green.log`：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -I"$UAV_RGB_ROOT/src/include" \
  "$UAV_RGB_ROOT/src/tests/test_batch_reader.cpp" "$UAV_RGB_ROOT/src/batch_reader.cpp" \
  "$UAV_RGB_ROOT/src/dataset.cpp" "$UAV_RGB_ROOT/src/tensor.cpp" \
  -Wl,--wrap=pread -o test_batch_reader
./test_batch_reader
```

修复前编译退出0、测试退出1，新增三项检查失败：

```text
FAIL: 符号链接后接父目录与旧 loader 元数据一致
FAIL: 符号链接后接父目录的像素与旧 loader 逐 bit 一致
FAIL: 原始路径符号链接改指向仍检出替换
test_batch_reader: FAIL
```

去掉构造函数中的 `lexically_normal()` 后，编译退出0、测试退出0，输出 `test_batch_reader: PASS`。用例包含不同样本数/seed的两份合法数据、标签/三类split比对、像素逐bit比对，以及运行中链接改指向另一inode的检测。

### 最终集成验证

两项修复冻结后执行：

```bash
cmake --build "$UAV_RGB_ROOT/build/resident-review" --parallel 8
ctest --test-dir "$UAV_RGB_ROOT/build/resident-review" --output-on-failure -j 1
```

构建退出0；CTest退出0，完整输出见 `final-ctest.log`。摘要：

```text
14/17 Test #14: test_device_state ................   Passed    0.55 sec
15/17 Test #15: test_device_transfer_errors ......   Passed    0.41 sec
16/17 Test #16: test_batch_reader ................   Passed    0.01 sec
17/17 Test #17: test_run_owner ...................   Passed    0.01 sec
100% tests passed, 0 tests failed out of 17
Total Test time (real) = 125.15 sec
```

原有训练、过拟合、加权训练及数据集测试也全部通过。该用时是功能回归耗时，不是吞吐基准，不用于计算加速比。

修复版源码清单为 `fixed-source-sha256.json`，最终二进制清单为 `fixed-binary-sha256.json`；旧首批清单不覆盖修复版本。测试源码和生产改动经交叉审核，收尾诊断已在memcheck日志核对。未做真实驱动损坏恢复、多GPU或训练性能验证。
