# 修复两份审核报告的「必须修」项（形状校验 / N=0 / CUDA 错误异常化 / int64 规模）

- 时间：2026-10-28 23:43
- 范围：`src/` 的 CPU Linear/Conv/softmax-CE、CUDA Linear/Conv/softmax-CE、对应测试与 `src/CMakeLists.txt`；审核报告 `docs/worklog/2026-10-28-2334-audit-cuda.md`、`docs/worklog/2026-10-28-2334-audit-cpp.md`
- 结论：完成（本次任务清单 1-5 全部落地并真跑验证）；两份审核报告里的「应该修/可以改」大部分按要求留给下一轮，逐条列在「没做什么」

## 做了什么

### 1. 形状/参数校验（任务 #1，同时覆盖 CUDA 审核 #9、CPU 审核 #1/#4/#12）

- `src/include/uavrgb/ops.hpp`：新增 `detail::check_linear_params()`（inline，CPU/CUDA 共用）：`in/out > 0`、`W.numel() == out*in`、`b.numel() == out`，消息带期望值与实际值。
- `src/include/uavrgb/conv.hpp`：新增 `detail::check_conv_params()`：`in/out/kh/kw > 0`、`pad >= 0`、`W.numel() == out*in*kh*kw`、`b.numel() == out`。
- `src/ops_cpu.cpp`：`linear_forward`、`linear_backward`、`gradcheck_linear` 入口调用参数校验；`gradcheck_linear` 增加 `x.ndim()==2`、列数匹配、`N>0` 校验（原来 rank<2 抛 `std::out_of_range`，现在抛 `std::invalid_argument`）。
- `src/conv_cpu.cpp`：`conv2d_forward`、`conv2d_backward` 入口调用参数校验；`oh/ow <= 0` 检查保留。
- `src/conv_gradcheck.cpp`（新）：`gradcheck_conv2d` 增加 `ndim==4`、通道数匹配、`N>0` 校验后再访问 `dims()[2]/[3]`；`oh/ow <= 0` 抛异常（原来 `c_in` 用 `m.in` 导致形状错误时静默返回 0）。
- `src/linear_cuda.cu`、`src/conv_cuda.cu`：CUDA 包装同样先调用共用参数校验，再校验 x/dy 形状。
- `src/softmax_cuda.cu`：`check_dims` 增加 `N==0`、`C==0` 抛 `std::invalid_argument`（与 CPU 语义一致）。

### 2. 空输入 N=0（任务 #2，同时覆盖 CUDA 审核 #2、CPU 审核 #5 的入口部分）

- `src/linear_cuda.cu`：`linear_forward_cuda` 在 `n==0` 时直接返回 `[0,out]` 空张量；`linear_backward_cuda` 返回 `dW[out,in]`/`db[out]` 全零、`dx[0,in]` 空张量。均在任何 `cudaMalloc`/launch 之前返回。
- `src/conv_cuda.cu`：`conv2d_forward_cuda` 在 `n==0` 时返回 `[0,out,oh,ow]` 空张量；`conv2d_backward_cuda` 返回零 dW/db 与空 dx；另外 `H==0||W==0`（pad 足够大时合法）会使 `dx` 元素数为 0，单独跳过 dx kernel 的 launch（dW/db 仍按 CPU 语义计算）。
- `src/ops_cpu.cpp`：`softmax_ce_loss` 在 N=0 时由「返回 NaN」改为抛 `std::invalid_argument("softmax_ce_loss: 空 batch（N=0）")`；`softmax_ce_grad` 选择同样抛异常（二选一中选“抛异常”，与 loss 一致）；`C==0` 单独抛「C 必须为正」，不再伪装成「label 越界」。
- `src/softmax_cuda.cu`：与 CPU 一致，N=0 / C=0 抛 `std::invalid_argument`，不 launch kernel。
- `gradcheck_linear` / `gradcheck_conv2d`：N=0 抛 `std::invalid_argument`，不再返回「看起来完美」的 0。

### 3. CUDA 错误不再 `std::exit`（任务 #3，CUDA 审核 #3）

- `src/linear_cuda.cu`、`src/conv_cuda.cu`、`src/softmax_cuda.cu` 三处的 `CUDA_CHECK` 与 `check_after_kernel` 从「打印 + `std::exit(EXIT_FAILURE)`」改为抛 `std::runtime_error`，消息含 `文件:行号` 与 `cudaGetErrorString`。
- 栈上 `DeviceBuf` 正常析构；用仓库外 OOM 探针实测：已分配的 68 MiB 设备缓冲在 catch 前已归还，全程净泄漏 0.000 MiB（输出见下）。

### 4. int32 规模溢出（任务 #4，CUDA 审核 #1）

- 三个 `.cu` 的 kernel 内 `idx`/`total` 全部改为 `std::int64_t`；下标表达式统一 int64。
- host 包装新增 `check_elems()`：`x`/`y`/`dlogits`/`W`/`dy` 元素数按 int64 计算，超过 `std::numeric_limits<int>::max()` 抛 `std::invalid_argument("... shape too large: ... > INT_MAX")`，不再静默回绕。
- `cudaMemcpy` 字节数一律 `std::size_t`（由 int64 元素数乘 `sizeof(float)` 得到），不再有 `n*out*sizeof(float)` 的 int 乘法。
- `blocks` 用 int64 计算后再转 int。
- 顺带把 conv 的 `oh/ow = h + 2*pad - kh + 1` 在 host 与 kernel 内都用 int64 计算（`2*pad` 在 int 下会溢出；CUDA 侧；CPU 侧未改，见「没做什么」）。
- 实跑负例：`Conv2d(1,65537,1,1,0)` + `x=[1,1,256,256]`（审核里会静默错值的形状）现在直接抛 `conv2d_cuda: shape too large: y/dy 元素数=4295032832 > INT_MAX`；`Linear(1,65536)` + `x=[32768,1]` 抛 `y 元素数=2147483648 > INT_MAX`。

### 5. 行数预算与拆文件（任务 #5，CPU 审核 #13）

- 把 `conv_forward_f64`、`conv_loss_f64`、`gradcheck_conv2d` 从 `src/conv_cpu.cpp` 移到新文件 `src/conv_gradcheck.cpp`（声明仍在 `conv.hpp`），未删注释、未压行。
- `src/CMakeLists.txt` 注册 `conv_gradcheck.cpp`；sm_75 FORCE 语句未动。

### 6. 负例测试（任务完成标准）

- 复用原 7 个 ctest（未新增可执行文件，负例并入对应测试二进制）：
  - `src/tests/test_gradcheck.cpp`：Linear W/b 元素数不足、`gradcheck_linear` rank<2、`gradcheck_conv2d` rank<4 / 通道不匹配 / N=0、Conv2d W 元素数不足、pad 太小、CPU softmax loss/grad N=0、C=0。
  - `src/tests/test_linear_cuda.cpp`：CUDA W 元素数不足、x 列数不匹配、dlogits 列数不匹配、元素数 > INT_MAX、N=0 前向返回 `[0,out]` 空张量、N=0 反向梯度形状正确且全零。
  - `src/tests/test_conv_cuda.cpp`：通道不匹配同时测 CPU 与 CUDA、CUDA W 元素数不足、CUDA pad 太小、元素数 > INT_MAX（65537×256×256）、N=0 前向空张量、N=0 反向零梯度。
  - `src/tests/test_softmax_cuda.cpp`：CPU/CUDA loss 与 grad 在 N=0 抛 `invalid_argument`，C=0 也抛。

## 没做什么

按任务要求「不要重写测试覆盖测试保护不足」「不做性能优化」「不改数值算法」，第 1 轮审核里以下条目明确未做：

### CUDA 审核（2026-10-28-2334-audit-cuda.md）

- **应该修 #4**：`test_softmax_cuda` 的「label 越界」用例仍然只有 2 个 label、实际命中「数量不一致」分支，范围检查仍未被测试（假绿未修）。原因：属于测试保护不足，按要求留到下一轮。
- **应该修 #5**：只补了与本次校验相关的 CUDA 负例（前向通道/W/pad/N=0/INT_MAX），没有补 `conv2d_backward_cuda` 的 dy 形状负例、也没有系统性补齐 linear/conv CUDA 校验的完整负例矩阵。原因：测试保护不足整体留到下一轮。
- **应该修 #6**：doc 04 §6 要求的 NaN/Inf 用例、固定种子随机对拍、非方核（kh≠kw）parity、「注入 bug 必须变红」自检，均未加。原因：测试覆盖范围，超出本次修复项。
- **可以改 #7**：每 kernel `cudaDeviceSynchronize` + 每次调用临时 malloc/free 的固定开销（实测 285-354 µs/call）未动，持久缓冲/异步化未做。原因：任务明确「不要做性能优化，另一条线在处理」。
- **可以改 #8**：softmax grad 每元素重算整行 max/sum 的 O(N·C²) 未改。原因：性能优化，超出范围。
- **可以改 #10**：测试里的 `max_abs_err` 仍不先比形状。原因：测试保护不足，下一轮。

### CPU 审核（2026-10-28-2334-audit-cpp.md）

- **必须修 #2**：gradcheck 仍不调用生产 `linear_backward`/`conv2d_forward`/`conv2d_backward`/`softmax_ce_loss` 做比对（`test_gradcheck` 对 conv 前/反向与 loss 归一化仍是零保护）。原因：任务明确「不要重写测试去覆盖所有审核提到的『测试保护不足』问题（那是下一轮）」。
- **应该修 #3**：CPU/CUDA 的「label 越界」测试仍无效（`{2,9}` 只有 2 个元素）。原因：同上，下一轮。
- **应该修 #6**：CPU 侧非方核（kh≠kw）零覆盖未补。原因：测试覆盖范围，下一轮。
- **应该修 #7**：独立 NumPy/PyTorch 测试级 oracle 未加（PyTorch 本身被铁律禁止）。原因：下一轮/超出范围。
- **可以改 #8**：`Tensor::numel()` 的乘法溢出检查未加。原因：超出本次修复项。
- **可以改 #9**：`operator[]` 越界检查未加。原因：超出本次修复项。
- **可以改 #10**：CPU 侧 `pad` 无上界、`2*pad` 有符号溢出仍未改（只修了 CUDA 侧）。原因：超出本次修复项；CUDA 审核 #10 的同类问题因本次 int64 改造顺带在 `.cu` 修了。
- **可以改 #11**：`ActionAdapter::quantize` 负 eps/阈值边界未改、未加测试。原因：超出本次修复项。
- **可以改 #12**：`gradcheck_linear` rank<2 抛 `std::out_of_range` —— 已随任务 #1 修好（现在 `invalid_argument`），这里记录为「实际做了」。
- **可以改 #13**：重复实现只做了「拆文件守住 280 行」，没有做 FP64/FP32 模板化合并（`conv_forward_f64` 仍与 `conv2d_forward` 重复）。原因：本轮只需守住行数预算，合并重构超出范围。
- **可以改 #14**：`test_tensor` 负维度、`conv2d_backward` dy 负例、`test_action` 诊断信息等测试缺口未补。原因：测试保护不足，下一轮。

### 其它明确未做

- `src/nn_cuda.cu` 仍保留 `std::exit` 版 `CUDA_CHECK`（不在任务点名的三个 `.cu` 内，且属于另一条线）。`src/nn_cpu.cpp`、`src/include/uavrgb/nn.hpp` 未动。
- 未改 `src/include/uavrgb/observation.hpp`、`src/include/uavrgb/event.hpp`（任务禁止）。
- 未碰 `sim/`、`ui/`、`tools/`、`experiments/`、`docs/`（仅写本 worklog）。
- 未改任何数值算法、阈值、归一化、下标约定；未加性能优化、持久显存、异步。
- 未跑 `compute-sanitizer --tool racecheck`（kernel 访存模式未变，上轮为 0 hazard；本轮只跑 memcheck）。
- 未做真实 OOM 之外的其他 CUDA 错误注入（如非法指针）；异常路径用 OOM 实测。
- CPU 侧 `conv2d_forward/backward/gradcheck_conv2d` 的 `int64 dims -> int` 截断、`n*in` 等 CPU 内乘法溢出未加显式上限（审核 CPU「未检查」项），本次 int64 上限校验只落在 CUDA host 包装。
- 空 batch 语义决策：linear/conv 的 CUDA 入口返回空/零（不抛），softmax loss/grad 与两个 gradcheck 统一抛 `invalid_argument`；如上层希望 softmax 空 batch 返回 0.0f 而不是异常，需要另行拍板（见「待确认」）。

## 待确认

- 空 batch 的接口语义：CPU/CUDA softmax 目前都抛 `invalid_argument`（按任务要求「空 batch 是调用方错误」）；linear/conv 的 CUDA 入口按任务要求返回空/零。如果训练器希望空 batch 被静默跳过，需要在调用方 catch，或未来统一改成返回 0.0f/空——这是产品语义，需确认。
- `gradcheck_conv2d`/`gradcheck_linear` 的 N=0 抛异常是本轮新增行为（原为静默返回 0），若有旧调用方依赖「返回 0」，需要同步调整。

## 验证到哪一步

全部在真机 RTX 2080 Ti（sm_75，CUDA 12.8）实跑，仓库 `src/build` 正式构建目录。

### 1. 构建（clean rebuild，无告警）

```console
$ source env.sh
$ cmake --build src/build --clean-first 2>&1 | tee /tmp/uavrgb-probe/clean_build.log | tail -3
[23/25] Linking CXX executable test_conv_cuda
[24/25] Linking CXX executable test_linear_cuda
[25/25] Linking CXX executable test_nn_cuda
$ grep -iE "warning|error" /tmp/uavrgb-probe/clean_build.log || echo "(none)"
(none)
```

（`clean_build.log` 是当时的临时构建日志；25/25 全部成功，逐行扫描无 warning/error。）

### 2. ctest 全绿（原 7 个，含本轮负例）

```console
$ ctest --test-dir src/build --output-on-failure
1/7 Test #1: test_action ......................   Passed    0.00 sec
2/7 Test #2: test_tensor ......................   Passed    0.00 sec
3/7 Test #3: test_gradcheck ...................   Passed    0.00 sec
4/7 Test #4: test_linear_cuda .................   Passed    0.37 sec
5/7 Test #5: test_conv_cuda ...................   Passed    0.37 sec
6/7 Test #6: test_softmax_cuda ................   Passed    0.36 sec
7/7 Test #7: test_nn_cuda .....................   Passed    0.37 sec

100% tests passed, 0 tests failed out of 7
Total Test time (real) =   1.49 sec
```

（负例并入上述 4 个测试二进制，ctest 条目数仍为 7；测试失败会以非零退出码让对应条目变红。）

### 3. 数值未回归（直接跑二进制）

```console
$ ./src/build/test_gradcheck
loss=0.829538 gradcheck_max_err=2.75298e-08
$ ./src/build/test_linear_cuda
linear parity (in=8 out=4 N=16)
  max|y_cpu - y_cuda|    = 4.76837e-07
  max|dW_cpu - dW_cuda|  = 1.90735e-06
  max|db_cpu - db_cuda|  = 0
  max|dx_cpu - dx_cuda|  = 2.38419e-07
  worst                   = 1.90735e-06
test_linear_cuda: PASS
$ ./src/build/test_conv_cuda
conv parity pad=1 ... worst = 1.52588e-05
conv parity pad=0 ... worst = 1.52588e-05
conv gradcheck_max_err = 6.02006e-09
test_conv_cuda: PASS
$ ./src/build/test_softmax_cuda
softmax_ce parity (N=6 C=5)
  loss_cpu = 5.85808, loss_cuda = 5.85808
  max|grad_cpu - grad_cuda| = 4.65661e-10
test_softmax_cuda: PASS
```

与审核报告基线（gradcheck 2.75e-8 / 6.02e-9、parity 1.5e-5）一致，数值算法未变。

### 4. 新增负例的真实异常消息（仓库外探针，链接 `libuavrgb_core.a`，非仓库文件）

```console
$ g++ -std=c++17 -O2 -I src/include -I /usr/local/cuda/include messages.cpp \
      src/build/libuavrgb_core.a -L/usr/local/cuda/lib64 -lcudart -o messages
$ ./messages
Linear W too small (cpu)         invalid_argument: Linear: W.numel() 应为 out*in=12（实际 2）
Linear b wrong (cuda)            invalid_argument: Linear: b.numel() 应为 out=3（实际 1）
gradcheck_linear x rank1         invalid_argument: gradcheck_linear: x 必须为 2 维 [N,in]（实际 ndim=1）
gradcheck_conv2d x rank3         invalid_argument: gradcheck_conv2d: x 必须为 4 维 [N,in,H,W]（实际 ndim=3）
gradcheck_conv2d channel         invalid_argument: gradcheck_conv2d: x 通道数应为 in=1（实际 3）
conv2d W too small (cuda)        invalid_argument: Conv2d: W.numel() 应为 out*in*kh*kw=18（实际 2）
conv2d pad too small (cuda)      invalid_argument: conv2d_cuda: 输出尺寸非正（pad 太小）
conv2d > INT_MAX (cuda)          invalid_argument: conv2d_cuda: shape too large: y/dy 元素数=4295032832 > INT_MAX
linear > INT_MAX (cuda)          invalid_argument: linear_cuda: shape too large: y 元素数=2147483648 > INT_MAX
softmax loss N=0 (cpu)           invalid_argument: softmax_ce_loss: 空 batch（N=0）
softmax loss N=0 (cuda)          invalid_argument: softmax_cuda: 空 batch（N=0）
softmax grad N=0 (cuda)          invalid_argument: softmax_cuda: 空 batch（N=0）
softmax C=0 (cpu)                invalid_argument: softmax_ce_loss: C 必须为正（实际 logits 为 [N,0]）
N=0 linear y=[0,3] numel=0 | dW=[3,4] db=[3] dx=[0,4] | conv y numel=0 dW numel=54 dx numel=0
```

### 5. CUDA 错误抛异常且不泄漏设备内存（仓库外 OOM 探针，真跑）

做法：`cudaMemGetInfo` 后一次性占满除 80 MiB 外的显存，调用 `linear_forward_cuda(m=1024x1024, x=[16384,1024])`，让 d_x(64 MiB)、d_w(4 MiB)、d_b 分配成功、d_y(64 MiB) 触发 OOM；在 catch 内、释放 filler 之前再测一次空闲显存。

```console
$ ./oom_raii
caught runtime_error: CUDA error /root/claude-code-backup-20261028/code/uav_rgb/src/linear_cuda.cu:40: out of memory
free0=21569.6 MiB, after filler=79.6 MiB, inside catch=79.6 MiB, final=21569.6 MiB
threw=1, d_x+d_w+d_b(68.0 MiB) 已在 catch 前释放: YES, 全程净泄漏=0.000 MiB
```

说明：`inside catch` 与 `after filler` 相同 → 失败的 `cudaMalloc` 之前已经分配的 68 MiB `DeviceBuf` 在栈展开时全部析构；最终显存回到 `free0`，净泄漏 0.000 MiB。（旧实现 `std::exit` 会跳过这些析构，审核已用 memcheck 证实泄漏。）

### 6. compute-sanitizer memcheck（三个 CUDA 测试）

```console
$ compute-sanitizer --tool memcheck --leak-check full ./src/build/test_linear_cuda
test_linear_cuda: PASS
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
$ compute-sanitizer --tool memcheck --leak-check full ./src/build/test_conv_cuda
test_conv_cuda: PASS
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
$ compute-sanitizer --tool memcheck --leak-check full ./src/build/test_softmax_cuda
test_softmax_cuda: PASS
========= LEAK SUMMARY: 0 bytes leaked in 0 allocations
========= ERROR SUMMARY: 0 errors
```

### 7. 架构确认（只含 sm_75）

```console
$ cuobjdump -lelf src/build/libuavrgb_core.a | grep -oE 'sm_[0-9]+' | sort -u
sm_75
```

### 8. 改动文件行数（硬限：C++ ≤280、CUDA ≤360）

| 文件 | 行数 | 限内 |
| --- | --- | --- |
| src/conv_cpu.cpp | 149 | 是（原 271，拆出 gradcheck） |
| src/conv_gradcheck.cpp（新） | 150 | 是 |
| src/ops_cpu.cpp | 240 | 是 |
| src/linear_cuda.cu | 212 | 是 |
| src/conv_cuda.cu | 293 | 是 |
| src/softmax_cuda.cu | 165 | 是 |
| src/include/uavrgb/ops.hpp | 67 | 是 |
| src/include/uavrgb/conv.hpp | 70 | 是 |
| src/tests/test_gradcheck.cpp | 116 | 是 |
| src/tests/test_linear_cuda.cpp | 144 | 是 |
| src/tests/test_conv_cuda.cpp | 193 | 是 |
| src/tests/test_softmax_cuda.cpp | 107 | 是 |
| src/CMakeLists.txt | 51 | 是（仅加 1 个源文件） |

## 风险与遗留

- **异常化的行为变化**：`CUDA_CHECK` 现在抛 `std::runtime_error`。任何直接调用这些 CUDA 包装且未 catch 的旧代码会终止（terminate），而不是像以前一样 `std::exit`；对训练器是期望行为，但需要调用方补 catch。`nn_cuda.cu` 仍保留 `std::exit`，同一个进程里两条错误处理策略并存，等该模块负责人统一。
- **空 batch 语义未在训练器侧验证**：本仓库还没有训练器；linear/conv 返回空、softmax 抛异常的组合需要未来 trainer/dataset 侧按约定处理（最后一个空 batch 应跳过而不是调用 softmax）。
- **CPU 侧仍无规模上限**：CPU 的 conv/linear 仍把 `int64` 维度截成 `int`，且 `2*pad` 在 CPU 实现里仍可能 int 溢出（审核 CPU #10）。CUDA 侧已拒绝 > INT_MAX，因此 CPU/CUDA 在超大形状上的行为不一致（CPU 可能先静默截断）。当前目标形状离该阈值有 4 个数量级，暂不触发。
- **0 维输入空间的边角**：conv 在 `H==0||W==0` 且 pad 足够大时，host 允许、dW/db 仍计算、dx kernel 跳过；已验证零字节 `cudaMemcpy`（含空指针）在 CUDA 12.8 返回成功，但该形状没有测试用例。
- **W/b 只校验元素数，不校验 dims 布局**：本任务要求的是 `numel()` 不变量；若有人把 W 填成转置布局（如 `[in,out]` 而非 `[out,in]`），numel 相同，仍会静默算错。要抓这类错误需要校验 `dims()`，不在本次范围（审核当时也列为「静默算错」而非必须修）。
- **仓库外临时探针**：`messages`/`oom_raii` 探针在 `/tmp/uavrgb-probe`（临时文件，验证后已删除，本文件保留其真实输出）；仓库内只保留 `src/tests/` 下的负例。
- **性能未改善**：每次调用临时 `cudaMalloc`/`cudaFree` + 全量 `cudaDeviceSynchronize` 的固定开销仍在（CUDA 审核 #7），本库目前只适合做 parity/参考；训练路径不要直接复用这些包装。
- **测试保护仍弱**：gradcheck 不比对生产 conv 前/反向（CPU 审核必须修 #2）、label 越界用例假绿（两份审核的应该修项）、NaN/Inf/随机/非方核覆盖缺失。改动这些生产函数后，CPU 侧测试仍可能抓不到回归，只能靠 CUDA parity（需要 GPU）兜底。
- **未覆盖的 sanitizer**：本轮没重跑 racecheck/synccheck/initcheck；kernel 访存模式未变，上轮 racecheck 为 0 hazard，但不能据此声明本轮没有并发问题。
- **两处共用校验放在 header 的 `detail` 命名空间**：CPU 与 CUDA 靠同一 inline 函数保证语义一致；如果以后有人绕过 `detail::check_*` 直接写新入口，仍会漏校验。
