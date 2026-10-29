# E5（数值内核）：ReLU / MaxPool2d / Adam + global-norm 梯度裁剪

- 时间：2026-10-28 23:25
- 范围：`src/`（新建 `nn.hpp` / `nn_cpu.cpp` / `nn_cuda.cu` / `test_nn_cuda.cpp`，并在 `src/CMakeLists.txt` 注册）
- 结论：任务 1/2/3 与可选项 4（梯度裁剪，二选一）全部完成，RTX 2080 Ti 真编译、真跑、数值为实测。

## 做了什么

- `src/include/uavrgb/nn.hpp`（79 行）：ReLU / MaxPool2d / Adam / 梯度裁剪的 CPU 与 CUDA 声明，
  并写死数值约定：
  - ReLU：`y = max(x,0)`，`dx = dy * (x > 0)`；**x == 0 处取次梯度 0**，CPU/CUDA 两处实现一致。
  - MaxPool2d：方形窗口 k×k、步长 s（本轮 k=s=2），NCHW，`H'=(H-k)/s+1` 向下取整；
    前向记录 argmax（单通道 flatten 下标 `ih*W+iw`，float 存储，<2^24 精确）；
    并列时按 `(ki,kj)` 行优先扫描、严格 `>` 才更新，即取第一个最大值，CPU/CUDA 一致。
  - Adam：`t` 为已完成 step 数，本次用 `t'=t+1` 做偏差校正；m/v/t 必须随 param 一起 checkpoint。
- `src/nn_cpu.cpp`（250 行）：CPU FP32 参考实现 `relu_forward/backward`、`maxpool2d_forward/backward`、
  `adam_make/adam_step`；FP64 中心差分 `gradcheck_relu`（loss=Σrelu(x)²）、
  `gradcheck_maxpool2d`（loss=Σy²，解析梯度 dy=2y 散回 argmax）；`global_grad_norm` /
  `clip_grad_norm`（coef = min(1, max_norm/(norm+1e-6))）。
- `src/nn_cuda.cu`（312 行）：手写 kernel `relu_forward/backward`、`maxpool_forward/backward`
  （反向用 float atomicAdd 把 dy 散回 argmax，重叠窗口也正确）、`adam_step`（每线程一元素原地更新）、
  `sumsq`（double atomicAdd 归约）+ `scale`；沿用 `conv_cuda.cu` 的 DeviceBuf RAII + CUDA_CHECK +
  每 kernel 后 `cudaGetLastError`+同步模板。argmax 直接作 GPU 写偏移，故在 host 先校验范围与整数性。
- `src/tests/test_nn_cuda.cpp`（216 行）：ReLU 前反向 parity（含精确 0 与次梯度约定自检）、
  MaxPool y/argmax/dx parity（H=5、W=7 奇数验证向下取整）、Adam 同初值跑 5 步逐步比较
  param/m/v、Adam 闭式自检（beta1=beta2=0 时一步 `p -= lr*sign(g)`）、两个 FP64 gradcheck、
  梯度裁剪的 CPU/CUDA parity + 独立 double 重算公式校验、形状异常自检。
- `src/CMakeLists.txt`：`nn_cpu.cpp` / `nn_cuda.cu` 加入 `uavrgb_core`，注册 `test_nn_cuda`；
  sm_75 FORCE 语句未动。

## 没做什么（含原因）

- **没做学习率调度**：可选项 4 是「梯度裁剪或学习率调度，任选其一」，选了前者；
  后者没有任务给定的公式形状，自造一个反而无法验证。
- **没做 MaxPool stride>1/padding/非方形窗口**：任务只要求 kernel 2、stride 2；
  结构支持任意 k=s 的方形窗口，非方形未做。
- **没做 CUDA 持久显存 / 异步流水**：延续现有 parity 自检模式，每次调用临时 cudaMalloc/cudaFree
  且每 kernel 同步；仅供正确性对拍，训练吞吐不是本轮目标。
- **没做 NaN/Inf 专项测试**：doc 04 §6 提到，但不在本任务自检清单内；当前行为是 NaN 直接传播，
  MaxPool 中 NaN 参与比较恒为 false（未验证，列在风险）。
- 没碰 `sim/`、`ui/`、`tools/`、`experiments/`、`docs/`（除本 worklog），
  未改 `src/include/uavrgb/event.hpp`，未改 sm_75 FORCE 语句。
- 未引入 PyTorch/TensorRT/cuDNN/cuBLAS，全部纯 C++/CUDA 手写。

## 验证到哪一步（真实命令 + 输出）

```bash
source /root/claude-code-backup-20261028/code/uav_rgb/env.sh
cmake --build /root/claude-code-backup-20261028/code/uav_rgb/src/build   # 11/11，无告警
ctest --test-dir /root/claude-code-backup-20261028/code/uav_rgb/src/build --output-on-failure
# 1/7..7/7 全部 Passed；100% tests passed, 0 tests failed out of 7
/root/claude-code-backup-20261028/code/uav_rgb/src/build/test_nn_cuda   # exit=0
cuobjdump -lelf /root/claude-code-backup-20261028/code/uav_rgb/src/build/libuavrgb_core.a \
  | grep -oE 'sm_[0-9]+' | sort -u                                      # 仅 sm_75
wc -l src/nn_cpu.cpp src/nn_cuda.cu src/include/uavrgb/nn.hpp src/tests/test_nn_cuda.cpp
# 250 / 312 / 79 / 216（C++ ≤280、CUDA ≤360，均未超）
```

`test_nn_cuda` 实测输出（RTX 2080 Ti，非推算）：

```text
relu parity        max|y_cpu-y_cuda|  = 0
                   max|dx_cpu-dx_cuda|= 0
maxpool parity     max|y_cpu-y_cuda|    = 0
                   max|arg_cpu-arg_cuda|= 0
                   max|dx_cpu-dx_cuda|  = 0
adam step 1        max|p|=0 max|m|=0 max|v|=0
adam step 2        max|p|=9.31323e-10 max|m|=9.31323e-10 max|v|=3.63798e-12
adam step 3        max|p|=1.86265e-09 max|m|=1.86265e-09 max|v|=3.63798e-12
adam step 4        max|p|=7.45058e-09 max|m|=2.79397e-09 max|v|=3.63798e-12
adam step 5        max|p|=1.49012e-08 max|m|=1.86265e-09 max|v|=7.27596e-12
relu gradcheck_max_err    = 5.00044e-07
maxpool gradcheck_max_err = 7.81526e-11
clip norm parity   |n_cpu-n_cuda|=0 coef cpu/cuda/ref=0.762492/0.762492/0.762492
                   max|a_cpu-a_cuda|=0 max|b_cpu-b_cuda|=0
test_nn_cuda: PASS
```

| 项 | 最大绝对误差 | 门槛 |
| --- | --- | --- |
| ReLU 前向 / 反向 CPU/CUDA parity | 0 / 0 | <1e-4 |
| MaxPool 前向 y / argmax / dx parity | 0 / 0 / 0 | <1e-4 |
| Adam 5 步 param（最差第 5 步） | 1.49012e-08 | <1e-4 |
| Adam 5 步 m（最差第 4 步） | 2.79397e-09 | <1e-4 |
| Adam 5 步 v（最差第 5 步） | 7.27596e-12 | <1e-4 |
| ReLU FP64 中心差分 gradcheck | 5.00044e-07 | <1e-5 |
| MaxPool FP64 中心差分 gradcheck | 7.81526e-11 | <1e-5 |
| 梯度裁剪 norm / 缩放后张量 parity | 0 / 0 | <1e-4 |

说明：ReLU gradcheck 的 5.0e-07 来自 x==0 处二阶不可导点的中心差分截断（`(eps²-0)/(2eps)=eps/2`，
eps=1e-6），不是实现误差；x>0 部分为二次函数，中心差分精确。梯度裁剪的 coef 与独立 double 重算
一致（0.762492），且测试值 norm≈0.656 > max_norm=0.5，确认裁剪分支被触发。

## 风险与遗留

- CUDA 包装每次调用临时分配 + 每 kernel 同步，训练吞吐会明显偏低；接训练器时应改持久显存/流水。
- argmax 用 float 存下标，仅保证 `<2^24` 时精确；当前 224×384=86016，安全。
- MaxPool 反向用 atomicAdd：本轮 k=s=2 窗口不重叠、无竞争；若将来 stride<k，求和顺序不确定
  （float 结果会有末位波动）。
- Adam 偏差校正因子在 host 用 double 算后转 float 传 CUDA，CPU/CUDA 一致；checkpoint 必须原样保存 t。
- NaN/Inf 未做专门测试：NaN 会在各算子中传播，MaxPool 的 `>` 比较对 NaN 恒 false，行为未验证。
- 自检过程中发现并修正了一个测试输入 bug（`0.2f * float(size_t i - 2)` 的无符号下溢会把值放大到
  1e19），说明独立 double 公式复算是必要的；最终数字均来自修正后的运行。
