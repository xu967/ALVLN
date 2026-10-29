# RTX 2080 Ti (sm_75, 22 GiB) Conv2d 吞吐 / 显存基准

- 日期：2026-10-28 23:27–23:35（真机实测）
- 设备：NVIDIA GeForce RTX 2080 Ti，sm_75，68 SM，nvidia-smi 报 22528 MiB，cudaMemGetInfo 报 22001 MiB 可用；FP32，driver 580.119.02，CUDA 12.8
- 被测实现：`src/conv_cuda.cu` 的 parity 版包装（只读链接 `src/build/libuavrgb_core.a`，未改 `src/`）
- 程序：`experiments/gpu_bench/gpu_bench.cu`；原始输出：`results_run1.txt`、`results_run2.txt`；kernel 计时：`kernel_times_nsys.csv`
- 形状：NCHW fp32，输入 128×128，3×3，pad=1，stride=1；层 3→16、16→32、32→64；batch 1/8/32/128/256

## 0. 结论（TL;DR）

1. **原始包装吞吐（真跑）**：batch 256 时，3 层网络前向 1.85 s（138 samples/s），反向 3.49 s（73 samples/s），训练一步（fwd+bwd）约 5.35 s（≈48 samples/s）。
2. **Kernel 本身（nsys 实测）**：前向 0.76–0.95 TFLOP/s，反向仅 0.16–0.29 TFLOP/s（dW kernel 是瓶颈）。3 层 batch 256 的 kernel-only 一步约 2.16 s（≈119 samples/s），约为原始包装的 2.5 倍。
3. **偏差来源**：每次调用 4/6 次 `cudaMalloc/cudaFree`、每个 kernel 后 `cudaDeviceSynchronize`、host 侧 `Tensor::zeros` 清零（本机仅 ~1.6 GB/s，page-fault 主导）、全量 pageable H2D/D2H。batch 256 的 32→64 前向中，host 清零 ≈610 ms、拷贝 ≈259 ms、kernel ≈169 ms（合计 ≈1008 ms 实测）。
4. **显存不是瓶颈（128×128）**：3 层网络训练每样本 ≈17.4 MiB（激活+梯度+dx 工作区），22 GiB 约可放 **batch ≈1200**；单层 32→64 反向约可放 **2632**。**时间才是瓶颈**。
5. **建议**：128×128 输入用 **batch 128–256**（batch 256 训练显存 ≈4.4 GiB）；256×256 输入（×4）用 **batch 64–128**（显存上限约 303）。要放大模型/实时训练，先优化反向 dW kernel 并把激活留在显存。

## 1. 构建与 sm_75 验证（真实输出）

```bash
source "$UAV_RGB_ROOT/env.sh"
cmake -S experiments/gpu_bench -B experiments/gpu_bench/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build experiments/gpu_bench/build
# -- CMAKE_CUDA_ARCHITECTURES = 75
# [2/2] Linking CUDA executable gpu_bench

cuobjdump -lelf experiments/gpu_bench/build/gpu_bench | grep -oE 'sm_[0-9]+' | sort -u
# sm_75
cuobjdump -lptx experiments/gpu_bench/build/gpu_bench | grep -oE 'sm_[0-9]+' | sort -u
# sm_75
```

`CMakeLists.txt` 在 `project()` 之前 `set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "" FORCE)`，且 target 再设 `CUDA_ARCHITECTURES 75`；ELF/PTX 均只有 sm_75。程序开头自检 CPU 参考 vs CUDA 前向：`max_abs_err=8.941e-08 PASS`，结尾 `RESULT: PASS`。

nvidia-smi（基准之前，2026/10/28 23:25:05）：

```
NVIDIA GeForce RTX 2080 Ti, 22528 MiB, 270 MiB used, 21732 MiB free, 11 %, compute_cap 7.5
```

`nvidia-smi --query-compute-apps`：仅 `selkies-venv/python 266 MiB`（桌面串流），无其他 CUDA 计算进程。

## 2. 测量方法与偏差说明（必须读）

现有包装（`src/conv_cuda.cu`）每次调用的真实行为：

- 前向：4 次 `cudaMalloc`（x/W/b/y）+ 3 次 H2D + 1 个 kernel + 1 次 `cudaDeviceSynchronize` + 1 次 D2H + 4 次 `cudaFree`；输出 host `Tensor::zeros` 先清零。
- 反向：6 次 `cudaMalloc`（x/W/dy/dW/db/dx）+ 3 次 H2D + **3 个 kernel，每个后都同步** + 3 次 D2H + 6 次 `cudaFree`；三个输出 host Tensor 都先清零。

因此**原始数字是该包装的真实端到端吞吐，不是 kernel 上限**，会显著偏低。本基准分两层给出：

- **实测**：`std::chrono` 包住完整包装调用，1 次暖机 + 5 次计时，表中给 mean/min；显存用解析精确值（同一次调用内同时存活的设备缓冲之和）与 200 µs `cudaMemGetInfo` 采样值互证。
- **~EST（估算）**：`kernel-only` 是 kernel 的 GPU 侧真实耗时（nsys 实测，见 §4）；`opt` = kernel 实测 + 单独实测的 pageable H2D/D2H 分量，即「复用缓冲、去掉 host 清零和每 kernel 同步，但仍保留每次调用搬数据」。`kernel-only` 列进一步假设数据常驻显存、层间零拷贝，是上界。
- 程序内还有一版“总时间 − 各分量”的估算并打印在 stdout；当 kernel 只占总量很小比例（小层/小 batch）时，分量相减误差可达 100%，**那些格子不要引用**；本节表格用 nsys 实测 kernel 时间替代，可信得多。

环境噪声：机器上并行跑着其他 agent（load ≈8/56 核）。两次完整运行中，host 侧 page-fault/拷页抖动使小层数字差异最多约 20%（例如 16→32 N=256 前向 482 vs 579 ms），但 nsys kernel 时间随 batch 严格线性（±2%）。本文表格采用 **run2**（内部一致）；run1 用于交叉检查。

## 3. 原始包装吞吐（实测，run2）

1 次暖机 + 5 次计时；`fwd/bwd ms` 为 mean / min。`samp/s` 由 mean 计算；`TF` 是包含全部包装开销的有效算力。

| layer | N | fwd ms (mean/min) | fwd samp/s | fwd TF | bwd ms (mean/min) | bwd samp/s | bwd TF | peak dev fwd/bwd MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 3→16 | 1 | 1.1 / 1.0 | 925 | 0.01 | 3.1 / 2.7 | 320 | 0.01 | 1.19 / 1.38 |
| 3→16 | 8 | 4.5 / 4.2 | 1798 | 0.03 | 22.0 / 20.9 | 364 | 0.01 | 9.50 / 11.00 |
| 3→16 | 32 | 31.3 / 30.8 | 1022 | 0.01 | 69.8 / 63.2 | 458 | 0.01 | 38.00 / 44.00 |
| 3→16 | 128 | 117.2 / 113.6 | 1093 | 0.01 | 257.2 / 248.2 | 498 | 0.01 | 152.00 / 176.00 |
| 3→16 | 256 | 216.0 / 208.9 | 1185 | 0.02 | 573.0 / 530.0 | 447 | 0.01 | 304.00 / 352.00 |
| 16→32 | 1 | 2.4 / 1.5 | 410 | 0.06 | 3.8 / 3.3 | 266 | 0.08 | 3.02 / 4.04 |
| 16→32 | 8 | 13.9 / 10.3 | 574 | 0.09 | 29.6 / 27.3 | 270 | 0.08 | 24.02 / 32.04 |
| 16→32 | 32 | 76.0 / 67.3 | 421 | 0.06 | 132.4 / 122.6 | 242 | 0.07 | 96.02 / 128.04 |
| 16→32 | 128 | 286.5 / 272.3 | 447 | 0.07 | 502.3 / 481.7 | 255 | 0.08 | 384.02 / 512.04 |
| 16→32 | 256 | 579.3 / 521.2 | 442 | 0.07 | 984.3 / 952.4 | 260 | 0.08 | 768.02 / 1024.04 |
| 32→64 | 1 | 4.2 / 3.6 | 239 | 0.14 | 8.0 / 7.2 | 125 | 0.15 | 6.07 / 8.14 |
| 32→64 | 8 | 37.2 / 35.0 | 215 | 0.13 | 58.8 / 56.3 | 136 | 0.16 | 48.07 / 64.14 |
| 32→64 | 32 | 138.0 / 132.0 | 232 | 0.14 | 243.9 / 231.5 | 131 | 0.16 | 192.07 / 256.14 |
| 32→64 | 128 | 546.3 / 520.9 | 234 | 0.14 | 932.7 / 922.8 | 137 | 0.17 | 768.07 / 1024.14 |
| 32→64 | 256 | 1056.0 / 1008.2 | 242 | 0.15 | 1937.5 / 1885.6 | 132 | 0.16 | 1536.07 / 2048.14 |

注意吞吐随 batch **几乎平坦**（每样本成本近似常数），说明开销是按 batch 线性增长的搬运/清零，而不是固定启动开销。**22 GiB 卡上把这套包装直接用于训练，batch 加大不会提升吞吐**。

## 4. Kernel 实测耗时（nsys）与优化空间估算

ncu 因 `ERR_NVGPUCTRPERM`（无 GPU 性能计数器权限）不可用；改用 **nsys CUPTI 活动跟踪**，读取每个 kernel 的 GPU 侧 duration（不改变代码，不影响 kernel 时间）。15 个配置 × (暖机+5 次计时) 共 91 个前向、各 90 个 dW/db/dx kernel，按 grid 大小与顺序逐一对齐验证（例如 32→64 N=256 前向 grid=1,048,576，与 `N·Cout·128·128/256` 一致）。原始数据：`kernel_times_nsys.csv`。

上表为 kernel mean 耗时与对应算力。**前向约 0.76–0.95 TFLOP/s，反向约 0.16–0.29 TFLOP/s**（2080 Ti FP32 峰值 13.4 TFLOP/s）——反向 dW/dx 直接卷积的实现效率远低于前向。

估算（~EST，非重新实现的实测；kernel 为 nsys 实测，拷贝为单独实测的 pageable 分量）：

- `opt`：复用设备缓冲 + 去掉 host 清零 + 去掉每 kernel 同步，但每次调用仍做全量 H2D/D2H。
- `kernel-only`：数据常驻显存、层间零拷贝的上界。

| layer | N | kernel fwd ms | kernel bwd ms | kernel fwd TF | kernel bwd TF | ~EST opt fwd samp/s | ~EST opt bwd samp/s | ~EST kernel-only fwd samp/s | ~EST kernel-only bwd samp/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3→16 | 1 | 0.03 | 1.61 | 0.46 | 0.02 | 1783 | 519 | 32376 | 622 |
| 3→16 | 8 | 0.21 | 15.49 | 0.55 | 0.01 | 4798 | 419 | 38567 | 516 |
| 3→16 | 32 | 0.59 | 48.09 | 0.77 | 0.02 | 6236 | 538 | 54104 | 666 |
| 3→16 | 128 | 2.35 | 191.15 | 0.77 | 0.02 | 5431 | 534 | 54546 | 670 |
| 3→16 | 256 | 4.79 | 383.08 | 0.76 | 0.02 | 5234 | 547 | 53489 | 668 |
| 16→32 | 1 | 0.17 | 1.61 | 0.89 | 0.19 | 1353 | 424 | 5910 | 622 |
| 16→32 | 8 | 1.29 | 14.73 | 0.94 | 0.16 | 1493 | 360 | 6212 | 543 |
| 16→32 | 32 | 5.12 | 59.05 | 0.94 | 0.16 | 1690 | 339 | 6252 | 542 |
| 16→32 | 128 | 20.83 | 236.23 | 0.93 | 0.16 | 1263 | 329 | 6146 | 542 |
| 16→32 | 256 | 42.04 | 473.12 | 0.92 | 0.16 | 1066 | 313 | 6089 | 541 |
| 32→64 | 1 | 0.64 | 4.09 | 0.95 | 0.30 | 397 | 172 | 1571 | 244 |
| 32→64 | 8 | 5.09 | 32.91 | 0.95 | 0.29 | 565 | 159 | 1572 | 243 |
| 32→64 | 32 | 20.27 | 131.86 | 0.95 | 0.29 | 540 | 165 | 1578 | 243 |
| 32→64 | 128 | 83.46 | 529.60 | 0.93 | 0.29 | 540 | 160 | 1534 | 242 |
| 32→64 | 256 | 169.40 | 1083.58 | 0.91 | 0.29 | 597 | 166 | 1511 | 236 |

以最大的 32→64、batch 256 为例：原始包装前向 1056 ms → 去 malloc/同步（仍拷贝）约 597 samples/s（≈2.5×）→ 数据常驻 kernel 上界 1511 samples/s（≈6.2×）；反向 1938 ms → 166 samples/s（≈1.25×）→ 236 samples/s（≈1.8×）。**反向受 kernel 本身限制，去同步帮助有限。**

### 整网 3 层（128×128）不同口径的一步耗时

| N | raw fwd ms (samp/s) | raw bwd ms (samp/s) | raw 一步 samp/s | ~EST opt 一步 samp/s | ~EST kernel-only 一步 samp/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 7.7 (130) | 14.9 (67) | 44 | 72 | 123 |
| 8 | 55.6 (144) | 110.4 (73) | 48 | 71 | 115 |
| 32 | 245.3 (130) | 446.1 (72) | 46 | 74 | 121 |
| 128 | 950.0 (135) | 1692.2 (76) | 48 | 72 | 120 |
| 256 | 1851.4 (138) | 3494.9 (73) | 48 | 72 | 119 |

“一步”= 前向 + 反向（不含优化器）。可以看到：原始包装约 48 samples/s，去掉包装开销但保留层间搬运约 72 samples/s（+50%），完全 device-resident 约 119 samples/s（+148%）。反向 kernel 占 kernel-only 一步的 90%，是首要优化对象。

## 5. 显存占用与 22 GiB 可容纳的 batch

预算：启动时 `cudaMemGetInfo` free = 21576 MiB，扣 512 MiB 余量 → **21064 MiB** 可用于本进程。峰值 = 同一次调用内同时存活的设备缓冲精确字节数；`cudaMemGetInfo` 200 µs 采样在 run1 中与解析值吻合（例如 32→64 N=256 反向采样 2050 MiB vs 解析 2048.14 MiB；整程序 GPU used 峰值 2476 MiB ≈ 2048 + 启动基线 426）。

| layer | peak fwd @N=1 MiB | fwd bytes/sample | max batch fwd | peak bwd @N=1 MiB | bwd bytes/sample | max batch bwd |
|---|---:|---:|---:|---:|---:|---:|
| 3→16 | 1.19 | 1,245,184 (1.19 MiB) | 17737 | 1.38 | 1,441,792 (1.38 MiB) | 15318 |
| 16→32 | 3.02 | 3,145,728 (3.00 MiB) | 7021 | 4.04 | 4,194,304 (4.00 MiB) | 5265 |
| 32→64 | 6.07 | 6,291,456 (6.00 MiB) | 3510 | 8.14 | 8,388,608 (8.00 MiB) | 2632 |

`max_batch` 为单层独占显存、只跑一次前向/反向的上限。全模型训练还需保存各层激活与梯度：

- 3 层 128×128：激活 115 ch = **7.19 MiB/样本**，梯度 112 ch = **7.00 MiB/样本**，dx 工作区 51 ch = **3.19 MiB/样本**，合计 **17.38 MiB/样本**；权重合计 <0.1 MiB。
- 全模型训练显存上限 ≈ 21064 / 17.38 ≈ **1212 samples**；推理（保留全部层激活）≈ **2930**；单层推理峰值（最大 96 ch）≈ **3510**。
- batch 256 时：训练激活+梯度+工作区 ≈ 4.45 GiB，远低于 22 GiB；实测整程序 GPU used 峰值 2476–2506 MiB（含 426 MiB 基线）。
- 输入 256×256 时每样本 ×4：训练 **69.5 MiB/样本** → 显存上限约 **303**。

**结论：128×128 下 22 GiB 根本不是限制，batch 可以开到上千；限制的是时间。** 若要走 256×256，显存上限降到 ~300，实际建议 64–128。

## 6. 运行中 nvidia-smi 采样（真实）

采样方式：`nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free --format=csv,noheader` 每 0.2 s 一次，整个基准运行期间后台记录。

| 运行 | 时长/采样数 | GPU util 均值 / 中位 / p95 / 峰值 | ≥90% 采样 | GPU used 峰值 |
|---|---|---:|---:|---:|
| run1（23:27:53–23:28:57） | 64 s / 297 | 55.5% / 58% / 91% / **98%** | 16 次 | 2476 MiB |
| run2（23:30:43–23:31:52） | 69 s / 282 | 49.7% / 48% / 77% / **85%** | 0 次 | 2506 MiB |

峰值 85–98% 说明 kernel 阶段确实把 GPU 用满；平均只有约 50%，因为包装中有大量 host 侧清零、pageable 拷贝、`cudaMalloc/Free` 阶段 GPU 空闲——这与 §3 的分解一致。两次运行的峰值差异来自 0.2 s 采样窗口与短 kernel 的对齐，不代表算力变化。

## 7. 22 GiB 下的建议（结论）

1. **batch size**
   - 128×128、3 层 CNN：**128–256**。显存可到 ~1200，但 batch 256 已使原始包装一步 5.35 s、kernel-only 一步 2.16 s；再大只会线性变慢且毫无吞吐收益（吞吐对 batch 平坦）。
   - 256×256：**64–128**（显存上限 ~303，时间约 ×4）。
   - 推理：batch 1 的 kernel-only 前向 0.84 ms（3 层合计）→ 上限 ~1200 samples/s；原始包装 batch 1 前向 7.7 ms（~130 samples/s）。实时性必须靠 device-resident 管线。
2. **模型规模**：22 GiB 对 128×128 的 3–5 层、≤128 通道小 CNN 非常宽裕（batch 256 训练 <5 GiB）。按 `max_batch ≈ 20.6 GiB / (2.4 × A)` 估算（A = Σ C_out·H·W·4 B/样本；2.4 是激活+梯度+工作区的经验系数）：A=7.2 MiB → ~1200；A=29 MiB → ~300；A=72 MiB（约 10× 本网络）→ ~120；A=230 MiB → ~38。想做 ResNet 级模型/224×224，batch 只能到几十，且本套 naive kernel 的算力（<1 TFLOP/s）会成为主要瓶颈。
3. **优化优先级**：a) 重写反向 dW kernel（当前 0.16–0.29 TFLOP/s，占 kernel-only 训练时间 90%）；b) 设备缓冲复用、去掉每 kernel 同步（fwd 提升约 2.5×）；c) 整网激活常驻显存、层间不搬 host（再提升约 1.6×）；d) 输入增大前先算显存（×4/×16）。

## 8. 复现命令

```bash
source /root/claude-code-backup-20261028/code/uav_rgb/env.sh
cmake -S experiments/gpu_bench -B experiments/gpu_bench/build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build experiments/gpu_bench/build
cd experiments/gpu_bench

# 基准本体（参数 = 计时迭代数；最后打印 RESULT: PASS）
./build/gpu_bench 5 | tee results_run2.txt

# 运行中 GPU 采样
( while :; do nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.free \
    --format=csv,noheader; sleep 0.2; done ) > /tmp/smi_gpu_run.csv &
SMI=$!; ./build/gpu_bench 5 > /dev/null; kill $SMI

# kernel-only 实测（ncu 无权限，用 nsys CUPTI 活动跟踪）
nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o /tmp/gpu_bench_nsys ./build/gpu_bench 5
nsys stats --report cuda_gpu_trace --format csv --force-export=true \
  /tmp/gpu_bench_nsys.nsys-rep > /tmp/gpu_trace.csv
```

## 9. 没做 / 局限（如实）

- **只测了 parity 版 Conv2d 包装**；linear/softmax 未测（本次任务只要求 Conv2d）。
- 只测 FP32、3×3、pad=1、stride=1、128×128；未测 FP16/Tensor Core、其他 kernel 尺寸、stride/padding 变体。
- **没有写优化版 kernel**：`opt` / `kernel-only` 是估算（kernel 时间由 nsys 实测，拷贝分量单独实测），不是重新实现的实测吞吐；实际优化版还会受 tiling/L2/occupancy 影响。
- 程序内“总量 − 分量”的逐格估算在小 kernel 格子里噪声大（甚至为 0），本文的 §4 表已用 nsys 实测替换 kernel 时间；stdout 中那些格子仅作诊断。
- ncu 不可用（`ERR_NVGPUCTRPERM`，未改系统权限）；nsys 的 kernel duration 是 GPU 侧时间，可信。
- 机器与并行 agent 共享：两次运行 host 侧差异最多约 20%（小层），表中为 run2；GPU util/显存峰值两次分别为 98%/2476 MiB 与 85%/2506 MiB。
- 显存上限是按“单进程独占、扣 512 MiB 余量”估算；未考虑显存碎片、CUDA context 增长、其他进程占用。
