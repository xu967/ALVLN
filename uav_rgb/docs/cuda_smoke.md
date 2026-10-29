# CUDA 冒烟测试报告 (vector add)

日期: 2026-10-28
机器: Ubuntu 24.04.4 LTS 容器, 1×GeForce RTX 2080 Ti (compute capability 7.5 / sm_75)
Toolkit: CUDA 12.8 (V12.8.93), driver 580.119.02, 内核 5.15.0-119-generic

工程位置: `/root/claude-code-backup-20261028/code/uav_rgb/experiments/cuda_smoke/`
- `vector_add.cu` —— 设备枚举 + 1M 元素向量加法 + 精度校验 + cudaEvent 计时
- `CMakeLists.txt` —— `CMAKE_CUDA_ARCHITECTURES 75`

## 构建与运行

```bash
source /root/claude-code-backup-20261028/code/uav_rgb/env.sh
cd /root/claude-code-backup-20261028/code/uav_rgb/experiments/cuda_smoke
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/cuda_smoke
```

构建输出：

```
-- CMAKE_CUDA_ARCHITECTURES = 75
-- CMAKE_CUDA_COMPILER = /usr/local/cuda-12.8/bin/nvcc
-- Configuring done (2.7s)
-- Generating done (0.0s)
[1/2] Building CUDA object CMakeFiles/cuda_smoke.dir/vector_add.cu.o
[2/2] Linking CUDA executable cuda_smoke
```

## 运行结果

```
=== CUDA device enumeration ===
visible device count: 1
GPU 0: NVIDIA GeForce RTX 2080 Ti
  compute capability : 7.5
  SM count           : 68
  global memory      : 21.49 GiB
  memory clock/bus   : 7000 MHz / 352 bit
  warp size          : 32
CUDA runtime version : 12.8
CUDA driver version  : 13.0
=== vectorAdd (n=1048576, blocks=4096, threads=256) ===
spot check: c[0]=0.000 c[1]=2.500 c[1048575]=2621437.500
expected  :           0.000          2.500              2621437.500
max abs error: 0.000000e+00
avg kernel time (10 iters): 0.0288 ms
RESULT: PASS
```

## 产物架构校验

```bash
cuobjdump -lelf build/cuda_smoke   # ELF file 1: cuda_smoke.1.sm_75.cubin
cuobjdump -lptx build/cuda_smoke   # PTX file 1: cuda_smoke.1.sm_75.ptx
```

可执行文件中只嵌入了 **sm_75** 的 cubin 与 PTX，没有任何 sm_120/其他架构，确认 `CUDA_ARCHITECTURES` 生效。

## 坑：CMake 默认架构不是 75

第一次构建时 `message(STATUS "CMAKE_CUDA_ARCHITECTURES")` 打印出 **52** —— 因为 CMake 3.28 在 `project()` 之后会把 `CMAKE_CUDA_ARCHITECTURES` 初始化为 nvcc 的编译器默认值（CUDA 12.8 下为 52），
函数开头的 `if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` 因此不会被执行。

修正做法：在 `project()` **之前** 强制写缓存变量：

```cmake
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures" FORCE)
project(cuda_smoke LANGUAGES CXX CUDA)
```

结论：本机所有 CUDA 工程的 CMakeLists 都要显式 `CUDA_ARCHITECTURES 75`（或 `set(CMAKE_CUDA_ARCHITECTURES 75)` 放在 project() 之前），
不要依赖 "native"/默认值，更不要照抄文档里的 120。

## 可用/不可用清单

| 项 | 状态 |
|---|---|
| nvcc / cmake / ninja / gcc | 可用（12.8.93 / 3.28.3 / 1.11.1 / 13.3.0） |
| ncu (Nsight Compute)、nsys (Nsight Systems) | 已存在（/usr/local/cuda/bin/ncu、/usr/local/bin/nsys） |
| cuobjdump / nvdisasm | 可用 |
| cuDNN | 未安装 |
| 多卡 | 容器只暴露 1 张 RTX 2080 Ti（/dev/nvidia0），不是 8 张 |

## 更新 2026-10-28（复核：GPU 实为 RTX 2080 Ti，非 T4）

`nvidia-smi -L` 当前实际输出 `NVIDIA GeForce RTX 2080 Ti`（22GB / 68 SM / 352-bit），早前记录为 Tesla T4 系信息陈旧。sm_75/Turing 架构结论不变，冒烟测试重跑仍 PASS：

```
GPU 0: NVIDIA GeForce RTX 2080 Ti
  compute capability : 7.5
  SM count           : 68
  global memory      : 21.49 GiB
  memory clock/bus   : 7000 MHz / 352 bit
max abs error: 0.000000e+00
avg kernel time (10 iters): 0.0288 ms
RESULT: PASS
```

`cuobjdump -lelf/-lptx` 仍只含 `cuda_smoke.1.sm_75.cubin` / `sm_75.ptx`。
