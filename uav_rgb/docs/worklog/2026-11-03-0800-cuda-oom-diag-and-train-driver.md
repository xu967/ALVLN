# CUDA 显存 OOM 报错补证据 + 学生训练驱动入库

- 时间：2026-11-03 08:00
- 范围：`src/*_cuda.cu`、`src/CMakeLists.txt`、`tools/train_students.sh`
- 结论：完成（代码改动已构建 + CTest 24/24 通过；驱动脚本用桩二进制自测过）
- 依据：`docs/worklog/2026-11-03-distill-round-retrospective.md` 的 §2.C 与 §4.1/4.3/4.5

## 做了什么

- `src/include/uavrgb/cuda_alloc.hpp`（新增）：`throw_alloc_failure(bytes, file, line)`
  把 `requested / free / total` 和 `file:line` 写进异常。先 `cudaGetLastError()` 清掉
  sticky error，否则 `cudaMemGetInfo` 会直接返回上一次的失败。
- `src/nn_cuda.cu`、`conv_cuda.cu`、`linear_cuda.cu`、`softmax_cuda.cu`：四份
  `DeviceBuf` 的 `cudaMalloc` 都改走这个函数（原来是 `CUDA_CHECK`，只报 `out of memory`）。
  四份都改是因为 OOM 可能从任意一个分配点抛出，只改 nn_cuda 会漏。
- `src/tests/test_cuda_alloc_errors.cpp`（新增）+ `src/CMakeLists.txt` 注册：
  断言报错含 `file:line`、请求字节数、`现可用/共`；再用 1 TiB 真实分配触发失败，
  验证失败后仍查得到显存（即 sticky error 确实被清掉了）。
- `tools/train_students.sh`（新增）：参数化学生训练驱动。组名 `base` / `a<alpha>`；
  显存预检（>1024 MiB 最多等 10 min 后放弃）；单组失败不中断其余组（对比旧脚本
  `set -eu` 让 a50 失败拖死 a80）；每阶段状态追加进 `<out>/run_status.json`；
  失败时自动 dump 最后 5 行日志 + `nvidia-smi --query-compute-apps`。

## 没做什么

- **没有重跑本轮 1000 集训练**。`student-a50` 07:30 已在跑（旧驱动），不动它；
  新驱动留给下一次实验。原因：重跑要 5 h，且 A/B 条件应保持一致。
- **没有加 per-process 的周期性显存采样**（复盘 §4.5 的一半）。原因：本次现场已用
  一次性 `nvidia-smi` 循环覆盖（`/tmp/gpu_mem_a50.log`），把它固化进驱动需要额外
  设计，超出本次「最小改动」范围；待下一次出现第二个 GPU 任务并存时再做。
- **没有把评测驱动（`eval_offline` / `eval_closed_loop`）入库**。原因：当前各跑一次
  的 `/tmp` 脚本还在用，收进仓库需要先统一参数口径，单独一次做。
- **没有改 `docs/EVAL_STANDARD.md` 的统计门槛（复盘 §4.4）**。原因：属于口径文件，
  需要用户拍板；本轮闭环对比仍按旧规则做，只在实验文档里标 `exploratory`。
- 没有验证「4 个 DeviceBuf 之外还有没有别的裸 `cudaMalloc`」。原因：`grep cudaMalloc`
  的结果里其余都是测试注入点或已带检查的路径，风险低；若下次 OOM 报错仍缺字段，
  优先查这里。

## 待确认

- `tools/train_students.sh` 是否该入库。`personal-skill` §3 说「不要为一次实验新建
  脚本并保留」，这条脚本是按复盘 §4.1 做的**参数化复用入口**（无硬编码路径、无数据/权重），
  我判断不属于「一次性脚本」，但仍需用户确认；若要删，删它不影响上面三项 CUDA 改动。
- 现有实验用 `--seed 1234`，与 `personal-skill`「固定 `seed=0`」的硬规则不一致。
  三组用的是**同一个** seed，A/B 内部可比；改 seed 需要三组全部重跑（约 5 h）。
  本轮不擅自改，先记在这里。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `cmake --build /tmp/build-diag`（CUDA 改动后全量） | 通过 |
| `ctest --test-dir /tmp/build-diag` | **24/24 通过**，118.89 s |
| `/tmp/build-diag/test_cuda_alloc_errors` | 通过（含 1 TiB 真失败路径） |
| `tools/train_students.sh` 桩二进制自测（base 成功 / a0.5 退出 7 / a0.8 继续） | 通过，`run_status.json` 三条记录正确，脚本 exit 1 |
| `student-a50` 现有训练是否受影响 | 未受影响：它在 07:30 已加载旧二进制，新构建在 `/tmp/build-diag`，未覆盖 `/tmp/build-soft/uavrgb_train` |

## 风险与遗留

- 新报错格式只在**下一次真的 OOM 时**才会被看到；现有训练仍用旧二进制。
- `tools/train_students.sh` 假定 `UAV_RGB_BUILD_DIR`（默认 `build/check`）里有
  `uavrgb_train`；如果用户平时用 `/tmp/build-*`，需要显式设这个变量，否则脚本直接退出 2。
- 显存预检阈值 1024 MiB 是经验值：本机训练常驻 250–430 MiB，留了余量；
  若以后模型变大，这个阈值要跟着调。
