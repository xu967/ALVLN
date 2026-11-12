# 卡 C1a：训练器的小修与数值测试

- 时间：2026-11-12 07:08
- 范围：`src/` 训练器与评估入口、checkpoint 元信息读取、CUDA 同步标志及新增 CTest 单测（分支 `p12-c1a`）
- 结论：完成

## 做了什么

- D3（提交 `249d37f`）：新增 `src/tests/test_lr_adamw.cpp` 并在 `src/CMakeLists.txt` 登记，手算并核对线性 warmup、余弦退火、纯余弦（卡 E2E-R1 口径）逐步学习率，以及 `dev::adam_step` 两步三类分量（零梯度区分 AdamW 与 L2 正则、正负翻转梯度、负参数负梯度）和 `DeviceCnn::backward_step` 全模型 12 个参数段的零梯度权重衰减接线。
- H8（提交 `4ea91cd`）：修改 `src/include/uavrgb/trainer.hpp` 与 `src/trainer.cpp`，将增强随机数发生器 `aug_rng_` 改为按 `cfg_.seed ^ 0x9e3779b9u` 播种，并在 `Trainer::run()` 的 `merge_resume_config` 之后按继承后的最终 `seed` 重新播种；新增 `src/tests/test_augment_seed.cpp` 隔离验证 `flip`、`photo`、`shift` 在不同种子下产生互异增强流且续训继承种子生效。
- 纯移动（提交 `97a328d`）：把 `src/train_main.cpp` 的 `--eval-only` 执行体原样移入 `src/trainer_eval.cpp` 的 `run_eval_only_cli`（`src/include/uavrgb/trainer.hpp` 声明），使 `train_main.cpp` 从 279 行降至 225 行，为后续开关留出 280 行预算且保留 CLI 分支供流水线测试静态扫描。
- H9 / 上一轮 D2（提交 `232bac4`）：在 `src/include/uavrgb/trainer.hpp`、`src/train_main.cpp`、`src/trainer.cpp` 新增 `--log-grad-norm`（默认关闭），开启时每轮向 `metrics.jsonl` 追加一条状态记录 `grad_norm epoch <E>: median=<...> max=<...> clip_frac=<...>`（裁剪前全局梯度范数的中位数、最大值及 `max_grad_norm / (gn + 1e-6f) < 1` 的步数占比）；新增 `src/tests/test_log_grad_norm.cpp`。
- D6（提交 `45bb34c`）：在 `src/include/uavrgb/trainer.hpp`、`src/train_main.cpp`、`src/class_weights.cpp`、`src/trainer.cpp` 补齐 `max_grad_norm_set`、`RunConfigSnapshot::max_grad_norm` 与 `merge_resume_config` 继承及非正数校验；新增 `src/tests/test_resume_grad_norm.cpp`。
- C6（提交 `e7ee56d`）：清查全仓库与在途工作树的 `--eval-only` 调用方后，在 `src/include/uavrgb/checkpoint.hpp`、`src/checkpoint.cpp` 让 `read_checkpoint_meta` 解析已有的 v5 113 B 训练契约（不改动磁盘 checkpoint 二进制格式），在 `src/trainer_eval.cpp` 让 `--eval-only` 依次读取同目录 `config.json` 与 checkpoint 契约的 `drop_collision`，两处矛盾、与 CLI `--drop-collision` 冲突、或两处均无记录时直接 `rc=2` 报错，不再静默回落；新增 `src/tests/test_eval_only_drop_collision.cpp`。
- D9（提交 `f7e80b5`）：新增 `src/tests/test_conv_dx_gradcheck.cpp` 并在 `src/CMakeLists.txt` 登记，对真实调用的 `conv2d_backward` 与 `conv2d_backward_cuda` 做加权二次损失下的 FP64 中心差分检查，覆盖 `3x3 pad=1`、`3x3 pad=0`、`3x5 pad=2` 非方核与 `need_dx=false`。
- 第 7 条（提交 `25c1d03`）：实测定位 `uavrgb_train` 占核约 95%–99% 来自 CUDA 默认 `cudaDeviceScheduleAuto` 在单线程进程下退化为用户态自旋等 GPU；在 `src/include/uavrgb/trainer.hpp`、`src/nn_cuda.cu`、`src/train_main.cpp` 新增默认关闭的 `--cuda-blocking-sync` 与环境变量 `UAVRGB_CUDA_BLOCKING_SYNC=1`（置 `cudaDeviceScheduleBlockingSync`），新增 `src/tests/test_cuda_blocking_sync.cpp`。

## 没做什么

- 没修改 checkpoint 二进制格式（未升 v7）、条件输入维度 `prev_dim`、FiLM 与深度辅助头挂接位置：
  - 原因：任务卡显式划归卡 E2E-C1，本卡禁止改动 checkpoint 格式以免影响在途的 E2E-1 与 E2E-P0。
- 没修改每轮末批不满半批时丢弃的训练逻辑（上一轮 `src.md` D5）：
  - 原因：任务卡显式排除，该改动会改变缺省训练轨迹，归入配方探索。
- 没修改 `tools/average_ckpt.py`（`read_val` 恒按 `flags[i] == "0"` 剔除碰撞步而不读契约位）与 `tools/select_ckpt.py`（缺 `config.json` 时告警回落 `drop_collision=False`）：
  - 原因：本卡边界限定只改动 `src/` 下文件；两处 Python 工具现状已在报告第 5 节列明，交主会话决定是否另排工具卡跟进。
- 没把 `--cuda-blocking-sync` 设为缺省开启，也没给常驻推理进程 `src/infer_main.cpp` 加该开关：
  - 原因：任务卡规定训练器新增开关默认关闭；推理进程单帧耗时约 0.4 ms 且闭环评测延迟敏感，不在本卡范围。
- 没把原已超过 280 行的历史大文件（`src/trainer.cpp` 716 行、`src/checkpoint.cpp` 319 行、`src/class_weights.cpp` 295 行）拆小：
  - 原因：遵循最小改动原则与 `src.md` D11 建议，不顺手重构无关代码；原在 280 行内的文件（含 `src/train_main.cpp` 249 行、`src/trainer_eval.cpp` 236 行）与全部 7 个新增测试文件均保持在 280 行以内，`src/nn_cuda.cu` 324 行保持在 360 行以内。
- 没在工作树跑全量 `tools/check.sh`（含 `cf_sitl` 编译与全部 44 项 CTest）：
  - 原因：交办指定本卡只跑受影响的 CTest 目标与 `tools/check.sh --python-only`，全量 `check.sh` 由主会话合并时统一执行。
- 没保留开 `--augment` 时旧版写死随机数种子 `0x9e3779b9u` 的行为：
  - 原因：交办第 2 条明确规定不开增强时逐位不变，开增强时的旧行为不保留。

## 待确认

- `tools/average_ckpt.py:168-169` 的 `read_val` 目前固定剔除碰撞步，未校验 checkpoint v5 契约的 `drop_collision` 位；当未来存在 `drop_collision=false` 训出的模型做校准时口径会不一致，需主会话决定是否在后续工具改动中补读契约位。
- 训练池守护脚本（如 E2E 系列后续训练入口）是否统一在环境变量加 `UAVRGB_CUDA_BLOCKING_SYNC=1` 以释放 CPU 核给 Gazebo 池，由主会话定夺（实测结果见报告第 5 节）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `cuobjdump -lelf uavrgb_train` | 通过：9 个 ELF cubin 全部为 `sm_75` |
| 受影响的 21 个 CTest 目标（`nice -n 10 ctest -j 1`） | 通过：21/21 PASS（含新增 7 项单测与 14 项关联回归） |
| 7 个新增单测的先红后绿与变异检查 | 通过：全部在草稿目录 `red/` 留存失败日志、`green/` 留存通过日志 |
| `tools/check.sh --python-only`（排 `check.lock`） | 通过：`rc=0`（2026-11-12 08:23:14，日志 `$D/tmp/phase12-cards/c1a/check_py/check_py.log`） |
| 真实数据集 `trunc/ds/dataset.uavr` 3 轮确定性逐位比对（`c1..c7` 及 `c7 --cuda-blocking-sync` 对 `main` 基线） | 通过：8 次重训的全部 6 个文件（`ckpt_ep01..03.bin`, `ckpt.bin`, `ckpt_best.bin`, `results.csv`）sha256 与 207 行 `metrics.jsonl`（除墙钟字段外）严格相等（`ALL_DONE rc=0`） |
| 第 7 条真实 3 轮训练开关前后对比（独占 `train.lock`） | 默认关：`0.929` 核（`utime=52.03s, stime=1.30s`），`19.13 s/轮`；开 `--cuda-blocking-sync`：`0.095` 核（`utime=4.49s, stime=1.43s`），`20.87 s/轮`（紧邻同负载默认运行 `c5/c6` 为 `20.77/20.79 s/轮`） |

## 风险与遗留

- 对于早期无 `config.json` 且为 v2..v4（无 v5 契约段）的历史归档 checkpoint，直接运行 `uavrgb_train --eval-only` 现在会以 `rc=2` 拒绝而非静默按 `drop_collision=false` 评估；若确需复评此类早期归档权重，需在其目录放置写明 `"drop_collision":true/false` 的 `config.json`。
- `--log-grad-norm` 通过 `metrics.jsonl` 的 `status` 事件输出每轮汇总，不改变冻结的五类 `EventType` 定义；若下游脚本解析它，按前缀 `grad_norm epoch ` 读取键值即可。
