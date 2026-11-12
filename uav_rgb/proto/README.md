# proto/：PyTorch 原型线

主线是自写的 C++/CUDA（`src/`、`uavrgb_train`、`uavrgb_infer`）。本目录是 PyTorch 原型线的地基：模型、C++ checkpoint 读写、行协议包装器、训练配方，
每一项都与主线逐项对拍。原型线的用途是在结构或训练配方上做快速试验，试验出来的权重能交回主线工具链（平均、校准、`uavrgb_infer`、Gazebo 评测）。
卡 E2E-P0 的结论与数字在 `$D/tmp/phase12-cards/reports/e2e-p0.md` 与 `docs/worklog/2026-11-12-*-proto-parity.md`，本文只写规则与用法。

## 规则

- 只改 `proto/` 下的文件。主线（`src/`、`tools/`、`sim/`）一行不动；原型要的功能在这里重写，不回改主线。
- 原型线的目标是「统计上与主线不可分」，不是逐位相同。浮点运算顺序不同（cuDNN 与手写卷积核、Adam 对微小梯度的放大），同种子的两条训练轨迹几步之后就分开；验收用前向逐帧对拍、动作序列对拍、训练指标的范围对拍与闭环成功数，不用逐步逐位比。
- 解释器：原型用 `$D/teacher-venv/bin/python`（torch、numpy、PIL；只借用，不装包，缺包就停下报告）；主线工具用主仓库的 `.venv-tools/bin/python`（numpy，无 torch）。
  `ckpt_io.py`、`metrics.py`、`data.py`、`compare_*.py`、`accept_offline.py`、`distribution_check.py`、`closedloop_summary.py`、`crosscheck_evalonly.py` 只依赖 numpy 或标准库，两个解释器都能跑。
- GPU：训练与推理前加 `CUDA_VISIBLE_DEVICES=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740`；单个训练进程显存约 1.9 GB（图像留在 CPU，按批搬进显存），约占 1 核。
- 数据、权重、实验输出、日志、临时文件放数据盘 `$D`（`/root/rivermind-data/uav_rgb`），不进仓库；`TMPDIR` 指到数据盘，`PYTHONDONTWRITEBYTECODE=1`（根分区空间小）。
- 排队：训练走训练池锁 `$D/tmp/phase12-cards/train.lock`，Gazebo 评测走 `machine.lock`，`tools/check.sh` 走 `check.lock`；长实验一律守护进程（见下），不在前台阻塞。
- 只支持 v6 checkpoint（`UAVRCK01`，带 113 字节训练契约）；读端校验与 C++ 一致：非有限值、维度不符、多余字节都拒绝。

## 文件

| 文件 | 作用 |
|---|---|
| `model.py` | `ProtoCnn`：C++ `CnnModel` 的复刻（3×[Conv3×3→ReLU→MaxPool2]→Linear；可选文本、prev-action、FiLM、辅助头、双帧）；`xavier_plan`、参数顺序、`model_from_ckpt`、`ckpt_from_model` |
| `ckpt_io.py` | C++ checkpoint v6 的读写（numpy），`make_contract` 生成 113 字节训练契约；读后再写逐字节一致 |
| `data.py`、`metrics.py` | `.uavr` 数据集读取（split、去碰撞、类别权重 auto、memmap 图像）；混淆矩阵、宏 F1、每类召回，口径同 C++ `trainer_eval.cpp` |
| `cpp_rng.py`、`cpp/rng_dump.cpp` | libstdc++ `std::mt19937`、`uniform_real_distribution<float>`、`std::shuffle` 的逐位复刻，与对拍程序逐项核对 |
| `train.py` | 训练配方复刻（初值、打乱、批次、加权 CE、全局梯度裁剪、Adam、逐轮评估、逐轮留档、训练后用 `ckpt_best.bin` 评估 test） |
| `export.py` | 把 `<run>/pt/*.pt` 写成 C++ 的 `ckpt_epNN.bin`、`ckpt.bin`、`ckpt_best.bin` |
| `infer_wrap.py`、`uavrgb_infer_proto` | `uavrgb_infer` 的 PyTorch 替身（同命令行、同行协议）与启动器；`--infer-bin` 指到启动器即可，`sim/`、`tools/` 不用改 |
| `proto_client.py` | 行协议客户端（离线对拍与测试用） |
| `parity_forward.py` | 第一关：前向与 `uavrgb_infer` 逐帧对拍（`real`：真实权重与真实帧；`variants`：随机权重的各结构变体） |
| `replay_compare.py` | 第二关离线：同一串帧分别灌给 C++ 与包装器，逐步比动作，记推理耗时 |
| `compare_episodes.py` | 第二关闭环：Gazebo 里两个臂对第二轮参照臂的逐步动作序列对照（首个分歧步、两边 `dispatch_lag_s` 与 `infer_ms`、推理时延、迟到步） |
| `accept_offline.py` | 第三关验收 (a) 的对拍表：val 与 test 的宏 F1、每类召回对 C++ 三个参照种子的范围放宽 ±0.03（字面读法）、去掉小样本类的读法、均值读法与 C++ 对 C++ 的基线。**字面读法已作废，不是判据**（任务卡 11-12 08:31 更正，见「验收口径与结果」），脚本只给描述性的表 |
| `distribution_check.py` | 验收 (a) 的事后补充读法：原型种子相对 C++ 全部种子（参照加额外种子）的均值、标准差、范围、z。11-12 07:00 曾被用作 (a) 的判据，08:31 作废，只描述；z 的标准误只用 sd/√n，漏掉 C++ 均值本身的不确定度，也没有 t 分布尾（第二轮审核 L-1，未改） |
| `crosscheck_evalonly.py` | 同一份 `ckpt_best.bin`，原型自己算的混淆矩阵与主线 `uavrgb_train --eval-only` 逐格相同 |
| `compare_training.py` | 同种子原型 vs C++：前几步 loss 差、逐轮指标、权重距离对 C++ 不同种子的基线 |
| `probe_divergence.py` | 分歧探针：从 C++ checkpoint 续训逐步对拍；1 ulp 初值扰动测噪声放大速度 |
| `closedloop_summary.py` | 第三关验收 (b)：闭环总成功数对第二轮 53/60 的预注册判定（R6 重跑合并、逐集配对） |
| `runs/` | 守护进程入口（见下） |
| `test_*.py` | 单元测试与带 `UAVRGB_PROTO_REAL=1` 守卫的真实数据对拍 |

## 训练

```bash
export D=/root/rivermind-data/uav_rgb TMPDIR=$D/tmp/<自己的草稿目录>/tmp PYTHONDONTWRITEBYTECODE=1
CUDA_VISIBLE_DEVICES=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740 \
  $D/teacher-venv/bin/python proto/train.py --seed 0 --out $D/exp-proto-parity-20261112/<名字>/seed0/run
```

- 缺省配置就是主线 e1-noise 的配方（`$D/exp-student-select-20261109/e1-noise/seed0/run/config.json`）：30 轮、batch 128、lr 0.001 常数、Adam（β 0.9/0.999、eps 1e-8）、类别权重 auto、去碰撞、无增强。常量集中在 `train.py` 顶部配置区，每项带出处注释。
- 数据缺省 `$D/tmp/agent6-distill/trunc/ds/dataset.uavr`（`--data` 可换）。split、drop-collision 与类别权重由数据集本身决定。
- 产物同 C++ 的 run 目录：`config.json`、`metrics.jsonl`、`results.csv`、`ckpt_epNN.bin`、`ckpt.bin`、`ckpt_best.bin`、`test_eval.jsonl`，另有 `pt/`（权重与 Adam 状态，`export.py` 的输入）与完成标记 `DONE`。
- `--out` 指向非空目录会拒绝；同一目录有 `.run.lock` 互斥。没有续训：中断了就换目录重跑。
- `--max-steps N` 只给对拍调试：跑到第 N 步停，不评估、不落盘。

三个种子并行、对拍与验收的整套流程是守护进程 `runs/run_step3_train.sh`（排训练池锁，用法见下）。

## 导出与交回主线

训练结束时 `train.py` 已经调用 `export.py` 把 `pt/` 写成 C++ 布局。单独导出：

```bash
python proto/export.py --run <run 目录>                    # 全部 pt 导成 ckpt_epNN.bin / ckpt.bin / ckpt_best.bin
python proto/export.py --pt <文件.pt> --out <文件.bin>     # 单个文件
```

目标文件已存在：内容一致就跳过，不一致报错（`--overwrite` 才覆盖）。导出的文件主线工具直接读：

```bash
# 末 10 轮权重平均 + 用真实 uavrgb_infer 在 val 帧上校准 FORWARD 先验（W1-5 之后流水线的交付权重）
CUDA_VISIBLE_DEVICES=<GPU> .venv-tools/bin/python tools/average_ckpt.py --run <run 目录> --data <dataset.uavr> \
    --out <产物.bin> --last-k 10 --infer-bin $D/bin-snapshots/main-97ae04b/uavrgb_infer
# 主线对 ckpt_best.bin 的 val/test 评估（核对混淆矩阵用）
uavrgb_train --eval-only <ckpt_best.bin> --data <dataset.uavr> --split all --drop-collision --out <目录>
```

`average_ckpt.py` 要求 `--data` 的 sha256 与 checkpoint 契约一致、`ckpt_epNN.bin` 的头里 epoch 与文件名一致；原型的导出满足这两条。

## 接入闭环（Gazebo）

两种用法，二选一：

- 评测原型训出的权重：导出并平均校准后，像主线权重一样传给 `sim/tools/eval_closed_loop.py --policy student --ckpt <产物.bin> --infer-bin $D/bin-snapshots/main-97ae04b/uavrgb_infer`。权重是 C++ 格式，推理端用主线 `uavrgb_infer`，与第二轮的评测条件相同。
- 评测原型的前向本身（包装器）：`--infer-bin proto/uavrgb_infer_proto`。环境变量 `UAV_RGB_PROTO_PYTHON`（解释器，缺省 teacher-venv）、`UAV_RGB_PROTO_DEVICE=cuda|cpu`（不设：有 CUDA 用 CUDA；设 `cuda` 而 CUDA 不可用则报错，不静默退回 CPU）、`UAV_RGB_PROTO_THREADS`（CPU 线程数，缺省 1）。
  包装器与 `uavrgb_infer` 的区别：概率输出 float64 全精度（C++ 为 6 位有效数字）、`infer_ms` 是 PyTorch 前向的耗时、只读 v6 checkpoint、不算辅助头。

## 守护进程入口（`runs/`）

长实验用 `setsid nohup`，写 `run.pid`（进程组号）、`run.log`、完成标记 `DONE`（`rc=<码> <时间> <原因>`）。公共函数在 `runs/common.sh`：单实例锁、等池锁（阻塞，每 10 分钟记一行）、起止各记一次 cgroup 的 `usage_usec` 与 `nr_throttled`。

| 脚本 | 做什么 | 池 |
|---|---|---|
| `run_cpp_prevaction.sh` | 用主线 `uavrgb_train` 训一个 1 轮的 prev-action 模型（第一关对拍 prev-action 结构的 C++ 参照） | 训练池 |
| `run_step2_gazebo.sh` | 第二关闭环：包装器臂与 C++ 臂各评 3 个冻结 seed，对第二轮参照臂出 `compare.md` | Gazebo 池 |
| `run_step3_train.sh` | 第三关离线：3 个种子并行训练、C++ eval-only 读回核对、验收 (a) 对拍表、C++ 重跑与额外种子的对照臂 | 训练池 |
| `run_step3_closedloop.sh` | 第三关验收 (b)：等离线部分结束、(a) 字面读法通过才继续（没通过 rc=5 退出，主会话决定越过时起跑前设 `ALLOW_OFFLINE_FAIL=1`）；平均与校准、3 个臂 × 20 个冻结 seed、2 条车道、R6 重跑、汇总与预注册判定 | Gazebo 池 |

用法（以第三关离线部分为例，其余同形）：

```bash
cd <worktree> && git status   # 脚本起跑时拿自己与 CODE_HEAD 里的同名文件比哈希：先提交再起跑，不一致 rc=4
mkdir -p $OUT && setsid nohup bash proto/runs/run_step3_train.sh > /dev/null 2>&1 < /dev/null &
tail -f $OUT/run.log          # 进度
cat $OUT/DONE                 # 结束后的 rc 与原因
kill -- -$(cat $OUT/run.pid)  # 停：只按进程组，不用 pkill -f
```

- 续跑：同一条命令重起；评完的种子与臂跳过，半成品挪到 `<名>.partial-<时刻>`（不删）；沿用起跑时的 `CODE_HEAD`。
- 起跑时冻结代码：`git archive <CODE_HEAD>` 解到 `$OUT/code`，逐文件核对 git blob 哈希；实验只用冻结副本，之后改仓库不影响跑着的实验。**跑着的守护进程所对应的脚本文件不要原地编辑**（bash 边读边执行）；要改就先按进程组停掉，改完提交再重起。
- 脚本头部的配置区每项一行带出处注释；带 `${X:-默认}` 的项可由环境变量覆盖（只给演练用）。

## 测试

`tools/check.sh` 只收 `sim/tools/`、`tools/`、`sim/3dgs/` 下的 `test_*.py`（`check.sh` 里的三个 `test_*.py` 循环），不收 `proto/test_*.py`，所以 `check.sh` 跑不到这里的测试；这里的测试单独跑（单进程，`nice -n 10`，每个文件在 2 分钟内）。
2026-11-12 16:32 在容器负载均值约 10 时实测（`$D/exp-proto-parity-takeover-20261112/tests/`）：默认模式 9 个文件共 117 例（跳过 10），墙钟合计 56 s，最慢 `test_infer_wrap` 36 s；`UAVRGB_PROTO_REAL=1` 的 4 个文件共 50 例（0 跳过），墙钟合计 103 s，最慢 `test_forward_parity` 75 s。

```bash
nice -n 10 $D/teacher-venv/bin/python -m unittest proto.test_ckpt_io proto.test_data_metrics proto.test_cpp_rng proto.test_model \
    proto.test_infer_wrap proto.test_compare_episodes proto.test_step3_tools proto.test_train proto.test_forward_parity -v
# 真实数据对拍（需要数据盘上的 C++ 参照物；GPU 那几条再加 UAVRGB_PROTO_DEVICE=cuda 与 CUDA_VISIBLE_DEVICES）
UAVRGB_PROTO_REAL=1 UAVRGB_PROTO_DEVICE=cuda nice -n 10 $D/teacher-venv/bin/python -m unittest proto.test_train proto.test_forward_parity -v
```

`test_ckpt_io.py`、`test_data_metrics.py`、`test_step3_tools.py` 只需 numpy，主线 `.venv-tools` 解释器也能跑（`cd proto && ../.venv-tools/bin/python -m unittest test_ckpt_io`）；需要 torch 的测试在没有 torch 的解释器里整份跳过。

## 验收口径与结果（卡 E2E-P0，2026-11-12 16:39 补）

数字与命令在 `docs/experiments/2026-11-12-proto-parity.md`，推导与逐项核对在 `$D/tmp/phase12-cards/reports/e2e-p0.md`。本节只写口径。

- **验收 (a) 的两种读法都已作废**（任务卡 11-12 08:31 更正，第二轮审核 L-1，已复现），不要当判据：
  - 字面读法（每个原型种子的每项指标落在 3 个 C++ 参照种子的范围上下各放宽 0.03 之内）：它对 33 个样本的类只容得下 1 个样本的差；同一条判据套在 5 个新的 C++ 种子上 4 个不过，全部 8 个 C++ 种子里任取 3 个做参照、其余当候选，逐个通过 45/280（16%）；
    原型三个种子 30 格里 28 格在范围内，2 格差 1 个样本（`$D/exp-proto-parity-20261112/step3-train/accept_with_control.md`）。
  - 「落在 8 个 C++ 种子的最小到最大之间」读法（11-12 07:00 改立）：两条线完全等价时，3 个候选种子的 24 格全部落在 8 个参照种子的极值之内的概率只有 3.3%（审核者的多元正态模拟；11 个真实 run 的 165 种分法里 5 种，3.0%），24 格全过是碰上的；
    它量的是没校准的 `ckpt_best`，单个转向类的召回要差 9 到 18 个点才检得出。
- **放行 (b) 的依据**：第二轮审核者在交付口径（末 10 轮平均加校准）上，对 val 加 test 的 1,458 帧比逐帧预测：跨线一致率 0.9446（C++ 内部 0.9374，原型内部 0.9479），双样本置换检验（165 种标号分法，精确）p = 0.78；
  概率的全变差距离同样检验 p = 0.87；九项指标的两样本 t 值绝对值都不超过 1.04（`$D/tmp/phase12-cards/review2-l/verify_all.log`）。
  (b) 因此在 `ALLOW_OFFLINE_FAIL=1` 下起跑；留痕只有 `step3-closedloop/run.log` 12:17:54 的一行警告，该变量没有写进 `inputs.txt`、`start.txt`、`closedloop.json`（第二轮审核 L-8）。
  `runs/run_step3_closedloop.sh` 的闸门与 `runs/run_step3_train.sh` 的 `RUN_CONTROL=auto` 仍按字面读法，代码没改。
- **以后的对拍判据**（任务卡 08:31）：在交付口径上，对逐帧预测做跨线对线内的双样本置换检验（p 不低于 0.05 记不可分），并报 test macro-F1 的均值差与区间；参照种子不少于 8 个，候选不少于 3 个；
  参照种子的个数在看结果之前定死，不许判不过再加种子。
- **验收 (b) 的结果**（预注册：总成功数与第二轮三个 K=1 臂的 53/60 相差不超过 6 集）：原型三臂 17、16、19，共 52/60，差 −1，**通过**。
  60 对里 4 对只有原型成功、5 对只有第二轮成功，精确 McNemar p = 1.000；按 20 个几何聚类的符号翻转检验 p = 1.000，整簇 bootstrap 95% 区间 [−5, +3] 集。三个原型臂没有迟到步、没有停顿步，相邻观测间隔全是 0.900 s；巡航步位移比第二轮多 0.19%，转角少 0.04%。
- **这个验收能检出多大的差**（估算，模型与假设见实验文档）：两条线等价时过门概率约 0.94 到 0.97；真实成功率差 0.10 时约 0.55 到 0.56，0.15 时约 0.22 到 0.28；要有 80% 的把握判不过，真实差要到 0.15 到 0.17（9 到 10 集）。所以「通过」排除不了 0.10 的差。

## 已知限制

- 训练只复刻了 e1-noise 配方用到的项：不含软标签蒸馏、深度辅助头、文本分支、双帧、prev-action、增强、warmup 与余弦、权重衰减、续训。其中 prev-action、双帧、文本、辅助头的**前向**与 checkpoint 读写已对拍（随机权重变体与一个 1 轮 C++ prev-action 模型）。
- 文本分支没有真实数据对拍（没有 C++ 训出的文本模型作参照），只有随机权重变体。
- 同种子的原型与 C++ 训练只有前几步在舍入量级内一致，之后分开；`probe_divergence.py` 给出分开速度与 1 ulp 初值扰动同量级的证据，验收按统计口径判，不按逐步判。
- 验收 (a) 的两种读法（字面 ±0.03；落在 C++ 全部种子的最小到最大之间）都已作废，`accept_offline.py` 与 `distribution_check.py` 的输出只是描述（见上一节）。
  代码里还留着按字面读法做的闸门：`runs/run_step3_closedloop.sh` 在字面读法不过时 rc=5 退出，`runs/run_step3_train.sh` 的 `RUN_CONTROL=auto` 只在字面读法不过时加 C++ 对照种子（通过时不加，是单向放宽）；这两处与 `distribution_check.py` 的标准误公式都没改（第二轮审核 L-1、L-8）。
