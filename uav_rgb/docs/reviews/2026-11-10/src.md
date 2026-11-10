# src/ 自研 C++/CUDA 训练推理栈审查报告

对象：`/root/claude-code-backup-20261028/code/uav_rgb` main @ c85a1ff，只读审查，2026-11-10。
记号：【已核实】= 读到了代码行或跑出了数字；【怀疑】= 推断，没有直接证据。`$D` = `/root/rivermind-data/uav_rgb`，`$S` = `$D/tmp/cursor-1110/agents/src`（本次草稿目录，脚本、输出、候选 ckpt 都在里面）。

## 0. 结论摘要

1. 数值面没有发现会让训练或推理算错的 bug：numpy 独立重写的前向与 `uavrgb_infer` 概率最大差 6.6e-6；ckpt 1,205,021 B 与 100,388 个参数逐项对上；11-07 与 11-09 两次独立训练的参数与 Adam 状态段 SHA256 相同；CTest 34/34。
2. 换种子后行为差别大的训练侧来源【已核实】：恒定 lr 1e-3 的 Adam 在末期是噪声驱动的来回过冲，「FWD 对 YAW」的类先验每轮摆 0.55–0.65 nat，相邻轮 17–22% 的帧换预测；只换 fc 权重就能复现其中的 75–89%，只换 fc 偏置时翻转的帧不到 0.1%。`ckpt_best` 是这条摆动序列上偏向 FWD 的一个瞬时状态。
3. 修法不需要新算子，但两件事要一起做：让权重停稳（`--cosine-lr`，或离线平均末 10 轮权重）并显式设定类先验（给 `fc.b[FWD]` 加一个标量，或换类权重）。离线结果：三个种子模型两两之间的预测分歧率，末轮权重 22–29%，现行 `ckpt_best` 10–14%，权重平均加校准 5–7%；test macro-F1 0.38–0.41，与 `ckpt_best` 在噪声内持平，收益在一致性，不在精度。候选 ckpt 已生成，闭环没测。
4. 新需求成本：多任务头、帧堆叠、软标签有现成路径（0–450 行）；BatchNorm 约 650–700 行，GRU 约 1,100–1,300 行，建议用余弦退火、权重平均、帧堆叠代替。
5. 部署：仓库里没有导出、量化、无 CUDA 的整网前向，从 `ckpt.bin` 到串口动作缺 10 环；numpy 模拟的 int8 量化（逐通道权重、最大值校准）与 FP32 动作一致率 99.4–99.8%。闭环数字加横向边界后是 10/19、12/20、9/20，种子间差异不显著。

## A. 学生模型规格

### A.1 逐层表（单帧默认结构）【已核实：`model.cpp:39-102`，`$S/ck.py` 读 seed1 `ckpt_best.bin`】

| # | 层 | k / s / p | 输出形状 [C,H,W] | 参数量 | 乘加次数 MAC |
|---|---|---|---|---|---|
| 0 | uint8 → float，`x/255.0f` | – | 3×120×160 | 0 | – |
| 1 | Conv c1 + ReLU | 3×3 / 1 / 1 | 16×120×160 | 448 | 8,294,400 |
| 2 | MaxPool | 2×2 / 2 / 0 | 16×60×80 | 0 | – |
| 3 | Conv c2 + ReLU | 3×3 / 1 / 1 | 32×60×80 | 4,640 | 22,118,400 |
| 4 | MaxPool | 2×2 / 2 / 0 | 32×30×40 | 0 | – |
| 5 | Conv c3 + ReLU | 3×3 / 1 / 1 | 64×30×40 | 18,496 | 22,118,400 |
| 6 | MaxPool | 2×2 / 2 / 0 | 64×15×20 | 0 | – |
| 7 | Flatten + Linear fc | – | 4 | 76,804 | 76,800 |
| | 合计 | | | **100,388** | **52,608,000** |

- fc 占参数的 76.5%，占 MAC 的 0.15%。双帧时 c1 为 880 参数 / 16,588,800 MAC，合计 100,820 / 60,902,400。
- 感受野：conv1 3 → pool1 4 → conv2 8 → pool2 10 → conv3 18 → pool3 22 像素，步距 8。15×20 的每个格子看 22×22 像素；全局信息只靠 fc 的逐位置权重汇总。
- 核对：`metrics.jsonl` 首行 `params=100388`；ckpt 大小 = 头 189 + 形状头 176 + 3×401,552（参数、Adam m、Adam v）= 1,205,021 B，与文件一致；v5 的旧 ckpt 少 8 B（1,205,013）。

### A.2 结构图

```text
Gazebo 320x240 RGB --PIL convert("RGB").resize((160,120), BILINEAR)--> uint8 [3,120,160]
                                         (双帧: [6,120,160] = [I_t ; I_t-1])
      | /255.0f，训练可选 photo / shift / flip（augment_dev.cu）
      v
  Conv c1 3x3 p1 (3|6 -> 16) + ReLU -> MaxPool2   [16, 60, 80]
  Conv c2 3x3 p1 (16 -> 32)  + ReLU -> MaxPool2   [32, 30, 40]
  Conv c3 3x3 p1 (32 -> 64)  + ReLU -> MaxPool2   [64, 15, 20] = p3
      |
      |   cond = [9 个 token 的 embedding (9*text_dim) | prev-action one-hot (4)]
      |          --Linear film--> [gamma(64) | beta(64)]
      +-- 可选 FiLM:  p3' = (1 + gamma_c) * p3 + beta_c
      |
      +---------------------------------+
      v                                 v  可选，只在训练时算
  Flatten 19200 -> Linear fc -> 4     Conv aux 1x1 (64 -> K=8) -> [K,15,20] -> 逐格 CE (255 = 屏蔽)
      |
  softmax -> argmax -> {0 SAFE_NEUTRAL, 1 FORWARD_SLOW, 2 YAW_LEFT, 3 YAW_RIGHT}
```

### A.3 输入预处理【已核实】

- 通道顺序 RGB；训练文件是 NCHW uint8（`tools/preprocess.py:618-638` 的 `load_frame`），闭环是 HWC uint8，`infer.cpp:68-79` 转 CHW。
- 缩放：PIL `BILINEAR`，320×240 → 160×120，训练（`preprocess.py:638`）与闭环（`sim/tools/student.py:116-117`）是同一种调用；逐字节对拍工具 `tools/check_input_parity.py` 已有。
- 归一化只有 `/255.0f`（`augment_dev.cu:21`），没有减均值除方差。训练与推理走同一个 kernel（`infer.cpp:56-59`）。

### A.4 可选分支与命令行开关【已核实：`train_main.cpp:51-148`，`model.hpp:17-41`】

| 分支 | 开关 | 接法 | 额外参数 | 前提 |
|---|---|---|---|---|
| 文本 FiLM | `--text-dim N`（数据集带指令侧车而要训纯视觉时须显式 `--vision-only`） | `text_emb[22,N]` 按 9 个 token 位置拼接 → `film` Linear(9N→128) → γ/β 调制 p3 | 22N + 9N·128 + 128 | `instructions.jsonl` 侧车 |
| 双帧 | `--two-frame` | c1 输入 6 通道，上一帧按侧车 `prev_index` 现读 | +432 | `dataset.aux` 侧车 |
| prev-action | `--prev-action` | one-hot(4) 并入 FiLM 条件向量 | +512（纯视觉模型另加 film 偏置，共 +640） | `dataset.aux` 侧车 |
| 深度分带头 | `--depth-aux`，`--aux-lambda 0.5` | 1×1 conv 64→K 接在 p3' 上，逐格 CE | K = 8 时 +520 | 侧车带深度目标 |
| 软标签蒸馏 | `--soft-targets f`，`--distill-alpha 0.5` | loss = α·CE(q) + (1−α)·CE(onehot)，都除以 N | 0 | UAVSOFT1 文件；与类权重、增强、标签平滑互斥（`trainer.cpp:438`、`trainer_dev.cpp:130`、`soft_targets.cpp:27`） |
| 逐轮留档 | `--save-every-epoch` | 多写 `ckpt_epNN.bin` | – | – |

### A.5 checkpoint 版本【已核实：`checkpoint.hpp:4-24`，`checkpoint.cpp:15-17`】

magic `UAVRCK01`，小端 float32，没有校验和，lr / β / eps 不入盘。v2 类权重 → v3 `best_macro_f1` → v4 `text_dim` → v5 训练契约 113 B（数据集 SHA、drop_collision、软标签 SHA、α、lr、seed、batch、split_scheme SHA）→ v6 `aux_classes`、`prev_dim`。只追加、旧偏移不变；读端接受 v2..v6。参数顺序固定为 c1.W, c1.b, c2.W, c2.b, c3.W, c3.b, fc.W, fc.b,（text_emb），（film.W, film.b），（aux.W, aux.b），之后是全部 m、全部 v。

### A.6 推理进程协议【已核实：`infer_main.cpp:17-75`，`infer.cpp:47-113`，实际运行】

- 启动 `uavrgb_infer --ckpt <ckpt.bin> [--text id,id,...]`，先在 stdout 打一行 hello。seed1 `ckpt_best.bin` 实测：`{"in_h":120,"in_w":160,"in_c":3,"num_classes":4,"text_dim":0,"prev_action_dim":0,"aux_classes":0}`。
- 每步从 stdin 读 `in_c/3` 张 `in_h×in_w×3` 字节的 HWC RGB（双帧先当前帧后上一帧），`prev_action_dim>0` 时再读 1 字节动作 id；回一行 `{"action":k,"probs":[...],"infer_ms":x}`（probs 是 6 位有效数字）。EOF 退出码 0，帧不完整退出码 1。辅助头在推理端不计算。
- 进程按步无状态。客户端 `sim/tools/student.py` 的 `StudentPolicy` 负责缩放、缓存上一帧与上一动作、`reset()`。

## B. 训练配方与随机性来源

### B.1 配方【已核实：三个 run 的 `config.json` 除 seed 外相同】

| 项 | 三个 run 的实际值 | 代码位置与默认值 |
|---|---|---|
| 优化器 | Adam，β1 0.9，β2 0.999，eps 1e-8，标准偏置修正 | `nn_cuda.cu:75-87`，`model_dev.cu:292-300` |
| 学习率 | 恒定 1e-3，没有 warmup，没有衰减 | `--warmup N` 线性，`--cosine-lr` 退火到 0（`trainer_dev.cpp:154-166`） |
| batch / epoch | 128 / 30，每轮 65 步共 1950 步，每轮最后一批只有 13 个样本，一个 run 约 7 分钟 | 默认 32 / 10（`trainer.hpp:24-25`） |
| 梯度裁剪 | 全局范数 5 | `model_dev.cu:279-291` |
| 权重衰减 / 标签平滑 | 0 / 0 | `--weight-decay` 是解耦式（AdamW）；`--label-smoothing ε` 的目标是 (1−ε)·onehot + ε/C，按标签的类权重加权后除以本批权重和，在 host 算（`trainer_dev.cpp:183-218`） |
| 类权重 | auto：w_c = N/(C·n_c) = [36.63, 0.298, 3.21, 3.28]，loss 除以本批权重和 | `class_weights.cpp:52-61`，`ops_weighted.cpp:76,102` |
| 增强 | 关 | photo（gamma 0.8–1.25、对比度 0.8–1.2、亮度 ±0.1）、shift ±4 px、flip（`trainer_dev.cpp:72-98`） |
| 数据 | train 8205 / val 574 / test 884 帧；标签 SAFE/FWD/L/R = train 56/6885/639/625，val 0/501/40/33，test 1/739/72/72 | `--drop-collision` |
| 选模 | 每轮 val macro-F1 严格大于历史最高才写 `ckpt_best.bin`；test 只在结束时用它评一次 | `trainer.cpp:632-641,656-660` |

增强的历史 A/B（`docs/worklog/2026-11-06-1740-src-algo.md:88-101`，T6 数据 10 轮，test macro-F1 相对基线 0.835）：全开 −0.099，只平移 −0.054，只翻转 −0.009，只光度 +0.004，种子噪声 ±0.005。该文档把平移的害处解释成「4 px 足以把前进变成该偏航」。【怀疑】这个解释与几何不符：仿真水平视场 60° 摊到 160 px 是 0.375°/px，4 px 只有 1.5°，教师的转向容差是 25°。更可能的原因在模型：三次池化后步距 8 px，±4 px 是半格，而 fc 是逐位置权重，没有平移不变性；平移臂的训练 CE 停在 0.53（基线 0.31）也是欠拟合的表现。没有做实验区分两种解释。

### B.2 随机性来源【已核实】

| 来源 | 种子与实现 | 位置 |
|---|---|---|
| 权重初始化 | `std::mt19937(seed)`，Xavier-uniform ±sqrt(6/(fan_in+fan_out))，顺序 c1.W→c2.W→c3.W→fc.W，偏置 0。上界 0.187 / 0.118 / 0.083 / 0.0177，卷积层是 He 上界的 0.40–0.58 倍 | `model.cpp:82-95` |
| 每轮打乱 | 另一个 `std::mt19937(seed)`，`std::shuffle` 整个 train 下标 | `trainer.cpp:507,515` |
| 增强 | `std::mt19937{0x9e3779b9}` 固定，不随 `--seed` | `trainer.hpp:187` |
| GPU 归约 | conv 的 dW/db 固定顺序归约；池化反传的 float `atomicAdd` 因窗口不重叠而确定 | `conv_cuda.cu:60-115`，`nn_cuda.cu:71` |
| 续训 | RNG 状态不入 ckpt，续训从种子重新开始 | `checkpoint.hpp:4-24` |

逐位可复现【已核实】：`$D/tmp/agent6-distill/trunc/run`（11-07，v5 二进制）与 `e1-noise/seed0/run`（11-09，v6）的 `ckpt.bin`、`ckpt_best.bin` 末尾 1,204,832 B（形状头 + 参数 + m + v）SHA256 相同。所以「换种子差别大」不是不确定性造成的，是轨迹本身对种子敏感。

### B.3 三个种子的曲线【已核实：`results.csv`、`metrics.jsonl`；我用真实 `uavrgb_infer` 重算的 val macro-F1 与训练器逐轮相差 1.5e-8 以内】

- `results.csv` 两列都是无权 CE（`trainer.cpp:566-570`、`trainer_eval.cpp:38-40`）。训练 CE 整体下降（1.23–1.26 → 0.47–0.52），val CE 不降（在 0.57–1.51 之间摆）。
- val acc 在相邻轮之间跳动：seed0 ep6→7→8 为 0.786 → 0.303 → 0.765；seed1 ep1→2 为 0.754 → 0.439。
- `ckpt_best` 是第 28 / 16 / 18 轮，val macro-F1 0.426 / 0.407 / 0.394。对 574 帧做自助法重采样，单个数字的 sd 是 0.024–0.026，95% 区间约 ±0.05，三个种子的差在一个 sd 左右。
- 口径提醒（已知）：`macro_f1` 按 4 类平均，没有样本的类记 0（`eval_metrics.cpp:26-40`）。val 没有 SAFE 样本，上限是 0.75；test 只有 1 个 SAFE 样本，四分之一的分数由这 1 帧决定。
- 闭环口径：`docs/worklog/2026-11-10-1010-phase10-audit.md:15-32`（工作区里未提交的文件）指出成功判据没有横向边界，加上之后 seed0 11/19 → 10/19，seed1 20/20 → 12/20（8 集是绕开柱阵），seed2 10/20 → 9/20，seed1 对 seed2 的配对检验 p = 0.508。

### B.4 机制：末期的类先验过冲【已核实，数字见 `$S/analyze*.out`、`analyze_mech.out`、`fc_grad_series_s*.out`】

做法：对 90 个逐轮 ckpt 用 `uavrgb_infer` 取 val + test 共 1458 帧的概率；另用 numpy 前向取第三层特征，把相邻两轮的「卷积特征」和「fc 权重」交叉组合。δ 的定义：只给 FWD 的 logit 加一个标量，使 val 上预测 FWD 的占比等于标签占比 0.873 所需的量（等价于改 `fc.b[1]`）。空画面帧 = 视锥内没有柱子的帧（`tools/select_ckpt.py:128` 的定义）。

| 末 10 轮（val，括号内 test） | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 相邻轮预测翻转率 | 0.224（0.240） | 0.173（0.162） | 0.173（0.183） |
| 空画面帧（val 155 帧）的翻转率 | 0.532 | 0.367 | 0.354 |
| 只换 fc 权重 / 只换卷积 / 只换 fc.b 时的翻转率 | 0.195 / 0.095 / 0.0005 | 0.154 / 0.081 / 0.0002 | 0.129 / 0.084 / 0.0003 |
| δ 的 sd / 极差（nat） | 0.43 / 1.43 | 0.52 / 1.98 | 0.48 / 1.51 |
| ep30 时 FWD 与最大 YAW 的 logit 差小于 0.5 nat 的帧 | 18.3% | 27.7% | 27.9% |
| 只校准 δ 之后的翻转率 | 0.048（0.055） | 0.045（0.040） | 0.037（0.046） |
| 单轮 val macro-F1：原始 → 校准 | 0.372±0.034 → 0.424±0.012 | 0.358±0.025 → 0.393±0.010 | 0.357±0.027 → 0.409±0.010 |
| 单轮 test macro-F1：原始 → 校准 | 0.341±0.023 → 0.383±0.012 | 0.357±0.021 → 0.391±0.014 | 0.355±0.019 → 0.390±0.007 |
| 空画面 YAW 率：原始均值（最小–最大）→ 校准 | 0.41（0.03–0.85）→ 0.01 | 0.29（0.01–0.67）→ 0.01 | 0.35（0.06–0.76）→ 0.01 |

读法：

1. 翻转主要是一个标量在摆。校准这一个数之后，同一种子相邻轮的翻转率从 16–24% 降到 4–6%，三个种子末轮模型两两之间的分歧率从 val 22–29% 降到 7.0–7.5%（test 22–28% → 8–10%）。种子之间真正不同的部分只有这 7–10%。
2. 这个标量不在偏置里。`fc.b` 每轮只动 0.001–0.003，只换 `fc.b` 几乎不翻转；类先验由 `fc.W` 乘以 19,200 维非负特征的「公共模」承载。特征均值 0.03–0.05、非零占比 7–15%，每帧 L1 约 530–890，按 lr × L1 × |m̂|/√v̂ 估算，Adam 每步在公共模方向上把 logit 推 0.04–0.16 nat。`fc.W` 的 L2 范数 30 轮从 2.83 涨到 15.7–19.6，末 10 轮每轮相对位移 6.7–8.2%。
3. 末期的更新是噪声。由 ckpt 的二阶矩反推，裁剪后梯度范数的 RMS 是 2.4–3.3（裁剪阈值 5，没有长期顶在阈值上）；第 30 轮一阶矩范数 0.50–0.61，纯噪声梯度的期望值是 0.55–0.59。
4. 动量在过冲。seed1 逐轮算 fc 层在固定 512 个训练帧上的精确梯度与 Adam 动量的夹角，30 轮里 22 轮余弦为负（−0.84 到 +0.53）。这 512 帧上预测 FWD 的占比逐轮在 0.41–0.83 之间摆（ep15→16→17 为 0.56 → 0.82 → 0.65），L 与 R 也来回换（ep28 为 0.28 / 0.06，ep29 为 0.12 / 0.17）。
5. `ckpt_best` 取的是偏向 FWD 的瞬时状态。三个种子的 best 轮在这 512 帧上预测 FWD 的占比是 0.81 / 0.82 / 0.75，前后相邻轮是 0.56–0.70（seed2 的证据弱一些：后一轮也有 0.70）；seed1 的第 16 轮同时是加权训练损失的一个尖峰（0.64，前后两轮 0.45）。解释：加权目标的最优点偏向 YAW，而 val macro-F1 按自然分布计分，偏向 FWD 的过冲时刻得分最高；依据是上表「原始 → 校准」两行，把先验推向 FWD 后 macro-F1 在三个种子上都升高。run 内 val 与 test 的 macro-F1 秩相关 +0.86 到 +0.91，说明 val 选的确实是「先验合适的那一轮」，不是纯噪声；它排不出的是种子之间的顺序和闭环结果。
6. 空画面正好落在决策边界上。训练集 2145 个空画面帧的标签是 SAFE 1 / FWD 1909 / L 126 / R 109（YAW 占 11%）；乘上 auto 类权重后的加权后验是 FWD 0.42 : YAW 0.56，两边接近对半。换成平方根逆频率权重是 0.71 : 0.29，不加权是 0.89 : 0.11。auto 权重下每个 batch 的有效样本数只有 11.8（一个 SAFE 样本占该批权重的 28%）。`DISTILL_20261107.md` §2.4 已用实验说明震荡随 YAW 权重缩放、与 SAFE 权重无关，这里的数字是它在这份数据上的补充。
7. 与闭环的对应（n = 3，只能说相容）：三个 `ckpt_best` 的 val 空画面 YAW 率是 seed1 0.02 < seed0 0.05 < seed2 0.10，修正口径的闭环成功数是 12 > 10 > 9。

### B.5 各怀疑项的判定

| 怀疑项 | 判定 | 依据 |
|---|---|---|
| 学习率偏大且不衰减 | 主因【已核实】 | B.4 第 2–4 条。旁证是历史 run（T6 数据，10 轮）的末三轮 val acc：恒定 lr 为 0.889 → 0.795 → 0.904（`A0_base_s0`）、0.907 → 0.839 → 0.835（`s1`）；余弦为 0.906 → 0.890 → 0.888（`A2_cos_s0`）、0.866 → 0.897 → 0.895（`s1`）。同一组 A/B 里余弦对 `ckpt_best` 的 test macro-F1 没有影响（−0.001），所以它当时没被采纳 |
| 类权重放大噪声 | 放大器【已核实】 | B.4 第 6 条；DISTILL §2.4 |
| val 太小 | 部分成立 | 自助法 sd 0.025 与种子间极差 0.032 同量级；run 内选轮有效（秩相关 +0.86 以上） |
| 最后一层偏置 | 否【已核实】 | 只换 `fc.b` 的翻转率 0.02–0.05% |
| 没有归一化层 | 间接相关 | fc 输入的 L1 大，放大了公共模方向的步长；不加归一化层也能压住（B.6） |
| 初始化尺度 | 未见证据 | 校准后跨种子分歧率（7–10%）接近同种子相邻轮的残余翻转率（4–6%） |
| 梯度裁剪 | 否 | RMS 2.4–3.3 低于阈值 5 |
| 每轮最后 13 个样本的那一步 | 【怀疑】影响小，没验证 | 该步之后紧接着评估和存盘；约 11% 的概率批内没有 YAW 样本 |

### B.6 最便宜的实验（按成本排序）

已跑出数字的（离线，test 884 帧；三个数对应 seed 0 / 1 / 2）：

| 模型 | acc | macro-F1 | 召回 FWD / L / R | 空画面 YAW 率 | 跨种子分歧率 |
|---|---|---|---|---|---|
| `ckpt_best`（现状） | 0.796 / 0.817 / 0.757 | 0.382 / 0.400 / 0.384 | 0.89/0.32/0.31，0.92/0.29/0.33，0.83/0.43/0.36 | 0.08 / 0.00 / 0.12 | 0.105–0.138 |
| 末 10 轮权重平均 | 0.707 / 0.755 / 0.749 | 0.356 / 0.369 / 0.378 | 0.77/0.42/0.40，0.83/0.36/0.36，0.82/0.39/0.43 | 0.32 / 0.20 / 0.20 | 0.131–0.172 |
| 末 10 轮权重平均 + δ | 0.829 / 0.839 / 0.829 | 0.403 / 0.408 / 0.382 | 0.93/0.28/0.32，0.95/0.26/0.32，0.94/0.19/0.32 | 0.00 / 0.00 / 0.00 | 0.053–0.068 |
| `ckpt_best` + δ | 0.827 / 0.824 / 0.833 | 0.396 / 0.381 / 0.400 | 0.93/0.31/0.28，0.94/0.24/0.26，0.94/0.31/0.26 | 0.00 / 0.00 / 0.00 | 0.054–0.079 |
| 三个 `ckpt_best` 概率平均 | 0.818 | 0.413 | 0.91/0.35/0.36 | 0.01 | – |

- 权重平均把「窗口挪一轮」的翻转率压到 1.6–3.3%（单轮是 15–24%），但不校准时它停在加权目标的最优点，空画面 YAW 率 20–32%，离线指标反而低于 `ckpt_best`。稳定和先验要一起处理，只做前者会得到一个稳定地偏向转弯的模型。
- 校准会压低 YAW 召回（test 上 L 0.19–0.31，R 0.26–0.32），对避障是好是坏要由闭环判定。δ 的取法（对齐标签占比）是我选的一个点。
- 离线 macro-F1 的差别（±0.02）都在自助法噪声（sd 0.025）以内。这张表能说明的是一致性：权重平均加校准后跨种子分歧率减半，空画面 YAW 率归零。
- 候选 ckpt 在 `$S/ckpt/`：`cal_swa_s{0,1,2}_k10.bin`、`swa_s*_k{05,10,15,20}.bin`、`cal_s*_best.bin`、`cal_s*_ep30.bin`，都能被 `uavrgb_infer` 直接加载（校准版的真实推理与离线平移的预测一致率 99.93–100%）。只供推理：文件头和 Adam 状态沿用源 ckpt，不要拿来续训。

建议接着做的（我没有权限启动训练和 Gazebo）：

1. 闭环评 `cal_swa_s*_k10.bin` 对 `ckpt_best.bin`，三个种子各 20 集，0 行代码。要扫 δ 时（建议 0.5 倍、0.75 倍各一档，看 YAW 召回与空画面 YAW 率的折中），在 `$S/calib.py:32` 给 `d` 乘上倍率重新生成。
2. 重训三个种子，加 `--cosine-lr --warmup 65 --save-every-epoch`，取末轮 `ckpt.bin`、不选模，看末 10 轮翻转率是否低于 5%。0 行代码。【怀疑】只做这一步会得到与「权重平均不校准」相同的偏转弯模型，要和第 3 步或 δ 配合。
3. 在 2 的基础上把类权重显式设成平方根逆频率 `--class-weights 7.79,0.70,2.31,2.33`。0 行代码。
4. 三个种子的概率平均当教师，用现成的 `--soft-targets` 蒸馏回单模型（它与类权重、增强互斥）。
5. 把初始化种子和打乱种子拆开（D4，约 8 行），做 2×2 分解方差来源。

## C. 算子库盘点

### C.1 已有【已核实】

| 算子 | CPU 参考 | CUDA | 梯度检查 / 对拍 |
|---|---|---|---|
| Conv2d（任意核与 pad；步长固定 1，无 dilation、groups） | `conv_cpu.cpp` | `conv_cuda.cu` | `gradcheck_conv2d`（自含 FP64，只查 dW、db）；CPU↔CUDA 对拍 |
| ReLU、MaxPool2d（方窗，k、s 可设） | `nn_cpu.cpp` | `nn_cuda.cu` | `gradcheck_relu`、`gradcheck_maxpool2d`；对拍 |
| Linear、softmax-CE、类权重 CE | `ops_cpu.cpp`、`ops_weighted.cpp` | `linear_cuda.cu`、`softmax_cuda.cu` | `gradcheck_linear(_weighted)`；对拍 |
| 软标签 CE、标签平滑 CE | `ops_soft.cpp`、`trainer_dev.cpp:183` | 无，N×C 小张量在 host 算 | `test_soft_loss` |
| 逐格 CE（255 屏蔽） | `ops_spatial.cpp` | `spatial_ce_cuda.cu` | `gradcheck_spatial_ce`；对拍 |
| FiLM、1×1 辅助头 | 只有 `model.cpp` 的 host 路径（FiLM 在 host 算，辅助头调 `conv2d_*_cuda`） | `model_dev_film.cu`、`model_dev_aux.cu` | `test_text_branch` FP32 中心差分；host↔device 对拍 |
| Adam / AdamW、全局范数裁剪 | `nn_cpu.cpp`（没有 weight decay 参数） | `nn_cuda.cu` | 对拍（wd = 0） |
| uint8→float 与三种增强 | 无 | `augment_dev.cu` | `test_augment_dev` |
| warmup + cosine | `trainer_dev.cpp:154`（host） | – | 没有数值测试（D3） |
| macro-F1、混淆矩阵 | `eval_metrics.cpp` | – | `test_eval_metrics` |

### C.2 缺失项：工作量与最省事的替代（行数按现有同类文件估算，含测试）

| 缺失项 | 要新增 / 改动的文件 | 估算 | 需要的测试 | 最省事的替代 |
|---|---|---|---|---|
| BatchNorm | 新 `bn_cpu.cpp`、`bn_cuda.cu`、`model_dev_bn.cu`；改 `model.{hpp,cpp}`、`checkpoint.cpp`（v7：running mean / var 是没有 Adam 状态的缓冲段） | 650–700 行 | FP64 gradcheck、对拍、训练与推理两种模式、同种子逐位复现（批统计要固定顺序归约）、折叠进 conv 后输出一致 | `--cosine-lr` 或权重平均（B.6）；输入减均值除方差只改 `augment_dev.cu:21` 一行 |
| GroupNorm | 新 `gn_cpu.cpp`、`gn_cuda.cu`，两条路径接线 | 约 400 行 | gradcheck、对拍 | 同上。它不能折叠进 conv，【怀疑】int8 NPU 上会回落到 CPU |
| 权重标准化 | `conv_cpu.cpp`、`conv_cuda.cu` 各加一对 W 的标准化前反向，`model_dev.cu` 接线 | 约 170 行 | gradcheck、对拍 | 需要归一化时的首选：导出时写出标准化后的 W，板端零成本 |
| Dropout | `nn_cpu.cpp`、`nn_cuda.cu` 加掩码 kernel（种子、步数、下标的整数哈希） | 约 100 行 | 掩码比例、同种子可复现、推理时关闭 | 现有 `--weight-decay`、`--augment photo` |
| GRU / LSTM | 新 `gru_cpu.cpp`、`gru_cuda.cu`、`model_dev_gru.cu`（沿时间存激活的 BPTT）；`trainer_dev.cpp` 按 `prev_index` 取连续片段；`infer.cpp` 与 `student.py` 带隐状态和复位；ckpt v7 | 1,100–1,300 行 | gradcheck、对拍、逐步推理与整段前向一致 | 帧堆叠：`--two-frame` 已有，扩到 k 帧约 30 行（`kMaxInC`、`read_u8_batch` 循环、`student.py` 换成队列）；prev-action 已有 |
| depthwise / 带步长的 conv | `ops_dev.hpp` 的 `ConvShape` 加字段，`conv_cpu.cpp` 与 `conv_cuda.cu` 的四个 kernel 改下标，`conv_gradcheck.cpp` 同步 | 步长约 120 行，groups 约 200 行 | 现有 gradcheck 与对拍加用例 | 52.6 M MAC 已经够小；降采样继续用 MaxPool |
| 全局平均池化 | `nn_cpu.cpp`、`nn_cuda.cu` 加一对前反向，ckpt 加结构标志 | 约 120 行 | gradcheck、对拍、ckpt 往返 | 第四个 MaxPool（k = s = 5 → 64×3×4，fc 输入 768 维）：不需要新算子，接线约 40 行，保留粗位置；效果没有实验 |
| 上采样 / 反卷积 | 最近邻上采样加在 `nn_*`；反卷积可复用 `conv2d_dx_kernel` 当前向，但要先有步长支持 | 100–280 行 | gradcheck、对拍 | 把目标降采样到 15×20 再用 1×1 头（深度头现在的做法） |
| sigmoid + BCE | 新 `ops_bce.cpp`（N×K 在 host 算）；逐格版本另加 `bce_cuda.cu` | 160–260 行 | FP64 gradcheck、对拍 | 分桶后用 softmax-CE 或逐格 CE |
| L1 / Huber | 新 `ops_huber.cpp`（host） | 约 130 行 | FP64 gradcheck | 把碰撞时间分带后走逐格 CE |
| 多头 loss 加权 | 现在只有一个辅助头和一个 `aux_lambda`；第二个头按 F 节 | 350–450 行 | 见 F 节 | 换目标不换头：侧车的 depth 段写成别的逐格类别，C++ 0 行 |
| EMA 权重 | `nn_cuda.cu` 加 5 行 kernel，`DeviceCnn` 加缓冲与下载，`trainer.cpp` 每轮换权重评估并另存 | 约 150 行 | 衰减算术、评估确实用了 EMA 权重 | 离线平均逐轮 ckpt：`$S/swa.py` + `$S/ck.py` 共 166 行 Python，已实测 |
| 梯度裁剪 | 已有（`model_dev.cu:279-291`） | – | – | 只缺日志（D2） |
| int8 量化感知训练 | 新 `fakequant_cuda.cu`，观测器状态入 ckpt，两条路径接线 | 约 450 行 | 直通估计的梯度、与导出的量化参数一致 | 训练后量化；E 节模拟一致率 99.4–99.8% |
| ONNX 导出 | 新 `tools/export_onnx.py`（读 ckpt，拼 3×(Conv, Relu, MaxPool) + Flatten + Gemm）与 numpy 参考前向 | 200–250 行 Python | 导出图的输出与 `uavrgb_infer` 相差小于 1e-4 | 无，部署第一步。`.venv-tools` 里只有 numpy 和 Pillow，onnx 包要另装 |

## D. bug 与架构问题

与历史报告去重：`AUDIT_SRC.md`、`REVIEW2_SRC_20261107.md` 里已修并带回归测试的项不再列。重点方向里查过且没发现问题的：Adam 偏置修正与 AdamW 公式（`nn_cuda.cu:75-87`）；各 loss 的分母（硬 CE 与软标签 CE 除以 N，加权 CE 除以权重和，逐格 CE 除以有效格数且全屏蔽时不除零，`model_dev_aux.cu:58,65`）；软标签与类权重的互斥查的是最终生效的权重（`trainer.cpp:436-441`）；conv 反传公式（`conv_cpu.cpp:84-143`、`conv_cuda.cu:65-142`，逐项推过）；UAVR 头的 magic、版本、偏移自洽、文件长度、标签范围（`dataset.cpp:472-509`）与取图下标检查（`:540`）；split 与碰撞过滤（我按 meta 的 `episodes[].split` 和碰撞标记用 Python 独立取出 val 574 帧重算 macro-F1，与训练器 90 个轮次的数字相差 1.5e-8 以内，test 的类别计数 1 / 739 / 72 / 72 也与训练器的混淆矩阵一致）；训练与推理共用预处理 kernel；best 指标入盘且先更新 best 再写 `ckpt.bin`（`trainer.cpp:632-643`）；显存由 `DeviceCnn` 一次性预分配、析构统一释放（`model_dev.cu:149-150`），构造中途失败也会释放（`:97-105`），没有用工具查泄漏。

**D1【新，已核实，优先级最高】默认配方没有让权重停稳的机制，交付权重取自摆动序列。**
证据：B.4。选模规则是

```632:634:/root/claude-code-backup-20261028/code/uav_rgb/src/trainer.cpp
    if (!dev_is_train && dev_stats.macro_f1 > static_cast<double>(state_.best_macro_f1)) {
      state_.best_macro_f1 = static_cast<float>(dev_stats.macro_f1);
      save_checkpoint(best_ckpt_path, model_, state_);
```

修法：不动 C++。流水线训练命令加 `--cosine-lr --warmup 65`，或把 `$S/swa.py`、`$S/ck.py` 收进 `tools/`（166 行）做末 k 轮平均；类先验用显式类权重或 δ 设定。验证：`tools/select_ckpt.py:194` 已经输出 `flip_rate`，在它的验收里加一条「末 10 轮相邻轮翻转率均值 < 0.05」，现在三个 run 是 0.17–0.22（红）；权重平均实测 0.016–0.033（绿），余弦没有实测。

**D2【新，已核实】步级事件不记梯度范数。** `backward_step` 返回裁剪前范数（`model_dev.cu:301`），`trainer.cpp:555` 丢弃，`event.hpp:78` 的 train_step 字段里没有它。这次只能从 Adam 二阶矩反推。修法：`MetricsEvent` 加 `grad_norm`，`make_train_step` 多一个参数，`metrics_json.cpp` 多写一项，约 6 行。测试：`test_trainer` 断言 train_step 行含 `"grad_norm":`，现在红。

**D3【新，已核实】warmup、余弦、AdamW 衰减没有数值测试。** 测试只检查续训时开关被继承（`test_trainer.cpp:540-570`）；CPU 的 `adam_step`（`nn.hpp:68`）没有 weight decay 参数，对拍覆盖不到 wd > 0；衰减对偏置和 embedding 也生效（`model_dev.cu:297-299`）。我用历史 run `A2_cos_s0` 的 1450 个 lr 值核过余弦公式，最大误差 6.5e-11，当前实现是对的。修法：`test_trainer` 加约 15 行，跑 1 轮 warmup = 3 加余弦，把 `metrics.jsonl` 的 lr 列与公式比对；`test_nn_cuda` 加约 10 行，梯度为 0、wd = 0.1 时一步后参数等于 p0·(1 − lr·wd)。把 `trainer_dev.cpp:162` 的 0.5 改掉应当变红。

**D4【新，已核实】初始化与打乱共用一个种子，增强种子固定，RNG 不入 ckpt。**

```82:82:/root/claude-code-backup-20261028/code/uav_rgb/src/model.cpp
  std::mt19937 rng(seed);
```

```507:507:/root/claude-code-backup-20261028/code/uav_rgb/src/trainer.cpp
  std::mt19937 rng(cfg_.seed);
```

影响：分不开「初始化方差」和「数据顺序方差」；续训轨迹与不中断训练不等价（`AUDIT_SRC.md:81` 实测过，已知未修）。修法：加 `--data-seed`（缺省等于 seed），`trainer.cpp:507` 改用它，增强种子改成 `data_seed ^ 0x9e3779b9`，config.json 多一项，约 8 行。测试：同 `--seed`、不同 `--data-seed` 的两次训练第 1 步 loss 不同；不给 `--data-seed` 时 ckpt 与现在逐字节相同。

**D5【新，已核实；影响没验证】每轮最后一个不满的批照常更新。** `trainer.cpp:518-519` 取 `min(order.size(), s + bs)`；本数据集 8205 = 64×128 + 13，最后一步 13 个样本，加权 CE 按这 13 个的权重和归一化，随后立刻评估并存盘。修法：循环里加 `if (s > 0 && e - s < bs / 2) break;`，`per_epoch`（`trainer.cpp:476-477`）的算式同步，约 4 行。测试：合成集训练样本数 = batch + 1 时每轮步数应为 1，现在是 2。

**D6【新，已核实】`--max-grad-norm` 续训时不继承。** CLI 可设（`train_main.cpp:76-77`）、会写进 config.json（`class_weights.cpp:246`），但 `RunConfigSnapshot`（`trainer.hpp:84-107`）没有这个字段，`merge_resume_config`（`trainer.cpp:215-252`）不读，续训回到 5。修法：照 `weight_decay` 的写法加三处，约 6 行。测试：`test_trainer.cpp:540` 那段加 `sw.max_grad_norm = 1.0f`，断言 `config.resume.json` 含 `"max_grad_norm":1`，现在红。

**D7【新，已核实】没有不依赖 CUDA 的整网前向。** `model.cpp:178-186,243` 的 host 路径调用的全是 `*_cuda` 包装，`infer.cpp:59` 用 `DeviceCnn`，`CMakeLists.txt:5-6` 无条件要求 CUDA。逐算子的 CPU 参考存在，但没有把它们串成整网的代码。影响：板端没有可复用的推理代码，也没有给 NPU 输出对拍用的参考实现。修法：不拆库，加 `tools/ref_forward.py`（numpy，约 50 行，`$S/np_forward.py` 可直接用）。测试：同一 ckpt 的 32 帧，numpy 概率与 `uavrgb_infer` 相差小于 1e-4（我实测 6.6e-6）。

**D8【已知未修，复核仍成立】未接入的设备层与 `BatchReader` 共 1,531 行。** `src/device/`、`include/uavrgb/device/`、`batch_reader.{cpp,hpp}` 和 3 个测试在 `src/`、`sim/`、`tools/`、`ui/` 的非测试代码里没有调用方；`device/state_validation.cpp:53-55,81` 写死 3 通道输入和 8 段参数，P2 结构（6 通道、FiLM、辅助头）过不了它的校验。修法：删除这些文件和 `CMakeLists.txt:37,51-58` 及三个测试目标。验证：`check.sh` 全绿，CTest 从 34 项变 31 项。

**D9【新，已核实】差分测试没有直接覆盖训练实际使用的 conv 反传，dx 没有任何差分检查。** `gradcheck_conv2d`（`conv_gradcheck.cpp:56-154`）自带 FP64 前向和解析式，不调用 `conv2d_backward`，而且只检查 dW、db；FP32 的 CPU 与 CUDA 实现之间只有互相对拍（`test_conv_cuda.cpp:64-65`），两份实现如果错成一样测试不会红；`test_text_branch` 的整网中心差分只查 embedding 和 film，梯度不经过 conv 的 dx。我逐项推过公式，没有发现错误。旁证：第 1 轮结束时各卷积层 Adam 一阶矩方向上的损失方向导数是随机方向的 100 倍以上（`$S/dircheck.out`），它说明方向相关，不能证明逐元素正确。修法：`test_conv_cuda.cpp` 加约 20 行，对 x 的 8 个元素做 FP32 中心差分（loss = Σy²）与 `g.dx` 比。把 `conv_cpu.cpp:131` 和 `conv_cuda.cu:132` 的 `ih + pad - ki` 同时改成 `ih - pad + ki` 应当变红。

**D10【架构，已核实】前反向有两份实现。** `model.cpp`（333 行，host 张量进出）与 `model_dev*.cu`（513 行，显存常驻）各写一遍；trainer 和 infer 只用后者，前者只有测试在调用，作用是对拍参考（`test_device_train_step`）。每加一个分支要写两遍。`dev::ConvShape`（`ops_dev.hpp:11-13`）没有 stride、dilation、groups。建议不合并，新增头时 host 侧写到够对拍为止（F 节按这个口径估工作量）。

**D11【小项，已核实】** `write_run_config` 放在 `class_weights.cpp:238-272`，加一个开关要改这个不相干的文件；`trainer.cpp` 686 行、`dataset.cpp` 561 行等 5 个文件超过 `docs/ENGINEERING_RULES.md:21` 的 280 行上限；`CMakeLists.txt` 222 行里是 34 段手写的测试目标。下次加开关时把 config 读写挪进单独文件（纯移动），不建议专门重构。

## E. 部署盘点（LicheeRV Nano，SG2002 TPU，int8）

现状【已核实】：全仓库搜索 onnx、tpu-mlir、cvimodel、量化等关键词，只有 `docs/worklog/2026-11-06-1740-src-algo.md:164-172` 的文字建议，没有任何脚本；没有 `real-drone/`、`board/` 目录；`docs/uart_action.md:3` 自述「占位协议，未联调、未上机」。

算子是否 NPU 友好：主干只有 Conv3×3、ReLU、MaxPool2×2、Flatten、Gemm，都是常规算子；最大中间张量 16×120×160 = 307 KB（int8），int8 权重约 100 KB。FiLM 分支的 γ/β 可以在 CPU 上算好当第二输入；双帧的 6 通道输入【怀疑】不能用工具链自带的 3 通道图像预处理，要自己喂张量。

int8 一阶估计【已核实，numpy fake-quant 模拟，不是 TPU-MLIR；`$S/ptq_sim.out`】：权重 int8、激活 uint8、偏置不量化，校准用 256 个训练帧，在 1458 帧上与 FP32 比。

| 模型（全 int8 的动作一致率） | 逐通道权重，激活按最大值校准 | 逐张量权重，激活按最大值校准 | 逐通道权重，激活按 99.99 分位截断 |
|---|---|---|---|
| seed1 `ckpt_best` | 99.73%，macro-F1 +0.001 | 99.79% | 99.66% |
| seed0 第 30 轮（边界帧多） | 99.45% | 98.70% | 98.22% |
| seed1 权重平均 + δ | 99.79% | 99.79% | 99.45% |

逐通道权重、按最大值校准时，量化带来的动作变化是 0.2–0.6%，比训练状态的轮间摆动（17–22%）小一到两个量级。第三层输出有长尾（最大 3.7–7.1，99.99 分位 2.4–3.3），校准不要用分位截断。旧文档担心的「19200→4 的 Gemm 逐张量量化掉精度」在模拟里没有出现。

从 `ckpt.bin` 到「板上出一个动作并经串口发给 ESP32」缺的每一环：

1. 导出：`ckpt.bin` → ONNX。没有。
2. 转换脚本：TPU-MLIR 的 transform（mean 0、scale 1/255、RGB、NCHW）→ 校准 → INT8 部署 → `.cvimodel`。没有脚本，也没有工具链环境的记录。【怀疑】SG2002 对应 cv181x 目标，没核实。
3. 校准集抽取：从 `dataset.uavr` 抽 200–500 帧。没有。
4. 一致性验收：FP32 对 int8 的动作一致率和 macro-F1，以及参考前向（D7）。没有。
5. 板端推理程序：加载 cvimodel、喂一帧、取 4 个 logit。没有。
6. 相机与前处理：GC4653 取帧、裁剪、缩放到 160×120、RGB 顺序，并与 PIL 双线性口径对齐。没有；真机水平视场 81°（4:3 裁切后约 65°）对仿真 60° 的匹配代码也没有（`SRC_20261107.md:44`）。
7. 状态与后处理：argmax、双帧或 prev-action 的缓存与复位、推理超时回落到 SAFE。只有 PC 端的 `student.py`。
8. 板端串口发送：`tools/uart_action.py`（382 行，9 字节帧、CRC8、10 Hz 心跳、300 ms TTL）是 PC 侧占位实现，没有 ACK（`docs/uart_action.md:178`），设备名与引脚复用未定（`:40`）。
9. ESP32 固件：CF-Drone 固件的串口只有 CLI 和遥控两种用途，没有四动作解析器（`docs/uart_action.md:5,19`），也没有动作到姿态设定值的映射和超时保护；`real-drone/uart_protocol.md` 不存在（`:129`）。
10. 端到端延迟与安全测试：相机到电机的延迟预算、断链行为。没有数据。

## F. 给新算法实现者：新增一个输出头和一个 loss 的步骤

先看能不能不写 C++：现有辅助头是「p3' 上的 1×1 conv + 逐格 K 类 CE（255 屏蔽）」。碰撞时间分带、每列可通行方向这类目标，只要能写成 15×20 网格上的类别（不参与的格子填 255），改 `tools/preprocess.py` 往侧车的 depth 段写新目标即可，C++ 0 行。下面的清单针对「要与深度头并存的第二个头」，以 K 维航向分布头（Linear 19200→K，CE 或软标签 CE）为例，括号里是稠密头的差别。合计约 350–450 行（含测试）。主标签固定 4 类（`dataset.cpp:504`），新头的标签只能走侧车。

1. 侧车带标签：`tools/preprocess.py:67` 的 `AUX_HEADER` 用一个新的 flags 位，追加 payload 段（[n] 的 uint8 类别，或 [n,K] 的分布）；同步 `tools/DATASET_FORMAT.md` §6 和 `tools/read_dataset.cpp` 的 `check_aux`；`aux_sidecar.hpp` 的 `AuxSidecar` 加字段，`aux_sidecar.cpp` 读入并校验取值范围；`test_aux_sidecar` 补正反例。
2. 模型结构：`model.hpp` 的 `CnnModel` 加 `Linear hdg; int hdg_classes`，`CnnGrad` 加对应梯度；`make_cnn`（`model.cpp:39-102`）加参数与 Xavier 初始化；`param_list`（`model.cpp:113-126`）和 `cnn_grad_tensors`（`model.cpp:134-148`）在末尾追加 `hdg.W, hdg.b`。只能追加，已有顺序不能动，ckpt 和 Adam 状态都按这个顺序。
3. host 路径（对拍参考）：`model.cpp` 仿 `aux_forward`（`:246-251`）和 `backward` 里的 `daux` 分支（`:263-272`）加 `hdg_forward` 与 `dhdg`，约 30 行。
4. 设备路径：`model_dev.hpp` 加段下标 `seg_hdg_` 和缓冲；`model_dev.cu:86-90` 追加段；新建 `model_dev_hdg.cu`，照 `model_dev_aux.cu`（86 行）写 `alloc_hdg`、`hdg_loss(targets, lambda)`、`hdg_backward(n)`，梯度用 `dev::add_inplace` 叠进 `dflat_`；`backward_step`（`model_dev.cu:261`）加一行调用。（稠密头直接复用 `dev::conv2d_forward/backward` 和 `dev::spatial_ce`。）
5. loss：硬 CE、类权重 CE、软标签 CE、逐格 CE 现成（`ops.hpp:61-93`）。N×K 的小张量在 host 算，不写 CUDA，先例是 `trainer_dev.cpp:183`。确实要 BCE 或 Huber 时新建 `ops_xxx.cpp`，带 CPU 参考和 FP64 gradcheck。
6. 训练器：`trainer.hpp` 的 `TrainConfig` 加开关、λ 和 `_set` 标志；`train_main.cpp` 解析；`trainer_dev.cpp` 的 `forward_batch` 取本批目标（照 `depth_batch_`，`:116-125`），开翻转增强时方向类目标要镜像（`flip_side_inputs`，`:47-65`）；`trainer.cpp:546-554` 调用；`trainer_eval.cpp:42` 评估，`EvalStats`（`trainer.hpp:125-136`）加两项。
7. 续训与配置：`write_run_config`（`class_weights.cpp:238-272`）、`read_run_config`、`RunConfigSnapshot`、`merge_resume_config`（`trainer.cpp:233-237`）各加一项，否则续训建不出同形模型。
8. checkpoint 升 v7：`checkpoint.cpp:16` 的 `kVersion` 改 7；在 `aux_classes / prev_dim` 之后追加 `i32 hdg_classes`；`read_checkpoint_meta`（`checkpoint.cpp:199-221`）和 `load_checkpoint` 按 `version >= 7` 读，旧版本读回 0；更新 `checkpoint.hpp` 的布局注释和 `CheckpointMeta`；`test_checkpoint_versions` 加 v6 读入与 v7 往返。
9. 推理：`infer_main.cpp:63-64` 给 `make_cnn` 传新参数，hello（`:69-72`）追加 `"hdg_classes"`。只做训练辅助就到此为止，推理不算它；要在板端输出时改 `infer.cpp:105-110` 的回复和 `student.py` 的解析。
10. 测试与构建：`test_device_train_step` 加一种结构（host 对 device，两步后参数和 Adam 状态相差小于 1e-5）；`test_overfit` 加一段（小样本上新头能拟合）；新 loss 的 gradcheck；`test_trainer` 的续训继承；`CMakeLists.txt` 加源文件和测试目标。

容易踩的地方：新头的梯度与动作头一起参加全局范数裁剪（阈值 5），λ 大时会挤占动作头；某一步没有目标时梯度清零但 Adam 照走（`model_dev_aux.cu:73-77`）；shift 增强不平移稠密目标（`trainer_dev.cpp:132-133` 的 shortcut）；255 是屏蔽码，类别数不能超过 254；池化反传用 float `atomicAdd`，只有窗口不重叠才逐位可复现，不要引入 k 大于 s 的池化。

## G. 没读 / 没验证的内容

没读全文：`docs/worklog/2026-11-07-0705-student-p2.md`（读的是对应的阶段报告 `STUDENT_P2_20261107.md`）；`tools/read_dataset.cpp`、`tools/uart_action.py` 只读了文件头和函数清单；`src/tests/` 下的测试只读了头注释和被引用的片段；`sim/` 下除 `student.py` 外的代码；`third_party/`；仓库外的 `ALVLN/docs/08`、`09`。

没验证：

- 没有启动训练。余弦退火、平方根类权重、丢弃末批、EMA 的效果只有历史 run 的旁证或离线替代实验。
- 没有跑闭环。校准和权重平均对成功率的影响未知，B.6 的表全部是离线指标。
- int8 结果是 numpy 模拟，舍入方式与 TPU-MLIR 可能不同；SG2002 的工具链参数没核实。
- 没有自己构建和跑 CTest，34/34 读自 `$D/tmp/cursor-1110/check-full.log`。
- 反传链没有逐元素的独立数值对拍（D9），依据是公式推导加现有对拍测试。
- 历史审计里的已知未修项（meta 扫描器不跳空格、`parse_episodes` 以首个 `}` 结项、混合几何下的焦距）没有逐条复核。
- 逐位复现只核对了 seed 0 的两次 run；空画面掩码用的是 `tools/select_ckpt.py` 的函数，没有另行核对它。
- 平移增强有害的原因（B.1）是推断；C.2 的行数是估算。

做了什么：读了 `src/` 的全部非测试源文件与头文件、任务书列出的工具与文档、历史报告；读了三个种子的全部训练产物；在草稿目录写了 12 个 Python 脚本（1,002 行），对 90 个逐轮 ckpt 和 24 个自己生成的候选 ckpt 做了真实推理与离线分析。
没做什么及原因：没改仓库任何文件、没写仓库内的 worklog 和 `progress.md`（只读约束；审查期间仓库工作区有其它会话的改动，与本次无关）；没训练、没跑 Gazebo、没装包（任务规则）；没有实现 C 节的缺失算子（只估工作量）。

跳过的检查：D5 与 B.1 的平移解释没有实验支撑。风险：B.6 的校准把 YAW 召回压到 0.19–0.32，上机前必须先过闭环。
