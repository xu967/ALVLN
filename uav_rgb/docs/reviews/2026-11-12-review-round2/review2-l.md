# 第二轮交叉审核报告 L：PyTorch 原型线 `proto/` 与主线对拍

- 审核对象：
  - 主仓库 `main`：`/root/claude-code-backup-20261028/code/uav_rgb` @ `ada0ca5`（只读）
  - 原型线工作树：`/root/claude-code-backup-20261028/code/wt/p12-e2e-p0`（分支 `p12-e2e-p0` @ `1e27bc4`，自 `main` `97c4228` 分出，只读，零字节改动，无字节码生成）
  - 在途扩展参考：`/root/claude-code-backup-20261028/code/wt/p12-e2e-p1`（分支 `p12-e2e-p1` @ `8d21245` 及工作区在编脚本，自 `1e27bc4` 分出，只读）
  - 实验产物与报告：`/root/rivermind-data/uav_rgb/exp-proto-parity-20261112/`、`/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/e2e-p0.md`
- 起止时刻：`2026-11-12 07:59:36 CST` 至 `2026-11-12 08:19 CST`
- 草稿与验证日志：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-l/`（`verify_all.py`、`verify_all.log`）

## 0. 结论

这个角度下最要紧的 3 件事：

1. **基础模型、随机数流与 e1-noise 训练配方的复刻本身等价，主会话放行 E2E-P0 进入闭环验收的决定成立；但任务卡 07:00 改立的「候选 24 格全落在 8 个参照种子最小到最大之间」判据在统计上不成立，不能用作后续通用对拍门禁（`L-1`）**。实测同分布下 `k=3` 候选在 `m=8` 参照上 24 格全过的零假设概率仅 `3.31%`（11 个真实 run 的 165 种划分仅 `5/165 = 3.03%` 通过），误拒率高达 `96.7%`；而在未校准 `ckpt_best` 口径下，因工作点漂移放大了种子方差，单个转向召回指标需要偏移 `12.0` 至 `18.3` 个百分点才能被可靠检出。将全部 8 个 C++ 与 3 个原型 run 换到真实交付口径（末 10 轮平均加 val FORWARD 校准 `cal_swa_k10`）并在 1,458 帧上做逐帧预测双样本 Energy 置换检验，跨线逐帧一致率为 `0.9446`（C++ 内部 `0.9374`、原型内部 `0.9479`，精确置换 `p = 0.7818`），概率全变差距离精确置换 `p = 0.8667`，确证两条线在交付口径上不可分。
2. **卡 E2E-P1 在地基上加 BatchNorm 后，交付链走了「每轮先 `fold_bn()` 成无 BN 权重、期末再算术平均 10 轮权重」及「EMA 权重直接套用在线末批 BN 统计量」的错误路径，正在扭曲 B1、B2、B3 与 C 组的筛选结果（`L-2`）**。多层卷积在参数线性平均后激活方差逐层收缩，先折叠掉 BN 再平均使归一化层失去重标定作用；在真实 `trunc` 子集上对 5 层 `ColHeadCnn(use_bn=True)` 实测，先折后均相对标准 SWA（平均参数后在训练集重估 BN 统计量再折叠）的 `logit` 标准差缩水至 `0.328`（缩减 `67.2%`），最大 `|Δlogit|` 达 `5.50` nat。
3. **从原型线到第二轮闭环评测存在三处接口断裂与一处侧车信任边界缺失（`L-3`、`L-4`）**：主线 `StudentPolicy` 硬编码只接受 `prev_action_dim` 为 `0` 或 `4`（E2E-P1 的 9 档码 `prev_dim=9` 与置空码 `prev_dim=10` 上闭环即报错中止，且 `p1` 在 `aux` 模式开置空时把 `prev_dim` 误设成 `10` 而非 `5`）；紧凑模型存于 `<run>/proto_ckpt/delivered_cal.pt` 导致 `nav_cond` 找不到父目录 `config.json` 而退化为 `unverified`；行协议无集边界且有两次全零预热，后续接列特征缓存或 GRU 会串入预热与跨集状态；`proto/data.py:read_aux` 跳过了 `uavr_sha256`、`payload_sha256`、样本数与取值范围校验。

---

## 1. 专项回答：验收 (a) 放行判据的统计审查、可分辨效应量与更强等价性检查

### 1.1 「3 个原型种子的 24 格落在 8 个 C++ 种子极值内」有什么漏洞

1. **否定原字面判据（3 个参照极值放宽 `±0.03`）是合理的**：`val` 分区类 3 真值仅 33 帧（1 帧 = `0.0303`），8 个 C++ 种子自测任取 3 参照对 3 候选同时通过率仅 `1/560`（`0.18%`，见 `step3-train/accept_with_control.md:38`），第 I 类错误率 `99.8%`。
2. **新判据的第 I 类错误率仍失控（零假设下约 `96.7%` 误拒）**：
   - 对任意单个连续同分布指标，8 个参照与 3 个候选共 11 个独立样本中，3 个候选全部落在 8 个参照 `[min, max]` 内的充要条件是总体最小值与总体最大值同时属于 8 个参照，精确概率为 `(8 × 7) / (11 × 10) = 56/110 = 50.91%`。仅看 1 个指标就有近一半概率误杀等价配方。
   - 同时要求 8 个指标（24 格）全部落在各自极值内时：在本次全部 11 个真实 run（`s0..s2, c3..c7, p0..p2`）的 `C(11, 3) = 165` 种划分中，24 格全过的仅有 `5/165 = 3.03%`；按 8 个 C++ 种子的经验均值与 `8×8` 协方差矩阵做 `100,000` 次多元正态蒙特卡洛模拟，零假设下 24 格全过率仅 `3.31%`（`verify_all.log:4-6`）。
   - 本次未校准 `ckpt_best` 恰好落入该 `3.03%`，含两阶段条件抽样偏差（`run_step3_train.sh:159` 设 `RUN_CONTROL=auto`，仅当 3 参照字面判据不过时才补跑 `c3..c7`，单向拓宽极值区间）：其中 `c4`、`c7` 在第 11 轮早停，把 `val recall3` 极值范围从 3 参照的 `[0.455, 0.515]`（宽 `0.060`）扩至 `[0.364, 0.636]`（宽 `0.272`，放大 4.5 倍）。若在真实交付口径 `cal_swa_k10` 上套同一条 24 格极值判据，`p0` 因 `val macro_f1 = 0.4433` 高于 C++ 最大值 `0.4287` 反被判为 `23/24` 不过（`verify_all.log:21`）。
3. **新判据能分辨多大的配方差异（检验力与口径噪声错配）**：
   - 模拟给出单指标极值规则（3 候选全落入 `[min_8, max_8]`）在不同均值偏移 `d·σ` 下的拒绝率：`0.5σ` 为 `55.2%`（零假设基线已达 `49.1%`，几乎无净区分力），`1.0σ` 为 `69.4%`，达到 `80%` 检出力需偏移 `1.40σ`，达到 `90%` 检出力需 `1.80σ`（`verify_all.log:7-13`）。若改用控制双侧第 I 类错误率 `α = 0.05` 的双样本 `t` 检验（`n1 = 8, n2 = 3, df = 9`），`80%` 检验力的最小可检效应量（MDE）为 `2.13σ`（8 项 Bonferroni 校正后为 `3.04σ`）。
   - 致命点在于验收 (a) 评的是未平均、未校准的 `ckpt_best`。审核 H2 已证实 `e1-noise` 下 `ckpt_best` 的波动主要由「FWD 对 YAW」一维工作点漂移主导（本次 11 个 run 的最佳轮横跨第 11 至第 28 轮）。按 `distribution_check.md` 的 C++ 标准差换算绝对效应量：

     | 指标 | `ckpt_best` C++ `σ` | 极值规则 80% 检出 (`1.40σ`) | 双样本 `t` 80% MDE (`2.13σ`) | `cal_swa_k10` C++ `σ` | `cal_swa_k10` `t` 80% MDE (`2.13σ`) |
     | --- | --- | --- | --- | --- | --- |
     | `val macro_f1` | 0.015 | 0.021 | 0.032 | 0.0150 | 0.032 |
     | `val recall1` | 0.029 | 0.041 | 0.062 | 0.0041 | 0.009 |
     | `val recall2` | 0.040 | 0.056 | 0.085 | 0.0490 | 0.104 |
     | `val recall3` | 0.086 | 0.120 | 0.183 | 0.0442 | 0.094 |
     | `test macro_f1` | 0.009 | 0.013 | 0.019 | 0.0085 | 0.018 |
     | `test recall1` | 0.036 | 0.050 | 0.077 | 0.0099 | 0.021 |
     | `test recall2` | 0.064 | 0.090 | 0.136 | 0.0385 | 0.082 |
     | `test recall3` | 0.031 | 0.043 | 0.066 | 0.0243 | 0.052 |

     可见在 `ckpt_best` 口径下，单个转向类的召回率要系统性恶化 `9` 至 `18` 个百分点才会被 `8 vs 3` 可靠检出。
   - 另有一处统计量公式偏差：`proto/distribution_check.py:66,69` 将 `z` 定义为 `(mean_p - mean_c) / (sd_c / sqrt(3))`，分母把 8 个 C++ 种子的样本均值当成无误差真值，漏计了 `sd_c^2 / 8` 项，使分母偏小 `sqrt(11/8) = 1.1726` 倍，且未计 `df = 7` 的 `t` 分布尾部。

### 1.2 续训逐步对拍覆盖不到什么，以及更有力的离线等价性检查

1. **续训 24 步探针（`probe_divergence.py`）覆盖不到的范围**：
   - 它覆盖了 `CppXavier` 初值、`CppMt19937.shuffle` 跨轮状态、单步前反向算子、加权 CE、`FlatAdam` 在 `t = 65, 650, 1885` 的更新，以及全局梯度裁剪触发分支（本次实测第 1 轮 65 步中有 `5` 步 `norm > 5.0`、最大 `15.52`；`ep10` 续训 24 步中有 `4` 步 `norm > 5.0`、最大 `19.46`，见 `verify_all.log:33-34`）。
   - **覆盖不到的盲区**：① `Trainer.run()` 的外层 epoch 控制流（`probe_divergence.py:59` 的 `run_steps` 直接调 `train_step`，绕过了验证集 `evaluate`、`best_f1` 比较选模、CSV/JSONL 统计与导出落盘）；② 1,950 步全程的微小系统性偏差累积以及末 10 轮权重平均后的盆地几何与校准量 `δ`；③ 所有非 `e1-noise` 的开关（余弦学习率、显式类权重、条件输入、双帧、文本、辅助头、BN、EMA、logit adjustment）。
2. **更有力的两项离线等价性检查（已在现成 11 个 run 上实测通过）**：
   - **检查一：直接检验闭环实际使用的交付产物 `cal_swa_k10`（消除工作点漂移噪声）**。对全部 8 个 C++ run 与 3 个原型 run 做末 10 轮平均加 val FORWARD 校准后，`val recall3`、`test recall2`、`val recall1`、`test recall1` 的标准差分别缩小 `49%`、`40%`、`86%`、`73%`；8 C++ 对 3 原型的校准量 `δ` 为 `1.457 ± 0.209` 对 `1.491 ± 0.118`（`t = +0.26`），`test macro_f1` 为 `0.3962 ± 0.0085` 对 `0.3937 ± 0.0053`（`t = -0.47`），全部 9 项指标 `|t| <= 1.04`（双侧 `p > 0.32`，`verify_all.log:20-28`）。
   - **检查二：跨线对线内的逐帧预测与概率分布双样本 Energy 置换检验（免多重比较、精确控 `α`）**。在 `val + test` 共 `1,458` 帧上计算两两模型的不一致率矩阵 `D_ij`（及概率全变差距离矩阵 `TV_ij`），构造双样本 Energy 统计量 `2·E[D(C++, Proto)] - E[D(C++, C++)] - E[D(Proto, Proto)]`，在全部 `C(11, 3) = 165` 种标号置换下做精确单侧检验（若原型学到不同决策边界，跨线距离会大于线内距离）。实测 `cal_swa_k10` 下：1,458 帧逐帧一致率 C++ 内部 `0.9374`、原型内部 `0.9479`、**跨线 `0.9446`**（精确 `p = 0.7818`）；平均 TV 距离 C++ 内部 `0.0850`、原型内部 `0.0752`、**跨线 `0.0774`**（精确 `p = 0.8667`，`verify_all.log:29-30`）。
   - **结论**：主会话放行 E2E-P0 起跑闭环验收的决定站得住；建议废止任务卡中「候选落在 `>= 8` 个参照极值内」的后续规则，改用「交付口径下跨线逐帧预测 Energy 置换检验（`p >= 0.05`）加 `test macro_f1` 双样本均值差置信区间」。

---

## 2. 发现清单（按严重程度排序）

### L-1【高】验收 (a) 改立的「24 格全落在 8 个参照极值内」判据零假设误拒率达 96.7%，单指标在 `ckpt_best` 下漏检 12 至 18 个百分点偏差

- 位置：`wt/p12-e2e-p0/proto/distribution_check.py:55-75`（提交 `1c35122`）、`wt/p12-e2e-p0/proto/accept_offline.py:90-129`（提交 `583626a`）、`wt/p12-e2e-p0/proto/runs/run_step3_train.sh:159`（提交 `583626a`）、`docs/briefs/2026-11-10-phase12-task-cards.md:856-858`。
- 现象：字面判据未过后，任务卡 11-12 07:00 将离线等价判据改为「候选 3 个种子的每项指标落在至少 8 个参照种子的最小到最大之间」，并写明以后同类对拍照此执行；`distribution_check.py:66,69` 另以 `sd_c / sqrt(len(p))` 作标准误计算 `z`。
- 为什么是缺陷：如第 1 节推导与实测，8 对 3 的 24 格极值盒在完全等价配方下的拒绝率高达 `96.7%`，以后照此执行会导致几乎所有对拍卡假阳性停工；同时因评估对象选了未校准的 `ckpt_best` 而非交付权重 `cal_swa_k10`，转向召回被工作点漂移噪声淹没，单指标偏移小于 `12` 至 `18` 个百分点时无法可靠检出；`RUN_CONTROL=auto` 只在初判失败时追加参照种子，构成单向放宽；`z` 分母漏了参照均值方差，偏小 `1.1726` 倍。本次 E2E-P0 的实质等价性虽经我们补做的 `cal_swa_k10` 置换检验确认无误，但规则与工具若保留原样会误导后续移植对拍（如 E2E-P1 胜出结构回迁 C++ 时的对拍卡）。
- 复现与验证：`nice -n 15 /root/rivermind-data/uav_rgb/teacher-venv/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-l/verify_all.py`（日志 `verify_all.log:3-30`）。
- 建议的最小修法：① 修改 `docs/briefs/2026-11-10-phase12-task-cards.md:858`，废除「落在 8 个参照极值内」的通用条款，改为在交付口径（末 K 轮平均加校准，或稳定配方 R2e）上做 `val+test` 逐帧预测双样本 Energy 置换检验（`C(m+k, k)` 划分，`p >= 0.05`）；② `distribution_check.py:66` 的双样本标准误改为合并标准差乘 `sqrt(1/len(p) + 1/len(c))`。
- 状态：**已复现**。
- 建议归属：修订任务卡公共条款，代码修改并入 `proto/` 合入 `main` 前的收尾提交。

### L-2【高】E2E-P1 地基延伸：含 BatchNorm 模型每轮先 `fold_bn()` 再做末 K 轮权重平均，且 EMA 权重混用在线末批 BN 统计量，致深层 logit 幅度缩水 67%

- 位置：地基 `wt/p12-e2e-p0/proto/train.py:173-182`（提交 `8b8283b`）；在途分支 `wt/p12-e2e-p1/proto/train.py:281-285, 498-506, 524-528, 639-646`（提交 `bcb2b07`、`63c1958`）与 `wt/p12-e2e-p1/proto/export.py:72-92`。
- 现象：地基 `flatten_params` 仅收集可训 `nn.Parameter`、不含模块 `buffers`。E2E-P1 加入带 BN 的基础结构（B1）与 5 层卷积列结构头 `ColHeadCnn`（B2、B3）后：
  1. `_save_compact_proto` 在每个 epoch 末先调 `self.model.fold_bn()` 把当轮 BN 折叠进卷积并把保存的 `arch["use_bn"]` 改为 `False`，训练结束时 `_finalize_compact_delivered` 再对末 10 轮已无 BN 层的权重直接做算术平均，未对平均权重重估 BN 统计量；
  2. 若同时启用 `--use-bn` 与 `--ema-decay`（如组合 A2 与 B2），评估与留档只把 `self.flat` 拷入 `ema.shadow`，而 `bn.running_mean` 与 `bn.running_var` 仍是在线未平均参数刚跑完末批留下的滑动值；
  3. `trunc` 训练集 `8205 % 128 = 13`（`round_3` 为 `17233 % 128 = 81`），每轮最后一步仅 13 个样本，在 PyTorch 默认 `momentum=0.1` 下与 128 样本大批次同占 `10%` 统计量权重，直接进入随后的 val 评估与 BN 折叠。
- 为什么是缺陷：不同轮次卷积权重线性平均后，输出特征方差严格小于单轮方差（`Var((Z1+Z2)/2) = (1+ρ)/2 · σ²`），经多层 ReLU 级联后激活尺度逐层衰减。标准 SWA 靠在平均参数上重跑训练集前向更新 BN 的 `running_mean/var` 来消除该收缩；先逐轮折叠再平均彻底丧失重标定能力，会让正在筛选的 B1、B2、B3（及胜出后进入 C 组的全部臂）交付模型 `delivered_raw.pt` 出现严重的幅度塌缩与工作点偏移，直接污染卡 E2E-P1 的离线选胜判决（`pick-b-winner` 主指标正是 `raw` 平均权重的 `macro_f1`）。
- 复现与验证：见 `verify_all.py` 第 5c 节（`verify_all.log:51`）：在真实 `trunc` 数据上训 5 层 `ColHeadCnn(use_bn=True)` 取 10 个检查点，「先逐轮 `fold_bn()` 再平均」在 val 574 帧上的 `logit_std = 0.6816`，标准 SWA（「先平均未折叠参数、在训练集精确重算 BN 均值方差再 `fold_bn()`」）的 `logit_std = 2.0811`，幅度缩水至 `32.8%`，最大 `|Δlogit| = 5.50` nat，预测翻转 `3.66%`。
- 建议的最小修法：`epNN.pt` 保存未折叠的原始 `state_dict`（`arch["use_bn"]=True`）；在 `_finalize_compact_delivered` 平均末 K 轮参数后（或 EMA 拷入 `shadow` 后），若 `model.use_bn` 为真，将所有 BN 层 `reset_running_stats()` 并设 `momentum=None`，在 `train_idx` 上 `with torch.no_grad()` 跑一遍前向精确统计全局均值与方差，最后再调 `fold_bn()` 写出交付文件。
- 状态：**已复现**。
- 建议归属：**卡 E2E-P1（高优，在 `run_p1_screen.sh` 正式出数前修）**。

### L-3【高】E2E-P1 地基延伸：包装器与主线闭环在三处接口断裂（`prev_dim > 4` 被拒、`proto_ckpt/` 绕空条件闸门、无集边界与双预热污染有状态模型）

- 位置：`wt/p12-e2e-p0/proto/infer_wrap.py:89-91, 118-128`（提交 `e9489d4`）、`wt/p12-e2e-p1/proto/infer_wrap.py:100-112`（提交 `8d21245`）、`wt/p12-e2e-p1/proto/train.py:501, 786`（提交 `bcb2b07`、`7d4ed7c`）；对照主线 `sim/tools/student.py:76-78, 89-105`、`sim/tools/nav_cond.py:94-151`、`sim/3dgs/eval_3dgs.py:265`。
- 现象：
  1. **条件维度白名单与编码缺失**：E2E-P1 的 C2、C3（`prev_dim = 9`）与 C4（`prev_dim = 10`）经 `infer_wrap.py` 在 `hello` 报出 `"prev_action_dim": 9` 或 `10`；主线 `sim/tools/student.py:77` 写死 `if self.prev_dim not in (0, 4): self._abort_start(...)`，闭环启动即失败；主线 `collect.py --student-cond` 亦无 9 档净转向数 `n9` 编码选项。此外 `wt/p12-e2e-p1/proto/train.py:786` 在 `--cond-mode aux`（4 值码）开启 `--cond-drop-prob > 0` 时，将 `prev_dim` 误赋为 `p1cd.CODE9_DROP_DIM`（`10` 而非 `5`）。
  2. **`proto_ckpt/` 子目录使 `nav_cond` 一致性闸门失效**：E2E-P1 将交付权重存为 `<run>/proto_ckpt/delivered_cal.pt`，而 `config.json` 在父目录 `<run>/config.json`，且未写 `average_ckpt.py` 风格的 `<ckpt>.json` 侧车。实测 `nav_cond.check_student_cond("heading-code", "<run>/proto_ckpt/delivered_cal.pt", 4)` 返回 `status="unverified"`，即使评测 C1 三值码模型也无法触发错喂拦截。同时 `p12-e2e-p0` 与 `p12-e2e-p1` 基于旧 `main` `97c4228`，分支内尚未合入 W1-9（`d816d2c`）的 `--student-cond`。
  3. **行协议无集边界且双重全零预热**：`infer_wrap.py:127` 在发 `hello` 前跑 1 次全零 `self.step`，`StudentPolicy._warmup()` 在收 `hello` 后再发 1 次全零 payload；且 `replay_compare.py:136` 与 `eval_3dgs.py:265` 跨多集复用同一个 `StudentPolicy` 子进程，`reset()` 仅清客户端缓存，不通知子进程。
- 为什么是缺陷：第 1、2 条会让 E2E-P1 第一轮选出的条件臂无法进入第二轮闭环或失去防呆闸门；第 3 条在后续按计划接入列特征缓存或 GRU 循环状态时，会让每集第 0 步携带两帧全黑预热的残留隐状态，并在 `replay_compare` 与 `eval_3dgs` 中发生跨集隐状态泄漏。
- 复现与验证：`verify_all.log:49` 已复现 `nav_cond` 返回 `unverified`；`student.py:77` 与 `p1` `train.py:786` 读代码确认。
- 建议的最小修法：① `p1` `train.py:786` 改为 `p1cd.HC3_DIM + 1 if a.cond_drop_prob > 0.0 else p1cd.HC3_DIM`；② `save_proto_model` 写交付权重时同步写 `<ckpt>.json`（含 `"run": os.path.abspath(self.run_dir)`），或在 `nav_cond.find_train_config` 中增查父目录；③ 第二轮闭环前合入最新 `main`，并在卡 E2E-C1 放开 `student.py` 的 `prev_dim` 限制与接入 `n9` 条件源；④ 接入有状态模型前，`Engine` 预热后显式调 `model.reset_state()`，并在行协议增加集首重置标志位。
- 状态：`nav_cond` 绕空**已复现**；其余**读代码确认**。
- 建议归属：卡 E2E-P1（修 `train.py:786` 与 `<ckpt>.json`）及卡 E2E-C1（主线客户端协议扩展）。

### L-4【中高】`proto/data.py` 缺失 `.aux` 侧车 SHA256 绑定校验与诊断性 split 门禁

- 位置：`wt/p12-e2e-p0/proto/data.py:63-93, 135-149`（提交 `ea519d8`）；对照 `src/aux_sidecar.cpp:48-109`、`src/dataset.cpp:176-211, 348-461`、`src/trainer.cpp:411-415`。
- 现象：
  1. `read_aux(uavr_path)` 只检查 8 字节 magic 与 payload 能否读出 `5*n` 字节，未校验头内 32B `uavr_sha256`（`hdr[28:60]`）、32B `payload_sha256`（`hdr[60:92]`）、文件总长、`n == ds.n`、`prev_index < n` 及 `prev_action` 范围；`test_data_metrics.py:114` 亦用全零 SHA256 构造用例。
  2. `Uavr` 不拦截 `diag*` 或 `map-hash:*` 诊断性划分（如卡 S1a 新增的 `diag-train-scene`），不核对 `split_policy` 与 `split_scheme` 一致，也不检查同目录是否存在 `instructions.jsonl` 语言侧车。
- 为什么是缺陷：主线 C++ 将 `.uavr` 与 `.aux` 视为信任边界以防异源侧车错配与场景泄漏。在 E2E-P1 的 C1 臂（`--cond-mode aux`）与 `parity_forward.py` 中直接调用 `read_aux`，一旦数据集重打包而旧 `.aux` 残留或拷错，`proto` 会静默用错位的条件码或上一帧训练而无任何报错。
- 复现与验证：`verify_all.log:37-38`：对 `ds.n=18` 的数据集配上 `aux.n=5`、伪造 SHA256、`prev_index=99999`、`prev_action=200` 的坏 `.aux`，`read_aux` 正常返回；对 `split_scheme="diag-train-scene"` 的 `.uavr`，`Uavr` 正常加载。
- 建议的最小修法：`read_aux(uavr_path, *, ds=None, max_prev_action=4)` 补齐 `n`、`uavr_sha256`、`payload_sha256`、文件精确长度与下标越界检查；`Uavr.__init__` 默认拒绝 `diag*` 与 `map-hash:*` 划分（加 `allow_diagnostic_split=False` 开关），并在存在 `instructions.jsonl` 且未显式声明纯视觉时报错。
- 状态：**已复现**。
- 建议归属：卡 E2E-P1 或 `proto/` 合入 `main` 前补齐。

### L-5【中】`proto/train.py` 对环境变量 `DATA_ROOT` 的定义与 `env.sh` 及启动器相反，`source env.sh` 后默认路径失效

- 位置：`wt/p12-e2e-p0/proto/train.py:48-50`（提交 `8b8283b`，`wt/p12-e2e-p1/proto/train.py:50-52` 同）；对照 `env.sh:10`、`proto/uavrgb_infer_proto:7`、`proto/test_train.py:38`。
- 现象：主仓库 `env.sh:10` 导出 `DATA_ROOT=/root/rivermind-data`（不含 `/uav_rgb`），启动器 `uavrgb_infer_proto` 与全部 `proto/test_*.py` 均按 `${DATA_ROOT}/uav_rgb/...` 拼接；唯独 `proto/train.py:48` 写成 `DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data/uav_rgb")` 并在后面直接拼 `/tmp/...` 与 `/exp-proto-parity-20261112/...`。
- 为什么是缺陷：按 `CLAUDE.md` 规范先 `source env.sh` 后，执行 `python proto/train.py --seed 0` 会因去找 `/root/rivermind-data/tmp/...` 而报文件不存在；若传了 `--data` 未传 `--out`，产物会越界写到 `/root/rivermind-data/exp-...`（落在项目数据目录 `/root/rivermind-data/uav_rgb/` 之外）。反之若按 `train.py` 设 `DATA_ROOT=/root/rivermind-data/uav_rgb`，`uavrgb_infer_proto` 会去找 `.../uav_rgb/uav_rgb/teacher-venv/bin/python` 并报错退出。
- 复现与验证：`bash -c 'source /root/claude-code-backup-20261028/code/uav_rgb/env.sh && PYTHONDONTWRITEBYTECODE=1 /root/rivermind-data/uav_rgb/teacher-venv/bin/python -c "import os,sys; sys.path.insert(0, \"/root/claude-code-backup-20261028/code/wt/p12-e2e-p0/proto\"); import train; print(train.DATA, os.path.exists(train.DATA))"'`，输出 `/root/rivermind-data/tmp/agent6-distill/trunc/ds/dataset.uavr False`（`verify_all.log:40`）。
- 建议的最小修法：`proto/train.py` 改为 `DATA_DIR = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")`，后续常量基于 `DATA_DIR` 拼接。
- 状态：**已复现**。
- 建议归属：卡 E2E-P1 顺手修正，`p12-e2e-p0` 等闭环跑完合并时修正。

### L-6【中】对拍测试与门禁存在四处盲区：续训探针绕过 `Trainer.run()`、新配方缺逐步对拍、变体未覆盖首帧与 `prev_dim != 4`、`check.sh` 未纳管

- 位置：`wt/p12-e2e-p0/proto/probe_divergence.py:59-67`（提交 `93549fd`）、`wt/p12-e2e-p0/proto/parity_forward.py:130-198`（提交 `ea519d8`）、`wt/p12-e2e-p0/proto/test_forward_parity.py:41`、`wt/p12-e2e-p1/proto/test_p1_recipe.py`、`tools/check.sh`。
- 现象：
  1. `probe_divergence.py` 绕过 `Trainer.run()`，不测验证评估、选模与导出（E2E-P1 新增的 `--compact-save` 交付分支连 `crosscheck_evalonly.py` 也不经过）；
  2. E2E-P1 新实现的 `--cosine-lr` 与显式类权重 `1e-6,1,1,1`（臂 A1）未与 C++ 在卡 E2E-R1 已跑完的同数据同种子 `R2e` run 做前几步 loss 逐位对拍，仅设了宏观稳定性闸门 P1（翻转率 `<= 0.03`、一致率 `>= 0.90`，容得下调度器步号差 1 或权重归一化差）；
  3. `parity_forward.py` 的 `variants` 双帧用随机抽样相邻项而非侧车 `prev_index`（漏测集首帧复制自身），未用数据盘已有的真实 C++ 双帧加 prev-action 权重 `run-tp10-two_prev_depth`，文本未测序列内部夹 `<pad>=0`，条件未测 `prev_dim != 4`；
  4. `proto/test_*.py` 不进 `tools/check.sh`，且不设 `UAVRGB_PROTO_REAL=1` 时 `test_forward_parity.py` 全部跳过仍显示 `OK`。
- 为什么是缺陷：后续在 `proto/` 迭代配方或改保存逻辑时，单测全绿挡不住 `step_lr` 步号偏移 1 步、`compact_save` 选错轮次或 `prev_dim > 4` 序列化不一致。
- 复现与验证：读代码确认；E2E-P0 工作日志 `2026-11-12-0215-proto-parity.md:59-61` 记录了默认跳过数与 `check.sh` 未收集的事实。
- 建议的最小修法：① 在 E2E-P1 给 A1（seed 0）补一条对 `$D/exp-recipe-stability-20261112/` 中 C++ `R2e` seed 0 前 5 步 `train_step` loss 的 `1e-5` 对拍；② `variants` 增加 `prev_dim=9`、首帧 `prev == cur`、中间含 `0` 的文本 ids；③ 合入 `main` 时在 `tools/check.sh` 增加 `proto` 合成单测段。
- 状态：**读代码确认**。
- 建议归属：卡 E2E-P1（第 ① 条）与 `proto/` 合入 `main` 的合并卡（第 ②③ 条）。

### L-7【中】E2E-P1 条件输入臂（C1 至 C4）的校准列仍采用单一全局标量 `delta`，存在与审核 H1 相同的回转压制混淆

- 位置：`wt/p12-e2e-p1/proto/train.py:537-550`（提交 `bcb2b07`）、`wt/p12-e2e-p1/proto/p1_metrics.py:26-41`（提交 `d46603d`）；对照 `docs/briefs/2026-11-10-phase12-task-cards.md:900-908, 915`。
- 现象：任务卡 E2E-P1 候选 3 写明「校准按码分开」，`p1` 分支在 `_finalize_compact_delivered` 中对所有条件臂仍对整份 val 求单一标量 `delta` 加到 `fc.bias[1]` 上，写出 `delivered_cal.pt`。
- 为什么是缺陷：虽然 `run_p1_screen.sh:20` 将离线主读法定在 `raw`（R2e 配方无 `auto` 类权重引入的 `+2.38` nat 偏移，`raw` 本身接近自然先验），但并列汇报的 `cal` 指标以及若第二轮闭环误取 `delivered_cal.pt`，在非零码（需纠偏回转）帧上仍会受到由 `n=0` 帧决定的 `delta` 干扰。
- 复现与验证：读代码确认（与第一轮审核 H1 同机理）。
- 建议的最小修法：`p1_summarize.py` 对 C1 至 C4 增加按 `n=0` 与 `|n|>=2` 分桶的 `raw`/`cal` 转向预测占比诊断；送闭环时按卡 E2E-1b 的结论选择 `delivered_raw.pt` 或按码校准。
- 状态：**读代码确认**。
- 建议归属：卡 E2E-P1 与卡 E2E-1b。

### L-8【中低】`closedloop_summary.py` 缺按 20 个几何聚类的配对检验，`proto/runs/*.sh` 未记录 `ALLOW_OFFLINE_FAIL` 等覆盖变量且沿用裸 PID 探活

- 位置：`wt/p12-e2e-p0/proto/closedloop_summary.py:80-88, 120-136`（提交 `320f9eb`）、`wt/p12-e2e-p0/proto/runs/common.sh:32`（提交 `e9489d4`）、`wt/p12-e2e-p0/proto/runs/run_step3_closedloop.sh:59, 91-93, 163-168`（提交 `f5b092f`）。
- 现象：
  1. 公共规则 §2（11-12 修订，审核 D2）要求 60 集比较按 20 个几何聚类报符号翻转精确检验与簇 bootstrap 区间；`closedloop_summary.py` 仅按单个训练种子分别算 3 个 `n=20` 的 `mcnemar_exact`，无 20 几何聚类检验与总配对表。
  2. `run_step3_closedloop.sh` 被 `ALLOW_OFFLINE_FAIL=1` 放行时，`inputs.txt`、`start.txt` 与 `closedloop.json` 均不记录 `ALLOW_OFFLINE_FAIL` 及 `SEEDS`、`LAST_K`、`EVAL` 等覆盖变量，正式运行也不拦截演练覆盖变量（审核 E-10）。
  3. `common.sh:32` 的 `run.pid` 仅写裸 PID，`run_step3_closedloop.sh:91` 用 `kill -0` 探活，容器重启后若 PID 被复用会死等（审核 E-4；注：`p1` 分支新建的 `run_p1_screen.sh:80-85, 110-114` 已按新规写全，但继承过来的 `common.sh` 与三个 `run_step*.sh` 未更新）。
- 为什么是缺陷：验收 (b) 跑完出的 `closedloop.md` 缺少主比较要求的聚类 p 值与越过离线闸门的持久元数据留痕。
- 复现与验证：读代码确认。
- 建议的最小修法：合入 `main` 后让 `closedloop_summary.py` 调用 `tools/paired_stats.py` 补报按几何聚类的符号翻转检验与区间，并在汇总 JSON 中记入 `offline_verdict` 与覆盖变量；`common.sh` 同步为带 `lstart` 与 `pgid` 的格式。
- 状态：**读代码确认**。
- 建议归属：`proto/` 合并时对齐卡 W1-14 与 W1-15a（当前闭环排队期不动 `wt/p12-e2e-p0`）。

### L-9【低】`proto/compare_episodes.py` 对 R6 迟到重跑目录的解析与 `closedloop_summary.py` 不一致

- 位置：`wt/p12-e2e-p0/proto/compare_episodes.py:52-58, 134-151`（提交 `e42a268`）；对照 `wt/p12-e2e-p0/proto/closedloop_summary.py:52-70`。
- 现象：`compare_episodes.py` 对参照臂 `ref_dir` 不查 `<ref>-retry1`；对被测臂只要 `<arm>-retry1` 下有 `actions.jsonl` 就无条件采用，不核对首次 `eval.json` 该 seed 是否真的在 `late_dispatch_episodes` 内，且当重跑仍迟到（落入 `retry1/eval.json` 的 `failed` 而不在 `per_episode`）时仍读取该废集并把结局显示为普通 `"未成功"`。
- 为什么是缺陷：本次第二关参照臂 `acc-L0.20-k1-s1` 与两个被测臂恰好均无迟到集，已出数字不受影响；但若复用到含迟到重跑的臂上，会读入作废轨迹。
- 复现与验证：读代码确认。
- 建议的最小修法：复用 `closedloop_summary.first_pass` 的 R6 规则，对 `ref` 与 `arm` 同等解析并在重跑仍迟到时标注 `迟到`。
- 状态：**读代码确认**。
- 建议归属：`proto/` 合入 `main` 后的清理。

### L-10【低】`proto/ckpt_io.py` 写端漏检非法类权重与 `best_macro_f1`，`proto/train.py` 的 `test_eval.jsonl` 缺 `run_id`

- 位置：`wt/p12-e2e-p0/proto/ckpt_io.py:182-197, 257-285`（提交 `ea519d8`）；`wt/p12-e2e-p0/proto/train.py:398-405` 对照 `src/trainer.cpp:341-351`。
- 现象：`encode_checkpoint` 写出前只校验 `params`、`adam_m`、`adam_v` 有限，未校验 `class_weights >= 0` 且有限、`best_macro_f1 >= -1.0` 且有限、以及 `contract` 内 `lr`/`distill_alpha` 有限，导致可写出自己与 C++ 都读不回来的 `.bin` 文件；`train.py` 写的 `uavrgb.test_eval.v1` 比 C++ 少 `"run_id"` 字段。
- 为什么是缺陷：写端信任边界不闭合，外部脚本构造 `Ckpt` 传入负权重或 `NaN` 指标时落盘不报错，延迟到下游加载才崩。
- 复现与验证：`verify_all.log:39`：构造 `class_weights=[-1,1,1,1]` 与 `best_macro_f1=nan`，`write_checkpoint` 成功写出，`read_checkpoint` 抛 `CkptError: checkpoint: class_weights 含非有限/负值`。
- 建议的最小修法：`encode_checkpoint` 增加与 `_read_mid` 对称的四项校验；`test_eval.jsonl` 补齐 `"run_id"`。
- 状态：**已复现**。
- 建议归属：`proto/` 合入 `main` 前的小修。

---

## 3. 第一轮清单相关条目核对（角度 C 与角度 H）

核对基准：`main` @ `ada0ca5`，并对照在途分支 `p12-e2e-1`、`p12-e2e-r1`、`p12-c1a`、`p12-e2e-p0`、`p12-e2e-p1`。

1. **修对了的（已合入 `main` `ada0ca5`）**：
   - `C2 / G1 / F1`（3DGS 线给 prev-action 学生喂模型自身输出而非实际执行动作）：卡 S1a（提交 `32caa35`，合并 `5387747`）已在 `sim/3dgs/collect_3dgs.py`、`guard.py` 修复，DAgger 与护栏改写后经 `note_executed` 回写，配 14 条测试。
   - `F4 / F5`（翻转增强互换与双帧 `prev_index` 缺带参考值的 C++ 测试）：卡 T1 第二批（提交 `32d0892`，合并 `b706b2f`）已入库 `test_flip_augment.cpp` 与 `test_two_frame_prev_index.cpp`。
   - `D2 / D4 / D5 / D7`（按几何聚类统计检验与 `stats_report` 分母）：卡 W1-14（合并 `0d7f3b5`）已在 `main` 入库 `tools/paired_stats.py`，但未同步到 `p12-e2e-p0` 的 `closedloop_summary.py`（见 `L-8`）。
2. **修了一半的（在分支上或仅覆盖部分情形，未闭环）**：
   - `C1 / D6 / H1`（`average_ckpt.py` 不支持条件/双帧/文本，且单一全局 `delta` 压制条件回转）：`main` 的 `tools/average_ckpt.py:210` 仍拒绝非 `(3, 0, 0)` 模型；分支 `p12-e2e-1` 的 `dc8e5ff` 仅补了单帧 `prev_dim == 4` 的全局校准（未合入 `main`），双帧、文本与 `prev_dim > 4` 仍不支持；按码校准（卡 E2E-1b）尚未实现，`p12-e2e-p1` 仍为全局 `delta`（见 `L-7`）。
   - `H2 / H3 / H6`（工作点漂移、恒定学习率、类权重 `auto`）：卡 E2E-R1 离线部分已在 C++ 证实 `R2e`（`--cosine-lr --class-weights 1e-6,1,1,1`）将翻转率压到 `0.008`、三种子一致率提到 `0.942`，闭环扫描在排队；`p12-e2e-p1` 已在 `proto/` 实现 EMA 与 logit adjustment，尚未实跑出数。
   - `H4 / H5 / H7`（列结构头、FiLM 前移加条件置空、BatchNorm）：`p12-e2e-p1` 已在 `proto/` 实现，但 BN 遇到 SWA/EMA 时存在先折后均的幅度收缩缺陷（见 `L-2`），且 `aux` 模式下 `cond_drop_prob` 的 `prev_dim` 误写为 10（见 `L-3`）。
   - `C6 / H8 / H9` 及上轮 `D2 / D3 / D6 / D9`（`--eval-only` 的 `drop_collision` 契约继承、增强固定种子、记录梯度范数、续训继承 `max_grad_norm`、lr/AdamW 测试、conv dx 差分检查）：`main` 上未变；分支 `p12-c1a` 已完成全部代码提交（`249d37f` 至 `f7e80b5`），正在排训练池锁做逐位回归验证。`p12-e2e-p1` 的 `cond_rng` 已随 `seed` 派生，未复犯 H8。
3. **没修的（等待卡 E2E-C1 / W1-13 / T1 剩余项）**：
   - `C3 / G7`：checkpoint 与推理 `hello` 仍无 `cond_kind` 与前处理契约；`student.py` 未传 `prev_action` 时仍静默回落 `_last_action`；C++ 与 `proto/train.py:324` 写进 `config.json` 的 `data` 仍保留命令行原样（可为相对路径）。
   - `C4 / H10`：C++ 训练器条件维仍绑死 `4`、翻转镜像绑死 `2↔3`、`student.py:77` 仍只认 `0` 或 `4`（导致 `L-3`）。
   - `C5`：`main` 与 `ProtoCnn` 的辅助头仍接在 FiLM 之后（`p12-e2e-p1` 的 `ColHeadCnn` 在 `film_pos="pool3"` 时改接到了未调制的 `p3` 上，但 `film_pos="conv2"` 时 `p3` 仍经受 `conv2` 的 FiLM 调制）。
   - `C7 / C8 / C9 / F2`：仍未修。

---

## 4. 查过但没有发现问题的地方

1. **`ProtoCnn` 前向算子、权重内存布局与分类解码**（`proto/model.py:43-163` 对照 `src/model.cpp`、`src/model_dev*.cu`、`src/infer.cpp`）：卷积 `[out,in,kh,kw]`、线性 `[out,in]`、NCHW 展平 `[c][h][w]`、`uint8` 按 `/ 255.0` float32 除法、FiLM 的 `(1+γ)·y + β` 通道切分、文本 `id==0 || id>=22` 掩零、float64 减最大值的 `softmax_probs`、平局取最小下标的 `first_argmax`，逐项与 C++ 对齐；真实 `cal_swa_s1_k10.bin` 2,000 帧最大概率差 `5.79e-06` 确由 C++ 侧 float32 累加顺序引起。
2. **libstdc++ 随机数流逐位复刻**（`proto/cpp_rng.py` 对照 GCC 13 `std::mt19937`、`uniform_real_distribution<float>`、`std::shuffle`）：`_legacy_seeding`、`generate_canonical<float,24>` 在 `1.0f` 处的 `nextafter` 截断、无 FMA 的两步线性映射、Lemire 区间抽样、`std::shuffle` 快慢双路径及奇偶首步分支、Xavier `fan_in/fan_out` 抽样顺序与单引擎跨轮持续状态，全部与 C++ 逐位一致。
3. **e1-noise 训练核与单步更新**（`proto/train.py:73-162, 258-413` 对照 `src/softmax_cuda.cu`、`src/nn_cuda.cu`、`src/trainer.cpp`）：`softmax_parts` 顺序 float32 求和、double 累加加权 CE、全局梯度裁剪 `max_norm / (float(norm) + 1e-6f)`（实测真实数据第 1 轮 5 步、`ep10` 续训 4 步触发裁剪，前 12 步 `|Δloss| <= 7.2e-07`）、`FlatAdam` 的 double `pow` 偏置修正转 float32 更新、末批 13 样本保留、每轮严格大于更新 `best_f1`、末尾从导出的 `ckpt_best.bin` 重载评 `test`，均与 C++ 一致。
4. **`ckpt_io.py` 读端校验与原子写盘**（`proto/ckpt_io.py:164-254, 287-294`）：v6 头部与 113 B 契约结构、逐张量形状核对、NaN/Inf 拒收、尾部多余字节拒收、`.tmp` 原子替换，真实 checkpoint 读写往返逐字节相同，导出产物通过 `tools/average_ckpt.py` 与 `uavrgb_train --eval-only` 校验。
5. **行协议包装器的 stdout 隔离与大帧管道读取**（`proto/infer_wrap.py:74-86, 154-192`）：`protect_stdout()` 在导入 `torch` 前把 fd 1 重定向到 stderr，防止底层 C++/CUDA 告警（如 cuBLAS 初始化提示）污染协议 JSON；阻塞 `sys.stdin.buffer.read(payload)` 在超过 Linux 64 KiB 默认管道缓冲的双帧（115,200 B）输入下能循环读满不截断；`--text` 越界/超长报错时机与退出码（1 与 2）同 `src/infer_main.cpp`。
6. **E2E-P1 的净转向计数 `n` 与子集掩码对齐**（`wt/p12-e2e-p1/proto/p1_cond_data.py:36-154`）：保留打包丢弃的碰撞步来累计已执行转向数 `n`，与 `preprocess.py` 和 `audit_features.py` 一致；`student_mask` 在 `round_3` 上选出 `val = 675` 帧、`test = 925` 帧，与卡 E2E-0 `audit_fit.py` 的 `"student"` 子集帧数一致（含 `round_1` 中 β 混合由专家执行的帧，`val` 中占 `94/675 = 13.9%`，读数时与 E2E-0 保持同一限定即可）。

---

## 5. 没有查的地方与原因（附验证记录）

### 5.1 没做什么与原因

- 未修改主仓库、`wt/p12-e2e-p0`、`wt/p12-e2e-p1` 中的任何文件，未写入仓库 `docs/worklog/` 或 `progress.md`，未触碰任何运行中的进程或池锁：严格执行任务书第 3 节只读禁令与 `wt/p12-e2e-p0` 零字节冻结要求。
- 未启动 Gazebo、未跑整轮训练、未跑 `tools/check.sh`：任务书第 3 节明确禁止。
- 未核查验收 (b) 的闭环实测结果：截至 `08:19 CST`，`$D/exp-proto-parity-20261112/step3-closedloop/` 仍停留在 `02:47` 因 `literal=FAIL` 退出的 `rc=5` 记录，带 `ALLOW_OFFLINE_FAIL=1` 的闭环排队结果尚未产出。
- 未直接运行 `proto/test_forward_parity.py`（真实数据模式）：执行者记录其实测耗时 `151.8 s` 超过单测 2 分钟上限；本次已通过审查其源码与 `$X/step1-forward/*.json`，并在 `verify_all.py` 中对 11 个 run × 1,458 帧做了 GPU 前向核对。
- 未逐行审完 `wt/p12-e2e-p1` 工作区尚未提交脚本的后半段（`proto/p1_summarize.py` 第 151 行以后、`proto/runs/run_p1_screen.sh` 第 151 行以后）：另一位执行者正在编写，本次聚焦地基向 E2E-P1 延伸的结构性风险（数据读取、评估口径、导出、包装器）。

### 5.2 验证到哪一步

| 检查项 / 命令 | 结果 |
| --- | --- |
| `TMPDIR=.../review2-l/tmp PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=GPU-b544b42e-… nice -n 15 $D/teacher-venv/bin/python .../review2-l/verify_all.py` | `rc=0`（耗时 `73.6 s`，单进程，显存 `< 0.5 GB`）；产出统计模拟、`cal_swa_k10` 11 个 run 对拍与 1,458 帧 Energy 置换检验、梯度裁剪触发数、`data.py`/`ckpt_io.py`/`env.sh` 复现及 `ColHeadCnn` BN 折叠平均收缩实测（详见 `verify_all.log`） |
| `git -C /root/claude-code-backup-20261028/code/wt/p12-e2e-p0 status --short` 及查找 `__pycache__` | 输出为空，`0` 处修改、`0` 个未跟踪或字节码文件 |
| `git -C /root/claude-code-backup-20261028/code/uav_rgb status --short` | 未触碰主仓库任何文件 |
