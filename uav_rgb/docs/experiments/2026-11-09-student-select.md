# STUDENT-SELECT：训练噪声、选模判据、双帧反事实（2026-11-09）

- 任务书：`tmp/phase3-20261107/BRIEF_ALGO.md` §3 + `tmp/phase8-20261109/BRIEF.md` §6。worktree `tmp/wt-student-select`（基于 main 2899428，未提交）。
- 问题：DISTILL §2.4 说 `ckpt_best.bin`（val macro-F1 严格最大）是噪声序列上的幸运轮，§1.3 说空画面 YAW 率随轮数升高与闭环「乱飞」同向。本实验量化这两件事，给出显式选模规则；另回答「`--two-frame` 是不是捷径」。
- 口径：离线四件套按 `docs/EVAL_STANDARD.md` §1.1（val 无 SAFE → macro-F1 上界 0.75）；闭环由主会话排在 eval4 之后，本文只给 ckpt 路径与 sha。
- 产物根 `$E = $DATA_ROOT/uav_rgb/exp-student-select-20261109/`；断点 `tmp/agent10-student-select/PROGRESS.md`。

## 0. 预注册（动手前写，做完逐条对照；结果见 §4）

### E1 训练噪声（`tmp/agent6-distill/trunc/ds`，e1_pipeline.CFG：30 轮 / batch 128 / lr 1e-3 / class-weights auto / `--drop-collision`，seed 0/1/2，`--save-every-epoch`）

- P1.1：三个 seed 的 best val macro-F1 彼此差 ≥ 0.02，best epoch 互不相同。
- P1.2：best epoch 上的 L/R 召回差在 ≥ 2/3 个 seed 上 ≥ 0.10，且方向不一致。
- P1.3：seed 0 逐位复现 `trunc/run`（best ep28 mF1 0.425699595、test acc 0.796）—— 训练器确定性 + `--save-every-epoch` 不改训练数学。
- 证伪：三 seed best mF1 极差 < 0.01 且 L/R 召回差都 < 0.05 → 「best-ckpt 是幸运轮」不成立；P1.3 不成立 → 二进制/改动有问题，先停。

### E2 选模判据（`tools/select_ckpt.py` 对 E1 的 3×30 个逐轮 ckpt 跑 val）

- P2.1：val macro-F1 的 argmax 轮不是空画面 YAW 率最低的轮；空画面 YAW 率随轮数上升。
- P2.2：逐轮预测翻转率末 10 轮仍 ≥ 5%，翻转集中在 YAW 帧。
- 证伪：空画面 YAW 率逐轮平坦（极差 < 5 pp）且 argmax 轮同时是空画面 YAW 率最低轮 → 现判据够用；翻转率末 10 轮 < 2% → 「幸运轮」被证伪。

### E3 双帧反事实（0417 collection 重打包带 UAVRAUX1 → 单帧 / `--two-frame` 各一个 seed 0 → 改侧车 `prev_index` 只做推理）

- 三个反事实：`self`（上一帧 = 当前帧）、`next`（上一帧 = 下一帧，时间反演）、`shuffle`（同集内随机帧）。
- P3.1：双帧比单帧 val mF1 多 ≤ 0.04。
- P3.2（捷径假设 H1）：`self` 下双帧 mF1 掉回单帧水平（差 < 0.01），YAW_L 帧的 L:R 预测比从 ≥ 4:1 掉到 < 2:1；`next` 下 L:R 反转。
- 证伪 H1：`self` 下预测变化率 < 5% 且 ΔmF1 < 0.01；或 mF1 下降但 L:R 比不变、`next` 不反转。
- 自检：单帧 seed 0 必须逐位复现 0417 run（best ep14 mF1 0.396405438、test acc 0.690431535）；新 ds 的 dataset.uavr 与 0417 ds 相同。

## 1. 方法

- `uavrgb_train --save-every-epoch`：每轮另存 `<run>/ckpt_ep{NN}.bin`，与 `ckpt.bin` 同格式同内容（CTest `test_save_every_epoch` 核字节相同、默认关）。
- `tools/select_ckpt.py --run <run> --data <ds>/dataset.uavr --out <dir> --infer-bin uavrgb_infer`：直接读 `dataset.uavr` 的 val 样本（与训练器同一份像素），每个 `ckpt_epNN.bin` 起真实 `uavrgb_infer` 子进程逐帧推理（与闭环同一条输入路），输出 `epochs.csv/json`、`preds_epNN.npy`。空画面 = 样本回溯到 `source_dir/actions.jsonl` 的 `pose_before/yaw_before` + `scene.json` 的 pillars 后 `teacher.visible_pillar_side(...) is None`。自检：每轮 macro-F1 与 `metrics.jsonl` 该轮 validation 事件对账（差 > 1e-5 则 rc=1）。
- 选模规则 `select_epoch()`：硬约束 ① 空画面 YAW 率 ≤ 空画面标签 YAW 率 + 0.05，② |recall_L − recall_R| ≤ 0.15；满足者中取 ±1 轮平均 macro-F1 的 argmax，并列取靠后轮；约束集为空则放宽并在输出标 `relaxed_constraints`。
- `tools/probe_two_frame_cf.py --ds <ds+aux> --ckpt <two-frame> --baseline-ckpt <single-frame> --train-bin --infer-bin`：按三个反事实重写 `dataset.aux` 的 `prev_index`（`preprocess.write_aux_sidecar`，uavr 硬链接），逐帧预测 orig / self / next / shuffle，报变化率、四件套、YAW_L/YAW_R 帧的 L:R 预测比；原始侧车的 mF1 与 `uavrgb_train --eval-only` 对账。

## 2. 数据与配置

| 项 | 值 |
| --- | --- |
| E1/E2 数据 | `tmp/agent6-distill/trunc/ds/dataset.uavr`（n=9663，val 574，过线帧已删，无 aux） |
| E3 数据 | `$E/ds-0417-aux/`：0417 collection 300 集重打包 `--drop-collision`（n=11818，train 10032 / val 720 / test 1066，`dataset.aux` prev_index/prev_action，无 depth） |
| 训练二进制 | `tmp/agent10-student-select/bin/uavrgb_train-wt`（worktree 构建，sha 3c75d9d5…，只多 `--save-every-epoch`） |
| 推理二进制 | `tmp/agent10-student-select/bin/uavrgb_infer`（main `build/check`，sha f33bafbd…，与 eval3 同） |
| 训练配置 | `tools/e1_pipeline.py CFG`：30 轮 / batch 128 / lr 1e-3 / class-weights auto / `--drop-collision` |

## 3. 结果

### 3.1 E1 训练噪声（trunc/ds val，n=574：FWD 501 / YAW_L 40 / YAW_R 33；空画面帧 155，其标签 YAW 率 0.123）

| seed | 训练器 best 轮 | best val mF1 | bal_acc | recall L / R | L/R 差 | 空画面 YAW 率 | test acc | 末 10 轮 mF1 均值 ± sd | `select_epoch` → 轮 | 选中轮 mF1 / 空画面 YAW / L/R 差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 28 | 0.4257 | 0.442 | 0.425 / 0.455 | 0.030 | 0.045 | 0.7964 | 0.372 ± 0.035 | 20 | 0.4067 / 0.013 / 0.074 |
| 1 | 16 | 0.4068 | 0.434 | 0.350 / 0.515 | 0.165 | 0.019 | 0.8167 | 0.358 ± 0.026 | 11 | 0.3773 / 0.026 / 0.120 |
| 2 | 18 | 0.3938 | 0.439 | 0.450 / 0.485 | 0.035 | 0.097 | 0.7568 | 0.357 ± 0.029 | 29 | 0.3871 / 0.110 / 0.036 |

- 三个 seed 的 best mF1 极差 0.032，best 轮 28 / 16 / 18；test acc 极差 0.06。
- seed 0 与 `tmp/agent6-distill/trunc/run` 的 31 条 validation 事件逐值相同；`ckpt_best.bin` 的权重与 Adam m/v 逐 float 相同（文件 sha 不同只因旧 run 是 ckpt v5、现在写 v6）。`--save-every-epoch` 没有改训练数学。
- 每个 seed 406 s（606 样本/s，两个并行）。

### 3.2 E2 逐轮行为（每 seed 30 轮，`select_ckpt.py` 真实 `uavrgb_infer` 推理；与 metrics.jsonl 对账最大差 1.5e-8）

| seed | 空画面 YAW 率：极差 / 前 10 轮均值 → 末 10 轮均值 | argmax 轮的空画面 YAW 率 | 末 10 轮里空画面 YAW 率 > 标签率+0.05 的轮数 | L/R 差 > 0.15 的轮数（30 轮） | 末 10 轮逐轮预测翻转率 均值 / 最小 | 末 10 轮 YAW 标签帧翻转率均值 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.00 / 0.13 → 0.41 | 0.045 | 7 | 12 | 0.224 / 0.117 | 0.266 |
| 1 | 0.92 / 0.36 → 0.29 | 0.019 | 7 | 13 | 0.173 / 0.091 | 0.201 |
| 2 | 0.90 / 0.34 → 0.35 | 0.097 | 6 | 10 | 0.173 / 0.077 | 0.190 |

seed 0 逐轮（ep1…ep30）：

```
mF1        0.256 0.308 0.339 0.340 0.358 0.398 0.206 0.394 0.382 0.367 0.387 0.328 0.390 0.390 0.353 0.380 0.276 0.316 0.402 0.407 0.383 0.345 0.374 0.308 0.362 0.417 0.345 0.426 0.367 0.396
空画面YAW   0.00  0.00  0.02  0.00  0.00  0.00  1.00  0.02  0.05  0.21  0.01  0.55  0.01  0.01  0.57  0.23  0.95  0.47  0.14  0.01  0.20  0.64  0.32  0.85  0.70  0.03  0.75  0.05  0.39  0.15
L−R 召回   +0.10 −0.21 −0.07 −0.31 −0.28 −0.12 −0.30 −0.23 −0.13 −0.09 −0.07 +0.02 −0.08 −0.13 +0.11 −0.21 +0.09 −0.21 −0.07 −0.07 −0.13 −0.33 −0.05 −0.27 +0.15 −0.08 +0.26 −0.03 −0.16 −0.12
翻转率      —    0.18  0.19  0.10  0.08  0.13  0.59  0.58  0.10  0.11  0.14  0.24  0.25  0.08  0.22  0.30  0.39  0.29  0.22  0.12  0.12  0.22  0.22  0.28  0.30  0.26  0.27  0.27  0.16  0.15
```

- 空画面 YAW 率不是「随轮数上升」，是逐轮在 0 与 1 之间摆：相邻两轮 0.03 → 0.75 → 0.05（seed 0 ep26–28）。mF1 对这件事只部分敏感：seed 0 空画面 YAW 率 ≥ 0.75 的轮 mF1 0.21–0.35，0.2–0.6 的轮 mF1 0.32–0.38，比 argmax 低 0.04–0.11。
- 五个 run（3 seed + 下面 0417 单帧 / 双帧）的训练器 best 轮，空画面 YAW 率都在该 run 的低端：trunc 0.045 / 0.019 / 0.097（各自 30 轮的中位数 0.15 / 0.20 / 0.28），0417 单帧 0.27、双帧 0.26（30 轮里最低 / 次低，中位数 0.89 / 0.85）—— **训练器现有判据（val mF1 最大）已经在隐式地挑「空画面不乱转」的轮**，因为空画面帧的标签 75–88% 是 FWD，在上面乱转会同时拉低 FWD 召回和 YAW 精度。
- 训练末期相邻两轮仍有 17–22% 的 val 预测翻转（YAW 标签帧 19–27%）；任何单轮 argmax 都建立在这个噪声上。
- 0417 数据（val 720，空画面 287，标签 YAW 率 0.251；`$E/select/0417-{single,two}`）：单帧 30 轮里只有 ep14 一轮空画面 YAW 率 ≤ 0.30（0.27，正是训练器 best），其余 29 轮 0.44–1.00（22 轮 ≥ 0.75）；双帧只有 ep6（0.17，mF1 0.298）和 ep20（0.26，训练器 best）两轮 ≤ 0.30。DISTILL §1.3 的「ep14 26–32% → ep30 84–89%」是拿幸运轮比典型轮，不是趋势。

### 3.3 E3 双帧反事实（0417 重打包，seed 0；val n=720 / test n=1066）

单帧 run 与 `e1-pipeline-20261107-0417/run` 的 31 条 validation 事件逐值相同（best ep14 mF1 0.396405438，test acc 0.6904）。双帧 best ep20 mF1 0.418478091，test acc 0.7195。

| split | 条件（双帧模型除最后一行） | acc | macro_f1 | bal_acc | recall F / L / R | 预测变化率 全 / YAW 帧 | YAW_L 帧预测 L:R | YAW_R 帧预测 R:L |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | orig（真实上一帧） | 0.751 | **0.418** | 0.450 | 0.81 / 0.45 / 0.53 | — | 35:6 | 32:7 |
| val | self（上一帧 = 当前帧） | 0.682 | 0.349 | 0.360 | 0.77 / 0.26 / 0.42 | 0.214 / 0.307 | 20:7 | 25:8 |
| val | next（上一帧 = 下一帧） | 0.650 | 0.337 | 0.364 | 0.72 / 0.34 / 0.40 | 0.282 / 0.431 | 26:14 | 24:14 |
| val | shuffle（同集随机帧） | 0.578 | 0.307 | 0.359 | 0.62 / 0.30 / 0.52 | 0.360 / 0.343 | 23:13 | 31:5 |
| val | 单帧对照 | 0.711 | **0.396** | 0.428 | 0.77 / 0.49 / 0.45 | — | 38:5 | 27:8 |
| test | orig | 0.720 | 0.388 | 0.409 | 0.80 / 0.41 / 0.42 | — | 44:20 | 48:16 |
| test | self | 0.649 | 0.336 | 0.345 | 0.73 / 0.30 / 0.35 | 0.257 / 0.364 | 32:14 | 39:25 |
| test | next | 0.659 | 0.363 | 0.396 | 0.72 / 0.33 / 0.54 | 0.296 / 0.464 | 35:34 | 61:22 |
| test | shuffle | 0.518 | 0.277 | 0.308 | 0.57 / 0.33 / 0.34 | 0.424 / 0.450 | 35:21 | 38:18 |
| test | 单帧对照 | 0.690 | 0.369 | 0.394 | 0.77 / 0.43 / 0.38 | — | 46:12 | 43:30 |

每个反事实侧车的 macro-F1 与 `uavrgb_train --eval-only` 对账，最大差 1.4e-8（`probe.json: eval_only_check`）。

按「上一步动作 vs 当前标签」拆帧（`$E/e3-two-frame/probe/transition_analysis.json`；延续帧 = 标签 == 上一步，切换帧 = 标签 != 上一步）：

| split | 模型 | 延续帧 acc | 切换帧 acc | 起转帧（FWD→YAW）acc | 停转帧（YAW→FWD）acc | 预测 == 上一步动作 的比例 |
| --- | --- | --- | --- | --- | --- | --- |
| val（延续 519 / 切换 183） | 单帧 | 0.803 | 0.421 | 0.421 (n=95) | 0.422 (n=83) | 0.717 |
| val | 双帧 orig | 0.850 | 0.448 | 0.421 | 0.482 | 0.752 |
| val | 双帧 self | 0.796 | 0.328 | 0.263 | 0.422 | 0.728 |
| val | 双帧 next | 0.701 | 0.475 | 0.379 | 0.590 | 0.615 |
| test（延续 742 / 切换 295） | 单帧 | 0.807 | 0.373 | 0.324 (n=148) | 0.432 (n=132) | 0.717 |
| test | 双帧 orig | 0.836 | 0.403 | 0.338 | 0.477 | 0.738 |
| test | 双帧 self | 0.768 | 0.319 | 0.277 | 0.394 | 0.698 |
| test | 双帧 next | 0.739 | 0.427 | 0.378 | 0.500 | 0.639 |

「照抄上一步动作」这条规则本身 acc = 0.739（val）/ 0.716（test），高于单帧 CNN 的 0.704 / 0.684。

## 4. 预注册对照

| 预测 | 结果 | 判定 |
| --- | --- | --- |
| P1.1 三 seed best mF1 差 ≥ 0.02、best 轮互不相同 | 极差 0.032；28 / 16 / 18 | 成立 |
| P1.2 best 轮 L/R 差 ≥ 0.10 于 ≥ 2/3 seed 且方向不一致 | 0.030 / 0.165 / 0.035，只有 1/3；三个都 R > L | **不成立**。mF1 的 argmax 本身偏向 L/R 平衡的轮（两类召回都高才有高 mF1）；不平衡出现在其它轮：30 轮里 10–13 轮 L/R 差 > 0.15，方向两边都有 |
| P1.3 seed 0 逐位复现 trunc/run | 31 条 validation 逐值相同，权重 / Adam 逐 float 相同 | 成立 |
| E1 证伪条件（极差 < 0.01 且 L/R 差 < 0.05） | 未触发 | — |
| P2.1 argmax 轮不是空画面 YAW 率最低轮；空画面 YAW 率随轮数上升 | 按字面 argmax 轮不是最低轮（最低轮是 ep1 之类全预测 FWD 的退化轮），但 argmax 轮的空画面 YAW 率都在各 run 低端（0417 是 30 轮里最低 / 次低）；无上升趋势，是 0 ↔ 1 的逐轮摆动 | **实质不成立**（第一句只在字面上成立，第二句不成立；见 §3.2） |
| P2.2 末 10 轮翻转率 ≥ 5%，集中在 YAW 帧 | 均值 0.17–0.22，最低 0.08；YAW 标签帧 0.19–0.27 vs 全体 0.17–0.22 | 成立，幅度是预测的 3–4 倍 |
| E2 证伪条件（空画面 YAW 率平坦且 argmax = 最低轮 / 翻转 < 2%） | 第一句半触发（argmax = 最低轮，但不平坦）；翻转远 > 2% | 「幸运轮」没有被证伪，但幸运的内容和预想不同：不是 mF1 幸运，是空画面行为幸运 |
| P3.1 双帧增益 ≤ 0.04 | +0.022（val mF1），test acc +0.029 | 成立；增益小于 E1 的 seed 间极差 0.032 |
| P3.2 / H1 `self` 掉回单帧水平（差 < 0.01）、YAW_L 帧 L:R 从 ≥ 4:1 到 < 2:1；`next` 反转 | `self` 掉到单帧以下 0.048（0.349 vs 0.396）；L:R 5.8:1 → 2.9:1；`next` 不反转（26:14、24:14 仍是正确方向占多数） | **不成立**：双帧不是「从光流读上一步方向再复读」 |
| H1 证伪条件（变化率 < 5% 且 ΔmF1 < 0.01；或 mF1 降但 L:R 不变、`next` 不反转） | 变化率 21–36%，ΔmF1 −0.07～−0.11；L:R 比有下降；`next` 不反转 | 第二条大体触发：模型用上一帧做的是别的事 |
| E3 自检（单帧复现 0417 run；dataset.uavr sha 相同） | 复现逐值相同；sha 不同（meta 多了 preprocess 11-07 后新增的字段与 `created_utc`），images / labels / collision_flags / 300 集 split 逐位相同 | 第一句成立；第二句按字面不成立，按意图成立 |

## 5. 结论与建议

**双帧是不是捷径**：不是 prev-action 那种（读出上一步方向再复读）—— 时间反演不翻转 L/R，增益也不是只落在延续帧（切换帧 +0.03，停转帧 +0.05～0.06，起转帧 ±0）。但它对「正常的相邻帧对」依赖很重：上一帧一换，21–36% 的预测改变，mF1 掉 0.07–0.11，`self`（零运动）时比单帧还差 0.05 且 YAW 召回掉到 0.26；预测与上一步动作一致的比例从 0.717 升到 0.752（轻微复读倾向）；+0.022 的离线增益小于 seed 间极差 0.032。闭环里悬停 / 起步 / 执行器没按指令动时帧对就是 `self` 那种状态。**建议：不进流水线默认配置**；若主会话想拿闭环裁决，`$E/e3-two-frame/two/run/ckpt_best.bin` 与同数据同 seed 的单帧 `$E/e3-two-frame/single/run/ckpt_best.bin` 可以直接排 A/B。

**选模判据（PLAN §8.1）**：

1. val 规模是第一问题：18 集 / 574 样本，YAW_L 40、YAW_R 33 —— 一个样本就是 0.025–0.03 的召回；L/R 差在 p≈0.45 下的二项 sd ≈ 0.12，观察到的 0.03–0.33 大多在噪声里；mF1 轮间 sd 0.03，seed 间极差 0.03。建议 val 提到 15–20% 的几何（≥ 45 集，每个 YAW 类 ≥ 150 样本，L/R 差 sd ≈ 0.06）；split 规则本轮未改。
2. 判据：保留 val macro-F1 做分数，但加两条硬约束 —— 空画面 YAW 率 ≤ 空画面标签 YAW 率 + 0.05、|recall_L − recall_R| ≤ 0.15 —— 并对 ±1 轮做平均、并列取靠后轮（`tools/select_ckpt.py::select_epoch`）。它在 0417 单帧 / 双帧上与训练器 best 同轮（14 / 20），在 trunc 三 seed 上改选 20 / 11 / 29（训练器 28 / 16 / 18）。两条硬约束对应闭环的两种坏法（空画面乱转、单边转），平滑只是对 20% 翻转率的弱防护；谁更好只能闭环裁决，建议至少排 seed 0 的一对（ep28 vs ep20）。
3. 更根本的：相邻轮 20% 预测翻转说明单轮 argmax 怎么选都脆；训练侧做末 k 轮权重平均（SWA 类）才是对症的，这是训练改动，本轮禁止、留给后续批次。

**闭环待跑**（主会话排在 eval4 之后，每个 20 seed）：

| 臂 | ckpt | sha256 前 16 位 |
| --- | --- | --- |
| seed0 训练器 best（ep28） | `$E/e1-noise/seed0/run/ckpt_best.bin` | 2b98084e57665e9b |
| seed1 训练器 best（ep16） | `$E/e1-noise/seed1/run/ckpt_best.bin` | a98b2607ff60a040 |
| seed2 训练器 best（ep18） | `$E/e1-noise/seed2/run/ckpt_best.bin` | 73181f379edd9d4f |
| seed0 `select_epoch`（ep20，可选） | `$E/e1-noise/seed0/run/ckpt_ep20.bin` | 6c6025f2d3f1c513 |
| seed1 `select_epoch`（ep11，可选） | `$E/e1-noise/seed1/run/ckpt_ep11.bin` | ce21b59469cd0be8 |
| seed2 `select_epoch`（ep29，可选） | `$E/e1-noise/seed2/run/ckpt_ep29.bin` | 2293d3470cf7d3a9 |
| 0417 单帧 best（ep14，E3 对照） | `$E/e3-two-frame/single/run/ckpt_best.bin` | 11fcac61c423ace2 |
| 0417 双帧 best（ep20） | `$E/e3-two-frame/two/run/ckpt_best.bin` | be2fe4fb716f46d7 |

闭环预测（预注册）：三个训练器 best 的 SR 极差 ≥ 3/20；n=20 的 Wilson 半宽 ±0.2 很可能盖住，按区间写。
