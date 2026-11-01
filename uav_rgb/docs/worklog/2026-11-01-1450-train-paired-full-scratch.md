# 从零训配对全集（L1 v3 文本实验 D 组）

- 时间：2026-11-01 13:33–14:50
- 范围：只跑训练与离线评测，未改任何代码。产物在 `$DATA_ROOT/uav_rgb/l1-v3-cf/`
- 结论：完成。D 组与 C 组打平（flip 0.828 / macro_f1 0.668），未兑现「从零训更好」
  的预期；YAW_RIGHT 召回从 0.500 提到 0.667，bal_acc 涨 2.3pp。

## 做了什么

- 从零训配对全集（无 resume），20 轮、batch 64、lr 5e-4、`--class-weights auto`、
  `--drop-collision`，输出 `l1-v3-cf/run-film-cf2`。228200 样本 / 3990.8 s（66.5 min）。
- `auto` 权重在配对集上算出 `[1.772, 0.437, 1.688, 1.807]`（配对集里 SAFE 和
  两个 YAW 的占比比基础集高，因为决定性状态各多一条反向标签）。
- 训练前先重建 `src/build`：旧二进制是 10-31 23:41 编的，早于 `kTextLen` 8→9，
  直接拿来训会产出 8-token 的 film 矩阵。
- 评测 `ckpt_best.bin`（第 18 轮，dev macro_f1 0.7153）：
  - 离线 test（697 样本）：acc 0.8393 / macro_f1 0.6683 / bal_acc 0.7639 /
    recall `[0.782, 0.873, 0.733, 0.667]`
  - 决定性帧探针（436 帧）：flip 0.8280 / acc_correct 0.7890 / acc_alt 0.7294 /
    follow 0.6055
  - 探针结果存 `l1-v3-cf/run-film-cf2/text_probe_test.json`
- 对照评了末轮 `ckpt.bin`：test macro_f1 0.6381，比 best 低 3.0pp。
- 更新 `docs/experiments/2026-11-01-l1-v3-text-training.md`：加 D 组行与
  「更新 2026-11-01 14:50」小节，改了开头结论行。

## 没做什么

- **没跑闭环评测。** 49 个 test 任务的闭环（学生 policy + 盲前进基线）还没跑。
  - 原因：这是独立的一步，要起 Gazebo、定 policy 参数；先出离线数字再决定派
    哪份权重进闭环，避免白跑两轮仿真。
- **没修 `carry_resume_best`。** 跨数据集续训会让 `ckpt_best.bin` 冻结的问题
  仍在 `src/trainer.cpp:40-70`。
  - 原因：本轮的训练不 resume，碰不到这个坑；修它要动 trainer 的
    checkpoint 语义（是否该在换数据集时重置 best），属于设计决策，不是我该
    单方面拍的。
- **没再调超参。** 没有扫学习率 / epoch 数 / 配对比例。C 组和 D 组用的是同一套
  超参，差别只在是否续训。
- **没动数据。** 三条 audit FAIL（`class_share` YAW 3.7%、`time_pairing` 1 步、
  `safe_collision` 1 步）按用户 2026-11-01 的决定（选 a）保留，训练用
  `--class-weights auto` 兜。
- 没提交、没推送：用户未授权。

## 待确认

- **下一份闭环权重用哪个**：D（flip 0.83 / bal_acc 0.76）还是 B（flip 0.50 /
  macro_f1 0.76）。离线表已经把两条路的取舍画清楚了，再在离线指标上选也选不出
  新信息。我倾向两份都在同一批 49 个 test 任务上跑一次闭环，用 SR / 碰撞率 /
  false_forward 决定，而不是离线 macro_f1。
- `carry_resume_best` 要不要加 `--reset-best`。不加的话，以后任何换数据集的续训
  都会踩同一个坑。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 训练 | 完成，20/20 轮，`EXIT=0`，228200 样本无报错 |
| `ckpt_best.bin` 是否真被选中 | 是。md5 `ef84aef4…`，与 A 组 `da793c99…`、末轮 `21ddd60d…` 均不同 |
| best 落在第几轮 | 第 18 轮（dev macro_f1 0.7153）；末轮 0.6897 |
| 离线 test（`ds`，697 样本） | acc 0.8393 / macro_f1 0.6683 / bal_acc 0.7639 |
| 决定性帧探针（436 帧） | flip 0.8280 / acc_correct 0.7890 / follow 0.6055 |
| 末轮对照 | test macro_f1 0.6381（比 best 低 3.0pp） |
| 闭环 / 盲前进基线 | 未跑 |

## 风险与遗留

- **dev 曲线在 0.695–0.715 之间无趋势地晃了 15 轮**，`ckpt_best` 挑的是这条
  曲线上的噪声极值。E18 比 E20 好 3.0pp 是真事，但换成另一个 seed 未必能复现
  这个 3pp。别把 best 选择当成可靠的增益来源。
- **YAW_RIGHT 的 0.667 vs 0.500 只有 6 个样本的差别**（test 里该类 30 个）。
  这个「改善」的置信度很低，闭环里才看得出来是真是假。
- 配对集让 SAFE 和 YAW 的占比上升，`auto` 权重把 FORWARD 压到 0.437；如果闭环
  里出现「不敢往前走」的行为，先怀疑这个权重而不是模型本身。
- 三组（B/C/D）各有短板，**目前没有任何一份权重同时过 flip ≥0.50 且
  macro_f1 ≥0.86**。第一阶段的验收线（闭环 SR ≥50%、false_forward 低于盲前进）
  还没有任何一份权重去碰过。
