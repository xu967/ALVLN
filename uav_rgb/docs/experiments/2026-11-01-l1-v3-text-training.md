# L1 v3 同色柱对的文本训练（三组对照）

- 时间：2026-11-01 07:20–08:50
- 数据：`$DATA_ROOT/uav_rgb/l1-v3-relation-500`（494 集 / 7104 步）、
  `l1-v3-dec`（决定性配对集 892 集 / 7702 样本）、
  `l1-v3-cf`（配对全集 890 集 / 12814 样本）
- 结论：**同色柱对确实逼出了文本**（flip 45.5% → 82–97%），但导航指标按
  配对比例单调下降；目前没有一组同时过「flip ≥0.50 且离线 macro_f1 ≥0.86」。
  D 组（从零训配对全集，2026-11-01 14:40 追加）与 C 组打平，未超过续训。

## 三组训练

```bash
# A：只训配对决定性集（每个决定性状态两条标签）
src/build/uavrgb_train --data l1-v3-dec/ds/dataset.uavr --text-dim 16 \
  --epochs 20 --batch 64 --lr 5e-4 --seed 1234 --class-weights auto \
  --drop-collision --out l1-v3-dec/run-film-v3

# B：A 的权重 → 在**非配对**基础集上微调
src/build/uavrgb_train --data l1-v3-relation-500/ds/dataset.uavr --text-dim 16 \
  --epochs 15 --batch 64 --lr 2e-4 --seed 1234 --drop-collision \
  --resume l1-v3-dec/run-film-v3/ckpt_best.bin --out .../run-film-v3-ft

# C：A 的权重 → 在**配对全集**上续训（原集 + make_counterfactual 的反事实集）
src/build/uavrgb_train --data l1-v3-cf/ds/dataset.uavr --text-dim 16 \
  --epochs 12 --batch 64 --lr 2e-4 --seed 1234 --drop-collision \
  --resume l1-v3-dec/run-film-v3/ckpt_best.bin --out l1-v3-cf/run-film-cf
```

## 结果

语言门槛探针（collection test split，436 个决定性帧）：

| 组 | 权重 | flip (≥0.50) | acc_correct (≥0.90) | acc_alt | follow |
| --- | --- | --- | --- | --- | --- |
| — | `run-film-dec-ft`（v2，上一版） | 0.4549 | 0.6275 | 0.5020 | 0.2824 |
| A | `run-film-v3/ckpt_best.bin` | **0.9702** | 0.7959 | 0.7890 | 0.6606 |
| B | `run-film-v3-ft/ckpt_best.bin` | 0.5023 | 0.8761 | 0.4037 | 0.3555 |
| C | `run-film-cf/ckpt.bin` | 0.8234 | 0.8303 | 0.6950 | 0.6078 |

离线（`ds` 的 test split，697 样本，未参与训练与模型选择）：

| 组 | acc | macro_f1 | bal_acc | 每类 recall |
| --- | --- | --- | --- | --- |
| v2 参考（`run-l2-text16`） | 0.9106 | 0.8595 | 0.8972 | 1.000 / 0.952 / 0.700 / 0.747 |
| A | 0.6801 | 0.4794 | 0.5641 | 0.769 / 0.688 / 0.533 / 0.267 |
| B | **0.8967** | **0.7610** | 0.7607 | 0.871 / 0.939 / 0.567 / 0.667 |
| C | 0.8465 | 0.6737 | **0.7412** | 0.816 / 0.882 / 0.767 / 0.500 |

## 结论

1. **场景升级有效。** flip 从 v2 的 0.4549 提到 0.9702（A 组）。同色柱对把
   「图像单独就能决定动作」这条路堵死了，这是之前三版融合设计都没做到的。
2. **只有配对监督能教文本。** B 组在非配对基础集上只微调了 15 轮，flip 就从
   0.97 掉到 0.50——尽管基础集有 60.2% 的决定性帧。决定性说的是「换指令动作
   会变」，但基础集里同一张图永远只配一条指令，模型靠图像记住映射就够了。
   **配对（同一帧两条指令）才是逼它读文本的唯一手段。**
3. **配对比例 = 文本/导航的旋钮。** A 组 100% 配对 → flip 0.97 / macro_f1 0.48；
   B 组 0% → flip 0.50 / macro_f1 0.76；C 组 50%（每帧都出现两次）→
   flip 0.82 / macro_f1 0.67。三组严格单调，没有免费的午餐。
4. 没有一组过 `acc_correct ≥ 0.90`。C 组离得最近（0.8303）。

## 已知问题（影响本轮的读数）

- **跨数据集 resume 会让 `ckpt_best.bin` 冻结。** `carry_resume_best` 把续训
  checkpoint 的 `best_macro_f1` 和 best 权重一起带进新 run，只有新 dev 分数
  **超过旧分数**才会覆盖。C 组续训自 A（旧 dev 0.8036），而 C 的配对 val
  最高只有 0.7302，于是 12 个 epoch 一次都没保存 best，
  `run-film-cf/ckpt_best.bin` 与 A 的 md5 完全相同（`da793c99…`）。
  上面的 C 组数字评的是 `ckpt.bin`（末轮，dev macro_f1 0.7197，
  不是峰值 0.7302）。
  - 这是代码的刻意设计（新 run 必须带着对应的 best 权重），但对
    「换数据集续训」是个坑：旧分数与新的 val split 不可比。
  - 尚未修，也没加 `--reset-best` 之类的开关。
- A、B 两组的 `ckpt_best` 选择正常（同数据集内训练）。
- 判据阈值 `acc_correct ≥0.90` 出自 `experiments/2026-10-31-basic-vln-plan.md`，
  不是公开基准。

## 更新 2026-11-01 14:50：D 组（从零训配对全集）

C 组评的是 `ckpt.bin`（末轮）而非 best，是因为 `ckpt_best.bin` 被冻结了。为了拿到
一份「best 真正按配对 val 选出来」的模型，追加一次从零训练（无 resume，故无继承分数）：

```bash
src/build/uavrgb_train --data l1-v3-cf/ds/dataset.uavr --text-dim 16 \
  --epochs 20 --batch 64 --lr 5e-4 --seed 1234 --class-weights auto \
  --drop-collision --out l1-v3-cf/run-film-cf2
# 228200 样本 / 20 轮 / 3990.8 s（66.5 min），class_weights auto
# = [1.772, 0.437, 1.688, 1.807]
```

`ckpt_best.bin` 落在第 18 轮（dev macro_f1 0.7153，md5 `ef84aef4…`，与 A 组
`da793c99…`、末轮 `21ddd60d…` 均不同），确认选权重这次真的动了。

| 组 | 权重 | flip | acc_correct | acc_alt | follow |
| --- | --- | --- | --- | --- | --- |
| D | `run-film-cf2/ckpt_best.bin`（E18） | 0.8280 | 0.7890 | 0.7294 | 0.6055 |

离线（`ds` 的 test split，697 样本）：

| 组 | acc | macro_f1 | bal_acc | 每类 recall |
| --- | --- | --- | --- | --- |
| D | 0.8393 | 0.6683 | **0.7639** | 0.782 / 0.873 / 0.733 / **0.667** |
| （对照）D 末轮 `ckpt.bin`（E20） | 0.8149 | 0.6381 | 0.7517 | 0.762 / 0.845 / 0.733 / 0.667 |

### 结论

5. **从零训没有赢过续训。** D ≈ C：flip 0.828 vs 0.823、macro_f1 0.668 vs 0.674
   （-0.5pp）、acc 0.839 vs 0.847。差值落在 dev 波动带内（第 6 轮起 dev macro_f1
   在 0.695–0.715 之间无趋势地晃），**不能算 C 更好的证据，也不能算 D 更好**。
   之前「从头训能让 flip 和 macro_f1 都好一些」的预期没兑现。
6. **唯一的实质变化是 YAW_RIGHT**：召回 0.500（C）→ 0.667（D），bal_acc
   0.741 → 0.764。A/C 两组里 YAW_RIGHT 一直是塌得最狠的类，从零训把这个坑填了
   一半。但 YAW_RIGHT 在 test 只有 30 个样本，6 个的差别（0.667 vs 0.500）
   就是 ±6.7pp 的粒度。
7. **best 选择有用，但收益有限。** 同一次训练里 E18 比 E20 的 test macro_f1 高
   3.0pp（0.6683 vs 0.6381），说明后半程在过拟合。不过 best 是从 0.695–0.715
   这条几乎水平的 dev 曲线上挑最高点，挑出来的是噪声极值，不是真的拐点。
8. 三条路都试完了，**最优折中点仍是 B 组**（flip 0.502 / macro_f1 0.761）。
   如果下一阶段要出闭环数字，D 是文本最强的一组里导航最好的（flip 0.83 +
   bal_acc 0.764），B 是导航最强的一组里文本唯一过 0.5 的。两者该各跑一次闭环，
   而不是继续在离线表上做选择。
