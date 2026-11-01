# L2 FiLM 微调（run-film-dec-ft）与语言门槛复测

- 时间：2026-11-01 00:49–02:27（训练）+ 02:30 探针
- 数据：`/root/rivermind-data/uav_rgb/l1-stop-500-173307`
- 结论：**语言门槛未通过，且微调把上一版的 flip 压低了 3.9 pp。**
  flip 45.5%（门槛 ≥50%）、acc_correct 62.8%（门槛 ≥90%）。
  同时导航能力也没有找回来：test acc 0.7482 / macro_f1 0.6935
  （对照 `run-l2-text16` 的 0.9106 / 0.8595）。这条路线两头都没赢。

## 命令

```bash
source env.sh
OUT=/root/rivermind-data/uav_rgb/l1-stop-500-173307
# 1) 从决定性集模型（20 轮）续训 30 轮，数据集换成 ds-cf
src/build/uavrgb_train --data "$OUT/ds-cf/dataset.uavr" --text-dim 16 \
  --epochs 30 --batch 64 --lr 5e-4 --seed 1234 --drop-collision \
  --resume "$OUT/run-film-dec/ckpt.bin" --out "$OUT/run-film-dec-ft"
# 2) 决定性帧探针（test split，255 帧）
"$UAV_RGB_PYTHON" sim/tools/probe_text_effect.py --src "$OUT" \
  --ckpt "$OUT/run-film-dec-ft/ckpt_best.bin" --split test \
  --out "$OUT/probe-ft-test.json"
```

环境：RTX 2080 Ti / sm_75，GPU 模式。训练 398,700 样本 / 5812.7 s / 68.6 样本/秒。

## 结果

### 语言门槛（决定性帧，test 255 帧）

| 权重 | acc_correct | acc_alt | follow | flip |
| --- | --- | --- | --- | --- |
| `run-film-dec/ckpt.bin`（决定性集 20 轮） | 0.5961 | 0.4392 | 0.2549 | **0.4941** |
| `run-film-dec-ft/ckpt_best.bin`（epoch 48） | 0.6275 | 0.5020 | 0.2824 | **0.4549** |
| 门槛 | ≥0.90 | — | — | ≥0.50 |

微调后 acc_correct +3.2 pp，但 flip −3.9 pp。两者都远未达标。

### 离线（`ds` 的 test split，973 样本，未参与训练与选择）

| 指标 | `run-l2-text16`（ds，10 轮） | `run-film-dec-ft`（ckpt.bin，末轮） |
| --- | --- | --- |
| acc | 0.9106 | **0.7482** |
| macro_f1 | 0.8595 | **0.6935** |
| balanced_accuracy | 0.8972 | **0.7072** |
| 每类 recall | 1.000 / 0.952 / 0.700 / 0.747 | 0.616 / 0.899 / 0.606 / 0.708 |

（`run-film-dec-ft` 的每类 recall 由混淆矩阵 `[250,144,7,5, 2,392,26,16, 3,18,40,5, 0,17,2,46]` 推得；
`eval-only` 事件里 `accuracy` 字段写成 null，只有 macro_f1/balanced/confusion，见"风险"。）

## 为什么两头都没赢

1. **`ds-cf` 里 78% 的样本不需要文本。** 决定性帧只占 astar 步的 35.3%，再加上安全的
   纯色集，微调集里真正需要颜色的比例很低。文本梯度和图像梯度一起更新，图像那条路的
   loss 更低，FiLM 的 γ/β 被拉回初始化附近——和 `run-l2-text16-cf` 的失败模式相同。
2. **`ckpt_best` 是按一个"没有文本信号"的 val 选的。** `ds-cf` 的 val split 只有原始集
   （`make_counterfactual` 只处理 `--splits train`），也就是目标可见、图像已经能猜出答案的
   那些帧。模型选择因此等价于"挑最会用图像捷径的 epoch"。epoch 48 的 dev macro_f1
   0.7894 是这条曲线上的最高点，而它的 flip 恰好偏低。
3. **更根本的：决定性帧上的图像本身泄漏答案。** 两根彩色柱都在视野内，最优轨迹朝目标飞，
   单帧图像已经足够判断该往哪边走。要让文本有用，得让图像单独不足——目标出视野、
   或两根同色柱靠关系词消歧。这是场景规格问题，不是训练问题。

## 更新 2026-11-01 03:20：49 个 test 任务闭环

```bash
source env.sh
OUT=/root/rivermind-data/uav_rgb/l1-stop-500-173307
"$UAV_RGB_PYTHON" sim/tools/eval_closed_loop.py --policy student \
  --ckpt "$OUT/run-film-dec-ft/ckpt_best.bin" --task l1 --split test \
  --episodes 49 --max-seeds 520 --steps 40 --hold 1.0 \
  --window-from observation --base-seed 700000 \
  --out "$OUT/eval-film-ft-keep-49"
```

`--window-from observation` 必须显式给：数据 meta 是 `observation`，而脚本默认是
`dispatch`（R01 之后的口径），不给就会训练/评测语义不一致。

### 结果（49/49 集，wall 2287.8 s）

| 指标 | 值 |
| --- | --- |
| SR | **0.3469**（17/49，95% CI 约 ±13 pp） |
| collision rate | **0.3061**（15/49） |
| stuck rate | 0.0 |
| false_forward rate | 0.2245（11/49） |
| 推理 p50 / p95 | 10.83 ms / 13.34 ms |
| 动作分布 | SAFE 125 / FWD 500 / YAW_L 102 / YAW_R 95 |

### 失败模式拆解

| 结局 | 集数 | 步数中位 | `final_waypoint_dist` 中位 |
| --- | --- | --- | --- |
| 成功 | 17 | 17 | 0.22 m |
| 碰撞 | 15 | 14 | 1.67 m |
| **提前停车**（无碰撞但自己 SAFE 停住） | **17** | 17 | **1.46 m** |

- **没有任何一集跑满 40 步**（最长 29）。所有失败都在目标附近之前就终止了。
- 17 个"提前停车"集的最后一步**全部是 SAFE**，且没有一集落在 `stop_radius+0.2 = 0.7 m` 内。
- 17/49 = 34.7% 的提前停车比例，与训练集里 **34.5% 的纯色 SAFE 帧**（90/498 集贴着柱子拍）
  在数值上重合。最可能的机制：那批帧教的是"画面被一种颜色填满 → SAFE"，
  于是学生一贴近任何柱子就停。**这是假设，本轮未验证。**

### 没做什么（本轮）

- 没跑盲前进基线（`--policy forward`，同 49 seed，约 38 分钟）。`EVAL_STANDARD.md`
  要求闭环同时报"我们的数字"和"盲前进基线"，现在只有前者，**不能对外报"false_forward
  低于盲前进"**。
- 没验证「纯色帧 → 提前停车」这个因果假设。最小的验证是重建去掉 90 个纯色集的
  `ds-cf2` 重训再评（约 30 分钟预处理 + 1.5 小时训练 + 38 分钟评测）。
- 没有和第二好的 FiLM ckpt（`run-l2-film16-cf`）做闭环对照。

## 没做什么

- 没跑闭环（SR/碰撞率）：语言门槛没过，闭环差异无法归因，跑了也是噪声。
- 没重建 `ds-cf2`（去掉 90 个纯色集）：本轮结论不依赖它。
- 没测"纯图像模型在同样 255 帧上的 acc"：`run-l2-text16` 的 ckpt 是改 FiLM 之前的
  mean-pool 结构，现在 `uavrgb_infer` 加载不了（报 `参数个数不符`）。要测必须用同一套
  结构重训一个 image-only 模型，本轮未做。
- 没改 `probe_text_effect.py` 支持 blank 文本模式。
- 没提交任何代码或数据。

## 风险与遗留

- `run-film-dec-ft` 的 `config.json` 写 `"epochs":30`，实际跑了 50 轮（20 续训 + 30）。
  `--epochs` 是"本次追加轮数"，配置里没有记录总轮数。
- `eval-only` 事件里 `accuracy`/`ce` 是 null，只有 macro_f1/balanced/confusion 有值；
  取 test acc 只能从 stdout 的 `test_accuracy` 行或混淆矩阵推。
- 探针的 255 帧 95% 置信区间约 ±6 pp，flip 45.5% 与门槛 50% 的差距（4.5 pp）在
  置信区间内，不能据此说"显著低于门槛"；但 acc_correct 62.8% vs 90% 是远超噪声的失败。
