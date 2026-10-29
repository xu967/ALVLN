# 公开评测口径 + 审查问题修复 + 200 条数据集

- 时间：2026-10-29 13:00
- 范围：`src/`（指标、训练器、类别权重）、`sim/tools/`、`tools/check.sh`、`docs/EVAL_STANDARD.md`
- 结论：完成。7 项审查问题全部落地；离线指标改为公开口径；首次在 200 条真实采集数据上训练

## 做了什么

### 1. 审查问题修复（7 项）

| # | 问题 | 改法 | 文件 |
| --- | --- | --- | --- |
| 1 | `train_ce` 是加权 CE、`dev_ce` 是无权 CE，画在同一张图上会读成假的过拟合 | 训练循环里对同一份 logits 另算无权 CE 计入 CSV，加权 loss 只留 metrics.jsonl | `src/trainer.cpp` |
| 2 | test split 解析了但从未评估 | 训练循环后对 `ds.test_idx` 评估一次，写 `Split::kTest` 事件；CLI 打印 `test_accuracy` | `src/trainer.cpp`、`src/train_main.cpp` |
| 3 | 重试时 `"wb"` 截断上一次的 sim 日志 | 改 `"ab"` | `sim/tools/collect_parallel.py` |
| 4 | 输入分辨率两个来源，已漂过一次（`collect.py` 还写着 128×128） | 修注释；`check.sh` 加一致性检查，不一致直接失败 | `sim/tools/collect.py`、`tools/check.sh` |
| 5 | `--max-grad-norm 0` 静默冻结模型 | 入口拒绝 `<= 0` | `src/train_main.cpp` |
| 6 | `trainer.hpp` 注释声称 `evaluate` 会计入 `samples_seen_`，实际没有 | 改正注释 | `src/include/uavrgb/trainer.hpp` |
| 7 | `auto` 权重要素为 0 的类静默无梯度 | `n_c=0` 打警告；**新增** `n_c=1` 退化警告（实测触发，见下） | `src/class_weights.cpp` |

### 2. 离线指标换成公开口径

新增 `src/include/uavrgb/eval_metrics.hpp` + `src/eval_metrics.cpp`（50 行）：

- 混淆矩阵（C×C，行=真值，列=预测）
- 宏平均 F1（与 `sklearn.metrics.f1_score(average="macro")` 同算法）
- 平衡准确率（与 `balanced_accuracy_score` 同算法）
- 每类召回

`validation` 事件追加 `macro_f1`、`balanced_accuracy`、`confusion` 三个字段，既有键名不变。
每 epoch 额外往 stdout 打一行宏指标 + 每类召回。

### 3. 评测标准文档

新增 `docs/EVAL_STANDARD.md`（约束文件）：

- 离线层：scikit-learn 定义的宏指标。
- 闭环层：**AvoidBench（TU Delft, ICRA 2023, arXiv:2301.07430）** 的七项指标
  —— SR / PO / EO / AGV / MP / 处理时间 / 对比因子，含全部公式。
- 明确写清：我们**不能跑** AvoidBench 套件（需要 ROS noetic + Unity3D + Flightmare），
  只实现同一套定义。任何暗示「跑了 AvoidBench」的表述都是错的。
- 数据切分标准、已知污染项、下一步映射表。

### 4. 200 条数据集 + 首次训练

- 采集：8 实例 × 25 集，200/200 成功，4000 步，端到端约 13 分钟。
- 预处理：`--drop-collision` 后 3707 样本，160×120，214 MB。
- 训练：30 epoch，batch 128，`--class-weights auto`。

## 关键实测结果

### 类别分布（丢碰撞步后）

| split | N | SAFE | FORWARD | LEFT | RIGHT |
| --- | --- | --- | --- | --- | --- |
| train | 2981 | **1** | 2196 | 408 | 376 |
| val | 370 | **0** | 271 | 42 | 57 |
| test | 375 | **0** | 282 | 65 | 28 |

### 训练曲线（dev，末 3 个 epoch）

```
epoch  acc       macro_f1  bal_acc  recall[SAFE,FWD,LEFT,RIGHT]
27     0.9784    0.725     0.735    0.000 0.973 1.000 0.965
28     0.9703    0.723     0.727    0.000 0.965 1.000 0.965
30     0.9757    0.723     0.731    0.000 0.978 1.000 0.947
```

### test split（只在训练结束后评一次，不参与任何选择）

```
N=375  acc=0.9733  macro_f1(4类)=0.7114  macro_f1(3类)=0.9486  bal_acc=0.7272

         预测: SAFE  FWD LEFT RIGHT
真值 SAFE       0    0    0    0   support 0
真值 FWD        0  275    2    5   support 282
真值 LEFT       0    2   63    0   support 65
真值 RIGHT      0    0    1   27   support 28
```

| 指标 | 数值 | 对照 |
| --- | --- | --- |
| test 准确率 | **0.9733** | 「全预测 FORWARD」= 0.752 |
| 三类宏平均 F1 | **0.9486** | FWD .984 / LEFT .962 / RIGHT .900 |
| 四类宏平均 F1 | 0.7114 | SAFE 在 test 里 support=0，按 sklearn 语义记 0 |
| 训练吞吐 | 62.2 样本/秒，24 分钟 30 epoch | CPU 打满，GPU 空闲 |

**结论**：模型确实学会了三分类转向，不是「只会前进」。但——
test 场景与 train 用的是同一套车道生成器，只换了 y 偏移。**这个 97.3% 证明的是
「记住了车道结构后能正确动作」，不证明「换个布局也能飞」。** 布局级泛化要靠闭环评测。

### 数据质量问题（本轮新发现）

1. **`stuck` 不是碰撞。** 274 个 collision 标记里 238 个是 `stuck`（位移 < 阈值），
   只有 30 个真 contact（另有 6 个两者都有）。
   - **94 次 stuck 集中在 step 3**：起飞后加速没完成，系统性假阳性。
   - 其余在 step 9–15：专家真的卡在柱子上，是**专家策略的缺陷**，不是碰撞。
   - 直接把 `collision` 当碰撞率会高估约 8 倍。
2. **SAFE_NEUTRAL 只有 1 个样本（train），val/test 各 0 个。**
   - 四动作实际退化成三动作，该类召回恒为 0。
   - `auto` 逆频率权重退化为 `[745.3, 0.339, 1.83, 1.98]`：那 1 个样本独占 1/4 总 loss。
   - 已加警告，但不是修复。

## 没做什么

- **没有跑 AvoidBench 套件**：不可能跑（ROS noetic + Unity3D + Flightmare，本仓库禁 ROS 2、无 ROS 1）。
  只沿用了它的指标定义。
- **没有实现 SR / PO / EO / AGV / MP 的计算**：`d_min` 需要 A\*、终点还没定义、`SimulatorEval` 整体未做。
  `EVAL_STANDARD.md` §3 已列出映射与缺口。
- **没有修 SAFE_NEUTRAL 缺失**：这是设计决策（合并进 FORWARD / 让专家更常输出 / 去掉这个类），
  需要用户拍板，我不单方面改动作空间。
- **没有修专家中段卡死**：本轮只统计了它发生的步号分布，没改 `teacher.py`。
- **没有做闭环评测**：学生模型还没在仿真里飞过。
- **没有优化训练吞吐**：实测 ~2 s/step，CPU 99.9%、GPU 基本闲着，瓶颈是 `dataset.cpp`
  逐样本 seek+read 和 host 侧 uint8→float 转换。属于下一轮工作。
- 没有做多 seed 重复实验，没有做超参搜索。

## 验证到哪一步

| 命令 | 结果 |
| --- | --- |
| `cmake --build src/build` | 通过 |
| `ctest --test-dir src/build` | **13/13 通过**（修好 `test_trainer` 的 schema 断言后复测） |
| `tools/preprocess.py --drop-collision` | 3707 样本，214 MB，7 s |
| `collect_parallel.py --instances 8 --episodes-per-instance 25` | 200/200，4000 步 |
| `uavrgb_train --epochs 30 --class-weights auto` | 进行中，dev acc 0.976 @ epoch 13 |
| `bash -n tools/check.sh` | 语法通过（未整跑，见下） |

## 风险与遗留

- **`test_trainer` 的 schema 检查曾因本次改动失败**，已同步修正参考事件。这说明该测试有效，
  以后改 event 字段必须同步它。
- **`check.sh` 没有整跑**：新加的分辨率一致性检查只做了语法检查。下次完整跑一遍再算验收。
- **训练吞吐 2 s/step** 让 30 epoch 要 29 分钟；数据量再涨一个量级就会卡住迭代。
- **`stuck` 阈值**（`min(STUCK_EPS, 0.5*SPEED_FWD*sim_dt)`）在低 RTF 下会缩小，
  可能造成漏报；本轮没改。
- 四类宏平均 F1 在当前数据上**不是学习信号**，报数字必须同时给 support。
