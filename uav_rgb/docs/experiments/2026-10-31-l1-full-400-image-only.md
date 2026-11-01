# L1 全量 400 集 image-only 基线

- 时间：2026-10-31 10:25–11:05（离线部分）
- 数据：`$DATA_ROOT/uav_rgb/l1-full-400-0803/ds/dataset.uavr`（16000 样本，160×120）
- 目的：给 L2 提供**同数据、同口径**的 image-only 对照（文本完全不用）
- 结论：离线 macro_f1 0.7686（先验 0.1283）；闭环 **SR 0/34**——但这是 **v1
  route-choice 废数据**：teacher 自己 0/400 到 goal（max_x 中位 2.58 m vs goal 6.0），
  0/34 不能归因于语言或「学生不知道目标色」，也**不能当 L2 的基线**。
  数据量不是主因，加数据不解决；v2 goal-stop 已重采。

## 1. 数据与训练

| 项 | 值 |
| --- | --- |
| split | train 12960 / val 1600 / test 1440 |
| audit | 15/15 PASS（`audit.json`） |
| 训练 | 10 轮，batch 64，lr 1e-3，`--class-weights auto --drop-collision` |
| 耗时 | 2300.2 s（129600 样本，56.3 样本/秒） |
| best 选择 | dev macro_f1（epoch 9） |

逐轮 dev：

| epoch | dev_ce | dev_acc | dev_macro_f1 |
| --- | --- | --- | --- |
| 1 | 0.6524 | 0.6663 | 0.6717 |
| 4 | 0.5246 | 0.7594 | 0.7511 |
| 6 | 0.5091 | 0.7738 | 0.7747 |
| 7 | 0.5143 | 0.7756 | 0.7770 |
| **9** | **0.5310** | **0.7756** | **0.7800** ← best |
| 10 | 0.5239 | 0.7612 | 0.7674 |

## 2. 离线指标（`ckpt_best.bin`，`--eval-only`）

| 指标 | val | test |
| --- | --- | --- |
| action_accuracy | 0.7756 | 0.7799 |
| **macro_f1** | **0.7800** | **0.7686** |
| balanced_accuracy | 0.7876 | 0.7747 |
| recall (SAFE/FWD/LEFT/RIGHT) | 1.000 / 0.731 / 0.715 / 0.704 | 1.000 / 0.736 / 0.698 / 0.664 |

多数类先验（`tools/baseline_priors.py`，train 里最多的 FORWARD_SLOW）：

| 指标 | val | test |
| --- | --- | --- |
| accuracy | 0.3719 | 0.3451 |
| macro_f1 | 0.1355 | **0.1283** |
| balanced_accuracy | 0.2500 | 0.2500 |

**我们的模型 test macro_f1 0.7686 vs 先验 0.1283**（+0.64），第一阶段目标里的
「明显高于多数类先验」这一条成立。

对比 120 集 pilot（同模型同超参）：test macro_f1 0.4954 → **0.7686**。数据量是主因。

## 3. 闭环（`--task l1 --window-from observation`，test 36 seed）

命令：`eval_closed_loop.py --task l1 --split test --seeds 310003,... --steps 80
--window-from observation --policy student --ckpt run/ckpt_best.bin`
（2026-10-31 11:05–11:58，wall 3160.9 s）

| 指标 | 值 |
| --- | --- |
| 评测集 | 34（36 seed，去重掉 2 集同 gid 同目标色） |
| **SR** | **0.0**（0/34） |
| collision_rate | **0.50**（17/34） |
| stuck_rate | 0.0 |
| false_forward_rate | **0.441**（15/34） |
| 推理延时 | p50 **9.65 ms** / p95 **21.38 ms**（warm n=1863） |
| 动作分布 | SAFE 83 / FORWARD 1260 / LEFT 257 / RIGHT 297 |

口径：`success = 无 collision 且 max_x ≥ goal_x(6.0) 且 waypoint_min_dist ≤ 1.0 m`。

**失败在哪一条**（三条同时满足才算成功）：

| 条件 | 通过 |
| --- | --- |
| waypoint_min_dist ≤ 1.0 m | 19/34 |
| max_x ≥ goal_x = 6.0 | **1/34**（310251，9.30，但它的 waypoint 是 1.49 m） |
| 无 collision | 17/34 |
| 三条同时 | 0/34 |

所以卡点是**没有一趟既靠近 waypoint 又越过 goal_x**：一半的集在撞柱时提前结束（平均
21–43 步），跑满 80 步的那 17 集里 max_x 中位数只有 2.5 m 左右——image-only 学生
不知道目标色，看到柱子就减速/转向，走不出 6 m。这正是 L2 要打的那个 0。

逐集（`eval.json` 的 `per_episode`）：

| seed | 目标色 | 步数 | 碰撞 | max_x | waypoint_min_dist |
| --- | --- | --- | --- | --- | --- |
| 310003 | blue | 21 | ✓ | 2.38 | 0.10 |
| 310016 | red | 21 | ✓ | 2.18 | 0.12 |
| 310018 | green | 80 |  | 3.15 | 0.03 |
| 310020 | yellow | 72 | ✓ | 3.41 | 0.04 |
| 310027 | yellow | 80 |  | 2.24 | 1.53 |
| 310038 | green | 80 |  | 3.03 | 0.06 |
| 310063 | blue | 36 | ✓ | 3.47 | 0.08 |
| 310071 | yellow | 40 | ✓ | 4.36 | 1.01 |
| 310075 | blue | 80 |  | 2.52 | 0.57 |
| 310081 | yellow | 80 |  | 2.99 | 1.57 |
| 310085 | green | 21 | ✓ | 2.42 | 1.51 |
| 310096 | blue | 30 | ✓ | 3.09 | 0.04 |
| 310098 | yellow | 80 |  | 2.88 | 1.39 |
| 310108 | green | 80 |  | 1.91 | 0.14 |
| 310111 | green | 26 | ✓ | 2.90 | 1.26 |
| 310126 | blue | 80 |  | 3.15 | 0.03 |
| 310135 | blue | 37 | ✓ | 3.13 | 0.22 |
| 310136 | yellow | 22 | ✓ | 3.07 | 0.09 |
| 310138 | blue | 80 |  | 2.85 | 0.07 |
| 310175 | green | 80 |  | 1.92 | 0.92 |
| 310177 | blue | 33 | ✓ | 3.08 | 1.26 |
| 310189 | red | 80 |  | 2.35 | 1.62 |
| 310204 | yellow | 43 | ✓ | 3.54 | 0.09 |
| 310230 | blue | 28 | ✓ | 2.44 | 1.19 |
| 310246 | yellow | 80 |  | 2.58 | 0.07 |
| 310250 | blue | 80 |  | 2.21 | 0.03 |
| 310251 | yellow | 80 |  | 9.30 | 1.49 |
| 310261 | green | 24 | ✓ | 2.89 | 1.25 |
| 310272 | yellow | 38 | ✓ | 2.91 | 1.67 |
| 310284 | red | 80 |  | 1.59 | 0.36 |
| 310311 | red | 80 |  | 5.98 | 1.63 |
| 310315 | green | 80 |  | 0.98 | 1.89 |
| 310329 | blue | 24 | ✓ | 2.34 | 0.47 |
| 310375 | blue | 21 | ✓ | 2.23 | 1.68 |

原始产物：`$DATA_ROOT/uav_rgb/l1-full-400-0803/eval-student-image-only/eval.json`。

## 4. 必须写进结论的限定

- **SAFE recall = 1.000 不是策略能力**：SAFE 样本 100% 来自 safe 出生集
  （`docs/experiments/2026-10-31-l1-data-health.md §2`），模型学的是
  「出生在柱阵里 → 悬停」这个场景级判据。评测全是默认出生，SAFE 不会触发。
- 闭环评测集只有 34 集，SR 的置信区间很宽。
