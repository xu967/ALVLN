# 特权蒸馏 A/B（2026-11-02 首轮）

- 目的：教师→学生的软标签蒸馏，在**除损失函数外完全相同的条件下**是否有收益。
- 结论（首轮，20 集）：**离线 macro-F1 +2.1 pp，闭环未见显著差异**（McNemar p=1.0）。
  两组的失败模式一致，瓶颈是终点前撞柱，不是软标签能直接补的。

## 数据

`$DATA_ROOT/uav_rgb/teacher-400-20261102`（带深度，330+ 集）打包成
`ds-teacher400`：5906 train / 582 val / **0 test**（本轮只传了 train+val，
test 那 39 集当时没纳入）。

> 缺陷：没有 test split，下面的离线指标全部是 val（参与选权重），偏乐观。
> 后续版本已改用 train+val+test 全量打包。

软标签：教师 `teacher-run-400-goal/ckpt_best.pth`（val macro-F1 0.8991，
闭环 SR 0.95）在有深度的帧上导出 logits，按 T=2 做 softmax，再按 `.uavr`
全局样本序拼平（`tools/join_soft_targets.py`，6488/6488 全覆盖）。

## 训练

两组命令只差 `--soft-targets`：

```bash
BIN=<build>/uavrgb_train
$BIN --data ds-teacher400/dataset.uavr --text-dim 16 \
     --epochs 20 --batch 64 --lr 5e-4 --seed 1234 --drop-collision \
     --out distill-20261102/student-base
$BIN ...同上一模一样... \
     --soft-targets ds-teacher400/soft_targets.bin --distill-alpha 0.5 \
     --out distill-20261102/student-distill
```

两组都是 118120 样本 / 20 轮，约 2080 s。

## 离线（dev split，**参与过选权重，偏乐观**）

| | acc | macro_f1 | 最优轮 |
| --- | --- | --- | --- |
| base | 0.8351 | 0.7648 | 17 |
| distill | 0.8436 | **0.7860** | 16 |

蒸馏在 minority 类上收益更明显（YAW_LEFT 召回 0.76 → 0.80）。

## 闭环（同一 20 个场景，`--text-mode keep`）

两种口径都列（定义见 `EVAL_STANDARD.md`，2026-11-02 起主指标是 `goal_reach_ever`）：

| 组 | SR（ever） | SR（stop，旧口径） | 碰撞率 |
| --- | --- | --- | --- |
| student-base | 0.600（12/20） | 0.450（9/20） | 0.25 |
| student-distill | 0.650（13/20） | 0.450（9/20） | 0.30 |

- 新口径下 base 独有成功 2 集、distill 独有 3 集 → **McNemar 精确 p = 1.0000**，
  **统计上不可区分**。20 集的 Wilson CI 宽约 ±20 pp，任何 <10 pp 的差异都测不出来。
- 旧口径下两组各 9/20，也是 p=1.0。

## 失败分解（新口径）

| | base | distill |
| --- | --- | --- |
| 成功 | 12 | 13 |
| 撞柱（多在距航点 0.7–0.9 m） | 5 | 6 |
| 始终没到（≥1.5 m） | 3 | 1 |

**瓶颈已经不是"停不下来"**（新口径把"到过又飞走"计入成功），而是**终点前的避障**。
教师在同一批场景能停到 0.12–0.58 m 且几乎不撞，差别在于教师有深度 + 目标向量。

## 复现

产物在 `$DATA_ROOT/uav_rgb/distill-20261102/`：
`soft_train.npz` / `soft_val.npz` / `student-base/` / `student-distill/` /
`eval-student-base/eval.json` / `eval-student-distill/eval.json`。

## 下一步

1. 数据量只有 328 训练集（v4 学生是 902 集），扩到 ~1000 集后重跑本 A/B。
2. 离线要有独立 test split（打包时带上 test 目录）。
3. 终点前撞柱是主要失分点，优先看教师软标签里的 clearance 信息是否被学生学走。
