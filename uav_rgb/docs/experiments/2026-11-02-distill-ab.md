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

## 第二轮（2026-11-03，1000 集 + 独立 test split）

### 数据

`ds-teacher1000`：815 train / 92 val / 93 test 集，18056 样本（14733 / 1607 / 1715）。
软标签 18056/18056 全覆盖，T=2.0。**本轮有独立 test，离线不再用 val 报数。**

三组命令只差 `--soft-targets` / `--distill-alpha`，其余完全相同（seed=1234，
`--drop-collision`，20 轮，batch 64，lr 5e-4）。

### 离线（独立 test split，n=1715，未参与训练与 best 选择）

| 组 | acc | macro-F1 | bal_acc | 召回 [SAFE, FWD, LEFT, RIGHT] |
| --- | --- | --- | --- | --- |
| base | 0.8799 | 0.8142 | 0.7832 | [0.792, 0.948, 0.664, 0.728] |
| a50 | 0.8781 | 0.8155 | 0.8023 | [0.849, 0.927, 0.699, 0.735] |
| **a80** | **0.8910** | **0.8336** | **0.8112** | [0.792, 0.951, 0.747, 0.755] |

收益集中在两个转向类（LEFT 0.664→0.747，RIGHT 0.728→0.755），FORWARD 基本不动。
**但 dev 上三组几乎相同**（base 0.8139 / a50 0.8158 / a80 0.8133），差值 ≤0.3 pp；
所以「a80 更好」只有 test 看得出来，**不能拿 test 做选择再报该组的 test 数**，
三组因此全部送闭环。

### 闭环（同 40 个 test seed，实际有效 39 集；`--text-mode keep`）

| 组 | SR（ever，主指标） | SR（stop） | 碰撞率 | false_forward |
| --- | --- | --- | --- | --- |
| base | 0.487（19/39） | 0.256（10/39） | 0.359 | 0.333 |
| a50 | 0.513（20/39） | 0.333（13/39） | 0.410 | 0.385 |
| a80 | **0.590（23/39）** | **0.487（19/39）** | 0.359 | 0.359 |

配对 McNemar 精确检验（同一批 39 集）：

| 对比 | SR ever | SR stop |
| --- | --- | --- |
| base vs a50 | +6/−5，p=1.000 | +6/−3，p=0.508 |
| base vs a80 | +7/−3，p=0.344 | **+11/−2，p=0.0225** |
| a50 vs a80 | +6/−3，p=0.508 | +8/−2，p=0.109 |

### 结论（必须按这个口径引用）

1. **主指标不显著**。a80 比 base 高 10.3 pp，但 n=39 配对设计的 MDE 约 15 pp，
   这个差距在噪声范围内，只能标 `exploratory`，**不能说「蒸馏提升了成功率」**。
2. **次要指标名义显著**：base→a80 的「真的停在航点 1.5 m 内」从 0.256 升到 0.487
   （+23 pp，p=0.0225）。但这是 3 组 × 2 指标 = 6 次比较，Bonferroni 校正后阈值
   约 0.008，**它不经过校正**。同样只能当线索。
3. **剂量-反应方向一致**：SR ever 0.487→0.513→0.590、SR stop 0.256→0.333→0.487
   都随 α 单调上升，离线 macro-F1 也是这样。三条独立证据同向，比单个 p 值更值得信，
   但仍不足以定论。
4. 机制上说得通：软标签的收益落在转向类，而「停在航点附近」正是需要更晚、更准的转向。

### 已知的评测污染与已核实的结论

- 三组不是同一时刻跑的（base 10:13–10:45、a50 10:45–11:14、a80 11:14–11:57），
  期间**另一个会话**改了 `eval_closed_loop.py`（`split_for_scene`、`goal_radius`、推理路径）
  和 `scene_gen.py`（新增 campus 任务、meta 加 `camera_profile`/`dynamics_profile`）。
- 已核实对本次三种运行方式**不改变语义**：l1 走 `split_for_scene` 时回退到 `split_for_gid`；
  `goal_radius` 缺省仍是 `REACH_RADIUS=1.5`；`task=l1` 的 world 字节生成路径未变。
- 推理二进制路径从 `src/build/uavrgb_infer` 换成了 `build/check/uavrgb_infer`（两个不同构建）。
  用同一 ckpt + 40 帧确定性随机输入对比：**动作与概率完全一致（40/40）**，结论不受影响。
- 剩余风险：这种跨会话改动会持续发生。以后跑 A/B 应把评测脚本与推理二进制路径**固定**下来
  （显式 `--infer-bin`，或先 commit 再跑），否则下次不一定这么走运。

### 复现

产物在 `$DATA_ROOT/uav_rgb/distill-1000-20261103/`：
`student-{base,a50,a80}/`、`eval-student-{base,a50,a80}/eval.json`。
命令：`/tmp/eval_1000.sh student-base student-a50 student-a80`（临时脚本，尚未入库）。
