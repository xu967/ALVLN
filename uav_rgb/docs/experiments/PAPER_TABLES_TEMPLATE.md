# 论文表格模板（空表，2026-11-07）

- 口径：`docs/EVAL_STANDARD.md` §5.1/§5.2/§5.3。每格 `点估计 [95% 区间] (n)`，n = 独立几何 / 路线数，
  不是 seed 数（§5.3「seed ≠ 几何」）。n < 20 的格标 `pilot`，不进正文结论。
- 生成：所有表由 `tools/stats_report.py` 从各组 `eval.json` 生成（`--out-dir` 下 `<stem>.md/.json`），
  手改数字一律不许；表里的「生成命令」直接可复制，`$O` 是该实验的输出根。
- 统计（§5.3）：二值指标 Wilson 95%；SPL 百分位 bootstrap；多训练种子 IQM + 分层 bootstrap；
  两方法 McNemar（共同 seed）；≥3 方法另给 Cochran's Q 作总体检验，两两 McNemar 照做（不以 Q 显著为前提）+ Holm（族 = 同场景同指标的全部两两行）；功效按 `--power p1 p2`。
- 标注必写：hold（Gazebo 1.0 s / 3DGS 0.5 s）、window_from（observation / render_observation）、
  载具（E1 外径 0.20 m vs X3 0.71 m）、`metric_version`。两域 hold 不同，跨域列不可直接比。
- E1 的 SAFE 是滑行不是停（§5.3）：task_success 列对 E1 必须加脚注，或等执行器位置保持后复跑。

---

## 表 1 主表：方法 × 场景 × 指标

行 = 方法与消融（PLAN §W4：A* 硬标签 BC / kino 硬标签 / kino 软标签 / P1 / P1+P2 / (P1+P2+P3)），
列 = 场景组 × {SR↑, CR↓, false_forward↓, SPL↑, task_success↑}。平均速度 / 平均步数 另表（`stats_report` 暂不出）。

| 方法 | Gazebo forest_e1 窄 | Gazebo forest_e1 中 | Gazebo forest_e1 宽 | 3DGS 训练场景 1..k | 3DGS 保留场景 1..2 | 自采场景 |
| --- | --- | --- | --- | --- | --- | --- |
| teacher（特权，上界） | SR / CR / ffwd / SPL / task | | | | | |
| forward（下界） | | | | | | |
| A* 硬标签 BC | | | | | | |
| kino 硬标签 | | | | | | |
| kino 软标签（无信念） | | | | | | |
| P1（信念软标签 + DAgger） | | | | | | |
| P1 + P2 | | | | | | |
| P1 + P2 + P3（探索） | | | | | | |

- 每格实际是 5 个子格（SR / CR / false_forward / SPL / task_success），排版时按 `stats_report` 的
  `cells` JSON 展开成 5 列一组。
- n：每个场景组 ≥ 20 个独立几何（M4 门槛），目标 ≥ 200 集/格（Wilson 半宽 p=0.8 时 ±5.5 pp）。
- 检验：每个场景组内 Cochran's Q（所有方法 × 共同 seed）作总体检验 + 全部两两 McNemar + Holm（族 = 本场景本指标的两两行）；写在表注。
- 训练种子：每方法 3 个种子，同名多次 `--eval`，格内自动变 IQM + 分层 bootstrap。
- 生成命令：

```
python tools/stats_report.py --out-dir $O/paper/main --stem table1_main \
  --metrics SR,CR,false_forward,SPL,task_success --pair-metrics SR,CR \
  --eval teacher forest-narrow $O/eval/teacher/forest-narrow/eval.json \
  --eval P1 forest-narrow $O/eval/P1-s0/forest-narrow/eval.json \
  --eval P1 forest-narrow $O/eval/P1-s1/forest-narrow/eval.json \
  --eval P1 forest-narrow $O/eval/P1-s2/forest-narrow/eval.json \
  --eval P1 gs-callinstrasse $O/eval/P1-s0/gs-callinstrasse/eval.json ...
```

## 表 2 教师消融（W2）

行 = 教师标签的构造方式，列 = 学生在各场景的 SR / SPL（学生架构固定为 P2 最终版，训练集与种子相同）。

| 教师 / 标签 | Gazebo forest_e1（三档合并） | 3DGS 训练场景 | 3DGS 保留场景 | 备注 |
| --- | --- | --- | --- | --- |
| A* 硬标签（现状 v5） | | | | 基线 |
| kino 硬标签 | | | | |
| kino 软标签，无信念 | | | | |
| P1 无 DAgger | | | | |
| P1 全量（DAgger 2–3 轮） | | | | |
| P1，K=1 / 4 / 16 | | | | 三行 |
| P1，QMDP vs CVaR 聚合 | | | | 两行 |
| P1，α × 0.5 / 1 / 2 | | | | 三行 |

- n：与表 1 同一请求集；每格 ≥ 20 独立几何，目标 200。
- 检验：以「P1 全量」为参照，两两 McNemar（SR）+ Holm；软标签还要报离线 KL / 宏 F1（§1.1）另表。
- 生成命令：同表 1，`--eval <教师变体名> <场景> <eval.json>`，`--pair-metrics SR`。

## 表 3 学生消融（W3）

行 = 学生输入 / 辅助头 / λ，列 = 场景 SR / CR / SPL，外加「动力学扰动（τ ±30%）后 SR 跌幅」一列。

| 学生 | Gazebo forest_e1 | 3DGS 训练 | 3DGS 保留 | τ −30% SR 跌幅 | τ +30% SR 跌幅 |
| --- | --- | --- | --- | --- | --- |
| 单帧，无辅助头（现状） | | | | | |
| 单帧 + 深度分带头 | | | | | |
| 双帧，无头 | | | | | |
| 双帧 + 分带头 | | | | | |
| 双帧 + 分带头 + prev-action（P2 最终） | | | | | |
| P2，TTC 头替代分带 | | | | | |
| P2，λ = 0.1 / 0.5 / 1 | | | | | |

- 「跌幅」= 基准 SR − 扰动 SR，两次评测同一 seed 集，区间用两比例差的 bootstrap（`stats_report` 的
  paired 输出里取 b/c 自行换算，或后续加 `--diff` 列；当前模板先手工列出 b/c）。
- n：同表 1。检验：每个扰动档 McNemar（基准 vs 扰动，同 seed）。
- 生成命令：每个扰动档一次 `stats_report`，场景名带后缀（如 `forest-tau-0.7`）。

## 表 4 跨场景泛化（训练集 A → 保留集 B）

行 = 训练集（`tools/compare_datasets.py` 的 A / B / AB 以及后续 3DGS 多场景），列 = 评测域 SR 与「泛化缺口」。

| 训练集 | 域内 SR（训练域） | Gazebo SR | 3DGS 训练场景 SR | 3DGS 保留场景 SR | 泛化缺口 = 域内 − 保留 |
| --- | --- | --- | --- | --- | --- |
| A = Gazebo 柱阵（e1-pipeline-20261107-0417/collection） | | | | | |
| B = 3DGS Callinstrasse（pilot-v3-fov60 → 放量） | | | | | |
| AB = A ∪ B（口径统一后才能打包，见 compare_datasets 预检） | | | | | |
| + Wanderland / DL3DV 场景（后续） | | | | | |

- 标注：Gazebo hold 1.0 s / observation，3DGS hold 0.5 s / render_observation；B 放量时统一到 1.0 /
  observation 后 AB 才能混包；在此之前 AB 行写「blocked」而不是空。
- n：Gazebo 15 个独立几何 seed（pilot）→ 目标 ≥ 20；3DGS 每路线 10 集 → 保留路线 ≥ 20 集。
- 检验：同一评测域内各训练集两两 McNemar + Holm；缺口列报 bootstrap 区间。
- 生成命令：`python tools/compare_datasets.py --out $O/compare`（守护进程，见文件头），产物
  `summary.md` / `summary.json` 即本表；重算只需 `--from summary`。

## 表 5 sim-to-real

行 = 方法，列 = 仿真 SR（同可穿越性桶）/ 真机 SR / 比值；真机 ≥ 100 次（SOUS VIDE 105 / Splat-Nav 124 口径）。

| 方法 | 可穿越性桶 | 仿真 SR (n) | 真机 SR (n) | 真机 CR (n) | 真机/仿真 SR 比 | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| P1 + P2 | 窄 | | | | | |
| P1 + P2 | 中 | | | | | |
| P1 + P2 | 宽 | | | | | |
| 基线 BC | 合并 | | | | | |

- 真机结果按 `eval.json` 同 schema 手工录入（`per_episode` 至少 `seed/success/collision/false_forward`），
  再走 `stats_report`，场景名 `real-<桶>`；SPL 若有外部定位（UWB/动捕）才填，否则「—」。
- n：真机每桶 ≥ 30、合计 ≥ 100；仿真同桶 ≥ 200。
- 检验：真机 vs 仿真是独立样本，不用 McNemar；报两比例差 + Wilson 区间，功效用 `--power`。
- 生成命令：`python tools/stats_report.py --eval P1P2 real-narrow $O/real/narrow/eval.json
  --eval P1P2 sim-narrow $O/eval/P1P2/forest-narrow/eval.json ...`

---

## 已有数据能填的格（2026-11-07 现状，全部 pilot）

- E1 四组（teacher / student-x3-old / student-e1-new / forward，Gazebo base，n=11 独立几何）：
  `$T/agent4-eval-matrix/e1-four-groups/e1_four_groups.md`。SPL 列为「—」：历史 eval 没留轨迹。
- 其余表都空；填表顺序按 PLAN §W4 的 M3/M4。
