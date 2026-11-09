# 2026-11-09 统一代价模型教师（costmap）：离线重标、闭环 ①②、软标签

状态：完成（闭环 ①② 与软标签闭环都跑完，结论见 §7）。任务书 `tmp/phase3-20261107/BRIEF_ALGO.md` §1 + `tmp/phase8-20261109/BRIEF.md` §4；
断点 `tmp/agent10-teacher-cost/PROGRESS.md`；原始输出 `$D/exp-teacher-cost-20261109/`（`$D=/root/rivermind-data/uav_rgb`）。

## 0. 代价模型（与 DISTILL §2.1 / `agent6-distill/q1_cost_to_go.py` 同源，一处修正）

- 目标 `(max_px + 2.0, 0)`（与 `_global_path` 同）；goal-stop 任务用 waypoint。
- 测地场 D：从目标格出发的 Dijkstra，栅格 0.1 m、膨胀 = A* 余量（E1 base 0.40）、8 邻接禁斜穿；栅格范围 / 移动集 / 可行判定与 `astar_path` 共用同一套 helper（`_grid_spec` / `_MOVES` / `_point_blocked`）。落在膨胀带内的点取 3×3 邻域里可行格的 D + 一跳（≈ legacy 0.30 局部余量的语义）；邻域全在带内 → 不可行。
- 四原语各滚一步：FWD 沿航向 0.25 m；YAW ±27° 并沿新航向平移 0.11 m；SAFE 原地（`TIE_*` 常量，与 `lr_tie_margin_steps` 同一组）。
- `Q(a) = 1 + D(x',y')/0.25 + max(0, |err'| − tol)/27`，err' = 滚出后航向与 D 在 (x',y') 处 0.7 m 胡萝卜方向之差；撞机身方框（半宽 0.102 + 0.01）或落进不可行格 → 40 步。默认 tol = 25°（legacy 容差同值）。
- 标签 = argmin；`|ΔQ| < 1e-6` 的平局：两个 YAW 平局按可见柱反侧，否则按枚举顺序（SAFE, FWD, L, R）。FWD/L/R 全是 40 → 退回 `plan_action`（info.fallback = `all_collide`）；SAFE 不参与这一判定，否则正对柱 0.4 m 处 Q=[15.9, 40, 40, 40] 会原地锁死。
- **修正（相对 q1）**：胡萝卜方向沿「最陡下降」走 0.7 m 时，下降率按单位长度比、同率先直走。q1 按「邻格 D 最小」取，octile 距离下直走与斜走每米降得一样多，只要有一点横向偏差就永远斜走——开阔走廊里横向偏 7 cm 就给出 45° 的胡萝卜，FWD 滚出后 err' ≈ 44° 吃航向罚。0417 上这一条让 2297 帧 legacy FWD 被判成 YAW；修正前一致率 72.5%（= q1 的 72.2%），修正后 84.5%。

版本常量 `teacher.TEACHER_COST_VERSION = "uavrgb.teacher_cost.v1"`；采集 / 重标的 manifest 写 `planner=costmap`、`costmap_tol_deg`、`costmap_lookahead_steps`；行内 `q_costs`（Q[SAFE, FWD, L, R]，步）。audit `label_recompute` 按 `planner` 选 `plan_action_costmap` 复算（重标副本 2 集 smoke：0/80 不一致）。

回归锁（legacy / visible 教师数值不变）：`sim/tools/test_teacher_costmap.py --data`——0417 300 集 12000 帧 legacy 复算 0 不一致、sgvis 300 集 10659 帧 visible 复算 0 不一致且行内 `tie_margin_steps` 与重算逐位相同；8 组硬编码 legacy Q/margin 锁、2 组 `astar_path` 哈希锁。

## 1. 离线重标（`tools/relabel_costmap.py`）

0417 全量 12000 帧（`$D/e1-pipeline-20261107-0417/collection`），默认 tol=25、单步：

| | SAFE | FWD | L | R |
|---|---|---|---|---|
| legacy 标签 | 229 | 9372 | 1196 | 1203 |
| costmap 标签 | 616 | 9995 | 709 | 680 |

一致率 **84.52%**（10142/12000），fallback `all_collide` 263 帧（= legacy 被围死 SAFE 的那些），平局 30。混淆（行 legacy、列 costmap）：

| legacy \ costmap | SAFE | FWD | L | R |
|---|---|---|---|---|
| SAFE | 213 | 14 | 2 | 0 |
| FWD | 37 | 8953 | 195 | 187 |
| L | 164 | 521 | 497 | 14 |
| R | 202 | 507 | 15 | 479 |

分区域（`q2_fov_explainability.zone_of` 口径：approach x<1.0 / in_field ≤ max_px+0.2 / exit ≤ max_px+1.0 / 线后）与分视锥：

| 子集 | n | 一致率 | legacy 分布 | costmap 分布 |
|---|---|---|---|---|
| approach | 2697 | 0.845 | [2, 2183, 235, 277] | [13, 2451, 113, 120] |
| in_field | 5978 | 0.903 | [227, 4980, 408, 363] | [228, 5394, 191, 165] |
| exit | 1170 | 0.827 | [0, 962, 112, 96] | [0, 1163, 5, 2] |
| post_success_line | 2155 | 0.695 | [0, 1247, 441, 467] | [375, 987, 400, 393] |
| 视锥内无柱 n_vis=0 | 4626 | 0.809 | [15, 3497, 561, 553] | [356, 3519, 384, 367] |
| 视锥内有柱 | 7374 | 0.868 | [214, 5875, 635, 650] | [260, 6476, 325, 313] |

分 legacy 航向误差 |err|（A* 胡萝卜）：[0,10) 98.3%（6385）；[10,25) 85.2%（3043）；**[25,45) 42.3%（1434）**；[45,90) 53.3%（291）；[90,180] 29.7%（303）；无路 78.3%（544）。

读法：
- 不一致集中在 legacy 刚过 25° 容差就转的帧：代价模型里 FWD 的路长收益（≈cos e 步）压过航向罚（(e−25)/27 步）直到 e ≈ 35°，costmap 比 legacy **晚转约 10°**。exit 区 legacy 的 208 个 YAW 在 costmap 里只剩 7 个（看不见的目标点的函数，DISTILL §3.1 要砍的那类）。
- costmap 多出的 SAFE（616 vs 229）里 375 帧在成功线后、离目标 (max_px+2) 不到一步：按代价模型到了目标就该停，这是结构性结果；截断集（trunc）里没有这些帧。
- 线后 legacy 的 908 个 YAW 在 costmap 里变成 SAFE / 左右交替（闭环 ① 的教师过线后也是这样，见 §4）。

`trunc/col`（到成功线截断，9845 帧）：一致率 **87.81%**；costmap 分布 [241, 9008, 309, 287]（legacy [229, 8125, 755, 736]）；视锥内无柱的 2625 帧里 costmap YAW 35 个（legacy 293）。

变体（只算统计，不进闭环）：

| 变体 | 一致率 | costmap 分布 | 平局 | 备注 |
|---|---|---|---|---|
| tol=25，单步（默认） | 84.52% | [616, 9995, 709, 680] | 30 | 闭环 ①② 与软标签都用它 |
| tol=0 | 80.81% | [690, 8412, 1483, 1415] | 3 | |err|∈[10,25) 一致率 55.5%：legacy 容差内的 FWD 被判成转 |
| 两步前瞻 | 83.72% | [808, 9133, 1042, 1017] | 390 | 第二步可取 SAFE 使 Q 并列，平局多 13 倍；线后 SAFE 591 |

修正前（q1 同款 descent）的数字留档于 `$D/exp-teacher-cost-20261109/relabel-flawed-descent/`：0417 72.47%、costmap YAW 占 31.6%、不一致以 legacy FWD → costmap YAW 为主（2297 帧）。

## 2. 软标签离线 4 臂 + 闭环 ② 学生（训练数据 = trunc/col 的 costmap 重标副本，9663 样本，`--drop-collision`）

训练器 = `$D/exp-teacher-cost-20261109/bin/uavrgb_train`（sha 625389a1，check.sh 17:35 构建），30 轮 / batch 128 / lr 1e-3 / seed 0。软标签：`tools/export_costmap_soft.py`（logits = log softmax(−Q/α)）→ `tools/join_soft_targets.py --temperature 1.0`（9663/9663 覆盖）→ `--soft-targets … --distill-alpha 1.0`（训练器禁止同时给类权重）。

| 臂 | 类权重 | best ep | val macro-F1 | val acc | 召回 L / R（val 仅 12 / 10 帧） | 末 10 轮 val acc sd | 空画面 YAW 率（0417 val+test n_vis=0，710 帧） | 有柱帧 YAW 率（1090 帧） |
|---|---|---|---|---|---|---|---|---|
| trunc-legacy（对照，agent6 `trunc/run`，legacy 标签） | auto | 28 | 0.426 | 0.831 | 0.43 / 0.45 | 0.083 | **8.2%** | 19.6% |
| relabel-cw（闭环 ② 学生） | auto | 21 | 0.348 | 0.878 | 0.42 / 0.40 | 0.036 | 0.4% | 17.5% |
| hard-argmin | 无 | 13 | 0.412 | 0.963 | 0.25 / 0.20 | 0.008 | 0.0% | 2.7% |
| soft α=0.25 | 无 | 16 | 0.423 | 0.967 | 0.42 / 0.10 | 0.006 | 0.0% | 2.4% |
| soft α=0.5 | 无 | 5 | 0.391 | 0.965 | 0.17 / 0.20 | 0.010 | 0.0% | 1.2% |
| soft α=1.0 | 无 | 5 | 0.391 | 0.965 | 0.17 / 0.20 | 0.006 | 0.0% | 1.0% |

读法：
- 空画面 YAW 率是 DISTILL §2.5 要压的指标：legacy 截断臂 8.2% → 重标臂 ≤ 0.4%。但代价是 YAW 整体变稀（costmap 标签 YAW 占 6%），无类权重的 4 臂在有柱帧上也几乎不转（1–3%），val 上 22 个 YAW 帧只召回 2–5 个。
- 预注册「软臂 L/R 召回差与末 10 轮 acc sd 不高于 hard_argmin」：α=1.0 两项都成立（0.033 vs 0.050；0.006 vs 0.008）；α=0.5 召回差成立、sd 高 0.002；α=0.25 召回差 0.32 不成立。val 只有 22 个 YAW 帧，这一项的分辨率是 1/10，只能说「没看出软标签让左右混淆变糟」，说不出更多。
- 离线 macro-F1 不能跨标签体系比（relabel 臂的 val 真值是 costmap 标签）。

## 3. 闭环条件（两臂同条件）

`cl1/conditions.json`：lib v3.1（`libCfSitlSystem.so` 4f41a41a、`libcf_sitl.so` 75ca0301）、`e1_mini.json` 683e7400、RTX 2080 Ti、worktree 2899428 + 本轮改动；`eval_closed_loop.py --task base --split test --seeds <CFG eval_seeds 20 个> --episodes 20 --steps 80 --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation`（= `e1_pipeline.evaluate`，不录像）；`metric_version sim09.v1.coast1s`。eval4 还没落地，对照臂全部自己跑。

## 4. 闭环 ①：costmap 教师 vs legacy 教师（各 20 集）

预注册：costmap SR ≥ 19/20 且碰撞 0；否则不可用作示范教师。**结果：成立。**

| 臂 | SR | 碰撞 | SPL | 1600 步动作分布 SAFE / FWD / L / R | max_x 中位 |
|---|---|---|---|---|---|
| costmap 教师（`--teacher-planner costmap`） | **20/20** [0.84, 1.00] | 0 | 0.948 | 407 / 655 / 262 / 276 | 7.10 |
| legacy 教师（同条件对照） | 20/20 | 0 | 0.963 | 0 / 796 / 358 / 446 | 7.22 |

两臂每集都跑满 80 步（sim09.v1 过线后继续飞）。柱区内两条动作序列几乎一样（例 seed 1 前 33 步：legacy `FFFRFFFFFFLFFFFFFFFFFFFFFFFLFFFFFFLL…`，costmap `FFFRFFFFFFFFLFFFFFFFFFFFFFFLFFFLFFF…`，costmap 少转 2–3 次）；过了目标 (max_px+2) 后 costmap 原地 SAFE / 左右一步交替（D≈0 时 SAFE 最便宜），legacy 继续连转十几步——costmap 的 407 个 SAFE 全在这一段，SPL 低 0.015 也来自这里（path_length 多走的那点）。`stats_report`：两教师 McNemar b=c=0。

## 5. 闭环 ②：重标学生（relabel-cw）vs 截断臂（`ckpt_trunc_distill.bin`）

预注册：重标学生不比截断臂差（McNemar 不一致对里截断赢的不多于重标赢的）；掉到截断臂一半以下即代价模型错。**结果：成立（14 vs 12，不显著）。**

| 臂 | 训练数据 | SR | 撞柱 / 乱飞 | SPL |
|---|---|---|---|---|
| relabel-cw（costmap 重标 trunc/col，类权重 auto） | 9663 样本 | **14/20** [0.48, 0.85] | 3 / 3 | 0.61 |
| trunc-legacy（eval3 的 `ckpt_trunc_distill.bin`，同条件重评） | 同一批图、legacy 标签 | 12/20 [0.39, 0.78] | 3 / 5 | 0.52 |

McNemar（同 20 seed）：重标赢 6、截断赢 4，精确二项 p=0.754；Cochran's Q 四臂 p=0.001 只由教师 vs 学生贡献。逐 seed：

| seed | relabel-cw | trunc-legacy |
|---|---|---|
| 1 | ✓ | ✓ |
| 113 | 撞柱 (x 3.5) | ✓ |
| 122 | 撞柱 (x 3.0) | 撞柱 (x 1.6) |
| 153 / 172 / 176 | ✓ | ✓ |
| 180 | ✓ | 乱飞 (x 4.2) |
| 230 | 乱飞 (x 3.3) | ✓ |
| 250 / 536 | ✓ | ✓ |
| 555 | 乱飞 (x 3.6) | 乱飞 (x 1.9) |
| 628 | 撞柱 (x 2.5) | ✓ |
| 683 | ✓ | 撞柱 (x 1.3) |
| 798 | 乱飞 (x 5.6) | ✓ |
| 929 | ✓ | 乱飞 (x 2.4) |
| 990 | ✓ | ✓ |
| 992 | ✓ | 乱飞 (x 5.9) |
| 997 | ✓ | 撞柱 (x 6.5) |
| 1113 | ✓ | 乱飞 (x 5.5) |
| 1654 | ✓ | ✓ |

动作分布：relabel-cw SAFE 13 / FWD 1318 / L 58 / R 41（1430 步）；trunc-legacy SAFE 2 / FWD 1270 / L 119 / R 48（1439 步）。重标学生转得少一半，乱飞少 2 集、撞柱持平。注意这个截断臂在 eval3（lib v3、4080）是 13/20，这里 lib v3.1 + 2080 Ti 同条件重评是 12/20，和 BRIEF §1 说的一致：旧数字不能直接拿来比。

## 6. 软标签闭环（α=0.5 一臂，排在 ①② 之后）

预注册：SR 不低于重标硬标签学生（14/20）；低于一半 → 软标签线停。**结果：被证伪，5/20，软标签线停。**

| 臂 | SR | 撞柱 / 乱飞 | 其中首柱前（x<1.5）撞 | SPL | 动作分布 SAFE / FWD / L / R |
|---|---|---|---|---|---|
| soft α=0.5（无类权重，`--distill-alpha 1.0`） | **5/20** [0.11, 0.47] | 14 / 1 | 8 | 0.22 | 0 / 632 / 14 / 8（654 步） |
| relabel-cw（§5） | 14/20 | 3 / 3 | 0 | 0.61 | 13 / 1318 / 58 / 41（1430 步） |

McNemar soft-0.5 vs relabel-cw：b=11（硬赢）、c=2，精确二项 p=0.022（Holm 0.135）；vs trunc-legacy 8:1，p=0.039。成功的 5 集（122 / 172 / 176 / 536 / 798）都是不用转就能过的布局。

读法：闭环结果与 §2 的离线指标一致——软臂在有柱帧上的 YAW 率只有 1.2%（val 22 个 YAW 帧只召回 4 个），closed loop 里就是直飞撞首柱。原因不在 α 本身：costmap 标签 YAW 只占 6%，训练器又禁止软标签和类权重同用，4 个无类权重臂（含 hard-argmin）全都把 YAW 学没了；而带类权重的 relabel-cw 用同一批标签能到 14/20。所以这一轮能下的结论是「**软标签 + 无类权重 在 costmap 标签分布下不可用**」，不是「软标签本身无用」；要再试软标签得先解决类不平衡（训练器允许软标签 + 类权重，或对 FWD 帧降采样），那是训练器文件域的事，本轮没做。

## 7. 结论

1. costmap 教师可用作示范教师（20/20、0 碰撞，与 legacy 同条件同成绩），但不是「更强的」教师——它的差异只在转弯阈值晚约 10° 与过目标后停下。
2. costmap 重标的学生（类权重 auto）14/20 vs 截断臂 12/20，不显著（p=0.754）；方向符合 DISTILL §2.5 的预期（空画面 YAW 率 8.2% → 0.4%，乱飞 5 → 3），但样本不够说「更好」。
3. 软标签 α=0.5 闭环 5/20，预注册证伪，软标签线按 BRIEF_ALGO §1.6 的证伪条件停。
4. 代价模型没动过任何动作切换惩罚 / 滞回 / 权重网格；唯一的「调参」是修了 descent_dir 的量化 bug，并写进 `test_teacher_costmap.py`。
