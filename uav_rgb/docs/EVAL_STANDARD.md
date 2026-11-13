# VLN 复核补充（2026-11-13）

`sim/3dgs/run_vln.py` 从 `3dgs.vln.v2` 起：`n_requested=n+n_failed`，SR/OSR 的分母固定为请求集，异常按未成功计；
SPL 只在有参考路径的有效集平均（`n_spl`），NE/碰撞率/停止率按有效集 `n`。每项分母写入 `denominators`，必须同时报告错误率。
有效集是记录非空且一致、无执行错误、物理和相机校验通过的集；导航碰撞仍是有效的失败结果，不能作为异常剔除。
评测 manifest 使用 `diag-vln-eval`，默认不得作为训练数据；SAGE 原始 split 和几何 split 是不同字段，不能混作场景泛化划分。

VLN 实际路径和最短参考路径均从首动作**之前**的位置起算；旧 v1 的最短参考路径漏掉首步，旧数保留版本，不覆盖。
当前距离是 xy 欧氏距离、最短参考路使用占据 A* 与软余量并截到目标圆盘，STOP 是连续3步SAFE的历史代理，
这些是本项目协议，不能称为与 VLN-CE/SAGE 官方完全同口径。下一版分离安全等待与任务STOP时须再次升版本并重跑对照。

# 当前判据补充（2026-11-03）

`--task campus` 使用 `visible_goal_reach_v1`：无contact且轨迹采样曾进入彩柱前0.5m目标区；
不要求停稳。默认dispatch。整拓扑留出，源于校园v1元数据，不使用旧L1的1.5m半径。
必须同时报告请求集成功率、完成比例、failed/skipped和共同有效场景的配对结果。
新报告记录appearance_version、goal_radius、模板/资源/world哈希及固定请求seed清单。
`stop_success` 仍仅为末位置诊断，不能称低速停稳成功。

历史 `goal_reach`（19/47=0.4043）与重算 `goal_reach_ever`（0.574）不是模型提升。
新碰撞事件修复意味着历史碰撞率也须复跑后才可直接比较。
具体地图与可见性边界见 `CAMPUS_BASELINE.md`。下方历史任务定义保留用于复现。

# 评测标准（公开口径）

- 更新：2026-10-29
- 适用：`uav_rgb` 的学生模型（单目 RGB → 四动作）与后续闭环评测
- 本文是**约束文件**：以后报数字必须按这里的口径，不许临时自造指标。

## 0. 为什么要这份文件

我们此前的评测只有两个数字：`action_accuracy`（准确率）和交叉熵。两者都是自己定的，
而且在这个数据集上会**主动掩盖问题**：

| 项目 | 数值 | 说明 |
| --- | --- | --- |
| 数据类别分布（4000 步） | FORWARD 75.2% / LEFT 13.1% / RIGHT 11.7% / SAFE 0.03% | 极不平衡 |
| 「全预测 FORWARD」基线 | 准确率 75.2% | 什么都不会也有 75% |
| 本轮实测（dev，epoch 13） | acc **0.976** / macro_f1 **0.724** / balanced_acc **0.735** | 准确率虚高 |
| SAFE_NEUTRAL 召回率 | **0.000**（每个 epoch 都是 0） | 准确率完全看不出来 |

所以：**准确率不能作为结论指标**，必须同时报宏平均指标；**离线指标不能作为可飞性证据**，
闭环指标才是。

## 1. 两层标准

模型有两类问题要回答，对应两套公开口径，缺一不可。

### 1.1 离线（分类）：scikit-learn 定义的宏指标

来源：`sklearn.metrics.f1_score(average="macro")`、`balanced_accuracy_score`、
`confusion_matrix`。这三者是类别不平衡下的事实标准，不是我们发明的。

实现：`src/include/uavrgb/eval_metrics.hpp` + `src/eval_metrics.cpp`。

| 指标 | 定义 |
| --- | --- |
| 混淆矩阵 | C×C，行 = 真值，列 = 预测 |
| 每类召回 recall_k | `M[k][k] / Σ_j M[k][j]`，该类无真值样本时记 0 |
| 每类精确率 precision_k | `M[k][k] / Σ_i M[i][k]` |
| 宏平均 F1 | `(1/C) Σ_k 2·P_k·R_k/(P_k+R_k)`；`P_k+R_k=0` 时该类的 F1 记 0 |
| 平衡准确率 | `(1/C) Σ_k recall_k` |

落盘位置：`metrics.jsonl` 的 `validation` 事件新增 `macro_f1`、`balanced_accuracy`、
`confusion`（16 个整数）。既有键名不变，只追加。

**报告规则**：任何一次训练结论必须同时给出 `acc`、`macro_f1`、`balanced_accuracy`
和**每类召回**。只报 acc 的结论视为无效。

### 1.2 闭环（避障）：AvoidBench 的七项指标

来源：
- 论文：**AvoidBench: A high-fidelity vision-based obstacle avoidance benchmarking suite for multi-rotor**，ICRA 2023，arXiv:2301.07430
- 代码：`github.com/tudelft/AvoidBench`（基于 Flightmare + Unity3D）

这是目前公开文献里唯一给出了**完整、可复现定义**的视觉避障评测协议。我们直接沿用它的
指标定义，不自己另起一套。

| 指标 | 公式 | 方向 |
| --- | --- | --- |
| 成功率 SR | `T_finished / T_whole` | ↑ |
| 路径最优性 PO | `(d_trav − d_min) / d_min × 100%` | ↓（0% = 没有绕路；100% = 飞了两倍距离） |
| 能量最优性 EO | `∫₀^{t_trial} ‖jerk‖₂ dt` | ↓ |
| 平均目标速度 AGV | `d_min / t_trial` | ↑ |
| 任务进度 MP | `(1 − |a·c / |a|²|) × 100%`，`a`=起点→终点，`c`=无人机终点→目标点 | ↑ |
| 处理时间 | 单次推理 + 决策的墙钟时间（机器相关） | ↓ |
| 对比因子 | `ln SR_Ξ1 / ln SR_Ξ2 = (d_minΞ2 · ln SR_Ξ1)/(d_minΞ1 · ln SR_Ξ2)` | 比较用 |

其中 `d_min` 是起终点之间的**最短自由路径长度**（AvoidBench 用 A\* 算），不是直线距离。

环境侧另有两个指标：**traversability**（可通行度）与 **relative gap size**（相对缝隙宽度）。
它们描述地图而不是算法；我们的 `scene_gen.py` 已有缝隙下限 `MIN_GAP=0.9 m` 和
车道间距 `LANE_SEP=1.6 m`（T1 起，横向车道缝 1.2 m），可以直接换算成 relative
gap size。旧 v2 几何（`LANE_SEP=1.302 m`，缝 0.902 m）的数据不能与 v3 混用。

## 2. 我们能不能直接跑 AvoidBench

**不能。** 原因：

- AvoidBench 依赖 ROS noetic（ROS 1）+ Unity3D + Flightmare + catkin 构建。
- 本项目**明令禁止引入 ROS 2**，本机也没有 ROS 1，且仿真栈是 Gazebo Harmonic。
- 引入 Unity3D 渲染需要替换现有 Gazebo/Ogre2 仿真链路（当前默认 NVIDIA EGL）。

所以我们的做法是：**实现同一套定义，不跑同一套套件**。报数字时必须写明
「指标定义沿用 AvoidBench (ICRA 2023)，评测环境为本仓库 Gazebo 场景，非 AvoidBench 套件」。
任何暗示「我们跑了 AvoidBench」的表述都是错的。

## 3. 我们的场景怎么映射到 AvoidBench

| AvoidBench 概念 | 本仓库对应物 | 当前状态 |
| --- | --- | --- |
| trial（一次起终点飞行） | 一个 episode：出生点 → `goal_x`（最右柱 x + `--goal-margin`）；L1 v2 例外：飞到目标柱 waypoint 并停 | ✅ 已实现；旧口径成功 = 无碰撞且 `max_x ≥ goal_x`；L1 v2 成功 = 无碰撞 + 最终距 waypoint ≤ `stop_radius`+0.2 + 末动作 `SAFE_NEUTRAL` |
| 起点 / 终点 | `(-0.5, 0, 0.15)` 出生点 / `goal_x` | ✅ 已实现；`CRUISE_END_X` 已废弃 |
| `d_trav`（实飞路径长度） | `oracle.jsonl` 的 `pose_gt` 轨迹累加 | ✅ 数据已在 |
| `d_min`（A\* 最短自由路径） | 由 `scene.json` 的柱子几何做栅格 A\* | ❌ **未实现** |
| 碰撞判定 | `contact` 传感器 + `stuck`（位移 < 阈值） | ⚠️ 见 §5.1 |
| 处理时间 | 前向 + 决策的 p50/p95 | ✅ 已实现（`eval.json` 的 `inference_ms`）；本机 GPU，不是板端 |
| SR / PO / EO / MP / AGV | 由上面几项算出 | SR ✅；PO / EO / AGV / MP ❌ |

## 4. 数据切分标准

- **按 scene_id 隔离**：同一场景的所有帧只能落在一个 split。`dataset.cpp` 的
  `assign_split()` 在加载时就拒绝跨 split 的 scene_id，不是靠约定。
- **按几何组分配 split**（2026-10-29 18:20 起）：`sim/tools/split_manifest.py` 把
  障碍几何规范化后取 sha256 得到 gid，再用固定 salt hash 决定
  `train / val / test`。同几何不同 seed 永远同 split，跨 split 交集为 0。
  旧策略 `scene_id % 10` 在 seed 0..1999 上实测有 241 组跨 split 泄漏，已废弃。
- **salt 已升到 v2**（2026-10-29 22:00）：v1 把 `-0.0` 格式化成 `-0.000`，
  seed 337 与 3651 的同几何被分到 test/train 两边。修复后 gid 变化，
  v1 时期采集的数据必须重采或显式重标，不能与 v2 数据混用。
- `collect.py` 在带 `--scene-meta` 时会核对 `--split` 与几何组的固定 split；
  `preprocess.py` 会拒绝同一 `geometry_gid` 出现在两个 split。
- **val 用来看，test 用来报**。test 只在训练结束时评估一次，不参与任何模型选择。
  实现：`src/trainer.cpp` 训练循环之后的 `Split::kTest` 事件；
  best-checkpoint 只按 dev 宏 F1 选择，不看 test。
- 同一 scene_id 生成两次必须字节级一致（`scene_gen.py --self-check` 用 sha256 保证）。
- 当前场景族只有 1 个（`family_holdout=false`）：几何不重复不等于跨族泛化。

## 5. 已知的、会污染评测的问题（必须写进任何结论里）

> **更新 2026-10-30**：下面第 0/1/2 条的旧数据问题已由 T1/T2 修复（v3 车道缝 1.2 m、
> SAFE 21.2%、599 集 23960 步 audit 11/11 PASS）；保留原文作为历史记录。§6 的
> SR 与 p50/p95 已实现，`d_min`/PO/EO/AGV/MP 仍缺。

0. **episode 太短，无人机从未进入障碍场。**（2026-10-29 上午实测，最严重的一条）

   > **2026-10-29 19:55 后续**：专家改用 GT carrot planner、默认 `--steps 80` 后，
   > seed 0..7 有 5/8 在 68–76 步穿过 x=5.0（见
   > `experiments/2026-10-29-teacher-traverse-ablations.md`）。下面这组旧数据
   > 仍说明旧数据集的分布，但不再是“当前专家一定进不去”；采集不可复现的问题仍在。

   | 事实 | 数字 |
   | --- | --- |
   | 柱子场 x 范围 | 1.5 – 5.0 m（纵深 3.5 m） |
   | 200 条 episode 终点 x | 中位数 **1.109 m**，p90 1.37 m，最大 2.053 m |
   | 到达首排柱子前表面（x=1.30） | 42 / 200 |
   | 到达首排柱子中心（x=1.50） | 6 / 200 |
   | **穿过整个场地（x≥5.00）** | **0 / 200** |

   根因：`--hold` 是**墙钟**窗口。8 路并行时软件 GL 把 CPU 打满，实测每步只推进
   0.42 s 仿真时间（RTF≈0.42）。`SPEED_FWD=0.3 m/s` 下每步只前进 0.126 m，
   20 步约 1.7 m —— 不到场地深度的一半。要用 40 步纯前进才能从 x=0 到 x=5.0，
   加上转向约需 45–50 步。

   **在这个问题修好之前：任何「避障」结论都不成立。** 模型学到的是「起飞后往哪边转」，
   不是「在障碍之间穿行」。连带后果：动作序列只剩 79 种（200 条 episode），
   最常见的是「转 3 下 → 直飞到底」。

1. **`stuck` 不是碰撞。** 采集数据里 274 个 collision 标记中，**238 个是 `stuck`
   （FORWARD_SLOW 位移 < 阈值），只有 30 个是真 contact**。
   - 其中 **94 次发生在 step 3**，是起飞后加速没完成，属系统性的假阳性。
   - 其余发生在 step 9–15，是专家真的卡在柱子上。
   - 直接把 `collision` 当碰撞率会**高估 8 倍**（6.85% vs 0.75%）。
2. **SAFE_NEUTRAL 在数据里几乎不存在。** 丢碰撞步后的实测：

   | split | N | SAFE | FORWARD | LEFT | RIGHT |
   | --- | --- | --- | --- | --- | --- |
   | train | 2981 | **1** | 2196 | 408 | 376 |
   | val | 370 | **0** | 271 | 42 | 57 |
   | test | 375 | **0** | 282 | 65 | 28 |

   后果有两个：
   - 四动作空间实际退化成三动作。val/test 里 SAFE 一个样本都没有，**这个类根本无法被评估**，
     四类宏平均 F1 因此被结构性地锁在 3/4 附近（实测 0.72）。
   - `--class-weights auto` 的逆频率权重在 `n_c=1` 时退化：训练出来的权重是
     `[745.3, 0.339, 1.83, 1.98]`，那一个样本独占 1/4 的总 loss。代码已改为在
     `n_c<=1` 时打警告，但这只是留痕，不是修复。

   **在这个问题解决前：不许说「模型支持四动作」；不许把四类宏平均 F1 当成学习信号；
   报空类指标时必须同时报 support（`confusion` 里已有）。**
3. **场景多样性低。** 所有场景都是同一套车道网格，只有 y 偏移不同。val/test 上的高准确率
   可能来自记住了车道结构，而不是看懂画面。**离线指标再高也不能证明泛化。**

### 5.1 闭环评测口径（2026-10-30 补充）

- **episode 终止条件 = `contact` ∨ `stuck`**（`eval_closed_loop.py --stop-on-collision`）。
  `stuck` 单独计率，不等同碰撞；被挡后不再继续转向脱困。
- **SR 分母 = `episodes_requested`**（2026-11-11 起，见 §5.6）。完成集 `episodes_done` 是 `collect_rc==0` 且 `steps>0` 的集；
  采集失败（含学生推理进程崩溃）进 `failed`，与跳过的集一起在分母里记 0，所以失败集不会抬高绝对 SR。
  报告必须同时给 `failed` 数。越低越好的指标（CR、stuck、false_forward）分母仍是完成集。
  `eval.json` 里 `success_rate` 等旧率字段仍是完成集分母，`requested_success_rate` 是请求集分母。
  盲前进基线同口径，两个基线对比仍公平。
- **处理时间 p50/p95** 是学生策略前向 + 决策，不含物理执行；本机 GPU 数字，不是板端。
- **L1 goal-reach-ever（2026-11-02 起，主指标，`success_criterion="goal_reach_ever"`）**：
  `stop_radius` 非空时 success = 无碰撞 + **轨迹上曾**进入 waypoint 的
  `REACH_RADIUS`（1.5 m）球内（`waypoint_min_dist ≤ 1.5`）。
  **改口径的原因**：waypoint 是柱阵外侧一个没有视觉标记的点，原口径要求「末位姿」
  落在 1.5 m 内，实际在考「停的时机」。实测有集次飞到距航点 0.03–0.17 m 却继续
  飞走 4–10 m，按末位姿算「完全没到」，和「根本没走过去」混为一谈，掩盖了导航能力。
  同一批数据两种口径的差距很大（v4 keep 47 集：0.574 vs 0.404）。
  **不要求末动作是 `SAFE_NEUTRAL`**：相机 60° 视场在真停止点上 40% 的帧一根柱子
  都看不到，「停」本身不可观测（见 `worklog/2026-11-01-2130-stop-point-unobservable.md`）。
  1.5 m 由几何定，不是调出来的：< 两候选 waypoint 最小间距 3.23 m 的一半
  （否则停在中间的无人机对 keep 和 shuffle 都算成功，反事实失效），
  < 出生点到 waypoint 的最小距离 2.50 m（否则原地不动也算成功）。
  **语言证据仍然成立**：v4 shuffle 47 集在新口径下依旧是 **0.000**（keep 0.574），
  说明放宽的是"到位判定"，不是语言判定。
- **L1 goal-reach-stop（2026-11-01 ~ 11-02 的历史口径，`success_criterion="goal_reach"`）**：
  success = 无碰撞 + **最终位姿**距 waypoint ≤ `REACH_RADIUS`（1.5 m）。
  现在仍照常计算并写入 `stop_success_rate` 与 per_episode 的 `stop_success`，用于对照
  「会不会停」，但不作为主指标。**两种口径的 SR 不可混报。**
  `eval.json` 顶层记录 `success_criterion`/`stop_radius`/`text_dim`，
  per_episode 记录 `waypoint_min_dist`/`final_waypoint_dist`；`stopped` 仅作诊断。
- **L1 goal-stop（2026-11-01 之前的历史口径，`success_criterion="goal_stop"`）**：
  success = 无碰撞 + 最终位姿距 waypoint ≤ `stop_radius + 0.2 m` + 末动作
  `SAFE_NEUTRAL`。**两种口径的 SR 不可混报。**
- **L1 route-choice（历史，v1 数据）**：success 还必须满足 `waypoint_min_dist ≤ 1.0 m`。
  v1 数据已作废，该口径只用于读旧产物。
- **L/I/S 文本消融**：`--text-mode keep/blank/shuffle`。blank/shuffle 前必须确认
  ckpt 的 `text_dim > 0`，否则与 keep 等价（eval 已强制检查）。shuffle 是**把关系词
  反过来**（left↔right）：v5 起两根目标柱同色，换颜色在图像里不可观测。
- **关系词定义**（l1.v4 起）：`left` 指**经过时**目标柱在机身左侧——waypoint 落在
  柱子外侧，无人机从外侧绕过去，柱子留在它的 left/right 那一侧（500/500 集实测成立）。
  这与 v3 的「出生时刻机身左侧」是**相反的口径**：v4 把 waypoint 移到外侧后，目标柱在
  出生时位于关系词的**另一侧**。两版数据不可混报。两根同色柱必须分居机头轴线两侧且横向跨距 ≤2.4 m
  （60° 水平视场装得下），否则该集不成立。
- **终止条件补充**：L1 v2 的 teacher 采集在连续 3 步 `SAFE_NEUTRAL` 后提前结束。
  （v4 起 safe 出生模式已删除，`safe_full_length` 闸门随之移除。）

### 5.2 SIM-09 指标合同（`metric_version="sim09.v1"`，2026-11-04）

旧 `success` / `stop_success` 的定义和数值**逐位不变**；下表字段是另写的诊断口径，
不得覆盖或重命名旧字段。per-episode 与 `eval.json` 顶层都写 `metric_version`。

| 字段 | 定义 | 分母 |
| --- | --- | --- |
| `reach_ever` | 无 contact 且 `waypoint_min_dist ≤ goal_radius_reach`（1.5 m） | 每集 |
| `final_in_radius` | 无 contact 且 `final_waypoint_dist ≤ goal_radius_stop` | 每集 |
| `stop_action_issued` | 最近 waypoint 点**前后各 2 步**内出现过 `SAFE_NEUTRAL`；不只看末步 | 每集 |
| `collision_free` / `stuck_free` | 无 contact / 无 stuck | 每集 |
| `task_success` | `reach_ever` + `final_in_radius` + `stop_action_issued` + `collision_free` + 未超时 | 每集 |
| `stable_termination` | 固定 `None`（未知）。没有可靠速度/姿态窗口数据时不得声称通过 | — |

- **半径**：L1 训练目标是 0.5 m 停、评测按 1.5 m 到达。`goal_radius_reach` 固定 1.5 m；
  `goal_radius_stop` 取 scene/meta 的 `stop_radius`（L1 为 0.5 m，campus 为 0.5 m）。
  `eval.json` 同时写 `goal_radius_reach` / `goal_radius_stop`，旧的 `goal_radius` 保留。
- **汇总分母**：`success_rate` 等旧率 = `episodes_done`；`requested_success_rate` =
  `episodes_requested`（固定请求集，skipped/failed 都算非成功）；`completion_rate` =
  实际尝试数（done+failed，**skipped 不进分母**）；`execution_rate` = `episodes_done` /
  `episodes_requested`（skipped 表现为未执行）。`failed_count` 与 `skipped_count` 分开。
  `requested_seeds` 只含实际尝试过的 seed；扫描范围与尝试清单另存
  `scanned_seeds` / `attempted_seeds`。`unique_gids` 是完成集的独立几何数。
- **Wilson 95%**：`wilson_ci.success` 用完成集分母，`wilson_ci.requested` 用请求集分母。
- **false_forward / stuck**：`false_forward` 只算 contact 碰撞且动作 `FORWARD_SLOW`；
  `stuck` 不算 collision，两者在 per-episode 与汇总里分开。
- **协议绑定**：`--protocol <json>` 可比对 `window_from/hold_s/steps/task/
  appearance_version/dynamics/camera_profile`。不一致时 `eval.json` 写 `ood=true`、
  `protocol_mismatches` 并打印警告；`--strict-protocol` 直接失败。默认不失败，
  保证旧 run 可复现。示例：`{"window_from":"observation","hold_s":0.2,"steps":80,
  "task":"l1","appearance_version":"v1","dynamics":"cf_sil",
  "camera_profile":"rgb320x240_fov60_v1"}`；未知键（如 `protocol_id`）忽略。
- **动力学档**：`--dynamics` 接受 `x3_velocity` / `cf_stab` / `cf_blackbox` / `cf_sitl`
  （`collect_parallel.DYNAMICS_CHOICES`；`cf_sitl` 必须配 `--vehicle e1_mini`）。`cf_sil` 已退出
  collect_parallel / eval 的矩阵，只在 `collect.py` 保留手动入口，不得当作已验证档。
  `cf_sitl` 的 `false_forward` 改为「碰撞前 1 s 窗口内下过 FORWARD」，`metric_version` 带后缀
  `sim09.v1.coast1s`（`COAST_WINDOW_S`），与无后缀的 X3 数字不混报。
- **报告规则**：`task_success` / `reach_ever` 与旧 `success` 不可混报；
  没有速度证据时 `stable_termination` 只能报未知。

### 5.3 SPL / CR / task_success / IQM 口径（2026-11-07，只加列，旧字段逐位不变）

本节只定义**新增输出列**和论文报表的统计方法；§5.1/§5.2 的字段定义与数值不变
（用 925 集历史 episode（base / L1 waypoint / campus / 3DGS / cf_sil）逐字段回归，
新旧 `summarize_episode` 与 `aggregate` 的旧键 0 处不一致，见 worklog `2026-11-07-*-eval-matrix.md`）。

| 字段 | 定义 | 分母 / 备注 |
| --- | --- | --- |
| `path_length_m`（p_i） | oracle `pose_gt` 的 xy 折线长，从首个位姿起、**截到第一次进入目标区**的那一点（线性插值到区边界）；从未到达则整条轨迹 | 每集；只有带 xy 的 oracle 才有，旧数据 None |
| `shortest_path_m`（l_i） | **物理可行最短路的下界**（2026-11-09 起）。Gazebo：`teacher.astar_path`（0.1 m 栅格，柱方框按**机体半径** `teacher.BODY_HALF_BY_VEHICLE[vehicle]` 膨胀：e1_mini 0.102 m / x3 0.235 m，机心离柱表面 ≥ 这个值就是碰撞模型下可行）再做视线抽捷径（string pulling，只走原路径经过的自由区）；2026-11-09 前膨胀用教师规划余量 `CLEARANCE_BY_VEHICLE`（E1 base 0.40 m），那是「教师参考路径」，两版 l_i / SPL 不可直接比。3DGS：`sim.3dgs.planner.route_check` 的 A* 路径（软余量 0.4 m / 代价 4.0）在硬栅格∪软余量上抽捷径（未改）。waypoint 任务同样**截到目标区边界**（与 p_i 同口径） | 每集；`eval.json` 顶层 `spl_reference` 写 `clearance_m`（实际膨胀）、`clearance_basis`、`teacher_clearance_m`（教师余量，只供对照），不同载具/任务/版本不得混报 |
| `spl` | `S_i · l_i / max(p_i, l_i)`，S_i = 该集旧 `success`（Anderson et al. 2018） | 每集；p_i 或 l_i 缺一个就是 None |
| 汇总 `spl` / `spl_n` | 有 `spl` 的完成集平均 / 个数 | `requested_spl` 只在全部完成集都有 spl 时给（请求集分母，skipped/failed 计 0） |
| CR | 就是已有的 `collision_rate`（contact 碰撞集比例）；**不新增字段**。stuck / 超时不算碰撞 | `episodes_done` |
| `collisions_per_100m` | `100 × 窗口内有接触碰撞的完成集数 ÷ 各完成集窗口内路程之和（m）`（2026-11-12 起，`collisions_per_100m_def = "v2.first_terminal_event"`）。窗口从起点到**第一个终局事件**（首次越出走廊、首次进入目标区或过终点线、首次接触碰撞；`stuck` 不算）；同一步既碰撞又越界（或过线）计为碰撞，越界或进目标区之后才发生的接触不进分子；单步即终止的集用第 0 步 `pose_before -> pose_gt` 折线长计路程。整集 `collision_rate` 不动 | 完成集窗口总路程；W1-4 旧产物缺 `collisions_per_100m_def` 键（分子计整集碰撞），新旧数不可混报 |
| task_success | 已在 §5.2 定义：`reach_ever` + `final_in_radius` + `stop_action_issued` + `collision_free` + 未超时；不另造 | `episodes_requested`（失败与跳过的集记 0，见 §5.6） |

- **目标区与 success 同口径**：waypoint 任务 = 以 waypoint 为心、`goal_radius` 为半径的圆盘
  （Gazebo L1 1.5 m，3DGS `reach_radius_m` 2.0 m）；验收线任务（base / forest_e1）= 半平面 `x ≥ goal_x`。
  p_i 与 l_i **都量到目标区边界**。截断的原因有两条：① base 没有 goal-stop，机体过线后继续飞到步数
  用完，整条轨迹会把这段算进分母，教师走直线 SPL 也只剩 0.7；② waypoint 任务里机体一进圆盘就算
  到达，若 l_i 量到圆心，p_i 系统性短一个半径（3DGS 2.0 m 对 11 m 路线差 18%），所有成功集都被
  `max(p_i, l_i)` 截成 1，这一列就没有区分度。实测：E1 base 教师采集集 12 集，成功 10 集的 SPL
  0.88–0.99（p_i 6.0–7.0 m，l_i 5.9–6.1 m）；3DGS Callinstrasse route_3 的 astar 教师 3 集 SPL
  0.994–1.000（p_i 9.3–9.9 m，l_i 9.2–9.8 m）。
- **采集端的成功线截断（`collect.py --stop-at-goal`，2026-11-07，默认关）**：与这里同一条线
  （`goal_line_x = 最远柱 x + 1.0`，看 oracle `pose_gt.x`），过线的那一步保留、之后不再采；manifest 记
  `stopped_on_goal=true` + `goal_line_x`，`validity` 按完整集填。它只改**训练数据**的集长（0417 的 12000 帧里
  过线后的 2155 帧 93% 视锥内无柱、38% 标签是 YAW，DISTILL §1.2），不改评测：评测仍跑 `eval_steps`、SPL 仍按上面的截断口径。
  用这类数据训出的学生与旧数据学生对照时，要在报告里写明训练集是否开了该开关（`collection_manifest.json`
  的 `stop_at_goal`），别把集长差异当成模型差异。（2026-11-09 起评测器对 route_choice 任务也在这条线上
  结束本集，见 §5.4。）
- **为什么 l_i 用机体半径而不是教师余量**：教师规划余量 0.40 m 比机身半宽 0.102 m 大得多，按它算的
  l_i 是「教师参考路径」，学生贴柱抄近路时 p_i < l_i 常见，`max(p_i, l_i)` 把一大片成功集截成 SPL=1，
  列失去区分度，而且 SPL=1 只表示「不比教师绕」。按机体半径膨胀，l_i 是碰撞模型下可行路的下界，
  p_i < l_i 只会差在 0.1 m 栅格 + 抽捷径的离散误差（几 cm）上，SPL 恢复「几何最优为 1」的本义。
  栅格 A* 只有 0°/45° 两种走向，不抽捷径时阶梯路径最多长 8%，所以必须抽。
- **哪些任务有 SPL**：`SPL_TASKS = (base, l1, forest_e1)` 和 3DGS 路线；campus 的建筑不在 `pillars`
  里，A* 算出来是错的，留 None。
- **历史 eval.json 不能补算 SPL**：2026-11-09 前的 `eval_closed_loop.py` 不带 `--keep-work` 时每集 oracle 随
  工作目录删除，`per_episode` 里没有轨迹。sim09.v2 起每集 `actions.jsonl` / `oracle.jsonl` 保留在
  `<out>/episodes/seed_<seed>/`（§5.4），之后的评测可以复算。
- **E1 的 SAFE 是滑行不是停**（DATA 报告 2026-11-07 §结论 3）：cf_sitl 杆量回中后首步仍滑行
  中位 0.065 m、最大 0.208 m，悬停漂移 0.012–0.027 m/步。所以 E1 上 `stop_action_issued` 只说明
  「下过 SAFE」，`final_in_radius` / `task_success` 不能当「停稳」报；`stable_termination` 有
  `speed_mps` 时才算，且 < 0.05 m/s 的门槛在滑行机体上通常不满足。写论文时 task_success 一列对 E1
  要标注这一点，或等执行器实现位置保持后复跑。
- **统计方法**（`tools/stats_report.py`，纯 numpy/标准库；单测 `tools/test_stats_report.py`）：
  - 二值指标：比例 + **Wilson 95%**。越高越好的（SR / task_success / reach_ever / stop_success）分母 =
    `episodes_requested`，失败与跳过的集记 0；越低越好的（CR / false_forward / stuck）分母 = `episodes_done`（§5.6）。
    新实验的 summarize 脚本用 `tools/paired_stats.py` 的 `success_counts` 取成功数与请求 / 完成 / 失败三个分母，不再各写各的 n。
  - 连续指标（SPL）：均值 + 百分位 bootstrap 95%（B=2000，seed 固定）。
  - 同一 (方法, 场景) 多个训练种子 run：**IQM**（去掉两端各 25% 后的均值，Agarwal et al. 2021 定义，
    与 `scipy.stats.trim_mean(x, 0.25)` 一致）作点估计，区间 = **几何整簇有放回重采样**的百分位 95%（同一几何上各 run 的集一起抽，
    每个 run 按各自的分母算比例）。run 级分层 bootstrap 把同一几何上的 3 集当独立样本，区间比整簇区间窄（calswa 的 60 集：
    旧 [0.60, 0.75]，整簇 [0.50, 0.78]）。跨场景的全场景 IQM 区间仍是 run 级分层 bootstrap。多 run 的方法也做配对检验，见 §5.6。
  - 两方法同一 seed 集：**McNemar**（不一致对 b+c < 25 用精确二项，否则带连续性校正的卡方）；
    ≥3 个单 run 方法另给 **Cochran's Q**（全体共同 seed）作总体检验；两两 McNemar 照做、不以 Q 显著为前提，
    **Holm** 族 = 同一场景、同一指标的全部两两行（`stats_report.py` 就是这么做的，表注会写）。配对按共同 seed 对齐
    （成功类指标取两侧请求 seed 交集，失败与占请求名额的跳过集记 0；越低越好指标取两侧都完成的 seed，见 §5.6）。
    默认表列 SR/CR/false_forward/SPL；`task_success` 在没有 waypoint 的 base 任务恒为 0，要显式 `--metrics` 加。
  - 样本量：两比例功效估算（正态近似，另给 Fleiss 连续性校正）。0.80 vs 0.90、α=0.05、功效 0.8
    每臂约 199 集（校正后 219）；配对设计可更少。n < 20 的格只算 pilot。
- **seed ≠ 几何**：请求集按 seed 给，完成集按 `geometry_gid` 去重。base 任务 0..2999 内双盐
  （v3 & v5）都是 test 的 53 个 seed 只有 15 个独立几何，固定 20 seed 的请求集实际只跑出 11 个
  几何、9 个 skipped，`requested_success_rate` 的分母因此偏大。论文表的 n 以独立几何数为准，
  请求集要直接用独立几何 seed 清单，唯一来源是 `tools/eval_seeds.py`（`EVAL_SEEDS_BASE_20` /
  `FOREST_SPARSE_SEEDS_20`，`tools/pick_test_seeds.py` 挑的各 20 个：几何两两不同、按各自 salt 都是 test、与 E1/X3
  训练集零重叠；`e1_pipeline.CFG["eval_seeds"]`、`compare_datasets`、`compare_teachers` 都从它读，`tools/test_eval_seeds.py` 锁定）。

### 5.4 sim09.v2：成功线即停（`metric_version="sim09.v2"` / `sim09.v2.coast1s`，2026-11-09）

**sim09.v2：成功线即停，SR 不随 `--steps` 变；v1 的数字过线后继续飞，碰撞可能发生在过线之后。**
`eval_closed_loop.py` 对 route_choice 任务（base / forest_e1，场景没有 waypoint/stop_radius）给
`collect.py` 加 `--stop-at-goal`：真值 `pose_gt.x` 首次 ≥ `goal_x`（最远柱 x + `collect.GOAL_MARGIN_M` = 1.0 m，
`--goal-margin` 只接受这个值）的那一步保留、本集结束，之后不再有碰撞、也不会超时。v1 下机体过线后继续飞到
`--steps` 用完，过线后的碰撞一样记成 collision、把 success 判成 False，所以 v1 的 SR/CR 随 `--steps` 变、系统性
偏低（偏多）；v2 与 v1 的 SR/CR **不可直接比**，`tools/stats_report.py` / `tools/e1_pipeline.py summary` 遇到
混合版本会在表头标出。goal_reach 类任务（l1 / campus）的 `stop_radius` goal-stop 逻辑不变，只是版本号同步成 v2。
每集新增字段：`stopped_on_goal`（collect manifest：route_choice = 首次过线结束；waypoint 任务 = goal-stop 连续
3 步 SAFE 结束）、`steps_to_goal`（第一次进入目标区用的步数，从未进入 None；v2 下 route_choice 成功集等于 `steps`）。
SPL 的 p_i 本来就截到第一次进入目标区，v2 下与整条轨迹一致，数值口径不变。cf_sitl 仍带 `.coast1s` 后缀。
中断（SIGTERM / Ctrl-C / 异常）时 eval.json 也会写出，带 `partial=true` 与已完成的 `per_episode`，不得当完整结果报；
正常结束时每集的 `actions.jsonl` / `oracle.jsonl` / `episode_manifest.json` 保留在 `<out>/episodes/seed_<seed>/`（帧仍删），
所以 v2 起的 eval.json 可以复算 SPL。

### 5.5 动作派发时刻固定到仿真时间（`--action-latency`，2026-11-10）

**新实验一律带 `--action-latency L`，报数时同时报 `dispatch_lag_s` 的中位数与最大值、迟到步数。**
`--window-from observation` 下动作窗口从观测戳起算，执行器截止 = 观测戳 + hold（1.0 s），所以每步动作实际生效的时长
= hold −（观测戳到派发的仿真秒）。这段延迟 = 帧龄（相机 10 Hz，固定约 0.114 s，出处 `docs/PLAN_2026-11-10_phase12.md`
longcourse 分段打点）+ 策略段的墙钟耗时折成的仿真时间；后一项随场景变（K=3 长课程里 Python A\* 每步重跑，冷缓存时
最慢），不带开关时 K=1 与 K=3 的生效时长因此不同（旧协议实测 K=1 0.871 s，K=3 第一轮 0.77–0.79 s，
`docs/worklog/2026-11-10-2011-long-course.md`），拿它比成功率会把时序差当成场景差。

- **口径**：`collect.py --action-latency L`（`eval_closed_loop.py` 透传同名参数）。策略出结果后等位姿话题的仿真戳 ≥ 观测戳 + L 再派发，
  执行器截止仍是观测戳 + hold，所以每步生效时长固定为 hold − L（再加 5–10 ms 的位姿网格过冲），与场景无关。
  只配 `--window-from observation`，L 必须是 (0, hold) 内的有限数。
- **迟到**：策略结果出来时仿真戳已超过观测戳 + L + 0.02 s（`LATE_DISPATCH_TOL_NS`）记一个迟到步，立即派发。
  `actions.jsonl` 每行有 `dispatch_lag_s`（观测戳到派发的仿真秒；不带开关时也写，只多这一个键），manifest 有
  `action_latency_s` 与 `late_dispatch_steps`。**含迟到步的集不计成功**：`eval_closed_loop.py` 把它记成基础设施失败
  （`failed[]` 里 `error` 以 `late_dispatch:` 开头，eval.json 另有 `late_dispatch_episodes`）：不进 `success_rate` 的完成集，在 `requested_success_rate` 里算非成功；
  `per_episode` 有 `late_dispatch_steps`、`dispatch_lag_s_median`、`dispatch_lag_s_max`。迟到集的轨迹仍留在 `<out>/episodes/`。
- **影子标签**：学生 + β=0 + 带开关时，影子教师的 A\* 标签挪到派发之后算（β>0、教师、规则 oracle 不动；`action_expert` 的值不变），
  A\* 耗时不再进延迟。**预热**：带开关时每集起飞前把教师的整张 A\* 自由栅格填满（`teacher.warm_free_cells(pillars, include_retries=True)`，
  一并覆盖标准余量与小于标准余量的 `ASTAR_RETRY_CLEARANCES` 重试档位），第 0 步与首次降余量重试均不再吃冷缓存；外部基线与学生本来就在构造时空推理一次（`StudentPolicy._warmup`）。不带开关时两者都不动，
  默认路径逐位不变（`sim/tools/test_action_latency.py` 的默认锁）。
- **停顿步与断帧旗统一判据**：相邻两步观测戳间隔超过 `max(0.1, hold - 0.1) + 0.2 s`（`collect.is_stall_gap`，`hold = 1.0 s` 时严格大于 `1.1 s`）记为停顿步（`stall_steps`）。带 `--action-latency` 时含停顿步的集与含迟到步的集同样计入 `failed[]`（`stall_episodes` 单列）；不带开关时只记不判。Gazebo 评测 `per_episode` 的 `frame_gap_flag` 与停顿步共用同一函数 `collect.is_stall_gap(max_frame_gap_s, hold)`；无 `hold` 参数的 3DGS 集无 10 Hz 相机相位差，保持 `max_frame_gap_s > 1.2 × sim_dt` 原判断。
- **协议键**：`action_latency_s` 在 eval 的 `PROTOCOL_KEYS` 和 `tools/preprocess.py` 的 `STRICT_KEYS` 里。
  协议文件（训练数据快照）里写了 `action_latency_s` 而评测不带或 L 不同，判协议不一致；旧协议文件没有这个键，不触发。打包时不同 L 的集、或新旧混合会被拒收。
  带开关与不带开关的 SR / CR **不可直接比**（生效时长 hold − L 与旧协议不同），报表头标 L。
- **`--rtf R`**：`scene_gen.py --rtf R`（`eval_closed_loop.py` 透传）把世界的目标实时因子压到 R，让仿真跑慢、墙钟耗时折成的仿真秒变少。
  只能压低，不能超过本机软渲染实际达到的实时因子（各集 `action_rtf` 均值约 0.8）。需要 `--rtf` 的 L 与 R 由标定定，见 `docs/experiments/2026-11-10-action-latency.md`。

### 5.6 主比较协议与统计表分母（2026-11-11）

**成功类指标的分母是请求集，失败集单列；主比较的规模与预注册在起跑前定。**

- **统计表分母**（`tools/stats_report.py`）。`eval.json` 里 `success_rate` 等旧率字段本身不变，仍是完成集分母：
  - 越高越好的二值指标（SR、task_success、reach_ever、stop_success）：分母 = `episodes_requested`。基础设施失败的集
    （`failed[]`，含 §5.5 的 `late_dispatch:` 集）与跳过的集都记 0，与 `eval.json` 的 `requested_success_rate`、`wilson_ci.requested` 同口径。
  - 越低越好的二值指标（CR、stuck、false_forward）：分母仍是完成集 `episodes_done`。失败集没有碰撞或乱飞的观测，记 0 会压低这几项。
  - SPL 是完成集上的均值，不变。
  - 表里单列「失败集数」（该方法全部 run、全部场景的 `failed_count` 之和）。失败集数不为 0 时，同一行里 SR 与 CR 的 n 不同；
    没有失败集时新旧数字逐位相同（`tools/test_stats_report.py` 锁定）。
- **主比较的规模**：20 个冻结几何（`tools/eval_seeds.py` 的 `EVAL_SEEDS_BASE_20`）× 每方法 3 个训练种子 = 60 集；
  主检验是按几何聚类的符号翻转检验与整簇 bootstrap 区间（下一条），逐集 McNemar 与 Wilson 区间照旧报，只作参考；
  两者结论不一致时照实写。T1（`--task course_base --blocks 3` 的长课程）起再报逐块存活。
- **配对检验与区间按几何聚类**（`tools/paired_stats.py`，只用标准库；`stats_report` 的配对表在多 run 的行里加两列）。
  60 集里同一几何上的 3 集不独立，独立单位是几何；逐集 McNemar 把 60 集当独立样本，逐集 p 低于几何聚类 p
  （calswa 对 best：逐集精确二项 0.036、带连续性校正的卡方 0.038，几何聚类 0.089）。
  - 检验：每个几何内 A 的成功数减 B 的成功数，得 d_g（整数）；差为 0 的几何不改变符号和，先剔除；全部几何差均为 0 时 p = 1.0。
    对非零的 d_g 做双侧符号翻转，统计量 |Σ d_g|，p = #{|S| ≥ |观测|} / 2^k（k = 非零几何数）。
    k ≤ 20 精确（与枚举 2^k 种符号等价；20 个几何的主比较恒为精确），k > 20 用固定种子蒙特卡洛（p = (命中 + 1) / (抽样 + 1)），
    表里写明抽样次数（默认 100000，`--n-draws`）。
  - 区间：A−B 成功率差的整簇 bootstrap 百分位 95%（B=2000，seed 固定）。簇键 = `per_episode[*].geometry_gid`，
    没有该字段或两侧同 seed 的 `geometry_gid` 不一致时整表回退为 seed，表注写明；两侧所有 run 都未完成（失败或跳过）的共同 seed
    单独按 seed 成簇，不把其余已完成集的 `geometry_gid` 聚类退回 seed。
  - 配对方式：同一方法的多个 run 按 `--eval` 的给出顺序对齐，第 i 个 run 对第 i 个 run，单元 = (run, seed)；
    两侧 run 数不同只配前 min 个。同一个 run 内 seed 重复直接报错，不静默覆盖。单 run 的方法配对表不加这两列。
  - 越高越好的指标配对与整簇区间里，未完成（基础设施失败或占请求名额的跳过，`failed` 与 `skipped` 合并去重）的 seed 记 0，
    与主表同分母，n 共同 = 两侧请求 seed 的交集（范围扫描里已满请求名额的多余 `skipped` 不混入）；全部失败或全部跳过的格显示 `0.00 (n=请求数)`。
    越低越好的指标失败与跳过集没有观测，只配两侧都完成的 seed。
  - Holm 校正只作用于逐集 McNemar 的 p；几何聚类 p 不做 Holm。
- **协议一致性**（`tools/stats_report.py`）。同一场景（同一列组）内，`action_latency_s`、`heading_guard`、`rtf`、`student_cond`、
  `dynamics_profile` 五个键必须一致，缺键 = 旧协议，`heading_guard=0` 与 `student_cond=prev-action` 等价于缺键。
  其中 `dynamics_profile` 在顶层没有该键时从 `per_episode[*].dynamics_profile` 取（单文件内出现多个取值也判不一致），
  `per_episode` 为空且顶层有 `cf_sitl_params` 时回退取 `values_sha256[:8]`。不一致报错（rc 2），
  信息列出场景、键、各取值与方法；确实要混表（例如有护栏对无护栏、不同固件库对照）加 `--allow-mixed-protocol`，表头标「混合协议」。
  `cf_sitl_params` 含路径，不直接比较路径，参数内容由 `dynamics_profile` 或 `values_sha256[:8]` 覆盖。`metric_version`、`success_criterion`、`hold_s`、
  `dynamics`、`vehicle`、`window_from`、`task` 仍只警告。
- **预注册与样本内外划分**：起跑前在入口脚本头部写下预测、证伪条件和主比较；主比较的配对不超过 3 个；参数在起跑前定死，不在评测 seed 上调。
  设计时看过冻结 20 个评测 seed（`EVAL_SEEDS_BASE_20`）上轨迹的方法（护栏阈值、规则、条件码阈值），在这 20 个 seed 上的数字算样本内；
  要在另外 40 个 test 分区几何（`tools/eval_seeds.py` 的 `EVAL_SEEDS_BASE_60` 去掉冻结的 20 个）上补评一遍，才算样本外的数字（2026-11-12，交叉审核 D3）。
- **单次 n=20 的比较不下结论**：同权重、同 seed 重评的逐集不一致率，此前写「约 24%」（出处 `docs/reviews/2026-11-10-phase12/longcourse.md`
  的「自重跑一致率约 76%」，该文档没有注明数据来源），**未核实**。W1-14 扫描数据根 `uav_rgb/` 四层目录内（不含 `tmp/`，按 `realpath` 去重）
  共 63 个顶层目录、220 个 `eval.json`（含 `e1-pipeline-20261107-*`、`eval3-gpucheck-20261109`、`exp-swa-calib-20261110`、
  `exp-heading-guard-20261110`、`exp-action-latency-{20261110,r2-20261111}`、`exp-long-course-{,v2-}20261110`、`exp-dagger-gz-20261109`、
  `exp-teacher-cost-20261109`、`exp-rule-oracle-{,v1-}20261110`、`exp-k1-control-20261110`、`exp-baselines-zeroshot-20261110`、
  `exp-fix-a-20261109`、`exp-soc100-parity-20261111`、`exp-2026-11-12-teacher-v7-parity`、`exp-proto-parity-20261112`、`exp-viz3-20261110`，
  以及 `l1-*`、`cfstab-*`、`campus-*`、`3dgs-*`、`xeval-*`、`teacher-eval-*`、`distill-*`、`t5b-*`、`t6-*`、`t7-*`、`demo-*`、`gpu-*` 等历史目录），
  按 `ckpt_sha256` 相同且 28 个配置键全同、共同完成 seed ≥ 10 筛选，得 9 对，逐集不一致率 0.20 到 0.47：
  `e1-pipeline-20261107-0417-eval3` 对 `eval3-gpucheck-20261109`（2 对，`new` 与 `trunc` 各 4/17 = 0.235，两次的差别是 GPU，
  RTX 4080 SUPER 对 RTX 2080 Ti，见 `docs/experiments/2026-11-07-e1-demo-plan.md`）；这两个目录的 `trunc` 臂各对
  `exp-teacher-cost-20261109/cl2/eval-trunc-legacy`（2 对，9/20 = 0.450 与 5/17 = 0.294）；`e1-pipeline-20261107-0417-eval4` 对
  `exp-dagger-gz-20261109/full-1109/eval20-trunc`（1 对，9/19 = 0.474）；`exp-long-course-20261110` 对 `exp-long-course-v2-20261110`
  （4 对，0.200 到 0.350）。另有 `exp-heading-guard-20261110/guard2-calswa-s1` 对 `exp-k1-control-20261110/guard2-calswa-s1`
  （4/20 = 0.200）因 `template_sha256` 不同未计入。9 对中只有换 GPU 那 2 对是有记录的受控重评，其余 7 对隔了日期或代码改动，
  没有一对能证明两次运行条件完全相同；「约 24%」与换 GPU 那 2 对的 4/17 数值吻合，是否同源未核实。要给出可引用的重评不一致率，
  需要补跑同权重、同 seed、同机器的两次评测。一次 20 集的差别可能落在这个波动里，所以结论看 60 集的配对结果（几何聚类检验），单次 20 集只作描述。

## 6. 下一步（按优先级）

1. 补 `d_min`（A\* 最短自由路径）→ 才能算 PO / MP / AGV（SR 已实现）。
2. `SimulatorEval` 已在 `eval_closed_loop.py` 落地；`failed`/`skipped` 已分开计数
   （2026-11-04），新指标合同见 §5.2。
3. 处理时间 p50/p95 —— 板端（LicheePi）未定，先报本机 GPU 的数，注明不是板端。
4. 修 §5 的 1 与 2，否则闭环指标本身不可信。

## 7. 引用

- AvoidBench, ICRA 2023. arXiv:2301.07430 <https://arxiv.org/abs/2301.07430>
- AvoidBench 代码 <https://github.com/tudelft/AvoidBench>
- scikit-learn `f1_score` / `balanced_accuracy_score` / `confusion_matrix`
  <https://scikit-learn.org/stable/modules/model_evaluation.html>
