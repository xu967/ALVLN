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
- **SR 分母 = `episodes_done`**（`collect_rc==0` 且 `steps>0` 的集）。采集失败
  （含学生推理进程崩溃）进 `failed`，不进分子分母。因此绝对 SR 对失败集是乐观的；
  报告必须同时给 `failed` 数。盲前进基线同口径，两个基线对比仍公平。
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

## 6. 下一步（按优先级）

1. 补 `d_min`（A\* 最短自由路径）→ 才能算 PO / MP / AGV（SR 已实现）。
2. `SimulatorEval` 已在 `eval_closed_loop.py` 落地；后续补终止原因落盘和
   `failed`/`skipped` 的完整对账。
3. 处理时间 p50/p95 —— 板端（LicheePi）未定，先报本机 GPU 的数，注明不是板端。
4. 修 §5 的 1 与 2，否则闭环指标本身不可信。

## 7. 引用

- AvoidBench, ICRA 2023. arXiv:2301.07430 <https://arxiv.org/abs/2301.07430>
- AvoidBench 代码 <https://github.com/tudelft/AvoidBench>
- scikit-learn `f1_score` / `balanced_accuracy_score` / `confusion_matrix`
  <https://scikit-learn.org/stable/modules/model_evaluation.html>
