# 教师端算法主线：用 RL / 规划思想提升学生（2026-11-07 立）

- 用途：第二条主线。学生是只看单目 RGB 的小 CNN（3 层卷积、19200→4 的全连接，约 10 万参数），模型本身不好做大（要上 SG2002 的 NPU）；能大幅提升的是**教它的人**：标签怎么来、在哪些状态上教、教的是什么形式。
- 原则：教师只在仿真里跑，可以用特权信息（真值位姿、障碍几何、深度），也可以用 PyTorch（`teacher/` 这条线已经在用 torch，有自己的 venv）；学生的训练/推理仍是自研 C++/CUDA，输入只有 RGB（+短指令），仿真真值只能当**标签**不能当**输入**（`ENGINEERING_RULES` 硬约束）。
- 现状基线：GT A* 教师（`sim/tools/teacher.py`）硬标签 → 行为克隆。X3 上学生 SR 0.90，但只换执行方式（cf_stab）就掉到 0.06；校园任务 0/80。说明学生学到的是「X3 在 X3 教师轨迹上的习惯」，不是「看到什么就该怎么飞」。

## 1. 现有教师代码盘点

| 代码 | 做什么 | 问题 |
| --- | --- | --- |
| `sim/tools/teacher.py` | 2D 网格 A*，几何常量按 X3，假设每步能停住；输出硬标签 | 不知道惯性；标签依赖相机看不见的信息；四个动作只给 argmax |
| `sim/tools/teacher_predictive.py` | 带简单动力学预测的教师（realism_rl_v1） | 第四套动力学常数，未与 SITL 对齐 |
| `rl_teacher/`（PPO，特权状态） | 想用 RL 学教师 | 依赖 stable_baselines3（未装）；闭环数据 `teacher=teacher_rl` 打不了包（AUDIT 候选 A4）；N1 观测分布错位（已修） |
| `teacher/`（PyTorch `PrivilegedTeacher` 序列模型 + `export_soft_targets.py`） | 特权输入 + 时序 → 软标签给学生蒸馏 | 蒸馏第二轮没有显著提升（`docs/experiments/2026-11-03`），因为软标签来自同一批状态、同一套 X3 习惯 |
| 本轮新增 `sim/tools/teacher_kino.py`（FAST-A agent 进行中） | 状态含速度的运动原语搜索，按 E1 实测响应建模 | 原型 |

结论：数据链路、软标签格式（`tools/SOFT_TARGETS_FORMAT.md`）、`action_expert` 与 `action_executed` 分开记录的 episode 格式都已经有，缺的是**算法**。下面五条按「收益 / 成本」排序，前三条本周可以做。

## 2. 五条算法（每条：思想 → 为什么对我们有效 → 最小实现 → 怎么验证）

### A1 DAgger：在学生自己的轨迹上让教师重新贴标签（本周，收益最大）

- 思想（Ross et al. 2011，RL 里的 on-policy 数据聚合）：行为克隆只见过教师走的「好状态」；学生一偏离就进入没见过的状态，错误复合。DAgger 让学生飞、教师在学生到的每个状态上给出它会怎么做，把这些 (学生看到的图, 教师标签) 加进训练集，迭代 2–3 轮。
- 为什么对我们有效：换载具后 SR 崩掉就是状态分布漂移；E1 的惯性让学生进入教师数据里没有的「带速度」状态。DAgger 直接补这部分。
- 最小实现（约 150 行，不加新格式）：`eval_closed_loop.py` 已经能让学生闭环飞；加 `--relabel teacher`：每步同时调用教师（`plan_action_global`，或 teacher_kino）得到 `action_expert`，学生动作进 `action_executed`，episode 写成标准格式，manifest 里 `teacher=plan_action`、`executed_by=student`、`dagger_round=k`、`beta`（前几轮按概率 β 混合执行教师动作）。`tools/preprocess.py` 的教师门禁放行这种 provenance（现在 `teacher=student` 被拒，是对的；DAgger 数据的标签仍是教师的）。训练用全部轮次的并集。
- 验证：同一 20 个 test seed（`docs/experiments/2026-11-07-e1-demo-plan.md` §1），BC 学生 vs DAgger 2 轮学生的 SR / 碰撞率；按 EVAL_STANDARD 报。预期：E1 上 SR 提升明显，尤其是 false_forward 下降。
- 风险：教师在学生造成的「坏状态」（贴着柱子、斜着）也要会给出合理动作，GT A* 能做到；kino 教师更好。

### A2 价值软标签：把四个动作的「代价差」教给学生（本周，和 A1 叠加）

- 思想：RL 里的 Q 值 / 优势函数。教师不只说「选 FORWARD」，而是对四个动作各算一个 cost-to-go（执行该动作一个 hold 窗口后，从预测终态到目标的 A* 代价 + 碰撞/净空惩罚），再 softmax(−cost/τ) 得到分布。学生用现成的 `--soft-targets` 蒸馏损失学这个分布。
- 为什么有效：很多状态下 YAW_LEFT 和 FORWARD 几乎等价，硬标签逼学生在噪声里二选一；软标签告诉它「哪些都行、哪个千万不行」。对小模型尤其重要：目标函数平滑，泛化更好。这也是 `teacher/` 蒸馏想做但没做到位的事：软标签要来自**规划代价**而不是另一个过拟合的网络。
- 最小实现：`teacher_kino.py` 里本来就要对四个原语各算终态，顺手输出 4 个 cost；`tools/export_soft_targets.py` 已有格式（`SOFT_TARGETS_FORMAT.md`），写一个 `tools/costs_to_soft_targets.py`（τ 放文件顶部配置区）。
- 验证：同数据、同 seed、同轮数，硬标签 vs 软标签（τ 两档）的离线 macro_f1 与闭环 SR。

### A3 可见性一致的标签：不教相机看不见的东西（本周，便宜）

- 思想：教师用全知几何，学生只有 60° 视场。教师因为视野外的柱子转弯，学生从图里找不到依据，只能记场景 → 过拟合。
- 最小实现：教师规划时记录「决定本步动作的障碍」（A* 路径上最近的阻挡柱），用相机位姿算它是否在视锥内且未被遮挡（有 GT 深度可以直接查）。不可见 → 该样本权重降为 0.2，或改为「只用可见障碍重新规划」的动作。`actions.jsonl` 加一列 `label_visible`，`preprocess.py` 支持 `--weight-by-visibility`（训练器已有 class weights，样本权重要加一个 per-sample 权重列：`src/dataset.cpp` 加可选字段，checkpoint 不受影响）。
- 验证：标签可见率统计（预期 X3 数据里 15–30% 的转向标签不可见）；去掉/降权后学生在新场景上的 SR。

### A4 PPO 特权教师 + 蒸馏（2–3 周，第二阶段）

- 思想（Learning by Cheating / 特权教师-学生，Loquercio 等人的敏捷飞行也是这条路）：用 RL 在**快的纯 Python 运动学仿真**里训练一个看特权状态（机体系下最近 K 根柱子的相对位置、速度、目标向量）的 MLP 策略，奖励 = 前进进度 − 碰撞 − 净空不足 − 动作抖动 − 时间；然后用它当 A1 的重贴标签教师、A2 的软标签来源。RL 教师比 A* 的优势：它学到的策略天然带惯性和时序（输入含速度），会「提前转、少刹车」，比逐步 A* 更像好飞手；而且奖励里能直接写安全余量。
- 现状可复用：`rl_teacher/core.py`（NavigationCore、DistanceField、奖励 `reward_v1.json`）、`gym_env.py`。缺训练器（SB3 未装）。
- 最小实现：不装 SB3，写一个 ≤300 行的 numpy/torch PPO（`rl_teacher/ppo_min.py`，torch 用 `teacher/.venv`），动力学换成 `teacher_kino` 同一套 E1 原语；4 个离散动作；训练 1e6 步在 CPU 上约 30–60 min。
- 验证：先在 Python 仿真里 SR > A* 教师（A* 在 E1 原语下的 SR 是上限参考）；再到 Gazebo SITL 闭环验 20 seed；最后作为教师做 A1/A2，比较学生 SR。

### A5 学生侧配套（和 SRC-ALGO 合并，模型不变大）

- 辅助深度头：训练时让学生额外预测 20×15 的粗深度（标签来自仿真深度相机，`--save-depth` 已有），损失加权 0.3，部署时丢掉这个头。小模型从辅助任务学到几何，4 动作头受益；深度是标签不是输入，不违反真值隔离。
- 双帧输入：把上一帧和当前帧堆成 6 通道，让学生看见自己在动（E1 有惯性，单帧无记忆是 REVIEW D3 指出的根本缺陷）。模型参数只多第一层卷积的 3×16×9=432 个；`student.py` 和 `infer_main` 要带一帧缓存；checkpoint 版本升级。
- 动作滞回：闭环推理时 YAW 切换要连续 2 帧同向才执行，抑制抖动；纯 Python，`sim/tools/student.py`。

## 3. 顺序与时间

| 周 | 做什么 | 过线 |
| --- | --- | --- |
| 本周（demo 后） | A1 DAgger（2 轮）+ A2 价值软标签（用 teacher_kino 的 cost）+ A3 可见性权重 | E1 SITL 20 seed：相对 BC 基线 SR 提升 ≥10 pp，false_forward 下降 |
| 第 2–3 周 | A4 PPO 特权教师（Python 仿真训练 → SITL 验证 → 当教师） | RL 教师 SR ≥ A* 教师；蒸馏后学生 ≥ A1 学生 |
| 并行 | A5 双帧 + 辅助深度（SRC-ALGO 分支）、动作滞回 | 离线 macro_f1 与闭环 SR 的 A/B |
| 之后 | 把 A1–A5 的组合作为「E1 学生 v2」进真机阶段 | 阶段 2 门禁 |

## 4. 要你定的

1. PPO 教师用 torch（走 `teacher/.venv`，仿真专用）还是纯 numpy？建议 torch，省一周。
2. DAgger 轮次数据是否进正式数据集（会改 split 统计口径）：建议单独成档 `dagger-round-k`，训练时拼接，评测只认 test split。
3. 双帧输入会改 checkpoint 格式与板端输入接口（两帧缓存），要不要进阶段 0 的 E1 学生 v1？建议 v1 单帧、v2 双帧，分开报数。

## 5. 代码规划（放哪、怎么管）

- `sim/tools/teacher.py`：保留，GT A* 基线教师（也是 A3 的宿主）。
- `sim/tools/teacher_kino.py`：动力学感知教师（FAST-A 原型 → 正式），同时输出 4 动作 cost（A2）。
- `rl_teacher/`：A4 的 PPO 教师；`core.py` 的动力学换成 teacher_kino 同一套原语，删掉第四套常数；`ppo_min.py` 训练器；`train_ppo.py` 改成配置区风格入口。
- `teacher/`：现有 PyTorch 特权序列教师，暂不删，等 A4 出来后二选一。
- `tools/`：`costs_to_soft_targets.py`（A2）、`preprocess.py` 放行 DAgger provenance + 样本权重列（A3）、`e1_pipeline.py`（配置区风格流水线，含 `--dagger-rounds`）。
- `sim/tools/eval_closed_loop.py`：`--relabel teacher`（A1）。
- `src/`：样本权重（A3）、辅助深度头与双帧输入（A5，在 SRC-ALGO 分支上做，格式升级 v6）。
- 淘汰：`cf_stab/cf_sil/cf_blackbox` 过 SITL 门禁后删；`teacher_predictive.py` 并入 teacher_kino 后删。
- 每条算法一个 `docs/experiments/` 文档记数字，一个 worklog 记做了什么；分支按主题开（`teacher-dagger`、`teacher-soft`、`rl-teacher-ppo`），合并前 `tools/check.sh` 全过，合并后 `tools/git_mirror_alvln.sh` 推 GitHub。
