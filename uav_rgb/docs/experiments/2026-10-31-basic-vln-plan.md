# 基础无人机视觉语言导航（VLN）路线图

- 时间：2026-10-31 15:17
- 范围：`uav_rgb/` 现有仿真、数据、训练、闭环链路
- 结论：**当前还不是 VLN，是「语言条件的目标选择 + 四动作避障」**；
  L1 v1 的 teacher 数据本身没走到 goal（§2.5，0/400），已用 **goal-stop v2**
  修复并通过 50 集 pilot（§2.5）。基础 VLN（颜色 + 左右关系 + 停止，
  单条指令，闭环，未见模板）：
  - 从今天算：**4–6 周（2026-12 上中旬）**；
  - 若 teacher 修复/重采一周内完成且导航门槛通过：**3–4 周（11 月底–12 月初）**；
  - 若导航门槛不过：**6–8 周（2027-01）**。
  最快 demo（颜色 + 停止）在数据修复后约 **1 周**；两步指令 VLN 约 **2–3 个月**。
  前提见 §5，失败则按 §6 回退。

## 1. 本文的「基础 VLN」定义

不是 AerialVLN 榜单级，也不是多步长指令。验收口径：

1. 指令同时包含**目标颜色 + 左/右空间关系 + 停止**；
2. 场景里至少有两根**同色**柱，关系词是消歧必需，不能只看颜色；
3. 策略在未知 seed 的柱阵中闭环飞行，成功 = 无碰撞 + 停在目标柱
   ≤1.0 m + 末段为 SAFE（悬停）；
4. 训练/测试指令模板分离，至少 25% 模板未在训练中出现；
5. 报告 L / I / S 三组：正常指令、空文本、打乱指令；
   语言版 SR 比 image-only 高 ≥10 pp，比打乱高 ≥10 pp。

满足以上 5 条，才称为「基础 VLN」。只做颜色目标选择、或只做离线准确率，
都只是 VLN 的前置能力。

## 2. 当前进度（git 与代码事实）

### 2.1 git 速度

- 仓库共 47 个提交，2026-10-29 18:35 建基线，到 10-31 08:03。
- 10-29：+29,984 行（导入基线/大量文档与工具）；
  10-30：+7,103 行；10-31：+1,367 行。
- 关键里程碑墙钟：
  - 10-29 18:35 基线 → 10-30 05:30 闭环 smoke（约 11 h）；
  - 10-31 01:00–08:03 语言词表/landmark/teacher/collect/audit/pilot/400 集
    （约 7 h 代码与数据链路）；
  - 10-31 11:15–11:50 文本分支代码（约 35 min）；
  - L1 image-only 训练 10 轮 2300 s（56.3 样本/s）；34 集闭环 3161 s；
  - 400 集采集约 3 h（1.5–1.67 步/s）。
- 结论：**代码不是主要瓶颈，任务设计、数据采集和闭环评测的墙钟时间是**。

### 2.2 已具备

| 能力 | 现状 |
| --- | --- |
| 仿真/采集 | Gazebo 柱阵、随机场景、teacher A*、并行采集、18 项 audit（v2 含 `teacher_stop`/`safe_full_length`） |
| 数据 | v1 400 集作废；v2 goal-stop 500 集采集中（base-seed 700000） |
| 模型 | 100k 参数 CNN + 16 维文本向量（CPU embedding + 忽略 pad 的 mean-pool + concat） |
| 训练 | 自研 C++/CUDA，Adam、加权 CE、best checkpoint、v4 checkpoint |
| 推理 | 常驻 `uavrgb_infer`，支持 `--text`，hello 报 `text_dim`；closed-loop student |
| 评测 | SR/碰撞/stuck/false_forward/p50/p95；L/I/S 文本模式已接通，goal-stop 成功判据 |
| 语言任务 | L1 v2 goal-stop：颜色目标 → 飞到目标柱并停；v2 数据尚无模型结果 |

### 2.3 还不是 VLN 的原因（代码级）

| 缺口 | 证据 | 影响 |
| --- | --- | --- |
| ~~没有 STOP~~ **已实现** | `teacher.py` v5 `stop_at_waypoint`；`stop_radius=0.5`；eval 用“最终距 waypoint ≤0.7 + 末动作 SAFE” | 剩余缺口不是 STOP，而是关系消歧和模板留出 |
| 关系词是装饰性的 | `landmarks.py` 选两根**不同色**柱；颜色已唯一确定目标 | “left/right”不参与消歧，模型可忽略 |
| 文本编码无词序 | `model.cpp` 只做 mean-pool | 单关系可行；两步/顺序指令不行 |
| 指令模板单一 | `instruction.py` 只有颜色模板 | 无法证明泛化到未见说法 |
| 模板留出未做 | `scene_gen` 无 `template_id`，test 未按模板隔离 | 无法量“未见模板” |
| 数据规模/场景族 | v2 500 集，1 个场景族，`family_holdout=false` | 结论只对当前柱阵成立 |

### 2.4 必须先过的「导航门槛」

> 下面是 **v1 数据**的拆解，只作历史证据。v2 改成 goal-stop 后，目标变成
> 「飞到目标柱并停」，门槛必须在 v2 数据上重测，不能沿用 0/34。

L1 image-only 闭环 SR 0/34 的逐条拆解（`2026-10-31-l1-full-400-image-only.md §3`）：

| 条件 | 通过 |
| --- | --- |
| waypoint_min_dist ≤1.0 m | 19/34 |
| max_x ≥ goal_x=6.0 | **1/34** |
| 无 collision | 17/34 |
| 三条同时 | 0/34 |

一半的集在 21–43 步撞柱；跑满的 17 集 max_x 中位数只有 2.5 m。**瓶颈是
“走不出 6 m”的导航能力，不是语言**。所以基础 VLN 之前必须加一个门槛：

> **M0.5**：文本模型/学生至少要在 L1 test 上 SR > 0（最好 ≥10%），
> 否则 L/I/S 的差异测不出来，语言增益没有意义。

如果 M0.5 不过，按顺序处理（每项 1–3 天）：
1. 先查 teacher/初始化/成功判据是不是又错了（用 audit + 逐集表，不要先加数据）；
2. 从 T6 base 模型初始化 CNN（图像权重迁移，文本分支新加），再微调；
3. 课程学习：先短路线/宽车道，再全量；
4. 仍不过 → 降低任务难度（goal 距离/柱密度），不要先加语言模型；
5. 只有前三步都排除后，才考虑再加数据（500 集是 v2 的最后一轮）。

### 2.5 Critical：L1 teacher 从未到达 goal（2026-10-31 实测）

对 `l1-full-400-0803` 全部 400 集逐集读 `oracle.jsonl`：

| 指标 | 结果 |
| --- | --- |
| `max_x >= max(pillar.x)+1.0`（eval goal） | **0 / 400** |
| `max_x >= max(pillar.x)+2.0`（teacher goal） | **0 / 400** |
| max_x p10 / p50 / p90 / max | 1.44 / **2.58** / 3.50 / **3.76 m** |
| eval goal_x 中位数 | 约 6.0 m |
| planner 分布 | astar 320 / plan_action 80 |

样例 `test/310003`：step 10 左右到达 waypoint（2.0, 0.87），随后 30 步在
x≈2.2–2.5 之间左右打转；末段 `dyaw≈±30°`、位移 <0.02 m。A* 的
`start→waypoint` 与 `waypoint→goal` 都存在，但纯追踪在 waypoint 附近
反复转向，走不出目标柱的侧袋。

现有 `sim/tools/traverse_rate.py` 其实能查出这个问题：

```
episodes=320 success=0 rate=0.0% goal_margin=1.0（safe 出生 80 集不计入）
train/val/test 全部 success=0
```

但 L1 的 audit/验收流程没有把它当硬门禁（audit 只查 waypoint 距离），
所以 15/15 PASS 与 0/320 traverse 同时成立。

**结论：当前 L1 数据不能用来训练 L2，也不能用来证明 image-only 的语言失败。**
0/34 的主因是 teacher 轨迹没有走完路线，不是学生“不知道颜色”。
这也解释了为什么 400 集训练出的 image-only 闭环 SR=0。

已做处置：
- 停掉正在跑的 L2 训练（`run-l2-text16-20261031`，epoch 2 后终止）；
- 本轮重打包的 `ds-l2-20261031` 保留作管线验证，但不能用于结论。

**2026-10-31 17:30 修复并验证**：
- `teacher.v5` 新增 `stop_at_waypoint`：waypoint 作为最终目标，距其
  ≤ `stop_radius=0.5` 输出 SAFE；`scene_gen` 写 `task_version=l1.v2`；
- `collect.py` 连续 3 步 SAFE 提前结束；safe 出生集不提前结束（跑满 40 步）；
- audit 新增 `teacher_stop`（最终距离 + 末动作 SAFE）；eval 新增 goal-stop 成功判据；
- 8 集 pilot、50 集 pilot（含 20% safe）全 16 项 audit PASS，
  `teacher_stop` 0 失败；50 集平均 14 步/集，SAFE 21.5%；
- 500 集全量采集中（5 实例，base-seed 700000）。

必须补的修复（保留为审计依据）：
1. audit 增加 `teacher_goal_reach`：`planner=astar` 的集必须
   `max_x >= eval goal`，否则 FAIL（当前 audit 只查 waypoint 距离）；
2. 把 `traverse_rate.py --min-rate 0.9` 作为采集后的硬门禁（它已存在，
   只是流程没用）；L1 当前是 0/320；
3. waypoint 改为**穿过后**的目标点（或 before+after 双点），避免在目标柱
   侧袋里原地打转；A* 的 `wp→goal` 存在性要单独断言；
4. 50 集 pilot 过 audit + goal-reach + traverse，再全量重采 400–600 集；
5. 重采前不恢复 L2 训练；文本分支代码本身可以先保留。

## 3. 路线图

日期按「1 人 + Claude，每天 4–6 h 有效工作，5 天/周，复用现有管线」估算；
若每天只有 2–3 h，日历时间约翻倍。

| 里程碑 | 内容 | 工作量 | 预计完成 | 验收 |
| --- | --- | --- | --- | --- |
| **M0.1** | **done（pilot）**：goal-stop teacher + `teacher_stop` audit；8/50 集 pilot | 已完成 2026-10-31 | 2026-10-31 | 50 集 16/16 PASS；teacher_stop 0 失败 |
| **M0** | **doing**：重采 500 集 + L2 颜色基座：训练 10 轮 + 离线/闭环 L/I/S | 1–2 天（采集约 2 h） | 2026-11-01 | 离线 `macro_f1_L − max(macro_f1_I, macro_f1_S) ≥ 0.10`；闭环 `SR_L ≥ SR_I+10 pp` 且 `SR_S ≤ SR_L−10 pp`（36 个 test 任务，先看离线） |
| **M0.5** | 导航门槛：文本模型闭环 SR > 0；否则加数据/迁移/课程 | 1–2 周（仅当不过） | 2026-11-20 | L1 test SR > 0，最好 ≥10% |
| **M1（可选）** | 颜色 + STOP：teacher 到目标后悬停，成功=停在目标 | 4–5 天 | 2026-11-13 | 停止误差 ≤0.5 m，SR 明显高于 image-only |
| **M2（推荐）** | 基础 VLN：同色双柱 + 左右关系 + STOP + 模板留出 | 8–12 天 | **2026-11-28 – 12-05** | §1 的 5 条全部满足 |
| **M3** | 两步指令：两个 waypoint、顺序语言、序列编码 | 4–6 周 | 2027-01 中下旬 | 全路线成功率、逐步成功率、S 组下降 |
| **M4** | 接近公开基准：更多场景族/长指令/paraphrase | 2–3 个月 | 2027 Q2+ | 与公开数字同口径（若届时仍有可比对象） |

**建议**：跳过 M1 单独立项，直接把「STOP + 同色双柱关系」合并进 M2；先做
50 集 pilot 过 audit，再全量 600–800 集。这样只采一轮数据，省 3–6 h 和一次训练。

## 4. M2 详细任务（推荐主路径）

### 4.1 任务与场景

- `task_version = l1.v2`，冻结指令模板与成功判据。
- `landmarks.assign_landmarks`：从可见前排选 **3 根**柱：
  - 两根同色（目标色），一左一右；
  - 一根干扰色；
  - 指令 `pass the {color} pillar on your {left|right} and stop`；
  - 关系以出生朝向 +x 为基准（y>0 为左，需在代码/测试中固定）。
- 指令长度：该模板 9 token，超过当前 `kTextLen=8`。M2 开始前把
  `src/include/uavrgb/text.hpp` 的 `kTextLen` 与 `tools/preprocess.py` 的
  `TEXT_LEN` 一起升到 10，并在 `tools/check.sh` 增加 Python/C++ 词表与长度
  的跨语言门禁（当前只有输入尺寸门禁）。没有文本 checkpoint，升级成本低。
- `scene_gen` 写 `relation`、`target_pillar`、`same_color_pair`、`stop_radius`。
- audit 新增：同色柱恰好 2 根、关系与几何一致、每 split 关系/颜色平衡、
  waypoint 属于被指令选中的那根。

### 4.2 教师与 STOP

- `teacher.plan_action_global` 增加 `stop_at_waypoint=True`（**已实现**，v5）：
  - 路径只到目标 waypoint；距 waypoint ≤ stop_radius（**已冻结为 0.5 m**，
    判成功容差 +0.2 m）后输出 `SAFE_NEUTRAL` 并保持；
  - 不再飞向最右柱终点。
- `collect.py` 记录 `stop_reached`、`final_dist`、`stop_steps`；
  连续 K 步 SAFE 可提前结束（K=3–5）。
- audit 复算：plan_action 集的末段必须是 SAFE，最终距离 ≤ stop_radius；
  否则 FAIL。
- **不新增第 5 个动作**：沿用 `SAFE_NEUTRAL` 作为悬停/停止，
  避免破坏 ESP32 四动作合同；若未来要显式 STOP，需硬件协议确认。

### 4.3 模型

- 先复用现有 mean-pool 文本分支：单关系只需要区分 left/right 词向量，
  不需要词序；M0 会验证文本分支是否真的被使用。
- 若 M0 的 L/I/S 差异 <3 pp：把文本特征改为**位置感知**（8×16 拼接 →
  Linear 32 → ReLU），仍不走 GRU/Transformer；约 2–3 天。
- 只有在 M3 两步指令时才实现序列模型（GRU/Transformer）；
  D-STACK 仍为自研 C++/CUDA，不引 PyTorch。

### 4.4 数据与训练

- pilot 50 集 → audit → 全量 600–800 集（约 5–6 h，4 实例）。
- 训练 10 轮，batch 64，lr 1e-3，seed 1234，`--text-dim 16`；
  约 40–60 min。
- 离线 L/I/S：同一 checkpoint，分别用正常/空/打乱 sidecar。
- 闭环 L/I/S：同一批 test seed，`--text-mode keep/blank/shuffle`；
  36 集 × 3 组约 2.5–3 h（可先 5 seed 冒烟）。
- 报告：SR、collision、false_forward、stop_error、逐集表；
  必须写 `eval.json` 的 task/window_from/text_mode/geometry_version。

### 4.5 代码改动清单（估时）

| 模块 | 改动 | 估时 |
| --- | --- | --- |
| `instruction.py` | 关系 + stop 模板，模板表 | 0.5 天 |
| `landmarks.py` | 同色双柱 + 干扰色 + 关系判定 | 1 天 |
| `scene_gen.py` | 写 relation/target/same-color pair/stop_radius | 0.5 天 |
| `teacher.py` | `stop_at_waypoint`，到点后 SAFE；不可达不再静默回退 | 0.5–1 天 |
| `collect.py` | 记录 stop_reached/final_dist/stop_steps，可提前结束 | 0.5 天 |
| `audit_dataset.py` | 同色双柱、关系一致性、stop 末段、模板 split 检查 | 1–1.5 天 |
| `eval_closed_loop.py` | stop 成功判据、stop_error；L/I/S 已接好 | 0.5 天 |
| `text.hpp`/`preprocess.py`/`check.sh` | `kTextLen` 8→10，词表/长度跨语言门禁 | 0.5 天 |
| 测试 | 上述每项的负例/回归 | 1 天 |
| 数据 | 50 集 pilot + 600–800 集全量 | 0.5–1 天（墙钟 5–6 h） |
| 训练/评测 | 10 轮 + 离线 + 36×3 闭环 | 0.5–1 天（墙钟 3–4 h） |

合计：**代码 5–7 天 + 数据/训练/评测 1–2 天 + buffer 2–3 天 ≈ 2–3 周**。

### 4.6 模板留出

- 至少 3 个模板/同义改写，训练只出现 70–75%，test 留 25–30%。
- `instruction.py` 增加模板表；`scene_gen` 写 `template_id`；
  audit 检查 test 模板不在 train。
- 文本增强不需要重采图像/动作，只改 sidecar；成本低。

## 5. 什么时候能实现（结论）

- **最快 demo（颜色 + 停止）**：teacher/数据修复后约 **1 周**，2026-11 中旬。
- **基础 VLN（颜色 + 左右 + 停止 + 模板留出）**：
  - 从今天算：**4–6 周**，**2026-12 上中旬**；
  - teacher 修复一周内完成 + 导航门槛通过：**3–4 周**，2026-11 底–12 初；
  - 导航门槛不过：**6–8 周**，2027-01。
- **两步指令 VLN**：约 **2–3 个月**，2027-01 中下旬。
- **接近公开基准**：2027 Q2 或更晚；当前只有 1 个场景族，不能声称泛化。

前提：
1. M0 证明文本分支真的被使用（L > I/S）；否则先加 3–5 天编码器改造；
2. **M0.5 导航门槛过**（文本模型 L1 test SR > 0）；不过则先加 1–2 周数据/迁移，
   基础 VLN 顺延到 **2026-12 上旬**；
3. 关系/停止的 pilot 过 audit，没有不可达/歧义场景；
4. Gazebo/GPU 单车道不被其它会话长期占用；
5. 不在中途改成功判据；M2 冻结后只改 bug。

## 6. 回退条件与范围

**回退触发**（按顺序）：
1. M0 训练 10 轮后，离线 `macro_f1_L - max(macro_f1_I, macro_f1_S) < 0.03`
   且闭环 L/I 差异 <20 pp；
2. 改成位置感知文本编码后重训，仍无差异；
3. M0.5 导航门槛：加数据/从 T6 初始化/课程学习后，文本模型 L1 SR 仍为 0；
   这时问题在导航任务难度，不在语言，先降难度或重做任务，不要继续加语言模型；
4. M2 pilot 的 audit/闭环显示关系任务不可学（例如同色柱不可见、关系歧义）。

**回退范围**：
- 只回退 L2 文本分支（`text.hpp`、`model.cpp` 文本路径、checkpoint v4、
  trainer/infer 的 text 参数、dataset sidecar）；保留 L1 语言任务与数据。
- 保留本轮的**安全修复**（eval 溯源、sidecar 绑定、audit 扫描范围、
  text_dim 上限），它们与 VLN 无关，不能一起回退。
- 不用 `git reset --hard`；按文件 `git checkout`/`git revert`，先提交或
  stash 一个已知可用基线；未经用户授权不 push。

## 7. 本周立即动作

1. **done**：teacher v5 goal-stop + `teacher_stop` audit；8/50 集 pilot
   全 PASS（`teacher_stop` 0 失败）。
2. **done**：500 集采集完成；排除 2 个含 collision 的集 → 498 集 / 10442 步；
   audit **18/18 PASS**；preprocess 完成（`ds/dataset.uavr` + 498 行侧车）。
3. **doing**：M0 训练 10 轮 + 离线/闭环 L/I/S；根据结果决定 M2 或先修编码器。
4. M2 设计冻结：同色双柱 + 左右关系 + STOP + 模板留出；50 集 pilot。
5. 本轮审核修复已落地（侧车严格解析、`text_dim` 消融前置校验、audit 新门禁、
   check.sh 跨语言门禁）；需跑全量 `check.sh` 后按主题提交（用户明确授权后才提交）。

**在新 500 集 audit PASS 前，不恢复 L2 训练，也不使用旧 L1/L2 数字做结论。**

## 8. M0 实验记录（阻塞）

| 项 | 结果 |
| --- | --- |
| 旧数据 | `l1-full-400-0803/ds-l2-20261031`（v1 route-choice）**作废**；teacher 0/400 到 goal |
| M0.1 pilot | 8 集（4 实例）+ 50 集（5 实例，20% safe）：audit 全 PASS；teacher_stop 0 失败；50 集 698 步。**当时 safe 集被提前截成 3 步，SAFE 21.5% 是假象**；修复后 safe 集跑满 40 步，500 集实测 SAFE ≈46% |
| 新全量 | 采集 500 集 → 排除 2 个含 collision 的集 → **498 集 / 10442 步**；audit **18/18 PASS**；数据卡 `2026-10-31-l1-v2-500-data-card.md` |
| 管线冒烟 | 50 集 ds 1 epoch text 训练 593 样本/10 s；eval-only exit 0（只证明链路，不作结论） |
| 训练 | `run-l2-text16` 启动（10 轮，text-dim 16，batch 64，lr 1e-3，seed 1234，class-weights auto） |
| 离线 L / I / S | 待训练 |
| 闭环 L / I / S | 待训练；`--text-mode` 已实现，test 49 个唯一任务 |
| 结论 | **M0.1 通过；M0 数据就绪，但语言门槛未过**：见 §11 |

## 11. M0 结果（2026-10-31 22:20）：文本没进决策，且不是数据问题

| 实验 | 结果 |
| --- | --- |
| 离线 L / I / S（498 集模型） | 0.8595 / 0.8621 / 0.8595 —— 无差异 |
| 同图换 `--text` 探针（147 帧） | 概率差 2e-4，argmax 翻转 **0** |
| 反事实训练集（311 集 / 1783 步换标签，35.3%） | 决定性帧 flip 1.6%，train 也一样（拟合不了） |
| 位置感知拼接（8×16 直接拼） | flip 3.9% |
| 位置感知 + 文本列放大初始化 | flip 2.75% |
| 目标无关状态（safe 出生集）两色标签差异 | **0%**（避障主导，颜色无用） |

**根因是结构性的，不是数据量**：当前 fusion 是

```
logits = W_img · img_feat + W_txt · text_feat
```

同一张图换文本时，logit 差是常数，与“目标柱在画面左侧还是右侧”无关。
模型无法表达“文本说红 → 去红柱（它可能在左也可能在右）”，最优解就是
忽略文本、退化成图像策略。加数据/加轮数/改初始化都不会改变这个表示能力上限。

加剧因素：训练状态全部来自“朝目标飞”的轨迹，图像本身泄漏目标是哪根柱
（无文本模型在决定性帧上 0.776），文本梯度被进一步压掉。

### 11.1 修订后的下一步（M0.6，替换原 M2 的位置）

1. **改融合结构**：文本 → 64 维 γ/β，对 conv 最后一层特征做 FiLM
   （`y' = γ·y + β`），或低秩双线性 `(W_img·img) ⊙ (W_txt·text)`。
   目标：同一张图换文本，动作必须能翻转。
2. **语言门槛（先于任何闭环）**：在 255 个决定性帧上
   `flip ≥ 0.5` 且 `acc_correct ≥ 0.9`；达不到就继续改结构，
   不进 M2、不采新数据。
3. 过了门槛再跑离线 L/I/S + 闭环 L/I/S；闭环 `SR_L − SR_I ≥ 10 pp`。
4. 若 FiLM 过了门槛但闭环 SR 仍低，再考虑状态覆盖（随机游走/DAgger，
   同一状态对两个目标各标一次）；这是**唯一**值得再采的一轮。
5. 不建议：继续 mean-pool、继续加同分布数据、继续堆轮数。

预计：M0.6 代码 2–4 天 + 训练/探针 1 天；基础 VLN 顺延到 **2026-12 中旬**。

## 9. 数据策略反思：为什么总在采集

**结论：不是“没有好数据集”，是“任务定义没冻结 + 采前门禁量错了东西 + 行为数据与文本标注没分开”。** 语言任务确实没有公开数据可替代，但同一任务版本只该采一次。

### 9.1 账本

| 数据 | 规模 | 结果 | 性质 |
| --- | --- | --- | --- |
| T5 视觉 599 集 | 23960 步 | audit 11/11；offline macro_f1 0.8589；SR 0.90 | **好数据，视觉够用** |
| T5b dispatch 596 集 | 23840 步 | SR 0.60（12/20） | 口径实验，不采纳 |
| T7-R（T6 模型、dispatch） | 20 seed | SR 0.55（11/20） | 口径实验，不采纳 |
| L1 120 集 | 4800 步 | offline 0.4954；SR 0 | 小，已过时 |
| L1 400 集 | 16000 步 | offline 0.7686；SR 0/34 | **无效**：teacher 0/400 到 goal |
| L1 v2 pilot 50 集 | 698 步 | 16/16 PASS | 有效，只作门禁 |
| L1 v2 500 集 | 采集中 | — | 当前唯一在采的 |
| AirSim 10k（公开） | 10000 帧 | 无 SAFE、无文本、连续指令 | 只能预训练，不能验收 |

2.5 天里全量采集 3 次（T5、T5b、L1 400），加本轮 L1 v2 是第 4 次。其中 **2 次是因为口径变了**，1 次是因为 teacher 根本走不到终点。

### 9.2 根因（按代价排序）

1. **任务语义改了 3 次**：route-choice → goal-stop；“and stop” → “pass” → 再改回；safe 出生集是否提前停。每次改都让旧数据整包作废。
2. **audit 量错了不变量**：旧 gate 只查 waypoint 距离，不查“teacher 是否到 goal、是否停下”。400 集 15/15 PASS 却是废数据。门禁存在，但测的不是成败。
3. **先全量、后验证**：120/400 集都是采完才发现 teacher 有问题。8 集 pilot + 逐集 `max_x`/`final_waypoint_dist` 10 分钟就能发现。
4. **没有把“行为”和“文本”分开**：指令写进逐集 meta，换措辞要重打包；换语义只能重采。模板/同义改写/blank/shuffle 本应只是 sidecar。
5. **SR=0 时的条件反射是“加数据”**：旧 L1 的 0/34 是数据无效，不是数据量不够；再加 400 集也还是 0。
6. **公开数据不匹配是事实，但不是主因**：AirSim 10k 没有 SAFE/文本/关系，替代不了；它只适合做视觉预训练，且会引入域差。

### 9.3 立即生效的四条规则

1. **冻结 L1 v2 规格**（写进 §4）：场景、出生、指令表、`stop_radius=0.5`、成功判据、split。改任何一条 = 新 `task_version` + 重采；只改措辞 = 只换 sidecar。`kTextLen`/词表是编码约束，改它不改变场景与行为，不算新任务版本（可重训，不重采）。
2. **采前门禁**（8 集 pilot，不过不开全量）：`teacher_stop` ≥ 7/8（半径 +0.2 内且末动作 SAFE）、safe 集跑满请求步数、每类动作 ≥ 5%、指令-几何一一对应。`traverse_rate` 的成功判据是 v1 的 `max_x≥goal`，**v2 上会 0/40 假失败，不能再用**。做成脚本，不靠人看。
3. **数据卡**：每个 ds 一页，写 `task_version`、集/步数、类分布、到点率、split、sha256、**能回答/不能回答的问题**。audit JSON 是机器证据，数据卡是人读的结论。
4. **停止条件**：本轮 v2 500 集是本任务版本的最后一轮；L/I/S 结果出来前不再采新数据。若 L≈I≈S，问题在导航/视觉，加数据无用，先改初始化/课程；只有“颜色能学、关系不会”才为 M2 采一轮（600–800 集，最后一次）。

### 9.4 复用设计

- **同轨迹、只换文本**：同义模板、blank/shuffle、跨语言 → 只重写 `instructions.jsonl`，零采集。
- **换几何/换行为**（goal-stop、同色双柱）→ 必须重采，但 pilot 过 gate 后只采一轮；之后所有措辞变体复用。
- **换目标色**：teacher 路径随目标变，动作不同，不能只改标注；归入新任务版本。

按此执行，本轮 500 集之后，基础 VLN 只剩 M2 一轮采集；采集不再是常态，而是任务版本升级时的一次性成本。

## 10. 2026-10-31 19:58 审核后的执行计划

### 10.1 审核结论（4 路只读 + 已修项）

- **C++ 无 Critical**。修了侧车整数解析的两个真实边界：`stoul` 截断到 uint32
  会把 `2^32+k` 回绕成 `k`，超大数抛 `out_of_range` 而不是 `runtime_error`；
  `[-1]` 会被当成 `1`。另修 3 处假绿测试（pad 断言恒真、text_dim 检查未被命中、
  NaN 注入无越界保护）。
- **采集门禁三个洞已补**：`safe_full_length`（safe 集被 3 连 SAFE 截短会悄悄
  饿死 SAFE 类）、`provenance_uniform`（v1/v2 混包时逐条复算会跳过旧版本）、
  `teacher_stop` 容差统一 +0.2 且无 astar 集时 FAIL 而不是 SKIP。
- **评测一个会让 L/I/S 结论作废的洞已补**：infer hello 现在报 `text_dim`，
  `student.py` 暴露它，`eval_closed_loop.py` 在 `blank/shuffle` 前强制校验；
  `eval.json` 记录 `text_dim` 和 `success_criterion`。此前纯视觉 ckpt 上
  blank/shuffle 与 keep 完全等价，会被读成「语言无增益」。
- **其它已修**：`preprocess` 先删旧侧车再替换数据集（不再可能留下新数据集 +
  旧侧车）、侧车 mkstemp；`check.sh` 加文本长度/词表跨语言门禁、
  `--no-tests=error`、safe 剔除按 episode_id 断言；shuffle 优先用场景里真实
  存在的干扰色。
- **未修（排期）**：`collect_parallel` 只信子进程 `stopped_on_goal` 布尔而不核
  最终距离；student eval 的 3 连 SAFE 提前停没有写终止原因；`src/device/`
  状态层不支持 text_dim；B4（infer 语义对拍）仍未做。这些不阻塞 M0，但 M2 前要补。

### 10.2 接下来 3 天（按顺序，卡点即停）

| 步骤 | 内容 | 判定 |
| --- | --- | --- |
| D1-A | 500 集收尾 → `audit_dataset.py`（18 项） | 全 PASS（`teacher_stop`/`safe_full_length`/`class_share` 必须 PASS）；不过只修数据/代码，不训练 |
| D1-B | `preprocess.py` 打包 + 数据卡（集/步、类占比、到点率、sha256） | SAFE ≈46% 记录在案；若训练后 SAFE 预测 >60% 或 stuck 上升，打包时抽样 safe 集（不重采） |
| D1-C | `bash tools/check.sh` | **已 PASS**（rc=0，21/21 CTest + 全部 Python 测试）；提交前再跑一次 |
| D2-A | L2 文本训练 10 轮 | 用 `--class-weights auto --drop-collision`，与 image-only 同超参 |
| D2-B | 离线 L/I/S | `macro_f1_L − max(macro_f1_I, macro_f1_S) ≥ 0.10` |
| D2-C | 闭环 L/I/S（先 8 seed 冒烟，再 36 任务） | `SR_L ≥ SR_I+10 pp` 且 `SR_S ≤ SR_L−10 pp`；同时记录 `SR_L > 0`（导航门槛） |
| D3 | 判定与决策 | 过 → M2 设计冻结 + 50 集 pilot；不过 → 按 §6：先改编码器/初始化/课程，最后才考虑加数据 |

**M0 的数据分母**：v2 test 任务数在 audit 后由 `eval.json` 的 `episodes_requested`
和 `skipped` 决定，不硬写 36。**L/I/S 结果出来前不再采新数据。**
