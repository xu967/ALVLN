# TinyAerialVLN 总规划 v2：可展示的 demo + 有算法意义的模型（2026-11-07 立，06:10 算法定稿）

- 状态：v2。算法选型（§3 W2/W3）按 `tmp/phase3-20261107/reports/ALGO_survey_20261107.md`（30 篇全文 + 本仓代码对照）定稿；场景清单吸收 3DGS 复核报告（`3DGS_20261107.md`）的发现。
- 用户 2026-11-07 05:50 拍板：
  1. **相机参数按老实验走**：Gazebo 与 3DGS 都用 60° HFOV、320×240（旧 X3 学生可比）。Callinstrasse `scene.json` 当前 64.9° 要改回 60°（等 3DGS pilot 本轮结束后改，再重跑 pilot）。§8 第 1 条关闭。
  2. 里程碑排期同意（M0 11-07 → M2 11-12 → M3 11-20 → M4 12-05）。
  3. 论文机器人与视觉两边并齐：场景数、真实外观对比、算法消融都按两边都能投的标准做。用户团队将**实地自采 3DGS 数据集**，流程与规范见 `DATASET_3DGS_SELF_CAPTURE.md`（现在只开线，不写代码）。
  4. **视频优先**：能出的视频先出。v0 短片已生成：`$DATA_ROOT/uav_rgb/demo-e1-20261107/reel/demo_reel_v0.mp4`（`tools/make_demo_reel.py`，296 s：3DGS 6 条路线 + Gazebo 森林）；v1 在流水线四组视频和新场景出来后替换。
- 关系：`ROADMAP.md` 管半年方向与门禁，`TEACHER_LINE_PLAN.md` 是教师线的早期草案（其中 A3「可见性加权」按用户 11-07 要求不再作为贡献点），`TASKS.md` 管任务状态。本文回答「接下来 4–6 周做什么、交付什么、怎么算过」。

## 0. 定位（2026-11-07 用户纠正后）

- 做的是**无人机视觉（语言）导航**。模型层只输出四个高层动作：SAFE_NEUTRAL / FORWARD_SLOW / YAW_LEFT / YAW_RIGHT。无人机自己飞：定高、定速、四个运动原语的执行都在飞控里，由硬件团队后续实现；模型不碰油门、杆量、PWM。
- 仿真里的 `sim/cf_sitl/exec/` 执行器只是飞控那部分功能的**参考实现**，目的是让仿真的闭环响应（每步位移、转角、惯性、悬停漂移）接近真机。固件的油门余量、杆量映射不是研究对象，相关数字只出现在 SITL 文档里，不进论文主线。
- 因此研究对象有三个：**教师**（监督从哪来）、**学生**（怎么从单目 RGB 学）、**场景**（在哪些世界里学和测）。

## 1. 两个交付物

### D1 Demo（目标 11-12 前）

一段 3–4 分钟视频 + 一页数字，给老师看：

1. E1 MINI 固件在环的无人机在 **≥3 个真实 3DGS 场景**（Callinstrasse + 至少 2 个新场景）里自主飞行到目标，每个场景 ≥4 条带转弯/避障的路线；三视角（机载 / 跟随 / 顶视）。
2. Gazebo 柱阵 + 随机圆柱森林（合成几何对照组），机体外观是 E1（四臂、电机、桨、保护罩），不是圆饼。
3. 同一批 test 场景上四组对照：教师（上界）/ 盲前进（下界）/ 旧 X3 学生 / 新 E1 学生，报 SR、碰撞率、false_forward、步数，n 与 Wilson 95% 区间。
4. SITL 自检曲线（起飞、悬停、打杆阶跃、电池）各一张图，说明「仿真的无人机是按真机参数建的」，并如实标注 `calibrated=false`。

门禁：`tools/check.sh` 全过；数字来自守护进程日志可追溯；视频里的每一帧能对回 `actions.jsonl`。

### D2 论文（目标 6–8 周出初稿）

工作名：**TinyAerialVLN：面向 100 g 单目微型无人机的特权教师–小模型学生导航与 sim-to-real**。要回答的科学问题：

- Q1 对只能装十几万参数小模型的微型机，**监督信号的形式**（硬标签 / 动作价值 / on-policy 聚合）比模型结构更能决定闭环表现吗？
- Q2 教师的**动力学感知程度**（几何 A* / 运动原语搜索 / RL）会不会传导到学生的闭环安全性？
- Q3 从合成几何（Gazebo）学到的避障能否迁移到真实外观（多场景 3DGS），混合训练是否两边都不掉？
- Q4 固件在环仿真训练的策略，sim-to-real 的差距有多大（真机阶段，依赖硬件团队）。

主表：行 = 方法与消融，列 = 场景（Gazebo 柱阵 / 森林 / 3DGS 场景 1..k）× 指标（SR / 碰撞率 / false_forward / SPL / 平均步数）。每格 ≥20 个独立几何或路线，报区间；成对比较用 McNemar。

### D3（D2 之后）语言导航

结构化指令（地标 + 关系 + 动作）在 3DGS 场景里落地；三组对照（正确指令 / 打乱指令 / 盲前进）；语言增益显著才叫 VLN（`ROADMAP.md` 阶段 3 门禁不变）。本文不展开。

## 2. 现状（2026-11-07 05:30 事实）

| 层 | 有 | 缺 |
| --- | --- | --- |
| 执行与动力学 | `sim/cf_sitl/`（固件在环 1 kHz，Gazebo 插件 + 3DGS 子进程共用） | 倒飞（route 5 seed 500）与悬停漂移 1–2 cm/s 的根因（SITL agent 在查） |
| 场景 | Gazebo 柱阵、forest_e1 三档；3DGS Callinstrasse（6 条路线）、Mip-NeRF360 bicycle/kitchen/room/bonsai/counter/playroom（有 PLY，未配路线/碰撞代理/相机对齐） | 新场景的尺度/重力对齐、路线、质量门禁；统一相机参数 |
| 教师 | 真值 A* + 纯追踪（`teacher.py`）；运动原语 kinodynamic A* 原型（`teacher_kino.py`）；3DGS 占据栅格 A*（`sim/3dgs/planner.py`） | 价值标签、on-policy 聚合、RL 教师训练器 |
| 学生 | 单帧 CNN + FiLM 文本，自研 C++/CUDA，训练 2669 样本/秒，推理 0.43 ms | 双帧输入、价值蒸馏头、辅助几何头；B4 输入一致性对拍（SRC agent 在做） |
| 数据与口径 | UAVR 格式、11+ 项 audit、几何组 split、20 个冻结 test seed、`EVAL_STANDARD` | preprocess 跨族打包（DATA agent 在做）；3DGS 数据 split 口径 |
| 工程 | 守护进程规则、worklog、`check.sh`、GitHub 镜像脚本 | — |

正在跑：E1 流水线（Gazebo 柱阵 300 集 → 训练 → 四组对照，约 07:30 出数）；3DGS pilot 重跑；6 个审计/调研 agent。

## 3. 工作流

### W1 场景（demo 与论文共用；最先动）

0. 已知短板（3DGS 复核报告）：碰撞代理只覆盖渲染可见表面的 24%，机体能穿过画面里的树篱而不判碰撞；深度是 alpha 加权期望。所以场景选择优先带**真实几何**（LiDAR 网格 / 激光扫描）的数据集，碰撞代理 v3 用网格或 Splat-Nav 式椭球测试做真值对照。
1. 选场景（按调研清单排序）：① Callinstrasse（已有，MIT，米制；地面高度 P0 已修）；② **Wanderland**（Apache-2.0，自带 3DGS + LiDAR 网格 + 导航 episode，人行道/校园，首选新增）；③ Mip-NeRF 360 garden/bicycle/stump（CC-BY 4.0 待在原页复核，Inria 30k 预训练 PLY，本地 7k mini 版先评估）；④ DL3DV-GS-960P 室外子集（6,939 场景带预训练 3DGS，需申请，**用户申请账号**）；⑤ ScanNet++ v2（米制重力对齐室内，需申请 + gsplat 自训 20–40 min/场景）；⑥ Zip-NeRF Alameda/Berlin/London/NYC（CC-BY 4.0，多房间室内，自训 30–60 min/场景）。主表用 ①②④ 作训练集，③ 与 ⑤/⑥ 各一个作保留集；室内室外各有保留场景；自采场景单列。
   M1（11-09）目标 ≥2 个新场景可飞：Wanderland 1 个 + Mip-NeRF 360 garden 或 bicycle。
2. 每个场景的接入流水线（已有工具，逐场景跑）：`tools/splat_to_ply.py` 转换 → 尺度与重力对齐写进 `scene.json`（`nav_to_world`、`ground_z`）→ 碰撞代理 v2 → 占据地图可视化 → 用 `probe_visual` 沿候选路线量机载帧清晰度（Laplacian 方差）→ 写 ≥4 条路线（`routes[]`，planner 离线校验）→ 教师 2 集 × 路线冒烟 → 三视角视频 → 20 集 pilot → 放量。
3. 相机统一：3DGS 渲染与 Gazebo E1 模板都用 **60° HFOV、320×240**（用户 11-07 拍板，沿用老实验）；`e1_mini.json camera_fov` 只作真机记录，sim-to-real 阶段再处理视场差。
4. 质量门禁：每帧 `frame_quality`（Laplacian 方差）落盘，低于阈值的帧比例进 manifest；路线级「雾区比例」超限不进训练集。
5. **W1b 自采 3DGS 数据集**（用户团队实地采集，我们出规范与处理流水线）：采集规范、处理步骤、验收标准见 `DATASET_3DGS_SELF_CAPTURE.md`；目录 `$DATA_ROOT/uav_rgb/3dgs-scenes/<site>/`；进论文时作为「自采场景组」与公开场景组分开报。

### W2 教师算法（已定稿：P1 主贡献）

为什么现在的学生会崩（SR 0.90 → 0.06）有两个可引用的形式化解释：离线行为克隆在部分可观测下会「锁住」历史动作，只有 on-policy 聚合能恢复（Swamy et al. 2022）；特权专家不知道学生看不见什么，给出的标签在部分信息下不可模仿甚至不安全（Warrington et al. 2021, A2D）。A* 硬标签 BC 同时踩中两条。

**P1 信念边际化的运动原语 cost-to-go 软标签 + DAgger**（教师侧，约 8 人日）

- 命题：标签不是「专家在真世界里的最优动作」，而是「在学生可能看到的所有世界里，每个动作的期望 cost-to-go」，按软最优转成分布，再 on-policy 聚合。
- 依据：非对称模仿最优解 π*(a|o)=Σ_s p(s|o)π_E(a|s)（Warrington 2021）；QMDP（Littman 1995）；cost-to-go 监督（AggreVaTe, Ross & Bagnell 2014）；搜索→软策略目标（Expert Iteration, Anthony 2017）；温度 = 代价尺度（Levine 2018）；必须 on-policy（DAgger, Ross 2011；Swamy 2022）。
- 我们能显式构造 p(s|o)：学生视锥内未遮挡的几何固定，不可见部分按场景先验（森林柱密度 / 3DGS 占据统计）重采样，同时采样执行延迟与漂移。
- 实现：`teacher_kino.py` 暴露 `plan_costs()`（四个首原语的 cost-to-go，不可行 = inf）→ 新 `sim/tools/teacher_belief.py`（可见集合、K=8–16 个假设、QMDP 平均、softmax(−Q̄/α)，α 取一步前进代价）→ 输出 UAVSOFT1（`tools/join_soft_targets.py` 已有格式）→ `collect_3dgs.py` 加 `--policy student`（Gazebo 侧 `collect.py` 已支持）→ DAgger 2–3 轮。学生侧零改动（`--soft-targets --distill-alpha`）。
- 对照与消融：A* 硬标签 BC / 真世界 kino 硬标签 / 真世界 kino 软标签（无信念）/ P1 无 DAgger / P1 全量；K∈{1,4,16}；QMDP vs CVaR 聚合；α×{0.5,1,2}。
- 它合并了旧计划的 A1+A2，**替代** A3（不是给样本降权，而是改变标签的定义）。
- RL 特权教师（旧 A4）不单独做：按 Xing et al. 2024 的顺序，先 P1 的模仿，再用非对称 critic（`rl_teacher/core.py` 的 13 维特权状态）做 RL 微调，放第三阶段，只在 P1 平台期后做。

### W3 学生算法（已定稿：P2；P3 探索）

**P2 双帧 + 低分辨率深度分带/TTC 辅助头**（学生侧，约 9 人日）

- 命题：学生输入 (I_t, I_{t−1}, 上一步动作)，第三层 64×20×15 特征上接 1×1 卷积预测每格 8 个深度分带（或「n 步内接触」二值 TTC），仿真深度监督，部署丢头只用 RGB。
- 依据：depth-as-loss 优于 depth-as-input，低分辨率分类即可（Mirowski 2017）；二值 TTC 分类稳健（Badki 2021）；12×16 几何足以避障并迁移真机（DiffPhysDrone 2024；Seeing Through Pixel Motion 2024）；训练期辅助头部署丢弃（NanoFlowNet 2023）；双帧的几何量：1 m 处障碍 0.5 s 内视差约 8.6°≈23 px，可分辨；偏航步的旋转光流由上一步动作输入补偿。
- 警告（Swamy 2022）：双帧学生离线 BC 会更差，**必须配 P1 的 DAgger**。
- 实现：`preprocess.py` UAVR 升版（6 通道、prev-action、20×15 分带目标，分带边界 0.3/0.5/0.8/1.2/1.8/2.5/4 m/∞ 与 ToF 量程对齐）；`src/model.cpp` `in_c=6` + `aux Conv2d(64,8,1)` + prev-action 经 FiLM 口；新算子逐格 softmax CE；L = CE_action + λ·CE_depth，λ=0.5 起；NPU 首层 MAC 翻倍后总量仍 <60M。
- 消融：单帧无头 / 单帧有头 / 双帧无头 / 双帧有头 / +prev-action；分带 vs TTC；λ∈{0.1,0.5,1}；关键检验是换动力学（τ ±30%）后 SR 跌幅是否小于基线。

**P3 以 3DGS 渲染器为一步世界模型的 Bellman 一致性正则**（联合，探索，约 9 人日，M4 后视 P1/P2 结果决定）：学生 logit 解释为 −Q，对渲染出的反事实下一帧加 TD 残差（DQfD 形式）。小 CNN 上 TD 稳定性无先例，排最后。

不做：可见性加权、门控、自适应权重、注意力小改。

### W4 评测与论文材料

- 主表：行 = {A* 硬标签 BC, kino 硬标签, kino 软标签, P1, P1+P2, (P1+P2+P3), 各消融}；列 = 场景 × {SR, 碰撞率 CR（超时不算碰撞）, SPL（l_i 取 A* 最短路长）, 平均速度, false_forward}。场景组：forest_e1 三档缝宽（Gazebo，按 AvoidBench 可穿越性分桶）、3DGS 训练场景 2–4 个、3DGS 保留场景 2 个（OpenFly 的 seen/unseen 口径）、自采场景。
- 样本量：每格同一起终点集合 ≥200 集（Wilson 95% 半宽 p=0.8 时约 ±5.5 pp）；两方法 McNemar（已有），多方法 Cochran's Q + 两两 McNemar + Holm；训练种子 3 个，报 IQM 与分层 bootstrap CI（Agarwal 2021），用 Colas 2018 的工具做功效复核。
- 跨场景：训练集 A → 保留集 B 的 SR 差作「泛化缺口」列。sim-to-real：真机 ≥100 次（SOUS VIDE 105 / Splat-Nav 124 口径），同可穿越性下报仿真/真机 SR 比。
- 口径沿用 `EVAL_STANDARD.md`，新增 SPL、CR、task_success（到达且停住）三列；val 太小的问题（18 集、SAFE 28 样本）在 §8 待拍板。
- 入口 `tools/compare_datasets.py`（骨架已进 main，`--dry-run` 可跑）→ 守护进程跑 → `summary.md` 直接可贴论文。

### W5 工程纪律（每个工作流都守）

- 代码风格按 personal-skill §6.10：入口脚本顶部配置区、每个参数一行并写取值依据、从上到下就是流程；库代码照常封装；注释写「为什么」。不为省 token 写半成品，不留重复定义。
- Git：一个主题一个分支（`scene-<name>`、`teacher-kino-value`、`teacher-rl`、`student-twoframe`…），合并前 `tools/check.sh` 全过，合并后立刻 `bash tools/git_mirror_alvln.sh` 推 GitHub；每个里程碑打 tag（`demo-2026-11-12`、`paper-v0`），方便回滚和 diff。
- 实验一律守护进程（`setsid nohup` + `run.pid` + `DONE`），日志在数据盘；用户的 YOLO 训练（PID 82761）不碰。
- 每个任务一份 worklog（做了什么 / 没做什么 / 验证到哪），每个实验一份 `experiments/` 文档。

## 4. 里程碑

| 里程碑 | 日期 | 交付 | 过线 |
| --- | --- | --- | --- |
| M0 审计与基线 | 11-07 | 6 份审计/调研报告收齐并合并修复；E1 流水线四组数字；3DGS pilot 20 集干净 | `check.sh` 全过；SITL 倒飞/漂移有根因且修复或有明确结论 |
| M1 场景 | 11-09 | ≥2 个新 3DGS 场景可飞（路线 + 教师视频）；Gazebo E1 外观；相机参数统一决策落地 | 每场景教师 ≥4 路线 × 2 集到达且 0 碰撞 |
| M2 Demo | 11-12 | 视频 + 一页数字 + SITL 曲线；规划 v2（算法定稿） | 用户验收 |
| M3 算法 v1 | 11-20 | T1 + T3 落地（T2 视调研结论）；S1/S2 落地；学生 v2 在 Gazebo ∪ 3DGS 上训练；多场景矩阵 v1 | 相对 BC 基线 SR 提升且区间不重叠（≥20 几何/路线） |
| M4 论文材料 | 12-05 | 完整矩阵 + 消融 + 方法/实验章节草稿 | 每格 n≥20、检验齐全 |
| M5 真机 | 依硬件团队 | 真机飞行记录回灌对比 | `ROADMAP.md` 阶段 2 门禁 |

## 5. agent 编队（并发上限 6）

| 批次 | agent | 分支 | 产物 |
| --- | --- | --- | --- |
| 第三阶段（进行中） | SITL / VIZ / DATA / 3DGS / SRC / ALGO | `*-20261107` | 审计报告 + 修复；算法调研 |
| 第四阶段（M0 合并后，11-07 起） | SCENE-A（Wanderland 接入 + 碰撞代理 v3 网格真值对照）、SCENE-B（Mip-NeRF 360 garden/bicycle 以 Inria PLY 重做 + 路线）；TEACHER-P1（`plan_costs` + `teacher_belief.py` + 3DGS `--policy student` + DAgger 脚本）；STUDENT-P2（UAVR 升版、`in_c=6`、辅助头、逐格 CE 算子、导出）；EVAL（compare_datasets 跑通、SPL/CR/task_success 列、IQM/bootstrap 脚本、论文表模板）；SITL 收尾（倒飞/漂移根因修复后真 Gazebo 验证） | `scene-wanderland`、`scene-mipnerf`、`teacher-p1`、`student-p2`、`eval-matrix`、`sitl-20261107` | 可飞场景、P1/P2 代码与单测、矩阵 v1 |
| 第五阶段（M3 后） | P3 探索；RL 微调（Xing 2024 第三阶段，非对称 critic）；自采场景接入（用户采集完成后） | `student-p3`、`teacher-rlft`、`scene-<site>` | 视 P1/P2 结果决定 |

每个 agent：自己的 worktree、自己的构建目录、不起 Gazebo（主会话统一跑真 Gazebo 验证）、不 commit（主会话合并）。

## 6. 不做的事

- 不改 CF-Drone 固件（硬件团队做）；不把固件油门余量当研究点。
- 不在学生侧引入 PyTorch/TensorRT；3DGS 的 gsplat venv 只做数据工具。
- 不用 VLN 基准的 20 m 成功半径与米级瞬移动作；不在一张表里混报 X3 与 E1。
- 不做「加权 / 门控 / 自适应」类小改当贡献。
- 不下载超过数据盘余量 1/2 的场景数据（现余 45 GB）。

## 7. 风险

- 3DGS 公开场景大多是手持地面采集，离采集路径远就发糊：每个场景要先量清晰度再定路线，可能只有部分区域可飞。
- RL 教师（T2）训练与调参不确定性大，放在 M3 之后；T1/T3 先保底。
- 学生格式升级（双帧、价值头）会让旧 checkpoint 不可比：新旧分开报数。
- 单机 12 核 + 软件渲染是 Gazebo 采集瓶颈；主采集向 3DGS（GPU 渲染）倾斜。

## 8. 需要用户拍板

已定（2026-11-07 05:50）：相机参数沿用老实验 60°/320×240；排期同意；论文机器人与视觉并齐；视频优先。

已定（06:10，按调研报告）：RL 教师不单独做，改为 P1 之后的 RL 微调（第五阶段）。

待定：

1. **评测口径**：val 只有 18 集、SAFE 约 28 个样本（SRC 报告），宏 F1 噪声大；是否把 val 比例从 10% 提到 15–20%（改 split 规则要 bump 几何版本），或 best-ckpt 改按 val 平衡准确率选。
2. **自采场景的采集设备与场地**：手机还是运动相机；室内走廊/实验室还是校园室外；决定后 `DATASET_3DGS_SELF_CAPTURE.md` 的参数表细化。
3. **需要用户申请的数据集账号**：DL3DV（Terms of Use，研究可用）、ScanNet++（账号申请）。不申请就只用 Callinstrasse + Wanderland + Mip-NeRF 360 + Zip-NeRF。
4. **SAFE 的语义**（DATA/SITL 报告都指出）：E1 的 SAFE 不是「停住」而是滑行 0.07–0.21 m 再漂移；教师在柱旁输出 SAFE 对 E1 不安全。两条路：执行器加位置保持（SITL 侧，优先）或教师近柱不输出 SAFE（改标签语义，要 bump 版本）。等 SITL 报告出来一起定。
