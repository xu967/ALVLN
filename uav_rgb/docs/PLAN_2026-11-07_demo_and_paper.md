# TinyAerialVLN 总规划 v1：可展示的 demo + 有算法意义的模型（2026-11-07 立）

- 状态：v1 草案。算法选型（§4）等 `tmp/phase3-20261107/reports/ALGO_survey_20261107.md` 调研报告出来后定稿为 v2；其余部分现在就按此执行。
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

1. 选场景：从调研清单里挑 ≥3 个「有现成 3DGS PLY、许可允许研究使用、2 m 高度可飞 8–30 m」的真实场景（室内一个、室外一个以上）。现有 bicycle / kitchen / room / playroom 先评估能不能直接用。
2. 每个场景的接入流水线（已有工具，逐场景跑）：`tools/splat_to_ply.py` 转换 → 尺度与重力对齐写进 `scene.json`（`nav_to_world`、`ground_z`）→ 碰撞代理 v2 → 占据地图可视化 → 用 `probe_visual` 沿候选路线量机载帧清晰度（Laplacian 方差）→ 写 ≥4 条路线（`routes[]`，planner 离线校验）→ 教师 2 集 × 路线冒烟 → 三视角视频 → 20 集 pilot → 放量。
3. 相机统一：3DGS 渲染、Gazebo E1 模板、`e1_mini.json camera_fov` 三处必须同一组内参；当前 Gazebo 是 X3 的 60°，待定（§8）。
4. 质量门禁：每帧 `frame_quality`（Laplacian 方差）落盘，低于阈值的帧比例进 manifest；路线级「雾区比例」超限不进训练集。

### W2 教师算法（候选，调研后定稿）

原则：教师可用特权信息和 PyTorch；产物是标签（硬标签或四动作价值分布），不进学生输入。候选三条，最终从中定 1 主 + 1–2 对照：

- T1 **动力学感知的运动原语规划教师**（`teacher_kino.py` 转正）：状态含速度，四个原语按 SITL 实测闭环响应积分，必撞状态 fail-closed；输出每个动作的 cost-to-go（到目标的规划代价 + 碰撞/净空惩罚），softmax(−cost/τ) 作为价值软标签。文献根基：Fast-Planner 的 kinodynamic A*、AlphaGo 式策略–价值目标、policy distillation。
- T2 **RL 特权教师**：在从 SITL 辨识出的快速 Python 代理动力学上，用非对称 actor-critic PPO 训练看特权状态（机体系最近 K 个障碍、速度、目标向量）的策略；奖励形式引用具体论文，不自造。学到提前转、少刹车的时序行为是规划教师给不了的。文献根基：Learning by Cheating、Agile Autonomy、RMA、D-NAV。
- T3 **on-policy 聚合（DAgger 族）**：学生闭环飞，教师在学生到达的状态上重贴标签，迭代 2–3 轮；直接对付「换载具 SR 0.90 → 0.06」这种状态分布漂移。`eval_closed_loop.py --relabel`、`preprocess.py` 放行 `executed_by=student` 的 provenance。

不做：可见性加权、门控、自适应权重这类小改。

### W3 学生算法（候选，调研后定稿）

模型要上 SG2002 NPU，参数量不大动；改的是**输入的几何信息**和**监督的形式**：

- S1 **双帧输入作为运动视差**：连续两帧给网络，几何上等价于给了光流/视差（单目测距的基本信号）；不叫「堆帧」，要在论文里按运动视差论证，并做单帧 vs 双帧对照。
- S2 **价值蒸馏头**：输出四个动作的价值（或分布），损失是对 T1/T2 软标签的 KL；闭环时取 argmax。与硬标签交叉熵对照。
- S3 **辅助几何头（仅训练期）**：粗深度（20×15）由仿真深度监督，部署时丢掉；多任务监督是成熟做法，要做有/无对照。
- 三者都要在 `src/` 加算子/格式（checkpoint v6、UAVR 多帧），SRC 线负责；`student.py` / `infer_main` 带一帧缓存。

### W4 评测与论文材料

- 口径沿用 `EVAL_STANDARD.md`：离线四件套 + 闭环 SR/碰撞/false_forward；新增 SPL（以教师路径长为参考）和「到达且停住」的 task_success。
- 两数据集对比矩阵（`experiments/2026-11-07-dataset-compare-plan.md`）扩成多场景矩阵：训练集 ∈ {Gazebo, 3DGS 场景集合, 混合}，测试域 ∈ {Gazebo 20 seed, 每个 3DGS 场景留出路线}。
- 入口 `tools/compare_datasets.py`（DATA agent 出骨架）→ 守护进程跑 → `summary.md` 直接可贴论文。

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
| 第四阶段（M0 合并后） | SCENE-A、SCENE-B（各接 1–2 个新 3DGS 场景）；TEACHER（T1 价值标签 + T3 聚合）；STUDENT（S1/S2 的 C++/CUDA 与格式）；EVAL（compare_datasets 跑通 + 论文表模板）；E1-VISUAL（Gazebo 机体外观，小任务） | `scene-*`、`teacher-value`、`student-v2`、`eval-matrix` | 可飞场景、算法 v1、矩阵 v1 |
| 第五阶段 | TEACHER-RL（T2）；视调研结论决定 | `teacher-rl` | RL 教师与对照 |

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

1. **相机内参**：Gazebo E1 模板现在沿用 X3 的 60° HFOV（为了旧 X3 学生可比）；E1 实际相机未定型号（参数表按 GC4653 假设 81°×51°，裁 4:3 后 64.9°）。建议：D1 demo 保持 60° 不动；论文主线从 M1 起三处统一到一组「E1 相机」内参并重训，旧 X3 学生只作历史对照。
2. **demo 日期**：本文按 11-12 排；若更早，M1 只保 1 个新场景。
3. **论文目标**：按机器人会议（ICRA/IROS/RA-L）口径排实验；若目标是视觉会议，场景数与真实外观对比权重要再加。
4. **T2 RL 教师**要不要进 M3（增加 1–2 周）：建议等调研报告的收益评估再定。
