# 当前实施状态（2026-11-04）

本轮范围与验收见 `CAMPUS_BASELINE.md` / `experiments/2026-11-03-campus-baseline.md`。
优先级：先让修复与校园pilot通过功能验收，再采集配对训练集和报告跨拓扑闭环。
2026-11-04 新增桌面 Studio 任务：L1/UI 冒烟已通过，目标 Gazebo 8 三视角渲染和 GPU L3 仍待用户桌面环境验证。
风、相机标定/成像扰动、实机参数拟合属于后续独立实验，不能由本轮纹理效果推断。
2026-11-07 04:10：SIM-E1 代码侧全部进 main 并推 GitHub（SITL 库、Gazebo 插件、3DGS 接入、可视化修复、学生 C++ 加速合并、`tools/e1_pipeline.py`）；
E1 流水线守护进程在跑（见 `../progress.md`），四组对照数字出来后此行改 done。
下方为历史任务表，不能覆盖以上最新状态。

# 任务板

- 建立：2026-10-29 22:19
- **唯一任务来源**：开工把状态改 `doing`，完成改 `done` 并链 worklog。HANDOFF / ENGINEERING_RULES 只保留优先级理由，不再各存一份状态。
- 文件域：`code-4b` = `src/ sim/tools/ tools/ docs/`；`code-3b` = `ui/`；子 agent 只读，或只写**新文件**。跨域先发消息。
- 验证三级：**L1** 相关单测（每次改）；**L2** `bash tools/check.sh`（每个任务完成，不许只跑子集）；**L3** 真实 Gazebo/训练（里程碑，先在本表登记，同一时刻只一个会话跑）。
- Git：一个主题一个提交，精确 `git add <path>`；首行「改了什么 + 为什么」，正文列行为变更与验证命令；文档单独提交；不 `git add -A`；用户明确授权后才 commit/push。

## 任务总表（进行中 + 已完成）

| ID | P | 状态 | owner | 文件域 | 任务 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| SIM-E1 | P0 | **doing**（2026-11-06，用户授权；边界见 `/root/rivermind-data/uav_rgb/grill-me/boundary-v4.txt` + `boundary-v5-addendum.txt`；21:05 起由 Cursor 会话接手，4 个 agent 并行） | 当前会话 | `sim/cf_sitl/`（新，GPLv3 隔离）、`sim/worlds/`、`sim/tools/`、`sim/3dgs/`、`tools/`、`ui/`、`docs/` | 仿真载具换成固件在环的琛光 E1 MINI（CF-Drone e7f507f 原样编入 + 电机/电池/刚体/IMU/ToF/光流模型 + 定高定速执行器参考实现），Gazebo 插件与 3DGS 共用；修可视化 REVIEW #5–#14/#17/#18 与 AUDIT 相关项；demo。**v5 增补**：主场景 Callinstrasse 真实校园 3DGS + Gazebo 柱阵对照；在 Gazebo 柱阵 + E1 SITL 上重采重训 E1 学生（20 集 pilot 过 audit 再放量，守护进程跑）；代码完成后推送 GitHub | SITL 自检；Gazebo 脚本序列 + 教师闭环三视角视频；E1 机体上同批 20 个冻结 test seed 四组对照（教师 / 旧 X3 学生 / 新 E1 学生 / 盲前进，observation 语义，单独成档，标明与 X3 不可比）；3DGS Callinstrasse 教师闭环三视角视频；`tools/check.sh` 全过。参数 `calibrated=false`，真机数字待实测 |
| STUDIO-1 | P1 | **doing**（2026-11-04；代码与 UI 冒烟完成，Gazebo L3 待用户桌面验证） | 当前会话（用户授权） | `ui/`、`sim/worlds/uav_rgb.sdf`、`docs/` | 原生桌面训练/实验工作台：真实 trainer 监控、实验自动归档、三视角 Gazebo 展示、学生 live demo、闭环评测入口 | `ui/test_studio.py` 8 项；`test_monitor.py` 74 项 + 1 skip；`test_trainer_demo.py` 92 项；Xvfb UI/训练假后端/PNG/topdown 冒烟通过；无 `gz` 的当前容器无法完成真实相机 L3。见 `worklog/2026-11-04-uav-navigation-studio.md` |
| T7-R | P0 | **done**（2026-10-31 00:00） | code-4b | `sim/tools/eval_closed_loop.py`、`docs/` | dispatch 语义 20 seed 复跑：SR 0.55（11/20），collision 0.45，false_forward 0.30；A/B observation 2/2 vs dispatch 0/2 | 见 `experiments/2026-10-31-t7-r01b-and-window-ab.md`；0.90 只适用于 observation 语义 |
| C0 | P0 | done | code-3b + code-4b | git | 清 37 文件积压，按主题提交 | 工作树干净；5 个主题提交 |
| E6-1 | P0 | done | code-4b | `src/infer_main.cpp`、`src/CMakeLists.txt`、`src/tests/` | 常驻推理进程：stdin 收 H×W RGB uint8，stdout 回 action/probs/infer_ms | 加载现有 ckpt；同帧 100 次输出一致；JSON 可解析（`2026-10-29-2226-infer-process.md`） |
| E6-2 | P0 | done | code-4b | `sim/tools/eval_closed_loop.py`、`sim/tools/collect.py` | 复用 collect 的 gz 连接，学生动作闭环飞，碰撞即停 | 1 个 test 场景跑通并输出 JSON（`2026-10-29-2232-closed-loop-eval.md`） |
| E6-3 | P0 | done | code-4b | `sim/tools/eval_closed_loop.py` | 固定 test split 场景集（几何 gid 与训练不重叠） | seed 5 / gid `c8afd118…`，split=test |
| E6-4 | P1 | done | code-4b | `sim/tools/eval_closed_loop.py` | SR、碰撞率、false_forward、推理 p50/p95、动作分布 | JSON + `docs/experiments/2026-10-29-closed-loop-smoke.md` |
| E6-5 | P1 | done | code-4b | `docs/` | 现有 checkpoint 基线 20 episode | 20/20 碰撞、SR=0；`2026-10-29-closed-loop-20ep.md` |
| E6-6 | P1 | done | code-3b | `ui/` | 只读载入 `eval.json`：ckpt 文件名 + split、SR/碰撞/false_forward、p50/p95(warm)、四动作分布、per_episode 的 max_x/碰撞 | `aa8838ac` + 真实 JSON 回归 `71fdc0cf`（未 push）；GUI 116 项 EXIT=0 |

## 下一步（2026-10-30 22:55 审核后）

> 自主执行任务编排（无固定截止）：`docs/experiments/2026-10-31-overnight-autonomous-plan.md`。

**当前 P0 是 v2 500 集采完后的 audit → L2 文本训练 → 离线/闭环 L/I/S。**
T8 不单独立项，失败分类并入 T7-R；D5/G1–G6 暂缓；V8、CF-Drone、SG2002、
AerialVLN-S 官方基准冻结。

语言线（L0–L2）已过 D-STACK 决策门（全自研 C++/CUDA）；ECPT 只在 L2 显示
语言增益后评估。语言线按 **v1 observation 语义**采集/评测；dispatch 未采纳。
前置项 B4（infer 语义对拍）/ B1（数据集防替换）**未立项就被 L0/L1 越过**：
本轮定为显式豁免（B1 部分由侧车身份绑定 + `preprocess` 原子写兜底），
若要补做，在 L2 正式训练前完成 B4 冒烟。

| ID | P | 状态 | owner | 任务 | 验收 |
| --- | --- | --- | --- | --- | --- |
| D-STACK | P0 | **done**（2026-10-31 10:03，用户拍板） | 用户 | **全自研 C++/CUDA**：文本侧 CPU embedding + mean-pool + concat，不新增 CUDA kernel，不引 PyTorch | 已选定；L2 按此栈立项 |
| T4 | P0 | **done**（2026-10-31 00:54） | code-4b | 完整 20 seed observation v1 基线（`eval-student-v1-observation`） | **SR 0.90（18/20）**，collision 0.05，p50 10.26 ms；与历史 0.90 一致，失败 seed 23/88；与 dispatch 不可混报 |
| T5b | P0 | **done**（2026-10-31 06:10） | code-4b | dispatch 全量重采 600 集 → 596 集 audit PASS → 训练 → 闭环 | 闭环 SR **0.60**（12/20）；离线 test macro_f1 0.8287；**未达 v1−5pp（0.85），不采纳为 baseline**；T5b 数据/模型保留作证据 |
| B1 | P1 | 待立项 | code-4b | 训练数据替换防护：把 `batch_reader.cpp` 的 `verify_unchanged` 接入 trainer，或训练启动时对 open fd 做 fstat 校验 | 训练中覆盖 `.uavr` 被拒/检测；负例测试通过 |
| B4 | P1 | 待立项 | code-4b | infer 语义对拍：已知像素图案 → logits 与 NCHW 直喂一致；`infer_main` 子进程冒烟 | 通道/数值对拍通过；在 L0 改 TEXT 协议前完成 |
| D-GEODUP | P3 | 待立项 | code-4b | 几何近重复检测：3 位舍入 gid 会把只差 5 mm 的场景当不同几何（T7-R 实测 seed 49/57 动作逐位相同）；按柱坐标容差聚类，报告 test 内近重复组 | 近重复组清单 + 20 seed 独立性结论；不阻塞 T7 |
| L0 | P1 | **done**（2026-10-31 11:35） | code-4b | 指令数据通路：episode 旁路 `instructions.jsonl`（`episode_id → instruction`）+ infer `--text` CSV + 文本权重/token 表随 checkpoint（v4）；文本侧 CPU embedding+mean-pool，不新增 CUDA kernel | 400 集带指令且侧车带 `text_ids`；超长/越界/缺字段在读端直接报错（`test_text_branch` 覆盖）；`--text` 已进模型 |
| L1 | P1 | **doing（v4，卡在「停得准」）** | code-4b | 最小语言任务：1 模板 + 同色柱对 + 关系词 + 外侧 waypoint | **v4 结果**：离线 test `macro_f1 0.7740`；探针 `follow 0.8090`；**2026-11-02 按 goal_reach + observation 重跑**：keep SR **0.4043**（19/47，Wilson [0.276, 0.547]）/ shuffle **0.000** / 盲前进 **0.000**（McNemar p<1e-5）。**未过 50% 验收线**：15 次碰撞 15/15 都在 ≤1.5 m 内——停止点在柱阵外侧 ±2.4 m，40% 的 SAFE 帧视野里零柱子。旧 0.426 是复算值，作废。方案 A（判据 goal_reach，`REACH_RADIUS=1.5`）已落地并重跑。**等用户拍板 B（改几何）或 C（改任务定义）**；当前并行推进 Teacher 蒸馏 + GRU 路线。见 `experiments/2026-11-01-l1-v4-text-training.md`、`docs/PLAN_2026-11-02_uav_rgb_restructure_distill_gru.md` |
| L2 | P1 | **done（v4 已过语言门槛）** | code-4b | 语言 baseline + 消融：FiLM 融合已落地并训练 | v4 断开了 v3 的权衡：flip 0.964 下 macro_f1 0.774（v3 A 组 flip 0.970 时只有 0.479）；`follow` 0.6055 → **0.8090**。`acc_correct` 0.8814 仍未过 0.90（该阈值出自内部计划，非公开基准） |
| ECPT | P2 | 条件项（未立项） | — | 只在 L2 显示语言增益后评估：证据置信/相位 + STOP 门控 | 届时有同参数同数据消融 |

## 阶段目标（2026-10-30 立，同日拍板口径）

**第一阶段目标：指标超过公开方案的公开数字。**

**对标对象：已拍板选 (a) 自建基线阶梯。** 理由：Kaggle AirSim 10k 这份数据
**没有任何公开基线**（`kernelCount=0`、无论文、无仓库、无 split、无指标），
「超过他们」对它无定义；(b) AvoidBench 要在 Gazebo 里重建同类场景，成本高；
(c) 要重做一次数据接入。所以先用自建基线把口径跑通。

| 基线 | 口径 | 怎么跑 |
| --- | --- | --- |
| 盲前进 policy | 闭环 SR / 碰撞率 / false_forward | `eval_closed_loop.py --policy forward`（已实现） |
| 多数类先验 | 离线 `macro_f1` / `balanced_accuracy` | `tools/baseline_priors.py`（已实现；T2 test `macro_f1=0.1543`） |
| 我们的模型 | 两个都要 | 同场景同口径 |

- 离线必须**同时**报 `acc` + `macro_f1` + `balanced_accuracy` + 每类召回
  （`EVAL_STANDARD.md §1.1`）；闭环用 AvoidBench 的 SR/PO/EO。
- **违规**：只报准确率；按「有样本的类」算宏 F1 当结论。
- 建议验收线（用户可改）：离线 macro_f1 显著高于多数类先验；闭环 SR ≥ 50%
  （≥20 个 test 场景）且 `false_forward_rate` 低于盲前进。
- 每次汇报必须同时写清「我们的数字」「对标数字」「对标数字出处（论文/榜单/仓库 + 日期）」。

## 数据来源决策（2026-10-29 22:30 立，22:51 改，10-30 01:30 有实测结论）

**实测结论：公开数据集可用但不能单独用——`SAFE_NEUTRAL` 为 0 条。**

- Kaggle AirSim 10k 已下载（2.0 GiB）、校验（30000 文件序号连续无缺口）、
  清点、离散化、打包（`/root/rivermind-data/uav_rgb/airsim10k-uavr/dataset.uavr`，
  10000 样本 549 MiB）并完成一次训练。详见 `2026-10-30-airsim10k.md`。
- **能过的一条**：标签与深度几何强相关（`YAW_LEFT` 左边更近 82.1%、
  `YAW_RIGHT` 16.8%，差 25.8 sigma）→ 图像/深度/标签是对齐的。
- **过不了的一条**：`vx` 最小 1.00 m/s，势场法从不输出「停」，
  `SAFE_NEUTRAL` = 0 条。四类接口里的第一类它教不了。
- 因此：公开数据只能当预训练/链路验证，**四类模型仍必须靠 sim 自采**。
  D1 修好后自采的 `sim_dt` 已是 0.0% 变异，且 `label_recompute` 可逐条复算。

历史决策记录（保留）：

- 首版结论是「不接入公开数据集」，理由是 ICL corridor / AirGoal-10k /
  drone-vla-toolkit 的标签语义与相机对不上我们的 4 类接口。同日闸门实测推翻
  了「自采可信」这个前提（旧 200 集 4 项 FAIL），所以两条线并行推进。
- 候选：Kaggle `lukpellant/droneflight-obs-avoidanceairsimrgb10k-320x320`
  （AirSim，10k 样本，RGB 320×320 + depth + 连续指令 `(vx,vy,vz,yaw_rate)`，
  CC BY 4.0）。标签由确定性势场法脚本生成，可复算。
- **卡点**：下载需 Kaggle API token（`~/.kaggle/kaggle.json`），本机没有，匿名 403。
  HuggingFace 全站在本机不可达（curl 000）；Padhy 走廊数据集无公开下载地址。
- 拿到后的顺序：`audit_dataset.py` 审类别分布与标签-物理一致性 → 定义
  `(vx,vy,vz,yaw_rate)` → 4 类的映射 → 转 `.uavr` → 训练。审不过就不训。
- 已知风险：势场法可能从不输出「停」，`SAFE_NEUTRAL` 可能又为空；320×320 方形与
  本机 320×240 / 160×120 不同，需裁剪或缩放；AirSim 世界不是我们的柱阵。
- sim 自采是主线。**旧的「1000 集 × 20 步 ≈ 2 万帧，4 实例 3.5 h」估算作废**：
  `--steps` 默认已是 80（实测每前进 0.105 m，穿 3.5 m 柱区要 68–76 步），
  20 步只走到 x≈2 m、还在柱区外面。按 8 实例 ≈3.0 步/s 重估：
  4 万步（500 集 × 80）≈ 3.7 h，8 万步（1000 集）≈ 7.4 h。见 T5。

## 主线任务（T1–T7，2026-10-30 完成；T7 同版本复跑见「进行中」）

> 新 AI 从这里开工。细节与理由见 `docs/HANDOFF.md §七`。
> **T1 与 T2 是 T3 的前置；T3 不过不放行 T5。** 这条顺序是为了避免「采完 8 万帧
> 才发现几何或标签有问题」——现有 3726 样本被审计判 5 项 FAIL 就是这么来的。

| ID | P | 状态 | 文件域 | 任务 | 验收 |
| --- | --- | --- | --- | --- | --- |
| T1 | P1 | **done** | `sim/tools/scene_gen.py`、`split_manifest.py`、`teacher.py` | 车道几何放宽 + A* 教师：v3 缝 1.2 m、出生点 x=-0.5；局部规划在门柱前摆振，普通集改 A* | 20 test 场景 **20/20，0 碰撞，100%**；2026-10-30 用当前 teacher 复跑复核（20/20、label_recompute 0/800）见 `2026-10-30-t1-lane-geometry.md §4b` |
| T2 | P1 | **done** | `sim/tools/collect.py`、`scene_gen.py`、`teacher.py` | SAFE_NEUTRAL 数据：给采集加初始位姿随机化/近距障碍开局，让 `plan_action` 天然输出 SAFE | 52 集 / 2080 步 0 碰撞，audit 11 项全 PASS，SAFE 21.2%，每 split 四类都有；见 `2026-10-30-t2-safe-neutral.md` |
| T3 | P1 | **done** | `collect_parallel.py`、`preprocess.py` | **放行闸门**：新几何+新方案采 ~20 集 → 打包 → 10 轮短训练 | 52 集打包 2080 步 → 10 轮 → test recall `[1.000 0.504 0.278 0.941]` 全 >0；macro_f1 0.5859（先验 0.1543）；见 `2026-10-30-0500-t3-rollgate.md` |
| T4 | P1 | **done** | `src/train_main.cpp`、`src/trainer_eval.cpp` | 加 `--eval-only <ckpt>`：test 现在只用末轮权重跑，`ckpt_best.bin` 从未被评过 | 对 `airsim10k-run/ckpt_best.bin` 跑出 val/test 四件套并与 `metrics.jsonl` 对账；CTest 20/20；见 `2026-10-30-0320-t4-eval-only.md` |
| T5 | P0 | **done**（2026-10-30 16:19） | `sim/tools/collect_parallel.py` | 大规模采集：600 集 × 40 步（base-seed 200000），4 实例。实测 **1.506 步/s / 135 集/h**、4 h 26 min；8 实例会起飞超时。见 `experiments/2026-10-30-t5-scaleup.md` | 599 集 / 23960 步 audit **11/11 PASS**；1 集接触集已隔离（`yaw_direction` 反转步口径修正，见 `worklog/2026-10-30-1650-t5-audit-fixes.md`） |
| T6 | P0 | **done**（2026-10-30 18:29） | `tools/preprocess.py`、`src/` | 训练 + 离线验收：打包（23960 步）→ 训练（`--class-weights auto --drop-collision`，10 轮）→ eval-only 评 **best 权重**。2026-10-30 18:29 完成：6501.9 s / 28.5 样本/秒，best=epoch 8 | 离线四件套齐全 + 与多数类先验对比表：test macro_f1 **0.8589** vs 先验 0.1968，四类 recall [1.000, 0.952, 0.700, 0.747]，见 `worklog/2026-10-30-1829-t6-train.md`、`experiments/2026-10-30-t6-train.md` |
| T7 | P0 | **done**（2026-10-30 19:29） | `sim/tools/eval_closed_loop.py` | 闭环验收 vs 基线阶梯（盲前进 / 多数类先验 / 我们的模型，同场景同口径）。2026-10-30 用 20 个冻结 test seed 跑 T6 best（`t6-full-1640/eval-student`）：**SR 0.90、collision 0.05、stuck 0.00、false_forward 0.05**；盲前进基线 SR 0 / collision 1.0 / false_forward 1.0。碰撞率按 contact 计、stuck 单列（`EVAL_STANDARD §5.1`）。R01 后的同版本复跑已完成：v1 observation **SR 0.90（18/20）**、dispatch T7-R **0.55（11/20）**，两者禁止混报（`experiments/2026-10-31-t7-r01b-and-window-ab.md`） | 三件套汇报齐；见 `worklog/2026-10-30-1929-t7-closed-loop.md`、`experiments/2026-10-30-t7-closed-loop.md`；本表 T1–T7 全部 done |

**已知约束（T1 必须重推，不能只改一个数）**：三条车道装进 ±1.5 m 要求
`2×LANE_SEP + 中心摆动 ≤ 3.0`；柱数上限 9 的推导随间距变；`GATE_Y_ABS_MAX=0.25`
的视锥契约要复查；改几何必须 bump `SPLIT_SALT`（`split_manifest.py:26` 是硬规矩）。

## 待排（E6 后）

| ID | P | 状态 | owner | 文件域 | 任务 | 验收 |
| --- | --- | --- | --- | --- | --- | --- |
| D0 | P0 | done | code-4b | `sim/tools/audit_dataset.py` | 数据集验收闸门：9 条不变量，FAIL 即非零退出 | `/tmp/ds200` 跑出 5 项 FAIL（`2026-10-29-2241-dataset-audit.md`） |
| D0b | P0 | done | code-4b | `sim/tools/collect.py`、`audit_dataset.py` | actions 记 `pose_before`/`yaw_before`，audit 真实现 `label_recompute` | 新采集 16 步 0 不一致；depth 存盘**未做**（无消费方），见 `2026-10-29-2338-simtime-hold-and-pose-before.md` |
| D1 | P0 | done | code-4b | `sim/tools/collect*.py`、`eval_closed_loop.py` | 仿真时间动作窗口（`--hold` 改 sim-time） | 同 seed 两次采集 `sim_dt` 全为 1.000 s，cv 0.0%（旧 18.3%）；同上 worklog |
| D0c | P1 | done | code-4b | `tools/airsim_to_uavr.py`、`tools/preprocess.py` | 公开数据集接入：离散化 + 裁剪 + 打包 `.uavr` | 10000 样本打包成功，11 项自检通过，训练跑通（`2026-10-30-airsim10k.md`） |
| D2 | P1 | **→ T1** | code-4b | `sim/tools/scene_gen.py` | 车道几何决策（缝 0.902 m vs 机体 0.711 m） | 已并入 T1，验收以 T1 为准 |
| D3 | P1 | **→ T2** | code-4b | `sim/tools/collect.py`、`teacher.py` | SAFE_NEUTRAL 数据/四类定义 | 已并入 T2，验收以 T2 为准 |
| D4 | P2 | **done**（2026-10-30） | code-4b | `sim/tools/collect.py` | depth/contact 停更门禁 | depth 与 RGB 差 >100 ms 或 contact 订阅退出即失败；5 个注入用例通过，见 `worklog/2026-10-30-2129-d4-staleness-gate.md` |
| D5 | P2 | todo | code-4b | `src/device/`、`src/trainer.cpp` | device 层接入训练路径、RNG/best 状态 | 吞吐对比 + resume 一致 |
| G1 | P3 | todo | 子 agent（只写新文件） | `src/tests/` | CUDA 多 block parity | 新增用例通过 |
| G2 | P3 | todo | code-4b | `src/checkpoint.cpp` | resume 数据签名校验（v4 已被 text_dim 占用，升 v5；v4 读端保持兼容） | 换数据 resume 被拒 |
| G3 | P3 | todo | code-4b | `src/checkpoint.cpp`、`dataset.cpp`、`run_owner.cpp` | fsync、64 位溢出、目录 fd 锁 | 单测 |
| G4 | P3 | todo | code-4b | `src/CMakeLists.txt` | `-Wall -Wextra` 全开、`.clang-format` | 0 warning（真） |
| G5 | P3 | todo | code-4b | `sim/tools/collect_parallel.py` | 同 `--out` 并发 TOCTOU 加锁 | 并发负例被拒 |
| G6 | P3 | todo | code-4b | `src/train_main.cpp` | 生效配置落盘 `<out>/config.json` | 字段与 CLI 一致 |

## 已完成（2026-10-29）

| 项 | worklog |
| --- | --- |
| best-checkpoint + checkpoint v3 | `2026-10-29-2155-best-checkpoint.md` |
| 数据完整性 P1×3（-0.0 / 几何门禁 / 位姿停更） | `2026-10-29-2200-data-integrity-fixes.md` |
| C++/CUDA 静默失败 P1×4 | `2026-10-29-2215-src-silent-failure-fixes.md` |
| 本地监控台（无 web、真实训练接入） | `2026-10-29-2140-monitor-review2-fixes.md`、`2026-10-29-2205-real-trainer-integration.md` |

## 已完成（2026-10-30）

| 项 | worklog / 记录 |
| --- | --- |
| 交接文档刷新（T7 复跑状态） | `worklog/2026-10-30-2155-handoff-refresh.md` |
| D4 depth/contact 停更门禁 | `worklog/2026-10-30-2129-d4-staleness-gate.md` |
| ponytail review 跟进修复（CLI 回退/resume 测试/审计对齐/文档口径） | `worklog/2026-10-30-2101-review-followup-fixes.md` |
| Codex 审核 R01–R07 + 上一轮 review 两项 | `worklog/2026-10-30-1935-deepseek-review-fixes.md` |
| T5 全量采集与审计 | `worklog/2026-10-30-1650-t5-audit-fixes.md`、`experiments/2026-10-30-t5-scaleup.md` |
| T6 训练与离线验收 | `worklog/2026-10-30-1829-t6-train.md`、`experiments/2026-10-30-t6-train.md` |
| T7 闭环验收 | `worklog/2026-10-30-1929-t7-closed-loop.md`、`experiments/2026-10-30-t7-closed-loop.md` |
