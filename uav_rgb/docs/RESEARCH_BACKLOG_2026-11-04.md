# 研究层任务台账（2026-11-04 另立）

- 立项依据：`code/TinyAerialVLN.md` 的「优先执行的修复门禁」第 2/3/4 条、§3 `SIM-08`–`SIM-20`、§4 `PR-01`–`PR-08`，以及 `docs/worklog/2026-11-04-1745-realism-rl-integration.md` 的「研究层遗留」。
- 状态：**全部未开工**。本轮只立项、划边界，不写实现、不改数据、不重算指标。
- 与实施主线的关系：`docs/TASKS.md` 仍是实施任务唯一来源；本台账**不并入**该表，也不改变其任何条目状态。
- 唯一可执行主线仍是 `code/uav_rgb/`；`code/ALVLN/uav_rgb/` 历史镜像禁止混用；CF-Drone 固件不改。

## 为什么必须另立

这些任务会改变 `dataset_version` / `metric_version` / `protocol_id` / `teacher_version`，一旦混进 `1e0d6fc` 这类集成提交，旧 `campus.v2`、480/1000 集、旧 checkpoint 与历史 SR 结论将失去可比性。`TinyAerialVLN.md` §5 已明确：从 `campus.v2` 到 v3、从 reached 到 stopped 的改动均为**下一期独立 PR**，需独立重算数据与历史分数，不得伪装成无副作用补丁。

**全局纪律（每条任务都适用）**

- 旧数据、旧 checkpoint、旧指标数值**冻结**；新任务独立命名、独立目录，不覆写 `campus.v2` 等旧标识。
- 每个任务开工前先写明新 `protocol_id` 与版本迁移方案；完成前不得与旧版本合并统计。
- 证据分级：离线单测 ≠ Gazebo verified ≠ CF-Drone SIL ≠ 实机；不得跨级宣称。
- 每项修复必须有可复现失败用例、单一原因假设、修后可比较证据；不顺手重构无关模块。
- 正式 A/B 用固定 identical episode IDs，报告全部失败与重试策略，以场景 ID 为聚类统计单元。

## 任务总表

| ID | 优先级 | 对应条目 | 主题 | 前置 |
| --- | --- | --- | --- | --- |
| R-01 | P0 | 门禁 2、`SIM-09`、`D04`、`PR-04` | 指标合同冻结与真实 STOP 口径 | 无 |
| R-02 | P0 | 门禁 2、`SIM-08`、`D01`、`PR-02` | Campus 颜色因果反事实与 `campus.v3` 版本化 | R-01（指标口径先定） |
| R-03 | P1 | `SIM-10`、`SIM-15`、`D15`、`PR-03` 扩展 | split 拓扑正交化与 20+ 独立地图配对闭环 | R-01 |
| R-04 | P0 | 门禁 3、`PR-01` 迁移、`PR-06`、`SIM-05`、`SIM-11`、`D02`、`D06` | 历史教师数据重导与在线/离线时序 parity | R-01 |
| R-05 | P0 | 门禁 4、`SIM-12/13/14/19`、§4 阶段 B–D | Gazebo 物理闭环、机体系速度与 Sim-to-Real 门禁 | R-01、R-03 |
| R-06 | P2 | `PR-05`、`PR-07`、`PR-08`、`SIM-17/18/20` | 训练/平台工程门禁与性能证据 | R-01 |

### R-01 指标合同冻结与真实 STOP 口径

- **目标**：把 `eval_closed_loop.summarize_episode` 的 reach-only `success` 保留为历史指标，另立并显式输出 `reach_ever`、`final_in_radius`、`stop_action_issued`、`stable_termination`、`collision_free`，预注册 `task_success`（到达半径 + 终止动作 + 速度/姿态稳定窗口 + 安全条件 + 未超时）。
- **必须产出**：`docs/EVAL_STANDARD.md` 修订、UI 读取契约、五类反例回归、所有历史结论标注旧 `metric_version`。
- **验收**：五口径可分别复现；无可靠速度数据时不得输出 `stable_termination` 通过；旧 `success` 数值逐位不变。
- **不做**：不重定义旧 `success`、不覆盖历史 `eval.json`、不改学生模型。

### R-02 Campus 颜色因果反事实与 `campus.v3`

- **目标**：现 `campus.layout` 左右 target 同色（`COLORS[seed%4]`），颜色是场景属性而非竞争性指称，消融颜色词无法证明语言能力。建同屏异色候选、颜色与左右正交随机化的反事实任务。
- **必须产出**：`campus.v3` 独立任务名与数据目录；同图换指令 / 同指令换材质都使正确目标切换的用例；`RGB-only` / `text-only` / `shuffled-text` / `blind` 四基线；目标 ID 与可见性校验。
- **验收**：`campus.v2` 冻结不覆写，旧 480/1000 集与旧 checkpoint 标记旧任务；color swap 导致目标切换且几何固定、任务可解、物理间隙满足安全门禁。
- **不做**：不改 `campus.py` 现有 v2 生成逻辑，不重算旧集。

### R-03 split 拓扑正交化与独立地图配对闭环

- **目标**：现 `campus.split_map` 把三种拓扑分别绑定 train/val/test，几何陌生度与外观、光照、排列高度耦合；A/B 增益可能只是外观鲁棒。建同族未见几何、跨族、正交外观扰动三套 split。
- **必须产出**：≥20 个独立地图的 T0/T1/T2 同场景配对数据；固定 identical episode IDs；以场景 ID 为聚类单元的统计；`split_for_seed` / `--verify` / 打包器三者 task 身份一致。
- **验收**：任一 split 内包含全部三种拓扑；伪造同图不同 split 必被 `--verify` 拒绝；报告全部失败 episode 而非只统计成功子集。
- **不做**：不改现有 8 seed 几何分组的 base/l1 默认行为。

### R-04 历史教师数据重导与时序 parity

- **目标**：交付包已修 `teacher/dataset.py` 的滑窗 `step` 取原始 row，但**门禁 3 指出离线导出 step 索引的历史污染未清算**；另需在线 `teacher_policy.py` 与离线 `dataset.py` 的 dt/pose 逐值一致，并测深度与 RGB 的 signed gap 分布后再定「最近帧」或「严格不晚于 RGB」。
- **必须产出**：全部历史教师 `.npz` / `UAVSOFT1` 的重新审计清单；有问题的按原数据哈希重导与 join；同一段 timestamped 轨迹在线/离线 pose 输入逐值相等的回放用例；缺帧/重复/跨坏帧 5 类用例。
- **验收**：窗口每个 sample 的最后 RGB 与 `(episode_id, step)` 严格一致；连续帧行为不变；换口径必须更换数据版本。
- **不做**：不直接收紧 `DEPTH_STALE_NS=100ms`（先测分布），不覆盖旧 `.npz`。

### R-05 Gazebo 物理闭环与 Sim-to-Real 门禁

- **目标**：`MulticopterVelocityControl` 的速度是**机体系**；`TopicSub.healthy()` 只证明子进程存活；校园纹理含绝对路径。需在 Gazebo Harmonic 上做真实物理、相机、坐标系、碰撞与延迟门禁。
- **必须产出**：yaw≈0 / yaw≈90° 短前进动作的世界系位姿对拍；collision-test world 注入确定性接触验证「事件锁存 → collision=True → 终止」并配无接触负样本；`model://` 资源跨服务器可移植性验证；接触/订阅失败率报告。
- **验收**：§4 阶段 B（服务器 Gazebo）、C（正式语义 VLN 基准）逐项留原始 stdout；阶段 D（CF-Drone / SG2002 / ESP32）在独立验收前**禁止宣称通过**。
- **不做**：不改 CF-Drone 固件；不把 Gazebo X3 通过写成 CF-Drone SIL 或实机自由飞行。

### R-06 训练/平台工程门禁与性能证据

- **目标**：`--appearance v1` 等参数曾被静默忽略（`PR-05`）；resume 非严格等价（`PR-07`）；CUDA arch 硬锁 `sm_75`（`PR-08`）；`gz topic -e` 有 Base64+JSON 开销（`SIM-17`）；`test_student.py` 1.5s 启动假超时（`SIM-18`）。
- **必须产出**：不支持选项直接报错的回归；连续跑 vs resume 的 batch IDs 一致；config 字段逐项落到 argv 且未处理字段 fail closed；CUDA arch 配置化与设备自检；`RTF` / `step_wall` / `p95_camera_to_action` 实测后再做性能优化；`tools/check.sh` 全量构建与 CTest 在有 GPU 环境复跑。
- **验收**：正确性门禁先过，再谈性能；不以理论 TOPS 代替实测。
- **不做**：不因 CLI 慢就改动作时间语义；不重写全部 CUDA。

## 建议顺序与并行纪律

1. **R-01 先行**：指标口径未冻结前，任何新数据/新任务的结论都不可比。
2. `PR-01/02/03` 一类共同决定数据身份的任务，旧数据冻结前禁止跨版本合并；R-02 / R-03 / R-04 在 R-01 后可并行。
3. R-05 需要 R-01 的口径与 R-03 的独立地图；R-06 的 UI / 性能可延后。
4. 任一任务开工时把状态改为 doing、完成改 done 并链 `docs/worklog/` 过程文档；不在本文件内维护状态。

## 环境与证据边界（2026-11-04 实测）

- 本机无卡：`nvidia-smi` 列不出设备；`gz` CLI 8.15.0 存在但 `DISPLAY` 未设置、无 EGL 设备。
- Python 3.11.15 / numpy 1.26.4 / Pillow 12.2.0；`gymnasium`、`stable_baselines3` 未安装；系统 `python3` 的 torch 缺 `libcudnn.so.9`，两条 `teacher/test_dataset_*.py` 需 `teacher-venv` 解释器。
- 因此 R-05 的 Gazebo 阶段、R-06 的 CUDA/CTest 全量在本环境**不可执行**，立项不等于已验证。

## 本轮（1e0d6fc 之后）对研究层做了什么

- 只立项与划边界；未开工 R-01–R-06，未产出任何新数据、新 protocol ID 或新指标数值。
- 同步了 `DELIVERY_MANIFEST.json` 的 9 个补丁后哈希，把 3 个顶层交付文档纳入主线仓库 `docs/delivery/`；均属记录性改动，不改变实验口径。
