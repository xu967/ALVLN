# uav_rgb 全仓代码与交接文档审核（5 路并行只读）

- 时间：2026-10-31 14:31
- 范围：`uav_rgb/` 的 docs、src、sim/tools、tools、ui；只读审核
- 结论：完成（审核完成，未改任何代码）

## 做了什么

- 完整读 `docs/HANDOFF.md`、`docs/TASKS.md`、`docs/EVAL_STANDARD.md`、
  `docs/ENGINEERING_RULES.md`、`README.md`、`CLAUDE.md`，以及 2026-10-31 的
  worklog 与 experiments。
- 开 5 个只读子代理并行审核：docs、`src/`、`sim/tools+preprocess`、
  `eval_closed_loop+student+ui`、旧审查 `CODE_REVIEW_DEEPSEEK_2026-10-30` 遗留项。
  其中 3 个首次因 API 402 中断，换 sonnet 重跑完成（不是 DeepSeek Pro）。
- 独立复核 `src/tests/test_trainer.cpp:305`：NaN 注入偏移仍是
  `48+4+ndim*8`，v4 头为 52 B、数据起点 `52+4+ndim*8`，测试实际写进
  `dims[3]` 高 4 字节，`check_shape` 先抛错，`check_finite` 未被执行，确认假绿。
- 汇总发现：无 Critical；High 7 条；Medium 约 20 条；Low 约 15 条；
  文档矛盾 24 条（重点列 10 条）。

High 摘要：

1. L1 指令 “...and stop” 与成功判据（越过最右柱、max_x≥goal）冲突。
2. 预注册的“同图换指令反事实对”在 400 集里实际为 0 对（C 组不成立）。
3. L1 eval 去重键 `(geometry_gid,target_color)` 过粗，36 集只评 34 集且 exit 1。
4. `eval.json` / collection_manifest / ds meta 不记录 `window_from`/`task`，
   observation 与 dispatch 无法从产物自证。
5. `instructions.jsonl` 未与数据集身份绑定；preprocess 先写 dataset 后非原子写
   sidecar，旧/半更新侧车可静默错配。
6. `audit_dataset.py` 递归扫描把 eval 产物当采集数据；在 l1-full-400-0803
   只读复跑得 434 集/17897 步并 FAIL。
7. `test_trainer` NaN checkpoint 测试假绿（已复核）。

## 没做什么

- 没改任何代码、配置、脚本或既有文档；只新增本 worklog。
- 没 commit / push。
- 没跑 `tools/check.sh`、CTest、训练、推理、Gazebo、UI；21/21 PASS 引用
  worklog，未独立复跑。
- 没深度审 `src/device/*`、`ui/` 全量、`collect_parallel` 的进程管理；
  没验证 T5/T2/pilot 数据。
- 没修任何发现的问题；没重训 L2；没跑 I/S/C 四组。
- 原因：本次是审核任务，用户未要求改动；运行态验证需独占 GPU/Gazebo，另行安排。

## 待确认

- L1 语义：改指令为 “pass the ... pillar”（需重采/重训），还是改任务为
  goal-stop；修复前是否冻结 L2。
- C 组：重新生成确定性反事实对，还是删除该预注册指标。
- T8 在 HANDOFF/TASKS/夜间计划里有三种含义，需用户确认。
- 是否授权按主题修复 High 并跑 `tools/check.sh`。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 文档/代码静态审核 | 完成，5 路覆盖 |
| `test_trainer` NaN 偏移 | 确认假绿（48 → 应为 52） |
| l1-full-400-0803 audit 只读复跑 | FAIL：434 集/17897 步、manifest_validity 17/434、teacher_waypoint 15 集 >1 m |
| `tools/check.sh` / CTest / 训练 / Gazebo | 未跑 |
| git | 未写；工作树保持原状 + 本 worklog |

## 风险与遗留

- 结论均来自静态审阅与既有产物；High 中 L1 语义、反事实对、去重/溯源
  会直接改变 L1/L2 验收结论，修复前不建议开 L2 训练。
- `eval.json` 不自描述，历史数字只能靠目录名/文档；引用前查产物 JSON。
- 3 个重跑代理使用 sonnet（非 DeepSeek Pro），结论已与 flash 代理交叉核对；
  如需严格同模型复核可再跑。

## 更新 2026-10-31 15:10：修复 High 并验证

结论：完成 6/7 条 High 的代码修复与验证；H2（反事实对）需要重采/设计拍板，
未改。未 commit/push。

### 做了什么

- `src/tests/test_trainer.cpp`：NaN 注入偏移 48→52，并断言异常消息含
  “含非有限值”；现在真正覆盖 checkpoint 有限性检查。
- `sim/tools/instruction.py`：模板去掉 `and stop`（`pass the {color} pillar`
  / `pass the ... on your {left|right}`），并新增 `upgrade_legacy()` 升级旧文本；
  `tools/preprocess.py` 在 `task_version=l1.v1` 时调用，旧数据不重采也能纠正。
- `sim/tools/eval_closed_loop.py`：去重键改为 `(gid, 彩色柱位置+颜色)`；
  `eval.json` 记录 `task/window_from/goal_margin/waypoint_radius/geometry_version/
  ckpt_sha256/infer_bin_sha256`，逐集记录 instruction/waypoint/task_version；
  `skipped` 与异常 `failed` 都落盘；补 goal-margin/timeout 有限正数、重复 seed 拒绝。
- `sim/tools/collect_parallel.py`：collection_manifest 写
  `task/window_from/geometry_version`。
- `tools/preprocess.py`：dataset meta 写 `window_from/hold_s/task_version/
  geometry_version` 与逐集 `waypoint_x/y/instruction/target_color`；拒绝同包
  混口径；侧车先校验后写、tmp+rename、无指令时删除旧侧车。
- `src/dataset.cpp`：侧车按 `episode_id + sample_start/count` 绑定 meta episodes[]，
  拒绝重复/缺集/异源侧车。
- `sim/tools/audit_dataset.py`：有 collection_manifest 时只审计登记的
  `(split, scene_id)`；无 manifest 时 walk 剪掉 `.work/.staging/failed/worlds/logs`。
- `src/model.cpp` / `src/checkpoint.cpp` / `text.hpp`：`text_dim` 上限 4096，
  并检查 `flat+text_dim <= INT_MAX`。
- `sim/tools/student.py`：要求 hello 的 `num_classes==4`；构造失败时杀掉子进程。
- `tools/make_synthetic_episode.py`：fixture 指令改为新模板。

### 没做什么

- **H2 反事实对未修**：现有 400 集里 0 对“同柱换色”，需要按几何确定性重采；
  属数据/科学口径决定，未擅自删预注册指标。
- 未重打包完整 L1 400 集（新代码已能把旧指令升级；L2 训练前需重新
  `preprocess` 到新目录，或覆盖 `ds` 前先确认）。
- 未实现 stop-on-goal、决策墙钟计时、`--text-override`、UI 默认目录、
  sm_75 门禁、audit 独立性等 Medium 项。
- 未 commit/push；工作树仍混有本轮之前的 L2 未提交改动。
- 未跑完整 36 集 L1 闭环复评；只跑 1 集冒烟。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `bash tools/check.sh` | PASS（含 21/21 CTest；trainer_demo 89 项） |
| `test_instruction` / `test_eval_closed_loop_args` / `test_eval_summary` / `test_audit_dataset` / `test_student` | PASS |
| l1-full-400-0803 重跑 audit | 400 集/16000 步，15/15 PASS（原目录含 eval `.work`） |
| test split 去重 | 36 个 test seed 全部保留；旧键 (gid,target_color) 只有 34 |
| C++ 侧车绑定 | 3 集真实数据：好侧车加载 960 token；改 episode_id 后被拒 |
| preprocess 指令迁移 | 旧 `fly to the ... and stop` → `pass the ...`；meta 带口径字段 |
| 闭环冒烟 | seed 310111，25 步，`eval.json` 带 task/window_from/hash 等字段 |
| 并行采集冒烟 | 1 集/5 步；manifest 带 task/window_from/geometry_version |
| 文本训练+复评 | 3 split 各 1 集，1 epoch + eval-only，exit 0（指标无意义） |

### 待确认

- 是否重采真正的反事实对（同几何、同柱对、换目标色），还是从 L2 预注册里删掉 C 组。
- 是否现在把完整 L1 400 集重新打包（新指令 + 新 meta）并开 L2 训练。
- 是否继续修 Medium（stop-on-goal、计时、UI、sm_75 门禁等）。

### 风险与遗留

- 修复未提交，仍有丢失风险；工作树混有并发会话的 L2 改动，提交时需逐文件确认。
- 新 `eval.json` 字段只对修复后的运行有效；历史产物仍不自描述。
- 旧 L1 400 集 sidecar 仍是旧指令，直到重新 preprocess；不要拿旧 ds 训 L2。

## 更新 2026-10-31 15:20：基础 VLN 计划与 M0 实验

结论：进行中。已按用户决定删除 C 组、重打包 L1 400 集、写 VLN 路线图；
L2 文本模型 10 轮训练已启动（15:12，ETA 约 15:50），L/I/S 离线与闭环待跑。

### 做了什么

- `docs/experiments/2026-10-31-l1-language-task-design.md`、`docs/TASKS.md`、
  `docs/HANDOFF.md`、`docs/experiments/2026-10-31-l1-data-health.md`：
  删除/标注 C 组（反事实对）移除；C 组实测 0 对同柱换色。
- 重新打包 L1 400 集到
  `$DATA_ROOT/uav_rgb/l1-full-400-0803/ds-l2-20261031`：400 集/16000 步，
  test 36 集，400 行新指令（无 “and stop”），meta 带口径字段。
- `sim/tools/eval_closed_loop.py`：新增 `--text-mode {keep,blank,shuffle}`
  与 `override_instruction`，用于闭环 L/I/S 消融；补测试。
- `docs/experiments/2026-10-31-basic-vln-plan.md`：基础 VLN 定义、差距、
  M0–M4 路线图、2–3 周估算、回退条件。
- 启动 L2 训练：`run-l2-text16-20261031`（10 轮，batch 64，lr 1e-3，
  seed 1234，text-dim 16，`--class-weights auto --drop-collision`）。
- 准备离线 L/I/S 数据集变体：`ds-l2-20261031/variants/{L,I,S}`，
  S 保证与 L 指令不同（颜色分布均衡）。

### 没做什么

- 未跑完 L2 训练/离线 L/I/S/闭环 L/I/S（训练进行中）。
- 未实现 M2 的同色双柱、关系、STOP、模板留出；只在计划里。
- 未 commit/push。
- 未删除旧 `ds`、pilot 数据、eval `.work`（保留为证据；本轮只清临时目录）。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 重打包 | 400 集/16000 步；test 36；400 行新指令；meta 口径字段齐全 |
| 旧指令残留 | 0 行 “and stop”；400 行 “pass the ...” |
| `--text-mode` 单测 | `test_eval_closed_loop_args` PASS |
| L2 训练 | 进行中（epoch 1 dev macro_f1 0.662；10 轮 ETA 约 15:50） |
| 离线/闭环 L/I/S | 未跑 |

### 待确认

- 第 4 点 “强你psus” 是否指 push；未确认前不写 git。
- 是否按 M2（关系+STOP）直接做，还是先做 M1（颜色+STOP）。
- 旧 `ds`/pilot/`.work` 是否删除。

## 更新 2026-10-31 15:40：L1 teacher 未到 goal，L2 训练作废

结论：**阻塞**。当前 L1 400 集不能用于任何 VLN/L2 结论；已停掉 L2 训练。

### 发现

- 逐集读 `oracle.jsonl`：400/400 集 `max_x < max(pillar.x)+1.0`；
  eval goal 约 6.0 m，teacher max_x 中位数 2.58 m、最大 3.76 m。
- planner 分布：astar 320 / plan_action 80；两类都没到 goal。
- 样例 `test/310003`：到达 waypoint (2.0,0.87) 后在 x≈2.2–2.5 原地
  左右打转 30 步，末段 dyaw≈±30°、位移 <0.02 m。
- A* 的 `start→waypoint` 与 `waypoint→goal` 都存在；问题在纯追踪/避障
  在目标柱侧袋里反复转向，不是路径不存在。
- 现有 audit 只查 waypoint 距离，不查 `max_x >= goal`；label_recompute
  复用同一 planner，所以也 PASS。image-only 0/34 的主因是数据没走完路线。
- 现有 `traverse_rate.py` 能查出：astar 320 集 **success=0/320（0.0%）**；
  但 L1 验收流程没有把它当硬门禁。

### 处置

- 终止 `run-l2-text16-20261031`（epoch 2 后），该 run 作废。
- `ds-l2-20261031` 保留只作管线验证，不能用于结论。
- 更新 `docs/experiments/2026-10-31-basic-vln-plan.md`：新增 §2.5、
  M0.1 teacher/waypoint 修复、goal-reach audit、重采后再训练；
  基础 VLN 时间改为 **4–6 周（12 月上中旬）**。

### 没做什么

- 尚未改 teacher/waypoint，也未加 `teacher_goal_reach` audit。
- 未重采 pilot，未跑 L/I/S。
- 未 commit/push。

### 下一步

1. audit 加 `teacher_goal_reach`（planner=astar 必须 `max_x >= eval goal`）。
2. 改 waypoint 为穿过后目标点/双点，或修控制器在 waypoint 的交接；
   50 集 pilot 验证。
3. 重采 400–600 集；再恢复 M0。

## 更新 2026-10-31 17:35：M0.1 goal-stop 已实现并 pilot 通过

结论：M0.1 完成（8 集 + 50 集 pilot 全 16 项 audit PASS）；500 集全量采集中。

### 做了什么

- **根因验证**：CPU 运动学仿真复现 current max_x 中位 2.49 m（与真实数据
  2.58 一致）；直接飞 goal 36/36 到达；side waypoint 作为**最终目标**并停下
  36/36 到达。因此改成 goal-stop，而不是继续 route-choice 的
  waypoint→终点。
- `teacher.py`：`TEACHER_VERSION=v5`，`plan_action_global` 新增
  `stop_at_waypoint/stop_radius`；半径内返回 SAFE_NEUTRAL；默认旧行为不变。
- `scene_gen.py`：L1 写 `task_version=l1.v2`、`stop_radius=0.5`。
- `collect.py`：读取 `stop_radius`；astar 集连续 3 步 SAFE 提前结束；
  **safe 出生集不提前结束**（跑满 40 步，否则 SAFE 类只有 3 步）。
- `collect_parallel.py`：`stopped_on_goal=true` 的提前结束集算成功；
  `XDG_RUNTIME_DIR` 支持环境变量覆盖（根盘紧张时放数据盘）。
- `audit_dataset.py`：新增 `teacher_stop`（最终距离 ≤ stop_radius+0.3 且
  末动作 SAFE）；`label_recompute` 透传 stop 参数。
- `eval_closed_loop.py`：`stop_radius` 非空时 success = 无碰撞 + 最终位姿
  距 waypoint ≤ stop_radius+0.2 + 末动作 SAFE；旧 route-choice 口径保留。
- `instruction.py`：恢复 `fly to the {color} pillar and stop`（goal-stop
  语义与旧措辞一致）；删除已不需要的 `upgrade_legacy`。
- `preprocess.py`：meta 记录 `stop_radius`；不再升级旧指令。
- 测试：`teacher.py` 自检、`test_instruction`、`test_eval_summary`、
  `test_eval_closed_loop_args`、`test_audit_dataset`、`test_collect` 全 PASS。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 8 集 pilot（无 safe） | 131 步；16/16 audit PASS；teacher_stop 0/8 失败 |
| 50 集 pilot（20% safe） | 698 步；16/16 audit PASS；teacher_stop 0/40 失败；平均 14 步/集 |
| 50 集 preprocess | 698 样本；`task_version=l1.v2`、`stop_radius=0.5`；侧车新指令 |
| 8 集 pilot（safe 修复后） | safe 集 40 步、astar 12–20 步；16/16 PASS |
| 50 集 1 epoch 文本训练 | 593 样本/10 s；eval-only exit 0（只证明链路） |
| 500 集全量 | 采集中（base-seed 700000，5 实例） |
| check.sh / CTest | 未跑（Gazebo 占用中，结束后补） |

### 没做什么

- 全量 500 集 audit/preprocess、L2 训练、离线/闭环 L/I/S 未跑。
- 未实现 M2 同色双柱/关系；未改 kTextLen。
- 未 commit/push。

### 待确认

- 500 集完成后是否直接训练 L2（text 10 轮）并跑 36 集 L/I/S。
- 旧 v1 数据（`l1-full-400-0803`）是否删除或归档；当前保留为“错误样本”。
- 第 4 点 push 仍待确认。
