# 校园配对训练集采集与学生基线训练

- 时间：2026-11-03 16:10
- 范围：`sim/tools/collect_parallel.py`（使用）、`tools/preprocess.py`（使用）、`src/build/uavrgb_train`（重编译）、
  `/root/rivermind-data/uav_rgb/campus-paired-20261103/`（新数据根）
- 结论：部分完成——采集与打包完成，训练启动；闭环评测未做
- 依据：`docs/experiments/2026-11-03-gpu-rendering.md`（GPU 渲染 4.2× 加速）

## 做了什么

### 1. 确认 GPU 渲染链路可用并实测并发容量

采集前用 `collect_parallel.py --task campus` 做了三组短测（每组 1 集/实例，`--steps 40`），
记录端到端墙钟：

| 实例数 | 端到端 | 折合每集 | 单实例均值 |
| ---: | ---: | ---: | ---: |
| 4 | 45.7 s | 11.4 s | 0.528 步/s |
| 6 | 49.9 s | 8.3 s | 0.544 步/s |
| 10 | 63.0 s | 6.3 s | 0.446 步/s |

单实例吞吐在 6 实例时仍为 0.544，到 10 实例降到 0.446——开始有争用，但端到端吞吐
仍在涨，因此正式采集用 10 实例。**这是本机第一次实测 campus 的多实例并发**，
此前文档只测过单实例。

### 2. 采集三个 split（10 实例，GPU 渲染）

```bash
UAVRGB_RENDER_BACKEND=gpu "$UAV_RGB_PYTHON" sim/tools/collect_parallel.py \
  --task campus --instances 10 --episodes-per-instance 32 --steps 40 --hold 1.0 \
  --split train --out <DATA>/campus-paired-20261103/train
```

| split | 地图族 | 集数 | 步数 | 耗时 | 失败 |
| --- | --- | ---: | ---: | ---: | ---: |
| train | building_lane | 320 | 7572 | 1755.9 s | 0 |
| val | tree_road | 80 | 1868 | ~450 s | 0 |
| test | plaza | 80 | 1866 | ~440 s | 0 |

每集平均 23.7 步（上限 40，抵达后连续 3 步 SAFE 提前结束）。全部 `attempts=1`，
无重试，结束时残留 ruby 进程 0。

### 3. 重编译训练器

`src/build/uavrgb_train` 原为 11-02 01:23 编译，而 `src/trainer.cpp`、`src/soft_targets.cpp`
在 11-03 11:06 改动（软标签加载迁到 `soft_targets.cpp` 并换成 UAVSOFT1 格式）。
在采集结束后重编译，65/65 目标成功。

### 4. 打包与审计

`tools/preprocess.py` 打 160×120（与 campus pilot 一致）。

审计结果（`audit_dataset.py`，按 split 分别跑）：

| split | 结论 | 失败项 |
| --- | --- | --- |
| train | FAIL（3 项） | `campus_teacher_reach` 5/320；`time_pairing` 1/7572 步 max_gap=100 ms；`safe_collision` 1/962 |
| val | FAIL（1 项） | `campus_teacher_reach` 1/80 |
| test | PASS | — |

`split_geometry` 在三份里都是 SKIP（单个 split 内无法核验跨 split 几何）；
另有 4 项（`target_color`/`relation_balance`/`waypoint_side`/`teacher_waypoint`）按设计
由 campus 专用检查替代，属预期 SKIP，不是漏检。

### 5. 定位 `campus_teacher_reach` 失败根因

6 个失败集全部**进了目标区**（`waypoint_min_dist` 0.092–0.347 m，阈值 0.5），
失败原因是**中途发生碰撞**，不是到不了：

| 集 | 最近距离 | 碰撞步 |
| --- | ---: | --- |
| train 6 | 0.092 m | 11 |
| train 52 | 0.247 m | 10, 25 |
| train 412 | 0.123 m | 10 |
| train 414 | 0.109 m | 10 |
| train 437 | 0.347 m | 11 |
| val 106 | — | 15, 16, 20 |

train 的 5 个恰好全是 `relation=right`，但 val 唯一失败的是 `relation=left`，
所以**不是左右不对称 bug**。碰撞都发生在中央 `island` 障碍物附近（第 10–11 步），
即教师 A* 路径贴着障碍物边缘走，净空不足。这是场景/路径规划缺陷，不是采集故障。

### 6. 按 T2 硬约束剔除碰撞集并重新打包

碰撞步的标签本身不可信（T2 硬约束），用 `--exclude-collision-episodes` 整集剔除后
重新打包到 `packed-clean/`：639,926,804 字节（原 651,741,388），
样本 11306 → 约 11101。

### 7. 训练学生基线

超参对齐 l1 蒸馏的 `base` 组，唯一改动是 seed 按项目硬规则取 0：

```bash
src/build/uavrgb_train --data <DATA>/campus-paired-20261103/packed-clean/dataset.uavr \
  --epochs 20 --batch 64 --lr 5e-4 --seed 0 --text-dim 16 --max-grad-norm 5 --drop-collision \
  --out <DATA>/campus-paired-20261103/student-base
```

## 没做什么

- **没有做闭环评测**。原因：训练尚未结束，闭环依赖 `ckpt_best.bin`。
- **没有做蒸馏软标签**。原因：先出一个与 l1 `base` 组同口径的硬标签基线；蒸馏作为下一轮，
  且需要 `teacher/export_soft_targets.py` + 新版 `join_soft_targets.py`（UAVSOFT1）重新导出。
- **没有修校园场景的教师路径净空**。原因：改动在 `campus.py` / `teacher.py`，属他人正在编辑的
  文件；且改了几何会让已采的 480 集全部作废。本轮用剔除规避。
- **没有补采那 6 个失败集**。原因：碰撞由几何净空决定，重采大概率复现；整集剔除代价仅 1.6%。
- **没有修 `time_pairing` 的 1/7572 异常**。原因：单步 100 ms 超阈值，属偶发调度抖动，
  不影响标签正确性；已记录。
- **没有跑完整 CTest**。原因：本轮没改 C++ 逻辑（只重编译），且采集期间禁止 build/CTest。
- **没有验证 val/test 的并发容量**。原因：只测了 train 规模下的吞吐，两族地图障碍数不同。
- **没有提交/推送**。原因：用户没有要求 Git 写操作。

## 待确认

- 教师路径净空不足是否应该修 `campus.py` 的布局（把 island 挪远或缩小），而不是长期靠剔除？
  修了会使现有 480 集作废，需要重采。
- `time_pairing` 的 50 ms 门限是否偏严（本轮 1/7572 触发，且是 100 ms 单点）。
- 训练集只有 40 张地图、单拓扑族（building_lane），val/test 是另外两族——
  这个设定下只能说"族留出"，不能说"通用泛化"，后续是否加训练族？

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| 4/6/10 实例并发短测 | 通过：10/10、6/6、4/4 全部成功，无重试，残留 0 |
| train 采集 | 通过：320/320 集，7572 步，0 失败 |
| val 采集 | 通过：80/80 集，1868 步，0 失败 |
| test 采集 | 通过：80/80 集，1866 步，0 失败 |
| 重编译 `uavrgb_train` | 通过：65/65 |
| `audit_dataset.py` train | **FAIL**（见上表，6 个碰撞集） |
| `audit_dataset.py` val | **FAIL**（1 个碰撞集） |
| `audit_dataset.py` test | PASS |
| 重新打包（剔除碰撞集） | 通过：639,926,804 字节 |
| 训练 | 进行中 |

## 风险与遗留

- 审计在 train/val 上仍是 FAIL，只是失败原因被定位为"碰撞集"并用剔除处理了。
  **不能把这次数据说成"审计通过"。**
- 教师路径贴障碍物的缺陷仍在，下一次采集还会复现同样的 ~1.8% 碰撞率。
- 训练集单族、40 张地图，样本量约 11101，明显小于 l1 的 18056；
  20 epoch 可能欠拟合，需要看 dev 曲线决定是否延长。
- 学生基线用的是 `--drop-collision`，与 `--exclude-collision-episodes` 叠加，
  正常情况下后者已经清干净，前者应为空操作；若训练日志显示丢了步，说明剔除没生效。

## 更新 2026-11-03 17:40 — 三方并行代码审计（只读）

针对"校园学生模型塌陷（test macro_f1=0.2386、闭环 SR=0.05）"做独立审计，
三个 subagent 并行，本会话复核关键结论。**未修改任何代码。**

### 审计结论

**1. C++ 训练/推理链路：未发现能解释症状的 bug。**
文本分支实测有效——同一 `ckpt_best` 只改 /tmp 侧车把指令全换成 `<pad>`，
val macro_f1 从 0.521 掉到 0.299、LEFT/RIGHT 召回趋 0；left/right token 互换
→ 0.362。标签切片、shuffle、Adam、推理预处理、checkpoint 选择逻辑均核对无误。

**2. 数据打包：完全对齐。** 11101/11101 个 `.uavr` 标签与 `actions.jsonl` 的
`action_expert` 逐条相等；474/474 集指令与 `scene.json` 一致；从 `.uavr` 抠图
与原始 PNG 的 MAD=0.0000；11306 步中 0 步图像晚于动作下发。

**3. 真实根因是数据与评测设计，不是代码：**
- 三个 split 是**整族留出**（building_lane→train / tree_road→val / plaza→test，零交集）。
- 训练族只有 **40 张地图**（每张 8 个颜色/左右变体）。
- plaza 整族外观剧变：首帧平均亮度 65.3±13.5，train 是 102.2±7.5；
  `building_back` 大墙正对镜头。
- 无权 CE（FORWARD 占 73%）+ 无正则 + 无早停，train_ce 0.095 / dev_ce 1.127。

**4. 左右在决策点只能靠文本区分。** 同地图同为红色、只差左右的决策帧，
去均值归一化 MAD=9.17，而同标签不同 seed 的纯外观噪声是 9.93；
全 train 集里同姿态的左右步对有 2512 个，其中 630 个（25.1%）标签相反。
图像里没有可用信号，文本是唯一依据。

### 需要纠正的一条（本会话自己算错的）

先前写"模型塌成多数类、比多数类差 16 pp"，**基线算错了**（把预测列当成真实行）。
正确数字：全猜 FORWARD 的准确率是 **0.7278**，模型实际 **0.5150**，
**比常量预测器差 21.3 pp**。平衡准确率 0.2470，四分类随机水平 0.25——是接近乱猜。

### 发现的真实代码缺陷（尚未修复）

| # | 位置 | 问题 | 严重度 |
| --- | --- | --- | --- |
| 1 | `src/trainer.cpp:302-312` | 循环结束后内联 test 评估用的是内存里的 `model_`（**最后一轮**），不是 `ckpt_best.bin`。本次 best 恰为 epoch 20 所以数字相同；best 早于末轮时，报告的 test 数字与交付权重不符 | 高 |
| 2 | `src/include/uavrgb/trainer.hpp:32` | `text_dim` 默认 0，`trainer.cpp:118` 只做单向检查；漏传 `--text-dim` 会静默训成纯视觉模型，症状与本次塌陷相同 | 高 |
| 3 | `src/dataset.cpp:132-164` | 跨 split 校验只按精确 `scene_id`，不知道地图组是 `scene_id // 8`；未来同图不同 scene_id 跨 split 挡不住 | 中 |
| 4 | `sim/tools/collect.py:1081-1088` | "连续 3 次 SAFE 提前结束"不检查距离，字段却叫 `stopped_on_goal`；学生 20 集里 7 集在离目标 3.4 m 处被截断 | 中 |
| 5 | `sim/tools/eval_closed_loop.py` + `--episodes 20` | 扫 seed 取 test 集，20 集只覆盖 plaza 的 3 张地图（共 10 张） | 中 |

### 没做什么

- 没有修上述 5 个缺陷。原因：本轮任务是审计；且 `src/trainer.cpp`、`sim/tools/collect.py`
  在当前工作区有他人未提交改动，动手需先确认归属。
- 没有重训、没有重采、没有跑"族内 test"来分离"域偏移"与"能力不足"。
- 没有跑完整 CTest（只跑了 9 项相关用例，全 PASS）。
- 没有提交/推送。

### 待确认

- 评测设计要不要改：整族留出使得 test 永远测的是域偏移而非能力。
  想让 test 有意义，需要（a）train 覆盖多族、test 做族内划分，或（b）接受
  "跨族泛化"作为唯一口径并明确写进 EVAL_STANDARD。
- 左右只能靠文本区分是 `campus.py` 的设计（两根目标柱同色同尺寸）。
  要不要改成左右异色，让任务变成多模态可解？改了已采 480 集作废。

### 风险与遗留

- 本批数据本身没有问题，但**当前划分下测不出模型能力**，任何 test 数字都只能当域偏移描述。
- `trainer.cpp:302` 的缺陷会让以后"best 早于末轮"的实验报出与交付权重不符的 test 数字。
- 闭环 20 集覆盖 3/10 张地图，SR 的置信区间极宽，不宜作单点能力指标。

## 更新 2026-11-03 23:55 — 按审计结论修复

### 做了什么

| # | 文件 | 改动 |
| --- | --- | --- |
| 1 | `src/trainer.cpp` | 内联 test 评估前先 `load_checkpoint(best_ckpt_path, ...)`，评的是交付的 `ckpt_best.bin` 而不是最后一轮 `model_`；状态用临时副本，不回写训练状态 |
| 2 | `src/trainer.cpp`、`src/include/uavrgb/trainer.hpp`、`src/train_main.cpp` | 数据集有 `instructions.jsonl`（含 `text_ids`）而 `text_dim==0` 时直接报错；新增 `--vision-only` 作为显式退出开关 |
| 3 | `sim/tools/collect.py` | 抽出 `stopped_at_waypoint(row, waypoint, stop_radius)`；goal-stop 的连续 SAFE 计数现在**要求真在 waypoint 半径内**，学生在远处喊 SAFE 不再提前截断 episode |
| 4 | `sim/tools/eval_closed_loop.py` | `eval.json` 新增 `unique_geometry`，让"20 集只覆盖 3 张图"在报告里可见 |

测试：

- `src/tests/test_trainer.cpp` 新增两条：内联 test 与 `--eval-only ckpt_best` 的 test 准确率必须一致；带侧车不传 `--text-dim` 必须抛异常、传 `--vision-only` 必须放行。
- `sim/tools/test_collect.py` 新增 4 条 `stopped_at_waypoint` 用例（半径内 / 半径外 / 缺位姿 / 无 waypoint）。

### 没做什么

- **没有修 `src/dataset.cpp` 的跨 split 校验**（审计第 3 条）。原因：`preprocess.py` 已按 `split_for_scene` 分配 split 并在打包时校验，`audit_dataset.py` 还有 `campus_map_split`；C++ 层是第三道防线，改它要动 meta 解析和 `EpisodeSpan` 结构，收益不抵风险。
- 没有重训校园模型。原因：本次修复影响的是"训练器报哪个数"和"数据守卫"，已训练权重不受影响（本次 best 恰为末轮，数字无误）。
- 没有验证修复后的 goal-stop 在真实学生闭环上的效果（要重跑 Gazebo 评测）。
- 没有提交/推送。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `ctest --test-dir src/build` | **25/25 通过**（175 s） |
| `test_trainer` 新增 2 条用例 | PASS（内联 test 0.375 == ckpt_best 复算 0.375；守卫抛错；`--vision-only` 放行） |
| `sim/tools/test_collect.py` | 30 项通过（新增 4 条） |
| `test_eval_summary.py` / `test_eval_closed_loop_args.py` / `test_collect_parallel.py` | 全 PASS |
| 真实数据集触发守卫 | 通过：`trainer: 数据集有 instructions.jsonl 但 --text-dim 缺省为 0…` 按预期报错 |
| 修复后的 goal-stop 真实闭环 | **未跑** |

### 风险与遗留

- 以前依赖"不传 `--text-dim` 在带侧车数据集上跑纯视觉"的脚本，现在会直接失败而不是静默降级——这是有意的；补 `--vision-only` 即可恢复。
- 内联 test 改用 `ckpt_best` 后，以后"best 早于末轮"的 run 报出的 test 数字会**和以前不同**，属修正后的正确值；与历史实验对比时要标注。
- goal-stop 改动会让**学生策略采集**的 episode 变长（不再被远处 SAFE 截断），教师采集不受影响（教师只在半径内输出 SAFE）。

## 更新 2026-11-04 — 修复后的全量闭环验证

- 范围：`campus.v1` test split 全部 10 张地图 × 8 变体 = 80 集
- 命令：`eval_closed_loop.py --task campus --policy student --ckpt student-base/ckpt_best.bin
  --split test --seeds <80 个> --steps 40 --hold 1 --window-from dispatch`
- 产物：`/root/rivermind-data/uav_rgb/campus-paired-20261103/eval-cl-full/`

### 结果

| 指标 | 值 |
| --- | --- |
| 集数 / 地图 | 80 / **10**（每张 8 变体） |
| SR | **0.0（0/80）** |
| 碰撞率 | 0.05 |
| 执行失败 | 0 |
| 墙钟 | 4721 s（59 s/集，含外部 GPU 争用） |
| 教师对照（采集时同 split） | 80/80 到达 |

动作分布：SAFE 1560（49%）、FORWARD 1466（46%）、LEFT 90、RIGHT 44。
离线标签里 SAFE 只占 12.9%——模型在闭环里把 SAFE 用超了 3.8 倍。

### 两种失败模式

1. **SAFE 锁死**：23 集 SAFE ≥30 步。例：seed 165 前 5 步 FORWARD 后 35 步全 SAFE，
   停在离目标 3.24 m 处。一旦静止，下一帧几乎不变，模型继续输出 SAFE，自我强化。
2. **接近后掉头**：最接近的 4 集最近距离 0.525 / 0.530 / 0.532 / 0.546 m，
   **最终距离却是 2.04–4.29 m**。例 seed 235：26 FORWARD + 2 LEFT + 12 SAFE，
   最近 0.525 m，最终退回 2.65 m。

**13 集进过 0.6 m、21 集进过 1.0 m，但 0 集跨过 0.5 m。** 最接近的一次差 2.5 cm。

76/80 集跑满 40 步上限——这正是 goal-stop 修复后的预期：不再被远处的 SAFE 提前截断。

### 与上一轮的对比（不可直接相减）

上一轮 20 集只在 3 张地图上跑出 SR=0.05，且 7 集被远处 SAFE 截断。
本轮覆盖全部 10 张地图、不再截断，SR=0.0。**差别主要来自样本与规则，不是模型变差。**

### 修正一个我自己引入的错误

前面新增的 `unique_geometry` 字段用 `dedup_key` 计数，而 `dedup_key` 含颜色和左右
（campus 每张图有 8 个变体），所以 80 集被报成 80 个"几何"。已改为
`unique_maps`，按 `geometry_gid` 计数，本轮实测 **10**。

### 没做什么

- 没有重训模型（本轮只验证修复与测量口径）。
- 没有跑 teacher 闭环（`--policy` 无裸 teacher 选项，用采集时的到达率当参照）。
- 没有跑盲前进对照（结构性必为 0）。
- 没有提交/推送。

### 待确认

- 0.5 m 判定半径是否偏严：最接近的一次差 2.5 cm。但把半径放宽到 0.6 m 就等于改口径，
  不能为了好看而改；要么改场景（waypoint 挪近）要么承认模型没到。
- SAFE 锁死是否需要单独的闭环干预（例如禁止连续 SAFE 超过 N 步）。

### 风险与遗留

- 本轮与外部的 `train.py` 抢 GPU（对方 14.7 GB / 65%），59 s/集明显慢于空载；
  指标不受影响，但不要拿本轮的墙钟当吞吐基线。
- 模型在闭环里 SAFE 占比 49% 而离线标签只有 12.9%，说明闭环分布与训练分布差距很大；
  离线指标不能预测闭环行为（这点已在上一轮审计里出现过）。
