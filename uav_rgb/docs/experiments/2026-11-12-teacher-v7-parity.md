# 2026-11-12 教师 v7（过柱阵后目标取线、A* 无路降余量重试）与教师失败集清理：闭环对拍、打包、审计

状态（2026-11-12 01:15）：代码冻结在 `3e03e7e`，闭环对拍 `DONE rc=0`（00:32:13 起跑，00:55:23 结束），0417 打包与审计已跑完。
任务卡 `docs/briefs/2026-11-10-phase12-task-cards.md` W1-7；入口 `experiments/2026-11-12-teacher-v7-parity/`；
原始输出 `$D/exp-2026-11-12-teacher-v7-parity/`；草稿与证据 `$D/tmp/phase12-cards/w1-7/`；报告 `$D/tmp/phase12-cards/reports/w1-7.md`（`$D=/root/rivermind-data/uav_rgb`）。

## 0. 结论

| 验收项 | 门槛 | 实测 | 判定 |
|---|---|---|---|
| `check.sh --python-only`（`3e03e7e`） | 全过 | `rc=0 2026-11-12 00:33:31` | 通过 |
| 教师 v7 闭环，20 个冻结 seed，2 条车道，不加 `--action-latency` | 20/20 | 20/20（撞柱 0、越界 0、超时 0） | 通过 |
| v7 出口区 YAW 步占比 | < 5% | 1/76 = 1.3%，Wilson 95% [0.2%, 7.1%] | 通过 |
| 0417 以 `--exclude-failed-teacher` 打包，报告跳过数 | 约 26 | 跳过 26 集（train 22、test 4、val 0） | 与预期相同 |
| 打包后 train 的 SAFE 帧 | 0 | train 0、val 0、test 0（旧包：train 56、test 1） | 通过 |
| 审计 `label_recompute`（版本分派） | 0 不一致 | 0417 的 v5 数据 0/12000；闭环 v7 臂 0/622；闭环 v5 臂 0/668 | 通过 |

同一批 seed、同参数、同时段的 v5 对照臂（不设门）：20/20，出口区 YAW 17/98 = 17.3%。任务卡的「约 17.7%」是 0417 离线数据的另一种口径（见第 3 节），
同口径重测旧教师得 17.3%，两个数字接近但不是同一个量。

版本号：任务卡与简报按「v6」写，但 `uavrgb.teacher.v6` 已被 forest_e1 档占用（`TEACHER_VERSION_BY_TASK`），本次的新版本是 `uavrgb.teacher.v7`，
入口目录因此叫 `teacher-v7-parity`。costmap 教师版本 `uavrgb.teacher_cost.v1` → `v2`（B2 对它同样生效）。

## 1. 改了什么（教师规则）

- B2（过柱阵后目标取线）：机体 x 大于所有柱的后表面（圆柱 x + radius，方柱 x + size[0]/2）的最大值后，base 目标 (最远柱 x + 2.0, 0) 的 y 取机体当前 y。
  只作用于 base 目标；waypoint 任务、forest_e1 档不变。
- B3（A* 无路降余量重试）：标准余量下 A* 无路时，按 0.30、0.20 m 依次重试，只试严格小于标准余量的档位（x3 标准 0.5：[0.5, 0.3, 0.2]；e1_mini 标准 0.4：[0.4, 0.3, 0.2]）。
  标准余量有路时只调一次 A*，路径与改动前逐位相同。每个余量用自己的可走性缓存钥匙（钥匙含余量）。forest_e1 档不重试。
- 版本：`TEACHER_VERSION` v5 → v7；`TEACHER_RULES_BY_VERSION` 记每个版本的规则档；`set_vehicle(name, task, version=)` 按版本切规则；
  `audit_dataset.py` 的 `label_recompute` 按 manifest 的 `teacher_version` 分派（旧数据按旧规则复算，不跳过）。
- B1：`tools/preprocess.py --exclude-failed-teacher`，整集跳过 oracle 最大 x < 最远柱 x + 1.0 m 的集；跳过数写进 meta `collision_filter.failed_teacher_skipped`。

## 2. 闭环对拍（入口 `run.sh`，守护进程，机器锁）

参数：`eval_closed_loop.py --policy teacher --task base --split test --episodes 10 --steps 80 --sim-timeout 120.0 --episode-timeout 1500.0 --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --corridor-half-width 3.0`，
与 `experiments/2026-11-10-heading-guard/run.sh` 同一套，去掉学生专用参数，不加 `--action-latency`。20 个冻结 seed 拆成前 10 / 后 10 两半（两条车道并行），v7 两半先跑，v5 两半后跑。
v7 臂冻结代码 `3e03e7e`，v5 对照臂冻结代码 main `a714b54`；`DONE`：`rc=0 2026-11-12 00:55:23 四个臂全部跑完`。

| 臂 | manifest 教师版本 | 集数 | 成功 | 撞柱 | 越界 | 超时 | 平均步数 | 出口区步数 | 出口区 YAW | YAW 占比 | Wilson 95% | 出口区 FWD | 出口区 SAFE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v7 | `uavrgb.teacher.v7` | 20 | 20 | 0 | 0 | 0 | 31.1 | 76 | 1 | 1.3% | [0.2%, 7.1%] | 75 | 0 |
| v5 | `uavrgb.teacher.v5` | 20 | 20 | 0 | 0 | 0 | 33.4 | 98 | 17 | 17.3% | [11.1%, 26.0%] | 81 | 0 |

| 臂 | 总步数 | 相邻 L↔R 换向 | 有换向的集 | SAFE 步 | 出口区里有 YAW 的集 |
|---|---|---|---|---|---|
| v7 | 622 | 13 | 3 | 0 | 1（seed 990，1/5） |
| v5 | 668 | 35 | 3 | 0 | 17 |

口径：出口区 = 步的 `pose_before.x` > 最远柱 x；YAW = `action_expert` ∈ {YAW_LEFT, YAW_RIGHT}；占比按步池化。v5 臂 20 集里 17 集在出口区各有 1 个 YAW 步，
v7 臂只有 seed 990 有 1 个。逐 seed 表见 `$D/exp-2026-11-12-teacher-v7-parity/summary.md` 第 3 节。

### 预注册（写在 `run.sh` 头部，起跑前提交）对照

| 预注册 | 内容 | 结果 |
|---|---|---|
| P1 | v7 臂 20/20 | 20/20 |
| P2 | v7 臂出口区 YAW < 5%；预测点估计约 3–4%，< 5% 的概率约 0.8（运动学代理，未核实动力学差异） | 1.3%，低于预测点估计，在预测区间 [2.57%, 5.17%] 之外偏下 |
| P3 | v5 臂出口区 YAW 占比高于 v7 臂（参照，不设门） | 17.3% 对 1.3% |

出口区只有约 80 步，1.3% 的点估计对应 1 个 YAW 步，Wilson 上界 7.1% 高于 5%：门槛判的是点估计，一次评测不能排除真实占比在 5% 以上。
运动学代理的 200 次重复给出 v7 3.65%、v5 17.77%，闭环结果与代理在方向和数量级上一致，代理的数字不替代闭环。

### 墙钟与计时健康度（旧协议，不带 `--action-latency`，按公共任务书要求记录）

`experiments/2026-11-10-long-course/summarize.py --timing`，参照值：每步生效时长 0.871 s、转角 26.1°，标记阈值 3%。

| 臂 | 集数 | 生效时长中位 s/步 | 对 0.871 | 稳态前飞位移 m/步 | 转角 °/步 | 对 26.1° | 标记 |
|---|---|---|---|---|---|---|---|
| v7 | 20 | 0.888 | +1.9% | 0.268 | 27.2 | +4.3% | 转角超 3% |
| v5 | 20 | 0.887 | +1.8% | 0.270 | 27.0 | +3.4% | 转角超 3% |

两个臂的生效时长都在阈值内；转角两个臂都超 3%，超出幅度相近（+4.3% 对 +3.4%），配对比较不受影响，但本次的绝对转角与 11-10 的参照值不能直接比。
容器负载（`run.log` 里的 cgroup 记录，配额 12 核）：v7 两半的窗口里容器平均 11.25 核、`nr_throttled` 增加约 4.7 千次；v5 两半的窗口平均 11.0 核、增加约 3.7 千次。
两个臂都在接近配额上限的负载下跑（11.25 核对 11.0 核）；教师不用推理，闭环动作时长由上表的生效时长反映。

## 3. 离线与代理证据（起跑前，写入预注册的依据；都不是 Gazebo 闭环）

### 3.1 0417 的 12000 步同状态重标（`offline_0417.py`，证据 `evidence/offline_0417.json`）

旧轨迹的状态不动，只换规则。记录标签与 v5 规则重算 0 步不同（旧数据按旧规则可复算）。

| 区域 | 步数 | 记录/v5 YAW 占比 | v7 重标 YAW 占比 |
|---|---|---|---|
| 全部 | 12000 | 19.99% | 29.11% |
| 最远柱后表面之后、成功线之前（exit_back_to_line） | 1170 | 17.78% | 34.79% |
| 其中机头偏离 +x ≤ 10° | 515 | 39.22% | 0.00% |
| 其中偏离 10–25° | 226 | 1.77% | 18.58% |
| 其中偏离 > 25° | 429 | 0.47% | 85.08% |

读法：任务卡的 17.7% 与这里的 17.78%（208/1170）是相近的两个口径（任务卡：data-algo 评审，train、视锥内无柱，175/987），没有逐帧对齐。
同状态重标预测不了闭环：机头已对准的状态上 v7 的 YAW 从 39.22% 降到 0.00%（B2 的直接效果）；机头偏离超过容差的状态上 v7 要把机头转回目标线，YAW 上升。
旧轨迹的机头偏差来自 v5 规则下出口区向 y=0 转向，v7 闭环轨迹没有这个偏差，所以 34.79% 不是 v7 闭环的预测。

A* 无路步（v5 规则，标准余量）：544 步（112 集），v7 重试后仍无路 100 步；这 544 步上 SAFE 标签 229 → 165，标签分布 v5 {L 78, FWD 161, R 76, SAFE 229}，v7 {L 147, R 116, FWD 116, SAFE 165}。
与 data-algo 评审（`docs/reviews/2026-11-10/data-algo.md` A4）的数字对口径（草稿 `reconcile_pathnone.py`，证据 `evidence/reconcile_pathnone.txt`）：
544 步 = 非 collision 364 步 + collision 180 步（16 集）。非 collision 的 364 步与评审的 364 相同；0.30 m 重试有路 293、0.30 或 0.20 m 累计 349，与评审的离线试算（293、349）相同，重试后仍无路 15 步。
collision 步 180 步里重试后有路 95、仍无路 85（15 + 85 = 100）。集数这里是 112，评审写 105/300（口径未对齐，未核实）。
相邻 L↔R 换向：记录序列 138 次（82 集），v7 重标序列 335 次（198 集）；闭环里 v7 13 次、v5 35 次（上表），同状态重标不代表闭环。
单次 `plan_action_global` 在 v5 无路步上的 CPU 时间（缓存就位的第二次调用）：v5 中位 7.83 ms、p95 16.11 ms；v7 中位 5.43 ms、p95 19.0 ms、最大 26.76 ms（v7 最多多跑 2 次 A*）。
B3 重试余量的缓存是冷的时（`--action-latency` 的每集预热只填标准余量那一份），第一次触发的单步耗时：40 个场景中位 16.6 ms、p95 30.0 ms、最大 30.2 ms
（草稿 `cold_retry_probe.py`，证据 `evidence/cold_retry_probe.txt`）。E1 的柱阵栅格小，这个量级不影响 0.9 s 的步长；K=3 长课程的栅格大，未测，未核实。

### 3.2 运动学代理（`kinematic_preview.py`，不是 Gazebo 动力学）

教师规则 + 运动基元 + 噪声，20 个冻结 seed × 10 次重复：v5 出口区 YAW 17.77%（161/906），v7 3.65%（30/821，Wilson 95% [2.57%, 5.17%]），两个臂都 200/200 成功。
按「每个 seed 随机抽 1 次、池化 20 集」模拟单次评测，v7 出口区占比 < 5% 的概率约 79.7%，v5 为 0。这是预注册 P2 的依据，实测 1.3% 落在代理区间下方。

## 4. 0417 打包与审计

### 4.1 `--exclude-failed-teacher` 打包（acceptance 3；`pack_0417_exclude.sh`，输出 `$D/tmp/phase12-cards/w1-7/pack-0417-exclude/`）

命令：`preprocess.py --drop-collision --exclude-failed-teacher` 读 `$D/e1-pipeline-20261107-0417/collection`（300 集）；墙钟 21.7 s，rc=0。

| | 集数 | 样本 | train 类计数 [SAFE, FWD, L, R] | val | test |
|---|---|---|---|---|---|
| 旧包（只 `--drop-collision`） | 300 | 11818 | [56, 7944, 1008, 1024] | [0, 583, 77, 60] | [1, 845, 107, 113] |
| 新包（再加 `--exclude-failed-teacher`） | 274 | 10960 | [0, 7519, 895, 906] | [0, 583, 77, 60] | [0, 743, 86, 91] |

跳过 26 集：train 22、test 4、val 0；集号 3103、3105、3116、3120、3127、3131、3134、3135、3142、3149、3150、3172、3173、3202、3210、3228、3244、3272、3285、3288、3290、3295、3321、3326、3331、3346。
被跳过集共 858 个保留样本；旧包丢弃的 182 个碰撞步也全在这 26 集里（858 + 182 = 26 × 40），新包 `collision: dropped 0`。
meta `collision_filter`：`exclude_failed_teacher: true`，`failed_teacher_skipped: 26`，`failed_teacher_goal_margin_m: 1.0`。

默认不带该开关时，新旧 `preprocess.py` 在 20 集真实子集上逐字节相同（`evidence/default_off_real_subset.txt`：aux、meta、像素与标签区、头 44 字节 sha256 全相同；旧 = main，新 = `3e03e7e`）。

### 4.2 审计（`audit_dataset.py`，`3e03e7e` 的版本分派）

| 数据 | 步数 | `label_recompute` | 其他变化 |
|---|---|---|---|
| 0417 全部 300 集（v5 数据，按 v5 规则复算） | 12000 | PASS，0 不一致 | `class_share` FAIL（SAFE 占比低于 5%）、`safe_collision` FAIL（172/229 个 SAFE 步带 collision），与旧代码的审计一致 |
| 0417 去掉被跳过的 26 集（符号链接视图 + `collection_manifest.json`） | 10960 | PASS，0 不一致 | `safe_collision` 变为 SKIP（审计集没有 SAFE 步）；`class_share` 仍 FAIL（SAFE_NEUTRAL 在三个 split 里都没有样本） |
| 闭环 v7 臂 20 集（`scene_gen` 重新生成 `scene.json`，`geometry_gid` 与 manifest 逐集相同） | 622 | PASS，0 不一致（按 v7 规则） | 主仓库的审计对同一批数据 SKIP（教师版本不是 v5，622 步不复算） |
| 闭环 v5 臂 20 集（同上） | 668 | PASS，0 不一致（按 v5 规则） | 主仓库的审计同样 PASS 0/668 |

审计没有 `--exclude-failed-teacher`，上表第二行用的是只含保留集的视图。去掉失败集后 SAFE_NEUTRAL 样本为 0，`class_share`（每类 ≥ 5%）不会通过，
这是 B1 的直接后果，不是新引入的缺陷；要不要为 SAFE 类放宽该检查、训练是否保留 SAFE 头，由主会话定。

## 5. 复现

```bash
D=/root/rivermind-data/uav_rgb; OUT=$D/exp-2026-11-12-teacher-v7-parity
cd /root/claude-code-backup-20261028/code/wt/p12-w1-7/experiments/2026-11-12-teacher-v7-parity
mkdir -p $OUT && cp run.sh $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &   # 阻塞式机器锁，约 25 分钟加排队
/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python summarize.py $OUT      # 解释器用主仓库 .venv-tools；随时可汇总
kill -- -$(cat $OUT/run.pid)                  # 停（只按进程组）
```

离线证据：`offline_0417.py --data <0417 collection> --out <json>`；代理：`kinematic_preview.py`；打包：`preprocess.py --drop-collision --exclude-failed-teacher ...`。

## 6. 没验证的部分

- 没有用学生评测验证 v7 标签训练出的学生：卡只要求教师闭环。出口区 YAW 标签占比下降是否改善学生出口区行为，未核实。
- 闭环对拍只覆盖 e1_mini 的 base 档 20 个 seed；forest_e1 档保持旧规则（v6），没有对它做 B2、B3 的闭环验证。
- 没有跑带 `--action-latency` 的闭环：B3 重试的冷缓存耗时只做了离线探针（第 3.1 节）。
- x3 载具（标准余量 0.5 m）的 B2、B3 只有单测，没有闭环。
- 同一场景集上 v7 打出的新数据没有采集（卡未要求）；v7 标签的真实采集集由主会话决定何时采。
