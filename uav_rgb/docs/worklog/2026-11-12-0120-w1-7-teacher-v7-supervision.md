# W1-7：监督清理（教师失败集不进训练、过柱阵后目标取线、A* 无路降余量重试）

- 时间：2026-11-11 23:37 起读规格、写代码与测试；2026-11-12 00:26:30 起检查与闭环守护进程（闭环 00:32:13–00:55:23）；01:02–01:20 打包、审计、补充证据与文档
- 范围：`sim/tools/teacher.py`、`sim/tools/audit_dataset.py`、`sim/tools/collect.py`、`tools/preprocess.py`、`tools/relabel_costmap.py`、`tools/export_costmap_soft.py`、各自的测试、`experiments/2026-11-12-teacher-v7-parity/`
- 分支：`p12-w1-7`（基于 main `a714b54`）；代码冻结在 `3e03e7e`，之后只有 docs 提交
- 结论：三项验收都通过。教师 v7 在 20 个冻结 seed 上 20/20，出口区 YAW 步占比 1.3%（门槛 < 5%，v5 对照臂同口径 17.3%）；0417 以 `--exclude-failed-teacher` 打包跳过 26 集、三个 split 的 SAFE 帧为 0；`check.sh --python-only` rc=0。新版本号是 v7，不是任务卡写的 v6（v6 已被 forest_e1 档占用）。

## 做了什么

- B2：`teacher.py` 的 `_global_path` 在机体 x 大于所有柱后表面的最大值后，把 base 目标 (最远柱 x + 2.0, 0) 的 y 换成当前 y。触发线用 `_pillars_back_x`（圆柱 x + radius，方柱 x + size[0]/2）。waypoint 任务不受影响。costmap 教师同一个开关（`_costmap_goal` 带机体位置时）。
- B3：`_astar_retry` 在标准余量无路时按 `ASTAR_RETRY_CLEARANCES` 里严格小于标准余量的档位依次重试（x3 [0.5, 0.3, 0.2]，e1_mini [0.4, 0.3, 0.2]，forest_e1 不重试）。重试各用自己的可走性缓存钥匙（钥匙含余量），标准余量有路时只调一次 A*，路径与改动前逐位相同，两点都有测试。
- 版本：`TEACHER_VERSION` v5 → v7（v6 被 `TEACHER_VERSION_BY_TASK["forest_e1"]` 占用）；`TEACHER_COST_VERSION` v1 → v2；`TEACHER_RULES_BY_VERSION` 记每个版本的规则档，`set_vehicle(name, task, version=)` 按版本切规则，未知版本报错，不回落。
- 审计：`audit_dataset.py` 的 `label_recompute` 按 manifest 的 `teacher_version` 分派（`recomputable_versions`），v5 数据按 v5 规则复算，不再因升版整集跳过；`campus_checks` 的版本判定同步认 `recomputable_versions("base")`。
- 采集与工具：`collect.py` 在 costmap 采集里按 `TEACHER_COST_VERSION` 切规则档（采集时生效的规则与审计按 manifest 版本复算的一致）；`relabel_costmap.py` 同样；`export_costmap_soft.py` 接受 cost v1 与 v2，一次导出只允许一个版本。
- B1：`tools/preprocess.py --exclude-failed-teacher`：整集跳过 oracle 最大 x < 最远柱 x + 1.0 m 的集，waypoint 任务不判，跳过数写进 meta `collision_filter.failed_teacher_skipped`；缺省关闭时输出逐字节不变。
- 测试（先红后绿，真实输出在草稿 `logs/`）：`test_teacher_v7.py` 31 项（红 failures=11 errors=14，绿 OK）、`test_preprocess_failed_teacher.py` 10 项（红 failures=8 errors=1，绿 OK）、`test_audit_teacher_dispatch.py`（红 `set_vehicle() got an unexpected keyword argument 'version'`，绿 PASS）、`test_export_costmap_soft.py` 6 项（红 failures=2，绿 OK）、`test_collect.py`（红 `FAIL: 开关: costmap 采集：生效规则 = cost 版本的规则`，绿 110 项通过）、随升版更新的 `test_teacher_costmap.py`（55 项，含 0417 300 集 12000 帧标签不一致 0）与 `test_latency_analysis.py`（红 failures=1，绿 21 项）。`teacher.py` 自检加了 v7 一条。
- 闭环对拍入口 `experiments/2026-11-12-teacher-v7-parity/`（`run.sh` 预注册写在头部，起跑前提交；`summarize.py`、`offline_0417.py`、`kinematic_preview.py`）。守护进程 PGID 252627，输出 `$D/exp-2026-11-12-teacher-v7-parity/`，`DONE` = `rc=0 2026-11-12 00:55:23`。
- 补充证据（草稿 `evidence/`）：0417 同状态重标、运动学代理、墙钟与计时健康度、0417 打包、四组审计（0417 全部、去掉失败集、闭环 v7 臂、闭环 v5 臂）、B3 冷缓存耗时探针、新旧 `preprocess.py` 在 20 集真实子集上逐字节相同。
- 提交：`009bca2` fix(teacher)、`f646c85` test、`1dad4aa` feat(preprocess)、`3e03e7e` feat(experiments)，另有本文所在的 docs 提交。
- 实验记录：`docs/experiments/2026-11-12-teacher-v7-parity.md`；报告：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/w1-7.md`。

## 没做什么

- 没改 forest_e1 档的规则（v6 = B2、B3 关）。原因：B2、B3 只在 base 档 20 个冻结 seed 上做过闭环验证，森林档标签要逐位不变。
- 没改 `make_counterfactual.py`、`experiments/2026-11-10-action-latency/profile_late_steps.py`。原因：不在卡内；两者合并后的行为变化记在「风险与遗留」。
- 没给 `audit_dataset.py` 加 `--exclude-failed-teacher`。原因：不在卡内；去掉失败集后的审计用符号链接视图做的（第 4.2 节）。
- 没采 v7 教师的新数据，也没用学生评测验证 v7 标签。原因：卡只要求教师闭环 20 个 seed 与打包；v7 真实采集集（data-algo A4 的「新采 100 集没到线 ≤ 1%」）要 Gazebo 重实验，由主会话排。
- 没做 x3 载具与 forest_e1 档的闭环。原因：卡的验收是 e1_mini base 档的 20 个冻结 seed；x3 与 forest 只有单测。
- 没跑带 `--action-latency` 的闭环。原因：卡明确不加。B3 冷缓存首次触发的耗时只做了离线探针。
- 没把 B3 的触发次数写进闭环输出。原因：`eval_closed_loop.py` 不记 path_none；离线次数在实验记录第 3.1 节。
- 没调用 data-algo 的 `a10_teacher_flipflop.py`（仓库里没有这个脚本，只有评审日志）。原因：换向次数按它的口径（本步与上一步都非碰撞、都是 YAW、方向不同）在 `offline_0417.py`、`summarize.py` 里自算。
- 没改 `progress.md`、`CLAUDE.md`，没合并、没推送，没删除任何文件、分支、工作树。原因：交办。

## 待确认

- 新版本号用 v7（v6 已被 forest_e1 档占用）：任务书、训练配置里写「v6」的地方要改成 v7；主会话确认。
- B1 之后 train 的 SAFE_NEUTRAL 为 0（旧包 56），`audit_dataset.py` 的 `class_share`（每类 ≥ 5%）对 v7 数据不会通过；是否放宽该检查、学生是否保留 SAFE 头、`--class-weights` 怎么设，由主会话定。
- 审计要不要同步加 `--exclude-failed-teacher`，让审计与打包同口径。
- v5 旧数据与 v7 新数据并集打包时，`provenance_uniform` 会判不一致；DAgger 并集流程怎么处理（只用 v7 新采、还是旧数据重标），由主会话定。A* 教师没有现成的整集重标工具（costmap 有 `relabel_costmap.py`）。
- B3 的重试余量降到 0.20 m 时空隙只剩约 0.10 m（机身半宽 0.102 m）：20 个 seed、622 步没有撞柱，但闭环里没记录触发次数，更大样本（新采 100 集）里是否擦柱未核实。

## 验证到哪一步

| 项 | 命令 / 条件 | 结果 |
| --- | --- | --- |
| 新增与更新的单测（先红后绿） | 各测试文件单进程 `nice -n 10`，`PYTHONDONTWRITEBYTECODE=1`，输出在草稿 `logs/red_*.txt`、`logs/green_*.txt` | 见上「测试」；绿全部通过 |
| `teacher.py` 自检 | `python sim/tools/teacher.py` | `teacher: 全部通过`（含 v7 一条） |
| `check.sh --python-only` | 守护脚本排 `check.lock`，HEAD `3e03e7e`，`logs/green_check_python_only.log` | `rc=0 2026-11-12 00:33:31`，`PASS（有 SKIP）` |
| 闭环 20 个冻结 seed（v7 臂） | `run.sh`，2 条车道，不加 `--action-latency`，守护进程，机器锁 | 20/20；出口区 YAW 1/76 = 1.3% |
| 闭环同时段 v5 对照臂（main `a714b54`） | 同上 | 20/20；出口区 YAW 17/98 = 17.3% |
| 计时健康度（旧协议） | `long-course/summarize.py --timing` | v7 生效时长 0.888 s（+1.9%）、转角 27.2°（+4.3%，超 3% 标记）；v5 0.887 s（+1.8%）、27.0°（+3.4%，超 3% 标记） |
| 0417 打包 | `preprocess.py --drop-collision --exclude-failed-teacher`，300 集 | 跳过 26 集（train 22、test 4），样本 11818 → 10960，三个 split 的 SAFE 为 0；`collision: dropped 0` |
| 默认关闭逐位不变（真实数据） | main 的 `preprocess.py` 对 W1-7 的，20 集（12 train + 4 val + 4 test），`--drop-collision` | aux、meta、像素与标签区、头 44 字节 sha256 全相同 |
| 审计 0417 全部 300 集（v5） | `audit_dataset.py --dataset .../collection`，`3e03e7e` | `label_recompute` 0/12000；`class_share`、`safe_collision` FAIL 与旧审计相同 |
| 审计去掉失败集后的 0417（符号链接视图） | 同上 | `label_recompute` 0/10960；`safe_collision` SKIP；`class_share` FAIL（SAFE 无样本） |
| 审计闭环 v7 臂 / v5 臂（`scene_gen` 重新生成 `scene.json`，`geometry_gid` 逐集相同） | 同上；主仓库审计作对照 | v7 臂 0/622（主仓库 SKIP）；v5 臂 0/668（主仓库 0/668） |
| 离线同状态重标 / 运动学代理 | `offline_0417.py`、`kinematic_preview.py` | 实验记录第 3 节；代理预测 v7 3.65%，闭环实测 1.3% |
| 未验证 | 学生评测、v7 新采集、x3 与 forest 闭环、`--action-latency` 闭环、完整 `check.sh` | 见「没做什么」 |

## 风险与遗留

- 一次评测的出口区只有 76 步，1.3% 对应 1 个 YAW 步，Wilson 95% 上界 7.1%；门槛判点估计，通过不等于真实占比一定在 5% 以下。
- `make_counterfactual.py` 把源 manifest 的 `teacher_version` 原样抄进副本，同时用当前默认规则重算标签；合并后对 v5 数据运行它，会得到按 v7 规则算的标签却标 v5，审计的 `label_recompute` 会报不一致。它是 L1 时期的历史工具，没人在用（未核实）。
- `experiments/2026-11-10-action-latency/profile_late_steps.py:377` 调 `set_vehicle("e1_mini", task)` 不传版本，合并后重跑它会用 v7 规则复算 11-10 的 v5 时期数据。它复制一份冻结教师副本作为证据，重跑前需要传版本。
- B3 的重试余量缓存在 `--action-latency` 的每集预热里没有预热（`warm_free_cells` 只填标准余量）：首次触发的单步耗时 40 个场景中位 16.6 ms、p95 30.0 ms、最大 30.2 ms，比热缓存的 6.2 / 19.4 / 20.1 ms 多约 10 ms，对 E1 的 0.9 s 步长可忽略；K=3 长课程的栅格大，未测，未核实。
- `_FREE_CACHE_MAX_SCENES = 4` 实际按钥匙计数（钥匙含余量与柱子几何）：一个场景最多占 3 个（标准、0.30、0.20），离线批量重标逐集换场景时更容易换出；只影响速度，不影响结果。
- 同状态重标的 v7 YAW 占比（34.79%）不是闭环预测，引用时注意口径；闭环数字用 1.3%。
- 容器负载接近 12 核配额（闭环窗口平均 11.0–11.25 核，`nr_throttled` 持续增加），旧协议的墙钟敏感；计时健康度已记录，两个臂同口径。
