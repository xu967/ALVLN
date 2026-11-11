# 交叉审核任务书（2026-11-11 晚立，给只读的审核 agent）

- 来由：用户 11-11 23:18 与 23:30 要求加派多个 agent、换一个模型家族，从不同角度审我们的代码还有哪些缺陷。
  仓库里的代码都是 Claude 系模型写的，这一轮要的是另一双眼睛的独立判断，不是复述已有结论。
- 只读：不改仓库里的任何文件，不提交，不建分支，不合并，不推送。只写自己的报告与草稿。
- 先读：`CLAUDE.md`；`docs/briefs/2026-11-10-phase12-common.md` 的 §1–§4（项目是什么、口径、路径、环境）；计划 `docs/PLAN_2026-11-10_phase12.md` 的 §0。
  上一轮全仓审查（11-10 下午）的六份报告在 `docs/reviews/2026-11-10/`（`sim-real`、`viz`、`data-algo`、`src`、`arch`、`lit`）：
  先读与自己角度对应的那一份，已经写在里面的问题不重复报，除非你发现它的结论有错或者问题仍然没修。

## 1. 角度（一个 agent 一个角度）

| 代号 | 角度 | 范围 | 重点 |
| --- | --- | --- | --- |
| A | Gazebo 线核心逻辑的正确性 | `sim/tools/collect.py`、`eval_closed_loop.py`、`teacher.py`、`rule_policy.py`、`scene_gen.py`、`scene_course.py`、`student.py`、`gz_io.py` | 坐标与角度的符号约定、时间戳与动作窗口、`--action-latency` 的派发与迟到判定、成功与碰撞判据、失败路径上的状态残留、边界条件 |
| B | 3DGS 线的几何与教师 | `sim/3dgs/*.py`（`planner`、`guard`、`collision`、`dynamics`、`dynamics_sitl`、`renderer`、`rule_oracle`、`collect_3dgs`、`eval_3dgs`） | 坐标系变换与相机模型、单位、占据栅格的膨胀与碰撞判定是否一致、教师与导引的边界情况、到点与碰撞的先后判定 |
| C | C++/CUDA 训练与推理栈 | `src/`（训练、模型、推理、数据读取、检查点）与 `src/include/` | 内存安全与未定义行为、CUDA 核的下标与同步、数值稳定、训练与推理是否同一套预处理、确定性、检查点的兼容 |
| D | 数据链路与统计 | `tools/preprocess.py`、`tools/DATASET_FORMAT.md`、`sim/tools/split_manifest.py`、`sim/tools/audit_dataset.py`、`tools/average_ckpt.py`、`tools/stats_report.py`、各实验目录的 `summarize*.py` 与 `analyze*.py` | 标签与图像的对齐、训练与评测之间的泄漏、分母、配对检验与区间的实现、汇总脚本里会悄悄算错的地方 |
| E | 实验入口与流水线的健壮性 | `experiments/*/run*.sh`、`tools/e1_pipeline.py`、`tools/dagger_gz_pipeline.py`、`tools/dagger_pipeline.py`、`tools/check.sh`、`env.sh` | 失败处理与续跑、锁与竞态、环境变量残留、静默降级、可复现性（冻结代码、seed、指纹）、容器重启后的恢复 |
| F | 测试的质量 | `sim/tools/test_*.py`、`sim/3dgs/test_*.py`、`tools/test_*.py`、`experiments/*/test_*.py`、`src/` 下的测试 | 永远不会失败的测试、断言过弱、关键路径没有测试、门禁（`check.sh`）没跑到的测试；对你认为最要紧的 5 个函数写出缺的用例 |
| G | 仿真与真机的一致性 | `sim/cf_sitl/`、`sim/dynamics/e1_mini.json`、动作协议、相机内外参、时延与电量模型；与 `docs/reviews/2026-11-10/sim-real.md` 对照 | 仿真里成立而真机上不成立的假设；学生输入里有没有混进仿真真值；部署链（int8、输入分辨率、推理协议）上的不一致 |

## 2. 怎么审

- 以代码为准，不以文档为准。每条发现都要落到文件与行号。
- 能验证的就验证：可以跑单个测试文件、写十几行的小脚本喂边界输入（单进程，前面加 `nice -n 10`，两分钟以内）。
  不起 Gazebo，不跑 3DGS 批量渲染，不训练，不跑 `tools/check.sh`，不占机器锁。草稿放 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review-<代号小写>/`。
- 机器上有别的 agent 在各自的工作树（`/root/claude-code-backup-20261028/code/wt/p12-*`）里改代码、有守护进程在跑实验。
  审的是主仓库 `/root/claude-code-backup-20261028/code/uav_rgb` 的 main；别的工作树与实验输出只读，进程不要动。
- 质量优先于数量：每个角度不超过 15 条。不报风格问题、命名问题、「可以更优雅」之类的意见。
- 分清「已复现」与「推断」。推断的要写出它成立需要什么条件、怎样验证。

## 3. 报告

写到 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/review-<代号小写>.md`，中文。结构：

1. 结论一段：这个角度下最要紧的 3 件事。
2. 发现清单，按严重程度排（会让实验结论出错的排最前，其次是会让数据或训练悄悄变坏的，再次是健壮性）。每条写：
   位置（文件:行）；现象；为什么是缺陷，会影响哪些已有的结论或后续实验；怎样复现或验证（命令或最小用例）；建议的最小修法；状态（已复现 / 推断）。
3. 查过但没有发现问题的地方（列出来，免得别人重复查）。
4. 没有查的地方与原因。

写作：不用程度副词（非常、极其、十分、相当），不用套话（总之、综上所述、值得注意的是、让我们），不用破折号作同位语；数字要有出处；没核实的写「未核实」。
最后给主会话的回复不超过 20 行：最要紧的 3 条发现与报告路径。
