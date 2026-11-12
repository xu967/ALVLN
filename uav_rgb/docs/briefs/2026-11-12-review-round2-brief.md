# 第二轮交叉审核任务书（2026-11-12 07:54 立，给只读的审核 agent）

- 来由：用户 11-12 07:46 要求再加派 agent 审我们的代码。第一轮（11-12 凌晨，八个角度 A–H，70 条发现）的清单在
  `docs/reviews/2026-11-12-cross-review/FIXLIST.md`，原报告在同目录 `review-<a…h>.md`。按用户 06:47 的指示，这一轮的审核 agent 用 Gemini 4 Carbon High。
- 这一轮审什么：第一轮之后有 17 张卡合进了 main（`git diff base-20261112..main`，`src`、`sim`、`tools` 下约 9,000 行），合并是主会话夜里逐张审的，没有第二个人看过；
  另有几块第一轮没有覆盖到：固件在环的 C 代码、PyTorch 原型线、端到端实验的设计与汇总器、部署链、文档与代码的一致性、Gazebo 闭环的耗时。
- 方向没有变（计划 `docs/PLAN_2026-11-10_phase12.md` §10）：交付路线是端到端，部署时只有一个网络，输入是机载图像、指令 token、由自己已执行的动作算出的历史码，输出 4 个动作。
- 只读：不改仓库里的任何文件，不提交，不建分支，不合并，不推送，不删任何东西。只写自己的报告与草稿。

## 1. 先读

1. `CLAUDE.md`；`docs/briefs/2026-11-10-phase12-common.md` 的 §1–§4（项目是什么、口径、路径、环境）；计划的 §0 与 §10（§10.7、§10.8 是第一轮审核之后的更正与进展）。
2. `docs/reviews/2026-11-12-cross-review/FIXLIST.md` 全文：第一轮的 70 条与各自的处理。已经在清单里的问题不重复报，除非你发现它的修复不对、修了一半，或者清单里写「已修」而代码里没有。
3. 自己角度对应的第一轮报告（下表「上一轮」一栏）。

## 2. 角度（一个 agent 一个角度）

| 代号 | 角度 | 范围 | 上一轮 | 重点 |
| --- | --- | --- | --- | --- |
| I1 | 11-12 合进 main 的 Gazebo 线改动 | `git diff base-20261112..main -- sim/tools ':!sim/tools/test_*'`，逐个合并提交看（W1-3 `bbc6132`、W1-4 `dd98f89`、W1-9 `d816d2c`、W1-7 `f383584`、W1-10 `41c474d`、K3 `df15994` 里 `rule_policy.py` 的部分） | `review-a.md` | 改动有没有引入回归；写着「缺省逐位不变」的地方是不是真的不变；四次冲突解决（清单「已合并」一节列了）有没有丢逻辑；教师 v7（过柱阵后目标取线、A* 无路时降余量重试）、航向码 `nav_cond.py`、评测前的一致性闸门、`--cf-sitl-params` 各自的边界情况；新开关两两组合 |
| I2 | 11-12 合进 main 的工具链改动 | `git diff base-20261112..main -- tools ':!tools/test_*'`（W1-1 `25e4b28`、W1-5 `8fadc83`、W1-2 `670c4b4`、W1-7 与 W1-8 `e41703e` 与 W1-9 里 `preprocess.py` 的三个新开关、W1-14 `0d7f3b5`、`tools/check.sh` 的两处）；分支 `p12-e2e-1` 上还没合的 `dc8e5ff`（`average_ckpt` 支持带条件输入的模型） | `review-d.md`、`review-e.md` | 打包在新开关各种组合下标签、图像、条件码、深度分带是否对齐；两条流水线交付的权重与评测口径；统计工具的新代码；门禁有没有因为这些改动出洞 |
| J | 3DGS 线 11-12 的改动 | K3 `df15994`（`sim/3dgs/rule_oracle.py` 的 v2、`sim/tools/rule_policy.py` 的 `rule_v2_*`）、W1-12 `dfb98c4`（`planner.py` 的 `clearance=swept`、出生两层判空）、S1a `5387747`；还没合的分支 `p12-s1b`（`git diff ba54628..p12-s1b`：碰撞口径 v2） | `review-b.md` | 规则 v2 的候选、分级通道、禁忌表的逻辑与状态残留；扫掠净空的几何；碰撞口径 v2 在判定、规划、规则臂、出生检查四处是不是同一个口径；`collect_3dgs.py` 与共用驱动 `experiments/2026-11-10-bearing-guard-3dgs/driver.py` 在 S1a 与 S1b 之间会怎样冲突 |
| K | 固件在环与执行器 | `sim/cf_sitl/` 全部（`exec/executor.c`、`src/`、`fw/`、`gz/` 插件、`proc/`、`selftest/`）、`sim/tools/cf_sitl_exec.py`、`sim/dynamics/e1_mini.json` | `review-g.md`；`docs/reviews/2026-11-10/sim-real.md` | 动作到期与被新动作取代的时序（卡 W1-13 第三部分「动作持续到下一次派发」要改的就是这里，草稿目录 `$D/tmp/phase12-cards/w1-13/handoff-from-39b5006b.md` 有一份读码笔记，核对它）；到期后回 SAFE 是位置保持还是滑行；电量模型；插件与 Python 之间话题的先后；单位与符号；内存与未定义行为；与 CF-Drone 固件原件的出入 |
| L | PyTorch 原型线 | 分支 `p12-e2e-p0` 的 `proto/`（工作树 `/root/claude-code-backup-20261028/code/wt/p12-e2e-p0`，只读；约 5,500 行）；报告 `$D/tmp/phase12-cards/reports/e2e-p0.md` | `review-c.md`、`review-h.md` | 复刻是不是真的等价（初值、随机数、打乱、批次、类权重、梯度裁剪、评估口径、导出）；对拍测试的盲区；包装器的行协议；「统计上不可分」的判据站不站得住；在它上面做结构探索（卡 E2E-P1）会踩到的坑 |
| M | 端到端实验的设计与汇总器 | main 上的 `experiments/2026-11-12-decidability-audit/`；分支 `p12-e2e-1` 的 `experiments/2026-11-12-e2e-heading-code/`；分支 `p12-e2e-r1` 的 `experiments/2026-11-12-recipe-stability/`；任务卡里 E2E-1、E2E-1b、E2E-R1、E2E-P1 的规格 | `review-d.md`、`review-h.md` | 预注册的文字与判定代码是否一致；评测几何的清单有没有泄漏（与训练数据、与调参用过的几何）；配对与按几何聚类的检验实现；结论能不能由设计支撑，缺什么对照；E2E-P1 的臂表与预期有没有逻辑漏洞。输出目录（`$D/exp-recipe-stability-20261112/`、`$D/exp-e2e-heading-code-20261112/`）有实验在跑，只读 |
| N | 部署链与真机接口 | `tools/uart_action.py`、`docs/uart_action.md`、`src/infer.cpp`、`src/infer_main.cpp`、`sim/tools/student.py`、仓库里与 int8、板端导出、输入分辨率有关的全部代码与文档；`$D/real-drone/`（只读） | `review-g.md` 的 G2、G7；`review-c.md` 的 C3 | 板端推理路径与训练路径的每一处不同；串口协议在丢包、乱序、超时、没有回执时的行为；量化与导出现在到了哪一步、缺什么；上真机之前必须先有的测试 |
| O | 文档与代码的一致性 | `README.md`、`CLAUDE.md`、`docs/HANDOFF.md`、`docs/HANDOFF_3DGS_20261105.md`、`docs/EVAL_STANDARD.md`、`docs/ENGINEERING_RULES.md`、`docs/ROADMAP.md`、`tools/DATASET_FORMAT.md`、`docs/uart_action.md`、`proto/README.md`（分支 `p12-e2e-p0`）对照代码 | 无 | 文档里已经不成立的说法、跑不通的命令、互相矛盾的口径；新人照文档从零做会卡在哪一步。只报会让人做错事的，不报措辞 |
| P | Gazebo 闭环一集的时间花在哪 | `sim/tools/collect.py` 的主循环、`sim/tools/gz_io.py`、`sim/tools/eval_closed_loop.py`、`sim/tools/collect_parallel.py`、`sim/tools/scene_gen.py` 与世界模板；现成的数据：`$D/exp-action-latency-r2-20261111/` 与 `$D/exp-recipe-stability-20261112/closed-loop/` 里各集 `actions.jsonl` 的 `timing` 字段、`eval.json` 的 `performance`、各臂的 `.cg` 记录（只读） | `review-a.md` | Gazebo 池是现在的瓶颈（12 个臂 240 集要 1.8 小时，2 条车道约 3.5 核）。量清一集的墙钟与 CPU 各花在哪（起 gz、起飞、每步等帧、推理、落盘、话题订阅子进程、软件渲染），给出不改变协议与数值的提速点，各自估收益与风险，排出先后。不起 Gazebo，只读代码与现成数据 |

## 3. 怎么审

- 以代码为准，不以文档为准。每条发现落到文件与行号（合并改动的角度另写是哪个提交带进来的）。
- 审的是主仓库 `/root/claude-code-backup-20261028/code/uav_rgb` 的 main。主会话这段时间还在往 main 合并，开工时记下 `git rev-parse --short main`，写进报告开头；
  读某个提交的内容用 `git show <提交>:<路径>`，不要 `git checkout`。别的工作树（`/root/claude-code-backup-20261028/code/wt/*`）只读。
- 能验证的就验证：可以跑单个测试文件、写十几行的小脚本喂边界输入。单进程，前面加 `nice -n 15`，两分钟以内，同一时刻最多一个；能靠读代码判断的就不跑。
  不起 Gazebo，不跑 3DGS 批量渲染，不训练，不跑 `tools/check.sh`，不拿任何池的锁（`$D/tmp/phase12-cards/*.lock`）。
  草稿放 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-<代号小写>/`，`export TMPDIR=<草稿目录>/tmp PYTHONDONTWRITEBYTECODE=1`。根分区只剩 2.8 GB，数据盘 14 GB，不写大文件。
- 机器上的情况（07:54）：12 核配额已经用满。Gazebo 池在跑闭环实验（对抢核敏感），训练池在训练，另有 6 个执行者在各自的工作树里改代码，一个清理 agent 在统计磁盘。
  进程都不要动；shell 里没有 `rg`，用 `grep`。解释器：`.venv-tools/bin/python`（无 torch），3DGS 相关用 `$D/3dgs-venv/bin/python`，原型线用 `$D/teacher-venv/bin/python`。
- 质量优先于数量：每个角度不超过 12 条。不报风格、命名、「可以更优雅」。分清「已复现」「读代码确认」「推断」；推断的写出它成立需要什么条件、怎样验证。

## 4. 报告

写到 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/review2-<代号小写>.md`，中文，边审边写。结构：

1. 开头：审的是哪个 main（提交号）、起止时刻。结论一段：这个角度下最要紧的 3 件事。
2. 发现清单，按严重程度排（会让已有结论或在跑的实验出错的排最前，其次是会让数据或训练悄悄变坏的，再次是健壮性）。编号用「代号-序号」（如 `I1-3`）。每条写：
   位置（文件:行，哪个提交）；现象；为什么是缺陷，影响哪些已有的结论或后续实验；怎样复现或验证（命令或最小用例）；建议的最小修法；状态；建议并入哪张现有的卡，或者要新立卡。
3. 第一轮清单里与本角度有关的条目，你核对过的：修对了的、修了一半的、没修的，各列出来。
4. 查过但没有发现问题的地方。
5. 没有查的地方与原因。

写作：不用程度副词（非常、极其、十分、相当），不用套话（总之、综上所述、值得注意的是、让我们），不用破折号作同位语；数字要有出处；没核实的写「未核实」。
最后给主会话的回复不超过 20 行：最要紧的 3 条发现、各自的状态、报告路径。
