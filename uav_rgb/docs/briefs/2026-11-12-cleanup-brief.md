# 清理任务书（2026-11-12 07:54 立，给清理 agent）

- 来由：用户 11-12 07:46：「加派一个 agent 清理我们文件夹里面不重要的内容，删除没有用的实验结果、实验测试脚本、实验说明文档、过程文档」。
  数据盘用了 96%（剩 14 GB），根分区剩 2.8 GB；仓库里 `docs/worklog/` 189 份、`docs/experiments/` 55 份、`experiments/` 20 个目录，`docs/` 顶层还有历次的计划、审查、调研。
- 范围只有 uav_rgb 这个项目的两处：仓库 `R=/root/claude-code-backup-20261028/code/uav_rgb` 与数据盘 `D=/root/rivermind-data/uav_rgb`（现在 106 GB，其中 `tmp/` 28.6 GB）。
  `/root/rivermind-data` 下别的目录（`MutilModel_*`、`YOLOVOmodal_*`、`RelateAnything` 等）是用户别的项目，`/root` 下别的目录也不是我们的：不看、不动、不列入清单。
- 两处的做法不同，原因是能不能找回：
  - 仓库里的文件删了能从 git 找回。在分支 `p12-cleanup` 上用 `git rm` 删，一类一个提交，主会话看过删除清单再合并；合并前主会话在 main 上打标签 `pre-cleanup-20261112`。
  - 数据盘上的东西删了找不回。**这一轮你在数据盘上一个文件都不删**：只出清单与删除脚本，主会话逐项看过、批准之后由主会话执行。

## 1. 现在不能碰的（机器上有 15 个 agent 与 3 个守护进程在跑）

- 第十二阶段的东西（2026-11-10 19:00 之后创建或改过的）在数据盘上一律不进「删」这一类：在跑的实验与在做的卡拿它们当输入。包括但不限于
  `exp-*-2026111[0-2]`、`tmp/phase12-cards/`、`tmp/phase12-20261110/`、`tmp/build-p12-*`、`tmp/build-check-main-*`、`tmp/cleanup-20261112/`。它们在清单里只能是「留」「可瘦身」或「问」。
- 基础设施一律留：`3dgs-venv`、`teacher-venv`、`3dgs-scenes`、`datasets`、`params`、`bin-snapshots`、`baselines`、`real-drone`、`grill-me`、`xdg`、`tmp/build-cf-sitl-lib31`、
  `tmp/agent6-distill/trunc`（示范数据集，多张卡在用）、`exp-dagger-gz-20261109`（DAgger 聚合集与它的原始集目录）、`exp-student-select-20261109`、`exp-swa-calib-20261110`、`exp-action-latency-r2-20261111`。
  这个名单不全：凡是被下面第 3 节的「引用」命中的，都留。
- 仓库里：`src/`、`sim/`、`tools/`、`proto/`、`ui/`、`teacher/`、`rl_teacher/` 下的代码与测试不删；`CLAUDE.md`、`progress.md`、`README.md` 不改（`progress.md` 的旧节由主会话自己裁）；
  `docs/briefs/2026-11-10-phase12-*.md`、`docs/briefs/2026-11-1[12]-*.md`、`docs/PLAN_2026-11-10_phase12.md`、`docs/reviews/2026-11-12-cross-review/`、`docs/reviews/2026-11-10/`、`docs/reviews/2026-11-10-phase12/` 不删。
  还没合并的分支在改的文件不删（合并时会冲突）：先 `git diff --name-only main...<分支>` 把 `p12-s1b`、`p12-w1-13`、`p12-w1-15a`、`p12-w1-15b`、`p12-w1-10`、`p12-e2e-1`、`p12-e2e-r1`、`p12-e2e-p0`、`p12-c1a` 碰到的文件列出来，这些文件与它们所在的实验目录这一轮不动。
- 不停任何进程，不拿任何池的锁，不 `git worktree remove`，不 `git branch -d`（工作树与分支只列清单）。

## 2. 第一部分：仓库（分支 `p12-cleanup`，从 main 起；可以直接删，一类一个提交）

判定「有用」的办法：从现行的入口出发能引用到。入口（根集合）：
`README.md`、`CLAUDE.md`、`progress.md` 里「第十二阶段」那一节（到 `## 2026-11-10 14:41` 之前）、`docs/PLAN_2026-11-10_phase12.md`、`docs/briefs/` 里上面列的几份、`docs/reviews/2026-11-12-cross-review/FIXLIST.md`、
`docs/EVAL_STANDARD.md`、`docs/ENGINEERING_RULES.md`、`docs/ROADMAP.md`、`docs/HANDOFF.md`、`docs/uart_action.md`、`tools/DATASET_FORMAT.md`、`tools/check.sh`，以及 `src/`、`sim/`、`tools/`、`proto/`、`ui/`、`teacher/`、`rl_teacher/` 下的全部代码。
一份文档或脚本被根集合里的文件按路径或文件名提到，就算被引用；被引用的文档再提到的文档也算（传递一层就够，写明你传递了几层）。引用图用脚本算，脚本与结果放草稿目录。

| 类 | 对象 | 删的条件 |
| --- | --- | --- |
| a 过程文档 | `docs/worklog/*.md` | 文件名里的日期早于 2026-11-10，而且不被引用。第十二阶段的（11-10 起）都留 |
| b 实验说明文档 | `docs/experiments/*.md` | 不被引用 |
| c 实验测试脚本 | `experiments/*/` 里的 `test_*.py`、`stub_*.py`、`rehearse*.sh` 与只给它们用的夹具 | 这个实验已经跑完（数据盘上对应的输出目录有 `DONE`）；不在 `tools/check.sh` 里；不被别的目录 import；不在第 1 节「没合并的分支在改」的名单里。共用驱动 `experiments/2026-11-10-bearing-guard-3dgs/driver.py` 的测试（各目录里的 `test_driver_*.py`）留。入口脚本、汇总脚本、配置、`README.md`、`RESULT.md` 留（计划 §1.3 的数字出自它们） |
| d 旧的计划、审查、调研、任务书 | `docs/` 顶层的 `*.md` 与 `docs/archive`、`docs/delivery`、`docs/realism_rl_v1`、`docs/briefs/` 里第十二阶段之前的 | 不被引用 |
| e 杂项 | 仓库里没人引用的图片、`.bak`、临时文件、空目录、`experiments/cuda_smoke`、`experiments/pipeline_profile` | 不被引用 |

- 每一类先出一张表（文件、行数、最后改动的日期、被谁引用或「无引用」、删还是留），写进报告，再按表 `git rm`，一类一个提交，说明第一行如 `chore(cleanup): 删第十二阶段之前无引用的过程文档 N 份`，正文写判定条件与怎样找回（`git show pre-cleanup-20261112:<路径>`）。
- 拿不准的不删，列在报告的「拿不准」一节，写清拿不准在哪。宁可少删。
- 删完之后在保留下来的文件里找一遍指向已删文件的引用，列进报告（死链清单）。入口文档里的死链不要自己改，交主会话。
- 验证：`tools/check.sh --python-only`（守护脚本排 `check.lock`，写法见公共规则 §4「23:15 起」第 2 款）rc=0；`git status --short` 为空。

## 3. 第二部分：数据盘（只出清单与脚本，不删）

- 现成的大小统计（主会话 07:48–07:53 跑的，单位 MiB）：`$D/tmp/cleanup-20261112/du_top.txt`（`$D` 顶层 173 项）、`du_tmp.txt`（`$D/tmp` 下一层）。要更深一层的自己统计，命令前加 `ionice -c3 nice -n 19`，不要并行跑多个 `du`。
- 清单一行一项（`$D` 顶层每一项、`$D/tmp` 下每一项；`tmp/phase12-cards` 再往下一层）：路径、大小、最后修改时刻、是什么（一句话，写依据）、引用情况、分类。
  - 引用情况查三处：仓库 main 上的文件里出现这个路径或目录名的地方（`grep -rn`，记命中数与最要紧的一两处：计划 §1.3、`progress.md` 第十二阶段一节、任务卡、公共规则、入口脚本的配置区）；
    在跑的守护进程与在做的卡的输入（`$D/tmp/phase12-cards/HANDOFF-1112-0700.md`、各实验输出目录的 `inputs.txt`、`config.env`、`launches.log`、`code_head.txt`，`ps -eo args` 里出现的路径）；
    别的保留项对它的依赖（符号链接、`manifest.json` 里的绝对路径）。
  - 分类：
    「留」：第 1 节的基础设施，或者被引用。
    「删」：早于第十二阶段、三处都没有引用，而且属于这几种之一：pip 与下载缓存；旧的构建目录；冒烟与试跑的输出；已经被后来的实验取代的旧实验输出；变异检查的副本；临时目录。每一项写它属于哪一种、取代它的是什么。
    「可瘦身」：结论要留但原始产物占地方的。留 `eval.json`、`summary*`、`manifest*`、`actions.jsonl`、`oracle.jsonl`、`DONE`、日志，删逐步的图像帧与视频；给出瘦身能省多少。第十二阶段的输出目录这一轮只估算，不进脚本。
    「问」：拿不准的，写清拿不准在哪、你倾向哪边。
- git 工作树与分支单列一张表：路径、分支、是否已全部进 main（`git branch --merged main`）、大小、有没有未提交的改动、是不是在跑的守护进程的工作目录或代码来源。只列，不动。
- 交付的文件（都放 `$D/tmp/cleanup-20261112/`）：
  `manifest.tsv`（机器可读）、`manifest.md`（按分类汇总，每类多少 GB，「删」与「可瘦身」各自按大小排）；
  `delete_approved.sh`：读同目录的 `approved.txt`（一行一个路径，主会话填），逐条核对后删。核对：路径规范化之后必须在 `/root/rivermind-data/uav_rgb/` 之内且不等于它本身；不是指向外面的符号链接；不在 `keep.txt`（「留」的全部路径，脚本内置一份）里，也不是其中任何一项的上级目录；
  没有进程正开着它下面的文件（查 `/proc/*/cwd` 与 `/proc/*/fd`）。用 `ionice -c3 nice -n 19 rm -rf --one-file-system`，每删一条往 `deleted.log` 追加一行（时刻、路径、删之前的大小）；任何一条核对不过就跳过并记进 `skipped.log`，不中止也不强删。支持 `--dry-run`（只打印会删什么）。
  `tools` 目录之外的脚本没有门禁，所以这个脚本自己要有演练：在草稿目录里造一棵假的目录树（把脚本里的根路径做成可由环境变量覆盖，只给演练用），验证四种拒绝与 dry-run 都成立，把演练的真实输出贴进报告。
- 只统计、只读文件。唯一允许写的地方是 `$D/tmp/cleanup-20261112/` 与你自己的草稿目录。

## 4. 报告与交回

- 报告 `$D/tmp/phase12-cards/reports/cleanup.md`，中文，边做边写；过程文档 `docs/worklog/2026-11-12-<开工时刻>-cleanup.md` 在 `p12-cleanup` 上提交（做了什么、没做什么与原因、待确认、验证到哪一步、风险与遗留）。
- 写作：不用程度副词，不用套话，不用破折号作同位语；数字要有出处；没核实的写「未核实」。
- 最后给主会话的回复不超过 30 行：仓库删了几类、各多少份、多少行，分支与提交号；数据盘「删」「可瘦身」「问」各多少 GB、最大的十项；工作树里能移除的有几个、多大；拿不准的清单在哪；验证结果。
