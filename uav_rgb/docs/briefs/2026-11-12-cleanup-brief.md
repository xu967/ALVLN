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

## 5. 第二轮（2026-11-12 17:05 立；用户 16:56：「加派 agent 去审核删除多余的结果文件、测试脚本、日志什么的，没有用的都可以删除了」）

- 第一轮的结果：仓库删了 157 个文件（合并 `40fd9a3`，恢复其中 8 份 `d33a114`）；数据盘清单 `$D/tmp/cleanup-20261112/manifest.{tsv,md}` 里「删」类 230 项，主会话 16:22 批准并删了 225 项，21.7 GiB
  （批准记录 `approval_notes_1112.md`，删除记录 `deleted.log`）。数据盘可用 40 GB，根分区可用 2.5 GB（紧）。没动的：「可瘦身」26 项、「问」59 项、72 个登记的 git 工作树（70 条分支里 58 条已全部进 main）。
- 这一轮分两张卡，都只动 uav_rgb 的东西（范围同第 0 节：数据盘 `$D`，仓库与它的工作树 `/root/claude-code-backup-20261028/code/{uav_rgb,wt}`；别的项目一律不看不动）：
  CLN2-A 管数据盘，CLN2-B 管 git 工作树与仓库。
- 机器现在很挤：12 核与磁盘是与用户另外两个项目共用的，IO 压力高。统计一律 `ionice -c3 nice -n 19`，同一时刻只跑一个 `du` / `find` / `sha256sum`；第一轮量过的大小直接用清单里的，不重量。

### 5.1 现在在用的，一律不进任何删除或瘦身清单

- `$D/tmp/phase12-cards/` 里：`reports/`、四把锁、两份 `HANDOFF-*.md`，以及正在做的卡的草稿目录
  `s1b`、`w1-15c`、`dep1`、`t2a`、`e2e-1`、`e2e-p0`、`e2e-p1`、`p1b`、`p1c`、`w1-10`、`bl1`、`bl2`、`bl3`、`bl4`、`cleanup`、`cln2-a`、`cln2-b`。
- 第十二阶段的全部实验输出目录（`exp-*2026111[0-2]*`、`exp-2026-11-12-*`）：有的正被别的卡读（样本外补评读 `exp-e2e-heading-code-20261112` 的冻结代码与权重；原型线读 `exp-proto-p1-20261112` 的逐轮权重；
  开源对比读 K1 的三个目录），这一轮整个不动，只在清单里记「以后可瘦身多少」。
- 基础设施：第一轮 `keep.txt` 与脚本内置名单里的全部（venv、`3dgs-scenes`、`datasets`、`baselines`、`bin-snapshots`、`torch-cache`、`tmp/build-cf-sitl*`、`tmp/agent6-distill/trunc`、`exp-dagger-gz-20261109`、
  `exp-student-select-20261109`、`exp-swa-calib-20261110`、`e1-pipeline-20261107-0417*`、`pubdata/sage3d` 等），外加 `tmp/phase12-20261110/`（主会话的脚本与日志）、`tmp/cleanup-20261112/`、
  `tmp/build-check-main-1112`、`tmp/build-check-main-1112-cf-sitl`（全量检查的构建缓存）。
- 工作树：当前 12 张卡的工作树（`wt/p12-s1b`、`p12-w1-15b`、`p12-dep1`、`p12-pk1`、`p12-e2e-1`、`p12-e2e-p0`、`p12-e2e-p1`、`p12-e2e-p1b`、`p12-w1-10`、`p12-bl2`、`p12-bl3`、`p12-bl4`）、
  主会话的 `wt/merge-try` 与 `wt/check-main`、主工作树。
- 旧 teacher 线的数据（`ds-teacher400`、`teacher-400-20261102`、`teacher-600-20261102`、`teacher-1000-merged`、`runs`）：等用户定 `teacher/` 这条线的去留，这一轮不动。

### 5.2 卡 CLN2-A：数据盘第二批（只出清单与脚本，不删；主会话逐项核对后执行）

数据盘上的东西删了找不回，所以执行权仍在主会话。你把「审核」做扎实，让主会话照着清单就能放心执行。

1. 第一轮「可瘦身」里第十二阶段之前的 10 项（`xeval-3dgs-20261109`、`tmp/agent6-viz-layout`、`tmp/agent4-teacher-p1`、`3dgs-smoke-20261107`、`exp-teacher-3dgs-20261109`、`xeval-3dgs-20261109-teacher-h05`、
   `exp-viz3-20261110`、`campus-paired-20261103`、`demo-e1-20261110`、`e1-pipeline-20261107-0417-eval3`；`tmp/cursor-1110` 里有两个登记的 git 工作树，归 CLN2-B，其余部分归你）：
   逐项给出要删的路径（目录或带通配规则展开后的文件清单），只删逐步的图像帧、未压缩的原始帧、已被成片取代的旧视频；`eval.json`、`summary*`、`manifest*`、`actions.jsonl`、`oracle.jsonl`、`trace`、`DONE`、日志、成片 mp4、
   每集的首帧都留。第一轮的草稿里有量好的分项（`$D/tmp/phase12-cards/cleanup/slim_breakdown.json`、`measure_slim_savings.py`），先用它。
   每一项写明：留下的东西够不够复原结论（哪份文档引用了它的哪个数）。
2. 已经合并、不在 5.1 名单里的第十二阶段任务卡草稿目录（`$D/tmp/phase12-cards/` 下的 `w1-0` 到 `w1-9`、`w1-11` 到 `w1-15b`、`k1`、`k3`、`c1a`、`pk1`、`t1`、`s1a`、`e2e-0`、`e2e-r1`、`e2e-1-cluster`、
   `review-*`、`review2-*`、`tmp-*`、`check-queue-sonnet`）：只列其中的大件，逐目录给出「删哪些子路径、各多大、留什么」。可删的是：构建目录、代码副本树（冻结副本、变异检查用的副本、基线快照）、`tmp/` 子目录、重复的数据副本、逐帧图像。
   留的是：`red/`、`green/`、各种日志、脚本、json 与 md 结果、`OWNER`（报告拿它们作证据）。某个子路径被报告 `reports/<卡>.md` 当作证据点名，就留。
3. 第一轮「问」里的：`downloads`（逐个大文件核对它在 `3dgs-scenes` 里有没有同 inode 或同 sha256 的副本，有才列为可删；重新下载的代价写明：11-09 记录过 Inria 直连 29 KB/s）；
   `pubdata/extract-airsim10k`（核对 `airsim10k-uavr` 确实是它打包出来的、`HANDOFF.md` 里确认不需要的那句话在哪一行）；11-10 19:00 之后留下的 nvcc 中间文件与 `tmp/tmp*` 残留（核对没有进程在用、不是今天的执行者正在写的）；
   `tmp/phase12-cards/tmp-*` 与 `check-queue-sonnet`。
4. 第一轮之后新出现或当时因为在用而留下的：`tmp/build-p12-c1a`、`tmp/build-p12-t1`（两张卡都已合并）、`tmp/build-check-main-1111*`（被 1112 取代）、`tmp/build-student-select`，以及 `$D` 顶层与 `$D/tmp` 下第一轮清单里没有的新条目（逐个说明是什么、谁建的、还在不在用）。
5. 日志：`$D` 下体积在 5 MiB 以上的 `*.log` 列一张表（路径、大小、属于哪个实验、还有没有文档引用），已结束实验的超大逐步日志可以列为可删或可压缩，结论性的日志（`run.log`、`launches.log`、汇总日志）留。
- 交付（都放 `$D/tmp/cleanup-20261112/round2/`）：`manifest2.tsv` 与 `manifest2.md`（每条：路径、大小、分类、依据、留下什么）；`approved2_candidate.txt`（一行一个要删的绝对路径，可以是 5.1 之外目录内部的子路径）；
  `delete_approved2.sh`：在第一轮脚本的四项核对之上，保护名单换成 5.1（内置），允许删保护名单之外目录的内部子路径，拒绝任何落在保护名单内或是其上级的路径，支持 `--dry-run`，照第一轮的办法在假目录树上演练并把真实输出贴进报告；
  用候选清单跑一遍 `--dry-run` 的日志。报告 `$D/tmp/phase12-cards/reports/cln2-a.md`，草稿 `$D/tmp/phase12-cards/cln2-a/`。
- 这张卡不提交仓库文件（脚本在数据盘上）；过程文档写在报告末尾三节里，主会话收进自己的过程文档。

### 5.3 卡 CLN2-B：git 工作树与仓库第二批

1. 工作树（72 个登记的；`/root/claude-code-backup-20261028/code/wt/` 在根分区上共 546 MB，根分区只剩 2.5 GB；另有 `$D/tmp/wt-*` 与 `$D/tmp/cursor-1110/{wt,wt-r1}`）：逐个列表（路径、分支或游离的提交、大小、
   是否已全部进 main、工作树是否干净含未跟踪文件、有没有进程的工作目录或打开的文件在里面、是不是某个还要用的冻结代码来源）。
   同时满足下面五条的可以由你直接移除（`git worktree remove <路径>`，不带 `--force`；分支一律保留，不删分支）：
   不在 5.1 的名单里；HEAD 是 main 的祖先（`git merge-base --is-ancestor`）；`git status --porcelain --untracked-files=all` 为空；`/proc/*/cwd` 与 `/proc/*/fd` 里没有它下面的路径；
   它的路径不出现在任何还没有 `DONE` 的实验输出目录的 `launches.log`、`config.env`、`inputs.txt` 里。移除前把清单写进报告，逐个移除、逐个记一行（时刻、路径、分支、HEAD、移除前的大小）。
   不满足的不动，列出来并写清卡在哪一条（比如有未提交的改动：列出是哪些文件）。
2. 仓库（分支 `p12-cleanup2`，新工作树 `/root/claude-code-backup-20261028/code/wt/p12-cleanup2`，从 main 起；`git rm`，一类一个提交，找回办法照第 2 节）：
   - 实验测试脚本（第 2 节的 c 类，判定条件不变）：第一轮之后跑完并已合并的实验，例如 `experiments/2026-11-12-decidability-audit/`、`experiments/2026-11-12-recipe-stability/` 的 `test_*.py` 与桩。
     还没合并的分支在动的实验目录不碰（先 `git diff --name-only main...<分支>` 把 `p12-s1b2`、`p12-w1-15c`、`p12-dep1`、`p12-t2a`、`p12-e2e-1`、`p12-e2e-p0`、`p12-e2e-p1b`、`p12-w1-10`、`p12-bl2`、`p12-bl3`、`p12-bl4` 碰到的文件列出来）；
     `experiments/2026-11-12-e2e-heading-code/`（样本外补评在用）、`experiments/2026-11-11-soc100-parity/`、`experiments/2026-11-12-3dgs-rebaseline/`、共用驱动的 `test_driver_*.py` 这一轮都留。
   - 第一轮恢复的 8 份（`docs/experiments/3dgs-site-*.md`、`docs/RESEARCH_2026-11-07_fastlab_*.md`、`docs/RESEARCH_BACKLOG_2026-11-04.md`）不删。
   - 仓库里跟踪着的结果文件、日志、图片：列一张表（路径、大小、谁引用）。被 `RESULT.md` / `README.md` 点名的结果文件留；没人引用的才删。
   - 再跑一遍第 2 节的引用图（根集合不变，传递一层），看今天的合并之后有没有新变成无引用的文档；`docs/worklog/` 里 11-10 之后的仍然都留。
   - 删完之后的死链清单接着第一轮的（94 处）更新，交主会话（归卡 DOC1）。
   - 验证：删掉的文件没有被任何代码 import 或调用（`git grep`）；剩下的实验测试逐个跑一遍；`tools/check.sh --python-only`（守护脚本排 `check.lock`）rc=0；`git status --short` 为空。
- 报告 `$D/tmp/phase12-cards/reports/cln2-b.md`，草稿 `$D/tmp/phase12-cards/cln2-b/`，过程文档 `docs/worklog/2026-11-12-<HHMM>-cleanup-round2.md` 在 `p12-cleanup2` 上提交。

### 5.4 第二轮交回之后的定案（11-12 19:45，主会话）

- 卡 CLN2-B 交回：工作树移除 37 个（分支保留）；仓库删 10 个文件 1988 行，已合并（`426198d`，合并前的标签 `pre-cleanup2-20261112`）。
- 拿不准的六组都留：`recipe-stability/test_recheck_sweep.py`（结果记录点名的归档件）；`w1-13-newprotocol` 的演练脚本与桩（卡 W1-13 第三轮在做）；
  `sim/assets/campus_textures/` 与 `sim/worlds/campus_textures/` 的六张图（后三张是生成脚本的输出，共 48 KB，数据盘上已生成的世界文件会不会按路径引用没查）；
  三份 11-12 的实验解读文档；第一轮恢复的 8 份调研与场地文档。
- 以后再跑引用图：根集合加上 `experiments/*/README.md` 与 `experiments/*/RESULT.md`，引用源加上 `$D/3dgs-scenes/*/scene.json`。
- 永久保留、不再评：`docs/experiments/3dgs-site-*.md`、`docs/RESEARCH_2026-11-07_fastlab_*.md`、`docs/RESEARCH_BACKLOG_2026-11-04.md`（用户要的调研）。
- 工作树：`wt/p12-w1-9` 等卡 E2E-1 合并后移除；`wt/p12-rulev1` 留（`experiments/2026-11-10-rule-oracle-v1/summarize.py` 缺省从它读代码）；`wt/p12-longcourse` 的两个漏合的提交已合进 main（`7b5b2ca`），工作树可以移除。
- 卡 CLN2-C（同一个执行者接着做）：数据盘上 15 个有未提交文件的老工作树，先把未提交的内容存档到 `$D/archive/wt-salvage-20261112/<名>/`（补丁、未跟踪文件的包、哈希），在临时检出上核对能逐字节还原，再 `git worktree remove --force`；
  `wt-sim-dyn` 的 `sim/tools/cf_sitl_sweep.py` 与 `wt-teacher-3dgs` 的 `sim/3dgs/test_teacher_occ.py` 等 main 里没有的内容由存档保住。`tmp/cursor-1110/wt` 与 `wt-r1` 是别的会话的，不动，等用户定。
- 死链 139 处交卡 DOC1（19:46 派）。`experiments/2026-11-12-e2e-heading-code/test_prepare_rehearsal.py` 有 1 个用例偶发失败（根因没查），等卡 E2E-1 的执行者交回时交给它。

### 5.5 卡 CLN2-C 交回之后（11-12 20:52，主会话）

- 数据盘上 15 个有未提交文件的老工作树已移除，`wt/p12-longcourse` 与 `wt/p12-cleanup2` 也已移除（后一个由主会话 20:51 移除）。分支都保留。登记的工作树从 72 个降到 23 个（含主工作树）。
- 存档 `$D/archive/wt-salvage-20261112/`（2.0 MB，15 个子目录，每个有补丁、未跟踪文件的包、哈希与还原核对的记录；`restore.sh` 与 `README.md` 说明怎样还原）。
  移除前与移除后各做了一遍还原核对：189 个脏文件里 188 个的哈希与可执行位一致，1 个已删文件确实不存在。
- **`$D/archive/` 进保护名单，任何清理都不碰。** 下面这些内容只有存档这一份，仓库的任何分支里都没有：
  `wt-sim-dyn` 的 `sim/tools/cf_sitl_sweep.py`（469 行）；`wt-teacher-3dgs` 的 `sim/3dgs/test_teacher_occ.py`（528 行）与 `planner.py`、`collect_3dgs.py` 的改动（一版没做完的 `astar_occ_tol`，任务卡里写过只作参考、不合并）。
- 被移除的工作树还被这些入口脚本当缺省的源码目录：`experiments/2026-11-10-long-course/{calibrate_timing,run,run_v2}.sh` 的 `SRC=`（`wt/p12-longcourse`）；CLN2-B 报告 1.7 节另列了 13 处。
  这些实验都已跑完；要重跑，先用 `$D/tmp/phase12-cards/cln2-b/worktrees_restore.sh <名字片段>` 把工作树建回原路径。
- `$D/tmp/build-student-select` 的源码目录随 `wt-student-select` 一起没了，里面的 `uavrgb_train` 二进制被 E2E 的实验记录引用，留着不清。
- 没动、等用户定：`tmp/cursor-1110/wt` 与 `tmp/cursor-1110/wt-r1`（别的会话的工作树）。等卡 E2E-1 合并后移除：`wt/p12-w1-9`。
