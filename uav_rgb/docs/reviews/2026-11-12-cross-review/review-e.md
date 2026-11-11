# 交叉审核 E：实验入口与流水线的健壮性（main @ 97c4228）

- 审核者：Claude Opus 5.5（只读）。日期 2026-11-12。任务书 `docs/briefs/2026-11-11-cross-review-brief.md` §1 代号 E。
- 范围：`experiments/*/run*.sh`（14 个）、`experiments/2026-11-10-bearing-guard-3dgs/driver.py`（三个 3DGS 入口共用）、`tools/e1_pipeline.py`、`tools/dagger_gz_pipeline.py`、`tools/dagger_pipeline.py`、`tools/pipeline_common.py`、`tools/check.sh`、`env.sh`。
- 行号取自主工作树（审核时 `git status --porcelain` 为空，与 HEAD 相同）。
- 上一轮 `docs/reviews/2026-11-10/arch.md` 已写的不重复：GPU UUID 不在代码里、插件目录缺省值、代码依赖 `$D/tmp` 草稿、主工作树被并发改（E2）、`--dynamics` 缺省 x3。
- 草稿与验证输出：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review-e/`（`verify_small.py`、`verify_small.txt`）。

## 1. 结论

1. 被容器重启打断的臂，续跑时整臂挪成 `.partial-<时刻>` 后从头重评，半成品里已经跑完的集不进汇总，`summary.md` 也不提重启。11-11 第二轮时延标定就是这样：`acc-L0.20-k3-s0` 被打断前已跑完的 8 个 seed 里成功 1 个，重评后同样 8 个 seed 成功 3 个（seed 113、172 由撞柱变成功）。读汇总的人看不出这个臂是重评的。按预注册「迟到步用首次记录」，标定臂如果被打断，同一机制会丢掉首次记录里的迟到步，门的判定会翻转。
2. 续跑不核对上次的输入，直接覆盖起跑记录（`inputs.txt`、`config.json`、`inputs.json`、`start.txt`）。大多数入口用活的工作树，不冻结代码。同一个输出目录里可以混进几份代码，而记录只剩最后一份。`exp-sage3d-pilot-20261110` 起了 3 次，HEAD 分别是 9098598、6182237、1c01078：教师臂是第一次跑的，其余三臂是第三次跑的，`inputs.txt` 只写了 1c01078。第三次起跑用环境变量 `GATE_SR=0` 越过了预注册闸门，`summary.md` 第 0 节的标题仍是「预注册 P1」，写着「≥ 0，全部通过」。
3. 三条 Python 流水线没有输出目录锁，`run.pid` 由操作者另写，只记一个 PID。容器重启后这个 PID 已经失效，但停进程的规矩仍是 `kill -- -$(cat run.pid)`。重复启动或旧实例还活着时，评测步的「不完整就挪开重评」会把另一个实例正在写的评测目录挪走，再起一份。shell 入口都有 `flock -n`，Python 流水线没有。

## 2. 发现清单（按严重程度）

### E-1 重启打断的臂整臂重评，半成品不进汇总，汇总不标注【已复现，用现有数据】

- 位置：`experiments/2026-11-10-action-latency/run_r2.sh:269-271`（`arm_complete`）、`:280-284`（挪成 `.partial-<时刻>`）；`summarize_r2.py` 里没有任何读取 `.partial` 的代码（`grep -n partial` 只命中 eval.json 的 partial 字段，在 `summarize.py`；`summarize_r2.py` 无匹配）。其他入口用的是同一种写法：`heading-guard/run.sh:62`、`swa-calib/run.sh:70`、`long-course/run_v2.sh:95`、`baselines-zeroshot/run.sh:121`、3DGS 的 `driver.py:156-157`（`*.incomplete-<时刻>`）。
- 现象：`exp-action-latency-r2-20261111/run.log` 记了 21:41:57 起跑、23:31:25 续跑（PGID 237956 → 16307），中间没有「收到终止信号」一行，说明进程是被 SIGKILL 杀掉的，trap 没有机会执行。续跑时 5 个已完成的臂被跳过，`acc-L0.20-k3-s0` 与 `acc-L0.20-k1-s2` 被挪成 `.partial-233200` 后从头重评。
  - `acc-L0.20-k3-s0.partial-233200.log` 里有 8 个 seed：1 成功；113、153、172、176、180、230 撞柱；122 在第 137 步因电池 FAILSAFE 采集中止。
  - 重评后（`acc-L0.20-k3-s0/eval.json` 的 `per_episode`）同样这 8 个 seed：1、113、172 成功，其余失败。
  - `summary.md` 里搜不到 partial、重启、中断、续跑（`grep -n -i -E "partial|重启|中断|restart|续跑" summary.md` 只命中第 18 行 `.cg` 的说明）。
- 为什么是缺陷：Gazebo 闭环不确定，同权重同 seed 重评的逐集一致率本身只有 0.55–0.76（run_r2.sh 头注释第 15 行与 R3 条款）。整臂重评等于换了一份样本。第二轮的验收 4 拿 K=3 第 1 块通过率与 K=1 成功率做配对，判定线是 0.08 与 0.12，n=60；一个臂里 2 集的变化就是 0.033。所以「哪一份算数」应当事先写下，并在汇总里标出来。标定臂的情况更直接：预注册写的是「迟到步数用首次记录，标定臂不重跑」（run_r2.sh:33），可只要标定臂被重启打断，同一段代码就会整臂重跑并丢掉首次记录。第二轮的标定臂恰好都在重启前跑完了，没有受影响。
- 验证：`diff <(grep -o '^\[seed [0-9]*\].*success=[A-Za-z]*' acc-L0.20-k3-s0.partial-233200.log) <(...)`，或用 `verify` 里的读法比 `per_episode`。本次读的是 `episode_manifest.json` 与 `eval.json`。
- 最小修法：二选一，先写进预注册。（a）续跑只补缺的 seed：从 `.partial-*/episodes/seed_*/episode_manifest.json` 读出已完成的 seed，用已有的 `--seeds` 只评剩下的，汇总时合并两份；（b）保留整臂重评，但汇总读 `<臂>.partial-*` 并新开一节，列出被重评的臂、重启时刻、半成品里每个 seed 的结果。不论选哪种，标定臂都按首次记录处理：被打断就停下报告，不自动重跑。
- 影响：`exp-action-latency-r2-20261111` 的 `acc-L0.20-k3-s0`、`acc-L0.20-k1-s2` 两个臂。数字本身不算错，缺的是标注。

### E-2 续跑覆盖起跑记录，活工作树让同一目录混入多份代码【已复现】

- 位置（每次起跑都用 `>` 重写，或用 `write_json` 覆盖）：
  - `heading-guard/run.sh:52`、`swa-calib/run.sh:59`（`inputs.txt`，代码取 `$D/tmp/cursor-1110/wt*` 活工作树，只记 `git diff --stat`）
  - `bearing-guard-3dgs/run.sh:84-88`、`sage3d-pilot/run.sh:112-115`、`rule-oracle-3dgs/run.sh:167-172`、`teacher-tol-3dgs/run.sh:177-181`（代码是活工作树 `$WT`，`inputs.txt` 只记当次 HEAD）
  - `run_r2.sh:195-198`（`inputs.txt`）与 `:246-259`（`start.txt`）：代码已冻结，但第一次起跑的条件记录被续跑覆盖（现存 `start.txt` 是 23:31:56 那次的）
  - `tools/e1_pipeline.py:352-357`、`tools/dagger_gz_pipeline.py:746-751`、`tools/dagger_pipeline.py:283-287`：`config.json` 里的 `git` 取当次 `REPO` 的 HEAD；`REPO` 是脚本所在的检出，没有冻结
  - `tools/dagger_gz_pipeline.py:390-395`：每次 preflight 都用当前 `BUILD` 的 `uavrgb_train` / `uavrgb_infer` / 插件 sha 重写 `inputs.json`；`baselines()`（:579-591）用这份新 sha 判断基线能不能复用
  - `tools/e1_pipeline.py:24`：`BUILD` 缺省是 `$REPO/build/check`，也就是 `check.sh:6` 的构建目录，全量 `check.sh` 会在原地重编；`e1_pipeline` 只记路径，不记二进制 sha
- 现象（实例）：`exp-sage3d-pilot-20261110/run.log` 有 3 行「开始」，WT 分别是 9098598（20:43）、6182237（20:59）、1c01078（21:06）。第三次起跑时教师臂的 40 条路线全部「done.json 已完成」被跳过，所以教师臂是 9098598 跑的，其余三臂是 1c01078 跑的，`inputs.txt` 末行只有 `git HEAD 1c010783…`。三个提交之间的差异落在共用的 `driver.py`（+3 −1，加了可选键 `teacher_tol_deg`，缺省不传时行为不变）与 `sage3d-pilot/run.sh`。这一次的差异不影响数字，但从产物本身判断不出来。
- 为什么是缺陷：续跑会把「哪段数据由哪份代码、哪个二进制产生」的证据抹掉。今天的三次重启都会走到这条路径。`dagger_gz_pipeline` 在第 2 轮之后续跑时，如果 `UAV_RGB_BUILD_DIR` 换了，前两轮用旧二进制训练、后面用新的，而 `inputs.json` 只记新的。
- 验证：`grep -E "开始 PGID" $D/exp-sage3d-pilot-20261110/run.log`；`git -C wt/p12-sage3d diff --stat 9098598 1c01078`。
- 最小修法：第一次起跑时写 `inputs.txt`，续跑时写 `inputs.<时刻>.txt`，并与第一份比较 sha，不同就拒绝起跑（仿照 `rule-oracle-3dgs` 对路线文件的 `cmp -s` 写法，约 5 行）。Python 流水线同理：`inputs.json` 已存在且内容不同就退出，另加 `--allow-input-change` 开关；`e1_pipeline` 补记两个二进制的 sha。代码一律仿照 `run_r2.sh:176-191` 冻结（`git archive` 加逐文件 blob 核对）。

### E-3 Python 流水线没有输出目录锁；重复实例会互相挪走评测目录【推断】

- 位置：`tools/e1_pipeline.py:341-386`、`tools/dagger_gz_pipeline.py:718-791`、`tools/dagger_pipeline.py:247-358` 都没有 `flock`；评测步的「挪开重评」在 `e1_pipeline.py:255-262`（调用 `pipeline_common.retire_unfinished_eval`，`pipeline_common.py:211-221`）、`dagger_gz_pipeline.py:536-544`、`dagger_pipeline.py:203-206`。
- 现象：判断评测目录是否完成只看 `eval.json`。另一个实例正在跑的评测目录还没有 `eval.json`，第二个实例会把它 `os.rename` 成 `.failed-<时刻>`，再起一个新的 `eval_closed_loop`。第一个实例的子进程继续往改名后的目录写，同时多出一组 gz。`e1_pipeline` 的子进程用 `start_new_session=True` 起（`pipeline_common.py:102`），流水线进程被 SIGKILL 时，这些子进程在自己的会话里继续跑。
- 为什么是缺陷：swa-calib 在 11-10 17:18 出过同类事故，同一个 shell 入口被启动两次，数据作废（`swa-calib/run.sh:47` 注释）。之后 shell 入口都加了 `flock -n`，Python 流水线没有补。容器重启后最常见的操作就是「原命令再起一次」，一旦旧实例其实还活着（例如只有父进程死了），就会触发上面的情况。两个实例同时跑 gz 还会抢核，带 `--action-latency` 的实验会因此出迟到步。
- 成立条件：同一个 `--out` 上同时有两个流水线进程。验证：在草稿目录用 `--dry-run` 以外的桩命令起两个实例（本次没跑，规则不允许执行入口脚本）。
- 最小修法：`main()` 开头加 4 行：`fcntl.flock(open(out/.lock, "a"), LOCK_EX | LOCK_NB)`，拿不到就退出 3。顺便由流水线自己写 `run.pid`（见 E-4）。

### E-4 run.pid 只有 PID，重启后失效，按它停进程会误杀；状态文件分不出「在跑」与「被重启杀掉」【推断，现状已核实】

- 位置：所有 shell 入口的 `echo $$ > "$OUT/run.pid"`（例：`run_r2.sh:153`、`heading-guard/run.sh:51`、`rule-oracle-3dgs/run.sh:132`）；三条 Python 流水线由操作者 `echo $! > run.pid`（`e1_pipeline.py:5-6`、`dagger_gz_pipeline.py:10-11`、`dagger_pipeline.py:15-16`）。停进程的规矩是 `kill -- -$(cat run.pid)`（公共任务书 §6，各入口头注释）。
- 现状（`ps -g <pid>` 逐个查，2026-11-12 00:4x）：`$D` 下共有 37 个 `run.pid`，4 个对应活着的进程组。`exp-teacher-cost-20261109/run.pid`（291424）、`cl1/cl2-flawed-descent-killed` 三个目录既没有 DONE，也没有活进程。容器重启后 PID 从小号重新分配，现在的活进程 PID 在 16307 到 362651 之间，已经覆盖了 11-09 留下的 279117、286266、291424、305320 这一段。审核时没有发现撞号。
- 为什么是缺陷：（1）对一个已经死掉的实验执行 `kill -- -$(cat run.pid)`，可能杀掉别的 agent 的守护进程组；（2）重启后 `status.json` 停在 `running`、没有 DONE，等 DONE 的脚本会一直等下去（`action-latency/run.sh:95` 的等待循环没有上限），人只能靠 `ps` 判断。
- 最小修法：`run.pid` 写三项：`PGID`、`/proc/$$/stat` 第 22 列（进程启动时刻）、`/proc/1/stat` 第 22 列（容器重启后会变；`/proc/sys/kernel/random/boot_id` 与 `uptime -s` 是宿主机的，容器重启时不变，本机 `uptime -s` 显示 2026-10-28）。另写一个 `tools/stop_run.sh <OUT>`：三项都对上才执行 `kill -- -PGID`，对不上就报「已失效」。等 DONE 的循环同时检查这三项。

### E-5 rule-oracle-3dgs 在拿 .lock 之前写 DONE，误起的第二个实例会盖掉正在跑的实例的状态【推断，按代码顺序】

- 位置：`experiments/2026-11-10-rule-oracle-3dgs/run.sh:117`（无卡时写 `DONE rc=4`）与 `:122`（formal 带覆盖变量时写 `DONE rc=2`），都在 `:125-126` 的 `flock -n` 之前。
- 现象：正式实例在跑（它在 `:131` 已删掉 DONE）。这时如果有人带着残留的环境变量，或者在重启后 GPU UUID 变了的情况下再起一次，第二个实例会直接写出 DONE 然后退出。
- 为什么是缺陷：下游靠 DONE 排队（`baselines-zeroshot/run.sh:57-61` 的 `WAIT_FOR`、`action-latency/run.sh:95`，以及公共任务书 §4 的完成标记约定），会以为上游已经结束，提前起跑。11-11 19:23 那次重启换了 GPU UUID，第二个触发条件是真实发生过的。`run_r2.sh` 的 `refuse()` 放在锁之后（:130-138），而且「DONE 只在还没有时才写」，是对的写法。
- 最小修法：把 `:113-124` 挪到 `:125-126` 之后（拿到锁再判断、再写 DONE）；拿不到锁的路径只打印到 stderr。

### E-6 DONE 的 rc 只反映汇总脚本，臂全失败也是 rc=0；公共任务书把这个脚本当样板【推断】

- 位置：`heading-guard/run.sh:75-77`（公共任务书 §6 点名的样板）、`swa-calib/run.sh:85-87`、`long-course/run_v2.sh:115-118`、`long-course/run.sh` 同构；`k1-control/run.sh:28` 的 DONE 取单臂 rc，但没有续跑逻辑（见 E-9）。
- 现象：`run_arm` 一律 `return 0`（`heading-guard/run.sh:70`），DONE 写的是 `summarize.py` 的退出码。
- 成立条件与例子：这些 11-10 的入口把 `CUDA_VISIBLE_DEVICES` 钉在 `GPU-dbc6435f-…`，11-11 19:23 重启后这张卡没有了。`uavrgb_infer` 没有 CPU 退路（`src/` 里搜不到 `cudaGetDeviceCount` 或 fallback），学生臂会在 StudentPolicy 握手阶段失败。用原命令续跑时，学生臂全部失败，DONE 仍然是 `rc=0 … 全部结束`。
- 为什么是缺陷：DONE 是排队与人工判读的唯一信号。`run_r2.sh:376-380` 的写法是对的：先逐臂查完整性，缺臂就 `finish 11`。
- 最小修法：照 `run_r2.sh:376-380` 在写 DONE 前逐臂调用完成判定，缺臂时 rc=11 并列出臂名。新入口应以 `run_r2.sh` 为样板，公共任务书 §6 的样板指向可以改。

### E-7 dagger_pipeline 的评测成功率分母是完成集，不是请求集【已核实，读代码】

- 位置：`tools/dagger_pipeline.py:230-238`：`n = len(rows)`，`rows` 只收 `per_episode`；被 `physics_verified=false` 剔除的集计入 `failed`（:223），不进 `n`。`success_rate = success / n`，Wilson 区间也用 `n`。
- 为什么是缺陷：违反「分母用请求集数」（公共任务书 §2、§6）。剔除集是 cf_sitl 失控保护触发的，多发生在飞得差的集，剔掉之后 SR 偏高。:207-209 的注释说「本轮只要 eval.json 在就继续汇总」，所以当轮 `summary.json` 里的数就是偏高的。
- 影响：3DGS DAgger（P1 信念软标签）线。该线目前是否还在用，未核实。
- 最小修法：`n_req = len(CFG["routes"]) * CFG["eval_eps_per_route"]`，SR 与 Wilson 区间都用 `n_req`，`failed` 另列（3 行）。

### E-8 四种「评完」口径并存；pipeline_common.eval_finished 把被 skipped 截短的评测当成完整【部分已复现】

- 位置与口径：
  - `tools/pipeline_common.py:194-208`：`failed == 0 and done > 0`，不和请求数比
  - `heading-guard/run.sh:59`、`swa-calib/run.sh:66`、`baselines-zeroshot/run.sh:112`：`episodes_done == 20 and failed_count == 0`
  - `run_r2.sh:270`、`action-latency/run.sh:117`、`long-course/run_v2.sh:93`：`done + failed == requested`
  - `driver.py:68-73`：`not failures and len(episodes) >= k`
- 已复现：`verify_small.py` 造了一份 `episodes_done=5, failed_count=0, skipped_count=15, partial=false`、请求 20 集的 eval.json，`pc.eval_finished` 返回 True（`verify_small.txt` 第 1 行）。`eval_closed_loop.py:908-911` 的 skipped 不计入 attempted，也不计入 failed。
- 为什么是缺陷：
  - 走 `pipeline_common` 的三条流水线会跳过这种被截短的评测。`dagger_gz_pipeline.reusable()`（:279）还会把它当作可复用的基线。
  - `run_r2` 的口径正好相反：只要出现 skipped 就永远判不完整，每次续跑都整臂重评。
  - `run_r2` 口径下，非迟到的基础设施失败（例如 gz 崩溃）算「评完」并计为不成功，R6 只重跑迟到集。预注册只覆盖了迟到与电池两种情况，其余失败没有写。
- 最小修法：`eval_finished` 增加 `episodes_requested` 比较（`done + failed + skipped == requested` 且 `skipped == 0`）；各入口的内联判定改为调用这个函数。

### E-9 采集与训练的半成品让「原命令续跑」直接失败，要人手删，与「不删除任何已有文件」冲突【已核实，读代码】

- 位置：`e1_pipeline.py:141-142`（采集目录非空就抛异常）、`:228-230`（训练目录非空且未完成就抛异常），不带 `--from` 时从 pilot 开始，第一步就会失败；`dagger_gz_pipeline.py:403-405`、`:464-466` 同样处理；`dagger_pipeline.py:100-108`、`:185-194` 对没有 `run.json` 的路线目录重新调用 `collect_3dgs`，而 `collect_3dgs.py:588-589` 遇到非空 `--out` 直接报错。`dagger_pipeline.py` 的文档（:18-19）却写着「断点续跑：同一 --out 重跑，完成的步骤跳过」。`k1-control/run.sh:23-27` 没有任何续跑处理，第二次起跑时 `eval_closed_loop` 会拒绝非空 `--out`。
- 为什么是缺陷：容器一天重启三次，每次打断采集都要人工改名才能继续。评测步已经会自动改名（`.failed-<时刻>`），采集与训练步没有，处理方式不一致。
- 最小修法：采集目录、训练目录、`collect_3dgs` 的路线目录都照评测步处理：自动改名成 `.incomplete-<时刻>` 再重跑，日志里记一行。

### E-10 演练与冒烟用的覆盖变量，正式运行既不拒绝也不记录【推断；sage3d 一例已发生】

- 位置：
  - `run_r2.sh:66-108` 的 `EVAL`、`SELFCHECK`、`NVIDIA_SMI`、`MACHINE_LOCK`、`GZ_PROBE`、`CPU_MAX_FILE`、`CPU_STAT_FILE`、`IDLE_*`、`MIN_CORES`、`LANES`、`OUT`、`SRC` 都可以用环境变量覆盖。`:155` 的开始日志与 `:194-198` 的 `inputs.txt` 都不记 `EVAL` 路径与 `SELFCHECK`。`rehearse.sh:17` 用 `export OUT=… EVAL=$HERE/stub_eval.py`，被 `source` 时会留在当前 shell。
  - `bearing-guard-3dgs/run.sh:31-50`（`WT`、`OUT`、`LANE_A/B`、`ROUTES`、`K`、`N_MAIN`）与 `sage3d-pilot/run.sh:46-60`（`SCENES`、`PER_SCENE`、`K`、`GATE_ARM`、`GATE_SR`）没有任何拒绝。
  - `rule-oracle-3dgs/run.sh:49-52`、`:121-124` 的 `OVERRIDDEN` 检查是对的写法。
- 已发生的例子：`exp-sage3d-pilot-20261110` 第三次起跑用了 `GATE_SR=0`（1c01078 的提交说明写明为此把 `GATE_SR` 改成可覆盖）。`config.json` 的 `gates[0].min_sr` 是 0.0，`summary.md` 第 0 节标题是「闸门（预注册 P1）」，写的是「≥ 0」、五个场景全部「通过」，DONE 写「闸门通过 [5 个场景]」。预注册的 0.8 门五个场景都没过（`gate_prereg.json`：0.50、0.67、0.71、0.71、0.58）。放行是主会话 20:59 的决定，记在 `docs/worklog/2026-11-10-1903-phase12-long-horizon.md:61`，结论没有错；错在实验目录里的产物前后矛盾。
- 为什么是缺陷：`EVAL` 残留会让正式运行接上桩评测器，产出格式合法的假 eval.json，而且 `run.log` 与 `inputs.txt` 里看不出来。
- 最小修法：所有正式入口照 `rule-oracle-3dgs` 的 `OVERRIDDEN` 写法处理：正式运行拒绝覆盖，冒烟与演练要显式写 `SMOKE=1` / `DRY_RUN=1`；生效的覆盖值写进 `inputs.txt` 与开始日志。预注册的门被人工放行时，汇总里同时列出预注册阈值与放行阈值。

### E-11 run_r2 在核对之前就覆盖 code_head.txt【已核实，读代码；低】

- 位置：`run_r2.sh:189` 先写 `code_head.txt`，`:190-191` 才核对冻结副本与 `CODE_HEAD` 是否一致。
- 现象：续跑时如果误带了另一个 `CODE_HEAD`，核对失败退出 rc=4，但 `code_head.txt` 已经换成错误的提交。下次不带变量续跑，会从这份错误的记录取 `CODE_HEAD`。`sim/` 不同时永远 rc=4；`sim/` 相同而 `experiments/` 不同时，会以「新提交」的名义用旧副本里的 `summarize_r2.py` 跑下去。`config.env` 保留了正确值，可以人工恢复。
- 最小修法：把 `:189` 挪到 `:192` 之后；或者 `code_head.txt` 已存在且与 `CODE_HEAD` 不同时，直接 `finish 4`。

### E-12 汇总与参数统计的 `r*` 通配会吃进 `r<N>.incomplete-<时刻>`【已复现模式；现有数据未受影响】

- 位置：`rule-oracle-3dgs/summarize.py:241`（`load_old`）、`rule-oracle-3dgs/yaw_step_stats.py:40`（冻结参数 `YAW_STEP_H10=31.7` 就来自这里）、`teacher-tol-3dgs/analyze.py:402`。`driver.py:156-157` 会生成 `r<N>.incomplete-<时刻>`。
- 已复现：`verify_small.txt` 第 2 行，`glob(.../r*/ep_*/metrics.json)` 同时命中 `r0/` 与 `r0.incomplete-231200/`。
- 现状：在 `exp-bearing-guard-3dgs-20261110`、`exp-sage3d-pilot-20261110`、`exp-rule-oracle-3dgs-20261110`、`exp-teacher-tol-3dgs-20261110` 下用 `find -maxdepth 4 -name '*.incomplete-*'` 都没有找到，已发布的数字不受影响。以后续跑过的目录会受影响：`yaw_step_stats` 会重复计数；`load_old` 按键建字典，正好由完整目录覆盖（`.` 排在 `/` 之前），这一处不受影响。
- 最小修法：通配改成 `r[0-9]*`，再过滤掉含 `.` 的目录名（1 行），或者按 `config.json` 的路线表逐个拼路径（`summarize.py:87-99` 已经是这种写法）。

### E-13 check.sh 的临时目录与字节码写在根分区【已核实；低】

- 位置：`tools/check.sh:40` 用 `mktemp -d`，没有设置 `TMPDIR` 时落在 `/tmp`；`check.sh` 与 `env.sh` 都不设 `PYTHONDONTWRITEBYTECODE`。
- 现状：根分区剩 3.0 GB（`df -h /`）；主仓库与 `wt/` 下有 11 个 `__pycache__`，合计 5.3 MB；`/tmp` 下有 5 个 `tmp.*` 残留（`trap … EXIT` 在被 SIGKILL 时不会执行）。现在的量不大。公共任务书 §4 的 check.lock 守护模板没有带 `TMPDIR`。
- 最小修法：`env.sh` 加 `export PYTHONDONTWRITEBYTECODE=1` 与 `TMPDIR="${TMPDIR:-$DATA_ROOT/uav_rgb/tmp}"`（2 行）。

## 3. 查过没有问题的地方

- flock 锁在容器重启后自动释放（内核随进程释放），锁文件残留不影响下一次拿锁。`machine.lock` 与各实验的 `.lock` 都用 `flock`，没有用「文件存在即占用」的写法。
- 半成品产物的原子性：
  - `preprocess.py:685-702` 的 `dataset.uavr` 用 mkstemp + `os.replace`，侧车同样处理；
  - `collect_parallel.py:551-556` 的 `collection_manifest.json`、`collect_3dgs.py:684` 的 `run.json` 都是原子写；
  - `eval_closed_loop.py:1413-1415` 的 `eval.json` 与 `driver.py:256-257` 的 `done.json` 不是原子写，但截断的 JSON 在所有续跑判定里都按「未完成」处理（`json.load` 失败即重跑），不会被当成结果；
  - `teacher_belief.py:565` 先写 npz 再写 stats，`dagger_pipeline.label()` 两个文件都在才跳过；
  - `audit.json` 不是原子写，截断时 `dagger_gz_pipeline.audit_dir` 会报错停下，不会放行。
- 训练完成判定看 `metrics.jsonl` 末行是否为 `done:`（`pipeline_common.py:181-191`、`dagger_gz_pipeline.py:173-182`），不看 ckpt 文件是否存在。
- 第二轮时延标定被重启打断后没有留下假的 DONE；续跑正确跳过 5 个已完成的臂，把两个半成品的目录、`.log`、`.cg` 一起改名（`run_r2.sh:281-284`）。cgroup 计数在重启后归零（`usage_usec` 从 5.48e10 回到 7.6e8），而每个臂的起止记录都落在同一次启动里，`.cg` 增量不会跨重启算错。
- `run_r2.sh` 的冻结代码核对（`git hash-object` 逐个比 `sim/`）与脚本自检（:180-185）逻辑正确；`DISP_CLAUSE` 续跑时与 `config.env` 比对。
- `rule-oracle-3dgs` 与 `sage3d-pilot` 的路线文件用 `cmp -s` 比对，不一致就拒绝续跑；`rule-oracle-3dgs` 要求本进程是进程组长（:107）。
- 信号处理：`pipeline_common.install_signal_handlers` 与 `terminate_children`、`dagger_gz_pipeline._on_term` 在 SIGTERM 时收掉子进程组，DONE 写真实 rc。
- Xorg `:20` 随容器重启自动拉起（审核时 PID 144，已运行 1:31），gz 软件渲染不依赖人工恢复。
- `TMPDIR=$OUT/tmp` 的残留很小（最大的是 `exp-action-latency-r2-20261111/tmp`，5.8 MB）。
- 当前没有活进程组的 PGID 撞上任何失效的 `run.pid`（逐个 `ps -g` 查过）。

## 4. 没有查的地方与原因

- 没有执行任何入口脚本、演练脚本（`rehearse.sh`、`rehearse_r2.py`）与 `check.sh`：规则禁止。E-3、E-5、E-6、E-10 是读代码得出的推断。
- `collect_parallel.py` 内部的重试与 staging（:836、:846 的 `os.replace`）只看了写盘方式，没有逐行读它被打断后的行为。`eval_closed_loop.py` 只读了续跑相关的写盘、skipped 与 partial 逻辑。
- `tools/compare_datasets.py` 不在 E 的文件清单里，没有读；它也调用 `pipeline_common`，E-8 的口径问题可能同样存在（未核实）。
- 主仓库以外的入口脚本（例如 `exp-rule-v2-3dgs-20261111` 用的 `wt/p12-k3`，11-11 23:53 到 00:12 起了 3 次）不在 main，没有审。
- `uavrgb_train` 自身的续训（RunOwner、`--resume`）属于 C 的范围，没有读。
- `dagger_pipeline.py` 现在是否还有人用，未核实；E-7 的影响范围因此未知。
- `experiments/*/test_*.py` 没有进 `check.sh`（只循环了 `sim/tools` 与 `tools`），属于 F 的范围，这里只记一句。
