# longcourse：长课程评测（K 个 base 柱阵块首尾相接，不训练，逐块存活率）

- 分支 / 工作树：`p12-longcourse`，`/root/claude-code-backup-20261028/code/wt/p12-longcourse`（基于 main 155e377，已含 `--heading-guard`）。
- 实验输出：`/root/rivermind-data/uav_rgb/exp-long-course-20261110/`；草稿：`/root/rivermind-data/uav_rgb/tmp/phase12-20261110/longcourse/`。
- 状态：代码、测试、教师冒烟完成；第一轮正式实验（第 5 节）在 20:46 因时序混杂被主会话中止，数据保留；第二轮（A* 栅格缓存修法）21:14:57 起跑，PGID 1558634，见文末「更新 2026-11-10 21:30」。

## 1. 代码改动

新任务 `course_base`：K（2–6）个 base 柱阵块沿 x 首尾相接，成功线 = 6.0 + 5.0·(K−1) m。

- 新增 `sim/tools/scene_course.py`：配置区在文件顶部，纯标准库，`build_scene(template, seed, scene_id, blocks, appearance)`、`--self-check`。
  第 0 块 = `scene_gen.sample_pillars(seed)` 原样；第 k≥1 块取子 seed `seed*1000 + k*100 + j`（j=0,1,…）里第一个同时满足「base 几何的 split 与第 0 块相同」「平移 k·5.0 m 后与已放置的柱子两两缝宽 ≥ `MIN_GAP`」的几何。
  meta 沿用 base 字段，另有 `geometry_version="uavrgb.course_base.v1"`、`split_policy="course_block0"`、`course={blocks, block_pitch_m, block_seeds, block_gids, base_geometry_version, block_goal_x}`；出生点同 base default。
- `sim/tools/split_manifest.py`：`SPLIT_POLICIES` 加 `course_block0`；冻结表 `COURSE_BASE_SALT = {"uavrgb.course_base.v1": "uavrgb.geom.v5"}`；`split_salt_for` 认这个版本串；
  `split_for_scene` 的 `course_block0` 分支用 `meta.course.block_gids[0]` 与 `base_geometry_version` 复算第 0 块的 split；缺 `course`、版本不符、课程版本不配该策略都抛 `ValueError`（不猜）。
- `sim/tools/scene_gen.py`：`--task course_base` + `--blocks`（与 `--density` 同样的互斥校验，范围 2–6，只支持 default 出生）；`build_scene(..., blocks=None)` 转给 `scene_course`。
- `sim/tools/collect_parallel.py`：`scene_gen_extra` 透传 `--blocks`；`task_geometry_version("course_base")` 返回课程版本串。
- `sim/tools/eval_closed_loop.py`：`--task course_base`、`--blocks`（互斥校验同上）；`SPL_TASKS` 加 `course_base`；有 `--blocks` 时 eval.json 顶层多 `blocks`，每集多 `block_goal_x`。
- `tools/check.sh`：`scene_gen.py --self-check` 之后加 `scene_course.py --self-check`。
- 新增测试：`sim/tools/test_scene_course.py`（15 项）、`sim/tools/test_course_summary.py`（12 项）。
- 实验入口：`experiments/2026-11-10-long-course/{run.sh,summarize.py,README.md}`。

偏离任务书的一处：任务书写 `collect_parallel.task_geometry_version` 让 course_base「走 base 的路径」。我让它返回课程版本串，理由是 eval.json 顶层的 `geometry_version` 要与 scene.json 一致，
否则课程结果与 base 的 K=1 结果在这个字段上无法区分。`teacher.task_of_scene` / `set_vehicle`、`eval_closed_loop.dedup_key`、`spl_clearance` 没改，课程 meta 本来就落在它们的 base 分支上
（在 `sim/tools` 里 grep 过调用方：`teacher.task_of_scene` 只看 `family` 与森林版本串，课程 meta 返回 `base`；`dedup_key` 对 base / course 都返回整张柱阵的 `geometry_gid`；`spl_clearance` 不看任务）。

没有 `--blocks` 时旧命令行与旧输出逐位不变，用测试锁住（见第 2 节 `DefaultUnchanged`）。

shortcut（已写进 `scene_course.py` 文件头与 `split_manifest.py` 注释）：训练打包没接课程。`tools/preprocess.py` 按 `KNOWN_GEOMETRY_VERSIONS` 拒收 `uavrgb.course_base.*`；
加 `--allow-frozen-geometry` 放开版本门后，episode manifest 不带 `course`，`split_for_scene` 抛错，仍拒收。这条用 `test_scene_course.py::TrainingPackingNotWired` 锁住。要用课程数据训练时再接。

## 2. 测试（先红后绿）

`sim/tools/test_scene_course.py` 覆盖：块 0 几何与 base 同 seed 逐项相同；各块 base split 相同（K=2/3/6）；跨块缝宽契约（含一个契约判得出「不过」的反例）；同 seed 同 K 字节级一致（含 CLI 两次写出逐字节相同）、K=3 课程的前两块 = K=2 课程；
20 个冻结 seed × K∈{2,3,4,6} 的 `split_for_scene` 全是 `test`；成功线 = 6.0 + 5.0·(K−1)（K=2..6）；6 个 seed 上 K=4 的 `teacher.astar_path` 从出生点到目标可解、K=3 的 SPL 最短路可算；
split 复算的 fail-closed 用例；`--blocks` 的互斥与范围（CLI rc=2）；`DefaultUnchanged`（6 组 base / forest / l1 / CLI 输出的 sha 与改动前相同，`scene_gen_extra` 无 `--blocks` 时为空）。

| 阶段 | 命令 | 结果 |
| --- | --- | --- |
| 红 1：`scene_course.py` 不存在 | `python sim/tools/test_scene_course.py`（输出 `…/longcourse/red1.txt`） | `Ran 14 tests in 0.140s`，`FAILED (failures=12)`（2 项默认不变用例已绿） |
| 红 2：生成器在、接线没做 | 同上（`red2.txt`） | `Ran 14 tests in 12.070s`，`FAILED (failures=3, errors=2)`：`build_scene() got an unexpected keyword argument 'blocks'`、`无法复算 split_salt：未知 geometry_version 'uavrgb.course_base.v1'`、`scene_gen_extra` 返回 `[]` 而不是 `['--blocks','3']`、`'course_block0' not found in SPLIT_POLICIES`、scene_gen CLI rc=2 |
| 绿 | 同上（`green1.txt`） | `Ran 14 tests in 16.004s`，`OK` |
| 补用例后终版 | 同上 | `Ran 15 tests in 21.331s`，`OK`（`python -X dev` 下无 ResourceWarning） |

`sim/tools/test_course_summary.py`（12 项，`Ran 12 tests in 0.109s OK`）覆盖逐块存活判据（通过 / 块 2 撞柱 / 块 1 越界后补过线不算 / 中途停 / stuck 不是碰撞而 contact+stuck 是 / 过线那一步的事件算 / 同步撞柱越界记撞柱）、
300 条随机轨迹上第 1 块通过与 `eval_closed_loop.summarize_episode` 的 K=1 判据逐条一致、Wilson 与 McNemar 数值、采集中止的集归块、整表汇总。这组测试在 summarize.py 写完之后才写，没有红的一步（见「没做什么」）。

`python sim/tools/scene_course.py --self-check`：`PASS（K∈{2,3,4,6} × 30 seed 同 seed 字节级一致；第 0 块同 base、各块同 split、缝宽契约与成功线全过；第 k≥1 块平均试 2.6 个子 seed，最多 35）`，用时 1.7 s。

## 3. 电池预算、教师冒烟与 K 的取舍

### 3.1 教师冒烟（K=3，2 个 seed，步数上限 240）

命令（脚本 `…/longcourse/smoke_teacher.sh`，输出 `…/exp-long-course-20261110/smoke/teacher-k3-s{1,113}`）：
`eval_closed_loop.py --policy teacher --task course_base --blocks 3 --split test --seeds <s> --episodes 1 --steps 240 --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --corridor-half-width 3.0`

| seed | 结果 | 步数 | 过 3 条线的步号（复位后仿真秒） | 末动作终点 | 末 20 步高度 z | 末 10 步均位移 / 前 10 步均位移 | SPL | wall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 成功，无碰撞，`y_max` 1.20 m | 81 | 36 (47.8 s) / 60 (69.4 s) / 80 (87.4 s) | 87.4 s | 1.98–2.00 m | 0.247 / 0.102 m | 0.956 | 125.5 s |
| 113 | 成功，无碰撞，`y_max` 1.07 m | 83 | 40 (49.8 s) / 61 (68.7 s) / 82 (87.6 s) | 87.6 s | 1.98–2.00 m | 0.224 / 0.113 m | 0.941 | 125.5 s |

出处：两个 `eval.json` 与 `episodes/seed_N/{oracle,actions}.jsonl`；`dynamics_profile` 都是 `cf_sitl_v1:1bc8fedf:lib31`，`metric_version` 是 `sim09.v2.coast1s.corridor3m`，`block_goal_x=[6.0,11.0,16.0]`，`skipped_count=0`，`failed_count=0`。
末 10 步的每步位移 0.22–0.25 m 与全程前进的正常值（0.25 m）一致，没有推力衰竭的减速；全程 z 在 1.98–2.02 m 之间。

### 3.2 电池参照

- 起始 SoC 0.5（`sim/dynamics/e1_mini.json` 的 `sim.value.battery_soc_init`）。11-09 探针 `tmp/analysis-20261109/probe/battery_fwd_s2.log`（时间从 READY 起算，READY 在复位后约 8.1 s）：
  z 在 101.5 s 仍是 1.986 m，106.5 s 起降到 1.947 m、111.5 s 1.921 m、116.5 s 1.912 m，123.0 s 进 FAILSAFE。换成复位后时间：掉高起点约 115 s、FAILSAFE 约 131 s。
- `collect.py` 里固件进入 FAILSAFE 会让采集进程报错退出，这一集进 eval.json 的 `failed` 而不在 `per_episode`；`summarize.py` 按中止前的轨迹把它归到相应的块，列为「采集中止」，计入没通过。
- 学生比教师慢：今天 K=1 评测里过线集的步数（`exp-heading-guard-20261110/guard2-calswa-s*`、`exp-swa-calib-20261110/calswa-s*`、`guard2-s*` 的 `eval.json`）：

| 臂（K=1，60 集） | 过线集数 | 过线集步数 均值 / 中位 / p90 / 最大 | 按 3 倍估 K=3 的步数 | 估算超过 95 步的占比 |
| --- | --- | --- | --- | --- |
| calswa + 护栏 2 | 50 | 32.7 / 31 / 39 / 71 | 约 98 | 0.32 |
| calswa 无护栏 | 39 | 30.2 / 29 / 34 / 35 | 约 91 | 0.15 |
| ckpt_best + 护栏 2 | 47 | 34.3 / 33 / 40 / 52 | 约 103 | 0.55 |

  教师 K=3 是 81–83 步（K=1 约 27 步）。按每步约 0.9 s 仿真时间加起飞前 13.5 s，学生 K=3 的典型一集约在复位后 100 s 结束，在掉高起点（约 115 s，约 112 步）之前；
  超过约 112 步的集进入掉压段，约 130 步进 FAILSAFE。这是「3 倍估算」，没有按 K=3 实测过学生（正式实验会给出每集 `sim_s`、末 10 步最低高度与采集中止数）。

### 3.3 K 的取舍

教师冒烟正常（过 3 条线、高度与位移无衰减、87.5 s 结束），按任务书的规则正式实验用 K=3。K=2 不做，除非 K=3 的学生臂里「采集中止」与「掉高」集占比高到无法把电池与导航失败分开（判据与数字看 `summary.md` 的第 3、5 节）。

### 3.4 起始 SoC 能否改成 1.0（只调查，不改）

- 能，不用重编库。`battery_soc_init` 不是编进库里的常量：`sim/cf_sitl/src/plant.h` 的 `Params::load` 读 `sim.value.battery_soc_init`（第 60 行），`reset()` 把 `soc = P.soc0`（第 146 行），每次 reset 都从 JSON 来。
  没有环境变量或命令行专门改它；插件按 SDF 里的仓库相对路径 `sim/dynamics/e1_mini.json` 解析，根目录取 `$UAVRGB_REPO_ROOT`（`sim/cf_sitl/gz/CfSitlSystem.cc` 的 `ResolveParams`，`collect_parallel.gz_env` 会设），
  Python 侧 `cf_sitl_exec.DEFAULT_PARAMS` 与 `collect.py --cf-sitl-params` 读同一路径。做法：把参数文件拷到另一个根目录的 `sim/dynamics/e1_mini.json`，改 `battery_soc_init` 为 1.0，评测进程设 `UAVRGB_REPO_ROOT` 指过去。
  库自带的 `sim/cf_sitl/selftest/scenario_battery.cpp` 也是这么做的（SoC ≠ 1.0 时写一份改过的参数副本再 init）。
- 会改指纹：`dynamics_profile = cf_sitl_v1:<sim.value 在内的各项 value 规范化 JSON 的 sha8>:lib31`（`cf_sitl_exec.params_values_sha256`）。
  用 `cf_sitl_exec` 的函数算：当前文件 `1bc8fedf`（与今天结果相同），SoC 改 1.0 后 `06e5f017`，只改注释文本不变（`1bc8fedf`）。所以结果会落到新指纹 `cf_sitl_v1:06e5f017:lib31`，与今天的 K=1 结果不是同一动力学档。
- 动力学也会变：推力 `T = 0.373·d^1.5·(V_bus/3.7)^1.5`（`e1_mini.json` 的 `thrust_model`），满电开路电压 4.20 V 对 0.5 SoC 的 3.82 V，同占空比推力约高 `(4.20/3.82)^1.5 ≈ 1.15` 倍。
  闭环行为（悬停占空比、控制器余量、学生成功率）会不会变，没有跑过，未核实。
- 正式实验不改起始 SoC，指纹保持 `cf_sitl_v1:1bc8fedf:lib31`，与今天的 K=1 参照同一档。

## 4. 实验设计（预注册写在 `run.sh` 文件头）

- 20 个冻结 seed `1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654`；`--task course_base --blocks 3 --split test --steps 240 --hold 1.0 --window-from observation --dynamics cf_sitl --vehicle e1_mini --corridor-half-width 3.0`，`--episode-timeout 2520`；推理二进制 `$D/bin-snapshots/main-97ae04b/uavrgb_infer`。
- 10 个臂，按优先级：① `teacher`（20 集）；② `guard2-calswa-s{0,1,2}`（calswa 权重 + `--heading-guard 2`，60 集，主臂）；③ `calswa-s{0,1,2}`（无护栏，60 集）；④ `guard2-s{0,1,2}`（原始 `ckpt_best` + 护栏，60 集，次要）。5 路并行。
- K=1 参照（今天同权重同 seed）：calswa + 护栏 50/60（0.83），calswa 无护栏 39/60（0.65），ckpt_best + 护栏 47/60（0.78）。
- 预测与证伪（原文在 `run.sh`）：P1 calswa + 护栏的 S_K 在 0.83^K ± 0.15 内（K=3：0.57）；明显低于 = 存在独立块模型之外的长航程失败机制，明显高于 = 风险集中在第一排柱子。
  P2 无护栏 S_K ≤ 0.65^K + 0.15 且失败以越界为主，护栏对无护栏的配对 McNemar（S_K）b > c，b ≤ c 则证伪。P3 第 1 块与今天 K=1 同权重同 seed 的逐集一致率 ≥ 70%（自己重跑约 76%）。教师必须 ≥ 19/20。
- 逐块存活：第 k 块通过 = 首次 `pose_gt.x ≥ block_goal_x[k]` 且到这一步（含）没有接触碰撞、`|y| ≤ 3.0`；`S_k` = 通过前 k 块的集数 / 20（或 60）。`summarize.py` 输出各变体 S_1…S_K（含 Wilson）、条件存活、与独立块模型的对照、失败类型按块（撞柱 / 越界 / 未到线 / 采集中止）、配对 McNemar（S_1 与 S_K）、步数与仿真秒、第 1 块对 K=1 的逐集一致率。

## 5. 正式实验的启动与监控

- 起跑：2026-11-10 20:10:56，守护进程，`run.sh` 从 `$OUT` 起（`$OUT=/root/rivermind-data/uav_rgb/exp-long-course-20261110`），**PGID 1372765**（`$OUT/run.pid`）。
  起跑时把 `git HEAD`（`f0fe697`，代码部分与其后的 `c2d9520` 相同，后者只加文档）用 `git archive` 冻结到 `$OUT/code`，之后只用这份副本；`$OUT/inputs.txt` 记了 HEAD、关键脚本与全部权重、推理二进制的 sha256。
- 10 个臂、5 路并行（`xargs -P5`）：先派发的 5 个是 `teacher`、`guard2-calswa-s0/1/2`、`calswa-s0`，其余 5 个（`calswa-s1/s2`、`guard2-s0/1/2`）排在后面。机器上同时有 7 个 gz（我的 5 个 + ruleoracle 的 2 个），没超过 7。
- 进度：`tail -f $OUT/run.log`；每臂 `grep -E '^\[seed' $OUT/<臂>.log`（每集一行，含步数、是否碰撞、是否成功）。`eval.json` 只在一个臂评完时才写，所以 `summarize.py` 中途只看得到已评完的臂，其余标「未完成」。
- 停：`kill -- -1372765`（只按 PGID，不用 `pkill -f`）；停后 `ps -eo pid,pgid,etime,cmd | grep -E "gz sim|eval_closed_loop"` 确认没有残留。
  注意 `gz sim`、`collect.py`、`uavrgb_infer` 各自在 `start_new_session` 起的独立进程组里（`ps -eo pid,ppid,pgid,sid,cmd` 看到它们的 PGID 不是 1372765），`kill -- -1372765` 不直接覆盖它们；
  清理靠 `eval_closed_loop.py` 的 SIGTERM 处理（第 1205 行注册，第 781 行 `stop_group` 兜底杀 collect）。这条路径我读了代码，没有实测（实测要杀正式实验），所以停后的 `ps` 确认不能省。
- 续跑：同一条命令重起（`cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &`）；评完的臂跳过，半成品挪成 `<臂>.partial-<时刻>`，冻结代码副本复用。
- 结束后：`$OUT/DONE`（rc + 时间）与 `$OUT/summary.md` 自动生成；重新汇总用 `/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python $OUT/summarize.py`。
- ETA：20:19:29 时 5 个臂的 `<臂>.log` 共记 18 集（teacher 3、guard2-calswa-s0 4、guard2-calswa-s1 4、guard2-calswa-s2 3、calswa-s0 4），自 20:10:56 起约 2.1 集/分；总共 200 集，剩 182 集按此速度约 87 分钟，
  加上最后几个臂跑不满 5 路的尾巴，预计 21:50–22:30 结束（撞柱的集提前结束，教师臂每集最长；按已有速度估，未核实）。

### 5.1 前几集的核对（seed 1，每臂第 1 集；`summarize.py` 的函数按 oracle / actions 算，`block_goal_x` 取 6/11/16）

| 臂 | 步数 | 复位后末动作终点 | 末 10 步最低高度 | 逐块通过 | 失败归属 | 护栏改写的步 |
| --- | --- | --- | --- | --- | --- | --- |
| teacher | 87 | 91.0 s | 1.98 m | 3 块全过 | 无 | 0 |
| guard2-calswa-s0 | 76 | 81.4 s | 1.98 m | 3 块全过 | 无 | 0 |
| guard2-calswa-s1 | 46 | 55.4 s | 1.99 m | 过第 1 块 | 块 2 撞柱 | 1 |
| guard2-calswa-s2 | 63 | 70.1 s | 1.97 m | 过前 2 块 | 块 3 撞柱 | 1 |
| calswa-s0（无护栏） | 22 | 32.6 s | 1.99 m | 都没过 | 块 1 撞柱 | 0 |

每个臂的 `episodes/seed_1/` 都有 `actions.jsonl`、`oracle.jsonl`、`episode_manifest.json`；manifest 里 `dynamics_profile=cf_sitl_v1:1bc8fedf:lib31`、`split=test`、`geometry_version=uavrgb.course_base.v1`、`split_policy=course_block0`，
与今天 K=1 结果同一指纹。教师第 1 集 87 步（冒烟同 seed 是 81 步，SITL 在不同负载下不逐位复现，步数差几步），成功。这 5 集只用来确认产物正常，不作结论。

### 5.2 第 2 集的核对（seed 113，同一口径，20:17 时读到的）

| 臂 | 步数 | 复位后末动作终点 | 末 10 步最低高度 | 逐块通过 | 失败归属 | 护栏改写的步 |
| --- | --- | --- | --- | --- | --- | --- |
| teacher | 80 | 85.0 s | 1.98 m | 3 块全过 | 无 | 0 |
| guard2-calswa-s0 | 92 | 95.8 s | 1.98 m | 3 块全过 | 无 | 5 |
| guard2-calswa-s1 | 51 | 58.3 s | 1.97 m | 过第 1 块 | 块 2 撞柱 | 6 |
| guard2-calswa-s2 | 77 | 82.1 s | 1.98 m | 3 块全过 | 无 | 1 |
| calswa-s0（无护栏） | 98 | 101.3 s | 1.98 m | 都没过 | 块 1 内越界（y 超出 ±3.0 m） | 0 |

- 两个 seed 下教师都过 3 块（87 / 80 步），与冒烟（81 / 83 步）同量级；10 集的指纹仍是 `cf_sitl_v1:1bc8fedf:lib31`。
- 目前出现的最长一集是 calswa-s0 seed 113：98 步、复位后 101.3 s，低于电池掉压起点（复位后约 115 s，第 3.2 节），末 10 步高度 1.97–1.99 m，没有掉高。
- calswa-s0 seed 113 的 `max_x`=16.17（过了终线 16.0），无碰撞，`eval_closed_loop` 日志记 `success=False`。官方判据的 `y_max` 取全集 `|y|` 的最大值（`eval_closed_loop.py` 第 500–504 行），超过 3.0 m 即整集失败；
  我的逐块判定按第一个坏事件归属，这一集在穿过第 1 条线之前就越界，所以 S_1 也不算它。这是预注册定义的行为，不是 bug。
- 两个臂已经有第 3 集（seed 122）：guard2-calswa-s0 撞柱（29 步，`max_x`=4.22，在块 1 内）、guard2-calswa-s1 成功（96 步）；这只是日志里读到的，不作结论。

### 5.3 20:19 读到的第一个采集失败（calswa-s0，seed 122）

- `calswa-s0.log` 记 `[seed 122] 采集失败 rc=1，占请求名额、不计入成功`；`episodes/seed_122/` 的 `actions` / `oracle` 留着：137 步，复位后末动作终点 135.5 s。
- 轨迹：第 42 步（复位后 50.9 s，x=4.72 m，y=-3.03 m）首次 `|y|>3.0`；至少从第 96 步起 y 在 -11 m 以外（x 14 m 上下游荡）；`block_outcomes` 的首个坏事件是块 1 内越界。z 的走势（复位后秒数）：99.5–115.7 s 为 1.98–1.99 m，118.4 s 1.95 m，121–128 s 1.92–1.93 m，130.1 s 1.90 m，
  131.0 s 1.86 m，131.9 s 1.76 m，132.8 s 1.58 m，133.7 s 1.26 m，134.6 s 0.79 m，135.5 s 0.14 m。
- 这条 z 曲线与第 3.2 节的电池参照一致（约 115 s 起下沉，约 131 s 后塌陷），是正式实验里第一次直接看到。z 首次低于 1.97 m 在第 116 步（117.5 s），比首次越界晚约 67 s，不改变这一集的块归属（块 1 越界）。
- 采集失败的确切原因要等 `eval.json` 的 `failed[].stderr_tail`（臂评完才写）；是不是固件 FAILSAFE，未核实。一集不作结论。之后看汇总里各臂复位后超过 110 s 的集数与「采集中止」数，判断 K=3 下学生臂的电池混淆有多大。

## 做了什么

- 在 `p12-longcourse` 分支提交 3 个：`47c187b feat(scene)`（`scene_course.py`、`split_manifest` 的 `course_block0`、`scene_gen` / `collect_parallel` / `eval_closed_loop` 接线、`check.sh` 自检、`test_scene_course.py` 15 项）、
  `f0fe697 feat(experiments)`（`run.sh`、`summarize.py`、`README.md`、`test_course_summary.py` 12 项）、`c2d9520 docs`（过程文档 `docs/worklog/2026-11-10-2011-long-course.md`，用户全局规则要求）。
- 测试先红两次再绿；`check.sh --python-only` rc=0（日志 `…/longcourse/check_python_only.log`）。
- 教师冒烟 K=3 两个 seed 全过，K 定为 3；电池预算、起始 SoC 调查的结论写在第 3 节。
- 正式实验起跑：预注册写在 `run.sh` 头部，10 个臂 × 20 集，PGID 1372765；前 5 个臂的前 2 集（seed 1、113）产物核对正常（第 5.1、5.2 节）；第 5.3 节记了第一个采集失败的集（z 曲线与电池探针一致，失败原因未核实）。

## 没做什么（每条带原因）

- 训练打包没接课程（`tools/preprocess.py`、`audit_dataset.py`）：任务书明确本轮不接；现状是 preprocess 拒收课程数据，用 `TrainingPackingNotWired` 锁住，`shortcut` 注释在 `scene_course.py` 与 `split_manifest.py`。
- 没等实验跑完、没有任何 S_k 结论：任务书要求起跑并确认前几集正常后返回；本报告里没有 P1/P2/P3 的判定。
- 没有按 K=3 实测学生的电池余量：学生臂就是正式实验；第 3.2 节的学生步数是按 K=1 步数 3 倍的估算，估算与实测可能不同。
- 没有跑起始 SoC=1.0 的闭环，也没有改参数文件：任务书只要求调查；改了会换指纹，且闭环行为变不变未核实。
- 没有给 K=2 起一组备份臂：教师冒烟正常就按规则用 K=3；如果 K=3 的采集中止占比高到分不开电池与导航失败，再由主会话决定补 K=2（`run.sh` 里改 `K=2`、`OUT` 换目录即可）。
- `test_course_summary.py` 没有先红后绿：它在 `summarize.py` 之后写，只有绿的输出。
- 没有改 `collect_parallel.py` 自己的 `--task` 选项与 `--blocks` 参数：任务书只要求 `scene_gen_extra` 透传与几何版本函数；长课程只给评测用，采集器不开这个任务。
- 没有对 `teacher.py`、`collect.py` 做任何改动：课程 meta 落在它们已有的 base 分支上，测试里确认了 `teacher.task_of_scene`、A* 余量档与成功线口径。

## 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| `python sim/tools/test_scene_course.py`（红 1，`scene_course.py` 不存在） | `Ran 14 tests`，`FAILED (failures=12)`（`…/longcourse/red1.txt`） |
| 同上（红 2，生成器在、接线没做） | `Ran 14 tests in 12.070s`，`FAILED (failures=3, errors=2)`（`red2.txt`） |
| 同上（绿） | `Ran 14 tests in 16.004s`，`OK`（`green1.txt`）；补用例后终版 `Ran 15 tests in 21.331s`，`OK` |
| `python sim/tools/test_course_summary.py`（`-W error::ResourceWarning`） | `Ran 12 tests in 0.109s`，`OK` |
| `python sim/tools/scene_course.py --self-check` | PASS（K∈{2,3,4,6} × 30 seed；平均试 2.6 个子 seed，最多 35） |
| `UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only`（终版代码） | rc=0，`…/longcourse/check_python_only.log` 末行 `rc=0` |
| 教师冒烟 K=3，seed 1 / 113 | 成功；81 / 83 步；过线步号 36/60/80 与 40/61/82；末动作终点 87.4 / 87.6 s；末 20 步 z 1.98–2.00 m；SPL 0.956 / 0.941 |
| 起始 SoC 调查 | 读 `plant.h` / `CfSitlSystem.cc` / `cf_sitl_exec.py`；用 `cf_sitl_exec.params_values_sha256` 算出指纹 `1bc8fedf` → `06e5f017`；闭环影响未核实 |
| 正式实验起跑与前 10 集 | `ps` 显示 PGID 1372765 在跑（20:16:51 时 etime 05:54）；`run.log` 5 个臂已开始；每臂 seed 1、113 的 `actions/oracle/manifest` 在，指纹全是 `cf_sitl_v1:1bc8fedf:lib31`；seed 1 的 split / 几何版本核对正确（第 5.1、5.2 节） |
| 电池参照对真实集 | calswa-s0 seed 122 一集：z 在复位后约 115 s 起下沉、131 s 后塌陷，与探针时间线一致；失败原因未核实（第 5.3 节） |
| 正式实验结果 | 未验证：实验在跑，预计 21:50–22:30 结束 |

---

## 更新 2026-11-10 21:30（第二轮：时序混杂的定位、标定、修法与起跑）

起因：主会话在 20:46 按 PGID 停掉第一轮（`kill -- -1372765`），原因是 `--window-from observation` 下动作窗口从观测帧起算，观测到派发的仿真时延越长，动作实际生效越短，K=3 比 K=1 长。
第一轮数据保留在 `/root/rivermind-data/uav_rgb/exp-long-course-20261110/`，没有动。证据文件都在草稿目录 `longcourse/`（下文只写文件名）。

### 6. 复核主会话的时序表

`timing_audit.py` 读 `episodes/seed_*/{episode_manifest,actions,oracle}`，不重跑仿真（输出 `timing_audit_v1.txt`）。生效时长 = `performance.action_sim_s / steps`，每集一个值，取集间中位数。

| 分组 | 集数 | 生效时长中位 s/步 | 集间最小 | 观测→派发 s | 帧龄 s | 位姿-RGB 戳差 ms | io 墙钟 s | 前进 m（集内中位的中位） | 转向 ° |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=1 护栏臂（下午） | 60 | 0.871 | 0.848 | 0.137 | 0.113 | 5.0 | 0.048 | 0.256 | 26.6 |
| K=1 规则 v0（今晚） | 56 | 0.890 | 0.885 | 0.118 | 0.113 | 5.0 | 0.018 | 0.262 | 27.4 |
| K=3 教师冒烟 | 2 | 0.782 | 0.777 | 0.226 | 0.113 | 5.0 | 0.164 | 0.208 | 22.7 |
| K=3 教师正式（第一轮） | 16 | 0.791 | 0.780 | 0.216 | 0.112 | 5.0 | 0.157 | 0.212 | 23.1 |
| K=3 护栏臂正式（第一轮） | 57 | 0.770 | 0.642 | 0.239 | 0.113 | 5.0 | 0.188 | 0.195 | 23.8 |
| K=3 calswa 无护栏（第一轮） | 25 | 0.745 | 0.638 | 0.264 | 0.113 | 5.0 | 0.235 | 0.178 | 22.7 |

主会话给的生效时长全部复现：0.871 / 0.890 / 0.782 / 0.791 / 0.770，K=3 护栏臂集间最小 0.642。
列的定义（`timing_audit.py` 文件头）：观测→派发 = 观测位姿到窗口末位姿的时长（hold + 末帧过冲，取自 `oracle.jsonl` 的 `action_start_ns` / `action_end_ns`）− 生效时长；帧龄 = 上一步窗口末位姿戳 − 本步 RGB 帧戳，即上一窗口结束时本步用的帧已经老了多少仿真秒；前进 / 转向两列是集内中位数的集间中位数，`summarize.py` 的池化口径见第 7、9 节。
帧龄与位姿-RGB 戳差在所有分组里相同，与 K 无关；随 K 变的只有 `io`，生效时长随 `io` 单调下降。

**6.1 延迟出在哪一段（直接证据）。** `make_diagcode.py` 把第一轮冻结的 `collect.py` 复制一份（`diagcode/`），在每步 5 个位置打点（等新帧、取深度并解码、配对位姿、策略、取派发位姿戳），原副本不动。
4 个任务同时跑，seed 1 各一集（`diag_run.sh`，`diag_summary.py` 汇总，输出 `diag_v1_table.txt`）：

| 任务 | 步数（去首步） | 等新帧 ms | 深度解码 ms | 位姿配对 ms | 策略段 ms | 策略段折仿真 ms | 帧龄（到手时）ms | 观测→派发 仿真 ms | io 合计 ms | 推理 ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=1 护栏 | 28 | 0.0 | 6.6 | 0.7 | 37.5 | 20.5 | 115.9 | 137.0 | 44.8 | 0.8 |
| K=1 教师 | 29 | 0.0 | 7.6 | 0.7 | 32.6 | 18.1 | 113.6 | 131.9 | 41.0 | - |
| K=3 护栏 | 85 | 0.1 | 7.4 | 0.7 | 144.7 | 105.0 | 113.8 | 218.6 | 152.9 | 1.0 |
| K=3 教师 | 81 | 0.0 | 8.1 | 0.7 | 147.0 | 105.2 | 114.7 | 220.1 | 156.0 | - |

- 观测→派发 = 帧龄 + 策略段折仿真：K=1 护栏 115.9 + 20.5 = 136.4（实测 137.0），K=3 教师 114.7 + 105.2 = 219.9（实测 220.1）。各段墙钟之和等于 io（K=3 教师 0.0 + 8.1 + 0.7 + 147.0 + 0.0 = 155.8，io 156.0）。
- 等新帧：0 ms，新帧在要用时已经到了，不是慢渲染或晚帧。深度解码 7–8 ms、位姿配对 0.7 ms、学生推理 0.8–1.0 ms，K=1 与 K=3 相同。
- 写盘：不在这段时延里。`collect.py` 在 `sil.dispatch` 之后才写 RGB / 深度 PNG（源码注释「PNG 落盘落在 hold 窗口内」）。
- 帧龄 114–116 ms 与 K 无关（来源没查）。策略段最短的规则 v0（io 0.018 s）观测→派发 0.118 s，生效时长 0.890 s，所以实测的上限在 0.89 附近。
- 差额在策略段：K=3 的 A* 每步约 145 ms，K=1 约 35 ms，多出的约 110 ms 墙钟按 RTF 0.8 折成约 85 ms 仿真时延，与 0.22 s 和 0.13 s 之差（约 0.085 s）一致。教师臂里 A* 是动作本身；学生臂里它是影子教师标签（写进 `actions.jsonl` 的 `action_expert`），
  不参与控制，但每步都算：K=3 护栏臂策略段 144.7 ms，推理只占 1.0 ms。
- A* 离线计时（`astar_time.py`，`astar_time_v1.txt`）：K=1 8 根柱，每个位姿约 20 ms；K=3 25 根柱，出生位姿 207–249 ms（5 次取中位），沿航线均值 48–63 ms。
  cProfile（`astar_profile.py`，`astar_profile_v1.txt`，第一轮代码，K=3 seed 1 共 30 次 `plan_action_global` 调用 6.27 s，含 cProfile 自身开销）：`astar_path` 累计 4.47 s，其中 `_pillar_blocked` 累计 3.87 s（87%），来自逐格逐柱的 `is_free` → `_point_blocked`。

**6.2 主会话假设的对照结论。** 慢软件渲染 / 晚到的相机帧：帧龄、位姿-RGB 戳差与 K 无关，等新帧 0 ms，不成立。辅助相机（chase / top）渲染：见第 7 节，去掉辅助相机后生效时长反而更短，不成立。
解码 / 推理 / 写盘：7–8 ms、1 ms、派发之后，不是差额。成立的是策略段里的 Python A*。

**6.3 `summarize.py` 池化口径下的 K=1 参照。** 60 集 K=1 护栏臂：生效时长中位 0.871 s，稳态前飞位移池化均值 0.258 m，转角池化均值 26.1°（第一轮目录 `summary.md` 第 8 节的参照行）。同权重无护栏 K=1 参照（`exp-swa-calib`，60 集）：生效时长 0.868 s，前飞 0.249 m，转角 26.6°。

### 7. 标定（教师，seed 1 / 113 各 2 集，2 条车道；`calibrate_timing.sh`，输出 `exp-long-course-v2-20261110/calib/`）

读表命令：`python summarize.py --tol 0.02 --timing "标签=目录1,目录2" ...`（`calib/timing_table_all.md`）；观测→派发、io、RTF 来自 `timing_audit.py`（`calib_v1_table.txt`、`calib_v2_table.txt`）。窗口 = 主会话给的 0.854–0.888。

| 配置 | 代码 | 生效时长中位 s/步 | 集间最小 | 对 0.871 | 稳态前飞 m | 转角 ° | 观测→派发 s | io 墙钟 s | RTF | 窗口 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=3 保留辅助相机（第一轮配置） | 第一轮 | 0.784 | 0.781 | −9.9% | 0.205 | 23.1 | 0.223 | 0.155 | 0.82 | 不在 |
| K=3 `--no-aux-cameras`（任务书手段 1） | 第一轮 | 0.763 | 0.755 | −12.4% | 0.197 | 23.1 | 0.246 | 0.159 | 0.89 | 不在，更差 |
| K=3 `--no-aux-cameras --camera-rate 5`（任务书手段 2） | 第一轮 | 0.643 | 0.634 | −26.2% | 0.157 | 18.5 | 0.364 | 0.164 | 1.00 | 不在，更差 |
| **K=3 A* 栅格缓存，保留辅助相机** | 第二轮 | **0.872** | 0.871 | +0.1% | 0.258 | 27.0 | 0.136 | 0.048 | 0.78 | **在** |
| K=1 保留辅助相机 | 第一轮 | 0.875 | 0.873 | +0.5% | 0.264 | 26.6 | 0.133 | 0.043 | 0.83 | 在 |
| K=1 `--no-aux-cameras` | 第一轮 | 0.865 | 0.862 | −0.7% | 0.260 | 26.6 | 0.144 | 0.048 | 0.87 | 在 |
| K=1 A* 栅格缓存，保留辅助相机 | 第二轮 | 0.887 | 0.885 | +1.8% | 0.273 | 27.6 | 0.120 | 0.030 | 0.76 | 在 |

- 手段 1 没用：去掉辅助相机后仿真更快（RTF 0.82 → 0.89），同样的 A* 墙钟折成更长的仿真时延（0.223 → 0.246 s），生效时长反而从 0.784 降到 0.763。K=3 与 K=1 的差距 0.091 s（保留）/ 0.102 s（去掉）基本不变。
- 手段 2：试了一次。先读了 `scene_gen.py --camera-rate` 的帮助文字：「E1 模板 10 Hz：每个 hold 窗口只消费 1 帧，两路同频配对误差≈0」（`collect.py` 里没有 camera-rate 的说明）。它没有给出降频能缩短动作窗口损失的依据；降频让帧更旧，RTF 更高。实测帧龄从 0.113 升到 0.212 s，生效时长 0.643，更差。
- 手段 3（任务书没列，针对 6.1 的原因）：A* 栅格缓存，见第 8 节。K=3 生效时长 0.872，与 K=1 下午参照 0.871 相差 +0.1%；稳态前飞 0.258 与参照 0.258 相同；转角 27.0° 对 26.1°（+3.5%，只有 2 集，样本小；第二轮正式各臂的转角见第 9 节）。
- K=1 在第二轮代码下 0.887（+1.8%）：A* 变快后 K=1 的延迟也缩短。第二轮 K=3 的延迟 0.136 s 与 K=1 第一轮代码的 0.133 s 一致，所以 K=3 与下午的 K=1 参照可比；再优化 A* 只会让 K=3 超过参照，没有做。
- 每个配置只有 2 集，生效时长在同配置内的离散度很小（集间最小与中位差 0.001–0.009 s），但样本数小，不做显著性判断。

### 8. 修法：`teacher.astar_path` 栅格可走性按场景缓存（提交 `fcbf2fe`）

- 为什么：柱子在一集内不变，A* 每步从当前位姿重跑，逐格判断「这格能不能走」的结果只取决于柱子几何、余量和步长，与起点无关，却每次重算。
  缓存键 = (余量, 步长, 全部柱子的 x / y / 半径 / 尺寸)；最多留 4 个场景。起点格的「强制可走」只对单次调用有效，不写进共享缓存（起点在余量内时，别的起点不能借用）。
- 路径逐位不变：`sim/tools/test_astar_cache.py` 7 项：第二次调用复用格子（`_point_blocked` 调用数低于第一次的 20%）；与 verbatim 复制的旧实现逐位相同（seed × K=1 / K=3 × 余量 0.4 / 0.5 × 6 个随机起点，多于 100 个用例）；起点在余量内不被缓存为可走；终点被堵 / 越界；圆柱与两个场景交替；柱子移动后结果与旧实现一致；缓存有上限。
  先红：2 项失败（`test_cache_is_bounded`：`teacher._FREE_CACHE 不存在`；`test_second_call_reuses_cells`：第二次调用仍重算 1390 / 3252 格），`red_astar.txt`；后绿：7 项 OK，`green_astar.txt`。
- 已有测试：`test_teacher_policy`、`test_teacher_costmap`（44 项）、`test_teacher_kino`、`test_teacher_belief`、`teacher.py --self-check` 通过；`check.sh --python-only` rc=0。
- 离线 A* 计时（`astar_time_v2.txt`，每位姿 5 次取中位）：K=3 出生位姿 249 → 60 ms（seed 1）、207 → 45 ms（seed 113）；沿航线均值 63 → 26 ms、48 → 16 ms；K=1 均值 16–19 ms，与之前约 20 ms 持平。
  cProfile（`astar_profile_v1.txt` 后半）：K=3 `astar_path` 累计 4.47 → 1.79 s。仍有一部分时间在 `plan_action_global` 里 A* 以外，没有继续优化（原因见第 7 节倒数第二条）。
- 没有开关：缓存只改耗时，不改任何返回值，默认就开；没有新开关需要「默认关闭行为不变」的锁定，路径等价由上面第二条测试锁定。

### 9. 第二轮起跑与前几集的时序健康

- 入口 `experiments/2026-11-10-long-course/run_v2.sh`（提交 `e7a2e22`；文件头写明第二轮、重跑原因、唯一参数变化、有效性条件；P1–P3 预注册与「起跑前已知」两块与 `run.sh` 用 `diff` 核对逐字相同）。
  冻结代码 = 提交 `fcbf2fe`（与标定同一份）；起跑时按 `git ls-tree` 的 blob 哈希逐文件核对 `sim/` 下 178 个跟踪文件，无差异。副本里比 git 树多 1 个 `sim/tools/__pycache__/gz_io.cpython-312.pyc`（21:06:14 标定期间生成的字节码，不是源文件）。输入哈希在 `$OUT/inputs.txt`。
- 臂：teacher（20 集）、guard2-calswa-s0..2（60 集）、calswa-s0..2（60 集）；`guard2-s*` 不跑；5 条车道；`--hold 1.0 --window-from observation --corridor-half-width 3.0 --steps 240`，与第一轮相同。
- 守护进程：**PGID 1558634**，2026-11-10 21:14:57 起跑；输出 `/root/rivermind-data/uav_rgb/exp-long-course-v2-20261110/`；日志 `run.log`、各臂 `<臂>.log`；pid `run.pid`。
  看进度：`tail -f $OUT/run.log`；任意时刻汇总：`python $OUT/summarize.py $OUT --present-only`。停：`kill -- -1558634`；续跑：`cd $OUT && setsid nohup bash run_v2.sh > /dev/null 2>&1 < /dev/null &`（评完的臂跳过，半成品挪到 `<臂>.partial-<时刻>`）。
  结束时脚本自己跑 `summarize.py`（不带 `--present-only`，分母 = 请求集数）并写 `DONE`（rc + 时间）。
  停后要用 `ps -eo pid,pgid,etime,cmd | grep exp-long-course-v2` 确认没有残留（`gz sim`、`collect.py`、`uavrgb_infer` 在各自的进程组里，靠 `eval_closed_loop.py` 的 SIGTERM 处理清理）。第一轮 20:46 的停止是这条路径的一次实测：21:40 的 `ps` 里没有任何第一轮目录（`exp-long-course-20261110`）的进程。
- 预计结束：21:36 已出 72 集（起跑后约 22 分钟），140 集的余量按此速度需 20–25 分钟；calswa-s1 / s2 要等车道空出来才起，预计 22:10–22:25（按已有速度估，未核实）。
- 其他进程：`exp-k1-control-20261110`（他人的守护进程，PGID 1593702）的 `eval_closed_loop.py` 路径在本目录 `code/` 下，也就是它在用这份冻结代码，所以 `$OUT/code/` 不能删、不能改。
  21:32 的 `ps` 显示机器上同时有 8 个 `gz sim`：本实验 5 个，`exp-rule-oracle-v1` 2 个，`exp-k1-control` 1 个，总数超过共享上限 7；本实验没有超自己的配额 5。多出的 CPU 竞争对生效时长的影响没有量化。

21:36 的前 72 集（`summarize.py --present-only` 第 8 节，快照 `summary_v2_snapshot.md`）：

| 臂 | 集数 | 生效时长中位 s/步 | 集间最小 | 对 0.871 | ≥40 步的集数 | 这些集的中位 | 稳态前飞 m/步 | 转角 °/步 | 标记 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=1 参照：护栏 2（下午） | 60 | 0.871 | 0.848 | -0.0% | 6 | 0.873 | 0.258 | 26.1 | 参照 |
| K=1 参照：calswa 无护栏 | 60 | 0.868 | 0.849 | -0.3% | 14 | 0.854 | 0.249 | 26.6 | 参照 |
| teacher | 11 | 0.871 | 0.861 | -0.0% | 11 | 0.871 | 0.260 | 26.5 | 3% 内 |
| guard2-calswa-s0 | 15 | 0.861 | 0.808 | -1.2% | 9 | 0.865 | 0.257 | 26.3 | 3% 内 |
| guard2-calswa-s1 | 15 | 0.861 | 0.817 | -1.2% | 10 | 0.867 | 0.255 | 26.2 | 3% 内 |
| guard2-calswa-s2 | 16 | 0.853 | 0.807 | -2.1% | 10 | 0.862 | 0.253 | 25.9 | 3% 内 |
| calswa-s0 | 15 | 0.854 | 0.795 | -2.0% | 8 | 0.859 | 0.238 | 26.3 | 3% 内 |

- 起跑后头一批集（21:18，各臂第 1 集，seed 1）：生效时长 0.864–0.870，在 0.854–0.888 内；稳态前飞 0.247–0.258 m。
- 按主会话的窗口 0.854–0.888 逐臂看（上表三位小数）：teacher 0.871、guard2-calswa-s0 0.861、s1 0.861 在窗内；calswa-s0 0.854（0.8538）压在下限上；guard2-calswa-s2 0.853 低于下限 0.001（对 0.871 偏 −2.1%，仍在 3% 内）。≥40 步的集中位数各臂 0.859–0.871，全在窗内。
- 全集中位数偏低的原因是短集。72 集拟合 eff ≈ 0.868 − 0.471 / 步数（残差标准差 0.0129，`eff_fit_v2.txt`）；≥40 步 48 集中位 0.866（范围 0.795–0.875），<40 步 24 集中位 0.835（范围 0.807–0.861）。
  对照：K=1 参照 60 集拟合 0.875 − 0.132 / 步数（残差 0.0053）；第一轮 K=3（旧 A*）98 集拟合 0.798 − 1.470 / 步数（残差 0.0186）。第二轮的渐近值 0.868 对 K=1 的 0.875 为 -0.8%。
  每集开头的一次性开销 0.47 s 仿真时延摊在这一集的步数上：40 步的集生效时长少 1.4%，60 步少 0.9%，100 步少 0.5%，8 步少 6.8%。
- 开销的来源：已测到的一项是首步 A* 的缓存是冷的（`astar_cold_v2.txt`，新进程第 1 次调用 K=3 323 ms、K=1 50 ms；第 2、3 次 K=3 58 / 47 ms；沿航线每次的均值 16–26 ms，见 `astar_time_v2.txt`）。
  K=3 比 K=1 多的冷调用约 0.27 s 墙钟，按 RTF 0.8 折约 0.22 s 仿真，对拟合里 K=3 比 K=1 多损的 0.34 s 解释了一部分；其余（后几步补缓存、首帧、首次推理）没有逐项拆，合计未核实。
  这笔开销落在第几步没有测：`actions.jsonl` 里没有逐步派发时刻，只有总的 `action_sim_s`。
- 长集里偏低的两个离群：calswa-s0 seed 536（135 步，0.795）与 seed 113（105 步，0.804），`mean_step_s.io` 为 0.135 / 0.126 s，≥40 步的集 `io` 中位数是 0.056 s（47 集，21:36 取）。
  这两集是无护栏臂飞得最久的集，io 是平时的 2.2–2.4 倍；原因没查（推测是飞偏后 A* 搜索范围变大，未核实）。它们不改变 calswa-s0 的 ≥40 步中位数。
- 前飞位移：calswa-s0 的 0.238 m 对护栏参照 -7.9%（K=1 无护栏参照本身对护栏参照是 -3.8%）。按步段拆开（`fwd_by_step_v2.txt`，21:37 取）：
  K=1 无护栏参照在步 30 以后步长缩到 0.232–0.233 m（步 1–9 为 0.265）；K=3 calswa-s0 同样，步 30–59 为 0.238、步 60 以后 0.220；护栏臂在 K=3 各步段都是 0.251–0.256。
  逐步段对同类 K=1 参照，差在 ±6% 内（最大 -5.6%，calswa-s0 步 60 以后）；护栏臂与 calswa 的步 1–9 比 K=1 低 3%–4%（样本 n=88–288），原因没查。
  所以这个差主要是无护栏臂后段的飞法，不是时序。`summarize.py` 的说明文字原来写「前飞位移是生效时长的直接后果」，不准，已改（提交 `596e52b`）。

### 10. 第一轮的汇总（`summarize.py --present-only`，写入 `exp-long-course-20261110/summary.md`；标题「第一轮，时序混杂，20:46 中止」）

分母 = 目录里有轨迹的集数；被 SIGTERM 中断的那一集（teacher seed 992、guard2-calswa-s2 seed 997、calswa-s1 seed 153、calswa-s2 seed 122、guard2-s0 seed 1）没有 `episodes/seed_N`，不计。时序混杂：7 个臂的生效时长对 K=1 参照偏离 −9% 至 −18%，结果与 K=1 参照不可比，只作参考。

| 臂 | 集数 | S_1 | S_2 | S_3 | 状态 |
| --- | --- | --- | --- | --- | --- |
| teacher | 16 | 16/16 | 16/16 | 15/16 | 中断 |
| guard2-calswa-s0 | 20 | 12/20 | 10/20 | 7/20 | 完整 |
| guard2-calswa-s1 | 20 | 12/20 | 6/20 | 4/20 | 完整 |
| guard2-calswa-s2 | 17 | 13/17 | 9/17 | 5/17 | 中断 |
| calswa-s0 | 20（含 2 集 FAILSAFE 中止） | 3/20 | 0/20 | 0/20 | 完整 |
| calswa-s1 | 3 | 0/3 | 0/3 | 0/3 | 中断 |
| calswa-s2 | 2 | 0/2 | 0/2 | 0/2 | 中断 |
| guard2-s0 | 0 | | | | 中断，没有轨迹 |
| 变体：calswa + 护栏 2（三个种子合计） | 57 | 37/57 = 0.65 | 25/57 = 0.44 | 16/57 = 0.28 [0.18, 0.41] | |
| 变体：calswa 无护栏 | 25 | 3/25 = 0.12 | 0/25 | 0/25 | |

- 护栏对无护栏的配对 McNemar（共同 25 对）：S_1 b = 16、c = 2，p = 0.001；S_3 b = 9、c = 0，p = 0.004。教师 S_3 15/16，唯一失败是第 3 块撞柱。
- FAILSAFE 原因已核实，更正第 5.3 节「失败原因未核实」：calswa-s0 seed 122 与 seed 929 的 `episode_manifest.json` 的 `error` 分别是「step 137: cf_sitl 执行器故障: 固件进入 FAILSAFE」「step 135」，`summary.md` 第 7 节统计 stderr 含 FAILSAFE 的 2 集。seed 122 的 z 曲线与第 5.3 节的电池参照一致；seed 929 没有逐步核对 z 曲线。
- 护栏臂「数步数」航向与真实航向的差（第 9 节；第一轮护栏臂的转角 23.1–23.9°，而计数按 26.1°，Δ 里含这部分系统偏差，没有拆分）：calswa + 护栏 2 的 Δ 均值 +3.8°（步 0–19）→ +5.4°（40–59）→ +8.2°（80–99），Δ 标准差 6.2° → 8.2° → 9.2°，|Δ| 90 分位 10.7° → 15.8° → 20.3°；
  K=1 参照（下午）步 0–19 的 Δ 均值 −0.1°、标准差 2.9°、|Δ| 90 分位 4.7°，步 20–39 为 −0.4° / 3.5° / 5.8°。100 步以后只有 71 个样本（均值 +21° 至 +28°，样本少且只含活到那一步的集）。
- 块末 `|y|`（第 10 节，第一轮）：教师三块均值 0.76 / 0.88 / 0.50 m（最大 1.33 m）；calswa + 护栏 2 三块均值 0.94 / 0.93 / 0.64 m，90 分位 1.38 / 1.25 / 1.14 m，最大 1.90 m，没有向块末累积变大。
  第二轮跑完后用同一脚本重算这两节，转角回到 26° 后偏差来源才干净。

### 11. 给主会话的建议与待拍板

- 第二轮 K=3 与 K=1 参照可比（渐近生效时长 0.868 对 0.875，转角 26°，前飞 0.25–0.26 m），代价是改了 `teacher.astar_path` 的耗时（不改结果）。这不在任务书列的两个手段里，是我按 6.1 的原因加的；任务书的两个手段在标定里都没有达到窗口（0.763、0.643），所以没有别的路径能起跑。若主会话不接受这个修法，第二轮要停（`kill -- -1558634`）。
- 根上的问题是 `--window-from observation` 让动作生效时长依赖策略耗时。可选的协议级修法（没做）：派发时刻固定为观测戳加固定仿真时延，迟到的步记为无效；或改用 `--window-from dispatch`（`collect.py` 里 R01 语义，窗口从派发位姿起算，生效时长恒为 hold）。
  两者都与今天的 K=1 参照协议不同，P1 的 K=1 参照需要重跑，要不要换由主会话定。
- 每集开头的一次性开销（约 0.47 s，其中首步 A* 缓存冷是已测到的一项）：想去掉就在 `collect.py` 首步前预热 A* 缓存，改动小但要换冻结代码重起。它摊到步数上：40 步的集少 1.4%，60 步少 0.9%，100 步少 0.5%；P1–P3 的比较量是通过块数，不是生效时长，影响没有量化（未核实）。我没有做。
- 有效性条件在 `run_v2.sh` 文件头：每个臂的生效时长中位数在 0.871 s 的 ±3%（0.845–0.897）内、转角在 26.1° 的 ±3%（25.3–26.9°）内，超出的臂其 P1–P3 判定作废。21:36 的 5 个臂都在条件内（生效时长 0.853–0.871，转角 25.9–26.5°）。如果后面某个臂只因短集多而越线（≥40 步的集仍在 3% 内），按「≥40 步的集」列判断还是作废，请主会话定；我没有预设。

### 做了什么（更新）

- 复核主会话的时序表，6 组数字全部复现；用打点副本得到分段证据：延迟来自策略段的 Python A*（K=3 145–147 ms，K=1 33–38 ms），等帧 0 ms、解码 7–8 ms、推理 1 ms、帧龄 114–116 ms 与 K 无关，写盘在派发之后。
- 标定 7 组 × 2 集：去辅助相机无效（0.763），降相机频率更差（0.643），A* 栅格缓存后 K=3 为 0.872（窗口 0.854–0.888 内）。
- `teacher.astar_path` 栅格可走性缓存（`fcbf2fe`，路径逐位不变，`test_astar_cache.py` 7 项先红后绿）。
- 第二轮入口 `run_v2.sh`（`e7a2e22`）与起跑：7 个臂（teacher、guard2-calswa-s0..2、calswa-s0..2）、5 条车道，PGID 1558634，21:14:57；头几集生效时长 0.864–0.870；21:36 的 72 集各臂全集中位数对 0.871 偏 −0.0% 至 −2.1%。
- `summarize.py`（`aa15397`）：第 8–10 节、`--present-only`、`--title`、`--timing`、`--tol`；`test_course_summary.py` 新增 14 项，先红（旧脚本 14 项报错）后绿（26 项 OK）。
- 第一轮目录：`summary.md`（present-only，带标题）与 `DONE.v1`（中止原因）；没有写 `DONE`。
- 过程文档追加「更新 2026-11-10 21:30」（`c48fdb1`）与「更新 2026-11-10 21:40」（`8f794b0`）；README 补第二轮说明（`aa15397`）。
- 起跑后补查：72 集重拟合一次性开销 0.47 s、测冷调用、前飞位移按步段拆开；`summarize.py` 的前飞位移说明文字按实测改正（`596e52b`，只改文字，26 项测试仍 OK）。

### 没做什么（更新；每条带原因）

- 没有改协议（`--window-from dispatch` / 观测戳加固定仿真时延）：与 K=1 参照协议不同，P1 的参照要重跑，需主会话拍板。
- 没有预热 A* 缓存：一次性开销摊到 40 步的集少 1.4%、60 步少 0.9%，改动要换冻结代码重起；等主会话看完第一批结果再定。
- 没有进一步优化 A*：第二轮 K=3 的延迟已与下午 K=1 参照一致（0.136 s 对 0.133 s），再快会让 K=3 超过参照（K=1 在第二轮代码下已到 0.887）。
- 没有跑 `guard2-s*`（ckpt_best，次要臂）：任务书这一轮不跑。
- 没有改 `progress.md`：它在 main 上由主会话维护，分支上改会合并冲突。
- 没有等第二轮结束：规则是起跑并确认前几集后返回；最终数字在 `$OUT/summary.md`，P1–P3 的判定还没有数据。
- 没有量化一次性开销对早期碰撞集结局的影响：需要重跑对照，第二轮结束前不做。
- 没有查 calswa-s0 两个长集（seed 536、113）`io` 偏高的原因：只影响这两集，不改 ≥40 步的中位数；要查需要逐步打点。
- 没有查帧龄 114–116 ms 的来源：与 K 无关，不影响本轮结论。
- 没有处理机器上 `gz sim` 总数超共享上限 7（21:32 为 8）：多出的进程属于其他实验，只在第 9 节报告。
- 没有碰第一轮目录里已有的数据，也没有写那里的 `DONE`（门禁文件由主会话写）。
- 第 5.3 节原文没有改，更正写在本节（FAILSAFE 原因已核实）。

### 验证到哪一步（更新）

| 命令或检查项 | 结果 |
| --- | --- |
| `python sim/tools/test_astar_cache.py`（改前 / 改后） | 失败 2 项（`F....F.`）/ 7 项 OK（13.6 s） |
| `python sim/tools/test_course_summary.py`（旧 summarize.py / 新） | 14 项报错（缺函数）/ 26 项 OK，证据 `red_summary.txt`、`green_summary.txt` |
| `test_teacher_policy` / `test_teacher_costmap`（44 项）/ `test_teacher_kino` / `test_teacher_belief` / `teacher.py --self-check` | 全部通过 |
| `UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only` | rc=0，「PASS（有 SKIP）」，日志 `check_python_only_r2.log` |
| 打点副本 4 任务（seed 1） | 策略段 K=3 144.7–147.0 ms，K=1 32.6–37.5 ms；观测→派发 = 帧龄 + 策略段折仿真，误差 ≤ 0.6 ms |
| 标定 7 组 × 2 集 | K=3 A* 缓存 0.872 s，窗口内；去辅助相机 0.763、降频 0.643，窗口外 |
| `run_v2.sh`：`bash -n`、预注册块与 `run.sh` 逐字 `diff`、冻结副本 blob 哈希核对 | 语法 OK、两块逐字相同、`sim/` 178 个文件无差异 |
| 第二轮 72 集（21:36） | 各臂全集中位数 0.853–0.871（对 0.871 偏 −0.0% 至 −2.1%），转角 25.9–26.5°，均在 ±3% 内；guard2-calswa-s2 的 0.853 低于主会话窗口下限 0.854，calswa-s0 的 0.8538 压在下限上；≥40 步的集中位数 0.859–0.871 |
| 一次性开销拟合与冷调用 | 72 集 eff ≈ 0.868 − 0.471 / 步数（残差标准差 0.0129）；冷调用 K=3 323 ms，K=1 50 ms（`eff_fit_v2.txt`、`astar_cold_v2.txt`） |
| 前飞位移按步段（21:37） | 对同类 K=1 参照差在 ±6% 内，最大 −5.6%（`fwd_by_step_v2.txt`） |
| 冻结副本与 git 树 | 178 个跟踪文件哈希一致；多 1 个 `__pycache__/gz_io.cpython-312.pyc`（字节码） |
| 第二轮 140 集全部结果 | 未结束，进行中（预计 22:10–22:25，未核实） |
| 第一轮 `summary.md` / `DONE.v1` | 已生成；`DONE` 不存在（核对过） |
