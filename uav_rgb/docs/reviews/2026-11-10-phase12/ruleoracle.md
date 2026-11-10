# ruleoracle 报告：规则层上限（真值列距离 + 固定规则 v0）

- 开始 2026-11-10 19:47，正式实验 20:05:21 起跑（时间盒 50 分钟内），报告首版 20:10，实验结果未出。
- 工作树 `/root/claude-code-backup-20261028/code/wt/p12-ruleoracle`，分支 `p12-ruleoracle`，基线 main `155e377`。
- 提交（均在 `p12-ruleoracle`）：`18f2ec0` feat(rule)（规则层 + collect/eval 接线 + 测试），`fb467b7` feat(experiments)（实验入口），`749112c` docs(worklog)（过程文档 `docs/worklog/2026-11-10-2010-rule-oracle.md`，任务书只要求前两个，第三个是用户全局规则要求的过程文档，主会话不要可跳过）。
- 实验输出 `/root/rivermind-data/uav_rgb/exp-rule-oracle-20261110/`（下文记 `$OUT`），草稿 `/root/rivermind-data/uav_rgb/tmp/phase12-20261110/ruleoracle/`（下文 `ruleoracle/` 指它）。

## 0 要点

1. 规则层与接线已入库：`--policy rule-oracle`、`--rule-hfov-deg`，其它策略的命令行与落盘行逐位不变（测试锁住）。
2. 冒烟：train 分区 seed 2、3，两集都成功（31 步与 35 步，SPL 0.948 与 0.904），转向方向正确（绕过门柱，没有撞向门柱），无左右摆动、无悬停死锁。
3. 正式实验在跑：rule60 与 rule81 各 3 次重复，每次 20 个冻结 seed，共 6 个臂 120 集，2 路并行，守护进程 PGID 1348128，预计 21:20 前后结束（范围 21:10 到 21:45）。
4. 规则设计上的一个发现（没有改规则）：规则 1 的例外分支在预注册参数下恒为悬停（SAFE_NEUTRAL），其中「不受阻则前进」的分支不可达，可能造成原地悬停到步数用完，详见第 5 节。

## 1 角度与坐标约定的核实

目标：`column_distances` 的列顺序、射线角、左右符号与 `sim/tools/teacher.py` 一致，且锁进测试。

**读代码**

| 约定 | 出处 | 结论 |
| --- | --- | --- |
| 世界系 x 前、y 左，yaw 度，0 指 +x，正为左转（逆时针） | `teacher.py:517-523` `_rollout`：`YAW_LEFT` 时 `nyaw = yaw_deg + TIE_YAW_STEP_DEG`（27°），`YAW_RIGHT` 减 | 采用 |
| 方位角正为左 | `teacher.py:565` `visible_pillar_side`：方位 = atan2(dy, dx) − yaw | 列 j 的中心方位 φ_j = (n/2 − 0.5 − j)·3°，列 0 在最左（+28.5°），列 19 在最右（−28.5°） |
| 射线角是世界系绝对角 | `teacher.py:241` `ray_clear(x, y, th_deg, length, pillars, clearance, ...)` 沿 (cos th, sin th) 走 | 列内射线世界角 = yaw + φ_j + 列内偏移 |
| 相机在机体前 0.045 m | `sim/dynamics/e1_mini.json` `camera_mount.xyz = [0.045, 0, 0.015]` | 射线起点 = 机体位置 + 0.045·(cos yaw, sin yaw)，y 向无偏置 |
| 机体半径 0.102 m | 同文件 `collision_envelope.radius_from_center` | r_safe = 0.102 + 0.25 余量 = 0.35（任务书给定值） |
| 柱子表示 | 场景 meta：方柱 `size = [sx, sy, sz]`，圆柱 `radius`；`rule_policy._hit` 读 `size[:2]` | 方柱 slab 法，圆柱判别式；高 3 m 按水平截面二维求交 |

**经验核对（实测转向符号）**：`python3 $D/tmp/phase12-20261110/ruleoracle/check_dyaw_sign.py` 读 `exp-heading-guard-20261110/guard2-calswa-s*/episodes/seed_*/actions.jsonl` 里实际执行步的 `dyaw_deg`：

```
YAW_LEFT: n=152 mean=+26.14 min=+24.4 max=+29.8
YAW_RIGHT: n=156 mean=-26.10 min=-29.2 max=-24.1
```

YAW_LEFT 使 yaw 增大，约 26°，与 `_rollout` 的 27° 一致。

**锁进测试**（`sim/tools/test_rule_policy.py`）：
- `test_pillar_on_the_left_shortens_only_left_columns`：柱子在左前，只有左半列变短，右半全是 max_range，并检查镜像对称。
- `test_yaw_left_moves_world_forward_pillar_to_the_right_side`：同一根世界系正前方的柱子，机头左转 20° 后左半 10 列全是 max_range，柱子只落在右半列（列 16、17 小于 2 m），机头右转 20° 则左右镜像。
- `test_side_agrees_with_visible_pillar_side`：对多组位姿，列距离判出的左右与 `teacher.visible_pillar_side` 一致。
- `test_ray_through_pillar_centre_is_blocked_in_teacher_ray_clear`：沿 `column_distances` 的射线方向，`teacher.ray_clear(..., clearance=0.01)` 判被挡。注意 `ray_clear` 用严格小于，clearance=0 永不挡，所以用 0.01。
- 突变检查：在临时副本里做 7 种突变（方位符号翻转、忽略 cam_forward、回转方向对调、左右和对调、圆柱当外接方块、不截断求和、平局不看 last_turn），测试分别有 10、6、5、6、1、1、4 项失败，全部被抓到；临时副本已删。

## 2 代码改动与测试

**`sim/tools/rule_policy.py`（112 行，纯 stdlib）**
- `column_distances(x, y, yaw_deg, pillars, hfov_deg=60.0, col_deg=3.0, max_range=4.0, cam_forward=0.045)`：左到右各列水平距离，每列 5 条射线含两条边缘（相邻 0.75°），取最小，封顶 max_range；60° 20 列，81° 27 列；|y| = 3 m 的走廊边界不是实体。
- `RULE_V0`（文件顶部，一行一个，行尾写依据）：r_safe 0.35，d_look 1.2，n_max 2，side_near 0.5，tie_frac 0.10，sum_clip 4.0。
- `rule_v0_entry` 返回 (动作, 条目)，`rule_v0` 只返回动作。条目：R1、R1-safe、R1-fwd、R2-sum、R2-last、R2-n、R2-left、R3。

**`sim/tools/collect.py`**
- `--policy rule-oracle`（没有 `--scene-meta` 时报错）；`--rule-hfov-deg`（默认 60.0，范围 [6, 180]，非 rule-oracle 且不等于 60 时报错）。
- 真值位姿用该步的 `pose_before` 与 `yaw_before`，柱子用 `--scene-meta`；缺任一项该步失败，不悄悄退回别的策略。
- 主循环复用 `net_yaw`，新增 `last_turn`（最近一次执行的转向）；`executed_by="rule_oracle"`；manifest 多记 `rule_oracle: {hfov_deg, params}`，`teacher="rule_oracle"`，`planner="rule_v0"`。
- 只在 rule-oracle 下，每步 actions.jsonl 多两个键：`rule_cols`（各列距离，2 位小数）与 `rule_entry`。其它策略的行与 manifest 逐位不变。

**`sim/tools/eval_closed_loop.py`**：`--policy rule-oracle`，透传 `--rule-hfov-deg`（只给 rule-oracle 传，其它策略命令行逐字不变）；`eval.json` 顶层多记 `rule_hfov_deg` 与 `rule_params`；`metric_version` 不变。

**策略名枚举处**（在 worktree 里 `grep -n 'teacher-nn'` 与 `grep -n '"forward"'` 查 `tools/`、`sim/tools/`、`sim/3dgs/` 的 `*.py`）：
- 加了：`sim/tools/collect.py`、`sim/tools/eval_closed_loop.py` 的 `--policy` choices 与帮助，以及两个测试文件。
- 没有策略枚举，没改：`tools/pipeline_common.py`、`tools/stats_report.py`（两者都没有命中）。
- 命中但不该加：`sim/tools/collect_parallel.py` 的 `--policy choices=("teacher", "forward", "teacher-rl", "student")` 是训练数据采集器，规则策略不产训练数据（它也没有 teacher-nn）；`tools/e1_pipeline.py`、`tools/compare_datasets.py`、`tools/dagger_gz_pipeline.py` 里的 `("forward", "forward", None)` 是这些流水线写死的评测臂清单，不是策略枚举；`sim/tools/teacher_policy.py` 是教师网络加载器；`sim/3dgs/*` 有自己的策略枚举，规则层的 3DGS 评测不在本任务。

**测试**
| 文件 | 项数 | 先红 | 后绿 |
| --- | --- | --- | --- |
| `sim/tools/test_rule_policy.py` | 27 | `ModuleNotFoundError: No module named 'rule_policy'`（`ruleoracle/red_test_rule_policy.txt`） | `Ran 27 tests ... OK`（`green_test_rule_policy.txt`） |
| `sim/tools/test_rule_oracle_wiring.py` | 16 | `Ran 16 tests ... FAILED (failures=8, errors=6)`（`red_test_rule_oracle_wiring.txt`） | `Ran 16 tests in 1.178s ... OK`（`green_test_rule_oracle_wiring.txt`） |

覆盖任务书要求的全部场景：正前 1.0 m 柱（中间列 0.955，两侧列 max_range，0.955 = 1.0 − 0.045）、柱在左前只缩左列、一圆一方、空场全 max_range 且前进、前方受阻朝空侧转、|n| ≥ 2 回转、平局沿用 last_turn、81° 视场 27 列。接线测试覆盖：`collect_one` 的行与 executor、`pose_before` 与 `yaw_before` 的使用、主循环 `net_yaw` 与 `last_turn` 的维护、manifest、参数校验、其它策略不留痕迹、eval 的命令行与 `eval.json` 键。

写测试时纠正了 6 处我自己的预期，都改测试，不是实现问题：
1. 左前柱会连带映出侧面，左边 4 列都变短，不是只有最左 2 列；
2. 81° 测试用的柱 (1.0, 1.4) 在方位 +55.7°，两种视场都看不到，换成 `square(1.2, 1.2)`（60° 看不到，81° 的最左 2 列看得到）；
3. `teacher.ray_clear(..., clearance=0.0)` 用严格小于，永不挡，交叉测试改用 0.01；
4. 规则 1 例外的「不受阻则前进」分支在默认参数下不可达（见 5.1），原先的预期写成了可达，测试改为：默认参数得 SAFE 与 R1-safe，自定义 `r_safe=0.1` 才得 FORWARD 与 R1-fwd；
5. 一处 n = −2 的预期写混了，正确答案是 YAW_LEFT（n < 0 朝左回转）；
6. 接线测试里 `round(0.6 − 0.045, 2)` 落在浮点舍入边界（0.555），改成容差 0.006 的近似比较（相机前移误差 0.045 仍会被抓到）。

`UAV_RGB_PYTHON=$R/.venv-tools/bin/python bash tools/check.sh --python-only`：`rc=0 2026-11-10 20:00:36`（日志 `ruleoracle/check_python_only.log`，约 2 分 45 秒）。脚本尾部声明「未检查：C++/CUDA 构建、CTest、GPU 测试、cf_sitl」，本次没改 C++，不需要全量。

## 3 冒烟（train 分区，seed 2 与 3，rule60，同正式口径）

命令 `ruleoracle/smoke.sh`（两集并行，2 个 gz，19:58:18 起，19:59:16 结束），输出 `exp-rule-oracle-20261110/smoke/`。

| seed | 门柱 y | 结果 | 步数 | SPL | 动作序列（F 前进，L 左转，R 右转） | 规则条目计数 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | −0.082 | 成功 | 31 | 0.948 | `FFFLLRFFFFRFFFFFFFFFFFFFFFFFFFF` | R3 27，R2-sum 2，R2-left 1，R1 1 |
| 3 | +0.019 | 成功 | 35 | 0.904 | `FFFRRLFFFFLFFFFFFLFFFFRFFFFFFFFFFFF` | R3 29，R2-sum 5，R1 1 |

- 方向：seed 3 在第 3 步（`pose_before` = (0.19, −0.04)，yaw −0.8°，门柱近面在正前 1.06 m）朝右转，两次右转后 yaw 到 −54.5°，第 5 步规则 1 回转（左），第 11 步机体已到 (0.99, −0.70)，绕过 y = +0.019 的门柱，没有撞向门柱。seed 2 对称：第 3 步（(0.17, −0.08)，门柱 y = −0.082，正前 1.08 m）朝左转，两次左转后回转，第 11 步到 (1.16, 0.46)，从门柱左侧绕过。
- 列距离合理：seed 2 第 0 步 `pose_before` = (−0.542, −0.059)，yaw 0.10°，中间列 1.80 = 门柱近面 x 1.3（柱心 1.5，边长 0.4）减去相机位置 −0.542 + 0.045；用 `scene_gen.sample_pillars(2)` 重算得到同样的 1.8，`geometry_gid` 与 manifest 一致（`2d89b6d2ee71b81c`），seed 3 同样一致（`5c3bc80128e15585`）。过了最后一根柱后各列回到 4.00（max_range）。
- 实测转向：seed 2 的 YAW_LEFT dyaw +26.8°、+30.1°，YAW_RIGHT −24.4°、−27.5°；seed 3 的 YAW_RIGHT −27.8°、−28.7°、−27.1°，YAW_LEFT +24.4°、+27.6°、+27.7°。与第 1 节一致。
- 落盘：actions.jsonl 每步有 `rule_cols`（20 个数）与 `rule_entry`，`executed_by` 全为 `rule_oracle`，`action_expert == action_executed`；`episode_manifest.json` 有 `rule_oracle` 键与 `teacher=rule_oracle`；eval.json 顶层有 `rule_hfov_deg=60.0` 与 `rule_params`。
- 没有出现左右反复摆动，也没有连续 SAFE（两集 SAFE 步数为 0）。两集都没触发规则 1 的例外分支，所以悬停风险（第 5.1）在冒烟里没有被检验。
- 两集 seed 的几何都在 train 分区（`sm.split_for_scene`），没有碰 20 个冻结 test seed。
- 冒烟后核对：没有残留的 gz、eval、collect 进程。

## 4 正式实验

| 项 | 值 |
| --- | --- |
| 启动 | 2026-11-10 20:05:21，`cd $D/exp-rule-oracle-20261110 && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &` |
| PGID | 1348128（`run.pid`），`ps -o pid,pgid,etime,cmd -p $(cat run.pid)` 确认存活 |
| 冻结代码 | `git archive 18f2ec0` 解到 `$D/tmp/phase12-20261110/ruleoracle/code-18f2ec0`；`inputs.txt` 记了提交与关键文件 sha256 |
| SITL 插件 | `UAVRGB_CF_SITL_PLUGIN_DIR=$D/tmp/build-cf-sitl-lib31/gz`（sha256 `2811ed50...bf399`，profile `cf_sitl_v1:1bc8fedf:lib31`） |
| 臂（优先级顺序） | rule60-rep0/1/2，rule81-rep0/1/2；每臂 20 个冻结 seed（`1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654`） |
| 口径 | `--task base --split test --steps 80 --hold 1.0 --window-from observation --dynamics cf_sitl --vehicle e1_mini --corridor-half-width 3.0`，`metric_version = sim09.v2.coast1s.corridor3m` |
| 并行 | 2 路（gz 同时 2 个），首批 rule60-rep0 与 rule60-rep1 |
| 日志 | `$OUT/run.log`（总），`$OUT/<臂>.log`（每臂，`[seed N] steps=... success=...` 一集一行） |
| 进度 | `tail -f $OUT/run.log`；逐臂一行：`cd $OUT; for a in rule{60,81}-rep{0,1,2}; do [ -f $a.log ] && echo "$a: $(grep -cE '^\[seed' $a.log)/20 集，成功 $(grep -c 'success=True' $a.log)，撞柱 $(grep -c 'collision=True' $a.log)"; done`（`eval.json` 要等一个臂跑完才落盘，`summarize.py` 在臂跑完前看不到它） |
| 停 | `kill -- -1348128`（eval 的信号处理会收掉各自的 gz 与 collect 进程组） |
| 汇总 | 随时 `cd $OUT && $R/.venv-tools/bin/python summarize.py`，写 `summary.md`；run.sh 结束时自己跑一次并写 `DONE` |
| 续跑 | 同一条启动命令；评完的臂跳过，半成品挪到 `<臂>.partial-<时刻>`；`.lock` 互斥 |
| 占用 | 20:07 时 1.6 MB（每集只留 actions.jsonl、oracle.jsonl、episode_manifest.json），预计远小于 100 MB |

**起跑后核对**：20:10:26 时 rule60-rep0 与 rule60-rep1 各完成 5 集，10 集全部成功，没有失败集（20:07:37 时 rep0 的 seed 1 共 33 步、seed 113 共 30 步，rep1 的 seed 1 共 33 步、seed 113 共 33 步）。以上只是前 10 集，不构成结果。同一 seed 两次重复的 max_x 不同（6.2105 与 6.1948），说明固件在环的噪声使每次重复并不逐位相同，3 次重复有意义。

**预计完成时间**：单集约 65 到 75 秒（20:05:22 起，首集 20:06:35 结束），每路要跑 60 集，约 70 分钟，预计 21:20 前后；撞柱集提前结束，未到线的集要跑满 80 步（约 2.5 分钟），范围 21:10 到 21:45。

**结束后**：`cat $OUT/DONE`（rc 与时间），`$OUT/summary.md` 是汇总，`$OUT/<臂>/eval.json` 是各臂原始结果；结果落到 `experiments/2026-11-10-rule-oracle/RESULT.md`（实验目录惯例，我没写，等结果）。

**预注册**（写在 `experiments/2026-11-10-rule-oracle/run.sh` 文件头，20:05 起跑前）：
- 预测：rule60 三次重复合并 60 集，走廊内成功 ≥ 54/60，撞柱 ≤ 3/60。
- 证伪：rule60 走廊内成功 < 48/60，说明规则层表达力不够，先加短期记忆（最近 3 步列距离按位移转到当前机体系）再评。
- 次要预测：rule81 撞柱数 ≤ rule60。
- 主比较 3 个：rule60 对 54 与 48 两条线；rule60 撞柱对 3；rule81 对 rule60 的撞柱数（配对精确 McNemar，60 对不独立，p 只作参考）。
- 并列参照不做检验结论：教师 A* 20/20（`exp-teacher-cost-20261109/cl1/eval-teacher-legacy/eval.json`，metric_version `sim09.v1.coast1s`，无走廊判据，只有 eval.json 没有逐集轨迹，走廊无法补算），学生最好配置 calswa + 护栏 50/60（`exp-heading-guard-20261110/guard2-calswa-s{0,1,2}`：15 + 17 + 18，`sim09.v2.coast1s.corridor3m`）。

**`summarize.py` 输出**：每个变体 60 集的走廊内成功（Wilson 区间）、撞柱、越界、未到线、各重复成功数、SPL、动作占比、每集左右相邻换向与转向方向变化、含 ≥ 3 连续 SAFE 的集数（悬停死锁的探针）、规则条目触发占比；撞柱集逐集列被撞柱坐标、撞击前方位与距离、被撞柱中心是否在视锥内（自身视锥与 60° 视锥两列）、撞前 3 步（含撞击步）最小列距离与（条目:动作）；越界与未到线集列末 8 步。验证过：冒烟 2 集（解析、表格、条目占比）、两条参照行（学生 50/60 = 15 + 17 + 18，SPL 0.783；教师 20/20）与我造的合成撞柱集（被撞柱、方位 −10.0°、距离 0.60 m、视锥分类与手算一致）；规则臂的真实撞柱集还没有，未验证；撞柱分析函数另在学生 calswa + 护栏的 10 个真实撞柱集上跑通（`load_arm(..., 60.0)`，被撞柱到机体中心的最近距离 0.10 到 0.18 m，机体半径 0.102 m，被撞的多是 x = 1.5 的门柱；视锥分类 心在内 7、仅边缘 1、全在外 2；这些集没有 `rule_cols`，撞前 3 步列距离为空）。20:11 为了验证参照行在实验目录里跑过一次，`summary.md` 里是当时的「进行中」内容，run.sh 结束时会重写。

## 5 对规则设计的观察与建议（规则未改）

**5.1 规则 1 的例外在预注册参数下只会悬停**

- 事实：例外条件是「回转一侧最外 2 列的 d < side_near（0.5）」。这两列的中心方位在 60° 视场下是 ±28.5°、±25.5°，blocked 判定用列中心方位，d < 0.5 时侧向偏移 d·|sin φ| < 0.5·sin 28.5° = 0.24 < r_safe（0.35），前向距离 d·cos φ < 0.5 < d_look（1.2），这两列自己就满足 blocked。81° 视场最外两列 ±39°、±36°，0.5·sin 39° = 0.315 < 0.35，结论相同。
- 所以例外分支恒走「blocked → SAFE_NEUTRAL」（条目 R1-safe），「不受阻 → FORWARD_SLOW」（R1-fwd）在 RULE_V0 下不可达（测试 `test_r1_exception_forward_branch_needs_a_smaller_r_safe` 要把 r_safe 降到 0.1 才走到）。
- 风险：SAFE_NEUTRAL 原地悬停，位姿、n、last_turn 都不变，下一步输入几乎相同，决策重复，一直悬停到 80 步用完（未到线），或悬停漂移撞柱。触发条件：|n| ≥ 2 且回转一侧的外缘贴近障碍。冒烟的两集没有触发。
- 怎么看：`summary.md` 第 3 节「含 ≥ 3 连续 SAFE 的集数」与第 4 节 R1-safe 占比，第 6 节的未到线集末 8 步。
- 建议（只在 train 分区几何上评，不用 20 个 test seed）：例外分支受阻时落到规则 2（朝较空的一侧转），或悬停连续 2 步后落到规则 2。|n| 超限时多转一步，比悬停到超时好。

**5.2 视锥外的柱子没有记忆**

两次同向转向（n = ±2，约 54°）后，触发转向的那根柱子已移出 60° 视锥，规则 3 随后前进，机体沿偏离路径 54° 的航向走，侧擦被绕开的柱子。这是计划 D6「短期记忆」的触发条件，学生 18 次撞柱里 16 次的被撞柱中心在 60° 视锥外（data-algo 审查）。看撞柱表的「自身视锥」列：若 rule60 的撞柱多为「全在外」，加记忆是对症的；若 rule81 的撞柱明显更少，则视场就是瓶颈。

**5.3 其他**

- n = 0 且左右和接近时固定朝左（R2-left）。冒烟 seed 2 的第一次转向就是 R2-left，实际结果成功，但对称场景下是左偏的，看 summary.md 第 4 节 R2-left 的占比与撞柱集是否偏向右侧。
- 81° 视场 27 列，奇数，中间列不计入左右两半的求和（左 13 列、右 13 列），不影响对称性，影响很小。
- rule81 只改规则的列几何，Gazebo 相机仍是 60°，因为规则用真值不看图像，这个臂回答的是「视场更宽时规则能否少侧擦」，不是「相机换了会怎样」。

## 做了什么

- 新增 `sim/tools/rule_policy.py`（112 行）与两个测试文件共 43 项测试，先红后绿，7 种突变都被抓到。
- `collect.py` 与 `eval_closed_loop.py` 接线 `--policy rule-oracle` 与 `--rule-hfov-deg`，其它策略逐位不变；提交 `18f2ec0`。
- `tools/check.sh --python-only` rc=0。
- 核实了角度约定：读代码、实测 dyaw、交叉测试 `visible_pillar_side` 与 `ray_clear`。
- 冒烟 train 分区 2 集并核对方向、列距离、落盘字段。
- 写 `experiments/2026-11-10-rule-oracle/{run.sh,summarize.py,README.md}` 并提交 `fb467b7`；以守护进程起跑正式实验（PGID 1348128），确认前 10 集正常（10/10 成功）。
- 写本报告与过程文档 `docs/worklog/2026-11-10-2010-rule-oracle.md`。

## 没做什么（每条带原因）

- 没改规则 v0 的任何参数或结构：任务书要求参数起跑前冻结，规则设计问题只记录（第 5 节）。
- 没给规则 1 的例外分支补可达路径，也没加短期记忆：前者是规则设计问题，后者是证伪后的下一步，都不在起跑前做。
- 没把 rule-oracle 加进 `collect_parallel.py` 与 3DGS 脚本（`sim/3dgs/*`）：前者是训练数据采集器，后者另有策略枚举，规则层的 3DGS 评测不在本任务。
- 没等正式实验结束：任务书要求起跑并确认前几集正常后返回，结果由主会话收。
- 没给教师参照行补走廊判据：11-09 的教师评测目录只留 eval.json，没有逐集轨迹，无法算 |y| ≤ 3 m；该行只作参照，口径是 v1。
- 没做多重比较校正与功效计算：主比较只有 3 个，60 个配对不独立，p 值只作参考。
- 没拿渲染图像里的柱子边缘列与 `column_distances` 的列逐像素对照：rule-oracle 不看图，不影响本实验；训练几何头时才需要，且相机内参与畸变要另行标定（未核实）。
- 没跑全量 `check.sh`：没改 C++；脚本自己声明未检查 C++/CUDA 构建、CTest、GPU 测试、cf_sitl。
- 没改 `progress.md`：5 条并行分支都改头部会在合并时冲突，留给主会话合并时统一写；已完成、关键决策、待办三块内容见本报告第 0、4、5 节。
- 没动 `ui/`、main、主仓库，没 push，没删任何已有文件。

## 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| `$PY sim/tools/test_rule_policy.py`（先红） | `ModuleNotFoundError: No module named 'rule_policy'` |
| 同上（后绿） | `Ran 27 tests in 0.048s OK` |
| `$PY sim/tools/test_rule_oracle_wiring.py`（先红 / 后绿） | `FAILED (failures=8, errors=6)` / `Ran 16 tests in 1.178s OK` |
| 7 种突变（临时副本） | 全部被测试抓到，副本已删 |
| `bash tools/check.sh --python-only` | `rc=0 2026-11-10 20:00:36` |
| `python3 ruleoracle/check_dyaw_sign.py` | YAW_LEFT 均值 +26.14°（n=152），YAW_RIGHT −26.10°（n=156） |
| 冒烟 seed 2、3（train，Gazebo cf_sitl，2 个 gz） | 两集成功，31 与 35 步，SPL 0.948 与 0.904，方向正确，无摆动无悬停 |
| 起跑后核对（20:10:26） | rule60-rep0、rep1 各 5 集，共 10/10 成功，只是前 10 集，不构成结果 |
| `summarize.py` | 冒烟集、合成撞柱集、学生 10 个真实撞柱集（撞柱分析）、两条参照行都通过；规则臂的真实撞柱集未出，未验证 |
| rule60 走廊内成功 ≥ 54/60 等预注册线 | 未出结果，实验进行中 |
| 规则 1 例外分支悬停风险 | 由参数与测试推出，闭环里未触发，未验证 |
| 3DGS 跨域评测与真机 | 没做，不在范围 |
| summarize.py 的几何对账 | 冒烟 2 集的 `geometry_gid` 与 `sample_pillars(seed)` 重算一致；20 个 test seed 的对账在汇总时自动做，对不上的集会标注，未对账的不做撞柱分析 |

## 更新（21:05，v0 终局与 §5.1 悬停预测的更正）

- v0 终局：rule60 走廊内成功 38/60（14/12/12），Wilson 95% [0.51, 0.74]，低于预注册证伪线 48/60，判负（`/root/rivermind-data/uav_rgb/exp-rule-oracle-20261110/summary.md`，metric_version `sim09.v2.coast1s.corridor3m`）。rule81-rep0 取消，rule81 的 v0 不再评。
- §5.1 预测的「规则 1 例外分支恒为 SAFE_NEUTRAL、悬停」**没有发生**：2990 步里 SAFE 为 0 步，含 3 连续 SAFE 的集 0 个。14 个未到线集里 13 个是规则 1（回转）与规则 2（受阻转向）在门柱前左右拉扯，末 20 步换向 15–19 次。当时只做了静态推导，没有估计例外分支的进入概率。逐集数据与时间线见 `reports/rulev1.md` §1。
