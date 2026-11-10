# rulev1 报告：规则 v1（短期障碍记忆 + 候选航向选择）的实现、离线检查、冒烟与正式实验

- 主会话 20:39 交办，v0 守护进程 20:56:27 收完，正式实验 20:59:08 起跑（时间盒 45 分钟内），报告 21:17 定稿，实验结果未出。
- 工作树 `/root/claude-code-backup-20261028/code/wt/p12-rulev1`，分支 `p12-rulev1`，基线 main `35be6d0`（含 p12-ruleoracle、长课程 `course_base`、`tools/average_ckpt.py`）。
- 提交（均在 `p12-rulev1`）：`cafb826` feat(rule)（规则 v1 + 离线模拟 + 接线 + 测试），`b3d06d9` feat(experiments)（实验入口），另有 `ea25ddf` test，以及 `edac2a9`、`db88fc5`、`9399447` 三个 docs(worklog)（见文末）。
- v1 实验输出 `/root/rivermind-data/uav_rgb/exp-rule-oracle-v1-20261110/`（下文记 `$OUT`），草稿 `/root/rivermind-data/uav_rgb/tmp/phase12-20261110/rulev1/`（下文 `rulev1/`）。
- v0 实验输出 `/root/rivermind-data/uav_rgb/exp-rule-oracle-20261110/`（下文记 `$V0`）。

## 0 要点

1. v0 守护进程已收：20:42:39 前后挂起 xargs 并只给 rule81-rep0 的 eval 发 SIGTERM（省出 1 个 gz；`run.log` 记该臂 20:42:39 结束），20:56:14 确认 rule60-rep2 的 20 集出齐，随即按 PGID 1348128 停守护进程，20:56:22 与 20:56:27 两次残留检查都为空（无 gz / collect 残留），20:56:27 `summarize.py` rc=0 并写 `DONE`（文字与交办一致）。rule81-rep0 的半成品改名 `rule81-rep0.cancelled-204303`。
2. v0 终局 rule60：走廊内成功 38/60（14/12/12，Wilson [0.51, 0.74]），撞柱 2，越界 6，未到线 14。**失败机制与我在 v0 报告 §5.1 的预测不同**：SAFE 占全部步 0.000，含 3 连续 SAFE 的集 0 个，没有悬停；14 个未到线集里 13 个是规则 1（回转）与规则 2（受阻转向）在门柱前左右拉扯（末 20 步换向 15–19 次，max_x 1.31–1.51 m）。`DONE` 里「悬停」二字是主会话原话，我按交办逐字写入，数据不支持它（详见 §1）。
3. v1 已实现并入库（`cafb826`、`b3d06d9`、`ea25ddf`）：规则与坐标函数是纯函数，记忆由 `collect.py` 主循环按集清零、逐步传入；v0 的行为与落盘逐位不变。新增测试 40 + 18 项，先红后绿；v0 的 27 + 16 项仍绿；19 种突变全部被抓到；`check.sh --python-only` rc=0（结果见文末验证表）。
4. 离线检查（运动学模拟，train 分区 20 个几何）：过线 18/20，撞柱 0/20，**正好在 18/20 的门槛上**（门槛是「低于 18 就停」，所以继续）。没过线的 seed 3、14 是 `V1-more ↔ V1-cap` 在 n=2 与 3 之间原地摆动 80 步（设计缺陷，见 §7，没有改）。同一份模拟里 81° 视场也是 18/20，没过的换成了 seed 11、20（同一类摆动），所以加宽视场不能消除这个缺陷。这 20 个几何只有 4 族，3 号与 14 号是同族且偏移只差 4 mm（§7 第 10 条），离线 18/20 的有效样本比 20 少。
5. Gazebo 冒烟（train 分区 seed 2、10、3、14）：4/4 过线（31 / 39 / 31 / 29 步），无碰撞、无左右拉扯；seed 3、14 在 Gazebo 里没有复现离线摆动。
6. 正式实验在跑：守护进程 PGID **1520162**，2 路并行，7 个臂（rule60v1 ×3 → 长课程 K=3 → rule81v1 ×3）。起跑后 17 分钟（21:16）rep0 与 rep1 共 33 集，32 成功、1 未到线、0 撞柱；**这 1 个未到线（rep0 seed 683）就是 §7 第 1 条的 `V1-more ↔ V1-cap` 两周期，在 Gazebo 的 test seed 上出现了**（详见 §6）。rule60v1 三次重复预计 21:50 前后出齐，全部预计 23:00–23:15（估计，依据见 §6）。
7. 需要主会话知道的风险：v1 与 v0 的失败是同一类结构（两个相反的规则在同一位置来回），v1 里的形式是 `V1-more ↔ V1-cap`。离线模拟对这类原地来回偏悲观（v0 离线 1/20，Gazebo 实测 38/60），Gazebo 里的真实发生率只能等正式结果（目前 29 集里 1 集，见 §6）；冒烟与离线只覆盖 train 分区。

## 1 v0 终局（收守护进程时的最终值）

数据来源：`$V0/summary.md`（`summarize.py` 20:56:27 重跑）、`$V0/rule60-rep{0,1,2}/episodes/seed_*/actions.jsonl`（本节的换向、SAFE、n、条目占比由 `rulev1/episode_stats.py` 读取，复核输出在 `rulev1/episode_stats_v0_rule60.txt`）。口径 `sim09.v2.coast1s.corridor3m`，分母 = 请求集数 60。

| 项 | 数值 |
| --- | --- |
| 走廊内成功 | **38/60**（三次重复 14 / 12 / 12），Wilson 95% [0.51, 0.74]；预注册证伪线 < 48，判负 |
| 撞柱 | 2（rep1、rep2 的 seed 992） |
| 越界 | 6（seed 536 三次，929 一次，990 两次） |
| 未到线 | 14 |
| SPL / 平均步数 | 0.582 / 49.8 |
| SAFE 占全部步 | **0.000**（2990 步里 0 步） |
| 含 ≥3 连续 SAFE 的集数 | **0** |
| 每集 L↔R 相邻换向 | 21.90（成功集的中位数 1） |
| 条目占比（2990 步） | R1 0.235，R1-safe 0.000，R1-fwd 0.000，R2-sum 0.276，R2-last 0.006，R2-n 0.000，R2-left 0.005，R3 0.478 |

按结果分的失败机制（逐集核对）：

| 结果 | 集数 | 形态 |
| --- | --- | --- |
| 未到线，门柱前拉扯 | 13 | 末 20 步相邻 L/R 换向 15–19 次，SAFE 0 步，max_x 1.31–1.51 m；末 8 步全是 `R1:R R2-sum:L` 交替（rep1 seed 929、rep2 seed 929 是 `R1:L R2-sum:R`）。seed 176、230、683 三次都是 |
| 未到线，步数耗尽 | 1 | rep0 seed 1654：max_x 5.07 m，全程换向 45 次，早期拉扯耗掉了步数（同 seed 的 rep1 成功、78 步，rep2 拉扯） |
| 越界 | 6 | 末 8 步全是 `R3:F`（前进），末 20 步 n 恒为 ±1（rep2 seed 536 是 1 与 2）：n=1 时 R1 不触发（阈值 n_max=2），26° 的航向偏差一直没被纠正，各集 `pose_before` 的最大 \|y\| 为 3.35–7.63 m（走廊判据 3 m；`episode_stats.py` 不含这一项，是另一段一次性代码读 `pose_before.y` 得到的） |
| 撞柱 | 2 | seed 992 的 rep1、rep2：拉扯 66–71 步后撞到身后的第一根柱（柱心 (1.50, −0.03)，方位 +158° 与 +174°，全在视锥外） |

**与 v0 报告 §5.1 的差异**：§5.1 预测规则 1 的例外分支（回转侧最外 2 列 d < 0.5 时不转）恒走 SAFE_NEUTRAL，会悬停。数据里 R1-safe 与 R1-fwd 的触发次数都是 0，例外分支一次也没有进。我当时只做了静态推导，没有估计这条分支的进入概率，也没有想到 R1（|n| >= 2 回转）与 R2（受阻转向）本身会互相抵消。v1 的设计让「受阻时向两边转开」与「转回」共用同一个通道判据，所以 v1 避开了这两个 v0 机制；但「两条相反规则在同一位置来回」这个结构在 v1 里换了形式还在（§7）。

收尾时间线（来源 `rulev1/v0_finish.log`、`$V0/run.log`）：

| 时刻 | 动作 |
| --- | --- |
| 20:42:39 前后 | `kill -STOP 1348145`（xargs，防止再起新臂）；只给 rule81-rep0 的 eval 进程发 SIGTERM（PID 是当时用 ps 取的，没有存档），它的处理函数收掉自己的 gz 与 collect；`run.log` 记该臂 20:42:39 以 `rc=0 SR=0.25 (5/20); completion=0.875 (7/8 attempted)` 结束：已出 7 集（成功 5），第 8 集 seed 230 中断 |
| 20:43:03 | rule81-rep0 目录与日志改名 `rule81-rep0.cancelled-204303[.log]`，起 `rulev1/v0_finish.sh` 守护进程等 rep2（`v0_finish.log` 第 1 行，eval pid 1422743） |
| 20:56:06 | `run.log` 记 rule60-rep2 结束：`SR=0.6 (12/20)`，20 集 |
| 20:56:14 | `v0_finish.log`：rep2 的 eval 进程已退出，`eval.json` 检查 ok；随后 `kill -TERM -- -1348128`，并 `kill -CONT 1348145` 让挂起的 xargs 能收到信号 |
| 20:56:22 | 停守护进程后的残留检查：`[]` |
| 20:56:27 | 二次检查残留 `[]`；`summarize.py` rc=0；`$V0/DONE` = `rc=0 2026-11-10 20:56:27 rule60 3 次重复完成；rule81-v0 取消（主会话 20:40 决定：v0 失败来自规则 1 例外分支的悬停，与视场无关）` |

统计脚本 `rulev1/episode_stats.py`（读 `actions.jsonl` 的 `action_executed` 与 `rule_entry`；动作 0/1/2/3 = SAFE/FWD/LEFT/RIGHT；换向 = 相邻两步分别是 2、3；n = 累计 2 减 3），对 `$V0/rule60-rep{0,1,2}` 的输出在 `rulev1/episode_stats_v0_rule60.txt`：60 集 2990 步，SAFE 0，含 3 连续 SAFE 的集 0，每集换向均值 21.90，条目占比 R3 0.478、R2-sum 0.276、R1 0.235、R2-last 0.006、R2-left 0.005，与上表一致；末 20 步换向 15–19 次的集共 15 个，其中 13 个未到线、2 个是 seed 992 的撞柱集。（上表第一次是在写本报告之前的会话里用内联片段算的，这里用落盘脚本复核，数字一致。）

## 2 v1 的实现

坐标与角度约定同 `rule_policy.py` v0 与 `teacher.py`：世界 x 前、y 左，yaw 度，0 = +x，正 = 左转；方位正 = 左；列从左到右排；相机在机体前 0.045 m（`sim/dynamics/e1_mini.json` `camera_mount`）。

**`sim/tools/rule_policy.py`（新增，v0 部分不动）**

- `RULE_V1`（文件顶部，一行一个，行尾写依据）：`r_safe 0.35`、`d_look 1.2`、`yaw_step_deg 26.0`（308 个转向步实测 26.12° ± 1.09°）、`mem_steps 8`、`n_cap 3`、`tie_frac 0.10`、`sum_clip 4.0`。`V1_ENTRIES` 列出 7 个条目。`n_cap` 的依据主会话没有给，我写成「主会话设计值」。
- `obstacle_points(d, hfov_deg, col_deg=3, max_range=4, cam_forward=0.045)`：每个 `d_j < max_range` 的列出一个机体系点 `(d_j cos φ_j + cam_forward, d_j sin φ_j)`，`φ_j = (k/2 − 0.5 − j)·3°`；列数必须等于 `hfov_deg / 3`（60° → 20、81° → 27），对不上抛 `ValueError`。
- `points_to_world`、`points_to_body`（互逆）、`update_memory(memory, step, new_world_points, mem_steps)`：记忆是 `[(X, Y, 产生步号)]`，丢弃年龄 `step − 产生步号 > mem_steps` 的点（年龄 = mem_steps 的点还在，下一步才丢），并入本步新点（年龄 0）。
- `corridor_clear(points_body, θ)`：没有点满足 `0 < u < d_look` 且 `|v| < r_safe`（严格不等号），`u = x cosθ + y sinθ`，`v = −x sinθ + y cosθ`。
- `rule_v1_entry(points_body, d, n, last_turn)` 按交办的决策顺序出 (动作, 条目)：`n≠0` 且转回通道畅通 → 转回 `V1-back`；当前航向通道畅通 → `FORWARD_SLOW` `V1-fwd`；否则 `n≠0` 且 `|n| < n_cap` → 沿 n 的符号再转 `V1-more`，`n=0` → 左右两半 `Σmin(d, sum_clip)` 取大者 `V1-side`、相差不到 `tie_frac` 沿用 `last_turn` `V1-last`、没有则向左 `V1-left`，`|n| >= n_cap` → 转回 `V1-cap`。永不输出 `SAFE_NEUTRAL`，`FORWARD` 只在 `clear(0)` 时输出。
- `rule_v1_step(memory, step, x, y, yaw_deg, d, n, last_turn, hfov_deg)`：collect 与离线模拟共用的一步（本步列距离 → 世界系点并入记忆 → 全部点换回当前机体系 → 规则），返回 (动作, 条目, 新记忆, `{"mem_n", "clear": [clear(+α), clear(0), clear(−α)]}`)。这个函数不在交办的函数清单里，是为了让 collect 与模拟走同一份代码而加的。

**`sim/tools/collect.py`**：`--rule-version {v0,v1}`（默认 v0，v0 的命令行、行为、落盘逐位不变，测试锁住）；v1 时主循环建一个空的记忆列表（每次进程一集，天然按集清零），逐步传给 `collect_one`，后者原地更新并多记 `rule_mem_n` 与 `rule_clear`（`[左转一步, 保持, 右转一步]` 三个布尔）；manifest 的 `rule_oracle` 在 v1 多一个 `version: "v1"` 键并记 `RULE_V1`，`planner = "rule_v1"`；stderr 步行多 `mem=` 与 `clear=`。记忆用真值位姿（`pose_before`、`yaw_before`）换算，不依赖 n。`--rule-version` 只能和 `--policy rule-oracle` 一起用。

**`sim/tools/eval_closed_loop.py`**：`--rule-version` 透传（只在 v1 时进命令行，v0 的命令行逐字不变），`eval.json` 顶层在 v1 另记 `rule_version`，`rule_params` 记 `RULE_V1`；`course_base` 走同一条路径（collect 只认 `--scene-meta` 里的 `pillars`，课程的柱子列表是跨块合并的）。

**`sim/tools/rule_sim.py`（新增）**：运动学模拟，前进 0.25 m、转向 26°、转向步不位移、`SAFE_NEUTRAL` 原地不动；前进段每 5 cm 查一次离柱面距离，进入 0.102 m 以内算撞柱；成功线 = 最远柱 x + 1.0，走廊 |y| ≤ 3 m。`python sim/tools/rule_sim.py --rule v1 --count 20` 就是 train 分区的离线检查。`train_geometries(count)` 按 `split_manifest.geometry_gid` + `split_for_gid` 取 train 分区前 count 个互不相同的几何。

## 3 测试（先红后绿）

| 文件 | 项数 | 红（实现前） | 绿 |
| --- | --- | --- | --- |
| `sim/tools/test_rule_v1.py` | 40 | `ModuleNotFoundError: No module named 'rule_sim'`（`rulev1/red_test_rule_v1.txt`） | `Ran 40 tests in 5.244s OK`（21:04 重跑，含离线检查输出行，`rulev1/green_test_rule_v1.txt`；加第 40 项之前的 39 项版本存为 `rulev1/green_test_rule_v1_39tests.txt`） |
| `sim/tools/test_rule_v1_wiring.py` | 18 | `FAILED (failures=4, errors=12)`（`rulev1/red_test_rule_v1_wiring.txt`） | `Ran 18 tests OK`（`rulev1/green_test_rule_v1_wiring.txt`） |
| `sim/tools/test_rule_policy.py`（v0） | 27 | | `Ran 27 tests OK`（`rulev1/green_v0_test_rule_policy.txt`） |
| `sim/tools/test_rule_oracle_wiring.py`（v0） | 16 | | `Ran 16 tests OK`（`rulev1/green_v0_test_rule_oracle_wiring.txt`） |

`test_rule_v1.py` 覆盖：`obstacle_points`（列方位、`max_range` 列不出点、81° 27 个点、列数与视场对不上抛错、点落在柱面上）；机体系 ↔ 世界系换算（已知旋转 + 200 组随机往返）；记忆过期边界（年龄 8 在、9 丢）与纯函数性；`clear(θ)` 的边界（`u = d_look`、`|v| = r_safe`、`u = 0` 都不算）与候选航向旋转，并与逐字公式在 3000 组随机输入上互相核对；决策顺序与 7 个条目各一项、`n_cap` 边界（n=2 还能再转、n=3 转回）、平局边界（差恰为 10% 不算平局）、奇数列中间一列不计入两半；6000 组随机输入上「永不 SAFE、FORWARD 只在 `clear(0)` 时输出、7 个条目全部出现」；三个 clear 的顺序在非对称场景下是 [左, 保持, 右]；单柱在正前方 1.0 m（柱面 1.0 m 与柱心 1.0 m 两种）与左右偏 0.15 m 的绕行（离柱面始终 ≥ 0.102 m、≤ 40 步、最终 n 回 0、连续换向少于 4 次、每个前进步都过 `clear(0)`）；模拟自身的力学（前进 / 转向 / 碰撞 / 距离 / 换向游程）；train 分区 20 个几何的离线门槛（≥ 18/20）与长课程柱子列表上的运行。

`test_rule_v1_wiring.py` 覆盖：`collect_one` 的 v1 行（条目、`rule_mem_n`、`rule_clear`、记忆原地更新）；记忆用真值位姿换算并按年龄过期；81° 点数；缺 `rule_mem` 时本步失败；v0 行不带新键；`action_row` 只在 v1 写两个新键；主循环每集传同一个记忆列表且首步为空；manifest 的 `version` 与参数；`--rule-version` 校验；`eval_closed_loop` 的透传、`eval.json` 记录、v0 与其它策略的命令行与报告逐位不变；`course_base` 路径。

第 40 项（三个 clear 的顺序）不是先红后绿：它是在 39 项都绿之后，为补「`rule_v1_step` 返回的 `clear` 三元组顺序」这一处无人把守的行加的（用 `square(1.2, ±0.65)` 的非对称场景，断言 `[False, True, True]` 与 `[True, True, False]`）；它的有效性由下面的突变「三个 clear 的顺序对调」红来证明。

**突变检查**（`rulev1/mutate.py`，输出 `rulev1/mutation_v1.txt`，21:03:57 完成）：在临时副本里逐个改坏实现，跑两个 v1 测试文件，期望至少一个变红。19 个突变全部被抓到：

| 类 | 突变（各 1 个，除注明外） | 由谁抓到 |
| --- | --- | --- |
| 通道判据 | `corridor_clear` 横向 `<` 改 `<=`；去掉 `u > 0`（身后的点也挡）；候选航向旋转方向取反 | `test_rule_v1` |
| 坐标 | `points_to_world` 的 y 符号取反；`obstacle_points` 方位符号取反；`obstacle_points` 忽略 `cam_forward` | `test_rule_v1` |
| 记忆 | `update_memory` 过期边界 `<=` 改 `<`；`rule_v1_step` 不带旧记忆 | 两个文件都红 |
| 决策 | `V1-more` 方向取反；`V1-back` 候选航向取反；`V1-cap` 方向取反；`n_cap` 判据 `<` 改 `<=`；`V1-side` 平局判据 `>=` 改 `>`；三个 clear 的顺序对调 | `test_rule_v1` |
| 接线 | `collect_one` 不把记忆写回主循环的列表；主循环不传 `rule_mem`；manifest 的 v1 参数写成 `RULE_V0`；eval 透传 v1 时不带 `--rule-version`；`eval.json` 的 v1 参数写成 `RULE_V0` | `test_rule_v1_wiring` |

最后一行输出 `共 19 个突变，抓到 19`。这只说明测试能抓到这 19 种改动，不说明规则本身对。

## 4 离线检查（运动学模拟，train 分区）

命令（仓库根）：`UAV_RGB_PYTHON=... .venv-tools/bin/python sim/tools/rule_sim.py --rule v1 --count 20`，输出 `rulev1/offline_v1_train20.txt`。20 个几何是 `split_manifest` 判为 train 的前 20 个互不相同的 base 几何：seed 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22（seed 1、5 的几何在 val 或 test，没有用）。只用 train 几何，没有碰 val / test。

| 规则 | 视场 | 过线 | 撞柱 | 说明 |
| --- | --- | --- | --- | --- |
| v1 | 60° | **18/20** | 0/20 | 过线集 31–46 步，离柱面最小距离 0.309–0.434 m（碰撞线 0.102），最长连续换向 ≤ 3（seed 12、16）；没过线的是 seed 3、14，最长换向 63 |
| v1 | 81° | 18/20 | 0/20 | 没过线的是 **seed 11、20**（不是 3、14；11 号 n 停在 −3 与 −2 之间，20 号停在 2 与 3 之间，都是 `V1-more ↔ V1-cap`），过线集最长换向 ≤ 5（`rulev1/offline_v1_81_train20.txt`） |
| v0 | 60° | 1/20 | 0/20 | 19 集没过线，多数停在 x=0.25 处（第一排柱前约 1.25 m）做 `R2-sum ↔ R1` 左右拉扯；仅 seed 15 过线（`rulev1/offline_v0_train20.txt`） |

**门槛判定**：交办写「离线过线数 < 18/20 就先停下，不起 Gazebo 正式实验」。结果是 18/20，不低于 18，所以继续。这是在门槛上，没有余量。这 20 个几何只有 4 族，没过的两个是同一族里偏移只差 4 mm 的两个几何（§7 第 10 条），所以 18/20 的有效样本比 20 少。

**seed 3 与 seed 14 的失败轨迹**（`rule_sim.py` 输出，两个 seed 的卡死位姿完全一致）：

```
seed 3  未到线 steps=80 max_x=1.76 n_final=2 最长换向 63
条目: V1-fwd V1-fwd V1-fwd V1-left V1-more V1-fwd V1-fwd V1-back V1-fwd V1-fwd V1-back V1-fwd V1-fwd V1-fwd
      V1-side V1-more V1-more V1-cap V1-more V1-cap V1-more V1-cap ...（此后 V1-more / V1-cap 交替到第 80 步）
卡死位姿 (x, y, yaw) = (1.76, 0.61, +52°) 与 (1.76, 0.61, +78°) 交替，n = 2 与 3 交替
seed 3 柱子: (1.5, 0.019) (2.0, -1.581) (2.5, 1.619) (3.0, 0.019) (3.5, -1.581) (4.0, 1.619) (4.5, 0.019) (5.0, -1.581)
```

卡死位姿的逐航向分析（`rulev1/stuck_state_analysis.py`，输出 `rulev1/stuck_state_seed3.txt`、`stuck_state_seed14.txt`，两个 seed 逐行相同；用模拟到第 72、73 步的真实记忆，对 7 个相对航向各算一次 `clear`）：

| 状态（位姿 x=1.757, y=0.613） | 决策 | 候选「转回一步」 | 候选「保持」 | 不在候选里的相对航向 |
| --- | --- | --- | --- | --- |
| yaw +52°（n=2，记忆 84 点） | `V1-more`（转到 n=3） | 相对 −26°（世界 +26°）被挡：最近点 u=1.13、v=+0.32，世界 (2.63, 1.40) | 相对 0°（世界 +52°）被挡：u=0.97、v=+0.08，世界 (2.29, 1.43) | **相对 −52°（世界 0°）畅通**，−78° 畅通，+26° 被挡（u=0.96、v=−0.33），+52°、+78° 畅通 |
| yaw +78°（n=3，记忆 78 点） | `V1-cap`（转回 n=2） | 相对 −26°（世界 +52°）被挡，同上一点 | 相对 0°（世界 +78°）被挡：u=0.96、v=−0.33，世界 (2.28, 1.49) | **相对 −78°（世界 0°）畅通**，−52° 被挡，+26°、+52°、+78° 畅通 |

**机理**：转回候选只看一步（θ = ∓α）。n=2 时世界航向 0° 的通道实际畅通，但它要连续转回两步，第一步（世界 +26°）的通道被挡，规则不会转过去；n=3 触到 `n_cap`，`V1-cap` 又把它转回 n=2，两个状态互为后继，位姿不再变化（转向步在模拟里不位移）。seed 3 与 seed 14 的 `clear` 标志、阻挡点个数与位姿逐项相同，阻挡点世界坐标差 0.01 m（几何差 4 mm）。

**挡住转回候选的只有一个物理点**：世界 (2.63, 1.40)，在柱 (2.5, 1.619) 南面（y = 1.419）附近；它在记忆里重复了 5 次（n=2 时）与 4 次（n=3 时），因为原地转向每步都在同一位姿重复观测它。这个点相对世界 +26° 通道的 u = 1.13（比 `d_look = 1.2` 小 0.07 m）、v = +0.32（比 `r_safe = 0.35` 小 0.03 m）。保持通道则被 69 个（n=2）与 66 个（n=3）点挡住，不敏感。所以这个两周期的触发条件是一个点以几厘米的余量落在候选通道里，对厘米量级的位置偏差敏感（§7 第 10 条）。

**模拟的局限**：模拟里转向步不位移，Gazebo 里每个转向步带实际位移（冒烟里 seed 2 第 3→4 步 x 从 0.26 退到 0.14），位置与航向的随机偏差会让两周期自己散开。证据是 v0：同一份模拟里 v0 过线 1/20，而 Gazebo 里 v0 在 test 上走廊内成功 38/60，所以模拟对「原地来回」类卡死偏悲观；v1 在 Gazebo 里的真实发生率没有核实（§5 的 4 个 train seed 全过，包括 3 与 14）。另一个方向的局限：模拟没有渲染、没有 SITL 的速度环误差与相机延迟，只验证规则逻辑，不验证闭环动力学。

## 5 Gazebo 冒烟

`rulev1/smoke.sh`：train 分区 seed 2、10、3、14，rule60 v1，与正式实验同口径与同参数，单路 gz（v0 的 rep2 当时还占 1 个，配额 2），20:52:36 起，20:56:34 结束，`$OUT/smoke/rule60v1-train/`。选法：2 与 10 是离线过线的两个几何，3 与 14 是离线唯一没过线的两个几何，用来看 Gazebo 里是否复现摆动。下表的条目计数、最大 \|n\|、记忆点数、最长连续换向由我读 `$OUT/smoke/rule60v1-train/episodes/seed_*/actions.jsonl`（`rule_entry`、`rule_mem_n`、`action_executed`）统计，片段是一次性的。

| seed | 离线模拟 | Gazebo | 步数 | 条目计数 | 最大 \|n\| | 记忆点数范围 | 最长连续换向 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 过线 33 步 | 过线，无碰撞 | 31 | fwd 27, left 1, more 1, back 2 | 2 | 5–119 | 0 |
| 10 | 过线 31 步 | 过线，无碰撞 | 39 | fwd 29, side 4, more 1, back 5 | 2 | 0–115 | 1 |
| 3 | **未到线 80 步** | 过线，无碰撞 | 31 | fwd 27, side 1, back 2, last 1 | 1 | 1–122 | 0 |
| 14 | **未到线 80 步** | 过线，无碰撞 | 29 | fwd 25, side 1, more 1, back 2 | 2 | 0–98 | 0 |

- 4/4 过线，撞柱 0，越界 0（`corridor: half_width=3m violations=0/4`）；eval 输出 SPL 0.932（n=4）、动作分布 FORWARD_SLOW 108 / YAW_LEFT 11 / YAW_RIGHT 11 / SAFE 0（来源 `$OUT/smoke/rule60v1-train.log` 末尾与 `eval.json`）。4 集条目计数相加与动作分布一致（fwd 108，转向 22）。
- 条目干净：只出现 V1-fwd / V1-back / V1-side / V1-left / V1-last / V1-more，**没有 V1-cap**；最长连续换向 1 次，没有左右拉扯。
- 记忆点数随位置变化：近柱时 98–122 个，过了柱阵后降到 0–5（每步新增点数受 `max_range=4 m` 与视场限制）；出了视场的柱子仍在记忆里（`test_memory_keeps_a_pillar_that_left_the_field_of_view` 在纯函数层验了，Gazebo 里没有单独看）。
- seed 3、14 没有复现离线摆动：Gazebo 里路径在前几步就与模拟分叉（seed 3 第 4 步是 `V1-side` 向右，模拟是 `V1-left` 向左，推测原因是 Gazebo 里起步后 y 已漂到约 −0.17，两半之和不再对称，未核实；seed 14 首个转向同向，转回时机比模拟早一步）。冒烟没有发现实现错误，也没有触发「设计问题就停下」的条件（4/4 过线，无拉扯）。

## 6 正式实验（守护进程）

预注册写在 `experiments/2026-11-10-rule-oracle-v1/run.sh` 文件头（第二轮、v1 在看过 v0 失败集后设计、参数没有在 test seed 上调、冒烟与离线只用 train 分区；预测 rule60v1 ≥ 54/60 且撞柱 ≤ 3/60，证伪 < 48/60；次要 rule81v1 撞柱 ≤ rule60v1、长课程 K=3 整条成功 ≥ 16/20；主比较 3 个；起跑前已知的离线结果、冒烟结果与 v0 失败机制核对）。

| 项 | 值 |
| --- | --- |
| 守护进程 | `bash run.sh`，PID = PGID = **1520162**，20:59:08 起，`$OUT/run.pid` |
| 日志 | `$OUT/run.log`（队列）、`$OUT/<臂>.log`（每臂，行尾 `[seed N] steps=.. collision=.. success=..`） |
| 看进度 | `tail -f $OUT/run.log`；`cd $OUT && $R/.venv-tools/bin/python summarize.py`（随时可跑，缺的臂标「进行中」，写 `$OUT/summary.md`；手动跑默认用工作树的 `sim/tools`，守护进程里的调用用环境变量 `RULE_TOOLS` 指向冻结副本，两份的 `rule_policy.py`、`scene_*.py` 内容相同） |
| 停 | `kill -- -1520162`（只按进程组；`eval` 会收掉自己的 gz 与 collect）；续跑同一条起跑命令，评完的臂跳过，半成品挪到 `<臂>.partial-<时刻>` |
| 冻结代码 | `rulev1/code-cafb826/`（`git archive cafb826`），`$OUT/inputs.txt` 有各文件 sha256 |
| 臂（优先级）| ① `rule60v1-rep{0,1,2}`（60 集，主）；② `rule60v1-course3`（`--task course_base --blocks 3 --steps 240`，20 集）；③ `rule81v1-rep{0,1,2}`（60 集） |
| gz 数 | 2（本任务配额；长课程实验占的 5 个不是我的，由主会话分配） |
| 配额闸 | `run.sh` 起跑前等 `$V0/DONE` 且 `$OUT/smoke` 的 gz 退出（本次起跑时两个条件都已满足，没有等待） |

起跑后 5 分钟（21:04，`$OUT/rule60v1-rep{0,1}.log`）：rep0 与 rep1 共 8 集，全部成功、无碰撞，步数 29–53。

起跑后 15 分钟（21:14，同上日志与 `episodes/seed_*/actions.jsonl`）：rep0 14 集、rep1 15 集，共 29 集，**28 成功、1 未到线、0 撞柱**。唯一的失败是 rep0 的 seed 683：80 步，max_x 4.99 m（成功线约 6.0 m），不是撞柱也不是越界。条目序列 `fwd×3, left, more, fwd, back, fwd×3, back, fwd×11, side, fwd, back, fwd×4, last, more, more, cap, more, cap, …`：第 31 步起 `V1-more ↔ V1-cap` 交替直到第 79 步（49 步），净转向 n 在 2 与 3 之间摆，最长连续换向 48。这 49 步里位置只在 x 4.81–4.91、y 1.08–1.26 m 的小范围内漂（每步位移 0.002–0.038 m），实际航向在 74° 与 91° 之间来回（转向步实测 23.7–30.0°，均值 24.8°，比标称 26° 略小）；`rule_clear` 在这 49 步里每一步都是 `[False, False, False]`（再转开、保持、转回三个候选的通道都被记忆里的点挡住）。这是 §7 第 1 条的缺陷在 Gazebo 动力学下、在 test seed 上的第一次实测：真实的转向漂移与位移没有把两周期打散。这只是前 29 集的一个点，最终发生率要看 60 集，`summarize.py` 第 8 节会把这类集标成「cap 摆动」。

21:16 复查（同上日志）：rep0 16 集（15 成功）、rep1 17 集（17 成功），共 33 集，32 成功、1 未到线（仍是 seed 683）、0 撞柱。

**ETA（估计，没有核实）**：v0 同口径每臂约 25 分钟（rep2 20:30:17 起、20:56:14 止）；v1 起跑后 5 分钟出了 8 集（两路各约 1.2 分钟一集），每臂按 25 分钟估。队列 2 路并行：rep0、rep1 约 21:24 完 → rep2 与 course3 接上（**rule60v1 三次重复约 21:50 出齐**）→ rep2 完后同一路接 rule81v1-rep0（约 22:13 完）、rep1（约 22:38）、rep2（约 23:03）。长课程每集约 100–120 步，按每步约 2.2 秒（本次 v1 实测约 33 步 1.2 分钟推算）约 4–4.5 分钟一集、20 集约 85 分钟，约 22:50 完（掉压中止的集更短）；它先完的话，空出的一路会提前接 rule81v1。全部约 23:05，上下 15 分钟。

## 7 对规则 v1 的观察（设计缺陷与风险，全部没有改）

1. **转回候选只看一步，`n_cap` 处 `V1-more ↔ V1-cap` 形成两周期**（§4，离线 2/20 train 几何）。n=2 时更远的转回航向（两步）实际畅通，但不在候选里。这与 v0 的 `R1:R ↔ R2-sum:L` 拉扯是同一种结构（两个方向相反的规则在同一位置互相抵消），只是换了条目。Gazebo 冒烟（train seed）没有复现；正式实验起跑 15 分钟后在 test 的 seed 683 上出现了（§6，29 集里 1 集，卡在 x≈4.9 处，三个候选通道全被挡）。如果正式结果里未到线集集中在 `V1-more / V1-cap` 交替，summarize.py 第 8 节会按「cap 摆动」标出来。
2. **`clear` 只看记忆里的可见表面点**：方柱只有朝向相机的面有点，背面与侧面没被看过就没有点；转向到柱子另一侧时，通道判定会偏乐观。记忆最多 8 步，柱子出视场超过 8 步后完全忘记，绕大弯时可能转回去撞已忘记的柱子。
3. **记忆用真值位姿（完美里程计）**：这是「上限」实验的前提，真机的里程计有漂移，v1 的记忆部分在真机上拿不到这个精度。结果只能当作「有完美距离与完美里程时规则层的天花板」。
4. **点由「列中心方位 + 该列最近距离」生成**：同一列内最近那条射线的方位可偏离列中心最多 1.5°，2 m 处横向最多约 5 cm，相对 `r_safe = 0.35` 可以忽略，但边界上的通道判定会受影响（`clear` 是硬阈值，没有滞回）。
5. **没有滞回**：`clear` 由于新观测出现而翻转时，可能出现 L/R 换向；离线最长连续换向 3 次（seed 12、16），正式实验的 `summarize.py` 报最长连续换向。
6. **`n` 与真实航向可以不一致**：记忆用真值位姿，候选航向用 `yaw_step_deg × 步数` 假设每步转 26°；Gazebo 里每步实测 26.1° ± 1.1°（RULE_V1 注释里的数，来自主会话给的 heading-guard 数据），累计 3 步误差约 ±2°，远小于 `r_safe` 对应的角裕度，但 n 计数本身不会修正这个偏差。seed 683 的摆动段里转向步均值是 24.8°（23.7–30.0°），比 26.1° 低 1.3°，说明实测转角会随位姿状况变化。
7. **成功判据里的 80 步上限**：v1 成功集 29–53 步，余量充足；摆动类失败会跑满 80 步，和 v0 一样表现为 `max_x` 停在门柱前。
8. **v0 失败机制的修正（§1）**：主会话对 v0 失败的归因（悬停）与数据不符。v1 去掉了回转例外分支、永不输出 SAFE_NEUTRAL，这对悬停有效，但 v0 实际的失败是拉扯（13 集）与 `n_max` 过小造成的越界（6 集）；v1 对这两类靠「转回前先查通道」「`n_cap` 放到 3」间接处理，设计时没有显式针对。这不影响实验的预注册线，只影响读结果时怎么归因。
9. **`run.sh` 文件头里的两处说法已冻结，读时以本报告 §1 为准**：预注册段第一条把 v0 的失败写成「规则 1 回转例外分支恒走 SAFE_NEUTRAL 的悬停」（主会话原话），下面「起跑前已知」段的第三条又写了事后核对的拉扯结论；「起跑前已知」第一条把离线 v0 的 1/20 称为「悬停死锁」，但 `rulev1/offline_v0_train20.txt` 的条目是 `R2-sum / R2-last / R1` 交替，也是拉扯。文件已被守护进程读取，没有改。
10. **base 几何高度同质，20 个几何不是 20 个独立样本**（`rulev1/geometry_families.py`，只读 train 分区）：train 的 20 个几何只有 4 种（车道序列, x 序列）：8 根柱、第二根在左（7 个），8 根柱、第二根在右（6 个），9 根柱、第二根在左（6 个），9 根柱、第二根在右（1 个）；同一族内只差第 1 根柱的 y 偏移 c（−0.098 到 +0.090 m）。离线没过的 seed 3、14 都是「8 根、第二根在右」族，c = +0.019 与 +0.023，几何只差 4 mm，卡死位姿相同；该族另外 4 个几何的 c（−0.098、−0.048、−0.014、+0.057）都过线，最近的两个（−0.014、+0.057）离失败的 c 各 0.033 与 0.034 m。再加上 §4 里挡住转回候选的只有一个余量 0.03 m 的点，这个两周期更像是厘米量级偏移触发的窄窗口，而不是这一族的必然结果（只有 6 个同族几何，不足以画出窗口的边界，未核实）。含义：① 离线 18/20 不能当 18 个独立证据；② Gazebo 里起步偏移与漂移本身就是厘米量级的随机量，会把这个窗口抹开，这与冒烟里 seed 3、14 都过线一致；③ test 的 20 个几何大概率也属于这 4 族，正式实验的 Wilson 区间与 McNemar p 要按「4 族 × 偏移」而不是「20 个几何」来读（test 的族分布没有看，未核实）。
11. **记忆里有重复点**：原地转向的每一步都在同一位姿重复观测同一批点，`update_memory` 不去重，所以 `rule_mem_n`（冒烟里最高 122）是带重复的点数，不是独立障碍点数；`clear` 只判存在性，不受重复影响，但 `rule_mem_n` 不能当作「看见了多少障碍」来读。

## 做了什么

- 收 v0 守护进程：挂起 xargs、取消 rule81-rep0、等 rule60-rep2 出齐后按 PGID 停进程、查残留、改名留证、跑汇总、写 `DONE`（§1，来源 `rulev1/v0_finish.log`）。
- v0 终局统计与失败机制逐集核对（§1），发现并写明与 v0 报告 §5.1 预测不同；在 v0 报告 `reports/ruleoracle.md` 与 `docs/worklog/2026-11-10-2010-rule-oracle.md` 末尾各追加一节「更新」，正文没动。
- 实现规则 v1：`RULE_V1`、`obstacle_points`、`points_to_world`、`points_to_body`、`update_memory`、`corridor_clear`、`rule_v1_entry`、`rule_v1_step`；`collect.py` 与 `eval_closed_loop.py` 接线（`--rule-version`）；`rule_sim.py` 离线模拟。
- 测试 40 + 18 项（先红后绿，第 40 项除外，见 §3），v0 的 27 + 16 项仍绿，19 种突变全部抓到，`check.sh --python-only` 在最终 HEAD 上 rc=0。
- 离线检查（train 分区 20 个几何）：v1 60° 18/20、81° 18/20，v0 1/20；失败轨迹与机理（§4）；train 几何只有 4 族（§7 第 10 条）。
- Gazebo 冒烟 4 个 train seed（§5）。
- 草稿里留了三个分析脚本与输出：`rulev1/episode_stats.py`（逐集动作统计，v0 与 v1 通用）、`rulev1/stuck_state_analysis.py`（离线卡死位姿的逐航向通道分析）、`rulev1/geometry_families.py`（train 几何分族）。
- 写实验入口 `experiments/2026-11-10-rule-oracle-v1/{run.sh,summarize.py,README.md}`，起正式实验守护进程（PGID 1520162）。
- 提交（均在 `p12-rulev1`）：`cafb826` feat(rule)、`b3d06d9` feat(experiments)、`ea25ddf` test（三个 clear 的顺序）、`edac2a9` docs(worklog)（v1 过程文档 `docs/worklog/2026-11-10-2105-rule-oracle-v1.md`，起草时文件名误写成 2110）、`db88fc5` docs(worklog)（v0 过程文档的更新小节）、`9399447` docs(worklog)（把误写的时刻与文件名改成实际的 21:05）。

## 没做什么（每条带原因）

- 没改 v1 的任何参数或结构：交办要求设计缺陷只记录、起跑前冻结（缺陷见 §7）。
- 没在 val / test 几何上跑离线模拟或冒烟：起跑前不窥视 test 几何，避免影响判断；离线与冒烟只用 train 分区。因此 test 的几何族分布、`V1-more ↔ V1-cap` 在 test 上的发生率都未核实。
- 没改 v0 报告 §5.1 正文与 v0 过程文档正文：它们是上一轮交付物，只在文末追加「更新」小节；是否同步改正文由主会话决定。
- 没改 `run.sh` 文件头里的两处旧说法（§7 第 9 条）：文件已被守护进程读取，改正在运行的脚本会让 bash 读错位置。
- 没等正式实验结束：交办要求起跑并确认前几集正常后返回。
- 没对 rule81v1 或长课程做 Gazebo 冒烟：冒烟的 4 个 seed 只覆盖 base 任务 60°；长课程路径靠单元测试（`sample_course(2, 3)` 的离线模拟能走出第 0 块）与 `eval_closed_loop` 接线测试覆盖，Gazebo 里的长课程行为未核实。
- 没做对「v0 悬停 / 拉扯」的前瞻性判断：v0 报告 §5.1 的静态推导没有估计分支进入概率，这是方法缺口，已在 §1 记下。
- 没改 `progress.md`：并行分支都改头部会在合并时冲突，留给主会话合并时统一写。
- 没跑全量 `check.sh`：没改 C++，用 `--python-only`。
- 没把 §5 冒烟统计与 §6 seed 683 的逐步查看落成脚本文件：一次性读 `actions.jsonl`，字段与口径写在文中；`episode_stats.py` 只覆盖动作、条目、换向与 SAFE，不含记忆点数与 `rule_clear`。
- 没把几何族统计放进测试：它是对 train 分区的描述，不是对代码的断言；脚本与输出在 `rulev1/geometry_families.{py,txt}`。

## 验证到哪一步

| 命令或检查项 | 结果 |
| --- | --- |
| `sim/tools/test_rule_v1.py` | 先红 `ModuleNotFoundError`，后绿 `Ran 40 tests in 5.244s OK`（第 40 项是事后补的，靠突变验证） |
| `sim/tools/test_rule_v1_wiring.py` | 先红 `FAILED (failures=4, errors=12)`，后绿 `Ran 18 tests OK` |
| `sim/tools/test_rule_policy.py`、`test_rule_oracle_wiring.py`（v0） | `Ran 27 tests OK`、`Ran 16 tests OK`（实现 v1 之后重跑，v0 行为逐位不变） |
| 突变检查（`rulev1/mutate.py`） | `共 19 个突变，抓到 19` |
| `UAV_RGB_PYTHON=$R/.venv-tools/bin/python UAV_RGB_BUILD_DIR=$D/tmp/build-p12-rulev1 bash tools/check.sh --python-only` | HEAD `db88fc5`：`rc=0 结束 2026-11-10 21:08:41`（`rulev1/check_final.log`）；之后只有 `9399447` 一个文档提交（重命名与改字，没碰 py 与 sh）；`cafb826` 时 rc=0（`rulev1/check_python_only.log`） |
| 离线检查 `rule_sim.py --rule v1 --count 20`（train 分区） | 过线 18/20，撞柱 0/20（等于门槛，未低于）；81° 同为 18/20；v0 1/20 |
| Gazebo 冒烟（train seed 2、10、3、14） | 4/4 过线，无碰撞，无拉扯，无 V1-cap；SPL 0.932 |
| `summarize.py` | 用冒烟数据（4 集）与长课程实验的学生臂（当作长课程与 rule81 的替身）跑通，rc=0；真实 v1 撞柱集的解析（`collision_info`）未验证，还没有 v1 撞柱集（v0 的同款函数验过）；停滞形态函数 `stall_class` 用真实失败集 seed 683 的 `actions.jsonl` 验过，标为「cap 摆动」，尾部文本 `V1-more:L n=2 clear=000 mem=157` 格式正常（走 `eval.json` 的整条汇总链路要等该臂跑完） |
| 正式实验起跑与前期 | 20:59:08 起，PGID 1520162；21:16 共 33 集（rep0 16、rep1 17）：32 成功、1 未到线（rep0 seed 683，`V1-more ↔ V1-cap` 摆动 49 步）、0 撞柱 |
| rule60v1 走廊内成功 ≥ 54/60、撞柱 ≤ 3/60（预注册线） | 未出结果 |
| rule81v1 撞柱 ≤ rule60v1；长课程 K=3 整条成功 ≥ 16/20 | 未出结果 |
| test 几何上 `V1-more ↔ V1-cap` 的发生率 | 33 集里见到 1 集（seed 683）；60 集的最终发生率未出；test 的几何族分布没看，未核实 |
