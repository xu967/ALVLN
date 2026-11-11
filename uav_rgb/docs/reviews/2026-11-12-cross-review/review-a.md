# 交叉审核 A：Gazebo 线核心逻辑的正确性

- 审核对象：主仓库 `/root/claude-code-backup-20261028/code/uav_rgb` main，审核基准 `97c4228`。审核期间 main 前进到 `c09e2fc`，两者之间只有两个文档提交（`git diff --stat 97c4228 c09e2fc -- sim/tools tools` 为空），下文行号对两者都成立。
- 范围：`sim/tools/collect.py`、`eval_closed_loop.py`、`teacher.py`、`rule_policy.py`、`scene_gen.py`、`scene_course.py`、`student.py`、`gz_io.py`，以及为核对调用关系读到的 `cf_sitl_exec.py`、`collect_parallel.py`、`tools/preprocess.py`、`tools/stats_report.py`。
- 重点：W1-0（`--action-latency`）的派发、迟到判定与口径。
- 证据脚本与输出在草稿目录 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review-a/`：`gap_stats.py` → `gap_stats.out`，`stalls.py` → `stalls.out`。两个脚本只读 `$D/exp-action-latency-20261110/` 与 `$D/exp-action-latency-r2-20261111/` 下各臂的 `actions.jsonl` 与 manifest。

## 结论

1. 新协议固定了「观测到派发」这一段，但没有固定「上一动作截止到下一动作派发」这一段。执行器在这段空档里回 SAFE（主动刹停）。L=0.20 时每步对都有 0.100 s 刹停，旧协议中位 0.015–0.025 s。新旧协议之间单步前飞位移差 16.7%、时长只差 8% 的「弹性 2.38」就出在这里（W1-0 报告记了现象，没有给机理）。空档的大小由相机相位与 RGB 到达延迟决定，不受任何开关约束，也不落盘（发现 1）。
2. 端到端线上，卡 E2E-1 用旧协议采的数据训练、用新协议 L=0.20 评测；上面这段刹停就是两者之间的运动学差别。评测端的 OOD 检查按设计不报（协议快照没有 `action_latency_s` 键时不比），`stats_report` 的协议一致性检查也不看这个键。采集侧 `collect_parallel.py`、DAgger 流水线都不支持这个开关，W2-2 的 DAgger 只能在旧协议下采、在新协议下评（发现 2、4）。
3. 迟到判据只看「观测戳到派发戳」，看不到步与步之间的停顿。第二轮过门的标定臂 `cal-L0.20-k3` 里，seed 4 有 frame_dt 1.6 / 2.6 / 3.6 s 三处停顿，都没有记成迟到。这段时间机体在 SAFE 悬停，属于新协议声称已经消除的那类「生效时长随负载变化」（发现 3；data-algo C8 已报过评测集的停顿，这里补充它在新协议下仍然存在，且出现在门的数据里）。

## 发现清单

按严重程度排序：先列会让实验结论出错的，再列会让数据或训练悄悄变坏的，最后是健壮性。

### 1. 新协议在相邻两步之间插入固定的主动刹停，长度由相机相位决定，不受控、不记录

- 位置：`sim/tools/collect.py:870`（下一帧取 `rgb latest_after(prev_rgb_ts)`）、`collect.py:1019-1032`（等到 观测戳 + L 再派发）、`collect.py:1064`（`sil.dispatch(action, action_start_ns)`，observation 语义下起点是观测戳）、`sim/tools/cf_sitl_exec.py:487-499`（`until = t/1e9 + hold_s`，到期回 SAFE）。
- 现象：
  - 第 k 步动作的截止是 观测戳_k + hold，第 k+1 步的派发是 观测戳_{k+1} + L。两者之差就是执行器在 SAFE 里的时长：空档 = L − (hold − frame_dt)，frame_dt = 观测戳_{k+1} − 观测戳_k。
  - `gap_stats.out` 的数如下：
    - frame_dt：第二轮全部学生臂（`acc-L0.20-k{1,3}-s{0,1,2}`，共 4,972 个步对）恒为 0.9 s。
    - L=0.20 时空档恒为 0.100 s；第一轮 L=0.16 时恒为 0.060 s；旧协议（森林三档，不带开关）中位 0.015–0.025 s。
    - 前一步与本步都是前进的步（F→F）的单步位移中位：L=0.20 为 0.213–0.216 m，L=0.16 为 0.238–0.243 m，旧协议为 0.25–0.29 m。
  - frame_dt 恒为 0.9 s 的原因：相机 10 Hz，RGB 经 gz topic/JSON 到达的延迟实测 70–175 ms（`collect.py:93` 注释）。动作窗口结束时（仿真时间），最新到达的帧是 0.1 s 前拍的那一帧。
- 为什么是缺陷：
  - W1-0 的契约是「每步生效时长恒为 hold − L，与场景、负载、策略的耗时无关」（任务卡第 73 行）。这一条成立。但每一步的运动由「生效段 + 一段刹停」组成，刹停段的长度没有被契约覆盖。
  - W1-0 报告第 128-130 行记了「时长短 8%、位移短 16.7%，弹性 2.38」，没有给机理。刹停段可以解释这一点：L 从 0.16 到 0.20，刹停从 0.06 s 变成 0.10 s，同时生效段少了 0.04 s，机体每步要从更低的速度重新加速。
  - 刹停长度依赖 RGB 到达延迟落在哪一侧：
    - 到达延迟超过 100 ms 时，frame_dt 变成 1.0 s，刹停变成 L = 0.2 s。L=0.16 的臂里有这种步：`acc-L0.16-k1-s1` 有 6 个 frame_dt=1.0 的步对，空档 0.16 s，位移中位 0.202 m，对比空档 0.06 s 的 0.240 m。
    - 渲染变快、延迟到 0 ms 附近时，frame_dt 可能变成 0.8 s，L=0.20 下刹停变成 0。
  - 换渲染后端、换机器或改相机频率，都会在不改任何开关的情况下改变每步的运动学。
- 影响：
  - 第二轮的门（K=3 对 K=1）不受影响，两边的 frame_dt 都是 0.9 s。
  - 受影响的是任何新旧协议之间、或不同机器与渲染负载之间的比较，以及卡 E2E-1 的训练与评测（见发现 2）。
- 怎样复现：`nice -n 15 env PYTHONDONTWRITEBYTECODE=1 .venv-tools/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-cards/review-a/gap_stats.py`。脚本对每个步对算 (pose_before_ts_{k+1} + L) − (pose_before_ts_k + hold)，再按空档分桶统计 F→F 的 `displacement_m`。
- 状态：数字已复现；「弹性来自刹停段」是推断。
  - 成立条件：cf_sitl 的 SAFE 是主动刹停而不是滑行。`cf_sitl_exec.py` 的注释写的是位置保持，未在插件源码里核实。
  - 验证方法：同一 L 下按空档分桶比较 F→F 位移（L=0.16 的 0.06 s 与 0.16 s 两桶已经差 0.038 m，样本 n=6，偏少）；或者在 state 话题上记录 IDX_ACTION 的切换时刻与速度。
- 最小修法：
  - 每步行里加 `safe_gap_s` 一列，取 下一步派发戳 − 本步截止，事后可算；`summarize_episode` 报它的中位数与最大值。
  - EVAL_STANDARD 里把「frame_dt 中位 0.9 s、刹停 L − 0.1 s」写成协议的一部分。门里加一条：刹停的中位数不偏离标定值超过 10 ms。
  - 如果要让刹停与相机相位无关，可以让执行器截止取 派发戳 + (hold − L)，这样两步首尾相接。这改变了协议，要重标定，不建议在 E2E-1 进行中改。

### 2. E2E-1 与 DAgger 线：训练数据是旧协议，评测是新协议，评测端不报 OOD

- 位置：
  - 任务卡 `docs/briefs/2026-11-10-phase12-task-cards.md:607-626`：两份数据都是现成的旧协议采集，评测带 `--action-latency L_STAR`。
  - `sim/tools/eval_closed_loop.py:217-223`：`protocol_mismatches` 只比快照里出现过的键。
  - `eval_closed_loop.py:1131`：注释写明快照没有 `action_latency_s` 时视为旧口径，不判 OOD。
  - `sim/tools/collect_parallel.py:400-430`：拼 collect.py 命令行的地方没有 `--action-latency`。`tools/dagger_gz_pipeline.py` 与 `tools/e1_pipeline.py` 里 grep 不到这个开关。
- 现象：A0–A3 四个臂的训练数据都在旧协议下采集，相邻动作之间几乎不刹停，F→F 约 0.26 m，转角 26.12°（W1-0 报告第 58 行）。评测在 L=0.20 下进行：每步刹停 0.1 s，F→F 约 0.213 m（发现 1），转角 25.06°（W1-0 报告第 75 行）。评测 manifest 不会因此报 OOD。
- 为什么是缺陷：
  - 上一轮 sim-real 报告第 217 行已经写过「改……窗口延迟……才会让旧数据作废」。W1-0 也把 `action_latency_s` 加进了 `PROTOCOL_KEYS` 与 preprocess 的 `STRICT_KEYS`。但评测端的比对规则让「旧数据训练 + 新协议评测」这个组合静默通过，问题仍没修。
  - 对 E2E-1 的预注册：
    - M1–M3 是臂间比较，四个臂都受同样的偏移，比较本身仍然公平。
    - 「是否达到 W2-2 判据 48/60」与「带护栏参照 53/60」是绝对数，含协议偏移带来的分布外成分。
    - 航向码按「已执行转向步数」计。每步转角从 26.1° 变成 25.1°，n=2 对应的角度从 52° 变成 50°，影响小。
    - 每步前进量少 18% 改变了图像在相邻帧之间的变化幅度；学生在 prev-action 通道里学到的自运动补偿是按旧幅度学的。
  - DAgger 的作用是覆盖学生自己走到的状态。在旧协议下采、在新协议下评，覆盖的是另一套动力学下的状态分布。卡 W2-2 若照搬 `dagger_gz_pipeline.py`，会把这个偏移带进每一轮。
- 影响：E2E-1 的绝对成功数与 W2-2 判据的比较会混进协议偏移，DAgger 轮次的收益估计也会受影响。偏移的方向与大小未核实。
- 怎样验证：拿 A0 的一个种子，在同一组 20 个冻结 seed 上分别跑「不带开关」与「L=0.20」两遍，逐集配对。11-10 第一轮验收 2 的 L=0.16 对旧协议已经有一次：一致 11/20（W1-0 报告第 83 行），但同协议重评也只有 16/20 一致，样本不足以定量。
- 最小修法：
  - `eval_closed_loop.py:1131` 处：协议快照没有 `action_latency_s`、而本次给了 L 时，在 eval.json 里记一条 `protocol_notes: train_protocol_without_action_latency`，不判 OOD，但报表能看到。约 5 行。
  - `collect_parallel.py:400-430` 透传 `--action-latency`，约 4 行；`dagger_gz_pipeline.py` 跟着透传。
  - 给 preprocess 加对行内 `late_dispatch` 的过滤，约 5 行。现在还没有新协议的训练数据，这一条可以和透传一起做。
  - E2E-1 的报告里把「训练旧协议、评测新协议」写进局限一节。
- 状态：代码路径已读代码确认；对成功率的影响是推断，成立条件与验证方法见上。

### 3. 迟到判据看不到步间停顿；第二轮过门的标定臂里有 3.6 s 的停顿

- 位置：`collect.py:1019-1032`（迟到 = 就绪时位姿戳 > 观测戳 + L + 20 ms）、`collect.py:870`（取下一帧时只要求戳比上一帧新，墙钟等待上限 `FRAME_WAIT = 2.0`，`collect.py:116`）、`eval_closed_loop.py:913`（只按 `late_dispatch_steps` 判失败）。
- 现象（`stalls.out`，frame_dt ≥ 1.0 s 的步对，都没有被标迟到）：
  - 第二轮 `cal-L0.20-k3`：seed 4 有 frame_dt 1.6、2.6、3.6 s 三处（步 9→10、66→67、29→30），另有 1.0–1.1 s 若干。停顿之后那一步的位移：前进步 0.065 m 与 0.154 m，正常约 0.215 m。
  - `cal-L0.20-k1` seed 7 有一处 1.5 s，之后前进步位移 0.073 m。
  - B 组 `cal-L0.20rtf0.5-k3` 有 4 处 1.0–1.2 s。
  - 第一轮 `acc-L0.16-k1-s1` 有 15 处 1.0–1.2 s。
  - 第二轮 `acc-*` 学生臂没有停顿。
- 为什么是缺陷：
  - 停顿期间执行器早已到期，机体在 SAFE 里悬停；下一帧来了以后，派发仍然准时在 新观测戳 + L，于是 `dispatch_lag_s` = 0.200，不迟到。
  - 从数据上看，这一步的「起始速度」与正常步不同（从静止起步），属于新协议声称已经消除的那类「随负载变化的时序」。
  - data-algo C8 已经指出评测集里有这种停顿，建议加 `max_frame_gap_s`，仍没修。这里补充：停顿在新协议下仍然存在，而且出现在 W1-0 第二轮过门所依据的标定臂里。
- 影响：
  - 对门的影响估算：`cal-L0.20-k3` 共 820 个步对，F→F 稳态位移的样本 n=654，受停顿影响的前进步不超过 3 步，对中位数的影响不到 0.5%，不改变过门结论。
  - 对以后在负载更重的机器上做的评测，这是一个不报警的口子。
- 怎样复现：`stalls.py` 列出 frame_dt > 0.95 s 的步对，并附上 late、lag、动作与位移。
- 最小修法：
  - 在 collect 的每步行里记 `obs_gap_s`，取 本观测戳 − 上一动作截止。
  - eval 的 `summarize_episode` 汇总最大值；超过阈值（建议 0.3 s，即正常值 L − 0.1 再加 0.2 s 余量）记 `stall_steps`，与 `late_dispatch_steps` 同等处理，进入 failed[]。约 10 行。
- 状态：已复现。

### 4. 新旧协议的 eval.json 能混进同一张表，不报警

- 位置：
  - `tools/stats_report.py:370`：一致性检查的键只有 `metric_version, success_criterion, hold_s, dynamics, vehicle, window_from, task`。
  - `sim/tools/eval_closed_loop.py:1303`：`report_metric_version` 只拼 `sim09.v2[.coast1s][.corridor3m]`，不含 L。
  - `eval_closed_loop.py:1375-1378`：只有给了开关才在 eval.json 顶层写 `action_latency_s`。
- 现象：同一个 `metric_version` 下，L=0.20、L=0.16 与不带开关的 eval.json 都能被 `stats_report` 当成同一口径汇总；`rtf` 与 `heading_guard` 也不在检查键里。
- 为什么是缺陷：W1-0 报告与 `--action-latency` 的 help 文字（`eval_closed_loop.py:1097`）都写了「新旧 SR 不可混报」，但汇总工具不执行这条规则。只靠人记得分开。
- 影响：之后的报告从多个实验目录取数时，可能把旧协议的成功率与新协议的放进同一个 McNemar 或同一行。
- 怎样验证：读代码确认。可以构造两份只差顶层 `action_latency_s` 的 eval.json 喂给 `stats_report`，预期不报错（未跑）。
- 最小修法：`stats_report.py:370` 的键元组加 `action_latency_s`、`heading_guard`、`rtf`，`load_eval`（234 行起）把这三项读进来。约 4 行。
- 状态：读代码确认，未跑。

### 5. `dispatch_lag_s` 与迟到判定用的是「已到达 Python 的位姿戳」，不是插件收到动作的仿真时刻

- 位置：`collect.py:1024-1026`（`ready_ts = subs["pose"].latest()` 的戳）、`collect.py:1036-1037`（`exec_ts` 同样取最新到达的位姿）、`collect.py:1244`（`dispatch_lag_s = (exec_ts − pose_before_ts)/1e9`）、`sim/tools/gz_io.py:125-141`（`latest_after` 只看到已经读进缓冲的帧）。
- 现象：
  - 位姿经 `gz topic -e --json-output` 子进程送进 Python，有一段到达延迟；派发消息从 Python 到插件又有一段发布延迟。两段都不在 `dispatch_lag_s` 里。
  - 第二轮所有学生臂的 `dispatch_lag_s` 中位与最大值都是 0.200（`gap_stats.out`），这是按构造得到的：程序等的就是这个戳。
  - 插件实际收到动作的仿真时刻没有逐步落盘。state 话题里有 IDX_ACTION（4）与 IDX_SIM_T（32），没有被记录。
- 为什么是缺陷：门里的「生效时长 hold − L」用的是 oracle 的 `action_end_ns − action_start_ns − dispatch_lag_s`（W1-0 报告第 54 行），三项都来自同一个 Python 侧时钟，在门里无法证伪。如果到达延迟随负载变化，实际生效时长就跟着变，记录上看不出来。
- 影响：现在的数据里看不到这个问题（frame_dt 与位移都稳定），所以列在发现 1–4 之后。它会在重负载或换机器时让「迟到 0」这一条失去意义。
- 怎样验证：在 collect 的轮询循环里顺带订阅 state 话题，记录每步 IDX_ACTION 切换到本步动作时的 IDX_SIM_T，与 观测戳 + L 比较。或者离线比较 `oracle` 里的位姿轨迹开始偏离 SAFE 的时刻。
- 最小修法：每步行里加 `plugin_ack_sim_t`，取 state 话题上 action 字段第一次等于本步动作时的 sim_t；迟到判据改用它。约 15 行。
- 状态：推断。成立条件：位姿管道的到达延迟大于 5 ms（位姿 200 Hz 的一个网格）。

### 6. `rule_v1` 与 heading guard 的转角常量按旧协议标定

- 位置：`sim/tools/rule_policy.py:32`（`yaw_step_deg = 26.0`，注释写的是 heading-guard 数据实测 26.12° ± 1.09°）、`rule_policy.py:22`（`n_max = 2`，「约 54°」）。
- 现象：新协议 L=0.20 下每步转角是 25.06°（W1-0 报告第 75-76 行）。`rule_v1` 的「转回一步」通道按 26° 检查，实际转回 25°。
- 影响：
  - 1° 的差在 1.2 m 通道末端约 2 cm 横向偏差，小于 `r_safe` 余量 0.25 m，不改变行为。
  - 「n=2 约 54°」在新协议下约 50°。规则臂若在新协议下与旧结果对比，要知道这一点。
- 最小修法：注释注明标定协议；新协议下的规则实验把 `yaw_step_deg` 改成按 L 查表，或在报告里写明。
- 状态：读代码确认；对规则成功率的影响是推断，量级按上面的几何估算。

## 查过但没有发现问题的地方

- 偏航与坐标的符号约定：
  - `collect._yaw_deg` 用 atan2(2(wz+xy), 1−2(y²+z²))。`teacher._pursuit_err_deg` 正值表示目标在左，YAW_LEFT 为 +wz。
  - `rule_policy.column_distances` 第 j 列方位 (n/2 − 0.5 − j)·col，正值在左，j=0 是最左列。`points_to_world` 与 `points_to_body` 互逆。
  - `heading_guard_action`（`collect.py:808`）回转方向与 `net_yaw` 符号一致。
- A\* 预热（`teacher.py:372` `warm_free_cells`）与按需填格用同一把缓存钥匙（clearance、step、柱几何）、同一栅格范围（`_grid_spec`）、同一判定函数。预热不会让结果与不预热时不同。预热只在 `planner_kind == "astar"` 时做（`collect.py:1757`），costmap 教师不做，不影响 W1-0 的臂。
- defer 标签：派发之后再算的影子标签（`collect.py:1081-1090`）与派发前的标签用同一个函数、同一组参数；失败时调 `stop_executor`。
- 迟到集的记账：`scan_episodes`（`eval_closed_loop.py:880` 起，913-937 行）把含迟到步的集放进 failed[]，占请求名额，不放进 `seen`，重跑不会重复计数。`success_rate` 的分母是 done，`requested_success_rate` 的分母是 requested，两者都写进了 eval.json。
- 走廊判据：`summarize_episode` 用 |y| ≤ 3.0（世界 y=0）。`scene_gen` 的柱子 y 在 ±1.7 以内，中心车道摆动不超过 0.1（`scene_gen.py:64-67`），出生点 y=0；`scene_course.shift_block`（`scene_course.py:50-52`）只沿 x 平移，没有横向偏移。所以世界 y=0 就是柱阵中线，判据与课程一致。
- 课程成功线：`block_goal_x` 每块取最远柱 x + 1.0。块间距 5.0 m，块内柱 x ∈ [1.5, 5.0]，所以第 k 块的成功线（≤ 6.0 + 5k）在第 k+1 块第一根柱（≥ 6.5 + 5k）之前；自检（`scene_course.py:190`）锁了整条课程的成功线。
- `--rtf`：`scene_gen.set_real_time_factor` 要求模板里恰好匹配 1 处；给了开关才写 `meta.real_time_factor` 并重算 `world_sha256`；eval.json 顶层记 `rtf`（`eval_closed_loop.py:1383`）。默认输出不变。
- `student.py`：预处理（PIL bilinear 缩放）、首帧 prev=SAFE、prev-action 取上一步执行的动作，与 `tools/preprocess.py:258-267`（prev 取上一行的 `action_executed`，缺了退回 `action_expert`，首帧 SAFE）一致。
- 失败路径：每集是一个新的 collect 进程加一个新的世界，`collect.py` 的 `finally` 先 `stop_executor` 再 `sil.stop`；`stop_executor`（`collect.py:359-368`）自身不抛异常。没有看到跨集残留的状态。
- `LATE_DISPATCH_TOL_NS`（`collect.py:124`）为 20 ms，与 W1-0 报告里的阈值 L + 0.02 s 一致；等派发的墙钟兜底是 L / `MIN_WINDOW_RTF`，超时返回错误，不会静默派发。

## 没有查的地方与原因

- cf_sitl 插件（`sitl/gz/CfSitlSystem.cc`）与库里 SAFE 的实际行为：只读了 Python 侧注释。发现 1 的机理依赖它，已在条目里写明成立条件。
- 没有起 Gazebo，没有跑任何闭环：按任务书的边界。发现 2、5 的定量影响因此只能是推断。
- `teacher.py` 的 costmap 教师与 `lr_tie_margin_steps` 推演：W1-0 的臂都是 A\* legacy tiebreak，本轮没有逐行审。
- `collect.py` 的 `dyn_twist`（x3_velocity 档）与 cf_sil 分支：已退出主线。
- `eval_closed_loop.py` 的参数段（1013-1085 行）只抽查了与 `--action-latency`、`--rtf` 相关的几行。
- 3DGS 线的同名逻辑：不在本角度范围。

## 做了什么 / 没做什么 / 验证到哪一步

- 做了：
  - 读了范围内 8 个文件中与派发、窗口、判据、坐标相关的段落，以及 `cf_sitl_exec.py`、`collect_parallel.py`、`preprocess.py`、`stats_report.py` 的相关段落。
  - 写了两个只读统计脚本，用 `nice -n 15` 与 `timeout 120` 各跑一次，rc=0，用时 1.5 s 与 0.6 s。
- 没做：
  - 没有改仓库里的任何文件，没有提交，没有建分支。
  - 没有起 Gazebo，没有训练，没有跑 `tools/check.sh`，没有拿锁。
  - 没有往仓库写过程文档：用户的只读边界优先，本节代替。
- 验证到哪一步：
  - 发现 1、3 的数字已用实验目录里的记录复现。
  - 发现 2、4、6 的代码路径已读代码确认，对成功率的影响未核实。
  - 发现 5 是推断，验证方法在条目里。
