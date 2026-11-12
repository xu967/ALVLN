# 第二轮交叉审核报告（角度 K：固件在环与执行器）

- 审核对象：主仓库 `/root/claude-code-backup-20261028/code/uav_rgb` 分支 `main`。开工时 `main` 为 `ada0ca5`（`2026-11-12 07:58:12 CST`）；审读期间主会话合入了卡 W1-15a（`f2c1107`）、卡 W1-13 第一二部分（`e40c693`，改了 `sim/tools/cf_sitl_exec.py`）与卡 W1-15b（`035b398`），`main` 前进到 `fb3d1a5`。本报告对 `sim/cf_sitl/`、`sim/dynamics/e1_mini.json` 的行号在 `ada0ca5` 与 `fb3d1a5` 上完全相同（其间仅 `sim/cf_sitl/gz/README.md` 在 `41c474d` 增加 5 行说明）；对 `sim/tools/cf_sitl_exec.py` 与 `sim/tools/collect.py` 同时覆盖 `e40c693` 合入的新代码并标明提交号。
- 固件原件：`/root/claude-code-backup-20261028/code/CF-Drone`（`e7f507f`，`git diff --ignore-cr-at-eol` 核实除 CRLF 行尾外零改动）。
- 起止时刻：`2026-11-12 07:58 CST` 至 `2026-11-12 08:32 CST`。
- 草稿与实测产物目录：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-k/`（探针脚本 `probe_w113.py`、结果 `probe_w113_results.json`、单测与自检日志 `sitl-tests.log`、`selftest-s1/selftest.txt`）。

## 1. 结论

本角度下最要紧的三件事：

1. **卡 W1-13 第三部分拟议的 `until = exec_ts + P`（不加管道余量）在 Gazebo 里做不到「步间不回 SAFE」（发现 `K-1`）**：Python 等到已到达的位姿帧戳 `>= o_{k+1} + L`（即 `exec_ts[k] + P`）才发下一条动作，此时 Gazebo 物理时钟早已走过 `exec_ts[k] + P`，再加 `gz topic -e` 子进程与 `--pub-server` 子进程的跨进程往返延迟 $\Delta_{\text{pipe}}$，下一动作到达插件 `DrainActions` 的时刻必然晚于 `exec_ts[k] + P`。执行器按严格不等式 `in->t_s < e->action_until_s` 判定，到点当拍即切回 `CF_EXEC_SAFE`（触发速度参考归零、位置保持外环介入与角速率环微分冲击），而 Python 侧账面 `safe_gap_s = exec_ts[k+1] - until[k]` 却恒显示 `0.000 s` 假过门。发给插件的截止必须加保持余量 `guard_s`（使新动作在旧动作到期前抢占覆盖），并用 `plugin_ack_sim_t` 验证插件侧真实空档。
2. **`--cf-sitl-params` 在采集侧存在单边生效缺口（发现 `K-2`），且满电档（`battery_soc_init=1.0`）起飞 40/40 集全部冲破 ToF 与光流 2.50 m 量程上限（发现 `K-3`）**：卡 W1-10 只打通了 `eval_closed_loop.py`，`collect_parallel.py` 既无 `--cf-sitl-params` 命令行入口也未在 `collect_command` 中向 `collect.py` 透传，且 `collect.py` 不校验 `scene_meta["cf_sitl_params"]` 与自身参数是否一致，直接调 `collect.py` 或后续补半截透传都会造出「Gazebo 飞一套电量、manifest 写另一套指纹」的静默脏数据。同时满电档未联动下调 `hover_duty_estimate=0.824`，`cf_sitl_proc` 实测 40/40 集起飞峰值达 `2.503–2.516 m`，每集经历最长 `300 ms` 的 ToF/光流双盲区，距 `tof_timeout_s = 500 ms` 触发 `FAILSAFE` 仅剩 `200 ms` 余量。
3. **刚刚合入 `main`（`e40c693`）的 `CfSitlExecutor.ack_sim_t()` 有四处边界缺陷（发现 `K-4`），且 `next-dispatch` 下连续转向比单步转向多转 `2.63°`（发现 `K-5`）**：`ack_sim_t()` 在窗口末才回扫 600 帧缓冲，迟到步贴近窗口末时受 `state`/`pose` 双子进程管道竞态影响会假阴性返回 `None`，`--hold >= 3.0` 时必溢出返回 `None`，不记游标在 `next-dispatch` 迟到步同截止时会错配到上一步，且含 `0..+4 ms` 的 200 Hz 采样阶梯上偏。`next-dispatch` 消去步间刹停后，`YAW→YAW` 连续步无起转滞后，整步周期转角达 `30.87°`（单步 `28.24°`、行内 0.8 s 为 `24.81°/27.44°`），标定需双口径并报。

---

## 给 W1-13 第三部分的事实

本节逐条回答任务书补充指定的五个执行器问题、核对上一轮读码笔记 `$D/tmp/phase12-cards/w1-13/handoff-from-39b5006b.md`，并用执行器源码与 `lib31` 数值试验裁决第一轮审核 A 的发现 1。

### A. 插件与执行器五项机制核实

#### 1. 动作截止 `until` 在客户端、插件与执行器里怎样生效
- **客户端计算与发送**（`sim/tools/cf_sitl_exec.py:494-507`）：`CfSitlExecutor.dispatch(action, t_ns)` 计算 Gazebo 仿真秒 `until = int(t_ns) / 1e9 + self.hold_s`（Python `float64`）。在 `.venv-tools` 下经 `_HelperPublisher.action`（`:254-255`）格式化为 `"a %d %.6f"`（`1 µs` 分辨率），由 `/usr/bin/python3` 子进程发布 `gz.msgs.Double_V [action, until]` 到 `/e1_mini/action`。
- **Gazebo 插件入队与换算**（`sim/cf_sitl/gz/CfSitlSystem.cc:86-93, 132-148, 226-229`）：`OnAction` 回调加互斥锁推入 `actions_` 队列。每个 `1 ms` 物理步的 `PreUpdate` 先调 `DrainCmds`，计算 `t_sitl = sim_s - base_s_`（`base_s_` 为本集 `reset` 时的 `sim_s`），再调 `DrainActions(t_sitl)`：对队列每条消息算 `until_sitl = a.until_sim_s - base_s_`；若 `until_sitl <= t_sitl` 则递增 `late_actions_` 并打印警告，但仍调 `cf_sitl_set_action(a.action, until_sitl)`。同一 `1 ms` 步内若到达多条消息，按入队顺序依次调用，末条生效。
- **SITL 单精度截断与每拍严格不等式判定**（`sim/cf_sitl/src/cf_sitl.cpp:49, 116-118`；`sim/cf_sitl/exec/executor.c:38-42, 180-185`）：
  - `cf_sitl_set_action` 将 `double until_s` 强转为 `float`（IEEE 754 binary32，在 `128–256 s` 区间 ULP 为 `1.53e-5 s`）写入 `e->action_until_s`；每步 `tick()` 将 `s.t` 强转为 `float` 写入 `in.t_s`。
  - 在 `CF_EXEC_READY` / `CF_EXEC_ACTING` 分支执行严格小于判断：`active = (e->action >= 0 && in->t_s < e->action_until_s)`。
  - **半开区间边界**：动作有效区间是 `[t_arrive, action_until_s)`。当 `in->t_s == e->action_until_s` 的那一个 `1 ms` 步到来时，若新动作尚未在同一拍 `DrainActions` 中更新 `action_until_s`，`active` 当拍即变为 `0`，执行器切回 `CF_EXEC_SAFE`、`e->state` 切回 `CF_EXEC_READY`（对比 3DGS 线 `sim/3dgs/dynamics.py:255-258` 专门多加 `+0.5 ms` 以避免窗口末拍掉回 `SAFE`）。

#### 2. 到期之后回 SAFE 是位置保持还是滑行，刹停的加速度多大
- **结论**：到期回 `SAFE` 是**主动刹停加位置保持（锁住到期瞬间的空间点并把前冲段往回拉）**，不是自由滑行。
- **代码位置与控制律**：
  - `sim/cf_sitl/exec/executor.c:182-195`：过期当拍 `act = CF_EXEC_SAFE`，因 `act != CF_EXEC_FORWARD` 进入 `position_hold(e, in, &vx, &vy)`（`:126-136`）。
  - `FORWARD` 期间 `e->holding` 恒为 `0`（`:192`），故过期首拍触发 `!e->holding`：将 `e->pos_err[0..1]` 清零并置 `e->holding = 1`，即以过期时刻的位置作为保持原点。此后每 `1 ms` 在机体系积分滤波光流速度 `pos_err[0] += (vel_est[0] + wz * py) * dt`，输出反向速度参考 `vx_ref = clamp(-0.5 * pos_err[0], -0.3, 0.3)`（`e1_mini.json:85-86`）。
  - `horizontal_loop`（`:97-120`）：过期首拍 `vx_ref` 从 `+0.30 m/s` 跳至 `0.0 m/s`（随后随前冲变为负值），速度误差 `err ≈ -0.30 m/s`，P 项立即输出反向抬头角 `vel_kp * err = 0.22 * (-0.30) = -0.066 rad = -3.78°`，叠加 `pos_err` 负速度参考与 `vel_ki = 0.30 rad/m` 反向积分。
- **刹停减速度与数值实测**（`probe_w113.py` 调 `$D/tmp/build-cf-sitl-lib31/cf_sitl_proc`，结果见 `probe_w113_results.json`）：
  - `FORWARD` 巡航 `3.0 s`（进入速度 `0.270 m/s`）后自然到期回 `SAFE`：峰值水平减速度为 **`-0.671 m/s²`**（出现在到期后 `210 ms`，因固件姿态环 `PITCH_P=6` 与有刷电机降速时间常数 `motor_tau_down = 0.12 s` 存在滞后）；`565 ms` 时前向真值速度过零点，最大前冲距离 **`0.087 m`**；随后位置保持外环驱动机体反向回退，最大回拉速度 **`-0.146 m/s`**。
  - 在 `L=0.20` 的周期性 `800 ms FORWARD + 100 ms SAFE` 下（`cycle_trace`）：`ms=800` 到期当拍 `pitch_stick` 从 `+0.0404` 阶跃跳至 `-0.0873`，到 `ms=900`（`0.1 s` 空档末）拉至 `-0.1512`（目标抬头角 `-4.54°`），真值俯仰角翻至 `-1.04°`；下一拍新 `FORWARD` 虽到达，机体仍因负俯仰角速率继续后仰至下一步 `ms=50` 的 `-1.72°`（周期内峰值减速度 **`-0.400 m/s²`**），速度在下一步 `ms=150` 跌至谷底 **`0.243 m/s`**，到 `ms=550` 才恢复至 `0.29 m/s`。

#### 3. 新动作在旧动作没到期时到来：是否立刻取代、速度参考有无跳变
- **结论**：新动作在到达后的下一个 `1 ms` `PreUpdate` 物理步**立刻无条件取代**旧动作；不同动作间切换时速度或偏航角速度参考**立即阶跃跳变**，相同动作续接时参考值**零跳变**。
- **代码位置**：`CfSitlSystem.cc:138-147` 排空队列调 `cf_sitl_set_action`；`executor.c:38-42` 直接覆写 `e->action = action; e->action_until_s = until_s;`，不触碰 `holding`、`pos_err`、`vel_est`、`vel_int` 等滤波与积分状态。
- **各切换组合的参考跳变**（`executor.c:184-195`）：
  - `FORWARD` 未到期切 `SAFE` 或 `YAW_*`：`vx_ref` 从 `+0.30 m/s` 阶跃跳至 `0.0 m/s`（首拍置 `holding=1` 清零 `pos_err`，后续变为负值回拉）；切 `YAW_*` 时 `s.yaw` 同步从 `0` 阶跃至 `∓0.1146`（对应 `±0.6 rad/s`）。
  - `SAFE` 或 `YAW_*` 未到期切 `FORWARD`：`holding` 置 `0`，`vx_ref` 从 `-0.5 * pos_err[0]` 阶跃跳至 `+0.30 m/s`，`vy_ref` 跳至 `0.0 m/s`，`s.yaw` 阶跃回 `0`（固件 `control.ino:162` 锁住切换瞬间的估计航向）。
  - `YAW_LEFT` 未到期切 `YAW_RIGHT`：`s.yaw` 从 `-0.1146` 阶跃至 `+0.1146`，`holding` 维持 `1` 不重置（保持点仍为最初进入非 `FORWARD` 时的位置）。

#### 4. 连着两步同一个动作首尾相接时状态机怎么走
- **真无缝覆盖（下一条动作在 `in->t_s < old_until_s` 期间到达插件）**：
  - `FORWARD → FORWARD`：`active` 恒为 `1`，`e->state` 全程保持 `CF_EXEC_ACTING(3)`，不经 `CF_EXEC_READY(2)`；`holding` 恒为 `0`，`vx_ref` 恒为 `+0.30 m/s`；`vel_est`、`vel_int` 与固件姿态/角速率 PID 连续运行，与单条 `1.8 s` 的 `FORWARD` 逐位等价。
  - `YAW_LEFT → YAW_LEFT`（或右转接右转）：`e->state` 全程保持 `CF_EXEC_ACTING(3)`，`s.yaw` 恒为 `∓0.1146`，固件不锁死 `yawTarget`、不打反向刹车力矩，`holding` 保持 `1` 持续锚定同一水平点原地匀速旋转。
  - `SAFE → SAFE`：`e->state` 保持 `CF_EXEC_ACTING(3)`，`holding` 保持 `1` 不清零 `pos_err`（即使中间过期回 `CF_EXEC_READY`，控制分支同样执行 `position_hold`，`holding` 仍为 `1` 不重置保持点，仅遥测 `state[1]` 与 `state[4]` 翻转）。
- **伪首尾相接（下一条动作比 `old_until_s` 晚到 $\Delta t \ge 1\text{ ms}$）**：
  - 在漏出的 $\Delta t$ 内，`e->state` 掉回 `CF_EXEC_READY(2)`、`state[4]` 变 `-1`。对 `FORWARD → FORWARD`，`holding` 被置 `1`、`vx_ref` 归零、`pitch_stick` 打出负向脉冲并激励固件 `pitchRatePID` 微分项，新动作到达后再清 `holding` 跳回 `+0.30 m/s`；`cf_sitl_proc` 实测仅 `10 ms` / `20 ms` / `30 ms` 空档即令 `0.8 s` 行内位移从 `0.2420 m` 降至 `0.2388 m` / `0.2358 m` / `0.2329 m`。对 `YAW → YAW`，漏出的 $\Delta t$ 内 `s.yaw` 归零触发固件航向锁定与反向制动。

#### 5. 转向动作的角度是按时长积分还是有目标角
- **结论**：**纯按有效时长对角速度积分，没有任何目标角闭环**。
- **代码位置**：
  - `sim/cf_sitl/exec/executor.c:138-141, 186, 194`：执行器仅将 `±0.6 rad/s`（`±34.38°/s`）除以固件满杆比例 `300°/s` 输出恒定杆量 `s.yaw = ∓0.11459`（刚过固件死区 `0.1`），无航向角状态或目标角。
  - `CF-Drone/control.ino:152, 161-166, 195-196, 208` 与 `CF-Drone/pid.h:19, 36`：`|controlYaw| >= 0.1` 期间，固件每 `1 ms` 执行 `yawTarget = attitude.getYaw()`，使外环角度误差 `yawError` 恒为 `0`，仅靠前馈 `ratesExtra.z = ±0.6 rad/s` 送入内环 `yawRatePID`（构造时未传 `windup` 默认 `0`，I 项恒为 `0`，纯 PD 跟踪陀螺角速率）。动作到期 `controlYaw` 归零首拍，`yawTarget` 停留在松杆那一拍的 `attitude.getYaw()`，外环 `yawPID`（`P=3`）将松杆后的惯性过冲拉回松杆时刻航向。
  - 数值后果（`probe_w113_results.json`）：单步 `YAW` 含起转加速滞后（约 `0.22 s` 到 90%），`next-dispatch`（`900 ms`）下第 1 步周期转角为 **`28.24°`**（`800 ms` 行内 **`24.81°`**），而连续同向第 2、3 步无起转滞后，周期转角升至 **`30.87°`**（`800 ms` 行内 **`27.44°`**）。

---

### B. 对草稿笔记 `$D/tmp/phase12-cards/w1-13/handoff-from-39b5006b.md` 的逐条核对

#### 核实成立的条目
1. **§1 离线核对结论与统计逻辑**（`:20-39`）：第二轮 6 个学生臂 4,972 个相邻观测对 `obs_gap_s` 全部为 `0.900 s`，W1-5 误起窗口内外巡航步位移相对差 `+0.55%`（`p=0.794`），转角差异来自 2 个碰撞截断步（剔除后 `-0.16%`，`p=0.522`），不重评两臂的依据成立（已由 `w1-13.md` §5 独立复跑逐字节对齐）。
2. **§2 评审 1（`NOMINAL_STEP_PERIOD_S` 写死 `0.9` 的缺陷）**（`:44-52`）：核实成立。`hold != 1.0`（如 `0.5` 或 `2.0`）时名义间隔为 `hold - 0.1 s`，写死 `0.9` 会漏报或每步误报，已在提交 `4d8a4ff`（合入 `e40c693`）修正。
3. **§2 评审 2（`safe_gap_s` 仅对 `cf_sitl` 具物理刹停含义）**（`:54-57`）：核实成立。`cf_sitl` 的 `until_from_window_start = True`（`cf_sitl_exec.py:333`），其余执行器或 `x3_velocity` 的截止机制不同。
4. **§3.1 时间代数更正**（`:68-73`）：核实成立。第一轮审核 A 建议的 `dispatch + (hold - L) = o_k + L + hold - L = o_k + hold` 确实恒等于现行 `deadline` 截止，没有改变任何行为；要覆盖完整步周期必须按名义步周期 `P = 0.9 s` 从派发戳起算。
5. **§3.3 行内量只覆盖 `0.8 s` 的口径提醒**（`:80-83`）：核实成立，且已由本次 `cf_sitl_proc` 实测定量验证：`next-dispatch` 无缝衔接时，行内 `[d_k, W_k]`（`0.800 s`）巡航位移为 `0.2420 m`，相邻观测 `0.900 s` 周期位移为 `0.2721 m`，若不区分两个口径会误判偏离旧协议（`0.2598 m`）超过 5%。用插件侧确认时刻核验物理衔接的建议也正确。
6. **§3.4 末步 `zero_now` 截断与首步 `safe_gap_s = null`**（`:86-91`）：核实成立。主循环退出后 `finally` 调 `stop_executor` → `sil.zero_now()` 发送 `action(0, now + 3600.0)`（`cf_sitl_exec.py:67, 531`；注：后任报告 `w1-13.md:135` 误写为 `now + 0.5`，交接笔记原文 `now + ZERO_UNTIL_S` 是准确的），在窗口末截断最后 `0.1 s` 动作，不影响已采样的行内位移。
7. **§3.5 协议一致性键同步清单**（`:93-98`）：`eval_closed_loop.PROTOCOL_KEYS`、`preprocess.STRICT_KEYS`、`stats_report.py` 一致性键同步加 `action_until`，以及默认 `deadline` 不写新键以保持逐字节不变，均成立。

#### 不成立或有遗漏的条目
1. **§3.2 `next-dispatch` 最小实现「`until = exec_ts + P` 即可让动作首尾相接、步间不回 SAFE」**（`:75-78`）：**物理上不成立（见发现 `K-1`）**。笔记把 Python 读到的已到达位姿戳 `exec_ts` 当成了动作到达插件的时刻。实际上 `exec_ts[k+1] = o_{k+1} + L = exec_ts[k] + P` 的位姿帧先在 Gazebo 生成，经跨进程管道传到 Python 后才触发 `dispatch`，再经跨进程管道传回插件 `DrainActions`，到达插件时 `t_sitl` 必定大于 `until_sitl = exec_ts[k] + P - base_s_`。不加管道保持余量 `guard_s` 时，每步之间仍会掉回 `CF_EXEC_SAFE` 达 $\Delta_{\text{pipe}}$ 秒，而账面 `safe_gap_s` 因同用 Python 侧 `exec_ts` 减法恒得 `0` 掩盖了该空档。
2. **§2 评审 3（`ack_sim_t` 匹配唯一、缓冲 600 帧足够、分辨率无偏）**（`:59-62`）：**四处边界不成立（见发现 `K-4`）**。
   - 笔记写「相邻步截止至少差 0.8 s，匹配唯一」：只对 `deadline` 成立；按该笔记 §3.4 自己的迟到步规则（`until = 实际派发戳 + P`），若第 $k$ 步迟到 $\delta \approx P$，则 `exec_ts[k] == exec_ts[k+1]`，两步 `until` 完全相等，无游标的 `ack_sim_t()` 从旧到新扫缓冲会把第 $k+1$ 步错配到第 $k$ 步的帧。
   - 笔记未考虑迟到步贴近窗口末（如 `exec_ts` 距 `hold_end_ns` 仅 `10 ms`）时，`/e1_mini/state` 与 `/model/e1_mini/pose` 两个独立 `gz topic -e` 子进程的管道竞态会让窗口末立即调用的 `ack_sim_t()` 漏读返回 `None`。
   - `600` 帧（`3.0 s`）在 `--hold >= 3.0` 时窗口末回扫必已被挤出缓冲。
   - `state` 每 `5` 个物理步发一帧（`CfSitlSystem.cc:244`），`ack_sim_t` 相对真实生效步有 `[0, +4 ms]` 的单向上偏（非对称舍入）。
3. **§3.3 与任务卡转角预期核对遗漏「单步 YAW 与连续 YAW 在 `next-dispatch` 下分化」**（`:80-83`）：笔记只讨论了前飞巡航速度提高，未指出转向在消除 `100 ms` 步间刹停后，连续第 2 步起不再经历 `0.22 s` 起转加速，周期转角从单步 `28.24°` 跃升至 `30.87°`（见发现 `K-5`）。
4. **§3.5 Part 2 透传只提 `--rtf` 与 `--action-latency`，遗漏 `--cf-sitl-params`**（`:97`）：`collect_parallel.py:204` 同样已有 `getattr(args, "cf_sitl_params", None)`，但 `argparse` 与 `collect_command` 均缺失（见发现 `K-2`）。

---

### C. 对第一轮审核 A 发现 1（「位移弹性来自刹停段」）的裁决

- **裁决结论**：**推断成立，已由执行器源码与 `cf_sitl_proc`（`lib31`）反事实数值实验定量证实；但需将总位移差精确分解为「行内测量窗口截断」与「SAFE 主动刹停加姿态-电机滞后凹坑」两部分**。
- **源码与数值证据**（`probe_w113.py` 跑 8 个 episode × 第 5–11 步稳态巡航均值，见 `probe_w113_results.json`）：

| 协议设定（步周期恒为 `900 ms`） | 动作有效时长 | 步间 SAFE 空档 | 行内测量窗口 | 行内位移 `win_disp_m` | 整周期真位移 `period_0.9s_disp_m` | 窗口末真速度 |
| --- | --- | --- | --- | --- | --- | --- |
| 旧协议等效（`lag=0.12 s`） | `880 ms` | `20 ms` | `880 ms` | `0.2598 m` | `0.2658 m` | `0.300 m/s` |
| 第一轮 `L=0.16` `deadline` | `840 ms` | `60 ms` | `840 ms` | `0.2357 m` | `0.2535 m` | `0.296 m/s` |
| 第二轮 `L=0.20` `deadline` | `800 ms` | `100 ms` | `800 ms` | `0.2118 m` | `0.2410 m` | `0.290 m/s` |
| 反事实对照：`next-dispatch` 无空档 | `900 ms` | `0 ms` | `800 ms` | `0.2420 m` | `0.2721 m` | `0.301 m/s` |

- **定量拆解**：
  1. 若无主动刹停（纯定速 `0.301 m/s`），窗口从 `880 ms` 缩至 `800 ms`（`-9.09%`）或从 `840 ms` 缩至 `800 ms`（`-4.76%`），位移弹性恒为 `1.00`。
  2. 实测从 `L=0.16` 到 `L=0.20`，窗口缩短 `4.76%` 而行内位移从 `0.2357 m` 跌至 `0.2118 m`（`-10.14%`，弹性 `2.13`，与 Gazebo 闭环实测一致）；从旧协议 `0.2598 m` 到 `L=0.20` `0.2118 m` 共减少 `0.0480 m`（`-18.5%`），其中窗口从 `880 ms` 截短为 `800 ms` 占 `0.0178 m`（对比无空档 `0.2420 m`，占 `37.1%`），而 `100 ms` `SAFE` 主动刹停使同为 `800 ms` 的窗口位移从 `0.2420 m` 净跌至 `0.2118 m`，占 **`0.0302 m`（`62.9%`）**；在完全剔除窗口长度差异的 `900 ms` 整周期位移上，`100 ms` `SAFE` 刹停仍造成 `0.2721 m → 0.2410 m`（`-11.4%`）的纯物理减速损失。
  3. 微观修正：审核 A 原文推测机体在 `0.1 s` 空档内已降至低速再重新加速；实测轨迹（`cycle_trace`）表明，因固件姿态环与电机 `tau_down=0.12 s` 滞后，`0.1 s` 空档结束时速度仍为 `0.305 m/s` 但俯仰角已翻负至 `-1.04°` 且具负角速率，真正的速度凹坑（最低 `0.243 m/s`）发生在新一步派发后的前 `400 ms` 内，直接吃掉了新一步窗口内的积分位移。

---

## 2. 发现清单（按严重程度排序）

### K-1【高，读代码确认 + `cf_sitl_proc` 已复现】卡 W1-13 第三部分拟议的 `until = exec_ts + P` 忽略跨进程管道延迟，实跑仍会每步掉回 SAFE 刹停，且账面 `safe_gap_s` 会虚假显示为 0

- **位置**：
  - `sim/tools/collect.py:1072-1096, 1298`（合并提交 `e40c693`）：等待已到达位姿戳 `>= pose_before_ts + L` 后取 `exec_ts` 派发，并按 `safe_gap_s = (exec_ts - prev_until_ns) / 1e9` 记账。
  - `sim/tools/cf_sitl_exec.py:494-507`：`until = t / 1e9 + hold_s`（绝对 Gazebo 仿真秒）。
  - `sim/cf_sitl/gz/CfSitlSystem.cc:138-147` 与 `sim/cf_sitl/exec/executor.c:182-195`：`in->t_s < e->action_until_s` 严格小于判定，到点当拍切回 `CF_EXEC_SAFE`。
  - 在途设计稿：`$D/tmp/phase12-cards/w1-13/handoff-from-39b5006b.md:75-78`、`$D/tmp/phase12-cards/reports/w1-13.md:129`。
- **现象**：两份 W1-13 设计稿均规定 `next-dispatch` 下发给执行器的截止为 `until_k = exec_ts[k] + P`（`P = 0.900 s`）。在第 $k+1$ 步，Gazebo 必须先推进到仿真时刻 $t_{\text{sim}} = o_{k+1} + L = \text{exec\_ts}[k] + P$ 才能由 `OdometryPublisher` 发出该帧位姿；在发出该帧的同一个 `1 ms` 物理步里，插件内 `t_sitl == until_sitl[k]`，第 $k$ 步动作已经过期切回 `CF_EXEC_SAFE`。此后位姿帧经 `gz topic -e --json-output` 子进程送达 Python、唤醒 `collect.py` 主线程、再经 `--pub-server` 子进程将新动作送回插件 `DrainActions`，中间经历跨进程往返延迟 $\Delta_{\text{pipe}} > 0$（Gazebo 未暂停，物理时钟继续走），第 $k+1$ 步动作在 $t_{\text{ack}, k+1} = \text{exec\_ts}[k] + P + \Delta_{\text{pipe}}$ 才生效。同时现行 `deadline` 模式下第 $k$ 步在固件内的真实有效时长也是 `hold - L - Δ_pipe`，真实刹停空档为 `0.100 s + Δ_pipe`。
- **为什么是缺陷及影响**：
  1. 在 `[until_k, t_ack_{k+1}]` 这 $\Delta_{\text{pipe}}$ 秒内，执行器每步都掉回 `CF_EXEC_SAFE`（`holding=1`、`vx_ref=0`、俯仰杆负跳变并冲击固件 `pitchRatePID` 微分项），违背卡 W1-13 第 7 条「动作持续到下一次派发，步间不回 SAFE」的设计目标，且步位移随机器负载改变 $\Delta_{\text{pipe}}$ 而波动（`probe_w113_results.json` 实测 `5 / 10 / 20 / 30 ms` 空档使行内巡航位移从 `0.2420 m` 依次掉到 `0.2402 / 0.2388 / 0.2358 / 0.2329 m`）。
  2. Python 侧账面 `safe_gap_s = exec_ts[k+1] - (exec_ts[k] + P)` 两端都用滞后的 Python 侧位姿戳，恒算出 `0.000 s`（完美通过「中位数为 0、最大 ≤ 0.02 s」的标定门），形成无法从 `safe_gap_s` 发现的假阳性。
- **怎样复现或验证**：
  - 数值复现：`nice -n 15 .venv-tools/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-k/probe_w113.py`（见 `L0.20_next_unpadded_gap*` 各行）。
  - Gazebo 验证：待排队中的 `exp-w1-13-ack-20261112`（PGID `78842`）跑完，读取 `summary.md` 里「确认 − Python 侧派发」一栏：只要该值 `> 0 ms`，不加余量的 `exec_ts + P` 就必然产生同量级的真实 `SAFE` 空档。
- **建议的最小修法**：
  1. `next-dispatch` 下发给插件的截止改为 `until_plugin = exec_ts + P + guard_s`，其中 `guard_s` 取大于满载 `plugin_ack_sim_t - (pose_ts + L)` 的 `p99` 加一个位姿周期（例如 `0.08 s`，小于停顿判据余量 `STALL_MARGIN_S = 0.20 s`）。正常步下一动作在到期前到达插件直接覆盖旧动作（`0` 拍回 `SAFE`，真实作用时长 $t_{\text{ack}, k+1} - t_{\text{ack}, k} \approx P$，一阶抵消固定管道延迟）；停顿步或异常时最多多走 `guard_s` 即安全回 `SAFE`；集末在窗口末由 `zero_now()` 截断。
  2. 记账分清两项：Python 派发周期抖动 `dispatch_period_err_s = exec_ts[k+1] - (exec_ts[k] + P)`，以及插件侧真实空档 `plugin_safe_gap_s = plugin_ack_sim_t[k+1] - until_plugin[k]`（无缝覆盖时应 `< 0`）。
- **状态**：时序逻辑读代码确认；微空档对位移的定量影响已用 `cf_sitl_proc` 复现。
- **归属建议**：并入**卡 W1-13 第三部分**（开工前修正规格）。

---

### K-2【高，读代码确认】`collect_parallel.py` 与 `collect.py` 存在 `--cf-sitl-params` 单边生效缺口：并行采集不支持满电档，单调 `collect.py` 时插件物理参数与 `manifest` 指纹可静默背离

- **位置**：
  - `sim/tools/collect_parallel.py:204-205, 403-446, 560-760, 889-980`（合并提交 `41c474d`、`e40c693`）。
  - `sim/tools/collect.py:1634-1647, 1672-1685`。
  - 对照：`sim/cf_sitl/gz/CfSitlSystem.cc:45, 70` 与 `sim/cf_sitl/gz/README.md:57-62`。
- **现象**：
  1. 卡 W1-10（`41c474d`）在 `collect_parallel.scene_gen_extra`（`:204-205`）加了 `if getattr(args, "cf_sitl_params", None): extra += ["--cf-sitl-params", args.cf_sitl_params]`（供 `eval_closed_loop.py` 复用），但 `collect_parallel.py` 自身的 `argparse` 没有 `--cf-sitl-params` 选项，`collect_command`（`:403-446`）也不向子进程 `collect.py` 传 `--cf-sitl-params`，`collection_manifest.json` 也不记录该键。
  2. `collect.py` 在 `:1676` 已解析 `scene_meta_obj`，其中由 `scene_gen --cf-sitl-params` 生成的世界带有 `scene_meta_obj["cf_sitl_params"]`，但 `collect.py` 完全不比对它与 `args.cf_sitl_params` 是否一致。
- **为什么是缺陷及影响**：
  - 已有实验不受影响（`eval_closed_loop.py:973-974` 自己拼了 `collect.py --cf-sitl-params`，两边一致）。
  - 对后续实验是高危陷阱：计划 §10.4 的 E3（及卡 W2-6）明确要求「新采，满电档」采集多块长课程数据，现行 `collect_parallel.py` 根本传不进满电参数；更危险的是，后续执行者若照 `scene_gen_extra` 已有代码只在 `collect_parallel.py` 的 `argparse` 补上 `--cf-sitl-params` 而漏改 `collect_command`，Gazebo 插件会跑满电档（`06e5f017`），而每集 `collect.py` 会按默认 `e1_mini.json` 在 `episode_manifest.json` 写半电指纹（`cf_sitl_v1:1bc8fedf:lib31`），`preprocess.py` 无法察觉，满电轨迹会静默混入半电数据集。同理，任何脚本直接调 `collect.py` 只传一边参数也会产生假指纹数据。
- **怎样复现或验证**：
  - 运行 `.venv-tools/bin/python sim/tools/collect_parallel.py --cf-sitl-params /tmp/x.json` 立即报 `unrecognized arguments: --cf-sitl-params`。
  - 读 `collect_parallel.py:403-446` 确认 `collect_command` 无 `cf_sitl_params`；读 `collect.py:1634-1685` 确认无 `scene_meta_obj.get("cf_sitl_params")` 校验。
- **建议的最小修法**：
  1. 在 `collect.py` 加载 `scene_meta_obj` 后（约 `:1685`）加 4 行一致性门禁：当 `args.dynamics == "cf_sitl"` 且给了 `--scene-meta` 时，比较 `os.path.realpath(args.cf_sitl_params or cf_sitl_exec.DEFAULT_PARAMS)` 与 `os.path.realpath(scene_meta_obj.get("cf_sitl_params") or cf_sitl_exec.DEFAULT_PARAMS)`，不一致立即 `p.error` 退出（从根源封死两边不一致）。
  2. 在 `collect_parallel.py` 的 `argparse`、`collect_command` 与 `collection_manifest.json` 补齐 `--cf-sitl-params`（缺省 `None` 不传、不写键，保持默认输出逐字节不变）。
- **状态**：读代码确认。
- **归属建议**：并入**卡 W1-13 第二轮**或 **E3 / W2-6 开工前的前置补丁**。

---

### K-3【高，`cf_sitl_proc` 已复现】卡 W1-10 满电档（`battery_soc_init=1.0`）未联动调整悬停前馈 `hover_duty_estimate=0.824`，导致 40/40 集起飞全部冲破 ToF 与光流 2.50 m 量程上限

- **位置**：
  - `sim/dynamics/e1_mini.json:37`（`hover_duty_estimate: 0.824`）、`:48`（`tof_range_max: 2.5`）、`:55`（`flow_max_height: 2.5`）、`:92`（`tof_timeout_s: 0.5`）、`:117`（`battery_soc_init`）。
  - `sim/cf_sitl/src/plant.h:94`；`sim/cf_sitl/exec/executor.c:24, 54, 69, 152`；`sim/cf_sitl/src/sensors.h:46, 61`。
  - 卡 W1-10（合并提交 `41c474d`，`docs/experiments/2026-11-11-soc100-parity.md:32-40`）。
- **现象**：`hover_duty_estimate = 0.824`（对应初始 `hover_stick = 0.910`）是按 50% SoC（带载 `3.44 V`）算出的；满电 `battery_soc_init = 1.0`（带载 `3.86 V`）真实悬停占空比只需 `0.734`（对应 `hover_stick = 0.799`，见 `e1_mini.json:143-144`）。卡 W1-10 的满电副本只将 `sim.battery_soc_init` 改为 `1.0`，未改 `hover_duty_estimate`。用 `cf_sitl_proc`（`lib31`）实测 seed 0 下 episode `0..39` 共 40 集起飞（`probe_w113_results.json` 的 `soc100`）：
  - **50% 电量档**：40 集起飞峰值高度 `2.204–2.224 m`（均值 `2.214 m`），0 集越过 `2.50 m`，`READY` 平均用时 `7.27 s`。
  - **100% 满电档**：**40/40 集（100%）起飞峰值高度全部越过 `2.50 m`（范围 `2.5025–2.5158 m`，均值 `2.5092 m`）**，40/40 集在起飞顶部同时触发 ToF 超量程（`tof_m = -1`）与光流超高度失效（`flow_valid = false`），单次连续失明时长达 **`300 ms`**，距 `tof_timeout_s = 500 ms` 触发 `CF_EXEC_FAILSAFE` 仅剩 **`200 ms`**（约 `1.5 cm` 过冲余量），`READY` 平均用时拉长至 `9.44 s`（最大 `11.07 s`）。
- **为什么是缺陷及影响**：
  - 正在排队运行的 `exp-soc100-parity-20261111`（PGID `508021`）在 Gazebo 外部刚体模式下若因积分微小差异使某集过冲多持续 `200 ms`，就会触发 `takeoff_failsafes` 重试（一旦满 4 次则整集失败）。
  - 后续若在满电档上叠加任何质量偏轻（如 `< 98 g`）、推力偏高或 ToF 量程随机化（第一轮 `sim-real.md` D4 计划），起飞会大面积坠入 `CF_EXEC_FAILSAFE` 开环爬升。
- **怎样复现或验证**：运行 `probe_w113.py` 的 `exp3_soc100_takeoff_overshoot()`，或查看 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-k/probe_w113_results.json` 的 `soc100` 字段。同时在 `exp-soc100-parity-20261111` 跑完后核查各集 `episode_manifest.json` 的 `takeoff_failsafe_retries` 是否大于 0。
- **建议的最小修法**：
  - 当前 W1-10 对拍实验不改（已冻结预注册指纹 `06e5f017`），但在分析报告中必查 `takeoff_failsafe_retries`。
  - 若 E3 正式大规模重采满电数据且暂不升级 C++ 库，可用 `make_params_variant.py --set sim.battery_soc_init=1.0 --set hover_duty_estimate=0.734` 生成匹配前馈的满电副本（只换参数 sha，不换 `lib31`）；升级 `lib32` 时按 `sim-real.md` E2 在 `plant.h` 按初始 `soc0` 自动解算初始悬停占空比。
- **状态**：已在 `cf_sitl_proc`（`lib31`）40 集上复现。
- **归属建议**：W1-10 报告核查重试计数；参数或代码修正并入 **E3 满电采集卡**或 **`cf_sitl` lib32 升级卡**。

---

### K-4【中，读代码确认】刚刚合入 `main` 的 `CfSitlExecutor.ack_sim_t()`（提交 `e40c693`）存在四处边界缺陷：双管道竞态假阴性、`next-dispatch` 迟到步错配、长 `hold` 缓冲溢出与 `+0..4 ms` 阶梯上偏

- **位置**：
  - `sim/tools/cf_sitl_exec.py:77-83, 459, 508-526`（提交 `579901d`，合并提交 `e40c693`）。
  - `sim/tools/collect.py:1321-1325`（合并提交 `e40c693`）。
  - `experiments/2026-11-12-w1-13-newprotocol/summarize_ack.py:77-85, 152-155, 267-271`（提交 `f4a2315`，合并提交 `e40c693`）。
- **现象与四处边界缺陷**：
  1. **窗口末跨进程管道异步竞态导致假阴性 `None`**：`collect_one` 在 `subs["pose"].latest() >= hold_end_ns` 退出循环后立即调 `sil.ack_sim_t()`。`/model/e1_mini/pose` 与 `/e1_mini/state` 由两个独立的 `gz topic -e --json-output` 子进程通过各自管道喂给 Python（`gz_io.py:76-80`）。当某步发生大迟到（例如 `exec_ts` 落在 `hold_end_ns - 10 ms`）或满载 CPU 抢占导致 `state` 子进程管道比 `pose` 管道晚冲洗几十毫秒墙钟时，动作虽已在插件生效，但窗口退出瞬间 `self._sub.frames()` 尚未收到该帧 `state`，`ack_sim_t()` 直接返回 `None`。
  2. **无已确认游标，在 `next-dispatch` 迟到步下会错配到上一步**：`ack_sim_t()` 每次从 `self._sub.frames()` 最老帧向最新帧扫描 `abs(until_sim - self._last_until) <= 1e-3`。在 `next-dispatch` 下截止为 `exec_ts + P`；若第 $k$ 步迟到 $\delta \approx P$（例如 A* 卡顿 `0.9 s`），第 $k$ 步与第 $k+1$ 步读到同一个最新位姿戳 `exec_ts[k] == exec_ts[k+1]`，发出完全相同的 `_last_until`，第 $k+1$ 步调用 `ack_sim_t()` 会命中 3 秒缓冲里第 $k$ 步的首帧，算出比第 $k+1$ 步派发还早或错误的确认时刻（触发 `summarize_ack.py:155` 的「早于派发戳」报错或污染延迟分布）。
  3. **`STATE_BUF = 600`（`3.0 s`）在 `--hold >= 3.0` 时必溢出**：`collect.py` 允许任意正数 `--hold`；当 `hold - L >= 3.0 s` 时，窗口末回扫时窗口初的生效帧已被后续 `200 Hz` 帧挤出环形缓冲，`ack_sim_t()` 恒为 `None`。
  4. **`200 Hz` 发布相对 `1 kHz` 物理步存在 `[0, +4 ms]` 单向上偏**：插件在 `CfSitlSystem.cc:244` 每 `state_every_ = 5` 步才调 `PublishState`。若动作在 `steps_ % 5 == 1` 的 `PreUpdate` 生效，首次出现在 `/e1_mini/state` 的时间戳是 `+4 ms` 后的那一拍（均匀分布期望 `+2 ms`）。`summarize_ack.py:31` 用 `p99 < 10 ms` 判定时，其中已有最高 `4 ms` 来自采样栅格而非传输延迟。
- **为什么是缺陷及影响**：`summarize_ack.py:270` 已预写建议判据 `plugin_ack_sim_t is None or plugin_ack_sim_t - (pose_ts/1e9 + L) > 阈值`；若后续按此切换迟到判据，上述竞态 `None` 与错配会直接误杀正常集或漏判迟到集。
- **怎样复现或验证**：构造 `FakeSub` 单测：(a) 连续两步传入相同 `until`、不同生效帧序号，断言第二步 `ack_sim_t()` 返回第二次生效帧而非第一次（现行代码会失败，返回第一次）；(b) `hold_s = 3.5` 时填充 700 帧后调 `ack_sim_t()`，现行代码返回 `None`。
- **建议的最小修法**：
  1. `CfSitlExecutor` 记录 `_ack_floor_seq = 0`：每次 `dispatch()` 把当前已处理位点推进（或在 `ack_sim_t()` 命中帧 `(seq, msg)` 时条件要求 `msg_sim_t >= dispatch_pose_sim_t - 0.005` 且更新 `_ack_floor_seq = seq`；注意连续两步同 `until` 时新动作不会改变 state 里的 `IDX_UNTIL`，所以 `next-dispatch` 下还应给重复 `until` 加 `2 ms` 微偏或避免同 `exec_ts` 派发两步）。
  2. `ack_sim_t()` 初次未扫到且 `_sub.healthy()` 时，若 `_sub` 最新帧时间戳小于 `hold_end_ns`，用 `_sub.latest_after(hold_end_ns - 5_000_000, timeout=0.05)` 对齐双管道后再扫一次。
  3. `buf = max(STATE_BUF, int(math.ceil(self.hold_s * 200)) + 400)`。
- **状态**：读代码确认。
- **归属建议**：并入**卡 W1-13 第三部分（第二轮）**。

---

### K-5【中，`cf_sitl_proc` 已复现】`next-dispatch` 消去步间刹停后，连续转向步比单步转向多转 `2.63°`（`30.87°` 对 `28.24°`），偏离任务卡预期 `26°` 达 `+8.6% ~ +18.7%`

- **位置**：
  - `sim/cf_sitl/exec/executor.c:138-141, 186, 194`；`CF-Drone/control.ino:161-166`。
  - 任务卡 W1-13 第 8 条（`docs/briefs/2026-11-10-phase12-task-cards.md:609`）；`sim/tools/rule_policy.py:22, 32`。
- **现象**：任务卡第 8 条写明预期 `next-dispatch` 下「转角回到旧协议附近（26° 上下）；偏离旧协议 5% 以上要写明原因」。`probe_w113.py` 实测（`probe_w113_results.json` 的 `yaw`）：
  - 现行 `deadline`（`800 ms YAW + 100 ms SAFE`）：每步末 `100 ms` 回 `SAFE` 把角速度刹停，因此第 1 步与第 2 步的 `800 ms` 行内转角相近（`24.81°` 与 `24.28°`，整周期 `27.25°` 与 `26.72°`）。
  - `next-dispatch` 无缝衔接（`900 ms YAW`）：第 1 步（从静止起转）行内 `800 ms` 转 **`24.81°`**、整周期 `900 ms` 转 **`28.24°`**；连续同向第 2、3 步因不松杆、省去 `0.22 s` 角速度上升瞬态，全程以稳态 `0.6 rad/s = 34.38°/s` 旋转，行内 `800 ms` 升至 **`27.44°`**，整周期 `900 ms` 升至 **`30.87°`**（相对 `26°` 分别高 `+8.6%` 与 `+18.7%`）。
- **为什么是缺陷及影响**：
  1. 标定若只看行内 `dyaw_deg`（`0.8 s` 切片），会低估真实每步转角 `3.43°`；若看整周期真实转角，必然超过 `26° ± 5%`（`27.3°`），需提前明确这是 `0.9 s × 34.38°/s` 消去刹停后的正常物理结果。
  2. 连转 2 步的实际航向变化从现行协议的约 `50.4°` 增至 `28.24° + 30.87° = 59.1°`，直接改变按步数计的航向码（`nav_cond.py`）与航向护栏（`--heading-guard 2`）对应的物理角度。
- **怎样复现或验证**：运行 `probe_w113.py` 的 `exp2b_yaw_protocols()`。
- **建议的最小修法**：W1-13 第三部分标定汇总同时报「行内 `0.8 s` `|dyaw_deg|`」与「相邻观测 `0.9 s` `|Δyaw_before|`」，并分列「首步转向（前一步非同向 YAW）」与「连续转向（前一步同向 YAW）」；若主线切到 `next-dispatch`，更新 `rule_policy.py` 的 `yaw_step_deg`，并在 E2 中优先采用按固件估计偏航角分档的条件码（计划 §10.7 第 6 条）。
- **状态**：已在 `cf_sitl_proc`（`lib31`）上复现。
- **归属建议**：并入**卡 W1-13 第三部分（标定汇总）**与**卡 E2**。

---

### K-6【中，`cf_sitl_proc` 已复现】SITL 在 `reset` 后 `t=0` 立即发解锁手势，因固件 `battery.ino` 首 `0.5 s` 未采样（`batteryVoltage=0.0 V` 视作未接电池）静默绕过 L1（`3.5 V`）低电禁解锁保护

- **位置**：
  - `CF-Drone/battery.ino:18-25`；`CF-Drone/control.ino:131`；`sim/cf_sitl/fw/fw_sitl.cpp:60, 117-121`；`sim/cf_sitl/exec/executor.c:157-160`。
  - `sim/cf_sitl/selftest/scenario_battery.cpp:78`（注释误以为设起始 SoC 再解锁即可测 L1）。
- **现象**：`fw_reset()` 将固件全局 `batteryVoltage` 复位为 `0.0f`、`fwl_lastCheck` 复位为 `0`，随后调用的固件 `setup()` 不执行 `updateBatteryVoltage()`。当 `reset` 后立即调 `cf_sitl_arm_takeoff`，第 0 步（`t = 0.000 s`）执行器发出满偏航解锁手势；固件 `loop()` 中 `updateBatteryVoltage()` 因 `t - lastCheck < 0.5f` 直接返回，`batteryVoltage` 保持 `0.0f`。紧接着 `control.ino:131` 判断 `batteryVoltage > VBAT_ABSENT_THRESHOLD (0.5f) && batteryVoltage < VBAT_WARN_THRESHOLD (3.5f)` 为假（固件把 `0.0 V` 当成未接电池的 USB 调试态），直接置 `armed = true`。等到 `t = 0.500 s` 首次采样出真实低电压时，推力已跨过 `BATTERY_FLYING_THRUST_MIN (0.15f)` 进入飞行态，L1 保护永久失效。
- **为什么是缺陷及影响**：当前默认 `50%` 与 `100%` 电量均高于 `3.5 V`，现有实验不受影响；但真机上电初始化与飞手解锁必晚于 `0.5 s`，低电（`< 3.5 V`）真机拒绝解锁而 SITL 照常起飞。这既让 `scenario_battery.cpp:78` 设想的 L1 测试无法生效，也会在后续做初始电量随机化时放行本该禁飞的低电集。
- **怎样复现或验证**：运行 `probe_w113.py` 的 `exp4_l1_battery_bypass()`（`probe_w113_results.json` 的 `l1_bypass`）：设 `battery_soc_init = 0.0`（`V_oc = 3.30 V < 3.50 V`），`reset` 后立即 `takeoff` 得到 `armed = 1`（`battery_v = 3.037 V`）；若 `reset` 后先 `step 600`（静置 `0.6 s`）再 `takeoff`，则 `armed = 0`（被固件 L1 正确拦截）。
- **建议的最小修法**：下次升级 `cf_sitl` 库版本（`lib32`）时，在 `fw_sitl.cpp:120` 的 `setup()` 之后加一行 `batteryVoltage = readBatteryVoltage();`（消除上电首拍 `0.0 V` 假象；注意正常 `SoC >= 0.3` 时行为不变，但严守版本号纪律随 `lib32` 一起做）。
- **状态**：已在 `cf_sitl_proc`（`lib31`）上复现。
- **归属建议**：并入 **`cf_sitl` lib32 升级卡**（与第一轮 `sim-real.md` E1–E6 同批处理）。

---

### K-7【低，读代码确认】执行器 `executor.c` 不感知固件 `mode == AUTO`（L3 低电或失联自动降落），持续变动的姿态杆量触发固件 `autoFailsafe()`，导致 `descend()` 每 `1 ms` 重置全部 6 个 PID

- **位置**：
  - `sim/cf_sitl/exec/executor.h:39`；`sim/cf_sitl/exec/executor.c:180-196`；`sim/cf_sitl/src/cf_sitl.cpp:49-59`。
  - `CF-Drone/safety.ino:39-46, 61-101, 163-178`；`CF-Drone/control.ino:114`。
- **现象**：当电池放电至带载 `< 3.0 V` 持续 `0.9 s` 触发固件 L3 自动降落（`l3Latched = true`）时，`batteryFailsafe()` 调用 `descend()` 将 `mode` 设为 `AUTO(4)` 并重置 6 个角速率/角度 PID。然而 `executor.c` 的输入结构体 `cf_exec_input` 没有 `fw_mode` 字段，执行器在 `READY/ACTING` 下继续每 `1 ms` 按光流噪声微调 `s.roll` 与 `s.pitch`。下一拍固件 `failsafe()` 先运行 `autoFailsafe()`（`safety.ino:93`：`roll != controlRoll || pitch != controlPitch`），误判为「飞手打杆接管」而把 `mode` 改回 `STAB(2)`；紧接着同一拍的 `batteryFailsafe()` 因 `l3Latched` 再次调用 `descend()`，看到 `mode != AUTO` 又走一遍「首次进入」分支，导致降落全程**每 `1 ms` 把全部 6 个 PID 的积分与微分低通清零一次**。若是非锁存的 `rcLossFailsafe`（将来真机移植方案下），执行器的杆量刷新更会直接打断 `AUTO` 降落。
- **为什么是缺陷及影响**：现有仿真集在电量耗尽前结束，不影响已有评测结果；但 `sim/cf_sitl/README.md:88-92` 规定 `executor.c` 目标是原样移植进 ESP32 固件，移植到真机后一旦触发低电自动降落，姿态内环 PID 每毫秒被清零会破坏降落稳定性。
- **怎样复现或验证**：读 `safety.ino:39-101, 163-178` 与 `executor.c:180-196` 即可确认控制流闭环；在 `selftest` 场景 5 的 `t > 278.9 s` 段打印 `rollRatePID.integral` 可见每拍恒为 `0`。
- **建议的最小修法**：真机移植或升 `lib32` 时，在 `cf_exec_input` 传入 `fw_mode`；当 `in->fw_mode == 4 /* AUTO */` 时，执行器停止闭环改写杆量（输出上拍保持值让固件 `descend()` 独占控制，避免触发 `autoFailsafe`）。
- **状态**：读代码确认。
- **归属建议**：列入**真机固件移植与 `cf_sitl` lib32 清单**（与 `sim-real.md` E1 合并）。

---

### K-8【低，读代码确认】计划 §0 第 10 条悬案「连续飞 180 s 后前进速度偏离 `0.3 m/s`」的源码根因结案，界定长课程单集时长包络

- **位置**：`sim/cf_sitl/exec/executor.c:71, 114-116`；`CF-Drone/estimate.ino:42-64`；`CF-Drone/control.ino:48, 229`；`docs/PLAN_2026-11-10_phase12.md:68`。
- **现象与根因**：计划 §0 第 10 条记有「连续飞 180 s 之后前进速度偏离 0.3 m/s，原因没查清」。经核查 `executor.c`、`estimate.ino` 与 `plant.h`，该现象由三项确定性机制共同决定：
  1. **姿态估计零偏累积触顶速度环积分限幅（主因）**：固件飞行中 `applyAcc()` 因 `!landed` 不运行、`levelWeight = levelBiasGain = 0`，姿态估计纯靠陀螺积分，倾角误差以残余零偏 $b$（`1σ = 0.097°/s`）加随机游走（`1e-4 rad/s/√s`）增长。`180 s` 时典型零偏累积达 `15°–30°`；一旦超过 `tilt_int_max_deg = 20°` 减去配平角 `0.7°`（约 `19.3°`，`1.1σ` 种子在 `180 s` 即触顶），`vel_int` 饱和冻结，速度环退化为纯 P，前向稳态速度按 $\Delta v_x = -(|\delta| - 20^\circ)/\text{vel\_kp}$ 跌落甚至倒飞。
  2. **光流去旋转零偏耦合与斜坡跟踪静差（次因，全程存在）**：`executor.c:71` 用含零偏的固件 `rates` 补偿光流旋转（`flow_vel[0] = (flow[0] + gyro[1]) * h`），加上 PI 环追斜坡零偏的稳态误差，真值前向速度恒带常值偏移 $-b_y h - b_y / \text{vel\_ki}$（随机游走使 $b_y(t)$ 在 `180 s` 末进一步漂移）。
  3. **电池放电在 `170–205 s` 触及固件 `motThrMax = 0.90` 推力上限**：满电起飞约 `180–205 s` 后悬停所需占空比逼近 `0.90`，机动倾角余量被 `desaturate` 压缩并伴随掉高，高度变化同步改变光流换算速度 `flow * height_est_m`。
- **影响与建议**：现有 `K=1`（约 `35 s`）与 `K=3`（半电约 `90 s`）绝大多数种子在 `20°` 积分预算内；但规划 `K > 3`（如 `blocks=4..6`，`> 120 s`）的长课程时，仅换满电档（W1-10）只能解决第 3 项电池掉高，解决不了第 1 项陀螺零偏在 `120–180 s` 吃满 `20°` 积分限幅的问题。跑超 `120 s` 的长集前，需在固件估计器启用小增益飞行中重力修正（如 `EST_LVL_BIAS_GAIN`）并升库版本。
- **状态**：读代码确认（与 `selftest/scenario_longrun.cpp` 实测包络一致）。
- **归属建议**：写入 **E3 / 卡 W2-6（多块长课程）** 的步数上限约束说明。

---

## 3. 第一轮清单中本角度相关条目的核对

对照 `docs/reviews/2026-11-12-cross-review/FIXLIST.md`、`review-g.md`、`review-a.md` 及 `docs/reviews/2026-11-10/sim-real.md`：

### 3.1 已修且核对正确
- **A1（步间空档记账部分）**：卡 W1-13 提交 `e9822de`、`5460af6`、`7353e10`（合入 `e40c693`）已在 `collect.py` 每步记 `safe_gap_s`、`eval_closed_loop.py` 汇总报中位数与最大值，并用文件级测试锁住默认输出除新增行键外逐字节不变。
- **A3（步间停顿记账与失败集处理）**：卡 W1-13 提交 `51ba557` 与修正提交 `4d8a4ff`（合入 `e40c693`）已记 `obs_gap_s` 与 `stall_steps`，修正了 `hold != 1.0` 时名义周期硬编码的问题，带 `--action-latency` 时含停顿步的集进 `failed[]`。
- **A2（协议偏移记录、并行器与流水线透传、打包过滤）**：`eval_closed_loop.py` 的 `protocol_notes`（`46cd88c`）、`collect_parallel.py` 透传 `--action-latency`/`--rtf`（`90ed8d7`）、`preprocess.py --drop-late-and-stall`（`aa9a6ff`）已在 `e40c693` 合入；两条 Gazebo 流水线透传已在卡 W1-15b（`402096a`，合并 `035b398`）合入。
- **A4（统计表协议一致性检查）**：卡 W1-14（`087d2a8`，合并 `0d7f3b5`）已将 `action_latency_s`、`heading_guard`、`rtf`、`student_cond`、`dynamics_profile` 纳入 `stats_report.py` 默认拦截键。
- **G1 / C2 / F1 前半（3DGS 学生适配器上一步实际执行动作）**：卡 S1a（`32caa35`，合并 `5387747`）已修并带 14 条测试。

### 3.2 修了一半或存在遗留边界问题
- **A5（插件侧实际生效时刻 `plugin_ack_sim_t`）**：记账代码（`579901d`）与测量入口（`f4a2315`）已随 `e40c693` 合入 `main`，测量守护进程正在排队；但 `CfSitlExecutor.ack_sim_t()` 存在本报告 **`K-4`** 的四处边界缺陷，切为正式迟到判据前须补游标与双管道对齐。
- **卡 W1-10（满电档 `--cf-sitl-params`，合并 `41c474d`）**：`scene_gen.py`、`eval_closed_loop.py` 与 `tools/make_params_variant.py` 核实正确；但 `collect_parallel.py` 缺 CLI 参数与 `collect_command` 透传、`collect.py` 缺与 `scene_meta` 的双向核对（本报告 **`K-2`**），且满电档未联动 `hover_duty_estimate` 导致起飞 100% 越过 ToF/光流量程上限（本报告 **`K-3`**）。
- **G1 后半（`sim/tools/student.py` 在 `prev_dim > 0` 且调用方未传 `prev_action` 时报错）**：仍静默回落 `self._last_action`，按 `FIXLIST` 归卡 E2E-C1。
- **G6（评测记录并比对训练侧动力学与时序口径）**：`e40c693` 的 `46cd88c` 只补了 `action_latency_s` 缺失时的 `protocol_notes`，尚未比对训练数据集的 `dynamics_profile`（旧 `bd0dcd43` / `683e7400` 对评测 `1bc8fedf:lib31`）。

### 3.3 未修（按排期保持原状）
- `git diff base-20261112..fb3d1a5 -- sim/cf_sitl sim/dynamics` 除 `sim/cf_sitl/gz/README.md` 增加 5 行文档外**零代码改动**。
- 上一轮 `sim-real.md` 的 **E1**（`FAILSAFE` 开环保持悬停油门无界爬升，`executor.c:197-200`）、**E2**（起飞前馈手填不联动，在满电档体现为 `K-3`）、**E3**（`ready_vz_tol_mps=0.1` 对 ToF 噪声余量偏紧）、**E4**（`tof_rate < 20 Hz` 加载期不拦截）、**E5**（`yaw_deadzone` 读入未使用，`yaw_rate_rps` 掉入死区不报错）、**E6**（光流失效时 `FORWARD` 固定倾角开环加速不进 `FAILSAFE`）、**E7**（3DGS 线仍为 `L=0`，`e1_mini.json:64` `camera_latency` 仍为 `0.10`）、**E8**（`cf_sitl_proc` 无原始杆量 `sticks` 命令）、**E9**、**E10**（`e1_mini.json:50` 字符串偏移笔误 `0x01259a` 应为 `0x0101ac`）全部维持原状，待攒成一次 `lib32` 升级。
- 第一轮 `review-g.md` 的 **G2**（真机无动作回执）、**G3**（按步数航向码跨协议角度漂移）、**G4**（视场/裁剪敏感）、**G5**（步间刹停协议与文档闭环，待卡 W1-13 第三部分完成）、**G7**（部署握手契约，归卡 E2E-C1）、**G8**、**G10** 均按 `FIXLIST` 排期未动。

---

## 4. 查过但没有发现问题的地方

1. **坐标系、单位与控制符号全链路**：世界系 ENU、机体系 FLU、四元数 `(w, x, y, z)`（机体到世界）在 `plant.h`、`sensors.h`、`executor.c`、`CfSitlSystem.cc`、`CF-Drone` 固件间完全一致。IMU 芯片安装旋转 `imuRotation = (0, 0, -pi/2)` 与 `sensors.h:18` 的逆旋转抵消；光流视运动公式 `flow[0] = vx/h - wy`、`flow[1] = vy/h + wx` 与 `executor.c:71-72` 反解符号一致；执行器 `+x` 前进给正 `pitch`（低头）、`+y` 左移给负 `roll`（左滚）、`YAW_LEFT`（`+z`）给负 `yaw` 杆（对应 `control.ino:166` `ratesExtra.z = -controlYaw * maxRate.z > 0`）、速度环积分 `vel_int` 与位置保持 `pos_err` 随偏航角速度按 $\dot u = -\omega \times u$ 旋转补偿、四电机 `[RL, RR, FR, FL]` 位置力矩与旋向反扭矩符号（CCW 给 `-z` 反扭矩）全部核实正确，`selftest` 场景 1/3/3.5 实跑验证通过。
2. **Gazebo 外部物理接口与话题先后**：`CfSitlSystem::ReadBody` 读取的 `WorldLinearAcceleration` 在 Gazebo Harmonic 中为运动学加速度（静止为 `0`、自由落体为 `-g`），传入 `cf_sitl.cpp:137` 加 `+g·z_hat` 后得到准确的比力真值；返回的机体系力与力矩不含重力、含气动阻力，经 `pose.Rot().RotateVector` 转世界系交 `AddWorldWrench` 作用在刚体原点，无重复计入重力。`PreUpdate` 内严格按 `DrainCmds → DrainActions → cf_sitl_step_external → AddWorldWrench → PublishState` 顺序单线程执行，`OnAction`/`OnCmd` 仅持互斥锁入队，`DrainCmds` 释放锁后在 `reset` 分支单独加锁清空 `actions_`，无死锁、无跨集残留动作。
3. **固件原件完整性与跨集复位确定性**：`/root/claude-code-backup-20261028/code/CF-Drone` 提交 `e7f507f` 除 CRLF 外无任何本地改动；`fw_sitl.cpp:35-103` 对照全部被 `#include` 的 `.ino` 文件逐一核查，控制与估计路径上的全局变量及 17 个函数内 `static` 变量全部复位，无跨集污染；编译选项 `-ffp-contract=off -fno-fast-math` 与自写 `splitmix64 + Box-Muller` 保证同进程重复 `reset`（`test_reset_determinism`）与跨进程（`selftest` 场景 4，FNV-1a 哈希 `6b0993fb0b761015`）逐位一致，独立积分与外部模式 3 s 位置差 `< 1 cm`（`test_external_vs_standalone`）。
4. **C/C++ 内存安全与进程隔离**：`executor.c` 无动态分配、无未初始化栈读、除数均有正值保护（`ts > 0.0f`、`duty_max > duty_min`、`per_stick > 0`）；`json_min.cpp` 设递归深度上限 `64`、`NaN/Inf` 写出为 `null`；`libcf_sitl.so` 用 `-fvisibility=hidden -Wl,-Bsymbolic` 隐藏全部固件全局符号；`cf_sitl_proc.cpp` 启动时 `dup(1)` 后将 `fd 1` 重定向到 `stderr`，固件任何打印不会污染 stdout JSON 行协议。
5. **电池功率与压降代数模型**：`plant.h:136-155` 的动量理论诱导功率 $T^{1.5}/\sqrt{2\rho A}/(\text{FM}\cdot\eta) + P_{\text{avionics}}$ 与二次方程母线电压大根公式量纲及数值同 `e1_mini.json` 的 `hover_check` 完全闭合；`disc <= 0` 时有 `voc / 2` 钳位保护，无 `sqrt` 负数未定义行为。
6. **3DGS 线步间衔接与 `make_params_variant.py`**：`sim/3dgs/dynamics.py:255-260` 在独立积分推进前下发 `until_s = t_sitl + dt + 0.5 ms`，渲染与推理期间 `cf_sitl_proc` 不步进，确认 3DGS 线无步间 `SAFE` 空档；`tools/make_params_variant.py` 原位替换与 `values_sha256` 指纹计算同 `cf_sitl_exec.py` 完全一致。

---

## 5. 没有查的地方与原因

- **未启动 Gazebo 实测真实跨进程话题往返延迟 $\Delta_{\text{pipe}}$**：任务书第 3 节明确禁止启动 Gazebo；该分布正由卡 W1-13 的守护进程 `/root/rivermind-data/uav_rgb/exp-w1-13-ack-20261112`（PGID `78842`）排队实测。
- **未重新编译 `sim/cf_sitl` 或运行 ASan / UBSan 插桩构建**：任务书规定现成构建目录 `$D/tmp/build-cf-sitl-lib31/` 只读、不在里面构建；内存与未定义行为通过全文件逐行静态审查加运行现成测试二进制核实。
- **未反汇编官方闭源固件二进制 `official_E1_MINI_20261102.bin`、未做真机硬件测试**：无真机硬件，闭源固件内部控制律参数与控制源仲裁逻辑不可见。
- **未审 `src/` C++/CUDA 训练推理栈、`proto/` 原型线、教师规划器与 3DGS 碰撞栅格几何、板端串口脚本 `tools/uart_action.py`**：分属本轮交叉审核角度 I1、I2、J、L、M、N、O、P 的职责范围。

---

## 过程记录

### 做了什么
- 按任务书第 1–3 节读完前置规则、计划 §0 与 §10、任务卡 W1-10 与 W1-13、`FIXLIST.md`、第一轮报告 `review-g.md`、`sim-real.md`、`review-a.md`，以及草稿笔记 `$D/tmp/phase12-cards/w1-13/handoff-from-39b5006b.md` 与报告 `reports/w1-13.md`。
- 逐文件通读 `sim/cf_sitl/` 全部源码（`include/`、`exec/`、`src/`、`fw/`、`gz/`、`proc/`、`selftest/`、`tests/`、`CMakeLists.txt`）、固件原件 `/root/claude-code-backup-20261028/code/CF-Drone/` 全部相关 `.ino/.h`、`sim/dynamics/e1_mini.json`、`sim/tools/cf_sitl_exec.py`、`sim/tools/gz_io.py`、`sim/tools/collect.py`、`sim/tools/collect_parallel.py`、`sim/3dgs/dynamics.py`、`tools/make_params_variant.py`。
- 编写只读数值探针 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-k/probe_w113.py`，用 `nice -n 15` 调用现成 `$D/tmp/build-cf-sitl-lib31/cf_sitl_proc` 完成前飞刹停弹性反事实分解、自由刹停减速度曲线、`next-dispatch` 转向单步与连续步测量、满电档 40 集起飞过冲测试、L1 低电禁解锁绕过复现，并运行现成 C++ 单测与 33 项 `cf_sitl_selftest`。
- 撰写本报告至 `/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/review2-k.md`。

### 没做什么
- 未修改 `/root/claude-code-backup-20261028/code/uav_rgb`、`/root/claude-code-backup-20261028/code/CF-Drone`、`/root/claude-code-backup-20261028/code/wt/*` 或 `$D/tmp/build-cf-sitl-lib31/` 中的任何文件（含未写仓库内 `docs/worklog/` 与 `progress.md`）：原因系严格遵守任务书只读要求。
- 未启动 Gazebo、未跑训练、未跑 `tools/check.sh`、未触碰任何池锁或外部守护进程：原因系严格遵守任务书资源与只读约束。
- 未跑 `test_forward_lateral`（30 集 × 30 s）与 `sweep_seeds.sh`（40 种子全量自检）：原因系控制单次验证耗时与 CPU 占用，避免干扰正在运行的 Gazebo 闭环实验。

### 验证到哪一步

| 检查项 / 命令 | 结果 |
| --- | --- |
| `git -C /root/claude-code-backup-20261028/code/CF-Drone diff --ignore-cr-at-eol --stat` | 通过（输出为空，与 `e7f507f` 零代码差异） |
| `nice -n 15 $B/gz/test_flatten_state` | 通过（`40 fields PASS`） |
| `nice -n 15 $B/tests/{test_json_min,test_params_file,test_reset_determinism,test_external_vs_standalone,test_proc_protocol}` | 全部通过（`0 项失败`，日志见 `sitl-tests.log`） |
| `nice -n 15 $B/cf_sitl_selftest --params .../e1_mini.json --out .../selftest-s1 --seed 1` | 通过（`0 项失败，33 项通过`） |
| `nice -n 15 .venv-tools/bin/python sim/tools/cf_sitl_exec.py --self-check` | 通过（`PASS`） |
| `nice -n 15 .venv-tools/bin/python .../review2-k/probe_w113.py` | 通过（输出 `probe_w113_results.json`，复现 `K-1`、`K-3`、`K-5`、`K-6` 及审核 A 发现 1 拆解） |


