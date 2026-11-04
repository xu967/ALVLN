# cf_sil 载具骨架：sim-only 串级 PID + 直接电机驱动（2026-11-04）

- 结论：完成 sim-only 骨架和纯函数自检；**未运行 Gazebo**，未做真机标定。
- 定位：**不是 CF-Drone SIL**。CF-Drone/control.ino 只有 STAB/ACRO，没有速度环/位置环；
  这里的水平速度环和高度保持都在伴飞电脑侧，高度用 Gazebo 真值 odometry。
- 文件：
  - `sim/tools/cf_sil.py`（新）：参数、控制器、Actuators 发布线程、飞行日志占位。
  - `sim/dynamics/cf_sil_v1.json`（新）：固件默认参数 + 仿真被控对象值 + 真机缺失项。
  - `sim/tools/collect.py`：新增 `--dynamics cf_sil`、`--cf-sil-params`、`--flight-log*`，
    后台电机线程接入四动作采集。
  - `sim/tools/test_cf_sil.py`（新）：49 项自检。
  - `sim/tools/test_collect.py`（只追加）：cf_sil 采集窗口/安全网回归，共 54 项。
  - `sim/worlds/uav_rgb.sdf`：**未改**。停用速度控制器走运行时 `/X3/enable=false`，
    不需要复制 world 或改模板；证据见下。

## 1. 控制链与接口

四动作 -> 水平速度/偏航角速率设定点 + 高度位置 PID -> 期望倾角/总油门
-> 角度 P -> 角速率 PID -> CF 混控 -> 4 个归一化油门 -> `omega = u * max_omega`
-> 发布 `/X3/gazebo/command/motor_speed`（`gz.msgs.Actuators.velocity`，单位 rad/s）。

- `SAFE_NEUTRAL`：水平速度设定点 0，高度保持 `altitude_target_m`。
- `FORWARD_SLOW`：机体系 +X 小速度设定点（默认 0.3 m/s）。
- `YAW_LEFT/RIGHT`：偏航角速率 ±0.6 rad/s，其余动作锁当前航向。
- 高度：`altitude_assumption=companion_hold`，真值 odometry 的 z 位置 PID。
  这是仿真假设，不是 CF 固件能力。
- 延迟：动作进入队列，`latency_s` 后生效（默认 0.02 s，占位）。
- 电机：Gazebo 的 MulticopterMotorModel 仍负责一阶滤波、推力 `kf*ω²`、
  反扭矩 `km*thrust`、`maxRotVelocity` 饱和。控制器不重复做电机滤波。

### CLI（给 eval agent）

```bash
# 采集/评测透传给 collect.py
--dynamics cf_sil
--cf-sil-params sim/dynamics/cf_sil_v1.json   # 默认值，可省略
--flight-log flight.ulg                        # 占位；需要 pyulog
--flight-log-csv flight.csv                    # 标准列，pyulog 缺失时用
```

`--cf-sil-params` 只能和 `--dynamics cf_sil` 一起用；`--flight-log` 与
`--flight-log-csv` 二选一，且只用于 cf_sil。

### profile / manifest

- `dynamics_profile = cf_sil_v1:<sha8>`，sha8 是参数 JSON 的 SHA-256 前 8 位；
  换参数后档名必变，`tools/preprocess.py` 已有的 dynamics_profile 混包门禁直接生效。
- `dynamics_params`（`CfSilParams.manifest()`）包含：
  `profile`、`sha256`、`calibrated`、`altitude_assumption`、
  `motor_command_units="rad/s"`、`firmware_source`、`hardware`、
  `hover_throttle_norm_sim`、`mass`/`inertia`/`kf`/`km`/`max_rpm`、
  `rotors`、`pid`、`pid_source`、`max_rate_rad_s`、`throttle_min/max`。
- manifest 顶层另有 `altitude_assumption="companion_hold"`（仅 cf_sil 集）和
  `flight_log` 路径（若给了日志）。
- 四动作、actions.jsonl/oracle.jsonl 字段不变；`action_executed` 仍是 0..3。
- 建议 cf_sil 采集加 `--window-from dispatch`：动作从电机线程派发时刻起算，
  窗口才和实际执行对齐；默认 observation 仍保留给旧基线复现。

## 2. 为什么 SDF 不用改

证据来自本机 Gazebo Harmonic 8.15 的 gz-sim 源码（只读核对）：

- `MulticopterVelocityControl.cc`：
  `enableTopic = robotNamespace + "/" + enableSubTopic`；SDF 是
  `<robotNamespace>X3</robotNamespace><enableSubTopic>enable</enableSubTopic>`，
  所以话题是 `/X3/enable`，消息 `gz.msgs.Boolean`。
  源码里 `controllerActive{true}` 是默认值，`OnEnable` 直接赋 `_msg.data()`。
- 收到 false 后，`PreUpdate` 在 `!controllerActive` 且 rotorVelocities 为零时直接 return，
  不会创建 `Actuators` component。
- `MulticopterMotorModel.cc`：如果 model 上有 `Actuators` component，它**优先于**
  transport 的 Actuators 消息；component 只在速度控制器真正发过电机指令后才会出现。
  cf_sil 路径不创建 twist publisher、不发 twist，所以不会触发 component。
- `Actuators.velocity` 注释单位和 MotorModel 源码均为 rad/s；`maxRotVelocity` 只做上限
  clamp。`momentConstant` 单位是 m（drag torque = km * thrust）。

因此 `cf_sil.py` 在启动时：
1. 等 `/X3/enable` 和 `/X3/gazebo/command/motor_speed` 有订阅者；
2. 在 `/X3/enable` 发布 `false`；
3. 启动 250 Hz 后台线程读 `/model/x3/odometry` 并发布 Actuators。

风险：如果同一条 sim 里别的工具又发 twist 给 `/X3/gazebo/command/twist`，
速度控制器仍可能重新激活并创建 component，之后 transport 电机会被忽略。
cf_sil 采集路径自身不会发 twist；混用旧工具时不要开 cf_sil。

## 3. 参数来源与占位

`sim/dynamics/cf_sil_v1.json` 分三层：

1. `firmware_default`：从 `CF-Drone/control.ino` 预填，标 `firmware_default`：
   - 角速率 PID：roll/pitch P=0.06 I=0.1 D=0.001 I_lim=0.3；
     yaw P=0.3 I=0.01 D=0.01 I_lim=0.3。
   - 角度环：roll/pitch P=6，yaw P=3，I/D=0。
   - 角速率上限 360/360/300 deg/s；最大倾角 30°。
   - 油门 MOT_THR_MIN=0.10、MOT_THR_MAX=0.90、解锁上限 0.05。
     SIL 没有解锁流程，0.05 只作为 provenance 记录。
   - IMU 1 kHz；D 低通 alpha=0.2（约 40 Hz）。SIL 控制频率 250 Hz，
     代码按时间常数换算 D 滤波系数（约 0.59）。
   - `pid.att` / `pid.rate` 直接使用这些值；`pid_source` 标了来源。
2. 仿真被控对象值（顶层 `mass`/`inertia`/`arm_length`/`kf`/`km`/`max_rpm`/`rotors`/`drag`）：
   来自 `sim/worlds/uav_rgb.sdf` 的 X3 等价模型，**不是 118 mm 真机参数**。
   `sim_plant_source` 已注明。
3. `hardware`：只确认了轴距 0.118 m（control.ino 注释“118mm 轴距”+ OSHWHub 硬件页）。
   mass/inertia/props/motors/thrust_curve/battery/hover_throttle **全部 null**，
   `calibrated=false`。没有用仿真值冒充真机。

`pid.pos` / `pid.vel` 是伴飞电脑侧占位值；`latency_s`、前进速度、偏航速率、
爬升率/垂直加速度也仍是占位值，等实测替换。

### 飞行日志接口（占位）

- `--flight-log <path.ulg>`：优先用 pyulog。当前环境**未安装 pyulog**，
  会明确报“需要 pyulog 或先用 --flight-log-csv 导出标准 CSV”，不会自动装依赖。
- `--flight-log-csv <path>`：接受 `t`（或 time/timestamp）、
  `qw/qx/qy/qz`、`wx/wy/wz`、`thrust|motor|throttle`、`x/y/z`（或 pos_x/pos_y/pos_z）。
  只做表头/时间列校验和摘要，**不做系统辨识**，控制参数仍来自 JSON。
- 日志路径会写进 manifest 的 `flight_log` 字段，便于追溯。

## 4. 自测（未运行 Gazebo）

```bash
python3 sim/tools/test_cf_sil.py
python3 sim/tools/test_collect.py
```

真实输出（2026-11-04，系统 `/usr/bin/python3`）：

```
test_cf_sil: 49 项通过
test_collect: 54 项通过
```

`test_cf_sil.py` 覆盖：固件默认值、参数档 sha8 变化、水平悬停推力≈mg、
高度 PID 上下修正、低 max_rpm 饱和、yaw 左右差动、FORWARD 俯仰/前向加速度方向、
延迟队列（含乱序派发）、动作越界拒绝、混控 desaturation、四元数/odometry 解析、
CSV 日志校验、`.ulg` 缺 pyulog 报错、未知扩展名拒绝。

`test_collect.py` 追加覆盖：cf_sil 动作交给 sil、窗口内不发 twist、不误判卡滞、
窗口超时/执行器故障时停桨。

CLI 占位接口的原始报错也已核对（不需要 Gazebo）：

```
collect.py: error: --flight-log 和 --flight-log-csv 只能二选一
collect.py: error: 飞行日志读取失败: 读取 .ulg 需要 pyulog，或先用 --flight-log-csv 导出标准 CSV；本仓库不自动安装该依赖
collect.py: error: --cf-sil-params 只能和 --dynamics cf_sil 一起用
```

## 5. Smoke test（未执行，留给有 Gazebo 的会话）

```bash
# 终端 1
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg gz sim -r sim/worlds/uav_rgb.sdf

# 终端 2：先短跑 5 步，确认起飞/线程/电机命令
python3 sim/tools/collect.py --dynamics cf_sil --policy forward \
  --steps 5 --hold 0.2 --window-from dispatch \
  --out /tmp/cf_sil_smoke --seed 0 --split train
```

检查点：
- stderr 出现 `cf_sil 执行器: cf_sil_v1:<sha8>（calibrated=False...）` 和 `起飞完成`；
- `gz topic -e -t /X3/enable` 能看到 false（或速度控制器不再发 motor_speed）；
- `/model/x3/odometry` 持续更新，episode_manifest.json 的 `dynamics_profile`
  是 `cf_sil_v1:<sha8>`，`altitude_assumption=companion_hold`；
- 退出后电机停桨，不残留 `gz topic -e` 子进程。

## 6. 与 x3 / cf_stab 的 A/B 口径

- 同场景集、同 seed、同仿真时间预算（例如 80 s）比较；cf_sil 建议 `--hold 0.2`
  （5 Hz），x3 旧档 `--hold 1.0`，两者不能按步数直接对齐。
- 报 SR/碰撞/stuck/false_forward 时注明 `dynamics_profile`；cf_sil 参数未标定，
  数字只能用于链路对照，不能当硬件性能结论。
- 本轮**没有跑任何对照实验**（未运行 Gazebo）。

## 7. 没做什么

- 没运行 Gazebo，没做 PID/悬停/延迟的闭环调参。
- 没做真机标定或系统辨识；`.ulg` 解析只留接口，pyulog 未安装。
- 没改 `sim/worlds/uav_rgb.sdf`（运行时 enable 已足够，避免影响 x3_velocity/cf_stab 默认路径）。
- 没改 `sim/tools/collect_parallel.py`、`eval_closed_loop.py`：它们的 `--dynamics`
  choices 仍只有 x3_velocity/cf_stab，需要各自 owner 加 cf_sil/透传 `--cf-sil-params`。
- 没实现 IMU 噪声/延迟、电池电压跌落、ESC 死区、真实推力曲线；模型仍是线性
  油门->转速假设。
- 没在 cf_sil 下启用 predictive teacher / teacher-rl（它们绑定 cf_stab 模型，
  collect 参数校验会拒绝）。
- 没写 `docs/worklog/`：本次文件所有权限定在指定文件，过程记录合并到本文件。

## 8. 风险与待确认

1. **Actuators component 优先级**：cf_sil 依赖“从不激活速度控制器”。若同一 sim
   有别的进程发 twist，速度控制器可能创建 component，之后 direct motor 命令会被忽略。
   smoke 时要检查 `/X3/enable` 和是否只有一个执行器在发命令。
2. **归一化映射**：`omega = u * max_omega` 是线性 ESC 假设；真机 thrust curve 缺失，
   悬停油门只在仿真里约 0.82，不能当实测值。
3. **控制频率**：固件 1 kHz，SIL 250 Hz；PID 增益直接照搬，闭环响应可能不同，需要实测重调。
4. **高度保持不是固件能力**：companion_hold 用真值 odometry；换真实状态估计后要重做。
5. **卡滞判定**：cf_sil 在 hold<0.5 s 时不启用 stuck 兜底，只靠 contact；长窗口阈值也需重测。
6. **参数档混包**：profile 带 sha8，但 `collect_parallel`/eval 还没透传 cf_sil，
   旧入口可能直接拒绝；这是预期的显式失败，不是静默降级。
7. **解释器**：cf_sil 的 Actuators 发布用进程内 `gz.transport13`。用没有该模块的
   venv python 跑 collect 会明确报错；应使用系统 `/usr/bin/python3`。
8. **max_rpm 舍入**：JSON 7639.437 rpm -> 799.99997 rad/s，与 SDF 800 rad/s 差 3e-5，
   可忽略；若要严格一致可写 7639.43727。
