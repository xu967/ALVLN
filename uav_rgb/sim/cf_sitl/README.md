# cf_sitl：E1 MINI（CF-Drone 固件）在环仿真库

把 CF-Drone 固件（ESP32 Arduino .ino，原样编译）、动作执行器、电机/电池/刚体模型、IMU/ToF/光流模型串成每 1 ms 一步的锁步仿真，
编成 `libcf_sitl.so`，供 Gazebo 插件（`gz/`，外部刚体模式）、3DGS 进程（`proc/cf_sitl_proc`，独立积分模式）与自检程序共用。
C 接口见 `include/cf_sitl.h`（签名冻结）。

## 目录

| 路径 | 内容 | 归属 |
|---|---|---|
| `include/cf_sitl.h` | C 接口（冻结；`cf_sitl_state` 只能在末尾追加字段） | 公共 |
| `src/plant.h` | 参数读取（`Params::load`）、RNG、电池、电机、阻力、刚体 | SITL-core |
| `src/sensors.h` | IMU（芯片旋转、零偏、随机游走、DLPF、噪声）、ToF、光流 | SITL-core |
| `src/cf_sitl.cpp` | 接口实现；两种模式共用的 `tick()` | SITL-core |
| `src/smoke_main.cpp` | 冒烟程序 `cf_sitl_smoke`：reset → arm_takeoff(2.0) → 独立模式 15 s | SITL-core |
| `exec/executor.{h,c}` | 动作执行器（纯 C，将来移植进固件） | SITL-core |
| `fw/` | 固件翻译单元 `fw_sitl.cpp` + Arduino/ESP32 桩 `fw/stub/` | 已完成（见 HANDOVER） |
| `src/json_min.{h,cpp}` | 最小 JSON 读写 | 已完成 |
| `proc/`、`selftest/`、`tests/`、`tools/check.sh` | 管道进程、自检、CTest | SITL-tools agent |
| `gz/` | Gazebo 插件 | Gazebo agent |
| `../dynamics/e1_mini.json` | 全部物理/传感器/执行器数值（本库唯一的数字来源） | SITL-core |

## 构建

依赖：CMake ≥ 3.22、Ninja、g++（C++17/C11）、Eigen 3.4（`/usr/include/eigen3`），以及只读的固件源码目录 `CF_DRONE_DIR`
（默认 `${CMAKE_CURRENT_SOURCE_DIR}/../../../CF-Drone`，即仓库旁的 `../CF-Drone`；不存在时 configure 报错，`tools/check.sh` 则整段 SKIP）。

固件源码不在本仓库，从零克隆时要自己放好：ALVLN 仓库 `references/snapshots/CF-Drone.zip` 解包即是
CF-Drone **e7f507f**（与 GitHub `https://github.com/songge8/CF-Drone` 同提交，只差 CRLF），解到任意目录后
`export CF_DRONE_DIR=<该目录>`（含 `CF-Drone.ino`）。`check.sh` 用同一个变量，`UAV_RGB_SITL_BUILD_DIR` 指定它的构建目录。

```bash
SRC="$UAV_RGB_ROOT/sim/cf_sitl"                                              # source env.sh 后可用
BUILD="$DATA_ROOT/uav_rgb/tmp/build-cf-sitl"                                 # 放数据盘，根分区装不下
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release -DCF_DRONE_DIR="$CF_DRONE_DIR"   # 可加 -DCF_SITL_BUILD_GZ=ON
cmake --build "$BUILD" --parallel 8                                          # 产物：$BUILD/libcf_sitl.so、$BUILD/cf_sitl_smoke；-j4 实测约 1 min
"$BUILD"/cf_sitl_smoke [e1_mini.json 路径] [seed]                            # 到 READY 退出码 0，否则 1
ctest --test-dir "$BUILD" --output-on-failure                                 # 9 项；-DCF_SITL_BUILD_GZ=ON 时多一项 gz test_flatten_state
"$BUILD"/cf_sitl_selftest --params sim/dynamics/e1_mini.json --out "$BUILD"/selftest-out [--seed 1]   # 33 项，退出码 = 失败数
sim/cf_sitl/selftest/sweep_seeds.sh "$BUILD"                                 # 种子 1..40 全量自检，汇总每种子 PASS/FAIL；改阈值/执行器参数后必跑
```

编译选项：Release（`-O3 -DNDEBUG`）加 `-ffp-contract=off -fno-fast-math`（两种模式、两次 reset 逐位一致），`-fvisibility=hidden -fvisibility-inlines-hidden
-fno-gnu-unique -Wl,-Bsymbolic -Wl,--no-undefined`，只导出 8 个 `cf_sitl_*` 符号（7 个接口 + 自检用的 `cf_sitl_set_sticks_raw`；`nm -D libcf_sitl.so` 可查）。
固件源码通过 `fw/fw_sitl.cpp` 的 `#include` 进入同一翻译单元，库内只有一个固件实例，一个进程只能有一个 SITL 实例。

`CF_SITL_BUILD_GZ=ON` 时 `add_subdirectory(gz)`；`proc/`、`selftest/`、`tests/` 的 `CMakeLists.txt` 存在时自动加入，`ctest` 在构建目录运行。

## 接口语义（对 `cf_sitl.h` 注释的补充）

- `cf_sitl_init(json, seed)`：读参数、建模型，并做一次隐式 `cf_sitl_reset(0, {0,0,ground_z}, 0)`。`sim.dt_s` 不是 0.001 → 返回 1。
  `CF_SITL_VERBOSE=1` 环境变量打开固件串口打印（到 stderr）。
- `cf_sitl_reset(episode, spawn, yaw)`：噪声种子 = `seed ^ episode`；SITL 时钟归零；固件全局变量回上电状态（等同 `setup()`）；执行器 DISARMED；
  电池回到 `sim.battery_soc_init`。独立模式出生在 `spawn`（`spawn` 为 NULL 时 `{0,0,ground_z}`）。
- `cf_sitl_arm_takeoff(h)`：执行器先打解锁手势（油门 < 0.05、偏航杆 = 1.0）直到固件 `armed`，再按 0.5 m/s 斜坡把高度参考抬到 `h`。
  FAILSAFE 下忽略（只能 reset 退出）。
- `cf_sitl_set_action(a, until_s)`：`a ∉ {0,1,2,3}` 视为无动作。过期自动回悬停，`exec_state` 回 READY。
- `cf_sitl_step_external(in, out)`：返回 0 成功；1 未 init；2 `t_s < 0` 或与上一次的间隔不是 0.001 s（容差 1e-6）；3 本集已用过独立模式。
  `omega`/`acc` 为世界系，库内换到机体系：`ω_b = q⁻¹ω_w`，比力 `f_b = q⁻¹(a_w + g·ẑ)`。返回 wrench 为机体系、作用在机体原点，含阻力不含重力。
- `cf_sitl_step_standalone(n)`：返回 0 / 1（未 init）/ 3（本集已用过外部模式）。地面是 `z = sim.ground_z_m` 的水平面，点接触；
  落地（z ≤ 地面且 vz ≤ 0）后速度、角速度清零，只保留偏航。
- `cf_sitl_get_state`：
  - `exec_state`、`action`（仅 ACTING 时为动作号，否则 −1）、`action_until_s`；
  - `motors[4]` 为固件 `motors[]`（RL, RR, FR, FL），`att_est` 为固件估计姿态（飞行中不用加计修正，带陀螺积分漂移）；
  - `battery_v` 是电池模型的母线电压 V_bus（固件自己以 2 Hz 读到的是同一个量的采样）；
  - `tof_m`（无效 −1）、`flow_vel[2]`（最近一帧光流换算的机体系速度，未滤波）、`height_est_m`、`vel_est[2]`（3 Hz 低通后）；
  - `hover_thrust` 是执行器当前悬停油门**杆量**（前馈 + 积分，∈[0,1]），不是占空比；
  - `pos/quat/vel`：独立模式为内部刚体，外部模式为插件最近传入的值；
  - `sticks[4]`（2026-11-06 追加在末尾）：执行器最近一个 tick 的 roll, pitch, yaw ∈[−1,1]、throttle ∈[0,1]。
  - `lib_version`（2026-11-09 追加在末尾）：= 头文件 `CF_SITL_LIB_VERSION`（31 = v3.1），init 前也填。采集侧（`sim/3dgs/dynamics.py`、
    `sim/tools/cf_sitl_exec.py`）把它拼进 `dynamics_profile = cf_sitl_v1:<e1_mini.json 各项 value 规范化 JSON 的 sha8>:lib<N>`：
    改参数注释不换 profile，改 value 或改库（版本号 +1）才换，`tools/preprocess.py` 据此拒绝混包。**改了会让同一参数下轨迹不同的代码就把
    版本号 +1**，并在下面「版本记录」写一行。
- FAILSAFE：飞行中 ToF 连续无效 > `executor.tof_timeout_s`（0.5 s）→ 执行器输出油门 = 当前悬停杆、水平、偏航 0，不再闭环；
  高度估计停止外推。头文件注释写的「固件 descend() 接管」在这版里**没有**发生（RC 流每 tick 都在刷新，固件不会判 RC 丢失），
  由调用方把 FAILSAFE 当作终止条件。目标高度 3 m（超 ToF 量程 2.5 m）的实测：5.0 s ToF 失效，5.5 s 进 FAILSAFE，之后以 ~0.5 m/s 开环漂升。

## 每一步做什么（`src/cf_sitl.cpp: tick()`）

1. 真值 → IMU 样本（机体 → 芯片系 `imuRotation` 偏航 −90°；零偏 + 随机游走；184 Hz 一阶 DLPF；按平均电机转速在静止/飞行噪声间插值）→ `fw_set_imu`；电池 V_bus → `fw_set_battery`。
2. 每 `1000/tof_rate` 步采一次 ToF：`(z − ground)/cos(倾角)` + 噪声，量程外 −1；每 `1000/flow_rate` 步采一次光流（见下节约定）。
3. 执行器 `cf_exec_tick`：输入上一 loop 的固件姿态/角速率、ToF、光流 → 四个杆量。
4. `fw_set_time_us` → `fw_set_sticks`（每 tick 一帧，固件永远不判 RC 丢失）→ `fw_loop` → `fw_get_state`（读 `duty[4]`）。
5. 电机一阶滞后（上升 τ 0.06 s / 下降 0.12 s）→ 推力 `T = T_full·(d·V_bus/3.7)^1.5`、反扭矩 `km·T`（CCW 旋翼给 −z 力矩）→ 合成机体 wrench；
   电池按功率模型 `P = ΣT^1.5/√(2ρA)/(FM·η) + 航电` 更新 I、SoC、V_bus；加线性 + 二次阻力。
6. 独立模式再做一步半隐式欧拉刚体积分。

## 执行器移植说明（`exec/executor.{h,c}`）

目标是原样搬进 ESP32 固件做「自动飞行层」，所以：纯 C、`float`、无动态分配、无 STL、固定 1 ms 步、所有数值由宿主填 `cf_exec_params`
（SITL 里从 `e1_mini.json` 的 `executor` 项读，代码中没有第二份默认值）。真机上宿主就是固件本身：把 `cf_exec_input` 换成固件的
`attitude`、`rates`、ToF/光流驱动读数，把 `cf_exec_sticks` 写进 `controls[]` 即可。

- **输出 = STAB 模式杆量**，按固件映射（`control.ino`）：倾角 = 杆 × 30°（`tilt_per_stick_deg`）；偏航角速度 = −杆 × 300°/s（正杆顺时针，
  所以 `YAW_LEFT`（+z）给负杆）；|偏航杆| < 0.1 置零，0.6 rad/s 对应 0.1146 刚过死区；油门杆 < 0.05 怠速，否则 duty = 0.10 + 0.80·(杆 − 0.05)/0.95。
  `cf_exec_stick_from_duty` 是油门映射的反函数，用于把 `hover_duty_estimate` 换成初始前馈杆量（0.824 → 0.910）。
- **符号**：+x 前进要低头 → `pitch = +tilt_x/30°`；+y（左）加速要左滚 → `roll = −tilt_y/30°`。两者都在 SITL 里闭环验证过（FORWARD 给机体 vx ≈ +0.3 m/s；
  YAW_LEFT 偏航角增加，约 34 °/s）。
- **高度观测**：`h = ToF 斜距`（2026-11-07 起不再乘 `cos(估计倾角)`：固件飞行中只积分陀螺，估计倾角含随时间增长的零偏误差，乘上去会让真值高度跟着漂，
  seed 500 60 s 后 2.00 → 2.10 m；不补偿时误差 = 1 − cos(真倾角)，机动倾角 ≤ 10° 时 ≤ 1.5%，不随时间增长）。`cos(估计倾角) = 1 − 2(qx² + qy²)` 只用来
  判断估计姿态朝上。g-h 观测器按实际 ToF 样本间隔更新（α 0.4、β 0.08），样本之间按 `vz` 外推，
  失效超过 `tof_timeout_s` 停止外推。起飞前 ToF 在最小量程内（0.04 m）无效，这时按高度 0 处理、不计失效；收到第一个有效样本后才开始失效计时。
- **高度环**：`throttle = hover_stick + kp·err + kd·(vz_ref − vz)`，`vz_ref` 在爬升斜坡期间等于爬升率（0.5 m/s），其余 0；
  `hover_stick` 以 `ki·err` 积分（输出饱和且误差同向时冻结）。悬停杆 0.91 离满杆只剩 0.09 的向上权限，所以增益必须低（kp 0.15、kd 0.20、ki 0.05，
  理由见 `e1_mini.json executor.basis`）。
- **光流约定**（`sensors.h` 与 `executor.c: observe()` 一致）：`flow[0] = vx/h − ω_y`，`flow[1] = vy/h + ω_x`（机体 FLU 系、rad/s），
  速度 = `(flow[0] + gyro_y)·h`、`(flow[1] − gyro_x)·h`；高度不在 [`flow_min_height`, `flow_max_height`]（0.08–2.5 m）或角速率 > 7.4 rad/s 无效，
  无效帧按速度 0、积分冻结。**真机 PMW3901 的两轴符号与安装朝向未测**，移植时先在地面手推机体标定这两个符号。
- **速度环**：PI → 倾角目标：kp 0.22 rad/(m/s)、ki 0.30 rad/m（lib v3.1 起；之前 0.15）。P 项限 ±`tilt_max_deg`（10°，机动倾角），I 项**单独**限
  ±`tilt_int_max_deg`（20°），P 饱和时积分冻结；两者之和 ≤ 30° = 杆量满程。积分吸收的是固件飞行中不修正的姿态估计漂移（陀螺零偏 0.1 °/s 1σ，
  seed 500 的 pitch 轴 0.22 °/s）与阻力配平。2026-11-07 之前积分与 P 共用 10° 限幅：姿态漂移吃满 10° 后速度环开环，FORWARD 反向加速
  （3DGS route 5 seed 500 t≈44 s 倒飞、−3 m/s 撞树），`tests/test_forward_longrun.cpp` 用同一噪声流复现并验证修复。20° 在 3σ 零偏（0.29 °/s）下
  可撑 69 s，覆盖 3DGS 单集 ≤ 47 s；selftest 场景 6 全程约 98 s，两轴合成零偏 ≥ 0.2 °/s（约 12% 的种子，40 个里 2 个）会在末段真饱和，
  场景 6b 只在 (20° − 4° 配平/卷绕余量)/|b| > 总时长 时才要求未饱和，否则记信息行（`selftest_config.h kLongrunInt*`；
  CTest `tests/test_longrun_int_budget.cpp` 用同一判据兜底）。更长的飞行要靠固件在飞行中估零偏（待办）。
  **FORWARD 侧偏（b/ki）**：PI 环追零偏斜坡的稳态误差是 b/ki，估计速度被压到 0 时真速度 = b_x/ki（侧向）、−b_y/ki（前向）；SAFE/YAW 有位置环把它
  变成常值偏离，FORWARD 没有，航迹相对机头偏一个固定角。R2-SITL 2026-11-07 实测：Gazebo seed 190 集 26 个稳态 FORWARD 步侧向速度均值 +0.033 m/s
  （7 m 航程偏 0.8 m，偏角 6.3°，该集零偏 0.27 °/s）；独立积分 30 集 FORWARD 30 s 稳态机体 |vy| 均值 ki 0.15 → 1.29 cm/s（2/30 超 3 cm/s，
  航迹偏角 1σ 2.5°），ki 0.30 → 0.80 cm/s（最大 2.0，0/30 超 3 cm/s），ki 0.45 → 0.64 但 FORWARD 欠调（3 s 末 0.271 m/s、交替段最小 vx 0.225）。
  所以取 0.30；代价：悬停倾角 RMS 0.43 → 0.49°，FORWARD 到 90% 0.72 → 0.61 s，场景 3 稳态 0.313 → 0.295 m/s。CTest `tests/test_forward_lateral.cpp`
  守住 30 集 |vy| 均值 < 1.0 cm/s、每集 < 3 cm/s。积分抵消的姿态估计误差在世界系固定（只积分陀螺），积分本身是机体系倾角，
  所以机体偏航时积分按 u̇ = −ω×u 随之反转（2026-11-07 审核补）：不反转时飞 30 s 后 YAW 6 s（≈206°）侧移 0.4–1.9 m，反转后 0.02–0.13 m
  （`tests/test_yaw_in_place.cpp`）。
- **位置保持**（2026-11-07 新增）：SAFE / 无动作 / 起飞段都保持进入时的位置：机体系偏离 `ṗ = v_est − ω×p` 积分光流速度，外环 P（`pos_kp` 0.5 1/s）
  给速度环参考（限 ±`pos_vmax_mps` 0.3），光流无效时冻结。只有速度环时姿态漂移让 SAFE 以 零偏/ki 匀速漂走（ki 0.15 时 ≈ 1.1 cm/s、ki 0.30 时
  ≈ 0.6 cm/s，1σ），有位置环后变成常值偏离 零偏/(pos_kp·ki)（≈ 2.3 / 1.1 cm）。**剩余漂移**：光流用带零偏的固件 `rates` 补偿旋转，估计速度偏
  `b×h`，速度环把它压到 0 时真速度 = −b×h（2 m 高、0.1 °/s → 0.35 cm/s，30 s 约 10 cm），位置环看不见，要消掉得估零偏；另有光流白噪声
  经位置环积成的随机游走（30 s 约 0.15 m，与零偏无关），selftest 场景 6a 的真值漂移判据是 0.20 m + 1.5·|b|·h·T（见 `selftest_config.h`）。
- **状态机**：DISARMED →(arm_takeoff)→ TAKEOFF（解锁手势 → 斜坡爬升；|h − 目标| < 0.1 m 且 |vz| < 0.1 m/s 持续 1 s）→ READY ↔ ACTING（动作未过期）；
  任一飞行状态 ToF 失效 > 0.5 s → FAILSAFE（只能 reset 退出）。
- 真机上还缺的：解锁前 2 s 静止让固件校准陀螺零偏（SITL 里立刻解锁，零偏残差就是 `gyro_bias_residual` 的抽样）；电池 L1 3.5 V 不解锁的处理；
  光流/ToF 驱动的真实延迟。

## 参数来源

全部数值在 `../dynamics/e1_mini.json`，每项 `{value, unit, source, status, basis}`，顶层 `calibrated: false`；`status` 为 `doc / firmware / assumed`，
`assumed` 项的 `basis` 写明推算过程与区间。SITL 读取的键：`sim.*`、`mass_flight`、`inertia_diag`、`motor_positions`、`motor_index_spin`、
`motor_thrust_full_duty`、`thrust_duty_exponent`、`battery_v_nominal`、`motor_tau_up/down`、`torque_thrust_ratio_km`、`drag_linear`、`drag_quadratic`、
`battery_capacity`、`battery_r_internal`、`battery_ocv_curve`、`battery_power_model`、`prop_diameter`、`imu_config.dlpf_hz/rotation_rpy`、
`gyro_*`、`acc_*`、`tof_*`、`flow_*`、`executor.*`、`hover_duty_estimate`。缺键直接报错退出（不补默认值）。
相对参数草案的改动：`flow_max_height` 2.0 → 2.5 m（目标高度 2.0 m 时光流必须有效，PMW3901 在室内 2 m 以上仍能输出，但精度下降，`basis` 已注明）；
新增 `battery_power_model`、`executor`、`sim`。

## 冒烟结果（2026-11-06，seed 1，`cf_sitl_smoke`）

```
READY 用时 6.643 s；READY 之后 8.4 s 内平均高度误差 0.014 m，末态 z=1.988 m，xy 漂移 0.165 m；平均油门杆 0.894，末态悬停前馈 0.894
```

seed 1–8 全部到 READY（6.5–8.1 s），高度误差 0.011–0.016 m，xy 漂移 0.10–0.49 m（开环漂移来源：固件姿态估计漂移，速度环积分补偿有滞后）。
外部模式用同一个刚体积分器在库外步进、同种子下与独立模式逐位一致（READY 6.643 s，末态 z 1.988 m）。

2026-11-07 加位置保持后同一命令 seed 1–8：READY 用时不变，xy 漂移 0.015–0.193 m（之前 0.097–0.487 m）；剩余部分主要是光流陀螺补偿零偏 `b×h`
（见「位置保持」一节）。`cf_sitl_selftest` 33 项通过，新增场景 6（SAFE 保持 30 s + FORWARD/YAW 交替 60 s）与 CTest `cf_sitl_forward_longrun`。

2026-11-07 **lib v3.1**（`e1_mini.json` `vel_ki` 0.15 → 0.30，`dynamics_profile` `cf_sitl_v1:cec5d6b0` → `cf_sitl_v1:683e7400`，旧数据按口径不能混包）：
FORWARD 稳态侧偏 b/ki 减半（见「速度环」）；selftest 阈值按 40 种子扫描修（场景 1 加计链 1.0° → 2.0°；6a 真值漂移常数项 0.05 → 0.20 m 补光流随机游走；
6b 积分饱和改成按零偏判断），`selftest/sweep_seeds.sh` seed 1..40：改前 11 红（6b 3、6a 4、场景 1 5），改后 40/40 全绿；seed 1 仍 33/33；
ctest 10/10（新增 `cf_sitl_forward_lateral`、`cf_sitl_longrun_int_budget`）。没做（待办）：飞行中估零偏（根治 b/ki 侧偏、SAFE 剩余漂移与积分饱和三件事）。

2026-11-09 **`CF_SITL_LIB_VERSION = 31`**（AUDIT_SIM #1；控制律不变，只加版本常量与 `cf_sitl_state.lib_version`）：profile 从
`cf_sitl_v1:<文件字节 sha8>`（683e7400）改成 `cf_sitl_v1:<value sha8>:lib31`。此前所有数据集（bd0dcd43 / cec5d6b0 / 683e7400）与新采集按 `preprocess`
口径都不能混包；gz 插件与 `cf_sitl_proc` 必须用本版头文件重建（state 多一个字段，Python 侧 `STATE_FIELDS=40`，旧插件会被明确拒绝）。

2026-11-09 **`CF_SITL_LIB_VERSION = 31`**（AUDIT_SIM #1；控制律不变，只加版本常量与 `cf_sitl_state.lib_version`）：profile 从
`cf_sitl_v1:<文件字节 sha8>`（683e7400）改成 `cf_sitl_v1:<value sha8>:lib31`。此前所有数据集（bd0dcd43 / cec5d6b0 / 683e7400）与新采集按 `preprocess`
口径都不能混包；gz 插件与 `cf_sitl_proc` 必须用本版头文件重建（state 多一个字段，Python 侧 `STATE_FIELDS=40`，旧插件会被明确拒绝）。
