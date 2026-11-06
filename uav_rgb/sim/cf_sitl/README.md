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
（默认 `${CMAKE_CURRENT_SOURCE_DIR}/../../../CF-Drone`，即 `/root/claude-code-backup-20261028/code/CF-Drone/`；不存在时 configure 报错）。

```bash
SRC=/root/claude-code-backup-20261028/code/uav_rgb/sim/cf_sitl
BUILD=/root/rivermind-data/uav_rgb/tmp/build-cf-sitl
cmake -S "$SRC" -B "$BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release            # 可加 -DCF_DRONE_DIR=... -DCF_SITL_BUILD_GZ=ON
cmake --build "$BUILD" --parallel 8                                          # 产物：$BUILD/libcf_sitl.so、$BUILD/cf_sitl_smoke
"$BUILD"/cf_sitl_smoke [e1_mini.json 路径] [seed]                            # 到 READY 退出码 0，否则 1
```

编译选项：Release（`-O3 -DNDEBUG`）加 `-ffp-contract=off -fno-fast-math`（两种模式、两次 reset 逐位一致），`-fvisibility=hidden -fvisibility-inlines-hidden
-fno-gnu-unique -Wl,-Bsymbolic -Wl,--no-undefined`，只导出 7 个 `cf_sitl_*` 符号（`nm -D libcf_sitl.so` 可查）。
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
- **高度观测**：`h = ToF·cos(倾角)`，`cos(倾角) = 1 − 2(qx² + qy²)`；g-h 观测器按实际 ToF 样本间隔更新（α 0.4、β 0.08），样本之间按 `vz` 外推，
  失效超过 `tof_timeout_s` 停止外推。起飞前 ToF 在最小量程内（0.04 m）无效，这时按高度 0 处理、不计失效；收到第一个有效样本后才开始失效计时。
- **高度环**：`throttle = hover_stick + kp·err + kd·(vz_ref − vz)`，`vz_ref` 在爬升斜坡期间等于爬升率（0.5 m/s），其余 0；
  `hover_stick` 以 `ki·err` 积分（输出饱和且误差同向时冻结）。悬停杆 0.91 离满杆只剩 0.09 的向上权限，所以增益必须低（kp 0.15、kd 0.20、ki 0.05，
  理由见 `e1_mini.json executor.basis`）。
- **光流约定**（`sensors.h` 与 `executor.c: observe()` 一致）：`flow[0] = vx/h − ω_y`，`flow[1] = vy/h + ω_x`（机体 FLU 系、rad/s），
  速度 = `(flow[0] + gyro_y)·h`、`(flow[1] − gyro_x)·h`；高度不在 [`flow_min_height`, `flow_max_height`]（0.08–2.5 m）或角速率 > 7.4 rad/s 无效，
  无效帧按速度 0、积分冻结。**真机 PMW3901 的两轴符号与安装朝向未测**，移植时先在地面手推机体标定这两个符号。
- **速度环**：PI → 倾角目标（限 ±10°）：kp 0.22 rad/(m/s)、ki 0.15 rad/m；积分器同时吸收固件飞行中不修正的姿态漂移（陀螺零偏 0.1 °/s 量级，
  20 s 约 2–4°；smoke 实测稳态 roll 杆 −0.15 ≈ −4.5° 刚好抵消估计漂移）。
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
