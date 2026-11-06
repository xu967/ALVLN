# CfSitlSystem — E1 MINI 固件在环的 gz-sim8 System 插件

把 `sim/cf_sitl` 的固件在环库（`cf_sitl.h`）接进 Gazebo Harmonic：每个 1 ms 物理步读一次刚体真值，
跑一次固件循环（`cf_sitl_step_external`），把返回的机体系力/力矩转到世界系施加到 `base_link`。
Gazebo 负责刚体积分与碰撞；固件负责姿态/高度/速度闭环。本目录只允许经 gz-transport 话题被目录外代码使用
（`cf_sitl.h` 头部约定：目录外不得链接）。

## 文件

| 文件 | 作用 |
|---|---|
| `CfSitlSystem.hh/.cc` | 插件：`ISystemConfigure` + `ISystemPreUpdate`；话题收发、时钟换算、施力、追尾相机 |
| `CfSitlState.hh/.cc` | `cf_sitl_state` → `gz.msgs.Double_V` 展平（索引表见下） |
| `CMakeLists.txt` | 独立构建或被 `sim/cf_sitl/CMakeLists.txt` 以 `add_subdirectory(gz)` 包含（`CF_SITL_BUILD_GZ=ON`） |
| `tests/test_flatten_state.cc` | 索引表回归（不需要 Gazebo） |
| `tests/stub_cf_sitl.cpp` | 测试桩：实现 `cf_sitl.h` 的真值 PD 控制器，只用于真库未就绪时联调，**不是**正式路径 |
| `tests/smoke_gz.py` | 真机 Gazebo 冒烟（插件加载 / 1 ms / 200 Hz / READY / FORWARD / YAW / contact / 相机 / RTF） |

## 构建

```bash
# 独立构建，链接真库（找 libcf_sitl.so；或传 -DCF_SITL_LIB=/path/libcf_sitl.so -DCF_SITL_INCLUDE=...）
cmake -S sim/cf_sitl/gz -B $DATA_ROOT/uav_rgb/tmp/build-cf-sitl/gz -G Ninja \
      -DCF_SITL_LIB_DIR=$DATA_ROOT/uav_rgb/tmp/build-cf-sitl
cmake --build $DATA_ROOT/uav_rgb/tmp/build-cf-sitl/gz --parallel 8
# 测试桩（真库不在时）
cmake -S sim/cf_sitl/gz -B $DATA_ROOT/uav_rgb/tmp/agent-gz/build-gz-stub -G Ninja -DCF_SITL_GZ_STUB=ON
cmake --build $DATA_ROOT/uav_rgb/tmp/agent-gz/build-gz-stub --parallel 8
$DATA_ROOT/uav_rgb/tmp/agent-gz/build-gz-stub/test_flatten_state
```

CMake 选项：`CF_SITL_GZ_STUB`（OFF）、`CF_SITL_TARGET`（父工程里的库目标名，默认 `cf_sitl`）、
`CF_SITL_INCLUDE`（默认 `../include`）、`CF_SITL_LIB` / `CF_SITL_LIB_DIR`。依赖 gz-sim8、gz-plugin2、gz-msgs10。

运行时 Gazebo 要能找到 `libCfSitlSystem.so`：`GZ_SIM_SYSTEM_PLUGIN_PATH=<构建目录>`。
`collect_parallel.gz_env` 自动设置：优先 `$UAVRGB_CF_SITL_PLUGIN_DIR`，否则按
`build-cf-sitl/gz` → `build-cf-sitl` → `agent-gz/build-gz-stub` 找第一个含该 .so 的目录。

## SDF 参数（见 `sim/worlds/uav_rgb_e1.sdf`）

```xml
<plugin filename="CfSitlSystem" name="uavrgb::CfSitlSystem">
  <link_name>base_link</link_name>
  <params>sim/dynamics/e1_mini.json</params>   <!-- 相对路径按 $UAVRGB_REPO_ROOT 解析，否则按当前目录 -->
  <seed>0</seed>
  <action_topic>/e1_mini/action</action_topic>
  <cmd_topic>/e1_mini/cmd</cmd_topic>
  <state_topic>/e1_mini/state</state_topic>
  <state_hz>200</state_hz>                      <!-- 1000/state_hz 必须是整数 -->
  <chase_model>chase_cam</chase_model>          <!-- 没有这个模型就不驱动追尾相机 -->
  <chase_offset>-1.0 0 0.5</chase_offset>       <!-- 机体 yaw 系下的偏移 -->
  <chase_pitch>0.45</chase_pitch>
</plugin>
```

物理步长必须是 `0.001`：其他步长插件只报一次 `gzerr` 并拒绝施力（固件按 1 kHz 写死）。

## 话题约定

| 话题 | 类型 | 方向 | 内容 |
|---|---|---|---|
| `/e1_mini/action` | `gz.msgs.Double_V` | collect → 插件 | `[action, until_s_sim]`：0 SAFE_NEUTRAL、1 FORWARD_SLOW（+x 机体 0.3 m/s）、2 YAW_LEFT（z 正向 0.6 rad/s）、3 YAW_RIGHT；`until_s_sim` 是 **Gazebo 仿真秒**，插件减去 reset 时刻的基准换算成 SITL 秒。到达时已过期的动作计入 `late_actions` 并告警 |
| `/e1_mini/cmd` | `gz.msgs.StringMsg` | collect → 插件 | `reset <episode>`（SITL 时钟归零、基准=当前 gz 时间、清空动作队列）；`arm_takeoff <height_m>` |
| `/e1_mini/state` | `gz.msgs.Double_V` | 插件 → 外 | 200 Hz，索引表如下；`Double_V` 没有 header，**时间戳取 `[32]`** |
| `/model/e1_mini/pose`、`/model/e1_mini/odometry` | `Pose_V` / `Odometry` | OdometryPublisher | 200 Hz 真值（collect 的位姿通道） |
| `/uav/camera` `/uav/depth` `/uav/contact` `/uav/chase_camera` `/uav/top_camera` | 同 X3 世界 | 传感器 | 相机 320×240 60°，`update_rate` 由 `scene_gen --camera-rate` 改（E1 模板默认 10 Hz） |

命令/动作回调只入队，所有固件调用都在 `PreUpdate`（同一线程）；仿真暂停时不步进、不施力。

采集侧（`sim/tools/cf_sitl_exec.py`）的两条兜底，次数都有记录：`reset` 发出后 1 s 内没在 state 上看到
「DISARMED 且 `t_s` < 0.5」就重发（gz-transport 刚连上时首条消息可能丢，解锁前 reset 幂等；日志 `reset 发了 N 次`）；
起飞阶段进 FAILSAFE 则 reset→落地→重新 arm，最多 3 次，次数写进 episode manifest `takeoff_failsafe_retries`
（库侧 `tof_airborne_m` 修复后实测 0 次）。

### `/e1_mini/state` 索引表（`CfSitlState.cc`，`test_flatten_state` 回归）

| 索引 | 字段 | 说明 |
|---|---|---|
| 0 | `t_s` | SITL 时钟（秒，reset 归零） |
| 1 | `exec_state` | 0 DISARMED / 1 TAKEOFF / 2 READY / 3 ACTING / 4 FAILSAFE |
| 2 | `fw_mode` | 固件 mode RAW/ACRO/STAB/ALTHOLD/AUTO = 0..4 |
| 3 | `armed` | 0/1 |
| 4 | `action` | 当前生效动作，-1 无 |
| 5 | `action_until_s` | SITL 秒 |
| 6..9 | `motors[4]` | RL RR FR FL |
| 10 | `battery_v` | |
| 11..14 | `att_est` | 固件估计姿态 w x y z |
| 15 | `tof_m` | 无效为 -1 |
| 16,17 | `flow_vel` | 光流机体系水平速度 |
| 18 | `height_est_m` | 执行器高度估计 |
| 19,20 | `vel_est` | 执行器机体系水平速度估计 |
| 21 | `hover_thrust` | |
| 22..24 | `pos` | 世界系（外部模式 = 插件传入的 Gazebo 真值） |
| 25..28 | `quat` | w x y z |
| 29..31 | `vel` | 世界系 |
| **32** | `gz_sim_s` | **Gazebo 仿真时间（秒）**，订阅方当时间戳用 |
| 33 | `late_actions` | 到达时窗口已过期的动作累计数 |
| 34 | `step_errors` | `cf_sitl_step_external` 返回非 0 的累计次数 |
| 35..38 | `sticks` | 执行器杆量 roll pitch yaw ∈[-1,1]、throttle ∈[0,1]（`cf_sitl.h` 2026-11-06 追加） |

Python 侧常量在 `sim/tools/cf_sitl_exec.py`（`IDX_*`、`STATE_FIELDS=39`）。

## 时钟与坐标

- gz → SITL：`t_sitl = t_gz - base`，`base` 在每次 `reset` 时记为当前 gz 时间（首步也隐式 reset 一次）。
- 传给固件的 `cf_sitl_body_in`：位置/姿态/线速度/角速度取 `Link::WorldPose/WorldLinearVelocity/WorldAngularVelocity`
  （世界系，角速度按头文件约定世界系），加速度取 `WorldLinearAcceleration`（运动学加速度，不含重力，符合头文件）。
- 返回的 wrench 是机体系：`F_w = R·f_b`，`τ_w = R·τ_b`，`Link::AddWorldWrench`。
- 追尾相机：独立 static 模型 `chase_cam`，每 20 步 `Model::SetWorldPoseCmd` 放到 `body.pos + R_yaw·offset`、俯仰 `chase_pitch`、
  只跟随位置与 yaw。

## 冒烟

```bash
OUT=$DATA_ROOT/uav_rgb/tmp/agent-gz/smoke; mkdir -p $OUT
DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg UAVRGB_RENDER_BACKEND=software \
  setsid nohup $UAV_RGB_PYTHON sim/cf_sitl/gz/tests/smoke_gz.py --out $OUT > $OUT/run.log 2>&1 < /dev/null & echo $! > $OUT/run.pid
# 结束写 $OUT/DONE；报告 $OUT/smoke.json；停止：kill -- -$(cat $OUT/run.pid)
```
