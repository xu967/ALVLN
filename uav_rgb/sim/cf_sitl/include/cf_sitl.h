// cf_sitl.h：E1 MINI 固件在环仿真的 C 接口（Gazebo 插件、3DGS 进程、自检程序共用）。
// 本目录按 GPLv3 + CF-Drone 附加条款发布（见 ../LICENSE）；目录外的代码只能经
// gz-transport topic 或 cf_sitl_proc 管道使用它，不得链接。
//
// 约定：
// - 坐标：世界 ENU（z 朝上），机体 FLU（x 前、y 左、z 上），四元数 w,x,y,z。
// - 时间：SITL 时钟每次 cf_sitl_reset 归零（对应真机上电），固件 micros() 取这个时钟。
//   Gazebo 插件在 reset 时记下 gz 时间基准，把 gz 时间换算成 SITL 时间再传入。
// - 动作：0=SAFE_NEUTRAL 1=FORWARD_SLOW 2=YAW_LEFT 3=YAW_RIGHT。
//   SAFE=原地保持（光流位置保持 + 保持高度，保持点 = 进入 SAFE 时的位置）；FORWARD=机体前向定速；
//   YAW_LEFT/RIGHT=原地偏航，YAW_LEFT 为绕 z 正向（逆时针）。动作在 until_s（SITL 时间）前有效（t < until_s），过期回 SAFE。
// - 单线程：所有函数在同一线程调用（插件在 PreUpdate 里调，topic 回调只入队）。
// - 固件代码是全局变量写法，一个进程只能有一个实例。
#ifndef CF_SITL_H
#define CF_SITL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 库版本（手工递增整数，2026-11-09 起；31 = lib v3.1）。改了会让同一参数文件下轨迹不同的东西
// （执行器/控制律/传感器模型/积分器、固件源码版本）就 +1，并在 README「版本记录」写一行。
// 经 cf_sitl_get_state().lib_version → cf_sitl_proc state / gz 插件 state[39] 带出，Python 侧拼进
// dynamics_profile = cf_sitl_v1:<参数 value sha8>:lib<N>，preprocess 据此拒绝换库不换参数的数据混包。
#define CF_SITL_LIB_VERSION 31

// 执行器状态
enum cf_sitl_exec_state {
  CF_SITL_DISARMED = 0,
  CF_SITL_TAKEOFF = 1,   // 已解锁，爬升到目标高度
  CF_SITL_READY = 2,     // 在目标高度稳定悬停，可以接收动作
  CF_SITL_ACTING = 3,    // 正在执行未过期的动作
  CF_SITL_FAILSAFE = 4,  // 飞行中 ToF 连续无效 > tof_timeout_s：执行器输出悬停杆、水平、不再闭环（固件 descend() 不会接管：
                         // 杆量每 tick 都在刷新，固件判不出 RC 丢失）；只能 reset 退出，调用方应把它当终止条件
};

// Gazebo 插件每个物理步传入的刚体状态（来自 Gazebo，世界系）
typedef struct {
  double t_s;          // SITL 时间（秒），由插件换算
  double pos[3];       // 机体原点位置 m
  double quat[4];      // 机体姿态 w,x,y,z（机体到世界）
  double vel[3];       // 线速度 m/s
  double omega[3];     // 角速度 rad/s（世界系；SITL 内部转到机体系）
  double acc[3];       // 线加速度 m/s^2（世界系，不含重力；需开启 EnableAccelerationChecks）
} cf_sitl_body_in;

// 返回给插件的力和力矩（机体系，作用在 link 原点；不含重力，含阻力）
typedef struct {
  double force[3];
  double torque[3];
} cf_sitl_wrench;

// 状态读出（插件发布到 state topic；3DGS 进程写进轨迹；自检画曲线）
typedef struct {
  double t_s;
  int exec_state;            // cf_sitl_exec_state
  int fw_mode;               // 固件 mode（RAW/ACRO/STAB/ALTHOLD/AUTO = 0..4）
  int armed;
  int action;                // 当前生效的动作，-1 表示无
  double action_until_s;
  double motors[4];          // 固件 motors[]，顺序同固件：RL, RR, FR, FL
  double battery_v;
  double att_est[4];         // 固件估计的姿态 w,x,y,z（含陀螺积分漂移）
  double tof_m;              // 最近一次 ToF 读数（无效为 -1）
  double flow_vel[2];        // 光流换算的机体系水平速度 m/s
  double height_est_m;       // 执行器估计的高度
  double vel_est[2];         // 执行器估计的机体系水平速度
  double hover_thrust;       // 执行器当前的悬停推力（前馈 + 积分）
  // 仅独立积分模式有效；外部模式为插件传入的值
  double pos[3];
  double quat[4];
  double vel[3];
  // 2026-11-06 追加（只能在末尾加字段）：执行器最近一个 tick 输出的杆量 roll, pitch, yaw ∈[-1,1]、throttle ∈[0,1]
  double sticks[4];
  // 2026-11-09 追加：= CF_SITL_LIB_VERSION（init 前也填），proc state / gz state[39] 原样带出
  int lib_version;
} cf_sitl_state;

// params_json：sim/dynamics/e1_mini.json 路径。seed：噪声随机数种子。失败返回非 0，错误写 stderr。
int cf_sitl_init(const char* params_json, uint64_t seed);

// 回到上电状态：SITL 时钟归零、固件全局状态重置（等同 setup()）、执行器 DISARMED。
// episode 参与噪声种子。spawn_*：独立积分模式的出生位姿；外部模式忽略。
void cf_sitl_reset(uint64_t episode, const double spawn_pos[3], double spawn_yaw);

// 解锁并起飞到 target_height_m；之后 exec_state 依次为 TAKEOFF、READY。
void cf_sitl_arm_takeoff(double target_height_m);

// 下发一个动作，在 until_s（SITL 时间）前有效。调用方按窗口语义算 until_s：
// observation 语义取拍图时刻 + hold，dispatch 语义取下发时刻 + hold。
void cf_sitl_set_action(int action, double until_s);

// 外部模式（Gazebo）：按 1 kHz 每步调用一次，跑一次固件 loop，返回要施加的 wrench。
// 返回值非 0 表示 dt 不是 0.001 s 等错误。
int cf_sitl_step_external(const cf_sitl_body_in* in, cf_sitl_wrench* out);

// 独立积分模式（3DGS、自检）：内部刚体积分 n_steps 个 1 ms 步，地面 z=0 平面接触。
int cf_sitl_step_standalone(int n_steps);

void cf_sitl_get_state(cf_sitl_state* out);

// 调试入口（只给自检用）：旁路执行器，每 tick 用这组 STAB 杆量刷新固件（roll/pitch/yaw ∈[-1,1]，throttle ∈[0,1]）。
// roll 传 NaN 取消旁路，执行器恢复接管。用途：固件级打杆阶跃（倾角=杆量×30°，偏航=杆量×300°/s）。
void cf_sitl_set_sticks_raw(double roll, double pitch, double yaw, double throttle);

#ifdef __cplusplus
}
#endif

#endif  // CF_SITL_H
