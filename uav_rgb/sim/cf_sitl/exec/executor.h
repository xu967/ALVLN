// executor.h：E1 MINI 动作执行器——用仿真 ToF/光流/固件姿态做高度环与水平速度环，输出固件 STAB
// 模式的四个杆量（代表将来改过的固件里的「自动飞行层」）。纯 C、float、无动态分配、固定 1 ms 步，
// 将来原样移植进 ESP32 固件（移植说明见 ../README.md）。
// 坐标：机体 FLU；四元数 w,x,y,z 为机体→世界（与固件 attitude 相同）。所有数值来自
// sim/dynamics/e1_mini.json 的 executor 项，由宿主填进 cf_exec_params；本文件不含第二份默认值。
#ifndef CF_EXEC_EXECUTOR_H
#define CF_EXEC_EXECUTOR_H

#ifdef __cplusplus
extern "C" {
#endif

enum cf_exec_state { CF_EXEC_DISARMED = 0, CF_EXEC_TAKEOFF = 1, CF_EXEC_READY = 2, CF_EXEC_ACTING = 3, CF_EXEC_FAILSAFE = 4 };
enum cf_exec_action { CF_EXEC_SAFE = 0, CF_EXEC_FORWARD = 1, CF_EXEC_YAW_LEFT = 2, CF_EXEC_YAW_RIGHT = 3 };

typedef struct {
	float dt_s;                                   // 步长（0.001）
	float forward_speed_mps, yaw_rate_rps;        // FORWARD / YAW_* 的目标值
	float tilt_max_deg, tilt_per_stick_deg;       // 倾角上限；固件 STAB 杆量→倾角比例（control.ino:165 tiltMax）
	float yaw_rate_per_stick_dps, yaw_deadzone;   // 固件偏航杆→角速度比例（maxRate.z）与死区（control.ino:152）
	float stick_deadband, duty_min, duty_max;     // 固件油门映射：duty = duty_min + (duty_max−duty_min)·(杆−deadband)/(1−deadband)
	float hover_duty;                             // 悬停占空比估计 → 初始油门前馈杆量
	float height_kp, height_kd, height_ki;        // 高度环：杆量/m、杆量/(m/s)、杆量/(m·s)
	float height_filter_alpha, height_filter_beta; // ToF 高度 g-h 观测器增益（按 ToF 样本更新）
	float vel_kp, vel_ki;                         // 速度环：倾角 rad/(m/s)、rad/m
	float vel_filter_cutoff_hz;                   // 光流速度一阶低通截止
	float takeoff_climb_rate_mps;                 // 起飞时高度参考的爬升速率
	float ready_height_tol_m, ready_vz_tol_mps, ready_hold_s; // 到达目标高度的判据
	float tof_timeout_s;                          // 飞行中 ToF 连续无效超过此时长 → FAILSAFE
	float arm_yaw_stick;                          // 解锁手势的偏航杆（固件要求 > 0.95，control.ino:121）
} cf_exec_params;

typedef struct {
	float t_s;            // 当前 SITL 时间
	int fw_armed;         // 固件 armed（真机里执行器与固件同在一片 MCU，直接读）
	float att_q[4];       // 固件估计姿态 w,x,y,z
	float gyro[3];        // 固件 rates（机体系 rad/s，已滤波去偏置）
	float tof_m;          // 最近 ToF 读数，< 0 无效
	int tof_new;          // 本 tick 有新 ToF 样本
	float flow[2];        // 光流角速率 rad/s（机体轴）：flow[0] = vx/h − gyro_y，flow[1] = vy/h + gyro_x（见 sensors.h）
	int flow_valid, flow_new;
} cf_exec_input;

typedef struct { float roll, pitch, yaw, throttle; } cf_exec_sticks; // roll/pitch/yaw ∈ [−1,1]，throttle ∈ [0,1]

typedef struct {
	cf_exec_params p;
	int state;                        // cf_exec_state
	int action; float action_until_s; // 当前动作（−1 无）与截止时间
	float target_height_m, height_ref_m;
	float height_est_m, vz_est_mps;   // g-h 观测器状态
	int height_valid;                 // 起飞后收到过有效 ToF（之前 ToF 在最小量程内，按高度 0 处理）
	float last_tof_valid_s;
	float hover_stick;                // 悬停油门杆：前馈 + 积分
	float flow_vel[2];                // 最近一帧光流换算速度（未滤波）
	float vel_est[2], vel_int[2];     // 滤波速度；速度环积分（单位 rad 倾角）
	int flow_ok;                      // 最近一帧光流有效
	float vel_alpha;                  // 由 vel_filter_cutoff_hz 与 dt 导出的低通系数
	float ready_since_s;              // 满足 READY 判据的起始时刻，< 0 未满足
	cf_exec_sticks sticks;            // 本 tick 输出
} cf_exec;

void cf_exec_init(cf_exec *e, const cf_exec_params *p);   // 回到 DISARMED（对应 cf_sitl_reset）
void cf_exec_arm_takeoff(cf_exec *e, float target_height_m);
void cf_exec_set_action(cf_exec *e, int action, float until_s);
void cf_exec_tick(cf_exec *e, const cf_exec_input *in);   // 每 1 ms 一次，结果写在 e->sticks
float cf_exec_stick_from_duty(const cf_exec_params *p, float duty); // 固件油门映射的反函数

#ifdef __cplusplus
}
#endif
#endif // CF_EXEC_EXECUTOR_H
