// CF-Drone 固件翻译单元的库内接口（不导出，只给 src/cf_sitl.cpp、自检与测试用）。
// 每个仿真步（1 ms）：fw_set_time_us → fw_set_imu → fw_set_sticks → fw_loop → fw_get_state。
// 固件是全局变量写法，一个进程一份；复位用 fw_reset()（恢复全部全局/静态变量后再跑 setup()）。
#pragma once
#include <stdint.h>

struct CfFwState {
	float t, dt, loop_rate;
	int armed, mode, landed, inverted;           // mode: 0 RAW 1 ACRO 2 STAB 3 ALTHOLD 4 AUTO
	float control_roll, control_pitch, control_yaw, control_throttle; // 固件内 control*（偏航死区之后）
	float thrust_target;
	float motors[4];          // 固件 motors[]，索引 0=RL 1=RR 2=FR 3=FL
	float duty[4];            // ledcWrite 计数 / 1023（真机 MOSFET 实际占空比），同索引
	float att_q[4];           // 估计姿态四元数 w,x,y,z（机体 FLU → 世界，固件约定）
	float att_euler[3];       // 固件 toEuler()：roll, pitch, yaw (rad)
	float att_target_euler[3];
	float rates[3], rates_target[3], torque_target[3];
	float gyro[3], acc[3], gyro_bias[3];   // readIMU() 之后：已减偏置、已转到机体系
	float battery_v;
	float rate_i[3];          // 角速率环 PID 积分累计 roll/pitch/yaw
};

void fw_set_verbose(int on);
void fw_set_time_us(uint64_t us);
void fw_set_imu(const float gyro_chip[3], const float acc_chip[3]); // 芯片系，rad/s、m/s^2
void fw_set_battery(float volts);
// 相当于 readRC() 收到一帧并 normalizeRC() 之后：roll/pitch/yaw ∈[-1,1]，throttle ∈[0,1]，mode=NAN 不切模式
void fw_set_sticks(float roll, float pitch, float yaw, float throttle, float mode);
void fw_reset(void);   // 上电状态：全部固件全局/静态变量回初值 + setup()
void fw_loop(void);
int fw_set_param(const char *name, float value);
float fw_get_param(const char *name);
void fw_get_state(CfFwState *s);
