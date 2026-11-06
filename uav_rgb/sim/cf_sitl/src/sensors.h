// sensors.h：传感器模型（宿主侧）：IMU（MPU6500 芯片系样本）、ToF（VL53L1X 机体 −z 测距）、光流（PMW3901 角速率）。
// 共用一个 Rng（种子 = init seed ^ episode），抽样顺序固定，结果逐位可复现。数值全部来自 e1_mini.json。
#pragma once
#include "plant.h"

namespace sitl {

// IMU：机体→芯片旋转 + 零偏（每集抽样 + 随机游走）+ 一阶 DLPF + 白噪声（DLPF 之后注入，std 就是固件看到的 std）。
// 量化/量程在固件替身驱动（fw/stub/sitl_drivers.h）里做。
struct Imu {
	const Params &P; Rng &rng;
	Quat q_cb = Quat::Identity();    // 机体→芯片：imu.ino:60 body = rotate(chip, fromEuler(imuRotation).inversed())
	Vec3 g_bias = Vec3::Zero(), a_bias = Vec3::Zero(), g_lp = Vec3::Zero(), a_lp = Vec3::Zero();
	double alpha = 1; bool init = false;
	Imu(const Params &p, Rng &r) : P(p), rng(r) {}
	Vec3 gauss3() { const double x = rng.gauss(), y = rng.gauss(), z = rng.gauss(); return Vec3(x, y, z); }
	void reset() { // 派生常数在这里算：构造时 P 还没从 JSON 载入
		q_cb = Quat(Eigen::AngleAxisd(-P.imu_yaw, Vec3::UnitZ())); // imuRotation 只有偏航分量
		alpha = 1 - std::exp(-2 * M_PI * P.dlpf_hz * P.dt);
		// 残余零偏：落地校准后仍留下的部分（gyro）与六面校准后的部分（acc），每轴 std = 参数值
		g_bias = gauss3() * P.gyro_bias_res;
		a_bias = gauss3() * P.acc_bias_res;
		init = false;
	}
	// motor_mean：平均归一化转速 0..1，噪声在静态与飞行值之间线性插值（振动随转速增大）
	void sample(const Vec3 &w_body, const Vec3 &sf_body, double motor_mean, float g_out[3], float a_out[3]) {
		g_bias += gauss3() * (P.gyro_rw * std::sqrt(P.dt));
		const Vec3 g = q_cb * w_body, a = q_cb * sf_body;
		if (!init) { g_lp = g; a_lp = a; init = true; }
		g_lp += alpha * (g - g_lp); a_lp += alpha * (a - a_lp);
		const double sg = P.gyro_std_static + (P.gyro_std_flight - P.gyro_std_static) * motor_mean;
		const double sa = P.acc_std_static + (P.acc_std_flight - P.acc_std_static) * motor_mean;
		const Vec3 go = g_lp + g_bias + gauss3() * sg, ao = a_lp + a_bias + gauss3() * sa;
		for (int i = 0; i < 3; i++) { g_out[i] = (float)go[i]; a_out[i] = (float)ao[i]; }
	}
};

// ToF：机体 −z 轴射线到地平面 z = ground_z 的距离 = (z − ground)/cos(倾角)；量程外或倒置返回 −1
struct Tof {
	const Params &P; Rng &rng;
	Tof(const Params &p, Rng &r) : P(p), rng(r) {}
	double sample(const Vec3 &pos, const Quat &q) {
		const double cos_tilt = (q * Vec3::UnitZ()).z();
		if (cos_tilt <= 1e-6) return -1;
		const double d = (pos.z() - P.ground_z) / cos_tilt + rng.gauss() * P.tof_std;
		return (d < P.tof_min || d > P.tof_max) ? -1 : d;
	}
};

// 光流：下视相机看到的地面角速率（机体轴）。输出约定（执行器按此反解）：
//   flow[0] = vx/h − ω_y，flow[1] = vy/h + ω_x，其中 v 为机体系水平速度、h 为离地高度、ω 为机体角速度。
//   推导：地面点 p=(0,0,−h) 在机体系的视运动 u = −v − ω×p，角速率 u/h = (−vx/h + ω_y, −vy/h − ω_x)，取反号。
// 真机 PMW3901 的像素轴向/符号取决于安装方向，移植时按实物标定（README「执行器移植」）。
// 高度不在 [flow_min_height, flow_max_height] 或角速率超过 flow_max_rate 时无效。
struct Flow {
	const Params &P; Rng &rng;
	Flow(const Params &p, Rng &r) : P(p), rng(r) {}
	bool sample(const Vec3 &pos, const Quat &q, const Vec3 &vel_world, const Vec3 &w_body, double out[2]) {
		const double h = pos.z() - P.ground_z;
		const Vec3 vb = q.conjugate() * vel_world;
		if (h < P.flow_min_h || h > P.flow_max_h) { out[0] = out[1] = 0; return false; }
		out[0] = vb.x() / h - w_body.y() + rng.gauss() * P.flow_std;
		out[1] = vb.y() / h + w_body.x() + rng.gauss() * P.flow_std;
		if (std::fabs(out[0]) > P.flow_max_rate || std::fabs(out[1]) > P.flow_max_rate) { out[0] = out[1] = 0; return false; }
		return true;
	}
};
} // namespace sitl
