// plant.h：E1 MINI 外部模型（宿主侧，不进固件）：参数读取、电机推力/反扭矩、电池、6DOF 刚体 + 地面。
// 世界 ENU、机体 FLU，四元数 q = 机体→世界（与固件 attitude 相同）。数值全部来自 e1_mini.json。
// 由 spike /root/rivermind-data/uav_rgb/tmp/sitl-spike/plant.h 搬来，补了电压相关推力、升降不同 τ、二次阻力、电池。
#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "json_min.h"
#include "../exec/executor.h"

namespace sitl {
using Vec3 = Eigen::Vector3d;
using Quat = Eigen::Quaterniond;
constexpr double G = 9.80665;

// ---- JSON 取值：每项形如 "key": {"value": ..., "unit", "source", "status", "basis"}，缺项直接抛错（init 返回非 0）----
inline const Json &item(const Json &j, const char *key) {
	const Json *v = j.get(key);
	if (!v) throw std::runtime_error(std::string("参数文件缺少 ") + key);
	return *v;
}
inline const Json &value(const Json &j, const char *key) { return item(item(j, key), "value"); }
inline double numf(const Json &obj, const char *field) {
	const Json &v = item(obj, field);
	if (!v.is_num()) throw std::runtime_error(std::string("参数 ") + field + " 不是数字");
	return v.num;
}
inline double num(const Json &j, const char *key) {
	const Json &v = value(j, key);
	if (!v.is_num()) throw std::runtime_error(std::string("参数 ") + key + ".value 不是数字");
	return v.num;
}
inline Vec3 arr3(const Json &a, const char *what) {
	if (a.type != Json::Array || a.arr.size() != 3) throw std::runtime_error(std::string("参数 ") + what + " 应为 3 元数组");
	return Vec3(a.arr[0].num, a.arr[1].num, a.arr[2].num);
}

struct Params {
	double dt, ground_z, soc0;
	double mass; Vec3 inertia; Vec3 r[4]; double spin[4];          // 电机位置/反扭矩方向，索引同固件 0=RL 1=RR 2=FR 3=FL
	double t_full, k_exp, v_ref, tau_up, tau_down, km;              // T = t_full·(n·V_bus/v_ref)^k
	double drag_xy, drag_z, drag_q;
	double cap_ah, r_int, fm, eta_motor, p_avionics, rho, prop_area; // 电池与功率模型
	std::vector<std::pair<double, double>> ocv;                      // [SoC, V_oc]，SoC 降序
	double imu_yaw, dlpf_hz, gyro_std_static, gyro_std_flight, gyro_bias_res, gyro_rw;
	double acc_std_static, acc_std_flight, acc_bias_res;
	double tof_min, tof_max, tof_std; int tof_steps;
	double flow_min_h, flow_max_h, flow_max_rate, flow_std; int flow_steps;
	cf_exec_params exec;

	static Params load(const Json &j) {
		Params P;
		const Json &sim = value(j, "sim");
		P.dt = numf(sim, "dt_s"); P.ground_z = numf(sim, "ground_z_m"); P.soc0 = numf(sim, "battery_soc_init");
		P.mass = num(j, "mass_flight"); P.inertia = arr3(value(j, "inertia_diag"), "inertia_diag");
		static const char *names[4] = {"RL", "RR", "FR", "FL"}, *idx[4] = {"0", "1", "2", "3"};
		const Json &mp = value(j, "motor_positions"), &sp = value(j, "motor_index_spin");
		for (int i = 0; i < 4; i++) {
			P.r[i] = arr3(item(mp, names[i]), names[i]);
			// 俯视 CCW 的桨给机体 −z 反扭矩（与 control.ino:224-227 混控符号一致）
			P.spin[i] = item(sp, idx[i]).str.find("CCW") != std::string::npos ? -1.0 : 1.0;
		}
		P.t_full = num(j, "motor_thrust_full_duty"); P.k_exp = num(j, "thrust_duty_exponent");
		P.v_ref = num(j, "battery_v_nominal"); P.tau_up = num(j, "motor_tau_up"); P.tau_down = num(j, "motor_tau_down");
		P.km = num(j, "torque_thrust_ratio_km");
		const Json &dl = value(j, "drag_linear");
		P.drag_xy = numf(dl, "xy"); P.drag_z = numf(dl, "z"); P.drag_q = num(j, "drag_quadratic");
		P.cap_ah = num(j, "battery_capacity"); P.r_int = num(j, "battery_r_internal");
		for (const Json &pt : value(j, "battery_ocv_curve").arr) P.ocv.emplace_back(pt.arr.at(0).num, pt.arr.at(1).num);
		if (P.ocv.size() < 2) throw std::runtime_error("battery_ocv_curve 至少 2 个点");
		const Json &pm = value(j, "battery_power_model");
		P.fm = numf(pm, "figure_of_merit"); P.eta_motor = numf(pm, "motor_efficiency");
		P.p_avionics = numf(pm, "avionics_w"); P.rho = numf(pm, "air_density_kgm3");
		const double d = num(j, "prop_diameter"); P.prop_area = M_PI * d * d / 4;
		const Json &imu = value(j, "imu_config");
		P.dlpf_hz = numf(imu, "dlpf_hz"); P.imu_yaw = arr3(item(imu, "rotation_rpy"), "rotation_rpy").z();
		P.gyro_std_static = num(j, "gyro_noise_std_static"); P.gyro_std_flight = num(j, "gyro_noise_std_flight");
		P.gyro_bias_res = num(j, "gyro_bias_residual"); P.gyro_rw = num(j, "gyro_bias_random_walk");
		P.acc_std_static = num(j, "acc_noise_std_static"); P.acc_std_flight = num(j, "acc_noise_std_flight");
		P.acc_bias_res = num(j, "acc_bias_residual");
		P.tof_min = num(j, "tof_range_min"); P.tof_max = num(j, "tof_range_max"); P.tof_std = num(j, "tof_noise_std");
		P.tof_steps = std::max(1, (int)std::lround(1.0 / (P.dt * num(j, "tof_rate"))));
		P.flow_min_h = num(j, "flow_min_height"); P.flow_max_h = num(j, "flow_max_height");
		P.flow_max_rate = num(j, "flow_max_rate"); P.flow_std = num(j, "flow_noise_std");
		P.flow_steps = std::max(1, (int)std::lround(1.0 / (P.dt * num(j, "flow_rate"))));
		const Json &ex = value(j, "executor");
		cf_exec_params &e = P.exec;
		e.dt_s = (float)P.dt; e.hover_duty = (float)num(j, "hover_duty_estimate");
		e.forward_speed_mps = (float)numf(ex, "forward_speed_mps"); e.yaw_rate_rps = (float)numf(ex, "yaw_rate_rps");
		e.tilt_max_deg = (float)numf(ex, "tilt_max_deg"); e.tilt_per_stick_deg = (float)numf(ex, "tilt_per_stick_deg");
		e.yaw_rate_per_stick_dps = (float)numf(ex, "yaw_rate_per_stick_dps"); e.yaw_deadzone = (float)numf(ex, "yaw_deadzone");
		e.stick_deadband = (float)numf(ex, "stick_deadband"); e.duty_min = (float)numf(ex, "duty_min"); e.duty_max = (float)numf(ex, "duty_max");
		e.height_kp = (float)numf(ex, "height_kp"); e.height_kd = (float)numf(ex, "height_kd"); e.height_ki = (float)numf(ex, "height_ki");
		e.height_filter_alpha = (float)numf(ex, "height_filter_alpha"); e.height_filter_beta = (float)numf(ex, "height_filter_beta");
		e.vel_kp = (float)numf(ex, "vel_kp"); e.vel_ki = (float)numf(ex, "vel_ki"); e.vel_filter_cutoff_hz = (float)numf(ex, "vel_filter_cutoff_hz");
		e.takeoff_climb_rate_mps = (float)numf(ex, "takeoff_climb_rate_mps");
		e.ready_height_tol_m = (float)numf(ex, "ready_height_tol_m"); e.ready_vz_tol_mps = (float)numf(ex, "ready_vz_tol_mps");
		e.ready_hold_s = (float)numf(ex, "ready_hold_s"); e.tof_timeout_s = (float)numf(ex, "tof_timeout_s");
		e.arm_yaw_stick = (float)numf(ex, "arm_yaw_stick");
		return P;
	}
};

// splitmix64 + Box-Muller：自带实现，不依赖 std::normal_distribution 的库实现，逐位可复现
struct Rng {
	uint64_t s = 0; bool has = false; double spare = 0;
	void seed(uint64_t v) { s = v; has = false; spare = 0; }
	uint64_t next() {
		uint64_t z = (s += 0x9E3779B97F4A7C15ull);
		z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull; z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
		return z ^ (z >> 31);
	}
	double uni() { return ((next() >> 11) + 0.5) * (1.0 / 9007199254740992.0); } // (0,1)
	double gauss() {
		if (has) { has = false; return spare; }
		const double r = std::sqrt(-2 * std::log(uni())), th = 2 * M_PI * uni();
		spare = r * std::sin(th); has = true;
		return r * std::cos(th);
	}
};

struct Wrench { Vec3 force = Vec3::Zero(), torque = Vec3::Zero(); }; // 机体系，不含重力

// 电池：OCV(SoC) − I·R_int；功率 = Σ 旋翼理想功率/(品质因数·电机效率) + 航电。T 依赖 V_bus，用上一步的 V_bus（1 ms 滞后）
struct Battery {
	const Params &P; double soc = 1, v_bus = 0, current = 0;
	explicit Battery(const Params &p) : P(p) {}
	double ocv(double s) const {
		const auto &c = P.ocv;
		if (s >= c.front().first) return c.front().second;
		for (size_t i = 1; i < c.size(); i++)
			if (s >= c[i].first) return c[i].second + (c[i - 1].second - c[i].second) * (s - c[i].first) / (c[i - 1].first - c[i].first);
		return c.back().second;
	}
	void reset() { soc = P.soc0; v_bus = ocv(soc); current = 0; }
	void step(const double thrust[4], double dt) {
		double p = P.p_avionics;
		for (int i = 0; i < 4; i++) p += std::pow(std::max(0.0, thrust[i]), 1.5) / std::sqrt(2 * P.rho * P.prop_area) / (P.fm * P.eta_motor);
		const double voc = ocv(soc), disc = voc * voc - 4 * p * P.r_int;
		v_bus = disc > 0 ? (voc + std::sqrt(disc)) / 2 : voc / 2; // 二次方程取大根；功率超出电池能力时钳在 V_oc/2
		current = p / v_bus;
		soc = std::max(0.0, soc - current * dt / 3600 / P.cap_ah);
	}
};

// 电机：占空比一阶滞后（升/降不同 τ），推力 T = t_full·(n·V_bus/v_ref)^k，反扭矩 km·T
struct Motors {
	const Params &P; double n[4] = {0, 0, 0, 0}, thrust[4] = {0, 0, 0, 0};
	explicit Motors(const Params &p) : P(p) {}
	void reset() { for (int i = 0; i < 4; i++) n[i] = thrust[i] = 0; }
	double mean_speed() const { return (n[0] + n[1] + n[2] + n[3]) / 4; }
	Wrench step(const float duty[4], double v_bus, double dt) {
		Wrench w;
		for (int i = 0; i < 4; i++) {
			const double d = std::min(1.0, std::max(0.0, (double)duty[i]));
			n[i] = d + (n[i] - d) * std::exp(-dt / (d > n[i] ? P.tau_up : P.tau_down));
			thrust[i] = P.t_full * std::pow(std::max(0.0, n[i] * v_bus / P.v_ref), P.k_exp);
			w.force.z() += thrust[i];
			w.torque += P.r[i].cross(Vec3(0, 0, thrust[i]));
			w.torque.z() += P.spin[i] * P.km * thrust[i];
		}
		return w;
	}
};

inline Vec3 drag_body(const Params &P, const Vec3 &vb) { // 线性（xy/z 不同）+ 二次
	const Vec3 lin(-P.drag_xy * vb.x(), -P.drag_xy * vb.y(), -P.drag_z * vb.z());
	return lin - P.drag_q * vb.norm() * vb;
}

// 独立积分模式的刚体：半隐式欧拉，硬地面 z = ground_z（点接触，落地后只保留偏航）
struct Body {
	const Params &P;
	Vec3 pos = Vec3::Zero(), vel = Vec3::Zero(), w = Vec3::Zero(); // 世界位置/速度，机体角速度
	Quat q = Quat::Identity();
	Vec3 sf_body = Vec3(0, 0, G); // 比力（加计真值），机体系
	bool grounded = true;
	explicit Body(const Params &p) : P(p) {}
	void reset(const Vec3 &spawn, double yaw) {
		pos = spawn; vel.setZero(); w.setZero();
		q = Quat(Eigen::AngleAxisd(yaw, Vec3::UnitZ()));
		grounded = pos.z() <= P.ground_z + 1e-9; if (grounded) pos.z() = P.ground_z;
		sf_body = Vec3(0, 0, G);
	}
	Vec3 vel_body() const { return q.conjugate() * vel; }
	void step(const Wrench &wr, double dt) { // wr 已含阻力，不含重力
		const Vec3 acc = (q * wr.force) / P.mass - Vec3(0, 0, G);
		const Vec3 vel_prev = vel;
		if (grounded && acc.z() <= 0) { // 推力不够离地：静止、水平，只保留偏航
			vel.setZero(); w.setZero(); pos.z() = P.ground_z;
			const double yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));
			q = Quat(Eigen::AngleAxisd(yaw, Vec3::UnitZ()));
		} else {
			grounded = false;
			const Vec3 &I = P.inertia;
			w += (wr.torque - w.cross(I.cwiseProduct(w))).cwiseQuotient(I) * dt; // 欧拉方程，半隐式
			const double wn = w.norm();
			if (wn > 0) q = (q * Quat(Eigen::AngleAxisd(wn * dt, w / wn))).normalized();
			vel += acc * dt;
			pos += vel * dt;
			if (pos.z() <= P.ground_z && vel.z() <= 0) { pos.z() = P.ground_z; vel.setZero(); w.setZero(); grounded = true; }
		}
		sf_body = q.conjugate() * ((vel - vel_prev) / dt + Vec3(0, 0, G));
	}
};
} // namespace sitl
