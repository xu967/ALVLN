// selftest_util.h：自检公用件：报告（stdout + TXT 同步、PASS/FAIL/SKIP 计数）、CSV 写出、
// 四元数几何、状态哈希、对 cf_sitl 的薄封装。场景实现见 scenario_*.cpp。
#pragma once
#include "cf_sitl.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <string>
#include <type_traits>
#include <vector>

struct Ctx {
	std::string params, out, exe;  // 参数文件、输出目录、本程序路径（确定性场景要再起两个进程）
	uint64_t seed = 1;
};

// 报告：每行同时写 stdout 与 <out>/selftest.txt；check 统计失败数（= 进程退出码）
struct Report {
	FILE *txt = nullptr;
	int fails = 0, passes = 0, skips = 0;
	explicit Report(const std::string &path) { txt = std::fopen(path.c_str(), "w"); }
	~Report() { if (txt) std::fclose(txt); }
	void line(const char *fmt, ...) {
		char b[1024];
		va_list ap; va_start(ap, fmt); std::vsnprintf(b, sizeof b, fmt, ap); va_end(ap);
		std::printf("%s\n", b); std::fflush(stdout);
		if (txt) { std::fprintf(txt, "%s\n", b); std::fflush(txt); }
	}
	void check(bool ok, const char *what) { line("  [%s] %s", ok ? "PASS" : "FAIL", what); ok ? passes++ : fails++; }
	void skip(const char *what) { line("  [SKIP] %s", what); skips++; }
};

// CSV：表头一次，之后每行 printf 风格
struct Csv {
	FILE *f = nullptr;
	Csv(const std::string &path, const char *header) {
		f = std::fopen(path.c_str(), "w");
		if (f) std::fprintf(f, "%s\n", header);
	}
	~Csv() { if (f) std::fclose(f); }
	void row(const char *fmt, ...) {
		if (!f) return;
		va_list ap; va_start(ap, fmt); std::vfprintf(f, fmt, ap); va_end(ap);
		std::fputc('\n', f);
	}
};

// ---- 四元数 (w,x,y,z)，机体→世界 ----
inline double tilt_deg(const double q[4]) {          // 机体 z 与世界 z 的夹角
	const double cz = 1 - 2 * (q[1] * q[1] + q[2] * q[2]);
	return std::acos(std::fmax(-1.0, std::fmin(1.0, cz))) * 180 / M_PI;
}
inline double yaw_rad(const double q[4]) {
	return std::atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
}
inline double roll_rad(const double q[4]) {
	return std::atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
}
inline double pitch_rad(const double q[4]) {
	return std::asin(std::fmax(-1.0, std::fmin(1.0, 2 * (q[0] * q[2] - q[3] * q[1]))));
}
inline double quat_angle_deg(const double a[4], const double b[4]) { // 两姿态之间的转角
	double d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
	const double na = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
	const double nb = std::sqrt(b[0] * b[0] + b[1] * b[1] + b[2] * b[2] + b[3] * b[3]);
	if (na < 1e-9 || nb < 1e-9) return 180.0;    // 未初始化的四元数当最大误差
	d = std::fabs(d / (na * nb));
	return 2 * std::acos(std::fmin(1.0, d)) * 180 / M_PI;
}
inline void rot_inv(const double q[4], const double v[3], double out[3]) { // 世界→机体
	const double w = q[0], x = q[1], y = q[2], z = q[3];
	out[0] = (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y + w * z) * v[1] + 2 * (x * z - w * y) * v[2];
	out[1] = 2 * (x * y - w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z + w * x) * v[2];
	out[2] = 2 * (x * z + w * y) * v[0] + 2 * (y * z - w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2];
}
inline void quat_mul(const double a[4], const double b[4], double o[4]) {
	o[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
	o[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
	o[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
	o[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}
inline void quat_from_axis(const double axis[3], double angle, double o[4]) { // 轴应为单位向量
	const double s = std::sin(angle / 2);
	o[0] = std::cos(angle / 2); o[1] = axis[0] * s; o[2] = axis[1] * s; o[3] = axis[2] * s;
}
inline void quat_normalize(double q[4]) {
	const double n = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
	for (int i = 0; i < 4; i++) q[i] /= n;
}
inline double hypot2(double x, double y) { return std::sqrt(x * x + y * y); }
inline std::string fmt_or(double v, const char *none = "未到") { // 负值表示「没发生」
	if (v < 0) return none;
	char b[32]; std::snprintf(b, sizeof b, "%.2f", v); return b;
}

// ---- cf_sitl 薄封装 ----
inline cf_sitl_state get_state() { cf_sitl_state s; std::memset(&s, 0, sizeof s); cf_sitl_get_state(&s); return s; }
inline bool step_ms(int n) { return cf_sitl_step_standalone(n) == 0; }

// cf_sitl_state 末尾可能追加 sticks[4]（core 计划中）；有就读，没有返回 NaN，两个版本的头文件都能编
template <class T, class = void> struct has_sticks : std::false_type {};
template <class T> struct has_sticks<T, std::void_t<decltype(std::declval<T>().sticks)>> : std::true_type {};
template <class S> double throttle_stick(const S &s) {
	if constexpr (has_sticks<S>::value) return s.sticks[3]; else { (void)s; return NAN; }
}

// FNV-1a，逐字段喂（不喂结构体原始字节，避免填充位）
inline uint64_t fnv(uint64_t h, const void *p, size_t n) {
	const unsigned char *b = (const unsigned char *)p;
	for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
	return h;
}
inline uint64_t hash_state(uint64_t h, const cf_sitl_state &s) {
	h = fnv(h, &s.t_s, sizeof s.t_s); h = fnv(h, &s.exec_state, sizeof s.exec_state);
	h = fnv(h, &s.fw_mode, sizeof s.fw_mode); h = fnv(h, &s.armed, sizeof s.armed);
	h = fnv(h, &s.action, sizeof s.action); h = fnv(h, &s.action_until_s, sizeof s.action_until_s);
	h = fnv(h, s.motors, sizeof s.motors); h = fnv(h, &s.battery_v, sizeof s.battery_v);
	h = fnv(h, s.att_est, sizeof s.att_est); h = fnv(h, &s.tof_m, sizeof s.tof_m);
	h = fnv(h, s.flow_vel, sizeof s.flow_vel); h = fnv(h, &s.height_est_m, sizeof s.height_est_m);
	h = fnv(h, s.vel_est, sizeof s.vel_est); h = fnv(h, &s.hover_thrust, sizeof s.hover_thrust);
	h = fnv(h, s.pos, sizeof s.pos); h = fnv(h, s.quat, sizeof s.quat); h = fnv(h, s.vel, sizeof s.vel);
	return h;
}

// 起飞到 READY；返回用时（s），超时返回 -1。每 10 ms 查一次状态。
inline double takeoff_until_ready(double height, double max_s) {
	const double t0 = get_state().t_s;
	cf_sitl_arm_takeoff(height);
	for (;;) {
		const cf_sitl_state s = get_state();
		if (s.exec_state == CF_SITL_READY) return s.t_s - t0;
		if (s.t_s - t0 >= max_s - 1e-9) return -1;
		if (!step_ms(10)) return -1;
	}
}

// 阶跃响应指标：y 为 1 ms 采样的响应，y0 为阶跃前的值；rise = 10%→90% 用时，overshoot 相对 (target−y0) 的百分比，
// ss = 末 ss_win_s 的均值。t90 < 0 表示没到 90%。
struct StepMetrics { double t10 = -1, t90 = -1, rise = -1, overshoot_pct = 0, ss = 0; };
inline StepMetrics step_metrics(const std::vector<double> &y, double y0, double target, double dt, double ss_win_s) {
	StepMetrics m; double peak = 0;
	for (size_t i = 0; i < y.size(); i++) {
		const double yn = (y[i] - y0) / (target - y0);
		if (m.t10 < 0 && yn >= 0.1) m.t10 = i * dt;
		if (m.t90 < 0 && yn >= 0.9) m.t90 = i * dt;
		peak = std::fmax(peak, yn);
	}
	if (m.t10 >= 0 && m.t90 >= 0) m.rise = m.t90 - m.t10;
	m.overshoot_pct = 100 * (peak - 1);
	const size_t n = std::min(y.size(), (size_t)std::lround(ss_win_s / dt));
	double s = 0; for (size_t i = y.size() - n; i < y.size(); i++) s += y[i];
	m.ss = n ? s / n : NAN;
	return m;
}

// 场景入口（各 scenario_*.cpp 实现）
void scenario_frame(Report &r, const Ctx &c);
void scenario_hover(Report &r, const Ctx &c);
void scenario_actuator(Report &r, const Ctx &c);
void scenario_sticks(Report &r, const Ctx &c);
void scenario_determinism(Report &r, const Ctx &c);
uint64_t determinism_hash_run(const Ctx &c);   // --hash-run 子进程用：跑固定脚本，返回哈希
void scenario_battery(Report &r, const Ctx &c);
