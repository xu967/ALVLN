// 场景 (3) 执行器级：一集里按顺序跑 arm_takeoff → 高度保持 → FORWARD 3 s → SAFE 停住 → YAW_LEFT 3 s → SAFE 10 s，
// 给出四类数字：READY 用时 / 高度稳态误差；FORWARD 稳态速度与到 90% 用时；停止距离与用时；YAW_LEFT 稳态角速度；SAFE 漂移。
// 速度与角速度都从真值 pos/quat/vel 算（quat 差分得偏航角速度；状态结构体没有 ω）。输出 <out>/actuator.csv。
#include "selftest_util.h"
#include "selftest_config.h"

#include <vector>

namespace {
constexpr int kYawWin = (int)(cfg::kYawRateWinS / cfg::kDt + 0.5);

struct Run {
	Csv csv;
	int phase = 0, k = 0;
	cf_sitl_state s{};
	double yaw_unw = 0, yaw_prev = 0;
	std::vector<double> yaw_buf = std::vector<double>(kYawWin + 1, 0.0);
	explicit Run(const std::string &path)
		: csv(path, "t,phase,action,exec_state,x,y,z,vx_b,vy_b,vz,roll_deg,pitch_deg,yaw_deg,yaw_rate,m0,m1,m2,m3,hover_thrust,throttle_stick,height_est,vel_est_x,vel_est_y,tof_m,battery_v") {
		s = get_state(); yaw_prev = yaw_rad(s.quat);
		for (double &y : yaw_buf) y = 0;
	}
	double yaw_rate() const {   // 过去 kYawRateWinS 的平均偏航角速度（真值差分）
		if (k < kYawWin) return 0;
		return (yaw_buf[k % (kYawWin + 1)] - yaw_buf[(k - kYawWin) % (kYawWin + 1)]) / (kYawWin * cfg::kDt);
	}
	void vbody(double out[3]) const { rot_inv(s.quat, s.vel, out); }
	bool tick() {
		if (!step_ms(1)) return false;
		s = get_state(); k++;
		const double y = yaw_rad(s.quat);
		yaw_unw += std::remainder(y - yaw_prev, 2 * M_PI); yaw_prev = y;
		yaw_buf[k % (kYawWin + 1)] = yaw_unw;
		if (k % cfg::kCsvEveryMs == 0) {
			double vb[3]; vbody(vb);
			const double R = 180 / M_PI;
			csv.row("%.3f,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.2f,%.4f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f",
			        s.t_s, phase, s.action, s.exec_state, s.pos[0], s.pos[1], s.pos[2], vb[0], vb[1], s.vel[2],
			        roll_rad(s.quat) * R, pitch_rad(s.quat) * R, yaw_rad(s.quat) * R, yaw_rate(),
			        s.motors[0], s.motors[1], s.motors[2], s.motors[3], s.hover_thrust, throttle_stick(s),
			        s.height_est_m, s.vel_est[0], s.vel_est[1], s.tof_m, s.battery_v);
		}
		return true;
	}
	bool run(double secs) {
		for (int i = 0, n = (int)std::lround(secs / cfg::kDt); i < n; i++) if (!tick()) return false;
		return true;
	}
};
}  // namespace

void scenario_actuator(Report &r, const Ctx &c) {
	r.line("== 3 执行器级（独立积分模式；速度/角速度由真值 pos/quat/vel 导出）");
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(4, spawn, 0);
	char what[256];

	// 3a. arm_takeoff(2.0) → READY 用时与高度稳态误差
	const double t_ready = takeoff_until_ready(cfg::kTargetHeight, cfg::kTakeoffMaxS);
	if (t_ready < 0) {
		const cf_sitl_state s = get_state();
		r.line("  起飞失败：t=%.2f s exec_state=%d armed=%d z=%.3f", s.t_s, s.exec_state, s.armed, s.pos[2]);
		r.check(false, "arm_takeoff 到 READY");
		return;
	}
	Run run(c.out + "/actuator.csv");
	run.phase = 1;
	double sum_dz = 0; int n = 0;
	for (int i = 0, m = (int)std::lround(cfg::kHeightHoldS / cfg::kDt); i < m; i++) {
		if (!run.tick()) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
		sum_dz += std::fabs(run.s.pos[2] - cfg::kTargetHeight); n++;
	}
	const double z_ss = sum_dz / n;
	r.line("  arm_takeoff(%.1f): READY 用时 %.2f s（上限 %.0f s）；READY 后 %.0f s 内平均 |z−%.1f| = %.3f m；悬停油门杆 %.3f",
	       cfg::kTargetHeight, t_ready, cfg::kTakeoffMaxS, cfg::kHeightHoldS, cfg::kTargetHeight, z_ss, run.s.hover_thrust);
	r.check(true, "arm_takeoff 到 READY（用时见上）");
	std::snprintf(what, sizeof what, "高度稳态误差 %.3f m < %.2f m", z_ss, cfg::kHeightSsTolM);
	r.check(z_ss < cfg::kHeightSsTolM, what);

	// 3b. FORWARD 3 s：稳态机体前向速度（末 0.5 s 均值）与到 90% 目标的用时
	run.phase = 2;
	const double t_fwd = run.s.t_s;
	cf_sitl_set_action(1, t_fwd + cfg::kForwardS);
	double t90 = -1, sum_v = 0, max_tilt = 0; n = 0;
	for (int i = 0, m = (int)std::lround(cfg::kForwardS / cfg::kDt); i < m; i++) {
		if (!run.tick()) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
		double vb[3]; run.vbody(vb);
		if (t90 < 0 && vb[0] >= 0.9 * cfg::kForwardTarget) t90 = run.s.t_s - t_fwd;
		if (run.s.t_s > t_fwd + cfg::kForwardS - 0.5) { sum_v += vb[0]; n++; }
		max_tilt = std::fmax(max_tilt, tilt_deg(run.s.quat));
	}
	const double v_ss = n ? sum_v / n : 0;
	double vb_end[3]; run.vbody(vb_end);
	r.line("  FORWARD %.0f s：末 0.5 s 平均前向速度 %.3f m/s（目标 %.1f），到 90%% 用时 %s s，侧向速度 %.3f m/s，最大倾角 %.2f°，exec_state=%d",
	       cfg::kForwardS, v_ss, cfg::kForwardTarget, fmt_or(t90).c_str(), vb_end[1], max_tilt, run.s.exec_state);
	std::snprintf(what, sizeof what, "FORWARD 稳态速度 %.3f 在 %.1f ± %.0f%% 内", v_ss, cfg::kForwardTarget, cfg::kSpeedTolFrac * 100);
	r.check(std::fabs(v_ss - cfg::kForwardTarget) <= cfg::kSpeedTolFrac * cfg::kForwardTarget, what);
	std::snprintf(what, sizeof what, "FORWARD 到 90%% 用时 %.2f s < %.1f s", t90, cfg::kRiseMaxS);
	r.check(t90 >= 0 && t90 < cfg::kRiseMaxS, what);

	// 3c. FORWARD 后 SAFE：停止距离与用时（|v_xy| 首次 < eps）
	run.phase = 3;
	const double t_safe = run.s.t_s, x0 = run.s.pos[0], y0 = run.s.pos[1];
	cf_sitl_set_action(0, t_safe + cfg::kStopMaxS + 1.0);
	double t_stop = -1, d_stop = -1;
	for (int i = 0, m = (int)std::lround(cfg::kStopMaxS / cfg::kDt); i < m; i++) {
		if (!run.tick()) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
		if (t_stop < 0 && hypot2(run.s.vel[0], run.s.vel[1]) < cfg::kStopSpeedEps) {
			t_stop = run.s.t_s - t_safe; d_stop = hypot2(run.s.pos[0] - x0, run.s.pos[1] - y0);
		}
	}
	const double d_end = hypot2(run.s.pos[0] - x0, run.s.pos[1] - y0);
	r.line("  SAFE 停止：|v_xy| 首次 < %.2f m/s 用时 %s s，此时距离 %s m；%.0f s 后距离 %.3f m、|v_xy| %.3f m/s",
	       cfg::kStopSpeedEps, fmt_or(t_stop, "未停").c_str(), fmt_or(d_stop, "-").c_str(), cfg::kStopMaxS, d_end, hypot2(run.s.vel[0], run.s.vel[1]));
	std::snprintf(what, sizeof what, "停止用时 %.2f s < %.1f s 且停止距离 %.3f m < %.1f m", t_stop, cfg::kStopMaxS, d_stop, cfg::kStopDistMaxM);
	r.check(t_stop >= 0 && d_stop < cfg::kStopDistMaxM, what);

	// 3d. YAW_LEFT 3 s：稳态偏航角速度（末 1 s 平均，绕 z 正向为正）与到 90% 用时
	run.phase = 4;
	const double t_yaw = run.s.t_s, yaw0 = run.yaw_unw;
	cf_sitl_set_action(2, t_yaw + cfg::kYawS);
	double t90y = -1, yaw_at_2s = 0, tilt_yaw = 0;
	for (int i = 0, m = (int)std::lround(cfg::kYawS / cfg::kDt); i < m; i++) {
		if (!run.tick()) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
		if (t90y < 0 && run.yaw_rate() >= 0.9 * cfg::kYawTarget) t90y = run.s.t_s - t_yaw;
		if (std::fabs(run.s.t_s - (t_yaw + cfg::kYawS - 1.0)) < 0.5 * cfg::kDt) yaw_at_2s = run.yaw_unw;
		tilt_yaw = std::fmax(tilt_yaw, tilt_deg(run.s.quat));
	}
	const double rate_ss = run.yaw_unw - yaw_at_2s;   // 末 1 s 转过的角度 = 平均角速度 rad/s
	r.line("  YAW_LEFT %.0f s：末 1 s 平均角速度 %+.3f rad/s（目标 %+.1f），到 90%% 用时 %s s，共转 %.1f°，期间最大倾角 %.2f°，z=%.3f",
	       cfg::kYawS, rate_ss, cfg::kYawTarget, fmt_or(t90y).c_str(),
	       (run.yaw_unw - yaw0) * 180 / M_PI, tilt_yaw, run.s.pos[2]);
	std::snprintf(what, sizeof what, "YAW_LEFT 稳态角速度 %+.3f 在 %+.1f ± %.0f%% 内（符号 = 绕 z 正向）", rate_ss, cfg::kYawTarget, cfg::kYawTolFrac * 100);
	r.check(std::fabs(rate_ss - cfg::kYawTarget) <= cfg::kYawTolFrac * cfg::kYawTarget, what);

	// 3e. SAFE 10 s：漂移
	run.phase = 5;
	const double t_s2 = run.s.t_s, xs = run.s.pos[0], ys = run.s.pos[1], zs = run.s.pos[2];
	cf_sitl_set_action(0, t_s2 + cfg::kSafeS);
	double max_r = 0, max_dz = 0;
	for (int i = 0, m = (int)std::lround(cfg::kSafeS / cfg::kDt); i < m; i++) {
		if (!run.tick()) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
		max_r = std::fmax(max_r, hypot2(run.s.pos[0] - xs, run.s.pos[1] - ys));
		max_dz = std::fmax(max_dz, std::fabs(run.s.pos[2] - cfg::kTargetHeight));
	}
	const double drift = hypot2(run.s.pos[0] - xs, run.s.pos[1] - ys);
	r.line("  SAFE %.0f s：水平漂移 %.3f m（最大 %.3f m），高度 %.3f → %.3f m（最大偏差 %.3f m），exec_state=%d，电压 %.3f V",
	       cfg::kSafeS, drift, max_r, zs, run.s.pos[2], max_dz, run.s.exec_state, run.s.battery_v);
	std::snprintf(what, sizeof what, "SAFE %.0f s 漂移 %.3f m < %.1f m", cfg::kSafeS, drift, cfg::kSafeDriftM);
	r.check(drift < cfg::kSafeDriftM, what);
}
