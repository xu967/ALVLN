// 场景 (3.5) 固件级打杆阶跃：READY 后用 cf_sitl_set_sticks_raw 旁路执行器，直接给固件 STAB 杆量
// （油门每 tick 取执行器当前的悬停杆 hover_thrust，执行器在旁路期间照常运行）。
//   俯仰 +0.5 → 固件估计俯仰稳态 15° ± 5%（control.ino：倾角 = 杆量 × 30°），松杆 2 s 内估计 |roll|,|pitch| < 1°；
//   偏航 +0.5 → 真值偏航角速度稳态 −150°/s ± 5%（杆量 × 300°/s；固件正杆 = 顺时针 = 绕 z 负向）。
// 判据用固件自己的姿态估计（它控制的就是估计值），真值与估计的差单独打印；记上升时间（10→90%）与超调。
// 两段之间取消旁路，等执行器刹住滑行、回到目标高度稳住再做偏航阶跃。输出 <out>/steps.csv。
#include "selftest_util.h"
#include "selftest_config.h"

namespace {
constexpr int kYawWin = (int)(cfg::kYawRateWinS / cfg::kDt + 0.5);
constexpr double R2D = 180 / M_PI;

struct Run {
	Csv csv;
	int phase = 0, k = 0;
	bool raw = false; double sticks[3] = {0, 0, 0};   // roll, pitch, yaw；raw=false 时不碰固件杆量
	cf_sitl_state s{};
	double yaw_unw = 0, yaw_prev = 0;
	std::vector<double> yaw_buf = std::vector<double>(kYawWin + 1, 0.0);
	explicit Run(const std::string &path)
		: csv(path, "t,phase,raw,stick_pitch,stick_yaw,throttle,roll_deg,pitch_deg,est_roll_deg,est_pitch_deg,yaw_deg,yaw_rate_dps,tilt_deg,z,vz,vx_b,vy_b,hover_thrust,exec_state,armed,fw_mode") {
		s = get_state(); yaw_prev = yaw_rad(s.quat);
	}
	double yaw_rate_dps() const {
		if (k < kYawWin) return 0;
		return (yaw_buf[k % (kYawWin + 1)] - yaw_buf[(k - kYawWin) % (kYawWin + 1)]) / (kYawWin * cfg::kDt) * R2D;
	}
	double est_pitch() const { return pitch_rad(s.att_est) * R2D; }
	double est_roll() const { return roll_rad(s.att_est) * R2D; }
	void set_raw(double roll, double pitch, double yaw) { raw = true; sticks[0] = roll; sticks[1] = pitch; sticks[2] = yaw; }
	void clear_raw() { raw = false; cf_sitl_set_sticks_raw(NAN, 0, 0, 0); }
	bool tick() {
		if (raw) cf_sitl_set_sticks_raw(sticks[0], sticks[1], sticks[2], s.hover_thrust);   // 油门 = 执行器上一 tick 的悬停杆
		if (!step_ms(1)) return false;
		s = get_state(); k++;
		const double y = yaw_rad(s.quat);
		yaw_unw += std::remainder(y - yaw_prev, 2 * M_PI); yaw_prev = y;
		yaw_buf[k % (kYawWin + 1)] = yaw_unw;
		if (k % cfg::kStepsCsvEveryMs == 0) {
			double vb[3]; rot_inv(s.quat, s.vel, vb);
			csv.row("%.3f,%d,%d,%.2f,%.2f,%.4f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.4f,%.3f,%.3f,%.3f,%.4f,%d,%d,%d",
			        s.t_s, phase, raw ? 1 : 0, raw ? sticks[1] : NAN, raw ? sticks[2] : NAN, raw ? s.hover_thrust : throttle_stick(s),
			        roll_rad(s.quat) * R2D, pitch_rad(s.quat) * R2D, est_roll(), est_pitch(), y * R2D, yaw_rate_dps(), tilt_deg(s.quat),
			        s.pos[2], s.vel[2], vb[0], vb[1], s.hover_thrust, s.exec_state, s.armed, s.fw_mode);
		}
		return true;
	}
	bool run(double secs, std::vector<double> *a = nullptr, double (*fa)(const Run &) = nullptr,
	         std::vector<double> *b = nullptr, double (*fb)(const Run &) = nullptr) {
		for (int i = 0, n = (int)std::lround(secs / cfg::kDt); i < n; i++) {
			if (!tick()) return false;
			if (a && fa) a->push_back(fa(*this));
			if (b && fb) b->push_back(fb(*this));
		}
		return true;
	}
	// 等执行器把高度稳住；返回用时，超时返回 -1
	double wait_settled(double max_s) {
		const double t0 = s.t_s; double since = -1;
		for (int i = 0, n = (int)std::lround(max_s / cfg::kDt); i < n; i++) {
			if (!tick()) return -1;
			const bool ok = std::fabs(s.pos[2] - cfg::kTargetHeight) < cfg::kSettleZM && std::fabs(s.vel[2]) < cfg::kSettleVzMps;
			if (!ok) since = -1; else if (since < 0) since = s.t_s;
			if (since >= 0 && s.t_s - since >= cfg::kSettleHoldS) return s.t_s - t0;
		}
		return -1;
	}
};
double true_pitch_of(const Run &r) { return pitch_rad(r.s.quat) * R2D; }
double est_pitch_of(const Run &r) { return r.est_pitch(); }
double yaw_rate_of(const Run &r) { return r.yaw_rate_dps(); }
}  // namespace

void scenario_sticks(Report &r, const Ctx &c) {
	r.line("== 3.5 固件级打杆阶跃（cf_sitl_set_sticks_raw 旁路执行器；油门 = 执行器悬停杆；判据用固件姿态估计）");
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(6, spawn, 0);
	const double t_ready = takeoff_until_ready(cfg::kTargetHeight, cfg::kTakeoffMaxS);
	if (t_ready < 0) { r.check(false, "arm_takeoff 到 READY"); return; }
	Run run(c.out + "/steps.csv");
	char what[256];
	auto bail = [&] { r.check(false, "cf_sitl_step_standalone 可用"); run.clear_raw(); };
	const double tgt = cfg::kStickPitchTargetDeg, tol = cfg::kStickTolFrac;

	// 俯仰阶跃 +0.5 → 15°（判据：固件估计；真值与估计差单独打印）
	if (!run.run(cfg::kStickPreHoldS)) return bail();
	const double z0 = run.s.pos[2], est0 = run.est_pitch(), tru0 = true_pitch_of(run);
	run.phase = 1; run.set_raw(0, cfg::kStickStep, 0);
	std::vector<double> est, tru;
	if (!run.run(cfg::kStickHoldS, &est, est_pitch_of, &tru, true_pitch_of)) return bail();
	const double sgn = est.back() < 0 ? -1 : 1;   // 判幅值，符号单独打印
	const StepMetrics em = step_metrics(est, est0, sgn * tgt, cfg::kDt, cfg::kStickSsWinS);
	const StepMetrics tm = step_metrics(tru, tru0, sgn * tgt, cfg::kDt, cfg::kStickSsWinS);
	double vb[3]; rot_inv(run.s.quat, run.s.vel, vb);
	r.line("  俯仰杆 +%.1f 保持 %.1f s：固件估计俯仰稳态 %+.2f°（目标 ±%.0f°），上升 10→90%% %s s，超调 %.1f%%；真值俯仰稳态 %+.2f°（估计误差 %+.2f°，上升 %s s）",
	       cfg::kStickStep, cfg::kStickHoldS, em.ss, tgt, fmt_or(em.rise).c_str(), em.overshoot_pct, tm.ss, tm.ss - em.ss, fmt_or(tm.rise).c_str());
	r.line("  掉高 %.3f m，前向速度 %.2f m/s，倾角 %.2f°，油门 %.3f", z0 - run.s.pos[2], vb[0], tilt_deg(run.s.quat), run.s.hover_thrust);
	std::snprintf(what, sizeof what, "俯仰阶跃稳态（固件估计）|%.2f°| 在 %.0f° ± %.0f%% 内", em.ss, tgt, tol * 100);
	r.check(std::fabs(std::fabs(em.ss) - tgt) <= tol * tgt, what);

	// 松杆：raw 中立杆，估计 |roll|,|pitch| 最后一次 > 1° 的时刻
	run.phase = 2; run.set_raw(0, 0, 0);
	const double t_rel = run.s.t_s;
	double last_off = t_rel, last_off_true = t_rel, emin = 0, emax = 0;
	for (int i = 0, n = (int)std::lround(cfg::kStickReleaseS / cfg::kDt); i < n; i++) {
		if (!run.tick()) return bail();
		const double p = run.est_pitch(), ro = run.est_roll();
		if (std::fabs(p) > cfg::kLevelDeg || std::fabs(ro) > cfg::kLevelDeg) last_off = run.s.t_s;
		if (std::fabs(true_pitch_of(run)) > cfg::kLevelDeg || std::fabs(roll_rad(run.s.quat) * R2D) > cfg::kLevelDeg) last_off_true = run.s.t_s;
		emin = std::fmin(emin, p); emax = std::fmax(emax, p);
	}
	const double settle = last_off - t_rel;
	rot_inv(run.s.quat, run.s.vel, vb);
	r.line("  松杆 %.0f s：固件估计 %.3f s 后 |roll|,|pitch| 一直 < %.0f°（回摆范围 [%.2f°, %.2f°]）；真值 %s s 后 < %.0f°，末态真值 roll %+.2f° pitch %+.2f°；速度 %.2f m/s，z=%.3f",
	       cfg::kStickReleaseS, settle, cfg::kLevelDeg, emin, emax, last_off_true - t_rel >= cfg::kStickReleaseS - 1e-6 ? "未" : fmt_or(last_off_true - t_rel).c_str(),
	       cfg::kLevelDeg, roll_rad(run.s.quat) * R2D, true_pitch_of(run), vb[0], run.s.pos[2]);
	std::snprintf(what, sizeof what, "松杆 %.3f s 内回平（固件估计，< %.0f s）", settle, cfg::kStickReleaseS);
	r.check(settle < cfg::kStickReleaseS - 1e-9, what);

	// 取消旁路：执行器接管，刹住滑行、补回高度并稳住
	run.phase = 3; run.clear_raw();
	const double t_rec = run.wait_settled(cfg::kRecoverMaxS);
	rot_inv(run.s.quat, run.s.vel, vb);
	r.line("  取消旁路后执行器接管：%s s 稳住（|z−%.1f| < %.2f m、|vz| < %.2f m/s 持续 %.1f s）；此时 |v_xy| %.3f m/s，z=%.3f，悬停杆 %.3f，exec_state=%d",
	       fmt_or(t_rec).c_str(), cfg::kTargetHeight, cfg::kSettleZM, cfg::kSettleVzMps, cfg::kSettleHoldS, hypot2(vb[0], vb[1]), run.s.pos[2], run.s.hover_thrust, run.s.exec_state);
	std::snprintf(what, sizeof what, "执行器在 %.0f s 内刹住滑行并回到目标高度稳住（用时 %s s）", cfg::kRecoverMaxS, fmt_or(t_rec).c_str());
	r.check(t_rec >= 0 && run.s.exec_state == CF_SITL_READY, what);

	// 偏航阶跃 +0.5 → −150°/s（真值角速度：固件速率环用陀螺，零偏 ~0.1°/s 相对 150°/s 可忽略）
	run.phase = 4; run.set_raw(0, 0, cfg::kStickStep);
	const double r0 = run.yaw_rate_dps(), yaw_a = run.yaw_unw;
	std::vector<double> rate;
	if (!run.run(cfg::kStickHoldS, &rate, yaw_rate_of)) return bail();
	const StepMetrics ym = step_metrics(rate, r0, cfg::kStickYawTargetDps, cfg::kDt, cfg::kStickSsWinS);
	r.line("  偏航杆 +%.1f 保持 %.1f s：真值偏航角速度稳态 %+.1f°/s（目标 %+.0f），上升 10→90%% %s s（含 %.0f ms 差分窗口），超调 %.1f%%，共转 %.1f°，倾角 %.2f°，z=%.3f",
	       cfg::kStickStep, cfg::kStickHoldS, ym.ss, cfg::kStickYawTargetDps, fmt_or(ym.rise).c_str(), cfg::kYawRateWinS * 1e3, ym.overshoot_pct,
	       (run.yaw_unw - yaw_a) * R2D, tilt_deg(run.s.quat), run.s.pos[2]);
	std::snprintf(what, sizeof what, "偏航阶跃稳态 %+.1f°/s 在 %+.0f ± %.0f%% 内（正杆 = 顺时针）", ym.ss, cfg::kStickYawTargetDps, tol * 100);
	r.check(std::fabs(ym.ss - cfg::kStickYawTargetDps) <= tol * std::fabs(cfg::kStickYawTargetDps), what);

	// 松偏航杆：|r| 最后一次 > 5°/s；再取消旁路看执行器接住
	run.phase = 5; run.set_raw(0, 0, 0);
	const double t_yrel = run.s.t_s, yaw_b = run.yaw_unw;
	double last_turn = t_yrel;
	for (int i = 0, n = (int)std::lround(cfg::kYawReleaseS / cfg::kDt); i < n; i++) {
		if (!run.tick()) return bail();
		if (std::fabs(run.yaw_rate_dps()) > cfg::kYawStopDps) last_turn = run.s.t_s;
	}
	r.line("  松偏航杆 %.0f s：%.3f s 后 |r| 一直 < %.0f°/s，多转 %.1f°", cfg::kYawReleaseS, last_turn - t_yrel, cfg::kYawStopDps, (run.yaw_unw - yaw_b) * R2D);
	run.phase = 6; run.clear_raw();
	if (!run.run(1.0)) return bail();
	std::snprintf(what, sizeof what, "全程取消旁路后执行器都接住（末态 exec_state=%d，armed=%d，z=%.3f）", run.s.exec_state, run.s.armed, run.s.pos[2]);
	r.check((run.s.exec_state == CF_SITL_READY || run.s.exec_state == CF_SITL_ACTING) && run.s.armed, what);
}
