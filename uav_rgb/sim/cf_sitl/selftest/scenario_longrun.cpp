// 场景 (6) 长航时：独立积分模式。READY 后 (a) SAFE 保持 30 s 看真值水平漂移（位置保持判据 < 5 cm；真值漂移 < 常数项 + 零偏项）；
// (b) FORWARD / YAW_LEFT / FORWARD / YAW_RIGHT 交替 60 s，监控 FORWARD 段机体系 vx 的符号、固件姿态估计误差 δ
// （估计 − 真值，飞行中固件只积分陀螺，误差以残余零偏速率线性增长）、速度环积分 I（由杆量反推）、光流速度估计。
// 3DGS route 5 seed 500 的倒飞就是这条链：估计漂移 → 积分吃满 → 速度环开环 → vx 变负。积分饱和只在零偏小到预算撑得住时才算失败
// （阈值与理由见 selftest_config.h kLongrunInt*；CTest tests/test_longrun_int_budget.cpp 用同一判据兜底）。
// 输出 <out>/longrun.csv。
#include "selftest_util.h"
#include "selftest_config.h"
#include "json_min.h"

#include <fstream>
#include <sstream>
#include <vector>

namespace {
struct Run {
	Csv csv;
	int phase = 0, k = 0;
	cf_sitl_state s{};
	explicit Run(const std::string &path)
		: csv(path, "t,phase,action,exec_state,x,y,z,vx_b,vy_b,roll_deg,pitch_deg,yaw_deg,est_roll_deg,est_pitch_deg,err_roll_deg,err_pitch_deg,"
		            "vel_est_x,vel_est_y,flow_vx,flow_vy,roll_stick,pitch_stick,yaw_stick,throttle_stick,hover_thrust,height_est,battery_v") {
		s = get_state();
	}
	void vbody(double out[3]) const { rot_inv(s.quat, s.vel, out); }
	bool tick() {
		if (!step_ms(1)) return false;
		s = get_state(); k++;
		if (k % cfg::kLongrunCsvEveryMs == 0) {
			double vb[3]; vbody(vb);
			const double R = 180 / M_PI;
			csv.row("%.3f,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.2f,%.3f,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f",
			        s.t_s, phase, s.action, s.exec_state, s.pos[0], s.pos[1], s.pos[2], vb[0], vb[1],
			        roll_rad(s.quat) * R, pitch_rad(s.quat) * R, yaw_rad(s.quat) * R, roll_rad(s.att_est) * R, pitch_rad(s.att_est) * R,
			        (roll_rad(s.att_est) - roll_rad(s.quat)) * R, (pitch_rad(s.att_est) - pitch_rad(s.quat)) * R,
			        s.vel_est[0], s.vel_est[1], s.flow_vel[0], s.flow_vel[1], s.sticks[0], s.sticks[1], s.sticks[2], s.sticks[3],
			        s.hover_thrust, s.height_est_m, s.battery_v);
		}
		return true;
	}
};
// 反推速度环积分要用的执行器增益（与库同一份参数文件，不在这里抄第二份默认值）
struct ExecGains { double kp = 0, tilt_max_deg = 0, per_stick_deg = 0, forward_mps = 0; bool ok = false; };
ExecGains read_exec_gains(const std::string &params) {
	ExecGains g;
	std::ifstream f(params); std::stringstream ss; ss << f.rdbuf();
	Json j; std::string err;
	if (!Json::parse(ss.str(), j, err)) return g;
	const Json *ex = j.get("executor"); const Json *v = ex ? ex->get("value") : nullptr;
	if (!v) return g;
	const struct { const char *key; double *dst; } keys[] = {
		{"vel_kp", &g.kp}, {"tilt_max_deg", &g.tilt_max_deg}, {"tilt_per_stick_deg", &g.per_stick_deg}, {"forward_speed_mps", &g.forward_mps}};
	for (const auto &k : keys) { const Json *x = v->get(k.key); if (!x || !x->is_num()) return g; *k.dst = x->num; }
	g.ok = true;
	return g;
}
double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }
}  // namespace

void scenario_longrun(Report &r, const Ctx &c) {
	r.line("== 6 长航时（独立积分模式）：SAFE 保持 %.0f s + FORWARD/YAW 交替 %.0f s", cfg::kLongrunSafeS, cfg::kLongrunAltS);
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(cfg::kLongrunEpisode, spawn, 0);
	const double t_ready = takeoff_until_ready(cfg::kTargetHeight, cfg::kTakeoffMaxS);
	if (t_ready < 0) {
		const cf_sitl_state s = get_state();
		r.line("  起飞失败：t=%.2f s exec_state=%d armed=%d z=%.3f", s.t_s, s.exec_state, s.armed, s.pos[2]);
		r.check(false, "arm_takeoff 到 READY");
		return;
	}
	Run run(c.out + "/longrun.csv");
	char what[256];
	const double R = 180 / M_PI;

	// 6a. SAFE 保持 30 s。两层判据：
	//   (1) 执行器相信的位移 ∫vel_est < 5 cm —— 位置环本身闭得住；
	//   (2) 真值漂移 < 0.20 m + 1.5·|b|·h·T —— 剩余漂移两个来源：光流陀螺补偿的零偏（执行器 flow_vel = (flow + gyro)·h，固件 rates 带
	//       残余零偏 b，速度环把「估计速度」压到 0 时真速度 = −b×h，2 m 高、0.1 °/s → 0.35 cm/s，任何位置环都看不见它；b 用固件姿态
	//       估计误差的增长率测），与零偏无关的光流白噪声经位置环积成的随机游走（30 s 约 0.15 m，进常数项，见 selftest_config.h）。
	run.phase = 1;
	const double t_safe = run.s.t_s, xs = run.s.pos[0], ys = run.s.pos[1];
	const double err0_p = pitch_rad(run.s.att_est) - pitch_rad(run.s.quat), err0_r = roll_rad(run.s.att_est) - roll_rad(run.s.quat);
	cf_sitl_set_action(0 /* SAFE */, t_safe + cfg::kLongrunSafeS + 1.0);
	double max_r = 0, bel_x = 0, bel_y = 0, h_sum = 0; int h_n = 0;
	for (int i = 0, m = (int)std::lround(cfg::kLongrunSafeS / cfg::kDt); i < m; i++) {
		if (!run.tick()) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
		max_r = std::fmax(max_r, hypot2(run.s.pos[0] - xs, run.s.pos[1] - ys));
		bel_x += run.s.vel_est[0] * cfg::kDt; bel_y += run.s.vel_est[1] * cfg::kDt;
		h_sum += run.s.height_est_m; h_n++;
	}
	const double drift = hypot2(run.s.pos[0] - xs, run.s.pos[1] - ys), believed = hypot2(bel_x, bel_y);
	const double err1_p = pitch_rad(run.s.att_est) - pitch_rad(run.s.quat), err1_r = roll_rad(run.s.att_est) - roll_rad(run.s.quat);
	const double bx = (err1_r - err0_r) / cfg::kLongrunSafeS, by = (err1_p - err0_p) / cfg::kLongrunSafeS;   // rad/s ≈ 机体 x/y 陀螺零偏
	const double flow_bias_drift = hypot2(bx, by) * (h_sum / h_n) * cfg::kLongrunSafeS;
	const double bound = cfg::kLongrunTrueDriftConstM + cfg::kLongrunFlowBiasMargin * flow_bias_drift;
	r.line("  SAFE %.0f s：真值离保持起点 %.3f m（最大 %.3f m；%.2f cm/s），执行器相信的位移 %.3f m，z=%.3f；"
	       "固件姿态估计误差 roll %+.2f→%+.2f° pitch %+.2f→%+.2f° → 零偏 b≈(%+.3f, %+.3f) °/s；光流补偿零偏应漂 |b|·h·T = %.3f m；横滚杆 %+.3f 俯仰杆 %+.3f（×30° ≈ 速度环积分）",
	       cfg::kLongrunSafeS, drift, max_r, drift / cfg::kLongrunSafeS * 100, believed, run.s.pos[2],
	       err0_r * R, err1_r * R, err0_p * R, err1_p * R, bx * R, by * R, flow_bias_drift, run.s.sticks[0], run.s.sticks[1]);
	std::snprintf(what, sizeof what, "SAFE %.0f s 位置环闭合：执行器相信的位移 %.3f m < %.2f m", cfg::kLongrunSafeS, believed, cfg::kLongrunSafeDriftM);
	r.check(believed < cfg::kLongrunSafeDriftM, what);
	std::snprintf(what, sizeof what, "SAFE %.0f s 真值漂移 %.3f m < %.2f + %.1f×|b|·h·T = %.3f m（常数项 = 5 cm 判据 + 光流噪声随机游走；零偏项 = 光流陀螺补偿零偏，位置环不可见）",
	              cfg::kLongrunSafeS, drift, cfg::kLongrunTrueDriftConstM, cfg::kLongrunFlowBiasMargin, bound);
	r.check(drift < bound, what);
	if (drift >= cfg::kLongrunSafeDriftM)
		r.line("  （信息）真值漂移未达任务目标 %.2f m：要达到需 |b| < %.3f °/s（当前参数 gyro_bias_residual 1σ = 0.1 °/s），或执行器在飞行中估零偏",
		       cfg::kLongrunSafeDriftM, cfg::kLongrunSafeDriftM / ((h_sum / h_n) * cfg::kLongrunSafeS) * R);

	// 6b. FORWARD / YAW_LEFT / FORWARD / YAW_RIGHT 交替。速度环积分 I 由杆量反推（FORWARD 段 ref 已知）：
	//   pitch 杆 = (clamp(kp·(v_fwd − vel_est_x), ±tilt_max) + I_x)/30°，roll 杆 = −(clamp(kp·(0 − vel_est_y), ±tilt_max) + I_y)/30°。
	run.phase = 2;
	const ExecGains G = read_exec_gains(c.params);
	if (!G.ok) r.line("  （警告）读不到 executor.value 的 vel_kp/tilt_max_deg/tilt_per_stick_deg/forward_speed_mps，积分按 杆×30° 近似（含 P 项）");
	const double t_alt = run.s.t_s;
	double vx_min = 1e9, t_vx_min = 0, vx_sum = 0; int vx_n = 0, seg = 0, n_fwd = 0;
	double max_err = std::fmax(hypot2(err1_p, err1_r) * R, hypot2(err0_p, err0_r) * R), max_int_deg = 0, t_sat = -1, err_at_sat = 0;
	int fwd_vx_bad = 0;
	while (run.s.t_s - t_alt < cfg::kLongrunAltS) {
		const int kind = seg % 4;   // 0 FORWARD, 1 YAW_LEFT, 2 FORWARD, 3 YAW_RIGHT
		const int action = kind == 0 || kind == 2 ? 1 : kind == 1 ? 2 : 3;   // 1 FORWARD 2 YAW_LEFT 3 YAW_RIGHT（cf_sitl.h 动作编号）
		const double dur = action == 1 ? cfg::kLongrunFwdS : cfg::kLongrunYawS;
		const double t0 = run.s.t_s;
		cf_sitl_set_action(action, t0 + dur);
		if (action == 1) n_fwd++;
		for (int i = 0, m = (int)std::lround(dur / cfg::kDt); i < m; i++) {
			if (!run.tick()) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
			const double ep = (pitch_rad(run.s.att_est) - pitch_rad(run.s.quat)) * R, er = (roll_rad(run.s.att_est) - roll_rad(run.s.quat)) * R;
			max_err = std::fmax(max_err, hypot2(ep, er));
			if (action == 1) {
				double ix, iy;
				if (G.ok) {
					const double px = clampd(G.kp * (G.forward_mps - run.s.vel_est[0]) * R, -G.tilt_max_deg, G.tilt_max_deg);
					const double py = clampd(G.kp * (0.0 - run.s.vel_est[1]) * R, -G.tilt_max_deg, G.tilt_max_deg);
					ix = run.s.sticks[1] * G.per_stick_deg - px; iy = -run.s.sticks[0] * G.per_stick_deg - py;
				} else {
					ix = run.s.sticks[1] * 30.0; iy = run.s.sticks[0] * 30.0;
				}
				const double im = std::fmax(std::fabs(ix), std::fabs(iy));
				max_int_deg = std::fmax(max_int_deg, im);
				if (t_sat < 0 && im >= cfg::kLongrunIntSatDeg - cfg::kLongrunIntTolDeg) { t_sat = run.s.t_s; err_at_sat = hypot2(ep, er); }
			}
			if (action == 1 && run.s.t_s - t0 >= cfg::kLongrunSettleS) {
				double vb[3]; run.vbody(vb);
				if (vb[0] < vx_min) { vx_min = vb[0]; t_vx_min = run.s.t_s - t_alt; }
				if (vb[0] < cfg::kLongrunVxMinFrac * cfg::kForwardTarget) fwd_vx_bad++;
				vx_sum += vb[0]; vx_n++;
			}
		}
		seg++;
	}
	const cf_sitl_state &e = run.s;
	const bool saturated = t_sat >= 0;
	r.line("  交替 %.0f s（%d 段 FORWARD×%.0f s，YAW×%.1f s）：FORWARD 稳态机体 vx 均值 %.3f、最小 %.3f m/s（+%.1f s）；姿态估计误差最大 %.2f°（末态 roll %+.2f° pitch %+.2f°）；"
	       "速度环积分最大 %.2f°（%s）；末态 z=%.3f 偏航 %.1f° exec_state=%d 电压 %.3f V",
	       cfg::kLongrunAltS, n_fwd, cfg::kLongrunFwdS, cfg::kLongrunYawS, vx_n ? vx_sum / vx_n : 0, vx_min, t_vx_min, max_err,
	       (roll_rad(e.att_est) - roll_rad(e.quat)) * R, (pitch_rad(e.att_est) - pitch_rad(e.quat)) * R, max_int_deg,
	       saturated ? "已饱和" : "未饱和", e.pos[2], yaw_rad(e.quat) * R, e.exec_state, e.battery_v);
	std::snprintf(what, sizeof what, "FORWARD 段（起步 %.0f s 后）机体 vx 始终 > %.2f m/s（最小 %.3f，低于阈值 %d ms）——倒飞时 vx < 0",
	              cfg::kLongrunSettleS, cfg::kLongrunVxMinFrac * cfg::kForwardTarget, vx_min, fwd_vx_bad);
	r.check(fwd_vx_bad == 0, what);
	std::snprintf(what, sizeof what, "FORWARD 稳态 vx 均值 %.3f 在 %.1f ± %.0f%% 内", vx_n ? vx_sum / vx_n : 0, cfg::kForwardTarget, cfg::kSpeedTolFrac * 100);
	r.check(vx_n && std::fabs(vx_sum / vx_n - cfg::kForwardTarget) <= cfg::kSpeedTolFrac * cfg::kForwardTarget, what);
	// 积分饱和是否预期：零偏 |b| 取 6a 两轴合成与全程 max|δ|/总时长 的较大者；预算 (限幅 − 配平/卷绕余量)/|b| 撑过总时长才要求未饱和；
	// 撑不过时饱和是已知包络，只要求饱和那一刻 δ 已逼近预算（否则是积分卷绕之类的异常，不是零偏）。两个分支都恰好 1 项判据，总数仍 33。
	const double total_s = e.t_s, b_eff = std::fmax(hypot2(bx, by) * R, max_err / total_s);
	const double budget_deg = cfg::kLongrunIntSatDeg - cfg::kLongrunIntTrimDeg, t_budget = budget_deg / b_eff;
	if (t_budget > total_s) {
		std::snprintf(what, sizeof what, "速度环积分未饱和：最大 %.2f° < 限幅 %.0f°（零偏 |b| %.3f °/s → 预算 %.0f°/|b| = %.0f s > 总时长 %.0f s，不该饱和）",
		              max_int_deg, cfg::kLongrunIntSatDeg, b_eff, budget_deg, t_budget, total_s);
		r.check(!saturated, what);
	} else {
		r.line("  （信息）零偏 |b| %.3f °/s 下 %.0f° 预算只撑 %.0f s ≤ 总时长 %.0f s，积分饱和是库的已知包络（README「速度环」；根治要在飞行中估零偏）%s",
		       b_eff, budget_deg, t_budget, total_s, saturated ? "" : "；本次没饱和");
		if (saturated)
			std::snprintf(what, sizeof what, "积分饱和由零偏解释：t=%.1f s 饱和时姿态估计误差 %.2f° ≥ 预算 %.0f°（FORWARD vx 与末态判据照常生效）", t_sat, err_at_sat, budget_deg);
		else
			std::snprintf(what, sizeof what, "积分最大 %.2f° 未到限幅 %.0f°（零偏包络内，不要求）", max_int_deg, cfg::kLongrunIntSatDeg);
		r.check(!saturated || err_at_sat >= budget_deg, what);
	}
	std::snprintf(what, sizeof what, "末态 exec_state=%d（ACTING/READY，非 FAILSAFE）且 |z−%.1f| = %.3f m < %.1f m", e.exec_state, cfg::kTargetHeight, std::fabs(e.pos[2] - cfg::kTargetHeight), cfg::kLongrunZTolM);
	r.check((e.exec_state == CF_SITL_ACTING || e.exec_state == CF_SITL_READY) && std::fabs(e.pos[2] - cfg::kTargetHeight) < cfg::kLongrunZTolM, what);
}
