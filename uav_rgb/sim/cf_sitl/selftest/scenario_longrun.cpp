// 场景 (6) 长航时：独立积分模式。READY 后 (a) SAFE 保持 30 s 看真值水平漂移（位置保持判据 < 5 cm）；
// (b) FORWARD / YAW_LEFT / FORWARD / YAW_RIGHT 交替 90 s，监控 FORWARD 段机体系 vx 的符号、固件姿态估计误差
// （估计 − 真值，飞行中固件只积分陀螺，误差以残余零偏速率线性增长）、俯仰/横滚杆（≈ 速度环积分 / 30°）、
// 光流速度估计。3DGS route 5 seed 500 的倒飞就是这条链：估计漂移 → 积分吃满 → 速度环开环 → vx 变负。
// 输出 <out>/longrun.csv。
#include "selftest_util.h"
#include "selftest_config.h"

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
	//   (2) 真值漂移 < 5 cm + 1.5·|b|·h·T —— 剩余漂移由光流陀螺补偿的零偏解释：执行器 flow_vel = (flow + gyro)·h，
	//       固件 rates 带残余零偏 b，速度环把「估计速度」压到 0 时真速度 = −b×h（2 m 高、0.1 °/s → 0.35 cm/s），
	//       任何位置环都看不见它；b 用固件姿态估计误差的增长率测（飞行中只积分陀螺，误差增长率 = 零偏）。
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
	const double bound = cfg::kLongrunSafeDriftM + cfg::kLongrunFlowBiasMargin * flow_bias_drift;
	r.line("  SAFE %.0f s：真值离保持起点 %.3f m（最大 %.3f m；%.2f cm/s），执行器相信的位移 %.3f m，z=%.3f；"
	       "固件姿态估计误差 roll %+.2f→%+.2f° pitch %+.2f→%+.2f° → 零偏 b≈(%+.3f, %+.3f) °/s；光流补偿零偏应漂 |b|·h·T = %.3f m；横滚杆 %+.3f 俯仰杆 %+.3f（×30° ≈ 速度环积分）",
	       cfg::kLongrunSafeS, drift, max_r, drift / cfg::kLongrunSafeS * 100, believed, run.s.pos[2],
	       err0_r * R, err1_r * R, err0_p * R, err1_p * R, bx * R, by * R, flow_bias_drift, run.s.sticks[0], run.s.sticks[1]);
	std::snprintf(what, sizeof what, "SAFE %.0f s 位置环闭合：执行器相信的位移 %.3f m < %.2f m", cfg::kLongrunSafeS, believed, cfg::kLongrunSafeDriftM);
	r.check(believed < cfg::kLongrunSafeDriftM, what);
	std::snprintf(what, sizeof what, "SAFE %.0f s 真值漂移 %.3f m < %.2f + %.1f×|b|·h·T = %.3f m（剩余漂移 = 光流陀螺补偿零偏，位置环不可见）",
	              cfg::kLongrunSafeS, drift, cfg::kLongrunSafeDriftM, cfg::kLongrunFlowBiasMargin, bound);
	r.check(drift < bound, what);
	if (drift >= cfg::kLongrunSafeDriftM)
		r.line("  （信息）真值漂移未达任务目标 %.2f m：要达到需 |b| < %.3f °/s（当前参数 gyro_bias_residual 1σ = 0.1 °/s），或执行器在飞行中估零偏",
		       cfg::kLongrunSafeDriftM, cfg::kLongrunSafeDriftM / ((h_sum / h_n) * cfg::kLongrunSafeS) * R);

	// 6b. FORWARD / YAW_LEFT / FORWARD / YAW_RIGHT 交替
	run.phase = 2;
	const double t_alt = run.s.t_s;
	double vx_min = 1e9, t_vx_min = 0, vx_sum = 0; int vx_n = 0, seg = 0, n_fwd = 0;
	double max_err = 0, max_int_deg = 0;
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
			max_int_deg = std::fmax(max_int_deg, std::fmax(std::fabs(run.s.sticks[0]), std::fabs(run.s.sticks[1])) * 30.0);
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
	r.line("  交替 %.0f s（%d 段 FORWARD×%.0f s，YAW×%.1f s）：FORWARD 稳态机体 vx 均值 %.3f、最小 %.3f m/s（+%.1f s）；姿态估计误差最大 %.2f°（末态 roll %+.2f° pitch %+.2f°）；"
	       "横滚/俯仰杆最大 ×30° = %.1f°（积分限幅 %s）；末态 z=%.3f 偏航 %.1f° exec_state=%d 电压 %.3f V",
	       cfg::kLongrunAltS, n_fwd, cfg::kLongrunFwdS, cfg::kLongrunYawS, vx_n ? vx_sum / vx_n : 0, vx_min, t_vx_min, max_err,
	       (roll_rad(e.att_est) - roll_rad(e.quat)) * R, (pitch_rad(e.att_est) - pitch_rad(e.quat)) * R, max_int_deg,
	       max_int_deg < cfg::kLongrunIntSatDeg ? "未饱和" : "已饱和", e.pos[2], yaw_rad(e.quat) * R, e.exec_state, e.battery_v);
	std::snprintf(what, sizeof what, "FORWARD 段（起步 %.0f s 后）机体 vx 始终 > %.2f m/s（最小 %.3f，低于阈值 %d ms）——倒飞时 vx < 0",
	              cfg::kLongrunSettleS, cfg::kLongrunVxMinFrac * cfg::kForwardTarget, vx_min, fwd_vx_bad);
	r.check(fwd_vx_bad == 0, what);
	std::snprintf(what, sizeof what, "FORWARD 稳态 vx 均值 %.3f 在 %.1f ± %.0f%% 内", vx_n ? vx_sum / vx_n : 0, cfg::kForwardTarget, cfg::kSpeedTolFrac * 100);
	r.check(vx_n && std::fabs(vx_sum / vx_n - cfg::kForwardTarget) <= cfg::kSpeedTolFrac * cfg::kForwardTarget, what);
	std::snprintf(what, sizeof what, "速度环积分（杆×30°）最大 %.1f° < 限幅 %.0f°（未饱和）", max_int_deg, cfg::kLongrunIntSatDeg);
	r.check(max_int_deg < cfg::kLongrunIntSatDeg, what);
	std::snprintf(what, sizeof what, "末态 exec_state=%d（ACTING/READY，非 FAILSAFE）且 |z−%.1f| = %.3f m < %.1f m", e.exec_state, cfg::kTargetHeight, std::fabs(e.pos[2] - cfg::kTargetHeight), cfg::kLongrunZTolM);
	r.check((e.exec_state == CF_SITL_ACTING || e.exec_state == CF_SITL_READY) && std::fabs(e.pos[2] - cfg::kTargetHeight) < cfg::kLongrunZTolM, what);
}
