// 场景 (1) 坐标系一致性：外部模式注入已知的世界系角速度/加速度，自己积分真值姿态，
// 看固件估计姿态（att_est）是否跟上。冻结接口不暴露固件原始 gyro/acc，所以用姿态这条链做判据：
//   陀螺链：世界 ω → SITL 转机体 → 芯片系旋转 → 固件 imuRotation 反转 → 积分。任一轴错/符号错 → 误差 ≥ 57°。
//   加计链：静止倾斜（gyro=0），只有 landed 时的加计修正能把估计拉到真实倾角，符号错则拉向反方向。
// 输出 <out>/frame.csv（t, phase, 真/估 欧拉角, 姿态误差）。
#include "selftest_util.h"
#include "selftest_config.h"

namespace {
struct Ext {   // 外部模式的刚体真值（只转不动，位置固定在地面）
	int k = 0;
	double q[4] = {1, 0, 0, 0};
	Csv &csv; int phase = 0;
	explicit Ext(Csv &c) : csv(c) {}
	// 推进一步：lib 看到的是本步开始时的姿态与常量 ω/acc；之后再把 ω 积进 q
	bool step(const double omega_w[3], const double acc_w[3]) {
		k++;
		cf_sitl_body_in in{};
		in.t_s = k * cfg::kDt;
		for (int i = 0; i < 4; i++) in.quat[i] = q[i];
		for (int i = 0; i < 3; i++) { in.omega[i] = omega_w[i]; in.acc[i] = acc_w[i]; }
		cf_sitl_wrench w;
		if (cf_sitl_step_external(&in, &w) != 0) return false;
		const double n = std::sqrt(omega_w[0] * omega_w[0] + omega_w[1] * omega_w[1] + omega_w[2] * omega_w[2]);
		if (n > 0) {
			const double axis[3] = {omega_w[0] / n, omega_w[1] / n, omega_w[2] / n};
			double dq[4], o[4];
			quat_from_axis(axis, n * cfg::kDt, dq);
			quat_mul(dq, q, o);   // 世界系角速度：左乘
			for (int i = 0; i < 4; i++) q[i] = o[i];
			quat_normalize(q);
		}
		if (k % cfg::kCsvEveryMs == 0) {
			const cf_sitl_state s = get_state();
			const double R = 180 / M_PI;
			csv.row("%.3f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f", k * cfg::kDt, phase,
			        roll_rad(q) * R, pitch_rad(q) * R, yaw_rad(q) * R,
			        roll_rad(s.att_est) * R, pitch_rad(s.att_est) * R, yaw_rad(s.att_est) * R, quat_angle_deg(q, s.att_est));
		}
		return true;
	}
	bool run(double secs, const double omega_w[3], const double acc_w[3]) {
		for (int i = 0, n = (int)std::lround(secs / cfg::kDt); i < n; i++) if (!step(omega_w, acc_w)) return false;
		return true;
	}
	double err_deg() const { return quat_angle_deg(q, get_state().att_est); }
};
const double kZero[3] = {0, 0, 0};
const double kLift[3] = {0, 0, cfg::kFrameLiftAcc};
}  // namespace

void scenario_frame(Report &r, const Ctx &c) {
	r.line("== 1 坐标系一致性（外部模式：注入世界系 ω/acc，比较固件 att_est 与真值姿态）");
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(1, spawn, 0);
	Csv csv(c.out + "/frame.csv", "t,phase,roll_deg,pitch_deg,yaw_deg,est_roll_deg,est_pitch_deg,est_yaw_deg,att_err_deg");
	Ext b(csv);
	char what[256];

	// 0. 静止：固件落地校准零偏（imu.ino:70）+ 加计修正把估计拉平
	if (!b.run(cfg::kFrameStillS, kZero, kZero)) { r.check(false, "cf_sitl_step_external 可用"); return; }
	const double e0 = b.err_deg();
	std::snprintf(what, sizeof what, "静止 %.0f s 后固件姿态估计 ≈ 单位四元数（误差 %.3f° < %.1f°）", cfg::kFrameStillS, e0, cfg::kFrameAttTolDeg);
	r.check(e0 < cfg::kFrameAttTolDeg, what);

	// 1-3. 陀螺链：绕世界 x/y/z 各转 +ω·T，再转回；转动期间注入竖直加速度让固件 landed=false
	const char *axis_name[3] = {"x(roll)", "y(pitch)", "z(yaw)"};
	for (int ax = 0; ax < 3; ax++) {
		double w_pos[3] = {0, 0, 0}, w_neg[3] = {0, 0, 0};
		w_pos[ax] = cfg::kFrameRateRps; w_neg[ax] = -cfg::kFrameRateRps;
		b.phase = 1 + ax;
		if (!b.run(cfg::kFrameSegS, w_pos, kLift)) { r.check(false, "cf_sitl_step_external 可用"); return; }
		const cf_sitl_state s = get_state();
		const double R = 180 / M_PI;
		const double tru = (ax == 0 ? roll_rad(b.q) : ax == 1 ? pitch_rad(b.q) : yaw_rad(b.q)) * R;
		const double est = (ax == 0 ? roll_rad(s.att_est) : ax == 1 ? pitch_rad(s.att_est) : yaw_rad(s.att_est)) * R;
		const double err = b.err_deg();
		r.line("  绕世界 %s 转 %.2f rad/s × %.0f s：真值 %+.2f°，固件估计 %+.2f°，姿态误差 %.3f°", axis_name[ax], cfg::kFrameRateRps, cfg::kFrameSegS, tru, est, err);
		std::snprintf(what, sizeof what, "陀螺链 %s 轴向与符号一致（误差 %.3f° < %.1f°）", axis_name[ax], err, cfg::kFrameAttTolDeg);
		r.check(err < cfg::kFrameAttTolDeg, what);
		b.run(cfg::kFrameSegS, w_neg, kLift);     // 转回
		b.phase = 0;
		b.run(cfg::kFrameSegS, kZero, kZero);     // 静止 1 s，加计修正吸收残差
	}

	// 4-5. 加计链：瞬间置成倾斜姿态、ω=0、acc=0（落地），看加计修正把 roll/pitch 估计拉到真值
	const double R = 180 / M_PI;
	for (int ax = 0; ax < 2; ax++) {
		double axis[3] = {0, 0, 0}; axis[ax] = 1;
		double q_yaw[4] = {std::cos(yaw_rad(b.q) / 2), 0, 0, std::sin(yaw_rad(b.q) / 2)}, dq[4];
		quat_from_axis(axis, cfg::kFrameTiltDeg / R, dq);
		quat_mul(q_yaw, dq, b.q);                  // 保留当前偏航，只加倾角（机体轴旋转：右乘）
		b.phase = 4 + ax;
		if (!b.run(cfg::kFrameTiltSettleS, kZero, kZero)) { r.check(false, "cf_sitl_step_external 可用"); return; }
		const cf_sitl_state s = get_state();
		const double tru = (ax == 0 ? roll_rad(b.q) : pitch_rad(b.q)) * R;
		const double est = (ax == 0 ? roll_rad(s.att_est) : pitch_rad(s.att_est)) * R;
		r.line("  静止倾斜 %s %+.1f° 持续 %.0f s：固件估计 %+.2f°", axis_name[ax], tru, cfg::kFrameTiltSettleS, est);
		std::snprintf(what, sizeof what, "加计链 %s 轴向与符号一致（|估计−真值| %.3f° < %.1f°）", axis_name[ax], std::fabs(est - tru), cfg::kFrameTiltTolDeg);
		r.check(std::fabs(est - tru) < cfg::kFrameTiltTolDeg, what);
		for (int i = 0; i < 4; i++) b.q[i] = q_yaw[i];   // 回到水平
		b.phase = 0;
		b.run(1.0, kZero, kZero);
	}
}
