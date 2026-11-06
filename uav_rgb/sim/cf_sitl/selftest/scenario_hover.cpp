// 场景 (2) 悬停 10 s：独立积分模式，arm_takeoff 到 READY 后不下发动作（执行器回悬停），
// 统计倾角 RMS、高度误差 RMS、水平漂移。输出 <out>/hover.csv。
#include "selftest_util.h"
#include "selftest_config.h"

void scenario_hover(Report &r, const Ctx &c) {
	r.line("== 2 悬停 %.0f s（独立积分模式，READY 后无动作）", cfg::kHoverS);
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(2, spawn, 0);
	const double t_ready = takeoff_until_ready(cfg::kTargetHeight, cfg::kTakeoffMaxS);
	if (t_ready < 0) {
		const cf_sitl_state s = get_state();
		r.line("  起飞失败：t=%.2f s exec_state=%d armed=%d z=%.3f（standalone 返回非 0 或 %.0f s 内未到 READY）", s.t_s, s.exec_state, s.armed, s.pos[2], cfg::kTakeoffMaxS);
		r.check(false, "arm_takeoff 到 READY");
		return;
	}
	Csv csv(c.out + "/hover.csv", "t,x,y,z,tilt_deg,yaw_deg,vx,vy,vz,height_est,tof_m,hover_thrust,battery_v,m0,m1,m2,m3,exec_state");
	const cf_sitl_state s0 = get_state();
	double sum_tilt2 = 0, sum_dz2 = 0, max_r = 0;
	int n = 0;
	const int steps = (int)std::lround(cfg::kHoverS / cfg::kDt);
	for (int k = 1; k <= steps; k++) {
		if (!step_ms(1)) { r.check(false, "cf_sitl_step_standalone 可用"); return; }
		const cf_sitl_state s = get_state();
		const double tilt = tilt_deg(s.quat);
		sum_tilt2 += tilt * tilt; sum_dz2 += (s.pos[2] - cfg::kTargetHeight) * (s.pos[2] - cfg::kTargetHeight); n++;
		max_r = std::fmax(max_r, hypot2(s.pos[0] - s0.pos[0], s.pos[1] - s0.pos[1]));
		if (k % cfg::kCsvEveryMs == 0)
			csv.row("%.3f,%.4f,%.4f,%.4f,%.3f,%.2f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f,%.3f,%.3f,%d",
			        s.t_s, s.pos[0], s.pos[1], s.pos[2], tilt, yaw_rad(s.quat) * 180 / M_PI, s.vel[0], s.vel[1], s.vel[2],
			        s.height_est_m, s.tof_m, s.hover_thrust, s.battery_v, s.motors[0], s.motors[1], s.motors[2], s.motors[3], s.exec_state);
	}
	const cf_sitl_state s1 = get_state();
	const double tilt_rms = std::sqrt(sum_tilt2 / n), z_rms = std::sqrt(sum_dz2 / n);
	const double drift = hypot2(s1.pos[0] - s0.pos[0], s1.pos[1] - s0.pos[1]);
	r.line("  READY 用时 %.2f s；窗口 [%.2f, %.2f] s：倾角 RMS %.3f°，高度误差 RMS %.3f m，水平漂移 %.3f m（最大离起点 %.3f m），末态 z=%.3f 偏航 %.1f°，悬停油门杆 %.3f，电压 %.3f V",
	       t_ready, s0.t_s, s1.t_s, tilt_rms, z_rms, drift, max_r, s1.pos[2], yaw_rad(s1.quat) * 180 / M_PI, s1.hover_thrust, s1.battery_v);
	char what[200];
	std::snprintf(what, sizeof what, "倾角 RMS %.3f° < %.1f°", tilt_rms, cfg::kHoverTiltRmsDeg);
	r.check(tilt_rms < cfg::kHoverTiltRmsDeg, what);
	std::snprintf(what, sizeof what, "高度误差 RMS %.3f m < %.2f m", z_rms, cfg::kHoverZRmsM);
	r.check(z_rms < cfg::kHoverZRmsM, what);
	std::snprintf(what, sizeof what, "水平漂移 %.3f m < %.1f m（%.0f s）", drift, cfg::kHoverDriftM, cfg::kHoverS);
	r.check(drift < cfg::kHoverDriftM, what);
	std::snprintf(what, sizeof what, "全程 exec_state 保持 READY（末态 %d）", s1.exec_state);
	r.check(s1.exec_state == CF_SITL_READY, what);
}
