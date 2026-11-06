// 冒烟：reset → arm_takeoff(2.0) → 独立积分 15 s，每秒打印一行，最后给出到 READY 的用时、悬停高度误差与油门杆。
// 用法：cf_sitl_smoke [e1_mini.json] [seed]；退出码 0 = 到达 READY。
#include "cf_sitl.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
	const char *params = argc > 1 ? argv[1] : CF_SITL_DEFAULT_PARAMS;
	const uint64_t seed = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1;
	if (cf_sitl_init(params, seed) != 0) return 2;
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(1, spawn, 0);
	cf_sitl_arm_takeoff(2.0);
	cf_sitl_state st{};
	double t_ready = -1, err_sum = 0, thr_sum = 0; int n_hover = 0;
	std::printf("   t  exec armed    z     h_est  vz_est  hover_stick  thr   vbat  motors\n");
	for (int i = 0; i < 15000; i++) {
		if (cf_sitl_step_standalone(1) != 0) return 3;
		cf_sitl_get_state(&st);
		if (t_ready < 0 && st.exec_state == CF_SITL_READY) t_ready = st.t_s;
		if (t_ready >= 0) { err_sum += std::fabs(st.pos[2] - 2.0); thr_sum += st.sticks[3]; n_hover++; }
		if ((i + 1) % 1000 == 0)
			std::printf("%5.1f  %d    %d    %6.3f %6.3f %7.3f   %6.3f   %5.3f %5.3f  %.2f %.2f %.2f %.2f\n", st.t_s, st.exec_state,
			            st.armed, st.pos[2], st.height_est_m, st.vel_est[0], st.hover_thrust, st.sticks[3], st.battery_v,
			            st.motors[0], st.motors[1], st.motors[2], st.motors[3]);
	}
	if (t_ready < 0) { std::printf("15 s 内未到 READY（exec_state=%d，z=%.3f）\n", st.exec_state, st.pos[2]); return 1; }
	std::printf("READY 用时 %.3f s；READY 之后 %.1f s 内平均高度误差 %.3f m，末态 z=%.3f m，xy 漂移 %.3f m；平均油门杆 %.3f，末态悬停前馈 %.3f\n",
	            t_ready, n_hover / 1000.0, err_sum / n_hover, st.pos[2], std::hypot(st.pos[0], st.pos[1]), thr_sum / n_hover, st.hover_thrust);
	return 0;
}
