// test_forward_longrun：复现 3DGS route 5 seed 500 的倒飞（demo-e1-20261107/3dgs-callinstrasse-routes/collect/route_5/ep_00000）。
// 同一噪声流（init seed 500、episode 0，与 collect_3dgs.py 一致）起飞后持续 FORWARD 60 s：固件飞行中不修正姿态，
// 陀螺残余零偏让姿态估计以 ~0.22 °/s 漂移，执行器速度环积分抵消它；修复前积分与机动倾角共用 10° 限幅，t≈45 s
// 吃满后速度环开环，机体前向速度在 t≈47 s 变负、70 s 到 −3 m/s。修复后（积分单独限幅 20°）全程 vx > 0。
// 判据：FORWARD 开始 2 s 之后机体系 vx 始终 > 0.5·目标速度；末态仍 ACTING、高度误差 < 0.3 m。
// 参数文件不存在或库没有独立积分模式 → 退出 77（SKIP）。用法：test_forward_longrun <e1_mini.json>
#include "cf_sitl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

// ---- 配置区 ----
static const uint64_t kSeed = 500, kEpisode = 0;   // route_5/ep_00000 的 episode_manifest.json：seed 500；collect_3dgs.py 固定 episode=0
static const double kHeight = 2.0, kTakeoffMaxS = 15.0;
static const double kForwardS = 60.0;               // route 5 在 FORWARD 第 37 s（t_sitl≈44 s）出事；60 s 留余量
static const double kSettleS = 2.0;                 // FORWARD 起步瞬态不算
static const double kVxMinFrac = 0.5;               // 机体 vx 下限 = 0.5 × forward_speed（0.15 m/s）；倒飞时 vx < 0
static const double kForwardTarget = 0.3;
static const double kZTolM = 0.3;

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

static cf_sitl_state st() { cf_sitl_state s; std::memset(&s, 0, sizeof s); cf_sitl_get_state(&s); return s; }
static double body_vx(const cf_sitl_state &s) {   // 世界速度转机体 x：R^T v 的第一行
	const double w = s.quat[0], x = s.quat[1], y = s.quat[2], z = s.quat[3];
	return (1 - 2 * (y * y + z * z)) * s.vel[0] + 2 * (x * y + w * z) * s.vel[1] + 2 * (x * z - w * y) * s.vel[2];
}
static double pitch_deg(const double q[4]) { return std::asin(std::fmax(-1.0, std::fmin(1.0, 2 * (q[0] * q[2] - q[3] * q[1])))) * 180 / M_PI; }

int main(int argc, char **argv) {
	if (argc < 2) { std::fprintf(stderr, "用法: %s <e1_mini.json>\n", argv[0]); return 2; }
	struct stat sb;
	if (stat(argv[1], &sb) != 0) { std::printf("SKIP: 参数文件不存在 %s\n", argv[1]); return 77; }
	if (cf_sitl_init(argv[1], kSeed) != 0) { std::printf("  [FAIL] cf_sitl_init 失败\n"); return 1; }
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(kEpisode, spawn, 0);
	cf_sitl_arm_takeoff(kHeight);
	double t_ready = -1;
	for (;;) {
		const cf_sitl_state s = st();
		if (s.exec_state == CF_SITL_READY) { t_ready = s.t_s; break; }
		if (s.t_s >= kTakeoffMaxS) break;
		if (cf_sitl_step_standalone(10) != 0) { std::printf("SKIP: 库没有独立积分模式（桩库）\n"); return 77; }
	}
	char what[200];
	std::snprintf(what, sizeof what, "arm_takeoff 到 READY（t=%.2f s）", t_ready);
	CHECK(t_ready >= 0, what);
	if (t_ready < 0) return g_fails;

	const double t0 = st().t_s;
	cf_sitl_set_action(1 /* FORWARD */, t0 + kForwardS + 1.0);
	double vx_min = 1e9, t_vx_min = 0, t_first_neg = -1, max_err_deg = 0;
	const int n = (int)std::lround(kForwardS / 0.001);
	for (int k = 1; k <= n; k++) {
		if (cf_sitl_step_standalone(1) != 0) { CHECK(false, "step_standalone 全程返回 0"); return g_fails; }
		const cf_sitl_state s = st();
		const double vx = body_vx(s), tf = s.t_s - t0;
		const double err = std::fabs(pitch_deg(s.att_est) - pitch_deg(s.quat));
		max_err_deg = std::fmax(max_err_deg, err);
		if (tf < kSettleS) continue;
		if (vx < vx_min) { vx_min = vx; t_vx_min = tf; }
		if (t_first_neg < 0 && vx < 0) t_first_neg = tf;
		if (k % 10000 == 0)
			std::printf("  FORWARD +%.0f s：机体 vx %.3f m/s，真值俯仰 %+.2f°，固件估计俯仰 %+.2f°（估计误差 %.2f°），俯仰杆 %+.3f，z=%.3f\n",
			            tf, vx, pitch_deg(s.quat), pitch_deg(s.att_est), pitch_deg(s.att_est) - pitch_deg(s.quat), s.sticks[1], s.pos[2]);
	}
	const cf_sitl_state e = st();
	char neg[32] = "无";
	if (t_first_neg >= 0) std::snprintf(neg, sizeof neg, "+%.1f s", t_first_neg);
	std::printf("== seed %llu episode %llu：FORWARD %.0f s 内机体 vx 最小 %.3f m/s（+%.1f s），首次 < 0：%s，姿态估计误差最大 %.2f°，末态 exec_state=%d z=%.3f x=%.2f\n",
	            (unsigned long long)kSeed, (unsigned long long)kEpisode, kForwardS, vx_min, t_vx_min, neg, max_err_deg, e.exec_state, e.pos[2], e.pos[0]);
	std::snprintf(what, sizeof what, "FORWARD 起步 %.0f s 后机体 vx 始终 > %.2f m/s（最小 %.3f）——修复前 seed 500 在 +37 s 变负", kSettleS, kVxMinFrac * kForwardTarget, vx_min);
	CHECK(vx_min > kVxMinFrac * kForwardTarget, what);
	CHECK(t_first_neg < 0, "全程没有倒飞（机体 vx 从未 < 0）");
	std::snprintf(what, sizeof what, "末态仍在 ACTING 且 |z−%.1f| = %.3f m < %.1f m", kHeight, std::fabs(e.pos[2] - kHeight), kZTolM);
	CHECK(e.exec_state == CF_SITL_ACTING && std::fabs(e.pos[2] - kHeight) < kZTolM, what);
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
