// test_yaw_in_place：YAW_LEFT/RIGHT 必须是原地偏航（cf_sitl.h 的约定）。飞 30 s 后固件姿态估计已漂 2–8°（只积分陀螺），
// 执行器速度环积分在机体系里抵消它；这个误差在世界系固定，机体偏航时若不把积分随之反转，转 180° 后补偿方向全反，
// 真倾角 = 2×误差，YAW 6 s（≈206°）侧移 0.4–1.9 m（修复前 seed 1 ep 0–4 实测 0.41–1.95 m；修复后 0.02–0.13 m）。
// 判据：SAFE 30 s 后 YAW_LEFT 6 s，偏航期间水平位移 < 0.25 m，且确实转过 > 150°。
// 参数文件不存在或库没有独立积分模式 → 退出 77（SKIP）。用法：test_yaw_in_place <e1_mini.json>
#include "cf_sitl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

// ---- 配置区 ----
static const uint64_t kSeed = 1;                    // 与 cf_sitl_smoke / selftest 默认一致
static const int kEpisodes = 5;                     // 5 个零偏抽样：修复前每个都 > 0.4 m
static const double kHeight = 2.0, kTakeoffMaxS = 15.0;
static const double kHoldS = 30.0;                  // 先悬停 30 s 让姿态估计漂 2–8°（1σ 零偏 0.1 °/s）
static const double kYawS = 6.0;                    // 0.6 rad/s × 6 s ≈ 206°：补偿方向最多反 180°
static const double kMaxDispM = 0.25;               // 修复后最大 0.13 m（含零偏×高度与光流噪声漂移）；修复前最小 0.41 m
static const double kMinTurnDeg = 150.0;            // 确认偏航真的执行了

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

static cf_sitl_state st() { cf_sitl_state s; std::memset(&s, 0, sizeof s); cf_sitl_get_state(&s); return s; }
static double yaw_deg(const double q[4]) { return std::atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3])) * 180 / M_PI; }
static double pitch_deg(const double q[4]) { return std::asin(std::fmax(-1.0, std::fmin(1.0, 2 * (q[0] * q[2] - q[3] * q[1])))) * 180 / M_PI; }
static double roll_deg(const double q[4]) { return std::atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])) * 180 / M_PI; }

int main(int argc, char **argv) {
	if (argc < 2) { std::fprintf(stderr, "用法: %s <e1_mini.json>\n", argv[0]); return 2; }
	struct stat sb;
	if (stat(argv[1], &sb) != 0) { std::printf("SKIP: 参数文件不存在 %s\n", argv[1]); return 77; }
	if (cf_sitl_init(argv[1], kSeed) != 0) { std::printf("  [FAIL] cf_sitl_init 失败\n"); return 1; }
	char what[256];
	for (int ep = 0; ep < kEpisodes; ep++) {
		const double spawn[3] = {0, 0, 0};
		cf_sitl_reset(ep, spawn, 0);
		cf_sitl_arm_takeoff(kHeight);
		double t_ready = -1;
		for (;;) {
			const cf_sitl_state s = st();
			if (s.exec_state == CF_SITL_READY) { t_ready = s.t_s; break; }
			if (s.t_s >= kTakeoffMaxS) break;
			if (cf_sitl_step_standalone(10) != 0) { std::printf("SKIP: 库没有独立积分模式（桩库）\n"); return 77; }
		}
		std::snprintf(what, sizeof what, "ep %d arm_takeoff 到 READY（t=%.2f s）", ep, t_ready);
		CHECK(t_ready >= 0, what);
		if (t_ready < 0) continue;
		cf_sitl_set_action(0 /* SAFE */, t_ready + kHoldS + 1.0);
		cf_sitl_step_standalone((int)std::lround(kHoldS / 0.001));
		const cf_sitl_state s0 = st();
		const double err0 = std::hypot(pitch_deg(s0.att_est) - pitch_deg(s0.quat), roll_deg(s0.att_est) - roll_deg(s0.quat));
		cf_sitl_set_action(2 /* YAW_LEFT */, s0.t_s + kYawS + 1.0);
		double vmax = 0;
		for (int k = 0, n = (int)std::lround(kYawS / 0.001); k < n; k++) {
			cf_sitl_step_standalone(1);
			const cf_sitl_state s = st();
			vmax = std::fmax(vmax, std::hypot(s.vel[0], s.vel[1]));
		}
		const cf_sitl_state s1 = st();
		const double disp = std::hypot(s1.pos[0] - s0.pos[0], s1.pos[1] - s0.pos[1]);
		double turned = yaw_deg(s1.quat) - yaw_deg(s0.quat);
		while (turned < -180) turned += 360;
		while (turned > 180) turned -= 360;
		turned = std::fabs(turned) < 1e-9 ? 0 : turned;   // 206° 会回绕成 −154°，取绝对值
		std::printf("  ep %d：悬停 %.0f s 后姿态估计误差 %.2f°，YAW_LEFT %.0f s 转过 %.1f°，期间水平位移 %.3f m（最大速度 %.3f m/s），z=%.3f\n",
		            ep, kHoldS, err0, kYawS, turned, disp, vmax, s1.pos[2]);
		std::snprintf(what, sizeof what, "ep %d YAW_LEFT %.0f s 原地偏航：水平位移 %.3f m < %.2f m", ep, kYawS, disp, kMaxDispM);
		CHECK(disp < kMaxDispM, what);
		std::snprintf(what, sizeof what, "ep %d 偏航确实执行：|转过| %.1f° > %.0f°", ep, std::fabs(turned), kMinTurnDeg);
		CHECK(std::fabs(turned) > kMinTurnDeg, what);
	}
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
