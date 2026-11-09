// test_forward_lateral：FORWARD 航迹侧偏预算（R2-SITL 2026-11-07 Should fix #1）。固件飞行中只积分陀螺，姿态估计以陀螺零偏 b
// 线性漂走；速度环 PI 追这条斜坡的稳态误差是 b/ki：估计速度被压到 0 时真速度 = b_x/ki（侧向）、−b_y/ki（前向）。SAFE/YAW 有位置环
// 把它变成常值偏离，FORWARD 没有：Gazebo seed 190 集 26 个稳态 FORWARD 步侧向速度均值 +0.033 m/s（7 m 航程侧偏 0.8 m，航迹偏机头 6.3°）。
// 30 个 episode 各 FORWARD 30 s 的稳态机体 |vy| 均值（独立积分，与 $T/agent7-r2-sitl/probe/lat_probe 同种子）：
//   vel_ki 0.15 → 1.29 cm/s、最大 3.2、2/30 超 3 cm/s（红）；0.30 → 0.80 cm/s、最大 2.0（绿）；0.45 起 FORWARD 欠调（3 s 末 0.271 m/s），不再加。
// 判据：30 集 |vy| 均值 < 1.0 cm/s，且每集 |vy| < 3.0 cm/s；顺带确认 vx 均值仍在 0.3 ± 20%（ki 加大不能换来欠调）。
// 参数文件不存在或库没有独立积分模式 → 退出 77（SKIP）。用法：test_forward_lateral <e1_mini.json>
#include "cf_sitl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

// ---- 配置区 ----
static const uint64_t kSeed = 0;                    // 与 R2-SITL 探针同种子（probe/vy_sweep*.txt 可逐集对照）
static const int kEpisodes = 30;                    // 30 个零偏抽样：均值的统计误差约 0.1 cm/s
static const double kHeight = 2.0, kTakeoffMaxS = 15.0;
static const double kForwardS = 30.0;               // 稳态侧偏是常值 b/ki，30 s 让均值压过光流噪声（0.1 m/s 白噪声 3 Hz 低通）
static const double kSettleS = 2.0;                 // 起步瞬态（到 90% 约 0.6–0.7 s）不算
static const double kForwardTarget = 0.3, kSpeedTolFrac = 0.20;   // 同 selftest 场景 3
static const double kMeanMaxMps = 0.010;            // 30 集 |vy| 均值上限：ki 0.30 实测 0.0080；ki 0.15 为 0.0129
static const double kEpMaxMps = 0.030;              // 单集上限：ki 0.30 最大 0.0201；ki 0.15 有 2 集 0.0313 / 0.0323

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

static cf_sitl_state st() { cf_sitl_state s; std::memset(&s, 0, sizeof s); cf_sitl_get_state(&s); return s; }
static void body_vxy(const cf_sitl_state &s, double *vx, double *vy) {   // R^T v 的前两行
	const double w = s.quat[0], x = s.quat[1], y = s.quat[2], z = s.quat[3];
	*vx = (1 - 2 * (y * y + z * z)) * s.vel[0] + 2 * (x * y + w * z) * s.vel[1] + 2 * (x * z - w * y) * s.vel[2];
	*vy = 2 * (x * y - w * z) * s.vel[0] + (1 - 2 * (x * x + z * z)) * s.vel[1] + 2 * (y * z + w * x) * s.vel[2];
}

int main(int argc, char **argv) {
	if (argc < 2) { std::fprintf(stderr, "用法: %s <e1_mini.json>\n", argv[0]); return 2; }
	struct stat sb;
	if (stat(argv[1], &sb) != 0) { std::printf("SKIP: 参数文件不存在 %s\n", argv[1]); return 77; }
	if (cf_sitl_init(argv[1], kSeed) != 0) { std::printf("  [FAIL] cf_sitl_init 失败\n"); return 1; }
	char what[256];
	double abs_sum = 0, vx_sum = 0, worst = 0; int n_ok = 0, n_over = 0;
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
		if (t_ready < 0) { std::snprintf(what, sizeof what, "ep %d arm_takeoff 到 READY", ep); CHECK(false, what); continue; }
		cf_sitl_set_action(1 /* FORWARD */, t_ready + kForwardS + 1.0);
		double vy_sum = 0, vx_ep = 0; int n = 0;
		for (int k = 1, m = (int)std::lround(kForwardS / 0.001); k <= m; k++) {
			if (cf_sitl_step_standalone(1) != 0) { CHECK(false, "step_standalone 全程返回 0"); return g_fails; }
			if (k * 0.001 < kSettleS) continue;
			const cf_sitl_state s = st();
			double vx, vy; body_vxy(s, &vx, &vy);
			vy_sum += vy; vx_ep += vx; n++;
		}
		const double vy_mean = vy_sum / n, vx_mean = vx_ep / n;
		abs_sum += std::fabs(vy_mean); vx_sum += vx_mean; n_ok++;
		worst = std::fmax(worst, std::fabs(vy_mean));
		if (std::fabs(vy_mean) >= kEpMaxMps) n_over++;
		std::printf("  ep %2d：FORWARD %.0f s 稳态机体 vy 均值 %+.4f m/s，vx 均值 %.3f m/s\n", ep, kForwardS, vy_mean, vx_mean);
	}
	if (!n_ok) return g_fails ? g_fails : 1;
	const double mean_abs = abs_sum / n_ok, mean_vx = vx_sum / n_ok;
	std::printf("== %d 集：稳态 |vy| 均值 %.4f m/s，最大 %.4f m/s，≥ %.0f cm/s 的集 %d；vx 均值 %.3f m/s\n",
	            n_ok, mean_abs, worst, kEpMaxMps * 100, n_over, mean_vx);
	std::snprintf(what, sizeof what, "FORWARD 稳态侧偏 |vy| 均值 %.4f m/s < %.3f m/s（b/ki 稳态误差：ki 0.15 为 0.0129，0.30 为 0.0080）", mean_abs, kMeanMaxMps);
	CHECK(mean_abs < kMeanMaxMps, what);
	std::snprintf(what, sizeof what, "每集 |vy| < %.3f m/s（最大 %.4f，超限 %d 集）", kEpMaxMps, worst, n_over);
	CHECK(n_over == 0, what);
	std::snprintf(what, sizeof what, "FORWARD 稳态 vx 均值 %.3f 在 %.1f ± %.0f%% 内（ki 加大后没有欠调）", mean_vx, kForwardTarget, kSpeedTolFrac * 100);
	CHECK(std::fabs(mean_vx - kForwardTarget) <= kSpeedTolFrac * kForwardTarget, what);
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
