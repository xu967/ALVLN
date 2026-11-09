// test_longrun_int_budget：速度环积分预算 vs 陀螺零偏——selftest 场景 6b 条件判据的 CTest 兜底（FIX-SITL 2026-11-07）。
// 固件飞行中只积分陀螺，姿态估计误差 δ 以零偏 |b| 线性增长（随机游走 1e-4 rad/s/√s 再加 ±3°/97 s）；执行器速度环积分 I（机体系，
// 随偏航反转）抵消它，I ≈ δ + 阻力配平（0.3 m/s 时 0.7°）+ FORWARD 起步卷绕瞬态（ki·∫(ref−v)dt ≈ 2.6°），限幅 ±tilt_int_max 20°。
// δ 真超过预算时 I 饱和是库的已知包络（selftest 6b 只记信息行）；这条测试拦住那个判据的漏洞：预算撑得住（(20° − 4°)/|b| > 总时长）
// 却仍饱和 → FAIL；撑不住时饱和那一刻 δ 必须已 ≥ 16°（否则是积分卷绕之类的异常，不是零偏）。
// 流程同 selftest 场景 6：起飞 → SAFE 30 s → FORWARD 3 s / YAW 1.5 s 交替 60 s，共 ≈97 s。|b| 取 SAFE 段估计误差增长率（两轴合成）
// 与全程 max|δ|/总时长 的较大者（随机游走让 30 s 窗口的 b 预测不到末段）。I 由杆量反推（只在 FORWARD 段，ref=(forward_speed, 0) 已知）：
// I = 杆×30° − clamp(kp·(ref − vel_est), ±tilt_max)；YAW 段跳过，1.5 s 内 I 最多再长 |b|·1.5 s ≈ 0.4°。
// 判据：每集 (a) 预算撑得住 → max|I| < 20°，撑不住只打印；(b) FORWARD 稳态机体 vx > 0.15 m/s（饱和后开环会倒飞）；(c) 至少 1 集落在 (a)。
// init seed 1、episode 0..7 → 噪声流 seed^ep ∈ {0..7}；流 3 = selftest --seed 5 场景 6 的噪声流（5^6）：零偏 0.226 °/s、97 s 真饱和，
// 覆盖「撑不住」分支。参数文件不存在或库没有独立积分模式 → 退出 77（SKIP）。用法：test_longrun_int_budget <e1_mini.json>
#include "cf_sitl.h"
#include "json_min.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

// ---- 配置区 ----
static const uint64_t kSeed = 1;
static const int kEpisodes = 8;                     // 8 个零偏抽样；P(|b| 合成 > 0.196 °/s = 撑不住) ≈ 15%，流 3 保证至少 1 个
static const double kHeight = 2.0, kTakeoffMaxS = 15.0;
static const double kSafeS = 30.0, kAltS = 60.0, kFwdS = 3.0, kYawS = 1.5;   // 同 selftest_config.h kLongrun*
static const double kSettleS = 1.0;                 // FORWARD 起步 1 s 后算稳态（到 90% 约 0.7 s）
static const double kVxMinMps = 0.15;               // = 0.5 × forward_speed；倒飞时 vx < 0
static const double kIntTrimDeg = 4.0;              // I 比 δ 多出的阻力配平 0.7° + FORWARD 起步卷绕 ki·0.3 m/s·0.5 s ≈ 2.6°（同 selftest_config.h kLongrunIntTrimDeg）
static const double kIntTolDeg = 0.05;              // 杆量反推 I 的 float 舍入 ~1e-4°；饱和时 I 恰等于限幅

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

static cf_sitl_state st() { cf_sitl_state s; std::memset(&s, 0, sizeof s); cf_sitl_get_state(&s); return s; }
static double body_vx(const cf_sitl_state &s) {
	const double w = s.quat[0], x = s.quat[1], y = s.quat[2], z = s.quat[3];
	return (1 - 2 * (y * y + z * z)) * s.vel[0] + 2 * (x * y + w * z) * s.vel[1] + 2 * (x * z - w * y) * s.vel[2];
}
static double roll_deg(const double q[4]) { return std::atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])) * 180 / M_PI; }
static double pitch_deg(const double q[4]) { return std::asin(std::fmax(-1.0, std::fmin(1.0, 2 * (q[0] * q[2] - q[3] * q[1])))) * 180 / M_PI; }
static double err_deg(const cf_sitl_state &s, double *er, double *ep) {   // 固件估计 − 真值（roll, pitch）与合成
	*er = roll_deg(s.att_est) - roll_deg(s.quat); *ep = pitch_deg(s.att_est) - pitch_deg(s.quat);
	return std::hypot(*er, *ep);
}
static double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Exec { double kp, tilt_max_deg, tilt_int_max_deg, per_stick_deg, forward_mps; };
static bool read_exec(const char *path, Exec *e) {
	std::ifstream f(path); std::stringstream ss; ss << f.rdbuf();
	Json j; std::string err;
	if (!Json::parse(ss.str(), j, err)) return false;
	const Json *ex = j.get("executor"); const Json *v = ex ? ex->get("value") : nullptr;
	if (!v) return false;
	const struct { const char *key; double *dst; } keys[] = {
		{"vel_kp", &e->kp}, {"tilt_max_deg", &e->tilt_max_deg}, {"tilt_int_max_deg", &e->tilt_int_max_deg},
		{"tilt_per_stick_deg", &e->per_stick_deg}, {"forward_speed_mps", &e->forward_mps}};
	for (const auto &k : keys) { const Json *x = v->get(k.key); if (!x || !x->is_num()) return false; *k.dst = x->num; }
	return true;
}

int main(int argc, char **argv) {
	if (argc < 2) { std::fprintf(stderr, "用法: %s <e1_mini.json>\n", argv[0]); return 2; }
	struct stat sb;
	if (stat(argv[1], &sb) != 0) { std::printf("SKIP: 参数文件不存在 %s\n", argv[1]); return 77; }
	Exec X;
	if (!read_exec(argv[1], &X)) { std::printf("  [FAIL] 读不到 executor.value 的 vel_kp/tilt_max_deg/tilt_int_max_deg/tilt_per_stick_deg/forward_speed_mps\n"); return 1; }
	if (cf_sitl_init(argv[1], kSeed) != 0) { std::printf("  [FAIL] cf_sitl_init 失败\n"); return 1; }
	const double R = 180 / M_PI, budget_deg = X.tilt_int_max_deg - kIntTrimDeg;
	char what[320];
	int n_required = 0;
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
		// SAFE 30 s：零偏 = 估计误差增长率
		double er0, ep0, er1, ep1;
		cf_sitl_state s = st(); err_deg(s, &er0, &ep0);
		cf_sitl_set_action(0 /* SAFE */, s.t_s + kSafeS + 1.0);
		double max_err = 0;
		for (int i = 0, m = (int)std::lround(kSafeS / 0.001); i < m; i++) {
			if (cf_sitl_step_standalone(1) != 0) { CHECK(false, "step_standalone 全程返回 0"); return g_fails; }
			s = st(); double a, b; max_err = std::fmax(max_err, err_deg(s, &a, &b));
		}
		err_deg(s, &er1, &ep1);
		const double b_safe = std::hypot(er1 - er0, ep1 - ep0) / kSafeS;   // °/s，两轴合成
		// FORWARD / YAW_LEFT / FORWARD / YAW_RIGHT 交替 60 s
		const double t_alt = s.t_s;
		double max_int = 0, t_sat = -1, err_at_sat = 0, vx_min = 1e9; int seg = 0, vx_bad = 0;
		while (s.t_s - t_alt < kAltS) {
			const int kind = seg % 4, action = kind == 0 || kind == 2 ? 1 : kind == 1 ? 2 : 3;
			const double dur = action == 1 ? kFwdS : kYawS, t0 = s.t_s;
			cf_sitl_set_action(action, t0 + dur);
			for (int i = 0, m = (int)std::lround(dur / 0.001); i < m; i++) {
				if (cf_sitl_step_standalone(1) != 0) { CHECK(false, "step_standalone 全程返回 0"); return g_fails; }
				s = st(); double a, b; const double e = err_deg(s, &a, &b); max_err = std::fmax(max_err, e);
				if (action != 1) continue;
				// 反推积分：pitch 杆 = (clamp(kp·(vx_ref − vel_est_x)) + I_x)/per_stick；roll 杆 = −(clamp(kp·(0 − vel_est_y)) + I_y)/per_stick
				const double px = clampd(X.kp * (X.forward_mps - s.vel_est[0]) * R, -X.tilt_max_deg, X.tilt_max_deg);
				const double py = clampd(X.kp * (0.0 - s.vel_est[1]) * R, -X.tilt_max_deg, X.tilt_max_deg);
				const double ix = s.sticks[1] * X.per_stick_deg - px, iy = -s.sticks[0] * X.per_stick_deg - py;
				const double im = std::fmax(std::fabs(ix), std::fabs(iy));
				if (im > max_int) max_int = im;
				if (t_sat < 0 && im >= X.tilt_int_max_deg - kIntTolDeg) { t_sat = s.t_s; err_at_sat = e; }
				if (s.t_s - t0 >= kSettleS) {
					const double vx = body_vx(s);
					vx_min = std::fmin(vx_min, vx);
					if (vx < kVxMinMps) vx_bad++;
				}
			}
			seg++;
		}
		const double total_s = s.t_s;
		const double b_eff = std::fmax(b_safe, max_err / total_s);        // °/s
		const double t_budget = budget_deg / b_eff;                        // 预算撑得住的时长
		const bool required = t_budget > total_s;
		if (required) n_required++;
		std::printf("  ep %d（噪声流 %llu）：零偏 |b| SAFE 段 %.3f °/s、全程等效 %.3f °/s → 预算 %.0f°/|b| = %.0f s %s 总时长 %.0f s；估计误差最大 %.2f°；"
		            "积分最大 %.2f°%s；FORWARD 稳态 vx 最小 %.3f m/s\n",
		            ep, (unsigned long long)(kSeed ^ (uint64_t)ep), b_safe, max_err / total_s, budget_deg, t_budget, required ? ">" : "≤", total_s,
		            max_err, max_int, t_sat >= 0 ? "（已饱和）" : "", vx_min);
		if (required) {
			std::snprintf(what, sizeof what, "ep %d 预算撑得住（%.0f s > %.0f s）→ 积分未饱和：最大 %.2f° < %.0f°", ep, t_budget, total_s, max_int, X.tilt_int_max_deg);
			CHECK(t_sat < 0, what);
		} else {
			std::printf("  （信息）ep %d 零偏 %.3f °/s 下 %.0f° 预算只撑 %.0f s ≤ 总时长 %.0f s，饱和是库的已知包络（README「速度环」）\n",
			            ep, b_eff, budget_deg, t_budget, total_s);
			if (t_sat >= 0) std::snprintf(what, sizeof what, "ep %d 饱和由零偏解释：t=%.1f s 饱和时估计误差 %.2f° ≥ 预算 %.0f°", ep, t_sat, err_at_sat, budget_deg);
			else std::snprintf(what, sizeof what, "ep %d 积分最大 %.2f° 未到限幅（零偏包络内，不要求）", ep, max_int);
			CHECK(t_sat < 0 || err_at_sat >= budget_deg, what);
		}
		std::snprintf(what, sizeof what, "ep %d FORWARD 段（起步 %.0f s 后）机体 vx 始终 > %.2f m/s（最小 %.3f，低于阈值 %d ms）", ep, kSettleS, kVxMinMps, vx_min, vx_bad);
		CHECK(vx_bad == 0, what);
	}
	std::snprintf(what, sizeof what, "至少 1 集落在「预算撑得住、要求未饱和」分支（%d / %d）", n_required, kEpisodes);
	CHECK(n_required > 0, what);
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
