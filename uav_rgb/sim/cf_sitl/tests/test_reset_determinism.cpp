// test_reset_determinism：同一进程里 reset 两次、喂同样的输入，逐步状态哈希必须逐位相同；
// 换 episode 则必须不同（噪声种子 = seed ^ episode）。参数文件不存在退出 77（SKIP）。
// 用法：test_reset_determinism <e1_mini.json>
#include "cf_sitl.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>

// ---- 配置区 ----
static const double kHeight = 2.0;      // BRIEF §4 目标高度
static const int kSteps = 3000;         // 3 s：覆盖解锁手势 + 爬升（smoke 里 READY 约 6.6 s，不必到 READY）
static const int kActionAt = 1500;      // 1.5 s 时下发 FORWARD，让执行器动作分支也进哈希

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
	const unsigned char *b = (const unsigned char *)p;
	for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 1099511628211ull;
	return h;
}
static uint64_t hash_state(uint64_t h, const cf_sitl_state &s) {   // 逐字段，不含结构体填充
	h = fnv(h, &s.t_s, sizeof s.t_s); h = fnv(h, &s.exec_state, sizeof s.exec_state); h = fnv(h, &s.armed, sizeof s.armed);
	h = fnv(h, s.motors, sizeof s.motors); h = fnv(h, &s.battery_v, sizeof s.battery_v); h = fnv(h, s.att_est, sizeof s.att_est);
	h = fnv(h, &s.tof_m, sizeof s.tof_m); h = fnv(h, s.flow_vel, sizeof s.flow_vel); h = fnv(h, &s.hover_thrust, sizeof s.hover_thrust);
	h = fnv(h, s.pos, sizeof s.pos); h = fnv(h, s.quat, sizeof s.quat); h = fnv(h, s.vel, sizeof s.vel);
	return h;
}
// 一次完整的 reset → 脚本；返回哈希，standalone 不可用返回 0
static uint64_t run(uint64_t episode, cf_sitl_state *first, cf_sitl_state *last) {
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(episode, spawn, 0.3);
	std::memset(first, 0, sizeof *first); cf_sitl_get_state(first);
	cf_sitl_arm_takeoff(kHeight);
	uint64_t h = 1469598103934665603ull;
	for (int k = 1; k <= kSteps; k++) {
		if (k == kActionAt) cf_sitl_set_action(1, k * 0.001 + 1.0);
		if (cf_sitl_step_standalone(1) != 0) return 0;
		cf_sitl_state s; std::memset(&s, 0, sizeof s); cf_sitl_get_state(&s);
		h = hash_state(h, s);
		if (k == kSteps) *last = s;
	}
	return h;
}

int main(int argc, char **argv) {
	if (argc < 2) { std::fprintf(stderr, "用法: %s <e1_mini.json>\n", argv[0]); return 2; }
	struct stat st;
	if (stat(argv[1], &st) != 0) { std::printf("SKIP: 参数文件不存在 %s\n", argv[1]); return 77; }
	if (cf_sitl_init(argv[1], 7) != 0) { std::printf("  [FAIL] cf_sitl_init 失败\n"); return 1; }
	cf_sitl_state f1, l1, f2, l2, f3, l3;
	const uint64_t h1 = run(2, &f1, &l1), h2 = run(2, &f2, &l2), h3 = run(3, &f3, &l3);
	std::printf("== reset 确定性：episode 2 两次 %016llx / %016llx，episode 3 %016llx\n", (unsigned long long)h1, (unsigned long long)h2, (unsigned long long)h3);
	CHECK(h1 != 0, "standalone 步进可用（哈希非 0）");
	CHECK(f1.t_s == 0 && f1.exec_state == CF_SITL_DISARMED && f1.armed == 0, "reset 后 t_s=0、DISARMED、未解锁");
	CHECK(f2.t_s == 0 && f2.exec_state == CF_SITL_DISARMED && f2.armed == 0, "第二次 reset 同样回到上电状态");
	CHECK(h1 == h2, "两次 reset 同输入 3000 步状态哈希逐位一致");
	CHECK(std::memcmp(l1.pos, l2.pos, sizeof l1.pos) == 0 && std::memcmp(l1.quat, l2.quat, sizeof l1.quat) == 0, "末态 pos/quat 逐位一致");
	CHECK(h1 != h3, "换 episode 哈希不同（噪声种子变了）");
	std::printf("  末态：t=%.3f s armed=%d exec_state=%d z=%.4f m 电压 %.3f V\n", l1.t_s, l1.armed, l1.exec_state, l1.pos[2], l1.battery_v);
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
