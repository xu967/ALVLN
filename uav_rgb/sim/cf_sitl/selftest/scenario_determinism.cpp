// 场景 (4) 确定性：同一脚本（reset → arm_takeoff → 悬停 → FORWARD → YAW_LEFT → SAFE）在本进程跑一遍、
// 再起两个独立进程（本程序 --hash-run）各跑一遍，三次逐步状态哈希（FNV-1a 逐字段）必须逐位相同。
// 跨进程比较排除了「同进程内残留状态恰好相同」的假阳性。
#include "selftest_util.h"
#include "selftest_config.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>

uint64_t determinism_hash_run(const Ctx &) {
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(cfg::kDetEpisode, spawn, 0);
	uint64_t h = 1469598103934665603ull;
	auto run = [&](double secs) {
		for (int i = 0, n = (int)std::lround(secs / cfg::kDt); i < n; i++) {
			if (!step_ms(1)) return false;
			h = hash_state(h, get_state());
		}
		return true;
	};
	cf_sitl_arm_takeoff(cfg::kTargetHeight);
	if (!run(cfg::kDetHoverS)) return 0;
	for (int act : {1, 2, 0}) {
		cf_sitl_set_action(act, get_state().t_s + cfg::kDetActS);
		if (!run(cfg::kDetActS)) return 0;
	}
	return h;
}

namespace {
// 起子进程跑同一脚本，读回 "HASH <hex>"；失败返回 0
uint64_t child_hash(const Ctx &c) {
	char cmd[8192];
	std::snprintf(cmd, sizeof cmd, "'%s' --hash-run --params '%s' --seed %llu 2>/dev/null", c.exe.c_str(), c.params.c_str(), (unsigned long long)c.seed);
	FILE *p = popen(cmd, "r");
	if (!p) return 0;
	char line[256];
	unsigned long long h = 0;
	while (std::fgets(line, sizeof line, p)) std::sscanf(line, "HASH %llx", &h);
	pclose(p);
	return h;
}
}  // namespace

void scenario_determinism(Report &r, const Ctx &c) {
	const double total = cfg::kDetHoverS + 3 * cfg::kDetActS;
	r.line("== 4 确定性（同脚本 %.0f s：起飞悬停 %.0f s + FORWARD/YAW_LEFT/SAFE 各 %.0f s；本进程 1 次 + 子进程 2 次）", total, cfg::kDetHoverS, cfg::kDetActS);
	const auto c0 = std::chrono::steady_clock::now();
	const uint64_t h0 = determinism_hash_run(c);
	const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
	const uint64_t h1 = child_hash(c), h2 = child_hash(c);
	r.line("  本进程 %016llx，子进程 A %016llx，子进程 B %016llx；本进程墙钟 %.2f s（%.2f µs/步，%.0f× 实时）",
	       (unsigned long long)h0, (unsigned long long)h1, (unsigned long long)h2, wall, wall / (total / cfg::kDt) * 1e6, total / wall);
	r.check(h0 != 0 && h1 != 0 && h2 != 0, "三次都跑完（哈希非 0；0 = standalone 步进失败或子进程没输出）");
	r.check(h0 != 0 && h0 == h1 && h1 == h2, "两个独立进程与本进程的逐步状态哈希逐位一致");
}
