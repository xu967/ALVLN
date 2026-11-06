// cf_sitl_selftest：对 libcf_sitl（固件在环 E1 MINI）跑 6 个场景，逐项 PASS/FAIL + 数字，退出码 = 失败数。
// 用法：cf_sitl_selftest --params sim/dynamics/e1_mini.json --out <dir> [--seed N]
//       画图：$UAV_RGB_PYTHON selftest/plot.py <dir>
// 超参数全部在 selftest_config.h（场景时长、注入量、阈值，行尾有依据）。
// 场景：(1) 坐标系一致性 (2) 悬停 10 s (3) 执行器级四个数字 (3.5) 固件级打杆阶跃（cf_sitl_set_sticks_raw 旁路执行器）
//       (4) 两进程确定性哈希 (5) 电池放电曲线 (6) 长航时：SAFE 保持 30 s + FORWARD/YAW 交替 90 s（姿态估计漂移/积分/倒飞监控）。
#include "selftest_util.h"
#include "selftest_config.h"

#include <cstdlib>
#include <sys/stat.h>
#include <unistd.h>

static void usage(const char *a0) {
	std::fprintf(stderr, "用法: %s --params <e1_mini.json> --out <dir> [--seed N]\n      %s --hash-run --params <e1_mini.json> [--seed N]\n", a0, a0);
}

int main(int argc, char **argv) {
	Ctx c;
	bool hash_run = false;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--params" && i + 1 < argc) c.params = argv[++i];
		else if (a == "--out" && i + 1 < argc) c.out = argv[++i];
		else if (a == "--seed" && i + 1 < argc) c.seed = std::strtoull(argv[++i], nullptr, 10);
		else if (a == "--hash-run") hash_run = true;
		else { usage(argv[0]); return 2; }
	}
	if (c.params.empty() || (!hash_run && c.out.empty())) { usage(argv[0]); return 2; }
	char exe[4096];
	const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
	c.exe = n > 0 ? std::string(exe, (size_t)n) : argv[0];

	if (cf_sitl_init(c.params.c_str(), c.seed) != 0) { std::fprintf(stderr, "cf_sitl_init 失败: %s\n", c.params.c_str()); return 2; }
	if (hash_run) {   // 确定性场景的子进程：只打印哈希
		std::printf("HASH %016llx\n", (unsigned long long)determinism_hash_run(c));
		return 0;
	}
	mkdir(c.out.c_str(), 0755);
	Report r(c.out + "/selftest.txt");
	r.line("cf_sitl_selftest  params=%s  seed=%llu  out=%s", c.params.c_str(), (unsigned long long)c.seed, c.out.c_str());
	r.line("（数字全部来自参数文件，calibrated=false；阈值与依据见 selftest_config.h）");

	scenario_frame(r, c);
	scenario_hover(r, c);
	scenario_actuator(r, c);
	scenario_sticks(r, c);
	scenario_determinism(r, c);
	scenario_battery(r, c);
	scenario_longrun(r, c);

	r.line("== 结果: %d 项失败，%d 项通过，%d 项跳过", r.fails, r.passes, r.skips);
	return r.fails;
}
