// test_params_file：sim/dynamics/e1_mini.json 的约定——顶层 calibrated=false；每个参数项（含 "value" 的对象）
// 都有 status ∈ {doc, firmware, assumed}；其他 agent 要读的键存在。文件不存在时退出 77（CTest SKIP）。
// 用法：test_params_file <e1_mini.json>
#include "../src/json_min.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

int main(int argc, char **argv) {
	if (argc < 2) { std::fprintf(stderr, "用法: %s <e1_mini.json>\n", argv[0]); return 2; }
	struct stat st;
	if (stat(argv[1], &st) != 0) { std::printf("SKIP: 参数文件不存在 %s（SITL-core 尚未生成）\n", argv[1]); return 77; }
	std::ifstream f(argv[1]);
	std::stringstream ss; ss << f.rdbuf();
	Json j; std::string err;
	if (!Json::parse(ss.str(), j, err)) { std::printf("  [FAIL] 解析失败: %s\n", err.c_str()); return 1; }
	std::printf("== %s\n", argv[1]);
	CHECK(j.type == Json::Object, "顶层是对象");
	const Json *cal = j.get("calibrated");
	CHECK(cal && cal->type == Json::Bool && !cal->b, "顶层 calibrated 存在且为 false（数字未经真机标定）");

	std::map<std::string, int> by_status;
	int items = 0, bad = 0;
	for (const auto &kv : j.obj) {
		const Json &v = kv.second;
		if (v.type != Json::Object || !v.get("value")) continue;   // 元数据（schema/date/frame…）不算参数项
		items++;
		const Json *s = v.get("status");
		const bool ok = s && s->type == Json::String && (s->str == "doc" || s->str == "firmware" || s->str == "assumed");
		if (!ok) { bad++; std::printf("    参数项 %s 的 status 缺失或非法\n", kv.first.c_str()); }
		else by_status[s->str]++;
		if (!v.get("unit") || !v.get("source") || !v.get("basis")) std::printf("    提示：%s 缺 unit/source/basis 之一\n", kv.first.c_str());
	}
	std::printf("  参数项 %d 个：doc %d，firmware %d，assumed %d\n", items, by_status["doc"], by_status["firmware"], by_status["assumed"]);
	CHECK(items >= 20, "参数项数量 ≥ 20（草案有 60 余项，少于 20 说明文件结构变了）");
	CHECK(bad == 0, "每个参数项都有 status ∈ {doc, firmware, assumed}");

	// BRIEF §4：其他 agent 从同一文件读这些键；sim/executor 是库 init 的必需项
	for (const char *key : {"mass_flight", "inertia_diag", "collision_envelope", "camera_mount", "motor_positions", "sim", "executor"}) {
		char what[128]; std::snprintf(what, sizeof what, "存在 %s.value", key);
		const Json *v = j.get(key);
		CHECK(v && v->get("value"), what);
	}
	const Json *sim = j.get("sim") ? j.get("sim")->get("value") : nullptr;
	const Json *dt = sim ? sim->get("dt_s") : nullptr;
	CHECK(dt && dt->is_num() && dt->num == 0.001, "sim.value.dt_s == 0.001（库接口固定 1 ms）");
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
