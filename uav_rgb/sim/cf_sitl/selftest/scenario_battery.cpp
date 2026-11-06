// 场景 (5) 电池：从 SoC 1.0 起悬停放电，记录母线电压与悬停油门杆曲线，看固件 L1（3.5 V）电平何时到、
// L3（3.0 V → 自动降落 fw_mode=AUTO）是否触发。SoC 不在冻结接口里，终点用「固件 L3 触发 / 落地上锁 / 上限时长」代替
// 「SoC 0.3」。起始 SoC 由参数文件 sim.value.battery_soc_init 决定，不是 1.0 时写一份改过的副本到 <out>/ 再 init。
// 输出 <out>/battery.csv。
#include "selftest_util.h"
#include "selftest_config.h"
#include "../src/json_min.h"

#include <fstream>
#include <sstream>

namespace {
Json *mut_get(Json &j, const char *key) {
	if (j.type != Json::Object) return nullptr;
	for (auto &kv : j.obj) if (kv.first == key) return &kv.second;
	return nullptr;
}
// 读参数文件；返回 battery_soc_init（缺失为 -1）、endurance 分钟（缺失为 -1）；soc_init ≠ 目标时写副本并返回其路径
std::string prepare_params(const Ctx &c, Report &r, double &soc_init, double &endurance_min) {
	soc_init = endurance_min = -1;
	std::ifstream f(c.params);
	std::stringstream ss; ss << f.rdbuf();
	Json j; std::string err;
	if (!Json::parse(ss.str(), j, err)) { r.line("  参数文件解析失败：%s", err.c_str()); return c.params; }
	if (const Json *e = j.get("endurance")) if (const Json *v = e->get("value")) if (v->is_num()) endurance_min = v->num;
	Json *sim = mut_get(j, "sim"); Json *sv = sim ? mut_get(*sim, "value") : nullptr; Json *soc = sv ? mut_get(*sv, "battery_soc_init") : nullptr;
	if (!soc || !soc->is_num()) { r.line("  参数文件没有 sim.value.battery_soc_init，按库默认起始 SoC"); return c.params; }
	soc_init = soc->num;
	if (std::fabs(soc_init - cfg::kBatSocInit) < 1e-12) return c.params;
	soc->num = cfg::kBatSocInit;
	const std::string path = c.out + "/e1_mini_soc1.json";
	std::ofstream o(path); o << j.dump() << "\n";
	r.line("  参数文件 battery_soc_init=%.2f ≠ %.1f，写副本 %s 并重新 init", soc_init, cfg::kBatSocInit, path.c_str());
	return path;
}
}  // namespace

void scenario_battery(Report &r, const Ctx &c) {
	r.line("== 5 电池放电（悬停到固件 L3 触发/落地/上限 %.0f s；SoC 不可观测，看 V_bus 与悬停油门杆）", cfg::kBatMaxS);
	double soc_init = -1, endurance_min = -1;
	const std::string params = prepare_params(c, r, soc_init, endurance_min);
	if (params != c.params && cf_sitl_init(params.c_str(), c.seed) != 0) { r.check(false, "用改过起始 SoC 的参数副本 init"); return; }
	const double spawn[3] = {0, 0, 0};
	cf_sitl_reset(5, spawn, 0);
	const double v0 = get_state().battery_v;
	const double t_ready = takeoff_until_ready(cfg::kTargetHeight, cfg::kTakeoffMaxS);
	if (t_ready < 0) {
		r.check(false, "arm_takeoff 到 READY");
		if (params != c.params) cf_sitl_init(c.params.c_str(), c.seed);
		return;
	}
	Csv csv(c.out + "/battery.csv", "t,battery_v,hover_thrust,throttle_stick,motor_mean,z,fw_mode,armed,exec_state");
	const cf_sitl_state s_ready = get_state();
	double t_l1 = -1, t_l3v = -1, t_auto = -1, t_disarm = -1, t_ground = -1, v_min = 1e9, z_max = 0;
	int k = 0;
	const int max_steps = (int)std::lround(cfg::kBatMaxS / cfg::kDt);
	cf_sitl_state s = s_ready;
	while (k < max_steps) {
		if (!step_ms(1)) { r.check(false, "cf_sitl_step_standalone 可用"); break; }
		s = get_state(); k++;
		v_min = std::fmin(v_min, s.battery_v); z_max = std::fmax(z_max, s.pos[2]);
		if (t_l1 < 0 && s.battery_v < cfg::kBatL1V) t_l1 = s.t_s;
		if (t_l3v < 0 && s.battery_v < cfg::kBatL3V) t_l3v = s.t_s;
		if (t_auto < 0 && s.fw_mode == 4) t_auto = s.t_s;
		if (t_disarm < 0 && !s.armed) t_disarm = s.t_s;
		if (t_ground < 0 && t_auto >= 0 && s.pos[2] < 0.05) t_ground = s.t_s;
		if (k % cfg::kBatCsvEveryMs == 0)
			csv.row("%.2f,%.4f,%.4f,%.4f,%.4f,%.3f,%d,%d,%d", s.t_s, s.battery_v, s.hover_thrust, throttle_stick(s),
			        (s.motors[0] + s.motors[1] + s.motors[2] + s.motors[3]) / 4, s.pos[2], s.fw_mode, s.armed, s.exec_state);
		if (t_disarm >= 0 || (t_ground >= 0 && s.t_s > t_ground + 2.0)) break;
	}
	r.line("  起始 SoC %s（参数文件）；标称续航 %s min；上电电压 %.3f V，READY（t=%.2f s）时 %.3f V、悬停油门杆 %.3f",
	       soc_init < 0 ? "未知" : fmt_or(soc_init).c_str(), endurance_min < 0 ? "未知" : fmt_or(endurance_min).c_str(), v0, s_ready.t_s, s_ready.battery_v, s_ready.hover_thrust);
	r.line("  结束 t=%.1f s：电压 %.3f V（最低 %.3f），悬停油门杆 %.3f，z=%.3f（最高 %.3f），fw_mode=%d armed=%d exec_state=%d",
	       s.t_s, s.battery_v, v_min, s.hover_thrust, s.pos[2], z_max, s.fw_mode, s.armed, s.exec_state);
	r.line("  V_bus 首次 < %.1f V（L1 电平）t=%s s；首次 < %.1f V（L3 电平）t=%s s；固件进入 AUTO 降落 t=%s s；落地 t=%s s；上锁 t=%s s",
	       cfg::kBatL1V, fmt_or(t_l1).c_str(), cfg::kBatL3V, fmt_or(t_l3v).c_str(), fmt_or(t_auto).c_str(), fmt_or(t_ground).c_str(), fmt_or(t_disarm).c_str());
	r.line("  （L1 在飞行中只阻止解锁/怠速上锁，悬停时固件不会因 L1 动作；要测 L1 需要能设起始 SoC 再解锁）");
	char what[256];
	std::snprintf(what, sizeof what, "母线电压随放电下降（%.3f → %.3f V）", s_ready.battery_v, s.battery_v);
	r.check(s.battery_v < s_ready.battery_v, what);
	std::snprintf(what, sizeof what, "悬停油门杆随电压下降而上升（%.3f → %.3f）", s_ready.hover_thrust, s.hover_thrust);
	r.check(s.hover_thrust > s_ready.hover_thrust, what);
	std::snprintf(what, sizeof what, "固件 L3 自动降落在 %.0f s 内触发（t=%s s）", cfg::kBatMaxS, fmt_or(t_auto).c_str());
	r.check(t_auto >= 0, what);
	if (params != c.params) cf_sitl_init(c.params.c_str(), c.seed);   // 恢复原参数
}
