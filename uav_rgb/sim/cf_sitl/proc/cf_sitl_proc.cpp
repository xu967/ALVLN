// cf_sitl_proc：把 libcf_sitl 包成独立进程，stdin 每行一个 JSON 请求、stdout 每行一个 JSON 应答
// （协议见 ../README.md 与 phase2 BRIEF §4）。3DGS 的 Python 侧经管道调用它，不链接 GPLv3 库。
// 用法：cf_sitl_proc --params sim/dynamics/e1_mini.json [--seed N]
// fd 1 在启动后改道到 stderr，应答只走 dup 出来的描述符：固件/库里任何杂散 printf 都进不了协议流。
#include "cf_sitl.h"
#include "../src/json_min.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <type_traits>
#include <unistd.h>

namespace {

Json arr(const double *v, int n) {
	Json a = Json::array();
	for (int i = 0; i < n; i++) a.push(Json::number(v[i]));
	return a;
}

// cf_sitl_state 末尾的 sticks[4]（执行器输出杆量）是后加的字段：有就输出，没有也能编
template <class T, class = void> struct has_sticks : std::false_type {};
template <class T> struct has_sticks<T, std::void_t<decltype(std::declval<T>().sticks)>> : std::true_type {};
template <class S> void add_sticks(Json &j, const S &s) {
	if constexpr (has_sticks<S>::value) j.set("sticks", arr(s.sticks, 4));
	else (void)j, (void)s;
}

// cf_sitl_state → JSON，键名与字段顺序同 include/cf_sitl.h
Json state_json() {
	cf_sitl_state s;
	std::memset(&s, 0, sizeof s);
	cf_sitl_get_state(&s);
	Json j = Json::object();
	j.set("t_s", Json::number(s.t_s));
	j.set("exec_state", Json::number(s.exec_state));
	j.set("fw_mode", Json::number(s.fw_mode));
	j.set("armed", Json::number(s.armed));
	j.set("action", Json::number(s.action));
	j.set("action_until_s", Json::number(s.action_until_s));
	j.set("motors", arr(s.motors, 4));
	j.set("battery_v", Json::number(s.battery_v));
	j.set("att_est", arr(s.att_est, 4));
	j.set("tof_m", Json::number(s.tof_m));
	j.set("flow_vel", arr(s.flow_vel, 2));
	j.set("height_est_m", Json::number(s.height_est_m));
	j.set("vel_est", arr(s.vel_est, 2));
	j.set("hover_thrust", Json::number(s.hover_thrust));
	j.set("pos", arr(s.pos, 3));
	j.set("quat", arr(s.quat, 4));
	j.set("vel", arr(s.vel, 3));
	add_sticks(j, s);
	return j;
}

Json fail(const std::string &why) {
	Json j = Json::object();
	j.set("ok", Json::boolean(false));
	j.set("error", Json::string(why));
	return j;
}
Json ok_with_state() {
	Json j = Json::object();
	j.set("ok", Json::boolean(true));
	j.set("state", state_json());
	return j;
}

// 取数字字段；缺失时用默认值（required=false），类型不对一律报错
bool num_field(const Json &req, const char *key, double &out, std::string &err, bool required, double def = 0) {
	const Json *v = req.get(key);
	if (!v) {
		if (required) { err = std::string("缺字段 ") + key; return false; }
		out = def;
		return true;
	}
	if (!v->is_num()) { err = std::string("字段 ") + key + " 不是数字"; return false; }
	out = v->num;
	return true;
}
bool vec_field(const Json &req, const char *key, double *out, int n, std::string &err) {
	const Json *v = req.get(key);
	if (!v) return true;   // 缺失保持默认
	if (v->type != Json::Array || (int)v->arr.size() != n) { err = std::string("字段 ") + key + " 应为长度 " + std::to_string(n) + " 的数组"; return false; }
	for (int i = 0; i < n; i++) {
		if (!v->arr[i].is_num()) { err = std::string("字段 ") + key + " 含非数字"; return false; }
		out[i] = v->arr[i].num;
	}
	return true;
}

// 独立积分推进 n 步；库报错时返回 false
bool step_n(int n, std::string &err) {
	const int rc = cf_sitl_step_standalone(n);
	if (rc != 0) { err = "cf_sitl_step_standalone 返回 " + std::to_string(rc); return false; }
	return true;
}

Json handle(const Json &req, bool &quit) {
	const Json *cmd = req.get("cmd");
	if (req.type != Json::Object || !cmd || cmd->type != Json::String) return fail("请求必须是含字符串字段 cmd 的对象");
	const std::string &c = cmd->str;
	std::string err;
	if (c == "quit") { quit = true; Json j = Json::object(); j.set("ok", Json::boolean(true)); return j; }
	if (c == "state") return ok_with_state();
	if (c == "reset") {
		double episode = 0, yaw = 0, spawn[3] = {0, 0, 0};
		if (!num_field(req, "episode", episode, err, false) || !num_field(req, "yaw", yaw, err, false) ||
		    !vec_field(req, "spawn", spawn, 3, err)) return fail(err);
		if (episode < 0 || episode != std::floor(episode)) return fail("episode 必须是非负整数");
		cf_sitl_reset((uint64_t)episode, spawn, yaw);
		return ok_with_state();
	}
	if (c == "takeoff") {
		double height = 0, max_s = 0;
		if (!num_field(req, "height", height, err, true) || !num_field(req, "max_s", max_s, err, false, 15.0)) return fail(err);
		if (!(height > 0) || !(max_s > 0)) return fail("height 与 max_s 必须为正");
		cf_sitl_state s;
		cf_sitl_get_state(&s);
		const double t_end = s.t_s + max_s;
		cf_sitl_arm_takeoff(height);
		// 每 10 ms 看一次状态；READY 或超时退出。到 READY 后不再多走。
		for (;;) {
			cf_sitl_get_state(&s);
			if (s.exec_state == CF_SITL_READY || s.t_s >= t_end - 1e-9) break;
			if (!step_n(10, err)) return fail(err);
		}
		Json j = ok_with_state();
		j.set("ready", Json::boolean(s.exec_state == CF_SITL_READY));
		return j;
	}
	if (c == "act") {
		double action = 0, until_s = 0;
		if (!num_field(req, "action", action, err, true) || !num_field(req, "until_s", until_s, err, true)) return fail(err);
		if (action < 0 || action > 3 || action != std::floor(action)) return fail("action 必须是 0..3");
		cf_sitl_set_action((int)action, until_s);
		Json j = Json::object();
		j.set("ok", Json::boolean(true));
		return j;
	}
	if (c == "step") {
		double n = 0;
		if (!num_field(req, "n", n, err, true)) return fail(err);
		if (n < 1 || n != std::floor(n) || n > 1e7) return fail("n 必须是 1..1e7 的整数");
		if (!step_n((int)n, err)) return fail(err);
		return ok_with_state();
	}
	return fail("未知命令 " + c);
}

// 应答只走协议 fd，write 被信号打断或短写时续写
void send(int fd, std::string line) {
	line += '\n';
	size_t off = 0;
	while (off < line.size()) {
		const ssize_t w = write(fd, line.data() + off, line.size() - off);
		if (w < 0) { if (errno == EINTR) continue; std::exit(3); }
		off += (size_t)w;
	}
}

}  // namespace

int main(int argc, char **argv) {
	const char *params = nullptr;
	uint64_t seed = 0;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		if (a == "--params" && i + 1 < argc) params = argv[++i];
		else if (a == "--seed" && i + 1 < argc) seed = std::strtoull(argv[++i], nullptr, 10);
		else { std::fprintf(stderr, "用法: %s --params <e1_mini.json> [--seed N]\n", argv[0]); return 2; }
	}
	if (!params) { std::fprintf(stderr, "用法: %s --params <e1_mini.json> [--seed N]\n", argv[0]); return 2; }
	const int proto_fd = dup(1);
	if (proto_fd < 0 || dup2(2, 1) < 0) { std::perror("dup"); return 3; }
	if (cf_sitl_init(params, seed) != 0) {
		send(proto_fd, fail(std::string("cf_sitl_init 失败: ") + params).dump());
		return 1;
	}
	std::string line;
	while (std::getline(std::cin, line)) {
		Json req;
		std::string err;
		bool quit = false;
		Json resp = line.empty() ? fail("空请求") : (Json::parse(line, req, err) ? handle(req, quit) : fail("JSON 解析失败: " + err));
		send(proto_fd, resp.dump());
		if (quit) return 0;
	}
	return 0;   // stdin 关闭等同 quit
}
