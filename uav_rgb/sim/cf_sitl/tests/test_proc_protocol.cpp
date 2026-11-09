// test_proc_protocol：起真的 cf_sitl_proc 进程做协议往返（BRIEF §4）：坏 JSON / 空行 / 未知命令 / 坏 action → ok=false；
// state / reset / act / step / takeoff → ok=true 且字段合理；quit 后进程退出码 0。参数文件不存在退出 77（SKIP）。
// 用法：test_proc_protocol <cf_sitl_proc> <e1_mini.json>
#include "../src/json_min.h"
#include "cf_sitl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

struct Proc {
	pid_t pid = -1; int to = -1, from = -1; std::string buf;
	bool start(const char *exe, const char *params) {
		int in[2], out[2];
		if (pipe(in) || pipe(out)) return false;
		pid = fork();
		if (pid < 0) return false;
		if (pid == 0) {
			dup2(in[0], 0); dup2(out[1], 1);
			close(in[0]); close(in[1]); close(out[0]); close(out[1]);
			execl(exe, exe, "--params", params, "--seed", "1", (char *)nullptr);
			_exit(127);
		}
		close(in[0]); close(out[1]);
		to = in[1]; from = out[0];
		return true;
	}
	bool send(const std::string &line) {
		const std::string l = line + "\n";
		return write(to, l.data(), l.size()) == (ssize_t)l.size();
	}
	bool recv(std::string &line) {   // 读到换行为止；EOF 返回 false
		for (;;) {
			const size_t nl = buf.find('\n');
			if (nl != std::string::npos) { line = buf.substr(0, nl); buf.erase(0, nl + 1); return true; }
			char tmp[4096];
			const ssize_t n = read(from, tmp, sizeof tmp);
			if (n <= 0) return false;
			buf.append(tmp, (size_t)n);
		}
	}
	// 一问一答；返回解析后的应答，失败时 type==Null
	Json call(const std::string &req) {
		Json j; std::string line, err;
		if (!send(req) || !recv(line)) { std::printf("    无应答: %s\n", req.c_str()); return j; }
		if (!Json::parse(line, j, err)) { std::printf("    应答不是 JSON: %s\n", line.c_str()); j = Json(); }
		return j;
	}
	int wait_exit() {
		close(to); int st = 0;
		waitpid(pid, &st, 0);
		return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
	}
};
static bool ok_of(const Json &j) { const Json *o = j.get("ok"); return o && o->type == Json::Bool && o->b; }
static bool is_fail(const Json &j) { const Json *e = j.get("error"); return j.get("ok") && !ok_of(j) && e && e->type == Json::String && !e->str.empty(); }
static double st_num(const Json &j, const char *key) { const Json *s = j.get("state"); const Json *v = s ? s->get(key) : nullptr; return v && v->is_num() ? v->num : NAN; }

int main(int argc, char **argv) {
	if (argc < 3) { std::fprintf(stderr, "用法: %s <cf_sitl_proc> <e1_mini.json>\n", argv[0]); return 2; }
	struct stat st;
	if (stat(argv[2], &st) != 0) { std::printf("SKIP: 参数文件不存在 %s\n", argv[2]); return 77; }
	Proc p;
	if (!p.start(argv[1], argv[2])) { std::printf("  [FAIL] 起不了 %s\n", argv[1]); return 1; }
	std::printf("== cf_sitl_proc 协议往返（%s）\n", argv[1]);

	Json r = p.call("garbage");
	CHECK(is_fail(r), "坏 JSON → ok=false + error");
	r = p.call("");
	CHECK(is_fail(r), "空行 → ok=false");
	r = p.call("{\"cmd\":\"nope\"}");
	CHECK(is_fail(r), "未知命令 → ok=false");
	r = p.call("[1,2]");
	CHECK(is_fail(r), "非对象请求 → ok=false");
	r = p.call("{\"cmd\":\"state\"}");
	CHECK(ok_of(r) && r.get("state") && st_num(r, "t_s") == 0 && st_num(r, "exec_state") == 0, "state → ok=true，t_s=0，DISARMED");
	const Json *s = r.get("state");
	bool shape = s && s->get("pos") && s->get("pos")->arr.size() == 3 && s->get("quat") && s->get("quat")->arr.size() == 4 &&
	             s->get("motors") && s->get("motors")->arr.size() == 4 && s->get("att_est") && s->get("flow_vel") && s->get("vel_est") &&
	             s->get("fw_mode") && s->get("armed") && s->get("action") && s->get("action_until_s") && s->get("battery_v") &&
	             s->get("tof_m") && s->get("height_est_m") && s->get("hover_thrust") && s->get("vel");
	CHECK(shape, "state 含 cf_sitl_state 全部字段，数组长度正确");
	// 库版本（AUDIT_SIM 2026-11-09 #1）：Python 侧拿它拼 dynamics_profile 的 lib<N>，缺了就不能区分换库不换参数的数据
	CHECK(st_num(r, "lib_version") == CF_SITL_LIB_VERSION && CF_SITL_LIB_VERSION >= 31, "state.lib_version == CF_SITL_LIB_VERSION（>= 31）");
	r = p.call("{\"cmd\":\"reset\",\"episode\":1,\"spawn\":[0,0,0],\"yaw\":0.25}");
	CHECK(ok_of(r) && st_num(r, "t_s") == 0, "reset → ok=true，t_s 归零");
	r = p.call("{\"cmd\":\"reset\",\"episode\":-1}");
	CHECK(is_fail(r), "reset episode=-1 → ok=false");
	r = p.call("{\"cmd\":\"reset\",\"spawn\":[0,0]}");
	CHECK(is_fail(r), "reset spawn 长度错 → ok=false");
	r = p.call("{\"cmd\":\"act\",\"action\":9,\"until_s\":1}");
	CHECK(is_fail(r), "act action=9 → ok=false");
	r = p.call("{\"cmd\":\"act\",\"action\":1}");
	CHECK(is_fail(r), "act 缺 until_s → ok=false");
	r = p.call("{\"cmd\":\"step\",\"n\":0}");
	CHECK(is_fail(r), "step n=0 → ok=false");
	r = p.call("{\"cmd\":\"step\",\"n\":10}");
	CHECK(ok_of(r) && std::fabs(st_num(r, "t_s") - 0.010) < 1e-9, "step n=10 → ok=true，t_s=0.010");
	r = p.call("{\"cmd\":\"act\",\"action\":1,\"until_s\":5.0}");
	CHECK(ok_of(r), "act FORWARD → ok=true");
	r = p.call("{\"cmd\":\"takeoff\",\"height\":2.0,\"max_s\":15.0}");
	const Json *ready = r.get("ready");
	const double t_ready = st_num(r, "t_s");
	std::printf("    takeoff: ready=%s t_s=%.3f exec_state=%.0f z=%.3f\n", ready && ready->b ? "true" : "false", t_ready, st_num(r, "exec_state"),
	            r.get("state") && r.get("state")->get("pos") ? r.get("state")->get("pos")->arr[2].num : NAN);
	CHECK(ok_of(r) && ready && ready->type == Json::Bool && ready->b && st_num(r, "exec_state") == 2 && t_ready <= 15.01, "takeoff → ok=true，ready=true，exec_state=READY，≤ 15 s");
	r = p.call("{\"cmd\":\"takeoff\",\"height\":-1}");
	CHECK(is_fail(r), "takeoff height<0 → ok=false");
	r = p.call("{\"cmd\":\"quit\"}");
	CHECK(ok_of(r), "quit → ok=true");
	std::string extra;
	CHECK(!p.recv(extra), "quit 后 stdout 关闭（没有多余输出）");
	const int rc = p.wait_exit();
	CHECK(rc == 0, "quit 后进程退出码 0");

	// 第二个进程：stdin 直接 EOF 也应以 0 退出
	Proc q;
	if (q.start(argv[1], argv[2])) {
		std::string l;
		q.call("{\"cmd\":\"state\"}");
		CHECK(q.wait_exit() == 0, "stdin EOF（无 quit）退出码 0");
	}
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
