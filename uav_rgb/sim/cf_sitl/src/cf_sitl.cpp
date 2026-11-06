// cf_sitl.cpp：include/cf_sitl.h 的实现——把固件翻译单元（fw/）、执行器（exec/）、电机/电池/刚体（plant.h）、
// 传感器（sensors.h）串成每 1 ms 一步的锁步仿真。两种模式共用同一条 tick()：
//   外部模式（Gazebo）：刚体状态由插件传入，返回机体系 wrench；独立模式（3DGS/自检）：内部 Body 积分。
// 一个进程一个实例（固件是全局变量写法）。只导出 cf_sitl_*（-fvisibility=hidden + 下面的 pragma）。
#pragma GCC visibility push(default)
#include "cf_sitl.h"
#pragma GCC visibility pop
#include "plant.h"
#include "sensors.h"
#include "json_min.h"
#include "../fw/cf_fw.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {
using namespace sitl;

struct Sitl {
	bool inited = false; uint64_t seed = 0;
	Params P; Rng rng;
	Battery bat; Motors mot; Body body; Imu imu; Tof tof; Flow flow; // 都只持有 P/rng 的引用，P 在 init 时整体赋值
	cf_exec exec{}; CfFwState fw{};
	uint64_t k = 0; double t = 0;                 // 步序号与最近一次固件 loop 的 SITL 时间
	bool used_external = false, used_standalone = false; double last_ext_t = -1;
	double tof_m = -1; int tof_new = 0; double flow_v[2] = {0, 0}; int flow_valid = 0, flow_new = 0;
	Vec3 ext_pos = Vec3::Zero(), ext_vel = Vec3::Zero(); Quat ext_q = Quat::Identity(); // 外部模式最近传入
	bool raw_on = false; float raw[4] = {0, 0, 0, 0}; // cf_sitl_set_sticks_raw：旁路执行器的调试杆量
	Sitl() : bat(P), mot(P), body(P), imu(P, rng), tof(P, rng), flow(P, rng) {}
};
Sitl *g = nullptr;

// 一步：传感器 → 执行器 → 固件 loop → 电机/电池 → wrench（机体系，含阻力，不含重力）
void tick(const Vec3 &pos, const Quat &q, const Vec3 &vel_w, const Vec3 &w_body, const Vec3 &sf_body, Wrench *out) {
	Sitl &s = *g; const Params &P = s.P;
	float gy[3], ac[3];
	s.imu.sample(w_body, sf_body, s.mot.mean_speed(), gy, ac);
	fw_set_imu(gy, ac);
	fw_set_battery((float)s.bat.v_bus);
	s.tof_new = (s.k % (uint64_t)P.tof_steps) == 0;
	if (s.tof_new) s.tof_m = s.tof.sample(pos, q);
	s.flow_new = (s.k % (uint64_t)P.flow_steps) == 0;
	if (s.flow_new) s.flow_valid = s.flow.sample(pos, q, vel_w, w_body, s.flow_v);

	cf_exec_input in{};           // 执行器只看固件上一 loop 的估计（真机里也是上一周期的值）
	in.t_s = (float)s.t; in.fw_armed = s.fw.armed;
	for (int i = 0; i < 4; i++) in.att_q[i] = s.fw.att_q[i];
	for (int i = 0; i < 3; i++) in.gyro[i] = s.fw.rates[i];
	in.tof_m = (float)s.tof_m; in.tof_new = s.tof_new;
	in.flow[0] = (float)s.flow_v[0]; in.flow[1] = (float)s.flow_v[1]; in.flow_valid = s.flow_valid; in.flow_new = s.flow_new;
	cf_exec_tick(&s.exec, &in);   // 旁路时执行器照常跑，只是输出不送固件，取消旁路后能接着接管
	const cf_exec_sticks &st = s.exec.sticks;
	fw_set_time_us((uint64_t)std::llround(s.t * 1e6));
	if (s.raw_on) fw_set_sticks(s.raw[0], s.raw[1], s.raw[2], s.raw[3], NAN);
	else fw_set_sticks(st.roll, st.pitch, st.yaw, st.throttle, NAN); // 每 tick 一帧，刷新固件 controlTime
	fw_loop();
	fw_get_state(&s.fw);

	Wrench w = s.mot.step(s.fw.duty, s.bat.v_bus, P.dt);
	s.bat.step(s.mot.thrust, P.dt);
	w.force += drag_body(P, q.conjugate() * vel_w);
	*out = w;
}

bool load_params(const char *path, Params &out) {
	std::ifstream f(path);
	if (!f) { std::fprintf(stderr, "cf_sitl: 无法打开参数文件 %s\n", path); return false; }
	std::stringstream ss; ss << f.rdbuf();
	Json j; std::string err;
	if (!Json::parse(ss.str(), j, err)) { std::fprintf(stderr, "cf_sitl: 参数文件 %s 解析失败：%s\n", path, err.c_str()); return false; }
	try {
		out = Params::load(j);
	} catch (const std::exception &e) {
		std::fprintf(stderr, "cf_sitl: 参数文件 %s：%s\n", path, e.what());
		return false;
	}
	if (std::fabs(out.dt - 0.001) > 1e-12) { std::fprintf(stderr, "cf_sitl: sim.dt_s 必须是 0.001\n"); return false; }
	return true;
}
} // namespace

extern "C" {

int cf_sitl_init(const char *params_json, uint64_t seed) {
	Params P;
	if (!load_params(params_json, P)) return 1;
	if (!g) g = new Sitl();
	g->P = P; g->seed = seed; g->inited = true;
	fw_set_verbose(std::getenv("CF_SITL_VERBOSE") != nullptr);
	const double spawn[3] = {0, 0, P.ground_z};
	cf_sitl_reset(0, spawn, 0);
	return 0;
}

void cf_sitl_reset(uint64_t episode, const double spawn_pos[3], double spawn_yaw) {
	if (!g || !g->inited) return;
	Sitl &s = *g;
	s.rng.seed(s.seed ^ episode);
	s.k = 0; s.t = 0; s.used_external = s.used_standalone = false; s.last_ext_t = -1;
	s.tof_m = -1; s.tof_new = 0; s.flow_v[0] = s.flow_v[1] = 0; s.flow_valid = s.flow_new = 0;
	s.bat.reset(); s.mot.reset(); s.imu.reset();
	s.body.reset(spawn_pos ? Vec3(spawn_pos[0], spawn_pos[1], spawn_pos[2]) : Vec3(0, 0, s.P.ground_z), spawn_yaw);
	s.ext_pos = s.body.pos; s.ext_vel.setZero(); s.ext_q = s.body.q;
	fw_reset();                 // 固件全局变量回上电状态 + setup()，SITL 时钟归零
	fw_get_state(&s.fw);
	cf_exec_init(&s.exec, &s.P.exec);
}

void cf_sitl_arm_takeoff(double target_height_m) {
	if (g && g->inited) cf_exec_arm_takeoff(&g->exec, (float)target_height_m);
}

void cf_sitl_set_action(int action, double until_s) {
	if (g && g->inited) cf_exec_set_action(&g->exec, action, (float)until_s);
}

void cf_sitl_set_sticks_raw(double roll, double pitch, double yaw, double throttle) {
	if (!g || !g->inited) return;
	if (std::isnan(roll)) { g->raw_on = false; return; }
	g->raw_on = true;
	g->raw[0] = (float)roll; g->raw[1] = (float)pitch; g->raw[2] = (float)yaw; g->raw[3] = (float)throttle;
}

int cf_sitl_step_external(const cf_sitl_body_in *in, cf_sitl_wrench *out) {
	std::memset(out, 0, sizeof *out);
	if (!g || !g->inited) return 1;
	Sitl &s = *g; const double dt = s.P.dt;
	if (s.used_standalone) return 3; // 同一集里不能混用两种模式
	if (in->t_s < 0 || (s.last_ext_t >= 0 && std::fabs(in->t_s - s.last_ext_t - dt) > 1e-6)) return 2;
	s.used_external = true; s.last_ext_t = in->t_s; s.t = in->t_s; s.k = (uint64_t)std::llround(in->t_s / dt);
	const Quat q = Quat(in->quat[0], in->quat[1], in->quat[2], in->quat[3]).normalized();
	const Vec3 pos(in->pos[0], in->pos[1], in->pos[2]), vel(in->vel[0], in->vel[1], in->vel[2]);
	const Vec3 omega_w(in->omega[0], in->omega[1], in->omega[2]), acc_w(in->acc[0], in->acc[1], in->acc[2]);
	const Vec3 w_body = q.conjugate() * omega_w, sf_body = q.conjugate() * (acc_w + Vec3(0, 0, G));
	s.ext_pos = pos; s.ext_vel = vel; s.ext_q = q;
	Wrench w;
	tick(pos, q, vel, w_body, sf_body, &w);
	for (int i = 0; i < 3; i++) { out->force[i] = w.force[i]; out->torque[i] = w.torque[i]; }
	return 0;
}

int cf_sitl_step_standalone(int n_steps) {
	if (!g || !g->inited) return 1;
	Sitl &s = *g;
	if (s.used_external) return 3;
	s.used_standalone = true;
	for (int i = 0; i < n_steps; i++) {
		s.k++; s.t = (double)s.k * s.P.dt;
		Wrench w;
		tick(s.body.pos, s.body.q, s.body.vel, s.body.w, s.body.sf_body, &w);
		s.body.step(w, s.P.dt);
	}
	return 0;
}

void cf_sitl_get_state(cf_sitl_state *o) {
	std::memset(o, 0, sizeof *o);
	if (!g || !g->inited) return;
	const Sitl &s = *g;
	o->t_s = s.t; o->exec_state = s.exec.state; o->fw_mode = s.fw.mode; o->armed = s.fw.armed;
	o->action = s.exec.state == CF_EXEC_ACTING ? s.exec.action : -1;
	o->action_until_s = s.exec.action_until_s;
	for (int i = 0; i < 4; i++) { o->motors[i] = s.fw.motors[i]; o->att_est[i] = s.fw.att_q[i]; }
	o->battery_v = s.bat.v_bus;
	o->tof_m = s.tof_m;
	for (int i = 0; i < 2; i++) { o->flow_vel[i] = s.exec.flow_vel[i]; o->vel_est[i] = s.exec.vel_est[i]; }
	o->height_est_m = s.exec.height_est_m;
	o->hover_thrust = s.exec.hover_stick;
	const Vec3 &p = s.used_external ? s.ext_pos : s.body.pos, &v = s.used_external ? s.ext_vel : s.body.vel;
	const Quat &q = s.used_external ? s.ext_q : s.body.q;
	for (int i = 0; i < 3; i++) { o->pos[i] = p[i]; o->vel[i] = v[i]; }
	o->quat[0] = q.w(); o->quat[1] = q.x(); o->quat[2] = q.y(); o->quat[3] = q.z();
	o->sticks[0] = s.exec.sticks.roll; o->sticks[1] = s.exec.sticks.pitch; o->sticks[2] = s.exec.sticks.yaw; o->sticks[3] = s.exec.sticks.throttle;
}

} // extern "C"
