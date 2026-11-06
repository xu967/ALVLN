// test_external_vs_standalone：同一集分别用两种模式跑 arm_takeoff：A = 库内独立积分；B = 本测试自己积分刚体
// （质量/惯量/地面高度从 e1_mini.json 读，半隐式欧拉 + 硬地面，与 src/plant.h 的 Body 同一套方程），每步把
// 位姿/速度/角速度/加速度传给 cf_sitl_step_external，拿回 wrench 再积分。末态位置差必须 < 1 cm。
// 参数文件不存在或库没有独立积分模式 → 退出 77（SKIP）。用法：test_external_vs_standalone <e1_mini.json>
#include "cf_sitl.h"
#include "../src/json_min.h"

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

// ---- 配置区 ----
static const double kDt = 0.001;        // 库接口固定
static const double kHorizonS = 3.0;    // 3 s：含解锁手势、电机起转、离地爬升（smoke 里 1 s 内还没离地，1 s 对比只是在地面比零）
static const double kHeight = 2.0;      // BRIEF §4 目标高度
static const double kPosTolM = 0.01;    // 任务判据：位置差 < 1 cm
static const double kG = 9.80665;

using Vec3 = Eigen::Vector3d; using Quat = Eigen::Quaterniond;
static int g_fails = 0;
#define CHECK(cond, what) do { const bool _ok = (cond); std::printf("  [%s] %s\n", _ok ? "PASS" : "FAIL", what); if (!_ok) g_fails++; } while (0)

struct Body {   // 与 plant.h Body 同一套：机体系 wrench（含阻力不含重力）→ 世界加速度；grounded 时推力不够就钉在地上
	double mass, ground_z; Vec3 I;
	Vec3 pos = Vec3::Zero(), vel = Vec3::Zero(), w = Vec3::Zero(), acc_w = Vec3::Zero();
	Quat q = Quat::Identity(); bool grounded = true;
	void step(const cf_sitl_wrench &wr) {
		const Vec3 f(wr.force[0], wr.force[1], wr.force[2]), tau(wr.torque[0], wr.torque[1], wr.torque[2]);
		const Vec3 acc = (q * f) / mass - Vec3(0, 0, kG);
		const Vec3 vel_prev = vel;
		if (grounded && acc.z() <= 0) {
			vel.setZero(); w.setZero(); pos.z() = ground_z;
			const double yaw = std::atan2(2 * (q.w() * q.z() + q.x() * q.y()), 1 - 2 * (q.y() * q.y() + q.z() * q.z()));
			q = Quat(Eigen::AngleAxisd(yaw, Vec3::UnitZ()));
		} else {
			grounded = false;
			w += (tau - w.cross(I.cwiseProduct(w))).cwiseQuotient(I) * kDt;
			const double wn = w.norm();
			if (wn > 0) q = (q * Quat(Eigen::AngleAxisd(wn * kDt, w / wn))).normalized();
			vel += acc * kDt;
			pos += vel * kDt;
			if (pos.z() <= ground_z && vel.z() <= 0) { pos.z() = ground_z; vel.setZero(); w.setZero(); grounded = true; }
		}
		acc_w = (vel - vel_prev) / kDt;   // 传给库的「世界系线加速度，不含重力」
	}
};

static bool load(const char *path, Body &b) {
	std::ifstream f(path); std::stringstream ss; ss << f.rdbuf();
	Json j; std::string err;
	if (!Json::parse(ss.str(), j, err)) { std::printf("  [FAIL] 参数解析失败: %s\n", err.c_str()); return false; }
	const Json *m = j.get("mass_flight"), *in = j.get("inertia_diag"), *sim = j.get("sim");
	m = m ? m->get("value") : nullptr; in = in ? in->get("value") : nullptr; sim = sim ? sim->get("value") : nullptr;
	const Json *gz = sim ? sim->get("ground_z_m") : nullptr;
	if (!m || !m->is_num() || !in || in->arr.size() != 3 || !gz || !gz->is_num()) { std::printf("  [FAIL] 缺 mass_flight / inertia_diag / sim.ground_z_m\n"); return false; }
	b.mass = m->num; b.I = Vec3(in->arr[0].num, in->arr[1].num, in->arr[2].num); b.ground_z = gz->num;
	return true;
}

int main(int argc, char **argv) {
	if (argc < 2) { std::fprintf(stderr, "用法: %s <e1_mini.json>\n", argv[0]); return 2; }
	struct stat st;
	if (stat(argv[1], &st) != 0) { std::printf("SKIP: 参数文件不存在 %s\n", argv[1]); return 77; }
	Body b;
	if (!load(argv[1], b)) return 1;
	if (cf_sitl_init(argv[1], 5) != 0) { std::printf("  [FAIL] cf_sitl_init 失败\n"); return 1; }
	const int n = (int)std::lround(kHorizonS / kDt);
	const double spawn[3] = {0, 0, b.ground_z};

	// A：库内独立积分
	cf_sitl_reset(1, spawn, 0);
	cf_sitl_arm_takeoff(kHeight);
	if (cf_sitl_step_standalone(n) != 0) { std::printf("SKIP: 库没有独立积分模式（桩库）\n"); return 77; }
	cf_sitl_state a; std::memset(&a, 0, sizeof a); cf_sitl_get_state(&a);

	// B：外部模式，本测试积分
	cf_sitl_reset(1, spawn, 0);
	cf_sitl_arm_takeoff(kHeight);
	b.pos = Vec3(spawn[0], spawn[1], spawn[2]); b.vel.setZero(); b.w.setZero(); b.acc_w.setZero(); b.q = Quat::Identity(); b.grounded = true;
	int rc = 0;
	double last_in_z = 0;   // 最后一次传给库的 z：get_state 回显的是传入值，不是本测试随后积分出的新值
	for (int k = 1; k <= n && rc == 0; k++) {
		cf_sitl_body_in in{};
		in.t_s = k * kDt;
		last_in_z = b.pos.z();
		const Vec3 omega_w = b.q * b.w;
		for (int i = 0; i < 3; i++) { in.pos[i] = b.pos[i]; in.vel[i] = b.vel[i]; in.omega[i] = omega_w[i]; in.acc[i] = b.acc_w[i]; }
		in.quat[0] = b.q.w(); in.quat[1] = b.q.x(); in.quat[2] = b.q.y(); in.quat[3] = b.q.z();
		cf_sitl_wrench wr;
		rc = cf_sitl_step_external(&in, &wr);
		if (rc == 0) b.step(wr);
	}
	cf_sitl_state e; std::memset(&e, 0, sizeof e); cf_sitl_get_state(&e);
	std::printf("== %.1f s 对比：standalone z=%.4f pos=(%.4f,%.4f,%.4f) vel=(%.3f,%.3f,%.3f) armed=%d exec=%d\n",
	            kHorizonS, a.pos[2], a.pos[0], a.pos[1], a.pos[2], a.vel[0], a.vel[1], a.vel[2], a.armed, a.exec_state);
	std::printf("              external   z=%.4f pos=(%.4f,%.4f,%.4f) vel=(%.3f,%.3f,%.3f) armed=%d exec=%d（step_external rc=%d）\n",
	            b.pos.z(), b.pos.x(), b.pos.y(), b.pos.z(), b.vel.x(), b.vel.y(), b.vel.z(), e.armed, e.exec_state, rc);
	const double dpos = (b.pos - Vec3(a.pos[0], a.pos[1], a.pos[2])).norm();
	const double dvel = (b.vel - Vec3(a.vel[0], a.vel[1], a.vel[2])).norm();
	const Quat qa(a.quat[0], a.quat[1], a.quat[2], a.quat[3]);
	const double dang = 2 * std::acos(std::fmin(1.0, std::fabs(qa.dot(b.q)))) * 180 / M_PI;
	char what[160];
	CHECK(rc == 0, "cf_sitl_step_external 全程返回 0");
	CHECK(a.pos[2] > 0.05, "standalone 3 s 内已离地（否则对比只是在地面比零）");
	std::snprintf(what, sizeof what, "末态位置差 %.5f m < %.2f m", dpos, kPosTolM);
	CHECK(dpos < kPosTolM, what);
	std::snprintf(what, sizeof what, "末态速度差 %.5f m/s、姿态差 %.4f°（信息）", dvel, dang);
	CHECK(true, what);
	CHECK(e.pos[2] == last_in_z && e.armed == a.armed, "外部模式 get_state 回显最后传入的位姿，固件 armed 状态与 standalone 一致");
	std::printf("== 结果: %d 项失败\n", g_fails);
	return g_fails;
}
