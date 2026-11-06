// stub_cf_sitl.cpp：cf_sitl.h 的**测试桩**，只给 gz 插件在真库（libcf_sitl）就绪前跑通链路用。
// 不进正式构建路径；CMake 选项 CF_SITL_GZ_STUB=ON 时才编译。
// 行为：理想的速度/高度 PD 闭环（用真值，不模拟固件、电机、传感器噪声）：
//   悬停 = 按 mass_flight·g 的恒定升力 + 位置/速度修正；FORWARD = 机体前向 0.3 m/s；
//   YAW_LEFT/RIGHT = ±0.6 rad/s（左转绕 z 正向）；动作在 until_s 前有效，过期回悬停。
// 返回的 wrench 是机体系、作用在 link 原点、不含重力（与头文件约定一致）。
#include "cf_sitl.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>

namespace {
constexpr double kG = 9.80665;
constexpr double kDt = 0.001;
constexpr double kSpeedFwd = 0.3;
constexpr double kYawRate = 0.6;

struct Stub {
  double mass = 0.102, ixx = 7.5e-5, iyy = 8.9e-5, izz = 1.49e-4;
  double t = 0.0, last_t = -1.0;
  int exec = CF_SITL_DISARMED;
  int action = -1;
  double until = 0.0;
  double target_h = 0.0;
  double ready_since = -1.0;
  cf_sitl_body_in in{};
  double motors[4] = {0, 0, 0, 0};
} g;

// 只认 "key": {"value": <数字>} 的扁平项（草案 e1_mini_params_draft.json 的格式）。
bool ReadValue(const std::string& text, const char* key, double* out) {
  const std::regex re("\"" + std::string(key) + "\"\\s*:\\s*\\{[^{}]*?\"value\"\\s*:\\s*(-?[0-9][0-9.eE+-]*)");
  std::smatch m;
  if (!std::regex_search(text, m, re)) return false;
  *out = std::stod(m[1]);
  return true;
}

// 机体->世界四元数 (w,x,y,z) 旋转矩阵列向量；RotateReverse = 世界->机体。
void RotateReverse(const double q[4], const double v[3], double out[3]) {
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  // conj(q) * v * q
  out[0] = (1 - 2 * (y * y + z * z)) * v[0] + 2 * (x * y + w * z) * v[1] + 2 * (x * z - w * y) * v[2];
  out[1] = 2 * (x * y - w * z) * v[0] + (1 - 2 * (x * x + z * z)) * v[1] + 2 * (y * z + w * x) * v[2];
  out[2] = 2 * (x * z + w * y) * v[0] + 2 * (y * z - w * x) * v[1] + (1 - 2 * (x * x + y * y)) * v[2];
}

double Yaw(const double q[4]) {
  return std::atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
}
double Roll(const double q[4]) {
  return std::atan2(2 * (q[0] * q[1] + q[2] * q[3]), 1 - 2 * (q[1] * q[1] + q[2] * q[2]));
}
double Pitch(const double q[4]) {
  const double s = 2 * (q[0] * q[2] - q[3] * q[1]);
  return std::asin(std::fmax(-1.0, std::fmin(1.0, s)));
}
}  // namespace

extern "C" {

int cf_sitl_init(const char* params_json, uint64_t) {
  std::ifstream f(params_json);
  if (!f) {
    std::fprintf(stderr, "stub_cf_sitl: 无法打开 %s\n", params_json);
    return 1;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();
  if (!ReadValue(text, "mass_flight", &g.mass)) {
    std::fprintf(stderr, "stub_cf_sitl: %s 缺 mass_flight\n", params_json);
    return 2;
  }
  // inertia_diag 是数组，桩只需要量级；读不到就用草案默认值。
  const std::regex re("\"inertia_diag\"\\s*:\\s*\\{[^{}]*?\"value\"\\s*:\\s*\\[([^\\]]*)\\]");
  std::smatch m;
  if (std::regex_search(text, m, re)) {
    std::stringstream vs(m[1]);
    char comma;
    vs >> g.ixx >> comma >> g.iyy >> comma >> g.izz;
  }
  std::fprintf(stderr, "stub_cf_sitl: 桩库已加载（mass=%.4f kg, I=%.2e/%.2e/%.2e）；不是固件在环\n",
               g.mass, g.ixx, g.iyy, g.izz);
  return 0;
}

void cf_sitl_reset(uint64_t, const double*, double) {
  g.t = 0.0;
  g.last_t = -1.0;
  g.exec = CF_SITL_DISARMED;
  g.action = -1;
  g.until = 0.0;
  g.ready_since = -1.0;
  std::memset(g.motors, 0, sizeof(g.motors));
}

void cf_sitl_arm_takeoff(double target_height_m) {
  g.target_h = target_height_m;
  g.exec = CF_SITL_TAKEOFF;
}

void cf_sitl_set_action(int action, double until_s) {
  g.action = action;
  g.until = until_s;
}

int cf_sitl_step_external(const cf_sitl_body_in* in, cf_sitl_wrench* out) {
  std::memset(out, 0, sizeof(*out));
  if (g.last_t >= 0.0 && std::fabs(in->t_s - g.last_t - kDt) > 1e-6) return 1;
  g.last_t = in->t_s;
  g.t = in->t_s;
  g.in = *in;
  if (g.exec == CF_SITL_DISARMED) return 0;

  const bool active = g.action >= 0 && g.t < g.until;
  const int act = active ? g.action : 0;
  if (g.exec >= CF_SITL_READY) g.exec = active ? CF_SITL_ACTING : CF_SITL_READY;

  // 机体系速度与期望：FORWARD 前向 0.3 m/s，其余水平速度 0。
  double v_b[3];
  RotateReverse(in->quat, in->vel, v_b);
  const double vx_des = (act == 1) ? kSpeedFwd : 0.0;
  const double yaw = Yaw(in->quat);
  // 期望世界系加速度：水平按速度误差，垂直按高度 + 垂直速度误差，再加重力补偿。
  const double kv = 2.0, kz = 4.0, kvz = 3.0;
  const double ax_b = kv * (vx_des - v_b[0]), ay_b = kv * (0.0 - v_b[1]);
  const double a_w[3] = {ax_b * std::cos(yaw) - ay_b * std::sin(yaw),
                         ax_b * std::sin(yaw) + ay_b * std::cos(yaw),
                         kz * (g.target_h - in->pos[2]) + kvz * (0.0 - in->vel[2]) + kG};
  double f_w[3] = {g.mass * a_w[0], g.mass * a_w[1], g.mass * std::fmax(0.0, a_w[2])};
  RotateReverse(in->quat, f_w, out->force);

  // 姿态：roll/pitch 回零 + 阻尼；偏航按 ±0.6 rad/s 速率环。角速度头文件约定是世界系。
  double w_b[3];
  RotateReverse(in->quat, in->omega, w_b);
  const double wz_des = (act == 2) ? kYawRate : (act == 3) ? -kYawRate : 0.0;
  const double kp_att = 40.0, kd_att = 6.0, kp_yaw = 8.0;
  out->torque[0] = g.ixx * (-kp_att * Roll(in->quat) - kd_att * w_b[0]);
  out->torque[1] = g.iyy * (-kp_att * Pitch(in->quat) - kd_att * w_b[1]);
  out->torque[2] = g.izz * kp_yaw * (wz_des - w_b[2]);

  if (g.exec == CF_SITL_TAKEOFF && std::fabs(in->pos[2] - g.target_h) < 0.05 &&
      std::fabs(in->vel[2]) < 0.05) {
    if (g.ready_since < 0.0) g.ready_since = g.t;
    if (g.t - g.ready_since >= 0.3) g.exec = CF_SITL_READY;   // 稳住 0.3 s 才算 READY
  }
  const double thrust_norm = f_w[2] / (4.0 * 0.373);   // 单电机满推力 0.373 N（草案值）
  for (double& m : g.motors) m = std::fmax(0.0, std::fmin(1.0, thrust_norm));
  return 0;
}

int cf_sitl_step_standalone(int) {
  std::fprintf(stderr, "stub_cf_sitl: 桩库没有独立积分模式\n");
  return -1;
}

void cf_sitl_get_state(cf_sitl_state* out) {
  std::memset(out, 0, sizeof(*out));
  out->t_s = g.t;
  out->exec_state = g.exec;
  out->fw_mode = 2;   // STAB
  out->armed = g.exec != CF_SITL_DISARMED;
  out->action = (g.action >= 0 && g.t < g.until) ? g.action : -1;
  out->action_until_s = g.until;
  std::memcpy(out->motors, g.motors, sizeof(g.motors));
  out->battery_v = 3.82;
  std::memcpy(out->att_est, g.in.quat, sizeof(out->att_est));
  out->tof_m = (g.in.pos[2] >= 0.04 && g.in.pos[2] <= 2.5) ? g.in.pos[2] : -1.0;
  double v_b[3];
  RotateReverse(g.in.quat, g.in.vel, v_b);
  out->flow_vel[0] = out->vel_est[0] = v_b[0];
  out->flow_vel[1] = out->vel_est[1] = v_b[1];
  out->height_est_m = g.in.pos[2];
  out->hover_thrust = g.mass * kG;
  std::memcpy(out->pos, g.in.pos, sizeof(out->pos));
  std::memcpy(out->quat, g.in.quat, sizeof(out->quat));
  std::memcpy(out->vel, g.in.vel, sizeof(out->vel));
}

}  // extern "C"
