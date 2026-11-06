// test_flatten_state.cc：FlattenState 的索引表必须与 README.md / cf_sitl.h 字段顺序一致。
// 每个字段填一个独一无二的值，再按 README 的索引逐个核对；长度 = 32 + 3 个插件字段。
#include <cstdio>
#include <cstdlib>

#include "CfSitlState.hh"

int main() {
  cf_sitl_state st{};
  double v = 100.0;
  st.t_s = v++;                                   // 100
  st.exec_state = CF_SITL_READY;                  // 2
  st.fw_mode = 2;
  st.armed = 1;
  st.action = 3;
  st.action_until_s = v++;                        // 101
  for (double& m : st.motors) m = v++;            // 102..105
  st.battery_v = v++;                             // 106
  for (double& q : st.att_est) q = v++;           // 107..110
  st.tof_m = v++;                                 // 111
  for (double& f : st.flow_vel) f = v++;          // 112,113
  st.height_est_m = v++;                          // 114
  for (double& e : st.vel_est) e = v++;           // 115,116
  st.hover_thrust = v++;                          // 117
  for (double& p : st.pos) p = v++;               // 118..120
  for (double& q : st.quat) q = v++;              // 121..124
  for (double& w : st.vel) w = v++;               // 125..127
  for (double& s : st.sticks) s = v++;            // 128..131

  gz::msgs::Double_V msg;
  uavrgb::FlattenState(st, 12.5, 7, 3, &msg);
  int fails = 0;
  auto expect = [&](int idx, double want, const char* name) {
    if (idx >= msg.data_size() || msg.data(idx) != want) {
      std::fprintf(stderr, "FAIL [%d] %s: got %s want %g\n", idx, name,
                   idx < msg.data_size() ? std::to_string(msg.data(idx)).c_str() : "<none>", want);
      ++fails;
    }
  };
  if (msg.data_size() != uavrgb::kStateFields) {
    std::fprintf(stderr, "FAIL size %d != %d\n", msg.data_size(), uavrgb::kStateFields);
    ++fails;
  }
  expect(0, 100, "t_s");
  expect(1, 2, "exec_state");
  expect(2, 2, "fw_mode");
  expect(3, 1, "armed");
  expect(4, 3, "action");
  expect(5, 101, "action_until_s");
  expect(6, 102, "motors[0] RL");
  expect(9, 105, "motors[3] FL");
  expect(10, 106, "battery_v");
  expect(11, 107, "att_est w");
  expect(14, 110, "att_est z");
  expect(15, 111, "tof_m");
  expect(16, 112, "flow_vel[0]");
  expect(17, 113, "flow_vel[1]");
  expect(18, 114, "height_est_m");
  expect(19, 115, "vel_est[0]");
  expect(20, 116, "vel_est[1]");
  expect(21, 117, "hover_thrust");
  expect(22, 118, "pos x");
  expect(24, 120, "pos z");
  expect(25, 121, "quat w");
  expect(28, 124, "quat z");
  expect(29, 125, "vel x");
  expect(31, 127, "vel z");
  expect(32, 12.5, "gz_sim_s");
  expect(33, 7, "late_actions");
  expect(34, 3, "step_errors");
  expect(35, 128, "sticks roll");
  expect(38, 131, "sticks throttle");
  if (fails) return 1;
  std::printf("test_flatten_state: PASS (%d fields)\n", msg.data_size());
  return 0;
}
