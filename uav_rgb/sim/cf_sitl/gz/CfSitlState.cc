// CfSitlState.cc：cf_sitl_state -> gz.msgs.Double_V 的展平（字段顺序 = cf_sitl.h 声明顺序）。
// 索引表在 README.md，改这里必须同步改 README 和 tests/test_flatten_state.cc。
#include "CfSitlState.hh"

namespace uavrgb {

void FlattenState(const cf_sitl_state& st, double gz_sim_s, uint64_t late_actions,
                  uint64_t step_errors, gz::msgs::Double_V* out) {
  out->clear_data();
  auto add = [out](double v) { out->add_data(v); };
  add(st.t_s);                                   // [0]
  add(st.exec_state);                            // [1]
  add(st.fw_mode);                               // [2]
  add(st.armed);                                 // [3]
  add(st.action);                                // [4]
  add(st.action_until_s);                        // [5]
  for (double m : st.motors) add(m);             // [6..9]  RL RR FR FL
  add(st.battery_v);                             // [10]
  for (double q : st.att_est) add(q);            // [11..14] w x y z
  add(st.tof_m);                                 // [15]
  for (double v : st.flow_vel) add(v);           // [16,17]
  add(st.height_est_m);                          // [18]
  for (double v : st.vel_est) add(v);            // [19,20]
  add(st.hover_thrust);                          // [21]
  for (double p : st.pos) add(p);                // [22..24]
  for (double q : st.quat) add(q);               // [25..28] w x y z
  for (double v : st.vel) add(v);                // [29..31]
  add(gz_sim_s);                                 // [32] 插件：gz 仿真时间（秒）
  add(static_cast<double>(late_actions));        // [33] 插件：到达时已过期的动作数
  add(static_cast<double>(step_errors));         // [34] 插件：step_external 返回非 0 的次数
  // cf_sitl.h 2026-11-06 末尾追加的字段放在插件字段之后，保证 [32] 永远是 gz 时间。
  for (double s : st.sticks) add(s);             // [35..38] 杆量 roll pitch yaw ∈[-1,1], throttle ∈[0,1]
}

}  // namespace uavrgb
