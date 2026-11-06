// CfSitlState.hh：cf_sitl_state -> gz.msgs.Double_V 展平（只依赖 gz-msgs，便于离线单测）。
#ifndef UAVRGB_CF_SITL_GZ_STATE_HH
#define UAVRGB_CF_SITL_GZ_STATE_HH

#include <cstdint>

#include <gz/msgs/double_v.pb.h>

#include "cf_sitl.h"

namespace uavrgb {

// 按 cf_sitl.h 原始字段顺序展平成 32 个值，追加 3 个插件诊断字段，再放 cf_sitl.h 后加的字段：
// [32] gz_sim_s（gz 仿真秒；Double_V 没有 header，订阅方用它当时间戳）
// [33] late_actions（到达时窗口已过期的动作累计数）  [34] step_errors（step_external 非 0 次数）
// [35..38] sticks roll/pitch/yaw/throttle（cf_sitl.h 2026-11-06 追加）
// 完整索引表见 README.md；改顺序必须同步改 README 与 tests/test_flatten_state.cc。
void FlattenState(const cf_sitl_state& st, double gz_sim_s, uint64_t late_actions,
                  uint64_t step_errors, gz::msgs::Double_V* out);
constexpr int kStateBaseFields = 32;
constexpr int kStateFields = kStateBaseFields + 3 + 4;

}  // namespace uavrgb

#endif  // UAVRGB_CF_SITL_GZ_STATE_HH
