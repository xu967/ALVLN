#include "uavrgb/action.hpp"

#include <cmath>

namespace uavrgb {

Action ActionAdapter::quantize(float forward_speed, float yaw_rate,
                               float fwd_eps, float yaw_eps) {
  // 偏航主导：先判定转弯，避免低速小偏航被前进指令掩盖。
  if (std::fabs(yaw_rate) > yaw_eps) {
    return yaw_rate > 0 ? Action::YAW_LEFT : Action::YAW_RIGHT;
  }
  // 无显著偏航时才允许前进。
  if (forward_speed > fwd_eps) {
    return Action::FORWARD_SLOW;
  }
  return Action::SAFE_NEUTRAL;
}

}  // namespace uavrgb
