// ActionAdapter::quantize 自检：无外部测试框架，返回 0 表示通过。
#include <cstdint>
#include <cstdio>

#include "uavrgb/action.hpp"

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    ++failures;
  }
}

}  // namespace

int main() {
  using uavrgb::Action;
  using uavrgb::ActionAdapter;

  // 枚举值契约：与 ESP32 协议编号一致。
  expect(static_cast<std::uint8_t>(Action::SAFE_NEUTRAL) == 0, "SAFE_NEUTRAL==0");
  expect(static_cast<std::uint8_t>(Action::FORWARD_SLOW) == 1, "FORWARD_SLOW==1");
  expect(static_cast<std::uint8_t>(Action::YAW_LEFT) == 2, "YAW_LEFT==2");
  expect(static_cast<std::uint8_t>(Action::YAW_RIGHT) == 3, "YAW_RIGHT==3");
  expect(uavrgb::kNumActions == 4, "kNumActions==4");

  // 量化规则：偏航主导，其次前进，否则安全中立。
  expect(ActionAdapter::quantize(1.0f, 0.0f) == Action::FORWARD_SLOW, "前进");
  expect(ActionAdapter::quantize(0.0f, 1.0f) == Action::YAW_LEFT, "左偏航");
  expect(ActionAdapter::quantize(0.0f, -1.0f) == Action::YAW_RIGHT, "右偏航");
  expect(ActionAdapter::quantize(0.0f, 0.0f) == Action::SAFE_NEUTRAL, "全零中立");
  // 低于默认阈值（fwd_eps=0.1, yaw_eps=0.05）应回退中立。
  expect(ActionAdapter::quantize(0.05f, 0.02f) == Action::SAFE_NEUTRAL, "低于阈值中立");
  // 偏航主导：即使前进也显著，偏航仍优先。
  expect(ActionAdapter::quantize(1.0f, 1.0f) == Action::YAW_LEFT, "偏航主导");

  return failures == 0 ? 0 : 1;
}
