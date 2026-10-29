#pragma once
// uav_rgb 四动作高层协议。模型只输出离散安全动作，绝不输出 PWM/转速/姿态；
// 具体语义由 ESP32 飞控定义并保留最终否决权（见 UBUNTU_BASE_GUIDE_V1 §0.1）。
#include <cstdint>

namespace uavrgb {

enum class Action : std::uint8_t {
  SAFE_NEUTRAL = 0,  // 安全中立/失效处理；具体行为由 ESP32 定义
  FORWARD_SLOW = 1,  // 低速短时前进
  YAW_LEFT = 2,      // 左偏航
  YAW_RIGHT = 3,     // 右偏航
};
inline constexpr int kNumActions = 4;

// ActionAdapter：把专家连续速度/偏航行为量化为四类短动作。
struct ActionAdapter {
  // forward_speed >= 0（前进标量），yaw_rate 带符号（左正/右负由调用方约定）。
  // 阈值对应 100–200ms 短指令语义；低于阈值回退 SAFE_NEUTRAL。
  static Action quantize(float forward_speed, float yaw_rate,
                         float fwd_eps = 0.1f, float yaw_eps = 0.05f);
};

}  // namespace uavrgb
