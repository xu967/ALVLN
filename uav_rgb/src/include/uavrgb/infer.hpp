#pragma once
// 常驻推理循环：checkpoint 固定，按帧从输入流收 RGB uint8，向输出流写一行 JSON。
// 不碰 Gazebo，也不做图像缩放；输入必须是模型尺寸的 HWC RGB。
#include "uavrgb/model.hpp"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <vector>

namespace uavrgb {

inline constexpr int kRgbChannels = 3;

// 每步读 (in_c/3) 张 in_h*in_w*3 的 HWC RGB（双帧模型：先当前帧、后上一帧；首步由调用方把
// 当前帧发两遍），prev_dim==4 时再读 1 字节动作 id，prev_dim==15 时读 3 字节 (prev_action, bearing_bin, near_goal)；
// 写 {"action":..,"probs":[..],"infer_ms":..}（5 维输出头时追加 "stop_prob" 与 "stop_requested"）。
std::size_t run_infer_loop(const CnnModel& model, std::istream& in, std::ostream& out,
                           std::vector<std::uint8_t> text_ids = {},
                           float stop_threshold = 0.5f);

}  // namespace uavrgb
