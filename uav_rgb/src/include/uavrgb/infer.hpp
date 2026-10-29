#pragma once
// 常驻推理循环：checkpoint 固定，按帧从输入流收 RGB uint8，向输出流写一行 JSON。
// 不碰 Gazebo，也不做图像缩放；输入必须是模型尺寸的 HWC RGB。
#include "uavrgb/model.hpp"

#include <cstddef>
#include <iosfwd>

namespace uavrgb {

// 每帧读 in_h*in_w*in_c 字节，写 {"action":..,"probs":[..],"infer_ms":..}。
// 干净 EOF 返回处理帧数；截断帧/写失败抛 std::runtime_error。
std::size_t run_infer_loop(const CnnModel& model, std::istream& in, std::ostream& out);

}  // namespace uavrgb
