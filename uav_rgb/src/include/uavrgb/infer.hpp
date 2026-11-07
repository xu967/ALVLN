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
// 当前帧发两遍），模型接 prev-action 时再读 1 字节动作 id；写 {"action":..,"probs":[..],"infer_ms":..}。
// 干净 EOF 返回处理帧数；截断帧/写失败抛 std::runtime_error。
// text_ids 是当前 episode 的指令 token；模型没有文本分支时忽略（采集链路一律带指令）。
std::size_t run_infer_loop(const CnnModel& model, std::istream& in, std::ostream& out,
                           std::vector<std::uint8_t> text_ids = {});

}  // namespace uavrgb
