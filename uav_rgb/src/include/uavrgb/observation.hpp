#pragma once
// ObservationSpec：学生在线唯一合法输入（RGB + timestamp + episode_id）。
// 禁止 GPS/深度/真值姿态/隐藏地图进入学生输入（UBUNTU_BASE_GUIDE_V1 §3）。
#include <cstdint>

namespace uavrgb {

// 模型输入尺寸。默认 160x120 = 仿真相机 320x240 的精确一半：保持 4:3 不失真，
// 像素数只比 128x128 多 17%，水平角分辨率却从 0.469°/px 改善到 0.375°/px
// （hfov=60° 时 10m 外 0.4m 立柱 4.9px → 6.1px）。
//
// 偏离 spec 的 128x128：320x240 直接拉伸到 1:1 会把画面横向压扁 2.5 倍，是失真不是缩放。
// 这是**模型超参数**，不是协议：改这里必须同时改训练数据（tools/preprocess.py --size）
// 和模型定义，三者必须一致；数据集头记录了真实 H/W，训练器按头读取。
inline constexpr int kObsW = 160;
inline constexpr int kObsH = 120;
inline constexpr int kObsC = 3;  // RGB, HWC uint8

struct SimObservation {
  std::uint8_t rgb[kObsH * kObsW * kObsC] = {};  // RGB 顺序（HWC），未归一化
  std::int64_t timestamp_ns = 0;                 // 与专家动作/碰撞同源时间戳
  std::int64_t episode_id = 0;
};

}  // namespace uavrgb
