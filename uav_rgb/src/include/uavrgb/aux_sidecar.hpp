#pragma once
// UAVRAUX1 侧车（STUDENT-P2 双帧 / prev-action / 深度分带目标），格式见 tools/DATASET_FORMAT.md §6。
// 与 UAVSOFT1 同一套绑定方式：头里记 dataset.uavr 整文件 SHA256，不匹配直接拒绝。
// dataset.uavr 本身字节不变，旧读端（join_soft_targets / baseline_priors / 视频训练脚本）不受影响。
#include <cstdint>
#include <string>
#include <vector>

namespace uavrgb {

inline constexpr int kAuxHeaderBytes = 128;
inline constexpr std::uint8_t kAuxUnknown = 255;  // 深度目标的「未知」格（损失屏蔽）
inline constexpr int kAuxBandEdges = 7;            // 8 带 = 7 个上界

struct AuxSidecar {
  bool present = false;    // 没有侧车时其余字段全空；训练器据此拒绝 --two-frame/--prev-action
  bool has_depth = false;  // flags bit0；false 时训练器拒绝 --depth-aux
  int depth_h = 0, depth_w = 0;
  int depth_classes = 0;   // 分带数 K（8）
  std::vector<float> band_edges;         // K-1 个上界（m），严格递增
  float envelope_radius = 0.0f;          // 膨胀用的机体包络半径（m）
  std::vector<std::uint32_t> prev_index;  // [n] 同集 step-1 的样本下标；首帧 = 自身
  std::vector<std::uint8_t> prev_action;  // [n] step-1 执行的动作；首帧 = 0（SAFE）
  std::vector<std::uint8_t> depth;        // [n*depth_h*depth_w]，0..K-1 或 255；无深度时空
  std::vector<std::uint8_t> first_frame;  // [n] 1 = prev_index 指向自身（首帧/前一步被丢）
};

// 侧车路径 = uavr 路径换后缀 .aux（同目录多个 .uavr 时不会串）。
std::string aux_sidecar_path(const std::string& uavr_path);

// 读取并校验（N/H/W 与 uavr 头、两个 SHA256、payload 取值范围）；文件不存在返回 present=false，
// 存在但不自洽抛 std::runtime_error——半坏的侧车比没有更危险。
AuxSidecar load_aux_sidecar(const std::string& uavr_path, int n, int h, int w,
                            std::uint32_t num_classes);

}  // namespace uavrgb
