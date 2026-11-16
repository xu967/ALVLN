#pragma once
// VLN 动态导航条件编码（15 维 FiLM one-hot）与第 5 维独立 STOP 头（二分类 BCE + 验证集阈值选择）。
// 编码单字节 code = prev_action + 4 * bearing_bin + 36 * near_goal ∈ [0, 71]，
// 展开为 [prev_action(4) | bearing_bin(9, 8=UNKNOWN) | near_goal(2)] 共 15 维。
#include "uavrgb/action.hpp"
#include "uavrgb/tensor.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb {

inline constexpr int kActionClasses = kNumActions;  // 4 个离散机动动作（0..3）
inline constexpr int kActionStopClasses = 5;        // 前 4 维机动动作 + 第 5 维独立 stop_logit
inline constexpr int kVlnNavDim = 15;               // 4 + 9 + 2
inline constexpr int kVlnNavCodes = 72;             // 4 * 9 * 2
inline constexpr int kBearingUnknown = 8;           // 视野未见目标 / 搜索态
inline constexpr float kDefaultStopThreshold = 0.5f;

inline std::uint8_t encode_vln_nav(int prev_action, int bearing_bin, int near_goal) {
  if (prev_action < 0 || prev_action >= kActionClasses || bearing_bin < 0 || bearing_bin > 8 ||
      (near_goal != 0 && near_goal != 1)) {
    throw std::invalid_argument("vln_nav: 条件越界（prev=" + std::to_string(prev_action) +
                                ", bearing=" + std::to_string(bearing_bin) +
                                ", near=" + std::to_string(near_goal) + "）");
  }
  return static_cast<std::uint8_t>(prev_action + 4 * bearing_bin + 36 * near_goal);
}

inline void decode_vln_nav(std::uint8_t code, int* prev_action, int* bearing_bin, int* near_goal) {
  if (code >= kVlnNavCodes) {
    throw std::invalid_argument("vln_nav: 编码字节越界 " + std::to_string(code));
  }
  *prev_action = static_cast<int>(code % 4u);
  *bearing_bin = static_cast<int>((code / 4u) % 9u);
  *near_goal = static_cast<int>(code / 36u);
}

inline std::uint8_t flip_vln_nav_code(std::uint8_t code) {
  int pa = 0, bb = 0, ng = 0;
  decode_vln_nav(code, &pa, &bb, &ng);
  if (pa == static_cast<int>(Action::YAW_LEFT)) {
    pa = static_cast<int>(Action::YAW_RIGHT);
  } else if (pa == static_cast<int>(Action::YAW_RIGHT)) {
    pa = static_cast<int>(Action::YAW_LEFT);
  }
  if (bb < kBearingUnknown) bb = (8 - bb) % 8;  // 逆时针 8 桶左右镜像：0->0, 1<->7, 2<->6, 3<->5, 4->4
  return encode_vln_nav(pa, bb, ng);
}

// 切出前 4 列动作 logits [N, 4]（若输入已是 [N, 4] 直接返回副本）。
Tensor slice_action_logits(const Tensor& logits);

// 将动作梯度 [N, 4] 与第 5 列 stop 梯度 [N] 拼回 [N, 5]。
Tensor merge_action_stop_grad(const Tensor& action_grad, const std::vector<float>& stop_grad);

// 训练集自动正例权重：clamp(neg / max(pos, 1), 1.0, 20.0)。
float auto_stop_pos_weight(const std::vector<std::uint8_t>& stop_targets,
                           const std::vector<std::uint32_t>& train_idx);

// 对 logits[:, 4] 计算带正例权重的稳定 Sigmoid BCE 损失（按 N 平均）；dstop 非空时写入对第 5 列 logit 的梯度。
float stop_bce_loss_and_grad(const Tensor& logits, const std::vector<std::uint8_t>& stop_y,
                             float pos_weight, float stop_lambda, std::vector<float>* dstop);

struct StopEvalMetrics {
  double bce = -1.0;
  double f1 = -1.0;
  double precision = -1.0;
  double recall = -1.0;
  float best_threshold = kDefaultStopThreshold;
};

// 从验证集 stop_logit 扫描阈值 {0.20..0.85} 选最大 F1（平局优先贴近 0.50）；测试集固定用 given_threshold 评估。
StopEvalMetrics evaluate_stop_head(const std::vector<float>& stop_logits,
                                   const std::vector<std::uint8_t>& stop_y,
                                   bool select_threshold,
                                   float given_threshold = kDefaultStopThreshold);

}  // namespace uavrgb
