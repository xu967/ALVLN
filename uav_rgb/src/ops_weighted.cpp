// 加权 softmax 交叉熵（CPU FP32 参考）：按样本权重和归一化，公式/梯度见 ops.hpp。
// 权重全为 1 时与 ops_cpu.cpp 的无权版逐位一致（tests/test_weighted_loss.cpp 有断言）。
#include "uavrgb/ops.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "uavrgb/softmax_detail.hpp"

namespace uavrgb {
namespace {

const char* const kLossName = "softmax_ce_loss_weighted";
const char* const kGradName = "softmax_ce_grad_weighted";

void check_logits(const Tensor& logits, const char* who) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument(std::string(who) + ": logits 形状必须为 [N, C]");
  }
  if (logits.dims().at(0) == 0) throw std::invalid_argument(std::string(who) + ": 空 batch（N=0）");
  if (logits.dims().at(1) == 0) {
    throw std::invalid_argument(std::string(who) + ": C 必须为正（实际 logits 为 [N,0]）");
  }
}

// 校验 labels/weights 并返回 Σ_i w[y_i]（double）。weights 长度须为 C；
// 每类权重须有限且非负；Σ 为 0 时归一化无定义，直接拒绝。
double check_weighted(const std::vector<int>& labels, const std::vector<float>& weights,
                      std::int64_t n, std::int64_t c, const char* who) {
  if (static_cast<std::int64_t>(labels.size()) != n) {
    throw std::invalid_argument(std::string(who) + ": labels 数量与 logits 行数不一致");
  }
  if (static_cast<std::int64_t>(weights.size()) != c) {
    throw std::invalid_argument(std::string(who) + ": weights 长度应等于 C=" +
                                std::to_string(c) + "（实际 " +
                                std::to_string(weights.size()) + "）");
  }
  for (float w : weights) {
    if (!(w >= 0.0f) || !std::isfinite(w)) {
      throw std::invalid_argument(std::string(who) + ": weights 必须为非负有限值");
    }
  }
  double sum = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[static_cast<std::size_t>(i)];
    if (label < 0 || label >= c) throw std::invalid_argument(std::string(who) + ": label 越界");
    sum += static_cast<double>(weights[static_cast<std::size_t>(label)]);
  }
  if (!(sum > 0.0)) {
    throw std::invalid_argument(std::string(who) + ": weights 按样本求和为 0，无法归一化");
  }
  return sum;
}

}  // namespace

float softmax_ce_loss_weighted(const Tensor& logits, const std::vector<int>& labels,
                               const std::vector<float>& weights) {
  check_logits(logits, kLossName);
  const std::int64_t n = logits.dims().at(0);
  const std::int64_t c = logits.dims().at(1);
  const double wsum = check_weighted(labels, weights, n, c, kLossName);
  double total = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[static_cast<std::size_t>(i)];
    const float* row = logits.data() + i * c;
    float mx = 0.0f;
    float sum = 0.0f;
    detail::softmax_row(row, c, &mx, &sum);
    // 与无权版同一表达式；乘 w=1.0 是精确运算，故全 1 权重逐位一致。
    total += static_cast<double>(weights[static_cast<std::size_t>(label)]) *
             (-static_cast<double>(row[label] - mx) + std::log(static_cast<double>(sum)));
  }
  return static_cast<float>(total / wsum);
}

Tensor softmax_ce_grad_weighted(const Tensor& logits, const std::vector<int>& labels,
                                const std::vector<float>& weights) {
  check_logits(logits, kGradName);
  const std::int64_t n = logits.dims().at(0);
  const std::int64_t c = logits.dims().at(1);
  const double wsum_d = check_weighted(labels, weights, n, c, kGradName);
  // kernel/参考实现都用 float 分母：Σw 超出 float 会变成 inf，梯度静默全 0。
  // 合同允许有限非负权重，所以这里必须拒绝，不能返回一个“看起来正常”的零梯度。
  if (!(wsum_d <= static_cast<double>(std::numeric_limits<float>::max()))) {
    throw std::invalid_argument(std::string(kGradName) +
                                ": 权重按样本求和超出 float 可表示范围，梯度会全为 0");
  }
  const float wsum = static_cast<float>(wsum_d);
  Tensor g = Tensor::zeros({n, c});
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[static_cast<std::size_t>(i)];
    const float w = weights[static_cast<std::size_t>(label)];
    const float* row = logits.data() + i * c;
    float mx = 0.0f;
    float sum = 0.0f;
    detail::softmax_row(row, c, &mx, &sum);
    for (std::int64_t j = 0; j < c; ++j) {
      const float p = std::exp(row[j] - mx) / sum;
      g[i * c + j] = w * (p - (j == label ? 1.0f : 0.0f)) / wsum;
    }
  }
  return g;
}

}  // namespace uavrgb
