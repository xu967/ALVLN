// 软标签交叉熵（蒸馏用，CPU FP32 参考）：公式/梯度见 ops.hpp。
// alpha=0 时退化为无权 softmax 交叉熵；tests/test_soft_loss.cpp 断言与 CPU 无权版逐位一致。
#include "uavrgb/ops.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

#include "uavrgb/softmax_detail.hpp"

namespace uavrgb {
namespace {

const char* const kLossName = "softmax_ce_loss_soft";
const char* const kGradName = "softmax_ce_grad_soft";

void check_shapes(const Tensor& logits, const std::vector<int>& labels,
                  const std::vector<float>& q, float alpha, const char* who) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument(std::string(who) + ": logits 形状必须为 [N, C]");
  }
  const std::int64_t n = logits.dims().at(0);
  const std::int64_t c = logits.dims().at(1);
  if (n == 0) throw std::invalid_argument(std::string(who) + ": 空 batch（N=0）");
  if (c == 0) throw std::invalid_argument(std::string(who) + ": C 必须为正");
  if (static_cast<std::int64_t>(labels.size()) != n) {
    throw std::invalid_argument(std::string(who) + ": labels 数量与 logits 行数不一致");
  }
  if (static_cast<std::int64_t>(q.size()) != n * c) {
    throw std::invalid_argument(std::string(who) + ": 软标签尺寸应为 N*C=" +
                                std::to_string(n * c) + "（实际 " + std::to_string(q.size()) + "）");
  }
  if (!(alpha >= 0.0f) || !(alpha <= 1.0f)) {
    throw std::invalid_argument(std::string(who) + ": alpha 必须在 [0,1]（实际 " +
                                std::to_string(alpha) + "）");
  }
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[static_cast<std::size_t>(i)];
    if (label < 0 || label >= c) throw std::invalid_argument(std::string(who) + ": label 越界");
    double s = 0.0;
    for (std::int64_t j = 0; j < c; ++j) {
      const float v = q[static_cast<std::size_t>(i * c + j)];
      if (!(v >= 0.0f) || !std::isfinite(v)) {
        throw std::invalid_argument(std::string(who) + ": 软标签必须为非负有限值");
      }
      s += static_cast<double>(v);
    }
    // 教师分布必须是归一化的；没归一化就直接拒绝，避免“看起来正常但量纲错”的 loss。
    if (std::fabs(s - 1.0) > 1e-3) {
      throw std::invalid_argument(std::string(who) + ": 第 " + std::to_string(i) +
                                  " 行软标签和为 " + std::to_string(s) + "，应为 1");
    }
  }
}

}  // namespace

float softmax_ce_loss_soft(const Tensor& logits, const std::vector<int>& labels,
                           const std::vector<float>& q, float alpha) {
  check_shapes(logits, labels, q, alpha, kLossName);
  const std::int64_t n = logits.dims().at(0);
  const std::int64_t c = logits.dims().at(1);
  double total = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[static_cast<std::size_t>(i)];
    const float* row = logits.data() + i * c;
    float mx = 0.0f;
    float sum = 0.0f;
    detail::softmax_row(row, c, &mx, &sum);
    const double log_norm = std::log(static_cast<double>(sum));
    double soft = 0.0;
    for (std::int64_t j = 0; j < c; ++j) {
      const double qi = q[static_cast<std::size_t>(i * c + j)];
      if (qi == 0.0) continue;
      // log p_ij = (logit_ij - mx) - log(Σexp(logit-mx))
      soft += qi * (-static_cast<double>(row[j] - mx) + log_norm);
    }
    const double hard = -static_cast<double>(row[label] - mx) + log_norm;
    total += static_cast<double>(alpha) * soft + (1.0 - static_cast<double>(alpha)) * hard;
  }
  return static_cast<float>(total / static_cast<double>(n));
}

Tensor softmax_ce_grad_soft(const Tensor& logits, const std::vector<int>& labels,
                            const std::vector<float>& q, float alpha) {
  check_shapes(logits, labels, q, alpha, kGradName);
  const std::int64_t n = logits.dims().at(0);
  const std::int64_t c = logits.dims().at(1);
  // 用除法而不是乘 1/N：与无权版 softmax_ce_grad 的舍入完全一致，alpha=0 才能逐位相等。
  const float fn = static_cast<float>(n);
  Tensor g = Tensor::zeros({n, c});
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[static_cast<std::size_t>(i)];
    const float* row = logits.data() + i * c;
    float mx = 0.0f;
    float sum = 0.0f;
    detail::softmax_row(row, c, &mx, &sum);
    for (std::int64_t j = 0; j < c; ++j) {
      const float p = std::exp(row[j] - mx) / sum;
      const float onehot = (j == label) ? 1.0f : 0.0f;
      const float qi = q[static_cast<std::size_t>(i * c + j)];
      g[i * c + j] = (alpha * (p - qi) + (1.0f - alpha) * (p - onehot)) / fn;
    }
  }
  return g;
}

}  // namespace uavrgb
