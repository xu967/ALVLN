// 逐格 softmax 交叉熵的 CPU 参考实现 + FP64 gradcheck（深度分带辅助头用）。
// CUDA 版在 softmax_cuda.cu（spatial_ce_*_cuda）与 dev::spatial_ce_*；三者口径以本文件为准。
#include "uavrgb/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace uavrgb {
namespace {

struct SpatialDims {
  std::int64_t n, k, hw;
};

SpatialDims check_spatial(const Tensor& logits, const std::vector<std::uint8_t>& targets,
                          const char* who) {
  if (logits.ndim() != 4) {
    throw std::invalid_argument(std::string(who) + ": logits 必须为 4 维 [N,K,H,W]（实际 ndim=" +
                                std::to_string(logits.ndim()) + "）");
  }
  const SpatialDims d{logits.dims()[0], logits.dims()[1], logits.dims()[2] * logits.dims()[3]};
  if (d.n == 0 || d.k == 0 || d.hw == 0) {
    throw std::invalid_argument(std::string(who) + ": N/K/H/W 都必须为正");
  }
  if (d.k >= kSpatialIgnore) {
    // 255 是屏蔽码；K 上限 254 与 checkpoint 的 kMaxAuxClasses、CUDA 版 check 同口径。
    throw std::invalid_argument(std::string(who) + ": K 必须 < 255");
  }
  if (static_cast<std::int64_t>(targets.size()) != d.n * d.hw) {
    throw std::invalid_argument(std::string(who) + ": targets 数量应为 N*H*W=" +
                                std::to_string(d.n * d.hw) + "（实际 " +
                                std::to_string(targets.size()) + "）");
  }
  for (std::uint8_t t : targets) {
    if (t != kSpatialIgnore && t >= d.k) {
      throw std::invalid_argument(std::string(who) + ": target 越界（" + std::to_string(t) + " >= K=" +
                                  std::to_string(d.k) + "，屏蔽码为 255）");
    }
  }
  return d;
}

// 一格的 softmax 统计：max 与 Σexp(x-max)，通道步长为 hw。
template <typename T>
void cell_stats(const T* x, std::int64_t k, std::int64_t hw, T* mx, T* sum) {
  *mx = x[0];
  for (std::int64_t c = 1; c < k; ++c) *mx = std::max(*mx, x[c * hw]);
  *sum = 0;
  for (std::int64_t c = 0; c < k; ++c) *sum += std::exp(x[c * hw] - *mx);
}

std::size_t count_valid(const std::vector<std::uint8_t>& targets) {
  return static_cast<std::size_t>(
      std::count_if(targets.begin(), targets.end(), [](std::uint8_t t) { return t != kSpatialIgnore; }));
}

double loss_f64(const std::vector<double>& x, const std::vector<std::uint8_t>& targets,
                const SpatialDims& d, std::size_t n_valid) {
  if (n_valid == 0) return 0.0;
  double total = 0.0;
  for (std::int64_t i = 0; i < d.n; ++i) {
    for (std::int64_t p = 0; p < d.hw; ++p) {
      const std::uint8_t t = targets[static_cast<std::size_t>(i * d.hw + p)];
      if (t == kSpatialIgnore) continue;
      const double* cell = x.data() + i * d.k * d.hw + p;
      double mx, sum;
      cell_stats(cell, d.k, d.hw, &mx, &sum);
      total += std::log(sum) - (cell[t * d.hw] - mx);
    }
  }
  return total / static_cast<double>(n_valid);
}

}  // namespace

float spatial_ce_loss(const Tensor& logits, const std::vector<std::uint8_t>& targets,
                      std::size_t* valid_out) {
  const SpatialDims d = check_spatial(logits, targets, "spatial_ce_loss");
  const std::size_t n_valid = count_valid(targets);
  if (valid_out != nullptr) *valid_out = n_valid;
  std::vector<double> x(logits.numel());
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = logits[i];
  return static_cast<float>(loss_f64(x, targets, d, n_valid));
}

Tensor spatial_ce_grad(const Tensor& logits, const std::vector<std::uint8_t>& targets) {
  const SpatialDims d = check_spatial(logits, targets, "spatial_ce_grad");
  const std::size_t n_valid = count_valid(targets);
  Tensor g = Tensor::zeros(logits.dims());
  if (n_valid == 0) return g;
  const float inv = 1.0f / static_cast<float>(n_valid);
  for (std::int64_t i = 0; i < d.n; ++i) {
    for (std::int64_t p = 0; p < d.hw; ++p) {
      const std::uint8_t t = targets[static_cast<std::size_t>(i * d.hw + p)];
      if (t == kSpatialIgnore) continue;
      const float* cell = logits.data() + i * d.k * d.hw + p;
      float mx, sum;
      cell_stats(cell, d.k, d.hw, &mx, &sum);
      for (std::int64_t c = 0; c < d.k; ++c) {
        const float prob = std::exp(cell[c * d.hw] - mx) / sum;
        g[static_cast<std::size_t>(i * d.k * d.hw + c * d.hw + p)] = (prob - (c == t ? 1.0f : 0.0f)) * inv;
      }
    }
  }
  return g;
}

double gradcheck_spatial_ce(const Tensor& logits, const std::vector<std::uint8_t>& targets,
                            double eps) {
  const SpatialDims d = check_spatial(logits, targets, "gradcheck_spatial_ce");
  const std::size_t n_valid = count_valid(targets);
  const Tensor g = spatial_ce_grad(logits, targets);
  std::vector<double> x(logits.numel());
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = logits[i];
  double max_err = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double orig = x[i];
    x[i] = orig + eps;
    const double lp = loss_f64(x, targets, d, n_valid);
    x[i] = orig - eps;
    const double lm = loss_f64(x, targets, d, n_valid);
    x[i] = orig;
    const double diff = std::fabs((lp - lm) / (2.0 * eps) - static_cast<double>(g[i]));
    if (!std::isfinite(diff)) return std::numeric_limits<double>::infinity();
    max_err = std::max(max_err, diff);
  }
  return max_err;
}

}  // namespace uavrgb
