// VLN 动作切片与第 5 维独立 STOP 头（稳定二分类 BCE + 阈值扫描）。
#include "uavrgb/vln_head.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace uavrgb {

Tensor slice_action_logits(const Tensor& logits) {
  if (logits.ndim() != 2 ||
      (logits.dims()[1] != kActionClasses && logits.dims()[1] != kActionStopClasses)) {
    throw std::invalid_argument("slice_action_logits: logits 列数必须是 4 或 5");
  }
  if (logits.dims()[1] == kActionClasses) return logits;
  const std::int64_t n = logits.dims()[0];
  const std::int64_t cols = logits.dims()[1];
  Tensor out({n, kActionClasses});
  for (std::int64_t i = 0; i < n; ++i) {
    const float* src = logits.data() + i * cols;
    float* dst = out.data() + i * kActionClasses;
    std::copy(src, src + kActionClasses, dst);
  }
  return out;
}

Tensor merge_action_stop_grad(const Tensor& action_grad, const std::vector<float>& stop_grad) {
  if (action_grad.ndim() != 2 || action_grad.dims()[1] != kActionClasses ||
      static_cast<std::size_t>(action_grad.dims()[0]) != stop_grad.size()) {
    throw std::invalid_argument("merge_action_stop_grad: 形状不匹配");
  }
  const std::int64_t n = action_grad.dims()[0];
  Tensor out({n, kActionStopClasses});
  for (std::int64_t i = 0; i < n; ++i) {
    const float* src = action_grad.data() + i * kActionClasses;
    float* dst = out.data() + i * kActionStopClasses;
    std::copy(src, src + kActionClasses, dst);
    dst[kActionClasses] = stop_grad[static_cast<std::size_t>(i)];
  }
  return out;
}

float auto_stop_pos_weight(const std::vector<std::uint8_t>& stop_targets,
                           const std::vector<std::uint32_t>& train_idx) {
  std::size_t pos = 0, neg = 0;
  for (std::uint32_t idx : train_idx) {
    if (stop_targets.at(idx) != 0) {
      ++pos;
    } else {
      ++neg;
    }
  }
  if (pos == 0) return 1.0f;
  const double ratio = static_cast<double>(neg) / static_cast<double>(pos);
  return static_cast<float>(std::clamp(ratio, 1.0, 20.0));
}

float stop_bce_loss_and_grad(const Tensor& logits, const std::vector<std::uint8_t>& stop_y,
                             float pos_weight, float stop_lambda, std::vector<float>* dstop) {
  if (logits.ndim() != 2 || logits.dims()[1] != kActionStopClasses ||
      static_cast<std::size_t>(logits.dims()[0]) != stop_y.size() || stop_y.empty()) {
    throw std::invalid_argument("stop_bce_loss_and_grad: logits 必须是 [N, 5] 且 N == stop_y.size()");
  }
  if (!(pos_weight > 0.0f) || !std::isfinite(pos_weight) ||
      !(stop_lambda >= 0.0f) || !std::isfinite(stop_lambda)) {
    throw std::invalid_argument("stop_bce_loss_and_grad: pos_weight/stop_lambda 非法");
  }
  const std::size_t n = stop_y.size();
  if (dstop) dstop->assign(n, 0.0f);
  const double inv_n = 1.0 / static_cast<double>(n);
  double loss_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    if (stop_y[i] > 1) throw std::invalid_argument("stop_bce_loss_and_grad: stop 标签必须是 0/1");
    const double y = static_cast<double>(stop_y[i]);
    const double z = static_cast<double>(logits[i * kActionStopClasses + kActionClasses]);
    const double w = (stop_y[i] != 0) ? static_cast<double>(pos_weight) : 1.0;
    const double bce = std::max(z, 0.0) - z * y + std::log1p(std::exp(-std::abs(z)));
    loss_sum += w * bce;
    if (dstop) {
      const double prob = (z >= 0.0) ? (1.0 / (1.0 + std::exp(-z)))
                                     : (std::exp(z) / (1.0 + std::exp(z)));
      (*dstop)[i] = static_cast<float>(static_cast<double>(stop_lambda) * w * (prob - y) * inv_n);
    }
  }
  return static_cast<float>(loss_sum * inv_n);
}

StopEvalMetrics evaluate_stop_head(const std::vector<float>& stop_logits,
                                   const std::vector<std::uint8_t>& stop_y,
                                   bool select_threshold, float given_threshold) {
  StopEvalMetrics m;
  if (stop_logits.empty() || stop_logits.size() != stop_y.size()) return m;
  const std::size_t n = stop_logits.size();
  std::vector<double> probs(n, 0.0);
  double bce_sum = 0.0;
  std::size_t total_pos = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const double z = static_cast<double>(stop_logits[i]);
    const double y = static_cast<double>(stop_y[i] != 0 ? 1 : 0);
    if (stop_y[i] != 0) ++total_pos;
    bce_sum += std::max(z, 0.0) - z * y + std::log1p(std::exp(-std::abs(z)));
    probs[i] = (z >= 0.0) ? (1.0 / (1.0 + std::exp(-z))) : (std::exp(z) / (1.0 + std::exp(z)));
  }
  m.bce = bce_sum / static_cast<double>(n);

  auto stats_at = [&](float th, double* f1, double* prec, double* rec) {
    std::size_t tp = 0, fp = 0, fn = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const bool pred = probs[i] >= static_cast<double>(th);
      const bool gt = stop_y[i] != 0;
      if (pred && gt) ++tp;
      else if (pred && !gt) ++fp;
      else if (!pred && gt) ++fn;
    }
    *prec = (tp + fp > 0) ? static_cast<double>(tp) / static_cast<double>(tp + fp) : 0.0;
    *rec = (tp + fn > 0) ? static_cast<double>(tp) / static_cast<double>(tp + fn) : 0.0;
    *f1 = (*prec + *rec > 0.0) ? (2.0 * (*prec) * (*rec) / (*prec + *rec)) : 0.0;
  };

  float chosen = given_threshold;
  if (select_threshold && total_pos > 0) {
    double best_f1 = -1.0;
    for (int step = 20; step <= 85; step += 5) {
      const float cand = static_cast<float>(step) * 0.01f;
      double f1 = 0.0, prec = 0.0, rec = 0.0;
      stats_at(cand, &f1, &prec, &rec);
      if (f1 > best_f1 + 1e-12 ||
          (std::abs(f1 - best_f1) <= 1e-12 &&
           std::abs(cand - kDefaultStopThreshold) < std::abs(chosen - kDefaultStopThreshold))) {
        best_f1 = f1;
        chosen = cand;
      }
    }
  }
  m.best_threshold = chosen;
  stats_at(chosen, &m.f1, &m.precision, &m.recall);
  return m;
}

}  // namespace uavrgb
