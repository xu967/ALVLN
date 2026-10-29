// 加权 softmax-CE 自检（无外部测试框架，返回 0 = 通过）：
//   1) 全 1 权重与无权版 loss/grad 逐位一致（memcmp）
//   2) 加权版 FP64 有限差分 gradcheck
//   3) 加权版 CPU/CUDA parity < 1e-4
//   4) 非法 weights（长度/负值/全零/NaN）抛 invalid_argument
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "uavrgb/ops.hpp"
#include "uavrgb/softmax_cuda.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}

template <typename F>
bool throws_invalid(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

// 逐位比较（float 的 == 会把 -0.0 与 +0.0 当相等，这里要求位模式完全一致）。
bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

double max_abs_diff(const uavrgb::Tensor& a, const uavrgb::Tensor& b) {
  double d = 0.0;
  for (std::size_t i = 0; i < a.numel(); ++i) {
    d = std::max(d, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return d;
}

}  // namespace

int main() {
  using namespace uavrgb;

  const int n = 6;
  const int c = 5;
  Tensor logits = Tensor::zeros({n, c});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      logits[i * c + j] = 0.7f * static_cast<float>((i * c + j) % 7 - 3) +
                          (j == 2 ? 8.0f : 0.0f) - (j == 4 ? 8.0f : 0.0f);
    }
  }
  const std::vector<int> labels = {2, 0, 3, 1, 4, 2};
  const std::vector<float> ones(static_cast<std::size_t>(c), 1.0f);
  const std::vector<float> w = {0.5f, 2.0f, 1.25f, 4.0f, 0.75f};

  // ---- 1) 全 1 权重必须与无权版逐位一致 ----
  const float base_loss = softmax_ce_loss(logits, labels);
  const float ones_loss = softmax_ce_loss_weighted(logits, labels, ones);
  const Tensor base_g = softmax_ce_grad(logits, labels);
  const Tensor ones_g = softmax_ce_grad_weighted(logits, labels, ones);
  bool grad_bits = base_g.numel() == ones_g.numel();
  for (std::size_t i = 0; grad_bits && i < base_g.numel(); ++i) {
    grad_bits = same_bits(base_g[i], ones_g[i]);
  }
  std::cout << "all-ones vs unweighted: loss_bits_equal=" << same_bits(base_loss, ones_loss)
            << " grad_bits_equal=" << grad_bits << " (loss=" << base_loss << ")\n";
  check(same_bits(base_loss, ones_loss), "全 1 权重 loss 与无权版逐位一致");
  check(grad_bits, "全 1 权重 grad 与无权版逐位一致");

  // ---- 2) 加权版 FP64 gradcheck ----
  Linear m(4, 3);
  for (std::size_t i = 0; i < m.W.numel(); ++i) {
    m.W[i] = 0.12f * (static_cast<float>(i) - 5.0f);
  }
  for (std::size_t i = 0; i < m.b.numel(); ++i) {
    m.b[i] = 0.12f * (static_cast<float>(i) - 5.0f);
  }
  const Tensor x = Tensor::from_flat(
      {2, 4}, {0.5f, -0.25f, 0.75f, -0.5f, -0.75f, 0.25f, 0.125f, -0.375f});
  const std::vector<int> labels2 = {2, 0};
  const std::vector<float> weights2 = {0.5f, 2.0f, 1.0f};
  const double gc_err = gradcheck_linear_weighted(m, x, labels2, weights2);
  std::cout << "weighted gradcheck_max_err = " << gc_err << "\n";
  check(std::isfinite(gc_err) && gc_err < 1e-5, "加权 gradcheck 最大绝对误差 < 1e-5");

  // ---- 3) CPU/CUDA parity（非平凡权重） ----
  const float cpu_loss = softmax_ce_loss_weighted(logits, labels, w);
  const float cuda_loss = softmax_ce_loss_weighted_cuda(logits, labels, w);
  const Tensor cpu_g = softmax_ce_grad_weighted(logits, labels, w);
  const Tensor cuda_g = softmax_ce_grad_weighted_cuda(logits, labels, w);
  const double e_loss = std::fabs(static_cast<double>(cpu_loss) - static_cast<double>(cuda_loss));
  const double e_grad = max_abs_diff(cpu_g, cuda_g);
  std::cout << "weighted parity: cpu_loss=" << cpu_loss << " cuda_loss=" << cuda_loss
            << " max|dloss|=" << e_loss << " max|dgrad|=" << e_grad << "\n";
  check(e_loss < 1e-4, "加权 loss CPU/CUDA 误差 < 1e-4");
  check(e_grad < 1e-4, "加权 grad CPU/CUDA 误差 < 1e-4");

  // 全 1 权重 CUDA 与无权 CPU 也应同量级一致（atomicAdd 顺序不保证逐位）。
  const float cuda_ones = softmax_ce_loss_weighted_cuda(logits, labels, ones);
  check(std::fabs(static_cast<double>(cuda_ones) - static_cast<double>(base_loss)) < 1e-5,
        "全 1 权重 CUDA loss 与无权 CPU 一致");

  // ---- 4) 非法输入 ----
  check(throws_invalid([&] { (void)softmax_ce_loss_weighted(logits, labels, {1.0f, 1.0f}); }),
        "weights 长度 != C 抛 invalid_argument");
  check(throws_invalid([&] {
          (void)softmax_ce_loss_weighted(logits, labels, {1.0f, -1.0f, 1.0f, 1.0f, 1.0f});
        }),
        "weights 含负值抛 invalid_argument");
  check(throws_invalid([&] {
          (void)softmax_ce_grad_weighted(logits, labels, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
        }),
        "weights 全零（样本权重和为 0）抛 invalid_argument");
  check(throws_invalid([&] {
          (void)softmax_ce_grad_weighted(logits, labels,
                                         {1.0f, 1.0f, std::nanf(""), 1.0f, 1.0f});
        }),
        "weights 含 NaN 抛 invalid_argument");
  check(throws_invalid([&] {
          (void)softmax_ce_loss_weighted_cuda(logits, labels, {1.0f, 1.0f});
        }),
        "CUDA weights 长度非法抛 invalid_argument");
  check(throws_invalid([&] { (void)gradcheck_linear_weighted(m, x, labels2, {1.0f}); }),
        "gradcheck 加权版 weights 长度非法抛 invalid_argument");

  if (g_failures == 0) {
    std::cout << "test_weighted_loss: PASS\n";
    return 0;
  }
  std::cerr << "test_weighted_loss: " << g_failures << " 项失败\n";
  return 1;
}
