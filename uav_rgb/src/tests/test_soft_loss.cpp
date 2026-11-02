// 软标签（蒸馏）交叉熵自检（无外部测试框架，返回 0 = 通过）：
//   1) alpha=0 与无权 softmax-CE 的 loss/grad 逐位一致（memcmp）
//   2) alpha=1 且 q 为 one-hot 时与硬标签版逐位一致
//   3) alpha=1 时 loss 等于 -Σ q·log p（手算）
//   4) 对 logits 做 FP64 有限差分，与 softmax_ce_grad_soft 在 alpha=0.5 下一致
//   5) 非法输入（尺寸/未归一化/alpha 越界）抛 invalid_argument
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "uavrgb/ops.hpp"

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

bool same_bits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }

constexpr int kN = 3;
constexpr int kC = 4;

std::vector<float> logits_data() {
  // 故意取非对称、量级不同的 logits，避免退化情形掩盖错误。
  return {0.5f, -1.0f, 2.0f, 0.25f,
          -2.0f, 3.0f, 0.0f, 1.0f,
          1.5f, 1.5f, 1.5f, 1.5f};
}

std::vector<int> labels() { return {2, 1, 0}; }

// 每行做 softmax 得到教师分布（等价于 T=1 的软标签）。
std::vector<float> softmax_rows(const std::vector<float>& logits) {
  std::vector<float> q(kN * kC);
  for (int i = 0; i < kN; ++i) {
    float mx = logits[i * kC];
    for (int j = 1; j < kC; ++j) mx = std::max(mx, logits[i * kC + j]);
    float s = 0.0f;
    for (int j = 0; j < kC; ++j) {
      q[i * kC + j] = std::exp(logits[i * kC + j] - mx);
      s += q[i * kC + j];
    }
    for (int j = 0; j < kC; ++j) q[i * kC + j] /= s;
  }
  return q;
}

std::vector<float> onehot(const std::vector<int>& y) {
  std::vector<float> q(kN * kC, 0.0f);
  for (int i = 0; i < kN; ++i) q[i * kC + y[static_cast<std::size_t>(i)]] = 1.0f;
  return q;
}

}  // namespace

int main() {
  using uavrgb::Tensor;
  const std::vector<float> ld = logits_data();
  Tensor logits({kN, kC});
  for (int i = 0; i < kN * kC; ++i) logits[i] = ld[static_cast<std::size_t>(i)];
  const std::vector<int> y = labels();
  const std::vector<float> q = softmax_rows(ld);
  const std::vector<float> oh = onehot(y);

  // 1) alpha=0 必须逐位等于无权版（同一表达式，不能只是"接近"）。
  {
    const float a = uavrgb::softmax_ce_loss(logits, y);
    const float b = uavrgb::softmax_ce_loss_soft(logits, y, q, 0.0f);
    check(same_bits(a, b), "alpha=0 的 loss 与无权版不逐位一致");
    const Tensor ga = uavrgb::softmax_ce_grad(logits, y);
    const Tensor gb = uavrgb::softmax_ce_grad_soft(logits, y, q, 0.0f);
    bool ok = ga.numel() == gb.numel();
    for (std::size_t i = 0; ok && i < ga.numel(); ++i) ok = same_bits(ga[i], gb[i]);
    check(ok, "alpha=0 的 grad 与无权版不逐位一致");
  }

  // 2) alpha=1 + one-hot 教师 = 硬标签版。
  {
    const float a = uavrgb::softmax_ce_loss(logits, y);
    const float b = uavrgb::softmax_ce_loss_soft(logits, y, oh, 1.0f);
    // one-hot 走的是 Σ q·(...) 求和路径，末位可能差 1 ulp；用相对误差判定。
    check(std::fabs(a - b) <= 1e-6f * std::max(1.0f, std::fabs(a)),
          "alpha=1 + one-hot 与硬标签 loss 不一致");
  }

  // 3) alpha=1 时 loss = -Σ q·log p，用 FP64 手算对照。
  {
    double expect = 0.0;
    for (int i = 0; i < kN; ++i) {
      double mx = -1e30, s = 0.0;
      for (int j = 0; j < kC; ++j) mx = std::max(mx, static_cast<double>(ld[i * kC + j]));
      for (int j = 0; j < kC; ++j) s += std::exp(static_cast<double>(ld[i * kC + j]) - mx);
      for (int j = 0; j < kC; ++j) {
        const double p = std::exp(static_cast<double>(ld[i * kC + j]) - mx) / s;
        expect -= static_cast<double>(q[i * kC + j]) * std::log(p);
      }
    }
    expect /= kN;
    const float got = uavrgb::softmax_ce_loss_soft(logits, y, q, 1.0f);
    check(std::fabs(got - static_cast<float>(expect)) < 1e-5,
          "alpha=1 的 loss 与手算 -Σq·log p 不符");
  }

  // 4) FP64 有限差分 gradcheck（alpha=0.5）。
  {
    const float alpha = 0.5f;
    const Tensor g = uavrgb::softmax_ce_grad_soft(logits, y, q, alpha);
    const double eps = 1e-3;
    double worst = 0.0;
    for (int i = 0; i < kN * kC; ++i) {
      Tensor plus({kN, kC});
      Tensor minus({kN, kC});
      for (int k = 0; k < kN * kC; ++k) {
        plus[k] = ld[static_cast<std::size_t>(k)];
        minus[k] = ld[static_cast<std::size_t>(k)];
      }
      plus[i] += static_cast<float>(eps);
      minus[i] -= static_cast<float>(eps);
      const double lp = uavrgb::softmax_ce_loss_soft(plus, y, q, alpha);
      const double lm = uavrgb::softmax_ce_loss_soft(minus, y, q, alpha);
      const double num = (lp - lm) / (2.0 * eps);
      worst = std::max(worst, std::fabs(num - static_cast<double>(g[i])));
    }
    check(worst < 1e-4, "gradcheck 误差超限（>1e-4）");
  }

  // 5) 非法输入必须抛 invalid_argument。
  {
    std::vector<float> short_q(q.begin(), q.end() - 1);
    check(throws_invalid([&] {
            uavrgb::softmax_ce_loss_soft(logits, y, short_q, 0.5f);
          }), "软标签尺寸不符未抛异常");

    std::vector<float> bad = q;
    for (int j = 0; j < kC; ++j) bad[j] *= 2.0f;  // 第一行和为 2
    check(throws_invalid([&] {
            uavrgb::softmax_ce_loss_soft(logits, y, bad, 0.5f);
          }), "未归一化的软标签未抛异常");

    check(throws_invalid([&] {
            uavrgb::softmax_ce_loss_soft(logits, y, q, 1.5f);
          }), "alpha>1 未抛异常");
    check(throws_invalid([&] {
            uavrgb::softmax_ce_loss_soft(logits, y, q, -0.1f);
          }), "alpha<0 未抛异常");

    std::vector<int> bad_y = y;
    bad_y[0] = kC;  // 越界
    check(throws_invalid([&] {
            uavrgb::softmax_ce_grad_soft(logits, bad_y, q, 0.5f);
          }), "label 越界未抛异常");
  }

  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << ")\n";
  return 1;
}
