// softmax 交叉熵 loss/grad 的 CPU/CUDA parity 自检：无外部测试框架，返回 0 表示通过。
// 确定性输入（含较大 logits 考验 stable softmax）；报告最大绝对误差（doc 04 §6 N3）。
#include <algorithm>
#include <cmath>
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

// 断言 f() 抛 std::invalid_argument；其它异常或未抛都算失败。
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

}  // namespace

int main() {
  using namespace uavrgb;

  const int n = 6;
  const int c = 5;

  // 确定性 logits：行内混入大值（±8 量级）验证先减 max 的 stable softmax。
  Tensor logits = Tensor::zeros({n, c});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      logits[i * c + j] = 0.7f * static_cast<float>((i * c + j) % 7 - 3) +
                          (j == 2 ? 8.0f : 0.0f) - (j == 4 ? 8.0f : 0.0f);
    }
  }
  const std::vector<int> labels = {2, 0, 3, 1, 4, 2};

  const float loss_cpu = softmax_ce_loss(logits, labels);
  const float loss_cuda = softmax_ce_loss_cuda(logits, labels);
  const Tensor g_cpu = softmax_ce_grad(logits, labels);
  const Tensor g_cuda = softmax_ce_grad_cuda(logits, labels);

  const double e_loss = std::fabs(static_cast<double>(loss_cpu) - static_cast<double>(loss_cuda));
  double e_grad = 0.0;
  for (std::size_t i = 0; i < g_cpu.numel(); ++i) {
    e_grad = std::max(e_grad, std::fabs(static_cast<double>(g_cpu[i]) - static_cast<double>(g_cuda[i])));
  }
  const double worst = std::max(e_loss, e_grad);

  std::cout << "softmax_ce parity (N=" << n << " C=" << c << ")\n";
  std::cout << "  loss_cpu = " << loss_cpu << ", loss_cuda = " << loss_cuda << "\n";
  std::cout << "  max|loss_cpu - loss_cuda| = " << e_loss << "\n";
  std::cout << "  max|grad_cpu - grad_cuda| = " << e_grad << "\n";
  std::cout << "  worst                      = " << worst << "\n";

  check(std::isfinite(loss_cuda) && loss_cuda >= 0.0f, "loss 有限且非负");
  check(worst < 1e-4f, "CPU/CUDA softmax-CE loss/grad 最大绝对误差 < 1e-4");

  // label 越界应抛异常（防越界读）。
  bool threw = false;
  try {
    const std::vector<int> bad = {2, 9};
    (void)softmax_ce_loss_cuda(logits, bad);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "label 越界抛 invalid_argument");

  // ---- 第 2 条负例：空 batch / C=0 是调用方错误，CPU 与 CUDA 都抛异常（不再 NaN / exit）----
  const Tensor empty = Tensor::zeros({0, c});
  check(throws_invalid([&] { (void)softmax_ce_loss(empty, {}); }),
        "softmax_ce_loss N=0 抛 invalid_argument");
  check(throws_invalid([&] { (void)softmax_ce_grad(empty, {}); }),
        "softmax_ce_grad N=0 抛 invalid_argument");
  check(throws_invalid([&] { (void)softmax_ce_loss_cuda(empty, {}); }),
        "softmax_ce_loss_cuda N=0 抛 invalid_argument");
  check(throws_invalid([&] { (void)softmax_ce_grad_cuda(empty, {}); }),
        "softmax_ce_grad_cuda N=0 抛 invalid_argument");
  const Tensor no_class = Tensor::zeros({1, 0});
  check(throws_invalid([&] { (void)softmax_ce_loss_cuda(no_class, {0}); }),
        "softmax_ce_loss_cuda C=0 抛 invalid_argument");
  check(throws_invalid([&] { (void)softmax_ce_loss(no_class, {0}); }),
        "softmax_ce_loss C=0 抛 invalid_argument");

  if (g_failures == 0) {
    std::cout << "test_softmax_cuda: PASS\n";
    return 0;
  }
  std::cerr << "test_softmax_cuda: " << g_failures << " 项失败\n";
  return 1;
}
