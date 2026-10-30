// softmax 交叉熵 + 线性层/卷积层前向/反向 gradcheck 自检：无外部测试框架，返回 0 表示通过。
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "uavrgb/conv.hpp"
#include "uavrgb/ops.hpp"

namespace {

int failures = 0;

void expect(bool cond, const char* msg) {
  if (!cond) {
    ++failures;
    std::cerr << "FAIL: " << msg << "\n";
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

  // 确定性非平凡参数：W[4,3]、b[3] 均取 0.12*(i-5)。
  Linear m(4, 3);
  for (std::size_t i = 0; i < m.W.numel(); ++i) {
    m.W[i] = 0.12f * (static_cast<float>(i) - 5.0f);
  }
  for (std::size_t i = 0; i < m.b.numel(); ++i) {
    m.b[i] = 0.12f * (static_cast<float>(i) - 5.0f);
  }

  // x: [2,4] 已知值；labels {2,0}。
  Tensor x = Tensor::from_flat(
      {2, 4}, {0.5f, -0.25f, 0.75f, -0.5f, -0.75f, 0.25f, 0.125f, -0.375f});
  std::vector<int> labels = {2, 0};

  Tensor logits = linear_forward(m, x);
  expect(logits.ndim() == 2 && logits.dims()[0] == 2 && logits.dims()[1] == 3,
         "linear_forward 输出形状 [2,3]");

  float loss = softmax_ce_loss(logits, labels);
  expect(std::isfinite(loss) && loss >= 0.0f, "loss 有限且非负");

  double err = gradcheck_linear(m, x, labels);
  std::cout << "loss=" << loss << " gradcheck_max_err=" << err << "\n";
  expect(err < 1e-5, "gradcheck 最大绝对误差 < 1e-5");
  Tensor x_nan = x;
  x_nan[0] = std::numeric_limits<float>::quiet_NaN();
  expect(!std::isfinite(gradcheck_linear(m, x_nan, labels)),
         "gradcheck_linear 遇到 NaN 必须返回非有限");

  // 形状不匹配应抛异常（防止越界读 W），见 ops_cpu.cpp 的校验。
  bool shape_threw = false;
  try {
    Tensor bad_x = Tensor::zeros({2, 5});  // in=5 != m.in=4
    (void)linear_forward(m, bad_x);
  } catch (const std::invalid_argument&) {
    shape_threw = true;
  }
  expect(shape_threw, "形状不匹配抛 invalid_argument");

  // ---- 第 1 条负例：W/b 形状不变量 + gradcheck 输入 rank/通道校验 ----
  expect(throws_invalid([&] {
           Linear bad(4, 3);
           bad.W = Tensor::zeros({2});  // 应为 out*in = 12 元素
           (void)linear_forward(bad, x);
         }),
         "Linear W 元素数不足抛 invalid_argument");
  expect(throws_invalid([&] {
           Linear bad(4, 3);
           bad.b = Tensor::zeros({1});  // 应为 out = 3 元素
           (void)linear_forward(bad, x);
         }),
         "Linear b 元素数不足抛 invalid_argument");
  expect(throws_invalid([&] { (void)gradcheck_linear(m, Tensor::zeros({4}), {}); }),
         "gradcheck_linear rank<2 抛 invalid_argument");
  expect(throws_invalid([&] { (void)gradcheck_conv2d(Conv2d(1, 1, 3, 3, 1), Tensor::zeros({1, 1, 4})); }),
         "gradcheck_conv2d rank<4 抛 invalid_argument");
  expect(
      throws_invalid([&] { (void)gradcheck_conv2d(Conv2d(1, 1, 3, 3, 1), Tensor::zeros({1, 3, 4, 4})); }),
      "gradcheck_conv2d 通道不匹配抛 invalid_argument");
  expect(throws_invalid([&] {
           Conv2d c(1, 2, 3, 3, 1);
           c.W = Tensor::zeros({2});  // 应为 out*in*kh*kw = 18 元素
           (void)conv2d_forward(c, Tensor::zeros({1, 1, 4, 4}));
         }),
         "Conv2d W 元素数不足抛 invalid_argument");
  expect(throws_invalid([&] { (void)conv2d_forward(Conv2d(1, 1, 3, 3, 0), Tensor::zeros({1, 1, 2, 2})); }),
         "conv2d pad 太小（输出尺寸非正）抛 invalid_argument");

  // ---- 第 2 条负例：空 batch 是调用方错误，CPU loss/grad 抛异常而非返回 NaN ----
  expect(throws_invalid([&] { (void)softmax_ce_loss(Tensor::zeros({0, 3}), {}); }),
         "softmax_ce_loss 空 batch 抛 invalid_argument");
  expect(throws_invalid([&] { (void)softmax_ce_grad(Tensor::zeros({0, 3}), {}); }),
         "softmax_ce_grad 空 batch 抛 invalid_argument");
  expect(throws_invalid([&] { (void)softmax_ce_loss(Tensor::zeros({1, 0}), {0}); }),
         "softmax_ce_loss C=0 抛 invalid_argument");
  expect(throws_invalid([&] { (void)gradcheck_linear(m, Tensor::zeros({0, 4}), {}); }),
         "gradcheck_linear 空 batch 抛 invalid_argument");
  expect(throws_invalid([&] { (void)gradcheck_conv2d(Conv2d(1, 1, 3, 3, 1), Tensor::zeros({0, 1, 4, 4})); }),
         "gradcheck_conv2d 空 batch 抛 invalid_argument");

  return failures == 0 ? 0 : 1;
}
