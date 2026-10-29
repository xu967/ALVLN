// Conv2d 层 CPU/CUDA 前向+反向 parity + FP64 gradcheck 自检：无外部测试框架，返回 0 表示通过。
// 确定性输入；报告每个输出张量的最大绝对误差（doc 04 §6 N1/N3）。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "uavrgb/conv.hpp"
#include "uavrgb/conv_cuda.hpp"

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

// 两个同形状 Tensor 的最大绝对误差。
double max_abs_err(const uavrgb::Tensor& a, const uavrgb::Tensor& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.numel(); ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return m;
}

// 对一组确定性 (m, x, dy) 跑 CPU/CUDA 前反向，返回四个最大绝对误差中最大者，并打印各值。
double run_parity(const uavrgb::Conv2d& m, const uavrgb::Tensor& x, const uavrgb::Tensor& dy,
                  const char* tag) {
  uavrgb::Tensor y_cpu = uavrgb::conv2d_forward(m, x);
  uavrgb::Tensor y_cuda = uavrgb::conv2d_forward_cuda(m, x);
  uavrgb::ConvGrad g_cpu = uavrgb::conv2d_backward(m, x, dy);
  uavrgb::ConvGrad g_cuda = uavrgb::conv2d_backward_cuda(m, x, dy);

  const double e_y = max_abs_err(y_cpu, y_cuda);
  const double e_dw = max_abs_err(g_cpu.dW, g_cuda.dW);
  const double e_db = max_abs_err(g_cpu.db, g_cuda.db);
  const double e_dx = max_abs_err(g_cpu.dx, g_cuda.dx);
  const double worst = std::max({e_y, e_dw, e_db, e_dx});

  std::cout << "conv parity " << tag << " (in=" << m.in << " out=" << m.out << " k=" << m.kh
            << "x" << m.kw << " pad=" << m.pad << " x=[" << x.dims()[0] << "," << x.dims()[1]
            << "," << x.dims()[2] << "," << x.dims()[3] << "])\n";
  std::cout << "  max|y_cpu - y_cuda|    = " << e_y << "\n";
  std::cout << "  max|dW_cpu - dW_cuda|  = " << e_dw << "\n";
  std::cout << "  max|db_cpu - db_cuda|  = " << e_db << "\n";
  std::cout << "  max|dx_cpu - dx_cuda|  = " << e_dx << "\n";
  std::cout << "  worst                   = " << worst << "\n";
  return worst;
}

}  // namespace

int main() {
  using namespace uavrgb;

  double worst = 0.0;

  // 配置 1：pad=1（输出同尺寸），非对称 H/W 防转置 bug。
  {
    Conv2d m(2, 3, 3, 3, 1);
    for (std::size_t i = 0; i < m.W.numel(); ++i) {
      m.W[i] = 0.04f * (static_cast<float>(i) - 30.0f);
    }
    for (std::size_t i = 0; i < m.b.numel(); ++i) {
      m.b[i] = 0.1f * (static_cast<float>(i) - 1.0f);
    }
    Tensor x = Tensor::zeros({2, 2, 6, 7});
    for (std::size_t i = 0; i < x.numel(); ++i) {
      x[i] = 0.02f * (static_cast<float>(i) - 80.0f);
    }
    Tensor dy = Tensor::zeros({2, 3, 6, 7});
    for (std::size_t i = 0; i < dy.numel(); ++i) {
      dy[i] = 0.03f * (static_cast<float>(i) - 100.0f);
    }
    worst = std::max(worst, run_parity(m, x, dy, "pad=1"));
  }

  // 配置 2：pad=0（默认值），输出缩小。
  {
    Conv2d m(1, 2, 3, 3, 0);
    for (std::size_t i = 0; i < m.W.numel(); ++i) {
      m.W[i] = 0.05f * (static_cast<float>(i) - 9.0f);
    }
    for (std::size_t i = 0; i < m.b.numel(); ++i) {
      m.b[i] = 0.1f * static_cast<float>(i);
    }
    Tensor x = Tensor::zeros({2, 1, 8, 9});
    for (std::size_t i = 0; i < x.numel(); ++i) {
      x[i] = 0.02f * (static_cast<float>(i) - 70.0f);
    }
    Tensor dy = Tensor::zeros({2, 2, 6, 7});
    for (std::size_t i = 0; i < dy.numel(); ++i) {
      dy[i] = 0.03f * (static_cast<float>(i) - 40.0f);
    }
    worst = std::max(worst, run_parity(m, x, dy, "pad=0"));
  }

  // FP64 有限差分 gradcheck：扰动 W/b 中心差分（loss = Σ y²）。
  {
    Conv2d m(1, 2, 3, 3, 1);
    for (std::size_t i = 0; i < m.W.numel(); ++i) {
      m.W[i] = 0.12f * (static_cast<float>(i) - 9.0f);
    }
    for (std::size_t i = 0; i < m.b.numel(); ++i) {
      m.b[i] = 0.12f * (static_cast<float>(i) - 5.0f);
    }
    Tensor x = Tensor::from_flat(
        {1, 1, 4, 4},
        {0.5f, -0.25f, 0.75f, -0.5f, -0.75f, 0.25f, 0.125f, -0.375f, 0.625f, 0.875f, -0.125f,
         -0.625f, 0.375f, -0.875f, 0.1875f, -0.1875f});
    const double err = gradcheck_conv2d(m, x);
    std::cout << "conv gradcheck_max_err = " << err << "\n";
    check(err < 1e-5, "conv gradcheck 最大绝对误差 < 1e-5");
  }

  // 形状不匹配应抛异常（防越界）：CPU 参考与 CUDA 包装都测。
  {
    Conv2d m(2, 3, 3, 3, 1);
    Tensor bad_x = Tensor::zeros({2, 3, 4, 4});  // in=3 != m.in=2
    check(throws_invalid([&] { (void)conv2d_forward(m, bad_x); }),
          "conv2d CPU 通道不匹配抛 invalid_argument");
    check(throws_invalid([&] { (void)conv2d_forward_cuda(m, bad_x); }),
          "conv2d_forward_cuda 通道不匹配抛 invalid_argument");
  }

  // ---- 第 1 条负例：W 形状不变量 / pad 太小（CPU 与 CUDA 都校验）----
  {
    Conv2d m(2, 3, 3, 3, 1);
    m.W = Tensor::zeros({2});  // 应为 out*in*kh*kw = 54 元素
    check(throws_invalid([&] { (void)conv2d_forward(m, Tensor::zeros({1, 2, 4, 4})); }),
          "conv2d CPU: W 元素数不足抛 invalid_argument");
    check(throws_invalid([&] { (void)conv2d_forward_cuda(m, Tensor::zeros({1, 2, 4, 4})); }),
          "conv2d_forward_cuda: W 元素数不足抛 invalid_argument");
  }
  check(
      throws_invalid([&] { (void)conv2d_forward_cuda(Conv2d(1, 1, 3, 3, 0), Tensor::zeros({1, 1, 2, 2})); }),
      "conv2d_forward_cuda: pad 太小抛 invalid_argument");

  // ---- 第 4 条负例：元素数 > INT_MAX 直接拒绝，绝不静默回绕后少算（审核 CUDA #1 的复现形状）----
  {
    Conv2d big(1, 65537, 1, 1, 0);  // W 仅 65537 元素；y 需 2^32+65536 个元素
    check(throws_invalid([&] { (void)conv2d_forward_cuda(big, Tensor::zeros({1, 1, 256, 256})); }),
          "conv2d_forward_cuda: 元素数 > INT_MAX 抛 invalid_argument");
  }

  // ---- 第 2 条负例：N=0 返回形状正确的空/零结果，不 launch 0 block 的 kernel ----
  {
    Conv2d m(2, 3, 3, 3, 1);
    Tensor x0 = Tensor::zeros({0, 2, 6, 7});
    Tensor y0 = conv2d_forward_cuda(m, x0);
    check(y0.dims() == std::vector<std::int64_t>({0, 3, 6, 7}) && y0.numel() == 0,
          "conv2d_forward_cuda: N=0 返回空张量");
    ConvGrad g0 = conv2d_backward_cuda(m, x0, Tensor::zeros({0, 3, 6, 7}));
    check(g0.dW.dims() == std::vector<std::int64_t>({3, 2, 3, 3}) &&
              g0.db.dims() == std::vector<std::int64_t>({3}) &&
              g0.dx.dims() == std::vector<std::int64_t>({0, 2, 6, 7}),
          "conv2d_backward_cuda: N=0 梯度形状正确");
    bool zeros = true;
    for (std::size_t i = 0; i < g0.dW.numel(); ++i) zeros = zeros && g0.dW[i] == 0.0f;
    for (std::size_t i = 0; i < g0.db.numel(); ++i) zeros = zeros && g0.db[i] == 0.0f;
    check(zeros, "conv2d_backward_cuda: N=0 梯度为零");
  }

  // float 累加项多（N*H'*W'*kh*kw 量级），1e-4 是宽松但明确的 parity 门槛。
  check(worst < 1e-4f, "CPU/CUDA Conv2d 前反向最大绝对误差 < 1e-4");

  if (g_failures == 0) {
    std::cout << "test_conv_cuda: PASS\n";
    return 0;
  }
  std::cerr << "test_conv_cuda: " << g_failures << " 项失败\n";
  return 1;
}
