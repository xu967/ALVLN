// Linear 层 CPU/CUDA 前向+反向 parity 自检：无外部测试框架，返回 0 表示通过。
// 确定性输入；报告每个输出张量的最大绝对误差（doc 04 §6 N3）。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "uavrgb/linear_cuda.hpp"
#include "uavrgb/ops.hpp"

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

// 两个同形状 Tensor 的最大绝对误差。NaN 不能被 std::max 吞掉（NaN 比较恒
// false），否则 CUDA 侧全是 NaN 时误差仍是 0，parity 测试会假绿。
double max_abs_err(const uavrgb::Tensor& a, const uavrgb::Tensor& b) {
  if (a.numel() != b.numel()) return std::numeric_limits<double>::infinity();
  double m = 0.0;
  for (std::size_t i = 0; i < a.numel(); ++i) {
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (!std::isfinite(d)) return std::numeric_limits<double>::infinity();
    m = std::max(m, d);
  }
  return m;
}

}  // namespace

int main() {
  using namespace uavrgb;

  // 守卫自身的负例：NaN 差值必须变成 inf，不能被 std::max 吞掉。
  check(!std::isfinite(max_abs_err(Tensor::from_flat({1}, {0.0f}),
                                   Tensor::from_flat(
                                       {1}, {std::numeric_limits<float>::quiet_NaN()}))),
        "max_abs_err 必须把 NaN 判成非有限");

  const int in = 8;
  const int out = 4;
  const int n = 16;

  // 确定性非平凡参数：W[k,j] = 0.05*(k*in+j-10)，b[k] = 0.1*(k-2)。
  Linear m(in, out);
  for (int k = 0; k < out; ++k) {
    for (int j = 0; j < in; ++j) {
      m.W[k * in + j] = 0.05f * static_cast<float>(k * in + j - 10);
    }
    m.b[k] = 0.1f * static_cast<float>(k - 2);
  }

  // 确定性输入 x 与上游梯度 dlogits（本层只做 Linear，dlogits 为任意上游梯度）。
  Tensor x = Tensor::zeros({n, in});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < in; ++j) {
      x[i * in + j] = 0.02f * static_cast<float>(i * in + j - 20);
    }
  }
  Tensor dlogits = Tensor::zeros({n, out});
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < out; ++k) {
      dlogits[i * out + k] = 0.03f * static_cast<float>(i * out + k - 6);
    }
  }

  // CPU 与 CUDA 前向。
  Tensor y_cpu = linear_forward(m, x);
  Tensor y_cuda = linear_forward_cuda(m, x);

  // CPU 与 CUDA 反向。
  LinearGrad g_cpu = linear_backward(m, x, dlogits);
  LinearGrad g_cuda = linear_backward_cuda(m, x, dlogits);

  const double e_y = max_abs_err(y_cpu, y_cuda);
  const double e_dw = max_abs_err(g_cpu.dW, g_cuda.dW);
  const double e_db = max_abs_err(g_cpu.db, g_cuda.db);
  const double e_dx = max_abs_err(g_cpu.dx, g_cuda.dx);
  const double worst = std::max({e_y, e_dw, e_db, e_dx});

  std::cout << "linear parity (in=" << in << " out=" << out << " N=" << n << ")\n";
  std::cout << "  max|y_cpu - y_cuda|    = " << e_y << "\n";
  std::cout << "  max|dW_cpu - dW_cuda|  = " << e_dw << "\n";
  std::cout << "  max|db_cpu - db_cuda|  = " << e_db << "\n";
  std::cout << "  max|dx_cpu - dx_cuda|  = " << e_dx << "\n";
  std::cout << "  worst                   = " << worst << "\n";

  // float 累加约 8~16 项，1e-4 是宽松但明确的 parity 门槛。
  check(worst < 1e-4f, "CPU/CUDA Linear 前反向最大绝对误差 < 1e-4");

  // ---- 第 1 条负例：参数/形状校验 ----
  {
    Linear bad_w(in, out);
    bad_w.W = Tensor::zeros({2});  // 应为 out*in = 32 元素
    check(throws_invalid([&] { (void)linear_forward_cuda(bad_w, x); }),
          "linear_forward_cuda: W 元素数不足抛 invalid_argument");
  }
  check(throws_invalid([&] { (void)linear_forward_cuda(m, Tensor::zeros({n, in + 1})); }),
        "linear_forward_cuda: x 列数不匹配抛 invalid_argument");
  check(throws_invalid([&] { (void)linear_backward_cuda(m, x, Tensor::zeros({n, out + 1})); }),
        "linear_backward_cuda: dlogits 列数不匹配抛 invalid_argument");

  // 元素数 > INT_MAX：入口直接拒绝，绝不静默回绕后少算（审核 CUDA #1 的同类形状）。
  {
    Linear big(1, 65536);  // W/b 合计约 0.5 MB
    check(throws_invalid([&] { (void)linear_forward_cuda(big, Tensor::zeros({32768, 1})); }),
          "linear_forward_cuda: 元素数 > INT_MAX 抛 invalid_argument");
  }

  // ---- 第 2 条负例：空 batch 返回形状正确的空/零结果，不 launch 0 block 的 kernel ----
  {
    Tensor x0 = Tensor::zeros({0, in});
    Tensor y0 = linear_forward_cuda(m, x0);
    check(y0.dims() == std::vector<std::int64_t>({0, out}) && y0.numel() == 0,
          "linear_forward_cuda: N=0 返回 [0,out] 空张量");
    LinearGrad g0 = linear_backward_cuda(m, x0, Tensor::zeros({0, out}));
    check(g0.dW.dims() == std::vector<std::int64_t>({out, in}) &&
              g0.db.dims() == std::vector<std::int64_t>({out}) &&
              g0.dx.dims() == std::vector<std::int64_t>({0, in}),
          "linear_backward_cuda: N=0 梯度形状正确");
    bool zeros = true;
    for (std::size_t i = 0; i < g0.dW.numel(); ++i) zeros = zeros && g0.dW[i] == 0.0f;
    for (std::size_t i = 0; i < g0.db.numel(); ++i) zeros = zeros && g0.db[i] == 0.0f;
    check(zeros, "linear_backward_cuda: N=0 梯度为零");
  }

  if (g_failures == 0) {
    std::cout << "test_linear_cuda: PASS\n";
    return 0;
  }
  std::cerr << "test_linear_cuda: " << g_failures << " 项失败\n";
  return 1;
}
