// D9：真实调用的 conv2d_backward / conv2d_backward_cuda 对输入梯度 dx 的有限差分检查（卡 C1a 第 6 条）。
//
// 背景：
//   原有 gradcheck_conv2d 自含独立解析式且只查 dW/db，test_conv_cuda 只做 CPU↔CUDA 互相对拍；
//   若 CPU 与 CUDA 的 dx 下标同时写反（如把 ih + pad - ki 写成 ih - pad + ki，漏掉卷积核 180° 翻转），
//   原有测试全部假绿。
//
// 方法：
//   取位置各异的加权二次损失 L(x, W, b) = 0.5 * Σ_j α_j * y_j(x, W, b)^2（二次型中心差分无截断误差），
//   上游梯度 dy_j = α_j * y_j。全程用 FP64 算中心差分 (L(x + eps*e_i) - L(x - eps*e_i)) / (2*eps)，
//   与 conv2d_backward（CPU）和 conv2d_backward_cuda（CUDA，底层即 DeviceCnn 使用的 dev::conv2d_backward）
//   产出的全部 dx / dW / db 元素逐一比对。覆盖 pad=1、pad=0、非方核 3x5 + pad=2 及 need_dx=false。
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "uavrgb/conv.hpp"
#include "uavrgb/conv_cuda.hpp"

using namespace uavrgb;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_fail;
  }
}

void fill_det(Tensor& t, float scale, int salt) {
  for (std::size_t i = 0; i < t.numel(); ++i) {
    t[i] = scale * (static_cast<float>((static_cast<int>(i) * 41 + salt) % 97) / 48.0f - 1.0f);
  }
}

// 独立 FP64 前向与加权二次损失 L = 0.5 * Σ_j alpha[j] * y[j]^2。
double forward_weighted_loss_f64(const std::vector<double>& w, const std::vector<double>& b,
                                 const std::vector<double>& x, const std::vector<double>& alpha,
                                 int n, int c_in, int h, int wd, int out, int kh, int kw, int pad,
                                 std::vector<double>* y_out = nullptr) {
  const int oh = h + 2 * pad - kh + 1;
  const int ow = wd + 2 * pad - kw + 1;
  if (y_out) y_out->assign(static_cast<std::size_t>(n) * out * oh * ow, 0.0);
  double loss = 0.0;
  for (int ni = 0; ni < n; ++ni) {
    for (int oc = 0; oc < out; ++oc) {
      for (int oi = 0; oi < oh; ++oi) {
        for (int oj = 0; oj < ow; ++oj) {
          double acc = b[static_cast<std::size_t>(oc)];
          for (int ic = 0; ic < c_in; ++ic) {
            for (int ki = 0; ki < kh; ++ki) {
              for (int kj = 0; kj < kw; ++kj) {
                const int ih = oi - pad + ki;
                const int iw = oj - pad + kj;
                if (ih >= 0 && ih < h && iw >= 0 && iw < wd) {
                  acc += x[((static_cast<std::size_t>(ni) * c_in + ic) * h + ih) * wd + iw] *
                         w[((static_cast<std::size_t>(oc) * c_in + ic) * kh + ki) * kw + kj];
                }
              }
            }
          }
          const std::size_t idx = ((static_cast<std::size_t>(ni) * out + oc) * oh + oi) * ow + oj;
          if (y_out) (*y_out)[idx] = acc;
          loss += 0.5 * alpha[idx] * acc * acc;
        }
      }
    }
  }
  return loss;
}

void run_dx_gradcheck_case(int n, int c_in, int h, int wd, int out, int kh, int kw, int pad,
                           int salt, const char* tag) {
  Conv2d m(c_in, out, kh, kw, pad);
  fill_det(m.W, 0.4f, salt + 3);
  fill_det(m.b, 0.2f, salt + 7);
  Tensor x = Tensor::zeros({n, c_in, h, wd});
  fill_det(x, 0.5f, salt + 11);

  const int oh = h + 2 * pad - kh + 1;
  const int ow = wd + 2 * pad - kw + 1;
  const std::size_t ny = static_cast<std::size_t>(n) * out * oh * ow;
  std::vector<double> wv(m.W.numel()), bv(m.b.numel()), xv(x.numel()), alpha(ny);
  for (std::size_t i = 0; i < m.W.numel(); ++i) wv[i] = m.W[i];
  for (std::size_t i = 0; i < m.b.numel(); ++i) bv[i] = m.b[i];
  for (std::size_t i = 0; i < x.numel(); ++i) xv[i] = x[i];
  for (std::size_t j = 0; j < ny; ++j) {
    alpha[j] = 0.3 + 0.9 * static_cast<double>((j * 29 + static_cast<std::size_t>(salt)) % 53) / 52.0;
  }

  std::vector<double> y0;
  (void)forward_weighted_loss_f64(wv, bv, xv, alpha, n, c_in, h, wd, out, kh, kw, pad, &y0);
  Tensor dy({n, out, oh, ow});
  for (std::size_t j = 0; j < ny; ++j) dy[j] = static_cast<float>(alpha[j] * y0[j]);

  const ConvGrad g_cpu = conv2d_backward(m, x, dy);
  const ConvGrad g_cuda = conv2d_backward_cuda(m, x, dy, /*need_dx=*/true);

  constexpr double kEps = 1e-5;
  double err_dx_cpu = 0.0, err_dx_cuda = 0.0, err_dw_cuda = 0.0, err_db_cuda = 0.0;

  for (std::size_t i = 0; i < xv.size(); ++i) {
    const double orig = xv[i];
    xv[i] = orig + kEps;
    const double lp = forward_weighted_loss_f64(wv, bv, xv, alpha, n, c_in, h, wd, out, kh, kw, pad);
    xv[i] = orig - kEps;
    const double lm = forward_weighted_loss_f64(wv, bv, xv, alpha, n, c_in, h, wd, out, kh, kw, pad);
    xv[i] = orig;
    const double num = (lp - lm) / (2.0 * kEps);
    err_dx_cpu = std::max(err_dx_cpu, std::fabs(static_cast<double>(g_cpu.dx[i]) - num));
    err_dx_cuda = std::max(err_dx_cuda, std::fabs(static_cast<double>(g_cuda.dx[i]) - num));
  }
  for (std::size_t i = 0; i < wv.size(); ++i) {
    const double orig = wv[i];
    wv[i] = orig + kEps;
    const double lp = forward_weighted_loss_f64(wv, bv, xv, alpha, n, c_in, h, wd, out, kh, kw, pad);
    wv[i] = orig - kEps;
    const double lm = forward_weighted_loss_f64(wv, bv, xv, alpha, n, c_in, h, wd, out, kh, kw, pad);
    wv[i] = orig;
    err_dw_cuda = std::max(err_dw_cuda,
                           std::fabs(static_cast<double>(g_cuda.dW[i]) - (lp - lm) / (2.0 * kEps)));
  }
  for (std::size_t i = 0; i < bv.size(); ++i) {
    const double orig = bv[i];
    bv[i] = orig + kEps;
    const double lp = forward_weighted_loss_f64(wv, bv, xv, alpha, n, c_in, h, wd, out, kh, kw, pad);
    bv[i] = orig - kEps;
    const double lm = forward_weighted_loss_f64(wv, bv, xv, alpha, n, c_in, h, wd, out, kh, kw, pad);
    bv[i] = orig;
    err_db_cuda = std::max(err_db_cuda,
                           std::fabs(static_cast<double>(g_cuda.db[i]) - (lp - lm) / (2.0 * kEps)));
  }

  std::cout << "dx gradcheck [" << tag << "] max|dx_cpu-num|=" << err_dx_cpu
            << " max|dx_cuda-num|=" << err_dx_cuda << " max|dW_cuda-num|=" << err_dw_cuda
            << " max|db_cuda-num|=" << err_db_cuda << "\n";
  check(std::isfinite(err_dx_cpu) && err_dx_cpu < 1e-5, std::string(tag) + ": CPU dx 有限差分误差 < 1e-5");
  check(std::isfinite(err_dx_cuda) && err_dx_cuda < 1e-5,
        std::string(tag) + ": CUDA dx 有限差分误差 < 1e-5");
  check(std::isfinite(err_dw_cuda) && err_dw_cuda < 1e-5,
        std::string(tag) + ": CUDA dW 有限差分误差 < 1e-5");
  check(std::isfinite(err_db_cuda) && err_db_cuda < 1e-5,
        std::string(tag) + ": CUDA db 有限差分误差 < 1e-5");

  // need_dx=false 分支：dx 应全零，dW/db 不受影响。
  const ConvGrad g_nodx = conv2d_backward_cuda(m, x, dy, /*need_dx=*/false);
  bool dx_zero = true, dw_same = true;
  for (std::size_t i = 0; i < g_nodx.dx.numel(); ++i) dx_zero = dx_zero && (g_nodx.dx[i] == 0.0f);
  for (std::size_t i = 0; i < g_nodx.dW.numel(); ++i) {
    dw_same = dw_same && (g_nodx.dW[i] == g_cuda.dW[i]);
  }
  check(dx_zero && dw_same, std::string(tag) + ": need_dx=false 时 dx 全零且 dW 不变");
}

}  // namespace

int main() {
  run_dx_gradcheck_case(2, 2, 5, 7, 3, 3, 3, 1, 10, "3x3_pad1");
  run_dx_gradcheck_case(2, 2, 6, 8, 2, 3, 3, 0, 20, "3x3_pad0");
  run_dx_gradcheck_case(1, 2, 5, 6, 3, 3, 5, 2, 30, "3x5_pad2_nonsquare");

  if (g_fail == 0) {
    std::cout << "test_conv_dx_gradcheck: PASS\n";
    return 0;
  }
  std::cerr << "test_conv_dx_gradcheck: " << g_fail << " 项失败\n";
  return 1;
}
