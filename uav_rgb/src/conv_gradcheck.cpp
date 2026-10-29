// Conv2d 的 FP64 有限差分 gradcheck（loss = Σ y²），从 conv_cpu.cpp 拆出以守住 280 行预算。
#include "uavrgb/conv.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb {
namespace {

// FP64 前向（gradcheck 专用）：直接卷积，全程 double 累加；下标约定与 conv2d_forward 一致。
std::vector<double> conv_forward_f64(const std::vector<double>& w, const std::vector<double>& b,
                                     const std::vector<double>& x, int n, int c_in, int h,
                                     int wd, int out, int kh, int kw, int pad) {
  const int oh = h + 2 * pad - kh + 1;
  const int ow = wd + 2 * pad - kw + 1;
  std::vector<double> y(static_cast<std::size_t>(n) * out * oh * ow, 0.0);
  for (int ni = 0; ni < n; ++ni) {
    for (int oc = 0; oc < out; ++oc) {
      for (int oi = 0; oi < oh; ++oi) {
        for (int oj = 0; oj < ow; ++oj) {
          double acc = b[oc];
          for (int ic = 0; ic < c_in; ++ic) {
            for (int ki = 0; ki < kh; ++ki) {
              for (int kj = 0; kj < kw; ++kj) {
                const int ih = oi - pad + ki;
                const int iw = oj - pad + kj;
                if (ih >= 0 && ih < h && iw >= 0 && iw < wd) {
                  acc += x[(static_cast<std::size_t>(ni) * c_in + ic) * h * wd + ih * wd + iw] *
                         w[(static_cast<std::size_t>(oc) * c_in + ic) * kh * kw + ki * kw + kj];
                }
              }
            }
          }
          y[(static_cast<std::size_t>(ni) * out + oc) * oh * ow + oi * ow + oj] = acc;
        }
      }
    }
  }
  return y;
}

// FP64 loss = Σ y²（二次型，对参数的中心差分无截断误差）。
double conv_loss_f64(const std::vector<double>& y) {
  double s = 0.0;
  for (double v : y) s += v * v;
  return s;
}

}  // namespace

double gradcheck_conv2d(const Conv2d& m, const Tensor& x, double eps) {
  // 先校验 rank/通道/W/b：rank<4 直接读 dims()[3] 会堆越界（审核 CPU #1）。
  detail::check_conv_params(m);
  if (x.ndim() != 4) {
    throw std::invalid_argument("gradcheck_conv2d: x 必须为 4 维 [N,in,H,W]（实际 ndim=" +
                                std::to_string(x.ndim()) + "）");
  }
  if (x.dims()[1] != m.in) {
    throw std::invalid_argument("gradcheck_conv2d: x 通道数应为 in=" + std::to_string(m.in) +
                                "（实际 " + std::to_string(x.dims()[1]) + "）");
  }
  if (x.dims()[0] == 0) {
    throw std::invalid_argument("gradcheck_conv2d: 空 batch（N=0）");
  }
  const int n = static_cast<int>(x.dims()[0]);
  const int c_in = m.in;
  const int h = static_cast<int>(x.dims()[2]);
  const int wd = static_cast<int>(x.dims()[3]);
  const int out = m.out;
  const int kh = m.kh;
  const int kw = m.kw;
  const int pad = m.pad;
  const int oh = h + 2 * pad - kh + 1;
  const int ow = wd + 2 * pad - kw + 1;
  if (oh <= 0 || ow <= 0) {
    throw std::invalid_argument("gradcheck_conv2d: 输出尺寸非正（pad 太小）");
  }

  // 参数/输入复制为 double，中心差分全程 FP64，避免 float 存储截断污染扰动步长。
  std::vector<double> wv(m.W.numel()), bv(m.b.numel()), xv(x.numel());
  for (std::size_t i = 0; i < m.W.numel(); ++i) wv[i] = m.W[i];
  for (std::size_t i = 0; i < m.b.numel(); ++i) bv[i] = m.b[i];
  for (std::size_t i = 0; i < x.numel(); ++i) xv[i] = x[i];

  // 解析梯度（FP64）：L = Σ y²，dL/dW、dL/db 直接用 dy = 2y 手工累加。
  const std::vector<double> y = conv_forward_f64(wv, bv, xv, n, c_in, h, wd, out, kh, kw, pad);
  std::vector<double> dW(static_cast<std::size_t>(out) * c_in * kh * kw, 0.0);
  std::vector<double> db(out, 0.0);
  for (int oc = 0; oc < out; ++oc) {
    for (int ic = 0; ic < c_in; ++ic) {
      for (int ki = 0; ki < kh; ++ki) {
        for (int kj = 0; kj < kw; ++kj) {
          double acc = 0.0;
          for (int ni = 0; ni < n; ++ni) {
            for (int oi = 0; oi < oh; ++oi) {
              for (int oj = 0; oj < ow; ++oj) {
                const int ih = oi - pad + ki;
                const int iw = oj - pad + kj;
                if (ih >= 0 && ih < h && iw >= 0 && iw < wd) {
                  acc += 2.0 * y[(static_cast<std::size_t>(ni) * out + oc) * oh * ow + oi * ow + oj] *
                         xv[(static_cast<std::size_t>(ni) * c_in + ic) * h * wd + ih * wd + iw];
                }
              }
            }
          }
          dW[(static_cast<std::size_t>(oc) * c_in + ic) * kh * kw + ki * kw + kj] = acc;
        }
      }
    }
    double accb = 0.0;
    for (int ni = 0; ni < n; ++ni) {
      for (int oi = 0; oi < oh; ++oi) {
        for (int oj = 0; oj < ow; ++oj) {
          accb += 2.0 * y[(static_cast<std::size_t>(ni) * out + oc) * oh * ow + oi * ow + oj];
        }
      }
    }
    db[oc] = accb;
  }

  double max_err = 0.0;
  auto central_diff = [&](double* p, double analytic) {
    const double orig = *p;
    *p = orig + eps;
    const double lp = conv_loss_f64(conv_forward_f64(wv, bv, xv, n, c_in, h, wd, out, kh, kw, pad));
    *p = orig - eps;
    const double lm = conv_loss_f64(conv_forward_f64(wv, bv, xv, n, c_in, h, wd, out, kh, kw, pad));
    *p = orig;
    max_err = std::max(max_err, std::fabs((lp - lm) / (2.0 * eps) - analytic));
  };

  for (int oc = 0; oc < out; ++oc) {
    for (int ic = 0; ic < c_in; ++ic) {
      for (int ki = 0; ki < kh; ++ki) {
        for (int kj = 0; kj < kw; ++kj) {
          central_diff(&wv[(static_cast<std::size_t>(oc) * c_in + ic) * kh * kw + ki * kw + kj],
                       dW[(static_cast<std::size_t>(oc) * c_in + ic) * kh * kw + ki * kw + kj]);
        }
      }
    }
    central_diff(&bv[oc], db[oc]);
  }
  return max_err;
}

}  // namespace uavrgb
