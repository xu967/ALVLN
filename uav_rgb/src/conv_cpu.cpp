#include "uavrgb/conv.hpp"

#include <stdexcept>

namespace uavrgb {

Conv2d::Conv2d(int in_channels, int out_channels, int kernel_h, int kernel_w, int padding)
    : in(in_channels), out(out_channels), kh(kernel_h), kw(kernel_w), pad(padding) {
  if (kh <= 0 || kw <= 0 || in_channels <= 0 || out_channels <= 0) {
    throw std::invalid_argument("Conv2d: 通道/核尺寸必须为正");
  }
  if (padding < 0) {
    throw std::invalid_argument("Conv2d: padding 不能为负");
  }
  W = Tensor::zeros({out, in, kh, kw});  // W[out][in][kh][kw]，row-major。
  b = Tensor::zeros({out});
}

Tensor conv2d_forward(const Conv2d& m, const Tensor& x) {
  // NCHW：out[n,oc,oi,oj] = b[oc] + Σ_{ic,ki,kj} x[n,ic,oi-pad+ki,oj-pad+kj]·W[oc,ic,ki,kj]。
  detail::check_conv_params(m);
  if (x.ndim() != 4 || x.dims()[1] != m.in) {
    throw std::invalid_argument("conv2d_forward: x 形状必须为 [N, in, H, W]");
  }
  const int n = static_cast<int>(x.dims()[0]);
  const int c_in = m.in;
  const int h = static_cast<int>(x.dims()[2]);
  const int wd = static_cast<int>(x.dims()[3]);
  const int oh = h + 2 * m.pad - m.kh + 1;
  const int ow = wd + 2 * m.pad - m.kw + 1;
  if (oh <= 0 || ow <= 0) {
    throw std::invalid_argument("conv2d_forward: 输出尺寸非正（pad 太小）");
  }
  Tensor y = Tensor::zeros({n, m.out, oh, ow});
  for (int ni = 0; ni < n; ++ni) {
    for (int oc = 0; oc < m.out; ++oc) {
      for (int oi = 0; oi < oh; ++oi) {
        for (int oj = 0; oj < ow; ++oj) {
          float acc = m.b[oc];
          for (int ic = 0; ic < c_in; ++ic) {
            for (int ki = 0; ki < m.kh; ++ki) {
              for (int kj = 0; kj < m.kw; ++kj) {
                const int ih = oi - m.pad + ki;
                const int iw = oj - m.pad + kj;
                if (ih >= 0 && ih < h && iw >= 0 && iw < wd) {
                  acc += x[(static_cast<std::size_t>(ni) * c_in + ic) * h * wd + ih * wd + iw] *
                         m.W[(static_cast<std::size_t>(oc) * c_in + ic) * m.kh * m.kw +
                             ki * m.kw + kj];
                }
              }
            }
          }
          y[(static_cast<std::size_t>(ni) * m.out + oc) * oh * ow + oi * ow + oj] = acc;
        }
      }
    }
  }
  return y;
}

ConvGrad conv2d_backward(const Conv2d& m, const Tensor& x, const Tensor& dy) {
  detail::check_conv_params(m);
  if (x.ndim() != 4 || x.dims()[1] != m.in) {
    throw std::invalid_argument("conv2d_backward: x 形状必须为 [N, in, H, W]");
  }
  const int n = static_cast<int>(x.dims()[0]);
  const int c_in = m.in;
  const int h = static_cast<int>(x.dims()[2]);
  const int wd = static_cast<int>(x.dims()[3]);
  const int oh = h + 2 * m.pad - m.kh + 1;
  const int ow = wd + 2 * m.pad - m.kw + 1;
  if (oh <= 0 || ow <= 0) {
    throw std::invalid_argument("conv2d_backward: 输出尺寸非正（pad 太小）");
  }
  if (dy.ndim() != 4 || dy.dims()[0] != n || dy.dims()[1] != m.out || dy.dims()[2] != oh ||
      dy.dims()[3] != ow) {
    throw std::invalid_argument("conv2d_backward: dy 形状必须为 [N, out, H', W']");
  }
  ConvGrad g;
  g.dW = Tensor::zeros({m.out, c_in, m.kh, m.kw});
  g.db = Tensor::zeros({m.out});
  g.dx = Tensor::zeros({n, c_in, h, wd});

  // dW[oc,ic,ki,kj] = Σ_{n,oi,oj} dy[n,oc,oi,oj]·x[n,ic,oi-pad+ki,oj-pad+kj]（越界项为 0）。
  for (int oc = 0; oc < m.out; ++oc) {
    for (int ic = 0; ic < c_in; ++ic) {
      for (int ki = 0; ki < m.kh; ++ki) {
        for (int kj = 0; kj < m.kw; ++kj) {
          float acc = 0.0f;
          for (int ni = 0; ni < n; ++ni) {
            for (int oi = 0; oi < oh; ++oi) {
              for (int oj = 0; oj < ow; ++oj) {
                const int ih = oi - m.pad + ki;
                const int iw = oj - m.pad + kj;
                if (ih >= 0 && ih < h && iw >= 0 && iw < wd) {
                  acc += dy[(static_cast<std::size_t>(ni) * m.out + oc) * oh * ow + oi * ow + oj] *
                         x[(static_cast<std::size_t>(ni) * c_in + ic) * h * wd + ih * wd + iw];
                }
              }
            }
          }
          g.dW[(static_cast<std::size_t>(oc) * c_in + ic) * m.kh * m.kw + ki * m.kw + kj] = acc;
        }
      }
    }
  }

  // db[oc] = Σ_{n,oi,oj} dy[n,oc,oi,oj]。
  for (int oc = 0; oc < m.out; ++oc) {
    float acc = 0.0f;
    for (int ni = 0; ni < n; ++ni) {
      for (int oi = 0; oi < oh; ++oi) {
        for (int oj = 0; oj < ow; ++oj) {
          acc += dy[(static_cast<std::size_t>(ni) * m.out + oc) * oh * ow + oi * ow + oj];
        }
      }
    }
    g.db[oc] = acc;
  }

  // dx[n,ic,ih,iw] = Σ_{oc,ki,kj: 0<=oh<H',0<=ow<W'} dy[n,oc,oh,ow]·W[oc,ic,ki,kj]，
  // 其中 oh = ih + pad - ki、ow = iw + pad - kj（stride=1）。
  for (int ni = 0; ni < n; ++ni) {
    for (int ic = 0; ic < c_in; ++ic) {
      for (int ih = 0; ih < h; ++ih) {
        for (int iw = 0; iw < wd; ++iw) {
          float acc = 0.0f;
          for (int oc = 0; oc < m.out; ++oc) {
            for (int ki = 0; ki < m.kh; ++ki) {
              for (int kj = 0; kj < m.kw; ++kj) {
                const int oi = ih + m.pad - ki;
                const int oj = iw + m.pad - kj;
                if (oi >= 0 && oi < oh && oj >= 0 && oj < ow) {
                  acc += dy[(static_cast<std::size_t>(ni) * m.out + oc) * oh * ow + oi * ow + oj] *
                         m.W[(static_cast<std::size_t>(oc) * c_in + ic) * m.kh * m.kw +
                             ki * m.kw + kj];
                }
              }
            }
          }
          g.dx[(static_cast<std::size_t>(ni) * c_in + ic) * h * wd + ih * wd + iw] = acc;
        }
      }
    }
  }
  return g;
}

}  // namespace uavrgb
