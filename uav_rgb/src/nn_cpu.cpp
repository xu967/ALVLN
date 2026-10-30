// ReLU / MaxPool2d / Adam 的 CPU FP32 参考实现，作为 CUDA kernel 的数值 oracle
// （doc 04 §6 N1/N3：FP64 中心差分 gradcheck + CPU/CUDA 对拍）。
#include "uavrgb/nn.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace uavrgb {
namespace {

// 4D NCHW 输出尺寸：H' = (H-k)/s+1；k/s 非法或输出非正即抛。
struct PoolDims {
  int n, c, h, wd, oh, ow;
};

PoolDims check_pool_dims(const MaxPool2d& m, const Tensor& x) {
  if (m.k <= 0 || m.s <= 0) throw std::invalid_argument("maxpool2d: k/s 必须为正");
  if (x.ndim() != 4) throw std::invalid_argument("maxpool2d: x 形状必须为 [N, C, H, W]");
  PoolDims d;
  d.n = static_cast<int>(x.dims()[0]);
  d.c = static_cast<int>(x.dims()[1]);
  d.h = static_cast<int>(x.dims()[2]);
  d.wd = static_cast<int>(x.dims()[3]);
  // C++ 负数除法向零截断，H=1,k=2,s=2 会误算出一个窗口并越界。
  if (d.h < m.k || d.wd < m.k) throw std::invalid_argument("maxpool2d: k 大于 H/W");
  d.oh = (d.h - m.k) / m.s + 1;
  d.ow = (d.wd - m.k) / m.s + 1;
  if (d.oh <= 0 || d.ow <= 0) throw std::invalid_argument("maxpool2d: 输出尺寸非正（k 大于 H/W）");
  return d;
}

// FP64 前向（gradcheck 专用）：同时返回 y 与每窗口 argmax（单通道 flatten 下标）。
void pool_forward_f64(const std::vector<double>& x, int n, int c, int h, int wd, int k, int s,
                      std::vector<double>* y, std::vector<int>* arg) {
  const int oh = (h - k) / s + 1;
  const int ow = (wd - k) / s + 1;
  y->assign(static_cast<std::size_t>(n) * c * oh * ow, 0.0);
  arg->assign(y->size(), 0);
  for (int ni = 0; ni < n; ++ni) {
    for (int ci = 0; ci < c; ++ci) {
      for (int oi = 0; oi < oh; ++oi) {
        for (int oj = 0; oj < ow; ++oj) {
          const std::size_t plane = (static_cast<std::size_t>(ni) * c + ci) * h * wd;
          int best_idx = (oi * s) * wd + oj * s;
          double best = x[plane + best_idx];
          for (int ki = 0; ki < k; ++ki) {
            for (int kj = 0; kj < k; ++kj) {
              const int idx = (oi * s + ki) * wd + oj * s + kj;
              if (x[plane + idx] > best) {  // 严格 >：并列取行优先第一个
                best = x[plane + idx];
                best_idx = idx;
              }
            }
          }
          const std::size_t o = ((static_cast<std::size_t>(ni) * c + ci) * oh + oi) * ow + oj;
          (*y)[o] = best;
          (*arg)[o] = best_idx;
        }
      }
    }
  }
}

}  // namespace

Tensor relu_forward(const Tensor& x) {
  Tensor y = Tensor::zeros(x.dims());
  // NaN 必须继续传播：压成 0 会把坏权重/溢出藏起来，训练照常 exit 0。
  for (std::size_t i = 0; i < x.numel(); ++i) {
    y[i] = (x[i] > 0.0f || std::isnan(x[i])) ? x[i] : 0.0f;
  }
  return y;
}

Tensor relu_backward(const Tensor& x, const Tensor& dy) {
  if (x.dims() != dy.dims()) throw std::invalid_argument("relu_backward: x/dy 形状不一致");
  Tensor dx = Tensor::zeros(x.dims());
  // 次梯度约定：x == 0 时导数取 0（与 CUDA kernel 注释及实现一致）；NaN 继续传播。
  for (std::size_t i = 0; i < x.numel(); ++i) {
    dx[i] = (x[i] > 0.0f || std::isnan(x[i])) ? dy[i] : 0.0f;
  }
  return dx;
}

MaxPool2d::MaxPool2d(int kernel, int stride) : k(kernel), s(stride) {
  if (k <= 0 || s <= 0) throw std::invalid_argument("MaxPool2d: k/s 必须为正");
}

MaxPoolResult maxpool2d_forward(const MaxPool2d& m, const Tensor& x) {
  const PoolDims d = check_pool_dims(m, x);
  MaxPoolResult r;
  r.y = Tensor::zeros({d.n, d.c, d.oh, d.ow});
  r.argmax = Tensor::zeros({d.n, d.c, d.oh, d.ow});
  for (int ni = 0; ni < d.n; ++ni) {
    for (int ci = 0; ci < d.c; ++ci) {
      for (int oi = 0; oi < d.oh; ++oi) {
        for (int oj = 0; oj < d.ow; ++oj) {
          const std::size_t plane = (static_cast<std::size_t>(ni) * d.c + ci) * d.h * d.wd;
          int best_idx = (oi * m.s) * d.wd + oj * m.s;
          float best = x[plane + best_idx];
          for (int ki = 0; ki < m.k; ++ki) {
            for (int kj = 0; kj < m.k; ++kj) {
              const int idx = (oi * m.s + ki) * d.wd + oj * m.s + kj;
              if (x[plane + idx] > best) {  // 严格 >：并列取行优先第一个
                best = x[plane + idx];
                best_idx = idx;
              }
            }
          }
          const std::size_t o = ((static_cast<std::size_t>(ni) * d.c + ci) * d.oh + oi) * d.ow + oj;
          r.y[o] = best;
          r.argmax[o] = static_cast<float>(best_idx);
        }
      }
    }
  }
  return r;
}

Tensor maxpool2d_backward(const MaxPool2d& m, const Tensor& x, const Tensor& argmax,
                          const Tensor& dy) {
  const PoolDims d = check_pool_dims(m, x);
  if (argmax.ndim() != 4 || argmax.dims()[0] != d.n || argmax.dims()[1] != d.c ||
      argmax.dims()[2] != d.oh || argmax.dims()[3] != d.ow) {
    throw std::invalid_argument("maxpool2d_backward: argmax 形状必须为 [N, C, H', W']");
  }
  if (dy.dims() != argmax.dims()) {
    throw std::invalid_argument("maxpool2d_backward: dy 形状必须与 argmax 一致");
  }
  Tensor dx = Tensor::zeros(x.dims());
  const int plane_in = d.h * d.wd;
  const std::size_t per_plane = static_cast<std::size_t>(d.oh) * d.ow;
  for (std::size_t i = 0; i < argmax.numel(); ++i) {
    const float a = argmax[i];
    const int idx = static_cast<int>(a);
    if (a != static_cast<float>(idx) || idx < 0 || idx >= plane_in) {
      throw std::invalid_argument("maxpool2d_backward: argmax 越界或非整数");
    }
    const std::size_t plane = i / per_plane;  // n*C+c；重叠窗口时同位置累加
    dx[plane * plane_in + idx] += dy[i];
  }
  return dx;
}

Adam adam_make(const Tensor& param, float lr, float beta1, float beta2, float eps) {
  Adam st;
  st.lr = lr;
  st.beta1 = beta1;
  st.beta2 = beta2;
  st.eps = eps;
  st.m = Tensor::zeros(param.dims());
  st.v = Tensor::zeros(param.dims());
  return st;
}

void adam_step(Adam& st, Tensor& param, const Tensor& grad) {
  if (grad.dims() != param.dims() || st.m.dims() != param.dims() || st.v.dims() != param.dims()) {
    throw std::invalid_argument("adam_step: param/grad/m/v 形状必须一致");
  }
  st.t += 1;
  // 偏差校正用 host double 算一次再转 float；CUDA 侧接收同一组 float，保证两边逐步同步。
  const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(st.beta1), st.t));
  const float bc2 = static_cast<float>(1.0 - std::pow(static_cast<double>(st.beta2), st.t));
  for (std::size_t i = 0; i < param.numel(); ++i) {
    const float g = grad[i];
    const float m = st.beta1 * st.m[i] + (1.0f - st.beta1) * g;
    const float v = st.beta2 * st.v[i] + (1.0f - st.beta2) * g * g;
    st.m[i] = m;
    st.v[i] = v;
    param[i] -= st.lr * (m / bc1) / (std::sqrt(v / bc2) + st.eps);
  }
}

float global_grad_norm(const std::vector<const Tensor*>& grads) {
  double sum = 0.0;
  for (const Tensor* t : grads) {
    if (!t) throw std::invalid_argument("global_grad_norm: 张量指针为空");
    for (std::size_t i = 0; i < t->numel(); ++i) {
      const double v = (*t)[i];
      sum += v * v;
    }
  }
  if (!std::isfinite(sum)) {
    throw std::runtime_error("global_grad_norm: 梯度含非有限值");
  }
  const double norm = std::sqrt(sum);
  // 每个分量都有限、但 norm 超出 float 时，转 float 会变 inf，clip 系数变成 0，
  // 梯度被静默清零；必须拒绝而不是继续 Adam。
  if (!(norm <= static_cast<double>(std::numeric_limits<float>::max()))) {
    throw std::runtime_error("global_grad_norm: 梯度范数超出 float 可表示范围");
  }
  return static_cast<float>(norm);
}

float clip_grad_norm(std::vector<Tensor*> grads, float max_norm) {
  if (max_norm < 0.0f) throw std::invalid_argument("clip_grad_norm: max_norm 不能为负");
  const std::vector<const Tensor*> views(grads.begin(), grads.end());
  const float norm = global_grad_norm(views);
  const float coef = std::min(1.0f, max_norm / (norm + 1e-6f));
  if (coef < 1.0f) {
    for (Tensor* t : grads) {
      for (std::size_t i = 0; i < t->numel(); ++i) (*t)[i] *= coef;
    }
  }
  return coef;
}

double gradcheck_relu(const Tensor& x, double eps) {
  std::vector<double> xv(x.numel());
  for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = x[i];
  auto loss = [&]() {
    double s = 0.0;
    for (double t : xv) {
      const double r = t > 0.0 ? t : 0.0;
      s += r * r;
    }
    return s;
  };
  double max_err = 0.0;
  for (std::size_t i = 0; i < xv.size(); ++i) {
    const double orig = xv[i];
    xv[i] = orig + eps;
    const double lp = loss();
    xv[i] = orig - eps;
    const double lm = loss();
    xv[i] = orig;
    const double analytic = orig > 0.0 ? 2.0 * orig : 0.0;  // x==0 取次梯度 0
    const double diff = std::fabs((lp - lm) / (2.0 * eps) - analytic);
    if (!std::isfinite(diff)) return std::numeric_limits<double>::infinity();
    max_err = std::max(max_err, diff);
  }
  return max_err;
}

double gradcheck_maxpool2d(const MaxPool2d& m, const Tensor& x, double eps) {
  const PoolDims d = check_pool_dims(m, x);
  std::vector<double> xv(x.numel());
  for (std::size_t i = 0; i < xv.size(); ++i) xv[i] = x[i];
  std::vector<double> y;
  std::vector<int> arg;
  auto loss = [&]() {  // 每次重算前向，顺带填充 y/arg
    pool_forward_f64(xv, d.n, d.c, d.h, d.wd, m.k, m.s, &y, &arg);
    double s = 0.0;
    for (double v : y) s += v * v;
    return s;
  };
  (void)loss();  // 解析梯度用：dy = 2y 散回 argmax
  std::vector<double> dx(xv.size(), 0.0);
  for (std::size_t o = 0; o < y.size(); ++o) {
    const std::size_t plane = o / (static_cast<std::size_t>(d.oh) * d.ow);
    dx[plane * d.h * d.wd + arg[o]] += 2.0 * y[o];
  }
  double max_err = 0.0;
  for (std::size_t i = 0; i < xv.size(); ++i) {
    const double orig = xv[i];
    xv[i] = orig + eps;
    const double lp = loss();
    xv[i] = orig - eps;
    const double lm = loss();
    xv[i] = orig;
    const double diff = std::fabs((lp - lm) / (2.0 * eps) - dx[i]);
    if (!std::isfinite(diff)) return std::numeric_limits<double>::infinity();
    max_err = std::max(max_err, diff);
  }
  return max_err;
}

}  // namespace uavrgb
