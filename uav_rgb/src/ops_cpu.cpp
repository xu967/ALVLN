#include "uavrgb/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "uavrgb/softmax_detail.hpp"

namespace uavrgb {
namespace {

// FP64 前向（gradcheck 专用）：logits[n,out] = b + x @ W^T，全程 double 累加。
std::vector<double> forward_f64(const std::vector<double>& w, const std::vector<double>& b,
                                const std::vector<double>& x, int n, int in, int out) {
  std::vector<double> y(static_cast<std::size_t>(n) * out, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int k = 0; k < out; ++k) {
      double acc = b[k];
      for (int j = 0; j < in; ++j) {
        acc += x[i * in + j] * w[k * in + j];
      }
      y[i * out + k] = acc;
    }
  }
  return y;
}

// FP64 stable softmax 交叉熵（按样本权重和归一化），避免 float 截断误差淹没中心差分。
// weights 为空表示无权重（w=1，wsum=n），此时与原实现逐位一致。
double softmax_ce_loss_f64(const std::vector<double>& logits, int n, int c,
                           const std::vector<int>& labels, const std::vector<float>& weights) {
  double total = 0.0;
  double wsum = 0.0;
  for (int i = 0; i < n; ++i) {
    const double* row = &logits[i * c];
    double mx = row[0];
    for (int j = 1; j < c; ++j) mx = std::max(mx, row[j]);
    double sum = 0.0;
    for (int j = 0; j < c; ++j) sum += std::exp(row[j] - mx);
    const double w = weights.empty() ? 1.0 : static_cast<double>(weights[labels[i]]);
    total += w * (-(row[labels[i]] - mx) + std::log(sum));
    wsum += w;
  }
  return total / wsum;
}

}  // namespace

Linear::Linear(int in_features, int out_features) : in(in_features), out(out_features) {
  W = Tensor::zeros({out, in});  // W[out][in]，row-major 下标 W[k*in + j]。
  b = Tensor::zeros({out});
}

Tensor linear_forward(const Linear& m, const Tensor& x) {
  // x[N,in] -> [N,out]：out[n,k] = b[k] + Σ_j x[n,j]·W[k,j]。
  detail::check_linear_params(m);
  if (x.ndim() != 2 || x.dims()[1] != m.in) {
    throw std::invalid_argument("linear_forward: x 形状必须为 [N, in]");
  }
  const std::int64_t n = x.dims()[0];
  Tensor y = Tensor::zeros({n, m.out});
  for (std::int64_t i = 0; i < n; ++i) {
    for (int k = 0; k < m.out; ++k) {
      float acc = m.b[k];
      for (int j = 0; j < m.in; ++j) {
        acc += x[i * m.in + j] * m.W[k * m.in + j];
      }
      y[i * m.out + k] = acc;
    }
  }
  return y;
}

LinearGrad linear_backward(const Linear& m, const Tensor& x, const Tensor& dlogits) {
  // 解析梯度：dW = dlogits^T @ x、db = dlogits 列和、dx = dlogits @ W。
  detail::check_linear_params(m);
  if (x.ndim() != 2 || x.dims()[1] != m.in) {
    throw std::invalid_argument("linear_backward: x 形状必须为 [N, in]");
  }
  if (dlogits.ndim() != 2 || dlogits.dims()[0] != x.dims()[0] || dlogits.dims()[1] != m.out) {
    throw std::invalid_argument("linear_backward: dlogits 形状必须为 [N, out]");
  }
  const std::int64_t n = x.dims()[0];
  LinearGrad g;
  g.dW = Tensor::zeros({m.out, m.in});
  g.db = Tensor::zeros({m.out});
  g.dx = Tensor::zeros({n, m.in});
  for (int k = 0; k < m.out; ++k) {
    for (int j = 0; j < m.in; ++j) {
      float acc = 0.0f;
      for (std::int64_t i = 0; i < n; ++i) {
        acc += dlogits[i * m.out + k] * x[i * m.in + j];
      }
      g.dW[k * m.in + j] = acc;
    }
    float accb = 0.0f;
    for (std::int64_t i = 0; i < n; ++i) {
      accb += dlogits[i * m.out + k];
    }
    g.db[k] = accb;
  }
  for (std::int64_t i = 0; i < n; ++i) {
    for (int j = 0; j < m.in; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < m.out; ++k) {
        acc += dlogits[i * m.out + k] * m.W[k * m.in + j];
      }
      g.dx[i * m.in + j] = acc;
    }
  }
  return g;
}

float softmax_ce_loss(const Tensor& logits, const std::vector<int>& labels) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("softmax_ce_loss: logits 形状必须为 [N, C]");
  }
  const std::int64_t n = logits.dims().at(0);
  const std::int64_t c = logits.dims().at(1);
  if (n == 0) {
    throw std::invalid_argument("softmax_ce_loss: 空 batch（N=0）");
  }
  if (c == 0) {
    throw std::invalid_argument("softmax_ce_loss: C 必须为正（实际 logits 为 [N,0]）");
  }
  if (static_cast<std::int64_t>(labels.size()) != n) {
    throw std::invalid_argument("labels 数量与 logits 行数不一致");
  }
  double total = 0.0;
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[i];
    if (label < 0 || label >= c) {
      throw std::invalid_argument("label 越界");
    }
    const float* row = logits.data() + i * c;
    float mx, sum;
    detail::softmax_row(row, c, &mx, &sum);
    // -log(softmax[label]) = -(logit[label]-mx) + log(Σ exp)。
    total += -static_cast<double>(row[label] - mx) + std::log(static_cast<double>(sum));
  }
  return static_cast<float>(total / static_cast<double>(n));
}

Tensor softmax_ce_grad(const Tensor& logits, const std::vector<int>& labels) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("softmax_ce_grad: logits 形状必须为 [N, C]");
  }
  const std::int64_t n = logits.dims().at(0);
  const std::int64_t c = logits.dims().at(1);
  if (n == 0) {
    throw std::invalid_argument("softmax_ce_grad: 空 batch（N=0）");
  }
  if (c == 0) {
    throw std::invalid_argument("softmax_ce_grad: C 必须为正（实际 logits 为 [N,0]）");
  }
  if (static_cast<std::int64_t>(labels.size()) != n) {
    throw std::invalid_argument("labels 数量与 logits 行数不一致");
  }
  Tensor g = Tensor::zeros({n, c});
  for (std::int64_t i = 0; i < n; ++i) {
    const int label = labels[i];
    if (label < 0 || label >= c) {
      throw std::invalid_argument("label 越界");
    }
    const float* row = logits.data() + i * c;
    float mx, sum;
    detail::softmax_row(row, c, &mx, &sum);
    for (std::int64_t j = 0; j < c; ++j) {
      const float p = std::exp(row[j] - mx) / sum;
      g[i * c + j] = (p - (j == label ? 1.0f : 0.0f)) / static_cast<float>(n);
    }
  }
  return g;
}

namespace {

// 无权/加权 gradcheck 共用的 FP64 中心差分主循环；weights 为空表示无权重。
double gradcheck_impl(const Linear& m, const Tensor& x, const std::vector<int>& labels,
                      const std::vector<float>& weights, double eps) {
  detail::check_linear_params(m);
  if (x.ndim() != 2) {
    throw std::invalid_argument("gradcheck_linear: x 必须为 2 维 [N,in]（实际 ndim=" +
                                std::to_string(x.ndim()) + "）");
  }
  if (x.dims()[1] != m.in) {
    throw std::invalid_argument("gradcheck_linear: x 列数应为 in=" + std::to_string(m.in) +
                                "（实际 " + std::to_string(x.dims()[1]) + "）");
  }
  if (x.dims()[0] == 0) {
    throw std::invalid_argument("gradcheck_linear: 空 batch（N=0）");
  }
  const int n = static_cast<int>(x.dims()[0]);
  const int in = m.in;
  const int out = m.out;

  // 解析梯度：dlogits = softmax_ce_grad；dW = x^T @ dlogits，db = dlogits 列和。
  const Tensor logits = linear_forward(m, x);
  const Tensor dlogits = weights.empty() ? softmax_ce_grad(logits, labels)
                                         : softmax_ce_grad_weighted(logits, labels, weights);
  std::vector<double> dW(static_cast<std::size_t>(out) * in, 0.0);
  std::vector<double> db(out, 0.0);
  for (int k = 0; k < out; ++k) {
    for (int j = 0; j < in; ++j) {
      double acc = 0.0;
      for (int i = 0; i < n; ++i) {
        acc += static_cast<double>(x[i * in + j]) * dlogits[i * out + k];
      }
      dW[k * in + j] = acc;
    }
    for (int i = 0; i < n; ++i) db[k] += dlogits[i * out + k];
  }

  // 参数/输入复制为 double，中心差分全程 FP64，避免 float 存储截断污染扰动步长。
  std::vector<double> w(static_cast<std::size_t>(out) * in), bvec(out), xvec(n * in);
  for (int i = 0; i < out * in; ++i) w[i] = m.W[i];
  for (int i = 0; i < out; ++i) bvec[i] = m.b[i];
  for (int i = 0; i < n * in; ++i) xvec[i] = x[i];

  double max_err = 0.0;
  auto central_diff = [&](double* p, double analytic) {
    const double orig = *p;
    *p = orig + eps;
    const double lp =
        softmax_ce_loss_f64(forward_f64(w, bvec, xvec, n, in, out), n, out, labels, weights);
    *p = orig - eps;
    const double lm =
        softmax_ce_loss_f64(forward_f64(w, bvec, xvec, n, in, out), n, out, labels, weights);
    *p = orig;
    max_err = std::max(max_err, std::fabs((lp - lm) / (2.0 * eps) - analytic));
  };

  for (int k = 0; k < out; ++k) {
    for (int j = 0; j < in; ++j) central_diff(&w[k * in + j], dW[k * in + j]);
    central_diff(&bvec[k], db[k]);
  }
  return max_err;
}

}  // namespace

double gradcheck_linear(const Linear& m, const Tensor& x, const std::vector<int>& labels,
                        double eps) {
  return gradcheck_impl(m, x, labels, {}, eps);
}

double gradcheck_linear_weighted(const Linear& m, const Tensor& x,
                                 const std::vector<int>& labels,
                                 const std::vector<float>& weights, double eps) {
  return gradcheck_impl(m, x, labels, weights, eps);
}

}  // namespace uavrgb
