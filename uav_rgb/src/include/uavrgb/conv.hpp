#pragma once
// Conv2d 前向/反向的 CPU FP32 参考实现（NCHW，stride=1，pad 可配），作为 CUDA 对拍 oracle
// （doc 04 §6 N1/N3：FP64 finite-difference gradcheck 与 CPU/CUDA 对拍）。
#include "uavrgb/tensor.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>

namespace uavrgb {

// 单层 2D 卷积（stride=1）：权重布局 W[out,in,kh,kw]，bias[out]；x 为 NCHW [N,in,H,W]。
struct Conv2d {
  int in = 0;
  int out = 0;
  int kh = 3;
  int kw = 3;
  int pad = 0;
  Tensor W;  // [out, in, kh, kw]
  Tensor b;  // [out]

  Conv2d() = default;
  Conv2d(int in_channels, int out_channels, int kernel_h, int kernel_w, int padding);
};

namespace detail {

// Conv2d 的 W/b 形状不变量：字段是 public，模型组装/checkpoint 赋值可能绕过构造函数，
// 所有入口都必须先校验，否则按 out*in*kh*kw 读 W 会越界读。
inline void check_conv_params(const Conv2d& m) {
  if (m.in <= 0 || m.out <= 0 || m.kh <= 0 || m.kw <= 0) {
    throw std::invalid_argument("Conv2d: in/out/kh/kw 必须为正（实际 in=" + std::to_string(m.in) +
                                ", out=" + std::to_string(m.out) + ", kh=" + std::to_string(m.kh) +
                                ", kw=" + std::to_string(m.kw) + "）");
  }
  if (m.pad < 0) {
    throw std::invalid_argument("Conv2d: pad 不能为负（实际 " + std::to_string(m.pad) + "）");
  }
  const std::size_t w_expected = static_cast<std::size_t>(m.out) * static_cast<std::size_t>(m.in) *
                                 static_cast<std::size_t>(m.kh) * static_cast<std::size_t>(m.kw);
  if (m.W.numel() != w_expected) {
    throw std::invalid_argument("Conv2d: W.numel() 应为 out*in*kh*kw=" +
                                std::to_string(w_expected) + "（实际 " +
                                std::to_string(m.W.numel()) + "）");
  }
  if (m.b.numel() != static_cast<std::size_t>(m.out)) {
    throw std::invalid_argument("Conv2d: b.numel() 应为 out=" + std::to_string(m.out) +
                                "（实际 " + std::to_string(m.b.numel()) + "）");
  }
}

}  // namespace detail

// x[N,in,H,W] -> y[N,out,H',W']，H' = H + 2*pad - kh + 1（stride=1，向下取整）。
Tensor conv2d_forward(const Conv2d& m, const Tensor& x);

// 卷积层反向输出：dW[out,in,kh,kw]、db[out]、dx[N,in,H,W]。
struct ConvGrad {
  Tensor dW;
  Tensor db;
  Tensor dx;
};

// CPU FP32 反向参考：给定上游梯度 dy[N,out,H',W']，返回 dW/db/dx。
ConvGrad conv2d_backward(const Conv2d& m, const Tensor& x, const Tensor& dy);

// FP64 有限差分 gradcheck：loss = Σ y²，对每个 W/b 扰动中心差分，返回 max abs 误差。
double gradcheck_conv2d(const Conv2d& m, const Tensor& x, double eps = 1e-6);

}  // namespace uavrgb
