#pragma once
// 单层 Conv2d 的 CUDA 前向/反向（纯 CUDA + C++，不依赖 cuBLAS/cuDNN；NCHW，stride=1）。
// 设备内存由实现内部临时分配，接口与 conv.hpp 的 CPU 参考完全对称，便于对拍（doc 04 §6 N3）。
#include "uavrgb/conv.hpp"

namespace uavrgb {

// 前向 y[N,out,H',W'] = x[N,in,H,W] * W[out,in,kh,kw] + b，结果写回 host Tensor。
Tensor conv2d_forward_cuda(const Conv2d& m, const Tensor& x);

// 反向：给定上游梯度 dy[N,out,H',W']，返回 dW[out,in,kh,kw]、db[out]、dx[N,in,H,W]。
ConvGrad conv2d_backward_cuda(const Conv2d& m, const Tensor& x, const Tensor& dy);

}  // namespace uavrgb
