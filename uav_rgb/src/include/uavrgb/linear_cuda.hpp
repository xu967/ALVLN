#pragma once
// 单层 Linear 的 CUDA 前向/反向（纯 CUDA + C++，不依赖 cuBLAS/cuDNN）。
// 设备内存由实现内部临时分配，接口与 ops.hpp 的 CPU 参考完全对称，便于对拍（doc 04 §6 N3）。
#include "uavrgb/ops.hpp"

namespace uavrgb {

// 前向 y[N,out] = x[N,in] @ W^T + b，结果写回 host Tensor。
Tensor linear_forward_cuda(const Linear& m, const Tensor& x);

// 反向：给定上游梯度 dlogits[N,out]，返回 dW[out,in]、db[out]、dx[N,in]。
LinearGrad linear_backward_cuda(const Linear& m, const Tensor& x, const Tensor& dlogits);

}  // namespace uavrgb
