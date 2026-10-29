#pragma once
// ReLU / MaxPool2d / Adam 的 CPU FP32 参考与 CUDA 实现声明（NCHW，纯手写，无 cuDNN/cuBLAS）。
// CUDA 接口与 CPU 参考一一对称，便于 doc 04 §6 N1/N3 的 FP64 gradcheck 与 CPU/CUDA 对拍。
#include "uavrgb/tensor.hpp"

#include <vector>

namespace uavrgb {

// ---- ReLU ----
// 前向 y = max(x, 0)；反向 dx = dy * (x > 0)。
// 导数约定：x == 0 处取次梯度 0（CPU 与 CUDA 两处实现必须一致）；前向 relu(0) = 0。
Tensor relu_forward(const Tensor& x);
Tensor relu_backward(const Tensor& x, const Tensor& dy);
Tensor relu_forward_cuda(const Tensor& x);
Tensor relu_backward_cuda(const Tensor& x, const Tensor& dy);

// FP64 中心差分 gradcheck（loss = Σ relu(x)²，逐元素扰 x），返回 max abs 误差。
double gradcheck_relu(const Tensor& x, double eps = 1e-6);

// ---- MaxPool2d ----
// 方形窗口 k×k、步长 s（本阶段使用 k=s=2），NCHW；H' = (H-k)/s+1（向下取整）。
// argmax 记录窗口最大值在“单通道内”的 flatten 下标 ih*W+iw（float 存储，<2^24 时精确）。
// 并列约定：按 (ki,kj) 行优先扫描，严格 > 才更新，即取第一个最大值；CPU/CUDA 一致。
struct MaxPool2d {
  int k = 2;
  int s = 2;
  MaxPool2d() = default;
  MaxPool2d(int kernel, int stride);
};

struct MaxPoolResult {
  Tensor y;       // [N,C,H',W']
  Tensor argmax;  // [N,C,H',W']，值域 [0, H*W)
};

MaxPoolResult maxpool2d_forward(const MaxPool2d& m, const Tensor& x);
// 反向：dx 先清零，再把 dy 散回每窗口 argmax 位置（重叠窗口则累加）。
Tensor maxpool2d_backward(const MaxPool2d& m, const Tensor& x, const Tensor& argmax,
                          const Tensor& dy);

MaxPoolResult maxpool2d_forward_cuda(const MaxPool2d& m, const Tensor& x);
Tensor maxpool2d_backward_cuda(const MaxPool2d& m, const Tensor& x, const Tensor& argmax,
                               const Tensor& dy);

// FP64 中心差分 gradcheck（loss = Σ y²，逐元素扰 x），返回 max abs 误差。
double gradcheck_maxpool2d(const MaxPool2d& m, const Tensor& x, double eps = 1e-6);

// ---- Adam ----
// 状态 m/v/t 必须与 param 一起 checkpoint；t 为已完成 step 数，本次更新用 t' = t+1：
//   m = b1*m + (1-b1)*g;  v = b2*v + (1-b2)*g²;
//   p -= lr * (m/(1-b1^t')) / (sqrt(v/(1-b2^t')) + eps)
struct Adam {
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
  int t = 0;
  Tensor m;  // [param.dims()]，一阶矩
  Tensor v;  // [param.dims()]，二阶矩
};

// 初始化：m/v 与 param 同形全零、t = 0。
Adam adam_make(const Tensor& param, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
               float eps = 1e-8f);

// 原地更新 param、m、v、t（CPU FP32 参考）。
void adam_step(Adam& st, Tensor& param, const Tensor& grad);
void adam_step_cuda(Adam& st, Tensor& param, const Tensor& grad);

// ---- 梯度裁剪（跨张量 global L2 norm） ----
// norm = sqrt(Σ_i Σ_j g_i[j]²)，double 累加后返回 float；空列表返回 0。
float global_grad_norm(const std::vector<const Tensor*>& grads);
float global_grad_norm_cuda(const std::vector<const Tensor*>& grads);
// 原地缩放所有梯度：coef = min(1, max_norm / (norm + 1e-6))，返回 coef。
float clip_grad_norm(std::vector<Tensor*> grads, float max_norm);
float clip_grad_norm_cuda(std::vector<Tensor*> grads, float max_norm);

}  // namespace uavrgb
