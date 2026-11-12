// ReLU / MaxPool2d / Adam / 梯度裁剪的 CUDA 实现：纯手写 kernel，不引入 cuBLAS/cuDNN。
// 只针对 sm_75（Turing）；接口与 nn.hpp 的 CPU 参考一一对称，供 CPU/CUDA 对拍（doc 04 §6 N3）。
// kernel 经 ops_dev.hpp 的设备指针级函数暴露给常驻路径；host Tensor 包装只做分配/拷贝/同步。
#include "uavrgb/nn.hpp"

#include "cuda_check.hpp"
#include "uavrgb/ops_dev.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace uavrgb {
namespace {

// ReLU 前向：x==0 → 0；NaN 继续传播（压成 0 会把坏权重藏起来）。
__global__ void relu_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                    int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  y[i] = (x[i] > 0.0f || isnan(x[i])) ? x[i] : 0.0f;
}

// ReLU 反向：dx = dy * (x > 0)；x==0 取次梯度 0；NaN 继续传播。
__global__ void relu_backward_kernel(const float* __restrict__ x, const float* __restrict__ dy,
                                     float* __restrict__ dx, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  dx[i] = (x[i] > 0.0f || isnan(x[i])) ? dy[i] : 0.0f;
}

// MaxPool 前向：每线程一个输出元素；窗口内严格 > 才更新 → 并列取行优先第一个。
__global__ void maxpool_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                       float* __restrict__ arg, int h, int wd, int oh, int ow,
                                       int k, int s, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int oj = idx % ow;
  const int oi = (idx / ow) % oh;
  const int plane = idx / (ow * oh);  // n*C+c
  const long long base = static_cast<long long>(plane) * h * wd;
  const int ih0 = oi * s;
  const int iw0 = oj * s;
  int best_idx = ih0 * wd + iw0;
  float best = x[base + best_idx];
  for (int ki = 0; ki < k; ++ki) {
    for (int kj = 0; kj < k; ++kj) {
      const int p = (ih0 + ki) * wd + iw0 + kj;
      const float v = x[base + p];
      if (v > best) {
        best = v;
        best_idx = p;
      }
    }
  }
  y[idx] = best;
  arg[idx] = static_cast<float>(best_idx);
}

// MaxPool 反向：每线程一个输出元素，把 dy 原子累加回 argmax 位置（重叠窗口正确）。
__global__ void maxpool_backward_kernel(const float* __restrict__ arg,
                                        const float* __restrict__ dy, float* __restrict__ dx,
                                        int h, int wd, int oh, int ow, int total) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int plane = idx / (ow * oh);  // n*C+c
  const int p = static_cast<int>(arg[idx]);
  atomicAdd(dx + static_cast<long long>(plane) * h * wd + p, dy[idx]);
}

// Adam 单步：每线程一个参数元素，m/v/param 原地更新；wd>0 即 AdamW 的解耦权重衰减。
__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ g,
                                 float* __restrict__ m, float* __restrict__ v, float lr,
                                 float b1, float b2, float bc1, float bc2, float eps, float wd,
                                 int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float gi = g[i];
  const float mi = b1 * m[i] + (1.0f - b1) * gi;
  const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  p[i] -= lr * ((mi / bc1) / (sqrtf(vi / bc2) + eps) + wd * p[i]);
}

// 梯度裁剪归约：double atomicAdd 累加平方和（与 CPU 的 double 累加对拍）。
__global__ void sumsq_kernel(const float* __restrict__ x, double* __restrict__ out, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const double v = x[i];
  atomicAdd(out, v * v);
}

// 梯度裁剪缩放：x *= coef。
__global__ void scale_kernel(float* __restrict__ x, float coef, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  x[i] *= coef;
}

// ---- host 侧形状校验（与 nn_cpu.cpp 同一合同） ----
dev::PoolShape check_pool_dims(const MaxPool2d& m, const Tensor& x) {
  if (m.k <= 0 || m.s <= 0) throw std::invalid_argument("maxpool2d: k/s 必须为正");
  if (x.ndim() != 4) throw std::invalid_argument("maxpool2d: x 形状必须为 [N, C, H, W]");
  dev::PoolShape d;
  d.n = static_cast<int>(x.dims()[0]);
  d.c = static_cast<int>(x.dims()[1]);
  d.h = static_cast<int>(x.dims()[2]);
  d.w = static_cast<int>(x.dims()[3]);
  d.k = m.k;
  d.s = m.s;
  // 必须在除法前拒绝不完整窗口，避免负数向零截断掩盖越界。
  if (d.h < m.k || d.w < m.k) throw std::invalid_argument("maxpool2d: k 大于 H/W");
  d.oh = (d.h - m.k) / m.s + 1;
  d.ow = (d.w - m.k) / m.s + 1;
  if (d.oh <= 0 || d.ow <= 0) throw std::invalid_argument("maxpool2d: 输出尺寸非正（k 大于 H/W）");
  return d;
}

DeviceBuf<float> upload(const Tensor& t) {
  DeviceBuf<float> d(t.numel() * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d.p, t.data(), t.numel() * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}

Tensor download(const std::vector<std::int64_t>& dims, const float* src) {
  Tensor t = Tensor::zeros(dims);
  CUDA_CHECK(cudaMemcpy(t.data(), src, t.numel() * sizeof(float), cudaMemcpyDeviceToHost));
  return t;
}

}  // namespace

namespace dev {

void relu_forward(const float* x, float* y, int total) {
  if (total == 0) return;  // 0 个元素时不能 launch：grid 为 0，cudaErrorInvalidConfiguration。
  relu_forward_kernel<<<blocks_for(total), kThreads>>>(x, y, total);
  launched("relu_forward_kernel");
}

void relu_backward(const float* x, const float* dy, float* dx, int total) {
  if (total == 0) return;
  relu_backward_kernel<<<blocks_for(total), kThreads>>>(x, dy, dx, total);
  launched("relu_backward_kernel");
}

void maxpool_forward(const PoolShape& d, const float* x, float* y, float* argmax) {
  const int total = d.n * d.c * d.oh * d.ow;
  if (total == 0) return;
  maxpool_forward_kernel<<<blocks_for(total), kThreads>>>(x, y, argmax, d.h, d.w, d.oh, d.ow, d.k,
                                                         d.s, total);
  launched("maxpool_forward_kernel");
}

void maxpool_backward(const PoolShape& d, const float* argmax, const float* dy, float* dx) {
  const int total = d.n * d.c * d.oh * d.ow;
  if (total == 0) return;
  CUDA_CHECK(cudaMemsetAsync(dx, 0, static_cast<std::size_t>(d.n) * d.c * d.h * d.w * sizeof(float)));
  maxpool_backward_kernel<<<blocks_for(total), kThreads>>>(argmax, dy, dx, d.h, d.w, d.oh, d.ow,
                                                          total);
  launched("maxpool_backward_kernel");
}

void adam_step(float* p, const float* g, float* m, float* v, float lr, float b1, float b2,
               float bc1, float bc2, float eps, float weight_decay, int total) {
  if (total == 0) return;
  adam_step_kernel<<<blocks_for(total), kThreads>>>(p, g, m, v, lr, b1, b2, bc1, bc2, eps,
                                                   weight_decay, total);
  launched("adam_step_kernel");
}

void sumsq(const float* x, double* accum, int total) {
  if (total == 0) return;
  sumsq_kernel<<<blocks_for(total), kThreads>>>(x, accum, total);
  launched("sumsq_kernel");
}

void scale(float* x, float coef, int total) {
  if (total == 0) return;
  scale_kernel<<<blocks_for(total), kThreads>>>(x, coef, total);
  launched("scale_kernel");
}

}  // namespace dev

Tensor relu_forward_cuda(const Tensor& x) {
  if (x.numel() == 0) return Tensor::zeros(x.dims());
  const DeviceBuf<float> d_x = upload(x);
  DeviceBuf<float> d_y(x.numel() * sizeof(float));
  dev::relu_forward(d_x.p, d_y.p, static_cast<int>(x.numel()));
  CUDA_CHECK(cudaDeviceSynchronize());
  return download(x.dims(), d_y.p);
}

Tensor relu_backward_cuda(const Tensor& x, const Tensor& dy) {
  if (x.dims() != dy.dims()) throw std::invalid_argument("relu_backward_cuda: x/dy 形状不一致");
  if (x.numel() == 0) return Tensor::zeros(x.dims());
  const DeviceBuf<float> d_x = upload(x);
  const DeviceBuf<float> d_dy = upload(dy);
  DeviceBuf<float> d_dx(x.numel() * sizeof(float));
  dev::relu_backward(d_x.p, d_dy.p, d_dx.p, static_cast<int>(x.numel()));
  CUDA_CHECK(cudaDeviceSynchronize());
  return download(x.dims(), d_dx.p);
}

MaxPoolResult maxpool2d_forward_cuda(const MaxPool2d& m, const Tensor& x) {
  const dev::PoolShape d = check_pool_dims(m, x);
  const std::vector<std::int64_t> out_dims{d.n, d.c, d.oh, d.ow};
  MaxPoolResult r;
  if (d.n * d.c * d.oh * d.ow == 0) {
    r.y = Tensor::zeros(out_dims);
    r.argmax = Tensor::zeros(out_dims);
    return r;
  }
  const DeviceBuf<float> d_x = upload(x);
  const std::size_t out_bytes = static_cast<std::size_t>(d.n) * d.c * d.oh * d.ow * sizeof(float);
  DeviceBuf<float> d_y(out_bytes);
  DeviceBuf<float> d_arg(out_bytes);
  dev::maxpool_forward(d, d_x.p, d_y.p, d_arg.p);
  CUDA_CHECK(cudaDeviceSynchronize());
  r.y = download(out_dims, d_y.p);
  r.argmax = download(out_dims, d_arg.p);
  return r;
}

Tensor maxpool2d_backward_cuda(const MaxPool2d& m, const Tensor& x, const Tensor& argmax,
                               const Tensor& dy) {
  const dev::PoolShape d = check_pool_dims(m, x);
  if (argmax.ndim() != 4 || argmax.dims()[0] != d.n || argmax.dims()[1] != d.c ||
      argmax.dims()[2] != d.oh || argmax.dims()[3] != d.ow) {
    throw std::invalid_argument("maxpool2d_backward_cuda: argmax 形状必须为 [N, C, H', W']");
  }
  if (dy.dims() != argmax.dims()) {
    throw std::invalid_argument("maxpool2d_backward_cuda: dy 形状必须与 argmax 一致");
  }
  // 信任边界：argmax 会直接作为 GPU 写偏移，先在 host 校验范围与整数性，防越界写。
  const int plane_in = d.h * d.w;
  for (std::size_t i = 0; i < argmax.numel(); ++i) {
    const float a = argmax[i];
    const int p = static_cast<int>(a);
    if (a != static_cast<float>(p) || p < 0 || p >= plane_in) {
      throw std::invalid_argument("maxpool2d_backward_cuda: argmax 越界或非整数");
    }
  }
  if (argmax.numel() == 0) return Tensor::zeros(x.dims());
  const DeviceBuf<float> d_arg = upload(argmax);
  const DeviceBuf<float> d_dy = upload(dy);
  DeviceBuf<float> d_dx(x.numel() * sizeof(float));
  dev::maxpool_backward(d, d_arg.p, d_dy.p, d_dx.p);
  CUDA_CHECK(cudaDeviceSynchronize());
  return download(x.dims(), d_dx.p);
}

void adam_step_cuda(Adam& st, Tensor& param, const Tensor& grad) {
  if (grad.dims() != param.dims() || st.m.dims() != param.dims() || st.v.dims() != param.dims()) {
    throw std::invalid_argument("adam_step_cuda: param/grad/m/v 形状必须一致");
  }
  st.t += 1;
  // 与 CPU 参考一致：host double 算偏差校正，再以同一 float 传入 kernel。
  const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(st.beta1), st.t));
  const float bc2 = static_cast<float>(1.0 - std::pow(static_cast<double>(st.beta2), st.t));
  if (param.numel() == 0) return;  // st.t 已自增，与 CPU 参考的步数一致
  DeviceBuf<float> d_p = upload(param);
  const DeviceBuf<float> d_g = upload(grad);
  DeviceBuf<float> d_m = upload(st.m);
  DeviceBuf<float> d_v = upload(st.v);
  dev::adam_step(d_p.p, d_g.p, d_m.p, d_v.p, st.lr, st.beta1, st.beta2, bc1, bc2, st.eps, 0.0f,
                 static_cast<int>(param.numel()));
  CUDA_CHECK(cudaDeviceSynchronize());
  param = download(param.dims(), d_p.p);
  st.m = download(param.dims(), d_m.p);
  st.v = download(param.dims(), d_v.p);
}

float global_grad_norm_cuda(const std::vector<const Tensor*>& grads) {
  DeviceBuf<double> d_sum(sizeof(double));
  CUDA_CHECK(cudaMemset(d_sum.p, 0, sizeof(double)));
  for (const Tensor* t : grads) {
    if (!t) throw std::invalid_argument("global_grad_norm_cuda: 张量指针为空");
    if (t->numel() == 0) continue;
    const DeviceBuf<float> d_x = upload(*t);
    dev::sumsq(d_x.p, d_sum.p, static_cast<int>(t->numel()));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  double sum = 0.0;
  CUDA_CHECK(cudaMemcpy(&sum, d_sum.p, sizeof(double), cudaMemcpyDeviceToHost));
  if (!std::isfinite(sum)) {
    throw std::runtime_error("global_grad_norm_cuda: 梯度含非有限值");
  }
  const double norm = std::sqrt(sum);
  // 每个分量都有限、但 norm 超出 float 时，转 float 会变 inf，clip 系数变成 0，
  // 梯度被静默清零；必须拒绝而不是继续 Adam。
  if (!(norm <= static_cast<double>(std::numeric_limits<float>::max()))) {
    throw std::runtime_error("global_grad_norm_cuda: 梯度范数超出 float 可表示范围");
  }
  return static_cast<float>(norm);
}

float clip_grad_norm_cuda(std::vector<Tensor*> grads, float max_norm) {
  if (max_norm < 0.0f) throw std::invalid_argument("clip_grad_norm_cuda: max_norm 不能为负");
  const std::vector<const Tensor*> views(grads.begin(), grads.end());
  const float norm = global_grad_norm_cuda(views);
  const float raw = max_norm / (norm + 1e-6f);
  const float coef = raw > 1.0f ? 1.0f : raw;  // 与 CPU 的 min(1, ...) 一致
  if (coef >= 1.0f) return coef;
  for (Tensor* t : grads) {
    if (t->numel() == 0) continue;
    DeviceBuf<float> d_x = upload(*t);
    dev::scale(d_x.p, coef, static_cast<int>(t->numel()));
    CUDA_CHECK(cudaDeviceSynchronize());
    *t = download(t->dims(), d_x.p);
  }
  return coef;
}

void enable_cuda_blocking_sync() {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
}

}  // namespace uavrgb
