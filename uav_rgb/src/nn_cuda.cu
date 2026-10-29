// ReLU / MaxPool2d / Adam 的 CUDA 实现：纯手写 kernel，不引入 cuBLAS/cuDNN。
// 只针对 sm_75（Turing）；接口与 nn.hpp 的 CPU 参考一一对称，供 CPU/CUDA 对拍（doc 04 §6 N3）。
#include "uavrgb/nn.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace uavrgb {
namespace {

// 统一 CUDA API 错误检查：抛 runtime_error 而不是 std::exit——与 conv/linear/softmax
// 三个 .cu 保持一致。训练中途显存或驱动出错时栈上 RAII 能正常析构，train_main 的
// catch 能收到，`save_checkpoint` 不会被静默跳过。
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err_ = (call);                                                  \
    if (err_ != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" +    \
                               std::to_string(__LINE__) + ": " +                \
                               cudaGetErrorString(err_));                       \
    }                                                                           \
  } while (0)

// 每次 kernel launch 后必查：先 cudaGetLastError 抓异步错误，再同步等 kernel 结束。
void check_after_kernel(const char* name) {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel ") + name + ": " +
                             cudaGetErrorString(err));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// 极简设备缓冲 RAII：析构自动释放，异常路径也不泄漏。
template <typename T>
struct DeviceBuf {
  T* p = nullptr;
  DeviceBuf() = default;
  explicit DeviceBuf(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
  ~DeviceBuf() {
    if (p) cudaFree(p);
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
};

constexpr int kThreads = 256;

// ReLU 前向：x==0 → 0。
__global__ void relu_forward_kernel(const float* __restrict__ x, float* __restrict__ y,
                                    int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}

// ReLU 反向：dx = dy * (x > 0)；x==0 取次梯度 0（与 nn_cpu.cpp 注释/实现一致）。
__global__ void relu_backward_kernel(const float* __restrict__ x, const float* __restrict__ dy,
                                     float* __restrict__ dx, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  dx[i] = x[i] > 0.0f ? dy[i] : 0.0f;
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
  const float a = arg[idx];
  const int p = static_cast<int>(a);
  atomicAdd(dx + static_cast<long long>(plane) * h * wd + p, dy[idx]);
}

// Adam 单步：每线程一个参数元素，m/v/param 原地更新。
__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ g,
                                 float* __restrict__ m, float* __restrict__ v, float lr,
                                 float b1, float b2, float bc1, float bc2, float eps,
                                 int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float gi = g[i];
  const float mi = b1 * m[i] + (1.0f - b1) * gi;
  const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
  m[i] = mi;
  v[i] = vi;
  p[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
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
  // 必须在除法前拒绝不完整窗口，避免负数向零截断掩盖越界。
  if (d.h < m.k || d.wd < m.k) throw std::invalid_argument("maxpool2d: k 大于 H/W");
  d.oh = (d.h - m.k) / m.s + 1;
  d.ow = (d.wd - m.k) / m.s + 1;
  if (d.oh <= 0 || d.ow <= 0) throw std::invalid_argument("maxpool2d: 输出尺寸非正（k 大于 H/W）");
  return d;
}

}  // namespace

Tensor relu_forward_cuda(const Tensor& x) {
  const int total = static_cast<int>(x.numel());
  // 0 个元素时不能 launch：grid 会算成 0，cudaErrorInvalidConfiguration。
  if (total == 0) return Tensor::zeros(x.dims());
  DeviceBuf<float> d_x(static_cast<std::size_t>(total) * sizeof(float));
  DeviceBuf<float> d_y(static_cast<std::size_t>(total) * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d_x.p, x.data(), static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyHostToDevice));
  relu_forward_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(d_x.p, d_y.p, total);
  check_after_kernel("relu_forward_kernel");
  Tensor y = Tensor::zeros(x.dims());
  CUDA_CHECK(cudaMemcpy(y.data(), d_y.p, static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return y;
}

Tensor relu_backward_cuda(const Tensor& x, const Tensor& dy) {
  if (x.dims() != dy.dims()) throw std::invalid_argument("relu_backward_cuda: x/dy 形状不一致");
  const int total = static_cast<int>(x.numel());
  if (total == 0) return Tensor::zeros(x.dims());
  DeviceBuf<float> d_x(static_cast<std::size_t>(total) * sizeof(float));
  DeviceBuf<float> d_dy(static_cast<std::size_t>(total) * sizeof(float));
  DeviceBuf<float> d_dx(static_cast<std::size_t>(total) * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d_x.p, x.data(), static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy.p, dy.data(), static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyHostToDevice));
  relu_backward_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(d_x.p, d_dy.p, d_dx.p,
                                                                       total);
  check_after_kernel("relu_backward_kernel");
  Tensor dx = Tensor::zeros(x.dims());
  CUDA_CHECK(cudaMemcpy(dx.data(), d_dx.p, static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return dx;
}

MaxPoolResult maxpool2d_forward_cuda(const MaxPool2d& m, const Tensor& x) {
  const PoolDims d = check_pool_dims(m, x);
  const int total_in = d.n * d.c * d.h * d.wd;
  const int total_out = d.n * d.c * d.oh * d.ow;
  if (total_out == 0) {
    MaxPoolResult empty;
    empty.y = Tensor::zeros({d.n, d.c, d.oh, d.ow});
    empty.argmax = Tensor::zeros({d.n, d.c, d.oh, d.ow});
    return empty;
  }
  DeviceBuf<float> d_x(static_cast<std::size_t>(total_in) * sizeof(float));
  DeviceBuf<float> d_y(static_cast<std::size_t>(total_out) * sizeof(float));
  DeviceBuf<float> d_arg(static_cast<std::size_t>(total_out) * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d_x.p, x.data(), static_cast<std::size_t>(total_in) * sizeof(float),
                        cudaMemcpyHostToDevice));
  maxpool_forward_kernel<<<(total_out + kThreads - 1) / kThreads, kThreads>>>(
      d_x.p, d_y.p, d_arg.p, d.h, d.wd, d.oh, d.ow, m.k, m.s, total_out);
  check_after_kernel("maxpool_forward_kernel");

  MaxPoolResult r;
  r.y = Tensor::zeros({d.n, d.c, d.oh, d.ow});
  r.argmax = Tensor::zeros({d.n, d.c, d.oh, d.ow});
  CUDA_CHECK(cudaMemcpy(r.y.data(), d_y.p, static_cast<std::size_t>(total_out) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(r.argmax.data(), d_arg.p,
                        static_cast<std::size_t>(total_out) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return r;
}

Tensor maxpool2d_backward_cuda(const MaxPool2d& m, const Tensor& x, const Tensor& argmax,
                               const Tensor& dy) {
  const PoolDims d = check_pool_dims(m, x);
  if (argmax.ndim() != 4 || argmax.dims()[0] != d.n || argmax.dims()[1] != d.c ||
      argmax.dims()[2] != d.oh || argmax.dims()[3] != d.ow) {
    throw std::invalid_argument("maxpool2d_backward_cuda: argmax 形状必须为 [N, C, H', W']");
  }
  if (dy.dims() != argmax.dims()) {
    throw std::invalid_argument("maxpool2d_backward_cuda: dy 形状必须与 argmax 一致");
  }
  // 信任边界：argmax 会直接作为 GPU 写偏移，先在 host 校验范围与整数性，防越界写。
  const int plane_in = d.h * d.wd;
  for (std::size_t i = 0; i < argmax.numel(); ++i) {
    const float a = argmax[i];
    const int p = static_cast<int>(a);
    if (a != static_cast<float>(p) || p < 0 || p >= plane_in) {
      throw std::invalid_argument("maxpool2d_backward_cuda: argmax 越界或非整数");
    }
  }
  const int total_in = d.n * d.c * d.h * d.wd;
  const int total_out = d.n * d.c * d.oh * d.ow;
  if (total_out == 0) return Tensor::zeros(x.dims());
  DeviceBuf<float> d_arg(static_cast<std::size_t>(total_out) * sizeof(float));
  DeviceBuf<float> d_dy(static_cast<std::size_t>(total_out) * sizeof(float));
  DeviceBuf<float> d_dx(static_cast<std::size_t>(total_in) * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d_arg.p, argmax.data(), static_cast<std::size_t>(total_out) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy.p, dy.data(), static_cast<std::size_t>(total_out) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_dx.p, 0, static_cast<std::size_t>(total_in) * sizeof(float)));
  maxpool_backward_kernel<<<(total_out + kThreads - 1) / kThreads, kThreads>>>(
      d_arg.p, d_dy.p, d_dx.p, d.h, d.wd, d.oh, d.ow, total_out);
  check_after_kernel("maxpool_backward_kernel");
  Tensor dx = Tensor::zeros(x.dims());
  CUDA_CHECK(cudaMemcpy(dx.data(), d_dx.p, static_cast<std::size_t>(total_in) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return dx;
}

void adam_step_cuda(Adam& st, Tensor& param, const Tensor& grad) {
  if (grad.dims() != param.dims() || st.m.dims() != param.dims() || st.v.dims() != param.dims()) {
    throw std::invalid_argument("adam_step_cuda: param/grad/m/v 形状必须一致");
  }
  st.t += 1;
  // 与 CPU 参考一致：host double 算偏差校正，再以同一 float 传入 kernel。
  const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(st.beta1), st.t));
  const float bc2 = static_cast<float>(1.0 - std::pow(static_cast<double>(st.beta2), st.t));
  const int total = static_cast<int>(param.numel());
  if (total == 0) return;  // st.t 已自增，与 CPU 参考的步数一致
  const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(float);
  DeviceBuf<float> d_p(bytes), d_g(bytes), d_m(bytes), d_v(bytes);
  CUDA_CHECK(cudaMemcpy(d_p.p, param.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_g.p, grad.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_m.p, st.m.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v.p, st.v.data(), bytes, cudaMemcpyHostToDevice));
  adam_step_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(
      d_p.p, d_g.p, d_m.p, d_v.p, st.lr, st.beta1, st.beta2, bc1, bc2, st.eps, total);
  check_after_kernel("adam_step_kernel");
  CUDA_CHECK(cudaMemcpy(param.data(), d_p.p, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(st.m.data(), d_m.p, bytes, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(st.v.data(), d_v.p, bytes, cudaMemcpyDeviceToHost));
}

float global_grad_norm_cuda(const std::vector<const Tensor*>& grads) {
  DeviceBuf<double> d_sum(sizeof(double));
  CUDA_CHECK(cudaMemset(d_sum.p, 0, sizeof(double)));
  for (const Tensor* t : grads) {
    if (!t) throw std::invalid_argument("global_grad_norm_cuda: 张量指针为空");
    const int total = static_cast<int>(t->numel());
    if (total == 0) continue;
    DeviceBuf<float> d_x(static_cast<std::size_t>(total) * sizeof(float));
    CUDA_CHECK(cudaMemcpy(d_x.p, t->data(), static_cast<std::size_t>(total) * sizeof(float),
                          cudaMemcpyHostToDevice));
    sumsq_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(d_x.p, d_sum.p, total);
    check_after_kernel("sumsq_kernel");
  }
  double sum = 0.0;
  CUDA_CHECK(cudaMemcpy(&sum, d_sum.p, sizeof(double), cudaMemcpyDeviceToHost));
  return static_cast<float>(std::sqrt(sum));
}

float clip_grad_norm_cuda(std::vector<Tensor*> grads, float max_norm) {
  if (max_norm < 0.0f) throw std::invalid_argument("clip_grad_norm_cuda: max_norm 不能为负");
  const std::vector<const Tensor*> views(grads.begin(), grads.end());
  const float norm = global_grad_norm_cuda(views);
  const float raw = max_norm / (norm + 1e-6f);
  const float coef = raw > 1.0f ? 1.0f : raw;  // 与 CPU 的 min(1, ...) 一致
  if (coef >= 1.0f) return coef;
  for (Tensor* t : grads) {
    const int total = static_cast<int>(t->numel());
    if (total == 0) continue;
    const std::size_t bytes = static_cast<std::size_t>(total) * sizeof(float);
    DeviceBuf<float> d_x(bytes);
    CUDA_CHECK(cudaMemcpy(d_x.p, t->data(), bytes, cudaMemcpyHostToDevice));
    scale_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(d_x.p, coef, total);
    check_after_kernel("scale_kernel");
    CUDA_CHECK(cudaMemcpy(t->data(), d_x.p, bytes, cudaMemcpyDeviceToHost));
  }
  return coef;
}

}  // namespace uavrgb
