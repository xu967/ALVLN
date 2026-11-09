// 单层 Linear 的 CUDA 实现：前向 y = x@W^T + b，反向 dW/db/dx。
// 只针对 sm_75（Turing）；不引入 cuBLAS/cuDNN，纯手写 kernel。
// kernel 经 ops_dev.hpp 的设备指针级函数暴露给常驻路径；host Tensor 包装只做分配/拷贝/同步，
// 供 CPU/CUDA 对拍。
#include "uavrgb/linear_cuda.hpp"

#include "cuda_check.hpp"
#include "uavrgb/ops_dev.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace uavrgb {
namespace {

// 元素数一律先用 int64 计算并校验上限：超过 INT_MAX 直接抛异常，绝不静默回绕后少算输出。
std::int64_t check_elems(std::int64_t total, const char* what) {
  if (total > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("linear_cuda: shape too large: ") + what + "=" +
                                std::to_string(total) + " > INT_MAX");
  }
  return total;
}

__global__ void linear_forward_kernel(const float* __restrict__ x,
                                      const float* __restrict__ w,
                                      const float* __restrict__ b,
                                      float* __restrict__ y, int n, int in, int out) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(n) * out) return;
  const std::int64_t i = idx / out;
  const int k = static_cast<int>(idx % out);
  float acc = b[k];
  for (int j = 0; j < in; ++j) {
    acc += x[i * in + j] * w[k * in + j];
  }
  y[i * out + k] = acc;
}

__global__ void linear_dw_kernel(const float* __restrict__ dlogits,
                                 const float* __restrict__ x,
                                 float* __restrict__ dw, int n, int in, int out) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(out) * in) return;
  const int k = static_cast<int>(idx / in);
  const int j = static_cast<int>(idx % in);
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) {
    acc += dlogits[static_cast<std::int64_t>(i) * out + k] *
           x[static_cast<std::int64_t>(i) * in + j];
  }
  dw[k * in + j] = acc;
}

__global__ void linear_db_kernel(const float* __restrict__ dlogits,
                                 float* __restrict__ db, int n, int out) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= out) return;
  float acc = 0.0f;
  for (int i = 0; i < n; ++i) {
    acc += dlogits[static_cast<std::int64_t>(i) * out + k];
  }
  db[k] = acc;
}

__global__ void linear_dx_kernel(const float* __restrict__ dlogits,
                                 const float* __restrict__ w,
                                 float* __restrict__ dx, int n, int in, int out) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(n) * in) return;
  const std::int64_t i = idx / in;
  const int j = static_cast<int>(idx % in);
  float acc = 0.0f;
  for (int k = 0; k < out; ++k) {
    acc += dlogits[i * out + k] * w[k * in + j];
  }
  dx[i * in + j] = acc;
}

DeviceBuf<float> upload(const Tensor& t) {
  DeviceBuf<float> d(t.numel() * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d.p, t.data(), t.numel() * sizeof(float), cudaMemcpyHostToDevice));
  return d;
}

void download(Tensor& dst, const float* src) {
  CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.numel() * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace

namespace dev {

void linear_forward(int n, int in, int out, const float* x, const float* w, const float* b,
                    float* y) {
  if (n == 0) return;  // 0 个 block 的 launch 会报 invalid configuration。
  linear_forward_kernel<<<blocks_for(static_cast<std::int64_t>(n) * out), kThreads>>>(x, w, b, y,
                                                                                    n, in, out);
  launched("linear_forward_kernel");
}

void linear_backward(int n, int in, int out, const float* x, const float* w, const float* dl,
                     float* dw, float* db, float* dx) {
  if (n == 0) return;
  linear_dw_kernel<<<blocks_for(static_cast<std::int64_t>(out) * in), kThreads>>>(dl, x, dw, n, in,
                                                                                out);
  launched("linear_dw_kernel");
  linear_db_kernel<<<blocks_for(out), kThreads>>>(dl, db, n, out);
  launched("linear_db_kernel");
  linear_dx_kernel<<<blocks_for(static_cast<std::int64_t>(n) * in), kThreads>>>(dl, w, dx, n, in,
                                                                              out);
  launched("linear_dx_kernel");
}

}  // namespace dev

Tensor linear_forward_cuda(const Linear& m, const Tensor& x) {
  detail::check_linear_params(m);
  if (x.ndim() != 2 || x.dims()[1] != m.in) {
    throw std::invalid_argument("linear_forward_cuda: x 形状必须为 [N, in]");
  }
  const std::int64_t n64 = x.dims()[0];
  check_elems(n64 * m.in, "x 元素数");
  check_elems(n64 * m.out, "y 元素数");
  check_elems(static_cast<std::int64_t>(m.out) * m.in, "W 元素数");
  Tensor y = Tensor::zeros({n64, m.out});
  if (n64 == 0) return y;  // 空 batch：直接返回形状正确的空张量。
  const DeviceBuf<float> d_x = upload(x);
  const DeviceBuf<float> d_w = upload(m.W);
  const DeviceBuf<float> d_b = upload(m.b);
  DeviceBuf<float> d_y(y.numel() * sizeof(float));
  dev::linear_forward(static_cast<int>(n64), m.in, m.out, d_x.p, d_w.p, d_b.p, d_y.p);
  CUDA_CHECK(cudaDeviceSynchronize());
  download(y, d_y.p);
  return y;
}

LinearGrad linear_backward_cuda(const Linear& m, const Tensor& x, const Tensor& dlogits) {
  detail::check_linear_params(m);
  if (x.ndim() != 2 || x.dims()[1] != m.in) {
    throw std::invalid_argument("linear_backward_cuda: x 形状必须为 [N, in]");
  }
  if (dlogits.ndim() != 2 || dlogits.dims()[0] != x.dims()[0] || dlogits.dims()[1] != m.out) {
    throw std::invalid_argument("linear_backward_cuda: dlogits 形状必须为 [N, out]");
  }
  const std::int64_t n64 = x.dims()[0];
  check_elems(n64 * m.in, "x/dx 元素数");
  check_elems(n64 * m.out, "dlogits 元素数");
  check_elems(static_cast<std::int64_t>(m.out) * m.in, "W/dW 元素数");
  LinearGrad g;
  g.dW = Tensor::zeros({m.out, m.in});
  g.db = Tensor::zeros({m.out});
  g.dx = Tensor::zeros({n64, m.in});
  if (n64 == 0) return g;  // 空 batch：全零梯度，不 launch kernel（0 block 非法）。
  const DeviceBuf<float> d_x = upload(x);
  const DeviceBuf<float> d_w = upload(m.W);
  const DeviceBuf<float> d_dl = upload(dlogits);
  DeviceBuf<float> d_dw(g.dW.numel() * sizeof(float));
  DeviceBuf<float> d_db(g.db.numel() * sizeof(float));
  DeviceBuf<float> d_dx(g.dx.numel() * sizeof(float));
  dev::linear_backward(static_cast<int>(n64), m.in, m.out, d_x.p, d_w.p, d_dl.p, d_dw.p, d_db.p,
                       d_dx.p);
  CUDA_CHECK(cudaDeviceSynchronize());
  download(g.dW, d_dw.p);
  download(g.db, d_db.p);
  download(g.dx, d_dx.p);
  return g;
}

}  // namespace uavrgb
