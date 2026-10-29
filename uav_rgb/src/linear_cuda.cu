// 单层 Linear 的 CUDA 实现：前向 y = x@W^T + b，反向 dW/db/dx。
// 只针对 sm_75（Turing）；不引入 cuBLAS/cuDNN，纯手写 kernel，供 CPU/CUDA 对拍。
#include "uavrgb/linear_cuda.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace uavrgb {
namespace {

// 统一 CUDA API 错误检查：抛 std::runtime_error 而不是 std::exit，调用方可以 catch 后
// 降 batch/重试；栈上 DeviceBuf 等 RAII 正常析构，checkpoint 落盘不会被跳过。
#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    const cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" +      \
                               std::to_string(__LINE__) + ": " +                  \
                               cudaGetErrorString(err_));                         \
    }                                                                             \
  } while (0)

// 每次 kernel launch 后必查：先 cudaGetLastError 抓异步错误，再同步等 kernel 结束。
void check_after_kernel(const char* name) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel ") + name + ": " + cudaGetErrorString(err));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

// 极简设备缓冲 RAII：析构自动释放，异常路径也不泄漏。
struct DeviceBuf {
  float* p = nullptr;
  DeviceBuf() = default;
  explicit DeviceBuf(std::size_t bytes) { CUDA_CHECK(cudaMalloc(&p, bytes)); }
  ~DeviceBuf() {
    if (p) cudaFree(p);
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
};

constexpr int kThreads = 256;

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

}  // namespace

Tensor linear_forward_cuda(const Linear& m, const Tensor& x) {
  detail::check_linear_params(m);
  if (x.ndim() != 2 || x.dims()[1] != m.in) {
    throw std::invalid_argument("linear_forward_cuda: x 形状必须为 [N, in]");
  }
  const std::int64_t n64 = x.dims()[0];
  const int in = m.in;
  const int out = m.out;
  const std::int64_t n_in = check_elems(n64 * in, "x 元素数");
  const std::int64_t n_out = check_elems(n64 * out, "y 元素数");
  check_elems(static_cast<std::int64_t>(out) * in, "W 元素数");
  if (n64 == 0) {
    // 空 batch：0 个 block 的 launch 会报 invalid configuration，直接返回形状正确的空张量。
    return Tensor::zeros({n64, out});
  }

  DeviceBuf d_x(static_cast<std::size_t>(n_in) * sizeof(float));
  DeviceBuf d_w(static_cast<std::size_t>(out) * in * sizeof(float));
  DeviceBuf d_b(static_cast<std::size_t>(out) * sizeof(float));
  DeviceBuf d_y(static_cast<std::size_t>(n_out) * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_x.p, x.data(), static_cast<std::size_t>(n_in) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.p, m.W.data(), static_cast<std::size_t>(out) * in * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.p, m.b.data(), static_cast<std::size_t>(out) * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int n = static_cast<int>(n64);
  const int blocks = static_cast<int>((n_out + kThreads - 1) / kThreads);
  linear_forward_kernel<<<blocks, kThreads>>>(d_x.p, d_w.p, d_b.p, d_y.p, n, in, out);
  check_after_kernel("linear_forward_kernel");

  Tensor y = Tensor::zeros({n, out});
  CUDA_CHECK(cudaMemcpy(y.data(), d_y.p, static_cast<std::size_t>(n_out) * sizeof(float),
                        cudaMemcpyDeviceToHost));
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
  const int in = m.in;
  const int out = m.out;
  const std::int64_t n_in = check_elems(n64 * in, "x/dx 元素数");
  const std::int64_t n_out = check_elems(n64 * out, "dlogits 元素数");
  const std::int64_t n_w = check_elems(static_cast<std::int64_t>(out) * in, "W/dW 元素数");

  LinearGrad g;
  g.dW = Tensor::zeros({out, in});
  g.db = Tensor::zeros({out});
  g.dx = Tensor::zeros({n64, in});
  if (n64 == 0) return g;  // 空 batch：全零梯度，不 launch kernel（0 block 非法）。

  DeviceBuf d_x(static_cast<std::size_t>(n_in) * sizeof(float));
  DeviceBuf d_w(static_cast<std::size_t>(n_w) * sizeof(float));
  DeviceBuf d_dl(static_cast<std::size_t>(n_out) * sizeof(float));
  DeviceBuf d_dw(static_cast<std::size_t>(n_w) * sizeof(float));
  DeviceBuf d_db(static_cast<std::size_t>(out) * sizeof(float));
  DeviceBuf d_dx(static_cast<std::size_t>(n_in) * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_x.p, x.data(), static_cast<std::size_t>(n_in) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.p, m.W.data(), static_cast<std::size_t>(n_w) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dl.p, dlogits.data(), static_cast<std::size_t>(n_out) * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int n = static_cast<int>(n64);
  int blocks = static_cast<int>((n_w + kThreads - 1) / kThreads);
  linear_dw_kernel<<<blocks, kThreads>>>(d_dl.p, d_x.p, d_dw.p, n, in, out);
  check_after_kernel("linear_dw_kernel");

  blocks = static_cast<int>((static_cast<std::int64_t>(out) + kThreads - 1) / kThreads);
  linear_db_kernel<<<blocks, kThreads>>>(d_dl.p, d_db.p, n, out);
  check_after_kernel("linear_db_kernel");

  blocks = static_cast<int>((n_in + kThreads - 1) / kThreads);
  linear_dx_kernel<<<blocks, kThreads>>>(d_dl.p, d_w.p, d_dx.p, n, in, out);
  check_after_kernel("linear_dx_kernel");

  CUDA_CHECK(cudaMemcpy(g.dW.data(), d_dw.p, static_cast<std::size_t>(n_w) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g.db.data(), d_db.p, static_cast<std::size_t>(out) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g.dx.data(), d_dx.p, static_cast<std::size_t>(n_in) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return g;
}

}  // namespace uavrgb
