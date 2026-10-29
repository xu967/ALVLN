// 单层 Conv2d 的 CUDA 实现：前向 + 反向（dW/db/dx），NCHW、stride=1、pad 可配。
// 只针对 sm_75（Turing）；不引入 cuBLAS/cuDNN，纯手写直接卷积 kernel，供 CPU/CUDA 对拍。
#include "uavrgb/conv_cuda.hpp"

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
    throw std::invalid_argument(std::string("conv2d_cuda: shape too large: ") + what + "=" +
                                std::to_string(total) + " > INT_MAX");
  }
  return total;
}

// 前向：每线程一个输出元素 (n,oc,oi,oj)，直接卷积累加。
__global__ void conv2d_forward_kernel(const float* __restrict__ x, const float* __restrict__ w,
                                      const float* __restrict__ b, float* __restrict__ y, int n,
                                      int c_in, int h, int wd, int out, int kh, int kw, int pad) {
  // int64 参与运算：2*pad 在 int 下可能溢出（host 侧已保证 oh/ow 落在 int 范围）。
  const int oh = static_cast<int>(static_cast<std::int64_t>(h) + 2 * static_cast<std::int64_t>(pad) -
                                  kh + 1);
  const int ow = static_cast<int>(static_cast<std::int64_t>(wd) + 2 * static_cast<std::int64_t>(pad) -
                                  kw + 1);
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(n) * out * oh * ow;
  if (idx >= total) return;
  const int oj = static_cast<int>(idx % ow);
  const int oi = static_cast<int>((idx / ow) % oh);
  const int oc = static_cast<int>((idx / (static_cast<std::int64_t>(ow) * oh)) % out);
  const int ni = static_cast<int>(idx / (static_cast<std::int64_t>(ow) * oh * out));
  float acc = b[oc];
  for (int ic = 0; ic < c_in; ++ic) {
    for (int ki = 0; ki < kh; ++ki) {
      for (int kj = 0; kj < kw; ++kj) {
        const int ih = oi - pad + ki;
        const int iw = oj - pad + kj;
        if (ih >= 0 && ih < h && iw >= 0 && iw < wd) {
          acc += x[((static_cast<std::int64_t>(ni) * c_in + ic) * h + ih) * wd + iw] *
                 w[((static_cast<std::int64_t>(oc) * c_in + ic) * kh + ki) * kw + kj];
        }
      }
    }
  }
  y[idx] = acc;
}

// dW：每线程一个权重元素，累加所有 (n,oi,oj) 位置。
__global__ void conv2d_dw_kernel(const float* __restrict__ dy, const float* __restrict__ x,
                                 float* __restrict__ dw, int n, int c_in, int h, int wd, int out,
                                 int kh, int kw, int pad) {
  // int64 参与运算：2*pad 在 int 下可能溢出（host 侧已保证 oh/ow 落在 int 范围）。
  const int oh = static_cast<int>(static_cast<std::int64_t>(h) + 2 * static_cast<std::int64_t>(pad) -
                                  kh + 1);
  const int ow = static_cast<int>(static_cast<std::int64_t>(wd) + 2 * static_cast<std::int64_t>(pad) -
                                  kw + 1);
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(out) * c_in * kh * kw;
  if (idx >= total) return;
  const int kj = static_cast<int>(idx % kw);
  const int ki = static_cast<int>((idx / kw) % kh);
  const int ic = static_cast<int>((idx / (static_cast<std::int64_t>(kw) * kh)) % c_in);
  const int oc = static_cast<int>(idx / (static_cast<std::int64_t>(kw) * kh * c_in));
  float acc = 0.0f;
  for (int ni = 0; ni < n; ++ni) {
    for (int oi = 0; oi < oh; ++oi) {
      for (int oj = 0; oj < ow; ++oj) {
        const int ih = oi - pad + ki;
        const int iw = oj - pad + kj;
        if (ih >= 0 && ih < h && iw >= 0 && iw < wd) {
          acc += dy[((static_cast<std::int64_t>(ni) * out + oc) * oh + oi) * ow + oj] *
                 x[((static_cast<std::int64_t>(ni) * c_in + ic) * h + ih) * wd + iw];
        }
      }
    }
  }
  dw[idx] = acc;
}

// db：每线程一个输出通道，累加所有 (n,oi,oj)。
__global__ void conv2d_db_kernel(const float* __restrict__ dy, float* __restrict__ db, int n,
                                 int oh, int ow, int out) {
  const int oc = blockIdx.x * blockDim.x + threadIdx.x;
  if (oc >= out) return;
  float acc = 0.0f;
  for (int ni = 0; ni < n; ++ni) {
    for (int oi = 0; oi < oh; ++oi) {
      for (int oj = 0; oj < ow; ++oj) {
        acc += dy[((static_cast<std::int64_t>(ni) * out + oc) * oh + oi) * ow + oj];
      }
    }
  }
  db[oc] = acc;
}

// dx：每线程一个输入元素，反向收集 (oh = ih+pad-ki, ow = iw+pad-kj) 的贡献（stride=1）。
__global__ void conv2d_dx_kernel(const float* __restrict__ dy, const float* __restrict__ w,
                                 float* __restrict__ dx, int n, int c_in, int h, int wd, int out,
                                 int kh, int kw, int pad) {
  // int64 参与运算：2*pad 在 int 下可能溢出（host 侧已保证 oh/ow 落在 int 范围）。
  const int oh = static_cast<int>(static_cast<std::int64_t>(h) + 2 * static_cast<std::int64_t>(pad) -
                                  kh + 1);
  const int ow = static_cast<int>(static_cast<std::int64_t>(wd) + 2 * static_cast<std::int64_t>(pad) -
                                  kw + 1);
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::int64_t total = static_cast<std::int64_t>(n) * c_in * h * wd;
  if (idx >= total) return;
  const int iw = static_cast<int>(idx % wd);
  const int ih = static_cast<int>((idx / wd) % h);
  const int ic = static_cast<int>((idx / (static_cast<std::int64_t>(wd) * h)) % c_in);
  const int ni = static_cast<int>(idx / (static_cast<std::int64_t>(wd) * h * c_in));
  float acc = 0.0f;
  for (int oc = 0; oc < out; ++oc) {
    for (int ki = 0; ki < kh; ++ki) {
      for (int kj = 0; kj < kw; ++kj) {
        const int oi = ih + pad - ki;
        const int oj = iw + pad - kj;
        if (oi >= 0 && oi < oh && oj >= 0 && oj < ow) {
          acc += dy[((static_cast<std::int64_t>(ni) * out + oc) * oh + oi) * ow + oj] *
                 w[((static_cast<std::int64_t>(oc) * c_in + ic) * kh + ki) * kw + kj];
        }
      }
    }
  }
  dx[idx] = acc;
}

// 校验 Conv2d 参数与 x 形状，计算输出尺寸；元素数一律按 int64 检查上限。
struct ConvDims {
  int n, c_in, h, wd, oh, ow;
};
ConvDims check_dims(const Conv2d& m, const Tensor& x) {
  detail::check_conv_params(m);
  if (x.ndim() != 4 || x.dims()[1] != m.in) {
    throw std::invalid_argument("conv2d_cuda: x 形状必须为 [N, in, H, W]");
  }
  ConvDims d;
  d.n = static_cast<int>(check_elems(x.dims()[0], "x.dims()[0]"));
  d.c_in = m.in;
  d.h = static_cast<int>(check_elems(x.dims()[2], "x.dims()[2]"));
  d.wd = static_cast<int>(check_elems(x.dims()[3], "x.dims()[3]"));
  check_elems(static_cast<std::int64_t>(d.n) * d.c_in * d.h * d.wd, "x 元素数");
  check_elems(static_cast<std::int64_t>(m.out) * m.in * m.kh * m.kw, "W 元素数");
  // pad 用 int64 参与运算：2*pad 在 int 下可能溢出（审核 CPU #10）。
  const std::int64_t oh = static_cast<std::int64_t>(d.h) + 2 * static_cast<std::int64_t>(m.pad) -
                          m.kh + 1;
  const std::int64_t ow = static_cast<std::int64_t>(d.wd) + 2 * static_cast<std::int64_t>(m.pad) -
                          m.kw + 1;
  if (oh <= 0 || ow <= 0) {
    throw std::invalid_argument("conv2d_cuda: 输出尺寸非正（pad 太小）");
  }
  d.oh = static_cast<int>(check_elems(oh, "输出 H'"));
  d.ow = static_cast<int>(check_elems(ow, "输出 W'"));
  check_elems(static_cast<std::int64_t>(d.n) * m.out * d.oh * d.ow, "y/dy 元素数");
  return d;
}

}  // namespace

Tensor conv2d_forward_cuda(const Conv2d& m, const Tensor& x) {
  const ConvDims d = check_dims(m, x);
  const int out = m.out;
  const std::int64_t n_w = static_cast<std::int64_t>(m.out) * m.in * m.kh * m.kw;
  const std::int64_t n_y = static_cast<std::int64_t>(d.n) * out * d.oh * d.ow;
  const std::int64_t n_x = static_cast<std::int64_t>(d.n) * d.c_in * d.h * d.wd;
  if (d.n == 0) {
    // 空 batch：0 个 block 的 launch 会报 invalid configuration，直接返回空张量。
    return Tensor::zeros({d.n, out, d.oh, d.ow});
  }

  DeviceBuf d_x(static_cast<std::size_t>(n_x) * sizeof(float));
  DeviceBuf d_w(static_cast<std::size_t>(n_w) * sizeof(float));
  DeviceBuf d_b(static_cast<std::size_t>(out) * sizeof(float));
  DeviceBuf d_y(static_cast<std::size_t>(n_y) * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_x.p, x.data(), static_cast<std::size_t>(n_x) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.p, m.W.data(), static_cast<std::size_t>(n_w) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b.p, m.b.data(), static_cast<std::size_t>(out) * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int blocks = static_cast<int>((n_y + kThreads - 1) / kThreads);
  conv2d_forward_kernel<<<blocks, kThreads>>>(d_x.p, d_w.p, d_b.p, d_y.p, d.n, d.c_in, d.h, d.wd,
                                              out, m.kh, m.kw, m.pad);
  check_after_kernel("conv2d_forward_kernel");

  Tensor y = Tensor::zeros({d.n, out, d.oh, d.ow});
  CUDA_CHECK(cudaMemcpy(y.data(), d_y.p, static_cast<std::size_t>(n_y) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return y;
}

ConvGrad conv2d_backward_cuda(const Conv2d& m, const Tensor& x, const Tensor& dy) {
  const ConvDims d = check_dims(m, x);
  if (dy.ndim() != 4 || dy.dims()[0] != d.n || dy.dims()[1] != m.out || dy.dims()[2] != d.oh ||
      dy.dims()[3] != d.ow) {
    throw std::invalid_argument("conv2d_backward_cuda: dy 形状必须为 [N, out, H', W']");
  }
  const int out = m.out;
  const std::int64_t n_w = static_cast<std::int64_t>(m.out) * m.in * m.kh * m.kw;
  const std::int64_t n_y = static_cast<std::int64_t>(d.n) * out * d.oh * d.ow;
  const std::int64_t n_x = static_cast<std::int64_t>(d.n) * d.c_in * d.h * d.wd;

  ConvGrad g;
  g.dW = Tensor::zeros({m.out, m.in, m.kh, m.kw});
  g.db = Tensor::zeros({m.out});
  g.dx = Tensor::zeros({d.n, d.c_in, d.h, d.wd});
  if (d.n == 0) return g;  // 空 batch：全零梯度，不 launch kernel（0 block 非法）。

  DeviceBuf d_x(static_cast<std::size_t>(n_x) * sizeof(float));
  DeviceBuf d_w(static_cast<std::size_t>(n_w) * sizeof(float));
  DeviceBuf d_dy(static_cast<std::size_t>(n_y) * sizeof(float));
  DeviceBuf d_dw(static_cast<std::size_t>(n_w) * sizeof(float));
  DeviceBuf d_db(static_cast<std::size_t>(out) * sizeof(float));
  DeviceBuf d_dx(static_cast<std::size_t>(n_x) * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_x.p, x.data(), static_cast<std::size_t>(n_x) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_w.p, m.W.data(), static_cast<std::size_t>(n_w) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_dy.p, dy.data(), static_cast<std::size_t>(n_y) * sizeof(float),
                        cudaMemcpyHostToDevice));

  int blocks = static_cast<int>((n_w + kThreads - 1) / kThreads);
  conv2d_dw_kernel<<<blocks, kThreads>>>(d_dy.p, d_x.p, d_dw.p, d.n, d.c_in, d.h, d.wd, out,
                                         m.kh, m.kw, m.pad);
  check_after_kernel("conv2d_dw_kernel");

  blocks = static_cast<int>((static_cast<std::int64_t>(out) + kThreads - 1) / kThreads);
  conv2d_db_kernel<<<blocks, kThreads>>>(d_dy.p, d_db.p, d.n, d.oh, d.ow, out);
  check_after_kernel("conv2d_db_kernel");

  if (n_x > 0) {  // H==0 或 W==0（pad 足够大时合法）时 dx 元素为 0，跳过 launch。
    blocks = static_cast<int>((n_x + kThreads - 1) / kThreads);
    conv2d_dx_kernel<<<blocks, kThreads>>>(d_dy.p, d_w.p, d_dx.p, d.n, d.c_in, d.h, d.wd, out,
                                           m.kh, m.kw, m.pad);
    check_after_kernel("conv2d_dx_kernel");
  }

  CUDA_CHECK(cudaMemcpy(g.dW.data(), d_dw.p, static_cast<std::size_t>(n_w) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g.db.data(), d_db.p, static_cast<std::size_t>(out) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(g.dx.data(), d_dx.p, static_cast<std::size_t>(n_x) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return g;
}

}  // namespace uavrgb
