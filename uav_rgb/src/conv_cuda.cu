// 单层 Conv2d 的 CUDA 实现：前向 + 反向（dW/db/dx），NCHW、stride=1、pad 可配。
// 只针对 sm_75（Turing）；不引入 cuBLAS/cuDNN，纯手写直接卷积 kernel。
// kernel 经 ops_dev.hpp 的设备指针级函数暴露给常驻路径；本文件末尾的 host Tensor 包装
// 只负责分配/拷贝/同步，供 CPU/CUDA 对拍。
#include "uavrgb/conv_cuda.hpp"
#include "uavrgb/cuda_alloc.hpp"
#include "uavrgb/ops_dev.hpp"

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

// launch 后立刻查配置错误（0 block、grid 超限）；执行期错误留给调用方的同步点。
void launched(const char* name) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel ") + name + ": " + cudaGetErrorString(err));
  }
}

// 极简设备缓冲 RAII：析构自动释放，异常路径也不泄漏。
struct DeviceBuf {
  float* p = nullptr;
  DeviceBuf() = default;
  explicit DeviceBuf(std::size_t bytes) {
    if (cudaMalloc(&p, bytes) != cudaSuccess) throw_alloc_failure(bytes, __FILE__, __LINE__);
  }
  ~DeviceBuf() {
    if (p) cudaFree(p);
  }
  DeviceBuf(DeviceBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
};

constexpr int kThreads = 256;
// dW 每线程只累加一个样本的 kDwRows 行：c1 在 batch 128 下从 432 个线程变成 432×128×8 个。
constexpr int kDwRows = 16;

int blocks_for(std::int64_t total) { return static_cast<int>((total + kThreads - 1) / kThreads); }

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
                                      int c_in, int h, int wd, int out, int kh, int kw, int pad,
                                      int oh, int ow) {
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

// dW：grid = (权重元素块, 样本 n, 行分块)。每线程对一个权重元素累加本样本 kDwRows 行内的
// 贡献，写进 part[(n*行块 + 行块号) * n_w + idx]，再由 reduce_rows_kernel 按固定顺序求和。
// 不用 atomicAdd：float 原子加的顺序随机，会让同 seed 训练不可逐位复现（test_soft_targets_train
// 的阈值就是按确定性轨迹定的）。旧实现每线程串行扫完 N×H'×W'，c1 只有 432 个线程（2 个
// block），一张几十个 SM 的卡上等于单 SM 串行，是改前一步 1.9 s 里最大的 kernel 开销（314 ms）。
__global__ void conv2d_dw_kernel(const float* __restrict__ dy, const float* __restrict__ x,
                                 float* __restrict__ part, int c_in, int h, int wd, int out,
                                 int kh, int kw, int pad, int oh, int ow) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= out * c_in * kh * kw) return;
  const int kj = idx % kw;
  const int ki = (idx / kw) % kh;
  const int ic = (idx / (kw * kh)) % c_in;
  const int oc = idx / (kw * kh * c_in);
  const int ni = blockIdx.y;
  const int oi_end = min(static_cast<int>(blockIdx.z + 1) * kDwRows, oh);
  const float* dyp = dy + (static_cast<std::int64_t>(ni) * out + oc) * oh * ow;
  const float* xp = x + (static_cast<std::int64_t>(ni) * c_in + ic) * h * wd;
  float acc = 0.0f;
  for (int oi = blockIdx.z * kDwRows; oi < oi_end; ++oi) {
    const int ih = oi - pad + ki;
    if (ih < 0 || ih >= h) continue;
    for (int oj = 0; oj < ow; ++oj) {
      const int iw = oj - pad + kj;
      if (iw >= 0 && iw < wd) acc += dyp[oi * ow + oj] * xp[ih * wd + iw];
    }
  }
  part[(static_cast<std::size_t>(ni) * gridDim.z + blockIdx.z) * (out * c_in * kh * kw) + idx] = acc;
}

// db：grid = (out, n)，一个 block 归约一个 (n, oc) 平面，写 part[n*out + oc]，再按 n 顺序求和。
// 旧实现每输出通道一个线程串行扫 N×H'×W'（c1 共 16 个线程，每线程 1.23M 次累加，110 ms）。
__global__ void conv2d_db_kernel(const float* __restrict__ dy, float* __restrict__ part, int out,
                                 int plane) {
  const float* p = dy + (static_cast<std::int64_t>(blockIdx.y) * out + blockIdx.x) * plane;
  float acc = 0.0f;
  for (int i = threadIdx.x; i < plane; i += blockDim.x) acc += p[i];
  __shared__ float sh[kThreads];
  sh[threadIdx.x] = acc;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) part[blockIdx.y * out + blockIdx.x] = sh[0];
}

// out[j] = Σ_i part[i*cols + j]，i 升序：结果只取决于形状，不取决于调度。
__global__ void reduce_rows_kernel(const float* __restrict__ part, float* __restrict__ out,
                                   int rows, int cols) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= cols) return;
  float acc = 0.0f;
  for (int i = 0; i < rows; ++i) acc += part[static_cast<std::size_t>(i) * cols + j];
  out[j] = acc;
}

// dx：每线程一个输入元素，反向收集 (oh = ih+pad-ki, ow = iw+pad-kj) 的贡献（stride=1）。
__global__ void conv2d_dx_kernel(const float* __restrict__ dy, const float* __restrict__ w,
                                 float* __restrict__ dx, int n, int c_in, int h, int wd, int out,
                                 int kh, int kw, int pad, int oh, int ow) {
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
dev::ConvShape check_dims(const Conv2d& m, const Tensor& x) {
  detail::check_conv_params(m);
  if (x.ndim() != 4 || x.dims()[1] != m.in) {
    throw std::invalid_argument("conv2d_cuda: x 形状必须为 [N, in, H, W]");
  }
  dev::ConvShape d;
  d.n = static_cast<int>(check_elems(x.dims()[0], "x.dims()[0]"));
  d.c_in = m.in;
  d.h = static_cast<int>(check_elems(x.dims()[2], "x.dims()[2]"));
  d.w = static_cast<int>(check_elems(x.dims()[3], "x.dims()[3]"));
  d.out = m.out;
  d.kh = m.kh;
  d.kw = m.kw;
  d.pad = m.pad;
  check_elems(static_cast<std::int64_t>(d.n) * d.c_in * d.h * d.w, "x 元素数");
  check_elems(static_cast<std::int64_t>(m.out) * m.in * m.kh * m.kw, "W 元素数");
  // pad 用 int64 参与运算：2*pad 在 int 下可能溢出（审核 CPU #10）。
  const std::int64_t oh = static_cast<std::int64_t>(d.h) + 2 * static_cast<std::int64_t>(m.pad) -
                          m.kh + 1;
  const std::int64_t ow = static_cast<std::int64_t>(d.w) + 2 * static_cast<std::int64_t>(m.pad) -
                          m.kw + 1;
  if (oh <= 0 || ow <= 0) {
    throw std::invalid_argument("conv2d_cuda: 输出尺寸非正（pad 太小）");
  }
  d.oh = static_cast<int>(check_elems(oh, "输出 H'"));
  d.ow = static_cast<int>(check_elems(ow, "输出 W'"));
  check_elems(static_cast<std::int64_t>(d.n) * m.out * d.oh * d.ow, "y/dy 元素数");
  return d;
}

DeviceBuf upload(const float* src, std::int64_t count) {
  DeviceBuf d(static_cast<std::size_t>(count) * sizeof(float));
  CUDA_CHECK(cudaMemcpy(d.p, src, static_cast<std::size_t>(count) * sizeof(float),
                        cudaMemcpyHostToDevice));
  return d;
}

void download(Tensor& dst, const float* src) {
  CUDA_CHECK(cudaMemcpy(dst.data(), src, dst.numel() * sizeof(float), cudaMemcpyDeviceToHost));
}

}  // namespace

namespace dev {

void conv2d_forward(const ConvShape& s, const float* x, const float* w, const float* b, float* y) {
  const std::int64_t n_y = static_cast<std::int64_t>(s.n) * s.out * s.oh * s.ow;
  if (n_y == 0) return;
  conv2d_forward_kernel<<<blocks_for(n_y), kThreads>>>(x, w, b, y, s.n, s.c_in, s.h, s.w, s.out,
                                                      s.kh, s.kw, s.pad, s.oh, s.ow);
  launched("conv2d_forward_kernel");
}

std::size_t conv2d_backward_workspace(const ConvShape& s) {
  const std::size_t chunks = (s.oh + kDwRows - 1) / kDwRows;
  return static_cast<std::size_t>(s.n) * chunks * s.out * s.c_in * s.kh * s.kw +
         static_cast<std::size_t>(s.n) * s.out;
}

void conv2d_backward(const ConvShape& s, const float* x, const float* w, const float* dy,
                     float* dw, float* db, float* dx, float* ws) {
  if (s.n == 0) return;
  if (s.n > 65535) throw std::invalid_argument("conv2d_cuda: batch > 65535 超出 grid.y 上限");
  const int n_w = s.out * s.c_in * s.kh * s.kw;
  const int chunks = (s.oh + kDwRows - 1) / kDwRows;
  float* part_w = ws;
  float* part_b = ws + static_cast<std::size_t>(s.n) * chunks * n_w;
  conv2d_dw_kernel<<<dim3(blocks_for(n_w), s.n, chunks), kThreads>>>(
      dy, x, part_w, s.c_in, s.h, s.w, s.out, s.kh, s.kw, s.pad, s.oh, s.ow);
  launched("conv2d_dw_kernel");
  reduce_rows_kernel<<<blocks_for(n_w), kThreads>>>(part_w, dw, s.n * chunks, n_w);
  launched("reduce_rows_kernel(dw)");
  conv2d_db_kernel<<<dim3(s.out, s.n), kThreads>>>(dy, part_b, s.out, s.oh * s.ow);
  launched("conv2d_db_kernel");
  reduce_rows_kernel<<<blocks_for(s.out), kThreads>>>(part_b, db, s.n, s.out);
  launched("reduce_rows_kernel(db)");
  const std::int64_t n_x = static_cast<std::int64_t>(s.n) * s.c_in * s.h * s.w;
  if (dx != nullptr && n_x > 0) {  // H==0 或 W==0（pad 足够大时合法）时 dx 元素为 0，跳过 launch。
    conv2d_dx_kernel<<<blocks_for(n_x), kThreads>>>(dy, w, dx, s.n, s.c_in, s.h, s.w, s.out, s.kh,
                                                   s.kw, s.pad, s.oh, s.ow);
    launched("conv2d_dx_kernel");
  }
}

}  // namespace dev

Tensor conv2d_forward_cuda(const Conv2d& m, const Tensor& x) {
  const dev::ConvShape d = check_dims(m, x);
  Tensor y = Tensor::zeros({d.n, m.out, d.oh, d.ow});
  if (d.n == 0) return y;  // 空 batch：0 个 block 的 launch 会报 invalid configuration。
  const DeviceBuf d_x = upload(x.data(), static_cast<std::int64_t>(x.numel()));
  const DeviceBuf d_w = upload(m.W.data(), static_cast<std::int64_t>(m.W.numel()));
  const DeviceBuf d_b = upload(m.b.data(), m.out);
  DeviceBuf d_y(y.numel() * sizeof(float));
  dev::conv2d_forward(d, d_x.p, d_w.p, d_b.p, d_y.p);
  CUDA_CHECK(cudaDeviceSynchronize());
  download(y, d_y.p);
  return y;
}

ConvGrad conv2d_backward_cuda(const Conv2d& m, const Tensor& x, const Tensor& dy, bool need_dx) {
  const dev::ConvShape d = check_dims(m, x);
  if (dy.ndim() != 4 || dy.dims()[0] != d.n || dy.dims()[1] != m.out || dy.dims()[2] != d.oh ||
      dy.dims()[3] != d.ow) {
    throw std::invalid_argument("conv2d_backward_cuda: dy 形状必须为 [N, out, H', W']");
  }
  ConvGrad g;
  g.dW = Tensor::zeros({m.out, m.in, m.kh, m.kw});
  g.db = Tensor::zeros({m.out});
  g.dx = Tensor::zeros({d.n, d.c_in, d.h, d.w});
  if (d.n == 0) return g;  // 空 batch：全零梯度，不 launch kernel（0 block 非法）。
  const DeviceBuf d_x = upload(x.data(), static_cast<std::int64_t>(x.numel()));
  const DeviceBuf d_w = upload(m.W.data(), static_cast<std::int64_t>(m.W.numel()));
  const DeviceBuf d_dy = upload(dy.data(), static_cast<std::int64_t>(dy.numel()));
  DeviceBuf d_dw(g.dW.numel() * sizeof(float));
  DeviceBuf d_db(g.db.numel() * sizeof(float));
  DeviceBuf d_dx(need_dx ? g.dx.numel() * sizeof(float) : sizeof(float));
  DeviceBuf d_ws(dev::conv2d_backward_workspace(d) * sizeof(float));
  dev::conv2d_backward(d, d_x.p, d_w.p, d_dy.p, d_dw.p, d_db.p, need_dx ? d_dx.p : nullptr,
                       d_ws.p);
  CUDA_CHECK(cudaDeviceSynchronize());
  download(g.dW, d_dw.p);
  download(g.db, d_db.p);
  if (need_dx) download(g.dx, d_dx.p);
  return g;
}

}  // namespace uavrgb
