// 逐格 softmax 交叉熵（深度分带辅助头）的 CUDA 实现：设备指针版 dev::spatial_ce 给常驻训练
// 路径在显存里直接算 loss/grad/分带命中数；host Tensor 版只是它的上传/下载包装，供与
// ops_spatial.cpp 的 CPU 参考对拍。口径（屏蔽码 255、按有效格数归一化）以 ops.hpp 为准。
#include "cuda_check.hpp"
#include "uavrgb/cuda_alloc.hpp"
#include "uavrgb/ops.hpp"
#include "uavrgb/ops_dev.hpp"
#include "uavrgb/softmax_cuda.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace uavrgb {
namespace {

constexpr int kThreads = 256;

// 每线程一格 (sample, p)：stable softmax 后写 K 个梯度（屏蔽格写 0）、double 原子累加 CE、
// 统计 argmax 命中。K 很小（8），一格一线程比一元素一线程少算 K 倍 softmax。
__global__ void spatial_ce_kernel(const float* __restrict__ logits,
                                  const std::uint8_t* __restrict__ tgt, int k, int hw, int cells,
                                  float scale, float* __restrict__ grad, double* __restrict__ loss,
                                  unsigned long long* __restrict__ correct) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= cells) return;
  const int s = i / hw, p = i % hw;
  const float* cell = logits + static_cast<std::size_t>(s) * k * hw + p;
  float* gcell = grad ? grad + static_cast<std::size_t>(s) * k * hw + p : nullptr;
  const int t = tgt[i];
  if (t == kSpatialIgnore) {
    for (int c = 0; gcell && c < k; ++c) gcell[c * hw] = 0.0f;
    return;
  }
  float mx = cell[0];
  int arg = 0;
  for (int c = 1; c < k; ++c) {
    const float v = cell[c * hw];
    if (v > mx) mx = v, arg = c;
  }
  float sum = 0.0f;
  for (int c = 0; c < k; ++c) sum += expf(cell[c * hw] - mx);
  for (int c = 0; gcell && c < k; ++c) {
    gcell[c * hw] = (expf(cell[c * hw] - mx) / sum - (c == t ? 1.0f : 0.0f)) * scale;
  }
  if (loss) atomicAdd(loss, -static_cast<double>(cell[t * hw] - mx) + log(static_cast<double>(sum)));
  if (correct && arg == t) atomicAdd(correct, 1ull);
}

__global__ void add_inplace_kernel(float* __restrict__ y, const float* __restrict__ x, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) y[i] += x[i];
}

template <typename T>
struct DeviceBuf {
  T* p = nullptr;
  explicit DeviceBuf(std::size_t count) {
    const std::size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
    if (cudaMalloc(&p, bytes) != cudaSuccess) throw_alloc_failure(bytes, __FILE__, __LINE__);
  }
  ~DeviceBuf() {
    if (p) cudaFree(p);
  }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
};

struct Dims {
  int n, k, hw;
  std::size_t valid;
};

// 与 ops_spatial.cpp 的 check_spatial 同一套校验（这里只是包装层，CPU 参考是口径来源）。
Dims check(const Tensor& logits, const std::vector<std::uint8_t>& targets) {
  if (logits.ndim() != 4) throw std::invalid_argument("spatial_ce_cuda: logits 必须为 [N,K,H,W]");
  const std::int64_t n = logits.dims()[0], k = logits.dims()[1];
  const std::int64_t hw = logits.dims()[2] * logits.dims()[3];
  if (n == 0 || k == 0 || hw == 0 || k >= kSpatialIgnore) {
    throw std::invalid_argument("spatial_ce_cuda: N/K/H/W 须为正且 K < 255");
  }
  if (static_cast<std::int64_t>(targets.size()) != n * hw || n * k * hw > 0x7fffffff) {
    throw std::invalid_argument("spatial_ce_cuda: targets 数量应为 N*H*W（且总元素数在 int 内）");
  }
  std::size_t valid = 0;
  for (std::uint8_t t : targets) {
    if (t == kSpatialIgnore) continue;
    if (t >= k) throw std::invalid_argument("spatial_ce_cuda: target 越界");
    ++valid;
  }
  return Dims{static_cast<int>(n), static_cast<int>(k), static_cast<int>(hw), valid};
}

}  // namespace

namespace dev {

void spatial_ce(const float* logits, const std::uint8_t* targets, int n, int k, int hw,
                float scale, float* grad, double* loss, unsigned long long* correct) {
  const int cells = n * hw;
  if (cells == 0) return;
  spatial_ce_kernel<<<(cells + kThreads - 1) / kThreads, kThreads>>>(logits, targets, k, hw, cells,
                                                                   scale, grad, loss, correct);
  launched("spatial_ce_kernel");
}

void add_inplace(float* y, const float* x, int total) {
  if (total <= 0) return;
  add_inplace_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(y, x, total);
  launched("add_inplace_kernel");
}

}  // namespace dev

float spatial_ce_loss_cuda(const Tensor& logits, const std::vector<std::uint8_t>& targets) {
  const Dims d = check(logits, targets);
  if (d.valid == 0) return 0.0f;
  DeviceBuf<float> x(logits.numel());
  DeviceBuf<std::uint8_t> t(targets.size());
  DeviceBuf<double> loss(1);
  CUDA_CHECK(cudaMemcpy(x.p, logits.data(), logits.numel() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(t.p, targets.data(), targets.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(loss.p, 0, sizeof(double)));
  dev::spatial_ce(x.p, t.p, d.n, d.k, d.hw, 0.0f, nullptr, loss.p, nullptr);
  double total = 0.0;
  CUDA_CHECK(cudaMemcpy(&total, loss.p, sizeof(double), cudaMemcpyDeviceToHost));
  return static_cast<float>(total / static_cast<double>(d.valid));
}

Tensor spatial_ce_grad_cuda(const Tensor& logits, const std::vector<std::uint8_t>& targets) {
  const Dims d = check(logits, targets);
  Tensor g = Tensor::zeros(logits.dims());
  if (d.valid == 0) return g;
  DeviceBuf<float> x(logits.numel());
  DeviceBuf<std::uint8_t> t(targets.size());
  DeviceBuf<float> dg(logits.numel());
  CUDA_CHECK(cudaMemcpy(x.p, logits.data(), logits.numel() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(t.p, targets.data(), targets.size(), cudaMemcpyHostToDevice));
  dev::spatial_ce(x.p, t.p, d.n, d.k, d.hw, 1.0f / static_cast<float>(d.valid), dg.p, nullptr,
                  nullptr);
  CUDA_CHECK(cudaMemcpy(g.data(), dg.p, g.numel() * sizeof(float), cudaMemcpyDeviceToHost));
  return g;
}

}  // namespace uavrgb
