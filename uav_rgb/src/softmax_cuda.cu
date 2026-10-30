// softmax 交叉熵 loss/grad 的 CUDA 实现：行内 stable softmax + double atomicAdd 归约。
// 只针对 sm_75（Turing）；纯手写 kernel，供 CPU/CUDA 对拍（doc 04 §6 N3）。
#include "uavrgb/softmax_cuda.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

// 极简设备缓冲 RAII：析构自动释放，异常路径也不泄漏。模板版用于 labels 的 int 缓冲。
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

// 元素数一律先用 int64 计算并校验上限：超过 INT_MAX 直接抛异常，绝不静默回绕。
std::int64_t check_elems(std::int64_t total, const char* what) {
  if (total > std::numeric_limits<int>::max()) {
    throw std::invalid_argument(std::string("softmax_cuda: shape too large: ") + what + "=" +
                                std::to_string(total) + " > INT_MAX");
  }
  return total;
}

// loss：每线程一行，stable softmax 后 double atomicAdd 到单个累加器。
__global__ void softmax_ce_loss_kernel(const float* __restrict__ logits,
                                       const int* __restrict__ labels, double* __restrict__ out,
                                       int n, int c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* row = logits + static_cast<std::int64_t>(i) * c;
  float mx = row[0];
  for (int j = 1; j < c; ++j) mx = fmaxf(mx, row[j]);
  float sum = 0.0f;
  for (int j = 0; j < c; ++j) sum += expf(row[j] - mx);
  const int label = labels[i];
  // -log(softmax[label]) = -(logit[label]-mx) + log(Σ exp)。
  atomicAdd(out, -static_cast<double>(row[label] - mx) + log(static_cast<double>(sum)));
}

// grad：每线程一个元素，重算本行 max/sum 后写 (p - onehot[label])/n。
__global__ void softmax_ce_grad_kernel(const float* __restrict__ logits,
                                       const int* __restrict__ labels, float* __restrict__ g,
                                       int n, int c) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(n) * c) return;
  const int j = static_cast<int>(idx % c);
  const int i = static_cast<int>(idx / c);
  const float* row = logits + static_cast<std::int64_t>(i) * c;
  float mx = row[0];
  for (int k = 1; k < c; ++k) mx = fmaxf(mx, row[k]);
  float sum = 0.0f;
  for (int k = 0; k < c; ++k) sum += expf(row[k] - mx);
  const float p = expf(row[j] - mx) / sum;
  g[idx] = (p - (j == labels[i] ? 1.0f : 0.0f)) / static_cast<float>(n);
}

// 加权 loss：每线程一行，atomicAdd(w[label] * ce)。
__global__ void softmax_ce_loss_weighted_kernel(const float* __restrict__ logits,
                                                const int* __restrict__ labels,
                                                const float* __restrict__ weights,
                                                double* __restrict__ out, int n, int c) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* row = logits + static_cast<std::int64_t>(i) * c;
  float mx = row[0];
  for (int j = 1; j < c; ++j) mx = fmaxf(mx, row[j]);
  float sum = 0.0f;
  for (int j = 0; j < c; ++j) sum += expf(row[j] - mx);
  const int label = labels[i];
  const double ce = -static_cast<double>(row[label] - mx) + log(static_cast<double>(sum));
  atomicAdd(out, static_cast<double>(weights[label]) * ce);
}

// 加权 grad：每线程一个元素，w[label]·(p - onehot)/wsum。
__global__ void softmax_ce_grad_weighted_kernel(const float* __restrict__ logits,
                                                const int* __restrict__ labels,
                                                const float* __restrict__ weights, float wsum,
                                                float* __restrict__ g, int n, int c) {
  const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= static_cast<std::int64_t>(n) * c) return;
  const int j = static_cast<int>(idx % c);
  const int i = static_cast<int>(idx / c);
  const float* row = logits + static_cast<std::int64_t>(i) * c;
  float mx = row[0];
  for (int k = 1; k < c; ++k) mx = fmaxf(mx, row[k]);
  float sum = 0.0f;
  for (int k = 0; k < c; ++k) sum += expf(row[k] - mx);
  const float p = expf(row[j] - mx) / sum;
  g[idx] = weights[labels[i]] * (p - (j == labels[i] ? 1.0f : 0.0f)) / wsum;
}

// 校验 weights 并返回 Σ_i w[label_i]（double）。labels 已由 check_dims 校验过范围。
double check_weight_sum(const std::vector<int>& labels, const std::vector<float>& weights,
                        int c) {
  if (static_cast<std::int64_t>(weights.size()) != c) {
    throw std::invalid_argument("softmax_cuda: weights 长度应等于 C=" + std::to_string(c) +
                                "（实际 " + std::to_string(weights.size()) + "）");
  }
  for (float w : weights) {
    if (!(w >= 0.0f) || !std::isfinite(w)) {
      throw std::invalid_argument("softmax_cuda: weights 必须为非负有限值");
    }
  }
  double sum = 0.0;
  for (int label : labels) sum += static_cast<double>(weights[static_cast<std::size_t>(label)]);
  if (!(sum > 0.0)) {
    throw std::invalid_argument("softmax_cuda: weights 按样本求和为 0，无法归一化");
  }
  return sum;
}

// 校验形状与 label 范围，返回 (n, c)。空 batch / C=0 是调用方错误，与 CPU 参考一致抛异常。
std::pair<int, int> check_dims(const Tensor& logits, const std::vector<int>& labels) {
  if (logits.ndim() != 2) {
    throw std::invalid_argument("softmax_cuda: logits 形状必须为 [N, C]");
  }
  const std::int64_t n64 = logits.dims().at(0);
  const std::int64_t c64 = logits.dims().at(1);
  if (n64 == 0) {
    throw std::invalid_argument("softmax_cuda: 空 batch（N=0）");
  }
  if (c64 == 0) {
    throw std::invalid_argument("softmax_cuda: C 必须为正（实际 logits 为 [N,0]）");
  }
  check_elems(n64 * c64, "logits 元素数");
  if (static_cast<std::size_t>(n64) != labels.size()) {
    throw std::invalid_argument("labels 数量与 logits 行数不一致");
  }
  for (int label : labels) {
    if (label < 0 || label >= c64) throw std::invalid_argument("label 越界");
  }
  return {static_cast<int>(n64), static_cast<int>(c64)};
}

}  // namespace

float softmax_ce_loss_cuda(const Tensor& logits, const std::vector<int>& labels) {
  const auto [n, c] = check_dims(logits, labels);
  const std::int64_t n_c = static_cast<std::int64_t>(n) * c;

  DeviceBuf<float> d_logits(static_cast<std::size_t>(n_c) * sizeof(float));
  DeviceBuf<int> d_labels(static_cast<std::size_t>(n) * sizeof(int));
  DeviceBuf<double> d_out(sizeof(double));

  CUDA_CHECK(cudaMemcpy(d_logits.p, logits.data(), static_cast<std::size_t>(n_c) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels.p, labels.data(), static_cast<std::size_t>(n) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.p, 0, sizeof(double)));

  const int blocks = static_cast<int>((static_cast<std::int64_t>(n) + kThreads - 1) / kThreads);
  softmax_ce_loss_kernel<<<blocks, kThreads>>>(d_logits.p, d_labels.p, d_out.p, n, c);
  check_after_kernel("softmax_ce_loss_kernel");

  double total = 0.0;
  CUDA_CHECK(cudaMemcpy(&total, d_out.p, sizeof(double), cudaMemcpyDeviceToHost));
  return static_cast<float>(total / n);
}

Tensor softmax_ce_grad_cuda(const Tensor& logits, const std::vector<int>& labels) {
  const auto [n, c] = check_dims(logits, labels);
  const std::int64_t total = static_cast<std::int64_t>(n) * c;

  DeviceBuf<float> d_logits(static_cast<std::size_t>(total) * sizeof(float));
  DeviceBuf<int> d_labels(static_cast<std::size_t>(n) * sizeof(int));
  DeviceBuf<float> d_g(static_cast<std::size_t>(total) * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_logits.p, logits.data(), static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels.p, labels.data(), static_cast<std::size_t>(n) * sizeof(int),
                        cudaMemcpyHostToDevice));

  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  softmax_ce_grad_kernel<<<blocks, kThreads>>>(d_logits.p, d_labels.p, d_g.p, n, c);
  check_after_kernel("softmax_ce_grad_kernel");

  Tensor g = Tensor::zeros({n, c});
  CUDA_CHECK(cudaMemcpy(g.data(), d_g.p, static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return g;
}

float softmax_ce_loss_weighted_cuda(const Tensor& logits, const std::vector<int>& labels,
                                    const std::vector<float>& weights) {
  const auto [n, c] = check_dims(logits, labels);
  const double wsum = check_weight_sum(labels, weights, c);
  const std::int64_t n_c = static_cast<std::int64_t>(n) * c;

  DeviceBuf<float> d_logits(static_cast<std::size_t>(n_c) * sizeof(float));
  DeviceBuf<int> d_labels(static_cast<std::size_t>(n) * sizeof(int));
  DeviceBuf<float> d_weights(static_cast<std::size_t>(c) * sizeof(float));
  DeviceBuf<double> d_out(sizeof(double));

  CUDA_CHECK(cudaMemcpy(d_logits.p, logits.data(), static_cast<std::size_t>(n_c) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels.p, labels.data(), static_cast<std::size_t>(n) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights.p, weights.data(), static_cast<std::size_t>(c) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out.p, 0, sizeof(double)));

  const int blocks = static_cast<int>((static_cast<std::int64_t>(n) + kThreads - 1) / kThreads);
  softmax_ce_loss_weighted_kernel<<<blocks, kThreads>>>(d_logits.p, d_labels.p, d_weights.p,
                                                        d_out.p, n, c);
  check_after_kernel("softmax_ce_loss_weighted_kernel");

  double total = 0.0;
  CUDA_CHECK(cudaMemcpy(&total, d_out.p, sizeof(double), cudaMemcpyDeviceToHost));
  return static_cast<float>(total / wsum);
}

Tensor softmax_ce_grad_weighted_cuda(const Tensor& logits, const std::vector<int>& labels,
                                     const std::vector<float>& weights) {
  const auto [n, c] = check_dims(logits, labels);
  const double wsum_d = check_weight_sum(labels, weights, c);
  // kernel 参数是 float：Σw 超出 float 会变成 inf，梯度静默全 0。
  if (!(wsum_d <= static_cast<double>(std::numeric_limits<float>::max()))) {
    throw std::invalid_argument("softmax_cuda: 权重按样本求和超出 float 可表示范围，梯度会全为 0");
  }
  const float wsum = static_cast<float>(wsum_d);
  const std::int64_t total = static_cast<std::int64_t>(n) * c;

  DeviceBuf<float> d_logits(static_cast<std::size_t>(total) * sizeof(float));
  DeviceBuf<int> d_labels(static_cast<std::size_t>(n) * sizeof(int));
  DeviceBuf<float> d_weights(static_cast<std::size_t>(c) * sizeof(float));
  DeviceBuf<float> d_g(static_cast<std::size_t>(total) * sizeof(float));

  CUDA_CHECK(cudaMemcpy(d_logits.p, logits.data(), static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_labels.p, labels.data(), static_cast<std::size_t>(n) * sizeof(int),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weights.p, weights.data(), static_cast<std::size_t>(c) * sizeof(float),
                        cudaMemcpyHostToDevice));

  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  softmax_ce_grad_weighted_kernel<<<blocks, kThreads>>>(d_logits.p, d_labels.p, d_weights.p,
                                                        wsum, d_g.p, n, c);
  check_after_kernel("softmax_ce_grad_weighted_kernel");

  Tensor g = Tensor::zeros({n, c});
  CUDA_CHECK(cudaMemcpy(g.data(), d_g.p, static_cast<std::size_t>(total) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return g;
}

}  // namespace uavrgb
