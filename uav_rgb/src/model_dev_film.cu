// DeviceCnn 的 FiLM 条件分支：条件向量 cond = [文本 embedding 查表 | prev-action one-hot]、
// film = Linear(cond)、第三层特征的 (1+γ)·y+β 调制及其反向。从 model_dev.cu 拆出以守住行数预算
// （STUDENT-P2 加了 prev-action 段）；kernel 口径与 model.cpp 的 host 路径逐位一致，
// 由 test_device_train_step / test_text_branch 对拍。
#include "cuda_check.hpp"
#include "uavrgb/model_dev.hpp"
#include "uavrgb/ops_dev.hpp"

#include <cuda_runtime.h>

namespace uavrgb {
namespace {

// FiLM 条件向量 cond[n, cdim] = [文本特征 tfeat | prev one-hot]（与 model.cpp 一致）：
// 文本段 <pad>(0) 与越界 id 的槽位置 0；prev 段只有 prev[s] 那一位是 1。
__global__ void cond_gather_kernel(const std::uint8_t* __restrict__ ids,
                                   const std::uint8_t* __restrict__ prev,
                                   const float* __restrict__ emb, float* __restrict__ cond,
                                   int text_dim, int tfeat, int cdim, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int s = i / cdim, j = i % cdim;
  if (j >= tfeat) {
    cond[i] = (prev[s] == j - tfeat) ? 1.0f : 0.0f;
    return;
  }
  const int id = ids[s * kTextLen + j / text_dim];
  cond[i] = (id == 0 || id >= kTextVocab) ? 0.0f : emb[id * text_dim + j % text_dim];
}

// demb[id, d] = Σ_{t: ids[t]==id} dcond[t 所在样本, t 的槽位, d]：每线程一个 (id, d)，按
// token 顺序扫一遍（原子加版本求和顺序随机，同 seed 训练不可逐位复现）。<pad>=0 的行恒为 0。
__global__ void text_demb_kernel(const std::uint8_t* __restrict__ ids,
                                 const float* __restrict__ dcond, float* __restrict__ demb,
                                 int tokens, int text_dim, int cdim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kTextVocab * text_dim) return;
  const int id = i / text_dim, d = i % text_dim;
  float acc = 0.0f;
  for (int t = 0; id != 0 && t < tokens; ++t) {
    if (ids[t] == id) acc += dcond[(t / kTextLen) * cdim + (t % kTextLen) * text_dim + d];
  }
  demb[i] = acc;
}

// y' = (1+γ)·y + β。用 __fmul_rn/__fadd_rn 禁止 nvcc 合并成 FMA，保持与 CPU 路径逐位一致。
__global__ void film_forward_kernel(const float* __restrict__ y, const float* __restrict__ film,
                                    float* __restrict__ out, int channels, int hw, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int nc = i / hw;
  const int s = nc / channels, c = nc % channels;
  const float gamma = film[s * 2 * channels + c];
  const float beta = film[s * 2 * channels + channels + c];
  out[i] = __fadd_rn(__fmul_rn(1.0f + gamma, y[i]), beta);
}

// 每 block 一个 (sample, c)：dγ = Σ dy'·y、dβ = Σ dy'（double 累加，同 CPU 路径），
// 然后把 dy' 原地改写成 dy = dy'·(1+γ)。
__global__ void film_backward_kernel(float* __restrict__ d, const float* __restrict__ y,
                                     const float* __restrict__ film, float* __restrict__ dfilm,
                                     int channels, int hw) {
  __shared__ double sg[kThreads];
  __shared__ double sb[kThreads];
  const int nc = blockIdx.x;
  const int s = nc / channels, c = nc % channels;
  const float gamma = film[s * 2 * channels + c];
  float* dp = d + static_cast<std::size_t>(nc) * hw;
  const float* yp = y + static_cast<std::size_t>(nc) * hw;
  double g = 0.0, b = 0.0;
  for (int k = threadIdx.x; k < hw; k += blockDim.x) {
    g += static_cast<double>(dp[k]) * yp[k];
    b += dp[k];
  }
  sg[threadIdx.x] = g;
  sb[threadIdx.x] = b;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      sg[threadIdx.x] += sg[threadIdx.x + stride];
      sb[threadIdx.x] += sb[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    dfilm[s * 2 * channels + c] = static_cast<float>(sg[0]);
    dfilm[s * 2 * channels + channels + c] = static_cast<float>(sb[0]);
  }
  for (int k = threadIdx.x; k < hw; k += blockDim.x) dp[k] = dp[k] * (1.0f + gamma);
}

}  // namespace

void DeviceCnn::film_forward(int n) {
  const LayerDims& L = layer_[2];
  const int tfeat = kTextLen * text_dim_;
  const int total = n * cond_dim_;
  cond_gather_kernel<<<blocks_for(total), kThreads>>>(
      text_u8_, prev_u8_, text_dim_ > 0 ? params_ + seg_[seg_emb_].off : nullptr, cond_,
      text_dim_, tfeat, cond_dim_, total);
  launched("cond_gather_kernel");
  dev::linear_forward(n, cond_dim_, 2 * L.c_out, cond_, params_ + seg_[seg_film_].off,
                      params_ + seg_[seg_film_ + 1].off, film_out_);
  const int hw = L.ph * L.pw;
  const int tot = n * L.c_out * hw;
  film_forward_kernel<<<blocks_for(tot), kThreads>>>(p3_, film_out_, p3mod_, L.c_out, hw, tot);
  launched("film_forward_kernel");
}

void DeviceCnn::film_backward(int n) {
  const LayerDims& L = layer_[2];
  film_backward_kernel<<<n * L.c_out, kThreads>>>(dflat_, p3_, film_out_, dfilm_, L.c_out,
                                                  L.ph * L.pw);
  launched("film_backward_kernel");
  dev::linear_backward(n, cond_dim_, 2 * L.c_out, cond_, params_ + seg_[seg_film_].off, dfilm_,
                       grads_ + seg_[seg_film_].off, grads_ + seg_[seg_film_ + 1].off, dcond_);
  if (text_dim_ == 0) return;  // prev one-hot 没有参数，dcond_ 的 prev 段到此为止
  text_demb_kernel<<<blocks_for(seg_[seg_emb_].len), kThreads>>>(
      text_u8_, dcond_, grads_ + seg_[seg_emb_].off, n * kTextLen, text_dim_, cond_dim_);
  launched("text_demb_kernel");
}

}  // namespace uavrgb
