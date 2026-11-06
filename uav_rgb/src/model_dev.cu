// 设备常驻 CNN（见 model_dev.hpp）。只有 uint8→float、文本 embedding 查表/散射、FiLM 调制
// 这几个 kernel 在本文件；卷积/池化/线性/Adam/范数全部走 ops_dev.hpp，与 host 路径同一份代码。
// 全部在默认流上；forward 末尾的 logits D2H 和 backward_step 里的范数 D2H 是仅有的两个同步点。
#include "uavrgb/model_dev.hpp"

#include "cuda_check.hpp"
#include "uavrgb/cuda_alloc.hpp"
#include "uavrgb/ops_dev.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace uavrgb {
namespace {

constexpr int kThreads = 256;
constexpr int kPoolK = 2;  // 与 model.cpp 的 kPool(2, 2) 一致
int blocks_for(std::size_t total) { return static_cast<int>((total + kThreads - 1) / kThreads); }

dev::ConvShape conv_shape(const DeviceCnn::LayerDims& L, int n) {
  return dev::ConvShape{n, L.c_in, L.h, L.w, L.c_out, L.kh, L.kw, L.pad, L.oh, L.ow};
}
dev::PoolShape pool_shape(const DeviceCnn::LayerDims& L, int n) {
  return dev::PoolShape{n, L.c_out, L.oh, L.ow, kPoolK, kPoolK, L.ph, L.pw};
}

// text_feat[n, kTextLen*text_dim]：<pad>(0) 与越界 id 的槽位置 0（与 model.cpp 一致）。
__global__ void text_gather_kernel(const std::uint8_t* __restrict__ ids,
                                   const float* __restrict__ emb, float* __restrict__ feat,
                                   int text_dim, int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int id = ids[i / text_dim];
  feat[i] = (id == 0 || id >= kTextVocab) ? 0.0f : emb[id * text_dim + i % text_dim];
}

// demb[id, d] = Σ_{t: ids[t]==id} dfeat[t, d]：每线程一个 (id, d)，按 token 顺序扫一遍
// （原子加版本求和顺序随机，同 seed 训练不可逐位复现）。<pad>=0 的行恒为 0。
__global__ void text_demb_kernel(const std::uint8_t* __restrict__ ids,
                                 const float* __restrict__ dfeat, float* __restrict__ demb,
                                 int tokens, int text_dim) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kTextVocab * text_dim) return;
  const int id = i / text_dim, d = i % text_dim;
  float acc = 0.0f;
  for (int t = 0; id != 0 && t < tokens; ++t) {
    if (ids[t] == id) acc += dfeat[t * text_dim + d];
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

template <typename T>
T* DeviceCnn::alloc(std::size_t count) {
  void* p = nullptr;
  const std::size_t bytes = (count == 0 ? 1 : count) * sizeof(T);
  if (cudaMalloc(&p, bytes) != cudaSuccess) throw_alloc_failure(bytes, __FILE__, __LINE__);
  owned_.push_back(p);
  return static_cast<T*>(p);
}

DeviceCnn::DeviceCnn(const CnnModel& model, int max_batch, const TrainState* state)
    : max_batch_(max_batch),
      in_c_(model.c1.in),
      h_(model.in_h),
      w_(model.in_w),
      text_dim_(model.text_dim),
      num_classes_(model.fc.out) {
  if (max_batch <= 0) throw std::invalid_argument("DeviceCnn: max_batch 必须为正");
  // 形状推导与 host 路径（conv stride 1 / pool 2,2）一致，最后用 fc.in 反查一次。
  int h = h_, w = w_, cin = in_c_;
  const Conv2d* convs[3] = {&model.c1, &model.c2, &model.c3};
  for (int l = 0; l < 3; ++l) {
    const Conv2d& c = *convs[l];
    if (c.in != cin) throw std::invalid_argument("DeviceCnn: 卷积层通道数不衔接");
    LayerDims& L = layer_[l];
    L = LayerDims{cin, c.out, c.kh, c.kw, c.pad, h, w,
                  h + 2 * c.pad - c.kh + 1, w + 2 * c.pad - c.kw + 1, 0, 0};
    if (L.oh < kPoolK || L.ow < kPoolK) throw std::invalid_argument("DeviceCnn: 输入尺寸过小");
    L.ph = (L.oh - kPoolK) / kPoolK + 1;
    L.pw = (L.ow - kPoolK) / kPoolK + 1;
    h = L.ph, w = L.pw, cin = c.out;
  }
  if (model.fc.in != cin * h * w) {
    throw std::invalid_argument("DeviceCnn: fc 输入维度与卷积输出不符");
  }
  if (text_dim_ > 0 && (model.film.in != kTextLen * text_dim_ || model.film.out != 2 * cin)) {
    throw std::invalid_argument("DeviceCnn: film 层维度与 text_dim/c3.out 不符");
  }
  for (const Tensor* t : cnn_params(model)) {
    seg_.push_back(Seg{total_params_, t->numel()});
    total_params_ += t->numel();
  }
  if (total_params_ > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("DeviceCnn: 参数量超出 int 范围");
  }
  params_ = alloc<float>(total_params_);
  grads_ = alloc<float>(total_params_);
  m_ = alloc<float>(total_params_);
  v_ = alloc<float>(total_params_);

  const std::size_t nb = static_cast<std::size_t>(max_batch_);
  x_u8_ = alloc<std::uint8_t>(nb * in_c_ * h_ * w_);
  x_ = alloc<float>(nb * in_c_ * h_ * w_);
  aug_ = alloc<dev::AugParams>(nb);
  float** conv_out[3] = {&c1_, &c2_, &c3_};
  float** pool_out[3] = {&p1_, &p2_, &p3_};
  float** pool_arg[3] = {&arg1_, &arg2_, &arg3_};
  float** d_conv[3] = {&dr1_, &dr2_, &dr3_};
  float** d_pool[3] = {&dp1_, &dp2_, &dflat_};  // 对各层池化输出的梯度；第三层即 fc 的 dx
  for (int l = 0; l < 3; ++l) {
    const LayerDims& L = layer_[l];
    const std::size_t conv_n = nb * L.c_out * L.oh * L.ow;
    const std::size_t pool_n = nb * L.c_out * L.ph * L.pw;
    *conv_out[l] = alloc<float>(conv_n);
    *d_conv[l] = alloc<float>(conv_n);
    *pool_out[l] = alloc<float>(pool_n);
    *pool_arg[l] = alloc<float>(pool_n);
    *d_pool[l] = alloc<float>(pool_n);
    ws_[l] = alloc<float>(dev::conv2d_backward_workspace(conv_shape(L, max_batch_)));
  }
  const std::size_t tfeat = static_cast<std::size_t>(kTextLen) * text_dim_;
  text_u8_ = alloc<std::uint8_t>(nb * kTextLen);
  text_feat_ = alloc<float>(nb * tfeat);
  dtext_feat_ = alloc<float>(nb * tfeat);
  film_out_ = alloc<float>(nb * 2 * cin);
  dfilm_ = alloc<float>(nb * 2 * cin);
  p3mod_ = alloc<float>(text_dim_ > 0 ? nb * model.fc.in : 0);
  logits_ = alloc<float>(nb * num_classes_);
  dlogits_ = alloc<float>(nb * num_classes_);
  d_sumsq_ = alloc<double>(1);
  upload(model, state);
}

DeviceCnn::~DeviceCnn() {
  for (void* p : owned_) cudaFree(p);
}

std::size_t DeviceCnn::seg_bytes(std::size_t k, std::size_t numel) const {
  if (numel != seg_[k].len) {
    throw std::invalid_argument("DeviceCnn: 第 " + std::to_string(k) + " 个张量元素数与构造时不一致");
  }
  return numel * sizeof(float);
}

void DeviceCnn::upload(const CnnModel& model, const TrainState* state) {
  const std::vector<const Tensor*> ps = cnn_params(model);
  if (ps.size() != seg_.size() || (state && state->adam.size() != seg_.size())) {
    throw std::invalid_argument("DeviceCnn: 参数/Adam 个数与构造时不一致");
  }
  for (std::size_t k = 0; k < seg_.size(); ++k) {
    CUDA_CHECK(cudaMemcpy(params_ + seg_[k].off, ps[k]->data(), seg_bytes(k, ps[k]->numel()),
                          cudaMemcpyHostToDevice));
    if (!state) continue;
    const Adam& a = state->adam[k];
    CUDA_CHECK(cudaMemcpy(m_ + seg_[k].off, a.m.data(), seg_bytes(k, a.m.numel()),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(v_ + seg_[k].off, a.v.data(), seg_bytes(k, a.v.numel()),
                          cudaMemcpyHostToDevice));
  }
}

void DeviceCnn::download(CnnModel& model, TrainState* state) const {
  std::vector<Tensor*> ps = cnn_params(model);
  if (ps.size() != seg_.size() || (state && state->adam.size() != seg_.size())) {
    throw std::invalid_argument("DeviceCnn: 参数/Adam 个数与构造时不一致");
  }
  for (std::size_t k = 0; k < seg_.size(); ++k) {
    CUDA_CHECK(cudaMemcpy(ps[k]->data(), params_ + seg_[k].off, seg_bytes(k, ps[k]->numel()),
                          cudaMemcpyDeviceToHost));
    if (!state) continue;
    Adam& a = state->adam[k];
    CUDA_CHECK(cudaMemcpy(a.m.data(), m_ + seg_[k].off, seg_bytes(k, a.m.numel()),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(a.v.data(), v_ + seg_[k].off, seg_bytes(k, a.v.numel()),
                          cudaMemcpyDeviceToHost));
  }
}

void DeviceCnn::film_forward(int n) {
  const LayerDims& L = layer_[2];
  const int tfeat = kTextLen * text_dim_;
  const int total = n * tfeat;
  text_gather_kernel<<<blocks_for(total), kThreads>>>(text_u8_, params_ + seg_[8].off, text_feat_,
                                                      text_dim_, total);
  launched("text_gather_kernel");
  dev::linear_forward(n, tfeat, 2 * L.c_out, text_feat_, params_ + seg_[9].off,
                      params_ + seg_[10].off, film_out_);
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
  const int tfeat = kTextLen * text_dim_;
  dev::linear_backward(n, tfeat, 2 * L.c_out, text_feat_, params_ + seg_[9].off, dfilm_,
                       grads_ + seg_[9].off, grads_ + seg_[10].off, dtext_feat_);
  text_demb_kernel<<<blocks_for(seg_[8].len), kThreads>>>(text_u8_, dtext_feat_,
                                                          grads_ + seg_[8].off, n * kTextLen,
                                                          text_dim_);
  launched("text_demb_kernel");
}

Tensor DeviceCnn::forward(const std::uint8_t* images, int n, const std::uint8_t* text,
                          const dev::AugParams* aug) {
  if (n <= 0 || n > max_batch_) {
    throw std::invalid_argument("DeviceCnn::forward: batch 必须在 1..max_batch 内");
  }
  if ((text_dim_ > 0) != (text != nullptr)) {
    throw std::invalid_argument("DeviceCnn::forward: 有文本分支必须传 text，无文本分支不能传");
  }
  n_ = 0;
  const std::size_t img = static_cast<std::size_t>(n) * in_c_ * h_ * w_;
  CUDA_CHECK(cudaMemcpy(x_u8_, images, img, cudaMemcpyHostToDevice));
  if (aug) CUDA_CHECK(cudaMemcpy(aug_, aug, n * sizeof(dev::AugParams), cudaMemcpyHostToDevice));
  dev::u8_to_float(x_u8_, x_, aug ? aug_ : nullptr, n, in_c_, h_, w_);
  const float* in = x_;
  float* conv_out[3] = {c1_, c2_, c3_};
  float* pool_out[3] = {p1_, p2_, p3_};
  float* pool_arg[3] = {arg1_, arg2_, arg3_};
  for (int l = 0; l < 3; ++l) {
    const LayerDims& L = layer_[l];
    dev::conv2d_forward(conv_shape(L, n), in, params_ + seg_[2 * l].off,
                        params_ + seg_[2 * l + 1].off, conv_out[l]);
    dev::relu_forward(conv_out[l], conv_out[l], n * L.c_out * L.oh * L.ow);
    dev::maxpool_forward(pool_shape(L, n), conv_out[l], pool_out[l], pool_arg[l]);
    in = pool_out[l];
  }
  if (text_dim_ > 0) {
    CUDA_CHECK(cudaMemcpy(text_u8_, text, static_cast<std::size_t>(n) * kTextLen,
                          cudaMemcpyHostToDevice));
    film_forward(n);
    in = p3mod_;
  }
  const LayerDims& L3 = layer_[2];
  dev::linear_forward(n, L3.c_out * L3.ph * L3.pw, num_classes_, in, params_ + seg_[6].off,
                      params_ + seg_[7].off, logits_);
  Tensor out({n, num_classes_});
  // 同步点：D2H 之前所有 kernel 的执行错误都会在这里报出来。
  CUDA_CHECK(cudaMemcpy(out.data(), logits_, out.numel() * sizeof(float), cudaMemcpyDeviceToHost));
  n_ = n;
  return out;
}

float DeviceCnn::backward_step(const Tensor& dlogits, float max_grad_norm, TrainState& state,
                               float weight_decay) {
  if (n_ == 0) throw std::logic_error("DeviceCnn::backward_step: 必须紧跟一次 forward");
  if (dlogits.ndim() != 2 || dlogits.dims()[0] != n_ || dlogits.dims()[1] != num_classes_) {
    throw std::invalid_argument("DeviceCnn::backward_step: dlogits 必须是 [n, num_classes]");
  }
  if (state.adam.size() != seg_.size()) throw std::invalid_argument("DeviceCnn: Adam 状态数不符");
  if (max_grad_norm < 0.0f) throw std::invalid_argument("DeviceCnn: max_grad_norm 不能为负");
  const int n = n_;
  n_ = 0;  // 参数即将更新，缓存的激活不再对应当前权重
  CUDA_CHECK(cudaMemcpy(dlogits_, dlogits.data(), dlogits.numel() * sizeof(float),
                        cudaMemcpyHostToDevice));
  const LayerDims& L3 = layer_[2];
  dev::linear_backward(n, L3.c_out * L3.ph * L3.pw, num_classes_, text_dim_ > 0 ? p3mod_ : p3_,
                       params_ + seg_[6].off, dlogits_, grads_ + seg_[6].off,
                       grads_ + seg_[7].off, dflat_);
  if (text_dim_ > 0) film_backward(n);  // dflat_ 原地变成对 p3 的梯度
  float* d_in = dflat_;
  float* conv_out[3] = {c1_, c2_, c3_};
  float* pool_arg[3] = {arg1_, arg2_, arg3_};
  float* d_conv[3] = {dr1_, dr2_, dr3_};
  float* d_prev[3] = {nullptr, dp1_, dp2_};  // 各层卷积对输入的梯度；第一层没有下游
  const float* conv_in[3] = {x_, p1_, p2_};
  for (int l = 2; l >= 0; --l) {
    const LayerDims& L = layer_[l];
    dev::maxpool_backward(pool_shape(L, n), pool_arg[l], d_in, d_conv[l]);
    // ReLU 输出作掩码：r>0 ⇔ 卷积输出>0，NaN 也同样透传，与保留卷积输出等价。
    dev::relu_backward(conv_out[l], d_conv[l], d_conv[l], n * L.c_out * L.oh * L.ow);
    dev::conv2d_backward(conv_shape(L, n), conv_in[l], params_ + seg_[2 * l].off, d_conv[l],
                         grads_ + seg_[2 * l].off, grads_ + seg_[2 * l + 1].off, d_prev[l],
                         ws_[l]);
    d_in = d_prev[l];
  }
  // 全局范数与裁剪系数的公式同 clip_grad_norm_cuda。
  CUDA_CHECK(cudaMemsetAsync(d_sumsq_, 0, sizeof(double)));
  dev::sumsq(grads_, d_sumsq_, static_cast<int>(total_params_));
  double sum = 0.0;
  CUDA_CHECK(cudaMemcpy(&sum, d_sumsq_, sizeof(double), cudaMemcpyDeviceToHost));
  if (!std::isfinite(sum)) throw std::runtime_error("DeviceCnn: 梯度含非有限值");
  const double norm = std::sqrt(sum);
  if (!(norm <= static_cast<double>(std::numeric_limits<float>::max()))) {
    throw std::runtime_error("DeviceCnn: 梯度范数超出 float 可表示范围");
  }
  const float raw = max_grad_norm / (static_cast<float>(norm) + 1e-6f);
  const float coef = raw > 1.0f ? 1.0f : raw;
  if (coef < 1.0f) dev::scale(grads_, coef, static_cast<int>(total_params_));
  for (std::size_t k = 0; k < seg_.size(); ++k) {
    Adam& a = state.adam[k];
    a.t += 1;
    const float bc1 = static_cast<float>(1.0 - std::pow(static_cast<double>(a.beta1), a.t));
    const float bc2 = static_cast<float>(1.0 - std::pow(static_cast<double>(a.beta2), a.t));
    dev::adam_step(params_ + seg_[k].off, grads_ + seg_[k].off, m_ + seg_[k].off,
                   v_ + seg_[k].off, a.lr, a.beta1, a.beta2, bc1, bc2, a.eps, weight_decay,
                   static_cast<int>(seg_[k].len));
  }
  return static_cast<float>(norm);
}

}  // namespace uavrgb
