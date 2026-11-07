// DeviceCnn 的深度分带辅助头（STUDENT-P2）：1x1 卷积 + 逐格 CE，全部在显存里完成，
// 每步只 D2H 一个 double（CE 和）和一个计数（命中格数）。从 model_dev.cu 拆出以守住行数预算。
#include "cuda_check.hpp"
#include "uavrgb/model_dev.hpp"
#include "uavrgb/ops.hpp"  // kSpatialIgnore
#include "uavrgb/ops_dev.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace uavrgb {
namespace {

dev::ConvShape aux_shape(const DeviceCnn::LayerDims& L3, int n, int k) {
  // 1x1、pad 0：输出尺寸就是第三层池化输出的 ph×pw。
  return dev::ConvShape{n, L3.c_out, L3.ph, L3.pw, k, 1, 1, 0, L3.ph, L3.pw};
}

}  // namespace

void DeviceCnn::alloc_aux(const CnnModel& model) {
  const std::size_t nb = static_cast<std::size_t>(max_batch_);
  const LayerDims& L3 = layer_[2];
  const std::size_t hw = static_cast<std::size_t>(L3.ph) * L3.pw;
  aux_logits_ = alloc<float>(nb * aux_classes_ * hw);
  daux_ = alloc<float>(nb * aux_classes_ * hw);
  dp3aux_ = alloc<float>(nb * L3.c_out * hw);
  aux_tgt_ = alloc<std::uint8_t>(nb * hw);
  ws_aux_ = alloc<float>(dev::conv2d_backward_workspace(aux_shape(L3, max_batch_, model.aux.out)));
  d_aux_loss_ = alloc<double>(1);
  d_aux_correct_ = alloc<unsigned long long>(1);
}

DeviceCnn::AuxStats DeviceCnn::aux_loss(const std::uint8_t* targets, float lambda) {
  if (aux_classes_ == 0) throw std::logic_error("DeviceCnn::aux_loss: 模型没有辅助头");
  if (n_ == 0) throw std::logic_error("DeviceCnn::aux_loss: 必须紧跟一次 forward");
  if (!(lambda >= 0.0f) || !std::isfinite(lambda)) {
    throw std::invalid_argument("DeviceCnn::aux_loss: lambda 必须为非负有限数");
  }
  const int n = n_;
  const LayerDims& L3 = layer_[2];
  const int hw = L3.ph * L3.pw;
  const std::size_t cells = static_cast<std::size_t>(n) * hw;
  AuxStats st;
  for (std::size_t i = 0; i < cells; ++i) {
    if (targets[i] == kSpatialIgnore) continue;
    if (targets[i] >= aux_classes_) throw std::invalid_argument("DeviceCnn::aux_loss: 目标越界");
    ++st.valid;
  }
  CUDA_CHECK(cudaMemcpy(aux_tgt_, targets, cells, cudaMemcpyHostToDevice));
  dev::conv2d_forward(aux_shape(L3, n, aux_classes_), fc_input(), params_ + seg_[seg_aux_].off,
                      params_ + seg_[seg_aux_ + 1].off, aux_logits_);
  CUDA_CHECK(cudaMemsetAsync(d_aux_loss_, 0, sizeof(double)));
  CUDA_CHECK(cudaMemsetAsync(d_aux_correct_, 0, sizeof(unsigned long long)));
  // 全部屏蔽时 scale 取 0：梯度全 0、loss 为 0，与 CPU 参考一致，不除零。
  const float scale = st.valid == 0 ? 0.0f : lambda / static_cast<float>(st.valid);
  dev::spatial_ce(aux_logits_, aux_tgt_, n, aux_classes_, hw, scale, daux_, d_aux_loss_,
                  d_aux_correct_);
  double total = 0.0;
  unsigned long long correct = 0;
  CUDA_CHECK(cudaMemcpy(&total, d_aux_loss_, sizeof(double), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&correct, d_aux_correct_, sizeof(correct), cudaMemcpyDeviceToHost));
  st.loss = st.valid == 0 ? 0.0f : static_cast<float>(total / static_cast<double>(st.valid));
  st.correct = static_cast<std::size_t>(correct);
  aux_pending_ = true;
  return st;
}

void DeviceCnn::aux_backward(int n) {
  const std::size_t w_len = seg_[seg_aux_].len, b_len = seg_[seg_aux_ + 1].len;
  if (!aux_pending_) {
    // 本步没有深度目标（纯动作步或 eval 后的续训）：辅助头梯度清 0，Adam 仍按统一顺序走。
    CUDA_CHECK(cudaMemsetAsync(grads_ + seg_[seg_aux_].off, 0, (w_len + b_len) * sizeof(float)));
    return;
  }
  const LayerDims& L3 = layer_[2];
  dev::conv2d_backward(aux_shape(L3, n, aux_classes_), fc_input(), params_ + seg_[seg_aux_].off,
                       daux_, grads_ + seg_[seg_aux_].off, grads_ + seg_[seg_aux_ + 1].off,
                       dp3aux_, ws_aux_);
  dev::add_inplace(dflat_, dp3aux_, n * L3.c_out * L3.ph * L3.pw);
  aux_pending_ = false;
}

}  // namespace uavrgb
