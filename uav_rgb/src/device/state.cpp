// 参数、梯度与 Adam m/v 常驻；快照只在明确同步边界生成。
#include "uavrgb/device/state.hpp"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uavrgb::device {
namespace {

// 收尾失败必须标记FAILED并留诊断，但不能替换导致收尾的首次异常。
void check_cleanup(Context& context, cudaError_t result, const char* operation) noexcept {
  try {
    context.check(result, operation);
  } catch (...) {
    std::fprintf(stderr, "device transfer cleanup: %s: %s\n", operation,
                 cudaGetErrorString(result));
  }
}

// 提交中途失败也须先排空，避免锁页源/目标仍在被 DMA 使用时被释放。
template <class F> void transfer(Context& context, F submit) {
  try {
    submit();
    context.finish();
  } catch (...) {
    check_cleanup(context, cudaSetDevice(context.device()), "cudaSetDevice");
    check_cleanup(context, cudaStreamSynchronize(context.stream()), "cudaStreamSynchronize");
    throw;
  }
}

}  // namespace

DeviceTrainState::DeviceTrainState(Context& context, const CnnModel& model,
                                   const TrainState& state) {
  import(context, model, state);
}

DeviceTrainState::DeviceTrainState(DeviceTrainState&& other) noexcept { swap(other); }

DeviceTrainState& DeviceTrainState::operator=(DeviceTrainState&& other) noexcept {
  if (this != &other) {
    DeviceTrainState previous(std::move(other));
    swap(previous);
  }
  return *this;
}

void DeviceTrainState::swap(DeviceTrainState& other) noexcept {
  std::swap(layout_, other.layout_);
  params_.swap(other.params_);
  grads_.swap(other.grads_);
  adam_m_.swap(other.adam_m_);
  adam_v_.swap(other.adam_v_);
  std::swap(in_h_, other.in_h_);
  std::swap(in_w_, other.in_w_);
  std::swap(num_classes_, other.num_classes_);
  std::swap(epoch_, other.epoch_);
  std::swap(step_, other.step_);
  std::swap(optimizer_, other.optimizer_);
  class_weights_.swap(other.class_weights_);
}

void DeviceTrainState::import(Context& context, const CnnModel& model, const TrainState& state) {
  validate_train_state(model, state);
  context.activate();
  DeviceTrainState candidate;
  candidate.layout_ = make_parameter_layout(model);
  candidate.in_h_ = model.in_h;
  candidate.in_w_ = model.in_w;
  candidate.num_classes_ = model.fc.out;
  candidate.epoch_ = state.epoch;
  candidate.step_ = static_cast<std::uint64_t>(state.adam.front().t);
  const auto& adam = state.adam.front();
  candidate.optimizer_ = {adam.lr, adam.beta1, adam.beta2, adam.eps};
  candidate.class_weights_ = state.class_weights;
  const auto count = candidate.layout_.count;
  const auto bytes = checked_multiply(count, sizeof(float));
  PinnedBuffer packed(checked_multiply(bytes, 3));
  auto* parameters = packed.at<float>(0, count);
  auto* moments = packed.at<float>(bytes, count);
  auto* variances = packed.at<float>(checked_multiply(bytes, 2), count);
  const auto model_parameters = cnn_params(model);
  for (std::size_t i = 0; i < candidate.layout_.segments.size(); ++i) {
    const auto& segment = candidate.layout_.segments[i];
    std::copy_n(model_parameters[i]->data(), segment.count, parameters + segment.offset);
    std::copy_n(state.adam[i].m.data(), segment.count, moments + segment.offset);
    std::copy_n(state.adam[i].v.data(), segment.count, variances + segment.offset);
  }
  candidate.params_ = DeviceBuffer(bytes, context.device());
  candidate.grads_ = DeviceBuffer(bytes, context.device());
  candidate.adam_m_ = DeviceBuffer(bytes, context.device());
  candidate.adam_v_ = DeviceBuffer(bytes, context.device());
  transfer(context, [&] {
    candidate.params_.copy_from_host(context, parameters, bytes);
    candidate.adam_m_.copy_from_host(context, moments, bytes);
    candidate.adam_v_.copy_from_host(context, variances, bytes);
    context.check(cudaMemsetAsync(candidate.grads_.data(), 0, bytes, context.stream()),
                  "cudaMemsetAsync grads");
  });
  swap(candidate);
}

HostSnapshot DeviceTrainState::snapshot(Context& context) const {
  if (layout_.count == 0 || device() != context.device() ||
      params_.data() == nullptr || adam_m_.data() == nullptr || adam_v_.data() == nullptr) {
    throw std::invalid_argument("device snapshot: 空状态或 context device 不符");
  }
  if (step_ > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw std::overflow_error("device snapshot: step 超出 host Adam int 范围");
  }
  HostSnapshot result;
  result.model = make_cnn(in_h_, in_w_, 0, num_classes_);
  result.state = make_train_state(result.model, optimizer_.lr);
  result.state.epoch = epoch_;
  result.state.class_weights = class_weights_;
  const auto bytes = checked_multiply(layout_.count, sizeof(float));
  PinnedBuffer packed(checked_multiply(bytes, 3));
  auto* parameters = packed.at<float>(0, layout_.count);
  auto* moments = packed.at<float>(bytes, layout_.count);
  auto* variances = packed.at<float>(checked_multiply(bytes, 2), layout_.count);
  transfer(context, [&] {
    params_.copy_to_host(context, parameters, bytes);
    adam_m_.copy_to_host(context, moments, bytes);
    adam_v_.copy_to_host(context, variances, bytes);
  });
  const auto model_parameters = cnn_params(result.model);
  for (std::size_t i = 0; i < layout_.segments.size(); ++i) {
    const auto& segment = layout_.segments[i];
    auto& adam = result.state.adam[i];
    std::copy_n(parameters + segment.offset, segment.count, model_parameters[i]->data());
    std::copy_n(moments + segment.offset, segment.count, adam.m.data());
    std::copy_n(variances + segment.offset, segment.count, adam.v.data());
    adam.lr = optimizer_.lr;
    adam.beta1 = optimizer_.beta1;
    adam.beta2 = optimizer_.beta2;
    adam.eps = optimizer_.eps;
    adam.t = static_cast<int>(step_);
  }
  validate_train_state(result.model, result.state);
  return result;
}

}  // namespace uavrgb::device
