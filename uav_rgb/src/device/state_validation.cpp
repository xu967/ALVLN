// 先验证完整 host 状态；形状或数值不合法时不提交任何 CUDA 工作。
#include "uavrgb/device/state.hpp"

#include <cmath>
#include <stdexcept>

namespace uavrgb::device {
namespace {

[[noreturn]] void bad(const std::string& message) {
  throw std::invalid_argument("device state: " + message);
}

std::size_t tensor_count(const Tensor& tensor, const std::vector<std::int64_t>& shape,
                         const std::string& name) {
  if (tensor.dims() != shape) bad(name + " shape 不符");
  std::size_t count = 1;
  for (const auto dimension : shape) {
    if (dimension <= 0) bad(name + " shape 非正");
    count = checked_multiply(count, static_cast<std::size_t>(dimension));
  }
  (void)checked_multiply(count, sizeof(float));
  if (tensor.numel() != count) bad(name + " 元素数不符");
  return count;
}

void check_conv(const Conv2d& layer, int input, int output, const char* name) {
  if (layer.in != input || layer.out != output || layer.kh != 3 || layer.kw != 3 ||
      layer.pad != 1) bad(std::string(name) + " 不符合固定 CNN 合同");
}

void finite_tensor(const Tensor& tensor, const std::string& name, bool nonnegative = false) {
  for (std::size_t j = 0; j < tensor.numel(); ++j) {
    if (!std::isfinite(tensor[j]) || (nonnegative && tensor[j] < 0.0f)) {
      bad(name + " 非法数值，index=" + std::to_string(j));
    }
  }
}

void check_optimizer(const Adam& adam) {
  if (adam.t < 0 || !std::isfinite(adam.lr) || adam.lr <= 0.0f ||
      !std::isfinite(adam.beta1) || adam.beta1 < 0.0f || adam.beta1 >= 1.0f ||
      !std::isfinite(adam.beta2) || adam.beta2 < 0.0f || adam.beta2 >= 1.0f ||
      !std::isfinite(adam.eps) || adam.eps <= 0.0f) bad("Adam t/lr/beta/eps 非法");
}

}  // namespace

ParameterLayout make_parameter_layout(const CnnModel& model) {
  if (model.in_h <= 0 || model.in_w <= 0 || model.in_h % 8 != 0 || model.in_w % 8 != 0) {
    bad("H/W 必须为正且是8的倍数");
  }
  check_conv(model.c1, 3, 16, "c1");
  check_conv(model.c2, 16, 32, "c2");
  check_conv(model.c3, 32, 64, "c3");
  const auto features = checked_multiply(64, checked_multiply(
      static_cast<std::size_t>(model.in_h / 8), static_cast<std::size_t>(model.in_w / 8)));
  if (model.fc.in <= 0 || static_cast<std::size_t>(model.fc.in) != features || model.fc.out <= 0) {
    bad("fc.in/out 与 CNN 不符");
  }
  const std::array<std::vector<std::int64_t>, 8> shapes{{
      {16, 3, 3, 3}, {16}, {32, 16, 3, 3}, {32}, {64, 32, 3, 3}, {64},
      {model.fc.out, model.fc.in}, {model.fc.out}}};
  constexpr const char* names[] = {"c1.W", "c1.b", "c2.W", "c2.b", "c3.W", "c3.b", "fc.W", "fc.b"};
  ParameterLayout layout;
  const auto parameters = cnn_params(model);
  for (std::size_t i = 0; i < layout.segments.size(); ++i) {
    auto& segment = layout.segments[i];
    segment.name = names[i];
    segment.shape = shapes[i];
    segment.count = tensor_count(*parameters[i], segment.shape, segment.name);
    segment.offset = layout.count;
    layout.count = checked_add(layout.count, segment.count);
  }
  (void)checked_multiply(layout.count, sizeof(float));
  return layout;
}

void validate_train_state(const CnnModel& model, const TrainState& state) {
  const auto layout = make_parameter_layout(model);
  if (state.adam.size() != layout.segments.size()) bad("Adam 状态数必须为8");
  if (state.epoch < 0) bad("epoch 不能为负");
  const auto parameters = cnn_params(model);
  const auto& first = state.adam.front();
  for (std::size_t i = 0; i < state.adam.size(); ++i) {
    const auto& adam = state.adam[i];
    const auto& segment = layout.segments[i];
    check_optimizer(adam);
    if (adam.t != first.t || adam.lr != first.lr || adam.beta1 != first.beta1 ||
        adam.beta2 != first.beta2 || adam.eps != first.eps) bad("八段 Adam 配置或 step 不一致");
    (void)tensor_count(adam.m, segment.shape, segment.name + ".m");
    (void)tensor_count(adam.v, segment.shape, segment.name + ".v");
    finite_tensor(*parameters[i], segment.name);
    finite_tensor(adam.m, segment.name + ".m");
    finite_tensor(adam.v, segment.name + ".v", true);
  }
  if (!state.class_weights.empty()) {
    if (state.class_weights.size() != static_cast<std::size_t>(model.fc.out)) {
      bad("class_weights 长度不符");
    }
    double sum = 0.0;
    for (const auto weight : state.class_weights) {
      if (!std::isfinite(weight) || weight < 0.0f) bad("class_weights 非有限或为负");
      sum += weight;
    }
    if (!(sum > 0.0)) bad("class_weights 总和必须为正");
  }
}

}  // namespace uavrgb::device
