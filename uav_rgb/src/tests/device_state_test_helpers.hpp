#pragma once
// 状态传输必须逐 bit 保真；这里不用浮点容差掩盖布局、符号零或复制错误。
#include "uavrgb/device/state.hpp"

#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace device_state_test {
using namespace uavrgb;
using namespace uavrgb::device;

inline void require(bool ok, const std::string& message) {
  if (!ok) throw std::runtime_error(message);
}

template <class Function> void rejects(Function&& function, const std::string& name) {
  try {
    function();
  } catch (const std::exception& error) {
    require(*error.what() != '\0', name + ": 缺少错误说明");
    return;
  }
  throw std::runtime_error(name + ": 未拒绝非法输入");
}

inline HostSnapshot fixture() {
  HostSnapshot host;
  host.model = make_cnn(120, 160, 73);
  host.state = make_train_state(host.model, 0.002f);
  host.state.epoch = 3;
  host.state.class_weights = {0.5f, 1.0f, 2.0f, 4.0f};
  const auto parameters = cnn_params(host.model);
  for (std::size_t segment = 0; segment < parameters.size(); ++segment) {
    auto& adam = host.state.adam[segment];
    adam.t = 17;
    for (std::size_t i = 0; i < parameters[segment]->numel(); ++i) {
      const float value = static_cast<float>(segment + 1) * 0.125f +
                          static_cast<float>(i) * 0.00000095367431640625f;
      (*parameters[segment])[i] = (i % 2 == 0) ? value : -value;
      adam.m[i] = -value * 0.5f;
      adam.v[i] = value * 0.25f;
    }
  }
  host.model.c1.W[0] = -0.0f;
  host.state.adam[7].m[0] = -0.0f;
  return host;
}

inline void same_tensor(const Tensor& expected, const Tensor& actual, const std::string& name) {
  require(expected.dims() == actual.dims(), name + ": shape 不同");
  for (std::size_t i = 0; i < expected.numel(); ++i) {
    if (std::memcmp(expected.data() + i, actual.data() + i, sizeof(float)) != 0) {
      throw std::runtime_error(name + ": 首次 bit 差异 index=" + std::to_string(i));
    }
  }
}

inline void same_snapshot(const HostSnapshot& expected, const HostSnapshot& actual) {
  require(expected.model.in_h == actual.model.in_h && expected.model.in_w == actual.model.in_w &&
              expected.model.fc.in == actual.model.fc.in &&
              expected.model.fc.out == actual.model.fc.out, "snapshot 模型元数据不同");
  require(expected.state.epoch == actual.state.epoch &&
              expected.state.class_weights == actual.state.class_weights, "snapshot 训练元数据不同");
  const auto first = cnn_params(expected.model), second = cnn_params(actual.model);
  require(expected.state.adam.size() == actual.state.adam.size(), "Adam 段数不同");
  for (std::size_t i = 0; i < first.size(); ++i) {
    const std::string name = "segment=" + std::to_string(i);
    same_tensor(*first[i], *second[i], name + "/p");
    const auto& a = expected.state.adam[i];
    const auto& b = actual.state.adam[i];
    require(a.t == b.t && a.lr == b.lr && a.beta1 == b.beta1 && a.beta2 == b.beta2 &&
                a.eps == b.eps, name + ": Adam 合同不同");
    same_tensor(a.m, b.m, name + "/m");
    same_tensor(a.v, b.v, name + "/v");
  }
}
}  // namespace device_state_test
