#pragma once
// 独立 host 快照与 GPU 常驻状态；首批只提供导入/导出，不提交训练更新。
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/device/buffer.hpp"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace uavrgb::device {

struct ParameterSegment {
  std::string name;
  std::vector<std::int64_t> shape;
  std::size_t offset = 0;  // float 元素偏移
  std::size_t count = 0;
};

struct ParameterLayout {
  std::array<ParameterSegment, 8> segments;
  std::size_t count = 0;
};

ParameterLayout make_parameter_layout(const CnnModel& model);
void validate_train_state(const CnnModel& model, const TrainState& state);

struct HostSnapshot {
  CnnModel model;
  TrainState state;
};

struct OptimizerConfig {
  float lr = 1e-3f;
  float beta1 = 0.9f;
  float beta2 = 0.999f;
  float eps = 1e-8f;
};

class DeviceTrainState {
 public:
  DeviceTrainState() noexcept = default;
  DeviceTrainState(Context& context, const CnnModel& model, const TrainState& state);
  DeviceTrainState(const DeviceTrainState&) = delete;
  DeviceTrainState& operator=(const DeviceTrainState&) = delete;
  DeviceTrainState(DeviceTrainState&& other) noexcept;
  DeviceTrainState& operator=(DeviceTrainState&& other) noexcept;

  // 全量校验、候选上传和 finish 成功后才替换已有状态。
  void import(Context& context, const CnnModel& model, const TrainState& state);
  HostSnapshot snapshot(Context& context) const;
  void swap(DeviceTrainState& other) noexcept;

  const ParameterLayout& layout() const noexcept { return layout_; }
  std::uint64_t step() const noexcept { return step_; }
  int device() const noexcept { return params_.device(); }
  DeviceBuffer& params() noexcept { return params_; }
  const DeviceBuffer& params() const noexcept { return params_; }
  DeviceBuffer& grads() noexcept { return grads_; }
  const DeviceBuffer& grads() const noexcept { return grads_; }
  DeviceBuffer& adam_m() noexcept { return adam_m_; }
  const DeviceBuffer& adam_m() const noexcept { return adam_m_; }
  DeviceBuffer& adam_v() noexcept { return adam_v_; }
  const DeviceBuffer& adam_v() const noexcept { return adam_v_; }

 private:
  ParameterLayout layout_;
  DeviceBuffer params_, grads_, adam_m_, adam_v_;
  int in_h_ = 0, in_w_ = 0, num_classes_ = 0, epoch_ = 0;
  std::uint64_t step_ = 0;
  OptimizerConfig optimizer_;
  std::vector<float> class_weights_;
};

}  // namespace uavrgb::device
