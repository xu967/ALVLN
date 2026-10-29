#pragma once
// 静态图张量（第一阶段仅 CPU、连续 row-major 存储）。CUDA 侧后续由独立 allocator
// 管理，Tensor 不隐式管理设备内存；所有权/生命周期规则见 doc 04 §3。
#include <cstddef>
#include <cstdint>
#include <vector>

namespace uavrgb {

class Tensor {
 public:
  Tensor() : dims_{}, data_(1, 0.0f) {}  // 默认构造为标量（1 元素），避免空存储被索引
  explicit Tensor(std::vector<std::int64_t> dims);  // 全零 float32
  static Tensor zeros(std::vector<std::int64_t> dims);
  static Tensor from_flat(std::vector<std::int64_t> dims, const std::vector<float>& data);

  const std::vector<std::int64_t>& dims() const { return dims_; }
  std::size_t ndim() const { return dims_.size(); }
  std::size_t numel() const;

  float* data() { return data_.data(); }
  const float* data() const { return data_.data(); }

  float& operator[](std::size_t i) { return data_[i]; }
  const float& operator[](std::size_t i) const { return data_[i]; }

  void zero();

 private:
  std::vector<std::int64_t> dims_;
  std::vector<float> data_;
};

}  // namespace uavrgb
