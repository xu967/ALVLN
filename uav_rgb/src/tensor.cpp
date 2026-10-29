#include "uavrgb/tensor.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace uavrgb {

Tensor::Tensor(std::vector<std::int64_t> dims) : dims_(std::move(dims)) {
  for (std::int64_t d : dims_) {
    if (d < 0) {
      throw std::invalid_argument("Tensor: 维度不能为负");
    }
  }
  // 连续 row-major float32，构造即全零。
  data_.assign(numel(), 0.0f);
}

Tensor Tensor::zeros(std::vector<std::int64_t> dims) {
  return Tensor(std::move(dims));
}

Tensor Tensor::from_flat(std::vector<std::int64_t> dims,
                         const std::vector<float>& data) {
  Tensor t(std::move(dims));
  if (data.size() != t.numel()) {
    throw std::invalid_argument("from_flat: data.size() != numel()");
  }
  t.data_ = data;
  return t;
}

std::size_t Tensor::numel() const {
  if (dims_.empty()) {
    return 1;  // 标量（ndim==0）按 1 个元素计。
  }
  std::size_t n = 1;
  for (std::int64_t d : dims_) {
    n *= static_cast<std::size_t>(d);
  }
  return n;
}

void Tensor::zero() {
  std::fill(data_.begin(), data_.end(), 0.0f);
}

}  // namespace uavrgb
