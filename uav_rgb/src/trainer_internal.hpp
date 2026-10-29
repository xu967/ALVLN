#pragma once
// trainer.cpp / trainer_eval.cpp 共用的 batch 切片工具；不对外安装。
#include "uavrgb/dataset.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace uavrgb {
namespace detail {

inline std::vector<int> take_labels(const UavrDataset& ds,
                                    const std::vector<std::uint32_t>& idx,
                                    std::size_t start, std::size_t end) {
  std::vector<int> y;
  y.reserve(end - start);
  for (std::size_t i = start; i < end; ++i) y.push_back(ds.labels[idx[i]]);
  return y;
}

inline std::vector<std::uint32_t> slice(const std::vector<std::uint32_t>& v, std::size_t s,
                                        std::size_t e) {
  return std::vector<std::uint32_t>(v.begin() + static_cast<std::ptrdiff_t>(s),
                                    v.begin() + static_cast<std::ptrdiff_t>(e));
}

}  // namespace detail
}  // namespace uavrgb
