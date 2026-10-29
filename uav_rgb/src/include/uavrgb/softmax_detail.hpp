#pragma once
// ops_cpu.cpp（无权）与 ops_weighted.cpp（加权）共用的行内 stable softmax 辅助。
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace uavrgb {
namespace detail {

// 行内 stable softmax：先减行最大值再 exp，避免上溢；输出行最大值与 exp 之和。
inline void softmax_row(const float* row, std::int64_t c, float* out_max, float* out_sum) {
  float mx = row[0];
  for (std::int64_t j = 1; j < c; ++j) mx = std::max(mx, row[j]);
  float sum = 0.0f;
  for (std::int64_t j = 0; j < c; ++j) sum += std::exp(row[j] - mx);
  *out_max = mx;
  *out_sum = sum;
}

}  // namespace detail
}  // namespace uavrgb
