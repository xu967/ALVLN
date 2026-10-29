#include "uavrgb/tensor.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    ++g_failures;
  }
}

}  // namespace

int main() {
  using uavrgb::Tensor;

  // numel：2x3 与标量。
  check(Tensor({2, 3}).numel() == 6, "Tensor({2,3}).numel()==6");
  check(Tensor(std::vector<std::int64_t>{}).numel() == 1, "标量 numel()==1");

  // zeros 全零。
  Tensor z = Tensor::zeros({2, 3});
  bool all_zero = true;
  for (std::size_t i = 0; i < z.numel(); ++i) {
    all_zero = all_zero && z[i] == 0.0f;
  }
  check(all_zero, "zeros({2,3}) 全零");

  // 标量 zeros 也是单元素全零。
  Tensor s = Tensor::zeros({});
  check(s.numel() == 1 && s[0] == 0.0f, "标量 zeros 全零");

  // 默认构造：标量（1 元素全零），避免空存储被索引。
  Tensor d;
  check(d.ndim() == 0 && d.numel() == 1, "默认构造为空标量");

  // from_flat：尺寸不符抛 std::invalid_argument。
  bool threw = false;
  try {
    Tensor::from_flat({2, 3}, {1.0f, 2.0f});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "from_flat 尺寸不符抛 invalid_argument");

  // from_flat：尺寸正确时通过 operator[] 往返。
  Tensor f = Tensor::from_flat({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
  bool roundtrip = true;
  for (std::size_t i = 0; i < 6; ++i) {
    roundtrip = roundtrip && f[i] == static_cast<float>(i + 1);
  }
  check(roundtrip, "from_flat 经 operator[] 往返");

  // zero() 清空已修改的张量。
  f[0] = 42.0f;
  f.zero();
  bool cleared = true;
  for (std::size_t i = 0; i < f.numel(); ++i) {
    cleared = cleared && f[i] == 0.0f;
  }
  check(cleared, "zero() 清空已修改张量");

  if (g_failures == 0) {
    std::cout << "test_tensor: PASS\n";
    return 0;
  }
  std::cerr << "test_tensor: " << g_failures << " 项失败\n";
  return 1;
}
