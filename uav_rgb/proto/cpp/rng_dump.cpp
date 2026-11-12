// 对拍用的 C++ 小程序：用与 uavrgb_train 同一套标准库调用（std::mt19937、uniform_real_distribution<float>、std::shuffle），
// 把结果打成文本，proto/test_cpp_rng.py 拿 Python 复刻（proto/cpp_rng.py）逐项比对。
// 编译（与主线同一个 g++，不开 -march=native，不开 -ffast-math）：
//   g++ -O3 -std=c++17 -o rng_dump proto/cpp/rng_dump.cpp
// 用法：
//   rng_dump raw    <seed> <n>                   前 n 个 mt19937 原始输出（每行一个十进制数）
//   rng_dump uniform <seed> <lim> <n>            uniform_real_distribution<float>(-lim, lim) 的前 n 个值（每行十六进制 float 位型）
//   rng_dump shuffle <seed> <N> <epochs>         iota(N) 用同一个引擎连续 shuffle epochs 次；每次输出整个排列（空格分隔）
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

int main(int argc, char** argv) {
  if (argc < 2) return 2;
  const std::string mode = argv[1];
  if (mode == "raw" && argc == 4) {
    std::mt19937 rng(static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)));
    const long n = std::strtol(argv[3], nullptr, 10);
    for (long i = 0; i < n; ++i) std::printf("%lu\n", static_cast<unsigned long>(rng()));
    return 0;
  }
  if (mode == "uniform" && argc == 5) {
    std::mt19937 rng(static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)));
    const float lim = std::strtof(argv[3], nullptr);
    const long n = std::strtol(argv[4], nullptr, 10);
    std::uniform_real_distribution<float> u(-lim, lim);
    for (long i = 0; i < n; ++i) {
      const float v = u(rng);
      std::uint32_t bits;
      std::memcpy(&bits, &v, sizeof bits);
      std::printf("%08x\n", bits);
    }
    return 0;
  }
  if (mode == "shuffle" && argc == 5) {
    std::mt19937 rng(static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)));
    const long n = std::strtol(argv[3], nullptr, 10);
    const long epochs = std::strtol(argv[4], nullptr, 10);
    std::vector<std::uint32_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), 0u);
    for (long e = 0; e < epochs; ++e) {
      std::shuffle(order.begin(), order.end(), rng);
      for (std::size_t i = 0; i < order.size(); ++i) std::printf(i ? " %u" : "%u", order[i]);
      std::printf("\n");
    }
    return 0;
  }
  std::fprintf(stderr, "用法见文件头\n");
  return 2;
}
