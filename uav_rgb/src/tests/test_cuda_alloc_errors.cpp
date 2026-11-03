// cudaMalloc 失败的报错必须带 requested/free/total（2026-11-03 a50 OOM 复盘）。
#include "uavrgb/cuda_alloc.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <stdexcept>
#include <string>

int main() {
  // 1) 报错必须能定位到「申请多少、还剩多少、卡有多大、哪一行」。
  bool ok = false;
  try {
    uavrgb::throw_alloc_failure(123456789u, "x.cu", 42);
  } catch (const std::runtime_error& e) {
    const std::string m = e.what();
    ok = m.find("x.cu:42") != std::string::npos && m.find("123456789") != std::string::npos &&
         m.find("现可用") != std::string::npos && m.find("共") != std::string::npos;
    if (!ok) std::fprintf(stderr, "FAIL: 报错缺字段: %s\n", m.c_str());
  }
  if (!ok) return 1;

  // 2) 真的失败过一次之后仍要查得到显存：不先清 sticky error，cudaMemGetInfo 会直接返回它。
  void* p = nullptr;
  const std::size_t huge = std::size_t{1} << 40;  // 1 TiB，任何一张卡都分配不出来
  if (cudaMalloc(&p, huge) == cudaSuccess) {
    cudaFree(p);
    std::printf("SKIP: 本卡分配到了 1 TiB\n");
    return 0;
  }
  bool got_numbers = false;
  try {
    uavrgb::throw_alloc_failure(huge, "y.cu", 7);
  } catch (const std::runtime_error& e) {
    got_numbers = std::string(e.what()).find("现可用") != std::string::npos;
    if (!got_numbers) std::fprintf(stderr, "FAIL: 失败后查不到显存: %s\n", e.what());
  }
  if (!got_numbers) return 1;

  std::printf("test_cuda_alloc_errors: PASS\n");
  return 0;
}
