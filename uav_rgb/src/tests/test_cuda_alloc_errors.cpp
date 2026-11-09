// cudaMalloc 失败的报错必须带 requested/free/total（2026-11-03 a50 OOM 复盘）。
#include "uavrgb/cuda_alloc.hpp"
#include "uavrgb/model_dev.hpp"

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

  // 3) DeviceCnn 构造中途 cudaMalloc 失败时，之前已分配的段必须释放（FIXLIST_R2 B4）：
  //    构造函数抛出后析构不会跑，owned_ 里的指针就是泄漏。960×1280 的模型 params/grads/m/v
  //    约 78 MB，先于按 max_batch 分配的激活；max_batch 取 2^20 让激活那一步必失败。
  //    连做 4 次，泄漏版会少掉约 300 MB，噪声（同卡其它进程）远小于阈值 100 MB。
  const uavrgb::CnnModel model = uavrgb::make_cnn(960, 1280, 1, 4);
  std::size_t free0 = 0, total0 = 0;
  if (cudaMemGetInfo(&free0, &total0) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cudaMemGetInfo 失败\n");
    return 1;
  }
  int thrown = 0;
  for (int i = 0; i < 4; ++i) {
    try {
      uavrgb::DeviceCnn dev(model, 1 << 20);
    } catch (const std::runtime_error&) {
      ++thrown;
    }
  }
  if (thrown != 4) {
    std::printf("SKIP: 本卡放得下 2^20 批的激活（%d/4 次失败），泄漏检查无法构造\n", thrown);
    return 0;
  }
  std::size_t free1 = 0, total1 = 0;
  cudaMemGetInfo(&free1, &total1);
  const double leaked_mb = (static_cast<double>(free0) - static_cast<double>(free1)) / 1048576.0;
  std::printf("DeviceCnn 构造失败 4 次后显存变化 %.1f MB\n", leaked_mb);
  if (leaked_mb > 100.0) {
    std::fprintf(stderr, "FAIL: DeviceCnn 构造失败后泄漏显存 %.1f MB（已分配段未释放）\n", leaked_mb);
    return 1;
  }

  std::printf("test_cuda_alloc_errors: PASS\n");
  return 0;
}
