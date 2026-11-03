#pragma once
// cudaMalloc 失败时的统一报错：必须带 requested/free/total 字节。
// 只报一句 "out of memory" 分不清「被别的进程占满卡」和「自己泄漏」——
// 2026-11-03 的 a50 训练 OOM 就因为缺这三个数，只能停在推测。
#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace uavrgb {

inline void throw_alloc_failure(std::size_t bytes, const char* file, int line) {
  cudaGetLastError();  // 清掉 cudaMalloc 留下的 sticky error，否则下面的查询直接返回它
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
  std::string msg = std::string("CUDA 显存分配失败 ") + file + ":" + std::to_string(line) +
                    "：申请 " + std::to_string(bytes) + " 字节";
  if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
    msg += "，现可用 " + std::to_string(free_bytes) + " / 共 " + std::to_string(total_bytes) + " 字节";
  } else {
    msg += "，cudaMemGetInfo 亦失败";
  }
  throw std::runtime_error(msg);
}

}  // namespace uavrgb
