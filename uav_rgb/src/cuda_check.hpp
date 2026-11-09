#pragma once
// .cu 内部共用的 CUDA 小工具（不对外暴露，故不放 include/）：错误检查、launch 后查错、
// 设备缓冲 RAII、线程块换算。此前 conv/nn/linear/softmax/spatial_ce 五个 .cu 各带一份
// 同样的拷贝（AUDIT_SRC #5），改一处漏四处；现在只留这一份。
// 抛 std::runtime_error 而不是 std::exit：调用方可以 catch 后降 batch/重试，栈上 RAII 正常析构，
// checkpoint 落盘不会被跳过。
#include "uavrgb/cuda_alloc.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                          \
  do {                                                                            \
    const cudaError_t err_ = (call);                                              \
    if (err_ != cudaSuccess) {                                                    \
      throw std::runtime_error(std::string("CUDA error ") + __FILE__ + ":" +      \
                               std::to_string(__LINE__) + ": " +                  \
                               cudaGetErrorString(err_));                         \
    }                                                                             \
  } while (0)

namespace uavrgb {

// launch 后立刻查配置错误（0 block、grid 超限）；执行期错误留给调用方的同步点。
inline void launched(const char* name) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA kernel ") + name + ": " + cudaGetErrorString(err));
  }
}

// host 侧对拍/独立算子用：抓异步错误后同步等 kernel 结束。训练主干不同步，只用 launched。
inline void check_after_kernel(const char* name) {
  launched(name);
  CUDA_CHECK(cudaDeviceSynchronize());
}

// 极简设备缓冲 RAII：析构自动释放，异常路径也不泄漏。按字节申请；0 字节按 1 字节申请，
// 省得调用方为「可选输出」单独写分支。
template <typename T>
struct DeviceBuf {
  T* p = nullptr;
  DeviceBuf() = default;
  explicit DeviceBuf(std::size_t bytes) {
    if (bytes == 0) bytes = 1;
    if (cudaMalloc(&p, bytes) != cudaSuccess) throw_alloc_failure(bytes, __FILE__, __LINE__);
  }
  ~DeviceBuf() {
    if (p) cudaFree(p);
  }
  DeviceBuf(DeviceBuf&& o) noexcept : p(o.p) { o.p = nullptr; }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
};

inline constexpr int kThreads = 256;
inline int blocks_for(std::int64_t total) {
  return static_cast<int>((total + kThreads - 1) / kThreads);
}

}  // namespace uavrgb
