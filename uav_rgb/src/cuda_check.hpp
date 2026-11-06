#pragma once
// .cu 内部共用的 CUDA 错误检查（不对外暴露，故不放 include/）。
// 抛 std::runtime_error 而不是 std::exit：调用方可以 catch 后降 batch/重试，栈上 RAII 正常析构。
#include <cuda_runtime.h>

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

}  // namespace uavrgb
