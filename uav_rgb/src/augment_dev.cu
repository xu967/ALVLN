// uint8 NCHW → float [0,1]，可选逐样本数据增强（见 ops_dev.hpp 的 AugParams）。
// 增强全部在设备侧做：host 每批只多传 n 个 AugParams（24 B/样本），像素流量不变。
#include "cuda_check.hpp"
#include "uavrgb/ops_dev.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace uavrgb::dev {
namespace {

constexpr int kThreads = 256;

__global__ void u8_to_float_kernel(const std::uint8_t* __restrict__ src, float* __restrict__ dst,
                                   const AugParams* __restrict__ aug, int c, int h, int w,
                                   int total) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  if (aug == nullptr) {
    dst[i] = static_cast<float>(src[i]) / 255.0f;
    return;
  }
  const int x = i % w, y = (i / w) % h, nc = i / (w * h);  // nc = n*c + ch
  const AugParams a = aug[nc / c];
  const int sx = min(max((a.flip ? w - 1 - x : x) + a.dx, 0), w - 1);
  const int sy = min(max(y + a.dy, 0), h - 1);
  float v = static_cast<float>(src[(static_cast<std::size_t>(nc) * h + sy) * w + sx]) / 255.0f;
  v = powf(v, a.gamma);
  v = (v - 0.5f) * a.contrast + 0.5f + a.brightness;
  dst[i] = fminf(fmaxf(v, 0.0f), 1.0f);
}

}  // namespace

void u8_to_float(const std::uint8_t* src, float* dst, const AugParams* aug, int n, int c, int h,
                 int w) {
  const int total = n * c * h * w;
  if (total == 0) return;
  u8_to_float_kernel<<<(total + kThreads - 1) / kThreads, kThreads>>>(src, dst, aug, c, h, w,
                                                                       total);
  launched("u8_to_float_kernel");
}

}  // namespace uavrgb::dev
