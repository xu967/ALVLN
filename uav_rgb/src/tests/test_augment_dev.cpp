// 设备侧 uint8→float（+增强）kernel 的下标与数值语义：恒等参数 = src/255；翻转/平移取对像素；
// 亮度/对比度/gamma 的公式与截断。增强下标算错会让 A/B 结论失效，所以单独钉死。
#include "uavrgb/ops_dev.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

// 跑一次 kernel：n 个样本的 NCHW uint8 → float。
std::vector<float> run(const std::vector<std::uint8_t>& src, const uavrgb::dev::AugParams* aug,
                       int n, int c, int h, int w) {
  std::uint8_t* d_src = nullptr;
  float* d_dst = nullptr;
  uavrgb::dev::AugParams* d_aug = nullptr;
  cudaMalloc(&d_src, src.size());
  cudaMalloc(&d_dst, src.size() * sizeof(float));
  cudaMemcpy(d_src, src.data(), src.size(), cudaMemcpyHostToDevice);
  if (aug) {
    cudaMalloc(&d_aug, sizeof(uavrgb::dev::AugParams) * n);
    cudaMemcpy(d_aug, aug, sizeof(uavrgb::dev::AugParams) * n, cudaMemcpyHostToDevice);
  }
  uavrgb::dev::u8_to_float(d_src, d_dst, d_aug, n, c, h, w);
  std::vector<float> out(src.size());
  cudaMemcpy(out.data(), d_dst, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_src);
  cudaFree(d_dst);
  if (d_aug) cudaFree(d_aug);
  return out;
}

}  // namespace

int main() {
  using uavrgb::dev::AugParams;
  const int n = 2, c = 3, h = 5, w = 7;
  std::vector<std::uint8_t> src(static_cast<std::size_t>(n) * c * h * w);
  for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>((i * 37 + 11) % 256);
  auto at = [&](const std::vector<float>& v, int s, int ch, int y, int x) {
    return v[((static_cast<std::size_t>(s) * c + ch) * h + y) * w + x];
  };
  auto px = [&](int s, int ch, int y, int x) {
    return src[((static_cast<std::size_t>(s) * c + ch) * h + y) * w + x] / 255.0f;
  };

  // 1) aug 为空：逐位等于 src/255.0f。
  const std::vector<float> plain = run(src, nullptr, n, c, h, w);
  bool ok = true;
  for (std::size_t i = 0; i < src.size(); ++i) ok = ok && plain[i] == src[i] / 255.0f;
  check(ok, "无增强时逐位等于 src/255.0f");

  // 2) 恒等参数：与无增强相差不超过 1 ulp（powf(v,1) 不保证逐位）。
  std::vector<AugParams> id(n, AugParams{1.0f, 1.0f, 0.0f, 0, 0, 0});
  const std::vector<float> same = run(src, id.data(), n, c, h, w);
  double worst = 0.0;
  for (std::size_t i = 0; i < src.size(); ++i) worst = std::fmax(worst, std::fabs(same[i] - plain[i]));
  check(worst < 1e-6, "恒等增强参数 ≈ 无增强");

  // 3) 样本 0 翻转，样本 1 平移 (dx=+2, dy=-1)，边缘复制。
  std::vector<AugParams> geo = {AugParams{1.0f, 1.0f, 0.0f, 0, 0, 1}, AugParams{1.0f, 1.0f, 0.0f, 2, -1, 0}};
  const std::vector<float> g = run(src, geo.data(), n, c, h, w);
  ok = true;
  for (int ch = 0; ch < c; ++ch) {
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        ok = ok && std::fabs(at(g, 0, ch, y, x) - px(0, ch, y, w - 1 - x)) < 1e-6;
        const int sx = std::min(x + 2, w - 1), sy = std::max(y - 1, 0);
        ok = ok && std::fabs(at(g, 1, ch, y, x) - px(1, ch, sy, sx)) < 1e-6;
      }
    }
  }
  check(ok, "翻转取 (y, w-1-x)，平移取 (clamp(y+dy), clamp(x+dx))");

  // 4) 光度：v' = clamp(((v^γ) - 0.5)·对比度 + 0.5 + 亮度, 0, 1)。
  std::vector<AugParams> photo(n, AugParams{0.8f, 1.2f, 0.1f, 0, 0, 0});
  const std::vector<float> p = run(src, photo.data(), n, c, h, w);
  worst = 0.0;
  bool clamped = true;
  for (std::size_t i = 0; i < src.size(); ++i) {
    const double v = src[i] / 255.0;
    const double want = std::fmin(1.0, std::fmax(0.0, (std::pow(v, 0.8) - 0.5) * 1.2 + 0.5 + 0.1));
    worst = std::fmax(worst, std::fabs(p[i] - want));
    clamped = clamped && p[i] >= 0.0f && p[i] <= 1.0f;
  }
  check(worst < 1e-5 && clamped, "gamma/对比度/亮度公式与截断");

  if (g_fail) {
    std::fprintf(stderr, "test_augment_dev: FAIL (%d)\n", g_fail);
    return 1;
  }
  std::printf("test_augment_dev: PASS\n");
  return 0;
}
