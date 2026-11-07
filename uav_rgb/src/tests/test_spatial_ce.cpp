// 逐格 softmax CE（深度分带辅助头）：CPU 参考 vs 手算、FP64 gradcheck、255 屏蔽、全屏蔽、
// CUDA 版与 CPU 版对拍、非法输入抛 invalid_argument。
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "uavrgb/ops.hpp"
#include "uavrgb/softmax_cuda.hpp"

namespace {

int g_fail = 0;

void expect(bool ok, const char* msg) {
  if (!ok) {
    ++g_fail;
    std::cerr << "FAIL: " << msg << "\n";
  }
}

template <typename F>
bool throws_invalid(F&& f) {
  try {
    f();
  } catch (const std::invalid_argument&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

double max_abs_diff(const uavrgb::Tensor& a, const uavrgb::Tensor& b) {
  if (a.dims() != b.dims()) return INFINITY;
  double worst = 0.0;
  for (std::size_t i = 0; i < a.numel(); ++i) worst = std::fmax(worst, std::fabs(double(a[i]) - b[i]));
  return worst;
}

}  // namespace

int main() {
  using namespace uavrgb;
  // [N=2, K=3, H=1, W=2]：4 个格，手算可核。通道步长 = H*W = 2。
  Tensor x = Tensor::from_flat({2, 3, 1, 2}, {// 样本 0：格 0 = (0,0,0)，格 1 = (1,2,3)
                                                0.f, 1.f, 0.f, 2.f, 0.f, 3.f,
                                                // 样本 1：格 0 = (2,0,-1)，格 1 = (0,0,0)
                                                2.f, 0.f, 0.f, 0.f, -1.f, 0.f});
  {
    // 全部有效：格 0 均匀 → ln3；格 1 目标 2 → ln(e1+e2+e3)-3；样本 1 格 0 目标 0 → ln(e2+1+e-1)-2；
    // 格 1 目标 1 → ln3。
    const std::vector<std::uint8_t> t = {0, 2, 0, 1};
    const double want = (std::log(3.0) + (std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0)) - 3.0) +
                         (std::log(std::exp(2.0) + 1.0 + std::exp(-1.0)) - 2.0) + std::log(3.0)) /
                        4.0;
    std::size_t valid = 0;
    const float loss = spatial_ce_loss(x, t, &valid);
    expect(valid == 4, "全有效时 valid=4");
    expect(std::fabs(loss - want) < 1e-6, "全有效 loss 与手算一致");
    const Tensor g = spatial_ce_grad(x, t);
    // 均匀格：grad = (1/3 - onehot)/4。
    expect(std::fabs(g[0] - (1.0 / 3.0 - 1.0) / 4.0) < 1e-6 && std::fabs(g[2] - (1.0 / 3.0) / 4.0) < 1e-6,
           "均匀格梯度 = (p - onehot)/n_valid");
    expect(gradcheck_spatial_ce(x, t) < 1e-6, "全有效 gradcheck < 1e-6");
  }
  {
    // 屏蔽两格：loss 只按剩下的 2 格平均，被屏蔽格梯度全 0。
    const std::vector<std::uint8_t> t = {kSpatialIgnore, 2, 0, kSpatialIgnore};
    std::size_t valid = 0;
    const float loss = spatial_ce_loss(x, t, &valid);
    const double want = ((std::log(std::exp(1.0) + std::exp(2.0) + std::exp(3.0)) - 3.0) +
                         (std::log(std::exp(2.0) + 1.0 + std::exp(-1.0)) - 2.0)) /
                        2.0;
    expect(valid == 2, "屏蔽后 valid=2");
    expect(std::fabs(loss - want) < 1e-6, "屏蔽格不计入 loss，按有效格数平均");
    const Tensor g = spatial_ce_grad(x, t);
    expect(g[0] == 0.0f && g[2] == 0.0f && g[4] == 0.0f, "样本 0 格 0 被屏蔽 → 梯度 0");
    expect(g[7] == 0.0f && g[9] == 0.0f && g[11] == 0.0f, "样本 1 格 1 被屏蔽 → 梯度 0");
    expect(std::fabs(g[1] - (std::exp(1.0) / (std::exp(1.0) + std::exp(2.0) + std::exp(3.0))) / 2.0) < 1e-6,
           "有效格梯度按 n_valid=2 归一");
    expect(gradcheck_spatial_ce(x, t) < 1e-6, "带屏蔽 gradcheck < 1e-6");
  }
  {
    // 全屏蔽：loss=0、梯度全 0、valid=0（训练器用 valid 判断本批是否有深度目标）。
    const std::vector<std::uint8_t> t(4, kSpatialIgnore);
    std::size_t valid = 7;
    expect(spatial_ce_loss(x, t, &valid) == 0.0f && valid == 0, "全屏蔽 loss=0 valid=0");
    const Tensor g = spatial_ce_grad(x, t);
    bool all_zero = true;
    for (std::size_t i = 0; i < g.numel(); ++i) all_zero = all_zero && g[i] == 0.0f;
    expect(all_zero, "全屏蔽梯度全 0");
  }
  {
    // CUDA 版对拍：K=8、H=15、W=20 与辅助头同形，随机 logits，约 1/3 格屏蔽。
    const int n = 3, k = 8, h = 15, w = 20;
    Tensor big({n, k, h, w});
    std::vector<std::uint8_t> t(static_cast<std::size_t>(n) * h * w);
    for (std::size_t i = 0; i < big.numel(); ++i) big[i] = std::sin(0.37f * static_cast<float>(i)) * 3.0f;
    for (std::size_t i = 0; i < t.size(); ++i) t[i] = (i % 3 == 0) ? kSpatialIgnore : static_cast<std::uint8_t>((i * 5) % k);
    std::size_t vh = 0;
    const float lh = spatial_ce_loss(big, t, &vh);
    const float ld = spatial_ce_loss_cuda(big, t);
    expect(vh == t.size() - (t.size() + 2) / 3, "CPU 有效格数 = 非屏蔽格数");
    expect(std::fabs(lh - ld) < 1e-5f, "CUDA loss 与 CPU 一致");
    const double dg = max_abs_diff(spatial_ce_grad(big, t), spatial_ce_grad_cuda(big, t));
    std::cout << "spatial_ce cuda parity: |Δloss|=" << std::fabs(lh - ld) << " max|Δgrad|=" << dg << "\n";
    expect(dg < 1e-6, "CUDA grad 与 CPU 一致");
    expect(gradcheck_spatial_ce(big, t, 1e-4) < 1e-5, "大张量 gradcheck < 1e-5");
  }
  // 非法输入：rank、targets 数量、target 越界、K 撞屏蔽码。
  expect(throws_invalid([&] { (void)spatial_ce_loss(Tensor::zeros({2, 3}), {0, 1}); }), "rank!=4 抛");
  expect(throws_invalid([&] { (void)spatial_ce_loss(x, {0, 1, 2}); }), "targets 数量不符抛");
  expect(throws_invalid([&] { (void)spatial_ce_grad(x, {0, 1, 3, 0}); }), "target>=K 抛");
  expect(throws_invalid([&] { (void)spatial_ce_loss(Tensor::zeros({1, 255, 1, 1}), {0}); }), "K>=255 抛");
  expect(throws_invalid([&] { (void)spatial_ce_loss_cuda(x, {0, 1, 2}); }), "CUDA 版 targets 数量不符抛");

  if (g_fail) {
    std::cerr << "test_spatial_ce: FAIL (" << g_fail << ")\n";
    return 1;
  }
  std::cout << "test_spatial_ce: PASS\n";
  return 0;
}
