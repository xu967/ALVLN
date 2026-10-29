// ReLU / MaxPool2d / Adam 层 CPU/CUDA parity + FP64 gradcheck 自检：无外部测试框架，返回 0 表示通过。
// 确定性输入；报告每个张量的最大绝对误差（doc 04 §6 N1/N3）。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "uavrgb/nn.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}

// 两个同形状 Tensor 的最大绝对误差。
double max_abs_err(const uavrgb::Tensor& a, const uavrgb::Tensor& b) {
  double m = 0.0;
  for (std::size_t i = 0; i < a.numel(); ++i) {
    m = std::max(m, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
  }
  return m;
}

}  // namespace

int main() {
  using namespace uavrgb;
  double worst = 0.0;

  // ---- ReLU：前向 + 反向 parity（输入含正/负/精确 0） ----
  {
    Tensor x = Tensor::zeros({3, 4, 5});
    Tensor dy = Tensor::zeros({3, 4, 5});
    for (std::size_t i = 0; i < x.numel(); ++i) {
      x[i] = 0.1f * (static_cast<float>(i % 11) - 5.0f);  // i%11==5 处为精确 0
      dy[i] = 0.03f * (static_cast<float>(i % 7) - 3.0f);
    }
    const Tensor y_cpu = relu_forward(x);
    const Tensor y_cuda = relu_forward_cuda(x);
    const Tensor dx_cpu = relu_backward(x, dy);
    const Tensor dx_cuda = relu_backward_cuda(x, dy);
    const double e_y = max_abs_err(y_cpu, y_cuda);
    const double e_dx = max_abs_err(dx_cpu, dx_cuda);
    worst = std::max({worst, e_y, e_dx});
    std::cout << "relu parity        max|y_cpu-y_cuda|  = " << e_y << "\n";
    std::cout << "                   max|dx_cpu-dx_cuda|= " << e_dx << "\n";

    // 导数约定自检：x==0 处 CPU/CUDA 都必须给出 0。
    bool zero_ok = true;
    for (std::size_t i = 0; i < x.numel(); ++i) {
      if (x[i] == 0.0f && (dx_cpu[i] != 0.0f || dx_cuda[i] != 0.0f)) zero_ok = false;
    }
    check(zero_ok, "ReLU x==0 的次梯度约定（0）CPU/CUDA 一致");
  }

  // ---- MaxPool2d（k=s=2）：前向 y/argmax + 反向 dx parity；H/W 为奇数验证向下取整 ----
  {
    const MaxPool2d m(2, 2);
    Tensor x = Tensor::zeros({2, 2, 5, 7});
    for (std::size_t i = 0; i < x.numel(); ++i) {
      // 窗口内下标差为 1/7/8，(37*Δ mod 101) 非零 → 同一 2×2 窗口内无并列。
      x[i] = 0.01f * static_cast<float>((static_cast<int>(i) * 37) % 101) - 0.5f;
    }
    Tensor dy = Tensor::zeros({2, 2, 2, 3});
    for (std::size_t i = 0; i < dy.numel(); ++i) {
      dy[i] = 0.1f * (static_cast<float>(i % 5) - 2.0f);
    }
    const MaxPoolResult f_cpu = maxpool2d_forward(m, x);
    const MaxPoolResult f_cuda = maxpool2d_forward_cuda(m, x);
    const Tensor dx_cpu = maxpool2d_backward(m, x, f_cpu.argmax, dy);
    const Tensor dx_cuda = maxpool2d_backward_cuda(m, x, f_cuda.argmax, dy);
    const double e_y = max_abs_err(f_cpu.y, f_cuda.y);
    const double e_arg = max_abs_err(f_cpu.argmax, f_cuda.argmax);
    const double e_dx = max_abs_err(dx_cpu, dx_cuda);
    worst = std::max({worst, e_y, e_dx});
    std::cout << "maxpool parity     max|y_cpu-y_cuda|    = " << e_y << "\n";
    std::cout << "                   max|arg_cpu-arg_cuda|= " << e_arg << "\n";
    std::cout << "                   max|dx_cpu-dx_cuda|  = " << e_dx << "\n";
    check(e_arg == 0.0, "MaxPool argmax CPU/CUDA 完全一致");
  }

  // ---- Adam：同初值跑 5 步，逐步比较 param / 一阶矩 m / 二阶矩 v ----
  {
    Tensor p0 = Tensor::zeros({4, 5, 6});
    for (std::size_t i = 0; i < p0.numel(); ++i) {
      p0[i] = 0.1f * static_cast<float>(static_cast<int>(i) % 13 - 6);
    }
    Tensor p_cpu = p0;
    Tensor p_cuda = p0;
    Adam st_cpu = adam_make(p0, 1e-2f);
    Adam st_cuda = adam_make(p0, 1e-2f);
    double worst_adam = 0.0;
    for (int step = 1; step <= 5; ++step) {
      Tensor g = Tensor::zeros({4, 5, 6});
      for (std::size_t i = 0; i < g.numel(); ++i) {
        g[i] = 0.02f * static_cast<float>((static_cast<int>(i) * 7 + step * 13) % 19 - 9);
      }
      adam_step(st_cpu, p_cpu, g);
      adam_step_cuda(st_cuda, p_cuda, g);
      const double e_p = max_abs_err(p_cpu, p_cuda);
      const double e_m = max_abs_err(st_cpu.m, st_cuda.m);
      const double e_v = max_abs_err(st_cpu.v, st_cuda.v);
      worst_adam = std::max({worst_adam, e_p, e_m, e_v});
      std::cout << "adam step " << step << "        max|p|=" << e_p << " max|m|=" << e_m
                << " max|v|=" << e_v << "\n";
    }
    worst = std::max(worst, worst_adam);

    // 闭式自检：beta1=beta2=0 时一步更新 p -= lr*sign(g)（eps 可忽略）。
    Tensor p = Tensor::from_flat({2}, {1.0f, -2.0f});
    const Tensor g = Tensor::from_flat({2}, {3.0f, -4.0f});
    Adam st = adam_make(p, 0.5f, 0.0f, 0.0f, 1e-8f);
    adam_step(st, p, g);
    check(std::fabs(p[0] - 0.5f) < 1e-5f && std::fabs(p[1] + 1.5f) < 1e-5f,
          "Adam 单步闭式更新 p -= lr*sign(g)");
  }

  // ---- FP64 中心差分 gradcheck：loss = Σ relu(x)² ----
  {
    Tensor x = Tensor::zeros({2, 3, 4});
    for (std::size_t i = 0; i < x.numel(); ++i) {
      x[i] = 0.2f * static_cast<float>(static_cast<int>(i) % 9 - 4);  // 含 0 与 ±0.2..±0.8
    }
    const double err = gradcheck_relu(x);
    std::cout << "relu gradcheck_max_err    = " << err << "\n";
    check(err < 1e-5, "ReLU FP64 中心差分 gradcheck 最大绝对误差 < 1e-5");
  }

  // ---- FP64 中心差分 gradcheck：loss = Σ y²（MaxPool k=s=2） ----
  {
    const MaxPool2d m(2, 2);
    Tensor x = Tensor::zeros({1, 2, 4, 6});
    for (std::size_t i = 0; i < x.numel(); ++i) {
      x[i] = 0.01f * static_cast<float>((static_cast<int>(i) * 37) % 101) - 0.5f;
    }
    const double err = gradcheck_maxpool2d(m, x);
    std::cout << "maxpool gradcheck_max_err = " << err << "\n";
    check(err < 1e-5, "MaxPool FP64 中心差分 gradcheck 最大绝对误差 < 1e-5");
  }

  // ---- 梯度裁剪：global L2 norm + 原地缩放 CPU/CUDA parity + 公式独立自检 ----
  {
    Tensor a = Tensor::zeros({2, 3});
    Tensor b = Tensor::zeros({4});
    for (std::size_t i = 0; i < a.numel(); ++i) {
      a[i] = 0.1f * static_cast<float>(static_cast<int>(i) % 7 - 3);
    }
    for (std::size_t i = 0; i < b.numel(); ++i) {
      b[i] = 0.2f * static_cast<float>(static_cast<int>(i) - 2);  // 先转 int，防 size_t 下溢
    }
    Tensor a_cuda = a;
    Tensor b_cuda = b;
    const float n_cpu = global_grad_norm({&a, &b});
    const float n_cuda = global_grad_norm_cuda({&a_cuda, &b_cuda});

    // 独立公式自检：double 直接重算 norm 与裁剪系数。
    double sq = 0.0;
    for (std::size_t i = 0; i < a.numel(); ++i) sq += static_cast<double>(a[i]) * a[i];
    for (std::size_t i = 0; i < b.numel(); ++i) sq += static_cast<double>(b[i]) * b[i];
    const float n_ref = static_cast<float>(std::sqrt(sq));
    const float max_norm = 0.5f;  // 实测 norm≈0.656 > 0.5，确保触发裁剪
    const float expected_coef = std::min(1.0f, max_norm / (n_ref + 1e-6f));
    const float c_cpu = clip_grad_norm({&a, &b}, max_norm);
    const float c_cuda = clip_grad_norm_cuda({&a_cuda, &b_cuda}, max_norm);
    const double e_a = max_abs_err(a, a_cuda);
    const double e_b = max_abs_err(b, b_cuda);
    worst = std::max({worst, e_a, e_b});
    std::cout << "clip norm parity   |n_cpu-n_cuda|=" << std::fabs(n_cpu - n_cuda)
              << " coef cpu/cuda/ref=" << c_cpu << "/" << c_cuda << "/" << expected_coef << "\n";
    std::cout << "                   max|a_cpu-a_cuda|=" << e_a
              << " max|b_cpu-b_cuda|=" << e_b << "\n";
    check(std::fabs(n_cpu - n_ref) < 1e-6f && std::fabs(n_cuda - n_ref) < 1e-6f,
          "global_grad_norm CPU/CUDA 与独立 double 重算一致");
    check(std::fabs(c_cpu - expected_coef) < 1e-6f && std::fabs(c_cuda - expected_coef) < 1e-6f,
          "clip 系数 min(1, max_norm/(norm+1e-6)) 与独立重算一致");
    check(c_cpu < 1.0f, "裁剪系数 <1（测试值确实超过 max_norm）");
  }

  // ---- 形状异常自检（防越界/错位） ----
  {
    bool threw = false;
    try {
      (void)relu_backward(Tensor::zeros({2, 3}), Tensor::zeros({3, 2}));
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "relu_backward 形状不一致抛 invalid_argument");

    threw = false;
    try {
      const MaxPool2d m(2, 2);
      Tensor x = Tensor::zeros({1, 1, 4, 4});
      (void)maxpool2d_backward(m, x, Tensor::zeros({1, 1, 2, 2}), Tensor::zeros({1, 1, 1, 1}));
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "maxpool2d_backward dy 形状不一致抛 invalid_argument");
  }

  // CPU/CUDA FP32 运算顺序不同（CPU 无 FMA、CUDA 默认 FMA），1e-4 是明确的 parity 门槛。
  for (const auto& dims : {std::vector<std::int64_t>{1, 1, 1, 4},
                           std::vector<std::int64_t>{1, 1, 4, 1}}) {
    const Tensor small(dims);
    for (bool gpu : {false, true}) {
      bool threw = false;
      try {
        if (gpu) (void)maxpool2d_forward_cuda(MaxPool2d(2, 2), small);
        else (void)maxpool2d_forward(MaxPool2d(2, 2), small);
      } catch (const std::invalid_argument&) { threw = true; }
      check(threw, "MaxPool CPU/CUDA 拒绝小于窗口的 H/W");
    }
  }
  check(worst < 1e-4, "ReLU/MaxPool/Adam/clip CPU/CUDA 最大绝对误差 < 1e-4");

  if (g_failures == 0) {
    std::cout << "test_nn_cuda: PASS\n";
    return 0;
  }
  std::cerr << "test_nn_cuda: " << g_failures << " 项失败\n";
  return 1;
}
