// CNN 组装：3→16→32→64 三层卷积（3x3, pad1）+ ReLU + 2x2 MaxPool，Flatten 后接 Linear。
// 全部算子走 CUDA 实现（src/*_cuda.cu）：CPU 参考是标量朴素循环，实测慢 20–100 倍
// （batch 32 下 c1 前向 CPU 0.64 s vs CUDA 0.032 s），训练循环只走 CUDA 路径。
#include "uavrgb/model.hpp"

#include "uavrgb/conv_cuda.hpp"
#include "uavrgb/linear_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace uavrgb {
namespace {

const MaxPool2d kPool(2, 2);  // 160x120 → 80x60 → 40x30 → 20x15

// Tensor 没有 view/reshape，只能按元素数拷贝一份（训练每步最多几 MB，可忽略）。
Tensor reshape_copy(const Tensor& t, std::vector<std::int64_t> dims) {
  Tensor o(std::move(dims));
  if (o.numel() != t.numel()) {
    throw std::invalid_argument("reshape_copy: 元素数不一致（" + std::to_string(t.numel()) +
                                " vs " + std::to_string(o.numel()) + "）");
  }
  std::copy(t.data(), t.data() + t.numel(), o.data());
  return o;
}

std::int64_t flat_dim(const CnnModel& m) {
  return static_cast<std::int64_t>(m.c3.out) * (m.in_h / 8) * (m.in_w / 8);
}

}  // namespace

CnnModel make_cnn(int in_h, int in_w, std::uint32_t seed, int num_classes) {
  if (in_h <= 0 || in_w <= 0) {
    throw std::invalid_argument("make_cnn: in_h/in_w 必须为正");
  }
  if (num_classes <= 0) {
    throw std::invalid_argument("make_cnn: num_classes 必须为正");
  }
  if (in_h % 8 != 0 || in_w % 8 != 0) {
    throw std::invalid_argument("make_cnn: 输入 H/W 必须是 8 的倍数（三次 2x2 pool）");
  }
  CnnModel m;
  m.in_h = in_h;
  m.in_w = in_w;
  m.c1 = Conv2d(3, 16, 3, 3, 1);
  m.c2 = Conv2d(16, 32, 3, 3, 1);
  m.c3 = Conv2d(32, 64, 3, 3, 1);
  const std::int64_t flat = flat_dim(m);
  if (flat > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("make_cnn: fc 输入维超出 int 范围");
  }
  m.fc = Linear(static_cast<int>(flat), num_classes);

  std::mt19937 rng(seed);
  auto xavier = [&rng](Tensor& w, int fan_in, int fan_out) {
    const float lim = std::sqrt(6.0f / static_cast<float>(fan_in + fan_out));
    std::uniform_real_distribution<float> u(-lim, lim);
    for (std::size_t i = 0; i < w.numel(); ++i) w[i] = u(rng);
  };
  xavier(m.c1.W, m.c1.in * 9, m.c1.out * 9);
  xavier(m.c2.W, m.c2.in * 9, m.c2.out * 9);
  xavier(m.c3.W, m.c3.in * 9, m.c3.out * 9);
  xavier(m.fc.W, m.fc.in, m.fc.out);
  // b 由 Conv2d/Linear 构造时清零，这里不再重复写。
  detail::check_conv_params(m.c1);
  detail::check_conv_params(m.c2);
  detail::check_conv_params(m.c3);
  detail::check_linear_params(m.fc);
  return m;
}

std::size_t num_params(const CnnModel& m) {
  std::size_t n = 0;
  for (const Tensor* t : cnn_params(m)) n += t->numel();
  return n;
}

std::vector<Tensor*> cnn_params(CnnModel& m) {
  return {&m.c1.W, &m.c1.b, &m.c2.W, &m.c2.b, &m.c3.W, &m.c3.b, &m.fc.W, &m.fc.b};
}

std::vector<const Tensor*> cnn_params(const CnnModel& m) {
  return {&m.c1.W, &m.c1.b, &m.c2.W, &m.c2.b, &m.c3.W, &m.c3.b, &m.fc.W, &m.fc.b};
}

std::vector<Tensor*> cnn_grad_tensors(CnnGrad& g) {
  return {&g.c1.dW, &g.c1.db, &g.c2.dW, &g.c2.db, &g.c3.dW, &g.c3.db, &g.fc.dW, &g.fc.db};
}

Tensor forward(const CnnModel& m, const Tensor& x, CnnCache* cache) {
  if (x.ndim() != 4 || x.dims()[1] != m.c1.in || x.dims()[2] != m.in_h ||
      x.dims()[3] != m.in_w) {
    throw std::invalid_argument("forward: x 必须是 [N," + std::to_string(m.c1.in) + "," +
                                std::to_string(m.in_h) + "," + std::to_string(m.in_w) + "]");
  }
  if (cache) cache->x = x;

  Tensor c1 = conv2d_forward_cuda(m.c1, x);
  Tensor r1 = relu_forward_cuda(c1);
  MaxPoolResult p1 = maxpool2d_forward_cuda(kPool, r1);
  Tensor c2 = conv2d_forward_cuda(m.c2, p1.y);
  Tensor r2 = relu_forward_cuda(c2);
  MaxPoolResult p2 = maxpool2d_forward_cuda(kPool, r2);
  Tensor c3 = conv2d_forward_cuda(m.c3, p2.y);
  Tensor r3 = relu_forward_cuda(c3);
  MaxPoolResult p3 = maxpool2d_forward_cuda(kPool, r3);

  Tensor flat = reshape_copy(p3.y, {x.dims()[0], flat_dim(m)});
  if (cache) {
    cache->c1 = std::move(c1);
    cache->r1 = std::move(r1);
    cache->p1 = std::move(p1);
    cache->c2 = std::move(c2);
    cache->r2 = std::move(r2);
    cache->p2 = std::move(p2);
    cache->c3 = std::move(c3);
    cache->r3 = std::move(r3);
    cache->p3 = std::move(p3);
    cache->flat = flat;  // 还要给 fc 反向用，保留副本
  }
  return linear_forward_cuda(m.fc, flat);
}

CnnGrad backward(const CnnModel& m, const CnnCache& c, const Tensor& dlogits) {
  if (c.flat.numel() == 0) {
    throw std::invalid_argument("backward: cache 为空，需用带 cache 的 forward 填充");
  }
  CnnGrad g;
  g.fc = linear_backward_cuda(m.fc, c.flat, dlogits);
  // fc 的 dx 是 [N, fc.in]，还原成 [N,64,h',w'] 再散回 pool3 的 argmax。
  Tensor dp3 = reshape_copy(g.fc.dx, c.p3.y.dims());
  Tensor dr3 = maxpool2d_backward_cuda(kPool, c.r3, c.p3.argmax, dp3);
  Tensor dc3 = relu_backward_cuda(c.c3, dr3);
  g.c3 = conv2d_backward_cuda(m.c3, c.p2.y, dc3);
  Tensor dr2 = maxpool2d_backward_cuda(kPool, c.r2, c.p2.argmax, g.c3.dx);
  Tensor dc2 = relu_backward_cuda(c.c2, dr2);
  g.c2 = conv2d_backward_cuda(m.c2, c.p1.y, dc2);
  Tensor dr1 = maxpool2d_backward_cuda(kPool, c.r1, c.p1.argmax, g.c2.dx);
  Tensor dc1 = relu_backward_cuda(c.c1, dr1);
  g.c1 = conv2d_backward_cuda(m.c1, c.x, dc1);
  return g;
}

CnnGrad backward(const CnnModel& m, const Tensor& x, const Tensor& dlogits) {
  CnnCache cache;
  forward(m, x, &cache);
  return backward(m, cache, dlogits);
}

}  // namespace uavrgb
