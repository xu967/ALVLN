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

CnnModel make_cnn(int in_h, int in_w, std::uint32_t seed, int num_classes, int text_dim) {
  if (in_h <= 0 || in_w <= 0) {
    throw std::invalid_argument("make_cnn: in_h/in_w 必须为正");
  }
  if (num_classes <= 0) {
    throw std::invalid_argument("make_cnn: num_classes 必须为正");
  }
  if (text_dim < 0 || text_dim > kMaxTextDim) {
    throw std::invalid_argument("make_cnn: text_dim 必须在 0.." +
                                std::to_string(kMaxTextDim));
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
  m.text_dim = text_dim;
  m.fc = Linear(static_cast<int>(flat), num_classes);
  if (text_dim > 0) {
    const std::int64_t tfeat = static_cast<std::int64_t>(kTextLen) * text_dim;
    if (tfeat > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("make_cnn: 文本特征维超出 int 范围");
    }
    m.text_emb = Tensor({kTextVocab, text_dim});
    m.film = Linear(static_cast<int>(tfeat), 2 * m.c3.out);
  }

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
  if (text_dim > 0) {
    xavier(m.text_emb, kTextVocab, text_dim);
    xavier(m.film.W, m.film.in, m.film.out);
  }
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
  std::vector<Tensor*> p = {&m.c1.W, &m.c1.b, &m.c2.W, &m.c2.b,
                            &m.c3.W, &m.c3.b, &m.fc.W, &m.fc.b};
  if (m.text_dim > 0) {
    p.push_back(&m.text_emb);
    p.push_back(&m.film.W);
    p.push_back(&m.film.b);
  }
  return p;
}

std::vector<const Tensor*> cnn_params(const CnnModel& m) {
  std::vector<const Tensor*> p = {&m.c1.W, &m.c1.b, &m.c2.W, &m.c2.b,
                                  &m.c3.W, &m.c3.b, &m.fc.W, &m.fc.b};
  if (m.text_dim > 0) {
    p.push_back(&m.text_emb);
    p.push_back(&m.film.W);
    p.push_back(&m.film.b);
  }
  return p;
}

std::vector<Tensor*> cnn_grad_tensors(CnnGrad& g) {
  std::vector<Tensor*> p = {&g.c1.dW, &g.c1.db, &g.c2.dW, &g.c2.db,
                            &g.c3.dW, &g.c3.db, &g.fc.dW, &g.fc.db};
  // 默认构造的 Tensor 是 1 元素标量（ndim 0），用它表示"没有文本分支"。
  if (g.text_demb.ndim() != 0) {
    p.push_back(&g.text_demb);
    p.push_back(&g.film.dW);
    p.push_back(&g.film.db);
  }
  return p;
}

Tensor forward(const CnnModel& m, const Tensor& x, CnnCache* cache,
               const std::vector<std::uint8_t>* text) {
  if (x.ndim() != 4 || x.dims()[1] != m.c1.in || x.dims()[2] != m.in_h ||
      x.dims()[3] != m.in_w) {
    throw std::invalid_argument("forward: x 必须是 [N," + std::to_string(m.c1.in) + "," +
                                std::to_string(m.in_h) + "," + std::to_string(m.in_w) + "]");
  }
  const std::size_t n = static_cast<std::size_t>(x.dims()[0]);
  if (m.text_dim > 0) {
    if (!text) throw std::invalid_argument("forward: 模型有文本分支，必须传 text ids");
    if (text->size() != n * static_cast<std::size_t>(kTextLen)) {
      throw std::invalid_argument("forward: text ids 长度必须是 N*kTextLen");
    }
  } else if (text) {
    throw std::invalid_argument("forward: 模型没有文本分支，不接受 text ids");
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

  // 文本分支：token id → embedding 查表 → 按位置拼接 [kTextLen × text_dim]；
  // <pad> 的槽位保持 0。再经 film 得到逐通道 γ/β，调制 conv 最后一层特征。
  Tensor text_feat;
  Tensor film_out;
  Tensor p3mod;               // 有文本时的调制结果
  const Tensor* p3y = &p3.y;  // 无文本时直接复用池化特征，避免整块深拷贝
  const std::int64_t tfeat =
      static_cast<std::int64_t>(kTextLen) * (m.text_dim > 0 ? m.text_dim : 0);
  if (m.text_dim > 0) {
    text_feat = Tensor({x.dims()[0], tfeat});
    for (std::size_t i = 0; i < n; ++i) {
      for (int l = 0; l < kTextLen; ++l) {
        const std::uint8_t id = (*text)[i * kTextLen + l];
        if (id == 0 || id >= kTextVocab) continue;  // 0 = <pad>，未知值按 pad 处理
        const std::size_t dst = i * tfeat + static_cast<std::size_t>(l) * m.text_dim;
        for (int d = 0; d < m.text_dim; ++d) {
          text_feat[dst + d] = m.text_emb[id * m.text_dim + d];
        }
      }
    }
    film_out = linear_forward_cuda(m.film, text_feat);
    const int channels = m.c3.out;
    const std::int64_t hh = p3.y.dims()[2];
    const std::int64_t ww = p3.y.dims()[3];
    p3mod = Tensor(p3.y.dims());
    for (std::size_t i = 0; i < n; ++i) {
      for (int c = 0; c < channels; ++c) {
        const float gamma = film_out[i * 2 * channels + c];
        const float beta = film_out[i * 2 * channels + channels + c];
        const std::size_t base = (i * channels + c) * hh * ww;
        for (std::int64_t k = 0; k < hh * ww; ++k) {
          p3mod[base + k] = (1.0f + gamma) * p3.y[base + k] + beta;
        }
      }
    }
    p3y = &p3mod;
  }
  Tensor flat = reshape_copy(*p3y, {x.dims()[0], flat_dim(m)});
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
    cache->text_feat = text_feat;
    cache->film_out = film_out;
    if (text) cache->text_ids = *text;
  }
  return linear_forward_cuda(m.fc, flat);
}

CnnGrad backward(const CnnModel& m, const CnnCache& c, const Tensor& dlogits) {
  if (c.flat.numel() == 0) {
    throw std::invalid_argument("backward: cache 为空，需用带 cache 的 forward 填充");
  }
  CnnGrad g;
  g.fc = linear_backward_cuda(m.fc, c.flat, dlogits);
  // fc 的 dx 是 [N, flat]，还原成 [N,64,h',w']；有文本时先反传 FiLM。
  Tensor dp3 = reshape_copy(g.fc.dx, c.p3.y.dims());
  if (m.text_dim > 0) {
    const std::size_t n = static_cast<std::size_t>(c.p3.y.dims()[0]);
    if (c.text_ids.size() != n * static_cast<std::size_t>(kTextLen)) {
      throw std::invalid_argument("backward: cache.text_ids 长度与 batch 不符");
    }
    const std::int64_t tfeat = static_cast<std::int64_t>(kTextLen) * m.text_dim;
    const int channels = m.c3.out;
    const std::int64_t hh = c.p3.y.dims()[2];
    const std::int64_t ww = c.p3.y.dims()[3];
    Tensor d_film({static_cast<std::int64_t>(n), 2 * channels});
    Tensor dym(c.p3.y.dims());
    for (std::size_t i = 0; i < n; ++i) {
      for (int c0 = 0; c0 < channels; ++c0) {
        const float gamma = c.film_out[i * 2 * channels + c0];
        const std::size_t base = (i * channels + c0) * hh * ww;
        double dg = 0.0;
        double db = 0.0;
        for (std::int64_t k = 0; k < hh * ww; ++k) {
          const float d = dp3[base + k];
          dg += static_cast<double>(d) * c.p3.y[base + k];
          db += d;
          dym[base + k] = d * (1.0f + gamma);
        }
        d_film[i * 2 * channels + c0] = static_cast<float>(dg);
        d_film[i * 2 * channels + channels + c0] = static_cast<float>(db);
      }
    }
    g.film = linear_backward_cuda(m.film, c.text_feat, d_film);
    g.text_demb = Tensor({kTextVocab, m.text_dim});
    for (std::size_t i = 0; i < n; ++i) {
      for (int l = 0; l < kTextLen; ++l) {
        const std::uint8_t id = c.text_ids[i * kTextLen + l];
        if (id == 0 || id >= kTextVocab) continue;
        for (int d = 0; d < m.text_dim; ++d) {
          g.text_demb[id * m.text_dim + d] +=
              g.film.dx[i * tfeat + static_cast<std::size_t>(l) * m.text_dim + d];
        }
      }
    }
    dp3 = std::move(dym);
  }
  // FC 的 dx（或 FiLM 调制后的梯度）还原成 [N,64,h',w'] 再散回 pool3 的 argmax。
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
