// Trainer 的设备常驻批前向 + 算法 A/B 开关的实现（增强参数、lr 日程、标签平滑 loss）。
// 单独成文件是因为 trainer.cpp / dataset.cpp 都已到行数上限。
#include "uavrgb/trainer.hpp"

#include "uavrgb/action.hpp"
#include "uavrgb/class_weights.hpp"
#include "uavrgb/ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb {
namespace {

// 读一批 NCHW uint8 样本；偏移公式与 UavrDataset::load_images 相同，只是不转 float。
// 长度不足即报错：截断文件不允许静默当成零填充。
void read_u8_batch(const UavrDataset& ds, const std::vector<std::uint32_t>& idx,
                   std::vector<std::uint8_t>& out) {
  const std::size_t plane = static_cast<std::size_t>(ds.c) * ds.h * ds.w;
  out.resize(idx.size() * plane);
  for (std::size_t i = 0; i < idx.size(); ++i) {
    if (idx[i] >= static_cast<std::uint32_t>(ds.n)) throw std::runtime_error("UAVR: 样本下标越界");
    ds.file.clear();
    ds.file.seekg(static_cast<std::streamoff>(ds.images_off +
                                              static_cast<std::uint64_t>(idx[i]) * plane));
    if (!ds.file) throw std::runtime_error("UAVR: seek 失败: 图像区");
    ds.file.read(reinterpret_cast<char*>(out.data() + i * plane),
                 static_cast<std::streamsize>(plane));
    if (static_cast<std::size_t>(ds.file.gcount()) != plane) {
      throw std::runtime_error("UAVR: 读取不足: 图像区");
    }
  }
}

// "--augment" 规格里是否含某组件（all 含全部）。
bool has_aug(const std::string& spec, const char* name) {
  return spec == "all" || ("," + spec + ",").find(std::string(",") + name + ",") != std::string::npos;
}

// 逐样本增强参数，幅度取保守值：gamma 0.8~1.25、对比度 0.8~1.2、亮度 ±0.1、平移 ±4px。
// 翻转只给纯视觉模型：镜像后左右偏航标签互换；带文本时"向左/向右"指令会与画面矛盾。
// 每个样本固定消耗 6 个随机数，开关组合不同时其余组件的抽样不变（便于消融对比）。
std::vector<dev::AugParams> make_aug(std::mt19937& rng, const std::string& spec, bool allow_flip,
                                     std::vector<int>& y) {
  const bool photo = has_aug(spec, "photo"), shift = has_aug(spec, "shift"),
             flip = has_aug(spec, "flip");
  if (!photo && !shift && !flip) throw std::invalid_argument("--augment 须是 all 或 photo/shift/flip 的组合");
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<dev::AugParams> a(y.size());
  for (std::size_t i = 0; i < y.size(); ++i) {
    const float r[6] = {u(rng), u(rng), u(rng), u(rng), u(rng), u(rng)};
    a[i].gamma = photo ? std::exp((2.0f * r[0] - 1.0f) * std::log(1.25f)) : 1.0f;
    a[i].contrast = photo ? 0.8f + 0.4f * r[1] : 1.0f;
    a[i].brightness = photo ? 0.2f * r[2] - 0.1f : 0.0f;
    a[i].dx = shift ? static_cast<int>(r[3] * 9.0f) - 4 : 0;  // -4..4
    a[i].dy = shift ? static_cast<int>(r[4] * 9.0f) - 4 : 0;
    a[i].flip = flip && allow_flip && r[5] < 0.5f;
    if (!a[i].flip) continue;
    if (y[i] == static_cast<int>(Action::YAW_LEFT)) {
      y[i] = static_cast<int>(Action::YAW_RIGHT);
    } else if (y[i] == static_cast<int>(Action::YAW_RIGHT)) {
      y[i] = static_cast<int>(Action::YAW_LEFT);
    }
  }
  return a;
}

}  // namespace

Tensor Trainer::forward_batch(const UavrDataset& ds, const std::vector<std::uint32_t>& part,
                              std::vector<int>* train_labels) {
  if (!dev_) throw std::logic_error("trainer: DeviceCnn 未初始化");
  read_u8_batch(ds, part, u8_batch_);
  std::vector<std::uint8_t> text;
  if (model_.text_dim > 0) ds.load_text(part, text);
  std::vector<dev::AugParams> aug;
  if (train_labels && !cfg_.augment.empty()) {
    if (train_labels->size() != part.size()) throw std::logic_error("trainer: 标签数与批大小不符");
    // 翻转后教师分布的左右也该互换，软标签按样本下标 gather 做不到，故不允许同开。
    if (!cfg_.soft_targets.empty()) throw std::invalid_argument("trainer: --augment 与 --soft-targets 不能同时开");
    aug = make_aug(aug_rng_, cfg_.augment, /*allow_flip=*/model_.text_dim == 0, *train_labels);
  }
  return dev_->forward(u8_batch_.data(), static_cast<int>(part.size()),
                       model_.text_dim > 0 ? text.data() : nullptr,
                       aug.empty() ? nullptr : aug.data());
}

float Trainer::step_lr(std::uint32_t step, std::uint32_t total_steps) {
  const std::uint32_t warm = static_cast<std::uint32_t>(std::max(0, cfg_.warmup_steps));
  float lr = cfg_.lr;
  if (warm > 0 && step <= warm) {
    lr = cfg_.lr * static_cast<float>(step) / static_cast<float>(warm);
  } else if (cfg_.cosine_lr && total_steps > warm) {
    const double p = std::min(1.0, static_cast<double>(step - warm) /
                                       static_cast<double>(total_steps - warm));
    lr = static_cast<float>(cfg_.lr * 0.5 * (1.0 + std::cos(M_PI * p)));
  }
  for (Adam& a : state_.adam) a.lr = lr;
  return lr;
}

Tensor Trainer::loss_and_grad(const Tensor& logits, const std::vector<int>& y,
                              const std::vector<float>& q, float* loss) {
  if (!q.empty()) {
    *loss = softmax_ce_loss_soft(logits, y, q, cfg_.distill_alpha);
    return softmax_ce_grad_soft(logits, y, q, cfg_.distill_alpha);
  }
  if (cfg_.label_smoothing <= 0.0f) {
    *loss = class_weighted_loss(logits, y, class_weights_);
    return class_weighted_grad(logits, y, class_weights_);
  }
  Tensor g;
  *loss = smoothed_weighted_loss(logits, y, class_weights_, cfg_.label_smoothing, &g);
  return g;
}

// 只在 host 算：logits 只有 N×C 个数，不值得再写一个 kernel。
float smoothed_weighted_loss(const Tensor& logits, const std::vector<int>& labels,
                             const std::vector<float>& weights, float eps, Tensor* grad) {
  if (logits.ndim() != 2 || static_cast<std::size_t>(logits.dims()[0]) != labels.size()) {
    throw std::invalid_argument("smoothed_weighted_loss: logits 必须是 [N, C] 且与标签数一致");
  }
  if (!(eps >= 0.0f) || !(eps < 1.0f)) throw std::invalid_argument("smoothed_weighted_loss: ε 须在 [0,1)");
  const std::size_t c = static_cast<std::size_t>(logits.dims()[1]);
  if (!weights.empty() && weights.size() != c) {
    throw std::invalid_argument("smoothed_weighted_loss: weights 长度须等于 C");
  }
  Tensor g(logits.dims());
  double total = 0.0, wsum = 0.0;
  for (std::size_t i = 0; i < labels.size(); ++i) {
    if (labels[i] < 0 || static_cast<std::size_t>(labels[i]) >= c) {
      throw std::invalid_argument("smoothed_weighted_loss: 标签越界");
    }
    const double wi = weights.empty() ? 1.0 : weights[static_cast<std::size_t>(labels[i])];
    wsum += wi;
    const float* row = logits.data() + i * c;
    double m = row[0];
    for (std::size_t k = 1; k < c; ++k) m = std::max(m, static_cast<double>(row[k]));
    double z = 0.0;
    for (std::size_t k = 0; k < c; ++k) z += std::exp(row[k] - m);
    const double lse = m + std::log(z);
    for (std::size_t k = 0; k < c; ++k) {
      const double t = (static_cast<int>(k) == labels[i] ? 1.0 - eps : 0.0) + eps / static_cast<double>(c);
      total -= wi * t * (row[k] - lse);
      g[i * c + k] = static_cast<float>(wi * (std::exp(row[k] - lse) - t));
    }
  }
  if (!(wsum > 0.0)) throw std::invalid_argument("smoothed_weighted_loss: 本批类别权重之和为 0");
  for (std::size_t i = 0; i < g.numel(); ++i) g[i] = static_cast<float>(g[i] / wsum);
  if (grad) *grad = std::move(g);
  return static_cast<float>(total / wsum);
}

}  // namespace uavrgb
