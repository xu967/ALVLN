// Trainer::evaluate 的独立实现：从 trainer.cpp 拆出，给 best-checkpoint 留出行数。
#include "uavrgb/trainer.hpp"

#include "trainer_internal.hpp"
#include "uavrgb/eval_metrics.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/softmax_cuda.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace uavrgb {

void Trainer::evaluate(const UavrDataset& ds, const std::vector<std::uint32_t>& idx,
                       EvalStats* out) {
  *out = EvalStats{};
  if (idx.empty()) return;
  const int c = static_cast<int>(ds.num_classes);
  Confusion conf;
  conf.reset(c);
  const std::size_t b = static_cast<std::size_t>(cfg_.batch);
  Tensor x;
  double sum = 0.0;
  std::uint64_t seen = 0;
  for (std::size_t s = 0; s < idx.size(); s += b) {
    const std::size_t e = std::min(idx.size(), s + b);
    const std::vector<std::uint32_t> part = detail::slice(idx, s, e);
    const std::vector<int> y = detail::take_labels(ds, idx, s, e);
    ds.load_images(part, x);
    const Tensor logits = forward(model_, x);
    // dev_ce 合同为无权 Teacher CE，不能随训练权重或 batch 划分改变。
    sum += static_cast<double>(softmax_ce_loss_cuda(logits, y)) *
           static_cast<double>(part.size());
    for (std::size_t i = 0; i < y.size(); ++i) {
      const float* row = logits.data() + i * static_cast<std::size_t>(c);
      int best = 0;
      for (int j = 1; j < c; ++j) {
        if (row[j] > row[best]) best = j;
      }
      conf.add(y[i], best);
    }
    seen += part.size();
  }
  std::uint64_t ok = 0;
  for (int k = 0; k < c; ++k) ok += conf.at(k, k);
  out->ce = sum / static_cast<double>(seen);
  out->acc = static_cast<double>(ok) / static_cast<double>(seen);
  out->macro_f1 = macro_f1(conf);
  out->balanced_acc = balanced_accuracy(conf);
  out->recall = per_class_recall(conf);
  out->confusion.assign(conf.cells.begin(), conf.cells.end());
}

}  // namespace uavrgb
