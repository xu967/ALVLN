// 分类评估指标实现；定义见 eval_metrics.hpp。
#include "uavrgb/eval_metrics.hpp"

namespace uavrgb {
namespace {

// F1 = 2PR/(P+R)；P+R=0 时按 0 记，不与 sklearn 的正负号约定打架。
double f1_of(double precision, double recall) {
  const double s = precision + recall;
  return s > 0.0 ? 2.0 * precision * recall / s : 0.0;
}

}  // namespace

std::vector<double> per_class_recall(const Confusion& c) {
  std::vector<double> out(static_cast<std::size_t>(c.classes), 0.0);
  for (int k = 0; k < c.classes; ++k) {
    std::uint64_t row = 0;
    for (int j = 0; j < c.classes; ++j) row += c.at(k, j);
    if (row > 0) out[static_cast<std::size_t>(k)] = static_cast<double>(c.at(k, k)) /
                                                    static_cast<double>(row);
  }
  return out;
}

double macro_f1(const Confusion& c) {
  double sum = 0.0;
  for (int k = 0; k < c.classes; ++k) {
    std::uint64_t row = 0;
    std::uint64_t col = 0;
    for (int j = 0; j < c.classes; ++j) {
      row += c.at(k, j);
      col += c.at(j, k);
    }
    const double recall = row > 0 ? static_cast<double>(c.at(k, k)) / static_cast<double>(row) : 0.0;
    const double precision =
        col > 0 ? static_cast<double>(c.at(k, k)) / static_cast<double>(col) : 0.0;
    sum += f1_of(precision, recall);
  }
  return c.classes > 0 ? sum / static_cast<double>(c.classes) : 0.0;
}

double balanced_accuracy(const Confusion& c) {
  const std::vector<double> r = per_class_recall(c);
  double sum = 0.0;
  for (double v : r) sum += v;
  return r.empty() ? 0.0 : sum / static_cast<double>(r.size());
}

}  // namespace uavrgb
