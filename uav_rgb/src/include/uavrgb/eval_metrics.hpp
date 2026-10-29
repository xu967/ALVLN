#pragma once
// 分类评估指标：混淆矩阵 + 宏平均 F1 + 平衡准确率。
//
// 为什么不用原始准确率：动作分布是 75% FORWARD 的不平衡集，"全预测 FORWARD"
// 就有 0.75 准确率，单看准确率分不清模型学到了没有。宏平均 F1 与平衡准确率
// 是类别不平衡下公认的口径（scikit-learn 的 f1_score(average='macro') 与
// balanced_accuracy_score 同一算法），每类等权，多数类吃不到便宜。
#include <cstddef>
#include <cstdint>
#include <vector>

namespace uavrgb {

// C×C 混淆矩阵，行优先，行 = 真值，列 = 预测。
struct Confusion {
  int classes = 0;
  std::vector<std::uint64_t> cells;

  void reset(int c) {
    classes = c;
    cells.assign(static_cast<std::size_t>(c) * static_cast<std::size_t>(c), 0);
  }
  void add(int truth, int pred) {
    ++cells[static_cast<std::size_t>(truth) * static_cast<std::size_t>(classes) +
           static_cast<std::size_t>(pred)];
  }
  std::uint64_t at(int truth, int pred) const {
    return cells[static_cast<std::size_t>(truth) * static_cast<std::size_t>(classes) +
                 static_cast<std::size_t>(pred)];
  }
};

// 宏平均 F1：每类 F1 等权平均；该类没有任何真值或没有预测命中时记 0（与 sklearn 一致）。
double macro_f1(const Confusion& c);

// 平衡准确率：每类召回率的等权平均。
double balanced_accuracy(const Confusion& c);

// 每类召回率，长度 = classes。
std::vector<double> per_class_recall(const Confusion& c);

}  // namespace uavrgb
