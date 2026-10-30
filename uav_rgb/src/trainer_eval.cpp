// Trainer::evaluate 的独立实现：从 trainer.cpp 拆出，给 best-checkpoint 留出行数。
#include "uavrgb/trainer.hpp"

#include "trainer_internal.hpp"
#include "uavrgb/eval_metrics.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/softmax_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
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

void Trainer::eval_only(const std::string& ckpt_path, bool eval_val, bool eval_test) {
  if (!eval_val && !eval_test) {
    throw std::invalid_argument("eval_only: 至少要评估 val 或 test 之一");
  }
  if (cfg_.batch <= 0) throw std::invalid_argument("eval_only: batch 必须为正");
  // 不读 train split 也可以评估；drop_collision 与训练期保持一致，否则同权重同数据
  // 的指标对不上（碰撞步在两边的去留不同）。
  UavrDataset ds = load_uavr(cfg_.data_path, LoadOptions{cfg_.drop_collision});
  std::filesystem::create_directories(cfg_.out_dir);
  model_ = make_cnn(ds.h, ds.w, cfg_.seed, static_cast<int>(ds.num_classes));
  state_ = make_train_state(model_, cfg_.lr);
  load_checkpoint(ckpt_path, model_, state_);

  const std::string metrics_path = cfg_.out_dir + "/metrics.jsonl";
  std::ofstream metrics(metrics_path, std::ios::app);
  if (!metrics) throw std::runtime_error("eval_only: 无法追加写 " + metrics_path);
  auto check_out = [](const std::ofstream& f, const char* name) {
    if (!f) throw std::runtime_error(std::string("eval_only: 写 ") + name + " 失败");
  };
  const std::uint32_t step = static_cast<std::uint32_t>(train_step(state_));
  metrics << metrics_json(make_status(
                 "eval-only: ckpt=" + ckpt_path + ", epoch=" + std::to_string(state_.epoch) +
                 ", split=" + std::string(eval_val ? "val" : "") +
                 (eval_val && eval_test ? "+" : "") + (eval_test ? "test" : "")))
          << '\n';
  metrics.flush();
  check_out(metrics, "metrics.jsonl");

  auto emit = [&](const std::vector<std::uint32_t>& idx, Split split, const char* name) {
    if (idx.empty()) {
      // 显式评估一个空 split 不是成功：调用方会把它当成「评过了」。
      metrics << metrics_json(make_error(Severity::kFatal, "EVAL_SPLIT_EMPTY", step,
                                         std::string("eval-only: ") + name +
                                         " split 没有样本"))
              << '\n';
      metrics.flush();
      check_out(metrics, "metrics.jsonl");
      throw std::runtime_error(std::string("eval_only: ") + name + " split 为空");
    }
    EvalStats s;
    evaluate(ds, idx, &s);
    if (!std::isfinite(s.ce) || !std::isfinite(s.acc) || !std::isfinite(s.macro_f1) ||
        !std::isfinite(s.balanced_acc)) {
      throw std::runtime_error(std::string("eval_only: ") + name + " 指标非有限值");
    }
    MetricsEvent ev = make_validation(step, static_cast<float>(s.ce),
                                      static_cast<float>(s.acc), split);
    ev.macro_f1 = static_cast<float>(s.macro_f1);
    ev.balanced_accuracy = static_cast<float>(s.balanced_acc);
    ev.confusion.assign(s.confusion.begin(), s.confusion.end());
    const std::string line = metrics_json(ev);
    metrics << line << '\n';
    metrics.flush();
    check_out(metrics, "metrics.jsonl");
    // 契约要求 stdout 打与 metrics.jsonl 相同的内容；先打 JSON 再打人读行。
    std::printf("%s\n", line.c_str());
    std::printf("eval-only %s: ce=%.4f acc=%.4f macro_f1=%.4f bal_acc=%.4f recall=[", name,
                s.ce, s.acc, s.macro_f1, s.balanced_acc);
    for (std::size_t k = 0; k < s.recall.size(); ++k) {
      std::printf("%s%.3f", k ? " " : "", s.recall[k]);
    }
    std::printf("] n=%zu\n", idx.size());
  };
  if (eval_val) emit(ds.val_idx, Split::kVal, "val");
  if (eval_test) emit(ds.test_idx, Split::kTest, "test");
}

}  // namespace uavrgb
