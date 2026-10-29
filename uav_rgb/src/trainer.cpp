// NativeTrainer 实现：分 batch → forward → softmax-CE → backward → 裁剪 → Adam → 指标。
#include "uavrgb/trainer.hpp"

#include "trainer_internal.hpp"
#include "uavrgb/class_weights.hpp"
#include "uavrgb/linear_cuda.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/softmax_cuda.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb {
namespace {

using Clock = std::chrono::steady_clock;

double since(const Clock::time_point& t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

// 非有限值（nan/inf）不能写进 JSON，统一退化成 null；训练循环另有守卫会直接中止。
std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", v);
  return b;
}

std::uint64_t count_correct(const Tensor& logits, const std::vector<int>& labels) {
  const int n = static_cast<int>(labels.size());
  const int c = static_cast<int>(logits.dims()[1]);
  std::uint64_t ok = 0;
  for (int i = 0; i < n; ++i) {
    int best = 0;
    for (int j = 1; j < c; ++j) {
      if (logits[static_cast<std::size_t>(i) * c + j] >
          logits[static_cast<std::size_t>(i) * c + best]) {
        best = j;
      }
    }
    ok += (best == labels[static_cast<std::size_t>(i)]) ? 1 : 0;
  }
  return ok;
}

}  // namespace

void Trainer::write_csv_header_if_needed() const {
  const std::string p = cfg_.out_dir + "/results.csv";
  std::error_code ec;
  const bool empty = !std::filesystem::exists(p, ec) || std::filesystem::file_size(p, ec) == 0;
  if (!empty) return;  // 续训直接追加，不重复写表头
  std::ofstream f(p, std::ios::app);
  if (!f) throw std::runtime_error("trainer: 无法写 " + p);
  f << kResultsCsvHeader << '\n';
}

void Trainer::run() {
  if (cfg_.epochs <= 0 || cfg_.batch <= 0) {
    throw std::invalid_argument("trainer: epochs/batch 必须为正");
  }
  UavrDataset ds = load_uavr(cfg_.data_path, LoadOptions{cfg_.drop_collision});
  if (ds.train_idx.empty()) throw std::runtime_error("trainer: 数据集没有 train split");
  std::filesystem::create_directories(cfg_.out_dir);

  class_weights_ = resolve_class_weights(cfg_.class_weights, ds);
  class_weights_source_ = class_weights_source(cfg_.class_weights);

  model_ = make_cnn(ds.h, ds.w, cfg_.seed, static_cast<int>(ds.num_classes));
  state_ = make_train_state(model_, cfg_.lr);
  if (!cfg_.resume.empty()) {
    load_checkpoint(cfg_.resume, model_, state_);
    class_weights_source_ = merge_resume_weights(cfg_.class_weights, class_weights_,
                                                 state_.class_weights);
  }
  state_.class_weights = class_weights_;
  write_run_config(cfg_.out_dir, cfg_, class_weights_, class_weights_source_);

  std::vector<std::uint32_t> dev = ds.val_idx;
  const bool dev_is_train = dev.empty();
  if (dev_is_train) dev = ds.train_idx;

  const std::string metrics_path = cfg_.out_dir + "/metrics.jsonl";
  const std::string csv_path = cfg_.out_dir + "/results.csv";
  const std::string ckpt_path = cfg_.out_dir + "/ckpt.bin";
  const std::string best_ckpt_path = cfg_.out_dir + "/ckpt_best.bin";
  write_csv_header_if_needed();
  std::ofstream metrics(metrics_path, std::ios::app);
  std::ofstream csv(csv_path, std::ios::app);
  if (!metrics || !csv) throw std::runtime_error("trainer: 无法追加写 " + cfg_.out_dir);

  const std::size_t bs = static_cast<std::size_t>(cfg_.batch);
  const std::uint32_t per_epoch =
      static_cast<std::uint32_t>((ds.train_idx.size() + bs - 1) / bs);
  std::uint32_t step = static_cast<std::uint32_t>(train_step(state_));
  const std::uint32_t total_steps = step + per_epoch * static_cast<std::uint32_t>(cfg_.epochs);

  metrics << metrics_json(make_status(
                 "resume from step " + std::to_string(step) + ", " + std::to_string(ds.n) +
                 " samples (" + std::to_string(ds.train_idx.size()) + " train / " +
                 std::to_string(dev.size()) + (dev_is_train ? " dev=fallback-train)" : " dev)") +
                 ", params=" + std::to_string(num_params(model_))))
          << '\n';
  if (dev_is_train) {
    metrics << metrics_json(make_error(Severity::kWarn, "DEV_SPLIT_EMPTY", step,
                                       "数据集没有 val split，dev 指标回退到 train 集"))
            << '\n';
  }
  metrics.flush();

  std::mt19937 rng(cfg_.seed);
  std::vector<std::uint32_t> order = ds.train_idx;
  const Clock::time_point t0 = Clock::now();
  const int epoch_base = state_.epoch;  // 续训时 results.csv 接着编号
  const std::uint32_t start_step = step;  // eta 只按本次运行的步数算
  Tensor x;
  samples_seen_ = 0;

  for (int ep = 1; ep <= cfg_.epochs; ++ep) {
    std::shuffle(order.begin(), order.end(), rng);
    double ce_sum = 0.0;
    std::uint64_t seen = 0;
    for (std::size_t s = 0; s < order.size(); s += bs) {
      const std::size_t e = std::min(order.size(), s + bs);
      const std::vector<std::uint32_t> part = detail::slice(order, s, e);
      const std::vector<int> y = detail::take_labels(ds, order, s, e);
      ds.load_images(part, x);

      CnnCache cache;
      const Tensor logits = forward(model_, x, &cache);
      const float loss = class_weighted_loss(logits, y, class_weights_);
      if (!std::isfinite(loss)) {
        metrics << metrics_json(make_error(Severity::kFatal, "TRAIN_NONFINITE_METRIC", step + 1,
                                           "训练 loss 非有限值，中止"))
                << '\n';
        metrics.flush();
        throw std::runtime_error("trainer: loss 非有限值（step " + std::to_string(step + 1) + "）");
      }
      const float acc = static_cast<float>(count_correct(logits, y)) /
                        static_cast<float>(part.size());
      const Tensor dlogits = class_weighted_grad(logits, y, class_weights_);
      CnnGrad g = backward(model_, cache, dlogits);

      std::vector<Tensor*> gts = cnn_grad_tensors(g);
      clip_grad_norm_cuda(gts, cfg_.max_grad_norm);
      std::vector<Tensor*> ps = cnn_params(model_);
      for (std::size_t i = 0; i < ps.size(); ++i) adam_step_cuda(state_.adam[i], *ps[i], *gts[i]);

      ++step;
      const double elapsed = since(t0);
      const double done = static_cast<double>(step - start_step);
      const double eta = elapsed / done * static_cast<double>(total_steps - step);
      metrics << metrics_json(make_train_step(step, total_steps, loss, acc, cfg_.lr, elapsed, eta))
              << '\n';
      metrics.flush();  // 监控台按追加偏移增量读，每步必须落盘

      // results.csv 的 train_ce 必须与 dev_ce 同口径（都是无权 CE），否则两列不可比。
      const float plain = class_weights_.empty() ? loss : softmax_ce_loss_cuda(logits, y);
      ce_sum += static_cast<double>(plain) * static_cast<double>(part.size());
      seen += part.size();
    }

    const double train_ce = ce_sum / static_cast<double>(seen);
    EvalStats dev_stats;
    evaluate(ds, dev, &dev_stats);
    state_.epoch = epoch_base + ep;
    csv << state_.epoch << ',' << num(train_ce) << ',' << num(dev_stats.ce) << ','
        << num(dev_stats.acc) << '\n';
    csv.flush();
    MetricsEvent vev = make_validation(step, static_cast<float>(dev_stats.ce),
                                       static_cast<float>(dev_stats.acc),
                                       dev_is_train ? Split::kTrain : Split::kVal);
    vev.macro_f1 = static_cast<float>(dev_stats.macro_f1);
    vev.balanced_accuracy = static_cast<float>(dev_stats.balanced_acc);
    vev.confusion.assign(dev_stats.confusion.begin(), dev_stats.confusion.end());
    metrics << metrics_json(vev) << '\n';
    // 类别不平衡下要看宏指标：原始 accuracy 被 75% 的 FORWARD 撑着。
    std::printf("epoch %d %s ce=%.4f acc=%.4f macro_f1=%.4f bal_acc=%.4f recall=[", ep,
                dev_is_train ? "train" : "dev  ", dev_stats.ce, dev_stats.acc,
                dev_stats.macro_f1, dev_stats.balanced_acc);
    for (std::size_t k = 0; k < dev_stats.recall.size(); ++k) {
      std::printf("%s%.3f", k ? " " : "", dev_stats.recall[k]);
    }
    std::printf("]\n");
    if (!class_weights_.empty()) {
      metrics << metrics_json(make_class_weights(static_cast<std::uint32_t>(state_.epoch), step,
                                                 class_weights_, class_weights_source_))
              << '\n';
    }
    metrics.flush();
    // 交付模型不能是最后一轮：dev 宏 F1 峰值常早于末轮（见 docs/results.md）。
    // dev 回退到 train 时不选模型，否则等于用训练集挑权重。
    if (!dev_is_train && dev_stats.macro_f1 > static_cast<double>(state_.best_macro_f1)) {
      state_.best_macro_f1 = static_cast<float>(dev_stats.macro_f1);
      save_checkpoint(best_ckpt_path, model_, state_);
      metrics << metrics_json(make_status("best checkpoint: epoch " +
                                          std::to_string(state_.epoch) + " macro_f1=" +
                                          num(dev_stats.macro_f1)))
              << '\n';
      metrics.flush();
    }
    // 必须在更新 best 之后写：否则续训从 ckpt.bin 读到的 best 比 ckpt_best.bin 落后一轮。
    save_checkpoint(ckpt_path, model_, state_);
    samples_seen_ += seen;
  }

  // test split 全程不参与训练和调参：只在最后评估一次，给出一个没被选择污染的数字。
  if (!ds.test_idx.empty()) {
    EvalStats test_stats;
    evaluate(ds, ds.test_idx, &test_stats);
    test_acc_ = test_stats.acc;
    MetricsEvent tev = make_validation(step, static_cast<float>(test_stats.ce),
                                       static_cast<float>(test_stats.acc), Split::kTest);
    tev.macro_f1 = static_cast<float>(test_stats.macro_f1);
    tev.balanced_accuracy = static_cast<float>(test_stats.balanced_acc);
    tev.confusion.assign(test_stats.confusion.begin(), test_stats.confusion.end());
    metrics << metrics_json(tev) << '\n';
  }

  elapsed_sec_ = since(t0);
  metrics << metrics_json(make_status("done: " + std::to_string(samples_seen_) + " samples in " +
                                      num(elapsed_sec_) + " s, ckpt=" + ckpt_path))
          << '\n';
  metrics.flush();
}

}  // namespace uavrgb
