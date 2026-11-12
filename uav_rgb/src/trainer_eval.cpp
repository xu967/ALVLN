// Trainer::evaluate 的独立实现：从 trainer.cpp 拆出，给 best-checkpoint 留出行数。
#include "uavrgb/trainer.hpp"

#include "trainer_internal.hpp"
#include "uavrgb/eval_metrics.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/run_owner.hpp"
#include "uavrgb/softmax_cuda.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
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
  double sum = 0.0, depth_sum = 0.0;
  std::uint64_t seen = 0, depth_valid = 0, depth_correct = 0;
  const bool want_depth = model_.aux_classes > 0 && ds.aux.has_depth;
  for (std::size_t s = 0; s < idx.size(); s += b) {
    const std::size_t e = std::min(idx.size(), s + b);
    const std::vector<std::uint32_t> part = detail::slice(idx, s, e);
    const std::vector<int> y = detail::take_labels(ds, idx, s, e);
    const Tensor logits = forward_batch(ds, part);
    // dev_ce 合同为无权 Teacher CE，不能随训练权重或 batch 划分改变。
    sum += static_cast<double>(softmax_ce_loss_cuda(logits, y)) *
           static_cast<double>(part.size());
    if (want_depth) {
      const DeviceCnn::AuxStats a = aux_batch(ds, /*lambda=*/0.0f);  // 只评估，不反向
      depth_sum += static_cast<double>(a.loss) * static_cast<double>(a.valid);
      depth_valid += a.valid;
      depth_correct += a.correct;
    }
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
  if (want_depth && depth_valid > 0) {
    out->depth_ce = depth_sum / static_cast<double>(depth_valid);
    out->depth_acc = static_cast<double>(depth_correct) / static_cast<double>(depth_valid);
  }
}

void Trainer::eval_only(const std::string& ckpt_path, bool eval_val, bool eval_test) {
  if (!eval_val && !eval_test) {
    throw std::invalid_argument("eval_only: 至少要评估 val 或 test 之一");
  }
  if (cfg_.batch <= 0) throw std::invalid_argument("eval_only: batch 必须为正");
  // 不读 train split 也可以评估；drop_collision 与训练期保持一致，否则同权重同数据
  // 的指标对不上（碰撞步在两边的去留不同）。
  LoadOptions load_opts;
  load_opts.drop_collision = cfg_.drop_collision;
  // test 评估要留痕数据身份：这里顺手算一次 SHA 填回 cfg_，train_main 读 config() 复用，
  // 不再对同一文件再读一遍。
  load_opts.want_file_sha256 = eval_test;
  UavrDataset ds = load_uavr(cfg_.data_path, load_opts);
  if (eval_test) cfg_.dataset_sha256 = sha256_hex(ds.file_sha256);
  std::filesystem::create_directories(cfg_.out_dir);
  // 模型结构以 checkpoint 为准；--text-dim / --two-frame 等只影响训练，不覆盖 checkpoint 记录。
  const CheckpointMeta meta = read_checkpoint_meta(ckpt_path);
  cfg_.text_dim = meta.text_dim;
  cfg_.two_frame = meta.in_c == 2 * ds.c;
  cfg_.prev_action = meta.prev_dim > 0;
  cfg_.depth_aux = meta.aux_classes > 0;
  if (cfg_.text_dim > 0 && ds.text_ids.empty()) {
    throw std::runtime_error("eval_only: checkpoint 有文本分支，但数据集没有 instructions 侧车");
  }
  if ((cfg_.two_frame || cfg_.prev_action) && !ds.aux.present) {
    throw std::runtime_error("eval_only: checkpoint 是双帧/prev-action 模型，但数据集没有 dataset.aux 侧车");
  }
  if (cfg_.depth_aux && !ds.aux.has_depth) {
    std::fprintf(stderr, "eval_only: 数据集没有深度目标，只评估动作头（depth_ce/depth_acc 记 -1）\n");
  }
  model_ = make_cnn(ds.h, ds.w, cfg_.seed, static_cast<int>(ds.num_classes), cfg_.text_dim,
                    meta.in_c, meta.aux_classes, meta.prev_dim);
  state_ = make_train_state(model_, cfg_.lr);
  load_checkpoint(ckpt_path, model_, state_);
  dev_ = std::make_unique<DeviceCnn>(model_, cfg_.batch);

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
    std::printf("] n=%zu", idx.size());
    if (s.depth_ce >= 0.0) std::printf(" depth_ce=%.4f depth_acc=%.3f", s.depth_ce, s.depth_acc);
    std::printf("\n");
  };
  if (eval_val) emit(ds.val_idx, Split::kVal, "val");
  if (eval_test) emit(ds.test_idx, Split::kTest, "test");
}

int run_eval_only_cli(TrainConfig& cfg, const std::string& eval_ckpt,
                      const std::string& eval_split) {
  // --out 是别的已完成 run 目录（有 config.json，而 ckpt 不在其中）时拒绝：
  // 否则另一模型的 validation 事件会被追加进那个 run 的 metrics.jsonl，事后分不出来。
  if (std::filesystem::exists(std::filesystem::path(cfg.out_dir) / "config.json") &&
      !same_run_dir(cfg.out_dir, eval_ckpt)) {
    std::fprintf(stderr,
                 "eval-only: %s 已是另一个 run 的目录（有 config.json），而 %s 不在其中；"
                 "请 --out 指向 ckpt 自己的 run 目录或一个新目录\n",
                 cfg.out_dir.c_str(), eval_ckpt.c_str());
    return 2;
  }
  // 输出目录通常就是训练 run-dir（非空），允许在已有产物上追加评估事件；
  // 仍用 RunOwner 防止与正在写同一目录的训练并发。
  std::optional<RunOwner> eval_owner;
  try {
    eval_owner.emplace(cfg.out_dir, true);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  try {
    // 碰撞过滤是数据身份的一部分：优先读同目录 config.json 与 checkpoint v5 训练契约；
    // 两处都无记录或互相矛盾时直接拒绝，不再静默回落到命令行缺省 false（C6）。
    // 放在 try 里：config.json / ckpt 损坏时抛异常走 rc=1，不能 abort。
    const std::filesystem::path ckpt_dir =
        std::filesystem::path(eval_ckpt).parent_path().empty()
            ? std::filesystem::path(".")
            : std::filesystem::path(eval_ckpt).parent_path();
    const CheckpointMeta ck_meta = read_checkpoint_meta(eval_ckpt);
    RunConfigSnapshot snap;
    const bool has_cfg = read_run_config(ckpt_dir.string(), &snap) && snap.has_drop_collision;
    const bool has_ck = ck_meta.has_drop_collision;
    if (has_cfg && has_ck && snap.drop_collision != ck_meta.drop_collision) {
      std::fprintf(stderr,
                   "eval-only: %s/config.json 的 drop_collision(%s) 与 checkpoint 契约(%s) 矛盾\n",
                   ckpt_dir.string().c_str(), snap.drop_collision ? "true" : "false",
                   ck_meta.drop_collision ? "true" : "false");
      return 2;
    }
    if (!has_cfg && !has_ck) {
      std::fprintf(stderr,
                   "eval-only: %s 无含 drop_collision 的 config.json，且 checkpoint 无训练契约，"
                   "拒绝静默回落\n",
                   ckpt_dir.string().c_str());
      return 2;
    }
    const bool trained_drop = has_cfg ? snap.drop_collision : ck_meta.drop_collision;
    if (cfg.drop_collision_set && cfg.drop_collision != trained_drop) {
      std::fprintf(stderr,
                   "eval-only: --drop-collision 与训练记录不一致（训练 %s）；"
                   "碰撞步去留不同会让指标不可比\n",
                   trained_drop ? "true" : "false");
      return 2;
    }
    cfg.drop_collision = trained_drop;
    Trainer tr(cfg);
    tr.eval_only(eval_ckpt, eval_split != "test", eval_split != "val");
    // test 评估单独留痕：查询次数/日期/数据身份，不改历史 metrics。
    // 数据集 SHA 由 eval_only 加载时算好（config().dataset_sha256），这里不再读一遍。
    if (eval_split != "val") {
      append_test_eval_record(cfg.out_dir, "eval_only", eval_ckpt, cfg.data_path,
                              tr.config().dataset_sha256, cfg.drop_collision, 0);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "评估失败: %s\n", e.what());
    return 1;
  }
  return 0;
}

}  // namespace uavrgb
