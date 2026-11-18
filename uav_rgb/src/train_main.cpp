// 训练器 CLI（表驱动参数解析，拒绝长串 if/else if 堆砌）：
//   uavrgb_train --data <dataset.uavr> --epochs N --batch B --lr L --out <run_dir>
//                [--resume <ckpt>] [--seed S] [--max-grad-norm G]
//                [--class-weights auto|<w0,w1,w2,w3>] [--drop-collision] [--save-every-epoch]
//   uavrgb_train --data <dataset.uavr> --eval-only <ckpt> --out <run_dir>
//                [--batch B] [--split val|test|all] [--drop-collision]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <functional>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "uavrgb/run_owner.hpp"
#include "uavrgb/trainer.hpp"

namespace {

void usage() {
  std::fprintf(stderr,
               "用法: uavrgb_train --data <dataset.uavr> --epochs N --batch B --lr L "
               "--out <run_dir> [--resume <ckpt>] [--seed S] [--max-grad-norm G] "
               "[--class-weights auto|<w0,w1,w2,w3>] [--drop-collision] "
               "[--text-dim N | --vision-only] [--soft-targets <bin> [--distill-alpha A]] "
               "[--allow-resume-mismatch] [--strict-distill] "
               "[--augment all|photo,shift,flip] [--warmup N] [--cosine-lr] [--weight-decay W] [--label-smoothing E]\n"
               "       [--two-frame] [--prev-action | --vln-nav] [--stop-head [--stop-lambda L] [--stop-pos-weight W]]\n"
               "       [--depth-aux [--aux-lambda L]]  （需要 dataset.aux 侧车）\n"
               "       [--save-every-epoch] [--log-grad-norm] [--cuda-blocking-sync]\n"
               "  或: uavrgb_train --data <dataset.uavr> --eval-only <ckpt> --out "
               "<run_dir> [--batch B] [--split val|test|all] [--drop-collision]\n");
}

unsigned parse_seed_u32(const std::string& s) {
  std::size_t used = 0;
  const bool digits = !s.empty() && s[0] >= '0' && s[0] <= '9';
  const unsigned long long v = digits ? std::stoull(s, &used) : 0ull;
  if (!digits || used != s.size() || v > 4294967295ull) {
    throw std::invalid_argument("--seed 必须是 0..4294967295 的整数: " + s);
  }
  return static_cast<unsigned>(v);
}

int parse_nonneg_int(const std::string& s, const char* msg) {
  const int v = std::stoi(s);
  if (v < 0) throw std::invalid_argument(msg);
  return v;
}

float parse_nonneg_finite(const std::string& s, const char* msg) {
  const float v = std::stof(s);
  if (!(v >= 0.0f) || !std::isfinite(v)) throw std::invalid_argument(msg);
  return v;
}

std::string parse_augment_spec(const std::string& s) {
  std::stringstream ss(s);
  for (std::string tok; std::getline(ss, tok, ',');) {
    if (tok != "all" && tok != "photo" && tok != "shift" && tok != "flip") {
      throw std::invalid_argument("--augment 只接受 all 或 photo/shift/flip 的逗号组合: " + tok);
    }
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  uavrgb::TrainConfig cfg;
  std::string eval_ckpt;
  std::string eval_split = "all";
  bool cuda_blocking_sync = false;

  const std::unordered_map<std::string, std::function<void()>> flag_opts = {
      {"--drop-collision", [&] { cfg.drop_collision = cfg.drop_collision_set = true; }},
      {"--vision-only", [&] { cfg.vision_only = true; cfg.text_dim = 0; cfg.text_dim_set = true; }},
      {"--allow-resume-mismatch", [&] { cfg.allow_resume_mismatch = true; }},
      {"--strict-distill", [&] { cfg.strict_distill = true; }},
      {"--cosine-lr", [&] { cfg.cosine_lr = cfg.cosine_lr_set = true; }},
      {"--two-frame", [&] { cfg.two_frame = cfg.two_frame_set = true; }},
      {"--prev-action", [&] { cfg.prev_action = cfg.prev_action_set = true; }},
      {"--vln-nav", [&] { cfg.vln_nav = cfg.vln_nav_set = true; }},
      {"--stop-head", [&] { cfg.stop_head = cfg.stop_head_set = true; }},
      {"--depth-aux", [&] { cfg.depth_aux = cfg.depth_aux_set = true; }},
      {"--save-every-epoch", [&] { cfg.save_every_epoch = true; }},
      {"--log-grad-norm", [&] { cfg.log_grad_norm = true; }},
      {"--cuda-blocking-sync", [&] { cuda_blocking_sync = true; }},
  };

  const std::unordered_map<std::string, std::function<void(const std::string&)>> val_opts = {
      {"--data", [&](const std::string& v) { cfg.data_path = v; }},
      {"--out", [&](const std::string& v) { cfg.out_dir = v; }},
      {"--resume", [&](const std::string& v) { cfg.resume = v; }},
      {"--epochs", [&](const std::string& v) { cfg.epochs = std::stoi(v); }},
      {"--batch", [&](const std::string& v) { cfg.batch = std::stoi(v); cfg.batch_set = true; }},
      {"--lr", [&](const std::string& v) { cfg.lr = std::stof(v); cfg.lr_set = true; }},
      {"--seed", [&](const std::string& v) { cfg.seed = parse_seed_u32(v); cfg.seed_set = true; }},
      {"--max-grad-norm", [&](const std::string& v) { cfg.max_grad_norm = std::stof(v); cfg.max_grad_norm_set = true; }},
      {"--class-weights", [&](const std::string& v) { cfg.class_weights = v; }},
      {"--text-dim", [&](const std::string& v) { cfg.text_dim = parse_nonneg_int(v, "--text-dim 不能为负"); cfg.text_dim_set = true; }},
      {"--soft-targets", [&](const std::string& v) { cfg.soft_targets = v; cfg.soft_targets_set = true; }},
      {"--distill-alpha", [&](const std::string& v) {
         cfg.distill_alpha = std::stof(v);
         if (!(cfg.distill_alpha >= 0.0f) || !(cfg.distill_alpha <= 1.0f)) throw std::invalid_argument("--distill-alpha 必须在 [0,1]");
         cfg.distill_alpha_set = true;
       }},
      {"--augment", [&](const std::string& v) { cfg.augment = parse_augment_spec(v); cfg.augment_set = true; }},
      {"--warmup", [&](const std::string& v) { cfg.warmup_steps = parse_nonneg_int(v, "--warmup 不能为负"); cfg.warmup_steps_set = true; }},
      {"--weight-decay", [&](const std::string& v) {
         cfg.weight_decay = std::stof(v);
         if (!(cfg.weight_decay >= 0.0f)) throw std::invalid_argument("--weight-decay 不能为负");
         cfg.weight_decay_set = true;
       }},
      {"--label-smoothing", [&](const std::string& v) {
         cfg.label_smoothing = std::stof(v);
         if (!(cfg.label_smoothing >= 0.0f) || !(cfg.label_smoothing < 1.0f)) throw std::invalid_argument("--label-smoothing 必须在 [0,1)");
         cfg.label_smoothing_set = true;
       }},
      {"--stop-lambda", [&](const std::string& v) { cfg.stop_lambda = parse_nonneg_finite(v, "--stop-lambda 必须为非负有限数"); cfg.stop_lambda_set = true; }},
      {"--stop-pos-weight", [&](const std::string& v) { cfg.stop_pos_weight = parse_nonneg_finite(v, "--stop-pos-weight 必须为非负有限数（0=自动）"); cfg.stop_pos_weight_set = true; }},
      {"--aux-lambda", [&](const std::string& v) {
         cfg.aux_lambda = std::stof(v);
         if (!(cfg.aux_lambda >= 0.0f)) throw std::invalid_argument("--aux-lambda 不能为负");
         cfg.aux_lambda_set = true;
       }},
      {"--eval-only", [&](const std::string& v) { eval_ckpt = v; }},
      {"--split", [&](const std::string& v) {
         if (v != "val" && v != "test" && v != "all") throw std::invalid_argument("--split 只能是 val/test/all");
         eval_split = v;
       }},
  };

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    try {
      if (auto fit = flag_opts.find(a); fit != flag_opts.end()) {
        fit->second();
        continue;
      }
      auto vit = val_opts.find(a);
      if (vit == val_opts.end()) {
        std::fprintf(stderr, "未知参数: %s\n", a.c_str());
        usage();
        return 2;
      }
      if (i + 1 >= argc) throw std::invalid_argument("缺参数值: " + a);
      vit->second(argv[++i]);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      usage();
      return 2;
    }
  }

  if (const char* env_bs = std::getenv("UAVRGB_CUDA_BLOCKING_SYNC")) {
    const std::string v(env_bs);
    if (v == "1") cuda_blocking_sync = true;
    if (!v.empty() && v != "0" && v != "1") {
      std::fprintf(stderr, "UAVRGB_CUDA_BLOCKING_SYNC 只接受 0 或 1（当前 '%s'）\n", env_bs);
      return 2;
    }
  }
  if (cfg.data_path.empty() || cfg.out_dir.empty()) {
    usage();
    return 2;
  }
  if (cuda_blocking_sync) {
    try {
      uavrgb::enable_cuda_blocking_sync();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 1;
    }
  }
  if (!eval_ckpt.empty()) {
    if (cfg.batch <= 0) {
      std::fprintf(stderr, "--batch 必须为正\n");
      return 2;
    }
    if (!cfg.resume.empty()) {
      std::fprintf(stderr, "--eval-only 与 --resume 不能同给（评估不续训；多半是命令拼错）\n");
      usage();
      return 2;
    }
    return uavrgb::run_eval_only_cli(cfg, eval_ckpt, eval_split);
  }
  if (!(cfg.max_grad_norm > 0.0f)) {
    std::fprintf(stderr, "--max-grad-norm 必须 > 0（当前 %g 会把梯度清零）\n", static_cast<double>(cfg.max_grad_norm));
    return 2;
  }
  if (!(std::isfinite(cfg.lr) && cfg.lr > 0.0f && cfg.lr <= 1.0f)) {
    std::fprintf(stderr, "--lr 必须是 0<lr<=1 的有限数（当前 %g）\n", static_cast<double>(cfg.lr));
    return 2;
  }

  const bool resume_same_dir = uavrgb::same_run_dir(cfg.out_dir, cfg.resume);
  std::optional<uavrgb::RunOwner> owner;
  try {
    owner.emplace(cfg.out_dir, resume_same_dir);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }

  try {
    uavrgb::Trainer tr(cfg);
    tr.run();
    const double sec = tr.last_elapsed_sec();
    std::printf("完成：%llu 样本，%.1f s，%.1f 样本/秒，输出 %s\n",
                static_cast<unsigned long long>(tr.last_samples()), sec,
                sec > 0 ? static_cast<double>(tr.last_samples()) / sec : 0.0,
                cfg.out_dir.c_str());
    const double test = tr.last_test_accuracy();
    if (test >= 0.0) std::printf("test_accuracy = %.4f（test split 未参与训练与模型选择）\n", test);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "训练失败: %s\n", e.what());
    return 1;
  }
  return 0;
}
