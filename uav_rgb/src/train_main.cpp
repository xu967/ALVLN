// 训练器 CLI：
//   uavrgb_train --data <dataset.uavr> --epochs N --batch B --lr L --out <run_dir>
//                [--resume <ckpt>] [--seed S] [--max-grad-norm G]
//                [--class-weights auto|<w0,w1,w2,w3>] [--drop-collision]
//   uavrgb_train --data <dataset.uavr> --eval-only <ckpt> --out <run_dir>
//                [--batch B] [--split val|test|all] [--drop-collision]
// 落盘：<run_dir>/results.csv、metrics.jsonl、config.json、ckpt.bin（每 epoch 覆盖）。
// eval-only 只向 metrics.jsonl 追加 validation 事件，不写 results.csv、不更新权重。
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <sstream>
#include <string>

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
               "  或: uavrgb_train --data <dataset.uavr> --eval-only <ckpt> --out "
               "<run_dir> [--batch B] [--split val|test|all] [--drop-collision]\n");
}

}  // namespace

int main(int argc, char** argv) {
  uavrgb::TrainConfig cfg;
  std::string eval_ckpt;
  std::string eval_split = "all";
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::invalid_argument(std::string("缺参数值: ") + name);
      return argv[++i];
    };
    try {
      if (a == "--data") {
        cfg.data_path = next("--data");
      } else if (a == "--epochs") {
        cfg.epochs = std::stoi(next("--epochs"));
      } else if (a == "--batch") {
        cfg.batch = std::stoi(next("--batch"));
        cfg.batch_set = true;
      } else if (a == "--lr") {
        cfg.lr = std::stof(next("--lr"));
        cfg.lr_set = true;
      } else if (a == "--out") {
        cfg.out_dir = next("--out");
      } else if (a == "--resume") {
        cfg.resume = next("--resume");
      } else if (a == "--seed") {
        cfg.seed = static_cast<unsigned>(std::stoul(next("--seed")));
        cfg.seed_set = true;
      } else if (a == "--max-grad-norm") {
        cfg.max_grad_norm = std::stof(next("--max-grad-norm"));
      } else if (a == "--class-weights") {
        cfg.class_weights = next("--class-weights");
      } else if (a == "--drop-collision") {
        cfg.drop_collision = true;
        cfg.drop_collision_set = true;
      } else if (a == "--text-dim") {
        cfg.text_dim = std::stoi(next("--text-dim"));
        if (cfg.text_dim < 0) throw std::invalid_argument("--text-dim 不能为负");
        cfg.text_dim_set = true;
      } else if (a == "--vision-only") {
        cfg.vision_only = true;
        cfg.text_dim = 0;
        cfg.text_dim_set = true;
      } else if (a == "--soft-targets") {
        cfg.soft_targets = next("--soft-targets");
        cfg.soft_targets_set = true;
      } else if (a == "--distill-alpha") {
        cfg.distill_alpha = std::stof(next("--distill-alpha"));
        if (!(cfg.distill_alpha >= 0.0f) || !(cfg.distill_alpha <= 1.0f)) {
          throw std::invalid_argument("--distill-alpha 必须在 [0,1]");
        }
        cfg.distill_alpha_set = true;
      } else if (a == "--allow-resume-mismatch") {
        cfg.allow_resume_mismatch = true;
      } else if (a == "--strict-distill") {
        cfg.strict_distill = true;
      } else if (a == "--augment") {
        cfg.augment = next("--augment");
        std::stringstream ss(cfg.augment);
        for (std::string tok; std::getline(ss, tok, ',');) {
          if (tok != "all" && tok != "photo" && tok != "shift" && tok != "flip") {
            throw std::invalid_argument("--augment 只接受 all 或 photo/shift/flip 的逗号组合: " + tok);
          }
        }
      } else if (a == "--warmup") {
        cfg.warmup_steps = std::stoi(next("--warmup"));
        if (cfg.warmup_steps < 0) throw std::invalid_argument("--warmup 不能为负");
      } else if (a == "--cosine-lr") {
        cfg.cosine_lr = true;
      } else if (a == "--weight-decay") {
        cfg.weight_decay = std::stof(next("--weight-decay"));
        if (!(cfg.weight_decay >= 0.0f)) throw std::invalid_argument("--weight-decay 不能为负");
      } else if (a == "--label-smoothing") {
        cfg.label_smoothing = std::stof(next("--label-smoothing"));
        if (!(cfg.label_smoothing >= 0.0f) || !(cfg.label_smoothing < 1.0f)) {
          throw std::invalid_argument("--label-smoothing 必须在 [0,1)");
        }
      } else if (a == "--eval-only") {
        eval_ckpt = next("--eval-only");
      } else if (a == "--split") {
        eval_split = next("--split");
        if (eval_split != "val" && eval_split != "test" && eval_split != "all") {
          throw std::invalid_argument("--split 只能是 val/test/all");
        }
      } else {
        std::fprintf(stderr, "未知参数: %s\n", a.c_str());
        usage();
        return 2;
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      usage();
      return 2;
    }
  }
  if (cfg.data_path.empty() || cfg.out_dir.empty()) {
    usage();
    return 2;
  }
  if (!eval_ckpt.empty()) {
    if (cfg.batch <= 0) {
      std::fprintf(stderr, "--batch 必须为正\n");
      return 2;
    }
    // 输出目录通常就是训练 run-dir（非空），允许在已有产物上追加评估事件；
    // 仍用 RunOwner 防止与正在写同一目录的训练并发。
    std::optional<uavrgb::RunOwner> eval_owner;
    try {
      eval_owner.emplace(cfg.out_dir, true);
    } catch (const std::exception& e) {
      std::fprintf(stderr, "%s\n", e.what());
      return 2;
    }
    // 碰撞过滤是数据身份的一部分：eval-only 默认继承训练 run 的 drop_collision，
    // 显式传入但与训练 config 不一致时拒绝（否则同权重同数据的指标对不上）。
    {
      const std::filesystem::path ckpt_dir =
          std::filesystem::path(eval_ckpt).parent_path().empty()
              ? std::filesystem::path(".")
              : std::filesystem::path(eval_ckpt).parent_path();
      uavrgb::RunConfigSnapshot snap;
      if (uavrgb::read_run_config(ckpt_dir.string(), &snap) && snap.has_drop_collision) {
        if (cfg.drop_collision_set && cfg.drop_collision != snap.drop_collision) {
          std::fprintf(stderr,
                       "eval-only: --drop-collision 与训练 config 不一致（训练 %s）；"
                       "碰撞步去留不同会让指标不可比\n",
                       snap.drop_collision ? "true" : "false");
          return 2;
        }
        cfg.drop_collision = snap.drop_collision;
      } else {
        std::fprintf(stderr,
                     "警告: %s 的 config 无 drop_collision，无法继承（按命令行值）\n",
                     ckpt_dir.string().c_str());
      }
    }
    try {
      uavrgb::Trainer tr(cfg);
      tr.eval_only(eval_ckpt, eval_split != "test", eval_split != "val");
      // test 评估单独留痕：查询次数/日期/数据身份，不改历史 metrics。
      if (eval_split != "val") {
        uavrgb::append_test_eval_record(cfg.out_dir, "eval_only", eval_ckpt, cfg.data_path,
                                        uavrgb::file_sha256_hex(cfg.data_path),
                                        cfg.drop_collision, 0);
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "评估失败: %s\n", e.what());
      return 1;
    }
    return 0;
  }
  // max_grad_norm=0 会把所有梯度按 0 缩放，训练照跑但模型完全不动，必须挡在入口。
  if (!(cfg.max_grad_norm > 0.0f)) {
    std::fprintf(stderr, "--max-grad-norm 必须 > 0（当前 %g 会把梯度清零）\n",
                 static_cast<double>(cfg.max_grad_norm));
    return 2;
  }
  // lr<=0 时 Adam 会原地不动（0）或做梯度上升（负），跑完全部 epoch 还报成功。
  // std::stof 也接受 "nan"，它会让参数在第 1 步后变成 NaN。上限 1.0：更大的 lr
  // 在 float 下会把参数推到溢出，但训练可能仍 exit 0（dev 指标检查会兜底）。
  if (!(std::isfinite(cfg.lr) && cfg.lr > 0.0f && cfg.lr <= 1.0f)) {
    std::fprintf(stderr, "--lr 必须是 0<lr<=1 的有限数（当前 %g）\n",
                 static_cast<double>(cfg.lr));
    return 2;
  }

  // 单写者锁 + 拒绝复用非空目录：否则两轮训练会往同一个 results.csv 追加、
  // 互相覆盖 ckpt.bin，产物无法区分。--resume 只有在**同一目录**续跑时才允许
  // 复用非空 out；跨目录 resume 必须写进空目录，不能拿别的 run 当覆盖目标。
  // 这是用法错误不是训练失败，所以单独 catch 后返回 2（与 --lr 等预检一致）；
  // 用 optional 是因为 RunOwner 必须活到 main 结束才持有锁，不能在 try 里构造。
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
    if (test >= 0.0) {
      std::printf("test_accuracy = %.4f（test split 未参与训练与模型选择）\n", test);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "训练失败: %s\n", e.what());
    return 1;
  }
  return 0;
}
