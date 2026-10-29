// 训练器 CLI：
//   uavrgb_train --data <dataset.uavr> --epochs N --batch B --lr L --out <run_dir>
//                [--resume <ckpt>] [--seed S] [--max-grad-norm G]
//                [--class-weights auto|<w0,w1,w2,w3>] [--drop-collision]
// 落盘：<run_dir>/results.csv、metrics.jsonl、config.json、ckpt.bin（每 epoch 覆盖）。
#include <cmath>
#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

#include "uavrgb/run_owner.hpp"
#include "uavrgb/trainer.hpp"

namespace {

void usage() {
  std::fprintf(stderr,
               "用法: uavrgb_train --data <dataset.uavr> --epochs N --batch B --lr L "
               "--out <run_dir> [--resume <ckpt>] [--seed S] [--max-grad-norm G] "
               "[--class-weights auto|<w0,w1,w2,w3>] [--drop-collision]\n");
}

}  // namespace

int main(int argc, char** argv) {
  uavrgb::TrainConfig cfg;
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
      } else if (a == "--lr") {
        cfg.lr = std::stof(next("--lr"));
      } else if (a == "--out") {
        cfg.out_dir = next("--out");
      } else if (a == "--resume") {
        cfg.resume = next("--resume");
      } else if (a == "--seed") {
        cfg.seed = static_cast<unsigned>(std::stoul(next("--seed")));
      } else if (a == "--max-grad-norm") {
        cfg.max_grad_norm = std::stof(next("--max-grad-norm"));
      } else if (a == "--class-weights") {
        cfg.class_weights = next("--class-weights");
      } else if (a == "--drop-collision") {
        cfg.drop_collision = true;
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
  // max_grad_norm=0 会把所有梯度按 0 缩放，训练照跑但模型完全不动，必须挡在入口。
  if (!(cfg.max_grad_norm > 0.0f)) {
    std::fprintf(stderr, "--max-grad-norm 必须 > 0（当前 %g 会把梯度清零）\n",
                 static_cast<double>(cfg.max_grad_norm));
    return 2;
  }
  // lr<=0 时 Adam 会原地不动（0）或做梯度上升（负），跑完全部 epoch 还报成功。
  // std::stof 也接受 "nan"，它会让参数在第 1 步后变成 NaN。
  if (!(std::isfinite(cfg.lr) && cfg.lr > 0.0f)) {
    std::fprintf(stderr, "--lr 必须是有限正数（当前 %g）\n", static_cast<double>(cfg.lr));
    return 2;
  }

  // 单写者锁 + 拒绝复用非空目录：否则两轮训练会往同一个 results.csv 追加、
  // 互相覆盖 ckpt.bin，产物无法区分。--resume 明确表示续跑同一目录时才放行。
  // 这是用法错误不是训练失败，所以单独 catch 后返回 2（与 --lr 等预检一致）；
  // 用 optional 是因为 RunOwner 必须活到 main 结束才持有锁，不能在 try 里构造。
  std::optional<uavrgb::RunOwner> owner;
  try {
    owner.emplace(cfg.out_dir, !cfg.resume.empty());
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
