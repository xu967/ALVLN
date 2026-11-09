// --save-every-epoch：每轮在 run 目录多写一份 ckpt_ep{NN}.bin（格式与 ckpt.bin 相同），默认关时一个都不写。
// 选模研究要对每一轮做离线探针 / 闭环（DISTILL §2.4：ckpt_best 是噪声序列上的最大值），
// 之前 run/ 里只有 ckpt.bin（末轮覆盖）与 ckpt_best.bin，其余轮的权重拿不到。
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <string>

#include "synthetic_dataset.hpp"
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/trainer.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}

std::string ep_file(const std::string& run_dir, int ep) {
  char b[32];
  std::snprintf(b, sizeof b, "ckpt_ep%02d.bin", ep);
  return run_dir + "/" + b;
}

std::string read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

int count_epoch_files(const std::string& run_dir) {
  int n = 0;
  for (const auto& e : fs::directory_iterator(run_dir)) {
    if (e.path().filename().string().rfind("ckpt_ep", 0) == 0) ++n;
  }
  return n;
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_save_every_epoch_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);
  // 48 样本：32 train / 16 val，batch 16 → 每轮 2 步。
  const uavrgb_test::SynthDataset sd =
      uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", 48, 120, 160, 32, 3);

  TrainConfig cfg;
  cfg.data_path = sd.path;
  cfg.out_dir = root + "/run";
  cfg.epochs = 3;
  cfg.batch = 16;
  cfg.lr = 1e-3f;
  cfg.seed = 1;
  cfg.save_every_epoch = true;
  Trainer tr(cfg);
  tr.run();

  check(count_epoch_files(cfg.out_dir) == cfg.epochs,
        "开 --save-every-epoch 应恰好写 epochs 个 ckpt_ep*.bin，实际 " +
            std::to_string(count_epoch_files(cfg.out_dir)));
  for (int ep = 1; ep <= cfg.epochs; ++ep) {
    const std::string p = ep_file(cfg.out_dir, ep);
    check(fs::exists(p), "缺 " + p);
    if (!fs::exists(p)) continue;
    // 推理端的读法（read_checkpoint_meta → make_cnn → load_checkpoint）必须原样可用。
    const CheckpointMeta meta = read_checkpoint_meta(p);
    check(meta.in_h == 120 && meta.in_w == 160 && meta.in_c == 3 && meta.num_classes == 4,
          p + " 的结构元信息不符");
    CnnModel m = make_cnn(120, 160, cfg.seed, 4);
    TrainState st = make_train_state(m, cfg.lr);
    load_checkpoint(p, m, st);
    check(st.epoch == ep, p + " 记录的 epoch 应为 " + std::to_string(ep) + "，实际 " +
                              std::to_string(st.epoch));
    check(train_step(st) == 2 * ep, p + " 的 step 数应为 " + std::to_string(2 * ep));
  }
  // 末轮的逐轮文件与 ckpt.bin 是同一份权重和状态（含 best_macro_f1），字节必须相同。
  check(read_bytes(ep_file(cfg.out_dir, cfg.epochs)) == read_bytes(cfg.out_dir + "/ckpt.bin"),
        "末轮 ckpt_epNN.bin 应与 ckpt.bin 字节相同");
  check(fs::exists(cfg.out_dir + "/ckpt_best.bin"), "ckpt_best.bin 仍应写出");

  // 默认关：原有产物不变，不写任何 ckpt_ep*.bin。
  TrainConfig off = cfg;
  off.out_dir = root + "/run_off";
  off.save_every_epoch = false;
  Trainer t2(off);
  t2.run();
  check(count_epoch_files(off.out_dir) == 0,
        "默认不应写 ckpt_ep*.bin，实际 " + std::to_string(count_epoch_files(off.out_dir)));
  check(fs::exists(off.out_dir + "/ckpt.bin") && fs::exists(off.out_dir + "/ckpt_best.bin"),
        "默认关时 ckpt.bin / ckpt_best.bin 照旧");

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << " 项)\n";
  return 1;
}
