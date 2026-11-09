// AUDIT_SRC #6：一次训练启动只把 dataset.uavr 完整读一遍算 SHA256，config.json 的 hex、
// checkpoint 契约、软标签校验、aux 侧车绑定四处共用这一个值，不能各算各的（681 MB 的 E1 集
// 每遍约 4 s，启动开销曾比 1 轮训练还长）。
// 用 /proc/self/io 的 rchar 计进程总读字节：1 epoch 训练本身读约 1 份文件（train + dev 各一遍），
// 每多算一次 SHA 就多读 1 份。修前 4 次 SHA ≈ 5 份，修后 1 次 ≈ 2 份，阈值取 3 份。
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "soft_targets_fixture.hpp"
#include "synthetic_dataset.hpp"
#include "uavrgb/model_dev.hpp"
#include "uavrgb/trainer.hpp"

namespace fs = std::filesystem;

namespace {

std::uint64_t read_chars() {
  std::ifstream f("/proc/self/io");
  std::string key;
  std::uint64_t v = 0;
  while (f >> key >> v) {
    if (key == "rchar:") return v;
  }
  return 0;
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_sha_once_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);
  // 32 样本 120×160（train 24 / val 8）+ 无深度的 aux 侧车 + one-hot 软标签：
  // 四条会算数据集 SHA 的路径全部走到。
  const int n = 32;
  const uavrgb_test::SynthDataset sd =
      uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", n, 120, 160, 24, 5);
  uavrgb_test::write_synthetic_aux(sd.path, n, 120, 160, {{0, 24}, {24, 8}}, sd.labels,
                                   /*with_depth=*/false);
  std::vector<float> q(static_cast<std::size_t>(n) * 4, 0.0f);
  for (int i = 0; i < n; ++i) q[static_cast<std::size_t>(i) * 4 + sd.labels[i]] = 1.0f;
  uavrgb_test::write_soft(root + "/soft.bin", sd.path, q);
  const double file_bytes = static_cast<double>(fs::file_size(sd.path));

  uavrgb::TrainConfig cfg;
  cfg.data_path = sd.path;
  cfg.out_dir = root + "/run";
  cfg.epochs = 1;
  cfg.batch = 8;
  cfg.lr = 1e-3f;
  cfg.seed = 1;
  cfg.soft_targets = root + "/soft.bin";
  cfg.distill_alpha = 0.5f;

  // 先把 CUDA 上下文建起来：驱动初始化的那些 read() 不该算进数据集的账。
  { uavrgb::DeviceCnn warm(uavrgb::make_cnn(120, 160, 1, 4), 1); }
  const std::uint64_t before = read_chars();
  if (before == 0) {
    std::cout << "SKIP: /proc/self/io 不可读，无法计读字节\n";
    fs::remove_all(root);
    return 0;
  }
  uavrgb::Trainer tr(cfg);
  tr.run();
  const double passes = static_cast<double>(read_chars() - before) / file_bytes;
  std::cout << "dataset " << file_bytes << " B，启动 + 1 epoch 共读 " << passes << " 份文件\n";
  fs::remove_all(root);
  if (passes >= 3.0) {
    std::cerr << "FAIL: 数据集被完整读了约 " << passes << " 遍：SHA256 在多处重复计算\n";
    return 1;
  }
  std::cout << "test_sha_once: PASS\n";
  return 0;
}
