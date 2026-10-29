// 过拟合测试：16 个固定样本训练 200 步，训练准确率必须 > 95%，证明整条链路能学。
// 数据集没有 val split → 训练器把 dev 指标回退到 train 集（并发 WARN 事件），
// 所以 results.csv 的 dev_accuracy 就是训练准确率。
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
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
std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream f(path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) out.push_back(line);
  }
  return out;
}
}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_overfit_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);
  const uavrgb_test::SynthDataset sd =
      uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", 16, 120, 160, 16, 11);

  TrainConfig cfg;
  cfg.data_path = sd.path;
  cfg.out_dir = root + "/run";
  cfg.epochs = 200;  // 16 样本 / batch 16 = 每 epoch 1 步 → 200 步
  cfg.batch = 16;
  cfg.lr = 1e-3f;
  cfg.seed = 5;

  Trainer tr(cfg);
  tr.run();

  const auto rows = read_lines(cfg.out_dir + "/results.csv");
  for (const auto& line : read_lines(cfg.out_dir + "/metrics.jsonl")) {
    if (line.find("\"event\":\"validation\"") != std::string::npos) {
      check(line.find("\"split\":\"train\"") != std::string::npos,
            "回退训练集评估必须标记 split=train");
    }
  }
  check(rows.size() == static_cast<std::size_t>(cfg.epochs) + 1, "results.csv 行数不符");
  double acc = -1.0;
  if (rows.size() >= 2) {
    const std::string& last = rows.back();
    const std::size_t c1 = last.find(',');
    const std::size_t c2 = last.find(',', c1 + 1);
    const std::size_t c3 = last.find(',', c2 + 1);
    if (c1 != std::string::npos && c2 != std::string::npos && c3 != std::string::npos) {
      acc = std::stod(last.substr(c3 + 1));
    }
  }
  const double sec = tr.last_elapsed_sec();
  std::cout << "过拟合测试：200 步，最终训练准确率=" << acc << "，耗时=" << sec
            << " s（" << tr.last_samples() / sec << " 样本/秒）\n";
  check(acc > 0.95, "过拟合测试准确率未超过 95%");

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << " 项)\n";
  return 1;
}
