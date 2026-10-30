// T4 回归：eval-only 用训练保存的 ckpt.bin 重评 val/test，必须与训练期同一权重、
// 同一数据算出的 validation 行逐字节一致；且不得改写 results.csv。
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
int failures = 0;
void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++failures;
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
std::vector<std::string> validations(const std::vector<std::string>& lines) {
  std::vector<std::string> out;
  for (const auto& l : lines) {
    if (l.find("\"event\":\"validation\"") != std::string::npos) out.push_back(l);
  }
  return out;
}
}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_eval_only_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);
  // 16 train / 4 val / 4 test；训练 1 epoch，val 和 test 各有一条 validation 事件。
  const auto sd =
      uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", 24, 120, 160, 16, 9, 4);

  TrainConfig cfg;
  cfg.data_path = sd.path;
  cfg.out_dir = root + "/run";
  cfg.epochs = 1;
  cfg.batch = 8;
  cfg.lr = 1e-3f;
  cfg.seed = 3;

  Trainer tr(cfg);
  tr.run();
  const std::string metrics_path = cfg.out_dir + "/metrics.jsonl";
  const std::string csv_path = cfg.out_dir + "/results.csv";
  const auto trained = validations(read_lines(metrics_path));
  check(trained.size() == 2, "训练期应有 val+test 两条 validation");
  check(read_lines(csv_path).size() == 2, "results.csv 应为表头 + 1 个 epoch");

  Trainer ev(cfg);
  ev.eval_only(cfg.out_dir + "/ckpt.bin", /*eval_val=*/true, /*eval_test=*/true);
  const auto all = validations(read_lines(metrics_path));
  check(all.size() == 4, "eval-only 应追加 val+test 两条 validation");
  if (all.size() == 4) {
    check(all[2] == all[0], "eval-only val 与训练期同权重结果不一致");
    check(all[3] == all[1], "eval-only test 与训练期同权重结果不一致");
    check(all[2].find("\"split\":\"val\"") != std::string::npos, "val 事件 split 字段错");
    check(all[3].find("\"split\":\"test\"") != std::string::npos, "test 事件 split 字段错");
    check(all[2].find("\"macro_f1\"") != std::string::npos, "val 事件缺 macro_f1");
    check(all[2].find("\"balanced_accuracy\"") != std::string::npos,
          "val 事件缺 balanced_accuracy");
    check(all[2].find("\"confusion\"") != std::string::npos, "val 事件缺 confusion");
  }
  check(read_lines(csv_path).size() == 2, "eval-only 不得改 results.csv");

  fs::remove_all(root);
  if (failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << failures << " 项)\n";
  return 1;
}
