// 过拟合测试：16 个固定样本训练 200 步，训练准确率必须 > 95%，证明整条链路能学。
// 数据集没有 val split → 训练器把 dev 指标回退到 train 集（并发 WARN 事件），
// 所以 results.csv 的 dev_accuracy 就是训练准确率。
// 第二遍跑 STUDENT-P2 结构（双帧 + prev-action + 深度分带辅助头）：数据集换成类别条件合成集
// （整图亮度编码类别）+ UAVRAUX1 侧车，深度目标 = 样本标签。纯噪声图上 1x1 头没法从局部特征
// 认出样本，亮度编码才让「辅助头真的在学」可判定：动作头过 95%，深度头训练命中率过 90%。
#include <cmath>
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

// 跑一遍训练，返回 results.csv 最后一行的 dev_accuracy（= 训练准确率）。
double run_overfit(TrainConfig cfg, const char* name) {
  Trainer tr(cfg);
  tr.run();
  const auto rows = read_lines(cfg.out_dir + "/results.csv");
  for (const auto& line : read_lines(cfg.out_dir + "/metrics.jsonl")) {
    if (line.find("\"event\":\"validation\"") != std::string::npos) {
      check(line.find("\"split\":\"train\"") != std::string::npos, "回退训练集评估必须标记 split=train");
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
  std::cout << "过拟合测试[" << name << "]：200 步，最终训练准确率=" << acc << "，耗时=" << sec
            << " s（" << tr.last_samples() / sec << " 样本/秒）\n";
  check(acc > 0.95, std::string("过拟合测试[") + name + "] 准确率未超过 95%");
  return acc;
}

// metrics.jsonl 里最后一条 depth_aux 状态行的某个字段（train_ce= / train_acc=）。
double last_depth_stat(const std::string& out_dir, const std::string& field) {
  double v = -1.0;
  for (const auto& line : read_lines(out_dir + "/metrics.jsonl")) {
    const std::size_t p = line.find("depth_aux epoch ");
    if (p == std::string::npos) continue;
    const std::size_t q = line.find(field, p);
    if (q != std::string::npos) v = std::stod(line.substr(q + field.size()));
  }
  return v;
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
  run_overfit(cfg, "single");

  // P2 变体：16 样本、4 类按亮度 0.2/0.4/0.6/0.8 编码，一个 episode；侧车带深度目标（= 标签，含 255 格）。
  std::vector<int> labels_cc;
  for (int i = 0; i < 16; ++i) labels_cc.push_back(i % 4);
  const uavrgb_test::SynthDataset cc = uavrgb_test::write_class_conditional_uavr(
      root + "/cc.uavr", 120, 160, labels_cc, {}, {0.2f, 0.4f, 0.6f, 0.8f}, 0.02f, 0.05f, 13);
  uavrgb_test::write_synthetic_aux(cc.path, 16, 120, 160, {{0, 16}}, cc.labels, /*with_depth=*/true);
  TrainConfig p2 = cfg;
  p2.data_path = cc.path;
  p2.out_dir = root + "/run_p2";
  p2.two_frame = true;
  p2.prev_action = true;
  p2.depth_aux = true;
  p2.aux_lambda = 0.5f;
  run_overfit(p2, "two-frame+prev+depth");
  const double depth_ce = last_depth_stat(p2.out_dir, "train_ce=");
  const double depth_acc = last_depth_stat(p2.out_dir, "train_acc=");
  std::cout << "深度辅助头最终训练 CE=" << depth_ce << "（均匀猜测 ln8=" << std::log(8.0)
            << "），命中率=" << depth_acc << "\n";
  check(depth_acc > 0.9, "深度辅助头训练命中率应超过 90%");
  check(depth_ce >= 0.0 && depth_ce < 0.5, "深度辅助头训练 CE 应降到 0.5 以下");
  // 旧数据集语义不变：去掉侧车后 --two-frame 必须报错而不是静默单帧。
  fs::remove(aux_sidecar_path(cc.path));
  bool threw = false;
  try {
    TrainConfig bad = p2;
    bad.out_dir = root + "/run_bad";
    Trainer(bad).run();
  } catch (const std::exception& e) {
    std::cout << "无侧车 + --two-frame 被拒: " << e.what() << "\n";
    threw = true;
  }
  check(threw, "无侧车时 two_frame 必须抛异常");

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << " 项)\n";
  return 1;
}
