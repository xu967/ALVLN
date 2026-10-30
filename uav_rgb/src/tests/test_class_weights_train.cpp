// 类别权重端到端自检（合成不平衡数据，90% 类 0 / 10% 类 1）：
//   1) 同一份数据分别「无权重 / auto 逆频率」训练，比较验证集少数类召回率（核心证据）
//   2) auto 权重公式 = N/(C·n_c)、每 epoch class_weights 事件、config.json 落盘
//   3) 权重进 checkpoint；resume 不传参数时沿用 checkpoint 权重；传了不一致必须报错
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/class_weights.hpp"
#include "uavrgb/dataset.hpp"
#include "uavrgb/trainer.hpp"
#include "uavrgb/softmax_cuda.hpp"

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

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  std::ostringstream os;
  os << f.rdbuf();
  return os.str();
}

std::vector<int> make_labels(int n0, int n1, std::mt19937& rng) {
  std::vector<int> v;
  v.reserve(static_cast<std::size_t>(n0 + n1));
  v.insert(v.end(), static_cast<std::size_t>(n0), 0);
  v.insert(v.end(), static_cast<std::size_t>(n1), 1);
  std::shuffle(v.begin(), v.end(), rng);
  return v;
}

// 逐类命中/总数；只在 idx 上评估（不参与训练）。
struct Recall {
  double hit[4] = {};
  double total[4] = {};
};

Recall eval_recall(const CnnModel& m, UavrDataset& ds, const std::vector<std::uint32_t>& idx) {
  Recall r;
  Tensor x;
  const std::size_t bs = 32;
  for (std::size_t s = 0; s < idx.size(); s += bs) {
    const std::size_t e = std::min(idx.size(), s + bs);
    const std::vector<std::uint32_t> part(idx.begin() + static_cast<std::ptrdiff_t>(s),
                                          idx.begin() + static_cast<std::ptrdiff_t>(e));
    ds.load_images(part, x);
    const Tensor logits = forward(m, x);
    const int c = static_cast<int>(logits.dims()[1]);
    for (std::size_t i = 0; i < part.size(); ++i) {
      int best = 0;
      for (int j = 1; j < c; ++j) {
        if (logits[i * static_cast<std::size_t>(c) + static_cast<std::size_t>(j)] >
            logits[i * static_cast<std::size_t>(c) + static_cast<std::size_t>(best)]) {
          best = j;
        }
      }
      const int label = ds.labels[part[i]];
      r.total[label] += 1.0;
      if (best == label) r.hit[label] += 1.0;
    }
  }
  return r;
}

double recall_of(const Recall& r, int c) {
  return r.total[c] > 0.0 ? r.hit[c] / r.total[c] : -1.0;
}

std::size_t count_lines_with(const std::string& path, const std::string& needle) {
  std::ifstream f(path);
  std::string line;
  std::size_t n = 0;
  while (std::getline(f, line)) {
    if (line.find(needle) != std::string::npos) ++n;
  }
  return n;
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_cw_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  // 90/10 不平衡训练集（180/20）；验证集故意均衡（30/30），让召回率估计更稳。
  std::mt19937 rng(1234);
  const std::vector<int> train_labels = make_labels(180, 20, rng);
  const std::vector<int> val_labels = make_labels(30, 30, rng);
  const std::vector<float> means = {0.35f, 0.45f, 0.0f, 0.0f};  // 类 2/3 不出现
  const auto sd = uavrgb_test::write_class_conditional_uavr(
      root + "/imbalanced.uavr", 32, 32, train_labels, val_labels, means, 0.15f, 0.05f, 7);

  TrainConfig base;
  base.data_path = sd.path;
  base.epochs = 30;
  base.batch = 32;
  base.lr = 1e-3f;
  base.seed = 11;

  // ---- 1) 无权重 vs auto 权重 ----
  base.out_dir = root + "/run_plain";
  Trainer plain(base);
  plain.run();

  TrainConfig wcfg = base;
  wcfg.out_dir = root + "/run_weighted";
  wcfg.class_weights = "auto";
  Trainer weighted(wcfg);
  weighted.run();

  UavrDataset ds = load_uavr(sd.path);
  const Recall r_plain = eval_recall(plain.model(), ds, ds.val_idx);
  const Recall r_weighted = eval_recall(weighted.model(), ds, ds.val_idx);
  const double plain_minority = recall_of(r_plain, 1);
  const double weighted_minority = recall_of(r_weighted, 1);
  const double plain_majority = recall_of(r_plain, 0);
  const double weighted_majority = recall_of(r_weighted, 0);

  std::cout << "imbalanced: train 180x class0 + 20x class1 (90/10), val 30/30, epochs="
            << base.epochs << "\n";
  std::cout << "no weights : minority(class1) recall=" << plain_minority
            << " majority(class0) recall=" << plain_majority << "\n";
  std::cout << "auto weights: minority(class1) recall=" << weighted_minority
            << " majority(class0) recall=" << weighted_majority << "\n";
  check(plain_minority >= 0.0 && weighted_minority >= 0.0, "验证集两类都有样本");
  check(weighted_minority > plain_minority,
        "加权后少数类召回率应高于无权训练（实际 " + std::to_string(weighted_minority) +
            " vs " + std::to_string(plain_minority) + "）");

  // ---- 2) auto 权重公式与落盘 ----
  auto near = [](double a, double b) { return std::fabs(a - b) < 1e-5; };
  const std::vector<float> auto_w = resolve_class_weights("auto", ds);
  check(auto_w.size() == 4, "auto 权重长度 = num_classes");
  check(near(auto_w[0], 200.0 / (4.0 * 180.0)), "auto w0 = N/(C*n0)");
  check(near(auto_w[1], 200.0 / (4.0 * 20.0)), "auto w1 = N/(C*n1)");
  check(near(auto_w[2], 0.0) && near(auto_w[3], 0.0), "缺失类权重为 0");
  std::cout << "auto weights = [" << auto_w[0] << ", " << auto_w[1] << ", " << auto_w[2]
            << ", " << auto_w[3] << "]\n";

  // 训练集存在的类被显式赋 0：纯该类 batch 会让 Σw=0，必须在入口拒绝，不能训练中途炸。
  bool zero_rejected = false;
  try {
    (void)resolve_class_weights("0,1,0,1", ds);
  } catch (const std::invalid_argument&) {
    zero_rejected = true;
  }
  check(zero_rejected, "训练集存在的类显式赋 0 权重必须被拒绝");
  const std::vector<float> absent_zero = resolve_class_weights("1,1,0,0", ds);
  check(absent_zero[0] == 1.0f && absent_zero[1] == 1.0f && absent_zero[2] == 0.0f &&
            absent_zero[3] == 0.0f,
        "训练集不存在的类允许 0 权重");

  const std::string cfg_json = read_file(wcfg.out_dir + "/config.json");
  check(cfg_json.find("\"class_weights_source\":\"auto\"") != std::string::npos,
        "config.json 记录 class_weights_source=auto");
  check(cfg_json.find("\"class_weights\":[") != std::string::npos, "config.json 含 class_weights");
  check(count_lines_with(wcfg.out_dir + "/metrics.jsonl", "\"event\":\"class_weights\"") ==
            static_cast<std::size_t>(wcfg.epochs),
        "每 epoch 一条 class_weights 事件");

  // ---- 3) checkpoint 权重 + resume ----
  bool weights_in_state = weighted.state().class_weights.size() == auto_w.size();
  for (std::size_t i = 0; weights_in_state && i < auto_w.size(); ++i) {
    weights_in_state = weighted.state().class_weights[i] == auto_w[i];
  }
  check(weights_in_state, "checkpoint state 记录训练时的类别权重");

  TrainConfig rcfg = base;
  rcfg.out_dir = root + "/run_resume";
  rcfg.resume = wcfg.out_dir + "/ckpt.bin";
  rcfg.epochs = 1;
  Trainer resumed(rcfg);  // 不传 --class-weights：应沿用 checkpoint 的
  resumed.run();
  check(resumed.state().class_weights == auto_w, "resume 后权重与 checkpoint 一致");
  check(read_file(rcfg.out_dir + "/config.json").find("\"class_weights_source\":\"checkpoint\"") !=
            std::string::npos,
        "resume 的 config.json 标记来源 checkpoint");
  // 跨目录 resume 必须把源 run 的历史 best 一并交付，不能只继承分数。
  check(fs::exists(rcfg.out_dir + "/ckpt_best.bin"),
        "跨目录 resume 输出目录必须包含历史 ckpt_best.bin");

  // 源 run 的 best 文件丢了：不能只继承分数，必须失败而不是悄悄少交付 best。
  const std::string missing_src = root + "/run_missing_best";
  fs::create_directories(missing_src);
  fs::copy_file(wcfg.out_dir + "/ckpt.bin", missing_src + "/ckpt.bin");
  TrainConfig mcfg = rcfg;
  mcfg.out_dir = root + "/run_missing_best_out";
  mcfg.resume = missing_src + "/ckpt.bin";
  bool missing_rejected = false;
  try {
    Trainer t(mcfg);
    t.run();
  } catch (const std::exception&) {
    missing_rejected = true;
  }
  check(missing_rejected, "源 run 缺 ckpt_best.bin 时 resume 必须失败");

  bool rejected = false;
  try {
    TrainConfig bad = rcfg;
    bad.out_dir = root + "/run_bad_resume";
    bad.class_weights = "1,1,1,1";
    Trainer t(bad);
    t.run();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "resume 时显式权重与 checkpoint 不一致必须被拒绝");

  // val 只有训练中未出现的类（auto 权重为 0），仍须报告有限的无权 dev_ce。
  const auto unseen = uavrgb_test::write_class_conditional_uavr(
      root + "/unseen.uavr", 8, 8, {0, 0}, {1, 1}, {0.2f, 0.8f}, 0, 0, 3);
  TrainConfig unseen_cfg;
  unseen_cfg.data_path = unseen.path;
  unseen_cfg.out_dir = root + "/run_unseen";
  unseen_cfg.epochs = 1;
  unseen_cfg.class_weights = "auto";
  Trainer unseen_trainer(unseen_cfg);
  unseen_trainer.run();
  auto unseen_ds = load_uavr(unseen.path);
  Tensor unseen_x;
  unseen_ds.load_images(unseen_ds.val_idx, unseen_x);
  const float expected_ce = softmax_ce_loss_cuda(forward(unseen_trainer.model(), unseen_x), {1, 1});
  std::ifstream csv(unseen_cfg.out_dir + "/results.csv");
  std::string row;
  std::getline(csv, row);
  std::getline(csv, row);
  std::istringstream fields(row);
  for (int column = 0; column < 3; ++column) std::getline(fields, row, ',');
  check(std::isfinite(std::stod(row)) && near(std::stod(row), expected_ce),
        "dev_ce 使用无权 CE，验证集未见类不因零类别权重崩溃");

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << " 项)\n";
  return 1;
}
