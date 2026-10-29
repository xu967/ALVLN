// 训练器端到端自检（合成数据，不依赖真实采集）：
//   1) 20 个 step 后 loss 下降（末 5 步均值 < 首 5 步均值）
//   2) 参数确实变了
//   3) checkpoint 存了再读回来，参数/Adam 状态逐位相同；结构不符的模型必须被拒
//   4) results.csv / metrics.jsonl 的字段名与 event.hpp 完全一致
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

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

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream f(path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(f, line)) {
    if (!line.empty()) out.push_back(line);
  }
  return out;
}

// 取出扁平 JSON 对象里的键名（只按 "key": 模式扫，够用于字段名比对）。
std::vector<std::string> keys_of(const std::string& line) {
  std::vector<std::string> ks;
  for (std::size_t i = 0; i < line.size(); ++i) {
    if (line[i] != '"') continue;
    const std::size_t e = line.find('"', i + 1);
    if (e == std::string::npos) break;
    if (e + 1 < line.size() && line[e + 1] == ':') ks.push_back(line.substr(i + 1, e - i - 1));
    i = e;
  }
  return ks;
}

double field_num(const std::string& line, const std::string& pat) {
  const std::size_t p = line.find(pat);
  return p == std::string::npos ? std::nan("") : std::strtod(line.c_str() + p + pat.size(), nullptr);
}

std::vector<std::string> split_csv(const std::string& line) {
  std::vector<std::string> out;
  std::string cur;
  for (char ch : line) {
    if (ch == ',') {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += ch;
    }
  }
  out.push_back(cur);
  return out;
}

double mean(const std::vector<double>& v, std::size_t from, std::size_t n) {
  double s = 0.0;
  for (std::size_t i = from; i < from + n; ++i) s += v[i];
  return s / static_cast<double>(n);
}

bool same_bits(const Tensor& a, const Tensor& b) {
  return a.dims() == b.dims() && a.numel() == b.numel() &&
         std::memcmp(a.data(), b.data(), a.numel() * sizeof(float)) == 0;
}

std::size_t count_diffs(const CnnModel& a, const CnnModel& b) {
  const auto pa = cnn_params(a);
  const auto pb = cnn_params(b);
  std::size_t n = 0;
  for (std::size_t i = 0; i < pa.size(); ++i) {
    for (std::size_t j = 0; j < pa[i]->numel(); ++j) n += ((*pa[i])[j] != (*pb[i])[j]) ? 1 : 0;
  }
  return n;
}

void check_metrics_jsonl(const std::string& path, std::vector<double>* losses) {
  const auto want_ts = keys_of(metrics_json(make_train_step(1, 1, 0.1f, 0.1f, 1e-3f, 1.0, 1.0)));
  // validation 现在带不平衡口径的宏指标；参考事件必须把这些字段也填上，否则
  // 期望键集会比实际行少，检查就失效（或误报）。confusion 长度 = num_classes^2。
  MetricsEvent vref = make_validation(1, 0.1f, 0.1f, Split::kVal);
  vref.macro_f1 = 0.1f;
  vref.balanced_accuracy = 0.1f;
  vref.confusion = {0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const auto want_val = keys_of(metrics_json(vref));
  const auto want_err = keys_of(metrics_json(make_error(Severity::kWarn, "X", 1, "m")));
  const auto want_status = keys_of(metrics_json(make_status("m")));
  int n_step = 0, n_val = 0, n_other = 0;
  for (const std::string& line : read_lines(path)) {
    if (line.find("\"event\":\"train_step\"") != std::string::npos) {
      ++n_step;
      check(keys_of(line) == want_ts, "train_step 字段名与 event.hpp 不一致: " + line);
      const double l = field_num(line, "\"loss\":");
      check(std::isfinite(l), "train_step loss 非法: " + line);
      losses->push_back(l);
    } else if (line.find("\"event\":\"validation\"") != std::string::npos) {
      ++n_val;
      check(keys_of(line) == want_val, "validation 字段名与 event.hpp 不一致: " + line);
    } else if (line.find("\"event\":\"status\"") != std::string::npos) {
      ++n_other;
      check(keys_of(line) == want_status, "status 字段名与 event.hpp 不一致: " + line);
    } else if (line.find("\"event\":\"error\"") != std::string::npos) {
      ++n_other;
      check(keys_of(line) == want_err, "error 字段名与 event.hpp 不一致: " + line);
    } else {
      check(false, "未知 event 行: " + line);
    }
    check(line.find("\"schema\":\"tinyvln.metrics.v1\"") != std::string::npos,
          "缺 schema 字段: " + line);
  }
  check(n_step == 20, "train_step 应为 20 条（10 epoch × 2 步），实际 " + std::to_string(n_step));
  check(n_val == 10, "validation 应为 10 条，实际 " + std::to_string(n_val));
  check(n_other >= 1, "缺 status 事件");
}

void check_results_csv(const std::string& path, int epochs) {
  const auto rows = read_lines(path);
  check(!rows.empty() && rows[0] == kResultsCsvHeader, "results.csv 表头不符");
  check(static_cast<int>(rows.size()) == epochs + 1,
        "results.csv 行数应为 epochs+1，实际 " + std::to_string(rows.size()));
  for (int i = 1; i <= epochs && i < static_cast<int>(rows.size()); ++i) {
    const auto cols = split_csv(rows[static_cast<std::size_t>(i)]);
    check(cols.size() == 4, "results.csv 列数不是 4: " + rows[static_cast<std::size_t>(i)]);
    if (cols.size() == 4) {
      check(std::stoi(cols[0]) == i, "results.csv epoch 列不连续");
      check(std::isfinite(std::stod(cols[1])) && std::isfinite(std::stod(cols[2])) &&
                std::isfinite(std::stod(cols[3])),
            "results.csv 含非有限值");
    }
  }
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_trainer_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  const uavrgb_test::SynthDataset sd =
      uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", 64, 120, 160, 48, 7);

  TrainConfig cfg;
  cfg.data_path = sd.path;
  cfg.out_dir = root + "/run";
  cfg.epochs = 10;
  cfg.batch = 32;
  cfg.lr = 1e-3f;
  cfg.seed = 42;

  Trainer tr(cfg);
  tr.run();

  // 1) loss 下降
  std::vector<double> losses;
  check_metrics_jsonl(cfg.out_dir + "/metrics.jsonl", &losses);
  check(losses.size() >= 10, "loss 样本不足");
  if (losses.size() >= 10) {
    const double first = mean(losses, 0, 5);
    const double last = mean(losses, losses.size() - 5, 5);
    std::cout << "loss 首 5 步均值=" << first << " 末 5 步均值=" << last << "\n";
    check(last < first, "loss 未下降");
  }

  // 4) results.csv 字段
  check_results_csv(cfg.out_dir + "/results.csv", cfg.epochs);

  // 2) 参数确实变了：同 seed 的 make_cnn 就是训练前的初始化（make_cnn 确定性）
  const CnnModel init = make_cnn(120, 160, cfg.seed, 4);
  const std::size_t ndiff = count_diffs(init, tr.model());
  std::cout << "参数量=" << num_params(tr.model()) << " 变化的参数个数=" << ndiff << "\n";
  check(ndiff > 0, "训练后参数没变");

  // 3) checkpoint 往返
  const std::string ckpt = root + "/ckpt.bin";
  save_checkpoint(ckpt, tr.model(), tr.state());
  CnnModel m2 = make_cnn(120, 160, cfg.seed, 4);
  TrainState st2 = make_train_state(m2, cfg.lr);
  load_checkpoint(ckpt, m2, st2);
  const auto pa = cnn_params(tr.model());
  const auto pb = cnn_params(m2);
  bool all_same = true;
  for (std::size_t i = 0; i < pa.size(); ++i) all_same = all_same && same_bits(*pa[i], *pb[i]);
  check(all_same, "checkpoint 参数读回后不完全相同");
  bool adam_same = tr.state().adam.size() == st2.adam.size();
  for (std::size_t i = 0; adam_same && i < st2.adam.size(); ++i) {
    adam_same = same_bits(tr.state().adam[i].m, st2.adam[i].m) &&
                same_bits(tr.state().adam[i].v, st2.adam[i].v) &&
                tr.state().adam[i].t == st2.adam[i].t;
  }
  check(adam_same, "checkpoint Adam m/v/t 读回后不完全相同");
  check(train_step(st2) == train_step(tr.state()) && train_step(st2) == 20,
        "checkpoint step 数不符（应为 20）");
  check(st2.epoch == tr.state().epoch && st2.epoch == cfg.epochs, "checkpoint epoch 数不符");

  // 结构不符的模型必须被拒，而不是静默加载一半
  CnnModel m3 = make_cnn(128, 128, cfg.seed, 4);
  TrainState st3 = make_train_state(m3, cfg.lr);
  bool rejected = false;
  try {
    load_checkpoint(ckpt, m3, st3);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check(rejected, "结构不符的 checkpoint 未被拒绝");

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << " 项)\n";
  return 1;
}
