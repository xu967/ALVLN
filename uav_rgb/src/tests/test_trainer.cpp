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
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/sha256.hpp"
#include "uavrgb/trainer.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;

namespace {

int g_failures = 0;

// v5 checkpoint 在 text_dim 之后的固定训练契约长度。
constexpr std::size_t kContractBytes = 32 + 1 + 32 + 4 + 4 + 4 + 4 + 32;

std::string hex_sha(const Sha256& h) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (unsigned char b : h) { s += d[b >> 4]; s += d[b & 15]; }
  return s;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}

// 从 metrics.jsonl 取最后一条 split=test 的 action_accuracy（eval-only 不设 test_acc_）。
double read_test_accuracy(const std::string& path) {
  std::ifstream f(path);
  std::string line;
  double v = -1.0;
  while (std::getline(f, line)) {
    if (line.find("\"split\":\"test\"") == std::string::npos) continue;
    const std::string key = "\"action_accuracy\":";
    const std::size_t p = line.find(key);
    if (p != std::string::npos) v = std::stod(line.substr(p + key.size()));
  }
  return v;
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

// 取文件里某个数值字段的最大值；没有匹配行时返回 NaN。
double max_field(const std::string& path, const std::string& pat) {
  double best = std::nan("");
  for (const std::string& line : read_lines(path)) {
    const double v = field_num(line, pat);
    if (std::isfinite(v) && !(v <= best)) best = v;
  }
  return best;
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
  // JSON 控制字符必须转义，否则 metrics.jsonl 不是合法 JSON（G0/B3）。
  {
    const std::string js = metrics_json(make_error(Severity::kWarn, "X", 1, "a\tb\rc"));
    check(js.find("\\t") != std::string::npos, "metrics_json 未转义 \\t");
    check(js.find("\\r") != std::string::npos, "metrics_json 未转义 \\r");
  }

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

  // 文本分支必须能从 run 目录复现：config.json 要记录 text_dim（L2 的 A/B 全靠它）。
  // 事实字段（数据 SHA/口径/优化器超参）也必须在场，旧字段保持原样。
  {
    std::ifstream cf(cfg.out_dir + "/config.json");
    const std::string cjs((std::istreambuf_iterator<char>(cf)),
                          std::istreambuf_iterator<char>());
    check(cjs.find("\"text_dim\":0") != std::string::npos, "config.json 缺 text_dim 字段");
    check(cjs.find("\"dataset_sha256\":\"" + hex_sha(sha256_file(sd.path)) + "\"") !=
              std::string::npos,
          "config.json 的 dataset_sha256 与数据文件不符");
    check(cjs.find("\"split_scheme\":\"scene_id\"") != std::string::npos,
          "config.json 缺 split_scheme（合成集为 scene_id）");
    check(cjs.find("\"drop_collision\":false") != std::string::npos,
          "config.json 缺 drop_collision");
    check(cjs.find("\"soft_targets_sha256\":\"\"") != std::string::npos,
          "config.json 缺 soft_targets_sha256（未蒸馏应为空）");
    check(cjs.find("\"beta1\":") != std::string::npos &&
              cjs.find("\"beta2\":") != std::string::npos &&
              cjs.find("\"eps\":") != std::string::npos,
          "config.json 缺 Adam beta/eps");
    check(cjs.find("\"seed\":42") != std::string::npos, "config.json 缺 seed");
    check(cjs.find("\"batch\":32") != std::string::npos, "config.json 缺 batch");
  }

  // 5) best-checkpoint：记录值必须等于 dev 宏 F1 峰值，且文件能读回。
  const double best_seen = max_field(cfg.out_dir + "/metrics.jsonl", "\"macro_f1\":");
  std::cout << "best dev macro_f1=" << best_seen
            << " state=" << tr.state().best_macro_f1 << "\n";
  check(std::isfinite(best_seen), "metrics.jsonl 无 macro_f1");
  check(tr.state().best_macro_f1 >= 0.0f, "best_macro_f1 未被记录");
  check(std::fabs(static_cast<double>(tr.state().best_macro_f1) - best_seen) < 1e-5,
        "best_macro_f1 与 metrics 峰值不一致");
  const std::string best_ckpt = cfg.out_dir + "/ckpt_best.bin";
  check(fs::exists(best_ckpt), "缺 ckpt_best.bin");
  CnnModel best_model = make_cnn(120, 160, cfg.seed, 4);
  TrainState best_state = make_train_state(best_model, cfg.lr);
  load_checkpoint(best_ckpt, best_model, best_state);
  check(best_state.best_macro_f1 == tr.state().best_macro_f1,
        "ckpt_best.bin 未保存 best_macro_f1");
  check(best_state.epoch >= 1 && best_state.epoch <= cfg.epochs, "ckpt_best.bin epoch 越界");
  // ckpt.bin 必须在更新 best 之后写，否则续训读到的峰值落后一轮、可能覆盖掉 ckpt_best。
  CnnModel last_model = make_cnn(120, 160, cfg.seed, 4);
  TrainState last_state = make_train_state(last_model, cfg.lr);
  load_checkpoint(cfg.out_dir + "/ckpt.bin", last_model, last_state);
  check(last_state.best_macro_f1 == tr.state().best_macro_f1,
        "ckpt.bin 的 best_macro_f1 落后于 ckpt_best.bin");

  // 2) 参数确实变了：同 seed 的 make_cnn 就是训练前的初始化（make_cnn 确定性）
  const CnnModel init = make_cnn(120, 160, cfg.seed, 4);
  const std::size_t ndiff = count_diffs(init, tr.model());
  std::cout << "参数量=" << num_params(tr.model()) << " 变化的参数个数=" << ndiff << "\n";
  check(ndiff > 0, "训练后参数没变");

  // 3) checkpoint 往返
  const std::string ckpt = root + "/ckpt.bin";
  save_checkpoint(ckpt, tr.model(), tr.state());
  const CheckpointMeta meta = read_checkpoint_meta(ckpt);
  check(meta.in_h == 120 && meta.in_w == 160 && meta.in_c == 3 && meta.num_classes == 4,
        "read_checkpoint_meta 读回的尺寸不符");
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
  check(st2.best_macro_f1 == tr.state().best_macro_f1,
        "checkpoint best_macro_f1 读回不一致");
  check(st2.contract_present && st2.contract.lr == cfg.lr && st2.contract.seed == cfg.seed &&
            st2.contract.batch == cfg.batch,
        "v5 checkpoint 未保存训练契约（lr/seed/batch）");
  check(hex_sha(st2.contract.dataset_sha256) == hex_sha(sha256_file(sd.path)),
        "v5 checkpoint 的 dataset_sha256 不符");

  // v2 兼容：旧 checkpoint 没有 best_macro_f1/text_dim/训练契约，读回应为 -1 且无契约。
  const std::string v2 = root + "/ckpt_v2.bin";
  {
    std::ifstream in(ckpt, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    std::uint32_t nw = 0;
    std::memcpy(&nw, &bytes[40], 4);
    check(nw == 0, "v2 兼容用例假设 checkpoint 无类别权重");
    const std::uint32_t two = 2;
    std::memcpy(&bytes[8], &two, 4);
    // 去掉 v6 aux_classes/prev_dim、v5 契约、v3 best_macro_f1、v4 text_dim，还原成 v2 布局。
    bytes.erase(bytes.begin() + 52 + kContractBytes, bytes.begin() + 52 + kContractBytes + 8);
    bytes.erase(bytes.begin() + 52, bytes.begin() + 52 + kContractBytes);
    bytes.erase(bytes.begin() + 44, bytes.begin() + 52);
    std::ofstream out(v2, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  CnnModel m4 = make_cnn(120, 160, cfg.seed, 4);
  TrainState st4 = make_train_state(m4, cfg.lr);
  load_checkpoint(v2, m4, st4);
  check(st4.best_macro_f1 == -1.0f, "v2 checkpoint 的 best_macro_f1 应为 -1");
  check(!st4.contract_present, "v2 checkpoint 不应有训练契约");

  // 旧 checkpoint 缺契约字段：resume 按旧行为放行（告警），不能拒绝。
  {
    const std::string v2src = root + "/v2src";
    fs::create_directories(v2src);
    fs::copy_file(v2, v2src + "/ckpt.bin");
    TrainConfig vc = cfg;
    vc.out_dir = root + "/run_resume_v2";
    vc.resume = v2src + "/ckpt.bin";
    vc.epochs = 1;
    bool ok = true;
    try {
      Trainer t(vc);
      t.run();
    } catch (const std::exception& e) {
      ok = false;
      std::cerr << "  v2 resume 失败: " << e.what() << "\n";
    }
    check(ok, "旧 v2 checkpoint 无训练契约字段时 resume 应按旧行为放行");
    check(fs::exists(vc.out_dir + "/config.resume.json"),
          "旧 run resume 也要写 config.resume.json 留痕");
  }
  const auto pc = cnn_params(m4);
  bool v2_same = true;
  for (std::size_t i = 0; i < pa.size(); ++i) v2_same = v2_same && same_bits(*pa[i], *pc[i]);
  check(v2_same, "v2 checkpoint 参数读回后不一致");

  // 含 NaN 参数的 checkpoint 必须在读入边界被拒绝（ReLU 会把中间 NaN 压成 0，
  // 不查 checkpoint 的话坏模型能被静默续训并写回）。
  const std::string nan_ckpt = root + "/ckpt_nan.bin";
  {
    std::ifstream in(ckpt, std::ios::binary);
    std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                            std::istreambuf_iterator<char>());
    std::uint32_t ndim = 0;
    const std::size_t tensor_off = 52 + kContractBytes + 8;  // v5 契约 + v6 两个 i32 之后的第一个参数
    std::memcpy(&ndim, &bytes[tensor_off], 4);
    const std::size_t off = tensor_off + 4 + static_cast<std::size_t>(ndim) * 8;
    check(off + 4 <= bytes.size(), "NaN 注入点超出 checkpoint 文件大小");
    if (off + 4 > bytes.size()) {
      std::cerr << "RESULT: FAIL（NaN 注入越界）\n";
      return 1;
    }
    const std::uint32_t nan_bits = 0x7fc00000u;
    std::memcpy(&bytes[off], &nan_bits, 4);
    std::ofstream out(nan_ckpt, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  CnnModel m5 = make_cnn(120, 160, cfg.seed, 4);
  TrainState st5 = make_train_state(m5, cfg.lr);
  bool nan_rejected = false;
  std::string nan_msg;
  try {
    load_checkpoint(nan_ckpt, m5, st5);
  } catch (const std::invalid_argument& e) {
    nan_rejected = true;
    nan_msg = e.what();
  }
  // 只断言“抛了 invalid_argument”不够：形状不符也会抛，必须确认走的是有限性检查。
  check(nan_rejected && nan_msg.find("含非有限值") != std::string::npos,
        "含 NaN 参数的 checkpoint 必须被有限性检查拒绝（实际: " + nan_msg + "）");

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

  // 内联 test 必须评 ckpt_best，不是最后一轮：dev 宏 F1 峰值常早于末轮，评 model_
  // 会报出与交付权重不符的数字（2026-11-03 审计）。用 eval-only 在同一份
  // ckpt_best 上复算，两边的 test 准确率必须一致。
  {
    const std::string ds2 = root + "/ds_best.uavr";
    const uavrgb_test::SynthDataset sd2 =
        uavrgb_test::write_synthetic_uavr(ds2, 64, 120, 160, 48, 9, /*test_n=*/8);
    TrainConfig c2;
    c2.data_path = sd2.path;
    c2.out_dir = root + "/run_best";
    c2.epochs = 6;
    c2.batch = 16;
    c2.lr = 1e-3f;
    c2.seed = 7;
    Trainer t2(c2);
    t2.run();
    const double inline_test = t2.last_test_accuracy();

    TrainConfig c2e = c2;
    c2e.out_dir = root + "/run_best_eval";
    Trainer ev(c2e);
    ev.eval_only(c2.out_dir + "/ckpt_best.bin", /*eval_val=*/false, /*eval_test=*/true);
    const double evalonly_test = read_test_accuracy(c2e.out_dir + "/metrics.jsonl");
    std::cout << "内联 test acc=" << inline_test << "  ckpt_best 复算=" << evalonly_test
              << "\n";
    check(inline_test >= 0.0 && std::abs(inline_test - evalonly_test) < 1e-9,
          "内联 test 与 ckpt_best 的 eval-only 结果不一致（评的不是交付权重）");

    // test 评估留痕：训练末尾的自动 test 也要记录次数/时间/数据身份。
    const auto rec = read_lines(c2.out_dir + "/test_eval.jsonl");
    check(rec.size() == 1 && rec[0].find("\"origin\":\"train_inline\"") != std::string::npos &&
              rec[0].find("\"eval_count\":1") != std::string::npos &&
              rec[0].find("\"split\":\"test\"") != std::string::npos &&
              rec[0].find("\"dataset_sha256\":\"" + hex_sha(sha256_file(sd2.path)) + "\"") !=
                  std::string::npos,
          "训练末尾 test 评估未写 test_eval.jsonl（次数/数据身份）");

    // resume 严格化：未显式传入的 lr/seed/batch 从源 run 的 config.json 继承；
    // 同目录续训只写 config.resume.json，绝不截断原 config.json。
    {
      const std::string cfg_before = read_file(c2.out_dir + "/config.json");
      TrainConfig rr = c2;
      rr.out_dir = c2.out_dir;
      rr.resume = c2.out_dir + "/ckpt.bin";
      rr.epochs = 1;
      rr.seed = 1234;  // 非源值，但 seed_set=false：应从 config 继承 7
      rr.batch = 64;
      bool ok = true;
      try {
        Trainer t3(rr);
        t3.run();
      } catch (const std::exception& e) {
        ok = false;
        std::cerr << "  resume 继承失败: " << e.what() << "\n";
      }
      check(ok, "未显式传 seed/batch 的 resume 应继承源 config（否则契约不一致）");
      check(read_file(c2.out_dir + "/config.json") == cfg_before,
            "resume 截断了原 config.json");
      const std::string rjs = read_file(c2.out_dir + "/config.resume.json");
      check(rjs.find("\"seed\":7") != std::string::npos &&
                rjs.find("\"batch\":16") != std::string::npos,
            "config.resume.json 未记录继承后的 seed/batch");
      check(rjs.find("\"resume\":") != std::string::npos,
            "config.resume.json 缺 resume 来源");
    }

    // 契约不一致：显式换 seed 必须拒绝；--allow-resume-mismatch 才放行。
    {
      TrainConfig bad = c2;
      bad.out_dir = root + "/run_resume_bad";
      bad.resume = c2.out_dir + "/ckpt.bin";
      bad.epochs = 1;
      bad.seed = 99;
      bad.seed_set = true;
      bool rejected = false;
      try {
        Trainer t4(bad);
        t4.run();
      } catch (const std::invalid_argument& e) {
        rejected = std::string(e.what()).find("契约") != std::string::npos;
      }
      check(rejected, "resume 与 checkpoint 训练契约不一致时必须拒绝");
      bad.allow_resume_mismatch = true;
      bad.out_dir = root + "/run_resume_allow";
      bool allowed = true;
      try {
        Trainer t5(bad);
        t5.run();
      } catch (const std::exception& e) {
        allowed = false;
        std::cerr << "  --allow-resume-mismatch 失败: " << e.what() << "\n";
      }
      check(allowed, "--allow-resume-mismatch 应放行契约不一致");
    }

    // 旧 config 缺事实字段（dataset_sha256/split_scheme/...）也能续训：
    // 只补全它能给出的 lr/seed/batch，其余按当前数据集/默认值走。
    {
      const std::string oldsrc = root + "/oldsrc";
      fs::create_directories(oldsrc);
      fs::copy_file(c2.out_dir + "/ckpt.bin", oldsrc + "/ckpt.bin");
      fs::copy_file(c2.out_dir + "/ckpt_best.bin", oldsrc + "/ckpt_best.bin");
      std::ofstream(oldsrc + "/config.json")
          << "{\"schema\":\"uavrgb.train_config.v1\",\"batch\":16,\"lr\":0.001,"
             "\"seed\":7,\"drop_collision\":false}\n";
      TrainConfig oc = c2;
      oc.out_dir = root + "/run_oldcfg";
      oc.resume = oldsrc + "/ckpt.bin";
      oc.epochs = 1;
      oc.seed = 1234;  // 未显式给出：应被旧 config 的 7 覆盖
      oc.batch = 64;
      bool ok = true;
      try {
        Trainer t6(oc);
        t6.run();
      } catch (const std::exception& e) {
        ok = false;
        std::cerr << "  旧 config resume 失败: " << e.what() << "\n";
      }
      check(ok, "旧 config 缺新字段时应兼容续训");
    }
  }

  // 数据集带 text_ids 侧车却没给 --text-dim：必须报错，不能静默训成纯视觉模型。
  {
    const std::string ds3 = root + "/ds_sidecar.uavr";
    const uavrgb_test::SynthDataset sd3 =
        uavrgb_test::write_synthetic_uavr(ds3, 32, 120, 160, 16, 11, /*test_n=*/8);
    {
      // 侧车必须覆盖全部 episode（train 0..15 / val 16..23 / test 24..31）。
      std::ofstream f(root + "/instructions.jsonl");
      f << "{\"episode_id\": 1,\"instruction\": \"go to the red pillar on your left\", "
           "\"sample_start\": 0, \"sample_count\": 16, \"text_ids\": [2, 3, 4]}\n"
           "{\"episode_id\": 2,\"instruction\": \"go to the red pillar on your right\", "
           "\"sample_start\": 16, \"sample_count\": 8, \"text_ids\": [2, 3, 5]}\n"
           "{\"episode_id\": 3,\"instruction\": \"go to the blue pillar on your left\", "
           "\"sample_start\": 24, \"sample_count\": 8, \"text_ids\": [2, 6, 4]}\n";
    }
    TrainConfig c3;
    c3.data_path = sd3.path;
    c3.out_dir = root + "/run_guard";
    c3.epochs = 1;
    c3.batch = 8;
    bool threw = false;
    try {
      Trainer t3(c3);
      t3.run();
    } catch (const std::runtime_error&) {
      threw = true;
    }
    check(threw, "带 instructions.jsonl 侧车但 text_dim=0 时必须报错（否则静默训纯视觉）");

    c3.vision_only = true;
    c3.out_dir = root + "/run_vision_only";
    bool ok = true;
    try {
      Trainer t3(c3);
      t3.run();
    } catch (const std::exception& e) {
      ok = false;
      std::cerr << "  --vision-only 失败: " << e.what() << "\n";
    }
    check(ok, "--vision-only 应允许在带侧车的数据集上训纯视觉模型");
  }

  // results.csv 的 train_ce 必须是无权、无平滑的 CE（与 dev_ce 同口径）。开 label_smoothing
  // 时 step loss 是平滑损失：模型一旦偏离均匀预测，平滑损失就明显高于无平滑 CE
  // （ε=0.2、C=4 的平滑最优点 CE≈0.16 而平滑损失≈0.59）。曾经的判断漏掉 ε，
  // train_ce 列记成了 step loss 的均值。两类亮度可分，几步就能学会。
  {
    const std::vector<int> tl(32, 0), vl(8, 0);
    std::vector<int> train_labels = tl, val_labels = vl;
    train_labels.insert(train_labels.end(), 32, 1);
    val_labels.insert(val_labels.end(), 8, 1);
    fs::create_directories(root + "/smooth");  // 单独目录：避开上面用例留下的 instructions.jsonl 侧车
    const uavrgb_test::SynthDataset sd4 = uavrgb_test::write_class_conditional_uavr(
        root + "/smooth/ds_smooth.uavr", 32, 32, train_labels, val_labels,
        {0.15f, 0.85f, 0.0f, 0.0f}, 0.02f, 0.02f, 21);
    TrainConfig c4;
    c4.data_path = sd4.path;
    c4.out_dir = root + "/run_smooth";
    c4.epochs = 10;
    c4.batch = 32;
    c4.lr = 1e-3f;
    c4.seed = 5;
    c4.label_smoothing = 0.2f;
    Trainer t4(c4);
    t4.run();
    const auto rows = read_lines(c4.out_dir + "/results.csv");
    std::vector<double> step_losses;
    for (const std::string& line : read_lines(c4.out_dir + "/metrics.jsonl")) {
      if (line.find("\"event\":\"train_step\"") != std::string::npos) {
        step_losses.push_back(field_num(line, "\"loss\":"));
      }
    }
    check(rows.size() == 11 && step_losses.size() == 20, "平滑口径用例应有 10 行 CSV、20 个 step");
    if (rows.size() == 11 && step_losses.size() == 20) {
      const double train_ce = std::stod(split_csv(rows[10])[1]);
      const double smoothed = mean(step_losses, 18, 2);  // 末轮 2 个 step 的平滑损失均值
      std::cout << "label_smoothing=0.2 末轮: train_ce=" << train_ce
                << " 平滑 step loss 均值=" << smoothed << "\n";
      check(train_ce < 0.9 * smoothed,
            "results.csv 的 train_ce 应是无平滑 CE，明显低于平滑 step loss（实际 " +
                std::to_string(train_ce) + " vs " + std::to_string(smoothed) + "）");
    }
  }

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << " 项)\n";
  return 1;
}
