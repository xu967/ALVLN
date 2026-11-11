// 双帧训练读的是侧车 prev_index 指向的那一帧，平面顺序与部署一致（交叉审核 F5，2026-11-12）。
// test_overfit 的双帧用例标签只由当前帧亮度决定，侧车是固定的 i-1 链：把上一帧读成当前帧自己、
// 读成 i-1、按批内位置而不是样本下标取 prev_index，准确率照样过 95%，三种读错都不会红。
//  A) 标签只由 prev_index 指向的那一帧决定。当前帧类别 i%4，上一帧类别 (i/4)%4，两者在 32 个样本里
//     16 种组合各出现 2 次：只看当前帧、或读成 i-1 的模型，最好也只有 25%（夹具自检里按组合表算出）。
//     只有真的读到 prev_index 那一帧才能过 95%。
//  B) 平面顺序。标签只由当前帧决定，上一帧类别与它无关。训练完把检查点装回，用部署路径 run_infer_loop
//     按「当前帧在前、上一帧在后」喂同样的帧：训练端把两个平面摆反时，模型学到的是通道 3..5，
//     部署端那里放的是上一帧，准确率回到 25% 左右。训练本身看不出顺序，所以必须过部署路径。
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "pattern_dataset.hpp"
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/infer.hpp"
#include "uavrgb/trainer.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;
using namespace uavrgb_test;

namespace {

int g_failures = 0;
void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}
void expect_above(double v, double thr, const std::string& what) {
  std::cout << what << " = " << v << std::endl;
  check(v > thr, what + "：实测 " + std::to_string(v) + "，须大于 " + std::to_string(thr));
}
void expect_below(double v, double thr, const std::string& what) {
  std::cout << what << " = " << v << std::endl;
  check(v < thr, what + "：实测 " + std::to_string(v) + "，须小于 " + std::to_string(thr));
}

constexpr int kH = 16, kW = 32, kN = 32;
constexpr std::uint8_t kLevel[4] = {30, 90, 150, 210};  // 四档亮度对应类别 0..3

int cur_class(int i) { return i % 4; }
int prev_class(int i) { return (i / 4) % 4; }
// 样本 i 的上一帧下标：类别 (i/4)%4，位置与 i 无规律（既不是 i 也不是 i-1）。
std::uint32_t prev_of(int i) { return static_cast<std::uint32_t>(4 * ((3 * i + 1) % 8) + prev_class(i)); }

// 夹具自检：模型只能看到 (当前帧类别, 读到的那一帧类别) 时，按组合表取多数标签能达到的最高准确率。
double ceiling(int (*label)(int), int (*read_class)(int)) {
  std::map<std::pair<int, int>, std::map<int, int>> table;
  for (int i = 0; i < kN; ++i) ++table[{cur_class(i), read_class(i)}][label(i)];
  int best = 0;
  for (const auto& kv : table) {
    int m = 0;
    for (const auto& c : kv.second) m = std::max(m, c.second);
    best += m;
  }
  return static_cast<double>(best) / kN;
}
int read_self(int i) { return cur_class(i); }
int read_prev_minus_1(int i) { return cur_class((i + kN - 1) % kN); }
int read_right(int i) { return prev_class(i); }

std::string make_dataset(const std::string& root, const std::string& tag, bool label_from_prev) {
  const std::string path = root + "/" + tag + ".uavr";
  write_synthetic_uavr(path, kN, kH, kW, kN, /*seed=*/9);
  std::vector<Plane> planes;
  std::vector<std::uint8_t> labels;
  std::vector<std::uint32_t> prev_index;
  for (int i = 0; i < kN; ++i) {
    planes.push_back(flat_plane(kH, kW, kLevel[cur_class(i)]));
    labels.push_back(static_cast<std::uint8_t>(label_from_prev ? prev_class(i) : cur_class(i)));
    prev_index.push_back(prev_of(i));
  }
  overwrite_samples(path, kH, kW, planes, labels);
  write_aux_explicit(path, kN, kH, kW, prev_index, std::vector<std::uint8_t>(kN, 0), {});
  return path;
}

TrainConfig make_cfg(const std::string& data, const std::string& out) {
  TrainConfig cfg;
  cfg.data_path = data;
  cfg.out_dir = out;
  cfg.epochs = 100;
  cfg.batch = 16;
  cfg.lr = 1e-3f;
  cfg.seed = 5;
  cfg.two_frame = true;
  return cfg;
}

// results.csv 最后一行的 dev_accuracy；数据集没有 val 时回退到训练集。
double last_dev_accuracy(const std::string& out_dir) {
  std::ifstream f(out_dir + "/results.csv");
  std::string line, last;
  while (std::getline(f, line)) {
    if (!line.empty()) last = line;
  }
  return std::stod(last.substr(last.rfind(',') + 1));
}

std::string hwc_frame(std::uint8_t level) {
  return std::string(static_cast<std::size_t>(kH) * kW * 3, static_cast<char>(level));
}

// 部署路径对 32 个样本逐一给出的动作：每步 [当前帧][上一帧]，各 HWC；输出一行一个 JSON。
std::vector<int> infer_actions(const CnnModel& model) {
  std::string bytes;
  for (int i = 0; i < kN; ++i) bytes += hwc_frame(kLevel[cur_class(i)]) + hwc_frame(kLevel[prev_class(i)]);
  std::istringstream in(bytes);
  std::ostringstream out;
  run_infer_loop(model, in, out);
  std::vector<int> actions;
  std::istringstream lines(out.str());
  for (std::string line; std::getline(lines, line);) {
    const std::size_t p = line.find("\"action\":");
    actions.push_back(p == std::string::npos ? -1 : std::stoi(line.substr(p + 9)));
  }
  return actions;
}

}  // namespace

int main() {
  const std::string root =
      (fs::temp_directory_path() / ("uavrgb_test_twoframe_" + std::to_string(std::random_device{}()))).string();
  fs::create_directories(root);

  // 夹具自检：标签 = 上一帧类别时，只看当前帧或读成 i-1 的上限；读对了才能到 1。
  expect_below(ceiling(prev_class, read_self), 0.3, "夹具自检：读成当前帧自己的准确率上限");
  expect_below(ceiling(prev_class, read_prev_minus_1), 0.3, "夹具自检：读成 i-1 的准确率上限");
  expect_above(ceiling(prev_class, read_right), 0.99, "夹具自检：读对 prev_index 的准确率上限");

  // A) 标签只由 prev_index 那一帧决定。
  {
    const TrainConfig cfg = make_cfg(make_dataset(root, "a", /*label_from_prev=*/true), root + "/a");
    Trainer(cfg).run();
    expect_above(last_dev_accuracy(cfg.out_dir), 0.95, "A) 标签只由 prev_index 帧决定：训练准确率");
  }

  // B) 平面顺序：训练 → 装回检查点 → 部署路径按「当前在前、上一帧在后」推理。
  {
    const TrainConfig cfg = make_cfg(make_dataset(root, "b", /*label_from_prev=*/false), root + "/b");
    Trainer(cfg).run();
    const std::string ckpt = cfg.out_dir + "/ckpt.bin";
    const CheckpointMeta meta = read_checkpoint_meta(ckpt);
    CnnModel model = make_cnn(meta.in_h, meta.in_w, 0, meta.num_classes, meta.text_dim, meta.in_c, meta.aux_classes,
                              meta.prev_dim);
    TrainState st = make_train_state(model, 0.0f);
    load_checkpoint(ckpt, model, st);
    const std::vector<int> actions = infer_actions(model);
    check(actions.size() == static_cast<std::size_t>(kN), "B) 部署路径应对每个样本输出一行");
    int ok = 0;
    for (std::size_t i = 0; i < actions.size(); ++i) ok += actions[i] == cur_class(static_cast<int>(i)) ? 1 : 0;
    expect_above(static_cast<double>(ok) / kN, 0.95, "B) 部署路径按「当前在前、上一帧在后」推理的准确率");
  }

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "test_two_frame_prev_index: PASS\n";
    return 0;
  }
  std::cerr << "test_two_frame_prev_index: " << g_failures << " 项失败\n";
  return 1;
}
