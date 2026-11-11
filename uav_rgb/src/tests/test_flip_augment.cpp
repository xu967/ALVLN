// 水平翻转增强的标签、prev-action、深度目标（交叉审核 F4，2026-11-12）。
// make_aug / flip_side_inputs 在 trainer_dev.cpp 的匿名命名空间里，没有测试经过它们的输出：删掉标签互换、
// prev 互换或深度列镜像，其余 CTest 照样通过。这里经 Trainer 端到端训练（--augment flip），
// 每条断言只在对应那处互换正确时成立。
//  1) 标签：训练集只有一个方向（左半亮 → YAW_LEFT），验证集只有另一个方向（右半亮 → YAW_RIGHT），再反过来做一次。
//     翻转真的发生且标签互换，模型才见过验证集的方向，验证准确率才高；互换被删或翻转没生效都在 0 附近。
//     不开翻转的对照必须低，证明夹具要求翻转。
//  2) prev-action：画面全灰，prev 在 YAW_LEFT/YAW_RIGHT 间交替，标签 = prev。灰图翻转后不变，模型无从判断
//     哪些样本被翻转过；prev 与标签不同步互换时，同一输入带两种标签，增强后批次上的训练准确率只有 0.48..0.74
//     （五种变异的实测）。
//  3) 深度目标：画面四个象限之一亮，亮象限里的深度格目标 near、其余 far。目标按列镜像才与翻转后的画面一致；
//     不镜像、整张倒序、列错位、镜像了别的样本的目标行时，同一张画面带两种目标，增强后批次上的格命中率只有
//     0.75..0.88（四种变异的实测）。
// 2)、3) 取 train_step 事件与 depth_aux 状态行里的增强后批次统计，不用评估集（评估集不加增强，
// 冲突样本的预测取决于平局时的偶然取向，不能当判据）。
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "pattern_dataset.hpp"
#include "uavrgb/action.hpp"
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
// 打印数值；越过阈值才算过，失败信息带上实测值。
void expect_above(double v, double thr, const std::string& what) {
  std::cout << what << " = " << v << std::endl;
  check(v > thr, what + "：实测 " + std::to_string(v) + "，须大于 " + std::to_string(thr));
}
void expect_below(double v, double thr, const std::string& what) {
  std::cout << what << " = " << v << std::endl;
  check(v < thr, what + "：实测 " + std::to_string(v) + "，须小于 " + std::to_string(thr));
}

constexpr int kH = 16, kW = 32;  // 深度网格 2x4
constexpr std::uint8_t kLeft = static_cast<std::uint8_t>(Action::YAW_LEFT);
constexpr std::uint8_t kRight = static_cast<std::uint8_t>(Action::YAW_RIGHT);
constexpr std::uint8_t kFwd = static_cast<std::uint8_t>(Action::FORWARD_SLOW);

std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream f(path);
  std::vector<std::string> out;
  for (std::string line; std::getline(f, line);) {
    if (!line.empty()) out.push_back(line);
  }
  return out;
}

// results.csv 最后一行的 dev_accuracy（有 val 时是验证准确率）。
double last_dev_accuracy(const std::string& out_dir) {
  const auto rows = read_lines(out_dir + "/results.csv");
  return rows.size() < 2 ? -1.0 : std::stod(rows.back().substr(rows.back().rfind(',') + 1));
}

// 文件里含 marker 的每一行，取 key 后面的数；取末尾 k 个的均值（不足 k 个返回 -1）。
double tail_mean(const std::string& path, const std::string& marker, const std::string& key, std::size_t k) {
  std::vector<double> v;
  for (const auto& line : read_lines(path)) {
    const std::size_t m = line.find(marker);
    const std::size_t p = m == std::string::npos ? m : line.find(key, m);
    if (p != std::string::npos) v.push_back(std::stod(line.substr(p + key.size())));
  }
  if (v.size() < k) return -1.0;
  double s = 0.0;
  for (std::size_t i = v.size() - k; i < v.size(); ++i) s += v[i];
  return s / static_cast<double>(k);
}

TrainConfig make_cfg(const std::string& data, const std::string& out, int epochs, bool flip) {
  TrainConfig cfg;
  cfg.data_path = data;
  cfg.out_dir = out;
  cfg.epochs = epochs;
  cfg.batch = 16;
  cfg.lr = 1e-3f;
  cfg.seed = 5;
  if (flip) cfg.augment = "flip";
  return cfg;
}

// 训练集 16 个样本全是 train_dir 方向，验证集 8 个全是另一个方向；左半亮 ↔ YAW_LEFT，右半亮 ↔ YAW_RIGHT。
void label_split(const std::string& root, const std::string& tag, bool train_left) {
  const std::string path = root + "/" + tag + ".uavr";
  write_synthetic_uavr(path, 24, kH, kW, /*train_n=*/16, /*seed=*/3);
  std::vector<Plane> planes;
  std::vector<std::uint8_t> labels;
  for (int i = 0; i < 24; ++i) {
    const bool left_bright = (i < 16) == train_left;
    planes.push_back(region_plane(kH, kW, half_region(kH, kW, left_bright)));
    labels.push_back(left_bright ? kLeft : kRight);
  }
  overwrite_samples(path, kH, kW, planes, labels);
  Trainer(make_cfg(path, root + "/" + tag + "_flip", 60, true)).run();
  Trainer(make_cfg(path, root + "/" + tag + "_plain", 60, false)).run();
  const std::string what = std::string(train_left ? "左训右验" : "右训左验");
  expect_above(last_dev_accuracy(root + "/" + tag + "_flip"), 0.95, what + "：翻转后验证准确率");
  expect_below(last_dev_accuracy(root + "/" + tag + "_plain"), 0.5, what + "：不开翻转的对照验证准确率");
}

}  // namespace

int main() {
  const std::string root =
      (fs::temp_directory_path() / ("uavrgb_test_flip_" + std::to_string(std::random_device{}()))).string();
  fs::create_directories(root);

  // 1) 标签互换，两个方向各一次。
  label_split(root, "label_lr", /*train_left=*/true);
  label_split(root, "label_rl", /*train_left=*/false);

  // 2) prev-action 与标签同步互换：灰图，prev 交替 LEFT/RIGHT，标签 = prev，每个样本自己是自己的上一帧。
  {
    const std::string path = root + "/prev.uavr";
    const int n = 32;
    write_synthetic_uavr(path, n, kH, kW, n, 4);
    std::vector<Plane> planes(n, flat_plane(kH, kW, 128));
    std::vector<std::uint8_t> labels;
    std::vector<std::uint32_t> self(n);
    for (int i = 0; i < n; ++i) {
      labels.push_back(i % 2 ? kRight : kLeft);
      self[i] = static_cast<std::uint32_t>(i);
    }
    overwrite_samples(path, kH, kW, planes, labels);
    write_aux_explicit(path, n, kH, kW, self, labels, {});
    TrainConfig cfg = make_cfg(path, root + "/prev", 60, true);
    cfg.prev_action = true;
    Trainer(cfg).run();
    expect_above(tail_mean(cfg.out_dir + "/metrics.jsonl", "\"event\":\"train_step\"", "\"action_accuracy\":", 60),
                 0.95, "prev-action 与标签同步互换：增强后批次的训练准确率");
  }

  // 3) 深度目标按列镜像：四个象限各 8 个样本，亮象限里的格 near=1，其余 far=6。
  {
    const std::string path = root + "/depth.uavr";
    const int n = 32;
    write_synthetic_uavr(path, n, kH, kW, n, 6);
    std::vector<Plane> planes;
    std::vector<std::uint8_t> labels(n, kFwd), depth;
    std::vector<std::uint32_t> self(n);
    for (int i = 0; i < n; ++i) {
      const Region r = quad_region(kH, kW, /*top=*/(i & 2) == 0, /*left=*/(i & 1) == 0);
      planes.push_back(region_plane(kH, kW, r));
      const Plane t = region_depth(kH, kW, r, 1, 6);
      depth.insert(depth.end(), t.begin(), t.end());
      self[i] = static_cast<std::uint32_t>(i);
    }
    overwrite_samples(path, kH, kW, planes, labels);
    write_aux_explicit(path, n, kH, kW, self, std::vector<std::uint8_t>(n, 0), depth);
    TrainConfig cfg = make_cfg(path, root + "/depth", 100, true);
    cfg.depth_aux = true;
    Trainer(cfg).run();
    expect_above(tail_mean(cfg.out_dir + "/metrics.jsonl", "depth_aux epoch ", "train_acc=", 20), 0.95,
                 "深度目标按列镜像：增强后批次的格命中率");
  }

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "test_flip_augment: PASS\n";
    return 0;
  }
  std::cerr << "test_flip_augment: " << g_failures << " 项失败\n";
  return 1;
}
