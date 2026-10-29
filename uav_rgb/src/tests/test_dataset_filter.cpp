// LoadOptions.drop_collision 自检（无外部测试框架）：
//   meta.collision_flags 里的 '1' 步必须从 train/val/test 三个下标表里删掉；
//   请求过滤但 meta 缺标记时必须报错（不得静默保留错误标签）。
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/dataset.hpp"

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
}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_filter_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  // 12 样本：train 全局 0..7，val 全局 8..11；碰撞步 = 2,4,7,9。
  const std::vector<int> train_labels(8, 0);
  const std::vector<int> val_labels(4, 0);
  const std::vector<float> means = {0.5f, 0.5f, 0.0f, 0.0f};
  const std::vector<std::uint8_t> collision = {0, 0, 1, 0, 1, 0, 0, 1,
                                               0, 1, 0, 0};
  const auto sd = uavrgb_test::write_class_conditional_uavr(
      root + "/filter.uavr", 64, 64, train_labels, val_labels, means, 0.0f, 0.0f, 5,
      collision);

  const UavrDataset all = load_uavr(sd.path);
  check(all.train_idx.size() == 8 && all.val_idx.size() == 4, "默认读取保留全部样本");

  const UavrDataset kept = load_uavr(sd.path, LoadOptions{true});
  const std::vector<std::uint32_t> want_train = {0, 1, 3, 5, 6};
  const std::vector<std::uint32_t> want_val = {8, 10, 11};
  std::cout << "samples: all train=" << all.train_idx.size() << " val=" << all.val_idx.size()
            << " -> drop_collision train=" << kept.train_idx.size()
            << " val=" << kept.val_idx.size() << "\n";
  check(kept.train_idx == want_train, "train 碰撞步被删除");
  check(kept.val_idx == want_val, "val 碰撞步被删除");
  check(kept.train_idx.size() + kept.val_idx.size() == 8, "共删除 4 个碰撞步");

  Tensor x;
  kept.load_images(kept.train_idx, x);
  check(x.dims() == std::vector<std::int64_t>({5, 3, 64, 64}), "过滤后仍能按下标读图像");

  // 旧格式（无 collision_flags）：默认读可以，请求过滤必须失败。
  const auto legacy = uavrgb_test::write_synthetic_uavr(root + "/legacy.uavr", 8, 64, 64, 8, 3);
  const UavrDataset legacy_ds = load_uavr(legacy.path);
  check(legacy_ds.train_idx.size() == 8, "无 collision_flags 的旧数据集默认可读");
  bool threw = false;
  try {
    (void)load_uavr(legacy.path, LoadOptions{true});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  check(threw, "请求 drop_collision 但缺 collision_flags 时抛异常");

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "test_dataset_filter: PASS\n";
    return 0;
  }
  std::cerr << "test_dataset_filter: " << g_failures << " 项失败\n";
  return 1;
}
