// LoadOptions.drop_collision 自检（无外部测试框架）：
//   meta.collision_flags 里的 '1' 步必须从 train/val/test 三个下标表里删掉；
//   请求过滤但 meta 缺标记时必须报错（不得静默保留错误标签）。
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
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

  // host 端 u8→float 必须与设备 kernel（augment_dev.cu：x/255.0f）逐位一致：
  // 写成 x*(1/255.0f) 会在 126 个取值上差 1 ulp。像素均匀随机，256 个取值全覆盖。
  {
    Tensor xf;
    legacy_ds.load_images(legacy_ds.train_idx, xf);
    const std::size_t plane = 3u * 64u * 64u;
    std::vector<unsigned char> raw(plane * 8);
    std::ifstream f(legacy.path, std::ios::binary);
    f.seekg(static_cast<std::streamoff>(legacy_ds.images_off));
    f.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    check(static_cast<std::size_t>(f.gcount()) == raw.size(), "读取原始像素字节");
    std::vector<bool> seen(256, false);
    std::size_t mismatch = 0;
    for (std::size_t j = 0; j < raw.size(); ++j) {
      seen[raw[j]] = true;
      if (xf.data()[j] != static_cast<float>(raw[j]) / 255.0f) ++mismatch;
    }
    const std::size_t covered = static_cast<std::size_t>(std::count(seen.begin(), seen.end(), true));
    std::cout << "u8->float: 取值覆盖=" << covered << "/256 与 x/255.0f 不一致=" << mismatch << "\n";
    check(covered == 256, "测试数据覆盖全部 256 个像素取值");
    check(mismatch == 0, "load_images 与 x/255.0f 逐位一致");
  }
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
