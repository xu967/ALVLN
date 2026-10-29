#pragma once
// UAVR 二进制数据集读取（格式见 tools/DATASET_FORMAT.md v1，只读，不引入第三方库）。
// H/W/C 一律从 48 B 文件头读，不假定 160x120；train/val/test 划分取 meta JSON 的
// episodes[].sample_start/sample_count/split（按 scene_id 隔离，不按帧随机拆）。
#include "uavrgb/tensor.hpp"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace uavrgb {

// 读取选项；默认与旧行为完全一致（不过滤）。只加选项，不改数据格式。
struct LoadOptions {
  // true：按 meta.collision_flags 丢掉碰撞步。标记由新版 preprocess.py 写入；缺失时
  // 直接报错而不是静默不过滤——碰撞步的专家标签本身是错的，静默保留会污染训练。
  bool drop_collision = false;
};

struct UavrDataset {
  int n = 0;
  int c = 0;
  int h = 0;
  int w = 0;
  std::uint32_t num_classes = 4;
  std::uint64_t images_off = 0;
  std::uint64_t labels_off = 0;
  std::vector<std::uint8_t> labels;         // [n]，已校验 0..num_classes-1
  std::vector<std::uint32_t> train_idx;     // 全局样本下标
  std::vector<std::uint32_t> val_idx;
  std::vector<std::uint32_t> test_idx;
  mutable std::ifstream file;  // 按需 seek 读样本，不整文件载入内存

  UavrDataset() = default;
  UavrDataset(UavrDataset&&) = default;
  UavrDataset& operator=(UavrDataset&&) = default;

  // 读 idx 指定的样本，写满 out[b,c,h,w]（float32，像素归一化到 [0,1]）；换 shape 时重新分配。
  void load_images(const std::vector<std::uint32_t>& idx, Tensor& out) const;
};

// 校验文件头/meta/标签后返回；任何不自洽抛 std::runtime_error。
UavrDataset load_uavr(const std::string& path);  // 等价于 load_uavr(path, LoadOptions{})
UavrDataset load_uavr(const std::string& path, const LoadOptions& opts);

}  // namespace uavrgb
