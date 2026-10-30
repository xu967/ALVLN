#pragma once
// 测试用合成 UAVR 数据集写入器（严格按 tools/DATASET_FORMAT.md v1 的 48 B 小端文件头）。
// 样本 = 纯均匀随机噪声图 + 随机标签。图像与标签之间没有真实规律，模型只能靠记忆拟合，
// 因此“训练集 loss/准确率上去”只能来自梯度链路真的通了（记忆型过拟合测试）。
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb_test {

struct SynthDataset {
  std::string path;
  int n = 0;
  int c = 3;
  int h = 0;
  int w = 0;
};

inline void put_u16(std::ostream& f, std::uint16_t v) {
  const char b[2] = {static_cast<char>(v & 0xff), static_cast<char>((v >> 8) & 0xff)};
  f.write(b, 2);
}
inline void put_u32(std::ostream& f, std::uint32_t v) {
  for (int i = 0; i < 4; ++i) f.put(static_cast<char>((v >> (8 * i)) & 0xff));
}
inline void put_u64(std::ostream& f, std::uint64_t v) {
  for (int i = 0; i < 8; ++i) f.put(static_cast<char>((v >> (8 * i)) & 0xff));
}

// n 个样本，前 train_n 个归 train episode，接着 test_n 个归 test episode，
// 其余归 val episode（val_n/test_n 为 0 时不写对应 episode）。
inline SynthDataset write_synthetic_uavr(const std::string& path, int n, int h, int w,
                                         int train_n, std::uint32_t seed, int test_n = 0) {
  if (n <= 0 || h <= 0 || w <= 0 || train_n < 0 || train_n > n || test_n < 0 ||
      train_n + test_n > n) {
    throw std::invalid_argument("write_synthetic_uavr: 参数非法");
  }
  const int val_n = n - train_n - test_n;
  const std::uint64_t plane = static_cast<std::uint64_t>(3) * h * w;
  const std::uint64_t images_off = 48;
  const std::uint64_t labels_off = images_off + static_cast<std::uint64_t>(n) * plane;
  const std::uint64_t meta_off = labels_off + static_cast<std::uint64_t>(n);

  std::string meta =
      "{\"schema\":\"uavrgb.dataset.v1\",\"images\":{\"layout\":\"NCHW\",\"dtype\":\"uint8\","
      "\"height\":" + std::to_string(h) + ",\"width\":" + std::to_string(w) +
      ",\"channels\":3},\"labels\":{\"dtype\":\"uint8\",\"num_classes\":4},"
      "\"split_policy\":\"scene_id\",\"num_samples\":" + std::to_string(n) +
      ",\"episodes\":[";
  auto episode = [&](int id, int scene, const char* split, int start, int count) {
    return std::string("{\"episode_id\":") + std::to_string(id) + ",\"scene_id\":" +
           std::to_string(scene) + ",\"seed\":" + std::to_string(seed) +
           ",\"version\":\"synthetic\",\"split\":\"" + split + "\",\"num_steps\":" +
           std::to_string(count) + ",\"sample_start\":" + std::to_string(start) +
           ",\"sample_count\":" + std::to_string(count) + ",\"source_dir\":\"/tmp\"}";
  };
  meta += episode(1, 1, "train", 0, train_n);
  if (val_n > 0) meta += "," + episode(2, 2, "val", train_n, val_n);
  if (test_n > 0) meta += "," + episode(3, 3, "test", train_n + val_n, test_n);
  meta += "]}";

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("write_synthetic_uavr: 无法写 " + path);
  f.write("UAVR", 4);
  put_u16(f, 1);
  put_u16(f, 48);
  put_u32(f, static_cast<std::uint32_t>(n));
  put_u16(f, static_cast<std::uint16_t>(h));
  put_u16(f, static_cast<std::uint16_t>(w));
  f.put(3);
  f.put(1);
  f.put(1);
  f.put(0);
  put_u64(f, images_off);
  put_u64(f, labels_off);
  put_u64(f, meta_off);
  put_u32(f, static_cast<std::uint32_t>(meta.size()));

  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> noise(0, 255);
  std::vector<std::uint8_t> labels(static_cast<std::size_t>(n));
  std::vector<std::uint8_t> img(static_cast<std::size_t>(plane));
  for (int i = 0; i < n; ++i) {
    labels[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(rng() % 4);
    for (std::size_t j = 0; j < img.size(); ++j) {
      img[j] = static_cast<std::uint8_t>(noise(rng));
    }
    f.write(reinterpret_cast<const char*>(img.data()),
            static_cast<std::streamsize>(img.size()));
  }
  f.write(reinterpret_cast<const char*>(labels.data()),
          static_cast<std::streamsize>(labels.size()));
  f.write(meta.data(), static_cast<std::streamsize>(meta.size()));
  if (!f) throw std::runtime_error("write_synthetic_uavr: 写入失败 " + path);
  return SynthDataset{path, n, 3, h, w};
}

// 类别条件合成集（不平衡实验/碰撞过滤测试用）：
//   每张图的整图亮度 ~ N(means[label]·255, sigma_img·255)，再叠加逐像素 N(0, sigma_px·255)
//   噪声并截断到 0..255。两类均值重叠时，类别先验（90/10）会把无权 CE 的决策边界推向
//   多数类，这正是本实验要复现的现象。labels 逐样本给定（train 在前、val 在后），
//   collision 非空时长度须 = train+val，写进 meta.collision_flags（1 = 碰撞步）。
inline SynthDataset write_class_conditional_uavr(
    const std::string& path, int h, int w, const std::vector<int>& train_labels,
    const std::vector<int>& val_labels, const std::vector<float>& means, float sigma_img,
    float sigma_px, std::uint32_t seed, const std::vector<std::uint8_t>& collision = {}) {
  const int train_n = static_cast<int>(train_labels.size());
  const int total = train_n + static_cast<int>(val_labels.size());
  const int ncls = static_cast<int>(means.size());
  if (total <= 0 || train_n <= 0 || h <= 0 || w <= 0 || ncls <= 0 || ncls > 4) {
    throw std::invalid_argument("write_class_conditional_uavr: 参数非法");
  }
  if (!collision.empty() && collision.size() != static_cast<std::size_t>(total)) {
    throw std::invalid_argument("write_class_conditional_uavr: collision 长度须为 train+val");
  }
  std::vector<int> labels = train_labels;
  labels.insert(labels.end(), val_labels.begin(), val_labels.end());
  for (int l : labels) {
    if (l < 0 || l >= ncls) throw std::invalid_argument("write_class_conditional_uavr: label 越界");
  }

  const std::uint64_t plane = static_cast<std::uint64_t>(3) * h * w;
  const std::uint64_t images_off = 48;
  const std::uint64_t labels_off = images_off + static_cast<std::uint64_t>(total) * plane;
  const std::uint64_t meta_off = labels_off + static_cast<std::uint64_t>(total);
  std::string flags(static_cast<std::size_t>(total), '0');
  for (int i = 0; i < total; ++i) {
    if (!collision.empty() && collision[static_cast<std::size_t>(i)] != 0) flags[i] = '1';
  }
  std::string meta =
      "{\"schema\":\"uavrgb.dataset.v1\",\"images\":{\"layout\":\"NCHW\",\"dtype\":\"uint8\","
      "\"height\":" + std::to_string(h) + ",\"width\":" + std::to_string(w) +
      ",\"channels\":3},\"labels\":{\"dtype\":\"uint8\",\"num_classes\":4},"
      "\"split_policy\":\"scene_id\",\"collision_flags\":\"" + flags +
      "\",\"num_samples\":" + std::to_string(total) + ",\"episodes\":[";
  auto episode = [&](int id, int scene, const char* split, int start, int count) {
    return std::string("{\"episode_id\":") + std::to_string(id) + ",\"scene_id\":" +
           std::to_string(scene) + ",\"seed\":" + std::to_string(seed) +
           ",\"version\":\"synthetic\",\"split\":\"" + split + "\",\"num_steps\":" +
           std::to_string(count) + ",\"sample_start\":" + std::to_string(start) +
           ",\"sample_count\":" + std::to_string(count) + ",\"source_dir\":\"/tmp\"}";
  };
  meta += episode(1, 1, "train", 0, train_n);
  if (total - train_n > 0) meta += "," + episode(2, 2, "val", train_n, total - train_n);
  meta += "]}";

  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("write_class_conditional_uavr: 无法写 " + path);
  f.write("UAVR", 4);
  put_u16(f, 1);
  put_u16(f, 48);
  put_u32(f, static_cast<std::uint32_t>(total));
  put_u16(f, static_cast<std::uint16_t>(h));
  put_u16(f, static_cast<std::uint16_t>(w));
  f.put(3);
  f.put(1);
  f.put(1);
  f.put(0);
  put_u64(f, images_off);
  put_u64(f, labels_off);
  put_u64(f, meta_off);
  put_u32(f, static_cast<std::uint32_t>(meta.size()));

  std::mt19937 rng(seed);
  std::normal_distribution<float> level_noise(0.0f, sigma_img * 255.0f);
  std::normal_distribution<float> pixel_noise(0.0f, sigma_px * 255.0f);
  std::vector<std::uint8_t> img(static_cast<std::size_t>(plane));
  for (int i = 0; i < total; ++i) {
    const float level = means[static_cast<std::size_t>(labels[static_cast<std::size_t>(i)])] *
                            255.0f +
                        level_noise(rng);
    for (std::size_t j = 0; j < img.size(); ++j) {
      const float v = level + pixel_noise(rng);
      img[j] = static_cast<std::uint8_t>(std::min(255.0f, std::max(0.0f, v)));
    }
    f.write(reinterpret_cast<const char*>(img.data()),
            static_cast<std::streamsize>(img.size()));
  }
  std::vector<std::uint8_t> lab(static_cast<std::size_t>(total));
  for (int i = 0; i < total; ++i) lab[static_cast<std::size_t>(i)] =
      static_cast<std::uint8_t>(labels[static_cast<std::size_t>(i)]);
  f.write(reinterpret_cast<const char*>(lab.data()),
          static_cast<std::streamsize>(lab.size()));
  f.write(meta.data(), static_cast<std::streamsize>(meta.size()));
  if (!f) throw std::runtime_error("write_class_conditional_uavr: 写入失败 " + path);
  return SynthDataset{path, total, 3, h, w};
}

}  // namespace uavrgb_test
