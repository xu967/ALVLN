#pragma once
// 可控内容的合成数据集夹具（翻转增强、双帧读帧的测试用，交叉审核 F4/F5）。
// 现有夹具给不出这两类测试要的内容：纯噪声图、整图亮度编码类别、侧车固定成 i-1 链。
// 这里先用 write_synthetic_uavr 建好文件头与 meta，再原地改写像素区与标签区，
// 最后写一份由调用方逐项给定的 UAVRAUX1 侧车（它绑定 uavr 的 SHA256，必须最后写）。
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/sha256.hpp"

namespace uavrgb_test {

using Plane = std::vector<std::uint8_t>;  // 一个样本的 CHW 像素，长 3*h*w

// 第 i 个样本的像素改成 planes[i]，标签区改成 labels；meta 不动（样本数、episode 划分沿用建文件时的）。
inline void overwrite_samples(const std::string& path, int h, int w, const std::vector<Plane>& planes,
                              const std::vector<std::uint8_t>& labels) {
  const std::uint64_t plane_bytes = 3ull * h * w;
  if (planes.size() != labels.size()) throw std::invalid_argument("overwrite_samples: 样本数不一致");
  std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
  if (!f) throw std::runtime_error("overwrite_samples: 无法打开 " + path);
  for (std::size_t i = 0; i < planes.size(); ++i) {
    if (planes[i].size() != plane_bytes) throw std::invalid_argument("overwrite_samples: 平面长度");
    f.seekp(static_cast<std::streamoff>(48 + i * plane_bytes));
    f.write(reinterpret_cast<const char*>(planes[i].data()), static_cast<std::streamsize>(plane_bytes));
  }
  f.seekp(static_cast<std::streamoff>(48 + planes.size() * plane_bytes));
  f.write(reinterpret_cast<const char*>(labels.data()), static_cast<std::streamsize>(labels.size()));
  if (!f) throw std::runtime_error("overwrite_samples: 写入失败 " + path);
}

// 整幅同一灰度（0..255）。
inline Plane flat_plane(int h, int w, std::uint8_t v) { return Plane(3ull * h * w, v); }

// 像素坐标里的半开区域 [r0,r1) x [c0,c1)。
struct Region {
  int r0, r1, c0, c1;
};
inline Region half_region(int h, int w, bool left) { return {0, h, left ? 0 : w / 2, left ? w / 2 : w}; }
inline Region quad_region(int h, int w, bool top, bool left) {
  return {top ? 0 : h / 2, top ? h / 2 : h, left ? 0 : w / 2, left ? w / 2 : w};
}

// 区域内亮 230、区域外暗 25，三个通道相同。
inline Plane region_plane(int h, int w, const Region& r) {
  Plane p(3ull * h * w);
  for (int ch = 0; ch < 3; ++ch)
    for (int y = 0; y < h; ++y)
      for (int x = 0; x < w; ++x) {
        const bool in = y >= r.r0 && y < r.r1 && x >= r.c0 && x < r.c1;
        p[(static_cast<std::size_t>(ch) * h + y) * w + x] = in ? 230 : 25;
      }
  return p;
}

// 与 region_plane 配套的深度分带目标（h/8 x w/8 格，行优先）：区域内的格 = near_band，其余 = far_band。
inline Plane region_depth(int h, int w, const Region& r, std::uint8_t near_band, std::uint8_t far_band) {
  const int dh = h / 8, dw = w / 8;
  Plane t(static_cast<std::size_t>(dh) * dw);
  for (int gy = 0; gy < dh; ++gy)
    for (int gx = 0; gx < dw; ++gx) {
      const bool in = gy * 8 >= r.r0 && gy * 8 < r.r1 && gx * 8 >= r.c0 && gx * 8 < r.c1;
      t[static_cast<std::size_t>(gy) * dw + gx] = in ? near_band : far_band;
    }
  return t;
}

// UAVRAUX1 侧车（tools/DATASET_FORMAT.md §6，写到 uavr 同名 .aux）。prev_index、prev_action 各 n 项；
// depth 非空时长 n*(h/8)*(w/8)，写进深度段，否则 flags=0。分带上界与 preprocess.py 相同。
inline void write_aux_explicit(const std::string& uavr_path, int n, int h, int w,
                               const std::vector<std::uint32_t>& prev_index,
                               const std::vector<std::uint8_t>& prev_action,
                               const std::vector<std::uint8_t>& depth) {
  if (static_cast<int>(prev_index.size()) != n || static_cast<int>(prev_action.size()) != n)
    throw std::invalid_argument("write_aux_explicit: prev 序列长度");
  const bool has_depth = !depth.empty();
  const int dh = h / 8, dw = w / 8;
  if (has_depth && depth.size() != static_cast<std::size_t>(n) * dh * dw)
    throw std::invalid_argument("write_aux_explicit: depth 长度");
  std::vector<std::uint8_t> payload;
  for (std::uint32_t v : prev_index)
    for (int b = 0; b < 4; ++b) payload.push_back(static_cast<std::uint8_t>((v >> (8 * b)) & 0xff));
  payload.insert(payload.end(), prev_action.begin(), prev_action.end());
  payload.insert(payload.end(), depth.begin(), depth.end());
  const uavrgb::Sha256 uavr_sha = uavrgb::sha256_file(uavr_path);
  std::istringstream payload_in(std::string(payload.begin(), payload.end()));
  const uavrgb::Sha256 payload_sha = uavrgb::sha256(payload_in);
  std::ofstream f(uavr_path.substr(0, uavr_path.rfind('.')) + ".aux", std::ios::binary | std::ios::trunc);
  f.write("UAVRAUX1", 8);
  put_u64(f, static_cast<std::uint64_t>(n));
  put_u32(f, has_depth ? 1u : 0u);
  put_u16(f, static_cast<std::uint16_t>(has_depth ? dh : 0));
  put_u16(f, static_cast<std::uint16_t>(has_depth ? dw : 0));
  f.put(8);
  f.put(static_cast<char>(255));
  put_u16(f, 0);
  f.write(reinterpret_cast<const char*>(uavr_sha.data()), 32);
  f.write(reinterpret_cast<const char*>(payload_sha.data()), 32);
  const float edges[7] = {0.3f, 0.5f, 0.8f, 1.2f, 1.8f, 2.5f, 4.0f};
  f.write(reinterpret_cast<const char*>(edges), sizeof edges);
  const float r_env = 0.102f;
  f.write(reinterpret_cast<const char*>(&r_env), 4);
  put_u32(f, 0);
  f.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
  if (!f) throw std::runtime_error("write_aux_explicit: 写入失败");
}

}  // namespace uavrgb_test
