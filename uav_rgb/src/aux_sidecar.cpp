// UAVRAUX1 侧车读取（布局见 tools/DATASET_FORMAT.md §6；写端 tools/preprocess.py）。
#include "uavrgb/aux_sidecar.hpp"

#include "uavrgb/sha256.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace uavrgb {
namespace {

[[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("dataset.aux: " + msg); }

std::uint64_t le(const unsigned char* p, int bytes) {
  std::uint64_t v = 0;
  for (int i = bytes - 1; i >= 0; --i) v = (v << 8) | p[i];
  return v;
}

float le_f32(const unsigned char* p) {
  const std::uint32_t u = static_cast<std::uint32_t>(le(p, 4));
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

}  // namespace

std::string aux_sidecar_path(const std::string& uavr_path) {
  return std::filesystem::path(uavr_path).replace_extension(".aux").string();
}

AuxSidecar load_aux_sidecar(const std::string& uavr_path, int n, int h, int w,
                            std::uint32_t num_classes, const Sha256* uavr_sha256) {
  AuxSidecar aux;
  const std::string path = aux_sidecar_path(uavr_path);
  std::error_code ec;
  if (!std::filesystem::exists(path, ec)) return aux;
  std::ifstream f(path, std::ios::binary);
  if (!f) fail("无法打开 " + path);
  unsigned char hb[kAuxHeaderBytes];
  if (!f.read(reinterpret_cast<char*>(hb), sizeof hb)) fail("文件头不足 128 B");
  if (std::memcmp(hb, "UAVRAUX1", 8) != 0) fail("magic 不是 UAVRAUX1");
  if (le(hb + 8, 8) != static_cast<std::uint64_t>(n)) fail("N 与 dataset.uavr 不符");
  const std::uint32_t flags = static_cast<std::uint32_t>(le(hb + 16, 4));
  if (flags > 1) fail("未知 flags（只定义了 bit0=深度目标）");
  aux.has_depth = (flags & 1u) != 0;
  aux.depth_h = static_cast<int>(le(hb + 20, 2));
  aux.depth_w = static_cast<int>(le(hb + 22, 2));
  aux.depth_classes = hb[24];
  if (hb[25] != kAuxUnknown) fail("unknown_code 必须是 255");
  if (aux.has_depth) {
    if (aux.depth_h != h / 8 || aux.depth_w != w / 8) fail("深度目标网格不是 H/8 × W/8");
    if (aux.depth_classes < 2 || aux.depth_classes > 254) fail("depth_classes 须在 2..254");
  } else if (aux.depth_h != 0 || aux.depth_w != 0) {
    fail("无深度目标时 depth_h/depth_w 必须为 0");
  }
  const Sha256 uavr_sha = uavr_sha256 ? *uavr_sha256 : sha256_file(uavr_path);
  if (!std::equal(uavr_sha.begin(), uavr_sha.end(), hb + 28)) {
    fail("UAVR SHA256 不匹配：侧车不是从这份 dataset.uavr 生成的，请重新 preprocess");
  }
  for (int i = 0; i < kAuxBandEdges; ++i) aux.band_edges.push_back(le_f32(hb + 92 + 4 * i));
  for (int i = 0; i < kAuxBandEdges; ++i) {
    const float e = aux.band_edges[static_cast<std::size_t>(i)];
    if (!std::isfinite(e) || e <= 0.0f || (i > 0 && e <= aux.band_edges[static_cast<std::size_t>(i) - 1])) {
      fail("分带上界必须为正、有限、严格递增");
    }
  }
  if (aux.has_depth && aux.depth_classes != kAuxBandEdges + 1) fail("depth_classes 须等于上界数+1");
  aux.envelope_radius = le_f32(hb + 120);
  if (!std::isfinite(aux.envelope_radius) || aux.envelope_radius < 0.0f) fail("包络半径非法");

  const std::uint64_t un = static_cast<std::uint64_t>(n);
  const std::uint64_t depth_bytes = aux.has_depth ? un * aux.depth_h * aux.depth_w : 0;
  const std::uint64_t expect = kAuxHeaderBytes + 5 * un + depth_bytes;
  const std::uint64_t actual = std::filesystem::file_size(path, ec);
  if (ec) fail("取文件大小失败");
  if (actual != expect) {
    fail("文件长 " + std::to_string(actual) + " != 期望 " + std::to_string(expect));
  }
  const Sha256 payload_sha = sha256(f);
  if (!std::equal(payload_sha.begin(), payload_sha.end(), hb + 60)) fail("payload SHA256 不匹配");
  f.clear();
  f.seekg(kAuxHeaderBytes);
  aux.prev_index.resize(static_cast<std::size_t>(n));
  aux.prev_action.resize(static_cast<std::size_t>(n));
  aux.depth.resize(static_cast<std::size_t>(depth_bytes));
  if (!f.read(reinterpret_cast<char*>(aux.prev_index.data()), 4 * static_cast<std::streamsize>(n)) ||
      !f.read(reinterpret_cast<char*>(aux.prev_action.data()), static_cast<std::streamsize>(n)) ||
      (depth_bytes != 0 &&
       !f.read(reinterpret_cast<char*>(aux.depth.data()), static_cast<std::streamsize>(depth_bytes)))) {
    fail("读取 payload 失败");
  }
  static_assert(sizeof(std::uint32_t) == 4, "prev_index 按小端 u32 落盘");
  const std::uint16_t endian = 1;
  if (*reinterpret_cast<const unsigned char*>(&endian) != 1) fail("仅支持小端 host");
  aux.first_frame.resize(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < aux.prev_index.size(); ++i) {
    if (aux.prev_index[i] >= static_cast<std::uint32_t>(n)) fail("prev_index 越界");
    if (aux.prev_action[i] >= num_classes) fail("prev_action 越界");
    aux.first_frame[i] = aux.prev_index[i] == i ? 1 : 0;
  }
  for (std::uint8_t v : aux.depth) {
    if (v != kAuxUnknown && v >= aux.depth_classes) fail("深度分带目标越界");
  }
  aux.present = true;
  return aux;
}

}  // namespace uavrgb
