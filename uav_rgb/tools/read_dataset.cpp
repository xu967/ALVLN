// 独立数据集读取/校验工具（不接入 src/CMakeLists.txt，不依赖 uavrgb 库）。
// 读取 tools/preprocess.py 产出的 UAVR 文件；H/W/C 全从文件头读取，不假定 128x128。
//
// 编译：
//   g++ -std=c++17 -O2 -Wall -Wextra tools/read_dataset.cpp -o /tmp/read_dataset
// 用法：
//   /tmp/read_dataset <dataset.uavr>                 仅校验结构并打印头/meta
//   /tmp/read_dataset <dataset.uavr> --selfcheck     额外校验 make_synthetic_episode.py
//                                                    生成的固定 15 样本数据集（像素/标签）
// --selfcheck 约定：ep1(scene 42,train,6步) ep2(scene 42,train,4步) ep3(scene 43,val,5步)，
// 标签 = episode 内 step%4；偶数全局样本为常量色帧，奇数为 R→x/G→y/B=77 梯度帧。
// 任意 --size 预处理结果都可跑 --selfcheck（尺寸取自文件头）。
// 同目录若有 dataset.aux（UAVRAUX1 侧车，DATASET_FORMAT.md §6）一并校验：头/长度/两个 SHA256/
// 取值范围；--selfcheck 再核对合成集的上一帧链与 prev-action（首帧指向自身、prev=SAFE）。
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// 最小 SHA-256（FIPS 180-4），只为校验侧车绑定；本工具不链接 uavrgb 库。
std::array<std::uint8_t, 32> sha256(const std::uint8_t* data, std::uint64_t len) {
  static const std::uint32_t k[64] = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
      0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
      0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
      0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
      0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
      0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
      0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
  std::uint32_t h[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::vector<std::uint8_t> msg(data, data + len);
  msg.push_back(0x80);
  while (msg.size() % 64 != 56) msg.push_back(0);
  const std::uint64_t bits = len * 8;
  for (int i = 7; i >= 0; --i) msg.push_back(static_cast<std::uint8_t>(bits >> (8 * i)));
  auto rotr = [](std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
  for (std::size_t off = 0; off < msg.size(); off += 64) {
    std::uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(msg[off + 4 * i]) << 24) |
             (static_cast<std::uint32_t>(msg[off + 4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(msg[off + 4 * i + 2]) << 8) | msg[off + 4 * i + 3];
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
      const std::uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
      hh = g, g = f, f = e, e = d + t1, d = c, c = b, b = a, a = t1 + t2;
    }
    h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e, h[5] += f, h[6] += g, h[7] += hh;
  }
  std::array<std::uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 4; ++j) out[4 * i + j] = static_cast<std::uint8_t>(h[i] >> (24 - 8 * j));
  }
  return out;
}

struct Header {
  std::uint16_t version = 0;
  std::uint16_t header_size = 0;
  std::uint32_t n = 0;
  std::uint16_t h = 0;
  std::uint16_t w = 0;
  std::uint8_t c = 0;
  std::uint8_t dtype = 0;
  std::uint8_t label_dtype = 0;
  std::uint8_t reserved = 0;
  std::uint64_t images_off = 0;
  std::uint64_t labels_off = 0;
  std::uint64_t meta_off = 0;
  std::uint32_t meta_len = 0;
};

std::uint16_t le16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
std::uint32_t le32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) |
         (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) |
         (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint64_t le64(const std::uint8_t* p) {
  return static_cast<std::uint64_t>(le32(p)) |
         (static_cast<std::uint64_t>(le32(p + 4)) << 32);
}

bool read_at(std::ifstream& f, std::uint64_t off, void* dst, std::size_t n) {
  f.clear();
  f.seekg(static_cast<std::streamoff>(off), std::ios::beg);
  if (!f) return false;
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
  return static_cast<std::size_t>(f.gcount()) == n;
}

bool contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}

int fail(const std::string& msg) {
  std::printf("FAIL: %s\nRESULT: FAIL\n", msg.c_str());
  return 1;
}

int fail_at(std::uint32_t sample, std::uint32_t x, std::uint32_t y, const std::string& msg) {
  std::printf("FAIL: sample=%u x=%u y=%u: %s\nRESULT: FAIL\n",
              sample, x, y, msg.c_str());
  return 1;
}

// 常量色帧期望：与 tools/make_synthetic_episode.py 保持一致。
void expected_constant(std::uint32_t g, int& r, int& gg, int& b) {
  r = 40 + static_cast<int>((g * 37u) % 180u);
  gg = 40 + static_cast<int>((g * 53u) % 180u);
  b = 40 + static_cast<int>((g * 71u) % 180u);
}

int selfcheck(std::ifstream& f, const Header& hd, const std::string& meta,
              std::uint64_t file_size) {
  constexpr std::uint32_t kExpectN = 15;
  const std::uint32_t ep_starts[3] = {0, 6, 10};
  const std::uint32_t ep_counts[3] = {6, 4, 5};
  const std::uint64_t sample_bytes = static_cast<std::uint64_t>(hd.c) * hd.h * hd.w;

  if (hd.n != kExpectN) {
    return fail("--selfcheck 只适用于合成数据集：n=" + std::to_string(hd.n) +
                "，期望 15");
  }
  std::vector<std::uint8_t> labels(hd.n);
  if (!read_at(f, hd.labels_off, labels.data(), labels.size())) {
    return fail("读取 labels 失败");
  }
  std::uint32_t label_ok = 0;
  for (std::uint32_t g = 0; g < kExpectN; ++g) {
    std::uint32_t local = 0;
    for (int e = 0; e < 3; ++e) {
      if (g >= ep_starts[e] && g < ep_starts[e] + ep_counts[e]) local = g - ep_starts[e];
    }
    if (labels[g] == local % 4u) {
      ++label_ok;
    } else {
      std::printf("FAIL: sample=%u label=%u 期望 %u\n", g, labels[g], local % 4u);
    }
  }
  if (label_ok != kExpectN) return fail("标签不符");
  std::printf("selfcheck: labels %u/%u OK\n", label_ok, kExpectN);

  std::vector<std::uint8_t> buf(sample_bytes);
  const std::size_t chan = static_cast<std::size_t>(hd.h) * hd.w;
  std::uint32_t even_ok = 0, odd_ok = 0;

  for (std::uint32_t g = 0; g < kExpectN; ++g) {
    if (!read_at(f, hd.images_off + g * sample_bytes, buf.data(), buf.size())) {
      return fail("读取 sample " + std::to_string(g) + " 像素失败");
    }
    if (g % 2 == 0) {
      int r, gg, b;
      expected_constant(g, r, gg, b);
      for (std::uint32_t y = 0; y < hd.h; ++y) {
        for (std::uint32_t x = 0; x < hd.w; ++x) {
          const std::size_t i = static_cast<std::size_t>(y) * hd.w + x;
          if (buf[i] != r || buf[chan + i] != gg || buf[2 * chan + i] != b) {
            return fail_at(g, x, y, "常量帧像素/通道顺序不符");
          }
        }
      }
      ++even_ok;
    } else {
      // B 通道恒 77；R 只随 x 变且非降；G 只随 y 变且非降（验证 NCHW 与宽高方向）。
      for (std::size_t i = 0; i < chan; ++i) {
        if (buf[2 * chan + i] != 77) {
          return fail("sample=" + std::to_string(g) + " 梯度帧 B 通道 != 77");
        }
      }
      for (std::uint32_t y = 0; y < hd.h; ++y) {
        const std::size_t row = static_cast<std::size_t>(y) * hd.w;
        for (std::uint32_t x = 0; x < hd.w; ++x) {
          if (buf[row + x] != buf[x]) {
            return fail_at(g, x, y, "R 通道随 y 变化（布局错）");
          }
          if (x + 1 < hd.w && buf[row + x] > buf[row + x + 1]) {
            return fail_at(g, x, y, "R 通道未随 x 单调递增");
          }
          if (buf[chan + row + x] != buf[chan + row]) {
            return fail_at(g, x, y, "G 通道随 x 变化（布局错）");
          }
        }
        if (y + 1 < hd.h && buf[chan + row] > buf[chan + row + hd.w]) {
          return fail_at(g, 0, y, "G 通道未随 y 单调递增");
        }
      }
      if (buf[hd.w - 1] - buf[0] < 100 ||
          buf[chan + static_cast<std::size_t>(hd.h - 1) * hd.w] - buf[chan] < 100) {
        return fail("sample=" + std::to_string(g) + " 梯度动态范围异常");
      }
      ++odd_ok;
    }
  }
  std::printf("selfcheck: constant frames %u/8 OK (逐像素精确, RGB 顺序)\n", even_ok);
  std::printf("selfcheck: gradient frames %u/7 OK (R->x, G->y, B=77, NCHW)\n", odd_ok);

  // meta 关键字段（紧凑 JSON，preprocess.py 固定分隔符）。
  const std::vector<std::string> needles = {
      "\"schema\":\"uavrgb.dataset.v1\"",
      "\"layout\":\"NCHW\"",
      "\"height\":" + std::to_string(hd.h),
      "\"width\":" + std::to_string(hd.w),
      "\"scene_id\":42",
      "\"scene_id\":43",
      "\"split\":\"train\"",
      "\"split\":\"val\"",
      "\"train\":10",
      "\"val\":5",
      "\"num_samples\":15",
  };
  for (const auto& needle : needles) {
    if (!contains(meta, needle)) {
      return fail("meta 缺片段: " + needle);
    }
  }
  if (file_size != hd.meta_off + hd.meta_len) {
    return fail("文件长度与 meta 末尾不一致");
  }
  std::printf("selfcheck: meta 字段/文件长度 OK\n");
  return 0;
}

// 侧车：uavr 路径换后缀 .aux；不存在返回 0（旧数据集没有侧车）。
int check_aux(const std::string& uavr_path, const Header& hd, bool do_selfcheck) {
  std::string aux_path = uavr_path;
  const std::size_t dot = aux_path.rfind('.');
  const std::size_t slash = aux_path.rfind('/');
  if (dot != std::string::npos && (slash == std::string::npos || dot > slash)) aux_path.erase(dot);
  aux_path += ".aux";
  std::ifstream af(aux_path, std::ios::binary);
  if (!af) {
    std::printf("aux: 无 %s（旧数据集；双帧/prev-action/深度分带不可用）\n", aux_path.c_str());
    return 0;
  }
  std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(af)), std::istreambuf_iterator<char>());
  if (bytes.size() < 128 || std::memcmp(bytes.data(), "UAVRAUX1", 8) != 0) return fail("aux: 头不足 128 B 或 magic 不是 UAVRAUX1");
  const std::uint64_t n = le64(bytes.data() + 8);
  const std::uint32_t flags = le32(bytes.data() + 16);
  const std::uint16_t dh = le16(bytes.data() + 20), dw = le16(bytes.data() + 22);
  const std::uint8_t classes = bytes[24], unknown = bytes[25];
  if (n != hd.n) return fail("aux: N=" + std::to_string(n) + " != uavr n=" + std::to_string(hd.n));
  if (flags > 1) return fail("aux: 未知 flags");
  if (unknown != 255) return fail("aux: unknown_code != 255");
  const bool has_depth = (flags & 1u) != 0;
  if (has_depth && (dh != hd.h / 8 || dw != hd.w / 8 || classes != 8)) return fail("aux: 深度网格/分带数与 H/8×W/8、8 带不符");
  if (!has_depth && (dh != 0 || dw != 0)) return fail("aux: 无深度时 depth_h/depth_w 须为 0");
  const std::uint64_t depth_bytes = has_depth ? n * dh * dw : 0;
  if (bytes.size() != 128 + 5 * n + depth_bytes) return fail("aux: 文件长度与 N/网格不符");
  // 绑定：头里的 dataset.uavr SHA256 与 payload SHA256 都要对得上，防止「新数据集 + 旧侧车」。
  std::ifstream uf(uavr_path, std::ios::binary);
  std::vector<std::uint8_t> ubytes((std::istreambuf_iterator<char>(uf)), std::istreambuf_iterator<char>());
  if (sha256(ubytes.data(), ubytes.size()) != *reinterpret_cast<const std::array<std::uint8_t, 32>*>(bytes.data() + 28)) {
    return fail("aux: UAVR SHA256 不匹配（侧车不是这份 dataset.uavr 的）");
  }
  if (sha256(bytes.data() + 128, bytes.size() - 128) != *reinterpret_cast<const std::array<std::uint8_t, 32>*>(bytes.data() + 60)) {
    return fail("aux: payload SHA256 不匹配");
  }
  const std::uint8_t* prev_idx = bytes.data() + 128;
  const std::uint8_t* prev_act = prev_idx + 4 * n;
  const std::uint8_t* depth = prev_act + n;
  std::uint64_t self_count = 0;
  for (std::uint64_t i = 0; i < n; ++i) {
    const std::uint32_t p = le32(prev_idx + 4 * i);
    if (p >= n) return fail("aux: prev_index 越界");
    if (prev_act[i] >= 4) return fail("aux: prev_action 越界");
    if (p == i) ++self_count;
  }
  std::uint64_t unknown_cells = 0;
  for (std::uint64_t i = 0; i < depth_bytes; ++i) {
    if (depth[i] == 255) ++unknown_cells;
    else if (depth[i] >= classes) return fail("aux: 深度分带目标越界");
  }
  std::printf("aux: %s N=%llu first/self=%llu depth=%s\n", aux_path.c_str(),
              static_cast<unsigned long long>(n), static_cast<unsigned long long>(self_count),
              has_depth ? ("grid " + std::to_string(dh) + "x" + std::to_string(dw) + " unknown " +
                           std::to_string(unknown_cells) + "/" + std::to_string(depth_bytes)).c_str()
                        : "none");
  if (!do_selfcheck) return 0;
  // 合成集：首帧 0/6/10 指向自身且 prev=SAFE(0)；其余指向 i-1，prev = 上一步标签（action_executed=label）。
  for (std::uint32_t i = 0; i < n; ++i) {
    const bool first = i == 0 || i == 6 || i == 10;
    const std::uint32_t want_idx = first ? i : i - 1;
    const std::uint32_t local_prev = first ? 0 : (i - 1) - (i - 1 >= 10 ? 10 : (i - 1 >= 6 ? 6 : 0));
    const std::uint8_t want_act = first ? 0 : static_cast<std::uint8_t>(local_prev % 4u);
    if (le32(prev_idx + 4 * i) != want_idx || prev_act[i] != want_act) {
      return fail("aux selfcheck: sample " + std::to_string(i) + " prev_index/prev_action 不符");
    }
  }
  if (has_depth) return fail("aux selfcheck: 合成集没有深度，侧车却带 depth 段");
  std::printf("selfcheck: aux prev chain %u/%u OK (首帧 0/6/10 指向自身, prev=SAFE)\n", hd.n, hd.n);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::strcmp(argv[2], "--selfcheck") != 0)) {
    std::printf("用法: %s <dataset.uavr> [--selfcheck]\n", argv[0]);
    return 2;
  }
  const bool do_selfcheck = argc == 3;
  std::ifstream f(argv[1], std::ios::binary);
  if (!f) return fail(std::string("打不开 ") + argv[1]);
  f.seekg(0, std::ios::end);
  const std::uint64_t file_size = static_cast<std::uint64_t>(f.tellg());
  f.seekg(0, std::ios::beg);

  std::uint8_t raw[48];
  if (!read_at(f, 0, raw, sizeof(raw))) return fail("文件短于 48 字节头");
  if (std::memcmp(raw, "UAVR", 4) != 0) return fail("magic 不是 UAVR");

  Header hd;
  hd.version = le16(raw + 4);
  hd.header_size = le16(raw + 6);
  hd.n = le32(raw + 8);
  hd.h = le16(raw + 12);
  hd.w = le16(raw + 14);
  hd.c = raw[16];
  hd.dtype = raw[17];
  hd.label_dtype = raw[18];
  hd.reserved = raw[19];
  hd.images_off = le64(raw + 20);
  hd.labels_off = le64(raw + 28);
  hd.meta_off = le64(raw + 36);
  hd.meta_len = le32(raw + 44);

  if (hd.version != 1) return fail("version=" + std::to_string(hd.version) + " 不支持");
  if (hd.header_size != 48) return fail("header_size != 48");
  if (hd.dtype != 1 || hd.label_dtype != 1) return fail("dtype/label_dtype 必须为 1(uint8)");
  if (hd.c != 3) return fail("channels != 3");
  if (hd.h == 0 || hd.w == 0) return fail("H/W 为 0");
  if (hd.n == 0) return fail("N 为 0");
  const std::uint64_t images_bytes =
      static_cast<std::uint64_t>(hd.n) * hd.c * hd.h * hd.w;
  if (hd.images_off != 48) return fail("images_offset != 48");
  if (hd.labels_off != hd.images_off + images_bytes) return fail("labels_offset 不一致");
  if (hd.meta_off != hd.labels_off + hd.n) return fail("meta_offset 不一致");
  if (hd.meta_len == 0 || file_size != hd.meta_off + hd.meta_len) {
    return fail("meta 长度/文件总长不一致");
  }

  std::vector<std::uint8_t> labels(hd.n);
  if (!read_at(f, hd.labels_off, labels.data(), labels.size())) return fail("读 labels 失败");
  std::uint32_t hist[4] = {0, 0, 0, 0};
  for (std::uint8_t v : labels) {
    if (v >= 4) return fail("标签 " + std::to_string(v) + " 超出 0..3");
    ++hist[v];
  }

  std::vector<char> meta_buf(hd.meta_len);
  if (!read_at(f, hd.meta_off, meta_buf.data(), meta_buf.size())) return fail("读 meta 失败");
  const std::string meta(meta_buf.begin(), meta_buf.end());
  if (meta.front() != '{' || meta.back() != '}') return fail("meta 不是 JSON 对象");

  std::printf("file: %s (%llu bytes)\n", argv[1],
              static_cast<unsigned long long>(file_size));
  std::printf("header: magic=UAVR version=1 header_size=48 n=%u h=%u w=%u c=%u "
              "images=uint8 labels=uint8\n", hd.n, hd.h, hd.w, hd.c);
  std::printf("offsets: images=%llu labels=%llu meta=%llu meta_len=%u\n",
              static_cast<unsigned long long>(hd.images_off),
              static_cast<unsigned long long>(hd.labels_off),
              static_cast<unsigned long long>(hd.meta_off), hd.meta_len);
  std::printf("labels: 0=%u 1=%u 2=%u 3=%u\n", hist[0], hist[1], hist[2], hist[3]);
  std::printf("meta: %s\n", meta.c_str());
  std::printf("OK: header/labels/meta structure valid\n");
  if (const int rc = check_aux(argv[1], hd, do_selfcheck); rc != 0) return rc;

  if (do_selfcheck) {
    const int rc = selfcheck(f, hd, meta, file_size);
    if (rc != 0) return rc;
    std::printf("RESULT: PASS (structure + synthetic pixel/label checks)\n");
  } else {
    std::printf("RESULT: PASS (structure only; 加 --selfcheck 做像素级校验)\n");
  }
  return 0;
}
