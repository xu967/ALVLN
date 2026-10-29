// UAVR v1 只读实现：48 B 小端头 + N×C×H×W uint8 NCHW + N uint8 标签 + meta JSON。
#include "uavrgb/dataset.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>

namespace uavrgb {
namespace {

std::uint16_t le16(const unsigned char* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}
std::uint32_t le32(const unsigned char* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}
std::uint64_t le64(const unsigned char* p) {
  return static_cast<std::uint64_t>(le32(p)) |
         (static_cast<std::uint64_t>(le32(p + 4)) << 32);
}

[[noreturn]] void fail(const std::string& msg) {
  throw std::runtime_error("UAVR: " + msg);
}

// 读整个区间；长度不足即报错（截断文件不允许静默当成零填充）。
void read_exact(std::ifstream& f, std::uint64_t off, void* dst, std::size_t len,
                const char* what) {
  f.clear();
  f.seekg(static_cast<std::streamoff>(off));
  if (!f) fail(std::string("seek 失败: ") + what);
  f.read(static_cast<char*>(dst), static_cast<std::streamsize>(len));
  if (static_cast<std::size_t>(f.gcount()) != len) fail(std::string("读取不足: ") + what);
}

// ---- meta 最小扫描 ----
// 只取 episodes[] 的 sample_start/sample_count/split；不实现通用 JSON 解析器，
// 字段缺失直接报错（宁可失败也不按错误的 split 训练）。
struct EpisodeSpan {
  std::uint32_t start = 0;
  std::uint32_t count = 0;
  std::string split;
  std::int64_t scene = 0;
};

std::string str_field(const std::string& obj, const char* key) {
  const std::string pat = std::string("\"") + key + "\":\"";
  const std::size_t p = obj.find(pat);
  if (p == std::string::npos) fail(std::string("episodes[] 缺字段 ") + key);
  const std::size_t b = p + pat.size();
  const std::size_t e = obj.find('"', b);
  if (e == std::string::npos) fail(std::string("episodes[] 字段未闭合 ") + key);
  return obj.substr(b, e - b);
}

std::int64_t int_field(const std::string& obj, const char* key) {
  const std::string pat = std::string("\"") + key + "\":";
  const std::size_t p = obj.find(pat);
  if (p == std::string::npos) fail(std::string("episodes[] 缺字段 ") + key);
  const std::size_t b = p + pat.size();
  const std::size_t e = obj.find_first_not_of("0123456789", b + (obj[b] == '-' ? 1 : 0));
  const std::string digits = obj.substr(b, e == std::string::npos ? e : e - b);
  if (digits.empty()) fail(std::string("episodes[] 字段非整数 ") + key);
  if (e != std::string::npos && obj[e] != ',' && obj[e] != '}' && obj[e] != ' ') {
    fail(std::string("episodes[] 字段非整数 ") + key);
  }
  return std::stoll(digits);
}

std::uint32_t num_field(const std::string& obj, const char* key) {
  const auto value = int_field(obj, key);
  if (value < 0 || value > std::numeric_limits<std::uint32_t>::max()) {
    fail(std::string("episodes[] 字段超出 uint32 ") + key);
  }
  return static_cast<std::uint32_t>(value);
}

std::vector<EpisodeSpan> parse_episodes(const std::string& meta) {
  const std::size_t k = meta.find("\"episodes\"");
  if (k == std::string::npos) fail("meta 缺 episodes[]");
  const std::size_t lb = meta.find('[', k);
  if (lb == std::string::npos) fail("meta episodes 不是数组");
  std::vector<EpisodeSpan> out;
  for (std::size_t i = lb + 1; i < meta.size() && meta[i] != ']'; ++i) {
    if (meta[i] != '{') continue;
    const std::size_t e = meta.find('}', i);  // episode 项内无嵌套对象
    if (e == std::string::npos) fail("meta episodes 项未闭合");
    const std::string obj = meta.substr(i + 1, e - i - 1);
    EpisodeSpan s;
    s.start = num_field(obj, "sample_start");
    s.count = num_field(obj, "sample_count");
    s.split = str_field(obj, "split");
    s.scene = int_field(obj, "scene_id");
    out.push_back(std::move(s));
    i = e;
  }
  if (out.empty()) fail("meta episodes[] 为空");
  return out;
}

void assign_split(UavrDataset& ds, const std::vector<EpisodeSpan>& eps) {
  std::vector<bool> assigned(static_cast<std::size_t>(ds.n), false);
  std::map<std::int64_t, std::string> scenes;
  for (const EpisodeSpan& e : eps) {
    const auto inserted = scenes.emplace(e.scene, e.split);
    if (!inserted.second && inserted.first->second != e.split) {
      fail("scene_id 跨 split（scene_id=" + std::to_string(e.scene) + "）");
    }
    if (static_cast<std::uint64_t>(e.start) + e.count > static_cast<std::uint64_t>(ds.n)) {
      fail("episode 区间越界（sample_start=" + std::to_string(e.start) + ", sample_count=" +
           std::to_string(e.count) + ", N=" + std::to_string(ds.n) + "）");
    }
    std::vector<std::uint32_t>* dst = nullptr;
    if (e.split == "train") {
      dst = &ds.train_idx;
    } else if (e.split == "val") {
      dst = &ds.val_idx;
    } else if (e.split == "test") {
      dst = &ds.test_idx;
    } else {
      fail("episode split 非法: " + e.split);
    }
    for (std::uint32_t i = 0; i < e.count; ++i) {
      const auto sample = e.start + i;
      if (assigned[sample]) fail("episode 区间重叠（sample=" + std::to_string(sample) + "）");
      assigned[sample] = true;
      dst->push_back(sample);
    }
  }
  if (std::find(assigned.begin(), assigned.end(), false) != assigned.end()) {
    fail("episode 区间未覆盖全部样本");
  }
}

// meta 追加字段 collision_flags 是 N 个 '0'/'1'（preprocess.py 写），1 = 碰撞步。
// 字段缺失返回空串：只有真要按它过滤时才强制要求。
std::string read_collision_flags(const std::string& meta) {
  const std::string key = "\"collision_flags\":\"";
  const std::size_t p = meta.find(key);
  if (p == std::string::npos) return {};
  const std::size_t b = p + key.size();
  const std::size_t e = meta.find('"', b);
  if (e == std::string::npos) fail("meta collision_flags 未闭合");
  return meta.substr(b, e - b);
}

// 校验并统计 flags 里 '1' 的个数；格式不合规直接报错。
std::size_t count_collisions(const UavrDataset& ds, const std::string& flags) {
  if (flags.size() != static_cast<std::size_t>(ds.n)) {
    fail("meta collision_flags 长度 != N（" + std::to_string(flags.size()) + " vs " +
         std::to_string(ds.n) + "）");
  }
  std::size_t n = 0;
  for (char c : flags) {
    if (c != '0' && c != '1') fail("meta collision_flags 只能含 '0'/'1'");
    if (c == '1') ++n;
  }
  return n;
}

// 没开 --drop-collision 时，专家在撞墙那一步的标签本身就是错的，却会照样进训练集。
// 不能静默——至少把比例打到 stderr。
void warn_collision_kept(const UavrDataset& ds, const std::string& meta) {
  const std::string flags = read_collision_flags(meta);
  if (flags.empty()) return;
  // 这里只做提示：字段格式坏了说明文件可疑，但不该把原本能训练的数据集挡在门外
  // （以前不带 --drop-collision 时这个字段根本不解析）。硬校验留给过滤路径。
  std::size_t n = 0;
  try {
    n = count_collisions(ds, flags);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "警告: %s（不影响训练，只是无法统计碰撞步）\n", e.what());
    return;
  }
  if (n == 0) return;
  std::fprintf(stderr,
               "警告: 数据集中 %zu/%zu 步 collision=true（%.1f%%），专家在这些帧上的标签是错的；"
               "加 --drop-collision 才会剔除\n",
               n, flags.size(),
               100.0 * static_cast<double>(n) / static_cast<double>(flags.size()));
}

// 按全局下标从三个 split 里删掉碰撞步；缺字段直接报错，不静默当"无碰撞"。
void drop_collision_samples(UavrDataset& ds, const std::string& meta) {
  const std::string flags = read_collision_flags(meta);
  if (flags.empty()) {
    fail("meta 缺 collision_flags，无法按碰撞过滤（用新版 tools/preprocess.py 重打包）");
  }
  count_collisions(ds, flags);
  auto drop = [&](std::vector<std::uint32_t>& idx) {
    idx.erase(std::remove_if(idx.begin(), idx.end(),
                             [&](std::uint32_t i) { return flags[i] == '1'; }),
              idx.end());
  };
  drop(ds.train_idx);
  drop(ds.val_idx);
  drop(ds.test_idx);
}

}  // namespace

UavrDataset load_uavr(const std::string& path) { return load_uavr(path, LoadOptions{}); }

UavrDataset load_uavr(const std::string& path, const LoadOptions& opts) {
  UavrDataset ds;
  ds.file.open(path, std::ios::binary);
  if (!ds.file) fail("无法打开 " + path);

  unsigned char hb[48];
  read_exact(ds.file, 0, hb, sizeof hb, "文件头");
  if (std::memcmp(hb, "UAVR", 4) != 0) fail("magic 不是 UAVR");
  if (le16(hb + 4) != 1) fail("version 不是 1");
  if (le16(hb + 6) != 48) fail("header_size 不是 48");
  ds.n = static_cast<int>(le32(hb + 8));
  ds.h = le16(hb + 12);
  ds.w = le16(hb + 14);
  ds.c = hb[16];
  if (ds.c != 3 || hb[17] != 1 || hb[18] != 1) fail("channels/dtype 不是 3/uint8/uint8");
  ds.images_off = le64(hb + 20);
  ds.labels_off = le64(hb + 28);
  const std::uint64_t meta_off = le64(hb + 36);
  const std::uint32_t meta_len = le32(hb + 44);
  if (ds.n <= 0 || ds.h <= 0 || ds.w <= 0) fail("N/H/W 非正");

  const std::uint64_t img_bytes =
      static_cast<std::uint64_t>(ds.n) * ds.c * ds.h * ds.w;
  if (ds.images_off != 48) fail("images_offset 不是 48");
  if (ds.labels_off != ds.images_off + img_bytes) fail("labels_offset 与 N*C*H*W 不自洽");
  if (meta_off != ds.labels_off + ds.n) fail("meta_offset 与 labels_offset+N 不自洽");
  const std::uint64_t expect = meta_off + meta_len;
  std::error_code ec;
  const std::uint64_t actual = std::filesystem::file_size(path, ec);
  if (ec) fail("取文件大小失败");
  if (actual != expect) {
    fail("文件长 " + std::to_string(actual) + " != meta_offset+meta_len " +
         std::to_string(expect));
  }

  ds.labels.resize(static_cast<std::size_t>(ds.n));
  read_exact(ds.file, ds.labels_off, ds.labels.data(), ds.labels.size(), "标签区");
  ds.num_classes = 4;  // tools/DATASET_FORMAT.md §1.3：标签 0..3（action.hpp）
  for (std::size_t i = 0; i < ds.labels.size(); ++i) {
    if (ds.labels[i] >= ds.num_classes) {
      fail("标签越界: label[" + std::to_string(i) + "]=" + std::to_string(ds.labels[i]));
    }
  }

  std::string meta(meta_len, '\0');
  read_exact(ds.file, meta_off, meta.data(), meta.size(), "meta 区");
  if (str_field(meta, "split_policy") != "scene_id") fail("split_policy 必须为 scene_id");
  assign_split(ds, parse_episodes(meta));
  if (opts.drop_collision) {
    drop_collision_samples(ds, meta);
  } else {
    warn_collision_kept(ds, meta);
  }
  return ds;
}

void UavrDataset::load_images(const std::vector<std::uint32_t>& idx, Tensor& out) const {
  const std::int64_t b = static_cast<std::int64_t>(idx.size());
  const std::vector<std::int64_t> want{b, c, h, w};
  if (out.dims() != want) out = Tensor(want);
  const std::size_t plane = static_cast<std::size_t>(c) * h * w;
  std::vector<unsigned char> buf(plane);
  const float inv = 1.0f / 255.0f;
  for (std::int64_t i = 0; i < b; ++i) {
    if (idx[static_cast<std::size_t>(i)] >= static_cast<std::uint32_t>(n)) fail("样本下标越界");
    read_exact(file, images_off + static_cast<std::uint64_t>(idx[i]) * plane, buf.data(), plane,
               "图像区");
    float* dst = out.data() + static_cast<std::size_t>(i) * plane;  // 文件已是 NCHW
    for (std::size_t j = 0; j < plane; ++j) dst[j] = static_cast<float>(buf[j]) * inv;
  }
}

}  // namespace uavrgb
