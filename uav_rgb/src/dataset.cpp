// UAVR v1 只读实现：48 B 小端头 + N×C×H×W uint8 NCHW + N uint8 标签 + meta JSON。
#include "uavrgb/dataset.hpp"

#include <algorithm>
#include <cctype>
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
  std::string id;  // episode_id 原样（字符串或数字）；旧数据可能缺省
  std::string gid; // geometry_gid；旧数据没有，空串
};

// 取标量原值（字符串去引号，数字原样）；缺字段返回空串。用于 episode_id 绑定，
// 不能要求所有旧数据集都有该字段。
std::string scalar_field(const std::string& obj, const char* key) {
  const std::string pat = std::string("\"") + key + "\":";
  const std::size_t p = obj.find(pat);
  if (p == std::string::npos) return {};
  std::size_t b = p + pat.size();
  while (b < obj.size() && obj[b] == ' ') ++b;
  if (b < obj.size() && obj[b] == '"') {
    const std::size_t e = obj.find('"', b + 1);
    if (e == std::string::npos) fail(std::string("字段未闭合 ") + key);
    return obj.substr(b + 1, e - b - 1);
  }
  std::size_t e = b;
  while (e < obj.size() && obj[e] != ',' && obj[e] != '}' && obj[e] != ' ' &&
         obj[e] != '\n' && obj[e] != '\r') {
    ++e;
  }
  return obj.substr(b, e - b);
}

// 可选标量：缺字段与 JSON null 都返回空串（preprocess 对无 gid 的集写 null）。
std::string opt_field(const std::string& obj, const char* key) {
  std::string v = scalar_field(obj, key);
  return v == "null" ? std::string() : v;
}

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
    s.id = scalar_field(obj, "episode_id");
    s.gid = opt_field(obj, "geometry_gid");
    out.push_back(std::move(s));
    i = e;
  }
  if (out.empty()) fail("meta episodes[] 为空");
  return out;
}

// ---- split 事实 ----
// preprocess 现在落 split_policy/split_scheme/split_salt；旧数据只有 scene_id。
// 诊断划分默认拒绝，旧数据缺新字段时保持旧行为但打 warning。
struct SplitMeta {
  std::string policy;
  std::string scheme;
  std::string salt;
};

bool is_diagnostic_split(const std::string& name) {
  return name.rfind("map-hash:", 0) == 0 || name.rfind("diag", 0) == 0;
}

SplitMeta read_split_meta(const std::string& meta, const LoadOptions& opts) {
  SplitMeta s;
  s.policy = opt_field(meta, "split_policy");
  s.scheme = opt_field(meta, "split_scheme");
  s.salt = opt_field(meta, "split_salt");
  if (s.policy.empty()) fail("meta 缺 split_policy");
  if (is_diagnostic_split(s.policy) || is_diagnostic_split(s.scheme)) {
    if (!opts.allow_diagnostic_split) {
      fail("诊断性 split（" + (s.scheme.empty() ? s.policy : s.scheme) +
           "）默认拒绝；确需读取请设 LoadOptions.allow_diagnostic_split");
    }
    return s;
  }
  if (s.scheme.empty()) {
    // 旧 .uavr：只有 split_policy=scene_id，没有方案/salt/gid。
    if (s.policy != "scene_id") {
      fail("旧 meta 缺 split_scheme 但 split_policy 不是 scene_id: " + s.policy);
    }
    std::fprintf(stderr,
                 "警告: 数据集 meta 缺 split_scheme/split_salt（旧格式），"
                 "按旧行为只校验 scene_id 隔离\n");
    return s;
  }
  if (s.policy != s.scheme) {
    fail("split_policy 与 split_scheme 不一致: " + s.policy + " vs " + s.scheme);
  }
  if (s.scheme != "scene_id" && s.scheme != "geometry_gid" &&
      s.scheme != "campus_map_v1" && s.scheme != "campus_assets_map_v1") {
    fail("未知 split_scheme: " + s.scheme);
  }
  return s;
}

void assign_split(UavrDataset& ds, const std::vector<EpisodeSpan>& eps,
                  const std::string& scheme) {
  std::vector<bool> assigned(static_cast<std::size_t>(ds.n), false);
  std::map<std::int64_t, std::string> scenes;
  std::map<std::string, std::string> gids;
  const bool need_gid = scheme == "geometry_gid";
  for (const EpisodeSpan& e : eps) {
    const auto inserted = scenes.emplace(e.scene, e.split);
    if (!inserted.second && inserted.first->second != e.split) {
      fail("scene_id 跨 split（scene_id=" + std::to_string(e.scene) + "）");
    }
    if (e.gid.empty()) {
      if (need_gid) fail("split_scheme=geometry_gid 但 episode 缺 geometry_gid");
    } else {
      const auto gi = gids.emplace(e.gid, e.split);
      if (!gi.second && gi.first->second != e.split) {
        fail("geometry_gid 跨 split（gid=" + e.gid + "）");
      }
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

// ---- 语言侧车 ----
// instructions.jsonl 每行：{"episode_id":..,"instruction":"..","sample_start":..,
//   "sample_count":..,"text_ids":[..]}；只取区间与 token id，按样本展开成 [n,kTextLen]。
// 没有侧车就不加载——无文本模型照常训练。
std::uint32_t side_num(const std::string& line, const char* key) {
  const std::string pat = std::string("\"") + key + "\":";
  const std::size_t p = line.find(pat);
  if (p == std::string::npos) fail(std::string("instructions.jsonl 缺字段 ") + key);
  std::size_t b = p + pat.size();
  while (b < line.size() && line[b] == ' ') ++b;  // json.dumps 默认写成 "key": 0
  std::size_t e = b;
  while (e < line.size() && std::isdigit(static_cast<unsigned char>(line[e])) != 0) ++e;
  if (e == b) fail(std::string("instructions.jsonl 字段非整数 ") + key);
  // 自己累加并检查上界：stoul 结果截断成 uint32 会让 2^32+k 回绕成 k，
  // 超大数还会抛 out_of_range 而不是本文件的 runtime_error。
  std::uint64_t v = 0;
  for (std::size_t i = b; i < e; ++i) {
    const std::uint64_t d = static_cast<std::uint64_t>(line[i] - '0');
    // 先检查再乘：20 位以上的数字会让 uint64 先回绕，绕过下面的上界比较。
    if (v > (0xFFFFFFFFull - d) / 10) {
      fail(std::string("instructions.jsonl 字段超 uint32: ") + key);
    }
    v = v * 10 + d;
  }
  return static_cast<std::uint32_t>(v);
}

void load_text_sidecar(UavrDataset& ds, const std::string& uavr_path,
                       const std::vector<EpisodeSpan>& eps) {
  const auto side = std::filesystem::path(uavr_path).parent_path() / "instructions.jsonl";
  std::error_code ec;
  if (!std::filesystem::exists(side, ec)) return;
  std::ifstream f(side);
  if (!f) fail("无法打开 " + side.string());
  std::vector<std::uint8_t> ids(static_cast<std::size_t>(ds.n) * kTextLen, 0);
  std::vector<char> seen(eps.size(), 0);
  std::string line;
  std::size_t lineno = 0;
  std::size_t used = 0;
  std::size_t with_ids = 0;      // 带 text_ids 的行数
  std::size_t missing_ids = 0;   // 缺 text_ids 的行数
  std::size_t first_missing = 0; // 第一处缺 text_ids 的行号（混合时报错用）
  while (std::getline(f, line)) {
    ++lineno;
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
    ++used;
    const std::uint32_t start = side_num(line, "sample_start");
    const std::uint32_t count = side_num(line, "sample_count");
    if (static_cast<std::uint64_t>(start) + count > static_cast<std::uint64_t>(ds.n)) {
      fail("instructions.jsonl 第 " + std::to_string(lineno) + " 行区间越界");
    }
    // 侧车必须按 episode 身份绑定：区间要唯一匹配 meta.episodes[]，同一集不能写两次。
    // 只查「样本被覆盖过」挡不住旧侧车/异源侧车配到新数据集上。
    std::size_t match = eps.size();
    for (std::size_t i = 0; i < eps.size(); ++i) {
      if (eps[i].start == start && eps[i].count == count) {
        if (match != eps.size()) {
          fail("instructions.jsonl 第 " + std::to_string(lineno) + " 行区间匹配多个 episode");
        }
        match = i;
      }
    }
    if (match == eps.size()) {
      fail("instructions.jsonl 第 " + std::to_string(lineno) +
           " 行区间与 meta episodes[] 不匹配（侧车不属于该数据集？）");
    }
    if (seen[match] != 0) {
      fail("instructions.jsonl 第 " + std::to_string(lineno) + " 行重复覆盖 episode");
    }
    seen[match] = 1;
    const std::string side_id = scalar_field(line, "episode_id");
    if (!side_id.empty() && !eps[match].id.empty() && side_id != eps[match].id) {
      fail("instructions.jsonl 第 " + std::to_string(lineno) + " 行 episode_id 不匹配（侧车 " +
           side_id + " vs meta " + eps[match].id + "）");
    }
    // 旧侧车（L2 之前）整份都没有 text_ids：当作没有侧车。image-only 照常训练，
    // 真需要文本的模型由 trainer 报「没有侧车」，不在读取层拦整个数据集。
    // 但只有部分行缺 text_ids 说明侧车半迁移/混源：静默丢弃会让文本模型误以为
    // 整份侧车不存在，必须报错。这里只记行号并继续，末尾校验对两种情况都要跑。
    const std::size_t kp = line.find("\"text_ids\":");
    if (kp == std::string::npos) {
      if (missing_ids == 0) first_missing = lineno;
      ++missing_ids;
      continue;
    }
    ++with_ids;
    const std::size_t lb = line.find('[', kp);
    const std::size_t rb = line.find(']', kp);
    if (lb == std::string::npos || rb == std::string::npos || rb < lb) {
      fail("instructions.jsonl text_ids 数组未闭合");
    }
    std::vector<std::uint8_t> tok;
    std::size_t i = lb + 1;
    while (i < rb) {
      const unsigned char c = static_cast<unsigned char>(line[i]);
      if (c == ' ' || c == ',' || c == '\t') {
        ++i;
        continue;
      }
      // 只接受十进制数字：旧实现跳过任意非数字字符，会把 [-1] 读成 1。
      if (std::isdigit(c) == 0) {
        fail("instructions.jsonl 第 " + std::to_string(lineno) + " 行 text_ids 含非数字");
      }
      std::uint32_t v = 0;
      while (i < rb && std::isdigit(static_cast<unsigned char>(line[i])) != 0) {
        v = v * 10 + static_cast<std::uint32_t>(line[i] - '0');
        if (v >= static_cast<std::uint32_t>(kTextVocab)) {
          fail("instructions.jsonl token id 越界: " + std::to_string(v));
        }
        ++i;
      }
      tok.push_back(static_cast<std::uint8_t>(v));
    }
    // 超长必须报错：静默截断会悄悄改掉指令的意思（preprocess.py 的 TEXT_LEN 与
    // text.hpp 的 kTextLen 是两份定义，漂了就靠这里兜）。
    if (tok.size() > static_cast<std::size_t>(kTextLen)) {
      fail("instructions.jsonl 第 " + std::to_string(lineno) + " 行 text_ids 有 " +
           std::to_string(tok.size()) + " 个 token，超过 kTextLen=" +
           std::to_string(kTextLen));
    }
    for (std::uint32_t k = 0; k < count; ++k) {
      const std::size_t s = static_cast<std::size_t>(start) + k;
      for (std::size_t t = 0; t < tok.size() && t < static_cast<std::size_t>(kTextLen); ++t) {
        ids[s * kTextLen + t] = tok[t];
      }
    }
  }
  if (used != eps.size()) {
    fail("instructions.jsonl 行数 " + std::to_string(used) + " != 数据集 episode 数 " +
         std::to_string(eps.size()) + "（侧车缺集或多集）");
  }
  for (std::size_t i = 0; i < seen.size(); ++i) {
    if (seen[i] == 0) fail("instructions.jsonl 未覆盖 episode " + std::to_string(i));
  }
  if (missing_ids != 0) {
    if (with_ids == 0) return;  // 整份旧侧车：视为无侧车（保持旧行为）
    fail("instructions.jsonl 第 " + std::to_string(first_missing) +
         " 行缺 text_ids（部分行有、部分行没有）");
  }
  ds.text_ids = std::move(ids);
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
  const SplitMeta split_meta = read_split_meta(meta, opts);
  ds.split_policy = split_meta.policy;
  ds.split_scheme = split_meta.scheme;
  ds.split_salt = split_meta.salt;
  const std::vector<EpisodeSpan> eps = parse_episodes(meta);
  assign_split(ds, eps, split_meta.scheme);
  if (opts.drop_collision) {
    drop_collision_samples(ds, meta);
  } else {
    warn_collision_kept(ds, meta);
  }
  load_text_sidecar(ds, path, eps);  // 无侧车时 text_ids 保持空
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

void UavrDataset::load_text(const std::vector<std::uint32_t>& idx,
                            std::vector<std::uint8_t>& out) const {
  if (text_ids.empty()) fail("数据集没有 instructions.jsonl 侧车，无法训练文本模型");
  out.resize(idx.size() * static_cast<std::size_t>(kTextLen));
  for (std::size_t i = 0; i < idx.size(); ++i) {
    if (idx[i] >= static_cast<std::uint32_t>(n)) fail("样本下标越界");
    std::copy_n(text_ids.begin() + static_cast<std::size_t>(idx[i]) * kTextLen, kTextLen,
                out.begin() + static_cast<std::ptrdiff_t>(i) * kTextLen);
  }
}

}  // namespace uavrgb
