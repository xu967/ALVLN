// 回归：读端必须独立拒绝场景泄漏、区间重叠/遗漏和整数溢出。
#include <filesystem>
#include <iostream>
#include <iterator>
#include <string>

#include "synthetic_dataset.hpp"
#include "uavrgb/dataset.hpp"

namespace fs = std::filesystem;

int main() {
  const auto root = fs::temp_directory_path() /
                    ("uavrgb_split_" + std::to_string(std::random_device{}()));
  fs::create_directories(root);
  const auto source = uavrgb_test::write_synthetic_uavr(
      (root / "valid.uavr").string(), 4, 8, 8, 2, 1);
  std::ifstream input(source.path, std::ios::binary);
  const std::string bytes((std::istreambuf_iterator<char>(input)), {});
  constexpr std::size_t meta_offset = 48 + 4 * 3 * 8 * 8 + 4;
  const auto original = bytes.substr(meta_offset);
  int failures = 0;
  auto sub = [](std::string s, const std::string& from, const std::string& to) {
    const auto pos = s.find(from);
    if (pos == std::string::npos) throw std::runtime_error("测试替换目标不存在");
    s.replace(pos, from.size(), to);
    return s;
  };
  auto write_and_load = [&](const char* name, const std::string& meta, bool should_pass,
                            const uavrgb::LoadOptions& opts) {
    const auto path = root / (std::string(name) + ".uavr");
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), 44);
    uavrgb_test::put_u32(out, static_cast<std::uint32_t>(meta.size()));
    out.write(bytes.data() + 48, meta_offset - 48);
    out << meta;
    out.close();
    bool accepted = true;
    try {
      (void)uavrgb::load_uavr(path.string(), opts);
    } catch (const std::exception& e) {
      accepted = false;
      std::cout << name << ": " << e.what() << '\n';
    }
    if (accepted != should_pass) {
      ++failures;
      std::cerr << "FAIL: " << name << '\n';
    }
  };
  auto run = [&](const char* name, const std::string& from, const std::string& to,
                 bool should_pass, const uavrgb::LoadOptions& opts = {}) {
    write_and_load(name, sub(original, from, to), should_pass, opts);
  };
  run("valid", "scene_id\":2", "scene_id\":2", true);
  run("same_scene", "scene_id\":2", "scene_id\":1", false);
  run("overlap", "sample_start\":2", "sample_start\":1", false);
  run("gap", "sample_count\":2", "sample_count\":1", false);
  run("overflow", "sample_start\":0", "sample_start\":4294967296", false);
  run("fraction", "sample_start\":0", "sample_start\":0.5", false);
  run("negative", "sample_start\":0", "sample_start\":-1", false);
  run("wrong_policy", "split_policy\":\"scene_id", "split_policy\":\"frame", false);
  run("large_scene", "scene_id\":2", "scene_id\":9223372036854775807", true);
  run("signed_scene", "scene_id\":2", "scene_id\":-42", true);
  // json.dumps 默认分隔符（": " / ", "）写出的 meta 是合法 JSON，扫描器必须接受（AUDIT_SRC #4）；
  // preprocess 用紧凑分隔符所以自家数据不触发，手改/indent 重写一次就撞上。
  run("spaced_int", "sample_start\":0", "sample_start\": 0", true);
  run("spaced_str", "split\":\"train", "split\": \"train", true);
  run("spaced_scene", "scene_id\":2", "scene_id\": 2", true);
  run("spaced_collision_flags", "split_policy\":\"scene_id",
      "collision_flags\": \"0000\",\"split_policy\":\"scene_id", true, uavrgb::LoadOptions{true});
  // 字符串值里的 '}' 不是 episode 项的结尾（REVIEW2_SRC #7）：version 排在 split/sample_* 之前，
  // 按首个 '}' 截断会让后面的必填字段「缺失」。
  run("brace_in_string", "version\":\"synthetic", "version\":\"syn}thetic", true);

  // 诊断性划分默认拒绝；显式 opt-in 才允许读取。
  run("diag_scheme", "split_policy\":\"scene_id",
      "split_policy\":\"scene_id\",\"split_scheme\":\"map-hash:diag1", false);
  run("diag_scheme_allowed", "split_policy\":\"scene_id",
      "split_policy\":\"scene_id\",\"split_scheme\":\"map-hash:diag1", true,
      uavrgb::LoadOptions{false, true});
  run("diag_policy", "split_policy\":\"scene_id", "split_policy\":\"diag1", false);
  // 新字段自洽：policy 与 scheme 不一致、声称按 gid 却没有 gid 都要拒绝。
  run("scheme_mismatch", "split_policy\":\"scene_id",
      "split_policy\":\"scene_id\",\"split_scheme\":\"campus_map_v1", false);
  run("campus_policy", "split_policy\":\"scene_id",
      "split_policy\":\"campus_map_v1\",\"split_scheme\":\"campus_map_v1", true);
  std::string gid_meta = sub(original, "split_policy\":\"scene_id",
                             "split_policy\":\"geometry_gid\",\"split_scheme\":\"geometry_gid");
  gid_meta = sub(gid_meta, "\"scene_id\":1", "\"geometry_gid\":\"g1\",\"scene_id\":1");
  gid_meta = sub(gid_meta, "\"scene_id\":2", "\"geometry_gid\":\"g2\",\"scene_id\":2");
  write_and_load("gid_ok", gid_meta, true, {});
  write_and_load("gid_cross_split",
                 sub(gid_meta, "\"geometry_gid\":\"g2\"", "\"geometry_gid\":\"g1\""),
                 false, {});
  write_and_load("gid_missing",
                 sub(original, "split_policy\":\"scene_id",
                     "split_policy\":\"geometry_gid\",\"split_scheme\":\"geometry_gid"),
                 false, {});

  // 旧 .uavr 缺 split_scheme：保持可读，并在 dataset 上暴露空方案。
  const auto legacy = uavrgb::load_uavr(source.path);
  if (legacy.split_scheme != "" || legacy.split_policy != "scene_id") {
    ++failures;
    std::cerr << "FAIL: 旧数据 split 字段被改写\n";
  }
  fs::remove_all(root);
  std::cout << "test_dataset_split: " << (failures ? "FAIL" : "PASS") << '\n';
  return failures ? 1 : 0;
}
