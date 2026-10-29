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
  auto run = [&](const char* name, const std::string& from, const std::string& to,
                 bool should_pass) {
    auto meta = original;
    const auto pos = meta.find(from);
    if (pos == std::string::npos) throw std::runtime_error("测试替换目标不存在");
    meta.replace(pos, from.size(), to);
    const auto path = root / (std::string(name) + ".uavr");
    std::ofstream out(path, std::ios::binary);
    out.write(bytes.data(), 44);
    uavrgb_test::put_u32(out, static_cast<std::uint32_t>(meta.size()));
    out.write(bytes.data() + 48, meta_offset - 48);
    out << meta;
    out.close();
    bool accepted = true;
    try { (void)uavrgb::load_uavr(path.string()); }
    catch (const std::exception& e) { accepted = false; std::cout << name << ": " << e.what() << '\n'; }
    if (accepted != should_pass) { ++failures; std::cerr << "FAIL: " << name << '\n'; }
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
  fs::remove_all(root);
  std::cout << "test_dataset_split: " << (failures ? "FAIL" : "PASS") << '\n';
  return failures ? 1 : 0;
}
