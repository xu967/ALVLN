// UAVRAUX1 侧车读写：夹具写出 → load_uavr 读回（prev 链 / prev-action / 深度目标 / 首帧标记），
// 无侧车时 present=false 且旧数据集照常加载；坏侧车（SHA 不匹配、截断、N 不符、prev_action
// 越界、flags 未知）一律 runtime_error，不能半读。
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/aux_sidecar.hpp"
#include "uavrgb/dataset.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;

namespace {

int g_fail = 0;
void check(bool ok, const std::string& msg) {
  if (!ok) {
    ++g_fail;
    std::cerr << "FAIL: " << msg << "\n";
  }
}

std::vector<char> slurp(const std::string& p) {
  std::ifstream f(p, std::ios::binary);
  return std::vector<char>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
void spit(const std::string& p, const std::vector<char>& b) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(b.data(), static_cast<std::streamsize>(b.size()));
}

bool load_fails(const std::string& uavr, const std::string& why) {
  try {
    (void)load_uavr(uavr);
  } catch (const std::runtime_error& e) {
    std::cout << "  [" << why << "] 拒绝: " << e.what() << "\n";
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_aux_sidecar_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);
  // 12 样本：train 8（集 1）、val 4（集 2）；h=16,w=24 → 深度网格 2x3。
  const int n = 12, h = 16, w = 24;
  const uavrgb_test::SynthDataset sd =
      uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", n, h, w, 8, 21);

  {
    const UavrDataset ds = load_uavr(sd.path);
    check(!ds.aux.present && !ds.aux.has_depth && ds.aux.prev_index.empty(), "无侧车：present=false");
  }
  const std::string aux_path =
      uavrgb_test::write_synthetic_aux(sd.path, n, h, w, {{0, 8}, {8, 4}}, sd.labels, /*with_depth=*/true);
  check(aux_path == aux_sidecar_path(sd.path), "侧车路径 = uavr 换后缀 .aux");
  const std::vector<char> good = slurp(aux_path);
  check(good.size() == 128 + 5 * n + static_cast<std::size_t>(n) * 2 * 3, "侧车长度 = 128 + 5N + N·dh·dw");
  {
    const UavrDataset ds = load_uavr(sd.path);
    check(ds.aux.present && ds.aux.has_depth, "有侧车：present/has_depth");
    check(ds.aux.depth_h == 2 && ds.aux.depth_w == 3 && ds.aux.depth_classes == 8, "深度网格 2x3、8 带");
    check(ds.aux.band_edges.size() == 7 && ds.aux.band_edges[0] == 0.3f && ds.aux.band_edges[6] == 4.0f,
          "分带上界读回");
    check(ds.aux.envelope_radius == 0.102f, "包络半径读回");
    bool chain_ok = ds.aux.prev_index.size() == static_cast<std::size_t>(n);
    for (int i = 0; chain_ok && i < n; ++i) {
      const bool first = i == 0 || i == 8;
      chain_ok = ds.aux.prev_index[i] == static_cast<std::uint32_t>(first ? i : i - 1) &&
                 ds.aux.first_frame[i] == (first ? 1 : 0) &&
                 ds.aux.prev_action[i] == (first ? 0 : sd.labels[static_cast<std::size_t>(i) - 1]);
    }
    check(chain_ok, "prev 链：首帧指向自身/prev=SAFE，其余指向 i-1、prev=上一步标签");
    check(ds.aux.depth.size() == static_cast<std::size_t>(n) * 6, "深度目标 N·dh·dw 字节");
    // 夹具规则：格值 = 样本标签，(i + r*3 + c) % 7 == 0 → 255：样本 0 的格 0、样本 2 的格 5、
    // 样本 7 的格 0 未知；样本 1 的 6 格全有效。
    check(ds.aux.depth[0] == kAuxUnknown && ds.aux.depth[1] == sd.labels[0] &&
              ds.aux.depth[6 * 2 + 5] == kAuxUnknown && ds.aux.depth[6 * 7] == kAuxUnknown &&
              ds.aux.depth[6 * 1 + 5] == sd.labels[1] && ds.aux.depth[6 * 1 + 0] == sd.labels[1],
          "深度目标取值与夹具规则一致（含 255 未知格）");
  }
  {  // 无深度段的侧车
    uavrgb_test::write_synthetic_aux(sd.path, n, h, w, {{0, 8}, {8, 4}}, sd.labels, /*with_depth=*/false);
    const UavrDataset ds = load_uavr(sd.path);
    check(ds.aux.present && !ds.aux.has_depth && ds.aux.depth.empty() && ds.aux.depth_h == 0,
          "无深度侧车：present、has_depth=false、depth 空");
  }
  // ---- 负例 ----
  {  // 头里的 UAVR SHA 改一字节
    std::vector<char> b = good;
    b[28] = static_cast<char>(b[28] ^ 0x5a);
    spit(aux_path, b);
    check(load_fails(sd.path, "uavr sha"), "UAVR SHA 不匹配被拒");
  }
  {  // payload 改一字节（prev_action 字段但仍在 0..3）→ payload SHA 不匹配
    std::vector<char> b = good;
    b[128 + 4 * n + 2] = static_cast<char>((b[128 + 4 * n + 2] + 1) % 4);
    spit(aux_path, b);
    check(load_fails(sd.path, "payload sha"), "payload SHA 不匹配被拒");
  }
  {  // 截断
    std::vector<char> b(good.begin(), good.end() - 5);
    spit(aux_path, b);
    check(load_fails(sd.path, "truncated"), "截断被拒");
  }
  {  // N 不符
    std::vector<char> b = good;
    b[8] = static_cast<char>(n + 1);
    spit(aux_path, b);
    check(load_fails(sd.path, "N"), "N 与 uavr 不符被拒");
  }
  {  // 未知 flags 位
    std::vector<char> b = good;
    b[16] = static_cast<char>(b[16] | 0x02);
    spit(aux_path, b);
    check(load_fails(sd.path, "flags"), "未知 flags 被拒");
  }
  {  // 重新写好的侧车但 dataset.uavr 换了内容（模拟「新数据集 + 旧侧车」）
    spit(aux_path, good);
    uavrgb_test::write_synthetic_uavr(sd.path, n, h, w, 8, /*seed=*/22);
    check(load_fails(sd.path, "stale"), "数据集重打包后旧侧车被拒");
  }

  fs::remove_all(root);
  if (g_fail) {
    std::cerr << "test_aux_sidecar: FAIL (" << g_fail << ")\n";
    return 1;
  }
  std::cout << "test_aux_sidecar: PASS\n";
  return 0;
}
