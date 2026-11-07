// checkpoint v4/v5/v6 交叉读：v6 写出的文件按格式说明剪成 v5/v4 字节流后仍能读进纯视觉/文本
// 模型（旧 run 的权重不作废）；v6 的 aux/prev 字段写回一致；结构不符（单帧 ckpt 读进双帧模型、
// 辅助头有无不同）必须被拒绝而不是错位读参数。字节布局见 checkpoint.hpp 文件头注释。
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "uavrgb/checkpoint.hpp"
#include "uavrgb/model.hpp"

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

// 固定头 8+4+4+4*4 = 32 B；之后 adam_t(4) epoch(4) nw(4) [nw 个 float] best(4) text_dim(4)
// 契约(113) aux_classes(4) prev_dim(4)。class_weights 为空时 text_dim 在 48，契约在 52，v6 字段在 165。
constexpr std::size_t kContractOff = 52;
constexpr std::size_t kV6Off = kContractOff + 113;

void set_version(std::vector<char>& b, std::uint32_t v) { std::memcpy(b.data() + 8, &v, 4); }

bool loads(const std::string& path, CnnModel& m) {
  TrainState st = make_train_state(m, 1e-3f);
  try {
    load_checkpoint(path, m, st);
    return true;
  } catch (const std::invalid_argument& e) {
    std::cout << "  拒绝: " << e.what() << "\n";
    return false;
  }
}

bool same_params(const CnnModel& a, const CnnModel& b) {
  const auto pa = cnn_params(a), pb = cnn_params(b);
  if (pa.size() != pb.size()) return false;
  for (std::size_t k = 0; k < pa.size(); ++k) {
    if (pa[k]->dims() != pb[k]->dims()) return false;
    for (std::size_t i = 0; i < pa[k]->numel(); ++i) {
      if ((*pa[k])[i] != (*pb[k])[i]) return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_ckpt_versions_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  // 1) v6 双帧 + prev + aux 模型 roundtrip，meta 读回结构。
  {
    CnnModel m = make_cnn(16, 24, 3, 4, 0, 6, 8, 4);
    TrainState st = make_train_state(m, 1e-3f);
    st.epoch = 2;
    save_checkpoint(root + "/p2.bin", m, st);
    const CheckpointMeta meta = read_checkpoint_meta(root + "/p2.bin");
    check(meta.in_c == 6 && meta.aux_classes == 8 && meta.prev_dim == 4 && meta.text_dim == 0 &&
              meta.in_h == 16 && meta.in_w == 24 && meta.num_classes == 4,
          "v6 meta 读回 in_c/aux_classes/prev_dim");
    CnnModel back = make_cnn(16, 24, 99, 4, 0, 6, 8, 4);
    check(loads(root + "/p2.bin", back) && same_params(m, back), "v6 roundtrip 参数逐元素一致");
    // 结构不符：单帧模型 / 没有辅助头的双帧模型 / prev_dim 不同，都要拒绝。
    CnnModel single = make_cnn(16, 24, 1, 4, 0, 3, 0, 0);
    check(!loads(root + "/p2.bin", single), "双帧 ckpt 读进单帧模型被拒");
    CnnModel no_aux = make_cnn(16, 24, 1, 4, 0, 6, 0, 4);
    check(!loads(root + "/p2.bin", no_aux), "有辅助头 ckpt 读进无辅助头模型被拒");
    CnnModel no_prev = make_cnn(16, 24, 1, 4, 0, 6, 8, 0);
    check(!loads(root + "/p2.bin", no_prev), "prev_dim 不同被拒");
  }

  // 2) 纯视觉与文本 FiLM 模型：v6 文件剪成 v5 / v4 后仍可读（模拟旧 run 留下的 checkpoint）。
  for (int text_dim : {0, 16}) {
    CnnModel m = make_cnn(16, 24, 5, 4, text_dim);
    TrainState st = make_train_state(m, 1e-3f);
    const std::string tag = text_dim ? "film" : "vision";
    save_checkpoint(root + "/" + tag + "_v6.bin", m, st);
    std::vector<char> b = slurp(root + "/" + tag + "_v6.bin");
    // v5：去掉契约之后的 8 B（aux_classes/prev_dim）。
    std::vector<char> v5 = b;
    v5.erase(v5.begin() + kV6Off, v5.begin() + kV6Off + 8);
    set_version(v5, 5);
    spit(root + "/" + tag + "_v5.bin", v5);
    // v4：再去掉 113 B 契约。
    std::vector<char> v4 = v5;
    v4.erase(v4.begin() + kContractOff, v4.begin() + kContractOff + 113);
    set_version(v4, 4);
    spit(root + "/" + tag + "_v4.bin", v4);
    for (const char* ver : {"v6", "v5", "v4"}) {
      const std::string path = root + "/" + tag + "_" + ver + ".bin";
      CnnModel back = make_cnn(16, 24, 77, 4, text_dim);
      const bool ok = loads(path, back) && same_params(m, back);
      check(ok, tag + " " + ver + " 可读且参数一致");
      const CheckpointMeta meta = read_checkpoint_meta(path);
      check(meta.in_c == 3 && meta.aux_classes == 0 && meta.prev_dim == 0 && meta.text_dim == text_dim,
            tag + " " + ver + " meta: in_c=3, 无 aux/prev");
      // 旧 ckpt 读进双帧模型：header in_c 不符，必须拒绝。
      CnnModel two = make_cnn(16, 24, 1, 4, text_dim, 6);
      check(!loads(path, two), tag + " " + ver + " 读进双帧模型被拒");
    }
  }

  // 3) v6 字段非法值被拒（aux_classes 超上限）。
  {
    std::vector<char> b = slurp(root + "/vision_v6.bin");
    const std::int32_t bad = kMaxAuxClasses + 1;
    std::memcpy(b.data() + kV6Off, &bad, 4);
    spit(root + "/bad_aux.bin", b);
    CnnModel m = make_cnn(16, 24, 1, 4, 0);
    check(!loads(root + "/bad_aux.bin", m), "aux_classes 超上限被拒");
    bool threw = false;
    try {
      (void)read_checkpoint_meta(root + "/bad_aux.bin");
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    check(threw, "read_checkpoint_meta 也拒绝非法 aux_classes");
  }

  fs::remove_all(root);
  if (g_fail) {
    std::cerr << "test_checkpoint_versions: FAIL (" << g_fail << ")\n";
    return 1;
  }
  std::cout << "test_checkpoint_versions: PASS\n";
  return 0;
}
