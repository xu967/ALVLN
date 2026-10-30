// 常驻推理进程入口。协议见 include/uavrgb/infer.hpp 与 docs/TASKS.md E6-1。
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "uavrgb/checkpoint.hpp"
#include "uavrgb/infer.hpp"
#include "uavrgb/model.hpp"

namespace {

int usage() {
  std::fprintf(stderr,
               "用法: uavrgb_infer --ckpt <ckpt.bin> [--text <id,id,...>]\n");
  return 2;
}

// --text 是当前 episode 的指令 token（逗号分隔 0..255）；空串=无指令。
bool parse_text(const std::string& s, std::vector<std::uint8_t>& out) {
  out.clear();
  if (s.empty()) return true;
  std::size_t start = 0;
  while (start <= s.size()) {
    const std::size_t comma = s.find(',', start);
    const std::string tok = s.substr(start, comma == std::string::npos
                                                ? std::string::npos : comma - start);
    if (tok.empty()) return false;
    char* end = nullptr;
    const long v = std::strtol(tok.c_str(), &end, 10);
    if (end == nullptr || *end != '\0' || v < 0 || v > 255) return false;
    out.push_back(static_cast<std::uint8_t>(v));
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt;
  std::vector<std::uint8_t> text_ids;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
      ckpt = argv[++i];
    } else if (std::strcmp(argv[i], "--text") == 0 && i + 1 < argc) {
      if (!parse_text(argv[++i], text_ids)) return usage();
    } else {
      return usage();
    }
  }
  if (ckpt.empty()) return usage();

  try {
    const uavrgb::CheckpointMeta meta = uavrgb::read_checkpoint_meta(ckpt);
    uavrgb::CnnModel model = uavrgb::make_cnn(meta.in_h, meta.in_w, 0, meta.num_classes);
    uavrgb::TrainState st = uavrgb::make_train_state(model, 0.0f);
    uavrgb::load_checkpoint(ckpt, model, st);
    std::printf("{\"in_h\":%d,\"in_w\":%d,\"in_c\":%d,\"num_classes\":%d}\n",
                meta.in_h, meta.in_w, meta.in_c, meta.num_classes);
    std::fflush(stdout);
    const std::size_t frames =
        uavrgb::run_infer_loop(model, std::cin, std::cout, text_ids);
    std::fprintf(stderr, "uavrgb_infer: %zu 帧\n", frames);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "uavrgb_infer: %s\n", e.what());
    return 1;
  }
  return 0;
}
