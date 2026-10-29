// 常驻推理进程入口。协议见 include/uavrgb/infer.hpp 与 docs/TASKS.md E6-1。
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>

#include "uavrgb/checkpoint.hpp"
#include "uavrgb/infer.hpp"
#include "uavrgb/model.hpp"

namespace {

int usage() {
  std::fprintf(stderr, "用法: uavrgb_infer --ckpt <ckpt.bin>\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string ckpt;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--ckpt") == 0 && i + 1 < argc) {
      ckpt = argv[++i];
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
    const std::size_t frames = uavrgb::run_infer_loop(model, std::cin, std::cout);
    std::fprintf(stderr, "uavrgb_infer: %zu 帧\n", frames);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "uavrgb_infer: %s\n", e.what());
    return 1;
  }
  return 0;
}
