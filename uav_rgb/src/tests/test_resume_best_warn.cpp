// AUDIT_SRC Nice #9 / #10 回归（小集 1 epoch，几秒跑完）：
//   #9  warmup_steps ≥ 总步数时 lr 全程爬坡到不了设定值：metrics.jsonl 必须有 WARMUP_EXCEEDS_STEPS 告警；
//   #10 epoch 末先写 ckpt_best.bin 再写 ckpt.bin，中间被杀时 best 文件比 ckpt 记的分数新：
//       resume 应接受这个更高的 best（告警）而不是报「不一致」把人卡住；best 文件分数反而更低
//       （拿错目录的 best）仍要拒绝，且报错里给出 --resume <dir>/ckpt_best.bin 的出路。
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "synthetic_dataset.hpp"
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/trainer.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_fail;
  }
}
std::string read_file(const std::string& path) {
  std::ifstream f(path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}
// 把 ckpt_best.bin 记录的 best_macro_f1 改成 v（模拟「best 已写新、ckpt 还是旧」的半截状态）。
void rewrite_best_score(const std::string& path, const CnnModel& like, float lr, float v) {
  CnnModel m = make_cnn(like.in_h, like.in_w, 0, like.fc.out);
  TrainState s = make_train_state(m, lr);
  load_checkpoint(path, m, s);
  s.best_macro_f1 = v;
  save_checkpoint(path, m, s);
}
std::string what_of(const TrainConfig& cfg) {
  try {
    Trainer(cfg).run();
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}
}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_resume_best_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);
  const auto sd = uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", 24, 120, 160, 16, 9);
  TrainConfig base;
  base.data_path = sd.path;
  base.epochs = 1;
  base.batch = 8;  // 16 train → 每 epoch 2 步
  base.lr = 1e-3f;
  base.seed = 3;

  // #9
  {
    TrainConfig c = base;
    c.out_dir = root + "/run_warm";
    c.warmup_steps = 1000;
    check(what_of(c).empty(), "warmup 过长只告警，不能让训练失败");
    const std::string m = read_file(c.out_dir + "/metrics.jsonl");
    check(m.find("WARMUP_EXCEEDS_STEPS") != std::string::npos,
          "warmup_steps ≥ 总步数时 metrics.jsonl 应有 WARMUP_EXCEEDS_STEPS 告警");
    TrainConfig ok = base;
    ok.out_dir = root + "/run_warm_ok";
    ok.warmup_steps = 1;
    check(what_of(ok).empty() &&
              read_file(ok.out_dir + "/metrics.jsonl").find("WARMUP_EXCEEDS_STEPS") == std::string::npos,
          "warmup 短于总步数不该告警");
  }

  // #10
  {
    TrainConfig src = base;
    src.out_dir = root + "/run_src";
    check(what_of(src).empty(), "源 run 应成功");
    CnnModel like = make_cnn(120, 160, 0, 4);
    TrainState st = make_train_state(like, base.lr);
    load_checkpoint(src.out_dir + "/ckpt.bin", like, st);
    const float recorded = st.best_macro_f1;
    check(recorded >= 0.0f && recorded < 0.9f, "源 run 应记录了一个可再抬高的 best 分数");

    // best 文件比 ckpt 新（分数更高）：接受并把更高的分数带进新 run。
    rewrite_best_score(src.out_dir + "/ckpt_best.bin", like, base.lr, recorded + 0.1f);
    TrainConfig newer = base;
    newer.out_dir = root + "/run_newer";
    newer.resume = src.out_dir + "/ckpt.bin";
    const std::string e1 = what_of(newer);
    check(e1.empty(), "ckpt_best 比 ckpt 新时 resume 不应失败: " + e1);
    check(read_file(newer.out_dir + "/metrics.jsonl").find("RESUME_BEST_NEWER") != std::string::npos,
          "接受更新的 ckpt_best 时应留 RESUME_BEST_NEWER 告警");
    if (fs::exists(newer.out_dir + "/ckpt.bin")) {
      TrainState s2 = make_train_state(like, base.lr);
      load_checkpoint(newer.out_dir + "/ckpt.bin", like, s2);
      check(s2.best_macro_f1 >= recorded + 0.1f - 1e-6f,
            "新 run 的 best 阈值应是 ckpt_best 文件里更高的那个分数");
    }

    // best 文件分数更低（不是这次 resume 的 best）：拒绝，并提示直接从 ckpt_best.bin 续。
    rewrite_best_score(src.out_dir + "/ckpt_best.bin", like, base.lr, recorded - 0.2f);
    TrainConfig older = base;
    older.out_dir = root + "/run_older";
    older.resume = src.out_dir + "/ckpt.bin";
    const std::string e2 = what_of(older);
    check(!e2.empty(), "ckpt_best 分数低于 ckpt 记录时必须拒绝");
    check(e2.find("ckpt_best.bin") != std::string::npos && e2.find("--resume") != std::string::npos,
          "拒绝信息应给出 --resume <dir>/ckpt_best.bin 的出路: " + e2);
  }

  fs::remove_all(root);
  if (g_fail) {
    std::cerr << "test_resume_best_warn: FAIL (" << g_fail << ")\n";
    return 1;
  }
  std::cout << "test_resume_best_warn: PASS\n";
  return 0;
}
