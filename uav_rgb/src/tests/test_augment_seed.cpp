// H8：数据增强随机数发生器种子随 --seed 变化（卡 C1a 第 2 条）。
//
// 验证设计：
//   用 16 个逐字节完全相同的非对称图案样本组成 train split，batch=16 一次取完，并从同一份
//   init.bin 载入相同初始权重（--allow-resume-mismatch 放行不同 seed）。
//   这样无论样本顺序如何打乱、初始化如何设，进入 forward_batch 前的输入批次与初始参数都逐位相同：
//   1) 不开增强时，seed=0/1/2 的第 1 步 loss 与更新后参数逐位相同（自证夹具剥离了其他随机源）；
//   2) 开 flip / photo / shift 时，seed=0/1/2 的第 1 步 loss 与更新后参数两两不同（旧实现写死
//      0x9e3779b9u，三个种子完全相同，在此变红）；
//   3) 同 seed 开增强跑两次，结果逐位可复现；
//   4) resume 未显式给 --seed 时，aug_rng_ 必须按从源 config.json 继承的 seed 播种，而非默认 1234。
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "pattern_dataset.hpp"
#include "uavrgb/action.hpp"
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/trainer.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;
using namespace uavrgb_test;

namespace {

int g_fail = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_fail;
  }
}

bool same_model_bits(const CnnModel& a, const CnnModel& b) {
  const auto pa = cnn_params(a);
  const auto pb = cnn_params(b);
  if (pa.size() != pb.size()) return false;
  for (std::size_t i = 0; i < pa.size(); ++i) {
    if (pa[i]->numel() != pb[i]->numel()) return false;
    if (std::memcmp(pa[i]->data(), pb[i]->data(), pa[i]->numel() * sizeof(float)) != 0) {
      return false;
    }
  }
  return true;
}

double first_step_loss(const std::string& metrics_path) {
  std::ifstream f(metrics_path);
  const std::string key = "\"loss\":";
  for (std::string line; std::getline(f, line);) {
    if (line.find("\"event\":\"train_step\"") == std::string::npos) continue;
    const std::size_t p = line.find(key);
    if (p != std::string::npos) return std::stod(line.substr(p + key.size()));
  }
  return -1.0;
}

struct StepOut {
  double loss = -1.0;
  CnnModel model;
};

StepOut run_from_fixed_init(const std::string& ds_path, const std::string& init_ckpt,
                            const std::string& out_dir, unsigned seed, const std::string& aug) {
  TrainConfig cfg;
  cfg.data_path = ds_path;
  cfg.out_dir = out_dir;
  cfg.resume = init_ckpt;
  cfg.allow_resume_mismatch = true;
  cfg.epochs = 1;
  cfg.batch = 16;
  cfg.lr = 1e-3f;
  cfg.lr_set = true;
  cfg.batch_set = true;
  cfg.seed = seed;
  cfg.seed_set = true;
  cfg.augment = aug;
  cfg.augment_set = true;
  Trainer tr(cfg);
  tr.run();
  return {first_step_loss(out_dir + "/metrics.jsonl"), tr.model()};
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_aug_seed_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  constexpr int kH = 16, kW = 32, kTrain = 16, kTotal = 20;
  const std::string ds_path = root + "/same.uavr";
  write_synthetic_uavr(ds_path, kTotal, kH, kW, kTrain, /*seed=*/11);
  // 全部样本用同一张左半亮图案与同一标签 YAW_LEFT，使任意打乱后的批次逐字节恒定。
  const Plane pat = region_plane(kH, kW, half_region(kH, kW, /*left=*/true));
  overwrite_samples(ds_path, kH, kW, std::vector<Plane>(kTotal, pat),
                    std::vector<std::uint8_t>(kTotal, static_cast<std::uint8_t>(Action::YAW_LEFT)));

  const std::string init_dir = root + "/init";
  fs::create_directories(init_dir);
  const std::string init_ckpt = init_dir + "/ckpt.bin";
  {
    CnnModel m = make_cnn(kH, kW, /*seed=*/99, /*num_classes=*/4);
    TrainState st = make_train_state(m, 1e-3f);
    save_checkpoint(init_ckpt, m, st);
  }

  // 1) 夹具隔离自检：不开增强时，seed=0/1/2 的单步更新逐位一致。
  {
    const StepOut s0 = run_from_fixed_init(ds_path, init_ckpt, root + "/plain_s0", 0, "");
    const StepOut s1 = run_from_fixed_init(ds_path, init_ckpt, root + "/plain_s1", 1, "");
    const StepOut s2 = run_from_fixed_init(ds_path, init_ckpt, root + "/plain_s2", 2, "");
    check(s0.loss == s1.loss && s1.loss == s2.loss && same_model_bits(s0.model, s1.model) &&
              same_model_bits(s1.model, s2.model),
          "不开增强时同初始权重 + 同构样本下 seed 0/1/2 应逐位一致");
  }

  // 2) 开 flip / photo / shift 时，seed=0/1/2 产生的增强参数序列必须互异。
  for (const char* m : {"flip", "photo", "shift"}) {
    const std::string mode(m);
    const StepOut s0 = run_from_fixed_init(ds_path, init_ckpt, root + "/" + mode + "_s0", 0, mode);
    const StepOut s1 = run_from_fixed_init(ds_path, init_ckpt, root + "/" + mode + "_s1", 1, mode);
    const StepOut s2 = run_from_fixed_init(ds_path, init_ckpt, root + "/" + mode + "_s2", 2, mode);
    std::cout << "augment=" << mode << " step1 loss: s0=" << s0.loss << " s1=" << s1.loss
              << " s2=" << s2.loss << "\n";
    check(s0.loss != s1.loss && s1.loss != s2.loss && s0.loss != s2.loss,
          "augment=" + mode + " 下 seed 0/1/2 的第 1 步 loss 应互不相同");
    check(!same_model_bits(s0.model, s1.model) && !same_model_bits(s1.model, s2.model) &&
              !same_model_bits(s0.model, s2.model),
          "augment=" + mode + " 下 seed 0/1/2 更新后的权重应互不相同");
  }

  // 3) 同 seed 开增强跑两次，结果逐位可复现。
  {
    const StepOut a = run_from_fixed_init(ds_path, init_ckpt, root + "/repro_a", 7, "all");
    const StepOut b = run_from_fixed_init(ds_path, init_ckpt, root + "/repro_b", 7, "all");
    check(a.loss == b.loss && same_model_bits(a.model, b.model),
          "同 seed=7 开 --augment all 两次运行应逐位可复现");
  }

  // 4) resume 未显式给 --seed 时，aug_rng_ 必须按继承自源 config.json 的 seed(42) 播种，而非默认 1234。
  {
    TrainConfig src;
    src.data_path = ds_path;
    src.out_dir = root + "/src42";
    src.epochs = 1;
    src.batch = 16;
    src.lr = 1e-3f;
    src.seed = 42;
    src.augment = "all";
    Trainer(src).run();

    TrainConfig rs_inherit;
    rs_inherit.data_path = ds_path;
    rs_inherit.out_dir = root + "/rs_inherit";
    rs_inherit.resume = src.out_dir + "/ckpt.bin";
    rs_inherit.epochs = 1;
    // rs_inherit.seed 保持默认 1234 且 seed_set=false：merge_resume_config 会把它改成 42。
    Trainer tr_inh(rs_inherit);
    tr_inh.run();

    TrainConfig rs_explicit = rs_inherit;
    rs_explicit.out_dir = root + "/rs_explicit42";
    rs_explicit.seed = 42;
    rs_explicit.seed_set = true;
    Trainer tr_exp(rs_explicit);
    tr_exp.run();

    TrainConfig rs_1234 = rs_inherit;
    rs_1234.out_dir = root + "/rs_1234";
    rs_1234.seed = 1234;
    rs_1234.seed_set = true;
    rs_1234.allow_resume_mismatch = true;
    Trainer tr_1234(rs_1234);
    tr_1234.run();

    check(same_model_bits(tr_inh.model(), tr_exp.model()),
          "resume 继承 seed=42 时增强随机流应与显式 --seed 42 逐位一致");
    check(!same_model_bits(tr_inh.model(), tr_1234.model()),
          "resume 继承 seed=42 时增强随机流不应停在构造期默认 seed=1234");
  }

  fs::remove_all(root);
  if (g_fail == 0) {
    std::cout << "test_augment_seed: PASS\n";
    return 0;
  }
  std::cerr << "test_augment_seed: " << g_fail << " 项失败\n";
  return 1;
}
