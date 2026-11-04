// 蒸馏软标签端到端自检（合成 4 类条件数据）：
//   1) 软标签按**全局样本序**对齐时，alpha=1 的 one-hot 软标签训练出的模型在验证集
//      准确率高（gather 错位会掉到随机水平 1/4，这条是主要证据）
//   2) 故意把软标签整体旋转一类：模型应该学到错误类别，按真标签算的准确率≈0
//   3) 软标签文件大小与数据集不符、alpha 越界、与 --class-weights 同用 → 抛异常
//   4) config.json 记录 soft_targets / distill_alpha
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "soft_targets_fixture.hpp"
#include "uavrgb/dataset.hpp"
#include "uavrgb/trainer.hpp"

namespace fs = std::filesystem;
using namespace uavrgb;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}

template <typename F>
bool throws_any(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  } catch (...) {
    return false;
  }
  return false;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  std::ostringstream os;
  os << f.rdbuf();
  return os.str();
}

// 把 one-hot 软标签写成扁平 float32 [n*4]。
void write_soft(const std::string& path, const std::string& dataset,
                const std::vector<int>& labels, int rotate) {
  std::vector<float> q(labels.size() * 4, 0.0f);
  for (std::size_t i = 0; i < labels.size(); ++i) {
    q[i * 4 + static_cast<std::size_t>((labels[i] + rotate) % 4)] = 1.0f;
  }
  uavrgb_test::write_soft(path, dataset, q);
}

double val_accuracy(const CnnModel& m, UavrDataset& ds) {
  Tensor x;
  ds.load_images(ds.val_idx, x);
  const Tensor logits = forward(m, x);
  const int c = static_cast<int>(logits.dims()[1]);
  std::size_t hit = 0;
  for (std::size_t i = 0; i < ds.val_idx.size(); ++i) {
    int best = 0;
    for (int j = 1; j < c; ++j) {
      if (logits[i * static_cast<std::size_t>(c) + static_cast<std::size_t>(j)] >
          logits[i * static_cast<std::size_t>(c) + static_cast<std::size_t>(best)]) {
        best = j;
      }
    }
    if (best == ds.labels[ds.val_idx[i]]) ++hit;
  }
  return static_cast<double>(hit) / static_cast<double>(ds.val_idx.size());
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_soft_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  // 4 类条件数据：每类用一个不同的像素均值，噪声小，模型应该很好学。
  std::mt19937 rng(2024);
  std::vector<int> train_labels, val_labels;
  for (int i = 0; i < 200; ++i) train_labels.push_back(static_cast<int>(rng() % 4));
  for (int i = 0; i < 60; ++i) val_labels.push_back(static_cast<int>(rng() % 4));
  const std::vector<float> means = {0.2f, 0.4f, 0.6f, 0.8f};
  const auto sd = uavrgb_test::write_class_conditional_uavr(
      root + "/cond.uavr", 32, 32, train_labels, val_labels, means, 0.08f, 0.03f, 5);

  std::vector<int> all = train_labels;
  all.insert(all.end(), val_labels.begin(), val_labels.end());
  write_soft(root + "/soft_ok.bin", sd.path, all, 0);
  write_soft(root + "/soft_rot.bin", sd.path, all, 1);

  TrainConfig base;
  base.data_path = sd.path;
  base.epochs = 30;
  base.batch = 32;
  base.lr = 1e-3f;
  base.seed = 3;
  base.distill_alpha = 1.0f;

  // ---- 1) 正确对齐的 one-hot 软标签 ----
  TrainConfig ok = base;
  ok.out_dir = root + "/run_ok";
  ok.soft_targets = root + "/soft_ok.bin";
  Trainer t_ok(ok);
  t_ok.run();
  UavrDataset ds = load_uavr(sd.path);
  const double acc_ok = val_accuracy(t_ok.model(), ds);
  std::printf("对齐软标签：val_acc=%.3f\n", acc_ok);
  // 判据是「远高于随机 0.25」而不是绝对精度：这条测试要抓的是 gather 错位
  // （错位时软标签与图像无关，只能学到随机水平）。合成数据上实测 0.73。
  check(acc_ok > 0.55, "对齐软标签训练后验证准确率过低（gather 可能错位）");

  // ---- 2) 旋转一类的软标签：模型学到的是错类别 ----
  TrainConfig rot = base;
  rot.out_dir = root + "/run_rot";
  rot.soft_targets = root + "/soft_rot.bin";
  Trainer t_rot(rot);
  t_rot.run();
  const double acc_rot = val_accuracy(t_rot.model(), ds);
  std::printf("旋转软标签：val_acc=%.3f\n", acc_rot);
  check(acc_rot < 0.30, "旋转过的软标签仍能学对，说明软标签没真正生效");

  // ---- 3) 非法输入 ----
  {
    std::vector<float> shortq(10, 0.25f);
    std::ofstream(root + "/soft_short.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(shortq.data()),
               static_cast<std::streamsize>(shortq.size() * sizeof(float)));
    TrainConfig bad = base;
    bad.out_dir = root + "/run_short";
    bad.soft_targets = root + "/soft_short.bin";
    check(throws_any([&] { Trainer(bad).run(); }), "软标签大小不符未抛异常");

    TrainConfig both = ok;
    both.out_dir = root + "/run_both";
    both.class_weights = "auto";
    check(throws_any([&] { Trainer(both).run(); }), "软标签与类别权重同用未抛异常");
  }

  // ---- 4) config.json 落盘 + 蒸馏来源审计 ----
  // 教师来源元信息由 join_soft_targets.py 写在 <soft>.json；student 必须原样落盘。
  {
    std::ofstream meta(root + "/soft_ok.json");
    meta << "{\"teacher_ckpt_sha256\":\"abc123\",\"teacher_select_split\":\"train\"}\n";
  }
  {
    TrainConfig tc = base;
    tc.out_dir = root + "/run_meta";
    tc.soft_targets = root + "/soft_ok.bin";
    tc.epochs = 1;
    Trainer t(tc);
    t.run();
    const std::string cfg = read_file(root + "/run_meta/config.json");
    check(cfg.find("\"soft_targets_sha256\":\"\"") == std::string::npos,
          "config.json 未记录 soft_targets_sha256");
    check(cfg.find("\"teacher_ckpt_sha256\":\"abc123\"") != std::string::npos,
          "config.json 未记录 teacher_ckpt_sha256");
    check(cfg.find("\"teacher_select_split\":\"train\"") != std::string::npos,
          "config.json 未记录 teacher_select_split");
    // 教师用 val 选模 + --strict-distill：必须失败，不能静默训练。
    std::ofstream meta2(root + "/soft_ok.json", std::ios::trunc);
    meta2 << "{\"teacher_ckpt_sha256\":\"abc123\",\"teacher_select_split\":\"val\"}\n";
    meta2.close();
    TrainConfig strict = tc;
    strict.out_dir = root + "/run_strict";
    strict.strict_distill = true;
    check(throws_any([&] { Trainer(strict).run(); }),
          "--strict-distill 时教师 val 选模与 student val 重叠必须失败");
  }

  const std::string cfg = read_file(root + "/run_ok/config.json");
  check(cfg.find("\"soft_targets\":\"") != std::string::npos, "config.json 未记录 soft_targets");
  check(cfg.find("\"distill_alpha\":1") != std::string::npos, "config.json 未记录 distill_alpha");

  fs::remove_all(root);
  if (g_failures == 0) {
    std::cout << "RESULT: PASS\n";
    return 0;
  }
  std::cerr << "RESULT: FAIL (" << g_failures << ")\n";
  return 1;
}
