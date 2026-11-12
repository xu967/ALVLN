// H9（上一轮 D2）：--log-grad-norm（默认关）记录每轮裁剪前梯度范数的中位数、最大值与被裁剪步数占比。
//
// 验证要点：
//   1) 默认不传 --log-grad-norm 时，metrics.jsonl 不含 grad_norm 行；
//   2) 偶数步（4 步/轮）与奇数步（3 步/轮）：用 DeviceCnn 按同 seed/同顺序独立逐步复算裁剪前范数，
//      手算中位数（偶数取中间两项均值、奇数取正中项）与最大值，核对 metrics.jsonl 输出值；
//   3) 裁剪占比 clip_frac：测全不裁（0）、全裁（1，且验证 max 是裁剪前范数 >> max_grad_norm）、
//      以及部分裁剪（lr=1e-7 下参数近似不动，取中间阈值使 4 步中恰好 2 步被裁，clip_frac=0.5）。
#include <sys/wait.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/class_weights.hpp"
#include "uavrgb/dataset.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/model_dev.hpp"
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

void check_near(double got, double want, double tol, const std::string& what) {
  const double err = std::fabs(got - want);
  if (!(err <= tol)) {
    std::cerr << "FAIL: " << what << " got=" << got << " want=" << want << " |err|=" << err << "\n";
    ++g_fail;
  }
}

int run_cli(const std::string& exe, const std::string& args, const std::string& log_path) {
  const int st = std::system((exe + " " + args + " >" + log_path + " 2>&1").c_str());
  return WIFEXITED(st) ? WEXITSTATUS(st) : -1;
}

struct GradLog {
  int epoch = -1;
  double median = -1.0;
  double max = -1.0;
  double clip_frac = -1.0;
};

double kv_num(const std::string& s, const std::string& key) {
  const std::size_t p = s.find(key);
  return p == std::string::npos ? std::nan("") : std::stod(s.substr(p + key.size()));
}

std::vector<GradLog> parse_grad_logs(const std::string& metrics_path) {
  std::ifstream f(metrics_path);
  std::vector<GradLog> out;
  const std::string tag = "grad_norm epoch ";
  for (std::string line; std::getline(f, line);) {
    const std::size_t p = line.find(tag);
    if (p == std::string::npos) continue;
    GradLog g;
    g.epoch = std::stoi(line.substr(p + tag.size()));
    g.median = kv_num(line, "median=");
    g.max = kv_num(line, "max=");
    g.clip_frac = kv_num(line, "clip_frac=");
    out.push_back(g);
  }
  return out;
}

// 独立用 DeviceCnn 复演第 1 轮每步的裁剪前全局梯度范数，手算中位数、最大值与裁剪占比。
GradLog reference_epoch1(const std::string& ds_path, int batch, float lr, float max_gn,
                         unsigned seed) {
  UavrDataset ds = load_uavr(ds_path);
  CnnModel model = make_cnn(ds.h, ds.w, seed, static_cast<int>(ds.num_classes));
  TrainState st = make_train_state(model, lr);
  DeviceCnn dev(model, batch, &st);

  std::mt19937 rng(seed);
  std::vector<std::uint32_t> order = ds.train_idx;
  std::shuffle(order.begin(), order.end(), rng);

  const std::size_t plane = static_cast<std::size_t>(ds.c) * ds.h * ds.w;
  std::vector<float> norms;
  int clipped = 0;
  for (std::size_t s = 0; s < order.size(); s += static_cast<std::size_t>(batch)) {
    const std::size_t e = std::min(order.size(), s + static_cast<std::size_t>(batch));
    const int n = static_cast<int>(e - s);
    std::vector<std::uint8_t> u8(static_cast<std::size_t>(n) * plane);
    std::vector<int> y(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const std::uint32_t idx = order[s + static_cast<std::size_t>(i)];
      ds.file.clear();
      ds.file.seekg(static_cast<std::streamoff>(ds.images_off + idx * plane));
      ds.file.read(reinterpret_cast<char*>(u8.data() + i * plane), static_cast<std::streamsize>(plane));
      y[static_cast<std::size_t>(i)] = ds.labels[idx];
    }
    const Tensor logits = dev.forward(u8.data(), n, nullptr, nullptr, nullptr);
    const Tensor dlogits = class_weighted_grad(logits, y, {});
    const float gn = dev.backward_step(dlogits, max_gn, st, /*weight_decay=*/0.0f);
    norms.push_back(gn);
    if (max_gn / (gn + 1e-6f) < 1.0f) ++clipped;
  }
  std::sort(norms.begin(), norms.end());
  const std::size_t m = norms.size();
  const double med = (m % 2 == 1)
                         ? static_cast<double>(norms[m / 2])
                         : 0.5 * (static_cast<double>(norms[m / 2 - 1]) +
                                  static_cast<double>(norms[m / 2]));
  return {1, med, static_cast<double>(norms.back()),
          static_cast<double>(clipped) / static_cast<double>(m)};
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "用法: test_log_grad_norm <uavrgb_train 路径>\n";
    return 2;
  }
  const std::string exe = argv[1];
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_grad_norm_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  // 8 个 train 样本（batch=2 时 4 步偶数；batch=3 时 3 步奇数）+ 4 个 val 样本。
  const auto sd = uavrgb_test::write_synthetic_uavr(root + "/ds.uavr", 12, 16, 16, 8, 17);
  const std::string base = "--data " + sd.path + " --epochs 2 --seed 5";

  // 1) 默认关闭：metrics.jsonl 不得出现 grad_norm 行。
  {
    const int rc = run_cli(exe, base + " --batch 2 --lr 1e-3 --out " + root + "/off", root + "/off.log");
    check(rc == 0, "默认不传 --log-grad-norm 应成功");
    check(parse_grad_logs(root + "/off/metrics.jsonl").empty(),
          "默认关闭时 metrics.jsonl 不应含 grad_norm 行");
  }

  // 2) 偶数步（4 步/轮）+ 极大阈值（clip_frac=0）：核对 epoch 1/2 都有记录，且 epoch 1 与独立手算一致。
  {
    const int rc = run_cli(exe,
                           base + " --batch 2 --lr 1e-7 --max-grad-norm 1000 --log-grad-norm --out " +
                               root + "/even_noclip",
                           root + "/even_noclip.log");
    check(rc == 0, "--log-grad-norm 偶数步应 rc=0");
    const auto logs = parse_grad_logs(root + "/even_noclip/metrics.jsonl");
    check(logs.size() == 2, "2 轮训练应产出 2 条 grad_norm 记录");
    const GradLog ref = reference_epoch1(sd.path, 2, 1e-7f, 1000.0f, 5);
    if (!logs.empty()) {
      check(logs[0].epoch == 1 && logs[1].epoch == 2, "epoch 编号应为 1, 2");
      check_near(logs[0].median, ref.median, 1e-5, "偶数步 median");
      check_near(logs[0].max, ref.max, 1e-5, "偶数步 max");
      check_near(logs[0].clip_frac, 0.0, 1e-9, "极大阈值下 clip_frac 应为 0");
      check(logs[0].median < logs[0].max, "4 个互异批次的 median 应严格小于 max");
    }
  }

  // 3) 奇数步（batch=3 -> 3 步/轮）：核对奇数中位数与最大值。
  {
    const int rc = run_cli(
        exe, "--data " + sd.path + " --epochs 1 --batch 3 --lr 1e-3 --seed 5 --log-grad-norm --out " +
                 root + "/odd",
        root + "/odd.log");
    check(rc == 0, "--log-grad-norm 奇数步应 rc=0");
    const auto logs = parse_grad_logs(root + "/odd/metrics.jsonl");
    const GradLog ref = reference_epoch1(sd.path, 3, 1e-3f, 5.0f, 5);
    check(logs.size() == 1, "1 轮训练应产出 1 条 grad_norm 记录");
    if (!logs.empty()) {
      check_near(logs[0].median, ref.median, 1e-5, "奇数步 median");
      check_near(logs[0].max, ref.max, 1e-5, "奇数步 max");
      check_near(logs[0].clip_frac, ref.clip_frac, 1e-9, "奇数步 clip_frac");
    }
  }

  // 4) 全裁剪（max_grad_norm=1e-4）：clip_frac=1，且记录的是裁剪前范数（max 远大于 1e-4）。
  {
    const int rc = run_cli(
        exe, "--data " + sd.path +
                 " --epochs 1 --batch 2 --lr 1e-7 --seed 5 --max-grad-norm 1e-4 --log-grad-norm --out " +
                 root + "/all_clip",
        root + "/all_clip.log");
    check(rc == 0, "全裁剪场景应 rc=0");
    const auto logs = parse_grad_logs(root + "/all_clip/metrics.jsonl");
    const GradLog ref = reference_epoch1(sd.path, 2, 1e-7f, 1e-4f, 5);
    check(logs.size() == 1, "全裁剪场景应有 1 条记录");
    if (!logs.empty()) {
      check_near(logs[0].clip_frac, 1.0, 1e-9, "极小阈值下 clip_frac 应为 1");
      check(logs[0].max > 1e-2, "记录的 max 必须是裁剪前梯度范数（应 >> 1e-4）");
      check_near(logs[0].median, ref.median, 1e-5, "全裁剪 median");
      check_near(logs[0].max, ref.max, 1e-5, "全裁剪 max");
    }

    // 5) 部分裁剪：lr=1e-7 下步间参数变化极微，取无裁剪时的中位数作阈值，4 步中恰好 2 步被裁（0.5）。
    const GradLog unclipped = reference_epoch1(sd.path, 2, 1e-7f, 1000.0f, 5);
    std::ostringstream thr;
    thr.precision(9);
    thr << unclipped.median;
    const int rc2 = run_cli(
        exe, "--data " + sd.path + " --epochs 1 --batch 2 --lr 1e-7 --seed 5 --max-grad-norm " +
                 thr.str() + " --log-grad-norm --out " + root + "/half_clip",
        root + "/half_clip.log");
    check(rc2 == 0, "半裁剪场景应 rc=0");
    const auto half_logs = parse_grad_logs(root + "/half_clip/metrics.jsonl");
    const GradLog half_ref = reference_epoch1(sd.path, 2, 1e-7f, static_cast<float>(unclipped.median), 5);
    check(!half_logs.empty() && half_ref.clip_frac == 0.5, "半裁剪参考夹具应触发 2/4=0.5 裁剪");
    if (!half_logs.empty()) {
      check_near(half_logs[0].clip_frac, 0.5, 1e-9, "半裁剪场景 clip_frac 应精确为 0.5");
      check_near(half_logs[0].median, half_ref.median, 1e-5, "半裁剪 median");
    }
  }

  fs::remove_all(root);
  if (g_fail == 0) {
    std::cout << "test_log_grad_norm: PASS\n";
    return 0;
  }
  std::cerr << "test_log_grad_norm: " << g_fail << " 项失败\n";
  return 1;
}
