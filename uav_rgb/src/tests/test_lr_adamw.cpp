// D3：warmup、余弦退火、AdamW 解耦权重衰减的手算数值测试（卡 C1a 第 1 条）。
//
// 1. 学习率日程（Trainer::step_lr，经 Trainer::run 写进 metrics.jsonl 的 train_step.lr）：
//    设基础学习率 lr_0，warmup 步数 W >= 0，总步数 T >= 1，步号 s = 1..T：
//      s <= W (W > 0):               lr(s) = lr_0 * (s / W)
//      s >  W 且 cosine_lr 且 T > W: p = (s - W) / (T - W), lr(s) = lr_0 * 0.5 * (1 + cos(pi * p))
//      其余:                         lr(s) = lr_0
//    手算参考值：
//    a) T = 7 (14 样本, batch=2, epochs=1), W = 3, lr_0 = 0.06, cosine_lr = true (T - W = 4)：
//       s=1: 0.06 * 1/3 = 0.02
//       s=2: 0.06 * 2/3 = 0.04
//       s=3: 0.06 * 3/3 = 0.06
//       s=4: p=0.25, cos(pi/4)=sqrt(2)/2   -> 0.03 * (1 + sqrt(2)/2) = 0.051213203435596426
//       s=5: p=0.50, cos(pi/2)=0           -> 0.03 * 1               = 0.03
//       s=6: p=0.75, cos(3pi/4)=-sqrt(2)/2 -> 0.03 * (1 - sqrt(2)/2) = 0.008786796564403574
//       s=7: p=1.00, cos(pi)=-1            -> 0.03 * 0               = 0.0
//    b) T = 4 (8 样本, batch=2, epochs=1), W = 0, lr_0 = 0.04, cosine_lr = true（E2E-R1 口径）：
//       s=1..4: p = 0.25, 0.50, 0.75, 1.00 ->
//               0.03414213562373095, 0.02, 0.00585786437626905, 0.0
//    c) T = 7, W = 3, lr_0 = 0.06, cosine_lr = false：
//       s=1..3 为 0.02, 0.04, 0.06；s=4..7 恒为 0.06。
//
// 2. AdamW 解耦权重衰减（dev::adam_step 与 DeviceCnn::backward_step）：
//    给定步号 t >= 1，lr = 0.1, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.25, m_0 = v_0 = 0：
//      m_t = b1*m_{t-1} + (1-b1)*g_t,  v_t = b2*v_{t-1} + (1-b2)*g_t^2
//      m_hat = m_t / (1 - b1^t),       v_hat = v_t / (1 - b2^t)
//      p_t = p_{t-1} - lr * (m_hat / (sqrt(v_hat) + eps) + wd * p_{t-1})
//    三个分量的两步手算值（float32 下 1-0.999f 有 ~1.3e-8 舍入，bc 按 model_dev.cu 同式由 float b1/b2 计算）：
//    - 分量 0（零梯度，区分 AdamW 解耦衰减与 L2 正则）：p_0 = 2.0, g_1 = 0, g_2 = 0
//      t=1: m_1=0, v_1=0, p_1 = 2.0 - 0.1*(0 + 0.25*2.0) = 1.95（L2 正则会算出 ~1.90）
//      t=2: m_2=0, v_2=0, p_2 = 1.95 - 0.1*(0 + 0.25*1.95) = 1.90125
//    - 分量 1（正负翻转梯度，同时约束 bc1/bc2/m/v/wd）：p_0 = 2.0, g_1 = 0.5, g_2 = -1.5
//      t=1 (bc1=0.1, bc2=0.001):
//        m_1 = 0.05 -> m_hat = 0.5; v_1 = 0.00025 -> v_hat = 0.25 -> sqrt(v_hat) = 0.5
//        adam_1 = 0.5 / (0.5 + 1e-8) = 0.99999998
//        p_1 = 2.0 - 0.1 * (0.99999998 + 0.5) = 1.850000002
//      t=2 (bc1=0.19, bc2=0.001999):
//        m_2 = 0.9*0.05 + 0.1*(-1.5) = -0.105 -> m_hat = -0.105 / 0.19 = -0.5526315789473685
//        v_2 = 0.999*0.00025 + 0.001*2.25 = 0.00249975 -> v_hat = 1.2505002501250626
//        sqrt(v_hat) = 1.118257685028841, adam_2 = -0.49418983134077555
//        decay_2 = 0.25 * 1.850000002 = 0.4625000005
//        p_2 = 1.850000002 - 0.1 * (-0.49418983134077555 + 0.4625000005) = 1.853168985
//    - 分量 2（负参数 + 负梯度）：p_0 = -4.0, g_1 = -2.0, g_2 = -2.0
//      t=1: m_1 = -0.2, v_1 = 0.004, adam_1 = -2/(2+1e-8) = -0.999999995, decay_1 = -1.0
//           p_1 = -4.0 - 0.1*(-1.999999995) = -3.8000000005
//      t=2: m_2 = -0.38, v_2 = 0.007996, adam_2 = -0.999999995, decay_2 = 0.25*(-3.8) = -0.95
//           p_2 = -3.8000000005 - 0.1*(-1.949999995) = -3.605000001
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/model_dev.hpp"
#include "uavrgb/ops_dev.hpp"
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

std::vector<double> read_step_lrs(const std::string& metrics_path) {
  std::ifstream f(metrics_path);
  std::vector<double> out;
  const std::string key = "\"lr\":";
  for (std::string line; std::getline(f, line);) {
    if (line.find("\"event\":\"train_step\"") == std::string::npos) continue;
    const std::size_t p = line.find(key);
    check(p != std::string::npos, "train_step 缺 lr 字段: " + line);
    if (p != std::string::npos) out.push_back(std::stod(line.substr(p + key.size())));
  }
  return out;
}

void run_lr_case(const std::string& ds_path, const std::string& out_dir, float lr0, int warmup,
                 bool cosine, unsigned seed, const std::vector<double>& want, const char* tag) {
  TrainConfig cfg;
  cfg.data_path = ds_path;
  cfg.out_dir = out_dir;
  cfg.epochs = 1;
  cfg.batch = 2;
  cfg.lr = lr0;
  cfg.warmup_steps = warmup;
  cfg.cosine_lr = cosine;
  cfg.seed = seed;
  Trainer tr(cfg);
  tr.run();
  const std::vector<double> got = read_step_lrs(cfg.out_dir + "/metrics.jsonl");
  check(got.size() == want.size(), std::string(tag) + " 步数不符");
  for (std::size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
    check_near(got[i], want[i], 1e-7, std::string(tag) + " step " + std::to_string(i + 1));
  }
  if (!want.empty()) {
    check_near(tr.state().adam[0].lr, want.back(), 1e-7, std::string(tag) + " 末步 adam.lr");
  }
}

void test_lr_schedules(const std::string& root) {
  const auto ds14 = uavrgb_test::write_synthetic_uavr(root + "/ds14.uavr", 18, 16, 16, 14, 3);
  const auto ds8 = uavrgb_test::write_synthetic_uavr(root + "/ds8.uavr", 12, 16, 16, 8, 5);
  const double s2 = std::sqrt(2.0) * 0.5;
  run_lr_case(ds14.path, root + "/lr_warm_cos", 0.06f, 3, true, 1,
              {0.02, 0.04, 0.06, 0.03 * (1.0 + s2), 0.03, 0.03 * (1.0 - s2), 0.0}, "1a) warm+cos");
  run_lr_case(ds8.path, root + "/lr_cos_only", 0.04f, 0, true, 2,
              {0.02 * (1.0 + s2), 0.02, 0.02 * (1.0 - s2), 0.0}, "1b) cos_only");
  run_lr_case(ds14.path, root + "/lr_warm_only", 0.06f, 3, false, 3,
              {0.02, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06}, "1c) warm_only");
}

void test_adamw_kernel_hand_calc() {
  constexpr int kN = 3;
  const float p0[kN] = {2.0f, 2.0f, -4.0f};
  const float g1[kN] = {0.0f, 0.5f, -2.0f};
  const float g2[kN] = {0.0f, -1.5f, -2.0f};
  const float z[kN] = {0.0f, 0.0f, 0.0f};

  float* buf = nullptr;  // [p, g, m, v] 紧邻分配，省重复样板
  cudaMalloc(&buf, 4 * sizeof p0);
  float *dp = buf, *dg = buf + kN, *dm = buf + 2 * kN, *dv = buf + 3 * kN;
  cudaMemcpy(dp, p0, sizeof p0, cudaMemcpyHostToDevice);
  cudaMemcpy(dm, z, sizeof z, cudaMemcpyHostToDevice);
  cudaMemcpy(dv, z, sizeof z, cudaMemcpyHostToDevice);

  const float lr = 0.1f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.25f;
  auto bc = [](float b, int t) {
    return static_cast<float>(1.0 - std::pow(static_cast<double>(b), t));
  };

  // 第 1 步
  cudaMemcpy(dg, g1, sizeof g1, cudaMemcpyHostToDevice);
  dev::adam_step(dp, dg, dm, dv, lr, b1, b2, bc(b1, 1), bc(b2, 1), eps, wd, kN);
  cudaDeviceSynchronize();
  float p1[kN]{}, m1[kN]{}, v1[kN]{};
  cudaMemcpy(p1, dp, sizeof p1, cudaMemcpyDeviceToHost);
  cudaMemcpy(m1, dm, sizeof m1, cudaMemcpyDeviceToHost);
  cudaMemcpy(v1, dv, sizeof v1, cudaMemcpyDeviceToHost);

  const double want_p1[kN] = {1.95, 1.850000002, -3.8000000005};
  const double want_m1[kN] = {0.0, 0.05, -0.2};
  const double want_v1[kN] = {0.0, 0.00025, 0.004};
  for (int i = 0; i < kN; ++i) {
    check_near(p1[i], want_p1[i], 1e-6, "2a) AdamW step1 p[" + std::to_string(i) + "]");
    check_near(m1[i], want_m1[i], 5e-7, "2a) AdamW step1 m[" + std::to_string(i) + "]");
    check_near(v1[i], want_v1[i], 5e-7, "2a) AdamW step1 v[" + std::to_string(i) + "]");
  }

  // 第 2 步
  cudaMemcpy(dg, g2, sizeof g2, cudaMemcpyHostToDevice);
  dev::adam_step(dp, dg, dm, dv, lr, b1, b2, bc(b1, 2), bc(b2, 2), eps, wd, kN);
  cudaDeviceSynchronize();
  float p2[kN]{}, m2[kN]{}, v2[kN]{};
  cudaMemcpy(p2, dp, sizeof p2, cudaMemcpyDeviceToHost);
  cudaMemcpy(m2, dm, sizeof m2, cudaMemcpyDeviceToHost);
  cudaMemcpy(v2, dv, sizeof v2, cudaMemcpyDeviceToHost);

  const double want_p2[kN] = {1.90125, 1.853168985, -3.605000001};
  const double want_m2[kN] = {0.0, -0.105, -0.38};
  const double want_v2[kN] = {0.0, 0.00249975, 0.007996};
  for (int i = 0; i < kN; ++i) {
    check_near(p2[i], want_p2[i], 1e-6, "2a) AdamW step2 p[" + std::to_string(i) + "]");
    check_near(m2[i], want_m2[i], 5e-7, "2a) AdamW step2 m[" + std::to_string(i) + "]");
    check_near(v2[i], want_v2[i], 5e-7, "2a) AdamW step2 v[" + std::to_string(i) + "]");
  }
  cudaFree(buf);
}

void test_device_cnn_weight_decay_wiring() {
  // 覆盖全部分支参数段（卷积/fc/text_emb/film/aux，且把偏置也置非零）：零梯度下 p_1 = p_0*(1 - lr*wd)。
  CnnModel m0 = make_cnn(16, 16, /*seed=*/9, /*num_classes=*/4, /*text_dim=*/8, /*in_c=*/6,
                         /*aux_classes=*/8, /*prev_dim=*/4);
  for (Tensor* t : cnn_params(m0)) {
    for (std::size_t i = 0; i < t->numel(); ++i) {
      if ((*t)[i] == 0.0f) (*t)[i] = 0.25f * static_cast<float>(static_cast<int>(i % 5) - 2) + 0.1f;
    }
  }
  const float lr = 0.1f, wd = 0.2f;  // 1 - lr*wd = 0.98
  TrainState st = make_train_state(m0, lr);
  DeviceCnn dev(m0, /*max_batch=*/2, &st);
  const std::vector<std::uint8_t> img(2 * 6 * 16 * 16, 100);
  const std::vector<std::uint8_t> txt(2 * kTextLen, 1);
  const std::vector<std::uint8_t> prv(2, 0);
  (void)dev.forward(img.data(), 2, txt.data(), nullptr, prv.data());
  dev.backward_step(Tensor::zeros({2, 4}), /*max_grad_norm=*/5.0f, st, wd);
  CnnModel m1 = m0;
  dev.download(m1, &st);

  const auto p_before = cnn_params(m0);
  const auto p_after = cnn_params(m1);
  double max_err = 0.0;
  for (std::size_t s = 0; s < p_before.size(); ++s) {
    for (std::size_t i = 0; i < p_before[s]->numel(); ++i) {
      max_err = std::max(max_err, std::fabs(static_cast<double>((*p_after[s])[i]) -
                                            0.98 * static_cast<double>((*p_before[s])[i])));
    }
  }
  std::cout << "DeviceCnn AdamW zero-grad wiring max_err = " << max_err << "\n";
  check(max_err < 1e-6, "2b) DeviceCnn::backward_step 零梯度下全参数段应精确乘 (1 - lr*wd)=0.98");
}

}  // namespace

int main() {
  const std::string root = (fs::temp_directory_path() /
                            ("uavrgb_test_lr_adamw_" + std::to_string(std::random_device{}())))
                               .string();
  fs::create_directories(root);

  test_lr_schedules(root);
  test_adamw_kernel_hand_calc();
  test_device_cnn_weight_decay_wiring();

  fs::remove_all(root);
  if (g_fail == 0) {
    std::cout << "test_lr_adamw: PASS\n";
    return 0;
  }
  std::cerr << "test_lr_adamw: " << g_fail << " 项失败\n";
  return 1;
}
