// 文本分支自检（L2 / D-STACK=自研）：
// 1) token 真的影响 logits，<pad> 不影响；
// 2) embedding 的解析梯度与中心差分一致（含「没出现的 token 梯度为 0」）；
// 3) instructions.jsonl 侧车能按样本展开成 token id；
// 4) checkpoint v4 能带着文本 embedding 往返，结构不符时拒绝加载。
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/dataset.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/text.hpp"

#include "synthetic_dataset.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(cond)                                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
      ++failures;                                                            \
    }                                                                        \
  } while (0)

using uavrgb::CnnCache;
using uavrgb::CnnGrad;
using uavrgb::CnnModel;
using uavrgb::Tensor;
using uavrgb::UavrDataset;

constexpr int kH = 8;
constexpr int kW = 8;
constexpr int kD = 2;
constexpr int kClasses = 4;

Tensor make_input(int n) {
  Tensor x({n, 3, kH, kW});
  for (std::size_t i = 0; i < x.numel(); ++i) x[i] = 0.001f * static_cast<float>(i % 97);
  return x;
}

double sum_logits(const CnnModel& m, const Tensor& x, const std::vector<std::uint8_t>& text) {
  const Tensor y = uavrgb::forward(m, x, nullptr, &text);
  double s = 0.0;
  for (std::size_t i = 0; i < y.numel(); ++i) s += static_cast<double>(y[i]);
  return s;
}

void test_text_affects_logits() {
  CnnModel m = uavrgb::make_cnn(kH, kW, 7, kClasses, kD);
  const Tensor x = make_input(1);
  std::vector<std::uint8_t> a(uavrgb::kTextLen, 0);
  std::vector<std::uint8_t> b = a;
  a[0] = 3;  // 两个不同的 color token
  b[0] = 4;
  const Tensor la = uavrgb::forward(m, x, nullptr, &a);
  const Tensor lb = uavrgb::forward(m, x, nullptr, &b);
  bool diff = false;
  for (std::size_t i = 0; i < la.numel(); ++i) diff = diff || (la[i] != lb[i]);
  CHECK(diff);  // 指令不同，logits 必须不同

  // 位置感知：同一个 token 放在位置 0 与位置 3，结果必须不同
  //（模板把颜色 token 放在固定槽位；mean-pool 会把这个区别抹掉）。
  std::vector<std::uint8_t> c(uavrgb::kTextLen, 0);
  c[3] = 3;
  const Tensor lc = uavrgb::forward(m, x, nullptr, &c);
  bool pos_diff = false;
  for (std::size_t i = 0; i < la.numel(); ++i) pos_diff = pos_diff || (la[i] != lc[i]);
  CHECK(pos_diff);

  // <pad>(0) 槽位不贡献：多一个全零位置不改变结果。
  std::vector<std::uint8_t> d = c;
  d[5] = 0;
  const Tensor ld = uavrgb::forward(m, x, nullptr, &d);
  for (std::size_t i = 0; i < la.numel(); ++i) CHECK(lc[i] == ld[i]);

  // 没有文本分支的模型不接受 text。
  CnnModel plain = uavrgb::make_cnn(kH, kW, 7, kClasses, 0);
  bool threw = false;
  try {
    uavrgb::forward(plain, x, nullptr, &a);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

void test_embedding_gradcheck() {
  CnnModel m = uavrgb::make_cnn(kH, kW, 11, kClasses, kD);
  const Tensor x = make_input(2);
  std::vector<std::uint8_t> text(2 * uavrgb::kTextLen, 0);
  text[0] = 6;  // 样本 0 用 token 6
  text[uavrgb::kTextLen] = 8;  // 样本 1 用 token 8

  CnnCache cache;
  uavrgb::forward(m, x, &cache, &text);
  Tensor dlogits({2, kClasses});
  for (std::size_t i = 0; i < dlogits.numel(); ++i) dlogits[i] = 1.0f;
  const CnnGrad g = uavrgb::backward(m, cache, dlogits);
  CHECK(g.text_demb.numel() == static_cast<std::size_t>(uavrgb::kTextVocab) * kD);

  const float eps = 1e-3f;
  double max_err = 0.0;
  std::size_t worst = 0;
  double worst_ana = 0.0, worst_num = 0.0;
  for (std::size_t i = 0; i < m.text_emb.numel(); ++i) {
    const float old = m.text_emb[i];
    m.text_emb[i] = old + eps;
    const double lp = sum_logits(m, x, text);
    m.text_emb[i] = old - eps;
    const double lm = sum_logits(m, x, text);
    m.text_emb[i] = old;
    const double num = (lp - lm) / (2.0 * static_cast<double>(eps));
    const double err = std::fabs(num - static_cast<double>(g.text_demb[i]));
    if (err > max_err) {
      max_err = err;
      worst = i;
      worst_ana = static_cast<double>(g.text_demb[i]);
      worst_num = num;
    }
  }
  std::printf("text gradcheck max_err=%.3e at emb[%zu] (ana=%.6f num=%.6f)\n", max_err, worst,
              worst_ana, worst_num);
  CHECK(max_err < 2e-2);

  // 没出现在 batch 里的 token 行梯度必须是精确 0。
  const int absent = 20;
  for (int d = 0; d < kD; ++d) {
    CHECK(g.text_demb[static_cast<std::size_t>(absent) * kD + d] == 0.0f);
  }
}

void test_film_gradcheck() {
  // FiLM 的 γ/β 必须真的收到正确梯度；否则训练时 film.W 会一直停在初始化，
  // 文本仍然进不了决策（这正是 concat 版本的失败模式）。
  CnnModel m = uavrgb::make_cnn(kH, kW, 13, kClasses, kD);
  const Tensor x = make_input(2);
  std::vector<std::uint8_t> text(2 * uavrgb::kTextLen, 0);
  text[0] = 6;
  text[uavrgb::kTextLen + 3] = 9;
  CnnCache cache;
  uavrgb::forward(m, x, &cache, &text);
  Tensor dlogits({2, kClasses});
  for (std::size_t i = 0; i < dlogits.numel(); ++i) dlogits[i] = 1.0f;
  const CnnGrad g = uavrgb::backward(m, cache, dlogits);
  CHECK(g.film.dW.numel() == m.film.W.numel());

  const float eps = 1e-3f;
  double max_err = 0.0;
  for (std::size_t i = 0; i < m.film.W.numel(); i += 17) {
    const float old = m.film.W[i];
    CnnModel mp = m;
    CnnModel mm = m;
    mp.film.W[i] = old + eps;
    mm.film.W[i] = old - eps;
    const double lp = sum_logits(mp, x, text);
    const double lm = sum_logits(mm, x, text);
    const double num = (lp - lm) / (2.0 * eps);
    max_err = std::max(max_err, std::abs(num - static_cast<double>(g.film.dW[i])));
  }
  CHECK(max_err < 1e-2);
}

void test_sidecar() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "uavrgb_text_branch_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::string uavr = (dir / "dataset.uavr").string();
  uavrgb_test::write_synthetic_uavr(uavr, 4, kH, kW, /*train_n=*/4, /*seed=*/42);

  // 没有侧车时 load_text 必须报错，而不是返回全 pad。
  UavrDataset plain = uavrgb::load_uavr(uavr);
  CHECK(plain.text_ids.empty());
  bool threw = false;
  std::vector<std::uint8_t> out;
  try {
    plain.load_text(plain.train_idx, out);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);

  {
    std::ofstream f(dir / "instructions.jsonl");
    // 与 preprocess.py（json.dumps 默认分隔符）一致：冒号后有空格。
    f << "{\"episode_id\": 1,\"instruction\": \"fly to the red pillar and stop\", "
         "\"sample_start\": 0, \"sample_count\": 4, \"text_ids\": [2, 4, 5, 6, 10, 11, 13]}\n";
  }
  UavrDataset ds = uavrgb::load_uavr(uavr);
  CHECK(ds.text_ids.size() == 4 * static_cast<std::size_t>(uavrgb::kTextLen));
  ds.load_text(ds.train_idx, out);
  CHECK(out.size() == ds.train_idx.size() * static_cast<std::size_t>(uavrgb::kTextLen));
  CHECK(out[0] == 2);
  CHECK(out[6] == 13);
  CHECK(out[7] == 0);  // 不足 kTextLen 的位置补 <pad>

  {
    // 旧侧车（L2 之前没有 text_ids）：当成没有侧车，不能因此拒绝整个数据集——
    // 否则旧数据集连 image-only 训练/eval 都跑不了。
    std::ofstream f(dir / "instructions.jsonl", std::ios::trunc);
    f << "{\"episode_id\": 1,\"instruction\": \"fly to the red pillar and stop\", "
         "\"sample_start\": 0, \"sample_count\": 4}\n";
  }
  CHECK(uavrgb::load_uavr(uavr).text_ids.empty());

  {
    // 3 行侧车中 1 行缺 text_ids：半迁移的侧车必须报错并指出坏行行号，
    // 不能静默丢掉整份侧车（否则文本模型会误报「没有侧车」）。
    // 6 样本 → train(0,2) / val(2,2) / test(4,2)，正好 3 个 episode。
    const fs::path mixed_dir = dir / "mixed";
    fs::create_directories(mixed_dir);
    const std::string mixed_uavr = (mixed_dir / "dataset.uavr").string();
    uavrgb_test::write_synthetic_uavr(mixed_uavr, 6, kH, kW, /*train_n=*/2, /*seed=*/43,
                                      /*test_n=*/2);
    {
      std::ofstream f(mixed_dir / "instructions.jsonl");
      f << "{\"episode_id\": 1, \"sample_start\": 0, \"sample_count\": 2, \"text_ids\": [2]}\n"
           "{\"episode_id\": 2, \"sample_start\": 2, \"sample_count\": 2}\n"
           "{\"episode_id\": 3, \"sample_start\": 4, \"sample_count\": 2, \"text_ids\": [3]}\n";
    }
    bool mixed_threw = false;
    std::string mixed_msg;
    try {
      (void)uavrgb::load_uavr(mixed_uavr);
    } catch (const std::runtime_error& e) {
      mixed_threw = true;
      mixed_msg = e.what();
    }
    CHECK(mixed_threw);
    CHECK(mixed_msg.find("第 2 行") != std::string::npos);
    CHECK(mixed_msg.find("缺 text_ids") != std::string::npos);

    // 全行缺 text_ids 但未覆盖全部 episode：仍要报错，不能提前 return 绕过覆盖校验。
    {
      std::ofstream f(mixed_dir / "instructions.jsonl", std::ios::trunc);
      f << "{\"episode_id\": 1, \"sample_start\": 0, \"sample_count\": 2}\n"
           "{\"episode_id\": 2, \"sample_start\": 2, \"sample_count\": 2}\n";
    }
    bool incomplete_threw = false;
    try {
      (void)uavrgb::load_uavr(mixed_uavr);
    } catch (const std::runtime_error&) {
      incomplete_threw = true;
    }
    CHECK(incomplete_threw);
  }

  {
    // 10 个 token 超过 kTextLen=9；静默截断会悄悄改掉指令的意思，必须报错。
    std::ofstream f(dir / "instructions.jsonl", std::ios::trunc);
    f << "{\"episode_id\": 1,\"instruction\": \"pass the red pillar on your left and stop now\", "
         "\"sample_start\": 0, \"sample_count\": 4, "
         "\"text_ids\": [4, 5, 7, 10, 13, 14, 15, 11, 12, 1]}\n";
  }
  bool too_long_threw = false;
  try {
    uavrgb::load_uavr(uavr);
  } catch (const std::runtime_error&) {
    too_long_threw = true;
  }
  CHECK(too_long_threw);

  // 侧车必须绑定到数据集身份：episode_id 不符、区间不符、重复/缺集都必须拒绝。
  auto rejects = [&](const std::string& body) {
    std::ofstream f(dir / "instructions.jsonl", std::ios::trunc);
    f << body;
    try {
      (void)uavrgb::load_uavr(uavr);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  const auto ep2_sidecar = "{\"episode_id\": 2, \"sample_start\": 0, \"sample_count\": 4, "
                            "\"text_ids\": [2]}\n";
  const auto bad_range = "{\"episode_id\": 1, \"sample_start\": 1, \"sample_count\": 3, "
                         "\"text_ids\": [2]}\n";
  const auto duplicate = "{\"episode_id\": 1, \"sample_start\": 0, \"sample_count\": 4, "
                         "\"text_ids\": [2]}\n"
                         "{\"episode_id\": 1, \"sample_start\": 0, \"sample_count\": 4, "
                         "\"text_ids\": [3]}\n";
  CHECK(rejects(ep2_sidecar));   // episode_id 不匹配
  CHECK(rejects(bad_range));     // sample_start/count 与 episodes[] 不符
  CHECK(rejects(duplicate));     // 重复覆盖同一 episode
  CHECK(rejects(""));            // 空侧车（缺集）
  fs::remove_all(dir);
}

void test_checkpoint_roundtrip() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() / "uavrgb_text_ckpt_test";
  fs::remove_all(dir);
  fs::create_directories(dir);
  const std::string path = (dir / "ckpt.bin").string();

  CnnModel m = uavrgb::make_cnn(kH, kW, 7, kClasses, kD);
  uavrgb::TrainState st = uavrgb::make_train_state(m, 0.01f);
  st.best_macro_f1 = 0.5f;
  uavrgb::save_checkpoint(path, m, st);
  CHECK(uavrgb::read_checkpoint_meta(path).text_dim == kD);

  CnnModel back = uavrgb::make_cnn(kH, kW, 7, kClasses, kD);
  uavrgb::TrainState st2 = uavrgb::make_train_state(back, 0.01f);
  uavrgb::load_checkpoint(path, back, st2);
  for (std::size_t i = 0; i < m.text_emb.numel(); ++i) CHECK(back.text_emb[i] == m.text_emb[i]);
  CHECK(st2.adam.size() == uavrgb::cnn_params(back).size());

  // 结构不符（无文本分支）必须拒绝，不能静默丢 embedding。
  CnnModel plain = uavrgb::make_cnn(kH, kW, 7, kClasses, 0);
  uavrgb::TrainState st3 = uavrgb::make_train_state(plain, 0.01f);
  bool threw = false;
  try {
    uavrgb::load_checkpoint(path, plain, st3);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);

  // 参数个数相同、只有 text_dim 不同：必须由 text_dim 检查拒绝，
  // 而不是靠参数个数不同偶然拦住。
  CnnModel wide = uavrgb::make_cnn(kH, kW, 7, kClasses, kD + 2);
  uavrgb::TrainState st4 = uavrgb::make_train_state(wide, 0.01f);
  threw = false;
  try {
    uavrgb::load_checkpoint(path, wide, st4);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
  fs::remove_all(dir);
}

}  // namespace

int main() {
  test_text_affects_logits();
  test_embedding_gradcheck();
  test_film_gradcheck();
  test_sidecar();
  test_checkpoint_roundtrip();
  if (failures != 0) {
    std::fprintf(stderr, "test_text_branch: %d 项失败\n", failures);
    return 1;
  }
  std::printf("test_text_branch: PASS\n");
  return 0;
}
