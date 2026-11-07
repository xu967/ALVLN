// 设备常驻路径（DeviceCnn）一步 vs host 算子逐层一步：同 batch、同初始权重，
// 参数/Adam m,v 差 < 1e-5。覆盖纯视觉与文本 FiLM 两种结构、n < max_batch、
// 裁剪生效/不生效、连续两步（t 递推与偏差校正）。
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/model_dev.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/ops.hpp"
#include "uavrgb/softmax_cuda.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}

double max_abs_diff(const uavrgb::Tensor& a, const uavrgb::Tensor& b) {
  if (a.dims() != b.dims()) return INFINITY;
  double worst = 0.0;
  for (std::size_t i = 0; i < a.numel(); ++i) {
    worst = std::fmax(worst, std::fabs(static_cast<double>(a[i]) - b[i]));
  }
  return worst;
}

void run_case(int text_dim, float max_norm, const char* name) {
  using namespace uavrgb;
  const int h = 16, w = 24, n = 5, max_batch = 8, num_classes = 4;
  CnnModel mh = make_cnn(h, w, /*seed=*/7, num_classes, text_dim);
  CnnModel md = make_cnn(h, w, /*seed=*/7, num_classes, text_dim);
  TrainState sh = make_train_state(mh, 1e-3f);
  TrainState sd = make_train_state(md, 1e-3f);

  std::vector<std::uint8_t> u8(static_cast<std::size_t>(n) * 3 * h * w);
  for (std::size_t i = 0; i < u8.size(); ++i) u8[i] = static_cast<std::uint8_t>((i * 131 + 17) % 256);
  Tensor x({n, 3, h, w});
  for (std::size_t i = 0; i < u8.size(); ++i) x[i] = u8[i] / 255.0f;  // 与设备侧同一表达式
  // token 里混入 <pad>=0 和越界 id（>= kTextVocab），两条路径都必须把它们当 pad。
  std::vector<std::uint8_t> text(static_cast<std::size_t>(n) * kTextLen);
  for (std::size_t i = 0; i < text.size(); ++i) text[i] = static_cast<std::uint8_t>((i * 5 + 3) % (kTextVocab + 2));
  const std::vector<int> y = {0, 1, 2, 3, 1};
  const std::vector<std::uint8_t>* text_h = text_dim > 0 ? &text : nullptr;
  const std::uint8_t* text_d = text_dim > 0 ? text.data() : nullptr;

  DeviceCnn dev(md, max_batch, &sd);
  for (int step = 0; step < 2; ++step) {
    CnnCache cache;
    const Tensor logits_h = forward(mh, x, &cache, text_h);
    const Tensor logits_d = dev.forward(u8.data(), n, text_d);
    const double dl = max_abs_diff(logits_h, logits_d);
    std::printf("[%s] step %d max|logits_h - logits_d| = %.3g\n", name, step, dl);
    check(dl < 1e-5, "logits 两条路径一致");

    CnnGrad g = backward(mh, cache, softmax_ce_grad_cuda(logits_h, y));
    std::vector<Tensor*> gts = cnn_grad_tensors(g);
    const float coef = clip_grad_norm_cuda(gts, max_norm);
    std::vector<Tensor*> ps = cnn_params(mh);
    for (std::size_t i = 0; i < ps.size(); ++i) adam_step_cuda(sh.adam[i], *ps[i], *gts[i]);

    const float norm = dev.backward_step(softmax_ce_grad_cuda(logits_d, y), max_norm, sd);
    std::printf("[%s] step %d host clip coef=%.4g, device grad norm=%.4g (max_norm=%g)\n", name,
                step, coef, norm, max_norm);
    check((coef < 1.0f) == (norm > max_norm), "裁剪是否生效两条路径一致");
  }

  dev.download(md, &sd);
  const std::vector<Tensor*> ph = cnn_params(mh);
  const std::vector<Tensor*> pd = cnn_params(md);
  check(ph.size() == pd.size() && sh.adam.size() == ph.size(), "参数个数一致");
  double worst_p = 0.0, worst_mv = 0.0;
  for (std::size_t k = 0; k < ph.size() && k < pd.size(); ++k) {
    const double dp = max_abs_diff(*ph[k], *pd[k]);
    const double dm = std::fmax(max_abs_diff(sh.adam[k].m, sd.adam[k].m),
                                max_abs_diff(sh.adam[k].v, sd.adam[k].v));
    std::printf("[%s] param %zu (%zu 元素) max|Δp|=%.3g max|Δm,v|=%.3g\n", name, k,
                ph[k]->numel(), dp, dm);
    worst_p = std::fmax(worst_p, dp);
    worst_mv = std::fmax(worst_mv, dm);
    check(sh.adam[k].t == 2 && sd.adam[k].t == 2, "两步后 Adam t == 2");
  }
  std::printf("[%s] worst |Δp| = %.3g, worst |Δm,v| = %.3g\n", name, worst_p, worst_mv);
  check(worst_p < 1e-5, "两步后参数差 < 1e-5");
  check(worst_mv < 1e-5, "两步后 Adam m/v 差 < 1e-5");

  // 更新后的权重再前向：设备侧用的必须是更新后的参数（download 不是唯一真相源）。
  const Tensor after_h = forward(mh, x, nullptr, text_h);
  const Tensor after_d = dev.forward(u8.data(), n, text_d);
  check(max_abs_diff(after_h, after_d) < 1e-5, "更新后前向一致");
}

// STUDENT-P2 结构：6 通道双帧 + prev-action FiLM + 深度分带辅助头。host 侧 L = CE_action +
// λ·CE_depth 的梯度 = backward(dlogits, λ·dL_depth/daux)；设备侧 = aux_loss(λ) + backward_step。
// 第二步不给深度目标（aux_loss 不调），验证辅助头梯度被清 0、Adam 仍按步推进。
void run_p2_case() {
  using namespace uavrgb;
  const int h = 16, w = 24, n = 5, max_batch = 8, num_classes = 4, in_c = 6, aux_k = 8, prev_dim = 4;
  const float lambda = 0.5f, max_norm = 100.0f;
  CnnModel mh = make_cnn(h, w, 7, num_classes, 0, in_c, aux_k, prev_dim);
  CnnModel md = make_cnn(h, w, 7, num_classes, 0, in_c, aux_k, prev_dim);
  check(has_film(mh) && mh.text_dim == 0 && cond_dim(mh) == prev_dim, "只有 prev-action 时 FiLM 条件维 = prev_dim");
  TrainState sh = make_train_state(mh, 1e-3f);
  TrainState sd = make_train_state(md, 1e-3f);

  std::vector<std::uint8_t> u8(static_cast<std::size_t>(n) * in_c * h * w);
  for (std::size_t i = 0; i < u8.size(); ++i) u8[i] = static_cast<std::uint8_t>((i * 131 + 17) % 256);
  Tensor x({n, in_c, h, w});
  for (std::size_t i = 0; i < u8.size(); ++i) x[i] = u8[i] / 255.0f;
  const std::vector<std::uint8_t> prev = {0, 3, 1, 2, 3};
  const std::vector<int> y = {0, 1, 2, 3, 1};
  const int ah = h / 8, aw = w / 8;
  std::vector<std::uint8_t> tgt(static_cast<std::size_t>(n) * ah * aw);
  for (std::size_t i = 0; i < tgt.size(); ++i) tgt[i] = (i % 5 == 0) ? kSpatialIgnore : static_cast<std::uint8_t>((i * 3) % aux_k);

  DeviceCnn dev(md, max_batch, &sd);
  check(dev.aux_h() == ah && dev.aux_w() == aw, "设备侧辅助头网格 = H/8 x W/8");
  for (int step = 0; step < 2; ++step) {
    const bool with_depth = step == 0;
    CnnCache cache;
    const Tensor logits_h = forward(mh, x, &cache, nullptr, &prev);
    const Tensor logits_d = dev.forward(u8.data(), n, nullptr, nullptr, prev.data());
    check(max_abs_diff(logits_h, logits_d) < 1e-5, "[p2] logits 两条路径一致");

    Tensor daux;
    if (with_depth) {
      const Tensor aux_h = aux_forward(mh, cache);
      check(aux_h.dims() == std::vector<std::int64_t>({n, aux_k, ah, aw}), "[p2] host 辅助头输出形状");
      const float loss_h = spatial_ce_loss(aux_h, tgt);
      const DeviceCnn::AuxStats st = dev.aux_loss(tgt.data(), lambda);
      std::printf("[p2] depth ce host=%.6f dev=%.6f valid=%zu correct=%zu\n", loss_h, st.loss, st.valid, st.correct);
      check(std::fabs(loss_h - st.loss) < 1e-5, "[p2] 辅助损失两条路径一致");
      check(st.valid == tgt.size() - (tgt.size() + 4) / 5, "[p2] 有效格数 = 非屏蔽格数");
      daux = spatial_ce_grad(aux_h, tgt);
      for (std::size_t i = 0; i < daux.numel(); ++i) daux[i] *= lambda;
    }
    CnnGrad g = backward(mh, cache, softmax_ce_grad_cuda(logits_h, y), with_depth ? &daux : nullptr);
    std::vector<Tensor*> gts = cnn_grad_tensors(g);
    clip_grad_norm_cuda(gts, max_norm);
    std::vector<Tensor*> ps = cnn_params(mh);
    for (std::size_t i = 0; i < ps.size(); ++i) adam_step_cuda(sh.adam[i], *ps[i], *gts[i]);
    dev.backward_step(softmax_ce_grad_cuda(logits_d, y), max_norm, sd);
  }
  dev.download(md, &sd);
  const std::vector<Tensor*> ph = cnn_params(mh);
  const std::vector<Tensor*> pd = cnn_params(md);
  check(ph.size() == 12 && pd.size() == 12, "[p2] 参数数 = 8 + film 2 + aux 2");
  double worst_p = 0.0, worst_mv = 0.0;
  for (std::size_t k = 0; k < ph.size(); ++k) {
    worst_p = std::fmax(worst_p, max_abs_diff(*ph[k], *pd[k]));
    worst_mv = std::fmax(worst_mv, std::fmax(max_abs_diff(sh.adam[k].m, sd.adam[k].m),
                                             max_abs_diff(sh.adam[k].v, sd.adam[k].v)));
  }
  std::printf("[p2] worst |Δp| = %.3g, worst |Δm,v| = %.3g\n", worst_p, worst_mv);
  check(worst_p < 1e-5 && worst_mv < 1e-5, "[p2] 两步后参数与 Adam 状态一致");
  // 第二步没给目标 → 辅助头梯度为 0，但 Adam 动量让参数仍会动；这里只验证 host/device 一致即可。
  bool threw = false;
  try {
    dev.aux_loss(tgt.data(), lambda);  // 没有 forward 在前
  } catch (const std::logic_error&) {
    threw = true;
  }
  check(threw, "[p2] backward_step 之后直接 aux_loss 必须抛 logic_error");
}

void run_misuse() {
  using namespace uavrgb;
  CnnModel m = make_cnn(16, 24, 1, 4, 0);
  DeviceCnn dev(m, 2);
  std::vector<std::uint8_t> u8(static_cast<std::size_t>(3) * 3 * 16 * 24, 7);
  bool threw = false;
  try {
    dev.forward(u8.data(), 3, nullptr);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  check(threw, "batch 超过 max_batch 必须抛 invalid_argument");
  threw = false;
  TrainState st = make_train_state(m, 1e-3f);
  try {
    dev.backward_step(Tensor({2, 4}), 1.0f, st);
  } catch (const std::logic_error&) {
    threw = true;
  }
  check(threw, "没有 forward 就 backward_step 必须抛 logic_error");
}

}  // namespace

int main() {
  try {
    run_case(/*text_dim=*/0, /*max_norm=*/100.0f, "vision");  // 裁剪不生效
    run_case(/*text_dim=*/16, /*max_norm=*/0.05f, "film");    // 裁剪生效
    run_p2_case();
    run_misuse();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: 异常 %s\n", e.what());
    return 1;
  }
  if (g_fail) {
    std::fprintf(stderr, "test_device_train_step: FAIL (%d)\n", g_fail);
    return 1;
  }
  std::printf("test_device_train_step: PASS\n");
  return 0;
}
