// 任务卡 S1 验证：15 维动态导航条件（prev_action + bearing_bin + near_goal）与第 5 维独立 STOP 头。
// 覆盖：编码/镜像对称性、CPU/CUDA 前向逐值对拍（atol=1e-5）、FP64 中心有限差分梯度检查、
// 常驻推理 3 字节条件协议与 stop_prob/stop_requested、以及 Trainer + eval_only 端到端 v7 权重往返。
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "synthetic_dataset.hpp"
#include "uavrgb/aux_sidecar.hpp"
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/infer.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/model_dev.hpp"
#include "uavrgb/sha256.hpp"
#include "uavrgb/softmax_cuda.hpp"
#include "uavrgb/trainer.hpp"
#include "uavrgb/vln_head.hpp"

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

void write_vln_aux(const std::string& uavr_path, int n,
                   const std::vector<std::uint8_t>& labels,
                   const std::vector<std::uint8_t>& stop_y,
                   const std::vector<std::uint8_t>& nav_cond) {
  const Sha256 uavr_sha = sha256_file(uavr_path);
  std::vector<std::uint8_t> payload;
  for (int i = 0; i < n; ++i) {
    const std::uint32_t pi = (i == 0 || i == 8) ? static_cast<std::uint32_t>(i)
                                                : static_cast<std::uint32_t>(i - 1);
    for (int b = 0; b < 4; ++b) payload.push_back(static_cast<std::uint8_t>((pi >> (8 * b)) & 0xffu));
  }
  for (int i = 0; i < n; ++i) {
    payload.push_back((i == 0 || i == 8) ? 0 : labels[static_cast<std::size_t>(i - 1)]);
  }
  payload.insert(payload.end(), stop_y.begin(), stop_y.end());
  payload.insert(payload.end(), nav_cond.begin(), nav_cond.end());

  std::istringstream pin(std::string(reinterpret_cast<const char*>(payload.data()), payload.size()));
  const Sha256 p_sha = sha256(pin);

  std::vector<std::uint8_t> hb(kAuxHeaderBytes, 0);
  std::memcpy(hb.data(), "UAVRAUX1", 8);
  const std::uint64_t un = static_cast<std::uint64_t>(n);
  std::memcpy(hb.data() + 8, &un, 8);
  const std::uint32_t flags = kAuxFlagStop | kAuxFlagNavCond;  // bit1 | bit2 = 6
  std::memcpy(hb.data() + 16, &flags, 4);
  hb[24] = 8;
  hb[25] = kAuxUnknown;
  std::copy(uavr_sha.begin(), uavr_sha.end(), hb.begin() + 28);
  std::copy(p_sha.begin(), p_sha.end(), hb.begin() + 60);
  const float edges[7] = {0.3f, 0.5f, 0.8f, 1.2f, 1.8f, 2.5f, 4.0f};
  std::memcpy(hb.data() + 92, edges, sizeof edges);
  const float env = 0.102f;
  std::memcpy(hb.data() + 120, &env, 4);

  std::ofstream out(aux_sidecar_path(uavr_path), std::ios::binary | std::ios::trunc);
  out.write(reinterpret_cast<const char*>(hb.data()), static_cast<std::streamsize>(hb.size()));
  out.write(reinterpret_cast<const char*>(payload.data()),
            static_cast<std::streamsize>(payload.size()));
}

}  // namespace

int main() {
  // 1) 72 个导航条件码编解码与翻转镜像自反性。
  for (int ng = 0; ng < 2; ++ng) {
    for (int bb = 0; bb <= 8; ++bb) {
      for (int pa = 0; pa < 4; ++pa) {
        const std::uint8_t c = encode_vln_nav(pa, bb, ng);
        int pa2 = -1, bb2 = -1, ng2 = -1;
        decode_vln_nav(c, &pa2, &bb2, &ng2);
        check(pa == pa2 && bb == bb2 && ng == ng2, "vln_nav 编解码往返");
        const std::uint8_t cf = flip_vln_nav_code(c);
        check(flip_vln_nav_code(cf) == c, "flip_vln_nav_code 二次镜像还原自身");
        int paf = -1, bbf = -1, ngf = -1;
        decode_vln_nav(cf, &paf, &bbf, &ngf);
        check(ngf == ng, "镜像不改变 near_goal");
        if (bb == 8) check(bbf == 8, "UNKNOWN(8) 镜像后仍为 8");
        if (bb == 1) check(bbf == 7, "左前(1) 镜像为右前(7)");
        if (bb == 2) check(bbf == 6, "正左(2) 镜像为正右(6)");
      }
    }
  }

  // 2) Host forward vs DeviceCnn CUDA forward 前向对拍（in_c=6, prev_dim=15, fc.out=5）。
  const int bsz = 4, h = 16, w = 24;
  CnnModel m = make_cnn(h, w, /*seed=*/42, kActionStopClasses, /*text_dim=*/0, /*in_c=*/6,
                        /*aux_classes=*/0, kVlnNavDim);
  std::vector<std::uint8_t> u8(static_cast<std::size_t>(bsz) * 6 * h * w);
  Tensor x({bsz, 6, h, w});
  for (std::size_t i = 0; i < u8.size(); ++i) {
    u8[i] = static_cast<std::uint8_t>((i * 37 + 13) & 0xffu);
    x[i] = static_cast<float>(u8[i]) / 255.0f;
  }
  const std::vector<std::uint8_t> nav = {
      encode_vln_nav(0, 0, 0),
      encode_vln_nav(1, 2, 0),
      encode_vln_nav(2, 8, 0),
      encode_vln_nav(3, 7, 1),
  };
  CnnCache cache;
  const Tensor host_logits = forward(m, x, &cache, nullptr, &nav);
  DeviceCnn dev(m, bsz);
  const Tensor dev_logits = dev.forward(u8.data(), bsz, nullptr, nullptr, nav.data());
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < host_logits.numel(); ++i) {
    max_diff = std::max(max_diff, std::abs(host_logits[i] - dev_logits[i]));
  }
  check(max_diff < 1e-5f, "Host vs DeviceCnn 15维条件+5维头前向对拍 atol=1e-5（max_diff=" +
                              std::to_string(max_diff) + "）");

  // 3) STOP BCE + 动作 CE 联合损失对模型参数的中心有限差分梯度校验。
  const std::vector<int> y_act = {1, 2, 3, 0};
  const std::vector<std::uint8_t> y_stop = {0, 0, 0, 1};
  const float pos_w = 3.0f, stop_lam = 1.2f;
  auto total_loss_fn = [&](const CnnModel& mdl) -> double {
    const Tensor lg = forward(mdl, x, nullptr, nullptr, &nav);
    const Tensor alg = slice_action_logits(lg);
    const double la = softmax_ce_loss_cuda(alg, y_act);
    const double ls = stop_bce_loss_and_grad(lg, y_stop, pos_w, stop_lam, nullptr);
    return la + static_cast<double>(stop_lam) * ls;
  };
  const Tensor act_lg = slice_action_logits(host_logits);
  const Tensor d_act = softmax_ce_grad_cuda(act_lg, y_act);
  std::vector<float> d_stop;
  (void)stop_bce_loss_and_grad(host_logits, y_stop, pos_w, stop_lam, &d_stop);
  const Tensor d_joint = merge_action_stop_grad(d_act, d_stop);
  CnnGrad g = backward(m, cache, d_joint);
  const float eps = 1e-3f;
  for (int idx : {0, 7, 19, static_cast<int>(m.fc.W.numel() - 1)}) {
    CnnModel mp = m, mm = m;
    mp.fc.W[idx] += eps;
    mm.fc.W[idx] -= eps;
    const double num_g = (total_loss_fn(mp) - total_loss_fn(mm)) / (2.0 * static_cast<double>(eps));
    const double ana_g = g.fc.dW[idx];
    check(std::abs(num_g - ana_g) < 2e-4,
          "fc.W 有限差分梯度校验 idx=" + std::to_string(idx) + " diff=" +
              std::to_string(std::abs(num_g - ana_g)));
  }
  for (int idx : {0, 5, 14, static_cast<int>(m.film.W.numel() - 1)}) {
    CnnModel mp = m, mm = m;
    mp.film.W[idx] += eps;
    mm.film.W[idx] -= eps;
    const double num_g = (total_loss_fn(mp) - total_loss_fn(mm)) / (2.0 * static_cast<double>(eps));
    const double ana_g = g.film.dW[idx];
    check(std::abs(num_g - ana_g) < 2e-4,
          "film.W 有限差分梯度校验 idx=" + std::to_string(idx) + " diff=" +
              std::to_string(std::abs(num_g - ana_g)));
  }

  // 4) 常驻推理 3 字节条件协议与 stop_prob / stop_requested。
  {
    const std::size_t frame_bytes = static_cast<std::size_t>(h) * w * 3 * 2;
    std::string req(frame_bytes + 3, '\x10');
    req[frame_bytes + 0] = 1;  // FORWARD_SLOW
    req[frame_bytes + 1] = 7;  // bearing_bin=7
    req[frame_bytes + 2] = 1;  // near_goal=1
    std::istringstream in(req, std::ios::binary);
    std::ostringstream out;
    const std::size_t n_frames = run_infer_loop(m, in, out, {}, /*stop_threshold=*/0.35f);
    check(n_frames == 1, "infer 处理 1 帧 3 字节条件请求");
    const std::string line = out.str();
    check(line.find("\"stop_prob\":") != std::string::npos &&
              line.find("\"stop_requested\":") != std::string::npos,
          "infer 输出包含 stop_prob 与 stop_requested");
  }

  // 5) Trainer + eval_only 端到端 v7 训练与侧车校验。
  {
    const std::string root =
        (fs::temp_directory_path() /
         ("uavrgb_test_vln_s1_" + std::to_string(std::random_device{}())))
            .string();
    fs::create_directories(root);
    const int n = 12;
    const auto sd = uavrgb_test::write_synthetic_uavr(root + "/dataset.uavr", n, h, w, 8, 31);
    std::vector<std::uint8_t> sy(n, 0), nc(n, 0);
    sy[7] = 1;
    sy[11] = 1;
    for (int i = 0; i < n; ++i) {
      nc[static_cast<std::size_t>(i)] =
          encode_vln_nav(sd.labels[static_cast<std::size_t>(i)], i % 9, sy[static_cast<std::size_t>(i)]);
    }
    write_vln_aux(sd.path, n, sd.labels, sy, nc);

    TrainConfig cfg;
    cfg.data_path = sd.path;
    cfg.out_dir = root + "/run";
    cfg.epochs = 3;
    cfg.batch = 4;
    cfg.lr = 1e-3f;
    cfg.two_frame = true;
    cfg.vln_nav = true;
    cfg.stop_head = true;
    cfg.augment = "flip";
    Trainer tr(cfg);
    tr.run();

    const CheckpointMeta meta = read_checkpoint_meta(cfg.out_dir + "/ckpt_best.bin");
    check(meta.num_classes == kActionStopClasses && meta.prev_dim == kVlnNavDim && meta.in_c == 6,
          "ckpt_best.bin 记录 num_classes=5, prev_dim=15, in_c=6");
    check(meta.aux_sidecar_sha256 == sha256_file(aux_sidecar_path(sd.path)),
          "v7 ckpt_best.bin 记录精确的 aux_sidecar_sha256");

    TrainConfig ev = cfg;
    ev.out_dir = root + "/eval";
    Trainer tev(ev);
    tev.eval_only(cfg.out_dir + "/ckpt_best.bin", /*eval_val=*/true, /*eval_test=*/false);
    fs::remove_all(root);
  }

  if (g_fail) {
    std::cerr << "test_vln_cond_stop: FAIL (" << g_fail << ")\n";
    return 1;
  }
  std::cout << "test_vln_cond_stop: PASS\n";
  return 0;
}
