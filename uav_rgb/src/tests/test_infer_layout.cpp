// 部署推理路径的输入布局（交叉审核 F2，2026-11-12）。
// run_infer_loop 把行协议里的 HWC uint8 摆成模型要的 NCHW，这是训练分布到部署输入的唯一接口。
// test_infer_io 的输入是 8x8 方形，只比「同输入同输出」与「换输入输出会变」：
// 源下标写成转置、两帧顺序写反、R/B 互换、条件字节取错位置，它都不会红。
//
// 本测试用非方形输入（H=16, W=24）。参考值由测试自己按「[当前帧 R,G,B ; 上一帧 R,G,B]」逐元素摆成
// NCHW（像素 = 字节/255），走 host 参考前向 forward（与设备路径的数值差 < 1e-5，见 test_device_train_step），
// 与 run_infer_loop 输出的 probs 逐值比对。每组还断言「对调后的参考与原参考不同」，保证夹具对该错误敏感。
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "uavrgb/action.hpp"
#include "uavrgb/infer.hpp"
#include "uavrgb/model.hpp"

namespace {

using Bytes = std::vector<std::uint8_t>;
using Probs = std::vector<double>;
using namespace uavrgb;

int g_failures = 0;
void check(bool ok, const std::string& msg) {
  if (!ok) {
    std::cerr << "FAIL: " << msg << "\n";
    ++g_failures;
  }
}

constexpr int kH = 16, kW = 24;  // 非方形；H、W 都是 8 的倍数（三次 2x2 池化）
constexpr double kTol = 1e-4;    // probs 以 6 位有效数字打印；设备与 host 前向的差约 1e-6
constexpr double kSensitive = 1e-3;

// 一帧 HWC RGB：像素值随 (y, x, 通道) 变化，salt 不同得到不同的帧。
Bytes make_frame(int salt) {
  Bytes f(static_cast<std::size_t>(kH) * kW * 3);
  for (int y = 0; y < kH; ++y)
    for (int x = 0; x < kW; ++x)
      for (int ch = 0; ch < 3; ++ch)
        f[(static_cast<std::size_t>(y) * kW + x) * 3 + ch] =
            static_cast<std::uint8_t>((y * 37 + x * 11 + ch * 71 + salt * 53 + 5) % 256);
  return f;
}

// 参考 NCHW 输入：通道 = 帧序 * 3 + (R,G,B)，帧序 0 = 当前帧。
Tensor reference_input(const std::vector<Bytes>& frames) {
  const int c = 3 * static_cast<int>(frames.size());
  std::vector<float> chw(static_cast<std::size_t>(c) * kH * kW);
  for (std::size_t f = 0; f < frames.size(); ++f)
    for (int ch = 0; ch < 3; ++ch)
      for (int y = 0; y < kH; ++y)
        for (int x = 0; x < kW; ++x)
          chw[((f * 3 + ch) * kH + y) * kW + x] = frames[f][(static_cast<std::size_t>(y) * kW + x) * 3 + ch] / 255.0f;
  return Tensor::from_flat({1, c, kH, kW}, chw);
}

// prev < 0：模型不接 prev-action。
Probs reference_probs(const CnnModel& m, const std::vector<Bytes>& frames, int prev) {
  const std::vector<std::uint8_t> p = {static_cast<std::uint8_t>(prev < 0 ? 0 : prev)};
  const Tensor logits = forward(m, reference_input(frames), nullptr, nullptr, prev < 0 ? nullptr : &p);
  double mx = logits[0], sum = 0.0;
  for (int k = 1; k < kNumActions; ++k) mx = std::max(mx, static_cast<double>(logits[k]));
  Probs out(kNumActions);
  for (int k = 0; k < kNumActions; ++k) sum += (out[k] = std::exp(static_cast<double>(logits[k]) - mx));
  for (double& v : out) v /= sum;
  return out;
}

// 行协议一步 = 各帧 HWC 字节依次排开，模型接 prev-action 时末尾再加 1 字节。
std::string step_bytes(const std::vector<Bytes>& frames, int prev) {
  std::string s;
  for (const Bytes& f : frames) s.append(f.begin(), f.end());
  if (prev >= 0) s.push_back(static_cast<char>(prev));
  return s;
}

// 解析输出行里的 "probs":[a,b,c,d]。
std::vector<Probs> run_infer(const CnnModel& m, const std::string& payload) {
  std::istringstream in(payload);
  std::ostringstream out;
  (void)run_infer_loop(m, in, out);
  std::vector<Probs> rows;
  std::istringstream lines(out.str());
  std::string line;
  while (std::getline(lines, line)) {
    const std::size_t at = line.find("\"probs\":[");
    check(at != std::string::npos, "输出缺 probs");
    if (at == std::string::npos) continue;
    std::istringstream nums(line.substr(at + 9, line.find(']', at) - at - 9));
    Probs p;
    std::string tok;
    while (std::getline(nums, tok, ',')) p.push_back(std::stod(tok));
    rows.push_back(p);
  }
  return rows;
}

double max_diff(const Probs& a, const Probs& b) {
  double d = 0.0;
  for (std::size_t k = 0; k < a.size() && k < b.size(); ++k) d = std::max(d, std::fabs(a[k] - b[k]));
  return a.size() == b.size() ? d : 1e9;
}

// infer 的一步输出必须等于参考；返回两者的最大差供调用方打印。
void expect_matches(const CnnModel& m, const std::vector<Bytes>& frames, int prev, const std::string& what) {
  const auto rows = run_infer(m, step_bytes(frames, prev));
  check(rows.size() == 1, what + "：应输出 1 行");
  if (rows.size() != 1) return;
  const double d = max_diff(rows[0], reference_probs(m, frames, prev));
  std::cout << what << "：与参考的最大差 " << d << "\n";
  check(d < kTol, what + "：probs 必须等于按 [当前帧;上一帧] 摆成 NCHW 的参考前向");
}

}  // namespace

int main() {
  const Bytes cur = make_frame(1), prev_frame = make_frame(2), third = make_frame(3);

  // 1) 单帧，非方形：HWC → CHW 的下标与 R,G,B 的通道顺序；同一输入流里的第二步不受第一步残留影响。
  {
    const CnnModel m = make_cnn(kH, kW, /*seed=*/7, kNumActions);
    expect_matches(m, {cur}, -1, "单帧");
    const auto rows = run_infer(m, step_bytes({cur}, -1) + step_bytes({third}, -1));
    check(rows.size() == 2, "单帧：两步应输出 2 行");
    if (rows.size() == 2) {
      check(max_diff(rows[0], reference_probs(m, {cur}, -1)) < kTol, "单帧：同流第 1 步等于参考");
      check(max_diff(rows[1], reference_probs(m, {third}, -1)) < kTol,
            "单帧：同流第 2 步等于参考（没有读到上一步的缓冲）");
    }
  }

  // 2) 纯双帧（不接 prev-action）：第一张是当前帧（通道 0..2），第二张是上一帧（通道 3..5）。
  {
    const CnnModel m = make_cnn(kH, kW, 7, kNumActions, /*text_dim=*/0, /*in_c=*/6);
    check(max_diff(reference_probs(m, {cur, prev_frame}, -1), reference_probs(m, {prev_frame, cur}, -1)) > kSensitive,
          "夹具自检：两帧对调后的参考输出必须不同");
    expect_matches(m, {cur, prev_frame}, -1, "双帧");
  }

  // 3) 双帧 + prev-action（行协议 = 当前帧 + 上一帧 + 1 字节）：四个 prev id 各自等于带同一个 id 的参考，且彼此不同。
  {
    const CnnModel m = make_cnn(kH, kW, 7, kNumActions, 0, 6, /*aux_classes=*/8, /*prev_dim=*/4);
    check(max_diff(reference_probs(m, {cur, prev_frame}, 2), reference_probs(m, {prev_frame, cur}, 2)) > kSensitive,
          "夹具自检：双帧 + prev 的参考对帧序敏感");
    std::vector<Probs> by_id;
    for (int id = 0; id < 4; ++id) {
      expect_matches(m, {cur, prev_frame}, id, "双帧+prev=" + std::to_string(id));
      by_id.push_back(reference_probs(m, {cur, prev_frame}, id));
    }
    for (int a = 0; a < 4; ++a)
      for (int b = a + 1; b < 4; ++b)
        check(max_diff(by_id[a], by_id[b]) > kSensitive,
              "夹具自检：prev id " + std::to_string(a) + " 与 " + std::to_string(b) + " 的参考输出必须不同");
    // 帧序反了的一步（先上一帧、后当前帧）不能得到「当前帧在前」的参考输出。
    const auto swapped = run_infer(m, step_bytes({prev_frame, cur}, 2));
    check(swapped.size() == 1 && max_diff(swapped[0], reference_probs(m, {cur, prev_frame}, 2)) > kSensitive,
          "对调两帧后 infer 的输出必须偏离「当前帧在前」的参考");
  }

  if (g_failures == 0) {
    std::cout << "test_infer_layout: PASS\n";
    return 0;
  }
  std::cerr << "test_infer_layout: " << g_failures << " 项失败\n";
  return 1;
}
