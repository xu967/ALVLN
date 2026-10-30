#include "uavrgb/infer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <vector>

#include "uavrgb/tensor.hpp"

namespace uavrgb {

std::size_t run_infer_loop(const CnnModel& model, std::istream& in, std::ostream& out,
                           std::vector<std::uint8_t> text_ids) {
  (void)text_ids;  // L0：先打通协议；L2 才把 token 送进文本编码器。
  const int h = model.in_h;
  const int w = model.in_w;
  const int c = model.c1.in;
  const int ncls = model.fc.out;
  if (h <= 0 || w <= 0 || c <= 0 || ncls <= 0) {
    throw std::invalid_argument("infer: 模型尺寸非法");
  }
  const std::size_t frame = static_cast<std::size_t>(h) * w * c;
  std::vector<unsigned char> rgb(frame);
  std::vector<double> probs(static_cast<std::size_t>(ncls));
  std::size_t frames = 0;

  while (true) {
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(frame));
    if (in.gcount() == 0) break;  // 干净 EOF
    if (in.gcount() != static_cast<std::streamsize>(frame)) {
      throw std::runtime_error("infer: 帧数据不足（截断）");
    }

    Tensor x({1, c, h, w});
    for (int ch = 0; ch < c; ++ch) {
      for (int y = 0; y < h; ++y) {
        for (int xx = 0; xx < w; ++xx) {
          const std::size_t src = (static_cast<std::size_t>(y) * w + xx) * c + ch;
          x[(static_cast<std::size_t>(ch) * h + y) * w + xx] = rgb[src] / 255.0f;
        }
      }
    }

    const auto t0 = std::chrono::steady_clock::now();
    const Tensor logits = forward(model, x);
    const auto t1 = std::chrono::steady_clock::now();
    if (logits.numel() != static_cast<std::size_t>(ncls)) {
      throw std::runtime_error("infer: logits 维度与 num_classes 不符");
    }

    double max_logit = logits[0];
    for (int k = 1; k < ncls; ++k) max_logit = std::max(max_logit, static_cast<double>(logits[k]));
    double sum = 0.0;
    int best = 0;
    for (int k = 0; k < ncls; ++k) {
      probs[k] = std::exp(static_cast<double>(logits[k]) - max_logit);
      sum += probs[k];
      if (logits[k] > logits[best]) best = k;
    }
    for (int k = 0; k < ncls; ++k) probs[k] /= sum;

    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out << "{\"action\":" << best << ",\"probs\":[";
    for (int k = 0; k < ncls; ++k) {
      if (k) out << ',';
      out << probs[k];
    }
    out << "],\"infer_ms\":" << ms << "}\n";
    out.flush();
    if (!out) throw std::runtime_error("infer: stdout 写入失败");
    ++frames;
  }
  return frames;
}

}  // namespace uavrgb
