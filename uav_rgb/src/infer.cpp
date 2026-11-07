#include "uavrgb/infer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "uavrgb/model_dev.hpp"
#include "uavrgb/tensor.hpp"

namespace uavrgb {

std::size_t run_infer_loop(const CnnModel& model, std::istream& in, std::ostream& out,
                           std::vector<std::uint8_t> text_ids) {
  // 文本分支：--text 的 id 补齐到 kTextLen（缺的补 <pad>=0），逐帧复用同一段指令。
  // 空串 = 无指令（全部 pad），L2 的「文本置空」消融就靠它。
  std::vector<std::uint8_t> text;
  if (model.text_dim > 0) {
    text.assign(kTextLen, 0);
    if (text_ids.size() > static_cast<std::size_t>(kTextLen)) {
      throw std::invalid_argument("infer: --text token 数超过 kTextLen");
    }
    for (std::uint8_t id : text_ids) {
      if (id >= kTextVocab) {
        throw std::invalid_argument("infer: token id 越界: " + std::to_string(id));
      }
    }
    if (text_ids.empty()) {
      // 文本分支对空输入不是中性的：FiLM 会把空文本当成另一条指令。
      std::fprintf(stderr, "uavrgb_infer: 文本模型没有 --text，按全 <pad> 推理\n");
    }
    std::copy(text_ids.begin(), text_ids.end(), text.begin());
  }
  // 没有文本分支时忽略 text_ids：L0 起 collect.py 一律带指令，image-only 权重必须照跑。
  const int h = model.in_h;
  const int w = model.in_w;
  const int c = model.c1.in;
  const int ncls = model.fc.out;
  if (h <= 0 || w <= 0 || c <= 0 || c % kRgbChannels != 0 || ncls <= 0) {
    throw std::invalid_argument("infer: 模型尺寸非法");
  }
  // 一步的输入 = images 张 HWC RGB（双帧模型：当前帧、上一帧，各 h*w*3 字节，顺序固定）
  // + prev-action 1 字节（模型接 prev-action 时）。CHW 里通道 0..2 是当前帧、3..5 是上一帧。
  const int images = c / kRgbChannels;
  const std::size_t one = static_cast<std::size_t>(h) * w * kRgbChannels;
  const std::size_t payload = one * images + (model.prev_dim > 0 ? 1 : 0);
  std::vector<std::uint8_t> rgb(payload);
  std::vector<std::uint8_t> chw(one * images);
  std::vector<double> probs(static_cast<std::size_t>(ncls));
  std::size_t frames = 0;
  // 权重常驻显存，每帧只上传 h*w*c 字节；/255 在设备侧做，与训练 DeviceCnn 同一个 kernel、
  // 同一舍入（host load_images 也写成 x/255.0f；闭环与打包两条输入路对拍见 tools/check_input_parity.py）。
  // 辅助头不在这里算：DeviceCnn 只有 aux_loss 才跑它，部署路径的 MAC 与单头模型相同。
  DeviceCnn dev(model, /*max_batch=*/1);

  while (true) {
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(payload));
    if (in.gcount() == 0) break;  // 干净 EOF
    if (in.gcount() != static_cast<std::streamsize>(payload)) {
      throw std::runtime_error("infer: 帧数据不足（截断）");
    }

    for (int f = 0; f < images; ++f) {  // 每张 HWC → CHW，按帧序堆叠
      const std::uint8_t* src_img = rgb.data() + one * f;
      std::uint8_t* dst_img = chw.data() + one * f;
      for (int ch = 0; ch < kRgbChannels; ++ch) {
        for (int y = 0; y < h; ++y) {
          for (int xx = 0; xx < w; ++xx) {
            const std::size_t src = (static_cast<std::size_t>(y) * w + xx) * kRgbChannels + ch;
            dst_img[(static_cast<std::size_t>(ch) * h + y) * w + xx] = src_img[src];
          }
        }
      }
    }
    const std::uint8_t prev = model.prev_dim > 0 ? rgb[payload - 1] : 0;
    if (model.prev_dim > 0 && prev >= model.prev_dim) {
      throw std::runtime_error("infer: prev-action 字节越界: " + std::to_string(prev));
    }

    const auto t0 = std::chrono::steady_clock::now();
    const Tensor logits = dev.forward(chw.data(), 1, text.empty() ? nullptr : text.data(), nullptr,
                                      model.prev_dim > 0 ? &prev : nullptr);
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
