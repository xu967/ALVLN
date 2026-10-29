// 类别权重：auto = 训练集逆频率 w_c = N/(C·n_c)；显式字符串直接解析；结果落盘。
//
// 为什么是 N/(C·n_c)：逆频率让每个类的总权重都等于 N/C（类均衡），归一化分母
// Σ_i w[y_i] = N（每类非空时），平均样本权重为 1，loss 数量级与无权版可比、日志直观。
// 与 (1/n_c) 相比只差一个全局常数；而加权 CE 按 Σw 归一化，全局常数会约掉，所以两者
// 数值等价，选前者只是为了让 metrics 里的权重值好读。训练集缺失的类权重记 0（无样本
// 就不贡献 loss），不会出现除零。
#include "uavrgb/class_weights.hpp"

#include "uavrgb/dataset.hpp"
#include "uavrgb/softmax_cuda.hpp"
#include "uavrgb/trainer.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb {
namespace {

std::vector<float> parse_explicit(const std::string& spec) {
  std::vector<float> w;
  std::size_t b = 0;
  while (true) {
    const std::size_t e = spec.find(',', b);
    const std::string tok = spec.substr(b, e == std::string::npos ? e : e - b);
    std::size_t used = 0;
    float v = 0.0f;
    try {
      v = std::stof(tok, &used);
    } catch (const std::exception&) {
      used = 0;
    }
    if (used != tok.size() || !(v >= 0.0f) || !std::isfinite(v)) {
      throw std::invalid_argument("--class-weights 非法权重 '" + tok +
                                  "'（应为非负有限数字）");
    }
    w.push_back(v);
    if (e == std::string::npos) break;
    b = e + 1;
  }
  return w;
}

std::vector<float> auto_weights(const UavrDataset& ds) {
  const int c = static_cast<int>(ds.num_classes);
  std::vector<std::size_t> count(static_cast<std::size_t>(c), 0);
  for (std::uint32_t i : ds.train_idx) ++count[ds.labels[i]];
  const double n = static_cast<double>(ds.train_idx.size());
  std::vector<float> w(static_cast<std::size_t>(c), 0.0f);
  for (int k = 0; k < c; ++k) {
    if (count[static_cast<std::size_t>(k)] > 0) {
      w[static_cast<std::size_t>(k)] =
          static_cast<float>(n / (static_cast<double>(c) * static_cast<double>(count[
                                     static_cast<std::size_t>(k)])));
    } else {
      // 权重 0 的类不产生任何梯度。训练集缺样本时必须留痕，否则模型永远学不会它。
      std::fprintf(stderr, "警告: 训练集没有类别 %d 的样本，该类权重记为 0\n", k);
    }
    if (count[static_cast<std::size_t>(k)] == 1) {
      // n_c=1 时逆频率退化：归一化后这一个样本独占 1/C 的总 loss，等于让一张图
      // 主导 25% 的梯度。不是报错，但结论必须写明这个类只有 1 个样本。
      std::fprintf(stderr,
                   "警告: 类别 %d 在训练集只有 1 个样本，逆频率权重退化为 %.1f"
                   "（该样本独占约 %.0f%% 的总 loss）\n",
                   k, static_cast<double>(w[static_cast<std::size_t>(k)]),
                   100.0 / static_cast<double>(c));
    }
  }
  return w;
}

std::string quote(const std::string& s) {
  std::string o = "\"";
  for (char ch : s) {
    if (ch == '"' || ch == '\\') o += '\\';
    if (ch == '\n') {
      o += "\\n";
      continue;
    }
    o += ch;
  }
  return o + '"';
}

}  // namespace

std::vector<float> resolve_class_weights(const std::string& spec, const UavrDataset& ds) {
  if (spec.empty()) return {};
  if (spec == "auto") return auto_weights(ds);
  std::vector<float> w = parse_explicit(spec);
  if (w.size() != ds.num_classes) {
    throw std::invalid_argument("--class-weights 需要 " + std::to_string(ds.num_classes) +
                                " 个权重（实际 " + std::to_string(w.size()) + "）");
  }
  return w;
}

const char* class_weights_source(const std::string& spec) {
  if (spec.empty()) return "none";
  return spec == "auto" ? "auto" : "explicit";
}

const char* merge_resume_weights(const std::string& spec, std::vector<float>& weights,
                                 const std::vector<float>& ckpt_weights) {
  if (!spec.empty() && weights != ckpt_weights) {
    throw std::invalid_argument("trainer: resume 的 --class-weights 与 checkpoint 记录不一致"
                                "（续训必须沿用同一组权重；不传该参数则自动用 checkpoint 的）");
  }
  if (spec.empty()) weights = ckpt_weights;
  if (weights.empty()) return "none";
  return spec.empty() ? "checkpoint" : class_weights_source(spec);
}

float class_weighted_loss(const Tensor& logits, const std::vector<int>& labels,
                          const std::vector<float>& weights) {
  return weights.empty() ? softmax_ce_loss_cuda(logits, labels)
                         : softmax_ce_loss_weighted_cuda(logits, labels, weights);
}

Tensor class_weighted_grad(const Tensor& logits, const std::vector<int>& labels,
                           const std::vector<float>& weights) {
  return weights.empty() ? softmax_ce_grad_cuda(logits, labels)
                         : softmax_ce_grad_weighted_cuda(logits, labels, weights);
}

void write_run_config(const std::string& out_dir, const TrainConfig& cfg,
                      const std::vector<float>& weights, const std::string& source) {
  std::ostringstream os;
  os.precision(9);
  os << "{\"schema\":\"uavrgb.train_config.v1\",\"data\":" << quote(cfg.data_path)
     << ",\"epochs\":" << cfg.epochs << ",\"batch\":" << cfg.batch << ",\"lr\":" << cfg.lr
     << ",\"seed\":" << cfg.seed << ",\"max_grad_norm\":" << cfg.max_grad_norm
     << ",\"drop_collision\":" << (cfg.drop_collision ? "true" : "false")
     << ",\"class_weights\":[";
  for (std::size_t i = 0; i < weights.size(); ++i) os << (i ? "," : "") << weights[i];
  os << "],\"class_weights_source\":" << quote(source) << "}\n";
  const std::string path = out_dir + "/config.json";
  std::ofstream f(path, std::ios::trunc);
  if (!f) throw std::runtime_error("trainer: 无法写 " + path);
  f << os.str();
  if (!f) throw std::runtime_error("trainer: 写入失败 " + path);
}

}  // namespace uavrgb
