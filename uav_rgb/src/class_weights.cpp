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
#include <filesystem>
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
    // tok 为空时必须单独判：std::stof("") 抛异常后 used=0，而 0 == tok.size()，
    // 原条件放行，空串会静默变成一个 0 权重，那个类从此拿不到任何梯度。
    if (tok.empty() || used != tok.size() || !(v >= 0.0f) || !std::isfinite(v)) {
      throw std::invalid_argument("--class-weights 非法权重 '" + tok +
                                  "'（应为非负有限数字，且不能有空项）");
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
  for (unsigned char ch : s) {
    if (ch == '"' || ch == '\\') {
      o += '\\';
      o += static_cast<char>(ch);
    } else if (ch == '\n') {
      o += "\\n";
    } else if (ch == '\r') {
      o += "\\r";
    } else if (ch == '\t') {
      o += "\\t";
    } else if (ch < 0x20) {
      // JSON 不允许裸控制字符；与 metrics_json.cpp 的 quoted 保持一致。
      char b[7];
      std::snprintf(b, sizeof b, "\\u%04x", static_cast<unsigned>(ch));
      o += b;
    } else {
      o += static_cast<char>(ch);
    }
  }
  return o + '"';
}

// config.json 是扁平对象：取 "key": 后的标量 token（字符串去掉引号）。
// 找不到返回空串；调用方用空串表示「旧 run 没记录该字段」。
std::string raw_field(const std::string& js, const std::string& key) {
  const std::string pat = "\"" + key + "\":";
  const std::size_t p = js.find(pat);
  if (p == std::string::npos) return {};
  const std::size_t b = p + pat.size();
  if (b < js.size() && js[b] == '"') {
    const std::size_t e = js.find('"', b + 1);
    return e == std::string::npos ? std::string{} : js.substr(b + 1, e - b - 1);
  }
  std::size_t e = b;
  while (e < js.size() && js[e] != ',' && js[e] != '}' && js[e] != '\n') ++e;
  return js.substr(b, e - b);
}

std::string quote_or_null(const std::string& s) { return s.empty() ? "null" : quote(s); }

}  // namespace

bool read_run_config(const std::string& run_dir, RunConfigSnapshot* out) {
  // 同目录续训会把最新事实写在 config.resume.json（原 config.json 不动）；
  // 二次续训应读最新那份，否则覆盖过的 CLI 会退回原始值。
  std::ifstream f(run_dir + "/config.resume.json");
  if (!f) f.open(run_dir + "/config.json");
  if (!f) return false;
  std::ostringstream os;
  os << f.rdbuf();
  const std::string js = os.str();
  auto str = [&](const char* k) { return raw_field(js, k); };
  out->has_lr = !str("lr").empty();
  if (out->has_lr) out->lr = std::stof(str("lr"));
  out->has_batch = !str("batch").empty();
  if (out->has_batch) out->batch = std::stoi(str("batch"));
  out->has_seed = !str("seed").empty();
  if (out->has_seed) out->seed = static_cast<unsigned>(std::stoul(str("seed")));
  out->has_drop_collision = !str("drop_collision").empty();
  if (out->has_drop_collision) out->drop_collision = str("drop_collision") == "true";
  out->soft_targets = str("soft_targets");
  out->has_soft_targets = !out->soft_targets.empty();
  out->has_distill_alpha = !str("distill_alpha").empty();
  if (out->has_distill_alpha) out->distill_alpha = std::stof(str("distill_alpha"));
  out->has_text_dim = !str("text_dim").empty();
  if (out->has_text_dim) out->text_dim = std::stoi(str("text_dim"));
  out->has_vision_only = !str("vision_only").empty();
  out->vision_only = str("vision_only") == "true";
  out->dataset_sha256 = str("dataset_sha256");
  return true;
}

std::vector<float> resolve_class_weights(const std::string& spec, const UavrDataset& ds) {
  if (spec.empty()) return {};
  if (spec == "auto") return auto_weights(ds);
  std::vector<float> w = parse_explicit(spec);
  if (w.size() != ds.num_classes) {
    throw std::invalid_argument("--class-weights 需要 " + std::to_string(ds.num_classes) +
                                " 个权重（实际 " + std::to_string(w.size()) + "）");
  }
  // 全 0 的加权求和恒为 0，loss 归一化时除零；与其训练到第一个 batch 才炸，不如入口就拒绝。
  bool any = false;
  for (float v : w) any = any || v > 0.0f;
  if (!any) throw std::invalid_argument("--class-weights 不能全为 0（每个样本权重都是 0）");
  // 训练集存在的类若权重为 0，一个只含该类的 batch 会让 Σw=0、归一化除零并中止训练。
  std::vector<std::size_t> count(w.size(), 0);
  for (std::uint32_t i : ds.train_idx) ++count[ds.labels[i]];
  for (std::size_t k = 0; k < w.size(); ++k) {
    if (w[k] == 0.0f && count[k] > 0) {
      throw std::invalid_argument("--class-weights 给训练集存在的类 " + std::to_string(k) +
                                  " 设了 0 权重（纯该类 batch 会除零）；用 auto 或正权重");
    }
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
     << ",\"seed\":" << cfg.seed << ",\"text_dim\":" << cfg.text_dim
     << ",\"vision_only\":" << (cfg.vision_only ? "true" : "false")
     << ",\"max_grad_norm\":" << cfg.max_grad_norm
     << ",\"drop_collision\":" << (cfg.drop_collision ? "true" : "false")
     << ",\"class_weights\":[";
  for (std::size_t i = 0; i < weights.size(); ++i) os << (i ? "," : "") << weights[i];
  os << "],\"class_weights_source\":" << quote(source)
     << ",\"soft_targets\":" << quote(cfg.soft_targets)
     << ",\"distill_alpha\":" << cfg.distill_alpha
     // 以下为事实字段：旧字段不动，只在末尾追加，旧 parser 忽略即可。
     << ",\"dataset_sha256\":" << quote(cfg.dataset_sha256)
     << ",\"split_scheme\":" << quote_or_null(cfg.split_scheme)
     << ",\"window_from\":" << quote_or_null(cfg.window_from)
     << ",\"hold_s\":" << (cfg.hold_s.empty() ? "null" : cfg.hold_s)
     << ",\"soft_targets_sha256\":" << quote(cfg.soft_targets_sha256)
     << ",\"teacher_ckpt_sha256\":" << quote(cfg.teacher_ckpt_sha256)
     << ",\"teacher_select_split\":" << quote(cfg.teacher_select_split)
     << ",\"beta1\":" << cfg.adam_beta1 << ",\"beta2\":" << cfg.adam_beta2
     << ",\"eps\":" << cfg.adam_eps << ",\"source_git\":" << quote(cfg.source_git)
     << ",\"augment\":" << quote(cfg.augment)
     << ",\"warmup_steps\":" << cfg.warmup_steps
     << ",\"cosine_lr\":" << (cfg.cosine_lr ? "true" : "false")
     << ",\"weight_decay\":" << cfg.weight_decay
     << ",\"label_smoothing\":" << cfg.label_smoothing;
  if (!cfg.resume.empty()) os << ",\"resume\":" << quote(cfg.resume);
  os << "}\n";
  auto write_to = [&](const std::string& path) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) throw std::runtime_error("trainer: 无法写 " + path);
    f << os.str();
    if (!f) throw std::runtime_error("trainer: 写入失败 " + path);
  };
  const std::string cfg_path = out_dir + "/config.json";
  const std::string resume_path = out_dir + "/config.resume.json";
  std::error_code ec;
  if (cfg.resume.empty()) {
    write_to(cfg_path);
    return;
  }
  // 续训不截断原 config.json：同目录续跑只写 config.resume.json；跨目录新 run
  // 没有旧 config，才把同一份事实同时写成主配置。
  if (!std::filesystem::exists(cfg_path, ec)) write_to(cfg_path);
  write_to(resume_path);
}

}  // namespace uavrgb
