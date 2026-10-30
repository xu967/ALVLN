// metrics.jsonl 单行 JSON 序列化（字段与 event.hpp 对齐）。从 trainer.cpp 拆出，
// 保持每个 C++ 文件在行数上限内；序列化规则见 event.hpp 文件头注释。
#include "uavrgb/trainer.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>

namespace uavrgb {
namespace {

// 非有限值（nan/inf）不能写进 JSON，统一退化成 null；训练循环另有守卫会直接中止。
std::string num(double v) {
  if (!std::isfinite(v)) return "null";
  char b[40];
  std::snprintf(b, sizeof b, "%.9g", v);
  return b;
}

std::string quoted(const std::string& s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
      o += static_cast<char>(c);
    } else if (c == '\n') {
      o += "\\n";
    } else if (c == '\r') {
      o += "\\r";
    } else if (c == '\t') {
      o += "\\t";
    } else if (c < 0x20) {
      // JSON 不允许裸控制字符；与 class_weights.cpp 的 quote 保持一致。
      char b[7];
      std::snprintf(b, sizeof b, "\\u%04x", static_cast<unsigned>(c));
      o += b;
    } else {
      o += static_cast<char>(c);
    }
  }
  o += '"';
  return o;
}

}  // namespace

std::string metrics_json(const MetricsEvent& e) {
  std::ostringstream os;
  os << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"" << to_string(e.type) << '"';
  switch (e.type) {
    case EventType::kTrainStep:
      os << ",\"step\":" << e.step << ",\"total_steps\":" << e.total_steps
         << ",\"loss\":" << num(e.loss) << ",\"action_accuracy\":" << num(e.action_accuracy)
         << ",\"lr\":" << num(e.lr) << ",\"elapsed_sec\":" << num(e.elapsed_sec)
         << ",\"eta_sec\":" << num(e.eta_sec);
      break;
    case EventType::kValidation:
      os << ",\"step\":" << e.step << ",\"loss\":" << num(e.loss)
         << ",\"action_accuracy\":" << num(e.action_accuracy) << ",\"split\":\""
         << to_string(e.split) << '"';
      // 追加字段，不改动既有键：监控台按已知键取值，多出来的会被忽略。
      // 四件套必须无条件输出（空 confusion 写 []）：任何新评估路径漏填 confusion
      // 时，不能静默丢掉 macro_f1 / balanced_accuracy，否则 EVAL_STANDARD 口径失效。
      os << ",\"macro_f1\":" << num(e.macro_f1)
         << ",\"balanced_accuracy\":" << num(e.balanced_accuracy) << ",\"confusion\":[";
      for (std::size_t i = 0; i < e.confusion.size(); ++i) {
        if (i != 0) os << ',';
        os << e.confusion[i];
      }
      os << ']';
      break;
    case EventType::kStatus:
      os << ",\"message\":" << quoted(e.message);
      break;
    case EventType::kError:
      os << ",\"severity\":\"" << to_string(e.severity) << "\",\"code\":" << quoted(e.code)
         << ",\"step\":" << e.step << ",\"message\":" << quoted(e.message);
      break;
    case EventType::kClassWeights:
      os << ",\"epoch\":" << e.epoch << ",\"step\":" << e.step << ",\"class_weights\":[";
      for (std::size_t i = 0; i < e.class_weights.size(); ++i) {
        if (i != 0) os << ',';
        os << num(e.class_weights[i]);
      }
      os << "],\"class_weights_source\":" << quoted(e.class_weights_source);
      break;
  }
  os << '}';
  return os.str();
}

}  // namespace uavrgb
