#pragma once
// Real training hook: progress on stderr and structured metrics in metrics.jsonl.
// Include from a FUTURE trainer; this header does not train a model by itself.
// Every line carries "schema":"tinyvln.metrics.v1"; numbers are always written in the C locale.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tinyvln {
inline constexpr const char* kMetricsSchema = "tinyvln.metrics.v1";

inline std::string JsonEscape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    if (c == '\n') { out += "\\n"; continue; }
    if (static_cast<unsigned char>(c) < 32) continue;
    out += c;
  }
  return out;
}

class TrainingMonitor {
 public:
  // resume_step: steps already completed when this process started (0 for a fresh run),
  // so that ETA after a checkpoint resume is not underestimated.
  explicit TrainingMonitor(const std::string& jsonl_path, std::size_t resume_step = 0)
      : file_(jsonl_path, std::ios::app), begin_(Clock::now()), resume_step_(resume_step) {
    if (!file_) throw std::runtime_error("METRICS_OPEN_FAILED: " + jsonl_path);
    file_.imbue(std::locale::classic());  // never "26,000" or "0,75" inside JSON
    file_ << std::setprecision(std::numeric_limits<double>::max_digits10);  // lossless round trip
  }

  void Step(std::size_t step, std::size_t total, double loss,
            double accuracy, double lr) {
    if (!std::isfinite(loss) || !std::isfinite(accuracy) || !std::isfinite(lr))
      Fail(step, "TRAIN_NONFINITE_METRIC", "loss/accuracy/lr is NaN or Inf");
    if (total == 0 || step > total || accuracy < 0 || accuracy > 1)
      Fail(step, "TRAIN_METRIC_RANGE", "step/total/accuracy out of range");
    const double elapsed = std::chrono::duration<double>(Clock::now() - begin_).count();
    const std::size_t done = step > resume_step_ ? step - resume_step_ : 0;
    const double eta = done ? elapsed * (static_cast<double>(total - step) / done) : 0;
    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"train_step\",\"step\":" << step
          << ",\"total_steps\":" << total << ",\"loss\":" << loss
          << ",\"action_accuracy\":" << accuracy << ",\"lr\":" << lr
          << ",\"elapsed_sec\":" << elapsed << ",\"eta_sec\":" << eta << "}\n";
    file_.flush();
    // Format in a private stream: std::cerr's flags/precision are never modified.
    std::ostringstream bar;
    bar.imbue(std::locale::classic());
    constexpr int kWidth = 28;
    const int filled = static_cast<int>(kWidth * static_cast<double>(step) / total);
    bar << '\r' << "[TRAIN] [" << std::string(filled, '#')
        << std::string(kWidth - filled, '-') << "] " << step << '/' << total
        << " loss=" << std::fixed << std::setprecision(4) << loss
        << " acc=" << std::setprecision(1) << accuracy * 100 << '%'
        << " ETA=" << std::setprecision(0) << eta << "s";
    if (step == total) bar << '\n';
    std::cerr << bar.str() << std::flush;
  }

  // split: e.g. "val_unseen"; n_episodes: episodes behind `sr`;
  // closed_loop=false marks teacher-forced / offline proxies so the UI cannot pass them off as SR.
  void Validation(std::size_t step, double sr, double stop_f1, const std::string& split,
                  std::size_t n_episodes, bool closed_loop) {
    if (!std::isfinite(sr) || sr < 0 || sr > 1 || !std::isfinite(stop_f1)
        || stop_f1 < 0 || stop_f1 > 1 || n_episodes == 0)
      Fail(step, "VAL_METRIC_INVALID", "sr/stop_f1 outside [0,1] or n_episodes == 0");
    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"validation\",\"step\":" << step
          << ",\"sr\":" << sr << ",\"stop_f1\":" << stop_f1
          << ",\"split\":\"" << JsonEscape(split) << "\",\"n_episodes\":" << n_episodes
          << ",\"closed_loop\":" << (closed_loop ? "true" : "false") << "}\n";
    file_.flush();
  }

  void Status(const std::string& event, const std::string& message) {
    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"" << JsonEscape(event)
          << "\",\"message\":\"" << JsonEscape(message) << "\"}\n";
    file_.flush();
  }

 private:
  using Clock = std::chrono::steady_clock;

  // Record why training stopped before throwing, so the dashboard can show the cause.
  [[noreturn]] void Fail(std::size_t step, const char* code, const char* message) {
    file_ << "{\"schema\":\"" << kMetricsSchema << "\",\"event\":\"error\",\"severity\":\"FATAL\""
          << ",\"code\":\"" << code << "\",\"step\":" << step
          << ",\"message\":\"" << JsonEscape(message) << "\"}\n";
    file_.flush();
    throw std::runtime_error(std::string(code) + ": " + message);
  }

  std::ofstream file_;
  Clock::time_point begin_;
  std::size_t resume_step_;
};
}  // namespace tinyvln
