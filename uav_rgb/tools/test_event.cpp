// EventSchema 自检（独立编译，不接入 src/CMakeLists.txt）：
//   g++ -std=c++17 -O2 -Wall -Wextra -I src/include tools/test_event.cpp -o /tmp/test_event
//   /tmp/test_event
// 校验枚举尺寸、结构体赋值/默认值、工厂函数与 to_string 映射；失败返回 1。
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "uavrgb/event.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++g_failures;
  }
}

bool str_eq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

bool near(double a, double b) { return (a > b ? a - b : b - a) < 1e-9; }

}  // namespace

int main() {
  using namespace uavrgb;

  static_assert(sizeof(EventType) == 1, "EventType 应为 1 字节（JSON 枚举）");
  static_assert(sizeof(Split) == 1, "Split 应为 1 字节");
  static_assert(sizeof(Severity) == 1, "Severity 应为 1 字节");
  static_assert(std::is_trivially_copyable<ResultsRow>::value,
                "ResultsRow 应可平凡拷贝");
  static_assert(std::is_same<decltype(ResultsRow::epoch), std::uint32_t>::value,
                "ResultsRow::epoch 必须是 uint32_t");

  std::printf("sizeof: MetricsEvent=%zu ResultsRow=%zu EventType=%zu Split=%zu Severity=%zu\n",
              sizeof(MetricsEvent), sizeof(ResultsRow), sizeof(EventType),
              sizeof(Split), sizeof(Severity));

  // 枚举 -> JSON 字符串
  check(str_eq(to_string(EventType::kTrainStep), "train_step"), "to_string(train_step)");
  check(str_eq(to_string(EventType::kValidation), "validation"), "to_string(validation)");
  check(str_eq(to_string(EventType::kStatus), "status"), "to_string(status)");
  check(str_eq(to_string(EventType::kError), "error"), "to_string(error)");
  check(str_eq(to_string(Split::kNone), ""), "to_string(split none)");
  check(str_eq(to_string(Split::kTrain), "train"), "to_string(split train)");
  check(str_eq(to_string(Split::kVal), "val"), "to_string(split val)");
  check(str_eq(to_string(Split::kTest), "test"), "to_string(split test)");
  check(str_eq(to_string(Severity::kFatal), "FATAL"), "to_string(FATAL)");

  // 默认值
  MetricsEvent def;
  check(def.type == EventType::kStatus, "默认 type=status");
  check(def.step == 0 && def.total_steps == 0, "默认 step/total_steps=0");
  check(def.loss == 0.0f && def.action_accuracy == 0.0f && def.lr == 0.0f,
        "默认 loss/acc/lr=0");
  check(def.split == Split::kNone && def.severity == Severity::kInfo,
        "默认 split=single severity=INFO");
  check(def.message.empty() && def.code.empty(), "默认 message/code 为空");

  // train_step
  MetricsEvent ts = make_train_step(100, 800, 1.105f, 0.40f, 0.001f, 12.5, 87.5);
  check(ts.type == EventType::kTrainStep, "train_step type");
  check(ts.step == 100 && ts.total_steps == 800, "train_step step/total");
  check(ts.loss == 1.105f && ts.action_accuracy == 0.40f && ts.lr == 0.001f,
        "train_step loss/acc/lr");
  check(near(ts.elapsed_sec, 12.5) && near(ts.eta_sec, 87.5), "train_step elapsed/eta");
  check(ts.split == Split::kNone, "train_step 不带 split");

  // validation
  MetricsEvent val = make_validation(200, 1.05f, 0.45f, Split::kVal);
  check(val.type == EventType::kValidation, "validation type");
  check(val.step == 200 && val.loss == 1.05f && val.action_accuracy == 0.45f,
        "validation step/loss/acc");
  check(val.split == Split::kVal, "validation split=val");

  // status
  MetricsEvent st = make_status("checkpoint saved");
  check(st.type == EventType::kStatus && st.message == "checkpoint saved", "status message");

  // error
  MetricsEvent er = make_error(Severity::kFatal, "TRAIN_NONFINITE_METRIC", 300, "loss is NaN");
  check(er.type == EventType::kError, "error type");
  check(er.severity == Severity::kFatal, "error severity");
  check(er.code == "TRAIN_NONFINITE_METRIC" && er.step == 300, "error code/step");
  check(er.message == "loss is NaN", "error message");

  // results.csv 行
  ResultsRow row;
  row.epoch = 3;
  row.train_ce = 0.874;
  row.dev_ce = 0.887;
  row.dev_accuracy = 0.510;
  check(row.epoch == 3 && near(row.train_ce, 0.874) && near(row.dev_ce, 0.887) &&
            near(row.dev_accuracy, 0.510), "ResultsRow 赋值");
  check(str_eq(kResultsCsvHeader, "epoch,train_ce,dev_ce,dev_accuracy"),
        "results.csv 表头");
  check(str_eq(kMetricsSchema, "tinyvln.metrics.v1"), "metrics schema");

  if (g_failures == 0) {
    std::printf("RESULT: PASS (all checks)\n");
    return 0;
  }
  std::printf("RESULT: FAIL (%d checks failed)\n", g_failures);
  return 1;
}
