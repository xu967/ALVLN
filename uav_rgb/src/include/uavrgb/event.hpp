#pragma once
// EventSchema：训练/评测事件与 results.csv 行格式（冻结接口）。
// 对齐 ../TinyAerialVLN_Final_V8/docs/07_API_LOGGING_CONTRACT.md 与 ui/monitor.py：
//   metrics.jsonl 每行一个 JSON 对象，schema="tinyvln.metrics.v1"，event 分五类：
//     train_step / validation / status / error / class_weights
//   字段名必须与 UI 解析层一致，新增字段只能追加，不得改名/改单位。
//   results.csv 表头固定为 epoch,train_ce,dev_ce,dev_accuracy（见 kResultsCsvHeader）。
//   train_ce / dev_ce 都是**无权**交叉熵：加权 loss 只进 metrics.jsonl 的 step loss，
//   否则开了 --class-weights 后两列不同口径，画在同一张图上会读成假的过拟合。
// 纯头文件，无第三方依赖；未接入 src/CMakeLists.txt（不产生构建目标）。
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace uavrgb {

inline constexpr char kMetricsSchema[] = "tinyvln.metrics.v1";
inline constexpr char kResultsCsvHeader[] = "epoch,train_ce,dev_ce,dev_accuracy";

// metrics.jsonl 的 event 字段。
enum class EventType : std::uint8_t {
  kTrainStep = 0,    // "train_step"：训练中周期上报的步级指标
  kValidation = 1,   // "validation"：在固定 dev/val 集上评估
  kStatus = 2,       // "status"：生命周期/检查点等文字状态
  kError = 3,        // "error"：可恢复告警或致命错误
  kClassWeights = 4, // "class_weights"：每 epoch 一条，记录生效的类别权重与来源
};

// validation 事件的数据集划分；kNone 表示该事件不带 split 字段。
enum class Split : std::uint8_t {
  kNone = 0,
  kTrain = 1,  // "train"
  kVal = 2,    // "val"（results.csv 中称 dev）
  kTest = 3,   // "test"
};

// error 事件的严重级别（JSON 大写）。
enum class Severity : std::uint8_t {
  kInfo = 0,   // "INFO"
  kWarn = 1,   // "WARN"
  kError = 2,  // "ERROR"：本步失败但进程可继续
  kFatal = 3,  // "FATAL"：必须终止训练
};

constexpr const char* to_string(EventType t) noexcept {
  switch (t) {
    case EventType::kTrainStep: return "train_step";
    case EventType::kValidation: return "validation";
    case EventType::kStatus: return "status";
    case EventType::kError: return "error";
    case EventType::kClassWeights: return "class_weights";
  }
  return "unknown";
}

constexpr const char* to_string(Split s) noexcept {
  switch (s) {
    case Split::kNone: return "";
    case Split::kTrain: return "train";
    case Split::kVal: return "val";
    case Split::kTest: return "test";
  }
  return "";
}

constexpr const char* to_string(Severity s) noexcept {
  switch (s) {
    case Severity::kInfo: return "INFO";
    case Severity::kWarn: return "WARN";
    case Severity::kError: return "ERROR";
    case Severity::kFatal: return "FATAL";
  }
  return "INFO";
}

// 一条 metrics.jsonl 事件。五类事件共用一个结构，按 type 决定哪些字段有效：
//   train_step:    step,total_steps,loss,action_accuracy,lr,elapsed_sec,eta_sec
//   validation:    step,loss,action_accuracy,split
//   status:        message
//   error:         severity,code,step,message
//   class_weights: epoch,step,class_weights,class_weights_source
// 无效字段写默认值；JSON 序列化器按 type 只输出有效字段，保持与 UI 兼容。
struct MetricsEvent {
  EventType type = EventType::kStatus;
  std::uint32_t step = 0;          // 训练步（1 起，参数更新计数）；status 无步时为 0
  std::uint32_t total_steps = 0;   // 计划总训练步数；仅 train_step 有意义
  float loss = 0.0f;               // 交叉熵，单位 nats/样本（自然对数，非 bit）
  float action_accuracy = 0.0f;    // 动作分类准确率，0..1（不是百分数）
  float lr = 0.0f;                 // 当前学习率（无量纲）
  double elapsed_sec = 0.0;        // 训练启动至今的墙钟秒数
  double eta_sec = 0.0;            // 预计剩余墙钟秒数；未知时置 0
  Split split = Split::kNone;      // validation 事件的数据集划分
  Severity severity = Severity::kInfo;  // 仅 error 事件使用
  std::string message;             // status/error 的人读文本；其他类型可为空
  std::string code;                // error 的稳定机器码，如 TRAIN_NONFINITE_METRIC
  std::uint32_t epoch = 0;         // 仅 class_weights：第几个 epoch（1 起）
  std::vector<float> class_weights;       // 仅 class_weights：长度 = num_classes
  std::string class_weights_source;       // 仅 class_weights："auto"/"explicit"/"checkpoint"
  // 仅 validation：类别不平衡下的标准口径（见 eval_metrics.hpp），与 action_accuracy 并存。
  float macro_f1 = 0.0f;                  // 宏平均 F1，0..1
  float balanced_accuracy = 0.0f;         // 每类召回的平均，0..1
  std::vector<std::uint32_t> confusion;   // C×C 行优先，行 = 真值，列 = 预测
};

inline MetricsEvent make_train_step(std::uint32_t step, std::uint32_t total_steps,
                                    float loss, float action_accuracy, float lr,
                                    double elapsed_sec, double eta_sec) {
  MetricsEvent e;
  e.type = EventType::kTrainStep;
  e.step = step;
  e.total_steps = total_steps;
  e.loss = loss;
  e.action_accuracy = action_accuracy;
  e.lr = lr;
  e.elapsed_sec = elapsed_sec;
  e.eta_sec = eta_sec;
  return e;
}

inline MetricsEvent make_validation(std::uint32_t step, float loss,
                                    float action_accuracy, Split split) {
  MetricsEvent e;
  e.type = EventType::kValidation;
  e.step = step;
  e.loss = loss;
  e.action_accuracy = action_accuracy;
  e.split = split;
  return e;
}

inline MetricsEvent make_status(std::string message) {
  MetricsEvent e;
  e.type = EventType::kStatus;
  e.message = std::move(message);
  return e;
}

inline MetricsEvent make_class_weights(std::uint32_t epoch, std::uint32_t step,
                                       std::vector<float> weights, std::string source) {
  MetricsEvent e;
  e.type = EventType::kClassWeights;
  e.epoch = epoch;
  e.step = step;
  e.class_weights = std::move(weights);
  e.class_weights_source = std::move(source);
  return e;
}

inline MetricsEvent make_error(Severity severity, std::string code,
                               std::uint32_t step, std::string message) {
  MetricsEvent e;
  e.type = EventType::kError;
  e.severity = severity;
  e.code = std::move(code);
  e.step = step;
  e.message = std::move(message);
  return e;
}

// results.csv 的一行（epoch 从 1 起；CE 单位 nats/样本；准确率 0..1）。
// 列顺序固定，序列化时不得增删或换序。
struct ResultsRow {
  std::uint32_t epoch = 0;    // 第几个 epoch（1 起）
  double train_ce = 0.0;      // 训练集平均交叉熵（nats/样本）
  double dev_ce = 0.0;        // dev/val 集平均交叉熵（nats/样本，无权 Teacher CE）
  double dev_accuracy = 0.0;  // dev/val 集四动作分类准确率，0..1
};

}  // namespace uavrgb
