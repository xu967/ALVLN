#pragma once
// NativeTrainer：UAVR 数据集 → CNN 前向/反向 → 梯度裁剪 → Adam → checkpoint + 指标落盘。
// 每 epoch 追加一行 results.csv（列固定见 event.hpp 的 kResultsCsvHeader），
// 每 step 追加一行 metrics.jsonl（字段见 event.hpp 的 MetricsEvent，监控台在读）。
// 纯单线程；前向/反向走 CUDA 算子，host 侧只做 IO 与标量统计。
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/dataset.hpp"
#include "uavrgb/event.hpp"
#include "uavrgb/model.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace uavrgb {

struct TrainConfig {
  std::string data_path;  // dataset.uavr
  std::string out_dir;    // results.csv / metrics.jsonl / ckpt 落盘目录
  int epochs = 10;
  int batch = 32;
  float lr = 1e-3f;
  float max_grad_norm = 5.0f;
  unsigned seed = 1234;
  std::string resume;     // 非空时从该 checkpoint 继续（Adam m/v/t + 参数 + 类别权重）
  // 空 = 无权重；"auto" = 训练集逆频率；"w0,w1,w2,w3" = 显式每类权重。
  std::string class_weights;
  bool drop_collision = false;  // 读数据集时丢弃 meta.collision_flags 标记的碰撞步
};

// 一条 MetricsEvent 的 JSON 序列化（单行、无换行）。字段名与 event.hpp 一致。
std::string metrics_json(const MetricsEvent& e);

// 一次评估的结果。ce/acc 是不加权口径；其余是类别不平衡下的标准指标
// （定义见 eval_metrics.hpp）。confusion 与 recall 长度都是 num_classes。
struct EvalStats {
  double ce = 0.0;
  double acc = 0.0;
  double macro_f1 = 0.0;
  double balanced_acc = 0.0;
  std::vector<double> recall;
  std::vector<std::uint32_t> confusion;  // C×C 行优先，行 = 真值，列 = 预测
};

class Trainer {
 public:
  explicit Trainer(TrainConfig cfg) : cfg_(std::move(cfg)) {}

  void run();

  // T4：只评估一个 checkpoint，不建优化器、不更新权重。向 cfg_.out_dir/metrics.jsonl
  // 追加 split=val/test 的 validation 事件（字段与训练期一致，监控台可直接读）。
  // eval_val/eval_test 至少一个为 true；显式请求的 split 为空时写 EVAL_SPLIT_EMPTY
  // FATAL 并抛错（调用方返回非零），不能把空集当成评估成功。
  void eval_only(const std::string& ckpt_path, bool eval_val, bool eval_test);

  const CnnModel& model() const { return model_; }
  CnnModel& model() { return model_; }
  TrainState& state() { return state_; }
  const TrainConfig& config() const { return cfg_; }
  // 最近一次 run() 的墙钟耗时（秒）与处理的样本数，用于吞吐统计。
  double last_elapsed_sec() const { return elapsed_sec_; }
  std::uint64_t last_samples() const { return samples_seen_; }
  // 最后一次 run() 在 test split 上的准确率；数据集没有 test split 时为负。
  double last_test_accuracy() const { return test_acc_; }

 private:
  void write_csv_header_if_needed() const;
  // 在 idx 上评估，结果写进 out。
  void evaluate(const UavrDataset& ds, const std::vector<std::uint32_t>& idx, EvalStats* out);

  TrainConfig cfg_;
  CnnModel model_;
  TrainState state_;
  std::vector<float> class_weights_;          // 最终生效权重（同时写进 state_/checkpoint）
  std::string class_weights_source_ = "none";  // "none"/"auto"/"explicit"/"checkpoint"
  double elapsed_sec_ = 0.0;
  double test_acc_ = -1.0;
  std::uint64_t samples_seen_ = 0;
};

}  // namespace uavrgb
