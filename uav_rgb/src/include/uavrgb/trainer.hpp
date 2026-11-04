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
  // >0 时启用文本分支：embedding[kTextVocab, text_dim] → 按位置拼接 → FiLM。
  // 需要 dataset 旁边有 instructions.jsonl 侧车（preprocess.py 写）。
  int text_dim = 0;
  // 数据集带 instructions.jsonl 却显式要纯视觉基线时必须置位：默认 0 的 text_dim
  // 会让"忘传 --text-dim"静默训成纯视觉模型，症状和训练失败一样（2026-11-03 审计）。
  bool vision_only = false;
  // 非空时启用蒸馏：扁平 float32 [n*C] 软标签（tools/join_soft_targets.py 产出），
  // 第 k 行对应数据集第 k 个样本。与 --class-weights 互斥（两种 loss 归一化口径不同）。
  std::string soft_targets;
  float distill_alpha = 0.5f;  // 软标签项权重；0 = 退化为纯硬标签
  // CLI 是否显式给出：resume 时只有未显式给出的项才从源 run 的 config.json 继承。
  bool lr_set = false;
  bool batch_set = false;
  bool seed_set = false;
  bool drop_collision_set = false;
  bool soft_targets_set = false;
  bool distill_alpha_set = false;
  bool text_dim_set = false;
  // checkpoint 训练契约不一致时默认拒绝；显式放行才继续（会打警告）。
  bool allow_resume_mismatch = false;
  // 教师选模 split 与 student val 重叠时默认只警告；strict 直接失败。
  bool strict_distill = false;
  // 以下由 trainer 在 run() 里按实际数据/环境补齐，再落进 config.json。
  std::string dataset_sha256, split_scheme, window_from, hold_s;
  std::string soft_targets_sha256, teacher_ckpt_sha256, teacher_select_split, source_git;
  float adam_beta1 = 0.9f, adam_beta2 = 0.999f, adam_eps = 1e-8f;
};

// 从 <run_dir>/config.json 读回的续训事实；没有文件时返回 false（旧 run 兼容）。
struct RunConfigSnapshot {
  bool has_lr = false, has_batch = false, has_seed = false, has_drop_collision = false;
  bool has_soft_targets = false, has_distill_alpha = false, has_text_dim = false;
  bool has_vision_only = false;
  float lr = 0.0f;
  int batch = 0;
  unsigned seed = 0;
  bool drop_collision = false;
  std::string soft_targets;
  float distill_alpha = 0.0f;
  int text_dim = 0;
  bool vision_only = false;
  std::string dataset_sha256;
};
bool read_run_config(const std::string& run_dir, RunConfigSnapshot* out);

// 文件 SHA256 的小写十六进制；文件不可读时抛 std::runtime_error。
std::string file_sha256_hex(const std::string& path);

// 追加 <out_dir>/test_eval.jsonl 一行：记录每次 test 评估（eval_count/时间/数据身份），
// 不改写历史 metrics.jsonl。samples=0 表示调用方未提供样本数。
void append_test_eval_record(const std::string& out_dir, const std::string& origin,
                             const std::string& ckpt_path, const std::string& data_path,
                             const std::string& dataset_sha256, bool drop_collision,
                             std::size_t samples);

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
  std::vector<float> soft_targets_;  // [n*C]；空 = 不蒸馏
  double elapsed_sec_ = 0.0;
  double test_acc_ = -1.0;
  std::uint64_t samples_seen_ = 0;
};

}  // namespace uavrgb
