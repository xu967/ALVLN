#pragma once
// NativeTrainer：UAVR 数据集 → CNN 前向/反向 → 梯度裁剪 → Adam → checkpoint + 指标落盘。
// 每 epoch 追加一行 results.csv（列固定见 event.hpp 的 kResultsCsvHeader），
// 每 step 追加一行 metrics.jsonl（字段见 event.hpp 的 MetricsEvent，监控台在读）。
// 纯单线程；前向/反向/Adam 在 DeviceCnn（设备常驻）上跑，host 侧只做 IO、loss 与标量统计。
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/dataset.hpp"
#include "uavrgb/event.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/model_dev.hpp"

#include <cstdint>
#include <memory>
#include <random>
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
  // 算法 A/B 开关，默认全关 = 原行为（实验记录见 docs/worklog 2026-11-06 src-algo）。
  // 设备侧增强，逗号分隔的组合：photo（gamma/对比度/亮度抖动）、shift（±4px 平移）、
  // flip（水平翻转，仅纯视觉模型，左右偏航标签互换）、all；空 = 关。
  std::string augment;
  int warmup_steps = 0;          // 线性 warmup 步数
  bool cosine_lr = false;        // warmup 之后余弦退火到 0
  float weight_decay = 0.0f;     // AdamW 解耦权重衰减（0 = 原 Adam）
  float label_smoothing = 0.0f;  // 标签平滑 ε（与类别权重叠加；与蒸馏互斥，蒸馏优先）
  // 同上：resume 时未显式给出的开关从源 run 继承，否则续训会静默换算法（AdamW 退回 Adam 等）。
  bool augment_set = false, warmup_steps_set = false, cosine_lr_set = false,
       weight_decay_set = false, label_smoothing_set = false;
  // STUDENT-P2（都需要 dataset.aux 侧车；旧数据集没有侧车时这些开关直接报错，不静默降级）：
  bool two_frame = false;    // 输入 6 通道 [I_t ; I_{t-1}]（上一帧按侧车 prev_index 读）
  bool prev_action = false;  // 上一步执行的动作 one-hot 经 FiLM 进第三层
  bool depth_aux = false;    // 1x1 深度分带辅助头，总损失 L = L_action + aux_lambda·L_depth
  float aux_lambda = 0.5f;
  bool two_frame_set = false, prev_action_set = false, depth_aux_set = false,
       aux_lambda_set = false;
  // 每轮多写一份 <out>/ckpt_ep{NN}.bin（NN = results.csv 的 epoch 列，格式同 ckpt.bin）。
  // 选模研究要对每一轮做离线探针/闭环（DISTILL §2.4：ckpt_best 是噪声序列上的最大值），
  // 默认关：30 轮 × 1.2 MB 对流水线 run 是多余的。
  bool save_every_epoch = false;
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
  bool has_two_frame = false, has_prev_action = false, has_depth_aux = false,
       has_aux_lambda = false;
  bool two_frame = false, prev_action = false, depth_aux = false;
  float aux_lambda = 0.0f;
  bool has_augment = false, has_warmup_steps = false, has_cosine_lr = false,
       has_weight_decay = false, has_label_smoothing = false;
  std::string augment;
  int warmup_steps = 0;
  bool cosine_lr = false;
  float weight_decay = 0.0f, label_smoothing = 0.0f;
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
  // 深度分带辅助头（模型有辅助头且数据集带深度目标时才有）：有效格平均 CE 与 argmax 命中率；
  // 没有时都是 -1。
  double depth_ce = -1.0;
  double depth_acc = -1.0;
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
  // 以下在 trainer_dev.cpp。
  // 读 part 这批样本（uint8 + token + 双帧的上一帧 + prev-action）并在 dev_ 上前向，返回 host
  // logits。train_labels 非空表示训练批：cfg_.augment 非空时做增强，翻转样本的左右偏航标签
  // （和 prev-action）就地互换、深度目标按列镜像。本批深度目标留在 depth_batch_ 给 aux_loss。
  Tensor forward_batch(const UavrDataset& ds, const std::vector<std::uint32_t>& part,
                       std::vector<int>* train_labels = nullptr);
  // 模型有辅助头且数据集带深度目标时，对最近一次 forward_batch 的批算辅助损失（lambda 为
  // 反向时的权重；评估传 0）；否则返回 valid=0。
  DeviceCnn::AuxStats aux_batch(const UavrDataset& ds, float lambda);
  // 按 cfg_/数据集结构建模型（in_c / prev_dim / aux_classes 都在这里决定）。
  CnnModel build_model(const UavrDataset& ds) const;
  // 按 warmup/余弦日程算第 step 步（1 起）的 lr，写进 state_.adam[*].lr 并返回。
  float step_lr(std::uint32_t step, std::uint32_t total_steps);
  // 训练 loss 与 dlogits：蒸馏 > 标签平滑(+类别权重) > 类别权重/无权。
  Tensor loss_and_grad(const Tensor& logits, const std::vector<int>& y,
                       const std::vector<float>& q, float* loss);

  TrainConfig cfg_;
  CnnModel model_;
  TrainState state_;
  std::unique_ptr<DeviceCnn> dev_;    // 权重/Adam/激活常驻显存；存 checkpoint 前 download
  std::vector<std::uint8_t> u8_batch_;  // forward_batch 复用的像素缓冲
  std::vector<std::uint8_t> depth_batch_;  // 最近一批的深度分带目标（已按翻转镜像）
  std::mt19937 aug_rng_{0x9e3779b9u};   // 增强专用，与打乱顺序的 rng 分开，A/B 时样本顺序不变
  std::vector<float> class_weights_;          // 最终生效权重（同时写进 state_/checkpoint）
  std::string class_weights_source_ = "none";  // "none"/"auto"/"explicit"/"checkpoint"
  std::vector<float> soft_targets_;  // [n*C]；空 = 不蒸馏
  double elapsed_sec_ = 0.0;
  double test_acc_ = -1.0;
  std::uint64_t samples_seen_ = 0;
};

}  // namespace uavrgb
