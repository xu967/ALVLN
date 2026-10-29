#pragma once
// 训练状态 + checkpoint 二进制格式（自定义 v1：小端、float32、无填充、无校验和）。
//
// 文件布局（偏移 B）：
//   0    char[8]  magic "UAVRCK01"
//   8    u32      version = 3（v2 起带类别权重；v3 追加 best_macro_f1。
//                读端接受 v2/v3；v1 无类别权重段，直接拒绝）
//   12   u32      param_count（固定 8）
//   16   i32 in_h, i32 in_w, i32 in_c, i32 num_classes   ← 模型结构元信息
//   32   u32      adam_t（已完成 step 数；8 个 Adam 共享同一 t）
//   36   u32      epoch（已完成 epoch 数，续训时 results.csv 接着编号）
//   40   u32      weight_count（0 或 num_classes），随后 f32 class_weights[weight_count]
//        (v3) f32 best_macro_f1（dev 宏 F1 峰值；-1 = 尚未评估）
//        逐参数（顺序同 cnn_params）：u32 ndim、i64 dims[ndim]、f32 data[numel]
//        再逐参数：f32 m[numel]
//        再逐参数：f32 v[numel]
//
// 读回时校验 magic/version/param_count/结构元信息/逐参数 dims，任何不符直接抛异常，
// 不做部分加载。lr/beta/eps 不入盘：恢复训练以 CLI 为准，避免「哪边生效」的歧义。
// 类别权重必须入盘：resume 时沿用训练时的权重，否则 loss 的归一化基准会悄悄改变。
#include "uavrgb/model.hpp"
#include "uavrgb/nn.hpp"

#include <string>
#include <vector>

namespace uavrgb {

struct TrainState {
  std::vector<Adam> adam;          // 与 cnn_params(model) 一一对应；adam[0].t 即全局 step
  int epoch = 0;                   // 已完成 epoch 数（results.csv 的 epoch 列从这里接着编）
  std::vector<float> class_weights;  // 空 = 无权训练；否则长度 = num_classes
  // dev 宏 F1 的当前最高值，-1 表示还没有可用 dev 评估。入盘是为了续训时
  // 不把已有的 ckpt_best.bin 用更差的模型覆盖掉。
  float best_macro_f1 = -1.0f;
};

inline int train_step(const TrainState& st) { return st.adam.empty() ? 0 : st.adam[0].t; }

// 只读文件头，给推理端在构造模型前拿结构尺寸（load_checkpoint 要求模型已存在且尺寸一致）。
struct CheckpointMeta {
  int in_h = 0;
  int in_w = 0;
  int in_c = 0;
  int num_classes = 0;
};
CheckpointMeta read_checkpoint_meta(const std::string& path);

// adam 为空时按 params 形状分配（m/v 全零），lr 由调用方给。
TrainState make_train_state(const CnnModel& model, float lr);

void save_checkpoint(const std::string& path, const CnnModel& model, const TrainState& st);

// model 必须已按目标结构构造、st.adam 已按 params 形状初始化，否则抛 std::invalid_argument。
void load_checkpoint(const std::string& path, CnnModel& model, TrainState& st);

}  // namespace uavrgb
