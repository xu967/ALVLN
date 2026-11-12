#pragma once
// 训练状态 + checkpoint 二进制格式（自定义 v1：小端、float32、无填充、无校验和）。
//
// 文件布局（偏移 B）：
//   0    char[8]  magic "UAVRCK01"
//   8    u32      version = 6（v2 起带类别权重；v3 追加 best_macro_f1；
//                v4 追加 text_dim；v5 追加训练契约；v6 追加 aux_classes/prev_dim。
//                读端接受 v2..v6；v1 无类别权重段，直接拒绝）
//   12   u32      param_count（顺序同 cnn_params：8 个 CNN/fc 参数 + 可选 text_emb
//                + 可选 film.W, film.b + 可选 aux.W, aux.b）
//   16   i32 in_h, i32 in_w, i32 in_c(3 单帧 / 6 双帧), i32 num_classes   ← 模型结构元信息
//   32   u32      adam_t（已完成 step 数；8 个 Adam 共享同一 t）
//   36   u32      epoch（已完成 epoch 数，续训时 results.csv 接着编号）
//   40   u32      weight_count（0 或 num_classes），随后 f32 class_weights[weight_count]
//        (v3) f32 best_macro_f1（dev 宏 F1 峰值；-1 = 尚未评估）
//        (v4) i32 text_dim（0 = 无文本分支）——只追加在 best 之后，旧偏移不变
//        (v5) 训练契约（固定 113 B）：32B dataset SHA256、u8 drop_collision(bit0)、
//             32B soft targets SHA256（全 0 = 未蒸馏）、f32 distill_alpha、f32 lr、
//             u32 seed、i32 batch、32B split_scheme 字符串的 SHA256（全 0 = 未记录）。
//             旧 v2..v4 没有该段：续训按旧行为放行，由 trainer 打警告。
//        (v6) i32 aux_classes、i32 prev_dim（旧版本读回 0/0）——追加在契约之后
//        逐参数（顺序同 cnn_params）：u32 ndim、i64 dims[ndim]、f32 data[numel]
//        再逐参数：f32 m[numel]
//        再逐参数：f32 v[numel]
//
// 读回时校验 magic/version/param_count/结构元信息/逐参数 dims，任何不符直接抛异常，
// 不做部分加载。lr/beta/eps 不入盘：恢复训练以 CLI 为准，避免「哪边生效」的歧义。
// 类别权重必须入盘：resume 时沿用训练时的权重，否则 loss 的归一化基准会悄悄改变。
// 文本分支只有 FiLM 一种实现：pre-FiLM checkpoint 不支持（旧 concat 分支 fc.in=19216），
// 加载会因参数个数/text_dim/逐参数形状不符被 load_checkpoint 直接拒绝，不做兼容加载。
#include "uavrgb/model.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/sha256.hpp"

#include <string>
#include <vector>

namespace uavrgb {

// v5 续训契约：只记录能证明「同一次训练事实」的量，不参与前向。
// 空/全 0 表示该字段未记录（旧 checkpoint 或非 trainer 写出）；比较时跳过空字段。
struct CheckpointContract {
  Sha256 dataset_sha256{};
  bool drop_collision = false;
  Sha256 soft_targets_sha256{};
  float distill_alpha = 0.0f;
  float lr = 0.0f;
  unsigned seed = 0;
  int batch = 0;
  Sha256 split_scheme_sha256{};
};

struct TrainState {
  std::vector<Adam> adam;          // 与 cnn_params(model) 一一对应；adam[0].t 即全局 step
  int epoch = 0;                   // 已完成 epoch 数（results.csv 的 epoch 列从这里接着编）
  std::vector<float> class_weights;  // 空 = 无权训练；否则长度 = num_classes
  // dev 宏 F1 的当前最高值，-1 表示还没有可用 dev 评估。入盘是为了续训时
  // 不把已有的 ckpt_best.bin 用更差的模型覆盖掉。
  float best_macro_f1 = -1.0f;
  CheckpointContract contract;     // v5 训练契约；contract_present=false 时为旧格式
  bool contract_present = false;
};

inline int train_step(const TrainState& st) { return st.adam.empty() ? 0 : st.adam[0].t; }

// 只读文件头，给推理端在构造模型前拿结构尺寸（load_checkpoint 要求模型已存在且尺寸一致）。
struct CheckpointMeta {
  int in_h = 0;
  int in_w = 0;
  int in_c = 0;
  int num_classes = 0;
  int text_dim = 0;  // 0 = 无文本分支（推理端据此决定要不要读 --text）
  int aux_classes = 0;  // v6：深度分带辅助头类别数（0 = 无；推理端不算它）
  int prev_dim = 0;     // v6：prev-action one-hot 维度（0 = 无；>0 时每帧多读 1 字节）
  bool has_drop_collision = false;  // v5+ 非全零训练契约在场时为 true
  bool drop_collision = false;      // v5+ 训练契约记录的碰撞过滤口径
};
CheckpointMeta read_checkpoint_meta(const std::string& path);

// adam 为空时按 params 形状分配（m/v 全零），lr 由调用方给。
TrainState make_train_state(const CnnModel& model, float lr);

void save_checkpoint(const std::string& path, const CnnModel& model, const TrainState& st);

// model 必须已按目标结构构造、st.adam 已按 params 形状初始化，否则抛 std::invalid_argument。
void load_checkpoint(const std::string& path, CnnModel& model, TrainState& st);

}  // namespace uavrgb
