#pragma once
// --class-weights 的解析、落盘与训练时 loss/grad 选择。
// auto 公式与理由写在 class_weights.cpp；这里只暴露调用方需要的接口。
#include "uavrgb/tensor.hpp"

#include <string>
#include <vector>

namespace uavrgb {

struct UavrDataset;
struct TrainConfig;

// 空 spec -> {}（无权重）；"auto" -> 训练集逆频率；"w0,w1,w2,w3" -> 显式。
// 非法 spec/长度 != num_classes/负权重抛 std::invalid_argument。
std::vector<float> resolve_class_weights(const std::string& spec, const UavrDataset& ds);

// spec -> "none" / "auto" / "explicit"（写进 config.json 与 metrics 的来源标记）。
const char* class_weights_source(const std::string& spec);

// resume 时把 checkpoint 的权重接过来：CLI 显式给了且不一致就报错，没给就用 checkpoint 的。
// weights 被就地更新，返回最终来源（"auto"/"explicit"/"checkpoint"/"none"）。
const char* merge_resume_weights(const std::string& spec, std::vector<float>& weights,
                                 const std::vector<float>& ckpt_weights);

// weights 为空时退回无权版 softmax-CE，否则走加权版（CPU/CUDA 分母一致，见 ops.hpp）。
float class_weighted_loss(const Tensor& logits, const std::vector<int>& labels,
                          const std::vector<float>& weights);
Tensor class_weighted_grad(const Tensor& logits, const std::vector<int>& labels,
                           const std::vector<float>& weights);

// 覆盖写 <out_dir>/config.json：记录本次训练参数与最终生效的类别权重。
void write_run_config(const std::string& out_dir, const TrainConfig& cfg,
                      const std::vector<float>& weights, const std::string& source);

}  // namespace uavrgb
