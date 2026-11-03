#pragma once
#include "uavrgb/dataset.hpp"
#include "uavrgb/trainer.hpp"

namespace uavrgb {
// UAVSOFT1 头绑定完整 UAVR、instructions.jsonl 和概率 payload，拒绝旧的裸数组。
std::vector<float> load_soft_targets(const TrainConfig& cfg, const UavrDataset& ds);
}  // namespace uavrgb
