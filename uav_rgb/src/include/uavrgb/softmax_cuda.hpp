#pragma once
// softmax 交叉熵 loss/grad 的 CUDA 实现（纯 CUDA + C++，reduction kernel）。
// 接口与 ops.hpp 的 CPU 参考 softmax_ce_loss / softmax_ce_grad 对称，便于对拍（doc 04 §6 N3）。
#include "uavrgb/tensor.hpp"

#include <vector>

namespace uavrgb {

// logits[N, C]、labels[N]（0 <= label < C）；返回平均交叉熵（同 CPU 参考）。
float softmax_ce_loss_cuda(const Tensor& logits, const std::vector<int>& labels);

// 返回 [N, C] 梯度（含 1/N 归一化，同 CPU 参考 softmax_ce_grad）。
Tensor softmax_ce_grad_cuda(const Tensor& logits, const std::vector<int>& labels);

// 加权版：归一化与梯度公式同 ops.hpp 的 softmax_ce_loss_weighted / softmax_ce_grad_weighted。
// weights 长度须等于 C、有限非负、按样本求和 > 0，否则抛 invalid_argument。
float softmax_ce_loss_weighted_cuda(const Tensor& logits, const std::vector<int>& labels,
                                    const std::vector<float>& weights);
Tensor softmax_ce_grad_weighted_cuda(const Tensor& logits, const std::vector<int>& labels,
                                     const std::vector<float>& weights);

}  // namespace uavrgb
