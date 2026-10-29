#pragma once
// 线性层 + softmax 交叉熵的 CPU FP32 参考实现，作为 CUDA 前向/反向的数值 oracle
// （doc 04 §6 N1/N2：FP64 finite-difference gradcheck 与 CPU/CUDA 对拍）。
#include "uavrgb/tensor.hpp"

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace uavrgb {

// 单层线性：权重布局 W[out][in]，bias[out]；forward 输出 logits[N, out]。
struct Linear {
  int in = 0;
  int out = 0;
  Tensor W;  // [out, in]
  Tensor b;  // [out]

  Linear() = default;
  Linear(int in_features, int out_features);
};

namespace detail {

// Linear 的 W/b 形状不变量：字段是 public，模型组装/checkpoint 赋值可能绕过构造函数，
// 所有前向/反向/gradcheck 入口都必须先校验，否则按 out*in 读 W 会越界读。
inline void check_linear_params(const Linear& m) {
  if (m.in <= 0 || m.out <= 0) {
    throw std::invalid_argument("Linear: in/out 必须为正（实际 in=" + std::to_string(m.in) +
                                ", out=" + std::to_string(m.out) + "）");
  }
  const std::size_t w_expected = static_cast<std::size_t>(m.out) * static_cast<std::size_t>(m.in);
  if (m.W.numel() != w_expected) {
    throw std::invalid_argument("Linear: W.numel() 应为 out*in=" + std::to_string(w_expected) +
                                "（实际 " + std::to_string(m.W.numel()) + "）");
  }
  if (m.b.numel() != static_cast<std::size_t>(m.out)) {
    throw std::invalid_argument("Linear: b.numel() 应为 out=" + std::to_string(m.out) +
                                "（实际 " + std::to_string(m.b.numel()) + "）");
  }
}

}  // namespace detail

// x: [N, in] -> [N, out]
Tensor linear_forward(const Linear& m, const Tensor& x);

// 线性层反向输出：dW[out,in]、db[out]、dx[N,in]（与 forward 的解析梯度一一对应）。
struct LinearGrad {
  Tensor dW;
  Tensor db;
  Tensor dx;
};

// CPU FP32 反向参考：给定上游梯度 dlogits[N,out]，返回 dW/db/dx。
LinearGrad linear_backward(const Linear& m, const Tensor& x, const Tensor& dlogits);

// softmax 交叉熵：logits[N, C]，labels[N]（0 <= label < C）；N=0 或 C=0 抛 invalid_argument。
float softmax_ce_loss(const Tensor& logits, const std::vector<int>& labels);
Tensor softmax_ce_grad(const Tensor& logits, const std::vector<int>& labels);

// 加权 softmax 交叉熵：weights[C] 为每类权重，按样本权重和归一化：
//   L = -Σ_i w[y_i]·log p_i / Σ_i w[y_i]，dlogits_ij = w[y_i]·(p_ij - 1[j==y_i]) / Σ_i w[y_i]。
// 分母不用 N，所以 loss 数量级不随权重整体缩放变化；权重全为 1 时与原版逐位一致。
// 权重须有限且非负、长度等于 C、Σ 按样本求和 > 0，否则抛 invalid_argument。
float softmax_ce_loss_weighted(const Tensor& logits, const std::vector<int>& labels,
                               const std::vector<float>& weights);
Tensor softmax_ce_grad_weighted(const Tensor& logits, const std::vector<int>& labels,
                                const std::vector<float>& weights);

// FP64 有限差分 gradcheck：对 Linear 的每个权重/偏置扰动，返回 max abs 误差。
double gradcheck_linear(const Linear& m, const Tensor& x, const std::vector<int>& labels,
                        double eps = 1e-6);
// 加权版的 FP64 gradcheck（损失函数换成 softmax_ce_loss_weighted）。
double gradcheck_linear_weighted(const Linear& m, const Tensor& x,
                                 const std::vector<int>& labels,
                                 const std::vector<float>& weights, double eps = 1e-6);

}  // namespace uavrgb
