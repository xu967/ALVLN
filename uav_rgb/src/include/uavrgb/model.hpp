#pragma once
// CNN 学生模型：3×(Conv3x3+ReLU+MaxPool2) → Flatten → Linear(4)。
// 前向/反向复用 src/*_cuda.cu 的算子（host Tensor 进出的 parity 模式：每次调用内部
// 临时 cudaMalloc + 同步）。未做 device-resident 优化，见 docs/worklog 的训练器记录。
#include "uavrgb/conv.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/ops.hpp"
#include "uavrgb/tensor.hpp"
#include "uavrgb/text.hpp"  // kTextLen / kTextVocab

#include <cstddef>
#include <cstdint>
#include <vector>

namespace uavrgb {

struct CnnModel {
  Conv2d c1;  // 3->16, 3x3, pad1
  Conv2d c2;  // 16->32
  Conv2d c3;  // 32->64
  Linear fc;  // 64*(in_h/8)*(in_w/8) -> num_classes
  int in_h = 0;
  int in_w = 0;
  // 文本分支（可选）：每个 token 位置各取 embedding，拼成 kTextLen*text_dim 维，
  // 再经 film 生成逐通道的 γ/β，调制 conv 最后一层特征（FiLM）：
  //   y' = (1+γ)·y + β
  // 纯 concat 是 late fusion，换文本只能加一个与图像无关的常数 bias，无法表达
  // “文本说红 → 去画面里的红柱”；实测三种训练都让模型忽略文本，故改 FiLM。
  // text_dim == 0 表示没有文本分支；此时 forward 必须不传 text。
  Tensor text_emb;  // [kTextVocab, text_dim]
  Linear film;      // [kTextLen*text_dim] -> 2*c3.out（前半 γ，后半 β）
  int text_dim = 0;
};

// 反向所需的中间激活；只有传了 cache 的 forward 才填充（否则全空，省内存）。
// r* 是 ReLU 输出（= pool 的输入），c* 是 ReLU 前的卷积输出。
struct CnnCache {
  Tensor x;  // [N,3,in_h,in_w]
  Tensor c1, r1;
  MaxPoolResult p1;
  Tensor c2, r2;
  MaxPoolResult p2;
  Tensor c3, r3;
  MaxPoolResult p3;
  Tensor flat;       // [N, 图像部分]：fc 的输入（FiLM 调制后的 flatten）
  Tensor text_feat;  // [N, kTextLen*text_dim]
  Tensor film_out;   // [N, 2*c3.out]：前半 γ，后半 β
  std::vector<std::uint8_t> text_ids;  // [N*kTextLen]，与 text_feat 同批
};

struct CnnGrad {
  ConvGrad c1, c2, c3;
  LinearGrad fc;
  LinearGrad film;   // 只有文本分支存在时有效
  Tensor text_demb;  // [kTextVocab, text_dim]；无文本分支时为空
};

// Xavier-uniform 初始化 W、b 置零；seed 固定则参数可复现。
// num_classes 默认 4（action.hpp 的 kNumActions），但保留参数用于结构自检。
// text_dim > 0 时额外建 [kTextVocab, text_dim] 的 embedding 表和 FiLM 层；
// fc 输入维度不变（文本只经 FiLM 调制，不拼进 fc 输入）。
CnnModel make_cnn(int in_h, int in_w, std::uint32_t seed = 0, int num_classes = 4,
                  int text_dim = 0);

std::size_t num_params(const CnnModel& m);

// 参数顺序固定：c1.W, c1.b, c2.W, c2.b, c3.W, c3.b, fc.W, fc.b；
// 有文本分支时追加 text_emb, film.W, film.b（共 11 个）。
// checkpoint 与 Adam 状态都依赖这个顺序，改顺序必须同步改 checkpoint 版本号。
std::vector<Tensor*> cnn_params(CnnModel& m);
std::vector<const Tensor*> cnn_params(const CnnModel& m);
std::vector<Tensor*> cnn_grad_tensors(CnnGrad& g);

// logits[N,num_classes]；cache 非空时顺带保存反向所需激活。
// text 非空时按 [N*kTextLen] 的 token id 逐位置取 embedding，拼成
// [N, kTextLen*text_dim]，再经 film 生成 γ/β 对 conv 最后一层做 FiLM 调制
// （<pad> 槽位保持 0，不做 mean-pool）；m.text_dim == 0 时必须为 nullptr，否则抛异常。
Tensor forward(const CnnModel& m, const Tensor& x, CnnCache* cache,
               const std::vector<std::uint8_t>* text = nullptr);
inline Tensor forward(const CnnModel& m, const Tensor& x) { return forward(m, x, nullptr); }

// 逐层反传：fc → pool3 → relu3 → conv3 → … → conv1。
CnnGrad backward(const CnnModel& m, const CnnCache& cache, const Tensor& dlogits);
// 无 cache 的便捷版：内部先跑一次带 cache 的 forward（训练循环应直接用上面的重载）。
CnnGrad backward(const CnnModel& m, const Tensor& x, const Tensor& dlogits);

}  // namespace uavrgb
