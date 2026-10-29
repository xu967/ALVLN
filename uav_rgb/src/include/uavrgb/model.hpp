#pragma once
// CNN 学生模型：3×(Conv3x3+ReLU+MaxPool2) → Flatten → Linear(4)。
// 前向/反向复用 src/*_cuda.cu 的算子（host Tensor 进出的 parity 模式：每次调用内部
// 临时 cudaMalloc + 同步）。未做 device-resident 优化，见 docs/worklog 的训练器记录。
#include "uavrgb/conv.hpp"
#include "uavrgb/nn.hpp"
#include "uavrgb/ops.hpp"
#include "uavrgb/tensor.hpp"

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
  Tensor flat;  // [N, fc.in]
};

struct CnnGrad {
  ConvGrad c1, c2, c3;
  LinearGrad fc;
};

// Xavier-uniform 初始化 W、b 置零；seed 固定则参数可复现。
// num_classes 默认 4（action.hpp 的 kNumActions），但保留参数用于结构自检。
CnnModel make_cnn(int in_h, int in_w, std::uint32_t seed = 0, int num_classes = 4);

std::size_t num_params(const CnnModel& m);

// 参数顺序固定：c1.W, c1.b, c2.W, c2.b, c3.W, c3.b, fc.W, fc.b。
// checkpoint 与 Adam 状态都依赖这个顺序，改顺序必须同步改 checkpoint 版本号。
std::vector<Tensor*> cnn_params(CnnModel& m);
std::vector<const Tensor*> cnn_params(const CnnModel& m);
std::vector<Tensor*> cnn_grad_tensors(CnnGrad& g);

// logits[N,num_classes]；cache 非空时顺带保存反向所需激活。
Tensor forward(const CnnModel& m, const Tensor& x, CnnCache* cache);
inline Tensor forward(const CnnModel& m, const Tensor& x) { return forward(m, x, nullptr); }

// 逐层反传：fc → pool3 → relu3 → conv3 → … → conv1。
CnnGrad backward(const CnnModel& m, const CnnCache& cache, const Tensor& dlogits);
// 无 cache 的便捷版：内部先跑一次带 cache 的 forward（训练循环应直接用上面的重载）。
CnnGrad backward(const CnnModel& m, const Tensor& x, const Tensor& dlogits);

}  // namespace uavrgb
