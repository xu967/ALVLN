#pragma once
// 设备指针级算子：不分配、不拷贝、不同步，只 launch 并查 cudaGetLastError。
// *_cuda 的 host Tensor 包装（CPU/CUDA parity 对拍用）与设备常驻训练/推理路径（model_dev.cu）
// 共用这一批 kernel。形状合法性由调用方保证：host 包装先过各自的 check_dims，常驻路径按模型
// 结构一次算好。全部 launch 在默认流上，调用方自己决定何时 cudaDeviceSynchronize。
#include <cstddef>
#include <cstdint>

namespace uavrgb::dev {

struct ConvShape {
  int n, c_in, h, w, out, kh, kw, pad, oh, ow;
};
void conv2d_forward(const ConvShape& s, const float* x, const float* w, const float* b, float* y);
// dw/db 先按 (样本, 行分块) 并行算部分和写进 ws，再按固定顺序归约：同输入逐位可复现。
// ws 至少 conv2d_backward_workspace(s) 个 float；dx 为空指针时不算（第一层的 dx 没有下游）。
std::size_t conv2d_backward_workspace(const ConvShape& s);
void conv2d_backward(const ConvShape& s, const float* x, const float* w, const float* dy,
                     float* dw, float* db, float* dx, float* ws);

// 逐样本数据增强参数（训练批才用）。输出像素 (y,x) 取自输入 (y+dy, flip ? w-1-x : x + dx)
// 边缘复制，再做 gamma → 对比度（绕 0.5）→ 亮度，截断回 [0,1]。
struct AugParams {
  float gamma, contrast, brightness;
  int dx, dy, flip;
};
// uint8 NCHW → float [0,1]；aug 为空指针时就是 src/255.0f（与 infer.cpp 同一舍入）。
void u8_to_float(const std::uint8_t* src, float* dst, const AugParams* aug, int n, int c, int h,
                 int w);

void relu_forward(const float* x, float* y, int total);                    // 允许 y == x
void relu_backward(const float* x, const float* dy, float* dx, int total);  // 允许 dx == dy

struct PoolShape {
  int n, c, h, w, k, s, oh, ow;
};
void maxpool_forward(const PoolShape& s, const float* x, float* y, float* argmax);
// dx 内部先清零；argmax 必须来自同形状的 maxpool_forward（设备侧不校验范围）。
void maxpool_backward(const PoolShape& s, const float* argmax, const float* dy, float* dx);

void linear_forward(int n, int in, int out, const float* x, const float* w, const float* b,
                    float* y);
void linear_backward(int n, int in, int out, const float* x, const float* w, const float* dl,
                     float* dw, float* db, float* dx);

// Adam 单步，bc1/bc2 = 1-β^t 由 host 算好；weight_decay > 0 即 AdamW 解耦衰减 p -= lr·wd·p。
void adam_step(float* p, const float* g, float* m, float* v, float lr, float b1, float b2,
               float bc1, float bc2, float eps, float weight_decay, int total);
void sumsq(const float* x, double* accum, int total);  // accum 由调用方清零
void scale(float* x, float coef, int total);

}  // namespace uavrgb::dev
