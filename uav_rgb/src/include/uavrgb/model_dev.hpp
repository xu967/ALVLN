#pragma once
// 设备常驻的 CNN 训练/推理路径：参数、Adam 状态、激活全部留在显存，每步只 H2D 一批 uint8
// 图像（+token id）、D2H logits 与梯度范数；loss 与 dlogits 仍在 host 用现有函数算。
// kernel 与 host Tensor 路径（model.cpp）共用 ops_dev.hpp，两条路径的数值一致性由
// test_device_train_step 对拍保证；host 路径继续作为 CPU/CUDA parity 测试的参考。
#include "uavrgb/checkpoint.hpp"
#include "uavrgb/model.hpp"
#include "uavrgb/ops_dev.hpp"  // dev::AugParams

#include <cstdint>
#include <vector>

namespace uavrgb {

class DeviceCnn {
 public:
  // 按 model 结构分配显存（参数/梯度/Adam m,v + max_batch 的激活）并上传参数；
  // state 非空时一并上传 m/v。
  DeviceCnn(const CnnModel& model, int max_batch, const TrainState* state = nullptr);
  ~DeviceCnn();
  DeviceCnn(const DeviceCnn&) = delete;
  DeviceCnn& operator=(const DeviceCnn&) = delete;

  // host → 显存：模型结构必须与构造时一致（只查元素数）。
  void upload(const CnnModel& model, const TrainState* state = nullptr);
  // 显存 → host：存 checkpoint 前调用。
  void download(CnnModel& model, TrainState* state = nullptr) const;

  // images：n 个 NCHW uint8 样本连续排布（像素在设备侧 /255；双帧模型每样本 6 通道）；
  // text：n*kTextLen 个 token，无文本分支时传空指针；aug 非空时按逐样本参数做数据增强
  // （只在训练批传）；prev：n 个 prev-action id，模型不接 prev-action 时传空指针。
  // 返回 host 上的 logits [n, num_classes]。
  Tensor forward(const std::uint8_t* images, int n, const std::uint8_t* text,
                 const dev::AugParams* aug = nullptr, const std::uint8_t* prev = nullptr);
  // 反向 + 全局梯度裁剪 + Adam；dlogits 对应最近一次 forward。lr/β/eps 取自
  // state.adam[k]，每个 adam[k].t 自增（与 host 路径的 adam_step_cuda 一致）。
  // weight_decay > 0 即 AdamW 解耦衰减。返回裁剪前的梯度范数。
  float backward_step(const Tensor& dlogits, float max_grad_norm, TrainState& state,
                      float weight_decay = 0.0f);

  // 深度分带辅助头（aux_classes>0 才可调，须紧跟 forward）：targets[n*aux_h*aux_w]（255 屏蔽）。
  // 在显存里跑 1x1 头 + 逐格 CE，返回 {平均 CE, argmax 命中格数, 有效格数}，并把
  // λ·dL/dlogits 留在显存给随后的 backward_step 叠加；只评估就不调 backward_step。
  // 推理路径不调它，辅助头的 MAC 不花。
  struct AuxStats {
    float loss = 0.0f;
    std::size_t correct = 0, valid = 0;
  };
  AuxStats aux_loss(const std::uint8_t* targets, float lambda);
  int aux_h() const { return layer_[2].ph; }
  int aux_w() const { return layer_[2].pw; }

  int max_batch() const { return max_batch_; }

  struct LayerDims {  // 一层 conv(+ReLU)+pool：输入 h,w → conv 输出 oh,ow → 池化输出 ph,pw
    int c_in, c_out, kh, kw, pad, h, w, oh, ow, ph, pw;
  };

 private:
  struct Seg {  // 参数在扁平缓冲里的位置
    std::size_t off = 0, len = 0;
  };
  template <typename T>
  T* alloc(std::size_t count);
  std::size_t seg_bytes(std::size_t k, std::size_t numel) const;  // 校验元素数后换算字节
  void film_forward(int n);
  void film_backward(int n);
  void alloc_aux(const CnnModel& model);  // model_dev_aux.cu
  void aux_backward(int n);               // 把辅助头梯度叠进 dflat_；本步没目标则梯度清 0
  const float* fc_input() const { return has_film_ ? p3mod_ : p3_; }

  int max_batch_ = 0, n_ = 0;           // n_：最近一次 forward 的 batch，0 = 还不能反向
  int in_c_ = 0, h_ = 0, w_ = 0, text_dim_ = 0, num_classes_ = 0;
  int prev_dim_ = 0, cond_dim_ = 0, aux_classes_ = 0;
  bool has_film_ = false, aux_pending_ = false;  // aux_pending_：本步已算 aux 梯度
  // 可选分支在 seg_ 里的下标（-1 = 没有）：text_emb / film.W（film.b 紧随）/ aux.W（aux.b 紧随）。
  int seg_emb_ = -1, seg_film_ = -1, seg_aux_ = -1;
  LayerDims layer_[3] = {};
  std::vector<Seg> seg_;                // 与 cnn_params 顺序一致
  std::size_t total_params_ = 0;
  std::vector<void*> owned_;            // 全部 cudaMalloc 结果，析构统一释放
  float *params_ = nullptr, *grads_ = nullptr, *m_ = nullptr, *v_ = nullptr;
  std::uint8_t* x_u8_ = nullptr;
  std::uint8_t* text_u8_ = nullptr;
  std::uint8_t* prev_u8_ = nullptr;
  dev::AugParams* aug_ = nullptr;
  // 激活（r = ReLU 原地覆盖 conv 输出；p/arg = 池化输出与 argmax）。
  float *x_ = nullptr, *c1_ = nullptr, *p1_ = nullptr, *arg1_ = nullptr;
  float *c2_ = nullptr, *p2_ = nullptr, *arg2_ = nullptr;
  float *c3_ = nullptr, *p3_ = nullptr, *arg3_ = nullptr;
  float *cond_ = nullptr, *film_out_ = nullptr, *p3mod_ = nullptr, *logits_ = nullptr;
  // 反向临时量。
  float *dlogits_ = nullptr, *dflat_ = nullptr, *dfilm_ = nullptr, *dcond_ = nullptr;
  float *dr3_ = nullptr, *dp2_ = nullptr, *dr2_ = nullptr, *dp1_ = nullptr, *dr1_ = nullptr;
  float* ws_[3] = {nullptr, nullptr, nullptr};  // 各层 conv 反向的部分和工作区
  double* d_sumsq_ = nullptr;
  // 辅助头：logits/梯度 [n, aux_classes, ph, pw]、对特征的梯度 [n, c3.out, ph, pw]、目标、统计。
  float *aux_logits_ = nullptr, *daux_ = nullptr, *dp3aux_ = nullptr, *ws_aux_ = nullptr;
  std::uint8_t* aux_tgt_ = nullptr;
  double* d_aux_loss_ = nullptr;
  unsigned long long* d_aux_correct_ = nullptr;
};

}  // namespace uavrgb
