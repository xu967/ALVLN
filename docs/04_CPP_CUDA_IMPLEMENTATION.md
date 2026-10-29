# 04｜C++17 / CUDA 专用训练器实施设计与优先级

> 此文件是代码开发合同；V8 实际已实现的只有 `cpp/include/tensor_ops.hpp` 及 `cpp/tests/gradcheck.cpp`（简单 Linear+Softmax-CE 双精度有限差分），以及从评审修复版继承的 `cpp/include/training_monitor.hpp`。**并没有**可训练的 CNN/GRU、真正的 CUDA 张量引擎或 TensorRT。

## 1. 确定目标：专用静态图，不造通用框架

训练对象：`RGB[B,3,224,384] + token_id[B,L] + episode_history + action_label[B,T]` 的视觉语言学生。首轮支持 `L` 覆盖训练语料 P99 甚至全长（例如先分配 `L=192`，实际根据长度统计而不是拍脑袋定 48）；`T` 为可配置 TBPTT chunk，如 16 或 32；输入独立 RGB 与语言指令，不读训练真值 metadata。

**不要在第一阶段实现**动态自动微分、任意控制流计算图、分布式训练器、自研 GEMM/NPU 编译器、通用 Python 绑定。允许调用 cuBLAS/cuDNN 优化卷积/GEMM，但数据布局、前反向合同、损失、优化器和模型导出/推理一致性由项目控制。

## 2. 建议代码组织（实际文件按需添加）

```text
cpp/
  CMakeLists.txt
  include/
    tensor_ops.hpp              # 现有 CPU Linear CE reference
    training_monitor.hpp        # 已修复 JSONL monitor
    tensor_view.hpp             # 待开发，纯 non-owning view
    episode_contract.hpp        # 待开发，学生白名单字段
  src/
    tensor.cpp                   # ownership, allocator, strides
    math_cpu.cpp                # independent FP64 reference
    conv_cpu.cpp                # Conv2D and backward reference
    train_main.cpp              # CLI, checkpoint, batch loop
    model/tiny_backbone.cpp     # static depthwise+pointwise CNN
    model/text_encoder.cpp      # embedding + minimal recurrent text
    model/nav_policy.cpp        # GRU/ECPT/STOP forward/backward
    export/onnx_adapter.cpp     # static export description (not hand-written protobuf)
  cuda/
    conv.cu                     # explicit forward/backward, reuse cuDNN if suitable
    embedding.cu
    recurrent.cu
    optimizer.cu
  tests/
    gradcheck.cpp               # 已实现
    conv_reference.cpp          # 待开发
    episode_reset.cpp           # 待开发
    export_parity.cpp           # 待开发
```

这些是职责建议，不是要求按目录凑足文件数量。业务 C++ 默认每文件约 280 行是审查提醒而不是挤压必要测试的机械指标；`sim/src` 允许在真实需求成立时批准扩大到 10 个实现文件/约 3500 有效行，禁止把真实控制/渲染/日志强塞进单文件。

## 3. Tensor 和 operator 合同

```cpp
struct TensorView {
  void* data;         // device/host pointer
  int64_t* dims;      // checked ranks, static graph
  int64_t* strides;   // explicit NHWC vs NCHW
  int32_t rank;
  DataType dtype;
  Device device;
};
struct OpContext {
  TensorView input, output, grad_input, grad_output;
  ScratchAllocator* scratch;
};
```

- FP32 CPU oracle 与 CUDA forward/backward 对拍；用于独立正确性测试的 FP64 finite differences `loss(w+eps)-loss(w-eps)`。
- Tensor 所有权与生命周期由 separate allocator 管理，不允许 TensorView 隐式管理内存。
- NCHW/NHWC 转换显式记录 copy/stride；不能让 GPU 计算正确但 ONNX 输入 RGB/BGR 顺序错误。
- 任何动量状态、优化器 step、随机数流、词表、GRU 隐状态都必须支持 checkpoint/resume。

## 4. 第一阶段静态学生网络

| 层 | 建议形状（以 [1,3,224,384] 为例） | 验收 |
|---|---|---|
| Conv3×3/s2 3→16 | [1,16,112,192] | 对拍、quantization axis |
| DWConv3×3/s2 16 + PW→24 | [1,24,56,96] | 正反向，FP32/INT8 |
| DWConv3×3/s2 24 + PW→32 | [1,32,28,48] | 算子是否能编译 CV181x |
| DWConv3×3/s2 32 + PW→48 | [1,48,14,24] | 显存与速度 |
| DW/PW 残差 ×2 + Pool/Projection | [1,64,7,12] | NPU 多输出/最终 logits |
| 文本 Embedding/GRU | [B,L,64] | 支持完整指令，reset |
| 视觉语言融合 + 策略记忆 | 64–128 维递归状态 | CPU/NPU 分图明确 |
| 移动动作 head + 独立 STOP head | 7+1 或 benchmark 8+stop | 分类、校准与协议一致 |

参数/MAC/峰值内存只作为**预算估计**，每个 PR 更新精确逐层统计，并以真实板端 P50/P95/p99 代替“1 TOPS 推算 0.2ms”的断言。此 backbone 非最佳模型结论，只是 bring-up 候选。

## 5. 训练损失、采样、梯度

阶段 1：`L_BC = CE(logits_action,expert_action)`，STOP 用 `BCEWithLogits`（配合类别权重或 focal loss，与普通 8 路 softmax 做等容量对照）。阶段 2：加有标签来源的进度辅助监督与证据校准（见 `05_ECPT_SPEC.md`）；阶段 3：只有强于学生且校准合格的教师才加 `L_KD=T² KL(p_teacher^T||p_student^T)`。

- 划分严格按 scene/path/episode group，不能把同路径不同语言指令随机分到 train 和 validation。
- 强制数据顺序/种子固定；梯度积累不能无意按 step 多次更新 AdamW 计数。
- TBPTT 的每个 chunk 必须携带 `valid_mask`，episode reset 后 `h=0`；不允许跨两条 episode 遗漏状态。
- 真实导航学习应加入学生自生状态的交互纠偏（DAgger 或有限纠错），只训练专家状态上的单步准确率不足以保证闭环成功。
- 如果引入 privileged deep/pose labels，需要 label policy 防止学生输入偷偷包含真值字段。

## 6. 数值正确性矩阵（严禁跳过）

| 关卡 | 方法 | 目标 |
|---|---|---|
| N1 | FP64 finite-difference gradcheck | `max abs/rel` 在算子预设容差内 |
| N2 | 与独立 NumPy/PyTorch **测试级 oracle** 对拍 | 前向 logits/梯度逐层一致，绝不进主训练 |
| N3 | CPU vs CUDA FP32 | 前向/反向、随机种子、stride/mask、padding 一致 |
| N4 | 小数据 16/64 episode 过拟合 | loss 可控下降，序列 reset 正确 |
| N5 | checkpoint 续训一致性 | 同配置续训等价或在可解释浮点波动内 |
| N6 | 导出前后 parity | ONNX/Python 测试级/SG2002 INT8 分段误差报告 |

要求测试 `padding`、全部 mask、长度 0、极长指令、极大 logits、NaN/Inf、卷积 stride/padding、DWConv channel multiplier、dropout off、训练/验证模式、相机 RGB/BGR。必须支持“故意注入一个转置/标签映射 bug”验证测试能抓到。

## 7. 编译/测试

```bash
./run.sh cpp-test
# 现阶段输出 TinyVLN CPU linear/CE FP64 gradcheck，不是 UAV 模型训练
```

真正训练器完成时计划命令（当前不会成功）：

```bash
cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/cpp -j
./build/cpp/tinyvln_train --config config/experiment.yaml --output artifacts/runs/<run_id>
```

当 `tinyvln_train` 真正实现，先在服务器 `config/registry.json` 将它登记为 `training_profiles`，经预检通过才允许 UI 启动；前端绝对不能传 arbitrary `argv`、脚本路径或 checkpoint 文件名。

## 8. 训练进度与日志

- C++ 终端 stderr 用固定刷新间隔打印 epoch/step、训练 loss、验证 loss、accuracy、STOP precision/recall、学习率、梯度范数、更新吞吐、ETA。
- 原始 stdout/stderr 归档，不覆盖或重写；结构化 `train_step`/`validation` 事件写 JSONL，浮点输出固定 C locale，NaN/Inf 先写 `TRAIN_NONFINITE_METRIC` 错误再安全退出。
- `results.csv` 由真实训练器按既定 schema 写出；NavLab 后端只读取它，绝不把动作 accuracy 推导为 navigation SR。
- run_id 单进程独占，checkpoint/best/last 持续持久化；崩溃后仍可导出 AI audit bundle。

## 9. 与旧 PyTorch 工程关系

`references/snapshots/web_code_review_20261027.zip` 是另一条对话的历史代码快照（包含 Python AerialVLN/LANI 训练和评测逻辑）。它提供动作类别、数据字段、缓存/评测合同的**参考实现**，也可以用于隔离的正确性 oracle；不是 V8 C++ 学生训练已实现的证据。不要把旧 DINOv3 模型权重、RGB-D 深度特征直接当作 SG2002 的 RGB-only 学生。
