# 09｜LicheeRV Nano (SG2002) 端侧部署链路和量化验收

## 1. 已知硬件与必须核验的事实

按 Sipeed 官方 Wiki：LicheeRV Nano SG2002 的主核运行在 1GHz，RISC-V C906/ARM A53 **择一**，辅助 C906 700MHz；集成 256MB DDR3；NPU 标称 1TOPS INT8（BF16 支持）。部分较早 PCB 版本有 NPU 频率/算力稳定性限制，因此实际可用吞吐需要查 PCB 丝印并实测。官方：https://wiki.sipeed.com/hardware/zh/lichee/RV_Nano/1_intro.html 。

## 2. 先进行算子与真实硬件探针（不要等论文模型训练完）

| 编号 | 操作 | 必须保存 |
|---|---|---|
| E0 | `uname`, SoC/固件/PCB 型号、启动模式、内存 | hardware.json、照片/固件版本 |
| E1 | 单层 Conv/DWConv/PWConv/Pool 小模型 FP32/INT8 转换 | ONNX hash、MLIR 编译日志、cvimodel |
| E2 | NPU runtime 单张输入推理和输出数值 parity | p50/p95/p99、max abs error |
| E3 | MIPI-CSI/ISP/VPSS 实际输出 RGB | 图像尺寸、FOV、颜色空间、时间戳 |
| E4 | CPU 后处理、文本编码和 GRU 原型 | 单核 CPU 利用率、内存、线程抖动 |
| E5 | NPU+CPU 端到端（真实相机） | camera→UART 时间段、RSS、丢帧、deadline miss |
| E6 | 串口 HIL | CRC/序号/TTL/ACK/错误/接管 trace |

工具：厂商 TPU-MLIR/SG200X SDK（版本必须与实际 SG2002 build 匹配）。ONNX 前端能解析某个算子不代表 CV181x 编译器后端有该算子的可用 INT8 实现；GRU/动态 shape/attention/LayerNorm 等先放 CPU 或导出为可验证静态算子。

## 3. 部署图划分

- **NPU**：静态 shape 视觉 backbone，通常选 batch 1；输入经过 ISP/VPS 硬件缩放和色彩转换后以 INT8/uint8 定标送入。是否支持动态双帧/多输出，以真实编译结果为准。
- **CPU**：一次编码完整指令并缓存 token、每步进度/视觉小向量记忆/独立 STOP 策略；管理已验证的相机时间戳与命令序号；UART 协议/日志。
- **ESP32**：实际底层安全仲裁/姿态稳定。不存在已验证的高度保持时不能默认 STOP 会悬停。

## 4. 推理预算：列出目标，但不声称已达到

初始候选目标 `camera→policy intent` P95 ≤ 150ms、单帧内存峰值按 256MB 总 RAM 的实际 Linux/NPU 分区留出足够余量、决策率 5–10Hz，**仅作为调参门槛草案**。最终值必须在该板上以准确时钟统计并以实测物理制动距离、无人机速度和飞控能力重新设定；`1 TOPS` 不能用来反推真实 ms 或系统 FPS。

```
T_total = T_exposure/ISP + T_VPSS + T_layout_copy + T_NPU
          + T_CPU_memory + T_decision + T_UART_serialize + scheduling_jitter
```

要同时报告 P50/P95/P99、30–60min 连续稳定性、内存 RSS、频率/功耗、摄像头曝光切换时延、drop frame 和时序漂移。单个 NPU 推理 latency 不等于实际无人机动作到达时延。

## 5. 导出流程

```text
C++/CUDA trained FP32 checkpoints -> deterministic state_dict export
-> static ONNX (pinned opset, raw preprocessing test vectors)
-> TPU-MLIR calibration set (TRAIN-only, varied light/texture)
-> INT8/BF16 CVIMODEL -> SG2002 SDK runtime
-> on-board parity and closed-loop tests
```

不允许把 Python 端 onnxruntime 可运行当成板端已通过。必须测试同一真实 RGB 输入经过 C++/Python/板端前处理后像素采样一致，INT8 校准只来自 train，不能泄漏 val_unseen。

## 6. 仍未完成

本包不包括 SG2002 SDK、转换后的 .cvimodel、摄像头实时捕获程序、NPU benchmark 或实测数据。当前没有 `model_fps=...` 的可引用数字。
