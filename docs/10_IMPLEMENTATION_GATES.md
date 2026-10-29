# 10｜最终实施计划、门禁、实验矩阵与停止条件

## A. 并行关系：不要把所有任务变成同一个阻塞链

| 阶段 | 必须交付的文件/证据 | 入口条件 | 不通过时 |
|---|---|---|---|
| M-1a 服务器 | doctor.json、API 测试、C++ gradcheck、GPU 枚举 | Ubuntu 目标机 | 修工具环境，不写算法 |
| M-1b Aerial 探针 | 1 scene/10 episodes 真实 AirSim trace、帧大小、RPC、reset | 有官方场景资产 | BLOCKED→优先 Gazebo Phys-VLN |
| GZ0 物理 | 四电机或可靠 SITL setpoint→真实 pose/twist，raw trace | Gazebo/PX4 | 不加语言模型 |
| GZ1 RGB | RGB 帧、时间戳、intrinsics、相机变换测试 | GZ0 | 修渲染器/相机，不做数据集 |
| GZ2 受控动作 | fixed policy → 新 RGB、定位/碰撞记录 | GZ1 | 修 ActionContract |
| DATA0 任务 | 可行路径/彩色语义地标/多步语言/遮挡/可解性测试 | 可渲染场景 | 不能宣称已生成 VLN 数据 |
| DATA1 真实 episode | 100+ 真实闭环 episodes、frame sha、真实指令/动作日志 | GZ2+DATA0 | 不启动训练主线 |
| CPP0 数值 | FP64 gradcheck/独立 oracle/CPU-CUDA parity | C++ 编译成功 | 不做长训练 |
| TRAIN0 baseline | 实际训练 loss/val 曲线、小集过拟合、至少一个真闭环数字 | DATA1+CPP0 | 不加 ECPT/KD |
| METHOD0 ECPT | 相同参数、相同 episodes、独立 STOP 对照+CI | TRAIN0 | 无净收益就删 ECPT claim |
| EDGE0 NPU | 编译/INT8/真实板端 P95/RSS/失败率 | 板卡+随机模型 | 不声称可部署 |
| CF0 控制内核 | CF PID/mixer 代码对拍、SIL 与姿态漂移模拟 | 源码许可/型号核验 | 不阻断 TRAIN0 |
| HIL0 串口安全 | 断链/重放/乱序/TTL/接管等可复现 trace | ESP32 后续已安全改造 | 不进行有桨测试 |

UI 状态 `ready` 必须来自可靠门禁证据与验证，不可以因 `artifacts/foo.txt` 存在就自动 PASS。本包的服务端当前只知道 profile 注册是否可信，不会捏造 GZ0 证据。

## B. 训练实验矩阵（同模型容量级别）

| ID | 学生 | 要检验的假设 |
|---|---|---|
| B0 | CNN + 文本均值 + 单 GRU + 官方动作 head | 任务下界/真实闭环 |
| B1 | B0 + 等容量的独立 STOP head | STOP 不平衡是否可校准 |
| B2 | B1 + 传统进度辅助头 | 是否仅由 progress supervision 带来收益 |
| B3 | B2 的证据/进度联合门控 ECPT | 新机制是否有独立增量 |
| B4 | B3 + 简单单帧/双帧历史消融 | 时序运动是否必要 |
| B5 | B3 + 几何教师（仅训练时特权） | KD 是否超越非 KD 同容量基线 |
| B6 | B3 在真实板上 INT8 | 能否达到精度–延迟–内存可用点 |

**不强制做 B5**：如果专家/教师闭环失效，删除 KD。若 B3 无显著增益，转向容量—性能曲线和风险校准的受控研究，不用扩大 ROI 去挽救无效假设。

## C. 评估协议

- 每一方法 3 个开发 seed，关键最终结果 5 seed，训练/阈值仅在 train/开发 split 调参。
- 保留固定 `pair_id`，策略 A/B 使用相同 scene/seed/起点/光照/指令/物理配置；采样单位按独立场景/轨迹聚类。
- 报 `SR`, `OSR`, `NE`, `nDTW`, `SDTW`, collision, time, distance, early_stop, missed_stop, stop ECE/Brier，不能只报 train CE。
- 官方 AerialVLN-S 的 5.1% LAG val_unseen 等历史数据仅是其原始论文特定设置下的参考，不能未经 RGB-only 复现就设成公平硬门槛。
- 统计能力有限时，报告置信区间和没有观察到分辨差异，不能因为小提升未达 p<0.05 就宣称方法无效；反过来也不能用几条成功案例声称“普适有效”。

## D. 优先级与工期说明

建议单人/一张 GPU 的最早两周以“验证能否运行 + 必须可产生证据”为中心：

**第 1–3 天**：服务器 doctor、API、C++ gradcheck、安装 Gazebo 与 RTX 5090 驱动核验；并开始独立 Aerial 兼容性探针。
**第 4–7 天**：启动官方 x500/X3；拿到真实电机/姿态 trace 与 RGB；完成学生输入隔离合同和第一条受控 episode。
**第 8–11 天**：语义场景/任务/专家日志，生成 16–64 条真实 Episode，盲策略和语言打乱对照。
**第 12–14 天**：至少完成 C++ Tensor/Conv 基本数值门禁，确定下一期训练器任务，不要在这两周堆新 UI 页面。

这只是资源安排建议，不是已执行记录，也不承诺一定在 14 天内完成真实物理集成；若硬件/旧 UE 构建阻断，优先记录明确 blocker 与替代路径。

## E. 版本门禁禁令

V8 禁止 ESP32 固件修改、发布真实飞行成绩、在论文摘要中用没有部署过的 `1TOPS/10Hz` 声称实测、向前端伪造训练 run、将任意一个 `replay` 输出命名为官方 SR。所有外部引用均应在论文前检索最新版本并核对 split/输入模态。
