# TinyAerialVLN V7.1 独立 AI / 研究人员评审核查包

> 可将本文和 `V7_1_RESEARCH_PLAN.md`、`CODE_RULES.md`、`docs/V7_1_UI_SYSTEM_DESIGN.md` 及工程 ZIP 一同给另一位 AI。要求评审员**质疑每个假设，而不是礼貌性认同**。
>
> 结论证据等级：`source_direct` = 源码中可定位；`local_tested` = 本次容器实际测试；`design_only` = 计划；`device_unverified` = 用户实际设备未验收；`literature_pending` = 需再次核实公开文献。

## 0. 项目任务与不可偷换的约束

本项目研究低算力空中 VLN：自然语言指令 + 前视 RGB 序列 + 过去动作 → 导航高层动作 + 进度记忆 + 终止判断。不能把任务降级为仅 RGB 的避障四分类而不作说明。

开发主机为 Ubuntu + RTX 5090；学生训练主路径优先 C++/CUDA 自研专用训练器，不依赖 PyTorch 运行；推理使用 LicheeRV Nano (SG2002)；导航端通过 UART 向 ESP32 发送有界高层指令。飞控内部使用 IMU 不等于视觉学生融合 IMU。公开 AerialVLN-S 与自建连续动力学场景必须分别评测，不可把离散 SetPose 视为真实自由飞行。

**V7.1 用户明确允许未来修改 ESP32 原代码，但当前阶段不得修改。** 本轮只调整界面、只读状态诊断、测试与文档，固件不刷写、不改变 PID、不改变引脚或通信协议。

## 1. 可直接在工程中复核的源码事实

| 编号 | 断言 | 定位 | 等级 | 可能的争议 |
|---|---|---|---|---|
| C01 | `run.sh` 是唯一的顶层脚本入口，调用 Python launcher 与 sim_preflight | `run.sh` | source_direct | `sim` 子命令有限 |
| C02 | 看板只绑定回环地址 127.0.0.1 | `tools/dashboard.py:serve` | source_direct | 尚未测试远程代理暴露情况 |
| C03 | 看板 GET 读取本地状态、场景与真实 JSONL | `tools/dashboard.py` | source_direct | 大规模/长时间运行稳定性尚需测试 |
| C04 | POST 返回 405，不提供训练/飞控控制 | `Handler.do_POST` | source_direct | 本地服务仍需最小权限部署 |
| C05 | 五个 UI 页面为总览、训练、场景、边缘、评审 | `assets/dashboard.html` | source_direct | 视觉设计需目标用户评价 |
| C06 | 日志尾部只读，拒绝非法 run ID / symlink / NaN / 损坏行 | `metrics_for` | source_direct+local_tested | 非有限嵌套数据尚未递归验证 |
| C07 | 实际已有场景是静态 SDF（forest/corridor） | `artifacts/scenes/*.json` | source_direct+local_tested | 不能证明 Gazebo 动态碰撞 |
| C08 | `sim smoke` 拒绝没有 multicopter plugin 的 world | `tools/sim_preflight.py` | source_direct+local_tested | 插件出现不等于插件实际可用 |
| C09 | `sim camera` / `sim cf-sil` 返回功能未实现 | `tools/sim_preflight.py` | source_direct+local_tested | 没有实际 Gazebo 验收 |
| C10 | CF-Drone 原固件未被打包入此工程 | 工程 ZIP 内容及独立 CF-Drone.zip | source_direct | 原固件仍需要单独许可、接线复核 |

## 2. 本次已执行的测试（不得越界解读）

- `python3 -m unittest discover -s code/tests -p 'test_*.py' -v`：13 例通过。
- `python3 -m unittest discover -s sim/tests -p 'test_*.py' -v`：2 例通过。
- `./run.sh doctor`：能运行，显示找不到 `nvcc`、`gz`，训练配置和可执行训练目标尚未建立。
- `./run.sh sim doctor`：在当前容器中返回 `GZ_NOT_INSTALLED`，这是预期阻断。
- 本地 Python HTTP 客户端实际 GET `/api/health/runs/scenes/status/audit` 返回有效 JSON；见未来 CI 可重复命令。
- Chromium 使用 `page.set_content` 加载 HTML，注入本地实际 status/scene 结果，切换总览/训练/仿真无 JS pageerror；**沙盒 Chromium 阻止浏览到 localhost 的直接导航，因此没有宣称浏览器—HTTP 完整链路已验证**。
- 所有截图来自无训练指标的真实空状态，绝不是学生训练结果图。

## 3. 未实现与未知事项（一级问题，不能省略）

| ID | 风险描述 | 当前状态 | 必须的 gate | 失败策略 |
|---|---|---|---|---|
| X01 | Ubuntu+5090 上 Gazebo Harmonic 与 GPU 渲染兼容性 | device_unverified | `gz sim` + RGB dump | 优先修复驱动/渲染，不假装模拟器已跑 |
| X02 | X3 电机模型真实产生力/矩和运动 | design_only | 电机输入与 pose/速度 trace | 检查 rad/s、motorConstant、方向与惯量 |
| X03 | 机载 RGB 与动力学同步 | design_only | frame timestamp + camera extrinsics | 不启用语言学生闭环 |
| X04 | CF-Drone PID/mixer 跨平台数值对拍 | design_only | golden vectors + 物理 SIL | 保留官方物理控制器对照，不改固件 |
| X05 | CF-Drone 原代码中的定高/速度闭环缺口 | source_from_previous_V7_audit | 控制能力/退化保护测试 | 禁止裸机自由飞行 |
| X06 | 自研 C++/CUDA 算子与梯度的正确性 | design_only | FP64 gradcheck + 独立 oracle | 回滚问题算子而非追加复杂模块 |
| X07 | 自研学生在未见场景闭环导航 | design_only | SR/OSR/NE/SDTW 和语言干预 | 返回数据/标签与基线验证 |
| X08 | 大型视觉教师真实增益 | design_only | 教师与几何专家闭环对照 | 无增益则不蒸馏 |
| X09 | SG2002 NPU 支持与 camera→UART p95 | device_unverified | cvimodel + 实机跟踪日志 | 缩模型/静态算子，不换板 |
| X10 | UART 的 ACK/EXEC_DONE、CRC、TTL、接管语义 | design_only | 无桨 HIL 故障注入 | 当前冻结固件，未来单独立项 |
| X11 | AerialVLN-S 官方仿真器 + 5090 兼容性 | device_unverified | 10 条真实闭环回放 | 延后官方 benchmark，不虚构成绩 |
| X12 | MiniWorld 训练的迁移价值 | literature_pending+design_only | 新三维未见场景测试 | 不把 MiniWorld 当真机证据 |

## 4. 独立评审优先核查的 18 个问题

### A. 任务与方法

1. 该课题的任务定义是否仍然是有自然语言指令的空中 VLN，而非普通图像避障？
2. 当前学生在线输入中，是否存在通过隐藏状态泄漏真值 pose、depth、map 或目标坐标的风险？
3. 在单目 RGB 无里程计下，哪些目标完成状态本质上存在不可观测性？如何定义“不确定”而非幻觉？
4. 证据校准式进度/终止方案与 Self-Monitoring、LookasideVLN、LightVLN、QAD、AirAnchor 的实质差异是什么？哪些只是既有模块组合？
5. STOP 单独 head 在相同数据、容量、计算下是否优于仅 8-way softmax？如何剔除阈值调参污染？
6. 研究是否过分依赖稀有成功事件，是否需要连续指标、路径/场景聚类统计？

### B. 数据与物理闭环

7. BabyAI / MiniWorld 的语言任务、视角、动作语义能否支撑空中导航结论？最小合法用途是什么？
8. 程序化 SDF 几何如何成为具有碰撞的渲染物体，机载 RGB 的坐标系是否正确？
9. 官方 X3 电机 API 的 rad/s 与 CF-Drone 归一化混控之间如何标定？
10. 动力学控制周期、策略观测周期、RGB 帧延迟与 simulation time 如何统一？
11. 当视觉 student 只看 RGB 时，CF 内部 IMU 是不是合理且明确的职责分离？
12. 离散 AerialVLN benchmark 的 5m Forward/15° 转向，是否被错误应用到实际飞控？

### C. 自研与部署

13. `code/src` 24 个源文件、`sim/src` 6 个源文件上限是否给核心模型留足工程空间？限制是否造成反向的巨大单文件问题？
14. 自研专用训练器是否有独立数学 oracle、gradcheck、checkpoint 恢复和 CPU/CUDA parity？
15. TPU-MLIR 到 SG2002 的算子支持是否真实在开发板上编译运行？是否错误地用理论 TOPS 推导 FPS？
16. 未来 ESP32 修改应如何分为协议、定高、串口安全、测试而不是大规模重写？

### D. UI 和证据

17. 看板的 ready / available_unverified / blocked / not_implemented 是否足够防止将工具存在误判为功能 PASS？
18. HTTP 指标读取的并发、异常、日志裁剪、目录穿越、可访问性和页面误导风险是否可接受？

## 5. 请评审者输出的格式（不是要求只报好话）

建议另一位 AI 按以下表格输出：

| 字段 | 填写要求 |
|---|---|
| P0 阻断 | 若不解决则无法实现功能/无法形成有效论文结论的问题 |
| P1 风险 | 不阻断最小演示，但将显著影响安全、效果、资源或可信度的问题 |
| P2 建议 | 维护性、可视化、文档等优化 |
| 证据 | 必须指出源码路径、能复核的公共资料或可执行测试；不得虚构跑过的硬件 |
| 争议/反例 | 至少写出一个可能推翻评审者自身观点的实验 |
| 最小改动 | 可被分支实现、可被测试、可回滚的方案 |
| 放弃条件 | 哪种观察结果出现时删除该模块，不为了创新堆代码 |
| 改进后实验 | 明确 seeds、数据划分、学生合法输入、统计单元和核心指标 |

**评审者不应**把本项目中 Python 工具用于场景与可视化等同于“学生训练依赖 PyTorch”；也不能因为网页好看就认为飞行链路完成；更不能根据外部公开文献结论直接替代用户 CF-Drone 的实际源码或板卡实测。

## 6. 复现命令与结果解释

```bash
# 工程根目录
./run.sh doctor                  # 工具存在性检查，不是功能验收
./run.sh scene --preset forest --seed 42 --count 24
./run.sh ui --port 8765          # 只读 UI；浏览器打开 http://127.0.0.1:8765
python3 -m unittest discover -s code/tests -p 'test_*.py' -v
python3 -m unittest discover -s sim/tests -p 'test_*.py' -v
./run.sh sim doctor              # 缺 Gazebo 时预期非零退出
./run.sh train --run-id smoke1   # 缺训练器时预期 BINARY_NOT_BUILT，不应生成假成绩
```

### 状态列表

- PASS：只读 UI 和 JSONL 日志处理的本地单元测试；静态场景 deterministic；未实现功能门禁。
- BLOCKED：Gazebo 动力学、真实 RGB 采集、CF 控制器 SIL、C++ 学生训练、SG2002 实机、ESP32 新协议和实体飞行。
- NOT CLAIMED：AerialVLN-S unseen SR、任何 real-flight SR、蒸馏收益、INT8 精度、板端 FPS。

## 7. 本次 V7.1 代码变更边界

- 修改 `assets/dashboard.html`：视觉、五页、真实指标与评审导出。
- 修改 `tools/dashboard.py`：只读 API、证据等级、安全读取。
- 修改 `tools/launcher.py`：更名 V7.1、新增 `ui` 子命令。
- 新增 `code/tests/test_dashboard_v7_1.py`：8 个测试。
- 修改 `README.md`,`CODE_RULES.md`、主方案与 docs 文档。
- **无** `.ino`、`.cpp`、`.cu`、飞控协议变更；`CF-Drone.zip` 仍为独立原始附件。

## 8. 独立评审结论应该达到的最低质量

请输出明确的“保留 / 修改 / 删除”三分判定、每项代价与复现门禁，并把关于研究发表创新性、物理飞行安全性、UI 可信度的判断**分别评分**。不允许将“没找到先例”写成“我们是第一篇”，或在没连接硬件的情况下保证真实实时性。
