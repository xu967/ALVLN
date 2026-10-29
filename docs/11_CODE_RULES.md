# 11｜最终工程代码限制与模块管理（非模板占位 KPI）

## 1. 项目内唯一主入口

`./run.sh` 为当前服务器启动入口；`frontend/` 只有一份生产使用的 NavLab Studio UI v3；`legacy/V7_1_patched` 仅供审计和对照，不提供任何默认进程。**禁止把两个 API 端口和两套 Dashboard 同时当成“同一实验结果”。**

## 2. 工程区域强约束

| 目录 | 放什么 | 禁止什么 |
|---|---|---|
| `frontend/` | HTML/CSS/JS 与静态截图 | 修改真实评测器/训练器或伪造成功标志 |
| `server/` | 白名单注册、任务状态机、审计数据/日志 | 接收 URL 指定脚本/可执行路径、`shell=True`、越权 Kill 其他进程 |
| `tools/` | 环境检查、静态场景工具、数据 manifest/小脚本 | 生产模型训练或不受控执行外部命令 |
| `cpp/` | 自研 Tensor/Conv/GRU/LOSS/优化器、导出接口 | 默认引入 PyTorch 主训练、未检查的 GPU cast |
| `simulation/` | Gazebo C++ 桥、相机、控制/任务执行 | 从 student 读 hidden oracle、将 PX4 测试称 CF 实测 |
| `artifacts/` | 运行时输出（数据、日志、检查点） | 直接 git commit 大模型/历史已验证证据覆盖 |
| `references/` | 只读源代码快照、外部评审 | 默默复制后忽略 LICENSE/源 commit |

## 3. 文件数/规模与简化纪律

第一版视觉学生建议 `cpp/src` ≤24 个实现文件、`cpp/cuda` ≤6、公共头文件 ≤10；仿真初版 `simulation/src` ≤10。默认单业务 C++ 文件 ≤280 有效行、CUDA ≤360 行；超过前写下唯一责任、接口测试与理由，而不是为了压行数把控制器/数据加载/训练器写成一个巨大文件。**这些限制是预算，不是要求先建满 40 个空目录**。

同一功能只有一个主实现；C++ 的 Eigen/数学库、cuBLAS/cuDNN、OpenCV/相机 SDK、protobuf/ONNX 仅在确有需要时通过锁定版本引入。禁止重复开发成熟底层图形/物理/矩阵乘法实现。不应把上游 ROS1/ROS2 AirSim 所有依赖安装到主机全局环境。

## 4. 接口冻结顺序

必须先冻结 `ObservationSpec`（合法学生字段）→ `ActionProtocol`（任务单位与安全语义）→ `EpisodeSchema`（标签/拆分/评测归属）→ `EventSchema`（日志）→ `NPUModelIO`（dtype/scale/shape）。UI 文档的字段不等于真实训练器已写出的字段；变更要维护一个 schema 兼容性测试。

## 5. 每个 PR 的最小验收

1. 明确问题、输入合同、失败条件与范围；不要为“代码整洁”修改无意义文件。
2. 与主方法无关的改动单独提交；一次变更尽量只改一个子系统（安全修复除外）。
3. 增加至少 1 个能区分“改前失败/改后通过”的测试；执行 `./run.sh test`、`./run.sh cpp-test`。
4. 若碰到训练/仿真，必须有完整 stderr、原始数据 hash、可重放命令；失败不可被 UI 绿色按钮掩盖。
5. 修改实验结果时使用新 run_id，不能覆盖历史 run/原始 JSONL；审计 manifest 由文件哈希生成。
6. 严禁把模型输出动作 logits、测试级单步预测等同于真实连续飞行安全性。

## 6. 安全约束

服务端只监听 loopback，训练/仿真只接受受管理员注册的 profile/id；不允许任意命令/路径。任何临时公网访问须先实现完整认证和安全审查。CF-Drone `.ino` 的变更需要新分支与无桨 HIL，不能在 UI 上加“Arm Motors”按钮。

## 7. 禁止事项摘要

**禁止**虚构训练曲线/仿真图像/真实部署结果；禁止将 `*.planned` 当成可用模块；禁止复制没有许可证核验的高飞/CF/Flix 源码到发布主仓库；禁止把单帧评估指标标成官方闭环 SR；禁止使用 dashboard 反向判定真实飞行安全；禁止为了强行复现好看的数字在 test split 上调参。
