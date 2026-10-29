# TinyAerialVLN / NavLab Studio Final V8 — 服务器交接入口

**日期：2026-10-28；交付类型：可启动的开发与研究工程骨架 + 已有前端 UI 源码 + 历史源码快照 + 全套实施文档和第三方审计材料。**

> 请不要把本工程理解为“已经训练好、已在真实无人机自主飞行”的交付。本次已实际测试：HTTP UI/API、任务注册门禁、静态 SDF+语义候选场景生成、日志审查、基础 C++ 数值梯度检查。**尚未实际测试：Gazebo 四旋翼物理闭环、真实 LANI/UE/AirSim 自动启动、自研 C++/CUDA CNN/GRU 完整训练、SG2002 的 cvimodel 运行、ESP32 飞控联调与真实飞行。**

## 1. Ubuntu 服务器解压后直接运行

```bash
unzip TinyAerialVLN_Final_V8_FULL_20261028.zip
cd TinyAerialVLN_Final_V8
chmod +x run.sh
./scripts/setup_ubuntu24.sh                 # 默认 dry-run；仅 Ubuntu 24.04
# ./scripts/setup_ubuntu24.sh --apply --with-gazebo   # 自愿安装基础工具和 Harmonic；不修改 GPU 驱动
./run.sh doctor             # 查看真实环境状态；FOUND != VERIFIED
./run.sh test               # Python 标准库测试
./run.sh cpp-test           # C++ FP64 前向/反向数值探针
./run.sh scene --seed 42 --difficulty S1
./run.sh ui                 # 本地 UI + API，默认 127.0.0.1:8765
```

服务器在远程时，在自己的电脑执行：

```bash
ssh -L 8765:127.0.0.1:8765 <username>@<server-ip>
```

浏览器访问 **http://127.0.0.1:8765**。默认不监听外网，不要直接开放训练/进程管理服务端口。当前 API 无面向公网的认证机制；如果需要多人访问，必须额外引入鉴权反向代理、HTTPS、用户隔离与资源限额，再由安全审查通过。

## 2. 首次进入 UI 会看到什么？

- 原始 NavLab UI v3 的训练、推理、闭环评测、实验历史、AI 审核页面均保留。
- 当同源 API 返回成功，页面切换为“REAL RUN/暂无运行记录”，训练曲线为空；若离线打开静态页面，初始数据明确标注 DEMO。
- 未注册真实训练可执行程序，点击启动会返回 `PROFILE_NOT_REGISTERED`，不会创建假的 loss 数据。
- 未注册且验证过的仿真控制程序，`simulation/capabilities.ready=false`，不会将 replay 冒充闭环 SR。
- 所有服务器实际启动的任务保留生命周期、stdout、stderr、事件及 AI 证据 ZIP；进程管理要求人工在 `config/registry.json` 中登记可信受控执行项。

首次传输后可运行 `./scripts/verify_package.sh` 以 SHA256 核验源文件，并执行本地 Python/C++ 测试；这不是 Gazebo/NPU/ESP32 验收。

## 3. 文件夹索引

| 目录 | 用途 | 当前真实状态 |
|---|---|---|
| `frontend/` | **唯一主 UI**：NavLab Studio v3，静态 HTML/CSS/JS、原始设计预览 | 可运行前端；部分后端功能待实现 |
| `server/` | 同源资源 API、审计 ZIP、受限外部任务启动器 | 可运行已测；未注册任务自动阻止 |
| `tools/` | doctor、Gazebo 探针、2.5D 语义场景发生器 | 可运行；不含四旋翼物理 |
| `cpp/` | C++17 基本数值算子/有限差分测试；未来放 CUDA 模型 | 数值测试通过，导航学生未实现 |
| `simulation/` | 计划中的 Gazebo ROS-free/ PX4 SITL 适配位置 | 只有计划文档，**不是**物理仿真成品 |
| `config/` | `registry.json`、受控外部执行配置示例 | 初始注册空，安全阻止未验证执行 |
| `docs/` | 研究、数据、算法、Gazebo、CUDA、部署、安全与 API 合同 | 详细设计 / 不等于功能实现 |
| `artifacts/` | 实验/仿真/场景/门禁证据，仅运行后才生成 | 默认无训练成果 |
| `legacy/V7_1_patched/` | 原 V7.1 已应用外部评审补丁的旧工程；供对照 | **非**主服务器，**非**主 UI |
| `references/snapshots/` | 原始上游/旧工程源码 ZIP（未经改动） | 只作资料，不默认编译，不默认运行 |
| `references/reviews/` | 独立评审报告、源码审计、补丁及复现脚本 | 原始证据，附历史结论 |

## 4. 首次开发顺序

**P0（运行环境）**：doctor → Python/API 测试 → C++ 数值测试 → GPU 驱动和 Gazebo Harmonic → PX4 `gz_x500` 能实际起飞并有前视 RGB 图 → 做出相机/姿态/控制统一的真实 trace。

**P0（外部基准 1–2 天探针）**：使用已授权的 AerialVLN-S 场景资源，在服务器记录 AirSim 实际启动能力。成功则优先复现 RGB-only 离散导航，失败记录证据后使用 Gazebo 小场景轨做算法闭环。不得无限期卡在不兼容 UE4。

**P1（数据/方法）**：场景-地标-语言-可行路径的离线合同、训练专家/学生输入隔离、最小 CNN+语言+GRU、动作/STOP 基线、ECPT 实现/消融。

**P2（端侧）**：先板端随机权重/INT8 算子编译与相机采集，才进入训练模型导出。ESP32 固件本期**冻结**；未来应根据实际芯片、接口、姿态估计和安全审查单独修改。

## 5. 明确没有包含的外部依赖

Gazebo/PX4/UE4 官方场景资源、NVIDIA CUDA/cuDNN、SG2002 TPU-MLIR/SDK、LANI 的 Unity/原始资产、论文数据和第三方权重不会由本包自动下载。`references/` 是用户上传的原始代码归档，**不是**可直接获得仿真资产的凭据；不保证第三方软件许可或版本兼容。

## 6. 更详细的入口

- 核心决策与任务边界：`docs/00_FINAL_DECISIONS.md`
- Ubuntu/5090/Gazebo 安装：`docs/01_UBUNTU_5090_SETUP.md`
- 场景、数据集、反捷径测试：`docs/02_DATA_AND_SCENE_PROTOCOL.md`
- Gazebo 实时闭环：`docs/03_GAZEBO_PHYSICS_CLOSED_LOOP.md`
- C++/CUDA 自研训练框架：`docs/04_CPP_CUDA_IMPLEMENTATION.md`
- ECPT 方法数学规格：`docs/05_ECPT_SPEC.md`
- 自动仿真服务端设计：`docs/06_SIMULATION_ORCHESTRATION.md`
- UI 审核/指标接口：`docs/07_API_LOGGING_CONTRACT.md`
- CF-Drone 固件风险与修改边界：`docs/08_CF_DRONE_SAFETY.md`
- SG2002 端侧路径：`docs/09_SG2002_DEPLOYMENT.md`
- 里程碑与实验验收：`docs/10_IMPLEMENTATION_GATES.md`
- 代码规则：`docs/11_CODE_RULES.md`
- 外部 AI 审查：`docs/12_AI_REVIEW_HANDOFF.md`
- 许可与资料追踪：`docs/13_PROVENANCE_LICENSE.md`
- 补充的细节：`docs/14_UI_BACKEND_OPERATION.md`

> **交付声明**：UI、API、场景工具、数值探针通过此环境的自动测试，不等于对远端 Ubuntu 5090 或 Gazebo/ESP32 的实测。任何别的 AI 审查必须区分 `source_observed`、`locally_tested`、`externally_documented`、`planned`、`unverified`。
