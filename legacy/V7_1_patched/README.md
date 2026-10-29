# TinyAerialVLN V7.1 — CF-Drone + Gazebo 研究工程（冻结固件阶段）

本 ZIP 是 **V7.1 基础设施与 UI 更新包**，不是已能真实飞行的完整项目。保持 V7 原有 Gazebo/CF-SIL/导航方法规划，新增可运行的本地只读研究看板、运行状态审计和测试。

## 一键进入工作台

在 Ubuntu 上解压并进入项目根目录：

```bash
chmod +x run.sh
./run.sh ui --port 8765
```

浏览器打开 **http://127.0.0.1:8765**。也可以使用旧入口 `./run.sh dashboard --port 8765`。不需要 Node、React、PyTorch、TensorBoard 或第三方 Web 运行库；只需 Python 3 标准库和浏览器。

主要页面：**项目总览 / 训练监控 / 场景与仿真 / 边缘部署 / 日志与评审**。完整设计见 `docs/V7_1_UI_SYSTEM_DESIGN.md`。

## 单命令开发诊断

```bash
./run.sh doctor
./run.sh sim doctor
./run.sh scene --preset forest --seed 42 --count 24
./run.sh train --run-id experiment_001    # 训练器尚未实现时应拒绝并报 BINARY_NOT_BUILT
```

注意：`./run.sh sim doctor` 检查 Gazebo 工具，不等于动力学 PASS；当前 ZIP 只有静态森林/走廊 SDF，**尚无可飞行的相机+四旋翼物理闭环**。

## 测试

```bash
python3 -m unittest discover -s code/tests -p 'test_*.py' -v
python3 -m unittest discover -s sim/tests -p 'test_*.py' -v
```

本次容器中分别 13 + 2 项测试通过；没有安装 `gz` 或 `nvcc`，没有真实 SG2002/ESP32 设备验收。

## 已实现 / 尚未实现

| 项目 | 状态 |
|---|---|
| 生成静态 SDF 场景、可复现随机种子 | 已实现并通过基础测试 |
| 只读 HTML 工作台、JSONL 指标读取、错误筛选、证据导出 | 已实现并通过单元/静态浏览器测试 |
| 训练器若写真实 JSONL 后自动展示 | 接口就绪，但无学生训练器可提供真实记录 |
| GZ0 四旋翼电机、RGB、物理碰撞闭环 | 未实现 |
| CF-Drone PID/mixer 软件在环 | 未实现 |
| 学生 C++/CUDA 训练、知识蒸馏 | 未实现 |
| SG2002 NPU 编译与板端通信 | 未验证 |
| ESP32 固件改动 | **本轮禁止，未来可单独评审后修改** |

## 评审阅读顺序

1. `docs/V7_1_INDEPENDENT_AI_REVIEW.md` — 对外独立评审包、风险与质疑问题。
2. `V7_1_RESEARCH_PLAN.md` — 完整 V7 研究规划及 V7.1 变更说明。
3. `docs/V7_1_UI_SYSTEM_DESIGN.md` — UI API、状态、测试及安全合同。
4. `CODE_RULES.md` — 模型/仿真源码上限、第三方许可、ESP32 冻结规定。
5. `tools/dashboard.py`, `assets/dashboard.html`, `code/tests/test_dashboard_v7_1.py` — 已实际实现的更改。

`CF-Drone.zip` 必须单独保存；为了避免许可混淆与误改，该附件没有放进此工程 ZIP。后续真正改固件必须新版本和专门硬件验收门禁。
