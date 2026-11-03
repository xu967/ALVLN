# campus.v1 可信基线实验记录

- 日期：2026-11-03（Asia/Shanghai）
- 仓库：`/root/claude-code-backup-20261028/code/uav_rgb`
- 数据根目录：`/root/rivermind-data/uav_rgb`
- 目的：验证校园三拓扑 pilot 的采集、审计、训练、学生闭环和工程检查链路。
- 结论：工程链路通过；学生小样本闭环已完成但测试成功率为 0/1，不能作为性能结论。

## 环境

```text
GPU: NVIDIA GeForce RTX 2080 Ti
CUDA: 12.8
Gazebo: 8.15.0
Python: .venv-tools, numpy 2.5.4, Pillow 13.0.0
渲染：此前实测 GL 为 llvmpipe（软件渲染，Accelerated:no）
```

GPU 可用于 C++/CUDA 训练和测试；Gazebo 的图形加速仍按软件渲染预算记录，不能把两者混为一谈。

## 采集与审计

真实 Gazebo 校园冒烟采集保存在：
`/root/rivermind-data/uav_rgb/campus-implementation-20261103/`。

- 6 个 episode、138 个步骤；building_lane、tree_road、plaza 三族均覆盖。
- 6/6 无碰撞进入 `visible_goal_reach_v1` 目标区。
- 数据审计 PASS；教师标签复算不一致为 `0/138`。
- train/val/test 三个 split 均有四类动作样本。
- 配对数据动作计数：SAFE=18、FORWARD=102、LEFT=9、RIGHT=9；每个 split 均覆盖四类。
- 采集保留 world、帧、轨迹和身份字段，目标半径为 0.5 m。

真实采集入口为 `sim/tools/smoke_campus.py --out <新目录>`，默认 seed 为
`0,4,8,12,16,20`（实际结果分在 `pilot-1` 与 `pilot-right` 两个目录，合并后通过
`audit-paired.json` 验收）。较早的仅左向 3 集试跑曾因 RIGHT 低于 5% 被审计拒绝；补齐右向配对任务后，6 集审计通过。这组结果只证明采集和审计链路可运行，没有证明学生策略泛化。

## 训练 smoke

使用 campus 数据进行了小型训练 smoke：48 个 train 样本、1 epoch，成功生成
`train-smoke/ckpt_best.bin`。该权重仅用于检查训练器到评测器的接口，不用于报告精度。
训练器记录的离线数值为：val `acc=0.636`、`macro_f1=0.197`、
`balanced_accuracy=0.219`；test `acc=0.739`、`macro_f1=0.213`、
`balanced_accuracy=0.250`。样本少且只训练 1 epoch，这些数值只作链路诊断。

## 学生闭环 smoke

命令输出日志：
`/root/rivermind-data/uav_rgb/campus-eval-smoke-fixed-20261103.log`。

对应结果文件：
`/root/rivermind-data/uav_rgb/campus-implementation-20261103/eval-smoke-fixed/eval.json`。
入口为 `sim/tools/eval_closed_loop.py --task campus --policy student --ckpt <smoke 权重> --split test --seeds 16:24 --keep-work --out <新目录>`。

真实输出摘要：

```text
episodes=1/1  SR=0.0  SR_stop=0.0  collision=0.0  stuck=0.0  false_forward=0.0
inference p50=7.08843ms p95=7.62194ms (warm n=2; cold first=[23.341])
seed 16: steps=3 collision=False success=False max_x=0.178617521376982
```

该 episode 在 `campus.v1/plaza` 测试地图完成采集，但只执行了 3 个 `FORWARD_SLOW`，最终距 waypoint 约 4.07 m，未进入 0.5 m 目标区。失败已计入请求集比例；不删除、不重试后挑选成功样本。

## 全量工程检查

执行：

```bash
source ./env.sh && bash tools/check.sh
```

结果：`PASS`，25/25 CTest 通过，Python 自检通过；GUI 冒烟因无 `DISPLAY` 跳过。检查脚本明确说明它不替代真实采集、训练或学生闭环，因此上面的独立结果仍需单独阅读。

## 解释和边界

- 当前是定高、四动作、单帧 RGB + 文本分支的校园 pilot；不是完整三维飞行或真机验证。
- 三个拓扑族足以形成最小留出，但不能支持充分的校园泛化结论。
- 未进行长训练、多 seed 学生评测、风场/相机畸变/曝光标定、控制延迟拟合或真实飞行。
- 当前不需要用户提供照片。若未来要贴合特定园区，应采集有重叠、多角度且带尺度/相机标定的影像，并单独检查碰撞几何；几张孤立照片或 3DGS 不能直接形成可信飞行空间。
