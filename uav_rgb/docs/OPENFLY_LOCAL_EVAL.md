# OpenFly 本地评测接入说明

本项目只把 OpenFly 当作外部语言导航基准和数据格式参考，不把 OpenFly 的城市尺度分数直接并入本项目的 3DGS/UAV 小步长排行榜。

## 当前结论

- 官方平台仓库提供 `train/eval.py` 和 `configs/eval_test.json` 的本地评测流程；没有发现官方远程提交接口或可直接上传模型的 evaluation server。
- 本地官方 evaluator 定义 0–9 十个动作 token：STOP、前进的三种距离、左右转、升降与左右平移；前进步长为3/6/9m，和本项目四个 primitive 的动力学不兼容。
- 2026-11-15 已下载的是 DataGen 场景/点云资产（清单124.37GB，最后Windows包仍续传）和103,226条Annotation；并未下载完整轨迹图像或OpenFly-Agent权重。场景资产、训练样本和模型不是同一个下载。

参考：

- [OpenFly-Platform 官方仓库](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
- [OpenFly 官方数据集](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly)
- [OpenFly 论文](https://arxiv.org/abs/2502.18041)

## 本地最小流程

1. 先只取得 `Annotation` 的小文件或官方已存在的 annotation 副本；不要把 `Image`、`traj` 全量拉到系统盘。
2. 用 `tools/audit_openfly_annotations.py` 统计指令、动作 token、轨迹长度和字段一致性：

   ```bash
   python tools/audit_openfly_annotations.py \
     --input /root/rivermind-data/uav_rgb/openfly/Annotation \
     --output /root/rivermind-data/uav_rgb/tmp/openfly-annotation-audit.json
   ```

   工具只读输入；输出目录由 `--output` 指定，建议始终使用 `/root/rivermind-data` 或 `/root/rivermind-fs`。
3. 运行官方 `train/eval.py` 前，先核对代码版本、成功半径、STOP、碰撞与 SPL 定义是否和目标论文协议一致，再准备对应的 ROS2/AirSim/UE 或 3DGS bridge 环境。官方脚本能运行不等于自动复现论文口径；这组结果单独报告，不能和本项目小型机动力学混排。
4. 若只做迁移基线，先记录动作翻译规则和 `cross_domain=true`，再把 OpenFly grounder 的指令理解输出送入本项目的局部控制器；迁移结果只能标成附加对照。

## 结果解释

审计报告是数据质量报告，字段 `records_with_action_and_pos` 只表示最小闭环字段存在，不表示视觉可用或任务成功。正式实验仍需固定 scene split、初始位姿、相机、时间预算、碰撞规则和随机种子，并同时报告 NE、SR、OSR、SPL、collision rate、latency 和显存。


## OpenFly到底是什么（2026-11-15核对）

OpenFly是户外无人机视觉语言导航研究体系，包含本地仿真/数据生成工具、数据集、参考策略和评测代码；评测是其中一部分。官方论文当前版本概括为100k轨迹、18场景，实际已下载Annotation计103,226条，论文概数与本地记录数分开报告。

| 名称 | 内容 | 本项目当前状态 |
|---|---|---|
| OpenFly-Platform | 多渲染环境接入、点云/语义/轨迹/指令生成、训练与本地eval代码 | 有官方快照，本轮env18真实RPC/渲染/动作smoke成功 |
| OpenFly_DataGen | AirSim/UE场景归档与PCD地图等资产 | 20资产124.37GB，21:36为19/20；不包含本轮所需模型权重，也不是全量训练图像 |
| OpenFly数据集 | 语言、轨迹、动作与视觉样本，供训练和基准任务 | 已有Annotation103,226条；图像/RLDS训练数据未全量准备 |
| OpenFly-Agent | 官方参考视觉语言导航模型，有独立预训练权重 | 未在本机加载或完成推理门禁；也可接自有策略，需遵循相同评测接口 |

例如任务指令可以是“沿道路前进，过路口后左转，在指定建筑前停下”。这是任务示例，不是本轮实际语言策略输出。要验证VLN，策略必须用指令与相机观测决定动作和停止；直接读取标注目标/参考轨迹、或只运行控制器，不构成语言理解证据。

官方资料核查：OpenFly-Platform README的Introduction/Simulation/Training/Evaluation，以及[论文](https://arxiv.org/abs/2502.18041)、[DataGen资产页](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen)。截至本次官方README仍将3DGS资产下载标为Coming soon；本地DataGen清单仅AirSim/UE/PCD，不能认为下载它已得到可替换Wanderland的全部3DGS场景。

## 本轮实测与开始正式评测的条件

- 已完成env18单场景：3次reset重复、9个短物理动作、0碰撞、RGB/depth正常；前三个官方参考位姿经pause/set/20ms tick后误差2.41–9.61mm。详细失败对照/命令见现有[实验文档](experiments/2026-11-14-v3-baseline-long-route-followup.md)21:40更新。
- 在线eval用环境实时取图，**不必等TB级历史图像库**；但必须接入可用的受训策略/模型、相应processor/statistics、协议适配与验证。数据资产完成不会自动产生成功率。
- 本地快照存在annotation −1/−2 token与eval0..9解码差异、xyz/xyzyaw混用及动作位姿不完全对齐；先明确转换，不直接开训练。
- 官方城市尺度的动作、20m判定半径与本机小型机SITL不同；结果分别报告，不混排名。不暂停的瞬移还会触发物理漂移，应固定取图时序后再做整条任务。
- 当前只保留最小env18解压目录，数据盘21:36余约23.6GB；不全量解压多个UE版本，不下载没有容量预算的大模型/图像库。
