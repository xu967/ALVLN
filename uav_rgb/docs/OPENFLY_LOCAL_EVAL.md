# OpenFly 本地评测接入说明

本项目只把 OpenFly 当作外部语言导航基准和数据格式参考，不把 OpenFly 的城市尺度分数直接并入本项目的 3DGS/UAV 小步长排行榜。

## 当前结论

- 2026-11-16本轮：causal.v2组首监督已完成；Seq2Seq/CMA各RGB-only和RGB+语言×3seed，共12个50轮模型已真实训练。v2完成200回合及200视频审计，但seen长序列暴露10次残余下落偏差；已新增无惯性v3协议并独立通过百步门禁，五臂新协议100回合已完成且协议/程序/录像错误为0；12次碰撞仍保留，语言版成功率未超过纯视觉。历史v2不是正式成绩。
- 当前复现用 env18_causal_bc_v2 清单和 exp-openfly-optimization-20261116/training 权重。旧权重只配旧源码，不关闭hash验证；本轮v2/v3 evaluator及trainer精确源码快照保存在 source_snapshots/；更早v1须使用对应Git历史与历史产物。

- 2026-11-16 03:11 +0800：并行审核后的代码回归 408/408 通过（44 C++/CUDA、58 OpenFly/下载、306 3DGS/SITL），另实际重训两模式并验证21闭环/20视频。**闭环不是全通过：正确语言第0路线有1次位姿/时间戳同步失败，现有门禁正确终止。** 复放又出现物理已更新而渲染滞后、或位置更新但时间戳不增；不能继续把当前适配器当稳定正式评测。诊断与视频：`/root/rivermind-data/uav_rgb/exp-code-audit-20261116/index.html`。
- 2026-11-15 23:47 +0800：完成盲基线、独立纯视觉与RGB+语言三seed对照，六训练/52闭环/52视频。平均终点误差分别30.19/16.05/13.73m；仅四条内部开发路线，语言作用仍弱，且已确认组末观测与完整宏动作的时序差异。最新观看页面在数据盘 exp-openfly-comparison-20261115/index.html。

- 官方平台仓库提供 `train/eval.py` 和 `configs/eval_test.json` 的本地评测流程；没有发现官方远程提交接口或可直接上传模型的 evaluation server。
- 本地官方 evaluator 定义 0–9 十个动作 token：STOP、前进的三种距离、左右转、升降与左右平移；前进步长为3/6/9m，和本项目四个 primitive 的动力学不兼容。
- 2026-11-16 核验：DataGen 在00:12:25 +0800完成20/20资产的下载器大小/哈希验证；当前20个ZIP/PCD合计124,369,388,594B，无 `.part`。另有103,226条Annotation；并未下载完整轨迹图像或OpenFly-Agent权重。场景资产、训练样本和模型不是同一个下载；本轮未重新计算全部124GB哈希。
- 22:23 更新：六个 AirSim Linux 场景已解压/校验，另下载 16 条路线的 150 张官方原图，完成 RGB+语言的 5/50 轮离线 BC。它复用 AirVLN seq2seq、保留 OpenFly 十类标签，不是官方 7B 或本机四动作学生；训练集拟合改善，验证显示过拟合。
- 23:00 更新：完成真实 env18 图像反馈的位姿闭环，正确／错配／零语言各4集，12段H264录像全部回读通过。零语言4/4动作序列不变，说明当前模型尚未证明使用语言完成任务；不是物理飞行或官方排名。

参考：

- [OpenFly-Platform 官方仓库](https://github.com/SHAILAB-IPEC/OpenFly-Platform)
- [OpenFly 官方数据集](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly)
- [OpenFly 论文](https://arxiv.org/abs/2502.18041)

## 本地最小流程

1. 先只取得 `Annotation` 的小文件或官方已存在的 annotation 副本；不要把 `Image`、`traj` 全量拉到系统盘。
2. 用 `tools/audit_openfly_annotations.py` 统计指令、动作 token、轨迹长度和字段一致性：

   ```bash
   python tools/audit_openfly_annotations.py \
     --input /root/rivermind-data/uav_rgb/downloads/OpenFly/Annotation \
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
| OpenFly_DataGen | AirSim/UE场景归档与PCD地图等资产 | 2026-11-16 00:12完成20/20，124.37GB；下载器校验日志及当前文件数/字节核对一致，不是全量训练图像 |
| OpenFly数据集 | 语言、轨迹、动作与视觉样本，供训练和基准任务 | 已有Annotation103,226条及249张唯一原图（旧150+补99）；当前causal训练使用150张，16路线BC已完成，全库/RLDS未准备 |
| OpenFly-Agent | 官方参考视觉语言导航模型，有独立预训练权重 | 未在本机加载或完成推理门禁；也可接自有策略，需遵循相同评测接口 |

例如任务指令可以是“沿道路前进，过路口后左转，在指定建筑前停下”。这是任务示例，不是本轮实际语言策略输出。要验证VLN，策略必须用指令与相机观测决定动作和停止；直接读取标注目标/参考轨迹、或只运行控制器，不构成语言理解证据。

官方资料核查：OpenFly-Platform README的Introduction/Simulation/Training/Evaluation，以及[论文](https://arxiv.org/abs/2502.18041)、[DataGen资产页](https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly_DataGen)。截至本次官方README仍将3DGS资产下载标为Coming soon；本地DataGen清单仅AirSim/UE/PCD，不能认为下载它已得到可替换Wanderland的全部3DGS场景。

## 本轮实测与开始正式评测的条件

- 已完成env18单场景：3次reset重复、9个短物理动作、0碰撞、RGB/depth正常；前三个官方参考位姿经pause/set/20ms tick后误差2.41–9.61mm。详细失败对照/命令见现有[实验文档](experiments/2026-11-14-v3-baseline-long-route-followup.md)21:40更新。
- 在线eval用环境实时取图，**不必等TB级历史图像库**；但必须接入可用的受训策略/模型、相应processor/statistics、协议适配与验证。数据资产完成不会自动产生成功率。
- 2026-11-15历史：快照存在annotation负token、xyz/xyzyaw及保留组末时序问题，当时只做同索引离线BC。2026-11-16已另建causal.v2并验证组首原语/宏转移；未覆盖的动作原语仍拒收，不能据此转为物理飞行或正式成绩。
- 官方城市尺度的动作、20m判定半径与本机小型机SITL不同；结果分别报告，不混排名。不暂停的瞬移还会触发物理漂移，应固定取图时序后再做整条任务。
- 六个Linux AirSim场景统一在 `/root/rivermind-data/uav_rgb/openfly-runtime-ubuntu/envs/airsim/`；env18软链接复用旧目录，其他五个新增约7.99GB。2026-11-15 22:23历史数据盘余约14.77GB、Windows包未完；2026-11-16 00:12全部DataGen已完成。当前可用空间以df为准，不全量解压多个UE版本。

## 可复现的小子集训练

`tools/train_openfly_subset.py` 复用本机已安装的 AirVLN `VLNCETrainer`，无需加载OpenFly7B；前置条件为既有airvln环境、ResNet50缓存、词表/深度构造权重和真实图像manifest。以下输出路径必须不存在，工具拒绝覆盖已有实验。

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache OMP_NUM_THREADS=4 \
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python tools/train_openfly_subset.py \
  --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_causal_bc_v2/manifest.json \
  --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN \
  --out /root/rivermind-data/uav_rgb/runs/openfly-env18-reproduction \
  --policy-type seq2seq --training-mode rgb_language --epochs 50 --seed 20261115
```

输入只含instruction和冻结ResNet的RGB特征，RNN逐完整轨迹reset，无未来帧、位置、终点、progress或真值前动作输入。标注、图像、词表、实现代码的hash写入checkpoint；保留原0..9分类，STOP只来自原始末尾0，不改成本机四动作。训练集12路线116帧，内部验证4路线34帧，全部来自官方train，非官方seen/unseen。

历史v1（非上方新causal命令）的50轮train正确102/116、STOP12/12；内部val正确19/34、STOP2/4，CE2.13148比5轮1.42110更差。不能只报告训练准确率。完整命令、混淆矩阵、语言消融、权重及限制见[实验记录](experiments/2026-11-14-v3-baseline-long-route-followup.md)22:23更新。

## 闭环与录像开关（2026-11-15 23:00）

入口 `tools/eval_openfly_subset.py` 使用当前渲染RGB和指令逐步决策，保留RNN hidden；起始位姿仅用于reset，终点仅用于评估与视频字幕。它使用OpenFly十类位姿动作，不读取下一标注位置，也不向策略传目标坐标。

2026-11-15历史观看入口：`/root/rivermind-data/uav_rgb/exp-openfly-closed-loop-20261115/index.html`，用服务器桌面浏览器打开即可对比12段视频。`correct_v2/episode_000..003/decisions.mp4` 是正确指令组，`swap_v2`、`zero_v2` 是对照；播放器有暂停、进度条和0.5倍速按钮。原始首末图、带字幕预览、trace和结果同目录。页面使用本地相对文件，不需要启动公网服务。

历史env18启动命令、cwd、非root运行用户与853×480/FOV90/Gamma1.5/相机Z=1设置保存在历史目录 `launch.json`、`settings.json`。RPC为41458，车辆drone_1，相机front_custom；先确认端口未被其他实验使用。该历史轮次结束已关闭其仿真进程；当前轮次状态以最新worklog为准，当前启动证据在 exp-openfly-optimization-20261116/launch_simulator.json 与 settings.json。

```bash
PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache OMP_NUM_THREADS=4 \
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python -u tools/eval_openfly_subset.py \
  --checkpoint /root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/training/seq2seq_rgb_language_20261115/model.pt \
  --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_causal_bc_v2/manifest.json \
  --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN \
  --out /root/rivermind-data/uav_rgb/runs/openfly-closed-loop-new \
  --execution-protocol openfly_eval_kinematic_step_v3 --port 41458 --limit 4 --max-steps 30 --language correct --record-video
```

- OpenFly录像默认开启；显式 `--record-video` 开启，替换成 `--no-record-video` 关闭。关闭后仍保留轻量首末PNG、trace和结果，已用真实回合验证。
- `--language swap` 循环换成另一验证路线的指令；`zero` 将语言编码输出清零。每次使用新输出目录，拒绝覆盖旧结果。
- `sim.3dgs.run_vln` 的既有开关是 `--videos N`（每场景前N集），`--videos 0` 关闭；Studio的“训练后保存视频（闭环1集）”仍只对应原C++学生训练入口，不能当作OpenFly开关。
- H264视频每观测一帧、2fps，没有平滑插帧；这是位姿跳转的决策录像，不能用画面跳动判断真实机体或PID。录像异常独立记录 `video_error`，不丢已完成导航结果；OpenFly门禁/录像失败使命令非零退出。
- 本轮3组均3/4停在20m内，但清空语言4/4动作完全不变、换语言3/4不变；该半径指标仅诊断，不是语言理解证据。详见实验记录23:00条目及计划§10.13。

## 三种策略的公平对照

仍使用上述训练和闭环入口，不增加启动脚本。`--training-mode rgb_language` 为默认模式；`--training-mode rgb_only` 从初始化到训练、验证、闭环都关闭语言分支，保留相同网络布局。它与训练好语言模型后使用 `--language zero` 的推理消融是两种不同实验。闭环自动读取 checkpoint 的训练模式。

本轮预先固定 `20261115/20261116/20261117` 三个 seed，每个模式训练50轮、600次更新，选择最终权重，不按验证表现选择某轮。复现实验分别替换原训练命令的 `--training-mode`、`--seed`、`--out`；每次输出目录必须不存在。初始化 state hash 和50轮样本顺序 hash 应在同 seed 的两种模式之间一致。关闭语言后，实际获得梯度的参数会减少，不能只用总参数数宣称有效容量完全相同。

盲前进基线不加载模型，不消费RGB或指令：只从12条训练路线统计前进动作众数和非STOP步数中位数，固定前进8次（每次9米）再STOP；若先触发碰撞或预算则提前终止。没有使用验证终点或验证路线长度设定停止时刻。与学习模型共用相机、位姿动作、反馈门禁及录像。

```bash
PYTHONDONTWRITEBYTECODE=1 \
  /root/rivermind-data/uav_rgb/envs/airvln/bin/python -u tools/eval_openfly_subset.py \
  --policy blind_forward \
  --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_causal_bc_v2/manifest.json \
  --out /root/rivermind-data/uav_rgb/runs/openfly-blind-new \
  --port 41458 --limit 4 --max-steps 30 --max-distance 300 --timeout 120 --record-video
```

盲基线无需 `--checkpoint` 或 `--airvln-root`，也不适用 `--language swap/zero`。学习策略沿用默认 `--policy student`；分别评估正确指令、交换指令和语言置零。视频字幕与 trace 标明策略、训练模式及停止原因，固定停止记录为 `fixed_stop`，模型自主停止记录为 `predicted_stop`。

所有权重仍严格校验训练源码、数据和词表hash。历史03:11版本只加强数据几何校验，仍改变了源码hash；当前示例已经更新为causal.v2及Seq2Seq/CMA新代码对应权重。两模式同seed重训各600更新后与旧模型499个张量逐项完全一致，但历史权重仍须其历史源码，不能据数值相同绕过hash门禁。

本轮新增运行边界：预算必须为有限正数；非有限模型输出/hidden/位姿与非法首帧时间戳拒绝。推理或编码跨时限后不再发送移动；期限内已决定STOP时，最终编码耗时不会撤销STOP。视频字幕 `predicted` 表示模型候选动作，`action_requested` 是通过门禁的请求，不是物理完成确认。`executed_actions`/`commanded_distance_m` 统计返回的命令次数/目标位移，不应替代实际反馈路程。RPC/运行异常写 `summary.run_error`，录像初始化失败仍保留导航trace/result并非零退出，此时可能没有PNG/MP4。

**2026-11-15历史动作标签边界（新causal.v2已单独解决所用子集）：** 本轮16份原始 `pose.jsonl` 的322次相邻转移均符合动作前置观测；连续前进压缩为1/8/9后，150条公布索引均保留动作组末端的原始观测。99个8/9观测比组首前进3或6米。从这些保留位姿直接执行完整宏动作，并不等价于原始动作重放。该历史对照冻结原manifest，不改步长或标签来改善成绩；本轮新manifest改为经raw验证的组首观测，旧证据保留。尚未找到官方压缩生成代码，不能据此宣布官方数据集有bug。后续因果观测版本应使用逐组校验的组首图像与起点、独立manifest，再重新训练全部基线。


## 当前运动学协议与官方seen子集

三种执行协议显式区分：v1为setPose+time(.02)；v2追加固定frames(2)；默认v3为一次simSetKinematics原子设置位姿并清零速度/加速度，再使用同样固定time/frames。短tick内物理仍运行，没有关闭重力或清空飞控内部状态。v2长序列保留物理速度会下落，v3只用于无惯性导航决策评测，不是飞控改善或正式官方协议。原反馈门槛0.2m/1°/递增timestamp、碰撞优先、无重试都不变。切换版本必须完整重跑同一比较组，不能混表。

CMA使用同一训练命令的 --policy-type cma，--training-mode rgb_only 为独立纯视觉训练。在线评测从checkpoint读取架构与模式，空间特征形状、前动作禁用契约和来源hash均校验，不手工将Seq2Seq权重当CMA加载。模型只有RGB/语言/过去hidden，无评估目标输入。

官方seen任务另用 --evaluation-tasks，不能塞入训练manifest假装开发集。下列参数追加到既有评估命令，仍使用新输出目录：

    --evaluation-tasks /root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/official_seen_tasks.json
    --execution-protocol openfly_eval_kinematic_step_v3 --limit 20 --max-steps 100 --max-distance 300 --timeout 120

完整可执行argv见 evaluation_v3_execution.json；原始官方起点和终点与训练causal组首分别处理。GPU UUID按运行时nvidia-smi确认，不能沿用过期编号。五模型臂沿用固定seed20261115，不从三seed里按seen成绩选最好一个。

结果必须报告总分母、STOP且距目标<20m、有效观测OSR、可信终帧NE及其n/N、碰撞、预算耗尽、协议/程序/视频错误。碰撞帧可能同时偏离命令位姿，不能把条件NE当完整均值排名。官方作者SR不要求STOP，旧SPL是直线距离/命令路径，和我们口径不同；没有正式全seen/unseen或远程榜单提交。

本轮视频和证据根目录：/root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/。新协议视频在 evaluation_v3/official_seen/<策略>/episode_000..019/decisions.mp4；历史v2在 evaluation/development 和 evaluation/official_seen，禁止覆盖。index.html提供分协议/路线/策略的视频查看入口。

可直接复制的完整示例（需已启动本机env18/41458；若目标目录已存在，请改为新目录）：

~~~bash
PYTHONDONTWRITEBYTECODE=1 TORCH_HOME=/root/rivermind-data/uav_rgb/torch-cache OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=GPU-e9d01956-88db-b317-ea53-be3fb6cf7190 /root/rivermind-data/uav_rgb/envs/airvln/bin/python tools/eval_openfly_subset.py --manifest /root/rivermind-data/uav_rgb/downloads/OpenFly/subsets/env18_causal_bc_v2/manifest.json --port 41458 --execution-protocol openfly_eval_kinematic_step_v3 --evaluation-tasks /root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/official_seen_tasks.json --limit 20 --max-steps 100 --max-distance 300 --timeout 120 --record-video --out /root/rivermind-data/uav_rgb/runs/openfly-seen-v3-reproduction-new --checkpoint /root/rivermind-data/uav_rgb/exp-openfly-optimization-20261116/training/seq2seq_rgb_language_20261115/model.pt --airvln-root /root/rivermind-data/uav_rgb/airvln-runtime-ubuntu/AirVLN
~~~


最新完成状态：12训练、历史v2 200回合、新v3 100回合全部完成，300视频/6690帧已独立审核。v3同20任务的STOP<20：盲1/20、Seq2Seq纯视觉5/20/语言4/20、CMA纯视觉5/20/语言3/20；这不是官方SR或榜单排名。原协议的10次高度偏差失败保留，不能混入v3平均。结果与下一步见现有实验报告末节及PLAN §10.17。

当前轮次已经收尾：本任务env18进程已关闭、41458已释放；若复现，按本轮launch_simulator.json和settings.json重新启动非root环境，不要假定服务仍运行。
