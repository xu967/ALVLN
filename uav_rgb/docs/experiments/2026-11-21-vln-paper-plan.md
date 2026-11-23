# 室内无人机视觉语言导航论文实验计划

2026-11-21。开发阶段使用 SAGE-3D 室内场景、本仓 3DGS 渲染和 Crazyflie 固件在环仿真。实验入口统一使用 `train.py`，已有批量预设用于复现历史诊断。正式任务清单和对照组在开发阶段验收后登记。

## 研究问题

**同一个起点，换一句指令，无人机能否通过指定门洞并在正确实例前自主停止？** 拟验证的方法是用当前 RGB 中可定位的区域支持阶段切换，再用连续帧中同一目标实例的证据确认停靠。门洞通过、实例匹配与 STOP 分别评分。测试应包含目标暂时不可见、同名异实例、误入另一房间和近墙无目标。若两个目标都可到达，互换指令后应分别到达各自的目标。

## 现有实现与候选贡献边界

现有 G3 只输出粗略方位、阶段模式、`near_goal` 和目标词。VLM worker 每次只接收最新 RGB 帧，`VlmBeliefTracker` 用重复的 `reason` 文本确认目标，再结合两次 `near_goal` 判定触发阶段或 STOP。代码没有区域框或像素级定位、目标实例 ID、跨帧视觉匹配，也没有独立几何接近验证。11 月 21 日冻结帧中，近目标标志仍有误判。因而“区域/实例证据”是待实现和验证的研究假设，不能描述为当前已有能力。

若实现并通过实验，候选主张限定为：在控制权重、VLM、推理预算和起点/朝向固定的同起点异目标任务中，显式区域与目标实例证据门控是否能提升正确自主停止率。阶段门控和实例门控分别做单因素消融，错目标停止、阶段顺序和碰撞作为安全结果。新颖性仍需与公开方法逐项对照后才能成立。阶段 1 的局控门槛通过前，不实现正式对照，也不训练新权重。

第一版比较现有 G3 的文本输出与带图像区域证据的阶段确认，保持视觉语言模型、局控权重和推理预算一致。下一组只去掉阶段确认，另一组只去掉目标实例确认。目标区域证据在开发集校准，近墙的全图中央深度只用于局部净空判断。C++ 双帧学生用于动作控制，三带列池化单独衡量参数、延迟与闭环代价。上述方法和消融是待验证假设。

分层语义规划、VLM 给航点、单目深度配合 VLM、3DGS 加低层控制均有公开先例。本文候选贡献限于可定位的阶段与实例证据，以及同起点异目标实验对语言作用的验证。至少在同任务预算下实现一个公开方法的可比适配，预先登记输入和动作接口差异，再讨论创新与收益。 [CityNavAgent](https://aclanthology.org/2025.acl-long.1511/) [See, Point, Fly](https://proceedings.mlr.press/v305/hu25e.html) [VLM-Nav](https://doi.org/10.1371/journal.pone.0345778) [GRaD-Nav++](https://arxiv.org/abs/2506.14009) [SAGE-3D](https://sage-3d.github.io/)

## 已有证据

- `exp-vln-3dgs-g3-1117-1718` 与 `exp-vln-3dgs-g3-1117-1817` 使用同一个种子 `20261117`。两次 R0 与 DAgger 的数据有效负载和对应最佳模型参数逐元素相同，总墙钟 6463.95 秒。室内直线方位 0/10 成功，教师 9/10。学生 1113 步中仅 25 步前进，集中在同一条最终碰撞的路线。黑屏和方位置空臂各 1171 步，均为右转。 [运行一](/root/rivermind-data/uav_rgb/exp-vln-3dgs-g3-1117-1718/summary.json) [运行二](/root/rivermind-data/uav_rgb/exp-vln-3dgs-g3-1117-1817/summary.json)
- `840865` 正确与错配指令均 0/4 自主成功、2/4 曾进入目标圈。第一对目标中心距离 0.906 m，两圈半径各 1 m，配对预算不一致。`841980` 为 0/5。当前六个本地 SAGE 场景里的旧多阶段路线缺少 `sage.multi_stage.v2` 合同，当前入口枚举的有效多阶段任务为 0。两场景用作开发回归。 [任务枚举](../../sim/3dgs/vln.py) [历史任务构造](../../train.py)
- 现有 R0 与 DAgger 数据里，所有训练、验证样本均满足 `near_goal == stop_target`。两套验证集各只有 14 个正例。STOP F1 = 1 可验证条件输入和标签对齐。G3 的视觉停止由两次同名居中近景观测确认，预测扇区深度用于避障。自主停止需要在独立目标实例与困难负例上测。 [打包条件](../../tools/preprocess.py) [在线停止](../../sim/3dgs/collect_3dgs.py)

## 实验顺序和放行条件

截至 2026-11-23 13:10 CST，阶段 1 开发门禁仍未通过，阶段 2、3 暂停。R2 direct-bearing 学生在 10 条开发路线的 SR/STOP 为 `0/10`、碰撞 `3/10`、平均 NE `6.709m`；同一学生改用占据图 A* 子目标后为 `0/10`、碰撞 `5/10`、平均 NE `6.917m`。短 hold 在 10 条路线多出一次成功，也多出一次碰撞，不能说明普遍减抖。碰撞前深度输出存在漏过当前前进门控的例子。R2 的训练标签使用 direct bearing，subgoal 是未训练过的条件分布，因此这组对照是压力测试，不能单独判断 subgoal 方法的上限。学生 checkpoint `text_dim=0`，当前闭环为 privileged bearing 控制诊断，不是语言条件导航结果。下一步先使训练标签与测试方位输入语义一致，再用固定更新数和其余超参数不变的单次配对训练验证，不推进阶段 2 或论文测试。

| 阶段 | 对照与输出 | 放行条件 |
| --- | --- | --- |
| 0 数据与评分，CPU | 先冻结约 6 对开发任务，覆盖同房间、门洞分叉、同名异实例。配对记录固定 ID、起点及朝向、目标区域、阶段参考路线、相机、动作预算与 seed。校验目标区互斥、每条路线可达、门洞连接折线和预算，重放现有 trace。 | 两个目标均可评分，原目标和实际提示目标各有评分。明确区分碰撞、走错门、错目标停靠和超时。 |
| 1 定位局控，复用权重 | 同路线、同权重比较真值局部路标、直线方位、黑屏。教师作为特权能力上界。记录前进率、到达首门洞、碰撞和动作反转。 | 真值路标能驱动学生通过门洞，再测试视觉路标。若失败，优先修动作与局部控制，复用结果定位故障后才安排单次训练。 |
| 2 验证语言，冻结权重 | 先用缓存的相同 RGB 和历史跑正确、异目标、空指令。通过后只跑同一小批任务的配对闭环，比较现有 G3、候选方法及两个单模块消融。 | 同一初始状态下观察到指令改变首个分岔和正确停止位置。按开发集冻结阈值、候选版本、报告指标与评测任务清单。 |
| 3 正式测试 | 冻结此前未参与调试或选模的场景与路线簇。主表比较同口径公开基线适配、G3 基线、完整方法及关键消融。每集登记 SR、SPL、碰撞、错误 STOP 和异常。 | 完整报告各指标分母和分子、按路线簇和场景配对的差值与区间。小样本保留为诊断，不在正文据此主张泛化。 |

主指标为依序通过阶段且在正确实例前无碰撞自主停止的请求集成功率。辅助指标用首个分岔、阶段完成率、wrong STOP、STOP precision/recall、OSR、SPL、导航误差和延迟。场景间隔离优先于随机分帧。跨场景的测试预算和样本数在看测试结果之前登记。OpenFly 的 AirSim 运动学结果和 Wanderland 的点目标控制另列补充表。

## 超参数和算力

- 同一个训练对照只改登记的实验因素。固定打包数据与辅助标签哈希、模型代码/二进制、初始化、batch、优化更新数、学习率曲线总步数、增强、损失权重和最佳权重选择规则。正式训练对照预设配对种子 `20261117, 20261118, 20261119`。固定权重的消融共用 checkpoint。当前 `augment=all` 会平移 RGB 最多 4 像素，深度监督标签只做镜像。若未修同步平移，候选正式训练全臂统一 `photo,flip`。 [增强实现](../../src/trainer_dev.cpp)
- 先用现有权重与轨迹。历史 R0 为 3240 次更新，DAgger 为 8375 次，后者在同一批教师路线采集了新访问状态。若研究 DAgger，固定总更新数和每步调度，其余参数不动。用既有训练记录估算单组，开发阶段按单卡串行。约 54 分钟是既有完整套件的单次墙钟，含渲染与录像。预估新小规模试验取各阶段实测，记录 GPU 型号、共享情况、精度、VLM 实际设备分配与峰值显存。
- `train.py --check` 检查路径和可复用运行契约。单跑 `g3_vlm` 必须指定 `UAVRGB_STUDENT_CKPT`。同一个运行目录内完成阶段才复用，已有旧目录继续供结果审计，新配置使用新名称。固定任务命令、配置、输入哈希、权重和逐集 trace。沿用已有录像要求，同一轨迹需要不同版式时复用已采帧合成。

## 执行更新 2026-11-23 11:52 CST

R2 DAgger 数据已按原 train/validation 场景重新打包，保持 15 个独立验证 episode，test split 为 0。打包后为 16,065 train、915 validation 样本，碰撞步过滤 38 条。由于阶段 1 仍未通过，本轮只执行一组预先限定的诊断训练，用来判断增加 learner-state 样本是否能改变已复现的动作塌缩；它不放行阶段 2，也不作为论文结果。

训练固定 8,375 次 optimizer updates，并保持 dropout 0 cleanaug 对照的 seed、batch、lr、cosine 总步数、双帧 VLN 输入、STOP/depth 辅助损失、增强、weight decay、类别权重和选模规则。训练成功后在同两条冻结路线复跑 direct-bearing 闭环。运行只暴露 GPU 0，PID/PGID `352920/352920`，预计约 12:00 CST 完成。配置、数据哈希和输出位置登记于 [R2 固定更新诊断](2026-11-23-s1-dagger-r2-fixed-updates.md)。结果出来后仍需先检查是否出现正确首个转向、非塌缩的动作分布和显式停止；两条路线不足以证明算法泛化。

## 执行更新 2026-11-23 12:20 CST

并行 trace 审查确认两条匹配路线每一步 `action_proposed == action_executed`，在线干预字段为 null，CF-SITL 的 10 Hz 遥测动作与决策窗口相符。转向振荡来自模型输出。学生以 0.5 秒间隔观察，并在每个窗口保持动作。CPU-only debounce 回放仅靠插入 SAFE 就能把直接反转数降至零，但动作切换增加，改写后的状态轨迹无法从原 trace 推断。

为检查时序频率，登记使用相同 checkpoint 和两条同起点路线、seed、direct bearing、显式 STOP 与模拟总时长，仅将 `--hold` 从 0.5 秒改为 0.25 秒的评测。该诊断不是训练对照，结果不能单独放行阶段 1。实验配置见 [S1 action-window diagnostic](2026-11-23-s1-action-window-diagnostic.md)。

该评测于 12:26 CST 以 `rc=0` 完成，共 2 集且无碰撞。合计 SR/OSR/SPL/显式 STOP 为 `1/2`，平均 NE `1.952 m`。`sage3d_839992/ti_0/0` 成功到达并停止，NE `0.905 m`，SPL `1.0`；FORWARD 动作由 `hold=0.5s` 的 11 次增至 60 次，左/右转反转由 16 次降至 2 次。`sage3d_841980/ts_0/0` 仍未前进或成功，NE `3.044 m -> 2.999 m`，反转 `10 -> 12`。因此缩短动作窗是值得继续验证的时序假设，但当前只有两条复用路线，且一条仍完全失败；阶段 1 不放行。结果和 trace 指标见 [S1 action-window diagnostic](2026-11-23-s1-action-window-diagnostic.md)。

由于 pilot 的收益集中在单条路线，预登记在同一 R2 checkpoint 下复跑原 `r2-direct` 的全部 10 条 task keys，只改 `--hold=0.25s`，其余 seed、direct bearing、显式 STOP 和模拟时长不变。基线全 10 集耗时 `130.7s`，按步骤翻倍估算约 `5` 分钟含余量。守护进程只使用 GPU 0，完成后逐集配对比较。小样本和重复使用路线仍不得用于论文泛化结论。

全量评测于约 12:34:50 CST 启动，PID/PGID `383117/383117`，GPU 0 only，预计 12:40 CST 完成。日志、`DONE` 和停止命令记录于数据盘实验目录 `closed-loop/hold-025-all10/`。

## 执行更新 2026-11-23 12:00 CST

R2 训练于 11:55:15 以 `rc=0` 完成。逐步指标确认正好 8,375 次更新，训练配置的 seed、batch、lr、优化器调度、增强、损失和选模设置与 dropout-0 对照一致。R2 的 best validation macro-F1 / balanced accuracy 为 `0.645596/0.650809`（step 503），基线为 `0.671536/0.676000`（step 1,005）。同一 915 条 validation 上 SAFE 召回保持 14/14，FORWARD 为 520/648 对 523/648，LEFT 为 51/134 对 56/134，RIGHT 为 50/119 对 57/119。变化主要落在左右转向分类，但单 seed 不能证明退化由新增数据导致。

第一次闭环命令因不支持 `student` 非采集模式下的 `--teacher-clearance braking` 在仿真前退出。去掉该参数后评测于 11:59:21 在 GPU0 启动，并于 12:01:33 以 `rc=0` 完成。命令漏设 `--limit 1`，所以两个场景的两个 split 共评估了 10 集。与旧基线完全相同的两条 task key 是 `sage3d_839992/ti_0/0` 和 `sage3d_841980/ts_0/0`，初始位置、朝向及评测条件一致。两条共有路线 SR/OSR/STOP `0/2`，无碰撞，平均 NE 从 `4.130 m` 降到 `3.413 m`；方向反转从基线的 1 次升至 26 次。全 10 集 SR/OSR/STOP `0/10`，碰撞 `3/10`，平均 NE `6.7093 m`。动作分布离开了先前单一右转，但仍以左右转向为主且反转频繁。生成四段视频，每个场景两段。阶段 1 仍未放行。

## 执行更新 2026-11-23 12:52 CST

GPU0 only 的全量配对评测以 `rc=0` 于 12:35:20 CST 完成，`vln_eval.json` 记录 `wall_s=51.8`。基线和 `hold=0.25s` 结果逐项核验了相同的 10 个 task key、checkpoint 哈希、seed、起点/朝向、split、direct bearing、显式 STOP 和 SITL profile，唯一改变项为 hold。短窗口组 SR/OSR/SPL/STOP 为 `1/10`，碰撞 `4/10`，平均 NE `6.281m`；`hold=0.5s` 基线为 `0/10`、`3/10`、`6.709m`。新增成功只发生在 `ti_0`，`ti_2` 同时出现一次新碰撞。前进动作增多，但偏航统计没有显示普遍减抖。三个前进碰撞的深度头输出未过当前门控阈值；第四次碰撞发生在左转。低质量灰帧占 `49.45%`，主要来自 `sage3d_841980`，但这些帧的渲染有效标志正常且两组均出现。

该结果说明动作窗口影响部分路线的进展和风险，不能作为已解决局控或抖动的证据。阶段 1 仍不放行，后续不得据此启动语言方法正式对比或论文测试。下一步应先用已有 CPU 测试与验证集指标审查深度头的风险覆盖，并准备有独立路线簇、固定配置、语言反事实和碰撞/错误 STOP 主指标的配对验证。完整逐集数据和限制见 [action-window diagnostic](2026-11-23-s1-action-window-diagnostic.md)。

## 执行更新 2026-11-23 13:10 CST

深度头验证审查确认现有 CPU 工具不能输出逐帧网格预测或碰撞风险漏报率，不能据总体深度分类准确率替代安全验证。阶段 1 下一项按既有计划比较局部特权路标与 direct bearing，不训练权重。

- 问题：相同学生在 A* 局部绕障子目标（`g0_mode=subgoal`）下，是否比 direct waypoint bearing 更能产生安全前进和路线进展？
- 固定条件：R2 best checkpoint SHA256 `11e67512be11e7d97c8f19fc5110b3470b7a8ff56ef6a06f69f737216e5bb719`，seed 0，原 `r2-direct` 的全部 10 个任务，两个 split、`hold=0.5s`、显式 STOP、`cf_sitl_v1:1bc8fedf:lib31`、模拟总时长与不录视频设置。
- 唯一改变：`--g0-mode direct` 改为 `--g0-mode subgoal`。
- 基线：SR/OSR/SPL/STOP `0/10`，碰撞 `3/10`，平均 NE `6.709m`。
- 记录：SR/OSR/SPL/STOP、碰撞、逐路线 NE、动作分布、反转以及 10 Hz yaw trace。该开发路线实验不单独放行阶段 1，也不支持语言条件导航主张。
- 输出：`/root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/closed-loop/r2-subgoal-all10/`。只暴露 GPU 0，不启动训练。按最近一次同样不录视频的 10 集评测 `wall_s=51.8` 估算，加启动余量约 1 分钟，预计 13:02 CST 完成。旧 `hold=0.5s` 基线 `wall_s=130.7` 含 4 段视频，作为本次估时参考较弱。
- 评测已于约 13:00 CST 启动，PID/PGID `403491/403491`。启动检查确认首条路线已到第 7 步，并开始执行 FORWARD。日志、PID 与完成标记预定保存在输出根目录。

评测以 `rc=0` 于 13:03:56 CST 完成，`wall_s=138.4`。逐集 task key、路线索引、逻辑 spawn、起始位姿/朝向、seed `args.seed + episode_index`、checkpoint 路径、hold、STOP 和 SITL profile 与 `r2-direct` 配对；运行时确认改变项为 `g0_mode`。结果为 SR/OSR/SPL/STOP `0/10`，碰撞 `5/10`，平均 NE `6.917m`、中位 NE `6.929m`。`direct` 基线为 `0/10`、碰撞 `3/10`、平均 NE `6.709m`、中位 NE `6.930m`。前进动作 `20 -> 52`，左右转 `402/392 -> 389/389`，总反转 `159 -> 146`。新增碰撞为 `ti_0`、`ti_4`，`ts_4` 的碰撞消失；`ti_2`、`ti_3`、`ts_1` 持续碰撞。

这项结果应解释为特权占据图子目标输入下的失败压力测试。R2 数据包元数据记录 `vln_bearing_source=direct`，而当前预处理器只支持 `recorded` 和 `direct`，没有 subgoal 标签来源。测试时改变在线 bearing 造成输入分布偏移，故不能据此否定经 subgoal 监督训练的学生。参考教师在相同 10 条路线为 SR `9/10`、STOP `10/10`、碰撞 `0/10`，表明场景中存在可行路径，但教师的地图/A* 能力不能归为学生视觉控制成功。

下一项训练前先做 CPU 数据合同审查，定义如何从 teacher/DAgger 状态生成与在线 A* 子目标同语义的 bearing 标签，并确认训练和评估共用同一 lookahead 规则。方案和标签生成测试通过后，再只用 GPU0 执行一个固定 8,375 updates 的诊断臂，其他训练超参数沿用 R2，不并行训练。独立场景和语言条件验证仍保持暂停。逐集结果写入数据盘 `closed-loop/r2-subgoal-all10/eval/episodes.jsonl`。

```bash
source env.sh && CUDA_VISIBLE_DEVICES=0 /root/rivermind-data/uav_rgb/3dgs-venv/bin/python -m sim.3dgs.run_vln --policy student --ckpt /root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/train_r2/model/ckpt_best.bin --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_839992/scene.json --scene /root/rivermind-data/uav_rgb/3dgs-scenes/sage3d_841980/scene.json --out /root/rivermind-data/uav_rgb/exp-s1-dagger-r2-cleanaug-20261123/closed-loop/r2-subgoal-all10/eval --hold 0.5 --g0-mode subgoal --stop-mode explicit --split test_Instruction_Unseen --split test_Scene_Unseen --seed 0 --videos 0
```

## 执行更新 2026-11-23 14:55 CST

subgoal bearing 标签合同已在 CPU 侧实现并完成全包验证。现有 R2 数据包按原 `meta.episodes` 顺序重打包 direct 和 subgoal 两份，均为 180 集、16,980 样本、16,065 train、915 validation、0 test，过滤 38 个碰撞步。原包与 direct 重包的图像、动作标签和 auxiliary payload 完全一致。direct 与 subgoal 两份的图像、动作、prev-frame、prev-action、depth、STOP、near-goal 和 search-dropout mask 均一致，只有 bearing 条件改变，9,746 个未屏蔽样本的 bearing 不同。完整数据哈希与运行状态见数据盘 `train_r2_subgoal/progress.md`。

阶段 1 仍未通过，因此这是一组固定 8,375 updates 的特权 bearing 控制诊断。使用同一 seed、训练参数和路线，将在新权重上依次评估 direct 和 subgoal 两种输入语义。训练与两次评测均只暴露物理 GPU 0，GPU 1 保持未使用。该学生的 `text_dim=0`，不消费自然语言；结果不能当作端到端 VLN 论文结果或泛化证据。

守护任务于 14:55:19 CST 启动，PID/PGID `482099/482099`，日志位于数据盘 `train_r2_subgoal/run.log`，完成标记为 `train_r2_subgoal/DONE`。按此前 R2 训练 211 秒、direct 评测约 52 秒、subgoal 评测 138 秒，并加 15% 余量，预计 15:03 CST 完成。当前没有训练指标或闭环结果，需用户回来要求查进度后再读取。

## 执行更新 2026-11-23 15:46 CST

8,375-update subgoal-bearing 诊断训练于 15:00:53 CST 以 `rc=0` 完成。实际训练步数为 8,375，固定 R2 的 seed、batch、lr、cosine 总步数、增强、损失、类别权重和选模规则，仅更换 bearing 标签包。最佳 validation macro-F1 / balanced accuracy 为 `0.787543/0.736004`，发生于 step 3,018；checkpoint SHA256 为 `513e95a7a26709ab38a0157453138ea28ef0dac663e8f267919c2c64e397e667`。

同一 checkpoint、10 个任务、逐集 seed、逻辑起点、`hold=0.5s`、显式 STOP 与 SITL profile 的 2x2 对照为：

| 训练标签 | 在线 bearing | SR / OSR / STOP | 碰撞 | 平均 NE | SPL |
| --- | --- | --- | --- | --- | --- |
| direct | direct | 0/10 | 3/10 | 6.709 m | 0 |
| direct | subgoal | 0/10 | 5/10 | 6.917 m | 0 |
| subgoal | direct | 0/10 | 10/10 | 5.315 m | 0 |
| subgoal | subgoal | 7/10 | 1/10 | 2.196 m | 0.631 |

匹配的 subgoal 条件给出局部控制可行性信号，但成功集中在 `sage3d_839992` 的 5 条路线；`sage3d_841980` 为 2/5，另有 1 次碰撞和 2 次超时。路线来自反复使用的开发集，没有独立门洞通过指标。四种试验的 `grounder=none`，学生 checkpoint 的 `text_dim=0`，因此任务自带英文指令没有进入动作决策。不得将这组结果作为语言贡献、端到端 VLN 或泛化证据，阶段 1 仍未正式放行。

按计划在启动闭环前，使用同一缓存 RGB 的 correct/swap/zero 查询做了小规模 VLM probe。初始 worker 输出有 4/8 非空响应触发安全解析回退，原因是二值 near 字段生成 `0.8` 一类小数或响应截断。worker 提示已明确第三字段必须为整数 0/1，最大生成 token 从 12 调至 20，解析器仍维持 fail-closed。修正后 8/8 非空响应可解析，但正确/交换指令在两个分岔样本上仍输出相同 bearing `R`；4 个样本中只有一个正确门目标查询返回 `final`，所有查询 `near_goal=0`。这没有通过视觉目标及停靠门槛，因此没有启动 G3 闭环或新训练。

下一步先改进目标区域/目标身份证据，让目标可见与近距离确认可以被独立、稳定地评分，再用相同冻结帧复测 correct/swap/zero。不得仅因 privileged subgoal 控制结果达到 7/10 就推进论文级对照或新增训练。当前评测报告新增 `active_bearing_source`，避免将 VLM bearing 误读成配置中的 `g0_mode`。实验明细见 [subgoal-bearing 训练与闭环记录](2026-11-23-s1-dagger-r2-fixed-updates.md) 和 [冻结帧语言探针](2026-11-23-vln-grounding-frame-probe.md)。

## 执行更新 2026-11-23 16:35 CST

复查发现冻结帧探针的 branch 样本取自 rollout 轨迹位置分离处，而非任务参考路径分叉处。`ts_2` / `ts_3` 参考路径前 20 个点相同，采样帧又仍在共享前缀，因此 correct/swap 都输出 `R` 不能作为语言失效证据。

修复了解析器接受截断 near-STOP JSON，以及 tracker 将任意重复 `reason` 当成目标身份的问题。当前要求完整 JSON、英文/中文目标短语与 VLM 标签相符、连续两次确认。此保护还没有目标框或像素级实例对应，无法解决同类实例歧义。

新版冻结帧探针使用相同 4 帧 correct/swap/zero，物理 GPU0 实测 18.86 秒；8/8 非空输出符合解析格式。near-door 正指令输出 `R/F/1/entrance door`，错误 lighting 指令输出 `R/S/0/none`。该帧仍为右侧方位，当前 tracker 不会据此直接 STOP；单正样本不足以估算停止性能。

阶段 1 和语言门禁仍未放行，没有启动 G3 闭环或训练。下一步先基于参考路线挑选或生成处于目标相关分岔的同位姿 RGB，并补足独立的正确目标、错目标、近景、遮挡和同类干扰帧，再决定闭环评测。详细结果见 [冻结帧探针](2026-11-23-vln-grounding-frame-probe.md)。

## 执行更新 2026-11-23 17:15 CST

并行路线审查找到了有效的同起点双目标开发配对。`sage3d_841980/ts_2` 与 `ts_3` 出生位姿相同，目标点相距约 `4.09m`，1m 目标圆不重叠。参考路径在前 20 点相同，并从 `(-4.9064,-0.505)` 分叉。冻结帧 `ts_2_3/rgb_00094.png` 距分叉约 `0.22m`，图像质量有效。它只用于开发探针，不能当独立测试。

在这同一 RGB 上，VLM 对果盘指令输出左方位，符合 1m 参考路径 lookahead；对 tuyere 指令输出右方位，且将 `bedroom` 写成原因，未符合目标路线。交换指令会改变输出，但目标路线并未都选对。因此语言只显示有影响的初步证据，门禁仍未通过。

新增的 STOP 冻结帧中，门目标从距离 waypoint 约 `8.9m`、`4.0m` 到 `0.87m`，正确指令在前两帧保持 `subgoal/near=0`，近景输出 `final/near=1`；换成 lighting 指令时三帧均为 `subgoal/near=0`。另一张约 `0.82m` 的画作近景正确命名了 `decorative painting`，但 `near=0`，是假阴性。距离只代表无人机到路线 waypoint 的代理距离。近门目标位于画面右侧，tracker 仍要求中心方位后才接受 STOP。

CPU 代码审查发现 `counterfactual=swap` 只替换了语言，成功率一直按原任务目标计算。现已保留原任务指标，并增加实际提示指令对应路线 waypoint 的 `supplied_goal_*` 行级与汇总指标，报告分别标注两种口径。新增同起点双目标回归测试验证飞到提示目标时两套 SR 给出不同结果。`test_vln.py` 71 项、`test_grounder.py` 27 项、`py_compile` 与 `git diff --check` 通过。

12 次非空冻结帧推理在物理 GPU0 耗时 `22.6s`，GPU1 未使用。结果文件位于数据盘 `language-grounding-probe-r2-subgoal-20261123/probe_after_branch_and_stop_v5.json`，逐帧记录和限制见 [冻结帧语言探针](2026-11-23-vln-grounding-frame-probe.md)。没有启动训练或 G3 闭环。阶段 1 和语言门禁仍未通过；下一步先扩充多条同起点分岔配对及不同物体的近/远、遮挡和同类实例负例，并用 CPU 校验 supplied-goal routewaypoint 与实际任务对象映射，再预登记闭环动作预算和双目标评分。

### 同起点路线候选筛查（17:47 CST）

2026-11-23 的只读筛查遍历 6 个 `sage3d_*` 场景、578 条路线。以出生位置差不超过 `0.05m`、yaw 差不超过 `1°`、同 split、不同目标点至少相距 `2m`、分叉后各保留至少 `1.2m` 为条件，得到 591 对候选。最大理论方位间隔的候选帧被黑墙遮挡，不适合视觉探针。第二候选 `sage3d_840865/tr_38/0` 与 `tr_95/0` 约差 2 个实际帧 bearing bin，指令分别是 “Take a warmed towel from the bathroom and place it beside the pillow in storage.” 和 “Please transfer a wet towel from the bathroom to the laundry area for heating.” 现有清晰帧 `same_spawn_to_cup/0/rgb_00022.png` 实际来自“去门口”的另一任务；虽然同 GS 几何且位姿接近两条路线，它不属于这两条 ref_path 的 rollout，scene JSON 哈希也不同。没有将其当作路线配对结果，也没有追加 GPU 查询。该候选只有在取得与目标路线合同匹配的帧后，才可用于开发探针；不能替代独立闭环测试。
