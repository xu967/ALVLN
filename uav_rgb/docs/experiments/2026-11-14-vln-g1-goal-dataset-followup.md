# VLN 实验与数据跟进（2026-11-14）

## 结论先行

当前没有活跃训练、导航仿真、SITL 或 Gazebo 任务，GPU 空闲。L1 语言/视觉训练与验证已经结束，但数据只有 train/val、`test=0`；它能说明旧模板动作任务会用到语言，不能说明 3DGS 自然语言导航已经完成。现有 3DGS teacher、reactive 和视觉避障结果都不是语言策略成绩。

本地 13 个已导入场景的 `scene.json` 所引用 PLY 均存在。数据集完整性不是“所有上游数据都已下载”：SAGE 是 VLN 标注加六场景运行子集，Wanderland 是一个可导航场景但缺原始图像归档，Inria 的 treehill checkpoint 不完整。`/root/rivermind-data` 尚有约 243 GB，根 overlay 仅约 1.3 GB，所有新实验、缓存和下载必须写入数据盘。

## 实验状态

| 实验/产物 | 状态与可报告结果 | 结论边界 |
|---|---|---|
| `/root/rivermind-data/uav_rgb/exp-l1-language-formal-20261114` | DONE；语言模型 3 seeds × 20 epochs。best checkpoint 在 val 的 accuracy 为 0.9005/0.8987/0.9023，macro-F1 为 0.8767/0.8739/0.8750，balanced accuracy 为 0.8839/0.8635/0.8793。 | 数据 16,184 帧/902 集，train 14,496、val 1,688、test 0；缺少 `split_scheme/split_salt`。是旧 Gazebo 模板动作任务验证，不是 3DGS VLN 或独立泛化结果。 |
| `/root/rivermind-data/uav_rgb/exp-l1-vision-parallel-20261114` | DONE；相同验证划分的视觉模型 3 seeds。eval-only accuracy 0.5071/0.5071/0.5095，macro-F1 0.5083/0.5138/0.5150。 | 与语言模型只支持“该模板任务上语言输入有用”的局部结论；没有独立 test，不能外推到开放语言。 |
| `/root/rivermind-data/uav_rgb/exp-l1-eval-20261114` | 六个 eval-only 子目录均有结束日志和 metrics，但父目录没有 `DONE` 或总汇总。 | 记账状态不完整；不要重跑或写成独立 test。后续可只读汇总并明确 `test=0`。 |
| `exp-vln-m1-20261113/{teacher,forward}` | 249 条 SAGE route：teacher SR 0.8434、OSR 0.8554、SPL 0.6770、collision 0.1486；forward SR 0、OSR 0.241、collision 0.964。 | teacher 直接拿 waypoint、占据图和 A*；是特权控制上界。forward 是盲飞下界。两者都不读取语言。 |
| `exp-baselines-3dgs-20261112` | DONE；室内/室外各 120 集。室内 forward+oracle-route-guard 72/120、PULP-tiny+guard 58/120、TrailNet+guard 53/120；室外分别 107/120、105/120（后两者为 forward+guard、TrailNet+guard）。 | 视觉避障/护栏工程基线，带 oracle route，不是 VLN 策略对比。跨任务分数不可和 VLN SR 直接排名。 |
| `tmp/exp-reactive-paired-20261114` | 两条 841980 route 的 reactive 与 A* 都 0/2；840865 同起点异目标配对中 reactive 1/2、A* 2/2。 | 每臂每路线只有 2 集，仅作控制器方向诊断；目标方向来自评测器，不是语言 grounder。 |
| `tmp/wanderland-long-20261114/route{0,1,2}` | 12.8、15.2、12.8 m；每条路线单次，分别 126、147、129 步上限，产物齐；均无碰撞并显式停止。 | 证明当前导入场景、renderer 和 cf_sitl 管线可运行，不证明长航线稳定或真机效果；dynamics manifest 为 `calibrated=false`。 |
| `tmp/vln-followup-teacher-20261114b` | SAGE 单集成功：SR/OSR 1、SPL 0.8629、NE 0.298 m、无碰撞，21 次动作切换（2.24 次/m）。 | 一集 smoke。前一目录 `tmp/vln-followup-teacher-20261114` 因输出目录非空而退出，不能计入成功重跑。 |
| `tmp/vln-grounder-endpoint-diagnostic-20261114.json` | 4,429 条指令中 1,077 接受、830 命中标注 end，总命中率 18.74%。 | 文件带 `leakage_warning=true`，候选来自路线 start/end 标注；不能作正式 G1 grounding 结果。 |

### 飞行平顺性证据

旧 teacher clearance 批次在 hold=1.0 的 120 个室内样本中，swept 净空版 112/120 到点、8 次碰撞；旧 ray 版 101/120 到点。该批次优化的是碰撞门槛，不是平滑控制，且验收与拟合共用 seed。历史教师 249 集的动作切换中位数为 11，原地转向占比中位数 0.18。近期 Wanderland 每路仅 1 集、动力学未标定，不能据此断言抖动改善。下一轮固定动力学 profile，配对记录 action switches/m、SAFE 窗、yaw 变化、刹停距离、终点误差和碰撞。

## 数据下载完整性

只读检查，没有下载、删除或重写数据。

| 数据 | 本地状态 | 是否足以用于当前任务 |
|---|---|---|
| `/root/rivermind-data/uav_rgb/3dgs-scenes/*/scene.json` | 13 个场景引用的 PLY 全部存在；可直接加载的有 bicycle、callinstrasse、garden、luigi、6 个 sage3d、stump、synthetic、wanderland。 | 能用于当前场景加载，不代表所有视角画质和飞行覆盖合格。 |
| SAGE VLN：`/root/rivermind-data/uav_rgb/pubdata/sage3d` | 原始 VLN 标注约 271 MB；本地已有六个导入场景及训练/测试 JSON 子集。不是全量官方 USDZ/3DGS 场景资产。`/root/rivermind-data/uav_datasets/sage3d_vln` 只有空目录。 | 可以继续当前六场景导航开发；正式泛化仍受场景数和 test route 数限制。 |
| Wanderland：`/root/rivermind-data/uav_rgb/downloads/wanderland/1MlUucugRAnrx86cpUuQt8fUtxFMddxju` | 下载记录 `DONE rc=0`，8 个文件约 1.71 GB；scene.usdz、3dgs/raw PLY、episodes/transforms 和 COLMAP sparse 文件存在。鱼眼/原始图像与 mask 的四个 tar 未下载。 | 已导入场景可做渲染和导航；缺图像归档时不适合从原图复建/重训。它只是一处场景，不是上游 335 场景全量。 |
| Inria 预训练：`/root/rivermind-data/uav_rgb/downloads/inria-3dgs-pretrained` | bicycle、garden、stump 有 `point_cloud_30000.ply` 和 `.ok` SHA 记录；本轮未全盘重算大文件哈希。treehill 缺 `point_cloud_30000.ply`，下载记录 rc=143。 | 三个完整目录可用；treehill 可选，不阻塞当前 VLN。 |
| Callinstrasse | 最终 `3dgs_data.ply` 存在，大小与 assemble 记录匹配；另有失败下载留下的 2.4 MB `.partial`。 | scene 引用的最终文件可用；`.partial` 不是当前 scene 资产，不要误当作完整下载。 |

代理检查：没有 proxy 环境变量，`127.0.0.1:7897` 无监听。treehill 上次直连约 29 KB/s，829 MB 压缩包估算约 8 小时后被停止。当前没有重试下载。若未来需要原始图像做重建，先将 Wanderland 四个 tar 下载到数据盘；treehill checkpoint 不是优先项。

## 下轮实验安排与验收

1. **先冻结 G1 独立候选表**：来源应是独立地图/标注，而不是 SAGE 测试端点或路线 start/end；保存来源、哈希、坐标系和单位，并人工审查与测试指令的独立性。至少设计同起点不同目标配对。
2. **跑 G1 反事实闭环**：正确指令、错配目标、空指令、否定/关系词分别统计 grounding coverage、拒绝率、接受准确率；导航报告目标 SR/SPL 与 route-waypoint proxy 两套口径。先用 `teacher` 验证 grounder 到目标接口，再用 reactive/条件学生拆分控制误差。
3. **做连续长路线门禁**：从 Wanderland 原始参考路径筛候选 30–50 m 段，校验坐标、逐段图像/深度覆盖、空洞、净空和碰撞网格后，再多个 seed 配对运行。55.7 m 参考轨迹不等于验证通过的可飞航线。
4. **定量处理抖动**：固定未变的 cf_sitl profile；先记录转换动作、方向反转、动作窗口内位移、刹停距离与 yaw 变化；每次只改一个策略因素，至少多路线、多 seed 做 paired 对照。未经 SITL 标定前不改真机 PID/固件。
5. **补实验账本**：对 L1 六个 eval 子目录做只读 summary，标明缺少父级 DONE 和 `test=0`；不要为补标记重训。新运行均在数据盘建立独立目录、固定 seed/config/data hash，结束后写结果与失败集。

本轮代码回归：项目 Python 环境下 grounder/VLN/video、SAGE/Wanderland importer 和 3DGS 控制层共 221 项 unittest 通过。正式 lexical G1 尚未执行，因为仓库当前没有通过独立来源审查的 catalog。
