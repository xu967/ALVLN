# bench：数据集 / 基准 / 场景资产调研与可达性探测（2026-11-10）

- 起止：19:20 起，时间盒 60 分钟。本报告在 19:5x 至 20:0x 之间写成。
- 路径约定：`B=/root/rivermind-data/uav_rgb/tmp/phase12-20261110/bench`，`D=/root/rivermind-data/uav_rgb`，`R=/root/claude-code-backup-20261028/code/uav_rgb`（只读）。
- 来源约定：
  - 标 `arXiv:ID` 的数字取自 `https://arxiv.org/abs/ID` 对应的 HTML 全文，本地缓存 `B/cache/ID.txt`。
  - 标 `HF:repo` 的数字取自 `https://huggingface.co/datasets/repo` 经 hf-mirror 的 API 目录树，原始 JSON 在 `B/meta/<org>__<name>.{info,tree}.json`，汇总在 `B/probe/hf_summary.txt`。
  - 标 `实测` 的数字来自本轮在本机执行的命令，脚本和输出在 `B/pilot/`、`B/probe/`。
  - 标 `未核实` 表示本轮没有拿到原文或没有执行验证；标 `没做` 的原因写在末尾「没做什么」。
- 本轮没有修改 `R` 下任何文件，没有创建 git 分支或提交。新增数据：`D/pubdata/sage3d/sample/` 205 MB，草稿目录 `B/` 114 MB（均为 `du -sh`），合计约 319 MB。

## 0. 摘要

**要不要换数据集**：不替换现有柱阵任务，在它旁边增加第二个任务族：InteriorGS 室内 3DGS 场景加 SAGE-3D 的轨迹与指令。理由有三条，都已在本机验证：

1. 取数不需要申请。`spatialverse/SAGE-3D_InteriorGS_usdz`（CC-BY-NC-4.0）和 `spatialverse/SAGE-3D_VLN_Data`（CC-BY-4.0）在 hf-mirror 上均为 `gated=False`，样本下载的字节数与 HEAD 返回一致。
2. 已解码为 INRIA PLY，仓库的 `read_gaussian_ply` 能读，`GsplatRenderer` 在 320×240、60° 视场下渲染 6 个视角成功，约 2 至 3 ms 每帧（首帧 953 ms）。渲染图见 `B/pilot/render_841150.png`。
3. 有 1,994,510 条轨迹指令对、1,000 个场景、5 类高层指令（HF:spatialverse/SAGE-3D_VLN_Data README），可以直接支撑长程多段任务和 ISR/CSR 类指标。

- **主选**：InteriorGS（经 SAGE-3D usdz 镜像取得）加 SAGE-3D VLN 数据。体量：每场景 usdz 中位 96.0 MB，VLN JSON 中位 9.8 MB。许可动作：取这两份不需要用户操作；要原版 InteriorGS 的 occupancy、structure、labels 需用户到 huggingface.co 点同意并提供 token。
- **备选 1**：Wanderland 全量库（ungated，apache-2.0，1,149.74 GB，目录树可见 335 个场景目录，我们已有 importer）。样例场景的点目标 episode 测地距离 30.7 至 75.4 m，落在 20 至 60 m 任务范围内，但指令是固定的一句话。体量：每场景 `scene.usdz` 中位 327 MB。许可动作：无。
- **备选 2**：SceneSplat-7K 系列中的 ScanNet++ v2 / Replica / Matterport3D 3DGS（gated=auto，CC-BY-SA-4.0 类许可），只有场景，路线和指令要自己生成。体量：ScanNet++ v2 每场景 354.0 MB。许可动作：用户到对应 HF 仓库页面点同意并提供 token。
- **对比协议**：SAGE-Bench 的指令划分与连续性指标（CSR/ICP/PS）加 LH-VLN 的多段指标（ISR/CSR/CGT），动作粒度按我们的 0.25 m / 25° 重新定义。官方评测器需要 Isaac Sim 5.0+，本机跑不了，只复用数据和指标定义。公开权重只核到模型 id 一级（3.1），gated 状态和大小未核实。

**必须让用户知道的三个风险**：

- 许可：SAGE 的场景和碰撞体是 CC-BY-NC-4.0，InteriorGS 原库是 Terms of Use（非商业科研教学，不得再分发）。只能用于研究，产物不能随意再分发。
- 路线偏短：已核实 5 个场景共 586 条 SAGE 路线（841150、839992，以及 usdz 最大的 840104、839938、840007），长度 p10/p50/p90/max 为 2.8/6.6/11.7/21.6 m，≥10 m 的占 21%，≥20 m 只有 2 条。20 至 60 m 的任务要靠多段拼接，或改用 Wanderland 的点目标 episode（样例场景 30.7 至 75.4 m）。其余 995 个场景的分布未核实。
- 净空与坐标系：公寓走廊对 e1_mini 包络偏窄，71 条可规划路线中只有 16 条净空 ≥ 机体半径 + 0.1 m，6 条满足仓库默认的 + 0.3 m。SAGE 轨迹坐标系到 PLY 坐标系的变换（绕 z 轴转 180°，无平移）是推断加统计支持，没有拿到官方 occupancy 对照。

需要用户决定的事列在 3.1 末尾。

## 1. 候选清单核实表

### 1.1 3DGS 场景资产与 3DGS 具身导航基准

| 条目与链接 | 提供内容 | 规模 | 体量 | 许可与申请 | 渲染或仿真需求（容器仅 CUDA） | 动作与成功判据 | 公开基线 |
|---|---|---|---|---|---|---|---|
| InteriorGS `https://huggingface.co/datasets/spatialverse/InteriorGS`；`https://github.com/manycore-research/InteriorGS` | 每场景 `3dgs_compressed.ply`（SuperSplat 压缩）、`labels.json`、`occupancy.json`、`occupancy.png`、`structure.json`（`B/meta/InteriorGS_README.md`；目录树扩展名计数 json 3000、ply 1000、png 1000） | 1,000 场景，752 住宅加 248 公共空间（arXiv:2510.21307） | HF:总 43.81 GB，5,003 文件；PLY 每场景 min/p50/p90/max 16.9/37.8/65.1/144.1 MB | `gated=auto`，镜像 HEAD 返回 403 `GatedRepo`（`B/probe/hf_head_tests.txt`）；Terms of Use：仅非商业科研与教学，不得再分发，需引用（`B/meta/InteriorGS_Terms_of_Use.pdf`）；需用户在 huggingface.co 点同意并提供 token | 读 PLY 不需仿真器；压缩 PLY 需先解压，`read_gaussian_ply` 只收全 float 属性 | 无（仅场景） | 无 |
| SAGE-3D_InteriorGS_usdz `https://huggingface.co/datasets/spatialverse/SAGE-3D_InteriorGS_usdz` | InteriorGS 同一批 3DGS 的 NuRec USDZ（zip 内含 `default.usda`、`gauss.usda`、`<id>.usdz.nurec`，后者是 gzip 加 msgpack，fp16 数组） | 1,000 场景 | HF:总 110.90 GB，1,003 文件；每场景 min/p50/p90/max 43.5/96.0/167.2/401.4 MB | CC-BY-NC-4.0，`gated=False`，hf-mirror 实测可下（HEAD 200，字节数一致） | 官方用 Isaac Sim 5.0+ 渲染（`B/gh/Galery23_SAGE-3D_Official_README.md` 第 98 至 99 行）；本机解码为 INRIA PLY 后用 gsplat 渲染成功（2.3） | 无 | 无 |
| SAGE-3D_Collision_Mesh `https://huggingface.co/datasets/spatialverse/SAGE-3D_Collision_Mesh` | 每场景 `.usd` 碰撞体，魔数 `PXR-USDC`（二进制） | 1,000 场景 | HF:总 39.42 GB；每场景 1.4/34.1/74.4/207.6 MB | CC-BY-NC-4.0，`gated=False` | 读取需要 `pxr`（usd-core），三个 python 解释器都报 `ModuleNotFoundError`，所以没解析 | 无 | 无 |
| SAGE-3D_VLN_Data `https://huggingface.co/datasets/spatialverse/SAGE-3D_VLN_Data`；论文 arXiv:2510.21307；代码 `https://github.com/Galery23/SAGE-3D_Official` | 轨迹 `points`（位置、四元数、相机内参，样例间距 0.05 m）加分层指令（高层 5 类加低层加 no-goal） | 1,994,510 对；929 训练场景加 45 测试场景，3 个泛化划分（Scene/Trajectory/Instruction-Unseen）（HF README） | HF:总 9.82 GB，1,071 文件；训练 JSON 每场景 2.1/9.8/14.7/35.7 MB（n=929） | CC-BY-4.0，`gated=False` | JSON 直接读；官方 SAGE-Bench 评测需要 Isaac Sim 5.0+（README 第 98 至 99 行） | 指标 CSR/ICP/PS 外加 SR 等；`r_tol` 与成功半径未核实。难度轴：资产数 >376 为 many、<184 为 few；路径 >29.0 m 为 long、<8.4 m 为 short（arXiv:2510.21307） | README 未列出公开权重（对 `checkpoint`、`weights`、`baseline` 的 grep 无命中）；论文正文提到 NaVILA 25 次、NaVid 16 次，具体评测对象与结果表未核实；NaVILA 权重见 3.1 |
| Wanderland 全量 `https://huggingface.co/datasets/ai4ce/wanderland`；arXiv:2511.20620 | 每场景 `3dgs.ply`、`scene.usdz`、`raw_pcd.ply`（LiDAR）、`episodes.json`、`transforms.json`、图像 tar.gz | 论文：530 场景、42 万帧、100 多小时、380 万 m²，单场景 5,000 至 10,000 m²。HF 目录树可见 335 个场景目录：`3dgs.ply` 334 个、`episodes.json` 293 个、`scene.usdz` 292 个。两个数字差异的原因未核实 | HF:总 1,149.74 GB，5,071 文件；`scene.usdz` 每场景 min/p50/max 209.0/327.0/487.7 MB（n=292）；样例场景 `3dgs.ply` 427,094,926 B（HEAD 200） | apache-2.0，`gated=False` | 官方整合到 Isaac Sim；我们已有 `tools/scene_import_wanderland.py`，gsplat 闭环已在 1 个场景跑通 | 点目标 episode：样例场景 18 条，`geodesic_distance` min/p50/max 30.7/39.5/75.4 m，`goals[0].radius`=3.0，指令固定为 "Navigate from the start position to the goal position."（`B/meta/wanderland/episodes_12mKex.json`） | 未核实 |
| Wanderland preview `https://huggingface.co/datasets/ai4ce/wanderland-preview` | 全量库的预览子集 | `scenes/` 下 128 个文件 | HF:总 1.34 GB | apache-2.0，`gated=False` | 已用于 `wanderland_1MlUucug` | 同上 | 未核实 |
| VLN-PE arXiv:2507.13019 | 基于 GRUTopia（Isaac Sim）的物理具身 VLN；场景含 MP3D、合成住宅、3DGS 扫描场景 | 场景数与轨迹数未核实 | 未核实 | 未核实；MP3D 需要授权 | Isaac Sim | 论文称 VLN-CE 模型迁移到物理环境后 SR 相对下降 34%；成功半径未核实 | 权重未核实 |
| GaussGym arXiv:2510.15352；`https://huggingface.co/datasets/escontra/gauss_gym_data` | splatfacto `splat.ply` 加 `meshes`，用于足式机器人 locomotion 训练 | gauss_gym_data 有 8 个场景：amazon_stairs、long_stairs、washington、bww_stairs、home_night、ryokan_bridge、grace_cathedral、cute_bridge | HF:gauss_gym_data 3.25 GB，最大 `splat.ply` 659.7 MB；`escontra/gauss_gym_arkit` 917.00 GB，30,031 文件 | gauss_gym_data：apache-2.0；gauss_gym_arkit：许可字段为空 | 3DGS 作为 IsaacGym 的渲染器 | 无 VLN 指令 | 无 |
| OpenFly 的 3DGS 部分 arXiv:2502.18041；`https://huggingface.co/datasets/IPEC-COMMUNITY/OpenFly`；`.../OpenFly_DataGen` | UE、GTA V、Google Earth、3D GS 四种渲染的场景和轨迹 | 100k 轨迹，18 个场景；4 个 3DGS 场景约占数据 20%（arXiv:2502.18041） | HF:`OpenFly_DataGen` 124.37 GB（22 文件）；`OpenFly` 目录树因 429 没取全，总大小未核实 | `OpenFly` apache-2.0 `gated=False`；`OpenFly_DataGen` `gated=auto` | 3DGS 场景用于城市尺度 | `Forward` 步长 3/6/9 m；成功为停在目标 20 m 内（arXiv:2502.18041） | HF 模型搜索命中 `IPEC-COMMUNITY/openfly-agent-7b`（533 次下载，`B/probe/hf_models_search.txt`） |
| SceneSplat-7K arXiv:2503.18052；`https://huggingface.co/datasets/GaussianWorld/scene_splat_7k` | 主仓只有 `3dgs_training_views/`（每帧位姿 JSON）、`data_splits/`、`statistics/`；高斯文件在独立仓库 `GaussianWorld/<name>_mcmc_3dgs`（README 列表） | 7,916 场景、112.7 亿高斯，来源 ScanNet、ScanNet++ v1/v2、Replica、Hypersim、3RScan、ARKitScenes、Matterport3D，不含 HM3D（README）。README 统计表：ScanNet++ v2 956 场景、Replica 18、Matterport3D 90（每场景均值 1,200 万高斯）、ScanNet++ 330 | HF:主仓 5.34 GB（12,915 文件，其中 12,888 个为 JSON）；`scannetpp_v2_mcmc_3dgs` 338.24 GB，每场景 `ckpts/point_cloud_30000.ply` 354.0 MB | 主仓 `gated=auto`，CC-BY-SA-4.0（license_name 为 scene-splat-7k-license）；`scannetpp_v2_mcmc_3dgs` HEAD 返回 403 `GatedRepo`；需用户点同意并提供 token | PLY 为 INRIA 格式（354.0 MB 对应 150 万高斯乘 236 B），读取无需仿真器 | 无路线、指令、碰撞体 | 无 |
| SceneSplat-49K arXiv:2506.08710；`https://huggingface.co/datasets/GaussianWorld/scene_splat_49k` | 主仓仅 14 个文件，`statistics/*.csv`（来源含 3rscan、aria_synthetic_env、arkitscenes、dl3dv、holicity、hypersim、matterport3d、replica、scannet、scannetpp v1/v2） | 约 49K 场景（arXiv:2506.08710） | 主仓 0.00 GB（只有统计 CSV） | `gated=auto`，CC-BY-SA-4.0 | 高斯文件位置未核实 | 无 | 无 |
| DL3DV-10K arXiv:2312.16256；`https://huggingface.co/datasets/DL3DV/DL3DV-GS-960P` | 3DGS 的 zip 与去畸变图像 | 10,510 个 4K 视频、51.2 百万帧（正文另一处写 51.3 百万）、65 类 POI（arXiv:2312.16256） | HF:`DL3DV-GS-960P` 4,251.24 GB，13,880 文件，最大 zip 4.17 GB | `gated=auto`，许可字段为空，HEAD 返回 403 | 无 | 无 | 无 |
| ScanNet++ arXiv:2308.11417 | 高分辨率重建、DSLR 与 iPhone 序列、语义标注 | 460 场景、28 万 DSLR 图、370 万以上 iPhone RGB-D 帧 | 未核实 | 需申请并同意使用条款，条款细节未核实 | 无官方 3DGS | 无 | 无 |
| RoboTidy arXiv:2511.14161 | 基于 InteriorGS 的 3DGS 家庭场景，带凸分解碰撞网格 | 500 个场景，6.4k 操作轨迹、1.5k 导航轨迹 | 发布位置与大小未核实 | 未核实 | 未核实 | 未核实 | 未核实 |
| GN0 / GN-Matrix arXiv:2606.03682 | 在 InteriorGS 1,000 场景上做网格与高斯对齐 | 1,000 场景（取自 InteriorGS） | 未核实 | 未核实 | 未核实 | 未核实 | 未核实 |

### 1.2 室内连续环境 VLN 与室内无人机语言导航

| 条目与链接 | 提供内容与规模 | 体量 | 许可与申请 | 渲染或仿真需求 | 动作与成功判据 | 公开基线 |
|---|---|---|---|---|---|---|
| VLN-CE / R2R-CE arXiv:2004.02857 | 4,475 条轨迹，由 R2R 训练与验证集转换；平均每条 55.88 个动作，R2R 为 5 个节点跳 | 场景为 Matterport3D，体量未核实 | MP3D 需要申请，细节未核实 | Habitat，需要 GL/EGL，本机不可行 | 4 个动作：前进 0.25 m、左转 15°、右转 15°、停止；成功半径本轮抽取未命中，未核实 | 未核实 |
| RxR / RxR-CE arXiv:2010.07954 | 126K 条指令，16.5K 条引导路径，每种语言 14K 条路径乘 3 条指令；路径平均比最短路径长 27.4% | 未核实 | 未核实 | Habitat | RxR-CE 的动作与成功半径未核实 | 未核实 |
| LH-VLN / LHPR-VLN arXiv:2412.09082；`https://huggingface.co/datasets/Starry123/LHPR-VLN` | 3,260 个任务，平均 150 个任务步，多子任务 | HF:91.36 GB，14 文件，中位 8.97 GB | HF 许可 mit，`gated=False`；场景为 HM3D，需另行申请 | Habitat | 前进 +0.25 m、左转 +30°、右转 -30°、停止；指标 ISR、CSR、CGT、TAR | 未核实 |
| GOAT-Bench arXiv:2404.06609 | 181 个 HM3DSem 场景、312 个物体类别、68 万 episode（145 个训练场景每个 5k episode，36 个验证场景每个 10 episode） | 真实数据位置未核实。`HKBU-NLP/GOAT-Bench` 是另一个仓库（8 文件，3.64 GB），与本基准无关 | HM3D 需申请 | Habitat | 多模态目标（图像、物体、语言）的终身导航，不是路径指令；SR 与 SPL | 未核实 |
| IndoorUAV arXiv:2512.19024；`https://huggingface.co/datasets/ml-resources/Indoor_UAV` | 从 Habitat 筛选 1,000 个以上室内场景（MP3D、Gibson、HM3D、ProcTHOR）；轨迹 CSV 在 `B/meta/indoor_uav/` | HF:1,384.96 GB，66 文件，中位 14.65 GB | apache-2.0，`gated=False`；场景本身需要 MP3D/HM3D 等授权 | Habitat 渲染 | 双尺度动作：前进 0.15 m 与 0.9 m；转向 3° 与 15°。VLA 任务：位置误差 <0.5 m 且偏航差 <π/4；VLN 任务：<2 m | 未核实 |
| SAGE-Bench（见 1.1） | 在 InteriorGS 上做 VLN 评测 | 同 1.1 | 同 1.1 | Isaac Sim 5.0+ | 指标 CSR/ICP/PS | 未核实 |
| VLN-PE（见 1.1） | 物理具身 VLN | 同 1.1 | 同 1.1 | Isaac Sim | 同 1.1 | 未核实 |

### 1.3 无人机 VLN 与语言飞行

最后一列是我对 100 g 低空室内外微型机的适用性判断。

| 条目与链接 | 规模 | 动作粒度 | 成功判据 | 离线图像与体量 | 许可 | 适用性判断 |
|---|---|---|---|---|---|---|
| UAV-Flow Colosseo arXiv:2505.15725；`https://huggingface.co/datasets/wangxiangyu0814/UAV-Flow`、`.../UAV-Flow-Sim` | 真实世界基准，仿真集 UAV-Flow-Sim 在 UE 校园场景，经 UnrealCV 控制 | 未核实 | 未核实 | HF:UAV-Flow 258.01 GB，UAV-Flow-Sim 35.93 GB | 许可字段为空 | 仿真端需 UE 渲染；只借鉴任务形式 |
| OpenFly arXiv:2502.18041 | 100k 轨迹，18 个场景，4 种渲染引擎 | 离散动作，Forward 为 3/6/9 m | 停在目标 20 m 内 | DataGen 124.37 GB（gated=auto） | `OpenFly` apache-2.0 | 尺度是城市，不适用 |
| AerialVLN arXiv:2308.06735；`https://huggingface.co/datasets/LeonMiao/AerialVLN` | 25 个城市级场景，870 多种物体，平均路径长 661.8（论文原文单位 units），每条指令约 83 词（原文 up to 83 words） | 未核实 | 停在目标 20 m 内（SR），OSR 同为 20 m | HF 仓库 0.39 GB（7 文件，仅标注）；仿真器另配 | apache-2.0 | 城市尺度，不适用 |
| TravelUAV / OpenUAV arXiv:2410.07087；`https://huggingface.co/datasets/wangxiangyu0814/TravelUAV`、`.../TravelUAV_env`、`.../TravelUAV_data_json` | UE4 加 AirSim，22 个场景、89 个物体；12,149 条轨迹，短于 250 m 为 easy，长于 250 m 为 hard | 连续轨迹，不拆离散动作 | NE 等；半径未核实 | HF:TravelUAV 482.70 GB，env 83.73 GB，data_json 0.20 GB | 许可字段为空 | 体量与渲染都不适用 |
| CityNav arXiv:2406.14240 | 真实城市点云；飞行高度随机取 100 至 150 m | 步长 5 m 取自任务书，本轮未复核 | 20 m 取自任务书，本轮未复核 | 未核实 | 未核实 | 高度与尺度不适用 |
| UAV-ON arXiv:2508.00288；`https://huggingface.co/datasets/Kyaren/UAV-ON-envs-train`、`.../UAV-ON-envs-test`、`.../UAV-ON-dataset` | 14 个 UE 环境；训练 10,000 个 episode，覆盖 10 个环境；目标在起点 50 units 半径内 | 例如 "move forward 3 units"、"rotate left 60 degrees" | 停在目标 20 units 内 | HF:train 26.80 GB，test 44.05 GB，dataset 0.01 GB | cc-by-4.0 | 需要 UE 二进制，不适用 |
| AVDN arXiv:2205.12219 | 对话式航拍导航，6,269 条子轨迹 | 未核实 | 未核实 | 未核实 | 未核实 | 基于卫星图，不适用 |
| CognitiveDrone arXiv:2503.01378；`https://huggingface.co/datasets/ArtemLykov/CognitiveDrone_dataset` | Gazebo 加 ArduPilot 基准，8,000 多条仿真轨迹，3 类任务（人物识别、符号理解、推理） | VLA 以 10 Hz 出 4D 动作，推理模块约 2 Hz | 以成功率评测，半径未核实 | HF:2.47 GB，251 文件 | 许可字段为空 | 近距离认知任务，可借鉴任务设计，场景不适用 |

### 1.4 避障飞行基准与难度度量

| 条目 | 难度或性能指标与公式 | 仿真器 | 本机可行性 |
|---|---|---|---|
| AvoidBench arXiv:2301.07430 | 环境指标两个：Traversability，Relative Gap Size（RGS，公式见 3.3，要求不小于 1）。性能指标七个：Success Rate（SR=T_finished/T_whole）、Path Optimality（PO=(d_trav-d_min)/d_min×100%）、Energy Optimality、Average Goal Velocity（AGV=d_min/t_trial）、Mission Progress（MP）、Processing Time、Contrast Factor。Traversability 公式未核实 | RotorS 加 Unity3D | 需 Unity 渲染，本机不可行；只借指标 |
| FlightBench arXiv:2406.05687 | 三个任务难度指标 TO、VO、AOL，公式见 3.3；成功为在 1.5 m 半径内到达目标且无碰撞；样例难度表：Forest TO 0.76 至 0.92、VO 0.30 至 0.60、AOL 7.64e-4 至 5.68e-3；Maze TO 1.42 至 1.54、VO 0.51 至 1.39、AOL 1.36e-3 至 0.61；MW TO 1.58 至 1.81、VO 0.55 至 1.13、AOL 0.08 至 0.94。感知距离 R 与权重 m_j 的取值未核实 | Flightmare 加 Gazebo 动力学 | 需渲染，本机不可行；只借指标 |
| Flightmare arXiv:2009.00563 | 由 Unity 渲染引擎和物理引擎两部分组成；DodgeDrone 挑战未核实 | Unity | 需渲染，不可行 |
| Aerial Gym | 没做 | 未核实 | 未核实 |

## 2. 本机可达性探测

### 2.1 hf-mirror 仓库探测

方法：`B/hfprobe.py`（单线程，429 时按 20 秒乘重试次数退避）取 `/api/datasets/<repo>` 与递归目录树；`B/probe/hf_head_tests.txt` 对 5 个文件做 HEAD（SAGE usdz 第一次路径写错返回 404，更正为 `InteriorGS_usdz/841150.usdz` 后返回 200）。汇总见 `B/probe/hf_summary.txt`。

| 仓库 | gated | 许可 | 文件数 / 总大小 | 单场景大小 | 访问结果 |
|---|---|---|---|---|---|
| spatialverse/InteriorGS | auto | other（Terms of Use） | 5,003 / 43.81 GB | PLY 16.9/37.8/65.1/144.1 MB（min/p50/p90/max） | 数据文件 HEAD 403 `GatedRepo`；README 可读 |
| spatialverse/SAGE-3D_InteriorGS_usdz | False | cc-by-nc-4.0 | 1,003 / 110.90 GB | 43.5/96.0/167.2/401.4 MB | 样本 `InteriorGS_usdz/841150.usdz` HEAD 200，46,488,397 B，与本地一致 |
| spatialverse/SAGE-3D_VLN_Data | False | cc-by-4.0 | 1,071 / 9.82 GB | 训练 JSON 2.1/9.8/14.7/35.7 MB | 下载成功 |
| spatialverse/SAGE-3D_Collision_Mesh | False | cc-by-nc-4.0 | 1,003 / 39.42 GB | 1.4/34.1/74.4/207.6 MB | 下载成功，格式 PXR-USDC |
| ai4ce/wanderland | False | apache-2.0 | 5,071 / 1,149.74 GB | `scene.usdz` 209.0/327.0/487.7 MB | 样例 `3dgs.ply` HEAD 200，427,094,926 B；`episodes.json` 22,148 B 已取 |
| ai4ce/wanderland-preview | False | apache-2.0 | 128 / 1.34 GB | 未统计 | 目录树可读 |
| GaussianWorld/scene_splat_7k | auto | cc-by-sa-4.0 | 12,915 / 5.34 GB（12,888 个为 JSON） | 不含高斯 | README 可读，目录树可读 |
| GaussianWorld/scannetpp_v2_mcmc_3dgs | auto | other（scene-splat-7k-license） | 2,870 / 338.24 GB | 354.0 MB | PLY HEAD 403 `GatedRepo` |
| GaussianWorld/interior_gs_preprocessed | auto | other（interiorgs-terms-of-use） | 7,008 / 21.95 GB（`.npy`；文件数 train 5,600、val 700、test 700） | 若每场景 7 个文件，则对应 1,000 个场景、均值 21.9 MB（推断，未核实） | 未取文件，需同意 InteriorGS 条款 |
| DL3DV/DL3DV-GS-960P | auto | 空 | 13,880 / 4,251.24 GB | 最大 zip 4.17 GB | zip HEAD 403 `GatedRepo` |
| IPEC-COMMUNITY/OpenFly | False | apache-2.0 | 目录树被 429 截断，未取全 | 未核实 | API 可达 |
| escontra/gauss_gym_data、escontra/gauss_gym_arkit | False | apache-2.0；空 | 120 / 3.25 GB；30,031 / 917.00 GB | gauss_gym_data 最大 splat.ply 659.7 MB；arkit 最大 426.4 MB | 目录树可读 |

hf-mirror 的行为：`/resolve/` 返回 307，需要 `curl -L`；连续多线程调用目录树会触发 HTTP 429，单线程加退避后可完成；`?author=` 列表接口返回 HTML 429 页面，没有使用。

### 2.2 GitHub 镜像前缀

测试脚本 `B/ghmirror_test.sh`，输出 `B/probe/gh_mirrors_round1.txt`（raw 文件）和 `B/probe/gh_mirrors_round2.txt`（ls-remote 与 archive）。clone 测试目标是 `Galery23/SAGE-3D_Official`（clone 后 7.2 MB），用 `git clone` 的浅克隆模式（depth 取 1）。

| 镜像前缀 | raw 文件（aerial_gym README，9,794 B） | `git ls-remote` | `archive/refs/heads/main.zip`（2,994,395 B） | 浅克隆 `git clone`（depth=1） | 结论 |
|---|---|---|---|---|---|
| `https://ghfast.top/` | 200，0.91 s | 成功 | 200，5.66 s | 成功，7 s | 可用，首选 |
| `https://githubfast.com/`（替换主机名，`https://githubfast.com/<org>/<repo>.git`） | 主页 HTML 200，7.18 s | 成功 | 200，8.10 s | 成功，24 s | 可用 |
| `https://gh-proxy.com/` | 200，1.57 s | 成功 | 200，19.0 s | 成功，36 s | 可用，较慢 |
| `https://ghproxy.net/` | 200，5.98 s | 25 s 超时无输出 | 25 s 内只收到 97,687 B，未完成 | 未测 | 小文件可用，git 与大文件不可靠 |
| `https://kkgithub.com/` | 000 | 证书校验失败（`server certificate verification`） | 000 | 未测 | 不可用 |
| `https://gitclone.com/github.com/` | 502 | 无输出 | 500 | 未测 | 不可用 |
| `https://mirror.ghproxy.com/` | 20 s 超时 | 无输出 | 000 | 未测 | 不可用 |
| `https://github.moeyy.xyz/` | 20 s 超时 | 未测 | 未测 | 未测 | 不可用 |
| `https://hub.gitmirror.com/` | 000 | 无法连接 | 000 | 未测 | 不可用 |

另一个数据点：`pip config list` 显示全局 index 为 `https://pypi.tuna.tsinghua.edu.cn/simple`，`curl https://pypi.org/simple/usd-core/` 在 8 s 内返回 200。本轮没有执行任何 `pip install`。

### 2.3 样例下载与 PLY 头检查

**样例**：SAGE 场景 841150（住宅），目录 `D/pubdata/sage3d/sample/`，合计 205 MB（`du -sh`），低于 500 MB 上限。

| 文件 | 字节数 | 说明 |
|---|---|---|
| `841150.usdz` | 46,488,397 | 与 HEAD 的 content-length 相同 |
| `841150_collision.usd` | 57,107,977 | 与目录树大小相同；魔数 `PXR-USDC`，没解析 |
| `train_trajectories_overall_0590_841150.json` | 13,054,819 | 与目录树大小相同；104 条轨迹，1,966 条指令（均值 13.9 词，中位 14，最大 28） |
| `converted/841150.ply` | 97,687,491 | 本轮由 `B/pilot/usdz_to_ply.py` 生成 |

**PLY 头与读取检查**：生成的 PLY 为 `binary_little_endian 1.0`，393,895 个顶点，62 个 `property float`（`x y z nx ny nz f_dc_0..2 f_rest_0..44 opacity scale_0..2 rot_0..3`），未压缩。`read_gaussian_ply` 读取成功，`sh_degree=3`，不透明度范围 0.0007 至 1.0，最大轴尺度超过 3 m 的高斯有 19 个（0.005%），渲染器按 `max_scale_m=3.0` 去掉后剩 393,876 个。不透明高斯中有 13.2% 落在同一个 5 cm 的 z 区间，地板在 z≈-0.05 m，与 InteriorGS README 的 (Right, Back, Up)、米制一致。

**NuRec 解码要点**（`B/pilot/usdz_to_ply.py`，25 行）：

- zip 内 `<id>.usdz.nurec` 先 `gzip.decompress`，再 `msgpack.unpackb`，取 `nre_data.state_dict` 中 `.gaussians_nodes.gaussians.` 前缀下的 `positions`、`rotations`、`scales`、`densities`、`features_albedo`、`features_specular`，数组为 fp16 字节串，用 `.shape` 项还原形状。
- `features_specular` 的布局是 [N, 15 个系数, 3 个通道]，INRIA 的 `f_rest` 是 [N, 3 个通道, 15 个系数]，写 PLY 前必须转置。第一版没有转置，渲染出现粉绿色斑块（`B/pilot/render_sh_compare.png` 的第一行）；转置后墙面颜色正常（`B/pilot/render_841150.png`）。
- `msgpack` 只在 `/opt/conda/bin/python3` 里有，3dgs-venv 里没有。接入时要么用系统 python 跑这一步，要么在 venv 里装 `msgpack`（本轮没装）。
- 位置是 fp16，在 8 至 16 m 范围内量化步长约 8 mm，16 至 32 m 约 1.6 cm。

**渲染检查**：`B/pilot/render_test.py` 用 `sim.3dgs.renderer.make_renderer` 读 `B/pilot/scene_841150.json`（`nav_to_world` 为绕 z 轴 180°），从 6 条 SAGE 轨迹起点以 z=1.2 m 出图。渲染器初始化 3.8 s，首帧 953 ms，之后 2 至 3 ms 每帧；深度中位数 2.45 至 3.26 m。

![841150 渲染](/root/rivermind-data/uav_rgb/tmp/phase12-20261110/bench/pilot/render_841150.png)

**SAGE 轨迹坐标系到 PLY 坐标系**：

- 官方 `Data/template.usda`（`B/gh/template.usda` 第 115 至 123 行）给 `gauss` 节点加了 `rotateXYZ=(-90,0,0)`；usdz 内 `gauss.usda` 自带 `xformOp:transform` 矩阵 [[-1,0,0],[0,0,-1],[0,-1,0]]（行向量约定）。手算复合后 (x,y,z) 变为 (-x,-y,z)，即绕 z 轴转 180°。这一步是我的手算，没有在 USD 库里验证。
- 数据检验（`B/pilot/frame_test4.py`、`frame_test5.py`、`frame_test6.py`）：104 条轨迹（z=0.5 m）的点，统计在 xy 平面距 1.1 至 1.3 m 高度层高斯（不透明度 >0.5）0.15 m 以内的比例。零平移时，旋转 0°、90°、180°、270° 依次为 25.3%、24.4%、11.3%、29.8%。180° 的最优平移是 (0.05, 0.00) m，比例 10.4%；其他三个角度要平移 1.25 m 以上才降到 11.9% 至 20.2%。天花板层（z 为 2.3 至 3.6 m，不透明度 >0.3）高斯的 xy 在 p1 至 p99 范围内为 x∈[-3.9, 5.8]、y∈[-11.6, 15.0]；轨迹为 x∈[-5.4, 3.4]、y∈[-10.1, 11.1]，不旋转时 x 下界越出天花板层范围，转 180° 后为 x∈[-3.4, 5.4]、y∈[-11.1, 10.1]，全部落入。
- 判断：绕 z 轴 180°、无平移是现有证据下最优的假设，仍有约 10% 的轨迹点离 1.2 m 高度的高斯过近，原因没有查清（可能是漂浮高斯、窗帘等薄物体，或配准残差）。要确认需要 InteriorGS 的 `occupancy.json` 与 `structure.json`（gated），或在 Isaac Sim 里对照。

**接入缺口：对照 `tools/scene_import_wanderland.py` 的步骤**

| 步骤 | Wanderland importer 现状 | InteriorGS（SAGE usdz）需要 | 新增量 |
|---|---|---|---|
| 下载 | 从 `ai4ce/wanderland-preview` 取文件 | `curl -L` 取 `InteriorGS_usdz/<id>.usdz` 与 `Trajectory_train/<id>/*.json`，走 hf-mirror | 约 20 行 |
| 格式转换 | SOG 或 `.splat` 转 INRIA PLY（`tools/splat_to_ply.py`） | NuRec 转 INRIA PLY，含 `f_rest` 转置 | `usdz_to_ply.py` 25 行，已写好 |
| 尺度 | 米制（LiDAR） | 米制，不需要缩放；`nav_to_world` 不允许缩放，本来也用不到 | 0 |
| 对齐与地面 | 用 COLMAP 上方向加局部 LiDAR 地面求 `nav_to_world`，`ground_z` 取地面 10% 分位 | 已是 z 向上，地面取 z 直方图主峰（样例约 -0.05 m）；`nav_to_world` 取绕 z 轴 180°（待核对） | 约 30 行 |
| 碰撞源 | LiDAR 点云（`collision.source="points"`），因为官方碰撞网格只覆盖相机高度上下 1.5 m | 方案 A：`source="gaussians"`，本轮已跑通；方案 B：InteriorGS `occupancy.json`（gated，没取）；方案 C：SAGE 碰撞 `.usd`（需 `pxr`，没装） | 方案 A 为 0 行 |
| 飞行高度与切片 | `FLY_Z=2.0`，`z_slice=[1.8,2.2]`，`voxel_m=0.2` | 天花板在 z≈3.0 m（不透明高斯的 z 直方图在 1.8 至 5 m 内主峰为 3.00 至 3.05 m，占 6.8%），取 `FLY_Z=1.2`，`z_slice=[1.0,1.4]`，`voxel_m=0.1` | 配置项 |
| 路线 | 手写 6 条，`planner.route_check` 回填 `max_steps` | 取 SAGE 轨迹的首点与末点为 spawn/waypoint，用 `route_check` 过滤，回填 `max_steps` | 约 60 行 |
| 相机 | 320×240，60° | 沿用 | 0 |

**路线可行性（`B/pilot/route_check_pilot.py`，输出 `B/pilot/route_check_pilot.json`）**：占据来自高斯（`z_slice=[1.0,1.4]`，`voxel_m=0.1`，包络 e1_mini，半径 0.102 m、半高 0.028 m）。

| 项 | 结果 |
|---|---|
| 路线数 | 104 |
| 起点无碰 / 终点无碰 / A* 有路径 | 102 / 90 / 71 |
| 最小净空 p10/p50/p90 | 0.2/0.2/0.4 m（体素量化 0.1 m） |
| 净空 ≥ 半径 + 0.0 / 0.1 / 0.2 / 0.3 m 的路线数（共 71） | 71 / 16 / 11 / 6 |
| SAGE 原路径长度 p10/p50/p90/max（全部 104 条） | 5.7/9.2/16.3/21.6 m；≥10 m 的 40 条；≥20 m 的 2 条 |
| 我方 A* 路径长度 p10/p50/p90/max（71 条） | 3.1/10.0/43.4/62.3 m |
| 我方 A* 长度除以 SAGE 长度 p50/p90 | 1.12/2.98；不超过 1.3 倍的占 68% |
| 步数估计（0.25 m / 25° 口径）p10/p50/p90/max | 31/72/262/362 |
| 另外 4 个场景的 SAGE 路径长度 p50/max 与 ≥20 m 条数 | 839992（测试，5 条）9.8/13.2 m，0 条；840104（210 条）5.5/13.2 m，0 条；839938（163 条）6.8/11.7 m，0 条；840007（104 条）7.6/17.4 m，0 条（`B/meta/sage3d_test_839992.json`、`B/meta/sage_big/*.json`） |
| 5 个场景合计 586 条 SAGE 路径长度 p10/p50/p90/max | 2.8/6.6/11.7/21.6 m；≥10 m 的 122 条（21%）；≥20 m 的 2 条 |

步数估计的公式沿用 `planner.route_check` 的结构：`ceil(1.25 × (路径长 / 0.25 + 总转向 / 25 + 8))`。仓库内原常数是 0.15 m 与 17°（`scene_import_wanderland.py` 第 541 行），这里换成用户给出的 0.25 m 与 25°，换算由 `B/pilot/route_check_pilot.json` 的 `path_m` 与 `turn_deg` 在命令行完成，未改仓库。

## 3. 结论与落地方案

### 3.1 推荐（主选 / 备选 / 对比口径）

**主选：InteriorGS 场景（经 `SAGE-3D_InteriorGS_usdz`）加 `SAGE-3D_VLN_Data` 轨迹与指令。**

- 它能回答用户的四点诉求：
  - 更复杂的场景：多房间、家具、门洞、走廊，不再是单根柱子。
  - 更长程：多段子任务。SAGE 单段路线中位 6.6 m（5 个场景合计），拼 3 至 5 段约 20 至 33 m，约 150 至 300 步，对应 LH-VLN 平均 150 步的量级（arXiv:2412.09082）和 5 分钟续航的 300 步。步数是粗估。
  - 更高精度：目标是具体物体附近，成功半径可分层（2.0 / 1.0 / 0.5 m，见 3.2）。
  - 与开源对比：指令与指标同源于 SAGE-Bench。
- 许可动作：取 usdz 与 VLN JSON 不需要用户操作。若要 `occupancy.json`、`structure.json`、`labels.json` 和原版压缩 PLY，用户需要在 `https://huggingface.co/datasets/spatialverse/InteriorGS` 页面同意 Terms of Use，并提供 HF token（镜像是否转发 token 本轮没测）。
- 风险：非商业许可；单段路线偏短（5 个场景 586 条中位 6.6 m，仅 2 条 ≥20 m，所以长程只能靠拼接）；坐标系是推断；净空窄（见 0 与 2.3）；3DGS 在 1.2 m 高度的漂浮高斯可能产生假障碍；InteriorGS 渲染图与真机相机图的域差距没有评估，未核实。

**备选 1：Wanderland 全量库。**

- 价值：已有 importer，零新增格式代码；样例场景 18 条点目标 episode 的测地距离 30.7 至 75.4 m（p50 39.5 m），落在 20 至 60 m 范围内；有 LiDAR 点云作碰撞真值。如果「20 至 60 m 的单段长路线」是硬要求，Wanderland 比 SAGE 直接。
- 缺点：指令是固定的一句话，没有语言多样性；goal radius 3.0 m，精度要求低，可自行收紧；单场景 5,000 至 10,000 m²，偏室外。
- 许可动作：无（apache-2.0，ungated）。体量：每场景 `scene.usdz` 中位 327 MB，样例 `3dgs.ply` 427 MB，按 ≤500 MB 的规则单次最多拿 1 个。论文 530 场景与目录树 335 个场景目录的差异未核实。

**备选 2：SceneSplat-7K 系列中的 ScanNet++ v2、Replica、Matterport3D。**

- 价值：场景类型多（956 + 18 + 90 个），PLY 是 INRIA 格式，`read_gaussian_ply` 可直接读，每场景 354 MB。
- 缺点：没有路线、指令、碰撞体；要自己用 `planner.route_check` 出路线，自己造指令。Matterport3D 场景每个约 1,200 万高斯，体量大（按 236 B 每高斯估算约 2.8 GB 每场景，估算值）。
- 许可动作：用户在 `https://huggingface.co/datasets/GaussianWorld/scannetpp_v2_mcmc_3dgs` 等仓库页面点同意（`gated=auto`），并提供 token；底层 ScanNet++ 的使用条款是否另有要求，未核实。

**对比协议：SAGE-Bench 子集加 LH-VLN 多段指标，动作粒度用我们的 0.25 m / 25°。**

- 数据：SAGE 的 Scene-Unseen / Trajectory-Unseen / Instruction-Unseen 三个测试划分，测试 JSON 139 个文件共 0.04 GB；场景取其中 10 个左右，避免体量过大（45 个测试场景按每场景 96 MB 加 2.1 倍 PLY 估算约 13 GB，估算值）。
- 指标：SR、SPL、NE，再加 SAGE 的 CSR/ICP/PS，再加 LH-VLN 的 ISR/CSR/CGT，定义和计算见 3.3。
- 对手：官方评测器要 Isaac Sim 5.0+，本机跑不了。能直接做的对手是随机、纯前进、A* 规划器（上界）和我们的小 CNN。和论文表格里的数字直接比较不可行，因为动作粒度不同。要比开源模型，需要把它们放进我们的 gsplat 闭环（单目 RGB 加指令出动作），工作量未评估。
- 公开权重（hf-mirror 模型搜索，`B/probe/hf_models_search.txt`，只取了搜索列表，没有逐个取 info、大小和 gated 状态）：

| 方法 | 搜索命中的模型 id | 下载量 | 备注 |
|---|---|---|---|
| NaVILA（SAGE 论文中出现 25 次） | `a8cheng/navila-llama3-8b-8f`、`a8cheng/navila-qwen2-7b-64k-64f` | 520、23 | 8B 与 7B 量级，显存需求未核实 |
| InternVLA-N1 | `InternRobotics/InternVLA-N1-DualVLN`、`InternRobotics/InternVLA-N1-System2` | 477、207 | 是否对应 VLN-PE 的基线未核实 |
| StreamVLN | `mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3` | 495 | 社区上传，官方来源未核实 |
| OpenFly | `IPEC-COMMUNITY/openfly-agent-7b` | 533 | 城市尺度的离散动作，和我们的步长不匹配 |
| TravelUAV | `wangxiangyu0814/traveluav-traj-model` | 0 | 下载量为 0 |
| NaVid（SAGE 论文中出现 16 次） | 搜索 `navid` 无对应结果 | 无 | 权重位置未核实 |

  这些模型输入的动作空间和我们的 4 个离散动作不同，接入需要写动作映射层。

**需要用户决定的事**：

1. 是否接受 CC-BY-NC-4.0 / InteriorGS Terms 的非商业限制。
2. 是否愿意去 huggingface.co 点同意 InteriorGS（换取 occupancy、structure、labels 和坐标系对照）。
3. 成功半径分层采用 2.0 / 1.0 / 0.5 m 是否合适（这是我的设计，不是公开标准）。
4. 20 至 60 m 的任务用多段拼接实现，还是以 Wanderland 点目标为主、InteriorGS 为辅。
5. 是否允许在 3dgs-venv 里安装 `msgpack`（转换步骤用）和 `usd-core`（解析 SAGE 碰撞体，可选）。

### 3.2 试点接入方案（≤ 10 GB）

**文件与目录**

- `D/pubdata/sage3d/usdz/<id>.usdz`：原始下载。
- `D/pubdata/sage3d/vln/<id>/train_trajectories_overall_*.json`：轨迹与指令。
- `D/3dgs-scenes/sage3d_<id>/{scene.ply, scene.json}`：与现有 `bicycle`、`callinstrasse`、`wanderland_1MlUucug` 的目录习惯一致。

**规模估算（估算值，仅 usdz 与 PLY 的比例 2.10 来自样例 97.69 MB / 46.49 MB）**：16 个住宅场景按 96.0 MB 加 4 个公共空间按 167.2 MB，usdz 共 2,205 MB；PLY 约 4,630 MB；VLN JSON 20 个按 9.8 MB 共 196 MB；合计约 7.0 GB。转换后可以删 usdz，剩约 4.8 GB。数据盘当前剩 15 GB（`df -h`），7.0 GB 可行。

**转换与接入**

1. 下载：hf-mirror 加 `curl -L`，单线程，遇 429 退避。
2. 转换：`usdz_to_ply.py`（已写，含 `f_rest` 转置）。
3. 新增 `tools/scene_import_sage3d.py`，约 170 行，仿照 `scene_import_wanderland.py`：写 `scene.json`（`nav_to_world` 取绕 z 轴 180°，`ground_z` 取 z 直方图主峰，碰撞取高斯源，`z_slice=[1.0,1.4]`，`FLY_Z=1.2`，相机 320×240、60°），从 VLN JSON 取路线，`route_check` 过滤后回填 `max_steps`。
4. 复用的函数：`scene.read_gaussian_ply`、`scene.save_scene`、`scene.load_scene`、`collision.from_scene`、`planner.OccPlanner`、`planner.route_check`、`renderer.make_renderer`、`scene_import_wanderland.py` 里的探针与冒烟步骤。
5. 路线过滤规则：A* 有路径，净空不小于机体半径加 0.1 m，A* 长度不超过 SAGE 长度的 1.3 倍。样例场景满足的有 15 条（16 条净空合格中有 15 条长度合格，`B/pilot/route_check_pilot.json`）。20 个场景估计可得到 100 至 300 条单段路线，这是外推，未核实。

**路线到动作的映射**

- SAGE 的点间距是 0.05 m，只取首点和末点作为 spawn 与 waypoint；中间路径由我们的 A* 加现有 oracle 控制器重新生成，动作为每 1 s 一步，0.25 m 前进或约 25° 转向。
- 指令字符串存为路线的元数据（`routes[].note` 或单独的 `instructions.jsonl`），是否作为模型输入由用户决定。
- 长程任务：把同一场景内 3 至 5 条首尾相接的单段路线串成一个任务。样例场景 841150 单段步数估计 p50 72，串 3 段约 216 步，在 5 分钟续航的 300 步以内。其余 4 个场景路线更短，步数估计没有单独计算。首尾相接的路线是否足够多，没有统计。

**成功半径建议**：粗 2.0 m（IndoorUAV 的 VLN 判据，也是仓库 Wanderland 现用值），标准 1.0 m，精 0.5 m 加偏航差 <π/4（IndoorUAV 的 VLA 判据）。FlightBench 用 1.5 m。UAV-ON、OpenFly、AerialVLN 的 20 m 或 20 units 与我们的尺度不符。

### 3.3 长程任务评测指标建议

数据来源：`oracle.jsonl` 每行有 `step`、`timestamp_ns`、`pose_gt{x,y,z}`、`action_start_ns`、`action_end_ns`；`actions.jsonl` 每行有 `step`、`action_expert`、`action_executed`、`collision`（布尔）、`dyaw_deg`、`displacement_m`、`pose_before`、`yaw_before` 等（样例 `D/previews/live_episode_900/`）。目标点 g 和起点 s 取自 `scene.json` 的 `routes[]`。

| 指标 | 来源 | 定义与公式 | 从我们的文件怎么算 |
|---|---|---|---|
| SR（半径 r） | 通用定义，本轮未逐字核对原文 | 最终位置到目标距离不超过 r 记 1，否则 0 | `pose_gt` 末行到 g 的水平距离，r 取 2.0/1.0/0.5 m |
| OSR | 通用定义，同上；AerialVLN 论文使用 OSR（arXiv:2308.06735） | 轨迹上任一点到目标距离不超过 r 记 1 | `pose_gt` 全程到 g 的最小距离 |
| NE | 通用定义（arXiv:2410.07087 也报告 NE） | 最终位置到目标的距离 | 同 SR |
| SPL | 通用定义，同上 | SR 乘 l* 除以 max(p, l*)；l* 为最短路径长，p 为实际飞行路径长 | l* 取 `route_check` 的 `path_m`，p 取 `pose_gt` 相邻点距离之和 |
| nDTW | 通用定义，同上 | 预测轨迹与参考轨迹的动态时间规整相似度 | 参考轨迹取规划路径，d_th 取成功半径 |
| ISR（LH-VLN） | arXiv:2412.09082 式 2 | ISR = Σ_j Σ_i s_{j,i} / (M·N)，M 为任务数，N 为子任务数，s_{j,i} 为第 j 个任务第 i 个子任务是否成功 | 子任务即串联的单段路线，s 用 SR |
| CSR（LH-VLN） | arXiv:2412.09082 式 3 | CSR = Σ_j Σ_i s_{j,i}(1+(N-1)s_{i-1}) / (M·N²)；i=1 项的 s_0 取值论文正文未写，取 s_0=1 时全部成功得 CSR=1，这是推断 | 同上 |
| CGT（LH-VLN） | arXiv:2412.09082 式 4 | CGT = Σ_j Σ_i (P_i/P)·s_{j,i}(1+(N-1)s_{j,i-1}) / (M·N)，P_i 为第 i 个子任务的真值路径长，P 为总和 | P_i 取各段 `route_check.path_m` |
| CSR（SAGE，连续成功率） | arXiv:2510.21307 | CSR = (1/T)Σ_t s(t)，s(t)=1 当 t 时刻位置在参考路径加 r_tol 缓冲的走廊内且满足任务条件；`r_tol` 取值未核实 | 参考路径取规划路径，用 `pose_gt` 逐步判断；r_tol 自定 |
| ICP（SAGE） | arXiv:2510.21307 | ICP = (1/T)Σ_t c(t)，c(t)∈[0,1] 为碰撞强度 | 我们只有布尔 `collision`，取 c(t)∈{0,1}，即碰撞步占比 |
| PS（SAGE） | arXiv:2510.21307 | PS = 1 - (1/(T-1))Σ_{t=2}^{T} min(|Δθ_t|/π, 1)，Δθ_t=θ_t-θ_{t-1} | θ 由 `yaw_before` 加 `dyaw_deg` 得到 |
| 难度分桶（SAGE） | arXiv:2510.21307 | 路径 >29.0 m 为 long，<8.4 m 为 short；资产 >376 为 many，<184 为 few | 路径长取 `route_check.path_m`；资产数需要 `labels.json`（gated），暂用高斯数代替，需另行定标 |
| PO（AvoidBench） | arXiv:2301.07430 式 4 | PO = (d_trav - d_min)/d_min × 100% | d_trav 为 `pose_gt` 路程，d_min 为 `route_check.path_m` |
| AGV（AvoidBench） | arXiv:2301.07430 式 6 | AGV = d_min / t_trial | t_trial 取 `timestamp_ns` 末行减首行 |
| MP（AvoidBench） | arXiv:2301.07430 式 7 | MP = (1 - \|a·c/\|a\|²\|) × 100%，a 为起点到目标向量，c 为终点到目标向量 | 用 `pose_gt` 首末行与 g |
| RGS（AvoidBench） | arXiv:2301.07430 式 2 | RGS = (1/(d_drone·N)) Σ s(p(i),h(i))，障碍间最小间隙除以机体直径，不应小于 1 | 代理算法：取沿路径的 `occ.clearance` 乘 2 作为间隙宽度，再除以机体直径；与论文按泊松半径定义的环境级 RGS 不同 |
| TO（FlightBench） | arXiv:2406.05687 式 2 | TO = (1/N_T)Σ_{i=1}^{⌊N_T/2⌋} R/r_i，r_i 为沿引导路径的球形走廊半径升序排列，R 为感知距离（取值未核实） | r_i 取沿 A* 路径的 `occ.clearance`，R 自定 |
| VO（FlightBench） | arXiv:2406.05687 式 3 | VO = (1/N_V)Σ_i Σ_{j=1}^{M} m_j R/s_{ij}，视场分 M 份，s_{ij} 为各份内最近障碍距离，m_j 为中心权重大的权重（取值未核实） | gsplat 渲染自带深度图，分 M 个扇区取最小深度，需新增约 15 行 |
| AOL（FlightBench） | arXiv:2406.05687 式 4 | AOL = (1/L)Σ_i (exp(θ_i/(π/6)) - 1)，θ_i 为路径折角，L 为路径长 | `route_check` 只给总转向量，逐折角需新增约 15 行 |
| 成功判据（FlightBench） | arXiv:2406.05687 | 在 1.5 m 半径内到达目标且无碰撞 | SR 的 r=1.5 m 并要求 `collision` 全 False |

TO、VO、AOL 可以用来回答「柱阵任务是否太容易」：对现有柱阵路线和新场景路线分别计算，数值越大越难。FlightBench 样例表给出了量级（Forest 的 TO 为 0.76 至 0.92，Maze 的 AOL 最高 0.61）。柱阵与新路线的实际数值本轮没有计算。

### 3.4 不推荐的选项

| 选项 | 理由与数字 |
|---|---|
| TravelUAV / OpenUAV | HF:TravelUAV 482.70 GB 加 env 83.73 GB；UE4 加 AirSim；12,149 条轨迹的分档是 250 m（arXiv:2410.07087），尺度不符 |
| UAV-ON | HF:UE 环境 train 26.80 GB 加 test 44.05 GB；成功半径 20 units（arXiv:2508.00288） |
| OpenFly 全套 | HF:DataGen 124.37 GB 且 `gated=auto`；Forward 步长 3/6/9 m，成功 20 m（arXiv:2502.18041） |
| AerialVLN | 25 个 UE4 城市场景，成功 20 m（arXiv:2308.06735），需 UE 渲染 |
| CityNav | 飞行高度 100 至 150 m（arXiv:2406.14240），与 1 至 2 m 高度不符 |
| IndoorUAV | HF:1,384.96 GB；场景需要 MP3D/HM3D 等授权；Habitat 渲染；动作 0.15/0.9 m 双尺度，0.9 m 大于我们的 0.25 m 步长（arXiv:2512.19024）。可借鉴的是 VLA 的 0.5 m 加偏航判据 |
| UAV-Flow | HF:258.01 GB 加仿真 35.93 GB，UE 校园场景 |
| LHPR-VLN 数据本体 | HF:91.36 GB，场景为 HM3D（需申请），Habitat 渲染。可借鉴的是 0.25 m / 30° 动作设定和 ISR/CSR/CGT 指标 |
| GOAT-Bench | HM3DSem 场景，Habitat；任务是多模态目标导航，不是路径指令；HF 上的同名仓库是无关数据集 |
| VLN-CE、RxR-CE | MP3D 授权与 Habitat 渲染；可借鉴 0.25 m / 15° 动作设定 |
| AvoidBench、FlightBench 的仿真器 | Unity 或 Flightmare 加 Gazebo，需要渲染，本机无 GL/EGL；只借指标 |
| SAGE-Bench 官方评测器 | 需要 Isaac Sim 5.0+（`B/gh/Galery23_SAGE-3D_Official_README.md` 第 98 至 99 行） |
| SceneSplat-49K | 主仓只有 14 个统计文件，高斯位置未核实，且 gated |
| GaussGym ARKit、DL3DV-GS-960P | HF:gauss_gym_arkit 917.00 GB；DL3DV-GS-960P 4,251.24 GB 且 gated |

## 做了什么

- 读完 `BRIEF_COMMON.md` 和仓库里的 `sim/3dgs/scene.py`、`tools/scene_import_wanderland.py`，以及 `collision.py`、`planner.py`、`renderer.py` 的相关函数。
- 用 `B/hfprobe.py` 对 28 个 HF 仓库取 info 与目录树，用 `B/summarize_meta.py` 汇总为 `B/probe/hf_summary.txt`；用 HEAD 测 5 个文件的可达性。
- 缓存 26 篇相关 arXiv 全文到 `B/cache/`（另有 1 篇误取的 2406.05344 与本题无关），用 `B/bf2.py` 按正则抽取规模、动作、成功半径、指标公式；输出在 `B/probe/facts_batch{A,B,C,D,E}.txt`。
- 测试 9 个 GitHub 镜像前缀的 raw、ls-remote、archive，并对 3 个可用前缀做浅克隆（depth=1）。
- 下载 SAGE 场景 841150 的 usdz、碰撞体和轨迹 JSON 共约 117 MB（加转换产物后该目录为 205 MB），写 `usdz_to_ply.py` 完成 NuRec 到 INRIA PLY 的转换，发现并修正 `f_rest` 布局，用 `read_gaussian_ply` 和 `GsplatRenderer` 渲染验证。
- 检验 SAGE 轨迹坐标系与 PLY 坐标系的对应，跑 `planner.route_check` 得到 104 条路线的可行性与步数估计。
- 取得 Wanderland 一个场景的 `episodes.json` 并统计测地距离与目标半径。
- 下载另外 3 个 usdz 最大场景的 VLN JSON（共约 40 MB，`B/meta/sage_big/`），统计 5 个场景 586 条 SAGE 路线的长度分布。
- 用 hf-mirror 模型搜索列出 NaVILA、InternVLA-N1、StreamVLN、OpenFly、TravelUAV 的公开权重（`B/probe/hf_models_search.txt`）。

## 没做什么（每条带原因）

- 没下载 InteriorGS 原版 PLY、`occupancy.json`、`structure.json`：`gated=auto`，镜像返回 403 `GatedRepo`，需要用户在 huggingface.co 点同意并提供 token。
- 没取 SceneSplat-7K 与 49K 的高斯文件：主仓只含位姿 JSON 与统计 CSV，高斯在独立的 gated 仓库，HEAD 返回 403。
- 没解析 SAGE 碰撞体 `.usd`：`PXR-USDC` 需要 `pxr`，三个 python 解释器都没有，安装属于环境变更，不在只调研的范围内。
- 没在 Isaac Sim 里核对 SAGE 轨迹与 3DGS 的坐标对应：本机没有 Isaac Sim 与 GPU 图形栈。改用高斯层统计检验，结论只到「现有证据下最优」。
- 没核实 VLN-CE、RxR-CE、VLN-PE、SAGE-Bench、UAV-Flow、AVDN、CognitiveDrone 的成功半径：正则抽取未命中，时间盒内没有逐篇精读。
- 没核实各公开基线权重的 gated 状态、文件大小、显存需求，以及它们与对应基准的官方关系：只做了模型搜索列表，没有逐个请求 info 与目录树。
- 没复核 CityNav 的 5 m 步长和 20 m 成功半径：这两个数取自任务书，本轮抽取只确认了 100 至 150 m 的飞行高度。
- 没取全 OpenFly 目录树、没读到 IndoorUAV 的 README：hf-mirror 429 限流与 HTTP 521。
- 没核实 Aerial Gym、DodgeDrone、Flightmare 的容器可行性：没有抓取 Aerial Gym 论文，Flightmare 只确认了 Unity 渲染引擎。
- 没核实 Wanderland 论文 530 场景与目录树 335 个场景目录的差异，没核实 RoboTidy、GN0 的发布位置：时间盒内没有逐个请求。
- 没安装 `msgpack` 与 `usd-core`：规则禁止系统安装，`pip` 默认指向清华镜像且 `pypi.org` 返回 200，但没执行安装。
- 没下载 Wanderland 的 427 MB `3dgs.ply`：已有同源场景，且单次下载接近 500 MB 上限。
- 没写 `3dgs-scenes/` 下的新场景，没改仓库任何文件：任务只做调研与探测。
- 没计算柱阵任务的 TO/VO/AOL：缺少 R、m_j 的取值，且本轮不改仓库。
- 没读 `tools/scene_import_mipnerf.py`：与本轮结论无关。

## 验证到哪一步

| 命令或检查项 | 结果 |
|---|---|
| `B/hfprobe.py` 批次 1 至 6，输出 `B/probe/hfprobe_batch*.txt` | 28 个仓库的 info 与目录树写入 `B/meta/`（`hf_summary.txt` 28 行）；`IPEC-COMMUNITY/OpenFly` 因 429 目录树被截断 |
| `B/probe/hf_head_tests.txt`：InteriorGS PLY、scannetpp_v2 PLY、DL3DV-GS zip | 均为 403 `GatedRepo` |
| 同文件：SAGE usdz 841150、Wanderland `3dgs.ply` | 200；46,488,397 B 与本地一致；427,094,926 B |
| `B/ghmirror_test.sh` 加浅克隆（depth=1，`B/probe/gh_mirrors_round2.txt`） | ghfast.top、githubfast.com、gh-proxy.com 可用；ghproxy.net 不稳定；其余 5 个不可用 |
| `ls -l D/pubdata/sage3d/sample/` 与 `du -sh` | 三个下载文件大小与目录树一致；合计 205 MB |
| `B/pilot/usdz_to_ply.py` 转换样例 | 393,895 高斯，62 个 float 属性，97,687,491 B |
| `read_gaussian_ply` 读取转换结果 | 成功，`sh_degree=3`，19 个高斯因尺度超 3 m 被渲染器剔除 |
| `B/pilot/render_test.py`（CUDA_VISIBLE_DEVICES 指定 GPU-dbc6435f-…） | 6 个视角渲染成功，2 至 3 ms 每帧；图 `B/pilot/render_841150.png` |
| `B/pilot/render_sh_test.py` | `f_rest` 不转置时出现粉绿色斑，转置后正常；对比图 `B/pilot/render_sh_compare.png`；这是目视判断，没有定量色差指标 |
| `B/pilot/frame_test4.py`、`frame_test5.py`、`frame_test6.py` | 绕 z 轴 180° 无平移最优（1.2 m 层 11.3% 对 24.4% 至 29.8%）；未与官方 occupancy 对照 |
| `B/pilot/route_check_pilot.py` | 104 条路线中 71 条有 A* 路径，16 条净空不小于半径加 0.1 m，6 条不小于半径加 0.3 m；SAGE 路径中位 9.2 m |
| 5 个场景 SAGE 路线长度统计（Python 读 JSON，不改任何文件） | 586 条，p50 6.6 m，最大 21.6 m，≥20 m 仅 2 条 |
| `B/probe/hf_models_search.txt`（模型搜索） | 搜索列表可取到 NaVILA、InternVLA-N1、StreamVLN、OpenFly、TravelUAV 的模型 id；NaVid 无命中；未逐个取 info |
| `B/bf2.py` 抽取与 `B/probe/facts_batch*.txt` | 规模、动作、半径、公式已核；半径与权重缺项见「没做什么」 |
| 写作风格自检：用 grep 检查任务书列出的禁用词、破折号与双连字符 | 无命中（Markdown 表格分隔行除外） |
| 仓库只读自检：`find R -newermt "2026-11-10 19:20" -type f` 排除 `.git` | 命中的文件在 `experiments/2026-11-10-*` 下，属于其他并行任务的分支；本任务没有改 `R`，没有提交，没有建分支 |
