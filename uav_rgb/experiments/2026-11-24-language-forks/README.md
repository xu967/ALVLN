# 11 月 24 日语言动作学习实验快照

按提交顺序保存三次实际运行的 `uav_rgb/train.py`。前缀训练、首个分岔训练和沿轨迹训练的入口文件均与训练产物逐字节核对。各提交里的训练入口和注册清单配套。数据包、R2 权重、BERT 与示教相机帧留在训练机的数据盘。

在原实验机复跑时，从对应数据盘目录调用其中的 `train.py`。入口从自身所在目录读取 `registration.json`，所需文件的位置和哈希见 `direct_dependencies.json`。Git 中的 `prefix_registration.json`、`sparse_registration.json` 和 `dense_registration.json` 分别对应三次训练目录中的同名登记文件 `registration.json`。本目录的 `registration.json` 是视觉特征解析器定位 R2 所用的登记。另一台机器复跑需先还原这些数据或调整登记路径。

最近一次提交保存指令停机对照，文件哈希与原始目录 `exp-vln-stop-head-20261124` 对应。样本归档和视觉编码器的依赖见 `stop_dependencies.json`。较早提交各自保存首次分岔及沿轨迹训练的完整入口。

区域语义复核将候选框裁图送给冻结模型自由命名，再按指令核验，使用上一轮固定的 30 条双帧样本。对应原实验目录 `exp-vln-stop-semantic-20261124`，登记和依赖见 `semantic_registration.json`、`semantic_dependencies.json`。

示教回放实验对应 `exp-vln-gallery-replay-20261124`。运行 `python train.py gallery_replay`。固定图库排除了八组查询路线及对应起点，图文和仅图像均答对 7/16 个动作，整对正确均为 0/8。逐集数据和图库开销见实验机的 `results.json`。

图文停机复核采用固定 CLIP ViT-B/32 与训练房间的 104 类词表，两张图中目标均列前五才保留原停机决策。开发五个目标上正确停机 0/5。入口的固定版本、输入与输出对应数据盘上的 exp-vln-stop-clip-20261124。

三个扇区的局部探索使用上一轮飞行基线的 default_runtime。在数据盘复制该基线并于副本根目录应用 sector_exploration/runtime.patch，把 visual_exploration.py 放到 sim/3dgs 后，按同目录 registration.json 的 arms 执行。两个实际运行源文件的 SHA256 在登记内。

整图深度对照来自 exp-vln-metric-frontier-839934-20261124。将 metric_frontier 目录的注册、校正和深度 worker 文件放入该实验目录，其内 runtime 从保存的 default_runtime 复制，应用 runtime.patch 后将 metric_frontier.py 放入 runtime/sim/3dgs。相机校准与策略输入在源码内固定。需要数据盘上的 Depth Anything V2 Hypersim ViT-S 权重及依赖。

重建飞行基线时，先取 GitHub 主线 4b376c3 的 uav_rgb 目录，从该目录应用 flight_base.patch，即可得到原 default_runtime 的 sim 下所有非测试 Python 与 JSON 源文件，逐个校验 flight_base_hashes.json。两个搜索方案的 runtime.patch 都以此版本为起点。固件、渲染资产、视觉模型和推理二进制依赖数据盘，整图方案的哈希见 metric_frontier/dependencies.json。

840865 跨房间的冻结距离与图文复核在 exp-vln-stop-transfer-score-20261124 的入口运行，两份注册资料见 transfer/。按原数据盘目录布局运行，近、错、远组各 4 个实例，冻结距离头的停机计数为 1、0、0。

目标短词对照有预先冻结和事后修正两个提交，训练入口分别与数据盘两个目录逐字节一致。命令、哈希与注册表见 noun_proposal/run_entries.json；原始指令及图像由实验机提供。

visible_approach 记录圈外可见起点诊断。还原 flight_base.patch 对应的运行时、应用该目录 runtime.patch，在原数据盘实验目录放置 approach.py、登记、标定与四个场景清单，按原位置提供模型和固件。visibility/fit.py 为内部可见性统计对照的原始源码，复跑需从数据盘恢复已核实训练清单、冻结图像与外部 CLIP 权重。
