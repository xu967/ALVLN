# 11 月 24 日语言动作学习实验快照

按提交顺序保存三次实际运行的 `uav_rgb/train.py`。前缀训练、首个分岔训练和沿轨迹训练的入口文件均与训练产物逐字节核对。各提交里的训练入口和注册清单配套。数据包、R2 权重、BERT 与示教相机帧留在训练机的数据盘。

在原实验机复跑时，从对应数据盘目录调用其中的 `train.py`。入口从自身所在目录读取 `registration.json`，所需文件的位置和哈希见 `direct_dependencies.json`。Git 中的 `prefix_registration.json`、`sparse_registration.json` 和 `dense_registration.json` 分别对应三次训练目录中的同名登记文件 `registration.json`。本目录的 `registration.json` 是视觉特征解析器定位 R2 所用的登记。另一台机器复跑需先还原这些数据或调整登记路径。

最近一次提交保存指令停机对照，文件哈希与原始目录 `exp-vln-stop-head-20261124` 对应。样本归档和视觉编码器的依赖见 `stop_dependencies.json`。较早提交各自保存首次分岔及沿轨迹训练的完整入口。

区域语义复核将候选框裁图送给冻结模型自由命名，再按指令核验，使用上一轮固定的 30 条双帧样本。对应原实验目录 `exp-vln-stop-semantic-20261124`，登记和依赖见 `semantic_registration.json`、`semantic_dependencies.json`。
