# 11 月 24 日语言动作学习实验快照

按提交顺序保存三次实际运行的 `uav_rgb/train.py`。前缀训练、首个分岔训练和沿轨迹训练的入口文件均与训练产物逐字节核对。各提交里的训练入口和注册清单配套。数据包、R2 权重、BERT 与示教相机帧留在训练机的数据盘。

在原实验机复跑时，从对应数据盘目录调用其中的 `train.py`。入口从自身所在目录读取 `registration.json`，所需文件的位置和哈希见 `direct_dependencies.json`。Git 中的 `prefix_registration.json`、`sparse_registration.json` 和 `dense_registration.json` 分别对应三次训练目录中的同名登记文件 `registration.json`。本目录的 `registration.json` 是视觉特征解析器定位 R2 所用的登记。另一台机器复跑需先还原这些数据或调整登记路径。
