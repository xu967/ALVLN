# 13｜源码来源、第三方许可、公开文献与引用纪律

## 资料来源与状态

| 文件 | 来源 | 使用方式 | 不能自动证明 |
|---|---|---|---|
| `frontend/` | 用户上传 `NavLab_Studio_UI_v3(1).zip` | 本项目唯一 UI 主源码 | 任何未开发 API 真能训练/仿真 |
| `docs/06_SIMULATION_ORCHESTRATION.md` | 用户新上传同名 MD | LANI/TravelUAV 自动仿真调度设计主依据 | 统一管理服务已在旧项目实现 |
| `references/snapshots/web_code_review_20261027.zip` | 用户既有 Aerial/LANI 代码快照 | 已有 Python 训练/推理/评测路径的参考 | C++/CUDA 学生已实现 |
| `legacy/V7_1_patched` | 原 V7.1 工程加 `review_v2.patch` | 原 dashboard、启动、训练进度/测试的已修复历史快照 | 已生成真实导航数据 |
| `references/snapshots/CF-Drone.zip` | 用户飞控源码 | 审计、数学核参考与未来 SIL | 已有定高/安全自动悬停 |
| `references/snapshots/OpenFly-Platform.zip` | 用户提供的 OpenFly 平台归档 | 可借鉴仿真数据格式/后期研究 | 与 Ubuntu24/RTX5090 即插即用 |
| `references/snapshots/code.zip` | 用户提供高飞团队多项目源码归档 | 离线参考 random_forest_sensing/mockamap/GCOPTER 等 | 可无差别转存到本项目、已兼容所有许可证 |
| `references/reviews/` | 用户交付评审/补丁/复现脚本 | 外部独立证据，供交叉质疑 | 所有评审意见都一定正确 |

**重要**：高飞/CF/Flix 相关源码可能涉及 GPL 或额外商业/品牌条件。对 `CF-Drone.zip` 的实际 LICENSE 文本应单独核验，不能因为另一个网页写 Apache-2.0 就覆盖本地 LICENSE。上游 GPL/NC/附加条款冲突应由发布前的实际权利审查解决。本交接包中保存的用户原始源码快照不等于获得公开再分发许可；**仅供授权研究和内部服务器使用**。

## 外部公开参考（来源链接，非效果保证）

- Gazebo Harmonic Ubuntu binaries：https://gazebosim.org/docs/harmonic/install_ubuntu/
- Gazebo Harmonic 版本与运行：https://gazebosim.org/docs/harmonic/
- PX4 Gazebo SITL x500：https://docs.px4.io/v1.18/en/sim_gazebo_gz/
- Sipeed LicheeRV Nano SG2002：https://wiki.sipeed.com/hardware/zh/lichee/RV_Nano/1_intro.html
- TPU-MLIR：https://github.com/sophgo/tpu-mlir
- SG200X SDK：https://github.com/milkv-duo/tpu-sdk-sg200x
- AerialVLN ICCV 2023：https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_AerialVLN_Vision-and-Language_Navigation_for_UAVs_ICCV_2023_paper.pdf
- OpenFly ICLR 2026：https://proceedings.iclr.cc/paper_files/paper/2026/hash/a6fcfd15cd01e4a550808c3e01f5583d-Abstract-Conference.html
- LightVLN arXiv：https://arxiv.org/abs/2610.05024
- AirAnchor arXiv：https://arxiv.org/abs/2609.08442
- LookasideVLN CVPR 2026：https://openaccess.thecvf.com/content/CVPR2026/html/Ning_LookasideVLN_Direction-Aware_Aerial_Vision-and-Language_Navigation_CVPR_2026_paper.html
- Query, Align, and Distill：https://arxiv.org/abs/2609.33097

论文前须核实在线出版时间、数据/代码版本、官方 split 和报告使用的传感器模态。不要把 `LightVLN Val-Seen 25.83%` 与自己的 Val-Unseen 混为同场对照，也不要将公开 LAG 的 RGB-D/其他输入设置的数字当作 RGB-only 公平下界。
