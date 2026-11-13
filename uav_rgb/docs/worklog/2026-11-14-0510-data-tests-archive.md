# 数据集、扩展测试与数据归档

- 时间：2026-11-14 05:20 CST
- 范围：公开数据准备、3DGS/SITL 回归与长航线实跑、`/root/rivermind-data` 低风险归档
- 结论：部分完成

## 做了什么

- 检查当前卷空间：`/root/rivermind-data` 可用约 243 GB；用户新给的 `/root/rivermind-fs` 为 200 GB 卷，当前可用约 165 GB。
- 尝试下载 SAGE-3D VLN 标注与 MatrixCity `aerial_street_fusion`，并核对 AirVLN 官方小型标注和模拟器下载入口；下载网络失败，没有留下数据半成品。
- 独立跑完当前 3DGS 单元测试 368 项（2 项 skip），并复跑 `tools/check.sh --python-only`。唯一失败为 `tools/test_exp_lib.py` 的 CPU quota 敏感断言，单独复测仍未通过；未改阈值。
- 在已有 Wanderland 3DGS + LiDAR 场景上用 `cf_sitl` 实跑 3 条路线：12.83 m/87 步、15.24 m/99 步、12.8 m/88 步均完成、0 碰撞，末端误差分别为 0.473/0.388/0.507 m，最低净空分别为 0.831/1.023/0.736 m，低质帧均为 0。运行使用 A* 特权教师与 legacy goal-stop，`vln=null`，不是语言导航成绩；动力学 profile 标记 `calibrated=false`。
- 将两份已解压且逐文件核对完整的重复压缩包移动到 `/root/rivermind-fs/archive/rivermind-data-archive-20261114/`，SHA256 和原路径记录在该目录的 `manifest.json`，可按清单迁回。

## 没做什么

- 没有完成任何新数据集下载：Hugging Face 请求超时；AirVLN Kaggle 下载地址返回 404，官方 S3 入口超时。`uav_datasets` 下对应目录为空。
- 没有下载 AirVLN 约 35 GB 的模拟器，先等小型指令标注和许可核验通过；没有下载约 620 MB 的 MatrixCity 单个 tar，因当前 HF 网络不可达且其许可为 CC-BY-NC-4.0。
- 没有运行新的 VLN 学生闭环或语言消融：当前没有已接通的语言 grounder / 目标条件学生，不能用教师冒充语言能力。
- 没有迁移 `root-offload-20261107`、`scratch_audit`、UAV 场景/虚拟环境、其他项目目录或 `archive/idea`：这些内容仍在使用、变化或被文档引用。后续审计确认 `claude-code-backup-20261028.zip` 是冷备份，已移动到 `/root/rivermind-fs/archive/rivermind-data-archive-20261114/`，SHA256 和恢复步骤见同目录的 `claude-code-backup-20261028.move-manifest.json`。
- 没有删除任何文件，也没有验证真机抖动、飞控标定或相机快门；本轮 SITL profile 本身未校准。

## 待确认

- 下载网络恢复后，优先获取 AirVLN 小型指令标注；使用前先确认仓库/论文许可条款。随后下载 MatrixCity 单个小样本用于覆盖和渲染探针，不要一次下载完整城市数据。
- 是否在未来引入受 CC-BY-NC-4.0 限制的 MatrixCity 实验，以及是否需要仅限非商业用途。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python -m unittest sim.3dgs.test_*` | 368 项通过，2 项 skip（代理实跑，145.7 s） |
| `bash tools/check.sh --python-only` | 其他 Python/工具/3DGS 测试通过；`tools/test_exp_lib.py::test_health_line_comes_from_the_start_and_end_snapshots` 因 CPU quota/调度测量断言失败；单测复跑仍失败，未调松阈值 |
| Wanderland `collect_3dgs` 长航线 | 完成，87/126 步，无碰撞，目标最近距离 0.356 m、末端距离 0.473 m；SITL 未校准，不代表真机安全性 |
| SAGE-3D 教师视频抽查 | 已显示指令、目标、top occupancy/path、位姿与成功状态；onboard/chase 视角可见运动模糊/重影 |
| 数据下载 | 未完成；网络超时/404，目录无下载文件 |
| 归档校验 | 两份 ZIP SHA256 与迁移清单一致；原目录文件内容逐项核验已存在 |

## 风险与遗留

- 旧采集接口仍用连续 SAFE 判定 legacy goal-stop，而 VLN 评测入口使用独立 STOP 意图。训练采集标签和 VLN STOP 监督之间还需要一条显式一致性测试。
- Wanderland 本次只跑 3 条路线、每条 1 个 seed；不能据此评估完整路线集合、渲染覆盖泛化或学生表现。
- CPU quota 指标测试会随调度波动；保留当前失败记录，后续应在可控 CPU 配额下复测。
