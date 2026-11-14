# OpenFly 本地评测接入说明

本项目只把 OpenFly 当作外部语言导航基准和数据格式参考，不把 OpenFly 的城市尺度分数直接并入本项目的 3DGS/UAV 小步长排行榜。

## 当前结论

- 官方平台仓库提供 `train/eval.py` 和 `configs/eval_test.json` 的本地评测流程；没有发现官方远程提交接口或可直接上传模型的 evaluation server。
- 官方动作有 6 类语义动作，但 evaluator 实际使用 10 个离散 token/距离变体；Forward 的步长是 3/6/9，和本项目的 4 primitive、约 0.25 m/25° 动力学不兼容。
- 官方数据集约 2.97 TB，3DGS 场景还需要单独的环境资产。当前阶段不下载全量数据，也不改本项目 evaluator。

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
3. 若要得到 OpenFly 论文口径的 NE/SR/OSR/SPL，需在独立的 Ubuntu 22.04、ROS2/AirSim/UE 或官方 3DGS bridge 环境中运行官方 `train/eval.py`。这组结果单独报告，不能和本项目的 0.25 m 小型机动力学混排。
4. 若只做迁移基线，先记录动作翻译规则和 `cross_domain=true`，再把 OpenFly grounder 的指令理解输出送入本项目的局部控制器；迁移结果只能标成附加对照。

## 结果解释

审计报告是数据质量报告，字段 `records_with_action_and_pos` 只表示最小闭环字段存在，不表示视觉可用或任务成功。正式实验仍需固定 scene split、初始位姿、相机、时间预算、碰撞规则和随机种子，并同时报告 NE、SR、OSR、SPL、collision rate、latency 和显存。
