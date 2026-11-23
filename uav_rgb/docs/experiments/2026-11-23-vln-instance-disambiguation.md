# 同画面同类目标实例诊断

- 时间：2026-11-23 20:47 CST
- 环境：RTX 2080 Ti，物理 GPU0
- checkpoint：固定 UAV-PUR-4B，冻结原 G3 worker 与 40-token 贪心解码
- 入口：按登记顺序读取 2 张已有 RGB，对每张询问上层植物、下层植物和缺席自行车；复用仓库 G3 grounder 与 JSONL worker，没有创建脚本
- 数据：/root/rivermind-data/uav_rgb/exp-vln-instance-disambiguation-20261123/

| 目标 | G3 输出 | 可辨认实例 | 方向 |
| --- | --- | ---: | ---: |
| 上层白圆盆植物 | 两帧均只返回 “leafy plant” | 0/2 | 2/2 |
| 下层浅色盆植物 | 两帧均返回 “smaller plant in pale bowl” | 2/2 | 0/2 |
| 画面外蓝色自行车 | 两帧均返回 S / none | 拒答 2/2 | 不评分 |

六次查询在 worker 启动后的墙钟为 6.16 秒，单次生成时间中位数为 793.8 ms。目标描述和框由人工依据原 RGB 冻结；SAGE 场景与 CAD 中没有相应植物实例标签。两张图是同一轨迹的相邻帧，不是独立测试样本。

登记文件原文把下层植物说成位于左侧。推理后复核归一化分区发现其中心为 116.5/129.5 像素，应计中央区，因此下层方向为 0/2。没有改写预注册原件，保留了带解释的[坐标更正](</root/rivermind-data/uav_rgb/exp-vln-instance-disambiguation-20261123/annotation_correction.json>)。

完整登记、模型原始回复及逐项评分分别见[登记](</root/rivermind-data/uav_rgb/exp-vln-instance-disambiguation-20261123/registration.json>)、[原始输出](</root/rivermind-data/uav_rgb/exp-vln-instance-disambiguation-20261123/raw.json>)和[评分](</root/rivermind-data/uav_rgb/exp-vln-instance-disambiguation-20261123/scores.json>)。结果可支持继续审计实体辨认与位置输出，不能证明停止距离或导航闭环有效。
