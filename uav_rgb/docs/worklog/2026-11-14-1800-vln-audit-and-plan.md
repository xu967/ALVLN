# VLN 全量审查与交接

- 时间：2026-11-14 18:00
- 范围：`uav_rgb` 的 VLN 任务层、3DGS、grounder、仿真控制、原型训练配置
- 结论：部分完成

## 做了什么

- 并行审查了 `sim/3dgs/run_vln.py`、`sim/3dgs/vln.py`、grounder、3DGS 场景链路和飞控仿真。
- 确认当前指令文本仍只用于任务组织/视频字幕；teacher 使用 waypoint，forward 盲飞，尚不能宣称语言理解。
- 确认 3DGS 主要问题是训练相机覆盖不足；短航线和视角外发糊不能靠渲染参数单独解决。
- 确认抖动主要来自 20 ms 离散动作直接改位姿，以及停-转-走动作切换；真机效果仍需实测校准。
- 保留并验证现有 G1 可解释词法 grounder、原型训练配置化改动和对应单测。
- 形成后续实施顺序：M2 G0 方位条件学生 → M3 G1/G2 语言落地 → M4 多段/长航线 → M5 数据扩容与真机协议。

## 没做什么

- 没有把字幕或 waypoint 偷换成语言理解结果；原因是这会污染 VLN 指标。
- 没有直接更换公开数据集；原因是 AerialVLN 等尺度与当前 E1 微型无人机不匹配，应先做覆盖度门禁和长航线小规模验证。
- 没有修改飞控硬件参数或宣称真机稳定；原因是当前只有仿真证据，缺少称重、推力曲线和传感器实测。
- 没有删除实验产物和废弃文档；原因是仓库/数据盘归属与保留清单尚未逐项确认，误删不可逆。
- 没有执行完整 CUDA/C++ 构建；本轮先完成 Python 逻辑回归，GPU/CUDA 长测留给 M2 实现后的专项验证。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `python -m unittest sim/3dgs/test_grounder.py tools/test_audit_vln_grounder.py proto/test_config.py` | 通过，13 项（1 项跳过） |
| `bash tools/check.sh --python-only` | 通过；C++/CUDA 按脚本明确跳过 |
| `nvidia-smi -L` | 发现 1 张 RTX 2080 Ti，UUID `GPU-4910bb04-2d00-ca81-f64a-2031568758ad` |

## 待确认

- 是否批准增加 500 GB 数据盘并下载 Wanderland/InteriorGS 场景。
- 真机 LicheePi 型号、ESP32 协议、成功半径和 STOP 语义。
- 删除历史实验目录/废弃 md 的明确清单。

## 风险与遗留

- 当前指标只能作为 M1 任务层上界/下界，不能作为语言模型成绩。
- 下一步必须实现 M2 bearing 条件学生，并单独报告 STOP、碰撞、SPL、nDTW/sDTW。
