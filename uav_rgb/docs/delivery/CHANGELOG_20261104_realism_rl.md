# 2026-11-04 代码变更清单

## 兼容性约束

- 原有 `base/l1/campus` world 生成路径和默认 `teacher=astar` 保留；新的路线只在明确使用 `campus_assets`、`--teacher-planner predictive`、`--policy teacher-rl` 时启用。
- 数据集修复属于行为变更：`teacher/dataset.py` 现在保留 `step` 原始编号，并拒绝跨断帧序列；历史训练权重不作自动迁移。
- 旧实验数据和模型不在本 ZIP 内，本包没有训练好的 PPO checkpoint。

## 新增

- `sim/tools/asset_campus.py`：真实网格管线入口、几何-碰撞-规划统一记录、四色目标组合及几何 map 分割。
- `sim/assets`：本地 `model://` 模型包，**原创离线示范资产**，不冒充真实第三方模型。
- `sim/tools/import_asset.py` + `register_ground_texture.py`：手工验证 CC0 模型/纹理后离线导入。
- `sim/tools/teacher_predictive.py`：T1 考虑速度和控制延迟的模型预测教师（实验）。
- `rl_teacher/`：T2 简化动力学 PPO 训练接口、PBRS 路径势函数奖励、预测风险、独立测试脚本。
- `teacher/test_dataset_step_gap.py`、`sim/tools/test_asset_campus.py`、`rl_teacher/test_core.py`。
- `docs/realism_rl_v1/README.md`、`VALIDATION.md`。

## 修改

- `sim/tools/scene_gen.py`：新增场景任务分支，不改默认。
- `sim/tools/split_manifest.py`：新增地图族隔离与分组。
- `sim/tools/collect.py`：新教师选择、PPO 权重、版本字段；保持旧参数默认。
- `sim/tools/collect_parallel.py`：新场景/新 teacher 的多实例入口，配置本地资产资源路径。
- `sim/tools/eval_closed_loop.py`：新场景与教师评测接口，防止错误去重。
- `teacher/dataset.py`：`step` 过滤索引错误修复和滑窗跨缺帧保护。
- `uav_rgb/README.md`：新文档索引。

## 不应当误读的结论

“代码已实现”不等于“性能已验证”。PPO 尚未在本环境训练；实际 Gazebo renderer 和 CF 动力学闭环并未执行；真实 PBR 资产必须使用许可合法的第三方模型替换离线原创示例网格。参数需要真实飞行平台标定，不能用本包宣称真机自主导航已达到某个成功率。
