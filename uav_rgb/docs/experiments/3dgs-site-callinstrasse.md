# 3DGS 场地记录：Callinstrasse（公开，校园街面）

- 日期：2026-11-07（主会话按 REVIEW-3DGS 审核结论收口；路线/pilot 由 3DGS agent 做，见 `docs/worklog/2026-11-07-0630-3dgs-routes.md`）
- 数据盘：`$DATA_ROOT/uav_rgb/3dgs-scenes/callinstrasse/`（`scene.json`、PLY 2M 高斯）；pilot 数据 `datasets/e1-3dgs-callinstrasse-pilot-v3-fov60/`（20 集 1457 帧，60°）；路线视频 `demo-e1-20261107/3dgs-callinstrasse-routes-v3-fov60/`
- 许可：MIT；来源：手持 Nikon D3200 684 张，COLMAP 米制，z-up

## 对齐

- 2026-11-07 06:05 地面标定 P0：原 `nav_to_world=I` 时 nav z=2 实际离地 3.6–3.8 m。按 366 台 COLMAP 相机脚下 1.5 m 内不透明高斯 z 的 10 分位（p50 −1.68）取地面，`nav_to_world` 平移 `t_z=−1.6`。
- FOV 按用户 11-07 决定改回 60°（备份 `scene.json.bak-20261107-0605-fov649`）。

## 飞行高度能不能核实（REVIEW-3DGS 2026-11-07 Must fix #2 的结论）

`t_z=−1.6` 由**路口**相机标定，不是全场地面。按路线逐条核（PLY 出生点 3 m 内不透明高斯 z 直方图 + 渲染器俯视 pitch −90° 中央 60×60 像素期望深度中位；同一方法在 Wanderland 6 个出生点给 1.97–2.15 m）：

| 路线 | 下方有无重建地面 | 俯视深度（出生 / 中点） | 结论 |
| --- | --- | --- | --- |
| r3_crossing_left_north | 有（nav +0.0…+0.2 众数 +0.15） | 1.59 / 2.16 m | 2 m AGL 可核，≈1.6–2.2 m |
| r4_crossing_right_south | 有 | 1.84 / 2.70 m | 同上 |
| r0_straight_east | 无（3 m 内 84 点散在 +0.2…+1.5） | 28.4 / 20.2 m | 看穿到远处浮子，高度不可核 |
| r1_hedge_left_turn | 无 | 8.6 / 37.0 m | 不可核 |
| r2_dogleg_plaza | 无（众数 +1.45 是树篱顶） | 8.6 / 2.60 m | 不可核（中点有树篱顶） |
| r5_west_field_s | 无 | 4.7 / 8.3 m | 不可核 |

366 台 COLMAP 相机里，r0/r1/r2/r5 出生点 6 m 内一台都没有，r3/r4 6 m 内 4 台。西侧走廊四条路线下方 3DGS 没有重建出地面，前视帧近地面也是无纹理灰面（pilot v3 route_0 第 10 帧）。

## 怎么用（主会话决定，2026-11-07 13:50）

- demo 视频：6 条路线都可用，字幕写「真实校园外观」，不写「2 m 低空」。
- 定量「2 m AGL」口径：只用 r3/r4。r0/r1/r2/r5 的集在 pilot 数据说明里标 `agl_unverified`，与 Gazebo 2 m 数据或真机 2 m 飞行对比时排除。
- P2 深度分带辅助目标：Callinstrasse 下半幅（近地面）深度不可信，辅助头训练优先用 Wanderland（LiDAR 真值地面）、garden / bicycle（俯视核 ≤0.1 m）；要用 Callinstrasse 就把近地面行掩成 255 未知。
- 「有真值地面」的主场地让位给 Wanderland；Callinstrasse 保留为外观多样性场景。`scene.json` 的 notes[1] 等 VIZ-LAYOUT 渲染结束后同步改写（渲染中不动数据盘文件）。
