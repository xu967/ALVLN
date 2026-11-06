# 第三/四阶段主会话：全库并行审计、修复合并、规划 v2、demo 视频、重评（2026-11-07 05:10 起）

- 时间：2026-11-07 05:10 → 进行中（本文随进度追加「更新」小节）
- 范围：主会话编排；代码改动来自 6 个审计 agent 与 5 个开发 agent 的分支，主会话负责审核、合并、构建、真 Gazebo 验证、推送
- 结论：部分完成（审计阶段完成并全部合并；开发阶段 5 个 agent 进行中；第二轮四组重评进行中）

## 做了什么

- 接手时流水线 `e1-pipeline-20261107-0417` 在采集；上一会话两位 agent 的未提交改动（forest 接管线、3DGS 多路线）跑单测全绿后按归属提交（`1825c8b`、`4a3f4f6`），并按 ponytail-review 逐行审核，两条必修项（demo 脚本重复定义、规划失败不落盘）在 3DGS 分支合并时修掉。
- 核实 CF-Drone 本地 36 文件改动只是 CRLF，固件与 e7f507f 一致。
- 读三处硬件网址（琛光 specs/faq/extended-module、oshwhub、GitHub CF-Drone）：规格与 `e1_mini.json` 一致；官方闭源 bin 才有定高定点，开源固件需自行实现。
- 建 5 个 worktree 派第三阶段审计 agent（SITL / VIZ / DATA / 3DGS / SRC）+ ALGO 调研；全部完成并合并：
  - SRC（`752fd9f`）：u8→float 舍入与设备 kernel 一致、label-smoothing 下 train_ce 口径、worktree 内 source_git、B4 输入一致性对拍脚本 `tools/check_input_parity.py`。
  - DATA（`9edf915`）：教师按 (载具, 任务) 取余量、圆柱按圆判定、forest 教师 v6、preprocess 跨族打包 `--allow-mixed-geometry`、`compare_datasets.py` 骨架。
  - 3DGS（`6842cab`）：Callinstrasse 地面高度 P0（之前 nav z=2 实际离地 3.7 m）、逐帧 `frame_quality`、`planner_failures`、demo 断点续跑。
  - SCENE-A（`88357bc`）：Wanderland 场景（Apache-2.0，LiDAR 真值）接入；碰撞代理 v3 points/mesh 真值源；代理 v2 在该场景覆盖率 98.7%。
  - VIZ（`b918ee1`）：E1 顶视相机覆盖到 x 7.4 m、E1 四旋翼外观、机体出画标注、3DGS 跟随格线框/比例尺/HFOV 标签、Studio forest_e1/E1、Monitor HiDPI。
  - SITL（`50cd5ee`）：倒飞与悬停漂移根因（陀螺零偏 → 固件姿态漂 → 速度环积分与机动倾角共用限幅吃满 → 开环倒飞）；积分单独限幅、SAFE 位置保持、ToF 斜距；复现 CTest；smoke 漂移 0.10–0.49 → 0.02–0.19 m。
- 规划：`docs/PLAN_2026-11-07_demo_and_paper.md` v1 → v2（定位纠正、D1/D2 交付、W1–W5、里程碑、P1/P2/P3 算法定稿、场景清单、评测协议、待拍板）；`docs/DATASET_3DGS_SELF_CAPTURE.md`（自采 3DGS 规范）。
- demo：`tools/make_demo_reel.py` + v0 短片（296 s）；Callinstrasse FOV 按用户决定改回 60°，v3 路线 + pilot（20/20 到达 0 碰撞）。
- 评测口径：发现 20 个冻结 test seed 只有 11 个独立几何；`tools/pick_test_seeds.py` 挑 20 个独立几何写进 `e1_pipeline.py`。
- 构建与验证：SITL 库 v2 重建到 `tmp/build-cf-sitl`（旧库保留 `build-cf-sitl-v1-20261106`），ctest 6/6、自检 33/33、smoke；`tools/check.sh` CTest 27/27 + Python 全过（Studio GUI 几何测试改成按显示器字体比例缩放窗口）；真 Gazebo 1 集教师录像验证新库 + 新模板 + 新录像代码链路。
- 第四阶段派 5 个开发 agent（TEACHER-P1、STUDENT-P2、SCENE-A、SCENE-B、EVAL）；第二轮四组重评守护进程（20 独立几何，lib v2）。
- 每个里程碑 `bash tools/git_mirror_alvln.sh` 推 GitHub。

## 没做什么

- 没重跑 forest v6 教师的真 Gazebo 验证（Gazebo 被重评占用；待重评结束后 2 集 sparse 验证穿林）。
- 没做 SITL 剩余 ≈0.35 cm/s 真漂移（零偏×高度经光流旋转补偿进入速度估计）：需要飞行中估零偏，属固件/执行器下一步，已写进待办。
- 没改真机参数（质量、桨径、推力曲线、零偏残差）：`calibrated=false`，等硬件团队实测。
- 没做 D-GEODUP 的系统性检查（只修了评测 seed 列表）；训练集内近重复几何未统计。
- 没申请 DL3DV / ScanNet++ 账号（需用户）。
- 没在 lib v2 上重采训练数据：第二轮重评的学生权重来自 lib v1 数据，表里标注。
- 3DGS 三视角原片仍是 960×276，短片 v1 前要调大 `sim/3dgs/video.py` 面板尺寸。

## 待确认

- val 太小（18 集、SAFE 28 样本）的口径处理；SAFE 语义（执行器位置保持已加，教师近柱 SAFE 是否改）；DL3DV/ScanNet++ 申请；自采场地与设备。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `tools/check.sh`（main，07:05） | CTest 27/27、cf_sitl 6/6、Python 全过；Studio GUI 一项按 DPI 缩放修后 52/52 |
| SITL lib v2：ctest / selftest / smoke | 6/6；33/33；seed 1–3 READY，漂移 0.105/0.193/0.041 m |
| 真 Gazebo 教师 1 集（seed 58，lib v2，新模板，录像） | 80 步到达 0 碰撞；视频 817 帧，机体外观与顶视覆盖正常 |
| 第一轮四组对照（n=11） | 教师 10/11、新 E1 学生 6/11、旧 X3 学生 2/11、盲前进 0/11（见 demo-plan §3.1） |
| 第二轮四组重评（n=20，lib v2） | 进行中（PGID 694826） |
| 3DGS v3（60°）路线 + pilot | rc=0；pilot 20/20 到达 0 碰撞，低质帧 1.3%，2 集倒飞隔离 |

## 风险与遗留

- lib v1 → v2 改变了 SAFE 行为（位置保持），新采数据与旧数据的动力学不同，混包时要分 `dynamics_profile`。
- `build/check` 二进制已按合并后 main 重建；`tmp/build-cf-sitl` 为 v2，所有后续采集/评测默认用它。
- 第四阶段 5 个 agent 的分支待审合并；worktree 清理在收尾做。
