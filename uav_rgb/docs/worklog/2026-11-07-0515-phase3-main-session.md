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

## 更新 2026-11-07 08:00（第五阶段：独立审核 + 并行验证）

用户 07:38 授权：加派 agent 并行审核；数据盘已扩（余 78 GB）可直接下载；可并行跑验证。

- 派出 4 个独立审核 agent（与 TEACHER-P1、SCENE-B 共 6）：REVIEW-P2（student-p2 分支 ed25f52，未合并）、REVIEW-SITL（2b479cc）、REVIEW-EVAL（8375325 + pick_test_seeds）、REVIEW-3DGS（c4be9fb + e4184c3）。简报 `tmp/phase3-20261107/BRIEF_REVIEW.md`：ponytail-review 格式，每条发现要有具体案例；Must fix ≤30 行在自己 worktree（`tmp/wt-review-*`，分支 `review-*`）先写会红的测试再修，不 commit；允许每人 1 个 gz 实例与小规模 3DGS 渲染做真实验证；构建目录 `tmp/build-review-*`。
- main `check.sh --python-only` 复跑：PASS（52 GUI 测试、inference/commands/dry-run/test_train_student_with_video 全过，91 s）。
- 独立核对 TEACHER-P1 软标签统计（`tmp/main-session/P1_soft_label_diagnosis_0805.md`）：硬 FORWARD 帧上 Q_SAFE−Q_min≈0.516 s，α=1 步 → p(SAFE)≈0.26；硬 YAW 帧上软 Q 的 argmin 78–85% 是 FORWARD。与 agent 报告 §3.2/§3.3/§4.3 一致：决定性消融「只把 8.7% 标签改成 kino argmax 的硬标签」SR 13/18 → 5/18，失败在 `teacher_kino` 代价模型对走廊小角度航向修正的判断，与软硬无关。方向：示范与标注统一到同一代价模型（给 `astar_occ` 实现 `plan_costs`，软标签 = 同一 A* 的逐动作 cost-to-go），再做软 vs 硬对照。
- cgroup CPU 配额实测 12 核（`cpu.max 1200000/100000`，20% 周期被限流，负载 32）：森林跨场景评测（已用 `pick_test_seeds.py --task forest_e1 --density sparse` 挑出 20 个独立几何 `7,8,19,22,23,39,53,64,71,86,90,102,110,134,135,136,142,145,155,161`）推迟到 eval2 结束释放 gz 后再起，避免拖慢正在跑的四组重评与审核构建。
- eval2 进度（07:55）：forward 20/20（SR 2/20）、x3-old 20/20（SR 2/20）、e1-new 8/20（4 成功）、teacher 6/20（6 成功）；带录像两组约 09:10 结束。

### 没做什么（本段）

- 没起森林跨场景评测：CPU 配额 12 核已满载，等 eval2 释放。
- 没下载新场景：第二个 Wanderland 场景需要人工看轨迹图选 ID 并设计路线，留给 SCENE-A agent 续做（slot 满）。
- 没合并 student-p2 / teacher-p1 / scene-mipnerf：等审核报告与 eval2 结束（build/check 正在被评测使用）。
