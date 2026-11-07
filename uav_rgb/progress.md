# progress.md（新开会话先读这个）

- 更新：2026-11-07 05:30（Cursor 新会话接手，第三阶段：全库并行精读 + 修 bug + demo）

## 第三阶段（2026-11-07 05:10 起）

用户原话：仿真里的无人机不像真机、可视化可能有 bug、要真实仿真环境、读代码/注释/git、用马尾辫 skill、跑测试和 demo、读三处硬件网址、调研浙大高飞/中科大、能改的直接改、并行 agent 不超上限。

- 05:15 接手时：E1 流水线 PID 424699 在 collect 步（156/300 集在 `.staging`）；forest agent 两个守护进程 DONE；3DGS 多路线守护进程 DONE rc=1（6 路线 × 2 集完成、11/12 到达；pilot 阶段 route_2 因 agent 边跑边改 `collect_3dgs.py` 报 `NameError: ANOMALY_BACK_M`，现文件已完整）。
- 05:18 forest / 3dgs-routes 两位 agent 的未提交改动跑单测全绿后按归属提交：`1825c8b feat(sim/tools) forest_e1 接管线`、`4a3f4f6 feat(3dgs) 多路线 + A* 教师 + demo_3dgs_routes`。两份 worklog（untracked）占位符待填。
- CF-Drone 本地 36 文件改动只是 CRLF（`git diff --ignore-all-space` 为空），固件与 e7f507f 一致。
- 05:25 建 5 个 worktree（`$DATA_ROOT/uav_rgb/tmp/wt-{sitl,viz,data,3dgs,src}-20261107`，分支同名），公共简报 `tmp/phase3-20261107/BRIEF.md`，报告目录 `tmp/phase3-20261107/reports/`。理由：流水线逐集起新进程读主工作树的 `sim/tools/*.py`、`e1_mini.json`，并 mmap `tmp/build-cf-sitl/*.so`、用 `build/check/` 二进制；agent 直接改主目录会让数据集前后集不一致。
- 05:28 派出 5 个并行 agent（上限 6）：SITL（固件在环保真度、倒飞/悬停漂移根因、`build-cf-sitl-s3`）、VIZ（看真实视频帧找可视化 bug）、DATA（教师/采集/评测/审计/preprocess，森林膨胀按 (vehicle,task)、preprocess 跨族打包、compare_datasets 骨架）、3DGS（pilot 重跑守护进程 + 真实场景评估 + 相机一致性）、SRC（C++/CUDA 读审 + B4 输入一致性对拍，`build-src-s3`）。都不 commit，不起 Gazebo，`--parallel 2`。
- 05:50 用户拍板：相机参数按老实验（60°/320×240，Gazebo 与 3DGS 统一；Callinstrasse scene.json 的 64.9° 待 3DGS pilot 结束后改回并重跑 pilot）；排期同意；论文机器人/视觉并齐；用户团队将实地自采 3DGS（规范 `docs/DATASET_3DGS_SELF_CAPTURE.md`）；视频优先。
- 05:55 v0 demo 短片：`$DATA_ROOT/uav_rgb/demo-e1-20261107/reel/demo_reel_v0.mp4`（296 s，`tools/make_demo_reel.py`，3DGS 6 路线 + 森林）。SRC 分支已审并合并（752fd9f）；GitHub 到 45336a4。
- 06:00 ALGO / 3DGS / DATA 三份报告收齐（`tmp/phase3-20261107/reports/`）。SRC(752fd9f)、DATA(9edf915)、3DGS(6842cab) 已合并进 main 并推送；VIZ 已在分支 fabecb3 落提交，等流水线 DONE 再合并（它改 E1 模板，会影响正在跑的评测）。
- 3DGS P0：Callinstrasse 之前把相机平面当地面，nav z=2 实际离地 3.7 m，已修（nav_to_world t_z −1.6）。按用户决定 FOV 改回 60°（备份 scene.json.bak-20261107-0605-fov649），v3 路线+pilot 守护进程 PID/PGID 607752（`demo-e1-20261107/3dgs-callinstrasse-routes-v3-fov60/run.log`，pilot 输出 `datasets/e1-3dgs-callinstrasse-pilot-v3-fov60`，停：`kill -- -607752`）。
- 06:10 规划 v2 算法定稿：P1 信念边际化 kino cost-to-go 软标签 + DAgger（主）、P2 双帧 + 深度分带辅助头、P3 探索；RL 教师改为 P1 后的 RL 微调；场景清单与评测协议进 `docs/PLAN_2026-11-07_demo_and_paper.md`。
- 06:20 第四阶段派出 5 个开发 agent（+SITL 共 6）：TEACHER-P1（`plan_costs` + `teacher_belief.py` + 3DGS `--policy student` + `dagger_pipeline.py`，Callinstrasse 小规模硬/软标签对照）、STUDENT-P2（UAVR 升版双帧/prev-action/深度分带、`in_c=6`、辅助头、逐格 CE 算子、checkpoint v6、infer 协议）、SCENE-A（Wanderland 调研下载接入 + 碰撞代理 v3 网格真值对照）、SCENE-B（Mip-NeRF 360 garden/bicycle 接入 + `tools/scene_probe_quality.py`）、EVAL（SPL/CR/task_success 列、`tools/stats_report.py`、compare_datasets 补全、论文表模板、四组对照表）。worktree `tmp/wt-{teacher-p1,student-p2,scene-wanderland,scene-mipnerf,eval-matrix}`，简报 `tmp/phase3-20261107/BRIEF4.md`。
- 06:40 流水线 DONE rc=0：四组对照 n=11（20 seed 只有 11 个独立几何）：教师 10/11、新 E1 学生 6/11、旧 X3 学生 2/11、盲前进 0/11（`docs/experiments/2026-11-07-e1-demo-plan.md` §3.1）。
- 07:00 SITL(50cd5ee)、VIZ(b918ee1)、SCENE-A(88357bc) 合并进 main；SITL 库 v2 重建到 `tmp/build-cf-sitl`（旧库 `build-cf-sitl-v1-20261106`），ctest/自检/smoke 全过；`check.sh` 全过（Studio GUI 测试按 DPI 缩放窗口 e325add）；真 Gazebo 1 集教师录像验证新链路（`tmp/gz-validate-20261107/teacher-seed58/videos/seed_58.mp4`）。
- 07:11 第二轮四组重评守护进程：`e1-pipeline-20261107-0417-eval2`，PGID 694826，20 独立几何 × 4 组，lib v2，`--from eval`（run/ ds/ 软链到 0417）；停：`kill -- -694826`。
- 07:20–07:30 STUDENT-P2 落提交 `ed25f52`（分支 student-p2，未合并，等 eval2 结束再合并并重建 build/check）；EVAL 合并 e33e816。
- 07:38 用户新授权：可加派 agent 并行审核；数据盘已扩（余 78 GB），需要的内容直接下载；可并行跑验证。
- 07:50 第五阶段派出 4 个独立审核 agent（+TEACHER-P1、SCENE-B 共 6，满上限）：REVIEW-P2（student-p2 分支 42 文件，worktree `tmp/wt-review-p2` 基于 ed25f52）、REVIEW-SITL（2b479cc 执行器修复，`tmp/wt-review-sitl`）、REVIEW-EVAL（8375325 SPL/stats_report/compare_datasets + pick_test_seeds，`tmp/wt-review-eval`）、REVIEW-3DGS（c4be9fb 碰撞代理 v3/Wanderland + e4184c3 规划/采集/demo，`tmp/wt-review-3dgs`）。简报 `tmp/phase3-20261107/BRIEF_REVIEW.md`；报告 `tmp/phase3-20261107/reports/REVIEW_{P2,SITL,EVAL,3DGS}_20261107.md`；各自构建目录 `tmp/build-review-<name>`。规则：ponytail-review 格式；Must fix ≤30 行直接在自己 worktree 修（先写会红的测试），不 commit；每人最多 1 个 gz 实例。
- TEACHER-P1 中间结果（07:28，Callinstrasse pilot 20 集、6 路线 × 3 集评测）：hard SR 12/18、soft(K=8, α=1 步) SR 4/18、hard_plain 13/18 —— 软标签第 0 轮明显更差，agent 在试 α=0.25 与软硬混合，并跑 DAgger 第 1 轮（PGID 713232）。合并 P1 前要看清原因，不能只看 argmax 一致率 91%。
- 07:55 独立核对 P1 统计（`tmp/main-session/P1_soft_label_diagnosis_0805.md`）与 TEACHER-P1 报告 §3–§4 一致：决定性消融「只把 8.7% 标签换成 kino argmax 硬标签」SR 13/18→5/18，问题在 `teacher_kino` 代价模型（走廊小角度修正判为不值得 + 剪枝键无速度让 SAFE 代价被低估），不是软标签本身。P1 下一步方向：示范与标注统一到同一代价模型（给 `astar_occ` 实现逐动作 cost-to-go），再做软 vs 硬。
- 机器约束：cgroup CPU 配额 12 核（不是 88），负载 32、20% 周期限流。排队的验证：森林跨场景评测（teacher / e1-new / forward × forest_e1 sparse 20 独立几何 `7,8,19,22,23,39,53,64,71,86,90,102,110,134,135,136,142,145,155,161`），等 eval2 结束释放 gz 再起。
- **08:03 会话中断**（IDE 会话重启，到 12:55 才恢复）：6 个子 agent 与它们起的 gz/渲染/构建进程全部被杀；只有 setsid 守护进程活下来。eval2 08:46 DONE rc=0：教师 19/20、新 E1 学生 7/20、旧 X3 学生 2/20、盲前进 2/20（碰撞 0.05/0.40/0.45/0.90；已填 `docs/experiments/2026-11-07-e1-demo-plan.md` §3.2；SPL 列空，因 eval2 启动早于 e33e816）。用户 YOLO 训练进程（82761）在此期间自行结束（不是我们动的；`YOLOVOmodal_3871538535/logs/full_e4_resume.log` 最后写于 12:05），GPU 空闲。
- 12:58 按用户要求恢复中断的 agent：TEACHER-P1、SCENE-B 的报告/worklog/PROGRESS 在中断前已定稿（待办全勾），不再恢复，直接收报告；4 个审核员用 `resume` 续做（各自 PROGRESS.md 断点：P2 在 F/G、SITL 等 Gazebo 1 集 + 报告、EVAL 在 D、3DGS 等 Wanderland 渲染 + 报告）。
- 13:00 新派 2 个 agent（共 6）：DISTILL（`tmp/phase3-20261107/BRIEF_DISTILL.md`：蒸馏环节 7 个问题，交判断不交改动，报告 `reports/DISTILL_20261107.md`）、VIZ-LAYOUT（`BRIEF_VIZ_LAYOUT.md`：3DGS 三视角从 960×276 统一到 Gazebo 的 1280×720 版式、一份版式实现、重合成已有 3DGS 集、demo 短片 v1；worktree `tmp/wt-viz-layout`）。用户 07:55 的「布局优化」问题（可视化版式 vs 规划排布）没等到回答：按可视化版式理解派了 VIZ-LAYOUT，规划文档更新由主会话自己做。
- 13:11 森林跨场景评测守护进程：`$DATA_ROOT/uav_rgb/xeval-forest-sparse-20261107-1310/run_xeval.sh`，PGID 941204，teacher → student-e1-new → forward 顺序各 20 独立几何（forest_e1 sparse，test），一次 1 个 gz，infer 二进制快照在 `…/bin/`；看 `run.log` / `eval-<组>.log`，结束落 `DONE` + `summary.md`；停 `kill -- -941204`。教训：gz 必须 `UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg`（同 `e1_pipeline.GZ_ENV`），默认 gpu(EGL) 后端在本容器段错误（nvidia EGL 库是 0 字节桩）；bash 里不能用 `GROUPS` 作变量名（只读数组）。
- 13:19 用户新要求：第五阶段改完并推 GitHub 后，加派多个 agent 按 **GitHub diff** 审核**全部代码**。已草拟第六阶段简报 `tmp/phase3-20261107/BRIEF_REVIEW2.md`：基线 ALVLN `8e7d488`（= uav_rgb `552daac`），6 个审核员按模块分域（R2-SRC / R2-SIMTOOLS / R2-3DGS / R2-SITL / R2-UIDOCS / R2-REPRO 从 GitHub 克隆按文档从零跑），启动时填 HEAD。启动条件：当前 6 个 agent 交付 → 合并 review-* 修复、student-p2（视 REVIEW-P2 + DISTILL 结论）、viz-layout → `check.sh` → 重建 `build/check` → 推 GitHub → 建 worktree `tmp/wt-r2-*` 派出。
- 13:20 scene-mipnerf 已提交 ea62e1f 并合并进 main（c435b49）；teacher-p1 已在分支落提交 a4b932e（**未合并**，等 W2 改向决定）；PLAN 追加 13:40 更新（99161b1）；GitHub 镜像推到 b2d7230。
- 主会话 worklog：`docs/worklog/2026-11-07-0515-phase3-main-session.md`（08:00 追加第五阶段小节）。
- 待办：收 4 份审核报告 → 按 Must fix 修/合并 review-* 分支的改动 → 收 TEACHER-P1 / SCENE-B 报告并合并 → 流水线结束后逐个合并分支到 main（先跑 check.sh）→ 真 Gazebo 验证 agent 列的清单 → 填四组对照数字 → demo 汇总页 → worklog → 镜像推送 GitHub（`bash tools/git_mirror_alvln.sh`）→ 清理 worktree。

---

- 上一段更新：2026-11-07 04:05
- 代码仓库：`/root/claude-code-backup-20261028/code/uav_rgb`（main）；GitHub 镜像 `git@github.com:xu967/ALVLN.git` 的 `uav_rgb/` 子目录，用 `bash tools/git_mirror_alvln.sh` 推
- 数据盘：`$DATA_ROOT/uav_rgb`（`/root/rivermind-data/uav_rgb`）；构建产物、数据集、实验输出都在这里，不进仓库
- 任务边界：`$DATA_ROOT/uav_rgb/grill-me/boundary-v4.txt` + `boundary-v5-addendum.txt`；任务板 `docs/TASKS.md`（SIM-E1）；实验计划 `docs/experiments/2026-11-07-e1-demo-plan.md`；长期规划 `docs/ROADMAP.md`、`docs/TEACHER_LINE_PLAN.md`

## 正在跑的守护进程（先看这里）

| 实验 | 启动 | 进程 | 看进度 | 停 |
| --- | --- | --- | --- | --- |
| 森林跨场景评测 `xeval-forest-sparse-20261107-1310/run_xeval.sh`（teacher → e1-new 学生 → forward，forest_e1 sparse 20 独立几何，一次 1 个 gz） | 2026-11-07 13:11 | PGID 941204，`$DATA_ROOT/uav_rgb/xeval-forest-sparse-20261107-1310/run.pid` | `tail -f …/run.log`、`…/eval-<组>.log`；结束落 `…/DONE` + `summary.md` | `kill -- -941204` |
| （已完成 08:46）第二轮四组重评 `e1-pipeline-20261107-0417-eval2`（20 独立几何，lib v2） | 2026-11-07 07:11 | PGID 694826 | `…-eval2/run.log` 末尾 summary；视频 `eval-teacher/videos/`、`eval-student-e1-new/videos/` | — |
| （已完成 06:40）E1 流水线 `tools/e1_pipeline.py`（300 集采集 → 审计 → 打包 → 训练 30 轮 → 20 seed 四组闭环含三视角视频 → 汇总；pilot 已过） | 2026-11-07 04:25（`--from collect`） | PID/PGID 424699，`$DATA_ROOT/uav_rgb/e1-pipeline-20261107-0417/run.pid` | `tail -f …/run.log`；每步状态 `…/status.json`；结束落 `…/DONE`（rc + 最后一步）；评测各组日志 `…/eval-<组>.log`，视频 `…/eval-teacher/videos/`、`…/eval-student-e1-new/videos/` | `kill -- -424699`（collect_parallel 已装 SIGTERM 处理器，会把 gz 一起收掉） |
| 子 agent「Gazebo forest_e1 场景接入」（随机圆柱森林三档密度接进 scene_gen/collect/eval/audit/e1_pipeline，真 Gazebo 测教师，录 1 段三视角） | 04:19 | 工作目录 `$DATA_ROOT/uav_rgb/tmp/agent-forest/`，断点 `PROGRESS.md` | 同左 | 它自己的 run.pid |
| 子 agent「3DGS 多路线 demo 与数据」（Callinstrasse ≥4 条带转弯路线、三视角视频、20 集 3DGS E1 pilot 数据集） | 04:19 | 工作目录 `$DATA_ROOT/uav_rgb/tmp/agent-3dgs-routes/`，断点 `PROGRESS.md`；产物 `$DATA_ROOT/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-routes/` | 同左 | 它自己的 run.pid |

第一轮 pilot（03:57，旧教师余量）：20/20 集采到，audit FAIL 2 项（8/20 集教师在柱旁 SAFE 死锁后漂进柱子；label_recompute 47/800 是 audit 用 X3 半宽复算的假阳性）。修 teacher/audit（5af664a）后第二轮 pilot（04:17）：17/20 集干净穿过 x≈7 m，label_recompute 0/800；剩 `class_share`（SAFE 3.9%）与 `safe_collision`（3 集柱旁左右摆后擦柱）两项在 E1 口径下放行（擦柱步由 preprocess --drop-collision 丢弃，教师真实 SR 由评测 teacher 组报）。

已有视频：3DGS Callinstrasse 教师闭环三视角（机载/跟随/顶视）3 集，`$DATA_ROOT/uav_rgb/demo-e1-20261107/3dgs-callinstrasse-teacher-48steps/collect/ep_0000{0,1,2}/views.webm`（48 步全 FORWARD，3/3 到达 0 碰撞；路线是直线，正由 3DGS agent 加转弯路线）。Gazebo 三视角视频由流水线评测步的 `--record-video` 产出（teacher 与新学生组），预计 06:30 前后。

另有用户自己的 YOLO 训练（`/root/rivermind-data/YOLOVOmodal_3871538535/train.py`，PID 82761）在用 GPU 约 16 GB、2 核，不要动。

## 已完成

- SITL：`sim/cf_sitl/` 固件在环库（CF-Drone e7f507f 原样编入，1 kHz，ToF/光流/电池/执行器），自检 27 项、打杆阶跃；`tof_airborne_m=0.15` 修起飞前误判 FAILSAFE（13b26df、0c26967、e103a9b）
- Gazebo：`sim/cf_sitl/gz/` CfSitlSystem 插件 + E1 模板 + `collect/collect_parallel/eval_closed_loop/audit --dynamics cf_sitl --vehicle e1_mini`；真 Gazebo 冒烟 0 FAILSAFE；1/4/6 实例采集链路通（cc8bbe6）。真库插件构建在 `$DATA_ROOT/uav_rgb/tmp/build-cf-sitl/gz/`，库在 `…/build-cf-sitl/libcf_sitl.so`
- 3DGS：Callinstrasse 发雾修复（SH3 + 2× 超采样 + 去 >3 m 巨型高斯）、碰撞代理 v2、`--dynamics cf_sitl` 经 `cf_sitl_proc`、三视角视频、`tools/demo_3dgs_callinstrasse.py`（f2e9e0f、c99b97c）
- 可视化：Studio/录像 REVIEW #5–#13/#17/#18 等（a221278）
- 学生 C++/CUDA（已合并 3f27c61）：训练 60 → 2669 样本/秒，推理 p50 6.57 → 0.43 ms，同 seed 逐位可复现；A/B 开关全部默认关（没有一项过门槛）
- 教师线：`sim/tools/teacher_kino.py`（E1 运动原语 kinodynamic A*）+ 新旧教师离线对比；`scene_forest.py` forest_e1 场景族；FAST Lab 精读报告两份
- 采集器健壮性：`collect_parallel` SIGTERM/Ctrl-C 后不留孤儿 gz、不重试、不起下一集（c171106，AUDIT Q1/Q2）
- 一键流水线 `tools/e1_pipeline.py`（552daac）；`tools/check.sh` 在合并后 main 上全过（CTest 27/27 + Python 全绿，2026-11-07 03:52）
- GitHub：ALVLN `main` 已推到 `8e7d488`（对应 uav_rgb `552daac`）

## 关键决策

- E1 与 X3 数字分档，不混报；E1 评测 20 seed 固定为 `180,206,250,485,503,555,586,708,715,721,799,855,932,992,1113,1116,1123,1145,1246,1282`（v3/v5 双 test）；采集 seed 从 3000/3100 起避开它们
- 评测语义 observation、hold 1.0、E1 机体四组对照（教师 / 旧 X3 学生 / 新 E1 学生 / 盲前进）；SR 分母 = 请求集 20
- `calibrated=false`：所有 E1 参数是硬件草案/假设值，demo 必须如实标注；悬停油门杆 0.894 余量只剩 10%，真机称重与推力曲线是下一阶段第一件事
- 采集默认锁 RTF 1.0 + `--no-aux-cameras` + 4 实例；仓库里没有解锁 RTF 的开关（性能余量只做了测量）
- 学生算法 A/B：平移增强对导航标签有破坏性（−0.054 macro_f1），其余在种子噪声内；默认全关
- 镜像脚本按 `--first-parent` 镜像：合并进来的分支在 GitHub 上是一个提交
- 本仓库纯个人学习用途，文档不写商业授权声明

## 用户 04:31 的新要求（两数据集对比）

- 数据集 A = Gazebo「柱阵 → 森林三档」，数据集 B = 3DGS「Callinstrasse 转弯路线」；学生 S_A / S_B / S_AB 在两个域的测试集上交叉评测（教师、盲前进、旧 X3 学生作参照）。计划与矩阵：`docs/experiments/2026-11-07-dataset-compare-plan.md`。对比入口 `tools/compare_datasets.py` 等两个 agent 的 CLI 定下来再写。
- 3DGS 画质结论（04:45）：Callinstrasse 是街面手持采集、2M 高斯铺一整条街，中景可用（Laplacian 方差 472，比 Gazebo 柱阵 140 还高），但靠近墙/绿篱的近景发糊重影、路线最长只有 8–15 m，是数据集本身的限制，不是渲染 bug。更好的选择：H3DGS Campus/SmallCity 预训练（Inria，2.3–9.8 GB，街面 1.5 km 级）、Mill-19/UrbanScene3D 无人机航拍自训 3DGS（gsplat 1–2 h/场景）、或自采校园（最贴合真机试飞场地）。
- 定位结论：当前是「微型无人机单目 RGB 的语言条件反应式导航」，按 AerialVLN/CityNav/OpenFly 口径还不算 VLN；最短补齐路径见计划文档 §6。

## 未完成 / 待办

0. 两数据集对比（上面）：等 forest agent 与 3DGS agent 交付 → B 放量 ≥300 集（建议 hold 统一 1.0）→ 写 `tools/compare_datasets.py` → 守护进程跑 → 填 §2 矩阵
1. 等 E1 流水线跑完（预计 04:00 起 ≈ 2 h）：看 `summary.md`，把四组数字填进 `docs/experiments/2026-11-07-e1-demo-plan.md` §3 的表；teacher / 新学生组的三视角视频在 `eval-<组>/videos/`
2. 若 pilot 的 audit FAIL（`class_share` 之类样本量问题 vs. 执行器问题）：先看 `pilot/audit.json`，确认是样本量再把检查名写进 `CFG["audit_ignore"]` 用 `--from collect` 续跑
3. demo 素材：SITL 自检曲线 PNG（`$DATA_ROOT/uav_rgb/tmp/agent-sitl-tools/selftest-run5/out/*.png`）、3DGS Callinstrasse 视频、Gazebo 三视角视频，汇总成一页给老师看
4. `docs/TASKS.md` SIM-E1 行在流水线出数后改 done 并链 worklog；`sim/README.md` 补 E1 用法
5. 清理：`fastlab-20261106`、`src-algo-20261106` 两个分支已全部进 main，worktree 在 `$DATA_ROOT/uav_rgb/tmp/wt-*`，确认不再用后 `git worktree remove` + 删分支（未删，等用户点头）
6. 教师线下一步（`docs/TEACHER_LINE_PLAN.md`）：DAgger 数据聚合、价值软标签、可见性权重；`teacher_kino` 接入 collect 的 `--teacher-planner`
7. 真机：称重、推力曲线、ToF/光流噪声实测，更新 `sim/dynamics/e1_mini.json` 并把 `calibrated` 置 true
