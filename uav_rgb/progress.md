# progress.md（新开会话先读这个）

- 更新：2026-11-07 04:05
- 代码仓库：`/root/claude-code-backup-20261028/code/uav_rgb`（main）；GitHub 镜像 `git@github.com:xu967/ALVLN.git` 的 `uav_rgb/` 子目录，用 `bash tools/git_mirror_alvln.sh` 推
- 数据盘：`$DATA_ROOT/uav_rgb`（`/root/rivermind-data/uav_rgb`）；构建产物、数据集、实验输出都在这里，不进仓库
- 任务边界：`$DATA_ROOT/uav_rgb/grill-me/boundary-v4.txt` + `boundary-v5-addendum.txt`；任务板 `docs/TASKS.md`（SIM-E1）；实验计划 `docs/experiments/2026-11-07-e1-demo-plan.md`；长期规划 `docs/ROADMAP.md`、`docs/TEACHER_LINE_PLAN.md`

## 正在跑的守护进程（先看这里）

| 实验 | 启动 | 进程 | 看进度 | 停 |
| --- | --- | --- | --- | --- |
| E1 流水线 `tools/e1_pipeline.py`（300 集采集 → 审计 → 打包 → 训练 30 轮 → 20 seed 四组闭环含三视角视频 → 汇总；pilot 已过） | 2026-11-07 04:25（`--from collect`） | PID/PGID 424699，`$DATA_ROOT/uav_rgb/e1-pipeline-20261107-0417/run.pid` | `tail -f …/run.log`；每步状态 `…/status.json`；结束落 `…/DONE`（rc + 最后一步）；评测各组日志 `…/eval-<组>.log`，视频 `…/eval-teacher/videos/`、`…/eval-student-e1-new/videos/` | `kill -- -424699`（collect_parallel 已装 SIGTERM 处理器，会把 gz 一起收掉） |
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
