# 2026-11-09 接手：现状分析（6 个只读 agent）→ 合并四个未提交 worktree → 结果入档 → 第八阶段派工

- 时间：2026-11-09 16:40 起（Cursor 主会话）
- 范围：`uav_rgb` 全仓（合并 `wt-r2-uidocs` / `wt-viz-layout` / `wt-fix-sitl` / `wt-fix-3dgs`）、`docs/`、`progress.md`、SITL 库重建、两个守护进程、第八阶段 6 个开发 agent
- 结论：部分完成（合并与入档完成并推 GitHub；eval4 基准、第八阶段 agent 的产物待收）

## 做了什么

- 16:40 盘点：main @ 1552899（11-07 17:18），GitHub 同步；两个守护进程 DONE 未入档；四个 worktree 未提交；第七阶段 agent 一行未写；GPU 换回 RTX 2080 Ti。派 6 个只读 agent，报告在 `$DATA_ROOT/uav_rgb/tmp/analysis-20261109/`：`PLAN_STATUS.md`（规划对账、26 条待办、18 处矛盾）、`WORKTREES.md`（四个 worktree 可合并性与测试实跑）、`EXPERIMENTS.md`（sgvis 判定、森林、基准 sha、文档错漏 11 条）、`AUDIT_SRC.md`（2080 Ti 构建 0 warning、CTest 30/30、吞吐 1.4–1.7k 样本/秒、p50 0.645 ms、同种子逐位复现；6 Should）、`AUDIT_SIMTOOLS.md`（3 Must：续跑判定 / 孤儿 gz）、`AUDIT_SIM.md`（profile 不盖库代码、3DGS 碰后穿模帧进训练集等 5 Should）。
- 合并（每个一个提交，均先跑该域测试）：
  - `045f759` r2-uidocs：5 个无冲突文档 + worklog；README/CLAUDE/HANDOFF/sim-README 与 main 8 块冲突取 main；`env.sh` 不带（worktree 把 GPU 写成 4080）。
  - `271e657` VIZ-LAYOUT 二轮：`e1_body.py` 两域 E1 机体、chase_cam 录像补丁、VIDEO_QUALITY 三档、ffmpeg libx264 `VideoSink`、reel v2 配置。合并时删 `eval_closed_loop` 重复键、`requirements.txt` 两行都留、不带 `demo_3dgs_callinstrasse.py` 改动。四组测试 + 3DGS video 自检 + reel self-check PASS。
  - `85eb7ef` REVIEW2_UIDOCS M1：`train_student_with_video` 对 cf_sitl 加 `--vehicle e1_mini` + 测试（修前 ValueError）。
  - `7c974ef` 文档：sgvis（10/20，判据不通过，非干净消融）与森林 lib v3（教师 20/20、BC 0/20、截断 4/20、盲前进 0/19）进 demo-plan §3.4/§3.5 与 scoreboard；demo-plan §3.1 dev F1 0.38→0.396、外径 0.18→0.20；GPU 文档更正；PLAN「更新 2026-11-09」五条决策；progress.md 头部。
  - `a156800` lib v3.1：`vel_ki` 0.30、selftest 阈值、两个新 CTest、`sweep_seeds.sh`（构建目录改必填）、worklog；`tmp/build-cf-sitl` 重建为 v3.1（旧 v3 → `build-cf-sitl-v3-20261107`），ctest 10/10、selftest 33/33、40 种子 0 红。
  - `2899428` FIX-3DGS D1/D5/D6 + worklog（test_collect_eval 21 OK、test_scene_import_wanderland 6 OK）。
- 全量 `tools/check.sh`（2899428）：CTest 30/30、cf_sitl 9/9、selftest 33/33、Python 全绿，PASS。`bash tools/git_mirror_alvln.sh` 推 GitHub：ALVLN main d875daf → 1e1a2ce（6 个提交）。
- 守护进程：
  - 短片 v2 `$DATA_ROOT/uav_rgb/demo-e1-20261107/reel/v2-run/`（17:11–17:13，rc=0）：`demo_reel_v2.mp4` 239 s 2560×1440 h264 61.7 MB、`demo_reel_v2_short.mp4` 57 s、`stills_v2/` 11 张。
  - GPU 对拍 `$DATA_ROOT/uav_rgb/eval3-gpucheck-20261109/`（17:22 起，PGID 188213，2 个 gz）：eval3 的 BC 与截断两臂在 2080 Ti + lib v3 + 同 infer 快照 + 同 20 seed 重跑，`summary.md` 给逐 seed 一致数；停 `kill -- -188213`。
- 第八阶段 6 个开发 agent 17:55 派出（简报 `tmp/phase8-20261109/BRIEF.md`，worktree `tmp/wt-{fix-a,teacher-cost,dagger-gz,student-select,fix-b,fix-sim}` 基于 2899428）：FIX-A（评测成功线停 + `sim09.v2` + 续跑判定 + SIGTERM + seed 单源 + helper 去重）、TEACHER-COST（统一代价模型教师，BRIEF_ALGO §1）、DAGGER-GZ（Gazebo DAgger 流水线，round0 = 截断臂，混包开关）、STUDENT-SELECT（`--save-every-epoch`、`select_ckpt.py`、3 seed 训练噪声、双帧反事实）、FIX-B（src 批次 B + AUDIT_SRC）、FIX-SIM（3DGS D2/D3/D7/D10/D11 + profile 加库指纹 + 碰后帧标碰撞）。

## 没做什么

- eval4 基准重跑（2080 Ti + lib v3.1 + `sim09.v2` 六臂）。原因：要等 FIX-A 的评测口径改完合并，否则又要重跑一次；预计今晚起。
- DAgger 正式 3 轮、TEACHER-COST 闭环、STUDENT-SELECT 三个 seed 的闭环。原因：agent 在做脚本与离线部分；闭环由主会话按 gz 闸门排在 eval4 之后。
- 删除已合并的 worktree（`wt-fix-sitl` / `wt-fix-3dgs` / `wt-r2-uidocs` / `wt-viz-layout`）与冻结快照 `wt-e1-sgvis`。原因：progress 旧待办 5 写「等用户点头」；`wt-xeval-forest` 正被 GPU 对拍守护进程使用，不能删。空的 `wt-fix-b` 已删并从新 main 重建给 FIX-B。
- r2-uidocs 对 README/CLAUDE/HANDOFF/sim-README 的改动。原因：与 main 冲突且 worktree 侧更旧、GPU 写错；信息量只有「CTest 30」且 main 已含。
- VIZ 二轮对 `tools/demo_3dgs_callinstrasse.py` 的 2 行。原因：FIXLIST D11 拍板删该脚本，FIX-SIM 本轮删。
- FIXLIST D2/D3/D7/D8/D10/D11、AUDIT_SIM 各条。原因：派给 FIX-SIM。
- `BRIEF_ALGO.md` §0.2/§0.3 的 4080 / lib v3 数字未改。原因：第八阶段 BRIEF §1 明文替代；旧简报保留原貌作记录。
- M1 补项（第二个 Wanderland、stump 路线、每场景 20 集 pilot）。原因：PLAN 11-09 决策 5 排到 M3 之后。
- 用户验收短片 v2 与打 tag `demo-2026-11-12`。原因：需用户看过。

## 待确认

1. 已合并 worktree 与 `wt-e1-sgvis` 是否现在删（`git worktree remove` + `git branch -d`）。
2. 短片 v2 是否符合预期（`reel/stills_v2/` 先看代表帧）；满意就打 tag。
3. PLAN §8 四个待拍板项仍悬：val 比例 / best-ckpt 判据、自采设备与场地、DL3DV / ScanNet++ 账号、SAFE 语义教师侧。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `tools/check.sh`（全量，2899428，`tmp/analysis-20261109/check-full-2899428.log`） | `100% tests passed, 0 tests failed out of 30`；cf_sitl `100% tests passed … out of 9`；selftest `0 项失败，33 项通过`；末行 `PASS: 所选检查完成` |
| viz 四组测试（test_view_layout / test_record_views / test_eval_closed_loop_args / test_e1_body） | 全 PASS（含 high 档 libx264 2560x1440） |
| `sim/3dgs/video.py` 自检 | `{"3dgs_video": "PASS", "frames": 3, "files": 4, "views_frames": 25, "body_px": 90.0, "high_backend": "ffmpeg libx264 crf 18"}` |
| `make_demo_reel.py --self-check` | 三档 PASS（high 2560x1440 20.5 KiB / 2 s） |
| `test_train_student_with_video.py` | 修前 `ValueError: '--vehicle' is not in list`；修后 PASS |
| cf_sitl v3.1 重建（`tmp/build-cf-sitl`） | ctest 10/10；selftest 33/33；`sweep_seeds.sh` 40 种子 0 红；sha `libcf_sitl.so` 75ca0301 / `gz/libCfSitlSystem.so` 4f41a41a / `e1_mini.json` 683e7400 |
| 3DGS 两组单测（3dgs-venv） | `Ran 21 tests OK`、`Ran 6 tests OK` |
| GitHub 推送 | `d875daf..1e1a2ce main -> main`；`ALVLN/uav_rgb/.mirror_src_commit` = 2899428 |
| 短片 v2 | `DONE rc=0`；ffprobe 239.0 s 2560x1440 h264 10 fps；短片 57 s |
| GPU 对拍守护进程 | 17:22 起跑，2 个 gz，`gpu.txt` = `NVIDIA GeForce RTX 2080 Ti, 580.119.02`；结果未出 |

## 风险与遗留

- 从 `a156800` 起所有采集 / 评测都是 lib v3.1 口径（`dynamics_profile` 683e7400）；eval3 表不再加臂；现成数据集（0417 bd0dcd43、sgvis cec5d6b0）与新采集的集 `preprocess` 会拒绝混包，DAgger 流水线要显式 `--allow-mixed-dynamics`（DAGGER-GZ 在做）。
- FIX-SIM 的「profile 加库指纹」会再换一次 profile 字符串，要在 DAgger 正式跑之前合并。
- 第八阶段 6 个 agent 并行改 `eval_closed_loop.py`（FIX-A 成功判定 / TEACHER-COST choices）与 `src/`（FIX-B / STUDENT-SELECT），合并时可能有小冲突，主会话解。
- 短片 v2 片中柱阵片段仍是 eval2（lib v2）录像，片尾表是 eval3 数字，字幕已注明；eval4 出数后可再出 v3。

## 更新 2026-11-09 19:30（用户追问后：3DGS 场景评测、下载、GPU 对拍）

### 做了什么

- 盘点已下载的 3DGS 数据：已配路线 4 个场景（Callinstrasse、Wanderland 1MlUucug、garden、bicycle），已下载未建场景 1 个（stump，`downloads/inria-3dgs-pretrained/stump/`，SCENE-B 只做了重力对齐，缺尺度参照）。
- 3DGS 跨域闭环矩阵守护进程 `$DATA_ROOT/uav_rgb/xeval-3dgs-20261109/`（PGID 321059）：Gazebo 柱阵训的 E1 学生（BC / 截断 / sgvis）+ 教师 A* + 盲前进 × 4 场景 × 6 路线 × 5 集，hold 1.0（学生训练口径），同 seed 同出生位姿可配对；冻结 worktree `tmp/wt-xeval-3dgs`（77927d4）；本轮专用驱动 `xeval_3dgs_driver.py` 每场景只加载一次；汇总 `summarize.py` 报 SR(stop)（到点即停口径，与 sim09.v2 同义）与 SR(v2)、碰撞、SPL、配对 McNemar、Callinstrasse AGL 可核子集。汇总脚本先在 garden 冒烟集上空跑通过。
- 对照轮 `xeval-3dgs-20261109-teacher-h05/`（PGID 337265）：等主轮 DONE 后只跑教师 hold 0.5（同出生位姿），给 3DGS 数据集 B 的 hold 决策。
- gsplat sm_75 缓存放到 `$DATA_ROOT/uav_rgb/torch-cache/extensions-sm75-20261109`（AUDIT-SIM 重编好的副本）；外层脚本写死的 `extensions-sm89-20261106` 不动，等 FIX-SIM 改成按算力拼目录名。
- GPU 对拍结果写进 demo-plan §3.3 补记与 scoreboard §1（BC 7/17、截断 15/17，逐集成功一致 13/17；6 集墙钟超时）。
- SCENE-3DGS 任务书写进 `tmp/phase8-20261109/BRIEF.md` §9（本轮只接 stump），等 agent 空位派。

### 没做什么

- treehill / flowers / truck / train 与第二个 Wanderland 场景的下载。原因：本机代理 127.0.0.1:7897 无进程监听，HuggingFace / GitHub 直连超时，Inria 直连实测 29 KB/s（一个场景约 8 h）；按用户指示放弃。下载脚本留在 `downloads/inria-3dgs-pretrained/_dl-20261109/run.sh`，代理恢复后可直接重跑。
- 3DGS 教师数据 pilot / 放量 / S_B 训练。原因：FIX-SIM 正在修三处会写进数据的问题（碰后穿模帧不标碰撞、pilot 碰撞即停与 preprocess 拒收的矛盾、`dynamics_profile` 不含库版本），合并前采的数据要么带这些问题、要么以后不能和新数据混包；另外 hold（0.5 / 1.0）待用户拍板。
- 跨域评测里没放旧 X3 学生、没录三视角视频。原因：控制时长；每集 rgb / actions / oracle / trace 都保留，之后可用 `python -m sim.3dgs.video --ep` 挑集合成。
- SCENE-3DGS agent 没有立即派。原因：已有 6 个 agent 在跑，沿用并发上限 6。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `summarize.py` 在 garden 冒烟集空跑 | 表格正常输出，2 分 9 秒（建占据栅格为主） |
| 驱动首场景加载 | garden / bicycle 各 71–72 s，`dynamics_profile=cf_sitl_v1:683e7400`；之后每路线 15–50 s |
| 下载测速 | `curl` 直连 20 MB 区间 60 s 超时，28.8 KB/s；代理端口无监听 |
| GPU 对拍 | `DONE rc=1`；summary 见 demo-plan §3.3 补记 |
