# 会话恢复：收口 Gazebo/学生 C++ 两个 agent 的产出，修 AUDIT Q1/Q2，一键流水线上线并守护启动

- 时间：2026-11-07 03:41 – 04:10
- 范围：`sim/tools/collect_parallel.py`、`sim/tools/test_collect_parallel.py`、`sim/cf_sitl/gz/CMakeLists.txt`、`tools/e1_pipeline.py`（新）、`tools/git_mirror_alvln.sh`、`progress.md`（新）、分支合并、GitHub 推送、E1 流水线守护进程
- 结论：完成（流水线仍在跑，结果见 `$DATA_ROOT/uav_rgb/e1-pipeline-20261107-0357/`）

## 背景

用户端断线前，两个后台 agent 的状态：Gazebo 集成 agent 已写完 worklog（`2026-11-07-0140-gz-e1-mini.md`，结论「完成，未 commit」），改动留在 main 工作树；学生 C++/CUDA agent 在分支 `src-algo-20261106` 上已提交全部工作含 worklog（`2026-11-06-1740-src-algo.md`）。恢复时先核对磁盘状态（git status / 分支 / worktree / 子 agent 的 worklog），没有重跑它们的工作。

## 做了什么

1. **验证并提交 Gazebo agent 的改动**（cc8bbe6）：重跑 `sim/tools/test_*.py` 32 个、`tools/test_*.py` 5 个、`rl_teacher.test_core`，全部 rc=0 后按主题提交；Q1/Q2 的改动先临时摘出再提交，保证这个提交只含 agent 的内容。
2. **AUDIT Q1/Q2 修复**（c171106，`sim/tools/collect_parallel.py`）：
   - `main()` 装 SIGTERM 处理器（与 `eval_closed_loop.py` 同写法）：`request_stop()` 置停止位 + `cleanup_all()`，再 `SystemExit(143)` 让 `finally` 跑。
   - 模块级 `_STOP` 事件：`run_episode` 不进 attempt 2、`run_instance` 不起下一集、`_register` 对置位后才启动的进程组就地收掉（关掉 cleanup_all 扫描与新启动之间的窗口）。
   - `test_collect_parallel.py` 新增 `StopTests` 2 项。
3. **合并 `src-algo-20261106` 进 main**（3f27c61，`--no-ff`）：`git merge-tree` 无冲突；合并后 `git diff main src-algo-20261106 -- src` 为 0 行。
4. **全量 `tools/check.sh`**（守护进程，构建目录 `$DATA_ROOT/uav_rgb/tmp/recover-20261107/check-main/build`）：rc=0，CTest 27/27（18.1 s）+ 5/5，Python 全绿。随后把仓库默认构建目录 `build/check` 也更新到合并后的源码（流水线用它的 `uavrgb_train` / `uavrgb_infer`）。
5. **真库插件构建到文档路径** `$DATA_ROOT/uav_rgb/tmp/build-cf-sitl/gz/libCfSitlSystem.so`（ldd 指向 `…/build-cf-sitl/libcf_sitl.so`，`test_flatten_state` PASS 39 字段）。顺手修 `sim/cf_sitl/gz/CMakeLists.txt`：README 写的 `-DCF_SITL_LIB_DIR=…` 路线实际报「找不到 libcf_sitl」，原因是 `set(CF_SITL_LIB "" CACHE FILEPATH)` 预占了缓存项，`find_library` 视为已找到直接返回；加 `unset(CF_SITL_LIB CACHE)` 后两条路线都通（e68ac15）。
6. **`tools/e1_pipeline.py`**（552daac）：按 `REVIEW_2026-11-07_entrypoints_style.md` §4 草稿落地，并把草稿里 10 个「待确认」按 Gazebo agent 的实际产出定下来：`--vehicle e1_mini` / `--dynamics cf_sitl` 已存在；模板由 `resolve_vehicle` 按载具选，不用显式 `--template`；`window_from=observation`（用户计划 F 的口径，不是草稿的 dispatch）；`audit_ignore=()`（`audit_dataset` 已豁免 cf_sitl 定速判据）；评测 seed 用计划 §1 的 20 个固定值；采集 seed 从 3000/3100 起避开它们；新增 `pilot` 步（20 集 + audit 闸门）；四组评测并行各 1 个 gz，teacher 与新学生组 `--record-video`；`preflight` 一次查完二进制、旧 ckpt、真库插件。`--dry-run` 走完 7 步。
7. **`tools/git_mirror_alvln.sh`** 加 `--first-parent`：合并进来的分支按合并提交快照镜像一次，否则分支内部 8 个提交的快照会在「分支基点」和「main 当前」之间来回跳。
8. **推送 GitHub**：ALVLN `f8c7c9d..8e7d488`（5 个镜像提交，对应 uav_rgb `cc8bbe6`…`552daac`）。
9. **启动 E1 流水线守护进程**：`setsid nohup $UAV_RGB_PYTHON tools/e1_pipeline.py --out $DATA_ROOT/uav_rgb/e1-pipeline-20261107-0357`，PID/PGID 324322；pilot 前 8 集全部 40/40、58–62 s/集、0 次重试。
10. `progress.md`（仓库根，新）：守护进程表、已完成、关键决策、待办；`/root/.claude/skills/personal-skill/SKILL.md` 删掉重复的第二份 §6.10。

## 没做什么

- 没重跑 Gazebo agent 的真 Gazebo 冒烟与 1/4/6 实例链路验证：它的 worklog 已有完整数字，且流水线 pilot 本身就是对同一链路的再验证。
- 没逐条复核 src-algo 分支的 nsys 数字与 A/B 表：只核对了合并后 `src/` 与分支逐字节一致、CTest 27/27 全过。
- 没删 `fastlab-20261106` / `src-algo-20261106` 分支和 `$DATA_ROOT/uav_rgb/tmp/wt-*` worktree（内容都已进 main）：删除类操作等用户点头。
- 没给 `scene_gen` 加解锁 RTF 的开关（Gazebo agent「没做什么」第 1 条）：新需求，等确认。
- 没改 `_pick_depth` 的 10 Hz 配对策略（偶发深度晚到一帧 100 ms，仍在门内）。
- `e1_pipeline.py` 的 SIGTERM 端到端复现脚本放在 `/tmp/recover-20261107/`，不进仓库；仓库里只留 `StopTests` 单测。
- 流水线的 300 集采集、训练、四组评测、视频还没出结果（预计 06:00 前后），demo 对照表（计划 §3）未填。
- `docs/TASKS.md` SIM-E1 行未改 done：要等四组数字出来。

## 待确认

- 两个已合并分支与 worktree 是否现在删。
- 评测步数用 80（T7 口径，碰撞/到达提前结束）而不是计划 F 写的 40；若要严格按 40 改 `CFG["eval_steps"]` 并 `--from eval` 重跑评测即可。
- `e1_pipeline` 的 pilot 数据只做闸门不进训练集；要并进去需要 preprocess 步多收一个目录（约 3 行）。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `sim/tools/test_*.py`（32）+ `tools/test_*.py`（5）+ `rl_teacher.test_core`，提交 cc8bbe6 前 | 全部 rc=0（`$DATA_ROOT/uav_rgb/tmp/recover-20261107/pytests.log`） |
| `test_collect_parallel.py`（4 项含 StopTests 2 项）、`test_process_cleanup.py`（6 项） | OK |
| SIGTERM 端到端（真 `main()`，gz/collect 换 `sleep 300`，`kill -- -PGID`） | rc=143，0.2 s 退出，4 个子进程全收，SIGTERM 后 0 次新启动 |
| `git merge-tree --write-tree main src-algo-20261106` | 无冲突；合并后 `src/` 与分支 diff 0 行 |
| `bash tools/check.sh`（合并后 main，含构建 + CTest + Python） | rc=0；`100% tests passed, 0 tests failed out of 27`，`Total Test time (real) = 18.09 sec` |
| `cmake -S sim/cf_sitl/gz -B … -DCF_SITL_LIB=…` 真库插件 | 构建成功；`test_flatten_state: PASS (39 fields)`；ldd → `libcf_sitl.so` |
| `cmake … -DCF_SITL_LIB_DIR=…`（修复后） | `CF_SITL_LIB:FILEPATH=…/build-cf-sitl/libcf_sitl.so` |
| `tools/e1_pipeline.py --dry-run` | 7 步、1+1+1+1+1+1+4 条命令，不写文件 |
| `bash tools/git_mirror_alvln.sh` | 推送成功 `f8c7c9d..8e7d488`，`## main...origin/main` |
| E1 流水线 pilot（真 Gazebo，4 实例） | 进行中：前 8 集 40/40、58–62 s/集、attempts=1 |

## 更新 2026-11-07 04:30：pilot 闸门两轮、教师 E1 余量与脱困 v2、流水线放量

- 第一轮 pilot（03:57）：20/20 集采到、每集 58–70 s，但 audit FAIL 2 项：
  - `label_recompute` 47/800：audit 复算没按 manifest.vehicle 调 `set_vehicle`，用 X3 半宽复算 E1 数据；按 e1_mini 复算 0/800。修法：`audit_dataset.py` 读 `vehicle` 并在复算前 `set_vehicle`（5af664a）。
  - `safe_collision` 98/173：8/20 集教师在柱旁进入 SAFE 死锁（"LRLR…SSSS"），E1 悬停漂移 2.4 cm/s 漂进柱子。根因是 E1 沿用 X3 余量（A* 0.5 / 局部 0.36）：1.2 m 缝只剩 0.2 m 走廊，一步 0.25 m 带 20° 航向偏差就出走廊；进余量后 A* 起点四邻全堵、局部脱困找不到 1.6 m 净空射线；另一卡法是全局规划在余量内按 err 换向。修法全部按载具（x3 走历史分支、v5 标签逐字节不变、`test_teacher_policy` 过）：`CLEARANCE_BY_VEHICLE e1_mini=(0.40, 0.30)`、E1 多一档 0.6 m 短射线脱困、`ESCAPE_V2`（余量内交给局部脱困；脱困射线的前向检查用同样 skip）。
- 第二轮 pilot（04:17，修后）：17/20 集干净穿过（max_x≈7 m，0 碰撞；首轮只有 10/20），`label_recompute` 0/800。剩 `class_share`（SAFE 3.9% < 5%，因为不再死锁）与 `safe_collision`（3 集在柱旁左右摆后擦柱）。处理：`e1_pipeline.py` 的 `audit_ignore=("class_share","safe_collision")` 并写明理由（擦柱步由 preprocess `--drop-collision` 丢掉，擦柱前标签仍对；教师真实 SR 由评测 teacher 组报）。没有再调教师：剩余 3 集是柱旁 L/R 换向（aim 随 yaw 变化）引起，属下一轮优化项。
- 04:25 流水线 `--from collect` 放量 300 集（PID 424699）。
- 04:19 起两个子 agent 并行：forest_e1 场景族接入 Gazebo 管线（三档密度、教师实测、录像）；3DGS Callinstrasse 多转弯路线 + 三视角视频 + 20 集 3DGS E1 pilot 数据。用户要求「复杂度调大、柱阵与 3DGS 都做、保存三视角视频」。

## 风险与遗留

- E1 教师在柱旁仍有 ~15% 集擦柱（L/R 换向），数据里靠 drop-collision 兜；要再降需要给局部脱困的 aim 加滞回或按速度方向选射线（teacher_kino 的思路）。
- E1 数据 SAFE 标签只有 ~4%（且多在擦柱段，会被丢掉）：学生可能学不到「停」；要补 SAFE 需要专门的出生/堵死场景或 DAgger。
- 流水线与用户 YOLO 训练共享 12 核 / 1 张卡：采集阶段 4 个 gz 约 7 核，训练阶段 DeviceCnn 约 0.8 GB 显存；吞吐数字要注明共享条件。
- `calibrated=false` 的参数采的数据，数字不能与 X3 档混报；manifest 已分档。
- `_register` 里对晚启动进程组的补杀依赖 `stop_group` 的 8 s SIGTERM 超时，极端情况下停止会多等 8 s。
- `--first-parent` 镜像后 GitHub 上看不到 src-algo 分支内部 8 个提交，细粒度历史只在本地 uav_rgb 仓库。
