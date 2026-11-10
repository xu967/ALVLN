# uav_rgb 架构审查报告（只读）

## 结论摘要

1. 两条仿真线同名口径、实际时序不同。同一动力学指纹 `cf_sitl_v1:1bc8fedf:lib31`、同为 hold 1.0：Gazebo 每 0.900 s 一帧观测，动作在观测后 0.15 s 才派发、生效约 0.85 s 后固件回 SAFE 约 0.05 s；3DGS 每 1.000 s 一帧，零延迟、无空档。相邻两帧之间的位姿变化：连续前进 0.255 对 0.299 m（+17%），连续转向 28.7° 对 34.4°（+20%），单发转向 22.6° 对 31.6°（+40%）。已有的跨域结论（学生 0.24 到 0.32 对盲前进 0.31）是在这个差别下测的。仓库内流水线读的还是到点不停的 `3dgs.pilot.v2`：同一批轨迹 BC 学生 18/120，按到点即停是 31/120。
2. 不建议现在抽「仿真后端接口 + 统一 episode runner」。漂移在判据、manifest 和时序口径，不在步循环。更便宜的办法：14 个纯判据函数（351 行，只依赖 `math`）移出 `eval_closed_loop.py`；库外的 xeval 驱动（5 份相同副本）并回 `eval_3dgs.py`；`sim/3dgs` 的 86 个测试接进 `tools/check.sh`（现在 0 个在门禁里，本次实测全过，约 65 s）。
3. 可清理的量大、风险低：19 个分支全部 `ahead=0`；24 个 worktree 里，除另一会话的 `cursor-1110/wt` 外只有 2 个含 main 没有的在途内容（`wt-teacher-3dgs`、`wt-sim-dyn`），1 个被运行中的流水线占用（`wt-eval4`）；数据盘 91 GB 里约 23 GB 在仓库内没有代码引用（其中 6.2 GB 是 pip 缓存）。
4. 工程风险第一位：完整历史没有机外副本（uav_rgb 无 remote，镜像只带 first-parent 快照），而 `.git` 只有 22 MB，多推一个分支即可。第二位：并发会话直接改主工作树，审查期间未提交项从 7 个增到 11 个，`uav_rgb_e1.sdf` 在工作树里先后出现两个未提交版本。
5. 代码依赖草稿目录：非测试代码里 `tmp/agent…` 出现 22 处，其中 3 处是流水线输入；32 个代码文件的注释引用了只存在于 `$D/tmp` 的任务书（BRIEF / FIXLIST，共 42 行）。

## 0. 范围与标注

- 对象：`/root/claude-code-backup-20261028/code/uav_rgb`，main @ c85a1ff，242 个提交；数据盘 `$D=/root/rivermind-data/uav_rgb`；审查日期 2026-11-10。
- 行号取自 HEAD（`git show HEAD:<文件>`）。并发会话正在改 `eval_closed_loop.py`、`record_views.py`、`e1_body.py`、`uav_rgb_e1.sdf`，主工作树里这几个文件的行号与 HEAD 不同。
- 【已核实】= 读了代码或用磁盘上的现有数据实测；【怀疑】= 有迹象，没有跑实验确认。
- 实测脚本与输出在 `$D/tmp/cursor-1110/agents/arch/`（`measure_*.py`、`measure_*.txt`、`wt_*.txt`、`import_graph.*`）。
- 任务书列出的 6 条已知问题只引用，不重复报。

## A. 架构地图

### A1. 分层图（下层不依赖上层；Python 与 C++ 只在文件格式处相接）

```text
L7 工作台   ui/studio.py(3606) --subprocess--> scene_gen / eval_closed_loop / trainer_native --exec--> uavrgb_train
            ui/monitor.py(1208)：独立 Tk 监控台；studio 只用它的 tail_text / tail_events
L6 流水线   tools/e1_pipeline.py        -> collect_parallel -> audit_dataset -> preprocess -> uavrgb_train -> eval_closed_loop
(守护进程,  tools/dagger_gz_pipeline.py -> 同一条链按轮次循环（import e1_pipeline 取 CFG 与 GZ_ENV）
 顶部 CFG)  tools/dagger_pipeline.py    -> collect_3dgs -> teacher_belief -> preprocess + join_soft_targets -> train -> eval_3dgs
            tools/compare_datasets.py   -> 上面两条链各跑一遍，再做两域交叉评测
            tools/pipeline_common.py(237)：log / run / status.json / DONE / 续跑判定 / wilson_ci(第二份)
L5 统计录像 tools/stats_report.py  tools/select_ckpt.py  tools/make_demo_reel.py
            sim/tools/record_views.py(511) --+
            sim/3dgs/video.py(495) ----------+--> sim/tools/view_layout.py(210) + e1_body.py(185)
L4 评测     sim/tools/eval_closed_loop.py(1311) --subprocess--> collect.py --policy student --stop-at-goal
            sim/3dgs/eval_3dgs.py(376) --函数调用--> collect_3dgs.run_episode；import eval_closed_loop 取判据函数
L3 采集     sim/tools/collect_parallel.py(982) --N x (gz sim + collect.py)--> sim/tools/collect.py(1884)
            sim/3dgs/collect_3dgs.py(706)：run_episode() 同步循环
L2 教师     sim/tools/teacher.py(1078：柱阵 A* v5/v6 + costmap)  teacher_kino / teacher_belief / teacher_predictive
            sim/3dgs/planner.py(391：占据栅格 A*)   teacher/(GRU 特权教师, torch)   rl_teacher/(PPO, 未训练)
L1 场景     Gazebo: scene_gen / scene_forest / campus / asset_campus -> world.sdf + scene.json；gz_io.py 订阅话题
            3DGS:   scene.py(scene.json + PLY) -> renderer.py(gsplat) + collision.py(体素占据代理)
L0 动力学   sim/cf_sitl/(C++：固件在环库 + gz 插件 CfSitlSystem + cf_sitl_proc) + sim/dynamics/e1_mini.json
            Gazebo: cf_sitl_exec.py(话题桥)      3DGS: dynamics.py(cf_sitl_proc 子进程 / cf_blackbox)
            旧档:  x3_velocity(三个 CLI 的默认值)  cf_stab(collect.py 内嵌)  cf_sil.py(987)  cf_blackbox.py(295)
=========================== 文件格式边界 ===========================
episode 目录(navlab.episode.v1) --preprocess.py--> dataset.uavr(UAVR v1, 内嵌 meta) + dataset.aux(UAVRAUX1) + soft_targets.bin
C++ src/(14.9k 行): uavrgb_train -> ckpt(UAVRCK01 v6)；uavrgb_infer(常驻进程) <-- sim/tools/student.py
```

### A2. 各层入口与对外接口

| 入口（行数） | 形式 | 产物 | 调用方（非测试 import 方数） |
| --- | --- | --- | --- |
| `sim/tools/scene_gen.py`（691）及 forest / campus / asset_campus | CLI + 库 | `world.sdf`、`scene.json`（几何、目标、gid、相机档） | collect_parallel、eval_closed_loop、studio（9） |
| `sim/tools/collect_parallel.py`（982） | CLI，33 个参数 | `<out>/{train,val,test}/<seed>/`、`collection_manifest.json` | 三条 Gazebo 流水线以子进程调用（4） |
| `sim/tools/collect.py`（1884） | CLI，31 个参数 | `episode_manifest.json`、`actions.jsonl`、`oracle.jsonl`、`rgb_%05d.png`、`scene.json` | collect_parallel、eval_closed_loop 以子进程调用（1） |
| `sim/3dgs/collect_3dgs.py`（706） | `python -m`，24 个参数；函数 `run_episode` | `<out>/ep_%05d/`（多一个 `trace.jsonl`）、`run.json` | dagger_pipeline、compare_datasets、eval_3dgs、库外驱动（3） |
| `sim/tools/audit_dataset.py`（1133） | CLI | 检查项与退出码 | e1_pipeline.py:175、dagger_gz_pipeline.py:423（0） |
| `tools/preprocess.py`（1068） | CLI，11 个参数 | `dataset.uavr`、`instructions.jsonl`、`dataset.aux` | 四条流水线（3） |
| `uavrgb_train` / `uavrgb_infer` | C++ CLI | `metrics.jsonl`、`ckpt_best.bin`、`results.csv` | 流水线、`ui/trainer_native.py`；`student.py` 管理 infer 子进程（8） |
| `sim/tools/eval_closed_loop.py`（1311） | CLI，35 个参数 | `eval.json`（`uavrgb.eval.v1`）+ 逐集目录 | 三条 Gazebo 流水线、studio（2） |
| `sim/3dgs/eval_3dgs.py`（376） | `python -m`，21 个参数 | 同 schema 名的 `eval.json`，字段集不同 | dagger_pipeline、compare_datasets（1） |
| 四条流水线 | 顶部 `CFG` + `--out` | `status.json`、`DONE`、`run.log`、`inputs.json` | 人工以守护进程启动 |

### A3. 版本与指纹字符串

| 字符串 | 定义处 | 谁校验 | 何时让旧数据作废 |
| --- | --- | --- | --- |
| 几何版本 `uavrgb.geom.v5`、森林、campus、asset_campus、`3dgs.v1` | scene_gen.py:58；scene_forest.py:39；campus.py:12；asset_campus.py:18；collect_3dgs.py:44 | preprocess.py:49 白名单与 :168–171（`3dgs.` 按前缀放行）；audit_dataset.py:787–789（3DGS 必 FAIL） | 升版后旧集要 `--allow-frozen-geometry`；`SPLIT_SALT = sg.GEOMETRY_VERSION`（split_manifest.py:35），升版同时改变每个 seed 的 split |
| split 策略与 salt | split_manifest.py:26, :35 | preprocess.py:134–141 与 split 复算门；check.sh 的 2 个 split 负例 | 改 salt 或布局常量 |
| 教师版本 `uavrgb.teacher.v5` / `v6`、`teacher_cost.v1`、`teacher_occ.v1` 等 | teacher.py:26, :33, :620；teacher_kino.py:38；teacher_belief.py:43；teacher_predictive.py:14；planner.py:21；collect.py:1818–1820 两个字面量 | 只有 audit_dataset.py:822–846；preprocess 不查，3DGS 集又过不了 audit，所以 3DGS 侧无人校验 | 人工升版 |
| 动力学指纹 `cf_sitl_v1:<参数 sha8>:lib<N>` | cf_sitl_exec.py:47, :121–132；`cf_sitl.h:27`（31） | preprocess.py:53；dagger_gz_pipeline.py:277–285；dynamics.py:177–185 对拍进程库版本 | e1_mini.json 任一 value 或库版本变。磁盘上 E1 数据已有 4 个值：`bd0dcd43`、`cec5d6b0`、`683e7400`、`1bc8fedf:lib31`（11-07 到 11-09） |
| 口径键 `window_from`、`hold_s` | collect.py:1282–1284 参数；collect_3dgs.py:477 字面量 | preprocess.py:53 `STRICT_KEYS` | 任一不同就不能同包 |
| 相机档 `rgb320x240_fov60_v1` | 5 处字面量：scene_gen.py:459、scene_forest.py:271、campus.py:159、asset_campus.py:185、eval_closed_loop.py:65；3DGS 不写 | preprocess.py:59（混包时允许部分缺） | 改相机要 5 处同步 |
| 模板 / 世界 / 资源 sha | split_manifest.py:122 | test_e1_template.py:108–113（live 模板须有同 sha 归档） | 模板任一字节变化 |
| 指标版本 `sim09.v2`（cf_sitl 带 `.coast1s`）、`3dgs.pilot.v2` | eval_closed_loop.py:52；eval_3dgs.py:44 | dagger_gz_pipeline.py:284 复用门；stats_report.py:377 与 e1_pipeline.py:303–307 只提示混合 | 判据变更时升版 |
| 碰撞代理 `3dgs.occ.v2`、`3dgs.occ.v3.points` / `.mesh` | collision.py:48, :50 | 无 | 无人据此作废 |
| `navlab.episode.v1` | collect.py:1782；collect_3dgs.py:436 | 无；两域键集合不同（B2） | 没有升版记录 |
| `UAVR` v1、`uavrgb.dataset.v1`、`UAVRAUX1` | preprocess.py:85–86, :533, :66；train_student_with_video.py:37（第二份版本号） | src/dataset.cpp:474；src/aux_sidecar.cpp:47；check.sh 用 read_dataset.cpp 对拍 | 头布局变 |
| `UAVRCK01` v6（可读 v2 起） | src/checkpoint.cpp:16–17 | checkpoint.cpp:88–92 | 低于 v2 的权重 |
| `uavrgb.eval.v1`、`uavrgb.3dgs_scene.v1`、`uavrgb.3dgs_collect.v1`、`uavrgb.train.v1` | eval.json 有 3 个写入方（eval_closed_loop、eval_3dgs.py:312、compare_datasets.py:295）；scene.py:225 | scene.py:263；monitor.py:43–44 | 无 |

写入方：前 7 行都由采集器写进 `episode_manifest.json`（几何版本另写进 `scene.json`），指标版本写进 `eval.json`。

### A4. import 环与跨层调用【已核实，`import_graph.py` 静态扫描】

- 顶层 import 无环。函数内延迟 import 有 7 个环，全在场景生成这 5 个模块之间：`scene_gen` 与 `scene_forest` 互相 import；`scene_gen` 与 `campus` 互相 import；`scene_forest` 与 `split_manifest` 互相 import；`asset_campus`、`campus`、`scene_gen` 三者成环；`scene_gen`、`scene_forest`、`split_manifest` 三者成环；另有 2 个经 `split_manifest` 回到 `campus` / `asset_campus` 的长环。
- 跨目录边数：tools 到 sim/3dgs 23，tools 到 sim/tools 17，sim/3dgs 到 sim/tools 11，ui 到 sim/tools 6，rl_teacher 与 sim/tools 互相 3 + 2，`sim/cf_sitl/gz/tests` 的冒烟脚本到 sim/tools 3，其余各 1。
- 向上或绕路的调用：
  - `sim/3dgs/eval_3dgs.py:41` 为 7 个名字（6 个纯函数、1 个常量）import `eval_closed_loop`，实测带进 24 个仓库模块（collect、collect_parallel、gz_io、cf_sil、scene_gen、campus 等整条 Gazebo 编排链）。
  - `sim/3dgs/eval_3dgs.py:106` 延迟 import `tools/train_student_with_video.read_uavr_meta`；`sim/tools/compare_teachers.py` import `tools/eval_seeds.py`。`pipeline_common.py:70` 的 docstring 说「那边不 import tools/」，这两处违反同一约定。
  - `eval_closed_loop.py:76` import 整个 `collect` 只为常量 `GOAL_MARGIN_M`。
  - `tools/dagger_gz_pipeline.py` import 另一个入口脚本 `e1_pipeline` 取 `CFG` 与 `GZ_ENV`。

## B. Gazebo 线与 3DGS 线对照

### B1. 重复实现

| 职责 | Gazebo | 3DGS | 已共用 | 仍各一份 |
| --- | --- | --- | --- | --- |
| 采集 | collect.py + collect_parallel.py | collect_3dgs.py | split_manifest（gid / split）、cf_sitl_exec 的指纹算法、student.py | 步循环、manifest 拼装、停止条件 |
| 评测 | eval_closed_loop.py | eval_3dgs.py；库外 `xeval_3dgs_driver.py`（115 行，5 份 sha 相同）；`compare_datasets.teacher_3dgs_eval_json` | `summarize_episode`、`aggregate`、SPL 函数 | 到点即停、成功判据包装、eval.json 拼装 |
| 教师 | teacher.py：柱子解析几何上的 A* | planner.py：占据栅格 A* | 无代码共用，只有「同值」注释（前瞻 0.7、前向净空 0.4、停止半径 0.5） | 航向容差、步长常量 |
| 录像 | record_views.py | video.py | view_layout.py、e1_body.py | 取帧与顶视图来源不同，属合理差异 |
| 审计 | audit_dataset.py | 无 | 无 | 3DGS 集不过任何审计 |
| 打包 | preprocess.py | 同一个 | 全部 | 无 |

### B2. 语义漂移逐项【已核实】

| 项 | Gazebo | 3DGS |
| --- | --- | --- |
| 成功判据 | `sim09.v2`：base / forest 首次 `x ≥ goal_x` 即结束本集（eval_closed_loop.py:708, :734） | `3dgs.pilot.v2`：欧氏半径，学生到点不停，整集任一碰撞即失败（eval_3dgs.py:246–250；eval_closed_loop.py:475） |
| 到达半径 | `REACH_RADIUS = 1.5`（:47）；campus 0.5（campus.py:14） | scene.json `reach_radius_m` 0.5 到 2.0；`reach_ever` 仍用 1.5（:525–526） |
| 碰撞 | 接触传感器话题加 stuck 启发式，逐窗口记（collect.py:1019–1048, :1109–1120） | 体素代理沿每步首尾连线扫掠；首撞后其余各行都记 true（collect_3dgs.py:337–341, :362） |
| 误前向 | cf_sitl 用「碰撞前 1 s 内下过 FORWARD」（:55, :785） | 不传滑行窗口（eval_3dgs.py:74–76），只看碰撞那一步 |
| 动作窗口 | `observation`：截止 = 观测位姿戳 + hold（collect.py:984–993, :1023；cf_sitl_exec.py:488–498）。实测观测间隔 0.900 s，派发比观测晚 0.152 s，动作生效约 0.85 s，之后固件回 SAFE 约 0.05 s | 字面量 `render_observation`（collect_3dgs.py:477）；SITL 时钟只在步进里走（dynamics.py:255–258），观测间隔 1.000 s，零延迟，无空档 |
| 相邻两帧的位姿变化（同指纹，hold 1.0，见 F2） | 连续前进 0.255 m；连续转向 28.7°；单发转向 22.6° | 0.299 m；34.4°；31.6° |
| `displacement_m` / `dyaw_deg` 字段 | 从派发位姿量到窗口末位姿，三维距离（collect.py:1105–1108）；连续前进中位 0.243 m | 量整个窗口，水平距离（collect_3dgs.py:341–344）；0.299 m。同名字段两域定义不同，不能直接比 |
| hold 缺省 | 1.0（collect.py:1282） | 采集取 scene.json `step_s` = 0.5（collect_3dgs.py:561）；评测强制显式给（eval_3dgs.py:123） |
| 步数缺省 | 采 40、评 80（split_manifest.py:39；eval_closed_loop.py:967） | scene.json `max_steps`，按 hold 0.5 的步长预算（planner.py:36），不随 hold 缩放（eval_3dgs.py:195） |
| manifest 键 | 独有 24 个：`vehicle`、`dynamics`、`camera_profile`、`success_criterion`、`goal_radius`、模板 / 世界 / 资源 sha 等 | 独有 19 个：`dynamics_backend`、`collision_proxy`、`renderer`、`route`、`scene`、`completed`、`planner_failures` 等；两域共有 37 个 |
| 同义不同名 | `dynamics` | `dynamics_backend`。`compare_datasets.py:129` 读 `dynamics`，对 3DGS 集得到 null |
| validity | 三个键都带「跑满」条件（collect.py:1854–1858） | 不带（collect_3dgs.py:500–505）；另写 `completed`（:508），全仓非测试代码没有读取方 |
| 教师 | `uavrgb.teacher.v5` / `v6`，航向容差 25°（teacher.py:457–461） | `uavrgb.teacher_occ.v1`，容差 12°（planner.py:32） |
| 随机化 | 每个 seed 生成一套柱阵与外观 | 场景固定，只有 `--spawn-jitter`（缺省 0，collect_3dgs.py:582） |
| 逐步字段 | 有 `depth_ts`、`pose_rgb_gap_ns`、`action_start_ns`；oracle 有 `speed_mps` | 有 `clearance_m`、`frame_quality`、`render_ms`、`pose_gt`；oracle 没有 `speed_mps`，`stable_termination` 恒为 None |
| split | 每个 seed 一个几何组 | 一个场景一个 gid；`exp-teacher-3dgs-20261109` 实测 train 19 / test 930（已知项） |

两域一致的部分【已核实】：图像 320×240、`fx = fy = 277.128`、near 0.05 / far 50、相机安装位 `0.045 0 0.015`、俯仰 0；动力学指纹算法同一份（dynamics.py:29 引用 cf_sitl_exec 的函数），最新数据两边都是 `1bc8fedf:lib31`；`yaw_before` 两边都是度。

### B3. 要不要抽「仿真后端接口 + 统一 runner」

结论：现在不抽。理由：

1. 两个步循环不是同一种东西。`collect.collect_one`（:802–1172，370 行）是异步的：订阅位姿 / 接触 / 相机话题、按仿真时钟等窗口结束、确认零速已发布、处理订阅进程退出与 RTF 过低。`collect_3dgs.run_episode`（:232–541）是同步的：渲染、策略、`dyn.step`、`occ.sweep`。前者多数行是传输层故障处理，后者没有对应物；能共用的只剩记账与 manifest 拼装。
2. B2 里的漂移项（判据、半径、窗口时序、manifest 键、教师容差）换成统一 runner 后仍要由各后端各自提供，接口本身挡不住。
3. `collect.py` 以 31 个参数的 CLI 被两层子进程调用，换掉它要用 Gazebo 重新验证数据等价；运行中的 DAgger 流水线还冻结在旧工作树上。
4. 项目处在实验密集期（11-07 一天 101 个提交），抽象层会让每次单域改动都要动两边。

什么时候再议：出现第三个后端，或同一个缺陷在两个循环里各修过三次以上。届时的最小接口草案（不建议现在实现）：

```python
class SimBackend(Protocol):
    def reset(self, seed: int) -> None: ...                 # 出生位姿、随机化、起飞到位
    def observe(self) -> "Obs": ...                         # rgb(HWC u8)、t_ns、pose_before(x, y, z, yaw_deg)
    def apply(self, action: int, hold_s: float) -> "Step":  # 推进一个窗口；返回 pose_gt、t_end_ns、collision、src
        ...
    def close(self) -> None: ...
```

`pose_gt()` 与 `collision()` 并进 `apply` 的返回值：Gazebo 的接触只能在窗口内读。迁移顺序应是：契约测试，3DGS 适配（现状已接近这个形状），Gazebo 适配，最后才是共用 runner；每步用「同 seed 的 actions.jsonl 逐行相同」验收。

更便宜的防漂移办法（建议做，按顺序）：

**B3-a 判据函数单源。**
- 改哪里：新建 `sim/tools/metrics.py`，从 `eval_closed_loop.py` 移入 `wilson_ci`、`coast_false_forward`、`trajectory_xy`、`path_length_m`、`shortcut_path`、`truncate_at_goal`、`goal_path_length_m`、`first_goal_step`、`spl_of`、`summarize_episode`、`aggregate` 等 14 个函数与 5 个常量。
- 改什么：`eval_closed_loop.py` 用 `from metrics import …` 保留原名；`eval_3dgs.py:41` 与 `pipeline_common.py:69` 改 import metrics，删掉第二份 `wilson_ci`。
- 收益：`eval_closed_loop.py` 从 1311 行降到约 950 行；`eval_3dgs` 不再加载 Gazebo 编排链；库外汇总脚本有地方 import。
- 风险：低。这些函数只引用 `math` 与 5 个常量；8 个测试文件没有对这些名字打补丁。
- 验证：`sim/tools/test_eval_*.py` 与 `sim/3dgs/test_collect_eval.py` 原样通过；import `sim.3dgs.eval_3dgs` 后 `sys.modules` 里没有 `collect`。

**B3-b 共享的 episode 必备键校验，放在两域唯一的共同必经点。**
- 改哪里：`tools/preprocess.py` 的 `load_episode`（:116 起），约 30 行。
- 改什么：一张必备键表逐集检查：`schema`、`hold_s`、`window_from`、`dynamics_profile`、`teacher`、`teacher_version`、`geometry_gid`、`geometry_version`、`split`、`camera{width,height,fx,fy,cx,cy}`、`num_steps`、`stopped_on_collision`、`validity`。外加不变量：actions 与 oracle 行数相等且等于 `num_steps`；`action_end_ns − action_start_ns` 与 `hold_s` 相差不超过 5%（Gazebo 窗口末实测晚 0.005 到 0.015 s）；`action_executed` 在 0 到 3。
- 再加一项时序记录：把相邻两行 `rgb_ts` 之差的中位数写进 meta（Gazebo 现为 0.900 s，3DGS 为 1.000 s），同包内各来源相差超过 5% 就拒绝。F2 那类「同名 hold、实际时序不同」靠窗口长度查不出来，靠它能查出来。
- 契约测试对三种夹具都跑：`make_synthetic_episode.py` 的产物；`test_collect_eval.py` 里 mock 渲染器跑出的真实 `collect_3dgs` 集；从磁盘拷一份真实 Gazebo manifest。
- 收益：任何一域改了键名或窗口语义，在 check.sh 里就红。
- 风险：旧数据缺键会被拒，沿用现有的 `--allow-frozen-geometry` 放行。
- 验证：删掉 3DGS 夹具的 `window_from`，测试失败。

**B3-c 把库外驱动并回 `eval_3dgs.py`。**
- 改什么：`--route` 接受多条路线，场景与渲染器只加载一次；`--policy` 增加 `astar`。同时删掉 `compare_datasets.py:261–262` 的教师特例与 `teacher_3dgs_eval_json`（约 35 行）以及 5 份库外副本。
- 收益：驱动的 docstring 写明每次调用重读 1.2 到 1.5 GB 的 PLY（garden 约 200 s），所以它「逐项照抄 eval_3dgs.main」并用了私有函数 `_sha256`；并回后只剩一份。
- 风险：eval.json 要按路线分目录，输出布局会变。
- 验证：用 xeval 的同一组 seed 重跑 1 个场景 1 个臂，逐集 success 与现有结果相同（需要 GPU，本次没跑）。

**B3-d 3DGS 测试进门禁。**
- 改哪里：`tools/check.sh` 末尾两个测试循环（:156, :161）之后加 3 行，遍历 `sim/3dgs/test_*.py`。
- 收益：6 个文件 1848 行、86 个测试现在只靠人手跑；`teacher/` 的 5 个测试与 `rl_teacher/test_core.py` 同样不在门禁里。
- 风险：增加约 65 s；`test_dynamics_sitl.py` 在没有 `cf_sitl_proc` 时自己跳过。
- 验证：本次在干净工作树 `wt-review3`（c85a1ff）用 `.venv-tools` 实跑，86 个全过，其中 3 个按设计跳过（需要真实 gsplat）。

## C. 大文件与配置蔓延

### C1. 五个大文件

| 文件 | 混在一起的职责 | 建议 |
| --- | --- | --- |
| `collect.py`（1884，import 方 1） | 参数解析与校验（`main` :1278–1880，约 600 行）；5 种动力学分支；5 种策略分支；步循环 `collect_one`（370 行、26 个参数）；起飞与安全网；manifest 拼装（:1781–1870） | 不按职责拆文件。先按 D1 删旧动力学与 teacher-rl 分支，预计降到约 1500 行。之后只把 manifest 拼装提成一个函数（约 90 行原地移动），供 B3-b 的契约测试直接调用 |
| `eval_closed_loop.py`（1311，import 方 2） | 纯判据函数（:214–633）；SPL 参考路径；单集编排（`run_episode` :666–828）；协议比对；报告；录像 | 抽 `metrics.py`（B3-a），降到约 950 行。这是五个文件里唯一有跨域收益的拆分 |
| `audit_dataset.py`（1133，import 方 0） | `audit()` 单函数 634 行（:403–1036）；`episode_facts`；campus 专项 | 不拆：只有 2 个子进程调用点，拆了没有复用。要补的是覆盖面（F6） |
| `preprocess.py`（1068，import 方 3） | 逐集门禁、口径一致性、split 推导、meta、UAVR 写出、两种侧车、CLI，沿数据流排列 | 不拆，内聚 |
| `ui/studio.py`（3606，import 方 0） | 顶层辅助约 815 行；`StudioApp` 约 2670 行、90 个方法、4 个页签；7 处子进程启动 | 不拆：归另一会话，没有 import 方。只有一处可删：:83–84 的 `ACTION_NAMES` 与 `ACTIONS` 是同值元组 |

补充【已核实】：`collect.py`、`collect_parallel.py`、`eval_closed_loop.py` 分别声明 31、33、35 个参数，其中 14 个三处都声明并逐层透传（`--dynamics --hold --steps --window-from --vehicle --policy --teacher-planner --costmap-tol --costmap-lookahead-steps --rl-checkpoint` 等）。可选值已经不一致：`--policy` 在 collect.py:1320 有 `teacher-nn`，collect_parallel.py:627 没有。删掉 teacher-rl 与旧动力学后这组重复会自然缩小，不必为它新建共享参数模块。

### C2. 同一参数多处定义【已核实】

| 参数 | 出现位置 | 不一致或风险 |
| --- | --- | --- |
| hold | collect.py:1282、collect_parallel.py:591、eval_closed_loop.py:968（都是 1.0）；e1_pipeline.py:35；compare_datasets.py:69（1.0）与 :84（0.5）；dagger_pipeline 的 CFG（0.5）；7 份 scene.json 的 `step_s`（0.5）；studio.py:77 | 3DGS 采集缺省 0.5，Gazebo 缺省 1.0；同为 1.0 时实际时序与步长也不同（F2） |
| 步数与步长常量 | split_manifest.py:39（40）；eval_closed_loop.py:967（80）；e1_pipeline.py:52, :88；studio.py:76；scene.json `max_steps`；planner.py:36（0.15 m / 17°）；teacher.py:508–511（0.25 m / 27° / 40 步） | 两域的步长常量分别按 hold 0.5 与 1.0 标定，互不引用 |
| 相机 | 两个 SDF 模板；renderer.py:32–33, :72, :186；scene.py:236；scene_import_mipnerf.py:41；scene_import_wanderland.py:46；7 份 scene.json；preprocess.py:75；teacher.py:513；e1_mini.json；相机档字面量 5 处 | 当前值一致；改相机要同步 15 处以上 |
| 到达 / 停止半径 | eval_closed_loop.py:42（1.0）、:47（1.5）；campus.py:14（0.5）；teacher.py:458 等 5 处与 scene_gen.py:503（0.5）；planner.py:34（0.5）；scene.json（0.5 到 2.0） | 一条 3DGS 教师集里同时出现 3 个半径（F3） |
| 成功线余量 | collect.py:1264（eval_closed_loop.py:81 引用同一个）；traverse_rate.py:49 另有缺省 1.0 | 目前同值，第二份不引用常量 |
| 机体半径 0.102 | e1_mini.json；teacher.py:147；preprocess.py:74；collision.py:257, :517, :615, :625；planner.py:330 | 只有 teacher_kino.py:79 从 json 读，其余是字面量 |
| 前进速度 0.3 | collect.py:123；audit_dataset.py:98 | AUDIT Q7 已记，仍在 |
| cf_sitl 插件与进程目录 | collect_parallel.py:64–69；e1_pipeline.py:37；compare_datasets.py:75；dynamics.py:35–37；scene_import_mipnerf.py:55 | 代码缺省 `tmp/build-cf-sitl`，运行中的流水线用 `tmp/build-cf-sitl-lib31`。3 个 lib31 构建目录里插件有 2 种 sha、`cf_sitl_proc` 有 3 种 sha，指纹里的 `lib31` 区分不了【怀疑：可能只是构建不可复现】 |
| GPU UUID | 代码 0 处；只在 CLAUDE.md（2 处）、progress.md（1 处） | 靠操作者的 shell 环境变量钉卡 |
| 默认动力学与模板 | `--dynamics` 默认 `x3_velocity`：collect.py:1289、collect_parallel.py:597、eval_closed_loop.py:974；`DEFAULT_TEMPLATE` 4 处（3 处指向 X3 模板，scene_forest.py:42 指向 E1 模板） | 现行流水线都显式传 cf_sitl；裸跑 CLI 得到的是 X3 运动学数据 |
| 窗口语义的说明 | collect.py:982–983 称 observation 是「旧语义，只用于诊断 / 兼容旧数据」，:962–963 写明它「会让实际执行窗口短于 hold」；e1_pipeline.py:36 称它是「唯一被采纳的基线语义」 | 两处注释互相矛盾；被采纳的正是会缩短执行窗口的那一种（F2） |

## D. 可删除 / 可归档清单（只列，不执行）

### D1. 代码

| 项 | 规模 | 证据 | 结论 |
| --- | --- | --- | --- |
| `rl_teacher/` | 7 个文件 508 行 | AUDIT Q5 已判死路径，至今未执行；`.venv-tools`、`teacher-venv`、`3dgs-venv` 都没装 stable_baselines3 / gymnasium；数据盘 3 层以内没有 PPO 产物 | 删，连同三个 CLI 的 `teacher-rl` 分支与 `--rl-checkpoint` |
| `cf_sil` | cf_sil.py 987 行 + 测试 293 行 + cf_sil_v1.json | 唯一 import 方是 collect.py；collect_parallel.py:54 注释「cf_sil 已退出」，:57 的可选项已不含它；数据只有 `cf_sil-pilot-20261104`（8 MB，8 集） | 删 |
| `cf_stab` | collect.py 内嵌 `CfStab`（:197 起）；audit_dataset.py 12 处 | 只被 teacher-rl 与 `--teacher-planner predictive` 需要（都强制 cf_stab 且 hold 0.2，collect_parallel.py:677–680）；`cfstab-*` 3 个目录共 85 MB，零引用 | 与上两项和 `teacher_predictive.py`（59 行）一起删，清单见 AUDIT Q6 |
| `cf_blackbox` | 295 行 + 测试 232 行 | Gazebo 侧选它必在参数校验处报错（collect.py:1394–1404）；3DGS 侧仍是可选后端，`test_collect_eval.py` 的 9 处 `run_episode` 调用靠它做轻量动力学 | Gazebo 三个 CLI 去掉该选项；模块留到 3DGS 测试有替身为止 |
| `x3_velocity` | 非测试 26 行，8 个文件 | 三个 CLI 的默认值；base / l1 / campus 任务在用；`old_x3_ckpt`（e1_pipeline.py:89、compare_datasets.py:66）依赖 `t6-full-1640` | 保留 |
| `teacher/` 与 `sim/tools/teacher_policy.py` | 12 个文件 1419 行 + 172 行 | 入口只有 `--policy teacher-nn`（collect_parallel 不提供）与 `export_soft_targets.py`；最后一次代码提交在 11-05；四条现行流水线都不用；依赖 6.7 GB 的 `teacher-venv` | 归档候选，需用户确认（`docs/TEACHER_LINE_PLAN.md` 仍在） |
| `teacher/student_gru_proto.py` 与测试、`experiments/pipeline_profile/` | 142 行；一个 CMake 项目 | AUDIT Q9 已列，仍在 | 删 |
| X3 模板归档 `sim/worlds/templates/uav_rgb.{3267bb7f,70ef2c0a,7d7f9472}.sdf` | 1413 行 | 分别等于提交 f852582、ac24c9d、dd5c9cb 的 `uav_rgb.sdf`，git 历史可复原；没有测试要求它们存在 | 删这三份（AUDIT Q9 当时列的就是它们）。之后加进来的 E1 归档不能删：test_e1_template.py:108–113 要求 live 模板有同 sha 归档，0417 集用 118fa38c、DAgger 集用 21c8bc9a |
| `ui/monitor.py` 的 `MonitorApp`（:529–1187）、`trainer_demo.py`、`test_trainer_demo.py`、`ui/demo/`、`launch_ui.sh` | 约 1100 行 | studio 只用 monitor 的两个函数；`trainer_demo.py` 只被 monitor.py:386, :746 与测试引用；`$D/monitor_runs` 为空；桌面入口只装 studio | 归档候选，需 ui 负责会话确认（`docs/监控系统交接.md` 仍在） |
| 过期脚本 | `tools/train_students.sh`（81）；`sim/tools/smoke_campus.py`（54）；`sim/tools/traverse_rate.py`（106）；`make_counterfactual.py` / `make_decisive.py` / `probe_text_effect.py`（467）；`tools/airsim_to_uavr.py`（248） | 前三个没有代码调用方（traverse_rate 只有自己的测试与 collect.py 一处注释），分别在 11-03、11-04、10-30 后未改；L1 三件只互相引用（另有 audit_dataset.py:243 一处注释），11-04 后未改；airsim 在 10-30 后未改，check.sh 仍跑它的自检 | 前三个可删；后两组取决于 L1 文本线与 AirSim 预训练是否继续 |
| 占位目录 `data/`、`models/`、`logs/`、`third_party/` | 9 个 `.gitkeep` | 磁盘上没有其他文件；代码里没有对这些仓库相对路径的引用 | 删。`config/` 与 `experiments/cuda_smoke/` 有引用，保留 |

### D2. 分支与 worktree【已核实，`wt_dirty.txt` / `wt_unique.txt` / `wt_lines.txt`】

19 个非 main 分支全部 `ahead=0`（`git branch --no-merged main` 为空）。逐个 worktree：

| worktree | 分支（领先 / 落后 main） | 未提交文件 | 未提交内容相对 main | 结论 |
| --- | --- | --- | --- | --- |
| wt-collect-stall、wt-e1-body、wt-review3、wt-viz3 | 同名分支（0 / 0） | 0 | 无 | 可删 |
| wt-e1-sgvis、wt-xeval-3dgs、wt-xeval-forest | detached c8b9ccf、77927d4、6c4a0b5（0 / 40、27、43） | 0 | 无 | 可删 |
| wt-eval4 | detached 97ae04b（0 / 17） | 0 | 无 | 在用：DAgger 流水线（PID 759495）的 cwd，落 DONE 前不可删 |
| wt-dagger-gz、wt-student-select | dagger-gz、student-select（0 / 29） | 7、11 | 全部等于 main 或 main 的历史版本 | 可删 |
| wt-fix-3dgs | fix-3dgs（0 / 43） | 4 | 同上 | 可删 |
| wt-nice-fixes、wt-refactor-tools | 同名分支（0 / 12） | 8、10 | 同上 | 可删 |
| wt-scene-3dgs | scene-3dgs（0 / 26） | 4 | 同上 | 可删 |
| wt-fix-a、wt-fix-b、wt-fix-sim、wt-teacher-cost | 同名分支（0 / 29） | 22、35、36、13 | 各有 1 到 6 个文件与 main 的任何版本都不同，但其新增行逐行都在 main 里 | 可删 |
| wt-viz-layout | viz-layout（0 / 52） | 16 | 同上（5 个文件）；另有 2 行改动落在 main 已删除的 `tools/demo_3dgs_callinstrasse.py` 上 | 看过这 2 行后可删 |
| wt-fix-sitl | fix-sitl（0 / 43） | 8 | `sweep_seeds.sh` 有 1 行不在 main（`BUILD` 缺省路径） | 看过这 1 行后可删 |
| wt-r2-uidocs | r2-uidocs（0 / 45） | 11 | 6 个文档与脚本共 33 行不在 main；其中 env.sh 的 2 行写「RTX 4080 SUPER（sm_89）」，与现机 2080 Ti 不符 | 过期内容，确认后可删 |
| wt-teacher-3dgs | teacher-3dgs（0 / 12） | 3 | planner.py 188 行、collect_3dgs.py 12 行不在 main；未跟踪 `test_teacher_occ.py` | 在途，不可删（F4） |
| wt-sim-dyn | sim-dyn（0 / 0） | 1 | 未跟踪 `sim/tools/cf_sitl_sweep.py`（469 行），main 没有 | 在途，先提交或另存 |
| cursor-1110/wt | detached c85a1ff | 4（已暂存） | 另一代理的在途修复 | 不在本清单内 |

合计：20 个可删（其中 3 个需先看一眼），1 个在用，2 个在途，1 个属于别的会话。

### D3. 数据盘 `$D`（共 91.1 GB；盘剩 19 GB，根分区剩 2.6 GB）

在用，不可清理【已核实，`ps`、`/proc/<pid>/cwd`、`/proc/<pid>/environ` 与该次运行的 config.json】：

- 运行中的 DAgger（PID 759495，写报告时在 eval20 阶段）依赖：`exp-dagger-gz-20261109`、`tmp/wt-eval4`、`bin-snapshots/main-97ae04b`、`tmp/build-cf-sitl-lib31`、`tmp/agent10-dagger-gz`、`tmp/agent6-distill`（round0 输入）。
- 被代码或测试引用：`tmp/agent4-teacher-p1`（occ 缓存）、`tmp/agent4-scene-wanderland`、`e1-pipeline-20261107-0417` 及其 eval3 / eval4 / sgvis、`t6-full-1640`、`t5-full-1155`、`campus-paired-20261103`、`3dgs-scenes`、`3dgs-venv`、`downloads`、`datasets`、`torch-cache`。
- 父会话的 `tmp/cursor-1110`。

| 目录 | 大小 | 仓库内引用（代码 / 文档） | 建议 |
| --- | --- | --- | --- |
| `3dgs-pip-cache` | 6.2 GB | 0 / 1 | 纯缓存，可删 |
| `tmp/research-acad`、`tmp/agent5-review-p2`、`tmp/gz-validate-20261107`、`tmp/3dgs-diag` | 2.0 + 1.4 + 1.2 + 0.2 GB | 代码 0 | 草稿目录；里面的报告先入库（E6）再删 |
| `tmp/build-cf-sitl-v2-20261107`、`-v3-20261107`、`-v3.1-20261109` 与十余个 `tmp/build-*` | 每个 4 到 81 MB | 0 | 过期构建，可删 |
| `pubdata`、`airsim10k-uavr` | 3.6 + 0.55 GB | 0 / 2 到 4 | AirSim 预训练线的输入与产物，随该线去留 |
| `demo-baseline-20261106`、`t5b-dispatch-20261031`、`campus-samefamily-diag-20261104` | 1.6 + 1.5 + 0.4 GB | 0 / 0 到 2 | 旧实验产物，归档候选 |
| `l1-v3-cf`、`l1-v4-cf`、`l1-v3-dec`、`l1-v3-relation-500`、`l1-v4-relation-500` | 共 3.1 GB | 0 / 2 到 4 | L1 文本线数据，随该线去留 |
| `teacher-venv`、`teacher-run-*`（6 个）、`ds-teacher400`、`ds-teacher1000`、`teacher-400`、`teacher-600` | 6.7 + 0.35 + 1.35 + 0.3 GB | teacher-venv 5 处，都在 teacher-nn 路径的说明与报错文案里 | 随 `teacher/` 的归档决定一并处理 |
| `xeval-3dgs-20261109` 等 4 个目录 | 4.7 + 1.5 + 1.4 + 0.5 GB | 0 / 1 | 实验证据：留 jsonl 与 manifest，删 `rgb_*.png` 可回收空间（PNG 占比没有统计） |
| `3dgs-smoke-20261107`、`demo-e1-20261107`、`tmp/agent6-viz-layout` | 2.6 + 0.8 + 3.7 GB | make_demo_reel.py:34, :56–60 | demo 短片的输入，定稿后可删 |
| `monitor_runs`、`studio_runs`、`xdg` | 空 | 无 | 可删 |
| `exp-viz3-20261110`、`demo-e1-20261110` | 0.5 + 0.3 GB | 0 / 0 到 1 | 今天的新产物，归属不明，不动 |

## E. git 与工程流程：6 条半天内可落地的改进

**E1. 完整历史没有机外副本。**
- 证据：`git remote` 为空；`tools/git_mirror_alvln.sh` 用 `rev-list --first-parent`（:31）加 `git archive | tar -x`（:37）逐快照重放。242 个提交里只有 213 个 first-parent 快照进镜像；29 个分支内提交、21 个合并的父关系、非 main 分支都不带。
- 改什么：镜像脚本末尾加一行 `git -C "$SRC" push "$(git -C "$DST" remote get-url origin)" main:refs/heads/uav_rgb-src`。不想动 GitHub 布局就改成 `git bundle create $D/backup/uav_rgb-$(date +%F).bundle --all`。
- 收益：`.git` 只有 22 MB，真实历史可恢复。风险：GitHub 仓库多一个与 main 无共同祖先的分支。
- 验证：`git ls-remote <url> uav_rgb-src` 等于 `git rev-parse main`。

**E2. 并发会话直接改主工作树。**
- 证据：审查期间主工作树的未提交项从 7 个增到 11 个。`sim/worlds/uav_rgb_e1.sdf` 在 HEAD 是 21c8bc9a，工作树里先后见到 c5466adf 与 2e554bcd 两个未提交版本；归档副本 `uav_rgb_e1.2e554bcd.sdf` 未跟踪，c5466adf 那份已不在。任务书注明同一时段另有进程在这棵树上跑 `tools/check.sh`。
- 现成的正确做法：长任务用冻结工作树加二进制快照（`wt-eval4` + `bin-snapshots/main-97ae04b`）。
- 改什么：规则写进 `docs/ENGINEERING_RULES.md`（主工作树只做合并，改动在各自 worktree 完成）；`check.sh` 开头与结尾各记一次 `git rev-parse HEAD` 与 `git status --porcelain | sha1sum`，不同就非零退出，约 6 行。
- 收益：一次检查结果对应一个确定的代码状态。风险：别人改树会让自己的检查变红，这正是要暴露的情况。
- 验证：检查运行中 touch 一个跟踪文件，结果为失败。

**E3. 提交标题过长。**
- 证据：标题长度中位 55、p90 196、最长 685 个字符；242 个里 90 个超过 72，23 个超过 200。
- 改什么：`tools/githooks/commit-msg`（约 5 行：标题超过 72 个字符或第二行非空就拒绝）加 `git config core.hooksPath tools/githooks`（现在未设）；这项配置各 worktree 共用。
- 风险：镜像提交发生在 ALVLN 仓库，不受影响。验证：100 个字符的标题被拒。

**E4. progress.md 单行超长。**
- 证据：HEAD 版本 195 行、33,350 个字符，最长一行 822 个字符，69 行超过 200；CLAUDE.md 最长 845，docs/TASKS.md 749。
- 改什么：同一目录加 `pre-commit`，只查本次新增行：`git diff --cached -U0 -- progress.md | grep '^+' | awk 'length > 200 {exit 1}'`；超过 7 天的已完成事项移进当月 worklog。
- 风险：只约束新增，存量不动。验证：暂存一行 300 个字符，被拒。

**E5. worklog 平铺 161 个文件。**
- 证据：HEAD 上 16,826 行；10 月 84 个、11 月 77 个；docs 占全部跟踪文件的 38%（241 / 636）；worklog 之外指向它的链接 59 处、目标 41 个，代码里 2 处。
- 改什么：`git mv docs/worklog/2026-10-* docs/worklog/2026-10/`，用一条 sed 修这 59 处链接，加一份每文件一行的 `README.md` 索引。
- 风险：`$D/tmp` 里未入库报告的链接会断。验证：`git grep -oh 'worklog/2026-[^) ]*\.md' | sort -u` 列出的路径全部存在。

**E6. 任务书、驱动脚本散在 `$D/tmp`，代码反过来依赖草稿目录。**
- 证据：`$D/tmp` 两层以内（不含 wt-* 与 cursor-1110）有 85 个 .md、194 个 .py / .sh；实验目录里还有 29 个未入库脚本；代码注释引用 BRIEF / FIXLIST 42 行（32 个文件）、`DISTILL §` 29 处，被引用的任务书与 `DISTILL_20261107.md` 0 个入库。
- 流水线输入指向草稿目录 3 处：`dagger_gz_pipeline.py:55`；`dagger_pipeline.py:54`（由未入库的 `build_occ_cache.py` 生成，:297 的报错文案点名该脚本）；`relabel_costmap.py:31`（绝对路径）。
- 改什么：被代码引用的任务书与报告拷进 `docs/briefs/` 入库；`build_occ_cache.py` 入 `tools/`；运行中的 DAgger 结束后，把两个输入目录挪到 `$D/datasets/` 下的稳定名字并改这 3 行 CFG；`check.sh` 加一行，非注释代码行出现 `tmp/agent` 就失败。
- 风险：`round0_collection` 正被 PID 759495 使用，必须等 DONE。
- 验证：该 grep 结果为 0；`dagger_gz_pipeline.py --stop-after-preflight` 通过。

## F. 3DGS 线的 bug 与口径问题

**F1. 评测到点不停，SR 随步数预算变。【已核实】**
- 证据：`eval_3dgs.py:246–250` 以 `stop_on_collision=True` 调 `run_episode`，学生策略没有 `stop_radius`（collect_3dgs.py:287），到点后继续飞到撞或步数用完；`summarize_episode` 对整集取 `any(collision)`（eval_closed_loop.py:475）。
- 现有 600 集重算（pilot.v2 对截到首次进半径）：教师 100 对 100；BC 18 对 31；截断 13 对 29；sgvis 18 对 38；盲前进 10 对 37。后一列与实验文档的 SR(stop) 列逐项相同。
- 步数敏感性：只给 60% 步数时，pilot.v2 下学生的成功数反而上升（BC 26、截断 22、sgvis 29），教师降到 46。
- 文档 `2026-11-09-xeval-3dgs.md` 已写「应统一成到点即停」，但 `dagger_pipeline.py:224–229` 与 `compare_datasets.py:297` 读的仍是 pilot.v2；文档的 SR(stop) 由库外的 `summarize.py` 算出。
- 最小修法：在 `summarize_3dgs_episode`（eval_3dgs.py:61–79）里用 `first_goal_step` 把 actions / oracle 截到首次进入半径那一步，再算成功、碰撞、SPL、步数；`task_success` 仍取完整轨迹的结果；`METRIC_VERSION` 升 `3dgs.pilot.v3`。约 5 行，旧轨迹可离线重算。
- 风险：v2 与 v3 的数字不能混报，stats_report 已有混合提示。
- 验证：对 `xeval-3dgs-20261109` 重算得到 100 / 31 / 29 / 38 / 37；加一个「先进半径后碰撞判成功」的单测。

**F2. 同名 hold 1.0，两域实际时序不同，相邻两帧的位姿变化差 17% 到 40%。【时序与步长已核实；对迁移结论的影响没有测】**
- 实测（同一指纹 `cf_sitl_v1:1bc8fedf:lib31`；Gazebo 为 DAgger 第 1 轮 100 集，3DGS 为 `exp-teacher-3dgs-20261109` 的 hold 1.0 共 465 集；都是中位数）：

| 量 | Gazebo | 3DGS |
| --- | --- | --- |
| 相邻观测间隔 | 0.900 s | 1.000 s |
| 观测位姿戳到动作派发 | 0.152 s | 0 |
| 动作生效时长 | 约 0.85 s，随后固件 SAFE 约 0.05 s | 1.000 s，无空档 |
| 连续前进，相邻两帧位移 | 0.255 m（n=1784） | 0.299 m（n=10965） |
| 连续转向，相邻两帧转角 | 28.7°（n=16） | 34.4°（n=48） |
| 单发转向（前后都是前进） | 22.6°（n=308） | 31.6°（n=951） |

- Gazebo 的机制：截止时刻 = 观测位姿戳 + hold（cf_sitl_exec.py:488–498，collect.py:990–993），到期固件自动回 SAFE（cf_sitl_exec.py:10）。仿真时钟在取帧和推理期间照走，下一帧观测又取在上一窗口结束前 0.105 s，所以每 0.9 s 里依次是：上一动作 0.105 s、SAFE 0.05 s、本动作 0.745 s。collect.py:962–963 的注释已写明这种语义「会让实际执行窗口短于 hold」。
- 3DGS 的机制：SITL 时钟只在 `dyn.step` 里走，截止多给半个仿真步以免出现空档（dynamics.py:255–258），动作占满整个窗口。两边的 docstring 都自称 observation 语义（dynamics.py:138）。
- Gazebo 一侧的时序还随机器负载变：0417 集的派发延迟是 0.130 s，DAgger 第 2 轮的 p90 是 0.174 s。
- 仓库里两组步长常量各自成立但互不引用：teacher.py:508–509 写 0.25 m / 27°；planner.py:30–36 写每 0.5 s 步 0.15 m / 17°。跨域实验文档把 hold 1.0 当作「三个学生的训练口径」直接用在 3DGS 上。
- 最小修法：不改代码。Gazebo 学生上 3DGS 评测时用 `--hold 0.85`（0.85 × 0.299 m/s = 0.254 m，0.85 × 34.4°/s = 29.2°），并把上表写进 EVAL_STANDARD。`eval_3dgs` 同时给 `--data` 且 hold 不一致会报错（:130–131），只能单给 `--hold`。
- 风险：一个旋钮对不齐 0.15 s 的派发延迟。Gazebo 里单发转向在下一帧只显示 22.6°，余下的落在再下一帧；3DGS 里一次显示完，hold 0.85 下的数值没有实测。彻底的办法有两个，都不建议现在做：Gazebo 改 `dispatch` 语义（全部 E1 数据作废）；或让 3DGS 步循环按上面的三段推进（多一种采集模式）。
- 验证：用 `measure_obs_delta.py` 量一轮 hold 0.85 的 3DGS 集，连续前进与连续转向落在 0.255 m、28.7° 的 ±5% 内；再重跑 1 个学生的 120 集，看 SR(stop) 是否变化。

**F3. 同一条记录里有三套半径。【已核实】**
- 证据：`success` 用场景半径（eval_closed_loop.py:513–517）；`reach_ever` 恒用 1.5（:525–526）；教师停止半径是 0.5（planner.py:34，写进 manifest 的 `stop_radius`）。
- 合成用例：场景半径 2.0、停在 1.75 m 处并发了 SAFE，得到 `success=True`、`final_in_radius=True`、`task_success=False`。现有 900 集里 `success` 与 `reach_ever` 不一致的有 23 集。`dagger_pipeline.py:226–227` 汇报的正是 `task_success`。
- 最小修法：只动 3DGS 包装函数，3 行：`reach_ever` 取 `success`，`goal_radius_reach` 写场景半径，`task_success` 按统一半径重算。Gazebo 的 sim09.v2 逐位不变。
- 验证：上面的合成用例变为 `task_success=True`。

**F4. 教师左右来回转（已知 D9）的补充证据。【已核实】**
- 翻转在全程均匀出现，不只在目标附近：hold 1.0 的教师集，离目标 3 m 以内 22.3%、3 m 以外 22.7%（1562 次 / 7025 步，116/120 集出现）；hold 0.5 只有 0.4%。
- 根因与 planner.py:30–31 的注释一致：容差 12° 是按 hold 0.5 的 17° 一步取的 0.7 步，在 hold 1.0 下只有 0.35 到 0.43 步。
- 在途工作树的产物 `exp-teacher-3dgs-20261109`（每组 150 集）已经给出对比：

| 变体 | hold 0.5 成功 / 翻转率 | hold 1.0 成功 / 翻转率 |
| --- | --- | --- |
| `astar_occ`（main 现状） | 148 / 0.4% | 127 / 21.9% |
| `astar_occ_hard` | 149 / 0.4% | 128 / 21.6% |
| `costmap_occ` | 145 / 0.3%（goal-stop 只有 118 集） | 149 / 0.1% |

- 读法：`astar_occ_hard` 不解决问题；`costmap_occ` 解决 hold 1.0，但 hold 0.5 变差。`astar_occ_tol` 的测试（11-10 06:06 写的 `test_teacher_occ.py`）要求「容差随 hold 推导，hold 0.5 下逐行不变」，而 planner.py（11-09 23:47）里还没有实现。
- 建议：补完 `astar_occ_tol`（约 5 行），在同样的 300 集上量过后只留一个规划器，不要并存 4 个。
- 追溯问题：该目录 949 集全部由这个工作树的未提交代码产出（manifest 里 `collision_proxy.envelope_source` 指向 `tmp/wt-teacher-3dgs`），其中 628 集写着 main 里不存在的版本串 `uavrgb.teacher_occ.v2`、`uavrgb.teacher_cost_occ.v1`，合并前无法从 main 复现。

**F5. A+B 混包在现有代码下恒被拦，且 3DGS manifest 缺两个键。【已核实】**
- 证据：`collect_3dgs.py:477` 的 `window_from` 是字面量；`preprocess.py:53` 把它列为任何模式都不放行的口径键；`compare_datasets.py:137` 据此标 `blocked_by`。该脚本的 docstring 写「B 改成 hold 1.0 + observation 后 AB 才成立」，但没有任何参数能让 3DGS 写出 `observation`。3DGS manifest 不写 `camera_profile` 与 `dynamics`（949 集全缺）。
- 建议：不要把字面量改成 `observation`，那会让时序不同的两批数据不带标记地同包（F2）。二选一：
  - （a）从 `compare_datasets` 的 CFG 里删掉 AB 臂，零成本。
  - （b）要做 AB，就按 F2 的换算采 B，并仿照已有的 `--allow-mixed-dynamics`（preprocess.py:342–343）加一个显式开关，把两种窗口语义记进 meta。
- 两种情况下都在 `collect_3dgs.py` 的 manifest 里补 `"dynamics"` 与 `"camera_profile"` 两个键（2 行）。验证：`dataset_protocol` 对 B 不再出现 null。

**F6. 3DGS 集没有审计门。【已核实；其中一项在现有数据里没有触发】**
- 证据：`audit_dataset.py:787–789` 对 `3dgs.v1` 必判 FAIL，gid 只从 `scene["pillars"]` 算（:162–171），所以 `dagger_pipeline` 与 `compare_datasets` 的 B 臂都不跑 audit。
- `teacher_version` 的一致性只在 audit 里查（:822–846），preprocess 不查；`exp-teacher-3dgs-20261109` 里三个教师版本并排，各 314 到 321 集。
- `planner_failures`（collect_3dgs.py:530 注释写「大于 0 的集不该进训练集」）在 preprocess 与 `dagger_pipeline.py:115–127` 里都没人看；现有 2008 集里该值全为 0。
- 最小修法：不给 audit 加 3DGS 分支。在两域共同的 `preprocess.load_episode` 加 2 行（`planner_failures` 为真则拒收），并把 `teacher_version` 加进 `FAMILY_KEYS`（1 行，跨族混包仍可用 `--allow-mixed-geometry` 放行）。
- 验证：按 check.sh 负例的写法加两条：改 manifest 后打包必须失败。

**F7. 碰撞代理的口径。【数值已核实；与渲染画面的一致性没有验证】**
- 膨胀格数是 `int(round(半径 / 体素))`（collision.py:409），5 个场景的 manifest 里都是 1 格。体素 0.1 的场景（bicycle、garden、stump）膨胀 0.1 m；体素 0.2 的场景（callinstrasse、wanderland）膨胀 0.2 m，是机体半径 0.102 的两倍。场景之间的碰撞余量不同。
- 代理来源也不同：wanderland 用激光点云（`3dgs.occ.v3.points`，每格至少 3 点），其余 4 个场景用不透明度筛过的高斯椭球（`3dgs.occ.v2`）。
- callinstrasse 的 2,062,370 个高斯只有 725,234 个（35%）通过不透明度与尺度筛选参与碰撞，其余画得出来但撞不到。
- 高度方面没有问题：480 集共 55,590 个采样点全部在 `[1.962, 2.066]`，没有离开 `z_slice [1.8, 2.2]`。
- 最小修法：`Occupancy.summary()` 加一个 `dilation_m` 字段（1 行），跨场景比较碰撞率时按体素分组报告。验证：manifest 里两类场景分别出现 0.1 与 0.2。

**F8. 步数预算与几个缺省值。【已核实】**
- `eval_3dgs.py:195` 的缺省步数取 scene.json 的 `max_steps`，那是按 hold 0.5 的步长算的（planner.py:36），hold 1.0 下时间预算翻倍；库外驱动自己换算 `ceil(max_steps × 0.5) + 15`。修法：缺省改成 `ceil(max_steps × step_s / hold)`，1 行；F1 修好后它只影响超时判定。
- `run_episode` 的缺省 `dynamics="cf_blackbox"`（collect_3dgs.py:237）与 CLI 缺省 `cf_sitl`（:578）不同。生产调用方都显式传参，只有 9 处测试调用吃这个缺省；`dynamics.py:4` 的 docstring 仍写「cf_blackbox（默认）」，改注释即可。
- 3DGS 的误前向不带滑行窗口，与 Gazebo cf_sitl 的 `.coast1s` 口径不同，两域的误前向率不可直接比。
- 相机内参、分辨率、near / far、安装位两域一致，不是问题；两者都与 e1_mini.json 记的真机标称视场（裁 4:3 后 64.9°）不同。

## G. 建议的落地顺序

1. E1（推一个分支备份历史）：一行，先做。
2. F1 + F3（3DGS 判据到点即停、半径统一）：约 8 行，旧轨迹离线重算。
3. B3-d（3DGS 测试进 check.sh）：3 行。
4. F2（把两域时序表写进 EVAL_STANDARD，用 hold 0.85 重测 1 个学生）：不改代码，要占一次 GPU。
5. F4（补完 `astar_occ_tol` 并合并 `wt-teacher-3dgs`，只留一个规划器）。
6. B3-a（`metrics.py`）与 B3-c（驱动并回 `eval_3dgs.py`）。
7. D1 的删除项（rl_teacher、cf_sil、cf_stab、predictive、X3 模板归档、占位目录），按 AUDIT Q6 的清单一次做完。
8. D2 / D3 的清理与 E2 到 E6，等运行中的 DAgger 落 DONE 之后做。

## 没读 / 没验证的内容

- **C++ 没有审。** `src/`（14.9k 行）只看了 CMake 目标与格式常量（dataset.cpp:474、checkpoint.cpp:16–17、aux_sidecar.cpp:47）；`sim/cf_sitl/`（56 个文件）只看了版本常量，没有构建。
- **没有逐行读的 Python。** `ui/studio.py` 与 `ui/monitor.py` 只看了结构（函数表、子进程启动点、import）。`teacher.py`、`scene_gen.py`、`scene_forest.py`、`campus.py`、`asset_campus.py`、`gz_io.py`、`cf_sil.py`、`teacher_kino.py`、`teacher_belief.py`、`stats_report.py`、两个 `scene_import_*.py` 只看了接口、常量与 import。`sim/3dgs` 的 8 个模块读了主体；`renderer.py` 的 gsplat 调用细节与 `scene.py` 的 PLY 读写、`nav_to_world` 推导没有逐行核对。
- **没有跑的东西。** Gazebo、训练、GPU 渲染、`tools/check.sh` 都没跑（按规则）。所以：碰撞代理与渲染画面是否一致没有验证；F2 的 `--hold 0.85` 换算没有实跑；B3-c 的等价性没有实跑；时序差对跨域 SR 的影响大小未知。
- **时序与步长对比的局限。** 用的是磁盘上的现有数据：Gazebo 为 DAgger 第 1、2 轮各 100 集与 0417 集，3DGS 为 `exp-teacher-3dgs-20261109` 的 949 集与 xeval 的 600 集；不是同一动作序列的对照实验。派发延迟由 manifest 的 `performance.action_sim_s` 与窗口长度相减得到（逐集平均），没有逐步的派发时刻记录。「SAFE 空档」来自代码语义加时序推算，没有读固件状态日志确认。连续转向的样本少（16 对 48）。
- **worktree 判定的方法。** 「新增行都在 main 里」按整行文本精确匹配，只有空白差异的行会被算作不在 main；没有逐个打开 20 个可删 worktree 人工复核。
- **数据盘。** 大小只量了一级目录与 `tmp` 的二级目录；「零引用」指仓库内跟踪文件里没有字面引用，`$D` 里未入库脚本之间的引用没有追。`xeval` 目录里 PNG 的占比没有统计。
- **git 与镜像。** 没有执行任何联网命令，GitHub 侧状态取自本地克隆与任务书；ALVLN 顶层的 `cpp/`、`frontend/`、`server/`、`legacy/`、`references/` 只数了文件，没有读内容。
- **并发会话的未提交修复没有审。** 这些文件一律按 HEAD 读。E2 里 c5466adf 那个版本是审查早些时候看到的，现在已无法复查。
- **`lib31` 三个构建目录的二进制 sha 不同**（C2）只看到现象，没有比对功能差异。
- **本次跑过的只有** `sim/3dgs/test_*.py`（6 个文件，在 `wt-review3` 里，`PYTHONDONTWRITEBYTECODE=1`，临时目录指向草稿目录；跑完该工作树含忽略文件在内仍无改动）与若干只读统计脚本。`test_dynamics_sitl.py` 启动过 CPU 上的 `cf_sitl_proc`。
- **过程文档。** 仓库只读，没有写 `docs/worklog` 与 `progress.md`；本报告即过程记录。没有删除任何文件。草稿目录里留下的文件：`REPORT.md`、`import_graph.{py,txt,json}`、`measure_3dgs.py`、`measure_steps.py`、`measure_obs_delta.py`、9 个 `measure_*.txt`、`wt_{dirty,unique,lines}.{sh,txt}`、`wt_dirty_input.txt`、`du_l1.txt`、`du_tmp.txt`、`tmp/`（6 个测试日志与 1 个生成脚本）。
