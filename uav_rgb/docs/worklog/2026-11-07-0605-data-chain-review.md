# 数据链/教师/评测精读与修复（DATA agent）

- 时间：2026-11-07 05:20–06:10
- 范围：`sim/tools/{teacher,collect,audit_dataset,split_manifest,test_audit_dataset,test_realism_rl_guards}.py`、`tools/{preprocess,e1_pipeline,test_preprocess_split}.py`、新文件 `tools/compare_datasets.py`；只读分析 `$DATA_ROOT/uav_rgb/e1-pipeline-20261107-0417/collection/.staging/`（295 集）与 `tmp/agent-forest/teacher3/sparse/`；工作目录 `$DATA_ROOT/uav_rgb/tmp/agent3-data/`
- 结论：数据链没有改动数字的 bug；有三处「口径」问题要标注（base 任务 SAFE 结构性 <5%、`--drop-collision` 留下余量内 SAFE/左右反转标签、5af664a 改 E1 余量没 bump 教师版本）。修了：教师 (载具, 任务) 余量 + 圆柱圆判定 + forest 档版本 v6；preprocess 放行 forest、`--allow-mixed-geometry` 混包并记逐族计数；split_manifest 对 forest 的几何组身份；加了 C++ 读端兼容门。全部测试绿（29 个 test_*.py、自测、self-check、preprocess 13 用例）；base 标签逐字节不变（.staging 11800 步复算 0 不一致）。未 commit。

## 做了什么

### 1. 教师 `sim/tools/teacher.py`
- `CLEARANCE_BY_VEHICLE` 改按 `(vehicle, task)` 取：`("e1_mini","forest_e1") → (0.20, 0.15)`，依据写在注释（离线 20 seed：0.40 时 sparse 20/20 绕林；0.20 时 sparse 20/20 穿、medium 10/20、dense 3/20；`tmp/agent-forest/offline_astar.json`）。未单列的任务（l1/safe/campus）走该载具 base 档。
- `set_vehicle(name, task="base")`；`TEACHER_TASK` 记当前档位；`teacher_version(task)`：forest_e1 → `uavrgb.teacher.v6`，base 仍 `uavrgb.teacher.v5`；`task_of_scene(scene_meta)` 由 `family`/`geometry_version` 前缀推断档位（collect 与 audit 共用）。
- `_pillar_blocked()`：带 `radius` 的柱按圆（圆心距 < r + clearance），无 radius 的方柱保留原算式（逐字节不变）；`ray_clear`/`_point_blocked` 都走它；`_body_hits_box` 对圆柱做「柱心转机身系 → 夹到方框 → 距离 ≤ r」。
- 自测加 3 组（档位/版本、圆判定 vs 方框、forest 档穿 0.6 m 缝而 base 档绕行），共 36 项。

### 2. 采集/审计跟随
- `collect.py:1391`：`teacher_task = task_of_scene(scene_meta_obj)`，`set_vehicle(vehicle, teacher_task)`，manifest `teacher_version = teacher_version(teacher_task)`。
- `audit_dataset.py:906`：label_recompute 的版本门按档位比（forest 集要 v6），`set_vehicle(vehicle, task)`；SKIP 文案「旧几何或教师版本」。
- `test_audit_dataset.py` case 21：forest 集按 (e1_mini, forest_e1) 产标签 + v6 → 复算 12/12；旧森林集（写 v5）→ SKIP 而不是假 FAIL。

### 3. 打包 `tools/preprocess.py`
- 白名单 `KNOWN_GEOMETRY_VERSIONS` 加 `scene_forest.GEOMETRY_VERSION_FOREST`。
- `--allow-mixed-geometry`：只放开族身份键 `FAMILY_KEYS`（geometry/appearance/split_policy/split_salt/task_version/stop_radius/success_criterion/goal_radius）；`STRICT_KEYS`（window_from/hold_s/dynamics_profile）任何模式都严格；`camera_profile` 混包时允许部分缺（3DGS pilot 没写）、不允许两种值。
- 混包时 `split_facts` 逐族推导，要求每族都是 `geometry_gid`（C++ 读端只认一个 scheme）；salt 不同则顶层 `split_salt=null`。meta 新增 `mixed_geometry`、`geometry_versions`（逐族 n_episodes/n_samples/family_split_scheme/family_split_salt/appearance/task_version/hold_s/camera_profile）；混包 `geometry_version="mixed:<a>+<b>"`（不以 `uavrgb.geom.` 开头，`train_student_with_video` 不会误认柱阵）。
- 写盘前的 C++ 兼容门：`"episodes":` 在 meta 里只能出现一次，`split_policy/split_scheme/split_salt/collision_flags` 第一次出现必须在 episodes 数组之前——`dataset.cpp` 是按子串找键的，第一版逐族表用了 `episodes` 键，读端实测报「episodes[] 为空」。
- `test_preprocess_split.py` 加 3 用例（forest 放行；混包需显式 flag + 逐族记录 + 读端键序；混包仍拒 hold_s 不一致、camera_profile 两值）。

### 4. 划分 `sim/tools/split_manifest.py`
- `group_id_for_task(seed, task, density=None)`：forest_e1 用该 density 的树干 gid（旧版静默回落到同 seed 的柱阵 gid，森林集跨 split 查不出来）；未知任务 `ValueError`（与 `split_for_seed` 同口径）。`--verify-task forest_e1` + `--verify-density`，缺省读 collection_manifest 的 task/density。`test_realism_rl_guards.py` 加 2 用例。

### 5. 流水线 `tools/e1_pipeline.py`
- 多集合时 preprocess 自动加 `--allow-mixed-geometry`；森林备选行与 `audit_ignore` 的注释改成实测结论；`steps` 注释标明 X3 0.27 / E1 实测 0.25。CFG 值没改。

### 6. 新文件 `tools/compare_datasets.py`（骨架）
- 配置区 `sources/datasets/gazebo/gs`，`GROUPS` 六组，四步 pack → train → eval → summary，status.json/DONE/`--from`/`--dry-run` 与 e1_pipeline 同约定。只保证 `--dry-run`。

## 没做什么

- 没改 SAFE 近柱语义（等 SITL agent 对执行器位置保持的结论）；两条路的改动量在报告「待确认」。
- 没 bump base 档教师版本：base 标签逐字节不变，bump 会让 audit 对流水线正在产出的 v5 集全部跳过复算。5af664a 的历史口径问题只记录。
- 没在 e1_pipeline 的评测里加 forest 域、没把 `drop_collision` 换成 `--exclude-collision-episodes`：是实验口径决定，报告里给数字请主会话拍板。
- 没起 Gazebo；森林 (0.20/0.15 + 圆判定) 教师在真 Gazebo 的 SR 没测。
- 没修 `ui/test_monitor.py`/`ui/test_studio.py` 两个 Tk 几何断言（不在可写域；ui/ 与主仓库逐字节相同，判断是共用 display :20 上的窗口几何差异）。
- `teacher_kino._boxes` 仍按外接方框读圆柱（不在默认采集路径，只记录）。

## 待确认

见报告 `$T/phase3-20261107/reports/DATA_20261107.md` §待确认。

## 验证到哪一步

| 命令 | 结果 |
| --- | --- |
| `sim/tools/teacher.py` | 36 ok，`teacher: 全部通过` rc=0 |
| `sim/tools/test_audit_dataset.py` | PASS rc=0 |
| `sim/tools/test_collect.py` | 80 项通过 rc=0 |
| `tools/test_preprocess_split.py` | Ran 13 tests OK rc=0 |
| `sim/tools/test_realism_rl_guards.py` | OK rc=0 |
| 29 个 `sim/tools/test_*.py` 逐个（`agent3-data/checks_table.txt`） | 全部 rc=0 |
| `scene_gen.py --self-check` / `scene_forest.py --self-check` | PASS / PASS |
| `tools/check.sh --python-only` | 到 `ui/test_monitor.py`「GUI 缩放后曲线有内容」rc=1 停；之前全过；之后 `ui/test_trainer_demo.py` 92 项过、`ui/test_studio.py` 1 失败（Tk 高度 49 ≤ 60）、`tools/test_train_student_with_video.py` PASS |
| 改后教师对 .staging 9800 步复算 | 0 不一致（`analyze_after_teacher_edit.log`） |
| 改后 audit 对 .staging 295 集 | label_recompute 0/11800；FAIL 只有 class_share、safe_collision（`audit_staging_base.json`） |
| 改后 audit 对 forest sparse 6 集 | label_recompute SKIP「旧几何或教师版本 240 步」，geometry_version PASS |
| preprocess 真数据：base 3 集 / forest 2 集 / base+forest `--allow-mixed-geometry` | 三个 .uavr 都被主仓库 `libuavrgb_core.a` 读通（`load_check`：n=120/80/200） |
| preprocess base + 3DGS pilot `--allow-mixed-geometry` | 被 `window_from` 拦（observation vs render_observation），符合预期 |
| `split_manifest.py --verify` forest sparse / base pilot 清单 | 跨 split 几何组 0 / 0 |
| `tools/compare_datasets.py --dry-run`、`tools/e1_pipeline.py --dry-run` | rc=0，命令全部打印 |

## 风险与遗留

- 混包数据集 `split_salt=null`：C++ 读端不校验 salt，只做日志留痕；审计工具若按顶层 salt 复算 split 要改读 `geometry_versions`。
- forest 档 medium/dense 在 0.20 膨胀下离线仍有一半以上绕行；要穿林得再缩余量或给 A* 加 `FOREST_Y` 墙（见报告）。
