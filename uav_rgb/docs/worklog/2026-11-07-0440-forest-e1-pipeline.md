# forest_e1 场景族接进采集/评测/审计管线 + E1 教师三档密度真 Gazebo 测评

- 时间：2026-11-07 04:40 起草，05:05 补齐 Gazebo/录像结果
- 范围：`sim/tools/{scene_gen,split_manifest,collect_parallel,eval_closed_loop,audit_dataset,test_scene_forest,test_audit_dataset}.py`、`tools/e1_pipeline.py`；实验目录 `$DATA_ROOT/uav_rgb/tmp/agent-forest/`
- 结论：管线接入完成（旧路径字节不变，测试全绿）；三档密度教师测评完成——**现版教师在三档都不穿林而是从森林带外绕行**，sparse 可采但不是「穿林」数据，medium/dense 目前不可作数据集，根因是教师膨胀余量（0.40 m）而非执行器；录像完成。未 commit。

## 做了什么

### 1. 场景生成 `sim/tools/scene_gen.py`
- `build_scene(..., task, appearance, density=None)`：`task == "forest_e1"` 时转 `scene_forest.build_scene(template, seed, scene_id, density, appearance)`；`density` 给了但 task 不是 forest_e1 → `ValueError`；forest_e1 不给 density / 非 default 出生 → `ValueError`（走廊保证是按固定出生点算的）。
- CLI：`--task` 多一个 `forest_e1`，新增 `--density sparse|medium|dense`；`--density` 与 `--task forest_e1` 必须同时给。base/l1/campus 路径一行没动（test_spawn legacy 字节断言照样过）。

### 2. 划分 `sim/tools/split_manifest.py`
- `split_salt_for`：`uavrgb.forest_e1.*` 与 `uavrgb.geom.*` 一样「版本号即 salt」。森林 episode manifest 没有 `split_salt` 字段，eval/audit 复算 split 全靠这一行；比往 `PROVENANCE_KEYS` 加 `split_salt`（会让 base manifest 多出 null 键）改动小。
- `split_for_seed(seed, task="base", density=None)`：forest 用 `scene_forest.sample_trunks(seed, density)` 的 gid + 森林 salt。

### 3. 采集 `sim/tools/collect_parallel.py`
- `--task forest_e1`、`--density`，`scene_gen_extra` 透传 `--density`；forest 必须 `--vehicle e1_mini`（走廊净空按 E1 半径算，X3 进去必撞）。
- 新 `task_geometry_version(task)`：集合 manifest 的 `geometry_version` 按族写（之前 base 以外都硬写 `sm.SPLIT_SALT`），manifest 多一个 `density` 字段。
- `collect.py` 不用改：pillars 里每根柱同时有 `radius` 与外接方框 `size`，教师/碰撞/split 都按已有字段读。

### 4. 评测 `sim/tools/eval_closed_loop.py`
- 同样加 `--task forest_e1 --density`、两条校验；报告 `geometry_version` 改用 `cp.task_geometry_version`，多 `density`。

### 5. 审计 `sim/tools/audit_dataset.py`
- `geometry_version`：白名单加 `uavrgb.forest_e1.v1`，仍拒绝 base/forest 混包；森林集按 `scene_metadata.density` 核对 `min_gap_required_m / trunk_count_range / forest_x / corridor_y / trunk_radius_m / trunk_height_m / spawn_clear_before_x_m` 七个常量（缺 density 记为不一致）。
- `label_recompute`：可复算版本集合 `RECOMPUTE_GEOMETRY_VERSIONS = (base, campus, forest)`，森林集 240/240 步复算一致。

### 6. 流水线 `tools/e1_pipeline.py`
- 配置区新增 `collect_sets=(("base", 300, 3100), …)`，森林三档写成注释掉的备选行（注释里写了启用前要先解决的两件事）。collect/audit 按集合逐个跑；base 集目录仍叫 `collection/`（正在跑的 PID 414432 与 `--from` 续跑不受影响），其它集 `collection-<task>-<density>/`；preprocess 把所有集合的 `{train,val,test}/*/` 合进一次调用。`--dry-run` 七步走通（见验证表）。

### 7. 测试
- `test_scene_forest.py`：scene_gen 入口 == scene_forest 直出（3 档）、`split_for_seed(density)`、`split_for_scene` 对森林 manifest、三种 ValueError、`cp.task_geometry_version / scene_gen_extra`。
- `test_audit_dataset.py` case 21：E1 森林集 geometry_version PASS + label_recompute 12/12、掺一集 base → FAIL「混用」、篡改 `min_gap_required_m` → FAIL。

### 8. 实验（都在 `$DATA_ROOT/uav_rgb/tmp/agent-forest/`，守护进程，`--instances 2`）
- 离线 A*（`offline_astar.py`，20 seed × 3 档 × 6 个膨胀值，不起 Gazebo）。
- 真 Gazebo + E1 SITL 教师：`run_teacher3.sh`（PGID 436329）sparse/medium/dense 各 6 集 × 40 步 + `audit_dataset.py`，输出 `teacher3/<density>/`，统计脚本 `stats_teacher3.py` → `teacher3/stats.json`。
- 余量 vs 执行器对照：`run_phase2.sh`（PGID 450270）用 `PYTHONPATH=clearance_override`（`sitecustomize.py`，**不入库**）临时把 E1 教师膨胀改成 sparse (0.20, 0.15)、medium (0.15, 0.10)，各 6 集 × 40 步。标签不是仓库教师算的，只用来回答问题，不能进数据集。
- 录像：`eval_closed_loop.py --policy teacher --record-video --task forest_e1 --density sparse --seeds 4103 --split train`（带 chase/top 相机）。

## 没做什么

- 没让教师真的穿林（没改 `teacher.py`）。原因：teacher.py 不在本次文件域；而且这是方案决策（见待确认）。
- 没改 `tools/preprocess.py`。原因：不在文件域。但它是森林集合进训练的硬阻塞：`load_episode` 的 geometry_version 白名单没有 `uavrgb.forest_e1.v1`，`_require_uniform` 又拒绝一个包里混两个 geometry_version。所以 `COLLECT_SETS` 的森林行只能先注释掉；启用后 pipeline 会在 preprocess 步停住。
- 没给 `scene_forest.py` 加横墙 / 没出 forest v2。原因：加墙改 SDF 就改几何版本，走廊保证、split、统计都要重做，应由用户决定是加墙还是改教师。
- 没把森林集并进 E1 流水线默认配置（仍 base 300 集）。原因：任务要求默认不变；且上面两条没解决前采了也进不了训练。
- 没测 `--instances > 2`、没测 cf_stab/X3 进森林（故意禁止）。
- 没跑 `eval --policy student` 在森林上。原因：没有在森林上训过的学生，评了也只是「分布外」数字。
- 余量对照实验只做了 sparse/medium 两档。原因：离线 A* 已表明 dense 穿林所需膨胀 ≤0.12 m，而 E1 物理半径 0.102 + 悬停漂移 2.4 cm/s 下每侧只剩 ≈0.04 m，SITL 跑了也只是撞；时间留给录像和文档。

## 待确认

1. **教师绕林还是穿林？** 森林带 y∈[−1.5,1.5] 两侧没有墙，A* 边界按柱阵外扩，现版 E1 教师（A* 膨胀 0.40 / 局部 0.30）三档全部从带外绕行——数据里学到的是「看见密林就拐出去」。要让它穿林，最小改法二选一：
   - a) 场景侧：forest v2 在 y=±1.5 加两面 3 m 高墙（或把 A* 边界夹到走廊内）；教师不变，但三档膨胀 0.40 下 A* 全部无路 → 教师整集 SAFE 死锁，仍要配 b)。
   - b) 教师侧：`CLEARANCE_BY_VEHICLE` 按 (vehicle, task) 取值，forest_e1 用 A* ≈0.20 / 局部 ≈0.15（离线 sparse 20/20、medium 20/20 可穿；dense 需 ≤0.12，不建议），并让 `_point_blocked/ray_clear` 认 `radius`（圆而不是外接方框，等价多赚 0.03–0.06 m）。Gazebo 对照结果见下表。
2. preprocess.py 两处改动（白名单 + 允许跨族多版本或按集合分包）谁来做；是否按集合分别打包而不是合并。
3. `collection_manifest.json` 新增 `density` 字段、`geometry_version` 按族写：下游有没有脚本把 `geometry_version` 当常量比对？（仓库内 grep 没有。）

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `sim/tools/test_*.py` 全部 34 个 + `scene_gen.py --self-check` + `scene_forest.py --self-check` | 全部 rc=0（`agent-forest/regress/summary.txt`，含 test_spawn legacy 字节断言、test_audit_dataset 21 例、test_scene_forest） |
| `scene_gen.py --task forest_e1 --density medium --seed 0 --out …/worlds/f0.sdf` | 生成 SDF + meta，`geometry_version=uavrgb.forest_e1.v1`，与 `scene_forest.py` 直出字节一致 |
| `scene_gen.py --density sparse`（不带 forest）/ `--task forest_e1` 不带 density | argparse 报错退出，提示「--density 与 --task forest_e1 必须同时给」 |
| `e1_pipeline.py --dry-run --out …/dry` | 7 步全部打印命令；collect 为 `collect_parallel.py --instances 4 --episodes-per-instance 75 --steps 40 --base-seed 3100 --task base …`（与改前一致）；临时把森林行解开，collect 多出 `--task forest_e1 … --density sparse`，audit 分别对两目录，preprocess 一次调用吃两目录 |
| 离线 A*（20 seed/档）穿林集数 vs 膨胀 | 下表 |
| 真 Gazebo E1 教师 3 档 × 6 集 × 40 步 | 下表 |
| `audit_dataset.py` 森林 sparse 集 | `geometry_version PASS（uavrgb.forest_e1.v1；常量核对 6/6 集）`、`label_recompute PASS 0/240 不一致`、`class_share FAIL（SAFE_NEUTRAL 占比 0，train 整类缺席）`——FAIL 是数据事实（教师全程无 SAFE），不是审计接错 |
| 录像 `eval_closed_loop.py --record-video`（forest sparse seed 4103，80 步） | rc=0，`videos/seed_4103.mp4` 816 帧，`video_error=None`；success=True、collision=False |
| 对照实验（改膨胀）两组 audit | `label_recompute` 51.7 % 不一致 → FAIL，`safe_collision` 19/30 → FAIL（预期，见对照表说明） |

### 离线 A*（`teacher.astar_path`，无 Gazebo）：穿林集数 / 20

| 膨胀 m | 0.40（现 E1） | 0.30 | 0.25 | 0.20 | 0.15 | 0.12 |
| --- | --- | --- | --- | --- | --- | --- |
| sparse（缝 ≥0.45） | 0 | 2 | 12 | 20 | 20 | 20 |
| medium（缝 ≥0.35） | 0 | 0 | 5 | 10 | 20 | 20 |
| dense（缝 ≥0.28） | 0 | 0 | 0 | 3 | 9 | 19 |

没穿的全是「绕行」（A* 从带外 |y|>1.5 绕过去），没有一例无路——因为 A* 边界没有墙。

### 真 Gazebo + E1 SITL，仓库教师原样（A* 0.40 / 局部 0.30），6 集 × 40 步，hold 1.0

seed 4100–4105，`teacher3/<density>/`，统计口径：到达 = 真值 max_x ≥ 最远柱 x + 1.0（与 eval 的 route_choice 同）；SAFE 死锁 = 末尾 ≥3 步连续 SAFE；绕行 = 森林 x 段内真值 max|y| > 1.5（带外）。

| 档 | 集 | 到达 | 碰撞 | SAFE 死锁 | 绕行 / 穿林 | SAFE 步占比 | 动作 fwd/yawL/yawR | 距柱面最小 m | s/集 | audit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse | 6/6 | 2/6 | 0 | 0 | 6 / 0 | 0 % | 193/23/24 | 0.286 | 81.5 | 仅 class_share FAIL（无 SAFE 类） |
| medium | 6/6 | 2/6 | 0 | 0 | 6 / 0 | 0 % | 203/18/19 | 0.314 | 110.1 | 同上 |
| dense | 6/6 | 2/6 | 0 | 0 | 6 / 0 | 0 % | 203/20/17 | 0.361 | 92.4 | 同上 |

- 三档结果几乎一样，因为教师根本没进森林：森林 x 段内 max|y| 1.8–2.9 m，全部贴着带外绕。到达 2/6 只是 40 步（≈8 m 航程，fwd 位移 ≈0.20 m/步）不够走完绕行路线，末步 x 普遍 5.7–6.3 m，差 0.1–0.6 m；不是死锁也不是碰撞。
- 距柱面最小 0.29–0.48 m：完全是带外路过时的距离，说明没有一次穿缝。
- 结论：**现状下三档都不能当「穿林」数据集**；sparse/medium/dense 采出来的是同一种「绕开整片障碍」行为，audit 还会因无 SAFE 类 FAIL。过不去不是执行器问题（0 碰撞、位移稳定 0.20 m/步、无 SAFE），是教师 0.40 m 膨胀把所有缝（最宽 0.45）都封死，A* 另找了带外的路。
- 可用性：如果只要「E1 在柱群边上绕行」的多样性，sparse 现在就能采（0 碰撞、label_recompute 100%），但要把 class_share 的 SAFE 要求对这族放开或掺 base 集；要「穿林」数据，三档都得先做待确认 1。

### 对照：临时缩小膨胀（不入库，`clearance_override/sitecustomize.py`）

同 seed 4100–4105，6 集 × 40 步，`phase2/<density>/`。A*/局部膨胀取离线表里「20/20 可穿」的最大值。

| 档 | 膨胀 A*/局部 | 到达 | 碰撞集 | 穿林 / 绕行 | SAFE 死锁 | SAFE 步占比 | 距柱面最小 m | 碰撞步按动作分（SAFE/FWD/YAW） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse | 0.20 / 0.15 | 3/6 | 2/6（20 步） | 6 / 0 | 0 | 12 % | 0.090 | 19 / 0 / 1 |
| medium | 0.15 / 0.10 | 1/6 | 5/6（21 步） | 5 / 1 | 0 | 3 % | 0.086 | 6 / 3 / 12 |

- 缩小膨胀后教师确实穿林（sparse 6/6、medium 5/6），到达 3/6 与 1/6 的差距全是碰撞。
- sparse 的 20 个碰撞步里 19 个发生在 **SAFE（悬停）** 期间：seed 4101 第 15 步贴柱 0.135 m 触发 SAFE 后，E1 悬停 12 s 横向漂了 0.23 m（≈2 cm/s，与 SITL 已知悬停漂移一致）撞上柱子；seed 4105 是偏航时擦柱后悬停继续漂。也就是说 sparse 在 0.20/0.15 下，**A* 路径本身是能过的，败在「SAFE = 原地悬停」对会漂的 E1 不安全**，这一条是执行器/原语问题，不是路径余量问题。
- medium 在 0.15/0.10（E1 半径 0.102 外只剩 0.05 m）碰撞分布到偏航（12）、前进（3）、悬停（6）：纯跟踪误差 + 0.1 m 滑行就能吃掉 0.05 m，这是**余量问题**——缝 0.35 m 对 0.2 m 宽的机体就是单侧 7.5 cm，再怎么改原语也不够。dense（缝 0.28，单侧 3.8 cm）更不用试。
- 结论：sparse 有救（改教师膨胀 + 把近柱 SAFE 换成「后退/侧移」或缩短 hold），medium 要么把 `DENSITY_GAP_M.medium` 提到 ≥0.45（那就是 sparse）、要么等 SITL 位置环把悬停漂移压到 <0.5 cm/s 再谈，dense 对这个动作原语教师不可行。
- audit 对这两组的 `label_recompute` 51.7 % 不一致 / `safe_collision` 19/30 FAIL 是预期：标签来自改了膨胀的教师，证明这两组**不能进数据集**，也证明 audit 抓得住。

### 录像

- 文件：`/root/rivermind-data/uav_rgb/tmp/agent-forest/video-sparse/videos/seed_4103.mp4`（1280×720，10 fps，816 帧 / 81.6 s，10 MB；原始帧 1.1 GB 已删），报告 `video-sparse/eval.json`。
- 命令：`eval_closed_loop.py --policy teacher --task forest_e1 --density sparse --dynamics cf_sitl --vehicle e1_mini --split train --seeds 4103 --episodes 1 --steps 80 --hold 1.0 --record-video --keep-work`（带 chase/top 相机，没加 `--no-aux-cameras`）。
- 结果：success=True（route_choice，max_x 7.05）、collision=False、80 步、动作 FWD 41 / YAW_L 27 / YAW_R 12 / SAFE 0。顶视图里黄色轨迹沿 y≈−1 到 −2.4 从森林带南侧绕过去，机载视角全程看不到柱子——就是上面说的「绕林」。

## 风险与遗留

- 森林 `split` 用自己的 salt（`uavrgb.forest_e1.v1`）：与 base 集的 train/val/test 划分互不相干，合并打包时同一 seed 在两族可能落不同 split，这是设计上的（几何不同），但看指标时别把两族 SR 混报。
- `collection_manifest.geometry_version` 现在按族写（campus 之前也是硬写 base salt，这次一起改了）；若有人拿旧 campus 集合 manifest 比对会看到不同值。
- `eval_closed_loop.py` 里 `import campus` 现在没用到，留着没删（最小改动）。
- `e1_pipeline.py` preprocess 合并多集合时要求各集 geometry_version 一致（preprocess.py 限制），森林行启用前必须先改 preprocess.py。
- 实验进程：`teacher3`（PGID 436329）、`phase2`（PGID 450270）都写 DONE 后自然退出；若中途要停：`kill -- -<PGID>`，不要 `pkill -f`。

## 更新 2026-11-07 05:35（主会话补记）

- 代码改动已按归属提交：`1825c8b feat(sim/tools): forest_e1 场景族接进 scene_gen/split/collect_parallel/eval/audit/e1_pipeline`（提交前复跑 test_scene_forest / test_audit_dataset / test_spawn / test_split_manifest 16 项、scene_gen 与 scene_forest `--self-check`，全绿）。
- 待确认 1（教师膨胀按 (vehicle, task) 取值 + 柱子按圆判定）与待确认 2（preprocess 几何版本白名单 / 跨族打包）已交给第三阶段 DATA agent 在 `data-20261107` 分支实现；「柱旁 SAFE 对会漂的 E1 不安全」交给 SITL agent 先查执行器是否该加位置保持，再决定教师语义改不改。
