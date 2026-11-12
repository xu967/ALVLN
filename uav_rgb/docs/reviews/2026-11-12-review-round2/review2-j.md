# 第二轮交叉审核报告：角度 J（3DGS 线 11-12 的改动）

- 审核代号：J（只读审核，未改任何仓库与工作树文件）
- 审查基线：开工时刻 `2026-11-12 07:57:20 CST`，主仓库 `main` 提交 `ada0ca5`（收尾时刻 `08:20 CST` 推进至 `7b6e71a`，期间 `sim/3dgs/` 与 `bearing-guard-3dgs/` 无新增改动）；在途分支 `p12-s1b` 开工提交 `44595b0`（收尾时刻推进至 `835d2ad`，新增提交 `460639d` 将开工时已审读的 `experiments/2026-11-12-3dgs-rebaseline/` 提交入库，`sim/3dgs/` 与 `driver.py` 无变化）。
- 涉及合并与分支提交：K3 合并 `df15994`（代码 `55d87e8`、`be64995`、`5058ae4`）、W1-12 合并 `dfb98c4`（代码 `bea8812`、`f9d938d`、`6f897ce`、`de56c36`）、S1a 合并 `5387747`（代码 `32caa35`、`8e2a383`）、以及未合入分支 `p12-s1b`（`6b3a2f4`..`460639d`）。
- 草稿与验证日志目录：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-j/`（含 `verify_all.py`、`verify_all.log`、`verify_unit.py`、`verify_unit.log`）。

## 1. 结论与两个独立判断

本角度最要紧的 3 件事：第一，分支 `p12-s1b` 的碰撞口径 `v2` 并未真正统一判定、规划、规则臂、出生检查四处的口径且主线评测入口漏接：`HEIGHT_MARGIN_M = 0.025 m` 只按第 3 步起的稳态 `pose_gt.z` 定出，漏算了第 0–2 步起飞过冲（`z - spawn.z` 实测达 `+0.070 m`），导致室内 46.7%–50.0% 集在起飞步跨入规划与出生检查看不见的第 3 层甚至丢掉第 1 层，同时 `sim/3dgs/eval_3dgs.py` 完全未接 `--collision-model` 仍停留在旧口径 `v1`，若直接用于卡 E4 会让端到端学生在宽松口径受评而基线在严格口径受评。第二，`v2` 按格中心离散的结构元素在 0.1 m 体素下变成 3×3 方核（斜向多扩 3.94 cm，重判 W1-12 教师多翻 11–13 集、K3 `rule2-depth` 碰撞数翻倍），而在 0.2 m 体素室外场景仍退化为十字核（斜向漏判达 10.2 cm、轴向多判达 9.8 cm），对不感知体素轴向的视觉学生有系统性偏倚。第三，将规则 v2 或 `swept` 教师接入 E4 训练采集与 DAgger 存在多处阻断与标签污染：`DaggerPolicy` 下 `swept` 教师的 `_waits` 与 `RuleOraclePolicy` 的 `last_action` 均按未执行的教师自身输出更新（连续 2 步学生未悬停即关掉扫掠保护打 `FORWARD_SLOW` 标签，或误解锁静止起步的 0.55 m 短通道），`StudentAdapter` 漏调 `nav_cond.check_student_cond` 会把航向码模型静默当成 `prev-action` 喂入非法码 `1`，且 `S1a` 与 `S1b` 合入时 `rebaseline/run.sh` 因禁止覆盖空 `SPLIT_FROM_IDENTITY` 会重回写死 `split="test"`。

对主会话补充的两个疑点的独立判断：
- **疑点 ①（`v2` 的 3×3 方核近似 vs 点到原始体素方块的精确欧氏距离查询）**：**应当把连续坐标查询（`collides`、`sweep`、`sweep_path`、`rule_oracle._hit`）改为点到原始体素方块的精确欧氏距离 $\le R$，而规划栅格 `OccPlanner.hard` 保留 `_dilate_euclid`（格中心欧氏核）**。定量依据见 `J-2`：在 0.1 m 体素下，A* 节点都在格中心，格中心处 3×3 方核与精确距离数学恒等；但在连续轨迹查询上，3×3 方核斜向把碰撞边界外推至 $0.1414\text{ m}$（窄门两侧合计缩窄最多 $7.88\text{ cm}$），已录轨迹重判使 W1-12 `swept` 教师多翻转 11–13 集（单层单弦 89 vs 102，层带折线 87 vs 98），K3 `formal40` 的 `rule2-occ` 翻转 1 集（精确 0 集）、`rule2-depth` 翻转 2 集且碰撞从 4 集升至 8 集；在 0.2 m 体素室外场景下，`_dilate_euclid` 退化回十字核，对角漏判整整 $10.2\text{ cm}$ 而轴向多判 $9.8\text{ cm}$（`wanderland` 的 `rule2-depth` 碰撞 8 vs 精确 3）。以后评测端到端视觉学生（E4）时，学生看的是连续 RGB 渲染表面，看不见对齐坐标轴的体素网格方角，3×3 方核与 0.2 m 十字核会造成直接的假阳性与假阴性。实测带粗筛的纯 NumPy 精确距离查询（`verify_all.py`）在 `sweep_path`（30 点）耗时 `1.77 ms`、`occ_columns`（8100 点）耗时 `5.88 ms`，无性能瓶颈。
- **疑点 ②（规划硬栅格取 `scene.spawn.z`：路线自身出生高度与顶层 `spawn.z` 不同时会怎样，现有场景是否有这种路线）**：遍历 `/root/rivermind-data/uav_rgb/3dgs-scenes/*/scene.json` 全部 13 个场景（10 个带 `routes` 的场景共 273 条路线），**现有场景中所有 `routes[*].spawn.z` 与 `waypoint.z` 均等于顶层 `scene.spawn.z`（差异路线数 0）**，当前重定基线不受影响。但代码层面（见 `J-9`），`driver.py` 与 `planner.py:main()` 均在 `apply_route` 之前用顶层 `scene.spawn.z` 只构建一次 `OccPlanner(occ)`；一旦后续场景（如批次 N 的多楼层或跨高度路线）出现路线级 `spawn.z` 与顶层不同，起飞高度、出生检查、碰撞判定、规则臂列距离都会跟路线自己的 `spawn.z` 走，唯独 A* 规划、`OracleRouteGuide`、`route_check` 与 SPL 最短路仍锁在顶层 `spawn.z` 的层带上；若顶层省略 `spawn`，`OccPlanner(occ)` 在 `v2` 下直接抛 `ValueError`。

---

## 2. 发现清单（按严重程度排序）

### J-1：碰撞口径 `v2` 的高度余量漏计起飞过冲（近半数集跨入层 3 或丢掉层 1），且禁止 `spawn_two_layer` 去掉水平余量，使四处口径在每集前 3 步再次分裂
- **位置**：`sim/3dgs/collision.py:220-229, 514-539`（分支 `p12-s1b` 提交 `51ac272`）；`sim/3dgs/collect_3dgs.py:240-254, 664-665`（提交 `51ac272`、`44595b0`）；`experiments/2026-11-12-3dgs-rebaseline/measure_z_margin.py:22, 47-55`（提交 `51ac272`）。另关联 `main` 上 W1-12 的 `sim/3dgs/collect_3dgs.py:250-256`（合并提交 `dfb98c4`）。
- **现象**：
  1. `measure_z_margin.py` 设 `STEADY_FROM_STEP = 3` 且只读步末 `pose_gt.z`，定出 `HEIGHT_MARGIN_M = 0.025 m`（层带半宽 `height_band_half = 0.028 + 0.025 = 0.053 m`）。`collision.py:225-227` 注释据此写「起飞过冲最大 +0.046 m；过冲时带下沿仍比出生高度低 0.007 m，出生高度所在的层仍在带内」。
  2. 现有全部 0.1 m 体素场景的 `spawn.z` 压在层 1 `[zlo+0.1, zlo+0.2)` 与层 2 `[zlo+0.2, zlo+0.3)` 的边界 `zlo + 0.200 m` 上。出生检查 `_spawn_for_episode` 与规划硬栅格 `OccPlanner.hard = occ.planning_grid(spawn.z)` 的 `layer_span(spawn.z)` 恒为 `(1, 2)`。
  3. 实测 5 组已有实验的 `actions.jsonl`（含 `pose_before.z`）与 `trace.jsonl`（`verify_all.log` 第 18–22 行）：起飞刚结束进入第 0 步时，`z - spawn.z` 最高达 `+0.0700 m`（并非 `+0.046 m`）。只要 `z - spawn.z > +0.047 m`，带上沿 `z + 0.053` 就跨过 `zlo + 0.300` 进入层 3；当 `z - spawn.z > +0.053 m` 时，带下沿 `z - 0.053 > zlo + 0.200`，连层 1 都退出，层带变成 `(2, 3)`。实测离开规划层带 `(1, 2)` 的集数占比：W1-11 室内 `56/120`（46.7%，其中 18 集为 `(1, 3)`、38 集为 `(2, 3)`），W1-12 室内 `56/120`（46.7%），K3 `holdout` `48/96`（50.0%，37 集为 `(2, 3)`），K3 `formal40` `56/120`（46.7%），W1-12 室外 `26/120`（21.7%）。5 个室内场景里层 3 相比层 1∪2 额外多出 `379` 至 `5005` 个膨胀占据格（如 `sage3d_841150` 多 5005 格、`sage3d_841012` 多 1187 格）。
  4. 同时，`p12-s1b` 在 `v2` 下报错禁止 `spawn_two_layer=True`，使出生检查只剩单点 `occ.collides(nx, ny, spawn.z)`，没有任何水平漂移余量。而 `cf_sitl` 起飞段存在中位 `3.1 cm`、p90 `5.5 cm`、最大 `8.2 cm` 的水平漂移（W1-12 报告 §5 实测）；此外 W1-12 在 `v1` 下的 `_spawn_two_layer_ok` 用 `occ.clearance >= body_radius + 0.05 = 0.152 m` 查的是到原始体素**中心**的距离，扣掉半体素 `0.05 m` 后离轴向膨胀边界只有 `2 mm`（解释了 W1-12 臂 B 的 `839992 r79 ep2` 过了判空后起飞漂 `3.9 cm` 仍在第 0 步撞）。
- **为什么是缺陷与影响**：直接影响卡 S1b 即将起跑的重定基线与后续 E4 采集。近一半的集在第 0–2 步中，碰撞判定 `_step_collision`、教师前向检查 `occ.sweep`、规则臂 `occ_columns` 查的是 `(1, 3)` 或 `(2, 3)`，而出生检查与 A* 规划查的仍是 `(1, 2)`：候选点通过出生检查后起飞可能因层 3 障碍或 `3–8 cm` 起飞水平漂移在第 0 步即撞；`OccPlanner` 认为层 3 障碍处畅通而教师前向检查判堵，产生规划与前向检查的视野分裂；规则臂在第 0 步扫到层 3 障碍还会写入 24 步世界系记忆，稳态回落后继续挡路。
- **复现与验证**：运行草稿脚本 `CUDA_VISIBLE_DEVICES="" nice -n 15 /root/rivermind-data/uav_rgb/3dgs-venv/bin/python /root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-j/verify_all.py`，输出见 `verify_all.log` 第 17–29 行。
- **建议的最小修法**：① 若保留稳态余量 `0.025 m`，出生检查应直接对 `cf_sitl.reset()` 起飞就绪后的实际首帧位姿校验（或在起飞前判空时覆盖起飞过冲上界 `+0.070 m` 与起飞水平漂移余量：离表面距离不低于 `body_radius + 0.06 m`）；若要求全集第 0 步起四处严格同层带，则 `HEIGHT_MARGIN_M` 需按含第 0 步 `pose_before.z` 的 p99（约 `0.065–0.070 m`）定值并同步进 `planning_grid`。② 修正 `collision.py:225-227` 中失实的 `+0.046 m` 与不丢层注释。
- **状态**：已复现。
- **建议归属**：并入正在执行的卡 `S1b`（在正式重定基线起跑前修正，或记录为重定基线已知边界）。

---

### J-2：碰撞口径 `v2` 的格中心膨胀核在 0.1 m 体素下斜向过严最多 3.94 cm，而在 0.2 m 体素室外场景下仍退化为十字核（斜向漏判达 10.2 cm）
- **位置**：`sim/3dgs/collision.py:478-505`（分支 `p12-s1b` 提交 `6b3a2f4`）。
- **现象**：
  1. `_dilate_euclid` 以「格中心到原始体素方块的欧氏距离 $\le R$」决定整格是否标为占据。
  2. 在 `voxel = 0.1 m`、`R = 0.102 m`（全部室内场景及 `garden`、`bicycle`）下，得到 3×3 实心方核（边长 `0.30 m`）。斜向邻格 `(±1, ±1)` 外角点距原始体素角点为 $\sqrt{0.10^2 + 0.10^2} = 0.1414\text{ m}$，比机体半径多判 `3.94 cm`（窄门两侧斜向合计缩窄最多 `7.88 cm`）。
  3. 在 `voxel = 0.2 m`、`R = 0.102 m`（室外 `callinstrasse`、`wanderland_1MlUucug`）下，斜向邻格中心距原始体素角点 `0.1414 m > 0.102 m` 不膨胀，`_dilate_euclid` 仍输出与 `v1` 完全相同的十字核 `dilation_cells = (1, 0)`：机体中心在 `(0.101, 0.101)` 距体素角点 `(0.100, 0.100)` 仅 `1.4 mm`（机体嵌入体素 `10.1 cm`）时不判碰（斜向漏判达 `10.2 cm`）；而在轴向邻格 `(1, 0)` 外沿 `(0.299, 0.0)` 距原始体素 `19.9 cm` 时却判碰（轴向多判达 `9.7 cm`）。
  4. 全量回放对比（`verify_all.log` 第 34–54 行）：
     - 室内 W1-12 `swept` 教师（120 集，原判 112）：单层单弦下 3×3 方核为 `89/120`（翻 23 集）、精确欧氏为 `102/120`（翻 10 集），方核多判负 **13 集**；叠加层带与 `trace` 折线后，方核为 `87/120`（碰撞 33）、精确欧氏为 `98/120`（碰撞 23），方核仍多判负 **11 集**。
     - 室内 K3 规则臂：在 `holdout`（96 集）上各规则臂成败 0 翻转（`rule2-depth` 在方核下碰撞多 1 集：3 vs 2）；在 `formal40`（120 集）上精确欧氏对三臂均 0 翻转，而 3×3 方核使 `rule2-occ(v2f)` 翻转 **1 集**（105→104）、`rule2-depth(v2d)` 翻转 **2 集**（106→104）且整集碰撞从 **4 集升至 8 集**。
     - 室外 `wanderland_1MlUucug`（`0.2 m` 体素，30 集）：`K3 rule2-depth` 在 `v2` 十字层带下判 8 集碰撞（到点 22/30），精确欧氏层带下仅 3 集碰撞（到点 24/30）。
- **为什么是缺陷与影响**：`rule2-occ` 因自身用同一个膨胀栅格测距并留 `0.15 m` 余量，对核形状不敏感；但基于视觉/深度的策略（`rule2-depth` 与未来 E4 的端到端 RGB 学生）看到的是连续 3DGS 表面，看不见与世界坐标轴对齐的离散网格方角，在 0.1 m 场景斜切拐角时会被 3.94 cm 隐形方角误杀，在 0.2 m 室外场景又同时存在 10.2 cm 斜向漏判与 9.7 cm 轴向误判。
- **复现与验证**：同 `J-1` 运行 `verify_all.py`，见 `verify_all.log` 第 24–54 行。`verify_all.py` 的 `exact_grid_hit_vec` 与暴力公式在 500 个随机点上 100% 一致；单进程纯 NumPy 粗筛精确查询实测耗时：`sweep_path`（30 点）`1.769 ms`（方核 `0.164 ms`），`occ_columns`（8100 点）`5.879 ms`（方核 `1.331 ms`）。
- **建议的最小修法**：`Occupancy._grid_hit` 在 `collision_model="v2"` 下改为「粗膨胀掩码初筛 + 候选点到邻域 `[-reach, reach]^2` 原始体素方块的精确水平欧氏距离 `<= body_radius + inflate`」（约 35 行纯 NumPy 代码，见 `verify_all.py:29-112`）；`OccPlanner.hard` 保持 `planning_grid` 的格中心欧氏核（0.1 m 体素下 A* 顶点在格中心，格中心处两者恒等）。
- **状态**：已复现。
- **建议归属**：并入卡 `S1b`。

---

### J-3：主线评测入口 `sim/3dgs/eval_3dgs.py` 完全漏接 `--collision-model`，合入 S1b 后仍硬编码使用旧口径 `v1`
- **位置**：`sim/3dgs/eval_3dgs.py:47, 245, 277-285, 345, 368`（`main` `7b6e71a` 与 `p12-s1b` `835d2ad` 均 0 行改动）。
- **现象**：`p12-s1b` 只在 `collect_3dgs.py` 与实验脚本 `experiments/2026-11-10-bearing-guard-3dgs/driver.py` 接入了 `collision_model`。主线学生闭环评测器 `sim/3dgs/eval_3dgs.py`：① 无 `--collision-model` CLI 参数；② 第 245 行 `occ = collision_mod.from_scene(scene, gaussians=gaussians)` 固定构造默认 `v1` 占据（十字核、单层跳变）；③ 第 277 行传入 `run_episode` 后因 `occ.collision_model == "v1"`，每步碰撞继续扫首尾直线弦而不扫 `trace` 折线；④ 第 345 行 `eval.json` 的 `metric_version` 恒写 `"3dgs.pilot.v2"`（不调 `collision_mod.metric_version`），顶层无 `collision_model` 字段。此外 `driver.py:287-289` 仅在 `config.json` 显式提供 `collision_model` 时才往 `metrics.json` 写 `metric_version`，缺省时不写该键。
- **为什么是缺陷与影响**：卡 S1 是端到端 E4 的前置。E4 以及 `tools/dagger_pipeline.py`、`tools/compare_datasets.py` 评测学生时走 `eval_3dgs.py`。若教师采集与规则基线用 `v2` 口径，而学生闭环评测走 `eval_3dgs.py` 默认的宽松 `v1` 口径，比较会混入偏袒学生的口径差（教师臂上 `v1` 与 `v2` 相差 14–25 集），且 `stats_report.py` 若不把 `collision_model` 列入协议一致性键还可能混表。
- **复现与验证**：核对 `git -C /root/claude-code-backup-20261028/code/uav_rgb diff ba54628..835d2ad -- sim/3dgs/eval_3dgs.py` 输出为空。
- **建议的最小修法**：在 `eval_3dgs.py` 增加 `--collision-model`（默认 `DEFAULT_COLLISION_MODEL` 以锁住旧行为），传给 `collision_mod.from_scene(..., collision_model=args.collision_model)`，`eval.json` 顶层记录 `"collision_model": occ.collision_model` 并将 `"metric_version"` 改为 `collision_mod.metric_version(METRIC_VERSION, occ.collision_model)`；同时在 `tools/stats_report.py` 的协议一致性键中加入 `collision_model`。
- **状态**：读代码确认。
- **建议归属**：`eval_3dgs.py` 并入卡 `S1b`；`stats_report.py` 协议键并入卡 `W1-14` 后续小补或 `S1b`。

---

### J-4：S1a 与 S1b 的合并冲突与 `rebaseline/run.sh` 锁死空 `SPLIT_FROM_IDENTITY` 陷阱
- **位置**：`experiments/2026-11-10-bearing-guard-3dgs/driver.py:41-46, 301-302`（`main` `5387747`）对 `:40-44, 308-310`（`p12-s1b` `44595b0`）；`test_driver_split.py:117` 对 `test_driver_collision_model.py:88, 106`；`experiments/2026-11-12-3dgs-rebaseline/run.sh:45, 84, 152`（`p12-s1b` 提交 `460639d`）。
- **现象**：
  1. `sim/3dgs/collect_3dgs.py` 在 `S1a` 与 `S1b` 之间 `git merge-file` 自动合并零冲突，且 `test_3dgs_student_prev_action.py:320-338` 以同次运行的 `ForwardPolicy` manifest 为动态基准集 `keys`，与 `S1b` 新增的 `manifest["collision_model"]` 兼容。
  2. `driver.py` 有两处直接文本冲突：① 文件头 docstring 末尾（两卡各自追加一段配置键说明）；② 第 301–310 行 `done.json` 末尾追加键（`S1a` 追加 `split, split_policy`，`S1b` 追加 `collision_model, metric_version`）。两卡的校验块在中部相隔数行，可自动合并。注意 `test_driver_split.py:117` 与 `test_driver_collision_model.py:88` 都用 `list(done) == OLD_DONE_KEYS + [...]` 严格锁了键顺序：单开其中一个键时两套测试都绿，但两键同开（重定基线正式运行场景）时，合并时选定的先后顺序需要补一条组合单测锁住。
  3. 更隐蔽的交互在 `p12-s1b` 的 `experiments/2026-11-12-3dgs-rebaseline/run.sh`：第 84 行写死默认空串 `SPLIT_FROM_IDENTITY=${SPLIT_FROM_IDENTITY:-}`（注释写合并进 main 后改为 `"true"`），同时第 45 行把 `SPLIT_FROM_IDENTITY` 放进了禁止环境变量覆盖列表，第 152 行在正式运行（非 `SMOKE=1`/`DRY_RUN=1`）时只要检测到覆盖就 `refuse_before_run 2` 退出。若主会话合并 `p12-s1b` 后照第 31 行命令直接正式起跑而不改第 84 行字面量，3DGS 重定基线全部 6 个套件产出的几千集 `episode_manifest.json` 仍会全部写死 `"split": "test", "split_policy": "diag-eval-fixed-test"`，使刚合入的 D1 修复在重定基线上失效；而若从命令行传 `SPLIT_FROM_IDENTITY=true` 起跑则直接报错 `rc=2`。
- **为什么是缺陷与影响**：直接导致 `p12-s1b` 合并冲突，且若漏改 `rebaseline/run.sh:84` 字面量，重定基线产物会继续污染 `"split": "test"` 标签。
- **复现与验证**：`git merge-tree --write-tree 7b6e71a 835d2ad` 复现 `driver.py` 冲突（草稿输出见 `review2-j/tmp/driver_merged.py`）；读 `rebaseline/run.sh:45, 84, 152` 确认覆盖拦截闭环。
- **建议的最小修法**：合并 `driver.py` 时保留两段 docstring，约定 `done.json` 先追加 `split, split_policy`（S1a 先入 main）再追加 `collision_model, metric_version`（S1b 后入 main）；合并后立即将 `experiments/2026-11-12-3dgs-rebaseline/run.sh:84` 改为 `SPLIT_FROM_IDENTITY=${SPLIT_FROM_IDENTITY:-true}`。
- **状态**：已复现。
- **建议归属**：主会话合并 `p12-s1b` 时一并处理（或交 `S1b` 收尾提交）。

---

### J-5：3DGS 学生条件通路（S1a）漏接 `nav_cond.check_student_cond`，传入航向码模型会静默当成 `prev-action` 喂入非法码 `1`
- **位置**：`sim/3dgs/collect_3dgs.py:167, 189-191, 708-709`、`sim/3dgs/eval_3dgs.py:265, 274`、`experiments/2026-11-10-bearing-guard-3dgs/driver.py:184, 229`（合并提交 `5387747` / 代码提交 `32caa35`）。
- **现象**：`StudentAdapter.__init__` 第 167 行仅凭 `getattr(student, "prev_dim", 0)` 非零就无条件设 `self.student_cond = "prev-action"`，每步向 `student.act` 传 `prev_action=self.last_executed`（取值 `{0, 1, 2, 3}`），并在 manifest 写 `"student_cond": "prev-action"`。虽然该文件第 42 行已 `import nav_cond`，但 `collect_3dgs.py`、`eval_3dgs.py`、`driver.py` 三处构造 `StudentPolicy(ckpt, infer_bin)` 后均未调用 `nav_cond.check_student_cond`。
- **为什么是缺陷与影响**：W1-9 与正在跑的 E2E-1 产出的 `heading-code` 模型与 `prev-action` 模型在 C++ checkpoint 握手行里都是 `prev_action_dim = 4`。一旦把 `heading-code` 权重传给 3DGS 的任一入口：Gazebo 侧 `check_student_cond` 会判 `mismatch` 拒绝运行，而 3DGS 侧完全不报错，把上一步动作 `FORWARD_SLOW = 1`（在 `heading-code` 定义中码 `1` 从未使用，属于纯分布外输入）当作条件字节喂给模型，且在 manifest 里错误标记为 `"prev-action"`。
- **复现与验证**：运行 `verify_unit.py`（日志 `verify_unit.log` 第 4–5 行）：对绑定了 `cond="heading_code.v1"` 数据集的权重，`nav_cond.check_student_cond("prev-action", ckpt, 4)` 返回 `mismatch`，而 `StudentAdapter` 静默输出 `student_cond='prev-action'` 并向推理进程发送字节 `[0, 1]`。
- **建议的最小修法**：在 `StudentAdapter.__init__(self, student, ckpt=None)`（或在 `collect_3dgs.py:708`、`eval_3dgs.py:265`、`driver.py:184` 创建 `StudentPolicy` 处；`student.ckpt` 若已在 `StudentPolicy` 上保存则直接在 `StudentAdapter.__init__` 读 `getattr(student, "ckpt", None)`）调用 `nav_cond.check_student_cond(nav_cond.COND_PREV_ACTION, ckpt, student.prev_dim)`，`status == "mismatch"` 时抛 `ValueError`。
- **状态**：已复现。
- **建议归属**：并入卡 `E2E-C1` 或卡 `E4` 开工前的前置小修。

---

### J-6：W1-12 扫掠教师 `OccAstarTeacher(clearance="swept")` 在 `DaggerPolicy` 下 `_waits` 状态污染，且 `_swept_blocked` 仍扫单高直线弦、不保护悬停滑行
- **位置**：`sim/3dgs/planner.py:278-295, 321-333`、`sim/3dgs/collect_3dgs.py:236-242, 660-665, 695-710`（合并提交 `dfb98c4` / 代码提交 `bea8812`、`f9d938d`）。
- **现象**：
  1. `collect_3dgs.py:660-665` 已显式允许 `--policy student --teacher-clearance swept`（DAgger 采集）。在 `DaggerPolicy.__call__` 中，每步先调 `self.expert(rgb, depth, state)` 取专家标签，再按概率 `beta` 决定执行学生动作还是专家动作。`OccAstarTeacher` 在进入 `wait_swept` 分支时将 `self._waits` 加 1，但从未核对本步最终实际执行的动作是不是 `SAFE_NEUTRAL`。当 `beta < 1`（例如 `beta = 0` 学生自飞）时，若学生连续 2 步未听从教师悬停而继续执行 `FORWARD_SLOW` 或转向，机体仍处于高速冲墙状态，第 3 步教师因 `waits == 2 == SWEPT_MAX_WAITS` 直接跳过扫掠否决，输出 `FORWARD_SLOW`（`last_branch = "forward"`）作为专家标签。
  2. 几何上：`_swept_blocked` 用起点高度 `z` 扫一条零宽直线弦 `(x, y, z) -> (x + wx + reserve*ux, y + wy + reserve*uy, z)`。在 `v2` 口径下，每步碰撞判定已改为沿 10 Hz `trace` 弧线折线扫（实测弧线离弦横偏 p90 `4.2 cm`、最大 `6.8 cm`），直线弦扫掠无法覆盖转向后首个前进步弧线外鼓的 `4–7 cm`；且被否决后输出的 `wait_swept`（`SAFE_NEUTRAL`）假定悬停不动，实际带速进入悬停时滑行中位 `0.123 m`、最大 `0.177 m`（W1-12 验收中 `840865 r26 ep2` 因此撞在 `wait_swept` 步，导致臂 A 室内撞在前 `8/120 = 0.067` 未过 `<= 0.06` 的预注册线）。
- **为什么是缺陷与影响**：第 1 点会在 `swept` 教师的 DAgger 采集中把本应刹停的危险状态错误打标为 `FORWARD_SLOW`，污染训练监督；第 2 点使 `swept` 教师在 `v2` 严格口径下与 B3 折线判据之间仍有几厘米的几何缺口（`verify_all.log` 回放显示已录 `swept` 轨迹在 `v2` 完整口径下从 `112/120` 掉到 `87/120`，精确欧氏下为 `98/120`，重跑在线闭环的数字待 S1b 实测）。
- **复现与验证**：运行 `verify_unit.py`（日志 `verify_unit.log` 第 1 行）：构造侧向带速 `vy = -0.35 m/s`、机头射线通而预测位移撞侧墙的状态，`DaggerPolicy(..., beta=0.0)` 连跑 3 步输出 `[(0, 'exec=1', 'expert_label=0', 'branch=wait_swept', '_waits=1'), (1, 'exec=1', 'expert_label=0', 'branch=wait_swept', '_waits=2'), (2, 'exec=1', 'expert_label=1', 'branch=forward', '_waits=0')]`。
- **建议的最小修法**：给 `OccAstarTeacher` 增加 `note_executed(action)`（若实际执行的不是 `SAFE_NEUTRAL` 则将 `self._waits = 0` 清零），并在 `DaggerPolicy.__call__` 确定最终 `action` 后统一调 `self.expert.note_executed(action)`（外层 `OracleGuardedPolicy` 改写时经 `DaggerPolicy.note_executed` 一并转发给 `expert`）。
- **状态**：第 1 点已复现；第 2 点读代码与 W1-12 实验记录确认。
- **建议归属**：新立小修或并入卡 `E4`（若 E4 仍用 `OccAstarTeacher` 对照）。

---

### J-7：规则 v2（K3）用作 E4 训练期教师或 DAgger 专家时的五处接线阻断与状态错位
- **位置**：`sim/3dgs/rule_oracle.py:145, 171, 195-200`（合并提交 `df15994` / 代码提交 `be64995`）；`sim/3dgs/collect_3dgs.py:212, 606`；`tools/preprocess.py:259`；`experiments/2026-11-10-bearing-guard-3dgs/driver.py:223`。
- **现象**：计划 §10.4 E4 与 §10.7 第 5 条已将 E4 的训练期教师定为规则 v2，但当前代码存在五处阻断或污染：
  1. `RuleOraclePolicy.__call__`（`:195, 199`）把 `prev_action=self.last_action` 传给 `rule_v2_step`，而 `self.last_action` 记的是规则自己上一步的返回值，且没有 `note_executed` 接口。在 DAgger（`beta < 1`）或外层改写下，学生上一步实际执行了 `FORWARD_SLOW`（机体高速前冲），影子规则 v2 若上一步建议的是转向，下一步会误判为「静止起步」，解锁仅限静止起步的第 2 级超短窄通道 `(0.12, 0.55, True)`（`v2f`），在距墙 `0.55 m` 高速状态下给出错误的 `FORWARD_SLOW` 标签；`self.last_turn` 同样记的是未执行的转向。
  2. `rule_policy.check_v2_params` 允许任意级数 `levels`，但 `RuleOraclePolicy.reset`（`:171`）用固定的 `rp.V2_ENTRIES`（写死 3 级）初始化 `self.entry_counts`，传入 4 级 `levels` 命中第 4 级时 `:200` 直接抛 `KeyError: 'V2-fwd3'`。
  3. `RuleOraclePolicy` 无 `stop_radius` 属性，传入 `DaggerPolicy` 作 `expert` 时 `collect_3dgs.py:212`（`self.stop_radius = expert.stop_radius`）直接抛 `AttributeError`。
  4. `collect_3dgs.py:606` 的 `--policy` 没有 `rule-oracle` 选项；且 `RuleOraclePolicy.name = "rule_oracle"`，会被 `tools/preprocess.py:259`（只认 `"plan_action", "expert_action", "synthetic"`）整集静默跳过。
  5. `RuleOraclePolicy` 永不输出 `SAFE_NEUTRAL` 且无 goal-stop，到点后会在 waypoint 附近持续绕圈输出转向动作；同时 `driver.py:223` 构造 `OracleRouteGuide` 未传 W1-12 新增的 `nearest_in_lookahead=True`，K3 报告里 3 个导引目标点停滞集会在目标点 `0.05 m` 处反复转圈刷坏标签。
- **为什么是缺陷与影响**：现在不影响 K3 的上限评测数字（K3 都是纯规则自飞，且只用 `metrics.json` 的首次到点判定 `success_stop`），但在卡 E4 开工用规则 v2 采示范或跑 DAgger 时会立刻崩溃、打包为 0 帧、或把高速冲墙与到点绕圈的脏标签喂进训练集。
- **复现与验证**：`verify_all.log` 第 13、15 行复现 `KeyError('V2-fwd3')` 与 `AttributeError("... has no attribute 'stop_radius'")`；`verify_unit.log` 第 2–3 行复现实际执行 `FORWARD_SLOW` 后 `RuleOraclePolicy` 误触发静止起步级 `V2-fwd1`。
- **建议的最小修法**：在卡 E4 采集前：① `RuleOraclePolicy` 增加 `note_executed(action)` 同步更新 `last_action` 与 `last_turn`，`entry_counts` 按 `len(self.params["levels"])` 动态生成键；② `DaggerPolicy.__init__` 改用 `getattr(expert, "stop_radius", None)`；③ 采集模式给规则教师加到点截断（首次进入 `reach_radius_m` 即结束本集，不录绕圈尾巴）、启用 `nearest_in_lookahead=True`，并在 `collect_3dgs.py` 与 `preprocess.py` 显式登记带 `oracle` 溯源标记的教师名。
- **状态**：已复现。
- **建议归属**：并入卡 `E4`（第 2 条 `entry_counts` 可顺手修）。

---

### J-8：规则 v2（K3）在 3DGS 大航向误差下未做 $360^\circ$ 环绕折回，且 `|m|=5..6` 的累积步长误差穿透禁忌表半宽引发 6–9 步周期折返
- **位置**：`sim/tools/rule_policy.py:286-293, 330, 339-346`、`sim/3dgs/rule_oracle.py:185-195`（合并提交 `df15994` / 代码提交 `55d87e8`、`be64995`）。
- **现象**：
  1. **环绕割线封杀近侧候选**：Gazebo 里 `n` 是相对固定走廊的无界累计步数，而 3DGS 里 `err = bearing_error_deg(...)` 是 `[-180, 180)` 的环绕角，`n = round(-err / 31.7)`，一整圈仅 `11.36` 步（半圈 `±5.68` 步）。`rule_v2_entry` 的 `seek`（`:340-345`）直接按非周期线性加法算 `abs(n + m) <= max(n_cap, abs(n))` 与排序键 `abs(e + m)`。当目标在右后方 `err = -160°`（`e = +5.05, n = +5, cap = 5`）且右侧 `m = -1..-5` 被墙堵死、左侧 `m = +1, +2` 畅通时，左转跨过 `±180°` 实际离目标更近（左转 2 步后误差从 `160°` 减至 `136.6°`），却因线性 `abs(5 + 1) = 6 > 5` 被剔除，迫使规则选 `m = -6` 往被堵的右侧转。
  2. **大 `|m|` 名义角偏差穿透禁忌表**：`m_max = 6` 在 `alpha = 31.7°` 下跨度达 `190.2° > 180°`，而 `cf_sitl` 连续转向真实步长约 `34.4°`（比 `31.7°` 大 `2.7°/步`）。在 K3 真实失败集 `holdout/rule2-occ+route/sage3d_841012/r27/ep_00000` 第 53–62 步（`experiments/2026-11-11-rule-v2-3dgs/failures.md:64-74`）：第 54 步在 `(1.09, 2.74)` 面对 `+88.0°` 被堵写入禁忌表，选 `m = -6` 连右转 5 步到 `-81.5°` 前进一步后，第 60 步在 `(1.16, 2.55)`（距原禁忌点仅 `0.20 m < tabu_radius 0.30 m`）评估 `m = +6`，算得名义候选角 `-81.5° + 6 × 31.7° = +108.7°`，与禁忌表里的 `+88.0°` 相差 `20.7° > alpha/2 = 15.85°`，禁忌表漏判放行，飞机左转 2 步后第 62 步又选 `m = -3` 右转折回，形成多步周期往复直至超时。
- **为什么是缺陷与影响**：K3 预注册结论（留出路线 86/96 对 v1 64/96）本身有效（这是固定参数规则的实际闭环表现），但这两条解释了 K3 剩余 7 个「末段小范围来回」超时集的成因；若用规则 v2 给 E3/E4 生成示范轨迹，这类多步来回折返会进入训练数据。
- **复现与验证**：`verify_all.log` 第 16 行复现 `err = -160°` 割线封杀（输出 `action=3 entry=V2-seek0 m=-6`）；多步禁忌穿透见仓库内 `experiments/2026-11-11-rule-v2-3dgs/failures.md:64-74`。
- **建议的最小修法**：仅在后续规则 v3 或 E4 教师筛选时改（不改冻结的 v2）：3DGS 导引模式下用折回角差 `abs(_wrap180((yaw_deg + m * alpha) - target_bearing_deg)) / alpha` 代替线性 `abs(e + m)` 与 `abs(n + m)`，并将候选范围限制在 `|m * alpha| <= 180°`（`31.7°` 下 `m_max = 5`）；训练采集时加 `--exclude-failed-teacher` 过滤未到点的规则超时集。
- **状态**：已复现。
- **建议归属**：记录入卡 `E4`（训练采集过滤超时集；若升级规则教师则入新版本，不动 `rule_v2` 冻结逻辑）。

---

### J-9：`driver.py` 与 `planner.py:main()` 在 `apply_route` 前固化 `OccPlanner(occ)`，路线级 `spawn.z` 与顶层不同时规划层带错位
- **位置**：`sim/3dgs/collision.py:730-731`、`sim/3dgs/planner.py:155-166, 470-476`、`experiments/2026-11-10-bearing-guard-3dgs/driver.py:166-169, 193-194`（分支 `p12-s1b` 提交 `51ac272`、`44595b0`）。
- **现象**：`from_scene(scene)` 读取顶层 `scene.get("spawn")["z"]` 赋给 `occ.cruise_z`。`collect_3dgs.py:main` 与 `eval_3dgs.py:main` 先调 `apply_route` 再调 `from_scene`，因此拿到的是目标路线的 `spawn.z`；但 `driver.py`（`:167-169`）与 `planner.py:main`（`:470`）为了跨路线复用同一个 `occ`，在调用 `apply_route` 之前就用顶层 `scene` 构造了唯一的 `occ_planner = OccPlanner(occ)`，把 `occ_planner.hard` 永久锁死在顶层 `scene.spawn.z` 的层带上。
- **为什么是缺陷与影响**：现有 13 个场景（273 条路线）核实 `routes[*].spawn.z` 与顶层 `scene.spawn.z` 全部相等，已有结果与本次 S1b 重定基线不受影响。但后续批次 N 导入新场景或构造跨高度路线时，若某条路线的 `spawn.z` 与顶层不同，动力学起飞、出生检查、碰撞判定、规则列距离全在路线高度层带，唯独教师 A*、`OracleRouteGuide`、`route_check` 与 SPL 最短路在顶层高度层带；若顶层无 `spawn` 键，`OccPlanner(occ)` 在 `v2` 下直接抛 `ValueError`。
- **复现与验证**：场景核查命令与输出已验（13 个 `scene.json` 全部 `diff_count=0`）；读 `planner.py:470-476` 与 `driver.py:166-208` 确认调用次序。
- **建议的最小修法**：`driver.py` 与 `planner.py:main` 按路线高度缓存规划器 `planners_by_z = {}`，在 `apply_route(sc, r)` 后取 `rz = float(sc["spawn"]["z"])`，用 `planners_by_z.setdefault(rz, planner_mod.OccPlanner(occ, z=rz))`（3D `occ` 仍只建一次，2D `OccPlanner` 构建仅数毫秒）。
- **状态**：读代码确认（现有场景已核实无跨高度路线）。
- **建议归属**：并入卡 `S1b`。

---

## 3. 第一轮清单（`FIXLIST.md`）本角度相关条目的核对结果

### 3.1 修对了的
- **D1（`eval_3dgs` 与 `driver` 本身）**：`main` 合入卡 S1a（`8e2a383` / `5387747`）。`eval_3dgs.py` 在加载 PLY 前检查 `scene_identity` 的哈希 `split`，`train`/`val` 默认以 `rc=2` 拒绝且不建目录，`--allow-train-scene` 放行并写真实 `split` 与 `"diag-train-scene"`，产物不再出现 `"split": "test"`；`driver.py` 新增可选布尔键 `split_from_identity`，缺省保持旧行为逐位不变。（注：3 个 `tools/` 外部调用方未传 `--allow-train-scene` 已在 `FIXLIST.md` 登记归卡 W1-15b 下一轮；`p12-s1b` 的 `rebaseline/run.sh` 需改默认值见 `J-4`。）
- **B6（`cf_blackbox` 固定 `z = 2.0` 在室内静默无碰撞）**：在未合入分支 `p12-s1b`（`e845cab`）已修并有单测锁住：`run_episode` 在 `occ.grid.size > 0` 且 `dyn.z` 不在 `z_range` 内时对 `v1`/`v2` 均直接抛 `ValueError`。待合入 `main`。
- **E-12（路线目录通配排除 `r<N>.incomplete-<时刻>`）**：在未合入分支 `p12-s1b`（`f64849b`）新增 `sim/3dgs/route_dirs.py`（严格匹配 `^r\d+$`），并在 `rebaseline/summarize.py` 与 `measure_z_margin.py` 使用，且 `rebaseline/summarize.py` 同时落实了 `E-1` 单独扫描列出 `*.incomplete-*` 半成品计数。既有已冻结实验目录（K1、K3、W1-11、W1-12）按公共规则不回头改。待合入 `main`。

### 3.2 修了一半的
- **C2 = G1（前半）= F1（3DGS 学生 `prev-action` 取实际执行动作）**：`main` 合入卡 S1a（`32caa35` / `5387747`）。`prev-action` 模型在裸适配器、DAgger `beta` 替换、护栏改写、两层叠套及跨集复用下的字节流与重置均修对且有 14 条测试锁住；但漏掉了 `nav_cond.check_student_cond` 一致性闸门，传入 `heading-code` 模型（同样 `prev_dim=4`）时会静默当成 `prev-action` 喂入非法码 `1`（见 `J-5`）。G1 后半（`sim/tools/student.py` 未收到 `prev_action` 时报错）按计划留给卡 `E2E-C1`。
- **B1（碰撞膨胀核斜向退化）**：在 `p12-s1b`（`6b3a2f4`）按格中心欧氏距离修改。0.1 m 体素下消除了斜向漏判但变成过严最多 `3.94 cm` 的 3×3 方核；0.2 m 体素室外场景下仍退化回十字核，斜向漏判 `10.2 cm` 未修（见 `J-2`）；且 `eval_3dgs.py` 未接入（见 `J-3`）。
- **B2、B4、B7（高度跳层与四处高度口径统一）**：在 `p12-s1b`（`51ac272`）引入 `layer_span` 与 `planning_grid`。第 3 步起的稳态飞行已消除层边界跳变；但第 0–2 步起飞过冲在室内近半数集跨入层 3 或丢掉层 1，出生检查与规划仍只看层 1、2 且无起飞水平漂移余量（见 `J-1`），多路线 `driver.py` / `planner.py:main` 未随路线级 `spawn.z` 切换规划层带（见 `J-9`），`eval_3dgs.py` 未接入（见 `J-3`）。另 `rule_oracle.depth_columns` 仍用独立的 `±0.128 m` 高度带（属策略自身参数，未随 `occ` 改变，应在报告中写明）。
- **B3（每步碰撞沿 SITL `trace` 折线逐段扫）**：在 `p12-s1b`（`256cb06`）实现了 `Occupancy.sweep_path` 与 `collect_3dgs._step_collision`，对 `collect_3dgs` 与 `driver` 在 `v2` 下生效（且验证与逐段 `sweep` 首个碰撞点逐位一致）；但 `eval_3dgs.py` 未接 `--collision-model` 仍走直线弦（见 `J-3`）。

### 3.3 没修的
- **B5（渲染深度有效阈值 `alpha >= 0.05` 对占据 `opacity >= 0.3`）**：`main` 与 `p12-s1b`（`835d2ad`）中 `sim/3dgs/renderer.py` 与 `collision.py` 的不透明度门限均无改动，garden 场景下按 `opacity >= 0.3` 过滤重渲的因果核实尚未执行。

---

## 4. 查过但没有发现问题的地方

- **旧口径与缺省开关的逐位兼容性**：
  - K3：`sim/tools/rule_policy.py` 对 `v0`/`v1` 原有行零修改；`RuleOraclePolicy` 缺省 `rule_version="v1"` 时类属性、返回动作、`rows` 键序与 `entry_counts` 保持不变；`driver.py` 不给 `rule_version` 时不传多余关键字、不改 `done.json` 与 `metrics.json` 键集。
  - W1-12：`OccAstarTeacher` 缺省 `clearance="ray"` 与旧决策逐步一致；`_spawn_for_episode` 缺省 `two_layer=False` 时因 Python `and` 短路，随机数 `rng` 消耗次数与出生位姿序列与改动前逐位一致；goal-stop 的 `stop_streak` 要求 `at_goal`（距 waypoint `<= stop_radius`）同时成立，而在 `at_goal` 内第 297 行先命中 `stop` 分支返回，不会在途中被 `wait_swept`（最多连续 2 步且不在 `stop_radius` 内）误触发提前结束。
  - S1b：`collision_model` 缺省 `"v1"` 时，`Occupancy.grid`、`collides`、`sweep`、`clearance`、`summary`（除新增 `"collision_model": "v1"` 外 `proxy_version` 串不变）与 `OccPlanner.hard` 均由 5 组金标 sha256 锁死不变。
- **规则 v2 与教师 `swept` 在纯自飞评估下的跨集状态重置**：
  - `RuleOraclePolicy.reset()` 完整清空 `memory`、`tabu`、`step_idx`、`last_turn`、`last_action`、`rows` 与 `entry_counts`；`driver.py` 每集新建 `RuleOraclePolicy` 与 `OracleRouteGuide` 实例，`run_episode` 集首还会调 `policy.reset()`，不存在跨集记忆或禁忌残留。
  - `OccAstarTeacher.reset()` 清零 `self._waits`；`driver.py` 虽在同一条路线的 `k` 集间复用同一个 `teacher` 实例，但 `run_episode:345` 每集开头均调用 `policy.reset()` 与 `policy.n_plan_fail = 0`，跨集无 `_waits` 残留；步内只要有一帧未走 `wait_swept` 分支，`__call__` 首行即把 `self._waits` 归零，非连续等待不会累加。
- **`_swept_blocked` 的坐标旋转与速度投影公式**：世界系水平速度 `(vx, vy)` 转动作前机体系 `(vbx, vby)`、线性预测 `(dbx, dby)` 再转回世界系 `(wx, wy)` 的正交变换符号与 ENU 逆时针 `yaw` 约定严格一致；`n <= 1e-6` 时退回机头单位向量无除零风险。
- **`OracleRouteGuide(nearest_in_lookahead=True)` 的单调性与插值**：搜索窗口从当前 `self.k` 起算且累计弧长不超过 `lookahead_m`，指针单调不减，不会倒退；剩余路径不足 `lookahead_m` 时返回 `waypoint`。
- **`Occupancy.sweep_path` 与 `_step_collision` 的切片边界**：`n_tr0 = len(dyn.trace)` 在 `dyn.step` 前记录，只取本步新增的 10 Hz `trace` 点，首点接 `(x0, y0, z0)`，末点若与 `(x, y, state["z"])` 不同则补齐，避免漏扫首段或末段。

---

## 5. 没有查的地方与原因，及验证记录

### 5.1 没有查的地方与原因
- **未起 GPU 渲染核实审核 B5（garden 深度异常的高斯不透明度过滤重渲）**：按任务书第 3 节硬性约束「不跑 3DGS 批量渲染」，且当前唯一显卡正跑训练池与闭环任务，未加载 1.45 GB garden PLY 上 GPU 重渲（未核实）。
- **未在 `cf_sitl` + gsplat 在线闭环中实跑 `collision_model="v2"` 或精确欧氏核下教师与规则 v2 的重跑成功率**：按任务书第 3 节约束不起仿真批量任务与不抢 `gs3d.lock`；报告中 `J-2` 的翻转集数是对已录轨迹的离线重判得分（教师与 `occ` 规则臂在线重跑时自身的规划栅格与列距离也会随新口径变化，重跑闭环数字由卡 `S1b` 守护进程产出，当前未核实）。
- **未查 C++ 训练推理内核（`src/`）、Gazebo 侧 `collect.py` / `eval_closed_loop.py`、`tools/` 统计与打包主体、PyTorch 原型线 `proto/`**：分别归属第二轮审核角度 K/N、I1/P、I2、L。
- **未跑 `tools/check.sh`**：按任务书第 3 节明确禁令不跑 `check.sh`、不拿任何池锁。

### 5.2 本轮执行的验证记录

| 检查项 / 命令 | 结果 | 证据文件 |
| --- | --- | --- |
| 全量 13 个 3DGS 场景 `scene.json` 顶层 `spawn.z` 对 `routes[*].spawn.z` 核查（`.venv-tools/bin/python` 只读解析） | 通过：10 个含路线场景共 273 条路线全部 `diff_count=0` | 会话只读命令输出（见 §1 疑点 ② 与 `J-9`） |
| `git merge-tree` / `git merge-file` 检查 `main`（含 S1a `5387747`）与 `p12-s1b`（`44595b0`/`835d2ad`）合并冲突 | `collect_3dgs.py` 自动合并 rc=0；`driver.py` 检出两处内容冲突 | `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-j/tmp/driver_merged.py` |
| `CUDA_VISIBLE_DEVICES="" nice -n 15 3dgs-venv/bin/python verify_all.py`（纯 CPU 单进程：4 项单元复现 + 5 组实验起飞过冲层带统计 + 5 室内与 2 室外场景 3×3 方核对精确欧氏回放 + 8100 点/30 点耗时基准） | `rc=0`：复现 `KeyError`、`AttributeError`、180° 割线封杀；算出过冲跨层率、回放翻转表与精确查询耗时 | `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-j/verify_all.log` |
| `CUDA_VISIBLE_DEVICES="" nice -n 15 3dgs-venv/bin/python verify_unit.py`（纯内存 < 1 s：复现 `DaggerPolicy + swept` 第 3 步误打 `FORWARD_SLOW`、`RuleOraclePolicy` 误解锁静止起步级、`StudentAdapter` 静默吞入 `heading-code` 模型） | `rc=0`：三项全部按预期复现 | `/root/rivermind-data/uav_rgb/tmp/phase12-cards/review2-j/verify_unit.log` |
