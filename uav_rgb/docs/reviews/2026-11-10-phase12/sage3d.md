# sage3d：SAGE-3D / InteriorGS 室内场景入库与 5 场景闭环试点（2026-11-10）

- 起止：20:02 起，时间盒 75 分钟（21:17 截止）。本文边做边写，21:05 前写完。
- 路径约定：`D=/root/rivermind-data/uav_rgb`，`WT=/root/claude-code-backup-20261028/code/wt/p12-sage3d`（分支 `p12-sage3d`，基于 main bd3a034），
  `S=$D/tmp/phase12-20261110/sage3d`（草稿），`PUB=$D/pubdata/sage3d`（下载物），试点输出 `$D/exp-sage3d-pilot-20261110/`。

## 0. 摘要

- 提交（分支 `p12-sage3d`，基于 main bd3a034，未 push）：`c1af32a`（`tools/` 两个入库工具与测试）、`9098598`（`experiments/2026-11-10-sage3d-pilot` 与参数化后共用的 bearing-guard 驱动 / 汇总）、
  `6182237`（`driver.py` 的可选键 `teacher_tol_deg`，诊断用）、`1c01078`（修 `run.sh` 里 `GATE_SR` 不能被环境变量覆盖的 bug，§4.7）。`check.sh --python-only` 在 `6182237` 上 rc=0（`1c01078` 只改了 `run.sh` 与 README 各一处，没重跑）。
- 飞行高度能按场景指定：`takeoff_height = spawn.z − ground_z`，`dynamics_profile` 不含场景与高度。导入器用 1.2 m，不改固件；真 `cf_sitl_proc` 起飞 1.2 m 的离地高度 min / p50 / max 为 1.179 / 1.192 / 1.236 m（§1）。
- 导入 6 个场景（试点 5 个，备用 1 个），都通过坐标系复核。通过过滤的路线数与高斯数：841150（393,895 个，14 条）、840865（762,677 个，40 条）、841012（583,028 个，36 条）、
  839992（807,699 个，123 条）、840139（1,388,583 个，34 条）；备用 841980（613,419 个，2 条，同起点，没进闭环）。新增磁盘约 1.1 GB（§3）。
- 零训练闭环：预注册闸门 P1（教师 SR(stop) >= 0.8）在 5 个场景上全部没过，合计 76/120 = 0.63（室外同一教师 0.83）。按预注册没跑其余 3 臂，P2、P3 没有数据，
  「室内对局部避障有没有检验力」这次没有回答（§4.4）。
- 原因查到教师自己：hold 1.0 下一个转角步 27–35°，教师航向容差 ±12°（死区宽 24°）更小，超时的 29 集里 25 集含左右交替原地打转；室外同一教师的 19 个超时集全含这种段（§4.5）。
  诊断实验（教师容差 18°，同 120 个路线与 seed，非预注册）：101/120 = 0.84，超时 29 → 0，4/5 个场景过 0.8，逐集配对 b=28、c=3，p = 4.6e-6；
  剩下的失败是碰撞（撞在前 19 集），说明这些室内路线对局部避障仍有难度（§4.6）。
- 需要主会话决定：放不放行阶段 2（我建议放行，命令与代价在 §4.7）；教师容差是否改默认值（共用代码，我没动）。


## 1. 飞行高度：能否按场景指定

结论：能按场景指定，1.2 m 不用改固件，也不换 `dynamics_profile`。证据三条。

1. 代码（读码，未改）：`sim/3dgs/dynamics.py:155-157` 里 `takeoff_height = scene["spawn"]["z"] − scene["ground_z"]`，要求落在 (0, 2.5) m；
   `reset()` 把它作为 `{"cmd":"takeoff","height":…}` 发给 `cf_sitl_proc`（同文件 236 行），执行器按这个目标高度做 ToF 定高；
   SITL 的地面是 z=0 平面，nav z = SITL z + `ground_z`（221 行）。
2. 指纹：`dynamics_profile` = `cf_sitl_v1:<e1_mini.json 各项 value 的 sha8>:lib<库版本>`，只取决于参数文件的值和库版本，不含场景与高度。
   `tools/test_scene_import_sage3d.py::TestFlightHeightIsPerScene` 用假的 `cf_sitl_proc` 锁定这一点：spawn 离地 1.2 m 与 2.0 m 的两个场景，
   `takeoff_height` 分别是 1.2 与 2.0，`params.profile()` 与 `manifest()["profile"]` 两次完全相同（形如 `cf_sitl_v1:xxxxxxxx:lib31`）；
   同一个类的另一个用例用导入器生成的 scene 验证 `takeoff_height == FLY_H_M`。
3. 实测（真 `cf_sitl_proc`，lib31，profile `cf_sitl_v1:1bc8fedf:lib31`，仅 CPU）。脚本 `$S/smoke_flight_height.py`，结果 `$S/smoke_flight_height.json`
   （`ground_z=−0.0004`，同一噪声种子）。起飞后盲前进 40 步、左转 6 步、前进 20 步，hold 1.0，共 66 步 660 个 trace 点：

   | 起飞高度 | 离地高度 min / p50 / max（m） | 标准差（m） | 最大 \|vz\|（m/s） | 最大横滚 / 俯仰（°） | 起飞用时（s） | 前 40 步行程（m） |
   |---|---|---|---|---|---|---|
   | 1.2 | 1.179 / 1.192 / 1.236 | 0.0062 | 0.036 | 1.15 / 3.97 | 6.28 | 11.97 |
   | 2.0（对照，现有场景的高度） | 1.977 / 1.992 / 2.035 | 0.0056 | 0.064 | 1.36 / 4.75 | 7.27 | 11.96 |

   两个高度都只出现 exec_state 3（ACTING），没有进 FAILSAFE。1.2 m 的稳态高度比目标低约 8 mm，稳定性不比 2.0 m 差。
   前进 11.97 m / 40 步 = 0.30 m 每步（含起步加速），比我先前估的 0.25 m 大，步数预算按 route_check 的 0.15 m 每 0.5 s 口径换算（见 §4）。

一致性：`z_slice`、`spawn.z`、渲染高度都来自 `tools/scene_import_sage3d.py` 配置区的 `FLY_H_M`（`build_scene` 同时写
`spawn.z = ground_z + FLY_H_M` 与 `z_slice = ground_z + FLY_H_M ± 0.2`，用例 `test_flight_height_consistency` 锁定；`step_render` 读 spawn.z）。
改高度只改这一个常数，再重跑 `align,scene,render` 三步。`terrain_from_scene` 要求 `z_slice` 下沿高于 0.3 m：1.2 m 下为 0.95 m，满足。

没有「最小必需改动」要写；剩一项没解决的差异：仿真里 ToF 是平地模型。飞行层（0.95–1.35 m）以下的家具（台面、床、沙发）在仿真里不存在，
真机飞过它们上方时 ToF 会读短、执行器会爬升。`route_check` 的 `agl_model_unverified` 只打标不改 ToF 输入；各场景被打标的路线数见 §3。

## 2. 入库的两个工具与测试

### 2.1 `tools/nurec_to_ply.py`

- 作用：SAGE-3D 的 usdz（zip）内的 `<id>.usdz.nurec`（gzip + msgpack，fp16 数组，键前缀 `.gaussians_nodes.gaussians.`）转成 INRIA 3DGS PLY
  （62 个 float32 属性，顺序与 `sim/3dgs/scene.py` 的 `PLY_PROPS` 相同）。
- 解释器：文件头注释写明 `/opt/conda/bin/python3`（msgpack 只在它里面）。msgpack 缺失时抛出带提示的错误，不往现有 venv 里装包。
- 要点：NuRec 的 `features_specular` 是 [N, 15 系数, 3 通道]，INRIA 的 `f_rest_{c*15+k}` 是通道在前，转换时做 `reshape(n,15,3).transpose(0,2,1)`。
  用它重转 841150，得到与 bench 转换同样的结果：393,895 个高斯，97,687,491 B，62 个属性（先前的转换产物与这次一致，字节数相同；逐字节 `cmp` 未做）。
- 测试 `tools/test_nurec_to_ply.py`（13 个用例，合成的极小 NuRec 字节）：字段顺序、f_rest 转置（值 `n*100+k*3+c`）、写出再读回、
  缺字段 / 字节数不符 / zip 里没有 .nurec 的报错、msgpack 缺失的提示。先红（`ModuleNotFoundError`，`$S/logs/test_nurec_red.log`）后绿。
  `.venv-tools` 的 python（无 msgpack）下 6 个依赖 msgpack 的用例打印 `SKIP` 后跳过，其余通过，退出码 0（`test_nurec_green_venvtools.log`）；
  `/opt/conda/bin/python3` 下 13 个全部运行并通过（`test_nurec_green_conda.log`）。变异检查：去掉 f_rest 转置，转置用例变红。

### 2.2 `tools/scene_import_sage3d.py`

- 步骤：`download → convert → align → scene → render`，一个场景一次（`python tools/scene_import_sage3d.py <场景号>`），每步落盘、产物在就跳过，
  `IMPORT_STEPS=align,scene` 选步骤。下载走 hf-mirror 的 `curl -L` 单线程，非 200 或字节数不符就删掉重来，第 n 次失败后等 n×20 s（429 退避），最多 6 次；
  usdz 超过 200 MB 直接停；转 PLY 后删自己下载的 usdz，删前写记录（文件名、字节数、时刻、PLY 字节数）。导入器自己的删除逻辑写 `pubdata/sage3d/usdz_deleted.jsonl`，
  这次 5 个 usdz 是 `$S/fetch_scenes.sh` 下载、转换并删的（5 个场景的 PLY 都在 `step_convert` 之前就已经存在，导入器的删除分支没有触发），记录在 `$S/usdz_deleted.jsonl`。
- 配置区（文件顶部）：`FLY_H_M=1.2`、`Z_SLICE_HALF_M=0.2`、`VOXEL_M=0.1`、相机 320×240 / 60°、`REACH_RADIUS_M=1.0`、`NAV_ROT_Z_DEG=180`、
  地面直方图参数、坐标系复核参数、路线过滤规则。
- 地面 `ground_z`：只留不透明度 ≥ 0.5、尺度 ≤ 3 m 的高斯，z 的 0.5–99.5 分位范围里「最低 40%」带内取最高的 0.05 m 桶，桶 ±1 桶内高斯 z 的中位数。
  限定在最低 40% 是为了天花板高斯更密时不会被当成地面（用例 `test_denser_ceiling_does_not_win`）。
- 坐标系复核（每个新场景都做，写 `align.json`）：沿用 bench `frame_test6.py` 的统计量。轨迹点（全部 VLN 点每 5 取 1，最多 2 万个）绕 z 转 0/90/180/270°，
  落进飞行层（`ground_z + 1.2 ± 0.1 m`）不透明高斯栅格（0.05 m，膨胀 0.15 m）的点占比，每个旋转再在 ±1.5 m 平移里找最小占比。
  通过条件：180° 的零平移占比最小，且 ≤ 0.7 × 第二名，且 180° 的最优平移 ≤ 0.5 m；不通过则 `step_align` 抛错停下，场景不入试点。
- 路线：起点 = SAGE 轨迹第一点，终点 = 最后一点（xy），朝向 = 朝路径上离起点 0.6 m 的点（几何朝向，JSON 的 `rotation` 是相机约定四元数，不用），
  z = `ground_z + 1.2`。过滤规则（配置区 `ROUTE_*`），逐条列出未过原因：
  ① A* 有路径且起终点不被占据；② 沿 A* 路径最小净空 ≥ 机体半径 0.102 m + 0.1 m（`route_check(margin_m=0.1)`，默认的 0.3 m 对室内走廊过严）；
  ③ A* 长度 ≤ 1.3 × SAGE 路径长；④（追加，任务书没有）起终点直线距离 ≥ 2.0 m，否则出生就在到达半径 1.0 m 的 2 倍之内，SR 恒为 1。
  通过的路线 `max_steps` 回填 `route_check.steps_estimate`（hold 0.5 口径）。
- 路线与指令的元数据字段（写在 `scene.json` 的 `routes[]` 里，每条路线一项；全部候选的过滤结果另存 `routes_check.json`，汇总存 `import_summary.json`）：

  ```
  routes[i] = {
    "name": "tr_2",                       # <划分缩写>_<轨迹号>：ti/ts/tt = test 的 Instruction/Scene/Trajectory-Unseen，tr = train
    "spawn": {"x","y","z","yaw_deg"}, "waypoint": {"x","y","z"}, "max_steps": int, "note": str,
    "sage": {
      "trajectory_id": int, "split": "train" | "test_Instruction_Unseen" | "test_Scene_Unseen" | "test_Trajectory_Unseen",
      "sage_path_m": float, "n_points": int,                      # SAGE 轨迹长度（米）与点数（间距 0.05 m）
      "astar_path_m": float, "turn_deg": float, "min_clearance_m": float,
      "agl_model_unverified": bool, "tof_terrain_m": float | null,  # ToF 平地模型对该路线不成立的打标
      "instructions": [ {"instruction_type": str, "start": str, "end": str, "text": str}, ... ]   # 该轨迹的全部指令，原顺序
    } }
  ```
  `instruction_type` 取 SAGE 原值（如 `Add_Object`、`Scenario_Driven`、`Attribute-based`、`Relative_Relationship`、`Add_Causality`），`start` / `end` 是物体实例名
  （如 `sofa_1`、`wall_64`），`text` 即 `generated_instruction` 原句。过滤掉的路线的指令不进 scene.json，只在 `routes_check.json` 里记条数。
- 测试 `tools/test_scene_import_sage3d.py`（32 个用例）：地面直方图 4 个、坐标系复核（合成 24 m 实心障碍场里挖 L 形走廊）3 个、VLN 解析与排序 4 个、
  过滤规则与原因标签 8 个、`route_check` 的余量与地形参数透传 1 个加真 `Occupancy` / `OccPlanner` 的 6 m 封闭房间 1 个、`scene.json` 构造与加载器往返 4 个、
  飞行高度写进 cf_sitl 起飞指令且 profile 不变（假 `cf_sitl_proc`）2 个、取值与统计辅助 2 个、下载 429 退避 3 个。先红
  （`ModuleNotFoundError`，`test_sage_import_red.log`）后绿（`test_sage_import_green2.log`）。
  变异检查（`$S/mutation_check_sage_import.sh`，结果 `$S/logs/mutation_check_sage_import.log`）：14 个变异里 13 个让用例变红；
  第 14 个（`m14`）改错了位置（改到了 `step_align` 的字段，那里没有单测，真数据跑过），补测试后改对位置的 `m14b` 变红。
  `reason_label` 与 `fly_height_m` 的用例是实现之后补的，不是先红后绿，用变异检查（`m12`、`m14b`）补验证。

## 3. 试点数据：5 个场景（另导入 1 个备用）

场景清单与来源（场景号 → 目录 `D/3dgs-scenes/sage3d_<id>/`，内含 `scene.ply`、`scene.json`、`align.json`、`routes_check.json`、`import_summary.json`；数据不进仓库）：

- 841150：bench 样本场景，PLY 用硬链接指向 bench 的转换产物（`pubdata/sage3d/sample/converted/841150.ply`，同文件、不占新空间），VLN 只有 train 划分。计作 1 个。
- 住宅 2 个：840865（test 划分 Trajectory-Unseen，有 2 条 test 轨迹）、841012（test 划分 Instruction-Unseen，5 条 test 轨迹）。
- 公共空间 2 个：839992（展陈厅，test Instruction-Unseen 5 条）、840139（娱乐休闲厅，test Trajectory-Unseen 4 条）。
- 备用 1 个：841980（住宅，test Scene-Unseen）。先选了它，导入后发现只有 5 条候选路线、通过 2 条、两条同起点，路线供给不够闭环用，改选 841012 补住宅名额；
  841980 的数据保留在盘上，不进闭环，它的数字照列在表里，作为「Scene-Unseen 场景没有 train JSON 可补」的证据。
- 选场景的依据：test 划分每场景只有 2–5 条轨迹（44 个有 test 数据的场景，共 204 条，`$S/scene_stats_test.json`），过滤后剩不下几条，所以同一场景的 train JSON 补进候选
  （`routes[].sage.split` 记来源）；Scene-Unseen 场景没有 train JSON，补不了。住宅 / 公共的归类依据是指令里的房间词与物体名，加渲染图人工看；
  InteriorGS 原库的房型标签没取（要点同意并给 token），所以归类未经官方标签核实。840139 渲染出来是带投影仪与沙发的影音厅，归公共空间的依据只有指令里的 party zone、guests、wine barrel，把握不大。

**表 3.1 体量与几何**

| 场景 | 类型（依据指令词与渲染图） | 高斯数 | PLY 字节 | usdz 字节（已删） | 地面 z / 天花板峰 z（m） | 地面积（m²） |
|---|---|---|---|---|---|---|
| 841150 | 住宅（厨房、餐厅区） | 393,895 | 97,687,491 | 46,488,397（bench 样本，未删） | 0.0 / 3.031 | 303.8 |
| 840865 | 住宅（客厅、餐厅、卫生间、卧室） | 762,677 | 189,145,427 | 90,007,987 | 0.001 / 2.5 | 103.2 |
| 841012 | 住宅（卧室、衣柜、餐桌） | 583,028 | 144,592,475 | 68,807,793 | 0.003 / 2.596 | 63.4 |
| 839992 | 公共（展陈厅：展柜、装饰画、轨道灯） | 807,699 | 200,310,883 | 95,320,986 | 0.0 / 1.925 | 144.8 |
| 840139 | 公共（娱乐休闲厅：影音沙发、酒桶、派对区） | 1,388,583 | 344,370,116 | 163,870,534 | 0.001 / 3.006 | 223.4 |
| 841980 | 住宅（浴室、卧室、客厅、娱乐区） | 613,419 | 152,129,443 | 72,394,206 | 0.0 / 2.404 | 116.7 |

**表 3.2 坐标系复核（飞行层栅格命中率，零平移；最后两列是 180° 对第二名的比值与 180° 的最优平移）**

| 场景 | 0° | 90° | 180° | 270° | 180° 对第二名 | 最优平移（m） | 判定 |
|---|---|---|---|---|---|---|---|
| 841150 | 0.240 | 0.247 | 0.106 | 0.283 | 0.44 | 0.05 | 通过 |
| 840865 | 0.289 | 0.513 | 0.007 | 0.261 | 0.03 | 0.05 | 通过 |
| 841012 | 0.280 | 0.224 | 0.016 | 0.163 | 0.10 | 0.05 | 通过 |
| 839992 | 0.066 | 0.111 | 0.020 | 0.119 | 0.31 | 0.158 | 通过 |
| 840139 | 0.054 | 0.068 | 0.030 | 0.080 | 0.57 | 0.0 | 通过 |
| 841980 | 0.444 | 0.459 | 0.000 | 0.290 | 0.00 | 0.0 | 通过 |

**表 3.3 路线供给与过滤**

| 场景 | 候选路线（划分：数） | 通过 | 通过中 test 划分 | 未过原因计数（一条路线可有多个原因） | 指令数（通过 / 全部候选） | agl 打标（通过路线中） |
|---|---|---|---|---|---|---|
| 841150 | 104（train 104） | 14 | 0 | 净空不足 55，A*无路径 33，A*长度比超限 23，终点被占据 14，起终点太近 6，起点被占据 2 | 273 / 1966 | 14 / 14 |
| 840865 | 100（t:Trajectory 2，train 98） | 40 | 0 | 净空不足 45，起终点太近 16，A*长度比超限 3，起点被占据 1，A*无路径 1，终点被占据 1 | 741 / 1815 | 27 / 40 |
| 841012 | 109（t:Instruction 5，train 104） | 36 | 3 | 净空不足 52，A*无路径 12，A*长度比超限 8，起终点太近 7，起点被占据 6，终点被占据 4 | 617 / 1929 | 29 / 36 |
| 839992 | 200（t:Instruction 5，train 195） | 123 | 4 | 净空不足 52，A*无路径 18，起点被占据 13，终点被占据 13，起终点太近 8，A*长度比超限 7 | 2176 / 3548 | 29 / 123 |
| 840139 | 158（t:Trajectory 4，train 154） | 34 | 1 | 净空不足 83，A*无路径 34，终点被占据 27，A*长度比超限 11，起点被占据 8，起终点太近 7 | 612 / 2878 | 23 / 34 |
| 841980 | 5（t:Scene 5） | 2 | 2 | 净空不足 3 | 10 / 25 | 0 / 2 |

**表 3.4 通过路线的长度、净空与步数分布（p10 / p50 / p90 / max）**

| 场景 | SAGE 路径（m） | A* 路径（m） | 最小净空（m） | steps_estimate（hold 0.5） | 全部候选的 SAGE 路径（m） |
|---|---|---|---|---|---|
| 841150 | 3.65 / 8.13 / 10.26 / 10.89 | 3.83 / 8.45 / 9.65 / 10.20 | 0.25 / 0.40 / 0.60 / 0.60 | 48.70 / 93.00 / 105.00 / 114.00 | 5.66 / 9.18 / 16.30 / 21.64 |
| 840865 | 4.09 / 6.57 / 8.81 / 12.71 | 3.98 / 5.74 / 8.33 / 8.95 | 0.22 / 0.32 / 0.54 / 0.60 | 47.90 / 66.00 / 86.10 / 95.00 | 1.78 / 6.57 / 10.08 / 14.80 |
| 841012 | 3.36 / 8.45 / 10.88 / 13.38 | 2.96 / 6.07 / 11.27 / 15.02 | 0.22 / 0.40 / 0.50 / 0.72 | 42.00 / 68.00 / 122.00 / 162.00 | 3.15 / 7.32 / 13.88 / 19.58 |
| 839992 | 3.66 / 9.41 / 15.04 / 18.02 | 3.72 / 9.87 / 14.12 / 16.35 | 0.22 / 0.32 / 0.60 / 0.78 | 47.00 / 106.00 / 143.80 / 162.00 | 3.09 / 8.53 / 14.23 / 18.02 |
| 840139 | 2.77 / 3.56 / 15.01 / 22.24 | 2.57 / 3.44 / 15.08 / 22.39 | 0.30 / 0.50 / 1.20 / 1.20 | 35.30 / 42.00 / 153.80 / 218.00 | 2.91 / 15.16 / 21.48 / 34.15 |
| 841980 | 8.94 / 9.82 / 10.69 / 10.91 | 9.76 / 10.65 / 11.55 / 11.77 | 0.30 / 0.30 / 0.30 / 0.30 | 108.60 / 115.00 / 121.40 / 123.00 | 6.11 / 9.98 / 14.88 / 17.52 |

**起点视角图**（每场景按 A* 长度分位取 4 条路线，2×2 拼图；单张在同目录）

- 841150：/root/rivermind-data/uav_rgb/tmp/phase12-20261110/sage3d/import/sage3d_841150/render/sage3d_841150_starts.png（路线 tr_88, tr_33, tr_2, tr_93）
- 840865：/root/rivermind-data/uav_rgb/tmp/phase12-20261110/sage3d/import/sage3d_840865/render/sage3d_840865_starts.png（路线 tr_81, tr_83, tr_27, tr_60）
- 841012：/root/rivermind-data/uav_rgb/tmp/phase12-20261110/sage3d/import/sage3d_841012/render/sage3d_841012_starts.png（路线 tr_31, tr_100, tr_17, tr_55）
- 839992：/root/rivermind-data/uav_rgb/tmp/phase12-20261110/sage3d/import/sage3d_839992/render/sage3d_839992_starts.png（路线 tr_126, tr_165, tr_5, tr_12）
- 840139：/root/rivermind-data/uav_rgb/tmp/phase12-20261110/sage3d/import/sage3d_840139/render/sage3d_840139_starts.png（路线 tr_96, tr_74, tr_21, tr_16）
- 841980：/root/rivermind-data/uav_rgb/tmp/phase12-20261110/sage3d/import/sage3d_841980/render/sage3d_841980_starts.png（路线 ts_3, ts_2）

读表要点：
- 6 个场景全部通过坐标系复核（180° 的零平移命中率都是最小，且不高于第二名的 0.7 倍，最优平移不超过 0.5 m），没有场景因此被剔除。
  区分度最弱的是 840139（0.57，宽敞稀疏的公共空间，飞行层障碍少）与 841150（0.44）。
- 过滤的主因是净空：1.2 m 飞行层上，墙边、门框与家具腿让沿路最小净空低于 0.202 m（机体半径 0.102 m + 0.1 m）。
  通过率 13%（841150）到 62%（839992）。840139 的 34 条通过路线里 SAGE 路径 p50 只有 3.56 m、p90 15.01 m，是两头多、中间少。
- 通过路线的 A* 长度 p50 为 3.4 到 10.7 m，最长 22.4 m（840139）。按步数预算换算后，约 18 m 以上的路线超过 100 步封顶（见 §4 的「需满电档」计数）。
- agl 打标：通过路线里被打标的比例是 0/2（841980）到 100%（841150 的 14/14）。打标意味着 A* 路径在飞行层下方 0.3–0.95 m 有占据（台面、床、沙发、墙根低柜），
  ToF 平地模型对这些位置不成立（§1）。
- 渲染：6 个场景的起点视角图颜色与朝向正常（每场景的拼图都人工看过），没有黑屏、倒置或大面积漂浮物。841980 的两条路线同起点，两张图相同；841012 的左下一张起点正对落地窗，画面是天空。

## 4. 试点闭环：预注册、守护进程、已出数字

### 4.1 设计（全文在 `experiments/2026-11-10-sage3d-pilot/run.sh` 文件头，提交 9098598）

不训练、不起 Gazebo。`driver.py` / `summarize.py` 是 bearing-guard 实验的那两个文件，参数化后共用（没有复制）：config.json 里 `routes` 可以是
`{场景: [路线下标]}` 字典、`step_cap` 封顶步数、`gates` / `arm_checks` / `main_pairs` 等；不给这些键时输出不变。
在 bearing-guard 的 120 集输出上，改前改后的 `summarize.py` 用 `3dgs-venv` 的 python 跑出的 `summary.md` 与 `summary.json` 逐字节相同（md5 对过，
与原目录里的 `summary.md` / `summary.json` 也相同；用系统 `python3` 跑时 `spl_stop` 的末位有 1e-16 量级差别，改前改后一致，来自解释器的浮点求和）。

| 项 | 取值 |
|---|---|
| 场景 | `sage3d_841150`、`840865`、`841012`、`840139`、`839992`，顺序跑，同一时刻只有 1 个 3DGS 驱动进程 |
| 路线 | `select_routes.py`：每场景最多 8 条，按 A* 长度取等间隔分位（含最短与最长）；真实数据选出 40 条：`<5 m` 14 条、`5–10 m` 17 条、`>=10 m` 9 条 |
| 集数 | 每路线 3 集，每臂 n = 120；seed = 8000 + 100·路线下标 + k；出生扰动 ±0.2 m / ±10°（室外是 ±0.5 m，室内净空 0.2–0.4 m 会抖进墙里）；同 seed 各臂出生位姿相同，逐集配对 |
| 臂 | `teacher`（占据 A* 教师）、`forward`（盲前进）、`forward+route`（盲前进 + oracle 路线方位护栏，阈值 45°，前视 1.5 m）、`calswa-s1+route`（学生权重 `exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin` + 同一护栏） |
| 阶段 | 阶段 1 只跑 `teacher`；`summarize.py --gate teacher:0.8` 按场景选出 SR(stop) >= 0.8 的场景，只有它们进阶段 2（其余 3 臂）；全不过则 DONE `rc=4`，不跑阶段 2 |
| 其余 | hold 1.0、`stop_on_collision`、cf_sitl（lib31，`CF_SITL_PROC=…/build-cf-sitl-lib31/cf_sitl_proc`，profile `cf_sitl_v1:1bc8fedf:lib31`）、推理二进制 `bin-snapshots/main-97ae04b/uavrgb_infer`；指标同 bearing-guard（`3dgs.pilot.v2+stop.v1`），另加按路线长度分桶与每条路线单列 |

护栏参数起跑前定死，没有在这批路线上调：阈值 45°（hold 1.0 一步转角约 31.6°，容许偏 1 步、偏 2 步就纠正，与室外同）；前视 1.5 m
（室外 3.0 m 放进室内会越过门框与拐角，护栏方向穿墙；约 5 步，仍是粗方向，不是路径跟踪）。

**预注册**（写于 20:27，冒烟与起跑之前，护栏参数、阈值、臂的清单、选路线规则已定；20:45 只改了「问题」段里的净空数字与场景名，预测与阈值没动）：

- P1（闸门）教师 SR(stop) >= 0.8，用过滤后的路线，各场景分别判。证伪：< 0.8，先查占据代理 / 坐标系 / 飞行高度，不跑其余臂。
- P2 `forward+route` 的 SR(stop) 低于室外的 0.89（107/120，`exp-bearing-guard-3dgs-20261110/summary.md` 第 1 节），预测 < 0.7（净空 0.2–0.4 m，45° 容差不足以避开墙 / 门框）。
  证伪：>= 0.89，说明室内对局部避障没有检验力。判决：过闸门场景合计的点估计 < 0.7 记支持，>= 0.89 记证伪，之间记「介于预测与证伪之间」，同时列 Wilson 95% 区间。
- P3 `calswa-s1+route` 对 `forward+route`：预测没有可检出的差别（逐集配对精确 McNemar p > 0.05），与室外一致（室外 b=1、c=3、p=0.625）。
  证伪：学生显著好于或差于盲飞（p <= 0.05）。
- 补充比较（不进判决）：`forward+route` 对 `forward`、`teacher` 对 `forward+route`、`teacher` 对 `calswa-s1+route`。

净空前提的核实（选路线之后、起跑之前）：选中的 40 条路线沿 A* 的最小净空（`scene.json` 的 `sage.min_clearance_m`）最小 0.22 m、p10 0.28 m、p50 0.40 m、p90 0.60 m、最大 1.2 m，
任务书的 0.2–0.4 m 与低端相符，中位数在上沿。

### 4.2 步数预算换算与「需满电档」

`route_check.steps_estimate = ceil((A* 长度 / 0.15 + 转角 / 17 + 8) × 1.25)`，口径是 hold 0.5（每步前进 0.15 m、转 17°），写进 `scene.json` 的 `max_steps`。
本试点 hold 1.0：实测每步前进约 0.30 m（`smoke_flight_height.json`：40 步 11.97 m；bearing-guard 的 forward 臂 `actions.jsonl` 里 `action_executed == 1` 的步，`displacement_m` 均值 0.298 m、n=4790，我复算的；`run.sh` 头注释写的 n=2477 是 subagent 的算法，均值相同，n 的口径没查），
步数换算同室外：`steps = ceil(max_steps × 0.5) + 15`，封顶 100（SITL 起始 50% 电量，约 100 步的电量限）。封顶对应 `max_steps > 170`，
直线路线约 19.2 m 以上，带转弯时更短。超过封顶的路线标「需满电档」，不进选取池、不跑，单列在 `routes_selected.json` 的 `need_full_battery`。

| 场景 | 过滤后路线 | 需满电档 | 进池 | 选中 | 桶 `<5` / `5–10` / `>=10`（m，A* 长度） |
|---|---|---|---|---|---|
| 841150 | 14 | 0 | 14 | 8 | 1 / 6 / 1 |
| 840865 | 40 | 0 | 40 | 8 | 3 / 5 / 0 |
| 841012 | 36 | 0 | 36 | 8 | 2 / 4 / 2 |
| 840139 | 34 | 2（tr_3：A* 21.65 m，未封顶 118 步；tr_16：A* 22.39 m，未封顶 124 步） | 32 | 8 | 6 / 0 / 2 |
| 839992 | 123 | 0 | 123 | 8 | 2 / 2 / 4 |
| 合计 | 247 | 2 | 245 | 40 | 14 / 17 / 9 |

选中路线的步数预算（封顶后）最小 30、中位 51、最大 96；A* 最长的两条选中路线（839992 tr_12，16.35 m；840139 tr_15，15.90 m）的预算都是 96 步，
在封顶之内，不是靠封顶截出来的。5 个场景里需要满电档的路线只有 840139 的 2 条，所以 100 步封顶只影响本次路线选择里的 2 条；
对任务卡 B 的多段长路线影响大（§6）。

### 4.3 守护进程

- 起跑：20:43:26，`PGID=1464051`，git HEAD `9098598`（工作树干净），输出 `/root/rivermind-data/uav_rgb/exp-sage3d-pilot-20261110/`（`$OUT`）。
- 看进度：`tail -f $OUT/run.log`；每场景日志 `grep -v '^step ' $OUT/driver-sage3d_841150.log | tail`；闸门 `cat $OUT/gate.json`（阶段 1 结束后才有）。
- 汇总（只读每集 `metrics.json`，随时可跑）：`/root/rivermind-data/uav_rgb/3dgs-venv/bin/python /root/claude-code-backup-20261028/code/wt/p12-sage3d/experiments/2026-11-10-bearing-guard-3dgs/summarize.py $OUT`，
  写 `$OUT/summary.md` 与 `summary.json`；结束时 `run.sh` 自己会跑一次，并写 `$OUT/DONE`（`rc=… 时刻 闸门通过 […] 未过 […]`）。
- 停：`kill -- -1464051`（只按 PGID，不用 `pkill -f`）。续跑：在 `experiments/2026-11-10-sage3d-pilot/` 下重跑同一条命令
  `setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &`，评完的路线跳过，半成品改名 `*.incomplete-<时刻>` 后重跑；`.lock` 保证重复启动的立即退出。
- 冒烟（起跑前）：场景 841150、最短 1 条路线（tr_88，A* 2.5 m）、1 集、4 臂全跑，输出 `$S/smoke/`，端到端 25 s（含两次场景加载各 7 s）：4 臂都到点；
  `forward` 到点后继续盲飞撞墙（SR(stop)=1，SR(v2)=0，与室外同口径）；`forward+route` 与 `calswa-s1+route` 的出生位姿逐位相同（首步位姿最大差 0）。
  每集墙钟 1–2 s（这条路线短，长路线按步数线性放大）。

### 4.4 正式实验的结果：P1 闸门五个场景全部没过，没有跑阶段 2

起止 20:43:26–20:52:00，`DONE` 内容 `rc=4`（所有场景都没过闸门），共 8 分 34 秒（20:59 有一次非我所起的重启，把 `DONE` 的时刻改成了 21:00:10，内容相同，见 §4.7）；阶段 1 的 120 集（5 场景 × 8 路线 × 3 集）失败集 0，`physics_verified` 全部为真。
原始结果：`$OUT/summary.md`、`summary.json`、`gate.json`（我另存了一份 `gate_prereg.json`，防止之后放行重跑时被覆盖）。

| 场景 | 教师 SR(stop) | 碰撞（含到点后） | 超时 | 超时集里含左右交替段 | 闸门（>= 0.8） |
|---|---|---|---|---|---|
| 841150 | 12/24 = 0.50 | 2 | 10 | 9 / 10 | 未过 |
| 840865 | 16/24 = 0.67 | 8 | 3 | 2 / 3 | 未过 |
| 841012 | 17/24 = 0.71 | 3 | 4 | 4 / 4 | 未过 |
| 840139 | 17/24 = 0.71 | 2 | 5 | 5 / 5 | 未过 |
| 839992 | 14/24 = 0.58 | 3 | 7 | 5 / 7 | 未过 |
| 合计 | 76/120 = 0.63 [Wilson 0.54, 0.71] | 18（撞在前 15） | 29 | 25 / 29 | |

合计行其余指标：SR(v2) 73/120，SPL(stop) 0.52，平均步数 45。室外同一个教师（bearing-guard，4 场景 × 6 路线 × 5 集）是 100/120 = 0.83。
按 A* 长度分桶的教师 SR(stop)：`<5 m` 38/42 = 0.90，`5–10 m` 28/51 = 0.55，`>=10 m` 10/27 = 0.37。

预注册判决：P1 在五个场景上都被证伪（教师 < 0.8）。按预注册，没跑 `forward` / `forward+route` / `calswa-s1+route`，所以 P2、P3 没有数据，
「室内对局部避障有没有检验力」这个问题这次没有回答。

### 4.5 P1 没过之后查了什么（占据代理、坐标系、飞行高度，以及教师自己）

1. 坐标系：6 个场景的统计复核都通过（§3 表 3.2）。闭环里教师沿 A* 走通了 76 集，若坐标系是错的，A* 路径会穿墙，不会有这个到点率。
2. 飞行高度：`spawn.z = ground_z + 1.2`，真 `cf_sitl_proc` 起飞到 1.2 m 的实测见 §1；120 集的 `physics_verified` 全为真，没有 FAILSAFE 导致的失败集。
3. 占据代理：失败里碰撞只有 18 集（撞在前 15 集），其中 840865 一个场景占 8 集；主要的失败是超时 29 集。29 个超时里只有 2 集最终停在离 waypoint 2 m 以内，
   中位的「最小到 waypoint 距离」是 3.05 m，也就是教师停在远处，没有差一点到。
4. 失败形态（`$S/teacher_osc.py`，只读 `actions.jsonl` 与 `metrics.json`，结果 `$S/logs/teacher_osc_main.txt`）：29 个超时集里 25 集含「左转 / 右转严格交替、连续 6 步以上」的段，
   段内水平位移最大 0.21 m（在原地转），其中 8 集这样的段占了一半以上的步数。典型的是 841150 的 tr_16（`r1`）：3 集全部超时，交替段占步数的 100% / 45% / 37%，段内位移最大 0.03 / 0.04 / 0.16 m；第 0 集 62 步全是 `2,3,2,3,…`。
   室外 bearing-guard 里同一个教师臂：19 个超时集 19 个含交替段，段内位移最大 0.27 m；到点的 100 集里也有 30 集含交替段（短暂的）。
5. 机理（读码加实测）：`sim/3dgs/planner.py` 的 `OccAstarTeacher` 在 `|航向误差| > YAW_TOL_DEG = 12°` 时转向，否则前向 0.4 m 的硬栅格 sweep 命中时按误差符号转。
   hold 1.0 下一个转角步实测 27–35°（室内教师：YAW_LEFT n=817，均值 29.3°、中位 27.7°；YAW_RIGHT n=838，均值 −29.8°；室外同量级）。
   死区宽 24° 小于一个转角步，误差落在 12°–16° 左右时转一步就过冲到另一侧的 12° 之外，再转回来，形成极限环。
   「前向被堵时按误差符号转」是另一条可能的来源，两条各占多少没有拆开（未核实）。
6. 与路线长度的关系：路线越长教师越差（0.90 → 0.55 → 0.37），与「长路线转弯多、交替的机会多、步数预算耗在原地」相符；这条也没有单独验证。

小结（只写查到的）：P1 没过的直接原因在教师，不在占据代理、坐标系、飞行高度这三项里，证据是 4 与 5；这不等于室内环境没有别的问题，
也不能由此推出 `forward+route` 在室内的 SR（它不用教师的转向逻辑，用的是 oracle 路线方位护栏）。

### 4.6 诊断实验：教师航向容差 18°（非预注册）

目的：验证 4.5 的机理。同一批 120 个（路线, seed），只改教师的航向容差（`teacher_tol_deg = 18°`，大于最大转角步 34.6° 的一半 17.3°），其余逐位不变，只跑 `teacher` 臂。
这不是预注册的 P1，是对 P1 未过原因的追查；结果不改变 4.4 的判决，也不替代它。
实现：`driver.py` 加可选 config 键 `teacher_tol_deg`（提交 6182237，缺省不传，行为不变，测试 +2 项，先红后绿）；脚本 `$S/diag_teacher_tol.sh`；
输出 `$D/exp-sage3d-pilot-20261110-diag-teachertol18/`；守护进程 `PGID=1495044`，20:52:44 起。

结果（20:52:44–20:59:00，6 分 16 秒，`DONE rc=0`，120 集失败集 0；表由 `$S/diag_table.py` 生成，`$S/logs/diag_table.txt`）：

| 场景 | 12°：SR(stop) | 碰撞（撞在前） | 超时 | 18°：SR(stop) | 碰撞（撞在前） | 超时 | 18° 下闸门（>= 0.8） |
|---|---|---|---|---|---|---|---|
| 841150 | 12/24 = 0.50 | 2（2） | 10 | 23/24 = 0.96 | 1（1） | 0 | 通过 |
| 840865 | 16/24 = 0.67 | 8（5） | 3 | 20/24 = 0.83 | 7（4） | 0 | 通过 |
| 841012 | 17/24 = 0.71 | 3（3） | 4 | 17/24 = 0.71 | 7（7） | 0 | 未过 |
| 840139 | 17/24 = 0.71 | 2（2） | 5 | 21/24 = 0.88 | 3（3） | 0 | 通过 |
| 839992 | 14/24 = 0.58 | 3（3） | 7 | 20/24 = 0.83 | 4（4） | 0 | 通过 |
| 合计 | 76/120 = 0.63 [0.54, 0.71] | 18（15） | 29 | 101/120 = 0.84 [0.77, 0.90] | 22（19） | 0 | 4 / 5 个场景通过 |

- 逐集配对（同场景同路线同 seed）：18° 成而 12° 败 b = 28，12° 成而 18° 败 c = 3，精确 McNemar p = 4.6e-6。
- 超时 29 → 0；左右交替段：超时集里 25/29 → 0 个超时集，到点的集里含交替段的从 20 集降到 2 集（`$S/logs/teacher_osc_diag18.txt`）。这与 4.5 的机理一致；它是单次对照，只改了容差，没有分别关掉两条来源，不能说明各自的贡献。
- 按 A* 长度分桶，12° → 18°：`<5 m` 38/42 → 41/42，`5–10 m` 28/51 → 42/51，`>=10 m` 10/27 → 18/27。
- 去掉打转之后，剩下的失败是碰撞：撞在前 19 集（12° 下 15 集），撞时离 waypoint 2.3–6.7 m；841012 一个场景 7/24 集撞在前（12° 下 3 集，原来的 4 个超时变成了碰撞，SR 没变，仍是 0.71）。
  也就是说，去掉教师自己的毛病之后，这些室内路线对「局部避障」仍有难度：A* 教师在 hold 1.0（每步约 0.30 m）下也有 16% 的集（19/120）撞在到点之前。
  这是教师的数字，不是 `forward+route` 的数字；P2 没有数据（§4.4）。
- 对 P1 的含义：事后换成容差 18° 的教师，4/5 个场景过 0.8 的线，841012 没过（0.71）。这不改变预注册的判决（P1 在 12° 下 5/5 个场景被证伪）。
- 对室外已有结果的含义（推断，未核实）：bearing-guard 的教师臂是同一个容差、同一个 hold（1.0），它的 0.83 里含有这种打转；室外 `forward+route` 的 0.89 高于教师的 0.83，
  这个打转可能是原因之一；教师在 hold 1.0 下不是上限。采集脚本的示例与缺省 hold 是 0.5（`collect_3dgs.py` 19 行与 592 行；`route_check` 的口径是每步转 17°，小于 24° 的死区宽度），
  不会出现这个极限环；采集数据用的 hold 我没有逐个核实。

### 4.7 下一步与放行命令（需要主会话决定）

P2 与 P3 的对象是 `forward+route` 与 `calswa-s1+route`，它们不走教师的转向逻辑，用的是 oracle 路线方位护栏；P1 的失败原因在教师（§4.5），所以我建议放行阶段 2，
同时把 P1 的原判决保留在 `gate_prereg.json` 与本报告 §4.4。放行是改动预注册流程的决定，我没有擅自做。

放行命令（在正式实验目录里续跑；教师臂 120 集已有 `done.json`，不重跑，保留的是容差 12° 的结果）：

```bash
cd /root/claude-code-backup-20261028/code/wt/p12-sage3d/experiments/2026-11-10-sage3d-pilot
GATE_SR=0 setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 看进度 / 汇总 / 停：同 §4.3（PGID 在 $OUT/run.pid）
```

会发生什么：`run.sh` 重写 `config.json`、`inputs.txt`（git HEAD 变成 `6182237`）、`gate.json`（闸门阈值写成 0，所有场景「通过」），追加 `run.log`，然后跑 3 个臂 × 5 场景 × 24 集 = 360 集。
耗时估计 20–25 分钟（未验证：教师的 120 集用了 8.5 分钟，盲前进类的臂到点后不停、会飞满预算，每集更长）。`summary.md` 的闸门一节会写「SR(stop) ≥ 0」，原判决看 `gate_prereg.json`。
这样跑出的 P2、P3 里，教师臂仍是容差 12° 的结果，只作参照。

**20:59 的一次重启与 `run.sh` 的 bug**：20:59:21 有人（不是我）在同一个 `$OUT` 里重启了 `run.sh`（PGID 1520727，进程环境里 `GATE_SR=0`，git HEAD `6182237`），
我是从 `run.log` 的第二条「开始」行和进程环境查到的，没有查到是谁起的，推测是主会话。这次重启在 21:00:10 又以 `rc=4` 结束：`run.sh` 的配置区写的是
`GATE_ARM=teacher; GATE_SR=0.8` 这样的硬赋值，环境变量覆盖无效（README 与文件头却写着可以覆盖；我的冒烟里教师 1/1 到点，闸门本来就过，所以没发现）。
这是我写的 bug，已修（提交 `1c01078`：`GATE_SR=${GATE_SR:-0.8}`，默认值不变），用冒烟验证了覆盖值会被读到（`GATE_SR=1.1` 时 `config.json` 的阈值是 1.1，闸门未过，`DONE rc=4`）。
这次重启的副作用：`config.json`、`inputs.txt`（git HEAD 从 `9098598` 变成 `6182237`）、`run.sh.snapshot`、`gate.json`、`summary.md`、`DONE`（时刻变成 21:00:10）被重写，`run.log` 追加；
闸门阈值仍是 0.8，教师臂的 120 集因 `done.json` 已完成而跳过，没有新的 episode，也没有碰任何结果文件。第一次起跑的输入记录只剩 `run.log` 第一行（`WT=9098598`）与 `gate_prereg.json`。
修好之后，上面的放行命令才真的有效。注意版本：main 已在 `938cae9` 合并了 `p12-sage3d` 的 `6182237`（我最后用 `git log main` 才看到；这也说明主会话当时在同时处理这个分支，重启大概率是它），
`1c01078` 还没进 main，main 检出里的 `run.sh` 仍是有 bug 的版本；放行续跑要用 `wt/p12-sage3d` 里已修的那份（上面命令里的路径就是），或者先把 `1c01078` 合进 main。

另一条路是把容差 18° 当作修订后的教师，重新定 P1 再跑（诊断实验在 §4.6 里已经跑了教师臂）：`run.sh` 现在没有传 `teacher_tol_deg` 的开关，要先加；这也是事后改仪器，需要同样由主会话批准。

## 5. 许可提醒

- 场景：`spatialverse/SAGE-3D_InteriorGS_usdz` 为 CC-BY-NC-4.0；InteriorGS 原库的使用条款是非商业科研教学、不得再分发（来源：`reports/bench.md` 第 16、27 行）。
  路线与指令 `spatialverse/SAGE-3D_VLN_Data` 为 CC-BY-4.0。
- 约束：只用于研究。本次产物（`scene.ply`、`scene.json`、渲染图、之后的教师轨迹与数据集）不再分发、不进仓库、不进任何公开发布物；仓库里只有脚本与测试。
  `scene.json` 的 `license` 字段已写入上面的许可文本，渲染图与 `import_summary.json` 不带许可字段，使用时要自己注明。
- 署名：引用时写 InteriorGS（SpatialVerse）与 SAGE-3D（arXiv:2510.21307）。
- 未核实：用这些数据训练出的模型权重算不算「再分发」，条款里没有明说；发布权重或把它们放进对外演示之前，需要用户确认。
- 本次取数：只取 hf-mirror 上 `gated=False` 可直接下载的文件，没有用 token，没有碰原版 InteriorGS 的 occupancy / structure / labels（那些要点同意并给 token）。
  usdz 转 PLY 后已删除（只删了自己下载的 5 个，共 490,401,506 B；841150 的 usdz 是 bench 的样本，没动），文件名与字节数见 §3 表 3.1 与 `$S/usdz_deleted.jsonl`。

## 6. 后续任务卡

### 任务卡 A：20 场景完整试点

目标：把导入从 5 个场景扩到 20 个（住宅与公共空间各 10，test 划分的三个子集都有），得到 20 个场景的过滤后路线集、起点视角图与零训练闭环基线，
为之后的训练数据采集（教师轨迹 + 指令）定场景清单。

| 文件 | 内容 |
|---|---|
| `tools/sage3d_pick_scenes.py`（新） | 读 bench 的两棵目录树（`bench/meta/spatialverse__SAGE-3D_*.tree.json`）与各场景候选路线统计，选场景。`classify_scene(instructions)`：按指令里的房间词与物体名分住宅 / 公共（规则写在文件顶部配置区，可改）；`score_scene(stats)`：路线长度 p50、通过率、地面积、是否有 test 划分，排序；`write_list(path)`：写 `tools/sage3d_scene_list.json`（场景号、类型、划分、依据）。 |
| `tools/scene_import_sage3d.py` | 加 `run_batch(ids)`：读清单逐场景跑五步，一个场景失败（含坐标系复核未过）记入 `failed.json` 不挡后面；下载总量与磁盘增量超限时停。 |
| `experiments/<日期>-sage3d-pilot20/{run.sh,README.md}` | 用本次的 `driver.py` / `summarize.py` 参数化版本，场景换成 20 个，臂与统计同本次。 |
| `tools/test_scene_import_sage3d.py` | 加 `classify_scene`、批处理失败隔离、磁盘预算停止的用例（先红后绿）。 |

验收：
1. 20 个场景里至少 15 个通过坐标系复核（180° 命中率最小、不高于第二名的 0.7 倍、最优平移不超过 0.5 m）；未过的写明数字。
2. 入选场景各有至少 8 条过滤后路线，合计至少 150 条；试点里 test 划分的场景给出通过率（本次 841980 只有 5 条候选、通过 2 条，说明 Scene-Unseen 场景没有 train JSON 可补，
   这类场景要么只当评测、要么另生成路线）。
3. 教师在过滤后路线上的 SR(stop) 不低于 0.8（闸门，同本次预注册 P1）。本次五个场景在容差 12° 下全部没过（合计 0.63，§4.4），原因是教师在 hold 1.0 下的转向极限环（§4.5）；扩到 20 场景之前要先定教师的航向容差（§4.6 的诊断），否则闸门会再被教师自己挡住。
4. 磁盘新增不超过 8 GB（每场景 PLY 约为 usdz 的 2.1 倍，usdz 中位 96.0 MB，bench 数字）；单场景导入墙钟不超过 5 分钟（本次 6 个场景的实测见 §3）。
5. 过滤原因计数、路线长度与净空分布、指令数逐场景入表。

预估：下载约 2 GB、PLY 约 4.2 GB；导入 20 × 1.5–3 分钟；零训练闭环按本次每集耗时线性外推（§4）。
风险：公共空间场景里的玻璃、镜面与密集小物件会让 1.2 m 飞行层的高斯占据代理出现假障碍，通过率可能低于住宅；
全部通过路线都带 `agl_model_unverified`（§3），真机迁移前要处理飞越家具时的 ToF 读数。

### 任务卡 B：多段（3–5 段）路线拼接

目标：把 SAGE 的单段路线（中位约 9 m）串成 3–5 段的长程任务，总长 30–60 m，指令为多句。

方案：同一场景内，段 i 的终点与段 i+1 的起点相距不超过 `REACH_RADIUS_M`（1.0 m），或段 i 的 `end` 物体名等于段 i+1 的 `start` 物体名（指令字段里有），连成链；
链上每一段都是已通过 §2.2 过滤的路线；衔接处再用 A* 验一次连通与净空；避免环路与重复经过同一段。

| 文件 | 内容 |
|---|---|
| `tools/scene_stitch_sage3d.py`（新） | `build_graph(routes)`：节点 = 物体名或位置簇，边 = 路线；`stitch_chains(routes, n_seg, min_total_m, max_total_m)`：返回链；`write_multi_routes(scene_json, chains)`：写 `scene["multi_routes"]`，每条含 `segments[]`（各段 spawn / waypoint / 名称 / 指令）、总步数预算与是否需满电档。 |
| `sim/3dgs/multiseg.py`（新） | 多段 episode 驱动：到达段 i 的 waypoint 后切到段 i+1 的目标，不重置位姿；指标 `segments_completed`、`SR_all_segments`、`SPL_total`；预算按各段 `max_steps` 之和换算（hold 1.0 口径，封顶 100 步，超过的标需满电档）。 |
| `tools/test_scene_stitch_sage3d.py`、`sim/3dgs/test_multiseg.py`（新） | 合成路线图：链连通、长度窗口、环路规避、同物体名衔接；段切换不重置、到点序列、预算换算。先红后绿。 |

验收：
1. 5 个场景合计至少 30 条 3–5 段链，每段衔接点 A* 可达且净空不低于 0.2 m。
2. 至少 3 条链 × 1 集的教师闭环冒烟，每段切换正确（`segments_completed` 与 trace 对得上）。
3. 总步数（hold 1.0）不超过 100 的链标「半电可跑」，其余标「满电档」，两类各自计数入报告。

依赖：任务卡 A 的过滤后路线集；本次 §4 的闭环驱动。风险：SAGE 的路线多数从物体 A 到物体 B，链式衔接需要同一物体既当终点又当起点，候选链数量未核实，可能不足。

## 做了什么

1. 飞行高度：查清能按场景指定。读码（`sim/3dgs/dynamics.py` 155–157、221、236 行）、假 `cf_sitl_proc` 的 2 个用例、真 `cf_sitl_proc`（lib31）起飞 1.2 m 与 2.0 m 的对照实测（§1）。
   导入器统一用 1.2 m，`z_slice`、`spawn.z`、渲染高度来自同一个常数 `FLY_H_M`；`dynamics_profile` 不随高度变，不用改固件。
2. 提交 `c1af32a`（`tools/`）：`nurec_to_ply.py`（13 个用例）、`scene_import_sage3d.py`（32 个用例），先红后绿（`reason_label` 与 `fly_height_m` 两处例外，见下一节），
   14 个变异里 13 个让用例变红，第 14 个改错位置后补断言再验（§2）。
3. 导入 6 个场景（试点 5 个，备用 1 个 841980）到 `$D/3dgs-scenes/sage3d_<id>/`：高斯数、PLY 字节、路线过滤与净空 / 长度 / 步数分布、指令数、
   坐标系复核（6 个全过）、每场景 4 张起点视角图（§3）。新增磁盘：5 个 PLY 共 1,030,548,344 B（841150 的 PLY 是 bench 文件的硬链接，不占新空间）、VLN JSON 68,944,765 B，
   合计约 1.1 GB（限额 2.5 GB）；下载的 usdz 共 490,401,506 B，转完已删（只删自己下的 5 个）。
4. 提交 `9098598`、`6182237`、`1c01078`（`experiments/`）：`2026-11-10-sage3d-pilot`（`run.sh` 含预注册、`select_routes.py`、README、34 + 2 个用例），
   参数化 bearing-guard 的 `driver.py` / `summarize.py`（同一解释器下 bearing-guard 输出逐字节不变）。任务书写的是 2 个提交，另外两个是追加的：`6182237` 是诊断要的 config 键（§4.6），`1c01078` 修 `run.sh` 的 `GATE_SR` 覆盖 bug（§4.7）。
5. 守护进程跑完正式实验的阶段 1（120 集，8 分 34 秒）：教师 SR(stop) 合计 0.63，P1 在 5 个场景上全部被证伪，按预注册没跑阶段 2（§4.4）。
6. 追查 P1 没过的原因：坐标系、飞行高度、占据代理都不是；教师在 hold 1.0 下的航向死区（±12°）小于一个转角步（27–35°），超时集里 25/29 含左右交替原地打转（§4.5）。
7. 诊断实验（教师容差 18°，同 120 个路线与 seed，非预注册）：结果见 §4.6。
8. 许可提醒（§5）与两张后续任务卡：20 场景完整试点、多段路线拼接（§6）。


## 没做什么（每条带原因）

1. 没跑 `forward`、`forward+route`、`calswa-s1+route` 三个臂，所以 P2、P3 没有数据：预注册规定闸门 P1 不过不跑其余臂；要不要放行由主会话定（命令见 §4.7）。
2. 没改教师（`sim/3dgs/planner.py` 的 `YAW_TOL_DEG`），也没改任何默认值：教师是共用代码，采集与其他实验都在用，改默认值要主会话决定；诊断实验只通过 config 键 `teacher_tol_deg` 传参。
3. 没拿容差 18° 的诊断结果去替换 P1 的判决：诊断是看到失败之后做的，不是预注册；4.4 的判决不变。
4. 没拆开极限环的两条来源（航向死区过冲、前向被堵时按误差符号转）：没有时间，也没有单独的消融。
5. 没取原版 InteriorGS 的 occupancy / structure / labels：要点同意并给 token。后果：占据用高斯代理，住宅 / 公共的归类只依据指令词与渲染图，未经官方标签核实。
6. 没用 SAGE 的碰撞体（`*_collision.usd`）做占据真值：读 USD 要 `pxr`，三个解释器里都没有，不往现有 venv 里装包。
7. 没用 low-level 与无目标的指令：试点只用 overall 指令，每条路线的全部指令字符串与类别都存了（§2.2）。
8. 没对 841980 跑闭环：过滤后只有 2 条同起点的路线（Scene-Unseen 场景没有 train JSON 可补）。
9. 没跑「需满电档」的路线（840139 的 tr_3、tr_16，A* 21.65 m 与 22.39 m，hold 1.0 下未封顶 118 与 124 步）：SITL 起始 50% 电量，约 100 步的电量限。
10. 没做 841150 的 PLY 与 bench 转换产物的逐字节 `cmp`：只核对了高斯数（393,895）与字节数（97,687,491）。
11. 没处理 ToF 平地模型与飞行层以下家具的差别：只给路线打 `agl_model_unverified`（§3 表 3.3），仿真里的 ToF 输入没改。
12. 没做真机与 Gazebo（任务书不要求）。
13. 没在真实数据上跑过导入器自己的 `download` 与 `convert` 两步：真实的下载、转换与删 usdz 是 `$S/fetch_scenes.sh` 做的（同样的逻辑，另一份脚本），导入器这两步只有单测（429 退避 3 项）；`align`、`scene`、`render` 三步在 6 个真实场景上都跑过。
14. `reason_label` 与 `fly_height_m` 的用例是实现之后补的，不是先红后绿；用变异检查 `m12`、`m14b` 补了验证。
15. 没验证到点半径 1.0 m（`REACH_RADIUS_M`）对室内是否合适；室外 4 个场景各自是 1.0 m 或 2.0 m。
16. Wilson 区间按「集」算，同一路线的 3 集共用路线与相近出生位姿，并不独立，区间偏窄；没做按路线聚类的区间。
17. 没复核 `run.sh` 头注释里「n=2477」的口径：我用全部 forward 臂复算得 n=4790、均值 0.298 m，均值一致，n 的差别没查。正式实验已起跑，头注释不能再改。
18. 没在最终 HEAD 上重跑 `sim/3dgs/test_*.py`：`git diff bd3a034 HEAD -- sim/` 为空，7 个文件 rc=0 的结果来自同一份 `sim/`（`$S/test-3dgs-sage3d.log`）。
19. 提交数是 4 个，任务书写的是 2 个：诊断实验要的 config 键（`6182237`）、`GATE_SR` 覆盖 bug 的修复（`1c01078`）各单独成了一个提交。
20. 没起放行续跑（`GATE_SR=0`）：放行改的是预注册流程，要主会话定；20:59 有人（不是我，疑为主会话）起过一次，因 bug 没放行，见 §4.7。
21. 没把 `1c01078` 的修复在「闸门未过、需要靠 `GATE_SR=0` 放行」的真实场景上跑通：只用冒烟验证了覆盖值被读到（`GATE_SR=1.1` 时 `config.json` 的阈值是 1.1，DONE rc=4）；`GATE_SR=0` 放行后的阶段 2 在真实闸门未过的场景上没跑过。


## 验证到哪一步

| 项目 | 验证到哪一步 | 证据 | 没验证到的 |
|---|---|---|---|
| NuRec 转 PLY | 13 个单测（`/opt/conda/bin/python3` 下全跑；`.venv-tools` 下 6 项依赖 msgpack 的打印 SKIP 后跳过，rc=0）；841150 重转的高斯数与字节数与 bench 产物相同 | `$S/logs/final_new_tests.log` | 逐字节 `cmp` 没做 |
| 场景导入器 | 32 个单测（先红后绿，两处补测例外）+ 14 个变异；`align` / `scene` / `render` 在 6 个真实场景上跑过 | `$S/logs/test_sage_import_green2.log`、`mutation_check_sage_import.log` | `download` / `convert` 两步只有单测，没在真实数据上跑（真实下载转换是 `fetch_scenes.sh`） |
| 飞行高度 1.2 m | 读码 + 单测（假进程）+ 真 `cf_sitl_proc` 起飞 1.2 / 2.0 m 对照 + 闭环 120 集 `physics_verified` 全为真 | §1、`smoke_flight_height.json`、`$OUT/summary.json` | ToF 平地模型对飞行层以下家具不成立的影响（未核实） |
| 坐标系（Rz 180°，零平移） | 6 个场景的统计复核全过；教师在 5 个场景上 A* 走通 76 集 | `align.json`、§3 表 3.2 | 没有独立的地标对照 |
| 路线过滤与 `scene.json` | 单测 + 6 个场景的真实输出；`scene.json` 经 `sim.3dgs.scene` 加载器往返 | `routes_check.json`、`import_summary.json` | 住宅 / 公共归类未经官方标签核实 |
| 起点视角渲染 | 6 个场景的拼图人工看过 | `$S/import/sage3d_<id>/render/` | |
| 闭环试点 | 冒烟（1 场景 1 路线 1 集 4 臂）；正式阶段 1：teacher 120 集；其余 3 臂 0 集 | `$OUT/summary.md`、`$S/smoke/` | P2、P3 没有数据 |
| P1（教师闸门） | 5 个场景全部 < 0.8，已证伪 | `$OUT/gate_prereg.json` | |
| 教师失败机理 | 读码 + 实测转角步（27–35°）+ 交替段统计（室内 25/29，室外 19/19 个超时集） | §4.5、`$S/logs/teacher_osc_main.txt` | 两条来源没拆；因果只有 §4.6 诊断的间接支持 |
| 诊断（容差 18°） | 同 120 个路线与 seed 跑完，逐集配对对 12° | §4.6 | 非预注册；单次；没换 seed 重复 |
| 实验侧测试 | `test_select_routes.py` 16、`test_summarize_params.py` 13、`test_driver_params.py` 7，共 36 项，rc=0（`.venv-tools` python，标准库） | 提交 `9098598`、`6182237` 的说明 | |
| bearing-guard 兼容 | 同一解释器（`3dgs-venv`）下，改前改后的 `summarize.py` 在 120 集输出上的 `summary.md` / `summary.json` md5 相同，与原目录相同 | `$S/bg_copy_a`、`bg_copy_b` | `driver.py` 只有桩测试（`test_driver_params.py`），没有真跑一遍 bearing-guard |
| `check.sh --python-only` | rc=0，HEAD `6182237`（脚本自己写明未检查 C++/CUDA 构建、CTest、GPU 测试、cf_sitl） | `$S/logs/check_python_only_final.log` / `.rc` | 上述未检查项 |
| `sim/3dgs/test_*.py` | 7 个文件 rc=0；`sim/` 与 main 相同 | `$S/test-3dgs-sage3d.log` | 最终 HEAD 上没重跑（`sim/` 未变） |
| 许可 | 读了条款（`reports/bench.md` 第 16、27 行） | §5 | 训练出的权重算不算「再分发」：未核实 |
