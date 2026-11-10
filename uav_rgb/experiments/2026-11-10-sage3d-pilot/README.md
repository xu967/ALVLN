# SAGE-3D 室内 3DGS 零训练试点（2026-11-10，sage3d）

不训练。把 `experiments/2026-11-10-bearing-guard-3dgs` 的驱动与汇总（参数化后共用，没有复制）用到 5 个 SAGE-3D / InteriorGS 室内 3DGS 场景，看室内走廊与门洞（任务书给的净空是 0.2–0.4 m；选中的 40 条路线沿 A* 的最小净空，来自 scene.json `sage.min_clearance_m`：最小 0.22 m，p50 0.40 m，p90 0.60 m）对局部避障有没有检验力。

- 室外对照：4 个场景上 `forward+route` 到点成功率 0.89（107/120），`calswa-s1+route` 对它 McNemar p = 0.625，教师 0.83（来源 `exp-bearing-guard-3dgs-20261110/summary.md`）。
- 预注册全文在 `run.sh` 文件头，三条：P1 闸门（教师 SR(stop) >= 0.8，按场景判）；P2 `forward+route` 合计 SR(stop) < 0.7 支持、>= 0.89 证伪；P3 `calswa-s1+route` 对 `forward+route` 逐集配对精确 McNemar p > 0.05 支持，p <= 0.05 证伪。
- 护栏起跑前定死：阈值 45°，路线前视 1.5 m（室外 3.0 m 会越过门框与拐角）。

## 设计

| 项 | 值 |
| --- | --- |
| 场景 | `sage3d_841150`、`840865`、`841012`、`840139`、`839992`（原定 841980 过滤后只剩 2 条且同起点，换成 841012；`$D/3dgs-scenes/<场景>/scene.json`），顺序跑，同一时刻只有 1 个 3DGS 驱动进程 |
| 路线 | `select_routes.py`：每场景最多 8 条，按 `sage.astar_path_m` 取 8 个等间隔分位（含最短与最长）；步数 = ceil(max_steps × 0.5) + 15，超过 100 的路线标「需满电档」，不进池、不跑、单列 |
| 集数 / seed | 每路线 3 集；seed = 8000 + 100·路线下标 + k；出生扰动 ±0.2 m / ±10°；同 seed 各臂出生位姿相同，逐集配对 |
| 臂 | `teacher`、`forward`、`forward+route`、`calswa-s1+route`（权重 `exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin`） |
| 阶段 | 阶段 1 只跑 `teacher`；`summarize.py --gate teacher:0.8` 选出 SR(stop) >= 0.8 的场景，只有它们进阶段 2；全不过则 DONE `rc=4`，不跑阶段 2 |
| 其余 | hold 1.0、`stop_on_collision`、cf_sitl lib31；指标同 bearing-guard（`3dgs.pilot.v2+stop.v1`），另加按路线长度分桶（`<5 m` / `5–10 m` / `>=10 m`）与每条路线单列 |

## 怎么跑

```bash
cd experiments/2026-11-10-sage3d-pilot
# 正式实验：守护进程（flock 互斥；写 run.pid / run.log / DONE；配置区在 run.sh 顶部，环境变量可覆盖）
setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 冒烟：1 个场景、取最短 1 条路线、每路线 1 集；GATE_SR=0 让阶段 2 也跑
SCENES=sage3d_841150 PER_SCENE=1 K=1 GATE_SR=0 OUT=<草稿目录>/smoke setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度
tail -f $OUT/run.log; grep -v '^step ' $OUT/driver-sage3d_841150.log | tail; cat $OUT/gate.json
# 汇总（只读每集 metrics.json，随时可跑；结束时 run.sh 自己会跑一次）
<3dgs-venv python> ../2026-11-10-bearing-guard-3dgs/summarize.py $OUT
# 停（只按 PGID）
kill -- -$(cat $OUT/run.pid)
```

续跑：同一条命令重起，已完成的路线（`done.json` 集数够且无失败）跳过，半成品改名为 `*.incomplete-<时刻>`。
放行闸门没过的场景继续跑阶段 2：`GATE_SR=0 setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &`（同一个 `OUT`；教师臂已完成的路线不重跑；`config.json` 与 `gate.json` 会被重写成阈值 0，原判决要先另存）。`routes_selected.json` 与上次不同（scene.json 变了）时拒绝续跑，新的写在 `routes_selected.json.new`。

## 产物（`$OUT` = `$D/exp-sage3d-pilot-20261110`）

- `routes_selected.json`：每场景选中的路线（`route` 是 scene.json routes 的下标）、`need_full_battery`、`n_filtered`、`counts`、分桶标签。
- `config.json`、`inputs.txt`（输入 sha）、`run.sh.snapshot`、`run.log`、`driver-<场景>.log`、`run.pid`、`DONE`（`rc=… 时刻 闸门通过 [...] 未过 [...]`）。
- `gate.json`：闸门逐场景的 k/n、是否通过、原因；`summary.md` / `summary.json`：汇总（第 0 节闸门，第 4 节 P3 与补充比较，P2 单臂判决，第 6 节分桶）。
- `<臂>/<场景>/r<N>/{done.json, ep_*/{metrics.json, guard.jsonl, actions.jsonl, oracle.jsonl, rgb_00000.png}}`：逐帧 PNG 只留首帧。

## 文件与测试

- `select_routes.py`：选路线（标准库）；`test_select_routes.py`（16 项）。
- `../2026-11-10-bearing-guard-3dgs/{driver.py,summarize.py}`：新增可选 config 键（`routes` 字典、`step_cap`、`gates`、`arm_checks`、`main_pairs` 等），缺省时 bearing-guard 的输出与原版一致；`test_summarize_params.py`（13 项，合成输出目录）、`test_driver_params.py`（5 项，桩模块，不碰 GPU）。
- 三个测试都是 `PYTHONDONTWRITEBYTECODE=1 python test_xxx.py`，只用标准库。

## 已知限制

- 到点半径取自 scene.json 的 `reach_radius_m`：SAGE 场景是 1.0 m（导入器配置区 `REACH_RADIUS_M`），室外 4 个场景各自是 1.0 m 或 2.0 m；室内 1.0 m 是否合适没有单独验证。
- 闸门分母 = 计划路线 × K，阶段 1 没跑完的场景不放行（`gate.json` 的 `kind` = `incomplete`）。
- 出生扰动在 0.2–0.4 m 净空里可能找不到自由点，该集记为失败集（算失败，单列）。
