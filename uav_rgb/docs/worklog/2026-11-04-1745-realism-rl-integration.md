# 2026-11-04 写实 Gazebo 资产 + Teacher/RL 交付包集成

- 时间：2026-11-04 17:45
- 范围：`code/`（唯一开发主线 `code/uav_rgb/`，git 仓库；顶层 3 个交付文档不在任何 git 仓库内）
- 结论：完成（代码集成 + 离线回归验证通过；Gazebo 实机闭环 / PPO 训练 / 真机 / CUDA 全量构建未验证）

## 做了什么

- **基线提交**：应用户「git 代码方便 diff」要求，先把集成前工作树（cf_stab 动力学等 15 个文件）提交为 `2426d04`，使本次改动可用 `git diff 2426d04..HEAD` 干净查看。
- **解压**：`UAV_Gazebo_RealAssets_TeacherRL_20261104.zip` 解到 `_UAV_zip_extract/`（只读参考，未直接覆盖生产目录）。
- **读提示**：`code/更新说明_必读.md`、`code/CHANGELOG_20261104_realism_rl.md`、`code/DELIVERY_MANIFEST.json`、`code/uav_rgb/docs/realism_rl_v1/README.md`（§0–§9，含 §7 验收顺序）、`VALIDATION.md`、`code/TinyAerialVLN.md`（审计门禁与 §7 审计指令）。
- **按清单合入**：`DELIVERY_MANIFEST.json` 登记的 23 个文件（7 改 + 16 新），合入前逐条 sha256 校验 23/23 一致；另合入 `code/` 顶层 3 个交付文档与 `uav_rgb/sim/assets/**` 8 个离线原创示范资产（manifest 未登记，靠 `asset_campus` 的 `resource_sha256` 兜底）。
- **未合入 3 个 `#U` 转义重名文件**：`TinyAerialVLN_#U...md` 与既有 `TinyAerialVLN.md` / `TinyAerialVLN_源码Bug审计及仿真专项_整合版_20261104.md` 内容完全相同（sha256 `bd732e90…`）；两处 `#U76d1...md` 解码为 `监控系统交接.md`，与已存在且被 git 跟踪的同名文件哈希相同。合入只会产生乱码文件名重复文件。
- **并行 subagent 分组审查**：4 组（资产场景 / Teacher-RL 链路 / 数据集与全量回归 / 文档清单与环境边界）并行只读审查，各自贴真实 stdout/stderr；本文件汇总其结论。
- **按审查结果应用 9 处最小补丁**（均为交付包离线单测未覆盖的确凿缺陷，非重构）：
  1. `sim/tools/import_asset.py:32` — mtllib 守卫由 `line.startswith('mtllib ')` 改为按首词小写比较，堵住前导空白/大写/Tab 绕过（否则带外部 .mtl 的 OBJ 被登记却缺贴图）。
  2. `sim/tools/build_demo_assets.py` — `main()` 开头检查 catalog 的 `types.*.provenance`，拒绝覆盖人工导入的 CC0 资产（README 原本只有文字警告）。
  3. `sim/tools/split_manifest.py` — 新增 `group_id_for_task`；`cross_split_groups(pairs, task)` 与 `--verify-task` 让 `--verify` 能查出 campus / campus_assets 同图跨 split 泄漏（对应审计文档声称但未落地的 SIM-04 / PR-03）；`split_for_seed` 未知 task、`split_for_scene` 的 asset-campus 版本错配改为 fail-closed。
  4. `rl_teacher/gym_env.py`、`train_ppo.py`、`evaluate.py` — 缺 `gymnasium` / `stable_baselines3` 时给明确安装提示，不再甩裸 traceback。
  5. `docs/realism_rl_v1/README.md`、`VALIDATION.md` — 标注两条 `teacher/test_dataset_*.py` 需带可用 torch 的解释器（本环境系统 python3 缺 `libcudnn.so.9`）。
  6. `uav_rgb/README.md` — 新增章节追加一句：25/25 CTest 与 EGL 结论属历史基线，本路线未跑 Gazebo/真机、未训练 PPO。
- **新增回归测试** `sim/tools/test_realism_rl_guards.py`（8 项 unittest）：覆盖 mtllib 4 种绕过写法、campus/campus_assets 跨 split 泄漏检出、base 默认不变、未知 task / 版本错配 fail-closed、可选依赖友好报错。
- **9 个补丁后文件哈希与交付清单不同**（清单保持为交付记录，未改写）。交付值 → 集成值：

| 文件 | delivery sha256（前 12） | integrated sha256（前 12） |
| --- | --- | --- |
| `sim/tools/split_manifest.py` | `4361158ebd9b` | `92cc18f487a8` |
| `sim/tools/build_demo_assets.py` | `10737c21c567` | `0825d6062d96` |
| `sim/tools/import_asset.py` | `ba0b5a95b6db` | `bbd72c57f1f0` |
| `rl_teacher/gym_env.py` | `b5622b059902` | `e196609358bf` |
| `rl_teacher/train_ppo.py` | `78a0e45734bf` | `e2e8e30483e5` |
| `rl_teacher/evaluate.py` | `eb108d0e20716` | `b3192ef8e9e4` |
| `docs/realism_rl_v1/README.md` | `a18b78fc3390` | `0ee7a749d765` |
| `docs/realism_rl_v1/VALIDATION.md` | `ce7057f3cb85` | `0031e28e058e` |
| `uav_rgb/README.md` | `abb0f23778b1` | `a6144b7efc38` |

其余 14 个清单文件仍与 ZIP 逐字节一致。`sim/assets/**` 8 个文件 sha256：`4cc6c1de`、`65f58b72`、`d2ec5fdb`、`0f2055c2`、`810961c7`、`06296f33`、`933cf59c`、`7eba06e5`（按 `find sim/assets -type f | sort` 顺序）。

## 没做什么

- **未跑真实 Gazebo 闭环 / `gz sim` / 渲染**：无卡模式（`nvidia-smi` 无输出、无 `/dev/nvidia*`），`DISPLAY` 未设置、无 EGL 设备；`gz` CLI 8.15.0 存在但无渲染条件。原因：环境不具备，且交付包 `更新说明_必读.md` / `VALIDATION.md` 明确声明未验证。
- **未训练 PPO、未产出 checkpoint**：`gymnasium`、`stable_baselines3` 均未安装（`ModuleNotFoundError` 已实测），且训练需在用户可用 Python 环境执行。原因：依赖缺失 + 禁止伪造。
- **未下载真实第三方 CC0 资产**：无外网；包内只有原创离线示范网格与人工导入工具。原因：交付本身声明，且许可需逐项人工核实。
- **未改 CF-Drone 固件**：`TinyAerialVLN.md` §7 与交付文档均禁止；`CF-Drone/**` 与基线哈希一致。
- **未同步 `code/ALVLN/uav_rgb/` 历史镜像**：审计文档明确其为旧镜像、禁止混用；本地实现后不自动同步。
- **未重算历史实验数据 / 未重导教师 `.npz`/UAVSOFT1**：`CHANGELOG` 要求旧数据冻结、按源哈希另行重导；超出本次范围，且 PR-01 只完成代码层修复。
- **未做 CUDA 编译 / `bash tools/check.sh` / CTest 全量构建**：无 GPU；`README.md` 的「25/25 CTest 通过」是历史结论，未复跑、未引用为本次结果。
- **未做研究协议变更**：Campus 颜色因果反事实、真实 STOP 指标（reach_ever / task_success / stable_termination）、split 拓扑正交化、`campus.v3` 数据重建、指标合同冻结（PR-02/04/05/06/07/08）全部未动。原因：这些会改变数据集/指标版本，必须与旧实验隔离另立 PR，不能伪装成无副作用补丁。
- **未跑 `teacher/` 下 5 个测试的系统 python3 版本**：`import torch` → `ImportError: libcudnn.so.9`，属环境依赖；已改用 `$DATA_ROOT/uav_rgb/teacher-venv/bin/python` 复跑关键两条并 PASS，其余 3 条未跑（同样依赖）。
- **未原样执行 `python3 -m compileall -q .`**：仓库自带 `.venv-tools` 为 Python 3.12，含 PEP695 语法，3.11 解释器会假红 8 处；改用 `-x 'venv-tools'` 通过。
- **未改 `split_manifest.py --out` 的 `build_manifest()`**：仍按 base 几何分组，超出最小补丁范围，列为遗留。
- **未清理运行产生的 `__pycache__/*.pyc`**：标准生成物且被 `.gitignore` 忽略；按「不删用户文件」保留。
- **未执行 `git push`**：用户只要求提交以便 diff，未授权推送。

## 待确认

- 是否要把 9 个补丁后的哈希同步进 `code/DELIVERY_MANIFEST.json`（当前清单是 ZIP 交付时的不可变记录，与集成版有意不同）。
- `code/TinyAerialVLN_源码Bug审计及仿真专项_整合版_20261104.md` 是与 `TinyAerialVLN.md` 完全相同的既有副本，是否保留（未删）。
- 是否把 `code/` 顶层 3 个交付文档纳入某个 git 仓库（它们目前不在任何仓库内，故不随本次 commit 提交）。
- 研究层遗留（颜色因果、STOP 指标、正交 split、20+ 独立地图 T0/T1/T2 配对闭环）是否另立任务。

## 验证到哪一步

环境：Python 3.11.15 / numpy 1.26.4 / Pillow 12.2.0；无卡；`gz` 8.15.0 但无 DISPLAY。全部命令在 `code/uav_rgb` 下执行。

| 项 | 结果 |
| --- | --- |
| `DELIVERY_MANIFEST.json` 23 条 sha256（合入时） | 通过，BAD=0 |
| `sim/tools/test_realism_rl_guards.py`（新增 8 项） | 通过 `Ran 8 tests ... OK` |
| `python3 -m unittest discover -s sim/tools -p 'test_*.py'` | 通过 `Ran 35 tests ... OK` |
| `sim/tools/` 全部 22 个 `test_*.py` 单独运行 | 21 个首跑 PASS；`test_teacher_policy.py` 首跑 90s 冷启动超时、复跑（含本轮）PASS |
| `python3 -m rl_teacher.test_core` | 通过 `rl_teacher core PASS` |
| `teacher/test_dataset_windows.py`、`test_dataset_step_gap.py`（teacher-venv） | 通过 `PASS` / `step-gap regression: PASS [0, 1, 5]` |
| 同上两条用系统 `python3` | 失败 `ImportError: libcudnn.so.9`（依赖，非代码回归）；已在文档标注 |
| `python3 -m compileall -q -x 'venv-tools' .` | 通过，无输出 |
| `scene_gen.py --task campus_assets --seed 0/19 --out /tmp/...` | 通过，SDF/JSON 生成、`ok=True`、目标四色齐全 |
| `scene_gen.py --task campus --seed 0`（旧任务回归） | 通过，输出与基线一致 |
| `test_asset_campus.py` 160 worlds | 通过 `{'train': 16, 'val': 2, 'test': 2}` |
| `test_campus.py` 48 scenes | 通过 |
| `test_collect.py` 46 项 / `test_split_manifest.py` 9 项 / 其余回归 | 全部通过 |
| `split_manifest.py --verify --verify-task campus_assets / campus` | 通过：均报 `跨 split 的几何组 1`（修复前为 0） |
| `split_manifest.py --verify`（默认 base） | 通过：`跨 split 的几何组 0`，默认行为未变 |
| `python3 -m rl_teacher.train_ppo/evaluate`（缺依赖） | 通过：退出码 2、无 traceback、含安装提示 |
| `build_demo_assets.py` 前后 8 个资产 sha256 | 通过：字节一致（新增守卫不改变 demo 生成行为） |
| `git diff --check` | 通过，rc=0 |
| Gazebo 闭环 / CUDA / CTest / PPO 训练 / 真机 | **未跑**（原因见「没做什么」），不得写「通过」 |

## 风险与遗留

- **离线测试不证明性能**：只证明 Python 逻辑与静态 SDF 契约；不得据此宣称 T1 优于 T0、成功率提升或 Sim-to-Real 可行。
- **T1/T2 未实机验证**：`teacher_predictive` 只用离线公式单测；`rl_teacher` 的简化 2.5D 动力学与 Gazebo 的 sim-sim 差异未做同动作轨迹定量对照；`SAFE_NEUTRAL` 非主动制动，稳定停靠不保证。
- **口径隐患**：`--teacher-planner predictive` 复用旧 `teacher="plan_action"` 枚举，shipped audit 已用版本守卫规避、文档已声明，但只按该单字段过滤的第三方消费者可能把 T1 标签当旧 A* 标签；`eval.json` 不记 `teacher_version`，`--keep-work` 关闭后 workdir 被删更难追溯。
- **T1 硬编码风险阈值 0.3** 与 `reward_v1.json` 的 `safety_distance=0.35` 不一致（不同组件，调参易误当成同一阈值）。
- **新分支单测覆盖仍不足**：`test_collect.py` 46 项与 `test_eval_closed_loop_args.py` 都是基线旧用例；`predictive` / `teacher-rl` / `dyn=None` 守卫 / 新 manifest 字段没有正式回归，本轮只补了 `test_realism_rl_guards.py` 覆盖其中一部分。
- **审计文档与主线代码曾有偏差**：`TinyAerialVLN.md` 声称 SIM-04 已用 `group_id_for_task` 修复，本轮补丁前全仓无此函数；本轮已补上 `--verify` 路径，但 `--out` 的 `build_manifest()` 仍 base-only。
- **资产完整性**：8 个示范资产未登记进 `DELIVERY_MANIFEST.json`，只靠运行时 `resource_sha256` 兜底；真实 CC0 资产导入后不能再跑 `build_demo_assets.py`（已加代码门禁）。
- **视觉/碰撞不一致是设计而非 bug**：树冠视觉 1.4m 宽 vs 碰撞 box 0.4m、巡航 z≈2.1–2.3m、`building_back` 网格被拉伸；`clearance.ok=True` 只代表 A* 可达，不是 Gazebo 碰撞验证。
- **旧 `--task campus` SDF 仍含 4 处绝对路径**（基线既有行为，`campus.py` 与基线逐字节相同）；新 `campus_assets` 为 0 处。
- **测试脆弱性**：`test_asset_campus.py` 用裸 `assert`，`python3 -O` 下会被剥离；`test_teacher_policy.py` 冷启动 `timeout=300` 大于外层 90s 会假超时；`compileall -q .` 因 py3.12 venv 假红。
- **文档环境差异**：`VALIDATION.md` 的 PASS 是打包环境结论，本环境 2 条 teacher 测试需换解释器；`uav_rgb/README.md` 的历史 CTest/EGL 表述已加限定句。
