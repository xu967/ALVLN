# sage3d-exp 报告：室内试点的实验脚本侧改造

- 时间：2026-11-10 20:15–20:37；子代理 `sage3d-exp`，给 `sage3d` 打下手
- 工作树：`/root/claude-code-backup-20261028/code/wt/p12-sage3d`（分支 p12-sage3d），未提交、未 add
- 结论：完成；真实驱动路径（需要 GPU）没有跑，用桩模块与桩驱动覆盖了新增逻辑

## 做了什么

- 新目录 `experiments/2026-11-10-sage3d-pilot/`
  - `select_routes.py`：读 scene.json 的 routes，先剔除「需满电档」（`ceil(max_steps×0.5)+15 > 100`），再按 `sage.astar_path_m` 升序取 8 个等间隔分位（索引 floor(i×(n-1)/(m-1)+0.5)，n<=m 全取，m=1 取最短），写 `routes_selected.json`（`scenes` / `need_full_battery` / `n_filtered` / `counts` / `params`）。`LEN_EDGES=(5.0, 10.0)` 在文件顶部配置区；步数参数由 run.sh 以必填命令行参数传入，不放第二份默认值。
  - `run.sh`：守护进程入口，配置区在顶部，头注释写预注册（问题、设计、护栏定死、P1–P3 判决规则、运行/进度/停止/续跑命令）。单车道，阶段 1 只跑 teacher，`summarize.py --gate teacher:0.8` 选场景，阶段 2 跑其余 3 臂；全不过则汇总后写 `DONE rc=4`。
  - `README.md`（55 行）、`test_select_routes.py`（16 项）、`test_summarize_params.py`（13 项）、`test_driver_params.py`（5 项）。
- 参数化 `experiments/2026-11-10-bearing-guard-3dgs/driver.py`：`routes` 可为 `{场景: [下标]}`（字典里没有该场景直接 KeyError，场景路线为空则不加载就退出）；可选 `step_cap`。
- 参数化 `summarize.py`：`routes` 字典、`--gate 臂:阈值`（stdout 只打印通过的场景名，写 `gate.json`，不写汇总）、`gates`（汇总第 0 节）、`routes_selected`（第 6 节分桶表，分路线表带路线名与 A* 长度）、`main_pairs` / `extra_pairs` / `verdict_texts`、`arm_checks`（单臂对固定阈值的判决）、`title` / `overall_label`；二节标题里写死的「× 5」改成 K。
- 顺手改了一处：`fmt_row` 在某臂一集都没成功（全是失败集）时 `steps_mean` 为 None 会崩，改成显示「—」。教师臂遇到 `route_check` 不可达的路线时会出现这种情况。

## 设计决定（run.sh）

- 闸门分母 = 该场景计划路线数 × K，没跑的集与失败集算失败；阶段 1 没跑完（`kind=incomplete`）或场景没有可跑路线（`no_routes`）都不放行，`gate.json` 写明原因；汇总里「闸门未过：先查占据代理/坐标系/飞行高度」只对 SR 不够的场景写。
- 续跑时把新选的路线写 `routes_selected.json.new`，与已有文件逐字节比较，不同就 `DONE rc=2` 拒绝（防止 scene.json 变了之后两批路线的结果混在一个目录）。
- 冒烟：`PER_SCENE=1` 取最短 1 条；`GATE_SR=0` 让阶段 2 也跑（教师 SR 为 0 的场景也放行）。
- P2 是单臂对固定阈值（< 0.7 支持，>= 0.89 证伪，之间记「介于预测与证伪之间」），阈值在配置区 `P2_SUPPORT_LT` / `P2_FALSIFY_GE`；P3 走 `main_pairs` + `verdict_texts`（p>0.05 支持，p<=0.05 两个方向都记证伪）；补充比较 x1–x3 不进判决。
- `DRIVER`、`SCENES_DIR` 可由环境变量覆盖，只为干跑测试（桩驱动，不碰 GPU）；`nvidia-smi` 查询限定到钉住的那张卡。

## 没做什么

- 没跑真实驱动、没起 cf_sitl、没跑任何 GPU 进程：任务约束。真实路径只有 `driver.py` 里 4 行新增，由 `test_driver_params.py` 用桩模块覆盖。
- 没跑 `tools/check.sh --python-only`：会读到主代理正在写的 `tools/`，结果不干净；合并前需要跑。
- 没做 `routes_selected.json` 的 JSON Schema 校验与功效估计：任务书没要求。
- 没改 `progress.md`、没写 `docs/worklog/`：这次不提交，由主代理统一写。
- 没动 `tools/`、没删任何已有文件。
- 没用整份 bearing-guard 输出目录做拷贝对比：它 404 MB，超过 50 MB 的限额；改为只复制 summarize 读的三类小文件，见下表。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `test_select_routes.py` 红（`select_routes` 不存在） | `ModuleNotFoundError`，日志 `$D/tmp/phase12-20261110/sage3d/logs/test_select_routes_red.log` |
| `test_select_routes.py` 绿 | 16 项通过，`…/logs/test_select_routes_green.log` |
| `test_summarize_params.py` | 对改动前的 summarize（`git show HEAD:` 取出）14 处失败（红灯是实现之后补跑的），新版 13 项通过；`…/logs/test_summarize_params_{red,green}.log` |
| `test_driver_params.py` | 对旧 driver 1 失败 3 错误，新版 5 项通过；`…/logs/test_driver_params_{red,green}.log` |
| bearing-guard 输出不变 | 原目录 404 MB；复制 config.json / done.json / metrics.json 共 1153 个文件（9.2 MB）到 `…/sage3d/bg_copy_a`、`bg_copy_b`；旧版对 a、新版对 b：`summary.md`、`summary.json`、stdout 的 `diff` 均为空；旧版在拷贝上的结果与原目录 19:57 生成的 `summary.md` / `summary.json` 也无差异 |
| 干跑 run.sh（桩驱动 + 假 nvidia-smi） | A 闸门部分通过、B 全不过 `rc=4` 且没有阶段 2、C 续跑全部跳过、D 第二实例 `flock` 退出 `rc=3`、E scene.json 变更 `rc=2`；用 5 个真实 scene.json（20:34 的版本）跑通整条流程 |
| 真实 scene.json 读取（只读） | 5 个场景 routes 数 14 / 40 / 2 / 34 / 123，选 8 / 8 / 2 / 8 / 8 条；840139 有 2 条需满电档（r1 118 步、r6 124 步） |
| `bash -n run.sh`、各 .py 的 `ast.parse` | 通过 |

## 风险与遗留

- 841980 只有 2 条路线，闸门分母 n = 6：5/6 通过，4/6 不过，一集之差。
- 合计 34 条路线，每臂最多 102 集；P3 的配对 McNemar 功效有限（没估）。
- 真实 scene.json 的 `reach_radius_m` 是 1.0（导入器写的），不是缺省 1.5；到点判定按它走，我没改。
- 出生扰动 ±0.2 m 在 0.2–0.4 m 净空里可能找不到自由点，该集记为失败集。
- 每步前进距离：任务书记 0.25 m，bearing-guard 输出里 forward 臂 FORWARD 步均值 0.298 m（n=2477，我从 actions.jsonl 算的），路线前视 1.5 m 约 5 步，不是 6 步。预注册数字没改，run.sh 头注释两个数都写了。
- 任务书里的 `.venv-tools` 路径 `/root/rivermind-data/uav_rgb/.venv-tools` 不存在，实际是 `/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools`；测试用后者，run.sh 用 3dgs-venv（选路线与汇总只用标准库）。
