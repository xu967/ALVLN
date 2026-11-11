# 2026-11-11 3DGS 上的规则 v2（rule-v2-3dgs，任务卡 K3）

计划 `docs/briefs/2026-11-10-phase12-task-cards.md` 的卡 K3。零训练，不起 Gazebo，不含教师臂。K1 量到「完美列距离 + 规则 v1 + 路线方位」在 3DGS 室内只有 80/120 = 0.67，
失败集大多以 `V1-more` 与 `V1-cap` 的两步原地循环收尾。K3 把规则换成 v2（两侧候选、分级通道、按世界航向记的禁忌表），量同样的感知下规则层在室内能到多高。
列距离仍来自占据真值（`occ`）或渲染深度（`depth`），航向参照来自 oracle 路线，都是特权信息：结果是规则层的上限，不是学生能力。

## 问什么与预注册

规则 v1 的循环去掉之后，室内到点成功率能比 v1 高多少。`run.sh` 文件头逐字抄了任务卡的预注册（起跑前定死）：

- 主终点（留出路线，`rule2-occ+route` 对 `rule-occ+route`，同路线同 seed 逐集配对）：SR(stop) 高 0.10 以上，精确 McNemar 双侧 p < 0.05；主臂碰撞率（整集任一步）≤ 0.10，撞在前 ≤ 0.08。证伪：差低于 0.05，或主臂碰撞率高于 0.15；两者之间记「介于预测与证伪之间」。
- 次终点（正式 40 条，同样的比较，只报告）：另报主臂是否到 0.80。
- 室外（不劣）：主臂 SR(stop) ≥ 0.93；对 v1（取 K1 的结果按键配对）「v1 成而 v2 败」减「v2 成而 v1 败」≤ 5；碰撞率 ≤ 0.10。
- depth 臂只报告。口径沿用 K1：SR(stop)、碰撞率含到点之后、点估计对阈值并列 Wilson 95% 区间。

执行者在 `run.sh` 头里另补了五条口径（任务卡没写，起跑前定，没有看结果改）：路线实际条数与「3/40」「多到点 3 集」按绝对集数读；主终点判决的机械算法；F0 对拍的字段；退出码；排队用 3DGS 池的锁 `gs3d.lock`。

## 设计

- 规则 v2（`sim/tools/rule_policy.py` 末尾追加，纯标准库；v1 的既有行没有改，该文件相对基线 `41335cd` 只有新增 175 行）：`rule_v2_step` 一步 = 本步 20 列距离 → 世界系障碍点并入记忆（24 步）→ 禁忌表清超龄条目、当前航向在第 0 级被堵就记一条 → `rule_v2_entry`。
  `rule_v2_entry` 的条目：`V2-back<ℓ>`（朝回转一步）、`V2-fwd<ℓ>`（前进）、`V2-seek<ℓ>`（两侧候选：m ≠ 0、|m| ≤ 6、|n + m| ≤ max(3, |n|)，按 |m|、|e + m|、v1 的左右两半求和偏向、last_turn、左 排序）、`V2-stuck`（全不通，沿 last_turn 再转一步，没有则左）；永不输出 SAFE_NEUTRAL。
  通道分级 `levels = ((r_safe, d_look, 只许静止起步), …)`，逐级不放宽；次序 `lexi`（逐级）或 `near`（先前进）；禁忌条目 (世界航向, x, y, 步号, lb) 半径 0.30 m、有效 24 步、航向窗 α/2。
- 3DGS 接线：`sim/3dgs/rule_oracle.py` 的 `RuleOraclePolicy(…, rule_version="v1"|"v2")`，v2 下 `n = round(−err/yaw_step)`、`e = −err/yaw_step`（α = 31.7°，hold 1.0 的实测中位）；driver 的臂配置新增可选键 `rule_version`，不给时旧臂逐位不变（F0 对拍证明，见下）；教师臂按臂的有效 hold 构造（`hold_s=hold`）。
- 路线三份，互不相交，都来自试点的 5 个室内场景，合格条件与取法复用 K1 的 `select_dev_routes.select_dev`（`select_routes_v2.py` 自检：同参数重算 K1 开发路线，逐条相同）：正式 40 条（K1 的同一批，seed 8000 + 100·路线 + k，每路线 3 集）；开发 38 条（K1 的 20 条加新取 18 条，seed 9000 + 100·路线，1 集）；留出 32 条（seed 5000 + 100·路线 + k，每路线 3 集，`sage3d_841150` 0 条）。
  `sage3d_841150` 的合格路线只有 14 条（试点 8、K1 开发 4、剩 2），所以开发路线是 38 而不是 40、留出是 32 而不是 40；合计 32 ≥ 24，没有触发「少于 24 条就停」。室外 4 场景 × 6 路线 × 5 集，seed 7000 + 100·路线 + k，同 K1。
- 变体表（任务卡原文，`run.sh` 配置区，`test_run_sh.py` 与任务卡逐行核对）与自动选取：六个 v2 变体 `v2a`…`v2f` 加 `v1` 在开发路线上各跑 1 集；撞在前多于 3 集的出局，其余取 SR(stop) 最高者，相差不超过 1 集算并列、取表里靠前的；选中的比 v1 多到点至少 3 集才进正式阶段。
  depth 臂用同结构的 0.25 一档（选中 `v2e` 或 `v2f` 时用 `v2a` 或 `v2d`，因为渲染深度量到表面，不是膨胀后的边界）。
- 臂：留出路线 `forward+route`、`rule-occ+route`（v1）、`rule2-occ+route`（主臂）、`rule2-depth+route`；正式 40 条 `rule-occ+route`（v1 重跑）、`rule2-occ+route`、`rule2-depth+route`；室外 `rule2-occ+route`、`rule2-depth+route`（v1 取 K1 结果按键配对）。
- 守护进程 `run.sh`：选路线 → 开发 → 自动选变体 → 闸门 → F0（正式 40 条上 v1 先重跑，与 K1 同键逐集对拍，一致才继续）→ 正式阶段 → 汇总。起跑时把已提交的 HEAD 用 `git archive` 冻到 `$OUT/code` 再从冻结副本重新 exec；
  formal 拒绝带未提交改动与环境变量覆盖起跑；实例锁 `$OUT/.lock`；产出体积看门狗 1500 MB（rc=5）与显存采样；排队用 3DGS 池的锁（`$D/tmp/phase12-cards/gs3d.lock`，阻塞式 `flock`，持有到脚本退出），最多 3 条车道；拿到锁与结束时各记一次 cgroup 的 `usage_usec`、`nr_throttled`（`$OUT/cgroup.txt`）；续跑：同一条命令重起。
  退出码：0 完成；2 配置 / 输入 / 路线；3 其他步骤失败；4 无卡；5 体积超限；6 F0 不一致；7 驱动任务失败；20 闸门不过；143 被终止。

## 结果（正式，2026-11-12 00:13:00 拿到锁起跑，00:36:51 结束，rc=0；表来自 `RESULT.md`）

开发路线（38 条 × 1 集，hold 1.0）：选中 `v2f`，到点 33/38，比 v1（26/38）多 7 集，闸门（至少多 3 集）通过。

| 臂 | 级别（r_safe:d_look，* = 只许静止起步） | 次序 | 到点 | 撞在前 | 超时 |
| --- | --- | --- | --- | --- | --- |
| v1 | 0.25:1.2（K1 冻结） | 规则 v1 | 26/38 | 2 | 10 |
| v2a | 0.25:1.2 | lexi | 27/38 | 1 | 10 |
| v2b | 0.25:1.2, 0.25:0.8 | lexi | 27/38 | 1 | 10 |
| v2c | 0.25:1.2, 0.25:0.8 | near | 28/38 | 1 | 9 |
| v2d | 0.25:1.2, 0.25:0.8, 0.20:0.55* | near | 31/38 | 1 | 6 |
| v2e | 0.15:1.2 | lexi | 30/38 | 1 | 7 |
| v2f（选中） | 0.15:1.2, 0.15:0.8, 0.12:0.55* | near | 33/38 | 1 | 4 |

开发阶段每个臂每条路线只有 1 集，v2f 与 v2d 差 2 集，v2e 差 3 集，在抽样噪声量级；留出路线是对选择的确认。

留出路线（32 条 × 3 集 = 96 集每臂，主终点）：

| 臂 | SR(stop) [Wilson 95%] | 碰撞率 | 撞在前 | 超时率 | <5 m | 5–10 m | ≥10 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `forward+route` | 65/96 = 0.677 [0.578, 0.762] | 0.406 | 0.302 | 0.02 | 39/45 | 17/33 | 9/18 |
| `rule-occ+route`（v1） | 64/96 = 0.667 [0.568, 0.753] | 0.031 | 0.031 | 0.30 | 38/45 | 17/33 | 9/18 |
| `rule2-occ+route`（主臂，v2f） | 86/96 = 0.896 [0.819, 0.942] | 0.021 | 0.021 | 0.08 | 43/45 | 29/33 | 14/18 |
| `rule2-depth+route`（v2d） | 87/96 = 0.906 [0.831, 0.950] | 0.021 | 0.021 | 0.07 | 43/45 | 29/33 | 15/18 |

正式 40 条（120 集每臂，次终点）：

| 臂 | SR(stop) [Wilson 95%] | 碰撞率 | 撞在前 | 超时率 |
| --- | --- | --- | --- | --- |
| `rule-occ+route`（v1，与 K1 逐集一致） | 80/120 = 0.667 [0.578, 0.745] | 0.033 | 0.033 | 0.30 |
| `rule2-occ+route`（主臂） | 105/120 = 0.875 [0.804, 0.923] | 0.025 | 0.025 | 0.10 |
| `rule2-depth+route` | 106/120 = 0.883 [0.814, 0.929] | 0.033 | 0.025 | 0.09 |

室外（120 集每臂）：

| 臂 | SR(stop) [Wilson 95%] | 碰撞率 | 撞在前 | 超时率 |
| --- | --- | --- | --- | --- |
| `rule-occ+route`（v1，K1 的结果） | 117/120 = 0.975 | 4/120 = 0.033 | 1/120 = 0.008 | 2/120 = 0.017 |
| `rule2-occ+route`（主臂） | 119/120 = 0.992 [0.954, 0.999] | 0.008 | 0.008 | 0.00 |
| `rule2-depth+route` | 108/120 = 0.900 [0.833, 0.942] | 0.033 | 0.033 | 0.07 |

## 预注册判决（点估计对阈值）

| 项 | 实测 | 阈值 | 判决 |
| --- | --- | --- | --- |
| 主终点：SR(stop) 之差 | +0.229（86/96 − 64/96；b = 22，c = 0） | ≥ +0.10 | 满足 |
| 主终点：精确 McNemar 双侧 p | 4.8e-07 | < 0.05 | 满足 |
| 主终点：主臂碰撞率 | 2/96 = 0.021 [0.006, 0.073] | ≤ 0.10 | 满足 |
| 主终点：主臂撞在前 | 2/96 = 0.021 [0.006, 0.073] | ≤ 0.08 | 满足 |
| 证伪：差低于 0.05 / 碰撞率高于 0.15 | +0.229 / 0.021 | < +0.05 / > 0.15 | 均不成立 |
| **主终点判决** | | | **达到预测** |
| 次终点：差（只报告） | +0.208（105/120 − 80/120；b = 27，c = 2），p = 1.6e-06 | 无阈值 | 报告 |
| 次终点：主臂是否到 0.80 | 0.875 [0.804, 0.923]，区间下限 0.804 | ≥ 0.80 | 到 |
| 室外：主臂 SR(stop) | 119/120 = 0.992 [0.954, 0.999] | ≥ 0.93 | 满足 |
| 室外：（v1 成 v2 败）−（v2 成 v1 败） | 0 − 2 = −2（120 对，p = 0.5） | ≤ 5 | 满足 |
| 室外：主臂碰撞率 | 1/120 = 0.008 [0.001, 0.046] | ≤ 0.10 | 满足 |
| **室外判决** | | | **不劣** |

辅助核对：F0（正式 40 条上 v1 重跑对 K1）120/120 逐集一致；开发路线上 K1 的 20 条同键 20/20 一致。`recount.py` 只读各集 `metrics.json`、不经 `summarize.py` 独立复算，29 项（各臂到点数、配对 b / c / p、主终点与室外结论）与 `summary.json` 全部一致（输出 `recount.txt`）。

## 失败形态

细节见 `failures.md`。要点：

- 主臂的碰撞（holdout 2、formal40 3、室外 1）全部是出生即碰撞，出生后没有碰撞；规则 v1 的两步循环（`超时·含交替` holdout 26、formal40 32）在两个 v2 臂里都是 0。
- 主臂剩下 20 个超时集（holdout 8、formal40 12）：3 集是导引目标点不前进（K1 §18 已登记，飞机到了目标点而目标点不动），7 集末段在小范围内来回，10 集仍在位移但步数预算用完；`V2-stuck` 收尾的失败集 0 个。
- 失败集中在 `sage3d_841012`（主臂 25 个失败集里 13 个）与 10 m 以上路线（holdout 14/18、formal40 21/27）。
- 失败集里「候选航向之间的缝里有路」没有核实（`rule_v2_entry` 的 `shortcut:` 注释写的升级条件）；`failures.md` 第 4 节写了要查的做法。

## 文件

| 文件 | 作用 |
| --- | --- |
| `run.sh` | 守护进程入口：阶段 0（PGID、实例锁、无卡 / 覆盖 / 脏树拒绝、冻结代码）与阶段 1（选路线、排 `gs3d.lock`、开发、选变体、闸门、F0、正式、汇总）；任务卡的预注册与执行者补的口径写在文件头 |
| `variants.py`、`select_routes_v2.py`、`select_variant.py`、`build_config.py`、`check_k1_match.py` | 变体表解析；三批路线；开发路线上自动选变体与闸门；写 driver 的 `config.json`（含 `--only`）；F0 对拍 |
| `summarize.py` | 汇总 → `summary.md` / `summary.json` / `verdict.txt`：预注册判决（`Fraction` 做边界比较）、分臂 / 分场景 / 分桶、McNemar、失败分类、规则条目占比、与 K1 同键对拍 |
| `recount.py`、`recount.txt` | 不经 `summarize.py` 的独立复算与输出 |
| `failure_analysis.py`、`failures.md` | 失败形态的描述性统计与结论（只读日志） |
| `dry_run_driver.py` | 干跑用桩驱动（不碰 GPU，数字是伪造的，只验流水线接线） |
| `mutation_check.py` | 变异检查：140 个一次一处的变异，等价变异在 `EQUIV` 里写明理由 |
| `test_*.py` | `test_build_config.py`（13）、`test_check_k1_match.py`（11）、`test_driver_v2.py`（9）、`test_run_chain.py`（9，整条链干跑）、`test_run_sh.py`（10，静态检查）、`test_select_routes_v2.py`（15）、`test_select_variant.py`（24）、`test_summarize.py`（19）；规则层 `sim/tools/test_rule_v2.py`（71）、`sim/3dgs/test_rule_oracle_v2.py`（21） |
| `RESULT.md` | 正式 `summary.md` 的副本 |

## 运行

```bash
cd /root/claude-code-backup-20261028/code/wt/p12-k3     # 或任何含本目录与 sim/ 的检出；正式实验不需要环境变量
setsid nohup bash experiments/2026-11-11-rule-v2-3dgs/run.sh > /dev/null 2>&1 < /dev/null &
tail -f /root/rivermind-data/uav_rgb/exp-rule-v2-3dgs-20261111/run.log
cat /root/rivermind-data/uav_rgb/exp-rule-v2-3dgs-20261111/phase.txt
kill -- -$(cat /root/rivermind-data/uav_rgb/exp-rule-v2-3dgs-20261111/run.pid)    # 停：只按进程组
# 只重做汇总：python3 experiments/2026-11-11-rule-v2-3dgs/summarize.py（参数见该文件头；run.sh 里的调用是唯一的完整写法）
# 独立复算：python3 experiments/2026-11-11-rule-v2-3dgs/recount.py
# 失败形态：python3 experiments/2026-11-11-rule-v2-3dgs/failure_analysis.py
```

GPU 用环境里唯一的那张卡（`run.sh` 配置区的 `GPU=` 行）；起跑前 `nvidia-smi -L` 核对。排队：阻塞式 `flock` 3DGS 池的锁 `$D/tmp/phase12-cards/gs3d.lock`（`machine.lock` 只管 Gazebo 池）。
输出：`/root/rivermind-data/uav_rgb/exp-rule-v2-3dgs-20261111/{dev,holdout,formal40,outdoor}/`（每臂 / 场景 / 路线 / 集一个目录：`metrics.json`、`rule.jsonl`、`actions.jsonl`、`oracle.jsonl`、`episode_manifest.json`、`rgb_00000.png`、`scene.json`；`drop_trace` 去掉了 `trace.jsonl`），
另有 `routes/`、`dev_selection.{md,json}`、`selected.json`、`k1_match.{md,json}`、`summary.{md,json}`、`verdict.txt`、`cgroup.txt`、`inputs.txt`、`code/`（冻结副本）、`DONE`。总量 571 MB，显存峰值 12,811 MiB。
排队期间留下的两份只有排队、没有数据的冻结副本与 DONE（PGID 54583、112512）改名保留为 `*.d6a7ab3-first-queue`、`*.8a14cdf-second-queue`，没有删除。
