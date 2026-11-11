# W1-12：3DGS 教师前向净空改扫掠、出生两层判空、导引指针（实现、验收实验、归因）

- 时间：2026-11-11 23:30 卡细化后开工；第 1 部分的量与离线反事实 23:37–23:40；代码与测试 11-12 00:24 前提交；验收实验守护进程 00:41:35–01:02:41；本文 01:25 写
- 范围：`sim/3dgs/{planner,collect_3dgs,guard}.py`、`sim/3dgs/test_teacher_clearance.py`、`experiments/2026-11-10-bearing-guard-3dgs/driver.py`（只加两个可选键）、`experiments/2026-11-11-teacher-clearance-3dgs/`
- 分支：`p12-w1-12`（基于 main `a714b54`）
- 结论：完成，预注册判决（臂 A，机械）是「介于预测与证伪之间」。室内到点 112/120 = 0.933（W1-11 101/120），撞在前 8/120 = 0.067（W1-11 19/120），超时 0；室外 119/120 = 0.992（与 W1-11 逐集相同）；
  配对 b = 11、c = 0、p = 0.000977。五条预测里「室内撞在前 <= 0.06」差 1 集没满足，其余四条满足，两条证伪条件都没命中。回归臂 `ray` 室外 30 集、室内 24 集与 W1-11 逐步逐字段相同。

## 做了什么

- 第 1 部分（先量，再离线反事实，达到门槛才改代码）：新教师臂 240 集 9174 步，路线下标偶数拟合、奇数检验。检验半前进步预测误差 p90：P1 上下文表 0.080 m，P2 速度外推 0.041 m，按任务卡取 P2；
  前进后转弯步位移模 p90 0.132 m，`reserve` = 0.132 + 0.041 = 0.1731 m。离线反事实（P2 交叉拟合，主口径）命中 13/16、误拒 0.11%（7/6391），过门槛（命中不到 10 或误拒高于 10% 停）。P1 交叉拟合旁证 16/16、0.25%。
- 教师（`sim/3dgs/planner.py`，`bea8812`）：`OccAstarTeacher(clearance="ray"|"swept")`。`ray`（缺省）与改动前逐步相同；`swept` 在机头线段通过后沿预测位移延长 `reserve` 再做一次 `occ.sweep`；
  被否决时 `|err| >= tol/2` 出 `turn_swept`，否则出 SAFE_NEUTRAL（`wait_swept`），连续最多等 2 步；goal-stop 的「连续 3 步 SAFE」不被等待步误触发（测试锁住）；版本 `uavrgb.teacher_occ.v3`；`swept` 只接受 hold 1.0。
- 采集入口（`collect_3dgs.py`，`f9d938d`）：`--teacher-clearance {ray,swept}`；manifest 写 `teacher_clearance`（教师带该属性才写）与 `spawn_two_layer`（总写）。
- 出生判空（同一提交）：`spawn_two_layer`（缺省关，`--spawn-two-layer` 要求 `--spawn-jitter > 0`）。候选点在出生高度与「巡航高度」两层都不碰撞，且 `occ.clearance` 不低于机体半径加 0.05 m。关闭时出生位姿与 rng 消耗逐位不变（测试锁住）。
- 导引指针（`guard.py`，`6f897ce`）：先在 K1 的两个停滞集上离线复现（重放的目标点与 `rule.jsonl` 逐步一致），再加 `nearest_in_lookahead`（缺省关，缺省行为不变）。
- driver（`de56c36`）：臂上两个可选键 `teacher_clearance`、`spawn_two_layer`，缺省与原版逐位相同；改动行：28–32、62–70、163–164、190、219、272–275（文件内行号）。
- 测试：`test_teacher_clearance.py` 53 项（旧树 52 项里 48 项报错，新树全过）、`test_driver_clearance.py` 15 项（旧驱动 9 失败 2 报错）、`test_analyze.py` 41 项；变异检查 `sim` 44/44、driver 15/15、`analyze.py` 43 个里 42 杀死 + 1 等价。
- 验收实验入口 `experiments/2026-11-11-teacher-clearance-3dgs/{run.sh,analyze.py,measure_disp.py,counterfactual.py,replay_guide.py,README.md}`：预注册与「另报」原文进 `run.sh` 头；拿 3DGS 池的锁 `gs3d.lock`；起止各记 cgroup；
  结束时自动出回归比对、配对汇总、预注册判决（`DONE` 与 `analysis/acceptance.{md,json}`）。
- 验收实验结果：见上「结论」。臂 B（swept + 出生两层，只报告）室内 114/120、撞在前 6；对 W1-11 b = 13、c = 0；对臂 A b = 2、c = 0、p = 0.5。室内臂 A 的 `wait_swept` 占 0.7% 的步、`turn_swept` 0.1%，SAFE_NEUTRAL 标签 9.7%（W1-11 8.8%），导航阶段 0.8%（W1-11 0）。
- 归因补充（草稿脚本，不入库，输出在草稿目录）：剩余 5 个非出生撞在前集的撞击点都在最后一个前进步的检查线段的沿线范围内，横向离线段 0.8–3.1 cm；等待步位移中位 0.123 m、最大 0.177 m（悬停不是停住），29 步里 1 步撞；
  出生判空只覆盖起飞前的候选点，候选点到实际起点（起飞漂移）中位 0.031 m、最大 0.082 m，臂 B 剩下的 1 个出生即撞集由这段漂移造成。
- 提交（`a714b54..HEAD`）：`bea8812`、`f9d938d`、`6f897ce`、`bb1ceb2`（`sim/3dgs`）；`de56c36`、`d0e8073`、`9a07011`、`1ae2eee`、`d3829d4`、`efba13b`、`07ddf18`（`experiments/`）；docs 提交 `fff0615`（README 结果）、`dda4093`（实验数字）和本文所在的提交。
- 报告：`/root/rivermind-data/uav_rgb/tmp/phase12-cards/reports/w1-12.md`；实验数字：`docs/experiments/2026-11-12-teacher-clearance-3dgs.md`。

## 没做什么

- 没改 A* 的规划余量、hold、航向容差，没动 `rule_oracle.py`、`rule_policy.py` 与 driver 里的规则臂。原因：任务卡「不做」。
- 没给转向步加否决，没改 `wait_swept` 的悬停动作。原因：任务卡规定转向不否决（转向没有替代动作）；等待步的减速设计超出任务卡，归因数据已写进实验文档，要不要开卡由主会话定。
- 没在新 seed 上重跑，没调 `reserve`、预测系数或等待上限。原因：任务卡规定判决不成立时不改参数、不在这批 seed 上重跑调参，也没有给新 seed。
- 没在闭环里验证 `nearest_in_lookahead`。原因：任务卡只要求离线重放说清停滞原因并加缺省关的参数；闭环验证要动 K1 的臂与导引，规则臂卡 K3 在改。
- 没处理与卡 K3 的合并冲突。原因：任务卡规定冲突由主会话处理，报告里写了改动行和 `hold_s` 重复的处理办法。
- 没跑全量 `tools/check.sh`。原因：公共规则规定全量 `check.sh` 由主会话跑，改动没有碰 C++。
- 没改 `progress.md`、`CLAUDE.md`，没合并、没推送，没改主仓库工作树。原因：任务书。
- 没清理默认 torch 扩展缓存 `~/.cache/torch_extensions/py312_cu128/gsplat_cuda/` 里的编译中间物。原因：补跑 `test_renderer.py`（`RUN_3DGS_REAL=1`）时第一次漏设 `TORCH_EXTENSIONS_DIR`，gsplat 开始在默认目录重编；约 2 分钟后按 PGID 停掉。
  已有的 `gsplat_cuda.so`（11-07 15:46，约 20 MB）没被改写；新增的 `.o` 与改写的 `build.ninja`、`.ninja_log`、`.ninja_deps`、`lock` 共 29 个文件 11.8 MiB（根分区，01:14–01:16 写入）。任务书规定不删除既有文件，别的会话是否在用这个目录未核实，交主会话决定。
- 没把 `inspect_remaining.py`、`inspect_segment.py`、`wait_stats.py`、`spawn_drift.py` 放进仓库。原因：它们是一次性的归因草稿，数字与结论已写进实验文档，脚本留在草稿目录 `$D/tmp/phase12-cards/w1-12/`。

## 待确认

- 臂 A 室内撞在前 8/120 差 1 集过 0.06，机械判决是「介于」。要不要把 `swept` 设为 3DGS 教师的默认（现在缺省仍是 `ray`），以及剩余失败要不要另开一卡，主会话定。
- 臂 B（出生两层）不是预注册主臂：室内 114/120、撞在前 6/120。是否把出生两层升为默认；巡航高度 `spawn.z + 0.05 m` 是一个选择（首步 `z − spawn.z` 室内实测 +0.012..+0.070，中位 +0.045）；
  出生判空要不要把起飞漂移（p90 0.055 m）算进余量，或起飞后再检一次。
- 验收与拟合共用同一批 seed（任务卡规定），+11 不是样本外结果；要不要换一批 seed 做样本外检验。
- 合并顺序与 `driver.py` 冲突：`git merge-tree` 只读试合并 `p12-w1-12` 与 `p12-k3`，只有 `driver.py` 冲突，4 处（文件头说明、臂校验循环、kw 字典定义、教师构造）；K3 在教师构造处传 `hold_s=hold`，本卡 `teacher_kw` 里也带 `hold_s`，同时存在会 `TypeError`，合并时去掉本卡 `teacher_kw` 里的 `hold_s`。
- `teacher_version` 出现 `uavrgb.teacher_occ.v3`：用 `swept` 采的数据与 v1、v2 数据是否混训，`preprocess` 是否按版本分层，未核实，主会话定。
- 默认 torch 扩展缓存里误产生的编译中间物（见「没做什么」）怎么处理。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `sim/3dgs/test_*.py` 全部 10 个文件（11-12 00:36–00:38，`3dgs-venv` python，HEAD `bb1ceb2`，之后 `sim/` 无改动） | 213 项全过：collect_eval 28、collision 26、dynamics 6、dynamics_sitl 11、guard 24、renderer 11（2 项跳过）、rule_oracle 32、teacher_clearance 53、teacher_tol 18（W1-11 的 18 项）、video 4 |
| `test_renderer.py` 补跑（01:17，`RUN_3DGS_REAL=1`，`TORCH_EXTENSIONS_DIR` 指向 `torch-cache/extensions-sm75`） | 11 项全过，0 跳过；真实 gsplat 渲染 320×240 约 3.0 ms |
| 新增测试 | `test_teacher_clearance.py` 53 项：旧树 `FAILED (errors=48)`（52 项）→ 新树 `OK`；`test_driver_clearance.py` 15 项：旧驱动 `FAILED (failures=9, errors=2)` → `OK`；`test_analyze.py` 41 项 `OK`（`.venv-tools` 与 `3dgs-venv` 的 python 各跑一遍；三个文件在 HEAD `07ddf18` 上 01:2x 复跑，53 / 15 / 41 项均 `OK`） |
| 变异检查 | `sim` 44 个变异全杀死（36 个杀 33 → 补测 → 44 个杀 43 → 补测 → `g06` 杀死）；driver 15/15；`analyze.py` 43 个里 42 杀死、1 个等价（`a16`） |
| 用到 driver 的既有测试 | K1 `test_driver_rule.py` 16、`test_driver_params.py` 7、`test_summarize_params.py` 13、W1-11 `test_analyze.py` 24，全过 |
| `tools/check.sh --python-only`（`check.lock` 排队 00:42:00 起，01:02:03 拿到锁，01:05:53 结束，HEAD `07ddf18`） | `rc=0`，末行「PASS（有 SKIP）：…未检查：C++/CUDA 构建、CTest、GPU 测试、cf_sitl」 |
| 冒烟（单车道，不拿锁，`ALLOW_DIRTY=1`） | 室内 sage3d_841150 50 s、室外 garden 188 s，各 3 个驱动（回归臂、臂 A、臂 B）每臂 1 集，加分析；回归臂逐步一致，swept 臂 manifest 为 `teacher_occ.v3` 与 `teacher_clearance=swept`，`ray` 臂为 v2 与 `ray` |
| 验收实验（真实 cf_sitl 仿真，渲染在 GPU） | 全部跑完 `rc=0`，数字见上；回放核对 0 步不一致 |
| 第 1 部分离线反事实 | 命中 13/16、误拒 0.11%，过门槛；这是离线重算，实际转成到点 11/16 |
| 第 3 部分导引指针 | 两个停滞集离线复现与重放；新参数只在同一条已记录位姿序列上重放，闭环未验证 |

## 风险与遗留

- 剩余 5 个非出生撞在前集：撞击点在检查线段范围内、横向差 1–3 cm，3 集撞在转向步（任务卡规定不否决），1 集撞在悬停步（`wait_swept`），1 集撞在前进步。前两类不在 `swept` 的作用范围内，改法没有设计。
- `wait_swept` 假设悬停等于不动；室内 29 个等待步位移中位 0.123 m，有速度时会滑进墙（1 集）。等待步还把导航阶段的 SAFE_NEUTRAL 标签占比抬高 0.8 个百分点。
- 位移系数只在 hold 1.0 下量过，用 240 集全量拟合，验收在同一批 seed 上；样本外表现未核实。
- 室外没有可比较的信号（A、B、W1-11 逐集相同，唯一失败集是出生即撞）。
- 出生两层判空对起飞漂移无能为力（臂 B 剩 1 个出生即撞集）。
- `nearest_in_lookahead` 在薄墙两侧路径靠近时可能把指针跳过墙；缺省关。
