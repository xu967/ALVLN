# 训练配方的稳定性筛选与工作点扫描（任务卡 E2E-R1）：第一部分（离线）、第二部分（闭环）的工具、入口与结果

- 时间：2026-11-12 01:07 接手（工作树创建时刻）；第一部分守护进程 01:38:02–02:22:42；第二部分守护进程 02:35:54 首次起跑（等锁时遇 02:52 整机停机），06:58:13 续起至 09:29:33 `DONE rc=0`；本文 02:40 首次写成，09:46 收尾更新。
- 范围：`experiments/2026-11-12-recipe-stability/`（全部新增，没有改库代码、C++、`src/`、`sim/tools/`、`tools/` 下的现有文件）
- 分支：`p12-e2e-r1`（基于 main `f56d54f`），未合并、未推送
- 结论：完成。第一部分（离线）：卡里的 R2、R3 被训练器拒绝零权重没有跑成，R1 不过线，替代臂 R2e、R3e 过线。第二部分（闭环，20 个 val 几何 × 3 训练种子）：三档成功数 `−0.5 nat` 36、`0` 档 40、`+0.5 nat` 39（极差 4 集，预注册判定一落在 4–7 的「不判」区间）；事后探索臂 R2e 为 41（14/14/13，极差 1，按 P4 公式过线，不作预注册过线结论）；12 臂共 8,791 步迟到步 0、停顿步 0。

## 做了什么

- 复现审核 H 的 R0 数字：01:19 先用审核 H 自己的脚本在参照 run 存下的逐轮预测上重跑一遍（`/root/rivermind-data/uav_rgb/tmp/phase12-cards/e2e-r1/h1_reviewer_rerun.log`，与审核 H 的 `h1_bias_drift.log` 的 `diff` 为空）；之后 `stability.py repro-h1` 用本卡自己的实现重算，18 行与审核 H 的日志逐行相同（`results/repro_h1.txt`）。原始翻转率 0.226 / 0.214 / 0.238，逐轮校准后 0.053 / 0.054 / 0.046，三种子全一致率原始 0.671、校准 0.899。
- 第一部分的入口与对账脚本（`run_offline.sh`、`gate_check.py`、`avg_preds.py`，提交 `e94cd2d`）：训练池锁、守护进程、实例锁、起跑前核对、复训闸门（R0 种子 0 重训，与参照 run 逐轮对账）、各臂三个种子训练、逐轮 val 预测、末 10 轮平均与平均加校准。
- 发现并处理 R2、R3 的 0 权重被训练器拒绝（提交 `f7a9632`）：演练时撞上，起跑后各用卡里的原配方试训一次，拒绝原话落进 `blocked/R2.txt`、`blocked/R3.txt`；另跑替代臂 R2e、R3e（SAFE 权重 1e-6）。DONE 用 rc=21 区别于全部成功的 0。
- 稳定性指标脚本 `stability.py`（`e8b9ef5`、`3f86fd1`，后续 `522f396`、`54dda2c`）：相邻轮翻转率（原始与逐轮校准，全程、末 9 对、最后 10 对）、三种子全一致率（末 10 轮均值、末轮、平均权重，原始与校准）、校准量 δ 的范围、val 无权 CE 曲线、平均权重的 macro-F1 与每类召回、过线判决（9 种读法组合）、噪声底、复现审核 H 的子命令 `repro-h1`、翻转率分段表。`test_stability.py` 25 例，合成数据的参考值手算，另有与审核 H 日志逐行对账、与 `select_ckpt` 的 `epochs.csv` 对账。
- 第一部分正式运行（PGID 665642，01:38:02 起，02:22:42 `DONE rc=21`）：闸门通过（复训 R0b-s0 与参照 run 的 30 个轮的权重逐位相同，macro-F1 差 0）；R1、R2e、R3e 各 3 个种子训练完，逐轮 val 预测与末 10 轮平均加校准全部完成；输出 426 MB。
- 第一部分结果（提交 `0623948`，`results/` 与 `RESULT.md`）：R0 末 10 轮原始翻转率（三种子最大）0.236，R1 0.097，R2e 0.008，R3e 0.012；三种子全一致率 0.671、0.793、0.942、0.926。R1 不过线，R2e、R3e 过线。详见 `RESULT.md` 第 1 节与报告。
- 验证 R0 的平均加校准流程等价于现成的 `cal_swa_s{0,1,2}_k10.bin`：三个种子逐字节相同（`sha256sum`）。第二部分的 0 档与加臂由此走同一套流程。
- 第二部分的几何清单 `select_val_geoms.py`（`7ecf6a3`）：base 任务只有 804 个不同几何，一个 seed 区间不能保证几何不重叠，所以按几何组 ID 挑：[30000, 50000) 内排除 591 个已被用过的几何组，val 分区 71 个几何里排除 40 个，余 31 个，取 seed 升序前 20 个；用 `scene_gen.py` 真生成逐个核对分区与几何组 ID；与训练数据集几何重叠 0 个。
- 第二部分的权重平移 `make_shifted_ckpts.py`（`7ecf6a3`）：`shift_forward_bias` 把现成 `cal_swa_s{0,1,2}_k10.bin` 的 FORWARD 偏置平移 ±0.5 nat，核对除 `fc.b[FORWARD]` 一个 float32 外逐字节相同、变化量等于平移量。
- 第二部分的入口与汇总（`c518eb7`、`ecf0804`）：`run_closed_loop.sh`（守护进程、实例锁、机器锁阻塞等待、冻结代码 `git archive`、起跑记录只追加、两条车道、迟到集重跑一次、DONE 的 rc 区分臂失败）、`summarize_sweep.py`（三档对比、几何聚类符号翻转精确检验、按几何 bootstrap、逐集 McNemar 并列、预注册判定、逐几何表、迟到与重跑记录）、`stub_eval.py`（演练桩）。`test_sweep.py` 21 例，`test_geoms_ckpts.py` 14 例，`test_rehearsal.py` 9 例（用桩把入口脚本从头跑到尾：全流程、整臂失败与续跑、等机器锁、正式模式拒绝覆盖项、输入缺失、平移权重生成失败、替代臂加臂），对脚本的 7 个人为变异全红。
- 看到第一部分结果后改了加臂规则（`54dda2c`，`542266d`）：卡里的臂都没过线、替代臂过线时，在过线的替代臂里按同一规则取一个加进第二部分，结果里标明替代；选出 R2e。这是看到第一部分结果之后的决定，不是预注册，已在 `run_offline.sh`、`run_closed_loop.sh` 的文件头、`RESULT.md` 1.5、报告里记下。
- 更正：`run_offline.sh` 文件头原写「训练集 24,615 样本、SAFE 168 个」，与训练器状态行的 8205 个训练样本不符（24,615 = 3 × 8205，来源没有追溯），更正为 8205 与 56（提交 `f18aacb`，只改注释）；`README.md`、`RESULT.md` 同步。数据盘上第一部分输出里的冻结副本 `code/` 保留原注释，没有动。
- 第二部分守护进程起跑（02:35:54，PID = PGID = 977092，冻结代码 `542266d`）：起跑条件 ①②、几何清单核对、配方臂判定都通过，在条件 ③ 等机器锁。起跑前把 `check.sh --python-only` 与全部测试放在前面跑完，避免与闭环评测争 CPU。

## 没做什么

- 没有按卡的写法训练 R2、R3。原因：训练器在入口拒绝给训练集里存在的类设 0 权重（`src/class_weights.cpp:199–206`），交办禁止改 C++。
- 没有验证替代臂 R2e、R3e 与卡里的 R2、R3 等价。原因：训练器不接受 0 权重，没有可对照的 run；1e-6 的影响只做了估算（SAFE 样本对加权 CE 的相对贡献约 7e-9：56 × 1e-6 / 8,149）。
- 没有跑「只加类权重、不加余弦学习率」的臂，没有给 R3e 起闭环臂。原因：卡里没有这两项，加了会改卡的臂表和额度（加臂只一个）。
- 没有对第一部分的差异做检验与区间。原因：val 只有 574 帧、三个训练种子，卡没有预注册检验；只报点估计与限定。
- 没有直接数训练集里各类标签的样本数。原因：由 auto 权重与训练样本数反推（56、6885、639、625，和为 8205）已能互相印证；读数据集标签要另写读取代码，不影响结论。
- 没有使用 test split，没有用训练器结束时打印的 `test_accuracy`。原因：卡与公共规则不让用 test 做选择。
- 没有做 logit adjustment、EMA，没有在冻结测试 seed 上跑任何档，没有加档或加种子。原因：卡划给 E2E-P1 或明令禁止。
- 没有改 C++、库代码，没有改 `progress.md`、`CLAUDE.md`，没有合并、没有推送、没有碰主仓库工作树与别人的目录和进程。原因：交办。
- 没有沿用 `run_r2.sh` 的「容器 CPU < 1.5 核」这一起跑条件。原因：它是为标定时序测量设的，这里整机在训练池并行工作，容器常年 20 核以上负载，沿用会一直等；窗口内平均核数仍记进 `start.txt`，每臂的限流增量记进 `<臂>.cg`（见 `run_closed_loop.sh` 条款 P7）。
- 第二部分尚未出结果的各项（三档对比、判定一、判定二、检验）：见下面「第二部分状态」。

## 待确认

- R2e、R3e 能否当作卡里 R2、R3 的等价写法。需要主会话确认；任务卡 W2-1（`docs/briefs/2026-11-10-phase12-task-cards.md` 第 321 行附近）写了 `--class-weights 0,1,2,2`，同一代码路径，大概率同样被拒绝，没有试过，未核实。
- 第二部分加的是替代臂 R2e，不是卡里的臂；这是看到第一部分结果之后的决定。主会话可以不采用这个臂的结果。要看 R3e 的闭环结果需要另起一轮（输入在 `exp-recipe-stability-20261112/avg/R3e-s{0,1,2}/avg_cal.bin`）。
- 第一部分的结论是配方（类权重）改了原始预测的工作点，校准量 δ 的符号翻转（R0、R1 为正，R2e 为负，R3e 在 0 附近）；第二部分的 ±0.5 nat 扫描是在 R0 配方的检查点上做的，结论不能直接搬到别的配方。
- 过线只度量稳定，不度量可用：R2e、R3e 的 L / R 召回（平均权重不校准）0.292 / 0.354、0.400 / 0.434，比 R0 的 0.450 / 0.495 低；要不要把它们当新的学生配方，需要闭环结果一起判断。
- 训练器逐位确定，同配方同种子的噪声底为 0；以后不能用「重训一次看差」来量噪声。

## 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `test_stability.py`（`.venv-tools` 解释器，`nice -n 10`，02:37） | `Ran 25 tests OK` |
| `test_sweep.py`（02:37） | `Ran 21 tests OK` |
| `test_geoms_ckpts.py`（02:37） | `Ran 14 tests OK` |
| `test_rehearsal.py`（02:37–02:39，用桩把 `run_closed_loop.sh` 跑完整流程） | `Ran 9 tests OK`（84 s） |
| `run_closed_loop.sh` 的人为变异（7 个：缺臂也 rc=0、不重跑迟到集、不等机器锁、正式模式不拒绝覆盖、配方臂输入不核对、臂永远算已完成、平移权重生成失败不报） | 7/7 被 `test_rehearsal.py` 抓到（草稿目录 `mutate_rehearsal.py`，不入库；第一轮 M7 漏掉，补了 `test_shift_generation_failure_is_rc11` 与 `RS_SWA` 覆盖项后抓到） |
| `stability.py` 加臂逻辑的人为变异（5 个：不回退到替代臂、替代臂不标替代、忽略卡里的臂、加臂不看过线、JSON 不写 `extra_arm_card`） | 5/5 被 `test_stability.py` 抓到（草稿目录 `mutate_stability.py`；第一轮 S5 漏掉，补了 `test_card_arm_has_priority` 后抓到） |
| `bash -n run_offline.sh`、`bash -n run_closed_loop.sh`；`py_compile` 全部 Python 文件 | 语法通过，`py_compile` rc=0（生成的 `__pycache__` 已删） |
| 复训闸门（R0b-s0 对参照 run） | 末轮 val macro-F1 差 0，30 个轮的权重逐位相同，`gate.json` 过 |
| 复现审核 H 的 R0 数字 | `repro-h1` 18 行与审核 H 日志逐行相同，rc=0 |
| R0 平均加校准权重对现成 `cal_swa_s{0,1,2}_k10.bin` | 三个种子 `sha256sum` 相同：`for s in 0 1 2; do sha256sum $D/exp-recipe-stability-20261112/avg/R0-s$s/avg_cal.bin $D/exp-swa-calib-20261110/ckpt/cal_swa_s${s}_k10.bin; done`，前 16 位 `2fb25b30b3375899`、`5cfe51f58f4f4a1d`、`abcae7cf0d09a475` |
| 第一部分正式运行 | `DONE rc=21`（预期：R2、R3 被拒绝，替代臂完成），输出 426 MB（`du -sh`） |
| `stability.py report` 在真实数据上 | rc=0，写出 `stability.json`、`stability.md`、`ce_curves.csv`、`curves.png`；读过图 |
| `tools/check.sh --python-only`（`check.lock` 守护进程，`UAV_RGB_PYTHON` 指向 `.venv-tools`） | 02:08:26 `rc=0`（结果 `PASS（有 SKIP）`：未检查 C++/CUDA 构建、CTest、GPU 测试、cf_sitl）；最终版本的复查见下一行 |
| `check.sh --python-only` 复查（`check2`，同样走 `check.lock` 守护进程，02:37:11 起跑） | 02:41:59 `rc=0`（`PASS（有 SKIP）`，跳过项同上） |
| 第二部分守护进程起跑检查 | 02:35:56 起跑条件 ①②、几何清单核对、配方臂 R2e、冻结代码核对通过；在条件 ③ 等机器锁；`launches.log` 记「覆盖项：无」 |

## 第二部分状态

已完成（`2026-11-12 09:29:33 DONE rc=0`），详见下方收尾更新。

## 更新 2026-11-12 09:46

接手口令：`e2e-r1-1112-0932`（写入 `$D/tmp/phase12-cards/e2e-r1/OWNER`）。只做第二部分结果独立核对、报告与文档归档，不改设计、不改预注册、不改冻结代码与已有实验输出。

### 做了什么

- 按接手规范核查环境与守护进程状态：`02:35:54` 首次起跑（PGID `977092`）在等待 `machine.lock` 期间遇 `02:52` 整机停机，未产生任何半成品臂（`*.partial-*` 为 `0` 个）；主会话 `06:58:13`（PGID `18148`）续起后于 `09:29:33` 完成全部 12 个臂（`DONE rc=0`，`*-retry1` 为 `0` 个），当前无残留守护进程。
- 新增独立核对与事后分析脚本 `experiments/2026-11-12-recipe-stability/recheck_sweep.py` 及测试 `test_recheck_sweep.py`（提交 `d3bc6f7`）：直接从 12 个臂的 `eval.json` 与 240 集的 `actions.jsonl` 按条款 P6 重数结局、动作与步数，调主仓库 `tools/paired_stats.py` 重算四组两两比较，扫描 12 臂 `.cg` 与逐步时序健康，并对撞柱与越界做按几何聚类的事后符号翻转检验及 20 集二项波动精确枚举；核对产物写至新建目录 `$D/exp-recipe-stability-20261112/closed-loop/post/recheck.{json,md}`。
- 完成 71 项逐格对账，`mismatches = 0`：`paired_stats.py` 重算的四组几何聚类符号翻转精确 `p`（`0.5493`、`1.0000`、`0.7070`、`1.0000`）与逐集精确 McNemar `p`（`0.5034`、`1.0000`、`0.6636`、`1.0000`）同 `summary.md` 逐位相等；文档写清 `summary.md`（60 集成功数差、`B=10000` 离散分位）与 `paired_stats`（成功率差、`n_boot=2000` 线性插值）两套 bootstrap 区间的口径差别。
- 完成 12 臂全量时序健康核查（补齐第二轮审核 M-10 未覆盖的后 6 臂）：全容器按核累计被限流时长占墙钟 `300.6%–1206.5%`（折合 12 核配额 `25.1%–100.5%`），但 240 集 `8,791` 步中迟到步为 `0`（派发延迟中位数均 `0.200 s`，最大 `0.205 s`），`8,551` 个相邻观测间隔中 `> 1.10 s` 停顿步为 `0`（最大 `1.100 s`），巡航步（`r >= 4`）前飞位移均值 `0.2098–0.2136 m`（极差 `1.8%`），稳态步（`r >= 2`）均值 `0.2175–0.2211 m`（极差 `1.6%`）。
- 归档第二部分结果（提交 `af929f4`）：将冻结汇总器生成的 `summary.md` 与 `summary.json` 原样拷入 `experiments/2026-11-12-recipe-stability/results/`（`cmp -s` 逐字节一致），并在 `RESULT.md` 补全第 2 节（含四条判读限定、时序健康表、事后失败构成分析与局限）；同步补全卡报告 `$D/tmp/phase12-cards/reports/e2e-r1.md`。

### 没做什么

- 未修改 `$D/exp-recipe-stability-20261112/` 下已有的任何文件（仅新建 `closed-loop/post/` 写入 `recheck.{json,md}`）。
  - 原因：交办硬约束，保护冻结实验记录。
- 未修改工作树里已冻结的 `run_offline.sh`、`run_closed_loop.sh`、`summarize_sweep.py`、`stability.py`（包括审核 M-10 指出的 `summarize_sweep.py` 未打印 `throttled_s / wall_s` 占比及区间单位与 `paired_stats` 不同这两点）。
  - 原因：交办规定冻结入口不改，发现的口径差异写进报告并由 `recheck_sweep.py` 补表。
- 未启动 Gazebo、未占用 GPU、未拿任何池的锁（含未重跑排 `check.lock` 的 `tools/check.sh --python-only` 与耗时演练 `test_rehearsal.py`）。
  - 原因：交办规定「不启动 Gazebo、不占 GPU、不拿任何池的锁」，机器 12 核配额满载且 Gazebo 池正跑其它实验；本次未改动 `src/`、`sim/`、`tools/` 下任何受 `check.sh` 检查的文件，也不用碰 `run_closed_loop.sh`（前任已在 `f935405` 前完成两次 `check.sh --python-only` 与 `test_rehearsal.py` 验证）。
- 未补跑「只改类权重 `1e-6,1,1,1`、不开 `--cosine-lr`」的离线消融臂，未补跑 `R2e` 未校准权重（`avg_raw.bin`）或 `R3e` 的闭环臂，未在 20 个冻结 test seed 上评测。
  - 原因：交办规定「只写报告，不改设计、不改预注册」，后续补臂与否作为选项交主会话定。
- 未更新主仓库的 `progress.md`，未合并进 `main`，未推送远端。
  - 原因：主仓库只读，合并、推送与全局进度汇总由主会话统一处理。

### 待确认（供主会话决策的选项清单，不代拍板）

1. **卡 W2-1（基线 v2）采用哪个训练配方**：
   - 选项 A：按卡 W2-1 现行文字采用 `R3e`（`--cosine-lr --class-weights 1e-6,1,2,2`）+ `calswa`（依据：离线过线，原始前进占比 `0.866–0.885` 自然贴近教师 val `0.873`，校准平移小，未校准转向召回高于 R2e；但无闭环实测）。
   - 选项 B：改用 `R2e`（`--cosine-lr --class-weights 1e-6,1,1,1`）+ `calswa`（依据：离线最稳，已有 60 集闭环实测 `41/60`、种子极差 `1`、撞柱 `3/60`；但在无护栏下越界 `16/60` 且受负向校准影响）。
   - 选项 C：上冻结 test 前先在同一批 20 个 val 几何补跑 `R3e` 闭环（60 集，约 45 分钟 Gazebo 池），与 `R2e` 配对后再定。
2. **是否补「只改类权重 `1e-6,1,1,1`、不开余弦退火」的消融**：
   - 选项 A：不补，直接将 `--cosine-lr` 与显式类权重捆绑作为标准配方（依据：余弦退火零额外开销，前半程高学习率段已排除纯退火假象）。
   - 选项 B：在训练池补 1 个 3 种子离线臂（约 7 分钟 GPU，不占 Gazebo 池），用于论文或报告严格拆分两项开关各自的独立效应。
3. **`R2e` 不做负向校准（或校准到更高工作点）是否补一个闭环臂**：
   - 选项 A：用现成 `$D/exp-recipe-stability-20261112/avg/R2e-s{0,1,2}/avg_raw.bin` 在同批 20 个 val 几何补跑 60 集闭环（零训练，约 45 分钟 Gazebo 池），直接检验 `16/60` 越界是否由 `−0.78` 至 `−0.98 nat` 负向校准造成。
   - 选项 B：不在纯图像模型上单独补，留到带航向码或原型线（卡 E2E-1b / E2E-P1）一并处理（依据：三档扫描显示纯图像标量偏置在 `−0.5` 至 `+0.5 nat` 内主要在撞柱与越界间置换，加高偏置减少越界的同时撞柱由 5 升至 12）。
4. **撞柱少而越界多的策略（如 `R2e`）加上航向码（卡 E2E-1 在测）的衔接方向**：
   - 选项 A（若 E2E-1 证实航向码有效压低越界）：将 `R2e` 配方（撞柱 `3/60`、种子极差 `1`，失败集中于越界 `16/19`）与三值航向码 + 按码 0 单独校准（卡 E2E-1b，避免全局校准压掉非零码回转）组合，作为端到端主候选。
   - 选项 B（若 E2E-1 在纯示范集上未压住越界）：说明示范集非零码帧占比过低（val 仅 `15/574` 帧），需将 `R2e` 配方放到 DAgger 聚合集或结合 FiLM 前移 + 条件置空（卡 E2E-P1 的 C1b/C4）再测闭环。

### 验证到哪一步

| 项 | 结果 |
| --- | --- |
| `nice -n 15 $R/.venv-tools/bin/python -m py_compile recheck_sweep.py test_recheck_sweep.py` | 通过（`rc=0`） |
| `nice -n 15 $R/.venv-tools/bin/python -m unittest -v test_recheck_sweep` | 通过（`Ran 3 tests in 16.7s, OK`：全量对账、篡改检出、P6 重跑与停顿步检出、事后符号翻转与二项极差） |
| `nice -n 15 $R/.venv-tools/bin/python -m unittest test_stability test_sweep test_geoms_ckpts` | 通过（`Ran 60 tests in 18.9s, OK`） |
| `nice -n 15 $R/.venv-tools/bin/python recheck_sweep.py`（真实闭环输出） | 通过（`rc=0`：`checked=71 mismatches=0 arms=12 steps=8791 late=0 stalls_gt_1p1=0`，写出 `closed-loop/post/recheck.{json,md}`） |
| `cmp -s` 核对 `results/summary.{md,json}` 与 `closed-loop/summary.{md,json}` | 通过（逐字节相同） |
| `tools/check.sh --python-only` | 本次未重跑（原因：交办禁止拿任何池的锁 `check.lock`，且本次未修改 `src/`、`sim/`、`tools/`；前任已在 `02:41:59` 验证 `rc=0`） |

### 风险与遗留

- 冻结的 `summarize_sweep.py` 有两处已知口径差异（第二轮审核 M-10）：一是逐臂明细表只打印 `nr_throttled` 次数而未打印 `throttled_s / wall_s` 占比；二是 `bootstrap_ci` 返回 60 集成功数差的离散分位（`B=10000`），而仓库标准库 `tools/paired_stats.py` 返回成功率差的线性插值分位（`n_boot=2000`）。本卡已通过 `recheck_sweep.py` 补齐并列，后续新实验应直接调用 `tools/paired_stats.py` 与 `tools/exp_lib.sh`（卡 W1-15c 正在修 `exp_lib.sh` 的按核累计限流比例展示）。
- 第二部分闭环运行在 `--action-latency 0.20` 新协议（含步间 0.10 s 回 SAFE 保持）上，待卡 W1-13 第三轮切换为「动作持续到下一次派发」后，每步前飞位移与转角将回到旧协议附近，工作点与偏置敏感度若搬到第三轮协议下需注意被控对象步长变化。
- 越界在 `−0.5` 对 `+0.5 nat` 上的聚类符号翻转 `p = 0.0391` 属于事后探索分析且未做多重比较校正，只能作为「标量偏置在撞柱与越界间置换」的机制线索，不能当成预注册验证结论引用。
