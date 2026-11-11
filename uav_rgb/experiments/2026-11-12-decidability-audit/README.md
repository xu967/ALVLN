# 可判定性审计（卡 E2E-0）

问题：特权 A* 教师的动作标签里，有多少能由端到端学生拿得到的信息决定；补哪一种输入能把缺口补上。结果用来定 E2E-1 之后条件码该带什么（计划 §10.4 的 E2）。
离线分析：不起仿真、不训练学生、不改库代码、不占任何池的锁。特征全部由仿真真值算（oracle），只用于分析，不进任何学生输入。
结果表与结论见 `RESULT.md`；全部数字的出处是 `$D/exp-decidability-audit-20261112/` 下的 `tables.md`、`results.json`。

## 文件

| 文件 | 作用 |
|---|---|
| `run.sh` | 守护进程入口：配置区、起跑前写下的预期与读法、四个阶段（extract → fit → bins → report） |
| `audit_features.py` | 选步（复用 `tools/preprocess.py::load_episode`）、真值特征（复用 `sim/tools/rule_policy.py::column_distances`）、与打包日志对账 |
| `audit_fit.py` | 特征组矩阵、子集掩码、小 MLP 拟合、混淆矩阵与指标、分箱核对 |
| `audit_report.py` | 自助法区间、与预期逐条对照、`tables.md` 与 `results.json` |
| `test_audit_features.py` `test_audit_fit.py` `test_audit_report.py` | 合成场景上的测试，手算的期望值 |
| `recheck_tables.py` | 只读核对：只用 `features.npz` 与 `preds_*.npz`，不调 `audit_*`，重算 `pooled_w`、`pooled_u` 全部 8 组在三个子集上的整体准确率、macro-F1（F/L/R）与「该转时转了」，并解析 `tables.md` 的主表，逐格比对均值（2 × 3 × 8 × 3 = 144 格，打印不一致的格子）；另打印相邻组的 macro-F1 差值，顺带数几个 `RESULT.md` 引用的计数（几何簇数、|n| ≥ 2 的帧占比）。`preds_*.npz` 只存 argmax，「转向帧左右对」（L、R 两个 logit 比）重算不了 |
| `mech_check.py` | 只读核对：转向标签的方向与 n 的符号的关系；测试帧按 \|n\| 分层看判成转向后方向对不对（`RESULT.md` §7 引用） |
| `RESULT.md` | 结果表与结论；表由 `tables.md` 原样摘录，数字出处与口径写在文件开头 |

## 数据与选步

| 名字 | 采集目录（只读） | 作为 |
|---|---|---|
| trunc | `$D/tmp/agent6-distill/trunc/col` | 教师驱动（示范，已在成功线截断） |
| more_demos | `$D/exp-dagger-gz-20261109/full-1109/more_demos/collect` | 教师驱动（示范） |
| round_1 | `…/full-1109/round_1/collect` | 学生驱动（β=0.5：每步按概率由教师或学生执行） |
| round_2、round_3 | `…/round_2/collect`、`…/round_3/collect` | 学生驱动（β=0，纯学生执行，影子教师给标签） |

进训练集的步直接调 `preprocess.load_episode(drop_collision=True)`：五份打包日志（trunc 的 `preprocess.log` 与 full-1109 各轮的 `ds.log`）都带 `--drop-collision`、不带
`--exclude-collision-episodes`。提取结束后，逐集的 split 与步数、各 split 的样本数与类别数、丢弃的碰撞步数都与这五份日志对账，对不上就停（`selection_check.json`）。
历史量（n、hist）按该集全部步算：被丢弃的碰撞步只是不当样本，它执行的动作仍然改变了航向与上一步动作。

与任务卡的出入（以实际文件为准）：柱子真值在每集目录的 `scene.json` 的 `pillars`，`episode_manifest.json` 里没有；动作前位姿在 `actions.jsonl` 的 `pose_before` 与 `yaw_before`，
`oracle.jsonl` 里的 `pose_gt` 是动作结束时的位姿，没有用。教师标签（`plan_action_global_tiebreak`）在 `collect.py` 里用的正是 `pose_before` 与 `yaw_before`，与特征同一个位姿。

## 特征

角度约定同 `rule_policy.py`：yaw 单位度，0 = +x，正 = 左转；方位正 = 左；列从左到右排。一步 = 悬停 1.0 s，约前进 0.25 m 或转 26°。

| 名字 | 定义 |
|---|---|
| `d_wide` | 动作前位姿上 180° 视场、60 列（3° 一列）、上限 4 m 的列距离（`column_distances(hfov_deg=180)`） |
| `d` | `d_wide` 的下标 20–39，即 60° 视场的 20 列；提取时在 100 个真实帧上与直接按 60° 视场调库函数的结果逐位核对 |
| `n` | 该步之前已执行的 YAW_LEFT 数减 YAW_RIGHT 数（`action_executed`，不是标签），分析时截到 ±4 |
| `hist` | 上一步执行的动作（首步按 SAFE_NEUTRAL，与 preprocess 一致）、最近一次转向的方向（没转过 / 左 / 右）、距最近一次转向过了几步（1…5，封顶 5，没转过记 5） |
| `priv` | 横向位置 y（世界系，航线沿 +x）、到最远柱的纵向距离 = 最大柱 x − 机体 x（最远柱 = x 最大的柱，教师的目标线在它前方 2.0 m）；学生拿不到 |
| 三值航向码 | W1-9 卡：\|n\| ≤ 1 → 0；n ≥ 2 → 2；n ≤ −2 → 3 |

## 特征组

卡里定的六组，输入矩阵：距离除以 4 m；n 截断后 9 位独热；hist = 上一步动作 4 位 + 最近转向方向 3 位 + 步数 5 位；priv 按训练帧的均值与标准差标准化。

| 组 | 输入 | 维数 |
|---|---|---|
| S1 | d | 20 |
| S2 | d + 上一步动作 | 24 |
| S3 | d + n | 29 |
| S4 | d + n + hist | 41 |
| S5 | d_wide + n + hist | 81 |
| S6 | S5 + priv | 83 |
| S3c（卡外附加） | d + 三值航向码（3 位） | 23 |
| S7（卡外附加） | S6 + 真实航向 sin、cos | 85 |

S3c 量「n 换成三值码丢多少」，S7 是天花板参照（教师标签是 x、y、yaw 与柱阵的确定函数），不属于卡里的特征组，报告里单独标注。

## 子集（都在测试帧上取数）

- 教师驱动的帧：trunc 与 more_demos。
- 学生驱动的帧：round_1、round_2、round_3（卡原文「学生驱动的状态 = round_{1,2,3}」）。附加子集 `student_r23` 只含纯学生执行的 round_2、round_3。
- 画面里 4 m 内没有障碍的帧：`d`（60° 视场 20 列）全是 4 m 封顶，各来源合在一起取。
- 附加：`all` 全部测试帧。

## 拟合器与协议

两层各 64 个单元的 ReLU MLP，Adam（lr 1e-3，batch 256），类别加权交叉熵（权重 N/(C·n_c)，缺的类 0，与 `src/class_weights.cpp` 的 auto 一致）。
train 拟合，val 的加权交叉熵最低的那一轮停止（最多 80 轮，10 轮没改善就停），test 只报数；按几何分组划分，用数据集 manifest 里的 split（同一 `geometry_gid` 同 split）。
每组 3 个随机种子（0、1、2：初始化与小批次顺序），报均值与极差。torch 只用 CPU 1 线程，不用 GPU。

拟合池，卡没有规定，所以跑四种配置，对照时都列：

| 配置 | 训练池 | 损失 | 取数 |
|---|---|---|---|
| `pooled_w`（主） | 全部训练帧 | 类别加权 | 各子集的测试帧 |
| `pooled_u` | 全部训练帧 | 不加权 | 各子集的测试帧 |
| `teacher_w` | 教师驱动的训练帧（验证帧同） | 类别加权 | 教师驱动的测试帧 |
| `student_w` | 学生驱动的训练帧（验证帧同） | 类别加权 | 学生驱动的测试帧 |

## 指标

- 整体准确率：四类。
- macro-F1（F/L/R）：FORWARD、LEFT、RIGHT 三类的宏平均（SAFE_NEUTRAL 在测试帧里只有个位数，单独列召回）；macro-F1（4 类）：四类宏平均，缺类按 0，与 C++ 训练器一致。
- 每类召回：SAFE、FWD、LEFT、RIGHT。
- 该转时转了（`turn_recall`）：教师标签是 L 或 R 的帧里，被判成 L 或 R 的比例。
- 转向帧左右对（`lr_acc`）：教师标签是 L 或 R 的帧里，网络 L 与 R 两个 logit 比出的方向对的比例，不要求先判成转向；`lr_cond` 只看判成转向的帧。
- 前进帧被判成转向（`false_turn`）：教师标签是 FWD 的帧里被判成 L 或 R 的比例。
- 基线：恒选 FORWARD_SLOW。

区间：测试集按 `geometry_gid` 分簇，有放回重抽样 1000 次，每次三个种子各自算指标再取均值，取 2.5% 与 97.5% 分位数；特征组之间的差值用同一批重抽样（配对）。
区间只反映测试几何有限带来的波动，不含换一批训练几何的波动。

## 分箱核对（不依赖拟合器）

把特征粗分箱，查每个箱里教师动作的多数占比，与 MLP 的准确率对量级：

- `d`：4 个扇区（各 5 列、15°）取最近距离，分 3 档（< 1.0 m、1.0–2.5 m、≥ 2.5 m）。
- `d_wide`：6 个扇区（各 10 列、30°）取最近距离，分 2 档（< 2.0 m、≥ 2.0 m）。
- n：三值航向码。hist：最近转向方向（3）× 距上次转向步数分 3 档（1 / 2–3 / 4–5）；上一步动作在步数为 1 时已由最近转向方向决定，不另分箱。
- priv：训练帧的三分位，y 3 档 × 到最远柱的纵向距离 3 档。
- S1 = d 的箱；S2 = d 的箱 × 上一步动作；S3 = d 的箱 × 航向码；S4 = d 的箱 × 航向码 × hist；S5 = d_wide 的箱 × 航向码 × hist；S6 = S5 的箱 × priv 的箱。

两个数：箱内多数占比（训练帧；箱多帧少时偏高）与分箱留出准确率（用训练帧的箱内多数类判测试帧，没见过的箱退回训练集多数类；箱多帧少时偏低）。分箱规则是不加权的多数规则，对量级用 `pooled_u`。

## 已知局限

- 几何种类有限（测试几何与训练几何同分布，数见 `RESULT.md`），测的是同分布下的可判定性，不是换布局的泛化（见 `docs/reviews/2026-11-10/data-algo.md` A8 的说明）。
- `d` 是无噪声的理想列距离，视觉前端读出的列距离只会更差，所以这里的可判定率是上界。
- 学生驱动的帧来自 DAgger 的各轮，状态分布取决于当时的学生；round_1 是 β=0.5 的混合执行。
- 小 MLP 拟合不到的结构会被读成不可判定；用分箱核对与 S7 的天花板作对照。
- 近平局帧的教师标签取决于 tiebreak 规则（`legacy`），不是所有标签都有稳定的「正确答案」；本审计不区分。

## 复现

```bash
mkdir -p $D/exp-decidability-audit-20261112 && cd $D/exp-decidability-audit-20261112
setsid nohup nice -n 10 bash <仓库>/experiments/2026-11-12-decidability-audit/run.sh > /dev/null 2>&1 < /dev/null &
# 进度 tail -f run.log；结束落 DONE；停 kill -- -$(cat run.pid)
```

测试（仓库根；numpy 即可，`test_audit_fit.py` 里的拟合用例需要 torch，没有则跳过；可用 `$D/teacher-venv/bin/python`）：

```bash
TMPDIR=$D/tmp/phase12-cards/e2e-0/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  $D/teacher-venv/bin/python experiments/2026-11-12-decidability-audit/test_audit_features.py
# 同目录的 test_audit_fit.py、test_audit_report.py
```
