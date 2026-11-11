#!/usr/bin/env bash
# 3DGS 上的规则 v2（2026-11-11，rule-v2-3dgs；任务卡 K3，docs/briefs/2026-11-10-phase12-task-cards.md「卡 K3」）。零训练，不起 Gazebo。
# 问题：K1 量到规则 v1 + 完美列距离在 3DGS 室内只有 80/120 = 0.67，失败集大多以 V1-more 与 V1-cap 的两步原地循环收尾。
#   规则 v2（sim/tools/rule_policy.py 的 rule_v2_step：两侧候选、分级通道、按世界航向记的禁忌表）把循环去掉，量同样的感知下规则层在室内能到多高。
# 一个守护进程把整条链串起来，中间不需要人改文件：
#   选路线（正式 40 / 开发 / 留出，互不相交）→ 开发阶段（v1 + 变体表里的 6 个 v2 变体，各 1 集）→ 自动选变体 → 闸门（不过：DONE 写明「开发路线上 v2 没有比 v1 好」）
#   → F0（正式 40 条上 v1 重跑，与 K1 的结果同键逐集对拍，不一致就停）→ 正式阶段（留出路线 × 3 集四个臂；正式 40 条 × 3 集三个臂；室外 24 条 × 5 集两个臂）→ 汇总。
# 代码：起跑时把已提交的 HEAD 用 git archive 冻成 $OUT_ROOT/code，再从冻结副本重新 exec 自己（之后工作树里的改动不影响这次运行；bash 逐段读脚本，原文件被改会让正在跑的实例读错位置）。
#   formal 拒绝带着未提交改动（git status 里已跟踪文件有改动）起跑；git HEAD 与未提交文件数写进 inputs.txt。
# 下面「任务卡原文」一节逐字抄自任务卡（test_run_sh.py 与 docs/briefs/…task-cards.md 逐行核对），起跑前定死，不改字。
# 执行者补的口径（任务卡没写；起跑前定，不看结果改）：
#   1. 开发阶段与留出路线实际取到的条数：sage3d_841150 的合格路线只有 14 条（试点 8、K1 开发 4、剩 2），开发 = 38 条（K1 的 20 + 新取 18），留出 = 32 条（该场景 0 条）；
#      「3/40」「多到点 3 集」都读成绝对集数（撞在前多于 3 集出局；多到点至少 3 集过闸）。比率读法（3/40 = 7.5%，上限取 floor(3·n/40) 集）只记进 dev_selection.json / .md，不参与选取。
#   2. 主终点判决（机械套用数字）：Δ = SR(stop)(rule2-occ) − SR(stop)(rule-occ) 在留出路线的全部集上（同路线同 seed 逐集配对）。
#      「达到预测」= Δ ≥ 0.10 且精确 McNemar 双侧 p < 0.05 且主臂碰撞率（整集任一步）≤ 0.10 且撞在前 ≤ 0.08；「证伪」= Δ < 0.05 或主臂碰撞率 > 0.15；其余记「介于预测与证伪之间」。
#      次终点（正式 40 条）同样的比较只报告，另报主臂 SR(stop) 是否到 0.80；室外：主臂 SR(stop) ≥ 0.93 且 b − c ≤ 5（b = v1 成而 v2 败，c = v2 成而 v1 败）且碰撞率 ≤ 0.10 才记「不劣」。
#      各条都对点估计，同时列 Wilson 95% 区间（沿用 K1 的口径：SR(stop)、碰撞率含到点之后）。
#   3. F0 对拍的字段：success_stop、collision、collide_first、timeout、steps、reach_step、collision_step 与出生位姿（容差 1e-6）；不一致 → DONE rc=6，不进正式阶段。
#   4. 退出码（DONE 第一个字段）：0 全部完成；2 配置 / 输入 / 路线问题；3 其他步骤失败（含汇总）；4 无卡；5 产出体积超限；6 F0 对拍不一致；7 有驱动任务失败；20 闸门不过；143 被终止。
#   5. 排队的锁：任务卡「资源与排队」原写机器锁 machine.lock；11-11 23:55 起主会话改为 3DGS 池的锁 $D/tmp/phase12-cards/gs3d.lock（公共规则 §4 同一时刻的那一条，
#      任务卡「资源与排队」一条已加同样的更新说明；machine.lock 只管 Gazebo 池）：一次一个 3DGS 批量实验，不超过 3 条车道（LANES=3）；守护进程起止各记一次 cgroup 的
#      usage_usec 与 nr_throttled（写 run.log 与 $OUT_ROOT/cgroup.txt）。预注册、路线三分、选变体规则、闸门都没有因此改动。
# ---------------- 任务卡原文（docs/briefs/2026-11-10-phase12-task-cards.md「卡 K3」，逐字抄，起跑前定死）----------------
# - 实验（入口 `experiments/2026-11-11-rule-v2-3dgs/`，输出 `$D/exp-rule-v2-3dgs-20261111/`；沿用 K1 入口的结构与共用的 driver；hold 1.0；`yaw_step_deg` 31.7；
#   一个守护进程把「开发 → 自动选变体 → 正式」串起来，中间不需要人改文件）。
#   路线分三份，互不重叠，都来自试点的 5 个室内场景，用 K1 的 `select_dev_routes.py` 同一套合格条件与按 A* 长度分位的取法：
#   正式 40 条 = K1 的同一批（同 seed，8000 + 100·路线下标 + k）；
#   开发 40 条 = K1 的 20 条开发路线加新取的 20 条（每场景 4 条；seed 9000 + 100·路线下标）；
#   留出路线 = 再新取的每场景 8 条（不够就有多少取多少；合计少于 24 条就停下报告）；seed 5000 + 100·路线下标 + k。
#   留出路线在这张卡之前没有被任何规则跑过，是主终点。正式 40 条的失败形态在设计 v2 时已经看过，单列为次终点。
#   1. 开发阶段（40 条 × 1 集）七个臂：`v1`（r_safe 0.25、d_look 1.2，K1 的冻结值）与六个 v2 变体。
#
#      | 变体 | levels（r_safe, d_look；带 * 的级只许静止起步） | order | 想分清什么 |
#      | --- | --- | --- | --- |
#      | `v2a` | (0.25, 1.2) | lexi | 只改决策逻辑 |
#      | `v2b` | (0.25, 1.2)、(0.25, 0.8) | lexi | 加一级短通道，绕不开时才用 |
#      | `v2c` | (0.25, 1.2)、(0.25, 0.8) | near | 加一级短通道，先前进 |
#      | `v2d` | (0.25, 1.2)、(0.25, 0.8)、(0.20, 0.55)* | near | 再加一级静止起步的更短通道 |
#      | `v2e` | (0.15, 1.2) | lexi | occ 列距离量到的是膨胀后的边界，半宽按它扣掉 0.10 |
#      | `v2f` | (0.15, 1.2)、(0.15, 0.8)、(0.12, 0.55)* | near | 窄半宽加分级 |
#
#      选变体的规则（写进入口脚本，自动执行）：撞在前多于 3/40 的臂出局；剩下的取到点成功 SR(stop) 最高的；与最高相差不超过 1 集的算并列，并列取表里靠前的。
#      闸门：选中的变体比 `v1` 在同一批 40 条上多到点 3 集以上才进正式阶段；否则停，DONE 写明「开发路线上 v2 没有比 v1 好」，照实报告。
#   2. 正式阶段（变体选定后冻结，写进 `$OUT/selected.json`，正式阶段只读它）：
#      留出路线 × 3 集：`forward+route`、`rule-occ+route`（v1）、`rule2-occ+route`（主臂）、`rule2-depth+route`；
#      正式 40 条 × 3 集：`rule-occ+route`（v1 重跑，先与 K1 的结果同键对拍，应逐集一致，不一致就停）、`rule2-occ+route`、`rule2-depth+route`；
#      室外 24 条 × 5 集（同 K1 的 seed）：`rule2-occ+route`、`rule2-depth+route`；v1 取 K1 的结果按键配对。
#      depth 臂的 levels：选中的是 `v2e` 或 `v2f` 时，depth 臂用同结构的 0.25 一档（`v2a` 或 `v2d`），因为渲染深度量到的是表面，不是膨胀后的边界。
# - 预注册（写进入口脚本头，起跑前定死；主会话 11-11 定，不许改）：
#   主终点（留出路线，`rule2-occ+route` 对 `rule-occ+route`，同路线同 seed 逐集配对）：到点成功率 SR(stop) 高 0.10 以上，精确 McNemar 双侧 p < 0.05；
#   主臂碰撞率（整集任一步）不高于 0.10，撞在前不高于 0.08。
#   证伪：到点成功率的差低于 0.05，或主臂碰撞率高于 0.15。含义是两侧搜索加分级通道补不上室内的缺口，下一步要的是更细的航向或更多几何，不是规则顺序。
#   落在两者之间记「介于预测与证伪之间」。
#   次终点（正式 40 条，同样的比较，只报告）：另报主臂是否到 0.80（K1 没达到的那条预测）。
#   室外（不劣）：主臂到点成功不低于 0.93；对 v1 逐集配对，「v1 成而 v2 败」的集数减去「v2 成而 v1 败」的集数不超过 5；碰撞率不高于 0.10。
#   depth 臂只报告。判读口径沿用 K1（SR(stop)、碰撞率含到点之后、点估计对阈值并列 Wilson 95% 区间）。
# - 另报：失败形态（沿用 K1 的分类，加 `V2-stuck` 步占比、各级 `V2-fwd` 的占比、每集转向步占比与平均步数）；按路线长度分桶，10 m 以上单列；出生即碰撞的集单列；
#   导引目标点不前进的集数；以 `V2-stuck` 收尾的集逐集列出位置与到占据的最小间隙（这些是规则顺序之外的缺口）。
# ---------------- 任务卡原文结束 ----------------
#
# 运行（正式实验不需要任何环境变量）：
#   cd /root/claude-code-backup-20261028/code/wt/p12-k3 && setsid nohup bash experiments/2026-11-11-rule-v2-3dgs/run.sh > /dev/null 2>&1 < /dev/null &
#   没有 GPU（nvidia-smi -L 列不出下面 GPU 的 UUID）时脚本立即退出，DONE 写「无卡模式，未起跑」，不退回 CPU。
#   formal 拒绝带着覆盖配置的环境变量起跑（OUT_ROOT / ROUTES_LIMIT / K_* / INDOOR_SCENES / OUTDOOR_SCENES / DRIVER / K1_ROOT / VARIANTS ...），冒烟与干跑才允许覆盖：
#   冒烟（有卡，单车道，不拿 3DGS 池的锁；2026-11-11 23:47 实测 2 分 3 秒）：
#     SMOKE=1 NO_GATE=1 FREEZE=tree LANES=1 INDOOR_SCENES=sage3d_840865 OUTDOOR_SCENES=callinstrasse ROUTES_LIMIT=1 K_DEV=1 K_HOLDOUT=1 K_FORMAL40=1 K_OUTDOOR=1 MIN_HOLDOUT=1 \
#       OUT_ROOT=<草稿目录> setsid nohup nice -n 10 bash run.sh > /dev/null 2>&1 < /dev/null &
#   干跑（无卡，桩驱动，不渲染）：DRY_RUN=1 FREEZE=tree DRIVER=<dry_run_driver.py> K1_ROOT=<桩 K1 目录（dry_run_driver.py --make-k1 造）> OUT_ROOT=<草稿目录> WATCH_S=0.3 setsid -w bash run.sh（test_run_chain.py 就是这样跑的）
# 进度：cat $OUT_ROOT/phase.txt；tail -f $OUT_ROOT/run.log；每场景日志 $OUT_ROOT/<dev|holdout|formal40|outdoor>/driver-<场景>.log（grep -v '^step ' 去掉逐步行）。
#   结束落 $OUT_ROOT/DONE（rc=…）、summary.md / summary.json（汇总）、dev_selection.md、selected.json（过闸才有）、k1_match.md（F0）。
# 排队：dev 之前阻塞式 flock 3DGS 池的锁 $D/tmp/phase12-cards/gs3d.lock（持有到脚本退出；不排 machine.lock，那把只管 Gazebo）；排队期间 phase.txt 写 queued。
#   拿到锁之后（冒烟、干跑是进入 dev 之前）与结束时各记一次 cgroup 的 usage_usec 与 nr_throttled：$OUT_ROOT/cgroup.txt（行首「起」/「止」）。
# 停：kill -- -$(cat $OUT_ROOT/run.pid)（只按 PGID，不用 pkill -f）。产出超过 SIZE_LIMIT_MB（1500）时守护自己停并写 DONE rc=5。
# 续跑：同一条命令重起；沿用已有的冻结副本 $OUT_ROOT/code；评完的路线（done.json 集数够且无失败）跳过，半成品改名 *.incomplete-<时刻> 后重跑；
#   routes 文件与上次不同、config.json 与上次不同、selected.json 与重算的选择不同都拒绝续跑。$OUT_ROOT/.lock 互斥：不会重复启动。
set -uo pipefail
SELF=$(readlink -f "${BASH_SOURCE[0]}"); HERE=$(dirname "$SELF")
# 起跑前记下哪些「可覆盖」的变量被环境预先设了（配置区的 ${VAR:-默认} 会把它们吃掉）；formal 用它拒绝带着覆盖起跑。
# LANES（并行度）与 WT 不在名单里：只影响速度 / 检出位置，起跑记录写进 inputs.txt。冻结副本的第二阶段继承第一阶段的结果。
if [[ -z "${FROZEN:-}" ]]; then
  OVERRIDDEN=""
  for v in OUT_ROOT SIZE_LIMIT_MB SCENES_DIR INDOOR_SCENES OUTDOOR_SCENES PILOT_ROUTES K1_ROOT ROUTES_LIMIT K_DEV K_HOLDOUT K_FORMAL40 K_OUTDOOR ROUTES_OUTDOOR \
           DEV_NEW_PER_SCENE HOLDOUT_PER_SCENE MIN_HOLDOUT DRIVER VARIANTS DEPTH_MAP NO_GATE PY FREEZE WATCH_S; do
    [[ -n "${!v+x}" ]] && OVERRIDDEN="$OVERRIDDEN $v"
  done
  export OVERRIDDEN
fi

# ---------------------------------------------------------------- 配置区（改这里；${VAR:-默认} 的项可被环境变量覆盖，其余写死）
D=/root/rivermind-data/uav_rgb
OUT_ROOT=${OUT_ROOT:-$D/exp-rule-v2-3dgs-20261111}                  # 产出根目录（数据盘）；阶段目录 $OUT_ROOT/{dev,holdout,formal40,outdoor}
PY=${PY:-$D/3dgs-venv/bin/python}                                   # torch 2.7.1 + gsplat 1.5.3；选路线、选变体、汇总只用标准库，也用它
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740                        # 机器上只有这一张卡（RTX 2080 Ti）；每次起跑前 nvidia-smi -L 核对
LANES=${LANES:-3}                                                   # 同时跑的 3DGS 驱动进程数：12 核，卡上只有本实验；K1 三车道峰值显存 14 GB（卡 22.5 GB）；只影响速度
SIZE_LIMIT_MB=${SIZE_LIMIT_MB:-1500}                                # K1 同值（按 0.46 MB/集推算，本实验约 1,700 集 ≈ 0.8 GB）；超了就停下报告
WATCH_S=${WATCH_S:-15}                                               # 体积与显存守护的采样间隔（秒）；干跑测试里调小
SCENES_DIR=${SCENES_DIR:-$D/3dgs-scenes}                            # 每个场景 <SCENES_DIR>/<场景>/scene.json
INDOOR_SCENES=${INDOOR_SCENES:-"sage3d_841150 sage3d_840865 sage3d_841012 sage3d_840139 sage3d_839992"}   # 试点的 5 个室内场景
OUTDOOR_SCENES=${OUTDOOR_SCENES:-"garden wanderland_1MlUucug bicycle callinstrasse"}                      # K1 / bearing-guard 的 4 个室外场景与顺序
PILOT_ROUTES=${PILOT_ROUTES:-$D/exp-sage3d-pilot-20261110/routes_selected.json}   # 正式 40 条 = 试点选定的 40 条（也是 K1 的正式室内路线）
K1_ROOT=${K1_ROOT:-$D/exp-rule-oracle-3dgs-20261110}                # K1 的输出（只读）：开发路线、正式室内 / 室外的 rule-occ+route 结果
K1_DEV_ROUTES=$K1_ROOT/dev/indoor/routes_dev.json                   # K1 的 20 条开发路线（进本实验的开发 40 条）
# 路线份数
DEV_NEW_PER_SCENE=${DEV_NEW_PER_SCENE:-4}                           # 开发路线每场景新取 4 条（任务卡）
HOLDOUT_PER_SCENE=${HOLDOUT_PER_SCENE:-8}                           # 留出路线每场景新取 8 条（任务卡），不够就有多少取多少
MIN_HOLDOUT=${MIN_HOLDOUT:-24}                                      # 留出合计少于 24 条就停下报告（任务卡）
# 开发阶段（任务卡）
K_DEV=${K_DEV:-1}                                                   # 每臂每路线 1 集
SEED0_DEV=9000                                                      # seed = 9000 + 100·路线下标（同 K1 的开发 seed，K1 的 20 条开发路线可与 K1 的 v1 结果逐集对照）
# 留出路线（任务卡）
K_HOLDOUT=${K_HOLDOUT:-3}                                           # 每路线 3 集
SEED0_HOLDOUT=5000                                                  # seed = 5000 + 100·路线下标 + k
# 正式 40 条（任务卡：K1 的同一批）
K_FORMAL40=${K_FORMAL40:-3}                                         # 每路线 3 集，40 路线共 120 集
SEED0_FORMAL40=8000                                                 # seed = 8000 + 100·路线下标 + k，与 K1 / 试点相同，v1 才能与 K1 逐集对拍
# 室外（同 K1）
K_OUTDOOR=${K_OUTDOOR:-5}                                           # 每路线 5 集，24 路线共 120 集
SEED0_OUTDOOR=7000                                                  # 同 K1 与 bearing-guard
JITTER_OUTDOOR=0.5                                                  # 同 K1
LOOKAHEAD_OUTDOOR=3.0                                               # 同 K1
ROUTES_OUTDOOR=${ROUTES_OUTDOOR:-"0 1 2 3 4 5"}                     # 每场景 6 条路线（路线下标）
# 室内共用（同 K1）
JITTER_INDOOR=0.2                                                   # 出生 ±0.2 m / ±10°：室内净空 0.2–0.4 m
LOOKAHEAD_INDOOR=1.5                                                # 路线前视距离（同试点与 K1）
STEP_CAP_INDOOR=100                                                 # 步数封顶（SITL 起始 50% 电量约 100 步）
ROUTES_LIMIT=${ROUTES_LIMIT:-}                                      # 冒烟用：每类只取前 N 条路线；空 = 全部
# 共用（同 K1）
HOLD=1.0                                                            # 动作保持时长（任务卡：hold 1.0）
STEP_FRAC=0.5; STEP_PAD=15                                          # 步数 = ceil(max_steps × 0.5) + 15（同前几个实验）
GUARD_DEG=45.0                                                      # forward+route 的护栏阈值（同前几个实验）
DROP_TRACE=true                                                     # 每集删 trace.jsonl（只有视频渲染要它）
# 规则参数（冻结）
YAW_STEP=31.7                                                       # 3DGS 动力学在 hold 1.0 下实测的转角步中位数（K1：22,988 个转向步中位数 31.68°，取一位小数；任务卡 yaw_step_deg 31.7）
V1_R_SAFE=0.25                                                      # K1 冻结值（开发路线上选定，14/20）
V1_D_LOOK=1.2                                                       # K1 冻结值
# 变体表（任务卡原文；一行一个：名字|r_safe:d_look[*],…|次序；带 * 的级只许静止起步）。test_run_sh.py 核对它与任务卡的表一致
VARIANTS=${VARIANTS:-"v2a|0.25:1.2|lexi
v2b|0.25:1.2,0.25:0.8|lexi
v2c|0.25:1.2,0.25:0.8|near
v2d|0.25:1.2,0.25:0.8,0.20:0.55*|near
v2e|0.15:1.2|lexi
v2f|0.15:1.2,0.15:0.8,0.12:0.55*|near"}
DEPTH_MAP=${DEPTH_MAP:-"v2e=v2a v2f=v2d"}                           # depth 臂的 levels：选中 v2e / v2f 时用同结构的 0.25 一档（任务卡）
# 选变体与闸门（任务卡）
MAX_COLLIDE_FIRST=3                                                 # 撞在前多于 3 集出局（读成绝对集数，见上「执行者补的口径」1）
TIE_SLACK=1                                                         # 与最高相差不超过 1 集算并列，并列取表里靠前的
GATE_MIN_GAIN=3                                                     # 选中的比 v1 多到点至少 3 集才进正式阶段
NO_GATE=${NO_GATE:-0}                                               # 1 = 闸门不过也往下走：只给冒烟用（路线太少闸门必然不过），formal 拒绝
GS3D_LOCK=$D/tmp/phase12-cards/gs3d.lock                            # 3DGS 池的锁（空文件）：2026-11-11 23:55 起 3DGS 批量实验拿它（公共规则 §4），不再排 machine.lock（只管 Gazebo）；阻塞式 flock，持有到本脚本退出；冒烟、干跑不拿
CF_SITL_PROC=$D/tmp/build-cf-sitl-lib31/cf_sitl_proc                # profile cf_sitl_v1:1bc8fedf:lib31，不要动这个目录
GSPLAT_EXT=$D/torch-cache/extensions-sm75                           # 已按 sm_75 编好，不应重编
# ----------------------------------------------------------------

WT=$(cd "$HERE/../.." && pwd)                                       # 第一阶段 = 起跑的检出；第二阶段（FROZEN=1）= $OUT_ROOT/code 冻结副本
EXPDIR=$HERE
DRIVER=${DRIVER:-$WT/experiments/2026-11-10-bearing-guard-3dgs/driver.py}   # 共用的驱动（同一个检出里的版本）；只有干跑才覆盖
DEFAULT_DRIVER=$WT/experiments/2026-11-10-bearing-guard-3dgs/driver.py
[[ "$(ps -o pgid= -p $$ | tr -d ' ')" == "$$" ]] || { echo "请用 setsid nohup bash run.sh 起（停进程按 PGID，要求本进程是组长）" >&2; exit 2; }
mkdir -p "$OUT_ROOT"
log() { echo "$(date '+%F %T') $*"; }
CG_STARTED=0
cgroup_note() {       # $1 = 起 | 止：当前 cgroup 的 usage_usec 与 nr_throttled（公共规则 §4：守护脚本起止各记一次）；写 run.log 与 $OUT_ROOT/cgroup.txt
  local f=/sys/fs/cgroup/cpu.stat u t
  u=$(awk '$1=="usage_usec"{print $2}' "$f" 2>/dev/null); t=$(awk '$1=="nr_throttled"{print $2}' "$f" 2>/dev/null)
  echo "$1 $(date '+%F %T') usage_usec=${u:-NA} nr_throttled=${t:-NA}" | tee -a "$OUT_ROOT/cgroup.txt"
}
cgroup_start() { cgroup_note 起; CG_STARTED=1; }
cgroup_end() { [[ "$CG_STARTED" == 1 ]] || return 0; CG_STARTED=0; cgroup_note 止; }       # 只在记过「起」之后记「止」；先于 DONE 落盘
fail_done() { log "$1"; cgroup_end; echo "rc=$2 $(date '+%F %T') $1" > "$OUT_ROOT/DONE"; exit "$2"; }
phase() { echo "$1 $(date '+%F %T')" > "$OUT_ROOT/phase.txt"; log "阶段：$1"; }
on_term() { cgroup_end; [[ -s "$OUT_ROOT/DONE" ]] || echo "rc=143 $(date '+%F %T') 被终止（kill -- -PGID）；已完成的路线下次起跑时跳过" > "$OUT_ROOT/DONE"; exit 143; }

if [[ -z "${FROZEN:-}" ]]; then
  # ======== 第一阶段：互斥、无卡检查、覆盖检查、冻结代码，然后从冻结副本重新 exec
  exec 9> "$OUT_ROOT/.lock"
  flock -n 9 || { echo "已有实例在跑（$OUT_ROOT/.lock），本次退出" >&2; exit 3; }
  exec >> "$OUT_ROOT/run.log" 2>&1
  trap on_term TERM INT
  rm -f "$OUT_ROOT/DONE"
  echo $$ > "$OUT_ROOT/run.pid"
  phase start
  log "开始 PGID=$$ WT=$WT OUT_ROOT=$OUT_ROOT LANES=$LANES SMOKE=${SMOKE:-0} DRY_RUN=${DRY_RUN:-0}"
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    [[ "$DRIVER" != "$DEFAULT_DRIVER" && -s "$DRIVER" ]] || fail_done "DRY_RUN=1 必须同时用 DRIVER=<桩驱动.py> 覆盖真驱动（不许无卡跑真渲染）" 2
  elif ! nvidia-smi -L 2>/dev/null | grep -q "$GPU"; then
    fail_done "无卡模式，未起跑（nvidia-smi -L 里没有 $GPU；不退回 CPU）" 4
  fi
  if [[ "${SMOKE:-0}" != 1 && "${DRY_RUN:-0}" != 1 && -n "${OVERRIDDEN:-}" ]]; then
    fail_done "formal 不接受环境变量覆盖：$OVERRIDDEN（冒烟请加 SMOKE=1，干跑请加 DRY_RUN=1）" 2
  fi
  DIRTY=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)
  if [[ "$DIRTY" != 0 && "${SMOKE:-0}" != 1 && "${DRY_RUN:-0}" != 1 ]]; then
    fail_done "已跟踪文件有 $DIRTY 个未提交改动：只从已提交的代码起跑（git status）" 2
  fi
  HEAD_NOW=$(git -C "$WT" rev-parse HEAD)
  if [[ -s "$OUT_ROOT/code_head.txt" && -d "$OUT_ROOT/code" ]]; then
    log "沿用已有的冻结副本 $OUT_ROOT/code（HEAD $(cat "$OUT_ROOT/code_head.txt")；当前检出 HEAD $HEAD_NOW）"
  else
    rm -rf "$OUT_ROOT/code" "$OUT_ROOT/code_head.txt"
    mkdir -p "$OUT_ROOT/code"
    if [[ "${FREEZE:-git}" == tree ]]; then      # 只给冒烟 / 干跑：冻结当前工作树（含未提交的改动），方便不提交就验改动；formal 一律 git archive HEAD
      [[ "${SMOKE:-0}" == 1 || "${DRY_RUN:-0}" == 1 ]] || fail_done "FREEZE=tree 只给冒烟与干跑用" 2
      (cd "$WT" && git ls-files -z --cached --others --exclude-standard | tar --null --ignore-failed-read -T - -cf -) | tar -x -C "$OUT_ROOT/code" || fail_done "冻结工作树失败" 2
      echo "$HEAD_NOW+tree" > "$OUT_ROOT/code_head.txt"
      echo "git HEAD $HEAD_NOW，冻结的是工作树（FREEZE=tree，含未提交改动 $DIRTY 个已跟踪文件）" > "$OUT_ROOT/code_state.txt"
    else
      git -C "$WT" archive HEAD | tar -x -C "$OUT_ROOT/code" || fail_done "git archive 失败" 2
      echo "$HEAD_NOW" > "$OUT_ROOT/code_head.txt"
      echo "git HEAD $HEAD_NOW 已跟踪文件未提交改动 $DIRTY 个（冻结副本只含已提交内容）" > "$OUT_ROOT/code_state.txt"
    fi
    log "冻结代码到 $OUT_ROOT/code（HEAD $HEAD_NOW，FREEZE=${FREEZE:-git}，未提交改动 $DIRTY 个）"
  fi
  [[ -s "$OUT_ROOT/code/experiments/2026-11-11-rule-v2-3dgs/run.sh" ]] || fail_done "冻结副本里没有 run.sh（入口没提交？）" 2
  export FROZEN=1 DIRTY HEAD_NOW
  exec bash "$OUT_ROOT/code/experiments/2026-11-11-rule-v2-3dgs/run.sh"
fi

# ======== 第二阶段（冻结副本里）：继承第一阶段的锁（fd 9）与日志重定向
trap on_term TERM INT
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT_ROOT/tmp   # 根分区只剩 3 GB，临时文件放数据盘
export TORCH_EXTENSIONS_DIR=$GSPLAT_EXT CF_SITL_PROC               # 其余环境与 K1 的入口一致（不设线程数变量），F0 才能逐集复现 K1
unset TORCH_CUDA_ARCH_LIST                                          # 带着旧值会让 gsplat 按别的架构重编
mkdir -p "$OUT_ROOT/tmp"
SMOKE=${SMOKE:-0}; DRY_RUN=${DRY_RUN:-0}
[[ "$SMOKE" == 1 || "$DRY_RUN" == 1 ]] || [[ "$NO_GATE" == 0 ]] || fail_done "NO_GATE 只给冒烟用" 2
phase preflight
log "冻结副本 WT=$WT；代码 HEAD $(cat "$OUT_ROOT/code_head.txt")；LANES=$LANES"
[[ "$DRY_RUN" == 1 ]] || log "GPU 占用（启动前）：$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i "$GPU" 2>&1)"

CLASSES_INDOOR="dev holdout formal40"
for f in "$WT/sim/3dgs/guard.py" "$WT/sim/3dgs/rule_oracle.py" "$WT/sim/tools/rule_policy.py" "$DRIVER" "$EXPDIR/select_routes_v2.py" "$EXPDIR/select_variant.py" \
         "$EXPDIR/check_k1_match.py" "$EXPDIR/build_config.py" "$EXPDIR/variants.py" "$PILOT_ROUTES" "$K1_DEV_ROUTES" \
         "$K1_ROOT/formal/indoor/config.json" "$K1_ROOT/formal/outdoor/config.json"; do
  [[ -s "$f" ]] || fail_done "缺输入 $f" 2
done
if [[ "$DRY_RUN" != 1 ]]; then
  for f in "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so"; do [[ -s "$f" ]] || fail_done "缺输入 $f" 2; done
fi
for s in $INDOOR_SCENES $OUTDOOR_SCENES; do [[ -s "$SCENES_DIR/$s/scene.json" ]] || fail_done "缺场景 $SCENES_DIR/$s/scene.json" 2; done
cd "$WT" || fail_done "进不去 $WT" 2

# ---- 变量导出（build_config.py 读环境变量）
export REPO=$WT SCENES_DIR HOLD STEP_FRAC STEP_PAD GUARD_DEG DROP_TRACE CF_SITL_PROC YAW_STEP V1_R_SAFE V1_D_LOOK VARIANTS DEPTH_MAP INDOOR_SCENES OUTDOOR_SCENES \
       ROUTES_OUTDOOR ROUTES_LIMIT K_DEV K_HOLDOUT K_FORMAL40 K_OUTDOOR SEED0_DEV SEED0_HOLDOUT SEED0_FORMAL40 SEED0_OUTDOOR JITTER_INDOOR JITTER_OUTDOOR \
       LOOKAHEAD_INDOOR LOOKAHEAD_OUTDOOR STEP_CAP_INDOOR
export ROUTES_DEV=$OUT_ROOT/routes/routes_dev.json ROUTES_HOLDOUT=$OUT_ROOT/routes/routes_holdout.json ROUTES_FORMAL40=$OUT_ROOT/routes/routes_formal40.json
export PY DRIVER OUT_ROOT LANES

# ---- 变体表先校验（levels 逐级不放宽、名字唯一）：错了在排队之前就报
"$PY" - <<'PYEOF' || fail_done "变体表 / DEPTH_MAP 校验失败（看 run.log）" 2
import os, sys
sys.path.insert(0, os.path.join(os.environ["REPO"], "experiments", "2026-11-11-rule-v2-3dgs"))
import variants as vr
vs = vr.parse_variants(os.environ["VARIANTS"])
m = vr.parse_depth_map(os.environ["DEPTH_MAP"], vs)
print("变体表 OK：", [v["name"] for v in vs], "depth 映射：", {k: v for k, v in m.items() if k != v})
PYEOF

# ---- 路线：先写 routes.new 与已有的比，不同就拒绝（续跑时 scene.json 变了会让两批路线的结果混在一个目录里）
phase routes
rm -rf "$OUT_ROOT/routes.new"
"$PY" "$EXPDIR/select_routes_v2.py" --scenes-dir "$SCENES_DIR" --scenes "$INDOOR_SCENES" --pilot-routes "$PILOT_ROUTES" --k1-dev-routes "$K1_DEV_ROUTES" \
      --out-dir "$OUT_ROOT/routes.new" --dev-new-per-scene "$DEV_NEW_PER_SCENE" --holdout-per-scene "$HOLDOUT_PER_SCENE" --min-holdout "$MIN_HOLDOUT" \
      --step-frac "$STEP_FRAC" --step-pad "$STEP_PAD" --step-cap "$STEP_CAP_INDOOR"
rc=$?
[[ $rc -eq 3 ]] && fail_done "留出路线合计少于 $MIN_HOLDOUT 条，停下报告（routes.new/counts.md）" 3
[[ $rc -eq 0 ]] || fail_done "选路线失败 rc=$rc（看 run.log；2 = 自检或重叠）" 2
if [[ -d "$OUT_ROOT/routes" ]]; then
  for f in routes_dev.json routes_holdout.json routes_formal40.json; do
    cmp -s "$OUT_ROOT/routes/$f" "$OUT_ROOT/routes.new/$f" || fail_done "$f 与上次不同（scene.json 或选路线参数变了）；新的在 $OUT_ROOT/routes.new，确认后改名旧目录再跑" 2
  done
  rm -rf "$OUT_ROOT/routes.new"
else
  mv "$OUT_ROOT/routes.new" "$OUT_ROOT/routes"
fi

# ---- 配置文件：先写 .new 与已有的比，不同就拒绝；F0 与正式阶段对 formal40 写不同的臂表，同一目录
write_config() {      # $1 类别，$2 目标 config.json，其余参数原样传给 build_config.py（--selected / --only）
  local cls=$1 target=$2; shift 2
  mkdir -p "$(dirname "$target")"
  "$PY" "$EXPDIR/build_config.py" "$cls" "$target.new" "$@" || fail_done "写 $cls 的 config.json 失败（看 run.log）" 2
  if [[ -s "$target" ]] && ! cmp -s "$target" "$target.new"; then
    fail_done "$target 与重算的不同（输入或代码变了）；新的在 $target.new，确认后改名旧目录再跑" 2
  fi
  mv -f "$target.new" "$target"
}

cp "$SELF" "$OUT_ROOT/run.sh.snapshot"
{ [[ "$DRY_RUN" == 1 ]] || sha256sum "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so"
  sha256sum "$WT/sim/dynamics/e1_mini.json" "$WT/sim/3dgs/guard.py" "$WT/sim/3dgs/rule_oracle.py" "$WT/sim/tools/rule_policy.py" "$WT/sim/3dgs/collision.py" \
            "$WT/sim/3dgs/planner.py" "$DRIVER" "$EXPDIR"/*.py "$PILOT_ROUTES" "$K1_DEV_ROUTES" "$OUT_ROOT"/routes/*.json
  for s in $INDOOR_SCENES $OUTDOOR_SCENES; do sha256sum "$SCENES_DIR/$s/scene.json"; done
  cat "$K1_ROOT/formal/DONE" 2>/dev/null | sed 's/^/K1 formal DONE: /'
  echo "git HEAD $(cat "$OUT_ROOT/code_head.txt") 未提交改动 ${DIRTY:-?} 个已跟踪文件（起跑时的检出；冻结副本只含已提交内容）"
  echo "起跑环境覆盖：${OVERRIDDEN:-无}  SMOKE=$SMOKE DRY_RUN=$DRY_RUN NO_GATE=$NO_GATE LANES=$LANES"; } > "$OUT_ROOT/inputs.txt" 2>&1
[[ "$DRY_RUN" == 1 ]] || nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" > "$OUT_ROOT/gpu.txt" 2>&1 || true   # 只查询，不起计算

write_config dev "$OUT_ROOT/dev/config.json"

# ---- 任务执行：一个 (类别, 场景) 一个驱动进程（场景加载一次，跑完该类的全部臂与路线）；LANES 并行；每 WATCH_S（15）秒看一次体积与显存
run_task() {
  local cls=$1 s=$2 rc
  log "==== [$cls] $s 开始"
  "$PY" "$DRIVER" "$OUT_ROOT/$cls" "$s" >> "$OUT_ROOT/$cls/driver-$s.log" 2>&1; rc=$?
  log "==== [$cls] $s 结束 rc=$rc"
  return $rc
}
export -f log run_task
PEAK=0
run_tasks() {         # stdin：每行「类别 场景」；返回 xargs 的 rc（123 = 有任务非零退出）
  local list; list=$(cat)
  log "任务 $(echo "$list" | wc -l) 个，并行 $LANES：$(echo "$list" | tr '\n' ';')"
  ( printf '%s\n' "$list" | xargs -P "$LANES" -L 1 bash -c 'run_task "$0" "$1"' ) &
  local XPID=$! tick=0 mb gpu_raw gpu_mb=0
  while kill -0 "$XPID" 2>/dev/null; do
    mb=$(du -sm "$OUT_ROOT" 2>/dev/null | cut -f1)
    if [[ "$DRY_RUN" != 1 ]]; then
      gpu_raw=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU" 2>/dev/null | head -1); gpu_mb=${gpu_raw% MiB}
      [[ "$gpu_mb" =~ ^[0-9]+$ ]] || gpu_mb=0
      (( gpu_mb > PEAK )) && PEAK=$gpu_mb
    fi
    (( tick % 8 == 0 )) && log "显存占用 ${gpu_mb} MiB（采样峰值 ${PEAK} MiB，整张卡）；产出 ${mb} MB"       # 每 8 次采样一行（默认 8 × 15 s = 2 分钟）
    tick=$((tick + 1))
    if [[ ${mb:-0} -gt $SIZE_LIMIT_MB ]]; then
      log "产出 ${mb} MB 超过 ${SIZE_LIMIT_MB} MB，停止整个进程组（任务书：超了停下报告）"
      echo "rc=5 $(date '+%F %T') 产出 ${mb} MB 超过 ${SIZE_LIMIT_MB} MB，已停（kill -- -$$）" > "$OUT_ROOT/DONE"
      kill -TERM -- -$$
    fi
    sleep "$WATCH_S"
  done
  wait "$XPID"
}
tasks_for() {         # $1 类别，$2.. 场景
  local cls=$1; shift
  mkdir -p "$OUT_ROOT/$cls"
  for s in "$@"; do echo "$cls $s"; done
}

# ---- 排队：3DGS 池的锁（阻塞式，持有到脚本退出）；冒烟与干跑不拿
if [[ "$SMOKE" != 1 && "$DRY_RUN" != 1 ]]; then
  phase queued
  log "等 3DGS 池的锁 $GS3D_LOCK"
  exec 8>> "$GS3D_LOCK"; flock 8
  log "拿到 3DGS 池的锁 $GS3D_LOCK（持有到本脚本退出）"
  nvidia-smi -L 2>/dev/null | grep -q "$GPU" || fail_done "无卡模式，未起跑（拿到 3DGS 池的锁后 nvidia-smi -L 里没有 $GPU）" 4
fi
T_START=$(date +%s)
cgroup_start

# ================================================================ 阶段 1：开发路线（v1 + 6 个 v2 变体）
if [[ ! -e "$OUT_ROOT/dev/.tasks_ok" ]]; then
  phase dev
  run_tasks < <(tasks_for dev $INDOOR_SCENES); rc=$?
  log "dev 驱动全部结束 rc=$rc；产出 $(du -sm "$OUT_ROOT" | cut -f1) MB；显存峰值 ${PEAK} MiB"
  [[ $rc -eq 0 ]] || fail_done "dev 有驱动任务失败 rc=$rc（看 $OUT_ROOT/dev/driver-*.log）" 7
  touch "$OUT_ROOT/dev/.tasks_ok"
fi

# ================================================================ 阶段 2：自动选变体与闸门
phase select
SEL_ARGS=(--dev "$OUT_ROOT/dev" --out-dir "$OUT_ROOT" --variants "$VARIANTS" --depth-map "$DEPTH_MAP" --max-collide-first "$MAX_COLLIDE_FIRST" \
          --tie-slack "$TIE_SLACK" --gate-min-gain "$GATE_MIN_GAIN")
[[ "$NO_GATE" == 1 ]] && SEL_ARGS+=(--no-gate)
"$PY" "$EXPDIR/select_variant.py" "${SEL_ARGS[@]}" >> "$OUT_ROOT/select_variant.log" 2>&1; rc=$?
log "select_variant rc=$rc：$(tail -3 "$OUT_ROOT/select_variant.log" | tr '\n' ' ' | cut -c1-300)"
if [[ $rc -eq 20 ]]; then
  fail_done "开发路线上 v2 没有比 v1 好（dev_selection.md；选中变体比 v1 多到点不足 $GATE_MIN_GAIN 集，或全部变体撞在前超限）" 20
fi
[[ $rc -eq 0 ]] || fail_done "select_variant 失败 rc=$rc（看 select_variant.log）" 3
SELECTED=$OUT_ROOT/selected.json
[[ -s "$SELECTED" ]] || fail_done "select_variant 没写 selected.json" 3
log "选中：$(tr -d '\n ' < "$SELECTED" | cut -c1-240)"

# ================================================================ 阶段 3：F0（正式 40 条上 v1 重跑，与 K1 逐集对拍）
if [[ ! -e "$OUT_ROOT/formal40/.f0_ok" ]]; then
  phase f0
  mkdir -p "$OUT_ROOT/formal40"
  write_config formal40 "$OUT_ROOT/formal40/config.f0.json" --selected "$SELECTED" --only rule-occ+route
  cp -f "$OUT_ROOT/formal40/config.f0.json" "$OUT_ROOT/formal40/config.json"
  run_tasks < <(tasks_for formal40 $INDOOR_SCENES); rc=$?
  log "F0 驱动全部结束 rc=$rc；显存峰值 ${PEAK} MiB"
  [[ $rc -eq 0 ]] || fail_done "F0 有驱动任务失败 rc=$rc（看 $OUT_ROOT/formal40/driver-*.log）" 7
  "$PY" "$EXPDIR/check_k1_match.py" --new "$OUT_ROOT/formal40" --k1 "$K1_ROOT/formal/indoor" --out-json "$OUT_ROOT/k1_match.json" --out-md "$OUT_ROOT/k1_match.md" \
        >> "$OUT_ROOT/check_k1_match.log" 2>&1; rc=$?
  log "check_k1_match rc=$rc：$(grep -E '判定|本次' "$OUT_ROOT/k1_match.md" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
  [[ $rc -ne 1 ]] || fail_done "F0 对拍不一致：v1 重跑与 K1 的结果不是逐集相同，停（$OUT_ROOT/k1_match.md）" 6
  [[ $rc -eq 0 ]] || fail_done "F0 对拍数据不全 rc=$rc（$OUT_ROOT/k1_match.md）" 3
  touch "$OUT_ROOT/formal40/.f0_ok"
fi

# ================================================================ 阶段 4：正式（留出路线、正式 40 条余下的臂、室外）
phase formal
write_config holdout "$OUT_ROOT/holdout/config.json" --selected "$SELECTED"
write_config formal40 "$OUT_ROOT/formal40/config.final.json" --selected "$SELECTED"
cp -f "$OUT_ROOT/formal40/config.final.json" "$OUT_ROOT/formal40/config.json"      # F0 用的单臂 config.json 换成完整的三臂版（已评完的 v1 臂按 done.json 跳过）
write_config outdoor "$OUT_ROOT/outdoor/config.json" --selected "$SELECTED"
run_tasks < <({ tasks_for holdout $INDOOR_SCENES; tasks_for formal40 $INDOOR_SCENES; tasks_for outdoor $OUTDOOR_SCENES; }); rc_all=$?
log "正式阶段驱动全部结束 rc=$rc_all；产出 $(du -sm "$OUT_ROOT" | cut -f1) MB；显存峰值 ${PEAK} MiB；本脚本拿锁后已用 $(( $(date +%s) - T_START )) s"
echo "$PEAK" > "$OUT_ROOT/gpu_peak_mib.txt"

# ================================================================ 阶段 5：汇总
phase summarize
msg="选中 $("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['variant'])" "$SELECTED" 2>/dev/null)"
if [[ -s "$EXPDIR/summarize.py" ]]; then
  "$PY" "$EXPDIR/summarize.py" "$OUT_ROOT" --k1-root "$K1_ROOT" > "$OUT_ROOT/summarize.log" 2>&1 \
    || { log "汇总失败（看 summarize.log）"; [[ $rc_all -ne 0 ]] || rc_all=3; }
  [[ -s "$OUT_ROOT/verdict.txt" ]] && msg="$msg；$(head -c 600 "$OUT_ROOT/verdict.txt" | tr '\n' ' ')"
else
  log "冻结副本里没有 summarize.py：结果已落盘，起跑后用已提交的 summarize.py 事后汇总"
  msg="$msg；未在链条里汇总"
fi
[[ $rc_all -eq 0 ]] || msg="$msg；有驱动任务失败（xargs rc=$rc_all），看 driver-*.log"
[[ $rc_all -eq 0 || $rc_all -eq 3 ]] || rc_all=7        # xargs 的 123 / 124 / 125 都按「有驱动任务失败」记 7；汇总失败 = 3
phase done
log "全部结束 rc=$rc_all"
cgroup_end
echo "rc=$rc_all $(date '+%F %T') $msg" > "$OUT_ROOT/DONE"
exit $rc_all
