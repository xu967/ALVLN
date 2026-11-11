#!/usr/bin/env bash
# 卡 E2E-0：可判定性审计。离线分析：特权 A* 教师的动作标签里，有多少能由端到端学生拿得到的信息决定，补哪一种输入能补上缺口。
# 卡：docs/briefs/2026-11-10-phase12-task-cards.md「卡 E2E-0」；背景：docs/PLAN_2026-11-10_phase12.md §10。
# 不起仿真、不训练学生、不改库代码、不拿任何池的锁。特征全部由仿真真值算，只用于分析（oracle），不进任何学生输入。
#
# 起跑前写下的预期（不是门槛，照实对照）：学生驱动的帧上，S1 的左右正确率不高于 0.70，S3 不低于 0.85；教师驱动的帧上 S1 与 S3 的整体准确率相差不到 5 个点；
# S5 比 S4 高不到 5 个点。S5 比 S4 高 10 个点以上，就说明端到端学生需要学出来的记忆，历史码不够。
#   （以上是任务卡原文，原样抄入。）
#
# 对上面这段话的预注册读法（2026-11-12 00:35 写下，起跑前）。写下时看过的只有：trunc 集的帧数、特征取值范围与对账结果；
# 一次冒烟跑（只含 trunc 的测试帧，每组 2 个种子、6 轮）的几行日志，用来确认三个阶段能连起来跑通。没有看过学生驱动的帧上的任何拟合结果。
# 卡没写清的三处，先定下来：
#   (a) 子集：教师驱动 = trunc 与 more_demos 的帧；学生驱动 = round_1、round_2、round_3 的帧（卡原文「学生驱动的状态 = round_{1,2,3}」；round_1 是 β=0.5 的混合执行，
#       纯学生执行的 round_2、round_3 另列为附加子集 student_r23）；画面里 4 m 内没有障碍 = 60° 视场的 20 列距离全是 4 m 封顶，各来源的帧合在一起取。
#   (b) 拟合池：主配置 pooled_w = 全部训练帧一起拟合、类别加权交叉熵，各子集只在测试帧上取数。卡没说拟合池，另跑三种敏感性配置，对照时都列：
#       pooled_u（不加权）、teacher_w（只用教师驱动的训练帧拟合）、student_w（只用学生驱动的训练帧拟合）。
#   (c) 「左右正确率」= 教师标签是 YAW_LEFT / YAW_RIGHT 的测试帧里，网络的 L、R 两个 logit 比出的方向对的比例（不要求先判成转向）；
#       「整体准确率」= 四类准确率；卡没写「S5 比 S4 高多少个点」用哪个指标与哪个子集，主读法用学生驱动帧上的 macro-F1（FORWARD / LEFT / RIGHT 三类的宏平均；
#       SAFE_NEUTRAL 在测试帧里只有个位数），整体准确率、左右正确率、该转时转了的比例以及另外两个子集上的差值同表列出。
# 逐条的主读法与判定（配置 pooled_w）：
#   E1  学生驱动的帧：lr_acc(S1) ≤ 0.70
#   E2  学生驱动的帧：lr_acc(S3) ≥ 0.85
#   E3  教师驱动的帧：|acc(S3) − acc(S1)| < 0.05
#   E4  学生驱动的帧：f1_frl(S5) − f1_frl(S4) < 0.05；E4'：该差 ≥ 0.10。差落在 [0.05, 0.10) 的话不属于卡里的任何一句，报告里照实写「落在两句之间」。
#   每条同时给点估计（3 个种子的均值与极差）与按测试几何簇自助法的 95% 区间；点估计与阈值的关系在区间里不稳定的，写「未能分辨」。
# 卡外附加（报告里标明）：S3c = {d, 三值航向码（W1-9 卡的 0/2/3）}，用来量「n 换成三值码会丢多少」；S7 = S6 + 真实航向 sin/cos，作天花板参照。
#
# 运行：mkdir -p $OUT && cd $OUT && setsid nohup nice -n 10 bash <仓库>/experiments/2026-11-12-decidability-audit/run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；结束落 $OUT/DONE（rc 与时刻）；结果表 $OUT/tables.md。停：kill -- -$(cat $OUT/run.pid)（只按 PGID）。
# 续跑：同一条命令重起；features.npz 与各配置的 stats_*.npz 已存在则跳过（FORCE=1 全部重算）。.lock 互斥，重复启动的那个立即退出。
# 资源：每个阶段只起 1 个 python 进程、1 个线程（nice 10）；不用 GPU（CUDA_VISIBLE_DEVICES 置空）。起止时记录 cgroup 的 usage_usec 与 nr_throttled。
set -u

# ---------------------------------------------------------------- 配置（改这里）
D=/root/rivermind-data/uav_rgb
WT=/root/claude-code-backup-20261028/code/wt/p12-e2e-0         # 本卡代码所在的工作树
PY=$D/teacher-venv/bin/python                                  # 只读借用：numpy 与 torch；不往里装包
OUT=$D/exp-decidability-audit-20261112                         # 全部输出
NICE=10
# 输入（只读）。名字=采集根目录，目录下是 train/val/test/<集号>/。名字要与下面的 *_SOURCES 对上。
FV=$D/exp-dagger-gz-20261109/full-1109
COLLECTIONS="trunc=$D/tmp/agent6-distill/trunc/col;more_demos=$FV/more_demos/collect;round_1=$FV/round_1/collect;round_2=$FV/round_2/collect;round_3=$FV/round_3/collect"
# 各数据集打包时的日志；提取结束后逐集、逐 split、逐类与它们对账，对不上就停
PACK_LOGS="$D/tmp/agent6-distill/trunc/preprocess.log;$FV/more_demos/ds.log;$FV/round_1/ds.log;$FV/round_2/ds.log;$FV/round_3/ds.log"
DROP_COLLISION=1                          # 两类日志都是 --drop-collision、没有 --exclude-collision-episodes
TEACHER_SOURCES=trunc,more_demos          # 教师驱动的帧
STUDENT_SOURCES=round_1,round_2,round_3   # 学生驱动的帧（卡里的口径）
STUDENT_PURE_SOURCES=round_2,round_3      # 纯学生执行（β=0）的轮，附加子集
# 拟合器（卡：两层各 64 个单元的 MLP，类别加权交叉熵，train 拟合、val 定停止轮数、test 报数，3 个种子）
SEEDS="0 1 2"
HIDDEN=64
LAYERS=2
LR=1e-3
BATCH=256
MAX_EPOCHS=80                             # val 加权交叉熵最低的那一轮；PATIENCE 轮没改善就停
PATIENCE=10
THREADS=1
FIT_CONFIGS=pooled_w,pooled_u,teacher_w,student_w
# 分箱核对：d 分 4 个扇区取最近距离、按 BIN_D_EDGES 分档；d_wide 分 6 个扇区、按 BIN_W_EDGES 分档（单位 m）
BIN_D_SECTORS=4
BIN_D_EDGES=1.0,2.5
BIN_W_SECTORS=6
BIN_W_EDGES=2.0
# 区间：测试几何簇自助法
BOOT_REPS=1000
BOOT_SEED=20261112
FORCE=0                                   # 1 = 忽略已有的 features.npz 与 stats_*.npz，全部重算
# ----------------------------------------------------------------

CODE=$WT/experiments/2026-11-12-decidability-audit
mkdir -p "$OUT/tmp"
export PYTHONDONTWRITEBYTECODE=1 TMPDIR="$OUT/tmp" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=   # 根分区只剩 3 GB，临时文件放数据盘
export AUDIT_OUT="$OUT" AUDIT_DATA_ROOT="$D" AUDIT_COLLECTIONS="$COLLECTIONS" AUDIT_PACK_LOGS="$PACK_LOGS" AUDIT_DROP_COLLISION="$DROP_COLLISION"
export AUDIT_TEACHER_SOURCES="$TEACHER_SOURCES" AUDIT_STUDENT_SOURCES="$STUDENT_SOURCES" AUDIT_STUDENT_PURE_SOURCES="$STUDENT_PURE_SOURCES"
export AUDIT_SEEDS="$SEEDS" AUDIT_HIDDEN="$HIDDEN" AUDIT_LAYERS="$LAYERS" AUDIT_LR="$LR" AUDIT_BATCH="$BATCH" AUDIT_MAX_EPOCHS="$MAX_EPOCHS"
export AUDIT_PATIENCE="$PATIENCE" AUDIT_THREADS="$THREADS" AUDIT_FIT_CONFIGS="$FIT_CONFIGS" AUDIT_FORCE="$FORCE"
export AUDIT_BIN_D_SECTORS="$BIN_D_SECTORS" AUDIT_BIN_D_EDGES="$BIN_D_EDGES" AUDIT_BIN_W_SECTORS="$BIN_W_SECTORS" AUDIT_BIN_W_EDGES="$BIN_W_EDGES"
export AUDIT_BOOT_REPS="$BOOT_REPS" AUDIT_BOOT_SEED="$BOOT_SEED"

exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
cg() { awk '/^(usage_usec|nr_periods|nr_throttled|throttled_usec) /{printf "%s=%s ", $1, $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null; }
echo "$(date '+%F %T') 开始 PGID=$$ nice=$NICE 配置=$FIT_CONFIGS 种子=[$SEEDS] cgroup: $(cg)"
{ echo "代码提交 $(git -C "$WT" rev-parse HEAD)；工作树改动：$(git -C "$WT" status --short | wc -l) 项"
  tr ';' '\n' <<< "$PACK_LOGS" | xargs sha256sum
  sha256sum "$CODE"/audit_features.py "$CODE"/audit_fit.py "$CODE"/audit_report.py "$CODE"/run.sh
  "$PY" -c 'import sys,numpy,torch; print("python", sys.version.split()[0], "numpy", numpy.__version__, "torch", torch.__version__, "cuda可见", torch.cuda.is_available())'
} > "$OUT/inputs.txt" 2>&1

run_stage() {
  local name=$1; shift
  echo "$(date '+%F %T') 阶段 $name 开始 cgroup: $(cg)"
  nice -n "$NICE" "$PY" "$@"
  local rc=$?
  echo "$(date '+%F %T') 阶段 $name rc=$rc cgroup: $(cg)"
  return $rc
}
fail() { echo "rc=$1 $(date '+%F %T')" > "$OUT/DONE"; echo "$(date '+%F %T') 失败 rc=$1，停"; exit "$1"; }

if [[ -f "$OUT/features.npz" && -f "$OUT/selection_check.json" && "$FORCE" != 1 ]]; then
  echo "$(date '+%F %T') features.npz 已存在，跳过 extract"
else
  run_stage extract "$CODE/audit_features.py" || fail $?
fi
run_stage fit "$CODE/audit_fit.py" fit || fail $?
run_stage bins "$CODE/audit_fit.py" bins || fail $?
run_stage report "$CODE/audit_report.py" || fail $?
echo "rc=0 $(date '+%F %T')" > "$OUT/DONE"
echo "$(date '+%F %T') 全部结束 rc=0 cgroup: $(cg)"
