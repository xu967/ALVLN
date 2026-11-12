#!/usr/bin/env bash
# 训练配方的稳定性筛选，第一部分（任务卡 E2E-R1，离线：只训练与离线推理，不起 Gazebo）。同一个守护进程：
#   闸门（参照配方复训种子 0，对账）→ R1 与 R2、R3（被训练器拒绝时改跑替代臂 R2e、R3e，见下面「起跑前发现」）各 3 个训练种子 → 逐轮 val 预测 →
#   末 10 轮平均与平均加校准 → 落盘。指标由 stability.py 另跑（stability.py report）。
# 背景：交叉审核 H（docs/reviews/2026-11-12-cross-review/review-h.md）的 H2、H3、H6：训练种子之间、相邻轮之间的不稳定大部分是「前进对转向」一个工作点在漂，
#   原因是恒定学习率的 Adam 作用在 19,200 维全连接上，加类权重 auto 引入的偏移。这里问：训练器现成的开关能不能在配方层把漂移消掉。
#
# 运行：RS_ 前缀的环境变量都不要设；cd $OUT 之外任何目录都行：
#   setsid nohup bash run_offline.sh > /dev/null 2>&1 < /dev/null &
#   起跑前必须是已提交的版本：脚本会拿自己与 CODE_HEAD（缺省 = SRC 的 HEAD）里的同名文件比哈希，不一致 rc=4。冻结代码 = git archive CODE_HEAD，解到 $OUT/code，
#   之后所有 Python 工具都从冻结副本里跑。续跑：同一条命令重起，已完成的训练 / 预测 / 平均跳过，半成品挪到 <名>.partial-<时刻>（不删），沿用起跑时的 CODE_HEAD。
# 进度：tail -f $OUT/run.log；每个 run 一个 runs/<臂>-s<种子>.log；结束落 DONE（rc、时间、原因）。
# 停：kill -- -$(cat $OUT/run.pid)，只按进程组，不用 pkill -f；停之前核对：ps -o lstart=,args= -p $(cat $OUT/run.pid) 的启动时刻晚于 ps -o lstart= -p 1，命令行是本脚本。
# DONE 的 rc：0 全部完成；3 已有实例（不写 DONE）；4 冻结代码或脚本版本不一致；5 拒绝起跑（正式运行遇到 RS_ 覆盖、数据集 sha 对不上）；11 有训练 / 预测 / 平均失败；
#   12 配置的 GPU 不在；14 等训练池锁超过 LOCK_MAX_S；20 闸门不过（复训与参照的末轮 val macro-F1 相差 >= 0.05，停下报告）；
#   21 其余全部完成，但卡里的 R2、R3 被训练器拒绝（SAFE 类权重 0，见下一段），已改跑替代臂 R2e、R3e（不是 0：下游只认 0 和 21 为「可以接着做」）；130 被信号终止。
#
# ---------------------------------------------------------------- 起跑前发现（2026-11-12 01:26，演练时撞上；发生在任何正式结果之前）
#   卡里的 R2 = `--class-weights 0,1,1,1`、R3 = `--class-weights 0,1,2,2` 被训练器在入口拒绝，rc=1，一个 batch 都没训：
#     「训练失败: --class-weights 给训练集存在的类 0 设了 0 权重（纯该类 batch 会除零）；用 auto 或正权重」（src/class_weights.cpp:199-206，resolve_class_weights）。
#   训练集里 SAFE 类有样本：训练器状态行写 9663 samples (8205 train / 574 dev)（runs/R0b-s0/metrics.jsonl 第 1 行），参照 run config.json 的 auto 权重 36.6294632 = N/(4·n_0)
#   （src/class_weights.cpp:52-62，N = 训练样本数），N = 8205 反推 n_0 = 56，其余三类 6885、639、625，四个数相加正好 8205；所以 0 权重不被接受。
#   （2026-11-12 02:32 更正：本段原先写「24,615 样本、SAFE 168 个」，与训练器状态行的 8205 不符（24,615 = 3 × 8205，来源没有追溯）；只改了这段注释里的数字，起跑时的脚本与结果不受影响。）
#   不改 C++、不改臂，所以做三件事：① 起跑后先用卡里的原配方各试训一次种子 0，把训练器的拒绝原话落进 blocked/R2.txt、blocked/R3.txt（证据）；
#   ② 被拒的臂记「未跑」，不算训练失败（DONE 用 rc=21 区别于全部成功的 0）；③ 另外跑两个标明「替代」的臂 R2e、R3e：把 SAFE 的 0 换成 1e-06，
#   其余权重和 --cosine-lr 与 R2、R3 逐项相同。替代的依据：SAFE 样本对加权 CE 的贡献与 0 权重的差是相对 1e-8 以下的量级（56·1e-6 / 8,149 ≈ 7e-9，R2e 口径；R3e 口径 56·1e-6 / 9,413 ≈ 6e-9），
#   而 SAFE logit 的梯度主要来自 8,149 个非 SAFE 样本的 w·p_0 项，不随这个 1e-6 变。等价性没有单独验证，由主会话确认；替代臂不进卡的过线判决与「第二部分加臂」的自动选择，
#   只在报告里单列。若训练器接受了卡里的配方（别的构建），照卡训满三个种子，不跑替代臂。
#   （2026-11-12 02:33 更新，第一部分出结果之后：R2e、R3e 都过线，「替代臂不进第二部分加臂」这一条改了：卡里的臂都没过线时，在过线的替代臂里按同一规则取一个加进第二部分，
#   结果里标明替代；这是看到第一部分结果之后的决定，不是预注册。理由与影响见 RESULT.md 1.5 第 3 条。过线判决本身仍只看卡里的臂与阈值，没有变。）
#
# ---------------------------------------------------------------- 预注册（第一部分，2026-11-12 起跑前写下；臂、配方、预期、通过线抄自任务卡 E2E-R1，不改）
#   数据同参照学生：$D/tmp/agent6-distill/trunc/ds/dataset.uavr。训练种子 0、1、2，30 轮，--save-every-epoch，增强关（增强的随机数种子写死、不随 --seed 变，审核 H8），
#   其余开关与参照配方（$D/exp-student-select-20261109/e1-noise/seed*/run/config.json）逐项相同，只改下表：
#     R0  参照配方：恒定学习率，类权重 auto。现成的三个 run 与它们的逐轮预测，不重训。另复训种子 0 一次，当闸门，并给「同配方同种子两次训练」的翻转率。
#     R1  加 --cosine-lr（类权重 auto）          R2  --cosine-lr，类权重 0,1,1,1（SAFE 不参与，三类不加权）          R3  --cosine-lr，类权重 0,1,2,2
#   预期（审核 H 的预测）：R1 的原始翻转率降到 0.08 以下（R0 是 0.21–0.24，全程 29 个相邻对的口径）。
#   通过线：不做事后校准时，末 10 轮翻转率不高于 0.06，并且三种子全一致率不低于 0.88。没有臂过线也照实报，第二部分只做工作点扫描。
#   起跑前补记的口径（任务卡对下面几处没有逐字规定，按最严的读法定下，起跑前写，结果出来后不改）：
#     1. 翻转率 = 同一训练种子相邻两轮在 val 574 帧上预测类别不同的帧占比，逐对算，取平均。「末 10 轮」= 第 21–30 轮内的 9 个相邻对（与 h1_bias_drift.py 的
#        slice(-9, None) 同）；另报「末 10 个相邻对」（第 20→30 轮）。判据取三个种子里最大的那个，均值另报。
#     2. 三种子全一致率 = 同一轮三个种子的预测在同一帧上完全相同的帧占比；判据口径 = 第 21–30 轮各轮的一致率取平均（与 h1 同，R0 为 0.671 / 校准后 0.899）。
#        另报末轮（第 30 轮）与末 10 轮平均权重（average_ckpt，不校准 / 校准）两种；各口径的判决不一致时并列写出。
#     3. 校准 = FORWARD logit 加 δ，使 val 上 FORWARD 预测占比等于 val 标签里的 FORWARD 占比（0.8728）；二分法同 h1_bias_drift.calib_pred 与 tools/average_ckpt.calibrate。
#     4. 多个臂同时过线时，第二部分只加一个臂：末 10 轮原始翻转率（三种子最大）最小的，并列取编号小的；其余过线的臂列在报告里交主会话定。
#
# 时间估计（实测，2026-11-12 烟测与演练）：单个训练独占时 1 轮 5.4–6 s（3 轮 16–18 s），30 轮约 3 分钟；3 个并行时每个进程 490 样本/秒、3 轮 50 s，30 轮约 8.4 分钟；
#   9 个训练分 3 批约 25 分钟，加闸门、逐轮预测与平均，整体 45 分钟上下。
set -u
export LC_ALL=C

# ---------------------------------------------------------------- 配置（改这里；RS_ 前缀的覆盖只在演练 RS_REHEARSAL=1 时允许）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=${RS_OUT:-$D/exp-recipe-stability-20261112}
SRC=${RS_SRC:-/root/claude-code-backup-20261028/code/wt/p12-e2e-r1}   # 代码源（分支 p12-e2e-r1）：只用来 git archive 冻结代码
CODE_HEAD=${RS_CODE_HEAD:-}                              # 冻结代码的提交；缺省 = 续跑时沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
TRAIN_BIN=${RS_TRAIN_BIN:-$D/tmp/build-check-main-1111/uavrgb_train}   # 11-11 全量检查的构建（main af5292f；src/ 自此到 f56d54f 无改动），只读地用
INFER=${RS_INFER:-$D/bin-snapshots/main-97ae04b/uavrgb_infer}          # 推理二进制快照（任务书指定）
DATA=${RS_DATA:-$D/tmp/agent6-distill/trunc/ds/dataset.uavr}           # 参照学生的训练集
DATA_SHA=534ca5904ea027b99a59137f1985a7b1b67646492bed124fb6bcae44309dfcac   # 参照 run config.json 里的 dataset_sha256
REF=${RS_REF:-$D/exp-student-select-20261109}            # 参照 run（e1-noise/seed*/run）与它们的逐轮预测（select/seed*），只读
GPU=${RS_GPU:-GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740}  # 11-11 容器重启后 uav_rgb 用的这张卡（RTX 2080 Ti），用户 11-11 19:23 指定
SEEDS=${RS_SEEDS:-"0 1 2"}                               # 训练种子：任务卡
EPOCHS=${RS_EPOCHS:-30}                                  # 30 轮：参照配方
BATCH=128                                                # 参照配方
LR=0.001                                                 # 参照配方（config.json 里是 float32 的 0.00100000005）
LAST_K=${RS_LAST_K:-10}                                  # 末 10 轮平均：tools/average_ckpt.py 的默认值，11-10 闭环只测过 10
PROCS=${RS_PROCS:-3}                                     # 训练池规则：组内最多 3 个训练进程（公共规则 §4）
TRAIN_LOCK=${RS_TRAIN_LOCK:-$D/tmp/phase12-cards/train.lock}   # 训练池的锁（阻塞式拿，持有到脚本退出）
LOCK_LOG_S=${RS_LOCK_LOG_S:-600}                         # 等锁期间每 10 分钟记一行
LOCK_MAX_S=${RS_LOCK_MAX_S:-10800}                       # 最长等 3 小时，超时 rc=14
GATE_MAX=0.05                                            # 任务书：复训与参照的末轮 val macro-F1 相差 0.05 以上就停下报告
SELFCHECK=${RS_SELFCHECK:-1}                             # 1 = 起跑前核对本脚本与 CODE_HEAD 里的同名文件一致（只有演练才关）
CPU_STAT_FILE=${RS_CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}   # 每个训练起止各记一次 usage_usec、nr_throttled
# 训练的公共参数：参照配方的 config.json 逐项（batch 128、lr 0.001、类权重见下面各臂、max_grad_norm 5 / Adam 的 β 与 ε / 无增强 / 无权重衰减是训练器默认值）
BASE_ARGS="--epochs $EPOCHS --batch $BATCH --lr $LR --drop-collision --save-every-epoch"
arm_args() {   # 各臂与参照配方的差别（任务卡的表）；R0 的 --class-weights auto 是参照 config.json 里 class_weights_source=auto 的来源（不写这个开关时训练器不加类权重）
  case $1 in
    R0) echo "--class-weights auto" ;;
    R1) echo "--class-weights auto --cosine-lr" ;;
    R2) echo "--class-weights 0,1,1,1 --cosine-lr" ;;
    R3) echo "--class-weights 0,1,2,2 --cosine-lr" ;;
    R2e) echo "--class-weights $SUBST_SAFE_W,1,1,1 --cosine-lr" ;;   # 替代臂：SAFE 的 0 换成 SUBST_SAFE_W
    R3e) echo "--class-weights $SUBST_SAFE_W,1,2,2 --cosine-lr" ;;
    *) return 1 ;;
  esac
}
ARMS="R1 R2 R3"                                          # 卡里新训的臂；R0 只复训种子 0（闸门）。R2、R3 先探训一次，被拒则改跑 R2e、R3e（见「起跑前发现」）
SUBST_SAFE_W=1e-06                                       # 替代臂里 SAFE 的权重：正数才过训练器的检查；取 1e-6 让它的贡献在 1e-8 量级以下
BLOCK_RE='设了 0 权重'                                    # 训练器拒绝零权重的报错里的固定字样；日志里认这个才算「被拒」，别的失败照常算失败
# ----------------------------------------------------------------

export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB，临时文件放数据盘
export OUT CODE=$OUT/code PY=$R/.venv-tools/bin/python TRAIN_BIN INFER DATA BASE_ARGS EPOCHS LAST_K CPU_STAT_FILE SUBST_SAFE_W
mkdir -p "$OUT/tmp" "$OUT/runs" "$OUT/select" "$OUT/avg"

log() { echo "$(date '+%F %T') $*"; }
export -f log arm_args

finish() {   # $1 = rc，$2 = 原因：写 DONE，退出
  trap - TERM INT
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

# 先拿实例锁，再写任何状态文件（run.log、run.pid、DONE、launches.log）
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }

refuse() {   # 拒绝起跑 = rc=5：原因追加进 refused.txt，DONE 只在还没有时才写（不盖掉上一次的结果）
  echo "$1" >&2
  echo "$(date '+%F %T') $1" >> "$OUT/refused.txt"
  [[ -f "$OUT/DONE" ]] || echo "rc=5 $(date '+%F %T') $1" > "$OUT/DONE"
  exit 5
}
OVERRIDES=$(compgen -v | grep '^RS_' | while read -r v; do echo "$v=${!v}"; done)
if [[ -n "$OVERRIDES" && "${RS_REHEARSAL:-}" != 1 ]]; then
  refuse "正式运行遇到覆盖配置的环境变量，拒绝起跑（演练请设 RS_REHEARSAL=1，覆盖项会逐个记进 launches.log）：$(tr '\n' ' ' <<< "$OVERRIDES")"
fi

exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；已完成的训练 / 预测 / 平均保留，同一条命令重起可续跑"' TERM INT
log "开始 PGID=$$ 训练池锁=$TRAIN_LOCK gpu=$GPU procs=$PROCS epochs=$EPOCHS last_k=$LAST_K"

# ---------------------------------------------------------------- 起跑条件与冻结代码
gpu_list=$(nvidia-smi -L 2>&1)
grep -qF "$GPU" <<< "$gpu_list" || finish 12 "配置的 GPU $GPU 不在 nvidia-smi -L 的列表里（$(head -c 300 <<< "$gpu_list" | tr '\n' ';')），没有起跑"
for f in "$TRAIN_BIN" "$INFER" "$DATA"; do [[ -f "$f" ]] || finish 5 "输入不存在：$f"; done
data_sha=$(sha256sum "$DATA" | cut -d' ' -f1)
[[ "$data_sha" == "$DATA_SHA" || "${RS_REHEARSAL:-}" == 1 ]] || finish 5 "数据集 sha256 $data_sha 与参照 run 的 $DATA_SHA 不同，不起跑"
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=experiments/2026-11-12-recipe-stability/run_offline.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/tools/select_ckpt.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败"
fi
echo "$CODE_HEAD" > "$OUT/code_head.txt"
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- tools sim/tools experiments/2026-11-12-recipe-stability | while read -r _m _t sha path; do
        [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
# 起跑记录只追加不覆盖：HEAD、输入的 sha256、被覆盖的配置项
{ echo "=== $(date '+%F %T') 起跑 PGID=$$ 模式=$([[ "${RS_REHEARSAL:-}" == 1 ]] && echo 演练 || echo 正式)"
  echo "code_head $CODE_HEAD（$SRC，分支 $(git -C "$SRC" rev-parse --abbrev-ref HEAD)）"
  echo "trainer_build main af5292f（$D/tmp/build-check-main-1111，CMakeCache 的源目录 = 主仓库 src）；src/ 自 af5292f 到 f56d54f 的 git diff 为空"
  echo "gpu $(grep -F "$GPU" <<< "$gpu_list" | head -n1)"
  sha256sum "$TRAIN_BIN" "$INFER" "$DATA" "$CODE/tools/select_ckpt.py" "$CODE/tools/average_ckpt.py" "${BASH_SOURCE[0]}"
  echo "覆盖项：${OVERRIDES:-无}"; } >> "$OUT/launches.log"

# ---------------------------------------------------------------- 阻塞式拿训练池锁
exec 8>> "$TRAIN_LOCK"
t_lock=$SECONDS
if ! flock -n 8; then
  log "训练池锁 $TRAIN_LOCK 被占用，阻塞等待（每 $LOCK_LOG_S s 记一行，最多 $LOCK_MAX_S s）"
  until flock -w "$LOCK_LOG_S" 8; do
    log "仍在等训练池锁（已等 $((SECONDS - t_lock)) s）"
    (( SECONDS - t_lock > LOCK_MAX_S )) && finish 14 "等训练池锁超过 $LOCK_MAX_S s，没有起跑"
  done
fi
log "拿到训练池锁，等了 $((SECONDS - t_lock)) s；持有到脚本退出"

# ---------------------------------------------------------------- 记账与单步
cg_snapshot() {   # 一行：epoch usage_usec nr_throttled throttled_usec（cgroup 整个容器）
  awk -v t="$(date +%s.%N)" '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} $1=="throttled_usec"{h=$2+0} END{printf "%s %d %d %d\n", t, u, n, h}' "$CPU_STAT_FILE"
}
cg_mark() {       # $1 = start|end，$2 = 名：追加到 $OUT/cg/<名>.cg
  mkdir -p "$OUT/cg"; echo "$1 $(cg_snapshot)" >> "$OUT/cg/$2.cg"
}
export -f cg_snapshot cg_mark

train_run() {   # $1 = 臂名，$2 = 种子，其余 = 额外训练参数。已完成的跳过；半成品挪开留证据，不删
  local name=$1 seed=$2 run ts rc; shift 2
  run=$OUT/runs/$name-s$seed
  if [[ -f "$run/TRAIN_OK" ]]; then log "$name-s$seed 训练已完成，跳过"; return 0; fi
  ts=$(date +%H%M%S)
  if [[ -d "$run" ]]; then
    mv "$run" "$run.partial-$ts"
    [[ -f "$run.log" ]] && mv "$run.log" "$run.partial-$ts.log"
  fi
  cg_mark start "train-$name-s$seed"
  log "$name-s$seed 训练开始：uavrgb_train $BASE_ARGS --seed $seed $*"
  # shellcheck disable=SC2086
  (cd "$OUT" && "$TRAIN_BIN" --data "$DATA" $BASE_ARGS --seed "$seed" "$@" --out "$run") > "$run.log" 2>&1   # 从非 git 目录起训：config.json 的 source_git 为空，构建来源记在 launches.log
  rc=$?
  cg_mark end "train-$name-s$seed"
  if (( rc == 0 )) && [[ -f "$run/ckpt_ep$(printf %02d "$EPOCHS").bin" ]]; then
    echo "rc=0 $(date '+%F %T')" > "$run/TRAIN_OK"
    log "$name-s$seed 训练完成：$(grep -E '^完成' "$run.log" | cut -c1-90)"
  else
    log "$name-s$seed 训练失败 rc=$rc，日志 $run.log"; return 1
  fi
}
select_run() {  # $1 = <臂名>-s<种子>：逐轮 val 预测（tools/select_ckpt.py 既有做法；它自检每轮 macro-F1 与 metrics.jsonl 一致，不一致 rc=1）
  local tag=$1 sel=$OUT/select/$1 rc
  [[ -f "$sel/SELECT_OK" ]] && return 0
  [[ -d "$sel" ]] && mv "$sel" "$sel.partial-$(date +%H%M%S)"
  "$PY" "$CODE/tools/select_ckpt.py" --run "$OUT/runs/$tag" --data "$DATA" --out "$sel" --infer-bin "$INFER" > "$sel.log" 2>&1
  rc=$?
  if (( rc == 0 )); then echo "rc=0 $(date '+%F %T')" > "$sel/SELECT_OK"; log "$tag 逐轮 val 预测完成"
  else log "$tag 逐轮预测失败 rc=$rc，日志 $sel.log"; return 1; fi
}
avg_run() {     # $1 = 标签，$2 = run 目录：末 k 轮平均与平均加校准，真实推理落盘 val 预测
  local tag=$1 run=$2 out=$OUT/avg/$1 rc
  [[ -f "$out/AVG_OK" ]] && return 0
  [[ -d "$out" ]] && mv "$out" "$out.partial-$(date +%H%M%S)"
  "$PY" "$CODE/experiments/2026-11-12-recipe-stability/avg_preds.py" --run "$run" --data "$DATA" --out "$out" --infer-bin "$INFER" --last-k "$LAST_K" > "$out.log" 2>&1
  rc=$?
  if (( rc == 0 )); then echo "rc=0 $(date '+%F %T')" > "$out/AVG_OK"; log "$tag 平均与校准完成：$(grep -E 'δ=' "$out.log" | cut -c1-120)"
  else log "$tag 平均失败 rc=$rc，日志 $out.log"; return 1; fi
}
job() { local name=$1 seed=$2; shift 2; train_run "$name" "$seed" "$@" && select_run "$name-s$seed"; }
export -f train_run select_run avg_run job

# ---------------------------------------------------------------- 阶段 1：闸门（参照配方复训种子 0）
cg_mark start offline
log "阶段 1：闸门。参照配方复训种子 0（R0b-s0），与 $REF/e1-noise/seed0/run 对账"
# shellcheck disable=SC2046
job R0b 0 $(arm_args R0) || finish 11 "闸门复训（R0b-s0）失败，见 $OUT/runs/R0b-s0.log"
"$PY" "$CODE/experiments/2026-11-12-recipe-stability/gate_check.py" --ref "$REF/e1-noise/seed0/run" --new "$OUT/runs/R0b-s0" --out "$OUT/gate.json" --max-dmf1 "$GATE_MAX" > "$OUT/gate.log" 2>&1
grc=$?
log "闸门 rc=$grc $(cat "$OUT/gate.log" | cut -c1-400)"
case $grc in
  0) ;;
  1) finish 20 "闸门不过：复训与参照的末轮 val macro-F1 相差 >= $GATE_MAX，停下报告。$(cut -c1-300 "$OUT/gate.log")" ;;
  *) finish 11 "闸门对账本身失败（输入不全或口径对不上）：$(cut -c1-300 "$OUT/gate.log")" ;;
esac

# ---------------------------------------------------------------- 阶段 2a：卡里的 R2、R3 先用原配方探训一次种子 0（见「起跑前发现」）
probe_card_arm() {  # $1 = R2|R3。返回 0 = 训练器接受（这个臂照常训）；1 = 被「零权重」拒绝，记 blocked；2 = 别的原因失败
  local a=$1 note=$OUT/blocked/$1.txt
  mkdir -p "$OUT/blocked"
  if [[ -f "$note" ]]; then   # 续跑沿用上次的判定，不重复探训（避免每次续跑都多一个 .partial 目录）
    case $(cut -d' ' -f1 "$note") in runnable) return 0 ;; blocked) return 1 ;; *) return 2 ;; esac
  fi
  # shellcheck disable=SC2046
  if train_run "$a" 0 $(arm_args "$a"); then echo "runnable $(date '+%F %T') 训练器接受卡里的配方" > "$note"; return 0; fi
  if grep -q "$BLOCK_RE" "$OUT/runs/$a-s0.log"; then
    echo "blocked $(date '+%F %T') 训练器拒绝卡里的配方 [$(arm_args "$a")]：$(grep "$BLOCK_RE" "$OUT/runs/$a-s0.log" | head -n1)" > "$note"; return 1
  fi
  echo "failed $(date '+%F %T') 训练失败但不是零权重被拒，见 runs/$a-s0.log" > "$note"; return 2
}
RUN_ARMS="R1"; BLOCKED=""; PROBE_FAILED=""
for a in R2 R3; do
  probe_card_arm "$a"; pr=$?
  case $pr in
    0) RUN_ARMS="$RUN_ARMS $a"; log "$a：训练器接受卡里的配方，照卡训满三个种子" ;;
    1) RUN_ARMS="$RUN_ARMS ${a}e"; BLOCKED="$BLOCKED $a"; log "$a：训练器拒绝卡里的配方（$(cut -c1-200 "$OUT/blocked/$a.txt")）；记未跑，改跑替代臂 ${a}e" ;;
    *) PROBE_FAILED="$PROBE_FAILED $a"; log "$a：探训失败，不是零权重被拒（$(cut -c1-200 "$OUT/blocked/$a.txt")）" ;;
  esac
done

# ---------------------------------------------------------------- 阶段 2b：R1 与 R2/R3（或替代臂）各 3 个训练种子（组内最多 $PROCS 个训练进程；按臂排序，被打断时先有整臂）
log "阶段 2b：$RUN_ARMS × 种子 $SEEDS，组内 $PROCS 个训练进程"
{ for a in $RUN_ARMS; do for s in $SEEDS; do echo "$a|$s"; done; done; } \
  | xargs -P "$PROCS" -I{} bash -c 'IFS="|" read -r n s <<< "{}"; job "$n" "$s" $(arm_args "$n")'

# ---------------------------------------------------------------- 阶段 3：末 k 轮平均与平均加校准（R0 的三个参照 run 也重算：与 11-10 的 swa / cal_swa 权重核对 sha256）
log "阶段 3：末 $LAST_K 轮平均与平均加校准"
for s in $SEEDS; do avg_run "R0-s$s" "$REF/e1-noise/seed$s/run"; done
avg_run R0b-s0 "$OUT/runs/R0b-s0"
for a in $RUN_ARMS; do for s in $SEEDS; do [[ -f "$OUT/runs/$a-s$s/TRAIN_OK" ]] && avg_run "$a-s$s" "$OUT/runs/$a-s$s"; done; done
cg_mark end offline

# ---------------------------------------------------------------- 收尾：核对每个产物的完成标记，DONE 的 rc 反映失败
missing=""
chk() { [[ -f "$1" ]] || missing="$missing ${1#$OUT/}"; }
chk "$OUT/runs/R0b-s0/TRAIN_OK"; chk "$OUT/select/R0b-s0/SELECT_OK"; chk "$OUT/avg/R0b-s0/AVG_OK"
for s in $SEEDS; do chk "$OUT/avg/R0-s$s/AVG_OK"; done
for a in $RUN_ARMS; do for s in $SEEDS; do chk "$OUT/runs/$a-s$s/TRAIN_OK"; chk "$OUT/select/$a-s$s/SELECT_OK"; chk "$OUT/avg/$a-s$s/AVG_OK"; done; done
[[ -z "$PROBE_FAILED" ]] || finish 11 "探训失败（不是零权重被拒）的臂：$PROBE_FAILED；见 blocked/*.txt 与 runs/*-s0.log"
[[ -z "$missing" ]] || finish 11 "有产物没完成：$missing；读 run.log，同一条命令重起可续跑"
if [[ -n "$BLOCKED" ]]; then
  finish 21 "完成，但卡里的臂${BLOCKED} 被训练器拒绝（SAFE 权重 0，证据见 blocked/），已改跑替代臂（SAFE 权重 $SUBST_SAFE_W）：$RUN_ARMS；闸门过；R0b-s0 与各臂 × 种子 $SEEDS 的训练、逐轮 val 预测、末 $LAST_K 轮平均与平均加校准完成；下一步：stability.py report"
fi
finish 0 "闸门过；R0b-s0 与 $RUN_ARMS × 种子 $SEEDS 训练、逐轮 val 预测、末 $LAST_K 轮平均与平均加校准全部完成；下一步：stability.py report"
