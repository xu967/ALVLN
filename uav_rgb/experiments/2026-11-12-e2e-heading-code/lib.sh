#!/usr/bin/env bash
# 卡 E2E-1 入口的共用部分（prepare.sh、run.sh 用 source 引入，不单独运行）：路径、臂表、日志、锁、cgroup 记录。
# 本文件随冻结代码（git archive CODE_HEAD → $OUT/code）一起走；两个脚本都从 $OUT/code/experiments/2026-11-12-e2e-heading-code/ 里启动。
# 带 ${X:-} 的项可由环境变量覆盖，只在演练（桩替换）时用；正式运行一律用这里的值。

D=/root/rivermind-data/uav_rgb
R=/root/claude-code-backup-20261028/code/uav_rgb          # 主仓库：只借用 .venv-tools 解释器（只读）
OUT=${OUT:-$D/exp-e2e-heading-code-20261112}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CODE=${CODE:-$(cd "$HERE/../.." && pwd)}                  # 冻结代码根（scripts 在 $CODE/experiments/2026-11-12-e2e-heading-code/）
ENTRY=$CODE/experiments/2026-11-12-e2e-heading-code
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-e2e-1}    # 代码源（分支 p12-e2e-1）：只用来 git archive 冻结代码与核对冻结副本
CODE_HEAD=${CODE_HEAD:-$(cat "$OUT/code_head.txt" 2>/dev/null || echo unknown)}
PY=${PY:-$R/.venv-tools/bin/python}
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740              # 公共任务书 11-11 19:23：uav_rgb 只用这张卡（RTX 2080 Ti）
TRAIN_SRC=${TRAIN_SRC:-$D/tmp/build-student-select/uavrgb_train}   # A0 训练用的同一个二进制（11-09 17:40 构建，A0 的 e1-noise 17:45 起跑）
TRAIN=$OUT/bin/uavrgb_train                               # 起跑时从 TRAIN_SRC 拷一份并记 sha256，之后只用这份
INFER=${INFER:-$D/bin-snapshots/main-97ae04b/uavrgb_infer}   # 与 11-10 / 11-11 的评测同一个推理二进制；支持条件字节（src/infer.cpp 核过）
PLUGIN=${PLUGIN:-$D/tmp/build-cf-sitl-lib31/gz}                      # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）

# 输入（只读）
DEMO_COL=${DEMO_COL:-$D/tmp/agent6-distill/trunc/col}                  # demo：截断的教师示范（300 集）
DEMO_DS=${DEMO_DS:-$D/tmp/agent6-distill/trunc/ds/dataset.uavr}       # demo 的现有数据集（A0 的训练数据）
DAGGER_DS=${DAGGER_DS:-$D/exp-dagger-gz-20261109/full-1109/round_3/ds/dataset.uavr}   # DAgger 第 3 轮聚合集（592 集、20,291 样本）
SWA=${SWA:-$D/exp-swa-calib-20261110/ckpt}                        # A0 的交付权重 cal_swa_s{0,1,2}_k10.bin
A0_RUN_DIR=${A0_RUN_DIR:-$D/exp-student-select-20261109/e1-noise}        # A0 的训练 run：seed{0,1,2}/run
L_STAR_FILE=${L_STAR_FILE:-$D/exp-action-latency-r2-20261111/L_STAR}
A0_SHA_EXPECT=(2fb25b30b337589999ad2385e2372bb38ae1ca6e9b0b28d6889d25bf4a327928 5cfe51f58f4f4a1d1afb4f79db0f11e777fd67271b9605f722b6bdf8372a6900 abcae7cf0d09a475f68356535e14c0afeaf201aefa0fd5f0375ceff7d4b17734)  # W1-5 的 inputs.txt
DEMO_SHA_EXPECT=534ca5904ea027b99a59137f1985a7b1b67646492bed124fb6bcae44309dfcac     # A0 训练契约里的数据集 sha256
SEEDS_TRAIN="0 1 2"
SEEDS_EVAL=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20

# 训练配方：与 A0 的 config.json 逐项相同（核过：除 data / dataset_sha256 / prev_action / source_git 外 29 个键全同）
TRAIN_FLAGS=${TRAIN_FLAGS:-"--epochs 30 --batch 128 --lr 0.001 --class-weights auto --drop-collision --save-every-epoch"}
LAST_K=${LAST_K:-10}                                      # 末 K 轮平均；文件名里的 k10 是固定的名字，演练时改 LAST_K 不改名字
SKIP_SHA_CHECK=${SKIP_SHA_CHECK:-0}                       # 1 = 不核 A0 权重与 demo 数据集的 sha256（只在演练时用）
SELFCHECK=${SELFCHECK:-1}                                 # 1 = 起跑前核对冻结副本与 CODE_HEAD 的提交逐文件一致（只在演练时关）

# 四个臂。名字 = 目录前缀；cond = 评测端 --student-cond（img = 不带条件输入，不传该开关）；
ARMS="a0 a1 a2 a3"
arm_name() { case $1 in a0) echo a0-img-demo ;; a1) echo a1-code-demo ;; a2) echo a2-code-dagger ;; a3) echo a3-img-dagger ;; esac; }
arm_label() { case $1 in a0) echo "A0 img/demo" ;; a1) echo "A1 code/demo" ;; a2) echo "A2 code/dagger" ;; a3) echo "A3 img/dagger" ;; esac; }
arm_is_code() { [[ $1 == a1 || $1 == a2 ]]; }
arm_data() {   # 训练数据集（也是校准与离线核对用的数据集）
  case $1 in a0) echo "$DEMO_DS" ;; a1) echo "$OUT/data/demo-hc/dataset.uavr" ;; a2) echo "$OUT/data/dagger-hc/dataset.uavr" ;; a3) echo "$DAGGER_DS" ;; esac
}
arm_run() {    # $1 = 臂，$2 = 训练种子：训练 run 目录
  if [[ $1 == a0 ]]; then echo "$A0_RUN_DIR/seed$2/run"; else echo "$OUT/runs/$(arm_name "$1")-s$2"; fi
}
arm_ckpt() {   # $1 = 臂，$2 = 训练种子：交付权重（末 10 轮平均 + FORWARD 先验校准）
  if [[ $1 == a0 ]]; then echo "$SWA/cal_swa_s$2_k10.bin"; else echo "$OUT/ckpt/cal_swa_$(arm_name "$1")_s$2_k10.bin"; fi
}
arm_evdir() { echo "$OUT/$(arm_name "$1")-s$2"; }   # 评测臂目录（<臂>-s<训练种子>）

CPU_STAT_FILE=${CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}
CPU_MAX_FILE=${CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}
TRAIN_LOCK=${TRAIN_LOCK:-$D/tmp/phase12-cards/train.lock}
MACHINE_LOCK=${MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}
LOCK_LOG_S=${LOCK_LOG_S:-600}                             # 等锁期间每多少秒记一行

export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=$GPU TMPDIR=$OUT/tmp
mkdir -p "$OUT/tmp"

log() { echo "$(date '+%F %T') $*"; }

cg_snapshot() {   # 一行：epoch usage_usec nr_throttled throttled_usec（整个容器的 cgroup）
  awk -v t="$(date +%s.%N)" '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} $1=="throttled_usec"{h=$2+0} END{printf "%s %d %d %d\n", t, u, n, h}' "$CPU_STAT_FILE"
}
cg_mark() {       # $1 = start|end，$2 = 名字：追加到 $OUT/<名字>.cg（start 时重写）
  if [[ $1 == start ]]; then echo "start $(cg_snapshot)" > "$OUT/$2.cg"; else echo "end $(cg_snapshot)" >> "$OUT/$2.cg"; fi
}
cg_brief() { awk 'END{printf "usage_usec=%s nr_throttled=%s", $2, $3}' <<< "$(cg_snapshot)"; }

acquire_lock() {  # $1 = 文件描述符号，$2 = 锁文件，$3 = 说明：阻塞等待，每 LOCK_LOG_S 秒记一行；拿到后持有到脚本退出
  local fd=$1 file=$2 what=$3 t0=$SECONDS
  mkdir -p "$(dirname "$file")"
  eval "exec $fd>> \"$file\""
  if ! flock -n "$fd"; then
    log "$what：锁 $file 被占用，阻塞等待（每 $LOCK_LOG_S s 记一行）"
    until flock -w "$LOCK_LOG_S" "$fd"; do log "$what：仍在等锁（已等 $((SECONDS - t0)) s）"; done
  fi
  log "$what：拿到锁 $file，等了 $((SECONDS - t0)) s；持有到脚本退出"
  LOCK_WAITED=$((SECONDS - t0))
}

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

verify_frozen() {   # 冻结副本 $CODE 与 $SRC 里 CODE_HEAD 提交的 sim/、tools/、本入口目录逐文件同哈希；返回 0/1，不一致的前 3 个文件放进 FROZEN_BAD
  FROZEN_BAD=""
  [[ "$CODE_HEAD" != unknown ]] || { FROZEN_BAD="没有 $OUT/code_head.txt（先运行 freeze.sh）"; return 1; }
  git -C "$SRC" cat-file -e "$CODE_HEAD^{commit}" 2>/dev/null || { FROZEN_BAD="$CODE_HEAD 在 $SRC 里不是有效提交"; return 1; }
  FROZEN_BAD=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim tools experiments/2026-11-12-e2e-heading-code \
    | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
  [[ -z "$FROZEN_BAD" ]]
}
