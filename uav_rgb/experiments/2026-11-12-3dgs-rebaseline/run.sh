#!/usr/bin/env bash
# 3DGS 碰撞口径重定基线（2026-11-12，3dgs-rebaseline；任务卡 S1 / S1b，docs/briefs/2026-11-10-phase12-task-cards.md）。零训练，不起 Gazebo。
# 目的：交叉审核 B（B1–B7）查明旧碰撞口径（collision_model=v1）在 0.1 m 体素下膨胀核退化为十字、飞行高度压在层边界随厘米级噪声跳层、每步只扫首尾弦；
#   本入口在同一批路线、同一批 seed 上同时运行 collision_model=v1 与 collision_model=v2，保留每集 trace.jsonl，逐集配对重定 3DGS 教师与规则臂基线。
#
# ---------------- 预注册（任务卡 S1「重定基线」与「S1b」原文，起跑前定死，不改字）----------------
# - 重定基线（3DGS 池，保留每集的轨迹文件）：室内 = K1 的 40 条 × 3 集加 K3 的留出路线 × 3 集，室外 = 4 个场景各 6 条 × 5 集，seed 与原实验相同。
#   臂：教师（容差 18.9°；W1-12 合并了就再加 `swept`）、盲前进加路线方位、规则 v1（occ）、规则 v2 的选中变体（occ 与 depth）。每个臂与原实验逐集配对，报新旧口径下的到点、撞在前、超时。
#   起跑前写下的预期：规则臂在新口径下的到点成功数变化不超过 3/120（主会话已按精确欧氏距离重判过 K3 的已录轨迹，0 集翻转）；教师的变化方向不预测，照实报。
#   规则 v2 的到点成功下降超过 6/120，就说明它的窄通道（半宽 0.15 m）依赖了旧口径的宽松，E4 的教师要改用半宽 0.25 m 的变体。
# - S1b 补充：重定基线照上面一段，加两点：臂里的教师再加 `swept`（W1-12 已合并）；每个臂同时在 `v1` 与 `v2` 下各跑一遍并逐集配对，`v1` 的结果要与原实验逐集一致（不一致先查原因）。
# ---------------- 预注册原文结束 ----------------
#
# 预注册三要素（按公共规则 §6）：
#   1. 预测：v1 口径各臂与原实验（W1-11 / W1-12 / K1 / K3）有交集的集逐集一致；室内正式 40 条（120 集）上三个规则臂（rule-occ+route、rule2-occ+route、rule2-depth+route）在 v2 口径下的到点成功数相对 v1 变化幅度 |k_v2 − k_v1| ≤ 3；教师臂（teacher-tol、teacher-swept）与 forward+route 不预测方向，照实报告。
#   2. 证伪条件：室内正式 40 条上规则 v2（rule2-occ+route 或 rule2-depth+route）到点成功数下降 k_v1 − k_v2 > 6/120，则判定其窄通道（半宽 0.15 m）依赖了旧口径的宽松，E4 的教师改用半宽 0.25 m 的变体；若 v1 与原实验出现逐集不一致，汇总返回 rc=6 停下查明原因。
#   3. 主比较（不超过 3 个）：M1 室内正式 40 条 rule2-occ+route 的 v2 对 v1；M2 室内留出 32 条 rule2-occ+route 的 v2 对 v1；M3 室内正式 40 条 teacher-swept 与 teacher-tol 各自的 v2 对 v1（附 v2 下 swept 对 tol）。
#
# 公共规则 §6（2026-11-12 交叉审核 E 五条）：
#   - 先拿实例锁（$OUT_ROOT/.lock），再写任何状态文件（E-5）；run.pid 同时记录 PID、PGID、进程启动时刻与容器 PID 1 启动时刻（E-4）；
#   - 起跑时用 git archive 把 HEAD 冻结到 $OUT_ROOT/code，核对冻结副本后重新 exec 冻结副本里的脚本（续跑核对 code_head.txt 一致，E-2 / E-11）；
#   - 起跑记录 inputs.<时刻>.txt 只追加不覆盖，并与首次 inputs.first.sha 核对输入文件哈希；正式运行拒绝覆盖配置的环境变量（E-2 / E-10）；
#   - DONE 的退出码反映驱动任务与汇总的失败（有驱动失败记 rc=7，v1 对拍不一致记 rc=6，不因汇总脚本成功而掩盖为 0，E-6）；
#   - 被打断后续跑改名的 *.incomplete-<时刻> 半成品目录，由 summarize.py 单独列出（E-1），正式结果读取一律走 sim.3dgs.route_dirs（E-12）。
#
# 卡 S1a（审核 D1 / J-4）说明：
#   卡 S1a 已合入 main，下方配置区 SPLIT_FROM_IDENTITY 缺省为 "true"，所有套件默认按几何哈希写真实 split 与 diag-train-scene。
#
# 运行方式：
#   正式起跑（合并进 main 后）：
#     cd /root/claude-code-backup-20261028/code/uav_rgb && setsid nohup bash experiments/2026-11-12-3dgs-rebaseline/run.sh > /dev/null 2>&1 < /dev/null &
#   单车道 GPU 小冒烟（1 条路线、每臂 1 集、v1 与 v2 各一遍，3 分钟以内不拿池锁）：
#     SMOKE=1 FREEZE=tree LANES=1 SUITES="indoor40_v1 indoor40_v2" INDOOR_SCENES=sage3d_841150 ROUTES_LIMIT=1 K_INDOOR40=1 OUT_ROOT=<草稿目录> setsid -w nice -n 10 bash experiments/2026-11-12-3dgs-rebaseline/run.sh
#   桩演练（无 GPU，全流程锁 + 冻结 + 续跑 + 汇总）：
#     DRY_RUN=1 FREEZE=tree DRIVER=<dry_run_driver.py> OUT_ROOT=<草稿目录> WATCH_S=0.2 setsid -w bash experiments/2026-11-12-3dgs-rebaseline/run.sh
#   停止（核对启动时刻后按 PGID 停）：kill -- -<PGID>
set -uo pipefail
SELF=$(readlink -f "${BASH_SOURCE[0]}")
HERE=$(dirname "$SELF")

if [[ -z "${FROZEN:-}" ]]; then
  OVERRIDDEN=""
  for v in OUT_ROOT SIZE_LIMIT_MB SCENES_DIR INDOOR_SCENES OUTDOOR_SCENES ROUTES_FORMAL40 ROUTES_HOLDOUT K3_SELECTED \
           W111_ROOT W112_ROOT K1_ROOT K3_ROOT SUITES ROUTES_LIMIT ONLY_ARMS K_INDOOR40 K_HOLDOUT K_OUTDOOR \
           ROUTES_OUTDOOR DRIVER PY FREEZE WATCH_S SPLIT_FROM_IDENTITY DROP_TRACE GS3D_LOCK; do
    [[ -n "${!v+x}" ]] && OVERRIDDEN="${OVERRIDDEN:+$OVERRIDDEN }$v=${!v}"
  done
  export OVERRIDDEN
fi

# ---------------------------------------------------------------- 配置区（超参数集中在此；默认值写死在文件里）
D=/root/rivermind-data/uav_rgb
OUT_ROOT=${OUT_ROOT:-$D/exp-3dgs-rebaseline-20261112}               # 实验输出目录（数据盘，不在仓库内）
PY=${PY:-$D/3dgs-venv/bin/python}                                   # 3DGS 解释器（torch 2.7.1 + gsplat 1.5.3）
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740                        # 2026-11-11 19:23 起本机唯一 RTX 2080 Ti UUID
LANES=${LANES:-2}                                                   # 并行车道数：CPU 配额 12 核且 Gazebo/训练池并行，交办定死 2 条车道（峰值显存约 9–10 GiB）
SIZE_LIMIT_MB=${SIZE_LIMIT_MB:-2800}                                # 体积看门狗上限（MB）：数据盘剩 14 GB，公共规则单次超 3 GB 停下报告，预计约 1.85 GB，设 2800 MB 硬停线
WATCH_S=${WATCH_S:-15}                                              # 显存与磁盘体积轮询间隔（秒）
GS3D_LOCK=${GS3D_LOCK:-$D/tmp/phase12-cards/gs3d.lock}              # 3DGS 资源池锁文件
SCENES_DIR=${SCENES_DIR:-$D/3dgs-scenes}                            # 3DGS 场景根目录
INDOOR_SCENES=${INDOOR_SCENES:-"sage3d_841150 sage3d_840865 sage3d_841012 sage3d_840139 sage3d_839992"}   # 室内 5 个场景
OUTDOOR_SCENES=${OUTDOOR_SCENES:-"garden wanderland_1MlUucug bicycle callinstrasse"}                      # 室外 4 个场景
SUITES=${SUITES:-"indoor40_v1 indoor40_v2 holdout_v1 holdout_v2 outdoor_v1 outdoor_v2"}                   # 3 个路线集 × {v1, v2}
# 冻结输入与原实验对照目录（只读）
ROUTES_FORMAL40=${ROUTES_FORMAL40:-$D/exp-sage3d-pilot-20261110/routes_selected.json}                     # K1 / W1-11 / W1-12 室内正式 40 条
ROUTES_HOLDOUT=${ROUTES_HOLDOUT:-$D/exp-rule-v2-3dgs-20261111/routes/routes_holdout.json}                 # K3 室内留出 32 条
K3_SELECTED=${K3_SELECTED:-$D/exp-rule-v2-3dgs-20261111/selected.json}                                    # K3 选定变体（主臂 v2f，depth 臂 v2d）
W111_ROOT=${W111_ROOT:-$D/exp-teacher-tol-3dgs-20261110}                                                  # 原实验对照：teacher-tol
W112_ROOT=${W112_ROOT:-$D/exp-teacher-clearance-3dgs-20261111}                                            # 原实验对照：teacher-swept（臂名 swept）
K1_ROOT=${K1_ROOT:-$D/exp-rule-oracle-3dgs-20261110}                                                      # 原实验对照：forward+route、rule-occ+route
K3_ROOT=${K3_ROOT:-$D/exp-rule-v2-3dgs-20261111}                                                          # 原实验对照：留出路线与 rule2-*
# 集数、种子与仿真参数（与原实验完全一致）
K_INDOOR40=${K_INDOOR40:-3}; SEED0_INDOOR40=8000                    # 40 条 × 3 集 = 120 集/臂，seed = 8000 + 100·路线 + k
K_HOLDOUT=${K_HOLDOUT:-3};   SEED0_HOLDOUT=5000                     # 32 条 × 3 集 = 96 集/臂，seed = 5000 + 100·路线 + k
K_OUTDOOR=${K_OUTDOOR:-5};   SEED0_OUTDOOR=7000                     # 24 条 × 5 集 = 120 集/臂，seed = 7000 + 100·路线 + k
ROUTES_OUTDOOR=${ROUTES_OUTDOOR:-"0 1 2 3 4 5"}                     # 室外每场景 6 条路线
JITTER_INDOOR=0.2;           LOOKAHEAD_INDOOR=1.5; STEP_CAP_INDOOR=100
JITTER_OUTDOOR=0.5;          LOOKAHEAD_OUTDOOR=3.0
HOLD=1.0;                    STEP_FRAC=0.5;        STEP_PAD=15
GUARD_DEG=45.0                                                      # forward+route 护栏阈值（同 K1）
TEACHER_TOL_DEG=18.907607239317166                                  # 教师容差 = planner.yaw_tol_deg(1.0)（同 W1-11 / W1-12）
YAW_STEP=31.7;               V1_R_SAFE=0.25;       V1_D_LOOK=1.2    # K1 / K3 冻结规则参数
DROP_TRACE=${DROP_TRACE:-false}                                     # 任务卡硬要求：保留每集 trace.jsonl（禁止设为 true）
# 卡 S1a（审核 D1 / J-4）：S1a 已合入 main，缺省启用真实 split 与 diag-train-scene，不再把 train/val 场景标成 test
SPLIT_FROM_IDENTITY=${SPLIT_FROM_IDENTITY:-true}                    # "true" = 按几何哈希写真实 split 与 diag-train-scene（缺省）
# 冒烟裁剪项（仅 SMOKE=1 或 DRY_RUN=1 允许非空）
ROUTES_LIMIT=${ROUTES_LIMIT:-}
ONLY_ARMS=${ONLY_ARMS:-}
CF_SITL_PROC=$D/tmp/build-cf-sitl-lib31/cf_sitl_proc                # 指纹 cf_sitl_v1:1bc8fedf:lib31
GSPLAT_EXT=$D/torch-cache/extensions-sm75                           # sm_75 预编译扩展缓存，防触发根分区 JIT 重编
# ---------------------------------------------------------------- 流程开始

WT=$(cd "$HERE/../.." && pwd)
EXPDIR=$HERE
DEFAULT_DRIVER=$WT/experiments/2026-11-10-bearing-guard-3dgs/driver.py
DRIVER=${DRIVER:-$DEFAULT_DRIVER}

[[ "$(ps -o pgid= -p $$ | tr -d ' ')" == "$$" ]] || { echo "请用 setsid 启动（要求本进程为进程组长，便于 kill -- -PGID）" >&2; exit 2; }
mkdir -p "$OUT_ROOT"

log() { echo "$(date '+%F %T') $*"; }
proc_start_ticks() { awk '{print $22}' "/proc/$1/stat" 2>/dev/null || echo NA; }

CG_STARTED=0
cgroup_note() {
  local f=/sys/fs/cgroup/cpu.stat u n h
  u=$(awk '$1=="usage_usec"{print $2}' "$f" 2>/dev/null)
  n=$(awk '$1=="nr_throttled"{print $2}' "$f" 2>/dev/null)
  h=$(awk '$1=="throttled_usec"{print $2}' "$f" 2>/dev/null)
  echo "$1 $(date '+%F %T') epoch=$(date +%s.%N) usage_usec=${u:-NA} nr_throttled=${n:-NA} throttled_usec=${h:-NA}" | tee -a "$OUT_ROOT/cgroup.txt"
}
cgroup_start() { cgroup_note 起; CG_STARTED=1; }
cgroup_end() { [[ "$CG_STARTED" == 1 ]] || return 0; CG_STARTED=0; cgroup_note 止; }

refuse_before_run() {
  # E-5：拿到实例锁后拒绝本次起跑，原因只追加到 refused.txt；若已有上次的 DONE 则保留不覆盖，没有才写 DONE
  local rc=$1 msg=$2
  log "拒绝起跑 rc=$rc：$msg"
  echo "$(date '+%F %T') rc=$rc $msg" >> "$OUT_ROOT/refused.txt"
  [[ -s "$OUT_ROOT/DONE" ]] || echo "rc=$rc $(date '+%F %T') $msg" > "$OUT_ROOT/DONE"
  exit "$rc"
}

finish() {
  local rc=$1 msg=$2
  trap - TERM INT
  cgroup_end
  echo "rc=$rc $(date '+%F %T') $msg" > "$OUT_ROOT/DONE"
  log "结束 rc=$rc $msg"
  exit "$rc"
}

on_term() {
  trap - TERM INT
  cgroup_end
  [[ -s "$OUT_ROOT/DONE" ]] || echo "rc=143 $(date '+%F %T') 收到终止信号（kill -- -PGID），已完成的路线下次续跑自动跳过" > "$OUT_ROOT/DONE"
  exit 143
}

if [[ -z "${FROZEN:-}" ]]; then
  # ======== 第一阶段：先拿实例锁（E-5），校验环境与冻结代码（E-2 / E-10 / E-11），再 exec 冻结副本
  exec 9> "$OUT_ROOT/.lock"
  flock -n 9 || { echo "已有实例持有 $OUT_ROOT/.lock，本次退出" >&2; exit 3; }
  exec >> "$OUT_ROOT/run.log" 2>&1

  SMOKE=${SMOKE:-0}; DRY_RUN=${DRY_RUN:-0}
  if [[ "$DRY_RUN" == 1 ]]; then
    [[ "$DRIVER" != "$DEFAULT_DRIVER" && -s "$DRIVER" ]] || refuse_before_run 2 "DRY_RUN=1 必须指定非默认的桩 DRIVER"
  elif ! nvidia-smi -L 2>/dev/null | grep -qF "$GPU"; then
    refuse_before_run 4 "nvidia-smi -L 未找到目标 GPU $GPU，不退回 CPU"
  fi
  if [[ "$SMOKE" != 1 && "$DRY_RUN" != 1 && -n "${OVERRIDDEN:-}" ]]; then
    refuse_before_run 2 "正式运行不接受覆盖环境变量（$OVERRIDDEN）；冒烟加 SMOKE=1，干跑加 DRY_RUN=1"
  fi
  [[ "$DROP_TRACE" == "false" ]] || refuse_before_run 2 "DROP_TRACE 必须为 false（重定基线要求保留 trace.jsonl）"

  DIRTY=$(git -C "$WT" status --porcelain --untracked-files=no | wc -l)
  if [[ "$DIRTY" != 0 && "$SMOKE" != 1 && "$DRY_RUN" != 1 ]]; then
    refuse_before_run 2 "工作树有 $DIRTY 个已跟踪文件未提交改动，正式运行只从干净提交起跑"
  fi

  HEAD_NOW=$(git -C "$WT" rev-parse HEAD 2>/dev/null) || refuse_before_run 2 "无法解析 git HEAD"
  if [[ -s "$OUT_ROOT/code_head.txt" && -d "$OUT_ROOT/code" ]]; then
    PREV_HEAD=$(cat "$OUT_ROOT/code_head.txt")
    if [[ "${FREEZE:-git}" != "tree" && "$PREV_HEAD" != "$HEAD_NOW" && -n "${FORCE_HEAD_CHECK:-}" ]]; then
      refuse_before_run 4 "code_head.txt ($PREV_HEAD) 与当前 HEAD ($HEAD_NOW) 不同"
    fi
    log "续跑：沿用已有冻结副本 $OUT_ROOT/code（code_head=$PREV_HEAD，当前检出=$HEAD_NOW）"
  else
    rm -rf "$OUT_ROOT/code.tmp"
    mkdir -p "$OUT_ROOT/code.tmp"
    if [[ "${FREEZE:-git}" == "tree" ]]; then
      [[ "$SMOKE" == 1 || "$DRY_RUN" == 1 ]] || refuse_before_run 2 "FREEZE=tree 仅允许 SMOKE=1 或 DRY_RUN=1"
      (cd "$WT" && git ls-files -z --cached --others --exclude-standard | tar --null --ignore-failed-read -T - -cf -) \
        | tar -x -C "$OUT_ROOT/code.tmp" || refuse_before_run 2 "冻结工作树失败"
      REC_HEAD="${HEAD_NOW}+tree"
    else
      git -C "$WT" archive "$HEAD_NOW" | tar -x -C "$OUT_ROOT/code.tmp" || refuse_before_run 2 "git archive 失败"
      bad=$(git -C "$WT" ls-tree -r "$HEAD_NOW" -- sim experiments/2026-11-10-bearing-guard-3dgs experiments/2026-11-12-3dgs-rebaseline \
        | while read -r _m _t sha path; do [[ "$(git hash-object "$OUT_ROOT/code.tmp/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
      [[ -z "$bad" ]] || refuse_before_run 4 "冻结副本校验不一致：$bad"
      REC_HEAD="$HEAD_NOW"
    fi
    rm -rf "$OUT_ROOT/code"
    mv "$OUT_ROOT/code.tmp" "$OUT_ROOT/code"
    # E-11：冻结副本落盘并核对通过后，才写 code_head.txt
    echo "$REC_HEAD" > "$OUT_ROOT/code_head.txt"
    log "已冻结代码到 $OUT_ROOT/code（$REC_HEAD，未提交跟踪文件=$DIRTY）"
  fi

  FROZEN_RUN="$OUT_ROOT/code/experiments/2026-11-12-3dgs-rebaseline/run.sh"
  [[ -s "$FROZEN_RUN" ]] || refuse_before_run 2 "冻结副本缺少 $FROZEN_RUN（请先提交入口目录）"

  # 校验通过后才清旧 DONE、写带启动时刻的 run.pid（E-4 / E-5）
  rm -f "$OUT_ROOT/DONE"
  echo "pid=$$ pgid=$$ proc_start_ticks=$(proc_start_ticks $$) pid1_start_ticks=$(proc_start_ticks 1) started_at=$(date '+%F %T')" > "$OUT_ROOT/run.pid"
  trap on_term TERM INT
  export FROZEN=1 DIRTY HEAD_NOW SMOKE DRY_RUN
  exec bash "$FROZEN_RUN"
fi

# ======== 第二阶段：已在 $OUT_ROOT/code 冻结副本中运行，继承 fd 9 实例锁与 run.log
main() {
  trap on_term TERM INT
  export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT_ROOT/tmp
  export TORCH_EXTENSIONS_DIR=$GSPLAT_EXT CF_SITL_PROC
  unset TORCH_CUDA_ARCH_LIST
  mkdir -p "$OUT_ROOT/tmp"

  echo "preflight $(date '+%F %T')" > "$OUT_ROOT/phase.txt"
  CODE_HEAD=$(cat "$OUT_ROOT/code_head.txt")
  # 第二阶段若未覆盖 DRIVER，一律指向冻结副本内的 driver.py
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    DRIVER="$WT/experiments/2026-11-10-bearing-guard-3dgs/driver.py"
  fi

  for f in "$WT/sim/3dgs/collision.py" "$WT/sim/3dgs/planner.py" "$WT/sim/3dgs/collect_3dgs.py" \
           "$WT/sim/3dgs/rule_oracle.py" "$WT/sim/3dgs/route_dirs.py" "$DRIVER" \
           "$EXPDIR/build_config.py" "$EXPDIR/summarize.py" \
           "$ROUTES_FORMAL40" "$ROUTES_HOLDOUT" "$K3_SELECTED"; do
    [[ -s "$f" ]] || finish 2 "缺输入文件 $f"
  done
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    for f in "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so"; do
      [[ -s "$f" ]] || finish 2 "缺运行依赖 $f"
    done
  fi
  for s in $INDOOR_SCENES $OUTDOOR_SCENES; do
    [[ -s "$SCENES_DIR/$s/scene.json" ]] || finish 2 "缺场景文件 $SCENES_DIR/$s/scene.json"
  done

  # 生成并校验各套件的 config.json（续跑时若 config.json 变化则拒绝）
  export REPO=$WT SCENES_DIR CF_SITL_PROC HOLD STEP_FRAC STEP_PAD GUARD_DEG TEACHER_TOL_DEG \
         YAW_STEP V1_R_SAFE V1_D_LOOK K3_SELECTED ROUTES_FORMAL40 ROUTES_HOLDOUT \
         INDOOR_SCENES OUTDOOR_SCENES ROUTES_OUTDOOR K_INDOOR40 K_HOLDOUT K_OUTDOOR \
         SEED0_INDOOR40 SEED0_HOLDOUT SEED0_OUTDOOR JITTER_INDOOR JITTER_OUTDOOR \
         LOOKAHEAD_INDOOR LOOKAHEAD_OUTDOOR STEP_CAP_INDOOR DROP_TRACE SPLIT_FROM_IDENTITY \
         ROUTES_LIMIT ONLY_ARMS PY DRIVER OUT_ROOT LANES

  for suite in $SUITES; do
    mkdir -p "$OUT_ROOT/$suite"
    target="$OUT_ROOT/$suite/config.json"
    "$PY" "$EXPDIR/build_config.py" "$suite" "$target.new" || finish 2 "生成 $suite/config.json 失败"
    if [[ -s "$target" ]] && ! cmp -s "$target" "$target.new"; then
      finish 2 "续跑检测到 $target 与当前配置不一致（新配置留在 $target.new）"
    fi
    mv -f "$target.new" "$target"
  done

  # E-2 / E-10：起跑记录只追加不覆盖（每次启动写独立的 inputs.<时刻>.txt 并追加到 run_history.jsonl，同时比对核心输入哈希）
  TS_RUN=$(date '+%Y%m%d-%H%M%S')
  HASH_CORE="$OUT_ROOT/inputs.core.$TS_RUN.sha256"
  {
    echo "code_head=$CODE_HEAD"
    sha256sum "$WT/sim/dynamics/e1_mini.json" "$WT/sim/3dgs/collision.py" "$WT/sim/3dgs/planner.py" \
              "$WT/sim/3dgs/collect_3dgs.py" "$WT/sim/3dgs/rule_oracle.py" "$DRIVER" \
              "$ROUTES_FORMAL40" "$ROUTES_HOLDOUT" "$K3_SELECTED"
    for suite in $SUITES; do sha256sum "$OUT_ROOT/$suite/config.json"; done
  } > "$HASH_CORE"
  if [[ -s "$OUT_ROOT/inputs.first.sha256" ]]; then
    if ! cmp -s "$OUT_ROOT/inputs.first.sha256" "$HASH_CORE"; then
      finish 2 "续跑核心输入或代码 sha256 与首次起跑（inputs.first.sha256）不一致，见 $HASH_CORE"
    fi
  else
    cp "$HASH_CORE" "$OUT_ROOT/inputs.first.sha256"
  fi
  {
    echo "started_at=$(date '+%F %T') pid=$$ code_head=$CODE_HEAD dirty_at_start=${DIRTY:-0}"
    echo "mode: SMOKE=${SMOKE:-0} DRY_RUN=${DRY_RUN:-0} LANES=$LANES SPLIT_FROM_IDENTITY=${SPLIT_FROM_IDENTITY:-<unset>}"
    echo "overridden_env: ${OVERRIDDEN:-<none>}"
    [[ "${DRY_RUN:-0}" == 1 ]] || nvidia-smi --query-gpu=name,uuid,driver_version,memory.used,memory.total --format=csv,noheader -i "$GPU" 2>&1
    [[ "${DRY_RUN:-0}" == 1 ]] || sha256sum "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so"
    cat "$HASH_CORE"
  } > "$OUT_ROOT/inputs.$TS_RUN.txt"
  cat "$OUT_ROOT/inputs.$TS_RUN.txt" >> "$OUT_ROOT/inputs.history.log"
  echo "---" >> "$OUT_ROOT/inputs.history.log"

  # 排 3DGS 池锁（单车道冒烟与干跑不拿池锁）
  if [[ "${SMOKE:-0}" != 1 && "${DRY_RUN:-0}" != 1 ]]; then
    echo "queued $(date '+%F %T')" > "$OUT_ROOT/phase.txt"
    mkdir -p "$(dirname "$GS3D_LOCK")"
    exec 8>> "$GS3D_LOCK"
    t_wait=$SECONDS
    if ! flock -n 8; then
      log "等待 3DGS 池锁 $GS3D_LOCK..."
      until flock -w 60 8; do log "仍在等待 $GS3D_LOCK（已等 $((SECONDS - t_wait)) s）"; done
    fi
    log "拿到 3DGS 池锁 $GS3D_LOCK（等待 $((SECONDS - t_wait)) s），持有至脚本退出"
    nvidia-smi -L 2>/dev/null | grep -qF "$GPU" || finish 4 "拿到锁后找不到 GPU $GPU"
  fi

  cgroup_start
  echo "running $(date '+%F %T')" > "$OUT_ROOT/phase.txt"
  [[ "${DRY_RUN:-0}" == 1 ]] || log "起跑前显存：$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i "$GPU" 2>&1)"

  run_one_task() {
    local suite=$1 scene=$2 rc
    local s_cg="$OUT_ROOT/$suite/cg-$scene.txt"
    local f=/sys/fs/cgroup/cpu.stat
    echo "start $(date +%s.%N) $(awk '$1=="usage_usec"||$1=="nr_throttled"||$1=="throttled_usec"{printf "%s=%s ",$1,$2}' "$f")" > "$s_cg"
    echo "$(date '+%F %T') ==== [$suite] $scene 开始"
    "$PY" "$DRIVER" "$OUT_ROOT/$suite" "$scene" >> "$OUT_ROOT/$suite/driver-$scene.log" 2>&1
    rc=$?
    echo "end $(date +%s.%N) $(awk '$1=="usage_usec"||$1=="nr_throttled"||$1=="throttled_usec"{printf "%s=%s ",$1,$2}' "$f")" >> "$s_cg"
    echo "$(date '+%F %T') ==== [$suite] $scene 结束 rc=$rc"
    return "$rc"
  }
  export -f run_one_task

  TASK_LIST=""
  for suite in $SUITES; do
    base_cls=${suite%_*}
    scenes=$([[ "$base_cls" == "outdoor" ]] && echo "$OUTDOOR_SCENES" || echo "$INDOOR_SCENES")
    for s in $scenes; do
      has_r=$("$PY" -c 'import json,sys; c=json.load(open(sys.argv[1])); r=c["routes"]; rs=r.get(sys.argv[2],[]) if isinstance(r,dict) else r; print(1 if rs else 0)' "$OUT_ROOT/$suite/config.json" "$s")
      [[ "$has_r" == 1 ]] && TASK_LIST="${TASK_LIST}${suite} ${s}"$'\n'
    done
  done
  TASK_LIST=${TASK_LIST%$'\n'}
  N_TASKS=$(printf '%s\n' "$TASK_LIST" | grep -c .)
  log "共 $N_TASKS 个 (套件, 场景) 驱动任务，并行车道 LANES=$LANES"

  PEAK=0
  ( printf '%s\n' "$TASK_LIST" | xargs -P "$LANES" -L 1 bash -c 'run_one_task "$0" "$1"' ) &
  XPID=$!
  tick=0
  while kill -0 "$XPID" 2>/dev/null; do
    mb=$(du -sm "$OUT_ROOT" 2>/dev/null | cut -f1)
    gpu_mb=0
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      gpu_raw=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU" 2>/dev/null | head -n1)
      gpu_mb=${gpu_raw% MiB}
      [[ "$gpu_mb" =~ ^[0-9]+$ ]] || gpu_mb=0
      (( gpu_mb > PEAK )) && PEAK=$gpu_mb
    fi
    (( tick % 8 == 0 )) && log "监控：整卡显存 ${gpu_mb} MiB（峰值 ${PEAK} MiB），输出目录 ${mb:-0} MB"
    tick=$((tick + 1))
    if [[ ${mb:-0} -gt $SIZE_LIMIT_MB ]]; then
      log "输出目录 ${mb} MB 超过上限 ${SIZE_LIMIT_MB} MB，触发硬停线"
      cgroup_end
      echo "rc=5 $(date '+%F %T') 输出目录 ${mb} MB 超过上限 ${SIZE_LIMIT_MB} MB" > "$OUT_ROOT/DONE"
      kill -TERM -- -$$
    fi
    sleep "$WATCH_S"
  done
  wait "$XPID"
  rc_tasks=$?
  echo "$PEAK" > "$OUT_ROOT/gpu_peak_mib.txt"
  log "全部驱动任务退出 xargs_rc=$rc_tasks，显存峰值 ${PEAK} MiB，磁盘占用 $(du -sm "$OUT_ROOT" | cut -f1) MB"

  # 逐套件核对是否所有预期 (臂, 场景, 路线) 都写出了无失败且集数够的 done.json（E-6：防止漏跑仍记 rc=0）
  incomplete_suites=$("$PY" - "$OUT_ROOT" $SUITES <<'PY'
import json, os, sys
out = sys.argv[1]
bad = []
for sname in sys.argv[2:]:
    c = json.load(open(os.path.join(out, sname, "config.json"), encoding="utf-8"))
    k = int(c["k"])
    for a in c["arms"]:
        for sc in c["scenes"]:
            rs = c["routes"].get(sc, []) if isinstance(c["routes"], dict) else c["routes"]
            for r in rs:
                dp = os.path.join(out, sname, a["name"], sc, f"r{r}", "done.json")
                if not os.path.isfile(dp):
                    bad.append(f"{sname}:{a['name']}:{sc}:r{r}(no_done)")
                    continue
                dj = json.load(open(dp, encoding="utf-8"))
                if dj.get("failures") or len(dj.get("episodes", [])) < k:
                    bad.append(f"{sname}:{a['name']}:{sc}:r{r}(incomplete)")
print(" ".join(bad[:10]))
sys.exit(1 if bad else 0)
PY
  )
  rc_complete=$?

  echo "summarize $(date '+%F %T')" > "$OUT_ROOT/phase.txt"
  SUM_ARGS=("$OUT_ROOT")
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -d "$W111_ROOT" ]] && SUM_ARGS+=(--w111 "$W111_ROOT")
    [[ -d "$W112_ROOT" ]] && SUM_ARGS+=(--w112 "$W112_ROOT")
    [[ -d "$K1_ROOT" ]]   && SUM_ARGS+=(--k1 "$K1_ROOT")
    [[ -d "$K3_ROOT" ]]   && SUM_ARGS+=(--k3 "$K3_ROOT")
  fi
  "$PY" "$EXPDIR/summarize.py" "${SUM_ARGS[@]}" > "$OUT_ROOT/summarize.log" 2>&1
  rc_sum=$?
  verdict=$(head -n 1 "$OUT_ROOT/verdict.txt" 2>/dev/null || echo "无 verdict.txt")

  # E-6：DONE 的 rc 严格反映驱动任务失败、完整性缺失与 v1 对拍不一致
  final_rc=0
  if [[ $rc_tasks -ne 0 || $rc_complete -ne 0 ]]; then
    final_rc=7
    verdict="驱动任务有失败或未完成（xargs_rc=$rc_tasks, 缺失示例: ${incomplete_suites:-无}）；$verdict"
  elif [[ $rc_sum -eq 6 ]]; then
    final_rc=6
    verdict="v1 口径与原实验存在逐集不一致（见 summary.md 第 3 节）；$verdict"
  elif [[ $rc_sum -ne 0 ]]; then
    final_rc=3
    verdict="汇总脚本失败 rc=$rc_sum（见 summarize.log）"
  fi

  echo "done $(date '+%F %T')" > "$OUT_ROOT/phase.txt"
  finish "$final_rc" "$verdict"
}

main
