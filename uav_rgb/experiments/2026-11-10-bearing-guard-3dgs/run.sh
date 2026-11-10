#!/usr/bin/env bash
# 路线方位护栏 × {盲前进, 学生} 在 3DGS 真实场景上的判别实验（2026-11-10，bearing3dgs；任务书 phase12-20261110）。不训练，不起 Gazebo。
# 问题：11-09 学生在 4 个真实场景 6 条带转弯路线上到点成功率 0.24–0.32，与盲前进 0.31 无差别（docs/experiments/2026-11-09-xeval-3dgs.md）。
#   两个因素混在一起：(a) 学生不知道目标在哪（柱阵是直线任务，观测里没有航向和目标）；(b) 学生没见过真实外观。
#   给策略加一层 oracle 路线方位护栏（真值位姿 + 占据 A* 全局路径的前视点，相当于上层给了粗方向），看
#   {盲前进, 学生} × {无护栏, 有护栏} 四格的到点成功率，决定下一步长程路线任务先补「目标条件」还是先补「真实外观上的感知」。
# 设计：garden / wanderland_1MlUucug / bicycle / callinstrasse × 路线 0–5 × 每路线 5 集；seed = 7000 + 100·路线 + k，出生扰动 ±0.5 m / ±10°
#   （与 11-09 相同，同 seed 出生位姿相同，逐集配对）；hold 1.0；每路线步数 = ceil(max_steps × 0.5) + 15（电池约 100 s 的限制）；
#   stop_on_collision；cf_sitl（lib31）。护栏参数起跑前定死、不在这批路线上调：阈值 45°（3DGS hold 1.0 单步转角约 31.6°，
#   容许偏 1 步、偏 2 步就纠正），路线前视 3.0 m。臂的顺序即优先级，前 4 个是主臂（先把四个场景的主臂都跑完，再补次臂）。
#
# 预注册（2026-11-10 19:45，冒烟之后、正式起跑之前写下；护栏参数与臂的清单冒烟前就已定，冒烟没有改它们）。主比较 3 个，逐集配对（同场景同路线同 seed），指标 SR(stop)，精确 McNemar：
#   P1 forward+route 对 forward：预测到点成功率提高（b > c），四场景合计至少高 0.15。
#      证伪：不高（b <= c）。那说明这些路线上「知道往哪走」不是盲飞失败的主因。
#   P2 calswa-s1+route 对 calswa-s1：预测提高（b > c）。证伪：b <= c。
#   P3 calswa-s1+route 对 forward+route：预测没有可检出的差别（p > 0.05），即柱阵学到的避障在真实外观上不起作用，
#      护栏之后剩下的失败以碰撞为主。证伪：学生加护栏显著好于盲飞加护栏（p < 0.05 且 b > c）。
#      那说明柱阵感知有一部分能迁移，11-09「约等于盲前进」的结论主要来自缺目标信息。
#   次要臂（不进判决）：calswa-s1+goal（目标方位护栏，直指 waypoint，不用路径）、另两个训练种子 +route、teacher（lib31 下重跑的上界）。
#   其余要看的：碰撞率、超时率（既没到点也没撞）、SPL(stop)、护栏改写的步占比；四场景合计与分场景，Wilson 95% 区间。
#   护栏方向的冒烟：1 场景 1 路线 1 集，看轨迹是否朝 waypoint 收敛、改写占比是否合理，通过后才起正式实验。
#
# 运行：setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &        （环境变量可覆盖配置区；冒烟用 OUT=… LANE_B= ROUTES=1 K=1）
# 进度：tail -f $OUT/run.log；每场景日志 $OUT/driver-<场景>.log；汇总随时可跑：python summarize.py $OUT。结束落 $OUT/DONE 与 summary.md。
# 停：kill -- -$(cat $OUT/run.pid)（只按 PGID，不用 pkill -f）。
# 续跑：同一条命令重起；评完的路线（done.json 集数够且无失败）跳过，半成品改名 *.incomplete-<时刻> 后重跑。.lock 互斥，重复启动的那个立即退出。
set -uo pipefail

# ---------------------------------------------------------------- 配置区（改这里）
D=/root/rivermind-data/uav_rgb
WT=${WT:-/root/claude-code-backup-20261028/code/wt/p12-bearing3dgs}   # 代码工作树（分支 p12-bearing3dgs）；起跑后别改里面的 sim/3dgs 与本目录
OUT=${OUT:-$D/exp-bearing-guard-3dgs-20261110}           # 实验输出（数据盘）
PY=$D/3dgs-venv/bin/python                               # torch 2.7.1 + gsplat 1.5.3
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）
SCENES_DIR=$D/3dgs-scenes
LANE_A=${LANE_A-"garden wanderland_1MlUucug"}            # 两条车道各跑两个场景：全机同时最多 2 个 3DGS 驱动进程，每进程显存约 2–3 GB
LANE_B=${LANE_B-"bicycle callinstrasse"}                 # callinstrasse r0/r1/r2/r5 下方无重建地面（场地文档）：汇总里每路线单列，可自行排除
ROUTES=${ROUTES:-"0 1 2 3 4 5"}                          # scene.json 里的 6 条路线
K=${K:-5}                                                # 每路线 5 集：每场景 n=30、四场景 n=120（与 11-09 同）
SEED0=7000                                               # 路线 r 第 k 集 seed = SEED0 + 100·r + k（与 11-09 同，出生位姿可逐集配对）
JITTER=0.5                                               # 出生 ±0.5 m / ±10°（collect_3dgs._spawn_for_episode）
HOLD=1.0                                                 # 学生是 hold 1.0 训的；盲前进、教师同口径
STEP_FRAC=0.5; STEP_PAD=15                               # 每路线步数 = ceil(max_steps×0.5)+15：max_steps 按 hold 0.5 算；最长 89 步 ≈ 89 s，落在电池掉压拐点约 100 s 之前
GUARD_DEG=45.0                                           # 护栏阈值：单步转角约 31.6°，容许偏 1 步，偏 2 步（≥ 45°）就纠正；起跑前定死
LOOKAHEAD_M=3.0                                          # 路线前视距离：粗方向，不是路径跟踪；起跑前定死
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 今天所有闭环实验用的同一个推理二进制
CF_SITL_PROC=$D/tmp/build-cf-sitl-lib31/cf_sitl_proc     # profile cf_sitl_v1:1bc8fedf:lib31，不要动这个目录
GSPLAT_EXT=$D/torch-cache/extensions-sm75                # = pipeline_common.torch_extensions_dir() 在 2080 Ti 上的返回值；已按 sm_75 编好，不应重编
SWA=$D/exp-swa-calib-20261110/ckpt                       # 末 10 轮权重平均 + 类先验校准的三个训练种子权重
N_MAIN=${N_MAIN:-4}                                      # 前 N_MAIN 个臂是预注册主臂（先跑）；冒烟时设 8 让所有臂一次加载跑完
# 臂：名字|底座（forward / student / teacher）|权重|护栏（none / route = 路线前视点方位 / goal = waypoint 方位）。顺序即优先级。
ARMS=(
  "forward|forward||none"
  "forward+route|forward||route"
  "calswa-s1|student|$SWA/cal_swa_s1_k10.bin|none"
  "calswa-s1+route|student|$SWA/cal_swa_s1_k10.bin|route"
  "calswa-s1+goal|student|$SWA/cal_swa_s1_k10.bin|goal"
  "calswa-s0+route|student|$SWA/cal_swa_s0_k10.bin|route"
  "calswa-s2+route|student|$SWA/cal_swa_s2_k10.bin|route"
  "teacher|teacher||none"
)
# ----------------------------------------------------------------

SELF=$(readlink -f "${BASH_SOURCE[0]}"); HERE=$(dirname "$SELF")
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2 GB，临时文件放数据盘
export TORCH_EXTENSIONS_DIR=$GSPLAT_EXT CF_SITL_PROC
unset TORCH_CUDA_ARCH_LIST                               # 带着旧值会让 gsplat 按别的架构重编
mkdir -p "$OUT/tmp"
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
log() { echo "$(date '+%F %T') $*"; }
fail_done() { log "$1"; echo "rc=$2 $(date '+%F %T') $1" > "$OUT/DONE"; exit "$2"; }
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
log "开始 PGID=$$ WT=$(git -C "$WT" rev-parse --short HEAD) OUT=$OUT lanes: [$LANE_A] [$LANE_B] 路线 [$ROUTES] K=$K"

names=(); ckpts=()
for a in "${ARMS[@]}"; do names+=("${a%%|*}"); IFS='|' read -r _ _ c _ <<< "$a"; [[ -n "$c" ]] && ckpts+=("$c"); done
for f in "$INFER" "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/3dgs/guard.py" "$HERE/driver.py" "$HERE/summarize.py" "${ckpts[@]}"; do
  [[ -s "$f" ]] || fail_done "缺输入 $f" 2
done
PHASE1="${names[*]:0:$N_MAIN}"; PHASE2="${names[*]:$N_MAIN}"   # 先把所有场景的主臂跑完，再补次臂（每个场景为此多加载一次，约 1–2 分钟）
cp "$SELF" "$OUT/run.sh.snapshot"
{ sha256sum "$INFER" "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/dynamics/e1_mini.json" \
            "$WT/sim/3dgs/guard.py" "$HERE/driver.py" "$HERE/summarize.py" "${ckpts[@]}"
  for s in $LANE_A $LANE_B; do sha256sum "$SCENES_DIR/$s/scene.json"; done
  echo "git HEAD $(git -C "$WT" rev-parse HEAD) 未提交改动 $(git -C "$WT" status --porcelain | wc -l) 个文件"; } > "$OUT/inputs.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt"
export WT SCENES_DIR ROUTES K SEED0 JITTER HOLD STEP_FRAC STEP_PAD GUARD_DEG LOOKAHEAD_M INFER CF_SITL_PROC LANE_A LANE_B
"$PY" - "$OUT/config.json" "${ARMS[@]}" <<'PYEOF'
import json, os, sys
e = os.environ
arms = []
for line in sys.argv[2:]:
    name, base, ckpt, guard = line.split("|")
    arms.append({"name": name, "base": base, "ckpt": ckpt or None, "guard": None if guard == "none" else guard})
json.dump({"repo": e["WT"], "scenes_dir": e["SCENES_DIR"], "scenes": (e["LANE_A"] + " " + e["LANE_B"]).split(),
           "routes": [int(r) for r in e["ROUTES"].split()], "k": int(e["K"]), "seed0": int(e["SEED0"]),
           "jitter": float(e["JITTER"]), "hold": float(e["HOLD"]), "step_frac": float(e["STEP_FRAC"]),
           "step_pad": int(e["STEP_PAD"]), "guard_threshold_deg": float(e["GUARD_DEG"]),
           "route_lookahead_m": float(e["LOOKAHEAD_M"]), "arms": arms, "infer_bin": e["INFER"],
           "cf_sitl_proc": e["CF_SITL_PROC"]}, open(sys.argv[1], "w"), ensure_ascii=False, indent=2)
PYEOF

run_lane() {    # $1 车道名，其余是场景：一个场景一个驱动进程（加载一次，跑完该阶段的全部臂与路线）；阶段 1 = 主臂，阶段 2 = 次臂
  local lane=$1 rc_lane=0 phase s rc; shift
  for phase in "$PHASE1" "$PHASE2"; do
    [[ -n "$phase" ]] || continue
    for s in "$@"; do
      log "[$lane] ==== $s 开始：$phase"
      "$PY" "$HERE/driver.py" "$OUT" "$s" $phase >> "$OUT/driver-$s.log" 2>&1; rc=$?
      log "[$lane] ==== $s 结束 rc=$rc"
      [[ $rc -eq 0 ]] || rc_lane=$rc
    done
  done
  return $rc_lane
}
cd "$WT" || fail_done "进不去 $WT" 2
# shellcheck disable=SC2086
run_lane laneA $LANE_A & p1=$!
# shellcheck disable=SC2086
run_lane laneB $LANE_B & p2=$!
wait $p1; rc1=$?; wait $p2; rc2=$?
rc_all=$(( rc1 != 0 ? rc1 : rc2 ))

log "汇总"
"$PY" "$HERE/summarize.py" "$OUT" > "$OUT/summarize.log" 2>&1 || { log "汇总失败（看 summarize.log）"; rc_all=3; }
log "全部结束 rc=$rc_all"
echo "rc=$rc_all $(date '+%F %T')" > "$OUT/DONE"
exit $rc_all
