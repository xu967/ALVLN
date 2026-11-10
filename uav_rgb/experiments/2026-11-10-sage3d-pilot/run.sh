#!/usr/bin/env bash
# SAGE-3D 室内 3DGS 零训练试点（2026-11-10，sage3d；任务书 phase12-20261110）。不训练，不起 Gazebo。
# 问题：室外 4 个 3DGS 场景里，盲前进加 oracle 路线方位护栏（forward+route）的到点成功率是 0.89（107/120，来源
#   exp-bearing-guard-3dgs-20261110/summary.md 第 1 节）。任务书给的室内走廊与门洞净空是 0.2–0.4 m；起跑前用 select_routes.py
#   选出的 40 条路线的 scene.json sage.min_clearance_m（沿 A* 的最小净空）核实：最小 0.22 m，p10 0.28 m，p50 0.40 m，p90 0.60 m。
#   想看这个环境对「局部避障」有没有检验力：护栏只给粗方向，不管前方有没有墙。
#   共用 bearing-guard 的 driver.py / summarize.py（参数化，不复制），护栏同一类，场景换成 5 个 SAGE-3D / InteriorGS 室内 3DGS。
# 设计：sage3d_841150 / 840865 / 841012 / 840139 / 839992（原定 841980 的过滤后路线只剩 2 条且同起点，换成 841012），
#   每场景最多 8 条路线（select_routes.py 按 A* 长度 sage.astar_path_m 分层取），
#   每路线 3 集；seed = 8000 + 100·路线下标（scene.json routes 的下标）+ k，出生扰动 ±0.2 m / ±10°，同 seed 各臂出生位姿相同，逐集配对；
#   hold 1.0；步数 = ceil(max_steps × 0.5) + 15，封顶 100（SITL 起始 50% 电量，约 100 步电量限）；超过 100 的路线标「需满电档」，
#   不进选取池、不跑，单列在 routes_selected.json；stop_on_collision；cf_sitl（lib31）。同一时刻只有 1 个 3DGS 驱动进程，场景顺序跑。
#   臂（顺序即优先级）：teacher（占据 A* 教师）、forward（盲前进）、forward+route（盲前进 + oracle 路线方位护栏）、
#   calswa-s1+route（学生权重 cal_swa_s1_k10 + 同一护栏）。
#   阶段 1 只跑 teacher；阶段 1 结束后按场景算教师 SR(stop)，>= 0.8 的场景才进阶段 2（跑其余 3 臂）。闸门由
#   summarize.py --gate 计算，结果写 gate.json；没过的场景在 summary.md 里标「闸门未过：先查占据代理/坐标系/飞行高度」。
#   所有场景都没过就停：DONE 写 rc=4 与原因，不跑阶段 2。
# 护栏参数起跑前定死，不在这批路线上调：阈值 45°（hold 1.0 单步转角约 31.6°，容许偏 1 步、偏 2 步就纠正，与室外同）；
#   路线前视 1.5 m（室外的 3.0 m 放进室内会越过门框与拐角，护栏方向穿墙；约 5–6 步，仍是粗方向，不是路径跟踪。每步前进：
#   任务书记约 0.25 m，bearing-guard 输出里 forward 臂 FORWARD 步位移均值 0.298 m，n=2477，我从 actions.jsonl 算的）。
#
# 预注册（写于 2026-11-10 20:27，冒烟与起跑之前；护栏参数、阈值、臂的清单、选路线规则起跑前已定，没有用室内数据调过。
#   20:45 只改了上面「问题」里的净空数字与场景名（核实前提、841980 换 841012），预测与阈值没动）。
#   P1（闸门）教师 SR(stop) >= 0.8，用过滤后的路线，各场景分别判。证伪：< 0.8，先查占据代理 / 坐标系 / 飞行高度，不跑其余臂。
#   P2 forward+route 的 SR(stop) 低于室外的 0.89，预测 < 0.7（净空 0.2–0.4 m，45° 容差不足以避开墙 / 门框）。
#      证伪：>= 0.89，说明室内对局部避障没有检验力。判决规则：过闸门场景合计的点估计 < 0.7 记支持，>= 0.89 记证伪，
#      之间记「介于预测与证伪之间」，同时列 Wilson 95% 区间。
#   P3 calswa-s1+route 对 forward+route：预测没有可检出的差别（逐集配对精确 McNemar p > 0.05），与室外一致
#      （室外 b=1, c=3, p=0.625，同一 summary.md 第 4 节）。证伪：学生显著好于或差于盲飞（p <= 0.05）。
#   补充比较（不进判决）：forward+route 对 forward、teacher 对 forward+route、teacher 对 calswa-s1+route。
#   其余要看的：碰撞率、超时率（既没到点也没撞）、SPL(stop)、护栏改写的步占比、按路线长度分桶（<5 m / 5–10 m / >=10 m，用 A* 长度）
#   的表、每条路线单列；Wilson 95% 区间。指标口径同 bearing-guard（metric_version 3dgs.pilot.v2+stop.v1）。
#
# 运行：setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
#   冒烟（1 个场景、每场景取最短 1 条路线、每路线 1 集，GATE_SR=0 让阶段 2 也跑）：
#   SCENES=sage3d_841150 PER_SCENE=1 K=1 GATE_SR=0 OUT=<草稿目录>/smoke setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；每场景日志 $OUT/driver-<场景>.log（grep -v '^step ' 去掉逐步行）；闸门 $OUT/gate.json；
#   汇总随时可跑：<3dgs-venv python> ../2026-11-10-bearing-guard-3dgs/summarize.py $OUT。结束落 $OUT/DONE 与 summary.md。
# 停：kill -- -$(cat $OUT/run.pid)（只按 PGID，不用 pkill -f）。
# 续跑：同一条命令重起；评完的路线（done.json 集数够且无失败）跳过，半成品改名 *.incomplete-<时刻> 后重跑。
#   routes_selected.json 与上次不同（scene.json 变了）就拒绝续跑，免得两批路线的结果混在一个目录里。.lock 互斥，重复启动的那个立即退出。
set -uo pipefail

# ---------------------------------------------------------------- 配置区（改这里）
D=/root/rivermind-data/uav_rgb
WT=${WT:-/root/claude-code-backup-20261028/code/wt/p12-sage3d}   # 代码工作树（分支 p12-sage3d）；起跑后别改里面的 sim/3dgs 与两个实验目录
OUT=${OUT:-$D/exp-sage3d-pilot-20261110}                 # 实验输出（数据盘）
PY=$D/3dgs-venv/bin/python                               # torch 2.7.1 + gsplat 1.5.3；选路线与汇总只用标准库，也用它
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）；同一时刻只许 1 个 3DGS 驱动进程
SCENES_DIR=${SCENES_DIR:-$D/3dgs-scenes}                 # 每个场景 <SCENES_DIR>/<场景>/scene.json（干跑测试时换成合成目录）
SCENES=${SCENES:-"sage3d_841150 sage3d_840865 sage3d_841012 sage3d_840139 sage3d_839992"}   # 场景顺序跑；冒烟只留 1 个
PER_SCENE=${PER_SCENE:-8}                                # 每场景最多 8 条路线（A* 长度等间隔分位）；冒烟设 1（取最短）
K=${K:-3}                                                # 每路线 3 集：每场景 n <= 24，五场景 n <= 120
SEED0=8000                                               # 第 k 集 seed = SEED0 + 100·路线下标 + k；与室外的 7000 错开
JITTER=0.2                                               # 出生 ±0.2 m / ±10°：室外是 0.5 m，室内净空 0.2–0.4 m，0.5 m 会抖进墙里
HOLD=1.0                                                 # 学生是 hold 1.0 训的；盲前进、教师同口径
STEP_FRAC=0.5; STEP_PAD=15; STEP_CAP=100                 # 步数 = ceil(max_steps×0.5)+15（max_steps 按 hold 0.5 算，与室外同口径），封顶 100：SITL 起始 50% 电量约 100 步电量限
GUARD_DEG=45.0                                           # 护栏阈值：单步转角约 31.6°，容许偏 1 步，偏 2 步（>= 45°）就纠正；起跑前定死，与室外同
LOOKAHEAD_M=1.5                                          # 路线前视距离：室外 3.0 m 会越过门框与拐角；起跑前定死
GATE_ARM=${GATE_ARM:-teacher}; GATE_SR=${GATE_SR:-0.8}      # 闸门：该臂按场景的 SR(stop) >= 0.8 才跑阶段 2（预注册 P1）；环境变量可覆盖，GATE_SR=0 放行所有场景（冒烟、放行续跑用）
P2_SUPPORT_LT=0.7; P2_FALSIFY_GE=0.89                    # 预注册 P2：forward+route 合计 SR(stop) < 0.7 支持；>= 0.89（室外值）证伪
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 今天所有闭环实验用的同一个推理二进制
CF_SITL_PROC=$D/tmp/build-cf-sitl-lib31/cf_sitl_proc     # profile cf_sitl_v1:1bc8fedf:lib31，不要动这个目录
GSPLAT_EXT=$D/torch-cache/extensions-sm75                # = pipeline_common.torch_extensions_dir() 在 2080 Ti 上的返回值；已按 sm_75 编好，不应重编
CKPT=$D/exp-swa-calib-20261110/ckpt/cal_swa_s1_k10.bin   # 末 10 轮权重平均 + 类先验校准，训练种子 1（与室外 P3 同一个权重）
# 臂：名字|底座（forward / student / teacher）|权重|护栏（none / route = 路线前视点方位 / goal = waypoint 方位）。顺序即优先级。
# GATE_ARM 的臂在阶段 1 单独跑，其余臂在阶段 2 跑；汇总里的比较编号与臂名见下面 config.json 的写出段。
ARMS=(
  "teacher|teacher||none"
  "forward|forward||none"
  "forward+route|forward||route"
  "calswa-s1+route|student|$CKPT|route"
)
# ----------------------------------------------------------------

SELF=$(readlink -f "${BASH_SOURCE[0]}"); HERE=$(dirname "$SELF")
BG=$HERE/../2026-11-10-bearing-guard-3dgs                # 共用的驱动与汇总（同一个检出里的版本）
DRIVER=${DRIVER:-$BG/driver.py}                          # 只有干跑测试（桩驱动，不碰 GPU）才会覆盖
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
log "开始 PGID=$$ WT=$(git -C "$WT" rev-parse --short HEAD) OUT=$OUT 场景 [$SCENES] 每场景 <= $PER_SCENE 条路线 K=$K"

names=(); ckpts=()
for a in "${ARMS[@]}"; do names+=("${a%%|*}"); IFS='|' read -r _ _ c _ <<< "$a"; [[ -n "$c" ]] && ckpts+=("$c"); done
for f in "$INFER" "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/3dgs/guard.py" "$DRIVER" "$BG/summarize.py" \
         "$HERE/select_routes.py" "${ckpts[@]}"; do
  [[ -s "$f" ]] || fail_done "缺输入 $f" 2
done
for s in $SCENES; do [[ -s "$SCENES_DIR/$s/scene.json" ]] || fail_done "缺场景 $SCENES_DIR/$s/scene.json" 2; done
[[ " ${names[*]} " == *" $GATE_ARM "* ]] || fail_done "GATE_ARM=$GATE_ARM 不在 ARMS 里" 2
REST=(); for n in "${names[@]}"; do [[ "$n" == "$GATE_ARM" ]] || REST+=("$n"); done   # 阶段 2 的臂

cd "$WT" || fail_done "进不去 $WT" 2
# 选路线：先写 .new，与已有的比；不同就拒绝（续跑时 scene.json 变了会让两批路线的结果混在一起）
"$PY" "$HERE/select_routes.py" --scenes-dir "$SCENES_DIR" --scenes "$SCENES" --out "$OUT/routes_selected.json.new" \
      --per-scene "$PER_SCENE" --step-frac "$STEP_FRAC" --step-pad "$STEP_PAD" --step-cap "$STEP_CAP" || fail_done "选路线失败（看 run.log）" 2
if [[ -s "$OUT/routes_selected.json" ]] && ! cmp -s "$OUT/routes_selected.json" "$OUT/routes_selected.json.new"; then
  fail_done "routes_selected.json 与上次不同（scene.json 或选路线参数变了）；新的在 routes_selected.json.new，确认后改名旧目录再跑" 2
fi
mv -f "$OUT/routes_selected.json.new" "$OUT/routes_selected.json"

cp "$SELF" "$OUT/run.sh.snapshot"
{ sha256sum "$INFER" "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/dynamics/e1_mini.json" \
            "$WT/sim/3dgs/guard.py" "$DRIVER" "$BG/summarize.py" "$HERE/select_routes.py" "$OUT/routes_selected.json" "${ckpts[@]}"
  for s in $SCENES; do sha256sum "$SCENES_DIR/$s/scene.json"; done
  echo "git HEAD $(git -C "$WT" rev-parse HEAD) 未提交改动 $(git -C "$WT" status --porcelain | wc -l) 个文件"; } > "$OUT/inputs.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" > "$OUT/gpu.txt" 2>&1 || true   # 只查询，不起计算

export WT SCENES_DIR SCENES K SEED0 JITTER HOLD STEP_FRAC STEP_PAD STEP_CAP GUARD_DEG LOOKAHEAD_M GATE_ARM GATE_SR P2_SUPPORT_LT P2_FALSIFY_GE \
       INFER CF_SITL_PROC OUT
"$PY" - "$OUT/config.json" "${ARMS[@]}" <<'PYEOF' || fail_done "写 config.json 失败" 2
import json, os, sys
e = os.environ
arms = []
for line in sys.argv[2:]:
    name, base, ckpt, guard = line.split("|")
    arms.append({"name": name, "base": base, "ckpt": ckpt or None, "guard": None if guard == "none" else guard})
sel = json.load(open(e["OUT"] + "/routes_selected.json", encoding="utf-8"))
p2 = {"id": "P2", "arm": "forward+route", "support_lt": float(e["P2_SUPPORT_LT"]), "falsify_ge": float(e["P2_FALSIFY_GE"]),
      "texts": {"support": f"支持预测（SR < {e['P2_SUPPORT_LT']}）",
                "falsify": f"证伪（SR >= {e['P2_FALSIFY_GE']}）：室内对局部避障没有检验力",
                "between": f"介于预测与证伪之间（{e['P2_SUPPORT_LT']} <= SR < {e['P2_FALSIFY_GE']}）：预测不成立，也没有被证伪"}}
json.dump({"repo": e["WT"], "scenes_dir": e["SCENES_DIR"], "scenes": e["SCENES"].split(),
           "routes": {s: [r["route"] for r in rs] for s, rs in sel["scenes"].items()},     # 字典：每个场景各自的路线下标
           "routes_selected": e["OUT"] + "/routes_selected.json", "k": int(e["K"]), "seed0": int(e["SEED0"]),
           "jitter": float(e["JITTER"]), "hold": float(e["HOLD"]), "step_frac": float(e["STEP_FRAC"]),
           "step_pad": int(e["STEP_PAD"]), "step_cap": int(e["STEP_CAP"]), "guard_threshold_deg": float(e["GUARD_DEG"]),
           "route_lookahead_m": float(e["LOOKAHEAD_M"]), "arms": arms, "infer_bin": e["INFER"], "cf_sitl_proc": e["CF_SITL_PROC"],
           "title": "SAGE-3D 室内零训练试点（sage3d-pilot）", "overall_label": "合计（teacher 含全部场景，其余臂只含过闸门的场景）",
           "gates": [{"id": "P1", "arm": e["GATE_ARM"], "min_sr": float(e["GATE_SR"]), "note": "闸门未过：先查占据代理/坐标系/飞行高度"}],
           "arm_checks": [p2],
           "main_pairs": [["P3", "calswa-s1+route", "forward+route"]],
           "extra_pairs": [["x1", "forward+route", "forward"], ["x2", "teacher", "forward+route"], ["x3", "teacher", "calswa-s1+route"]],
           "verdict_texts": {"P3": {"ns": "支持预测（p>0.05，无可检出差别）", "up": "证伪（学生+护栏显著好于盲飞+护栏）",
                                    "down": "证伪（学生+护栏显著差于盲飞+护栏）"}}},
          open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PYEOF

run_phase() {    # $1 阶段名，$2 臂（空格分隔），其余是场景：一个场景一个驱动进程（场景加载一次，跑完该阶段的全部臂与路线）；顺序跑，不并行
  local name=$1 arms=$2 s rc rc_all=0; shift 2
  for s in "$@"; do
    log "==== [$name] $s 开始：$arms"
    "$PY" "$DRIVER" "$OUT" "$s" $arms >> "$OUT/driver-$s.log" 2>&1; rc=$?
    log "==== [$name] $s 结束 rc=$rc"
    [[ $rc -eq 0 ]] || rc_all=$rc
  done
  return $rc_all
}
summarize() { "$PY" "$BG/summarize.py" "$OUT" > "$OUT/summarize.log" 2>&1 || { log "汇总失败（看 summarize.log）"; return 1; }; }

# shellcheck disable=SC2086
run_phase "阶段 1" "$GATE_ARM" $SCENES; rc1=$?
PASSED=$("$PY" "$BG/summarize.py" "$OUT" --gate "$GATE_ARM:$GATE_SR") || fail_done "闸门计算失败（summarize.py --gate，看 run.log）" 3
FAILED=""; for s in $SCENES; do [[ " $PASSED " == *" $s "* ]] || FAILED="$FAILED $s"; done
log "闸门 $GATE_ARM SR(stop) >= $GATE_SR：通过 [$PASSED] 未过 [${FAILED# }]（详见 $OUT/gate.json）"
if [[ -z "${PASSED// /}" ]]; then
  summarize
  fail_done "所有场景都没过闸门（$GATE_ARM SR(stop) < $GATE_SR 或阶段 1 没跑完）：先查占据代理 / 坐标系 / 飞行高度，没跑阶段 2；详见 gate.json" 4
fi

# shellcheck disable=SC2086
run_phase "阶段 2" "${REST[*]}" $PASSED; rc2=$?
rc_all=$(( rc1 != 0 ? rc1 : rc2 ))

log "汇总"
summarize || rc_all=3
log "全部结束 rc=$rc_all"
echo "rc=$rc_all $(date '+%F %T') 闸门通过 [$PASSED] 未过 [${FAILED# }]" > "$OUT/DONE"
exit $rc_all
