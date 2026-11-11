#!/usr/bin/env bash
# 3DGS 占据 A* 教师的航向容差随转角步推导：验收实验（2026-11-10，任务卡 W1-11，分支 p12-w1-11）。不训练，不起 Gazebo。
#
# 问题：hold 1.0 下教师（sim/3dgs/planner.py OccAstarTeacher）航向容差 12° 的死区宽 24°，小于一个转角步（实测 26.3–34.7°，
#   measure_turn_step.py 在旧数据上量的：室外 2597 步、室内 2967 步，全部 > 24°），误差落在 (12°, 步长 − 12°) 时转一步过冲到另一侧容差之外，
#   再转回来，左右来回转（极限环）。旧数据（analyze.py 复算，与 sage3d / bearing3dgs 报告的数字一致）：室外教师 100/120 = 0.83，
#   19 个超时全含交替段；室内教师 76/120 = 0.63，29 个超时里 25 个含交替段；室内容差 18° 的诊断（非预注册）101/120 = 0.84，超时 0。
# 改动：yaw_tol_deg(hold) = max(12°, 0.55 × 名义转角步)，名义转角步 = yaw_rate_rps × hold（e1_mini.json 的 0.6 rad/s）：
#   hold 0.5 仍是 12°（与旧教师逐步相同），hold 1.0 为 18.91°。系数 0.55 来自任务卡，核对见 measure_turn_step.py 的输出：
#   同向连转实测中位数 34.38° = 名义值；按最大实测步 34.7° 算，无环的最小系数是 0.504，0.55 留 9% 余量。系数不在评测 seed 上调。
# 设计：新臂 teacher-tol = 新容差的教师，旧臂 teacher = 旧实验已有的教师臂（容差 12°，旧输出目录的只读符号链接）；
#   路线、集数、seed、出生扰动、步数预算逐项从旧实验的 config.json 复制（同路线同 seed 才能逐集配对），驱动 / 汇总共用 bearing-guard 的
#   driver.py 与 summarize.py（只读使用，不改），新容差通过 config.json 已有的 teacher_tol_deg 键传入，数值由 planner.yaw_tol_deg(HOLD) 算出。
#   室外：garden / wanderland_1MlUucug / bicycle / callinstrasse × 路线 0–5 × 每路线 5 集，seed = 7000 + 100·路线 + k，出生扰动 ±0.5 m；
#   室内：sage3d_841150 / 840865 / 841012 / 840139 / 839992 × 各 8 条路线（共 40）× 每路线 3 集，seed = 8000 + 100·路线下标 + k，±0.2 m，步数封顶 100。
#   hold 1.0、stop_on_collision、cf_sitl（lib31）。阶段：0 确定性回归 → 1 室外 → 2 室内 → 3 分析（含回放归因）。同一时刻只有 1 个 3DGS 驱动进程。
#   阶段 0（回归）：显式 tol_deg = 12° 重跑旧教师的室外 garden 与室内 sage3d_841150 各一整个场景，与旧 actions.jsonl 逐步逐字段精确比对，
#   证明「新代码 + 显式 12°」与旧教师轨迹逐位相同（也量出仿真自己的确定性）。回归不一致不拦截阶段 1–2，结果写进 DONE 与报告。
#
# 预注册（写于 2026-11-10 22:24，冒烟与起跑之前；容差公式、系数、阈值、场景、路线、seed 起跑前已定，没有在这批 seed 上调过）。
#   写预注册之前已经在旧数据上跑过的东西（不是本实验的结果）：measure_turn_step.py（转角步分布）、analyze.py（复算旧教师与 18° 诊断的数字，
#   与两份报告一致）、analyze.py --replay（回放旧教师与 18° 诊断的交替段：室内旧教师 12° 的 61 段里 49 段只有航向误差超容差、12 段两种都有、
#   0 段只有前向被堵转向；18° 诊断残留 2 集含交替段，其中 1 段（sage3d_839992 r93 ep1，7 步）也是只有航向误差超容差类；回放复算与记录的
#   action_expert 不一致 0/2880 步）。所以下面不预测「新教师没有交替段」：目标点（前视点）随位姿跳变时，不靠极限环也会有 turn_err 型交替。
#   主比较只有三个：
#   M1 室外：新教师到点成功（SR(stop)）点估计 >= 0.92（任务卡阈值；旧 100/120 = 0.83），逐集配对（精确 McNemar）预测 b > c 且 p <= 0.05。
#      证伪：点估计 < 0.92。
#   M2 室内：新教师到点成功点估计 >= 0.80（任务卡阈值；旧 76/120 = 0.63），逐集配对预测 b > c 且 p <= 0.05。证伪：点估计 < 0.80。
#   M3 超时：新教师超时集数不超过旧教师的 1/4：室外 <= 4（旧 19）、室内 <= 7（旧 29）。1/4 是我定的，依据是 18° 诊断室内超时 29 → 0；
#      室外没有对照。证伪：任一边超过。它检验「超时的主因是转角步大于死区宽造成的左右交替」，超过时看剩下的超时是不是还含交替段。
#   判决由 analyze.py 机械给出，结论以主会话为准。判决不成立时不改系数、不在这批 seed 上重跑调参，写现象和证据。
#   不进判决但要报的：超时集里含交替段的集数、碰撞率（任一步 / 撞在前）、SPL(stop)、室内按路线长度分桶（<5 m / 5–10 m / >=10 m，A* 长度）、
#   每场景数字、仍含交替段的集的清单与回放归因（只有航向误差超容差 / 只有前向被堵转向 / 两种都有，规则不改只报告）、回归结果。
#
# 运行：setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
#   冒烟（每场景 1 条路线 × 1 集，只跑 2 个场景，回归也只跑这些）：
#   OUT=<草稿目录>/smoke ONLY_SCENES="garden sage3d_841150" ROUTES_LIMIT=1 K_LIMIT=1 ALLOW_DIRTY=1 setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；每个驱动的日志 $OUT/driver-<阶段>-<场景>.log（grep -v '^step ' 去掉逐步行）。
#   结束落 $OUT/DONE（rc、时间、回归与两个判决）、$OUT/analysis/acceptance.{md,json}、turn_step.{md,json}、<suite>/summary.md。
# 停：kill -- -$(cat $OUT/run.pid)（只按 PGID，不用 pkill -f）。
# 续跑：同一条命令重起；评完的路线（done.json 集数够且无失败）跳过，半成品改名 *.incomplete-<时刻> 后重跑。.lock 互斥，重复启动的立即退出。
set -uo pipefail

# ---------------------------------------------------------------- 配置区（改这里；路线 / 集数 / seed / 步数预算不在这里：从旧 config.json 复制）
D=/root/rivermind-data/uav_rgb
WT=${WT:-/root/claude-code-backup-20261028/code/wt/p12-w1-11}   # 代码工作树（分支 p12-w1-11）；起跑后别改里面的 sim/3dgs 与本目录
OUT=${OUT:-$D/exp-teacher-tol-3dgs-20261110}             # 实验输出（数据盘；根分区只剩 2 GB）
PY=$D/3dgs-venv/bin/python                               # torch + gsplat；分析（回放要加载场景占据）也用它
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）
GPU_MAX_USED_MIB=19000                                   # 起跑时这张卡已用显存超过它就不跑：另一个 3DGS 实验占 1–3 GB，本驱动再要 1–3 GB，卡共 22528 MiB
SCENES_DIR=${SCENES_DIR:-$D/3dgs-scenes}                 # 每个场景 <SCENES_DIR>/<场景>/scene.json
OLD_OUTDOOR=$D/exp-bearing-guard-3dgs-20261110           # 旧教师臂所在目录（室外 4 场景 × 6 路线 × 5 集，容差 12°）
OLD_INDOOR=$D/exp-sage3d-pilot-20261110                  # 旧教师臂所在目录（室内 5 场景 × 8 路线 × 3 集，容差 12°）
OLD_ARM=teacher                                          # 旧实验里教师臂的目录名（summarize.py 按 metrics.json 里记的臂名过滤，所以必须同名）
NEW_ARM=teacher-tol                                      # 本实验新教师臂的目录名
HOLD=1.0                                                 # 与旧实验同；旧 config.json 的 hold 不等于它就退出
ACCEPT_OUTDOOR=0.92                                      # 预注册 M1 阈值（任务卡）
ACCEPT_INDOOR=0.80                                       # 预注册 M2 阈值（任务卡）
TIMEOUTS_MAX_OUTDOOR=4                                   # 预注册 M3 上界：旧教师超时 19 的 1/4 取整（见上面预注册）
TIMEOUTS_MAX_INDOOR=7                                    # 预注册 M3 上界：旧教师超时 29 的 1/4 取整
REGRESS=${REGRESS:-1}                                    # 1 = 先做阶段 0 确定性回归；0 = 跳过
REGRESS_OUTDOOR_SCENE=garden                             # 回归用的室外场景：旧教师 30 集里 6 个超时、全含交替段，既有环也有正常集
REGRESS_INDOOR_SCENE=sage3d_841150                       # 回归用的室内场景：旧教师 24 集里 10 个超时（9 个含交替段）
REGRESS_TOL=12.0                                         # 回归的显式容差 = 旧默认 YAW_TOL_DEG
ONLY_SCENES=${ONLY_SCENES:-}                             # 冒烟：只跑这些场景（空 = 旧实验的全部场景）
ROUTES_LIMIT=${ROUTES_LIMIT:-0}                          # 冒烟：每场景只取前 N 条路线（0 = 全部）
K_LIMIT=${K_LIMIT:-0}                                    # 冒烟：每路线只跑前 N 集（0 = 全部）；seed 仍是 seed0 + 100·路线 + k，与旧教师逐集对得上
ALLOW_DIRTY=${ALLOW_DIRTY:-0}                            # 1 = 允许 sim/ 与本目录有未提交改动（只用于冒烟）；正式跑必须是提交过的代码，inputs.txt 才对得上
CF_SITL_PROC=$D/tmp/build-cf-sitl-lib31/cf_sitl_proc     # profile cf_sitl_v1:1bc8fedf:lib31，与旧实验同一个，不要动这个目录
GSPLAT_EXT=$D/torch-cache/extensions-sm75                # 已按 sm_75 编好的 gsplat 扩展，不应重编
# ----------------------------------------------------------------

SELF=$(readlink -f "${BASH_SOURCE[0]}"); HERE=$(dirname "$SELF")
BG=$HERE/../2026-11-10-bearing-guard-3dgs                # 共用的驱动与汇总（同一个检出里的版本，只读使用）
DRIVER=${DRIVER:-$BG/driver.py}
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2 GB，临时文件放数据盘
export TORCH_EXTENSIONS_DIR=$GSPLAT_EXT CF_SITL_PROC
unset TORCH_CUDA_ARCH_LIST                               # 带着旧值会让 gsplat 按别的架构重编
mkdir -p "$OUT/tmp" "$OUT/analysis"
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
log() { echo "$(date '+%F %T') $*"; }
fail_done() { log "$1"; echo "rc=$2 $(date '+%F %T') $1" > "$OUT/DONE"; exit "$2"; }
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
log "开始 PGID=$(ps -o pgid= -p $$ | tr -d ' ') WT=$(git -C "$WT" rev-parse --short HEAD) OUT=$OUT 场景限制=[${ONLY_SCENES:-全部}] 路线限制=$ROUTES_LIMIT 集限制=$K_LIMIT 回归=$REGRESS"

# ---- 起跑前检查：输入、代码是否提交、显存、磁盘
for f in "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$DRIVER" "$BG/summarize.py" "$HERE/analyze.py" "$HERE/measure_turn_step.py" \
         "$WT/sim/3dgs/planner.py" "$WT/sim/dynamics/e1_mini.json" "$OLD_OUTDOOR/config.json" "$OLD_INDOOR/config.json" \
         "$OLD_INDOOR/routes_selected.json"; do
  [[ -s "$f" ]] || fail_done "缺输入 $f" 2
done
[[ -d "$OLD_OUTDOOR/$OLD_ARM" && -d "$OLD_INDOOR/$OLD_ARM" ]] || fail_done "旧教师臂目录不存在（$OLD_OUTDOOR/$OLD_ARM 或 $OLD_INDOOR/$OLD_ARM）" 2
if [[ "$ALLOW_DIRTY" != 1 && -n "$(git -C "$WT" status --porcelain -- sim experiments/2026-11-10-teacher-tol-3dgs)" ]]; then
  fail_done "sim/ 或 experiments/2026-11-10-teacher-tol-3dgs 有未提交改动：先提交再起跑（冒烟可设 ALLOW_DIRTY=1）" 2
fi
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -d ' ')
[[ "$USED" =~ ^[0-9]+$ ]] || fail_done "查不到 GPU $GPU 的显存占用（nvidia-smi 输出 [$USED]）" 2
(( USED <= GPU_MAX_USED_MIB )) || fail_done "GPU $GPU 已用 ${USED} MiB > ${GPU_MAX_USED_MIB} MiB，不起跑（另一个 3DGS 实验占着）" 2
AVAIL_GB=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc 0-9)
(( AVAIL_GB >= 4 )) || fail_done "数据盘只剩 ${AVAIL_GB} GB < 4 GB，不起跑" 2
log "GPU 已用 ${USED} MiB，数据盘剩 ${AVAIL_GB} GB"

TOL=$("$PY" -c "import importlib, sys; sys.path.insert(0, '$WT'); print(repr(importlib.import_module('sim.3dgs.planner').yaw_tol_deg($HOLD)))") \
  || fail_done "算不出 yaw_tol_deg($HOLD)" 2
log "新教师容差 yaw_tol_deg(hold=$HOLD) = $TOL°；旧教师 12°；回归显式容差 $REGRESS_TOL°"

# ---- 写 config.json：路线 / 集数 / seed / 步数预算从旧 config 复制
export WT OUT D SCENES_DIR OLD_OUTDOOR OLD_INDOOR OLD_ARM NEW_ARM HOLD TOL REGRESS_TOL ACCEPT_OUTDOOR ACCEPT_INDOOR TIMEOUTS_MAX_OUTDOOR TIMEOUTS_MAX_INDOOR ONLY_SCENES ROUTES_LIMIT K_LIMIT \
       REGRESS REGRESS_OUTDOOR_SCENE REGRESS_INDOOR_SCENE CF_SITL_PROC
"$PY" - <<'PYEOF' || fail_done "写 config.json 失败（看 run.log）" 2
import json, os
e = os.environ
only = e["ONLY_SCENES"].split()
rl, kl, hold = int(e["ROUTES_LIMIT"]), int(e["K_LIMIT"]), float(e["HOLD"])
COPY = ("scenes_dir", "scenes", "routes", "k", "seed0", "jitter", "hold", "step_frac", "step_pad", "step_cap",
        "guard_threshold_deg", "route_lookahead_m", "routes_selected")     # 后两个只为 summarize.py 的表头
arms = [{"name": e["NEW_ARM"], "base": "teacher", "ckpt": None, "guard": None},
        {"name": e["OLD_ARM"], "base": "teacher", "ckpt": None, "guard": None}]
verdict = {"T1": {"ns": "新旧教师无可检出差别（p>0.05）", "up": "新教师显著好于旧教师", "down": "新教师显著差于旧教师"}}


def make(old_dir, name, scenes_only, tol, accept, tmax, title):
    c0 = json.load(open(os.path.join(old_dir, "config.json"), encoding="utf-8"))
    if abs(float(c0["hold"]) - hold) > 1e-9:
        raise SystemExit(f"{old_dir}/config.json 的 hold={c0['hold']} != HOLD={hold}")
    c = {k: c0[k] for k in COPY if k in c0}
    c["scenes_dir"] = e["SCENES_DIR"]
    scenes = [s for s in c["scenes"] if (not scenes_only or s in scenes_only)]
    if not scenes:
        raise SystemExit(f"{name}: 没有可跑的场景（ONLY_SCENES={e['ONLY_SCENES']}）")
    routes = {s: (c["routes"][s] if isinstance(c["routes"], dict) else c["routes"]) for s in scenes}
    if rl:
        routes = {s: r[:rl] for s, r in routes.items()}
    c.update(scenes=scenes, routes=routes)
    if kl:
        c["k"] = min(kl, c["k"])
    c.update(repo=e["WT"], arms=arms, new_arm=e["NEW_ARM"], old_arm=e["OLD_ARM"], teacher_tol_deg=tol, accept_sr=accept, accept_timeouts_max=tmax,
             title=title, overall_label="合计", main_pairs=[["T1", e["NEW_ARM"], e["OLD_ARM"]]], verdict_texts=verdict,
             cf_sitl_proc=e["CF_SITL_PROC"])
    out = os.path.join(e["OUT"], name)
    os.makedirs(out, exist_ok=True)
    json.dump(c, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return c


outdoor = make(e["OLD_OUTDOOR"], "outdoor", only, float(e["TOL"]), float(e["ACCEPT_OUTDOOR"]), int(e["TIMEOUTS_MAX_OUTDOOR"]),
               "3DGS 教师航向容差随转角步推导：室外（新教师 vs 旧教师）")
indoor = make(e["OLD_INDOOR"], "indoor", only, float(e["TOL"]), float(e["ACCEPT_INDOOR"]), int(e["TIMEOUTS_MAX_INDOOR"]),
              "3DGS 教师航向容差随转角步推导：室内（新教师 vs 旧教师）")
if e["REGRESS"] == "1":
    for name, old_dir, scene in (("regress-outdoor", e["OLD_OUTDOOR"], e["REGRESS_OUTDOOR_SCENE"]),
                                 ("regress-indoor", e["OLD_INDOOR"], e["REGRESS_INDOOR_SCENE"])):
        if only and scene not in only:
            continue
        make(old_dir, name, [scene], float(e["REGRESS_TOL"]), None, None, f"确定性回归：显式 tol_deg={e['REGRESS_TOL']} 重跑旧教师（{scene}）")
print("config 已写：", {n: (len(c['scenes']), c['k'], sum(len(r) for r in c['routes'].values())) for n, c in (("outdoor", outdoor), ("indoor", indoor))})
PYEOF

# 旧教师臂 = 旧输出目录的只读符号链接（summarize.py / analyze.py 按 <suite>/<臂>/ 读）；已存在且不是符号链接就拒绝
for pair in "outdoor:$OLD_OUTDOOR" "indoor:$OLD_INDOOR"; do
  suite=${pair%%:*}; old=${pair#*:}; link=$OUT/$suite/$OLD_ARM
  if [[ -e "$link" && ! -L "$link" ]]; then fail_done "$link 已存在且不是符号链接，不覆盖" 2; fi
  ln -sfn "$old/$OLD_ARM" "$link"
done

cp "$SELF" "$OUT/run.sh.snapshot"
ALL_SCENES=$("$PY" -c "
import json
s = set()
for d in ('outdoor', 'indoor'):
    s.update(json.load(open('$OUT/' + d + '/config.json'))['scenes'])
print(' '.join(sorted(s)))")
for s in $ALL_SCENES; do [[ -s "$SCENES_DIR/$s/scene.json" ]] || fail_done "缺场景 $SCENES_DIR/$s/scene.json" 2; done
{ sha256sum "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/dynamics/e1_mini.json" "$WT/sim/3dgs/planner.py" \
            "$WT/sim/3dgs/collect_3dgs.py" "$DRIVER" "$BG/summarize.py" "$HERE/analyze.py" "$HERE/measure_turn_step.py" \
            "$OLD_OUTDOOR/config.json" "$OLD_INDOOR/config.json" "$OLD_INDOOR/routes_selected.json" "$OUT"/*/config.json
  for s in $ALL_SCENES; do sha256sum "$SCENES_DIR/$s/scene.json"; done
  echo "git HEAD $(git -C "$WT" rev-parse HEAD) 分支 $(git -C "$WT" rev-parse --abbrev-ref HEAD) 未提交改动 $(git -C "$WT" status --porcelain | wc -l) 个文件"; } > "$OUT/inputs.txt"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader -i "$GPU" > "$OUT/gpu.txt" 2>&1 || true   # 只查询，不起计算

run_suite() {    # $1 阶段名，$2 config 目录名（<OUT>/<name>）：一个场景一个驱动进程（场景加载一次，跑完该场景全部路线）；顺序跑，不并行
  local label=$1 name=$2 s rc rc_all=0 scenes
  scenes=$("$PY" -c "import json; print(' '.join(json.load(open('$OUT/$name/config.json'))['scenes']))")
  for s in $scenes; do
    log "==== [$label] $s 开始"
    "$PY" "$DRIVER" "$OUT/$name" "$s" "$NEW_ARM" >> "$OUT/driver-$name-$s.log" 2>&1; rc=$?
    log "==== [$label] $s 结束 rc=$rc"
    [[ $rc -eq 0 ]] || rc_all=$rc
  done
  return $rc_all
}

rc_all=0; REG_RESULT="未做"
if [[ "$REGRESS" == 1 ]]; then
  for name in regress-outdoor regress-indoor; do
    [[ -s "$OUT/$name/config.json" ]] || continue
    # 回归臂目录名也叫 $NEW_ARM（驱动按臂名找 base=teacher）；容差由该 config 的 teacher_tol_deg = $REGRESS_TOL 给
    run_suite "阶段 0 回归" "$name" || rc_all=$?
  done
  REG_RC=0
  for pair in "regress-outdoor:$OLD_OUTDOOR" "regress-indoor:$OLD_INDOOR"; do
    name=${pair%%:*}; old=${pair#*:}
    [[ -d "$OUT/$name/$NEW_ARM" ]] || continue
    "$PY" "$HERE/analyze.py" regress "$OUT/$name/$NEW_ARM" "$old/$OLD_ARM" --json "$OUT/analysis/$name.json" >> "$OUT/analysis/regress.log" 2>&1 || REG_RC=1
  done
  [[ -s "$OUT/analysis/regress.log" ]] && log "回归：$(grep -h '^确定性回归' "$OUT/analysis/regress.log" | tr '\n' ';')"
  REG_RESULT=$([[ $REG_RC -eq 0 ]] && echo "逐步一致" || echo "有差异（看 analysis/regress.log）")
fi

run_suite "阶段 1 室外" outdoor || rc_all=$?
run_suite "阶段 2 室内" indoor || rc_all=$?

log "阶段 3 分析"
# summarize.py（bearing-guard 的，不改）把旧臂符号链接里的全部集都数进去；冒烟压低了路线 / 集数时旧臂集数比请求多，分母会小于分子，所以只在完整跑时用它
if [[ "$ROUTES_LIMIT" == 0 && "$K_LIMIT" == 0 && -z "$ONLY_SCENES" ]]; then
  for suite in outdoor indoor; do
    "$PY" "$BG/summarize.py" "$OUT/$suite" > "$OUT/$suite/summarize.log" 2>&1 || { log "$suite 的 summarize.py 失败（看 summarize.log）"; rc_all=3; }
  done
fi
"$PY" "$HERE/measure_turn_step.py" --hold "$HOLD" --out "$OUT/analysis" \
      "室外旧教师 12°=$OLD_OUTDOOR/$OLD_ARM" "室内旧教师 12°=$OLD_INDOOR/$OLD_ARM" \
      "室外新教师=$OUT/outdoor/$NEW_ARM" "室内新教师=$OUT/indoor/$NEW_ARM" > "$OUT/analysis/measure_turn_step.log" 2>&1 || { log "measure_turn_step.py 失败"; rc_all=3; }
"$PY" "$HERE/analyze.py" acceptance "$OUT" --replay > "$OUT/analysis/acceptance.stdout" 2>&1 || { log "analyze.py acceptance 失败（看 analysis/acceptance.stdout）"; rc_all=3; }
VERDICTS=$("$PY" -c "
import json
j = json.load(open('$OUT/analysis/acceptance.json'))['suites']
print(' '.join(f\"{k}={v['verdict']}({v['arms']['$NEW_ARM']['k_stop']}/{v['arms']['$NEW_ARM']['n_req']})\" for k, v in j.items()))" 2>/dev/null || echo "判决未取到")
log "全部结束 rc=$rc_all 回归=$REG_RESULT 判决 $VERDICTS"
echo "rc=$rc_all $(date '+%F %T') 回归=$REG_RESULT $VERDICTS" > "$OUT/DONE"
exit $rc_all
