#!/usr/bin/env bash
# 3DGS 占据 A* 教师的前向净空检查改形状（clearance=swept）与出生两层判空：验收实验（2026-11-11，任务卡 W1-12，分支 p12-w1-12）。
# 不训练，不起 Gazebo。
#
# 问题：W1-11 新教师（容差 18.9°）室内 19 个没到点的集全是撞在前：3 集出生即撞，15 集撞在前进与转向互相切换的那一步，1 集撞在连续前进时，超时 0。
#   教师的前向检查是沿机头一条 0.4 m 的零宽线段（clearance="ray"），看不到转向后首个前进步的位移偏向（偏机头中位 25.5°）和前进后转弯步的前滑
#   （中位 0.066 m）。
# 改动（sim/3dgs/planner.py、collect_3dgs.py、guard.py；数字来源 measure_disp.py / counterfactual.py 的离线输出）：
#   clearance="swept"：机头线段通过之后，再沿「本步前进的预测位移（动作前机体系速度的线性函数）延长 reserve」做一次 occ.sweep；
#   被否决时 |err| 不小于容差一半就按 err 的符号转（turn_swept），否则出 SAFE_NEUTRAL 等一步（wait_swept），连续最多等 2 步。教师版本 v3。
#   出生两层判空（spawn_two_layer，缺省关）：抖动出的候选点在出生高度与巡航高度两层都不碰撞，且 occ.clearance 不低于机体半径加 0.05 m。
#   导引指针（guard.OracleRouteGuide 的 nearest_in_lookahead，缺省关）：本实验不用，K1 的两个停滞集的离线重放见 replay_guide.py。
# 设计：路线、集数、seed、出生扰动、步数预算逐项从 W1-11 验收实验的 config.json 复制（同路线同 seed 才能逐集配对）；
#   驱动共用 bearing-guard 的 driver.py（臂上的可选键 teacher_clearance / spawn_two_layer，缺省时行为与改动前逐位相同）。
#   室外：garden / wanderland_1MlUucug / bicycle / callinstrasse × 路线 0–5 × 每路线 5 集，seed = 7000 + 100·路线 + k，出生扰动 ±0.5 m；
#   室内：sage3d_841150 / 840865 / 841012 / 840139 / 839992 × 各 8 条路线（共 40）× 每路线 3 集，seed = 8000 + 100·路线下标 + k，±0.2 m，步数封顶 100。
#   hold 1.0、容差 18.9°（W1-11 config 里的 teacher_tol_deg）、stop_on_collision、cf_sitl（lib31）。
#   臂 A = swept（出生开关关），与 W1-11 的新教师臂 teacher-tol（W1-11 输出目录的只读符号链接）逐集配对；
#   臂 B = swept-spawn（swept 加出生开关开），只报告；
#   回归臂 ray = 显式 clearance="ray"：室内 sage3d_841150 全部 8 条路线 × 3 集 = 24 集，室外 garden 6 条路线 × 5 集 = 30 集，
#   与 W1-11 的 teacher-tol 臂 actions.jsonl 逐步逐字段比对（浮点按 == 比），不一致不拦截后面的阶段，结果写进 DONE 与报告。
#   阶段：回归 → A → B（同一个任务队列，按这个顺序出队，最多 LANES 个驱动同时跑）→ 分析（含回放归因）。
#
# 预注册（写于 2026-11-12 起跑之前，任务卡 W1-12 原文，不许改；臂 A 的判决由 analyze.py 机械给出，结论以主会话为准）：
# 预注册（写进入口脚本头，不许改）：臂 A 室内到点成功率不低于 0.90，撞在前不高于 0.06，超时不高于 0.03；室外到点不低于 0.97；
# 对 W1-11 的新教师臂逐集配对，室内「新成旧败」减「旧成新败」不少于 8 集。证伪：室内到点低于 0.87，或超时高于 0.05。落在之间记「介于预测与证伪之间」。
# 另报：`wait_swept` 与 `turn_swept` 的步占比；每集平均步数；SAFE_NEUTRAL 标签的占比（教师数据的类别分布会不会被等待步拉偏）；仍然撞在前的集逐集归类。
#   判决不成立时不改参数、不在这批 seed 上重跑调参，写现象和证据。
#
# 运行：setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &      （本进程必须是进程组长，停进程按 PGID）
#   冒烟（单车道，3 分钟以内，不拿锁；只用于看链路通不通，产出不是结果）：
#   SMOKE=1 OUT=<草稿目录>/smoke ONLY_SCENES="sage3d_841150" ROUTES_LIMIT=1 K_LIMIT=1 ALLOW_DIRTY=1 LANES=1 setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；每个驱动的日志 $OUT/driver-<suite>-<臂>-<场景>.log（grep -v '^step ' 去掉逐步行）。
#   结束落 $OUT/DONE（rc、时间、回归、臂 A 的判决、cgroup 起止）、$OUT/analysis/acceptance.{md,json}、regress-*.json。
# 停：kill -- -$(cat $OUT/run.pid)（只按 PGID，不用 pkill -f）。
# 续跑：同一条命令重起；评完的 (臂, 场景, 路线)（done.json 集数够且无失败）跳过，半成品改名 *.incomplete-<时刻> 后重跑。.lock 互斥，重复启动的立即退出。
set -uo pipefail

# ---------------------------------------------------------------- 配置区（改这里）
D=/root/rivermind-data/uav_rgb
SELF=$(readlink -f "${BASH_SOURCE[0]}"); HERE=$(dirname "$SELF")
WT=${WT:-$(cd "$HERE/../.." && pwd)}                       # 代码检出（本脚本所在的工作树）；起跑后别改里面的 sim/ 与本目录
OUT=${OUT:-$D/exp-teacher-clearance-3dgs-20261111}         # 实验输出（数据盘；根分区只剩 3 GB）
PY=$D/3dgs-venv/bin/python                                 # torch + gsplat；分析（回放要加载场景占据）也用它
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740               # 2026-11-11 23:15 起这台机器只有这一张卡，uav_rgb 独占；起跑前用 nvidia-smi -L 核对
GS3D_LOCK=$D/tmp/phase12-cards/gs3d.lock                   # 3DGS 池的锁（11-11 23:55 起 3DGS 池单独一把锁，machine.lock 只管 Gazebo）
LANES=${LANES:-3}                                          # 同时跑的 3DGS 驱动进程数，池规定不超过 3；只影响速度
GPU_MAX_USED_MIB=19000                                     # 拿到锁之后这张卡已用显存超过它就不跑（卡共 22528 MiB）
SIZE_LIMIT_MB=2500                                         # 产出超过它就停整个进程组并写 DONE（任务书：新任务数据盘上限 3 GB）
SCENES_DIR=${SCENES_DIR:-$D/3dgs-scenes}                   # 每个场景 <SCENES_DIR>/<场景>/scene.json
W111=$D/exp-teacher-tol-3dgs-20261110                      # W1-11 验收输出（只读）：config 来源、配对对象、回归对象
W111_ARM=teacher-tol                                       # W1-11 里新教师臂的目录名
ARM_A=swept                                                # 臂 A：clearance=swept，出生开关关
ARM_B=swept-spawn                                          # 臂 B：clearance=swept，出生开关开
ARM_R=ray                                                  # 回归臂：显式 clearance=ray
HOLD=1.0                                                   # 与 W1-11 同；W1-11 config.json 的 hold 不等于它就退出
REGRESS=${REGRESS:-1}                                      # 1 = 做回归；0 = 跳过
REGRESS_OUTDOOR_SCENE=garden                               # 回归用的室外场景：6 路线 × 5 集 = 30 集
REGRESS_INDOOR_SCENE=sage3d_841150                         # 回归用的室内场景：8 路线 × 3 集 = 24 集
RUN_B=${RUN_B:-1}                                          # 1 = 跑臂 B（只报告）；0 = 不跑
SMOKE=${SMOKE:-0}                                          # 1 = 冒烟：不拿锁、不查显存占用、不检查未提交改动之外的东西
ONLY_SCENES=${ONLY_SCENES:-}                               # 冒烟：只跑这些场景（空 = W1-11 的全部场景）
ROUTES_LIMIT=${ROUTES_LIMIT:-0}                            # 冒烟：每场景只取前 N 条路线（0 = 全部）
K_LIMIT=${K_LIMIT:-0}                                      # 冒烟：每路线只跑前 N 集（0 = 全部）；seed 仍是 seed0 + 100·路线 + k，与 W1-11 逐集对得上
ALLOW_DIRTY=${ALLOW_DIRTY:-0}                              # 1 = 允许 sim/ 与本目录有未提交改动（只用于冒烟）；正式跑必须是提交过的代码
CF_SITL_PROC=$D/tmp/build-cf-sitl-lib31/cf_sitl_proc       # profile cf_sitl_v1:1bc8fedf:lib31，与 W1-11 同一个，不要动这个目录
GSPLAT_EXT=$D/torch-cache/extensions-sm75                  # 已按 sm_75 编好的 gsplat 扩展，不应重编
# ----------------------------------------------------------------

BG=$HERE/../2026-11-10-bearing-guard-3dgs                  # 共用的驱动（同一个检出里的版本，只读使用，除臂上两个可选键之外未改）
DRIVER=${DRIVER:-$BG/driver.py}
[[ "$(ps -o pgid= -p $$ | tr -d ' ')" == "$$" ]] || { echo "请用 setsid nohup bash run.sh 起（停进程按 PGID，要求本进程是组长）" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB，临时文件放数据盘
export TORCH_EXTENSIONS_DIR=$GSPLAT_EXT CF_SITL_PROC
unset TORCH_CUDA_ARCH_LIST                                 # 带着旧值会让 gsplat 按别的架构重编
mkdir -p "$OUT/tmp" "$OUT/analysis"
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
log() { echo "$(date '+%F %T') $*"; }
fail_done() { log "$1"; echo "rc=$2 $(date '+%F %T') $1" > "$OUT/DONE"; exit "$2"; }
cg() { awk '/^(usage_usec|nr_throttled|throttled_usec) / {printf "%s=%s ", $1, $2}' /sys/fs/cgroup/cpu.stat 2>/dev/null; }   # cgroup 的 CPU 用量与被限流次数
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
HEAD0=$(git -C "$WT" rev-parse HEAD)
log "开始 PGID=$$ WT=$WT HEAD=${HEAD0:0:7} OUT=$OUT LANES=$LANES SMOKE=$SMOKE 场景限制=[${ONLY_SCENES:-全部}] 路线限制=$ROUTES_LIMIT 集限制=$K_LIMIT 回归=$REGRESS 臂B=$RUN_B"

# ---- 起跑前检查（排队之前）：输入、代码是否提交、有没有这张卡、磁盘
for f in "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$DRIVER" "$HERE/analyze.py" "$WT/sim/3dgs/planner.py" "$WT/sim/3dgs/collect_3dgs.py" \
         "$WT/sim/dynamics/e1_mini.json" "$W111/outdoor/config.json" "$W111/indoor/config.json" "$W111/outdoor/$W111_ARM/garden/r0/done.json" \
         "$W111/indoor/$W111_ARM/sage3d_841150/r9/done.json"; do
  [[ -s "$f" ]] || fail_done "缺输入 $f" 2
done
dirty() { git -C "$WT" status --porcelain -- sim experiments/2026-11-10-bearing-guard-3dgs experiments/2026-11-11-teacher-clearance-3dgs; }
if [[ "$ALLOW_DIRTY" != 1 && -n "$(dirty)" ]]; then
  fail_done "sim/、bearing-guard 驱动目录或本目录有未提交改动：先提交再起跑（冒烟可设 ALLOW_DIRTY=1）" 2
fi
nvidia-smi -L 2>/dev/null | grep -q "$GPU" || fail_done "nvidia-smi -L 里没有 $GPU（无卡模式不跑：正式实验必须是 GPU 渲染，不退回 CPU）" 4
AVAIL_GB=$(df --output=avail -BG "$OUT" | tail -1 | tr -dc 0-9)
(( AVAIL_GB >= 4 )) || fail_done "数据盘只剩 ${AVAIL_GB} GB < 4 GB，不起跑" 2

# ---- 3DGS 池的锁：阻塞式，持有到本脚本退出（冒烟：单车道、3 分钟以内，不拿）
if [[ "$SMOKE" != 1 ]]; then
  log "等 3DGS 池的锁 $GS3D_LOCK（前面可能排着别的卡的实验，每 10 分钟看一次，不空转）"
  exec 8>> "$GS3D_LOCK"; flock 8   # 11-11 23:55 起 3DGS 池单独一把锁（machine.lock 只管 Gazebo）
  log "拿到 3DGS 池的锁（持有到本脚本退出）"
  [[ "$(git -C "$WT" rev-parse HEAD)" == "$HEAD0" ]] || fail_done "排队期间 $WT 的 HEAD 变了（起跑时 ${HEAD0:0:7}），不跑" 2
  if [[ "$ALLOW_DIRTY" != 1 && -n "$(dirty)" ]]; then fail_done "排队期间 sim/、驱动目录或本目录出现未提交改动，不跑" 2; fi
  USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" 2>/dev/null | head -1 | tr -d ' ')
  [[ "$USED" =~ ^[0-9]+$ ]] || fail_done "查不到 GPU $GPU 的显存占用（nvidia-smi 输出 [$USED]）" 2
  (( USED <= GPU_MAX_USED_MIB )) || fail_done "拿到锁后 GPU $GPU 已用 ${USED} MiB > ${GPU_MAX_USED_MIB} MiB，不起跑" 2
  log "GPU 已用 ${USED} MiB，数据盘剩 ${AVAIL_GB} GB"
fi
T_START=$(date +%s)
CG_START=$(cg)
log "cgroup 起点：$CG_START"

TOL=$("$PY" -c "import importlib, sys; sys.path.insert(0, '$WT'); print(repr(importlib.import_module('sim.3dgs.planner').yaw_tol_deg($HOLD)))") \
  || fail_done "算不出 yaw_tol_deg($HOLD)" 2
log "教师容差 yaw_tol_deg(hold=$HOLD) = $TOL°（臂 A / B / 回归臂都用这个值，与 W1-11 的 teacher-tol 臂同）"

# ---- 写 config.json：路线 / 集数 / seed / 步数预算从 W1-11 的 config 复制
export WT OUT D SCENES_DIR W111 W111_ARM ARM_A ARM_B ARM_R HOLD TOL ONLY_SCENES ROUTES_LIMIT K_LIMIT REGRESS REGRESS_OUTDOOR_SCENE REGRESS_INDOOR_SCENE RUN_B CF_SITL_PROC
"$PY" - <<'PYEOF' || fail_done "写 config.json 失败（看 run.log）" 2
import json, os
e = os.environ
only = e["ONLY_SCENES"].split()
rl, kl, hold, tol = int(e["ROUTES_LIMIT"]), int(e["K_LIMIT"]), float(e["HOLD"]), float(e["TOL"])
COPY = ("scenes_dir", "scenes", "routes", "k", "seed0", "jitter", "hold", "step_frac", "step_pad", "step_cap",
        "guard_threshold_deg", "route_lookahead_m", "routes_selected")


def arm(name, **kw):
    return {"name": name, "base": "teacher", "ckpt": None, "guard": None, **kw}


ARMS_MAIN = [arm(e["ARM_A"], teacher_clearance="swept")]
if e["RUN_B"] == "1":
    ARMS_MAIN.append(arm(e["ARM_B"], teacher_clearance="swept", spawn_two_layer=True))
ARMS_REGRESS = [arm(e["ARM_R"], teacher_clearance="ray")]


def make(suite, scenes_only, arms, title):
    c0 = json.load(open(os.path.join(e["W111"], suite.replace("regress-", ""), "config.json"), encoding="utf-8"))
    if abs(float(c0["hold"]) - hold) > 1e-9:
        raise SystemExit(f"{suite}: W1-11 config 的 hold={c0['hold']} != HOLD={hold}")
    if abs(float(c0["teacher_tol_deg"]) - tol) > 1e-12:
        raise SystemExit(f"{suite}: W1-11 config 的 teacher_tol_deg={c0['teacher_tol_deg']!r} != yaw_tol_deg({hold}) = {tol!r}")
    c = {k: c0[k] for k in COPY if k in c0}
    c["scenes_dir"] = e["SCENES_DIR"]
    scenes = [s for s in c["scenes"] if (not scenes_only or s in scenes_only)]
    if not scenes:
        return None
    routes = {s: (c["routes"][s] if isinstance(c["routes"], dict) else c["routes"]) for s in scenes}
    if rl:
        routes = {s: r[:rl] for s, r in routes.items()}
    c.update(scenes=scenes, routes=routes)
    if kl:
        c["k"] = min(kl, c["k"])
    c.update(repo=e["WT"], arms=arms, teacher_tol_deg=tol, cf_sitl_proc=e["CF_SITL_PROC"], title=title,
             arm_a=e["ARM_A"], arm_b=e["ARM_B"] if e["RUN_B"] == "1" else None, w111_arm=e["W111_ARM"], w111_dir=os.path.join(e["W111"], suite.replace("regress-", "")))
    out = os.path.join(e["OUT"], suite)
    os.makedirs(out, exist_ok=True)
    json.dump(c, open(os.path.join(out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return c


made = {"outdoor": make("outdoor", only, ARMS_MAIN, "3DGS 教师前向净空 swept：室外（臂 A / B 对 W1-11 新教师臂）"),
        "indoor": make("indoor", only, ARMS_MAIN, "3DGS 教师前向净空 swept：室内（臂 A / B 对 W1-11 新教师臂）")}
if e["REGRESS"] == "1":
    for suite, scene in (("regress-outdoor", e["REGRESS_OUTDOOR_SCENE"]), ("regress-indoor", e["REGRESS_INDOOR_SCENE"])):
        if only and scene not in only:
            continue
        made[suite] = make(suite, [scene], ARMS_REGRESS, f"确定性回归：显式 clearance=ray 重跑 W1-11 的新教师臂（{scene}）")
print("config 已写：", {n: (None if c is None else (len(c["scenes"]), c["k"], sum(len(r) for r in c["routes"].values()))) for n, c in made.items()})
PYEOF

# W1-11 新教师臂 = 只读符号链接（analyze.py 按 <suite>/<臂>/ 读）；已存在且不是符号链接就拒绝
for suite in outdoor indoor; do
  [[ -s "$OUT/$suite/config.json" ]] || continue
  link=$OUT/$suite/$W111_ARM
  if [[ -e "$link" && ! -L "$link" ]]; then fail_done "$link 已存在且不是符号链接，不覆盖" 2; fi
  ln -sfn "$W111/$suite/$W111_ARM" "$link"
done

cp "$SELF" "$OUT/run.sh.snapshot"
ALL_SCENES=$("$PY" -c "
import json, os
s = set()
for d in ('outdoor', 'indoor'):
    p = '$OUT/' + d + '/config.json'
    if os.path.exists(p):
        s.update(json.load(open(p))['scenes'])
print(' '.join(sorted(s)))")
for s in $ALL_SCENES; do [[ -s "$SCENES_DIR/$s/scene.json" ]] || fail_done "缺场景 $SCENES_DIR/$s/scene.json" 2; done
{ sha256sum "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/dynamics/e1_mini.json" "$WT/sim/3dgs/planner.py" \
            "$WT/sim/3dgs/collect_3dgs.py" "$WT/sim/3dgs/guard.py" "$WT/sim/3dgs/collision.py" "$DRIVER" "$HERE/analyze.py" "$HERE/measure_disp.py" \
            "$HERE/counterfactual.py" "$W111/outdoor/config.json" "$W111/indoor/config.json" "$OUT"/*/config.json
  for s in $ALL_SCENES; do sha256sum "$SCENES_DIR/$s/scene.json"; done
  echo "git HEAD $(git -C "$WT" rev-parse HEAD) 分支 $(git -C "$WT" rev-parse --abbrev-ref HEAD) 未提交改动 $(git -C "$WT" status --porcelain | wc -l) 个文件"; } > "$OUT/inputs.txt"
nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader -i "$GPU" > "$OUT/gpu.txt" 2>&1 || true   # 只查询，不起计算

# ---- 任务队列：一个任务 = 一个驱动进程（场景加载一次，跑完该 suite 里该臂的全部路线）。出队顺序：回归 → 臂 A → 臂 B
run_task() {    # $1 suite，$2 场景，$3 臂
  local suite=$1 s=$2 a=$3 rc
  log "==== [$suite/$a] $s 开始"
  "$PY" "$DRIVER" "$OUT/$suite" "$s" "$a" >> "$OUT/driver-$suite-$a-$s.log" 2>&1; rc=$?
  log "==== [$suite/$a] $s 结束 rc=$rc"
  return $rc
}
export -f log run_task
export PY DRIVER OUT
scenes_of() { "$PY" -c "import json, os; p = '$OUT/$1/config.json'; print(' '.join(json.load(open(p))['scenes']) if os.path.exists(p) else '')"; }
TASKS=()
if [[ "$REGRESS" == 1 ]]; then
  for suite in regress-outdoor regress-indoor; do for s in $(scenes_of $suite); do TASKS+=("$suite $s $ARM_R"); done; done
fi
for arm in $ARM_A $([[ "$RUN_B" == 1 ]] && echo "$ARM_B"); do
  for suite in outdoor indoor; do for s in $(scenes_of $suite); do TASKS+=("$suite $s $arm"); done; done
done
log "任务 ${#TASKS[@]} 个，并行 $LANES：${TASKS[*]}"
( printf '%s\n' "${TASKS[@]}" | xargs -P "$LANES" -L 1 bash -c 'run_task "$0" "$1" "$2"' ) &
XPID=$!
# 产出体积与显存守护：主进程每 15 s 看一次，不另起后台进程（免得退出后残留）；超过体积上限停整个进程组
tick=0; peak=0
while kill -0 "$XPID" 2>/dev/null; do
  mb=$(du -sm "$OUT" 2>/dev/null | cut -f1)
  gpu_raw=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU" 2>/dev/null | head -1); gpu_mb=${gpu_raw% MiB}
  [[ "$gpu_mb" =~ ^[0-9]+$ ]] || gpu_mb=0
  (( gpu_mb > peak )) && peak=$gpu_mb
  (( tick % 8 == 0 )) && log "GPU 显存占用 ${gpu_raw:-无法读取}（15 s 采样的峰值 ${peak} MiB，整张卡含别的进程）；产出 ${mb} MB"    # 每 2 分钟一行
  tick=$((tick + 1))
  if [[ ${mb:-0} -gt $SIZE_LIMIT_MB ]]; then
    log "产出 ${mb} MB 超过 ${SIZE_LIMIT_MB} MB，停止整个进程组（任务书：超了停下报告）"
    echo "rc=5 $(date '+%F %T') 产出 ${mb} MB 超过 ${SIZE_LIMIT_MB} MB，已停（kill -- -$$）" > "$OUT/DONE"
    kill -TERM -- -$$
  fi
  sleep 15
done
wait "$XPID"; rc_all=$?
echo "$peak" > "$OUT/gpu_peak_mib.txt"
log "驱动全部结束 rc=$rc_all（xargs：123 = 有任务非零退出）；产出 $(du -sm "$OUT" | cut -f1) MB；显存峰值 ${peak} MiB（15 s 采样，整张卡含别的进程）"

# ---- 回归：臂 ray 对 W1-11 的 teacher-tol 臂，逐集逐步逐字段
REG_RESULT="未做"
if [[ "$REGRESS" == 1 ]]; then
  REG_RC=0
  for suite in regress-outdoor regress-indoor; do
    [[ -d "$OUT/$suite/$ARM_R" ]] || continue
    "$PY" "$HERE/analyze.py" regress "$OUT/$suite/$ARM_R" "$W111/${suite#regress-}/$W111_ARM" --json "$OUT/analysis/$suite.json" >> "$OUT/analysis/regress.log" 2>&1 || REG_RC=1
  done
  [[ -s "$OUT/analysis/regress.log" ]] && log "回归：$(grep -h '^确定性回归' "$OUT/analysis/regress.log" | tr '\n' ';')"
  REG_RESULT=$([[ $REG_RC -eq 0 ]] && echo "逐步一致" || echo "有差异（看 analysis/regress.log）")
fi

log "阶段 分析"
"$PY" "$HERE/analyze.py" acceptance "$OUT" --replay --occ-cache "$OUT/tmp/occ_cache" > "$OUT/analysis/acceptance.stdout" 2>&1 || { log "analyze.py acceptance 失败（看 analysis/acceptance.stdout）"; rc_all=3; }
VERDICT=$("$PY" -c "
import json
j = json.load(open('$OUT/analysis/acceptance.json'))
print(j['verdict_a']['text'])" 2>/dev/null || echo "判决未取到")
CG_END=$(cg)
log "cgroup 终点：$CG_END；用时 $(( $(date +%s) - T_START )) s"
log "全部结束 rc=$rc_all 回归=$REG_RESULT 臂 A 判决 $VERDICT"
echo "rc=$rc_all $(date '+%F %T') 回归=$REG_RESULT 臂A判决=$VERDICT cgroup起点=[$CG_START] 终点=[$CG_END]" > "$OUT/DONE"
exit $rc_all
