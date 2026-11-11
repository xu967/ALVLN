#!/usr/bin/env bash
# 3DGS 上的规则层上限（2026-11-10，rule-oracle-3dgs；任务卡 K1，docs/briefs/2026-11-10-phase12-task-cards.md「卡 K1」）。零训练，不起 Gazebo。
# 问题：感知完美时，「各列可通行距离 + 固定规则 v1 + 路线方向」这套分解在室内外 3DGS 场景上能到多高。11-10 晚已量的另外两个角：
#   盲飞加路线护栏（forward+route）室内 0.60、10 m 以上 0.30、碰撞率 0.43，室外 0.89（107/120）；教师（特权 A*）室内 0.63，容差修正后 0.84。
# 设计（全部来自任务卡，没有改）：
#   策略 sim/3dgs/rule_oracle.py：每步 20 列（3°，60° 视场，上限 4.0 m）列距离，两个来源：occ = 碰撞判定用的占据栅格沿飞行高度步进，
#   depth = 渲染深度反投影到机体系、留机体包络上下各 0.10 m 内的点按方位分列取最近；航向计数 n = round(−err / yaw_step_deg)，
#   err = bearing_error_deg(...)（正 = 目标在左），目标在左 n < 0，规则回转动作是 YAW_LEFT；规则文件 sim/tools/rule_policy.py 未改，只传参数覆盖。
#   室内 = 试点（sage3d-pilot）的 5 个场景、同 40 条路线、每条 3 集、同 seed（8000 + 100·路线 + k）；
#   室外 = garden、bicycle、wanderland_1MlUucug、callinstrasse 各 6 条路线 × 5 集、同 seed（7000 + 100·路线 + k，与 bearing-guard 相同）；hold 1.0。
#   臂（顺序即优先级）：forward+route（参照，重跑用于配对；护栏阈值 45°、路线前视室内 1.5 m / 室外 3.0 m，与前两个实验相同）、
#   rule-occ+route（主臂）、rule-depth+route、rule-occ+goal（航向参照是 waypoint）；仅室内再加 rule-occ+route@0.5 与 forward+route@0.5（hold 0.5，步数预算 ×2）。
#   没有教师臂。
# 开发路线（只用来选 r_safe 与 d_look，不碰正式的 40 条室内与 24 条室外路线）：室内 5 个场景里没进试点 40 条的合格路线每场景取 4 条（A* 长度等间隔分位），
#   r_safe ∈ {0.20, 0.25, 0.35} × d_look ∈ {0.8, 1.2} 各跑 1 集（hold 1.0，seed = 9000 + 100·路线），取到点成功 SR(stop) 最高的一组；
#   并列取 r_safe 大的（任务卡）；仍并列时的补充规则（任务卡没写，dev 起跑前补的）：撞在前率低的，再取 d_look 大的。
#   选定值写进下面配置区后冻结；其余规则参数（mem_steps 8、n_cap 3、tie_frac、sum_clip）不动；室外用同一组，不另调。
#   dev 已跑完（2026-11-10 22:38–22:44，$OUT_ROOT/dev/）：6 组里 r_safe 0.25、d_look 1.2 的到点成功最高（14/20 = 0.70，次高三组各 13/20），已冻结在配置区。
#   yaw_step_deg 取 3DGS 动力学在该 hold 下实测的转角步中位数：hold 1.0 = 31.7°（yaw_step_stats.py 量 bearing-guard 与 sage3d-pilot 全部轨迹 22,988 个转向步，
#   中位数 31.68°）；hold 0.5 = 17.13°（dev 阶段 forward+route@0.5 的 590 个转向步中位数）；两个值同样冻结在配置区。
# 预注册（任务卡原文，逐字抄自「卡 K1」，起跑前定死；下面三行是原文，不改字）：
#   主臂室内到点成功不低于 0.80，10 m 以上的路线不低于 0.60，碰撞率不高于 0.15；室外不低于 0.84（参照 0.89 减 0.05），碰撞率不高于 0.10。
#   证伪：主臂室内低于 0.70。含义是地平线带的列距离加这条规则在室内不够，要么动作粒度太粗（看 hold 0.5 的臂），要么需要更多几何（门框、家具高度）。
#   主比较三个：`rule-occ+route` 对 `forward+route`（室内，配对 McNemar）；`rule-depth+route` 对 `rule-occ+route`（室内）；`rule-occ+goal` 对 `rule-occ+route`（室内）。
# 判读口径（执行者补的，任务卡没写；起跑前定，不看结果改）：「主臂」= rule-occ+route；「到点成功」= SR(stop)（首次进入 waypoint 半径早于首次碰撞），
#   室内 120 集合计、室外 120 集合计，点估计对阈值；「10 m 以上」= 室内路线 A* 长度 ≥ 10 m（试点 routes_selected.json 的 >=10 m 桶）；
#   「碰撞率」= 整集任一步碰撞（含到点之后，与前两个实验的表同口径，因为规则臂到点后不停，会在 waypoint 附近继续飞），
#   撞在前（没到点就撞）与超时率并列报告；判决：SR ≥ 阈值记「达到预测」，室内 < 0.70 记「证伪」，之间记「介于预测与证伪之间」，同时列 Wilson 95% 区间。
#   三个主比较用 SR(stop) 的逐集配对精确 McNemar（同场景同路线同 seed）。
# 另报：depth 与 occ 两种列距离在同一位姿上的一致性（每列绝对差的中位数与 p90；depth 比 occ 远 0.3 m 以上的列占比），失败集按「撞、超时（含左右交替）、其他」分类，
#   按路线长度分桶（室内 <5 m / 5–10 m / >=10 m），规则各条目（V1-back / fwd / more / side / last / left / cap）的触发占比。
#
# 运行（正式实验不需要任何环境变量；dev 已跑完，PHASE 缺省 = formal）：
#   cd /root/claude-code-backup-20261028/code/wt/p12-k1 && setsid nohup bash experiments/2026-11-10-rule-oracle-3dgs/run.sh > /dev/null 2>&1 < /dev/null &
#   没有 GPU（nvidia-smi -L 列不出下面 GPU 的 UUID）时脚本立即退出，DONE 写「无卡模式，未起跑」，不退回 CPU。
#   formal 拒绝带着覆盖配置的环境变量起跑（ROUTES_LIMIT / K_INDOOR / K_OUTDOOR / INDOOR_SCENES / OUTDOOR_SCENES / ROUTES_OUTDOOR / OUT_ROOT / DRIVER ...），
#   免得上一条命令残留的环境变量悄悄改了正式设计；冒烟与干跑才允许覆盖：
#   冒烟（有卡）：SMOKE=1 INDOOR_SCENES=sage3d_841150 OUTDOOR_SCENES=garden ROUTES_LIMIT=1 K_INDOOR=1 K_OUTDOOR=1 OUT_ROOT=<草稿目录>/smoke setsid nohup bash run.sh ...
#   干跑（无卡，桩驱动，不渲染）：DRY_RUN=1 DRIVER=<桩驱动.py> OUT_ROOT=<草稿目录>/dry setsid -w bash run.sh；DRY_RUN=1 拒绝与真驱动同用（不会退回 CPU 渲染）。
#   开发路线选参（已跑完，复现用）：PHASE=dev setsid nohup bash run.sh ...
# 进度：tail -f $OUT_ROOT/<阶段>/run.log；每场景日志 $OUT/<室内|室外>/driver-<场景>.log（grep -v '^step ' 去掉逐步行）。结束落 $OUT/DONE 与 summary.md（formal）/ dev_selection.md（dev）。
# 停：kill -- -$(cat $OUT/run.pid)（只按 PGID，不用 pkill -f）。产出超过 SIZE_LIMIT_MB（1500）时守护自己停并写 DONE rc=5。
# 续跑：同一条命令重起；评完的路线（done.json 集数够且无失败）跳过，半成品改名 *.incomplete-<时刻> 后重跑。routes 文件与上次不同就拒绝续跑。
#   $OUT_ROOT/.lock 互斥：dev 与 formal 不会同时跑，也不会重复启动。
set -uo pipefail
SELF=$(readlink -f "${BASH_SOURCE[0]}"); HERE=$(dirname "$SELF")
# 起跑前记下哪些「可覆盖」的变量被环境预先设了（配置区的 ${VAR:-默认} 会把它们吃掉）；formal 用它拒绝带着覆盖起跑。
# WT（代码检出）与 LANES（并行度）不在名单里：WT 缺省 = 本脚本所在的检出，起跑记录写进 inputs.txt（git HEAD）；LANES 只影响速度
OVERRIDDEN=""
for v in OUT_ROOT SIZE_LIMIT_MB SCENES_DIR INDOOR_SCENES OUTDOOR_SCENES PILOT_ROUTES ROUTES_LIMIT K_INDOOR K_OUTDOOR ROUTES_OUTDOOR DEV_PER_SCENE DRIVER; do
  [[ -n "${!v+x}" ]] && OVERRIDDEN="$OVERRIDDEN $v"
done

# ---------------------------------------------------------------- 配置区（改这里；${VAR:-默认} 的项可被环境变量覆盖，其余写死）
D=/root/rivermind-data/uav_rgb
WT=${WT:-$(cd "$HERE/../.." && pwd)}                                # 代码检出 = 本脚本所在的检出（现在是工作树 wt/p12-k1，分支 p12-k1）；起跑后别改里面的 sim/3dgs 与两个实验目录
PHASE=${PHASE:-formal}                                              # dev = 室内开发路线选参；formal = 正式（室内 + 室外）
OUT_ROOT=${OUT_ROOT:-$D/exp-rule-oracle-3dgs-20261110}              # 产出根目录（数据盘）；阶段目录 $OUT_ROOT/dev 与 $OUT_ROOT/formal
PY=$D/3dgs-venv/bin/python                                          # torch 2.7.1 + gsplat 1.5.3；选路线与汇总只用标准库，也用它
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740                        # 11-11 起机器上只有这一张卡（RTX 2080 Ti），用户 11-11 19:23 指示 uav_rgb 放在这张卡上跑；dev 阶段（11-10）用的是 GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70
LANES=${LANES:-3}                                                   # 同时跑的 3DGS 驱动进程数：cgroup 12 核，卡上没有别的 3DGS 实验；并行度只影响速度
SIZE_LIMIT_MB=${SIZE_LIMIT_MB:-1500}                                # 主会话 11-11 批准；按 dev 实测约 0.46 MB/集推算正式约 0.6–0.8 GB，1000 的余量太小；数据盘剩 11 GB。超了就停下报告
SCENES_DIR=${SCENES_DIR:-$D/3dgs-scenes}                            # 每个场景 <SCENES_DIR>/<场景>/scene.json
INDOOR_SCENES=${INDOOR_SCENES:-"sage3d_841150 sage3d_840865 sage3d_841012 sage3d_840139 sage3d_839992"}   # 试点的 5 个场景
OUTDOOR_SCENES=${OUTDOOR_SCENES:-"garden wanderland_1MlUucug bicycle callinstrasse"}                      # bearing-guard 的 4 个场景与顺序
PILOT_ROUTES=${PILOT_ROUTES:-$D/exp-sage3d-pilot-20261110/routes_selected.json}   # 室内正式路线 = 试点选定的 40 条（同路线同 seed 才能配对）
ROUTES_LIMIT=${ROUTES_LIMIT:-}                                      # 冒烟用：每场景只取前 N 条路线（试点文件里按 A* 长度升序）；空 = 全部
# 室内（同 sage3d-pilot）
K_INDOOR=${K_INDOOR:-3}                                             # 每路线 3 集，40 路线共 120 集
SEED0_INDOOR=8000                                                   # seed = 8000 + 100·路线下标 + k，与试点相同，forward+route 才能逐集对上旧结果
JITTER_INDOOR=0.2                                                   # 出生 ±0.2 m / ±10°：室内净空 0.2–0.4 m，0.5 m 会抖进墙里（同试点）
LOOKAHEAD_INDOOR=1.5                                                # 路线前视距离（同试点）：室外的 3.0 m 放进室内会越过门框与拐角
STEP_CAP_INDOOR=100                                                 # 步数封顶（SITL 起始 50% 电量约 100 步），同试点；hold 0.5 的臂在封顶后乘 2
# 室外（同 bearing-guard）
K_OUTDOOR=${K_OUTDOOR:-5}                                           # 每路线 5 集，24 路线共 120 集
SEED0_OUTDOOR=7000                                                  # 同 bearing-guard
JITTER_OUTDOOR=0.5                                                  # 同 bearing-guard
LOOKAHEAD_OUTDOOR=3.0                                               # 同 bearing-guard
ROUTES_OUTDOOR=${ROUTES_OUTDOOR:-"0 1 2 3 4 5"}                     # 每场景 6 条路线（路线下标）
# 共用
HOLD=1.0                                                            # 与前两个实验相同；@0.5 的臂在臂上覆盖
STEP_FRAC=0.5; STEP_PAD=15                                          # 步数 = ceil(max_steps × 0.5) + 15（同前两个实验）
GUARD_DEG=45.0                                                      # forward+route 的护栏阈值（同前两个实验）
DROP_TRACE=true                                                     # 每集删 trace.jsonl（只有视频渲染要它）：1,320 集留着约多 0.2 GB，actions / oracle 的位姿足够复原轨迹
# 开发路线选参（PHASE=dev）
DEV_PER_SCENE=${DEV_PER_SCENE:-4}                                   # 每场景 4 条开发路线
DEV_K=1                                                             # 每组合每路线 1 集
DEV_SEED0=9000                                                      # 开发 seed，与正式 8000 / 7000 错开
DEV_R_GRID="0.20 0.25 0.35"                                         # r_safe 候选（规则的 r_safe 已含机体半径，占据栅格再膨胀一次，所以含小于 0.35 的值）
DEV_D_GRID="0.8 1.2"                                                # d_look 候选
# 规则参数（冻结；mem_steps 8、n_cap 3、tie_frac、sum_clip 不传，用 RULE_V1 的值；这四个值写死，不接受环境变量覆盖）
YAW_STEP_H10=31.7                                                   # 来源 $D/tmp/phase12-cards/k1/yaw_step_h10.json：hold 1.0 转向步中位数 31.68°（22,988 步），取一位小数
R_SAFE=0.25                                                         # 来源 $D/exp-rule-oracle-3dgs-20261110/dev/indoor/dev_selection.md 第 1 名：14/20 = 0.70
D_LOOK=1.2                                                          # 来源同上（臂 rocc-r0.25-d1.2）
YAW_STEP_H05=17.13                                                  # 来源 $D/exp-rule-oracle-3dgs-20261110/dev/yaw_step_h05.json：hold 0.5 转向步中位数 17.13°（590 步）
OLD_INDOOR=$D/exp-sage3d-pilot-20261110                             # 汇总时逐集对照 forward+route 用的旧室内实验（只读）
OLD_OUTDOOR=$D/exp-bearing-guard-3dgs-20261110                      # 同上，旧室外实验（只读）
MACHINE_LOCK=$D/tmp/phase12-cards/machine.lock                      # 今晚重实验排队用的机器锁（空文件）：formal 阻塞式 flock，持有到本脚本退出；dev、冒烟、干跑不拿
CF_SITL_PROC=$D/tmp/build-cf-sitl-lib31/cf_sitl_proc                # profile cf_sitl_v1:1bc8fedf:lib31，不要动这个目录
GSPLAT_EXT=$D/torch-cache/extensions-sm75                           # 已按 sm_75 编好，不应重编
# ----------------------------------------------------------------

BG=$HERE/../2026-11-10-bearing-guard-3dgs                           # 共用的驱动（同一个检出里的版本）
DRIVER=${DRIVER:-$BG/driver.py}                                     # 只有干跑测试（桩驱动，不碰 GPU）才会覆盖
OUT=$OUT_ROOT/$PHASE
[[ "$PHASE" == "dev" || "$PHASE" == "formal" ]] || { echo "PHASE 只能是 dev 或 formal，得到 [$PHASE]" >&2; exit 2; }
[[ "$(ps -o pgid= -p $$ | tr -d ' ')" == "$$" ]] || { echo "请用 setsid nohup bash run.sh 起（停进程按 PGID，要求本进程是组长）" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2 GB，临时文件放数据盘
export TORCH_EXTENSIONS_DIR=$GSPLAT_EXT CF_SITL_PROC
unset TORCH_CUDA_ARCH_LIST                                          # 带着旧值会让 gsplat 按别的架构重编
mkdir -p "$OUT/tmp"
# 无卡检查：钉死的 GPU 不在 nvidia-smi -L 里就退出，不退回 CPU（正式实验必须是预注册的 GPU 渲染；桩渲染器 / CPU 渲染的数字不能当结果）。
# 干跑（DRY_RUN=1）只允许与桩驱动同用，桩驱动不渲染，所以不查卡。
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  [[ "$DRIVER" != "$BG/driver.py" && -s "$DRIVER" ]] || { echo "DRY_RUN=1 必须同时用 DRIVER=<桩驱动.py> 覆盖真驱动（不许无卡跑真渲染）" >&2; exit 2; }
elif ! nvidia-smi -L 2>/dev/null | grep -q "$GPU"; then
  echo "rc=4 $(date '+%F %T') 无卡模式，未起跑（nvidia-smi -L 里没有 $GPU；不退回 CPU）" > "$OUT/DONE"
  echo "无卡模式，未起跑（nvidia-smi -L 里没有 $GPU；不退回 CPU）" >&2
  exit 4
fi
if [[ "$PHASE" == "formal" && -n "$OVERRIDDEN" && "${SMOKE:-0}" != 1 && "${DRY_RUN:-0}" != 1 ]]; then
  echo "rc=2 $(date '+%F %T') formal 不接受环境变量覆盖：$OVERRIDDEN（冒烟请加 SMOKE=1，干跑请加 DRY_RUN=1）" > "$OUT/DONE"
  echo "formal 不接受环境变量覆盖：$OVERRIDDEN（冒烟请加 SMOKE=1，干跑请加 DRY_RUN=1）" >&2
  exit 2
fi
exec 9> "$OUT_ROOT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT_ROOT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
log() { echo "$(date '+%F %T') $*"; }
fail_done() { log "$1"; echo "rc=$2 $(date '+%F %T') $1" > "$OUT/DONE"; exit "$2"; }
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
log "开始 PGID=$$ PHASE=$PHASE WT=$(git -C "$WT" rev-parse --short HEAD) OUT=$OUT LANES=$LANES"
log "GPU 占用（启动前）：$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i "$GPU" 2>&1)"
if [[ "$PHASE" == "formal" && "${SMOKE:-0}" != 1 && "${DRY_RUN:-0}" != 1 ]]; then
  log "等机器锁 $MACHINE_LOCK"
  exec 8>> "$MACHINE_LOCK"; flock 8
  log "拿到机器锁 $MACHINE_LOCK（持有到本脚本退出）"
fi

if [[ "$PHASE" == "formal" ]]; then
  [[ -n "$R_SAFE" && -n "$D_LOOK" && -n "$YAW_STEP_H05" ]] || fail_done "formal 要求 R_SAFE / D_LOOK / YAW_STEP_H05 已冻结在配置区" 2
fi
CLASSES="indoor"; [[ "$PHASE" == "formal" ]] && CLASSES="indoor outdoor"
for f in "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/3dgs/guard.py" "$WT/sim/3dgs/rule_oracle.py" "$WT/sim/tools/rule_policy.py" \
         "$DRIVER" "$HERE/select_dev_routes.py" "$HERE/select_params.py" "$HERE/yaw_step_stats.py" "$PILOT_ROUTES"; do
  [[ -s "$f" ]] || fail_done "缺输入 $f" 2
done
for s in $INDOOR_SCENES $OUTDOOR_SCENES; do [[ -s "$SCENES_DIR/$s/scene.json" ]] || fail_done "缺场景 $SCENES_DIR/$s/scene.json" 2; done
cd "$WT" || fail_done "进不去 $WT" 2

# 路线文件：dev 现选、formal 室内沿用试点；先写 .new 与已有的比，不同就拒绝（续跑时 scene.json 变了会让两批路线的结果混在一个目录里）
mkdir -p "$OUT/indoor"
if [[ "$PHASE" == "dev" ]]; then
  ROUTES_FILE=$OUT/indoor/routes_dev.json
  "$PY" "$HERE/select_dev_routes.py" --scenes-dir "$SCENES_DIR" --scenes "$INDOOR_SCENES" --pilot-routes "$PILOT_ROUTES" --out "$ROUTES_FILE.new" \
        --per-scene "$DEV_PER_SCENE" --step-frac "$STEP_FRAC" --step-pad "$STEP_PAD" --step-cap "$STEP_CAP_INDOOR" || fail_done "选开发路线失败（看 run.log）" 2
else
  ROUTES_FILE=$OUT/indoor/routes_selected.json
  cp "$PILOT_ROUTES" "$ROUTES_FILE.new" || fail_done "复制试点路线文件失败" 2
fi
if [[ -s "$ROUTES_FILE" ]] && ! cmp -s "$ROUTES_FILE" "$ROUTES_FILE.new"; then
  fail_done "$(basename "$ROUTES_FILE") 与上次不同（scene.json 或选路线参数变了）；新的在 $ROUTES_FILE.new，确认后改名旧目录再跑" 2
fi
mv -f "$ROUTES_FILE.new" "$ROUTES_FILE"

cp "$SELF" "$OUT/run.sh.snapshot"
{ sha256sum "$CF_SITL_PROC" "$GSPLAT_EXT/gsplat_cuda/gsplat_cuda.so" "$WT/sim/dynamics/e1_mini.json" "$WT/sim/3dgs/guard.py" "$WT/sim/3dgs/rule_oracle.py" \
            "$WT/sim/tools/rule_policy.py" "$WT/sim/3dgs/collision.py" "$WT/sim/3dgs/planner.py" "$DRIVER" "$HERE/select_dev_routes.py" "$HERE/select_params.py" \
            "$HERE/summarize.py" "$PILOT_ROUTES" "$ROUTES_FILE"
  for s in $INDOOR_SCENES $OUTDOOR_SCENES; do sha256sum "$SCENES_DIR/$s/scene.json"; done
  echo "git HEAD $(git -C "$WT" rev-parse HEAD) 未提交改动 $(git -C "$WT" status --porcelain | wc -l) 个文件"; } > "$OUT/inputs.txt" 2>&1
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "$GPU" > "$OUT/gpu.txt" 2>&1 || true   # 只查询，不起计算

export WT SCENES_DIR PHASE OUT ROUTES_FILE ROUTES_LIMIT HOLD STEP_FRAC STEP_PAD GUARD_DEG DROP_TRACE CF_SITL_PROC \
       INDOOR_SCENES OUTDOOR_SCENES K_INDOOR SEED0_INDOOR JITTER_INDOOR LOOKAHEAD_INDOOR STEP_CAP_INDOOR \
       K_OUTDOOR SEED0_OUTDOOR JITTER_OUTDOOR LOOKAHEAD_OUTDOOR ROUTES_OUTDOOR \
       DEV_K DEV_SEED0 DEV_R_GRID DEV_D_GRID YAW_STEP_H10 R_SAFE D_LOOK YAW_STEP_H05
for cls in $CLASSES; do
  mkdir -p "$OUT/$cls"
  CLS=$cls "$PY" - "$OUT/$cls/config.json" <<'PYEOF' || fail_done "写 $cls/config.json 失败" 2
import json, os, sys
e = os.environ
cls, phase = e["CLS"], e["PHASE"]
H10, H05 = float(e["YAW_STEP_H10"]), (float(e["YAW_STEP_H05"]) if e["YAW_STEP_H05"] else None)
FWD = {"base": "forward", "ckpt": None}


def arms_dev():
    out = [{"name": f"rocc-r{float(r):.2f}-d{float(d):.1f}", "base": "rule-occ", "ckpt": None, "guard": "route",
            "rule_params": {"r_safe": float(r), "d_look": float(d)}} for r in e["DEV_R_GRID"].split() for d in e["DEV_D_GRID"].split()]
    out.append({"name": "forward+route@0.5", **FWD, "guard": "route", "hold": 0.5, "step_mult": 2})      # 只为量 hold 0.5 的转角步
    return out


def arms_formal():
    out = [{"name": "forward+route", **FWD, "guard": "route"},
           {"name": "rule-occ+route", "base": "rule-occ", "ckpt": None, "guard": "route"},
           {"name": "rule-depth+route", "base": "rule-depth", "ckpt": None, "guard": "route"},
           {"name": "rule-occ+goal", "base": "rule-occ", "ckpt": None, "guard": "goal"}]
    if cls == "indoor":
        out += [{"name": "rule-occ+route@0.5", "base": "rule-occ", "ckpt": None, "guard": "route", "hold": 0.5, "step_mult": 2,
                 "rule_params": {"yaw_step_deg": H05}},
                {"name": "forward+route@0.5", **FWD, "guard": "route", "hold": 0.5, "step_mult": 2}]
    return out


common = {"repo": e["WT"], "scenes_dir": e["SCENES_DIR"], "hold": float(e["HOLD"]), "step_frac": float(e["STEP_FRAC"]),
          "step_pad": int(e["STEP_PAD"]), "guard_threshold_deg": float(e["GUARD_DEG"]), "drop_trace": e["DROP_TRACE"] == "true",
          "infer_bin": "unused (no student arm)", "cf_sitl_proc": e["CF_SITL_PROC"], "phase": phase, "class": cls}
if cls == "indoor":
    sel = json.load(open(e["ROUTES_FILE"], encoding="utf-8"))
    lim = int(e["ROUTES_LIMIT"]) if e["ROUTES_LIMIT"] else None
    scenes = e["INDOOR_SCENES"].split()
    routes = {s: [r["route"] for r in sel["scenes"].get(s, [])][:lim] for s in scenes}
    dev = phase == "dev"
    cfg = {**common, "scenes": scenes, "routes": routes, "routes_selected": e["ROUTES_FILE"],
           "k": int(e["DEV_K"] if dev else e["K_INDOOR"]), "seed0": int(e["DEV_SEED0"] if dev else e["SEED0_INDOOR"]),
           "jitter": float(e["JITTER_INDOOR"]), "step_cap": int(e["STEP_CAP_INDOOR"]), "route_lookahead_m": float(e["LOOKAHEAD_INDOOR"]),
           "title": "室内（SAGE-3D）" + ("开发路线选参" if dev else "规则层上限")}
else:
    scenes = e["OUTDOOR_SCENES"].split()
    cfg = {**common, "scenes": scenes, "routes": [int(x) for x in e["ROUTES_OUTDOOR"].split()][:int(e["ROUTES_LIMIT"]) if e["ROUTES_LIMIT"] else None],
           "k": int(e["K_OUTDOOR"]), "seed0": int(e["SEED0_OUTDOOR"]), "jitter": float(e["JITTER_OUTDOOR"]),
           "route_lookahead_m": float(e["LOOKAHEAD_OUTDOOR"]), "title": "室外（3DGS 四场景）规则层上限"}
cfg["arms"] = arms_dev() if phase == "dev" else arms_formal()
cfg["rule_params"] = {"yaw_step_deg": H10} if phase == "dev" else {"r_safe": float(e["R_SAFE"]), "d_look": float(e["D_LOOK"]), "yaw_step_deg": H10}
cfg["frozen"] = {"yaw_step_h10": H10, "yaw_step_h05": H05, "r_safe": e["R_SAFE"] or None, "d_look": e["D_LOOK"] or None}
json.dump(cfg, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=2)
PYEOF
done

run_task() {    # $1 类别（indoor / outdoor），$2 场景：一个场景一个驱动进程（场景加载一次，跑完该类别的全部臂与路线）
  local cls=$1 s=$2 rc
  log "==== [$cls] $s 开始"
  "$PY" "$DRIVER" "$OUT/$cls" "$s" >> "$OUT/$cls/driver-$s.log" 2>&1; rc=$?
  log "==== [$cls] $s 结束 rc=$rc"
  return $rc
}
export -f log run_task
export PY DRIVER
TASKS=()
if [[ "$PHASE" == "formal" ]]; then for s in $OUTDOOR_SCENES; do TASKS+=("outdoor $s"); done; fi
for s in $INDOOR_SCENES; do TASKS+=("indoor $s"); done
log "任务 ${#TASKS[@]} 个，并行 $LANES：${TASKS[*]}"
( printf '%s\n' "${TASKS[@]}" | xargs -P "$LANES" -L 1 bash -c 'run_task "$0" "$1"' ) &
XPID=$!
# 产出体积守护：整个 $OUT_ROOT 超过上限就停掉整个进程组并写 DONE（任务书：超了停下报告）；主进程每 15 s 看一次，不另起后台进程（免得退出后残留）
tick=0; peak=0
while kill -0 "$XPID" 2>/dev/null; do
  mb=$(du -sm "$OUT_ROOT" 2>/dev/null | cut -f1)
  gpu_raw=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i "$GPU" 2>/dev/null | head -1); gpu_mb=${gpu_raw% MiB}
  [[ "$gpu_mb" =~ ^[0-9]+$ ]] || gpu_mb=0
  (( gpu_mb > peak )) && peak=$gpu_mb
  (( tick % 8 == 0 )) && log "GPU 显存占用 ${gpu_raw:-无法读取}（15 s 采样的峰值 ${peak} MiB，整张卡含别的进程）；产出 ${mb} MB"    # 每 8 × 15 s = 2 分钟一行
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
log "驱动全部结束 rc=$rc_all（xargs：123 = 有任务非零退出）；产出 $(du -sm "$OUT_ROOT" | cut -f1) MB；显存峰值 ${peak} MiB（15 s 采样，整张卡含别的进程）"

msg=""
if [[ "$PHASE" == "dev" ]]; then
  sel=$("$PY" "$HERE/select_params.py" --dev "$OUT/indoor" 2>&1) && sel_rc=0 || sel_rc=$?
  echo "$sel" >> "$OUT/select_params.log"
  log "select_params rc=$sel_rc：$(echo "$sel" | tail -1)"
  [[ $sel_rc -eq 0 ]] || rc_all=3
  "$PY" "$HERE/yaw_step_stats.py" "$OUT/indoor" --hold 0.5 --out "$OUT/yaw_step_h05.json" > "$OUT/yaw_step_h05.log" 2>&1 \
    && log "hold 0.5 转角步中位数：$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['yaw_step_deg_median'], 'n =', d['pooled']['n'])" "$OUT/yaw_step_h05.json")" \
    || { log "yaw_step_stats --hold 0.5 失败（看 yaw_step_h05.log）"; rc_all=3; }
  msg="$(echo "$sel" | tail -1) yaw_step_h05=$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['yaw_step_deg_median'])" "$OUT/yaw_step_h05.json" 2>/dev/null)"
else
  log "汇总"
  "$PY" "$HERE/summarize.py" "$OUT" --old-indoor "$OLD_INDOOR" --old-outdoor "$OLD_OUTDOOR" > "$OUT/summarize.log" 2>&1 \
    || { log "汇总失败（看 summarize.log）"; rc_all=3; }
fi
log "全部结束 rc=$rc_all"
echo "rc=$rc_all $(date '+%F %T') $msg" > "$OUT/DONE"
exit $rc_all
