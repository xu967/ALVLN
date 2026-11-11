#!/usr/bin/env bash
# 任务卡 W1-8 验收 2（对拍）的采集段：1 集教师轨迹，带 --save-depth（Gazebo 的真实深度图），1 个 gz 实例。
# 采完后的 file / analytic 两种打包与逐格比较是 CPU 小活，不在这里做：见同目录 README.md 与 compare.py（手动跑）。
#
# ---------------------------------------------------------------- 预注册（2026-11-12，起跑前写下，起跑后不改；要改就换新文件名与输出目录）
#   主比较：同一集分别用 `preprocess.py --depth-source file` 与 `--depth-source analytic` 打包（其余参数相同，包络膨胀同为默认 0.102 m），
#     比较「柱子所在单元」的分带一致率（侧车里的最终分带，膨胀之后）。
#     柱子所在单元 = 稠密解析渲染（同一位姿，每格 16×16 个像素中心各一条射线）里至少有 1 个柱面像素的格；分母 = 全集这类格的总数，逐步汇总。
#   P1（任务卡验收 2）：一致率 ≥ 95%。证伪：< 95%。判不通过，不调参数、不补洞、不换采样方式，把分类与分带差直方图贴进报告。
#   不一致的格按下面的次序归类（每格只归一类；归类只看稠密渲染里该格 16×16 个像素各属于哪根柱 / 地面 / 天空与远处）：
#     地面：该格有某一列，柱面像素的正下方紧邻地面像素（柱脚的轮廓线经过这格）；
#     柱边缘：不是「地面」，且格内像素不全属于同一根柱（柱的左右缘、顶缘经过这格，或两根不同距离的柱在这格交界）；
#     其它：格内像素全是同一根柱的柱面，分带不同来自深度值本身（位姿、机身倾角、分带边界处的量化）。
#   另报（不设门）：全部格（含非柱格）的一致率；不一致格的分带差（analytic − file）直方图；不一致格在地平线以上 / 以下的分布。
#   预测（把握不足）：一致率在 90%–98%。不一致里柱边缘格占多数；地面格很少（相机高约 2.0 m 时柱脚只在 4.6 m 以外的柱上可见，那里两种来源都落第 8 带）；
#     任务卡写明的取舍（机体滚转与俯仰不计）会让近处窄带（0.3–0.8 m）的柱面格错一档，占比取决于取样时刻的机身倾角，
#     数据里没有姿态量，无法事先估（未核实）。
#   口径：位姿取 actions.jsonl 的 pose_before 与 yaw_before（观测时刻起算，与 RGB / 深度帧同一刻，pose_rgb_gap_ns 记在每行里）。
#
# 起跑条件：阻塞式拿机器锁 $MACHINE_LOCK（Gazebo 池；每 10 分钟记一行，持有到脚本退出）；拿到后记 cgroup 的 usage_usec 与 nr_throttled，结束再记一次。
# 运行：mkdir -p $OUT && cp run.sh $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
#   起跑前必须是已提交的版本：脚本会拿自己与 CODE_HEAD 里的同名文件比哈希，不一致 rc=4。冻结代码 = git archive，解到 $OUT/code，并逐文件核对 sim/ 的 git blob 哈希。
# 进度：tail -f $OUT/run.log。结束落 $OUT/DONE（rc 加时间）；采集产物在 $OUT/collection/。
# 停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。
# 续跑：同一条命令重起；collection 已采完（collection_manifest.json 里 episodes_ok == 1）就跳过，半成品挪到 collection.partial-<时刻>（不删）。
#   .lock 互斥，重复启动的那个立即退出（rc=3，不写 DONE）。
# DONE 的 rc：0 采完；4 冻结代码或脚本版本不一致；11 采集没成功；130 被信号终止。
# 时间估计（按 0417 的 5277 s / 75 集 / 4 实例折算，每集约 70 s，1 实例，仅为估计）：采集约 1–2 分钟，另加等锁。
set -u
export LC_ALL=C

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器（只读）
D=/root/rivermind-data/uav_rgb
NAME=2026-11-12-analytic-depth-parity
OUT=${OUT:-$D/exp-$NAME}
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-w1-8}   # 代码源（分支 p12-w1-8）：只用来 git archive 冻结代码
CODE_HEAD=${CODE_HEAD:-}                                 # 冻结代码的提交；缺省 = 续跑时沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
SEEDS=${SEEDS:-3100}                                     # 1 集：scene_id = seed = 3100，与 0417 collection 的第一集同一张图（train 分区）
STEPS=40                                                 # 同 0417 collection（--steps 40）
EPT=${EPT:-900}                                          # 每集墙钟上限（秒）；0417 每集约 70 s，留 10 倍余量
SIM_TIMEOUT=${SIM_TIMEOUT:-120.0}                        # gz 起不来的上限（秒）：同 heading-guard 实验
MACHINE_LOCK=${MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}   # Gazebo 池的整机锁（公共任务书 §4：exec 8>> 锁文件; flock 8；持有到脚本退出）
LOCK_LOG_S=${LOCK_LOG_S:-600}                            # 等锁期间每多少秒记一行
CPU_STAT_FILE=${CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}  # 记录用：usage_usec、nr_throttled
COLLECT=${COLLECT:-}                                     # 采集入口；缺省 = 冻结代码里的 collect_parallel.py（演练时才覆盖）
SELFCHECK=${SELFCHECK:-1}                                # 1 = 起跑前核对本脚本与 CODE_HEAD 里的同名文件一致（演练时才关）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp         # 根分区只剩 3 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg

log() { echo "$(date '+%F %T') $*"; }
cg_brief() { awk '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} END{printf "usage_usec=%s nr_throttled=%s", u, n}' "$CPU_STAT_FILE"; }

finish() {   # $1 = rc，$2 = 原因：写 DONE，退出
  trap - TERM INT
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；同一条命令重起可续跑"' TERM INT
log "开始 PGID=$$ seeds=$SEEDS steps=$STEPS 机器锁=$MACHINE_LOCK"

# ---------------------------------------------------------------- 冻结代码（先于等锁：代码问题立刻暴露）
CODE=$OUT/code
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=experiments/$NAME/run.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/sim/tools/collect_parallel.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败：$SRC $CODE_HEAD"
fi
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
echo "$CODE_HEAD" > "$OUT/code_head.txt"
grep -q 'TEACHER_VERSION = "uavrgb.teacher.v7"' "$CODE/sim/tools/teacher.py" || finish 4 "冻结代码里 TEACHER_VERSION 不是 v7（采集用的教师版本要与分支一致）"
{ echo "code_head $CODE_HEAD"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/collect_parallel.py" "$CODE/sim/tools/teacher.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  sha256sum "$PLUGIN/libCfSitlSystem.so"; } > "$OUT/inputs.txt" 2>&1

# ---------------------------------------------------------------- 机器锁（阻塞等待，持有到脚本退出）
mkdir -p "$(dirname "$MACHINE_LOCK")"
exec 8>> "$MACHINE_LOCK"
t_lock=$SECONDS
if ! flock -n 8; then
  log "机器锁 $MACHINE_LOCK 被占用，阻塞等待（每 $LOCK_LOG_S s 记一行）"
  until flock -w "$LOCK_LOG_S" 8; do log "仍在等机器锁（已等 $((SECONDS - t_lock)) s）"; done
fi
log "拿到机器锁，等了 $((SECONDS - t_lock)) s；持有到脚本退出（$(cg_brief)）"

# ---------------------------------------------------------------- 采集
collection_ok() {   # collection_manifest.json 存在、episodes_ok == 1、episodes_failed == 0
  "$PY" -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if (m.get("episodes_ok")==1 and not m.get("episodes_failed")) else 1)' \
    "$OUT/collection/collection_manifest.json" 2>/dev/null
}
if collection_ok; then
  log "collection 已采完，跳过"
else
  if [[ -d "$OUT/collection" ]]; then   # 半成品挪开留证据，不删（collect_parallel 要求 --out 不存在或为空）
    ts=$(date +%H%M%S); mv "$OUT/collection" "$OUT/collection.partial-$ts"
    [[ -f "$OUT/collect.log" ]] && mv "$OUT/collect.log" "$OUT/collect.partial-$ts.log"
  fi
  entry=${COLLECT:-$CODE/sim/tools/collect_parallel.py}
  log "采集开始（$(cg_brief)）入口 $entry seeds $SEEDS"
  # 与 0417 collection 同一套参数（e1_mini + cf_sitl、hold 1.0、observation 窗口、无辅助相机、默认出生点、A* 教师），
  # 差别：1 实例 1 集、固定 seed、--save-depth。
  "$PY" "$entry" --instances 1 --episodes-per-instance 1 --steps "$STEPS" --seeds "$SEEDS" --task base \
      --out "$OUT/collection" --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation \
      --no-aux-cameras --save-depth --episode-timeout "$EPT" --sim-timeout "$SIM_TIMEOUT" > "$OUT/collect.log" 2>&1
  log "采集 rc=$?（$(cg_brief)）"
fi
collection_ok || finish 11 "采集没成功：读 $OUT/collect.log 与 $OUT/collection/collection_manifest.json"
n_depth=$(find "$OUT/collection" -name 'depth_*.png' | wc -l)
log "深度帧 $n_depth 个"
finish 0 "采完 1 集，深度帧 $n_depth 个；下一步：README.md 里的打包与比较"
