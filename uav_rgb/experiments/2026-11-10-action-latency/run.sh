#!/usr/bin/env bash
# 动作派发时刻固定到仿真时间（任务卡 W1-0，collect.py --action-latency）的标定与验收：两个阶段，同一个守护进程。零训练，用既有权重。
# 背景：--window-from observation 下动作窗口从观测戳起算，动作实际生效时长 = hold −（观测戳到派发的延迟）。延迟 = 帧龄（相机 10 Hz，约 0.114 s）
#   + 策略段的墙钟耗时折成的仿真时间；柱阵 K=1 每步生效 0.871 s，长课程 K=3 的第一步因 A* 缓存冷只生效约一半（11-10 晚实测，计划 phase12 §2.4）。
#   --action-latency L 之后每步在「观测戳 + L」派发，生效时长恒为 hold − L；就绪时仿真戳已超过 观测戳 + L + 0.02 s 的步记「迟到」，含迟到步的集算基础设施失败。
#   每集起飞前预热 A* 栅格，学生 + β=0 的影子教师标签挪到派发之后算。冻结代码 = 提交 CODE_HEAD（分支 p12-w1-0：feat(collect) 69a5515 + feat(scene) b85aa6b）。
#
# 阶段 1（不依赖闸）：
#   · 标定（验收 1）：教师，柱阵 K=1 与长课程 K=3，各 10 个 train 分区 seed（2,3,4,6,7,8,9,10,11,12：K=1 与 K=3 的 train 分区前 10 个 seed 相同，离线核过），L=0.16。
#   · 验收 2：学生 calswa 种子 1 + 护栏 2，柱阵 K=1，20 个冻结 seed，L=0.16（同时是新口径下的 K=1 参照）。
#   · 验收 3：森林 forest_e1 三档（sparse / medium / dense），教师，不带 --action-latency，每档 3 集，量延迟分布并算需要的 --rtf。
# 标定门（summarize.py --gate L）：K=3 对 K=1 的生效时长、稳态前飞位移、转角三项相对差都 ≤ 2%，且两个臂迟到步都为 0（含每集第 0 步）。
#   L=0.16 未过 → 试 L=0.20（只补跑教师标定，两个臂各 10 集）；两个 L 都没过，或数据不全 → 脚本停，原因写进 DONE，不进入阶段 2。
# 阶段 2（只在标定门过了才进；用第一个过门的 L）：验收 4，学生 calswa 三个训练种子 + 护栏 2，K=1 与 K=3（第 1 块）各 3 × 20 个冻结 seed。
#   L=0.16 过门时 K=1 的种子 1 复用验收 2 那一组（同配置同 L，已评完则跳过）。最后用哪个 L 由主会话定，本脚本只报告每个 L 的数。
# 运行：mkdir -p $OUT && cp run.sh summarize.py $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
# 进度：tail -f $OUT/run.log；汇总随时可跑：python $OUT/summarize.py（缺的臂标「未完成」）；只看标定门：python $OUT/summarize.py --gate 0.16
# 结束落 DONE（rc、时间、原因）与 summary.md。停：kill -- -$(cat $OUT/run.pid)（只按进程组，别用 pkill -f）。
# 续跑：同一条命令重起；评完的臂跳过，半成品挪到 <臂>.partial-<时刻>。.lock 互斥，重复启动的那个立即退出。
# gz 配额：本任务最多 2 个实例（LANES=2）。起跑前等 rule-oracle v1 的守护进程收掉（$V1/DONE 出现），所以先于它起跑也不会超配额。
#
# 预注册（2026-11-10，起跑前写下；下面 1–4 条抄自任务卡 W1-0「标定与验收」，括号里是我补的执行口径）：
#   1. L 取 0.16 s：帧龄 0.114 s 加教师在 K=3 上的策略段（缓存后沿航线均值 16–26 ms 墙钟，折仿真约 13–21 ms）再留余量。
#      柱阵现状的延迟中位数是 0.129–0.137 s，所以新协议下每步生效 0.84 s，比现在短约 3.5%，旧的 K=1 参照要在新协议下重跑一遍（见第 2 条）。
#      教师策略，柱阵 K=1 与长课程 K=3 各 10 个 train 分区 seed：每步动作生效时长、稳态前进位移、转向角度三项，K=3 与 K=1 的相对差都不超过 2%；
#      迟到步占比为 0，含每集第一步。做不到时先查预热，再试 L = 0.20，再考虑 --rtf；试过的每个取值的迟到步占比都写进报告，最后用哪个由主会话定。
#      （生效时长 = 各集「每步生效」均值的集间中位数，与 11-10 长课程汇总同口径；稳态前进位移 = 本步与上一步都执行 FORWARD_SLOW 且本步无碰撞的步的位移池化均值；
#       转向角度 = YAW_LEFT / YAW_RIGHT 步 |dyaw_deg| 池化均值；相对差 = (K=3 − K=1) / K=1；迟到步占比 = 迟到步数 / 全部步数，每集第 0 步另列。）
#   2. 学生 calswa 种子 1 加 --heading-guard 2，柱阵 20 个冻结 seed，带 --action-latency 0.16：走廊内成功数与 11-10 的两次同配置评测
#      （下午 17/20，晚上对照 17/20，两者逐集一致 16/20）配对比较。预测：差别在同权重同 seed 重评的波动之内（逐集一致率不低于 70%）。这一组同时是新协议下的 K=1 参照。
#   3. 森林 forest_e1 三档各量一次延迟分布，给出各自需要的 --rtf。
#   4. 长课程第 1 块的对照：学生 calswa 三个训练种子加 --heading-guard 2，带 --action-latency，柱阵 K=1 与长课程 K=3 各 20 个冻结 seed × 3 个种子。
#      比较量是 K=3 的第 1 块通过率对 K=1 的成功率（同权重同 seed 配对，精确 McNemar）。11-10 晚的旧协议下是 39/60 对 50/60，但 K=3 的第一步因为冷缓存只生效了约一半时长，
#      分不清是时序还是画面。预注册：新协议下两者相差不超过 0.08，则那次下降来自第一步的时序；K=3 仍低 0.12 以上，则来自后面几块的柱子进了画面（学生没见过远处的柱子）。
#      （差在 0.08 与 0.12 之间：两条线都没覆盖，如实记录，不归因。含迟到集的臂：迟到集不计成功，按没成功配对。）
# 起跑前已知（写进来，免得事后当发现）：
#   · 冒烟（train 分区 seed 2，1 个 gz，eval 全链路）：教师 K=1 34 步，迟到 0，dispatch_lag_s 全部 0.160；学生 calswa s1 + 护栏 2 的 K=1 两集（seed 2、3）各 32 步，迟到 0，
#     lag 中位 0.160、最大 0.165 / 0.160；教师 K=3 79 步，迟到 3 步（步号 0、3、8，lag 0.195 / 0.190 / 0.185 s，阈值 0.18 s），lag 中位 0.160，该集按预期进 failed[] 与 late_dispatch_episodes。
#     也就是说 L=0.16 在 K=3 上的第 0 步仍迟到：预热后第 0 步的 A* 约 51 ms 墙钟（冷缓存 284 ms，离线实测），仍是沿航线最重的一步。
#     预期标定门在 L=0.16 不过，脚本自动试 L=0.20（阈值 0.22 s，冒烟里 K=3 最大 0.195 s）。这是冒烟的 1 集，不当 K=3 的发生率。
#   · 学生冒烟的第 0 步推理耗时 0.69 ms，其余中位 0.72 ms：StudentPolicy 构造时已有 _warmup，没有首帧冷启动，所以没有新加空推理。
#   · 阶段 2 的 K=3 每集约 80–100 步，对应 11-10 的电池掉压起点（复位后约 115 s）有余量；采集中止仍算没通过（lc.block_outcomes 的 aborted）。
set -u

# ---------------------------------------------------------------- 配置（改这里）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=${OUT:-$D/exp-action-latency-20261110}
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-w1-0}   # 代码源（分支 p12-w1-0）：只用来 git archive 冻结代码
CODE_HEAD=${CODE_HEAD:-b85aa6bec16749bc839e70818e943b9156a711ef}                          # 冻结代码的提交：feat(scene): --rtf（含 feat(collect) 69a5515）
V1=${V1:-$D/exp-rule-oracle-v1-20261110}                 # rule-oracle v1 守护进程的目录：其 DONE 出现后才起跑（配额）
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与 11-10 的 K=1 / K=3 评测同一个推理二进制
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
SWA=$D/exp-swa-calib-20261110/ckpt                       # 权重平均 + 校准权重 cal_swa_s{0,1,2}_k10.bin
GPU=GPU-dbc6435f-4db3-f704-bd1d-3b40bff73a70             # uav_rgb 只许用这张卡（按 UUID 钉）
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
CAL_SEEDS=2,3,4,6,7,8,9,10,11,12                         # 标定的 train 分区 seed：K=1 与 K=3 前 10 个 train seed（离线跑 scene_gen 核过，两边相同）
FOREST_SEEDS=4100:4140                                   # 森林：从中按 train 分区取前 FOREST_EPISODES 个
FOREST_EPISODES=3
L_MAIN=0.16                                              # 预注册：先 0.16
L_ALT=0.20                                               # 不过再试 0.20
EPISODES=20                                              # 验收臂的请求集数 = 冻结 seed 个数
CAL_EPISODES=10                                          # 标定臂的集数
GUARD=2                                                  # 与 R1 实验同一个护栏阈值
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
K1_ARGS="--task base --steps 80"                         # 柱阵 K=1
K3_ARGS="--task course_base --blocks 3 --steps 240"      # 长课程 K=3（80·K 步预算，到成功线即停）
K1_EPT=1500                                              # 每集墙钟上限（秒）：同 heading-guard 实验
K3_EPT=2520                                              # 600 + 8·240：同 long-course 实验
LANES=${LANES:-2}                                        # 同时 2 个 gz：本任务的配额（总共 7 个，另一个 agent 占 5 个）
EVAL=${EVAL:-}                                           # 评测入口；缺省 = 冻结代码里的 eval_closed_loop.py（只有演练流程时才覆盖）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 2.3 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT CODE=$OUT/code PY=$R/.venv-tools/bin/python
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }
exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
echo "$(date '+%F %T') 开始 PGID=$$ lanes=$LANES L=$L_MAIN→$L_ALT guard=$GUARD code=$CODE_HEAD"

finish() {   # $1 = rc，$2 = 原因
  "$PY" "$OUT/summarize.py" "$OUT" > "$OUT/summarize.log" 2>&1
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  echo "$(date '+%F %T') 全部结束 rc=$1 $2"
  exit "$1"
}

# 配额闸：rule-oracle v1 收掉（DONE 出现）、且我的冒烟 gz 退出之后才起跑；只读检查，不杀任何进程
waited=0
while [[ ! -f "$V1/DONE" ]] || ps -eo cmd | grep -F "phase12-cards/w1-0/smoke" | grep -v grep | grep -q .; do
  (( waited % 300 == 0 )) && echo "$(date '+%F %T') 等 $V1/DONE 与冒烟结束（配额）"
  sleep 20; waited=$((waited + 20))
done
echo "$(date '+%F %T') 配额闸通过：$(cat "$V1/DONE" 2>/dev/null | head -c 80)"

# 冻结代码：已有副本就复用（续跑必须同一份），没有就从 CODE_HEAD 解出；然后逐文件核对 sim/ 下的 git blob 哈希
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || { echo "冻结代码失败"; finish 4 "冻结代码失败"; }
fi
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || { echo "$(date '+%F %T') 冻结副本与 $CODE_HEAD 不一致：$bad"; finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"; }
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}
export EVAL
{ echo "code_head $(git -C "$SRC" rev-parse "$CODE_HEAD")"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/teacher.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  sha256sum "$SWA"/cal_swa_s{0,1,2}_k10.bin "$INFER" "$PLUGIN/libCfSitlSystem.so"; } > "$OUT/inputs.txt" 2>&1

# 一个评测臂：名字|评测参数串|请求集数|单集墙钟上限。评完 = eval.json 不是 partial 且 出结果 + 采集失败（含迟到集）= 请求数；迟到集是测量结果，不重跑。
run_job() {
  local name=$1 args=$2 episodes=$3 ept=$4
  [[ -n "$name" && -n "$args" ]] || return 0
  if "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$name/eval.json" "$episodes" 2>/dev/null; then
    echo "$(date '+%F %T') $name 已评完，跳过"; return 0
  fi
  [[ -d "$OUT/$name" ]] && mv "$OUT/$name" "$OUT/$name.partial-$(date +%H%M%S)"   # 半成品挪开留证据，不删
  echo "$(date '+%F %T') $name 开始"
  # shellcheck disable=SC2086
  "$PY" "$EVAL" $args --episodes "$episodes" --sim-timeout 120.0 --episode-timeout "$ept" \
      --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --out "$OUT/$name" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$(date '+%F %T') $name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)"
  return 0                                   # 单臂失败（含迟到集使 eval 退出码非 0）不停整个队列，汇总里按 failed 标出来
}
export -f run_job
run_queue() {   # 标准输入：每行一个臂 名字|评测参数串|请求集数|单集墙钟上限；顺序即优先级
  xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n a e t <<< "{}"; run_job "$n" "$a" "$e" "$t"'
}

cal_job() {   # $1 = L，$2 = 1|3
  local ta=$K1_ARGS ept=$K1_EPT; [[ $2 == 3 ]] && { ta=$K3_ARGS; ept=$K3_EPT; }
  echo "cal-L$(printf %.2f "$1")-k$2|--policy teacher $ta --split train --seeds $CAL_SEEDS --corridor-half-width $CORRIDOR --action-latency $1|$CAL_EPISODES|$ept"
}
acc_job() {   # $1 = L，$2 = 1|3，$3 = 训练种子
  local ta=$K1_ARGS ept=$K1_EPT; [[ $2 == 3 ]] && { ta=$K3_ARGS; ept=$K3_EPT; }
  echo "acc-L$(printf %.2f "$1")-k$2-s$3|--policy student --ckpt $SWA/cal_swa_s$3_k10.bin --infer-bin $INFER --heading-guard $GUARD $ta --split test --seeds $SEEDS --corridor-half-width $CORRIDOR --action-latency $1|$EPISODES|$ept"
}
forest_job() {   # $1 = 密度
  echo "forest-$1|--policy teacher --task forest_e1 --density $1 --steps 80 --split train --seeds $FOREST_SEEDS|$FOREST_EPISODES|$K1_EPT"
}
gate() {   # $1 = L；stdout 一行判定，退出码 0 PASS / 1 FAIL / 2 INCOMPLETE
  "$PY" "$OUT/summarize.py" --gate "$1" "$OUT"
}

# ---------------------------------------------------------------- 阶段 1
echo "$(date '+%F %T') 阶段 1：标定 L=$L_MAIN（教师 K=3、K=1 各 $CAL_EPISODES 集）+ 验收 2（学生 K=1 种子 1）+ 验收 3（森林三档）"
{ cal_job "$L_MAIN" 3; cal_job "$L_MAIN" 1; acc_job "$L_MAIN" 1 1; forest_job sparse; forest_job medium; forest_job dense; } | run_queue

verdict=$(gate "$L_MAIN"); grc=$?
echo "$(date '+%F %T') 标定门 L=$L_MAIN rc=$grc $verdict"
L_STAR=""
if [[ $grc -eq 0 ]]; then
  L_STAR=$L_MAIN
elif [[ $grc -eq 1 ]]; then
  echo "$(date '+%F %T') 标定门 L=$L_MAIN 未过，试 L=$L_ALT（只补跑教师标定）"
  { cal_job "$L_ALT" 3; cal_job "$L_ALT" 1; } | run_queue
  verdict2=$(gate "$L_ALT"); grc2=$?
  echo "$(date '+%F %T') 标定门 L=$L_ALT rc=$grc2 $verdict2"
  if [[ $grc2 -eq 0 ]]; then
    L_STAR=$L_ALT
  else
    finish 10 "标定门未过，没有进入验收 4。${verdict#* }；${verdict2#* }。下一步（预注册）：先查预热，再考虑 --rtf，最后用哪个 L 由主会话定"
  fi
else
  finish 11 "标定数据不全，没有进入验收 4：${verdict#* }"
fi
echo "$L_STAR" > "$OUT/L_STAR"

# ---------------------------------------------------------------- 阶段 2：验收 4（只在标定门过了才进）
echo "$(date '+%F %T') 阶段 2：验收 4，L=$L_STAR，学生 calswa 三个训练种子 + 护栏 $GUARD，K=1 与 K=3 各 $EPISODES 个冻结 seed"
{ for s in 0 1 2; do acc_job "$L_STAR" 3 "$s"; done; for s in 0 1 2; do acc_job "$L_STAR" 1 "$s"; done; } | run_queue

finish 0 "标定门通过的 L=$L_STAR，验收 1–4 全部跑完；最后用哪个 L 由主会话定。读 summary.md"
