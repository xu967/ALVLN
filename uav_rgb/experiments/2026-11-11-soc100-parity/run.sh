#!/usr/bin/env bash
# 满电档（SOC=100%）对 50% 档的验收实验（任务卡 W1-10「11-11 更新」）：同一批权重、同一批 seed、同一协议，只换起始电量。
# 先核对 Gazebo 插件确实读了满电副本（探针），再跑六个学生臂，结束时自动出配对汇总与预注册判决。同一个守护进程，零训练，用既有权重。
#
# 背景（任务卡 W1-10；来源 docs/reviews/2026-11-10-phase12/complexity.md §5）：SITL 每集从 50% 电量起飞，约 100 s 后推力余量耗尽，一集最多约 100 步；
#   长任务要从满电起飞。起始电量只能经 params JSON 的副本覆盖（sim.battery_soc_init 0.5→1.0），dynamics_profile 从 1bc8fedf 变成 06e5f017。
#   起跑前查清的事实（提交 b436c90，测试 sim/tools/test_scene_cf_sitl_params.py）：gz 插件只读世界 SDF 的 <params>，
#   此前 collect.py --cf-sitl-params 只改 Python 侧的指纹，物理仍按默认参数（电量 50%），两边静默不一致。现在 scene_gen.py --cf-sitl-params
#   把 <params> 写成副本绝对路径，collect_parallel 与 eval_closed_loop 透传；不给该参数时世界文件逐字节不变（测试锁住）。
#   满电副本 = $D/params/e1_mini_soc100.json（tools/make_params_variant.py 产物，不进仓库）：与 sim/dynamics/e1_mini.json 的 diff 只有一行（battery_soc_init），
#   整文件 sha256 cd822f80…99819。对照 = 第二轮（exp-action-latency-r2-20261111）里 50% 电量的六个学生臂 acc-L0.20-k{1,3}-s{0,1,2}。
#
# 起跑前已知的事实（写在这里免得事后当发现；来源与可信度如注）：
#   · CPU 试探（complexity.md §5，未核实可复现）：50% 电量解锁到 READY 约 7.3–8.1 s（SITL 时钟），V_bus 约 3.43 V，悬停油门约 0.89；
#     100% 约 9.6–10.3 s，约 3.81 V，约 0.79；50% 约 106 s 开始掉高，约 123 s 落地；100% 约 260–268 s 落地。
#   · 第二轮对照臂的首步观测时刻（actions.jsonl 第 0 行 timestamp_ns，仿真秒）：K=1 三个训练种子中位约 13.5 s（13.0–15.3），K=3 s1 中位约 14.3 s；
#     第二轮 K=1 三个臂（11-11 23:52 已评完，60 集）：巡航位移（cruise，步 3–30）池化均值 0.2097 m（n=905），转角池化均值 25.19°（n=222）。
#   · 对照臂的 manifest 全部是 cf_sitl_v1:1bc8fedf:lib31（108 集，params_sha256 683e7400…）。
#
# ---------------------------------------------------------------- 预注册（卡 W1-10「11-11 更新」，逐字）
#   K=1 的每步前飞位移与转角（口径同 `summarize_r2.py`）与 50% 档相差不超过 5%，成功数配对无可检出的差别（精确 McNemar p > 0.05）；
#   K=3 满电档里飞到电量掉高的集数为 0，第 1 块通过率与 50% 档相差不超过 0.08，三块全过的集数不少于 50% 档。证伪：K=1 的位移或转角相差超过 10%。
#   差超过 10% 则两档的数据与结论不能互用，报告里写明并停下。
#   第二轮没有 `DONE` 或 `DONE` 的 rc 不是 0 时先停下报告，不要自己起对照臂。
# ---------------------------------------------------------------- 逐项落成（summarize_parity.py 的 judge 函数实现，有单测；起跑后不改）
#   P1 K=1 每步前飞位移：满电对 50% 的相对差 |f/c − 1| ≤ 5%。口径 = 第二轮 config.env 的 DISP_CLAUSE（cruise：连续前进第 4 步起、步 3–30），
#      三个训练种子的 60 集池化均值，函数直接用 summarize_r2.py 的 fwd_samples。位移中位数另报，不进判决。
#   P2 K=1 每步转角：YAW 步 |dyaw_deg| 的池化均值，相对差 ≤ 5%。转角中位数另报。
#   P3 K=1 走廊内成功数（success，横向边界 3 m）：同 seed 同训练种子逐集配对，三个种子合并 60 对，精确 McNemar p > 0.05。
#   P4 K=3 满电档里 z_tail_min < 1.5 m（末 10 步最低高度，电池压降的症状，长课程汇总脚本的口径）的集数 = 0。
#   P5 K=3 第 1 块通过率（通过数 / 60）：|满电 − 50%| ≤ 0.08。
#   P6 K=3 三块全过的集数：满电 ≥ 50%。
#   证伪 F：P1 或 P2 的相对差 > 10%。F 触发时判决 FALSIFIED：两档数据与结论不能互用，报告写明并停下（所有臂已在同一个守护进程里跑完，没有后续阶段可停）。
#   判决 PASS = P1..P6 全过；FAIL = 有项没过且没触发 F。指纹（各集 manifest.dynamics_profile，对照 1bc8fedf、满电 06e5f017）或配对协议键（权重与推理二进制 sha256、
#   护栏、hold、window_from、L、走廊、模板、seed 等）不一致时判决 INVALID，不给 P1..P6。判决写入 DONE 文本、VERDICT、summary.md；DONE 的 rc 不随判决变。
#   没有设提前停止：判决量是 60 集的池化量，单个臂 20 集太噪，不据它中途停。
#   另报不进判决：位移与转角中位数、首步观测仿真秒（起飞到就绪的代理）、K=3 的 S_1/S_2/S_3 与 sim_s、探针的 READY 用时与悬停油门。
#   条款 S1（探针，起跑后第一步，满电臂之前）：probe_plugin_params.py 起 default / copy 两种世界各 3 个 gz sim（-v 3），读静止电压与 READY 遥测与插件日志的 params= 行；
#     判据 G1 每个都到 READY、G2 copy 的静止电压中位数比 default 高 ≥ 0.25 V、G3 日志里的 params= 与世界一致、G4 库版本一致；不过 rc=21，不起任何满电臂。
#   条款 S2（对照核对，拿到机器锁之后）：$R2/DONE 存在且 rc=0、$R2/L_STAR 与 config.env 的 DISP_CLAUSE 存在、对照六个臂评完；
#     权重、推理二进制、插件的 sha256 与 $R2/inputs.txt 一致；sim/tools、sim/dynamics、sim/worlds、sim/cf_sitl 下与第二轮冻结代码只许有允许清单里的差异
#     （scene_gen / collect_parallel / eval_closed_loop 三个文件、测试文件、md）。不满足：第二轮的问题 rc=20，输入或代码不一致 rc=23，都不起任何臂。
#   条款 S3（含迟到步的集）：与第二轮条款 R6 相同，学生臂里含迟到步的集自动重跑一次，新输出子目录 <臂>-retry1，两份记录都留，汇总以重跑结果为准。
#   条款 S4（资源与起跑条件）：与第二轮条款 R7 相同。LANES=2 全程；起跑条件 ① nvidia-smi -L 里有配置的 GPU；② cgroup 配额 ≥ 12 核；③ 拿到机器锁 $MACHINE_LOCK
#     （阻塞等待，每 10 分钟记一行，持有到脚本退出）；④ 拿到锁后没有 gz sim 进程，且 30 s 窗口内容器 CPU < 1.5 核。①② 不满足立刻退出并写 DONE；④ 每分钟复查，最多 3 小时。
#
# 运行：mkdir -p $OUT && cp run.sh $OUT/ && cd $OUT && setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &
#   起跑前必须是已提交的版本：脚本拿自己与 CODE_HEAD（缺省 SRC 的 HEAD）里的同名文件比哈希，不一致 rc=4。冻结代码 = git archive CODE_HEAD，解到 $OUT/code，
#   并逐文件核对 sim/ 的 git blob 哈希；汇总脚本也从冻结副本里跑。续跑：同一条命令重起，评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删），探针通过后不重跑。
# 进度：tail -f $OUT/run.log；汇总随时可跑：<解释器> $OUT/code/experiments/2026-11-11-soc100-parity/summarize_parity.py $OUT --control $R2（缺的臂标「未完成」）。
#   停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。结束落 DONE（rc、时间、原因、判决）、VERDICT 与 summary.md。
# DONE 的 rc：0 六个满电臂跑完，汇总与判决已出（判决 PASS / FAIL / FALSIFIED 都是 0，看 VERDICT）；3 已有实例（不写 DONE）；4 冻结代码或脚本版本不一致；
#   11 数据不全；12 配置的 GPU 不在；13 cgroup 配额不足；14 等空闲超过 3 小时；20 第二轮没有 DONE / DONE 的 rc 不是 0 / 对照臂不全；
#   21 探针没过或基础设施失败；22 满电副本缺失或与预期指纹 / sha256 不符；23 对照与本实验的输入或协议代码不一致；24 汇总判 INVALID；130 被信号终止。
# 时间估计（按第二轮的用时折算，仅为估计）：探针约 5 分钟；K=1 学生 20 集约 18 分钟；K=3 学生 20 集第二轮约 25–30 分钟，满电不掉高会飞更多步，可能更长；
#   2 条车道约 1.5–2.5 小时，另加等锁与等空闲的时间。
set -u
export LC_ALL=C

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
R2=${R2:-$D/exp-action-latency-r2-20261111}              # 第二轮输出：对照臂（50% 电量）、L_STAR、config.env、inputs.txt 的来源，只读
OUT=${OUT:-$D/exp-soc100-parity-20261111}                # 本实验输出
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-w1-10}   # 代码源（分支 p12-w1-10）：只用来 git archive 冻结代码
CODE_HEAD=${CODE_HEAD:-}                                 # 冻结代码的提交；缺省 = 续跑时沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与对照臂同一个推理二进制（与 $R2/inputs.txt 的 sha256 核对）
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # 与对照臂同一个 SITL 插件（公共任务书：不要动这个目录；与 $R2/inputs.txt 核对）
SWA=$D/exp-swa-calib-20261110/ckpt                       # 与对照臂同一批权重 cal_swa_s{0,1,2}_k10.bin（与 $R2/inputs.txt 核对）
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740             # 同第二轮：11-11 容器重启后 uav_rgb 用的这张卡（RTX 2080 Ti），用户 11-11 19:23 指定
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20，同第二轮
PARAMS=${PARAMS:-$D/params/e1_mini_soc100.json}          # 满电副本（make_params_variant.py --set sim.battery_soc_init=1.0 的产物，数据盘，不进仓库）
PARAMS_PROFILE=cf_sitl_v1:06e5f017:lib31                 # 副本应有的 dynamics_profile：只改 battery_soc_init 0.5→1.0，任务卡给的 06e5f017；不符 rc=22
PARAMS_SHA256=cd822f80a6c09e324d1324aa8a29bdb40d10e6286c0b6cc2efeb4bddb2e99819   # 副本文件字节 sha256（make_params_variant 的输出）；不符 rc=22
PROBE_EPISODES=0,1,2                                     # 条款 S1：探针里每种世界起几个 gz sim（reset episode 号）
EPISODES=20                                              # 验收臂的请求集数 = 冻结 seed 个数
GUARD=2                                                  # 同第二轮：护栏阈值
CORRIDOR=3.0                                             # 同第二轮：横向边界 = scene_gen.Y_RANGE
K1_ARGS="--task base --steps 80"                         # 柱阵 K=1，同第二轮
K3_ARGS="--task course_base --blocks 3 --steps 240"      # 长课程 K=3（80·K 步预算，到成功线即停），同第二轮
K1_EPT=1500                                              # 每集墙钟上限（秒），同第二轮
K3_EPT=2520                                              # 600 + 8·240，同第二轮
LANES=${LANES:-2}                                        # 条款 S4：全程 2 条车道，与第二轮相同
EVAL=${EVAL:-}                                           # 评测入口；缺省 = 冻结代码里的 eval_closed_loop.py（只有演练流程时才覆盖）
PROBE_PY=${PROBE_PY:-}                                   # 探针入口；缺省 = 冻结代码里的 probe_plugin_params.py（演练时换成桩）
SELFCHECK=${SELFCHECK:-1}                                # 1 = 起跑前核对本脚本与 CODE_HEAD 里的同名文件一致（演练时才关）
CHECK_R2_INPUTS=${CHECK_R2_INPUTS:-1}                    # 1 = 条款 S2 的输入与代码核对（演练时关，演练的第二轮目录是假的）
# 允许与第二轮冻结代码不同的文件（条款 S2）：本卡改的三个文件、测试文件、文档
ALLOW_DIFF='(sim/tools/(collect_parallel|eval_closed_loop|scene_gen)\.py|test_[A-Za-z0-9_]+\.py|README\.md)'
# 起跑条件（条款 S4）
MIN_CORES=${MIN_CORES:-12}                               # ② cgroup 配额下限（核）
IDLE_MAX_CORES=${IDLE_MAX_CORES:-1.5}                    # ④ 窗口内容器平均核数上限
IDLE_WINDOW_S=${IDLE_WINDOW_S:-30}                       # ④ 测量窗口（秒）
IDLE_RECHECK_S=${IDLE_RECHECK_S:-60}                     # ④ 不满足时的复查间隔（秒）
IDLE_LOG_S=${IDLE_LOG_S:-600}                            # ④ 等待期间每多少秒记一行
IDLE_LIMIT_S=${IDLE_LIMIT_S:-10800}                      # ④ 最长等 3 小时
LOCK_LOG_S=${LOCK_LOG_S:-600}                            # ③ 等锁期间每多少秒记一行
MACHINE_LOCK=${MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}   # 重实验的整机锁（主会话定的惯用法：exec 8>> 锁文件; flock 8；持有到脚本退出）
NVIDIA_SMI=${NVIDIA_SMI:-nvidia-smi}                     # ① 探针（演练时换成桩）
CPU_MAX_FILE=${CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}     # ② 探针：配额 微秒 周期
CPU_STAT_FILE=${CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}  # ④ 与每臂记录用：usage_usec、nr_throttled、throttled_usec
GZ_PATTERN=${GZ_PATTERN:-'(^|[ /])gz sim( |$)'}          # ④ 探针：pgrep -f 的模式（匹配 gz sim 进程的命令行）
GZ_PROBE=${GZ_PROBE:-}                                   # 非空时用这条命令的第一行输出当 gz sim 进程数（演练用）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT R2 CODE=$OUT/code PY=$R/.venv-tools/bin/python CPU_STAT_FILE
SUMM_R2=$CODE/experiments/2026-11-10-action-latency/summarize_r2.py       # 口径函数与 --late-seeds
SUMM=$CODE/experiments/2026-11-11-soc100-parity/summarize_parity.py       # 配对汇总与预注册判决
export SUMM_R2 SUMM
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg

log() { echo "$(date '+%F %T') $*"; }
export -f log

finish() {   # $1 = rc，$2 = 原因：刷新 summary.md（冻结代码已就位时），写 DONE，退出
  trap - TERM INT
  [[ -f "$SUMM" && -f "$OUT/config.env" ]] && "$PY" "$SUMM" "$OUT" --control "$R2" --note "$(date '+%F %T') 生成，DONE rc=$1" > "$OUT/summarize.log" 2>&1
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }

exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；评完的臂保留，同一条命令重起可续跑"' TERM INT
log "开始 PGID=$$ lanes=$LANES gpu=$GPU 对照=$R2 副本=$PARAMS（$PARAMS_PROFILE）机器锁=$MACHINE_LOCK"

# ---------------------------------------------------------------- 起跑条件 ①②（不满足立刻退出）
gpu_list=$("$NVIDIA_SMI" -L 2>&1)
if ! grep -qF "$GPU" <<< "$gpu_list"; then
  finish 12 "条件 ① 不满足：配置的 GPU $GPU 不在 nvidia-smi -L 的列表里（$(head -c 300 <<< "$gpu_list" | tr '\n' ';')），没有起跑"
fi
log "条件 ① 满足：$(grep -F "$GPU" <<< "$gpu_list" | head -n1)"
read -r quota period < "$CPU_MAX_FILE"
if [[ "$quota" == max ]]; then
  cores=max; quota_ok=1
else
  cores=$(awk -v q="$quota" -v p="${period:-100000}" 'BEGIN{printf "%.2f", q/p}')
  quota_ok=$(awk -v c="$cores" -v m="$MIN_CORES" 'BEGIN{print (c>=m) ? 1 : 0}')
fi
if [[ "$quota_ok" != 1 ]]; then
  finish 13 "条件 ② 不满足：cgroup 配额 $cores 核 < $MIN_CORES 核（$CPU_MAX_FILE：$quota ${period:-?}），没有起跑"
fi
log "条件 ② 满足：cgroup 配额 $cores 核（要求 ≥ $MIN_CORES）"

# ---------------------------------------------------------------- 冻结代码（先于等锁：代码问题立刻暴露，冻结的是起跑这一刻的提交）
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=experiments/2026-11-11-soc100-parity/run.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败"
fi
echo "$CODE_HEAD" > "$OUT/code_head.txt"
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim | while read -r _m _t sha path; do [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
[[ -f "$SUMM" && -f "$SUMM_R2" ]] || finish 4 "冻结副本里没有 summarize_parity.py 或 summarize_r2.py（CODE_HEAD $CODE_HEAD 之前的提交？）"
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}
PROBE_PY=${PROBE_PY:-$CODE/experiments/2026-11-11-soc100-parity/probe_plugin_params.py}
export EVAL PROBE_PY

# 满电副本核对（不起 gz，立刻做）：缺文件、指纹或 sha256 不符就不往下走
"$PY" "$PROBE_PY" --params "$PARAMS" --check-only --expect-profile "$PARAMS_PROFILE" --expect-sha256 "$PARAMS_SHA256" > "$OUT/params_check.log" 2>&1
rc=$?
[[ $rc == 0 ]] || finish 22 "满电副本核对没过（rc=$rc）：$(tr '\n' ';' < "$OUT/params_check.log" | head -c 400)"
log "满电副本核对通过：$(grep -E '指纹 ' "$OUT/params_check.log" | head -n1 | sed 's/^ *//')"

{ echo "code_head $CODE_HEAD"
  echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/teacher.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  sha256sum "$SWA"/cal_swa_s{0,1,2}_k10.bin "$INFER" "$PLUGIN/libCfSitlSystem.so" "$PARAMS"; } > "$OUT/inputs.txt" 2>&1
DISP_CLAUSE=$(sed -n 's/^DISP_CLAUSE=//p' "$R2/config.env" 2>/dev/null | head -n1)
[[ -f "$OUT/config.env" ]] || cat > "$OUT/config.env" <<EOF
# run.sh 起跑配置（summarize_parity.py 读 CONTROL、DISP_CLAUSE、PARAMS_PROFILE；续跑沿用这里的值）
CONTROL=$R2
DISP_CLAUSE=${DISP_CLAUSE:-cruise}
PARAMS=$PARAMS
PARAMS_PROFILE=$PARAMS_PROFILE
PARAMS_SHA256=$PARAMS_SHA256
CODE_HEAD=$CODE_HEAD
GPU=$GPU
LANES=$LANES
EOF

# ---------------------------------------------------------------- 起跑条件 ③ 机器锁，④ 空闲（等到满足才开始）
mkdir -p "$(dirname "$MACHINE_LOCK")"
exec 8>> "$MACHINE_LOCK"
t_lock=$SECONDS
if ! flock -n 8; then
  log "条件 ③：机器锁 $MACHINE_LOCK 被占用，阻塞等待（每 $LOCK_LOG_S s 记一行）"
  until flock -w "$LOCK_LOG_S" 8; do log "条件 ③：仍在等机器锁（已等 $((SECONDS - t_lock)) s）"; done
fi
lock_wait=$((SECONDS - t_lock))
log "条件 ③ 满足：拿到机器锁，等了 $lock_wait s；持有到脚本退出"

gz_count() {
  local n
  if [[ -n "$GZ_PROBE" ]]; then n=$(bash -c "$GZ_PROBE" 2>/dev/null | head -n1); else n=$(pgrep -fc "$GZ_PATTERN" 2>/dev/null); fi
  echo "${n:-0}"
}
usage_usec() { awk '$1=="usage_usec"{print $2+0}' "$CPU_STAT_FILE"; }
t_idle=$SECONDS; last_idle_log=-999999; idle_cores=""; gz_n=""
while true; do
  u0=$(usage_usec); w0=$(date +%s.%N); sleep "$IDLE_WINDOW_S"; u1=$(usage_usec); w1=$(date +%s.%N)
  idle_cores=$(awk -v a="$u0" -v b="$u1" -v t0="$w0" -v t1="$w1" 'BEGIN{printf "%.2f", (b-a)/1e6/(t1-t0)}')
  gz_n=$(gz_count)
  idle_ok=$(awk -v c="$idle_cores" -v m="$IDLE_MAX_CORES" -v g="$gz_n" 'BEGIN{print (g==0 && c<m) ? 1 : 0}')
  [[ "$idle_ok" == 1 ]] && break
  waited=$((SECONDS - t_idle))
  if (( waited >= IDLE_LIMIT_S )); then
    finish 14 "条件 ④ 等了 $waited s 仍不满足（gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核，要求 0 个且 < $IDLE_MAX_CORES 核），超过 $IDLE_LIMIT_S s，没有起跑"
  fi
  if (( SECONDS - last_idle_log >= IDLE_LOG_S )); then
    log "条件 ④ 不满足：gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核（要求 0 个且 < $IDLE_MAX_CORES 核）；已等 $waited s，每 $IDLE_RECHECK_S s 复查"
    last_idle_log=$SECONDS
  fi
  sleep "$IDLE_RECHECK_S"
done
idle_wait=$((SECONDS - t_idle))
log "条件 ④ 满足：gz sim 进程 $gz_n 个，$IDLE_WINDOW_S s 窗口内容器 $idle_cores 核（< $IDLE_MAX_CORES）；等了 $idle_wait s"
cat > "$OUT/start.txt" <<EOF
gpu_ok=是
cpu_quota_cores=$cores
min_cores=$MIN_CORES
lock_wait_s=$lock_wait
gz_count=$gz_n
idle_window_s=$IDLE_WINDOW_S
idle_cores=$idle_cores
idle_max_cores=$IDLE_MAX_CORES
idle_wait_s=$idle_wait
started_at=$(date '+%F %T')
gpu=$GPU
lanes=$LANES
EOF

# ---------------------------------------------------------------- 条款 S2：第二轮与对照核对（拿到锁之后：第二轮持锁到它退出，拿到锁时它要么结束要么已死）
arm_complete_in() {  # $1 = 根目录，$2 = 臂名，$3 = 请求集数：eval.json 不是 partial 且 出结果 + 采集失败（含迟到集）= 请求数
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$1/$2/eval.json" "$3" 2>/dev/null
}
if [[ ! -f "$R2/DONE" ]]; then
  finish 20 "第二轮 $R2 没有 DONE（还没结束就拿到了机器锁，或进程已死）：按任务卡先停下报告，没有起任何臂"
fi
r2_rc=$(sed -n 's/^rc=\([0-9]*\).*/\1/p' "$R2/DONE" | head -n1)
if [[ "$r2_rc" != 0 ]]; then
  finish 20 "第二轮 DONE 的 rc 不是 0（$(head -c 300 "$R2/DONE")）：按任务卡先停下报告，没有起任何臂"
fi
[[ -f "$R2/L_STAR" ]] || finish 20 "第二轮没有 L_STAR（标定没有定下 L），没有起任何臂"
L_STAR=$(tr -d '[:space:]' < "$R2/L_STAR")
[[ -n "$DISP_CLAUSE" ]] || finish 20 "第二轮 $R2/config.env 里没有 DISP_CLAUSE，位移口径无从对齐，没有起任何臂"
missing=""
for s in 0 1 2; do for k in 1 3; do arm_complete_in "$R2" "acc-L$(printf %.2f "$L_STAR")-k$k-s$s" "$EPISODES" || missing="$missing acc-k$k-s$s"; done; done
[[ -z "$missing" ]] || finish 20 "第二轮的对照臂没有评完：$missing（L*=$L_STAR），没有起任何臂"
log "第二轮核对通过：DONE $(head -c 200 "$R2/DONE")；L*=$L_STAR；位移口径 $DISP_CLAUSE；对照六个臂评完"
if [[ "$CHECK_R2_INPUTS" == 1 ]]; then
  badin=""
  for f in "$SWA"/cal_swa_s{0,1,2}_k10.bin "$INFER" "$PLUGIN/libCfSitlSystem.so"; do
    have=$(sha256sum "$f" 2>/dev/null | cut -d' ' -f1); want=$(awk -v p="$f" '$2==p{print $1}' "$R2/inputs.txt" 2>/dev/null | head -n1)
    [[ -n "$want" && "$have" == "$want" ]] || badin="$badin $(basename "$f")"
  done
  [[ -z "$badin" ]] || finish 23 "权重 / 推理二进制 / 插件与第二轮 inputs.txt 的 sha256 不一致或没有记录：$badin；配对不成立，没有起任何臂"
  : > "$OUT/code_diff_vs_control.txt"
  for d in sim/tools sim/dynamics sim/worlds sim/cf_sitl; do
    diff -rq -x __pycache__ "$R2/code/$d" "$CODE/$d" 2>&1 | sed "s#$R2/code/#第二轮:#g; s#$CODE/#本实验:#g" >> "$OUT/code_diff_vs_control.txt"
  done
  [[ -s "$R2/code/sim/tools/collect.py" ]] || finish 23 "第二轮冻结代码 $R2/code 不在，无法核对代码差异；没有起任何臂"
  unexpected=$(grep -vE "$ALLOW_DIFF" "$OUT/code_diff_vs_control.txt" | head -n 5)
  [[ -z "$unexpected" ]] || finish 23 "本实验冻结代码与第二轮在协议相关目录下有允许清单之外的差异：$(tr '\n' ';' <<< "$unexpected")；配对不成立，没有起任何臂"
  log "输入与代码核对通过：权重、推理二进制、插件与第二轮同一份；协议相关目录的差异只有允许清单内的文件（$(grep -c . "$OUT/code_diff_vs_control.txt") 行，见 code_diff_vs_control.txt）"
fi

# ---------------------------------------------------------------- 条款 S1：探针（满电臂之前；通过一次后续跑不再重跑）
if [[ -f "$OUT/probe/PASS" ]]; then
  log "探针已通过（$OUT/probe/PASS），跳过"
else
  mkdir -p "$OUT/probe"
  log "探针开始：default 与 copy 两种世界各 episode $PROBE_EPISODES 的 gz sim（-v 3）"
  "$PY" "$PROBE_PY" --params "$PARAMS" --out "$OUT/probe" --episodes "$PROBE_EPISODES" --expect-profile "$PARAMS_PROFILE" --expect-sha256 "$PARAMS_SHA256" > "$OUT/probe.log" 2>&1
  rc=$?
  log "探针 rc=$rc：$(grep -E '^(PASS|FAIL)' "$OUT/probe.log" | tail -n1 | cut -c1-200)"
  case $rc in
    0) touch "$OUT/probe/PASS" ;;
    1) finish 21 "探针判据没过：插件没有按满电副本在算，或起飞失败；没有起任何满电臂。$(grep -E '^- 未过' "$OUT/probe/probe.md" 2>/dev/null | head -n 3 | tr '\n' ' ')" ;;
    3) finish 22 "探针的副本指纹核对没过：$(tr '\n' ';' < "$OUT/probe.log" | head -c 300)" ;;
    *) finish 21 "探针基础设施失败（rc=$rc）：$(grep -E '^FAIL' "$OUT/probe.log" | head -n 2 | tr '\n' ' ')；没有起任何满电臂" ;;
  esac
fi

# ---------------------------------------------------------------- 评测臂
cg_snapshot() {   # 一行：epoch usage_usec nr_throttled throttled_usec（cgroup 整个容器）
  awk -v t="$(date +%s.%N)" '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} $1=="throttled_usec"{h=$2+0} END{printf "%s %d %d %d\n", t, u, n, h}' "$CPU_STAT_FILE"
}
cg_mark() {       # $1 = start|end，$2 = 臂名：追加到 <臂>.cg（start 时重写）
  if [[ $1 == start ]]; then echo "start $(cg_snapshot)" > "$OUT/$2.cg"; else echo "end $(cg_snapshot)" >> "$OUT/$2.cg"; fi
}
cg_brief() { awk 'END{printf "usage_usec=%s nr_throttled=%s", $2, $3}' <<< "$(cg_snapshot)"; }
arm_complete() {  # $1 = 臂名，$2 = 请求集数：在 $OUT 下
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$1/eval.json" "$2" 2>/dev/null
}
export -f cg_snapshot cg_mark cg_brief arm_complete

# 一个评测臂：名字|评测参数串|请求集数|单集墙钟上限|种类（acc|none）。单臂失败不停整个队列，汇总里按 failed 标出来。
run_job() {
  local name=$1 args=$2 episodes=$3 ept=$4 kind=${5:-none}
  [[ -n "$name" && -n "$args" ]] || return 0
  if arm_complete "$name" "$episodes"; then
    log "$name 已评完，跳过"
  else
    local ts; ts=$(date +%H%M%S)
    if [[ -d "$OUT/$name" ]]; then   # 半成品挪开留证据，不删
      mv "$OUT/$name" "$OUT/$name.partial-$ts"
      for ext in log cg; do [[ -f "$OUT/$name.$ext" ]] && mv "$OUT/$name.$ext" "$OUT/$name.partial-$ts.$ext"; done
    fi
    cg_mark start "$name"
    log "$name 开始（$(cg_brief)）"
    # shellcheck disable=SC2086
    "$PY" "$EVAL" $args --episodes "$episodes" --sim-timeout 120.0 --episode-timeout "$ept" \
        --dynamics cf_sitl --vehicle e1_mini --hold 1.0 --window-from observation --out "$OUT/$name" > "$OUT/$name.log" 2>&1
    local rc=$?
    cg_mark end "$name"
    log "$name rc=$rc $(grep -E '^requested:' "$OUT/$name.log" | cut -c1-90)（$(cg_brief)）"
  fi
  [[ $kind == acc ]] && retry_job "$name" "$args" "$ept"
  return 0
}
retry_job() {   # 含迟到步的集重跑一次（条款 S3）：新输出子目录 <臂>-retry1，只含迟到的 seed，原目录保留
  local name=$1 args=$2 ept=$3 late n rargs
  late=$("$PY" "$SUMM_R2" --late-seeds "$OUT/$name" 2>/dev/null)
  [[ -n "$late" ]] || return 0
  n=$(tr ',' '\n' <<< "$late" | grep -c .)
  rargs=$(sed -E "s/--seeds [^ ]+/--seeds $late/" <<< "$args")
  log "$name 含迟到步的集 [$late]，重跑一次 → $name-retry1"
  run_job "$name-retry1" "$rargs" "$n" "$ept" none
}
export -f run_job retry_job

run_queue() {   # 标准输入：每行一个臂 名字|评测参数串|请求集数|单集墙钟上限|种类；顺序即优先级
  xargs -P "$LANES" -I{} bash -c 'IFS="|" read -r n a e t k <<< "{}"; run_job "$n" "$a" "$e" "$t" "$k"'
}

acc_job() {   # $1 = 1|3，$2 = 训练种子（学生臂不带 --rtf）；与第二轮的 acc_job 的命令行只差 --cf-sitl-params
  local ta=$K1_ARGS ept=$K1_EPT; [[ $1 == 3 ]] && { ta=$K3_ARGS; ept=$K3_EPT; }
  echo "soc100-L$(printf %.2f "$L_STAR")-k$1-s$2|--policy student --ckpt $SWA/cal_swa_s$2_k10.bin --infer-bin $INFER --heading-guard $GUARD $ta --split test --seeds $SEEDS --corridor-half-width $CORRIDOR --action-latency $L_STAR --cf-sitl-params $PARAMS|$EPISODES|$ept|acc"
}
refresh_summary() { "$PY" "$SUMM" "$OUT" --control "$R2" --note "$(date '+%F %T') 进行中：$1" > "$OUT/summarize.log" 2>&1; }

log "六个满电学生臂：calswa 三个训练种子 + 护栏 $GUARD，K=1 与 K=3 各 $EPISODES 个冻结 seed，L=$L_STAR，副本 $PARAMS；顺序 k1-s1 k3-s1 k1-s0 k3-s0 k1-s2 k3-s2（同第二轮）"
{ acc_job 1 1; acc_job 3 1; acc_job 1 0; acc_job 3 0; acc_job 1 2; acc_job 3 2; } | run_queue
refresh_summary "六个臂结束"

missing=""
for s in 0 1 2; do for k in 1 3; do
  arm="soc100-L$(printf %.2f "$L_STAR")-k$k-s$s"
  arm_complete "$arm" "$EPISODES" || missing="$missing $arm"
  late=$("$PY" "$SUMM_R2" --late-seeds "$OUT/$arm" 2>/dev/null)
  if [[ -n "$late" ]]; then arm_complete "$arm-retry1" "$(tr ',' '\n' <<< "$late" | grep -c .)" || missing="$missing $arm-retry1"; fi
done; done
[[ -z "$missing" ]] || finish 11 "有臂没跑完：$missing；读 summary.md，同一条命令重起可续跑"
"$PY" "$SUMM" "$OUT" --control "$R2" --note "$(date '+%F %T') 六个臂跑完" > "$OUT/summarize.log" 2>&1
sc=$?
verdict=$(head -n1 "$OUT/VERDICT" 2>/dev/null | cut -c1-300)
[[ $sc == 4 ]] && finish 24 "六个臂跑完，但汇总判 INVALID（指纹或配对协议键不一致）：$verdict；两档数据不可配对，读 summary.md 第 2 节"
[[ $sc == 2 ]] && finish 11 "六个臂跑完但汇总判数据不全：$verdict；读 summary.md"
finish 0 "六个满电臂跑完（L=$L_STAR，位移口径 $DISP_CLAUSE，副本 $PARAMS_PROFILE）；预注册判决：$verdict；读 summary.md"
