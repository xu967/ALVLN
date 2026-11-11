#!/usr/bin/env bash
# 动作派发时刻固定到仿真时间（任务卡 W1-0，collect.py --action-latency）第二轮：标定阶梯 → 学生验收 → 森林。同一个守护进程，零训练，用既有权重。
#
# 第二轮是看到第一轮失败之后设计的。第一轮（2026-11-10，run.sh，输出 exp-action-latency-20261110）的标定门在 L=0.16 与 L=0.20 都没过，
#   没有进入验收 4，DONE rc=10。第一轮的预注册与判定原样保留（run.sh、summarize.py 一个字没动），第二轮的数据不与第一轮混报。
#   本文件头的门、阶梯、验收的预测与证伪逐条写在起跑前，起跑后不改；要改就换一个新的文件名与输出目录。
#
# 起跑前已知的第一轮事实（来源：exp-action-latency-20261110 下的 analysis-round1.md、profile-late-steps.md，11-11 离线生成，提交 0600d00；写在这里免得事后当发现）：
#   · 迟到步：K=3 在 L=0.16 为 36/823 步，在 L=0.20 为 3/821 步（seed 6 步 2、seed 7 步 0、seed 10 步 0，各超出阈值 5 ms）；K=1 在 L=0.16 为 1/335，在 L=0.20 为 0/338。
#   · 慢步的来源是教师那一步的 A*，不是左右并排的 lr_tie_margin（第一轮用 tiebreak=legacy，只跑一次 plan_action_global）。K=3 L=0.16 按 A* 计数分类的迟到率：
#     A* 无路径、退回局部规划 16/59（27.1%），A* 有路径且出堆数 ≥ 1500（长路径、近起点）19/56（33.9%），其余 1/708（0.1%）；L=0.20 为 0/34、3/56（5.4%）、0/731。
#     迟到步的离线 CPU 耗时中位 41.1 ms，随机不迟到步 15.6 ms（比值 2.63，Mann–Whitney p = 2.7e-8）。
#   · 位移差：K=3 对 K=1 的稳态前飞位移（上一步也是前进，第一轮的门口径，全部步）低 3.3%（L=0.16）与 2.6%（L=0.20）。收窄到步 3–30 后为 -2.5% [-4.1, -0.9] 与 -1.8% [-3.8, -0.2]，
#     所以「步段 / 电量状态不同」只解释 0.8 与 0.7 个百分点（占 25% 与 28%）；主要来源是动作上下文的构成（连续前进的第几步：r=2、r=3 的步位移比巡航步大 0.02–0.07 m，K=3 的直线更长，
#     这两类步占比更低），按 K=1 的构成加权后为 -1.2% 与 -0.1%。只比巡航步（连续前进第 4 步起，步 3–30）为 -0.8% [-2.3, +0.5] 与 +0.0% [-0.8, +0.9]。
#   · 验收 2（学生 calswa s1 + 护栏 2，K=1，L=0.16）：逐集一致率 新对下午 11/20=0.55，新对晚上 13/20=0.65，低于预测的 70%；含 1 个迟到集（seed 172 步 0）。
#
# ---------------------------------------------------------------- 预注册（第二轮，2026-11-11 起跑前写下）
#   条款 R1（阶梯）。教师，柱阵 K=1 与长课程 K=3 各 10 个 train 分区 seed（2,3,4,6,7,8,9,10,11,12），全部重跑，不复用第一轮数据。
#     A 组：L=0.20，默认 RTF，K=1 与 K=3。过门则 L=0.20、不带 --rtf，后面的组不跑。
#     B 组（A 组未过且原因里含迟到步）：K=3 带 --rtf 0.5，L=0.20；K=1 复用 A 组的臂（同 L、同 seed、同参数）。过门则 L=0.20，教师 K=3 要带 --rtf 0.5。
#     C 组（B 组仍含迟到步）：L=0.24，默认 RTF，K=1 与 K=3 都重跑。过门则 L=0.24。
#     某组未过且原因里没有迟到步（只有位移 / 转角 / 生效时长）：停，DONE 写明原因，不往下试。三组都没过：停。
#     预期（先写下）：A 组大概率仍有迟到步（第一轮 L=0.20 的 K=3 有 3 步，按均值 3 的泊松估算 0 步的概率约 5%）；是否过 B 组取决于容器负载，L=0.24 的阈值（0.26 s）余量最大。
#   条款 R2（门，逐条，每条都过才算过；数的口径见 summarize_r2.py 文件头）。
#     1. 迟到步：K=1 与 K=3 都为 0，含每集第 0 步。
#     2. 逐步生效时长（oracle 窗口 − dispatch_lag_s；各集均值的集间中位）：K=3 对 K=1 的相对差 ≤ 2%。
#     3. 转角（YAW 步 |dyaw_deg| 的池化均值）：相对差 ≤ 2%。
#     4. 稳态前飞位移，只比步 3–30：相对差 ≤ 2%。口径由 DISP_CLAUSE 选，二选一，起跑前必须显式给（缺省不起跑，rc=5），实际用的写进 config.env：
#        cruise = 本步及前 3 步都是 FORWARD_SLOW、本步无碰撞（连续前进第 4 步起）。第一轮归因后的推荐口径：它剔掉了 r=2、r=3 的构成差。
#        steady = 本步与上一步都是 FORWARD_SLOW、本步无碰撞（任务卡与第一轮的稳态口径，限步 3–30）。
#        两种口径的数都写进汇总，不进门的那一种只报告。全部步的口径（第一轮的门口径）只报告，不进门。整集位移差只报告，不进门。
#     迟到步数用首次记录，标定臂不重跑（迟到步就是测量对象）。
#     证伪：若第一轮的归因成立，cruise 口径在 L=0.20 的 K=3 对 K=1 应在 ±2% 内（第一轮 +0.0%）；若本轮 cruise 口径超出 2%，归因里「主要来自构成」的说法被推翻，要重看。
#   条款 R3（验收 2）。门过之后，学生 calswa 三个训练种子（$SWA/cal_swa_s{0,1,2}_k10.bin）+ --heading-guard 2 + --action-latency L*，K=1 的 20 个冻结 seed 每个种子各一组。
#     种子 1 的 20 集对 exp-heading-guard-20261110/guard2-calswa-s1（下午，17/20）与 exp-k1-control-20261110/guard2-calswa-s1（晚上对照，17/20）逐集配对。
#     预测：逐集一致率对两者都不低于 70%（差别在同权重同 seed 重评的波动之内）。证伪：任一不足 70%。三个训练种子合起来（60 集）是新协议下的 K=1 参照。学生臂不带 --rtf。
#   条款 R4（验收 4，任务卡 W1-0 第 4 条原文）。同一批权重与 seed，K=3（--task course_base --blocks 3 --steps 240）三个训练种子各 20 个冻结 seed（共 60 集），
#     比较量 = K=3 第 1 块通过率对 K=1 成功率，配对精确 McNemar，差 = K=1 − K=3。预注册：新协议下相差不超过 0.08，则 11-10 的下降（旧协议 39/60 对 50/60）来自第一步的时序；
#     K=3 仍低 0.12 以上，则来自后面几块的柱子进了画面（学生没见过远处的柱子）；在 0.08 与 0.12 之间：两条线都没覆盖，如实记录，不归因。
#     新协议每步生效 hold − L（L=0.20 为 0.80 s，比旧协议的 0.871 s 短 8%）。每步前飞位移的缩幅比时长的缩幅大：第一轮教师 K=1 从 L=0.16 到 0.20，逐步生效时长 -4.7%，稳态位移 -10.8%（弹性 2.38，analysis-round1.md 第 3 节）；
#     照这个弹性外推到 0.80 s（未核实，是估计），每步位移约比旧协议小 17%–18%，同一段航程约多飞 20%–22% 的步，高于「多飞约 8% 的步」的粗估。以汇总里 K=3 的实测步数与 sim_s 为准。多飞的步数让 K=3 离半电池（约 100 步）近一些；
#     验收 4 只看第 1 块，汇总另报 K=3 逐块存活 S_1/S_2/S_3、掉高集数与 sim_s 分布，并排旧协议（K=1 50/60；K=3 逐块 39/27/19，共 60 集）。
#   条款 R5（森林）。排在最后，低优先级：L* 下 forest_e1 三档（sparse / medium / dense），教师，各 3 集，带 --action-latency L*（不带 --rtf）。只报迟到步与 dispatch_lag_s 分布，不设门。
#   条款 R6（含迟到步的集）。学生臂里含迟到步的集（基础设施失败，不计成功）自动重跑一次：新输出子目录 <臂>-retry1，只含迟到的 seed，原目录保留，两份记录都留；
#     汇总以重跑结果为准（重跑仍迟到则按没成功），首次记录另列。不改库代码，只用评测入口已有的 --seeds / --episodes。标定臂与森林臂不重跑。
#     电池掉压造成的采集中止是测量结果，不重跑（沿用第一轮口径）。
#   条款 R7（资源与起跑条件）。LANES=2 全程。起跑条件：① nvidia-smi -L 里有配置的 GPU；② cgroup 配额 ≥ 12 核；③ 拿到机器锁 $MACHINE_LOCK（阻塞等待，每 10 分钟记一行）；
#     ④ 拿到锁后没有 gz sim 进程，且 30 s 窗口内容器 CPU（cpu.stat usage_usec 增量）< 1.5 核。①② 不满足：立刻退出并写 DONE；④ 不满足：每分钟复查，每 10 分钟记一行，最多 3 小时，超时写 DONE。
#     不再用「1 分钟 load < 8」（/proc/loadavg 是整机的，不代表本容器）。每个臂开始与结束各记一次 usage_usec 与 nr_throttled（run.log 与 <臂>.cg），
#     汇总给每臂平均核数与限流增量，增量大于 0 的臂标 ⚠，不自动作废，是否采信由主会话定。
#
# 运行：mkdir -p $OUT && cp run_r2.sh $OUT/ && cd $OUT && DISP_CLAUSE=<cruise|steady> setsid nohup bash run_r2.sh > /dev/null 2>&1 < /dev/null &
#   起跑前必须是已提交的版本：脚本会拿自己与 CODE_HEAD（缺省 SRC 的 HEAD）里的同名文件比哈希，不一致 rc=4。冻结代码 = git archive CODE_HEAD，解到 $OUT/code，
#   并逐文件核对 sim/ 的 git blob 哈希；汇总脚本也从冻结副本里跑。续跑：同一条命令重起，评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删），沿用起跑时的 CODE_HEAD 与 DISP_CLAUSE。
# 进度：tail -f $OUT/run.log；汇总随时可跑：<解释器> $OUT/code/experiments/2026-11-10-action-latency/summarize_r2.py $OUT（缺的臂标「未完成」）；
#   只看某组标定门：同一脚本加 --gate A|B|C。结束落 DONE（rc、时间、原因）与 summary.md。停：kill -- -$(cat $OUT/run.pid)（只按进程组，不用 pkill -f）。
# DONE 的 rc：0 全部跑完；3 已有实例（不写 DONE）；4 冻结代码或脚本版本不一致；5 没给 DISP_CLAUSE 或与 config.env 不一致；10 阶梯结束但没有组过门（含只因位移 / 转角 / 生效时长未过）；
#   11 数据不全；12 配置的 GPU 不在；13 cgroup 配额不足；14 等空闲超过 3 小时；130 被信号终止。
# 时间估计（按第一轮 1 条车道的用时折算，仅为估计）：K=3 教师 10 集约 20 分钟，K=1 教师 10 集约 10 分钟，学生 K=3 20 集约 45 分钟，学生 K=1 20 集约 20 分钟；
#   A 组过门约 2.2 小时（标定 20 分钟 + 学生六个臂 + 森林），走到 C 组约 3.2 小时，另加等锁与等空闲的时间。
set -u
export LC_ALL=C

# ---------------------------------------------------------------- 配置（改这里；带 ${X:-} 的可由环境变量覆盖，演练用）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
OUT=${OUT:-$D/exp-action-latency-r2-20261111}
SRC=${SRC:-/root/claude-code-backup-20261028/code/wt/p12-w1-0}   # 代码源（分支 p12-w1-0）：只用来 git archive 冻结代码
CODE_HEAD=${CODE_HEAD:-}                                 # 冻结代码的提交；缺省 = 续跑时沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与 11-10 的 K=1 / K=3 评测同一个推理二进制
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
SWA=$D/exp-swa-calib-20261110/ckpt                       # 权重平均 + 校准权重 cal_swa_s{0,1,2}_k10.bin
GPU=GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740             # 11-11 容器重启后 uav_rgb 用的这张卡（RTX 2080 Ti），用户 11-11 19:23 指定；只在本任务卡里改，第一轮 run.sh 写的旧 UUID 是冻结的实验记录
SEEDS=1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654   # tools/eval_seeds.py EVAL_SEEDS_BASE_20
CAL_SEEDS=2,3,4,6,7,8,9,10,11,12                         # 标定的 train 分区 seed：K=1 与 K=3 前 10 个 train seed（离线核过，两边相同）
FOREST_SEEDS=4100:4140                                   # 森林：从中按 train 分区取前 FOREST_EPISODES 个
FOREST_EPISODES=3
L_A=0.20                                                 # 条款 R1：A 组与 B 组的 L
L_C=0.24                                                 # 条款 R1：C 组的 L
RTF_B=0.5                                                # 条款 R1：B 组 K=3 的 --rtf
EPISODES=20                                              # 验收臂的请求集数 = 冻结 seed 个数
CAL_EPISODES=10                                          # 标定臂的集数
GUARD=2                                                  # 与 11-10 同一个护栏阈值
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
K1_ARGS="--task base --steps 80"                         # 柱阵 K=1
K3_ARGS="--task course_base --blocks 3 --steps 240"      # 长课程 K=3（80·K 步预算，到成功线即停）
K1_EPT=1500                                              # 每集墙钟上限（秒）：同 heading-guard 实验
K3_EPT=2520                                              # 600 + 8·240：同 long-course 实验
LANES=${LANES:-2}                                        # 条款 R7：全程 2 条车道
DISP_CLAUSE=${DISP_CLAUSE:-}                             # 条款 R2.4：cruise | steady，必须显式给
EVAL=${EVAL:-}                                           # 评测入口；缺省 = 冻结代码里的 eval_closed_loop.py（只有演练流程时才覆盖）
SELFCHECK=${SELFCHECK:-1}                                # 1 = 起跑前核对本脚本与 CODE_HEAD 里的同名文件一致（演练时才关）
# 起跑条件（条款 R7）
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
export OUT CODE=$OUT/code PY=$R/.venv-tools/bin/python CPU_STAT_FILE
SUMM=$CODE/experiments/2026-11-10-action-latency/summarize_r2.py
export SUMM
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg

log() { echo "$(date '+%F %T') $*"; }
export -f log

finish() {   # $1 = rc，$2 = 原因：刷新 summary.md（冻结代码已就位时），写 DONE，退出
  trap - TERM INT
  [[ -f "$SUMM" ]] && "$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 生成，DONE rc=$1" > "$OUT/summarize.log" 2>&1
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }

# 口径先于其它一切：DISP_CLAUSE 必须显式给；续跑时与 config.env 里的必须一致。拒绝起跑 = rc=5：原因写 refused.txt，DONE 只在还没有时才写（不盖掉上一次的结果）
refuse() {
  echo "$1" >&2
  echo "$(date '+%F %T') $1" >> "$OUT/refused.txt"
  [[ -f "$OUT/DONE" ]] || echo "rc=5 $(date '+%F %T') $1" > "$OUT/DONE"
  exit 5
}
if [[ -f "$OUT/config.env" ]]; then
  prev=$(sed -n 's/^DISP_CLAUSE=//p' "$OUT/config.env" | head -n1)
  if [[ -n "$DISP_CLAUSE" && -n "$prev" && "$DISP_CLAUSE" != "$prev" ]]; then
    refuse "DISP_CLAUSE=$DISP_CLAUSE 与 $OUT/config.env 里起跑时的 $prev 不一致；续跑沿用起跑时的口径，本次不起跑"
  fi
  DISP_CLAUSE=${DISP_CLAUSE:-$prev}
fi
if [[ "$DISP_CLAUSE" != cruise && "$DISP_CLAUSE" != steady ]]; then
  refuse "没给 DISP_CLAUSE（cruise|steady）：门的位移条款口径要由主会话定，见文件头条款 R2.4；本次不起跑"
fi
export DISP_CLAUSE

exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；评完的臂保留，同一条命令重起可续跑"' TERM INT
log "开始 PGID=$$ lanes=$LANES gpu=$GPU 位移口径=$DISP_CLAUSE 阶梯 L=$L_A→(B: --rtf $RTF_B)→$L_C 机器锁=$MACHINE_LOCK"

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
  rel=experiments/2026-11-10-action-latency/run_r2.sh
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
[[ -f "$SUMM" ]] || finish 4 "冻结副本里没有 summarize_r2.py（CODE_HEAD $CODE_HEAD 之前的提交？）"
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}
export EVAL
{ echo "code_head $CODE_HEAD"
  echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/teacher.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  sha256sum "$SWA"/cal_swa_s{0,1,2}_k10.bin "$INFER" "$PLUGIN/libCfSitlSystem.so"; } > "$OUT/inputs.txt" 2>&1
[[ -f "$OUT/config.env" ]] || cat > "$OUT/config.env" <<EOF
# run_r2.sh 起跑配置（summarize_r2.py 读 DISP_CLAUSE；续跑沿用这里的口径）
DISP_CLAUSE=$DISP_CLAUSE
CODE_HEAD=$CODE_HEAD
GPU=$GPU
LANES=$LANES
L_A=$L_A
L_C=$L_C
RTF_B=$RTF_B
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

# ---------------------------------------------------------------- 评测臂
cg_snapshot() {   # 一行：epoch usage_usec nr_throttled throttled_usec（cgroup 整个容器）
  awk -v t="$(date +%s.%N)" '$1=="usage_usec"{u=$2+0} $1=="nr_throttled"{n=$2+0} $1=="throttled_usec"{h=$2+0} END{printf "%s %d %d %d\n", t, u, n, h}' "$CPU_STAT_FILE"
}
cg_mark() {       # $1 = start|end，$2 = 臂名：追加到 <臂>.cg（start 时重写）
  if [[ $1 == start ]]; then echo "start $(cg_snapshot)" > "$OUT/$2.cg"; else echo "end $(cg_snapshot)" >> "$OUT/$2.cg"; fi
}
cg_brief() { awk 'END{printf "usage_usec=%s nr_throttled=%s", $2, $3}' <<< "$(cg_snapshot)"; }
arm_complete() {  # $1 = 臂名，$2 = 请求集数：eval.json 不是 partial 且 出结果 + 采集失败（含迟到集）= 请求数
  "$PY" -c 'import json,sys; e=json.load(open(sys.argv[1])); sys.exit(0 if (not e.get("partial") and e.get("episodes_done",0)+e.get("failed_count",0)==int(sys.argv[2])) else 1)' "$OUT/$1/eval.json" "$2" 2>/dev/null
}
export -f cg_snapshot cg_mark cg_brief arm_complete

# 一个评测臂：名字|评测参数串|请求集数|单集墙钟上限|种类（cal|acc|forest|none）。单臂失败不停整个队列，汇总里按 failed 标出来。
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
retry_job() {   # 学生臂含迟到步的集重跑一次（条款 R6）：新输出子目录 <臂>-retry1，只含迟到的 seed，原目录保留
  local name=$1 args=$2 ept=$3 late n rargs
  late=$("$PY" "$SUMM" --late-seeds "$OUT/$name" 2>/dev/null)
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

cal_job() {   # $1 = L，$2 = 1|3，$3 = --rtf（可空）
  local L=$1 k=$2 rtf=${3:-} ta=$K1_ARGS ept=$K1_EPT
  [[ $k == 3 ]] && { ta=$K3_ARGS; ept=$K3_EPT; }
  echo "cal-L$(printf %.2f "$L")${rtf:+rtf$rtf}-k$k|--policy teacher $ta --split train --seeds $CAL_SEEDS --corridor-half-width $CORRIDOR --action-latency $L${rtf:+ --rtf $rtf}|$CAL_EPISODES|$ept|cal"
}
acc_job() {   # $1 = L，$2 = 1|3，$3 = 训练种子（学生臂不带 --rtf）
  local ta=$K1_ARGS ept=$K1_EPT; [[ $2 == 3 ]] && { ta=$K3_ARGS; ept=$K3_EPT; }
  echo "acc-L$(printf %.2f "$1")-k$2-s$3|--policy student --ckpt $SWA/cal_swa_s$3_k10.bin --infer-bin $INFER --heading-guard $GUARD $ta --split test --seeds $SEEDS --corridor-half-width $CORRIDOR --action-latency $1|$EPISODES|$ept|acc"
}
forest_job() {   # $1 = L，$2 = 密度（不带 --rtf，只报分布）
  echo "forest-L$(printf %.2f "$1")-$2|--policy teacher --task forest_e1 --density $2 --steps 80 --split train --seeds $FOREST_SEEDS --action-latency $1|$FOREST_EPISODES|$K1_EPT|forest"
}
gate() {   # $1 = A|B|C：判门，结果放 GATE_RC / GATE_TEXT 并记一行
  GATE_TEXT=$("$PY" "$SUMM" --gate "$1" --disp-clause "$DISP_CLAUSE" "$OUT"); GATE_RC=$?
  log "标定门 $1 rc=$GATE_RC $GATE_TEXT"
}
refresh_summary() { "$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 进行中：$1" > "$OUT/summarize.log" 2>&1; }

# ---------------------------------------------------------------- 阶段 1：标定阶梯（条款 R1、R2）
run_group() {   # $1 = A|B|C：跑该组缺的臂（评完的跳过），然后判门
  log "标定组 $1 开始"
  case $1 in
    A) { cal_job "$L_A" 3 ""; cal_job "$L_A" 1 ""; } | run_queue ;;
    B) { cal_job "$L_A" 3 "$RTF_B"; } | run_queue ;;
    C) { cal_job "$L_C" 3 ""; cal_job "$L_C" 1 ""; } | run_queue ;;
  esac
  gate "$1"
  refresh_summary "标定组 $1 判门 rc=$GATE_RC"
}
L_STAR=""; RTF_STAR=""; G_STAR=""; TXT_A=""; TXT_B=""; TXT_C=""
run_group A; TXT_A=$GATE_TEXT
case $GATE_RC in
  0) G_STAR=A; L_STAR=$L_A ;;
  1) log "A 组未过且原因含迟到步，试 B 组（K=3 带 --rtf $RTF_B）"
     run_group B; TXT_B=$GATE_TEXT
     case $GATE_RC in
       0) G_STAR=B; L_STAR=$L_A; RTF_STAR=$RTF_B ;;
       1) log "B 组未过且原因含迟到步，试 C 组（L=$L_C）"
          run_group C; TXT_C=$GATE_TEXT
          case $GATE_RC in
            0) G_STAR=C; L_STAR=$L_C ;;
            1) finish 10 "标定阶梯三组都没过（都含迟到步），没有进入验收。A：${TXT_A#* }；B：${TXT_B#* }；C：${TXT_C#* }。下一步由主会话定" ;;
            3) finish 10 "C 组未过，原因里没有迟到步，按规则停，没有进入验收。${TXT_C#* }。A：${TXT_A#* }；B：${TXT_B#* }" ;;
            *) finish 11 "C 组标定数据不全，没有进入验收：${TXT_C#* }" ;;
          esac ;;
       3) finish 10 "B 组未过，原因里没有迟到步（只有位移 / 转角 / 生效时长），按规则停，没有进入验收。${TXT_B#* }。A：${TXT_A#* }" ;;
       *) finish 11 "B 组标定数据不全，没有进入验收：${TXT_B#* }" ;;
     esac ;;
  3) finish 10 "A 组未过，原因里没有迟到步（只有位移 / 转角 / 生效时长），按规则停，没有进入验收。${TXT_A#* }" ;;
  *) finish 11 "A 组标定数据不全，没有进入验收：${TXT_A#* }" ;;
esac
echo "$L_STAR" > "$OUT/L_STAR"
[[ -n "$RTF_STAR" ]] && echo "$RTF_STAR" > "$OUT/RTF_STAR"
log "标定阶梯结束：$G_STAR 组过门，新协议参数 L=$L_STAR${RTF_STAR:+，教师 K=3 带 --rtf $RTF_STAR}；学生臂与森林不带 --rtf"

# ---------------------------------------------------------------- 阶段 2：验收 2 与 4（条款 R3、R4、R6）
log "阶段 2：学生 calswa 三个训练种子 + 护栏 $GUARD，K=1 与 K=3 各 $EPISODES 个冻结 seed，L=$L_STAR；顺序 k1-s1 k3-s1 k1-s0 k3-s0 k1-s2 k3-s2（先把验收 2 的那一组评完）"
{ acc_job "$L_STAR" 1 1; acc_job "$L_STAR" 3 1; acc_job "$L_STAR" 1 0; acc_job "$L_STAR" 3 0; acc_job "$L_STAR" 1 2; acc_job "$L_STAR" 3 2; } | run_queue
refresh_summary "阶段 2 结束"

# ---------------------------------------------------------------- 阶段 3：森林三档（条款 R5，低优先级，最后）
log "阶段 3：森林三档，教师，各 $FOREST_EPISODES 集，L=$L_STAR（只报迟到步与 dispatch_lag_s 分布，不设门）"
{ for d in sparse medium dense; do forest_job "$L_STAR" "$d"; done; } | run_queue

missing=""
for s in 0 1 2; do for k in 1 3; do arm_complete "acc-L$(printf %.2f "$L_STAR")-k$k-s$s" "$EPISODES" || missing="$missing acc-k$k-s$s"; done; done
for d in sparse medium dense; do arm_complete "forest-L$(printf %.2f "$L_STAR")-$d" "$FOREST_EPISODES" || missing="$missing forest-$d"; done
[[ -z "$missing" ]] || finish 11 "标定 $G_STAR 组过门（L=$L_STAR），但有臂没跑完：$missing；读 summary.md，同一条命令重起可续跑"
finish 0 "标定 $G_STAR 组过门（L=$L_STAR${RTF_STAR:+，教师 K=3 带 --rtf $RTF_STAR}），验收 2、4 与森林全部跑完；位移口径 $DISP_CLAUSE；读 summary.md，最后用哪个 L 由主会话定"
