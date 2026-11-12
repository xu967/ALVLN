#!/usr/bin/env bash
# 工作点扫描（任务卡 E2E-R1 第二部分，闭环：Gazebo，守护进程，排整机锁，2 条车道，新协议）。零训练。
#   现成的 cal_swa_s{0,1,2}_k10.bin（末 10 轮平均 + 前进校准）的 FORWARD 偏置，在现有校准值上再加 −0.5 / 0 / +0.5 nat 三档，
#   3 个训练种子 × 20 个 val 分区几何 = 每档 60 集；第一部分有卡里的臂过线时，再加一个配方臂（该配方末 10 轮平均 + 校准，不平移，60 集）。
# 背景：前进偏置现在取的是教师状态分布上的前进占比 0.873（val 标签里的 FORWARD 占比），从来没有在闭环上调过（任务卡 E2E-R1「回答的问题」）。
#
# 运行：
#   mkdir -p $PART1/closed-loop && cd $PART1/closed-loop && setsid nohup bash <本脚本的路径> > /dev/null 2>&1 < /dev/null &      # PART1 = $D/exp-recipe-stability-20261112
#   起跑前必须是已提交的版本：脚本拿自己与 CODE_HEAD（缺省 = SRC 的 HEAD；续跑沿用 $OUT/code_head.txt）里的同名文件比哈希，不一致 rc=4。冻结代码 = git archive CODE_HEAD，
#   解到 $OUT/code，逐文件核对 sim/、tools/、本目录的 git blob 哈希；之后所有 Python 工具（评测入口、权重平移、汇总）都从冻结副本里跑。
#   续跑：同一条命令重起，评完的臂跳过，半成品挪到 <臂>.partial-<时刻>（不删），沿用起跑时的 CODE_HEAD 与配方臂的判定。
# 进度：tail -f $OUT/run.log；汇总随时可跑（缺的臂标「未完成」）：<解释器> $OUT/code/experiments/2026-11-12-recipe-stability/summarize_sweep.py $OUT；结束落 DONE 与 summary.md。
# 停：kill -- -$(cat $OUT/run.pid)，只按进程组，不用 pkill -f；停之前核对：ps -o lstart=,args= -p $(cat $OUT/run.pid) 的启动时刻晚于 ps -o lstart= -p 1，命令行是本脚本。
# DONE 的 rc：0 全部臂跑完；3 已有实例（不写 DONE）；4 冻结代码或脚本版本不一致；5 拒绝起跑（正式运行遇到 RS_ 覆盖、第一部分结果没提交、冻结几何清单核对不过、L* 不是 0.20、输入文件缺失）；
#   11 有臂没跑完（含采集整体失败）；12 配置的 GPU 不在；13 cgroup 配额不足；14 拿到机器锁后等 gz sim 进程退出超过 3 小时；130 被信号终止。
#
# ---------------------------------------------------------------- 起跑前已知的事实（2026-11-12，写在这里免得事后当发现）
#   · 几何空间只有 804 个不同几何（base 任务 seed 0–49999：train 662 / val 71 / test 71，split_manifest 实测），71 个 val 几何的首次出现 seed 全部 ≤ 2632。
#     任务卡写的「一段没被别的实验用过的 seed 区间」因此不能保证几何没被用过：高位 seed 只是重复低位几何。清单改按几何组 ID（gid）挑，见 select_val_geoms.py 文件头：
#     候选 seed 取 [30000, 50000)（这段 seed 没被任何已有清单或评测用过，脚本核对），排除训练数据集 300 集的全部几何（含它的 14 个 val 几何，校准与选 ckpt 用过）、
#     所有 base-v5 采集清单与 eval.json 里出现过的几何、tools/eval_seeds.py 的全部清单；71 个 val 几何里排除 40 个，余 31 个，取 seed 升序前 20 个（30012…30944）。
#     与训练数据集的几何重合 0 个，与任一 base-v5 清单 / 评测的几何重合 0 个（按扫描时刻 2026-11-12 01:43；别的卡之后新采新评的不在内）。清单经 scene_gen 真生成核对：gid 与分区全部一致。
#   · 三个训练种子的 cal_swa 权重 fc.b[FORWARD] 分别为 +1.6166、+1.4694、+1.5600（校准后的值；make_shifted_ckpts.py 输出），平移 ±0.5 后改的就是这一个数。
#   · 对照的新协议：run_r2.sh 的 acc_job 命令行去掉 --heading-guard，--action-latency 取 $D/exp-action-latency-r2-20261111/L_STAR（0.20），学生臂不带 --rtf；
#     --task base --steps 80 --split val。口径 sim09.v2.coast1s.corridor3m（成功 = 无碰撞、过成功线、全程 |y| ≤ 3 m）。
#   · 第一部分已写进报告并提交（卡的要求：先于本脚本起跑，提交 0623948）；本脚本读冻结副本里 results/stability.json 的 extra_arm 决定是否加配方臂（空 = 不加）。
#     第一部分的结果：卡里的 R2、R3 被训练器拒绝零权重没有跑成，R1 不过线，替代臂 R2e、R3e（SAFE 权重 1e-6）都过线；stability.json 里 extra_arm_card = null，
#     extra_arm = R2e（替代臂，extra_arm_is_substitute = true）。所以本次的配方臂是 R2e，不是卡里的臂（见条款 P4 的 2026-11-12 更新）。
#
# ---------------------------------------------------------------- 预注册（第二部分，起跑前写下；档、几何数、阈值抄自任务卡 E2E-R1，起跑后不改；要改就换新文件名与新输出目录）
#   条款 P1（档）。FORWARD 偏置在现有校准值上加 −0.5、0、+0.5 nat。0 档 = 原权重文件（不拷贝、不重写）；±0.5 档由 tools/average_ckpt.py 的 shift_forward_bias 生成，
#     不训练（make_shifted_ckpts.py；核对除 fc.b[FORWARD] 一个 float32 外与原文件逐字节相同、变化量等于平移量）。
#   条款 P2（几何）。20 个 val 分区几何，清单冻结在下面的 GEOM_SEEDS（sha256 记在 GEOM_SHA256，起跑时核对）；不在冻结的 test seed 上跑任何一档。
#   条款 P3（判定一，工作点敏感度）。三档成功总数（各 60 集）最高与最低相差 ≥ 8 集：工作点在闭环上是一个要显式调的量；相差 < 4 集：在 ±0.5 nat 内闭环对它不敏感；
#     相差 4–7 集：两条线都没覆盖，如实记录，不下「敏感」或「不敏感」的结论。
#   条款 P4（判定二，配方臂；只在第一部分有卡里的臂过线时有）。配方臂（该配方三个训练种子的末 10 轮平均加校准，不平移）过线 = 三个训练种子的成功数极差 ≤ 3，
#     并且总数 ≥ 同批几何上 0 档的总数 − 3。几个臂同时过线时只加一个（stability.py 的 choose_extra_arm：末 10 轮原始翻转率最小，并列取编号小的）。
#     （2026-11-12 更新，起跑前，看到第一部分结果之后：卡里的臂都没过线、替代臂 R2e 与 R3e 过线。卡没有规定替代臂的加臂；按同一规则在过线的替代臂里取一个，
#     choose_extra_arm_run，取 R2e（0.008 < 0.012），标明替代，过线判据原样。这不是预注册，是看到第一部分结果之后的决定；主会话可以不采用这个臂的结果。）
#   条款 P5（另报）。每档的撞柱、越界、未到线，迟到与中止；动作占比；三个训练种子各自的成功数；逐几何成功数。主比较按几何聚类
#     （同一几何上三个训练种子的成败差先加总，再对 20 个几何做符号翻转精确检验；区间 = 按几何重抽样 bootstrap），逐集配对精确 McNemar 并列，两者结论不一致时照实写。
#     三个两两比较（−0.5 对 0、+0.5 对 0、−0.5 对 +0.5）各报原始 p，不做多重比较校正。单次 n=20 的比较不下结论（公共规则 §2）。
#   条款 P6（含迟到步的集；run_r2.sh 条款 R6）。学生臂里含迟到步的集（基础设施失败，不计成功，占请求分母）自动重跑一次：新输出子目录 <臂>-retry1，只含迟到的 seed，
#     原目录保留，两份记录都留；汇总以重跑结果为准（重跑仍迟到 / 中止则按没成功），首次记录另列，有重跑的臂在汇总里标「重评」。采集中止（不是迟到）是测量结果，不重跑。
#   条款 P7（资源与起跑条件）。LANES=2 全程。起跑条件：① nvidia-smi -L 里有配置的 GPU；② cgroup 配额 ≥ 12 核；③ 拿到机器锁 $MACHINE_LOCK（阻塞等待，每 10 分钟记一行）；
#     ④ 拿到锁后没有 gz sim 进程（每分钟复查，最多 3 小时）。run_r2.sh 的「30 s 窗口容器 CPU < 1.5 核」这一条不沿用：它是为标定时序测量设的，这里整机在训练池并行
#     工作，容器常年 20 核以上负载，沿用会没完没了地等；窗口内的容器平均核数仍记进 start.txt 供对照。每个臂起止各记一次 usage_usec 与 nr_throttled（<臂>.cg），
#     汇总给每臂平均核数与限流增量，增量大于 0 的臂标 ⚠，不自动作废，是否采信由主会话定。
#   条款 P8（臂的顺序）。按训练种子分组、组内 0 档、−0.5、+0.5、配方臂相邻，使各档的时段尽量重叠，减少「哪一档碰上哪段负载」的混杂。
#   不做：改 C++ 与库代码；logit adjustment 与权重 EMA（卡 E2E-P1）；在冻结的 test seed 上跑任何一档；加档（±0.5 之外）或加种子；看到结果后换几何清单。
#   预期：任务卡对方向没有预期，这里也不写。
#
# 时间估计（按 11-11 R2 的实测折算，仅为估计）：每集约 55 s（2 条车道各 1 个进程），每臂 20 集约 18 分钟；9 个档臂 + 3 个配方臂 = 12 臂，2 条车道约 1.8 小时（无配方臂 1.4 小时），
#   另加等锁与迟到集重跑的时间。等锁时间不可预估（卡里估计 4–5 小时）。
set -u
export LC_ALL=C

# ---------------------------------------------------------------- 配置（改这里；RS_ 前缀的覆盖只在演练 RS_REHEARSAL=1 时允许）
R=/root/claude-code-backup-20261028/code/uav_rgb         # 主仓库：只借用 .venv-tools 解释器
D=/root/rivermind-data/uav_rgb
PART1=${RS_PART1:-$D/exp-recipe-stability-20261112}      # 第一部分的输出（读：avg/<臂>-s<种子>/avg_cal.bin，DONE）
OUT=${RS_OUT:-$PART1/closed-loop}                        # 本部分的输出，放在卡指定的输出目录下
SRC=${RS_SRC:-/root/claude-code-backup-20261028/code/wt/p12-e2e-r1}   # 代码源（分支 p12-e2e-r1）：只用来 git archive 冻结代码
CODE_HEAD=${RS_CODE_HEAD:-}                              # 冻结代码的提交；缺省 = 续跑时沿用 $OUT/code_head.txt，首次 = SRC 的 HEAD
INFER=$D/bin-snapshots/main-97ae04b/uavrgb_infer         # 与 11-10 / 11-11 的 K=1 评测同一个推理二进制
PLUGIN=$D/tmp/build-cf-sitl-lib31/gz                     # SITL 插件，profile cf_sitl_v1:1bc8fedf:lib31（公共任务书：不要动这个目录）
SWA=${RS_SWA:-$D/exp-swa-calib-20261110/ckpt}            # 现成的 cal_swa_s{0,1,2}_k10.bin（只读；只有演练才覆盖）
GPU=${RS_GPU:-GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740}  # 11-11 容器重启后 uav_rgb 用的这张卡（RTX 2080 Ti），用户 11-11 19:23 指定
L_STAR_FILE=$D/exp-action-latency-r2-20261111/L_STAR     # 新协议的动作延迟（R2 选定值）；读出来必须等于 L_EXPECT，不等 rc=5
L_EXPECT=0.20                                            # 任务卡：新协议 = run_r2.sh 选定的 L*
# 冻结的 20 个 val 分区几何（select_val_geoms.py 挑，出处 results/val_geoms.json）。起跑后不改；改了就换新文件名与新输出目录
GEOM_SEEDS=30012,30043,30084,30094,30115,30119,30130,30131,30168,30189,30263,30384,30400,30409,30420,30475,30627,30632,30776,30944
GEOM_SHA256=5058a33b8b0387e812e693c438d4edaec70de6ef357c80aaf19d67a9134a24ab   # GEOM_SEEDS 逗号连接的 sha256
EPISODES=20                                              # 每臂请求集数 = 冻结几何个数
TRAIN_SEEDS="0 1 2"                                      # 训练种子：任务卡
LEVELS="0 m0.5 p0.5"                                     # 档（nat）：0 = 原校准，m0.5 = −0.5，p0.5 = +0.5；顺序即臂的排队顺序（条款 P8）
CORRIDOR=3.0                                             # 横向边界 = scene_gen.Y_RANGE
K1_ARGS="--task base --steps 80"                         # 柱阵 K=1
K1_EPT=1500                                              # 每集墙钟上限（秒）：同 heading-guard 与 R2 实验
LANES=${RS_LANES:-2}                                     # 条款 P7：全程 2 条车道
EVAL=${RS_EVAL:-}                                        # 评测入口；缺省 = 冻结代码里的 eval_closed_loop.py（只有演练才覆盖）
SELFCHECK=${RS_SELFCHECK:-1}                             # 1 = 起跑前核对本脚本与 CODE_HEAD 里的同名文件一致（只有演练才关）
STABILITY_JSON=${RS_STABILITY_JSON:-}                    # 第一部分的 stability.json；缺省 = 冻结副本里的 results/stability.json（没提交就拒绝起跑）
MIN_CORES=${RS_MIN_CORES:-12}                            # ② cgroup 配额下限（核）
IDLE_WINDOW_S=${RS_IDLE_WINDOW_S:-30}                    # 起跑时测一次容器平均核数的窗口（秒），只记录，不作条件
GZ_RECHECK_S=${RS_GZ_RECHECK_S:-60}                      # ④ 复查间隔（秒）
GZ_LOG_S=${RS_GZ_LOG_S:-600}                             # ④ 等待期间每多少秒记一行
GZ_WAIT_MAX_S=${RS_GZ_WAIT_MAX_S:-10800}                 # ④ 最长等 3 小时
LOCK_LOG_S=${RS_LOCK_LOG_S:-600}                         # ③ 等锁期间每多少秒记一行
MACHINE_LOCK=${RS_MACHINE_LOCK:-$D/tmp/phase12-cards/machine.lock}   # 重实验的整机锁（主会话定的惯用法：exec 8>> 锁文件; flock 8；持有到脚本退出）
NVIDIA_SMI=${RS_NVIDIA_SMI:-nvidia-smi}                  # ① 探针（演练时换成桩）
CPU_MAX_FILE=${RS_CPU_MAX_FILE:-/sys/fs/cgroup/cpu.max}  # ② 探针：配额 微秒 周期
CPU_STAT_FILE=${RS_CPU_STAT_FILE:-/sys/fs/cgroup/cpu.stat}   # 每臂记录用：usage_usec、nr_throttled、throttled_usec
GZ_PATTERN=${RS_GZ_PATTERN:-'(^|[ /])gz sim( |$)'}       # ④ 探针：pgrep -f 的模式（匹配 gz sim 进程的命令行）
GZ_PROBE=${RS_GZ_PROBE:-}                                # 非空时用这条命令的第一行输出当 gz sim 进程数（演练用）
# ----------------------------------------------------------------

export UAVRGB_RENDER_BACKEND=software DISPLAY=:20 XDG_RUNTIME_DIR=/tmp/xdg   # 本容器 gz 只能软件渲染，Xorg :20
export CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 TMPDIR=$OUT/tmp   # 根分区只剩 3 GB，临时文件放数据盘
export UAVRGB_CF_SITL_PLUGIN_DIR=$PLUGIN
export OUT CODE=$OUT/code PY=$R/.venv-tools/bin/python CPU_STAT_FILE INFER GEOM_SEEDS CORRIDOR K1_ARGS EPISODES
mkdir -p "$OUT/tmp" /tmp/xdg && chmod 700 /tmp/xdg
SUMM=$CODE/experiments/2026-11-12-recipe-stability/summarize_sweep.py
export SUMM

log() { echo "$(date '+%F %T') $*"; }
export -f log

finish() {   # $1 = rc，$2 = 原因：刷新 summary.md（冻结代码已就位时），写 DONE，退出
  trap - TERM INT
  [[ -f "$SUMM" ]] && "$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 生成，DONE rc=$1" > "$OUT/summarize.log" 2>&1
  echo "rc=$1 $(date '+%F %T') $2" > "$OUT/DONE"
  log "全部结束 rc=$1 $2"
  exit "$1"
}

# 先拿实例锁，再写任何状态文件（run.log、run.pid、DONE、launches.log）
exec 9> "$OUT/.lock"
flock -n 9 || { echo "已有实例在跑（$OUT/.lock），本次退出" >&2; exit 3; }

refuse() {   # 拒绝起跑 = rc=5：原因追加进 refused.txt，DONE 只在还没有时才写（不盖掉上一次的结果）
  echo "$1" >&2
  echo "$(date '+%F %T') $1" >> "$OUT/refused.txt"
  [[ -f "$OUT/DONE" ]] || echo "rc=5 $(date '+%F %T') $1" > "$OUT/DONE"
  exit 5
}
OVERRIDES=$(compgen -v | grep '^RS_' | while read -r v; do echo "$v=${!v}"; done)
if [[ -n "$OVERRIDES" && "${RS_REHEARSAL:-}" != 1 ]]; then
  refuse "正式运行遇到覆盖配置的环境变量，拒绝起跑（演练请设 RS_REHEARSAL=1，覆盖项会逐个记进 launches.log）：$(tr '\n' ' ' <<< "$OVERRIDES")"
fi

exec >> "$OUT/run.log" 2>&1
rm -f "$OUT/DONE"
echo $$ > "$OUT/run.pid"
trap 'finish 130 "收到终止信号（kill 进程组），停在这里；评完的臂保留，同一条命令重起可续跑"' TERM INT
log "开始 PGID=$$ lanes=$LANES gpu=$GPU 机器锁=$MACHINE_LOCK 档=$LEVELS 训练种子=$TRAIN_SEEDS 几何 $EPISODES 个"

# ---------------------------------------------------------------- 起跑条件 ①②（不满足立刻退出）与输入核对
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

geom_sha=$(printf %s "$GEOM_SEEDS" | sha256sum | cut -d' ' -f1)
[[ "$geom_sha" == "$GEOM_SHA256" ]] || refuse "GEOM_SEEDS 的 sha256 $geom_sha 与冻结值 $GEOM_SHA256 不一致：冻结清单被改过，不起跑"
n_geom=$(tr ',' '\n' <<< "$GEOM_SEEDS" | grep -c .)
[[ "$n_geom" == "$EPISODES" ]] || refuse "GEOM_SEEDS 有 $n_geom 个 seed，不等于 EPISODES=$EPISODES，不起跑"
l_star=$(tr -d ' \n' < "$L_STAR_FILE" 2>/dev/null)
[[ "$l_star" == "$L_EXPECT" ]] || refuse "$L_STAR_FILE 读出 '$l_star'，不等于预注册的新协议 L=$L_EXPECT，不起跑（主会话改了协议要换新文件名）"
for s in $TRAIN_SEEDS; do
  [[ -f "$SWA/cal_swa_s${s}_k10.bin" ]] || refuse "输入不存在：$SWA/cal_swa_s${s}_k10.bin"
done
for f in "$INFER" "$PLUGIN/libCfSitlSystem.so"; do [[ -f "$f" ]] || refuse "输入不存在：$f"; done
part1_done=$(cut -d' ' -f1 "$PART1/DONE" 2>/dev/null)
[[ "$part1_done" == rc=0 || "$part1_done" == rc=21 || "${RS_REHEARSAL:-}" == 1 ]] || refuse "第一部分 $PART1/DONE 不是 rc=0 / rc=21（'$part1_done'）：第一部分没跑完，不起跑"

# ---------------------------------------------------------------- 冻结代码（先于等锁：代码问题立刻暴露，冻结的是起跑这一刻的提交）
if [[ -z "$CODE_HEAD" ]]; then
  if [[ -f "$OUT/code_head.txt" ]]; then CODE_HEAD=$(cat "$OUT/code_head.txt"); else CODE_HEAD=$(git -C "$SRC" rev-parse HEAD); fi
fi
CODE_HEAD=$(git -C "$SRC" rev-parse "$CODE_HEAD^{commit}" 2>/dev/null) || finish 4 "CODE_HEAD 在 $SRC 里不是有效提交"
if [[ "$SELFCHECK" == 1 ]]; then
  rel=experiments/2026-11-12-recipe-stability/run_closed_loop.sh
  want=$(git -C "$SRC" show "$CODE_HEAD:$rel" 2>/dev/null | sha256sum | cut -d' ' -f1)
  have=$(sha256sum < "${BASH_SOURCE[0]}" | cut -d' ' -f1)
  [[ "$want" == "$have" ]] || finish 4 "本脚本与 $CODE_HEAD 里的 $rel 不一致（sha256 $have 对 $want）：先提交再起跑"
fi
if [[ ! -f "$CODE/sim/tools/eval_closed_loop.py" ]]; then
  mkdir -p "$CODE" && git -C "$SRC" archive "$CODE_HEAD" | tar -x -C "$CODE" || finish 4 "冻结代码失败"
fi
echo "$CODE_HEAD" > "$OUT/code_head.txt"
bad=$(git -C "$SRC" ls-tree -r "$CODE_HEAD" -- sim tools experiments/2026-11-12-recipe-stability | while read -r _m _t sha path; do
        [[ "$(git hash-object "$CODE/$path" 2>/dev/null)" == "$sha" ]] || echo "$path"; done | head -n 3)
[[ -z "$bad" ]] || finish 4 "冻结副本与 $CODE_HEAD 不一致：$bad"
[[ -f "$SUMM" ]] || finish 4 "冻结副本里没有 summarize_sweep.py（CODE_HEAD $CODE_HEAD 之前的提交？）"
EVAL=${EVAL:-$CODE/sim/tools/eval_closed_loop.py}
export EVAL

# 冻结的几何清单自身核对（分区都是 val、gid 两两不同、与 tools/eval_seeds.py 全部清单不相交）
geom_chk=$("$PY" "$CODE/experiments/2026-11-12-recipe-stability/select_val_geoms.py" --frozen-check "$GEOM_SEEDS" 2>&1); geom_rc=$?
(( geom_rc == 0 )) || refuse "冻结几何清单核对不过：$geom_chk"
log "冻结几何清单核对：$geom_chk"

# 第一部分的结论：配方臂
STAB=${STABILITY_JSON:-$CODE/experiments/2026-11-12-recipe-stability/results/stability.json}
[[ -f "$STAB" ]] || refuse "没有 $STAB：第一部分的结果还没有提交到 $CODE_HEAD（任务卡：第一部分写进报告并提交之后才能起第二部分）"
EXTRA_ARM=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("extra_arm") or "")' "$STAB") || refuse "读不了 $STAB"
if [[ -f "$OUT/config.env" ]]; then   # 续跑沿用起跑时的判定；与现在冻结副本里的不一致说明第一部分的结论被改过
  prev=$(sed -n 's/^EXTRA_ARM=//p' "$OUT/config.env" | head -n1)
  [[ "$prev" == "$EXTRA_ARM" ]] || refuse "配方臂判定与 $OUT/config.env 里起跑时的 '$prev' 不一致（现在 '$EXTRA_ARM'）：续跑沿用起跑时的判定，不起跑"
fi
export EXTRA_ARM
if [[ -n "$EXTRA_ARM" ]]; then
  for s in $TRAIN_SEEDS; do
    [[ -f "$PART1/avg/$EXTRA_ARM-s$s/AVG_OK" && -f "$PART1/avg/$EXTRA_ARM-s$s/avg_cal.bin" ]] || refuse "配方臂 $EXTRA_ARM 的输入不存在或没做完：$PART1/avg/$EXTRA_ARM-s$s/avg_cal.bin"
  done
fi
log "配方臂：${EXTRA_ARM:-无}（来自 $STAB）"

{ echo "code_head $CODE_HEAD"
  echo "gpu_list $(tr '\n' ';' <<< "$gpu_list")"
  echo "geom_seeds $GEOM_SEEDS  sha256 $geom_sha"
  echo "l_star $l_star  extra_arm ${EXTRA_ARM:-无}"
  sha256sum "$CODE/sim/tools/collect.py" "$CODE/sim/tools/eval_closed_loop.py" "$CODE/sim/tools/scene_gen.py" "$CODE/sim/dynamics/e1_mini.json"
  sha256sum "$SWA"/cal_swa_s{0,1,2}_k10.bin "$INFER" "$PLUGIN/libCfSitlSystem.so" "${BASH_SOURCE[0]}"
  [[ -n "$EXTRA_ARM" ]] && sha256sum "$PART1/avg/$EXTRA_ARM"-s{0,1,2}/avg_cal.bin; } > "$OUT/inputs.txt" 2>&1
# 起跑记录只追加不覆盖：HEAD、覆盖项
{ echo "=== $(date '+%F %T') 起跑 PGID=$$ 模式=$([[ "${RS_REHEARSAL:-}" == 1 ]] && echo 演练 || echo 正式)"
  echo "code_head $CODE_HEAD（$SRC，分支 $(git -C "$SRC" rev-parse --abbrev-ref HEAD)）；配方臂 ${EXTRA_ARM:-无}；L=$l_star"
  echo "覆盖项：${OVERRIDES:-无}"; } >> "$OUT/launches.log"
[[ -f "$OUT/config.env" ]] || cat > "$OUT/config.env" <<EOF
# run_closed_loop.sh 起跑配置（summarize_sweep.py 读；续跑沿用这里的 EXTRA_ARM）
SEEDS=$GEOM_SEEDS
L_STAR=$l_star
CODE_HEAD=$CODE_HEAD
EXTRA_ARM=$EXTRA_ARM
GPU=$GPU
LANES=$LANES
EOF

# ---------------------------------------------------------------- 起跑条件 ③ 机器锁，④ 没有别人的 gz sim
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
t_gz=$SECONDS; last_gz_log=-999999; gz_n=$(gz_count)
while (( gz_n > 0 )); do
  waited=$((SECONDS - t_gz))
  if (( waited >= GZ_WAIT_MAX_S )); then
    finish 14 "条件 ④ 等了 $waited s 仍有 gz sim 进程 $gz_n 个（拿到机器锁之后别人还在跑 Gazebo），超过 $GZ_WAIT_MAX_S s，没有起跑"
  fi
  if (( SECONDS - last_gz_log >= GZ_LOG_S )); then
    log "条件 ④ 不满足：拿到锁后仍有 gz sim 进程 $gz_n 个；已等 $waited s，每 $GZ_RECHECK_S s 复查"
    last_gz_log=$SECONDS
  fi
  sleep "$GZ_RECHECK_S"; gz_n=$(gz_count)
done
gz_wait=$((SECONDS - t_gz))
u0=$(usage_usec); w0=$(date +%s.%N); sleep "$IDLE_WINDOW_S"; u1=$(usage_usec); w1=$(date +%s.%N)
idle_cores=$(awk -v a="$u0" -v b="$u1" -v t0="$w0" -v t1="$w1" 'BEGIN{printf "%.2f", (b-a)/1e6/(t1-t0)}')
log "条件 ④ 满足：gz sim 进程 $gz_n 个（等了 $gz_wait s）；$IDLE_WINDOW_S s 窗口内容器平均 $idle_cores 核（只记录，不作条件）"
cat > "$OUT/start.txt" <<EOF
gpu_ok=是
cpu_quota_cores=$cores
min_cores=$MIN_CORES
lock_wait_s=$lock_wait
gz_count=$gz_n
gz_wait_s=$gz_wait
idle_window_s=$IDLE_WINDOW_S
idle_cores=$idle_cores
started_at=$(date '+%F %T')
gpu=$GPU
lanes=$LANES
EOF

# ---------------------------------------------------------------- 档的权重：±0.5 档由 shift_forward_bias 生成（0 档用原文件）
CKPT=$OUT/ckpt
if ! "$PY" "$CODE/experiments/2026-11-12-recipe-stability/make_shifted_ckpts.py" --src-dir "$SWA" --out-dir "$CKPT" --shifts=-0.5,0.5 > "$OUT/make_shifted.log" 2>&1; then
  finish 11 "生成平移权重失败，见 $OUT/make_shifted.log：$(tail -n 3 "$OUT/make_shifted.log" | tr '\n' ' ')"
fi
log "平移权重已生成并逐字节核对：$(grep -c 'sha256' "$OUT/make_shifted.log") 行记录，见 $OUT/make_shifted.log 与 $CKPT/shifts.json"
export CKPT SWA PART1

ckpt_of() {   # $1 = 档 key，$2 = 训练种子 → 权重路径
  case $1 in
    0)    echo "$SWA/cal_swa_s$2_k10.bin" ;;
    m0.5) echo "$CKPT/cal_swa_s$2_k10_fwdm0.5.bin" ;;
    p0.5) echo "$CKPT/cal_swa_s$2_k10_fwdp0.5.bin" ;;
    *) return 1 ;;
  esac
}

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
refresh_summary() {   # 两条车道可能同时刷新：拿不到锁就跳过，下一个臂结束时会再刷
  ( flock -n 7 || exit 0; "$PY" "$SUMM" "$OUT" --note "$(date '+%F %T') 进行中：$1" > "$OUT/summarize.log" 2>&1 ) 7> "$OUT/.summary.lock"
}
export -f cg_snapshot cg_mark cg_brief arm_complete refresh_summary

# 一个评测臂：名字|评测参数串|请求集数|单集墙钟上限|种类（acc|none）。单臂失败不停整个队列，汇总里按未完成标出来。
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
  refresh_summary "$name 结束"
  return 0
}
retry_job() {   # 含迟到步的集重跑一次（条款 P6）：新输出子目录 <臂>-retry1，只含迟到的 seed，原目录保留
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

arm_line() {   # $1 = 臂名，$2 = 权重路径：学生臂，不带 --heading-guard 与 --rtf，新协议 --action-latency L，val 分区冻结几何
  echo "$1|--policy student --ckpt $2 --infer-bin $INFER $K1_ARGS --split val --seeds $GEOM_SEEDS --corridor-half-width $CORRIDOR --action-latency $l_star|$EPISODES|$K1_EPT|acc"
}

# ---------------------------------------------------------------- 排队（条款 P8：按训练种子分组，组内档与配方臂相邻）
log "开始评测：训练种子 $TRAIN_SEEDS × 档 $LEVELS${EXTRA_ARM:+ + 配方臂 $EXTRA_ARM}，每臂 $EPISODES 集，L=$l_star，$LANES 条车道"
{ for s in $TRAIN_SEEDS; do
    for lv in $LEVELS; do arm_line "lvl-$lv-s$s" "$(ckpt_of "$lv" "$s")"; done
    [[ -n "$EXTRA_ARM" ]] && arm_line "x-$EXTRA_ARM-s$s" "$PART1/avg/$EXTRA_ARM-s$s/avg_cal.bin"
  done; } | run_queue

# ---------------------------------------------------------------- 收尾
missing=""
for s in $TRAIN_SEEDS; do
  for lv in $LEVELS; do arm_complete "lvl-$lv-s$s" "$EPISODES" || missing="$missing lvl-$lv-s$s"; done
  [[ -n "$EXTRA_ARM" ]] && { arm_complete "x-$EXTRA_ARM-s$s" "$EPISODES" || missing="$missing x-$EXTRA_ARM-s$s"; }
done
[[ -z "$missing" ]] || finish 11 "有臂没跑完：$missing；读 summary.md，同一条命令重起可续跑"
finish 0 "全部臂跑完（档 $LEVELS${EXTRA_ARM:+，配方臂 $EXTRA_ARM}，L=$l_star）；读 summary.md，判定由主会话定"
