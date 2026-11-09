#!/bin/bash
# sweep_seeds.sh：cf_sitl_selftest 多种子扫描。每个种子独立输出目录，末尾汇总每种子 PASS/FAIL 与失败项，退出码 = 有失败项的种子数。
# 为什么要扫：33 项判据里场景 1（加计链）、6a（SAFE 漂移）、6b（积分饱和）的阈值是统计量，只看默认 seed 1 的 33/33 会漏掉
# 换种子才红的判据（R2-SITL 2026-11-07：改阈值前 40 种子 11 红）。改任何阈值 / 执行器参数后跑一遍，1..40 全绿才算过。
# 用法：sim/cf_sitl/selftest/sweep_seeds.sh [构建目录] [输出目录]
#       SEEDS="$(seq 1 80)" JOBS=8 sim/cf_sitl/selftest/sweep_seeds.sh /path/build
set -u
# ---- 配置区 ----
SEEDS=${SEEDS:-$(seq -s ' ' 1 40)}   # 40 个种子：三类统计判据各 5–10% 的误报率在 40 个里都会露头（R2-SITL 实测 5/40、4/40、3/40）
BUILD=${BUILD:-${1:?用法: sweep_seeds.sh <构建目录> [输出目录]（含 cf_sitl_selftest 的构建目录，README「构建」；本机是 \$DATA_ROOT/uav_rgb/tmp/build-cf-sitl）}}
OUT=${OUT:-${2:-$BUILD/selftest-seeds}}                                 # 每种子一个 s<N>/（selftest.txt、CSV、log.txt），落数据盘
PARAMS=${PARAMS:-$(cd "$(dirname "$0")/../../.." && pwd)/sim/dynamics/e1_mini.json}   # 按脚本位置定位本仓库的参数文件
JOBS=${JOBS:-4}                      # 并行度：每种子单线程约 2 s，场景 4 再起 2 个子进程；4 路 40 种子约 25 s，别占满 12 核的 cgroup

# ---- 执行 ----
SELFTEST=$BUILD/cf_sitl_selftest
if [ ! -x "$SELFTEST" ]; then echo "找不到 $SELFTEST：先按 sim/cf_sitl/README.md 构建" >&2; exit 2; fi
mkdir -p "$OUT"
run_one() {   # $1 = seed；log.txt 末行记退出码（= 失败项数）
	local d="$2/s$1"
	mkdir -p "$d"
	"$3" --params "$4" --out "$d" --seed "$1" > "$d/log.txt" 2>&1
	echo "rc=$?" >> "$d/log.txt"
}
export -f run_one
printf '%s\n' $SEEDS | xargs -P "$JOBS" -I{} bash -c 'run_one "$@"' _ {} "$OUT" "$SELFTEST" "$PARAMS"

# ---- 汇总 ----
n=0; fail=0
for s in $SEEDS; do
	n=$((n + 1))
	log="$OUT/s$s/log.txt"
	rc=$(tail -n 1 "$log" | sed -n 's/^rc=//p')
	summary=$(grep '^== 结果' "$log")
	if [ "${rc:-99}" = "0" ]; then
		echo "seed=$s PASS  $summary"
	else
		fail=$((fail + 1))
		echo "seed=$s FAIL  ${summary:-（没跑完，rc=${rc:-?}）}"
		grep '\[FAIL\]' "$log" | sed 's/^ */    /'
	fi
done
echo "== $n 个种子，$fail 个有失败项（输出在 $OUT）"
exit $fail
