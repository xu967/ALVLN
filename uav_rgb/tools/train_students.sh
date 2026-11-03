#!/usr/bin/env bash
# 学生训练驱动：一组失败不影响其余组，每阶段状态写进 <out>/run_status.json。
# 存在理由见 docs/worklog/2026-11-03-distill-round-retrospective.md：
# 旧的 /tmp 脚本用 set -eu 顺序跑，a50 失败后 a80 根本没启动，等待器也没有失败分支，
# 白等 5 h 41 min。判完成请读 run_status.json，不要用 pgrep/sleep 猜。
#
# 用法：
#   tools/train_students.sh --data ds/dataset.uavr --out runs/student-1000 \
#       --groups base,a0.5,a0.8 [--soft ds/soft_targets.bin] -- --text-dim 16 --epochs 20 ...
# 组名 base 走无权 CE；a<alpha>（如 a0.5）加 --soft-targets + --distill-alpha。
set -u
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$root/env.sh"
bin="${UAV_RGB_BUILD_DIR:-$root/build/check}/uavrgb_train"

data="" out="" soft="" groups="base"
while [[ $# -gt 0 ]]; do
  case $1 in
    --data) data="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    --soft) soft="$2"; shift 2 ;;
    --groups) groups="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
  esac
done
[[ -n $data && -n $out ]] || { echo "用法见文件头" >&2; exit 2; }
[[ -x $bin ]] || { echo "找不到训练器 $bin（先跑 tools/check.sh 或设 UAV_RGB_BUILD_DIR）" >&2; exit 2; }

# 显存预检：被别人占着的卡不要抢，抢输了就是 2026-11-03 那次 OOM。
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  (( used <= 1024 )) && break
  echo "GPU 已占用 ${used} MiB，等 10 s（最多 10 min）" >&2
  sleep 10
done
(( ${used:-0} <= 1024 )) || { echo "GPU 仍被占用 ${used} MiB，放弃启动" >&2; exit 1; }

mkdir -p "$out"
status="$out/run_status.json"
record() {  # stage status exit_code
  local entry
  entry=$(jq -nc --arg s "$1" --arg st "$2" --argjson c "$3" --arg t "$(date '+%F %T')" \
    '{stage:$s,status:$st,exit_code:$c,ended:$t}')
  if [[ -f $status ]]; then
    jq -c --argjson e "$entry" '. + [$e]' "$status" > "$status.tmp" && mv "$status.tmp" "$status"
  else
    jq -nc --argjson e "$entry" '[$e]' > "$status"
  fi
}

failed=0
for g in ${groups//,/ }; do
  dir="$out/student-$g"
  if [[ -e $dir ]]; then
    echo "跳过 $g：$dir 已存在（不覆盖既有 run）" >&2
    record "$g" skipped 0
    continue
  fi
  args=(--data "$data" --out "$dir")
  if [[ $g != base ]]; then
    [[ -n $soft ]] || { echo "--groups 含 $g 但没给 --soft" >&2; exit 2; }
    args+=(--soft-targets "$soft" --distill-alpha "${g#a}")
  fi
  args+=("$@")
  echo "[$(date '+%F %T')] 开始 $g"
  # 不 set -e：单组失败继续下一组。
  if "$bin" "${args[@]}" > "$out/$g.log" 2>&1; then
    record "$g" ok 0
  else
    code=$?
    record "$g" failed "$code"
    failed=1
    echo "=== $g 失败（exit $code），现场：" >&2
    tail -5 "$out/$g.log" >&2
    nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv >&2
  fi
done
echo "[$(date '+%F %T')] 结束，状态见 $status"
cat "$status"
exit "$failed"
