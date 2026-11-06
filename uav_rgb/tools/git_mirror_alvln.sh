#!/usr/bin/env bash
# 把 uav_rgb 仓库的提交逐个镜像到 ALVLN 仓库的 uav_rgb/ 子目录并推送 GitHub。
# 一个 uav_rgb 提交 = 一个 ALVLN 提交（同一条提交说明），GitHub 上能按主题看 diff。
# 用法：bash tools/git_mirror_alvln.sh            # 镜像 上次镜像点..HEAD 并推送
#       PUSH=0 bash tools/git_mirror_alvln.sh     # 只提交不推送
# 只镜像「已提交」的内容；工作树里未提交的改动不会被带过去。
set -euo pipefail

# ───── 配置区 ─────
SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"   # uav_rgb 仓库根
DST="${ALVLN_DIR:-$SRC/../ALVLN}"                              # ALVLN 完整克隆（远端 git@github.com:xu967/ALVLN.git）
SUBDIR="uav_rgb"                                               # ALVLN 里的子目录名
MARK="$SUBDIR/.mirror_src_commit"                              # 记录最近镜像到的 uav_rgb 提交哈希，提交进 ALVLN
BRANCH="${ALVLN_BRANCH:-main}"
PUSH="${PUSH:-1}"                                              # 1 推送；0 只提交（想先看 git log 再推时用）
# ──────────────────

cd "$DST"
[[ -z "$(git status --short)" ]] || { echo "ALVLN 工作树不干净，先处理：$(git status --short | head -3)" >&2; exit 1; }
git fetch -q origin "$BRANCH"
# 本地必须包含远端（能快进推送）；本地领先几个未推的提交没关系，落后就先 pull
git merge-base --is-ancestor "origin/$BRANCH" HEAD || { echo "ALVLN 本地 $BRANCH 落后于 origin，先 git pull --ff-only" >&2; exit 1; }

last="$(cat "$MARK" 2>/dev/null || true)"
if [[ -z "$last" ]]; then
  # 首次：以 ALVLN 里 uav_rgb/ 当前内容对应的 uav_rgb 提交为起点（由人确认后写入 MARK）
  echo "缺 $MARK；请先 echo <uav_rgb提交哈希> > $DST/$MARK 并提交" >&2; exit 1
fi
mapfile -t commits < <(git -C "$SRC" rev-list --reverse "${last}..HEAD")
[[ ${#commits[@]} -gt 0 ]] || { echo "没有新提交需要镜像（$last 已是 HEAD）"; exit 0; }

for c in "${commits[@]}"; do
  git rm -rq --ignore-unmatch "$SUBDIR"
  mkdir -p "$SUBDIR"
  git -C "$SRC" archive "$c" | tar -x -C "$SUBDIR"
  echo "$c" > "$MARK"
  git add -A "$SUBDIR"
  subject="$(git -C "$SRC" log -1 --format=%s "$c")"
  body="$(git -C "$SRC" log -1 --format=%b "$c")"
  git commit -q -m "$subject" -m "$body" -m "uav_rgb 源提交：$c"
  echo "镜像 $c → $(git rev-parse --short HEAD)  $subject"
done
if [[ "$PUSH" == "1" ]]; then
  git push origin "$BRANCH"
  git status --short --branch | head -1
fi
