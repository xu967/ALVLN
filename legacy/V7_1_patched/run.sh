#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
  echo '[ERR][PYTHON_NOT_FOUND] 未找到 python3。请先安装 Python 3。' >&2
  exit 127
fi
if [[ "${1:-}" == "sim" ]]; then
  shift
  exec python3 "$ROOT/tools/sim_preflight.py" "$@"
fi
exec python3 "$ROOT/tools/launcher.py" "$@"
