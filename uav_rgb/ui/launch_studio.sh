#!/usr/bin/env bash
# Launch the native desktop workbench. Source project environment first so DATA_ROOT,
# UAV_RGB_PYTHON and CUDA/build paths match the current uav_rgb installation.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/env.sh"
if [[ -z "${DISPLAY:-}" && -z "${WAYLAND_DISPLAY:-}" ]]; then
  echo "launch_studio.sh: no desktop display found; run this from the Ubuntu desktop session." >&2
  exit 1
fi
for py in /usr/bin/python3 python3; do
  command -v "$py" >/dev/null 2>&1 || continue
  if "$py" -c 'import tkinter' >/dev/null 2>&1; then
    exec "$py" "$ROOT/ui/studio.py" "$@"
  fi
done
echo "launch_studio.sh: Python with tkinter not found. Install Ubuntu package python3-tk." >&2
exit 1
