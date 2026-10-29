#!/usr/bin/env bash
# 启动 uav_rgb 训练监控台（本地原生桌面程序，Tkinter）。
# 自动选择带 tkinter 的 Python 3（本机系统 Python 已装 python3-tk，conda Python 亦可用）。
# 参数透传 monitor.py，如：launch_ui.sh --run-dir /path/to/run --refresh 3
set -u
if [ -z "${DISPLAY:-}" ]; then
  echo "launch_ui.sh: 没有 DISPLAY；请在 XFCE 桌面会话（X11）里运行" >&2
  exit 1
fi
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for py in python3 /usr/bin/python3 /opt/conda/bin/python3; do
  command -v "$py" >/dev/null 2>&1 || continue
  if "$py" -c 'import tkinter' >/dev/null 2>&1; then
    exec "$py" "$DIR/monitor.py" "$@"
  fi
done
echo "launch_ui.sh: 找不到带 tkinter 的 Python 3（Ubuntu 可装 python3-tk）" >&2
exit 1
