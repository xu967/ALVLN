#!/usr/bin/env bash
# 只准备 Python 数据工具。系统 CUDA/Gazebo 由 doctor 单独检查。
set -euo pipefail
uav_setup_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
uav_setup_python="${UAV_RGB_SYSTEM_PYTHON:-/usr/bin/python3}"
"$uav_setup_python" -m venv "$uav_setup_root/.venv-tools"
# venv 重建会更新 activate，但已安装的 console scripts 仍可能引用备份前的路径。
# 只修复本 venv bin/ 内、指向同名 venv 的 Python shebang，不改包源码或外部文件。
"$uav_setup_python" - "$uav_setup_root/.venv-tools" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
for script in (root / "bin").iterdir():
    if script.is_symlink() or not script.is_file():
        continue
    raw = script.read_bytes()
    first, sep, rest = raw.partition(b"\n")
    if first.startswith(b"#!/"):
        old = Path(first[2:].decode("utf-8", errors="replace"))
        if old.parent.name == "bin" and old.parent.parent.name == root.name:
            if old.name.startswith("python") and old.parent != root / "bin":
                script.write_bytes(b"#!" + str(root / "bin/python").encode() + sep + rest)
                print("relocated:", script.name)
PY
"$uav_setup_root/.venv-tools/bin/python" -m pip install -r "$uav_setup_root/tools/requirements.txt"
source "$uav_setup_root/env.sh"
"$UAV_RGB_PYTHON" "$uav_setup_root/tools/doctor.py"
