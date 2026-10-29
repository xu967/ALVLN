#!/usr/bin/env python3
"""当前环境探测；工具存在不等于完成训练或 Gazebo 闭环验证。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def probe(name, argv, required=True):
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        code, detail = proc.returncode, (proc.stdout + proc.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        code, detail = 1, str(exc)
    return {"name": name, "required": required, "command": argv,
            "status": "PASS" if code == 0 else "UNAVAILABLE",
            "exit_code": code, "detail": detail[:4000]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--require-sim", action="store_true")
    args = parser.parse_args()
    py = os.environ.get("UAV_RGB_PYTHON", str(ROOT / ".venv-tools/bin/python"))
    gz_py = os.environ.get("UAVRGB_GZ_PYTHON", "/usr/bin/python3")
    checks = [probe("gpu", ["nvidia-smi", "-L"], args.require_gpu)]
    # nvidia-smi 的成功退出本身不能证明列出了卡。
    if checks[0]["status"] == "PASS" and "GPU " not in checks[0]["detail"]:
        checks[0]["status"] = "UNAVAILABLE"
    for name, command in (("cmake", ["cmake", "--version"]),
                          ("ninja", ["ninja", "--version"]),
                          ("cxx", ["g++", "--version"]),
                          ("cuda_toolkit", ["nvcc", "--version"])):
        checks.append(probe(name, command))
    checks.append(probe("data_python", [py, "-c",
        "import sys,numpy,PIL; print(sys.executable); "
        "print('numpy',numpy.__version__,'Pillow',PIL.__version__) "]))
    checks.append(probe("gazebo", ["gz", "sim", "--versions"], args.require_sim))
    checks.append(probe("gazebo_python", [gz_py, "-I", "-c",
        "from gz.transport13 import Node; from gz.msgs10.twist_pb2 import Twist; "
        "print('gz.transport13 / gz.msgs10 imports OK')"], args.require_sim))
    checks.append(probe("tkinter", [gz_py, "-c",
        "import tkinter; print(tkinter.TkVersion)"], False))
    data_root = Path(os.environ.get("DATA_ROOT", "/root/rivermind-data")).expanduser().resolve()
    # 数据根不能在整个 code 树里；不为检查而创建数据目录。
    outside_code = not data_root.is_relative_to(ROOT.parent)
    checks.append({"name": "data_root", "required": True,
                   "status": "PASS" if outside_code else "UNAVAILABLE",
                   "detail": str(data_root), "exists": data_root.is_dir()})
    failed = any(c["required"] and c["status"] != "PASS" for c in checks)
    print(json.dumps({"schema": "uavrgb.environment.v1", "root": str(ROOT),
        "mode": "GPU" if checks[0]["status"] == "PASS" else "NO_GPU",
        "result": "FAIL" if failed else "PASS", "checks": checks,
        "scope": "只验证命令及导入；未验证 CUDA 算子、相机渲染或学生闭环。",
        "display": os.environ.get("DISPLAY")}, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
