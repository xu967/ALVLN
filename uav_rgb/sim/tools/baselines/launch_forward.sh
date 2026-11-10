#!/usr/bin/env bash
# 启动器：盲前进（协议自检，应与 --policy forward 逐步一致）。--infer-bin 只认一个可执行文件、不能带环境变量，所以每个基线一个启动器。
export BASELINE=forward PYTHONDONTWRITEBYTECODE=1
exec /root/rivermind-data/uav_rgb/teacher-venv/bin/python "$(cd "$(dirname "$0")" && pwd)/baseline_infer.py" "$@"
