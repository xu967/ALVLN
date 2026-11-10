#!/usr/bin/env bash
# 启动器：Tiny-PULP-Dronet v3 dw-pw-0.125（2,784 参数）零样本，忠实映射同 pulp_v3。--infer-bin 只认一个可执行文件、不能带环境变量，所以每个基线一个启动器，环境变量写死在这里。
# STEER_SIGN=+1：同上口径，柱在 native 裁剪视场右侧的帧比左侧的帧 steer 高 0.193（95% 区间 [0.176, 0.211]，n=352/297）。
export BASELINE=pulp_tiny PYTHONDONTWRITEBYTECODE=1 STEER_SIGN=1
exec /root/rivermind-data/uav_rgb/teacher-venv/bin/python "$(cd "$(dirname "$0")" && pwd)/baseline_infer.py" "$@"
