#!/usr/bin/env bash
# 启动器：DroNet 重标定臂：碰撞阈值取离线校准帧上碰撞概率的中位数（见 experiments/2026-11-10-baselines-zeroshot）、S 取 90 °/s 与 PULP 对齐；两处改动都偏离原文。--infer-bin 只认一个可执行文件、不能带环境变量，所以每个基线一个启动器，环境变量写死在这里。
# P_CRIT=0.9994456：DroNet 在 train 分区 10,200 帧上碰撞概率的中位数 0.99944562（取 7 位小数）；p 几乎处处饱和（p10 0.9968），原文 0.7 刹停等于永远不前进。
# STEER_SIGN=+1：同 launch_dronet.sh，符号判不出，沿用约定。
export BASELINE=dronet PYTHONDONTWRITEBYTECODE=1 STEER_SIGN=1 P_CRIT=0.9994456 YAW_SCALE_DRONET=90
exec /root/rivermind-data/uav_rgb/teacher-venv/bin/python "$(cd "$(dirname "$0")" && pwd)/baseline_infer.py" "$@"
