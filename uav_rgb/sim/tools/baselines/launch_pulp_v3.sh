#!/usr/bin/env bash
# 启动器：PULP-Dronet v3 ResBlock-1.0（320,928 参数）零样本，忠实映射 S=90 °/s、阈值 0.7、转向门槛 12.5°。--infer-bin 只认一个可执行文件、不能带环境变量，所以每个基线一个启动器，环境变量写死在这里。
# STEER_SIGN=+1：train 分区教师帧（10,200 帧，offline_check.py）上，柱在 native 裁剪视场右侧的帧比左侧的帧 steer 高 0.162（按集自助法 95% 区间 [0.129, 0.194]，n=352/297），实测符号与「steer>0=左转」的约定一致。
export BASELINE=pulp_v3 PYTHONDONTWRITEBYTECODE=1 STEER_SIGN=1
exec /root/rivermind-data/uav_rgb/teacher-venv/bin/python "$(cd "$(dirname "$0")" && pwd)/baseline_infer.py" "$@"
