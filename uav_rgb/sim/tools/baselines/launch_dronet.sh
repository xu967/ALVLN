#!/usr/bin/env bash
# 启动器：DroNet 原版 ResNet-8（约 32 万参数）零样本，忠实映射 S=20 °/s、阈值 0.7、转向门槛 12.5°。--infer-bin 只认一个可执行文件、不能带环境变量，所以每个基线一个启动器，环境变量写死在这里。
# STEER_SIGN=+1 只是沿用约定，不是实测结论：native 裁剪视场内左右分组的 steer 均值差 -0.0002（95% 区间 [-0.064, 0.073]），符号判不出；
# steer 均值 0.313 且镜像输入不翻号（只有 12% 的帧变号），DroNet 在这批画面上没有有效的转向信号。
export BASELINE=dronet PYTHONDONTWRITEBYTECODE=1 STEER_SIGN=1
exec /root/rivermind-data/uav_rgb/teacher-venv/bin/python "$(cd "$(dirname "$0")" && pwd)/baseline_infer.py" "$@"
