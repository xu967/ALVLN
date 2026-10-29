#!/usr/bin/env bash
# uav_rgb 环境变量 —— 幂等：重复 source 不会重复追加 PATH/LD_LIBRARY_PATH
# 适配本机实际环境: Turing (sm_75) + CUDA Toolkit 12.8
# 不要改成 CUDA 13.1: 驱动 580.119.02 最高支持 CUDA 13.0 API，且 Turing 为 sm_75

# 仓库根目录：从脚本自身位置推导，避免硬编码 $HOME/code（本机实际路径与历史不同）
export UAV_RGB_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_HOME="/usr/local/cuda-12.8"
export CUDA_PATH="$CUDA_HOME"
export DATA_ROOT="${DATA_ROOT:-/root/rivermind-data}"
# 直接使用解释器路径，避免备份后 activate / pip 中的旧绝对路径。
export UAV_RGB_PYTHON="${UAV_RGB_PYTHON:-$UAV_RGB_ROOT/.venv-tools/bin/python}"
export UAVRGB_GZ_PYTHON="${UAVRGB_GZ_PYTHON:-/usr/bin/python3}"

# 幂等追加函数：只有目标目录尚未出现时才前置
__uav_rgb_prepend() {
  local var_name="$1" dir="$2"
  # 必须用 ${!var_name-}：set -u 下引用未定义变量会直接退出 shell，
  # 而 LD_LIBRARY_PATH 在干净容器/CI 里通常就是未定义的。
  case ":${!var_name-}:" in
    *":$dir:"*) ;;
    *) export "${var_name}=$dir${!var_name-:+:${!var_name-}}" ;;
  esac
}
__uav_rgb_prepend PATH "$CUDA_HOME/bin"
__uav_rgb_prepend LD_LIBRARY_PATH "$CUDA_HOME/lib64"
unset -f __uav_rgb_prepend

export CUDNN_HOME=""           # 本环境未安装 cuDNN
