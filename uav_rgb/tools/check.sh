#!/usr/bin/env bash
# 一键检查；默认不启动仿真、不生成训练数据。失败立即返回非零。
set -euo pipefail
uav_check_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$uav_check_root/env.sh"
uav_check_build="${UAV_RGB_BUILD_DIR:-$uav_check_root/build/check}"

if [[ $# -gt 1 || ($# -eq 1 && "$1" != "--python-only") ]]; then
  echo "用法: tools/check.sh [--python-only]" >&2
  exit 2
fi
# 输入分辨率有两个来源（冻结合同 observation.hpp 与真正的实现 preprocess.py），
# 两边漂过一次；这里对齐，不一致直接红。
uav_check_obs="$(sed -n 's/.*kObsW = \([0-9]*\).*/\1/p' \
  "$uav_check_root/src/include/uavrgb/observation.hpp" | head -1)x$(sed -n \
  's/.*kObsH = \([0-9]*\).*/\1/p' "$uav_check_root/src/include/uavrgb/observation.hpp" | head -1)"
uav_check_pre="$(sed -n 's/.*DEFAULT_SIZE = (\([0-9]*\), \([0-9]*\)).*/\1x\2/p' \
  "$uav_check_root/tools/preprocess.py" | head -1)"
if [[ "$uav_check_obs" != "$uav_check_pre" ]]; then
  echo "输入分辨率不一致: observation.hpp=${uav_check_obs} preprocess.py=${uav_check_pre}" >&2
  exit 1
fi
if [[ "${1:-}" != "--python-only" ]]; then
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/doctor.py" --require-gpu
  cmake -S "$uav_check_root/src" -B "$uav_check_build" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$uav_check_build" --parallel "${UAV_RGB_BUILD_JOBS:-8}"
  ctest --test-dir "$uav_check_build" --output-on-failure

  # UAVR 格式有三份实现：preprocess.py（写）、dataset.cpp（读）、synthetic_dataset.hpp
  # （测试夹具，另一份写）。CTest 只用夹具喂 C++ 读端，真正的 Python 产物没人校验——
  # 这条链路把它接起来，格式漂移（例如头里的 H/W 换位）会在这里红。
  uav_check_fmt="$(mktemp -d)"
  trap 'rm -rf "$uav_check_fmt"' EXIT
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/make_synthetic_episode.py" \
    --out "$uav_check_fmt/ep" >/dev/null
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" --out "$uav_check_fmt/ds" \
    "$uav_check_fmt"/ep/ep1 "$uav_check_fmt"/ep/ep2 "$uav_check_fmt"/ep/ep3 >/dev/null
  g++ -std=c++17 -O2 -Wall -Wextra "$uav_check_root/tools/read_dataset.cpp" \
    -o "$uav_check_fmt/read_dataset"
  "$uav_check_fmt/read_dataset" "$uav_check_fmt/ds/dataset.uavr" --selfcheck >/dev/null
else
  echo "SKIP: 用户指定 --python-only，未验证 C++/CUDA 构建和 GPU 测试。"
fi
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/teacher.py"
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/scene_gen.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/gz_io.py" --self-check
for uav_check_test in "$uav_check_root"/sim/tools/test_*.py; do
  "$UAV_RGB_PYTHON" "$uav_check_test"
done
"$UAV_RGB_PYTHON" "$uav_check_root/ui/test_monitor.py"
"$UAV_RGB_PYTHON" "$uav_check_root/ui/test_trainer_demo.py"
echo "PASS: 所选检查完成；模拟采集、真实训练和学生闭环须独立验收。"
