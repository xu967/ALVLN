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
# 格式/门禁的夹具与负例都是纯 Python，--python-only 也要跑；C++ 读端仍在下面。
uav_check_fmt="$(mktemp -d)"
trap 'rm -rf "$uav_check_fmt"' EXIT
"$UAV_RGB_PYTHON" "$uav_check_root/tools/make_synthetic_episode.py" \
  --out "$uav_check_fmt/ep" >/dev/null

if [[ "${1:-}" != "--python-only" ]]; then
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/doctor.py" --require-gpu
  cmake -S "$uav_check_root/src" -B "$uav_check_build" -G Ninja -DCMAKE_BUILD_TYPE=Release
  cmake --build "$uav_check_build" --parallel "${UAV_RGB_BUILD_JOBS:-8}"
  ctest --test-dir "$uav_check_build" --output-on-failure

  # UAVR 格式有三份实现：preprocess.py（写）、dataset.cpp（读）、synthetic_dataset.hpp
  # （测试夹具，另一份写）。CTest 只用夹具喂 C++ 读端，真正的 Python 产物没人校验——
  # 这条链路把它接起来，格式漂移（例如头里的 H/W 换位）会在这里红。
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" --out "$uav_check_fmt/ds" \
    "$uav_check_fmt"/ep/ep1 "$uav_check_fmt"/ep/ep2 "$uav_check_fmt"/ep/ep3 >/dev/null
  # L0：带 instruction 的集必须写出旁路侧车，否则文本通路静默丢失。
  [[ -s "$uav_check_fmt/ds/instructions.jsonl" ]] || {
    echo "缺 instructions.jsonl（L0 侧车未写出）" >&2; exit 1; }
  [[ "$(wc -l < "$uav_check_fmt/ds/instructions.jsonl")" -eq 3 ]] || {
    echo "instructions.jsonl 行数 != 3" >&2; exit 1; }
  g++ -std=c++17 -O2 -Wall -Wextra "$uav_check_root/tools/read_dataset.cpp" \
    -o "$uav_check_fmt/read_dataset"
  "$uav_check_fmt/read_dataset" "$uav_check_fmt/ds/dataset.uavr" --selfcheck >/dev/null
else
  echo "SKIP: 用户指定 --python-only，未验证 C++/CUDA 构建和 GPU 测试。"
fi

# 以下纯 Python 打包门禁负例在 --python-only 下也执行。
  # 盲策略/学生自采的标签不是监督信号，必须在打包关口被拒绝。
  "$UAV_RGB_PYTHON" - "$uav_check_fmt/ep" <<'PY'
import json, sys
path = f"{sys.argv[1]}/ep2/episode_manifest.json"
man = json.load(open(path, encoding="utf-8"))
man["teacher"] = "student"
json.dump(man, open(path, "w", encoding="utf-8"))
PY
  if "$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" \
       --out "$uav_check_fmt/ds_student" "$uav_check_fmt/ep/ep2" >/dev/null 2>&1; then
    echo "教师标签门禁未生效：teacher=student 被当训练数据接受" >&2
    exit 1
  fi

  # 同几何（同 geometry_gid）跨 split 必须在打包关口被拒绝；只查 scene_id 挡不住。
  "$UAV_RGB_PYTHON" - "$uav_check_fmt/ep" <<'PY'
import json, sys
root = sys.argv[1]
for name in ("ep1", "ep3"):
    path = f"{root}/{name}/episode_manifest.json"
    man = json.load(open(path, encoding="utf-8"))
    man["geometry_gid"] = "check.shared_geometry"
    json.dump(man, open(path, "w", encoding="utf-8"))
PY
  if "$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" \
       --out "$uav_check_fmt/ds_leak" "$uav_check_fmt/ep/ep1" "$uav_check_fmt/ep/ep3" \
       >/dev/null 2>&1; then
    echo "几何 split 门禁未生效：同 geometry_gid 跨 train/val 被接受" >&2
    exit 1
  fi

  # 单集包内没有对照集可查「同 gid 跨 split」；必须直接校验 split == split_for_gid(gid)。
  # 同一 episode 目录被传两次也不能静默双打包（同一批样本会被算两遍）。
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/make_synthetic_episode.py" \
    --out "$uav_check_fmt/ep_gate" >/dev/null
  if "$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" \
       --out "$uav_check_fmt/ds_dup" "$uav_check_fmt/ep_gate/ep1" "$uav_check_fmt/ep_gate/ep1" \
       >/dev/null 2>&1; then
    echo "同一 episode 目录被重复打包未报错" >&2
    exit 1
  fi
  "$UAV_RGB_PYTHON" - "$uav_check_fmt/ep_gate" "$uav_check_root/sim/tools" <<'PY'
import json, sys
root, sm_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, sm_dir)
import split_manifest as sm
path = root + "/ep2/episode_manifest.json"
man = json.load(open(path, encoding="utf-8"))
gid = "check.split_mismatch"
man["geometry_gid"] = gid
man["split"] = next(s for s in ("train", "val", "test") if s != sm.split_for_gid(gid))
json.dump(man, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
PY
  if "$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" \
       --out "$uav_check_fmt/ds_mismatch" "$uav_check_fmt/ep_gate/ep2" >/dev/null 2>&1; then
    echo "split 与几何 gid 不符被接受：防泄漏门未生效" >&2
    exit 1
  fi
echo "PASS: preprocess 教师标签/几何隔离/重复目录/split 门禁负例"
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/teacher.py"
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/scene_gen.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/gz_io.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/tools/airsim_to_uavr.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/tools/baseline_priors.py" --self-check
for uav_check_test in "$uav_check_root"/sim/tools/test_*.py; do
  "$UAV_RGB_PYTHON" "$uav_check_test"
done
"$UAV_RGB_PYTHON" "$uav_check_root/ui/test_monitor.py"
"$UAV_RGB_PYTHON" "$uav_check_root/ui/test_trainer_demo.py"
echo "PASS: 所选检查完成；模拟采集、真实训练和学生闭环须独立验收。"
