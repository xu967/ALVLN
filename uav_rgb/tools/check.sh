#!/usr/bin/env bash
# 一键检查；默认不启动仿真、不生成训练数据。失败立即返回非零。
set -euo pipefail
uav_check_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$uav_check_root/env.sh"
# 字节码与临时目录不写根分区（只剩几 GB；E-13）：TMPDIR 已设就照用，没设且数据盘的 tmp 在就用它，都没有沿用系统缺省。
export PYTHONDONTWRITEBYTECODE=1
if [[ -z "${TMPDIR:-}" && -d "$DATA_ROOT/uav_rgb/tmp" ]]; then export TMPDIR="$DATA_ROOT/uav_rgb/tmp"; fi
uav_check_build="${UAV_RGB_BUILD_DIR:-$uav_check_root/build/check}"
uav_check_skipped=()   # 被 SKIP 的段；末行 PASS 要把它们列出来，免得「没检查」被读成「检查过了」（R2-REPRO 2026-11-07）

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
# 文本长度/词表同样是手抄两份（text.hpp vs preprocess.py/instruction.py）；
# 漂了要到真实打包或训练才炸，这里先红。
uav_check_tlen_c="$(sed -n 's/.*kTextLen = \([0-9]*\).*/\1/p' \
  "$uav_check_root/src/include/uavrgb/text.hpp" | head -1)"
uav_check_tlen_py="$(sed -n 's/^TEXT_LEN = \([0-9]*\).*/\1/p' \
  "$uav_check_root/tools/preprocess.py" | head -1)"
uav_check_vocab_c="$(sed -n 's/.*kTextVocab = \([0-9]*\).*/\1/p' \
  "$uav_check_root/src/include/uavrgb/text.hpp" | head -1)"
uav_check_vocab_py="$("$UAV_RGB_PYTHON" -c 'import sys; sys.path.insert(0, sys.argv[1]); import instruction; print(len(instruction.VOCAB))' \
  "$uav_check_root/sim/tools")"
if [[ "$uav_check_tlen_c" != "$uav_check_tlen_py" || "$uav_check_vocab_c" != "$uav_check_vocab_py" ]]; then
  echo "文本常量不一致: text.hpp len=${uav_check_tlen_c} vocab=${uav_check_vocab_c}" \
       "preprocess.py len=${uav_check_tlen_py} instruction.VOCAB=${uav_check_vocab_py}" >&2
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
  # --no-tests=error：注册表为空时 ctest 默认 rc=0，会让「没跑测试」显示成 PASS。
  ctest --test-dir "$uav_check_build" --output-on-failure --no-tests=error
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/test_join_soft_targets.py" \
    --native-check "$uav_check_build/test_soft_targets_identity"

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

  # cf_sitl（固件在环 E1 MINI）：固件源码在仓库外（CF_DRONE_DIR，默认仓库旁的 ../CF-Drone），
  # 没有就 SKIP 而不是红。构建目录跟 UAV_RGB_BUILD_DIR 走（根分区装不下，别用仓库内 build/）。
  uav_check_cfd="${CF_DRONE_DIR:-$uav_check_root/../CF-Drone}"
  if [[ -f "$uav_check_cfd/CF-Drone.ino" && -f "$uav_check_root/sim/cf_sitl/CMakeLists.txt" ]]; then
    uav_check_sitl="${UAV_RGB_SITL_BUILD_DIR:-${uav_check_build}-cf-sitl}"
    cmake -S "$uav_check_root/sim/cf_sitl" -B "$uav_check_sitl" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCF_DRONE_DIR="$uav_check_cfd"
    cmake --build "$uav_check_sitl" --parallel "${UAV_RGB_BUILD_JOBS:-8}"
    ctest --test-dir "$uav_check_sitl" --output-on-failure --no-tests=error
    # 自检退出码 = 失败场景数；输出（TXT/CSV）落在构建目录，画图另跑 sim/cf_sitl/selftest/plot.py
    "$uav_check_sitl/cf_sitl_selftest" --params "$uav_check_root/sim/dynamics/e1_mini.json" \
      --out "$uav_check_sitl/selftest-out"
  else
    echo "SKIP: cf_sitl 未检查（CF_DRONE_DIR=$uav_check_cfd 没有 CF-Drone.ino，或 sim/cf_sitl/CMakeLists.txt 缺失；" \
         "固件来源见 README「环境与一键检查」）" >&2
    uav_check_skipped+=("cf_sitl（固件在环库、CTest、selftest）")
  fi
else
  echo "SKIP: 用户指定 --python-only，未验证 C++/CUDA 构建和 GPU 测试。" >&2
  uav_check_skipped+=("C++/CUDA 构建、CTest、GPU 测试、cf_sitl")
  "$UAV_RGB_PYTHON" "$uav_check_root/tools/test_join_soft_targets.py"
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
  # 这道防线有两层，各配一个负例，并断言报错来自它要测的那一层（只看退出码，另一层的报错会顶替它）：
  #   a) load_episode 的 split 复算门：默认策略下 split 必须等于 gid 按记录策略应属的 split；
  #   b) scan 的 gid 唯一性：诊断策略（自带 split）跳过复算门，同一个几何出现在两个 split 仍要被拒。
  "$UAV_RGB_PYTHON" - "$uav_check_fmt/ep" <<'PY'
import json, sys
root = sys.argv[1]
for name in ("ep1", "ep3"):
    path = f"{root}/{name}/episode_manifest.json"
    man = json.load(open(path, encoding="utf-8"))
    man["geometry_gid"] = "check.shared_geometry"
    json.dump(man, open(path, "w", encoding="utf-8"))
PY
  if uav_check_err="$("$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" \
       --out "$uav_check_fmt/ds_leak" "$uav_check_fmt/ep/ep1" "$uav_check_fmt/ep/ep3" 2>&1 >/dev/null)"; then
    echo "几何 split 门禁未生效：同 geometry_gid 跨 train/val 被接受" >&2
    exit 1
  fi
  [[ "$uav_check_err" == *"按记录策略应属的 split"* ]] || {
    echo "几何 split 负例 a) 没有被 split 复算门挡下，报错是：$uav_check_err" >&2
    exit 1
  }
  "$UAV_RGB_PYTHON" - "$uav_check_fmt/ep" <<'PY'
import json, sys
root = sys.argv[1]
for name in ("ep1", "ep3"):
    path = f"{root}/{name}/episode_manifest.json"
    man = json.load(open(path, encoding="utf-8"))
    man["split_policy"] = "diag-manual-split"
    json.dump(man, open(path, "w", encoding="utf-8"))
PY
  if uav_check_err="$("$UAV_RGB_PYTHON" "$uav_check_root/tools/preprocess.py" --allow-diagnostic-split \
       --out "$uav_check_fmt/ds_leak_diag" "$uav_check_fmt/ep/ep1" "$uav_check_fmt/ep/ep3" 2>&1 >/dev/null)"; then
    echo "几何 split 门禁未生效：诊断策略下同 geometry_gid 跨 train/val 被接受" >&2
    exit 1
  fi
  # 「同几何必须同 split」是 scan() 这一句独有的；build_meta 里还有一句「几何组 … 跨 split」，也含「几何组」，不能当判据。
  [[ "$uav_check_err" == *"同几何必须同 split"* ]] || {
    echo "几何 split 负例 b) 没有走到 scan() 的「同一个几何出现在两个 split」那一道，报错是：$uav_check_err" >&2
    exit 1
  }

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
echo "PASS: preprocess 教师标签/几何隔离/重复目录/split 门禁 负例"
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/teacher.py"
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/scene_gen.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/scene_course.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/sim/tools/gz_io.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/tools/airsim_to_uavr.py" --self-check
"$UAV_RGB_PYTHON" "$uav_check_root/tools/baseline_priors.py" --self-check
for uav_check_test in "$uav_check_root"/sim/tools/test_*.py; do
  "$UAV_RGB_PYTHON" "$uav_check_test"
done
# tools/ 下的纯 Python 测试（打包侧车/深度分带、seed 独立性、统计公式、3DGS 工具的 mock 路径）之前没进门禁
# （REVIEW-P2 #4 / R2-SRC Should fix #2）：P2 的全部打包侧测试只靠人手跑。test_train_student_with_video 在下面单列。
for uav_check_test in "$uav_check_root"/tools/test_*.py; do
  [[ "$uav_check_test" == */test_join_soft_targets.py || "$uav_check_test" == */test_train_student_with_video.py ]] && continue
  "$UAV_RGB_PYTHON" "$uav_check_test"
done
# sim/3dgs/test_*.py（mock 渲染器、纯 CPU，不起 Gazebo）要 3dgs-venv 的解释器，.venv-tools 不是同一套依赖。
# 文件个数不写死（新增的 test_*.py 自动进来）；解释器不在就 SKIP 并让末行 PASS 列出，免得「没检查」被读成「检查过了」。
# --python-only 也跑这一段。UAV_RGB_3DGS_PYTHON 可换解释器路径（也用来演示 SKIP 路径）；建 3dgs-venv 见 docs/HANDOFF_3DGS_20261105.md。
uav_check_3dgs_py="${UAV_RGB_3DGS_PYTHON:-$DATA_ROOT/uav_rgb/3dgs-venv/bin/python}"
if [[ -x "$uav_check_3dgs_py" ]]; then
  uav_check_3dgs_n=0
  for uav_check_test in "$uav_check_root"/sim/3dgs/test_*.py; do
    echo "3DGS 测试: ${uav_check_test#"$uav_check_root"/}"
    "$uav_check_3dgs_py" "$uav_check_test"
    uav_check_3dgs_n=$((uav_check_3dgs_n + 1))
  done
  echo "PASS: sim/3dgs 测试 ${uav_check_3dgs_n} 个文件（$uav_check_3dgs_py）"
else
  echo "SKIP: 3DGS 测试未检查（$uav_check_3dgs_py 不存在或不可执行；sim/3dgs/test_*.py 要 3dgs-venv，" \
       "建法见 docs/HANDOFF_3DGS_20261105.md）" >&2
  uav_check_skipped+=("3DGS 测试")
fi
"$UAV_RGB_PYTHON" "$uav_check_root/ui/test_monitor.py"
"$UAV_RGB_PYTHON" "$uav_check_root/ui/test_trainer_demo.py"
"$UAV_RGB_PYTHON" "$uav_check_root/ui/test_studio.py"
"$UAV_RGB_PYTHON" "$uav_check_root/tools/test_train_student_with_video.py"
if [[ ${#uav_check_skipped[@]} -gt 0 ]]; then
  echo "PASS（有 SKIP）：所选检查完成，但未检查：$(IFS=';'; echo "${uav_check_skipped[*]}")；模拟采集、真实训练和学生闭环须独立验收。"
else
  echo "PASS: 所选检查完成；模拟采集、真实训练和学生闭环须独立验收。"
fi
