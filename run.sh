#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
CMD="${1:-help}"
if (( $# )); then shift; fi
case "$CMD" in
  ui|serve) exec python3 -m server.app --host "${NAVLAB_HOST:-127.0.0.1}" --port "${NAVLAB_PORT:-8765}" "$@" ;;
  doctor) exec python3 tools/doctor.py "$@" ;;
  test) exec python3 -m unittest discover -s tests -p 'test_*.py' -v ;;
  cpp-test) cmake -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release && cmake --build build/cpp -j2 && ctest --test-dir build/cpp --output-on-failure ;;
  scene) exec python3 tools/scene_generator.py "$@" ;;
  sim-probe) exec python3 tools/gazebo_probe.py "$@" ;;
  train|eval|sim|board|uart) echo "[BLOCKED][PROFILE_NOT_VERIFIED] $CMD requires a registered, tested backend. Read docs/00_FINAL_DECISIONS.md and config/registry.example.json" >&2; exit 3 ;;
  help|--help|-h) cat <<'HELP'
TinyAerialVLN Final V8 -- server bootstrap (no fictitious training or simulation)
  ./run.sh doctor           Environment / driver / Gazebo / toolchain status
  ./run.sh test             Self tests (stdlib, no GPU required)
  ./run.sh cpp-test         C++17 FP64 linear/softmax gradcheck, not drone training
  ./run.sh scene --seed 42 --difficulty S1  Generate RGB-ready semantic SDF scene/task (no drone yet)
  ./run.sh sim-probe        Check Gazebo command availability, no flight claimed
  ./run.sh ui              NavLab Studio + same-origin API at http://127.0.0.1:8765
  ./run.sh train            Explicitly BLOCKED until real C++ trainer registered
  ./run.sh sim              Explicitly BLOCKED until physics and evaluator registered
See START_HERE.md for server deployment and safety caveats.
HELP
  ;;
  *) echo "[ERR][UNKNOWN_COMMAND] $CMD" >&2; exit 2 ;;
esac
