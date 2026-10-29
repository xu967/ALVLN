#!/usr/bin/env bash
# Explicitly opt-in. Never changes GPU drivers, CUDA, ESP32 firmware or CUDA paths.
set -Eeuo pipefail
mode=dry-run
gazebo=no
for arg in "$@"; do
  case "$arg" in
    --apply) mode=apply ;;
    --with-gazebo) gazebo=yes ;;
    -h|--help) echo "Usage: scripts/setup_ubuntu24.sh [--apply] [--with-gazebo]"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2;;
  esac
done
if [[ ! -f /etc/os-release ]];then echo '[BLOCKED][OS_UNKNOWN]' >&2; exit 3;fi
. /etc/os-release
if [[ "${ID:-}" != ubuntu || "${VERSION_ID:-}" != 24.04 ]];then
  echo "[BLOCKED][UBUNTU_24_REQUIRED] detected ${ID:-?} ${VERSION_ID:-?}" >&2; exit 3
fi
cat <<'TXT'
Will install (when --apply): git, curl, ca-certificates, gnupg, cmake,
python3, python3-venv, python3-pip, build-essential, pkg-config.
Optional --with-gazebo adds Gazebo's official apt repository and gz-harmonic.
Will NOT install/replace NVIDIA GPU driver, CUDA Toolkit, PX4, AirSim or flash ESP32.
TXT
if [[ $mode != apply ]];then
  echo '[DRY_RUN] No packages installed. Add --apply to proceed.'
  exit 0
fi
sudo apt-get update
sudo apt-get install -y git curl ca-certificates gnupg cmake python3 python3-venv python3-pip build-essential pkg-config
if [[ $gazebo == yes ]];then
  keyring=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
  curl -fsSL https://packages.osrfoundation.org/gazebo.gpg | sudo tee "$keyring" >/dev/null
  echo "deb [arch=$(dpkg --print-architecture) signed-by=$keyring] https://packages.osrfoundation.org/gazebo/ubuntu-stable noble main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y gz-harmonic
fi
echo '[SETUP_FINISHED][BASE_PACKAGES_ONLY] Validate with ./run.sh doctor; GPU/CUDA and physics are still unverified.'
