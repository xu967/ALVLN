#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")/.."
if [[ ! -f SHA256SUMS.txt ]];then echo '[ERR][HASH_LIST_MISSING]' >&2;exit 3;fi
sha256sum -c SHA256SUMS.txt
./run.sh test
./run.sh cpp-test
echo '[PASS][PACKAGE_LOCAL_INTEGRITY_AND_TESTS] Not a Gazebo/NPU/ESP32 test.'
