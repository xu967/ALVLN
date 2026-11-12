#!/usr/bin/env python3
"""DAgger 第 3 轮打包的集目录清单：与 tools/dagger_gz_pipeline.py 的 round0_settings(...)["dirs"] + run_round 里的 new 同一规则
（每个 collect 目录下 train/val/test 里带 episode_manifest.json 的集目录，各自 sorted；先 round0，再 round_1..3）。"""
import glob
import os
import sys

D = "/root/rivermind-data/uav_rgb"
FULL = f"{D}/exp-dagger-gz-20261109/full-1109"
ROUND0 = f"{D}/tmp/agent6-distill/trunc/col"


def episode_dirs(col):
    return sorted(d.rstrip("/") for s in ("train", "val", "test") for d in glob.glob(f"{col}/{s}/*/")
                  if os.path.isfile(os.path.join(d, "episode_manifest.json")))


dirs = episode_dirs(ROUND0) + [d for j in (1, 2, 3) for d in episode_dirs(f"{FULL}/round_{j}/collect")]
out = sys.argv[1]
with open(out, "w") as f:
    f.write("\n".join(dirs) + "\n")
print(f"{len(dirs)} 个集目录 → {out}（round0 {len(episode_dirs(ROUND0))}，" +
      "，".join(f"round_{j} {len(episode_dirs(f'{FULL}/round_{j}/collect'))}" for j in (1, 2, 3)) + "）")
