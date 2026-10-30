#!/usr/bin/env python3
"""生成用于预处理流水线自检的合成 episode（不使用任何真实采集数据）。

固定产出 3 条 episode（tools/read_dataset.cpp --selfcheck 依赖这套约定）：
  ep1: episode_id=1 scene_id=42 split=train 6 步
  ep2: episode_id=2 scene_id=42 split=train 4 步
  ep3: episode_id=3 scene_id=43 split=val   5 步
共 15 个样本，标签 action_expert = 每条 episode 内 step % 4。

像素约定（全局样本序号 g，从 0 起）：
  g 偶数：整帧常量色 R=40+(g*37)%180, G=40+(g*53)%180, B=40+(g*71)%180
          （双线性缩放常量图仍精确等于该常量，用于逐像素精确校验）
  g 奇数：R 随 x 线性递增、G 随 y 线性递增、B 恒为 77
          （用于校验 NCHW 布局、通道顺序、宽高方向；不做逐像素精确比对）

用法：
  .venv-tools/bin/python tools/make_synthetic_episode.py --out /tmp/synth
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sim", "tools"))
import split_manifest as sm  # noqa: E402

SRC_W, SRC_H = 320, 240
EPISODES = [
    {"name": "ep1", "episode_id": 1, "scene_id": 42, "seed": 7, "split": "train", "steps": 6,
     "instruction": "fly to the red pillar and stop"},
    {"name": "ep2", "episode_id": 2, "scene_id": 42, "seed": 8, "split": "train", "steps": 4,
     "instruction": "fly to the blue pillar and stop"},
    {"name": "ep3", "episode_id": 3, "scene_id": 43, "seed": 9, "split": "val", "steps": 5,
     "instruction": "fly to the red pillar and stop"},
]


def make_frame(g):
    if g % 2 == 0:
        color = (40 + (g * 37) % 180,
                 40 + (g * 53) % 180,
                 40 + (g * 71) % 180)
        return np.full((SRC_H, SRC_W, 3), color, dtype=np.uint8)
    x = np.arange(SRC_W, dtype=np.uint32)[None, :]
    y = np.arange(SRC_H, dtype=np.uint32)[:, None]
    r = (x * 255 // (SRC_W - 1)).astype(np.uint8)
    gch = (y * 255 // (SRC_H - 1)).astype(np.uint8)
    return np.stack([np.broadcast_to(r, (SRC_H, SRC_W)),
                     np.broadcast_to(gch, (SRC_H, SRC_W)),
                     np.full((SRC_H, SRC_W), 77, dtype=np.uint8)], axis=-1)


def write_episode(root, spec, g0):
    ep_dir = os.path.join(root, spec["name"])
    os.makedirs(ep_dir, exist_ok=True)
    actions = []
    for step in range(spec["steps"]):
        g = g0 + step
        Image.fromarray(make_frame(g), "RGB").save(
            os.path.join(ep_dir, "rgb_%05d.png" % step))
        label = step % 4
        actions.append({"step": step,
                        "timestamp_ns": 1_000_000_000 + 100_000_000 * step,
                        "action_expert": label,
                        "action_executed": label,
                        "collision": False})
    with open(os.path.join(ep_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
        for row in actions:
            f.write(json.dumps(row) + "\n")
    with open(os.path.join(ep_dir, "oracle.jsonl"), "w", encoding="utf-8") as f:
        for row in actions:
            f.write(json.dumps({"step": row["step"], "timestamp_ns": row["timestamp_ns"],
                                "pose_gt": {"x": 0.1 * row["step"], "y": 0.0, "z": 1.5}}) + "\n")
    manifest = {
        "schema": "navlab.episode.v1",
        "episode_id": spec["episode_id"],
        "scene_id": spec["scene_id"],
        "seed": spec["seed"],
        "version": "synthetic",
        "split": spec["split"],
        "teacher": "synthetic",
        "instruction": spec.get("instruction"),
        "geometry_version": sm.SPLIT_SALT,
        "actions_ref": "actions.jsonl",
        "oracle_ref": "oracle.jsonl",
        "camera": {"width": SRC_W, "height": SRC_H,
                   "fx": None, "fy": None, "cx": None, "cy": None},
        # 合成 fixture 是格式契约测试，不是物理采集；validity 只表示文件完整。
        "validity": {"camera_verified": True, "physics_verified": True,
                     "episode_executed": True},
        "num_steps": spec["steps"],
    }
    with open(os.path.join(ep_dir, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return g0 + spec["steps"]


def main():
    p = argparse.ArgumentParser(description="生成合成 episode（预处理自检用，非真实数据）")
    p.add_argument("--out", required=True, help="输出根目录（生成 ep1/ep2/ep3）")
    args = p.parse_args()
    g = 0
    for spec in EPISODES:
        g = write_episode(args.out, spec, g)
    print("synthetic episodes: %d  samples: %d  -> %s"
          % (len(EPISODES), g, os.path.abspath(args.out)))
    print("  " + "  ".join("%s(scene=%d,%s,%d步)" % (e["name"], e["scene_id"],
                                                    e["split"], e["steps"])
                           for e in EPISODES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
