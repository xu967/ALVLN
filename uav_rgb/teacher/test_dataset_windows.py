#!/usr/bin/env python3
"""滑窗数据集自测：标签必须取窗口末帧，不能恒为收尾的 SAFE。

回归的是这个 bug：旧实现只取每集最后 seq_len 帧、标签取最后一帧，
而教师集都以停在 waypoint 的 SAFE 收尾，于是所有样本标签都是 0，
训练 acc 假性 1.0。
"""
import json
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import EpisodeSequenceDataset

N_FRAMES = 6
LABELS = [1, 1, 2, 2, 3, 0]  # 最后一步是收尾 SAFE
SEQ = 3
INSTRUCTION = "pass the red pillar on your left and stop"


def _make_episode(root: Path):
    ep = root / "910001"
    (ep / "depth").mkdir(parents=True)
    rows = []
    for i in range(N_FRAMES):
        Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(ep / f"rgb_{i:05d}.png")
        d = np.full((8, 8), 1000 + i, dtype=np.uint16)
        Image.fromarray(d).save(ep / "depth" / f"depth_{i:05d}.png")
        rows.append({
            "step": i, "frame": f"rgb_{i:05d}.png",
            "depth_file": f"depth/depth_{i:05d}.png",
            "pose_before": {"x": 0.5 * i, "y": 0.0, "z": 2.0},
            "yaw_before": 0.0,
            "rgb_ts": 1_000_000_000 + i * 1_000_000_000,
            "rgb_ok": True,
            "action_expert": LABELS[i],
        })
    with open(ep / "actions.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(ep / "episode_manifest.json", "w", encoding="utf-8") as f:
        json.dump({
            "episode_id": "910001",
            "instruction": INSTRUCTION,
            "depth_ref": "depth/depth_%05d.png",
            "depth_format": "png16_mm",
        }, f)


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        _make_episode(root)

        ds = EpisodeSequenceDataset(root, seq_len=SEQ, image_size=8)
        expected = N_FRAMES - SEQ + 1
        assert len(ds) == expected, f"滑窗数 {len(ds)} != {expected}"

        labels = [ds.samples[i]["label"] for i in range(len(ds))]
        assert labels == LABELS[SEQ - 1:], f"标签应以窗口末帧为准，实际 {labels}"

        uniq = sorted(set(labels))
        assert uniq == [0, 2, 3], f"标签不应恒为收尾 SAFE，实际 {uniq}"

        # 每个窗口的最后一帧必须就是标签所在帧
        for i in range(len(ds)):
            s = ds.samples[i]
            assert s["step"] == SEQ - 1 + i, f"窗口 {i} 末帧 {s['step']} 不符"
            assert len(s["rgb_paths"]) == SEQ

        it = ds[0]
        assert tuple(it["depth"].shape) == (SEQ, 1, 8, 8)
        # 深度必须按米读：1000 mm -> 1.0 m
        assert abs(float(it["depth"][0, 0, 0, 0]) - 1.0) < 1e-3, \
            f"深度单位错误: {float(it['depth'][0,0,0,0])}"

        ds2 = EpisodeSequenceDataset(root, seq_len=SEQ, image_size=8, stride=2)
        assert len(ds2) == 2, f"stride=2 应为 2 个窗口，实际 {len(ds2)}"

    print("PASS")


if __name__ == "__main__":
    main()
