#!/usr/bin/env python3
""".splat (antimatter15 32B/point) -> INRIA 3DGS PLY。

输入不是训练数据，只是把公开/自建 .splat 转成 gsplat 可读的 PLY。
默认把 y-up 转成 nav 的 z-up：nav=(x,-z,y)，尺度可用 --scale 调。
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim", "3dgs"))

import scene as scene_mod  # noqa: E402

DT = np.dtype([("pos", "<f4", (3,)), ("scale", "<f4", (3,)),
               ("rgba", "u1", (4,)), ("rot", "u1", (4,))])


def convert(src, dst, rotate_y_up=True, scale=1.0, flip_height=False,
            z_offset=0.0):
    a = np.fromfile(src, dtype=DT)
    if a.size == 0:
        raise ValueError(f"{src}: 空 .splat")
    means = a["pos"].astype(np.float32)
    if rotate_y_up:
        means = np.stack([means[:, 0], -means[:, 2], means[:, 1]], axis=1)
    means *= float(scale)
    if flip_height:
        means[:, 2] *= -1.0
    means[:, 2] += float(z_offset)
    colors = a["rgba"][:, :3].astype(np.float32) / 255.0
    scales = a["scale"].astype(np.float32) * float(scale)
    quats = (a["rot"].astype(np.float32) - 128.0) / 128.0
    n = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / np.maximum(n, 1e-9)
    opac = a["rgba"][:, 3].astype(np.float32) / 255.0
    os.makedirs(os.path.dirname(os.path.abspath(dst)) or ".", exist_ok=True)
    scene_mod.write_gaussian_ply(dst, means, colors, scales, quats, opac)
    print("wrote %s n=%d bounds min=%s max=%s"
          % (dst, len(means), means.min(axis=0).round(3).tolist(),
             means.max(axis=0).round(3).tolist()))
    return means


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--no-rotate-y-up", action="store_true")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--flip-height", action="store_true")
    p.add_argument("--z-offset", type=float, default=0.0)
    a = p.parse_args()
    convert(a.src, a.dst, not a.no_rotate_y_up, a.scale, a.flip_height,
            a.z_offset)


if __name__ == "__main__":
    main()
