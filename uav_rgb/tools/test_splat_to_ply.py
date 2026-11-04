#!/usr/bin/env python3
"""splat_to_ply 最小回归：2 点 .splat -> PLY -> scene.read_gaussian_ply。"""
import os
import struct
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim", "3dgs"))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import scene as scene_mod  # noqa: E402
import splat_to_ply  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "a.splat")
        with open(src, "wb") as f:
            # pos, linear scale, rgba, quat bytes (w,x,y,z)
            f.write(struct.pack("<6f4B4B", 1.0, 2.0, 3.0, 0.1, 0.2, 0.3,
                                10, 20, 30, 200, 255, 128, 128, 128))
            f.write(struct.pack("<6f4B4B", -1.0, 0.0, 0.5, 0.02, 0.03, 0.04,
                                255, 0, 128, 100, 255, 128, 128, 128))
        dst = os.path.join(d, "a.ply")
        splat_to_ply.convert(src, dst, rotate_y_up=False, scale=1.0)
        g = scene_mod.read_gaussian_ply(dst)
        assert g["means"].shape == (2, 3), g["means"].shape
        assert np.allclose(g["means"][0], [1, 2, 3], atol=1e-5)
        assert np.allclose(g["scales"][0], [0.1, 0.2, 0.3], atol=1e-5)
        assert np.allclose(g["colors"][0], [10 / 255, 20 / 255, 30 / 255],
                           atol=1e-5)
        assert abs(float(g["opacities"][0]) - 200 / 255) < 1e-4
        assert abs(float(np.linalg.norm(g["quats"][0])) - 1.0) < 1e-5
    print("splat_to_ply: PASS")


if __name__ == "__main__":
    main()
