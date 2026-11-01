#!/usr/bin/env python3
"""深度单位自测：png16_mm 换算成米，旧归一化口径保持不变。"""
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from dataset import _load_depth

SIZE = 8


def _write_png16(path, value):
    arr = np.full((SIZE, SIZE), value, dtype=np.uint16)
    Image.fromarray(arr).save(path)


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "depth_00000.png"

        _write_png16(p, 1500)
        metres = _load_depth(p, SIZE, "png16_mm")
        assert np.allclose(metres, 1.5), f"png16_mm: 期望 1.5 m，实际 {metres[0,0]}"

        _write_png16(p, 65535)
        capped = _load_depth(p, SIZE, "png16_mm")
        assert np.allclose(capped, 65.535), f"png16_mm 量程: {capped[0,0]}"

        _write_png16(p, 32767)
        legacy = _load_depth(p, SIZE)
        assert 0.49 < legacy[0, 0] < 0.51, f"旧口径应归一化，实际 {legacy[0,0]}"

        _write_png16(p, 0)
        zero = _load_depth(p, SIZE, "png16_mm")
        assert np.allclose(zero, 0.0), "0 应表示无效深度"

    print("PASS")


if __name__ == "__main__":
    main()
