#!/usr/bin/env python3
"""save_frame.py 自检：stride padding、通道交换、异常格式/截断。

覆盖旧审核 S2/S3 的真实 bug：2x2 step=7、4x2 step=13、320x2 step=1024 的
行 padding，BGR/BGRA 交换，灰度/16-bit/截断/非法尺寸必须 rc=1 且不建文件。
用法：python3 sim/tools/test_save_frame.py
"""
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import zlib

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import save_frame


def msg(w, h, step, fmt, data):
    return {"width": w, "height": h, "step": step,
            "pixelFormatType": fmt, "data": base64.b64encode(data).decode()}


def png_pixels(png):
    assert png[:8] == b'\x89PNG\r\n\x1a\n'
    pos, idat, w, h, ctype = 8, b'', None, None, None
    while pos < len(png):
        n = struct.unpack('>I', png[pos:pos + 4])[0]
        tag = png[pos + 4:pos + 8]
        body = png[pos + 8:pos + 8 + n]
        if tag == b'IHDR':
            w, h, depth, ctype = struct.unpack('>IIBB', body[:10])
        elif tag == b'IDAT':
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    nch = {2: 3, 6: 4}[ctype]
    rows = []
    for y in range(h):
        line = raw[y * (1 + w * nch):(y + 1) * (1 + w * nch)]
        assert line[0] == 0
        rows.append([tuple(line[1 + i * nch:1 + (i + 1) * nch]) for i in range(w)])
    return rows


def check(name, m, want):
    _, _, _, png = save_frame.png_bytes(m)
    got = png_pixels(png)
    ok = got == want
    print(("ok  " if ok else "FAIL") + f" {name}: {got}")
    return ok


def expect_fail(name, m):
    with tempfile.TemporaryDirectory() as d:
        dst = os.path.join(d, "out.png")
        p = subprocess.run(
            [sys.executable, os.path.join(TOOLS, "save_frame.py"), "-", dst],
            input=json.dumps(m).encode(), capture_output=True)
        exists = os.path.exists(dst)
        ok = p.returncode != 0 and not exists
        print(("ok  " if ok else "FAIL")
              + f" {name}: rc={p.returncode} file={exists} "
              + f"err={p.stderr.decode().strip()[:80]}")
        return ok


def main():
    ok = True
    ok &= check("1x1 rgb", msg(1, 1, 3, "RGB_INT8", bytes((10, 20, 30))),
                [[(10, 20, 30)]])

    # 2x2, step=7: 每行 1 字节 0xaa padding
    raw = bytes((1, 2, 3, 4, 5, 6, 0xaa, 10, 20, 30, 40, 50, 60, 0xaa))
    ok &= check("2x2 step7 pad", msg(2, 2, 7, "RGB_INT8", raw),
                [[(1, 2, 3), (4, 5, 6)], [(10, 20, 30), (40, 50, 60)]])

    # 4x2, step=13: 每行 1 字节 padding
    r1 = bytes(range(1, 13))
    r2 = bytes(range(21, 33))
    ok &= check("4x2 step13 pad", msg(4, 2, 13, "RGB_INT8", r1 + b'\xaa' + r2 + b'\xaa'),
                [[(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12)],
                 [(21, 22, 23), (24, 25, 26), (27, 28, 29), (30, 31, 32)]])

    # 320x2, step=1024: 每行 64 字节 padding
    row = bytes((i * 7) % 256 for i in range(960))
    raw = row + b'\xcc' * 64 + bytes((255 - i) % 256 for i in range(960)) + b'\xcc' * 64
    w, h, nch, png = save_frame.png_bytes(msg(320, 2, 1024, "RGB_INT8", raw))
    px = png_pixels(png)
    good = px[0][0] == (0, 7, 14) and px[1][0] == (255, 254, 253)
    ok &= good
    print(("ok  " if good else "FAIL")
          + f" 320x2 step1024 pad: row0[0]={px[0][0]} row1[0]={px[1][0]}")

    ok &= check("2x2 rgba",
                msg(2, 2, 8, "RGBA_INT8",
                    bytes((1, 2, 3, 255, 4, 5, 6, 128,
                           10, 20, 30, 255, 40, 50, 60, 0))),
                [[(1, 2, 3, 255), (4, 5, 6, 128)],
                 [(10, 20, 30, 255), (40, 50, 60, 0)]])

    # BGR -> RGB 交换
    ok &= check("1x1 bgr", msg(1, 1, 3, "BGR_INT8", bytes((30, 20, 10))),
                [[(10, 20, 30)]])

    # 失败路径必须 rc!=0，且不能留下 0 字节坏文件
    ok &= expect_fail("gray", msg(2, 1, 2, "L_INT8", bytes((1, 2))))
    ok &= expect_fail("16bit rgb", msg(1, 1, 6, "RGB_INT16", bytes((1, 2, 3, 4, 5, 6))))
    ok &= expect_fail("truncated", msg(2, 2, 6, "RGB_INT8", bytes(range(8))))
    ok &= expect_fail("bad size", msg(0, 2, 6, "RGB_INT8", b""))

    print("test_save_frame:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
