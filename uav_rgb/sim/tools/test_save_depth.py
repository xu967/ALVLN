#!/usr/bin/env python3
"""--save-depth 自检：白名单投影 + 16 位毫米 PNG 编码。

不需要 Gazebo / 订阅 / 发布：只测 collect.py 的纯函数。
用法：python3 sim/tools/test_save_depth.py
"""
import os
import struct
import sys
import tempfile
import zlib

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import collect

OLD_KEYS = (
    "step", "timestamp_ns", "action_expert", "action_executed",
    "collision", "rgb_ok", "frame", "src", "rgb_ts", "depth_ts",
    "depth_rgb_gap_ns", "pose_ts", "pose_rgb_gap_ns", "dyaw_deg",
    "displacement_m", "infer_ms", "action_start_ns", "pose_before",
    "yaw_before",
)


def base_row():
    return {
        "step": 3, "timestamp_ns": 10, "action_expert": 1,
        "action_executed": 1, "collision": False, "rgb_ok": True,
        "frame": "rgb_00003.png", "src": None, "rgb_ts": 10,
        "depth_ts": 9, "depth_rgb_gap_ns": 1, "pose_before_ts": 11,
        "pose_rgb_gap_ns": 1, "dyaw_deg": 0.0, "displacement_m": 0.1,
        "infer_ms": None, "action_start_ns": 12,
        "pose_before": {"x": 0.0, "y": 0.0, "z": 2.0}, "yaw_before": 0.0,
        "depth_file": "depth/depth_00003.png",
    }


def png16_values(png, w, h):
    """解析本工具写出的 16 位灰度 PNG，返回全部样本值（校验字节序/位深）。"""
    assert png[:8] == b"\x89PNG\r\n\x1a\n", "PNG signature"
    ihdr = struct.unpack(">IIBBBBB", png[16:29])
    assert ihdr == (w, h, 16, 0, 0, 0, 0), f"IHDR 非 16 位灰度: {ihdr}"
    pos, idat = 8, b""
    while pos < len(png):
        n = struct.unpack(">I", png[pos:pos + 4])[0]
        if png[pos + 4:pos + 8] == b"IDAT":
            idat += png[pos + 8:pos + 8 + n]
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = 1 + 2 * w
    assert len(raw) == h * stride, f"扫描行长度 {len(raw)} != {h * stride}"
    vals = []
    for y in range(h):
        row = raw[y * stride:(y + 1) * stride]
        assert row[0] == 0, "仅支持 filter=None"
        vals.extend(struct.unpack(">%dH" % w, row[1:]))
    return vals


def main():
    # 1) 白名单投影：关闭时字段集与旧版完全一致，开启时只多 depth_file。
    row = base_row()
    off = collect.action_row(row, save_depth=False)
    assert "depth_file" not in off, "save_depth=False 不得落 depth_file"
    assert tuple(off) == OLD_KEYS, f"旧字段集/顺序变了: {tuple(off)}"
    on = collect.action_row(row, save_depth=True)
    assert on["depth_file"] == "depth/depth_00003.png"
    assert tuple(on) == OLD_KEYS + ("depth_file",)

    # 2) 深度编码：米 -> 毫米 uint16，inf/负值 -> 0，超量程截到 65535。
    depth = [1.234, float("inf"), -1.0, 70.0]
    png = collect._depth_png16(depth, 2, 2)
    assert png16_values(png, 2, 2) == [1234, 0, 0, 65535]

    # 3) 落盘路径：write_depth_png 真能写文件，内容与内存编码一致。
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "depth_00003.png")
        assert collect.write_depth_png(depth, 2, 2, path)
        with open(path, "rb") as f:
            assert png16_values(f.read(), 2, 2) == [1234, 0, 0, 65535]

    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
