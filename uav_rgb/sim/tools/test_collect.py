#!/usr/bin/env python3
"""collect.py 纯函数自检：contact_fired 的坏消息、read_depth 的格式/stride。

不需要 Gazebo：contact_fired 用假进程，read_depth 用假 echo_json。
用法：python3 sim/tools/test_collect.py
"""
import base64
import contextlib
import io
import os
import struct
import subprocess
import sys

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)
import collect

passed = 0


def check(name, cond):
    global passed
    if not cond:
        raise AssertionError(f"FAIL: {name}")
    passed += 1
    print(f"  ok  {name}")


class FakeProc:
    def __init__(self, out=b"", rc=0, timeout=False):
        self.out = out
        self.returncode = rc
        self.timeout = timeout

    def communicate(self, timeout=None):
        if self.timeout:
            raise subprocess.TimeoutExpired("gz", timeout)
        return self.out, b""

    def kill(self):
        pass

    def wait(self):
        return 0


def contact(name, out, rc=0, want=False):
    got = collect.contact_fired(FakeProc(out, rc))
    check(name, got is want)


def read_depth_with(msg):
    old = collect.echo_json
    collect.echo_json = lambda _topic: msg
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return collect.read_depth()
    finally:
        collect.echo_json = old


def depth_msg(w, h, step, fmt, raw):
    return {"width": w, "height": h, "step": step,
            "pixelFormatType": fmt,
            "data": base64.b64encode(raw).decode()}


def main():
    contact("contact 有元素 -> True", b'{"contact":[{}]}', want=True)
    contact("contact 空列表 -> False", b'{"contact":[]}')
    contact("接触 JSON null -> False", b'null')
    contact("接触 JSON 列表 -> False", b'[]')
    contact("接触空输出 -> False", b"")
    contact("接触坏 JSON -> False", b'not json')
    contact("接触 rc!=0 -> False", b'{"contact":[{}]}', rc=1)
    got = collect.contact_fired(FakeProc(timeout=True))
    check("接触超时 -> False", got is False)

    d = read_depth_with(depth_msg(2, 2, 8, "R_FLOAT32",
                                  struct.pack("<4f", 1, 2, 3, 4)))
    check("深度紧凑 2x2", d is not None and d[0] == [1, 2, 3, 4]
          and d[1:3] == (2, 2))

    row1 = struct.pack("<2f", 1, 2)
    row2 = struct.pack("<2f", 3, 4)
    raw = row1 + b"\x00" * 4 + row2 + b"\x00" * 4
    d = read_depth_with(depth_msg(2, 2, 12, "R_FLOAT32", raw))
    check("深度 padded step=12", d is not None and d[0] == [1, 2, 3, 4])

    check("深度 float64 格式拒绝",
          read_depth_with(depth_msg(2, 2, 16, "R_FLOAT64",
                                    b"\x00" * 32)) is None)
    check("深度 RGB 格式拒绝",
          read_depth_with(depth_msg(2, 2, 8, "RGB_INT8",
                                    b"\x00" * 16)) is None)
    check("深度 payload 截断拒绝",
          read_depth_with(depth_msg(2, 2, 8, "R_FLOAT32",
                                    b"\x00" * 8)) is None)

    print(f"test_collect: {passed} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
