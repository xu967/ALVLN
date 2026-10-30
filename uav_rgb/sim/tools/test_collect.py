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
import tempfile
from unittest.mock import patch

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


class FakeSub:
    def __init__(self, frames=(), latest=None, after=None, healthy=True):
        self._frames, self._latest, self._after = list(frames), latest, after
        self._healthy = healthy

    def frames(self):
        return self._frames

    def latest(self):
        return self._latest

    def healthy(self):
        return self._healthy

    def latest_after(self, _ts, timeout=0.0):
        return self._after

    def close(self):
        pass


class DyingContact(FakeSub):
    """第一次 healthy() 为真（步首检查），之后为假（动作窗口内退出）。"""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def healthy(self):
        self.calls += 1
        return self.calls == 1


class ClockPoseSub(FakeSub):
    """latest() 返回当前假时钟；测试通过 act/sleep 推进它。"""

    def __init__(self, initial):
        super().__init__(frames=[(1, initial)], latest=initial)
        self.current = initial

    def latest(self):
        return self.current


class FakePub:
    def __init__(self):
        self.t = 0.0

    def start(self, *args):
        self.t += 0.001
        return self.t

    def wait(self, handle):
        return handle

    def abort(self, handle):
        pass

    def close(self):
        pass


def image_msg(w, h, sec):
    data = bytes((i * 7) % 256 for i in range(w * h * 3))
    return {"width": w, "height": h, "step": w * 3,
            "pixelFormatType": "RGB_INT8",
            "data": base64.b64encode(data).decode(),
            "header": {"stamp": {"sec": sec, "nsec": 0}}}


def pose_msg(sec, z=2.1):
    whole = int(sec)
    return {"pose": [{"position": {"x": 0.0, "y": 0.0, "z": z},
                      "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                      "header": {"stamp": {"sec": whole,
                                           "nsec": int(round((sec - whole) * 1e9))}}}]}


def _fresh_depth(w, h, sec=10):
    dep = depth_msg(w, h, 4 * w, "R_FLOAT32",
                    struct.pack("<%df" % (w * h), *([5.0] * (w * h))))
    dep["header"] = {"stamp": {"sec": sec, "nsec": 0}}
    return dep


def stale_depth_fails():
    """depth 缓冲里只有旧帧：不能把旧深度当本步观测。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    dep = _fresh_depth(w, h, sec=9)       # 比 RGB 旧 1 s
    pos = pose_msg(10.5)
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, dep)]),
            "pose": FakeSub(frames=[(1, pos)], latest=pos), "contact": FakeSub()}
    with tempfile.TemporaryDirectory() as d:
        row, err = collect.collect_one(d, 0, 0.01, "teacher", subs, FakePub())
    return row, err


def dead_depth_fails():
    """depth 订阅进程退出：即使缓冲里还有旧帧也必须失败。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    dep = _fresh_depth(w, h)
    pos = pose_msg(10.5)
    subs = {"rgb": FakeSub(after=rgb),
            "depth": FakeSub(frames=[(1, dep)], healthy=False),
            "pose": FakeSub(frames=[(1, pos)], latest=pos), "contact": FakeSub()}
    with tempfile.TemporaryDirectory() as d:
        row, err = collect.collect_one(d, 0, 0.01, "teacher", subs, FakePub())
    return row, err


def dead_contact_fails():
    """contact 订阅进程退出：步首就拒绝，不能默认「无接触」。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pos = pose_msg(10.5)
    subs = {"rgb": FakeSub(after=rgb),
            "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": FakeSub(frames=[(1, pos)], latest=pos),
            "contact": FakeSub(healthy=False)}
    with tempfile.TemporaryDirectory() as d:
        row, err = collect.collect_one(d, 0, 0.01, "teacher", subs, FakePub())
    return row, err


def dying_contact_window_fails():
    """动作窗口内 contact 订阅退出：必须零速并失败，不能把碰撞漏成 False。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.5))
    subs = {"rgb": FakeSub(after=rgb),
            "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": DyingContact()}
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)):
        row, err = collect.collect_one(d, 0, 1.0, "teacher", subs, FakePub())
    return row, err


def dead_contact_takeoff_fails():
    """起飞前 contact 订阅已退出：直接拒绝，不能带着瞎掉的碰撞检测起飞。"""
    pos = FakeSub(latest=pose_msg(0.2, z=0.2))
    try:
        collect.take_off(pos, FakePub(), FakeSub(healthy=False), timeout=0.01)
    except RuntimeError as e:
        return "contact" in str(e)
    return False


def stale_pose_fails():
    """窗口内位姿不前进：动作窗口走不满，该步必须失败。

    动作窗口改成仿真时间后，位姿停更在窗口内就被抓住（旧实现是墙钟跑满、
    事后才发现没有新位姿），两种都是「不能把动作前位姿当 oracle 写成功行」。
    """
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    dep = depth_msg(w, h, 4 * w, "R_FLOAT32",
                    struct.pack("<%df" % (w * h), *([5.0] * (w * h))))
    dep["header"] = {"stamp": {"sec": 10, "nsec": 0}}
    pos = pose_msg(10.5)
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, dep)]),
            "pose": FakeSub(frames=[(1, pos)], latest=pos), "contact": FakeSub()}
    with tempfile.TemporaryDirectory() as d:
        row, err = collect.collect_one(d, 0, 0.01, "teacher", subs, FakePub())
    return row, err


def delayed_inference_window():
    """推理期间仿真推进：执行窗口必须从派发时刻算，不能从观测帧算。

    回归 R01：旧实现 action_start_ns=观测戳(10.2)，act 期间仿真到 11.2，
    hold=1.0 的终点 11.2 已经过去，零速会和动作在同一仿真时刻发出。
    """
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    dep = depth_msg(w, h, 4 * w, "R_FLOAT32",
                    struct.pack("<%df" % (w * h), *([5.0] * (w * h))))
    dep["header"] = {"stamp": {"sec": 10, "nsec": 0}}
    pose = ClockPoseSub(pose_msg(10.2))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, dep)]),
            "pose": pose, "contact": FakeSub()}

    class Student:
        def act(self, msg):
            pose.current = pose_msg(11.2)   # 推理期间仿真走到 11.2
            return 1, 500.0, [0.0, 1.0, 0.0, 0.0]

    class RecordingPub(FakePub):
        def __init__(self):
            super().__init__()
            self.sent = []

        def start(self, *args):
            self.sent.append((collect._pose_stamp(pose.latest()), args))
            return super().start(*args)

    def advance(_secs):
        pose.current = pose_msg(12.2)       # 窗口走满 1 s

    pub = RecordingPub()
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        row, err = collect.collect_one(d, 0, 1.0, "student", subs, pub,
                                       student=Student())
    return (err is None and row is not None
            and row["pose_before_ts"] == 10_200_000_000
            and row["action_start_ns"] == 11_200_000_000
            and row["sim_dt"] == 1.0
            and [s for s, _ in pub.sent] == [11_200_000_000, 12_200_000_000])


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

    row, err = stale_pose_fails()
    check("pose 停更 -> 该步失败", row is None and "动作窗口超时" in (err or ""))
    row, err = stale_depth_fails()
    check("depth 停更 -> 该步失败", row is None and "depth 停更" in (err or ""))
    row, err = dead_depth_fails()
    check("depth 订阅退出 -> 该步失败", row is None and "depth 订阅进程已退出" in (err or ""))
    row, err = dead_contact_fails()
    check("contact 订阅退出 -> 该步失败",
          row is None and "contact 订阅进程已退出" in (err or ""))
    row, err = dying_contact_window_fails()
    check("动作窗口内 contact 退出 -> 该步失败",
          row is None and "动作窗口内 contact" in (err or ""))
    check("起飞前 contact 退出 -> 拒绝起飞", dead_contact_takeoff_fails())
    check("推理延迟不缩短执行窗口", delayed_inference_window())

    print(f"test_collect: {passed} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
