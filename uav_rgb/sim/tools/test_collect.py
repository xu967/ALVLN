#!/usr/bin/env python3
"""collect.py 纯函数自检：contact_fired 的坏消息、read_depth 的格式/stride。

不需要 Gazebo：contact_fired 用假进程，read_depth 用假 echo_json。
用法：python3 sim/tools/test_collect.py
"""
import base64
import contextlib
import io
import math
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


def depth_gap_step(extra_ns):
    """depth/RGB 配对边界：刚好 100 ms 放行，超过 100 ms 该步失败。

    audit_dataset.py 的 depth 阈值依赖这个 `>` 语义（等于阈值算通过）。
    """
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    dep = _fresh_depth(w, h, sec=10)
    dep["header"] = {"stamp": {"sec": 10, "nsec": extra_ns}}
    pose = ClockPoseSub(pose_msg(10.5))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, dep)]),
            "pose": pose, "contact": FakeSub()}

    def advance(_secs):
        pose.current = pose_msg(11.5)      # 动作窗口走满 1 s

    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        return collect.collect_one(d, 0, 1.0, "teacher", subs, FakePub())


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


def delayed_inference_window(window_from="dispatch", contact_phase=None):
    """推理期间仿真推进：dispatch 从派发时刻算，observation 保留旧语义。

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
            if contact_phase == "inference":
                subs["contact"]._frames.append((1, {"contact": [{}]}))
            return 1, 500.0, [0.0, 1.0, 0.0, 0.0]

    class RecordingPub(FakePub):
        def __init__(self):
            super().__init__()
            self.sent = []

        def start(self, *args):
            self.sent.append((collect._pose_stamp(pose.latest()), args))
            if len(self.sent) == 2 and contact_phase == "zero":
                subs["contact"]._frames.append((1, {"contact": [{}]}))
            return super().start(*args)

    def advance(_secs):
        pose.current = pose_msg(12.2)       # 窗口走满 1 s

    pub = RecordingPub()
    baseline = None
    if contact_phase in ("between", "ground"):
        subs["contact"]._frames.append((1, {"contact": [{}]}))
        baseline = 0 if contact_phase == "between" else 1
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        row, err = collect.collect_one(d, 0, 1.0, "student", subs, pub,
                                       student=Student(), window_from=window_from,
                                       contact_seq0=baseline)
    if err is not None or row is None:
        return False
    if contact_phase is not None:
        if contact_phase == "ground":
            return "contact" not in row["src"] and row["contact_seq_end"] == 1
        return "contact" in row["src"]
    if window_from == "dispatch":
        return (row["pose_before_ts"] == 10_200_000_000
                and row["action_start_ns"] == 11_200_000_000
                and row["sim_dt"] == 1.0
                and [s for s, _ in pub.sent] == [11_200_000_000, 12_200_000_000])
    return (row["pose_before_ts"] == 10_200_000_000
            and row["action_start_ns"] == 10_200_000_000
            and row["sim_dt"] == 1.0
            and [s for s, _ in pub.sent] == [11_200_000_000, 11_200_000_000])


def takeoff_contact_baseline():
    """地面接触不算撞柱，但升空后零速发布期间的新接触不能被巡航基线吞掉。"""
    contact_sub = FakeSub(frames=[(1, {"contact": [{}]})])

    class RisingPose(FakeSub):
        heights = iter((0.15, 0.6, 2.1))

        def latest(self):
            return pose_msg(1, z=next(self.heights))

    class Pub(FakePub):
        def start(self, *args):
            if self.t > 0:
                contact_sub._frames.append((2, {"contact": [{}]}))
            return super().start(*args)

    with patch.object(collect.time, "sleep"):
        baseline = collect.take_off(RisingPose(), Pub(), contact_sub)
    return baseline == 1 and collect._contact_since(contact_sub, baseline)


def cf_stab_model():
    """CfStab 解析解：终端速度、延迟、滑行、分段积分一致、机体系换算。"""
    S = 1_000_000_000
    out = {}
    d = collect.CfStab(tilt_deg=10.0, drag_k=1.0, latency_s=0.0)
    d.dispatch(collect.FORWARD_SLOW, 0)
    d.step(0, 0.0)
    vx, _, _ = d.step(20 * S, 0.0)
    out["terminal"] = abs(vx - 9.81 * math.tan(math.radians(10)) / 1.0) < 1e-6
    a = collect.CfStab(latency_s=0.0)
    b = collect.CfStab(latency_s=0.0)
    for m in (a, b):
        m.dispatch(collect.FORWARD_SLOW, 0)
        m.step(0, 0.0)
    for k in range(1, 11):
        a.step(k * S // 10, 0.0)
    b.step(S, 0.0)
    out["split_exact"] = abs(a.vx - b.vx) < 1e-9 and abs(a.path_m - b.path_m) < 1e-9
    lat = collect.CfStab(latency_s=0.15)
    lat.dispatch(collect.FORWARD_SLOW, 0)
    v_early = lat.step(S // 10, 0.0)[0]
    v_late = lat.step(S // 5, 0.0)[0]
    out["latency"] = v_early == 0.0 and v_late > 0.0
    c = collect.CfStab(latency_s=0.0)
    c.dispatch(collect.FORWARD_SLOW, 0)
    c.step(0, 0.0)
    v0 = c.step(5 * S, 0.0)[0]
    c.dispatch(collect.SAFE_NEUTRAL, 5 * S)
    v1 = c.step(6 * S, 0.0)[0]
    out["coast"] = abs(v1 - v0 * math.exp(-collect.CF_DRAG_K)) < 1e-9 and v1 > 0
    bx, by, _ = c.step(6 * S, 90.0)   # 机头转到 +Y，动量仍沿世界 +X -> 机体右侧(-Y)
    out["body_frame"] = abs(bx) < 1e-9 and abs(by + v1) < 1e-9
    y = collect.CfStab(latency_s=0.0)
    y.dispatch(collect.YAW_LEFT, 0)
    out["yaw"] = y.step(0, 0.0)[2] == collect.SPEED_YAW
    try:
        collect.CfStab(tilt_deg=31.0)
        out["tilt_cap"] = False
    except ValueError:
        out["tilt_cap"] = True
    out["profile"] = collect.CfStab().profile().startswith(collect.CF_STAB_PROFILE + ":")
    return out


class FakeSil:
    """collect_one 的 cf_sil 后端替身：记录派发/停桨，不碰 Gazebo。"""

    def __init__(self, error=None):
        self.actions = []
        self.zeroed = 0
        self.failed = error

    def dispatch(self, action, t_ns):
        self.actions.append((action, t_ns))

    def error(self):
        return self.failed

    def zero_now(self):
        self.zeroed += 1


def cf_sil_window():
    """collect_one + cf_sil：动作交给 sil，不走 twist，也不误判卡滞。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.02))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}

    class RecordingPub(FakePub):
        def __init__(self):
            super().__init__()
            self.sent = []

        def start(self, *args):
            self.sent.append(args)
            return super().start(*args)

    t = [10.02]

    def advance(_secs):
        t[0] = round(t[0] + 0.05, 3)
        pose.current = pose_msg(t[0])

    pub, sil = RecordingPub(), FakeSil()
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        row, err = collect.collect_one(d, 0, 0.2, "forward", subs, pub, sil=sil)
    return {
        "ran": row is not None and err is None,
        "dispatched": sil.actions == [(collect.FORWARD_SLOW, 10_020_000_000)],
        "no_twist": pub.sent == [],
        "no_false_stuck": row is not None and row["src"] is None,
        "action_unchanged": row is not None
                            and row["action_executed"] == collect.FORWARD_SLOW,
    }


def cf_sil_stale_pose_zeroes():
    """cf_sil 动作窗口超时：必须让 sil 停桨，而不是只发 twist 零速。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.02))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}
    sil = FakeSil()
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)):
        row, err = collect.collect_one(d, 0, 0.01, "teacher", subs, FakePub(), sil=sil)
    return (row is None and "动作窗口超时" in (err or "")
            and sil.zeroed > 0 and len(sil.actions) == 1)


def cf_sil_error_fails():
    """后台线程报错后，本步必须失败并停桨。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.02))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}
    sil = FakeSil(error="boom")
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)):
        row, err = collect.collect_one(d, 0, 0.2, "forward", subs, FakePub(), sil=sil)
    return (row is None and "cf_sil 执行器故障" in (err or "")
            and sil.zeroed > 0)


def cf_stab_window(cruise=False):
    """collect_one + cf_stab：窗口内逐帧更新指令、窗口末不刹停、卡滞判定。

    假位姿只走时间不动位置。cruise=False 从静止起步：预测位移 < STUCK_EPS，不能判卡滞
    （X3 跟踪滞后时起步必误报）；cruise=True 先把模型推到巡航速度：贴柱不动必须判卡滞。
    """
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.02))   # 位姿必须晚于 RGB(10 s)
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}

    class RecordingPub(FakePub):
        def __init__(self):
            super().__init__()
            self.sent = []

        def start(self, *args):
            self.sent.append(args)
            return super().start(*args)

    t = [10.02]

    def advance(_secs):
        t[0] = round(t[0] + 0.05, 3)
        pose.current = pose_msg(t[0])

    pub = RecordingPub()
    dyn = collect.CfStab(latency_s=0.0)
    if cruise:
        dyn.dispatch(collect.FORWARD_SLOW, 0)
        dyn.step(0, 0.0)
        dyn.step(10_020_000_000, 0.0)
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        row, err = collect.collect_one(d, 0, 0.2, "forward", subs, pub, dyn=dyn)
    ran = "ran_cruise" if cruise else "ran"
    if err is not None or row is None:
        return {ran: False}
    vxs = [s[0] for s in pub.sent]
    if cruise:
        return {ran: True, "stuck_at_cruise": row["src"] == "stuck"}
    return {"ran": True,
            "updates": len(pub.sent) >= 4 and vxs == sorted(vxs),
            "no_brake": vxs[-1] > 0.0,
            "no_false_stuck_from_rest": row["src"] is None}


def main():
    m = cf_stab_model()
    for k, v in m.items():
        check(f"cf_stab 模型: {k}", v)
    for k, v in (cf_stab_window() | cf_stab_window(cruise=True)).items():
        check(f"cf_stab 采集窗口: {k}", v)
    check("cf_sil 在 DYNAMICS 且默认参数存在",
          "cf_sil" in collect.DYNAMICS and os.path.isfile(collect.DEFAULT_CF_SIL_PARAMS))
    check("cf_sil 只保留为手动实验档", collect.DYNAMICS_EXPERIMENTAL == ("cf_sil",))
    # 2026-11-05 决定：cf_sil 退出默认训练/评测矩阵，cf_blackbox 只留名字给 3DGS。
    import collect_parallel as cp
    check("编排器默认矩阵去掉 cf_sil", "cf_sil" not in cp.DYNAMICS_CHOICES)
    check("编排器默认矩阵含 cf_blackbox",
          "cf_blackbox" in cp.DYNAMICS_CHOICES
          and "x3_velocity" in cp.DYNAMICS_CHOICES
          and "cf_stab" in cp.DYNAMICS_CHOICES)
    check("cf_blackbox 在 collect 且默认参数存在",
          "cf_blackbox" in collect.DYNAMICS
          and os.path.isfile(collect.DEFAULT_CF_BLACKBOX_PARAMS))
    for k, v in cf_sil_window().items():
        check(f"cf_sil 采集窗口: {k}", v)
    check("cf_sil 窗口超时 -> 停桨", cf_sil_stale_pose_zeroes())
    check("cf_sil 执行器故障 -> 该步失败并停桨", cf_sil_error_fails())
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
    row, err = depth_gap_step(100_000_000)
    check("depth gap 恰好 100 ms -> 放行", row is not None and err is None)
    row, err = depth_gap_step(101_000_000)
    check("depth gap 101 ms -> 该步失败", row is None and "depth 停更" in (err or ""))
    row, err = dead_depth_fails()
    check("depth 订阅退出 -> 该步失败", row is None and "depth 订阅进程已退出" in (err or ""))
    row, err = dead_contact_fails()
    check("contact 订阅退出 -> 该步失败",
          row is None and "contact 订阅进程已退出" in (err or ""))
    row, err = dying_contact_window_fails()
    check("动作窗口内 contact 退出 -> 该步失败",
          row is None and "动作窗口内 contact" in (err or ""))
    check("起飞前 contact 退出 -> 拒绝起飞", dead_contact_takeoff_fails())
    check("推理延迟不缩短执行窗口（dispatch）", delayed_inference_window("dispatch"))
    check("observation 语义保留旧行为（窗口被推理吃掉）",
          delayed_inference_window("observation"))
    check("推理期间 contact 不漏报", delayed_inference_window(contact_phase="inference"))
    check("零速发布后 contact 不漏报", delayed_inference_window(contact_phase="zero"))
    check("步间漂移 contact 归下一步", delayed_inference_window(contact_phase="between"))
    check("历史地面 contact 不污染巡航", delayed_inference_window(contact_phase="ground"))
    check("起飞基线保留最后零速期间的新 contact", takeoff_contact_baseline())
    # goal-stop 只数动作不看距离，会把学生在 3.4 m 外喊的 SAFE 当到达（2026-11-03 审计）。
    wp = {"x": 4.0, "y": 1.5}
    check("waypoint 半径内的 SAFE 算停稳",
          collect.stopped_at_waypoint({"pose_before": {"x": 4.1, "y": 1.45}}, wp, 0.5))
    check("半径外的 SAFE 不算停稳",
          not collect.stopped_at_waypoint({"pose_before": {"x": 1.0, "y": 0.0}}, wp, 0.5))
    check("缺位姿的 SAFE 不算停稳",
          not collect.stopped_at_waypoint({"pose_before": None}, wp, 0.5))
    check("没有 waypoint 时不算停稳",
          not collect.stopped_at_waypoint({"pose_before": {"x": 4.0, "y": 1.5}}, None, 0.5))

    print(f"test_collect: {passed} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
