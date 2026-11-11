#!/usr/bin/env python3
"""collect.py 纯函数自检：contact_fired 的坏消息、read_depth 的格式/stride。

不需要 Gazebo：contact_fired 用假进程，read_depth 用假 echo_json。
用法：python3 sim/tools/test_collect.py
"""
import base64
import contextlib
import hashlib
import io
import json
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
import nav_cond
import split_manifest as sm
import teacher

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
        def act(self, msg, prev_action=None):   # 与 StudentPolicy.act 同签名
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


class FakeSitl(FakeSil):
    """cf_sitl 执行器替身：截止从窗口起点算、关卡滞、带末速度。"""
    until_from_window_start = True
    stuck_detection = False
    name = "cf_sitl"

    def speed_mps(self):
        return 0.012


def cf_sitl_window():
    """collect_one + cf_sitl：observation 语义下派发时间戳 = 观测位姿戳；长窗口静止也不判卡滞。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.02))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}
    t = [10.02]

    def advance(_secs):
        t[0] = round(t[0] + 0.05, 3)
        pose.current = pose_msg(t[0])

    out = {}
    for window_from in ("observation", "dispatch"):
        sil = FakeSitl()
        pose.current = pose_msg(10.02)
        t[0] = 10.02
        with tempfile.TemporaryDirectory() as d, \
                patch.object(collect, "write_rgb_png", return_value=(True, None)), \
                patch.object(collect.time, "sleep", side_effect=advance):
            row, err = collect.collect_one(d, 0, 1.0, "forward", subs, FakePub(), sil=sil,
                                           window_from=window_from)
        if row is None:
            out[f"ran_{window_from}"] = False
            continue
        out[f"ran_{window_from}"] = err is None
        # 两种语义下 FakeSub 的观测位姿与派发位姿是同一帧（10.02 s），截止起点都应等于 action_start_ns
        out[f"anchor_{window_from}"] = sil.actions == [(collect.FORWARD_SLOW, row["action_start_ns"])]
        # 1 s 窗口、位姿不动：cf_sil 会判 stuck，cf_sitl 关掉卡滞只认 contact
        out[f"no_stuck_{window_from}"] = row["src"] is None and row["collision"] is False
        out[f"speed_{window_from}"] = row["speed_after_mps"] == 0.012
    # 对照：同样的静止 1 s 窗口，cf_sil 应判 stuck（证明豁免只对 cf_sitl 生效）
    sil = FakeSil()
    pose.current = pose_msg(10.02)
    t[0] = 10.02
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        row, err = collect.collect_one(d, 0, 1.0, "forward", subs, FakePub(), sil=sil)
    out["cf_sil_control_stuck"] = row is not None and row["src"] == "stuck" and row["speed_after_mps"] is None
    return out


def cf_sitl_error_text():
    """cf_sitl 执行器故障时错误文案点名 cf_sitl。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.02))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}
    sil = FakeSitl(error="FAILSAFE")
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)):
        row, err = collect.collect_one(d, 0, 0.2, "forward", subs, FakePub(), sil=sil)
    return row is None and "cf_sitl 执行器故障" in (err or "") and sil.zeroed > 0


def vehicle_checks():
    x3 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "worlds", "uav_rgb.sdf"), encoding="utf-8").read()
    e1 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "worlds", "uav_rgb_e1.sdf"), encoding="utf-8").read()
    out = {
        "x3_template": collect.vehicle_name(x3) == "x3",
        "e1_template": collect.vehicle_name(e1) == "e1_mini",
        "fallback": collect.vehicle_name("<sdf/>") == "x3",
        "pose_topic": collect.VEHICLES["e1_mini"]["pose_topic"] == "/model/e1_mini/pose"
                      and collect.VEHICLES["x3"]["pose_topic"] == collect.POSE_TOPIC,
        # x3 位姿缓冲保持历史 6 帧；e1_mini 200 Hz 位姿缓冲要盖过 RGB 到达延迟（≥ 0.3 s）
        "pose_buf": collect.VEHICLES["x3"]["pose_buf"] == 6
                    and collect.VEHICLES["e1_mini"]["pose_buf"] / 200.0 >= 0.3,
        "cf_sitl_in_dynamics": "cf_sitl" in collect.DYNAMICS
                               and collect.DYNAMICS_VEHICLE["cf_sitl"] == "e1_mini",
    }
    info = {"header": {"data": [{"key": "frame_id", "value": ["e1_mini::base_link::camera"]}]},
            "width": 320, "height": 240,
            "intrinsics": {"k": [277.0, 0, 160, 0, 277.0, 120, 0, 0, 1]}}
    out["camera_info_by_vehicle"] = (
        collect.camera_from_info(info) is None
        and collect.camera_from_info(info, collect.VEHICLES["e1_mini"]["rgb_frame_id"])["fx"] == 277.0)
    return out


def rl_delay_checks():
    """N1：teacher-rl 的 delay_remaining 按仿真时间算，不再恒等于 latency_s。"""
    dyn = collect.CfStab(latency_s=0.15)
    out = {"idle_zero": collect.rl_delay_remaining(dyn, 5_000_000_000) == 0.0}
    dyn.dispatch(collect.FORWARD_SLOW, 10_000_000_000)        # 生效于 10.15 s
    out["mid_delay"] = abs(collect.rl_delay_remaining(dyn, 10_050_000_000) - 0.10) < 1e-9
    out["after_hold"] = collect.rl_delay_remaining(dyn, 10_200_000_000) == 0.0   # 0.2 s 后：训练侧也是 0
    dyn.step(10_200_000_000, 0.0)                               # 推进后 pending 清空
    out["popped_zero"] = collect.rl_delay_remaining(dyn, 10_250_000_000) == 0.0 and not dyn.pending
    return out


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


# ---------------------------------------------------------------- 可开关采集功能（2026-11-07 DEV-DATA；默认行为不变）
# action_row 在 save_depth=False 时的旧字段集：新开关关着时必须一字不差（下游/审计按它读）。
ROW_KEYS_LEGACY = {"step", "timestamp_ns", "action_expert", "action_executed", "collision", "rgb_ok",
                   "frame", "src", "rgb_ts", "depth_ts", "depth_rgb_gap_ns", "pose_ts", "pose_rgb_gap_ns",
                   "dyaw_deg", "displacement_m", "infer_ms", "action_start_ns", "pose_before", "yaw_before"}
# W1-0：每步行多记 dispatch_lag_s（派发时的仿真戳减观测戳；不给 --action-latency 也记，只加键、排在最后）。
ROW_KEYS_DEFAULT = ROW_KEYS_LEGACY | {"dispatch_lag_s"}
# 成功线 = 最远柱 4.97 + 1.0 = 5.97
SCENE = {"pillars": [{"x": 1.0, "y": 0.3, "z": 1.5, "size": [0.4, 0.4, 3.0]},
                     {"x": 4.97, "y": -0.5, "z": 1.5, "size": [0.4, 0.4, 3.0]}]}


def fake_row(step, x, **over):
    """collect_one 成功返回的 row（主循环与 action_row 用到的全部键）。"""
    s = 10 + step
    r = {"contact_seq_end": 0, "step": step, "timestamp_ns": s * 10**9, "rgb_ts": s * 10**9,
         "action_expert": 1, "action_executed": 1, "collision": False,
         "pose_gt": {"x": x, "y": 0.0, "z": 2.1}, "pose_ts": s * 10**9 + 10**9,
         "pose_before": {"x": x - 0.25, "y": 0.0, "z": 2.1}, "yaw_before": 0.0, "rgb_ok": True,
         "frame": "rgb_%05d.png" % step, "depth_file": None, "depth_ts": s * 10**9,
         "pose_before_ts": s * 10**9 + 10**7, "action_start_ns": s * 10**9 + 10**7,
         "action_end_ns": s * 10**9 + 10**9, "displacement_m": 0.25, "dyaw_deg": 0.0, "sim_dt": 1.0,
         "speed_after_mps": None, "infer_ms": None, "action_wall_s": 1.0, "src": None,
         "tie_margin_steps": None, "executed_by": None,
         "dispatch_lag_s": 0.129, "late_dispatch": None,   # 与真 collect_one 的返回一致：late_dispatch 只有 --action-latency 才非 None
         "timing": {"io": 0.01, "window": 1.0, "block": 1.0, "tail": 0.01, "total": 1.02,
                    "dispatch_sim_dt": 1.0}}
    r.update(over)
    return r


class FixedRng:
    def __init__(self, v):
        self.v = v

    def random(self):
        return self.v


def shadow_teacher_step(beta, rng_value=0.0, tiebreak="legacy", prev_executed=None, **guard):
    """真 collect_one + 假学生（恒 FORWARD）+ 正前方一根柱：标签来自 A* 影子教师，执行按 β。

    guard：透传 heading_guard / net_yaw（航向护栏）。
    """
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.2))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}
    seen = []

    class Student:
        def act(self, msg, prev_action=None):
            seen.append(prev_action)
            return 1, 5.0, [0.0, 1.0, 0.0, 0.0]

    def advance(_secs):
        pose.current = pose_msg(11.2)

    pillars = [{"x": 0.8, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        row, err = collect.collect_one(d, 0, 1.0, "student", subs, FakePub(), student=Student(),
                                       pillars=pillars, tiebreak=tiebreak, beta=beta,
                                       rng=FixedRng(rng_value), prev_executed=prev_executed, **guard)
    return row, err, seen


def probs_step(policy, probs=None):
    """真 collect_one：policy == "student" 时接一个假学生（act 返回 (动作, 毫秒, probs)），其它策略不接学生。

    夹具同 shadow_teacher_step（一帧 RGB/深度，位姿推进 1 s），不给柱子：学生没有影子教师，教师走深度专家。返回 (row, err)。
    """
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.2))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}

    class Student:
        def act(self, msg, prev_action=None):
            return 1, 5.0, probs

    def advance(_secs):
        pose.current = pose_msg(11.2)

    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        return collect.collect_one(d, 0, 1.0, policy, subs, FakePub(),
                                   student=Student() if policy == "student" else None)


def as_actions_line(row):
    """collect_one 的行经 action_row 写进 actions.jsonl 的样子（补上主循环加的两个间隔键）。"""
    return collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})


def probs_checks():
    """W1-3：学生每步的四类概率落进 actions.jsonl 行（4 位小数）；其它策略的行逐键不变。"""
    out = {}
    raw = [0.1234567, 0.5000004, 0.2000009, 0.176542]            # 推理进程给的原始概率（和为 1，不止 4 位小数）
    row, err = probs_step("student", raw)
    out["collect_one 学生行有 probs：4 个数，保留 4 位小数"] = (
        err is None and row.get("probs") == [0.1235, 0.5, 0.2, 0.1765]
        and all(isinstance(p, float) and round(p, 4) == p for p in row.get("probs", ())))
    out["学生行的 probs 之和为 1（±1e-3）"] = err is None and abs(sum(row.get("probs", ())) - 1.0) <= 1e-3
    line = as_actions_line(row)
    keys = list(line)
    out["action_row 把 probs 写进 actions.jsonl 行；W1-0 的 dispatch_lag_s 仍是最后一个键"] = (
        line.get("probs") is not None and line["probs"] == row["probs"] and keys[-1] == "dispatch_lag_s")
    out["学生行只比默认行多 probs"] = set(line) == ROW_KEYS_DEFAULT | {"probs"}
    # 边界：学生没给 probs（None）→ 行里没有这个键，步照常成功；给的不是数字 → 派发前判本步失败，不留到动作发出之后
    row_none, err_none = probs_step("student", None)
    out["学生给 None：行里没有 probs 键，步成功"] = err_none is None and "probs" not in row_none
    row_bad, err_bad = probs_step("student", ["x", 1.0, 0.0, 0.0])
    out["probs 不是数字：本步失败（学生推理失败）"] = row_bad is None and "学生推理失败" in (err_bad or "")
    # 零样本基线（baselines/baseline_infer.py）经 --policy student --infer-bin 走同一条学生分支；它没有四类概率，给 one-hot，照常落盘
    row_oh, err_oh = probs_step("student", [0.0, 1.0, 0.0, 0.0])
    out["零样本基线的 one-hot probs 照常落盘（走同一条学生分支）"] = (
        err_oh is None and row_oh.get("probs") == [0.0, 1.0, 0.0, 0.0]
        and as_actions_line(row_oh).get("probs") == [0.0, 1.0, 0.0, 0.0])
    # probs 是学生自己的输出，与谁执行无关：β=1 时执行教师动作，probs 仍是学生的
    row_b, err_b, _ = shadow_teacher_step(beta=1.0, rng_value=0.3, prev_executed=3)
    out["probs 不随执行者变（β=1 执行教师动作，probs 仍是学生的）"] = (
        err_b is None and row_b["executed_by"] == "expert" and row_b.get("probs") == [0.0, 1.0, 0.0, 0.0])
    # 其它策略：行里没有 probs 键（collect_one 的行与 action_row 的行都没有），键集合与默认行一字不差
    for policy in ("teacher", "forward"):
        row_o, err_o = probs_step(policy)
        out[f"{policy} 的行没有 probs 键"] = (
            err_o is None and "probs" not in row_o and "probs" not in as_actions_line(row_o)
            and set(as_actions_line(row_o)) == ROW_KEYS_DEFAULT)
    # 主循环落盘：student 的 actions.jsonl 每行带 probs，教师的每行都没有
    rows_s = [fake_row(i, 1.0 + 0.25 * i, executed_by="student", probs=[0.1, 0.6, 0.2, 0.1]) for i in range(3)]
    _, acts_s, code_s, _ = run_main(["--policy", "student", "--student-ckpt", "CKPT"], rows_s,
                                    student_cls=FakeStudentPolicy)
    out["主循环：student 的 actions.jsonl 每行带 probs"] = (
        code_s is None and len(acts_s) == 3 and all(a.get("probs") == [0.1, 0.6, 0.2, 0.1] for a in acts_s)
        and all(set(a) == ROW_KEYS_DEFAULT | {"executed_by", "probs"} for a in acts_s))
    _, acts_t, code_t, _ = run_main([], [fake_row(i, 1.0 + 0.25 * i) for i in range(3)])
    out["主循环：教师的 actions.jsonl 没有 probs"] = (
        code_t is None and len(acts_t) == 3 and all("probs" not in a and set(a) == ROW_KEYS_DEFAULT for a in acts_t))
    return out


class MainPub(FakePub):
    mode = "fake"


def run_main(extra_argv, rows, scene=SCENE, student_cls=None, steps=None):
    """跑 collect.main() 的主循环与 manifest 收尾，Gazebo 全部替身，collect_one 按 rows 依次返回。

    返回 (manifest, actions.jsonl 行, SystemExit code, ckpt sha256)；参数校验失败时 manifest 为 None。
    """
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "ep")
        meta = os.path.join(d, "scene.json")
        with open(meta, "w") as f:
            json.dump(scene, f)
        ckpt = os.path.join(d, "ckpt.bin")
        with open(ckpt, "wb") as f:
            f.write(b"not-a-real-checkpoint")
        ckpt_sha = hashlib.sha256(b"not-a-real-checkpoint").hexdigest()
        argv = ["collect.py", "--out", out, "--scene-meta", meta, "--split", sm.split_for_scene(scene),
                "--steps", str(steps if steps is not None else len(rows))]
        argv += [a.replace("CKPT", ckpt) for a in extra_argv]
        subs = {k: FakeSub(latest=pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}
        it = iter(rows)
        code = None
        with contextlib.ExitStack() as st, contextlib.redirect_stderr(io.StringIO()):
            st.enter_context(patch.object(sys, "argv", argv))
            st.enter_context(patch.object(collect, "open_subs", return_value=subs))
            st.enter_context(patch.object(collect, "wait_ready", return_value=None))
            st.enter_context(patch.object(collect, "read_camera_info", return_value=None))
            st.enter_context(patch.object(collect.gz_io, "open_twist_publisher", return_value=MainPub()))
            st.enter_context(patch.object(collect, "take_off", return_value=0))
            st.enter_context(patch.object(collect, "stop_executor"))
            st.enter_context(patch.object(collect, "close_subs"))
            st.enter_context(patch.object(collect, "collect_one",
                                          side_effect=lambda *a, **kw: (next(it), None)))
            if student_cls is not None:
                st.enter_context(patch("student.StudentPolicy", student_cls))
            try:
                collect.main()
            except SystemExit as e:
                code = e.code
        man_path = os.path.join(out, "episode_manifest.json")
        if not os.path.isfile(man_path):
            return None, [], code, ckpt_sha
        with open(man_path) as f:
            manifest = json.load(f)
        with open(os.path.join(out, "actions.jsonl")) as f:
            actions = [json.loads(line) for line in f if line.strip()]
        return manifest, actions, code, ckpt_sha


class FakeStudentPolicy:
    def __init__(self, ckpt, infer_bin, text_ids=None):
        pass

    def close(self):
        pass


def active_rules_match(man):
    """collect.main 跑完后 teacher 模块里生效的规则档（B2/B3）== manifest.teacher_version 对应的规则档。"""
    return teacher.rules_for_version(man["teacher_version"]) == {
        "goal_line": teacher.GOAL_LINE_AFTER_PILLARS, "retry": teacher.ASTAR_RETRY_CLEARANCES}


def switch_checks():
    """--stop-at-goal / --policy student 影子教师 / --tiebreak 三个开关，以及关着时与旧版一致。"""
    out = {}
    # 1) 成功线判定（与 eval 同口径：最远柱 x + 1.0，看 oracle pose_gt.x）
    gx = collect.goal_line_x(SCENE["pillars"])
    out["goal_line_x = 最远柱+1.0"] = abs(gx - 5.97) < 1e-9
    out["过线判定看 pose_gt.x ≥ 线"] = (collect.crossed_goal_line({"pose_gt": {"x": 5.97}}, gx)
                                       and not collect.crossed_goal_line({"pose_gt": {"x": 5.96}}, gx)
                                       and not collect.crossed_goal_line({"pose_gt": None}, gx))
    # 2) 主循环：真值 x 5.5 → 5.9 → 6.0（过线）→ 6.3；开了开关第 3 步后结束且算完整集，关着跑满 4 步
    rows = [fake_row(i, x) for i, x in enumerate((5.5, 5.9, 6.0, 6.3))]
    man, acts, code, _ = run_main(["--stop-at-goal"], rows)
    out["--stop-at-goal 过线即止"] = (code is None and man is not None and man["num_steps"] == 3
                                    and len(acts) == 3 and man["stopped_on_goal"] is True
                                    and abs(man["goal_line_x"] - 5.97) < 1e-9
                                    and man["validity"]["episode_executed"] is True)
    man, acts, code, _ = run_main([], rows)
    out["默认不截断，manifest goal_line_x=None"] = (code is None and man["num_steps"] == 4
                                                  and man["stopped_on_goal"] is False
                                                  and man["goal_line_x"] is None
                                                  and man["teacher_tiebreak"] == "legacy"
                                                  and man["teacher"] == "plan_action" and man["planner"] == "astar"
                                                  and "executed_by" not in man and "beta" not in man
                                                  and all(set(a) == ROW_KEYS_DEFAULT for a in acts))
    out["A* 采集：生效规则 = manifest.teacher_version 的规则（v7：B2 开、A* 重试开）"] = (
        man["teacher_version"] == teacher.TEACHER_VERSION and active_rules_match(man)
        and teacher.GOAL_LINE_AFTER_PILLARS and teacher.ASTAR_RETRY_CLEARANCES == (0.30, 0.20))
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()), \
            patch.object(sys, "argv", ["collect.py", "--out", os.path.join(d, "x"), "--stop-at-goal"]):
        try:
            collect.main()
            out["--stop-at-goal 没有 scene-meta 拒绝"] = False
        except SystemExit as e:
            out["--stop-at-goal 没有 scene-meta 拒绝"] = e.code == 2
    wp_scene = SCENE | {"waypoint": {"x": 3.0, "y": 0.5}, "stop_radius": 0.5}
    out["--stop-at-goal 与 waypoint 任务互斥"] = run_main(["--stop-at-goal"], rows, scene=wp_scene)[2] == 2
    # 3) 影子教师：学生恒 FORWARD，正前方 0.8 m 有柱 → 教师标签是转向；β=0 学生执行，β=1 教师执行
    row, err, seen = shadow_teacher_step(beta=0.0)
    out["影子教师：标签=A*，执行=学生"] = (err is None and row["action_expert"] in (collect.YAW_LEFT, collect.YAW_RIGHT)
                                        and row["action_executed"] == collect.FORWARD_SLOW
                                        and row["executed_by"] == "student" and row["tie_margin_steps"] is None
                                        and seen == [None])
    row_b, err_b, seen_b = shadow_teacher_step(beta=1.0, rng_value=0.3, prev_executed=3)
    out["影子教师：β=1 执行教师动作，prev_executed 喂给学生"] = (
        err_b is None and row_b["action_executed"] == row_b["action_expert"] == row["action_expert"]
        and row_b["executed_by"] == "expert" and seen_b == [3])
    row_v, err_v, _ = shadow_teacher_step(beta=0.0, tiebreak="visible")
    out["--tiebreak visible 每步记 tie_margin_steps"] = (err_v is None
                                                        and isinstance(row_v["tie_margin_steps"], float))
    ar = collect.action_row(row | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})
    # W1-3：student 的行另有 probs（学生四类概率，见 probs_checks）；教师行仍是默认字段集
    out["action_row 只在开关开着时多字段"] = (set(ar) == ROW_KEYS_DEFAULT | {"executed_by", "probs"}
                                           and set(collect.action_row(fake_row(0, 1.0) | {
                                               "depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0})) == ROW_KEYS_DEFAULT)
    # 4) student 模式 manifest：字段名与 sim/3dgs/collect_3dgs.py 一致
    rows_s = [fake_row(i, 1.0 + 0.25 * i, action_expert=2, action_executed=1 if i != 1 else 2,
                       executed_by="expert" if i == 1 else "student") for i in range(3)]
    man, acts, code, sha = run_main(["--policy", "student", "--student-ckpt", "CKPT", "--beta", "0.5",
                                     "--dagger-round", "2", "--tiebreak", "visible"],
                                    rows_s, student_cls=FakeStudentPolicy)
    out["student manifest：plan_action/astar + DAgger 字段"] = (
        code is None and man["teacher"] == "plan_action" and man["planner"] == "astar"
        and man["executed_by"] == "student" and man["beta"] == 0.5 and man["dagger_round"] == 2
        and man["student_ckpt_sha256"] == sha and man["student_ckpt"].endswith("ckpt.bin")
        and man["n_expert_executed"] == 1 and man["teacher_tiebreak"] == "visible"
        and man["teacher_version"] is not None
        and [a["executed_by"] for a in acts] == ["student", "expert", "student"])
    out["--beta 只配 student"] = run_main(["--beta", "0.5"], rows)[2] == 2
    out["--tiebreak visible 只配 A* 教师"] = run_main(["--tiebreak", "visible", "--policy", "forward"], rows)[2] == 2
    # 5) --teacher-planner costmap（BRIEF_ALGO §1）：manifest planner/teacher_version/两个显式参数；行多 q_costs
    rows_c = [fake_row(i, 1.0 + 0.25 * i, q_costs=[17.0, 16.0, 16.5, 18.0]) for i in range(3)]
    man_c, acts_c, code_c, _ = run_main(["--teacher-planner", "costmap"], rows_c)
    out["costmap manifest：planner/teacher_version/tol/前瞻"] = (
        code_c is None and man_c["planner"] == "costmap" and man_c["teacher"] == "plan_action"
        and man_c["teacher_version"] == collect.TEACHER_COST_VERSION
        and man_c["costmap_tol_deg"] == 25.0 and man_c["costmap_lookahead_steps"] == 1
        and man_c["experimental_policy"] is False and man_c["teacher_tiebreak"] == "legacy")
    out["costmap 采集：生效规则 = cost 版本的规则（B2 开、A* 不重试）"] = (
        active_rules_match(man_c) and teacher.GOAL_LINE_AFTER_PILLARS and teacher.ASTAR_RETRY_CLEARANCES == ())
    out["costmap 行多 q_costs（4 浮点）"] = all(set(a) == ROW_KEYS_DEFAULT | {"q_costs"} and len(a["q_costs"]) == 4
                                              for a in acts_c)
    man_c2, _, code_c2, _ = run_main(["--teacher-planner", "costmap", "--costmap-tol", "0", "--costmap-lookahead-steps", "2"], rows_c)
    out["costmap 变体参数进 manifest"] = (code_c2 is None and man_c2["costmap_tol_deg"] == 0.0
                                         and man_c2["costmap_lookahead_steps"] == 2)
    out["legacy manifest 不带 costmap 字段"] = "costmap_tol_deg" not in man and "costmap_lookahead_steps" not in man
    out["costmap 只配 --policy teacher"] = run_main(["--teacher-planner", "costmap", "--policy", "forward"], rows_c)[2] == 2
    out["costmap 参数只配 costmap"] = run_main(["--costmap-tol", "0"], rows)[2] == 2
    out["costmap 前瞻 ≥ 1"] = run_main(["--teacher-planner", "costmap", "--costmap-lookahead-steps", "0"], rows_c)[2] == 2
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(io.StringIO()), \
            patch.object(sys, "argv", ["collect.py", "--out", os.path.join(d, "x"), "--teacher-planner", "costmap"]):
        try:
            collect.main()
            out["costmap 没有 scene-meta 拒绝"] = False
        except SystemExit as e:
            out["costmap 没有 scene-meta 拒绝"] = e.code == 2
    # 6) 真 collect_one 走 costmap 分支：正前方 0.8 m 一根柱 → 标签是转向，q_costs 里 FWD = 撞柱代价
    row_c, err_c = costmap_teacher_step()
    out["collect_one costmap：标签=argmin Q，q_costs 落 row"] = (
        err_c is None and row_c["action_expert"] in (collect.YAW_LEFT, collect.YAW_RIGHT)
        and row_c["action_executed"] == row_c["action_expert"] and len(row_c["q_costs"]) == 4
        and row_c["q_costs"][collect.FORWARD_SLOW] == max(row_c["q_costs"])
        and row_c["tie_margin_steps"] is None and row_c["executed_by"] is None)
    return out


def costmap_teacher_step():
    """真 collect_one + --teacher-planner costmap（costmap_opts）+ 正前方一根柱。"""
    w, h = 4, 6
    rgb = image_msg(w, h, 10)
    pose = ClockPoseSub(pose_msg(10.2))
    subs = {"rgb": FakeSub(after=rgb), "depth": FakeSub(frames=[(1, _fresh_depth(w, h))]),
            "pose": pose, "contact": FakeSub()}

    def advance(_secs):
        pose.current = pose_msg(11.2)

    pillars = [{"x": 0.8, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]
    with tempfile.TemporaryDirectory() as d, \
            patch.object(collect, "write_rgb_png", return_value=(True, None)), \
            patch.object(collect.time, "sleep", side_effect=advance):
        row, err = collect.collect_one(d, 0, 1.0, "teacher", subs, FakePub(), pillars=pillars,
                                       costmap_opts={"tol": 25.0, "lookahead_steps": 1})
    return row, err


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
    for k, v in cf_sitl_window().items():
        check(f"cf_sitl 采集窗口: {k}", v)
    check("cf_sitl 执行器故障 -> 文案点名 cf_sitl", cf_sitl_error_text())
    for k, v in vehicle_checks().items():
        check(f"载具: {k}", v)
    for k, v in rl_delay_checks().items():
        check(f"N1 delay_remaining: {k}", v)
    check("编排器矩阵含 cf_sitl", "cf_sitl" in cp.DYNAMICS_CHOICES
          and cp.DYNAMICS_VEHICLE["cf_sitl"] == "e1_mini")
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
    for k, v in switch_checks().items():
        check(f"开关: {k}", v)
    for k, v in probs_checks().items():
        check(f"学生概率: {k}", v)

    # 航向护栏（--heading-guard）：|已执行左转数 − 右转数| 达到阈值就朝回转一步；0 = 关闭。
    check("护栏关闭时不干预", collect.heading_guard_action(5, 0) is None)
    check("护栏未到阈值不干预",
          collect.heading_guard_action(1, 2) is None and collect.heading_guard_action(-1, 2) is None)
    check("护栏：左偏 2 步朝右回", collect.heading_guard_action(2, 2) == collect.YAW_RIGHT)
    check("护栏：右偏 3 步朝左回", collect.heading_guard_action(-3, 2) == collect.YAW_LEFT)
    row_g, err_g, _ = shadow_teacher_step(beta=0.0, heading_guard=2, net_yaw=2)
    check("护栏触发：覆盖学生的 FORWARD 并留痕",
          err_g is None and row_g["action_executed"] == collect.YAW_RIGHT
          and row_g["executed_by"] == "heading_guard")
    row_n, err_n, _ = shadow_teacher_step(beta=0.0, heading_guard=2, net_yaw=1)
    check("护栏未触发：执行学生动作",
          err_n is None and row_n["action_executed"] == 1 and row_n["executed_by"] == "student")

    # 航向码（--student-cond heading-code，W1-9）：传给学生的值 = nav_cond.heading_code(net_yaw)；默认与改动前逐位相同。
    # 主循环 / 行字段 / manifest / eval 透传的接线在 test_nav_cond_wiring.py，这里锁 collect_one 这一层。
    for n in (-3, -2, -1, 0, 1, 2, 3):
        row_c, err_c, seen_c = shadow_teacher_step(beta=0.0, prev_executed=collect.YAW_LEFT, net_yaw=n,
                                                   student_cond="heading-code")
        check(f"航向码：net_yaw={n} → 传给学生 {nav_cond.heading_code(n)}（与 prev_executed 无关）",
              err_c is None and seen_c == [nav_cond.heading_code(n)] and row_c["cond_code"] == nav_cond.heading_code(n))
    row_p, err_p, seen_p = shadow_teacher_step(beta=0.0, prev_executed=collect.YAW_RIGHT, net_yaw=5)
    check("默认：传给学生的仍是 prev_executed（net_yaw 不起作用），行里没有 cond_code",
          err_p is None and seen_p == [collect.YAW_RIGHT] and row_p.get("cond_code") is None
          and "cond_code" not in collect.action_row(row_p | {"depth_rgb_gap_ns": 0, "pose_rgb_gap_ns": 0}))

    print(f"test_collect: {passed} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
