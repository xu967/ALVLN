#!/usr/bin/env python3
"""cf_sitl_exec 的离线单测（不需要 Gazebo）：dynamics_profile = cf_sitl_v1:<参数 value 规范化 sha8>:lib<N>。

AUDIT_SIM 2026-11-09 #1：profile 以前只是 e1_mini.json 原始字节的 sha8，改注释就换 profile、换库不换 profile。
现在参数 sha 只算各项的 value，库版本从插件 state[IDX_LIB_VERSION] 读。运行：
  .venv-tools/bin/python -m unittest sim.tools.test_cf_sitl_exec
"""
import json
import os
import re
import subprocess
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_sitl_exec as ce  # noqa: E402

PROFILE_RE = re.compile(r"^cf_sitl_v1:[0-9a-f]{8}:lib\d+$")


def _params(obj, lib_version=None):
    raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    return ce.CfSitlParams("x.json", raw, obj, lib_version=lib_version)


BASE = {"calibrated": False,
        "mass_flight": {"value": 0.102, "unit": "kg", "basis": "a"},
        "executor": {"value": {"vel_kp": 0.22, "vel_ki": 0.30}, "basis": "b"}}


class TestProfileFingerprint(unittest.TestCase):
    def test_comment_change_keeps_profile_value_change_breaks_it(self):
        p0 = _params(BASE, 31)
        self.assertRegex(p0.profile(), PROFILE_RE)
        self.assertTrue(p0.profile().endswith(":lib31"))
        commented = json.loads(json.dumps(BASE))
        commented["executor"]["basis"] = "改了说明文字"
        commented["mass_flight"]["unit"] = "千克"
        p1 = _params(commented, 31)
        self.assertNotEqual(p1.sha256, p0.sha256)             # 文件字节变了
        self.assertEqual(p1.profile(), p0.profile())           # 只算 value：profile 不变
        tuned = json.loads(json.dumps(BASE))
        tuned["executor"]["value"]["vel_ki"] = 0.15
        self.assertNotEqual(_params(tuned, 31).profile(), p0.profile())
        self.assertNotEqual(_params(BASE, 32).profile(), p0.profile())     # 库版本变 → 变
        # 旧格式 cf_sitl_v1:<原始字节 sha8> 与新格式不可能相等（长度就不同）
        self.assertNotEqual(p0.profile(), f"{ce.PROFILE_PREFIX}:{p0.sha256[:8]}")

    def test_unknown_lib_version_is_visible_not_silent(self):
        p = _params(BASE)
        self.assertIsNone(p.lib_version)
        self.assertEqual(p.profile(), f"{ce.PROFILE_PREFIX}:{p.values_sha256[:8]}:lib?")   # start() 前只供日志
        self.assertIsNone(p.manifest()["lib_version"])
        p.lib_version = 31
        self.assertRegex(p.profile(), PROFILE_RE)
        self.assertEqual(p.manifest()["lib_version"], 31)

    def test_values_sha_is_canonical(self):
        a = {"b": {"value": [1, 2.5, {"y": 1, "x": 2}]}, "a": {"value": "s"}, "calibrated": True}
        b = {"a": {"value": "s", "basis": "zzz"}, "b": {"value": [1, 2.5, {"x": 2, "y": 1}]}}
        self.assertEqual(ce.params_values_sha256(a), ce.params_values_sha256(b))   # 键序、注释、非 value 顶层项都不算
        self.assertEqual(len(ce.params_values_sha256(a)), 64)

    def test_real_params_file(self):
        p = ce.CfSitlParams.from_json(ce.DEFAULT_PARAMS)
        p.lib_version = 31
        self.assertRegex(p.profile(), PROFILE_RE)
        self.assertEqual(p.profile(), ce.format_profile(p.values_sha256, 31))

    def test_state_layout_has_lib_version_after_sticks(self):
        self.assertEqual(ce.IDX_LIB_VERSION, ce.IDX_STICKS + 4)
        self.assertEqual(ce.STATE_FIELDS, ce.IDX_LIB_VERSION + 1)
        ex = ce.CfSitlExecutor(_params(BASE), hold_s=0.5)

        class Sub:
            def __init__(self, n):
                self.n = n

            def latest(self):
                return {"data": [0.0] * self.n}

            def healthy(self):
                return True

        ex._sub = Sub(ce.STATE_FIELDS)
        self.assertEqual(len(ex.state()), ce.STATE_FIELDS)
        ex._sub = Sub(ce.STATE_FIELDS - 1)                     # 旧插件（没有 lib_version）：明确报错，不是静默等超时
        self.assertIsNone(ex.state())
        self.assertIn("重建", ex.error())


class FakeSim:
    """假 state 源 + 假墙钟，顶替 cf_sitl_exec 的 time、gz_io.TopicSub 与发布端。

    每 sleep(dt)：墙钟走 dt，gz 仿真走 rtf·dt。arm_takeoff 后 ready_after_s 仿真秒进 READY（None = 一直 TAKEOFF）；
    failsafe_once：第一次起飞立刻 FAILSAFE（z=0.3），下一次 reset 后 fall_s 仿真秒落地；no_state：一帧 state 都不发。
    """
    mode = "fake"

    def __init__(self, rtf, ready_after_s=None, failsafe_once=False, fall_s=0.0, no_state=False):
        self.rtf, self.ready_after_s, self.failsafe_once, self.fall_s, self.no_state = (
            rtf, ready_after_s, failsafe_once, fall_s, no_state)
        self.wall = self.sim = self.t_sitl = self.z = 0.0
        self.exec, self.arm_sim, self.fall_at = ce.EXEC_DISARMED, None, None

    def monotonic(self):
        return self.wall

    def sleep(self, dt):
        self.wall += dt
        self.sim += self.rtf * dt
        self.t_sitl += self.rtf * dt
        if self.fall_at is not None and self.sim >= self.fall_at:
            self.z, self.fall_at = 0.0, None
        if self.exec == ce.EXEC_TAKEOFF:
            if self.failsafe_once:
                self.exec, self.failsafe_once, self.z = ce.EXEC_FAILSAFE, False, 0.3
            elif self.ready_after_s is not None and self.sim - self.arm_sim >= self.ready_after_s:
                self.exec, self.z = ce.EXEC_READY, 2.0

    def latest(self):
        if self.no_state:
            return None
        st = [0.0] * ce.STATE_FIELDS
        st[ce.IDX_T], st[ce.IDX_EXEC], st[ce.IDX_SIM_T] = self.t_sitl, float(self.exec), self.sim
        st[ce.IDX_HEIGHT] = st[ce.IDX_POS + 2] = self.z
        st[ce.IDX_LIB_VERSION] = 31.0
        return {"data": st}

    def healthy(self):
        return True

    def close(self):
        pass

    def action(self, _a, _u):
        pass

    def cmd(self, text):
        if text.startswith("reset"):
            self.fall_at = self.sim + self.fall_s if self.z > 0.0 else None
            self.exec, self.t_sitl = ce.EXEC_DISARMED, 0.0
        elif text.startswith("arm_takeoff"):
            self.exec, self.arm_sim = ce.EXEC_TAKEOFF, self.sim


def _start(sim, run=None):
    ex = ce.CfSitlExecutor(_params(BASE), hold_s=1.0, log=lambda _m: None)
    with mock.patch.object(ce, "time", sim), mock.patch.object(ce.gz_io, "TopicSub", lambda *a, **k: sim), \
            mock.patch.object(ce, "open_publisher", lambda *a, **k: sim), \
            mock.patch.object(ce.subprocess, "run", run or subprocess.run):
        ex.start()
    return ex


class TestTakeoffWaitsOnSimTime(unittest.TestCase):
    """起飞等待按 state 的 gz 仿真时间计，墙钟只兜底：多个 gz + 三视角录像满载时 RTF 可低到 0.08，
    同样的起飞要多花十几倍墙钟，按墙钟判超时会把正常起飞判成失败集。"""

    def test_slow_wall_clock_alone_is_not_a_timeout(self):
        sim = FakeSim(rtf=2.0 / 90.0, ready_after_s=2.5)    # 墙钟 90 s 仿真才走 2 s；仿真 2.5 s 后 READY
        _start(sim)
        self.assertEqual(sim.exec, ce.EXEC_READY)
        self.assertGreater(sim.wall, 90.0)

    def test_still_takeoff_after_sim_limit_times_out(self):
        sim = FakeSim(rtf=1.0)                              # 一直 TAKEOFF
        with self.assertRaises(ce.WaitTimeout) as cm:
            _start(sim)
        msg = str(cm.exception)
        self.assertIn("仿真", msg)
        self.assertIn("exec_state=TAKEOFF", msg)
        self.assertLessEqual(sim.sim - sim.arm_sim, ce.READY_SIM_S + 0.1)   # 仿真到上限就报，不等墙钟

    def test_wall_cap_bounds_a_crawling_sim(self):
        sim = FakeSim(rtf=0.01)                             # 仿真在走但极慢：仿真上限到不了，墙钟兜底
        with self.assertRaises(ce.WaitTimeout) as cm:
            _start(sim)
        self.assertIn("墙钟", str(cm.exception))
        self.assertGreaterEqual(sim.wall, ce.SIM_WAIT_WALL_S)
        self.assertLess(sim.sim - sim.arm_sim, ce.READY_SIM_S)

    def test_stale_state_still_detected_on_wall_clock(self):
        sim = FakeSim(rtf=0.0)                              # state 戳不再前进：STATE_STALE_S 墙钟判停更，不变
        with self.assertRaises(RuntimeError) as cm:
            _start(sim)
        self.assertIn("停更", str(cm.exception))
        self.assertLess(sim.wall, 2 * ce.STATE_STALE_S)

    def test_settle_after_failsafe_waits_on_sim_time(self):
        sim = FakeSim(rtf=0.2, ready_after_s=7.0, failsafe_once=True, fall_s=0.8)   # 落地 0.8 s 仿真 = 4 s 墙钟
        ex = _start(sim)
        self.assertEqual(ex.takeoff_failsafes, 1)
        self.assertEqual(sim.exec, ce.EXEC_READY)


class TestNoStateNamesWorldStep(unittest.TestCase):
    """插件只接受 1 ms 物理步长，不符时只在 gz 日志 gzerr 一次、不发 state；首帧超时的报错要把世界步长说出来。"""

    def _fail(self, stats_stdout):
        calls = []

        def run(cmd, **_kw):
            calls.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=stats_stdout, stderr="")

        with self.assertRaises(ce.WaitTimeout) as cm:
            _start(FakeSim(rtf=1.0, no_state=True), run)
        return str(cm.exception), calls

    def test_wrong_world_step_is_named(self):
        # X3 模板（0.004 s）的世界配 E1 插件；gz topic -e -t /stats --json-output 一行（int64 是字符串）
        msg, calls = self._fail('{"simTime": {"sec": "12", "nsec": 500000000}, "iterations": "3125"}\n')
        self.assertIn("0.004", msg)
        self.assertIn("max_step_size", msg)
        self.assertIn("/stats", calls[0])

    def test_right_world_step_points_to_gz_log(self):
        msg, _ = self._fail('{"simTime": {"sec": "12", "nsec": 500000000}, "iterations": "12500"}\n')
        self.assertIn("无 state", msg)
        self.assertIn("[CfSitlSystem]", msg)
        self.assertNotIn("max_step_size", msg)

    def test_unreadable_stats_keeps_original_error(self):
        msg, _ = self._fail("")
        self.assertIn("无 state", msg)
        self.assertIn("[CfSitlSystem]", msg)


if __name__ == "__main__":
    unittest.main()
