#!/usr/bin/env python3
"""CfSitlDynamics（E1 MINI 固件在环，经 cf_sitl_proc 子进程）与 viewmat_from_quat 的测试。

用真进程：$CF_SITL_PROC（缺省 $DATA_ROOT/uav_rgb/tmp/build-cf-sitl/cf_sitl_proc）不存在时
进程相关用例 skip，纯数学用例照跑。运行：
  $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_dynamics_sitl.py
"""
import importlib
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
dyn = importlib.import_module("sim.3dgs.dynamics")
scene_mod = importlib.import_module("sim.3dgs.scene")

HAVE_PROC = os.path.isfile(dyn.CF_SITL_PROC)
SCENE = {"spawn": {"x": -14.0, "y": 0.0, "z": 2.0, "yaw_deg": 0.0}, "ground_z": 0.0}


def _yaw_quat(yaw_deg):
    h = math.radians(yaw_deg) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


class TestViewmatFromQuat(unittest.TestCase):
    def test_yaw_only_matches_viewmat_from_pose(self):
        for yaw in (0.0, 37.0, -120.0, 179.0):
            want = scene_mod.viewmat_from_pose(1.0, -2.0, 2.0, yaw)
            got = scene_mod.viewmat_from_quat((1.0, -2.0, 2.0), _yaw_quat(yaw))
            np.testing.assert_allclose(got, want, atol=1e-9)

    def test_mount_offset_and_pitch(self):
        mount = {"xyz": [0.045, 0.0, 0.015], "pitch_deg": 0.0}
        v = scene_mod.viewmat_from_quat((0.0, 0.0, 2.0), _yaw_quat(90.0), mount)
        r, t = v[:3, :3], v[:3, 3]
        center = -r.T @ t
        np.testing.assert_allclose(center, [0.0, 0.045, 2.015], atol=1e-9)   # 机头朝 +y，相机在机头前
        # 安装俯仰 +10°（抬头）等价于 viewmat_from_pose 的 pitch=+10
        up = scene_mod.viewmat_from_quat((0, 0, 2), _yaw_quat(0), {"xyz": [0, 0, 0], "pitch_deg": 10.0})
        np.testing.assert_allclose(up, scene_mod.viewmat_from_pose(0, 0, 2, 0, 10.0), atol=1e-9)

    def test_body_pitch_down_looks_down(self):
        # 机体绕 +y（左）转 +15° = 低头：相机前向 z 分量为负
        h = math.radians(15.0) / 2.0
        v = scene_mod.viewmat_from_quat((0, 0, 2), (math.cos(h), 0.0, math.sin(h), 0.0))
        fwd = v[2, :3]
        self.assertAlmostEqual(fwd[2], -math.sin(math.radians(15.0)), places=9)
        self.assertAlmostEqual(fwd[0], math.cos(math.radians(15.0)), places=9)
        # nav_to_world 平移：相机中心跟着走
        m = np.eye(4)
        m[:3, 3] = [10.0, 0.0, -1.0]
        v2 = scene_mod.viewmat_from_quat((0, 0, 2), _yaw_quat(0), None, m)
        np.testing.assert_allclose(-v2[:3, :3].T @ v2[:3, 3], [10.0, 0.0, 1.0], atol=1e-9)


class TestCfSitlParams(unittest.TestCase):
    def test_manifest(self):
        p = dyn.CfSitlParams.from_json(lib_version=31)     # 不起进程：库版本直接给
        m = p.manifest()
        self.assertEqual((m["dynamics"], m["vehicle"]), ("cf_sitl", "e1_mini"))
        self.assertEqual(len(m["params_sha256"]), 64)
        self.assertFalse(m["calibrated"])
        self.assertEqual(m["camera_mount"]["xyz"], [0.045, 0.0, 0.015])
        self.assertRegex(p.profile(), r"^cf_sitl_v1:[0-9a-f]{8}:lib31$")
        self.assertEqual((m["lib_version"], m["params_values_sha256"]), (31, p.values_sha256))

    def test_profile_fingerprint(self):
        """AUDIT_SIM #1：参数 sha 只算 value（改 basis 注释不变），改 vel_ki 变，库版本变也变；
        与 Gazebo 侧 sim/tools/cf_sitl_exec.py 同一算法，同一参数文件两条链路 profile 一致。"""
        import json
        import importlib
        ce = importlib.import_module("cf_sitl_exec")
        p0 = dyn.CfSitlParams.from_json(lib_version=31)
        with open(dyn.E1_MINI_PARAMS_PATH, encoding="utf-8") as f:
            obj = json.load(f)
        with tempfile.TemporaryDirectory() as tmp:
            def variant(name, mutate):
                o = json.loads(json.dumps(obj))
                mutate(o)
                path = os.path.join(tmp, name)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(o, f, ensure_ascii=False, indent=1)
                return dyn.CfSitlParams.from_json(path, lib_version=31)
            p_doc = variant("basis.json", lambda o: o["executor"].__setitem__("basis", "换个说法"))
            self.assertNotEqual(p_doc.sha256, p0.sha256)
            self.assertEqual(p_doc.profile(), p0.profile())
            p_ki = variant("ki.json", lambda o: o["executor"]["value"].__setitem__("vel_ki", 0.15))
            self.assertNotEqual(p_ki.profile(), p0.profile())
        self.assertNotEqual(dyn.CfSitlParams.from_json(lib_version=32).profile(), p0.profile())
        gz = ce.CfSitlParams.from_json(dyn.E1_MINI_PARAMS_PATH)
        gz.lib_version = 31
        self.assertEqual(gz.profile(), p0.profile())
        with open(dyn.E1_MINI_PARAMS_PATH, "rb") as f:
            raw = f.read()
        with self.assertRaises(RuntimeError):            # 版本未知时 3DGS 侧不给出可写进 manifest 的 profile
            dyn.CfSitlParams(dyn.E1_MINI_PARAMS_PATH, raw, obj).profile()
        with self.assertRaises(FileNotFoundError):       # 探测用的进程不存在：早在构造参数时就报，不等到第一集
            dyn.CfSitlParams.from_json(proc=os.path.join("/nonexistent", "cf_sitl_proc"))

    @unittest.skipUnless(HAVE_PROC, f"无 cf_sitl_proc: {dyn.CF_SITL_PROC}")
    def test_probe_lib_version_from_proc(self):
        p = dyn.CfSitlParams.from_json()                    # 起一次 cf_sitl_proc 读 state.lib_version
        self.assertGreaterEqual(p.lib_version, 31)
        self.assertRegex(p.profile(), r"^cf_sitl_v1:[0-9a-f]{8}:lib\d+$")
        d = dyn.CfSitlDynamics(SCENE, params=p, seed=0)
        try:
            self.assertEqual(d.profile(), p.profile())
            self.assertEqual(d.manifest()["lib_version"], p.lib_version)
        finally:
            d.close()
        wrong = dyn.CfSitlParams.from_json(lib_version=p.lib_version + 1)
        with self.assertRaises(RuntimeError):            # 进程的库版本与 params 声称的不一致
            dyn.CfSitlDynamics(SCENE, params=wrong, seed=0)


@unittest.skipUnless(HAVE_PROC, f"无 cf_sitl_proc: {dyn.CF_SITL_PROC}")
class TestCfSitlDynamics(unittest.TestCase):
    def setUp(self):
        self.d = dyn.CfSitlDynamics(SCENE, seed=0)

    def tearDown(self):
        self.d.close()

    def test_reset_takeoff_ready(self):
        s = self.d.reset(-14.0, 0.0, 0.0)
        self.assertEqual(s["exec_state"], dyn.EXEC_READY)
        self.assertAlmostEqual(s["t"], 0.0)                    # 起飞不计入 episode
        self.assertGreater(self.d.takeoff_s, 3.0)
        self.assertLess(self.d.takeoff_s, 15.0)
        self.assertLess(abs(s["z"] - 2.0), 0.2)
        self.assertLess(math.hypot(s["x"] + 14.0, s["y"]), 0.3)
        self.assertLess(abs(s["yaw_deg"]), 10.0)
        self.assertEqual(len(s["quat"]), 4)
        self.assertEqual(self.d.path_m, 0.0)
        m = self.d.manifest()
        self.assertEqual(m["dynamics"], "cf_sitl")
        self.assertIn("action_window", m)
        with self.assertRaises(ValueError):
            self.d.step(7, 0.5)

    def test_forward_2s(self):
        s0 = self.d.reset(-14.0, 0.0, 0.0)
        for _ in range(4):
            s = self.d.step(dyn.FORWARD_SLOW, 0.5)
        along = ((s["x"] - s0["x"]) * math.cos(s0["yaw"]) + (s["y"] - s0["y"]) * math.sin(s0["yaw"]))
        self.assertGreater(along, 0.3, s)
        self.assertLess(along, 0.8, s)
        self.assertAlmostEqual(s["t"], 2.0, places=6)
        self.assertLess(abs(s["z"] - 2.0), 0.3)
        self.assertEqual(s["active_action"], dyn.FORWARD_SLOW)
        self.assertEqual(len(self.d.trace), 20)                # 0.5 s / 0.1 s × 4
        self.assertEqual([r["step"] for r in self.d.trace][::5], [0, 1, 2, 3])
        self.assertTrue(all(abs(s["pitch_deg"]) < 20 and abs(s["roll_deg"]) < 20 for s in [s]))

    def test_hold_requires_whole_sitl_ticks(self):
        self.d.reset(-14.0, 0.0, 0.0)
        before = self.d._state()
        for hold in (0.0004, 0.1004, float("nan"), float("inf"), 0.0):
            with self.subTest(hold=hold), self.assertRaises(ValueError):
                self.d.step(dyn.FORWARD_SLOW, hold)
            self.assertEqual(self.d._state(), before)
            self.assertEqual(self.d.step_index, 0)
            self.assertEqual(self.d.trace, [])
        # 浮点加法形成的合法窗口不应被拒绝，日志时间必须等于实际推进时间。
        s = self.d.step(dyn.FORWARD_SLOW, 0.1 + 0.2)
        self.assertAlmostEqual(s["t"], 0.3, places=9)
        self.assertEqual(len(self.d.trace), 3)

    def test_yaw_left_1s(self):
        s0 = self.d.reset(-14.0, 0.0, 0.0)
        s = self.d.step(dyn.YAW_LEFT, 0.5)
        s = self.d.step(dyn.YAW_LEFT, 0.5)
        dyaw = (s["yaw_deg"] - s0["yaw_deg"] + 180.0) % 360.0 - 180.0
        self.assertGreater(dyaw, 20.0, s)
        self.assertLess(dyaw, 45.0, s)
        self.assertGreater(s["yaw_rate"], 0.0)
        self.assertLess(math.hypot(s["x"] - s0["x"], s["y"] - s0["y"]), 0.5)   # 原地偏航

    def test_proc_death_raises_not_hangs(self):
        self.d.reset(-14.0, 0.0, 0.0)
        self.d.proc.kill()
        self.d.proc.wait()
        with self.assertRaises(RuntimeError):
            self.d.step(dyn.FORWARD_SLOW, 0.5)

    def test_missing_proc_is_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                dyn.CfSitlDynamics(SCENE, proc=os.path.join(tmp, "nope"))


if __name__ == "__main__":
    unittest.main()
