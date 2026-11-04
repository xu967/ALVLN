#!/usr/bin/env python3
"""cf_blackbox 参数单源 + 3DGS 动力学包装的测试（不依赖 torch）。"""
import importlib
import math
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
dyn = importlib.import_module("sim.3dgs.dynamics")


class TestDynamics(unittest.TestCase):
    def test_params_single_source(self):
        p = dyn.CfBlackboxParams.from_json()
        self.assertFalse(p.calibrated)
        self.assertEqual(p.z_target_m, 2.0)
        self.assertEqual(dyn.DEFAULT_PARAMS_PATH,
                         os.path.join(ROOT, "sim", "dynamics", "cf_blackbox_v1.json"))
        self.assertEqual(p.profile(), dyn.CfBlackbox(p).profile())

    def test_fixed_z_and_forward(self):
        d = dyn.CfBlackbox(dyn.CfBlackboxParams.from_json(), 0, 0, 0)
        for _ in range(20):
            s = d.step(dyn.FORWARD_SLOW, 0.5)
        self.assertEqual(s["z"], 2.0)
        self.assertGreater(s["x"], 0.5)
        self.assertAlmostEqual(s["y"], 0.0, places=6)
        self.assertEqual(s["active_action"], dyn.FORWARD_SLOW)

    def test_yaw_left_right(self):
        d = dyn.CfBlackbox(dyn.CfBlackboxParams.from_json(), 0, 0, 0)
        for _ in range(4):
            s = d.step(dyn.YAW_LEFT, 0.5)
        self.assertGreater(s["yaw_deg"], 5)
        self.assertGreater(s["yaw_rate"], 0.0)
        self.assertEqual(s["z"], 2.0)
        d2 = dyn.CfBlackbox(dyn.CfBlackboxParams.from_json(), 0, 0, 0)
        for _ in range(4):
            s2 = d2.step(dyn.YAW_RIGHT, 0.5)
        self.assertLess(s2["yaw_deg"], -5)
        self.assertLess(s2["yaw_rate"], 0.0)

    def test_latency(self):
        d = dyn.CfBlackbox(dyn.CfBlackboxParams.from_json(), 0, 0, 0)
        lat = d.params.latency_s
        self.assertGreater(lat, 0)
        d.step(dyn.FORWARD_SLOW, lat * 0.5)
        self.assertAlmostEqual(d.x, 0.0, places=9)
        d.step(None, lat)          # 只推进，不重复派发
        self.assertGreater(d.x, 0.0)

    def test_yaw_first_order(self):
        d = dyn.CfBlackbox(dyn.CfBlackboxParams.from_json(), 0, 0, 0)
        d.dispatch(dyn.YAW_LEFT, 0)
        d.step(None, d.params.latency_s + 0.01)
        target = d.params.yaw_rate_rad_s
        self.assertGreater(d.yaw_rate, 0.0)
        self.assertLess(d.yaw_rate, target)
        d.step(None, 2.0)
        self.assertAlmostEqual(d.yaw_rate, target, places=4)

    def test_invalid_action(self):
        d = dyn.CfBlackbox(dyn.CfBlackboxParams.from_json(), 0, 0, 0)
        with self.assertRaises(ValueError):
            d.step(9, 0.1)
        with self.assertRaises(ValueError):
            d.step(dyn.SAFE_NEUTRAL, -0.1)


if __name__ == "__main__":
    unittest.main()
