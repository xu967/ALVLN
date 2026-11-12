#!/usr/bin/env python3
"""measure_z_margin.py 的单测：合成轨迹，不读真实实验输出。

锁定：起飞过冲（前 STEADY_FROM_STEP 步）不进稳态分布；r<N>.incomplete-<时刻> 半成品不读；余量 = 各来源 max(|p1|, |p99|) 的最大值向上取到 5 mm；
没有 scene.json 的集跳过。

  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python experiments/2026-11-12-3dgs-rebaseline/test_measure_z_margin.py
"""
import importlib.util
import json
import os
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("measure_z_margin", os.path.join(HERE, "measure_z_margin.py"))
mz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mz)

_DATA = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
if "TMPDIR" not in os.environ and os.path.isdir(os.path.join(_DATA, "tmp")):
    tempfile.tempdir = os.path.join(_DATA, "tmp")


def write_episode(root, scene, route, ep, zs, spawn_z=1.2, with_scene=True):
    d = os.path.join(root, scene, route, ep)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "actions.jsonl"), "w") as f:
        for i, z in enumerate(zs):
            f.write(json.dumps({"step": i, "pose_gt": {"x": 0.0, "y": 0.0, "z": z}}) + "\n")
    if with_scene:
        with open(os.path.join(d, "scene.json"), "w") as f:
            json.dump({"spawn": {"z": spawn_z}}, f)


class TestMeasureZMargin(unittest.TestCase):
    def test_takeoff_overshoot_excluded(self):
        with tempfile.TemporaryDirectory() as t:
            # 前 3 步过冲 +0.05，之后 100 步在 spawn.z 上下 ±0.02 均匀分布
            zs = [1.25, 1.25, 1.25] + list(1.2 + np.linspace(-0.02, 0.02, 100))
            write_episode(t, "s", "r0", "ep_00000", zs)
            dz, n_ep = mz.steady_dz(t, depth=1)
            self.assertEqual(n_ep, 1)
            self.assertEqual(len(dz), 100)
            self.assertLess(abs(dz.max() - 0.02), 1e-9)

    def test_incomplete_and_sceneless_episodes_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            write_episode(t, "s", "r0", "ep_00000", [1.2] * 6)
            write_episode(t, "s", "r0.incomplete-020101", "ep_00000", [1.5] * 6)
            write_episode(t, "s", "r1", "ep_00000", [1.5] * 6, with_scene=False)
            dz, n_ep = mz.steady_dz(t, depth=1)
            self.assertEqual(n_ep, 1)
            self.assertEqual(float(np.abs(dz).max()), 0.0)

    def test_margin_is_max_over_sources_rounded_up(self):
        a = {"p1": -0.0231, "p99": 0.009}
        b = {"p1": -0.0188, "p99": 0.0090}
        margin, need = mz.pick_margin({"a": a, "b": b})
        self.assertAlmostEqual(need, 0.0231)
        self.assertEqual(margin, 0.025)
        # 恰为 5 mm 整数倍时不再往上加一档
        self.assertEqual(mz.pick_margin({"c": {"p1": -0.02, "p99": 0.01}})[0], 0.02)
        # p99 在上侧更大时取上侧
        self.assertEqual(mz.pick_margin({"d": {"p1": -0.001, "p99": 0.0301}})[0], 0.035)

    def test_percentiles_use_pct_config(self):
        dz = np.linspace(-0.05, 0.05, 10001)
        s = mz.stats(dz)
        self.assertAlmostEqual(s["p1"], -0.049, places=6)
        self.assertAlmostEqual(s["p99"], 0.049, places=6)
        self.assertEqual(s["n"], 10001)


if __name__ == "__main__":
    unittest.main()
