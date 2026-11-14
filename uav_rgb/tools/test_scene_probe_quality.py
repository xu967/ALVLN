#!/usr/bin/env python3
"""scene_probe_quality 回归：阈值/判定函数与采集端同一对象、路径采样间距与航向、路线汇总与门槛、
候选解析、拼图尺寸、--self-check 子进程（mock 渲染器，不要 GPU）。

  $DATA_ROOT/uav_rgb/3dgs-venv/bin/python tools/test_scene_probe_quality.py
"""
import importlib
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import scene_probe_quality as spq  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP


def _frame(low, lapvar=None, hole=0.0):
    return {"pose": [0, 0, 2, 0], "lapvar": lapvar if lapvar is not None else (10.0 if low else 500.0),
            "hole_frac": hole, "low": bool(low), "depth_med_m": 3.0, "depth_valid_frac": 1.0 - hole}


class SharedThreshold(unittest.TestCase):
    def test_same_objects_as_collector(self):
        collect = importlib.import_module("sim.3dgs.collect_3dgs")
        self.assertIs(spq.frame_quality, collect.frame_quality)
        self.assertEqual(spq.FRAME_LAPVAR_MIN, collect.FRAME_LAPVAR_MIN)
        self.assertEqual(spq.LOOKAHEAD_M, importlib.import_module("sim.3dgs.planner").LOOKAHEAD_M)

    def test_summary_threshold_is_collectors(self):
        s = spq.summarize("r", [_frame(False)], 1.0)
        self.assertEqual(s["lapvar_min_threshold"], spq.FRAME_LAPVAR_MIN)


class SamplePoses(unittest.TestCase):
    def test_straight_spacing_and_heading(self):
        path = [(0.1 * k, 0.0) for k in range(101)]           # 10 m，0.1 m 栅格
        poses = spq.sample_poses(path, spawn_yaw_deg=90.0, step_m=0.5)
        self.assertEqual(poses[0], (0.0, 0.0, 90.0))          # 首帧用出生朝向
        self.assertAlmostEqual(poses[-1][0], 10.0, places=6)  # 末点必含
        gaps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poses, poses[1:])]
        self.assertTrue(all(0.5 - 1e-6 <= g <= 0.5 + 1e-6 for g in gaps[:-1]), gaps)
        self.assertEqual(len(poses), 21)
        for p in poses[1:]:
            self.assertAlmostEqual(p[2], 0.0, places=6)       # 沿 +x 航向 0°

    def test_corner_heading_turns(self):
        path = [(0.1 * k, 0.0) for k in range(31)] + [(3.0, 0.1 * k) for k in range(1, 31)]   # 3 m 东再 3 m 北
        poses = spq.sample_poses(path, spawn_yaw_deg=0.0, step_m=0.5)
        yaws = [p[2] for p in poses]
        self.assertAlmostEqual(yaws[1], 0.0, places=6)
        self.assertAlmostEqual(yaws[-1], 90.0, places=6)
        self.assertTrue(any(0.0 < y < 90.0 for y in yaws), yaws)   # 拐角处前视点在转角另一侧，航向过渡

    def test_degenerate_paths(self):
        self.assertEqual(spq.sample_poses([], 0.0), [])
        self.assertEqual(spq.sample_poses([(1.0, 2.0)], 30.0), [(1.0, 2.0, 30.0)])


class Summarize(unittest.TestCase):
    def test_low_frac_and_pass(self):
        frames = [_frame(False)] * 9 + [_frame(True)]
        s = spq.summarize("r", frames, 5.0)
        self.assertEqual((s["n_frames"], s["n_low"], s["low_frac"]), (10, 1, 0.1))
        self.assertTrue(s["pass"])                            # 刚好 10% 过线
        frames.append(_frame(True))
        s = spq.summarize("r", frames, 5.0)
        self.assertGreater(s["low_frac"], 0.10)
        self.assertFalse(s["pass"])

    def test_stats_and_empty(self):
        frames = [_frame(False, 100.0, 0.1), _frame(False, 300.0, 0.3), _frame(False, 200.0, 0.2)]
        s = spq.summarize("r", frames, 1.0)
        self.assertEqual((s["lapvar_min"], s["lapvar_med"], s["hole_frac_max"]), (100.0, 200.0, 0.3))
        e = spq.summarize("empty", [], 0.0)
        self.assertFalse(e["pass"])
        self.assertIsNone(e["low_frac"])


class DepthGate(unittest.TestCase):
    def test_high_frequency_cannot_hide_invalid_depth(self):
        rgb = np.repeat(((np.indices((16, 16)).sum(axis=0) % 2) * 255)[..., None], 3, axis=2).astype(np.uint8)
        class Renderer:
            def render_pose(self, *args, **kwargs):
                return rgb, self.depth, 0.0
        renderer = Renderer()
        for value in (0.0, -1.0, float("nan"), float("inf")):
            renderer.depth = np.full((16, 16), value)
            frames, _ = spq.probe_poses(renderer, [(0, 0, 0)], 2)
            self.assertFalse(frames[0]["low"])
            self.assertEqual(frames[0]["depth_valid_frac"], 0.0)
            self.assertIsNone(frames[0]["depth_med_m"])
            self.assertFalse(spq.summarize("bad", frames, 1)["pass"])
        renderer.depth = np.ones((16, 16))
        frames, _ = spq.probe_poses(renderer, [(0, 0, 0)], 2)
        self.assertTrue(spq.summarize("good", frames, 1)["pass"])

    def test_holes_are_independent_gate(self):
        self.assertTrue(spq.summarize("r", [_frame(False, hole=0.1)], 1)["depth_pass"])
        self.assertFalse(spq.summarize("r", [_frame(False, hole=0.11)], 1)["depth_pass"])
        for threshold in (0, -1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                spq.summarize("r", [_frame(False)], 1, min_depth_valid_frac=threshold)


class Candidate(unittest.TestCase):
    def test_parse(self):
        name, sp, wp = spq.parse_candidate("r9:-4,0.5,90,5,2")
        self.assertEqual(name, "r9")
        self.assertEqual(sp, {"x": -4.0, "y": 0.5, "z": 2.0, "yaw_deg": 90.0})
        self.assertEqual(wp, {"x": 5.0, "y": 2.0, "z": 2.0})
        _, sp, wp = spq.parse_candidate("low:0,0,0,1,1,1.0")
        self.assertEqual((sp["z"], wp["z"]), (1.0, 1.0))
        for bad in ("noname", "r:1,2,3", "r:1,2,3,4,5,6,7", ":1,2,3,4,5"):
            with self.assertRaises(ValueError):
                spq.parse_candidate(bad)


class ContactSheet(unittest.TestCase):
    def test_grid_shape_and_subsample(self):
        rgbs = [np.full((240, 320, 3), i * 10, np.uint8) for i in range(7)]
        sheet = spq.contact_sheet(rgbs, [str(i) for i in range(7)], cols=6, max_frames=18)
        self.assertEqual(sheet.shape, (480, 1920, 3))         # 7 帧 -> 2 行，补黑
        sheet = spq.contact_sheet(rgbs * 5, [str(i) for i in range(35)], cols=6, max_frames=18)
        self.assertEqual(sheet.shape, (720, 1920, 3))         # 35 帧抽到 18 -> 3 行


class SelfCheck(unittest.TestCase):
    def test_cli_self_check(self):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "scene_probe_quality.py"), "--self-check"],
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("self-check PASS", r.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
