#!/usr/bin/env python3
"""measure_z_margin.py（整集高度包络与起飞水平漂移测量，卡 S1b 第二轮 J-1）的单测：合成轨迹，不读真实实验输出。

锁定：
  1. 全部步（含第 0 步动作前位姿 pose_before、步末 pose_gt 与 trace.jsonl 全部中间点）的 z − spawn.z 最小与最大值都计入包络（不再跳过前 3 步起飞过冲）；
  2. 起飞水平漂移 = 第 0 步动作前位姿 (pose_before.x, pose_before.y) 到候选出生点 (cx, cy) 的欧氏距离最大值；
  3. Z_ENVELOPE_DOWN_M、Z_ENVELOPE_UP_M、SPAWN_DRIFT_MARGIN_M 均为各来源极值向外取到 5 mm 的整数倍；
  4. r<N>.incomplete-<时刻> 半成品不读。
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


def write_episode(root, scene, route, ep, before_zs, gt_zs, trace_zs=(), spawn_z=1.2,
                  cand_xy=(0.0, 0.0), start_xy=(0.0, 0.0), seed=8000, with_scene=True):
    d = os.path.join(root, scene, route, ep)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
        for i, (zb, zg) in enumerate(zip(before_zs, gt_zs)):
            sx, sy = start_xy if i == 0 else (0.1 * i, 0.0)
            f.write(json.dumps({
                "step": i,
                "pose_before": {"x": sx, "y": sy, "z": zb},
                "pose_gt": {"x": sx + 0.1, "y": sy, "z": zg},
            }) + "\n")
    if trace_zs:
        with open(os.path.join(d, "trace.jsonl"), "w", encoding="utf-8") as f:
            for i, zt in enumerate(trace_zs):
                f.write(json.dumps({"step": 0, "t": 0.1 * (i + 1), "pos": [0.0, 0.0, zt]}) + "\n")
    if with_scene:
        with open(os.path.join(d, "scene.json"), "w", encoding="utf-8") as f:
            json.dump({"spawn": {"x": cand_xy[0], "y": cand_xy[1], "z": spawn_z, "yaw_deg": 0.0}}, f)
    with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"scene": scene, "route": 0, "seed": seed}, f)


class TestMeasureEnvelopeAndDrift(unittest.TestCase):
    def test_includes_step0_pose_before_and_trace_extremes(self):
        with tempfile.TemporaryDirectory() as t:
            # 第 0 步 pose_before 过冲到 1.270（+0.070），trace 跌到 1.166（-0.034），稳态在 1.200
            write_episode(t, "s", "r0", "ep_00000",
                          before_zs=[1.270, 1.200, 1.200, 1.200],
                          gt_zs=[1.215, 1.200, 1.200, 1.200],
                          trace_zs=[1.260, 1.166],
                          spawn_z=1.200, cand_xy=(1.0, 2.0), start_xy=(1.06, 2.08))
            res = mz.scan_source(t, depth=1, jitter=0.0)
            self.assertEqual(res["episodes"], 1)
            self.assertAlmostEqual(res["dz_max"], 0.070, places=6)
            self.assertAlmostEqual(res["dz_min"], -0.034, places=6)
            self.assertAlmostEqual(res["drift_max"], 0.100, places=6)   # hypot(0.06, 0.08) = 0.10

    def test_incomplete_and_sceneless_episodes_skipped(self):
        with tempfile.TemporaryDirectory() as t:
            write_episode(t, "s", "r0", "ep_00000", [1.2], [1.2])
            write_episode(t, "s", "r0.incomplete-020101", "ep_00000", [1.9], [0.5], start_xy=(5.0, 5.0))
            write_episode(t, "s", "r1", "ep_00000", [1.9], [0.5], with_scene=False)
            res = mz.scan_source(t, depth=1, jitter=0.0)
            self.assertEqual(res["episodes"], 1)
            self.assertEqual((res["dz_min"], res["dz_max"], res["drift_max"]), (0.0, 0.0, 0.0))

    def test_pick_envelope_and_drift_rounds_outward_to_5mm(self):
        sources = {
            "a": {"dz_min": -0.034197, "dz_max": 0.070023, "drift_max": 0.0824},
            "b": {"dz_min": -0.028900, "dz_max": 0.065700, "drift_max": 0.102369},
        }
        picked = mz.pick_constants(sources)
        self.assertEqual(picked["z_envelope_down_m"], 0.035)
        self.assertEqual(picked["z_envelope_up_m"], 0.075)
        self.assertEqual(picked["spawn_drift_margin_m"], 0.105)
        # 恰为 5 mm 整数倍时不再多加一档
        exact = mz.pick_constants({"c": {"dz_min": -0.030, "dz_max": 0.070, "drift_max": 0.080}})
        self.assertEqual((exact["z_envelope_down_m"], exact["z_envelope_up_m"], exact["spawn_drift_margin_m"]),
                         (0.030, 0.070, 0.080))


if __name__ == "__main__":
    unittest.main()
