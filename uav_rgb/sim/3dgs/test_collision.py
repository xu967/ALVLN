#!/usr/bin/env python3
"""碰撞代理测试：切片/体素/膨胀/地面/净空，以及合成场景端到端。"""
import importlib
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
col = importlib.import_module("sim.3dgs.collision")
scene_mod = importlib.import_module("sim.3dgs.scene")


class TestCollision(unittest.TestCase):
    def setUp(self):
        pts = []
        for x in np.linspace(4, 5, 11):
            for y in np.linspace(-0.5, 0.5, 11):
                pts.append((x, y, 2.0))
        pts += [(0, 10, 0.0), (30, 0, 2.0), (5, 0, 0.5)]
        self.occ = col.Occupancy(np.array(pts), z_slice=(1.65, 2.35),
                                 voxel_m=0.1, inflate_m=0.3, drone_radius_m=0.3)

    def test_inside_and_inflation(self):
        self.assertTrue(self.occ.collides(4.5, 0, 2.0))
        self.assertTrue(self.occ.collides(4.5, 0.7, 2.0))
        self.assertFalse(self.occ.collides(4.5, 1.2, 2.0))
        self.assertFalse(self.occ.collides(15, 0, 2.0))

    def test_ground_and_out_of_slice(self):
        self.assertTrue(self.occ.collides(4.5, 0, 0.2))
        self.assertFalse(self.occ.collides(4.5, 0, 0.5))   # 低于切片，仅地面判定
        self.assertFalse(self.occ.collides(4.5, 0, 3.0))   # 高于切片，且离地够高

    def test_clearance(self):
        self.assertLess(self.occ.clearance(4.5, 0, 2.0), 0.01)
        self.assertGreater(self.occ.clearance(4.5, 2.0, 2.0), 1.0)
        self.assertLessEqual(self.occ.clearance(0, 0, 0.2), 0.2)

    def test_empty(self):
        occ = col.Occupancy(np.zeros((0, 3)))
        self.assertFalse(occ.collides(0, 0, 2.0))
        self.assertAlmostEqual(occ.clearance(0, 0, 2.0), 2.0)

    def test_scene_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_mod.make_corridor(tmp, seed=1, length=10.0)
            scene = scene_mod.load_scene(os.path.join(tmp, "scene.json"))
            occ = col.from_scene(scene)
            self.assertGreater(occ.n_sliced, 100)
            # 出生点自由、走廊中央自由、墙边碰撞。
            sp = scene["spawn"]
            self.assertFalse(occ.collides(sp["x"], sp["y"], sp["z"]))
            self.assertFalse(occ.collides(3.0, 0.0, 2.0))
            self.assertTrue(occ.collides(3.0, 3.9, 2.0))


if __name__ == "__main__":
    unittest.main()
