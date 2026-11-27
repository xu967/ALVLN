import importlib
import math
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from tools.vln_fork_geometry import label_goal


scene = importlib.import_module("sim.3dgs.scene")


class TestForkGeometry(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(x=1.0, y=-2.0, z=1.2, yaw=0.0, vx=0.3, vy=-0.4)
        self.waypoint = dict(x=6.0, y=-2.0, z=1.2)
        self.camera = dict(width=100, height=80, fx=100.0, fy=100.0, cx=50.0, cy=40.0)
        self.nav = np.array([[0, -1, 0, 4], [1, 0, 0, -3], [0, 0, 1, 2], [0, 0, 0, 1]])
        self.mount = dict(xyz=[0.1, 0, 0.05], pitch_deg=0.0)
        self.planner = Mock()
        self.planner.plan.return_value = [(1, -2), (6, -2)]
        self.planner.aim_point.return_value = (2, -2)
        self.planner.occ.sweep.return_value = (False, None)

    def label(self, action):
        pos = (self.state.x, self.state.y, self.state.z)
        view = scene.viewmat_from_quat(pos, (1, 0, 0, 0), self.mount, self.nav)
        return label_goal(self.planner, self.state, self.waypoint, 0.4, self.camera, view, self.nav,
                          action, 12.0)

    def test_continuous_bearing_and_action_have_different_thresholds(self):
        dy = math.tan(math.radians(18))
        self.planner.aim_point.return_value = (2.0, self.state.y + dy)
        label = self.label(action=2)
        self.assertEqual(label["planning_status"], "astar")
        self.assertEqual((label["local_bearing_bin"], label["runtime_bin"], label["teacher_runtime_bin"]), (0, 0, 0))
        self.assertEqual((label["coarse_action_if_representable"], label["replayed_teacher_action"]), (1, 2))
        self.assertFalse(label["agrees_with_original_action"])
        self.assertAlmostEqual(label["relative_yaw_ccw_deg"], 18.0)
        self.assertAlmostEqual(label["lookahead_m"], 0.65 + 0.8 * 0.5)
        self.assertTrue(label["inside_actual_image"])
        self.assertAlmostEqual(label["camera_z_m"], 0.9)
        np.testing.assert_allclose(label["image_uv_same_altitude_px"], [50 - 100 * dy / 0.9, 40 + 5 / 0.9])

    def test_horizontal_and_vertical_bounds_use_the_recorded_camera(self):
        self.planner.aim_point.return_value = (2, -1)
        label = self.label(action=2)
        self.assertEqual(label["local_bearing_bin"], 1)
        self.assertLess(label["image_uv_same_altitude_px"][0], 0)
        self.assertFalse(label["inside_actual_camera_horizontal_fov"])
        self.planner.aim_point.return_value = (2, -2)
        self.mount["pitch_deg"] = 45
        label = self.label(action=1)
        self.assertTrue(label["inside_actual_camera_horizontal_fov"])
        self.assertFalse(label["inside_actual_image"])
        self.assertGreater(label["image_uv_same_altitude_px"][1], self.camera["height"])

    def test_clipping_and_rear_bearings(self):
        for aim, action, expected_bin in [((1.2, -2), 1, 0), ((0, -2.1), 3, 4)]:
            with self.subTest(aim=aim):
                self.planner.aim_point.return_value = aim
                label = self.label(action)
                self.assertEqual(label["local_bearing_bin"], expected_bin)
                self.assertIsNone(label["image_uv_same_altitude_px"])
                self.assertFalse(label["inside_actual_camera_horizontal_fov"])
                self.assertFalse(label["inside_actual_image"])
        self.assertIsNone(label["coarse_action_if_representable"])
        self.assertIsNone(label["agrees_with_original_action"])

    def test_missing_path_excludes_runtime_direct_fallback(self):
        self.waypoint["y"] = -7
        self.planner.plan.return_value = None
        label = self.label(action=0)
        self.assertEqual(label["planning_status"], "no_astar_path_runtime_used_direct_fallback")
        self.assertEqual((label["runtime_bin"], label["teacher_runtime_bin"]), (7, 8))
        self.assertIsNone(label["local_bearing_bin"])
        self.assertNotIn("relative_yaw_ccw_deg", label)

    def test_near_and_degenerate_aim_are_excluded(self):
        self.waypoint["x"] = 1.25
        label = self.label(action=0)
        self.assertEqual((label["planning_status"], label["runtime_near"]), ("near_goal_excluded", 1))
        self.assertIsNone(label["local_bearing_bin"])
        self.planner.plan.assert_not_called()
        self.waypoint["x"] = 6.0
        self.planner.aim_point.return_value = (self.state.x, self.state.y)
        label = self.label(action=1)
        self.assertEqual(label["planning_status"], "degenerate_local_vector")
        self.assertIsNone(label["local_bearing_bin"])

    def test_invalid_state_calibration_and_action_raise(self):
        with self.assertRaisesRegex(ValueError, "registered action 3, replayed 1"):
            self.label(action=3)
        self.camera["fx"] = float("nan")
        with self.assertRaisesRegex(ValueError, "camera calibration"):
            self.label(action=1)
        self.state.vx = float("inf")
        with self.assertRaisesRegex(ValueError, "Non-finite pose, velocity"):
            self.label(action=1)


if __name__ == "__main__":
    unittest.main()
