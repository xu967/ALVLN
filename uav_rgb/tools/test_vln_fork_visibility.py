from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from tools import vln_fork_visibility as visibility


CAMERA = {"width": 21, "height": 21, "fx": 10, "fy": 10, "cx": 10, "cy": 10}
BOX = np.array([[-1, -1, 2], [1, -1, 2], [1, 1, 2], [-1, 1, 2],
                [-1, -1, 4], [1, -1, 4], [1, 1, 4], [-1, 1, 4]], dtype=float)


def observation(step, *, strict=False, pair="a", heading=0, x=1):
    return dict(step=step, pair=pair, scene="839992", split="development", verified_shared_prefix=True,
                spawn=dict(x=x, y=2, z=1.2, yaw_deg=heading), status="different_coarse_both_in_image" if strict else "same_bearing",
                labels=[dict(planning_status="astar", local_bearing_bin=0, inside_actual_image=True),
                        dict(planning_status="astar", local_bearing_bin=int(strict), inside_actual_image=True)])


class TestForkVisibility(unittest.TestCase):
    @unittest.skipUnless((Path("/root/rivermind-data/uav_rgb") / visibility.RAY_LIBRARY).is_file(), "缺已登记的旧帧射线库")
    def test_mesh_ray_returns_camera_z(self):
        root = Path("/root/rivermind-data/uav_rgb")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "surface.ply"
            path.write_text("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\n"
                            "property float z\nelement face 1\nproperty list uchar int vertex_indices\n"
                            "end_header\n-1 -1 2.5\n1 -1 2.5\n0 1 2.5\n3 0 1 2\n")
            depth = visibility.collision_depth(root, path, CAMERA, np.eye(4))
            self.assertAlmostEqual(float(depth[10, 10]), 2.5, delta=1e-6)
            self.assertTrue(np.isnan(depth[0, 0]))

    def test_compatible_depth_is_geometry_only(self):
        depth = np.full((21, 21), 2.5)
        result = visibility.classify_box(BOX, np.eye(4), CAMERA, depth, 0.2, 50)
        self.assertEqual(result["status"], "unobstructed_candidate")
        self.assertEqual(result["box_xyxy_inclusive"], [5, 5, 15, 15])
        self.assertEqual((result["compatible_pixels"], result["behind_pixels"], result["box_pixels"]), (81, 40, 121))

    def test_foreground_missing_depth_and_outside(self):
        result = visibility.classify_box(BOX, np.eye(4), CAMERA, np.full((21, 21), 1.0), 0.2, 50)
        self.assertEqual(result["reason"], "all_pixels_have_nearer_surface")
        self.assertEqual(result["front_pixels"], result["box_pixels"])
        result = visibility.classify_box(BOX, np.eye(4), CAMERA, np.full((21, 21), np.nan), 0.2, 50)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["invalid_pixels"], result["box_pixels"])
        shifted = BOX + [100, 0, 0]
        self.assertEqual(visibility.classify_box(shifted, np.eye(4), CAMERA, None, 0.2, 50)["status"],
                         "outside_image")
        self.assertEqual(visibility.classify_box(BOX, np.eye(4), CAMERA, None, 0.2, 50)["status"], "unknown")
        self.assertEqual(visibility.classify_box(None, np.eye(4), CAMERA, None, 0.2, 50)["status"], "unknown")

    def test_choose_strict_then_flown_then_reset_per_physical_point(self):
        rows = [observation(0, strict=True), observation(1), observation(7, strict=True),
                observation(3, strict=True, pair="first", heading=90),
                observation(3, strict=True, pair="second", heading=180), observation(6, x=5), observation(0, x=8)]
        self.assertEqual([(role, row["pair"], row["step"]) for role, row in visibility.representatives(rows)],
                         [("primary", "first", 3), ("flown_control", "a", 6), ("reset_only", "a", 0)])
        with self.assertRaises(ValueError):
            visibility.representatives([dict(observation(1), verified_shared_prefix=False)])
        with self.assertRaises(ValueError):
            visibility.representatives([dict(observation(1), scene="840818")])

    def test_unreviewed_phrase_keeps_identity_unknown(self):
        spans = visibility.read_reference_spans(visibility.REFERENCE_FILE)
        new = visibility.reference_review(spans, "A newly supplied sentence.", "chair_20")
        self.assertEqual((new["status"], new["phrases"], new["instance_ids"]), ("pending_text_review", None, None))
        self.assertNotEqual(new["instruction_goal_sha256"],
                            visibility.reference_review(spans, "A newly supplied sentence.", "lamp_2")["instruction_goal_sha256"])


if __name__ == "__main__":
    unittest.main()
