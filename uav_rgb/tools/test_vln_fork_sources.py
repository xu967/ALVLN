import copy
import hashlib
import math
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from tools.vln_fork_sources import ARCHIVE, ForkSources, decoded_rgb_sha, pre_action_state, shared_fork


class TestPreActionState(unittest.TestCase):
    def setUp(self):
        self.action = dict(step=2, timestamp_ns=1_000_000_000, rgb_ts=1_000_000_000, pose_ts=1_000_000_000,
                           pose_before=dict(x=1, y=2, z=3), yaw_before=90.0, speed_mps=99)
        self.trace = dict(step=1, t=1.0, pos=[1, 2, 3], quat=[math.sqrt(0.5), 0, 0, math.sqrt(0.5)],
                          vel=[0.1, 0.2, -0.3])

    def test_measured_velocity_and_heading_at_dispatch(self):
        state = pre_action_state(self.action, self.trace)
        self.assertEqual((state.x, state.y, state.z, state.vx, state.vy), (1, 2, 3, 0.1, 0.2))
        self.assertAlmostEqual(state.yaw, math.pi / 2)

    def test_rejects_wrong_clock_body_or_velocity(self):
        for changes in (dict(step=2), dict(t=0.9), dict(pos=[1, 2, 2.99]), dict(quat=[1, 0, 0, 0]),
                        dict(vel=[0.1, math.nan, 0.0])):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                pre_action_state(self.action, dict(self.trace, **changes))
        with self.assertRaisesRegex(ValueError, "clocks"):
            pre_action_state(dict(self.action, pose_ts=self.action["timestamp_ns"] - 1), self.trace)
        with self.assertRaisesRegex(ValueError, "previous"):
            pre_action_state(dict(self.action, step=0), dict(self.trace, step=-1))


class TestSharedFork(unittest.TestCase):
    def setUp(self):
        actions = [dict(step=i, timestamp_ns=i, rgb_ts=i, pose_ts=i, pose_before={}, yaw_before=0,
                        action_executed=1, frame=f"{i}.png") for i in range(3)]
        left = dict(logs=Path("A"), actions=actions, trace=list(range(15)), camera={})
        right = copy.deepcopy(left)
        right["logs"] = Path("B")
        right["actions"][1]["action_executed"] = 2
        right["trace"][5:] = [99] * 10
        right["actions"][2]["yaw_before"] = 27
        self.arms = [left, right]
        self.mask = dict(pair="pair", status="strict_rgb_fork", rgb_equal_through_divergence=True,
                         physics_equal_through_divergence=True, first_unequal_executed_action_step=1)

    def test_shared_image_includes_first_step_with_different_actions(self):
        paths = []

        def rgb(path, camera):
            paths.append(path)
            return path.name

        self.assertEqual(shared_fork(self.arms, self.mask, rgb), 1)
        self.assertEqual(paths, [Path("A/0.png"), Path("B/0.png"), Path("A/1.png"), Path("B/1.png")])

    def test_rejects_changed_pixels_and_changed_flight_history(self):
        with self.assertRaisesRegex(ValueError, "Shared RGB"):
            shared_fork(self.arms, self.mask, lambda path, _: str(path))
        self.arms[1]["trace"][4] = 99
        with self.assertRaisesRegex(ValueError, "history"):
            shared_fork(self.arms, self.mask, lambda path, _: path.name)


class TestSourceFingerprint(unittest.TestCase):
    def test_decoded_hash_is_independent_of_png_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.png", Path(tmp) / "b.png"
            image = Image.new("RGB", (10, 8), (20, 65, 70))
            image.save(a, compress_level=0)
            image.save(b, compress_level=9)
            self.assertNotEqual(a.read_bytes(), b.read_bytes())
            expected = hashlib.sha256(image.tobytes()).hexdigest()
            self.assertEqual(decoded_rgb_sha(a, image.size), expected)
            self.assertEqual(decoded_rgb_sha(b, image.size), expected)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                decoded_rgb_sha(b, (9, 8))

    def test_unregistered_archive_fails_before_loading_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / ARCHIVE
            archive.mkdir()
            (archive / "registration.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                ForkSources(tmp)


if __name__ == "__main__":
    unittest.main()
