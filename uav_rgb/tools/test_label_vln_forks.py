import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.label_vln_forks import build, counts, grouped_counts, sha256


def arm(bearing, inside=True):
    if bearing is None:
        return {"planning_status": "near_goal_excluded", "replayed_teacher_action": 0, "original_action": 0}
    return {
        "planning_status": "astar", "local_bearing_bin": bearing, "inside_actual_image": inside,
        "inside_actual_camera_horizontal_fov": inside, "replayed_teacher_action": 1, "original_action": 1,
        "agrees_with_original_action": {0: True, 1: False, 7: False}.get(bearing),
    }


class TestForkBearingCounts(unittest.TestCase):
    def test_all_requests_keep_their_denominator(self):
        bins = [(arm(0), arm(1)), (arm(7), arm(0, False)), (arm(4, False), arm(1)), (arm(0), arm(0)), (arm(None), arm(7))]
        rows = [dict(pair="a" if i < 2 else "b", spawn_group="g", verified_shared_prefix=i == 0,
                     registered_shared_prefix=False, labels=pair) for i, pair in enumerate(bins)]
        result = counts(rows)
        self.assertEqual((result["states"], result["instruction_requests"], result["both_valid"]), (5, 10, 4))
        self.assertEqual((result["same_bin"], result["different_bin"], result["different_both_coarse"]), (1, 3, 2))
        self.assertEqual((result["different_coarse_both_in_image"], result["different_coarse_shared"]), (1, 1))
        self.assertEqual((result["different_coarse_route_pairs"], result["different_coarse_spawns"]), (1, 1))
        self.assertEqual((result["in_image"], result["outside_image"], result["outside_coarse_bins"]), (7, 2, 1))
        self.assertEqual((result["coarse_action_agrees"], result["coarse_action_differs"]), (4, 4))
        self.assertEqual(grouped_counts(rows, "pair")["b"]["states"], 3)

    def test_existing_results_are_preserved_before_opening_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with self.assertRaises(FileExistsError):
                build(out / "absent", out / "absent", out)


@unittest.skipUnless(os.environ.get("UAVRGB_FORK_LABEL_DIR"), "需要四房间标签和旧十对数据盘产物")
class TestFrozenForkCorpus(unittest.TestCase):
    def setUp(self):
        self.folder = Path(os.environ["UAVRGB_FORK_LABEL_DIR"])
        self.rows = {
            split: [json.loads(line) for line in (self.folder / f"{split}.jsonl").read_text().splitlines()]
            for split in ("train", "development")
        }

    def test_split_hashes_and_independent_denominators(self):
        summary = json.loads((self.folder / "summary.json").read_text())
        train, dev = self.rows["train"], self.rows["development"]
        self.assertEqual((len(train), len(dev)), (832, 286))
        self.assertEqual({r["scene"] for r in train}, {"840139", "841012", "841150"})
        self.assertEqual({r["scene"] for r in dev}, {"839992"})
        for key in ("pair", "spawn_group", "rgb_sha256"):
            self.assertTrue({r[key] for r in train}.isdisjoint(r[key] for r in dev), key)
        self.assertEqual(len({r["rgb_sha256"] for r in train + dev}), 1118)
        self.assertEqual(sum(r["verified_shared_prefix"] for r in train + dev), 270)
        for split, rows in self.rows.items():
            self.assertEqual(sha256(self.folder / f"{split}.jsonl"), summary["output_sha256"][f"{split}.jsonl"])
            self.assertEqual(counts(rows), summary["by_split"][split])

    def test_ten_historical_pairs_match_the_original_replay(self):
        prior = self.folder.parents[1] / "exp-vln-local-waypoint-transfer-20261127/fork_bearing_labels.json"
        self.assertEqual(sha256(prior), "20bb1a49d52dbcd2b2c7c276506a9f02e74a867e2b2c594b3f10548cf4d0c876")
        expected = json.loads(prior.read_text())["rows"]
        self.assertEqual(len(expected), 10)
        dev = {r["rgb_sha256"]: r for r in self.rows["development"]}
        renamed = {"instance_label": "goal_instance_label", "target_instance": "goal_instance",
                   "target_object": "goal_object", "waypoint_nav_m": "waypoint"}
        for old in expected:
            new = dev[old["rgb_sha256_decoded"]]
            for key in ("step", "pair", "registered_shared_prefix", "verified_shared_prefix", "pose_nav_m",
                        "body_quat_wxyz", "action_timestamp_ns", "pre_action_velocity_xyz_mps", "yaw_before_deg"):
                self.assertEqual(old[key], new[key], (old["pair"], key))
            for before, after in zip(old["labels"], new["labels"], strict=True):
                for key, value in before.items():
                    self.assertEqual(value, after[renamed.get(key, key)], (old["pair"], before["route"], key))


if __name__ == "__main__":
    unittest.main()
