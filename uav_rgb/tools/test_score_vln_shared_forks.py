import unittest

from tools.score_vln_shared_forks import MODES, score_group


class TestSharedForkTransfer(unittest.TestCase):
    def test_primary_counts_sites_and_keeps_failed_requests(self):
        predictions = {m: ["L", "C"] for m in MODES}
        rows = [dict(primary=True, physical_site=("room", 1, 2, 1), pair="A", truth=["L", "C"], predicted=predictions),
                dict(primary=False, physical_site=("room", 1, 2, 1), pair="A", truth=[None, "C"], predicted=predictions)]
        scores = score_group(["A", "B"], rows)
        self.assertEqual((scores["requested_pairs"], scores["pairs_with_flown_observation"], scores["primary_sites"]), (2, 1, 1))
        self.assertEqual(scores["scores"]["region_text"]["all_flown_both_correct"], 1)
        self.assertEqual(scores["scores"]["region_text"]["primary_pair_accuracy"], 1.0)
        self.assertIsNone(score_group(["A"], [])["scores"]["region_text"]["primary_pair_accuracy"])

    def test_neighboring_frames_cannot_count_as_two_primary_sites(self):
        row = dict(primary=True, physical_site=("room", 1, 2, 1))
        with self.assertRaisesRegex(ValueError, "重复"):
            score_group(["A"], [row, row])


if __name__ == "__main__":
    unittest.main()
