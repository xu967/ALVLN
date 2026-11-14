import importlib.util
import os
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vln_grounder", os.path.join(HERE, "grounder.py"))
grounder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(grounder)


class TestLexicalGrounder(unittest.TestCase):
    def setUp(self):
        self.g = grounder.LexicalGrounder()

    def test_matches_landmark_without_using_coordinates(self):
        result = self.g.ground("Go to the track light above the wall art", ["track_light_1", "sofa_1"])
        self.assertEqual(result.target, "track_light_1")
        self.assertTrue(result.accepted)

    def test_rejects_unknown_instruction(self):
        result = self.g.ground("Fly to the red plant", ["track_light_1", "sofa_1"])
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "below_min_score")

    def test_rejects_ambiguous_candidates(self):
        result = self.g.ground("Go to the chair", ["chair_left", "chair_right"])
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "ambiguous")

    def test_goal_phrase_beats_source_distractor(self):
        result = self.g.ground(
            "Leave the cabinet within the display area and proceed to the entry door",
            ["display_cabinet_1", "door_1"],
        )
        self.assertEqual(result.target, "door_1")
        self.assertTrue(result.accepted)

    def test_instance_numbers_do_not_change_matching(self):
        result = self.g.ground("Go to door 17", ["door_1", "sofa_1"])
        self.assertEqual(result.target, "door_1")

    def test_empty_inputs_are_safe(self):
        result = self.g.ground("", [])
        self.assertFalse(result.accepted)
        self.assertIsNone(result.target)


if __name__ == "__main__":
    unittest.main()
