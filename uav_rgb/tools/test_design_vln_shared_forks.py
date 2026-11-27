from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tools import design_vln_shared_forks as design


def route(name, goal, yaw=0.0, start="11", end=None, spawn=(0, 0, 1), text=None):
    ins = dict(start="desk", end=name, instruction_type="Relative_Relationship", text=text or f"Go to {name}.")
    source = dict(name=name, waypoint=dict(zip("xyz", goal)), spawn=dict(zip(("x", "y", "z", "yaw_deg"), (*spawn, yaw))),
                  ref_path=[list(spawn[:2]), list(goal[:2])], sage=dict(split="train", instructions=[ins]))
    return dict(route=name, text=ins["text"], instruction=ins, start_id=start, end_id=end or name,
                source_instruction_index=0, native_route=source)


class TestSharedDesign(unittest.TestCase):
    def test_native_identity_and_goal_separation(self):
        a = route("a", (2, 0, 1))
        b = route("b", (0, 2, 1), yaw=20, spawn=(1e-6, 0, 1))
        c = route("c", (-2, 0, 1), start="12")
        clone = route("d", (2.5, 0, 1), end="a")
        counts = Counter()
        sites = design.candidate_sites([a, b, c, clone], 0.5, counts)
        pairs = sites[(0, 0, 1)]
        self.assertEqual([(x["route"], y["route"]) for x, y in pairs], [("a", "b"), ("b", "d")])
        self.assertEqual(counts["different_start"], 3)
        self.assertEqual(counts["same_goal_or_sentence"], 1)
        close = route("close", (3, 0, 1))
        self.assertFalse(design.candidate_sites([a, close], 0.5, counts))

    def test_headings_and_native_control(self):
        a, b = route("a", (3, 0, 1), yaw=180), route("b", (0, 3, 1), yaw=-179)
        pair = a, b
        headings = design.candidate_headings(pair, (160, -160))
        self.assertEqual(headings[:2], [-180, -179])
        self.assertLessEqual(len(headings), 9)
        self.assertIsNone(design.choose_control("839992", {(0, 0, 1): [pair]}))
        b["native_route"]["spawn"]["yaw_deg"] = -180
        pos, selected = design.choose_control("839992", {(0, 0, 1): [pair]})
        self.assertEqual(selected, pair)
        control = design.pair_spec("839992", "control", pos, selected, 180, None)
        self.assertEqual((control["selection_group"], control["heading_source"]), ("control", "native_common"))
        change = design.pair_spec("839992", "targeted", pos, selected, 171, {"step": 3})
        self.assertEqual((change["heading_source"], change["seed"]), ("designed_common", control["seed"]))

    def test_first_unequal_action_uses_only_the_common_pre_action_state(self):
        class Teacher:
            stop_requested, last_path = False, [(0, 0), (1, 1)]

            def __init__(self, sequence):
                self.sequence = iter(sequence)

            def __call__(self, rgb, depth, state):
                return next(self.sequence), None

        pair = route("a", (3, 0, 1)), route("b", (0, 3, 1))
        drone = Mock(x=0, y=0, z=1, vx=0, vy=0, vz=0, quat=[1, 0, 0, 0])
        drone.step.return_value = {"pose": (0, 0, 1)}
        planner = Mock()
        planner.occ.sweep.return_value = False, None
        scene = {"reach_radius_m": 0.4}
        with patch.object(design.firmware, "CfSitlDynamics", return_value=drone), \
                patch.object(design.planner_mod, "OccAstarTeacher", side_effect=[Teacher([1, 2]), Teacher([1, 1])]), \
                patch.object(design, "local_view", return_value=({"bins": [0, 1]}, "both_inside_different")) as view:
            hits, reasons = design.simulate(planner, scene, None, None, (0, 0, 1), 30, pair, 101)
        drone.step.assert_called_once_with(1, 0.5)
        drone.close.assert_called_once()
        view.assert_called_once()
        self.assertEqual((hits[0]["step"], hits[0]["common_prefix_actions"], hits[0]["actions"]), (1, [1], [2, 1]))
        self.assertEqual(reasons["first_unequal_action"], 1)

    def test_fixtures_keep_original_instruction_endpoint_and_camera(self):
        pair = route("a", (3, 0, 1)), route("b", (0, 3, 1), yaw=40)
        record = design.pair_spec("839992", "targeted", (0, 0, 1), pair, 20, {"step": 1})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = dict(ply="view.ply", collision=dict(geometry="mesh.ply"), camera=dict(fov_deg=60),
                         routes=[arm["native_route"] for arm in pair])
            src = root / "scene.json"
            src.write_text(json.dumps(scene))
            report = dict(source_scene_path=str(src), assets_sha256={str(src): design.file_sha(src)}, pairs=[record])
            design.write_fixtures(root, root, root, report)
            for arm, original in zip(record["arms"], pair):
                fixture = Path(arm["fixture"])
                doc = json.loads(fixture.read_bytes())
                self.assertEqual(arm["fixture_sha256"], design.file_sha(fixture))
                self.assertEqual(doc["spawn"], doc["routes"][0]["spawn"])
                self.assertEqual(doc["spawn"]["yaw_deg"], 20)
                self.assertEqual(doc["max_steps"], 8)
                self.assertEqual(doc["routes"][0]["sage"]["instructions"], [original["instruction"]])
                self.assertEqual(doc["waypoint"], original["native_route"]["waypoint"])
                self.assertEqual(doc["camera"], scene["camera"])
                self.assertIn(arm["out"], arm["argv"])


if __name__ == "__main__":
    unittest.main()
