import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools import verify_vln_shared_forks as verify


def pair_at(key, x):
    return dict(pair=key, scene="839992", split="development", seed=1, common_budget=3,
                spawn=dict(x=x, y=0, z=1, yaw_deg=0), heading_source="native_common", selection_group="targeted",
                arms=[dict(route="A", out="A"), dict(route="B", out="B")])


def sample_arm(pair, spec):
    name = spec["route"]
    heading = 0 if name == "A" else 7
    spawn = pair["spawn"]
    pos = {k: spawn[k] for k in "xyz"}
    rows, trace = [], []
    for step in range(pair["common_budget"]):
        stamp = step * 500_000_000
        rows.append(dict(step=step, timestamp_ns=stamp, rgb_ts=stamp, pose_ts=stamp, pose_before=pos, yaw_before=0,
                         action_executed=1, action_expert=1, stop_requested=False, bearing_bin=heading, near_goal=0,
                         rgb_ok=True, collision=False, frame_quality={"low": False}, frame=f"{step}.png"))
        for j in range(5):
            trace.append(dict(t=step * .5 + (j + 1) * .1, step=step, pos=list(pos.values()),
                              vel=[.1, .1, 0], quat=[1, 0, 0, 0]))
    return dict(actions=rows, trace=trace, frames=[f"rgb{i}" for i in range(len(rows))],
                depths=[f"depth{i}" for i in range(len(rows))],
                camera=dict(width=320, height=240, mount={}), logs=Path("/virtual") / name, hashes={},
                manifest=dict(stop_radius=1., teacher_yaw_tol_deg=12., renderer=dict(near_plane_m=.2),
                              terminal_reason="budget_exhausted"), waypoint=dict(x=4, y=6 if name == "A" else 0, z=1),
                task=dict(route=name, instruction=name, goal_object=name, goal_instance=name, start_object="door",
                          start_instance="1", native_spawn=spawn))


def geometry(planner, state, waypoint, radius, camera, view, nav, action, tol, *, near_plane_m):
    bearing = 0 if waypoint["y"] else 7
    return dict(teacher_runtime_bin=bearing, runtime_near=0, local_bearing_bin=bearing,
                planning_status="astar", inside_actual_image=True)


def registered_design(root, pairs):
    join = root / "exp-vln-stage0-pairs-20261123/interiorgs_instance_join.json"
    join.parent.mkdir()
    join.write_text('{"rows":[]}', encoding="utf-8")
    rooms = {scene: split for split, values in verify.ROOMS.items() for scene in values}
    protocol = root / "protocol.json"
    protocol.write_text(json.dumps(dict(instance_join_sha256=verify.sha(join), acquisition={"steps": 8},
                             design={"cap": {s: 8 for s in rooms}},
                             roles={k: sorted(v) for k, v in verify.ROOMS.items()}, scene_sha256={s: s for s in rooms})))
    reports = {}
    for scene in rooms:
        selected = [dict(pair=p["pair"], status="selected_targeted") for p in pairs if p["scene"] == scene]
        reports[scene] = dict(scene=scene, sites=selected, counts=dict(eligible_sites=len(selected),
                              selected_targeted=len(selected), selected_control=0), source_scene_path="scene.json",
                              assets_sha256={"scene.json": scene})
    design = dict(schema=verify.DESIGN_SCHEMA, pairs=pairs, protocol_sha256=verify.sha(protocol), join_sha256=verify.sha(join),
                  scoring_steps=list(range(1, 8)), split_roles=rooms, rooms=reports)
    path = root / "design.json"
    path.write_text(json.dumps(design), encoding="utf-8")
    return path, design, join


class TestSharedWindows(unittest.TestCase):
    def setUp(self):
        self.pair = pair_at("a", 0)
        self.arms = [sample_arm(self.pair, arm) for arm in self.pair["arms"]]

    def test_equal_history_includes_final_control_window(self):
        self.assertEqual(list(verify.shared_steps(self.arms)), [(0, False), (1, False), (2, False)])
        self.arms[1]["trace"][-1]["quat"] = [0, 0, 0, 1]
        with self.assertRaisesRegex(ValueError, "history differs after step 2"):
            list(verify.shared_steps(self.arms))

    def test_stop_divergence_and_depth_before_dispatch(self):
        self.arms[1]["actions"][1]["stop_requested"] = True
        self.assertEqual(list(verify.shared_steps(self.arms)), [(0, False), (1, True)])
        self.arms[1]["depths"][1] = "changed"
        with self.assertRaisesRegex(ValueError, "depth differs at step 1"):
            list(verify.shared_steps(self.arms))

    def test_replay_checks_both_logged_bearings(self):
        room = dict(scene={"_nav_to_world": np.eye(4)}, planner=None)
        self.arms[1]["actions"][1]["bearing_bin"] = 2
        with patch.object(verify, "label_goal", side_effect=geometry), self.assertRaisesRegex(ValueError, "bearing/STOP"):
            verify.score_step(self.pair, room, self.arms, 1)

    def test_missing_arm_and_equal_budget_stay_in_denominators(self):
        pairs = [self.pair, pair_at("missing", 3)]
        for pair in pairs:
            pair["common_budget"] = 8

        def load(pair, arm, room, base):
            if pair["pair"] == "missing" and arm["route"] == "B":
                raise FileNotFoundError("B/episodes.jsonl")
            return sample_arm(pair, arm)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, design, _ = registered_design(root, pairs)
            room = dict(scene={"_nav_to_world": np.eye(4)}, planner=None, hashes={})
            out = root / "verify"
            with patch.object(verify, "load_room", return_value=room), patch.object(verify, "load_arm", side_effect=load), \
                    patch.object(verify, "label_goal", side_effect=geometry), patch.object(verify, "PROTOCOL_SHA", design["protocol_sha256"]):
                summary = verify.verify(path, out, root)
            reported = summary["all"]
            self.assertEqual(reported["requested_pairs"], 2)
            self.assertEqual(reported["requested_arms"], 4)
            self.assertEqual(reported["verified_arms"], 3)
            self.assertEqual(reported["pair_status"], {"same_through_budget": 1, "missing_or_unverified_arm": 1})
            self.assertEqual(reported["different_coarse_by_visible_points"], {"0": 0, "1": 0, "2": 7})
            self.assertEqual(reported["primary_distinct_starts"], 1)
            rows = verify.jsonl(out / "observations.jsonl")
            self.assertEqual([r["step"] for r in rows], list(range(1, 8)))
            self.assertEqual([r["step"] for r in rows if r["primary"]], [1])
            self.assertEqual(rows[0]["pre_action_velocity_xyz_mps"], [.1, .1, 0])
            self.assertEqual(verify.sha(out / "observations.jsonl"), summary["observations_sha256"])

    def test_omitted_room_request_or_changed_budget_fails_before_flight_io(self):
        self.pair["common_budget"] = 8
        with tempfile.TemporaryDirectory() as folder:
            path, design, join = registered_design(Path(folder), [self.pair])
            with patch.object(verify, "PROTOCOL_SHA", design["protocol_sha256"]):
                for change in ("schema", "protocol", "join", "room", "pair", "budget"):
                    bad = copy.deepcopy(design)
                    if change == "schema":
                        bad["schema"] = "unregistered"
                    elif change == "protocol":
                        bad["protocol_sha256"] = "wrong"
                    elif change == "join":
                        bad["join_sha256"] = "wrong"
                    elif change == "room":
                        del bad["rooms"]["841150"]
                    elif change == "pair":
                        bad["pairs"].clear()
                    else:
                        bad["pairs"][0]["common_budget"] = 7
                    with self.subTest(field=change), self.assertRaises(ValueError):
                        verify.registered_inputs(path, bad, join)


if __name__ == "__main__":
    unittest.main()
