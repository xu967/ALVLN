import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from tools import probe_vln_waypoints as probe


class TestWaypointProbe(unittest.TestCase):
    def test_first_record_per_pair_checks_decoded_pixels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = Image.new("RGB", (4, 3), (20, 90, 150))
            image.save(root / "view.png", compress_level=0)
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            row = {"pair": "a", "scene": "839992", "split": "development", "step": 8, "arm": 0,
                   "rgb": "view.png", "rgb_sha256": digest, "shared_executed_state": False,
                   "routes": ["a1", "a2"], "actions": [2, 1], "instructions": ["Go to the lamp.", "Go to the door."]}
            manifest = root / "development.jsonl"
            rows = [row, dict(row, step=2), dict(row, pair="b", step=5)]
            manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
            image.save(root / "view.png", compress_level=9)
            manifest_digest, samples = probe.load_pairs(manifest)
            self.assertEqual(manifest_digest, hashlib.sha256(manifest.read_bytes()).hexdigest())
            self.assertEqual([(p["pair"], p["step"]) for p, _ in samples], [("a", 8), ("b", 5)])
            self.assertEqual(samples[0][0]["queries"][0]["teacher"], "L")
            Image.new("RGB", image.size).save(root / "view.png")
            with self.assertRaisesRegex(ValueError, "像素 SHA256"):
                probe.load_pairs(manifest)

    def test_same_pixels_reset_each_query_and_unknown_counts_as_wrong(self):
        observation = importlib.import_module("sim.3dgs.grounder").VisualGrounding
        samples = [({"pair": "a", "queries": [{"instruction": "lamp", "teacher": "L"},
                                              {"instruction": "door", "teacher": "C"}]},
                    np.zeros((2, 2, 3), dtype=np.uint8))]
        replies = [observation(1, 0, 1.0, "aisle", goal_mode="subgoal"),
                   observation(0, 1, 1.0, "door", goal_mode="final")]
        for condition, n_correct in (("local-waypoint", 1), ("baseline", 2)):
            with self.subTest(condition=condition):
                worker = mock.Mock()
                worker.ground_step.side_effect = replies
                results = probe.query_pairs(samples, worker, condition)
                self.assertEqual(worker.method_calls, [mock.call.reset_episode(),
                                 mock.call.ground_step(samples[0][1], "lamp"), mock.call.reset_episode(),
                                 mock.call.ground_step(samples[0][1], "door")])
                report = probe.scores(results)
                self.assertEqual((report["correct"], report["requests"], report["both_correct"], report["unknown"]),
                                 (n_correct, 2, n_correct - 1, 2 - n_correct))
                self.assertEqual(report["by_teacher"]["L"], {"correct": 1, "total": 1})
                center = probe.scores(results, always_center=True)
                self.assertEqual((center["correct"], center["both_correct"], center["unknown"]), (1, 0, 0))

    def test_first_token_scored_apart_from_strict_json_and_n_remains_unknown(self):
        observation = importlib.import_module("sim.3dgs.grounder").VisualGrounding
        sample = {"queries": [{"instruction": "lamp", "teacher": "L"}, {"instruction": "door", "teacher": "C"}]}
        model = mock.Mock()

        def ground(frame, instruction):
            model.last_waypoint_probe = {"first_token_choice": {"lamp": "L", "door": "N"}[instruction],
                                         "raw_generation": "incomplete"}
            return observation(8, 0, 0., "parse_fallback", goal_mode="search")

        model.ground_step.side_effect = ground
        results = probe.query_pairs([(sample, np.zeros((3, 3, 3), np.uint8))], model, "local-waypoint")
        strict = probe.scores(results)
        first = probe.scores(results, prediction_key="first_token_prediction")
        self.assertEqual((strict["correct"], strict["unknown"]), (0, 2))
        self.assertEqual((first["correct"], first["unknown"], first["both_correct"]), (1, 1, 0))
        self.assertEqual(model.ground_step.call_count, 2)
        self.assertEqual(results[0]["queries"][0]["waypoint_probe"]["raw_generation"], "incomplete")


def write_teacher_run(root):
    folder = root / "dev" / "route_0"
    folder.mkdir(parents=True)
    goal = {"x": 3.0, "y": 0.0}
    scene = {"name": "dev", "routes": [{"name": "route", "waypoint": goal,
             "sage": {"split": "train", "instructions": [{"text": "Go to the lamp.", "end": "lamp_1"}]}}]}
    original = importlib.import_module("sim.3dgs.vln").episodes(scene)[0]
    task = dict(original, counterfactual="correct", supplied_text=original["text"],
                supplied_instruction_key=original["key"], supplied_goal_waypoint=goal, blind_rgb=False)
    episode = dict(task, scene="dev", seed=1, start_pose={"x": 0.0}, steps=5, dir=str(folder))
    manifest = {"scene": {"name": "dev", "spawn": episode["start_pose"]}, "route": {"name": "route"}, "vln": task, "seed": 1,
                "waypoint": goal, "num_steps": 5, "stop_radius": 1.0, "renderer": {"backend": "gsplat"},
                "executed_by": "teacher", "planner": "astar_occ", "camera_source": "gsplat",
                "camera": {"width": 4, "height": 3}, "actions_ref": "actions.jsonl", "completed": True,
                "validity": {"camera_verified": True, "physics_verified": True, "episode_executed": True},
                "teacher_clearance": "braking", "dynamics_backend": "cf_sitl"}
    report = {"policy": "teacher", "dynamics": "cf_sitl", "counterfactual": "correct", "blind_rgb": False,
              "failures": [], "overall": {"n_requested": 1}}
    for path, value in ((folder / "scene.json", scene), (folder / "episode_manifest.json", manifest),
                        (root / "vln_eval.json", report), (root / "episodes.jsonl", episode)):
        path.write_text(json.dumps(value) + "\n")
    actions = []
    for step, bearing in enumerate([2, 7, 7, 0, 1]):
        filename = f"rgb_{step:05d}.png"
        Image.new("RGB", (4, 3), (step, 50, 30)).save(folder / filename)
        near = int(step == 4)
        actions.append({"step": step, "frame": filename, "rgb_ok": True, "pose_ts": step, "rgb_ts": step,
                        "timestamp_ns": step, "action_executed": 1, "action_expert": 1,
                        "bearing_bin": bearing, "near_goal": near, "pose_before": {"x": 3.0 * near, "y": 0.0}})
    (folder / "actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in actions))
    return folder


class TestTeacherWaypointProbe(unittest.TestCase):
    def test_first_present_bins_exclude_near_and_measure_bearing_independently_of_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_teacher_run(root)
            meta, samples = probe.load_teacher_runs([root])
            self.assertEqual(meta["scope"], "offline_astar_local_bearing")
            self.assertEqual(meta["routes"][0]["outside_bearing_bins"], {0: 1, 2: 1, 7: 2})
            self.assertEqual(meta["routes"][0]["near_steps"], 1)
            self.assertEqual([(s["step"], s["queries"][0]["teacher"]) for s, _ in samples], [(1, "R"), (3, "C")])
            self.assertEqual([int(frame[0, 0, 0]) for _, frame in samples], [1, 3])
            result = probe.scores([s for s, _ in samples], always_center=True, paired=False)
            self.assertEqual((result["requests"], result["correct"], result["balanced_accuracy"]), (2, 1, 1 / 3))
            self.assertEqual(result["recall_by_teacher"], {"C": 1.0, "L": 0.0, "R": 0.0})
            self.assertEqual(result["by_teacher"]["L"]["total"], 0)
            self.assertIsNone(probe.scores([], paired=False)["balanced_accuracy"])
            self.assertNotIn("both_correct", result)
            self.assertNotIn("pairs", result)
            with self.assertRaisesRegex(ValueError, "重复教师"):
                probe.load_teacher_runs([root, root])

    def test_refuses_mismatched_renderer_instruction_or_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = write_teacher_run(root)
            manifest_path = folder / "episode_manifest.json"
            original = json.loads(manifest_path.read_bytes())
            for change, message in (({"renderer": {"backend": "mock"}}, "gsplat"),
                                    ({"validity": dict(original["validity"], camera_verified=False)}, "未核验"),
                                    ({"validity": dict(original["validity"], physics_verified=False)}, "未核验"),
                                    ({"vln": dict(original["vln"], supplied_text="Go elsewhere.")}, "原句")):
                with self.subTest(change=change):
                    manifest_path.write_text(json.dumps(dict(original, **change)))
                    with self.assertRaisesRegex(ValueError, message):
                        probe.load_teacher_runs([root])
            manifest_path.write_text(json.dumps(original))
            actions_path = folder / "actions.jsonl"
            actions = [json.loads(line) for line in actions_path.read_bytes().splitlines()]
            actions[1]["frame"] = "rgb_00003.png"
            actions_path.write_text("".join(json.dumps(a) + "\n" for a in actions))
            with self.assertRaisesRegex(ValueError, "RGB"):
                probe.load_teacher_runs([root])

    def test_existing_and_competing_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch("sys.stderr", new_callable=io.StringIO):
            root = Path(tmp)
            folder = write_teacher_run(root)
            unselected_rgb = folder / "rgb_00000.png"
            before = unselected_rgb.read_bytes()
            with mock.patch.object(probe, "load_teacher_runs") as load:
                with self.assertRaises(SystemExit):
                    probe.main(["--teacher-run", tmp, "--out", str(unselected_rgb)])
                load.assert_not_called()
            self.assertEqual(unselected_rgb.read_bytes(), before)

            output = root / "result.json"
            link = os.link

            def publish_first(src, dest):
                output.write_text("first writer")
                link(src, dest)

            with mock.patch.object(probe.os, "link", side_effect=publish_first), self.assertRaises(SystemExit):
                probe.main(["--teacher-run", tmp, "--check", "--out", str(output)])
            self.assertEqual(output.read_text(), "first writer")
            self.assertEqual({p.name for p in root.iterdir()}, {"dev", "vln_eval.json", "episodes.jsonl", "result.json"})


if __name__ == "__main__":
    unittest.main()
