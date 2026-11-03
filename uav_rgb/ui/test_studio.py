#!/usr/bin/env python3
"""Headless regression tests for UAV Navigation Studio data/geometry helpers."""
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import studio


class StudioHelperTests(unittest.TestCase):
    def test_parse_pose_v(self):
        msg = {"pose": [{"name": "x3", "position": {"x": 1.0, "y": -2.0, "z": 0.7},
                         "orientation": {"x": 0, "y": 0, "z": 0.70710678, "w": 0.70710678},
                         "header": {"stamp": {"sec": 12, "nsec": 345}}}]}
        pose = studio.parse_pose_msg(msg)
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose[0], 1.0)
        self.assertAlmostEqual(pose[1], -2.0)
        self.assertAlmostEqual(pose[2], 0.7)
        self.assertAlmostEqual(pose[3], 1.5707963, places=5)
        self.assertEqual(pose[4], 12_000_000_345)
        self.assertEqual(studio.pose_stamp_ns(msg), 12_000_000_345)

    def test_parse_pose_single(self):
        msg = {"position": {"x": 0, "y": 0, "z": 1},
               "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}
        pose = studio.parse_pose_msg(msg)
        self.assertEqual(pose[:3], (0.0, 0.0, 1.0))
        self.assertAlmostEqual(pose[3], 0.0)

    def test_parse_sdf_collision_footprints(self):
        sdf = '''<sdf version="1.6"><world name="test">
          <model name="ground_plane"><link name="l"><collision name="c"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision></link></model>
          <model name="x3"><link name="l"><collision name="c"><geometry><box><size>1 1 1</size></box></geometry></collision></link></model>
          <model name="tree_1"><pose>2 3 1 0 0 0.5</pose><link name="l"><collision name="c"><geometry><box><size>0.4 0.6 2</size></box></geometry></collision></link></model>
          <model name="post"><pose>4 -1 0 0 0 0</pose><link name="l"><collision name="c"><geometry><cylinder><radius>0.2</radius><length>2</length></cylinder></geometry></collision></link></model>
        </world></sdf>'''
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "world.sdf"
            p.write_text(sdf, encoding="utf-8")
            obs = studio.parse_sdf_obstacles(p)
        self.assertEqual(len(obs), 2)
        tree = next(o for o in obs if o["name"] == "tree_1")
        self.assertEqual(tree["kind"], "box")
        self.assertAlmostEqual(tree["x"], 2.0)
        self.assertAlmostEqual(tree["y"], 3.0)
        self.assertAlmostEqual(tree["yaw"], 0.5)
        post = next(o for o in obs if o["name"] == "post")
        self.assertEqual(post["kind"], "circle")
        self.assertAlmostEqual(post["r"], 0.2)

    def test_summary_keeps_missing_test_null_and_separates_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "results.csv").write_text(
                "epoch,train_ce,dev_ce,dev_accuracy\n1,1.0,1.2,0.7\n2,0.7,0.9,0.8\n",
                encoding="utf-8")
            events = [
                {"event": "train_step", "step": 1},
                {"event": "validation", "split": "val", "loss": 0.9,
                 "action_accuracy": 0.8, "macro_f1": 0.72, "balanced_accuracy": 0.7},
                {"event": "status", "message": "best checkpoint: epoch 2 macro_f1=0.72"},
            ]
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
            (run / "ckpt_best.bin").write_bytes(b"checkpoint")
            meta = {"run_id": "test_run", "started_at": "2026-11-04T00:00:00Z",
                    "started_monotonic": 1.0, "config": {"epochs": 2, "text_dim": 16}}
            summary = studio.summarize_run(run, meta, 0, "finished")
        m = summary["metrics"]
        self.assertEqual(m["epochs_completed"], 2)
        self.assertAlmostEqual(m["best_validation_macro_f1"], 0.72)
        self.assertEqual(m["best_checkpoint_epoch"], 2)
        self.assertIsNone(m["test_macro_f1"])
        self.assertFalse(m["test_evaluated"])
        self.assertTrue(summary["artifacts"]["checkpoint_best"])
        self.assertIn("not navigation success rate", summary["notes"][0])

    def test_current_world_template_has_three_camera_sensors_and_keeps_fpv(self):
        template = studio.REPO_ROOT / "sim" / "worlds" / "uav_rgb.sdf"
        root = ET.parse(template).getroot()
        fpv = root.find(".//sensor[@name='camera']")
        chase = root.find(".//sensor[@name='chase_camera']")
        top = root.find(".//sensor[@name='top_camera']")
        self.assertIsNotNone(fpv)
        self.assertIsNotNone(chase)
        self.assertIsNotNone(top)
        self.assertEqual(fpv.findtext("topic"), "/uav/camera")
        self.assertEqual(chase.findtext("topic"), "/uav/chase_camera")
        self.assertEqual(chase.findtext("camera/image/width"), "640")
        self.assertEqual(chase.findtext("camera/image/height"), "360")
        self.assertEqual(top.findtext("topic"), "/uav/top_camera")
        self.assertEqual(top.findtext("camera/image/width"), "480")
        self.assertEqual(top.findtext("camera/image/height"), "360")

    def test_summary_derives_per_class_recall_and_report_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            confusion = [8, 2, 0, 0, 1, 9, 0, 0, 0, 1, 8, 1, 0, 0, 2, 8]
            event = {"event": "validation", "split": "test", "action_accuracy": 0.82,
                     "macro_f1": 0.81, "balanced_accuracy": 0.80, "confusion": confusion}
            (run / "metrics.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
            meta = {"run_id": "recall_test", "config": {"text_dim": 0}}
            summary = studio.summarize_run(run, meta, 0, "finished")
            studio.write_experiment_report(run, summary)
            report = (run / "experiment_report.md").read_text(encoding="utf-8")
        self.assertEqual(len(summary["metrics"]["test_per_class_recall"]), 4)
        self.assertAlmostEqual(summary["metrics"]["test_per_class_recall"][0]["recall"], 0.8)
        self.assertIn("Test confusion matrix", report)
        self.assertIn("SAFE_NEUTRAL", report)

    def test_read_jsonl_keeps_tail_for_final_test_events(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.jsonl"
            p.write_text("".join(json.dumps({"event": "train_step", "step": i}) + "\n"
                                   for i in range(10)) +
                         json.dumps({"event": "validation", "split": "test", "macro_f1": 0.7}) + "\n",
                         encoding="utf-8")
            rows = studio.read_jsonl(p, max_lines=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1].get("split"), "test")

    def test_read_jsonl_skips_bad_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.jsonl"
            p.write_text('{"event":"ok"}\nnot-json\n{"event":"done"}\n', encoding="utf-8")
            rows = studio.read_jsonl(p)
        self.assertEqual([x["event"] for x in rows], ["ok", "done"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
