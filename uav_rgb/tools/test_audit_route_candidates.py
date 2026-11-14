#!/usr/bin/env python3
"""audit_route_candidates 的标准库单测。"""
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("audit_route_candidates", ROOT / "audit_route_candidates.py")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class TestAuditRouteCandidates(unittest.TestCase):
    def write_scene(self, root):
        scene_dir = pathlib.Path(root) / "scene_a"
        scene_dir.mkdir()
        scene = {"name": "scene_a", "routes": [{"name": "r_long"}, {"name": "r_short"}]}
        checks = [
            {"name": "r_long", "trajectory_id": 7, "split": "train", "sage_path_m": 12.5,
             "astar_path_m": 13.1, "max_steps": 120, "turn_deg": 90.0, "min_clearance_m": 0.4,
             "n_instructions": 2, "ok": True, "reasons": []},
            {"name": "r_short", "trajectory_id": 8, "split": "train", "sage_path_m": 4.0,
             "astar_path_m": 4.2, "max_steps": 50, "turn_deg": 30.0, "min_clearance_m": 0.5,
             "n_instructions": 1, "ok": True, "reasons": []},
            {"name": "r_rejected", "trajectory_id": 9, "split": "train", "sage_path_m": 15.0,
             "astar_path_m": 20.0, "ok": False, "reasons": ["净空不足"]},
        ]
        (scene_dir / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
        (scene_dir / "routes_check.json").write_text(json.dumps(checks), encoding="utf-8")
        return scene_dir / "scene.json"

    def test_selects_only_accepted_in_scene_long_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = AUDIT.audit_scene(str(self.write_scene(tmp)), min_path_m=10.0, top_k=5)
        self.assertEqual(report["n_candidates"], 3)
        self.assertEqual(report["n_accepted"], 2)
        self.assertEqual(report["n_accepted_in_scene"], 2)
        self.assertEqual([row["name"] for row in report["long_candidates"]], ["r_long"])
        self.assertEqual(report["filter_reason_counts"], {"净空不足": 1})
        self.assertEqual(report["length_summary_m"], {"p50": 8.25, "p90": 11.65, "max": 12.5})

    def test_missing_check_record_is_integrity_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = self.write_scene(tmp)
            check_path = pathlib.Path(scene_path).with_name("routes_check.json")
            checks = json.loads(check_path.read_text(encoding="utf-8"))
            check_path.write_text(json.dumps([checks[0]]), encoding="utf-8")
            report = AUDIT.audit_scene(str(scene_path))
        self.assertTrue(report["integrity_errors"])
        self.assertIn("r_short", report["integrity_errors"][0])

    def test_invalid_top_level_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_path = self.write_scene(tmp)
            pathlib.Path(scene_path).with_name("routes_check.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                AUDIT.audit_scene(str(scene_path))


if __name__ == "__main__":
    unittest.main()
