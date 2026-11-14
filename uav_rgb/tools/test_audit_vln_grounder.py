import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_vln_grounder import audit_scene


class TestAuditVlnGrounder(unittest.TestCase):
    def test_report_is_explicitly_marked_as_leaky_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.json"
            path.write_text(json.dumps({"name": "toy", "routes": [{"sage": {"instructions": [
                {"instruction_type": "x", "start": "cabinet_1", "end": "door_1",
                 "text": "Leave the cabinet and proceed to the entry door."}
            ]}}]}), encoding="utf-8")
            report = audit_scene(path)
            self.assertTrue(report["leakage_warning"])
            self.assertEqual(report["instruction_count"], 1)
            self.assertEqual(report["accepted_target_hits"], 1)


if __name__ == "__main__":
    unittest.main()
