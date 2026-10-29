"""Infrastructure tests; no simulator, student model or GPU required."""
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import dashboard  # noqa: E402
import scene_tool  # noqa: E402


class InfrastructureTest(unittest.TestCase):
    def test_scene_deterministic_and_well_formed(self):
        from xml.etree import ElementTree as ET
        (ROOT / 'artifacts').mkdir(exist_ok=True)  # artifacts/ is gitignored: absent in a fresh clone
        with tempfile.TemporaryDirectory(dir=ROOT / 'artifacts') as d:
            output = Path(d)
            first = scene_tool.generate('forest', 123, 30, output)
            a = first.read_bytes()
            m = json.loads(first.with_suffix('.json').read_text())
            assert m['sha256'] == hashlib.sha256(a).hexdigest()
            assert m['schema_version'] == 'scene-v0.1'
            ET.fromstring(a)
            second = scene_tool.generate('forest', 123, 30, output)
            self.assertEqual(a, second.read_bytes())

    def test_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            scene_tool.validate_path(Path('/tmp/escape'))

    def test_rejects_invalid_obstacle_count(self):
        with self.assertRaises(ValueError):
            scene_tool.generate('forest', 42, -1, ROOT / 'artifacts' / 'scenes')

    def test_refuses_nonexistent_training_binary(self):
        if (ROOT / 'build' / 'tinyvln_train').exists():
            self.skipTest('trainer is built; this test must never launch real training')
        p = subprocess.run([str(ROOT / 'run.sh'), 'train', '--run-id', 'forbidden_test'],
                           text=True, capture_output=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn('BINARY_NOT_BUILT', p.stderr)

    def test_dashboard_empty_no_fake_scores(self):
        # All newly created run folders lacking metrics must be ignored.
        run_ids = dashboard.runs_list()
        self.assertIsInstance(run_ids, list)
        self.assertTrue(all((ROOT / 'artifacts' / 'runs' / name / 'metrics.jsonl').is_file()
                            for name in run_ids))


if __name__ == '__main__':
    unittest.main()
