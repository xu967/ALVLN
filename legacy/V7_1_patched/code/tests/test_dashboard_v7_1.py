"""V7.1 dashboard tests: truthful state, log safety, artifact boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import dashboard


class DashboardV71Tests(unittest.TestCase):
    def test_status_no_false_flight_claim(self):
        data = dashboard.status()
        lookup = {e['key']: e for e in data['checks']}
        self.assertEqual(lookup['physics']['state'], 'not_implemented')
        self.assertEqual(lookup['cf_sil']['state'], 'not_implemented')
        self.assertEqual(lookup['firmware']['state'], 'frozen')
        self.assertEqual(lookup['edge']['state'], 'not_implemented')
        self.assertTrue(data['read_only'])
        self.assertTrue(data['no_fake_metrics'])

    def test_generated_scenes_are_not_flight_simulator(self):
        for scene in dashboard.scene_list():
            self.assertIn(scene['status'], ('not_a_flight_simulator', 'manifest_invalid'))

    def test_no_run_no_fabricated_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(dashboard, 'RUNS', Path(tmp)):
                self.assertEqual(dashboard.runs_list(), [])
                with self.assertRaises(ValueError):
                    dashboard.metrics_for('fictional_success')

    def test_runs_reject_symlink_and_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'safe_run').mkdir()
            (base / 'safe_run' / 'metrics.jsonl').write_text('', encoding='utf-8')
            (base / 'invalid-run').mkdir()
            (base / 'invalid-run' / 'metrics.jsonl').write_text('', encoding='utf-8')
            (base / 'linked').symlink_to(base / 'safe_run', target_is_directory=True)
            with patch.object(dashboard, 'RUNS', base):
                self.assertEqual(dashboard.runs_list(), ['safe_run'])
                for n in ('../safe_run', 'invalid-run', 'linked', ''):
                    with self.assertRaises(ValueError):
                        dashboard.metrics_for(n)

    def test_reads_only_completed_lines_and_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            d = base / 'myrun'; d.mkdir()
            (d / 'metrics.jsonl').write_bytes(
                b'{"event":"train_step","step":3,"total_steps":9,"loss":0.7}\n'
                b'not-json\n'
                b'{"event":"train_step","step":4')
            with patch.object(dashboard, 'RUNS', base):
                result = dashboard.metrics_for('myrun')
                self.assertEqual(len(result['events']), 1)
                self.assertEqual(result['events'][0]['step'], 3)
                self.assertTrue(any('LOG_PARTIAL_WRITE' in x for x in result['warnings']))
                self.assertTrue(any('LOG_JSON_INVALID' in x for x in result['warnings']))

    def test_nans_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); d=base/'run_1';d.mkdir()
            (d/'metrics.jsonl').write_text('{"event":"train_step","loss":NaN}\n', encoding='utf-8')
            with patch.object(dashboard,'RUNS',base):
                data=dashboard.metrics_for('run_1')
                self.assertEqual(data['events'],[])
                self.assertTrue(any('LOG_NONFINITE' in x for x in data['warnings']))

    def test_audit_explicitly_states_limits(self):
        report = dashboard.audit()
        self.assertEqual(report['schema'], 'tinyvln.audit.v1')
        self.assertTrue(any('没有修改' in s for s in report['explicit_limitations']))
        self.assertTrue(any('闭环' in s for s in report['explicit_limitations']))

    def test_html_has_five_pages_without_fabricated_metrics(self):
        html = dashboard.HTML.read_text(encoding='utf-8')
        for key in ['overview','training','simulation','deployment','auditing']:
            self.assertIn('id="page-'+key+'"',html)
        self.assertIn('尚无可展示的真实训练记录',html)
        self.assertNotIn('演示成功率: 95%', html)


if __name__ == '__main__':
    unittest.main()
