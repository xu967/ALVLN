"""Regression tests for the V7.1 review (v2) findings B01–B16 and gate P0-2."""
import http.client
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import dashboard  # noqa: E402
import scene_tool  # noqa: E402

GOOD = b'{"event":"train_step","step":1,"total_steps":2,"loss":0.5}\n'


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def start_server():
    port = free_port()
    threading.Thread(target=dashboard.serve, args=(port,), daemon=True).start()
    for _ in range(50):
        try:
            socket.create_connection(('127.0.0.1', port), timeout=0.1).close()
            return port
        except OSError:
            time.sleep(0.05)
    raise RuntimeError('dashboard did not start')


def get(port, path, host=None):
    conn = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
    conn.request('GET', path, headers={'Host': host} if host else {})
    resp = conn.getresponse()
    return resp.status, resp.read()


class MetricsRobustness(unittest.TestCase):
    def run_with(self, payload):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        base = Path(tmp.name)
        (base / 'r1').mkdir()
        (base / 'r1' / 'metrics.jsonl').write_bytes(GOOD + payload + GOOD)
        with patch.object(dashboard, 'RUNS', base):
            return dashboard.metrics_for('r1')

    def test_B01_nested_nonfinite_line_is_skipped_not_fatal(self):
        data = self.run_with(b'{"event":"train_step","extra":{"grad_norm":NaN}}\n')
        self.assertEqual(len(data['events']), 2)
        self.assertTrue(any('LOG_NONFINITE' in w for w in data['warnings']))

    def test_B02_deep_nesting_and_huge_int_are_skipped(self):
        for bad in (b'{"event":"x","v":' + b'[' * 30000 + b']' * 30000 + b'}\n',
                    b'{"event":"x","v":' + b'9' * 5000 + b'}\n'):
            self.assertEqual(len(self.run_with(bad)['events']), 2)


class SceneManifests(unittest.TestCase):
    def test_B03_B06_invalid_manifests_never_break_or_count_as_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / 'a.json').write_text('[]')
            (base / 'b.json').write_text('{broken')
            with patch.object(dashboard, 'SCENES', base):
                scenes = dashboard.scene_list()
                self.assertTrue(all(s['status'] == 'manifest_invalid' for s in scenes))
                check = {c['key']: c for c in dashboard.status()['checks']}['scenes']
                self.assertEqual(check['state'], 'blocked')


class RunListing(unittest.TestCase):
    def test_B07_total_is_reported_beyond_listing_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            for i in range(85):
                (base / f'r{i:03}').mkdir()
                (base / f'r{i:03}' / 'metrics.jsonl').write_bytes(GOOD)
            with patch.object(dashboard, 'RUNS', base):
                self.assertEqual(dashboard.status()['run_count'], 85)
                self.assertEqual(len(dashboard.runs_list()), dashboard.MAX_RUNS_LISTED)
                dashboard.metrics_for('r000')  # listed or not, a valid run stays readable


class HttpLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.port = start_server()

    def test_B05_foreign_host_header_is_rejected(self):
        self.assertEqual(get(self.port, '/api/audit', host='evil.example:%d' % self.port)[0], 421)
        self.assertEqual(get(self.port, '/api/audit')[0], 200)

    def test_B04_io_error_with_non_ascii_path_still_answers_500(self):
        def boom(_name):
            raise PermissionError(13, 'Permission denied', '/home/u/项目_工程/metrics.jsonl')
        with patch.object(dashboard, 'metrics_for', boom):
            self.assertEqual(get(self.port, '/api/metrics?run=x')[0], 500)


class LauncherContract(unittest.TestCase):
    def run_sh(self, *args):
        return subprocess.run([str(ROOT / 'run.sh'), *args], capture_output=True, text=True)

    def test_B08_non_ascii_run_id_rejected_like_dashboard(self):
        for bad in ('实验一', 'ｒｕｎ１'):
            r = self.run_sh('train', '--run-id', bad)
            self.assertIn('RUN_ID_INVALID', r.stderr)

    def test_B10_planned_subcommands_report_not_implemented(self):
        for args in (('board', 'doctor'), ('uart', 'dry-run')):
            r = self.run_sh(*args)
            self.assertEqual(r.returncode, 3)
            self.assertIn('FEATURE_NOT_IMPLEMENTED', r.stderr)


@unittest.skipUnless(shutil.which('g++'), 'g++ not available')
class TrainingMonitorHeader(unittest.TestCase):
    PROGRAM = r'''
#include "tinyvln/training_monitor.hpp"
#include <locale>
struct Comma : std::numpunct<char> {
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};
int main(int, char** argv) {
  std::locale::global(std::locale(std::locale::classic(), new Comma));
  tinyvln::TrainingMonitor m(argv[1]);
  m.Step(26000, 52000, 0.75, 0.5, 2e-4);
  std::cerr << "\nprobe=" << 0.37 << "\n";
  try { m.Step(26001, 52000, 0.0 / 0.0, 0.5, 2e-4); } catch (const std::exception&) { std::cerr << "THROWN\n"; }
}
'''

    def test_B12_B13_B14_locale_stream_state_and_error_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, exe, log = Path(tmp) / 't.cpp', Path(tmp) / 't', Path(tmp) / 'm.jsonl'
            src.write_text(self.PROGRAM)
            subprocess.run(['g++', '-std=c++17', '-I', str(ROOT / 'code/include'), str(src), '-o', str(exe)],
                           check=True, capture_output=True)
            r = subprocess.run([str(exe), str(log)], capture_output=True, text=True)
            events = [json.loads(line) for line in log.read_text().splitlines()]  # B13: valid JSON
            self.assertEqual(events[0]['step'], 26000)
            self.assertIn('probe=0.37', r.stderr)  # B12: std::cerr state untouched
            self.assertIn('THROWN', r.stderr)
            self.assertEqual(events[-1]['code'], 'TRAIN_NONFINITE_METRIC')  # B14: logged before throw


class SceneGate(unittest.TestCase):
    @unittest.expectedFailure  # P0-2: remove the decorator once the generator stops reserving a straight corridor
    def test_P0_2_scenes_require_more_than_flying_straight(self):
        blocked = sum(any(abs(o['y']) - o['r'] < 0.3 for o in scene_tool.create_obstacles('forest', seed, 60))
                      for seed in range(200))
        self.assertGreater(blocked / 200, 0.5)


if __name__ == '__main__':
    unittest.main()
