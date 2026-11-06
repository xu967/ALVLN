#!/usr/bin/env python3
"""tools/demo_3dgs_routes.py 的断点续跑判定（complete_episodes）小测，不需要 GPU/torch。

  python tools/test_demo_3dgs_routes.py
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
spec = importlib.util.spec_from_file_location("demo_3dgs_routes", os.path.join(ROOT, "tools", "demo_3dgs_routes.py"))
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


class TestResume(unittest.TestCase):
    def test_complete_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "route_0")
            self.assertIsNone(demo.complete_episodes(d, 2))                      # 没跑过
            _write(os.path.join(d, "run.json"), {"episodes": [{"dir": "ep_00000"}, {"dir": "ep_00001"}], "failures": []})
            _write(os.path.join(d, "ep_00000", "episode_manifest.json"), {"error": None})
            # ep_00001 不在（已隔离到 quarantine/）：仍算完整，只返回还在的
            self.assertEqual(demo.complete_episodes(d, 2), [os.path.join(d, "ep_00000")])
            self.assertIsNone(demo.complete_episodes(d, 3))                      # 集数不够
            _write(os.path.join(d, "ep_00001", "episode_manifest.json"), {"error": "RuntimeError: x"})
            self.assertIsNone(demo.complete_episodes(d, 2))                      # 有集出错
            _write(os.path.join(d, "ep_00001", "episode_manifest.json"), {"error": None})
            _write(os.path.join(d, "run.json"), {"episodes": [{"dir": "ep_00000"}, {"dir": "ep_00001"}],
                                                 "failures": [{"episode": 1}]})
            self.assertIsNone(demo.complete_episodes(d, 2))                      # run.json 记了失败

    def test_no_duplicate_config(self):
        """配置区同名常量不能出现两次（上一版 PILOT_RETRY_MAX / anomaly_str 各写了两遍）。"""
        src = open(os.path.join(ROOT, "tools", "demo_3dgs_routes.py"), encoding="utf-8").read()
        self.assertEqual(src.count("\nPILOT_RETRY_MAX ="), 1)
        self.assertEqual(src.count("\ndef anomaly_str("), 1)


if __name__ == "__main__":
    unittest.main()
