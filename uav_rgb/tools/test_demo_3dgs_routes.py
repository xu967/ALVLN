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

    def test_complete_episodes_rejects_other_config(self):
        """run.json 记的几何 gid / 起终点 / FOV / hold / 教师 / 动力学与当前配置不一致时不能复用
        （默认 PILOT_OUT 里还躺着 t_z 修复前的 pilot v1：gid 893fc2a757638edf、离地 3.7 m）。"""
        with tempfile.TemporaryDirectory() as tmp:
            d = os.path.join(tmp, "route_0")
            info = {"episodes": [{"dir": "ep_00000"}], "failures": [], "scene": {"geometry_gid": "893fc2a757638edf"},
                    "renderer": {"fov_deg": 60.0}, "hold_s": 0.5, "policy": "astar", "dynamics": "cf_sitl",
                    "spawn": {"x": -14.0, "y": 0.0, "z": 2.0, "yaw_deg": 0.0}, "waypoint": {"x": -6.0, "y": 0.0, "z": 2.0}}
            _write(os.path.join(d, "run.json"), info)
            _write(os.path.join(d, "ep_00000", "episode_manifest.json"), {"error": None})
            expect = {"geometry_gid": "893fc2a757638edf", "fov_deg": 60.0, "hold_s": 0.5, "policy": "astar",
                      "dynamics": "cf_sitl", "spawn": info["spawn"], "waypoint": info["waypoint"]}
            self.assertEqual(demo.complete_episodes(d, 1, expect), [os.path.join(d, "ep_00000")])   # 同配置：复用
            for k, v in (("geometry_gid", "86b0f5cb953fbb03"), ("fov_deg", 64.9), ("hold_s", 1.0), ("policy", "teacher"),
                         ("dynamics", "cf_blackbox"), ("spawn", dict(info["spawn"], x=-2.0)), ("waypoint", dict(info["waypoint"], y=1.0))):
                self.assertIsNone(demo.complete_episodes(d, 1, dict(expect, **{k: v})), k)
            self.assertEqual(demo.complete_episodes(d, 1), [os.path.join(d, "ep_00000")])           # 不传 expect：老行为

    def test_no_duplicate_config(self):
        """配置区同名常量不能出现两次（上一版 PILOT_RETRY_MAX / anomaly_str 各写了两遍）。"""
        src = open(os.path.join(ROOT, "tools", "demo_3dgs_routes.py"), encoding="utf-8").read()
        self.assertEqual(src.count("\nPILOT_RETRY_MAX ="), 1)
        self.assertEqual(src.count("\ndef anomaly_str("), 1)

    def test_torch_cache_dir_named_by_gpu_capability(self):
        """AUDIT_SIM #5：gsplat JIT 缓存目录按本机算力拼（以前写死 extensions-sm89-20261106，换回 2080 Ti 后目录名与 cubin 对不上）。
        同一规则也用于 tools/scene_import_wanderland.py；两个脚本源码里都不能再有 sm89 字面量。"""
        self.assertEqual(os.path.basename(demo.torch_extensions_dir((7, 5))), "extensions-sm75")
        self.assertEqual(os.path.basename(demo.torch_extensions_dir((8, 9))), "extensions-sm89")
        self.assertEqual(os.path.dirname(demo.torch_extensions_dir((7, 5))),
                         os.path.join(demo.DATA_ROOT, "uav_rgb", "torch-cache"))
        for name in ("demo_3dgs_routes.py", "scene_import_wanderland.py"):
            src = open(os.path.join(ROOT, "tools", name), encoding="utf-8").read()
            self.assertNotIn('"extensions-sm89', src, name)       # 不再有写死的目录名字面量（注释里提历史可以）
            self.assertNotIn("sm_89 编", src, name)


if __name__ == "__main__":
    unittest.main()
