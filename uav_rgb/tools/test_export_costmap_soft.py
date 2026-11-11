#!/usr/bin/env python3
"""tools/export_costmap_soft.py 的版本门：costmap 教师 v1（点目标）与 v2（过柱阵后线目标）都收，各集必须同版本。

W1-7 把 TEACHER_COST_VERSION 从 v1 升到 v2（costmap 的目标也在过柱阵后取线）。升版前写出的 costmap 集
（manifest.teacher_version=uavrgb.teacher_cost.v1）仍要能导；v1 与 v2 混在一个 --src 里要报错（出口区标签口径不同）；
npz 的 teacher_version 键写各集实际的版本，不写当前常量。

  python3 tools/test_export_costmap_soft.py
EXPORTER 环境变量可指向另一份导出器脚本（先红后绿时用旧版对照）。
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTER = os.environ.get("EXPORTER") or os.path.join(REPO, "tools", "export_costmap_soft.py")
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import teacher  # noqa: E402

V1, V2 = "uavrgb.teacher_cost.v1", "uavrgb.teacher_cost.v2"


def write_episode(root, split, eid, version, planner="costmap"):
    d = os.path.join(root, split, f"{eid:04d}")
    os.makedirs(d)
    with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"episode_id": eid, "planner": planner, "teacher_version": version}, f)
    rows = [
        {"step": 0, "action_expert": 1, "q_costs": [40.0, 10.0, 12.0, 13.0]},
        {"step": 1, "action_expert": 2, "q_costs": [40.0, 11.0, 9.0, 13.0]},
        {"step": 2, "action_expert": 1, "q_costs": None, "costmap_fallback": "goal_stop"},
    ]
    with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


class ExportVersionGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "collection")
        self.out = os.path.join(self.tmp.name, "soft.npz")

    def run_export(self):
        return subprocess.run([sys.executable, EXPORTER, "--src", self.src, "--alpha", "0.5", "--out", self.out],
                              capture_output=True, text=True)

    def test_constant_is_v2(self):
        self.assertEqual(teacher.TEACHER_COST_VERSION, V2)
        self.assertEqual(teacher.recomputable_versions(costmap=True), (V1, V2))

    def test_v1_collection_still_exports_and_meta_keeps_v1(self):
        write_episode(self.src, "train", 1, V1)
        write_episode(self.src, "val", 2, V1)
        r = self.run_export()
        self.assertEqual(r.returncode, 0, r.stderr)
        z = np.load(self.out)
        self.assertEqual(str(z["teacher_version"]), V1)
        self.assertEqual(int(z["n_frames"]), 6)

    def test_v2_collection_exports_and_meta_is_v2(self):
        write_episode(self.src, "train", 1, V2)
        r = self.run_export()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(str(np.load(self.out)["teacher_version"]), V2)

    def test_mixed_versions_are_refused(self):
        write_episode(self.src, "train", 1, V1)
        write_episode(self.src, "train", 2, V2)
        r = self.run_export()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("不一致", r.stderr)
        self.assertFalse(os.path.exists(self.out))

    def test_non_costmap_or_unknown_version_is_refused(self):
        write_episode(self.src, "train", 1, "uavrgb.teacher.v7", planner="astar")
        r = self.run_export()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("不是 costmap 集", r.stderr)

    def test_unknown_cost_version_is_refused(self):
        write_episode(self.src, "train", 1, "uavrgb.teacher_cost.v9")
        r = self.run_export()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("不是 costmap 集", r.stderr)


if __name__ == "__main__":
    unittest.main()
