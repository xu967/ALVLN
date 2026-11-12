#!/usr/bin/env python3
"""route_dirs（任务卡 S1b，审核 E-12）的单测：纯标准库。

背景：3DGS 实验驱动（experiments/2026-11-10-bearing-guard-3dgs/driver.py）续跑时把半成品路线目录改名成 r<N>.incomplete-<时刻> 再重跑，
汇总脚本里的通配 r*/ep_*/metrics.json 会把半成品一起读进来（审核 E-12 在 verify_small.txt 里复现：一次通配同时命中 r0/ 与 r0.incomplete-231200/）。
route_dirs / episode_files 只收名字恰为 r<数字> 的目录。

用 3dgs-venv 或 .venv-tools 跑：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_route_dirs.py
"""
import glob
import importlib
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
rd = importlib.import_module("sim.3dgs.route_dirs")

_DATA = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb")
if "TMPDIR" not in os.environ and os.path.isdir(os.path.join(_DATA, "tmp")):
    tempfile.tempdir = os.path.join(_DATA, "tmp")


def touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").close()


class TestRouteDirs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        # <臂>/<场景>/r<N>/ep_*/metrics.json，其中 r3 有一个续跑留下的半成品
        for r in ("r0", "r3", "r10", "r3.incomplete-020101", "r10.incomplete-231200", "rX", "r", "r3x", "r3.bak"):
            for ep in ("ep_00000", "ep_00001"):
                touch(os.path.join(self.root, "armA", "scene1", r, ep, "metrics.json"))
                touch(os.path.join(self.root, "armA", "scene1", r, ep, "actions.jsonl"))
        touch(os.path.join(self.root, "armB", "scene1", "r0", "ep_00000", "metrics.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def names(self, paths):
        return sorted(os.path.relpath(p, self.root) for p in paths)

    def test_old_glob_picks_up_incomplete_dirs(self):
        """旧通配的缺陷（锁定）：r* 同时命中 r3 与 r3.incomplete-020101。"""
        old = glob.glob(os.path.join(self.root, "*", "*", "r*", "ep_*", "metrics.json"))
        self.assertTrue(any(".incomplete-" in p for p in old))

    def test_episode_files_excludes_incomplete_and_foreign_names(self):
        got = self.names(rd.episode_files(self.root, "metrics.json", depth=2))
        want = sorted(f"{arm}/scene1/{r}/{ep}/metrics.json"
                      for arm, rs in (("armA", ("r0", "r3", "r10")), ("armB", ("r0",)))
                      for r in rs for ep in (("ep_00000", "ep_00001") if arm == "armA" else ("ep_00000",)))
        self.assertEqual(got, want)

    def test_route_dirs_only_complete_names(self):
        got = [os.path.relpath(p, self.root) for p in rd.route_dirs(self.root, depth=2)]
        self.assertEqual(got, ["armA/scene1/r0", "armA/scene1/r10", "armA/scene1/r3", "armB/scene1/r0"])

    def test_equals_old_glob_when_no_incomplete(self):
        """没有半成品时与旧通配逐项相同（新函数不丢任何完整的集）。"""
        with tempfile.TemporaryDirectory() as t:
            for arm in ("a", "b"):
                for r in ("r0", "r1", "r12"):
                    for ep in ("ep_00000", "ep_00001", "ep_00002"):
                        touch(os.path.join(t, arm, "s", r, ep, "metrics.json"))
            old = sorted(glob.glob(os.path.join(t, "*", "*", "r*", "ep_*", "metrics.json")))
            self.assertEqual(rd.episode_files(t, "metrics.json", depth=2), old)

    def test_depth_levels(self):
        with tempfile.TemporaryDirectory() as t:
            touch(os.path.join(t, "r1", "ep_00000", "actions.jsonl"))
            touch(os.path.join(t, "r1.incomplete-1", "ep_00000", "actions.jsonl"))
            touch(os.path.join(t, "sceneA", "r2", "ep_00000", "actions.jsonl"))
            touch(os.path.join(t, "sceneA", "r2.incomplete-1", "ep_00000", "actions.jsonl"))
            self.assertEqual([os.path.relpath(p, t) for p in rd.episode_files(t, "actions.jsonl", depth=0)],
                             ["r1/ep_00000/actions.jsonl"])
            self.assertEqual([os.path.relpath(p, t) for p in rd.episode_files(t, "actions.jsonl", depth=1)],
                             ["sceneA/r2/ep_00000/actions.jsonl"])

    def test_missing_root_is_empty(self):
        self.assertEqual(rd.episode_files(os.path.join(self.root, "nope"), "metrics.json"), [])
        self.assertEqual(rd.route_dirs(os.path.join(self.root, "nope")), [])

    def test_file_named_like_route_is_not_a_dir(self):
        with tempfile.TemporaryDirectory() as t:
            touch(os.path.join(t, "a", "s", "r5"))      # 同名的文件
            self.assertEqual(rd.route_dirs(t, depth=2), [])

    def test_real_experiment_output_unchanged_when_no_incomplete(self):
        """真实输出（teacher-tol 室内臂，没有半成品目录）：新函数与旧通配逐项相同。"""
        root = os.path.join(_DATA, "exp-teacher-tol-3dgs-20261110", "indoor")
        if not os.path.isdir(root):
            self.skipTest(f"{root} 不存在")
        old = sorted(glob.glob(os.path.join(root, "*", "*", "r*", "ep_*", "metrics.json")))
        self.assertEqual([p for p in old if ".incomplete-" in p], [], "前提：这个目录里没有半成品")
        self.assertEqual(rd.episode_files(root, "metrics.json", depth=2), old)
        self.assertGreater(len(old), 100)


if __name__ == "__main__":
    unittest.main()
