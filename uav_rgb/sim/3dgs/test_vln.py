#!/usr/bin/env python3
"""vln.py（VLN 集与指标）、run_episode 的智能体喊停、录像指令横幅的小测（mock 渲染器，纯 CPU）。

  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_vln.py
"""
import importlib
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
vln = importlib.import_module("sim.3dgs.vln")
collect = importlib.import_module("sim.3dgs.collect_3dgs")
scene_mod = importlib.import_module("sim.3dgs.scene")

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

SCENE = {"name": "s", "routes": [
    {"name": "r0", "sage": {"split": "train", "instructions": [
        {"instruction_type": "Attribute-based", "start": "sofa_1", "end": "lamp_2", "text": "Go to the tall lamp."},
        {"instruction_type": "Area-based", "start": "sofa_1", "end": "lamp_2", "text": "Head to the reading corner."}]}},
    {"name": "r1", "sage": {"split": "test_Instruction_Unseen", "instructions": [
        {"instruction_type": "Attribute-based", "start": "bed_1", "end": "tv_1", "text": "Find the black TV."}]}},
    {"name": "old"}]}     # 不是 SAGE 导入的航线：没有指令，不出集
GOAL = {"x": 4.0, "y": 0.0}


def _steps(xy, collide_at=None):
    """xy[0] 是出发点，之后每步一个终点；collide_at 起每步都标撞（run_episode 撞后整段标撞的语义）。"""
    actions = [{"pose_before": {"x": x, "y": y}, "collision": collide_at is not None and i >= collide_at}
               for i, (x, y) in enumerate(xy[:-1])]
    oracle = [{"pose_gt": {"x": x, "y": y}} for x, y in xy[1:]]
    return actions, oracle


class TestEpisodes(unittest.TestCase):
    def test_every_instruction_is_an_episode(self):
        eps = vln.episodes(SCENE)
        self.assertEqual([e["key"] for e in eps], ["s/r0/0", "s/r0/1", "s/r1/0"])
        self.assertEqual((eps[2]["route_index"], eps[2]["goal_object"], eps[2]["split"]),
                         (1, "tv_1", "test_Instruction_Unseen"))

    def test_filters(self):
        self.assertEqual([e["key"] for e in vln.episodes(SCENE, splits=["test_Instruction_Unseen"])], ["s/r1/0"])
        self.assertEqual([e["key"] for e in vln.episodes(SCENE, types=["Area-based"])], ["s/r0/1"])

    def test_one_per_route_rotates_by_type_name(self):
        # r0 按句式名排序是 [Area-based(1), Attribute-based(0)]，与指令在航线里的原顺序无关
        eps = vln.episodes(SCENE)
        self.assertEqual([e["key"] for e in vln.one_per_route(eps, seed=0)], ["s/r0/1", "s/r1/0"])
        self.assertEqual([e["key"] for e in vln.one_per_route(eps, seed=1)], ["s/r0/0", "s/r1/0"])


class TestMetrics(unittest.TestCase):
    def test_stop_inside_radius_is_success(self):
        a, o = _steps([(0, 0), (1, 0), (2, 0), (3.5, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True, shortest_m=3.0)
        self.assertTrue(m["success"] and m["oracle_success"])
        self.assertAlmostEqual(m["ne_m"], 0.5)
        self.assertAlmostEqual(m["path_m"], 3.5)
        self.assertAlmostEqual(m["spl"], 3.0 / 3.5)

    def test_flying_through_without_stop_is_only_oracle_success(self):
        # 旧 3dgs.pilot.v2 判据会算成功；VLN 要求自己停在半径里
        a, o = _steps([(0, 0), (2, 0), (4, 0), (6, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=False, shortest_m=3.0)
        self.assertEqual((m["success"], m["oracle_success"], m["spl"]), (False, True, 0.0))
        self.assertAlmostEqual(m["ne_m"], 2.0)

    def test_stop_short_of_goal_fails(self):
        a, o = _steps([(0, 0), (1, 0)])
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True)
        self.assertEqual((m["success"], m["oracle_success"], m["spl"]), (False, False, None))

    def test_poses_after_collision_do_not_count(self):
        # 第 1 步（0 起）撞：它的终点 (4, 0) 已在障碍里，不算进过半径
        a, o = _steps([(0, 0), (2.5, 0), (4, 0), (4, 0)], collide_at=1)
        m = vln.episode_metrics(a, o, GOAL, 1.0, stopped=True)
        self.assertEqual((m["success"], m["oracle_success"], m["collision"]), (False, False, True))

    def test_summarize(self):
        rows = [{"success": True, "oracle_success": True, "spl": 0.8, "ne_m": 0.5, "collision": False, "stopped": True, "type": "A"},
                {"success": False, "oracle_success": True, "spl": 0.0, "ne_m": 2.5, "collision": True, "stopped": False, "type": "B"},
                {"success": False, "oracle_success": False, "spl": None, "ne_m": 4.0, "collision": False, "stopped": True, "type": "B"}]
        s = vln.summarize(rows)
        self.assertEqual((s["n"], s["n_spl"], s["ne_median_m"]), (3, 2, 2.5))
        for key, want in (("sr", 1 / 3), ("osr", 2 / 3), ("spl", 0.4), ("ne_mean_m", 7 / 3),
                          ("collision_rate", 1 / 3), ("stop_rate", 2 / 3)):
            self.assertAlmostEqual(s[key], want, msg=key)
        self.assertEqual(vln.summarize_by(rows, "type")["B"]["n"], 2)
        self.assertEqual(vln.summarize([]), {"n": 0})


class Scripted:
    """按顺序出动作的策略；序列用完就重复最后一个。"""
    name = "forward"

    def __init__(self, seq):
        self.seq, self.i = list(seq), 0

    def __call__(self, rgb, depth, state):
        a = self.seq[min(self.i, len(self.seq) - 1)]
        self.i += 1
        return a, None


class TestAgentStop(unittest.TestCase):
    def test_safe_streak_ends_episode_only_when_enabled(self):
        F, S = collect.FORWARD_SLOW, collect.SAFE_NEUTRAL
        seq = [F, S, S, F, S, S, S, F]
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
            renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(scene, True)
            occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)
            params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
            identity = collect.scene_identity(scene)
            scene["vln"] = {"text": "Go to the end of the corridor."}
            m, err = collect.run_episode(scene, os.path.join(tmp, "stop"), renderer, occ, params, Scripted(seq), 10, 0.5,
                                         split=identity["split"], identity=identity, stop_after_safe=vln.STOP_STREAK)
            self.assertIsNone(err)
            self.assertEqual((m["num_steps"], m["stopped_by_agent"], m["completed"]), (7, True, True))
            self.assertEqual(m["vln"], {"text": "Go to the end of the corridor."})
            m, _ = collect.run_episode(scene, os.path.join(tmp, "old"), renderer, occ, params, Scripted(seq), 10, 0.5,
                                       split=identity["split"], identity=identity)
            self.assertEqual((m["num_steps"], m["stopped_by_agent"], m["stop_after_safe"]), (10, False, None))


class TestBanner(unittest.TestCase):
    def test_wraps_and_keeps_at_most_four_lines(self):
        import cv2
        video = importlib.import_module("sim.3dgs.video")
        canvas = np.full((360, 640, 3), 200, np.uint8)
        video._banner(cv2, canvas, "word " * 80, 0, 0, 640)
        band = canvas[32:32 + 90]                       # 4 行 * 20 px + 10 px 边距
        self.assertEqual(band.max(), 255)               # 白字
        self.assertLess(np.median(band), 100)           # 黑底
        self.assertTrue((canvas[:32] == 200).all() and (canvas[32 + 90:] == 200).all())


if __name__ == "__main__":
    unittest.main()
