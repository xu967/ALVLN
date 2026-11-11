#!/usr/bin/env python3
"""bearing-guard-3dgs/driver.py 为 teacher-clearance-3dgs 新增的两个臂上可选键（teacher_clearance、spawn_two_layer）的单测，
加一组「不给新键时输出逐位不变」的锁。

沿用 rule-oracle-3dgs/test_driver_rule.py 的 run()（它再沿用 sage3d-pilot/test_driver_params.py 的桩）：不碰 GPU、不起 sitl、不读 PLY；
桩的 OccAstarTeacher 记下构造参数（tdp.TEACHER_KW），桩的 run_episode 记下收到的关键字（REC["runs"][i]["kw"]）。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_driver_clearance.py
  DRIVER=<旧版 driver.py> python test_driver_clearance.py     拿改动前的版本跑：「新键」类用例应当红，「逐位不变」类应当绿
"""
import importlib.util
import json
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tdr", os.path.join(HERE, "..", "2026-11-10-rule-oracle-3dgs", "test_driver_rule.py"))
tdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tdr)            # 复用桩、run() 与 DRIVER 的取法（环境变量 DRIVER 优先）
tdp = tdr.tdp
REC = tdr.REC
OLD_RUN_KW = tdr.TestDefaultsLocked.OLD_RUN_KW
OLD_DONE_KEYS = tdr.TestDefaultsLocked.OLD_DONE_KEYS
DONE = tdr.DONE
TOL = 18.9


def arm(name, base="teacher", guard=None, **extra):
    return {"name": name, "base": base, "ckpt": None, "guard": guard, **extra}


def run(cfg, arms, **kw):
    tdp.TEACHER_KW.clear()
    return tdr.run({"routes": [0], "teacher_tol_deg": TOL, **cfg}, arms, **kw)


def done_of(snap, name, r=0):
    return json.loads(snap[DONE.format(arm=name, r=r)])


class TestDefaultsLocked(unittest.TestCase):
    """不给新键：对教师构造与 run_episode 的调用、done.json 的键与顺序，与没有本卡时一致。
    教师构造里的 hold_s 不是本卡的键：K3 起 driver 对所有教师臂都传这个臂的生效 hold（合并 K3 与本卡时由主会话对齐）。"""

    def test_teacher_arm_gets_exactly_the_old_constructor_arguments(self):
        rc, snap = run({}, [arm("t")])
        self.assertEqual(rc, 0)
        self.assertEqual(tdp.TEACHER_KW, [{"tol_deg": TOL, "hold_s": 1.0}])   # 不多传 clearance；hold_s 是 K3 对所有教师臂传的
        (run0,) = REC["runs"]
        self.assertEqual(run0["kw"], OLD_RUN_KW)                      # 不多传 spawn_two_layer
        self.assertEqual(list(done_of(snap, "t")), OLD_DONE_KEYS)

    def test_teacher_arm_without_tol_key_gets_only_hold_s(self):
        tdp.TEACHER_KW.clear()
        cfg = {"routes": [0]}
        tdr.run(cfg, [arm("t")])
        self.assertEqual(tdp.TEACHER_KW, [{"hold_s": 1.0}])

    def test_non_teacher_arm_is_unchanged(self):
        rc, snap = run({}, [arm("forward", base="forward")])
        self.assertEqual(rc, 0)
        self.assertEqual(REC["runs"][0]["kw"], OLD_RUN_KW)
        self.assertEqual(list(done_of(snap, "forward")), OLD_DONE_KEYS)
        self.assertEqual(tdp.TEACHER_KW, [])


class TestTeacherClearanceKey(unittest.TestCase):
    def test_swept_arm_passes_clearance_and_the_arm_hold(self):
        rc, snap = run({}, [arm("a", teacher_clearance="swept")])
        self.assertEqual(rc, 0)
        self.assertEqual(tdp.TEACHER_KW, [{"tol_deg": TOL, "hold_s": 1.0, "clearance": "swept"}])
        self.assertEqual(REC["runs"][0]["kw"], OLD_RUN_KW)            # 只给教师键，run_episode 的调用不变
        done = done_of(snap, "a")
        self.assertEqual(list(done), OLD_DONE_KEYS + ["teacher_clearance"])      # 新键追加在最后
        self.assertEqual(done["teacher_clearance"], "swept")

    def test_explicit_ray_arm(self):
        _, snap = run({}, [arm("r", teacher_clearance="ray")])
        self.assertEqual(tdp.TEACHER_KW, [{"tol_deg": TOL, "hold_s": 1.0, "clearance": "ray"}])
        self.assertEqual(done_of(snap, "r")["teacher_clearance"], "ray")

    def test_hold_s_is_the_arms_effective_hold(self):
        run({}, [arm("r", teacher_clearance="ray", hold=0.5, step_mult=2)])
        self.assertEqual(tdp.TEACHER_KW[0]["hold_s"], 0.5)
        run({"hold": 0.5}, [arm("r", teacher_clearance="ray")])
        self.assertEqual(tdp.TEACHER_KW[0]["hold_s"], 0.5)           # 全局 hold 也算生效值
        run({"hold": 0.5}, [arm("s", teacher_clearance="swept", hold=1.0)])
        self.assertEqual(tdp.TEACHER_KW[0]["hold_s"], 1.0)           # 臂上的 hold 盖过全局

    def test_one_teacher_per_route_and_arm(self):
        run({"routes": [0, 1]}, [arm("a", teacher_clearance="swept"), arm("b")])
        self.assertEqual([kw.get("clearance") for kw in tdp.TEACHER_KW], ["swept", "swept", None, None])

    def test_swept_with_other_hold_is_rejected_before_loading_the_scene(self):
        for cfg, a in (({"hold": 0.5}, arm("s", teacher_clearance="swept")), ({}, arm("s", teacher_clearance="swept", hold=0.5))):
            with self.assertRaises(ValueError):
                run(cfg, [a])
            self.assertEqual(tdp.LOADED, [])                          # 配置错在加载场景（约 200 s）之前就报

    def test_bad_value_or_non_teacher_arm_is_rejected_before_loading_the_scene(self):
        for a in (arm("x", teacher_clearance="sweep"), arm("x", teacher_clearance=None), arm("x", base="forward", teacher_clearance="ray")):
            with self.assertRaises(ValueError):
                run({}, [a])
            self.assertEqual(tdp.LOADED, [])


class TestSpawnTwoLayerKey(unittest.TestCase):
    def test_true_is_handed_to_run_episode_and_recorded(self):
        rc, snap = run({}, [arm("b", teacher_clearance="swept", spawn_two_layer=True)])
        self.assertEqual(rc, 0)
        self.assertEqual(REC["runs"][0]["kw"], {**OLD_RUN_KW, "spawn_two_layer": True})
        done = done_of(snap, "b")
        self.assertEqual(list(done), OLD_DONE_KEYS + ["teacher_clearance", "spawn_two_layer"])
        self.assertIs(done["spawn_two_layer"], True)

    def test_false_is_handed_over_too(self):
        _, snap = run({}, [arm("b", spawn_two_layer=False)])
        self.assertEqual(REC["runs"][0]["kw"], {**OLD_RUN_KW, "spawn_two_layer": False})
        self.assertEqual(list(done_of(snap, "b")), OLD_DONE_KEYS + ["spawn_two_layer"])
        self.assertIs(done_of(snap, "b")["spawn_two_layer"], False)

    def test_spawn_key_works_without_teacher_key_and_on_other_bases(self):
        run({}, [arm("f", base="forward", spawn_two_layer=True)])
        self.assertEqual(REC["runs"][0]["kw"]["spawn_two_layer"], True)
        self.assertEqual(tdp.TEACHER_KW, [])

    def test_two_arms_share_seeds_and_differ_only_in_the_switch(self):
        run({}, [arm("a", teacher_clearance="swept"), arm("b", teacher_clearance="swept", spawn_two_layer=True)])
        a, b = REC["runs"]
        self.assertEqual((a["kw"]["seed"], b["kw"]["seed"]), (8000, 8000))
        self.assertNotIn("spawn_two_layer", a["kw"])
        self.assertIs(b["kw"].get("spawn_two_layer"), True)
        self.assertEqual({k: v for k, v in b["kw"].items() if k != "spawn_two_layer"}, a["kw"])

    def test_bad_values_are_rejected_before_loading_the_scene(self):
        for cfg, a in (({}, arm("x", spawn_two_layer="yes")), ({}, arm("x", spawn_two_layer=1)),
                       ({"jitter": 0.0}, arm("x", spawn_two_layer=True))):
            with self.assertRaises(ValueError):
                run(cfg, [a])
            self.assertEqual(tdp.LOADED, [])

    def test_false_is_allowed_without_jitter(self):
        rc, _ = run({"jitter": 0.0}, [arm("x", spawn_two_layer=False)])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
