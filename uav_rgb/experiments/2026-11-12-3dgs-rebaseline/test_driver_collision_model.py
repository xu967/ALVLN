#!/usr/bin/env python3
"""bearing-guard-3dgs/driver.py 为 3dgs-rebaseline（任务卡 S1b）新增的全局配置键 collision_model 的单测，加一组「不给键时输出逐位不变」的锁。

沿用 rule-oracle-3dgs/test_driver_rule.py 的 run() 与桩（它再沿用 sage3d-pilot/test_driver_params.py）：不碰 GPU、不起 sitl、不读 PLY，
用 runpy 跑真实的 driver.py。这里把桩的 from_scene 换成会记录关键字的版本（旧桩只收一个位置参数），
并给桩 collision 模块补 metric_version（与 sim/3dgs/collision.py 同义，真函数的字串由 test_collision_model.py 锁定）。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_driver_collision_model.py
  DRIVER=<旧版 driver.py> python test_driver_collision_model.py    拿改动前的版本跑：「键」类用例应当红，「逐位不变」类应当绿
"""
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tdr", os.path.join(HERE, "..", "2026-11-10-rule-oracle-3dgs", "test_driver_rule.py"))
tdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tdr)            # 复用桩、run() 与 DRIVER 的取法（环境变量 DRIVER 优先）
tdp, REC, _ns = tdr.tdp, tdr.REC, tdr._ns
OLD = tdr.TestDefaultsLocked
METRIC_BASE = "3dgs.pilot.v2+stop.v1"

FROM_SCENE = []                          # 桩 from_scene 每次收到的关键字参数
_orig_install = tdr.install


def _install(routes):
    names = _orig_install(routes)
    coll = sys.modules["sim.3dgs.collision"]
    coll.from_scene = lambda base, **kw: FROM_SCENE.append(dict(kw)) or _ns(collision_model=kw.get("collision_model", "v1"))
    coll.metric_version = lambda base, cm="v1": base + {"v1": "", "v2": "+coll.v2"}[cm]
    return names


tdr.install = _install


def run(cfg, arms, **kw):
    FROM_SCENE.clear()
    return tdr.run({"routes": [0], **cfg}, arms, **kw)


def done_of(snap, name, r=0):
    return tdr.jload(snap, tdr.DONE.format(arm=name, r=r))


def metrics_of(snap, name, r=0):
    return tdr.jload(snap, tdr.EP.format(arm=name, r=r, f="metrics.json"))


def arm(name, base="forward", guard=None, **extra):
    return tdr.arm(name, base, guard, **extra)


class TestDefaultsLocked(unittest.TestCase):
    """不给 collision_model：from_scene 原样调用（不多传关键字），run_episode 的调用、done.json 与 metrics.json 的键与顺序都与改动前一致。"""

    def test_no_key_changes_nothing(self):
        rc, snap = run({}, [arm("f")])
        self.assertEqual(rc, 0)
        self.assertEqual(FROM_SCENE, [{}])
        self.assertEqual(list(done_of(snap, "f")), OLD.OLD_DONE_KEYS)
        self.assertEqual(list(metrics_of(snap, "f")), OLD.OLD_METRIC_KEYS)
        self.assertEqual(REC["runs"][0]["kw"], OLD.OLD_RUN_KW)

    def test_no_key_on_teacher_and_rule_arms(self):
        rc, snap = run({}, [arm("t", base="teacher"), arm("r", base="rule-occ", guard="route")])
        self.assertEqual(rc, 0)
        self.assertEqual(FROM_SCENE, [{}])
        self.assertEqual(list(done_of(snap, "t")), OLD.OLD_DONE_KEYS)
        self.assertNotIn("collision_model", done_of(snap, "r"))
        self.assertNotIn("collision_model", metrics_of(snap, "r"))
        self.assertNotIn("metric_version", metrics_of(snap, "r"))


class TestCollisionModelKey(unittest.TestCase):
    def test_v2_reaches_from_scene_and_is_recorded_last(self):
        rc, snap = run({"collision_model": "v2"}, [arm("f")])
        self.assertEqual(rc, 0)
        self.assertEqual(FROM_SCENE, [{"collision_model": "v2"}])
        done, m = done_of(snap, "f"), metrics_of(snap, "f")
        self.assertEqual(list(done), OLD.OLD_DONE_KEYS + ["collision_model", "metric_version"])      # 新键追加在最后
        self.assertEqual(list(m), OLD.OLD_METRIC_KEYS + ["collision_model", "metric_version"])
        self.assertEqual((done["collision_model"], m["collision_model"]), ("v2", "v2"))
        self.assertEqual((done["metric_version"], m["metric_version"]), (METRIC_BASE + "+coll.v2",) * 2)

    def test_explicit_v1_is_recorded_with_the_unsuffixed_version(self):
        rc, snap = run({"collision_model": "v1"}, [arm("f")])
        self.assertEqual(rc, 0)
        self.assertEqual(FROM_SCENE, [{"collision_model": "v1"}])
        self.assertEqual((done_of(snap, "f")["collision_model"], done_of(snap, "f")["metric_version"]), ("v1", METRIC_BASE))
        self.assertEqual((metrics_of(snap, "f")["collision_model"], metrics_of(snap, "f")["metric_version"]), ("v1", METRIC_BASE))

    def test_run_episode_call_is_unchanged_by_the_key(self):
        run({"collision_model": "v2"}, [arm("f")])
        self.assertEqual(REC["runs"][0]["kw"], OLD.OLD_RUN_KW)            # 口径只经 occ 到 run_episode，不多传参数

    def test_key_order_after_other_new_keys_and_rule_summary(self):
        _, snap = run({"collision_model": "v2"}, [arm("t", base="teacher", teacher_clearance="swept", spawn_two_layer=False),
                                                  arm("r", base="rule-occ", guard="route")])
        self.assertEqual(list(done_of(snap, "t")),
                         OLD.OLD_DONE_KEYS + ["teacher_clearance", "spawn_two_layer", "collision_model", "metric_version"])
        self.assertEqual(list(metrics_of(snap, "r"))[-3:], ["rule", "collision_model", "metric_version"])
        self.assertEqual(list(done_of(snap, "r"))[-3:], ["rule", "collision_model", "metric_version"])

    def test_bad_values_are_rejected_before_loading_the_scene(self):
        for bad in ("v3", "V2", None, 2, ""):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    run({"collision_model": bad}, [arm("f")])
                self.assertEqual(tdp.LOADED, [])

    def test_v2_rejects_spawn_two_layer_true_before_loading_the_scene(self):
        with self.assertRaises(ValueError):
            run({"collision_model": "v2"}, [arm("t", base="teacher", spawn_two_layer=True)])
        self.assertEqual(tdp.LOADED, [])

    def test_v2_with_spawn_two_layer_false_and_v1_with_true_still_run(self):
        rc, _ = run({"collision_model": "v2"}, [arm("t", base="teacher", spawn_two_layer=False)])
        self.assertEqual(rc, 0)
        self.assertEqual(REC["runs"][0]["kw"]["spawn_two_layer"], False)
        rc, _ = run({"collision_model": "v1"}, [arm("t", base="teacher", spawn_two_layer=True)])
        self.assertEqual(rc, 0)
        self.assertEqual(REC["runs"][0]["kw"]["spawn_two_layer"], True)


if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")
