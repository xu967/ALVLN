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
PLANNER_ZS = []                          # 桩 OccPlanner 每次构造收到的 z 关键字参数
DEPTH_OP_CALLS = []                      # 桩 renderer.set_depth_opacity_min 每次收到的门限
ROUTE_ZS = {}                            # 可选覆盖每条路线的 spawn.z：{route_idx: z}
IDENTITY = {}                            # 可选覆盖桩 scene_identity 的返回字典
_orig_install = tdr.install


def _install(routes):
    names = _orig_install(routes)
    coll = sys.modules["sim.3dgs.collision"]
    coll.from_scene = lambda base, **kw: FROM_SCENE.append(dict(kw)) or _ns(collision_model=kw.get("collision_model", "v1"))
    coll.metric_version = lambda base, cm="v1": base + {"v1": "", "v2": "+coll.v2"}[cm]
    rend_mod = sys.modules["sim.3dgs.renderer"]
    rend_mod.make_renderer = lambda *a, **kw: (tdp.LOADED.append(1)
                                               or _ns(set_depth_opacity_min=lambda v: DEPTH_OP_CALLS.append(v)))
    pl_mod = sys.modules["sim.3dgs.planner"]
    pl_mod.OccPlanner = lambda occ, **kw: (PLANNER_ZS.append(kw.get("z"))
                                           or _ns(plan=(tdr.PLAN if "z" not in kw else (tdr.PLAN, kw["z"]))))
    if IDENTITY:
        sys.modules["sim.3dgs.collect_3dgs"].scene_identity = lambda sc, ply_sha256=None: dict(IDENTITY)
        sys.modules["sim.3dgs.eval_3dgs"].eval_split = (
            lambda ident, allow: ("test", "diag-eval-fixed-test") if ident["split"] == "test" else (ident["split"], "diag-train-scene")
        )
    if ROUTE_ZS:
        sc_mod = sys.modules["sim.3dgs.scene"]
        orig_apply = sc_mod.apply_route

        def _apply(sc, r):
            orig_apply(sc, r)
            if r in ROUTE_ZS:
                sc["spawn"]["z"] = ROUTE_ZS[r]

        sc_mod.apply_route = _apply
    return names


tdr.install = _install


def run(cfg, arms, route_zs=None, identity=None, **kw):
    FROM_SCENE.clear()
    PLANNER_ZS.clear()
    DEPTH_OP_CALLS.clear()
    ROUTE_ZS.clear()
    IDENTITY.clear()
    if route_zs:
        ROUTE_ZS.update(route_zs)
    if identity:
        IDENTITY.update(identity)
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

    def test_v2_caches_planner_by_each_routes_own_spawn_height(self):
        """J-9：driver 在 v2 下按每条路线自己的 sc['spawn']['z'] 构建并缓存 OccPlanner（同高度跨路线与跨臂复用，三维 occ 只建一次）；
        v1 不给 collision_model=v2 时仍无参数调一次 OccPlanner(occ)。"""
        rc, _ = run(
            {"collision_model": "v2", "routes": [0, 1, 2]},
            [arm("t", base="teacher"), arm("f", base="forward", guard="route")],
            route_zs={0: 1.2, 1: 1.8, 2: 1.2},
        )
        self.assertEqual(rc, 0)
        self.assertEqual(len(FROM_SCENE), 1, "三维占据只建一次")
        self.assertEqual(PLANNER_ZS, [1.2, 1.8], "两个不同路线高度各建一次规划器，r2 与第二臂复用缓存")
        # forward+route 的 OracleRouteGuide 在 r0/r1/r2 拿到的 plan 分别对应各自高度 1.2 / 1.8 / 1.2
        self.assertEqual([g.plan[1] for g in REC["guides"]], [1.2, 1.8, 1.2])

        # 对照：v1 下仍只无关键字调一次 OccPlanner(occ)
        rc1, _ = run({"routes": [0, 1]}, [arm("f")], route_zs={0: 1.2, 1: 1.8})
        self.assertEqual(rc1, 0)
        self.assertEqual(PLANNER_ZS, [None])

    def test_split_from_identity_and_collision_model_combined_done_key_order(self):
        """J-4：split_from_identity 与 collision_model 两键同开时（重定基线正式运行配置），done.json 末尾键顺序锁为：
        split、split_policy 在前（S1a），collision_model、metric_version 在后（S1b）。"""
        rc, snap = run(
            {"split_from_identity": True, "collision_model": "v2"},
            [arm("t", base="teacher", teacher_clearance="swept"), arm("r", base="rule-occ", guard="route", rule_version="v2")],
            identity={"split": "train", "geometry_gid": "a" * 16},
        )
        self.assertEqual(rc, 0)
        dt = done_of(snap, "t")
        dr = done_of(snap, "r")
        self.assertEqual(list(dt), OLD.OLD_DONE_KEYS + ["teacher_clearance", "split", "split_policy", "collision_model", "metric_version"])
        self.assertEqual((dt["split"], dt["split_policy"], dt["collision_model"], dt["metric_version"]),
                         ("train", "diag-train-scene", "v2", METRIC_BASE + "+coll.v2"))
        self.assertEqual(list(dr), OLD.OLD_DONE_KEYS + ["rule", "split", "split_policy", "collision_model", "metric_version"])

    def test_depth_opacity_min_per_arm_resets_for_other_arms_and_records_key(self):
        """B5：driver 支持臂级 depth_opacity_min（给深度规则臂用）：切到带该键的臂时调 set_depth_opacity_min(0.30)，
        下一个不带该键的臂自动调回 None，不留状态；没有臂用这个键时从不调用 set_depth_opacity_min（旧单参桩兼容）。"""
        rc, snap = run({}, [arm("f")])
        self.assertEqual((rc, DEPTH_OP_CALLS), (0, []), "没有任何臂设 depth_opacity_min 时不调用")

        rc, snap = run(
            {"collision_model": "v2"},
            [arm("r0", base="rule-depth", guard="route", rule_version="v2"),
             arm("r_op", base="rule-depth", guard="route", rule_version="v2", depth_opacity_min=0.30),
             arm("f", base="forward")],
        )
        self.assertEqual(rc, 0)
        self.assertEqual(DEPTH_OP_CALLS, [None, 0.30, None])
        self.assertNotIn("depth_opacity_min", done_of(snap, "r0"))
        self.assertNotIn("depth_opacity_min", metrics_of(snap, "r0"))
        self.assertEqual(done_of(snap, "r_op")["depth_opacity_min"], 0.30)
        self.assertEqual(metrics_of(snap, "r_op")["depth_opacity_min"], 0.30)
        self.assertNotIn("depth_opacity_min", done_of(snap, "f"))

        for bad in (-0.01, 1.5, "0.3", True, None):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                run({}, [arm("r_bad", base="rule-depth", guard="route", depth_opacity_min=bad)])
            self.assertEqual(tdp.LOADED, [])


if __name__ == "__main__":
    unittest.main(verbosity=1, warnings="ignore")
