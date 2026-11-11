#!/usr/bin/env python3
"""driver.py 的全局可选 config 键 split_from_identity（卡 S1a / 审核 D1）的单测，加一组「不给键时输出逐位不变」的锁。

driver 以前把每集的 split 写死成 "test"、split_policy 写死成 "diag-eval-fixed-test"，场景按几何哈希是 train / val 也一样
（9 个评测场景里 8 个 train、1 个 val）。新键 split_from_identity = true：每条路线的身份算出来以后交给 eval_3dgs.eval_split(identity, True)，
拿它给的 (split, split_policy) 传给 run_episode：哈希 test 的场景仍是 ("test", "diag-eval-fixed-test")，train / val 的场景写真实 split
与 "diag-train-scene"。给了这个键（true 或 false），done.json 末尾多记 split 与 split_policy；没给就没有。eval_split 本身的单测在
sim/3dgs/test_eval_split.py，这里只测 driver 怎么用它。

沿用 rule-oracle-3dgs/test_driver_rule.py 的 run()（它再沿用 sage3d-pilot/test_driver_params.py 的桩）：不碰 GPU、不起 sitl、不读 PLY。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_driver_split.py
  DRIVER=<旧版 driver.py> python test_driver_split.py     拿改动前的版本跑：「新键」类用例应当红，「逐位不变」类应当绿
"""
import importlib.util
import json
import os
import sys
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

SPLIT_CALLS = []                         # 桩 eval_split 收到的 (identity, allow_train_scene)
IDENTITY = {}                            # 桩 scene_identity 返回的内容；空 = 用原来的桩（返回 {}）


def fake_eval_split(identity, allow_train_scene):
    """桩 eval_3dgs.eval_split：只复述约定（真函数的用例在 sim/3dgs/test_eval_split.py）。"""
    SPLIT_CALLS.append((dict(identity), allow_train_scene))
    if identity["split"] == "test":
        return "test", "diag-eval-fixed-test"
    return identity["split"], "diag-train-scene"


_orig_install = tdr.install


def _install(routes):
    names = _orig_install(routes)
    if IDENTITY:
        sys.modules["sim.3dgs.collect_3dgs"].scene_identity = lambda sc, ply_sha256=None: dict(IDENTITY)
        sys.modules["sim.3dgs.eval_3dgs"].eval_split = fake_eval_split
    return names


def setUpModule():
    tdr.install = _install


def tearDownModule():
    tdr.install = _orig_install


def arm(name="forward", base="forward", guard=None, **extra):
    return {"name": name, "base": base, "ckpt": None, "guard": guard, **extra}


def run(cfg, identity=None, arms=None, **kw):
    IDENTITY.clear()
    IDENTITY.update(identity or {})
    del SPLIT_CALLS[:]
    return tdr.run({"routes": [0], **cfg}, arms or [arm()], **kw)


def done_of(snap, r=0, name="forward"):
    return json.loads(snap[DONE.format(arm=name, r=r)])


class TestDefaultsLocked(unittest.TestCase):
    """不给 split_from_identity：对 run_episode 的调用、done.json 的键与顺序，与没有这个键时逐位一致；
    即使场景哈希是 train，缺省仍写 test（旧实验的配置要能逐位复现）。桩的 eval_3dgs 没有 eval_split，调到它就会炸。"""

    def test_default_with_the_stock_stubs(self):
        rc, snap = run({})
        self.assertEqual(rc, 0)
        (run0,) = REC["runs"]
        self.assertEqual(run0["kw"], OLD_RUN_KW)
        self.assertEqual(list(done_of(snap)), OLD_DONE_KEYS)

    def test_default_on_a_hash_train_scene_still_writes_test(self):
        rc, snap = run({}, identity={"split": "train", "geometry_gid": "g" * 16})
        self.assertEqual(rc, 0)
        kw = REC["runs"][0]["kw"]
        self.assertEqual((kw["split"], kw["split_policy"]), ("test", "diag-eval-fixed-test"))
        self.assertEqual({k: v for k, v in kw.items() if k != "identity"}, {k: v for k, v in OLD_RUN_KW.items() if k != "identity"})
        self.assertEqual(SPLIT_CALLS, [])
        self.assertEqual(list(done_of(snap)), OLD_DONE_KEYS)

    def test_other_bases_and_guards_are_unchanged(self):
        for a in (arm("fr", guard="route"), arm("t", base="teacher")):
            rc, snap = run({}, arms=[a])
            self.assertEqual(rc, 0)
            self.assertEqual(REC["runs"][0]["kw"], OLD_RUN_KW)
            self.assertEqual(list(done_of(snap, name=a["name"])), OLD_DONE_KEYS)


class TestSplitFromIdentityKey(unittest.TestCase):
    def test_true_on_a_train_scene_writes_the_real_split_and_diag_train_scene(self):
        ident = {"split": "train", "geometry_gid": "g" * 16}
        rc, snap = run({"split_from_identity": True}, identity=ident)
        self.assertEqual(rc, 0)
        kw = REC["runs"][0]["kw"]
        self.assertEqual((kw["split"], kw["split_policy"]), ("train", "diag-train-scene"))
        self.assertEqual(kw["identity"], ident)
        self.assertEqual({k: v for k, v in kw.items() if k not in ("split", "split_policy", "identity")},
                         {k: v for k, v in OLD_RUN_KW.items() if k not in ("split", "split_policy", "identity")})
        self.assertEqual(SPLIT_CALLS, [(ident, True)])           # 开了键就是明确要在这个场景上评测：allow_train_scene 恒为 True
        done = done_of(snap)
        self.assertEqual(list(done), OLD_DONE_KEYS + ["split", "split_policy"])      # 新键追加在最后
        self.assertEqual((done["split"], done["split_policy"]), ("train", "diag-train-scene"))

    def test_true_on_a_val_scene(self):
        _, snap = run({"split_from_identity": True}, identity={"split": "val", "geometry_gid": "g" * 16})
        self.assertEqual((REC["runs"][0]["kw"]["split"], REC["runs"][0]["kw"]["split_policy"]), ("val", "diag-train-scene"))
        self.assertEqual(done_of(snap)["split"], "val")

    def test_true_on_a_test_scene_keeps_the_old_pair(self):
        rc, snap = run({"split_from_identity": True}, identity={"split": "test", "geometry_gid": "g" * 16})
        self.assertEqual(rc, 0)
        kw = REC["runs"][0]["kw"]
        self.assertEqual((kw["split"], kw["split_policy"]), ("test", "diag-eval-fixed-test"))
        self.assertEqual(list(done_of(snap)), OLD_DONE_KEYS + ["split", "split_policy"])

    def test_false_is_the_old_behaviour_and_is_recorded(self):
        rc, snap = run({"split_from_identity": False}, identity={"split": "train", "geometry_gid": "g" * 16})
        self.assertEqual(rc, 0)
        kw = REC["runs"][0]["kw"]
        self.assertEqual((kw["split"], kw["split_policy"]), ("test", "diag-eval-fixed-test"))
        self.assertEqual(SPLIT_CALLS, [])
        done = done_of(snap)
        self.assertEqual(list(done), OLD_DONE_KEYS + ["split", "split_policy"])
        self.assertEqual((done["split"], done["split_policy"]), ("test", "diag-eval-fixed-test"))

    def test_every_route_asks_for_its_own_split(self):
        rc, snap = run({"split_from_identity": True, "routes": [0, 1]}, identity={"split": "train", "geometry_gid": "g" * 16})
        self.assertEqual(rc, 0)
        self.assertEqual(len(SPLIT_CALLS), 2)
        self.assertEqual([r["kw"]["split_policy"] for r in REC["runs"]], ["diag-train-scene"] * 2)
        self.assertEqual([done_of(snap, r)["split_policy"] for r in (0, 1)], ["diag-train-scene"] * 2)

    def test_no_split_test_anywhere_in_a_train_scenes_artifacts(self):
        """产物 = done.json，加传给 run_episode 的参数（真 run_episode 据此写 episode_manifest）。只断言「没有 test 标签」在旧代码上也成立
        （旧代码不认这个键，什么 split 都不记），所以每处再加一条正面断言：写的是真实的 train。"""
        _, snap = run({"split_from_identity": True}, identity={"split": "train", "geometry_gid": "g" * 16})

        def labelled_test(obj):
            if isinstance(obj, dict):
                return any((k in ("split", "split_policy") and v in ("test", "diag-eval-fixed-test")) or labelled_test(v)
                           for k, v in obj.items())
            return isinstance(obj, list) and any(labelled_test(v) for v in obj)

        checked = [rel for rel in snap if rel.endswith("done.json")]
        self.assertTrue(checked)
        for rel in checked:
            d = json.loads(snap[rel])
            self.assertFalse(labelled_test(d), rel)
            self.assertEqual(d.get("split"), "train", rel)
        self.assertTrue(REC["runs"])
        for r in REC["runs"]:
            self.assertFalse(labelled_test(r["kw"]), r["kw"])
            self.assertEqual(r["kw"]["split"], "train")

    def test_key_applies_to_every_base(self):
        for a in (arm("fr", guard="route"), arm("t", base="teacher")):
            run({"split_from_identity": True}, identity={"split": "train", "geometry_gid": "g" * 16}, arms=[a])
            self.assertEqual(REC["runs"][0]["kw"]["split_policy"], "diag-train-scene", a["name"])

    def test_bad_values_are_rejected_before_loading_the_scene(self):
        for bad in ("yes", 1, None, [True]):
            with self.assertRaises(ValueError):
                run({"split_from_identity": bad}, identity={"split": "train", "geometry_gid": "g" * 16})
            self.assertEqual(tdp.LOADED, [])                          # 配置错在加载场景（约 200 s）之前就报


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
