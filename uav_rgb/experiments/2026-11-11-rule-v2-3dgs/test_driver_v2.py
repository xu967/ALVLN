#!/usr/bin/env python3
"""bearing-guard-3dgs/driver.py 为 rule-v2-3dgs（任务卡 K3）新增的可选键 rule_version 的单测，加「不给新键时输出逐位不变」的锁。

新键（driver.py 文件头有说明）：臂 rule_version = "v1" | "v2"，只给规则层臂。
沿用 K1 的 test_driver_rule.py 的桩（把 driver 依赖的 sim.3dgs.* 换成桩模块，runpy 跑真实 driver.py），在其上把桩 RuleOraclePolicy 换成
会记下构造时全部关键字参数的子类。不碰 GPU、不起 sitl、不读 PLY。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_driver_v2.py
  DRIVER=<旧版 driver.py> python test_driver_v2.py      拿改动前的版本跑：「新键」类用例应当红，「逐位不变」类应当绿
"""
import importlib.util
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tdr", os.path.join(HERE, "..", "2026-11-10-rule-oracle-3dgs", "test_driver_rule.py"))
tdr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tdr)
tdp = tdr.tdp
run, arm, jload, jlines, REC, EP, DONE = tdr.run, tdr.arm, tdr.jload, tdr.jlines, tdr.REC, tdr.EP, tdr.DONE

K1_RULE_KEYS = ["source", "n_steps", "entries", "n_steps_pre", "entries_pre", "plan_failed", "params"]
K1_DONE_RULE_KEYS = ["source", "overrides"]
V2_PARAMS = {"levels": [[0.25, 1.2, False], [0.25, 0.8, False]], "order": "near"}


class SpyRule(tdr.FakeRule):
    """kw = 除 occ / target_fn / source / params 之外 driver 传来的关键字参数（旧臂应为空）。"""

    def __init__(self, occ, target_fn, source="occ", params=None, **kw):
        super().__init__(occ, target_fn, source=source, params=params)
        self.kw = kw


class Base(unittest.TestCase):
    def setUp(self):
        self._orig = tdr.FakeRule
        tdr.FakeRule = SpyRule

    def tearDown(self):
        tdr.FakeRule = self._orig


class TestDefaultsLocked(Base):
    """不给 rule_version：策略构造不带这个参数，metrics / done 的键与顺序同 K1。"""

    def test_k1_rule_arm_is_unchanged(self):
        rc, snap = run({"routes": [0], "rule_params": {"yaw_step_deg": 31.7}},
                       [arm("rocc", "rule-occ", "route", rule_params={"r_safe": 0.25, "d_look": 1.2})])
        self.assertEqual(rc, 0)
        (pol,) = REC["rules"]
        self.assertEqual(pol.kw, {}, "没给 rule_version 就不传，旧臂的构造调用与 K1 逐位相同")
        self.assertEqual(pol.params_arg, {"yaw_step_deg": 31.7, "r_safe": 0.25, "d_look": 1.2})
        m = jload(snap, EP.format(arm="rocc", r=0, f="metrics.json"))
        self.assertEqual(list(m["rule"]), K1_RULE_KEYS)
        done = jload(snap, DONE.format(arm="rocc", r=0))
        self.assertEqual(list(done["rule"]), K1_DONE_RULE_KEYS)
        self.assertNotIn("rule_version", m["rule"])

    def test_old_arm_stays_clean_next_to_a_v2_arm(self):
        rc, snap = run({"routes": [0]}, [arm("v1", "rule-occ", "route"), arm("v2", "rule-occ", "route", rule_version="v2")])
        self.assertEqual(rc, 0)
        self.assertEqual(list(jload(snap, EP.format(arm="v1", r=0, f="metrics.json"))["rule"]), K1_RULE_KEYS)
        self.assertEqual(list(jload(snap, DONE.format(arm="v1", r=0))["rule"]), K1_DONE_RULE_KEYS)
        self.assertEqual(REC["rules"][0].kw, {})
        self.assertEqual(REC["rules"][1].kw, {"rule_version": "v2"})


class TestRuleVersionKey(Base):
    def test_v2_reaches_the_policy_with_its_params(self):
        rc, snap = run({"routes": [1], "rule_params": {"yaw_step_deg": 31.7}},
                       [arm("rule2", "rule-occ", "route", rule_version="v2", rule_params=V2_PARAMS)])
        self.assertEqual(rc, 0)
        (pol,) = REC["rules"]
        self.assertEqual(pol.kw, {"rule_version": "v2"})
        self.assertEqual(pol.params_arg, {"yaw_step_deg": 31.7, **V2_PARAMS})
        self.assertEqual(pol.source, "occ")
        m = jload(snap, EP.format(arm="rule2", r=1, f="metrics.json"))
        self.assertEqual(list(m["rule"]), K1_RULE_KEYS + ["rule_version"], "新键只追加在后面")
        self.assertEqual(m["rule"]["rule_version"], "v2")
        done = jload(snap, DONE.format(arm="rule2", r=1))
        self.assertEqual(list(done["rule"]), K1_DONE_RULE_KEYS + ["rule_version"])
        self.assertEqual(done["rule"]["rule_version"], "v2")
        self.assertEqual(done["rule"]["overrides"], {"yaw_step_deg": 31.7, **V2_PARAMS})

    def test_depth_source_with_v2(self):
        rc, snap = run({"routes": [0]}, [arm("rd2", "rule-depth", "route", rule_version="v2")])
        self.assertEqual(rc, 0)
        (pol,) = REC["rules"]
        self.assertEqual((pol.source, pol.kw), ("depth", {"rule_version": "v2"}))

    def test_explicit_v1_is_passed_through(self):
        rc, snap = run({"routes": [0]}, [arm("v1x", "rule-occ", "goal", rule_version="v1")])
        self.assertEqual(rc, 0)
        self.assertEqual(REC["rules"][0].kw, {"rule_version": "v1"})
        self.assertEqual(jload(snap, EP.format(arm="v1x", r=0, f="metrics.json"))["rule"]["rule_version"], "v1")

    def test_rule_files_are_still_written_for_v2(self):
        rc, snap = run({"routes": [0]}, [arm("rule2", "rule-occ", "route", rule_version="v2")])
        rows = jlines(snap, EP.format(arm="rule2", r=0, f="rule.jsonl"))
        self.assertEqual(len(rows), 3)
        m = jload(snap, EP.format(arm="rule2", r=0, f="metrics.json"))
        self.assertEqual(m["rule"]["n_steps"], 3)
        self.assertIsNone(m["guard"])

    def test_bad_version_fails_before_loading_the_scene(self):
        with self.assertRaises(ValueError):
            run({"routes": [0]}, [arm("bad", "rule-occ", "route", rule_version="v3")])
        self.assertEqual(len(tdp.LOADED), 0, "配置错在加载场景之前就报")

    def test_version_on_a_non_rule_arm_fails_before_loading(self):
        for base in ("forward", "teacher"):
            with self.subTest(base=base), self.assertRaises(ValueError):
                run({"routes": [0]}, [arm("x", base, None, rule_version="v2")])
            self.assertEqual(len(tdp.LOADED), 0)

    def test_guard_still_required_for_rule_arms(self):
        with self.assertRaises(ValueError):
            run({"routes": [0]}, [arm("x", "rule-occ", None, rule_version="v2")])
        self.assertEqual(len(tdp.LOADED), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
