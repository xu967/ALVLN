#!/usr/bin/env python3
"""cf_sitl_exec 的离线单测（不需要 Gazebo）：dynamics_profile = cf_sitl_v1:<参数 value 规范化 sha8>:lib<N>。

AUDIT_SIM 2026-11-09 #1：profile 以前只是 e1_mini.json 原始字节的 sha8，改注释就换 profile、换库不换 profile。
现在参数 sha 只算各项的 value，库版本从插件 state[IDX_LIB_VERSION] 读。运行：
  .venv-tools/bin/python -m unittest sim.tools.test_cf_sitl_exec
"""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cf_sitl_exec as ce  # noqa: E402

PROFILE_RE = re.compile(r"^cf_sitl_v1:[0-9a-f]{8}:lib\d+$")


def _params(obj, lib_version=None):
    raw = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    return ce.CfSitlParams("x.json", raw, obj, lib_version=lib_version)


BASE = {"calibrated": False,
        "mass_flight": {"value": 0.102, "unit": "kg", "basis": "a"},
        "executor": {"value": {"vel_kp": 0.22, "vel_ki": 0.30}, "basis": "b"}}


class TestProfileFingerprint(unittest.TestCase):
    def test_comment_change_keeps_profile_value_change_breaks_it(self):
        p0 = _params(BASE, 31)
        self.assertRegex(p0.profile(), PROFILE_RE)
        self.assertTrue(p0.profile().endswith(":lib31"))
        commented = json.loads(json.dumps(BASE))
        commented["executor"]["basis"] = "改了说明文字"
        commented["mass_flight"]["unit"] = "千克"
        p1 = _params(commented, 31)
        self.assertNotEqual(p1.sha256, p0.sha256)             # 文件字节变了
        self.assertEqual(p1.profile(), p0.profile())           # 只算 value：profile 不变
        tuned = json.loads(json.dumps(BASE))
        tuned["executor"]["value"]["vel_ki"] = 0.15
        self.assertNotEqual(_params(tuned, 31).profile(), p0.profile())
        self.assertNotEqual(_params(BASE, 32).profile(), p0.profile())     # 库版本变 → 变
        # 旧格式 cf_sitl_v1:<原始字节 sha8> 与新格式不可能相等（长度就不同）
        self.assertNotEqual(p0.profile(), f"{ce.PROFILE_PREFIX}:{p0.sha256[:8]}")

    def test_unknown_lib_version_is_visible_not_silent(self):
        p = _params(BASE)
        self.assertIsNone(p.lib_version)
        self.assertEqual(p.profile(), f"{ce.PROFILE_PREFIX}:{p.values_sha256[:8]}:lib?")   # start() 前只供日志
        self.assertIsNone(p.manifest()["lib_version"])
        p.lib_version = 31
        self.assertRegex(p.profile(), PROFILE_RE)
        self.assertEqual(p.manifest()["lib_version"], 31)

    def test_values_sha_is_canonical(self):
        a = {"b": {"value": [1, 2.5, {"y": 1, "x": 2}]}, "a": {"value": "s"}, "calibrated": True}
        b = {"a": {"value": "s", "basis": "zzz"}, "b": {"value": [1, 2.5, {"x": 2, "y": 1}]}}
        self.assertEqual(ce.params_values_sha256(a), ce.params_values_sha256(b))   # 键序、注释、非 value 顶层项都不算
        self.assertEqual(len(ce.params_values_sha256(a)), 64)

    def test_real_params_file(self):
        p = ce.CfSitlParams.from_json(ce.DEFAULT_PARAMS)
        p.lib_version = 31
        self.assertRegex(p.profile(), PROFILE_RE)
        self.assertEqual(p.profile(), ce.format_profile(p.values_sha256, 31))

    def test_state_layout_has_lib_version_after_sticks(self):
        self.assertEqual(ce.IDX_LIB_VERSION, ce.IDX_STICKS + 4)
        self.assertEqual(ce.STATE_FIELDS, ce.IDX_LIB_VERSION + 1)
        ex = ce.CfSitlExecutor(_params(BASE), hold_s=0.5)

        class Sub:
            def __init__(self, n):
                self.n = n

            def latest(self):
                return {"data": [0.0] * self.n}

            def healthy(self):
                return True

        ex._sub = Sub(ce.STATE_FIELDS)
        self.assertEqual(len(ex.state()), ce.STATE_FIELDS)
        ex._sub = Sub(ce.STATE_FIELDS - 1)                     # 旧插件（没有 lib_version）：明确报错，不是静默等超时
        self.assertIsNone(ex.state())
        self.assertIn("重建", ex.error())


if __name__ == "__main__":
    unittest.main()
