#!/usr/bin/env python3
"""--rtf（W1-0：世界物理的目标实时因子）的测试。不起 Gazebo。

覆盖：
  scene_gen --rtf：只改世界里 <real_time_factor> 那一行（x3 / e1_mini 两个模板），meta 只在给了 --rtf 时多
    real_time_factor 一个键、world_sha256 随世界更新，默认输出（世界文本与 meta）不变；非法值走我们自己的校验。
  collect_parallel.scene_gen_extra：给了才透传，没给不传（旧命令行逐字不变）。
  eval_closed_loop：--rtf 透传给 scene_gen、eval.json 顶层记 rtf（没给没有这个键）、非法值拒绝。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_scene_rtf.py
"""
import contextlib
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_parallel as cp
import scene_gen
import test_action_latency as tal   # 以模块名引用，借 EvalHarness / eval_rows


def gen_world(extra, seed=1, vehicle=None):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "w.sdf")
        argv = ["scene_gen.py", "--seed", str(seed), "--out", out] + (["--vehicle", vehicle] if vehicle else []) + extra
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            rc = scene_gen.main()
        assert rc == 0
        return Path(out).read_text(), json.loads(Path(out).with_suffix(".json").read_text())


class SceneGenRtf(unittest.TestCase):
    def test_rtf_changes_only_the_real_time_factor_line(self):
        for vehicle in (None, "e1_mini"):
            base, meta0 = gen_world([], vehicle=vehicle)
            world, meta = gen_world(["--rtf", "0.5"], vehicle=vehicle)
            self.assertEqual(base.count("<real_time_factor>"), 1, vehicle)
            self.assertEqual(world.count("<real_time_factor>0.5</real_time_factor>"), 1, vehicle)
            old = base[base.index("<real_time_factor>"):base.index("</real_time_factor>") + len("</real_time_factor>")]
            self.assertEqual(world, base.replace(old, "<real_time_factor>0.5</real_time_factor>"), "只改这一行")
            self.assertEqual(meta["real_time_factor"], 0.5)
            self.assertEqual(meta["world_sha256"], hashlib.sha256(world.encode("utf-8")).hexdigest())
            rest = {k: v for k, v in meta.items() if k not in ("real_time_factor", "world_sha256")}
            rest0 = {k: v for k, v in meta0.items() if k != "world_sha256"}
            self.assertEqual(rest, rest0, "其余 meta 逐项不变（几何、gid、split 不随 rtf 变）")

    def test_default_output_has_no_rtf_trace(self):
        for vehicle in (None, "e1_mini"):
            world, meta = gen_world([], vehicle=vehicle)
            self.assertNotIn("real_time_factor", meta)
            self.assertEqual(world.count("<real_time_factor>1.0</real_time_factor>"), 1, vehicle)
            self.assertEqual(meta["world_sha256"], hashlib.sha256(world.encode("utf-8")).hexdigest())

    def test_invalid_rtf_rejected(self):
        for bad in ("0", "-1", "nan", "inf"):
            err = io.StringIO()
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err), \
                    patch.object(sys, "argv", ["scene_gen.py", "--seed", "1", "--out", os.path.join(d, "w.sdf"),
                                               "--rtf", bad]):
                with self.assertRaises(SystemExit) as cm:
                    scene_gen.main()
                self.assertEqual(cm.exception.code, 2, bad)
            self.assertIn("--rtf 必须是有限正数", err.getvalue(), bad)   # 不是「不认识的参数」那种 2

    def test_scene_gen_extra_passes_rtf_only_when_given(self):
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(rtf=0.5)), ["--rtf", "0.5"])
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(rtf=None)), [])
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace()), [])


class EvalRtf(unittest.TestCase):
    def test_default_has_no_rtf_flag_or_key(self):
        report, cmds = tal.EvalHarness.run([], tal.eval_rows([0.129], flagged=False))
        self.assertNotIn("--rtf", cmds["scene_gen"])
        self.assertNotIn("rtf", report)

    def test_rtf_is_passed_to_scene_gen_and_recorded(self):
        report, cmds = tal.EvalHarness.run(["--rtf", "0.5"], tal.eval_rows([0.129], flagged=False))
        s = cmds["scene_gen"]
        self.assertEqual(s[s.index("--rtf") + 1], "0.5")
        self.assertEqual(report["rtf"], 0.5)

    def test_invalid_rtf_rejected(self):
        for bad in ("0", "-1", "inf", "nan"):
            code, cmds = tal.EvalHarness.run(["--rtf", bad], tal.eval_rows([0.16]), expect_exit=True)
            self.assertEqual(code, 2, bad)
            self.assertIn("--rtf 必须", cmds["stderr"], f"{bad}：要是我们的校验报的，不是「不认识的参数」")


if __name__ == "__main__":
    unittest.main()
