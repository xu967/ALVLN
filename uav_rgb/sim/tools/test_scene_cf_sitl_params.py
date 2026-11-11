#!/usr/bin/env python3
"""--cf-sitl-params（任务卡 W1-10：满电档，Gazebo 插件读参数副本）的测试。不起 Gazebo。

背景：E1 世界模板里 CfSitlSystem 插件的 <params> 写死仓库相对路径 sim/dynamics/e1_mini.json，插件在 gz 进程里自己读它。
collect.py --cf-sitl-params 只让 Python 一侧读副本（manifest 的 dynamics_profile、params_sha256），gz 里的物理仍是 50% 电量：
manifest 写满电指纹、动力学却是半电。修法是生成世界时把插件的 <params> 改写成副本的绝对路径（scene_gen --cf-sitl-params），
评测入口把同一个路径同时交给 scene_gen 与 collect.py。

覆盖：
  scene_gen：默认世界与 meta 不变（仓库相对路径、meta 无新键）；给了才只改插件块里的 <params> 一行，meta 多 cf_sitl_params，
    world_sha256 随世界更新；相对路径存成绝对路径；模板注释里的 "<params>" 字样不被碰；X3 模板、不存在的文件、
    路径含 XML 特殊字符都拒绝；与 --rtf / --camera-rate 同用；forest_e1 与 course_base 的世界同样适用。
  collect_parallel.scene_gen_extra：给了才透传，没给不传（旧命令行逐字不变）。
  eval_closed_loop：--cf-sitl-params 同时透传给 scene_gen 与 collect.py；eval.json 顶层记 cf_sitl_params（路径与两个 sha）
    与 dynamics_profile，没给没有这两个键；只能配 --dynamics cf_sitl；文件不存在拒绝；dynamics_profile_of 的三种情形。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_scene_cf_sitl_params.py
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
import eval_closed_loop as ec
import scene_gen
import test_action_latency as tal   # 以模块名引用，借 EvalHarness / eval_rows

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PARAMS_LINE = "<params>sim/dynamics/e1_mini.json</params>"
PROFILE_FULL = "cf_sitl_v1:06e5f017:lib31"


def write_copy(directory, name="soc100.json", soc=1.0):
    """e1_mini.json 的一份副本，只把 sim.battery_soc_init 改成 soc（不借 make_params_variant：本文件只测世界与评测入口）。"""
    with open(os.path.join(REPO, "sim", "dynamics", "e1_mini.json"), encoding="utf-8") as f:
        obj = json.load(f)
    obj["sim"]["value"]["battery_soc_init"] = soc
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def gen_world(extra, seed=1, vehicle="e1_mini"):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "w.sdf")
        argv = ["scene_gen.py", "--seed", str(seed), "--out", out] + (["--vehicle", vehicle] if vehicle else []) + extra
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            rc = scene_gen.main()
        assert rc == 0
        return Path(out).read_text(), json.loads(Path(out).with_suffix(".json").read_text())


def gen_world_error(extra, vehicle="e1_mini"):
    err = io.StringIO()
    with tempfile.TemporaryDirectory() as d, contextlib.redirect_stderr(err), \
            patch.object(sys, "argv", ["scene_gen.py", "--seed", "1", "--out", os.path.join(d, "w.sdf")]
                         + (["--vehicle", vehicle] if vehicle else []) + extra):
        try:
            scene_gen.main()
        except SystemExit as e:
            return e.code, err.getvalue()
    return None, err.getvalue()


class SceneGenCfSitlParams(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.copy = write_copy(self._tmp.name)

    def test_default_world_keeps_the_repo_relative_params_and_meta_has_no_new_key(self):
        world, meta = gen_world([])
        self.assertEqual(world.count(DEFAULT_PARAMS_LINE), 1)
        self.assertNotIn("cf_sitl_params", meta)
        self.assertEqual(meta["world_sha256"], hashlib.sha256(world.encode("utf-8")).hexdigest())

    def test_flag_rewrites_only_the_plugin_params_line(self):
        base, meta0 = gen_world([])
        world, meta = gen_world(["--cf-sitl-params", self.copy])
        self.assertEqual(world, base.replace(DEFAULT_PARAMS_LINE, f"<params>{self.copy}</params>"), "只改这一行")
        self.assertEqual(world.count(f"<params>{self.copy}</params>"), 1)
        self.assertEqual(meta["cf_sitl_params"], self.copy)
        self.assertEqual(meta["world_sha256"], hashlib.sha256(world.encode("utf-8")).hexdigest())
        rest = {k: v for k, v in meta.items() if k not in ("cf_sitl_params", "world_sha256")}
        rest0 = {k: v for k, v in meta0.items() if k != "world_sha256"}
        self.assertEqual(rest, rest0, "其余 meta 逐项不变（几何、gid、split 不随参数副本变）")

    def test_relative_path_is_stored_absolute(self):
        old = os.getcwd()
        os.chdir(self._tmp.name)
        try:
            world, meta = gen_world(["--cf-sitl-params", "soc100.json"])
        finally:
            os.chdir(old)
        expect = os.path.join(os.path.realpath(self._tmp.name), "soc100.json")   # chdir 之后 getcwd() 给的是真实路径
        self.assertEqual(meta["cf_sitl_params"], expect)
        self.assertIn(f"<params>{expect}</params>", world)

    def test_template_comment_mentioning_params_is_left_alone(self):
        base, _ = gen_world([])
        world, _ = gen_world(["--cf-sitl-params", self.copy])
        marker = "<params> is repo-relative"
        self.assertEqual(base.count(marker), 1, "模板注释里确有这个字样，测试才有意义")
        self.assertEqual(world.count(marker), 1)

    def test_combined_with_rtf_and_camera_rate(self):
        base, _ = gen_world(["--rtf", "0.5", "--camera-rate", "10"])
        world, meta = gen_world(["--rtf", "0.5", "--camera-rate", "10", "--cf-sitl-params", self.copy])
        self.assertEqual(world, base.replace(DEFAULT_PARAMS_LINE, f"<params>{self.copy}</params>"))
        self.assertEqual(meta["real_time_factor"], 0.5)
        self.assertEqual(meta["world_sha256"], hashlib.sha256(world.encode("utf-8")).hexdigest())

    def test_other_tasks_use_the_same_plugin_block(self):
        for extra in (["--task", "course_base", "--blocks", "3"], ["--task", "forest_e1", "--density", "sparse"]):
            base, _ = gen_world(extra)
            world, meta = gen_world(extra + ["--cf-sitl-params", self.copy])
            self.assertEqual(world, base.replace(DEFAULT_PARAMS_LINE, f"<params>{self.copy}</params>"), extra)
            self.assertEqual(meta["cf_sitl_params"], self.copy, extra)

    def test_x3_template_rejects_the_flag(self):
        for vehicle in (None, "x3"):
            code, err = gen_world_error(["--cf-sitl-params", self.copy], vehicle=vehicle)
            self.assertEqual(code, 2, vehicle)
            self.assertIn("--cf-sitl-params 只能配 E1", err, f"{vehicle}：要是我们的校验报的，不是「不认识的参数」")

    def test_missing_file_rejected(self):
        code, err = gen_world_error(["--cf-sitl-params", os.path.join(self._tmp.name, "nope.json")])
        self.assertEqual(code, 2)
        self.assertIn("--cf-sitl-params", err)
        self.assertIn("不存在", err)

    def test_path_with_xml_special_characters_rejected(self):
        for name in ("a&b.json", "a<b>.json"):
            path = write_copy(self._tmp.name, name=name)
            code, err = gen_world_error(["--cf-sitl-params", path])
            self.assertEqual(code, 2, name)
            self.assertIn("XML", err, name)


class SceneGenExtra(unittest.TestCase):
    def test_passes_the_flag_only_when_given(self):
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(cf_sitl_params="/x/y.json")),
                         ["--cf-sitl-params", "/x/y.json"])
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace(cf_sitl_params=None)), [])
        self.assertEqual(cp.scene_gen_extra(SimpleNamespace()), [], "collect_parallel 的参数里没有这个键：旧命令行不变")

    def test_other_options_keep_their_order_and_flag_is_appended(self):
        ns = SimpleNamespace(camera_rate=10.0, rtf=0.5, density=None, blocks=3, cf_sitl_params="/x/y.json")
        self.assertEqual(cp.scene_gen_extra(ns), ["--camera-rate", "10", "--rtf", "0.5", "--blocks", "3",
                                                  "--cf-sitl-params", "/x/y.json"])


class EvalCfSitlParams(unittest.TestCase):
    CF = ["--dynamics", "cf_sitl", "--vehicle", "e1_mini"]

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.copy = write_copy(self._tmp.name)
        patcher = patch.object(cp, "find_plugin_dir", return_value="/nonexistent-plugin-dir")
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_eval(self, extra, manifest=None, **kw):
        return tal.EvalHarness.run(self.CF + extra, tal.eval_rows([0.2], flagged=False), manifest, **kw)

    def test_default_has_no_flag_and_no_new_top_keys(self):
        report, cmds = self.run_eval([], {"dynamics_profile": "cf_sitl_v1:1bc8fedf:lib31"})
        self.assertNotIn("--cf-sitl-params", cmds["scene_gen"])
        self.assertNotIn("--cf-sitl-params", cmds["collect"])
        for key in ("cf_sitl_params", "dynamics_profile"):
            self.assertNotIn(key, report)
        self.assertEqual(report["per_episode"][0]["dynamics_profile"], "cf_sitl_v1:1bc8fedf:lib31", "逐集的记录本来就有")

    def test_flag_goes_to_scene_gen_and_collect_and_is_recorded(self):
        report, cmds = self.run_eval(["--cf-sitl-params", self.copy], {"dynamics_profile": PROFILE_FULL})
        for name in ("scene_gen", "collect"):
            c = cmds[name]
            self.assertEqual(c[c.index("--cf-sitl-params") + 1], self.copy, name)
            self.assertEqual(c.count("--cf-sitl-params"), 1, name)
        rec = report["cf_sitl_params"]
        raw = Path(self.copy).read_bytes()
        self.assertEqual(rec["path"], self.copy)
        self.assertEqual(rec["sha256"], hashlib.sha256(raw).hexdigest())
        obj = json.loads(raw)
        values = {k: v["value"] for k, v in obj.items() if isinstance(v, dict) and "value" in v}
        canon = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self.assertEqual(rec["values_sha256"], hashlib.sha256(canon.encode("utf-8")).hexdigest())
        self.assertEqual(report["dynamics_profile"], PROFILE_FULL)

    def test_relative_path_is_made_absolute_before_use(self):
        rel = os.path.relpath(self.copy, os.getcwd())
        self.assertFalse(os.path.isabs(rel))
        report, cmds = self.run_eval(["--cf-sitl-params", rel], {"dynamics_profile": PROFILE_FULL})
        c = cmds["collect"]
        self.assertEqual(c[c.index("--cf-sitl-params") + 1], os.path.abspath(rel))
        self.assertEqual(report["cf_sitl_params"]["path"], os.path.abspath(rel))

    def test_only_with_cf_sitl_dynamics(self):
        code, cmds = tal.EvalHarness.run(["--dynamics", "x3_velocity", "--cf-sitl-params", self.copy],
                                         tal.eval_rows([0.2], flagged=False), expect_exit=True)
        self.assertEqual(code, 2)
        self.assertIn("--cf-sitl-params 只能和 --dynamics cf_sitl 一起用", cmds["stderr"])

    def test_missing_file_rejected(self):
        code, cmds = self.run_eval(["--cf-sitl-params", os.path.join(self._tmp.name, "nope.json")], expect_exit=True)
        self.assertEqual(code, 2)
        self.assertIn("--cf-sitl-params", cmds["stderr"])
        self.assertIn("不存在", cmds["stderr"])

    def test_dynamics_profile_of(self):
        self.assertIsNone(ec.dynamics_profile_of([]))
        self.assertIsNone(ec.dynamics_profile_of([{"seed": 1}, {"seed": 2, "dynamics_profile": None}]))
        self.assertEqual(ec.dynamics_profile_of([{"dynamics_profile": "a"}, {"dynamics_profile": "a"}, {}]), "a")
        self.assertEqual(ec.dynamics_profile_of([{"dynamics_profile": "b"}, {"dynamics_profile": "a"}]),
                         ["a", "b"], "各集不一致是数据错误：列出全部，不挑一个")


if __name__ == "__main__":
    unittest.main()
