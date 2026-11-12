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

    def test_bad_params_file_and_xml_chars_fail_fast_before_running_episodes(self):
        bad_syntax = os.path.join(self._tmp.name, "corrupt.json")
        Path(bad_syntax).write_text("{not-json", encoding="utf-8")
        bad_list = os.path.join(self._tmp.name, "list.json")
        Path(bad_list).write_text("[1, 2, 3]", encoding="utf-8")
        bad_empty = os.path.join(self._tmp.name, "empty.json")
        Path(bad_empty).write_text("{}", encoding="utf-8")
        bad_xml = write_copy(self._tmp.name, name="bad&char.json")
        for bad_path in (bad_syntax, bad_list, bad_empty, bad_xml):
            code, cmds = self.run_eval(["--cf-sitl-params", bad_path], expect_exit=True)
            self.assertEqual(code, 2, bad_path)
            self.assertNotIn("scene_gen", cmds, f"{bad_path}: 应在起任何子进程前当场 rc=2")
            self.assertNotIn("collect", cmds, bad_path)
            self.assertIn("--cf-sitl-params", cmds["stderr"], bad_path)

    def test_all_failed_episodes_still_record_non_null_top_level_dynamics_profile(self):
        # 1) collect 子进程 rc=1 全失败：per_episode 为空，顶层 dynamics_profile 仍必须等于参数副本的指纹
        report_rc1, _ = self.run_eval(["--cf-sitl-params", self.copy], {}, collect_rc=1)
        self.assertEqual(report_rc1["per_episode"], [])
        self.assertEqual(report_rc1["failed_count"], 1)
        self.assertEqual(report_rc1["dynamics_profile"], PROFILE_FULL)
        self.assertNotIn("dynamics_profile", report_rc1["failed"][0])

        # 2) 迟到步导致全失败（collect_rc=0，但进 failed[]）：per_episode 为空，顶层 dynamics_profile 同样是 PROFILE_FULL
        report_late, _ = tal.EvalHarness.run(
            self.CF + ["--cf-sitl-params", self.copy, "--action-latency", "0.2"],
            tal.eval_rows([0.35], late_ids=(0,)),
            {"dynamics_profile": PROFILE_FULL, "action_latency_s": 0.2, "late_dispatch_steps": 1})
        self.assertEqual(report_late["per_episode"], [])
        self.assertEqual(report_late["failed_count"], 1)
        self.assertEqual(report_late["dynamics_profile"], PROFILE_FULL)


class CollectParallelCfSitlParams(unittest.TestCase):
    """I1-2：collect_parallel.py 命令行 --cf-sitl-params、collect_command 透传、collection_manifest 记录与默认逐字不变。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.copy = write_copy(self._tmp.name)
        patcher = patch.object(cp, "find_plugin_dir", return_value="/fake-plugin-dir")
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def _ns(**over):
        base = dict(steps=40, hold=1.0, policy="teacher", teacher_planner="astar", rl_checkpoint="",
                    window_from="observation", dynamics="cf_sitl", save_depth=False, vehicle="e1_mini",
                    student_ckpt=None, infer_bin=None, beta=0.0, dagger_round=0,
                    stop_at_goal=False, tiebreak="legacy")
        base.update(over)
        return SimpleNamespace(**base)

    def test_collect_command_default_unchanged_and_forwards_when_set(self):
        t = cp.Task(0, 0, 2, 2, "train", "world.sdf", "scene.json")
        cmd_def = cp.collect_command(t, self._ns())
        self.assertNotIn("--cf-sitl-params", cmd_def)
        self.assertEqual(cp.collect_command(t, self._ns(cf_sitl_params=None)), cmd_def)
        cmd_with = cp.collect_command(t, self._ns(cf_sitl_params=self.copy))
        self.assertEqual(cmd_with[:-2], cmd_def)
        self.assertEqual(cmd_with[-2:], ["--cf-sitl-params", self.copy])

    def _run_parallel(self, extra_argv, expect_exit=False):
        from unittest.mock import Mock
        seen = {"scene_gen": [], "collect": []}
        pillars = [{"x": 1.5, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]}]

        def fake_run(cmd, **_kw):
            if cp.SCENE_GEN in cmd:
                seen["scene_gen"].append(list(cmd))
                w = Path(cmd[cmd.index("--out") + 1])
                w.write_text("<sdf/>")
                meta = {"pillars": pillars, "pillar_count": 1, "vehicle": "e1_mini"}
                if "--cf-sitl-params" in cmd:
                    meta["cf_sitl_params"] = cmd[cmd.index("--cf-sitl-params") + 1]
                w.with_suffix(".json").write_text(json.dumps(meta))
            return Mock(returncode=0, stdout="", stderr="")

        def fake_popen(item, args_obj, _env, _log):
            seen["collect"].append(cp.collect_command(item, args_obj))
            os.makedirs(item.staging, exist_ok=True)
            Path(item.staging, "episode_manifest.json").write_text(json.dumps({
                "num_steps": args_obj.steps,
                "validity": {"episode_executed": True, "camera_verified": True, "physics_verified": True},
            }))
            proc, pump = Mock(), Mock()
            proc.wait.return_value = 0
            return proc, pump

        err = io.StringIO()
        out = os.path.join(self._tmp.name, f"pout_{len(os.listdir(self._tmp.name))}")
        argv = ["collect_parallel.py", "--instances", "1", "--episodes-per-instance", "1",
                "--seeds", "2", "--steps", "3", "--dynamics", "cf_sitl", "--vehicle", "e1_mini",
                "--out", out] + extra_argv
        with patch.object(sys, "argv", argv), \
             patch.object(cp.subprocess, "run", side_effect=fake_run), \
             patch.object(cp, "_start_sim", return_value=Mock(pid=77)), \
             patch.object(cp, "wait_ready"), \
             patch.object(cp, "_popen_collect", side_effect=fake_popen), \
             patch.object(cp, "stop_group"), patch.object(cp, "cleanup_all"), \
             patch.object(cp, "_sweep_world_ruby", return_value=[]), \
             patch.object(cp, "_ruby_pids", return_value=[]), \
             patch.object(cp, "XDG_RUNTIME_DIR", os.path.join(self._tmp.name, "xdg")), \
             patch.object(cp.time, "sleep"), patch.object(cp.signal, "signal"), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            if expect_exit:
                with self.assertRaises(SystemExit) as cm:
                    cp.main()
                return cm.exception.code, err.getvalue(), seen, None
            rc = cp.main()
        man = json.loads(Path(out, "collection_manifest.json").read_text())
        return rc, err.getvalue(), seen, man

    def test_parallel_main_wires_both_sides_and_keeps_default_unchanged(self):
        rc0, _, seen0, man0 = self._run_parallel([])
        self.assertEqual(rc0, 0)
        self.assertNotIn("--cf-sitl-params", seen0["scene_gen"][0])
        self.assertNotIn("--cf-sitl-params", seen0["collect"][0])
        self.assertNotIn("cf_sitl_params", man0)

        rel = os.path.relpath(self.copy, os.getcwd())
        rc1, _, seen1, man1 = self._run_parallel(["--cf-sitl-params", rel])
        self.assertEqual(rc1, 0)
        want = os.path.abspath(rel)
        self.assertEqual(seen1["scene_gen"][0][-2:], ["--cf-sitl-params", want])
        self.assertEqual(seen1["collect"][0][-2:], ["--cf-sitl-params", want])
        self.assertEqual(man1["cf_sitl_params"], want)

    def test_parallel_main_rejects_bad_usage(self):
        bad_json = os.path.join(self._tmp.name, "bad.json")
        Path(bad_json).write_text("{not valid json")
        bad_xml = write_copy(self._tmp.name, name="a&b.json")
        for extra, frag in (
            (["--dynamics", "x3_velocity", "--vehicle", "x3", "--cf-sitl-params", self.copy], "只能和 --dynamics cf_sitl"),
            (["--cf-sitl-params", os.path.join(self._tmp.name, "missing.json")], "不存在"),
            (["--cf-sitl-params", bad_xml], "XML"),
            (["--cf-sitl-params", bad_json], "--cf-sitl-params"),
        ):
            code, stderr, _, _ = self._run_parallel(extra, expect_exit=True)
            self.assertEqual(code, 2, extra)
            self.assertIn(frag, stderr, extra)


class CollectSceneMetaParamsGuard(unittest.TestCase):
    """K-2：collect.py 在 --dynamics cf_sitl + --scene-meta 下核对自身参数文件与 scene_meta['cf_sitl_params'] 的真实路径；
    并验证 eval_closed_loop.py（默认与带 --cf-sitl-params、含 l1 指令覆写）跑真 scene_gen + 真 collect 校验不被误伤。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.soc100 = write_copy(self._tmp.name, "soc100.json", soc=1.0)
        self.soc80 = write_copy(self._tmp.name, "soc80.json", soc=0.8)
        patcher = patch.object(cp, "find_plugin_dir", return_value="/fake-plugin-dir")
        patcher.start()
        self.addCleanup(patcher.stop)

    def _make_scene_meta(self, extra_scene_argv, seed=2):
        w = os.path.join(self._tmp.name, f"w_{len(os.listdir(self._tmp.name))}.sdf")
        argv = ["scene_gen.py", "--seed", str(seed), "--vehicle", "e1_mini", "--out", w] + extra_scene_argv
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            assert scene_gen.main() == 0
        return os.path.splitext(w)[0] + ".json"

    def _run_collect_main(self, meta_path, collect_extra):
        import collect
        import split_manifest as sm
        import teacher
        import test_collect as tc
        from unittest.mock import Mock
        meta_obj = json.loads(Path(meta_path).read_text())
        out = os.path.join(self._tmp.name, f"ep_{len(os.listdir(self._tmp.name))}")
        argv = ["collect.py", "--out", out, "--scene-meta", meta_path,
                "--split", sm.split_for_scene(meta_obj),
                "--dynamics", "cf_sitl", "--vehicle", "e1_mini", "--steps", "1"] + collect_extra
        subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}
        fake_sil = Mock(takeoff_failsafes=0)
        fake_sil.params = collect.cf_sitl_exec.CfSitlParams.from_json(
            collect_extra[collect_extra.index("--cf-sitl-params") + 1]
            if "--cf-sitl-params" in collect_extra else collect.cf_sitl_exec.DEFAULT_PARAMS)
        err = io.StringIO()
        code = 0
        with patch.object(sys, "argv", argv), \
             patch.object(collect, "open_subs", return_value=subs), \
             patch.object(collect, "wait_ready", return_value=None), \
             patch.object(collect, "read_camera_info", return_value=None), \
             patch.object(collect, "take_off_cf_sitl", return_value=0), \
             patch.object(collect, "stop_executor"), \
             patch.object(collect, "close_subs"), \
             patch.object(collect.cf_sitl_exec, "CfSitlExecutor", return_value=fake_sil), \
             patch.object(collect, "collect_one", return_value=(tc.fake_row(0, 1.0), None)), \
             patch.object(teacher, "warm_free_cells", return_value=0), \
             contextlib.redirect_stderr(err):
            try:
                collect.main()
            except SystemExit as e:
                code = e.code
        return code, err.getvalue(), out

    def test_collect_rejects_one_sided_or_mismatched_params_and_accepts_matching_paths(self):
        import collect
        meta_def = self._make_scene_meta([])
        meta_100 = self._make_scene_meta(["--cf-sitl-params", self.soc100])
        link_100 = os.path.join(self._tmp.name, "link_soc100.json")
        os.symlink(self.soc100, link_100)

        # 匹配情形：两边都缺省、两边同一文件（含软链）、一边显式传默认 e1_mini.json 另一边缺省
        self.assertEqual(self._run_collect_main(meta_def, [])[0], 0)
        self.assertEqual(self._run_collect_main(meta_100, ["--cf-sitl-params", self.soc100])[0], 0)
        self.assertEqual(self._run_collect_main(meta_100, ["--cf-sitl-params", link_100])[0], 0, "按 realpath 比，软链一致放行")
        self.assertEqual(self._run_collect_main(meta_def, ["--cf-sitl-params", collect.cf_sitl_exec.DEFAULT_PARAMS])[0], 0)

        # 不匹配情形：只给世界不给 collect、只给 collect 不给世界、两边给不同副本
        for m, extra in (
            (meta_100, []),
            (meta_def, ["--cf-sitl-params", self.soc100]),
            (meta_100, ["--cf-sitl-params", self.soc80]),
        ):
            code, stderr, out = self._run_collect_main(m, extra)
            self.assertEqual(code, 2, (m, extra))
            self.assertIn("cf_sitl_params", stderr)
            self.assertFalse(os.path.isfile(os.path.join(out, "episode_manifest.json")))

    def test_eval_closed_loop_runs_real_scene_gen_and_real_collect_guard_without_false_alarm(self):
        import collect
        import teacher
        import test_collect as tc
        from unittest.mock import Mock

        def run_real_wired_eval(extra_argv):
            def real_scene_gen(cmd, **_kw):
                with patch.object(sys, "argv", [cmd[1]] + list(cmd[2:])), contextlib.redirect_stdout(io.StringIO()):
                    rc = scene_gen.main()
                return Mock(returncode=rc)

            def real_collect_proc(cmd, **_kw):
                fake_sil = Mock(takeoff_failsafes=0)
                p_arg = cmd[cmd.index("--cf-sitl-params") + 1] if "--cf-sitl-params" in cmd else collect.cf_sitl_exec.DEFAULT_PARAMS
                fake_sil.params = collect.cf_sitl_exec.CfSitlParams.from_json(p_arg)
                fake_sil.params.lib_version = 31
                subs = {k: tc.FakeSub(latest=tc.pose_msg(1.0)) for k in ("rgb", "depth", "pose", "contact", "camera_info")}
                err_buf = io.StringIO()
                rc = 0
                with patch.object(sys, "argv", [cmd[1]] + list(cmd[2:])), \
                     patch.object(collect, "open_subs", return_value=subs), \
                     patch.object(collect, "wait_ready", return_value=None), \
                     patch.object(collect, "read_camera_info", return_value=None), \
                     patch.object(collect, "take_off_cf_sitl", return_value=0), \
                     patch.object(collect, "stop_executor"), \
                     patch.object(collect, "close_subs"), \
                     patch.object(collect.cf_sitl_exec, "CfSitlExecutor", return_value=fake_sil), \
                     patch.object(collect, "collect_one", return_value=(tc.fake_row(0, 6.5), None)), \
                     patch.object(teacher, "warm_free_cells", return_value=0), \
                     contextlib.redirect_stderr(err_buf):
                    try:
                        collect.main()
                    except SystemExit as e:
                        rc = e.code or 0
                proc = Mock(returncode=rc)
                proc.communicate.return_value = ("", err_buf.getvalue())
                return proc

            out = os.path.join(self._tmp.name, f"ev_{len(os.listdir(self._tmp.name))}")
            argv = ["eval_closed_loop.py", "--policy", "teacher", "--dynamics", "cf_sitl", "--vehicle", "e1_mini",
                    "--seeds", "1", "--steps", "1", "--out", out] + extra_argv
            with patch.object(ec.sys, "argv", argv), \
                 patch.object(ec.subprocess, "run", side_effect=real_scene_gen), \
                 patch.object(ec.subprocess, "Popen", side_effect=real_collect_proc), \
                 patch.object(ec.cp, "_start_sim", return_value=Mock()), \
                 patch.object(ec.cp, "wait_ready"), patch.object(ec.cp, "_register"), \
                 patch.object(ec.cp, "stop_group"), patch.object(ec.cp, "cleanup_all"), \
                 patch.object(ec.cp, "XDG_RUNTIME_DIR", os.path.join(self._tmp.name, "xdg")), \
                 patch.object(ec.time, "sleep"), patch.object(ec.signal, "signal"), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                rc = ec.main()
            return rc, json.loads(Path(out, "eval.json").read_text())

        # 1) 默认不传 --cf-sitl-params
        rc_def, rep_def = run_real_wired_eval([])
        self.assertEqual((rc_def, rep_def["failed_count"], rep_def["episodes_done"]), (0, 0, 1))
        # 2) 传相对路径的 --cf-sitl-params
        rel_100 = os.path.relpath(self.soc100, os.getcwd())
        rc_100, rep_100 = run_real_wired_eval(["--cf-sitl-params", rel_100])
        self.assertEqual((rc_100, rep_100["failed_count"], rep_100["dynamics_profile"]), (0, 0, PROFILE_FULL))


if __name__ == "__main__":
    unittest.main()
