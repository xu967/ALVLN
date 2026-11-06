#!/usr/bin/env python3
"""Headless regression tests for UAV Navigation Studio data/geometry helpers.

StudioGuiTests 需要 tkinter + DISPLAY（本机 DISPLAY=:20），没有显示时跳过；其余用替身对象，不开窗口。
"""
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import studio


def _write_uavr(path, meta):
    """最小 UAVR：48 B 头 + meta JSON（0 样本，只测尾部 meta 读取）。"""
    raw = json.dumps(meta).encode("utf-8")
    header = struct.pack("<4sHHIHHBBBBQQQI", b"UAVR", 1, 48, 0, 120, 160, 3,
                         1, 1, 0, 48, 48, 48, len(raw))
    Path(path).write_bytes(header + raw)


class _Var:
    def __init__(self, value=""):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _Widget:
    def configure(self, **_kwargs):
        pass


class _Text(_Widget):
    def __init__(self):
        self.text = ""

    def insert(self, _index, text):
        self.text += text

    def delete(self, *_args):
        self.text = ""

    def see(self, *_args):
        pass


def _headless_app(run_dir=None):
    """不建 Tk 窗口的 StudioApp：只放被测方法会碰到的属性。"""
    app = object.__new__(studio.StudioApp)
    app.run_dir = Path(run_dir).resolve() if run_dir else None
    app.sidecar = Path(str(app.run_dir) + ".monitor") if app.run_dir else None
    app.run_meta = {}
    app.proc = app.proc_log = None
    app.eval_proc = app.eval_log = app.eval_out = None
    app._eval_auto, app._eval_owner = False, None
    app._live_owner = None
    app._detached_pid = None
    app._metric_sig = None
    app._events, app._val_events = deque(maxlen=300), []
    app._latest_train_event, app._event_tail = None, {}
    app._train_log_state, app._train_log_lines = {}, 0
    app._rows = []
    app._eval_index, app._standalone_evals = {}, []
    app._closing = False
    app._vehicle = studio.DEFAULT_VEHICLE
    app._sim_last_seq = {"fpv": 0, "chase": 0, "top": 0}
    app._sim_last_shown = {}
    app.render_backend_var = _Var(studio.DEFAULT_RENDER_BACKEND)
    app.card_vars = {k: _Var("—") for k in ("epoch", "train_ce", "dev_ce", "dev_macro",
                                            "test_macro", "test_bal", "closed_loop_sr")}
    app.hero_sub = {"test_macro": _Var(), "closed_loop_sr": _Var()}
    app.test_summary_var = _Var()
    for name in ("_eval_status_var", "_training_status_var", "_sim_status_var",
                 "_sim_stream_var", "ckpt_var", "run_title_var", "sim_info_var"):
        setattr(app, name, _Var())
    for name in ("eval_btn", "eval_stop_btn", "start_train_btn", "stop_train_btn",
                 "stop_detached_btn"):
        setattr(app, name, _Widget())
    app.event_text, app.train_log_text = _Text(), _Text()
    app.tabs = type("Tabs", (), {"select": lambda self, _tab: None})()
    app.dashboard_tab = None
    return app


def _eval_json(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema": "uavrgb.eval.v1", "success_rate": 0.5, "episodes_done": 1,
              "episodes_requested": 1, "failed_count": 0}
    report.update(fields)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def _patched_runs_root(test, root):
    old = studio.RUNS_ROOT
    studio.RUNS_ROOT = Path(root)
    test.addCleanup(setattr, studio, "RUNS_ROOT", old)


def _restore_env(key, value):
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value


def _patched_messagebox(test, **answers):
    """替换 studio.messagebox：showerror/showinfo 只记录，askyesno 按 answers[标题] 回答（缺省 True）。"""
    log = []

    def show(kind):
        return lambda title, *_a, **_k: log.append((kind, title))

    def ask(title, *_a, **_k):
        log.append(("askyesno", title))
        return answers.get(title, True)
    box = type("Box", (), {"showerror": staticmethod(show("showerror")),
                           "showinfo": staticmethod(show("showinfo")),
                           "showwarning": staticmethod(show("showwarning")),
                           "askyesno": staticmethod(ask)})
    old = studio.messagebox
    studio.messagebox = box
    test.addCleanup(setattr, studio, "messagebox", old)
    return log


def _patched_popen(test):
    """subprocess.Popen 替身：记录命令与 env，不真正启动进程。"""
    calls = []

    class Proc:
        pid = 4242

        def poll(self):
            return None

    def popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return Proc()
    old = studio.subprocess.Popen
    studio.subprocess.Popen = popen
    test.addCleanup(setattr, studio.subprocess, "Popen", old)
    return calls


def _training_run(root, meta, name="run"):
    """带训练数据的 run：<run>/ckpt_best.bin + <run>.monitor/job.json 指向一个只有 meta 的 .uavr。"""
    run = Path(root) / name
    run.mkdir()
    (run / "ckpt_best.bin").write_bytes(b"weights")
    data = Path(root) / f"{name}.uavr"
    _write_uavr(data, meta)
    side = Path(str(run) + ".monitor")
    side.mkdir()
    (side / "job.json").write_text(json.dumps({"config": {"data": str(data)}}), encoding="utf-8")
    return run, data


class StudioHelperTests(unittest.TestCase):
    def test_parse_pose_v(self):
        msg = {"pose": [{"name": "x3", "position": {"x": 1.0, "y": -2.0, "z": 0.7},
                         "orientation": {"x": 0, "y": 0, "z": 0.70710678, "w": 0.70710678},
                         "header": {"stamp": {"sec": 12, "nsec": 345}}}]}
        pose = studio.parse_pose_msg(msg)
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose[0], 1.0)
        self.assertAlmostEqual(pose[1], -2.0)
        self.assertAlmostEqual(pose[2], 0.7)
        self.assertAlmostEqual(pose[3], 1.5707963, places=5)
        self.assertEqual(pose[4], 12_000_000_345)
        self.assertEqual(studio.pose_stamp_ns(msg), 12_000_000_345)

    def test_parse_pose_single(self):
        msg = {"position": {"x": 0, "y": 0, "z": 1},
               "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}
        pose = studio.parse_pose_msg(msg)
        self.assertEqual(pose[:3], (0.0, 0.0, 1.0))
        self.assertAlmostEqual(pose[3], 0.0)

    def test_parse_sdf_collision_footprints(self):
        sdf = '''<sdf version="1.6"><world name="test">
          <model name="ground_plane"><link name="l"><collision name="c"><geometry><plane><normal>0 0 1</normal></plane></geometry></collision></link></model>
          <model name="x3"><link name="l"><collision name="c"><geometry><box><size>1 1 1</size></box></geometry></collision></link></model>
          <model name="tree_1"><pose>2 3 1 0 0 0.5</pose><link name="l"><collision name="c"><geometry><box><size>0.4 0.6 2</size></box></geometry></collision></link></model>
          <model name="post"><pose>4 -1 0 0 0 0</pose><link name="l"><collision name="c"><geometry><cylinder><radius>0.2</radius><length>2</length></cylinder></geometry></collision></link></model>
        </world></sdf>'''
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "world.sdf"
            p.write_text(sdf, encoding="utf-8")
            obs = studio.parse_sdf_obstacles(p)
        self.assertEqual(len(obs), 2)
        tree = next(o for o in obs if o["name"] == "tree_1")
        self.assertEqual(tree["kind"], "box")
        self.assertAlmostEqual(tree["x"], 2.0)
        self.assertAlmostEqual(tree["y"], 3.0)
        self.assertAlmostEqual(tree["yaw"], 0.5)
        post = next(o for o in obs if o["name"] == "post")
        self.assertEqual(post["kind"], "circle")
        self.assertAlmostEqual(post["r"], 0.2)

    def test_summary_keeps_missing_test_null_and_separates_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "results.csv").write_text(
                "epoch,train_ce,dev_ce,dev_accuracy\n1,1.0,1.2,0.7\n2,0.7,0.9,0.8\n",
                encoding="utf-8")
            events = [
                {"event": "train_step", "step": 1},
                {"event": "validation", "split": "val", "loss": 0.9,
                 "action_accuracy": 0.8, "macro_f1": 0.72, "balanced_accuracy": 0.7},
                {"event": "status", "message": "best checkpoint: epoch 2 macro_f1=0.72"},
            ]
            (run / "metrics.jsonl").write_text(
                "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")
            (run / "ckpt_best.bin").write_bytes(b"checkpoint")
            meta = {"run_id": "test_run", "started_at": "2026-11-04T00:00:00Z",
                    "started_monotonic": 1.0, "config": {"epochs": 2, "text_dim": 16}}
            summary = studio.summarize_run(run, meta, 0, "finished")
        m = summary["metrics"]
        self.assertEqual(m["epochs_completed"], 2)
        self.assertAlmostEqual(m["best_validation_macro_f1"], 0.72)
        self.assertEqual(m["best_checkpoint_epoch"], 2)
        self.assertIsNone(m["test_macro_f1"])
        self.assertFalse(m["test_evaluated"])
        self.assertTrue(summary["artifacts"]["checkpoint_best"])
        self.assertIn("not navigation success rate", summary["notes"][0])

    def test_current_world_template_has_three_camera_sensors_and_keeps_fpv(self):
        template = studio.REPO_ROOT / "sim" / "worlds" / "uav_rgb.sdf"
        root = ET.parse(template).getroot()
        fpv = root.find(".//sensor[@name='camera']")
        chase = root.find(".//sensor[@name='chase_camera']")
        top = root.find(".//sensor[@name='top_camera']")
        self.assertIsNotNone(fpv)
        self.assertIsNotNone(chase)
        self.assertIsNotNone(top)
        self.assertEqual(fpv.findtext("topic"), "/uav/camera")
        self.assertEqual(chase.findtext("topic"), "/uav/chase_camera")
        self.assertEqual(chase.findtext("camera/image/width"), "640")
        self.assertEqual(chase.findtext("camera/image/height"), "360")
        self.assertEqual(top.findtext("topic"), "/uav/top_camera")
        self.assertEqual(top.findtext("camera/image/width"), "480")
        self.assertEqual(top.findtext("camera/image/height"), "360")

    def test_summary_derives_per_class_recall_and_report_matrix(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            confusion = [8, 2, 0, 0, 1, 9, 0, 0, 0, 1, 8, 1, 0, 0, 2, 8]
            event = {"event": "validation", "split": "test", "action_accuracy": 0.82,
                     "macro_f1": 0.81, "balanced_accuracy": 0.80, "confusion": confusion}
            (run / "metrics.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
            meta = {"run_id": "recall_test", "config": {"text_dim": 0}}
            summary = studio.summarize_run(run, meta, 0, "finished")
            studio.write_experiment_report(run, summary)
            report = (run / "experiment_report.md").read_text(encoding="utf-8")
        self.assertEqual(len(summary["metrics"]["test_per_class_recall"]), 4)
        self.assertAlmostEqual(summary["metrics"]["test_per_class_recall"][0]["recall"], 0.8)
        self.assertIn("Test confusion matrix", report)
        self.assertIn("SAFE_NEUTRAL", report)

    def test_read_jsonl_keeps_tail_for_final_test_events(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.jsonl"
            p.write_text("".join(json.dumps({"event": "train_step", "step": i}) + "\n"
                                   for i in range(10)) +
                         json.dumps({"event": "validation", "split": "test", "macro_f1": 0.7}) + "\n",
                         encoding="utf-8")
            rows = studio.read_jsonl(p, max_lines=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[-1].get("split"), "test")

    def test_read_jsonl_skips_bad_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "metrics.jsonl"
            p.write_text('{"event":"ok"}\nnot-json\n{"event":"done"}\n', encoding="utf-8")
            rows = studio.read_jsonl(p)
        self.assertEqual([x["event"] for x in rows], ["ok", "done"])

    def test_discover_runs_two_levels_and_skip_rules(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "group" / "run-a").mkdir(parents=True)
            (root / "group" / "run-a" / "metrics.jsonl").write_text("", encoding="utf-8")
            (root / "run-b").mkdir()
            (root / "run-b" / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
            (root / "group" / "not-a-run").mkdir()
            (root / "group" / "not-a-run" / "readme.txt").write_text("x", encoding="utf-8")
            for skipped in ("ep_0001", "simulation_logs", ".hidden"):
                (root / skipped).mkdir()
                (root / skipped / "eval.json").write_text("{}", encoding="utf-8")
            found = {p.name for p in studio.discover_runs([root])}
        self.assertEqual(found, {"run-a", "run-b"})

    def test_eval_index_maps_ckpt_parent_and_keeps_standalone(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "group" / "run-a"
            run.mkdir(parents=True)
            ckpt = run / "ckpt_best.bin"
            ckpt.write_bytes(b"weights")
            attached = root / "group" / "eval-1" / "eval.json"
            attached.parent.mkdir(parents=True)
            attached.write_text(json.dumps({"ckpt": str(ckpt), "success_rate": 0.5}),
                                encoding="utf-8")
            standalone = root / "teacher-eval" / "eval.json"
            standalone.parent.mkdir()
            standalone.write_text(json.dumps({"success_rate": 0.9}), encoding="utf-8")
            index, loose, stale = studio.eval_index([root])
        self.assertEqual(index[run.resolve()], [attached.resolve()])
        self.assertEqual(loose, [standalone.resolve()])
        self.assertEqual(stale, [])

    def test_eval_index_separates_stale_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            run = root / "run-a"
            run.mkdir()
            ckpt = run / "ckpt_best.bin"
            ckpt.write_bytes(b"new weights")
            good = run / "eval-good" / "eval.json"
            good.parent.mkdir()
            good.write_text(json.dumps({
                "ckpt": str(ckpt),
                "ckpt_sha256": hashlib.sha256(b"new weights").hexdigest(),
                "success_rate": 0.5}), encoding="utf-8")
            stale = run / "eval-old" / "eval.json"
            stale.parent.mkdir()
            stale.write_text(json.dumps({
                "ckpt": str(ckpt),
                "ckpt_sha256": hashlib.sha256(b"old weights").hexdigest(),
                "success_rate": 0.9}), encoding="utf-8")
            # 旧 eval 没有 sha 时不能判定为 stale，保持挂载但显示“未标注”。
            unlabeled = run / "eval-unlabeled" / "eval.json"
            unlabeled.parent.mkdir()
            unlabeled.write_text(json.dumps({
                "ckpt": str(ckpt), "success_rate": 0.7}), encoding="utf-8")
            index, loose, stale_paths = studio.eval_index([root])
        self.assertEqual(set(index[run.resolve()]),
                         {good.resolve(), unlabeled.resolve()})
        self.assertEqual(loose, [])
        self.assertEqual(stale_paths, [stale.resolve()])

    def test_eval_display_fields_do_not_invent_missing_values(self):
        fields = studio.eval_display_fields({"task": "l1", "dynamics": "x3_v1"})
        self.assertEqual(fields["task"], "l1")
        self.assertEqual(fields["dynamics"], "x3_v1")
        self.assertEqual(fields["criterion"], "未标注")
        self.assertEqual(fields["window_from"], "未标注")
        context = studio.eval_rate_context({"success_criterion": "goal_reach_ever"})
        self.assertIn("requested=未标注", context)
        self.assertIn("failed=未标注", context)
        self.assertIn("episodes=—/未标注", context)

    def test_uavr_meta_infers_auto_eval_task_split_dynamics(self):
        cases = [
            ({"task_version": "campus.assets.v1"}, "campus_assets"),
            ({"task_version": "campus.v2"}, "campus"),
            ({"task_version": "l1.v4"}, "l1"),
            ({"geometry_version": "uavrgb.geom.v5"}, "base"),
            ({}, None),
        ]
        for meta, want in cases:
            self.assertEqual(studio.video_cli.infer_task(meta), want, meta)
        self.assertEqual(studio.video_cli.infer_dynamics({"dynamics_profile": "x3_v1"}),
                         "x3_velocity")
        self.assertEqual(studio.video_cli.infer_dynamics(
            {"episodes": [{"dynamics_profile": "cf_stab_v1:tilt2"}]}), "cf_stab")

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ds.uavr"
            _write_uavr(path, {
                "task_version": "campus.v2", "splits": {"train": 9, "val": 0, "test": 2},
                "dynamics_profile": "x3_v1", "window_from": "dispatch", "hold_s": 1.0})
            meta = studio.video_cli.read_uavr_meta(path)
            params, reason = studio.video_cli.auto_eval_params(meta)
            self.assertIsNone(reason)
            self.assertEqual(params["task"], "campus")
            self.assertEqual(params["split"], "test")
            self.assertEqual(params["dynamics"], "x3_velocity")
            self.assertEqual(params["window_from"], "dispatch")
            self.assertEqual(params["hold_s"], 1.0)

            # test=0 时回退 val；两者都空则跳过。
            _write_uavr(path, {"task_version": "l1.v4",
                               "splits": {"train": 1, "val": 4, "test": 0}})
            params, _ = studio.video_cli.auto_eval_params(
                studio.video_cli.read_uavr_meta(path))
            self.assertEqual(params["split"], "val")
            _write_uavr(path, {"task_version": "l1.v4",
                               "splits": {"train": 1, "val": 0, "test": 0}})
            params, reason = studio.video_cli.auto_eval_params(
                studio.video_cli.read_uavr_meta(path))
            self.assertIsNone(params)
            self.assertIn("test/val", reason)

    def test_run_data_path_prefers_trainer_config_and_falls_back(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            run.mkdir()
            self.assertEqual(studio.run_data_path(run, {"data": "/fallback.uavr"}),
                             Path("/fallback.uavr"))
            (run / "config.json").write_text(json.dumps({"data": "/trainer.uavr"}),
                                             encoding="utf-8")
            self.assertEqual(studio.run_data_path(run, {"data": "/fallback.uavr"}),
                             Path("/trainer.uavr"))

    def test_report_links_closed_loop_videos_relative_to_run(self):
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            video = run / "closed_loop_eval_1" / "videos" / "seed_1.mp4"
            video.parent.mkdir(parents=True)
            video.write_bytes(b"mp4")
            summary = {
                "run_id": "r", "status": "finished", "return_code": 0,
                "config": {}, "metrics": {}, "artifacts": {}, "live_demos": [],
                "monitor_dir": str(run) + ".monitor",
                "closed_loop_evaluations": [{
                    "task": "campus", "split": "test", "success_rate": 0.5,
                    "video_dir": str(video.parent), "videos": [str(video)]}],
            }
            studio.write_experiment_report(run, summary)
            report = (run / "experiment_report.md").read_text(encoding="utf-8")
        self.assertIn("[seed_1.mp4](closed_loop_eval_1/videos/seed_1.mp4)", report)

    def test_auto_eval_command_is_non_blocking_and_skips_confirm(self):
        """自动录像复用参数化入口：不弹确认框，命令带录像/推断参数。"""
        class Widget:
            def configure(self, **_kwargs):
                pass

        class Var:
            def __init__(self):
                self.value = None

            def set(self, value):
                self.value = value

        class Proc:
            pass

        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            run.mkdir()
            (run / "ckpt_best.bin").write_bytes(b"weights")
            app = object.__new__(studio.StudioApp)
            app.run_dir = run
            app.eval_proc = None
            app.eval_log = None
            app.eval_out = None
            app._eval_auto = False
            app._sim_proc = None
            app._sim_subs = {}
            app.eval_btn = Widget()
            app.eval_stop_btn = Widget()
            app._eval_status_var = Var()
            app.render_backend_var = _Var("software")
            calls = []
            old_popen = studio.subprocess.Popen
            studio.subprocess.Popen = lambda cmd, **kwargs: calls.append(cmd) or Proc()
            old_backend = os.environ.get("UAVRGB_RENDER_BACKEND")
            try:
                started = app.start_closed_loop_eval(auto={
                    "task": "campus", "split": "test", "dynamics": "cf_stab",
                    "episodes": 1, "window_from": "dispatch", "hold_s": 0.2})
            finally:
                studio.subprocess.Popen = old_popen
                _restore_env("UAVRGB_RENDER_BACKEND", old_backend)
            self.assertTrue(started)
            self.assertEqual(len(calls), 1)
            cmd = calls[0]
            self.assertIn("--record-video", cmd)
            self.assertEqual(cmd[cmd.index("--episodes") + 1], "1")
            self.assertEqual(cmd[cmd.index("--dynamics") + 1], "cf_stab")
            self.assertEqual(cmd[cmd.index("--window-from") + 1], "dispatch")
            self.assertIn("--text-mode", cmd)
            self.assertTrue(app.eval_out.name.startswith("closed_loop_eval_"))

    def test_classify_episode_prioritises_success_then_failure_mode(self):
        self.assertEqual(studio.classify_episode({"success": True, "collision": True}), "成功")
        # eval 的 false_forward 一定同时 collision=True，必须归为假前进。
        self.assertEqual(studio.classify_episode({"collision": True, "false_forward": True}), "假前进")
        self.assertEqual(studio.classify_episode({"collision": True, "stuck": True}), "碰撞")
        self.assertEqual(studio.classify_episode({"stuck": True, "false_forward": True}), "假前进")
        self.assertEqual(studio.classify_episode({"stuck": True}), "卡住")
        # 实测 20 集里 13 集所有 flag 全 false，必须归为“未到达”而不是静默成功。
        self.assertEqual(studio.classify_episode({"success": False, "collision": False,
                                                  "stuck": False, "false_forward": False}),
                         "未到达")
        self.assertEqual(studio.classify_episode({}), "未到达")


class EvalOwnershipTests(unittest.TestCase):
    """REVIEW #5 / AUDIT #5：异步评测、自动录像备注、演示只写启动它的 run。"""

    def _runs(self, d):
        a, b = Path(d) / "run-a", Path(d) / "run-b"
        for run in (a, b):
            run.mkdir()
            (run / "ckpt_best.bin").write_bytes(run.name.encode())
        side_b = Path(str(b) + ".monitor")
        side_b.mkdir()
        (side_b / "lifecycle.json").write_text(json.dumps({"status": "running", "pid": 1}))
        (side_b / "job.json").write_text(json.dumps({"run_id": "run-b"}))
        out = a / "closed_loop_eval_1"
        _eval_json(out / "eval.json", episodes_requested=20, episodes_done=20, success_rate=0.4)
        return a.resolve(), b.resolve(), out

    def _finish_after(self, a, out, switch):
        app = _headless_app(a)
        app.eval_proc, app.eval_out, app._eval_owner = object(), out, a
        switch(app)
        app._finish_closed_loop_eval(0)
        return app

    def test_switch_run_during_eval_archives_to_starting_run(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, out = self._runs(d)
            app = self._finish_after(a, out, lambda app: setattr(app, "run_dir", b))
            summary_a = json.loads((a / "studio_summary.json").read_text())
            self.assertEqual(summary_a["closed_loop_evaluations"][0]["eval_json"],
                             str(out / "eval.json"))
            # 缺摘要时不编造 finished / rc=0：run A 没有 lifecycle，也没有 done: 事件
            self.assertEqual(summary_a["status"], "incomplete_unobserved")
            self.assertIsNone(summary_a["return_code"])
            self.assertFalse((b / "studio_summary.json").exists())
            self.assertIn("run-a", app._eval_status_var.value)

    def test_start_training_b_during_eval_leaves_b_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, out = self._runs(d)

            def start_b(app):
                app.run_dir, app.proc = b, object()   # start_training 把 run_dir 换成新 run
                app.run_meta = {"run_id": "run-b", "config": {"epochs": 3}}
            self._finish_after(a, out, start_b)
            self.assertTrue((a / "studio_summary.json").is_file())
            self.assertFalse((b / "studio_summary.json").exists())
            life_b = json.loads((Path(str(b) + ".monitor") / "lifecycle.json").read_text())
            self.assertEqual(life_b["status"], "running")

    def test_auto_eval_note_goes_to_owner_run(self):
        with tempfile.TemporaryDirectory() as d:
            a, b, _ = self._runs(d)
            for run in (a, b):
                (run / "studio_summary.json").write_text(json.dumps(
                    {"run_id": run.name, "status": "finished", "notes": [], "metrics": {},
                     "config": {}, "artifacts": {}}))
            app = _headless_app(b)
            app._auto_eval_note("自动录像评测失败 rc=1", a)
            self.assertEqual(json.loads((a / "studio_summary.json").read_text())["notes"],
                             ["自动录像评测失败 rc=1"])
            self.assertEqual(json.loads((b / "studio_summary.json").read_text())["notes"], [])

    def test_eval_attached_mid_training_survives_training_finish(self):
        """新 bug：训练中途评测结束会写 finished/rc=0 摘要，训练结束又整份覆盖把评测抹掉。"""
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / "run"
            run.mkdir()
            side = Path(str(run) + ".monitor")
            side.mkdir()
            (side / "job.json").write_text(json.dumps({"run_id": "run"}))
            (side / "lifecycle.json").write_text(json.dumps({"status": "running", "pid": 1}))
            out = _eval_json(run / "closed_loop_eval_1" / "eval.json", episodes_requested=5).parent
            app = _headless_app(run)
            app._attach_eval_to_run(run.resolve(), out, json.loads((out / "eval.json").read_text()), 0)
            mid = json.loads((run / "studio_summary.json").read_text())
            self.assertEqual((mid["status"], mid["return_code"]), ("running", None))
            _patched_runs_root(self, Path(d) / "no-runs-root")
            app._finish_training(0)
            final = json.loads((run / "studio_summary.json").read_text())
            self.assertEqual(final["status"], "finished")
            self.assertEqual([e["eval_json"] for e in final["closed_loop_evaluations"]],
                             [str(out / "eval.json")])

    def test_recover_finalizes_running_summary_and_keeps_evals(self):
        with tempfile.TemporaryDirectory() as d:
            _patched_runs_root(self, d)
            run = Path(d) / "run"
            run.mkdir()
            side = Path(str(run) + ".monitor")
            side.mkdir()
            (side / "job.json").write_text(json.dumps({"run_id": "run"}))
            (side / "lifecycle.json").write_text(json.dumps({"status": "running",
                                                             "pid": 2 ** 31 - 2}))
            (run / "studio_summary.json").write_text(json.dumps(
                {"status": "running", "closed_loop_evaluations": [{"eval_json": "x"}]}))
            _headless_app()._recover_detached_runs()
            final = json.loads((run / "studio_summary.json").read_text())
            self.assertEqual(final["status"], "incomplete_unobserved")
            self.assertEqual(final["closed_loop_evaluations"], [{"eval_json": "x"}])


class StartingRaceTests(unittest.TestCase):
    """AUDIT #16：另一个 Studio 窗口不能把启动窗口里的 starting run 回收成失败。"""

    def test_recover_skips_starting_run_with_live_or_unknown_starter(self):
        with tempfile.TemporaryDirectory() as d:
            _patched_runs_root(self, d)
            cases = {"live": {"studio_pid": os.getpid()}, "legacy": {},
                     "dead": {"studio_pid": 2 ** 31 - 2}}
            for name, extra in cases.items():
                run = Path(d) / name
                run.mkdir()
                side = Path(str(run) + ".monitor")
                side.mkdir()
                (side / "job.json").write_text(json.dumps({"run_id": name}))
                (side / "lifecycle.json").write_text(json.dumps({"status": "starting", **extra}))
            _headless_app()._recover_detached_runs()
            self.assertEqual(os.listdir(Path(d) / "live"), [])
            self.assertEqual(os.listdir(Path(d) / "legacy"), [])
            self.assertTrue((Path(d) / "dead" / "studio_summary.json").is_file())


class DashboardHistoryTests(unittest.TestCase):
    def test_best_val_uses_full_history_beyond_tail_window(self):
        """REVIEW #6：末尾 1 MiB / 200 条读不到早期 val；首页 Best Val 要与全量一致。"""
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            lines = []
            for epoch in range(30):
                for step in range(120):
                    lines.append({"event": "train_step", "step": step, "pad": "x" * 300})
                f1 = 0.9 if epoch == 2 else 0.5 + epoch * 0.005
                lines.append({"event": "validation", "split": "val", "macro_f1": f1,
                              "action_accuracy": 0.8, "balanced_accuracy": 0.7})
            (run / "metrics.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
            self.assertGreater((run / "metrics.jsonl").stat().st_size, 1 << 20)
            app = _headless_app(run)
            app.refresh_run(force=True)
            self.assertEqual(len(app._val_events), 30)
            self.assertEqual(app.card_vars["dev_macro"].value, "0.900")


class EvalCheckpointTests(unittest.TestCase):
    def test_eval_of_last_ckpt_is_current_not_stale(self):
        """REVIEW #7：评末轮 ckpt.bin 的报告要和 ckpt.bin 比，不能拿 ckpt_best.bin 比。"""
        with tempfile.TemporaryDirectory() as d:
            run = Path(d)
            (run / "ckpt.bin").write_bytes(b"last epoch")
            (run / "ckpt_best.bin").write_bytes(b"best epoch")
            last = hashlib.sha256(b"last epoch").hexdigest()
            self.assertEqual(studio.eval_ckpt_state(run, {"ckpt": str(run / "ckpt.bin"),
                                                          "ckpt_sha256": last}), "current")
            self.assertEqual(studio.eval_ckpt_state(run, {"ckpt": str(run / "ckpt.bin"),
                                                          "ckpt_sha256": "0" * 64}), "stale")
            # 报告没写 ckpt 时退回 ckpt_best.bin；路径失效（数据盘搬家）时找 run 里同名文件
            best = hashlib.sha256(b"best epoch").hexdigest()
            self.assertEqual(studio.eval_ckpt_state(run, {"ckpt_sha256": best}), "current")
            self.assertEqual(studio.eval_ckpt_state(run, {"ckpt": "/moved/run/ckpt.bin",
                                                          "ckpt_sha256": last}), "current")

    def test_single_episode_recording_is_not_shown_as_sr(self):
        """REVIEW #8：训练后自动录像只有 1 集，大卡和 SR 列不能显示 1.000。"""
        with tempfile.TemporaryDirectory() as d:
            run = Path(d).resolve()
            video = _eval_json(run / "closed_loop_eval_1" / "eval.json", success_rate=1.0)
            app = _headless_app(run)
            app._eval_index = {run: [video]}
            app._update_sr_card()
            self.assertEqual(app.card_vars["closed_loop_sr"].value, "单集录像")
            self.assertIn("不是成功率", app.hero_sub["closed_loop_sr"].value)
            self.assertEqual(studio.sr_text([video]), "单集录像(1)")
            formal = _eval_json(run / "closed_loop_eval_2" / "eval.json", success_rate=0.55,
                                episodes_requested=20, episodes_done=20)
            app._eval_index = {run: [video, formal]}
            app._update_sr_card()
            self.assertEqual(app.card_vars["closed_loop_sr"].value, "0.550")
            self.assertIn("1 份单集录像未计入", app.hero_sub["closed_loop_sr"].value)
            self.assertEqual(studio.sr_text([video, formal]), "0.550")


class SimPageTests(unittest.TestCase):
    def test_pose_missing_zero_field_defaults_to_zero(self):
        """REVIEW #17：protobuf 转 JSON 省略 0 值字段，出生点 y=0 时没有 "y"。"""
        pose = studio.parse_pose_msg({"pose": [{"name": "x3", "position": {"x": 1.5, "z": 2.1},
                                                "orientation": {"w": 1.0}}]})
        self.assertIsNotNone(pose)
        self.assertEqual(pose[:3], (1.5, 0.0, 2.1))

    def test_trail_clears_when_sim_time_goes_back(self):
        """REVIEW #18：世界重置/重启后，新一集轨迹不能接到上一集末尾。"""
        app = _headless_app()
        app._sim_pose, app._sim_path, app._sim_last_pose_wall = None, deque(maxlen=3000), None
        for x, ns in ((0.0, 10), (0.5, 20), (0.5, 20), (1.0, 30)):
            app._ingest_pose((x, 0.0, 2.0, 0.0, ns))
        self.assertEqual(list(app._sim_path), [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0)])
        app._ingest_pose((5.0, 3.0, 2.0, 0.0, 5))
        self.assertEqual(list(app._sim_path), [(5.0, 3.0)])

    def test_map_note_names_sdf_and_marks_unverified(self):
        """REVIEW #18：底图写出 SDF 文件名；与 Studio 正在运行的世界不一致时标未核对。"""
        app = _headless_app()

        class Proc:
            def poll(self):
                return None
        world = Path("/w/scene_42.sdf")
        app._map_world, app._map_world_sha, app._map_source = world, "aaa", "SDF"
        app._sim_proc, app._sim_world_path, app._sim_world_sha256 = None, None, None
        text, _ = app._map_world_note()
        self.assertIn("scene_42.sdf", text)
        self.assertIn("未核对", text)
        app._sim_proc, app._sim_world_path, app._sim_world_sha256 = Proc(), world, "aaa"
        self.assertIn("已核对", app._map_world_note()[0])
        app._sim_world_sha256 = "bbb"
        self.assertIn("未核对", app._map_world_note()[0])
        app._map_world = None
        self.assertIn("未加载", app._map_world_note()[0])

    def _stubborn_gazebo(self, ignore):
        code = ("import signal, sys, time\n"
                + "".join(f"signal.signal(signal.{s}, signal.SIG_IGN)\n" for s in ignore)
                + "print('ready', flush=True)\ntime.sleep(60)\n")
        proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                start_new_session=True)
        self.assertEqual(proc.stdout.readline().strip(), b"ready")
        self.addCleanup(lambda: proc.poll() is None and os.killpg(proc.pid, signal.SIGKILL))
        return proc

    def test_stop_gazebo_escalates_sigint_sigterm_sigkill(self):
        """REVIEW #12：停止只发一次 SIGINT 时，忽略 SIGINT 的 gz sim 会一直留着。"""
        for ignore, want in ((("SIGINT",), -signal.SIGTERM),
                             (("SIGINT", "SIGTERM"), -signal.SIGKILL)):
            app = _headless_app()
            app._sim_stop, app._sim_proc = None, self._stubborn_gazebo(ignore)
            proc = app._sim_proc
            app._stop_gazebo_blocking(grace_s=0.3)
            self.assertEqual(proc.wait(timeout=5), want)
            self.assertIsNone(app._sim_stop)
            proc.stdout.close()

    def test_stream_status_does_not_overwrite_event_message(self):
        """REVIEW #12：『Gazebo 已退出 rc=…』不能在下一个 80 ms 被流状态盖掉。"""
        class Sub:
            def frames(self):
                return []

            def latest(self):
                return {"pose": [{"name": "x3", "position": {"x": 1.0, "y": 0.0, "z": 2.0},
                                  "orientation": {"w": 1.0},
                                  "header": {"stamp": {"sec": 1, "nsec": 0}}}]}

            def healthy(self):
                return True

        class Exited:
            returncode = 1

            def poll(self):
                return 1
        app = _headless_app()
        app._closing = True
        app._sim_subs = {k: Sub() for k in ("fpv", "chase", "top", "pose")}
        app._sim_pose, app._sim_path, app._sim_last_pose_wall = None, deque(maxlen=3000), None
        app._sim_frame_stamps = {"fpv": None, "chase": None, "top": None}
        app._cam_status_vars = {k: _Var() for k in ("fpv", "chase", "top")}
        app._live_proc = app._scene_proc = app._sim_log_handle = app._sim_stop = None
        app._sim_proc, app._sim_world_path, app._sim_world_sha256 = Exited(), None, None
        app._sim_tick()
        self.assertEqual(app._sim_status_var.value, "Gazebo 已退出 rc=1")
        app._sim_tick()
        self.assertEqual(app._sim_status_var.value, "Gazebo 已退出 rc=1")
        self.assertIn("pose:OK", app._sim_stream_var.value)

    def test_delayed_connect_skipped_when_gazebo_already_exited(self):
        """新 bug：Gazebo 在 3.5 s 内退出（坏 SDF），延迟的 connect_sim 仍会订阅并盖掉退出原因。"""
        app = _headless_app()
        calls = []
        app.connect_sim = lambda: calls.append(1)

        class Exited:
            def poll(self):
                return 1
        proc = Exited()
        app._sim_proc = proc
        app._connect_after_start(proc)
        app._sim_proc = None
        app._connect_after_start(proc)
        self.assertEqual(calls, [])

    def test_failed_gazebo_start_restores_partition(self):
        """新 bug：start_gazebo 准备环境失败后 GZ_PARTITION 仍被锁在本次分区，之后连接用错分区。"""
        with tempfile.TemporaryDirectory() as d:
            world = Path(d) / "w.sdf"
            world.write_text("<sdf/>")
            app = _headless_app()
            app._scene_proc = app._sim_proc = app._sim_stop = None
            app._sim_partition_active = None
            app.world_var, app.partition_var = _Var(str(world)), _Var("")
            app.render_backend_var = _Var("bogus")   # collect_parallel.gz_env 会抛 ValueError
            old_env = {k: os.environ.get(k) for k in ("GZ_PARTITION", "UAVRGB_RENDER_BACKEND",
                                                      "GZ_SIM_RESOURCE_PATH")}
            old_box = studio.messagebox
            studio.messagebox = type("Box", (), {"showerror": staticmethod(lambda *a: None)})
            try:
                app.start_gazebo()
                partition_after = os.environ.get("GZ_PARTITION")
            finally:
                studio.messagebox = old_box
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            self.assertIsNone(app._sim_partition_active)
            self.assertEqual(partition_after, old_env["GZ_PARTITION"])
            self.assertIn("环境准备失败", app._sim_status_var.value)


WORLD_X3 = studio.REPO_ROOT / "sim" / "worlds" / "uav_rgb.sdf"
WORLD_E1 = studio.REPO_ROOT / "sim" / "worlds" / "uav_rgb_e1.sdf"
CF_META = {"task_version": "l1.v4", "splits": {"train": 9, "val": 0, "test": 2},
           "dynamics_profile": "cf_stab_v1", "window_from": "observation", "hold_s": 0.2}


class VehicleTests(unittest.TestCase):
    """新载具：载具名从世界 SDF 推断，位姿话题/障碍物过滤/Pose_V 选择都跟着走，不再写死 x3。"""

    def test_vehicle_from_sdf_and_pose_topic(self):
        self.assertEqual(studio.vehicle_from_sdf(WORLD_X3), "x3")
        self.assertEqual(studio.vehicle_from_sdf(WORLD_E1), "e1_mini")
        self.assertEqual(studio.vehicle_from_sdf("/nonexistent/world.sdf"), "x3")
        self.assertEqual(studio.vehicle_from_sdf(""), "x3")
        self.assertEqual(studio.pose_topic("e1_mini"), "/model/e1_mini/pose")

    def test_parse_sdf_obstacles_excludes_inferred_vehicle(self):
        # uav_rgb_e1.sdf 的注释含 "--"，expat 严格解析失败；去注释重试后仍要拿到障碍物
        with self.assertRaises(ET.ParseError):
            ET.parse(str(WORLD_E1))
        names = {o["name"] for o in studio.parse_sdf_obstacles(WORLD_E1)}
        self.assertIn("pillar_center", names)
        self.assertNotIn("e1_mini", names, "e1_mini 机体不能被画成障碍物")
        sdf = '''<sdf version="1.6"><world name="w">
          <model name="e1_mini"><link name="base_link">
            <collision name="c"><geometry><box><size>0.2 0.2 0.1</size></box></geometry></collision>
            <sensor name="camera" type="camera"/></link></model>
          <model name="pillar_0"><link name="l">
            <collision name="c"><geometry><box><size>0.4 0.4 2</size></box></geometry></collision></link></model>
        </world></sdf>'''
        with tempfile.NamedTemporaryFile("w", suffix=".sdf", delete=False) as f:
            f.write(sdf)
        self.addCleanup(os.unlink, f.name)
        self.assertEqual([o["name"] for o in studio.parse_sdf_obstacles(f.name)], ["pillar_0"])

    def test_parse_pose_v_prefers_named_vehicle(self):
        msg = {"pose": [{"name": "pillar_0", "position": {"x": 1.5, "z": 1.5}, "orientation": {"w": 1}},
                        {"name": "e1_mini", "position": {"x": 0.3, "y": 0.2, "z": 2.0},
                         "orientation": {"w": 1}}]}
        self.assertEqual(studio.parse_pose_msg(msg, "e1_mini")[:3], (0.3, 0.2, 2.0))
        # 没有同名条目时退回第一条（旧行为），不报错
        self.assertEqual(studio.parse_pose_msg(msg)[:3], (1.5, 0.0, 1.5))

    def test_connect_sim_subscribes_vehicle_pose_topic(self):
        topics = []

        class Sub:
            def __init__(self, topic, **_kwargs):
                topics.append(topic)

            def close(self):
                pass
        old_sub = studio.gz_io.TopicSub
        studio.gz_io.TopicSub = Sub
        self.addCleanup(setattr, studio.gz_io, "TopicSub", old_sub)
        old_part = os.environ.get("GZ_PARTITION")
        self.addCleanup(_restore_env, "GZ_PARTITION", old_part)
        app = _headless_app()
        app._sim_subs, app._sim_partition_active = {}, None
        app.partition_var, app.meta_var = _Var(""), _Var("")
        app.world_var = _Var(str(WORLD_E1))
        app._sim_path, app._sim_pose, app._sim_last_pose_wall = deque(maxlen=10), None, None
        app.connect_sim()
        self.assertEqual(app._vehicle, "e1_mini")
        self.assertIn("/model/e1_mini/pose", topics)
        self.assertNotIn("/model/x3/pose", topics)


class ProtocolTests(unittest.TestCase):
    """N2：手动闭环评测 / live demo 也按训练数据的执行协议传 --dynamics/--window-from/--hold。"""

    def _eval_app(self, d, meta=CF_META, task="l1"):
        run, data = _training_run(d, meta)
        app = _headless_app(run)
        app.eval_episodes_var, app.eval_task_var = _Var("5"), _Var(task)
        app.eval_split_var, app.eval_appearance_var = _Var("val"), _Var("v1")
        app._sim_proc, app._sim_subs = None, {}
        self.addCleanup(_restore_env, "UAVRGB_RENDER_BACKEND", os.environ.get("UAVRGB_RENDER_BACKEND"))
        return app, data

    def test_run_eval_protocol_reads_the_runs_own_job_json(self):
        with tempfile.TemporaryDirectory() as d:
            run, data = _training_run(d, CF_META)
            params, reason = studio.run_eval_protocol(run)
            self.assertIsNone(reason)
            self.assertEqual((params["task"], params["dynamics"], params["window_from"],
                              params["hold_s"], params["data"]),
                             ("l1", "cf_stab", "observation", 0.2, str(data)))
            bare = Path(d) / "bare"
            bare.mkdir()
            self.assertEqual(studio.run_eval_protocol(bare), (None, "run 配置里没有训练数据路径"))

    def test_manual_eval_passes_training_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            app, _ = self._eval_app(d)
            box = _patched_messagebox(self)
            calls = _patched_popen(self)
            self.assertTrue(app.start_closed_loop_eval())
            cmd, kwargs = calls[0]
            self.assertEqual(cmd[cmd.index("--dynamics") + 1], "cf_stab")
            self.assertEqual(cmd[cmd.index("--window-from") + 1], "observation")
            self.assertEqual(cmd[cmd.index("--hold") + 1], "0.2")
            self.assertEqual(cmd[cmd.index("--episodes") + 1], "5")
            self.assertEqual(kwargs["env"]["UAVRGB_RENDER_BACKEND"], studio.DEFAULT_RENDER_BACKEND)
            self.assertEqual(box, [], "协议一致时不弹任何框")

    def test_manual_eval_asks_on_task_mismatch_and_unknown_protocol(self):
        with tempfile.TemporaryDirectory() as d:
            app, _ = self._eval_app(d, task="base")
            box = _patched_messagebox(self, **{"Task 与训练数据不一致": False})
            calls = _patched_popen(self)
            self.assertFalse(app.start_closed_loop_eval())
            self.assertEqual(box, [("askyesno", "Task 与训练数据不一致")])
            self.assertEqual(calls, [])
        with tempfile.TemporaryDirectory() as d:
            app, _ = self._eval_app(d, meta={"task_version": "l1.v4", "splits": {"test": 1},
                                             "dynamics_profile": "unknown_v9"})
            box = _patched_messagebox(self, **{"无法推断执行协议": True})
            calls = _patched_popen(self)
            self.assertTrue(app.start_closed_loop_eval())
            self.assertEqual(box[0][1], "无法推断执行协议")
            self.assertNotIn("--dynamics", calls[0][0], "推断不到就按 eval_closed_loop 默认值")

    def test_manual_eval_refuses_3dgs_training_data(self):
        with tempfile.TemporaryDirectory() as d:
            app, _ = self._eval_app(d, meta=dict(CF_META, renderer_backend="gsplat"))
            box = _patched_messagebox(self)
            calls = _patched_popen(self)
            self.assertFalse(app.start_closed_loop_eval())
            self.assertEqual(box, [("showerror", "数据来自 3DGS")])
            self.assertEqual(calls, [])

    def test_live_ckpt_prefills_hold_and_demo_passes_dynamics(self):
        with tempfile.TemporaryDirectory() as d:
            run, _ = _training_run(d, CF_META)
            scene_meta = Path(d) / "scene_1.json"
            scene_meta.write_text(json.dumps({"seed": 1, "split": "test"}), encoding="utf-8")
            app = _headless_app(run)
            app.live_hold_var, app.live_steps_var = _Var("1.0"), _Var("80")
            app._set_live_ckpt(run / "ckpt_best.bin")
            self.assertEqual(app.live_hold_var.value, "0.2", "选中权重后 Hold 预填训练数据的 hold_s")
            app.meta_var, app.world_var = _Var(str(scene_meta)), _Var(str(WORLD_X3))
            app._live_proc, app._sim_proc, app._sim_subs = None, None, {"pose": object()}
            app._sim_partition_active, app._sim_path = None, deque(maxlen=10)
            app.live_action_var = _Var()
            app.live_btn, app.live_stop_btn = _Widget(), _Widget()
            _patched_messagebox(self)   # 外部仿真 World 无法核验 → 继续
            calls = _patched_popen(self)
            app.start_live_student()
            self.assertEqual(len(calls), 1)
            cmd = calls[0][0]
            self.assertEqual(cmd[cmd.index("--dynamics") + 1], "cf_stab")
            self.assertEqual(cmd[cmd.index("--window-from") + 1], "observation")
            self.assertEqual(cmd[cmd.index("--hold") + 1], "0.2")
            self.assertEqual(app._live_config["dynamics"], "cf_stab")
            app._live_log.close()


class ViewThrottleTests(unittest.TestCase):
    def test_chase_and_top_limited_to_10hz_fpv_every_frame(self):
        """REVIEW #9：chase 15 Hz 每帧缩放拖慢主线程；chase/顶视限 10 Hz，机载画面（模型输入）不限。"""
        class Sub:
            seq = 0

            def frames(self):
                Sub.seq += 1
                return [(Sub.seq, {"width": 2, "height": 2})]

            def latest(self):
                return None

            def healthy(self):
                return True
        shown = []
        app = _headless_app()
        app._closing = True
        app._show_camera_frame = lambda key, _msg: shown.append(key)
        app._sim_subs = {k: Sub() for k in ("fpv", "chase", "top", "pose")}
        app._sim_pose, app._sim_path, app._sim_last_pose_wall = None, deque(maxlen=10), None
        app._sim_frame_stamps = {"fpv": None, "chase": None, "top": None}
        app._cam_status_vars = {k: _Var() for k in ("fpv", "chase", "top")}
        app._live_proc = app._scene_proc = app._sim_proc = app._sim_stop = None
        clock = [100.0]
        old_mono = studio.time.monotonic
        studio.time.monotonic = lambda: clock[0]
        self.addCleanup(setattr, studio.time, "monotonic", old_mono)
        for _ in range(13):            # 1 s 内 13 个 tick（80 ms 一个）
            app._sim_tick()
            clock[0] += studio.SIM_TICK_MS / 1000
        counts = {k: shown.count(k) for k in ("fpv", "chase", "top")}
        self.assertEqual(counts["fpv"], 13, counts)
        self.assertLessEqual(counts["chase"], 10, counts)
        self.assertGreaterEqual(counts["chase"], 6, counts)
        self.assertEqual(counts["chase"], counts["top"], counts)


class RenderBackendTests(unittest.TestCase):
    def test_start_gazebo_uses_selected_render_backend(self):
        """本容器 gpu 后端段错误：Studio 启动 Gazebo 必须能选 software，并透传给 gz_env。"""
        keys = ("GZ_PARTITION", "UAVRGB_RENDER_BACKEND", "GZ_SIM_RESOURCE_PATH", "XDG_RUNTIME_DIR")
        for k in keys:
            self.addCleanup(_restore_env, k, os.environ.get(k))
        with tempfile.TemporaryDirectory() as d:
            _patched_runs_root(self, d)
            calls = _patched_popen(self)
            for backend, headless in (("software", False), ("gpu", True)):
                app = _headless_app()
                app.root = type("Root", (), {"after": lambda self, *_a: None})()
                app._scene_proc = app._sim_proc = app._sim_stop = app._sim_log_handle = None
                app._sim_partition_active = None
                app.world_var, app.meta_var, app.partition_var = _Var(str(WORLD_X3)), _Var(""), _Var("")
                app.render_backend_var = _Var(backend)
                app.start_gazebo()
                cmd, kwargs = calls[-1]
                self.assertEqual(kwargs["env"]["UAVRGB_RENDER_BACKEND"], backend)
                self.assertEqual("--headless-rendering" in cmd, headless, cmd)
                self.assertIn(f"渲染={backend}", app._sim_status_var.value)
                app._sim_log_handle.close()
                app._restore_partition()
        self.assertEqual(studio.DEFAULT_RENDER_BACKEND, "software")


class HistoryScanTests(unittest.TestCase):
    def test_refresh_history_scans_off_main_thread(self):
        """REVIEW #11：扫描放后台线程；主线程只排一次 after，表格在主线程回调里填。"""
        class Root:
            def __init__(self):
                self.pending = []

            def after(self, _ms, fn):
                self.pending.append(fn)

        class Tree:
            def __init__(self):
                self.rows = []

            def get_children(self):
                return list(self.rows)

            def delete(self, *items):
                self.rows = [r for r in self.rows if r not in items]

            def insert(self, _parent, _where, iid, values):
                self.rows.append(iid)
        with tempfile.TemporaryDirectory() as d:
            data = Path(d)
            run = data / "uav_rgb" / "run-a"
            run.mkdir(parents=True)
            (run / "results.csv").write_text("epoch,train_ce\n1,1.0\n")
            old = studio.DATA_ROOT
            studio.DATA_ROOT = data
            self.addCleanup(setattr, studio, "DATA_ROOT", old)
            _patched_runs_root(self, data / "uav_rgb" / "studio_runs")
            app = _headless_app()
            app.root, app.history_tree = Root(), Tree()
            app._history_thread, app._history_again = None, False
            app._scan_sig = app._scan_cache = None
            gate = threading.Event()
            real_scan = app._scan_runs
            app._scan_runs = lambda force=False: (gate.wait(5), real_scan(force))[1]
            app.refresh_history(force=True)
            self.assertEqual(app.history_tree.rows, [])     # 主线程没被扫描阻塞
            app.refresh_history(force=True)                 # 扫描中再请求：只记一笔
            self.assertTrue(app._history_again)
            gate.set()
            app._history_thread.join(5)
            app.root.pending.pop(0)()
            self.assertEqual(app.history_tree.rows, [str(run)])
            self.assertIsNotNone(app._history_thread)       # 补扫已开始
            app._history_thread.join(5)

    def test_eval_walk_prunes_venv_and_episode_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            keep = _eval_json(root / "group" / "eval-1" / "eval.json")
            for decoy in ("venv/lib/site-packages/pkg", "group/episode_x/sub", ".hidden/e"):
                _eval_json(root / decoy / "eval.json")
            (root / "venv" / "pyvenv.cfg").write_text("home = /usr\n")
            (root / "group" / "episode_x" / "episode_manifest.json").write_text("{}")
            self.assertEqual(studio._walk_eval_json(root), [keep])


@unittest.skipIf(studio.tk is None or not os.environ.get("DISPLAY"), "需要 tkinter + DISPLAY")
class StudioGuiTests(unittest.TestCase):
    """真实 Tk 窗口（DISPLAY=:20）。DATA_ROOT/RUNS_ROOT 指向临时目录，不扫真实数据盘。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        old_data, old_runs = studio.DATA_ROOT, studio.RUNS_ROOT
        studio.DATA_ROOT = Path(self.tmp.name)
        studio.RUNS_ROOT = studio.DATA_ROOT / "uav_rgb" / "studio_runs"
        self.addCleanup(setattr, studio, "DATA_ROOT", old_data)
        self.addCleanup(setattr, studio, "RUNS_ROOT", old_runs)
        self.root = studio.tk.Tk()
        self.app = studio.StudioApp(self.root)
        self.addCleanup(self._destroy)

    def _destroy(self):
        if not self.app._closing:
            self.app._closing = True
            self.root.destroy()

    def _pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end and not self.app._closing:
            self.root.update()
            time.sleep(0.01)

    def test_sim_panels_keep_weight_ratio(self):
        """REVIEW #10：顶视和地图不能被压扁；格子按权重 3:2、两列等宽。"""
        import base64
        self.root.geometry("1500x960")
        self.app.tabs.select(self.app.sim_tab)
        msgs = {k: {"width": w, "height": h, "step": w * 3, "pixelFormatType": "RGB_INT8",
                    "data": base64.b64encode(bytes(range(256)) * (w * h * 3 // 256)).decode()}
                for k, (w, h) in {"chase": (640, 360), "fpv": (320, 240), "top": (480, 360)}.items()}
        for _ in range(5):
            for key, msg in msgs.items():
                self.app._show_camera_frame(key, msg)
            self._pump(0.1)
        self.assertEqual(set(self.app._sim_photos), {"chase", "fpv", "top"})
        chase, top = self.app.chase_panel, self.app.topcam_panel
        ratio = chase.winfo_height() / top.winfo_height()
        self.assertAlmostEqual(ratio, 1.5, delta=0.15)
        self.assertEqual(chase.winfo_width(), self.app.fpv_panel.winfo_width())
        self.assertGreater(self.app.top_canvas.winfo_height(), 60)

    def test_map_reuses_static_layer_and_labels_sdf(self):
        """REVIEW #11/#18：轨迹每帧只改坐标，不重建；地图写 SDF 文件名并标未核对。"""
        self.app.tabs.select(self.app.sim_tab)
        self.app._load_world_geometry()
        self._pump(0.3)
        app, c = self.app, self.app.top_canvas
        app._sim_path.extend((0.002 * i, 0.0) for i in range(3000))
        app._sim_pose = (6.0, 0.0, 2.0, 0.0, 1)
        app._draw_topdown()
        items = c.find_all()
        app._sim_path.append((6.001, 0.0))
        app._sim_pose = (6.001, 0.0, 2.0, 0.0, 2)
        app._draw_topdown()
        self.assertEqual(c.find_all(), items)
        self.assertEqual(len(c.coords("map_path")), 2 * len(app._sim_path))
        note = c.itemcget("map_world", "text")
        self.assertIn("uav_rgb.sdf", note)
        self.assertIn("未核对", note)

    def test_sigint_closes_window_and_stops_owned_gazebo(self):
        """REVIEW #12：终端 Ctrl+C 走 _on_close（不弹框），收掉 Studio 启动的 Gazebo 进程组。"""
        code = ("import signal, time\nsignal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                "print('ready', flush=True)\ntime.sleep(60)\n")
        child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                                 start_new_session=True)
        self.assertEqual(child.stdout.readline().strip(), b"ready")
        self.addCleanup(child.stdout.close)
        self.addCleanup(lambda: child.poll() is None and os.killpg(child.pid, signal.SIGKILL))
        old = {s: signal.getsignal(s) for s in (signal.SIGINT, signal.SIGTERM)}
        for s, h in old.items():
            self.addCleanup(signal.signal, s, h)
        self.app._sim_proc = child
        studio.install_quit_signals(self.app)
        os.kill(os.getpid(), signal.SIGINT)
        self._pump(10)
        self.assertTrue(self.app._closing)
        self.assertEqual(child.wait(timeout=5), -signal.SIGTERM)


if __name__ == "__main__":
    unittest.main(verbosity=2)
