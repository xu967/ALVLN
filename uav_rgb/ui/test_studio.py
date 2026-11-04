#!/usr/bin/env python3
"""Headless regression tests for UAV Navigation Studio data/geometry helpers."""
import hashlib
import json
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import studio


def _write_uavr(path, meta):
    """最小 UAVR：48 B 头 + meta JSON（0 样本，只测尾部 meta 读取）。"""
    raw = json.dumps(meta).encode("utf-8")
    header = struct.pack("<4sHHIHHBBBBQQQI", b"UAVR", 1, 48, 0, 120, 160, 3,
                         1, 1, 0, 48, 48, 48, len(raw))
    Path(path).write_bytes(header + raw)


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
            calls = []
            old_popen = studio.subprocess.Popen
            studio.subprocess.Popen = lambda cmd, **kwargs: calls.append(cmd) or Proc()
            try:
                started = app.start_closed_loop_eval(auto={
                    "task": "campus", "split": "test", "dynamics": "cf_stab",
                    "episodes": 1, "window_from": "dispatch", "hold_s": 0.2})
            finally:
                studio.subprocess.Popen = old_popen
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
