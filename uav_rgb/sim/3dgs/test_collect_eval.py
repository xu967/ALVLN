#!/usr/bin/env python3
"""collect_3dgs / eval_3dgs 端到端小测（mock renderer，不需要 torch/学生权重）。

用 3dgs-venv 或 .venv-tools 跑（要 numpy；视频那一项要 cv2，打包那一项要 PIL）：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_collect_eval.py
没设 TMPDIR 时临时目录落 $DATA_ROOT/uav_rgb/tmp（根分区很小，不往 /tmp 写）。
"""
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sim", "tools"))
scene_mod = importlib.import_module("sim.3dgs.scene")
collect = importlib.import_module("sim.3dgs.collect_3dgs")
eval_mod = importlib.import_module("sim.3dgs.eval_3dgs")
import split_manifest as sm  # noqa: E402

_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP
    os.environ["TMPDIR"] = _DATA_TMP   # 子进程也不要写 /tmp


def _run(args, env=None):
    return subprocess.run([sys.executable, "-m"] + args, cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=180)


def _run_script(path, args):
    return subprocess.run([sys.executable, path] + args, cwd=ROOT,
                          capture_output=True, text=True, timeout=180)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect(scene_json, out, *extra):
    return _run(["sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", out,
                 "--steps", "3", "--mock-renderer", *extra])


def _eval(scene_json, out, *extra):
    return _run(["sim.3dgs.eval_3dgs", "--scene", scene_json, "--out", out,
                 "--policy", "forward", "--mock-renderer", "--episodes", "1",
                 "--steps", "3", *extra])


class TestCollect3dgs(unittest.TestCase):
    def test_forward_mock_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            out = os.path.join(tmp, "ep")
            r = _collect(os.path.join(scene_dir, "scene.json"), out,
                         "--policy", "forward", "--save-depth")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load(os.path.join(out, "ep_00000", "episode_manifest.json"))
            self.assertEqual(m["renderer"]["backend"], "mock")
            self.assertEqual(m["num_steps"], 3)
            self.assertEqual(m["dynamics_params"]["z_target_m"], 2.0)
            self.assertEqual(m["dynamics_profile"], m["dynamics_params"]["profile"])
            self.assertTrue(m["validity"]["episode_executed"])
            with open(os.path.join(out, "ep_00000", "actions.jsonl")) as f:
                rows = [json.loads(x) for x in f]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(r["action_executed"] == 1 for r in rows))
            self.assertTrue(all(r["pose_gt"]["z"] == 2.0 for r in rows))
            self.assertTrue(all("collision_point" in r for r in rows))
            for i in range(3):
                png = os.path.join(out, "ep_00000", "rgb_%05d.png" % i)
                with open(png, "rb") as f:
                    self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")
                self.assertTrue(os.path.isfile(
                    os.path.join(out, "ep_00000", "depth", "depth_%05d.npy" % i)))
            run = _load(os.path.join(out, "run.json"))
            self.assertEqual(run["dynamics_params"]["z_target_m"], 2.0)
            # AUDIT #1：身份来自 PLY 指纹，split 由几何组哈希决定并可按 manifest 复算
            self.assertEqual(m["split_policy"], "geometry_gid")
            self.assertEqual(m["scene_id"], int(m["geometry_gid"][:8], 16))
            self.assertEqual(m["split"], sm.split_for_gid(m["geometry_gid"], m["split_salt"]))
            self.assertEqual(m["split"], sm.split_for_scene(m))
            self.assertEqual(run["scene"]["geometry_gid"], m["geometry_gid"])
            self.assertEqual(run["split"], m["split"])
            self.assertEqual(m["collision_proxy"]["proxy_version"], "3dgs.occ.v2")

    def test_route_select_and_astar_teacher(self):
        """--route N 覆盖 spawn/waypoint/max_steps；astar 教师朝 +y 出生、目标在 +x 时先右转；越界报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene = scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene["routes"] = [
                {"name": "r0", "spawn": scene["spawn"], "waypoint": scene["waypoint"]},
                {"name": "r1", "spawn": {"x": 3.0, "y": 0.0, "z": 2.0, "yaw_deg": 90.0},
                 "waypoint": {"x": 10.0, "y": 0.0, "z": 2.0}, "max_steps": 4}]
            scene_json = os.path.join(scene_dir, "scene.json")
            scene_mod.save_scene({k: v for k, v in scene.items() if not k.startswith("_") and k not in ("path", "dir", "ply_path")}, scene_json)
            out = os.path.join(tmp, "r1")
            r = _run(["sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", out, "--mock-renderer",
                      "--policy", "astar", "--route", "1"])
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load(os.path.join(out, "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["route"]["index"], m["route"]["name"]), (1, "r1"))
            self.assertEqual(m["scene"]["spawn"]["yaw_deg"], 90.0)
            self.assertEqual(m["waypoint"]["x"], 10.0)
            self.assertEqual(m["requested_steps"], 4)            # 没给 --steps 时用 routes[1].max_steps
            self.assertEqual((m["teacher"], m["planner"], m["stop_radius"]), ("plan_action", "astar_occ", 0.5))
            self.assertEqual(m["teacher_version"], "uavrgb.teacher_occ.v1")
            with open(os.path.join(out, "ep_00000", "actions.jsonl")) as f:
                rows = [json.loads(x) for x in f]
            self.assertEqual(rows[0]["action_executed"], 3)      # YAW_RIGHT：航向 +y，目标在 +x
            # 默认 route（None）= 顶层 spawn/waypoint = routes[0]；task_id 与 route 1 不同
            r0 = _run(["sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", os.path.join(tmp, "r0"),
                       "--mock-renderer", "--policy", "forward", "--steps", "2"])
            self.assertEqual(r0.returncode, 0, r0.stderr[-2000:])
            m0 = _load(os.path.join(tmp, "r0", "ep_00000", "episode_manifest.json"))
            self.assertEqual(m0["route"]["index"], 0)
            self.assertEqual(m0["scene"]["spawn"], scene["spawn"])
            self.assertNotEqual(m0["task_id"], m["task_id"])
            self.assertEqual(m0["geometry_gid"], m["geometry_gid"])
            bad = _run(["sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", os.path.join(tmp, "bad"),
                        "--mock-renderer", "--route", "2", "--steps", "1"])
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("route 2", bad.stderr)


class TestSceneIdentity(unittest.TestCase):
    """AUDIT #1：以前 scene_id=seed、split 由命令行随便给，同一 PLY 可同时进 train/val。"""

    def test_gid_independent_of_seed_and_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene_json = os.path.join(scene_dir, "scene.json")
            r0 = _collect(scene_json, os.path.join(tmp, "a"), "--policy", "forward", "--seed", "0")
            r7 = _collect(scene_json, os.path.join(tmp, "b"), "--policy", "forward", "--seed", "7")
            self.assertEqual((r0.returncode, r7.returncode), (0, 0), r0.stderr + r7.stderr)
            ma = _load(os.path.join(tmp, "a", "ep_00000", "episode_manifest.json"))
            mb = _load(os.path.join(tmp, "b", "ep_00000", "episode_manifest.json"))
            self.assertNotEqual(ma["seed"], mb["seed"])
            self.assertEqual(ma["geometry_gid"], mb["geometry_gid"])
            self.assertEqual(ma["scene_id"], mb["scene_id"])
            self.assertEqual(ma["split"], mb["split"])
            self.assertNotEqual(ma["scene_id"], ma["seed"])
            # 同 PLY 换 waypoint：几何组不变，任务 id 变
            scene = scene_mod.load_scene(scene_json)
            base = collect.scene_identity(scene)
            scene["waypoint"] = dict(scene["waypoint"], x=scene["waypoint"]["x"] + 1.0)
            moved = collect.scene_identity(scene)
            self.assertEqual(moved["geometry_gid"], base["geometry_gid"])
            self.assertNotEqual(moved["task_id"], base["task_id"])
            # 另一份 PLY -> 另一个几何组
            other_dir = os.path.join(tmp, "scene2")
            scene_mod.make_corridor(other_dir, seed=3, length=12.0)
            other = collect.scene_identity(scene_mod.load_scene(os.path.join(other_dir, "scene.json")))
            self.assertNotEqual(other["geometry_gid"], base["geometry_gid"])

    def test_forced_split_is_marked_diagnostic(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene_json = os.path.join(scene_dir, "scene.json")
            want = collect.scene_identity(scene_mod.load_scene(scene_json))["split"]
            other = "val" if want != "val" else "test"
            r = _collect(scene_json, os.path.join(tmp, "x"), "--policy", "forward", "--split", other)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn("diag-manual-split", r.stderr)
            m = _load(os.path.join(tmp, "x", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["split"], m["split_policy"]), (other, "diag-manual-split"))
            self.assertTrue(sm.is_diagnostic_scheme(m["split_policy"]))
            # 直接调 run_episode 给不一致的 split 而不声明策略 -> 拒绝
            scene = scene_mod.load_scene(scene_json)
            renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(scene, True)
            occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)
            params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
            with self.assertRaises(ValueError):
                collect.run_episode(scene, os.path.join(tmp, "y"), renderer, occ, params,
                                    collect.ForwardPolicy(), 1, 0.5, split=other)

    def test_preprocess_rejects_same_scene_in_two_splits(self):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("无 PIL，不跑 preprocess")
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene_json = os.path.join(scene_dir, "scene.json")
            out = os.path.join(tmp, "c")
            r = _collect(scene_json, out, "--policy", "teacher", "--episodes", "2")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            eps = [os.path.join(out, "ep_%05d" % i) for i in range(2)]
            pre = os.path.join(ROOT, "tools", "preprocess.py")
            ok = _run_script(pre, ["--out", os.path.join(tmp, "ds_ok")] + eps)
            self.assertEqual(ok.returncode, 0, ok.stderr[-2000:] + ok.stdout[-1000:])
            meta = _load(os.path.join(tmp, "ds_ok", "meta.json")) if os.path.isfile(
                os.path.join(tmp, "ds_ok", "meta.json")) else None
            if meta is not None:
                self.assertEqual(meta.get("split_policy"), "geometry_gid")
            # 把第二集的 split 改掉：同一 geometry_gid 跨 split，按记录策略复算也对不上 -> 拒绝
            man_path = os.path.join(eps[1], "episode_manifest.json")
            man = _load(man_path)
            man["split"] = "val" if man["split"] != "val" else "test"
            with open(man_path, "w", encoding="utf-8") as f:
                json.dump(man, f)
            bad = _run_script(pre, ["--out", os.path.join(tmp, "ds_bad")] + eps)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("不一致", bad.stderr + bad.stdout)


class TestEval3dgs(unittest.TestCase):
    def test_forward_mock_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=3, length=12.0)
            out = os.path.join(tmp, "eval")
            r = _eval(os.path.join(scene_dir, "scene.json"), out, "--hold", "0.5")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            e = _load(os.path.join(out, "eval.json"))
            self.assertEqual(e["episodes_done"], 1)
            self.assertIn("success_rate", e)
            self.assertIn("collision_rate", e)
            self.assertEqual(e["metric_version"], "3dgs.pilot.v2")
            self.assertEqual(e["success_criterion"], eval_mod.SUCCESS_CRITERION)
            self.assertEqual(e["renderer"]["backend"], "mock")
            self.assertEqual(len(e["per_episode"]), 1)
            self.assertEqual((e["hold_s"], e["hold_source"]), (0.5, "cli --hold"))
            self.assertEqual(e["failed"], [])
            # 评测产物的身份：gid 来自 PLY，split 固定 test 且标成诊断策略，打包端默认拒绝
            m = _load(os.path.join(out, "ep_00000", "episode_manifest.json"))
            self.assertEqual(m["geometry_gid"], e["scene"]["geometry_gid"])
            self.assertEqual((m["split"], m["split_policy"]), ("test", "diag-eval-fixed-test"))
            self.assertTrue(sm.is_diagnostic_scheme(m["split_policy"]))

    def test_video_fps_follows_hold_and_reports_task_identity(self):
        try:
            import cv2
        except ImportError:
            self.skipTest("无 cv2")
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=3, length=12.0)
            scene_json = os.path.join(scene_dir, "scene.json")
            out = os.path.join(tmp, "eval")
            r = _run(["sim.3dgs.eval_3dgs", "--scene", scene_json, "--out", out,
                      "--policy", "forward", "--mock-renderer", "--episodes", "1",
                      "--steps", "4", "--hold", "0.2", "--record-video"])
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            e = _load(os.path.join(out, "eval.json"))
            self.assertEqual((e["hold_s"], e["hold_source"]), (0.2, "cli --hold"))
            ep = e["per_episode"][0]
            self.assertNotIn("video_error", ep)
            # 每帧代表一个 hold 窗口：fps=1/hold，仿真时间 1:1（以前固定 5 fps）
            self.assertAlmostEqual(ep["video_fps"], 5.0)
            self.assertTrue(ep["video"].endswith(".webm"), ep["video"])
            path = os.path.join(out, ep["video"])
            with open(path, "rb") as f:
                self.assertIn(b"V_VP8", f.read(4096))
            cap = cv2.VideoCapture(path)
            back = 0
            while cap.read()[0]:
                back += 1
            cap.release()
            self.assertEqual((ep["video_frames"], back), (4, 4))
            raw = _load(scene_json)
            with open(scene_json, "rb") as f:
                want_sha = hashlib.sha256(f.read()).hexdigest()
            self.assertEqual(e["scene"]["scene_json_sha256"], want_sha)
            self.assertEqual(e["scene"]["waypoint"], raw["waypoint"])
            self.assertEqual(e["scene"]["spawn"], raw["spawn"])


class TestHoldSource(unittest.TestCase):
    """AUDIT #12：评测的 hold 必须来自命令行或数据集，不能静默用 scene.step_s。"""

    def test_missing_hold_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=3, length=12.0)
            r = _eval(os.path.join(scene_dir, "scene.json"), os.path.join(tmp, "eval"))
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("--hold", r.stderr)
            self.assertFalse(os.path.isfile(os.path.join(tmp, "eval", "eval.json")))

    def test_hold_from_data_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=3, length=12.0)
            scene_json = os.path.join(scene_dir, "scene.json")
            data = os.path.join(tmp, "collect")
            r = _collect(scene_json, data, "--policy", "forward", "--hold", "0.2")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            # 只给 --data：hold 从 run.json 读
            r = _eval(scene_json, os.path.join(tmp, "e1"), "--data", data)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            e = _load(os.path.join(tmp, "e1", "eval.json"))
            self.assertEqual((e["hold_s"], e["hold_source"]), (0.2, "data run.json"))
            # episode 目录也行（episode_manifest.json）
            r = _eval(scene_json, os.path.join(tmp, "e2"), "--data", os.path.join(data, "ep_00000"))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            e = _load(os.path.join(tmp, "e2", "eval.json"))
            self.assertEqual((e["hold_s"], e["hold_source"]), (0.2, "data episode_manifest.json"))
            # --hold 与数据集不一致 -> 报错，不出报告
            r = _eval(scene_json, os.path.join(tmp, "e3"), "--data", data, "--hold", "0.5")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("不一致", r.stderr)
            self.assertFalse(os.path.isfile(os.path.join(tmp, "e3", "eval.json")))
            # 一致则通过
            r = _eval(scene_json, os.path.join(tmp, "e4"), "--data", data, "--hold", "0.2")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])

    def test_resolve_hold_unit(self):
        self.assertEqual(eval_mod.resolve_hold(0.5, None), (0.5, "cli --hold"))
        with self.assertRaises(ValueError):
            eval_mod.resolve_hold(None, None)
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "run.json"), "w") as f:
                json.dump({"hold_s": 0.25}, f)
            self.assertEqual(eval_mod.resolve_hold(None, tmp), (0.25, "data run.json"))
            with self.assertRaises(ValueError):
                eval_mod.resolve_hold(0.5, tmp)
            with open(os.path.join(tmp, "run.json"), "w") as f:
                json.dump({"hold_s": None}, f)
            with self.assertRaises(ValueError):
                eval_mod.resolve_hold(None, tmp)


class TestReachEuclid(unittest.TestCase):
    """AUDIT #10：到达判定按欧氏距离，目标两侧同距离同判；以前 x 不到 waypoint.x 一律失败。"""

    @staticmethod
    def _episode(xs, y=0.0):
        actions = [{"step": i, "action_executed": 1, "collision": False, "infer_ms": 1.0}
                   for i in range(len(xs))]
        oracle = [{"step": i, "pose_gt": {"x": x, "y": y, "z": 2.0, "yaw_deg": 0.0}}
                  for i, x in enumerate(xs)]
        return actions, oracle

    def test_symmetric_around_waypoint(self):
        scene = {"waypoint": {"x": -6.0, "y": 0.0}, "reach_radius_m": 1.0}
        a, o = self._episode([-9.0, -8.0, -6.5])     # 停在目标前 0.5 m（x 没到 -6）
        s_before = eval_mod.summarize_3dgs_episode(a, o, scene)
        a, o = self._episode([-9.0, -8.0, -5.5])     # 冲过目标 0.5 m
        s_after = eval_mod.summarize_3dgs_episode(a, o, scene)
        self.assertTrue(s_before["success"], s_before)
        self.assertTrue(s_after["success"], s_after)
        self.assertAlmostEqual(s_before["waypoint_min_dist"], s_after["waypoint_min_dist"])
        self.assertEqual(s_before["metric_version"], "3dgs.pilot.v2")
        self.assertEqual(s_before["reach_radius_m"], 1.0)
        # 对照：老口径（stop_radius=None）对同样两集给出不对称结果
        old_before = eval_mod.summarize_episode(*self._episode([-9.0, -8.0, -6.5]),
                                                goal_x=-6.0, waypoint=scene["waypoint"])
        old_after = eval_mod.summarize_episode(*self._episode([-9.0, -8.0, -5.5]),
                                               goal_x=-6.0, waypoint=scene["waypoint"])
        self.assertFalse(old_before["success"])
        self.assertTrue(old_after["success"])

    def test_outside_radius_or_collision_fails(self):
        scene = {"waypoint": {"x": -6.0, "y": 0.0}, "reach_radius_m": 1.0}
        a, o = self._episode([-9.0, -8.0, -7.2])
        self.assertFalse(eval_mod.summarize_3dgs_episode(a, o, scene)["success"])
        a, o = self._episode([-9.0, -6.0, -4.0])
        a[1]["collision"], a[1]["src"] = True, "voxel_proxy"
        s = eval_mod.summarize_3dgs_episode(a, o, scene)
        self.assertTrue(s["collision"])
        self.assertFalse(s["success"])
        # 默认半径 = eval_closed_loop.REACH_RADIUS
        s = eval_mod.summarize_3dgs_episode(*self._episode([-9.0, -7.4]),
                                            {"waypoint": {"x": -6.0, "y": 0.0}})
        self.assertEqual(s["reach_radius_m"], eval_mod.REACH_RADIUS)
        self.assertTrue(s["success"])
        with self.assertRaises(ValueError):
            eval_mod.summarize_3dgs_episode(a, o, {"waypoint": {"x": 0, "y": 0}, "reach_radius_m": 0})


class TestEpisodeFailureExit(unittest.TestCase):
    """AUDIT #4：单集异常必须进 failed 列表并让 eval 以非零退出，而不是只写进 summary。"""

    def test_episode_error_lists_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=3, length=12.0)
            scene_json = os.path.join(scene_dir, "scene.json")
            # 通过 sitecustomize 在子进程里给 run_episode 打补丁：第 2 集抛异常
            hook = os.path.join(tmp, "hook")
            os.makedirs(hook)
            with open(os.path.join(hook, "sitecustomize.py"), "w", encoding="utf-8") as f:
                f.write(
                    "import importlib\n"
                    "c = importlib.import_module('sim.3dgs.collect_3dgs')\n"
                    "_orig = c.run_episode\n"
                    "def boom(scene, out_dir, *a, **k):\n"
                    "    if out_dir.endswith('ep_00001'):\n"
                    "        raise RuntimeError('渲染器炸了')\n"
                    "    return _orig(scene, out_dir, *a, **k)\n"
                    "c.run_episode = boom\n")
            env = dict(os.environ, PYTHONPATH=hook + os.pathsep + ROOT)
            out = os.path.join(tmp, "eval")
            r = _run(["sim.3dgs.eval_3dgs", "--scene", scene_json, "--out", out,
                      "--policy", "forward", "--mock-renderer", "--episodes", "3",
                      "--steps", "2", "--hold", "0.5"], env=env)
            self.assertEqual(r.returncode, 1, r.stderr[-2000:])
            self.assertIn("FAILED episode=1", r.stderr)
            e = _load(os.path.join(out, "eval.json"))
            self.assertEqual(len(e["failed"]), 1)
            self.assertEqual(e["failed"][0]["episode"], 1)
            self.assertIn("渲染器炸了", e["failed"][0]["error"])
            self.assertEqual(e["episodes_done"], 2)
            self.assertEqual(e["episodes_requested"], 3)
            self.assertEqual([p["episode"] for p in e["per_episode"]], [0, 2])


class TestDyawUnits(unittest.TestCase):
    """AUDIT #13：yaw（度）与 yaw0（弧度）必须先统一单位；非零起始航向、跨 ±180° 都要对。"""

    def test_dyaw_wraps_in_degrees_across_180(self):
        renderer = importlib.import_module("sim.3dgs.renderer")
        collision = importlib.import_module("sim.3dgs.collision")
        dynamics = importlib.import_module("sim.3dgs.dynamics")

        class Turn:
            def __init__(self, action):
                self.action, self.name = action, "turn"

            def __call__(self, rgb, depth, state):
                return self.action, None

        for action, spawn_yaw, sign in ((dynamics.YAW_LEFT, 175.0, 1),
                                        (dynamics.YAW_RIGHT, -175.0, -1)):
            with tempfile.TemporaryDirectory() as tmp:
                scene_dir = os.path.join(tmp, "scene")
                scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
                scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
                scene["spawn"]["yaw_deg"] = spawn_yaw
                params = dynamics.CfBlackboxParams.from_json(None)
                identity = collect.scene_identity(scene)
                collect.run_episode(scene, os.path.join(tmp, "ep"),
                                    renderer.make_renderer(scene, True),
                                    collision.from_scene(scene), params, Turn(action),
                                    6, 0.5, split=identity["split"], identity=identity)
                with open(os.path.join(tmp, "ep", "actions.jsonl")) as f:
                    rows = [json.loads(x) for x in f]
            crossed = False
            for row in rows:
                before, after = row["yaw_before"], row["pose_gt"]["yaw_deg"]
                want = (after - before + 180.0) % 360.0 - 180.0
                self.assertAlmostEqual(row["dyaw_deg"], want, places=6, msg=row)
                self.assertLess(abs(row["dyaw_deg"]), 90.0, row)
                self.assertGreaterEqual(sign * row["dyaw_deg"], -1e-9, row)
                crossed |= (before > 0) != (after > 0) and abs(before) > 90
            self.assertTrue(crossed, ("要真的跨过 ±180°", spawn_yaw, rows[-1]["pose_gt"]))
            self.assertGreater(abs(sum(r["dyaw_deg"] for r in rows)), 30.0)


if __name__ == "__main__":
    unittest.main()
