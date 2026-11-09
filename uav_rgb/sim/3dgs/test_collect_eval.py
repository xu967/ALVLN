#!/usr/bin/env python3
"""collect_3dgs / eval_3dgs 端到端小测（mock renderer，不需要 torch/学生权重）。

用 3dgs-venv 或 .venv-tools 跑（要 numpy；视频那一项要 cv2，打包那一项要 PIL）：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_collect_eval.py
没设 TMPDIR 时临时目录落 $DATA_ROOT/uav_rgb/tmp（根分区很小，不往 /tmp 写）。
"""
import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

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


# AUDIT_SIM #9 之后 --dynamics 默认 cf_sitl（要 cf_sitl_proc）；mock 链路显式走黑盒。
BLACKBOX = ["--dynamics", "cf_blackbox"]


def _collect(scene_json, out, *extra):
    return _run(["sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", out,
                 "--steps", "3", "--mock-renderer", *BLACKBOX, *extra])


def _eval(scene_json, out, *extra):
    return _run(["sim.3dgs.eval_3dgs", "--scene", scene_json, "--out", out,
                 "--policy", "forward", "--mock-renderer", "--episodes", "1",
                 "--steps", "3", *BLACKBOX, *extra])


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
                      *BLACKBOX, "--policy", "astar", "--route", "1"])
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load(os.path.join(out, "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["route"]["index"], m["route"]["name"]), (1, "r1"))
            self.assertEqual(m["scene"]["spawn"]["yaw_deg"], 90.0)
            self.assertEqual(m["waypoint"]["x"], 10.0)
            self.assertEqual(m["requested_steps"], 4)            # 没给 --steps 时用 routes[1].max_steps
            self.assertEqual((m["teacher"], m["planner"], m["stop_radius"]), ("plan_action", "astar_occ", 0.5))
            self.assertEqual(m["teacher_version"], "uavrgb.teacher_occ.v1")
            self.assertEqual(m["planner_failures"], 0)            # 走廊畅通：A* 没有 fail-closed 过
            with open(os.path.join(out, "ep_00000", "actions.jsonl")) as f:
                rows = [json.loads(x) for x in f]
            self.assertEqual(rows[0]["action_executed"], 3)      # YAW_RIGHT：航向 +y，目标在 +x
            # 默认 route（None）= 顶层 spawn/waypoint = routes[0]；task_id 与 route 1 不同
            r0 = _run(["sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", os.path.join(tmp, "r0"),
                       "--mock-renderer", *BLACKBOX, "--policy", "forward", "--steps", "2"])
            self.assertEqual(r0.returncode, 0, r0.stderr[-2000:])
            m0 = _load(os.path.join(tmp, "r0", "ep_00000", "episode_manifest.json"))
            self.assertEqual(m0["route"]["index"], 0)
            self.assertIsNone(m0["planner_failures"])            # forward 策略没有规划器
            self.assertEqual(m0["scene"]["spawn"], scene["spawn"])
            self.assertNotEqual(m0["task_id"], m["task_id"])
            self.assertEqual(m0["geometry_gid"], m["geometry_gid"])
            bad = _run(["sim.3dgs.collect_3dgs", "--scene", scene_json, "--out", os.path.join(tmp, "bad"),
                        "--mock-renderer", *BLACKBOX, "--route", "2", "--steps", "1"])
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("route 2", bad.stderr)

    def test_frame_quality_marked_not_dropped(self):
        """每帧 frame_quality（lapvar / hole_frac / low）写进 actions.jsonl、manifest 汇总 n_low；雾帧只标不丢。"""
        import numpy as np
        flat = np.full((240, 320, 3), 128, np.uint8)                  # 纯雾：lapvar 0
        q = collect.frame_quality(flat, np.zeros((240, 320), np.float32))
        self.assertEqual((q["lapvar"], q["hole_frac"], q["low"]), (0.0, 1.0, True))
        noisy = np.random.default_rng(0).integers(0, 255, (240, 320, 3), np.uint8)
        self.assertFalse(collect.frame_quality(noisy, np.ones((240, 320), np.float32))["low"])
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            out = os.path.join(tmp, "ep")
            r = _collect(os.path.join(scene_dir, "scene.json"), out, "--policy", "forward")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load(os.path.join(out, "ep_00000", "episode_manifest.json"))
            with open(os.path.join(out, "ep_00000", "actions.jsonl")) as f:
                rows = [json.loads(x) for x in f]
            self.assertEqual(len(rows), 3)                              # 不丢帧
            self.assertTrue(all(set(x["frame_quality"]) == {"lapvar", "hole_frac", "low"} for x in rows))
            self.assertEqual(m["frame_quality"]["lapvar_min"], collect.FRAME_LAPVAR_MIN)
            self.assertEqual(m["frame_quality"]["n_low"], sum(x["frame_quality"]["low"] for x in rows))
            self.assertEqual(m["camera"]["fov_deg"], 60.0)              # 合成场景 scene.camera.fov_deg
            self.assertEqual(m["camera"]["mount"]["pitch_deg"], 0.0)

    def test_dynamics_default_is_cf_sitl(self):
        """AUDIT_SIM #9：不给 --dynamics 走 cf_sitl（固件在环，proc 缺失要明确报错而不是悄悄退回黑盒）；
        cf_blackbox 只能显式指定。collect / eval 两个入口同口径。"""
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            scene_json = os.path.join(scene_dir, "scene.json")
            env = dict(os.environ, CF_SITL_PROC=os.path.join(tmp, "no_such_cf_sitl_proc"))
            cases = (("sim.3dgs.collect_3dgs", ["--policy", "forward", "--steps", "2"]),
                     ("sim.3dgs.eval_3dgs", ["--policy", "forward", "--steps", "2", "--hold", "0.5", "--episodes", "1"]))
            for i, (mod, extra) in enumerate(cases):
                base = [mod, "--scene", scene_json, "--mock-renderer", *extra]
                r = _run(base + ["--out", os.path.join(tmp, f"default{i}")], env=env)
                self.assertNotEqual(r.returncode, 0, f"{mod} 默认应是 cf_sitl，proc 缺失必须失败")
                self.assertIn("cf_sitl_proc 不存在", r.stderr, mod)
                self.assertFalse(os.path.isdir(os.path.join(tmp, f"default{i}", "ep_00000")))
                r = _run(base + ["--out", os.path.join(tmp, f"bb{i}"), "--dynamics", "cf_blackbox"], env=env)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            run = _load(os.path.join(tmp, "bb0", "run.json"))
            self.assertEqual(run["dynamics"], "cf_blackbox")


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
    def test_student_adapter_resets_policy_at_episode_start(self):
        # eval_3dgs 跨集复用一个 StudentPolicy 进程、每集新建一个 adapter：双帧 / prev-action 模型的
        # 帧缓存必须在集边界清掉，否则第 k≥1 集首帧拿的是上一集末帧（REVIEW-P2 Must fix #1）。
        class FakeStudent:
            resets = 0

            def reset(self):
                self.resets += 1

        s = FakeStudent()
        collect.StudentAdapter(s)
        collect.StudentAdapter(s)
        self.assertEqual(s.resets, 2)
        # DAgger 采集整轮复用一个 DaggerPolicy：run_episode 开头调 policy.reset() 也要落到学生
        class FakeExpert:
            planner_name, stop_radius, n_plan_fail = "astar_occ", 0.5, 0

        import random
        dp = collect.DaggerPolicy(s, FakeExpert(), 0.5, random.Random(0))
        dp.last_expert = 1
        dp.reset()
        self.assertEqual((s.resets, dp.last_expert), (4, None))

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
                      "--policy", "forward", "--mock-renderer", *BLACKBOX, "--episodes", "1",
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


def _rows(ep_dir):
    with open(os.path.join(ep_dir, "actions.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def _mock_env(tmp, seed=2):
    """合成走廊 + mock 渲染器 + cf_blackbox，直接调 run_episode 用。"""
    scene_dir = os.path.join(tmp, "scene")
    scene_mod.make_corridor(scene_dir, seed=seed, length=12.0)
    scene = scene_mod.load_scene(os.path.join(scene_dir, "scene.json"))
    renderer = importlib.import_module("sim.3dgs.renderer").make_renderer(scene, True)
    occ = importlib.import_module("sim.3dgs.collision").from_scene(scene)
    params = importlib.import_module("sim.3dgs.dynamics").CfBlackboxParams.from_json(None)
    return scene, renderer, occ, params, collect.scene_identity(scene)


class FakeStudent:
    """student.StudentPolicy 的最小替身：恒定动作，不起 uavrgb_infer。"""

    def __init__(self, action):
        self.action, self.resets = action, 0

    def reset(self):
        self.resets += 1

    def act(self, msg):
        return self.action, 1.0, [0.0] * 4


class TestExecutedBy(unittest.TestCase):
    """R2-3DGS #2：executed_by 缺省按策略名；非 DAgger 学生集（eval_3dgs）action_expert=None、beta=0；
    以前缺省写 teacher / beta 1.0，117 步学生动作被记成「教师标签、教师执行」。"""

    def test_student_adapter_episode_is_marked_student(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene, renderer, occ, params, identity = _mock_env(tmp)
            policy = collect.StudentAdapter(FakeStudent(action=2))
            m, err = collect.run_episode(scene, os.path.join(tmp, "s"), renderer, occ, params, policy,
                                         3, 0.5, split=identity["split"], identity=identity)
            self.assertIsNone(err)
            self.assertEqual((m["teacher"], m["executed_by"], m["beta"], m["dagger_round"], m["n_expert_executed"]),
                             ("student", "student", 0.0, 0, None))
            rows = _rows(os.path.join(tmp, "s"))
            self.assertEqual([r["action_executed"] for r in rows], [2, 2, 2])
            self.assertEqual([r["action_expert"] for r in rows], [None, None, None])   # 学生动作不是标签
            # 对照：forward 对照集 / 深度教师的缺省不变
            m_f, _ = collect.run_episode(scene, os.path.join(tmp, "f"), renderer, occ, params, collect.ForwardPolicy(),
                                         1, 0.5, split=identity["split"], identity=identity)
            self.assertEqual((m_f["executed_by"], m_f["beta"]), ("forward", 1.0))
            m_t, _ = collect.run_episode(scene, os.path.join(tmp, "t"), renderer, occ, params, collect.DepthTeacher(),
                                         2, 0.5, split=identity["split"], identity=identity)
            self.assertEqual((m_t["teacher"], m_t["executed_by"], m_t["beta"]), ("expert_action", "teacher", 1.0))
            self.assertTrue(all(r["action_expert"] == r["action_executed"] for r in _rows(os.path.join(tmp, "t"))))

    def test_dagger_beta_one_executes_expert_every_step(self):
        import random
        planner = importlib.import_module("sim.3dgs.planner")
        with tempfile.TemporaryDirectory() as tmp:
            scene, renderer, occ, params, _ = _mock_env(tmp)
            scene["waypoint"] = {"x": 10.0, "y": 0.0, "z": 2.0}      # 走廊默认 waypoint x=30 在尽头墙里，A* 无路
            identity = collect.scene_identity(scene)
            expert = planner.OccAstarTeacher(planner.OccPlanner(occ), scene["waypoint"])
            # 学生恒 SAFE；β=1 每步都换成教师动作，β=0 每步都执行学生动作，两种情况标签都是教师的
            for beta, out in ((1.0, "b1"), (0.0, "b0")):
                policy = collect.DaggerPolicy(FakeStudent(action=0), expert, beta, random.Random(0))
                dagger = {"executed_by": "student", "dagger_round": 1, "beta": beta}
                m, err = collect.run_episode(scene, os.path.join(tmp, out), renderer, occ, params, policy, 6, 0.5,
                                             dagger=dagger, split=identity["split"], identity=identity)
                self.assertIsNone(err)
                rows = _rows(os.path.join(tmp, out))
                self.assertEqual(m["num_steps"], 6)
                self.assertEqual((m["teacher"], m["planner"], m["executed_by"], m["beta"], m["dagger_round"]),
                                 ("plan_action", "astar_occ", "student", beta, 1))
                self.assertTrue(any(r["action_expert"] != 0 for r in rows))       # 走廊畅通：教师不是全程 SAFE
                if beta == 1.0:
                    self.assertTrue(all(r["action_executed"] == r["action_expert"] for r in rows), rows)
                    self.assertEqual(m["n_expert_executed"], m["num_steps"])
                else:
                    self.assertEqual([r["action_executed"] for r in rows], [0] * 6)
                    self.assertEqual(m["n_expert_executed"], 0)


class ScriptedDyn:
    """假动力学：每步按脚本给世界系位移 (dx, dy) 与窗口末世界系速度 (vx, vy)，航向不变；测倒飞判据用。"""

    def __init__(self, moves):
        self.moves, self.i = moves, 0
        self.x = self.y = self.t = 0.0
        self.z, self.yaw = 2.0, 0.0

    def step(self, action, dt):
        dx, dy, vx, vy = self.moves[self.i % len(self.moves)]
        self.i += 1
        self.x, self.y, self.t = self.x + dx, self.y + dy, self.t + dt
        return {"t": self.t, "x": self.x, "y": self.y, "z": self.z, "vx": vx, "vy": vy,
                "yaw": self.yaw, "yaw_deg": math.degrees(self.yaw)}

    def profile(self):
        return "scripted"

    def manifest(self):
        return {"dynamics": "scripted"}


class TestBackwardAnomaly(unittest.TestCase):
    """REVIEW-3DGS #3：倒飞判据用机体系前向速度连续两窗为负，不用世界位移沿起始航向的投影。
    拐角后机体带着横向动量时位移沿航向分量可为 -0.06 m（旧判据误判隔离），机体并没有后退。"""

    def _run(self, tmp, name, moves, spawn_yaw=0.0, steps=4):
        scene, renderer, occ, params, _ = _mock_env(tmp)
        scene["spawn"]["yaw_deg"] = spawn_yaw
        identity = collect.scene_identity(scene)
        real = collect.make_blackbox
        collect.make_blackbox = lambda scene, path, params: ScriptedDyn(moves)
        try:
            m, err = collect.run_episode(scene, os.path.join(tmp, name), renderer, occ, params, collect.ForwardPolicy(),
                                         steps, 0.5, split=identity["split"], identity=identity)
        finally:
            collect.make_blackbox = real
        self.assertIsNone(err)
        return m

    def test_lateral_momentum_is_not_backward(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 航向 +x：每窗位移 (-0.06, +0.14)（沿航向 -6 cm，旧阈值 5 cm 会判倒飞），窗口末速度 (+0.05, +0.20) 机体前向为正
            m = self._run(tmp, "lat", [(-0.06, 0.14, 0.05, 0.20)])
            self.assertIsNone(m["dynamics_anomaly"])
            self.assertEqual((m["num_steps"], m["validity"]["physics_verified"]), (4, True))
            # 航向 +y：世界系 -x 的速度是横向，不是后退
            m = self._run(tmp, "lat90", [(-0.15, 0.0, -0.3, 0.0)], spawn_yaw=90.0)
            self.assertIsNone(m["dynamics_anomaly"])
            # 单个窗口后退再恢复：不连续两窗，不判
            m = self._run(tmp, "once", [(-0.15, 0.0, -0.3, 0.0), (0.15, 0.0, 0.3, 0.0)])
            self.assertIsNone(m["dynamics_anomaly"])
            self.assertEqual(m["num_steps"], 4)

    def test_true_backward_flight_is_flagged_on_second_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            for yaw, moves in ((0.0, [(-0.15, 0.0, -0.3, 0.0)]),            # 航向 +x，世界 -x 后退
                               (90.0, [(0.0, -0.15, 0.0, -0.3)])):          # 航向 +y，世界 -y 后退
                m = self._run(tmp, f"back{int(yaw)}", moves, spawn_yaw=yaw)
                a = m["dynamics_anomaly"]
                self.assertIsNotNone(a, m)
                self.assertEqual((a["kind"], a["step"]), ("backward_under_forward", 1))
                self.assertLess(a["vx_body_mps"], -collect.ANOMALY_BACK_VX)
                self.assertEqual(m["num_steps"], 2)                           # 第二窗判定后本集结束
                self.assertFalse(m["validity"]["physics_verified"])
                self.assertEqual(len(_rows(os.path.join(tmp, f"back{int(yaw)}"))), 2)


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

    def test_spl_columns(self):
        """SPL（2026-11-07 只加列）：l_i 来自占据代理 A*，p_i 是 oracle xy 折线截到进入 reach 圆盘；不给 l_i 时 spl 为 None。"""
        import numpy as np
        collision = importlib.import_module("sim.3dgs.collision")
        planner = importlib.import_module("sim.3dgs.planner")
        scene = {"waypoint": {"x": 8.0, "y": 0.0}, "reach_radius_m": 1.0}
        a, o = self._episode([2.0, 4.0, 6.0, 7.5])
        self.assertIsNone(eval_mod.summarize_3dgs_episode(a, o, scene)["spl"])
        s = eval_mod.summarize_3dgs_episode(a, o, scene, shortest_path_m=4.5)
        self.assertTrue(s["success"])
        self.assertAlmostEqual(s["path_length_m"], 5.0)      # x=2 起飞，x=7 进 1.0 m 圆盘，7.5 那半米不算
        self.assertAlmostEqual(s["spl"], 4.5 / 5.0)
        # 一堵 x=5、y∈[-3,1] 的墙（planner 自检同款）：直线被挡，l_i 必须大于直线距离 6 m。
        ys = np.linspace(-3.0, 1.0, 21)
        occ = collision.Occupancy(np.array([(5.0, y, 2.0) for y in ys]), np.tile([0.01, 0.2, 0.3], (len(ys), 1)),
                                  None, np.full(len(ys), 0.95), z_range=(1.65, 2.35), voxel_m=0.1,
                                  body_radius_m=0.102, body_half_height_m=0.028, pad_m=4.0)
        pl = planner.OccPlanner(occ)
        start = [{"pose_gt": {"x": 2.0, "y": 0.0, "z": 2.0, "yaw_deg": 0.0}}]
        l_i = eval_mod.route_shortest_path_m(pl, start, scene)
        # 绕墙：直线 6 m 被挡，绕到 y>1 再回来；量到 1.0 m 圆盘边界后仍明显长于 5 m 直线。
        self.assertGreater(l_i, 5.5)
        # 圆盘越大 l_i 越短；两次截点之间的弧长至少是半径差 1 m（末段不是正对圆心时更长），但不会离谱。
        l_big = eval_mod.route_shortest_path_m(pl, start, dict(scene, reach_radius_m=2.0))
        self.assertLess(l_big, l_i)
        self.assertGreaterEqual(l_i - l_big, 1.0 - 1e-6)
        self.assertLess(l_i - l_big, 2.0)
        self.assertIsNone(eval_mod.route_shortest_path_m(None, o, scene))
        self.assertIsNone(eval_mod.route_shortest_path_m(pl, [], scene))
        # 目标在墙里：无路 -> None，评测不崩。
        self.assertIsNone(eval_mod.route_shortest_path_m(pl, o, {"waypoint": {"x": 5.0, "y": 0.0}}))


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
                      "--policy", "forward", "--mock-renderer", *BLACKBOX, "--episodes", "3",
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


class ScriptedOcc:
    """真占据代理的包装：第 hit_step 步的 sweep 强制判碰撞，其余步按真代理（走廊畅通 = 不撞）。"""

    def __init__(self, occ, hit_step):
        self._occ, self.hit_step, self.calls = occ, hit_step, 0

    def sweep(self, p0, p1):
        i, self.calls = self.calls, self.calls + 1
        return (True, [float(v) for v in p1]) if i == self.hit_step else self._occ.sweep(p0, p1)

    def __getattr__(self, name):
        return getattr(self._occ, name)


class TestCollisionRows(unittest.TestCase):
    """AUDIT_SIM #3 + FIXLIST D2：3DGS 没有接触物理，撞后机体穿模继续飞；撞后每一行都要 collision=True
    （--drop-collision 一并丢），--truncate-on-collision 撞到就结束本集但不算闭环评测产物（stopped_on_collision 不变）。"""

    HIT = 2

    def _episode(self, tmp, name, steps, **kw):
        scene, renderer, occ, params, identity = _mock_env(tmp)
        occ = ScriptedOcc(occ, self.HIT)
        # 深度教师（teacher=expert_action）：preprocess 的教师门禁只收教师标签，forward 对照集会被拒
        m, err = collect.run_episode(scene, os.path.join(tmp, name), renderer, occ, params, collect.DepthTeacher(),
                                     steps, 0.5, split=identity["split"], identity=identity, **kw)
        self.assertIsNone(err)
        return m, _rows(os.path.join(tmp, name))

    def _preprocess(self, tmp, name, ep_dir):
        try:
            import PIL  # noqa: F401
        except ImportError:
            self.skipTest("无 PIL，不跑 preprocess")
        out = os.path.join(tmp, name)
        r = _run_script(os.path.join(ROOT, "tools", "preprocess.py"), ["--out", out, "--drop-collision", ep_dir])
        if r.returncode != 0:
            return r, None
        with open(os.path.join(out, "dataset.uavr"), "rb") as f:     # meta JSON 在文件尾部（DATASET_FORMAT §1.4）
            data = f.read()
        import struct
        meta_off, meta_len = struct.unpack_from("<QI", data, 4 + 2 + 2 + 4 + 2 + 2 + 4 + 8 + 8)
        return r, json.loads(data[meta_off:meta_off + meta_len].decode("utf-8"))

    def test_rows_after_collision_all_flagged_and_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            m, rows = self._episode(tmp, "cont", self.HIT + 4)
            self.assertEqual(m["num_steps"], self.HIT + 4)
            self.assertEqual([r["collision"] for r in rows], [False] * self.HIT + [True] * 4)
            self.assertEqual(rows[self.HIT]["src"], "voxel_proxy")                     # 真撞的那一步
            self.assertEqual([r["src"] for r in rows[self.HIT + 1:]], ["post_collision"] * 3)   # 之后穿模的三步
            self.assertIsNotNone(rows[self.HIT]["collision_point"])
            self.assertFalse(m["stopped_on_collision"])
            self.assertFalse(m["truncated_on_collision"])
            self.assertTrue(m["validity"]["physics_verified"])
            r, meta = self._preprocess(tmp, "ds", os.path.join(tmp, "cont"))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:] + r.stdout[-1000:])
            self.assertEqual((meta["num_samples"], meta["collision_dropped"]), (self.HIT, 4))

    def test_truncate_on_collision_keeps_episode_and_drops_only_hit_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            m, rows = self._episode(tmp, "trunc", self.HIT + 4, truncate_on_collision=True)
            self.assertEqual(m["num_steps"], self.HIT + 1)                             # 撞的那一步写完就结束
            self.assertEqual([r["collision"] for r in rows], [False] * self.HIT + [True])
            self.assertEqual((m["stopped_on_collision"], m["truncated_on_collision"]), (False, True))
            self.assertTrue(m["validity"]["physics_verified"])
            r, meta = self._preprocess(tmp, "ds", os.path.join(tmp, "trunc"))
            self.assertEqual(r.returncode, 0, r.stderr[-2000:] + r.stdout[-1000:])
            self.assertEqual((meta["num_samples"], meta["collision_dropped"]), (self.HIT, 1))
            # 对照：--stop-on-collision 仍是闭环评测语义，preprocess 照旧整集拒收
            m2, rows2 = self._episode(tmp, "stop", self.HIT + 4, stop_on_collision=True)
            self.assertEqual((m2["num_steps"], m2["stopped_on_collision"], m2["truncated_on_collision"]),
                             (self.HIT + 1, True, False))
            r2, _ = self._preprocess(tmp, "ds2", os.path.join(tmp, "stop"))
            self.assertNotEqual(r2.returncode, 0)
            self.assertIn("stopped_on_collision", r2.stderr + r2.stdout)
            with self.assertRaises(ValueError):
                self._episode(tmp, "both", 3, stop_on_collision=True, truncate_on_collision=True)

    def test_completed_aligned_with_gazebo(self):
        """AUDIT_SIM Nice 7：manifest.completed = 跑满请求步数 or goal-stop（与 sim/tools/collect.py 同义）；
        validity.physics_verified 保持「有帧、无 error、无动力学异常」：碰撞截断 / 评测即停的集 completed=False
        但 physics_verified 仍 true（截断集可打包）。"""
        with tempfile.TemporaryDirectory() as tmp:
            m, _ = self._episode(tmp, "full", self.HIT + 4)                                  # 撞了但跑满
            self.assertEqual((m["completed"], m["validity"]["physics_verified"]), (True, True))
            m, _ = self._episode(tmp, "trunc", self.HIT + 4, truncate_on_collision=True)
            self.assertEqual((m["completed"], m["validity"]["physics_verified"]), (False, True))
            m, _ = self._episode(tmp, "stop", self.HIT + 4, stop_on_collision=True)
            self.assertEqual((m["completed"], m["stopped_on_collision"]), (False, True))
            # goal-stop：出生点就在 waypoint 半径内，教师连续 3 步 SAFE 提前结束，也算 completed
            scene, renderer, occ, params, identity = _mock_env(tmp)
            scene["waypoint"] = {"x": scene["spawn"]["x"], "y": scene["spawn"]["y"], "z": 2.0}

            class HoverAtGoal:
                name, stop_radius = "plan_action", 0.5

                def __call__(self, rgb, depth, dyn):
                    return collect.SAFE_NEUTRAL, None

            m, err = collect.run_episode(scene, os.path.join(tmp, "goal"), renderer, occ, params, HoverAtGoal(),
                                         8, 0.5, split=identity["split"], identity=identity)
            self.assertIsNone(err)
            self.assertEqual((m["stopped_on_goal"], m["num_steps"], m["completed"]), (True, 3, True))
        with open(os.path.join(ROOT, "tools", "DATASET_FORMAT.md"), encoding="utf-8") as f:
            doc = f.read()
        self.assertIn("`completed`", doc)
        self.assertIn("physics_verified", doc)

    def test_cli_flag_and_pilot_uses_truncate(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = os.path.join(tmp, "scene")
            scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
            r = _collect(os.path.join(scene_dir, "scene.json"), os.path.join(tmp, "c"), "--policy", "forward",
                         "--truncate-on-collision")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load(os.path.join(tmp, "c", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["truncated_on_collision"], m["stopped_on_collision"]), (False, False))   # 畅通：没撞
            bad = _collect(os.path.join(scene_dir, "scene.json"), os.path.join(tmp, "b"), "--policy", "forward",
                           "--truncate-on-collision", "--stop-on-collision")
            self.assertNotEqual(bad.returncode, 0)
        with open(os.path.join(ROOT, "tools", "demo_3dgs_routes.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn('"--truncate-on-collision"', src)
        self.assertNotIn('"--stop-on-collision"', src)       # 教师 pilot 不再用闭环评测语义采训练数据


class TestTofTerrain(unittest.TestCase):
    """AUDIT_SIM #2：SITL 的 ToF 是平地模型（(z-ground_z)/cos），飞行层以下的桌面/长椅在仿真里不存在，真机 ToF 到
    桌面会读短、执行器按「离地 2 m」爬升。只打标不改 ToF 输入：route_check 用同一份几何在 [TOF_TERRAIN_Z_MIN, z_slice 下沿)
    再切一层占据，沿路径查有没有东西 → tof_terrain_m（最高格顶）；它不为 None 或路线名含 over_ → agl_model_unverified，
    写进 manifest 与 planner CLI 输出（场地文档据此标注）。"""

    WP = {"x": 10.0, "y": 0.0, "z": 2.0}

    def _scene(self, tmp, table):
        scene_dir = os.path.join(tmp, "scene_table" if table else "scene_plain")
        scene_mod.make_corridor(scene_dir, seed=2, length=12.0)
        if table:
            # 桌面：x∈[3.5,4.5] 横贯走廊 y∈[-3.5,3.5]、z=0.75 的一层高斯（σ 0.06）；到 waypoint 的任何路都从它上方过
            ply = os.path.join(scene_dir, "scene.ply")
            g = scene_mod.read_gaussian_ply(ply)
            xs, ys = np.meshgrid(np.arange(3.5, 4.5, 0.1), np.arange(-3.5, 3.5, 0.1))
            tab = np.stack([xs.ravel(), ys.ravel(), np.full(xs.size, 0.75)], 1).astype(np.float32)
            n = len(tab)
            scene_mod.write_gaussian_ply(
                ply, np.vstack([g["means"], tab]), np.vstack([g["colors"], np.tile([0.6, 0.4, 0.2], (n, 1))]),
                np.vstack([g["scales"], np.full((n, 3), 0.06, np.float32)]),
                np.vstack([g["quats"], np.tile([1.0, 0, 0, 0], (n, 1))]),
                np.concatenate([np.full(len(g["means"]), 4.6, np.float32), np.full(n, 4.6, np.float32)]))   # logit 4.6 ≈ 0.99
        return os.path.join(scene_dir, "scene.json")

    def test_route_check_flags_low_furniture_and_over_routes(self):
        collision = importlib.import_module("sim.3dgs.collision")
        planner = importlib.import_module("sim.3dgs.planner")
        with tempfile.TemporaryDirectory() as tmp:
            for table in (False, True):
                scene = scene_mod.load_scene(self._scene(tmp, table))
                occ = collision.from_scene(scene)
                terrain = collision.terrain_from_scene(scene)
                self.assertEqual(terrain.z_range, (collision.TOF_TERRAIN_Z_MIN, 1.65))   # 走廊 z_slice 下沿 1.65
                self.assertEqual(terrain.voxel, occ.voxel)
                pl = planner.OccPlanner(occ)
                r = planner.route_check(pl, scene["spawn"], self.WP, terrain_occ=terrain, route_name="r9_straight")
                self.assertTrue(r["feasible"], r)
                if table:
                    self.assertIsNotNone(r["tof_terrain_m"])
                    self.assertTrue(0.7 <= r["tof_terrain_m"] <= 1.1, r["tof_terrain_m"])   # 桌面 0.75 + 3σ 外扩，格顶
                    self.assertEqual((r["agl_model_unverified"], r["agl_model_unverified_reason"]), (True, ["tof_terrain"]))
                else:
                    self.assertEqual((r["tof_terrain_m"], r["agl_model_unverified"], r["agl_model_unverified_reason"]),
                                     (None, False, []))
                    r2 = planner.route_check(pl, scene["spawn"], self.WP, terrain_occ=terrain,
                                             route_name="r3_north_south_over_table")
                    self.assertEqual((r2["tof_terrain_m"], r2["agl_model_unverified"], r2["agl_model_unverified_reason"]),
                                     (None, True, ["route_name"]))
                    r3 = planner.route_check(pl, scene["spawn"], self.WP)            # 不给 terrain：不判，字段仍在
                    self.assertEqual((r3["tof_terrain_m"], r3["agl_model_unverified"]), (None, False))

    def test_manifest_and_planner_cli_carry_agl_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_json = self._scene(tmp, True)
            raw = _load(scene_json)
            raw["routes"] = [{"name": "r0_over_table", "spawn": raw["spawn"], "waypoint": self.WP, "max_steps": 6},
                             {"name": "r1_plain_name", "spawn": raw["spawn"], "waypoint": self.WP, "max_steps": 6}]
            scene_mod.save_scene(raw, scene_json)
            r = _collect(scene_json, os.path.join(tmp, "a"), "--policy", "astar", "--route", "0", "--steps", "2")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load(os.path.join(tmp, "a", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["agl_model_unverified"], m["agl_model_unverified_reason"]), (True, ["tof_terrain", "route_name"]))
            self.assertTrue(0.7 <= m["tof_terrain_m"] <= 1.1, m["tof_terrain_m"])
            # 非 A* 策略没有路径：沿 spawn→waypoint 直线段查
            r = _collect(scene_json, os.path.join(tmp, "f"), "--policy", "forward", "--route", "1", "--steps", "2")
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            m = _load(os.path.join(tmp, "f", "ep_00000", "episode_manifest.json"))
            self.assertEqual((m["agl_model_unverified"], m["agl_model_unverified_reason"]), (True, ["tof_terrain"]))
            out = _run(["sim.3dgs.planner", "--scene", scene_json, "--json"])
            rows = json.loads(out.stdout)
            self.assertEqual([x["agl_model_unverified_reason"] for x in rows], [["tof_terrain", "route_name"], ["tof_terrain"]])
            self.assertTrue(all(0.7 <= x["tof_terrain_m"] <= 1.1 for x in rows))


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
