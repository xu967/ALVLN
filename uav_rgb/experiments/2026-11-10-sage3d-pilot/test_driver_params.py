#!/usr/bin/env python3
"""bearing-guard-3dgs/driver.py 新增的三个可选 config 键（routes 字典、step_cap、teacher_tol_deg）与教师构造时传 hold_s 的单测。

不碰 GPU、不起 sitl、不读 PLY：把 driver 依赖的 sim.3dgs.* 与 student 换成桩模块，在进程内用 runpy 跑真实的 driver.py，
桩版 run_episode 只记下收到的 steps 并写出 episode_metrics 需要的最小 actions / oracle。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_driver_params.py
  DRIVER=<旧版 driver.py> python test_driver_params.py      拿改动前的版本跑，新功能的用例应当失败（红）
"""
import json
import os
import runpy
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.environ.get("DRIVER") or os.path.join(HERE, "..", "2026-11-10-bearing-guard-3dgs", "driver.py")
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

CALLS = []            # 桩 run_episode 收到的 (场景名, 路线下标, steps, seed)
LOADED = []           # 桩 make_renderer 被调用的次数（= 加载了场景）
TEACHER_KW = []       # 桩 OccAstarTeacher 每次构造收到的关键字参数


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def install_stubs(routes_in_scene_json):
    class Scene(dict):
        pass

    def load_scene(path):
        return Scene(path=path, ply_path=path, routes=routes_in_scene_json, reach_radius_m=1.5)

    def apply_route(sc, r):
        sc["spawn"], sc["waypoint"] = {"x": 0.0, "y": 0.0, "z": 1.2}, {"x": 1.0, "y": 0.0, "z": 1.2}
        sc["route"] = r

    def run_episode(sc, ep_dir, renderer, occ, params, policy, steps, hold, **kw):
        CALLS.append((os.path.basename(os.path.dirname(os.path.dirname(ep_dir))), sc["route"], steps, kw["seed"]))
        os.makedirs(ep_dir, exist_ok=True)
        with open(os.path.join(ep_dir, "actions.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"collision": False, "pose_before": {"x": 0.0, "y": 0.0}, "yaw_before": 0.0}) + "\n")
        with open(os.path.join(ep_dir, "oracle.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"pose_gt": {"x": 1.0, "y": 0.0}}) + "\n")
        return {"validity": {"physics_verified": True}, "stopped_on_goal": False, "performance": {"wall_s": 0.1}}, None

    mods = {
        "sim": types.ModuleType("sim"), "sim.3dgs": types.ModuleType("sim.3dgs"),
        "sim.3dgs.scene": _ns(load_scene=load_scene, apply_route=apply_route),
        "sim.3dgs.collision": _ns(from_scene=lambda base: object()),
        "sim.3dgs.renderer": _ns(make_renderer=lambda *a: LOADED.append(1) or object()),
        "sim.3dgs.planner": _ns(OccPlanner=lambda occ: _ns(plan=None), route_check=lambda *a, **k: {"feasible": True},
                                OccAstarTeacher=lambda planner, wp, **kw: TEACHER_KW.append(kw) or _ns(name="teacher")),
        "sim.3dgs.collect_3dgs": _ns(_sha256=lambda p: "0" * 64, scene_identity=lambda sc, ply_sha256=None: {},
                                     ForwardPolicy=lambda: _ns(name="forward"), StudentAdapter=None, run_episode=run_episode),
        "sim.3dgs.eval_3dgs": _ns(reach_radius_of=lambda sc: 1.5, route_shortest_path_m=lambda *a: 1.0,
                                  summarize_3dgs_episode=lambda *a, **k: {"success": True, "shortest_path_m": 1.0,
                                                                          "path_length_m": 1.0, "spl": 1.0, "waypoint_min_dist": 0.0}),
        "sim.3dgs.guard": _ns(),
        "sim.3dgs.dynamics": _ns(CfSitlParams=_ns(from_json=lambda p: _ns(profile=lambda: "stub"))),
        "student": _ns(StudentPolicy=None),
    }
    sys.modules.update(mods)
    return list(mods)


def run_driver(cfg, scene, scene_routes, arm="forward", arm_extra=None):
    CALLS.clear()
    LOADED.clear()
    TEACHER_KW.clear()
    with tempfile.TemporaryDirectory() as out:
        full = {"repo": HERE, "scenes_dir": out, "scenes": [scene], "k": 1, "seed0": 8000, "jitter": 0.2, "hold": 1.0,
                "step_frac": 0.5, "step_pad": 15, "guard_threshold_deg": 45.0, "route_lookahead_m": 1.5,
                "arms": [{"name": arm, "base": arm, "ckpt": None, "guard": None, **(arm_extra or {})}],
                "infer_bin": "x", "cf_sitl_proc": "y", **cfg}
        os.makedirs(os.path.join(out, scene))
        with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(full, f)
        with open(os.path.join(out, scene, "scene.json"), "w", encoding="utf-8") as f:
            json.dump({"routes": scene_routes}, f)
        names = install_stubs(scene_routes)
        argv, sys.argv = sys.argv, ["driver.py", out, scene, arm]
        try:
            with self_silence():
                try:
                    runpy.run_path(DRIVER, run_name="__main__")
                    rc = 0
                except SystemExit as e:
                    rc = e.code or 0
        finally:
            sys.argv = argv
            for n in names:
                sys.modules.pop(n, None)
        done = {}
        for r in range(len(scene_routes)):
            p = os.path.join(out, arm, scene, f"r{r}", "done.json")
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    done[r] = json.load(f)
        return rc, list(CALLS), len(LOADED), done


class self_silence:   # driver 的 log 走 stdout，测试里不要刷屏
    def __enter__(self):
        self._o, sys.stdout = sys.stdout, open(os.devnull, "w")

    def __exit__(self, *a):
        sys.stdout.close()
        sys.stdout = self._o


ROUTES = [{"max_steps": 60}, {"max_steps": 170}, {"max_steps": 400}]      # 未封顶步数 45 / 100 / 215


class TestDriverParams(unittest.TestCase):
    def test_legacy_list_routes_no_cap(self):
        rc, calls, loaded, done = run_driver({"routes": [0, 2]}, "sA", ROUTES)
        self.assertEqual((rc, loaded), (0, 1))
        self.assertEqual([(c[1], c[2]) for c in calls], [(0, 45), (2, 215)])      # 缺省不封顶：215 步照跑，与原版一致
        self.assertEqual({r: d["steps"] for r, d in done.items()}, {0: 45, 2: 215})

    def test_step_cap(self):
        rc, calls, _, done = run_driver({"routes": [0, 1, 2], "step_cap": 100}, "sA", ROUTES)
        self.assertEqual([c[2] for c in calls], [45, 100, 100])                   # min(100, 45 / 100 / 215)
        self.assertEqual({r: d["steps"] for r, d in done.items()}, {0: 45, 1: 100, 2: 100})

    def test_routes_dict_picks_this_scenes_routes(self):
        cfg = {"routes": {"sA": [1], "sB": [0, 2]}, "scenes": ["sA", "sB"]}
        rc, calls, _, _ = run_driver(cfg, "sA", ROUTES)
        self.assertEqual([(c[0], c[1], c[3]) for c in calls], [("sA", 1, 8100)])  # seed = 8000 + 100·路线下标 + k

    def test_routes_dict_empty_scene_exits_without_loading(self):
        rc, calls, loaded, _ = run_driver({"routes": {"sA": []}}, "sA", ROUTES)
        self.assertEqual((rc, calls, loaded), (0, [], 0))

    def test_routes_dict_missing_scene_is_an_error(self):
        with self.assertRaises(KeyError):
            run_driver({"routes": {"sB": [0]}}, "sA", ROUTES)

    def test_teacher_tol_deg_reaches_the_teacher(self):
        rc, calls, _, _ = run_driver({"routes": [0, 1], "teacher_tol_deg": 18.0}, "sA", ROUTES, arm="teacher")
        self.assertEqual(rc, 0)
        # 每条路线构造一次教师，都带上容差，也都带上 hold_s；显式的 tol_deg 在 OccAstarTeacher 里优先于 hold_s 的推导
        self.assertEqual(TEACHER_KW, [{"hold_s": 1.0, "tol_deg": 18.0}, {"hold_s": 1.0, "tol_deg": 18.0}])
        self.assertEqual(len(calls), 2)

    def test_teacher_default_tolerance_not_passed(self):
        rc, _, _, _ = run_driver({"routes": [0]}, "sA", ROUTES, arm="teacher")
        self.assertEqual(rc, 0)
        # 不给 teacher_tol_deg = 不传 tol_deg，只传 hold_s：教师按 hold 推导容差（hold 1.0 → 18.9°）。
        # 2026-11-11 之前这里是 [{}]（教师退回 12°），旧配置要复现旧结果须显式写 teacher_tol_deg: 12.0
        self.assertEqual(TEACHER_KW, [{"hold_s": 1.0}])

    def test_teacher_receives_the_global_hold(self):
        rc, _, _, _ = run_driver({"routes": [0, 1], "hold": 0.5}, "sA", ROUTES, arm="teacher")
        self.assertEqual(rc, 0)
        self.assertEqual(TEACHER_KW, [{"hold_s": 0.5}, {"hold_s": 0.5}])

    def test_teacher_receives_the_arms_effective_hold(self):
        rc, _, _, done = run_driver({"routes": [0]}, "sA", ROUTES, arm="teacher", arm_extra={"hold": 0.5})
        self.assertEqual(rc, 0)
        self.assertEqual(TEACHER_KW, [{"hold_s": 0.5}], "臂级 hold 盖过全局 hold，教师容差按实际执行的窗口推导")
        self.assertEqual(done[0]["hold"], 0.5)

    def test_hold_is_not_passed_to_other_bases(self):
        rc, calls, _, _ = run_driver({"routes": [0]}, "sA", ROUTES, arm="forward")
        self.assertEqual((rc, TEACHER_KW), (0, []))


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")     # driver 里 json.load(open(...)) 不关文件，会刷 ResourceWarning；那是原有写法，不是本次改动
