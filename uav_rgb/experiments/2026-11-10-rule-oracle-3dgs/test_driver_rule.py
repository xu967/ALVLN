#!/usr/bin/env python3
"""bearing-guard-3dgs/driver.py 为 rule-oracle-3dgs 新增的可选配置键的单测，加一组「不给新键时输出逐位不变」的锁。

新键（driver.py 文件头有说明）：臂 base = rule-occ / rule-depth、臂 hold、臂 step_mult、臂与全局 rule_params、全局 drop_trace。
不碰 GPU、不起 sitl、不读 PLY：沿用 sage3d-pilot/test_driver_params.py 的桩（把 driver 依赖的 sim.3dgs.* 与 student 换成桩模块，
用 runpy 跑真实的 driver.py），在其上补桩 guard / rule_oracle 与会记录调用的 run_episode。只用标准库。

  PYTHONDONTWRITEBYTECODE=1 python test_driver_rule.py
  DRIVER=<旧版 driver.py> python test_driver_rule.py      拿改动前的版本跑：「新键」类用例应当红，「逐位不变」类应当绿
"""
import importlib.util
import json
import os
import runpy
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("tdp", os.path.join(HERE, "..", "2026-11-10-sage3d-pilot", "test_driver_params.py"))
tdp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tdp)             # 复用桩与 DRIVER 的取法（环境变量 DRIVER 优先）
DRIVER = tdp.DRIVER
_ns = tdp._ns

PLAN = object()                            # 桩 OccPlanner.plan：断言 driver 把它原样交给 OracleRouteGuide
ENTRIES = ["V1-fwd", "V1-side", "V1-fwd"]  # 桩规则策略每集 3 步，条目依次如此
REC = {}                                   # 每次 run() 重置：runs / rules / guides / guarded


class FakeRouteGuide:
    def __init__(self, plan, wp, lookahead):
        self.plan, self.wp, self.lookahead, self.plan_failed = plan, wp, lookahead, False
        REC["guides"].append(self)

    def target(self, x, y):
        return self.wp


class FakeGuarded:
    name = "guarded"

    def __init__(self, inner, target_fn, thr):
        self.inner, self.target_fn, self.thr = inner, target_fn, thr
        self.trace, self.n_steps, self.n_triggered, self.n_rewritten = [], 0, 0, 0
        REC["guarded"].append(self)

    def reset(self):
        self.trace, self.n_steps, self.n_triggered, self.n_rewritten = [], 0, 0, 0

    def __call__(self, rgb, depth, state):
        self.trace.append({"step": self.n_steps, "triggered": True, "rewritten": False})
        self.n_steps += 1
        self.n_triggered += 1
        return 0, None


class FakeRule:
    """桩 RuleOraclePolicy：构造参数原样留存；每集 3 步，rows 的 step / entry 是 driver 汇总所需的最小字段。"""
    name, planner_name, version = "rule_oracle", "rule_v1", "stub"

    def __init__(self, occ, target_fn, source="occ", params=None, **kw):
        self.occ, self.target_fn, self.source, self.params_arg = occ, target_fn, source, params
        self.params = {"r_safe": 0.35, "d_look": 1.2, "yaw_step_deg": 26.0, **(params or {})}      # 桩：有效参数 = 缺省 + 覆盖
        self.rows = []
        REC["rules"].append(self)

    def reset(self):
        self.rows = []

    def __call__(self, rgb, depth, state):
        i = len(self.rows)
        self.rows.append({"step": i, "entry": ENTRIES[i % 3], "action": 0, "d_occ": None, "d_depth": None})
        return 0, None


def _recording(orig):
    def run_episode(sc, ep_dir, renderer, occ, params, policy, steps, hold, **kw):
        REC["runs"].append({"scene": os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(ep_dir)))),
                            "route": sc["route"], "steps": steps, "hold": hold, "kw": dict(kw), "policy": policy, "occ": occ,
                            "wp": (float(sc["waypoint"]["x"]), float(sc["waypoint"]["y"]))})
        out = orig(sc, ep_dir, renderer, occ, params, policy, steps, hold, **kw)      # 写最小 actions / oracle
        if callable(policy):
            if hasattr(policy, "reset"):
                policy.reset()
            for _ in range(3):
                policy(None, None, None)
        with open(os.path.join(ep_dir, "trace.jsonl"), "w", encoding="utf-8") as f:   # 真 run_episode 会写；用来测 drop_trace
            f.write("{}\n")
        return out
    return run_episode


def install(routes):
    names = tdp.install_stubs(routes)
    c3 = sys.modules["sim.3dgs.collect_3dgs"]
    c3.run_episode = _recording(c3.run_episode)
    guard = sys.modules["sim.3dgs.guard"]
    guard.OracleRouteGuide, guard.OracleGuardedPolicy = FakeRouteGuide, FakeGuarded
    sys.modules["sim.3dgs.planner"].OccPlanner = lambda occ: _ns(plan=PLAN)
    sys.modules["sim.3dgs.rule_oracle"] = _ns(RuleOraclePolicy=FakeRule)
    return names + ["sim.3dgs.rule_oracle"]


def run(cfg, arms, scene="sA", routes=None):
    """跑真实 driver.py（桩依赖）。返回 (rc, 快照 {相对路径: 文本})；快照在临时目录删除前取。"""
    routes = routes or tdp.ROUTES
    REC.update(runs=[], rules=[], guides=[], guarded=[])
    tdp.LOADED.clear()
    snap = {}
    with tempfile.TemporaryDirectory() as out:
        full = {"repo": HERE, "scenes_dir": out, "scenes": [scene], "k": 1, "seed0": 8000, "jitter": 0.2, "hold": 1.0,
                "step_frac": 0.5, "step_pad": 15, "guard_threshold_deg": 45.0, "route_lookahead_m": 1.5, "routes": [0, 1, 2],
                "arms": arms, "infer_bin": "x", "cf_sitl_proc": "y", **cfg}
        os.makedirs(os.path.join(out, scene))
        with open(os.path.join(out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(full, f)
        with open(os.path.join(out, scene, "scene.json"), "w", encoding="utf-8") as f:
            json.dump({"routes": routes}, f)
        names = install(routes)
        argv, sys.argv = sys.argv, ["driver.py", out, scene]
        try:
            with tdp.self_silence():
                try:
                    runpy.run_path(DRIVER, run_name="__main__")
                    rc = 0
                except SystemExit as e:
                    rc = e.code or 0
        finally:
            sys.argv = argv
            for n in names:
                sys.modules.pop(n, None)
        for root, _, files in os.walk(out):
            for fn in files:
                p = os.path.join(root, fn)
                with open(p, encoding="utf-8") as f:
                    snap[os.path.relpath(p, out)] = f.read()
    return rc, snap


def arm(name, base, guard, **extra):
    return {"name": name, "base": base, "ckpt": None, "guard": guard, **extra}


def jload(snap, rel):
    return json.loads(snap[rel])


def jlines(snap, rel):
    return [json.loads(x) for x in snap[rel].splitlines() if x.strip()]


EP = "{arm}/sA/r{r}/ep_00000/{f}"
DONE = "{arm}/sA/r{r}/done.json"


class TestDefaultsLocked(unittest.TestCase):
    """不给新键：driver 的产物、对 run_episode 的调用、done / metrics 的键与顺序，与改动前逐位一致。"""

    OLD_DONE_KEYS = ["arm", "scene", "route", "steps", "hold", "ckpt", "guard", "dynamics_profile", "episodes", "failures", "wall_s"]
    OLD_METRIC_KEYS = ["success_stop", "success_v2", "collision", "collide_first", "timeout", "steps", "reach_step", "collision_step",
                       "spl", "spl_stop", "final_wp_dist_m", "min_wp_dist_m", "start_pose", "arm", "scene", "route", "episode",
                       "seed", "stopped_on_goal", "wall_s", "guard"]
    OLD_RUN_KW = {"split": "test", "seed": 8000, "episode_id": 8000, "save_depth": False, "stop_on_collision": True,
                  "dynamics_path": None, "spawn_jitter": 0.2, "identity": {}, "split_policy": "diag-eval-fixed-test",
                  "dynamics": "cf_sitl"}

    def test_forward_arm(self):
        rc, snap = run({"routes": [0]}, [arm("forward", "forward", None)])
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(snap), sorted(["config.json", "sA/scene.json", DONE.format(arm="forward", r=0)]
                                              + [EP.format(arm="forward", r=0, f=f)
                                                 for f in ("actions.jsonl", "oracle.jsonl", "metrics.json", "trace.jsonl")]))
        done = jload(snap, DONE.format(arm="forward", r=0))
        self.assertEqual(list(done), self.OLD_DONE_KEYS)
        self.assertEqual((done["steps"], done["hold"], done["guard"], done["ckpt"]), (45, 1.0, None, None))
        m = jload(snap, EP.format(arm="forward", r=0, f="metrics.json"))
        self.assertEqual(list(m), self.OLD_METRIC_KEYS)
        self.assertIsNone(m["guard"])
        (run0,) = REC["runs"]
        self.assertEqual((run0["steps"], run0["hold"], run0["kw"]), (45, 1.0, self.OLD_RUN_KW))

    def test_route_guarded_arm(self):
        rc, snap = run({"routes": [1]}, [arm("fr", "forward", "route")])
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(k for k in snap if k.startswith("fr/")),
                         sorted([DONE.format(arm="fr", r=1)] + [EP.format(arm="fr", r=1, f=f)
                                                                for f in ("actions.jsonl", "oracle.jsonl", "metrics.json",
                                                                          "trace.jsonl", "guard.jsonl")]))
        done = jload(snap, DONE.format(arm="fr", r=1))
        self.assertEqual(list(done), self.OLD_DONE_KEYS)
        m = jload(snap, EP.format(arm="fr", r=1, f="metrics.json"))
        self.assertEqual(list(m), self.OLD_METRIC_KEYS)
        self.assertEqual(sorted(m["guard"]), sorted(["n_steps", "n_triggered", "n_rewritten", "n_steps_pre", "n_triggered_pre",
                                                     "n_rewritten_pre", "plan_failed"]))
        (g,) = REC["guides"]
        (gp,) = REC["guarded"]
        self.assertEqual((g.plan, g.lookahead, gp.thr, gp.target_fn), (PLAN, 1.5, 45.0, g.target))
        self.assertEqual(REC["runs"][0]["steps"], 100)      # ceil(170 × 0.5) + 15

    def test_trace_is_kept_without_drop_trace(self):
        _, snap = run({"routes": [0]}, [arm("forward", "forward", None)])
        self.assertIn(EP.format(arm="forward", r=0, f="trace.jsonl"), snap)


class TestRuleArms(unittest.TestCase):
    def test_rule_occ_route_builds_the_rule_policy_on_the_route_target(self):
        rc, snap = run({"routes": [1], "rule_params": {"r_safe": 0.25, "yaw_step_deg": 31.7}},
                       [arm("ro", "rule-occ", "route")])
        self.assertEqual(rc, 0)
        (rule,) = REC["rules"]
        (guide,) = REC["guides"]
        (run0,) = REC["runs"]
        self.assertEqual((rule.source, rule.params_arg), ("occ", {"r_safe": 0.25, "yaw_step_deg": 31.7}))
        self.assertIs(rule.occ, run0["occ"])                         # 与碰撞判定同一个 Occupancy
        self.assertEqual((guide.plan, guide.wp, guide.lookahead), (PLAN, run0["wp"], 1.5))
        self.assertEqual(rule.target_fn, guide.target)               # 航向参照 = 路线前视点
        self.assertIs(run0["policy"], rule)                          # 没有再套 guard 包装
        self.assertEqual(REC["guarded"], [])

    def test_rule_depth_source_and_goal_target(self):
        rc, _ = run({"routes": [0]}, [arm("rd", "rule-depth", "goal")])
        self.assertEqual(rc, 0)
        (rule,) = REC["rules"]
        self.assertEqual(rule.source, "depth")
        self.assertEqual(REC["guides"], [])                          # goal：不规划路线
        self.assertEqual(rule.target_fn(5.0, 5.0), REC["runs"][0]["wp"])     # 与当前位置无关，恒为 waypoint

    def test_rule_params_arm_overrides_global(self):
        run({"routes": [0], "rule_params": {"r_safe": 0.25, "d_look": 0.8, "yaw_step_deg": 31.7}},
            [arm("a", "rule-occ", "route"), arm("b", "rule-occ", "route", rule_params={"yaw_step_deg": 16.0, "d_look": 1.2})])
        self.assertEqual([r.params_arg for r in REC["rules"]],
                         [{"r_safe": 0.25, "d_look": 0.8, "yaw_step_deg": 31.7}, {"r_safe": 0.25, "d_look": 1.2, "yaw_step_deg": 16.0}])

    def test_no_rule_params_passes_an_empty_dict(self):
        run({"routes": [0]}, [arm("a", "rule-occ", "route")])
        self.assertEqual(REC["rules"][0].params_arg, {})

    def test_rule_arm_writes_rule_jsonl_and_metrics_rule(self):
        _, snap = run({"routes": [0], "rule_params": {"r_safe": 0.25}}, [arm("ro", "rule-occ", "route")])
        rows = jlines(snap, EP.format(arm="ro", r=0, f="rule.jsonl"))
        self.assertEqual([r["step"] for r in rows], [0, 1, 2])
        self.assertNotIn(EP.format(arm="ro", r=0, f="guard.jsonl"), snap)
        m = jload(snap, EP.format(arm="ro", r=0, f="metrics.json"))
        self.assertIsNone(m["guard"])
        # 桩的 oracle 只有 1 行且就在 waypoint 半径内：reach_step = 0，到点前只有第 0 步
        self.assertEqual(m["reach_step"], 0)
        self.assertEqual(m["rule"], {"source": "occ", "n_steps": 3, "entries": {"V1-fwd": 2, "V1-side": 1},
                                     "n_steps_pre": 1, "entries_pre": {"V1-fwd": 1}, "plan_failed": False,
                                     "params": {"r_safe": 0.25, "d_look": 1.2, "yaw_step_deg": 26.0}})

    def test_goal_arm_has_no_plan_flag(self):
        _, snap = run({"routes": [0]}, [arm("rg", "rule-occ", "goal")])
        self.assertIsNone(jload(snap, EP.format(arm="rg", r=0, f="metrics.json"))["rule"]["plan_failed"])

    def test_done_json_records_rule_source_and_overrides(self):
        _, snap = run({"routes": [0], "rule_params": {"r_safe": 0.25}}, [arm("ro", "rule-occ", "route")])
        done = jload(snap, DONE.format(arm="ro", r=0))
        self.assertEqual(list(done)[:11], TestDefaultsLocked.OLD_DONE_KEYS)     # 旧键与顺序不动，新键只追加在后面
        self.assertEqual(done["rule"], {"source": "occ", "overrides": {"r_safe": 0.25}})
        self.assertNotIn("step_mult", done)                                      # 乘数为 1 时不写

    def test_rule_arm_without_route_or_goal_is_rejected_before_loading_the_scene(self):
        with self.assertRaises(ValueError):
            run({"routes": [0]}, [arm("bad", "rule-occ", None)])
        self.assertEqual(tdp.LOADED, [])                              # 配置错在加载场景（约 200 s）之前就报


class TestHoldAndSteps(unittest.TestCase):
    def test_arm_hold_and_step_mult(self):
        rc, snap = run({"routes": [1], "step_cap": 100},
                       [arm("h10", "forward", "route"), arm("h05", "forward", "route", hold=0.5, step_mult=2)])
        self.assertEqual(rc, 0)
        by_arm = {("h05" if r["hold"] == 0.5 else "h10"): r for r in REC["runs"]}
        self.assertEqual((by_arm["h10"]["hold"], by_arm["h10"]["steps"]), (1.0, 100))
        self.assertEqual((by_arm["h05"]["hold"], by_arm["h05"]["steps"]), (0.5, 200))        # 封顶 100 之后再乘 2
        self.assertEqual(jload(snap, DONE.format(arm="h05", r=1))["hold"], 0.5)
        self.assertEqual(jload(snap, DONE.format(arm="h05", r=1))["steps"], 200)
        self.assertEqual(jload(snap, DONE.format(arm="h05", r=1))["step_mult"], 2)
        self.assertEqual(jload(snap, DONE.format(arm="h10", r=1))["hold"], 1.0)

    def test_step_mult_without_cap_multiplies_the_uncapped_budget(self):
        run({"routes": [0]}, [arm("h05", "forward", None, hold=0.5, step_mult=2)])
        self.assertEqual(REC["runs"][0]["steps"], 90)                    # (ceil(60 × 0.5) + 15) × 2

    def test_hold_override_reaches_rule_arm_and_same_seed_is_kept(self):
        run({"routes": [1]}, [arm("a", "rule-occ", "route"), arm("b", "rule-occ", "route", hold=0.5, step_mult=2)])
        a, b = REC["runs"]
        self.assertEqual((a["hold"], b["hold"]), (1.0, 0.5))
        self.assertEqual((a["kw"]["seed"], b["kw"]["seed"]), (8100, 8100))   # 同 seed -> 各臂同一出生位姿，配对


class TestDropTrace(unittest.TestCase):
    def test_drop_trace_removes_only_the_trace(self):
        _, snap = run({"routes": [0], "drop_trace": True}, [arm("ro", "rule-occ", "route")])
        self.assertNotIn(EP.format(arm="ro", r=0, f="trace.jsonl"), snap)
        for f in ("actions.jsonl", "oracle.jsonl", "metrics.json", "rule.jsonl"):
            self.assertIn(EP.format(arm="ro", r=0, f=f), snap)

    def test_drop_trace_false_keeps_it(self):
        _, snap = run({"routes": [0], "drop_trace": False}, [arm("ro", "rule-occ", "route")])
        self.assertIn(EP.format(arm="ro", r=0, f="trace.jsonl"), snap)


if __name__ == "__main__":
    unittest.main(verbosity=2, warnings="ignore")
