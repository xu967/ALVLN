#!/usr/bin/env python3
"""teacher-tol-3dgs 两个分析脚本（analyze.py、measure_turn_step.py）的单测：只用标准库，合成数据，不加载场景、不占 GPU。

  PYTHONDONTWRITEBYTECODE=1 python test_analyze.py
没设 TMPDIR 时临时目录落数据盘（根分区只剩 2 GB，不往 /tmp 写）。
覆盖：交替段判据、Wilson / 精确 McNemar（与旧报告里的数值对上）、每集读取与汇总（超时 / 含交替段的超时 / 碰撞 / SPL / 分桶）、
新旧配对、验收判决、回放归因（桩教师）、确定性回归的逐步比对、转角步分上下文的分布。
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


analyze = load("analyze")
measure = load("measure_turn_step")

S, F, L, R = 0, 1, 2, 3          # SAFE_NEUTRAL / FORWARD_SLOW / YAW_LEFT / YAW_RIGHT


class TestAltRuns(unittest.TestCase):
    def test_strict_alternation_of_at_least_six(self):
        self.assertEqual(analyze.alt_runs([L, R, L, R, L, R]), [(0, 5)])
        self.assertEqual(analyze.alt_runs([R, L, R, L, R, L, R]), [(0, 6)])

    def test_five_is_not_enough(self):
        self.assertEqual(analyze.alt_runs([L, R, L, R, L]), [])

    def test_same_direction_turns_do_not_alternate(self):
        self.assertEqual(analyze.alt_runs([L, L, L, L, L, L, L]), [])
        self.assertEqual(analyze.alt_runs([L, L, R, L, R, L, R, L]), [(1, 7)])      # 前面的重复 L 不算进段

    def test_forward_or_safe_breaks_the_run(self):
        self.assertEqual(analyze.alt_runs([L, R, L, R, F, L, R, L, R]), [])
        self.assertEqual(analyze.alt_runs([L, R, L, R, S, L, R, L, R]), [])

    def test_several_runs_and_min_run_parameter(self):
        seq = [L, R, L, R, L, R, F, R, L, R, L, R, L, R, L]
        self.assertEqual(analyze.alt_runs(seq), [(0, 5), (7, 14)])
        self.assertEqual(analyze.alt_runs([L, R, L, R], min_run=4), [(0, 3)])
        self.assertEqual(analyze.alt_runs([]), [])


class TestStats(unittest.TestCase):
    def test_exact_mcnemar_matches_reported_values(self):
        self.assertAlmostEqual(analyze.exact_mcnemar(28, 3), 4.649162e-06, places=10)      # sage3d 18° 诊断：p = 4.6e-6
        self.assertEqual(analyze.exact_mcnemar(1, 3), 0.625)                               # bearing-guard P3：b=1, c=3, p=0.625
        self.assertEqual(analyze.exact_mcnemar(0, 0), 1.0)
        self.assertEqual(analyze.exact_mcnemar(5, 5), 1.0)
        self.assertAlmostEqual(analyze.exact_mcnemar(10, 0), 0.001953125)

    def test_wilson(self):
        lo, hi = analyze.wilson(100, 120)
        self.assertAlmostEqual(lo, 0.7565456, places=6)
        self.assertAlmostEqual(hi, 0.8894409, places=6)
        self.assertEqual(analyze.wilson(0, 10)[0], 0.0)
        self.assertEqual(analyze.wilson(10, 10)[1], 1.0)
        self.assertTrue(all(math.isnan(v) for v in analyze.wilson(0, 0)))


# ---- 合成实验目录：<out>/<suite>/<arm>/<scene>/r<route>/{done.json, ep_*/{metrics.json, actions.jsonl}}
def write_episode(suite_dir, arm, scene, route, ep, seq, outcome, spl=0.0, manifest_tol=None):
    """seq = action_executed 序列；outcome = reach / collide / timeout。"""
    d = os.path.join(suite_dir, arm, scene, f"r{route}", "ep_%05d" % ep)
    os.makedirs(d, exist_ok=True)
    reach = outcome == "reach"
    rows = [{"step": i, "action_expert": a, "action_executed": a, "collision": outcome == "collide" and i == len(seq) - 1,
             "pose_before": {"x": float(i), "y": 0.0, "z": 2.0}, "yaw_before": 10.0 * i, "dyaw_deg": 0.0, "displacement_m": 0.1}
            for i, a in enumerate(seq)]
    with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    m = {"success_stop": reach, "success_v2": reach, "collision": outcome == "collide", "collide_first": outcome == "collide",
         "timeout": outcome == "timeout", "steps": len(seq), "reach_step": len(seq) - 1 if reach else None,
         "collision_step": len(seq) - 1 if outcome == "collide" else None, "spl": spl, "spl_stop": spl if reach else 0.0,
         "final_wp_dist_m": 0.4 if reach else 3.0, "min_wp_dist_m": 0.4, "arm": arm, "scene": scene, "route": route,
         "episode": ep, "seed": 8000 + 100 * route + ep}
    with open(os.path.join(d, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f)
    if manifest_tol is not None:
        with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"teacher_yaw_tol_deg": manifest_tol}, f)


def finish_route(suite_dir, arm, scene, route, k):
    d = os.path.join(suite_dir, arm, scene, f"r{route}")
    with open(os.path.join(d, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": [{"episode": i} for i in range(k)], "failures": []}, f)


CYCLE = [L, R] * 5                       # 10 步左右交替
GOOD = [L, F, F, F, F]                   # 一次转向后直飞


class Synthetic(unittest.TestCase):
    """室内合成：2 场景 × 每场景 2 路线 × K=3；旧臂 teacher 有 4 个超时（3 个含交替段）、1 个撞；新臂 teacher-tol 把其中 3 个超时变成到点。"""

    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.addCleanup(self._td.cleanup)
        self.out = self._td.name
        self.suite = os.path.join(self.out, "indoor")
        self.cfg = {"scenes": ["sA", "sB"], "routes": {"sA": [0, 1], "sB": [4]}, "k": 3, "hold": 1.0, "seed0": 8000,
                    "arms": [{"name": "teacher-tol", "base": "teacher", "ckpt": None, "guard": None},
                             {"name": "teacher", "base": "teacher", "ckpt": None, "guard": None}],
                    "new_arm": "teacher-tol", "old_arm": "teacher", "teacher_tol_deg": 18.9, "accept_sr": 0.8,
                    "accept_timeouts_max": 1,
                    "routes_selected": os.path.join(self.suite, "routes_selected.json")}
        os.makedirs(self.suite)
        with open(os.path.join(self.suite, "routes_selected.json"), "w", encoding="utf-8") as f:
            json.dump({"params": {"bucket_labels": ["<5 m", "5–10 m", ">=10 m"]},
                       "scenes": {"sA": [{"route": 0, "name": "a0", "astar_path_m": 3.0, "bucket": "<5 m"},
                                         {"route": 1, "name": "a1", "astar_path_m": 7.0, "bucket": "5–10 m"}],
                                  "sB": [{"route": 4, "name": "b4", "astar_path_m": 12.0, "bucket": ">=10 m"}]}}, f)
        with open(os.path.join(self.suite, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.cfg, f)
        # 旧臂：sA r0 三集 = 超时（交替）、超时（交替）、到点；sA r1 = 超时（无交替：原地悬停）、到点、到点；sB r4 = 超时（交替）、撞、到点
        old = {("sA", 0): [(CYCLE, "timeout"), (CYCLE, "timeout"), (GOOD, "reach")],
               ("sA", 1): [([S] * 8, "timeout"), (GOOD, "reach"), (GOOD, "reach")],
               ("sB", 4): [(CYCLE + [F], "timeout"), (GOOD, "collide"), (GOOD, "reach")]}
        # 新臂：sA r0 = 到点 × 3；sA r1 = 超时（无交替）、到点、到点；sB r4 = 到点、撞、到点（交替段在到点集里残留一个）
        new = {("sA", 0): [(GOOD, "reach"), (GOOD, "reach"), (GOOD, "reach")],
               ("sA", 1): [([S] * 8, "timeout"), (GOOD, "reach"), (GOOD, "reach")],
               ("sB", 4): [(CYCLE + [F], "reach"), (GOOD, "collide"), (GOOD, "reach")]}
        for arm, data, tol in (("teacher", old, None), ("teacher-tol", new, 18.9)):
            for (scene, route), eps in data.items():
                for i, (seq, outcome) in enumerate(eps):
                    write_episode(self.suite, arm, scene, route, i, seq, outcome, spl=0.5, manifest_tol=tol)
                finish_route(self.suite, arm, scene, route, 3)

    def test_load_and_summarize_old_arm(self):
        rows = analyze.load_episodes(self.suite, self.cfg, "teacher")
        self.assertEqual(len(rows), 9)
        a = analyze.summarize(rows, n_req=9)
        self.assertEqual((a["k_stop"], a["timeouts"], a["timeouts_alt"], a["episodes_alt"], a["collisions"]), (4, 4, 3, 3, 1))
        self.assertAlmostEqual(a["sr_stop"], 4 / 9)
        self.assertAlmostEqual(a["spl_stop"], 4 * 0.5 / 9)
        self.assertEqual(a["n"], 9)

    def test_new_arm_and_pairing(self):
        old = analyze.load_episodes(self.suite, self.cfg, "teacher")
        new = analyze.load_episodes(self.suite, self.cfg, "teacher-tol")
        a = analyze.summarize(new, n_req=9)
        self.assertEqual((a["k_stop"], a["timeouts"], a["timeouts_alt"], a["episodes_alt"]), (7, 1, 0, 1))   # 到点集里还留着 1 个交替段
        p = analyze.pair(new, old)
        self.assertEqual((p["pairs"], p["b"], p["c"]), (9, 3, 0))       # b = 新成旧败：sA r0 两集 + sB r4 一集
        self.assertAlmostEqual(p["p"], 0.25)                            # 2 × (1/8) = 0.25
        self.assertAlmostEqual(p["delta"], 3 / 9)

    def test_episodes_beyond_k_are_ignored(self):
        """冒烟（K_LIMIT）时旧臂的符号链接里集数比 config 的 k 多：只读 episode < k 的，不然分母小于分子。"""
        write_episode(self.suite, "teacher", "sA", 0, 3, GOOD, "reach", spl=0.5)
        write_episode(self.suite, "teacher", "sA", 0, 4, GOOD, "reach", spl=0.5)
        rows = analyze.load_episodes(self.suite, self.cfg, "teacher")
        self.assertEqual(len(rows), 9)
        self.assertEqual(sorted(k[2] for k in rows), [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertLessEqual(analyze.summarize(rows, n_req=9)["k_stop"], 9)

    def test_unfinished_routes_are_not_counted(self):
        os.remove(os.path.join(self.suite, "teacher", "sB", "r4", "done.json"))
        rows = analyze.load_episodes(self.suite, self.cfg, "teacher")
        self.assertEqual(len(rows), 6)
        self.assertEqual(analyze.done_routes(self.suite, self.cfg, "teacher"), {("sA", 0), ("sA", 1)})

    def test_buckets(self):
        with open(self.cfg["routes_selected"], encoding="utf-8") as f:
            rs = analyze.route_buckets(json.load(f))
        self.assertEqual(rs, {("sA", 0): "<5 m", ("sA", 1): "5–10 m", ("sB", 4): ">=10 m"})
        rows = analyze.load_episodes(self.suite, self.cfg, "teacher")
        done = analyze.done_routes(self.suite, self.cfg, "teacher")
        t = analyze.bucket_table(rows, rs, ["<5 m", "5–10 m", ">=10 m"], k=3, done=done)
        self.assertEqual(t["<5 m"], {"n": 3, "k_stop": 1, "timeouts": 2, "timeouts_alt": 2, "collisions": 0})
        self.assertEqual(t["5–10 m"], {"n": 3, "k_stop": 2, "timeouts": 1, "timeouts_alt": 0, "collisions": 0})
        self.assertEqual(t[">=10 m"], {"n": 3, "k_stop": 1, "timeouts": 1, "timeouts_alt": 1, "collisions": 1})

    def test_verdict_against_preregistered_threshold(self):
        self.assertEqual(analyze.verdict(0.92, 0.92), "达标")
        self.assertEqual(analyze.verdict(0.9166, 0.92), "未达标")
        self.assertEqual(analyze.verdict(0.80, 0.80), "达标")

    def test_timeout_verdict_is_an_upper_bound(self):
        self.assertEqual(analyze.verdict_max(4, 4), "达标")
        self.assertEqual(analyze.verdict_max(0, 4), "达标")
        self.assertEqual(analyze.verdict_max(5, 4), "未达标")

    def test_main_writes_report_and_json(self):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = analyze.main(["acceptance", self.out, "--suites", "indoor"])
        self.assertEqual(rc, 0)
        with open(os.path.join(self.out, "analysis", "acceptance.json"), encoding="utf-8") as f:
            s = json.load(f)["suites"]["indoor"]
        self.assertEqual(s["arms"]["teacher-tol"]["k_stop"], 7)
        self.assertEqual(s["arms"]["teacher"]["timeouts_alt"], 3)
        self.assertEqual((s["pair"]["b"], s["pair"]["c"]), (3, 0))
        self.assertEqual(s["accept_sr"], 0.8)
        self.assertEqual(s["verdict"], "未达标")                          # 7/9 = 0.78 < 0.80
        self.assertEqual((s["accept_timeouts_max"], s["verdict_timeouts"]), (1, "达标"))   # 新臂超时 1 <= 1
        self.assertEqual([e["episode"] for e in s["alt_episodes"]["teacher-tol"]], [0])
        with open(os.path.join(self.out, "analysis", "acceptance.md"), encoding="utf-8") as f:
            md = f.read()
        for head in ("到点成功", "超时", "含交替段", "碰撞", "SPL", "McNemar", "<5 m", "5–10 m", ">=10 m", "超时集数"):
            self.assertIn(head, md)


class StubOcc:
    def __init__(self, blocked):
        self.blocked = blocked

    def sweep(self, a, b):
        return self.blocked, None


class StubPlanner:
    """目标永远在 +x 方向很远处：误差 = −yaw。"""

    def __init__(self, blocked=False):
        self.occ = StubOcc(blocked)

    def plan(self, start, goal):
        return [start, goal]

    def aim_point(self, path, x, y, lookahead):
        return path[-1]


class TestReplay(unittest.TestCase):
    def test_branches_and_fidelity(self):
        planner = importlib_planner()
        wp = {"x": 1e4, "y": 0.0}
        teacher = planner.OccAstarTeacher(StubPlanner(), wp, tol_deg=12.0)
        actions = [
            {"pose_before": {"x": 0.0, "y": 0.0, "z": 2.0}, "yaw_before": -20.0, "action_expert": L},   # err +20 > 12：turn_err
            {"pose_before": {"x": 0.0, "y": 0.0, "z": 2.0}, "yaw_before": 0.0, "action_expert": F},     # 对准且不堵：forward
            {"pose_before": {"x": 0.0, "y": 0.0, "z": 2.0}, "yaw_before": 0.0, "action_expert": F},     # 记录说 F，下面用堵塞教师复算
        ]
        res = analyze.replay_branches(teacher, actions)
        self.assertEqual([r["branch"] for r in res], ["turn_err", "forward", "forward"])
        self.assertEqual([r["match"] for r in res], [True, True, True])
        blocked = planner.OccAstarTeacher(StubPlanner(blocked=True), wp, tol_deg=12.0)
        res = analyze.replay_branches(blocked, actions)
        self.assertEqual([r["branch"] for r in res], ["turn_err", "turn_blocked", "turn_blocked"])
        self.assertEqual([r["match"] for r in res], [True, False, False])           # 复算与记录不一致会被数出来

    def test_resolve_tol_prefers_the_recorded_value(self):
        self.assertEqual(analyze.resolve_tol({"teacher_yaw_tol_deg": 18.9}, 12.0), 18.9)     # 记录优先
        self.assertEqual(analyze.resolve_tol({}, 18.0), 18.0)                                # 旧代码产出的集没有键：用该臂在 config 里的值
        self.assertEqual(analyze.resolve_tol(None, None), 12.0)                              # 都没有 = 旧教师
        self.assertEqual(analyze.resolve_tol({"teacher_yaw_tol_deg": None}, None), 12.0)

    def test_classify_run(self):
        self.assertEqual(analyze.classify_run(["turn_err"] * 6), "只有航向误差超容差")
        self.assertEqual(analyze.classify_run(["turn_blocked"] * 6), "只有前向被堵转向")
        self.assertEqual(analyze.classify_run(["turn_err", "turn_blocked"] * 3), "两种都有")
        self.assertEqual(analyze.classify_run(["turn_err", "forward"]), "其他")


def importlib_planner():
    import sys
    root = os.path.abspath(os.path.join(HERE, "..", ".."))
    sys.path.insert(0, root)
    return importlib.import_module("sim.3dgs.planner")


class TestRegressCompare(unittest.TestCase):
    def rows(self, seq):
        return [{"step": i, "action_expert": a, "action_executed": a, "pose_before": {"x": 0.1 * i, "y": 0.0, "z": 2.0},
                 "yaw_before": 1.5 * i, "dyaw_deg": 0.0, "displacement_m": 0.1, "collision": False} for i, a in enumerate(seq)]

    def test_identical(self):
        r = analyze.compare_actions(self.rows([L, F, F]), self.rows([L, F, F]))
        self.assertEqual((r["identical"], r["first_diff"]), (True, None))

    def test_first_difference_is_located(self):
        a, b = self.rows([L, F, F]), self.rows([L, F, F])
        b[2]["pose_before"]["x"] += 1e-12
        r = analyze.compare_actions(a, b)
        self.assertFalse(r["identical"])
        self.assertEqual((r["first_diff"]["step"], r["first_diff"]["key"]), (2, "pose_before"))

    def test_different_length(self):
        r = analyze.compare_actions(self.rows([L, F, F]), self.rows([L, F]))
        self.assertFalse(r["identical"])
        self.assertEqual(r["first_diff"]["key"], "length")


class TestMeasure(unittest.TestCase):
    @staticmethod
    def row(ex, dyaw):
        return {"action_executed": ex, "dyaw_deg": dyaw}

    def test_contexts_and_sign(self):
        rows = [self.row(L, 31.0), self.row(L, 34.0), self.row(R, -27.0), self.row(F, 0.5), self.row(R, -31.5), self.row(S, 0.0),
                self.row(L, 32.0)]
        got = measure.turn_steps(rows)
        self.assertEqual(got, [("after_hover", 31.0), ("same_dir", 34.0), ("reversal", 27.0), ("after_forward", 31.5),
                               ("after_hover", 32.0)])

    def test_wrong_sign_is_kept_as_negative_not_dropped(self):
        got = measure.turn_steps([self.row(L, -3.0)])
        self.assertEqual(got, [("after_hover", -3.0)])

    def test_describe(self):
        d = measure.describe([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual((d["n"], d["median"], d["min"], d["max"], d["mean"]), (5, 30.0, 10.0, 50.0, 30.0))
        self.assertAlmostEqual(d["p10"], 14.0)                      # 线性插值：位置 0.4 → 10 + 0.4 × 10
        self.assertAlmostEqual(d["p90"], 46.0)
        self.assertIsNone(measure.describe([])["median"])


if __name__ == "__main__":
    unittest.main()
