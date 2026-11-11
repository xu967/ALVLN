#!/usr/bin/env python3
"""summarize.py 的单测：分母口径（已写 done.json 的路线 × K，失败集算失败）、Wilson / 精确 McNemar、路线长度桶、失败集分类（含左右交替）、
规则条目占比、depth 与 occ 的列统计（中位数 / 90 分位 / 漏检）、预注册判决的边界、与旧实验同名臂的逐集对照。只用标准库，合成目录，不碰 GPU。

  PYTHONDONTWRITEBYTECODE=1 python test_summarize.py
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import summarize as sm  # noqa: E402

FWD = [1] * 10


def write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def make_class(root, cls, arms, scenes, routes, k, ep_fn, route_info=None, seed0=8000, lost=(), skip_done=()):
    """arms = [(名, base, guard)]；ep_fn(arm, scene, route, ep) -> dict(ok, coll, to, acts, steps, rule, cols)。
    lost = {(臂, 场景, 路线, 集)}：这一集没有 metrics.json，done.json 记失败；skip_done = {(臂, 场景, 路线)}：整条路线没有 done.json。"""
    cdir = os.path.join(root, cls)
    cfg = {"scenes": scenes, "routes": routes, "k": k, "hold": 1.0, "jitter": 0.2, "seed0": seed0, "guard_threshold_deg": 45.0,
           "route_lookahead_m": 1.5, "rule_params": {"r_safe": 0.25, "d_look": 1.2},
           "arms": [{"name": n, "base": b, "ckpt": None, "guard": g} for n, b, g in arms]}
    if route_info:
        write(os.path.join(cdir, "routes_selected.json"),
              {"params": {"bucket_labels": ["<5 m", "5–10 m", ">=10 m"]},
               "scenes": {s: [{"route": r, "name": f"n{r}", "astar_path_m": L, "bucket": b} for r, (L, b) in rs.items()]
                          for s, rs in route_info.items()}})
        cfg["routes_selected"] = os.path.join(cdir, "routes_selected.json")
    write(os.path.join(cdir, "config.json"), cfg)
    for name, base, _ in arms:
        for s in scenes:
            for r in (routes[s] if isinstance(routes, dict) else routes):
                if (name, s, r) in skip_done:
                    continue
                d = os.path.join(cdir, name, s, f"r{r}")
                fails = []
                for e in range(k):
                    if (name, s, r, e) in lost:
                        fails.append({"episode": e})
                        continue
                    x = ep_fn(name, s, r, e)
                    ok, coll, to = x.get("ok", False), x.get("coll", False), x.get("to", False)
                    m = {"success_stop": ok, "success_v2": ok and not coll, "collision": coll, "collide_first": coll and not ok, "timeout": to,
                         "steps": x.get("steps", 10), "reach_step": x.get("reach", 5 if ok else None),
                         "collision_step": x.get("hit", 7 if coll else None), "spl_stop": 1.0 if ok else 0.0,
                         "start_pose": x.get("pose", {"x": 1.0, "y": 2.0, "yaw_deg": 3.0}), "arm": name, "scene": s, "route": r,
                         "episode": e, "seed": seed0 + 100 * r + e}
                    if base.startswith("rule-"):
                        m["rule"] = x.get("rule", {"n_steps": 10, "entries": {"V1-fwd": 10}, "n_steps_pre": 5, "entries_pre": {"V1-fwd": 5}})
                    ed = os.path.join(d, f"ep_{e:05d}")
                    write(os.path.join(ed, "metrics.json"), m)
                    write_jsonl(os.path.join(ed, "actions.jsonl"), [{"step": i, "action_executed": a} for i, a in enumerate(x.get("acts", FWD))])
                    if "cols" in x:
                        write_jsonl(os.path.join(ed, "rule.jsonl"), [{"step": i, "d_occ": o, "d_depth": dd} for i, (o, dd) in enumerate(x["cols"])])
                write(os.path.join(d, "done.json"), {"failures": fails, "episodes": [{"episode": e} for e in range(k) if (name, s, r, e) not in lost]})
    return cdir


def run_main(root, *extra):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = sm.main([root, *extra])
    return rc, out.getvalue(), err.getvalue()


ARMS = [("forward+route", "forward", "route"), ("rule-occ+route", "rule-occ", "route"), ("rule-depth+route", "rule-depth", "route")]


class TestBasics(unittest.TestCase):
    def test_percentile_linear_interpolation(self):
        self.assertEqual(sm.percentile([1, 2, 3, 4], 0.5), 2.5)
        self.assertAlmostEqual(sm.percentile([1, 2, 3, 4], 0.9), 3.7)
        self.assertEqual(sm.percentile([5], 0.9), 5)
        self.assertIsNone(sm.percentile([], 0.5))

    def test_exact_mcnemar_never_switches_to_chi_square(self):
        self.assertEqual(sm.exact_mcnemar(0, 0), 1.0)
        self.assertAlmostEqual(sm.exact_mcnemar(7, 1), 2 * (1 + 8) / 256)
        b, c = 20, 5                                    # b + c = 25：tools/stats_report 会在这里换卡方，这里仍要精确
        want = 2 * sum(math.comb(25, i) for i in range(6)) / 2 ** 25
        self.assertAlmostEqual(sm.exact_mcnemar(b, c), want)
        self.assertEqual(sm.exact_mcnemar(3, 3), 1.0)

    def test_alt_runs_requires_six_strict_alternations(self):
        L, R = sm.YAW_LEFT, sm.YAW_RIGHT
        self.assertEqual(sm.alt_runs([L, R, L, R, L, R]), [(0, 5)])
        self.assertEqual(sm.alt_runs([L, R, L, R, L]), [])                        # 5 步不够
        self.assertEqual(sm.alt_runs([L, L, R, L, R, L, R]), [(1, 6)])             # 同向重复断开前一段
        self.assertEqual(sm.alt_runs([L, R, L, 1, R, L, R, L, R, L]), [(4, 9)])    # 前进动作断开
        self.assertEqual(sm.alt_runs([L, R, L, R, L, R, R, L, R, L, R, L]), [(0, 5), (6, 11)])
        self.assertEqual(sm.alt_runs([L, 4, R, 4, L, 4, R, 4, L, 4, R]), [])      # 非转向动作不算交替

    def test_wilson_matches_reference_values(self):
        lo, hi = sm.wilson_ci(8, 10)
        self.assertAlmostEqual(lo, 0.4902, places=3)
        self.assertAlmostEqual(hi, 0.9433, places=3)

    def test_column_stats_known_columns(self):
        # (occ, depth)：|差| = 0, 0.1, 0.3, 0.5, 0, 0.3。0.5 − 0.2 在浮点里正好等于 0.3，两个方向的「含等号」都靠这两对锁住
        pairs = [(1.0, 1.0), (1.0, 1.1), (0.2, 0.5), (2.0, 1.5), (4.0, 4.0), (0.5, 0.2)]
        st = sm.column_stats(pairs)
        self.assertEqual(st["n"], 6)
        self.assertAlmostEqual(st["median"], 0.2)                                   # 排序 [0, 0, 0.1, 0.3, 0.3, 0.5]，pos = 2.5
        self.assertAlmostEqual(st["p90"], 0.4)                                      # pos = 4.5：(0.3 + 0.5) / 2
        self.assertAlmostEqual(st["depth_farther"], 1 / 6)                          # 只有 (0.2, 0.5) 的 depth − occ >= 0.3（含等号）
        self.assertAlmostEqual(st["occ_farther"], 2 / 6)                            # (2.0, 1.5) 与 (0.5, 0.2)（含等号）
        self.assertEqual(sm.obstacle_cols(pairs), [q for q in pairs if q != (4.0, 4.0)])   # 两侧都是 4.0 的列没有障碍
        self.assertEqual(sm.obstacle_cols([(4.0, 3.0), (3.0, 4.0)]), [(4.0, 3.0), (3.0, 4.0)])   # 只要一侧低于上限就算
        self.assertIsNone(sm.column_stats([])["median"])


class TestClass(unittest.TestCase):
    def build(self, d, ep_fn, **kw):
        info = kw.pop("route_info", {"sA": {0: (3.0, "<5 m"), 1: (12.0, ">=10 m")}})
        make_class(d, "indoor", kw.pop("arms", ARMS), ["sA"], {"sA": [0, 1]}, 2, ep_fn, route_info=info, **kw)
        data = sm.load_class(os.path.join(d, "indoor"))
        return data, sm.summarize_class("indoor", data, [])

    def test_denominator_is_done_routes_times_k_and_lost_episodes_fail(self):
        def ep(arm, s, r, e):
            return {"ok": True}
        with tempfile.TemporaryDirectory() as d:
            data, res = self.build(d, ep, lost={("rule-occ+route", "sA", 0, 1)}, skip_done={("rule-depth+route", "sA", 1)})
            a = res["overall"]["rule-occ+route"]
            self.assertEqual((a["n_req"], a["n"], a["failed"], a["k_stop"]), (4, 3, 1, 3))      # 丢的那集算失败
            self.assertAlmostEqual(a["sr_stop"], 3 / 4)
            self.assertEqual(a["ci_stop"], sm.wilson_ci(3, 4))                                      # 区间用请求数 4，不是有结果的 3
            b = res["overall"]["rule-depth+route"]
            self.assertEqual((b["n_req"], b["n"], b["k_stop"]), (2, 2, 2))                        # 半条路线（没 done.json）不进分母
            self.assertEqual(res["failed"], [{"arm": "rule-occ+route", "scene": "sA", "route": 0, "episode": 1}])

    def test_collision_counts_whole_episode_but_collide_first_only_before_reach(self):
        def ep(arm, s, r, e):
            if (r, e) == (0, 0):
                return {"ok": True, "coll": True, "reach": 3, "hit": 8}          # 到点后又撞：成功、碰撞，但不是撞在前
            if (r, e) == (0, 1):
                return {"coll": True, "hit": 4}                                    # 没到点就撞
            return {"ok": True}
        with tempfile.TemporaryDirectory() as d:
            _, res = self.build(d, ep)
            a = res["overall"]["rule-occ+route"]
            self.assertEqual(a["k_stop"], 3)
            self.assertAlmostEqual(a["collision"], 2 / 4)
            self.assertAlmostEqual(a["collide_first"], 1 / 4)
            self.assertEqual(a["k_coll"], 2)
            self.assertEqual(a["k_v2"], 2)                                          # 过点后再撞，v2 算失败
            row = next(p for p in res["prereg"] if p["item"].startswith("碰撞率"))
            self.assertEqual((row["k"], row["n"]), (2, 4))                          # 预注册的碰撞率用整集碰撞数（含到点后），不是撞在前

    def test_buckets_use_routes_selected_and_done_routes_only(self):
        def ep(arm, s, r, e):
            return {"ok": r == 0}                                                   # r0（<5 m）全成，r1（>=10 m）全败
        with tempfile.TemporaryDirectory() as d:
            _, res = self.build(d, ep, skip_done={("rule-depth+route", "sA", 1)})
            self.assertEqual(res["buckets"]["<5 m|rule-occ+route"]["k_stop"], 2)
            self.assertEqual(res["buckets"][">=10 m|rule-occ+route"]["n_req"], 2)
            self.assertEqual(res["buckets"][">=10 m|rule-occ+route"]["k_stop"], 0)
            self.assertEqual(res["buckets"][">=10 m|rule-depth+route"]["n_req"], 0)   # 这条路线没 done.json：桶里该臂没有请求
            self.assertEqual(res["buckets"]["5–10 m|rule-occ+route"]["n_req"], 0)

    def test_outdoor_style_config_routes_list_without_route_info(self):
        def ep(arm, s, r, e):
            return {"ok": r == 0}
        with tempfile.TemporaryDirectory() as d:
            make_class(d, "outdoor", ARMS, ["g1", "g2"], [0, 1], 2, ep, seed0=7000)     # 室外：所有场景共用一个路线列表，没有 routes_selected
            res = sm.summarize_class("outdoor", sm.load_class(os.path.join(d, "outdoor")), [])
            a = res["overall"]["rule-occ+route"]
            self.assertEqual((a["n_req"], a["k_stop"]), (8, 4))                         # 2 场景 × 2 路线 × 2 集
            self.assertEqual(res["per_scene"]["rule-occ+route|g2"]["n_req"], 4)
            self.assertEqual(res["buckets"], {})                                         # 没有 routes_selected：不出桶
            self.assertEqual({p["item"] for p in res["prereg"]}, {"到点成功 SR(stop)", "碰撞率（整集任一步碰撞）"})

    def test_failure_classes_split_collision_and_timeouts_by_alternation(self):
        L, R = sm.YAW_LEFT, sm.YAW_RIGHT

        def ep(arm, s, r, e):
            return {(0, 0): {"coll": True, "hit": 4}, (0, 1): {"to": True, "acts": [L, R] * 5},
                    (1, 0): {"to": True, "acts": [1, L, L, R, R, 1, 1, 1]}, (1, 1): {"ok": True}}[(r, e)]
        with tempfile.TemporaryDirectory() as d:
            _, res = self.build(d, ep)
            c = res["fail"]["rule-occ+route"]["all"]
            self.assertEqual(c, {"撞在前": 1, "超时·含交替": 1, "超时·无交替": 1, "超时·无动作序列": 0, "其他": 0})
            self.assertEqual(res["fail"]["rule-occ+route"]["<5 m"]["超时·含交替"], 1)
            self.assertEqual(res["fail"]["rule-occ+route"][">=10 m"]["超时·无交替"], 1)
            self.assertEqual(res["fail"]["rule-occ+route"][">=10 m"]["撞在前"], 0)

    def test_failure_class_other_and_missing_actions(self):
        m = {"success_stop": False, "collide_first": False, "timeout": False, "_dir": "/nonexistent"}
        self.assertEqual(sm.classify(m), "其他")
        m["timeout"] = True
        self.assertEqual(sm.classify(m), "超时·无动作序列")                           # 读不到动作序列不猜
        self.assertIsNone(sm.classify({"success_stop": True, "collide_first": False, "timeout": False, "_dir": "/x"}))

    def test_paired_counts_only_keys_in_both_arms(self):
        def ep(arm, s, r, e):
            if arm == "rule-occ+route":
                return {"ok": (r, e) != (1, 1)}                                      # A 只有 (1, 1) 败
            return {"ok": (r, e) in ((1, 1), (0, 0))}                                  # B 只有 (0, 0)、(1, 1) 成
        with tempfile.TemporaryDirectory() as d:
            make_class(d, "indoor", ARMS, ["sA"], {"sA": [0, 1, 2]}, 2, ep, lost={("forward+route", "sA", 0, 1)})   # B 丢一集：配对数 5
            res = sm.summarize_class("indoor", sm.load_class(os.path.join(d, "indoor")), [])
            t = next(p for p in res["pairs"] if p["id"] == "M1")
            self.assertEqual((t["a"], t["b"]), ("rule-occ+route", "forward+route"))
            # (0,0) 两臂都成；(1,0)、(2,0)、(2,1) A 成 B 败；(1,1) A 败 B 成
            self.assertEqual((t["pairs"], t["b_count"], t["c_count"]), (5, 3, 1))
            self.assertAlmostEqual(t["delta_sr"], (4 - 2) / 5)                        # SR(A) − SR(B)，A 在这 5 集成 4 个、B 成 2 个
            self.assertAlmostEqual(t["p"], 2 * (1 + 4) / 16)                           # n = 4，min(b, c) = 1
            self.assertTrue(t["main"])
            self.assertFalse(next(p for p in res["pairs"] if p["id"] == "x1")["main"])    # 补充比较不是预注册的主比较
            self.assertIn("M2", [p["id"] for p in res["pairs"]])
            self.assertNotIn("x3", [p["id"] for p in res["pairs"]])                  # 没有 @0.5 臂，补充比较不出现

    def test_spawn_pose_max_diff_across_arms(self):
        def ep(arm, s, r, e):
            if arm == "rule-occ+route":
                return {"ok": True, "pose": {"x": 1.5, "y": 2.0, "yaw_deg": 203.0}}   # 与其余臂：x 差 0.5，航向差 200° → 2.0
            return {"ok": True}
        with tempfile.TemporaryDirectory() as d:
            make_class(d, "indoor", ARMS, ["sA"], {"sA": [0]}, 1, ep)
            res = sm.summarize_class("indoor", sm.load_class(os.path.join(d, "indoor")), [])
            self.assertAlmostEqual(res["spawn_pair_max_diff"], 2.0)
        with tempfile.TemporaryDirectory() as d:
            make_class(d, "indoor", ARMS, ["sA"], {"sA": [0]}, 1, lambda *a: {"ok": True})
            res = sm.summarize_class("indoor", sm.load_class(os.path.join(d, "indoor")), [])
            self.assertEqual(res["spawn_pair_max_diff"], 0.0)

    def test_entry_shares_pool_steps_not_episode_means(self):
        def ep(arm, s, r, e):
            n = 10 if e == 0 else 30                                                # 两集步数不同：占比要按总步数池化
            return {"rule": {"n_steps": n, "entries": {"V1-fwd": n // 2, "V1-back": n - n // 2}, "n_steps_pre": 4,
                             "entries_pre": {"V1-fwd": 4}}}
        with tempfile.TemporaryDirectory() as d:
            _, res = self.build(d, ep)
            e = res["entries"]["rule-occ+route"]
            self.assertEqual((e["n_steps"], e["n_steps_pre"]), (80, 16))
            self.assertEqual(e["whole"], {"V1-fwd": 40, "V1-back": 40})
            self.assertEqual(e["pre"], {"V1-fwd": 16})
            self.assertNotIn("forward+route", res["entries"])                       # 非规则臂没有规则条目

    def test_depth_occ_reads_rule_jsonl_and_cuts_at_nav_end(self):
        cols = [([1.0, 4.0], [1.0, 4.0]), ([1.0, 2.0], [1.4, 2.0]), ([0.5, 4.0], [0.5, 3.0])]   # 步 0、1、2
        def ep(arm, s, r, e):
            return {"ok": True, "reach": 1, "cols": cols}                           # 到点步 = 1：pre 只含步 0、1
        with tempfile.TemporaryDirectory() as d:
            data, res = self.build(d, ep)
            v = res["depth_occ"]["rule-occ+route"]
            self.assertEqual(len(v["all"]), 4 * 6)                                   # 4 集 × 3 步 × 2 列
            self.assertEqual(len(v["pre"]), 4 * 4)                                   # 4 集 × 2 步 × 2 列
            st = sm.column_stats(v["pre"])
            self.assertAlmostEqual(st["depth_farther"], 4 / 16)                      # 每集步 1 的 (1.0, 1.4)
            st_obs = sm.column_stats(sm.obstacle_cols(v["all"]))
            self.assertEqual(st_obs["n"], 4 * 5)                                     # 只有 (4.0, 4.0) 这一列被排除
            self.assertNotIn("forward+route", res["depth_occ"])


    def test_depth_occ_timeout_episode_pre_is_whole_episode(self):
        cols = [([1.0, 2.0], [1.0, 2.0])] * 3
        def ep(arm, s, r, e):
            return {"to": True, "steps": 3, "cols": cols}                           # 没到点没撞：末步 = steps - 1 = 2，3 步都算导航阶段
        with tempfile.TemporaryDirectory() as d:
            _, res = self.build(d, ep)
            v = res["depth_occ"]["rule-occ+route"]
            self.assertEqual((len(v["all"]), len(v["pre"])), (4 * 6, 4 * 6))

    def test_depth_occ_cuts_at_first_collision_when_not_reached(self):
        cols = [([1.0, 2.0], [1.0, 2.0])] * 4
        def ep(arm, s, r, e):
            return {"coll": True, "hit": 1, "steps": 4, "cols": cols}               # 第 1 步撞：pre = 步 0、1
        with tempfile.TemporaryDirectory() as d:
            _, res = self.build(d, ep)
            v = res["depth_occ"]["rule-occ+route"]
            self.assertEqual((len(v["all"]), len(v["pre"])), (4 * 8, 4 * 4))


class TestPrereg(unittest.TestCase):
    def verdict(self, k_stop, k_coll, k_long, n=20, cls="indoor", n_long=5):
        """主臂：n 集里 k_stop 集到点、k_coll 集有碰撞（含到点后再撞）；>=10 m 桶 n_long 集里 k_long 集到点。"""
        res = {"overall": {sm.MAIN_ARM: {"n_req": n, "k_stop": k_stop, "k_coll": k_coll}},
               "buckets": {f"{sm.LONG_BUCKET}|{sm.MAIN_ARM}": {"k_stop": k_long, "n_req": n_long}}}
        return {p["item"]: p for p in sm.prereg(cls, res, None)}

    def test_indoor_thresholds_are_inclusive_where_card_says_not_below(self):
        v = self.verdict(16, 3, 6, n_long=10)           # 0.80、0.60、0.15 正好在线上：都算达成
        self.assertTrue(v["到点成功 SR(stop)"]["met"])
        self.assertTrue(v[f"{sm.LONG_BUCKET} 路线的到点成功"]["met"])
        self.assertTrue(v["碰撞率（整集任一步碰撞）"]["met"])
        self.assertFalse(v["证伪条件：SR(stop) 低于 0.70"]["met"])
        v = self.verdict(15, 4, 5, n_long=10)           # 0.75、0.50、0.20：三项都未达成，也没到证伪线
        self.assertFalse(v["到点成功 SR(stop)"]["met"])
        self.assertFalse(v[f"{sm.LONG_BUCKET} 路线的到点成功"]["met"])
        self.assertFalse(v["碰撞率（整集任一步碰撞）"]["met"])
        self.assertFalse(v["证伪条件：SR(stop) 低于 0.70"]["met"])

    def test_falsification_is_strictly_below_070(self):
        self.assertFalse(self.verdict(14, 0, 5)["证伪条件：SR(stop) 低于 0.70"]["met"])   # 0.70 正好：没有低于
        self.assertTrue(self.verdict(13, 0, 5)["证伪条件：SR(stop) 低于 0.70"]["met"])    # 0.65

    def test_outdoor_has_its_own_thresholds_and_no_falsification_or_bucket(self):
        v = self.verdict(17, 2, 0, cls="outdoor")        # 0.85 ≥ 0.84，碰撞 0.10 ≤ 0.10
        self.assertEqual(set(v), {"到点成功 SR(stop)", "碰撞率（整集任一步碰撞）"})
        self.assertTrue(v["到点成功 SR(stop)"]["met"])
        self.assertTrue(v["碰撞率（整集任一步碰撞）"]["met"])
        v = self.verdict(16, 3, 0, cls="outdoor")        # 0.80 < 0.84，碰撞 0.15 > 0.10
        self.assertFalse(v["到点成功 SR(stop)"]["met"])
        self.assertFalse(v["碰撞率（整集任一步碰撞）"]["met"])

    def test_ci_straddle_flag(self):
        v = self.verdict(16, 3, 3)                       # 16/20 = 0.80，Wilson 区间 [0.58, 0.92] 跨过 0.80
        self.assertTrue(v["到点成功 SR(stop)"]["ci_straddles"])
        v = self.verdict(20, 0, 5)                       # 20/20，Wilson 下界 0.84 > 0.80
        self.assertFalse(v["到点成功 SR(stop)"]["ci_straddles"])

    def test_no_main_arm_gives_no_rows(self):
        self.assertEqual(sm.prereg("indoor", {"overall": {}, "buckets": {}}, None), [])


class TestOld(unittest.TestCase):
    def test_rerun_agreement_and_reference_arms(self):
        def new_ep(arm, s, r, e):
            return {"ok": e == 0, "steps": 12, "reach": 4 if e == 0 else None, "pose": {"x": 1.0, "y": 2.0, "yaw_deg": 3.0}}

        def old_ep(arm, s, r, e):
            ok = e == 0 or (arm == "forward+route" and (r, e) == (1, 1))             # 旧 forward+route 比新的多成一集
            return {"ok": ok, "coll": (r, e) == (0, 1), "steps": 12, "reach": 4 if ok else None,     # 旧的 (0, 1) 还撞了一次
                    "pose": {"x": 1.0, "y": 2.0, "yaw_deg": 3.0 if e == 0 else 3.5}}
        with tempfile.TemporaryDirectory() as d:
            new = os.path.join(d, "new")
            old = os.path.join(d, "old")
            make_class(new, "indoor", ARMS[:1], ["sA"], {"sA": [0, 1]}, 2, new_ep)
            make_class(old, "indoor", [("forward+route", "forward", "route"), ("teacher", "forward", None)], ["sA"], {"sA": [0, 1, 2]}, 2, old_ep)
            data = sm.load_class(os.path.join(new, "indoor"))
            o = sm.old_comparison(data["rows"], sm.load_old(os.path.join(old, "indoor")))
            r = o["rerun"]
            self.assertEqual((r["matched"], r["success_equal"], r["k_new"], r["k_old"]), (4, 3, 2, 3))
            self.assertEqual(r["collision_equal"], 3)                                  # (0, 1) 旧撞新没撞
            self.assertEqual(r["trajectory_equal"], 2)                                 # (0, 1) 多了撞步，(1, 1) 多了到点步
            self.assertEqual(r["start_pose_equal"], 2)                                 # e == 1 的两集旧出生航向 3.5
            self.assertEqual(o["reference"]["teacher"]["n_req"], 4)                    # 只取同键的集，路线 2 不进
            self.assertEqual(o["reference"]["forward+route"]["k_stop"], 3)

    def test_missing_old_dir_gives_none_not_crash(self):
        with tempfile.TemporaryDirectory() as d:
            make_class(d, "indoor", ARMS[:1], ["sA"], {"sA": [0]}, 1, lambda *a: {"ok": True})
            data = sm.load_class(os.path.join(d, "indoor"))
            self.assertIsNone(sm.old_comparison(data["rows"], sm.load_old(None))["rerun"])
            self.assertIsNone(sm.old_comparison(data["rows"], sm.load_old(os.path.join(d, "nope")))["rerun"])


class TestMain(unittest.TestCase):
    def test_main_writes_summary_files_and_handles_one_class(self):
        def ep(arm, s, r, e):
            return {"ok": True, "rule": {"n_steps": 10, "entries": {"V1-fwd": 6, "V1-more": 4}, "n_steps_pre": 5, "entries_pre": {"V1-fwd": 5}},
                    "cols": [([1.0, 4.0], [1.3, 4.0])] * 4}
        with tempfile.TemporaryDirectory() as d:
            make_class(d, "indoor", ARMS, ["sA"], {"sA": [0, 1]}, 2, ep, route_info={"sA": {0: (3.0, "<5 m"), 1: (12.0, ">=10 m")}})
            rc, out, _ = run_main(d)
            self.assertEqual(rc, 0)
            with open(os.path.join(d, "summary.md"), encoding="utf-8") as f:
                md = f.read()
            with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
                js = json.load(f)
            self.assertEqual(out.strip(), md.strip())
            for need in ("预注册判决", "逐集配对", "失败集分类", "规则条目触发占比", "depth 与 occ 列距离一致性", "与旧实验同名臂", "按路线长度分桶"):
                self.assertIn(need, md)
            self.assertNotIn("室外（3DGS 四场景）", md)                                 # 没有 outdoor 目录就不出室外小节
            self.assertEqual(js["indoor"]["overall"]["rule-occ+route"]["k_stop"], 4)
            self.assertAlmostEqual(js["indoor"]["depth_occ"]["rule-occ+route"]["all"]["all_cols"]["depth_farther"], 0.5)
            self.assertAlmostEqual(js["indoor"]["depth_occ"]["rule-occ+route"]["all"]["obstacle_cols"]["depth_farther"], 1.0)

    def test_main_marks_preregistered_comparisons_only_for_indoor(self):
        def ep(arm, s, r, e):
            return {"ok": e == 0}
        with tempfile.TemporaryDirectory() as d:
            make_class(d, "indoor", ARMS, ["sA"], {"sA": [0, 1]}, 2, ep)
            make_class(d, "outdoor", ARMS, ["g1"], [0, 1], 2, ep, seed0=7000)
            rc, out, _ = run_main(d)
            self.assertEqual(rc, 0)
            self.assertEqual(out.count("| 主比较 |"), 2)                                # 室内 M1、M2；室外同名比较写「补充（未预注册）」
            self.assertIn("室外（3DGS 四场景）", out)

    def test_main_without_any_class_fails_with_rc2(self):
        with tempfile.TemporaryDirectory() as d:
            rc, _, err = run_main(d)
            self.assertEqual(rc, 2)
            self.assertIn("config.json", err)

    def test_main_reports_old_comparison_when_dirs_given(self):
        def ep(arm, s, r, e):
            return {"ok": True}
        with tempfile.TemporaryDirectory() as d:
            make_class(os.path.join(d, "new"), "indoor", ARMS[:1], ["sA"], {"sA": [0]}, 2, ep)
            make_class(os.path.join(d, "old"), "indoor", ARMS[:1], ["sA"], {"sA": [0]}, 2, ep)
            rc, out, _ = run_main(os.path.join(d, "new"), "--old-indoor", os.path.join(d, "old", "indoor"))
            self.assertEqual(rc, 0)
            self.assertIn("同键集 2", out)
            self.assertIn("到点判定一致 2/2", out)


if __name__ == "__main__":
    unittest.main()
