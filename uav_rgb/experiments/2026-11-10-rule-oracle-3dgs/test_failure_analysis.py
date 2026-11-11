#!/usr/bin/env python3
"""dev_failure_analysis.py / formal_failure_analysis.py 的判定函数测试（合成集，只用标准库）。

运行：python experiments/2026-11-10-rule-oracle-3dgs/test_failure_analysis.py
"""
import importlib.util
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dfa = _load("dev_failure_analysis")
ffa = _load("formal_failure_analysis")

FWD, L, R = 1, 2, 3


def make_ep(entries, actions=None, ns=None, targets=None, xy=None, clears=None, success=False, collide_first=False, timeout=None,
            reach_step=None, collision_step=None, clr=None):
    """合成一集：entries 每步规则条目；actions 默认按条目给（more / cap 交替给 L R）；targets / xy 每步一项。"""
    n_steps = len(entries)
    actions = actions or [L if i % 2 == 0 else R for i in range(n_steps)]
    ns = ns or [0] * n_steps
    targets = targets or [[5.0, 5.0]] * n_steps
    xy = xy or [(0.0, 0.0)] * n_steps
    clears = clears or [[0, 0, 0]] * n_steps
    rule = [{"step": i, "err_deg": 0.0, "n": ns[i], "target": list(targets[i]), "action": actions[i], "entry": entries[i], "mem_n": 0,
             "clear": list(clears[i]), "src": "occ", "d_occ": [4.0] * 20, "d_depth": [4.0] * 20} for i in range(n_steps)]
    act = [{"step": i, "action_executed": actions[i], "collision": False, "collision_point": None,
            "pose_before": {"x": xy[i][0], "y": xy[i][1], "z": 1.2}, "yaw_before": 0.0, "clearance_m": (clr or [0.5] * n_steps)[i]} for i in range(n_steps)]
    if timeout is None:
        timeout = not success and not collide_first
    m = {"success_stop": success, "collision": collide_first, "collide_first": collide_first, "timeout": timeout, "steps": n_steps,
         "reach_step": reach_step, "collision_step": collision_step, "final_wp_dist_m": 3.0, "min_wp_dist_m": 2.5,
         "start_pose": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0}, "scene": "s", "route": 1, "episode": 0}
    return {"m": m, "rule": rule, "act": act}


MC = ["V1-more", "V1-cap"] * 10          # 20 步严格交替


class TestAltMoreCap(unittest.TestCase):
    def test_strict_alternation(self):
        self.assertEqual(dfa.alt_more_cap(MC), (19, 20))

    def test_back_in_the_middle_breaks_the_run(self):
        e = ["V1-more", "V1-cap", "V1-more", "V1-back", "V1-more", "V1-cap"]
        self.assertEqual(dfa.alt_more_cap(e), (3, 3))      # 互换对：more→cap、cap→more、back 之后的 more→cap 共 3 对；最长连续段是开头的 more, cap, more

    def test_pairs_count_adjacent_swaps_only(self):
        e = ["V1-more", "V1-more", "V1-cap", "V1-cap", "V1-more"]
        self.assertEqual(dfa.alt_more_cap(e)[0], 2)         # more→cap（第 2、3 项）、cap→more（第 4、5 项）；同条目相邻不算

    def test_no_more_cap(self):
        self.assertEqual(dfa.alt_more_cap(["V1-fwd", "V1-back"]), (0, 0))

    def test_longest_run_requires_swaps_not_just_more_or_cap(self):
        self.assertEqual(dfa.alt_more_cap(["V1-more", "V1-more", "V1-cap"]), (1, 2))      # 前两个都是 more，段从第 2 项重新数


class TestBackDir(unittest.TestCase):
    def test_signs(self):
        self.assertEqual(dfa.back_dir(2), R)       # n > 0 = 目标在右，回转 YAW_RIGHT
        self.assertEqual(dfa.back_dir(-1), L)
        self.assertIsNone(dfa.back_dir(0))


class TestCycleAndStall(unittest.TestCase):
    def test_cycle_true_for_20_alternating(self):
        self.assertTrue(dfa.ended_in_cycle(make_ep(MC)))

    def test_cycle_false_for_19_steps(self):
        self.assertFalse(dfa.ended_in_cycle(make_ep(MC[:19])))

    def test_cycle_false_when_one_step_is_not_more_cap(self):
        e = list(MC)
        e[10] = "V1-back"
        self.assertFalse(dfa.ended_in_cycle(make_ep(e)))

    def test_cycle_false_when_only_19_of_20_steps_alternate(self):
        e = ["V1-back"] + MC[:19]            # 末 20 步里第一步不是 more / cap，其余 19 步严格交替
        self.assertFalse(dfa.ended_in_cycle(make_ep(e)))

    def test_cycle_false_for_two_more_in_a_row(self):
        e = list(MC)
        e[11] = "V1-more"          # 第 10 项本来就是 more，两个 more 相邻
        self.assertFalse(dfa.ended_in_cycle(make_ep(e)))

    def test_tail_target_run_counts_from_the_end(self):
        t = [[1.0, 1.0]] * 3 + [[2.0, 2.0]] * 5
        self.assertEqual(dfa.tail_target_run(make_ep(["V1-fwd"] * 8, targets=t)["rule"]), 5)

    def test_target_run_rounds_to_two_decimals(self):
        t = [[2.001, 2.0]] * 4 + [[2.004, 2.0]] * 4
        self.assertEqual(dfa.tail_target_run(make_ep(["V1-fwd"] * 8, targets=t)["rule"]), 8)

    def test_stalled_needs_both_constant_target_and_near_final_pose(self):
        e = ["V1-back"] * 25
        near = make_ep(e, targets=[[1.0, 0.0]] * 25, xy=[(0.8, 0.0)] * 25)
        far = make_ep(e, targets=[[1.0, 0.0]] * 25, xy=[(-2.0, 0.0)] * 25)
        moving = make_ep(e, targets=[[float(i), 0.0] for i in range(25)], xy=[(float(i), 0.0) for i in range(25)])
        self.assertTrue(dfa.target_stalled(near))
        self.assertFalse(dfa.target_stalled(far))
        self.assertFalse(dfa.target_stalled(moving))

    def test_stalled_needs_at_least_20_unchanged_steps(self):
        t20 = dict(targets=[[1.0, 0.0]] * 20, xy=[(0.9, 0.0)] * 20)
        t19 = dict(targets=[[1.0, 0.0]] * 19, xy=[(0.9, 0.0)] * 19)
        self.assertTrue(dfa.target_stalled(make_ep(["V1-back"] * 20, **t20)))
        self.assertFalse(dfa.target_stalled(make_ep(["V1-back"] * 19, **t19)))

    def test_stalled_boundary_is_strictly_below_half_meter(self):
        e = ["V1-back"] * 25
        self.assertFalse(dfa.target_stalled(make_ep(e, targets=[[0.5, 0.0]] * 25, xy=[(0.0, 0.0)] * 25)))
        self.assertTrue(dfa.target_stalled(make_ep(e, targets=[[0.49, 0.0]] * 25, xy=[(0.0, 0.0)] * 25)))


class TestEpisodeReport(unittest.TestCase):
    def test_toward_and_away_follow_the_sign_of_n(self):
        # n > 0 = 目标在右：YAW_RIGHT 朝路线，YAW_LEFT 背离；n < 0 反之；n = 0 单列
        entries = ["V1-back"] * 5
        actions = [R, L, L, R, L]
        ns = [2, 2, -1, -1, 0]
        ep = make_ep(entries, actions=actions, ns=ns)
        x = dfa.episode_report(("s", 1), ep, {}, 20)
        self.assertEqual((x["toward"], x["away"], x["n_zero_turn"]), (2, 2, 1))     # R/n=2 朝，L/n=2 背，L/n=-1 朝，R/n=-1 背，L/n=0 单列

    def test_forward_steps_are_not_counted_as_turns(self):
        ep = make_ep(["V1-fwd", "V1-back"], actions=[FWD, L], ns=[0, -1])
        x = dfa.episode_report(("s", 1), ep, {}, 20)
        self.assertEqual((x["toward"], x["away"], x["n_zero_turn"]), (1, 0, 0))

    def test_longest_left_right_alternation_over_the_whole_episode(self):
        actions = [L, R, L, R, L, FWD, L, R]
        ep = make_ep(["V1-more"] * 8, actions=actions)
        x = dfa.episode_report(("s", 1), ep, {}, 20)
        self.assertEqual((x["alt_run_len"], x["alt_run_at"]), (5, 0))

    def test_bbox_and_min_clearance_use_only_the_last_steps(self):
        xy = [(10.0, 10.0)] * 5 + [(0.0, 0.0), (0.1, 0.3)]
        clr = [0.01] * 5 + [0.4, 0.3]
        ep = make_ep(["V1-more"] * 7, xy=xy, clr=clr)
        x = dfa.episode_report(("s", 1), ep, {}, 2)
        self.assertAlmostEqual(x["bbox"][0], 0.1)
        self.assertAlmostEqual(x["bbox"][1], 0.3)
        self.assertAlmostEqual(x["min_clr_tail"], 0.3)


class TestOutcomeAndNavEnd(unittest.TestCase):
    def test_outcome_labels(self):
        self.assertEqual(dfa.outcome(make_ep(["V1-fwd"], success=True)["m"]), "到点")
        self.assertEqual(dfa.outcome(make_ep(["V1-fwd"], collide_first=True, collision_step=0)["m"]), "撞在前")
        self.assertEqual(dfa.outcome(make_ep(["V1-fwd"])["m"]), "超时")

    def test_nav_end_is_the_earlier_of_reach_and_collision(self):
        self.assertEqual(dfa.nav_end({"steps": 50, "reach_step": 12, "collision_step": 30}), 12)
        self.assertEqual(dfa.nav_end({"steps": 50, "reach_step": 40, "collision_step": 30}), 30)
        self.assertEqual(dfa.nav_end({"steps": 50, "reach_step": None, "collision_step": None}), 49)


class TestFailureForm(unittest.TestCase):
    def test_success_is_not_a_failure_even_if_it_cycles_afterwards(self):
        ep = make_ep(MC, success=True, reach_step=2)
        self.assertEqual(ffa.failure_form(ep), "到点")

    def test_collide_first_wins_over_cycle(self):
        ep = make_ep(MC, collide_first=True, collision_step=19)
        self.assertEqual(ffa.failure_form(ep), "撞在前")

    def test_cycle_wins_over_stall_and_alternation(self):
        self.assertEqual(ffa.failure_form(make_ep(MC)), "2 步循环收尾")

    def test_stall_when_target_constant_and_near_but_not_cycle(self):
        ep = make_ep(["V1-back"] * 25, actions=[R] * 25, targets=[[1.0, 0.0]] * 25, xy=[(0.9, 0.0)] * 25)
        self.assertEqual(ffa.failure_form(ep), "目标点停滞")

    def test_other_alternation_when_run_of_six_but_not_stalled(self):
        entries = ["V1-fwd"] * 14 + ["V1-back"] * 11
        actions = [FWD] * 14 + [L, R, L, R, L, R, L, R, L, R, L]
        xy = [(float(i), 0.0) for i in range(25)]
        ep = make_ep(entries, actions=actions, xy=xy, targets=[[float(i), 5.0] for i in range(25)])
        self.assertEqual(ffa.failure_form(ep), "其他交替")

    def test_other_timeout(self):
        xy = [(float(i), 0.0) for i in range(25)]
        ep = make_ep(["V1-fwd"] * 25, actions=[FWD] * 25, xy=xy, targets=[[float(i), 5.0] for i in range(25)])
        self.assertEqual(ffa.failure_form(ep), "其他超时")

    def test_no_rule_log(self):
        ep = make_ep(["V1-fwd"] * 3)
        ep["rule"] = []
        self.assertEqual(ffa.failure_form(ep), "无规则日志")


class TestStepStats(unittest.TestCase):
    def test_all_vs_pre_split_at_nav_end(self):
        clears = [[1, 1, 1]] * 3 + [[0, 0, 0]] * 3
        entries = ["V1-fwd"] * 3 + ["V1-cap"] * 3
        ep = make_ep(entries, clears=clears, success=True, reach_step=2)
        s = ffa.step_stats([ep])
        self.assertEqual((s["all"]["n"], s["all"]["blk"], s["all"]["cap"]), (6, 3, 3))
        self.assertEqual((s["pre"]["n"], s["pre"]["blk"], s["pre"]["cap"]), (3, 0, 0))


class TestCollisionRow(unittest.TestCase):
    def test_fields_at_the_collision_step(self):
        ep = make_ep(["V1-fwd", "V1-fwd", "V1-more"], actions=[FWD, FWD, L], collide_first=True, collision_step=2, clr=[0.3, 0.2, 0.1])
        ep["m"]["start_pose"] = {"x": 1.0, "y": 1.0, "yaw_deg": 0.0}
        ep["act"][2]["collision"] = True
        ep["act"][2]["collision_point"] = [4.0, 5.0, 1.2]
        ep["rule"][1]["d_occ"] = [0.1] + [4.0] * 6 + [0.5] * 6 + [4.0] * 7        # 第 0 列近，但不在正前方 6 列里
        ep["rule"][1]["d_depth"] = [4.0] * 20
        r = ffa.collision_row(ep)
        self.assertEqual((r["step"], r["action"], r["entry"]), (2, "YAW_LEFT", "V1-more"))
        self.assertAlmostEqual(r["clr0"], 0.3)
        self.assertAlmostEqual(r["clr_hit"], 0.1)
        self.assertAlmostEqual(r["spawn_to_hit"], 5.0)
        self.assertAlmostEqual(r["occ_c_prev"], 0.5)        # 前一步 occ 正前方 6 列（第 7–12 列）最小
        self.assertAlmostEqual(r["dep_c_prev"], 4.0)


class TestArmStats(unittest.TestCase):
    def test_cycle_and_stall_counts_skip_successful_episodes(self):
        ok = make_ep(MC, success=True, reach_step=1)                         # 到点后徘徊成循环：不算
        bad = make_ep(MC)                                                    # 没到点的循环：算
        s = dfa.arm_stats({("a", 1): ok, ("a", 2): bad})
        self.assertEqual((s["cyc"], s["stall"], s["n"], s["k"]), (1, 0, 2, 1))


if __name__ == "__main__":
    unittest.main(verbosity=1)
