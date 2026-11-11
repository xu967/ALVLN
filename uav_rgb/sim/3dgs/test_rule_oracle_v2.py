#!/usr/bin/env python3
"""rule_oracle.py 的 rule_version="v2" 接线单测（任务卡 K3）：纯 Python + numpy，不需要 torch / GPU。

用 3dgs-venv 或 .venv-tools 跑：
  TMPDIR=$DATA_ROOT/uav_rgb/tmp $DATA_ROOT/uav_rgb/3dgs-venv/bin/python sim/3dgs/test_rule_oracle_v2.py
覆盖：
  1. 版本开关：缺省 = v1 且行为、日志键、条目计数、版本串与 K1 相同；显式 v1 与缺省逐步相同；v2 的 planner_name / version / 参数表 / 条目表；
     未知键按所选版本的参数表判；v2 的 levels 不逐级收紧在构造时报错；rule_version 不认识报错；
  2. v2 的符号与规则调用：目标在左 → n < 0、e < 0、YAW_LEFT（V2-back0）；目标在右反之；e = −err / yaw_step（浮点，不取整）、n = round(e)；
     规则调用与 rule_policy.rule_v2_step 逐位相同（跨步的记忆与禁忌表）；上一步返回的动作作为 prev_action 传入（「只许静止起步」的级据此判断）；
     永不 SAFE_NEUTRAL；reset 清禁忌表与上一步动作；rows 多出 e / level / m / tabu_n / ok；闭环里能转向身后的目标。
没设 TMPDIR 时临时目录落 $DATA_ROOT/uav_rgb/tmp（根分区很小，不往 /tmp 写）。
"""
import importlib.util
import math
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("test_rule_oracle_k1", os.path.join(HERE, "test_rule_oracle.py"))
tro = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tro)                       # 复用 K1 测试里的桩占据与桩状态（模块级只有定义，不跑用例）
ro, rp, guard = tro.ro, tro.rp, tro.guard
State, make_policy, stub_occ = tro.State, tro.make_policy, tro.stub_occ
SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT = tro.SAFE_NEUTRAL, tro.FORWARD_SLOW, tro.YAW_LEFT, tro.YAW_RIGHT

K1_ROW_KEYS = ["step", "err_deg", "n", "target", "action", "entry", "mem_n", "clear", "src", "d_occ", "d_depth"]
V2_ROW_KEYS = ["step", "err_deg", "n", "e", "target", "action", "entry", "mem_n", "clear", "level", "m", "tabu_n", "ok", "src", "d_occ", "d_depth"]
TWO_LEVELS = ((0.35, 1.2, False), (0.25, 0.8, True))


def v2(**kw):
    return make_policy(rule_version="v2", **kw)


class TestVersionSwitch(unittest.TestCase):
    def test_default_is_v1_and_unchanged(self):
        p = make_policy(target=(5.0, 0.0))
        self.assertEqual((p.rule_version, p.planner_name, p.version), ("v1", "rule_v1", "uavrgb.rule_oracle.v1"))
        self.assertEqual(p.params, rp.RULE_V1)
        self.assertEqual(list(p.entry_counts), list(rp.V1_ENTRIES))
        p(None, None, State(0.0, 0.0, 0.0))
        self.assertEqual(list(p.rows[0]), K1_ROW_KEYS, "v1 的日志键与键序不变")
        self.assertEqual(p.rows[0]["entry"], "V1-fwd")

    def test_explicit_v1_equals_default_step_by_step(self):
        occ = stub_occ(door=True)
        a = make_policy(occ, target=(6.0, 0.4))
        b = make_policy(occ, target=(6.0, 0.4), rule_version="v1")
        rng = random.Random(5)
        for _ in range(40):
            st = State(rng.uniform(0.0, 2.4), rng.uniform(-1.5, 1.5), rng.uniform(-60, 60))
            self.assertEqual(a(None, None, st), b(None, None, st))
        self.assertEqual(a.rows, b.rows)
        self.assertEqual(a.entry_counts, b.entry_counts)

    def test_v2_attributes(self):
        p = v2(target=(5.0, 0.0))
        self.assertEqual((p.rule_version, p.planner_name, p.version), ("v2", "rule_v2", "uavrgb.rule_oracle.v2"))
        self.assertEqual(p.params, rp.RULE_V2)
        self.assertEqual(list(p.entry_counts), list(rp.V2_ENTRIES))
        self.assertEqual(ro.RuleOraclePolicy.planner_name, "rule_v1", "类属性不动，v1 实例仍读到 rule_v1")
        self.assertEqual(ro.RuleOraclePolicy.version, "uavrgb.rule_oracle.v1")

    def test_v2_params_override_and_do_not_touch_module_dict(self):
        p = v2(params={"levels": [[0.25, 1.2, False], [0.25, 0.8, False]], "order": "near", "yaw_step_deg": 31.7, "mem_steps": 30})
        self.assertEqual((p.params["order"], p.params["yaw_step_deg"], p.params["mem_steps"]), ("near", 31.7, 30))
        self.assertEqual(p.params["tabu_steps"], rp.RULE_V2["tabu_steps"])
        self.assertEqual(rp.RULE_V2["order"], "lexi")
        self.assertEqual(rp.RULE_V2["levels"], ((0.35, 1.2, False),))

    def test_unknown_keys_judged_against_the_chosen_version(self):
        with self.assertRaises(ValueError):
            v2(params={"r_safe": 0.2})                        # r_safe 是 v1 的键；v2 用 levels
        with self.assertRaises(ValueError):
            make_policy(params={"levels": [[0.25, 1.2, False]]})   # levels 是 v2 的键
        with self.assertRaises(ValueError):
            v2(params={"order_": "near"})
        make_policy(params={"r_safe": 0.2})
        v2(params={"levels": [[0.25, 1.2, False]]})

    def test_bad_v2_params_rejected_at_construction(self):
        with self.assertRaises(ValueError):
            v2(params={"levels": [[0.25, 1.2, False], [0.30, 0.8, False]]})
        with self.assertRaises(ValueError):
            v2(params={"order": "first"})
        with self.assertRaises(ValueError):
            v2(params={"levels": []})

    def test_bad_version_rejected(self):
        with self.assertRaises(ValueError):
            make_policy(rule_version="v3")
        with self.assertRaises(ValueError):
            make_policy(rule_version=None)


class TestV2SignsAndRule(unittest.TestCase):
    def test_target_left_gives_negative_n_and_e_and_yaw_left(self):
        p = v2(target=(0.0, 5.0))                            # 目标正左，err = +90°
        action, _ = p(None, None, State(0.0, 0.0, 0.0))
        row = p.rows[0]
        self.assertGreater(row["err_deg"], 80.0)
        self.assertLess(row["n"], 0)
        self.assertLess(row["e"], 0.0)
        self.assertEqual((action, row["entry"]), (YAW_LEFT, "V2-back0"))      # n < 0 时「转回」是 YAW_LEFT

    def test_target_right_gives_positive_n_and_e_and_yaw_right(self):
        p = v2(target=(0.0, -5.0))
        action, _ = p(None, None, State(0.0, 0.0, 0.0))
        self.assertGreater(p.rows[0]["n"], 0)
        self.assertGreater(p.rows[0]["e"], 0.0)
        self.assertEqual((action, p.rows[0]["entry"]), (YAW_RIGHT, "V2-back0"))

    def test_e_is_minus_err_over_yaw_step_and_n_is_its_rounding(self):
        for err, step in ((66.0, 33.0), (-66.0, 33.0), (15.0, 33.0), (20.0, 33.0), (-100.0, 26.0), (47.0, 31.7)):
            with self.subTest(err=err, step=step):
                tgt = (5.0 * math.cos(math.radians(err)), 5.0 * math.sin(math.radians(err)))
                p = v2(target=tgt, params={"yaw_step_deg": step})
                p(None, None, State(0.0, 0.0, 0.0))
                row = p.rows[0]
                self.assertAlmostEqual(row["err_deg"], err, places=1)
                self.assertAlmostEqual(row["e"], -err / step, places=1)
                if abs(err / step - round(err / step)) > 0.05:
                    self.assertNotEqual(row["e"], float(row["n"]), "e 是浮点值，不取整")
                self.assertEqual(row["n"], int(round(-err / step)))

    def test_target_ahead_goes_forward_at_level_zero(self):
        p = v2(target=(5.0, 0.0))
        action, _ = p(None, None, State(0.0, 0.0, 0.0))
        row = p.rows[0]
        self.assertEqual((action, row["n"], row["entry"], row["level"], row["m"]), (FORWARD_SLOW, 0, "V2-fwd0", 0, 0))

    def test_rows_carry_v2_info_in_the_documented_order(self):
        p = v2(target=(6.0, 0.0), params={"levels": TWO_LEVELS, "order": "near"})
        p(None, None, State(2.0, 0.0, 0.0))
        row = p.rows[0]
        self.assertEqual(list(row), V2_ROW_KEYS)
        self.assertEqual(len(row["ok"]), 2)
        self.assertTrue(all(len(r) == 2 * p.params["n_cap"] + 1 for r in row["ok"]))
        self.assertEqual(row["src"], "occ")
        self.assertIsNotNone(row["d_occ"])
        self.assertIsInstance(row["tabu_n"], int)

    def test_rule_call_is_rule_v2_step_bit_for_bit(self):
        occ = stub_occ(door=True)
        params = {"levels": [[0.25, 1.2, False], [0.25, 0.8, False], [0.20, 0.55, True]], "order": "near", "yaw_step_deg": 31.7}
        pol = v2(occ=occ, target=(6.0, 0.4), params=params)
        mem, tabu, last, prev = [], [], None, None
        rng = random.Random(3)
        seen = set()
        for i in range(60):
            st = State(rng.uniform(0.0, 2.4), rng.uniform(-1.5, 1.5), rng.uniform(-60, 60))
            action, _ = pol(None, None, st)
            yaw_deg = math.degrees(st.yaw)
            d = ro.occ_columns(occ, st.x, st.y, st.z, yaw_deg)
            err = guard.bearing_error_deg(st.x, st.y, yaw_deg, (6.0, 0.4))
            n, e = int(round(-err / pol.params["yaw_step_deg"])), -err / pol.params["yaw_step_deg"]
            want, entry, mem, tabu, info = rp.rule_v2_step(mem, tabu, i, st.x, st.y, yaw_deg, [float(v) for v in d], n, last, e=e,
                                                           prev_action=prev, hfov_deg=60.0, params=pol.params)
            row = pol.rows[-1]
            self.assertEqual((action, row["entry"], row["level"], row["m"], row["tabu_n"], row["ok"], row["mem_n"]),
                             (want, entry, info["level"], info["m"], info["tabu_n"], info["ok"], info["mem_n"]), i)
            self.assertEqual(row["clear"], [int(c) for c in info["clear"]])
            seen.add(entry)
            if want in (YAW_LEFT, YAW_RIGHT):
                last = want
            prev = want
        self.assertGreaterEqual(len(seen), 3, seen)

    def test_previous_returned_action_is_passed_as_prev_action(self):
        calls, orig = [], rp.rule_v2_step

        def spy(*args, **kw):
            calls.append(kw)
            return orig(*args, **kw)

        rp.rule_v2_step = spy
        try:
            p = v2(target=(0.0, 5.0), params={"yaw_step_deg": 33.0})
            acts = [p(None, None, State(0.0, 0.0, yaw))[0] for yaw in (0.0, 33.0, 66.0, 99.0, 99.0)]
        finally:
            rp.rule_v2_step = orig
        self.assertEqual(len(calls), 5)
        self.assertIsNone(calls[0]["prev_action"])
        self.assertEqual([c["prev_action"] for c in calls[1:]], acts[:-1])
        # 目标在正左（err = 90°, 57°, 24°, −9°）：e = −2.73, −1.73, −0.73, +0.27，n = −3, −2, −1, 0；n < 0 转回 = YAW_LEFT，n = 0 前进
        self.assertEqual(acts, [YAW_LEFT, YAW_LEFT, YAW_LEFT, FORWARD_SLOW, FORWARD_SLOW])
        self.assertEqual(calls[3]["prev_action"], YAW_LEFT)
        self.assertEqual(calls[4]["prev_action"], FORWARD_SLOW, "前进之后的下一步，传入的是 FORWARD_SLOW")
        self.assertAlmostEqual(calls[0]["e"], -90.0 / 33.0, places=1)
        self.assertAlmostEqual(calls[3]["e"], 9.0 / 33.0, places=1)
        self.assertEqual(calls[0]["params"], p.params)

    def test_stationary_only_level_follows_the_policys_own_previous_action(self):
        occ = stub_occ(door=False)                           # 墙面在 x = 3.0；机体中心 x = 2.0 时，墙面点在机体系 x = 1.0：只堵第 0 级（d_look 1.2），不堵第 1 级（0.8）
        p = v2(occ=occ, target=(6.0, 0.0), params={"levels": TWO_LEVELS, "order": "near"})
        st = State(2.0, 0.0, 0.0)
        a0, _ = p(None, None, st)
        self.assertEqual((a0, p.rows[0]["entry"]), (FORWARD_SLOW, "V2-fwd1"), "第 0 步：没有上一步动作，第 1 级可用")
        a1, _ = p(None, None, st)
        # 上一步刚前进：第 1 级只许静止起步，不可用；视场只有 ±30°：m = ±1（±26°）的通道被墙点堵住，m = ±2（±52°）超出视场、记忆里也没有点，判畅通
        self.assertEqual((a1, p.rows[1]["entry"], p.rows[1]["level"], p.rows[1]["m"]), (YAW_LEFT, "V2-seek0", 0, 2))
        a2, _ = p(None, None, st)
        self.assertEqual((a2, p.rows[2]["entry"]), (FORWARD_SLOW, "V2-fwd1"), "上一步是转向：第 1 级重新可用")
        self.assertEqual(p.last_action, a2)

    def test_never_safe_neutral(self):
        occ = stub_occ(door=True, dilate=True)
        rng = random.Random(11)
        pol = v2(occ=occ, target=(6.0, 0.0), params={"levels": TWO_LEVELS, "order": "lexi"})
        seen = set()
        for _ in range(300):
            pol.target_fn = lambda x, y, t=(rng.uniform(-6, 6), rng.uniform(-6, 6)): t
            st = State(rng.uniform(-2.0, 2.9), rng.uniform(-2.5, 2.5), rng.uniform(-180, 180))
            a, _ = pol(None, None, st)
            seen.add(a)
        self.assertTrue(seen <= {FORWARD_SLOW, YAW_LEFT, YAW_RIGHT}, seen)
        self.assertGreaterEqual(len(seen), 2)

    def test_reset_clears_the_episode(self):
        occ = stub_occ(door=False)
        p = v2(occ=occ, target=(6.0, 0.0))
        for i in range(5):
            p(None, None, State(2.2 + 0.1 * i, 0.0, 0.0))
        self.assertGreater(len(p.memory), 0)
        self.assertGreater(len(p.tabu), 0)
        self.assertIsNotNone(p.last_action)
        p.reset()
        self.assertEqual((p.memory, p.tabu, p.rows, p.step_idx, p.last_turn, p.last_action), ([], [], [], 0, None, None))
        self.assertEqual(sum(p.entry_counts.values()), 0)
        self.assertEqual(list(p.entry_counts), list(rp.V2_ENTRIES))
        p(None, None, State(-2.0, 0.0, 0.0))
        self.assertEqual((p.rows[0]["step"], p.rows[0]["mem_n"], p.rows[0]["tabu_n"]), (0, 0, 0))

    def test_entry_counts_follow_the_rows(self):
        occ = stub_occ(door=True)
        p = v2(occ=occ, target=(6.0, 0.4))
        rng = random.Random(9)
        for _ in range(50):
            p(None, None, State(rng.uniform(0.0, 2.4), rng.uniform(-1.5, 1.5), rng.uniform(-60, 60)))
        from collections import Counter
        self.assertEqual({k: v for k, v in p.entry_counts.items() if v}, dict(Counter(r["entry"] for r in p.rows)))

    def test_interface_has_no_goal_stop_and_no_guard_trace(self):
        p = v2()
        self.assertEqual(p.name, "rule_oracle")
        for attr in ("stop_radius", "trace", "prev"):
            self.assertFalse(hasattr(p, attr), attr)


class TestV2ClosedLoop(unittest.TestCase):
    """与 K1 的闭环同一个最小运动学：FORWARD 前进 0.3 m，YAW_LEFT / RIGHT 转 ±33°。n、e 的符号错了会越转越远。"""

    def run_loop(self, start_yaw, target, steps=40, occ=None, params=None):
        p = v2(occ=occ, target=target, params={"yaw_step_deg": 33.0, **(params or {})})
        x, y, yaw = 0.0, 0.0, start_yaw
        for _ in range(steps):
            a, _ = p(None, None, State(x, y, yaw))
            if a == YAW_LEFT:
                yaw += 33.0
            elif a == YAW_RIGHT:
                yaw -= 33.0
            else:
                x += 0.3 * math.cos(math.radians(yaw))
                y += 0.3 * math.sin(math.radians(yaw))
            if math.hypot(x - target[0], y - target[1]) < 0.8:
                return True, p
        return False, p

    def test_turns_around_and_reaches_a_target_behind(self):
        for yaw0 in (180.0, 135.0, -135.0, 90.0, -90.0):
            with self.subTest(yaw0=yaw0):
                ok, p = self.run_loop(yaw0, (6.0, 1.0))
                self.assertTrue(ok, [r["entry"] for r in p.rows])

    def test_levels_variant_also_reaches_the_target(self):
        ok, p = self.run_loop(135.0, (6.0, 1.0), params={"levels": TWO_LEVELS, "order": "near"})
        self.assertTrue(ok, [r["entry"] for r in p.rows])


if __name__ == "__main__":
    unittest.main(verbosity=2)
