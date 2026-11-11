#!/usr/bin/env python3
"""规则 v2（任务卡 K3：两侧候选、分级通道、禁忌航向）的纯函数测试与离线运动学模拟。不起 Gazebo、不渲染。

坐标与角度约定同 rule_policy / teacher（x 前、y 左；yaw 度，0 = +x，正 = 左转；m、n 正 = 向左）。
覆盖：RULE_V2 数值；v1 / v0 的常量与函数与 BASE_REV 逐位一致；levels 不逐级收紧报错；禁忌条目的三种失效（太远、过期、级别）
与航向容差、绕圈换算、按世界航向记；两种次序（lexi / near）与 a / b / c 三个基本动作、两侧候选的排序键与范围、
只许静止起步的级；四条不变式（不输出 SAFE_NEUTRAL；前进只在某个可用级的当前航向畅通时出现；同一位置面对过并判堵的航向在禁忌有效期内
不会再被选作候选；v1 的两步原地循环场景里 v2 在 2·n_cap + 3 步内前进或进入 V2-stuck）；随机输入上与按任务卡文字另写的参照实现逐项一致；
rule_sim 里 20 个 train 分区几何上过线数与撞柱数不差于 v1。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_rule_v2.py
"""
import ast
import math
import os
import random
import subprocess
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rule_policy as rp
import rule_sim
from rule_policy import (RULE_V1, RULE_V2, V2_ENTRIES, SAFE_NEUTRAL, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT,
                         check_v2_params, corridor_clear, rule_v2_entry, rule_v2_step, v2_tabu_hit)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE_REV = "41335cd"          # K3 起点（main）；v1 / v0 在这个版本的 rule_policy.py 里的样子是「不许动」的参照
ALPHA = RULE_V2["yaw_step_deg"]
OPEN20 = [4.0] * 20
# 任务卡的变体表（实验入口 run.sh 里有同一份；experiments/2026-11-11-rule-v2-3dgs/test_run_sh.py 核对两处与任务卡一致）
VARIANTS = {
    "v2a": (((0.25, 1.2, False),), "lexi"),
    "v2b": (((0.25, 1.2, False), (0.25, 0.8, False)), "lexi"),
    "v2c": (((0.25, 1.2, False), (0.25, 0.8, False)), "near"),
    "v2d": (((0.25, 1.2, False), (0.25, 0.8, False), (0.20, 0.55, True)), "near"),
    "v2e": (((0.15, 1.2, False),), "lexi"),
    "v2f": (((0.15, 1.2, False), (0.15, 0.8, False), (0.12, 0.55, True)), "near"),
}
TWO_LEVELS = ((0.35, 1.2, False), (0.25, 0.8, False))


def square(x, y, size=0.4):
    return {"name": "p", "x": x, "y": y, "z": 1.5, "size": [size, size, 3.0]}


def at(m, u, alpha=ALPHA):
    """机体系点：在朝 m·α 的通道坐标里 (u, 0)，即正好在航向 m·α 的轴线上、距离 u。"""
    th = math.radians(m * alpha)
    return (u * math.cos(th), u * math.sin(th))


def entry(points=(), n=0, last_turn=None, tabu=(), step=10, pose=(0.0, 0.0, 0.0), prev=None, e=None, d=OPEN20, **over):
    """rule_v2_entry 的简写：points 是机体系点；over 覆盖 RULE_V2 里的键。"""
    return rule_v2_entry(list(points), d, n, last_turn, list(tabu), step, pose, prev, e, {**RULE_V2, **over})


def action_of(res):
    return res[0], res[1]


class ParamsTests(unittest.TestCase):
    def test_frozen_values(self):
        self.assertEqual(RULE_V2, {"yaw_step_deg": 26.0, "n_cap": 3, "tie_frac": 0.10, "sum_clip": 4.0, "mem_steps": 24,
                                   "levels": ((0.35, 1.2, False),), "order": "lexi", "tabu_radius": 0.30, "tabu_steps": 24, "m_max": 6})

    def test_shared_values_equal_v1(self):
        for key in ("yaw_step_deg", "n_cap", "tie_frac", "sum_clip"):
            self.assertEqual(RULE_V2[key], RULE_V1[key], key)
        self.assertEqual(RULE_V2["levels"][0][:2], (RULE_V1["r_safe"], RULE_V1["d_look"]))
        self.assertGreater(RULE_V2["mem_steps"], RULE_V1["mem_steps"])

    def test_entries(self):
        self.assertEqual(len(V2_ENTRIES), 10)
        self.assertEqual(V2_ENTRIES[-1], "V2-stuck")
        for lv in range(3):
            for kind in ("back", "fwd", "seek"):
                self.assertIn(f"V2-{kind}{lv}", V2_ENTRIES)


class V1UntouchedTests(unittest.TestCase):
    """v1 / v0 的常量与函数与 BASE_REV 的 rule_policy.py 逐位一致（v2 全是新增）。"""

    @classmethod
    def setUpClass(cls):
        try:
            cls.old_src = subprocess.run(["git", "show", f"{BASE_REV}:sim/tools/rule_policy.py"], capture_output=True, text=True, check=True,
                                         cwd=HERE).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise unittest.SkipTest(f"取不到 {BASE_REV} 的 rule_policy.py（{exc}）")
        cls.old = types.ModuleType("rule_policy_base")
        exec(compile(cls.old_src, "rule_policy_base.py", "exec"), cls.old.__dict__)

    def test_every_old_top_level_node_unchanged(self):
        new_nodes = {}
        with open(os.path.join(HERE, "rule_policy.py"), encoding="utf-8") as f:
            new_body = ast.parse(f.read()).body
        for node in new_body:
            key = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else ast.dump(node)[:80]
            new_nodes[key] = ast.dump(node)
        checked = 0
        for node in ast.parse(self.old_src).body:
            key = node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else ast.dump(node)[:80]
            self.assertIn(key, new_nodes, f"BASE_REV 里的 {key} 在当前文件里不见了")
            self.assertEqual(ast.dump(node), new_nodes[key], f"{key} 与 BASE_REV 不同")
            checked += 1
        self.assertGreater(checked, 15)

    def test_constants_equal(self):
        for name in ("RULE_V0", "RULE_V1", "V1_ENTRIES", "COL_DEG", "N_RAYS"):
            self.assertEqual(getattr(rp, name), getattr(self.old, name), name)

    def test_v1_step_bit_identical_on_random_inputs(self):
        rng = random.Random(20261111)
        for i in range(300):
            memory, new_memory = [], []
            n, last = 0, None
            x = y = yaw = 0.0
            for step in range(rng.randint(1, 6)):
                d = [4.0 if rng.random() < 0.5 else round(rng.uniform(0.3, 4.0), 3) for _ in range(20)]
                got = rp.rule_v1_step(new_memory, step, x, y, yaw, d, n, last)
                ref = self.old.rule_v1_step(memory, step, x, y, yaw, d, n, last)
                self.assertEqual(got, ref, f"case {i} step {step}")
                memory, new_memory = ref[2], got[2]
                if got[0] == YAW_LEFT:
                    n, last, yaw = n + 1, YAW_LEFT, yaw + 26.0
                elif got[0] == YAW_RIGHT:
                    n, last, yaw = n - 1, YAW_RIGHT, yaw - 26.0
                else:
                    x += 0.25 * math.cos(math.radians(yaw))
                    y += 0.25 * math.sin(math.radians(yaw))

    def test_v0_decision_bit_identical_on_random_inputs(self):
        rng = random.Random(7)
        for _ in range(300):
            d = [4.0 if rng.random() < 0.4 else round(rng.uniform(0.3, 4.0), 3) for _ in range(20)]
            n, last = rng.randint(-3, 3), rng.choice([None, YAW_LEFT, YAW_RIGHT])
            self.assertEqual(rp.rule_v0_entry(d, n, last), self.old.rule_v0_entry(d, n, last))


class LevelsTests(unittest.TestCase):
    def test_default_and_card_variants_accepted(self):
        self.assertEqual(check_v2_params(RULE_V2), ((0.35, 1.2, False),))
        for name, (levels, order) in VARIANTS.items():
            self.assertEqual(check_v2_params({**RULE_V2, "levels": levels, "order": order}), tuple(tuple(lv) for lv in levels), name)

    def test_json_lists_accepted_and_normalised(self):
        got = check_v2_params({**RULE_V2, "levels": [[0.25, 1.2, 0], [0.25, 0.8, 1]]})
        self.assertEqual(got, ((0.25, 1.2, False), (0.25, 0.8, True)))

    def test_equal_levels_are_not_a_relaxation(self):
        check_v2_params({**RULE_V2, "levels": ((0.25, 1.2, False), (0.25, 1.2, True))})

    def test_relaxing_r_safe_rejected(self):
        with self.assertRaises(ValueError):
            check_v2_params({**RULE_V2, "levels": ((0.25, 1.2, False), (0.30, 0.8, False))})

    def test_relaxing_d_look_rejected(self):
        with self.assertRaises(ValueError):
            check_v2_params({**RULE_V2, "levels": ((0.25, 0.8, False), (0.25, 1.2, False))})

    def test_relaxing_at_a_later_pair_rejected(self):
        with self.assertRaises(ValueError):
            check_v2_params({**RULE_V2, "levels": ((0.35, 1.2, False), (0.25, 0.8, False), (0.30, 0.5, True))})

    def test_malformed_levels_rejected(self):
        bad = ((), None, [(0.25, 1.2)], [(0.25, 1.2, False, 1)], [(0.0, 1.2, False)], [(0.25, -1.0, False)],
               [(float("nan"), 1.2, False)], [(0.25, float("inf"), False)], [0.25])
        for levels in bad:
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                check_v2_params({**RULE_V2, "levels": levels})

    def test_other_params_validated(self):
        for key, value in (("order", "first"), ("n_cap", 0), ("n_cap", 2.5), ("m_max", 0), ("m_max", True), ("tabu_radius", -0.1),
                           ("tabu_steps", float("nan")), ("tabu_steps", -1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check_v2_params({**RULE_V2, key: value})

    def test_step_and_entry_raise_on_bad_levels(self):
        bad = {**RULE_V2, "levels": ((0.25, 1.2, False), (0.30, 0.8, False))}
        with self.assertRaises(ValueError):
            rule_v2_step([], [], 0, 0.0, 0.0, 0.0, OPEN20, 0, None, params=bad)
        with self.assertRaises(ValueError):
            rule_v2_entry([], OPEN20, 0, None, [], 0, (0.0, 0.0, 0.0), None, None, bad)


class TabuTests(unittest.TestCase):
    POSE = (1.0, 2.0, 30.0)

    def hit(self, tabu, m=1, level=0, step=10, pose=POSE, **over):
        x, y, yaw = pose
        return v2_tabu_hit(tabu, yaw + m * ALPHA, level, step, x, y, {**RULE_V2, **over})

    def test_matching_entry_hits(self):
        self.assertTrue(self.hit([(self.POSE[2] + ALPHA, 1.0, 2.0, 8, 0)], m=1))

    def test_heading_tolerance_half_step_inclusive(self):
        base = self.POSE[2] + ALPHA
        self.assertTrue(self.hit([(base + ALPHA / 2, 1.0, 2.0, 8, 0)]))
        self.assertTrue(self.hit([(base - ALPHA / 2, 1.0, 2.0, 8, 0)]))
        self.assertFalse(self.hit([(base + ALPHA / 2 + 0.01, 1.0, 2.0, 8, 0)]))
        self.assertFalse(self.hit([(base - ALPHA / 2 - 0.01, 1.0, 2.0, 8, 0)]))

    def test_heading_compared_modulo_360(self):
        self.assertTrue(v2_tabu_hit([(-179.0, 0.0, 0.0, 5, 0)], 181.0, 0, 6, 0.0, 0.0, RULE_V2))
        self.assertFalse(v2_tabu_hit([(-179.0, 0.0, 0.0, 5, 0)], 181.0 + 20.0, 0, 6, 0.0, 0.0, RULE_V2))

    def test_far_entry_does_not_hit(self):
        psi = self.POSE[2] + ALPHA
        edge = (psi, 1.0 + 0.25, 2.0, 8, 0)             # 0.25 在二进制里精确，边界比较不受舍入影响
        far = (psi, 1.0 + 0.26, 2.0, 8, 0)
        self.assertTrue(self.hit([edge], tabu_radius=0.25), "半径内（含边界）")
        self.assertFalse(self.hit([far], tabu_radius=0.25), "半径外")
        self.assertTrue(self.hit([far], tabu_radius=0.5), "半径是参数")
        self.assertFalse(self.hit([(psi, 1.0 + 0.5, 2.0, 8, 0)]), "默认半径 0.30 m：0.5 m 外不禁")

    def test_expired_entry_does_not_hit(self):
        psi = self.POSE[2] + ALPHA
        self.assertTrue(self.hit([(psi, 1.0, 2.0, 10 - 24, 0)], step=10), "步龄 == tabu_steps 仍有效")
        self.assertFalse(self.hit([(psi, 1.0, 2.0, 10 - 25, 0)], step=10), "步龄 > tabu_steps 失效")
        self.assertTrue(self.hit([(psi, 1.0, 2.0, 10 - 25, 0)], step=10, tabu_steps=30))

    def test_level_above_lb_does_not_hit(self):
        psi = self.POSE[2] + ALPHA
        e0, e1 = (psi, 1.0, 2.0, 8, 0), (psi, 1.0, 2.0, 8, 1)
        self.assertTrue(self.hit([e0], level=0))
        self.assertFalse(self.hit([e0], level=1), "lb = 0 只禁第 0 级")
        self.assertTrue(self.hit([e1], level=0))
        self.assertTrue(self.hit([e1], level=1))
        self.assertFalse(self.hit([e1], level=2))

    def test_entry_recorded_when_heading_blocked_at_level_zero(self):
        tabu = rule_v2_step([], [], 3, 1.0, 2.0, 30.0, OPEN20, 0, None)[3]
        self.assertEqual(tabu, [], "当前航向畅通：不记")
        d = rp.column_distances(1.0, 2.0, 30.0, [square(1.0 + 0.8 * math.cos(math.radians(30)), 2.0 + 0.8 * math.sin(math.radians(30)))])
        action, ent, memory, tabu, info = rule_v2_step([], [], 3, 1.0, 2.0, 30.0, d, 0, None)
        self.assertEqual(tabu, [(30.0, 1.0, 2.0, 3, 0)], "条目 = (世界 yaw, x, y, 步号, lb)")
        self.assertNotEqual(ent, "V2-fwd0")

    def test_lb_is_deepest_blocked_level_by_geometry(self):
        params = {**RULE_V2, "levels": ((0.25, 1.2, False), (0.25, 0.8, False), (0.20, 0.55, True))}
        for face, lb in ((1.0, 0), (0.7, 1), (0.4, 2)):       # 正前方柱的近面距机体中心 face（柱心 = face + 0.2）
            d = rp.column_distances(0.0, 0.0, 0.0, [square(face + 0.2, 0.0)])
            tabu = rule_v2_step([], [], 0, 0.0, 0.0, 0.0, d, 0, None, params=params)[3]
            self.assertEqual(len(tabu), 1, face)
            self.assertEqual(tabu[0][4], lb, face)

    def test_lb_ignores_stationary_usability(self):
        params = {**RULE_V2, "levels": ((0.25, 1.2, False), (0.25, 0.8, False), (0.20, 0.55, True)), "order": "near"}
        d = rp.column_distances(0.0, 0.0, 0.0, [square(0.45 + 0.2, 0.0)])
        for prev in (None, FORWARD_SLOW, YAW_LEFT):
            tabu = rule_v2_step([], [], 0, 0.0, 0.0, 0.0, d, 0, None, prev_action=prev, params=params)[3]
            self.assertEqual(tabu[0][4], 2, prev)

    def test_old_entries_removed_each_step(self):
        old = [(0.0, 5.0, 5.0, 0, 0)]
        tabu = rule_v2_step([], old, 24, 0.0, 0.0, 0.0, OPEN20, 0, None)[3]
        self.assertEqual(tabu, old, "步龄 24 仍留")
        tabu = rule_v2_step([], old, 25, 0.0, 0.0, 0.0, OPEN20, 0, None)[3]
        self.assertEqual(tabu, [], "步龄 25 清掉")

    def test_step_does_not_mutate_inputs(self):
        memory, tabu = [(0.5, 0.0, 0)], [(0.0, 0.0, 0.0, 0, 0)]
        before = (list(memory), list(tabu))
        d = rp.column_distances(0.0, 0.0, 0.0, [square(0.8, 0.0)])
        rule_v2_step(memory, tabu, 1, 0.0, 0.0, 0.0, d, 0, None)
        self.assertEqual((memory, tabu), before)

    def test_tabu_is_keyed_by_world_heading_not_n(self):
        tabu = [(ALPHA, 0.0, 0.0, 5, 0)]                # 世界航向 26° 被禁；机头在 0° 时它就是 m = +1，不管 n 是几
        block0 = [at(0, 1.0)]
        for n in (0, 1, -1, 2, -2):
            res = entry(block0, n=n, tabu=tabu, step=6, pose=(0.0, 0.0, 0.0))
            self.assertNotEqual(res[2]["m"], 1, f"n={n}：m = +1 的世界航向被禁，不能选")
        control = entry(block0, n=0, pose=(0.0, 0.0, 0.0))
        self.assertEqual(control[2]["m"], 1, "对照：没有禁忌条目时 n = 0 选 m = +1（左）")
        self.assertEqual(entry(block0, n=-2, pose=(0.0, 0.0, 0.0))[2]["m"], 1, "对照：n = −2 时转回（m = +1）")

    def test_tabu_heading_never_chosen_while_valid(self):
        """同一位置：当前航向被堵，m=+1 的世界航向在禁忌表里 → 不选 m=+1 去转（即使几何上它畅通）；过期后可以选。"""
        pose = (0.0, 0.0, 0.0)
        tabu = [(ALPHA, 0.0, 0.0, 5, 0)]
        res = entry([at(0, 1.0)], tabu=tabu, step=10, pose=pose)
        self.assertEqual(res[2]["m"], -1)
        res = entry([at(0, 1.0)], tabu=tabu, step=40, pose=pose)
        self.assertEqual(res[2]["m"], 1, "过期后几何畅通的 m=+1 又可选（调用方通常已清掉条目，这里直接传过期条目）")
        res = entry([at(0, 1.0)], tabu=[(ALPHA, 0.5, 0.0, 5, 0)], step=10, pose=pose)
        self.assertEqual(res[2]["m"], 1, "位置差 0.5 m > 0.30 m：不禁")

    def test_back_candidate_also_respects_tabu(self):
        tabu = [(-ALPHA, 0.0, 0.0, 5, 0)]               # 世界 −26° 被禁；n = 1 时「转回」就是它
        res = entry([], n=1, tabu=tabu, step=6, pose=(0.0, 0.0, 0.0))
        self.assertEqual((res[0], res[1]), (FORWARD_SLOW, "V2-fwd0"), "转回被禁 → 当前航向畅通就前进")

    def test_current_heading_exempt_from_tabu(self):
        tabu = [(0.0, 0.0, 0.0, 5, 0)]
        res = entry([], n=0, tabu=tabu, step=6)
        self.assertEqual((res[0], res[1]), (FORWARD_SLOW, "V2-fwd0"), "m = 0 不查禁忌表")


class DecisionTests(unittest.TestCase):
    def test_back_turns_toward_guidance(self):
        self.assertEqual(action_of(entry(n=1)), (YAW_RIGHT, "V2-back0"))
        self.assertEqual(action_of(entry(n=-2)), (YAW_LEFT, "V2-back0"))
        self.assertEqual(entry(n=1)[2]["m"], -1)
        self.assertEqual(entry(n=-2)[2]["m"], 1)

    def test_forward_when_clear_and_no_deviation(self):
        res = entry(n=0)
        self.assertEqual(action_of(res), (FORWARD_SLOW, "V2-fwd0"))
        self.assertEqual((res[2]["level"], res[2]["m"]), (0, 0))

    def test_forward_when_back_is_blocked_but_heading_clear(self):
        self.assertEqual(action_of(entry([at(-1, 1.0)], n=1)), (FORWARD_SLOW, "V2-fwd0"))

    def test_seek_prefers_smallest_m_left_on_tie(self):
        res = entry([at(0, 1.0)])
        self.assertEqual(action_of(res), (YAW_LEFT, "V2-seek0"))
        self.assertEqual(res[2]["m"], 1)

    def test_seek_other_side_when_one_side_blocked(self):
        self.assertEqual(action_of(entry([at(0, 1.0), at(1, 1.0)])), (YAW_RIGHT, "V2-seek0"))
        res = entry([at(0, 1.0), at(1, 1.0), at(-1, 1.0)])
        self.assertEqual((action_of(res), res[2]["m"]), ((YAW_LEFT, "V2-seek0"), 2))

    def test_seek_key_e_breaks_equal_m(self):
        block = [at(0, 1.0)]
        self.assertEqual(entry(block, n=0, e=0.4)[2]["m"], -1, "e > 0 = 机头偏左：转右更接近导引")
        self.assertEqual(entry(block, n=0, e=-0.4)[2]["m"], 1)
        block2 = [at(0, 1.0), at(1, 1.0), at(-1, 1.0)]
        self.assertEqual(entry(block2, n=0, e=0.4)[2]["m"], -2)
        self.assertEqual(entry(block2, n=0, e=-0.4)[2]["m"], 2)

    def test_seek_key_m_beats_e(self):
        res = entry([at(0, 1.0)], n=0, e=2.5)
        self.assertEqual(res[2]["m"], -1, "|m| 小优先于 |e + m|")
        self.assertEqual(entry([at(0, 1.0), at(-1, 1.0)], n=0, e=-2.5)[2]["m"], 1)

    def test_seek_key_side_preference_then_last_turn_then_left(self):
        block = [at(0, 1.0)]
        right_open = [1.0] * 10 + [4.0] * 10           # 左半近、右半远 → 偏向右
        left_open = [4.0] * 10 + [1.0] * 10
        self.assertEqual(entry(block, d=right_open, e=0.0)[2]["m"], -1)
        self.assertEqual(entry(block, d=left_open, e=0.0)[2]["m"], 1)
        self.assertEqual(entry(block, d=right_open, e=0.0, last_turn=YAW_LEFT)[2]["m"], -1, "左右求和的偏向先于 last_turn")
        self.assertEqual(entry(block, d=OPEN20, e=0.0, last_turn=YAW_RIGHT)[2]["m"], -1, "平局沿用 last_turn")
        self.assertEqual(entry(block, d=OPEN20, e=0.0, last_turn=YAW_LEFT)[2]["m"], 1)
        self.assertEqual(entry(block, d=OPEN20, e=0.0, last_turn=None)[2]["m"], 1, "都没有则向左")

    def test_seek_side_preference_boundary_is_inclusive(self):
        """左右求和差正好等于 tie_frac × max 时算有偏向（>=，与 v1 同）；差略小则是平局，改由 last_turn 决定。"""
        block = [at(0, 1.0)]
        left20_right18 = [2.0] * 10 + [2.0] * 9 + [0.0]          # 左 20.0、右 18.0，差 2.0 = 0.10 × 20.0（浮点下恰好相等）
        left18_right20 = [2.0] * 9 + [0.0] + [2.0] * 10
        left20_right185 = [2.0] * 10 + [2.0] * 9 + [0.5]          # 差 1.5 < 2.0
        self.assertEqual(entry(block, d=left20_right18, e=0.0, last_turn=YAW_RIGHT)[2]["m"], 1, "差正好 = tie_frac × max：偏向左，先于 last_turn")
        self.assertEqual(entry(block, d=left18_right20, e=0.0, last_turn=YAW_LEFT)[2]["m"], -1, "对称：偏向右")
        self.assertEqual(entry(block, d=left20_right185, e=0.0, last_turn=YAW_RIGHT)[2]["m"], -1, "差略小于边界：平局，沿用 last_turn")

    def test_seek_key_e_before_side_preference(self):
        block = [at(0, 1.0)]
        self.assertEqual(entry(block, d=[1.0] * 10 + [4.0] * 10, e=-0.4)[2]["m"], 1, "e 先于左右求和偏向")

    def test_seek_limited_by_n_cap(self):
        wall = [at(m, 1.0) for m in range(-3, 4)]
        res = entry(wall, n=0)
        self.assertEqual(action_of(res), (YAW_LEFT, "V2-stuck"), "|m| = 4、5、6 的航向畅通也不去：|n + m| 不超过 n_cap")
        self.assertEqual((res[2]["level"], res[2]["m"]), (None, None))
        res = entry([at(0, 1.0), at(-1, 1.0)], n=3)
        self.assertEqual((action_of(res), res[2]["m"]), ((YAW_RIGHT, "V2-seek0"), -2), "n = 3：m = +1（n + m = 4）超出 n_cap，不如 m = −2")

    def test_seek_with_large_n_never_goes_further_out(self):
        pts = [at(0, 1.0)] + [at(m, 1.0) for m in range(-6, 0)]
        self.assertEqual(entry(pts, n=4)[1], "V2-stuck", "n = 4：m > 0 的候选 n + m ≥ 5 > max(n_cap, |n|) 被排除，m < 0 的全堵")
        res = entry([at(0, 1.0), at(-1, 1.0)], n=4)
        self.assertEqual((res[1], res[2]["m"]), ("V2-seek0", -2), "n = 4 时 m = −2（n + m = 2）可达，m = +1（n + m = 5）不可")

    def test_seek_cap_relaxes_to_abs_n_when_already_far_off(self):
        """n = 6 > n_cap = 3：候选上限放宽到 |n|，m = −2（|n + m| = 4 > n_cap）可去；上限若仍是 n_cap，只剩 m ≤ −3，这里全堵，会掉到 V2-stuck。"""
        pts = [at(m, 1.0) for m in (0, -1, -3, -4, -5, -6)]            # 只留 m = −2 的航向畅通（相邻航向相差 26°，侧向 0.44 m > r_safe 0.35 m，互不遮挡）
        res = entry(pts, n=6)
        self.assertEqual((res[0], res[1], res[2]["m"]), (YAW_RIGHT, "V2-seek0", -2))

    def test_info_clear_is_left_keep_right_like_v1(self):
        """info["clear"] 与 v1 同次序 [朝左一步, 保持, 朝右一步]（日志与 v1 的 rule.jsonl 逐列可比）。"""
        self.assertEqual(entry([at(1, 1.0)])[2]["clear"], [0, 1, 1], "只堵朝左一步")
        self.assertEqual(entry([at(-1, 1.0)])[2]["clear"], [1, 1, 0], "只堵朝右一步")
        self.assertEqual(entry([at(0, 1.0)])[2]["clear"], [1, 0, 1], "只堵保持")

    def test_seek_limited_by_m_max(self):
        wall = [at(m, 1.0) for m in (-1, 0, 1)]
        self.assertEqual(entry(wall, n=0, m_max=2)[2]["m"], 2)
        self.assertEqual(entry(wall + [at(2, 1.0), at(-2, 1.0)], n=0, m_max=2)[1], "V2-stuck")
        self.assertEqual(entry(wall + [at(2, 1.0), at(-2, 1.0)], n=0, m_max=3)[2]["m"], 3)

    def test_stuck_continues_last_turn_or_left(self):
        wall = [at(m, 1.0) for m in range(-3, 4)]
        self.assertEqual(action_of(entry(wall, last_turn=YAW_RIGHT)), (YAW_RIGHT, "V2-stuck"))
        self.assertEqual(action_of(entry(wall, last_turn=YAW_LEFT)), (YAW_LEFT, "V2-stuck"))
        self.assertEqual(action_of(entry(wall, last_turn=None)), (YAW_LEFT, "V2-stuck"))
        self.assertEqual(action_of(entry(wall, last_turn=FORWARD_SLOW)), (YAW_LEFT, "V2-stuck"), "last_turn 不是转向时按没有算")

    def test_lexi_tries_seek_at_level_zero_before_forward_at_level_one(self):
        kw = dict(levels=TWO_LEVELS, yaw_step_deg=45.0)
        block = [at(0, 1.0, 45.0)]                     # 只堵第 0 级（u = 1.0 > 第 1 级的 d_look 0.8）
        self.assertEqual(action_of(entry(block, order="lexi", **kw)), (YAW_LEFT, "V2-seek0"))
        res = entry(block, order="near", **kw)
        self.assertEqual(action_of(res), (FORWARD_SLOW, "V2-fwd1"))
        self.assertEqual((res[2]["level"], res[2]["m"]), (1, 0))

    def test_near_forward_at_level_one_before_back_at_level_one(self):
        kw = dict(levels=TWO_LEVELS, yaw_step_deg=45.0, m_max=3)
        a = 45.0
        block = [at(m, 1.0, a) for m in (-3, -2, -1, 0, 1, 2)]       # 只堵第 0 级：n = 1 时的候选范围 m ∈ [−3, 2] 加上 m = 0 全堵，第 1 级全通
        res_lexi = entry(block, n=1, order="lexi", **kw)
        res_near = entry(block, n=1, order="near", **kw)
        self.assertEqual(action_of(res_lexi), (YAW_RIGHT, "V2-back1"), "lexi：a(0) b(0) c(0) 都无 → a(1)")
        self.assertEqual(action_of(res_near), (FORWARD_SLOW, "V2-fwd1"), "near：a(0) b(0) b(1) 在 c(0)、a(1) 之前")

    def test_near_order_with_three_levels(self):
        kw = dict(levels=((0.35, 1.2, False), (0.25, 0.8, False), (0.20, 0.55, False)), yaw_step_deg=45.0, m_max=3)
        a = 45.0
        block = [at(0, 1.0, a), at(0, 0.7, a)]         # 第 0、1 级当前航向堵，第 2 级通
        self.assertEqual(action_of(entry(block, order="near", **kw)), (FORWARD_SLOW, "V2-fwd2"), "b(2) 先于 c(0)")
        self.assertEqual(action_of(entry(block, order="lexi", **kw)), (YAW_LEFT, "V2-seek0"))

    def test_seek_at_deeper_level_only_when_shallower_fail(self):
        kw = dict(levels=TWO_LEVELS, yaw_step_deg=45.0, m_max=3, order="lexi")
        a = 45.0
        block = [at(0, 0.6, a)] + [at(m, 1.0, a) for m in (-3, -2, -1, 1, 2, 3)]       # 当前航向两级都堵；两侧只在第 0 级堵
        res = entry(block, **kw)
        self.assertEqual((res[1], res[2]["level"], res[2]["m"]), ("V2-seek1", 1, 1), "第 0 级无路，第 1 级 c(1) 找到两侧候选")
        self.assertEqual(action_of(res), (YAW_LEFT, "V2-seek1"))

    def test_stationary_only_level_needs_non_forward_previous_action(self):
        kw = dict(levels=((0.35, 1.2, False), (0.25, 0.8, True)), yaw_step_deg=45.0, order="near")
        block = [at(0, 1.0, 45.0)]                      # 只堵第 0 级
        for prev, expect in ((None, "V2-fwd1"), (YAW_LEFT, "V2-fwd1"), (YAW_RIGHT, "V2-fwd1"), (SAFE_NEUTRAL, "V2-fwd1"),
                             (FORWARD_SLOW, "V2-seek0")):
            res = entry(block, prev=prev, **kw)
            self.assertEqual(res[1], expect, f"prev={prev}")
            self.assertEqual(res[2]["usable"], [1, 0 if prev == FORWARD_SLOW else 1])

    def test_unusable_level_is_skipped_for_back_and_seek_too(self):
        kw = dict(levels=((0.35, 1.2, False), (0.25, 0.8, True)), yaw_step_deg=45.0, order="lexi", m_max=3)
        a = 45.0
        block = [at(m, 1.0, a) for m in (-3, -2, -1, 0, 1, 2, 3)]       # 第 0 级全堵；第 1 级全通
        self.assertEqual(entry(block, prev=FORWARD_SLOW, **kw)[1], "V2-stuck")
        self.assertEqual(entry(block, prev=None, **kw)[1], "V2-fwd1")

    def test_info_fields(self):
        res = entry([at(0, 1.0)], n=1, e=0.7, levels=TWO_LEVELS)
        info = res[2]
        self.assertEqual(set(info), {"level", "m", "e", "usable", "ok", "clear"})
        self.assertEqual(info["e"], 0.7)
        self.assertEqual(len(info["ok"]), 2)
        self.assertTrue(all(len(row) == 2 * RULE_V2["n_cap"] + 1 for row in info["ok"]))
        self.assertEqual(info["ok"][0][RULE_V2["n_cap"]], 0, "m = 0 在第 0 级被堵")
        self.assertEqual(info["ok"][1][RULE_V2["n_cap"]], 1, "m = 0 在第 1 级畅通（u = 1.0 > 0.8）")
        self.assertEqual(info["clear"], [1, 0, 1])
        self.assertEqual(entry([])[2]["e"], 0.0, "e 缺省取 n")
        self.assertEqual(entry([], n=-2)[2]["e"], -2.0)

    def test_ok_bitmap_marks_tabu_for_nonzero_m_only(self):
        res = entry([], tabu=[(ALPHA, 0.0, 0.0, 9, 0), (0.0, 0.0, 0.0, 9, 0)], step=10)
        row = res[2]["ok"][0]
        n_cap = RULE_V2["n_cap"]
        self.assertEqual(row[n_cap + 1], 0, "m = +1 被禁")
        self.assertEqual(row[n_cap], 1, "m = 0 不查禁忌")
        self.assertEqual(row[n_cap - 1], 1)


# ------------------------------------------------------------------ 参照实现：只按任务卡文字另写一遍，不调用 rule_policy 里的 v2 函数

def reference_v2(points, d, n, last_turn, tabu, step, pose, prev_action, e, params):
    alpha = params["yaw_step_deg"]
    levels = [(float(r), float(dl), bool(st)) for r, dl, st in params["levels"]]
    x, y, yaw = pose
    e = float(n) if e is None else e

    def geo_blocked(m, lv):
        th = math.radians(m * alpha)
        for px, py in points:
            u = px * math.cos(th) + py * math.sin(th)
            v = -px * math.sin(th) + py * math.cos(th)
            if 0.0 < u < levels[lv][1] and abs(v) < levels[lv][0]:
                return True
        return False

    def tabu_blocked(m, lv):
        psi = yaw + m * alpha
        for ye, xe, yye, se, lb in tabu:
            dyaw = (psi - ye + 180.0) % 360.0 - 180.0
            if abs(dyaw) <= alpha / 2 and math.hypot(x - xe, y - yye) <= params["tabu_radius"] and step - se <= params["tabu_steps"] and lv <= lb:
                return True
        return False

    def ok(m, lv):
        return (not geo_blocked(m, lv)) and (m == 0 or not tabu_blocked(m, lv))

    usable = [not (st and prev_action == FORWARD_SLOW) for _r, _d, st in levels]
    half = len(d) // 2
    left, right = sum(min(v, params["sum_clip"]) for v in d[:half]), sum(min(v, params["sum_clip"]) for v in d[len(d) - half:])
    side = 0
    if abs(left - right) >= params["tie_frac"] * max(left, right) and left != right:
        side = 1 if left > right else -1
    last = 1 if last_turn == YAW_LEFT else (-1 if last_turn == YAW_RIGHT else 0)

    def choose(lv):
        cap = max(params["n_cap"], abs(n))
        cands = [m for m in range(-params["m_max"], params["m_max"] + 1) if m != 0 and abs(n + m) <= cap and ok(m, lv)]
        cands.sort(key=lambda m: (abs(m), abs(e + m), 0 if side == 0 or side * m > 0 else 1, 0 if last == 0 or last * m > 0 else 1, -m))
        return cands[0] if cands else None

    nl = len(levels)
    seq = [f"{k}{l}" for l in range(nl) for k in "abc"] if params["order"] == "lexi" else \
        ["a0"] + [f"b{l}" for l in range(nl)] + ["c0"] + [f"{k}{l}" for l in range(1, nl) for k in "ac"]
    for item in seq:
        kind, lv = item[0], int(item[1:])
        if not usable[lv]:
            continue
        if kind == "a" and n != 0 and ok(-1 if n > 0 else 1, lv):
            m = -1 if n > 0 else 1
            return (YAW_RIGHT if m < 0 else YAW_LEFT), f"V2-back{lv}", lv, m
        if kind == "b" and ok(0, lv):
            return FORWARD_SLOW, f"V2-fwd{lv}", lv, 0
        if kind == "c":
            m = choose(lv)
            if m is not None:
                return (YAW_LEFT if m > 0 else YAW_RIGHT), f"V2-seek{lv}", lv, m
    return (last_turn if last_turn in (YAW_LEFT, YAW_RIGHT) else YAW_LEFT), "V2-stuck", None, None


class InvariantTests(unittest.TestCase):
    N_CASES = 1500

    @staticmethod
    def random_case(rng):
        name = rng.choice(["default"] + sorted(VARIANTS))
        params = dict(RULE_V2)
        if name != "default":
            params["levels"], params["order"] = VARIANTS[name]
        if rng.random() < 0.3:
            params["order"] = rng.choice(["lexi", "near"])
        params["yaw_step_deg"] = rng.choice([26.0, 31.7, 45.0])
        alpha = params["yaw_step_deg"]
        pose = (rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-180, 180))
        step = rng.randint(0, 60)
        points = [(rng.uniform(-0.3, 1.6), rng.uniform(-1.2, 1.2)) for _ in range(rng.randint(0, 14))]
        if rng.random() < 0.5:                            # 在若干候选航向的轴线上放点，让「堵」的情形足够多
            points += [at(rng.randint(-4, 4), rng.uniform(0.3, 1.3), alpha) for _ in range(rng.randint(1, 8))]
        d = [4.0 if rng.random() < 0.5 else round(rng.uniform(0.3, 4.0), 3) for _ in range(20)]
        tabu = [(pose[2] + rng.randint(-4, 4) * alpha + rng.uniform(-0.7, 0.7) * alpha, pose[0] + rng.uniform(-0.4, 0.4),
                 pose[1] + rng.uniform(-0.4, 0.4), step - rng.randint(0, 30), rng.randint(0, 2)) for _ in range(rng.randint(0, 5))]
        n = rng.randint(-5, 5)
        last_turn = rng.choice([None, YAW_LEFT, YAW_RIGHT])
        prev = rng.choice([None, FORWARD_SLOW, YAW_LEFT, YAW_RIGHT, SAFE_NEUTRAL])
        e = None if rng.random() < 0.3 else n + rng.uniform(-0.6, 0.6)
        return points, d, n, last_turn, tabu, step, pose, prev, e, params

    def cases(self):
        rng = random.Random(1111)
        for i in range(self.N_CASES):
            yield i, self.random_case(rng)

    def test_matches_reference_implementation(self):
        stuck = fwd = seek = back = 0
        for i, case in self.cases():
            got = rule_v2_entry(*case)
            ref = reference_v2(*case)
            self.assertEqual((got[0], got[1], got[2]["level"], got[2]["m"]), ref, f"case {i}")
            stuck += got[1] == "V2-stuck"
            fwd += got[1].startswith("V2-fwd")
            seek += got[1].startswith("V2-seek")
            back += got[1].startswith("V2-back")
        for name, count in (("stuck", stuck), ("fwd", fwd), ("seek", seek), ("back", back)):
            self.assertGreater(count, 20, f"随机用例里 {name} 太少（{count}），覆盖不到")

    def test_invariant_never_safe_neutral_and_entries_known(self):
        for i, case in self.cases():
            action, ent, info = rule_v2_entry(*case)
            self.assertIn(action, (FORWARD_SLOW, YAW_LEFT, YAW_RIGHT), f"case {i}")
            self.assertIn(ent, V2_ENTRIES, f"case {i}")
            self.assertEqual(info["level"] is None, ent == "V2-stuck")
            self.assertEqual(info["m"] is None, ent == "V2-stuck")

    def test_invariant_forward_only_if_some_usable_level_clear_ahead(self):
        for i, case in self.cases():
            points, d, n, last_turn, tabu, step, pose, prev, e, params = case
            action, ent, info = rule_v2_entry(*case)
            levels = params["levels"]
            clear_usable = [lv for lv, (r, dl, st) in enumerate(levels)
                            if not (st and prev == FORWARD_SLOW) and corridor_clear(points, 0.0, {"r_safe": r, "d_look": dl})]
            if action == FORWARD_SLOW:
                self.assertTrue(clear_usable, f"case {i}：前进但没有可用级的当前航向畅通")
                self.assertIn(info["level"], clear_usable)
                self.assertEqual(ent, f"V2-fwd{info['level']}")
            else:
                self.assertNotEqual(ent[:6], "V2-fwd")

    def test_invariant_chosen_heading_not_tabu_not_blocked(self):
        for i, case in self.cases():
            points, d, n, last_turn, tabu, step, pose, prev, e, params = case
            action, ent, info = rule_v2_entry(*case)
            if info["m"] in (None, 0):
                continue
            m, lv = info["m"], info["level"]
            r, dl, _st = params["levels"][lv]
            self.assertTrue(corridor_clear(points, m * params["yaw_step_deg"], {"r_safe": r, "d_look": dl}), f"case {i}")
            self.assertFalse(v2_tabu_hit(tabu, pose[2] + m * params["yaw_step_deg"], lv, step, pose[0], pose[1], params), f"case {i}")
            self.assertEqual(action, YAW_LEFT if m > 0 else YAW_RIGHT)
            self.assertLessEqual(abs(m), params["m_max"] if ent.startswith("V2-seek") else 1)

    def test_invariant_ok_bitmap_matches_geometry_and_tabu(self):
        for i, case in self.cases():
            points, d, n, last_turn, tabu, step, pose, prev, e, params = case
            info = rule_v2_entry(*case)[2]
            n_cap, alpha = params["n_cap"], params["yaw_step_deg"]
            self.assertEqual(len(info["ok"]), len(params["levels"]))
            for lv, (r, dl, _st) in enumerate(params["levels"]):
                for m in range(-n_cap, n_cap + 1):
                    want = corridor_clear(points, m * alpha, {"r_safe": r, "d_look": dl}) and \
                        (m == 0 or not v2_tabu_hit(tabu, pose[2] + m * alpha, lv, step, pose[0], pose[1], params))
                    self.assertEqual(info["ok"][lv][m + n_cap], int(want), f"case {i} level {lv} m {m}")

    def test_stuck_means_no_option_at_any_usable_level(self):
        for i, case in self.cases():
            points, d, n, last_turn, tabu, step, pose, prev, e, params = case
            action, ent, info = rule_v2_entry(*case)
            if ent != "V2-stuck":
                continue
            for lv, (r, dl, st) in enumerate(params["levels"]):
                if st and prev == FORWARD_SLOW:
                    continue
                ok = lambda m: corridor_clear(points, m * params["yaw_step_deg"], {"r_safe": r, "d_look": dl}) and \
                    (m == 0 or not v2_tabu_hit(tabu, pose[2] + m * params["yaw_step_deg"], lv, step, pose[0], pose[1], params))
                self.assertFalse(ok(0), f"case {i}")
                if n != 0:
                    self.assertFalse(ok(-1 if n > 0 else 1), f"case {i}")
                cap = max(params["n_cap"], abs(n))
                for m in range(-params["m_max"], params["m_max"] + 1):
                    if m != 0 and abs(n + m) <= cap:
                        self.assertFalse(ok(m), f"case {i} m {m}")


class StepTests(unittest.TestCase):
    def test_memory_kept_24_steps_not_8(self):
        d = rp.column_distances(0.0, 0.0, 0.0, [square(0.8, 0.0)])
        memory = rule_v2_step([], [], 0, 0.0, 0.0, 0.0, d, 0, None)[2]
        self.assertTrue(memory)
        mem_n = len(memory)
        later = rule_v2_step(memory, [], 20, 0.0, 0.0, 90.0, OPEN20, 0, None)
        self.assertEqual(later[4]["mem_n"], mem_n, "第 20 步仍保留第 0 步的点（v1 的 8 步窗口早已忘掉）")
        gone = rule_v2_step(memory, [], 25, 0.0, 0.0, 90.0, OPEN20, 0, None)
        self.assertEqual(gone[4]["mem_n"], 0, "年龄 25 > 24 清掉")

    def test_step_info_has_mem_n_and_tabu_n(self):
        d = rp.column_distances(0.0, 0.0, 0.0, [square(0.8, 0.0)])
        action, ent, memory, tabu, info = rule_v2_step([], [], 0, 0.0, 0.0, 0.0, d, 0, None)
        self.assertEqual(info["mem_n"], len(memory))
        self.assertEqual(info["tabu_n"], len(tabu))
        self.assertEqual(info["tabu_n"], 1)
        self.assertEqual(set(info), {"level", "m", "e", "usable", "ok", "clear", "mem_n", "tabu_n"})

    def test_memory_point_outside_current_view_still_blocks_candidate(self):
        params = {**RULE_V2, "yaw_step_deg": 45.0}
        d0 = rp.column_distances(0.0, 0.0, 0.0, [square(0.9 + 0.2, 0.0)])         # 正前方 0.9 m 处的柱
        memory = rule_v2_step([], [], 0, 0.0, 0.0, 0.0, d0, 0, None, params=params)[2]
        self.assertTrue(memory)
        *_, info = rule_v2_step(memory, [], 1, 0.0, 0.0, 90.0, OPEN20, 2, YAW_LEFT, params=params)    # 左转 90° 后柱子在视场外
        n_cap = params["n_cap"]
        self.assertEqual(info["ok"][0][n_cap - 2], 0, "朝柱子的航向（m = −2）由记忆判堵，虽然当前视场里什么也没有")
        self.assertEqual(info["ok"][0][n_cap], 1)
        self.assertEqual(info["ok"][0][n_cap + 2], 1)

    def test_deterministic(self):
        d = rp.column_distances(0.0, 0.0, 0.0, [square(0.8, 0.0)])
        a = rule_v2_step([], [], 0, 0.0, 0.0, 0.0, d, 0, None)
        b = rule_v2_step([], [], 0, 0.0, 0.0, 0.0, d, 0, None)
        self.assertEqual(a, b)


class LoopScenarioTests(unittest.TestCase):
    """K1 的失败形态：飞机在原地的 V1-more ↔ V1-cap 两步循环（三个候选通道都被判堵，规则没有出口）。"""
    WALL = [square(0.15, y) for y in (-2.0, -1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.6, 2.0)]   # 机头前 0.45 m 起的一堵宽墙（START_X = −0.5）

    def test_v1_loops_in_place_for_20_steps(self):
        r = rule_sim.simulate(self.WALL, "v1", steps=40, stop_at_goal=False)
        self.assertFalse(r["collided"])
        tail = r["entries"][-20:]
        self.assertNotIn("V1-fwd", r["entries"])
        self.assertTrue(set(tail) <= {"V1-more", "V1-cap"}, tail)
        self.assertEqual(rule_sim.max_reversal_run(r["actions"][-20:]), 19, "最近 20 步全是换向（L R L R …）")

    def test_v2_leaves_the_loop_within_2_n_cap_plus_3_steps(self):
        r = rule_sim.simulate(self.WALL, "v2", steps=40, stop_at_goal=False)
        first = next((i + 1 for i, (a, ent) in enumerate(zip(r["actions"], r["entries"])) if a == FORWARD_SLOW or ent == "V2-stuck"), None)
        self.assertIsNotNone(first, r["entries"])
        self.assertLessEqual(first, 2 * RULE_V2["n_cap"] + 3, r["entries"])
        self.assertGreater(min(rule_sim.min_gap(self.WALL, x, y) for x, y, _yaw in r["traj"][:first + 1]), rule_sim.BODY_RADIUS_M)

    def test_v2_has_no_period_two_turn_loop_there(self):
        r = rule_sim.simulate(self.WALL, "v2", steps=40, stop_at_goal=False)
        self.assertLess(rule_sim.max_reversal_run(r["actions"]), 2 * RULE_V2["n_cap"], r["entries"])

    def test_offline_sim_hands_the_previous_action_to_v2(self):
        """「只许静止起步」的级不许紧跟在前进步之后用：第 0 步（上一步算静止）走第 1 级的前进，下一步必须改走别的。离线模拟要记上一步执行的动作并传给规则。"""
        levels = ((0.35, 1.2, False), (0.20, 0.55, True))
        pillar = [square(0.7, -0.5)]                 # 柱面离机体中心 1.0 m，离航向轴最近 0.3 m：第 0 级（半宽 0.35）被堵，第 1 级（半宽 0.20）畅通
        r = rule_sim.simulate(pillar, "v2", steps=3, stop_at_goal=False, v2_params={**RULE_V2, "levels": levels, "order": "near"})
        self.assertEqual(r["entries"][:2], ["V2-fwd1", "V2-seek0"], r["entries"])
        self.assertEqual(r["actions"][:2], [FORWARD_SLOW, YAW_LEFT])


class OfflineTrainCheckTests(unittest.TestCase):
    def test_v2_not_worse_than_v1_on_20_train_geometries(self):
        geoms = rule_sim.train_geometries(20)
        v1 = [rule_sim.simulate(p, "v1", steps=80) for _s, p in geoms]
        v2 = [rule_sim.simulate(p, "v2", steps=80) for _s, p in geoms]
        p1, c1 = sum(r["success"] for r in v1), sum(r["collided"] for r in v1)
        p2, c2 = sum(r["success"] for r in v2), sum(r["collided"] for r in v2)
        print(f"\n[离线检查] train 分区 20 个几何（seed {[s for s, _ in geoms]}）：v1 过线 {p1}/20 撞柱 {c1}/20；"
              f"v2（RULE_V2 默认）过线 {p2}/20 撞柱 {c2}/20；v2 步数 {[r['steps'] for r in v2]}；"
              f"v2 最长换向 {[rule_sim.max_reversal_run(r['actions']) for r in v2]}", file=sys.stderr)
        self.assertGreaterEqual(p2, p1, "v2 过线数比 v1 少：照实报，不改 v2")
        self.assertLessEqual(c2, c1, "v2 撞柱数比 v1 多：照实报，不改 v2")

    def test_v2_never_outputs_safe_neutral_in_simulation(self):
        for _s, p in rule_sim.train_geometries(5):
            r = rule_sim.simulate(p, "v2", steps=80)
            self.assertNotIn(SAFE_NEUTRAL, r["actions"])
            self.assertTrue(set(r["entries"]) <= set(V2_ENTRIES))


if __name__ == "__main__":
    unittest.main()
