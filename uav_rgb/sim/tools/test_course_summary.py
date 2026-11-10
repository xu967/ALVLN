#!/usr/bin/env python3
"""experiments/2026-11-10-long-course/summarize.py 的单测：逐块存活判据、与 K=1 官方判据一致、采集中止归块、整表汇总（合成数据，不起 gz）。

运行：python3 sim/tools/test_course_summary.py（纯标准库 + eval_closed_loop，tools/check.sh 自动带上 test_*.py）
"""
import contextlib
import importlib.util
import io
import json
import os
import random
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_closed_loop as ec  # noqa: E402

_SPEC = importlib.util.spec_from_file_location(
    "long_course_summarize", os.path.join(HERE, "..", "..", "experiments", "2026-11-10-long-course", "summarize.py"))
sm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sm)

GOALS = [6.0, 11.0, 16.0]


def traj(xs, ys=None, collide=(), src=None):
    """xs[i] = 第 i 步结束后的 x，ys 同；collide = 第 i 步 collision=True 的步号，src 写进这些步。返回 (actions, oracle)。"""
    ys = ys or [0.0] * len(xs)
    actions = [{"step": i, "collision": i in collide, "src": src if i in collide else None, "action_executed": 1}
               for i in range(len(xs))]
    oracle = [{"step": i, "pose_gt": {"x": x, "y": y, "z": 2.0},
               "action_start_ns": int((13.5 + 0.9 * i) * 1e9), "action_end_ns": int((14.4 + 0.9 * i) * 1e9)}
              for i, (x, y) in enumerate(zip(xs, ys))]
    return actions, oracle


def ramp(top, step=0.3):
    n = int(top / step) + 1
    return [round(step * i, 3) for i in range(n)]


class BlockOutcomes(unittest.TestCase):
    def test_all_blocks_pass(self):
        xs = ramp(16.2)
        r = sm.block_outcomes(*traj(xs), GOALS)
        self.assertEqual(r["pass"], [True, True, True])
        self.assertIsNone(r["fail"])
        self.assertEqual(r["cross_step"], [next(i for i, x in enumerate(xs) if x >= g) for g in GOALS])

    def test_collision_in_second_block(self):
        xs = ramp(8.4)
        r = sm.block_outcomes(*traj(xs, collide={len(xs) - 1}, src="contact"), GOALS)
        self.assertEqual(r["pass"], [True, False, False])
        self.assertEqual(r["fail"], (1, "collision"))

    def test_corridor_before_crossing_blocks_every_later_block(self):
        xs = ramp(16.2)
        ys = [0.0] * len(xs)
        ys[10] = 3.2   # x = 3.0，第 1 块内越界；之后回到走廊并过了全部线，也不算
        r = sm.block_outcomes(*traj(xs, ys), GOALS)
        self.assertEqual(r["pass"], [False, False, False])
        self.assertEqual(r["fail"], (0, "corridor"))

    def test_stops_short_is_not_reached_or_aborted(self):
        xs = ramp(9.0)
        a, o = traj(xs)
        self.assertEqual(sm.block_outcomes(a, o, GOALS)["fail"], (1, "not_reached"))
        self.assertEqual(sm.block_outcomes(a, o, GOALS, aborted=True)["fail"], (1, "aborted"))
        self.assertEqual(sm.block_outcomes(a, o, GOALS)["pass"], [True, False, False])

    def test_stuck_is_not_collision_but_contact_stuck_is(self):
        xs = ramp(16.2)
        self.assertIsNone(sm.block_outcomes(*traj(xs, collide={5}, src="stuck"), GOALS)["fail"])
        r = sm.block_outcomes(*traj(xs, collide={5}, src="contact+stuck"), GOALS)
        self.assertEqual(r["fail"], (0, "collision"))
        self.assertEqual(r["pass"], [False, False, False])

    def test_event_on_the_crossing_step_counts(self):
        xs = ramp(7.0)
        i = next(i for i, x in enumerate(xs) if x >= 6.0)
        r = sm.block_outcomes(*traj(xs[:i + 1], collide={i}, src="contact"), GOALS)
        self.assertEqual(r["pass"], [False, False, False])
        self.assertEqual(r["fail"], (0, "collision"))
        ys = [0.0] * (i + 1)
        ys[i] = 3.5
        self.assertEqual(sm.block_outcomes(*traj(xs[:i + 1], ys), GOALS)["fail"], (0, "corridor"))

    def test_same_step_collision_and_corridor_is_collision(self):
        r = sm.block_outcomes(*traj([1.0, 2.0], [0.0, 3.4], collide={1}, src="contact"), GOALS)
        self.assertEqual(r["fail"], (0, "collision"))

    def test_first_block_matches_k1_judge_on_random_trajectories(self):
        """第 1 块的通过 = K=1 官方判据 success（collect 在首次过线即停，所以轨迹截到过线那一步）。"""
        rng = random.Random(20261110)
        agree = 0
        for _ in range(300):
            n = rng.randint(5, 60)
            xs, x = [], 0.0
            for _ in range(n):
                x += rng.uniform(-0.1, 0.4)
                xs.append(round(x, 3))
            ys = [round(rng.gauss(0, 1.4), 3) for _ in range(n)]
            collide = {i for i in range(n) if rng.random() < 0.03}
            src = rng.choice(["contact", "stuck", "contact+stuck"])
            first = next((i for i, x_ in enumerate(xs) if x_ >= 6.0), None)
            end = n if first is None else first + 1
            actions, oracle = traj(xs[:end], ys[:end], {i for i in collide if i < end}, src)
            official = ec.summarize_episode(actions, oracle, 6.0, corridor_half_width=3.0)["success"]
            mine = sm.block_outcomes(actions, oracle, [6.0], 3.0)["pass"][0]
            self.assertEqual(official, mine, (xs[:end], ys[:end], sorted(collide), src))
            agree += official
        self.assertGreater(agree, 10)   # 随机数据里要有足够多的通过集，不然上面的相等是空转


class Stats(unittest.TestCase):
    def test_wilson_and_mcnemar(self):
        lo, hi = sm.wilson(50, 60)
        self.assertAlmostEqual(lo, 0.7197, places=3)
        self.assertAlmostEqual(hi, 0.9069, places=3)
        self.assertEqual(sm.wilson(0, 0)[0] != sm.wilson(0, 0)[0], True)   # n=0 给 nan
        self.assertAlmostEqual(sm.mcnemar_exact(10, 2), 158 / 4096, places=9)
        self.assertEqual(sm.mcnemar_exact(0, 0), 1.0)
        self.assertEqual(sm.mcnemar_exact(3, 7), sm.mcnemar_exact(7, 3))

    def test_episode_stats(self):
        a, o = traj(ramp(3.0))
        s = sm.episode_stats(a, o)
        self.assertEqual(s["steps"], len(a))
        self.assertAlmostEqual(s["flight_s"], 0.9 * len(a) - 0.0, places=6)   # 起点 13.5、末点 14.4 + 0.9(n-1)
        self.assertAlmostEqual(s["sim_s"], 14.4 + 0.9 * (len(a) - 1), places=6)
        self.assertEqual(s["z_tail_min"], 2.0)


def write_episode(root, name, seed, xs, ys=None, collide=(), src="contact"):
    ep = os.path.join(root, name, "episodes", f"seed_{seed}")
    os.makedirs(ep, exist_ok=True)
    actions, oracle = traj(xs, ys, collide, src)
    for fname, rows in (("actions.jsonl", actions), ("oracle.jsonl", oracle)):
        with open(os.path.join(ep, fname), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    return os.path.join("episodes", f"seed_{seed}")


class WholeTable(unittest.TestCase):
    def test_teacher_arm_table_and_failed_episode_attribution(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            per_episode = []
            xs_pass = ramp(16.2)
            per_episode.append({"seed": 1, "success": True, "episode_dir": write_episode(tmp, "teacher", 1, xs_pass),
                                "block_goal_x": GOALS, "goal_x": 16.0})
            xs_hit = ramp(8.4)
            per_episode.append({"seed": 2, "success": False, "episode_dir": write_episode(tmp, "teacher", 2, xs_hit, collide={len(xs_hit) - 1}),
                                "block_goal_x": GOALS, "goal_x": 16.0})
            xs_abort = ramp(12.0)   # 采集中途失败（FAILSAFE）：不在 per_episode，在 failed；已过第 2 块的线
            write_episode(tmp, "teacher", 3, xs_abort)
            eval_json = {"blocks": 3, "corridor_half_width": 3.0, "partial": False, "episodes_requested": 3, "episodes_done": 2,
                         "failed_count": 1, "per_episode": per_episode,
                         "failed": [{"seed": 3, "rc": 1, "stderr_tail": ["RuntimeError: 固件进入 FAILSAFE"]}]}
            with open(os.path.join(tmp, "teacher", "eval.json"), "w", encoding="utf-8") as f:
                json.dump(eval_json, f)
            arm = sm.load_arm(os.path.join(tmp, "teacher"))
            self.assertTrue(arm["complete"])
            self.assertEqual(arm["eps"][3]["fail"], (2, "aborted"))
            self.assertEqual(arm["eps"][3]["pass"], [True, True, False])
            self.assertEqual(arm["eps"][2]["fail"], (1, "collision"))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sm.main(tmp), 0)
            with open(os.path.join(tmp, "summary.md"), encoding="utf-8") as f:
                text = f.read()
            self.assertIn("| 教师 | 3 | 3/3=1.00", text)           # S_1：三集都过了第 1 块
            self.assertIn("2/3=0.67", text)                         # S_2：集 1 与集 3
            self.assertIn("1/3=0.33", text)                         # S_3：只有集 1
            self.assertIn("| 教师 | 2 | 3 | 2 | 1 | 0 | 0 | 0 |", text)   # 块 2：进入 3、通过 2、撞柱 1
            self.assertIn("| 教师 | 3 | 2 | 1 | 0 | 0 | 0 | 1 |", text)   # 块 3：进入 2、通过 1、采集中止 1
            self.assertIn("S_K 与 eval.json 的 success 逐集不一致：0 集", text)
            self.assertIn("stderr 含 FAILSAFE 的 1 集", text)

    def test_no_data_returns_nonzero(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sm.main(tmp), 1)


if __name__ == "__main__":
    unittest.main()
