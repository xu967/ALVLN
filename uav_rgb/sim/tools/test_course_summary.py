#!/usr/bin/env python3
"""experiments/2026-11-10-long-course/summarize.py 的单测：逐块存活判据、与 K=1 官方判据一致、采集中止归块、整表汇总（合成数据，不起 gz）。

运行：python3 sim/tools/test_course_summary.py（纯标准库 + eval_closed_loop，tools/check.sh 自动带上 test_*.py）
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import random
import sys
import tempfile
import unittest
from unittest import mock

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


# ---------------------------------------------------------------- 第二轮新增：时序健康 / 航向漂移 / 块末 |y| / --present-only

def act(executed, disp=0.0, dyaw=0.0, collision=False, yaw_before=None):
    row = {"action_executed": executed, "collision": collision, "displacement_m": disp, "dyaw_deg": dyaw}
    if yaw_before is not None:
        row["yaw_before"] = yaw_before
    return row


def write_manifest_episode(root, name, seed, rows, steps=None, sim_s=None, goal_line_x=None, error=None):
    """合成一集的 episode_manifest.json（performance、goal_line_x、error）与 actions.jsonl；返回集目录。"""
    ep = os.path.join(root, name, "episodes", f"seed_{seed}")
    os.makedirs(ep, exist_ok=True)
    man = {"seed": seed, "error": error}
    if goal_line_x is not None:
        man["goal_line_x"] = goal_line_x
    if steps is not None:
        man["performance"] = {"steps": steps, "action_sim_s": sim_s}
    with open(os.path.join(ep, "episode_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    with open(os.path.join(ep, "actions.jsonl"), "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return ep


def yaw_rows(yaw0, steps):
    """steps = [(执行的动作, 该步 dyaw_deg)]；理想情形下一步的 yaw_before = 上一步的 yaw_before + dyaw。"""
    rows, y = [], yaw0
    for ex, dy in steps:
        rows.append({"action_executed": ex, "yaw_before": y, "dyaw_deg": dy})
        y += dy
    return rows


class TimingHealth(unittest.TestCase):
    def test_percentile_and_std(self):
        self.assertAlmostEqual(sm.percentile([0, 10, 20, 30, 40], 90), 36.0)   # 线性插值：位置 3.6
        self.assertEqual(sm.percentile([1, 2, 3, 4], 50), 2.5)
        self.assertEqual(sm.percentile([7], 90), 7)
        self.assertTrue(math.isnan(sm.percentile([], 50)))
        self.assertAlmostEqual(sm.std([2, 4, 4, 4, 5, 5, 7, 9]), 2.0)           # 总体标准差
        self.assertTrue(math.isnan(sm.std([])))
        lo, hi = sm.wilson(0, 20)
        self.assertEqual(lo, 0.0)                                              # 0/n 的下界不能是 -0.00
        self.assertEqual(sm.wilson(20, 20)[1], 1.0)

    def test_episode_timing_steady_forward_turns_and_eff(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            rows = [act(1, 0.26), act(1, 0.27), act(1, 0.25, collision=True), act(1, 0.26),
                    act(2, 0.04, dyaw=26.0), act(1, 0.20), act(3, 0.04, dyaw=-24.0), act(3, 0.04, dyaw=-22.0)]
            ep = write_manifest_episode(tmp, "a", 1, rows, steps=8, sim_s=7.0)
            t = sm.episode_timing(ep)
            self.assertAlmostEqual(t["eff"], 0.875)                  # action_sim_s / steps
            self.assertEqual(t["steps"], 8)
            self.assertEqual(t["fwd"], [0.27, 0.26])                 # 第 1、3 步：本步与上一步都是前飞、本步无碰撞；第 2 步本步碰撞，第 5 步上一步是转向
            self.assertEqual(t["yaw"], [26.0, 24.0, 22.0])           # |dyaw|，只取转向步
            self.assertIsNone(sm.episode_timing(write_manifest_episode(tmp, "a", 2, rows)))   # 没有 performance 的半截集
            self.assertIsNone(sm.episode_timing(os.path.join(tmp, "a", "episodes", "seed_404")))

    def test_arm_timing_pools_steps_but_takes_median_over_episodes(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            write_manifest_episode(tmp, "arm", 1, [act(1, 0.1), act(1, 0.2), act(1, 0.2), act(2, 0.0, dyaw=20.0)], steps=10, sim_s=8.0)
            write_manifest_episode(tmp, "arm", 2, [act(1, 0.3), act(1, 0.3), act(2, 0.0, dyaw=30.0), act(2, 0.0, dyaw=30.0), act(2, 0.0, dyaw=30.0)],
                                   steps=10, sim_s=9.0)
            write_manifest_episode(tmp, "arm", 3, [act(0)], steps=10, sim_s=10.0)
            write_manifest_episode(tmp, "arm", 4, [act(0)], steps=sm.LONG_STEPS, sim_s=0.85 * sm.LONG_STEPS)   # 刚到门槛的长集
            t = sm.arm_timing([os.path.join(tmp, "arm")])
            self.assertEqual(t["n"], 4)
            self.assertAlmostEqual(t["eff_med"], 0.875)               # 0.8 0.85 0.9 1.0 的中位数
            self.assertAlmostEqual(t["eff_min"], 0.8)
            self.assertEqual(t["n_long"], 1)                          # 只有集 4 够长
            self.assertAlmostEqual(t["eff_long_med"], 0.85)
            self.assertAlmostEqual(t["fwd"], (0.2 + 0.2 + 0.3) / 3)   # 池化：集 1 两步 + 集 2 一步
            self.assertAlmostEqual(t["yaw"], (20 + 30 * 3) / 4)       # 池化 27.5；按集平均会是 25
            self.assertIsNone(sm.arm_timing([os.path.join(tmp, "nope")]))

    def test_flags_use_three_percent_of_the_k1_reference(self):
        self.assertEqual(sm.timing_flags({"eff_med": 0.871 * 1.029, "yaw": 26.1 * 0.971}), [])
        bad = {"eff_med": 0.871 * 0.90, "yaw": 26.1 * 1.04}
        self.assertEqual(sm.timing_flags(bad), ["生效时长 -10.0%", "转角 +4.0%"])
        self.assertEqual(sm.timing_flags(bad, tol=0.05), ["生效时长 -10.0%"])
        self.assertEqual(sm.timing_flags({"eff_med": 0.871, "yaw": float("nan")}), [])    # 没有转向步：不标
        full = {"n": 4, "eff_med": 0.87, "eff_min": 0.85, "n_long": 3, "eff_long_med": 0.866, "fwd": 0.26, "yaw": 26.0}
        lines, flagged = sm.timing_lines([("参照行", full, True), ("偏短", {**full, "eff_med": 0.78}, False), ("空", None, False),
                                          ("无长集", {**full, "n_long": 0, "eff_long_med": None}, False)], ref_fwd=0.26)
        self.assertEqual(flagged, ["偏短"])                          # 参照行不进被标记列表
        self.assertIn("参照", lines[2])
        self.assertIn("⚠ 生效时长 -10.4%", lines[3])
        self.assertIn("| 3 | 0.866（-0.6%） |", lines[3])             # 长集单列：数量与它们的中位数（对 0.871）
        self.assertIn("没有可用的集", lines[4])
        self.assertIn("| 0 | - |", lines[5])                          # 没有长集：中位数留 -
        self.assertEqual(lines[0].count("|"), lines[2].count("|"))   # 表头与数据行列数一致

    def test_timing_only_label_may_contain_equals_sign(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            write_manifest_episode(tmp, "x1", 1, [act(1, 0.26), act(1, 0.26)], steps=2, sim_s=1.74)
            write_manifest_episode(tmp, "x2", 2, [act(1, 0.26), act(1, 0.26)], steps=2, sim_s=1.76)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                sm.timing_only([f"K=3 标定={tmp}/x1,{tmp}/x2", f"{tmp}/x1"])
            out = buf.getvalue()
            self.assertIn("| K=3 标定 | 2 | 0.875 |", out)         # 标签里的 = 不破坏切分；两集合并
            self.assertIn("| x1 | 1 | 0.870 |", out)               # 单个目录：标签取目录名


class HeadingDrift(unittest.TestCase):
    def test_counted_heading_vs_true_heading(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            steps = [(1, 0.0), (2, 20.0), (2, 20.0), (3, -20.0), (1, 0.0)]
            ep = write_manifest_episode(tmp, "g", 1, yaw_rows(10.0, steps))      # 第 0 步 yaw 不是 0：必须按相对值算
            d = sm.heading_drift([ep], seg=20, angle=26.1)
            self.assertEqual(sorted(d), [0])
            want = [0.0, 20 - 26.1, 40 - 52.2, 20 - 26.1, 20 - 26.1]            # 真实 - 26.1° × 含当步的净转向数
            for got, w in zip(d[0], want):
                self.assertAlmostEqual(got, w, places=6)
            d2 = sm.heading_drift([ep], seg=2, angle=26.1)                       # 按 2 步一段切
            self.assertEqual({g: len(v) for g, v in d2.items()}, {0: 2, 1: 2, 2: 1})

    def test_wraparound_at_180_degrees(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            rows = yaw_rows(170.0, [(2, 20.0), (1, 0.0)])
            rows[1]["yaw_before"] = -170.0                                       # 仿真给的 yaw 已折回 (-180, 180]
            d = sm.heading_drift([write_manifest_episode(tmp, "w", 1, rows)], seg=20, angle=26.1)
            self.assertAlmostEqual(d[0][0], 20 - 26.1, places=6)
            self.assertAlmostEqual(d[0][1], 20 - 26.1, places=6)                 # 跨 180° 后真实航向仍是相对第 0 步的 +20°

    def test_rows_missing_yaw_are_skipped_and_empty_is_empty(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            rows = yaw_rows(0.0, [(1, 0.0), (1, 0.0)])
            del rows[1]["dyaw_deg"]
            d = sm.heading_drift([write_manifest_episode(tmp, "m", 1, rows)])
            self.assertEqual(d[0], [0.0])
            self.assertEqual(sm.heading_drift([write_manifest_episode(tmp, "m", 2, [])]), {})

    def test_drift_lines_statistics(self):
        line, = sm.drift_lines("变体甲", {0: [3.0, -3.0, 3.0, -3.0]})
        self.assertEqual(line, "| 变体甲 | 0–19 | 4 | +0.0 | 3.0 | 3.0 | 3.0 | 3.0 |")
        line, = sm.drift_lines("变体甲", {2: [float(i) for i in range(1, 11)]}, seg=20)
        self.assertEqual(line, "| 变体甲 | 40–59 | 10 | +5.5 | 2.9 | 5.5 | 9.1 | 10.0 |")


class BlockEndY(unittest.TestCase):
    def test_cross_y_is_abs_y_at_the_crossing_step(self):
        xs = ramp(16.2)
        i = [next(j for j, x in enumerate(xs) if x >= g) for g in GOALS]
        ys = [0.0] * len(xs)
        ys[i[0]], ys[i[1]], ys[i[2]] = -1.5, 0.7, 2.2
        self.assertEqual(sm.block_outcomes(*traj(xs, ys), GOALS)["cross_y"], [1.5, 0.7, 2.2])
        short = sm.block_outcomes(*traj(ramp(8.4), [0.4] * len(ramp(8.4))), GOALS)
        self.assertEqual(short["cross_y"], [0.4, None, None])

    def test_block_end_lines(self):
        rows = {1: {"cross_y": [1.0, 2.0, None]}, 2: {"cross_y": [3.0, None, None]}, 3: {"cross_y": [2.0, None, None]}, 4: {}}
        lines = sm.block_end_y_lines("变体乙", rows, 3)
        self.assertEqual(lines[0], "| 变体乙 | 1 | 3 | 2.00 | 2.00 | 2.80 | 3.00 |")
        self.assertEqual(lines[1], "| 变体乙 | 2 | 1 | 2.00 | 2.00 | 2.00 | 2.00 |")
        self.assertEqual(lines[2], "| 变体乙 | 3 | 0 | | | | |")


class PresentOnly(unittest.TestCase):
    def _interrupted_arm(self, tmp):
        """calswa-s1：评了 2 集，第 3 集被 SIGTERM 中断（只出现在 failed，没有 episodes/seed_N）；请求 20 集。"""
        per_episode = []
        for seed, top in ((1, 16.2), (2, 8.4)):
            per_episode.append({"seed": seed, "success": top > 16, "episode_dir": write_episode(tmp, "calswa-s1", seed, ramp(top)),
                                "block_goal_x": GOALS, "goal_x": 16.0})
        eval_json = {"blocks": 3, "corridor_half_width": 3.0, "partial": True, "episodes_requested": 20, "episodes_done": 2, "failed_count": 1,
                     "interrupted": {"seed": 9, "reason": "SIGTERM"}, "per_episode": per_episode,
                     "failed": [{"seed": 9, "error": "interrupted: SIGTERM"}]}
        with open(os.path.join(tmp, "calswa-s1", "eval.json"), "w", encoding="utf-8") as f:
            json.dump(eval_json, f)
        return os.path.join(tmp, "calswa-s1")

    def test_interrupted_episode_is_not_a_data_point_in_present_only(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            d = self._interrupted_arm(tmp)
            full = sm.load_arm(d)
            self.assertEqual((full["requested"], len(full["eps"]), full["failed"]), (20, 3, 1))   # 默认口径：分母 20，被中断的集算没通过
            po = sm.load_arm(d, present_only=True)
            self.assertEqual((po["requested"], sorted(po["eps"]), po["failed"]), (2, [1, 2], 0))
            self.assertTrue(po["interrupted"] and not po["complete"])

    def test_arm_without_eval_json_is_read_from_episode_dirs(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            for seed, top, err in ((5, 16.2, None), (6, 8.4, None), (7, 12.0, "step 137: cf_sitl 执行器故障: 固件进入 FAILSAFE")):
                ep = os.path.join(tmp, "arm", write_episode(tmp, "arm", seed, ramp(top)))
                with open(os.path.join(ep, "episode_manifest.json"), "w", encoding="utf-8") as f:
                    json.dump({"seed": seed, "goal_line_x": 16.0, "error": err}, f)
            d = os.path.join(tmp, "arm")
            self.assertIsNone(sm.load_arm(d))                                  # 默认口径：没有 eval.json 就是没数据
            arm = sm.load_arm(d, present_only=True)
            self.assertEqual((arm["K"], arm["requested"], arm["failed"], arm["complete"]), (3, 3, 1, False))
            self.assertTrue(arm["eps"][5]["success"] and not arm["eps"][6]["success"])
            self.assertEqual(arm["eps"][7]["fail"], (2, "aborted"))           # 过了第 2 块的线，采集中止在第 3 块
            self.assertIn("FAILSAFE", arm["eps"][7]["stderr"])
            self.assertIsNone(sm.load_arm(os.path.join(tmp, "nope"), present_only=True))

    def test_main_title_present_only_and_new_sections(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            self._interrupted_arm(tmp)
            os.rename(os.path.join(tmp, "calswa-s1"), os.path.join(tmp, "calswa-s0"))          # 变体「calswa（无护栏）」的第 0 个训练种子
            # 教师臂：没有 eval.json，只有轨迹与 manifest；生效时长 1.56 / 2 = 0.78 s，比参照 0.871 短 10.4%，应被标 ⚠
            write_episode(tmp, "teacher", 3, ramp(16.2))
            with open(os.path.join(tmp, "teacher", "episodes", "seed_3", "episode_manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"seed": 3, "goal_line_x": 16.0, "error": None, "performance": {"steps": 2, "action_sim_s": 1.56}}, f)
            with mock.patch.object(sm, "GUARD", os.path.join(tmp, "no-guard-refs")), mock.patch.object(sm, "SWA", os.path.join(tmp, "no-swa-refs")):
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(sm.main(tmp, title="第一轮，时序混杂，20:46 中止", present_only=True), 0)
            with open(os.path.join(tmp, "summary.md"), encoding="utf-8") as f:
                text = f.read()
            self.assertIn("（第一轮，时序混杂，20:46 中止）", text.splitlines()[0])
            self.assertIn("| 变体 | 有轨迹集数 |", text)
            self.assertIn("| calswa（无护栏） | 2 | 2/2=1.00", text)             # 分母 = 有轨迹的 2 集，不是请求的 20 集
            self.assertIn("1/2=0.50", text)
            self.assertIn("已中断（有轨迹 2 集）", text)
            self.assertIn("> ⚠ 时序健康：1 个臂", text)
            self.assertIn("| teacher | 1 | 0.780 | 0.780 | -10.4% |", text)
            for head in ("## 8. 时序健康", "## 9. 护栏臂", "## 10. 块末"):
                self.assertIn(head, text)
            self.assertIn("| 教师 | 1 | 1 |", text)                             # 块末 |y| 表里有教师行


if __name__ == "__main__":
    unittest.main()
