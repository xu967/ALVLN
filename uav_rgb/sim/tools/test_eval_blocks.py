#!/usr/bin/env python3
"""W1-4「逐块存活与每百米碰撞数进 eval.json」的测试。不起 Gazebo。

背景：长课程 --task course_base 的逐块存活原先只在实验目录的汇总脚本里算（experiments/2026-11-10-long-course/summarize.py
的 block_outcomes），eval.json 里没有。现在判据只有一份，在 eval_closed_loop.py：
  blocks_passed(oracle, actions, block_goal_x, corridor_half_width)  → 通过的块数
  block_outcomes(actions, oracle, goals, half_width, aborted)         → 逐块明细（summarize.py 改为 import 它）
  flight_path_m(oracle, actions, goal_x, ...)                         → 一集量到首次碰撞 / 越界 / 过线的路程
eval.json 的新键（只在下面写明的条件下出现，其余键与值不变）：
  per_episode.blocks_passed / blocks_total   只在场景 meta 带 course（course_base）时
  block_survival                             只在 course_base；长度 K，S_k = blocks_passed ≥ k 的集数 ÷ 请求集数
  collisions_per_100m                        所有任务；100 × 接触碰撞的集数 ÷ 各完成集路程之和

覆盖：blocks_passed 三种轨迹（第 1 块撞 → 0、第 2 块越界 → 1、全程通过 → 3）与边界；flight_path_m 的截断规则；
  完整 ec.main()（假 scene_gen 出真 meta、假 collect 写指定轨迹）走到 eval.json：course_base 与 base 的键集合
  （旧键集冻结自 main a714b54 的真实输出）、block_survival 的分母是请求集数、失败集算没通过、per_episode 不泄漏内部键；
  summarize.py 与 eval_closed_loop 共用同一份 block_outcomes；ec.aggregate / summarize_episode 的键集合不变
  （3DGS 评测与 tools/compare_datasets.py 共用它们）。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_eval_blocks.py
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_closed_loop as ec  # noqa: E402
import scene_course  # noqa: E402
import scene_gen as sg  # noqa: E402

LONG_COURSE_SUMMARIZE = os.path.join(HERE, "..", "..", "experiments", "2026-11-10-long-course", "summarize.py")
GOALS = [6.0, 11.0, 16.0]          # 3 块课程：第 k 块最远柱 x + 1.0（scene_course 的 block_goal_x 的形状）

# 旧代码（main a714b54）下 ec.main() / ec.aggregate / ec.summarize_episode 的真实键集合，是本文件的 run_eval 在旧树上的输出。
# course_base 比 base 多的旧键只有 blocks（顶层）与 block_goal_x（每集）。
OLD_TOP_KEYS_BASE = {
    'action_distribution', 'appearance_version', 'attempted_seeds', 'aux_cameras', 'camera_rate_hz', 'ckpt',
    'ckpt_sha256', 'coast_window_s', 'collision_free_rate', 'collision_rate', 'completion_rate',
    'corridor_half_width', 'corridor_violations', 'density', 'dynamics', 'episodes_done', 'episodes_requested',
    'execution_rate', 'failed', 'failed_count', 'false_forward_rate', 'final_in_radius_rate',
    'geometry_version', 'goal_margin', 'goal_radius', 'goal_radius_reach', 'goal_radius_stop', 'hold_s',
    'infer_bin', 'infer_bin_sha256', 'inference_ms', 'interrupted', 'metric_version', 'ood', 'partial',
    'per_episode', 'policy', 'protocol', 'protocol_mismatches', 'reach_ever_rate', 'requested_seeds',
    'requested_spl', 'requested_success_rate', 'rl_checkpoint', 'scanned_seeds', 'schema', 'seeds', 'skipped',
    'skipped_count', 'spl', 'spl_n', 'spl_reference', 'split', 'stable_termination_rate', 'steps',
    'stop_action_issued_rate', 'stop_radius', 'stop_success_rate', 'strict_protocol', 'stuck_free_rate',
    'stuck_rate', 'success_criterion', 'success_rate', 'task', 'task_success_rate', 'teacher_planner',
    'template_sha256', 'text_dim', 'text_mode', 'unique_gids', 'unique_maps', 'vehicle', 'video_quality',
    'wall_s', 'waypoint_radius', 'wilson_ci', 'window_from'
}

OLD_EPISODE_KEYS_BASE = {
    'actions', 'appearance_version', 'camera_profile', 'collect_rc', 'collision', 'collision_free',
    'corridor_violation', 'dedup_key', 'dynamics', 'dynamics_profile', 'episode_dir', 'false_forward', 'family',
    'final_in_radius', 'final_waypoint_dist', 'geometry_gid', 'goal_radius', 'goal_radius_reach',
    'goal_radius_stop', 'goal_x', 'instruction', 'map_id', 'max_x', 'metric_version', 'path_length_m',
    'performance', 'pillars', 'policy', 'reach_ever', 'render_backend_requested', 'resource_sha256', 'seed',
    'shortest_path_m', 'spl', 'split', 'split_policy', 'stable_termination', 'stderr_tail', 'steps',
    'steps_to_goal', 'stop_action_issued', 'stop_radius', 'stop_success', 'stopped', 'stopped_on_goal', 'stuck',
    'stuck_free', 'success', 'success_criterion', 'target_color', 'task', 'task_success', 'task_version',
    'template_sha256', 'text_mode', 'timed_out', 'vehicle', 'wall_s', 'waypoint', 'waypoint_min_dist',
    'waypoint_radius', 'window_from', 'world_sha256', 'y_max_m'
}

OLD_AGGREGATE_KEYS = {
    'action_distribution', 'collision_free_rate', 'collision_rate', 'completion_rate', 'corridor_violations',
    'episodes_done', 'execution_rate', 'failed_count', 'false_forward_rate', 'final_in_radius_rate',
    'inference_ms', 'reach_ever_rate', 'requested_spl', 'requested_success_rate', 'skipped_count', 'spl',
    'spl_n', 'stable_termination_rate', 'stop_action_issued_rate', 'stop_success_rate', 'stuck_free_rate',
    'stuck_rate', 'success_rate', 'task_success_rate', 'unique_gids', 'wilson_ci'
}

OLD_SUMMARY_KEYS = {
    'collision', 'collision_free', 'corridor_violation', 'false_forward', 'final_in_radius',
    'final_waypoint_dist', 'goal_radius_reach', 'goal_radius_stop', 'infer_ms', 'max_x', 'metric_version',
    'path_length_m', 'reach_ever', 'shortest_path_m', 'spl', 'stable_termination', 'steps', 'steps_to_goal',
    'stop_action_issued', 'stop_success', 'stopped', 'stuck', 'stuck_free', 'success', 'task_success',
    'timed_out', 'waypoint_min_dist', 'y_max_m'
}

OLD_TOP_KEYS_COURSE = OLD_TOP_KEYS_BASE | {"blocks"}
OLD_EPISODE_KEYS_COURSE = OLD_EPISODE_KEYS_BASE | {"block_goal_x"}


def ramp(top, step=0.3):
    return [round(step * i, 3) for i in range(int(round(top / step)) + 1)]


def traj(xs, ys=None, collide=(), src="contact"):
    """xs[i] = 第 i 步结束后的 x；collide = collision=True 的步号。返回 (oracle, actions)（blocks_passed 的参数顺序）。"""
    ys = ys or [0.0] * len(xs)
    actions = [{"step": i, "collision": i in collide, "src": src if i in collide else None, "action_executed": 1}
               for i in range(len(xs))]
    oracle = [{"step": i, "pose_gt": {"x": x, "y": y, "z": 2.0}} for i, (x, y) in enumerate(zip(xs, ys))]
    return oracle, actions


def polyline(points):
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def first_block_collision():
    """x 走到 4.5（第 1 块内）在最后一步撞柱。"""
    xs = ramp(4.5)
    return traj(xs, collide={len(xs) - 1})


CORRIDOR_EXIT_X = 8.4


def second_block_corridor():
    """过了第 1 块线（x=6.0），在第 2 块内（x=8.4）一步 |y|=3.4 出走廊，下一步回到中线，继续飞过第 2 块线（x=11.0）到 11.4。

    出界之后飞行没有结束：第 2 块过没过线由「出界」决定，不是由轨迹在哪一步断了决定。
    """
    xs = ramp(11.4)
    ys = [0.0] * len(xs)
    ys[xs.index(CORRIDOR_EXIT_X)] = 3.4
    return traj(xs, ys)


def path_to_corridor_exit():
    """second_block_corridor 的位姿折线量到出走廊那一步为止（含那一步）。"""
    xs = ramp(11.4)
    i = xs.index(CORRIDOR_EXIT_X)
    return [(x, 0.0) for x in xs[:i]] + [(xs[i], 3.4)]


def full_pass():
    return traj(ramp(16.2))


class BlocksPassed(unittest.TestCase):
    def test_collision_in_first_block_passes_none(self):
        self.assertEqual(ec.blocks_passed(*first_block_collision(), GOALS, 3.0), 0)

    def test_corridor_exit_in_second_block_passes_one(self):
        self.assertEqual(ec.blocks_passed(*second_block_corridor(), GOALS, 3.0), 1)

    def test_full_course_passes_three(self):
        self.assertEqual(ec.blocks_passed(*full_pass(), GOALS, 3.0), 3)

    def test_argument_order_is_oracle_then_actions(self):
        """卡里的签名是 (oracle, actions, ...)，与同文件 summarize_episode(actions, oracle, ...) 相反：顺序换了会红。"""
        oracle, actions = full_pass()
        self.assertEqual(ec.blocks_passed(oracle, actions, GOALS, 3.0), 3)
        self.assertEqual(ec.blocks_passed(actions, oracle, GOALS, 3.0), 0)

    def test_without_corridor_the_lateral_check_is_skipped(self):
        """评测没给 --corridor-half-width（None）时 success 也不查横向，逐块判据与它同口径。"""
        oracle, actions = second_block_corridor()
        self.assertEqual(ec.blocks_passed(oracle, actions, GOALS, None), 2, "第 2 块内的 |y|=3.4 不算越界，飞过 11.0 就过第 2 块")
        self.assertEqual(ec.blocks_passed(oracle, actions, GOALS, 3.0), 1)
        xs = ramp(16.2)
        ys = [0.0] * len(xs)
        ys[3] = 9.0
        self.assertEqual(ec.blocks_passed(*traj(xs, ys), GOALS, None), 3)
        self.assertEqual(ec.blocks_passed(*traj(xs, ys), GOALS, 3.0), 0, "给了半宽：第 1 块内越界后一块都不算")

    def test_stuck_is_not_a_collision_but_contact_stuck_is(self):
        xs = ramp(16.2)
        self.assertEqual(ec.blocks_passed(*traj(xs, collide={5}, src="stuck"), GOALS, 3.0), 3)
        self.assertEqual(ec.blocks_passed(*traj(xs, collide={5}, src="contact+stuck"), GOALS, 3.0), 0)

    def test_event_on_the_crossing_step_blocks_that_block(self):
        xs = ramp(7.0)
        i = next(i for i, x in enumerate(xs) if x >= 6.0)
        self.assertEqual(ec.blocks_passed(*traj(xs[:i + 1], collide={i}), GOALS, 3.0), 0)
        self.assertEqual(ec.blocks_passed(*traj(xs[:i + 1]), GOALS, 3.0), 1)

    def test_empty_and_short_trajectories(self):
        self.assertEqual(ec.blocks_passed([], [], GOALS, 3.0), 0)
        self.assertEqual(ec.blocks_passed(*traj(ramp(5.0)), GOALS, 3.0), 0)

    def test_summarize_py_shares_the_single_copy_of_block_outcomes(self):
        spec = importlib.util.spec_from_file_location("long_course_summarize_w14", LONG_COURSE_SUMMARIZE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertIs(mod.block_outcomes, ec.block_outcomes, "summarize.py 不许再留自己的那份逐块判据")
        oracle, actions = full_pass()
        self.assertEqual(sum(ec.block_outcomes(actions, oracle, GOALS)["pass"]), 3, "缺省走廊半宽 3.0 m")


class FlightPath(unittest.TestCase):
    def test_stops_at_the_first_contact_collision_inclusive(self):
        oracle, actions = first_block_collision()
        self.assertAlmostEqual(ec.flight_path_m(oracle, actions, 6.0), 4.5, places=9)
        xs = ramp(4.5) + [4.8, 5.1]          # 碰撞之后的位姿（真实评测 --stop-on-collision 不会有）不进路程
        oracle, actions = traj(xs, collide={len(ramp(4.5)) - 1})
        self.assertAlmostEqual(ec.flight_path_m(oracle, actions, 6.0), 4.5, places=9)

    def test_stuck_is_not_a_stopping_event(self):
        oracle, actions = traj(ramp(3.0), collide={4}, src="stuck")
        self.assertAlmostEqual(ec.flight_path_m(oracle, actions, 6.0), 3.0, places=9)

    def test_stops_at_the_first_corridor_exit_only_when_a_half_width_is_given(self):
        oracle, actions = second_block_corridor()
        self.assertAlmostEqual(ec.flight_path_m(oracle, actions, 16.0, corridor_half_width=3.0),
                               polyline(path_to_corridor_exit()), places=9, msg="出界之后回到中线的那段不算")
        whole = [(o["pose_gt"]["x"], o["pose_gt"]["y"]) for o in oracle]
        self.assertAlmostEqual(ec.flight_path_m(oracle, actions, 16.0), polyline(whole), places=9,
                               msg="没给半宽：整条轨迹（没到 16.0 的线）")

    def test_stops_at_the_goal_line_with_interpolation_even_if_the_flight_continues(self):
        n = len(ramp(16.2))
        oracle, actions = traj(ramp(16.2) + [16.5], collide={n})   # 过线之后才撞
        self.assertAlmostEqual(ec.flight_path_m(oracle, actions, 16.0, corridor_half_width=3.0), 16.0, places=9)

    def test_waypoint_task_stops_in_the_reach_disc_and_ignores_the_corridor(self):
        oracle, actions = traj([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 3.6, 3.6, 3.6, 3.6])
        wp = {"x": 4.0, "y": 3.6}
        got = ec.flight_path_m(oracle, actions, 6.0, waypoint=wp, stop_radius=0.5, goal_radius=1.5,
                               corridor_half_width=3.0)
        pts = [(0.0, 0.0), (1.0, 3.6), (2.0, 3.6), (2.5, 3.6)]   # 进半径 1.5 的圆的点：(2.5, 3.6)；|y|=3.6 不触发越界
        self.assertAlmostEqual(got, polyline(pts), places=9)

    def test_no_pose_rows_gives_none(self):
        self.assertIsNone(ec.flight_path_m([], [], 6.0))
        self.assertIsNone(ec.flight_path_m([{"step": 0}], [{"collision": False}], 6.0))

    def test_equals_the_spl_path_when_no_collision_or_exit(self):
        """没有碰撞与越界时量到的就是 summarize_episode 的 p_i（同一个目标区），两处的目标区写法不许漂移。"""
        for stop_radius, wp in ((None, None), (0.5, {"x": 3.0, "y": 1.0})):
            oracle, actions = traj([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.5, 1.0, 1.0, 0.0])
            s = ec.summarize_episode(actions, oracle, 6.0, wp, stop_radius=stop_radius)
            self.assertAlmostEqual(ec.flight_path_m(oracle, actions, 6.0, wp, stop_radius=stop_radius),
                                   s["path_length_m"], places=9)


class AggregateHelpers(unittest.TestCase):
    def test_collisions_per_100m(self):
        eps = [{"collision": True, ec.FLIGHT_PATH_KEY: 4.5}, {"collision": False, ec.FLIGHT_PATH_KEY: 15.5},
               {"collision": True, ec.FLIGHT_PATH_KEY: 10.0}]
        self.assertAlmostEqual(ec.collisions_per_100m(eps), 100.0 * 2 / 30.0, places=9)
        self.assertEqual(ec.collisions_per_100m([{"collision": False, ec.FLIGHT_PATH_KEY: 8.0}]), 0.0)

    def test_collisions_per_100m_is_none_without_distance(self):
        self.assertIsNone(ec.collisions_per_100m([]))
        self.assertIsNone(ec.collisions_per_100m([{"collision": True, ec.FLIGHT_PATH_KEY: None}]))
        self.assertIsNone(ec.collisions_per_100m([{"collision": True, ec.FLIGHT_PATH_KEY: 0.0}]))

    def test_block_survival_denominator_is_the_requested_count(self):
        eps = [{"blocks_passed": 0}, {"blocks_passed": 1}, {"blocks_passed": 3}]
        self.assertEqual(ec.block_survival(eps, 3, 3), [2 / 3, 1 / 3, 1 / 3])
        self.assertEqual(ec.block_survival(eps, 4, 3), [2 / 4, 1 / 4, 1 / 4], "多出的 1 个请求集（失败集）算没通过")
        self.assertEqual(ec.block_survival([], 5, 3), [0.0, 0.0, 0.0], "一集也没跑完：K 个 0，不是缺键")

    def test_block_survival_is_non_increasing(self):
        eps = [{"blocks_passed": n} for n in (0, 2, 2, 3, 1, 3)]
        s = ec.block_survival(eps, 6, 3)
        self.assertEqual(s, [5 / 6, 4 / 6, 2 / 6])
        self.assertEqual(s, sorted(s, reverse=True))


# ---------------------------------------------------------------- 完整 ec.main()（假 scene_gen / 假 collect）

def run_eval(task, trajectories, failing=(), corridor="3.0", real_spl=False, extra_argv=()):
    """完整 ec.main()：假 scene_gen 用真生成器出 meta，假 collect 把 trajectories[seed] = (oracle, actions) 写成集目录。

    failing 里的 seed：collect 返回 rc=1 且不写轨迹（算采集失败）。corridor=None 不给 --corridor-half-width。返回 eval.json。
    SPL 的参考路径要跑 A*（K=3 每集约 2 s），与本卡无关，默认打桩成 None；real_spl=True 走真函数（新旧树差分用）。
    extra_argv：追加到命令行末尾（W1-3 的 test_eval_frame_gap.py 用它给 --hold）。
    """
    seeds = list(trajectories) + [s for s in failing if s not in trajectories]

    def generate(cmd, **_kw):
        world = Path(cmd[cmd.index("--out") + 1])
        seed = int(cmd[cmd.index("--seed") + 1])
        template = sg._read_template(cmd[cmd.index("--template") + 1])
        if task == "course_base":
            _, meta = scene_course.build_scene(template, seed, seed, int(cmd[cmd.index("--blocks") + 1]), "v1")
        else:
            _, meta = sg.build_scene(template, seed, seed, task=task, appearance="v1")
        world.write_text("<sdf/>")
        world.with_suffix(".json").write_text(json.dumps(meta))
        return Mock(returncode=0)

    def collect_proc(cmd, **_kw):
        ep = Path(cmd[cmd.index("--out") + 1])
        seed = int(cmd[cmd.index("--seed") + 1])
        proc = Mock(returncode=1 if seed in failing else 0)
        proc.communicate.return_value = ("", "")
        if seed in failing:
            return proc
        oracle, actions = trajectories[seed]
        (ep / "actions.jsonl").write_text("".join(json.dumps(a) + "\n" for a in actions))
        (ep / "oracle.jsonl").write_text("".join(json.dumps(o) + "\n" for o in oracle))
        (ep / "episode_manifest.json").write_text("{}")
        return proc

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "eval"
        argv = ["eval_closed_loop.py", "--policy", "forward", "--task", task, "--seeds", ",".join(map(str, seeds)),
                "--steps", "200", "--out", str(out)]
        if corridor is not None:
            argv += ["--corridor-half-width", corridor]
        if task == "course_base":
            argv += ["--blocks", "3"]
        argv += list(extra_argv)
        with contextlib.ExitStack() as st:
            st.enter_context(patch.object(ec.sys, "argv", argv))
            st.enter_context(patch.object(ec.cp, "XDG_RUNTIME_DIR", str(Path(d) / "xdg")))
            st.enter_context(patch.object(ec.subprocess, "run", side_effect=generate))
            st.enter_context(patch.object(ec.subprocess, "Popen", side_effect=collect_proc))
            st.enter_context(patch.object(ec.cp, "_start_sim", return_value=Mock()))
            for target, name in ((ec.cp, "wait_ready"), (ec.cp, "_register"), (ec.cp, "stop_group"),
                                 (ec.cp, "cleanup_all"), (ec.time, "sleep"), (ec.signal, "signal")):
                st.enter_context(patch.object(target, name))
            if not real_spl:
                st.enter_context(patch.object(ec, "episode_shortest_path_m", return_value=None))
            st.enter_context(contextlib.redirect_stdout(io.StringIO()))
            st.enter_context(contextlib.redirect_stderr(io.StringIO()))
            ec.main()
        return json.loads((out / "eval.json").read_text())


def course_trajectories():
    return {1: first_block_collision(), 113: second_block_corridor(), 122: full_pass()}


def base_trajectories():
    """base 柱阵（成功线 x=6.0）：撞柱 / 出走廊 / 过线。出走廊的那条在倒数第 3 步 |y|=3.5 后回到走廊。"""
    xs = ramp(2.4)
    ys = [0.0] * len(ramp(6.3))
    ys[-3] = 3.5
    return {1: traj(xs, collide={len(xs) - 1}), 113: traj(ramp(6.3), ys), 122: traj(ramp(6.3))}


class CourseReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_eval("course_base", course_trajectories())

    def test_per_episode_blocks_passed_and_total(self):
        eps = {e["seed"]: e for e in self.report["per_episode"]}
        self.assertEqual({s: e["blocks_passed"] for s, e in eps.items()}, {1: 0, 113: 1, 122: 3})
        self.assertEqual({e["blocks_total"] for e in eps.values()}, {3})
        self.assertEqual(len(eps[122]["block_goal_x"]), 3, "block_goal_x 照旧")

    def test_block_survival_denominator_is_requested(self):
        self.assertEqual(self.report["block_survival"], [2 / 3, 1 / 3, 1 / 3])
        self.assertEqual(self.report["episodes_requested"], 3)

    def test_last_block_survival_matches_requested_success_rate(self):
        """S_K 与 success 逐集应当一致：开了走廊判据时 S_K = requested_success_rate。"""
        self.assertEqual(self.report["block_survival"][-1], self.report["requested_success_rate"])

    def test_collisions_per_100m_uses_path_to_first_event(self):
        """分子 1（seed 1 撞柱）；分母 = 4.5（撞柱那步）+ 到出走廊那步的折线 + 16.0（量到过线点，不含 16.0 之后的 0.2）。"""
        total = 4.5 + polyline(path_to_corridor_exit()) + 16.0
        self.assertAlmostEqual(self.report["collisions_per_100m"], 100.0 * 1 / total, places=9)

    def test_failed_collection_counts_in_the_denominator_as_not_passed(self):
        report = run_eval("course_base", course_trajectories(), failing=(153,))
        self.assertEqual(report["episodes_requested"], 4)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["block_survival"], [2 / 4, 1 / 4, 1 / 4])
        self.assertEqual(len(report["per_episode"]), 3, "失败集不进 per_episode")

    def test_nothing_finished_gives_zero_survival_and_no_collision_rate(self):
        report = run_eval("course_base", {}, failing=(1, 113))
        self.assertEqual(report["block_survival"], [0.0, 0.0, 0.0])
        self.assertIsNone(report["collisions_per_100m"])

    def test_without_the_corridor_flag_the_lateral_check_is_skipped(self):
        report = run_eval("course_base", course_trajectories(), corridor=None)
        eps = {e["seed"]: e for e in report["per_episode"]}
        self.assertEqual(eps[113]["blocks_passed"], 2, "没开走廊：第 2 块内的 |y|=3.4 不算越界，与 success 同口径")
        self.assertEqual(report["block_survival"], [2 / 3, 2 / 3, 1 / 3], "seed 1 / 113 / 122 通过 0 / 2 / 3 块")
        self.assertEqual(report["block_survival"][-1], report["requested_success_rate"])

    def test_key_sets(self):
        self.assertEqual(set(self.report) - OLD_TOP_KEYS_COURSE, {"collisions_per_100m", "block_survival"})
        self.assertEqual(OLD_TOP_KEYS_COURSE - set(self.report), set(), "旧键一个不少")
        ep = self.report["per_episode"][0]
        self.assertEqual(set(ep) - OLD_EPISODE_KEYS_COURSE, {"blocks_passed", "blocks_total"})
        self.assertEqual(OLD_EPISODE_KEYS_COURSE - set(ep), set(), "旧键一个不少")
        self.assertNotIn(ec.FLIGHT_PATH_KEY, ep, "分母的逐集路程只给汇总用，不落进 per_episode")


class BaseReport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_eval("base", base_trajectories())

    def test_no_block_keys_for_base(self):
        self.assertNotIn("block_survival", self.report)
        self.assertNotIn("blocks", self.report)
        for e in self.report["per_episode"]:
            for key in ("blocks_passed", "blocks_total", "block_goal_x"):
                self.assertNotIn(key, e)

    def test_collisions_per_100m_for_every_task(self):
        n = len(ramp(6.3))
        xs = ramp(6.3)
        pts_b = [(x, 3.5 if i == n - 3 else 0.0) for i, x in enumerate(xs)][:n - 2]   # 到出走廊那一步（倒数第 3 个位姿）为止
        total = 2.4 + polyline(pts_b) + 6.0
        self.assertAlmostEqual(self.report["collisions_per_100m"], 100.0 * 1 / total, places=9)

    def test_key_sets_are_the_old_ones_plus_collisions_per_100m(self):
        self.assertEqual(set(self.report), OLD_TOP_KEYS_BASE | {"collisions_per_100m"})
        self.assertEqual(set(self.report["per_episode"][0]), OLD_EPISODE_KEYS_BASE)

    def test_old_values_are_untouched(self):
        r = self.report
        self.assertEqual(r["episodes_done"], 3)
        self.assertEqual(r["requested_success_rate"], 1 / 3)
        self.assertEqual(r["collision_rate"], 1 / 3)
        self.assertEqual(r["corridor_violations"], 1)
        self.assertEqual(r["metric_version"], "sim09.v2.corridor3m")


class SharedFunctionsUnchanged(unittest.TestCase):
    def test_aggregate_and_summarize_episode_keys(self):
        """3DGS 评测（sim/3dgs/eval_3dgs.py）与 tools/compare_datasets.py 共用 ec.aggregate / ec.summarize_episode：键集合不许因本卡变。"""
        oracle, actions = full_pass()
        s = ec.summarize_episode(actions, oracle, 16.0)
        self.assertEqual(set(ec.aggregate([s], requested=1)), OLD_AGGREGATE_KEYS)
        self.assertEqual(set(s), OLD_SUMMARY_KEYS)


if __name__ == "__main__":
    unittest.main()
