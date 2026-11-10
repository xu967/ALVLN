#!/usr/bin/env python3
"""sim09.v2（FIXLIST_R2 批次 A1/A2）：route_choice 任务评测器在成功线停；中断写部分 eval.json；正常结束保留逐步日志。

不起 Gazebo：scene_gen / collect / gz 全部 mock，走完整 main()。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_eval_sim09v2.py
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import eval_closed_loop as ec
import split_manifest as sm


def pillars_for(seed):
    """每个 seed 一套几何（gid 不同，否则第二集被 dedup 跳过）；x 偏移小到不改变 A* 可行性。"""
    dx = 0.01 * (seed % 50)
    return [{"x": 1.5 + dx, "y": 0.0, "z": 1.5, "size": [0.4, 0.4, 3.0]},
            {"x": 3.0 + dx, "y": 1.6, "z": 1.5, "size": [0.4, 0.4, 3.0]}]


def base_meta(seed, waypoint=False):
    meta = {"schema": "navlab.scene.v1", "scene_id": seed, "seed": seed,
            "geometry_version": sm.SPLIT_SALT, "pillars": pillars_for(seed), "pillar_count": 2,
            "camera_profile": ec.CAMERA_PROFILE, "dynamics_profile": "x3_v1"}
    if waypoint:   # l1 口径：waypoint + stop_radius，彩色柱给 dedup_key 用
        meta["pillars"][0]["color"] = "red"
        meta.update(waypoint={"x": 5.0, "y": -2.0}, stop_radius=0.5, relation="left",
                    target_color="red", instruction="pass the red pillar on your left and stop")
    return meta


def test_seeds(n, waypoint=False):
    """前 n 个落在 test split 的 seed（split 由几何 gid 决定，不是 seed）。"""
    out = []
    for seed in range(1, 200):
        if sm.split_for_scene(base_meta(seed, waypoint)) == "test":
            out.append(seed)
            if len(out) == n:
                return out
    raise AssertionError("找不到足够的 test 几何")


class Sim09V2Tests(unittest.TestCase):
    def run_eval(self, seeds, task="base", steps=10, extra_argv=(), on_collect=None, oracle_x=(0.0, 2.0, 4.6),
                 stopped_on_goal=True, waypoint=False):
        """mock 掉 gz / scene_gen / collect，返回 (rc, report 或 None, 捕获的 collect 命令, workdirs, out)。"""
        cmds, workdirs = [], []

        def generate(cmd, **_kwargs):
            world = Path(cmd[cmd.index("--out") + 1])
            seed = int(cmd[cmd.index("--seed") + 1])
            world.write_text("<sdf/>")
            world.with_suffix(".json").write_text(json.dumps(base_meta(seed, waypoint)))
            return Mock(returncode=0)

        def collect(cmd, **_kwargs):
            episode = Path(cmd[cmd.index("--out") + 1])
            seed = int(cmd[cmd.index("--seed") + 1])
            cmds.append(cmd)
            workdirs.append(episode.parent)
            if on_collect:
                on_collect(seed)
            rows = [json.dumps({"action_executed": 1, "collision": False, "infer_ms": 2.0}) for _ in oracle_x]
            (episode / "actions.jsonl").write_text("\n".join(rows) + "\n")
            # oracle_x 的元素可以是 x（y=0）或 (x, y)：横向边界测试要构造绕开柱阵的轨迹。
            (episode / "oracle.jsonl").write_text(
                "\n".join(json.dumps({"pose_gt": {"x": p[0], "y": p[1]}} if isinstance(p, tuple)
                                      else {"pose_gt": {"x": p, "y": 0.0}})
                           for p in oracle_x) + "\n")
            (episode / "rgb_00000.png").write_bytes(b"\x89PNG fake frame")
            (episode / "episode_manifest.json").write_text(json.dumps(
                {"num_steps": len(oracle_x), "stopped_on_goal": stopped_on_goal,
                 "goal_line_x": 4.0 if stopped_on_goal else None, "dynamics_profile": "x3_v1"}))
            proc = Mock(returncode=0)
            proc.communicate.return_value = ("", "")
            return proc

        self.directory = tempfile.TemporaryDirectory()
        out = Path(self.directory.name) / "eval"
        argv = ["eval_closed_loop.py", "--policy", "forward", "--task", task, "--split", "test",
                "--seeds", ",".join(map(str, seeds)), "--steps", str(steps), "--out", str(out), *extra_argv]
        stdout = io.StringIO()
        with patch.object(ec.sys, "argv", argv), \
             patch.object(ec.subprocess, "run", side_effect=generate), \
             patch.object(ec.subprocess, "Popen", side_effect=collect), \
             patch.object(ec.cp, "_start_sim", return_value=Mock()), \
             patch.object(ec.cp, "wait_ready"), patch.object(ec.cp, "_register"), \
             patch.object(ec.cp, "stop_group"), patch.object(ec.cp, "cleanup_all"), \
             patch.object(ec.cp, "find_plugin_dir", return_value="/fake/cf_sitl/plugin"), \
             patch.object(ec.cp, "XDG_RUNTIME_DIR", str(Path(self.directory.name) / "xdg")), \
             patch.object(ec.time, "sleep"), patch.object(ec.signal, "signal"), \
             contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            rc = ec.main()
        self.last_stdout = stdout.getvalue()
        path = out / "eval.json"
        return rc, (json.loads(path.read_text()) if path.is_file() else None), cmds, workdirs, out

    def tearDown(self):
        if getattr(self, "directory", None):
            self.directory.cleanup()

    # ---- A1：成功线即停 ----
    def test_route_choice_passes_stop_at_goal_and_records_goal_fields(self):
        seed = test_seeds(1)[0]
        rc, report, cmds, _w, _out = self.run_eval([seed])
        self.assertEqual(rc, 0)
        self.assertIn("--stop-at-goal", cmds[0], "base 任务必须让 collect.py 在成功线停")
        self.assertEqual(report["metric_version"], "sim09.v2")
        self.assertEqual(report["success_criterion"], "route_choice")
        ep = report["per_episode"][0]
        self.assertTrue(ep["success"])
        self.assertTrue(ep["stopped_on_goal"])
        self.assertEqual(ep["steps_to_goal"], 3)       # oracle x: 0, 2, 4.6；goal_x = 3.0 + 1.0 = 4.0 → 第 3 步过线
        self.assertEqual(ep["metric_version"], "sim09.v2")

    def test_route_choice_without_crossing_has_no_steps_to_goal(self):
        seed = test_seeds(1)[0]
        _rc, report, _c, _w, _out = self.run_eval([seed], oracle_x=(0.0, 1.0, 2.0), stopped_on_goal=False)
        ep = report["per_episode"][0]
        self.assertFalse(ep["success"])
        self.assertFalse(ep["stopped_on_goal"])
        self.assertIsNone(ep["steps_to_goal"])

    def test_waypoint_task_keeps_stop_radius_logic(self):
        seed = test_seeds(1, waypoint=True)[0]
        rc, report, cmds, _w, _out = self.run_eval([seed], task="l1", waypoint=True, stopped_on_goal=False,
                                                   oracle_x=(0.0, 2.0, 4.6))
        self.assertEqual(rc, 0)
        self.assertNotIn("--stop-at-goal", cmds[0], "waypoint 任务按 stop_radius 的 goal-stop 结束，不能再加成功线")
        self.assertEqual(report["success_criterion"], "goal_reach_ever")
        self.assertEqual(report["metric_version"], "sim09.v2")
        self.assertFalse(report["per_episode"][0]["stopped_on_goal"])

    # ---- 2026-11-10：route_choice 的横向边界（opt-in） ----
    def test_corridor_off_by_default_keeps_legacy_numbers(self):
        """默认不开：轨迹绕到 y=-8 m 也按旧口径判成功，metric_version 逐字 sim09.v2。"""
        seed = test_seeds(1)[0]
        rc, report, _c, _w, _out = self.run_eval([seed], oracle_x=((0.0, 0.0), (2.0, -4.0), (4.6, -8.0)))
        self.assertEqual(rc, 0)
        self.assertEqual(report["metric_version"], "sim09.v2")
        self.assertIsNone(report["corridor_half_width"])
        ep = report["per_episode"][0]
        self.assertTrue(ep["success"])
        self.assertAlmostEqual(ep["y_max_m"], 8.0)
        self.assertFalse(ep["corridor_violation"])

    def test_corridor_rejects_detour_around_pillar_field(self):
        """开 3 m：同一集判失败、y_max/corridor_violation 落盘、metric_version 带后缀。"""
        seed = test_seeds(1)[0]
        rc, report, _c, _w, _out = self.run_eval([seed], extra_argv=("--corridor-half-width", "3.0"),
                                                 oracle_x=((0.0, 0.0), (2.0, -4.0), (4.6, -8.0)))
        self.assertEqual(rc, 0)
        self.assertEqual(report["metric_version"], "sim09.v2.corridor3m")
        self.assertEqual(report["corridor_half_width"], 3.0)
        ep = report["per_episode"][0]
        self.assertFalse(ep["success"], "轨迹 |y| 到 8 m，超过 3 m 边界不能算成功")
        self.assertTrue(ep["corridor_violation"])
        self.assertAlmostEqual(ep["y_max_m"], 8.0)
        self.assertEqual(report["corridor_violations"], 1)

    def test_corridor_passes_straight_through(self):
        """开 3 m：直线穿过（|y|<=0.5）仍然成功。"""
        seed = test_seeds(1)[0]
        rc, report, _c, _w, _out = self.run_eval([seed], extra_argv=("--corridor-half-width", "3.0"),
                                                 oracle_x=((0.0, 0.0), (2.0, 0.4), (4.6, -0.5)))
        self.assertEqual(rc, 0)
        ep = report["per_episode"][0]
        self.assertTrue(ep["success"])
        self.assertFalse(ep["corridor_violation"])
        self.assertAlmostEqual(ep["y_max_m"], 0.5)

    # ---- A7：l_i 净空 = 机体半径（物理可行下界），不是教师规划余量 ----
    def test_spl_reference_uses_body_radius(self):
        body = ec.teacher.BODY_HALF_BY_VEHICLE
        self.assertEqual(ec.spl_clearance("e1_mini", "base"), body["e1_mini"])        # 0.102，不是 0.40
        self.assertEqual(ec.spl_clearance("e1_mini", "forest_e1"), body["e1_mini"])   # 不随任务变
        self.assertEqual(ec.spl_clearance("x3", "base"), body["x3"])                  # 0.235，不是 0.5
        self.assertAlmostEqual(ec.spl_clearance("e1_mini", "base"), 0.102)
        seed = test_seeds(1)[0]
        rc, report, _c, _w, _out = self.run_eval([seed], extra_argv=("--vehicle", "e1_mini", "--dynamics", "cf_sitl"))
        self.assertEqual(rc, 0)
        self.assertEqual(report["metric_version"], "sim09.v2.coast1s")   # cf_sitl 的后缀照旧
        ref = report["spl_reference"]
        self.assertAlmostEqual(ref["clearance_m"], 0.102)
        self.assertIn("body", ref["clearance_basis"].lower())
        self.assertAlmostEqual(ref["teacher_clearance_m"], 0.40)   # 教师规划余量另记，读表的人能看出两者不同
        ep = report["per_episode"][0]
        # 直线 y=0 被第一根柱挡住：绕柱的 l_i 随膨胀余量单调增，0.102 下比 0.40 的教师参考路径短。
        pillars = pillars_for(seed)
        goal_x = max(p["x"] for p in pillars) + ec.GOAL_MARGIN_M
        l_body = ec.teacher_shortest_path_m(pillars, (0.0, 0.0), goal_x=goal_x, clearance=0.102)
        l_teacher = ec.teacher_shortest_path_m(pillars, (0.0, 0.0), goal_x=goal_x, clearance=0.40)
        self.assertAlmostEqual(ep["shortest_path_m"], l_body)
        self.assertLess(ep["shortest_path_m"], l_teacher)

    def test_goal_margin_must_match_collect_goal_line(self):
        seed = test_seeds(1)[0]
        with self.assertRaises(SystemExit) as raised:
            self.run_eval([seed], extra_argv=("--goal-margin", "2.0"))
        self.assertNotEqual(raised.exception.code, 0)

    # ---- A2：中断写部分 eval.json；正常结束保留逐步日志 ----
    def test_sigterm_mid_run_writes_partial_eval_json_and_cleans_workdir(self):
        seeds = test_seeds(3)

        def interrupt_second(seed):
            if seed == seeds[1]:
                raise SystemExit(143)       # main() 装的 SIGTERM handler 就是这样抛的

        rc, report, _cmds, workdirs, out = self.run_eval(seeds, on_collect=interrupt_second)
        self.assertEqual(rc, 143)
        self.assertIsNotNone(report, "中断后也要写出 eval.json")
        self.assertTrue(report["partial"])
        self.assertEqual(report["episodes_done"], 1)
        self.assertEqual([e["seed"] for e in report["per_episode"]], seeds[:1])
        self.assertEqual(report["interrupted"], {"seed": seeds[1], "reason": "SIGTERM"})
        self.assertIn(seeds[1], [f["seed"] for f in report["failed"]])
        self.assertFalse(workdirs[0].exists(), "临时工作目录中断后也要删")
        self.assertTrue((out / "episodes" / f"seed_{seeds[0]}" / "oracle.jsonl").is_file(),
                        "已完成集的逐步日志要留下来")

    def test_keyboard_interrupt_writes_partial_eval_json(self):
        seeds = test_seeds(2)

        def ctrl_c(seed):
            if seed == seeds[0]:
                raise KeyboardInterrupt

        rc, report, _cmds, _w, _out = self.run_eval(seeds, on_collect=ctrl_c)
        self.assertEqual(rc, 130)
        self.assertTrue(report["partial"])
        self.assertEqual(report["episodes_done"], 0)
        self.assertEqual(report["interrupted"]["reason"], "KeyboardInterrupt")

    def test_normal_end_keeps_step_logs_drops_frames_and_workdir(self):
        seeds = test_seeds(2)
        rc, report, _cmds, workdirs, out = self.run_eval(seeds)
        self.assertEqual(rc, 0)
        self.assertFalse(report.get("partial"))
        self.assertFalse(workdirs[0].exists(), "正常结束删临时工作目录")
        for seed in seeds:
            ep = out / "episodes" / f"seed_{seed}"
            for name in ("actions.jsonl", "oracle.jsonl", "episode_manifest.json"):
                self.assertTrue((ep / name).is_file(), f"{name} 要保留")
            self.assertEqual(report["per_episode"][seeds.index(seed)]["episode_dir"], f"episodes/seed_{seed}")
        self.assertEqual(list(out.rglob("*.png")), [], "帧不保留")


if __name__ == "__main__":
    unittest.main()
