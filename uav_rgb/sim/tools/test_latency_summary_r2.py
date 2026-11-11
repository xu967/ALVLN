#!/usr/bin/env python3
"""experiments/2026-11-10-action-latency/summarize_r2.py 的单测（合成数据，不起 gz）。

背景：第一轮标定门没过（L=0.16 与 0.20），第二轮要按「标定阶梯 → 学生验收 → 森林」自动走，阶梯每一步靠 summarize_r2.py 的门判定，
最后 summary.md 要能独立读懂（主会话拿它填计划）。这些是新的判定与汇总逻辑，数据全是合成的，格式照真实 eval.json / episodes/seed_N。

覆盖：
  Gate          逐条款判定（迟到步、生效时长、转角、位移）、两种位移口径在「直线更长」的数据上给出不同结论、数据不全、行里没有 late_dispatch、
                退出码 0 / 1 / 2 / 3、组 B 复用组 A 的 K=1；
  Ladder        阶梯规则：A 过 / A 迟到→B / B 迟到→C / 只有位移不过就停 / 三组都没过；
  Retry         含迟到步的集重跑后采用重跑结果、重跑仍迟到按没成功、没有重跑记录、首次记录口径；
  Acceptance    验收 4 的三档判定（≤0.08 / 之间 / ≥0.12）与 b、c 计数；验收 2 的逐集一致率与预测；
  CgroupAndK3   cgroup 记录的平均核数与限流增量、⚠ 标记；K=3 逐块存活 S_k 与掉高集数；
  Summary       main 写出的 summary.md 含结论一览与各节；fwd_samples 与 analyze_round1 的同名函数逐位一致。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_latency_summary_r2.py
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
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# LATENCY_EXP_DIR：指向一份被故意改坏的脚本副本，用来确认这些测试真的会红（变异检查）；缺省是仓库里的脚本
EXP = os.path.abspath(os.environ.get("LATENCY_EXP_DIR") or os.path.join(HERE, "..", "..", "experiments", "2026-11-10-action-latency"))
sys.path.insert(0, EXP)


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXP, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


r2 = load("latency_summarize_r2", "summarize_r2.py")
s1 = r2.s1
an = load("latency_analyze_round1_for_r2", "analyze_round1.py")

F, Y = 1, 2          # FORWARD_SLOW / YAW_LEFT
STRATA = {1: 0.08, 2: 0.40, 3: 0.28}   # 连续前进第 1、2、3 步的位移（m）；第 4 步起用各测试给的 disp。r=2、3 的步比巡航大，与第一轮归因里的现象同向
HOLD_NS = 1_005_000_000
FROZEN = r2.FROZEN_SEEDS


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def cruise_steps(n=40, disp=0.23, lag=0.20, yaw_deg=26.0):
    """一集的逐步清单 [(动作, 位移, lag, dyaw)]：每 10 步里 [Y, F×9]（r=1..9）；r=1..3 的位移取 STRATA，r≥4 为 disp。"""
    out = []
    for i in range(n):
        j = i % 10
        if j == 0:
            out.append((Y, 0.0, lag, yaw_deg))
        else:
            out.append((F, STRATA.get(j, disp), lag, 0.1))
    return out


def long_run_steps(n=40, disp=0.23, lag=0.20, yaw_deg=26.0):
    """直线更长的集（K=3 的样子）：每 20 步里 [Y, F×19]；r 构成里巡航占比大，层内位移与 cruise_steps 相同。"""
    out = []
    for i in range(n):
        j = i % 20
        if j == 0:
            out.append((Y, 0.0, lag, yaw_deg))
        else:
            out.append((F, STRATA.get(j, disp), lag, 0.1))
    return out


def write_arm(root, name, eps, L=0.20, k=1, requested=None, blocks=None, corridor=3.0, rtf=0.8, with_flag=True):
    """一个臂目录。eps = {seed: dict(steps=[(动作, 位移, lag, dyaw)], success=bool, passes=[bool×3]（K=3）, z_tail=高度)}。
    lag > L + 0.02 的步记迟到；含迟到步的集进 failed[] 与 late_dispatch_episodes（与真实 eval 一致，迟到集不进 per_episode）。"""
    arm = os.path.join(root, name)
    per_episode, failed, late_eps = [], [], []
    for seed, spec in eps.items():
        ep = os.path.join(arm, "episodes", f"seed_{seed}")
        os.makedirs(ep, exist_ok=True)
        steps = spec["steps"]
        n = len(steps)
        passes = spec.get("passes")
        x_end = 6.5 if k == 1 and spec.get("success", True) else (2.0 if k == 1 else [0.5, 6.5, 11.5, 16.5][sum(passes or [])])
        z_tail = spec.get("z_tail", 2.0)
        actions, oracle, late_ids = [], [], []
        for i, (a, d, lag, dyaw) in enumerate(steps):
            late = lag > L + 0.02 + 1e-9
            if late:
                late_ids.append(i)
            start = 10_000_000_000 + i * 1_000_000_000
            row = {"step": i, "action_executed": a, "collision": False, "displacement_m": d, "dyaw_deg": dyaw if a == Y else 0.1,
                   "dispatch_lag_s": lag, "src": None, "action_start_ns": start}
            if with_flag:
                row["late_dispatch"] = late
            actions.append(row)
            oracle.append({"step": i, "action_start_ns": start, "action_end_ns": start + HOLD_NS,
                           "pose_gt": {"x": x_end * (i + 1) / n, "y": 0.0, "z": z_tail if i >= n - 10 else 2.0}})
        write_jsonl(os.path.join(ep, "actions.jsonl"), actions)
        write_jsonl(os.path.join(ep, "oracle.jsonl"), oracle)
        sim_s = sum((o["action_end_ns"] - o["action_start_ns"]) / 1e9 for o in oracle) - sum(s[2] for s in steps)
        with open(os.path.join(ep, "episode_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"performance": {"steps": n, "action_sim_s": sim_s, "action_rtf": rtf}, "action_latency_s": L}, f)
        lags = [s[2] for s in steps]
        entry = {"seed": seed, "success": bool(spec.get("success", False if k == 3 else True)), "steps": n, "pillars": 16,
                 "episode_dir": os.path.join("episodes", f"seed_{seed}"), "dispatch_lag_s_max": max(lags)}
        if k == 3:
            entry["block_goal_x"] = [6.0, 11.0, 16.0]
        if late_ids:
            failed.append({"seed": seed, "rc": 0, "error": "late_dispatch: stub", "late_dispatch_steps": len(late_ids), "late_step_ids": late_ids})
            late_eps.append({"seed": seed, "late_dispatch_steps": len(late_ids), "late_step_ids": late_ids,
                             "dispatch_lag_s_median": sorted(lags)[len(lags) // 2], "dispatch_lag_s_max": max(lags)})
        else:
            per_episode.append(entry)
    report = {"episodes_requested": requested or len(eps), "episodes_done": len(per_episode), "failed_count": len(failed), "failed": failed,
              "partial": False, "per_episode": per_episode, "corridor_half_width": corridor, "blocks": blocks, "action_latency_s": L,
              "late_dispatch_episodes": late_eps}
    with open(os.path.join(arm, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(report, f)
    return arm


def cal_eps(maker, n_eps=10, **kw):
    return {s: {"steps": maker(**kw), "success": True} for s in range(2, 2 + n_eps)}


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        for patcher in (mock.patch.object(r2, "OUT", self.root),
                        mock.patch.object(s1, "REF_K3", [os.path.join(self.root, "no-ref")] * 3)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def group(self, gid="A"):
        return r2.GROUP_BY_ID[gid]

    def build_group(self, gid, k1, k3, L=None, **kw):
        g = self.group(gid)
        L = g["L"] if L is None else L
        if not os.path.isdir(os.path.join(self.root, g["k1"])) and k1 is not None:
            write_arm(self.root, g["k1"], k1, L=L, k=1, **kw)
        if k3 is not None:
            write_arm(self.root, g["k3"], k3, L=L, k=3, blocks=3, **kw)


class Gate(Base):
    def test_pass_when_every_clause_holds(self):
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps))
        res = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual(res["status"], "PASS", res["text"])
        self.assertEqual(res["reasons"], [])
        self.assertTrue(all(r["ok"] for r in res["rows"] if r["ok"] is not None))

    def test_late_step_in_k3_fails_with_late_reason(self):
        k3 = cal_eps(cruise_steps)
        k3[2]["steps"] = [(a, d, 0.26 if i == 0 else lag, y) for i, (a, d, lag, y) in enumerate(k3[2]["steps"])]   # 第 0 步 lag 0.26 > 0.22
        self.build_group("A", cal_eps(cruise_steps), k3)
        res = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual(res["status"], "FAIL_LATE", res["text"])
        self.assertIn("late", res["reasons"])
        self.assertIn("迟到步 K=1 0/", res["text"])
        self.assertFalse(res["rows"][0]["ok"])

    def test_late_step_in_k1_also_counts(self):
        k1 = cal_eps(cruise_steps)
        k1[3]["steps"] = [(a, d, 0.25 if i == 5 else lag, y) for i, (a, d, lag, y) in enumerate(k1[3]["steps"])]
        self.build_group("A", k1, cal_eps(cruise_steps))
        self.assertEqual(r2.gate_group(self.group("A"), "cruise")["status"], "FAIL_LATE")

    def test_displacement_only_failure_has_no_late_reason(self):
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps, disp=0.23 * 0.95))   # K=3 巡航位移低 5%
        res = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual(res["status"], "FAIL_NOLATE", res["text"])
        self.assertEqual(res["reasons"], ["disp"])

    def test_yaw_clause(self):
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps, yaw_deg=26.0 * 1.05))
        res = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual((res["status"], res["reasons"]), ("FAIL_NOLATE", ["yaw"]))

    def test_effective_duration_clause(self):
        """K=3 每步 lag 0.219（≤ 阈值 0.22，不迟到）→ 生效时长 0.786 对 0.805，低 2.4% > 2%。"""
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps, lag=0.219))
        res = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual((res["status"], res["reasons"]), ("FAIL_NOLATE", ["eff"]), res["text"])

    def test_clause_modes_disagree_on_composition_confounded_data(self):
        """K=1 的直线短（每 10 步一次转向）、K=3 的直线长（每 20 步一次）；层内位移完全相同。
        上一步也是前进的口径被构成拉低（steady 未过），连续前进第 4 步起的口径不受影响（cruise 过）。"""
        self.build_group("A", cal_eps(cruise_steps), cal_eps(long_run_steps))
        steady = r2.gate_group(self.group("A"), "steady")
        cruise = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual((steady["status"], steady["reasons"]), ("FAIL_NOLATE", ["disp"]), steady["text"])
        self.assertEqual(cruise["status"], "PASS", cruise["text"])
        # 不进门的行：cruise 口径时另一种口径与全部步口径作为对照列出
        names = [r["name"] for r in cruise["rows"] if r["ok"] is None]
        self.assertEqual(len(names), 3)   # 不进门的三行：上一步前进（步 3–30）、上一步前进（全部步）、巡航（全部步）
        self.assertTrue(any("第一轮的门口径" in n for n in names))

    def test_incomplete_when_an_arm_is_missing(self):
        self.build_group("A", cal_eps(cruise_steps), None)
        res = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual(res["status"], "INCOMPLETE")
        self.assertIn("cal-L0.20-k3", res["text"])

    def test_missing_late_flag_is_a_data_error_not_a_late_failure(self):
        """行里没有 late_dispatch（没带 --action-latency）：不能当迟到去试下一组，按「原因里没有迟到步」停。"""
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps), with_flag=False)
        res = r2.gate_group(self.group("A"), "cruise")
        self.assertEqual(res["status"], "FAIL_NOLATE", res["text"])
        self.assertIn("nolate_flag", res["reasons"])

    def test_group_b_reuses_group_a_k1(self):
        """B 组的 K=1 是 A 组的 cal-L0.20-k1：只补跑 cal-L0.20rtf0.5-k3 就能判。"""
        self.build_group("A", cal_eps(cruise_steps), None)
        write_arm(self.root, "cal-L0.20rtf0.5-k3", cal_eps(cruise_steps), L=0.20, k=3, blocks=3)
        res = r2.gate_group(self.group("B"), "cruise")
        self.assertEqual(res["status"], "PASS", res["text"])

    def test_cli_exit_codes_and_one_line(self):
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps))
        for gid, want in (("A", 0), ("B", 2), ("C", 2)):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = r2.cli(["--gate", gid, self.root, "--disp-clause", "cruise"])
            self.assertEqual(rc, want)
            self.assertEqual(len(buf.getvalue().strip().splitlines()), 1)
        late = cal_eps(cruise_steps)
        late[2]["steps"][0] = (F, 0.1, 0.30, 0.1)
        write_arm(self.root, "cal-L0.20-k3", late, L=0.20, k=3, blocks=3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(r2.cli(["--gate", "A", self.root, "--disp-clause", "cruise"]), 1)
        write_arm(self.root, "cal-L0.20-k3", cal_eps(cruise_steps, disp=0.20), L=0.20, k=3, blocks=3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(r2.cli(["--gate", "A", self.root, "--disp-clause", "cruise"]), 3)

    def test_disp_clause_from_config_env(self):
        with open(os.path.join(self.root, "config.env"), "w", encoding="utf-8") as f:
            f.write("# 起跑配置\nDISP_CLAUSE=steady\nL_STAR=0.20\n")
        self.assertEqual(r2.disp_clause_of(), "steady")
        self.assertEqual(r2.disp_clause_of("cruise"), "cruise")        # 命令行优先
        os.remove(os.path.join(self.root, "config.env"))
        self.assertEqual(r2.disp_clause_of(), r2.DEFAULT_DISP_CLAUSE)
        with self.assertRaises(ValueError):
            r2.disp_clause_of("bogus")


class Ladder(Base):
    def late_k3(self):
        k3 = cal_eps(cruise_steps)
        k3[2]["steps"][0] = (F, 0.1, 0.30, 0.1)
        return k3

    def test_a_passes_selects_a_and_does_not_look_further(self):
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps))
        results, selected, state = r2.ladder_state("cruise")
        self.assertEqual(selected, "A")
        self.assertEqual(list(results), ["A"])
        self.assertIn("A 组通过", state)

    def test_a_late_then_b_passes(self):
        self.build_group("A", cal_eps(cruise_steps), self.late_k3())
        write_arm(self.root, "cal-L0.20rtf0.5-k3", cal_eps(cruise_steps), L=0.20, k=3, blocks=3)
        results, selected, _state = r2.ladder_state("cruise")
        self.assertEqual((results["A"]["status"], results["B"]["status"], selected), ("FAIL_LATE", "PASS", "B"))
        self.assertNotIn("C", results)

    def test_a_and_b_late_then_c_passes(self):
        self.build_group("A", cal_eps(cruise_steps), self.late_k3())
        write_arm(self.root, "cal-L0.20rtf0.5-k3", self.late_k3(), L=0.20, k=3, blocks=3)
        write_arm(self.root, "cal-L0.24-k1", cal_eps(cruise_steps, lag=0.24), L=0.24, k=1)
        write_arm(self.root, "cal-L0.24-k3", cal_eps(cruise_steps, lag=0.24), L=0.24, k=3, blocks=3)
        _results, selected, _state = r2.ladder_state("cruise")
        self.assertEqual(selected, "C")

    def test_displacement_only_failure_stops_the_ladder(self):
        """A 组只因位移未过（没有迟到步）：不往下试，状态写明原因。"""
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps, disp=0.20))
        results, selected, state = r2.ladder_state("cruise")
        self.assertIsNone(selected)
        self.assertEqual(list(results), ["A"])
        self.assertIn("没有迟到步", state)
        self.assertIn("disp", state)

    def test_all_three_fail_stops(self):
        self.build_group("A", cal_eps(cruise_steps), self.late_k3())
        write_arm(self.root, "cal-L0.20rtf0.5-k3", self.late_k3(), L=0.20, k=3, blocks=3)
        write_arm(self.root, "cal-L0.24-k1", cal_eps(cruise_steps, lag=0.24), L=0.24, k=1)
        late24 = cal_eps(cruise_steps, lag=0.24)
        late24[2]["steps"][0] = (F, 0.1, 0.40, 0.1)
        write_arm(self.root, "cal-L0.24-k3", late24, L=0.24, k=3, blocks=3)
        results, selected, state = r2.ladder_state("cruise")
        self.assertIsNone(selected)
        self.assertEqual(set(results), {"A", "B", "C"})
        self.assertIn("三组都没过", state)

    def test_in_progress(self):
        self.build_group("A", cal_eps(cruise_steps), self.late_k3())
        _results, selected, state = r2.ladder_state("cruise")
        self.assertIsNone(selected)
        self.assertIn("B 组还没跑", state)


def student_eps(n_ok, pass3=None, late=(), k3=False):
    """20 个冻结 seed 的学生臂：前 n_ok 个成功（K=1）/ 通过第 1 块（K=3，passes=[T,F,F]）；late 里的 seed 含一个迟到步。"""
    eps = {}
    for i, seed in enumerate(FROZEN):
        ok = i < n_ok
        steps = cruise_steps(n=30)
        if seed in late:
            steps[0] = (F, 0.1, 0.30, 0.1)
        spec = {"steps": steps, "success": ok}
        if k3:
            spec["passes"] = [ok, False, False]
            spec["success"] = False
        eps[seed] = spec
    return eps


class Retry(Base):
    NAME = "acc-L0.20-k1-s1"

    def build(self, retry_late=False, retry_missing_seed=False, retry_success=True):
        write_arm(self.root, self.NAME, student_eps(15, late=(172,)), L=0.20, k=1, requested=20)
        if retry_missing_seed:
            return
        steps = cruise_steps(n=30)
        if retry_late:
            steps[3] = (F, 0.1, 0.30, 0.1)
        write_arm(self.root, self.NAME + "-retry1", {172: {"steps": steps, "success": retry_success}}, L=0.20, k=1, requested=1)

    def test_clean_retry_replaces_the_late_first_run(self):
        self.build()
        m = r2.merged_rows(os.path.join(self.root, self.NAME), 20)
        it = m[172]
        self.assertEqual((it["first_late"], it["retry"], it["final"]), (True, "ok", "retry"))
        ok = r2.success_map(m, 1)
        self.assertTrue(ok[172])
        self.assertEqual(sum(ok.values()), 15)           # 前 15 个 seed 成功；seed 172（第 5 个）首次迟到不算，重跑成功补回，仍是 15
        first = r2.first_run_map(m, 1)
        self.assertFalse(first[172])                     # 首次记录：迟到集按没成功

    def test_retry_that_is_late_again_counts_as_failure(self):
        self.build(retry_late=True)
        m = r2.merged_rows(os.path.join(self.root, self.NAME), 20)
        self.assertEqual((m[172]["retry"], m[172]["final"]), ("late", "none"))
        self.assertFalse(r2.success_map(m, 1)[172])

    def test_no_retry_record_counts_as_failure(self):
        self.build(retry_missing_seed=True)
        m = r2.merged_rows(os.path.join(self.root, self.NAME), 20)
        self.assertEqual((m[172]["retry"], m[172]["final"]), ("missing", "none"))
        self.assertFalse(r2.success_map(m, 1)[172])

    def test_clean_retry_that_fails_the_task_is_a_real_failure(self):
        self.build(retry_success=False)
        m = r2.merged_rows(os.path.join(self.root, self.NAME), 20)
        self.assertEqual(m[172]["final"], "retry")
        self.assertFalse(r2.success_map(m, 1)[172])

    def test_episodes_without_late_steps_are_untouched(self):
        self.build()
        m = r2.merged_rows(os.path.join(self.root, self.NAME), 20)
        self.assertTrue(all(it["final"] == "first" and it["retry"] is None for s, it in m.items() if s != 172))

    def test_late_seeds_cli(self):
        self.build()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r2.cli(["--late-seeds", os.path.join(self.root, self.NAME)])
        self.assertEqual(buf.getvalue().strip(), "172")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r2.cli(["--late-seeds", os.path.join(self.root, "nope")])
        self.assertEqual(buf.getvalue().strip(), "")

    def test_retry_section_lists_the_episode(self):
        self.build()
        text = "\n".join(r2.retry_section(0.20))
        self.assertIn("重跑后干净，采用重跑结果", text)
        self.assertIn("| acc-L0.20-k1-s1 | 172 |", text)


class Acceptance(Base):
    def build_student(self, n_k1, n_k3):
        """三个训练种子，每个 20 个冻结 seed：K=1 成功 n_k1 个、K=3 第 1 块通过 n_k3 个（每个种子都是前 n 个）。"""
        for s in r2.SEEDS3:
            write_arm(self.root, f"acc-L0.20-k1-s{s}", student_eps(n_k1), L=0.20, k=1, requested=20)
            write_arm(self.root, f"acc-L0.20-k3-s{s}", student_eps(n_k3, k3=True), L=0.20, k=3, blocks=3, requested=20)

    def verdict(self):
        _lines, verdict, _merged = r2.acceptance4_section(0.20)
        return verdict

    def test_small_gap_means_first_step_timing(self):
        self.build_student(17, 16)     # 60 集：51 对 48，差 0.05
        kind, gap, text, _counts = self.verdict()
        self.assertEqual(kind, "small")
        self.assertAlmostEqual(gap, 0.05, places=9)
        self.assertIn("来自第一步的时序", text)

    def test_gap_between_the_two_lines_is_not_attributed(self):
        self.build_student(17, 15)     # 差 2/20 = 0.10
        kind, gap, text, _counts = self.verdict()
        self.assertEqual(kind, "between")
        self.assertIn("不归因", text)

    def test_large_gap_means_later_blocks(self):
        self.build_student(17, 14)     # 差 3/20 = 0.15
        kind, gap, text, _counts = self.verdict()
        self.assertEqual(kind, "large")
        self.assertIn("后面几块的柱子进了画面", text)

    def test_boundaries(self):
        self.build_student(20, 19)     # 差 0.05
        self.assertEqual(self.verdict()[0], "small")
        self.build_student(20, 12)     # 差 0.4
        self.assertEqual(self.verdict()[0], "large")

    def test_b_and_c_counts_and_mcnemar(self):
        self.build_student(17, 14)
        lines, _v, _m = r2.acceptance4_section(0.20)
        text = "\n".join(lines)
        # 前 14 个两边都过，第 15–17 个只 K=1 过 → 每个种子 b=3，共 9；c=0；p = 2·(1/2)^9 = 0.004
        self.assertIn("配对 b（只 K=1 过）=9，c（只 K=3 过）=0，精确 McNemar p = 0.004", text)
        self.assertIn("K=1 成功 51/60", text)
        self.assertIn("K=3 第 1 块通过 42/60", text)

    def test_late_episode_pairs_use_the_retry_result(self):
        """K=3 的 seed 1（前 15 个里）含迟到步，重跑通过：按重跑结果配对；首次记录口径另列。"""
        self.build_student(17, 15)
        write_arm(self.root, "acc-L0.20-k3-s0", student_eps(15, k3=True, late=(1,)), L=0.20, k=3, blocks=3, requested=20)
        steps = cruise_steps(n=30)
        write_arm(self.root, "acc-L0.20-k3-s0-retry1", {1: {"steps": steps, "success": False, "passes": [True, False, False]}}, L=0.20, k=3, blocks=3, requested=1)
        lines, _v, _m = r2.acceptance4_section(0.20)
        text = "\n".join(lines)
        self.assertIn("K=3 第 1 块通过 45/60", text)       # 重跑后 45
        self.assertIn("只看首次记录", text)
        self.assertIn("K=3 第 1 块 44/60", text)           # 首次记录：迟到集按没通过 → 44

    def test_acceptance2_agreement_and_prediction(self):
        ref_a = write_arm(self.root, "ref-afternoon", {s: {"steps": cruise_steps(n=30), "success": i < 17} for i, s in enumerate(FROZEN)}, k=1)
        ref_e = write_arm(self.root, "ref-evening", {s: {"steps": cruise_steps(n=30), "success": i < 17} for i, s in enumerate(FROZEN)}, k=1)
        write_arm(self.root, "acc-L0.20-k1-s1", student_eps(17), L=0.20, k=1, requested=20)
        with mock.patch.object(s1, "REF_K1_AFTERNOON_S1", ref_a), mock.patch.object(s1, "REF_K1_EVENING_S1", ref_e):
            lines, verdict = r2.acceptance2_section(0.20)
        self.assertEqual(verdict[0], True)
        self.assertEqual(verdict[1], [1.0, 1.0])
        self.assertIn("逐集一致率 ≥ 70%）：**成立**", "\n".join(lines))

    def test_acceptance2_prediction_fails_below_seventy_percent(self):
        flipped = {s: {"steps": cruise_steps(n=30), "success": i >= 6} for i, s in enumerate(FROZEN)}    # 与新协议的前 17 个成功几乎相反
        ref_a = write_arm(self.root, "ref-afternoon", flipped, k=1)
        ref_e = write_arm(self.root, "ref-evening", flipped, k=1)
        write_arm(self.root, "acc-L0.20-k1-s1", student_eps(17), L=0.20, k=1, requested=20)
        with mock.patch.object(s1, "REF_K1_AFTERNOON_S1", ref_a), mock.patch.object(s1, "REF_K1_EVENING_S1", ref_e):
            _lines, verdict = r2.acceptance2_section(0.20)
        self.assertFalse(verdict[0])
        self.assertLess(verdict[1][0], 0.70)

    def test_sections_without_selected_l_say_so(self):
        self.assertIn("没有选定的 L", "\n".join(r2.acceptance2_section(None)[0]))
        self.assertIn("没有选定的 L", "\n".join(r2.acceptance4_section(None)[0]))


class CgroupAndK3(Base):
    def write_cg(self, name, start, end):
        with open(os.path.join(self.root, name + ".cg"), "w", encoding="utf-8") as f:
            f.write("start " + " ".join(str(x) for x in start) + "\nend " + " ".join(str(x) for x in end) + "\n")

    def test_read_cg_average_cores_and_throttle(self):
        self.write_cg("armA", (1000.0, 5_000_000, 10, 2_000_000), (1100.0, 605_000_000, 14, 5_000_000))
        cg = r2.read_cg("armA")
        self.assertAlmostEqual(cg["avg_cores"], 6.0, places=9)        # 600 s CPU / 100 s
        self.assertEqual(cg["throttled"], 4)
        self.assertAlmostEqual(cg["throttled_s"], 3.0, places=9)
        self.assertAlmostEqual(cg["wall_s"], 100.0, places=9)

    def test_read_cg_unfinished_or_absent_is_none(self):
        with open(os.path.join(self.root, "armB.cg"), "w", encoding="utf-8") as f:
            f.write("start 1000.0 5 10 0\n")
        self.assertIsNone(r2.read_cg("armB"))
        self.assertIsNone(r2.read_cg("armC"))

    def test_timing_row_flags_throttled_arms_without_voiding_them(self):
        write_arm(self.root, "acc-L0.20-k1-s1", student_eps(15), L=0.20, k=1, requested=20)
        self.write_cg("acc-L0.20-k1-s1", (0.0, 0, 5, 0), (100.0, 800_000_000, 9, 1_000_000))
        row = r2.timing_row("acc-L0.20-k1-s1", os.path.join(self.root, "acc-L0.20-k1-s1"), 20)
        self.assertIn("| 8.0 | 4 ⚠ |", row)
        self.write_cg("acc-L0.20-k1-s1", (0.0, 0, 5, 0), (100.0, 800_000_000, 5, 0))
        self.assertIn("| 8.0 | 0 |", r2.timing_row("acc-L0.20-k1-s1", os.path.join(self.root, "acc-L0.20-k1-s1"), 20))

    def test_k3_block_survival_and_altitude_drop(self):
        eps = {}
        for seed, passes, z in ((1, [True, True, True], 2.0), (2, [True, True, False], 2.0), (3, [True, False, False], 1.2),
                                (4, [False, False, False], 1.0)):
            eps[seed] = {"steps": cruise_steps(n=30), "passes": passes, "z_tail": z}
        write_arm(self.root, "acc-L0.20-k3-s0", eps, L=0.20, k=3, blocks=3, requested=4)
        m = r2.merged_rows(os.path.join(self.root, "acc-L0.20-k3-s0"), 4)
        st = r2.k3_block_stats(m)
        self.assertEqual(st["S"], [3, 2, 1])
        self.assertEqual(st["drops"], 2)                 # z_tail 1.2 与 1.0 < 1.5
        self.assertEqual(st["n"], 4)
        self.assertEqual(len(st["steps"]), 4)

    def test_k3_late_episode_without_clean_result_is_not_counted_as_passed(self):
        """含迟到步的集（基础设施失败）轨迹上哪怕过了三块，没有干净的重跑结果就不计入 S_k；有干净重跑才按重跑算。"""
        late_steps = cruise_steps(n=30)
        late_steps[0] = (F, 0.1, 0.30, 0.1)
        eps = {1: {"steps": late_steps, "passes": [True, True, True]}, 2: {"steps": cruise_steps(n=30), "passes": [True, True, False]}}
        write_arm(self.root, "acc-L0.20-k3-s0", eps, L=0.20, k=3, blocks=3, requested=2)
        m = r2.merged_rows(os.path.join(self.root, "acc-L0.20-k3-s0"), 2)
        self.assertEqual(r2.k3_block_stats(m)["S"], [1, 1, 0])               # 只有 seed 2 计入
        write_arm(self.root, "acc-L0.20-k3-s0-retry1", {1: {"steps": cruise_steps(n=30), "passes": [True, False, False]}}, L=0.20, k=3, blocks=3, requested=1)
        m = r2.merged_rows(os.path.join(self.root, "acc-L0.20-k3-s0"), 2)
        self.assertEqual(r2.k3_block_stats(m)["S"], [2, 1, 0])               # seed 1 按重跑结果（只过第 1 块）

    def test_k3_section_has_old_protocol_side_by_side(self):
        eps = {seed: {"steps": cruise_steps(n=30), "passes": [True, False, False]} for seed in FROZEN[:4]}
        old = [write_arm(self.root, f"old-k3-s{s}", eps, L=0.20, k=3, blocks=3) for s in r2.SEEDS3]
        for s in r2.SEEDS3:
            write_arm(self.root, f"acc-L0.20-k3-s{s}", eps, L=0.20, k=3, blocks=3, requested=4)
        with mock.patch.object(s1, "REF_K3", old):
            text = "\n".join(r2.k3_section(0.20, {s: r2.merged_rows(os.path.join(self.root, f"acc-L0.20-k3-s{s}"), 4) for s in r2.SEEDS3}))
        self.assertIn("新协议 K=3 合计", text)
        self.assertIn("旧协议 K=3（11-10 晚长课程第二轮，s0..2）", text)
        self.assertIn("12/12 | 0/12 | 0/12", text)       # 合计 S_1/S_2/S_3
        self.assertIn("比旧协议的 0.871 s 短 8%", text)


class Summary(Base):
    def build_all(self):
        self.ref_a = write_arm(self.root, "ref-afternoon", {s: {"steps": cruise_steps(n=30), "success": i < 17} for i, s in enumerate(FROZEN)}, k=1)
        self.ref_e = write_arm(self.root, "ref-evening", {s: {"steps": cruise_steps(n=30), "success": i < 16} for i, s in enumerate(FROZEN)}, k=1)
        self.build_group("A", cal_eps(cruise_steps), cal_eps(cruise_steps))
        for s in r2.SEEDS3:
            write_arm(self.root, f"acc-L0.20-k1-s{s}", student_eps(17), L=0.20, k=1, requested=20)
            write_arm(self.root, f"acc-L0.20-k3-s{s}", student_eps(14, k3=True), L=0.20, k=3, blocks=3, requested=20)
        for d in ("sparse", "medium", "dense"):
            write_arm(self.root, f"forest-L0.20-{d}", {4100: {"steps": cruise_steps(n=20)}, 4101: {"steps": cruise_steps(n=20)}}, L=0.20, k=1, requested=3)
        with open(os.path.join(self.root, "L_STAR"), "w", encoding="utf-8") as f:
            f.write("0.20\n")
        with open(os.path.join(self.root, "start.txt"), "w", encoding="utf-8") as f:
            f.write("gpu_ok=是\ncpu_quota_cores=12.0\nmin_cores=12\nlock_wait_s=0\ngz_count=0\nidle_window_s=30\nidle_cores=0.4\nidle_max_cores=1.5\nidle_wait_s=0\n"
                    "started_at=2026-11-11 21:00:00\ngpu=GPU-test\nlanes=2\n")

    def test_main_writes_a_readable_summary(self):
        self.build_all()
        with open(os.path.join(self.root, "DONE"), "w", encoding="utf-8") as f:
            f.write("rc=0 2026-11-11 23:59:59 测试\n")
        with mock.patch.object(s1, "REF_K1_AFTERNOON_S1", self.ref_a), mock.patch.object(s1, "REF_K1_EVENING_S1", self.ref_e), \
                contextlib.redirect_stdout(io.StringIO()):
            r2.main(note="测试用合成数据")
        with open(os.path.join(self.root, "summary.md"), encoding="utf-8") as f:
            text = f.read()
        for needle in ("> 测试用合成数据", "## 0. 结论一览", "DONE：`rc=0", "标定阶梯：A 组通过", "新协议参数：L = 0.20 s（每步生效 0.80 s）",
                       "验收 2：逐集一致率 新对下午 1.00、新对晚上 0.95", "验收 4：K=1 成功 51/60，K=3 第 1 块通过 42/60，差（K=1 − K=3）+0.150", "## 1. 起跑条件与资源", "## 2. 标定阶梯", "## 3. 验收 2", "## 4. 验收 4",
                       "## 5. K=3 逐块存活", "## 6. 含迟到步的集的重跑记录", "## 7. 森林三档", "## 8. 全部臂的时序健康", "位移条款口径：cruise"):
            self.assertIn(needle, text)
        self.assertIn("预注册判定", text)

    def test_main_on_empty_dir_does_not_crash(self):
        with contextlib.redirect_stdout(io.StringIO()):
            r2.main()
        with open(os.path.join(self.root, "summary.md"), encoding="utf-8") as f:
            text = f.read()
        self.assertIn("标定阶梯：A 组还没跑", text)

    def test_fwd_samples_match_analyze_round1(self):
        rng = random.Random(7)
        for _ in range(50):
            acts = [{"action_executed": rng.choice([0, 1, 1, 1, 2, 3]), "collision": rng.random() < 0.05,
                     "displacement_m": None if rng.random() < 0.05 else rng.random()} for _ in range(rng.randint(1, 60))]
            self.assertEqual(r2.fwd_samples(acts), an.fwd_samples(acts))


if __name__ == "__main__":
    unittest.main(verbosity=2)
