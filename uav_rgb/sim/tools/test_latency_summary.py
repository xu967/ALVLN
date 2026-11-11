#!/usr/bin/env python3
"""experiments/2026-11-10-action-latency/summarize.py 的单测（合成数据，不起 gz）。

背景：第一轮标定守护进程（2026-11-10）结束时自动跑的 summarize.py 在 forest_section 崩了：真实 eval.json 的
per_episode[].pillars 是整数（柱子个数），代码按列表取 len()，TypeError，所以实验目录没有 summary.md。
演练用的假评测入口写的是列表，没覆盖到。这里按真实格式（整数）建夹具。

覆盖：forest_section 对整数 / 列表两种 pillars、缺 pillars 的处理；main(note=...) 把说明写在 summary.md 开头。
运行（仓库根）：.venv-tools/bin/python sim/tools/test_latency_summary.py
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

EXP = os.path.join(HERE, "..", "..", "experiments", "2026-11-10-action-latency")
_SPEC = importlib.util.spec_from_file_location("latency_summarize", os.path.join(EXP, "summarize.py"))
sm = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sm)


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_arm(root, name, episodes, pillars=16, rtf=0.8, hold_s=1.0):
    """一个臂目录：episodes = {seed: [每步 dispatch_lag_s]}。

    per_episode[].pillars 写成整数 `pillars`（真实 eval.json 的格式）；pillars=None 时不写这个键，
    pillars 是列表时原样写进去（旧演练夹具的格式）。
    """
    arm = os.path.join(root, name)
    per_episode = []
    for seed, lags in episodes.items():
        ep = os.path.join(arm, "episodes", f"seed_{seed}")
        os.makedirs(ep)
        write_jsonl(os.path.join(ep, "actions.jsonl"), [
            {"step": i, "action_executed": 1, "collision": False, "displacement_m": 0.25, "dyaw_deg": 0.0,
             "dispatch_lag_s": lag} for i, lag in enumerate(lags)])
        write_jsonl(os.path.join(ep, "oracle.jsonl"), [
            {"step": i, "action_start_ns": int((10.0 + i * hold_s + lag) * 1e9),
             "action_end_ns": int((10.0 + (i + 1) * hold_s) * 1e9)} for i, lag in enumerate(lags)])
        with open(os.path.join(ep, "episode_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"performance": {"steps": len(lags), "action_rtf": rtf, "action_sim_s": hold_s * len(lags)}}, f)
        row = {"seed": seed, "steps": len(lags), "success": True}
        if pillars is not None:
            row["pillars"] = pillars
        per_episode.append(row)
    with open(os.path.join(arm, "eval.json"), "w", encoding="utf-8") as f:
        json.dump({"per_episode": per_episode, "episodes_done": len(per_episode), "failed_count": 0}, f)
    return arm


class ForestSection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(sm, "OUT", self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)

    def build(self, **kw):
        for dens in ("sparse", "medium", "dense"):
            write_arm(self.tmp.name, f"forest-{dens}", {4100: [0.25, 0.13, 0.14, 0.13], 4101: [0.24, 0.13, 0.13, 0.15]}, **kw)

    def rows(self):
        return [ln for ln in sm.forest_section() if ln.startswith("| ") and "未完成" not in ln and not ln.startswith("| 密度")]

    def test_integer_pillars_as_in_real_eval_json(self):
        """真实 eval.json：per_episode[].pillars = 柱子个数（整数）。旧代码在这里 TypeError: object of type 'int' has no len()。"""
        self.build(pillars=16)
        rows = self.rows()
        self.assertEqual(len(rows), 3)
        cells = [c.strip() for c in rows[0].strip("|").split("|")]
        self.assertEqual(cells[0], "sparse")
        self.assertEqual(cells[3], "16", "柱子数一列应是 eval.json 里的整数 16")

    def test_list_pillars_still_counted(self):
        self.build(pillars=[{"x": 1.0, "y": 0.0}] * 8)
        cells = [c.strip() for c in self.rows()[0].strip("|").split("|")]
        self.assertEqual(cells[3], "8")

    def test_missing_pillars_key_does_not_crash(self):
        self.build(pillars=None)
        self.assertEqual(len(self.rows()), 3)

    def test_rtf_need_uses_steady_steps_only(self):
        """第 0 步（冷缓存）单列，不进稳态：稳态最大 0.15 s 低于 L=0.16 的阈值 0.18 s，所以不需要 --rtf。"""
        self.build(pillars=16)
        row = self.rows()[0]
        self.assertIn("0.0% → 不需要", row)


class HeaderNote(unittest.TestCase):
    def test_note_is_written_below_the_title(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = ([], {})
            with mock.patch.object(sm, "calib_section", return_value=empty), \
                    mock.patch.object(sm, "item2_section", return_value=[]), \
                    mock.patch.object(sm, "forest_section", return_value=[]), \
                    mock.patch.object(sm, "item4_section", return_value=[]), \
                    mock.patch.object(sm, "overview_section", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                sm.main(tmp, note="事后用修复后的脚本生成，提交 abc1234")
            with open(os.path.join(tmp, "summary.md"), encoding="utf-8") as f:
                lines = f.read().splitlines()
        self.assertTrue(lines[0].startswith("# "))
        self.assertIn("事后用修复后的脚本生成，提交 abc1234", "\n".join(lines[:4]))

    def test_no_note_leaves_the_header_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sm, "calib_section", return_value=([], {})), \
                    mock.patch.object(sm, "item2_section", return_value=[]), \
                    mock.patch.object(sm, "forest_section", return_value=[]), \
                    mock.patch.object(sm, "item4_section", return_value=[]), \
                    mock.patch.object(sm, "overview_section", return_value=[]), \
                    contextlib.redirect_stdout(io.StringIO()):
                sm.main(tmp)
            with open(os.path.join(tmp, "summary.md"), encoding="utf-8") as f:
                text = f.read()
        self.assertNotIn("事后", text)


if __name__ == "__main__":
    unittest.main()
