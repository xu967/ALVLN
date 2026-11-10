#!/usr/bin/env python3
"""bearing-guard-3dgs/summarize.py 新增的可选 config 键（routes 字典、gates / --gate、routes_selected 分桶、
main_pairs + verdict_texts、arm_checks）的单测，以及「不给这些键时输出不变」的锁定。只用标准库；每个用例造一个合成输出目录，
用子进程跑 summarize.py。

  PYTHONDONTWRITEBYTECODE=1 python test_summarize_params.py
  SUMMARIZE=<旧版 summarize.py> python test_summarize_params.py      拿改动前的版本跑，新功能的用例应当失败（红）
没设 TMPDIR 时临时目录落数据盘（根分区只剩 2 GB，不往 /tmp 写）。
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARIZE = os.environ.get("SUMMARIZE") or os.path.join(HERE, "..", "2026-11-10-bearing-guard-3dgs", "summarize.py")
_DATA_TMP = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "tmp")
if "TMPDIR" not in os.environ and os.path.isdir(_DATA_TMP):
    tempfile.tempdir = _DATA_TMP

ARMS = [{"name": "teacher", "base": "teacher", "ckpt": None, "guard": None},
        {"name": "fwd+r", "base": "forward", "ckpt": None, "guard": "route"},
        {"name": "stu+r", "base": "student", "ckpt": "x.bin", "guard": "route"}]
BASE_CFG = {"scenes": ["sA", "sB"], "k": 2, "seed0": 8000, "jitter": 0.2, "hold": 1.0, "guard_threshold_deg": 45.0,
            "route_lookahead_m": 1.5, "arms": ARMS}
GUARD = {"n_steps": 10, "n_triggered": 2, "n_rewritten": 1, "n_steps_pre": 8, "n_triggered_pre": 1, "n_rewritten_pre": 1,
         "plan_failed": False}


def write_arm_route(out, arm, scene, route, flags, k):
    """flags = 每集是否到点（长度 = 已写的集数）；done.json 的 episodes 与 metrics.json 一一对应。"""
    d = os.path.join(out, arm, scene, f"r{route}")
    guard = next(a["guard"] for a in ARMS if a["name"] == arm)
    for i, ok in enumerate(flags):
        ed = os.path.join(d, "ep_%05d" % i)
        os.makedirs(ed)
        seed = 8000 + 100 * route + i
        m = {"success_stop": ok, "success_v2": ok, "collision": not ok, "collide_first": not ok, "timeout": False, "steps": 10,
             "reach_step": 5 if ok else None, "collision_step": None if ok else 5, "spl": 0.5, "spl_stop": 0.5 if ok else 0.0,
             "final_wp_dist_m": 0.5, "min_wp_dist_m": 0.4, "start_pose": {"x": float(seed), "y": 0.0, "yaw_deg": 0.0},
             "arm": arm, "scene": scene, "route": route, "episode": i, "seed": seed, "stopped_on_goal": ok, "wall_s": 1.0,
             "guard": dict(GUARD) if guard else None}
        with open(os.path.join(ed, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(m, f)
    with open(os.path.join(d, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"episodes": [{"episode": i} for i in range(len(flags))], "failures": []}, f)


def run(out, *args):
    return subprocess.run([sys.executable, SUMMARIZE, out, *args], capture_output=True, text=True,
                          env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})


class Base(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.out = self._td.name
        self.addCleanup(self._td.cleanup)

    def build(self, cfg, data):
        """data = {(臂, 场景, 路线): [是否到点, ...]}"""
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        for (arm, scene, route), flags in data.items():
            write_arm_route(self.out, arm, scene, route, flags, cfg["k"])

    def summary(self):
        r = run(self.out)
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(os.path.join(self.out, "summary.md"), encoding="utf-8") as f:
            return f.read()


class TestLegacyUnchanged(Base):
    def test_scene_header_uses_k(self):
        self.build({**BASE_CFG, "routes": [0]}, {("teacher", "sA", 0): [True, False]})
        self.assertIn("## 2. 分场景 SR(stop)（每格 n = 已完成路线数 × 2，括号内是碰撞率）", self.summary())

    def test_no_new_keys_keeps_legacy_output(self):
        flags = [True, False, True, True, False]
        self.build({**BASE_CFG, "routes": [0, 1], "k": 5}, {("teacher", s, r): flags for s in ("sA", "sB") for r in (0, 1)})
        md = self.summary()
        self.assertTrue(md.startswith("# 3DGS 路线方位护栏 × {盲前进, 学生} 判别实验（bearing-guard-3dgs）\n"))
        self.assertIn("## 1. 四场景合计", md)
        self.assertIn("## 2. 分场景 SR(stop)（每格 n = 已完成路线数 × 5，括号内是碰撞率）", md)
        self.assertIn("| sA r0 | 3/5 |", md)                                  # 分路线表的行标不带路线名
        self.assertIn("P1–P3 是预注册的主比较（见 run.sh 文件头）；x1–x7 是补充比较，不进判决。", md)
        for absent in ("## 0.", "## 6.", "单臂预注册判决"):
            self.assertNotIn(absent, md)
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            self.assertEqual(sorted(json.load(f)), ["config", "failed", "metric_version", "overall", "pairs", "per_scene",
                                                    "routes_done", "spawn_pair_max_diff"])

    def test_arm_with_only_failed_episodes_does_not_crash(self):
        cfg = {**BASE_CFG, "routes": [0]}
        self.build(cfg, {})
        os.makedirs(os.path.join(self.out, "teacher", "sA", "r0"))
        with open(os.path.join(self.out, "teacher", "sA", "r0", "done.json"), "w", encoding="utf-8") as f:
            json.dump({"episodes": [], "failures": [{"episode": None, "error": "不可达"}]}, f)
        self.assertIn("0/2 = 0.00", self.summary())


class TestRoutesDictAndGate(Base):
    def cfg(self, **kw):
        return {**BASE_CFG, "routes": {"sA": [3, 7], "sB": [2]}, **kw}

    def data(self):
        d = {("teacher", "sA", 3): [True, True], ("teacher", "sA", 7): [True, False],      # sA 3/4 = 0.75
             ("teacher", "sB", 2): [True, True]}                                           # sB 2/2 = 1.0
        return d

    def test_routes_dict_read_per_scene(self):
        self.build(self.cfg(), self.data())
        md = self.summary()
        for row in ("| sA r3 | 2/2 |", "| sA r7 | 1/2 |", "| sB r2 | 2/2 |"):
            self.assertIn(row, md)
        self.assertNotIn("| sB r3 |", md)

    def test_gate_cli_prints_passing_scenes_and_writes_gate_json(self):
        self.build(self.cfg(), self.data())
        r = run(self.out, "--gate", "teacher:0.8")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "sB"))
        self.assertFalse(os.path.exists(os.path.join(self.out, "summary.md")))            # 闸门模式不写汇总
        with open(os.path.join(self.out, "gate.json"), encoding="utf-8") as f:
            g = json.load(f)
        self.assertEqual(g["passed"], ["sB"])
        self.assertEqual([x["scene"] for x in g["failed"]], ["sA"])
        self.assertEqual((g["scenes"]["sA"]["k_stop"], g["scenes"]["sA"]["n_req"], g["scenes"]["sA"]["kind"]), (3, 4, "low_sr"))
        self.assertEqual(run(self.out, "--gate", "teacher:0.75").stdout.split(), ["sA", "sB"])   # 恰好 0.75 算过（>=）

    def test_gate_does_not_pass_unfinished_or_routeless_scene(self):
        data = self.data()
        del data[("teacher", "sA", 7)]                                                    # sA 少一条路线的 done.json
        self.build(self.cfg(routes={"sA": [3, 7], "sB": [2], "sC": []}, scenes=["sA", "sB", "sC"]), data)
        r = run(self.out, "--gate", "teacher:0.5")
        self.assertEqual(r.stdout.split(), ["sB"])
        with open(os.path.join(self.out, "gate.json"), encoding="utf-8") as f:
            g = json.load(f)["scenes"]
        self.assertEqual((g["sA"]["kind"], g["sA"]["routes_done"], g["sA"]["routes_planned"]), ("incomplete", 1, 2))
        self.assertEqual(g["sC"]["kind"], "no_routes")

    def test_gate_unknown_arm_exits_nonzero(self):
        self.build(self.cfg(), self.data())
        self.assertNotEqual(run(self.out, "--gate", "nope:0.8").returncode, 0)

    def test_summary_marks_gate_failures(self):
        self.build(self.cfg(gates=[{"id": "P1", "arm": "teacher", "min_sr": 0.8, "note": "闸门未过：先查占据代理"}]), self.data())
        md = self.summary()
        self.assertIn("## 0. 闸门（预注册 P1）", md)
        self.assertIn("| sA | 3/4 = 0.75 | 2/2 | 闸门未过：先查占据代理（SR(stop) 0.75 < 0.8） |", md)
        self.assertIn("| sB | 2/2 = 1.00 | 1/1 | 通过 |", md)


class TestBucketsPairsChecks(Base):
    def build_pilot(self, **kw):
        sel = {"params": {"bucket_labels": ["<5 m", "5–10 m", ">=10 m"]},
               "scenes": {"sA": [{"route": 3, "name": "tt_3", "astar_path_m": 3.0, "bucket": "<5 m"},
                                 {"route": 7, "name": "tt_7", "astar_path_m": 12.0, "bucket": ">=10 m"}], "sB": []}}
        with open(os.path.join(self.out, "routes_selected.json"), "w", encoding="utf-8") as f:
            json.dump(sel, f)
        cfg = {**BASE_CFG, "routes": {"sA": [3, 7], "sB": []}, "routes_selected": os.path.join(self.out, "routes_selected.json"),
               "title": "试点", "overall_label": "合计", "main_pairs": [["P3", "stu+r", "fwd+r"]], **kw}
        data = {("fwd+r", "sA", 3): [True, True], ("fwd+r", "sA", 7): [False, False],
                ("stu+r", "sA", 3): [True, True], ("stu+r", "sA", 7): [True, False]}
        self.build(cfg, data)

    def test_bucket_table_and_route_labels(self):
        self.build_pilot()
        md = self.summary()
        self.assertIn("## 6. 按路线长度分桶", md)
        self.assertIn("| <5 m（1） | — | 2/2 = 1.00 | 2/2 = 1.00 |", md)
        self.assertIn("| 5–10 m（0） | — | — | — |", md)
        self.assertIn("| >=10 m（1） | — | 0/2 = 0.00 | 1/2 = 0.50 |", md)
        self.assertIn("| sA r7（tt_7，12.0 m） |", md)
        with open(os.path.join(self.out, "summary.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["buckets"][">=10 m|stu+r"], {"k_stop": 1, "n_req": 2})

    def test_missing_routes_selected_file_fails_loudly(self):
        self.build_pilot()
        os.remove(os.path.join(self.out, "routes_selected.json"))
        self.assertNotEqual(run(self.out).returncode, 0)

    def test_custom_main_pairs_use_verdict_texts(self):
        texts = {"P3": {"ns": "无差别", "up": "学生更好", "down": "学生更差"}}
        self.build_pilot(verdict_texts=texts)
        md = self.summary()
        self.assertIn("| P3 | `stu+r` vs `fwd+r` | 4 | 1 | 0 |", md)              # b=1（r7 的一集），c=0，p=1.0
        self.assertIn("| 无差别 |", md)
        self.assertIn("P3 是预注册的主比较（见 run.sh 文件头）；判决由脚本", md)       # 没有补充比较时不写 x1–xN
        self.assertNotIn("支持预测（b>c 且", md)                                       # 不套用内置的 P1–P3 判决

    def test_pair_verdict_direction(self):
        sA = {("fwd+r", "sA", r): [False] * 2 for r in range(3, 11)}                  # 8 条路线，盲飞全败
        sA.update({("stu+r", "sA", r): [True] * 2 for r in range(3, 11)})             # 学生全成：b=16, c=0, p≈3e-5
        up = {"P3": {"ns": "NS", "up": "UP", "down": "DOWN"}}
        cfg = {**BASE_CFG, "routes": {"sA": list(range(3, 11)), "sB": []}, "main_pairs": [["P3", "stu+r", "fwd+r"]], "verdict_texts": up}
        self.build(cfg, sA)
        self.assertIn("| UP |", self.summary())
        cfg["main_pairs"] = [["P3", "fwd+r", "stu+r"]]                                # 对调 A/B：显著但方向相反
        with open(os.path.join(self.out, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        self.assertIn("| DOWN |", self.summary())

    def test_arm_checks_three_outcomes(self):
        chk = {"id": "P2", "arm": "fwd+r", "support_lt": 0.7, "falsify_ge": 0.89, "texts": {"support": "支持", "falsify": "证伪", "between": "之间"}}
        for flags, want in (([True, False, False, False], "支持"), ([True, True, False, True], "之间"), ([True] * 4, "证伪")):
            with self.subTest(want=want):
                self.out = tempfile.mkdtemp(dir=self._td.name)
                cfg = {**BASE_CFG, "routes": {"sA": [3, 7], "sB": []}, "arm_checks": [chk]}
                self.build(cfg, {("fwd+r", "sA", 3): flags[:2], ("fwd+r", "sA", 7): flags[2:]})
                md = self.summary()
                self.assertRegex(md, r"\| P2 \| `fwd\+r` \| \d/4 = [\d.]+ \[[\d., ]+\] \| 0\.7 \| 0\.89 \| " + want + r" \|")


if __name__ == "__main__":
    unittest.main(verbosity=2)
