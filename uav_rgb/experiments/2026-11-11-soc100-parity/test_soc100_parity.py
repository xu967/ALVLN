#!/usr/bin/env python3
"""满电档实验（任务卡 W1-10）的单测：预注册判决 judge、配对计数、位移口径与 summarize_r2.py 一致、探针的纯函数、run.sh 头部的预注册逐字核对。
不起 Gazebo、不读真实实验目录（造数据用 stub_eval.py）。用法：<.venv-tools 解释器> experiments/2026-11-11-soc100-parity/test_soc100_parity.py
（不在 tools/check.sh 的发现范围里：check.sh 只扫 sim/tools 与 tools 下的 test_*.py；需要时单独跑。）
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
sys.path.insert(0, os.path.join(REPO, "tools"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import summarize_parity as sp  # noqa: E402
import probe_plugin_params as pp  # noqa: E402

s1, s2 = sp.s1, sp.s2
STUB = os.path.join(HERE, "stub_eval.py")
SEEDS = ",".join(str(x) for x in sp.FROZEN_SEEDS)
PARAMS_COPY = "/root/rivermind-data/uav_rgb/params/e1_mini_soc100.json"   # 只在这台机器上有；没有就跳过用到它的测试


def read_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def base_m(**over):
    m = {"k1": {"disp": (0.2100, 0.2100), "turn": (25.0, 25.0), "b": 3, "c": 3, "n": 60},
         "k3": {"drops": (0, 0), "s1": (39, 39), "s3": (19, 19), "n": 60}}
    for k, v in over.items():
        grp, key = k.split("__")
        m[grp][key] = v
    return m


def verdicts(m):
    J = sp.judge(m)
    return J["verdict"], {i["id"]: ("证伪" if i["falsified"] else "过" if i["ok"] else "未过") for i in J["items"]}


class JudgeTests(unittest.TestCase):
    def test_all_pass(self):
        v, items = verdicts(base_m())
        self.assertEqual(v, "PASS")
        self.assertEqual(set(items.values()), {"过"})

    def test_displacement_thresholds(self):
        cases = [(1.00, "过"), (1.05, "过"), (0.95, "过"), (1.0501, "未过"), (0.94, "未过"), (1.10, "未过"), (0.90, "未过"), (1.1001, "证伪"), (0.89, "证伪"), (1.30, "证伪")]
        for scale, want in cases:
            with self.subTest(scale=scale):
                _v, items = verdicts(base_m(k1__disp=(1.0, scale)))
                self.assertEqual(items["P1"], want)

    def test_turn_thresholds_and_falsify(self):
        for scale, want, verdict in [(1.05, "过", "PASS"), (1.07, "未过", "FAIL"), (1.15, "证伪", "FALSIFIED"), (0.85, "证伪", "FALSIFIED")]:
            with self.subTest(scale=scale):
                v, items = verdicts(base_m(k1__turn=(25.0, 25.0 * scale)))
                self.assertEqual((items["P2"], v), (want, verdict))

    def test_falsify_beats_other_failures(self):
        v, items = verdicts(base_m(k1__disp=(1.0, 0.8), k3__drops=(0, 5)))
        self.assertEqual(v, "FALSIFIED")
        self.assertEqual(items["P4"], "未过")

    def test_nan_is_not_ok_and_not_falsified(self):
        v, items = verdicts(base_m(k1__disp=(float("nan"), 0.2)))
        self.assertEqual((items["P1"], v), ("未过", "FAIL"))
        v, items = verdicts(base_m(k1__turn=(0.0, 25.0)))     # 分母 0：rel 为 nan
        self.assertEqual(items["P2"], "未过")

    def test_mcnemar(self):
        for b, c, want in [(0, 0, "过"), (3, 3, "过"), (6, 2, "过"), (9, 1, "未过"), (0, 8, "未过"), (10, 0, "未过")]:
            with self.subTest(b=b, c=c):
                _v, items = verdicts(base_m(k1__b=b, k1__c=c))
                self.assertEqual(items["P3"], want)
        # 精确值：n=8, (6,2) → 2·(1+8+28)/256
        self.assertAlmostEqual(sp.mcnemar_exact(6, 2), 2 * (1 + 8 + 28) / 256)
        self.assertAlmostEqual(sp.mcnemar_exact(9, 1), 2 * (1 + 10) / 1024)

    def test_drops_must_be_zero_for_full_only(self):
        _v, items = verdicts(base_m(k3__drops=(4, 0)))
        self.assertEqual(items["P4"], "过")          # 对照有掉高不算数，只看满电
        _v, items = verdicts(base_m(k3__drops=(0, 1)))
        self.assertEqual(items["P4"], "未过")

    def test_first_block_gap(self):
        for full, want in [(39, "过"), (43, "过"), (35, "过"), (44, "未过"), (34, "未过"), (60, "未过")]:
            with self.subTest(full=full):
                _v, items = verdicts(base_m(k3__s1=(39, full)))
                self.assertEqual(items["P5"], want)          # 4/60 = 0.067 过；5/60 = 0.083 > 0.08 未过

    def test_all_three_blocks_not_fewer(self):
        for full, want in [(19, "过"), (25, "过"), (18, "未过"), (0, "未过")]:
            with self.subTest(full=full):
                _v, items = verdicts(base_m(k3__s3=(19, full)))
                self.assertEqual(items["P6"], want)

    def test_fail_when_one_item_fails_without_falsify(self):
        v, items = verdicts(base_m(k3__s1=(39, 50)))
        self.assertEqual((v, items["P5"]), ("FAIL", "未过"))

    def test_item_rows_render(self):
        for it in sp.judge(base_m())["items"]:
            row = sp.item_row(it)
            self.assertEqual(row.count("|"), 8, row)          # 7 列，表格里不能多出竖线


def fake_merged(rows):
    """rows = {seed: (success, passes[3], final, first_late)} → s2.merged_rows 的形状。"""
    out = {}
    for seed, (succ, passes, final, first_late) in rows.items():
        row = {"success": succ, "pass": passes, "z_tail_min": 2.0, "sim_s": 30.0, "steps": 30}
        out[seed] = {"row": row, "first": row, "first_late": first_late, "retry": None, "final": final}
    return out


class PairTests(unittest.TestCase):
    def test_pair_counts_k1(self):
        c = fake_merged({1: (True, [], "first", False), 113: (True, [], "first", False), 122: (False, [], "first", False), 153: (False, [], "first", False)})
        f = fake_merged({1: (True, [], "first", False), 113: (False, [], "first", False), 122: (True, [], "first", False), 153: (False, [], "first", False)})
        r = sp.pair_counts(c, f, 1)
        self.assertEqual(r, {"n": 4, "c_ok": 2, "f_ok": 2, "b": 1, "c": 1})

    def test_late_without_clean_result_counts_as_failure(self):
        c = fake_merged({1: (True, [], "first", False), 113: (True, [], "first", False)})
        f = fake_merged({1: (True, [], "first", True), 113: (True, [], "none", True)})     # 首次迟到且没拿到干净结果
        r = sp.pair_counts(c, f, 1)
        self.assertEqual((r["c_ok"], r["f_ok"], r["b"], r["c"]), (2, 0, 2, 0))

    def test_retry_result_counts(self):
        c = fake_merged({1: (False, [], "first", False)})
        f = fake_merged({1: (True, [], "retry", True)})
        r = sp.pair_counts(c, f, 1)
        self.assertEqual((r["f_ok"], r["c"]), (1, 1))

    def test_pair_counts_k3_first_block(self):
        c = fake_merged({1: (False, [True, False, False], "first", False), 113: (False, [False, False, False], "first", False)})
        f = fake_merged({1: (False, [True, True, False], "first", False), 113: (False, [True, False, False], "first", False)})
        r = sp.pair_counts(c, f, 3)
        self.assertEqual((r["c_ok"], r["f_ok"], r["b"], r["c"]), (1, 2, 0, 1))

    def test_only_seeds_in_both(self):
        c = fake_merged({1: (True, [], "first", False), 113: (True, [], "first", False)})
        f = fake_merged({1: (True, [], "first", False)})
        self.assertEqual(sp.pair_counts(c, f, 1)["n"], 1)


def run_stub(out, task="base", k_args=(), full=False, env=None, seeds=SEEDS, episodes=20, ckpt="x_s1_k10.bin"):
    cmd = [sys.executable, STUB, "--policy", "student", "--ckpt", ckpt, "--infer-bin", "infer", "--heading-guard", "2", "--task", task,
           "--steps", "80" if task == "base" else "240", "--split", "test", "--seeds", seeds, "--corridor-half-width", "3.0", "--action-latency", "0.20",
           "--episodes", str(episodes), "--dynamics", "cf_sitl", "--vehicle", "e1_mini", "--hold", "1.0", "--window-from", "observation", "--out", out]
    if task == "course_base":
        cmd += ["--blocks", "3"]
    if full:
        cmd += ["--cf-sitl-params", "/nonexistent/soc100.json"]
    r = subprocess.run(cmd + list(k_args), env={**os.environ, "STUB_SLEEP": "0", **(env or {})}, capture_output=True, text=True)
    if r.returncode not in (0, 1):            # 与真入口一致：有集失败（含迟到）退 1，其余退 0
        raise RuntimeError(f"桩异常退出 rc={r.returncode}\n{r.stderr[-2000:]}")


class SamplesTests(unittest.TestCase):
    """位移与转角的池化口径与 summarize_r2.py / summarize.py 的原函数一致（同一份数据，两种算法）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="soc100-test-")
        cls.arm = os.path.join(cls.tmp, "arm")
        run_stub(cls.arm, env={"STUB_STUDENT_LATE_SEEDS": "1", "STUB_STUDENT_LATE_ARM": "arm"})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_matches_disp_stats(self):
        for clause, key in (("cruise", "cruise_common"), ("steady", "steady_common")):
            with self.subTest(clause=clause):
                want_mean, want_n = s2.disp_stats(self.arm)[key]
                got = sp.pooled_samples([self.arm], clause)["disp"]
                self.assertEqual(len(got), want_n)
                self.assertAlmostEqual(sum(got) / len(got), want_mean, places=12)

    def test_matches_timing_stats_yaw(self):
        t = s1.timing_stats([self.arm])
        got = sp.pooled_samples([self.arm], "cruise")["yaw"]
        self.assertEqual(len(got), t["n_yaw"])
        self.assertAlmostEqual(sum(got) / len(got), t["yaw"], places=12)

    def test_pool_over_several_dirs_is_concatenation(self):
        other = os.path.join(self.tmp, "arm2")
        run_stub(other, ckpt="x_s2_k10.bin")
        a, b = sp.pooled_samples([self.arm], "cruise"), sp.pooled_samples([other], "cruise")
        both = sp.pooled_samples([self.arm, other], "cruise")
        self.assertEqual(both["disp"], a["disp"] + b["disp"])
        self.assertEqual(both["yaw"], a["yaw"] + b["yaw"])

    def test_step0_by_seed(self):
        t = sp.step0_by_seed(self.arm)
        self.assertEqual(len(t), 20)
        self.assertTrue(all(13.0 <= v <= 14.6 for v in t.values()), t)


class AnalyzeTests(unittest.TestCase):
    """两个目录（对照 + 满电）从桩生成，走完整的 analyze → 判决。"""

    def build(self, full_env=None, profile_bad=False):
        tmp = tempfile.mkdtemp(prefix="soc100-analyze-")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        ctrl, out = os.path.join(tmp, "ctrl"), os.path.join(tmp, "out")
        os.makedirs(out)
        for k in (1, 3):
            for s in (0, 1, 2):
                task = "base" if k == 1 else "course_base"
                run_stub(os.path.join(ctrl, f"acc-L0.20-k{k}-s{s}"), task=task, ckpt=f"x_s{s}_k10.bin")
                env = dict(full_env or {})
                if profile_bad:
                    env["STUB_FULL_PROFILE_BAD"] = "1"
                run_stub(os.path.join(out, f"soc100-L0.20-k{k}-s{s}"), task=task, ckpt=f"x_s{s}_k10.bin", full=True, env=env)
        return ctrl, out

    def test_pass_and_numbers(self):
        ctrl, out = self.build()
        A = sp.analyze(out, ctrl, 0.20, "cruise")
        self.assertEqual(A["status"], "PASS")
        c, f = A["judge_input"]["k1"]["disp"]
        self.assertAlmostEqual(c, 0.23)
        self.assertAlmostEqual(f, 0.23)
        self.assertEqual(A["judge_input"]["k1"]["n"], 60)
        self.assertEqual(A["judge_input"]["k3"]["n"], 60)
        self.assertTrue(all(abs(d - 2.5) < 1e-6 for d in A["step0"][1]["diff"]), "桩里满电多 2.5 s，配对差应都是 2.5")
        self.assertEqual(len(A["step0"][1]["diff"]), 60)
        text = "\n".join(sp.render(A))
        self.assertIn("判决：PASS", text)
        self.assertEqual(A["metric_versions"], ["sim09.v2.coast1s.corridor3m"])
        self.assertIn("metric_version `sim09.v2.coast1s.corridor3m`", text)
        self.assertIn("成功数的分母 = 请求集数（每臂 20 个冻结 seed）", text)

    def test_falsified_by_displacement(self):
        ctrl, out = self.build({"STUB_FULL_DISP_SCALE": "0.88"})
        A = sp.analyze(out, ctrl, 0.20, "cruise")
        self.assertEqual(A["status"], "FALSIFIED")
        self.assertLess(A["judge"]["items"][0]["diff"], -0.10)

    def test_invalid_profile(self):
        ctrl, out = self.build(profile_bad=True)
        A = sp.analyze(out, ctrl, 0.20, "cruise")
        self.assertEqual(A["status"], "INVALID")
        self.assertTrue(A["bad_profiles"])
        self.assertNotIn("judge", A)

    def test_invalid_when_pair_protocol_differs(self):
        ctrl, out = self.build()
        p = os.path.join(out, "soc100-L0.20-k1-s1", "eval.json")
        with open(p, encoding="utf-8") as f:
            e = json.load(f)
        e["heading_guard"] = 0
        with open(p, "w", encoding="utf-8") as f:
            json.dump(e, f)
        A = sp.analyze(out, ctrl, 0.20, "cruise")
        self.assertEqual(A["status"], "INVALID")
        self.assertEqual([k for k, _c, _f in A["proto_diff"][(1, 1)]], ["heading_guard"])

    def test_retried_arm_keeps_pair_protocol(self):
        """真入口里含迟到步的集进 failed[]，eval.json 的 seeds 不含它，requested_seeds 才是请求的全集；对照臂没有迟到集。
        配对协议键不能因此判 INVALID（第一个真实臂 soc100-L0.20-k1-s1 暴露：旧版把 seeds 放进 PAIR_KEYS，重跑过的臂必判 INVALID）。"""
        late = {"STUB_STUDENT_LATE_SEEDS": "113", "STUB_STUDENT_LATE_ARM": "soc100-L0.20-k1-s1"}
        ctrl, out = self.build(late)
        run_stub(os.path.join(out, "soc100-L0.20-k1-s1-retry1"), task="base", ckpt="x_s1_k10.bin", full=True, seeds="113", episodes=1, env=late)
        with open(os.path.join(out, "soc100-L0.20-k1-s1", "eval.json"), encoding="utf-8") as f:
            e = json.load(f)
        self.assertNotIn(113, e["seeds"], "桩应与真入口一致：迟到的集不在 seeds 里")
        self.assertIn(113, e["requested_seeds"])
        A = sp.analyze(out, ctrl, 0.20, "cruise")
        self.assertEqual(A["proto_diff"], {})
        self.assertEqual(A["status"], "PASS")

    def test_invalid_when_requested_seeds_differ(self):
        ctrl, out = self.build()
        p = os.path.join(out, "soc100-L0.20-k3-s0", "eval.json")
        with open(p, encoding="utf-8") as f:
            e = json.load(f)
        e["requested_seeds"] = e["requested_seeds"][:-1]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(e, f)
        A = sp.analyze(out, ctrl, 0.20, "cruise")
        self.assertEqual(A["status"], "INVALID")
        self.assertEqual([k for k, _c, _f in A["proto_diff"][(3, 0)]], ["requested_seeds"])

    def test_incomplete_when_arm_missing(self):
        ctrl, out = self.build()
        shutil.rmtree(os.path.join(out, "soc100-L0.20-k3-s2"))
        A = sp.analyze(out, ctrl, 0.20, "cruise")
        self.assertEqual(A["status"], "INCOMPLETE")
        self.assertTrue(any("K=3 s2" in x for x in A["missing"]))
        self.assertEqual(sp.EXIT[A["status"]], 2)
        sp.render(A)      # 不完整时也要能渲染

    def test_main_writes_summary_and_verdict(self):
        ctrl, out = self.build()
        with open(os.path.join(ctrl, "L_STAR"), "w") as f:
            f.write("0.20\n")
        with open(os.path.join(out, "config.env"), "w") as f:
            f.write(f"CONTROL={ctrl}\nDISP_CLAUSE=cruise\nPARAMS_PROFILE=cf_sitl_v1:06e5f017:lib31\n")
        rc = subprocess.run([sys.executable, os.path.join(HERE, "summarize_parity.py"), out], capture_output=True).returncode
        self.assertEqual(rc, 0)
        self.assertTrue(read_text(os.path.join(out, "VERDICT")).startswith("PASS"))
        self.assertIn("## 0. 结论一览", read_text(os.path.join(out, "summary.md")))


def synth_runs(copy_path, gap=0.38, default_path="/code/sim/dynamics/e1_mini.json"):
    runs = []
    for e in (0, 1, 2):
        for v in ("default", "copy"):
            rest = 3.80 + (gap if v == "copy" else 0.0)
            runs.append({"variant": v, "episode": e, "error": None, "params_in_world": copy_path if v == "copy" else "sim/dynamics/e1_mini.json",
                         "params_log": copy_path if v == "copy" else default_path, "lib_version": 31,
                         "rest": {"battery_v": rest, "hover_thrust": 0.9}, "ready": {"battery_v": rest - 0.3, "hover_thrust": 0.8, "throttle": 0.6, "sitl_t_s": 8.0 + e * 0.1}})
    return runs


class ProbeTests(unittest.TestCase):
    COPY = "/data/params/e1_mini_soc100.json"
    DEFAULT = "/code/sim/dynamics/e1_mini.json"

    def judge(self, runs, lib=31):
        return pp.judge_probe(runs, self.COPY, self.DEFAULT, lib)

    def test_parse_params_log(self):
        text = ("[Msg] Loading plugin\n[Msg] [CfSitlSystem] 就绪: link=base_link params=/a/b.json seed=0 action=/e1_mini/action cmd=/e1_mini/cmd state=/e1_mini/state@200Hz\n"
                "[Msg] [CfSitlSystem] reset episode=0 base_s=0\n")
        self.assertEqual(pp.parse_params_log(text), "/a/b.json")
        self.assertEqual(pp.parse_params_log(text + text.replace("/a/b.json", "/c/d.json")), "/c/d.json")
        self.assertIsNone(pp.parse_params_log("[Msg] nothing\n"))
        self.assertIsNone(pp.parse_params_log(""))

    def test_parse_params_log_real_gz_color_codes(self):
        # 真实 gz sim -v 3 的输出：每个 token 前后都有 ANSI 颜色码（2026-11-12 01:43 探针首次真跑的
        # $OUT/probe/gz_copy_e0.log 第 27 行，截到 action= 处）。上一条用的是手写的无颜色样例，没有拦住这个格式
        line = ("\x1b[1;32m[Msg] \x1b[0m\x1b[1;32m[CfSitlSystem] 就绪: link=\x1b[0m\x1b[1;32mbase_link\x1b[0m\x1b[1;32m params=\x1b[0m"
                "\x1b[1;32m/root/rivermind-data/uav_rgb/params/e1_mini_soc100.json\x1b[0m\x1b[1;32m seed=\x1b[0m\x1b[1;32m0\x1b[0m"
                "\x1b[1;32m action=\x1b[0m\x1b[1;32m/e1_mini/action\x1b[0m")
        reset = "\x1b[1;32m[Msg] \x1b[0m\x1b[1;32m[CfSitlSystem] reset episode=\x1b[0m\x1b[1;32m0\x1b[0m\x1b[1;32m base_s=\x1b[0m\x1b[1;32m0.398\x1b[0m"
        self.assertEqual(pp.parse_params_log(line + "\n" + reset + "\n"), "/root/rivermind-data/uav_rgb/params/e1_mini_soc100.json")
        self.assertIsNone(pp.parse_params_log(reset + "\n"))

    def test_world_params_path(self):
        one = '<plugin filename="CfSitlSystem" name="x"><link>b</link><params>sim/dynamics/e1_mini.json</params></plugin>'
        self.assertEqual(pp.world_params_path(one), "sim/dynamics/e1_mini.json")
        self.assertIsNone(pp.world_params_path(one + one))
        self.assertIsNone(pp.world_params_path("<sdf/>"))

    def test_pass(self):
        v = self.judge(synth_runs(self.COPY, default_path=self.DEFAULT))
        self.assertTrue(v["ok"], v["reasons"])
        self.assertAlmostEqual(v["rest_v_gap"], 0.38, places=6)
        self.assertEqual(v["log_lines_checked"], 6)

    def test_fails_when_plugin_ignored_the_copy(self):
        runs = synth_runs(self.COPY, gap=0.0)
        for r in runs:
            if r["variant"] == "copy":
                r["params_log"] = self.DEFAULT        # 日志也显示读的是默认
        v = self.judge(runs)
        self.assertFalse(v["ok"])
        self.assertTrue(any(x.startswith("G2") for x in v["reasons"]))
        self.assertTrue(any(x.startswith("G3") for x in v["reasons"]))

    def test_gap_threshold(self):
        self.assertTrue(self.judge(synth_runs(self.COPY, gap=0.25))["ok"])
        self.assertFalse(self.judge(synth_runs(self.COPY, gap=0.2499))["ok"])

    def test_missing_log_line_is_unverified_not_fail(self):
        runs = synth_runs(self.COPY)
        for r in runs:
            r["params_log"] = None
        v = self.judge(runs)
        self.assertTrue(v["ok"])
        self.assertEqual(v["log_lines_checked"], 0)

    def test_takeoff_error_fails(self):
        runs = synth_runs(self.COPY)
        runs[1]["error"] = "RuntimeError: 固件进入 FAILSAFE"
        v = self.judge(runs)
        self.assertFalse(v["ok"])
        self.assertTrue(any(x.startswith("G1") for x in v["reasons"]))

    def test_no_runs_for_a_variant_fails(self):
        v = self.judge([r for r in synth_runs(self.COPY) if r["variant"] == "default"])
        self.assertFalse(v["ok"])

    def test_world_params_mismatch_fails(self):
        runs = synth_runs(self.COPY)
        runs[1]["params_in_world"] = "sim/dynamics/e1_mini.json"      # copy 世界里没写副本
        v = self.judge(runs)
        self.assertTrue(any(x.startswith("G0") for x in v["reasons"]))

    def test_lib_version_mismatch_fails(self):
        v = self.judge(synth_runs(self.COPY), lib=30)
        self.assertTrue(any(x.startswith("G4") for x in v["reasons"]))

    def test_report_md_renders(self):
        runs = synth_runs(self.COPY, default_path=self.DEFAULT)
        verdict = self.judge(runs)
        text = pp.report_md({"path": self.COPY, "default_params": self.DEFAULT, "profile": "cf_sitl_v1:06e5f017:lib31", "sha256": "x"}, runs, verdict)
        self.assertIn("判决：通过", text)
        self.assertEqual(text.count("| copy |"), 4)    # 三次逐次记录 + 汇总行

    @unittest.skipUnless(os.path.exists(PARAMS_COPY), "满电副本只在数据盘上")
    def test_fingerprint_of_real_copy(self):
        problems, info = pp.fingerprint_problems(PARAMS_COPY, "cf_sitl_v1:06e5f017:lib31", None)
        self.assertEqual(problems, [])
        self.assertTrue(info["sha256"].startswith("cd822f80"))
        problems, _ = pp.fingerprint_problems(PARAMS_COPY, "cf_sitl_v1:1bc8fedf:lib31", "0" * 64)
        self.assertEqual(len(problems), 2)
        problems, _ = pp.fingerprint_problems("/nonexistent.json")
        self.assertIn("副本不存在", problems[0])


class RunShTests(unittest.TestCase):
    RUN = os.path.join(HERE, "run.sh")
    CARD = os.path.join(REPO, "docs", "briefs", "2026-11-10-phase12-task-cards.md")

    @staticmethod
    def squash(text):
        return re.sub(r"\s+", "", text)

    def header_block(self):
        lines, on = [], False
        with open(self.RUN, encoding="utf-8") as f:
            for line in f:
                if line.startswith("# ---") and "预注册（卡 W1-10" in line:
                    on = True
                    continue
                if on and line.startswith("# ---"):
                    break
                if on:
                    lines.append(line[1:])
        return "".join(lines)

    def test_preregistration_is_verbatim_from_card(self):
        block = self.squash(self.header_block())
        card = self.squash(read_text(self.CARD))
        pieces = ["K=1的每步前飞位移与转角（口径同`summarize_r2.py`）与50%档相差不超过5%，成功数配对无可检出的差别（精确McNemarp>0.05）；",
                  "K=3满电档里飞到电量掉高的集数为0，第1块通过率与50%档相差不超过0.08，三块全过的集数不少于50%档。证伪：K=1的位移或转角相差超过10%。",
                  "差超过10%则两档的数据与结论不能互用，报告里写明并停下。",
                  "第二轮没有`DONE`或`DONE`的rc不是0时先停下报告，不要自己起对照臂。"]
        last = -1
        for p in pieces:
            self.assertIn(p, card, f"卡里没有这句：{p}")
            at = block.find(p)
            self.assertGreater(at, last, f"run.sh 头部缺这句或顺序不对：{p}")
            last = at

    def test_thresholds_match_preregistration(self):
        self.assertEqual((sp.REL_PASS, sp.REL_FALSIFY, sp.P_MIN, sp.S1_GAP_MAX, sp.DROPS_MAX), (0.05, 0.10, 0.05, 0.08, 0))
        head = read_text(self.RUN).split("set -u", 1)[0]
        for needle in ("≤ 5%", "> 10%", "p > 0.05", "≤ 0.08", "z_tail_min < 1.5 m"):
            self.assertIn(needle, head)

    def test_bash_syntax(self):
        r = subprocess.run(["bash", "-n", self.RUN], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_allow_diff_pattern(self):
        m = re.search(r"^ALLOW_DIFF='(.*)'\s*(#.*)?$", read_text(self.RUN), re.M)
        self.assertIsNotNone(m)
        pat = re.compile(m.group(1))
        for line in ("Files 第二轮:sim/tools/scene_gen.py and 本实验:sim/tools/scene_gen.py differ",
                     "Files 第二轮:sim/tools/collect_parallel.py and 本实验:sim/tools/collect_parallel.py differ",
                     "Files 第二轮:sim/tools/eval_closed_loop.py and 本实验:sim/tools/eval_closed_loop.py differ",
                     "Only in 本实验:sim/tools: test_scene_cf_sitl_params.py",
                     "Files 第二轮:sim/cf_sitl/gz/README.md and 本实验:sim/cf_sitl/gz/README.md differ"):
            self.assertRegex(line, pat)
        for line in ("Files 第二轮:sim/tools/teacher.py and 本实验:sim/tools/teacher.py differ",
                     "Files 第二轮:sim/tools/collect.py and 本实验:sim/tools/collect.py differ",
                     "Files 第二轮:sim/dynamics/e1_mini.json and 本实验:sim/dynamics/e1_mini.json differ",
                     "Files 第二轮:sim/worlds/uav_rgb_e1.sdf and 本实验:sim/worlds/uav_rgb_e1.sdf differ",
                     "Only in 本实验:sim/tools: cf_sitl_exec2.py"):
            self.assertNotRegex(line, pat)

    def test_run_sh_is_executable_text_with_command_in_header(self):
        text = read_text(self.RUN)
        self.assertTrue(text.startswith("#!/usr/bin/env bash"))
        self.assertIn("setsid nohup bash run.sh", text.split("set -u", 1)[0])

    def test_acc_job_differs_from_round2_only_by_params_and_name(self):
        """命令行与第二轮 run_r2.sh 的 acc_job 只差臂名前缀与 --cf-sitl-params（逐词比较）。"""
        r2 = read_text(os.path.join(REPO, "experiments", "2026-11-10-action-latency", "run_r2.sh"))
        mine = read_text(self.RUN)
        pick = lambda t: re.search(r'echo "(?:acc|soc100)-L\$\(printf %\.2f "\$[1L_STAR]+"\)-k\$[12]-s\$[23]\|(.*?)\|\$EPISODES\|\$ept\|acc"', t)   # noqa: E731
        a, b = pick(r2), pick(mine)
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        norm = lambda s: s.replace("$3", "$S").replace("$2", "$S").replace("$1", "$L").replace("$L_STAR", "$L")   # noqa: E731
        wa, wb = norm(a.group(1)).split(), norm(b.group(1)).split()
        self.assertEqual([x for x in wb if x not in wa], ["--cf-sitl-params", "$PARAMS"])
        self.assertEqual([x for x in wa if x not in wb], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
