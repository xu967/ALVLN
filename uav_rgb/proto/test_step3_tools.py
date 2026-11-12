#!/usr/bin/env python3
"""第三关的对拍与验收脚本的测试：compare_training.py、accept_offline.py、distribution_check.py、crosscheck_evalonly.py、probe_divergence.first_exceed。
只用合成的 run 目录（几十毫秒）；numpy 即可，probe_divergence 那一条没有 torch 时跳过。
运行：nice -n 10 python -m unittest proto.test_step3_tools -v
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accept_offline as ao  # noqa: E402
import ckpt_io as cio  # noqa: E402
import compare_training as ct  # noqa: E402
import closedloop_summary as cl  # noqa: E402
import crosscheck_evalonly as cx  # noqa: E402
import distribution_check as dc  # noqa: E402
import metrics as M  # noqa: E402
from test_ckpt_io import make_ck  # noqa: E402

try:
    import probe_divergence as pd  # noqa: E402
except ImportError:      # 没有 torch
    pd = None

ST = dict(in_h=16, in_w=24, in_c=3, num_classes=4)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **k)


def jl(path, events):
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")


def conf_event(split, step, conf, loss=0.5):
    conf = np.asarray(conf, np.int64)
    s = M.summarize(conf)
    return {"schema": "tinyvln.metrics.v1", "event": "validation", "step": step, "loss": loss, "action_accuracy": s["accuracy"],
            "split": split, "macro_f1": s["macro_f1"], "balanced_accuracy": s["balanced_accuracy"], "confusion": conf.reshape(-1).tolist()}


def write_run(d, *, best_epoch, val_confs, test_conf, step_losses=None, results=None, ckpts=None, val_f1=None):
    """合成一个 run 目录：metrics.jsonl（状态、train_step、val 事件、best 提示、test 事件）、results.csv、ckpt_epNN.bin。"""
    os.makedirs(d, exist_ok=True)
    ev = [{"schema": "tinyvln.metrics.v1", "event": "status", "message": "start"}]
    for i, loss in enumerate(step_losses or [], 1):
        ev.append({"schema": "tinyvln.metrics.v1", "event": "train_step", "step": i, "loss": float(loss), "action_accuracy": 0.5})
    for i, conf in enumerate(val_confs, 1):
        e = conf_event("val", i * 65, conf)
        if val_f1 is not None:
            e["macro_f1"] = val_f1[i - 1]
        ev.append(e)
    ev.append({"schema": "tinyvln.metrics.v1", "event": "status", "message": f"best checkpoint: epoch {best_epoch} macro_f1=0.5"})
    ev.append(conf_event("test", len(val_confs) * 65, test_conf))
    jl(os.path.join(d, "metrics.jsonl"), ev)
    with open(os.path.join(d, "results.csv"), "w", encoding="utf-8") as f:
        f.write("epoch,train_ce,dev_ce,dev_accuracy\n")
        for i, (tr, dv) in enumerate(results or [(1.0, 1.0)] * len(val_confs), 1):
            f.write(f"{i},{tr},{dv},0.5\n")
    for e, ck in (ckpts or {}).items():
        cio.write_checkpoint(os.path.join(d, f"ckpt_ep{e:02d}.bin"), ck)
    return d


BASE = [[5, 2, 1, 0], [3, 20, 2, 1], [1, 4, 12, 3], [0, 3, 2, 10]]


def jitter(base, k):
    c = np.array(base, np.int64)
    c[1, 1] += k
    c[1, 2] -= k
    c[2, 2] -= k
    c[2, 1] += k
    return c


class TestCompareTraining(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        n = 24
        self.ref_loss = [1.0 + 0.05 * i for i in range(n)]
        d = np.zeros(n)
        d[1] = 2e-6          # 第 2 步
        d[2:] = 3e-5         # 第 3 步起
        d[5:] = 3e-4         # 第 6 步起
        d[8:] = 3e-3         # 第 9 步起
        d[11:] = 3e-2        # 第 12 步起
        self.prot_loss = list(np.array(self.ref_loss) + d)
        self.ck_ref = {1: make_ck(1, **ST), 2: make_ck(2, **ST)}
        self.ck_pro = {}
        for e, ck in self.ck_ref.items():
            cp = make_ck(e, **ST)
            cp.params = [p * np.float32(1.001) for p in cp.params]
            self.ck_pro[e] = cp
        self.ref = write_run(os.path.join(self.root, "ref"), best_epoch=2, val_confs=[BASE, BASE], test_conf=BASE,
                             step_losses=self.ref_loss, results=[(1.0, 0.9), (0.8, 0.7)], ckpts=self.ck_ref, val_f1=[0.4, 0.5])
        self.pro = write_run(os.path.join(self.root, "pro"), best_epoch=2, val_confs=[BASE, BASE], test_conf=BASE,
                             step_losses=self.prot_loss, results=[(1.1, 0.8), (0.9, 0.6)], ckpts=self.ck_pro, val_f1=[0.45, 0.48])
        self.other = write_run(os.path.join(self.root, "other"), best_epoch=1, val_confs=[BASE], test_conf=BASE, ckpts={
            1: make_ck(77, **ST), 2: make_ck(78, **ST)})

    def tearDown(self):
        self.tmp.cleanup()

    def test_first_exceed_steps(self):
        r = ct.compare_steps(self.pro, self.ref)
        self.assertEqual(r["steps_compared"], 24)
        self.assertEqual(r["first_exceed"], {"1e-05": 3, "0.0001": 6, "0.001": 9, "0.01": 12})
        self.assertEqual(len(r["first_rows"]), ct.N_FIRST)
        self.assertEqual(r["first_rows"][0]["diff"], 0.0)
        self.assertAlmostEqual(r["max_abs_diff_all"], 3e-2, places=9)
        self.assertEqual(r["accuracy_identical_steps"], 24)

    def test_identical_runs_never_exceed(self):
        r = ct.compare_steps(self.ref, self.ref)
        self.assertEqual(r["first_exceed"], {"1e-05": None, "0.0001": None, "0.001": None, "0.01": None})
        self.assertEqual(r["max_abs_diff_all"], 0.0)

    def test_epoch_table(self):
        r = ct.compare_epochs(self.pro, self.ref)
        self.assertEqual(r["epochs_compared"], 2)
        self.assertEqual(r["rows"][0]["val_macro_f1"], [0.45, 0.4])
        self.assertAlmostEqual(r["mean_abs_diff"]["dev_ce"], 0.5 * (abs(0.8 - 0.9) + abs(0.6 - 0.7)))

    def test_weight_distance_matches_numpy(self):
        w = ct.compare_weights(self.pro, self.ref, (1, 2, 3))
        self.assertEqual(sorted(w), [1, 2])          # 第 3 轮没有 checkpoint，跳过
        a = np.concatenate([np.asarray(p, np.float64).reshape(-1) for p in self.ck_pro[1].params])
        b = np.concatenate([np.asarray(p, np.float64).reshape(-1) for p in self.ck_ref[1].params])
        self.assertAlmostEqual(w[1]["rel_l2"], float(np.linalg.norm(a - b) / np.linalg.norm(b)), places=12)
        self.assertAlmostEqual(w[1]["rel_l2"], 1e-3, delta=1e-4)
        self.assertEqual(list(w[1]["per_tensor"]), list(ct.LAYER_NAMES))

    def test_baseline_dedupes_same_run_and_uses_independent_seeds(self):
        res = ct.build({"p": (self.pro, self.ref), "again": (self.pro, self.ref)}, {"o": self.other}, (1, 2))
        b = res["baseline_other_seeds"]
        self.assertEqual(sorted(b[1]["pairs"]), ["extra:o|ref:p"])        # ref 只算一次，不会出现距离 0 的重复对
        self.assertGreater(b[1]["mean"], 1.0)                             # 独立初值的随机权重，距离在 1.4 附近
        self.assertEqual(sorted(res["pairs"]), ["again", "p"])

    def test_cli_writes_files_and_rejects_missing(self):
        out_j, out_m = os.path.join(self.root, "o.json"), os.path.join(self.root, "o.md")
        rc = quiet(ct.main, ["--pair", f"p:{self.pro}:{self.ref}", "--extra", f"o:{self.other}", "--epochs", "1", "2",
                             "--out-json", out_j, "--out-md", out_m])
        self.assertEqual(rc, 0)
        with open(out_j, encoding="utf-8") as f:
            self.assertIn("pairs", json.load(f))
        with open(out_m, encoding="utf-8") as f:
            text = f.read()
        self.assertIn("第 3 步", text)
        self.assertEqual(quiet(ct.main, ["--pair", f"p:{self.root}/nope:{self.ref}"]), 2)
        self.assertEqual(quiet(ct.main, []), 2)


class TestAcceptOffline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_dir(self, name, conf_val, conf_test, best_epoch=2):
        return write_run(os.path.join(self.root, name), best_epoch=best_epoch, val_confs=[BASE, conf_val], test_conf=conf_test)

    def refs(self):
        return {f"s{k}": self.run_dir(f"s{k}", jitter(BASE, k), jitter(BASE, -k)) for k in range(3)}

    def cli(self, refs, protos, controls=None):
        args = []
        for n, d in refs.items():
            args += ["--ref", f"{n}={d}"]
        for n, d in protos.items():
            args += ["--proto", f"{n}={d}"]
        for n, d in (controls or {}).items():
            args += ["--control", f"{n}={d}"]
        out_j = os.path.join(self.root, "a.json")
        rc = quiet(ao.main, args + ["--out-json", out_j, "--out-md", os.path.join(self.root, "a.md")])
        with open(out_j, encoding="utf-8") as f:
            return rc, json.load(f)

    def test_delivered_uses_best_epoch_for_val_and_last_test(self):
        d = write_run(os.path.join(self.root, "x"), best_epoch=1, val_confs=[BASE, jitter(BASE, 3)], test_conf=jitter(BASE, 2))
        r = ao.delivered(d)
        self.assertEqual(r["best_epoch"], 1)
        self.assertEqual(r["val"]["confusion"], np.asarray(BASE).reshape(-1).tolist())
        self.assertEqual(r["test"]["confusion"], jitter(BASE, 2).reshape(-1).tolist())
        self.assertLess(r["logged_f1_diff"], 1e-12)

    def test_run_inside_range_passes_literal(self):
        rc, res = self.cli(self.refs(), {"p0": self.run_dir("p0", jitter(BASE, 1), jitter(BASE, -1))})
        self.assertEqual(rc, 0)
        self.assertTrue(res["literal"]["pass"])
        self.assertEqual(res["literal"]["failures"], [])

    def test_collapsed_run_fails_and_names_the_items(self):
        collapsed = [[0, 8, 0, 0], [0, 26, 0, 0], [0, 20, 0, 0], [0, 15, 0, 0]]
        rc, res = self.cli(self.refs(), {"p0": self.run_dir("p0", collapsed, collapsed)})
        self.assertEqual(rc, 1)
        self.assertFalse(res["literal"]["pass"])
        self.assertIn("p0 val macro_f1", res["literal"]["failures"])
        self.assertIn("p0 test recall2", res["literal"]["failures"])

    def test_class_with_one_true_sample_separates_literal_and_supplementary(self):
        one = lambda row0: [row0, [3, 20, 2, 1], [1, 4, 12, 3], [0, 3, 2, 10]]   # noqa: E731  类 0 只有 1 个真值样本
        refs = {f"s{k}": self.run_dir(f"s{k}", jitter(BASE, k), one([0, 1, 0, 0])) for k in range(3)}      # 参照：这 1 帧被判成类 1，召回 0
        p = self.run_dir("p0", jitter(BASE, 1), one([1, 0, 0, 0]))                                          # 原型：判对，召回 1
        rc, res = self.cli(refs, {"p0": p})
        self.assertEqual(rc, 1)
        self.assertIn("p0 test recall0", res["literal"]["failures"])
        # 去掉该类的召回后，recall0 不再判；宏 F1 仍含该类的 F1，那 1 帧判对使宏 F1 变化，仍判为未通过
        self.assertEqual(res["excluding_tiny_classes"]["failures"], ["p0 test macro_f1"])
        tiny = {(r["split"], r["metric"]): r["tiny"] for r in res["rows"]}
        self.assertTrue(tiny[("test", "recall0")])
        self.assertFalse(tiny[("test", "recall1")])

    def test_controls_and_baseline_need_six_runs(self):
        controls = {f"c{k}": self.run_dir(f"c{k}", jitter(BASE, k + 3), jitter(BASE, -k - 3)) for k in range(3)}
        rc, res = self.cli(self.refs(), {"p0": self.run_dir("p0", jitter(BASE, 1), jitter(BASE, -1))}, controls)
        self.assertIn("baseline_literal", res["controls"])
        self.assertEqual(res["controls"]["baseline_literal"]["per_seed_trials"], 20 * 3)     # C(6,3)=20 个参照三元组 × 3 个候选
        rc2, res2 = self.cli(self.refs(), {"p0": self.run_dir("p0b", jitter(BASE, 1), jitter(BASE, -1))},
                             {"c0": controls["c0"]})
        self.assertNotIn("baseline_literal", res2["controls"])

    def test_bad_input_returns_2(self):
        self.assertEqual(quiet(ao.main, ["--ref", f"s0={self.root}/none", "--ref", f"s1={self.root}/none", "--proto", f"p={self.root}/none"]), 2)
        self.assertEqual(quiet(ao.main, ["--ref", "s0=x", "--proto", "p=y"]), 2)


class TestCrosscheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, cpp_val=None, cpp_test=None, cpp_loss=0.5):
        run = write_run(os.path.join(self.root, "run"), best_epoch=2, val_confs=[jitter(BASE, 1), BASE], test_conf=jitter(BASE, 2))
        ev = os.path.join(self.root, "eval")
        os.makedirs(ev, exist_ok=True)
        jl(os.path.join(ev, "metrics.jsonl"), [conf_event("val", 1, BASE if cpp_val is None else cpp_val, cpp_loss),
                                               conf_event("test", 1, jitter(BASE, 2) if cpp_test is None else cpp_test, cpp_loss)])
        return run, ev

    def test_identical(self):
        run, ev = self.make()
        out = os.path.join(self.root, "x.json")
        self.assertEqual(quiet(cx.main, ["--run", run, "--eval-dir", ev, "--out", out]), 0)
        with open(out, encoding="utf-8") as f:
            text = f.read()
        self.assertIn('"identical": true', text)        # run_step3_train.sh 靠这一行判断是否已通过

    def test_confusion_differs(self):
        run, ev = self.make(cpp_test=jitter(BASE, 3))
        out = os.path.join(self.root, "x.json")
        self.assertEqual(quiet(cx.main, ["--run", run, "--eval-dir", ev, "--out", out]), 1)
        with open(out, encoding="utf-8") as f:
            self.assertIn('"identical": false', f.read())

    def test_ce_differs(self):
        run, ev = self.make(cpp_loss=0.5 + 1e-3)
        self.assertEqual(quiet(cx.main, ["--run", run, "--eval-dir", ev]), 1)

    def test_missing_files(self):
        self.assertEqual(quiet(cx.main, ["--run", f"{self.root}/a", "--eval-dir", f"{self.root}/b"]), 2)


SEEDS20 = [1, 113, 122, 153, 172, 176, 180, 230, 250, 536, 555, 628, 683, 798, 929, 990, 992, 997, 1113, 1654]


def make_arm(d, wins, late=(), failed=()):
    """合成一个评测臂目录：wins = 成功的 seed 集合（其余 seed 失败）；late = 含迟到步的 seed（只进 late_dispatch_episodes 与 failed，不进 per_episode）；
    failed = 其他中止的 seed。"""
    os.makedirs(d, exist_ok=True)
    gone = set(late) | set(failed)
    per = [{"seed": s, "success": s in wins, "steps": 30} for s in SEEDS20 if s not in gone]
    ev = {"schema": "uavrgb.eval.v1", "partial": False, "per_episode": per, "episodes_done": len(per),
          "failed_count": len(gone),
          "late_dispatch_episodes": [{"seed": s, "late_dispatch_steps": 1, "late_step_ids": [1]} for s in late],
          "failed": [{"seed": s, "rc": 0, "error": "late_dispatch: 1 步迟到"} for s in late] + [{"seed": s, "rc": 1, "error": "collect failed"} for s in failed]}
    with open(os.path.join(d, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f)
    return d


def make_retry(d, seeds, wins, late=()):
    """<臂>-retry1：只含重跑的 seed。"""
    os.makedirs(d, exist_ok=True)
    per = [{"seed": s, "success": s in wins} for s in seeds if s not in late]
    ev = {"partial": False, "per_episode": per, "episodes_done": len(per), "failed_count": len(late),
          "late_dispatch_episodes": [{"seed": s} for s in late], "failed": [{"seed": s, "error": "late_dispatch"} for s in late]}
    with open(os.path.join(d, "eval.json"), "w", encoding="utf-8") as f:
        json.dump(ev, f)
    return d


class TestClosedloopSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def arms(self, proto_wins, ref_wins):
        """proto_wins / ref_wins：{训练种子名: 成功 seed 集合}。"""
        proto, ref = {}, {}
        for n in proto_wins:
            proto[n] = make_arm(os.path.join(self.root, "proto", n), proto_wins[n])
            ref[n] = make_arm(os.path.join(self.root, "ref", n), ref_wins[n])
        return proto, ref

    def test_mcnemar(self):
        self.assertEqual(cl.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(cl.mcnemar_exact(5, 0), 2 / 32)
        self.assertEqual(cl.mcnemar_exact(3, 3), 1.0)

    def test_equal_totals_pass(self):
        wins = {"s0": set(SEEDS20[:18]), "s1": set(SEEDS20[:16]), "s2": set(SEEDS20[:19])}
        proto, ref = self.arms(wins, wins)
        r = cl.summarize(proto, ref, SEEDS20)
        self.assertEqual((r["total"]["proto_successes"], r["total"]["ref_successes"], r["total"]["delta"]), (53, 53, 0))
        self.assertTrue(r["total"]["within_tolerance"] and r["total"]["complete"])
        self.assertTrue(r["verdict"].startswith("预注册通过"))
        self.assertEqual(r["pairs"]["s0"]["both"], 18)
        self.assertEqual(r["pairs"]["s0"]["proto_only"] + r["pairs"]["s0"]["ref_only"], 0)

    def test_tolerance_boundary_is_six(self):
        ref_w = {"s0": set(SEEDS20[:18]), "s1": set(SEEDS20[:16]), "s2": set(SEEDS20[:19])}      # 53/60，第二轮三个 K=1 臂
        split = lambda total: [min(20, total), min(20, max(0, total - 20)), max(0, total - 40)]  # noqa: E731  把总数摊到 3 个臂
        for total, ok in ((59, True), (60, False), (47, True), (46, False), (53, True)):
            parts = split(total)
            wins = {f"s{i}": set(SEEDS20[:parts[i]]) for i in range(3)}
            r = cl.summarize(*self.arms(wins, ref_w), SEEDS20)
            self.assertEqual(r["total"]["proto_successes"], total)
            self.assertEqual(r["total"]["delta"], total - 53)
            self.assertEqual(r["total"]["within_tolerance"], ok, total)
            self.assertTrue(r["verdict"].startswith("预注册通过" if ok else "预注册未通过"), (total, r["verdict"]))

    def test_late_episode_uses_retry_and_incomplete_when_still_late(self):
        wins = set(SEEDS20[:15])
        late = [SEEDS20[3], SEEDS20[4]]
        proto = {"s0": make_arm(os.path.join(self.root, "proto", "s0"), wins - set(late), late=late)}
        ref = {"s0": make_arm(os.path.join(self.root, "ref", "s0"), wins)}
        # 没有重跑目录：迟到的两集按缺失（不成功）
        r = cl.summarize(proto, ref, SEEDS20)
        self.assertEqual(r["arms"]["s0"]["proto"]["counts"]["missing"], 2)
        self.assertEqual(r["arms"]["s0"]["proto"]["detail"]["first_late_seeds"], sorted(late))
        # 重跑：一集干净成功，一集仍迟到
        make_retry(os.path.join(self.root, "proto", "s0-retry1"), late, wins={late[0]}, late=[late[1]])
        r = cl.summarize(proto, ref, SEEDS20)
        c = r["arms"]["s0"]["proto"]["counts"]
        self.assertEqual((c["success"], c["late"], c["missing"]), (len(wins - set(late)) + 1, 1, 0))
        self.assertEqual(r["arms"]["s0"]["proto"]["detail"]["how"][late[0]], "retry")
        self.assertFalse(r["total"]["complete"])
        # 迟到的那一集重跑也干净：完整
        make_retry(os.path.join(self.root, "proto", "s0-retry1"), late, wins=set(late))
        r = cl.summarize(proto, ref, SEEDS20)
        self.assertEqual(r["arms"]["s0"]["proto"]["counts"]["success"], len(wins - set(late)) + 2)
        self.assertTrue(r["total"]["complete"])

    def test_missing_arm_is_not_judged_and_cli_returns_2(self):
        wins = set(SEEDS20[:10])
        proto = {"s0": make_arm(os.path.join(self.root, "proto", "s0"), wins), "s1": os.path.join(self.root, "proto", "nope")}
        ref = {"s0": make_arm(os.path.join(self.root, "ref", "s0"), wins), "s1": make_arm(os.path.join(self.root, "ref", "s1"), wins)}
        r = cl.summarize(proto, ref, SEEDS20)
        self.assertFalse(r["total"]["all_arms_present"])
        self.assertTrue(r["verdict"].startswith("未下判"))
        args = []
        for n, d in proto.items():
            args += ["--proto", f"{n}={d}"]
        for n, d in ref.items():
            args += ["--ref", f"{n}={d}"]
        out_j = os.path.join(self.root, "c.json")
        rc = quiet(cl.main, args + ["--seeds", ",".join(map(str, SEEDS20)), "--out-json", out_j, "--out-md", os.path.join(self.root, "c.md")])
        self.assertEqual(rc, 2)
        with open(out_j, encoding="utf-8") as f:
            self.assertIn("arms", json.load(f))

    def test_cli_exit_codes(self):
        ref_w = {"s0": set(SEEDS20[:18]), "s1": set(SEEDS20[:16]), "s2": set(SEEDS20[:19])}
        for n, w in (("same", ref_w), ("bad", {"s0": set(SEEDS20[:8]), "s1": set(SEEDS20[:8]), "s2": set(SEEDS20[:8])})):
            proto, ref = {}, {}
            for k in ref_w:
                proto[k] = make_arm(os.path.join(self.root, n, "proto", k), w[k])
                ref[k] = make_arm(os.path.join(self.root, n, "ref", k), ref_w[k])
            args = [x for k in ref_w for x in ("--proto", f"{k}={proto[k]}", "--ref", f"{k}={ref[k]}")]
            rc = quiet(cl.main, args + ["--seeds", ",".join(map(str, SEEDS20))])
            self.assertEqual(rc, 0 if n == "same" else 1)

    def test_cg_average_cores(self):
        d = os.path.join(self.root, "x")
        os.makedirs(d, exist_ok=True)
        with open(d + ".cg", "w", encoding="utf-8") as f:
            f.write("start 100.0 1000000 5 0\nend 200.0 301000000 7 50000\n")
        g = cl.read_cg(d + ".cg")
        self.assertAlmostEqual(g["avg_cores"], 3.0)
        self.assertEqual(g["throttled"], 2)
        self.assertIsNone(cl.read_cg(os.path.join(self.root, "none.cg")))


def seed_metrics(val, test, nv=(0, 501, 40, 33), nt=(1, 739, 72, 72)):
    """合成 accept JSON 里一个种子的 val / test 指标；val、test 各给 (宏 F1, 类 1, 类 2, 类 3 召回)。"""
    def one(t, n):
        return {"macro_f1": t[0], "recall0": 0.0, "recall1": t[1], "recall2": t[2], "recall3": t[3], "n_true": list(n)}
    return {"val": one(val, nv), "test": one(test, nt)}


class TestDistributionCheck(unittest.TestCase):
    """事后补充读法：原型种子相对 C++ 全部种子的均值、标准差、范围、z。"""

    BASE = (0.40, 0.80, 0.30, 0.50)

    def acc(self, cpp_val, proto_val, test=None, **kw):
        """cpp_val / proto_val 是 val 元组的列表；test 缺省时 test 全取 BASE（每个种子相同）。"""
        t = test or self.BASE
        refs = {f"s{i}": seed_metrics(v, t, **kw) for i, v in enumerate(cpp_val[:3])}
        ctrl = {f"c{i + 3}": seed_metrics(v, t, **kw) for i, v in enumerate(cpp_val[3:])}
        out = {"refs": refs, "protos": {f"p{i}": seed_metrics(v, t, **kw) for i, v in enumerate(proto_val)}}
        if ctrl:
            out["controls"] = {"runs": ctrl}
        return out

    def row(self, res, split, metric):
        return [r for r in res["rows"] if r["split"] == split and r["metric"] == metric][0]

    def four_cpp(self):
        # val 宏 F1：0.40 0.42 0.38 0.40，均值 0.40，样本标准差 sqrt(0.0008 / 3) = 0.0163299
        return [(0.40, 0.80, 0.30, 0.50), (0.42, 0.82, 0.32, 0.54), (0.38, 0.78, 0.28, 0.46), (0.40, 0.80, 0.30, 0.50)]

    def test_numbers_by_hand(self):
        proto = [(0.41, 0.80, 0.30, 0.50), (0.43, 0.80, 0.30, 0.50), (0.45, 0.80, 0.30, 0.50)]
        res = dc.analyse(*dc.collect(self.acc(self.four_cpp(), proto)))
        r = self.row(res, "val", "macro_f1")
        self.assertEqual(r["cpp_n"], 4)
        self.assertAlmostEqual(r["cpp_mean"], 0.40)
        self.assertAlmostEqual(r["cpp_sd"], 0.0163299, places=6)
        self.assertAlmostEqual(r["cpp_min"], 0.38)
        self.assertAlmostEqual(r["cpp_max"], 0.42)
        self.assertAlmostEqual(r["proto_mean"], 0.43)
        # z = (0.43 - 0.40) / (0.0163299 / sqrt(3)) = 3.18198
        self.assertAlmostEqual(r["z"], 3.18198, places=4)
        self.assertEqual(r["in_range"], {"p0": True, "p1": False, "p2": False})
        r1 = self.row(res, "val", "recall1")                 # 原型三个种子都等于 C++ 均值：z = 0，三格都在范围内
        self.assertAlmostEqual(r1["z"], 0.0)
        self.assertEqual(list(r1["in_range"].values()), [True, True, True])
        # 8 个指标 × 3 个种子 = 24 格；val 宏 F1 里 2 格在范围外
        self.assertEqual((res["cells_in_range"], res["cells_total"]), (22, 24))
        self.assertAlmostEqual(res["max_abs_z"], 3.18198, places=4)

    def test_val_and_test_are_separate_rows(self):
        proto = [(0.40, 0.80, 0.30, 0.50)] * 3
        res = dc.analyse(*dc.collect(self.acc(self.four_cpp(), proto, test=(0.30, 0.70, 0.20, 0.10))))
        self.assertEqual(len(res["rows"]), 8)
        self.assertAlmostEqual(self.row(res, "test", "macro_f1")["cpp_mean"], 0.30)   # test 取的是另一组数
        self.assertAlmostEqual(self.row(res, "val", "macro_f1")["cpp_mean"], 0.40)

    def test_class_with_at_most_one_sample_is_not_reported(self):
        proto = [self.BASE] * 3
        res = dc.analyse(*dc.collect(self.acc(self.four_cpp(), proto, nv=(0, 501, 40, 1))))
        keys = {(r["split"], r["metric"]) for r in res["rows"]}
        self.assertNotIn(("val", "recall3"), keys)             # val 类 3 只有 1 个样本
        self.assertIn(("test", "recall3"), keys)
        self.assertEqual(len(res["rows"]), 7)
        self.assertEqual(res["cells_total"], 21)
        res2 = dc.analyse(*dc.collect(self.acc(self.four_cpp(), proto, nv=(0, 501, 40, 2))))     # 2 个样本：报
        self.assertIn(("val", "recall3"), {(r["split"], r["metric"]) for r in res2["rows"]})

    def test_range_edges_are_inclusive(self):
        proto = [(0.42, 0.80, 0.30, 0.50), (0.38, 0.80, 0.30, 0.50), (0.4201, 0.80, 0.30, 0.50)]
        r = self.row(dc.analyse(*dc.collect(self.acc(self.four_cpp(), proto))), "val", "macro_f1")
        self.assertEqual(r["in_range"], {"p0": True, "p1": True, "p2": False})

    def test_zero_sd_gives_no_z(self):
        same = [self.BASE] * 4
        res = dc.analyse(*dc.collect(self.acc(same, [self.BASE] * 3)))
        self.assertIsNone(self.row(res, "val", "macro_f1")["z"])
        self.assertIsNone(res["max_abs_z"])
        self.assertIn("无（C++ 标准差为 0）", dc.render_md(res))

    def test_controls_are_merged_and_too_few_seeds_rejected(self):
        cpp, protos = dc.collect(self.acc(self.four_cpp(), [self.BASE] * 3))
        self.assertEqual(list(cpp), ["s0", "s1", "s2", "c3"])
        self.assertEqual(list(protos), ["p0", "p1", "p2"])
        with self.assertRaises(ValueError):                    # 只有 2 个 C++ 种子
            dc.collect({"refs": {"s0": seed_metrics(self.BASE, self.BASE), "s1": seed_metrics(self.BASE, self.BASE)},
                        "protos": {"p0": seed_metrics(self.BASE, self.BASE)}})
        with self.assertRaises(ValueError):                    # 没有原型
            dc.collect({"refs": {f"s{i}": seed_metrics(self.BASE, self.BASE) for i in range(3)}, "protos": {}})
        with self.assertRaises(ValueError):                    # 缺 protos 段
            dc.collect({"refs": {}})

    def test_main_writes_files_and_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "accept.json")
            proto = [(0.41, 0.80, 0.30, 0.50)] * 3
            with open(src, "w", encoding="utf-8") as f:
                json.dump(self.acc(self.four_cpp(), proto), f)
            oj, om = os.path.join(d, "o.json"), os.path.join(d, "o.md")
            self.assertEqual(quiet(dc.main, ["--accept", src, "--out-json", oj, "--out-md", om]), 0)
            with open(oj, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["cells_total"], 24)
            with open(om, encoding="utf-8") as f:
                md = f.read()
            self.assertIn("事后分析，不是卡的判据", md)
            self.assertIn("| val | macro_f1 |", md)
            self.assertEqual(quiet(dc.main, ["--accept", os.path.join(d, "none.json")]), 2)       # 文件不存在
            bad = os.path.join(d, "bad.json")
            with open(bad, "w", encoding="utf-8") as f:
                json.dump({"refs": {}, "protos": {}}, f)
            self.assertEqual(quiet(dc.main, ["--accept", bad]), 2)                                # 种子不够


@unittest.skipIf(pd is None, "没有 torch")
class TestProbeHelpers(unittest.TestCase):
    def test_first_exceed(self):
        d = [0, 2e-6, 3e-5, 3e-5, 3e-4, 3e-3, 3e-2]
        self.assertEqual(pd.first_exceed(d), {"1e-05": 3, "0.0001": 5, "0.001": 6, "0.01": 7})
        self.assertEqual(pd.first_exceed([0.0] * 5), {"1e-05": None, "0.0001": None, "0.001": None, "0.01": None})


if __name__ == "__main__":
    unittest.main()
