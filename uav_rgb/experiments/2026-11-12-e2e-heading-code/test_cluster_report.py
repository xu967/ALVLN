#!/usr/bin/env python3
"""cluster_report.py 的单测（全部使用合成桩数据，不读任何真实闭环评测目录，不起 Gazebo，不用 GPU）。

覆盖：
  1. 干净数据（含 R6 迟到重跑干净通过）上与冻结汇总器 summarize_e2e.py 的成功数、越界数、M1–M3 判决逐项一致；
  2. 存在非迟到崩溃集（不在 per_episode、缺 episode_manifest.json，含 failed 缺 seed、无 seed 键报错与带 seed 三种形态）时分母恒为 60，按预注册同阈值重算并列出原因；
  3. 20 个簇下 7 个几何全同向聚类 p = 0.0156（< 0.05）、8 个几何全同向 p < 0.01；
  4. 逐集 15:0 而差集中在 5 个几何（每几何 +3）时，落在「逐集检验下成立，按几何聚类不显著，效应集中在少数几何」；以及证伪、不判分支；
  5. M2 越界集数对等剔除任一侧基础设施失败单元（重跑仍迟到、rgb_ts 相邻差 > 1.1 s 停顿步、采集崩溃；恰好 1.1 s 不剔），并标注读不到逐步记录的集；
  6. 20 个 seed 与 20 个 geometry_gid 不一一对应或缺 geometry_gid 时报错退出（不猜）；
  7. 未跑完的臂标「未完成」，不出判决；命令行默认输出到 <OUT>/cluster/ 与显式 --out-dir。

运行（仓库根）：nice -n 10 .venv-tools/bin/python experiments/2026-11-12-e2e-heading-code/test_cluster_report.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))

import cluster_report as cr  # noqa: E402
import e2e_arms as ea  # noqa: E402
import paired_stats as ps  # noqa: E402
import split_manifest as sm  # noqa: E402
import summarize_e2e as se  # noqa: E402

STUB = os.path.join(HERE, "stub_eval_e2e.py")
SEEDS_CSV = ",".join(str(s) for s in se.FROZEN_SEEDS)
REAL_GID = {s: sm.group_id_for_task(s, "base") for s in se.FROZEN_SEEDS}
NS = 1_000_000_000


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(json.dumps(r) + "\n" for r in rows))


def enrich_arm_dir(arm_dir, gid_map=None, step_period_ns=900_000_000):
    """给 stub_eval_e2e.py 产出的评测臂目录补上真实 eval_closed_loop/collect 写出的 geometry_gid 与每步 rgb_ts。"""
    gmap = REAL_GID if gid_map is None else gid_map
    ev_path = os.path.join(arm_dir, "eval.json")
    ev = read_json(ev_path)
    for entry in ev.get("per_episode", []):
        sd = int(entry["seed"])
        if sd in gmap and gmap[sd] is not None:
            entry["geometry_gid"] = gmap[sd]
    write_json(ev_path, ev)
    eps_root = os.path.join(arm_dir, "episodes")
    if not os.path.isdir(eps_root):
        return
    for name in sorted(os.listdir(eps_root)):
        if not name.startswith("seed_"):
            continue
        sd = int(name.split("_", 1)[1])
        ep_dir = os.path.join(eps_root, name)
        man_path = os.path.join(ep_dir, "episode_manifest.json")
        if os.path.isfile(man_path) and sd in gmap and gmap[sd] is not None:
            man = read_json(man_path)
            man["geometry_gid"] = gmap[sd]
            write_json(man_path, man)
        act_path = os.path.join(ep_dir, "actions.jsonl")
        if os.path.isfile(act_path) and step_period_ns is not None:
            rows = read_jsonl(act_path)
            for idx, r in enumerate(rows):
                r["rgb_ts"] = 10 * NS + idx * step_period_ns
            write_jsonl(act_path, rows)


def run_stub(out, arm, train_seed, env=None, seeds=SEEDS_CSV, episodes=20, retry=False, gid_map=None):
    name = f"{ea.NAME[arm]}-s{train_seed}" + ("-retry1" if retry else "")
    arm_dir = os.path.join(out, name)
    cmd = [sys.executable, STUB, "--policy", "student", "--ckpt", ea.ckpt_path(out, arm, train_seed), "--infer-bin", "x"]
    if ea.IS_CODE[arm]:
        cmd += ["--student-cond", "heading-code"]
    cmd += [
        "--task", "base", "--steps", "80", "--split", "test", "--seeds", seeds,
        "--corridor-half-width", "3.0", "--action-latency", "0.20", "--episodes", str(episodes),
        "--sim-timeout", "120.0", "--episode-timeout", "1500", "--dynamics", "cf_sitl",
        "--vehicle", "e1_mini", "--hold", "1.0", "--window-from", "observation", "--out", arm_dir,
    ]
    full_env = {**os.environ, "STUB_SLEEP": "0", **(env or {})}
    res = subprocess.run(cmd, env=full_env, capture_output=True, text=True)
    enrich_arm_dir(arm_dir, gid_map=gid_map)
    return res


def make_stub_world(out, env=None, gid_map=None):
    for a in ea.ARMS:
        for s in ea.SEEDS_TRAIN:
            run_stub(out, a, s, env=env, gid_map=gid_map)


def write_custom_world(out, outcome_fn, gid_map=None):
    """按给定的逐单元结局构造 12 个评测臂目录：outcome_fn(arm, train_seed, eval_seed) -> 'success' | 'collision' | 'corridor' | 'not_reached'。"""
    gmap = REAL_GID if gid_map is None else gid_map
    for a in ea.ARMS:
        for s in ea.SEEDS_TRAIN:
            arm_dir = os.path.join(out, f"{ea.NAME[a]}-s{s}")
            os.makedirs(arm_dir, exist_ok=True)
            per_ep = []
            for seed in se.FROZEN_SEEDS:
                kind = outcome_fn(a, s, seed)
                ok = kind == "success"
                ep_dir = os.path.join(arm_dir, "episodes", f"seed_{seed}")
                os.makedirs(ep_dir, exist_ok=True)
                actions, oracle = [], []
                for i in range(6):
                    collide = (kind == "collision" and i == 3)
                    row = {
                        "step": i, "action_executed": 1, "action_expert": 1, "collision": collide,
                        "dispatch_lag_s": 0.2, "late_dispatch": False, "src": None,
                        "action_start_ns": 10 * NS + i * 900_000_000,
                        "rgb_ts": 10 * NS + i * 900_000_000,
                    }
                    if ea.IS_CODE[a]:
                        row["cond_code"] = 0
                    actions.append(row)
                    x = 6.5 * (i + 1) / 6 if ok else (2.0 * (i + 1) / 6 if kind == "not_reached" else 3.0 * (i + 1) / 6)
                    y = 3.5 if (kind == "corridor" and i >= 3) else 0.0
                    oracle.append({
                        "step": i, "action_start_ns": 10 * NS + i * 900_000_000,
                        "action_end_ns": 10 * NS + i * 900_000_000 + 1_005_000_000,
                        "pose_gt": {"x": x, "y": y, "z": 2.0}, "speed_mps": 0.3,
                    })
                    if collide:
                        break
                write_jsonl(os.path.join(ep_dir, "actions.jsonl"), actions)
                write_jsonl(os.path.join(ep_dir, "oracle.jsonl"), oracle)
                man = {"goal_line_x": 6.0, "action_latency_s": 0.2, "late_dispatch_steps": 0}
                if gmap.get(seed) is not None:
                    man["geometry_gid"] = gmap[seed]
                write_json(os.path.join(ep_dir, "episode_manifest.json"), man)
                entry = {
                    "seed": seed, "success": ok, "steps": len(actions),
                    "episode_dir": os.path.join("episodes", f"seed_{seed}"),
                    "goal_x": 6.0, "collision": kind == "collision",
                    "corridor_violation": kind == "corridor",
                    "dispatch_lag_s_median": 0.2, "dispatch_lag_s_max": 0.2, "late_dispatch_steps": 0,
                }
                if gmap.get(seed) is not None:
                    entry["geometry_gid"] = gmap[seed]
                per_ep.append(entry)
            rep = {
                "episodes_requested": 20, "episodes_done": 20, "failed_count": 0, "failed": [],
                "partial": False, "skipped_count": 0, "per_episode": per_ep,
                "corridor_half_width": 3.0, "metric_version": "sim09.v2.coast1s.corridor3m",
                "action_latency_s": 0.2, "late_dispatch_episodes": [],
            }
            write_json(os.path.join(arm_dir, "eval.json"), rep)


class ClusterReportTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(self._tmp.cleanup)
        self.out = self._tmp.name

    def test_clean_stub_matches_frozen_summarizer(self):
        """干净数据（含首次迟到但 retry1 干净完成的集）上，与冻结汇总器的成功数、越界数、M1–M3 判决逐项一致。"""
        late = {"a1-code-demo-s0": [113, 172], "a2-code-dagger-s2": [1]}
        env = {"STUB_P": "0.50,0.80,0.95,0.20", "STUB_FAIL_MIX": "0.1,0.8,0.1", "STUB_LATE": json.dumps(late)}
        make_stub_world(self.out, env=env)
        run_stub(self.out, "a1", 0, env=env, seeds="113,172", episodes=2, retry=True)
        run_stub(self.out, "a2", 2, env=env, seeds="1", episodes=1, retry=True)

        _, frozen_v, _ = se.build(self.out)
        md, rep = cr.build_cluster_report(self.out, n_boot=400, boot_seed=0)

        self.assertTrue(frozen_v["complete"] and rep["complete"])
        self.assertFalse(rep["recomputed"])
        self.assertEqual(rep["recomputed_reasons"], [])
        self.assertEqual(rep["unreadable_step_logs"], [])
        for a in ea.ARMS:
            self.assertEqual(rep["arms"][a]["n"], 60)
            self.assertEqual(rep["arms"][a]["success"], frozen_v["success"][a], a)
            self.assertEqual(rep["arms"][a]["corridor"], frozen_v["classes"][a]["corridor"], a)
        for key in ("m1", "m2", "m3"):
            self.assertEqual(rep[key]["verdict"], frozen_v[key]["verdict"], key)
            self.assertEqual(rep[key]["frozen_verdict"], frozen_v[key]["verdict"], key)
            self.assertEqual(rep[key]["n"], 60)
            self.assertAlmostEqual(rep[key]["episode_mcnemar_p"], frozen_v[key]["p"], places=12)
        self.assertEqual(rep["m2"]["excluded_count"], 0)
        self.assertEqual((rep["m2"]["clean_corridor_a0"], rep["m2"]["clean_corridor_a1"]),
                         (frozen_v["m2"]["a0_corridor"], frozen_v["m2"]["a1_corridor"]))
        for phrase in ("判决以冻结的汇总器", "补报", "样本内", "另外 40 个", "样本外复核"):
            self.assertIn(phrase, md)

    def test_crash_without_manifest_keeps_denom_60_and_recomputes(self):
        """崩溃集（不在 per_episode、没有 episode_manifest.json）：分母恒为 60，重算判决并列出具体集与原因。"""
        def outcome(a, s, seed):
            idx = se.FROZEN_SEEDS.index(seed)
            if a == "a2":
                return "success" if idx < 15 else "collision"
            if a == "a3":
                return "success" if idx < 8 or (idx == 8 and s == 0) else "collision"
            if a == "a0":
                return "success" if idx < 10 else ("corridor" if idx < 14 else "collision")
            return "success" if idx < 12 else ("corridor" if (idx == 12 and s < 2) else "collision")

        write_custom_world(self.out, outcome)

        # 形态 A1：failed_count=1、failed=[]（未写出该崩溃 seed）、无 episode_manifest.json -> 冻结汇总器漏计为 n=59 并判「未完成，不判」；
        # cluster_report 补齐 60 分母并给出重算判决。
        a2_s0 = os.path.join(self.out, "a2-code-dagger-s0")
        shutil.rmtree(os.path.join(a2_s0, "episodes", "seed_1654"))
        ev_path = os.path.join(a2_s0, "eval.json")
        ev = read_json(ev_path)
        ev["per_episode"] = [x for x in ev["per_episode"] if x["seed"] != 1654]
        ev["episodes_done"] = 19
        ev["failed_count"] = 1
        ev["failed"] = []
        write_json(ev_path, ev)

        _, frozen_v, _ = se.build(self.out)
        self.assertFalse(frozen_v["complete"])
        self.assertEqual(frozen_v["m1"]["verdict"], se.UNDONE)
        self.assertEqual(frozen_v["m1"]["n"], 59)

        md, rep = cr.build_cluster_report(self.out, n_boot=300, boot_seed=0)
        self.assertTrue(rep["complete"] and rep["recomputed"])
        self.assertEqual(rep["arms"]["a2"]["n"], 60)
        self.assertEqual(rep["m1"]["n"], 60)
        self.assertEqual(rep["m1"]["verdict"], se.PRED)
        self.assertEqual(rep["m1"]["frozen_verdict"], se.UNDONE)
        self.assertTrue(any("a2-code-dagger-s0" in r and "1654" in r for r in rep["recomputed_reasons"]), rep["recomputed_reasons"])
        self.assertIn("重算", md)
        self.assertIn("1654", md)

        # 形态 A2：failed 里含缺 seed 键的条目（冻结汇总器会抛 KeyError）-> cluster_report 照常完成并记录冻结器异常。
        ev["failed"] = [{"rc": -9, "error": "gz crash before seed recorded"}]
        write_json(ev_path, ev)
        _, rep_err = cr.build_cluster_report(self.out, n_boot=100, boot_seed=0)
        self.assertTrue(rep_err["complete"] and rep_err["recomputed"])
        self.assertEqual(rep_err["m1"]["n"], 60)
        self.assertEqual(rep_err["m1"]["verdict"], se.PRED)
        self.assertIn("KeyError", rep_err["m1"]["frozen_verdict"])

        # 形态 B：failed 里带 {"seed": 1113, "rc": 1}、无 episode_manifest.json（发生在 A0 上）：分母仍是 60，M2 对等剔除该单元。
        ev["failed"] = [{"seed": 1654, "rc": 1, "error": "collect crash"}]
        write_json(ev_path, ev)
        a0_s1 = os.path.join(self.out, "a0-img-demo-s1")
        shutil.rmtree(os.path.join(a0_s1, "episodes", "seed_1113"))
        ev0_path = os.path.join(a0_s1, "eval.json")
        ev0 = read_json(ev0_path)
        ev0["per_episode"] = [x for x in ev0["per_episode"] if x["seed"] != 1113]
        ev0["episodes_done"] = 19
        ev0["failed_count"] = 1
        ev0["failed"] = [{"seed": 1113, "rc": 1, "error": "collect crash"}]
        write_json(ev0_path, ev0)

        _, rep2 = cr.build_cluster_report(self.out, n_boot=200, boot_seed=0)
        self.assertTrue(rep2["complete"] and rep2["recomputed"])
        self.assertEqual((rep2["arms"]["a0"]["n"], rep2["arms"]["a2"]["n"]), (60, 60))
        self.assertTrue(any("1113" in r for r in rep2["recomputed_reasons"]))
        self.assertEqual(rep2["m2"]["excluded_count"], 1)

    def test_discrete_signflip_7_and_8_geometries(self):
        """20 个簇时：7 个非零差几何全同向 p = 2/2^7 = 0.0156（< 0.05 显著）；8 个全同向 p = 2/2^8 = 0.0078（< 0.01）。"""
        def outcome_7(a, s, seed):
            idx = se.FROZEN_SEEDS.index(seed)
            if a == "a2" and (idx < 5 or (idx in (5, 6) and s == 0)):
                return "success"
            return "collision"

        write_custom_world(self.out, outcome_7)
        md7, rep7 = cr.build_cluster_report(self.out, n_boot=200, boot_seed=0)
        m1_7 = rep7["m1"]
        self.assertEqual((m1_7["diff"], m1_7["b"], m1_7["c"]), (17, 17, 0))
        self.assertEqual((m1_7["cluster"]["n_clusters"], m1_7["cluster"]["n_nonzero"], m1_7["cluster"]["n_same_dir"]), (20, 7, 7))
        self.assertAlmostEqual(m1_7["cluster"]["p_value"], 2 / 128, places=12)
        self.assertEqual(round(m1_7["cluster"]["p_value"], 4), 0.0156)
        self.assertEqual(m1_7["reading_code"], "established_cluster_sig")
        self.assertEqual(m1_7["reading_text"], "成立，按几何聚类也显著")
        self.assertIn("0.0156", md7)

        def outcome_8(a, s, seed):
            idx = se.FROZEN_SEEDS.index(seed)
            if a == "a2" and idx < 8 and s in (0, 1):
                return "success"
            return "collision"

        write_custom_world(self.out, outcome_8)
        _, rep8 = cr.build_cluster_report(self.out, n_boot=200, boot_seed=0)
        m1_8 = rep8["m1"]
        self.assertEqual((m1_8["cluster"]["n_nonzero"], m1_8["cluster"]["n_same_dir"]), (8, 8))
        self.assertAlmostEqual(m1_8["cluster"]["p_value"], 2 / 256, places=12)
        self.assertLess(m1_8["cluster"]["p_value"], 0.01)
        self.assertEqual(m1_8["reading_text"], "成立，按几何聚类也显著")

    def test_15_to_0_in_5_geometries_and_other_readings(self):
        """逐集 15:0 集中在 5 个几何（每几何 +3）：预注册判预测成立，但聚类 p = 0.0625 >= 0.05，落在「成立而聚类不显著」写法；
        同时校验「证伪」与「不判」两种写法。"""
        def outcome_5(a, s, seed):
            idx = se.FROZEN_SEEDS.index(seed)
            if a == "a2" and idx < 5:
                return "success"
            if a == "a0" and idx < 5:
                return "success"
            if a == "a0" and 5 <= idx < 9:
                return "corridor"
            if a == "a1" and idx < 5:
                return "success"
            if a == "a1" and (5 <= idx < 7 or (idx == 7 and s == 0)):
                return "corridor"
            return "collision"

        write_custom_world(self.out, outcome_5)
        md, rep = cr.build_cluster_report(self.out, n_boot=200, boot_seed=0)
        m1 = rep["m1"]
        self.assertEqual((m1["diff"], m1["b"], m1["c"], m1["verdict"]), (15, 15, 0, se.PRED))
        self.assertLess(m1["episode_mcnemar_p"], 0.01)
        self.assertEqual((m1["cluster"]["n_nonzero"], m1["cluster"]["n_same_dir"]), (5, 5))
        self.assertAlmostEqual(m1["cluster"]["p_value"], 2 / 32, places=12)
        self.assertEqual(m1["reading_code"], "established_cluster_ns")
        self.assertEqual(m1["reading_text"], "逐集检验下成立，按几何聚类不显著，效应集中在少数几何")
        self.assertIn("逐集检验下成立，按几何聚类不显著，效应集中在少数几何", md)

        self.assertEqual(rep["m2"]["reading_code"], "inconclusive")
        self.assertEqual(rep["m2"]["reading_text"], se.MID)
        self.assertEqual(rep["m3"]["reading_code"], "falsified")
        self.assertEqual(rep["m3"]["reading_text"], se.FALS)

    def test_m2_exclusions_late_stall_crash_and_unreadable_logs(self):
        """M2 只在两侧都干净完成的单元上比：重跑仍迟到、停顿步（rgb_ts 相邻差 > 1.1 s；恰好 1.1 s 不算）、采集崩溃两侧一起剔除；
        两种算法都报、判决按剔除后的；越界簇差符号翻转 p 正确；读不到逐步记录的集照实标出。"""
        s_u1, s_u2, s_u3 = se.FROZEN_SEEDS[1], se.FROZEN_SEEDS[3], se.FROZEN_SEEDS[4]
        s_exact_11 = se.FROZEN_SEEDS[2]

        # A0 成功 29 集，A1 成功 33 集（succ_a1 >= succ_a0 恒满足）。
        # 57 个干净单元上：A0 越界 10 集（idx 10..12 共 9 集 + (s=0, idx=13) 1 集），A1 越界 5 集（idx 10 共 3 集 + idx 11 的 s=0,1 共 2 集） -> 2*5 <= 10（预测成立）。
        # 3 个被剔除单元上：A1 在 u2、u3 越界 2 次，故未剔除时 A1 越界 7 集（2*7 > 10 -> 介于之间，不判），剔除后 5 集 -> 预测成立。
        def outcome_m2(a, s, seed):
            idx = se.FROZEN_SEEDS.index(seed)
            if a == "a1" and (s, seed) in ((1, s_u2), (2, s_u3)):
                return "corridor"
            if a == "a0":
                if idx < 10:
                    return "success"
                if 10 <= idx <= 12 or (idx == 13 and s == 0):
                    return "corridor"
                return "collision"
            if a == "a1":
                if idx < 10 or idx in (14, 15):
                    return "success"
                if idx == 10 or (idx == 11 and s in (0, 1)):
                    return "corridor"
                return "collision"
            return "success" if idx < 10 else "collision"

        write_custom_world(self.out, outcome_m2)

        # 1) u1: a1-s0 seed=s_u1 首次迟到 + retry1 仍迟到
        a1_s0 = os.path.join(self.out, "a1-code-demo-s0")
        ev = read_json(os.path.join(a1_s0, "eval.json"))
        ev["per_episode"] = [x for x in ev["per_episode"] if x["seed"] != s_u1]
        ev["episodes_done"] = 19
        ev["failed_count"] = 1
        ev["failed"] = [{"seed": s_u1, "rc": 0, "error": "late_dispatch: 1", "late_dispatch_steps": 1, "late_step_ids": [0]}]
        ev["late_dispatch_episodes"] = [{"seed": s_u1, "late_dispatch_steps": 1, "late_step_ids": [0]}]
        write_json(os.path.join(a1_s0, "eval.json"), ev)
        rdir = a1_s0 + "-retry1"
        os.makedirs(os.path.join(rdir, "episodes", f"seed_{s_u1}"), exist_ok=True)
        write_json(os.path.join(rdir, "eval.json"), {
            "episodes_requested": 1, "episodes_done": 0, "failed_count": 1,
            "failed": [{"seed": s_u1, "rc": 0, "error": "late_dispatch: 1", "late_dispatch_steps": 1, "late_step_ids": [2]}],
            "partial": False, "skipped_count": 0, "per_episode": [],
            "action_latency_s": 0.2, "late_dispatch_episodes": [{"seed": s_u1, "late_dispatch_steps": 1, "late_step_ids": [2]}],
        })

        # 2) u2: a0-s1 seed=s_u2 第 2 步制造 1.6 s 停顿；同时在 seed=s_exact_11 制造恰好 1.1 s（不该判停顿）
        for target_seed, gap_ns in ((s_u2, 1_600_000_000), (s_exact_11, 1_100_000_000)):
            ap = os.path.join(self.out, "a0-img-demo-s1", "episodes", f"seed_{target_seed}", "actions.jsonl")
            rows = read_jsonl(ap)
            ts = 10 * NS
            for idx, r in enumerate(rows):
                if idx > 0:
                    ts += gap_ns if idx == 2 else 900_000_000
                r["rgb_ts"] = ts
            write_jsonl(ap, rows)

        # 3) u3: a0-s2 seed=s_u3 采集进程崩溃（无 episode_manifest.json）
        a0_s2 = os.path.join(self.out, "a0-img-demo-s2")
        shutil.rmtree(os.path.join(a0_s2, "episodes", f"seed_{s_u3}"))
        ev02 = read_json(os.path.join(a0_s2, "eval.json"))
        ev02["per_episode"] = [x for x in ev02["per_episode"] if x["seed"] != s_u3]
        ev02["episodes_done"] = 19
        ev02["failed_count"] = 1
        ev02["failed"] = [{"seed": s_u3, "rc": 139, "error": "segfault"}]
        write_json(os.path.join(a0_s2, "eval.json"), ev02)

        # 4) 在 a2-s0 seed=1 删掉 rgb_ts，验证「读不到逐步记录的集照实标出」
        ap_unreadable = os.path.join(self.out, "a2-code-dagger-s0", "episodes", "seed_1", "actions.jsonl")
        rows_u = read_jsonl(ap_unreadable)
        for r in rows_u:
            r.pop("rgb_ts", None)
        write_jsonl(ap_unreadable, rows_u)

        md, rep = cr.build_cluster_report(self.out, n_boot=300, boot_seed=0)
        m2 = rep["m2"]
        self.assertEqual(m2["excluded_count"], 3)
        self.assertEqual(m2["clean_n"], 57)
        excl_pairs = [(x["train_seed"], x["eval_seed"]) for x in m2["excluded_units"]]
        self.assertEqual(sorted(excl_pairs), sorted([(0, s_u1), (1, s_u2), (2, s_u3)]))
        self.assertEqual((m2["raw_corridor_a0"], m2["raw_corridor_a1"], m2["raw_verdict"]), (10, 7, se.MID))
        self.assertEqual((m2["clean_corridor_a0"], m2["clean_corridor_a1"], m2["verdict"]), (10, 5, se.PRED))
        self.assertEqual(m2["corridor_cluster"]["n_nonzero"], 3)
        self.assertAlmostEqual(m2["corridor_cluster"]["p_value"], ps.signflip_test([1, 3, 1])["p_value"], places=12)
        self.assertEqual(len(rep["unreadable_step_logs"]), 1)
        self.assertEqual((rep["unreadable_step_logs"][0]["arm"], rep["unreadable_step_logs"][0]["train_seed"], rep["unreadable_step_logs"][0]["eval_seed"]),
                         ("a2", 0, 1))
        self.assertIn("a2-code-dagger-s0", md)
        self.assertIn(str(s_u2), md)

    def test_seed_gid_mismatch_raises_error(self):
        """seed 与 geometry_gid 不一一对应（两个 seed 共用 gid、同一 seed 跨臂 gid 不同、缺 geometry_gid）时报错退出，不猜。"""
        dup_map = dict(REAL_GID)
        dup_map[se.FROZEN_SEEDS[1]] = dup_map[se.FROZEN_SEEDS[0]]
        write_custom_world(self.out, lambda *_a: "success", gid_map=dup_map)
        with self.assertRaises(cr.ClusterReportError):
            cr.build_cluster_report(self.out, n_boot=20)
        rc = subprocess.run([sys.executable, os.path.join(HERE, "cluster_report.py"), self.out], capture_output=True, text=True).returncode
        self.assertNotEqual(rc, 0)

        write_custom_world(self.out, lambda *_a: "success")
        p = os.path.join(self.out, "a1-code-demo-s0", "eval.json")
        ev = read_json(p)
        ev["per_episode"][0]["geometry_gid"] = "other_gid_conflict"
        mp = os.path.join(self.out, "a1-code-demo-s0", "episodes", f"seed_{ev['per_episode'][0]['seed']}", "episode_manifest.json")
        man = read_json(mp)
        man["geometry_gid"] = "other_gid_conflict"
        write_json(p, ev)
        write_json(mp, man)
        with self.assertRaises(cr.ClusterReportError):
            cr.build_cluster_report(self.out, n_boot=20)

        no_gid_map = dict(REAL_GID)
        no_gid_map[se.FROZEN_SEEDS[0]] = None
        write_custom_world(self.out, lambda *_a: "success", gid_map=no_gid_map)
        with self.assertRaises(cr.ClusterReportError):
            cr.build_cluster_report(self.out, n_boot=20)

    def test_incomplete_arms_do_not_judge_and_cli_outputs(self):
        """未跑完的臂标「未完成」且不出判决；完整跑完时 CLI 默认写 <OUT>/cluster/cluster_report.{md,json} 并支持 --out-dir。"""
        write_custom_world(self.out, lambda a, *_r: "success" if a == "a2" else "collision")
        r_ok = subprocess.run([sys.executable, os.path.join(HERE, "cluster_report.py"), self.out, "--n-boot", "100"],
                              capture_output=True, text=True)
        self.assertEqual(r_ok.returncode, 0, r_ok.stderr)
        self.assertTrue(os.path.isfile(os.path.join(self.out, "cluster", "cluster_report.md")))
        self.assertTrue(os.path.isfile(os.path.join(self.out, "cluster", "cluster_report.json")))

        alt = os.path.join(self.out, "custom_out")
        r_alt = subprocess.run([sys.executable, os.path.join(HERE, "cluster_report.py"), self.out, "--out-dir", alt, "--n-boot", "100"],
                               capture_output=True, text=True)
        self.assertEqual(r_alt.returncode, 0, r_alt.stderr)
        self.assertTrue(os.path.isfile(os.path.join(alt, "cluster_report.json")))

        shutil.rmtree(os.path.join(self.out, "a3-img-dagger-s2"))
        md, rep = cr.build_cluster_report(self.out, n_boot=50)
        self.assertFalse(rep["complete"])
        for k in ("m1", "m2", "m3"):
            self.assertEqual(rep[k]["verdict"], se.UNDONE)
            self.assertEqual(rep[k]["reading_code"], "undone")
        self.assertIn("`a3-img-dagger-s2`：未完成", md)
        self.assertIn("未完成，不判", md)

    def test_pairing_aligns_by_train_seed_and_eval_seed(self):
        """M-4 第 3 点：配对单元严格按 (train_seed, eval_seed) 对齐，簇内先加总 3 个训练种子的成败差再做符号翻转。
        每个几何上 A2 只在 s=0 成功、A3 只在 s=1 成功：逐集配对 b=20, c=20（若错配 s=0 对 s=1 会变成 b=0, c=0），几何簇差全为 0。"""
        def outcome_align(a, s, _seed):
            if a == "a2":
                return "success" if s == 0 else "collision"
            if a == "a3":
                return "success" if s == 1 else "collision"
            return "collision"

        write_custom_world(self.out, outcome_align)
        _, rep = cr.build_cluster_report(self.out, n_boot=100, boot_seed=0)
        m1 = rep["m1"]
        self.assertEqual((m1["b"], m1["c"], m1["diff"]), (20, 20, 0))
        self.assertEqual((m1["cluster"]["n_nonzero"], m1["cluster"]["p_value"], m1["cluster"]["method"]), (0, 1.0, "none"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
