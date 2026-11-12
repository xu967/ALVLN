#!/usr/bin/env python3
"""dagger_gz_pipeline 的纯 CPU 回归：round-0 开关读取与缺字段退出、seed 按 gid 排除、dry-run 命令行逐字、
more_demos 集数、基线复用四条件、续跑判定、专家不一致率、失败时 status/DONE、交付权重（calswa / best）。不起 gz、不读真实数据。

运行（仓库根）：.venv-tools/bin/python tools/test_dagger_gz_pipeline.py
"""
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import dagger_gz_pipeline as dg  # noqa: E402
import split_manifest as sm  # noqa: E402

ROUND0_MANIFEST = {   # trunc/col 的真实字段子集（lib v1 采、离线截断；没有 teacher_tiebreak）
    "schema": "navlab.episode.v1", "teacher": "plan_action", "planner": "astar",
    "hold_s": 1.0, "window_from": "observation", "dynamics": "cf_sitl", "vehicle": "e1_mini",
    "dynamics_profile": "cf_sitl_v1:bd0dcd43", "stopped_on_goal": False, "truncated_at_success_line": True,
    "num_steps": 30, "requested_steps": 40, "geometry_version": "uavrgb.geom.v5",
    "validity": {"camera_verified": True, "physics_verified": True, "episode_executed": True},
}


def make_round0(root, n=3, **override):
    """伪 round-0 集合：train/<seed>/episode_manifest.json（+ 空 actions.jsonl），字段按 override 改。"""
    for i in range(n):
        d = os.path.join(root, "train" if i < n - 1 else "val", str(3100 + i))
        os.makedirs(d)
        man = dict(ROUND0_MANIFEST, episode_id=3100 + i, scene_id=3100 + i, seed=3100 + i,
                   split=os.path.basename(os.path.dirname(d)), **override)
        with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(man, f)
        open(os.path.join(d, "actions.jsonl"), "w").close()
    return root


class Round0Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_dagger_gz_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_settings_from_episode_manifests(self):
        col = make_round0(os.path.join(self.tmp, "col"))
        s = dg.round0_settings(col)
        self.assertEqual((s["hold"], s["window_from"], s["dynamics"], s["vehicle"]), (1.0, "observation", "cf_sitl", "e1_mini"))
        self.assertEqual(s["tiebreak"], "legacy")             # 旧 manifest 没有 teacher_tiebreak = legacy
        self.assertTrue(s["stop_at_goal"])                    # 离线截断 ⇒ 新轮次开 --stop-at-goal，分布对齐
        self.assertEqual(s["dynamics_profiles"], ["cf_sitl_v1:bd0dcd43"])
        self.assertEqual((s["episodes"], s["steps"]), (3, 90))

    def test_collection_manifest_wins_when_present(self):
        col = make_round0(os.path.join(self.tmp, "col"))
        with open(os.path.join(col, "collection_manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"hold_s": 1.0, "window_from": "observation", "dynamics": "cf_sitl", "vehicle": "e1_mini",
                       "stop_at_goal": False, "teacher_tiebreak": "visible"}, f)
        s = dg.round0_settings(col)
        self.assertEqual((s["stop_at_goal"], s["tiebreak"]), (False, "visible"))

    def test_missing_or_mixed_field_exits(self):
        for override, word in (({"hold_s": None}, "hold_s"), ({"teacher": "forward"}, "teacher"),
                               ({"window_from": "dispatch"}, "window_from")):
            col = make_round0(os.path.join(self.tmp, "col_" + word), 2)
            if word == "window_from":   # 只改一集：两种值混在一起
                p = os.path.join(col, "train", "3100", "episode_manifest.json")
                with open(p, encoding="utf-8") as f:
                    man = json.load(f)
                man.update(override)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(man, f)
            else:
                shutil.rmtree(col)
                make_round0(col, 2, **override)
            with self.assertRaises(RuntimeError, msg=word) as cm:
                dg.round0_settings(col)
            self.assertIn(word, str(cm.exception))

    def test_no_episodes_exits(self):
        os.makedirs(os.path.join(self.tmp, "empty", "train"))
        with self.assertRaises(RuntimeError):
            dg.round0_settings(os.path.join(self.tmp, "empty"))


class SeedAndCommandTests(unittest.TestCase):
    SETTINGS = dict(hold=1.0, window_from="observation", dynamics="cf_sitl", vehicle="e1_mini",
                    tiebreak="visible", stop_at_goal=True)

    def test_round_seeds_skip_test_geometries_and_refill(self):
        excl = {sm.group_id_for_task(5100, "base"), sm.group_id_for_task(5102, "base")}
        seeds = dg.pick_round_seeds(5100, 4, excl)
        self.assertEqual(len(seeds), 4)
        self.assertNotIn(5100, seeds)
        self.assertNotIn(5102, seeds)
        self.assertEqual(seeds[0], 5101)
        self.assertTrue(all(sm.group_id_for_task(s, "base") not in excl for s in seeds))
        self.assertEqual(seeds, sorted(seeds))

    def test_test_gids_cover_all_eval_seeds(self):
        gids = dg.test_gids("1,113,122")
        self.assertEqual(gids, {sm.group_id_for_task(s, "base") for s in (1, 113, 122)})

    def test_seeds60_is_e1_rule_list(self):
        # 用户 2026-11-09 定的 E1 口径：eval60 = eval_seeds.EVAL_SEEDS_BASE_60，不再按采集集合排除（那样只剩 24 个）
        import eval_seeds as es
        out = tempfile.mkdtemp(prefix="test_dagger_gz_")
        try:
            self.assertEqual(dg.seeds60(out), es.csv(es.EVAL_SEEDS_BASE_60))
            with open(os.path.join(out, "eval_seeds60.json"), encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["seeds"], list(es.EVAL_SEEDS_BASE_60))
            self.assertEqual(doc["count"], 60)
        finally:
            shutil.rmtree(out, ignore_errors=True)

    def test_student_collect_command(self):
        cmd = dg.collect_cmd("/out/round_1/collect", [5101, 5103], self.SETTINGS, instances=2,
                             steps=12, policy="student", ckpt="/ck/r0.bin", beta=0.5, dagger_round=1)
        s = " ".join(cmd)
        for piece in ("collect_parallel.py", "--instances 2", "--steps 12", "--seeds 5101,5103", "--task base",
                      "--policy student", "--student-ckpt /ck/r0.bin", "--beta 0.5", "--dagger-round 1",
                      "--stop-at-goal", "--tiebreak visible", "--dynamics cf_sitl", "--vehicle e1_mini",
                      "--hold 1.0", "--window-from observation", "--out /out/round_1/collect", "--keep-worlds",
                      f"--infer-bin {dg.BUILD}/uavrgb_infer"):
            self.assertIn(piece, s, piece)
        # β=0：collect.py 的默认值，不传；round 仍传
        s0 = " ".join(dg.collect_cmd("/o", [1], self.SETTINGS, 1, 40, "student", "/c.bin", 0.0, 2))
        self.assertNotIn("--beta", s0)
        self.assertIn("--dagger-round 2", s0)

    def test_teacher_collect_command_has_no_student_flags(self):
        settings = dict(self.SETTINGS, tiebreak="legacy", stop_at_goal=False)
        s = " ".join(dg.collect_cmd("/out/more_demos/collect", [8100], settings, 4, 40))
        for absent in ("--policy", "--student-ckpt", "--beta", "--dagger-round", "--stop-at-goal", "--tiebreak", "--infer-bin"):
            self.assertNotIn(absent, s, absent)

    def test_more_demos_episode_count(self):
        # 3 轮新增 3000 帧，round0 每集 32.8 帧 → 91.5 → 92 集，4 实例整数倍
        self.assertEqual(dg.more_demos_n(3000, 9845 / 300, 4), 92)
        self.assertEqual(dg.more_demos_n(2960, 9845 / 300, 4), 92)   # 90.2 → 91 → 补到 92
        self.assertEqual(dg.more_demos_n(0, 32.8, 4), 0)
        with self.assertRaises(RuntimeError):
            dg.more_demos_n(100, 0, 4)


class ResumeAndReuseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_dagger_gz_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write(self, rel, obj=None, text=None):
        p = os.path.join(self.tmp, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(text if text is not None else json.dumps(obj))
        return p

    def test_train_done_needs_done_line(self):
        run = os.path.join(self.tmp, "run")
        self.assertFalse(dg.train_done(run))
        self._write("run/metrics.jsonl", text='{"event":"validation","step":1}\n')
        self._write("run/ckpt_best.bin", text="x")   # 有 ckpt 不算训完：末行必须 done:
        self.assertFalse(dg.train_done(run))
        self._write("run/metrics.jsonl", text='{"event":"validation","step":1}\n'
                    '{"event":"status","message":"done: 100 samples in 1 s, ckpt=run/ckpt.bin"}\n')
        self.assertTrue(dg.train_done(run))

    def test_eval_done_needs_zero_failed_and_some_done(self):
        edir = os.path.join(self.tmp, "eval")
        self.assertFalse(dg.eval_done(edir))
        self._write("eval/eval.json", {"failed_count": 1, "episodes_done": 19})
        self.assertFalse(dg.eval_done(edir))
        self._write("eval/eval.json", {"failed_count": 0, "episodes_done": 0})
        self.assertFalse(dg.eval_done(edir))
        self._write("eval/eval.json", {"failed_count": 0, "episodes_done": 2})
        self.assertTrue(dg.eval_done(edir))
        # 中断写出的半成品（eval_closed_loop 收到 SIGTERM 写 partial=true）：续跑要挪开重评，不能当评完跳过
        self._write("eval/eval.json", {"failed_count": 0, "episodes_done": 7, "partial": True})
        self.assertFalse(dg.eval_done(edir))
        # E-8：被跳过或截短的评测不算评完；metric_version 或 ckpt 对不上当前配置时也不算评完
        self._write("eval/eval.json", {"episodes_requested": 20, "episodes_done": 5, "failed_count": 0, "skipped_count": 15})
        self.assertFalse(dg.eval_done(edir))
        self._write("eval/eval.json", {"episodes_requested": 20, "episodes_done": 12, "failed_count": 0, "skipped_count": 0})
        self.assertFalse(dg.eval_done(edir))
        ck1, ck2 = self._write("a.bin", text="a"), self._write("b.bin", text="b")
        self._write("eval/eval.json", {"episodes_requested": 2, "episodes_done": 2, "failed_count": 0, "skipped_count": 0,
                                       "metric_version": "sim09.v2.coast1s", "ckpt": ck1})
        self.assertTrue(dg.eval_done(edir, metric_version="sim09.v2.coast1s", ckpt=ck1))
        self.assertFalse(dg.eval_done(edir, metric_version="sim09.v2.coast1s.corridor3m", ckpt=ck1))
        self.assertFalse(dg.eval_done(edir, metric_version="sim09.v2.coast1s", ckpt=ck2))
        # E-8：reusable 基线如果本身是被跳过截短的 eval.json，不能当可复用基线放行
        trunc_base = self._baseline("trunc_base", eval={"episodes_requested": 20, "episodes_done": 2, "skipped_count": 18})
        ok, _why = dg.reusable(trunc_base, seeds={1, 113}, metric_version="sim09.v1.coast1s",
                               plugin_sha="plug-a", infer_sha="infer-a", profile="cf_sitl_v1:683e7400")
        self.assertFalse(ok)

    def test_collect_done_is_manifest(self):
        col = os.path.join(self.tmp, "col")
        self.assertFalse(dg.collect_done(col))
        self._write("col/collection_manifest.json", {"episodes_ok": 1})
        self.assertTrue(dg.collect_done(col))

    def test_partial_collect_ds_and_train_retired_on_resume_and_rename_failure_stops(self):
        """E-9：半成品采集 / 打包 / 训练目录自动挪成 *.partial-<时刻> 后重做，记进 retired_dirs.jsonl；
        改名失败时停下报错，不在旧目录上重做。"""
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        dg.pc.lock_out(out)
        self.addCleanup(dg.pc.unlock_out)
        saved_cfg, saved_run, saved_dry = dict(dg.CFG), dg.run, dg.DRY
        self.addCleanup(lambda: (dg.CFG.clear(), dg.CFG.update(saved_cfg),
                                 setattr(dg, "run", saved_run), setattr(dg, "DRY", saved_dry)))
        dg.DRY = False
        dg.CFG.update(gz_total_max=0, deliver="best", epochs=2)
        settings = dict(hold=1.0, window_from="observation", dynamics="cf_sitl", vehicle="e1_mini",
                        tiebreak="legacy", stop_at_goal=False)
        calls = []

        def fake_run(cmd, **_kw):
            first = os.path.basename(cmd[0])
            tool = first if first == "uavrgb_train" else os.path.basename(cmd[1])
            target = _flag(cmd, "--out")
            self.assertFalse(os.path.exists(target), f"{target} 应已让空")
            calls.append(tool)
            if tool == "collect_parallel.py":
                os.makedirs(target, exist_ok=True)
                dg.write_json(os.path.join(target, "collection_manifest.json"),
                              {"episodes_ok": 4, "episodes_requested": 4, "steps_completed_total": 40,
                               "splits": {"train": ["5101"], "val": [], "test": []}, "end_to_end_wall_s": 1.0})
            elif tool == "preprocess.py":
                os.makedirs(target, exist_ok=True)
                open(os.path.join(target, "dataset.uavr"), "wb").close()
            elif tool == "uavrgb_train":
                os.makedirs(target, exist_ok=True)
                open(os.path.join(target, "ckpt_best.bin"), "wb").close()
                with open(os.path.join(target, "metrics.jsonl"), "w", encoding="utf-8") as f:
                    f.write('{"event":"status","message":"done: 1"}\n')
            return 0

        dg.run = fake_run
        col = os.path.join(out, "round_1", "collect")
        ds = os.path.join(out, "round_1", "ds")
        run_dir = os.path.join(out, "round_1", "run")
        self._write("out/round_1/collect/logs/partial.log", text="half-col")
        self._write("out/round_1/ds/tmp.part", text="half-ds")
        self._write("out/round_1/run/ckpt.bin", text="half-run")
        ep_dir = self._write("ep0/episode_manifest.json", {"seed": 1})
        dg.collect_into(col, [5101, 5102], settings)
        dg.preprocess_into(ds, [os.path.dirname(ep_dir)])
        dg.train_into(ds, run_dir)
        self.assertEqual(calls, ["collect_parallel.py", "preprocess.py", "uavrgb_train"])
        rd_entries = os.listdir(os.path.join(out, "round_1"))
        for prefix in ("collect.partial-", "ds.partial-", "run.partial-"):
            self.assertTrue(any(d.startswith(prefix) for d in rd_entries), (prefix, rd_entries))
        self.assertEqual([(r["kind"], r["path"]) for r in dg.pc.read_retired(out)],
                         [("partial", "round_1/collect"), ("partial", "round_1/ds"), ("partial", "round_1/run")])

        # 改名失败：抛 RuntimeError，不调子命令，旧目录保留
        calls.clear()
        col_busy = os.path.join(out, "busy_col")
        run_busy = os.path.join(out, "busy_run")
        self._write("out/busy_col/x", text="keep1")
        self._write("out/busy_run/ckpt.bin", text="keep2")
        real_rename = dg.pc.os.rename
        dg.pc.os.rename = lambda *_a: (_ for _ in ()).throw(PermissionError("locked"))
        try:
            with self.assertRaises(RuntimeError):
                dg.collect_into(col_busy, [5101], settings)
            with self.assertRaises(RuntimeError):
                dg.train_into(ds, run_busy)
        finally:
            dg.pc.os.rename = real_rename
        self.assertEqual(calls, [])
        self.assertTrue(os.path.isfile(os.path.join(col_busy, "x")))
        self.assertTrue(os.path.isfile(os.path.join(run_busy, "ckpt.bin")))

    def _baseline(self, name="eval-x", **over):
        e = {"failed_count": 0, "episodes_done": 2, "seeds": [1, 113], "metric_version": "sim09.v1.coast1s",
             "infer_bin_sha256": "infer-a", "per_episode": [{"seed": 1, "dynamics_profile": "cf_sitl_v1:683e7400"}]}
        e.update(over.pop("eval", {}))
        self._write(f"{name}/eval.json", e)
        inputs = {"libCfSitlSystem.so": "plug-a"}
        inputs.update(over.pop("inputs", {}))
        if over.pop("no_inputs", False) is False:
            self._write(f"{name}/inputs.json", inputs)
        return os.path.join(self.tmp, name)

    def test_reuse_requires_all_four_conditions(self):
        want = dict(seeds={1, 113}, metric_version="sim09.v1.coast1s", plugin_sha="plug-a",
                    infer_sha="infer-a", profile="cf_sitl_v1:683e7400")
        ok, why = dg.reusable(self._baseline("ok"), **want)
        self.assertTrue(ok, why)
        for i, over in enumerate((dict(eval={"seeds": [1, 122]}), dict(eval={"metric_version": "sim09.v2.coast1s"}),
                                  dict(inputs={"libCfSitlSystem.so": "plug-b"}), dict(eval={"infer_bin_sha256": "infer-b"}),
                                  dict(eval={"per_episode": [{"seed": 1, "dynamics_profile": "cf_sitl_v1:cec5d6b0"}]}),
                                  dict(no_inputs=True), dict(eval={"failed_count": 1}))):
            ok, why = dg.reusable(self._baseline(f"bad{i}", **over), **want)
            self.assertFalse(ok, (i, over, why))
        # 非学生行（teacher/forward）没有 infer sha：只比其余条件
        ok, why = dg.reusable(self._baseline("teacher", eval={"infer_bin_sha256": None}), **dict(want, infer_sha=None))
        self.assertTrue(ok, why)
        # inputs.json 也可以放在 eval 目录的父目录（流水线根）
        d = self._baseline("parent/eval-y", no_inputs=True)
        self._write("parent/inputs.json", {"libCfSitlSystem.so": "plug-a"})
        ok, why = dg.reusable(d, **want)
        self.assertTrue(ok, why)
        # 主会话的 eval 目录用 sha256sum 格式的 inputs.sha256（eval3-gpucheck-20261109 实例）：同样认
        d = self._baseline("parent2/eval-z", no_inputs=True)
        with open(os.path.join(self.tmp, "parent2", "inputs.sha256"), "w", encoding="utf-8") as f:
            f.write("deadbeef  /x/bin/uavrgb_infer\nplug-a  /x/build-cf-sitl/gz/libCfSitlSystem.so\n")
        ok, why = dg.reusable(d, **want)
        self.assertTrue(ok, why)
        with open(os.path.join(self.tmp, "parent2", "inputs.sha256"), "w", encoding="utf-8") as f:
            f.write("plug-b  /x/build-cf-sitl/gz/libCfSitlSystem.so\n")
        ok, why = dg.reusable(d, **want)
        self.assertFalse(ok)
        self.assertIn("libCfSitlSystem.so", why)

    def test_disagreement_stats(self):
        col = os.path.join(self.tmp, "col")
        rows = [  # expert, executed, executed_by, collision
            (1, 1, "student", False), (1, 2, "student", False), (2, 1, "student", True),
            (3, 3, "expert", False), (0, 0, None, False)]
        for i, split in enumerate(("train", "val")):
            d = os.path.join(col, split, str(5100 + i))
            os.makedirs(d)
            with open(os.path.join(d, "actions.jsonl"), "w", encoding="utf-8") as f:
                for a, x, who, c in rows:
                    f.write(json.dumps({"step": 0, "action_expert": a, "action_executed": x, "executed_by": who,
                                        "collision": c}) + "\n")
            with open(os.path.join(d, "episode_manifest.json"), "w", encoding="utf-8") as f:
                json.dump({"stopped_on_goal": i == 0, "num_steps": len(rows)}, f)
        st = dg.disagreement_stats(col)
        self.assertEqual((st["episodes"], st["steps"], st["student_steps"]), (2, 10, 6))
        self.assertEqual(st["disagree"], 4)
        self.assertAlmostEqual(st["disagree_rate"], 4 / 6)
        self.assertEqual(st["by_expert"]["YAW_LEFT"], {"n": 2, "disagree": 2})
        self.assertEqual(st["by_expert"]["FORWARD_SLOW"], {"n": 4, "disagree": 2})
        self.assertEqual((st["reached_goal"], st["collision_episodes"], st["collision_steps"]), (1, 2, 2))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_dagger_gz_")
        self.col = make_round0(os.path.join(self.tmp, "col"))
        self.ckpt = os.path.join(self.tmp, "r0.bin")
        open(self.ckpt, "wb").close()
        self.saved, self.saved_build = dict(dg.CFG), dg.BUILD
        # preflight 查的二进制与插件：worktree 里没有 build/check，用空文件顶替（只查存在与 sha）
        dg.BUILD = os.path.join(self.tmp, "bin")
        os.makedirs(os.path.join(self.tmp, "bin"))
        os.makedirs(os.path.join(self.tmp, "gz"))
        for rel in ("bin/uavrgb_train", "bin/uavrgb_infer", "gz/libCfSitlSystem.so"):
            open(os.path.join(self.tmp, rel), "wb").close()
        dg.CFG.update(plugin_dir=os.path.join(self.tmp, "gz"),
                      deliver="best")   # 既有用例按 smoke（--epochs 2）写，走 deliver=best；calswa 的用例各自显式设

    def tearDown(self):
        dg.CFG.clear()
        dg.CFG.update(self.saved)
        dg.BUILD = self.saved_build
        dg.pc.DRY = dg.DRY = False
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = dg.main(["--out", os.path.join(self.tmp, "out"), *argv])
        return rc, buf.getvalue()

    def test_dry_run_prints_round_chain_and_writes_nothing(self):
        dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt)
        rc, out = self._run("--dry-run", "--rounds", "2", "--eps-per-round", "2", "--instances", "2",
                            "--steps", "12", "--eval-seeds", "1,113", "--epochs", "2", "--baselines", "trunc")
        self.assertEqual(rc, 0, out)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "out")), "dry-run 不建目录")
        o = os.path.join(self.tmp, "out")
        # 第 1 轮：round0 ckpt、β=0.5、轮次 1；第 2 轮：上一轮 ckpt、β=0、轮次 2
        self.assertIn(f"--student-ckpt {self.ckpt} --beta 0.5 --dagger-round 1", out)
        self.assertIn(f"--student-ckpt {o}/round_1/run/ckpt_best.bin --dagger-round 2", out)
        self.assertIn("--stop-at-goal", out)
        self.assertNotIn("--tiebreak", out)                     # round0 是 legacy
        # 并集打包：round0 的 3 集 + 第 1 轮目录；动力学混包开关
        self.assertIn(f"preprocess.py --out {o}/round_1/ds --drop-collision --allow-mixed-dynamics", out)
        self.assertIn(f"uavrgb_train --data {o}/round_2/ds/dataset.uavr --out {o}/round_2/run --epochs 2", out)
        # more_demos 对照臂是教师采，没有 student 开关；评测只有 dagger / more_demos / trunc 三行 + 60 seed 的 dagger
        self.assertIn(f"--out {o}/more_demos/collect", out)
        self.assertIn(f"eval_closed_loop.py --policy student --task base --split test --seeds 1,113 --episodes 2", out)
        self.assertIn(f"--out {o}/eval20-dagger", out)
        self.assertIn(f"--out {o}/eval20-more_demos", out)
        self.assertIn(f"--out {o}/eval20-trunc", out)
        self.assertNotIn("eval20-teacher", out)
        self.assertIn(f"--out {o}/eval60-dagger", out)
        self.assertNotIn("average_ckpt.py", out)                # deliver=best：没有校准平均步，训练命令不带 --save-every-epoch
        self.assertNotIn("--save-every-epoch", out)

    def _dry_eval_cmds(self):
        """dry-run 打印出的每条评测命令（token 列表）：dagger / more_demos / 两条基线 / 60 seed 的 dagger 共 5 条。"""
        dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt)
        rc, out = self._run("--dry-run", "--rounds", "1", "--eps-per-round", "2", "--instances", "2", "--steps", "12",
                            "--eval-seeds", "1,113", "--epochs", "2", "--baselines", "teacher,trunc", "--eval60-arms", "dagger")
        self.assertEqual(rc, 0, out)
        cmds = [line.split(" $ ", 1)[1].split("  > ", 1)[0].split() for line in out.splitlines() if "eval_closed_loop.py" in line]
        self.assertEqual(len(cmds), 5, out)
        return cmds

    def test_eval_commands_carry_corridor_half_width(self):
        """W1-1：评测命令默认带 --corridor-half-width 3（与 e1_pipeline 同口径）；CFG 设 0 回到旧口径，命令只少这两项。"""
        on = self._dry_eval_cmds()
        for cmd in on:
            self.assertIn("--corridor-half-width", cmd, " ".join(cmd))
            self.assertEqual(cmd[cmd.index("--corridor-half-width") + 1], "3", cmd)
        dg.CFG["corridor_half_width"] = 0.0
        off = self._dry_eval_cmds()
        self.assertTrue(all("--corridor-half-width" not in cmd for cmd in off), off)
        strip = lambda c: [t for i, t in enumerate(c) if t != "--corridor-half-width" and (i == 0 or c[i - 1] != "--corridor-half-width")]  # noqa: E731
        self.assertEqual(off, [strip(c) for c in on], "设 0 只应少这两项，其余逐字不变")

    def test_corridor_half_width_env_override(self):
        """CFG 在 import 时读 E1_CORRIDOR_HALF_WIDTH：默认 3.0，设 0 关闭，其它值照读（新进程里验证）。"""
        import subprocess
        code = ("import sys; sys.path[:0] = [%r, %r]; import dagger_gz_pipeline as dg; print(dg.CFG['corridor_half_width'])"
                % (os.path.join(REPO, "sim", "tools"), os.path.join(REPO, "tools")))
        for env_value, want in ((None, "3.0"), ("0", "0.0"), ("2.5", "2.5")):
            env = {k: v for k, v in os.environ.items() if k != "E1_CORRIDOR_HALF_WIDTH"}
            if env_value is not None:
                env["E1_CORRIDOR_HALF_WIDTH"] = env_value
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout.strip(), want, (env_value, r.stdout, r.stderr))

    def test_dry_run_calswa_chain(self):
        """W1-5：deliver=calswa 的 dry-run 链。每次训练（第 1、2 轮与 more_demos）命令带 --save-every-epoch，紧跟一条 average_ckpt.py
        （同一 run / dataset，产物 run/ckpt_calswa.bin，末 10 轮，推理二进制取 BUILD）；下一轮学生自采、20/60 seed 评测的 --ckpt 都是 ckpt_calswa.bin；
        基线行的 ckpt 不变；全程没有 ckpt_best.bin。"""
        dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt, deliver="calswa")
        rc, out = self._run("--dry-run", "--rounds", "2", "--eps-per-round", "2", "--instances", "2", "--steps", "12",
                            "--eval-seeds", "1,113", "--epochs", "10", "--baselines", "trunc", "--eval60-arms", "dagger")
        self.assertEqual(rc, 0, out)
        o = os.path.join(self.tmp, "out")
        cmds = _dry_cmds(out)
        trains = [i for i, c in enumerate(cmds) if os.path.basename(c[0]) == "uavrgb_train"]
        self.assertEqual([_flag(cmds[i], "--out") for i in trains], [f"{o}/round_1/run", f"{o}/round_2/run", f"{o}/more_demos/run"])
        for i in trains:
            train, avg = cmds[i], cmds[i + 1]
            run_dir = _flag(train, "--out")
            self.assertIn("--save-every-epoch", train)
            self.assertEqual(avg[:2], [dg.PY, os.path.join(REPO, "tools", "average_ckpt.py")], avg)
            self.assertEqual(_flag(avg, "--run"), run_dir)
            self.assertEqual(_flag(avg, "--data"), _flag(train, "--data"))
            self.assertEqual(_flag(avg, "--out"), f"{run_dir}/ckpt_calswa.bin")
            self.assertEqual(_flag(avg, "--last-k"), "10")
            self.assertEqual(_flag(avg, "--infer-bin"), f"{dg.BUILD}/uavrgb_infer")
        self.assertEqual(sum(1 for c in cmds if os.path.basename(c[1]) == "average_ckpt.py"), 3, "三次训练各一次校准平均")
        self.assertIn(f"--student-ckpt {self.ckpt} --beta 0.5 --dagger-round 1", out)             # round0 学生不变
        self.assertIn(f"--student-ckpt {o}/round_1/run/ckpt_calswa.bin --dagger-round 2", out)   # 下一轮自采用交付权重
        evals = {os.path.basename(_flag(c, "--out")): c for c in cmds if os.path.basename(c[1]) == "eval_closed_loop.py"}
        self.assertEqual(sorted(evals), ["eval20-dagger", "eval20-more_demos", "eval20-trunc", "eval60-dagger"])
        self.assertEqual(_flag(evals["eval20-dagger"], "--ckpt"), f"{o}/round_2/run/ckpt_calswa.bin")
        self.assertEqual(_flag(evals["eval60-dagger"], "--ckpt"), f"{o}/round_2/run/ckpt_calswa.bin")
        self.assertEqual(_flag(evals["eval20-more_demos"], "--ckpt"), f"{o}/more_demos/run/ckpt_calswa.bin")
        self.assertEqual(_flag(evals["eval20-trunc"], "--ckpt"), dg.TRUNC_CKPT)
        self.assertNotIn("ckpt_best.bin", out)

    def test_calswa_needs_enough_epochs_at_preflight(self):
        """calswa 要末 avg_last_k 轮：--epochs 小于它时 preflight 就失败（status/DONE 记原因，提示 E1_DELIVER），不等采集几小时后训练步才炸；
        deliver=best 不受限。"""
        dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt, deliver="calswa")
        rc, out = self._run("--rounds", "1", "--baselines", "", "--epochs", "9", "--stop-after-preflight")
        self.assertEqual(rc, 1, out)
        with open(os.path.join(self.tmp, "out", "status.json"), encoding="utf-8") as f:
            st = json.load(f)
        self.assertEqual(st["preflight"]["state"], "failed")
        self.assertIn("E1_DELIVER", st["preflight"]["error"])
        self.assertIn("epochs=9", st["preflight"]["error"])
        dg.CFG["deliver"] = "best"
        rc, out = self._run("--rounds", "1", "--baselines", "", "--epochs", "9", "--stop-after-preflight")
        self.assertEqual(rc, 0, out)

    def _dry_lines(self, **cfg):
        """dry-run 打印的命令行（每行一条）；cfg 改 CFG（tearDown 会恢复）。"""
        dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt, **cfg)
        rc, out = self._run("--dry-run", "--rounds", "2", "--eps-per-round", "2", "--instances", "2",
                            "--steps", "12", "--eval-seeds", "1,113", "--epochs", "2", "--baselines", "trunc")
        self.assertEqual(rc, 0, out)
        return out.splitlines()

    def test_default_student_cond_adds_no_flag_anywhere(self):
        self.assertEqual(dg.CFG["student_cond"], "prev-action")
        lines = self._dry_lines()
        for flag in ("--student-cond", "--cond", "--prev-action"):
            self.assertEqual([ln for ln in lines if flag in ln], [], flag)

    def test_heading_code_is_passed_to_collect_preprocess_train_and_eval(self):
        lines = self._dry_lines(student_cond="heading-code")
        o = os.path.join(self.tmp, "out")
        collects = [ln for ln in lines if "collect_parallel.py" in ln]
        self.assertTrue(collects)
        for ln in collects:
            if "--policy student" in ln:
                self.assertIn("--student-cond heading-code", ln)
            else:
                self.assertNotIn("--student-cond", ln, "more_demos 是教师采集：没有学生，也就没有条件输入")
        self.assertTrue([ln for ln in collects if "--policy student" in ln])
        packs = [ln for ln in lines if "preprocess.py" in ln]
        self.assertTrue(packs)
        for ln in packs:
            self.assertIn("--cond heading-code", ln)
        trains = [ln for ln in lines if "uavrgb_train" in ln]
        self.assertTrue(trains)
        for ln in trains:
            self.assertIn("--prev-action", ln)   # 不加就训出忽略条件输入的模型，评测端闸门会拒绝

        def eval_line(name):
            hit = [ln for ln in lines if "eval_closed_loop.py" in ln and f"--out {o}/{name}" in ln]
            self.assertEqual(len(hit), 1, name)
            return hit[0]
        for name in ("eval20-dagger", "eval20-more_demos", "eval60-dagger"):
            self.assertIn("--student-cond heading-code", eval_line(name), name)
        self.assertNotIn("--student-cond", eval_line("eval20-trunc"), "基线 trunc 没有条件输入，评测不带这个开关")

    def test_preflight_failure_writes_status_and_done(self):
        dg.CFG.update(round0_collection=self.col, round0_ckpt=os.path.join(self.tmp, "missing.bin"))
        rc, out = self._run("--rounds", "1", "--baselines", "")
        self.assertEqual(rc, 1)
        o = os.path.join(self.tmp, "out")
        with open(os.path.join(o, "DONE"), encoding="utf-8") as f:
            self.assertTrue(f.read().startswith("rc=1 "))
        with open(os.path.join(o, "status.json"), encoding="utf-8") as f:
            st = json.load(f)
        self.assertEqual(st["preflight"]["state"], "failed")
        self.assertIn("missing.bin", st["preflight"]["error"])
        import fcntl
        self.assertTrue(os.path.exists(os.path.join(o, ".pipeline.lock")), "E-3：main 起跑应拿过实例锁")
        with open(os.path.join(o, ".pipeline.lock"), "a+", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)   # E-3：收尾（DONE 写完）已放锁，同目录马上能再起
        with open(os.path.join(o, "run.pid"), encoding="utf-8") as f:   # E-4：run.pid 带启动时刻
            rec = dict(ln.split("=", 1) for ln in f.read().splitlines())
        self.assertEqual(rec["pid"], str(os.getpid()))
        self.assertEqual(rec["start"], dg.pc.proc_stat_field(os.getpid(), 22))

    def test_second_instance_exits_before_touching_out(self):
        """E-3：输出目录已被另一个实例占着时，main 在写任何文件之前退出 3：第一个实例的 config.json、DONE、run.pid 原样留着，不进 preflight
        （preflight 换成会抛的替身，没拦住也起不了任何子进程）。"""
        import fcntl
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        holder = open(os.path.join(out, ".pipeline.lock"), "a+", encoding="utf-8")   # 另一个实例：约定的锁文件名 + flock
        self.addCleanup(holder.close)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, text in (("DONE", "rc=0 first-instance\n"), ("config.json", '{"first": true}'),
                           ("run.pid", "pid=1\npgid=1\nstart=1\ncmd=first-instance\n")):
            with open(os.path.join(out, name), "w", encoding="utf-8") as f:
                f.write(text)
        dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt)

        def boom(*_a, **_k):
            raise AssertionError("拿不到锁却走到了 preflight")

        saved = dg._preflight_all
        dg._preflight_all = boom
        self.addCleanup(setattr, dg, "_preflight_all", saved)
        with self.assertRaises(SystemExit) as cm:
            self._run("--rounds", "1", "--baselines", "")
        self.assertEqual(cm.exception.code, 3)
        self.assertEqual(sorted(os.listdir(out)), [".pipeline.lock", "DONE", "config.json", "run.pid"])
        with open(os.path.join(out, "DONE"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "rc=0 first-instance\n")
        with open(os.path.join(out, "config.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"first": True})
        with open(os.path.join(out, "run.pid"), encoding="utf-8") as f:
            self.assertEqual(f.read(), "pid=1\npgid=1\nstart=1\ncmd=first-instance\n")

    def test_preflight_records_both_dynamics_profiles(self):
        dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt)
        rc, out = self._run("--rounds", "1", "--baselines", "", "--stop-after-preflight")
        self.assertEqual(rc, 0, out)
        with open(os.path.join(self.tmp, "out", "status.json"), encoding="utf-8") as f:
            st = json.load(f)
        prof = st["preflight"]["dynamics_profiles"]
        self.assertEqual(prof["round0"], ["cf_sitl_v1:bd0dcd43"])
        self.assertTrue(prof["current"].startswith("cf_sitl_v1:"))
        self.assertNotEqual(prof["current"], prof["round0"][0])   # 不一致只记录，不退出（混包开关兜底）
        with open(os.path.join(self.tmp, "out", "inputs.json"), encoding="utf-8") as f:
            inputs = json.load(f)
        for key in ("libCfSitlSystem.so", "e1_mini.json", "uavrgb_train", "uavrgb_infer", "round0_ckpt"):
            self.assertIn(key, inputs)

    def test_evaluate_and_summary_retire_stale_and_truncated(self):
        """E-8：evaluate 与 baselines 续跑遇到 metric_version / ckpt 不一致挪 .stale-<时刻>，遇到截断/跳过挪 .failed-<时刻>，
        一致的跳过；summary 把挪走的目录列进 summary.md 与 summary.json['retired']，没挪过时不改输出。"""
        import subprocess
        out = os.path.join(self.tmp, "out")
        os.makedirs(out)
        dg.pc.lock_out(out)
        self.addCleanup(dg.pc.unlock_out)
        settings = dict(hold=1.0, window_from="observation", dynamics="cf_sitl", vehicle="e1_mini",
                        tiebreak="legacy", stop_at_goal=False)
        dg.CFG.update(corridor_half_width=3.0, gz_total_max=0, eval60_arms=())
        mv_now, mv_old = "sim09.v2.coast1s.corridor3m", "sim09.v2.coast1s"
        calswa = os.path.join(out, "round_1", "run", "ckpt_calswa.bin")
        best = os.path.join(out, "round_1", "run", "ckpt_best.bin")
        md_ckpt = os.path.join(out, "more_demos", "run", "ckpt_calswa.bin")
        trunc_ckpt = os.path.join(self.tmp, "trunc.bin")
        for p in (calswa, best, md_ckpt, trunc_ckpt):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "wb").close()
        dg.CFG["baselines"] = {"trunc": ("student", trunc_ckpt, "")}
        dg.write_json(os.path.join(out, "inputs.json"), {"libCfSitlSystem.so": "p", "uavrgb_infer": "i"})
        saved_popen, saved_run = subprocess.Popen, dg.run
        dg.run = lambda *_a, **_k: 0   # summary 末尾的 stats_report 子进程替身
        self.addCleanup(setattr, subprocess, "Popen", saved_popen)
        self.addCleanup(setattr, dg, "run", saved_run)

        def put(rel, **over):
            doc = {"episodes_requested": 2, "episodes_done": 2, "failed_count": 0, "skipped_count": 0,
                   "metric_version": mv_now, "ckpt": None, "requested_success_rate": 0.5,
                   "per_episode": [{"seed": 1, "success": True, "dynamics_profile": "p"},
                                   {"seed": 113, "success": False, "dynamics_profile": "p"}]}
            doc.update(over)
            p = os.path.join(out, rel, "eval.json")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            dg.write_json(p, doc)

        # 未挪走任何目录时 summary 不带 retired
        put("eval20-dagger", ckpt=calswa)
        put("eval20-more_demos", ckpt=md_ckpt)
        res0 = dg.summary(out, 1, [("dagger", "student", calswa), ("more_demos", "student", md_ckpt)], {}, "")
        self.assertNotIn("retired", res0)
        with open(os.path.join(out, "summary.md"), encoding="utf-8") as f:
            self.assertNotIn("续跑挪走", f.read())

        # 造三种续跑现场：dagger 评的是旧 best 权重（stale），more_demos 一致（done），基线 trunc 旧口径无走廊（stale），另测截断（failed）
        put("eval20-dagger", ckpt=best)
        put("eval20-trunc", metric_version=mv_old, ckpt=trunc_ckpt)
        put("eval20-forward_tmp", episodes_done=1, skipped_count=1)
        dg.CFG["baselines"]["forward_tmp"] = ("forward", None, "")
        started = []

        class FakeEvalProc:
            returncode = 0
            def poll(self):
                return 0
            def wait(self):
                return 0

        def fake_popen(cmd, **_kw):
            edir = _flag(cmd, "--out")
            self.assertFalse(os.path.exists(edir), f"{edir} 应已挪空")
            started.append(os.path.basename(edir))
            ck = _flag(cmd, "--ckpt") if "--ckpt" in cmd else None
            put(os.path.relpath(edir, out), ckpt=ck)
            return FakeEvalProc()

        subprocess.Popen = fake_popen

        arms = [("dagger", "student", calswa), ("more_demos", "student", md_ckpt)]
        dg.evaluate(out, arms, "1,113", "eval20", settings)
        decisions = dg.baselines(out, ["trunc", "forward_tmp"], "1,113", settings)
        self.assertEqual(started, ["eval20-dagger", "eval20-trunc", "eval20-forward_tmp"])
        names = os.listdir(out)
        self.assertTrue(any(d.startswith("eval20-dagger.stale-") for d in names), names)
        self.assertTrue(any(d.startswith("eval20-trunc.stale-") for d in names), names)
        self.assertTrue(any(d.startswith("eval20-forward_tmp.failed-") for d in names), names)
        res = dg.summary(out, 1, arms, decisions, "")
        self.assertEqual([(r["kind"], r["path"]) for r in res["retired"]],
                         [("stale", "eval20-dagger"), ("stale", "eval20-trunc"), ("failed", "eval20-forward_tmp")])
        with open(os.path.join(out, "summary.md"), encoding="utf-8") as f:
            md = f.read()
        self.assertIn("续跑挪走的目录", md)
        self.assertIn("eval20-dagger.stale-", md)
        self.assertIn("eval20-forward_tmp.failed-", md)

    def test_action_latency_and_rtf_passthrough_and_validation(self):
        """A2：CFG 的 action_latency 与 rtf 缺省均为 None（从 e1_pipeline.CFG 继承，E1_ACTION_LATENCY / E1_RTF 可覆盖），
        默认 dry-run 命令行逐字不变；给了之后透传到全部采集（各轮学生自采 + more_demos 教师采）与全部评测（eval20 / 基线重评 / eval60）；
        非法取值在 preflight 报错。"""
        import subprocess
        self.assertIsNone(self.saved["action_latency"])
        self.assertIsNone(self.saved["rtf"])
        base_cmds = _dry_cmds("\n".join(self._dry_lines()))
        both_cmds = _dry_cmds("\n".join(self._dry_lines(action_latency=0.20, rtf=0.5)))
        self.assertEqual(len(base_cmds), len(both_cmds))

        def strip_pair(cmd, flag):
            return [t for i, t in enumerate(cmd) if t != flag and (i == 0 or cmd[i - 1] != flag)]

        collects, evals = 0, 0
        for b, w in zip(base_cmds, both_cmds):
            first = os.path.basename(b[0])
            tool = first if first == "uavrgb_train" else os.path.basename(b[1])
            self.assertNotIn("--action-latency", b)
            self.assertNotIn("--rtf", b)
            if tool in ("collect_parallel.py", "eval_closed_loop.py"):
                collects += tool == "collect_parallel.py"
                evals += tool == "eval_closed_loop.py"
                self.assertEqual(_flag(w, "--action-latency"), "0.2", w)
                self.assertEqual(_flag(w, "--rtf"), "0.5", w)
                self.assertEqual(strip_pair(strip_pair(w, "--action-latency"), "--rtf"), b)
            else:
                self.assertEqual(w, b, tool)
        self.assertEqual((collects, evals), (3, 4))   # round_1 + round_2 + more_demos；eval20 3 条 + eval60 1 条

        # 新进程环境变量覆盖验证
        code = ("import sys; sys.path[:0] = [%r, %r]; import dagger_gz_pipeline as dg; "
                "print(dg.CFG['action_latency'], dg.CFG['rtf'])"
                % (os.path.join(REPO, "sim", "tools"), os.path.join(REPO, "tools")))
        env = {k: v for k, v in os.environ.items() if k not in ("E1_ACTION_LATENCY", "E1_RTF")}
        r0 = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual((r0.returncode, r0.stdout.strip()), (0, "None None"), r0.stderr)
        r1 = subprocess.run([sys.executable, "-c", code],
                            env=dict(env, E1_ACTION_LATENCY="0.2", E1_RTF="0.5"),
                            capture_output=True, text=True, timeout=120)
        self.assertEqual((r1.returncode, r1.stdout.strip()), (0, "0.2 0.5"), r1.stderr)

        # 非法取值在 preflight 失败
        for bad in (dict(action_latency=0.0, rtf=None), dict(action_latency=1.0, rtf=None), dict(action_latency=None, rtf=0.0)):
            dg.CFG.update(round0_collection=self.col, round0_ckpt=self.ckpt, **bad)
            rc, _out = self._run("--rounds", "1", "--baselines", "", "--epochs", "2", "--stop-after-preflight")
            self.assertEqual(rc, 1, bad)


def _dry_cmds(out):
    """dry-run 输出里每条 `$ ` 命令的 token 列表（去掉时间戳与 `  > 日志` 后缀）。"""
    return [line.split(" $ ", 1)[1].split("  > ", 1)[0].split() for line in out.splitlines() if " $ " in line]


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


class DeliverTests(unittest.TestCase):
    """W1-5：训练后的交付权重。calswa = 末 10 轮平均 + 校准（run/ckpt_calswa.bin），best = ckpt_best.bin（改前行为）。
    dg.run 换成替身（按真实工具的产物落文件），不起任何子进程。"""
    DONE = '{"event":"status","message":"done: 100 samples in 1 s, ckpt=run/ckpt.bin"}\n'

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_dagger_gz_")
        self.ds, self.run_dir = os.path.join(self.tmp, "ds"), os.path.join(self.tmp, "run")
        self.swa = os.path.join(self.run_dir, "ckpt_calswa.bin")
        self.calls, self.fail_on = [], ()
        self.saved, self.real_run, self.real_dry = dict(dg.CFG), dg.run, dg.DRY
        dg.run, dg.DRY = self._fake_run, False
        dg.CFG.update(deliver="calswa", epochs=30, avg_last_k=10, drop_collision=True)
        self._touch(os.path.join(self.ds, "dataset.uavr"))

    def tearDown(self):
        dg.run, dg.DRY = self.real_run, self.real_dry
        dg.CFG.clear()
        dg.CFG.update(self.saved)
        shutil.rmtree(self.tmp, ignore_errors=True)

    @staticmethod
    def _touch(path, text=""):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    @staticmethod
    def _tool(cmd):
        first = os.path.basename(cmd[0])
        return first if first == "uavrgb_train" else os.path.basename(cmd[1])

    def _fake_run(self, cmd, env_extra=None, log_path=None):
        self.calls.append((list(cmd), env_extra, log_path))
        tool = self._tool(cmd)
        if tool in self.fail_on:
            return 1
        if tool == "uavrgb_train":
            out = _flag(cmd, "--out")
            self._touch(os.path.join(out, "ckpt_best.bin"))
            self._touch(os.path.join(out, "metrics.jsonl"), self.DONE)
        elif tool == "average_ckpt.py":
            self._touch(_flag(cmd, "--out"))
            self._touch(_flag(cmd, "--out") + ".json", "{}")
        return 0

    def _tools(self):
        return [self._tool(c) for c, _, _ in self.calls]

    def test_calswa_train_then_average(self):
        dg.train_into(self.ds, self.run_dir)
        self.assertEqual(self._tools(), ["uavrgb_train", "average_ckpt.py"])
        (train, _, train_log), (avg, avg_env, avg_log) = self.calls
        self.assertEqual(train.count("--save-every-epoch"), 1, train)
        self.assertEqual(train_log, self.run_dir + ".log")
        self.assertEqual(avg[:2], [dg.PY, os.path.join(REPO, "tools", "average_ckpt.py")])
        self.assertEqual(_flag(avg, "--run"), self.run_dir)
        self.assertEqual(_flag(avg, "--data"), f"{self.ds}/dataset.uavr")
        self.assertEqual(_flag(avg, "--out"), self.swa)
        self.assertEqual(_flag(avg, "--last-k"), "10")
        self.assertEqual(_flag(avg, "--infer-bin"), f"{dg.BUILD}/uavrgb_infer")
        self.assertNotIn("--no-calibrate", avg)
        self.assertEqual(avg_env, {"CUDA_VISIBLE_DEVICES": dg.CFG["gpu"]})
        self.assertEqual(avg_log, self.run_dir + ".average.log")

    def test_calswa_flags_exist_in_tools(self):
        """流水线拼出的开关在被调工具里真实存在。替身 run 只看命令内容，开关笔误抓不到；
        这里静态读 average_ckpt.py 的 add_argument 与 train_main.cpp 的开关分支，对照训练命令与校准平均命令里的每个 `--开关`。"""
        dg.train_into(self.ds, self.run_dir)
        train, avg = self.calls[0][0], self.calls[1][0]

        def known(path, pattern):
            with open(os.path.join(REPO, *path), encoding="utf-8") as f:
                return set(re.findall(pattern, f.read()))
        avg_known = known(("tools", "average_ckpt.py"), r'add_argument\("(--[\w-]+)"')
        train_known = known(("src", "train_main.cpp"), r'a == "(--[\w-]+)"')
        avg_used = {a for a in avg if a.startswith("--")}
        train_used = {a for a in train if a.startswith("--")}
        self.assertIn("--save-every-epoch", train_used)
        self.assertTrue(avg_used)
        self.assertLessEqual(avg_used, avg_known, f"average_ckpt.py 不认的开关：{sorted(avg_used - avg_known)}")
        self.assertLessEqual(train_used, train_known, f"uavrgb_train 不认的开关：{sorted(train_used - train_known)}")

    def test_best_keeps_old_command(self):
        dg.CFG["deliver"] = "best"
        dg.train_into(self.ds, self.run_dir)
        want = [f"{dg.BUILD}/uavrgb_train", "--data", f"{self.ds}/dataset.uavr", "--out", self.run_dir,
                "--epochs", str(dg.CFG["epochs"]), "--batch", str(dg.CFG["batch"]), "--lr", repr(float(dg.CFG["lr"])),
                "--seed", str(dg.CFG["seed"]), "--class-weights", dg.CFG["class_weights"], "--drop-collision"]
        self.assertEqual([c for c, _, _ in self.calls], [want])

    def test_resume_trained_run_only_averages(self):
        """训练做完但没有 ckpt_calswa.bin：只补平均，不重训；产物（.bin 与 .json）齐了什么都不跑；只剩 .bin 不当完成。"""
        self._touch(os.path.join(self.run_dir, "ckpt_best.bin"))
        self._touch(os.path.join(self.run_dir, "metrics.jsonl"), self.DONE)
        dg.train_into(self.ds, self.run_dir)
        self.assertEqual(self._tools(), ["average_ckpt.py"])
        self.calls.clear()
        dg.train_into(self.ds, self.run_dir)
        self.assertEqual(self.calls, [], "产物齐了不该再跑")
        os.remove(self.swa + ".json")
        dg.train_into(self.ds, self.run_dir)
        self.assertEqual(self._tools(), ["average_ckpt.py"])

    def test_average_failure_stops_without_fallback(self):
        self.fail_on = ("average_ckpt.py",)
        with self.assertRaises(RuntimeError) as cm:
            dg.train_into(self.ds, self.run_dir)
        self.assertIn("average_ckpt.py", str(cm.exception))
        self.assertFalse(os.path.exists(self.swa), "失败时不能留下交付权重")

    def test_calswa_needs_enough_epochs_and_known_value(self):
        dg.CFG["epochs"] = 9
        with self.assertRaises(RuntimeError) as cm:
            dg.train_into(self.ds, self.run_dir)
        self.assertIn("E1_DELIVER", str(cm.exception))
        self.assertEqual(self.calls, [], "报错前不该起命令")
        dg.CFG.update(epochs=30, deliver="swa")
        with self.assertRaises(RuntimeError) as cm:
            dg.train_into(self.ds, self.run_dir)
        self.assertTrue("calswa" in str(cm.exception) and "best" in str(cm.exception), cm.exception)

    def test_deliver_env_overrides(self):
        """CFG 在 import 时读 e1_pipeline.CFG（单源）：deliver 缺省 calswa、E1_DELIVER=best 切回；gpu 缺省 11-11 起机器上唯一那张卡的 UUID，
        E1_GPU 覆盖；avg_last_k 缺省 10（新进程里验证）。"""
        import subprocess
        code = ("import sys; sys.path[:0] = [%r, %r]; import dagger_gz_pipeline as dg; "
                "print(dg.CFG['deliver'], dg.CFG['gpu'], dg.CFG['avg_last_k'])"
                % (os.path.join(REPO, "sim", "tools"), os.path.join(REPO, "tools")))

        def read(**env_extra):
            env = {k: v for k, v in os.environ.items() if k not in ("E1_DELIVER", "E1_GPU")}
            env.update(env_extra)
            r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 0, r.stderr)
            return r.stdout.split()

        self.assertEqual(read(), ["calswa", "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740", "10"])
        self.assertEqual(read(E1_DELIVER="best")[0], "best")
        self.assertEqual(read(E1_GPU="GPU-test")[1], "GPU-test")

    def test_delivered_ckpt_path(self):
        self.assertEqual(dg.deliver_ckpt(self.run_dir), self.swa)
        dg.CFG["deliver"] = "best"
        self.assertEqual(dg.deliver_ckpt(self.run_dir), os.path.join(self.run_dir, "ckpt_best.bin"))


if __name__ == "__main__":
    unittest.main()
