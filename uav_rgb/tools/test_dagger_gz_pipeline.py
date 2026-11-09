#!/usr/bin/env python3
"""dagger_gz_pipeline 的纯 CPU 回归：round-0 开关读取与缺字段退出、seed 按 gid 排除、dry-run 命令行逐字、
more_demos 集数、基线复用四条件、续跑判定、专家不一致率、失败时 status/DONE。不起 gz、不读真实数据。

运行（仓库根）：.venv-tools/bin/python tools/test_dagger_gz_pipeline.py
"""
import contextlib
import io
import json
import os
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

    def test_pick_up_to_returns_partial_list_instead_of_dying(self):
        # base 任务全部只有 71 个 test 几何，排掉训练集合后不足 60：清单取能取到的，不能让跑了几小时的流水线在这一步退出
        seeds, gids = dg.pick_up_to(5, "test", set(), seed_max=400)
        self.assertEqual(len(seeds), 5)
        self.assertEqual(len(set(gids)), 5)
        self.assertTrue(all(sm.split_for_seed(s, "base") == "test" for s in seeds))
        self.assertEqual(seeds, sorted(seeds))
        few, few_gids = dg.pick_up_to(10_000, "test", set(gids), seed_max=400)   # 要得太多：拿到剩余全部，不抛
        self.assertLess(len(few), 10_000)
        self.assertTrue(set(few_gids).isdisjoint(gids))

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

    def test_collect_done_is_manifest(self):
        col = os.path.join(self.tmp, "col")
        self.assertFalse(dg.collect_done(col))
        self._write("col/collection_manifest.json", {"episodes_ok": 1})
        self.assertTrue(dg.collect_done(col))

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
        dg.CFG.update(plugin_dir=os.path.join(self.tmp, "gz"))

    def tearDown(self):
        dg.CFG.clear()
        dg.CFG.update(self.saved)
        dg.BUILD = self.saved_build
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


if __name__ == "__main__":
    unittest.main()
