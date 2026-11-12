#!/usr/bin/env python3
"""compare_datasets 的不起 gz / 不训练单测：续跑判定（AUDIT Must 1/2）、孤儿 gz（Must 3）、dry-run 不写盘。

运行（仓库根）：.venv-tools/bin/python tools/test_compare_datasets.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import compare_datasets as cd  # noqa: E402

DONE_LINE = '{"schema":"tinyvln.metrics.v1","event":"status","message":"done: 1 samples in 1 s, ckpt=x"}\n'


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


class Fixture:
    """只留 gazebo 域、一个训练集 A、一个训练种子；ckpt 都是空文件。"""

    def __enter__(self):
        self.tmp = tempfile.mkdtemp(prefix="test_compare_datasets_")
        self.saved = (dict(cd.CFG), cd.EVAL_DOMAINS, cd.DRY, subprocess.Popen, cd.sh, cd.run, cd.GROUPS)
        cd.GROUPS = tuple(g for g in cd.GROUPS if g[2] in (None, "old_x3_ckpt", "A"))
        cd.CFG["datasets"] = dict(A=("A",))
        cd.CFG["train_seeds"] = (0,)
        cd.CFG["old_x3_ckpt"] = os.path.join(self.tmp, "old_x3.bin")
        cd.EVAL_DOMAINS = ("gazebo",)
        cd.DRY = False
        cd._BLOCKED.clear()
        _touch(cd.CFG["old_x3_ckpt"])
        _touch(os.path.join(cd.ds_dir(self.tmp, "A"), "dataset.uavr"))
        return self

    def __exit__(self, *_):
        cfg, cd.EVAL_DOMAINS, cd.DRY, subprocess.Popen, cd.sh, cd.run, cd.GROUPS = self.saved
        cd.CFG.clear()
        cd.CFG.update(cfg)
        cd._BLOCKED.clear()
        shutil.rmtree(self.tmp, ignore_errors=True)


class FakeProc:
    waited = []

    def __init__(self, label, edir, write_eval):
        self.label, self.edir, self.write_eval, self.returncode = label, edir, write_eval, None

    def poll(self):   # pipeline_common.children() 用它判活
        return self.returncode

    def wait(self):
        FakeProc.waited.append(self.label)
        if self.write_eval:
            _touch(os.path.join(self.edir, "eval.json"), json.dumps({"episodes_done": 1, "failed_count": 0}))
            self.returncode = 0
        else:
            self.returncode = 1
        return self.returncode


def _fake_popen(broken=(), started=None):
    def popen(cmd, **_kw):
        edir = cmd[cmd.index("--out") + 1]
        label = os.path.basename(edir)
        if started is not None:
            started.append(label)
        assert not os.path.exists(edir), f"{edir} 应先让出来（eval_closed_loop 要求 --out 为空）"
        return FakeProc(label, edir, write_eval=label not in broken)
    return popen


def test_train_resume_by_metrics_done():
    with Fixture() as fx:
        calls = []
        cd.sh = lambda cmd, *a, **k: calls.append(cmd)
        _label, run_dir = cd.train_runs(fx.tmp, "A")[0]
        _touch(os.path.join(run_dir, "ckpt_best.bin"))
        try:
            cd.train(fx.tmp)
        except RuntimeError as e:
            assert "非空" in str(e), e
        else:
            raise AssertionError("有 ckpt_best.bin 但 metrics.jsonl 没有 done 行：不是完成，run 目录非空应报错")
        _touch(os.path.join(run_dir, "metrics.jsonl"), DONE_LINE)
        cd.train(fx.tmp)
        assert calls == [], calls
    print("ok train：按 metrics.jsonl done 行续跑")


def test_evaluate_resume_and_waits_all_before_need():
    with Fixture() as fx:
        _label, run_dir = cd.train_runs(fx.tmp, "A")[0]
        _touch(os.path.join(run_dir, "ckpt_best.bin"))
        _touch(os.path.join(run_dir, "metrics.jsonl"), DONE_LINE)
        cd.CFG["gazebo"] = dict(cd.CFG["gazebo"], parallel=8)     # 全部一批，一个 wait 循环
        jobs = cd.eval_jobs(fx.tmp)
        labels = [os.path.basename(j[5]) for j in jobs]
        assert len(labels) >= 4, labels
        finished, stale, broken = labels[0], labels[1], labels[2]
        _touch(os.path.join(jobs[0][5], "eval.json"), json.dumps({"episodes_done": 20, "failed_count": 0}))
        _touch(os.path.join(jobs[1][5], "eval.json"), json.dumps({"episodes_done": 19, "failed_count": 1}))
        started, FakeProc.waited = [], []
        subprocess.Popen = _fake_popen(broken=(broken,), started=started)
        try:
            cd.evaluate(fx.tmp)
        except RuntimeError as e:
            err = str(e)
        else:
            raise AssertionError(f"{broken} 没写 eval.json，evaluate 应报错")
        assert broken in err, err
        assert finished not in started, f"完成的组被重跑：{started}"
        assert stale in started, f"failed_count>0 的组没重跑：{started}"
        parent = os.path.dirname(jobs[1][5])
        assert any(d.startswith(stale + ".failed-") for d in os.listdir(parent)), os.listdir(parent)
        # Must 3：报错前等齐这一批所有组，否则其余组的 gz 留成孤儿。
        assert FakeProc.waited == started, f"报错前没等齐：等了 {FakeProc.waited}，起了 {started}"
    print("ok evaluate：按 eval_finished 续跑；缺 eval.json 时等齐全批再报错")


def test_evaluate_retires_skipped_or_truncated_eval():
    """E-8：compare_datasets 共用 pipeline_common.eval_finished / retire_unfinished_eval。
    被跳过（skipped_count>0）或完成数少于请求数（截断）的 eval.json 不再当评完跳过，挪到 .failed-* 后重跑。"""
    with Fixture() as fx:
        _label, run_dir = cd.train_runs(fx.tmp, "A")[0]
        _touch(os.path.join(run_dir, "ckpt_best.bin"))
        _touch(os.path.join(run_dir, "metrics.jsonl"), DONE_LINE)
        cd.CFG["gazebo"] = dict(cd.CFG["gazebo"], parallel=8)
        jobs = cd.eval_jobs(fx.tmp)
        labels = [os.path.basename(j[5]) for j in jobs]
        _touch(os.path.join(jobs[0][5], "eval.json"),
               json.dumps({"episodes_requested": 20, "episodes_done": 20, "failed_count": 0, "skipped_count": 0}))
        _touch(os.path.join(jobs[1][5], "eval.json"),
               json.dumps({"episodes_requested": 20, "episodes_done": 5, "failed_count": 0, "skipped_count": 15}))
        _touch(os.path.join(jobs[2][5], "eval.json"),
               json.dumps({"episodes_requested": 20, "episodes_done": 12, "failed_count": 0, "skipped_count": 0}))
        for j in jobs[3:]:
            _touch(os.path.join(j[5], "eval.json"),
                   json.dumps({"episodes_requested": 20, "episodes_done": 20, "failed_count": 0, "skipped_count": 0}))
        started, FakeProc.waited = [], []
        subprocess.Popen = _fake_popen(started=started)
        cd.evaluate(fx.tmp)
        assert started == [labels[1], labels[2]], f"应只重跑跳过与截断的两组，实际 {started}"
        parent = os.path.dirname(jobs[1][5])
        assert any(d.startswith(labels[1] + ".failed-") for d in os.listdir(parent)), os.listdir(parent)
        assert any(d.startswith(labels[2] + ".failed-") for d in os.listdir(parent)), os.listdir(parent)
    print("ok evaluate（E-8）：跳过与截断的 eval.json 在 compare_datasets 中也不算评完，挪 .failed- 重跑")


def test_main_records_interrupt_with_real_rc():
    import signal
    import pipeline_common as pc
    tmp = tempfile.mkdtemp(prefix="test_compare_datasets_")
    saved = (cd.RUN["summary"], sys.argv, {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)})

    def interrupted(_out):
        raise pc.Interrupted(signal.SIGINT)

    try:
        cd.RUN["summary"] = interrupted
        sys.argv = ["compare_datasets.py", "--out", tmp, "--from", "summary"]
        rc = cd.main()
        assert rc == 130, rc
        status = json.load(open(os.path.join(tmp, "status.json"), encoding="utf-8"))
        assert status["summary"]["state"] == "interrupted" and status["summary"]["signal"] == "SIGINT", status
        assert open(os.path.join(tmp, "DONE"), encoding="utf-8").read().startswith("rc=130 ")
    finally:
        cd.RUN["summary"], sys.argv, handlers = saved
        for s, h in handlers.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok main：中断 → status interrupted + DONE rc=130")


def test_summary_warns_legacy_no_corridor():
    """卡 W1-1 #2 / W1-5 #4（W1-15b 第 6 条）：compare_datasets 不扩功能，输出必须提示「旧口径，未带走廊」。"""
    import contextlib
    import io
    with Fixture() as fx:
        # 1) 尚无 eval.json 时（含 dry-run）summary 打印 notes 也带该提示
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cd.summary(fx.tmp)
        assert "旧口径，未带走廊" in buf.getvalue(), buf.getvalue()
        # 2) 有 eval.json 写出 summary.md / summary.json 时两处都有该提示，且评测命令本身不带 --corridor-half-width
        for _d, _n, _s, cmd, _env, edir, _r in cd.eval_jobs(fx.tmp, check_ckpt=False):
            assert "--corridor-half-width" not in cmd, cmd
            _touch(os.path.join(edir, "eval.json"), json.dumps({
                "episodes_requested": 1, "episodes_done": 1, "failed_count": 0, "skipped_count": 0,
                "hold_s": 1.0, "metric_version": "sim09.v2.coast1s", "success_criterion": "goal_line",
                "dynamics": "cf_sitl", "vehicle": "e1_mini", "window_from": "observation", "task": "base",
                "per_episode": [{"seed": 1, "success": True, "collision": False, "stuck": False,
                                 "false_forward": False, "spl": 0.8}]}))
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            cd.summary(fx.tmp)
        md = open(os.path.join(fx.tmp, "summary.md"), encoding="utf-8").read()
        doc = json.load(open(os.path.join(fx.tmp, "summary.json"), encoding="utf-8"))
        assert "旧口径，未带走廊" in md, md
        assert any("旧口径，未带走廊" in n for n in doc["notes"]), doc["notes"]
    print("ok summary（第 6 条）：输出、summary.md、summary.json 均提示「旧口径，未带走廊」")


def test_d1_3dgs_allow_train_scene_and_teacher_eval_split():
    """D1：compare_datasets 的 3DGS 学生与 forward 评测命令显式带 --allow-train-scene（Gazebo 不带）；
    3DGS 教师采集命令不写死 --split test（让 collect_3dgs 按几何哈希写实际 split，与 teacher_3dgs_eval_json 一致）；
    teacher_3dgs_eval_json 用 eval_3dgs.eval_split(identity, True) 写入真实 split 与 split_policy。"""
    import importlib
    sys.path.insert(0, REPO)
    c3 = importlib.import_module("sim.3dgs.collect_3dgs")
    scene_mod = importlib.import_module("sim.3dgs.scene")
    with Fixture() as fx:
        cd.EVAL_DOMAINS = ("gazebo", "3dgs")
        _label, run_dir = cd.train_runs(fx.tmp, "A")[0]
        _touch(os.path.join(run_dir, "ckpt_best.bin"))
        jobs = cd.eval_jobs(fx.tmp)
        gz_cmds = [cmd for dom, _n, _s, cmd, _e, _ed, _r in jobs if dom == "gazebo"]
        gs_eval_cmds = [cmd for dom, n, _s, cmd, _e, _ed, _r in jobs if dom == "3dgs" and n != "teacher"]
        gs_teacher_cmds = [cmd for dom, n, _s, cmd, _e, _ed, _r in jobs if dom == "3dgs" and n == "teacher"]
        assert gz_cmds and gs_eval_cmds and len(gs_teacher_cmds) == 1
        assert all("--allow-train-scene" not in c for c in gz_cmds), gz_cmds
        assert all("--allow-train-scene" in c and "sim.3dgs.eval_3dgs" in c for c in gs_eval_cmds), gs_eval_cmds
        t_cmd = gs_teacher_cmds[0]
        assert "sim.3dgs.collect_3dgs" in t_cmd and "--stop-on-collision" in t_cmd, t_cmd
        assert "--split" not in t_cmd, f"教师采集不应硬编码 --split test（会与非 test 场景哈希冲突）：{t_cmd}"

        # 在哈希为 train（seed=3）与 test（seed=0）的合成走廊上实跑 collect_3dgs + teacher_3dgs_eval_json，核对两处一致
        py3dgs = cd.CFG["gs"]["python"]
        bb_json = os.path.join(REPO, "sim", "dynamics", "cf_blackbox_v1.json")
        for seed, want_split, want_policy in ((3, "train", "diag-train-scene"),
                                              (0, "test", "diag-eval-fixed-test")):
            sdir = os.path.join(fx.tmp, f"corridor_s{seed}")
            sc = scene_mod.make_corridor(sdir, seed=seed, length=12.0)
            sc["waypoint"] = {"x": 5.0, "y": 0.0, "z": 2.0}   # length=12 时默认 x=30 在走廊外，挪到走廊内无柱自由区
            sjson = os.path.join(sdir, "scene.json")
            scene_mod.save_scene(sc, sjson)
            assert c3.scene_identity(scene_mod.load_scene(sjson))["split"] == want_split
            cd.CFG["gs"] = dict(cd.CFG["gs"], scene=sjson, routes=(0,), episodes_per_route=1,
                                steps=3, hold=0.5, dynamics="cf_blackbox", dynamics_params=bb_json, spawn_jitter=0.0)
            cmd, env, edir = cd.eval_3dgs_cmd(fx.tmp, f"teacher_s{seed}", "teacher", None, 0)
            r = subprocess.run(cmd + ["--mock-renderer"], cwd=REPO, env=dict(os.environ, **env),
                               capture_output=True, text=True, timeout=120)
            assert r.returncode == 0, r.stderr
            run_meta = json.load(open(os.path.join(edir, "run.json"), encoding="utf-8"))
            ep_man = json.load(open(os.path.join(edir, "ep_00000", "episode_manifest.json"), encoding="utf-8"))
            rep = cd.teacher_3dgs_eval_json(edir, 0)
            assert run_meta["split"] == want_split and ep_man["split"] == want_split, (run_meta["split"], ep_man["split"])
            assert rep["scene"]["split"] == want_split, rep["scene"]
            assert rep["scene"].get("split_policy") == want_policy, rep["scene"]
    print("ok D1（compare_datasets）：3DGS 学生/forward 带 --allow-train-scene，教师采集与 eval.json 的 split / split_policy 一致")


def main():
    test_train_resume_by_metrics_done()
    test_evaluate_resume_and_waits_all_before_need()
    test_evaluate_retires_skipped_or_truncated_eval()
    test_main_records_interrupt_with_real_rc()
    test_summary_warns_legacy_no_corridor()
    test_d1_3dgs_allow_train_scene_and_teacher_eval_split()
    print("test_compare_datasets: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
