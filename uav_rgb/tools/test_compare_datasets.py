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


def main():
    test_train_resume_by_metrics_done()
    test_evaluate_resume_and_waits_all_before_need()
    test_main_records_interrupt_with_real_rc()
    print("test_compare_datasets: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
