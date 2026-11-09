#!/usr/bin/env python3
"""tools/pipeline_common.py：三条流水线（e1_pipeline / dagger_pipeline / compare_datasets）共用的续跑判定与 helper。

AUDIT_SIMTOOLS Must 1/2：训练完成 = run/metrics.jsonl 末行 status done:（不是 ckpt 文件存在——trainer 每个 epoch 都写
ckpt.bin，中途被杀也有）；评测完成 = eval.json 存在且 failed_count == 0 且 episodes_done > 0（run_xeval.sh done_ok()）。

运行（仓库根）：.venv-tools/bin/python tools/test_pipeline_common.py
"""
import json
import os
import sys
import tempfile
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import pipeline_common as pc  # noqa: E402


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_train_finished():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        run = os.path.join(tmp, "run")
        assert pc.train_finished(run) is False, "目录不存在"
        _write(os.path.join(run, "ckpt.bin"), "x")
        assert pc.train_finished(run) is False, "只有 ckpt.bin（每轮都写）不算完成"
        _write(os.path.join(run, "metrics.jsonl"),
               '{"schema":"tinyvln.metrics.v1","event":"status","message":"start"}\n'
               '{"schema":"tinyvln.metrics.v1","event":"validation","step":10,"loss":0.9}\n')
        assert pc.train_finished(run) is False, "末行不是 done:"
        with open(os.path.join(run, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write('{"schema":"tinyvln.metrics.v1","event":"status","message":"done: 300960 samples in 102.6 s, ckpt=/x/ckpt.bin"}\n')
        assert pc.train_finished(run) is True
        with open(os.path.join(run, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write("\n\n")      # 末尾空行不影响
        assert pc.train_finished(run) is True
        _write(os.path.join(tmp, "run2", "metrics.jsonl"), "not json\n")
        assert pc.train_finished(os.path.join(tmp, "run2")) is False, "坏行不抛，算未完成"
    print("ok train_finished")


def test_eval_finished():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        edir = os.path.join(tmp, "eval-teacher")
        assert pc.eval_finished(edir) is False
        _write(os.path.join(edir, "eval.json"), json.dumps({"episodes_done": 20, "failed_count": 0}))
        assert pc.eval_finished(edir) is True
        _write(os.path.join(edir, "eval.json"), json.dumps({"episodes_done": 19, "failed_count": 1}))
        assert pc.eval_finished(edir) is False, "有失败集不算完成（run_xeval done_ok）"
        _write(os.path.join(edir, "eval.json"), json.dumps({"episodes_done": 0, "failed_count": 0}))
        assert pc.eval_finished(edir) is False, "0 集不算完成"
        _write(os.path.join(edir, "eval.json"), json.dumps({"episodes_done": 20, "failed_count": 0, "partial": True}))
        assert pc.eval_finished(edir) is False, "SIGTERM 写出的 partial 不算完成"
        _write(os.path.join(edir, "eval.json"), json.dumps({"per_episode": [{"seed": 1}], "failed": []}))
        assert pc.eval_finished(edir) is True, "旧 eval.json 没有 *_count 字段时按列表长度算"
        _write(os.path.join(edir, "eval.json"), "{broken")
        assert pc.eval_finished(edir) is False, "坏 JSON 不抛，算未完成"
    print("ok eval_finished")


def test_retire_unfinished_eval():
    """未完成的评测目录改名 <dir>.failed-<时间> 让出路径（eval_closed_loop 要求 --out 为空）。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        edir = os.path.join(tmp, "eval-forward")
        assert pc.retire_unfinished_eval(edir) is None, "目录不存在：什么都不做"
        _write(os.path.join(edir, "eval.json"), json.dumps({"episodes_done": 3, "failed_count": 2}))
        moved = pc.retire_unfinished_eval(edir)
        assert moved and os.path.basename(moved).startswith("eval-forward.failed-") and not os.path.exists(edir), moved
        assert os.path.isfile(os.path.join(moved, "eval.json"))
    print("ok retire_unfinished_eval")


def test_signal_handlers_and_child_groups():
    """SIGTERM 到流水线：run() 里等着的子进程组先收到 SIGTERM（给 eval_closed_loop 写部分 eval.json 的机会），再抛 Interrupted。"""
    import signal
    import threading
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        pc.install_signal_handlers()
        timer = threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            pc.run([sys.executable, "-c", "import time; time.sleep(30)"], grace_s=3.0)
        except pc.Interrupted as e:
            assert (e.signum, e.rc, e.name) == (signal.SIGTERM, 143, "SIGTERM"), (e.signum, e.rc, e.name)
        else:
            raise AssertionError("SIGTERM 期间 run() 应抛 Interrupted")
        finally:
            timer.cancel()
        assert pc.children() == [], "子进程组应已被 SIGTERM 收掉"
        # 直接起的子进程（evaluate 的并行组）也登记，terminate_children 一并收。
        procs = [pc.popen([sys.executable, "-c", "import time; time.sleep(30)"]) for _ in range(2)]
        assert len(pc.children()) == 2
        assert pc.terminate_children(grace_s=3.0) == 2
        assert all(p.returncode is not None for p in procs) and pc.children() == []
        # 不理会 SIGTERM 的子进程：宽限期后 SIGKILL。
        stubborn = pc.popen([sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"])
        time.sleep(0.5)
        assert pc.terminate_children(grace_s=1.0) == 1 and stubborn.returncode == -9, stubborn.returncode
        # Ctrl-C 同样走 Interrupted，rc=130。
        try:
            os.kill(os.getpid(), signal.SIGINT)
            time.sleep(0.5)
        except pc.Interrupted as e:
            assert e.rc == 130, e.rc
        else:
            raise AssertionError("SIGINT 应抛 Interrupted")
    finally:
        for s, h in saved.items():
            signal.signal(s, h)
    print("ok 信号处理：子进程组 SIGTERM → 宽限 → SIGKILL；Interrupted rc=128+signum")


def test_shared_helpers():
    """log/need/read_json/write_json/mark/wilson_ci 只在 pipeline_common 里有一份（A4），三条流水线直接用它的。"""
    import io
    import contextlib
    sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
    import e1_pipeline as ep
    import compare_datasets as cd
    import dagger_pipeline as dp
    import stats_report as sr
    import eval_closed_loop as ec
    for mod in (ep, cd, dp):
        for name in ("log", "need", "read_json", "write_json", "mark", "sh"):
            assert getattr(mod, name) is getattr(pc, name), f"{mod.__name__}.{name} 不是 pipeline_common 的那一份"
        assert not hasattr(mod, "wilson"), f"{mod.__name__} 还有自己的 wilson"
    assert sr.wilson_ci is pc.wilson_ci, "stats_report 应直接用 pipeline_common.wilson_ci"
    for k, n in ((0, 1), (3, 12), (11, 12), (20, 20), (7, 20)):
        lo, hi = pc.wilson_ci(k, n)
        assert abs(lo - ec.wilson_ci(k, n)[0]) < 1e-12 and abs(hi - ec.wilson_ci(k, n)[1]) < 1e-12, (k, n)
    assert pc.wilson_ci(0, 0) is None
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        saved = pc.DRY
        try:
            pc.DRY = False
            pc.mark(tmp, "pack", "ok", n=3)
            status = pc.read_json(os.path.join(tmp, "status.json"))
            assert status["pack"]["state"] == "ok" and status["pack"]["n"] == 3, status
            try:
                pc.need(os.path.join(tmp, "nope"), "提示")
            except RuntimeError as e:
                assert "nope" in str(e) and "提示" in str(e)
            else:
                raise AssertionError("真跑缺文件应抛")
            pc.DRY = True
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                pc.need(os.path.join(tmp, "nope"), "提示")      # dry-run 只提醒
                pc.mark(tmp, "train", "ok")                      # dry-run 不写
            assert "[dry]" in buf.getvalue() and "nope" in buf.getvalue(), buf.getvalue()
            assert "train" not in pc.read_json(os.path.join(tmp, "status.json"))
        finally:
            pc.DRY = saved
    print("ok 共享 helper 单一份；wilson_ci 与 eval_closed_loop 一致")


def main():
    test_train_finished()
    test_eval_finished()
    test_retire_unfinished_eval()
    test_signal_handlers_and_child_groups()
    test_shared_helpers()
    print("test_pipeline_common: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
