#!/usr/bin/env python3
"""e1_pipeline 的不起 gz 单测。

1. evaluate 的等待顺序：四组各带 1 个 gz 并行跑，一组没写出 eval.json 时必须等其余几组
   跑完再报错。中途抛出会让 main 写 DONE(rc=1) 退出，剩下几组的 gz 还在跑；续跑 --from eval 会再起 4 个。
2. summary：各组 eval.json 的 metric_version 不同（sim09.v1 过线后继续飞 / sim09.v2 成功线即停）时表头必须标出。

运行（仓库根）：.venv-tools/bin/python tools/test_e1_pipeline.py
"""
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import e1_pipeline as ep  # noqa: E402


class FakeProc:
    """替身子进程：wait() 记录被等过；write_eval=False 的组模拟 eval_closed_loop 崩在写 eval.json 之前。"""
    waited = []

    def __init__(self, name, edir, write_eval):
        self.name, self.edir, self.write_eval, self.returncode = name, edir, write_eval, None

    def poll(self):   # pipeline_common.children() 用它判活
        return self.returncode

    def wait(self):
        FakeProc.waited.append(self.name)
        if self.write_eval:
            os.makedirs(self.edir, exist_ok=True)
            with open(os.path.join(self.edir, "eval.json"), "w", encoding="utf-8") as f:
                json.dump({"episodes_done": 0}, f)
            self.returncode = 0
        else:
            self.returncode = 1
        return self.returncode


def test_evaluate_waits_all_groups():
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    real_popen, real_ckpt = subprocess.Popen, ep.CFG["old_x3_ckpt"]
    try:
        for rel in ("old_x3.bin", os.path.join("run", "ckpt_best.bin")):   # evaluate 开头 need() 的两个 ckpt
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)) or tmp, exist_ok=True)
            open(os.path.join(tmp, rel), "wb").close()
        ep.CFG["old_x3_ckpt"] = os.path.join(tmp, "old_x3.bin")
        groups = [g[0] for g in ep.eval_groups()]
        broken = groups[0]   # 第一个被 wait 的组没写 eval.json

        def fake_popen(cmd, **_kw):
            edir = cmd[cmd.index("--out") + 1]
            name = os.path.basename(edir)[len("eval-"):]
            return FakeProc(name, edir, write_eval=(name != broken))

        subprocess.Popen = fake_popen
        try:
            ep.evaluate(tmp)
        except RuntimeError as e:
            err = str(e)
        else:
            raise AssertionError("有一组没写 eval.json，evaluate 应报错")
        assert broken in err, err
        assert FakeProc.waited == groups, f"报错前没等齐所有组：等了 {FakeProc.waited}，应等 {groups}"
    finally:
        subprocess.Popen, ep.CFG["old_x3_ckpt"] = real_popen, real_ckpt
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"ok evaluate：{len(groups)} 组，{broken} 缺 eval.json 时其余组仍被等完")


def _eval_json(out, name, metric_version, sr=0.5):
    os.makedirs(os.path.join(out, f"eval-{name}"), exist_ok=True)
    doc = {"success_rate": sr, "requested_success_rate": sr, "collision_rate": 0.1, "stuck_rate": 0.0,
           "false_forward_rate": 0.0, "stable_termination_rate": None, "episodes_done": 20, "failed_count": 0,
           "skipped_count": 0, "metric_version": metric_version, "dynamics": "cf_sitl", "hold_s": 1.0,
           "steps": 80, "ood": False, "wilson_ci": {"requested": [0.3, 0.7]}, "inference_ms": {"p50": 3.0, "p95": 5.0}}
    with open(os.path.join(out, f"eval-{name}", "eval.json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)


def test_summary_marks_mixed_metric_version():
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    try:
        groups = [g[0] for g in ep.eval_groups()]
        for i, name in enumerate(groups):
            _eval_json(tmp, name, "sim09.v1.coast1s" if i == 0 else "sim09.v2.coast1s")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ep.summary(tmp)
        printed = buf.getvalue()
        md = open(os.path.join(tmp, "summary.md"), encoding="utf-8").read()
        for text, where in ((printed, "stdout"), (md, "summary.md")):
            assert "sim09.v1.coast1s" in text and "sim09.v2.coast1s" in text, f"{where} 没标出混合版本:\n{text}"
            assert "metric_version" in text, f"{where} 表头没有 metric_version 列:\n{text}"
        # 每行都带自己的版本，读者能分清哪组是哪种口径。
        assert any(groups[0] in line and "sim09.v1.coast1s" in line for line in printed.splitlines()), printed
        # 全部同版本：不报混合。
        for name in groups:
            _eval_json(tmp, name, "sim09.v2.coast1s")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ep.summary(tmp)
        assert "混合" not in buf.getvalue(), buf.getvalue()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok summary：metric_version 混合时表头标出")


DONE_LINE = '{"schema":"tinyvln.metrics.v1","event":"status","message":"done: 300960 samples in 102.6 s, ckpt=x"}\n'


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_train_resume_by_metrics_done():
    """训练完成的判据是 metrics.jsonl 末行 done:，不是 ckpt 文件在（trainer 每轮都写 ckpt.bin，被杀的 run 也有）。"""
    real_sh = ep.sh
    calls = []
    ep.sh = lambda cmd, *a, **k: calls.append(cmd)
    try:
        tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
        _touch(os.path.join(tmp, "ds", "dataset.uavr"))
        _touch(os.path.join(tmp, "run", "ckpt_best.bin"))
        try:
            ep.train(tmp)
        except RuntimeError as e:
            assert "非空" in str(e), e
        else:
            raise AssertionError("run/ 非空但没跑完（没有 done 行）应报错，不能当训练完成")
        _touch(os.path.join(tmp, "run", "metrics.jsonl"), DONE_LINE)
        ep.train(tmp)      # 完成 → 跳过
        assert calls == [], f"训练已完成不该再起 uavrgb_train：{calls}"
    finally:
        ep.sh = real_sh
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok train：按 metrics.jsonl done 行续跑")


def test_evaluate_resume_by_eval_finished():
    """eval.json 在但 failed_count>0 的组不算完成：目录改名 .failed-* 后重跑；完成的组跳过。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    real_popen, real_ckpt = subprocess.Popen, ep.CFG["old_x3_ckpt"]
    FakeProc.waited = []
    started = []
    try:
        for rel in ("old_x3.bin", os.path.join("run", "ckpt_best.bin")):
            _touch(os.path.join(tmp, rel))
        ep.CFG["old_x3_ckpt"] = os.path.join(tmp, "old_x3.bin")
        groups = [g[0] for g in ep.eval_groups()]
        finished, broken = groups[0], groups[1]
        _touch(os.path.join(tmp, f"eval-{finished}", "eval.json"), json.dumps({"episodes_done": 20, "failed_count": 0}))
        _touch(os.path.join(tmp, f"eval-{broken}", "eval.json"), json.dumps({"episodes_done": 19, "failed_count": 1}))

        def fake_popen(cmd, **_kw):
            edir = cmd[cmd.index("--out") + 1]
            name = os.path.basename(edir)[len("eval-"):]
            started.append(name)
            assert not os.path.exists(edir), f"{edir} 应先让出来（eval_closed_loop 要求 --out 为空）"
            return FakeProc(name, edir, write_eval=True)

        subprocess.Popen = fake_popen
        ep.evaluate(tmp)
        assert finished not in started, f"完成的组被重跑：{started}"
        assert broken in started, f"failed_count>0 的组没重跑：{started}"
        retired = [d for d in os.listdir(tmp) if d.startswith(f"eval-{broken}.failed-")]
        assert len(retired) == 1 and os.path.isfile(os.path.join(tmp, retired[0], "eval.json")), os.listdir(tmp)
    finally:
        subprocess.Popen, ep.CFG["old_x3_ckpt"] = real_popen, real_ckpt
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok evaluate：按 eval_finished 续跑，未完成目录改名重跑")


def test_main_records_interrupt_with_real_rc():
    """SIGTERM/Ctrl-C：status.json 该步标 interrupted，DONE 写真实 rc（以前 Ctrl-C 写 DONE rc=0）。"""
    import signal
    import pipeline_common as pc
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    saved = (ep.RUN["summary"], ep.preflight, sys.argv, {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)})

    def interrupted(_out):
        raise pc.Interrupted(signal.SIGTERM)

    try:
        ep.RUN["summary"], ep.preflight = interrupted, lambda: None
        sys.argv = ["e1_pipeline.py", "--out", tmp, "--from", "summary"]
        rc = ep.main()
        assert rc == 143, rc
        status = json.load(open(os.path.join(tmp, "status.json"), encoding="utf-8"))
        assert status["summary"]["state"] == "interrupted" and status["summary"]["signal"] == "SIGTERM", status
        done = open(os.path.join(tmp, "DONE"), encoding="utf-8").read()
        assert done.startswith("rc=143 "), done
        for s in (signal.SIGTERM, signal.SIGINT):
            assert signal.getsignal(s) not in (signal.SIG_DFL, None), "main 应装 SIGTERM/SIGINT 处理器"
    finally:
        ep.RUN["summary"], ep.preflight, sys.argv, handlers = saved
        for s, h in handlers.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok main：中断 → status interrupted + DONE rc=143")


def test_dry_run_creates_nothing():
    """A6：--dry-run 只打印命令，不建 --out、不写 config.json/status.json/DONE（以前 main 一进来就 makedirs + 写 config）。"""
    import signal
    import pipeline_common as pc
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    out = os.path.join(tmp, "never")
    saved = (sys.argv, {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)})
    try:
        sys.argv = ["e1_pipeline.py", "--out", out, "--dry-run"]
        rc = ep.main()
        assert rc == 0, rc
        assert not os.path.exists(out), os.listdir(out)
        assert os.listdir(tmp) == [], os.listdir(tmp)
    finally:
        sys.argv, handlers = saved
        pc.DRY = ep.DRY = False
        for s, h in handlers.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok --dry-run 不建目录不写文件")


def test_teacher_planner_passthrough():
    """CFG teacher_planner：默认 astar 时采集 / 评测命令行逐字不变；costmap 时采集与 teacher 组带 --teacher-planner，学生组不带。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    real_run, real_popen, real_ckpt = ep.run, subprocess.Popen, ep.CFG["old_x3_ckpt"]
    real_planner, real_dry = ep.CFG.get("teacher_planner"), ep.DRY
    seen = {}
    try:
        ep.run = lambda cmd, **_kw: seen.setdefault("collect", []).append(cmd) or 0
        ep.DRY = True
        for planner in ("astar", "costmap"):
            ep.CFG["teacher_planner"] = planner
            ep.collect_into(os.path.join(tmp, f"col-{planner}"), 4, 3100)
        a, c = seen["collect"]
        assert "--teacher-planner" not in a, a
        assert c[c.index("--teacher-planner") + 1] == "costmap", c
        same_out = lambda cmd: [("OUT" if i and cmd[i - 1] == "--out" else x) for i, x in enumerate(cmd)]
        assert [x for x in same_out(c) if x not in ("--teacher-planner", "costmap")] == same_out(a), "costmap 只多这两项"
        ep.DRY = False
        for rel in ("old_x3.bin", os.path.join("run", "ckpt_best.bin")):
            os.makedirs(os.path.dirname(os.path.join(tmp, rel)) or tmp, exist_ok=True)
            open(os.path.join(tmp, rel), "wb").close()
        ep.CFG["old_x3_ckpt"] = os.path.join(tmp, "old_x3.bin")
        cmds = {}

        def fake_popen(cmd, **_kw):
            edir = cmd[cmd.index("--out") + 1]
            name = os.path.basename(edir)[len("eval-"):]
            cmds[name] = cmd
            return FakeProc(name, edir, write_eval=True)

        subprocess.Popen = fake_popen
        ep.evaluate(tmp)
        for name, cmd in cmds.items():
            policy = cmd[cmd.index("--policy") + 1]
            if policy == "teacher":
                assert cmd[cmd.index("--teacher-planner") + 1] == "costmap", (name, cmd)
            else:
                assert "--teacher-planner" not in cmd, (name, cmd)
    finally:
        ep.run, subprocess.Popen, ep.CFG["old_x3_ckpt"], ep.DRY = real_run, real_popen, real_ckpt, real_dry
        ep.CFG["teacher_planner"] = real_planner
        FakeProc.waited.clear()   # 类级记录：别让后面 test_evaluate_waits_all_groups 看到本测试等过的组
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok teacher_planner：astar 命令行不变，costmap 只给采集与 teacher 组加 --teacher-planner")


def main():
    test_teacher_planner_passthrough()
    test_evaluate_waits_all_groups()
    test_summary_marks_mixed_metric_version()
    test_train_resume_by_metrics_done()
    test_evaluate_resume_by_eval_finished()
    test_main_records_interrupt_with_real_rc()
    test_dry_run_creates_nothing()
    print("test_e1_pipeline: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
