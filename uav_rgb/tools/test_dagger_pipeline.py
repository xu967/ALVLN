#!/usr/bin/env python3
"""dagger_pipeline 的不跑 GPU 单测：续跑判定（AUDIT Must 1/2）、dry-run 不建目录（A6）、无死代码 gazebo_commands。

运行（仓库根）：.venv-tools/bin/python tools/test_dagger_pipeline.py
"""
import json
import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import dagger_pipeline as dp  # noqa: E402

DONE_LINE = '{"schema":"tinyvln.metrics.v1","event":"status","message":"done: 1 samples in 1 s, ckpt=x"}\n'


def _touch(path, text=""):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def test_train_resume_by_metrics_done():
    saved_sh, saved_dry = dp.sh, dp.DRY
    dp.DRY = False
    calls = []
    dp.sh = lambda cmd, *a, **k: calls.append(cmd)
    tmp = tempfile.mkdtemp(prefix="test_dagger_")
    try:
        run_dir = os.path.join(tmp, "run_soft")
        _touch(os.path.join(run_dir, "ckpt.bin"))
        try:
            dp.train(os.path.join(tmp, "ds"), run_dir, "soft")
        except RuntimeError as e:
            assert "非空" in str(e), e
        else:
            raise AssertionError("只有 ckpt.bin（每轮都写）不是训练完成，run 目录非空应报错")
        _touch(os.path.join(run_dir, "metrics.jsonl"), DONE_LINE)
        _touch(os.path.join(run_dir, "ckpt_best.bin"))
        assert dp.train(os.path.join(tmp, "ds"), run_dir, "soft") == os.path.join(run_dir, "ckpt_best.bin")
        assert calls == [], calls
    finally:
        dp.sh, dp.DRY = saved_sh, saved_dry
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok train：按 metrics.jsonl done 行续跑")


def test_evaluate_resume_by_eval_finished():
    saved_run, saved_dry, saved_routes = dp.run, dp.DRY, dp.CFG["routes"]
    dp.DRY = False
    dp.CFG["routes"] = (0, 1)
    tmp = tempfile.mkdtemp(prefix="test_dagger_")
    ran = []

    def fake_run(cmd, *_a, **_k):
        edir = cmd[cmd.index("--out") + 1]
        ran.append(os.path.basename(edir))
        assert not os.path.exists(edir), f"{edir} 应先让出来"
        _touch(os.path.join(edir, "eval.json"), json.dumps(
            {"episodes_done": 1, "failed_count": 0, "failed": [],
             "per_episode": [{"success": True, "task_success": False, "collision": False, "stuck": False, "steps": 5}]}))
        return 0

    dp.run = fake_run
    try:
        _touch(os.path.join(tmp, "route_0", "eval.json"), json.dumps(
            {"episodes_done": 1, "failed_count": 0, "failed": [],
             "per_episode": [{"success": False, "task_success": False, "collision": True, "stuck": False, "steps": 3}]}))
        _touch(os.path.join(tmp, "route_1", "eval.json"), json.dumps(
            {"episodes_done": 2, "failed_count": 1, "failed": [{"episode": 0}],
             "per_episode": [{"success": True, "task_success": False, "collision": False, "stuck": False, "steps": 3}]}))
        rep = dp.evaluate("ckpt.bin", tmp)
        assert ran == ["route_1"], f"完成的 route_0 应跳过、有失败集的 route_1 应重跑：{ran}"
        assert any(d.startswith("route_1.failed-") for d in os.listdir(tmp)), os.listdir(tmp)
        assert rep["n"] == 2 and rep["success"] == 1 and rep["failed"] == 0, rep
    finally:
        dp.run, dp.DRY, dp.CFG["routes"] = saved_run, saved_dry, saved_routes
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok evaluate：按 eval_finished 续跑，未完成目录改名重跑")


def test_main_records_interrupt_with_real_rc():
    import signal
    import pipeline_common as pc
    tmp = tempfile.mkdtemp(prefix="test_dagger_")
    saved = (dp.need, dp.round0_dirs, dp.torch_extensions_dir, sys.argv,
             {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)})

    def interrupted(_out):
        raise pc.Interrupted(signal.SIGTERM)

    try:
        dp.need, dp.round0_dirs = (lambda *a, **k: None), interrupted
        dp.torch_extensions_dir = lambda *a, **k: None     # 不去问 3DGS venv 的 torch
        sys.argv = ["dagger_pipeline.py", "--out", tmp, "--rounds", "0"]
        rc = dp.main()
        assert rc == 143, rc
        status = json.load(open(os.path.join(tmp, "status.json"), encoding="utf-8"))
        assert status["preflight"]["state"] == "interrupted" and status["preflight"]["signal"] == "SIGTERM", status
        assert open(os.path.join(tmp, "DONE"), encoding="utf-8").read().startswith("rc=143 ")
    finally:
        dp.need, dp.round0_dirs, dp.torch_extensions_dir, sys.argv, handlers = saved
        for s, h in handlers.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok main：中断 → status interrupted + DONE rc=143")


def test_no_dead_gazebo_branch_and_no_hardcoded_paths():
    """A4：死代码 gazebo_commands 删掉（--domain 旗标保留，gazebo 值直接报错指路）；绝对路径 / sm89 缓存名不再写死。"""
    assert not hasattr(dp, "gazebo_commands"), "gazebo_commands 是死代码，应删"
    assert dp.CFG["dynamics_params"].startswith(dp.REPO), dp.CFG["dynamics_params"]
    assert os.path.isfile(dp.CFG["dynamics_params"]), dp.CFG["dynamics_params"]
    assert dp.torch_extensions_dir(capability=(7, 5)).endswith("extensions-sm75")
    assert dp.torch_extensions_dir(capability=(8, 9)).endswith("extensions-sm89")
    assert dp.torch_extensions_dir(capability=None, python="/nonexistent/python") is None, "问不到 torch 就不设"
    src = open(dp.__file__, encoding="utf-8").read()
    assert "/root/claude-code-backup" not in src and "extensions-sm89" not in src, "源码里不该再有写死的绝对路径 / sm89"
    saved_argv = sys.argv
    try:
        sys.argv = ["dagger_pipeline.py", "--out", "/nonexistent/x", "--domain", "gazebo", "--dry-run"]
        try:
            dp.main()
        except SystemExit as e:
            assert e.code not in (0, None), e.code
        else:
            raise AssertionError("--domain gazebo 应报错退出并指向 e1_pipeline / collect_parallel")
    finally:
        sys.argv = saved_argv
    print("ok 无 gazebo_commands 死代码；dynamics_params 相对仓库；缓存名按 GPU 计算能力")


def test_dry_run_creates_nothing():
    """A6：--dry-run 只打印命令，不建 --out / round_0、不写 config.json/summary.json/DONE、不去问 torch。"""
    import signal
    import pipeline_common as pc
    tmp = tempfile.mkdtemp(prefix="test_dagger_")
    out = os.path.join(tmp, "never")
    saved = (dp.torch_extensions_dir, sys.argv, dict(dp.CFG),
             {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)})

    def boom(*a, **k):
        raise AssertionError("dry-run 不该探测 GPU")

    try:
        dp.torch_extensions_dir = boom
        sys.argv = ["dagger_pipeline.py", "--out", out, "--dry-run", "--rounds", "0", "--collect-round0"]
        rc = dp.main()
        assert rc == 0, rc
        assert not os.path.exists(out), os.listdir(out)
        assert os.listdir(tmp) == [], os.listdir(tmp)
    finally:
        dp.torch_extensions_dir, sys.argv, cfg, handlers = saved
        dp.CFG.clear()
        dp.CFG.update(cfg)
        pc.DRY = dp.DRY = False
        for s, h in handlers.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok --dry-run 不建目录不写文件")


def main():
    test_train_resume_by_metrics_done()
    test_evaluate_resume_by_eval_finished()
    test_main_records_interrupt_with_real_rc()
    test_no_dead_gazebo_branch_and_no_hardcoded_paths()
    test_dry_run_creates_nothing()
    print("test_dagger_pipeline: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
