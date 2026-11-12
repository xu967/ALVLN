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
    tmp = tempfile.mkdtemp(prefix="test_dagger_")

    def fake_sh(cmd, *_a, **_k):
        out = cmd[cmd.index("--out") + 1]
        assert not os.path.exists(out), f"{out} 半成品应先挪空"
        calls.append(cmd)
        _touch(os.path.join(out, "ckpt_best.bin"), "fresh")
        _touch(os.path.join(out, "metrics.jsonl"), DONE_LINE)

    dp.sh = fake_sh
    try:
        run_dir = os.path.join(tmp, "run_soft")
        _touch(os.path.join(run_dir, "ckpt.bin"), "half")
        # E-9：只有 ckpt.bin（无 done 行）不当训练完成，挪成 run_soft.partial-* 后重起 uavrgb_train
        assert dp.train(os.path.join(tmp, "ds"), run_dir, "soft") == os.path.join(run_dir, "ckpt_best.bin")
        assert len(calls) == 1, calls
        partials = [d for d in os.listdir(tmp) if d.startswith("run_soft.partial-")]
        assert len(partials) == 1 and open(os.path.join(tmp, partials[0], "ckpt.bin")).read() == "half"
        calls.clear()
        assert dp.train(os.path.join(tmp, "ds"), run_dir, "soft") == os.path.join(run_dir, "ckpt_best.bin")
        assert calls == [], calls
    finally:
        dp.sh, dp.DRY = saved_sh, saved_dry
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok train：半成品 run 挪 .partial-* 后重训，metrics.jsonl 有 done 行跳过")


def test_partial_collect_and_pack_retired_and_excluded_from_glob():
    """E-9：round_0 现采、collect_dagger 自采与 pack 打包的半成品目录改名 *.partial-<时刻> 后重做；
    good_episode_dirs 不扫进 route_*.partial-* 里的残余集；改名失败时停下报错，不在旧目录上重做。"""
    import pipeline_common as pc
    saved = (dp.sh, dp.DRY, dp.CFG["routes"], dp.CFG["round0_episodes"])
    dp.DRY = False
    dp.CFG["routes"] = (0, 1)
    dp.CFG["round0_episodes"] = None
    tmp = tempfile.mkdtemp(prefix="test_dagger_")
    ran = []

    def ep_manifest(path):
        _touch(os.path.join(path, "episode_manifest.json"), json.dumps(
            {"error": None, "dynamics_anomaly": False, "validity": {"ok": True}}))
        _touch(os.path.join(path, "actions.jsonl"), "{}\n")

    def fake_sh(cmd, *_a, **_k):
        if "sim.3dgs.collect_3dgs" in cmd:
            out = cmd[cmd.index("--out") + 1]
            assert not os.path.exists(out), f"{out} 半成品应先挪空（collect_3dgs 拒绝非空目录）"
            ran.append(("collect_3dgs", os.path.basename(out)))
            ep_manifest(os.path.join(out, "ep_00000"))
            _touch(os.path.join(out, "run.json"), "{}")
        elif any("preprocess.py" in c for c in cmd):
            ds = cmd[cmd.index("--out") + 1]
            assert not os.path.exists(ds), f"{ds} 半成品应先挪空"
            ran.append(("preprocess", os.path.basename(ds)))
            _touch(os.path.join(ds, "dataset.uavr"), "u")
        elif any("join_soft_targets.py" in c for c in cmd):
            out = cmd[cmd.index("--out") + 1]
            ran.append(("join", os.path.basename(out)))
            _touch(out, "s")

    dp.sh = fake_sh
    pc.lock_out(tmp)
    try:
        # round_0：route_0 已完成，route_1 是没有 run.json 的半成品（含旧 ep_00099）
        r0 = os.path.join(tmp, "round_0", "collect")
        ep_manifest(os.path.join(r0, "route_0", "ep_00000"))
        _touch(os.path.join(r0, "route_0", "run.json"), "{}")
        ep_manifest(os.path.join(r0, "route_1", "ep_00099"))
        dirs = dp.round0_dirs(tmp)
        assert ran == [("collect_3dgs", "route_1")], ran
        assert any(d.startswith("route_1.partial-") for d in os.listdir(r0)), os.listdir(r0)
        assert [os.path.relpath(d, r0) for d in dirs] == ["route_0/ep_00000", "route_1/ep_00000"], \
            f"route_1.partial-* 里的 ep_00099 不能混进训练集：{dirs}"

        # collect_dagger + pack 半成品
        ran.clear()
        c1 = os.path.join(tmp, "round_1", "collect")
        _touch(os.path.join(c1, "route_0", "run.json"), "{}")
        ep_manifest(os.path.join(c1, "route_0", "ep_00000"))
        ep_manifest(os.path.join(c1, "route_1", "ep_00088"))   # 无 run.json → 半成品
        ds1 = os.path.join(tmp, "round_1", "ds")
        _touch(os.path.join(ds1, "dataset.uavr"), "only-uavr-no-soft")   # 缺 soft_targets.bin → 半成品
        dp.collect_dagger(1, "ck.bin", 0.5, c1)
        new_dirs = dp.good_episode_dirs(c1)
        assert [os.path.relpath(d, c1) for d in new_dirs] == ["route_0/ep_00000", "route_1/ep_00000"], new_dirs
        dp.pack(dirs + new_dirs, ["a.npz"], ds1)
        assert ran == [("collect_3dgs", "route_1"), ("preprocess", "ds"), ("join", "soft_targets.bin")], ran
        assert any(d.startswith("ds.partial-") for d in os.listdir(os.path.join(tmp, "round_1")))
        assert [(r["kind"], r["path"]) for r in pc.read_retired(tmp)] == [
            ("partial", "round_0/collect/route_1"),
            ("partial", "round_1/collect/route_1"),
            ("partial", "round_1/ds")], pc.read_retired(tmp)

        # 改名失败：停下报错，不调 collect_3dgs 或 uavrgb_train
        ran.clear()
        _touch(os.path.join(tmp, "busy_c", "route_0", "ep_00000", "x"), "keep")
        _touch(os.path.join(tmp, "busy_run", "ckpt.bin"), "keep")
        real_rename = pc.os.rename
        pc.os.rename = lambda *_a: (_ for _ in ()).throw(PermissionError("denied"))
        try:
            for fn in (lambda: dp.collect_dagger(1, "ck.bin", 0.5, os.path.join(tmp, "busy_c")),
                       lambda: dp.train(ds1, os.path.join(tmp, "busy_run"), "soft")):
                try:
                    fn()
                except RuntimeError as e:
                    assert "denied" in str(e), e
                else:
                    raise AssertionError("改名失败应停下报错")
        finally:
            pc.os.rename = real_rename
        assert ran == [], f"改名失败不该重做：{ran}"
    finally:
        dp.sh, dp.DRY, dp.CFG["routes"], dp.CFG["round0_episodes"] = saved
        pc.unlock_out()
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok E-9（dagger_pipeline）：采集 / 打包 / 训练半成品挪 .partial-*，glob 排除半成品，改名失败停下")


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


def test_evaluate_retires_stale_and_truncated_and_lists_in_summary():
    """E-8：3DGS 评测同样走统一判据。ckpt 或 metric_version 对不上的路线挪 .stale-*，被跳过/截短的路线挪 .failed-*，
    一致的跳过；台账记进 retired_dirs.jsonl 供汇总列出。"""
    import pipeline_common as pc
    saved_run, saved_dry, saved_routes, saved_eps = dp.run, dp.DRY, dp.CFG["routes"], dp.CFG["eval_eps_per_route"]
    dp.DRY = False
    dp.CFG["routes"] = (0, 1, 2)
    dp.CFG["eval_eps_per_route"] = 2
    tmp = tempfile.mkdtemp(prefix="test_dagger_")
    ran = []
    pc.lock_out(tmp)
    try:
        ckpt_now, ckpt_old = os.path.join(tmp, "now.bin"), os.path.join(tmp, "old.bin")
        _touch(ckpt_now, "n")
        _touch(ckpt_old, "o")
        ep_row = {"success": True, "task_success": True, "collision": False, "stuck": False, "steps": 4}

        def put(r, **over):
            doc = {"episodes_requested": 2, "episodes_done": 2, "failed_count": 0, "skipped_count": 0,
                   "failed": [], "metric_version": "3dgs.pilot.v2", "ckpt": ckpt_now, "per_episode": [ep_row, ep_row]}
            doc.update(over)
            _touch(os.path.join(tmp, "eval", f"route_{r}", "eval.json"), json.dumps(doc))

        put(0)                                      # 一致 → 跳过
        put(1, ckpt=ckpt_old)                       # 旧 ckpt → .stale-
        put(2, episodes_done=1, skipped_count=1)    # 跳过截短 → .failed-

        def fake_run(cmd, *_a, **_k):
            edir = cmd[cmd.index("--out") + 1]
            assert not os.path.exists(edir), f"{edir} 应先挪空"
            r = int(os.path.basename(edir).split("_")[1])
            ran.append(r)
            put(r)
            return 0

        dp.run = fake_run
        dp.evaluate(ckpt_now, os.path.join(tmp, "eval"))
        assert ran == [1, 2], f"应只重评 route_1 与 route_2，实际 {ran}"
        eval_names = os.listdir(os.path.join(tmp, "eval"))
        assert any(d.startswith("route_1.stale-") for d in eval_names), eval_names
        assert any(d.startswith("route_2.failed-") for d in eval_names), eval_names
        rows = pc.read_retired(tmp)
        assert [(x["kind"], x["path"]) for x in rows] == [("stale", "eval/route_1"), ("failed", "eval/route_2")], rows
    finally:
        dp.run, dp.DRY, dp.CFG["routes"], dp.CFG["eval_eps_per_route"] = saved_run, saved_dry, saved_routes, saved_eps
        pc.unlock_out()
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok evaluate（E-8）：旧 ckpt → .stale-，截断跳过 → .failed-，一致跳过，记台账")


def test_evaluate_sr_uses_requested_denominator():
    """E-7：evaluate 的成功率与成功类 Wilson 区间分母用请求集数（physics_verified=false 剔除的失败集计入分母、成功记 0），
    不能只除完成集而虚高；碰撞率分母仍为完成集（与 W1-2 / stats_report 同口径），failed 单列。"""
    import pipeline_common as pc
    saved = (dp.run, dp.DRY, dp.CFG["routes"], dp.CFG["eval_eps_per_route"])
    dp.DRY = False
    dp.CFG["routes"] = (0, 1)
    dp.CFG["eval_eps_per_route"] = 3   # 2 条路线 × 3 集 = 请求 6 集
    tmp = tempfile.mkdtemp(prefix="test_dagger_")

    def fake_run(cmd, *_a, **_k):
        edir = cmd[cmd.index("--out") + 1]
        r = int(os.path.basename(edir).split("_")[1])
        if r == 0:
            # 3 集全完成：2 成功，1 碰撞
            doc = {"episodes_requested": 3, "episodes_done": 3, "failed_count": 0, "failed": [],
                   "per_episode": [
                       {"success": True, "task_success": True, "collision": False, "stuck": False, "steps": 5},
                       {"success": True, "task_success": False, "collision": False, "stuck": False, "steps": 6},
                       {"success": False, "task_success": False, "collision": True, "stuck": False, "steps": 4}]}
            rc = 0
        else:
            # 请求 3 集，仅完成 1 集（成功），2 集因 physics_verified=false 被剔除进 failed
            doc = {"episodes_requested": 3, "episodes_done": 1, "failed_count": 2,
                   "failed": [{"episode": 1, "error": "physics_verified=false"},
                              {"episode": 2, "error": "physics_verified=false"}],
                   "per_episode": [
                       {"success": True, "task_success": True, "collision": False, "stuck": False, "steps": 5}]}
            rc = 1
        _touch(os.path.join(edir, "eval.json"), json.dumps(doc))
        return rc

    dp.run = fake_run
    try:
        rep = dp.evaluate("ckpt.bin", os.path.join(tmp, "eval"))
        # 请求 6 集，完成 4 集（3 成功，2 task_success，1 碰撞），失败剔除 2 集：
        # 旧代码 n=4, SR=3/4=0.75；修复后 n=6（请求集数），SR=3/6=0.5，CR=1/4=0.25（完成集分母）
        assert rep["n"] == 6 and rep["episodes_done"] == 4 and rep["failed"] == 2, rep
        assert abs(rep["success_rate"] - 3 / 6) < 1e-12, rep["success_rate"]
        assert abs(rep["task_success_rate"] - 2 / 6) < 1e-12, rep.get("task_success_rate")
        assert abs(rep["collision_rate"] - 1 / 4) < 1e-12, rep["collision_rate"]
        want_sr_ci = [round(x, 3) for x in pc.wilson_ci(3, 6)]
        want_task_ci = [round(x, 3) for x in pc.wilson_ci(2, 6)]
        want_cr_ci = [round(x, 3) for x in pc.wilson_ci(1, 4)]
        assert rep["success_ci95"] == want_sr_ci, (rep["success_ci95"], want_sr_ci)
        assert rep["task_success_ci95"] == want_task_ci, (rep["task_success_ci95"], want_task_ci)
        assert rep["collision_ci95"] == want_cr_ci, (rep["collision_ci95"], want_cr_ci)
        assert rep["per_route"]["1"]["n"] == 3 and rep["per_route"]["1"]["episodes_done"] == 1 and rep["per_route"]["1"]["failed"] == 2, rep["per_route"]["1"]
    finally:
        dp.run, dp.DRY, dp.CFG["routes"], dp.CFG["eval_eps_per_route"] = saved
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok E-7：dagger_pipeline 成功率与成功类 Wilson 区间分母为请求集数，failed 单列")


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
        assert os.path.exists(os.path.join(tmp, ".pipeline.lock")), "E-3：main 起跑应拿过实例锁"
        _hold_lock(tmp).close()   # E-3：收尾（DONE 写完）已放锁，同目录马上能再起
        rec = dict(ln.split("=", 1) for ln in open(os.path.join(tmp, "run.pid"), encoding="utf-8").read().splitlines())
        assert rec["pid"] == str(os.getpid()) and rec["start"] == pc.proc_stat_field(os.getpid(), 22), rec   # E-4：run.pid 带启动时刻
    finally:
        dp.need, dp.round0_dirs, dp.torch_extensions_dir, sys.argv, handlers = saved
        for s, h in handlers.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok main：中断 → status interrupted + DONE rc=143")


def _hold_lock(out):
    """另一个实例占着输出目录：按约定的锁文件名 <out>/.pipeline.lock 加 flock。返回文件对象，关掉就放锁；已被占会抛 BlockingIOError。"""
    import fcntl
    os.makedirs(out, exist_ok=True)
    f = open(os.path.join(out, ".pipeline.lock"), "a+", encoding="utf-8")
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return f


def test_main_second_instance_exits_before_touching_out():
    """E-3：输出目录已被另一个实例占着时，main 在写任何文件之前退出 3：config.json、DONE、run.pid 原样，不去问 torch、不进 preflight。"""
    import signal
    tmp = tempfile.mkdtemp(prefix="test_dagger_")
    holder = _hold_lock(tmp)
    saved = (dp.need, dp.round0_dirs, dp.torch_extensions_dir, sys.argv,
             {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)})

    def boom(*_a, **_k):
        raise AssertionError("拿不到锁却走到了后面的步骤")

    try:
        _touch(os.path.join(tmp, "DONE"), "rc=0 first-instance\n")
        _touch(os.path.join(tmp, "config.json"), '{"first": true}')
        _touch(os.path.join(tmp, "run.pid"), "pid=1\npgid=1\nstart=1\ncmd=first-instance\n")
        dp.need, dp.round0_dirs, dp.torch_extensions_dir = boom, boom, boom
        sys.argv = ["dagger_pipeline.py", "--out", tmp, "--rounds", "0"]
        try:
            dp.main()
        except SystemExit as e:
            assert e.code == 3, e.code
        else:
            raise AssertionError("输出目录被占着，main 应退出 3")
        assert sorted(os.listdir(tmp)) == [".pipeline.lock", "DONE", "config.json", "run.pid"], os.listdir(tmp)
        assert open(os.path.join(tmp, "DONE"), encoding="utf-8").read() == "rc=0 first-instance\n"
        assert json.load(open(os.path.join(tmp, "config.json"), encoding="utf-8")) == {"first": True}
        assert open(os.path.join(tmp, "run.pid"), encoding="utf-8").read() == "pid=1\npgid=1\nstart=1\ncmd=first-instance\n"
    finally:
        holder.close()
        dp.need, dp.round0_dirs, dp.torch_extensions_dir, sys.argv, handlers = saved
        for s, h in handlers.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok main：输出目录被占 → 退出 3，config.json / DONE 原样，不进任何步骤")


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


def test_torch_extensions_dir_uses_pipeline_common():
    """缓存目录规则只在 pipeline_common.torch_extensions_dir：这里只补默认的 PY3DGS；问不到返回 None（main 不设，用 torch 默认缓存）。"""
    import pipeline_common as pc
    assert dp.torch_extensions_dir((7, 5)) == pc.torch_extensions_dir((7, 5))
    calls, saved = [], pc.torch_extensions_dir
    pc.torch_extensions_dir = lambda capability=None, python=None: calls.append((capability, python))
    try:
        assert dp.torch_extensions_dir() is None
        assert dp.torch_extensions_dir(python="/x/python") is None
    finally:
        pc.torch_extensions_dir = saved
    assert calls == [(None, dp.PY3DGS), (None, "/x/python")], calls
    print("ok torch_extensions_dir 走 pipeline_common：默认问 PY3DGS，问不到返回 None")


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
    test_partial_collect_and_pack_retired_and_excluded_from_glob()
    test_evaluate_resume_by_eval_finished()
    test_evaluate_retires_stale_and_truncated_and_lists_in_summary()
    test_evaluate_sr_uses_requested_denominator()
    test_main_records_interrupt_with_real_rc()
    test_main_second_instance_exits_before_touching_out()
    test_no_dead_gazebo_branch_and_no_hardcoded_paths()
    test_torch_extensions_dir_uses_pipeline_common()
    test_dry_run_creates_nothing()
    print("test_dagger_pipeline: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
