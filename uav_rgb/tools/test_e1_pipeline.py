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
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
import e1_pipeline as ep  # noqa: E402


DELIVERED = os.path.join("run", "ckpt_calswa.bin")   # evaluate 默认评的权重：末 10 轮平均 + 校准（W1-5）


@contextlib.contextmanager
def _cfg(**over):
    """临时改 CFG；退出时还原，原本没有的键删掉。也可当装饰器用。"""
    absent = object()
    saved = {k: ep.CFG.get(k, absent) for k in over}
    ep.CFG.update(over)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is absent:
                ep.CFG.pop(k, None)
            else:
                ep.CFG[k] = v


@contextlib.contextmanager
def _patched(obj, **attrs):
    saved = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


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
        for rel in ("old_x3.bin", DELIVERED):   # evaluate 开头 need() 的两个 ckpt（默认交付权重 calswa，W1-5）
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


@_cfg(deliver="best")   # 这条只管训练续跑，用 deliver=best 排除校准平均步（calswa 的续跑见 test_calswa_resume）
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
    """eval.json 在但 failed_count>0 的组、录像组中断后只剩 videos/ 的组都不算完成：目录改名 .failed-* 后重跑；
    完成的组跳过。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    real_popen, real_ckpt = subprocess.Popen, ep.CFG["old_x3_ckpt"]
    FakeProc.waited = []
    started, cmds = [], {}
    try:
        for rel in ("old_x3.bin", DELIVERED):
            _touch(os.path.join(tmp, rel))
        ep.CFG["old_x3_ckpt"] = os.path.join(tmp, "old_x3.bin")
        groups = [g[0] for g in ep.eval_groups()]
        finished, broken, recorded = groups[0], groups[1], ep.CFG["record_groups"][-1]
        assert recorded not in (finished, broken), (finished, broken, recorded)
        _touch(os.path.join(tmp, f"eval-{finished}", "eval.json"), json.dumps({"episodes_done": 20, "failed_count": 0}))
        _touch(os.path.join(tmp, f"eval-{broken}", "eval.json"), json.dumps({"episodes_done": 19, "failed_count": 1}))
        # eval_closed_loop 一启动就建 videos/，录像暂存帧在 videos/.frames_<seed>/；中断在写 eval.json 之前
        _touch(os.path.join(tmp, f"eval-{recorded}", "videos", ".frames_7", "chase_000001.png"))

        def fake_popen(cmd, **_kw):
            edir = cmd[cmd.index("--out") + 1]
            name = os.path.basename(edir)[len("eval-"):]
            started.append(name)
            cmds[name] = cmd
            assert not os.path.exists(edir), f"{edir} 应先让出来（eval_closed_loop 要求 --out 为空）"
            return FakeProc(name, edir, write_eval=True)

        subprocess.Popen = fake_popen
        ep.evaluate(tmp)
        assert finished not in started, f"完成的组被重跑：{started}"
        assert broken in started, f"failed_count>0 的组没重跑：{started}"
        retired = [d for d in os.listdir(tmp) if d.startswith(f"eval-{broken}.failed-")]
        assert len(retired) == 1 and os.path.isfile(os.path.join(tmp, retired[0], "eval.json")), os.listdir(tmp)
        assert recorded in started and "--record-video" in cmds[recorded], f"只剩 videos/ 的录像组没重跑：{started}"
        retired = [d for d in os.listdir(tmp) if d.startswith(f"eval-{recorded}.failed-")]
        assert len(retired) == 1 and os.path.isdir(os.path.join(tmp, retired[0], "videos")), os.listdir(tmp)
    finally:
        subprocess.Popen, ep.CFG["old_x3_ckpt"] = real_popen, real_ckpt
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok evaluate：按 eval_finished 续跑，未完成目录（含只剩 videos/ 的录像组）改名重跑")


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
        for rel in ("old_x3.bin", DELIVERED):
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


def _dry_run_eval_cmds(out, **env_extra):
    """子进程跑 `e1_pipeline.py --from eval --dry-run`，返回打印出的每条评测命令（token 列表）。
    CFG 在 import 时读环境变量，默认值与环境变量覆盖只能在新进程里验证。--out 由调用方给，几次调用同一个路径才能逐字比较。"""
    env = {k: v for k, v in os.environ.items() if k != "E1_CORRIDOR_HALF_WIDTH"}
    env.update(env_extra)
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "e1_pipeline.py"),
                        "--out", out, "--from", "eval", "--dry-run"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return [line.split(" $ ", 1)[1].split("  > ", 1)[0].split()
            for line in r.stdout.splitlines() if "eval_closed_loop.py" in line]


def _without_pair(cmd, flag):
    """去掉命令里的 `flag 值` 两项。"""
    return [t for i, t in enumerate(cmd) if t != flag and (i == 0 or cmd[i - 1] != flag)]


def test_corridor_half_width_in_eval_commands():
    """W1-1：四组评测命令默认带 --corridor-half-width 3（柱阵评测的横向边界 = scene_gen.Y_RANGE）；
    E1_CORRIDOR_HALF_WIDTH=0 回到旧口径，命令与默认相比只少这两项；其它取值按 :g 格式透传。"""
    groups = len(ep.eval_groups())
    out = os.path.join(tempfile.gettempdir(), "test_e1_pipeline_corridor_never")   # dry-run 不建目录，只当路径用
    on = _dry_run_eval_cmds(out)
    assert len(on) == groups, f"dry-run 应打印 {groups} 条评测命令，实际 {len(on)}"
    for cmd in on:
        assert "--corridor-half-width" in cmd, f"评测命令缺 --corridor-half-width：{' '.join(cmd)}"
        assert cmd[cmd.index("--corridor-half-width") + 1] == "3", cmd
    off = _dry_run_eval_cmds(out, E1_CORRIDOR_HALF_WIDTH="0")
    assert len(off) == groups, off
    assert all("--corridor-half-width" not in cmd for cmd in off), off
    assert off == [_without_pair(cmd, "--corridor-half-width") for cmd in on], "设 0 只应少这两项，其余逐字不变"
    custom = _dry_run_eval_cmds(out, E1_CORRIDOR_HALF_WIDTH="2.5")
    assert all(cmd[cmd.index("--corridor-half-width") + 1] == "2.5" for cmd in custom), custom
    print(f"ok corridor：{groups} 组评测命令默认带 --corridor-half-width 3，设 0 时没有，设 2.5 透传")


# ───────────── W1-5：交付权重 calswa（末 10 轮权重平均 + FORWARD 先验校准） ─────────────
def _tool(cmd):
    """命令的工具名：训练器 uavrgb_train，或脚本名（cmd[1]）。"""
    first = os.path.basename(cmd[0])
    return first if first == "uavrgb_train" else os.path.basename(cmd[1])


def _flag(cmd, name):
    return cmd[cmd.index(name) + 1]


def _fake_sh(calls, fail_on=()):
    """替身 ep.sh：记录 (cmd, kwargs)，并按真实工具的产物落文件（训练器：ckpt_best.bin 与 metrics.jsonl 的 done 行；
    average_ckpt.py：--out 与 --out.json）。工具名在 fail_on 里就抛 RuntimeError，同真实 sh 对非零退出码的处理。"""
    def fake(cmd, **kw):
        calls.append((list(cmd), kw))
        tool = _tool(cmd)
        if tool in fail_on:
            raise RuntimeError(f"退出码 1: {tool}")
        if tool == "uavrgb_train":
            run_dir = _flag(cmd, "--out")
            _touch(os.path.join(run_dir, "ckpt_best.bin"))
            _touch(os.path.join(run_dir, "metrics.jsonl"), DONE_LINE)
        elif tool == "average_ckpt.py":
            _touch(_flag(cmd, "--out"))
            _touch(_flag(cmd, "--out") + ".json", "{}")
    return fake


def test_calswa_train_commands():
    """W1-5：deliver=calswa 时训练命令带 --save-every-epoch，训练步之后跟一条 average_ckpt.py：
    末 10 轮、同一份 dataset、产物 run/ckpt_calswa.bin、推理二进制取 BUILD、CUDA_VISIBLE_DEVICES 钉 CFG 的 gpu；不带 --no-calibrate。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    calls = []
    try:
        _touch(os.path.join(tmp, "ds", "dataset.uavr"))
        with _cfg(deliver="calswa"), _patched(ep, sh=_fake_sh(calls)):
            ep.train(tmp)
        tools = [_tool(c) for c, _ in calls]
        assert tools == ["uavrgb_train", "average_ckpt.py"], tools
        train_cmd, (avg_cmd, avg_kw) = calls[0][0], calls[1]
        run_dir, data = os.path.join(tmp, "run"), os.path.join(tmp, "ds", "dataset.uavr")
        assert "--save-every-epoch" in train_cmd, train_cmd
        assert avg_cmd[:2] == [ep.PY, os.path.join(REPO, "tools", "average_ckpt.py")], avg_cmd[:2]
        assert _flag(avg_cmd, "--run") == run_dir
        assert _flag(avg_cmd, "--data") == _flag(train_cmd, "--data") == data
        assert _flag(avg_cmd, "--out") == os.path.join(run_dir, "ckpt_calswa.bin")
        assert _flag(avg_cmd, "--last-k") == "10"
        assert _flag(avg_cmd, "--infer-bin") == f"{ep.BUILD}/uavrgb_infer"
        assert "--no-calibrate" not in avg_cmd, avg_cmd
        assert avg_kw.get("env_extra") == {"CUDA_VISIBLE_DEVICES": ep.CFG["gpu"]}, avg_kw
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok calswa：训练带 --save-every-epoch，之后一条 average_ckpt.py（末 10 轮、校准、钉卡）")


def _cli_flags(path, pattern):
    """静态读被调工具的源码，取它认的命令行开关（不跑工具）。"""
    with open(path, encoding="utf-8") as f:
        return set(re.findall(pattern, f.read()))


def test_calswa_flags_exist_in_tools():
    """W1-5：流水线拼出的开关在被调工具里真实存在。替身 sh 只看命令内容，开关笔误抓不到；
    这里静态读 average_ckpt.py 的 add_argument 与 train_main.cpp 的开关分支，对照训练命令与校准平均命令里的每个 `--开关`。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    calls = []
    try:
        _touch(os.path.join(tmp, "ds", "dataset.uavr"))
        with _cfg(deliver="calswa"), _patched(ep, sh=_fake_sh(calls)):
            ep.train(tmp)
        train_cmd, avg_cmd = calls[0][0], calls[1][0]
        avg_known = _cli_flags(os.path.join(REPO, "tools", "average_ckpt.py"), r'add_argument\("(--[\w-]+)"')
        train_known = _cli_flags(os.path.join(REPO, "src", "train_main.cpp"), r'a == "(--[\w-]+)"')
        avg_used = {a for a in avg_cmd if a.startswith("--")}
        train_used = {a for a in train_cmd if a.startswith("--")}
        assert "--save-every-epoch" in train_used and avg_used, (train_used, avg_used)
        assert avg_used <= avg_known, f"average_ckpt.py 不认的开关：{sorted(avg_used - avg_known)}，它认 {sorted(avg_known)}"
        assert train_used <= train_known, f"uavrgb_train 不认的开关：{sorted(train_used - train_known)}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"ok calswa 开关：average_ckpt.py {len(avg_used)} 个、uavrgb_train {len(train_used)} 个开关在被调工具里都存在")


def test_deliver_best_keeps_old_behavior():
    """deliver=best：训练命令与改前逐项相同（没有 --save-every-epoch），没有 average_ckpt.py，评测评 ckpt_best.bin。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    calls, cmds = [], {}
    real_popen = subprocess.Popen
    FakeProc.waited = []
    try:
        _touch(os.path.join(tmp, "ds", "dataset.uavr"))
        _touch(os.path.join(tmp, "old_x3.bin"))
        with _cfg(deliver="best", drop_collision=True, text_dim=0, old_x3_ckpt=os.path.join(tmp, "old_x3.bin")), \
                _patched(ep, sh=_fake_sh(calls)):
            ep.train(tmp)
            assert [_tool(c) for c, _ in calls] == ["uavrgb_train"], [_tool(c) for c, _ in calls]
            want = [f"{ep.BUILD}/uavrgb_train", "--data", f"{tmp}/ds/dataset.uavr", "--out", f"{tmp}/run",
                    "--epochs", str(ep.CFG["epochs"]), "--batch", str(ep.CFG["batch"]), "--lr", repr(float(ep.CFG["lr"])),
                    "--seed", str(ep.CFG["seed"]), "--class-weights", ep.CFG["class_weights"], "--drop-collision"]
            assert calls[0][0] == want, calls[0][0]

            def fake_popen(cmd, **_kw):
                edir = _flag(cmd, "--out")
                name = os.path.basename(edir)[len("eval-"):]
                cmds[name] = cmd
                return FakeProc(name, edir, write_eval=True)

            subprocess.Popen = fake_popen
            ep.evaluate(tmp)
        assert _flag(cmds["student-e1-new"], "--ckpt") == os.path.join(tmp, "run", "ckpt_best.bin"), cmds["student-e1-new"]
    finally:
        subprocess.Popen = real_popen
        FakeProc.waited.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok deliver=best：训练命令逐项不变、没有 average_ckpt.py、评测评 ckpt_best.bin")


def test_evaluate_uses_delivered_ckpt_without_fallback():
    """calswa 下评 run/ckpt_calswa.bin；它不在时（只有 ckpt_best.bin）一组评测都不起，不静默退回 best。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    cmds, real_popen = {}, subprocess.Popen
    FakeProc.waited = []
    try:
        _touch(os.path.join(tmp, "old_x3.bin"))
        _touch(os.path.join(tmp, "run", "ckpt_best.bin"))   # 只有 best

        def fake_popen(cmd, **_kw):
            edir = _flag(cmd, "--out")
            name = os.path.basename(edir)[len("eval-"):]
            cmds[name] = cmd
            return FakeProc(name, edir, write_eval=True)

        subprocess.Popen = fake_popen
        with _cfg(deliver="calswa", old_x3_ckpt=os.path.join(tmp, "old_x3.bin")):
            try:
                ep.evaluate(tmp)
            except RuntimeError as e:
                assert "ckpt_calswa.bin" in str(e), e
            else:
                raise AssertionError("calswa 的权重不存在，evaluate 应报错而不是评 ckpt_best.bin")
            assert cmds == {}, f"缺交付权重时不该起任何评测：{list(cmds)}"
            _touch(os.path.join(tmp, "run", "ckpt_calswa.bin"))
            ep.evaluate(tmp)
        assert _flag(cmds["student-e1-new"], "--ckpt") == os.path.join(tmp, "run", "ckpt_calswa.bin"), cmds["student-e1-new"]
        assert _flag(cmds["student-x3-old"], "--ckpt") == os.path.join(tmp, "old_x3.bin"), cmds["student-x3-old"]
    finally:
        subprocess.Popen = real_popen
        FakeProc.waited.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok evaluate：calswa 评 ckpt_calswa.bin，缺它就停，不退回 ckpt_best.bin")


def test_calswa_needs_enough_epochs_and_known_value():
    """训练轮数小于 avg_last_k 时 calswa 取不到窗口：train 与 preflight 都在起任何命令之前报错并提示 E1_DELIVER；deliver=best 不受限；
    deliver 的值不认识也报错。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    calls = []
    try:
        _touch(os.path.join(tmp, "ds", "dataset.uavr"))
        with _cfg(deliver="calswa", epochs=9), _patched(ep, sh=_fake_sh(calls)):
            for name, fn in (("train", lambda: ep.train(tmp)), ("preflight", ep.preflight)):
                try:
                    fn()
                except RuntimeError as e:
                    assert "E1_DELIVER" in str(e) and "epochs=9" in str(e), (name, e)
                else:
                    raise AssertionError(f"{name}：calswa 且 epochs=9 < 10 应报错")
            assert calls == [], f"报错前不该起命令：{calls}"
        with _cfg(deliver="best", epochs=9), _patched(ep, sh=_fake_sh(calls)):
            ep.train(tmp)
            assert [_tool(c) for c, _ in calls] == ["uavrgb_train"], calls
        with _cfg(deliver="swa"):
            try:
                ep.train(tmp)
            except RuntimeError as e:
                assert "calswa" in str(e) and "best" in str(e), e
            else:
                raise AssertionError("deliver 的值不认识应报错")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok calswa：epochs < 10 报错并提示 E1_DELIVER，deliver=best 不受限，未知取值报错")


def test_calswa_resume():
    """续跑：训练已完成但没有 ckpt_calswa.bin 时只补校准平均，不重训；产物（.bin 与 .json）齐了什么都不跑；
    只剩 .bin（上次在两个文件之间被杀）不当作完成，再起一次让工具自己拒绝覆盖。"""
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    calls = []
    try:
        _touch(os.path.join(tmp, "ds", "dataset.uavr"))
        _touch(os.path.join(tmp, "run", "ckpt_best.bin"))
        _touch(os.path.join(tmp, "run", "metrics.jsonl"), DONE_LINE)
        calswa = os.path.join(tmp, "run", "ckpt_calswa.bin")
        with _cfg(deliver="calswa"), _patched(ep, sh=_fake_sh(calls)):
            ep.train(tmp)
            assert [_tool(c) for c, _ in calls] == ["average_ckpt.py"], [_tool(c) for c, _ in calls]
            calls.clear()
            ep.train(tmp)
            assert calls == [], f"产物齐了不该再跑：{calls}"
            os.remove(calswa + ".json")
            ep.train(tmp)
            assert [_tool(c) for c, _ in calls] == ["average_ckpt.py"], calls
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok calswa 续跑：训完只补平均，产物齐了跳过，只剩 .bin 不当完成")


def test_calswa_failure_stops_pipeline():
    """校准平均失败（工具非零退出）→ train 步标 failed、DONE rc=1、后面的 eval 不起；不退回 ckpt_best。"""
    import signal
    tmp = tempfile.mkdtemp(prefix="test_e1_pipeline_")
    calls = []
    saved_argv, saved_sig = sys.argv, {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    def boom(_out):
        raise AssertionError("校准失败后流水线不该再进后面的步骤")

    try:
        _touch(os.path.join(tmp, "ds", "dataset.uavr"))
        sys.argv = ["e1_pipeline.py", "--out", tmp, "--from", "train"]
        # RUN 里 train 之后的步骤换成 boom：实现若退步让 train 误成功，也只会记一条失败，不会真起 Gazebo 评测
        run = dict(train=ep.RUN["train"], eval=boom, summary=boom)
        with _cfg(deliver="calswa"), \
                _patched(ep, sh=_fake_sh(calls, fail_on=("average_ckpt.py",)), preflight=lambda: None, RUN=run):
            rc = ep.main()
        assert rc == 1, rc
        assert [_tool(c) for c, _ in calls] == ["uavrgb_train", "average_ckpt.py"], calls
        status = json.load(open(os.path.join(tmp, "status.json"), encoding="utf-8"))
        assert status["train"]["state"] == "failed" and "average_ckpt.py" in status["train"]["error"], status
        assert "eval" not in status, f"校准失败后不该进 eval：{status}"
        assert open(os.path.join(tmp, "DONE"), encoding="utf-8").read().startswith("rc=1 ")
    finally:
        sys.argv = saved_argv
        for s, h in saved_sig.items():
            signal.signal(s, h)
        shutil.rmtree(tmp, ignore_errors=True)
    print("ok calswa 失败：train 标 failed、DONE rc=1、不进 eval")


def _dry_run_cmds(out, from_step, **env_extra):
    """子进程跑 `e1_pipeline.py --from <步> --dry-run`，返回打印出的每条命令（token 列表，去掉时间戳与重定向）。"""
    env = {k: v for k, v in os.environ.items() if k not in ("E1_CORRIDOR_HALF_WIDTH", "E1_DELIVER", "E1_GPU")}
    env.update(env_extra)
    r = subprocess.run([sys.executable, os.path.join(REPO, "tools", "e1_pipeline.py"),
                        "--out", out, "--from", from_step, "--dry-run"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return [line.split(" $ ", 1)[1].split("  > ", 1)[0].split() for line in r.stdout.splitlines() if " $ " in line]


def test_dry_run_deliver_commands():
    """dry-run 的完整链（train → eval）：默认训练命令含 --save-every-epoch，其后是 average_ckpt.py，评测 --ckpt 指向 ckpt_calswa.bin；
    E1_DELIVER=best 时三条都回到旧行为。"""
    out = os.path.join(tempfile.gettempdir(), "test_e1_pipeline_deliver_never")   # dry-run 不建目录，只当路径用
    new = _dry_run_cmds(out, "train")
    tools = [_tool(c) for c in new]
    assert tools == ["uavrgb_train", "average_ckpt.py"] + ["eval_closed_loop.py"] * 4, tools
    assert "--save-every-epoch" in new[0], new[0]
    assert _flag(new[1], "--out") == f"{out}/run/ckpt_calswa.bin" and _flag(new[1], "--last-k") == "10", new[1]
    e1_new = [c for c in new[2:] if f"{out}/eval-student-e1-new" in c][0]
    assert _flag(e1_new, "--ckpt") == f"{out}/run/ckpt_calswa.bin", e1_new
    old = _dry_run_cmds(out, "train", E1_DELIVER="best")
    assert [_tool(c) for c in old] == ["uavrgb_train"] + ["eval_closed_loop.py"] * 4, [_tool(c) for c in old]
    assert "--save-every-epoch" not in old[0], old[0]
    assert _flag([c for c in old[1:] if f"{out}/eval-student-e1-new" in c][0], "--ckpt") == f"{out}/run/ckpt_best.bin"
    assert old[0] == [t for t in new[0] if t != "--save-every-epoch"], "best 的训练命令只少 --save-every-epoch"
    print("ok dry-run：calswa 链 = 训练(--save-every-epoch) → average_ckpt.py → 评 ckpt_calswa.bin；best 回到旧命令")


def test_deliver_env_overrides():
    """CFG 在 import 时读环境变量：deliver 缺省 calswa、E1_DELIVER=best 切回；gpu 缺省是 11-11 起机器上唯一那张卡的 UUID，E1_GPU 覆盖；
    avg_last_k 缺省 10。"""
    code = ("import sys; sys.path.insert(0, %r); import e1_pipeline as ep; "
            "print(ep.CFG['deliver'], ep.CFG['gpu'], ep.CFG['avg_last_k'])" % os.path.join(REPO, "tools"))

    def read(**env_extra):
        env = {k: v for k, v in os.environ.items() if k not in ("E1_DELIVER", "E1_GPU")}
        env.update(env_extra)
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr
        return r.stdout.split()

    assert read() == ["calswa", "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740", "10"], read()
    assert read(E1_DELIVER="best")[0] == "best"
    assert read(E1_GPU="GPU-test")[1] == "GPU-test"
    print("ok CFG：deliver 缺省 calswa / 环境变量切 best，gpu 缺省 UUID / 环境变量覆盖")


def main():
    test_corridor_half_width_in_eval_commands()
    test_teacher_planner_passthrough()
    test_evaluate_waits_all_groups()
    test_summary_marks_mixed_metric_version()
    test_train_resume_by_metrics_done()
    test_evaluate_resume_by_eval_finished()
    test_main_records_interrupt_with_real_rc()
    test_dry_run_creates_nothing()
    test_calswa_train_commands()
    test_calswa_flags_exist_in_tools()
    test_deliver_best_keeps_old_behavior()
    test_evaluate_uses_delivered_ckpt_without_fallback()
    test_calswa_needs_enough_epochs_and_known_value()
    test_calswa_resume()
    test_calswa_failure_stops_pipeline()
    test_dry_run_deliver_commands()
    test_deliver_env_overrides()
    print("test_e1_pipeline: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
