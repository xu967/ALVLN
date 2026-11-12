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


def test_parse_stat_field():
    """E-4：/proc/<pid>/stat 第 2 列 comm 里可以有空格和括号，列号要从最后一个「) 」往后数。手写样本，不依赖本机进程。
    第 3 列起：S 1 777 700 0 -1 4194560 100 0 0 0 1 2 0 0 20 0 1 0 987654 …，即第 3 列 state=S、第 5 列 pgrp=777、第 22 列 starttime=987654。"""
    rest = "S 1 777 700 0 -1 4194560 100 0 0 0 1 2 0 0 20 0 1 0 987654 12345678 99 18446744073709551615"
    for comm in ("python3", "a b) c (d", "x) y"):
        text = f"4242 ({comm}) {rest}\n"
        assert pc.parse_stat_field(text, 3) == "S", comm
        assert pc.parse_stat_field(text, 5) == "777", comm
        assert pc.parse_stat_field(text, 22) == "987654", comm
        assert pc.parse_stat_field(text, 25) == "18446744073709551615", comm
        assert pc.parse_stat_field(text, 26) is None, "列号超出返回 None"
    assert pc.parse_stat_field("garbage", 22) is None, "格式不对返回 None，不抛"
    print("ok parse_stat_field：comm 带空格和括号时列号不错位")


def test_proc_stat_field_matches_awk_for_child():
    """E-4：真子进程（sleep，comm 里没有空格，awk 按列取值可信）上，PGID 与启动时刻与 awk 取的一致；进程收掉后返回 None。"""
    import subprocess
    child = subprocess.Popen(["sleep", "30"], start_new_session=True)
    try:
        want = {n: subprocess.run(["awk", "{print $%d}" % n, f"/proc/{child.pid}/stat"],
                                  capture_output=True, text=True).stdout.strip() for n in (5, 22)}
        assert want[5] == str(child.pid), want            # start_new_session：它自己是进程组长
        assert want[22].isdigit() and int(want[22]) > 0, want
        assert pc.proc_stat_field(child.pid, 5) == want[5], (pc.proc_stat_field(child.pid, 5), want)
        assert pc.proc_stat_field(child.pid, 22) == want[22], (pc.proc_stat_field(child.pid, 22), want)
        assert pc.proc_cmdline(child.pid) == "sleep 30", pc.proc_cmdline(child.pid)
    finally:
        child.kill()
        child.wait()
    assert pc.proc_stat_field(child.pid, 22) is None and pc.proc_cmdline(child.pid) is None, "收掉的进程没有 stat"
    print("ok proc_stat_field / proc_cmdline：与 awk 取的一致，进程不存在返回 None")


def test_write_run_pid_records_start_time():
    """E-4：run.pid 四行 pid= / pgid= / start= / cmd=（与 tools/exp_lib.sh 的 run.pid 逐键一致，stop_daemon.sh 一个解析器读两种入口）。
    start 是进程启动时刻（stat 第 22 列，自宿主机开机起的节拍，容器重启不归零）：PID 被别的进程拿走后这一列对不上，停进程前核对它就不会停错。
    旧写法 echo $! > run.pid 只有 PID。start 与 cmd 用 shell 另写一遍对账（exp_lib.sh 的 exp_stat / exp_cmdline 就是这两个写法）。"""
    import subprocess
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        path = os.path.join(tmp, "run.pid")
        _write(path, "424242\n")                      # 旧流水线留下的只有 PID 的记录
        pc.write_run_pid(tmp)
        lines = open(path, encoding="utf-8").read().splitlines()
        assert [ln.split("=", 1)[0] for ln in lines] == ["pid", "pgid", "start", "cmd"], lines
        rec = dict(ln.split("=", 1) for ln in lines)
        me = os.getpid()
        assert rec["pid"] == str(me) and rec["pgid"] == str(os.getpgid(0)), rec
        sh_start = subprocess.run(["bash", "-c", 's=$(< /proc/$1/stat); read -ra a <<< "${s##*) }"; echo "${a[19]}"', "_", str(me)],
                                  capture_output=True, text=True).stdout.strip()
        sh_cmd = subprocess.run(["bash", "-c", "tr '\\0' ' ' < /proc/$1/cmdline | sed 's/ *$//'", "_", str(me)],
                                capture_output=True, text=True).stdout.rstrip("\n")
        assert rec["start"] == sh_start and sh_start.isdigit(), (rec, sh_start)
        assert rec["cmd"] == sh_cmd and rec["cmd"], (rec, sh_cmd)
        assert os.listdir(tmp) == ["run.pid"], "先写临时文件再改名，不留 .tmp"
        pc.write_run_pid(tmp)                          # 重写同一个目录（续跑）：覆盖旧记录，格式不变
        assert open(path, encoding="utf-8").read().splitlines() == lines
    print("ok write_run_pid：pid / pgid / start / cmd 四行，与 shell 取法一致，原子写")


def test_write_run_pid_warns_when_not_group_leader():
    """E-4：停进程按 PGID，前提是本进程是进程组长（setsid nohup 起）；不是的话 run.pid 照实记，日志里提醒一句，不拒绝起跑。"""
    import contextlib
    import io
    saved = pc.proc_stat_field
    try:
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            for pgid, expect_warning in (("1", True), (str(os.getpid()), False)):
                pc.proc_stat_field = lambda pid, n, _g=pgid: _g if n == 5 else saved(pid, n)
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    pc.write_run_pid(tmp)
                assert ("不是进程组长" in buf.getvalue()) == expect_warning, (pgid, buf.getvalue())
                assert f"pgid={pgid}\n" in open(os.path.join(tmp, "run.pid"), encoding="utf-8").read()
    finally:
        pc.proc_stat_field = saved
    print("ok write_run_pid：不是进程组长时提醒，记录照实写")


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


_DROP = object()   # _eval_dir 里值为它的字段整个删掉（造旧格式 eval.json）


def _eval_dir(root, name, **fields):
    """造一个评测目录：默认是完整的 20/20 评测，fields 覆盖字段，值为 _DROP 的键删掉。返回目录。"""
    rep = {"episodes_requested": 20, "episodes_done": 20, "failed_count": 0, "skipped_count": 0, "partial": False}
    rep.update(fields)
    _write(os.path.join(root, name, "eval.json"), json.dumps({k: v for k, v in rep.items() if v is not _DROP}))
    return os.path.join(root, name)


def test_eval_state_unified_criterion():
    """E-8：「评完了」只有 pipeline_common.eval_state 一处判据，四条流水线都走它。
    没评完：没有或读不了 eval.json、partial、有失败集、有跳过集、完成数不等于请求数（被跳过的 seed 没评就是截断）、0 集。
    评完了但与当前配置对不上（stale）：metric_version 或权重路径与当前配置不同；旧 eval.json 没有这个键时不判 stale。"""
    mv, old_mv = "sim09.v2.coast1s.corridor3m", "sim09.v2.coast1s"
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        best, calswa = os.path.join(tmp, "run", "ckpt_best.bin"), os.path.join(tmp, "run", "ckpt_calswa.bin")
        _write(best, "b")
        _write(calswa, "c")
        link = os.path.join(tmp, "link.bin")
        os.symlink(calswa, link)
        cases = [   # (说明, eval.json 字段覆盖, 当前配置, 期望状态, 原因里该有的字)
            ("20/20 完整", {}, {}, "done", ""),
            ("E-8 复现：完成 5 集、跳过 15 集，旧判据当评完", dict(episodes_done=5, skipped_count=15), {}, "unfinished", "skipped_count=15"),
            ("截断：完成 12 集，没有失败也没有跳过", dict(episodes_done=12), {}, "unfinished", "episodes_done=12"),
            ("跳过 1 集但完成数等于请求数：字面照卡，跳过的评测不算评完", dict(skipped_count=1), {}, "unfinished", "skipped_count=1"),
            ("有失败集", dict(episodes_done=19, failed_count=1), {}, "unfinished", "failed_count=1"),
            ("中断写出的 partial", dict(partial=True), {}, "unfinished", "partial"),
            ("0 集", dict(episodes_done=0, episodes_requested=0), {}, "unfinished", "episodes_done=0"),
            ("旧 eval.json 没有 episodes_requested 与 skipped_count", dict(episodes_requested=_DROP, skipped_count=_DROP), {}, "done", ""),
            ("旧 eval.json 只有列表（没有 *_count）", dict(episodes_requested=_DROP, skipped_count=_DROP, failed_count=_DROP,
                                                   episodes_done=_DROP, failed=[], per_episode=[{"seed": 1}]), {}, "done", ""),
            ("metric_version 一致", dict(metric_version=mv), dict(metric_version=mv), "done", ""),
            ("metric_version 不同：旧口径没有走廊", dict(metric_version=old_mv), dict(metric_version=mv), "stale", old_mv),
            ("旧 eval.json 没有 metric_version 键：不判 stale", {}, dict(metric_version=mv), "done", ""),
            ("没给 metric_version：不核对", dict(metric_version=old_mv), {}, "done", ""),
            ("权重路径一致", dict(ckpt=calswa), dict(ckpt=calswa), "done", ""),
            ("权重路径不同：评的是 ckpt_best，当前交付 ckpt_calswa（W1-5 第 5 项）", dict(ckpt=best), dict(ckpt=calswa), "stale", "ckpt_best.bin"),
            ("eval.json 的 ckpt 是 null 而当前有权重", dict(ckpt=None), dict(ckpt=calswa), "stale", "ckpt"),
            ("旧 eval.json 没有 ckpt 键：不判 stale", {}, dict(ckpt=calswa), "done", ""),
            ("符号链接指向同一个文件：算一致", dict(ckpt=link), dict(ckpt=calswa), "done", ""),
            ("没评完又对不上：先算没评完", dict(episodes_done=19, failed_count=1, ckpt=best), dict(ckpt=calswa), "unfinished", "failed_count=1"),
        ]
        for i, (why, fields, want, state, word) in enumerate(cases):
            d = _eval_dir(tmp, f"case{i}", **fields)
            got, reason = pc.eval_state(d, **want)
            assert got == state, f"{why}：期望 {state}，得到 {got}（{reason}）"
            assert word in reason, f"{why}：原因 {reason!r} 里没有 {word!r}"
            assert pc.eval_finished(d, **want) is (state == "done"), why
        # 目录本身的几种状态
        assert pc.eval_state(os.path.join(tmp, "nope"))[0] == "missing"
        os.makedirs(os.path.join(tmp, "empty"))
        assert pc.eval_state(os.path.join(tmp, "empty"))[0] == "missing", "空目录没有东西要让出"
        _write(os.path.join(tmp, "videos-only", "videos", "x.png"), "")
        assert pc.eval_state(os.path.join(tmp, "videos-only")) == ("unfinished", "没有 eval.json"), "只剩 videos/：要让出目录重跑"
        _write(os.path.join(tmp, "broken", "eval.json"), "{broken")
        assert pc.eval_state(os.path.join(tmp, "broken"))[0] == "unfinished"
        _write(os.path.join(tmp, "listjson", "eval.json"), "[1, 2]")
        assert pc.eval_state(os.path.join(tmp, "listjson"))[0] == "unfinished"
    print("ok eval_state：没评完 / 对不上 / 评完的判据一处定义")


def test_settle_eval_retires_unfinished_and_stale():
    """E-8：settle_eval = 判定 + 让出目录。没评完的挪到 <目录>.failed-<时刻>，对不上的挪到 <目录>.stale-<时刻>，都不删；
    评完的、不存在的不动；dry-run 只判不挪；挪走的记进输出目录的 retired_dirs.jsonl（流水线收尾时列进 summary）。"""
    mv_now, mv_old = "sim09.v2.coast1s.corridor3m", "sim09.v2.coast1s"
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        pc.lock_out(tmp)
        saved = pc.time.strftime
        pc.time.strftime = lambda fmt, *a: "20261112-030000" if fmt == "%Y%m%d-%H%M%S" else saved(fmt, *a)
        try:
            _eval_dir(tmp, "ok", metric_version=mv_now)
            _eval_dir(tmp, "trunc", episodes_done=5, skipped_count=15)
            _eval_dir(tmp, "stale", metric_version=mv_old)
            assert pc.settle_eval(os.path.join(tmp, "ok"), metric_version=mv_now) == ("done", None, "")
            assert pc.settle_eval(os.path.join(tmp, "none"), metric_version=mv_now) == ("missing", None, "目录不存在")
            state, moved, why = pc.settle_eval(os.path.join(tmp, "trunc"), metric_version=mv_now)
            assert (state, moved) == ("unfinished", os.path.join(tmp, "trunc.failed-20261112-030000")), (state, moved)
            assert "skipped_count=15" in why and not os.path.exists(os.path.join(tmp, "trunc"))
            assert os.path.isfile(os.path.join(moved, "eval.json")), "旧目录原样留着"
            state, moved, why = pc.settle_eval(os.path.join(tmp, "stale"), metric_version=mv_now)
            assert (state, moved) == ("stale", os.path.join(tmp, "stale.stale-20261112-030000")), (state, moved)
            assert mv_old in why and mv_now in why and os.path.isfile(os.path.join(moved, "eval.json"))
            # 同一秒内再挪同名目录：加 -2，不覆盖
            _eval_dir(tmp, "stale", metric_version=mv_old)
            _s, moved2, _w = pc.settle_eval(os.path.join(tmp, "stale"), metric_version=mv_now)
            assert moved2 == os.path.join(tmp, "stale.stale-20261112-030000-2"), moved2
            assert os.path.isdir(os.path.join(tmp, "ok")), "评完的不动"
            rows = pc.read_retired(tmp)
            assert [(r["kind"], r["path"], r["moved_to"]) for r in rows] == [
                ("failed", "trunc", "trunc.failed-20261112-030000"),
                ("stale", "stale", "stale.stale-20261112-030000"),
                ("stale", "stale", "stale.stale-20261112-030000-2")], rows
            assert "skipped_count=15" in rows[0]["why"] and mv_old in rows[1]["why"] and rows[0]["at"]
            # dry-run：只判不挪，不记账
            _eval_dir(tmp, "dry", episodes_done=3)
            pc.DRY = True
            try:
                assert pc.settle_eval(os.path.join(tmp, "dry"))[:2] == ("unfinished", None)
            finally:
                pc.DRY = False
            assert os.path.isdir(os.path.join(tmp, "dry")) and len(pc.read_retired(tmp)) == 3
        finally:
            pc.time.strftime = saved
            pc.unlock_out()
    print("ok settle_eval：没评完 → .failed-，对不上 → .stale-，都不删，记台账；dry-run 只判不挪")


def test_retire_dir_generic_and_retired_lines():
    """E-8 / E-9：retire_dir 是通用的「改名让路」：<路径>.<后缀>-<时刻>（同一秒加 -2），目录不存在返回 None，不删，记台账。
    retired_lines 把台账排成 summary 里的几行；没有台账什么都不输出（没挪过东西的流水线 summary 与改前逐字相同）。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        assert pc.read_retired(tmp) == [] and pc.retired_lines(tmp) == []
        # 没拿过输出目录锁：照样改名，不写台账
        _write(os.path.join(tmp, "a", "x.txt"), "1")
        moved = pc.retire_dir(os.path.join(tmp, "a"), "partial", "没有 collection_manifest.json")
        assert os.path.basename(moved).startswith("a.partial-") and os.path.isfile(os.path.join(moved, "x.txt"))
        assert pc.read_retired(tmp) == []
        pc.lock_out(tmp)
        try:
            assert pc.retire_dir(os.path.join(tmp, "nope"), "partial") is None, "不存在：什么都不做"
            _write(os.path.join(tmp, "round_1", "collect", "x.txt"), "1")
            moved = pc.retire_dir(os.path.join(tmp, "round_1", "collect"), "partial", "没有 collection_manifest.json")
            assert os.path.basename(moved).startswith("collect.partial-") and not os.path.exists(os.path.join(tmp, "round_1", "collect"))
            (row,) = pc.read_retired(tmp)
            assert row["kind"] == "partial" and row["path"] == "round_1/collect" and row["moved_to"] == os.path.relpath(moved, tmp), row
            assert row["why"] == "没有 collection_manifest.json"
            lines = pc.retired_lines(tmp)
            assert lines[0].startswith("续跑挪走的目录") and "没删" in lines[0], lines
            assert len(lines) == 2 and "partial" in lines[1] and "round_1/collect" in lines[1] and "没有 collection_manifest.json" in lines[1], lines
            with open(os.path.join(tmp, "retired_dirs.jsonl"), "a", encoding="utf-8") as f:
                f.write("{半行没写完")                # 台账末行写到一半被杀：读时跳过，不抛
            assert len(pc.read_retired(tmp)) == 1
            # 改名失败：抛 RuntimeError，不留假台账，旧目录不动
            _write(os.path.join(tmp, "busy", "f.txt"), "keep")
            real_rename = pc.os.rename
            pc.os.rename = lambda *_a: (_ for _ in ()).throw(PermissionError("busy"))
            try:
                try:
                    pc.retire_dir(os.path.join(tmp, "busy"), "partial", "半成品")
                except RuntimeError as e:
                    assert "busy" in str(e)
                else:
                    raise AssertionError("改名失败应抛 RuntimeError")
            finally:
                pc.os.rename = real_rename
            assert os.path.isfile(os.path.join(tmp, "busy", "f.txt")) and len(pc.read_retired(tmp)) == 1
        finally:
            pc.unlock_out()
    print("ok retire_dir / read_retired / retired_lines（含改名失败停下）")


def test_metric_version_for_matches_eval_closed_loop():
    """E-8：流水线要在评测前就知道「这条命令会得到的 metric_version」，规则在 pipeline_common 里抄了一份（tools/ 不 import sim/）。
    对着 eval_closed_loop.report_metric_version 逐组对账，两边谁改了规则这里就红；3DGS 评测的常量对着 eval_3dgs.py 的源码读。"""
    import types
    sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
    import eval_closed_loop as ec
    assert pc.SIM_METRIC_VERSION == ec.METRIC_VERSION and pc.COAST_WINDOW_S == ec.COAST_WINDOW_S, (pc.SIM_METRIC_VERSION, ec.METRIC_VERSION)
    for dynamics in ("cf_sitl", "x3_velocity", "cf_stab", None):
        for corridor in (None, 3.0, 2.5, 4):
            want = ec.report_metric_version(types.SimpleNamespace(dynamics=dynamics, corridor_half_width=corridor))
            assert pc.metric_version_for(dynamics, corridor) == want, (dynamics, corridor, pc.metric_version_for(dynamics, corridor), want)
    assert pc.metric_version_for("cf_sitl", 3.0) == "sim09.v2.coast1s.corridor3m"
    import re
    with open(os.path.join(REPO, "sim", "3dgs", "eval_3dgs.py"), encoding="utf-8") as f:
        src_version = re.search(r'^METRIC_VERSION = "([^"]+)"', f.read(), re.M).group(1)
    import dagger_pipeline as dp
    assert dp.EVAL_METRIC_VERSION == src_version, (dp.EVAL_METRIC_VERSION, src_version)
    print("ok metric_version_for 与 eval_closed_loop / eval_3dgs 的规则一致")


def test_lock_out_refuses_second_instance():
    """E-3：同一输出目录同时只许一个流水线实例。第二个实例（另一个进程）拿不到 <out>/.pipeline.lock，立刻 SystemExit(3)，
    消息带持锁者的 pid；拿锁失败不改目录里任何已有文件。持锁者主动放锁、或被 SIGKILL，锁都不残留，之后能再拿（容器重启后原命令再起）。"""
    import subprocess
    tools = os.path.join(REPO, "tools")
    take = f"import sys; sys.path.insert(0, {tools!r}); import pipeline_common as pc; pc.lock_out(sys.argv[1])"

    def second(tmp):
        return subprocess.run([sys.executable, "-c", take, tmp], capture_output=True, text=True, timeout=60)

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        _write(os.path.join(tmp, "DONE"), "rc=0 first\n")
        pc.lock_out(tmp)                      # 本进程当第一个实例
        try:
            r = second(tmp)
            assert r.returncode == 3, (r.returncode, r.stdout, r.stderr)
            assert "已有实例在跑" in r.stderr and f"pid={os.getpid()}" in r.stderr, r.stderr
            assert open(os.path.join(tmp, "DONE"), encoding="utf-8").read() == "rc=0 first\n"
            assert sorted(os.listdir(tmp)) == [".pipeline.lock", "DONE"], os.listdir(tmp)
        finally:
            pc.unlock_out()
        r = second(tmp)
        assert r.returncode == 0, (r.returncode, r.stderr)   # 主动放锁后第二个能拿
        pc.unlock_out()                                      # 没持锁时再调不报错

        # 持锁的是另一个进程，被 SIGKILL：内核放锁，不留残余
        holder = subprocess.Popen([sys.executable, "-c", take + "; print('ready', flush=True); import time; time.sleep(60)", tmp],
                                  stdout=subprocess.PIPE, text=True)
        try:
            assert holder.stdout.readline().strip() == "ready"
            r = second(tmp)
            assert r.returncode == 3 and f"pid={holder.pid}" in r.stderr, (r.returncode, r.stderr)
            holder.kill()
            holder.wait(timeout=30)
            r = second(tmp)
            assert r.returncode == 0, (r.returncode, r.stderr)
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.wait(timeout=30)
    print("ok lock_out：第二个实例退出 3、带持锁者 pid、不改目录；放锁或持锁者被杀后可再拿")


def test_torch_extensions_dir():
    """gsplat JIT 缓存目录：同一算力同一路径（2080 Ti = $DATA_ROOT/uav_rgb/torch-cache/extensions-sm75）；
    问不到算力（python 不存在 / 退出非零 / 输出不是两个整数）返回 None。假 python 是 shell 脚本，不起 torch。"""
    root = os.path.join(os.environ.get("DATA_ROOT", "/root/rivermind-data"), "uav_rgb", "torch-cache")
    assert pc.torch_extensions_dir((7, 5)) == os.path.join(root, "extensions-sm75")
    assert pc.torch_extensions_dir((8, 9)) == os.path.join(root, "extensions-sm89")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        def fake_python(name, body):
            path = os.path.join(tmp, name)
            _write(path, "#!/bin/sh\n" + body + "\n")
            os.chmod(path, 0o755)
            return path

        assert pc.torch_extensions_dir(python=fake_python("ok", "echo 7 5")) == os.path.join(root, "extensions-sm75")
        for py in (fake_python("fail", "exit 1"), fake_python("junk", "echo no cuda"), fake_python("three", "echo 7 5 0"),
                   os.path.join(tmp, "missing-python")):
            assert pc.torch_extensions_dir(python=py) is None, py
    print("ok torch_extensions_dir：同一算力同一路径，问不到返回 None")


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
    import dagger_gz_pipeline as dg
    for name in ("log", "read_json", "write_json"):   # run/sh/mark/need 在 dagger_gz 里语义不同，见该文件注释
        assert getattr(dg, name) is getattr(pc, name), f"dagger_gz_pipeline.{name} 不是 pipeline_common 的那一份"
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
    test_eval_state_unified_criterion()
    test_settle_eval_retires_unfinished_and_stale()
    test_retire_dir_generic_and_retired_lines()
    test_metric_version_for_matches_eval_closed_loop()
    test_lock_out_refuses_second_instance()
    test_parse_stat_field()
    test_proc_stat_field_matches_awk_for_child()
    test_write_run_pid_records_start_time()
    test_write_run_pid_warns_when_not_group_leader()
    test_signal_handlers_and_child_groups()
    test_torch_extensions_dir()
    test_shared_helpers()
    print("test_pipeline_common: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
