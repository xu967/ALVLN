#!/usr/bin/env python3
"""tools/exp_lib.sh 与 tools/stop_daemon.sh 的测试（任务卡 W1-15a，审核 E）。

不起 Gazebo、不用 GPU。锁文件、cgroup 文件、输出目录全在临时目录里，假臂命令只用 sleep、false、echo；
真实的 $D/tmp/phase12-cards/*.lock 在这里只作为「绝不能出现」的断言对象，用例的 LOCK_DIR 一律指向临时目录。
每个用例造一个临时 git 仓库（带 tools/exp_lib.sh 的副本与一个假入口），像正式运行那样 setsid 起入口，再读它留下的文件。

覆盖（E 开头是交叉审核 E 的发现编号）：
  E-5   实例锁在最前：重复实例 rc=3、不写任何文件；带着会被拒绝的配置起第二个实例，也不写 DONE 与 refused.txt
  DONE  入口没走到 exp_finish 就退出（未定义变量、中途 exit）也补一个 rc≠0 的 DONE，等 DONE 的脚本不会一直等
  E-4   run.pid 记 pid、pgid、启动时刻、命令行；入口不是进程组长时拒绝；被 SIGTERM 时落 rc=130 的 DONE
  E-2   代码冻结到 $OUT/code（git archive），续跑沿用首次的提交与冻结副本，起跑记录 launches.log 只追加，输入文件的 sha256 与首次不同就拒绝
  E-10  环境变量覆盖过的配置项逐个进起跑记录；入口与库没提交（或与提交不一致）拒绝起跑
  E-11  续跑给了别的 CODE_HEAD：先核对、拒绝，记录一个字节都不动
  池    同池排队、异池不互等、POOL=none 不碰锁文件、锁文件名与公共规则 §4 一致；GPU（rc=12）与配额（rc=13）核对在拿锁之前
  Gazebo 池的起跑条件（01:40 规则）：拿锁后空闲就起；不空闲每隔一阵复查，中途空闲了就起；到点照样起并记下当时的核数；等待有上限
  E-6   臂失败时 DONE 的 rc 不为 0（rc=11 并列出失败的臂），其余臂照跑；并行的臂也算；信号终止落 rc=130 且不留 .ok
  E-9   续跑：已完成的臂跳过，没跑完的臂目录改名 *.partial-<时刻> 留着、重做，改名进起跑记录；改名失败就停，不在旧目录上重做
  时序  每个臂起止各记一次 cgroup 快照，打一行健康（平均核数、nr_throttled 增量、被限流时长及占墙钟的比例）
  E-4   stop_daemon.sh：记录对得上（启动时刻、命令行、进程组）才 kill -- -PGID；伪造启动时刻（模拟容器重启后 PID 被别的进程用了）不杀、报告、rc≠0；
        旧格式 run.pid、没有 run.pid、入口已不在但进程组里还有进程、TERM 之后仍有进程，各有各的 rc
  E-13  tools/check.sh：设 PYTHONDONTWRITEBYTECODE=1；临时目录跟 TMPDIR 走，没设时落数据盘的 tmp，数据盘不在就沿用系统缺省
  样板  experiments/_template/run.sh 不超过 80 行，只选一个池就能整条链路跑通并续跑，不选池被拒绝，Gazebo 池的起跑条件也走得通；
        起跑后入口文件被原地改写（编辑器、工具直接写文件）不影响已启动的实例（bash 边跑边从文件里读后面的行，样板把脚本体包进 main() 避开）

运行（仓库根）：nice -n 10 .venv-tools/bin/python tools/test_exp_lib.py
"""
import contextlib
import hashlib
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
REAL_LOCK_DIR = "/root/rivermind-data/uav_rgb/tmp/phase12-cards"   # 真实的池锁目录：只用来断言测试没有指向它
KEEP = os.environ.get("EXP_LIB_TEST_KEEP") == "1"                 # 1 = 不删临时目录，排查用


def proc_stat(pid):
    """/proc/<pid>/stat 里 comm 之后的字段，下标 0 对应第 3 列（state）；第 5 列 pgrp = [2]，第 22 列 starttime = [19]。"""
    with open(f"/proc/{pid}/stat") as f:
        s = f.read()
    return s[s.rindex(")") + 2:].split()


def group_members(pgid):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if int(proc_stat(d)[2]) == pgid:
                    out.append(int(d))
            except (OSError, ValueError, IndexError):
                pass
    return out


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def wait_for(pred, timeout=15.0, what="条件", proc=None):
    """轮询到 pred() 为真；proc 给了就在它提前退出时立刻失败并带上它的 stderr，不空等到超时。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return
        if proc is not None and proc.poll() is not None:
            raise AssertionError(f"{what}：进程已退出 rc={proc.returncode}，stderr：{proc.stderr.read().decode()[-1500:]}")
        time.sleep(0.05)
    raise AssertionError(f"等了 {timeout} s，{what} 仍不成立")


def sh(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pairs(text):
    """run.pid 之类的 key=value 行 → dict。"""
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


class Sandbox:
    """一个用例的临时世界：git 仓库（含库的副本与假入口）、输出目录、锁目录、假 cgroup 文件。"""

    def __init__(self, base):
        self.base = tempfile.mkdtemp(prefix="sb-", dir=base)
        self.repo, self.out, self.locks = (os.path.join(self.base, d) for d in ("repo", "out", "locks"))
        for d in (self.repo, self.locks, os.path.join(self.repo, "tools"), os.path.join(self.repo, "experiments", "e")):
            os.makedirs(d, exist_ok=True)
        assert os.path.realpath(self.locks) != REAL_LOCK_DIR, "测试的锁目录不能是真实的 phase12-cards"
        self.cpu_max, self.cpu_stat = os.path.join(self.base, "cpu.max"), os.path.join(self.base, "cpu.stat")
        self.write_file(self.cpu_max, "1200000 100000\n")
        self.set_cpu_stat(0, 0, 0)
        self.fakebin = os.path.join(self.base, "fakebin")      # 顶替 pgrep（数 gz sim）与 nvidia-smi，个数与卡号由环境变量给
        for name, body in (("pgrep", 'echo "${FAKE_GZ_COUNT:-0}"'),
                           ("nvidia-smi", 'echo "GPU 0: Fake Card (UUID: ${FAKE_GPU_UUID:-GPU-aaaa})"')):
            self.write_file(os.path.join(self.fakebin, name), f"#!/bin/sh\n{body}\n")
            os.chmod(os.path.join(self.fakebin, name), 0o755)
        for name in ("exp_lib.sh", "stop_daemon.sh"):          # 库的副本进临时仓库，入口按 ../../tools/ 找它（与正式布局相同）
            src = os.path.join(HERE, name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.repo, "tools", name))
        sh("git", "init", "-q", cwd=self.repo)
        sh("git", "config", "user.email", "t@t", cwd=self.repo)
        sh("git", "config", "user.name", "t", cwd=self.repo)
        self.procs = []

    # ---- 文件
    def write_file(self, path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def set_cpu_stat(self, usage_usec, nr_throttled, throttled_usec):
        tmp = self.cpu_stat + ".tmp"
        self.write_file(tmp, f"usage_usec {usage_usec}\nnr_throttled {nr_throttled}\nthrottled_usec {throttled_usec}\n")
        os.replace(tmp, self.cpu_stat)

    @contextlib.contextmanager
    def busy_cpu(self, cores, throttle=0.0):
        """后台线程按 cores 个核的速度推进假 cpu.stat 的 usage_usec（throttle = 每秒墙钟被限流的秒数）；退出后文件停在最后的值。"""
        stop = threading.Event()

        def pump():
            last, u, n, th = time.time(), 5_000_000_000, 1000, 2_000_000_000     # 起点取大数：列取错或没做差，结果会差几个数量级
            while not stop.wait(0.02):
                now = time.time()
                dt, last = now - last, now
                u, th, n = u + int(cores * 1e6 * dt), th + int(throttle * 1e6 * dt), n + (1 if throttle else 0)
                self.set_cpu_stat(u, n, th)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join()

    def commit(self, msg="c"):
        sh("git", "add", "-A", cwd=self.repo)
        sh("git", "commit", "-q", "--allow-empty", "-m", msg, cwd=self.repo)
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()

    def head(self):
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.repo, text=True).strip()

    def cfg(self, **over):
        """配置区的缺省值：与 experiments/_template/run.sh 的变量一一对应，数值取测试用的短值。"""
        c = dict(OUT=self.out, POOL="none", LOCK_DIR=self.locks, LOCK_LOG_S="1", SRC=self.repo, CODE_HEAD="", INPUTS="",
                 SELFCHECK="1", GPU="", MIN_CORES="0", CPU_MAX_FILE=self.cpu_max, CPU_STAT_FILE=self.cpu_stat,
                 IDLE_MAX_CORES="8", IDLE_WINDOW_S="0.3", IDLE_RECHECK_S="0.3", IDLE_LIMIT_S="3",
                 GZ_PATTERN="(^|[ /])gz sim( |$)")
        c.update(over)
        return c

    def entry(self, body, commit=True, name="run.sh", **cfg):
        """写假入口（配置区 + body）到 repo/experiments/e/<name>，缺省随即提交（入口自检要求它已提交）。"""
        head = ["#!/usr/bin/env bash", "set -u", 'HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)',
                'source "$HERE/../../tools/exp_lib.sh"']
        head += [f"exp_cfg {k} {shlex.quote(str(v))}" for k, v in self.cfg(**cfg).items()]
        path = os.path.join(self.repo, "experiments", "e", name)
        self.write_file(path, "\n".join(head) + "\n" + textwrap.dedent(body).lstrip("\n"))
        if commit:
            self.commit(f"entry {name}")
        return path

    # ---- 进程：像 setsid nohup bash run.sh 那样起，环境只留 PATH 等，免得外面残留的变量被当成覆盖
    def launch(self, entry, env=None, group=True):
        e = {"PATH": self.fakebin + ":" + os.environ["PATH"], "HOME": os.environ.get("HOME", "/root"), "TMPDIR": os.environ.get("TMPDIR", "/tmp")}
        e.update(env or {})
        p = subprocess.Popen(["bash", entry], cwd=self.repo, env=e, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, start_new_session=group)
        self.procs.append((p, group))
        return p

    def run(self, entry, env=None, group=True, timeout=30):
        """起入口并等它结束，返回 (rc, stderr)。"""
        p = self.launch(entry, env, group)
        err = p.communicate(timeout=timeout)[1].decode()
        return p.returncode, err

    def snapshot(self):
        """输出目录里每个文件的 sha256（不含 .lock），用来断言「什么都没动」。"""
        snap = {}
        for root, _, files in os.walk(self.out):
            for f in files:
                p = os.path.join(root, f)
                if f != ".lock":
                    snap[os.path.relpath(p, self.out)] = sha(p)
        return snap

    def cleanup(self):
        for p, group in self.procs:           # 只杀自己起的进程组（组长就是 Popen 的 pid）
            if group and p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
            if p.poll() is None:
                p.kill()
            p.wait()
            if p.stderr:
                p.stderr.close()
        if not KEEP:
            shutil.rmtree(self.base, ignore_errors=True)


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp(prefix="exp_lib_test_", dir=os.environ.get("TMPDIR"))

    def setUp(self):
        self.sb = Sandbox(self.base)
        self.addCleanup(self.sb.cleanup)

    @classmethod
    def tearDownClass(cls):
        left = []                              # 测试结束不许留下进程：命令行里带本次临时目录的都算
        for d in os.listdir("/proc"):
            if d.isdigit() and int(d) != os.getpid():
                try:
                    with open(f"/proc/{d}/cmdline", "rb") as f:
                        if cls.base.encode() in f.read():
                            left.append(d)
                except OSError:
                    pass
        if not KEEP:
            shutil.rmtree(cls.base, ignore_errors=True)
        assert not left, f"测试留下了进程 {left}"

    # 常用断言
    def out(self, name):
        return os.path.join(self.sb.out, name)

    def done(self):
        return read(self.out("DONE")) if os.path.exists(self.out("DONE")) else None


# 入口里「臂」的占位：起来后留个标记，睡到被停；commit 4 之前还没有 exp_arm，用裸命令
LONG = """
    exp_start
    touch "$OUT/started"
    sleep 30
    exp_finish 0 "不应走到这里"
"""


class InstanceLockAndPid(Base):
    """E-5：先拿实例锁再写任何状态文件。E-4：run.pid 带启动时刻，信号终止落 rc=130。"""

    def start_long(self):
        entry = self.sb.entry(LONG)
        p = self.sb.launch(entry)
        wait_for(lambda: os.path.exists(self.out("started")), what="第一个实例起来", proc=p)
        return entry, p

    def test_second_instance_refused_and_state_untouched(self):
        entry, p = self.start_long()
        before = self.sb.snapshot()
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 3, err)
        self.assertIn("已有实例", err)
        self.assertEqual(self.sb.snapshot(), before, "第二个实例改动了状态文件")
        self.assertIsNone(p.poll(), "第一个实例被第二个实例弄死了")

    def test_refusing_config_after_lock_writes_no_done(self):
        # 第二个实例本来会因为「不是进程组长」被拒绝（rc=2，写 DONE 与 refused.txt）；拿锁在前，所以应当先因为锁退出 rc=3。
        entry, p = self.start_long()
        rc, err = self.sb.run(entry, group=False)
        self.assertEqual(rc, 3, err)
        self.assertIsNone(self.done(), "拿不到锁的实例写了 DONE（E-5）")
        self.assertFalse(os.path.exists(self.out("refused.txt")))

    def test_run_pid_records_pid_pgid_start_and_cmd(self):
        entry, p = self.start_long()
        rec = pairs(read(self.out("run.pid")))
        self.assertEqual(rec["pid"], str(p.pid))
        self.assertEqual(rec["pgid"], str(p.pid), "setsid 起的入口应当是进程组长")
        self.assertEqual(rec["start"], proc_stat(p.pid)[19], "start 应是 /proc/<pid>/stat 第 22 列")
        self.assertEqual(rec["cmd"], f"bash {entry}")

    def test_not_group_leader_refused(self):
        entry = self.sb.entry(LONG)
        rc, err = self.sb.run(entry, group=False)
        self.assertEqual(rc, 2, err)
        self.assertIn("进程组长", err)
        self.assertTrue(self.done().startswith("rc=2 "), self.done())
        self.assertFalse(os.path.exists(self.out("run.pid")), "被拒绝的实例不该写 run.pid")

    def test_sigterm_writes_done_130(self):
        entry, p = self.start_long()
        os.killpg(p.pid, signal.SIGTERM)       # 与 kill -- -PGID 相同：入口与它的子进程一起收到
        p.wait(timeout=15)
        self.assertEqual(p.returncode, 130)
        self.assertTrue(self.done().startswith("rc=130 "), self.done())
        self.assertIn("收到终止信号", read(self.out("run.log")))
        self.assertEqual(group_members(p.pid), [], "进程组里还有残留进程")


FIN = """
    exp_start
    exp_finish 0 "ok"
"""


class FrozenCodeAndLaunchRecord(Base):
    """E-2、E-10、E-11：冻结代码、只追加的起跑记录、续跑时核对。"""

    def setUp(self):
        super().setUp()
        self.sb.write_file(os.path.join(self.sb.repo, "code", "hello.txt"), "v1\n")
        self.sb.write_file(os.path.join(self.sb.base, "input.bin"), "weights-1\n")
        self.inp = os.path.join(self.sb.base, "input.bin")
        self.entry = self.sb.entry(FIN, INPUTS=self.inp, LANES=2)
        self.head1 = self.sb.head()

    def log(self):
        return read(self.out("launches.log"))

    def test_freeze_is_a_git_archive_of_the_commit(self):
        rc, err = self.sb.run(self.entry)
        self.assertEqual(rc, 0, err)
        self.assertEqual(read(self.out("code/code/hello.txt")), "v1\n")
        self.assertEqual(read(self.out("code/experiments/e/run.sh")), read(self.entry))
        # 起跑之后工作树里的改动（含已提交的新版本）不影响冻结副本，续跑也不换
        self.sb.write_file(os.path.join(self.sb.repo, "code", "hello.txt"), "v2\n")
        self.sb.commit("v2")
        rc, err = self.sb.run(self.entry)
        self.assertEqual(rc, 0, err)
        self.assertEqual(read(self.out("code/code/hello.txt")), "v1\n", "续跑换了冻结副本")
        self.assertEqual(self.log().count(f"code_head {self.head1}\n"), 2, "续跑应沿用首次的提交")

    def test_record_lists_head_inputs_and_only_the_overridden_items(self):
        rc, err = self.sb.run(self.entry, env={"LANES": "3"})
        self.assertEqual(rc, 0, err)
        log = self.log()
        self.assertIn(f"code_head {self.head1}\n", log)
        self.assertIn(f"input {self.inp} {sha(self.inp)}\n", log)
        self.assertIn("override LANES=3（缺省 2）\n", log)
        self.assertEqual(log.count("override "), 1, "没被覆盖的配置项不该进记录：\n" + log)
        self.assertIn(f"entry {self.entry} {sha(self.entry)}\n", log)

    def test_record_is_append_only(self):
        self.sb.run(self.entry)
        first = self.log()
        self.sb.run(self.entry, env={"LANES": "4"})
        second = self.log()
        self.assertTrue(second.startswith(first), "第二次起跑改写了第一次的记录")
        self.assertEqual(second.count("== 起跑"), 2)
        self.assertIn("override LANES=4（缺省 2）", second[len(first):])
        self.assertNotIn("override", first)

    def test_lib_does_not_clobber_the_entrys_own_variables(self):
        # exp_begin 里的循环变量 v、f 是局部的：入口在 exp_start 之前用过同名变量，之后还要能读到原值
        self.sb.write_file(os.path.join(self.sb.base, "input.bin"), "weights-1\n")
        entry = self.sb.entry("""
            f=keep-f; v=keep-v
            exp_start
            [[ $f == keep-f && $v == keep-v ]] && touch "$OUT/vars-intact"
            exp_finish 0 "ok"
        """, INPUTS=os.path.join(self.sb.base, "input.bin"), LANES=2)
        rc, err = self.sb.run(entry, env={"LANES": "3"})
        self.assertEqual(rc, 0, err)
        self.assertTrue(os.path.exists(self.out("vars-intact")), "库改掉了入口里名为 f 或 v 的变量")

    def test_resume_refuses_changed_input_and_leaves_state_alone(self):
        self.assertEqual(self.sb.run(self.entry)[0], 0)
        before = self.sb.snapshot()
        self.sb.write_file(self.inp, "weights-2\n")
        rc, err = self.sb.run(self.entry)
        self.assertEqual(rc, 4, err)
        self.assertIn(self.inp, err)
        after = self.sb.snapshot()
        refused = after.pop("refused.txt")
        self.assertEqual(after, before, "拒绝起跑却改动了状态文件（DONE 与起跑记录应原样保留）")
        self.assertIn("sha256", read(self.out("refused.txt")))
        self.assertNotEqual(refused, "")

    def test_wrong_code_head_is_refused_before_anything_is_written(self):
        # E-11：run_r2 先写 code_head.txt 再核对，误带的提交会留在记录里。这里核对在前，记录一个字节都不动
        self.assertEqual(self.sb.run(self.entry)[0], 0)
        self.sb.write_file(os.path.join(self.sb.repo, "code", "hello.txt"), "v2\n")
        head2 = self.sb.commit("v2")
        before = self.sb.snapshot()
        rc, err = self.sb.run(self.entry, env={"CODE_HEAD": head2})
        self.assertEqual(rc, 4, err)
        self.assertIn(self.head1, err)
        after = self.sb.snapshot()
        after.pop("refused.txt")
        self.assertEqual(after, before)
        rc, err = self.sb.run(self.entry)                 # 下次不带变量续跑，仍取首次的提交
        self.assertEqual(rc, 0, err)
        self.assertEqual(read(self.out("code/code/hello.txt")), "v1\n")

    def test_resume_refuses_tampered_frozen_copy(self):
        self.assertEqual(self.sb.run(self.entry)[0], 0)
        self.sb.write_file(self.out("code/code/hello.txt"), "edited\n")
        rc, err = self.sb.run(self.entry)
        self.assertEqual(rc, 4, err)
        self.assertIn("冻结副本", err)

    def test_uncommitted_entry_refused_and_selfcheck_override_is_recorded(self):
        with open(self.entry, "a", encoding="utf-8") as f:
            f.write("# 起跑后才改的一行\n")
        rc, err = self.sb.run(self.entry)
        self.assertEqual(rc, 4, err)
        self.assertIn("先提交再起跑", err)
        os.remove(self.out("DONE"))
        rc, err = self.sb.run(self.entry, env={"SELFCHECK": "0"})      # 演练：放行，但覆盖项进记录
        self.assertEqual(rc, 0, err)
        self.assertIn("override SELFCHECK=0（缺省 1）", self.log())

    def test_missing_input_refused(self):
        os.remove(self.inp)
        rc, err = self.sb.run(self.entry)
        self.assertEqual(rc, 2, err)
        self.assertIn("缺输入", err)

    def test_half_frozen_copy_from_a_dead_first_launch_is_renamed_not_reused(self):
        self.sb.write_file(self.out("code/junk.txt"), "首次起跑死在解包途中留下的半成品\n")
        rc, err = self.sb.run(self.entry)
        self.assertEqual(rc, 0, err)
        parts = [d for d in os.listdir(self.sb.out) if d.startswith("code.partial-")]
        self.assertEqual(len(parts), 1, os.listdir(self.sb.out))
        self.assertTrue(os.path.exists(os.path.join(self.sb.out, parts[0], "junk.txt")))
        self.assertFalse(os.path.exists(self.out("code/junk.txt")))
        self.assertEqual(read(self.out("code/code/hello.txt")), "v1\n")


def read_or_empty(path):
    return read(path) if os.path.exists(path) else ""


def rec_line(text, key):
    """起跑记录里以 key 开头的第一行 → dict（key=value 各项）。"""
    for line in text.splitlines():
        if line.startswith(key + " "):
            return dict(kv.split("=", 1) for kv in line.split()[1:] if "=" in kv)
    raise AssertionError(f"起跑记录里没有 {key} 行：\n{text}")


# 拿到池锁之后一直占着，直到 RELEASE 文件出现（假的「长实验」）
HOLD = """
    exp_start
    touch "$OUT/started"
    while [[ ! -e "$RELEASE" ]]; do sleep 0.1; done
    exp_finish 0 "ok"
"""
GO = """
    exp_start
    touch "$OUT/started"
    exp_finish 0 "ok"
"""


class Pools(Base):
    """池锁：同池排队、异池并行、POOL=none 不碰锁、锁文件名与公共规则一致；GPU 与配额的核对在拿锁之前。"""

    def second_out(self):
        return os.path.join(self.sb.base, "out2")

    def test_same_pool_queues_until_the_holder_exits(self):
        rel = os.path.join(self.sb.base, "release")
        entry = self.sb.entry(HOLD, POOL="gazebo", RELEASE=rel)
        p1 = self.sb.launch(entry)
        wait_for(lambda: os.path.exists(self.out("started")), what="第一个拿到池锁", proc=p1)
        p2 = self.sb.launch(entry, env={"OUT": self.second_out()})
        log2 = os.path.join(self.second_out(), "run.log")
        wait_for(lambda: "仍在排队" in read_or_empty(log2), what="第二个排队并记日志", proc=p2)
        self.assertFalse(os.path.exists(os.path.join(self.second_out(), "started")), "第一个还占着锁，第二个不该往下走")
        self.assertIsNone(p2.poll())
        open(rel, "w").close()
        self.assertEqual((p1.wait(timeout=15), p2.wait(timeout=15)), (0, 0))
        self.assertTrue(os.path.exists(os.path.join(self.second_out(), "started")))
        line = rec_line(read(os.path.join(self.second_out(), "launches.log")), "pool")
        self.assertGreaterEqual(int(line["lock_wait_s"]), 1, line)

    def test_different_pools_do_not_wait_for_each_other(self):
        rel = os.path.join(self.sb.base, "release")
        entry = self.sb.entry(HOLD, POOL="gazebo", RELEASE=rel)
        p1 = self.sb.launch(entry)
        wait_for(lambda: os.path.exists(self.out("started")), what="Gazebo 池的实验起来", proc=p1)
        p2 = self.sb.launch(entry, env={"OUT": self.second_out(), "POOL": "gs3d"})
        wait_for(lambda: os.path.exists(os.path.join(self.second_out(), "started")), what="3DGS 池的实验不等 Gazebo 池", proc=p2)
        self.assertIsNone(p1.poll())
        open(rel, "w").close()
        self.assertEqual((p1.wait(timeout=15), p2.wait(timeout=15)), (0, 0))

    def test_lock_file_names_follow_the_common_rules_and_none_takes_no_lock(self):
        entry = self.sb.entry(GO)
        for i, pool in enumerate(("none", "gazebo", "gs3d", "train")):
            rc, err = self.sb.run(entry, env={"POOL": pool, "OUT": os.path.join(self.sb.base, f"o{i}")})
            self.assertEqual(rc, 0, f"{pool}: {err}")
            if pool == "none":
                self.assertEqual(os.listdir(self.sb.locks), [], "POOL=none 不该碰锁目录")
        self.assertEqual(sorted(os.listdir(self.sb.locks)), ["gs3d.lock", "machine.lock", "train.lock"])
        self.assertFalse(os.path.exists(os.path.join(REAL_LOCK_DIR, "machine.lock.fake")))

    def test_unknown_pool_refused(self):
        entry = self.sb.entry(GO, POOL="gpu")
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 2, err)
        self.assertIn("POOL", err)
        self.assertEqual(os.listdir(self.sb.locks), [])

    def test_gpu_missing_refused_before_any_lock(self):
        entry = self.sb.entry(GO, POOL="train", GPU="GPU-bbbb")
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 12, err)
        self.assertIn("GPU-bbbb", err)
        self.assertEqual(os.listdir(self.sb.locks), [], "核对在拿锁之前，不该碰锁目录")
        self.assertFalse(os.path.exists(self.out("run.pid")))
        rc, err = self.sb.run(entry, env={"GPU": "GPU-aaaa"})
        self.assertEqual(rc, 0, err)
        self.assertIn("Fake Card (UUID: GPU-aaaa)", read(self.out("launches.log")))

    def test_cpu_quota_below_minimum_refused(self):
        entry = self.sb.entry(GO, MIN_CORES="16")
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 13, err)
        self.assertIn("配额", err)
        rc, err = self.sb.run(entry, env={"MIN_CORES": "8"})
        self.assertEqual(rc, 0, err)
        self.assertIn("cpu_quota_cores 12.00", read(self.out("launches.log")))
        self.sb.write_file(self.sb.cpu_max, "max 100000\n")             # 没有配额上限也算够
        rc, err = self.sb.run(entry, env={"MIN_CORES": "16"})
        self.assertEqual(rc, 0, err)

    def test_nonsense_numbers_refused(self):
        entry = self.sb.entry(GO, POOL="gazebo")
        for k, v in (("IDLE_LIMIT_S", "abc"), ("IDLE_LIMIT_S", "-5"), ("IDLE_WINDOW_S", "0"), ("LOCK_LOG_S", "0"), ("IDLE_MAX_CORES", "")):
            rc, err = self.sb.run(entry, env={k: v})
            self.assertEqual(rc, 2, f"{k}={v!r}: {err}")
            self.assertIn(k, err)


class GazeboStartCondition(Base):
    """01:40 规则：拿到锁之后核对没有 gz sim、窗口内核数低于门槛；不满足就复查，有上限，到点照样起并记下核数。"""

    def go(self, env=None, **cfg):
        entry = self.sb.entry(GO, POOL="gazebo", **cfg)
        t0 = time.time()
        rc, err = self.sb.run(entry, env=env, timeout=30)
        return rc, err, time.time() - t0, rec_line(read(self.out("launches.log")), "start")

    def test_idle_machine_starts_at_once(self):
        rc, err, took, line = self.go()
        self.assertEqual(rc, 0, err)
        self.assertEqual((line["gz_count"], line["idle_timed_out"]), ("0", "0"), line)
        self.assertEqual(float(line["idle_cores"]), 0.0)
        self.assertLess(took, 2.5, "空闲的机器不该等")

    def test_busy_machine_starts_at_the_deadline_and_records_the_cores(self):
        with self.sb.busy_cpu(10):
            rc, err, took, line = self.go(IDLE_LIMIT_S="2")
        self.assertEqual(rc, 0, err)
        self.assertEqual(line["idle_timed_out"], "1", line)
        self.assertGreaterEqual(float(line["idle_cores"]), 8.0, "到点起跑时要把当时的核数记下来")
        self.assertGreaterEqual(took, 1.0)
        self.assertLess(took, 8.0, "等待必须有上限")
        self.assertIn("到点照样起跑", read(self.out("run.log")))

    def test_gz_processes_hold_it_until_the_deadline(self):
        rc, err, took, line = self.go(env={"FAKE_GZ_COUNT": "2"}, IDLE_LIMIT_S="2")
        self.assertEqual(rc, 0, err)
        self.assertEqual((line["gz_count"], line["idle_timed_out"]), ("2", "1"), line)
        self.assertGreaterEqual(took, 1.0)
        self.assertLess(took, 8.0)

    def test_starts_as_soon_as_the_machine_goes_idle(self):
        entry = self.sb.entry(GO, POOL="gazebo", IDLE_LIMIT_S="20")
        t0 = time.time()
        with self.sb.busy_cpu(10):
            p = self.sb.launch(entry)
            time.sleep(1.0)
        self.assertEqual(p.wait(timeout=30), 0)                  # 退出 with 之后 cpu.stat 停住 = 空闲
        line = rec_line(read(self.out("launches.log")), "start")
        self.assertEqual(line["idle_timed_out"], "0", line)
        self.assertLess(time.time() - t0, 10.0, "空闲后应在下一次复查时起跑，不该等到上限")

    def test_other_pools_ignore_the_cpu_load(self):
        entry = self.sb.entry(GO, POOL="gs3d")
        t0 = time.time()
        with self.sb.busy_cpu(10):
            rc, err = self.sb.run(entry)
        self.assertEqual(rc, 0, err)
        self.assertLess(time.time() - t0, 1.5)
        self.assertNotIn("idle_timed_out", read(self.out("launches.log")))

    def test_core_threshold_is_configurable_and_defaults_to_8(self):
        # 其它池并行占约 5 核（高于旧条款 1.5 核，低于新缺省 8 核）：缺省 8 核立即起跑，改配 3 核则等到点起跑
        with self.sb.busy_cpu(5):
            rc, err, took, line = self.go(IDLE_LIMIT_S="2")
        self.assertEqual(rc, 0, err)
        self.assertEqual((line["idle_max_cores"], line["idle_timed_out"]), ("8", "0"), line)
        self.assertLess(took, 2.0, "5 核低于缺省 8 核门槛，不应空等")
        self.sb.out = os.path.join(self.sb.base, "out_strict")
        with self.sb.busy_cpu(5):
            rc2, err2, took2, line2 = self.go(IDLE_MAX_CORES="3", IDLE_LIMIT_S="2")
        self.assertEqual(rc2, 0, err2)
        self.assertEqual((line2["idle_max_cores"], line2["idle_timed_out"]), ("3", "1"), line2)
        self.assertGreaterEqual(took2, 1.0, "5 核高于 3 核门槛，应等到上限才到点起跑")


class Arms(Base):
    """E-6、E-9 与每臂的 cgroup 快照：臂的成败、续跑、半成品改名、时序健康一行。"""

    WORK = """
    work() {   # 假臂：数一次运行次数，在自己的目录里留结果；FAILFLAG 存在时臂 b 失败
        echo x >> "$OUT/count.$ARM"; mkdir -p "$ARM_DIR"; echo "$ARM-data" > "$ARM_DIR/result.txt"
        [[ $ARM != b || ! -e "$FAILFLAG" ]]
    }
    exp_start
    exp_arm a work
    exp_arm b work
    exp_arm c work
    exp_finish 0 "跑完"
    """

    def count(self, arm):
        return len(read_or_empty(self.out(f"count.{arm}")).splitlines())

    def test_failed_arm_makes_done_nonzero_and_the_queue_goes_on(self):
        entry = self.sb.entry("""
            exp_start
            exp_arm good true
            exp_arm bad false
            exp_arm later touch "$OUT/later.touched"
            exp_finish 0 "跑完"
        """)
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 11, err)
        self.assertTrue(self.done().startswith("rc=11 "), self.done())
        self.assertIn("bad", self.done())
        self.assertNotIn("good", self.done())
        self.assertTrue(os.path.exists(self.out("later.touched")), "一个臂失败不该停掉后面的臂")
        self.assertTrue(os.path.exists(self.out("good/.ok")))
        self.assertFalse(os.path.exists(self.out("bad/.ok")))
        log = read(self.out("launches.log"))
        self.assertIn("arm good rc=0\n", log)
        self.assertIn("arm bad rc=1\n", log)

    def test_explicit_failure_rc_is_not_changed(self):
        entry = self.sb.entry("""
            exp_start
            exp_arm bad false
            exp_finish 7 "自己指定的失败"
        """)
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 7, err)
        self.assertTrue(self.done().startswith("rc=7 "), self.done())

    def test_exp_arm_returns_command_rc_to_caller(self):
        entry = self.sb.entry("""
            exp_start
            exp_arm step1 bash -c 'exit 7'
            echo "arm_rc=$?" > "$OUT/caller_seen"
            exp_finish 0 "跑完"
        """)
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 11, err)
        self.assertEqual(read(self.out("caller_seen")), "arm_rc=7\n")

    def test_failures_of_parallel_arms_are_collected(self):
        entry = self.sb.entry("""
            exp_start
            exp_arm a false & exp_arm b true & exp_arm c false &
            wait
            exp_finish 0 "跑完"
        """)
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 11, err)
        done = self.done()
        self.assertTrue("a " in done and "c " in done and "b " not in done.split("：", 1)[1], done)

    def test_resume_skips_finished_arms_and_redoes_the_failed_one_after_renaming_it(self):
        flag = os.path.join(self.sb.base, "failflag")
        open(flag, "w").close()
        entry = self.sb.entry(self.WORK, FAILFLAG=flag)
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 11, err)
        self.assertEqual([self.count(x) for x in "abc"], [1, 1, 1])
        os.remove(flag)
        rc, err = self.sb.run(entry)
        self.assertEqual(rc, 0, err)
        self.assertEqual([self.count(x) for x in "abc"], [1, 2, 1], "只该重跑失败的臂 b")
        parts = [d for d in os.listdir(self.sb.out) if d.startswith("b.partial-")]
        self.assertEqual(len(parts), 1, os.listdir(self.sb.out))
        self.assertEqual(read(os.path.join(self.sb.out, parts[0], "result.txt")), "b-data\n")
        self.assertTrue(os.path.exists(self.out("b/.ok")))
        self.assertFalse(os.path.exists(os.path.join(self.sb.out, parts[0], ".ok")))
        self.assertIn(f"partial b -> {parts[0]}\n", read(self.out("launches.log")))
        self.assertTrue(self.done().startswith("rc=0 "), self.done())

    def failing_mv_env(self):
        """PATH 最前面放一个只在带 -T 时失败的 mv（库里改名半成品用 mv -T），其余照常转给真的 mv。"""
        d = os.path.join(self.sb.base, "failbin")
        self.sb.write_file(os.path.join(d, "mv"), '#!/bin/bash\nfor a in "$@"; do [[ $a == -T ]] && exit 1; done\nexec /bin/mv "$@"\n')
        os.chmod(os.path.join(d, "mv"), 0o755)
        return {"PATH": d + ":" + self.sb.fakebin + ":" + os.environ["PATH"]}

    def test_failed_rename_of_an_old_arm_directory_stops_the_run(self):
        flag = os.path.join(self.sb.base, "failflag")
        open(flag, "w").close()
        entry = self.sb.entry(self.WORK, FAILFLAG=flag)
        self.assertEqual(self.sb.run(entry)[0], 11)
        os.remove(flag)
        rc, err = self.sb.run(entry, env=self.failing_mv_env())
        self.assertEqual(rc, 2, err)
        self.assertIn("改名失败", self.done())
        self.assertEqual(self.count("b"), 1, "旧目录没改走，臂 b 不该在上面重做")
        self.assertFalse([d for d in os.listdir(self.sb.out) if ".partial-" in d])

    def test_failed_rename_of_a_half_frozen_code_directory_stops_the_run(self):
        entry = self.sb.entry(FIN)
        self.sb.write_file(self.out("code/junk.txt"), "首次起跑死在解包途中留下的半成品\n")
        rc, err = self.sb.run(entry, env=self.failing_mv_env())
        self.assertEqual(rc, 4, err)
        self.assertIn("冻结代码失败", self.done())
        self.assertEqual(os.listdir(self.out("code")), ["junk.txt"], "不能在旧的半成品上解包")

    def test_sigterm_during_an_arm_leaves_no_ok_and_done_130(self):
        entry = self.sb.entry("""
            exp_start
            exp_arm slow bash -c 'mkdir -p "$ARM_DIR"; touch "$OUT/in_arm"; sleep 30'
            exp_finish 0 "不应走到这里"
        """)
        p = self.sb.launch(entry)
        wait_for(lambda: os.path.exists(self.out("in_arm")), what="臂起来", proc=p)
        os.killpg(p.pid, signal.SIGTERM)
        self.assertEqual(p.wait(timeout=15), 130)
        self.assertTrue(self.done().startswith("rc=130 "), self.done())
        self.assertFalse(os.path.exists(self.out("slow/.ok")), "被终止的臂不能留下完成标记")
        self.assertNotIn("arm slow rc=", read(self.out("launches.log")))
        self.assertEqual(group_members(p.pid), [])

    def test_health_line_comes_from_the_start_and_end_snapshots(self):
        entry = self.sb.entry("""
            exp_start
            exp_arm timed sleep 1.2
            exp_finish 0 "跑完"
        """)
        with self.sb.busy_cpu(3, throttle=0.5):
            rc, err = self.sb.run(entry)
        self.assertEqual(rc, 0, err)
        marks = read(self.out("timed.cg")).splitlines()
        self.assertEqual([m.split()[0] for m in marks], ["start", "end"])
        self.assertTrue(all(len(m.split()) == 5 for m in marks), marks)
        line = read(self.out("health.txt")).strip()
        import re
        m = re.fullmatch(r"timed 墙钟 (\d+)s 平均 ([\d.]+) 核 nr_throttled \+(\d+) 被限流 ([\d.]+)s（占墙钟 (\d+)%）", line)
        self.assertIsNotNone(m, line)
        wall, cores, nthr, thr_s, pct = (float(x) for x in m.groups())
        self.assertTrue(2.5 <= cores <= 3.5, line)
        self.assertTrue(10 < nthr < 200, line)
        self.assertTrue(0.4 <= thr_s <= 0.9, line)
        self.assertTrue(35 <= pct <= 65, line)
        self.assertIn(line, read(self.out("run.log")))


STOP = os.path.join(HERE, "stop_daemon.sh")


class StopDaemon(Base):
    """E-4：tools/stop_daemon.sh 核对 run.pid 记的身份，对得上才按 PGID 停；对不上不杀。"""

    def stop(self, out=None, wait="3"):
        r = subprocess.run(["bash", STOP, out or self.sb.out], capture_output=True, text=True, timeout=60,
                           env={"PATH": os.environ["PATH"], "STOP_WAIT_S": wait})
        return r.returncode, r.stdout, r.stderr

    def start_long(self):
        entry = self.sb.entry(LONG)
        p = self.sb.launch(entry)
        wait_for(lambda: os.path.exists(self.out("started")), what="入口起来", proc=p)
        return entry, p

    def bystander(self):
        """一个与实验无关的进程（自成一组），代表「PID 被别的进程用了」时会被误杀的对象。"""
        q = subprocess.Popen(["sleep", "60"], start_new_session=True, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: (q.kill(), q.wait()))
        return q

    def test_matching_record_stops_the_whole_group(self):
        entry, p = self.start_long()
        members = group_members(p.pid)
        self.assertGreaterEqual(len(members), 2, "入口加上它的 sleep 子进程")
        rc, out, err = self.stop()
        self.assertEqual(rc, 0, out + err)
        self.assertIn("核对通过", out)
        self.assertEqual(p.wait(timeout=15), 130)
        self.assertEqual(group_members(p.pid), [], "进程组里还有残留进程")
        self.assertTrue(self.done().startswith("rc=130 "), self.done())

    def test_forged_start_time_is_refused_and_nothing_is_killed(self):
        # 模拟容器重启后 PID 被别的进程用了：PID 与命令行都对得上，启动时刻对不上
        entry, p = self.start_long()
        rec = pairs(read(self.out("run.pid")))
        rec["start"] = str(int(rec["start"]) - 5000)
        self.sb.write_file(self.out("run.pid"), "".join(f"{k}={v}\n" for k, v in rec.items()))
        rc, out, err = self.stop()
        self.assertEqual(rc, 1, out + err)
        self.assertIn("启动时刻", err)
        self.assertIsNone(p.poll(), "对不上的记录不该杀掉进程")
        self.assertIsNone(self.done(), "没杀，入口不该写 DONE")
        self.assertGreaterEqual(len(group_members(p.pid)), 2)

    def test_unrelated_process_with_the_recorded_pid_survives(self):
        q = self.bystander()
        now = proc_stat(q.pid)[19]
        # 命令行与进程组都对得上，只有启动时刻不同（记录是更早的进程留下的）：不杀
        self.sb.write_file(self.out("run.pid"), f"pid={q.pid}\npgid={q.pid}\nstart={int(now) - 100}\ncmd=sleep 60\n")
        rc, out, err = self.stop()
        self.assertEqual(rc, 1, out + err)
        self.assertIn("启动时刻", err)
        self.assertIsNone(q.poll(), "无关进程被杀了")
        # 启动时刻对得上，但命令行不是记录里的入口：同样不杀
        self.sb.write_file(self.out("run.pid"), f"pid={q.pid}\npgid={q.pid}\nstart={now}\ncmd=bash /x/run.sh\n")
        rc, out, err = self.stop()
        self.assertEqual(rc, 1, out + err)
        self.assertIn("命令行", err)
        self.assertIsNone(q.poll(), "无关进程被杀了")

    def test_legacy_plain_pid_file_is_refused(self):
        q = self.bystander()
        self.sb.write_file(self.out("run.pid"), f"{q.pid}\n")
        rc, out, err = self.stop()
        self.assertEqual(rc, 2, out + err)
        self.assertIn("旧格式", err)
        self.assertIsNone(q.poll())

    def test_no_run_pid_is_a_usage_error(self):
        rc, out, err = self.stop(out=os.path.join(self.sb.base, "nowhere"))
        self.assertEqual(rc, 2, out + err)
        self.assertIn("run.pid", err)
        self.assertEqual(subprocess.run(["bash", STOP], capture_output=True).returncode, 2, "不带参数也是用法错误")

    def test_finished_entry_is_nothing_to_stop(self):
        entry = self.sb.entry(FIN)
        self.assertEqual(self.sb.run(entry)[0], 0)
        rc, out, err = self.stop()
        self.assertEqual(rc, 0, out + err)
        self.assertIn("已不存在", out)

    def test_orphans_left_after_the_entry_died_are_reported_not_killed(self):
        entry, p = self.start_long()
        members = [m for m in group_members(p.pid) if m != p.pid]
        os.kill(p.pid, signal.SIGKILL)                    # 只杀入口：它的 sleep 子进程成了孤儿，还在这个进程组里
        p.wait(timeout=15)
        try:
            rc, out, err = self.stop()
            self.assertEqual(rc, 4, out + err)
            self.assertIn(str(p.pid), err)
            self.assertEqual(group_members(p.pid), members, "无法核对身份的进程不该被杀")
        finally:
            os.killpg(p.pid, signal.SIGKILL)              # 测试自己起的组，自己清掉

    def test_survivors_after_term_are_reported_without_escalating(self):
        entry = self.sb.entry("""
            exp_start
            exp_arm stubborn bash -c 'trap "" TERM; touch "$OUT/in_arm"; sleep 30'
            exp_finish 0 "不应走到这里"
        """)
        p = self.sb.launch(entry)
        wait_for(lambda: os.path.exists(self.out("in_arm")), what="臂起来", proc=p)
        rc, out, err = self.stop(wait="2")
        self.assertEqual(rc, 3, out + err)
        self.assertIn("仍有进程", err)
        self.assertGreaterEqual(len(group_members(p.pid)), 2, "脚本不该自己升级成 SIGKILL")


class CheckShTemp(Base):
    """E-13：tools/check.sh 的临时目录与字节码不写根分区。

    用一个假解释器顶替 UAV_RGB_PYTHON：普通调用转给真解释器，走到 make_synthetic_episode.py（check.sh 里 mktemp 之后的第一个 Python 调用）
    时把环境与 --out 记下来就退出 77，check.sh 到不了后面的测试循环，不会套娃。
    """

    WRAPPER = """#!/bin/bash
for a in "$@"; do
  if [[ $a == *make_synthetic_episode.py ]]; then
    { echo "bytecode=${PYTHONDONTWRITEBYTECODE-<unset>}"; echo "tmpdir=${TMPDIR-<unset>}"
      while [[ $# -gt 0 ]]; do [[ $1 == --out ]] && echo "out=$2"; shift; done; } > "$PROBE"
    exit 77
  fi
done
exec "$REAL_PYTHON" "$@"
"""

    @staticmethod
    def tmp_parent(got):
        """--out 是 <mktemp 目录>/ep，mktemp 目录在 TMPDIR 下：返回 mktemp 目录所在的那一层。"""
        return os.path.dirname(os.path.dirname(got["out"]))

    def check(self, tmpdir=None, data_ready=True):
        data = os.path.join(self.sb.base, "data")
        if data_ready:
            os.makedirs(os.path.join(data, "uav_rgb", "tmp"))
        else:
            os.makedirs(data)
        wrapper, probe = os.path.join(self.sb.base, "py-wrapper"), os.path.join(self.sb.base, "probe")
        self.sb.write_file(wrapper, self.WRAPPER)
        os.chmod(wrapper, 0o755)
        env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/root"), "DATA_ROOT": data,
               "UAV_RGB_PYTHON": wrapper, "REAL_PYTHON": sys.executable, "PROBE": probe}
        if tmpdir:
            env["TMPDIR"] = tmpdir
        r = subprocess.run(["bash", os.path.join(REPO, "tools", "check.sh"), "--python-only"], env=env,
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        got = dict(line.split("=", 1) for line in read(probe).splitlines())
        return data, got

    def test_python_does_not_write_bytecode(self):
        data, got = self.check()
        self.assertEqual(got["bytecode"], "1")

    def test_default_goes_to_the_data_disk(self):
        data, got = self.check()
        self.assertEqual(got["tmpdir"], os.path.join(data, "uav_rgb", "tmp"))
        self.assertEqual(self.tmp_parent(got), os.path.join(data, "uav_rgb", "tmp"), "mktemp 的目录没跟着 TMPDIR 走")

    def test_callers_tmpdir_wins(self):
        mine = os.path.join(self.sb.base, "mytmp")
        os.makedirs(mine)
        data, got = self.check(tmpdir=mine)
        self.assertEqual(got["tmpdir"], mine)
        self.assertEqual(self.tmp_parent(got), mine)

    def test_without_data_disk_the_system_default_stays(self):
        data, got = self.check(data_ready=False)
        self.assertEqual(got["tmpdir"], "<unset>")
        self.assertEqual(self.tmp_parent(got), "/tmp")


TEMPLATE = os.path.join(REPO, "experiments", "_template", "run.sh")
TEMPLATE_README = os.path.join(REPO, "experiments", "_template", "README.md")


class Template(Base):
    """experiments/_template/run.sh 与 README.md：长度、配置区行尾依据与缺省值、必须选池、整条链路跑通并续跑。"""

    def put(self):
        dst = os.path.join(self.sb.repo, "experiments", "tpl", "run.sh")     # 与正式布局同深度，样板里的 ../../tools/exp_lib.sh 才找得到
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(TEMPLATE, dst)
        self.sb.commit("样板的副本")
        return dst

    def env(self, **kw):
        e = dict(OUT=self.sb.out, LOCK_DIR=self.sb.locks, CPU_MAX_FILE=self.sb.cpu_max, CPU_STAT_FILE=self.sb.cpu_stat, POOL="none")
        e.update({k: str(v) for k, v in kw.items()})
        return e

    def test_at_most_80_lines(self):
        self.assertLessEqual(len(read(TEMPLATE).splitlines()), 80)

    def test_config_lines_have_rationale_comments_and_defaults(self):
        cfg_lines = [ln for ln in read(TEMPLATE).splitlines() if ln.startswith("exp_cfg ")]
        self.assertGreaterEqual(len(cfg_lines), 17)
        for ln in cfg_lines:
            self.assertRegex(ln, r"^exp_cfg\s+\w+\s+.+#\s*\S+", f"配置行缺行尾取值依据注释：{ln}")
        parsed = {}
        for ln in cfg_lines:
            toks = shlex.split(ln.split("#", 1)[0])
            parsed[toks[1]] = toks[2]
        self.assertEqual(parsed["POOL"], "")
        self.assertEqual(parsed["IDLE_MAX_CORES"], "8")
        self.assertEqual(parsed["IDLE_WINDOW_S"], "30")
        self.assertEqual(parsed["IDLE_RECHECK_S"], "60")
        self.assertEqual(parsed["IDLE_LIMIT_S"], "600")
        self.assertEqual(parsed["LOCK_LOG_S"], "600")

    def test_readme_documents_how_to_start_a_new_experiment(self):
        self.assertTrue(os.path.isfile(TEMPLATE_README), "缺 experiments/_template/README.md")
        doc = read(TEMPLATE_README)
        for token in ("experiments/", "预注册", "POOL", "gazebo", "gs3d", "train",
                      "setsid nohup", "tools/stop_daemon.sh", "launches.log", "DONE", ".partial-"):
            self.assertIn(token, doc, f"README.md 缺关键说明项：{token}")

    def test_runs_end_to_end_and_resumes(self):
        tpl = self.put()
        rc, err = self.sb.run(tpl, env=self.env())
        self.assertEqual(rc, 0, err)
        self.assertTrue(self.done().startswith("rc=0 "), self.done())
        self.assertEqual(read(self.out("demo-a/result.txt")), "lanes=2 arm=demo-a\n")
        for arm in ("demo-a", "demo-b"):
            self.assertTrue(os.path.exists(self.out(f"{arm}/.ok")) and os.path.exists(self.out(f"{arm}.cg")), arm)
        self.assertEqual(os.path.exists(self.out("run.pid")) and pairs(read(self.out("run.pid")))["cmd"], f"bash {tpl}")
        overridden = re.findall(r"^override (\w+)=", read(self.out("launches.log")), re.M)
        self.assertEqual(sorted(overridden), ["CPU_MAX_FILE", "CPU_STAT_FILE", "LOCK_DIR", "OUT", "POOL"], "只该列出测试自己覆盖的项")
        rc, err = self.sb.run(tpl, env=self.env())                      # 续跑：两个臂都已完成
        self.assertEqual(rc, 0, err)
        self.assertEqual(read(self.out("run.log")).count("已完成，跳过"), 2)

    def test_editing_the_entry_in_place_while_it_runs_does_not_change_the_run(self):
        # bash 执行外部命令之后会按旧偏移量重新读脚本文件：文件在跑的时候被原地改写，后面的行就读错位。
        tpl = self.put()
        text = re.sub(r"(exp_arm demo-b arm_demo )1", r"\g<1>3", read(tpl))        # 让第二个臂睡 3 秒，给改文件留出时间
        self.assertNotEqual(text, read(tpl))
        self.sb.write_file(tpl, text)
        self.sb.commit("样板的副本，第二个臂睡 3 秒")
        p = self.sb.launch(tpl, env=self.env())
        wait_for(lambda: os.path.isdir(self.out("demo-b")), what="第二个臂开始", proc=p)
        with open(tpl, "w", encoding="utf-8") as f:                                  # 原地改写：同一个 inode，前面多出 400 多字节
            f.write("# " + "起跑后才加的一行注释，把后面的内容整体后移。" * 20 + "\n" + text)
        self.assertEqual(p.wait(timeout=30), 0, p.stderr.read().decode())
        log = read(self.out("run.log"))
        self.assertEqual(log.count("全部结束"), 1, log)
        self.assertEqual(log.count("臂 demo-b：开始"), 1, log)
        self.assertNotIn("not found", log)
        self.assertTrue(self.done().startswith("rc=0 "), self.done())

    def test_lanes_override_reaches_the_arm_and_the_record(self):
        tpl = self.put()
        rc, err = self.sb.run(tpl, env=self.env(LANES=3))
        self.assertEqual(rc, 0, err)
        self.assertEqual(read(self.out("demo-a/result.txt")), "lanes=3 arm=demo-a\n")
        self.assertIn("override LANES=3（缺省 2）", read(self.out("launches.log")))

    def test_refuses_until_a_pool_is_chosen(self):
        tpl = self.put()
        env = self.env()
        del env["POOL"]
        rc, err = self.sb.run(tpl, env=env)
        self.assertEqual(rc, 2, err)
        self.assertIn("POOL", err)
        self.assertEqual(os.listdir(self.sb.locks), [])

    def test_gazebo_pool_goes_through_lock_and_start_condition(self):
        tpl = self.put()
        env = self.env(POOL="gazebo", IDLE_WINDOW_S="0.3", IDLE_RECHECK_S="0.3", IDLE_LIMIT_S="2")
        rc, err = self.sb.run(tpl, env=env)
        self.assertEqual(rc, 0, err)
        self.assertEqual(os.listdir(self.sb.locks), ["machine.lock"])
        log = read(self.out("launches.log"))
        self.assertEqual(rec_line(log, "start")["idle_timed_out"], "0")
        self.assertIn("lock_wait_s=0", log)


class CrashGuard(Base):
    """入口没走到 exp_finish 就退出：补写 DONE，rc 取退出码（0 改成 1）；正常结束不重复写。"""

    def run_body(self, body):
        entry = self.sb.entry(body)
        rc, err = self.sb.run(entry)
        return rc, err

    def test_unbound_variable_leaves_a_done(self):
        rc, err = self.run_body("""
            exp_start
            echo "$NOT_DEFINED_ANYWHERE"
            exp_finish 0 "不应走到这里"
        """)
        self.assertEqual(rc, 1)
        self.assertTrue(self.done().startswith("rc=1 "), self.done())
        self.assertIn("没走到 exp_finish", self.done())
        self.assertIn("end rc=1", read(self.out("launches.log")))

    def test_exit_zero_in_the_middle_is_not_a_success(self):
        rc, err = self.run_body("""
            exp_start
            exit 0
        """)
        self.assertEqual(rc, 1)
        self.assertTrue(self.done().startswith("rc=1 "), self.done())

    def test_exit_code_is_kept(self):
        rc, err = self.run_body("""
            exp_start
            exit 5
        """)
        self.assertEqual(rc, 5)
        self.assertTrue(self.done().startswith("rc=5 "), self.done())

    def test_normal_finish_is_written_once(self):
        rc, err = self.run_body(FIN)
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.done().split()[0], "rc=0")
        self.assertEqual(read(self.out("run.log")).count("全部结束"), 1)
        self.assertEqual(read(self.out("launches.log")).count("end rc="), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
