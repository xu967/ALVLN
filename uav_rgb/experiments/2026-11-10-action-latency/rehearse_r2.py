#!/usr/bin/env python3
"""演练 run_r2.sh（第二轮）的每个分支：假评测入口 stub_eval_r2.py 写真实格式的 eval.json 与轨迹，探针（nvidia-smi、cpu.max、cpu.stat、gz 进程数、机器锁）
全部换成临时目录里的桩，run_r2.sh 照常走 起跑条件 → 冻结代码 → 阶梯 → 阶段 2 → 森林 → summarize_r2.py → DONE。不起 Gazebo，不碰真实机器锁，不占 gz。
第一轮的 rehearse.sh 不动；本文件是第二轮的版本，带自动核对（每个场景若干条断言，最后一张总表，有一条不过就退出码 1）。

用法：python rehearse_r2.py [场景 ...|all]    场景见 SCENARIOS；临时目录在 $TMPDIR（缺省系统临时目录）下 rehearse-r2-<场景>-*，不自动删。
冻结代码用真实提交（git archive HEAD，不起任何仿真）；summarize_r2.py 也从那份冻结副本里跑，所以修了汇总脚本要先提交再演练。
SELFCHECK 默认关（脚本未提交时无法核对），场景 selfcheck 单独演练它（脚本已提交时才有意义）。
环境变量 RUN_R2_SH 指向另一份 run_r2.sh（变异检查：往副本里注入缺陷，看对应场景是否变红）；缺省用本目录的 run_r2.sh。
"""
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.environ.get("RUN_R2_SH") or os.path.join(HERE, "run_r2.sh")   # 变异检查时指向注入过缺陷的副本
STUB = os.path.join(HERE, "stub_eval_r2.py")
TMP = os.environ.get("TMPDIR") or tempfile.gettempdir()
GPU = "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740"
ACC = ["acc-L{L}-k{k}-s{s}".format(L="{L}", k=k, s=s) for k in (1, 3) for s in (0, 1, 2)]
FOREST = ["forest-L{L}-sparse", "forest-L{L}-medium", "forest-L{L}-dense"]


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def head():
    return sh(["git", "-C", SRC, "rev-parse", "HEAD"]).stdout.strip()


def shared_code():
    """冻结代码只解一次，其它场景用符号链接指过去（passA / resume 走完整的 git archive 路径）。"""
    d = os.path.join(TMP, f"rehearse-r2-code-{head()[:10]}")
    if not os.path.isfile(os.path.join(d, "sim", "tools", "eval_closed_loop.py")):
        os.makedirs(d, exist_ok=True)
        arc = subprocess.Popen(["git", "-C", SRC, "archive", head()], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", d], stdin=arc.stdout, check=True)
        arc.wait()
    return d


class Ctx:
    def __init__(self, name, clause="cruise", gpu=GPU, cpu_max="1200000 100000", extra_env=None, link_code=True):
        self.name = name
        self.T = tempfile.mkdtemp(prefix=f"rehearse-r2-{name}-", dir=TMP)
        self.out = os.path.join(self.T, "out")
        os.makedirs(self.out)
        os.makedirs(os.path.join(self.T, "bin"))
        nv = os.path.join(self.T, "bin", "nvidia-smi")
        with open(nv, "w") as f:
            f.write(f"#!/usr/bin/env bash\necho 'GPU 0: NVIDIA Stub (UUID: {gpu})'\n")
        os.chmod(nv, 0o755)
        self.write("cpu.max", cpu_max + "\n")
        self.write("cpu.stat", "usage_usec 1000000\nuser_usec 0\nsystem_usec 0\nnr_periods 0\nnr_throttled 0\nthrottled_usec 0\n")
        self.write("gz_count", "0\n")
        open(os.path.join(self.T, "machine.lock"), "a").close()
        if link_code:
            os.symlink(shared_code(), os.path.join(self.out, "code"))
        with open(RUN, "rb") as src, open(os.path.join(self.out, "run_r2.sh"), "wb") as dst:
            dst.write(src.read())
        self.env = {**os.environ, "OUT": self.out, "MACHINE_LOCK": os.path.join(self.T, "machine.lock"), "NVIDIA_SMI": nv,
                    "CPU_MAX_FILE": os.path.join(self.T, "cpu.max"), "CPU_STAT_FILE": os.path.join(self.T, "cpu.stat"),
                    "GZ_PROBE": f"cat {self.T}/gz_count", "EVAL": STUB, "SELFCHECK": "0", "LANES": "2", "IDLE_WINDOW_S": "1",
                    "IDLE_RECHECK_S": "1", "IDLE_LOG_S": "2", "LOCK_LOG_S": "2", "IDLE_LIMIT_S": "60", "STUB_K3_LATE_AT": "",
                    "TMPDIR": os.path.join(self.T, "tmp")}
        os.makedirs(self.env["TMPDIR"], exist_ok=True)
        if clause:
            self.env["DISP_CLAUSE"] = clause
        else:
            self.env.pop("DISP_CLAUSE", None)
        self.env.update(extra_env or {})
        self.bg = []
        self.checks = []

    def write(self, name, text):
        with open(os.path.join(self.T, name), "w") as f:
            f.write(text)

    def read(self, rel, default=""):
        p = os.path.join(self.out, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                return f.read()
        return default

    def popen(self, script=None):
        return subprocess.Popen(["bash", script or os.path.join(self.out, "run_r2.sh")], env=self.env, cwd=self.out, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True, start_new_session=True)

    def run(self, timeout=420):
        p = self.popen()
        try:
            _o, err = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            _o, err = p.communicate()
            self.check(False, f"超时 {timeout}s，已按进程组终止")
        return p.returncode, err

    def spawn(self, code):
        """后台辅助进程（占锁、造 CPU 负载…），场景结束时按进程组收掉。"""
        p = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
        self.bg.append(p)
        return p

    def cleanup(self):
        for p in self.bg:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass

    def check(self, ok, text):
        self.checks.append((bool(ok), text))

    # ---- 常用断言
    def done(self):
        return self.read("DONE").strip()

    def done_rc(self):
        m = re.match(r"rc=(\d+)", self.done())
        return int(m.group(1)) if m else None

    def arm(self, pattern, **fmt):
        return os.path.isdir(os.path.join(self.out, pattern.format(**fmt)))

    def expect_done(self, rc, *needles):
        self.check(self.done_rc() == rc, f"DONE rc={rc}（实际：{self.done()[:140]!r}）")
        for n in needles:
            self.check(n in self.done(), f"DONE 含「{n}」")

    def expect_log(self, *needles):
        log = self.read("run.log")
        for n in needles:
            self.check(n in log, f"run.log 含「{n}」")

    def expect_arms(self, present=(), absent=(), **fmt):
        for a in present:
            self.check(self.arm(a, **fmt), f"臂 {a.format(**fmt)} 存在")
        for a in absent:
            self.check(not self.arm(a, **fmt), f"臂 {a.format(**fmt)} 不存在")

    def args_of(self, arm):
        return self.read(os.path.join(arm, "stub_args.txt")).strip()


def busy_writer(cpu_stat, seconds, cores=4.0):
    """后台：seconds 秒内每 0.2 s 给 cpu.stat 的 usage_usec 加 cores × 0.2 s，模拟容器被占满。"""
    return (f"import time,os\nt=time.time()\nwhile time.time()-t<{seconds}:\n"
            f" kv=dict(l.split() for l in open('{cpu_stat}') if l.strip())\n kv['usage_usec']=str(int(kv['usage_usec'])+int({cores}*0.2*1e6))\n"
            f" open('{cpu_stat}.tmp','w').write(''.join(f'{{k}} {{v}}\\n' for k,v in kv.items()))\n os.replace('{cpu_stat}.tmp','{cpu_stat}')\n time.sleep(0.2)\n")


def lock_holder(lock, seconds):
    return f"import fcntl,time\nf=open('{lock}','a')\nfcntl.flock(f,fcntl.LOCK_EX)\ntime.sleep({seconds})\n"


def full_arms(L="0.20", with_forest=True):
    return [a.format(L=L) for a in ACC] + ([f.format(L=L) for f in FOREST] if with_forest else [])


# ---------------------------------------------------------------- 场景
def sc_passA(c):
    rc, _ = c.run()
    c.check(rc == 0, f"进程退出码 0（实际 {rc}）")
    c.expect_done(0, "标定 A 组过门（L=0.20", "位移口径 cruise")
    c.expect_arms(["cal-L0.20-k1", "cal-L0.20-k3"] + full_arms(), ["cal-L0.20rtf0.5-k3", "cal-L0.24-k1", "cal-L0.24-k3", "acc-L0.20-k1-s1-retry1"])
    c.check(c.read("L_STAR").strip() == "0.20", "L_STAR = 0.20")
    c.check(not os.path.exists(os.path.join(c.out, "RTF_STAR")), "没有 RTF_STAR")
    c.check(all("--rtf" not in c.args_of(a) for a in full_arms()), "学生臂与森林臂的命令行都没有 --rtf")
    c.check(all("--action-latency 0.20" in c.args_of(a) for a in full_arms()), "学生臂与森林臂都带 --action-latency 0.20")
    c.check("--heading-guard 2" in c.args_of("acc-L0.20-k1-s1") and "--ckpt" in c.args_of("acc-L0.20-k1-s1"), "学生臂带 --heading-guard 2 与权重")
    c.check("--blocks 3" in c.args_of("acc-L0.20-k3-s0") and "--task course_base" in c.args_of("acc-L0.20-k3-s0"), "K=3 学生臂是 course_base --blocks 3")
    cgs = [a for a in full_arms() + ["cal-L0.20-k1", "cal-L0.20-k3"] if re.search(r"^start .*\nend ", c.read(a + ".cg"))]
    c.check(len(cgs) == len(full_arms()) + 2, f"每个臂都有 .cg 的 start 与 end 两行（{len(cgs)}/{len(full_arms()) + 2}）")
    log = c.read("run.log")
    c.check(len(re.findall(r"usage_usec=\d+ nr_throttled=\d+", log)) >= 2 * (len(full_arms()) + 2), "run.log 每个臂开始与结束各记 usage_usec 与 nr_throttled")
    c.expect_log("条件 ① 满足", "条件 ② 满足", "条件 ③ 满足", "条件 ④ 满足", "标定门 A rc=0")
    c.check(c.read("code_head.txt").strip() == head(), "code_head.txt = 当前 HEAD")
    st = c.read("start.txt")
    c.check("gpu_ok=是" in st and "lanes=2" in st, "start.txt 记录起跑条件与 LANES=2")
    c.check("DISP_CLAUSE=cruise" in c.read("config.env"), "config.env 记录位移口径 cruise")
    s = c.read("summary.md")
    c.check("标定阶梯：A 组通过" in s and "新协议参数：L = 0.20 s" in s and "## 7. 森林三档" in s, "summary.md 有阶梯结论、新协议参数与森林节")
    c.check(os.path.isfile(os.path.join(c.out, "inputs.txt")) and GPU in c.read("inputs.txt").replace("\n", ""), "inputs.txt 记录 GPU 列表")


def sc_ladderB(c):
    c.env["STUB_K3_LATE_AT"] = "0.20"
    rc, _ = c.run()
    c.expect_done(0, "标定 B 组过门（L=0.20，教师 K=3 带 --rtf 0.5）")
    c.expect_arms(["cal-L0.20-k1", "cal-L0.20-k3", "cal-L0.20rtf0.5-k3"] + full_arms(), ["cal-L0.24-k1", "cal-L0.24-k3"])
    c.check("--rtf 0.5" in c.args_of("cal-L0.20rtf0.5-k3"), "B 组 K=3 带 --rtf 0.5")
    c.check("--rtf" not in c.args_of("cal-L0.20-k1"), "A 组 K=1 不带 --rtf（B 组复用它）")
    c.check(all("--rtf" not in c.args_of(a) for a in full_arms()), "学生臂与森林臂都不带 --rtf（即使选了 B 组）")
    c.check(c.read("RTF_STAR").strip() == "0.5" and c.read("L_STAR").strip() == "0.20", "L_STAR = 0.20，RTF_STAR = 0.5")
    c.expect_log("标定门 A rc=1", "A 组未过且原因含迟到步，试 B 组", "标定门 B rc=0")
    c.check("教师 K=3 要带 `--rtf 0.5` 才过门，学生臂不带 `--rtf`" in c.read("summary.md"), "summary.md 写明教师 K=3 要带 --rtf 0.5、学生臂不带")


def sc_ladderC(c):
    c.env["STUB_K3_LATE_AT"] = "0.20,0.20rtf0.5"
    rc, _ = c.run()
    c.expect_done(0, "标定 C 组过门（L=0.24")
    c.expect_arms(["cal-L0.20-k1", "cal-L0.20-k3", "cal-L0.20rtf0.5-k3", "cal-L0.24-k1", "cal-L0.24-k3"] + full_arms("0.24"), full_arms("0.20"))
    c.check(c.read("L_STAR").strip() == "0.24" and not os.path.exists(os.path.join(c.out, "RTF_STAR")), "L_STAR = 0.24，没有 RTF_STAR")
    c.check(all("--action-latency 0.24" in c.args_of(a) for a in full_arms("0.24")), "学生臂与森林臂带 --action-latency 0.24")
    c.expect_log("标定门 A rc=1", "标定门 B rc=1", "B 组未过且原因含迟到步，试 C 组", "标定门 C rc=0")


def sc_failAll(c):
    c.env["STUB_K3_LATE_AT"] = "0.20,0.20rtf0.5,0.24"
    rc, _ = c.run()
    c.expect_done(10, "三组都没过")
    c.expect_arms(["cal-L0.20-k1", "cal-L0.20-k3", "cal-L0.20rtf0.5-k3", "cal-L0.24-k1", "cal-L0.24-k3"], [a.format(L="0.20") for a in ACC] + [a.format(L="0.24") for a in ACC])
    c.check(not os.path.exists(os.path.join(c.out, "L_STAR")), "没有 L_STAR")
    c.check("标定阶梯：C 组未过（含迟到步），三组都没过，停" in c.read("summary.md"), "summary.md 写明三组都没过")


def sc_dispOnly(c):
    c.env.update(STUB_K3_LATE_AT="", STUB_K3_FWD_AT="0.20", STUB_K3_FWD_SCALE="0.95")
    rc, _ = c.run()
    c.expect_done(10, "A 组未过，原因里没有迟到步", "按规则停")
    c.expect_arms(["cal-L0.20-k1", "cal-L0.20-k3"], ["cal-L0.20rtf0.5-k3", "cal-L0.24-k1", "cal-L0.24-k3"] + [a.format(L="0.20") for a in ACC])
    c.expect_log("标定门 A rc=3")


def clause_case(c, clause, rc_want):
    c.env.update(STUB_K3_LATE_AT="", STUB_K3_RUN_LEN="20")      # K=3 直线更长，层内位移不变：steady 口径不过，cruise 口径过
    rc, _ = c.run()
    c.check(c.done_rc() == rc_want, f"DONE rc={rc_want}（口径 {clause}；实际 {c.done()[:120]!r}）")
    c.check(f"DISP_CLAUSE={clause}" in c.read("config.env"), f"config.env 记录 DISP_CLAUSE={clause}")
    c.expect_log(f"位移口径={clause}")


def sc_clauseCruise(c):
    clause_case(c, "cruise", 0)
    c.expect_log("标定门 A rc=0")


def sc_clauseSteady(c):
    clause_case(c, "steady", 10)
    c.expect_log("标定门 A rc=3")
    c.expect_done(10, "没有迟到步")


def sc_noclause(c):
    rc, err = c.run()
    c.check(rc == 5, f"进程退出码 5（实际 {rc}）")
    c.expect_done(5, "没给 DISP_CLAUSE")
    c.check("DISP_CLAUSE" in err, "stderr 提示没给 DISP_CLAUSE")
    c.check(not os.path.exists(os.path.join(c.out, "run.pid")) and not c.arm("cal-L0.20-k1"), "没有起跑（无 run.pid、无臂）")


def sc_nogpu(c):
    rc, _ = c.run()
    c.check(rc == 12, f"进程退出码 12（实际 {rc}）")
    c.expect_done(12, "条件 ① 不满足")
    c.check(not c.arm("cal-L0.20-k1"), "没有起跑")


def sc_lowquota(c):
    rc, _ = c.run()
    c.check(rc == 13, f"进程退出码 13（实际 {rc}）")
    c.expect_done(13, "条件 ② 不满足", "8.00 核")
    c.check(not c.arm("cal-L0.20-k1"), "没有起跑")


def sc_lockwait(c):
    c.spawn(lock_holder(os.path.join(c.T, "machine.lock"), 12))
    time.sleep(0.5)
    rc, _ = c.run()
    c.expect_done(0)
    c.expect_log("条件 ③：机器锁", "被占用，阻塞等待", "条件 ③：仍在等机器锁", "条件 ③ 满足：拿到机器锁，等了")
    m = re.search(r"拿到机器锁，等了 (\d+) s", c.read("run.log"))
    c.check(m and int(m.group(1)) >= 3, f"等锁秒数 ≥ 3（实际 {m.group(1) if m else None}）")
    m2 = re.search(r"lock_wait_s=(\d+)", c.read("start.txt"))
    c.check(m2 and int(m2.group(1)) >= 3, "start.txt 的 lock_wait_s 与之一致")
    log = c.read("run.log")
    c.check(log.index("条件 ② 满足") < log.index("条件 ③：机器锁") < log.index("标定组 A 开始"), "顺序：②、等锁、开跑")


def sc_gzwait(c):
    c.write("gz_count", "2\n")
    c.spawn(f"import time\ntime.sleep(12)\nopen('{c.T}/gz_count','w').write('0\\n')\n")
    rc, _ = c.run()
    c.expect_done(0)
    c.expect_log("条件 ④ 不满足：gz sim 进程 2 个", "条件 ④ 满足：gz sim 进程 0 个")
    m = re.search(r"idle_wait_s=(\d+)", c.read("start.txt"))
    c.check(m and int(m.group(1)) >= 3, "start.txt 的 idle_wait_s ≥ 3")
    log = c.read("run.log")
    c.check(log.index("条件 ④ 满足") < log.index("标定组 A 开始"), "④ 满足之后才开跑")


def sc_cpuwait(c):
    c.spawn(busy_writer(os.path.join(c.T, "cpu.stat"), 12, cores=4.0))
    time.sleep(0.3)
    rc, _ = c.run()
    c.expect_done(0)
    c.expect_log("条件 ④ 不满足", "条件 ④ 满足")
    c.check(re.search(r"条件 ④ 不满足：gz sim 进程 0 个，1 s 窗口内容器 [3-4]\.\d+ 核", c.read("run.log")) is not None, "不满足的那行写了窗口内容器核数（约 4 核）与 0 个 gz")
    m = re.search(r"idle_wait_s=(\d+)", c.read("start.txt"))
    c.check(m and int(m.group(1)) >= 3, "start.txt 的 idle_wait_s ≥ 3")


def sc_idletimeout(c):
    c.env["IDLE_LIMIT_S"] = "6"
    c.spawn(busy_writer(os.path.join(c.T, "cpu.stat"), 90, cores=4.0))
    time.sleep(0.3)
    rc, _ = c.run()
    c.check(rc == 14, f"进程退出码 14（实际 {rc}）")
    c.expect_done(14, "条件 ④ 等了", "超过 6 s", "没有起跑")
    c.check(not c.arm("cal-L0.20-k1"), "没有起跑")
    c.check(not os.path.exists(os.path.join(c.out, "start.txt")), "没有 start.txt")


def sc_retry(c):
    c.env.update(STUB_STUDENT_LATE_SEEDS="172", STUB_STUDENT_LATE_ARM="acc-L0.20-k1-s1")
    rc, _ = c.run()
    c.expect_done(0)
    c.expect_arms(["acc-L0.20-k1-s1", "acc-L0.20-k1-s1-retry1"], ["acc-L0.20-k3-s1-retry1", "acc-L0.20-k1-s0-retry1"])
    a = c.args_of("acc-L0.20-k1-s1-retry1")
    c.check("--seeds 172 " in a + " " and "--episodes 1" in a and "--heading-guard 2" in a and "--ckpt" in a, f"重跑臂只含 seed 172，其余参数同原臂（{a[:160]}）")
    e = json.loads(c.read("acc-L0.20-k1-s1/eval.json"))
    c.check(e["failed_count"] == 1 and e["late_dispatch_episodes"][0]["seed"] == 172, "原臂保留，迟到集 seed 172 仍在 failed[]")
    e2 = json.loads(c.read("acc-L0.20-k1-s1-retry1/eval.json"))
    c.check(e2["failed_count"] == 0 and e2["episodes_done"] == 1, "重跑臂 1 集干净")
    s = c.read("summary.md")
    c.check("重跑后干净，采用重跑结果" in s and "| acc-L0.20-k1-s1 | 172 |" in s, "summary.md 第 6 节列出重跑")
    c.expect_log("含迟到步的集 [172]，重跑一次 → acc-L0.20-k1-s1-retry1")


def sc_retryLate(c):
    c.env.update(STUB_STUDENT_LATE_SEEDS="172", STUB_STUDENT_LATE_ARM="acc-L0.20-k1-s1", STUB_RETRY_LATE="1")
    rc, _ = c.run()
    c.expect_done(0)
    e2 = json.loads(c.read("acc-L0.20-k1-s1-retry1/eval.json"))
    c.check(e2["failed_count"] == 1, "重跑仍迟到（failed 1）")
    c.check("重跑后仍含迟到步，按没成功算" in c.read("summary.md"), "summary.md 写「重跑后仍含迟到步，按没成功算」")
    retries = [d for d in os.listdir(c.out) if d.endswith("-retry1")]
    c.check(retries == ["acc-L0.20-k1-s1-retry1"], f"只重跑一次（{retries}）")


def sc_throttle(c):
    c.env["STUB_THROTTLE"] = "1"
    rc, _ = c.run()
    c.expect_done(0)
    s = c.read("summary.md")
    c.check(re.search(r"\| acc-L0\.20-k1-s1 \|.*\| \d+ ⚠ \|\n", s) is not None, "summary.md 里学生臂的限流增量带 ⚠")
    c.check("有 ⚠ 的臂不自动作废" in s, "summary.md 说明 ⚠ 不自动作废")
    c.check(re.search(r"\| cal-L0\.20-k1 \|.*\| \d+ ⚠ \|\n", s) is not None, "标定臂也带 ⚠")


def sc_resume(c):
    rc, _ = c.run()
    c.expect_done(0)
    n1 = len(re.findall(r"已评完，跳过", c.read("run.log")))
    head1 = c.read("code_head.txt")
    summary1 = c.read("L_STAR")
    rc2, _ = c.run()
    n2 = len(re.findall(r"已评完，跳过", c.read("run.log")))
    c.check(rc2 == 0 and c.done_rc() == 0, "第二次起跑 DONE rc=0")
    c.check(n2 - n1 == len(full_arms()) + 2, f"第二次所有臂都跳过（跳过 {n2 - n1} 行，期望 {len(full_arms()) + 2}）")
    c.check(c.read("code_head.txt") == head1 and c.read("L_STAR") == summary1, "code_head.txt 与 L_STAR 不变")
    c.env["DISP_CLAUSE"] = "steady"
    rc3, err3 = c.run()
    c.check(rc3 == 5 and "不一致" in err3, "续跑时给了与 config.env 不同的口径：rc=5，不起跑")
    c.check(c.done_rc() == 0, "拒绝起跑不覆盖已有的 DONE（仍是上一次的 rc=0）")
    c.check("不一致" in c.read("refused.txt"), "拒绝原因写进 refused.txt")
    del c.env["DISP_CLAUSE"]
    rc4, _ = c.run()
    c.check(rc4 == 0, "续跑时不给 DISP_CLAUSE：沿用 config.env 里的口径，rc=0")


def sc_dup(c):
    c.env["STUB_SLEEP"] = "2"
    p1 = c.popen()
    t0 = time.time()
    while "标定组 A 开始" not in c.read("run.log") and time.time() - t0 < 60:
        time.sleep(0.3)
    p2 = c.popen()
    try:
        _o, err2 = p2.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p2.pid, signal.SIGTERM)
        _o, err2 = p2.communicate()
    c.check(p2.returncode == 3 and "已有实例在跑" in err2, f"第二个实例 rc=3 且提示已有实例（rc={p2.returncode}）")
    try:
        p1.communicate(timeout=420)
    except subprocess.TimeoutExpired:
        os.killpg(p1.pid, signal.SIGTERM)
        p1.communicate()
    c.check(p1.returncode == 0 and c.done_rc() == 0, "第一个实例照常跑完 DONE rc=0")
    c.check(len(re.findall(r"开始 PGID=", c.read("run.log"))) == 1, "run.log 里只有一次「开始 PGID」")


def sc_selfcheck(c):
    """脚本版本核对：与 HEAD 里同名文件一致才往下走；改过一个字节的副本 rc=4。HEAD 里还没有 run_r2.sh（未提交）时本场景只报告跳过。"""
    rel = "experiments/2026-11-10-action-latency/run_r2.sh"
    committed = sh(["git", "-C", SRC, "show", f"HEAD:{rel}"])
    if committed.returncode != 0:
        c.check(False, "SKIP：run_r2.sh 还没提交，无法演练脚本版本核对（提交后再跑本场景）")
        return
    with open(RUN, "rb") as f:
        same = committed.stdout.encode() == f.read()
    # 与 HEAD 一致的副本：通过核对，之后在条件 ④ 停下（gz 进程数恒为 1，只等 3 s），所以 rc=14 = 过了版本核对
    c.write("gz_count", "1\n")
    c.env.update(SELFCHECK="1", IDLE_LIMIT_S="3")
    rc, _ = c.run(timeout=60)
    if same:
        c.check(c.done_rc() == 14, f"与 HEAD 一致的脚本通过版本核对，后面在条件 ④ 停下 rc=14（实际 {c.done()[:100]!r}）")
    else:
        c.check(c.done_rc() == 4, f"工作区的脚本与 HEAD 里的不一致：rc=4（实际 {c.done()[:100]!r}）")
    # 改一个字节的副本必须被拒绝，且没有起跑
    c2 = Ctx("selfcheck2", extra_env={"SELFCHECK": "1"})
    try:
        with open(os.path.join(c2.out, "run_r2.sh"), "ab") as f:
            f.write(b"# tamper\n")
        rc2, _ = c2.run(timeout=60)
        c.check(rc2 == 4 and c2.done_rc() == 4 and "不一致" in c2.done(), f"改过一个字节的副本：rc=4 且 DONE 说不一致（rc={rc2}）")
        c.check(not c2.arm("cal-L0.20-k1"), "被拒绝后没有起跑")
    finally:
        c2.cleanup()


def sc_partial(c):
    d = os.path.join(c.out, "cal-L0.20-k1")
    os.makedirs(os.path.join(d, "episodes", "seed_2"))
    with open(os.path.join(d, "eval.json"), "w") as f:
        json.dump({"partial": True, "episodes_requested": 10, "episodes_done": 1, "failed_count": 0, "per_episode": []}, f)
    with open(os.path.join(c.out, "cal-L0.20-k1.log"), "w") as f:
        f.write("old partial log\n")
    rc, _ = c.run()
    c.expect_done(0)
    parts = [x for x in os.listdir(c.out) if x.startswith("cal-L0.20-k1.partial-")]
    c.check(any(os.path.isdir(os.path.join(c.out, x)) for x in parts), f"半成品目录挪到 cal-L0.20-k1.partial-*（{parts}）")
    c.check(any(x.endswith(".log") for x in parts), "旧日志一并挪走，没有被覆盖")
    e = json.loads(c.read("cal-L0.20-k1/eval.json"))
    c.check(not e["partial"] and e["episodes_done"] == 10, "重跑后的臂完整（10 集）")


def sc_sigterm(c):
    c.env["STUB_SLEEP"] = "4"
    p = c.popen()
    t0 = time.time()
    while "标定组 A 开始" not in c.read("run.log") and time.time() - t0 < 60:
        time.sleep(0.3)
    time.sleep(1.5)
    os.killpg(p.pid, signal.SIGTERM)
    try:
        p.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
    c.check(c.done_rc() == 130, f"DONE rc=130（实际 {c.done()[:100]!r}）")
    time.sleep(0.5)
    left = sh(["pgrep", "-f", c.T]).stdout.split()
    left = [x for x in left if int(x) != os.getpid()]
    c.check(not left, f"按进程组终止后没有残留进程（{left}）")
    with open(os.path.join(c.T, "machine.lock"), "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            c.check(True, "机器锁已释放")
        except OSError:
            c.check(False, "机器锁仍被占用")


SCENARIOS = {
    "passA": (sc_passA, dict(link_code=False)),
    "ladderB": (sc_ladderB, {}),
    "ladderC": (sc_ladderC, {}),
    "failAll": (sc_failAll, {}),
    "dispOnly": (sc_dispOnly, {}),
    "clauseCruise": (sc_clauseCruise, dict(clause="cruise")),
    "clauseSteady": (sc_clauseSteady, dict(clause="steady")),
    "noclause": (sc_noclause, dict(clause=None)),
    "nogpu": (sc_nogpu, dict(gpu="GPU-00000000-0000-0000-0000-000000000000")),
    "lowquota": (sc_lowquota, dict(cpu_max="800000 100000")),
    "lockwait": (sc_lockwait, {}),
    "gzwait": (sc_gzwait, {}),
    "cpuwait": (sc_cpuwait, {}),
    "idletimeout": (sc_idletimeout, {}),
    "retry": (sc_retry, {}),
    "retryLate": (sc_retryLate, {}),
    "throttle": (sc_throttle, {}),
    "resume": (sc_resume, dict(link_code=False)),
    "dup": (sc_dup, {}),
    "selfcheck": (sc_selfcheck, {}),
    "partial": (sc_partial, {}),
    "sigterm": (sc_sigterm, {}),
}


def main(argv):
    names = list(SCENARIOS) if not argv or argv == ["all"] else argv
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"未知场景 {unknown}；可选 {list(SCENARIOS)}")
        return 2
    rows, failed = [], 0
    for name in names:
        fn, kw = SCENARIOS[name]
        t0 = time.time()
        c = Ctx(name, **kw)
        try:
            fn(c)
        except Exception as e:                       # 场景自己出错也记成不过，不拖垮后面的场景
            c.check(False, f"场景抛出异常：{type(e).__name__}: {e}")
        finally:
            c.cleanup()
        bad = [t for ok, t in c.checks if not ok]
        failed += bool(bad)
        rows.append((name, len(c.checks) - len(bad), len(c.checks), time.time() - t0, c.out))
        print(f"\n=== {name}：{len(c.checks) - len(bad)}/{len(c.checks)} 条断言通过（{time.time() - t0:.0f} s）  {c.out}")
        for ok, t in c.checks:
            print(f"  {'通过' if ok else '未过'}  {t}")
        print(f"  DONE: {c.done()[:200]}")
        sys.stdout.flush()
    print("\n场景总表")
    print("| 场景 | 通过 / 总断言 | 用时 s |")
    print("|---|---|---|")
    for name, ok, n, dt, _out in rows:
        print(f"| {name} | {ok}/{n}{'' if ok == n else ' 未过'} | {dt:.0f} |")
    print(f"\n{len(rows) - failed}/{len(rows)} 个场景全部断言通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
