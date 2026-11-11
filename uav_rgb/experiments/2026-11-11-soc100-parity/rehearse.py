#!/usr/bin/env python3
"""演练 run.sh 与 summarize_parity.py 的每个分支（不起 Gazebo）：假评测入口 stub_eval.py 写真实格式的 eval.json 与轨迹，探针入口换成桩
（--check-only 转给真实的 probe_plugin_params.py，其余用真实 judge_probe / report_md 在合成遥测上出 probe.json），
nvidia-smi、cpu.max、cpu.stat、gz 进程数、机器锁都换成临时目录里的桩。run.sh 照常走：起跑条件 → 冻结代码 → 副本核对 → 等锁 → 第二轮与输入核对 →
探针 → 六个满电臂 → summarize_parity.py → DONE。不起 Gazebo，不碰真实机器锁，不占 gz，不动真实的第二轮目录（演练里的第二轮是桩生成的假目录）。
每个场景若干条断言，最后一张总表，有一条不过就退出码 1。

用法：python rehearse.py [场景 ...|all]    场景见 SCENARIOS；临时目录在 $TMPDIR（缺省系统临时目录）下 rehearse-soc100-<场景>-*，不自动删。
冻结代码用真实提交（git archive HEAD，不起任何仿真）；summarize_parity.py 也从那份冻结副本里跑，所以改了脚本要先提交再演练。
环境变量 REHEARSE_SRC 指向另一个 git 仓库（改了还没提交时，把工作树快照成临时仓库再演练）；RUN_SH 指向另一份 run.sh（变异检查：往副本里注入缺陷，
看对应场景是否变红）；缺省用本目录的 run.sh。SELFCHECK 默认关（脚本未提交时无法核对），场景 selfcheck 单独演练它（脚本已提交时才有意义）。
"""
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SRC = os.environ.get("REHEARSE_SRC") or REPO
RUN = os.environ.get("RUN_SH") or os.path.join(HERE, "run.sh")
STUB = os.path.join(HERE, "stub_eval.py")
TMP = os.environ.get("TMPDIR") or tempfile.gettempdir()
D = "/root/rivermind-data/uav_rgb"
PY = "/root/claude-code-backup-20261028/code/uav_rgb/.venv-tools/bin/python"
GPU = "GPU-b544b42e-15d3-c9c8-1bdb-4c339775a740"
PARAMS = os.path.join(D, "params", "e1_mini_soc100.json")
SEEDS = "1,113,122,153,172,176,180,230,250,536,555,628,683,798,929,990,992,997,1113,1654"
SWA = f"{D}/exp-swa-calib-20261110/ckpt"
INFER = f"{D}/bin-snapshots/main-97ae04b/uavrgb_infer"
PLUGIN_SO = f"{D}/tmp/build-cf-sitl-lib31/gz/libCfSitlSystem.so"
ARMS = [(k, s) for k in (1, 3) for s in (0, 1, 2)]

PROBE_STUB = r'''#!/usr/bin/env python3
"""演练用探针桩：--check-only 转给真实探针；其余在合成遥测上用真实 judge_probe / report_md 出 probe.json，退出码取 PROBE_STUB_RC（缺省 0）。"""
import json, os, subprocess, sys
real_dir = os.environ["REAL_PROBE_DIR"]
args = sys.argv[1:]
if "--check-only" in args:
    sys.exit(subprocess.call([os.environ["REAL_PY"], os.path.join(real_dir, "probe_plugin_params.py")] + args))
sys.path.insert(0, real_dir)
import probe_plugin_params as pp
rc = int(os.environ.get("PROBE_STUB_RC", "0"))
out = args[args.index("--out") + 1]
params = os.path.abspath(args[args.index("--params") + 1])
os.makedirs(out, exist_ok=True)
default = "/stub/sim/dynamics/e1_mini.json"
runs = []
gap = 0.38 if rc != 1 else 0.0
for e in (0, 1, 2):
    for v in ("default", "copy"):
        v_rest = 3.80 + (gap if v == "copy" else 0.0)
        runs.append({"variant": v, "episode": e, "error": None, "params_in_world": params if v == "copy" else "sim/dynamics/e1_mini.json",
                     "params_log": params if v == "copy" else default, "lib_version": 31, "profile_python": "stub",
                     "rest": {"battery_v": v_rest, "hover_thrust": 0.9 - (0.1 if v == "copy" else 0.0)},
                     "ready": {"battery_v": v_rest - 0.35, "hover_thrust": 0.89 - (0.1 if v == "copy" else 0.0), "throttle": 0.6,
                               "sitl_t_s": 7.5 + (2.3 if v == "copy" else 0.0), "gz_t_s": 7.6, "wall_s": 12.0}})
verdict = pp.judge_probe(runs, params, default, 31)
info = {"path": params, "default_params": default, "profile": "stub", "sha256": "stub"}
json.dump({"schema": "soc100-probe.v1", "info": info, "runs": runs, "verdict": verdict}, open(os.path.join(out, "probe.json"), "w"), ensure_ascii=False)
open(os.path.join(out, "probe.md"), "w").write(pp.report_md(info, runs, verdict))
print(("PASS" if verdict["ok"] else "FAIL") + " 桩探针")
sys.exit(2 if rc == 2 else (0 if verdict["ok"] else 1))
'''


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def head():
    return sh(["git", "-C", SRC, "rev-parse", "HEAD"]).stdout.strip()


def shared_code():
    """冻结代码只解一次，作第二轮桩目录里 code/ 的来源（run.sh 自己仍走完整的 git archive 路径）。"""
    d = os.path.join(TMP, f"rehearse-soc100-code-{head()[:10]}")
    if not os.path.isfile(os.path.join(d, "sim", "tools", "eval_closed_loop.py")):
        os.makedirs(d, exist_ok=True)
        arc = subprocess.Popen(["git", "-C", SRC, "archive", head()], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-x", "-C", d], stdin=arc.stdout, check=True)
        arc.wait()
    return d


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_HASHES = {}


def input_lines():
    """第二轮 inputs.txt 里条款 S2 要核对的文件（真实文件的真实哈希）。"""
    files = [f"{SWA}/cal_swa_s{s}_k10.bin" for s in (0, 1, 2)] + [INFER, PLUGIN_SO]
    for f in files:
        _HASHES.setdefault(f, sha256(f))
    return [(_HASHES[f], f) for f in files]


class Ctx:
    def __init__(self, name, gpu=GPU, cpu_max="1200000 100000", extra_env=None):
        self.name = name
        self.T = tempfile.mkdtemp(prefix=f"rehearse-soc100-{name}-", dir=TMP)
        self.out, self.r2 = os.path.join(self.T, "out"), os.path.join(self.T, "r2")
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
        self.probe = os.path.join(self.T, "probe_stub.py")
        with open(self.probe, "w") as f:
            f.write(PROBE_STUB)
        with open(RUN, "rb") as src, open(os.path.join(self.out, "run.sh"), "wb") as dst:
            dst.write(src.read())
        self.env = {**os.environ, "OUT": self.out, "R2": self.r2, "SRC": SRC, "MACHINE_LOCK": os.path.join(self.T, "machine.lock"), "NVIDIA_SMI": nv,
                    "CPU_MAX_FILE": os.path.join(self.T, "cpu.max"), "CPU_STAT_FILE": os.path.join(self.T, "cpu.stat"),
                    "GZ_PROBE": f"cat {self.T}/gz_count", "EVAL": STUB, "PROBE_PY": self.probe, "SELFCHECK": "0", "LANES": "2", "IDLE_WINDOW_S": "1",
                    "IDLE_RECHECK_S": "1", "IDLE_LOG_S": "2", "LOCK_LOG_S": "2", "IDLE_LIMIT_S": "60", "TMPDIR": os.path.join(self.T, "tmp"),
                    "REAL_PY": PY, "REAL_PROBE_DIR": HERE, "PARAMS": PARAMS}
        for k in [k for k in self.env if k.startswith(("STUB_", "PROBE_STUB_"))]:
            del self.env[k]
        os.makedirs(self.env["TMPDIR"], exist_ok=True)
        self.env.update(extra_env or {})
        self.bg, self.checks = [], []

    def write(self, name, text):
        with open(os.path.join(self.T, name), "w") as f:
            f.write(text)

    def read(self, rel, default=""):
        p = os.path.join(self.out, rel)
        if os.path.isfile(p):
            with open(p, encoding="utf-8", errors="replace") as f:
                return f.read()
        return default

    # ---- 第二轮桩目录：六个对照臂由同一个桩生成（不带 --cf-sitl-params）
    def make_r2(self, done="rc=0 2026-11-12 00:30:00 桩：第二轮跑完", L="0.20", inputs=True, code_edit=None, late_seed=None):
        os.makedirs(self.r2, exist_ok=True)
        for k, s in ARMS:
            ta = "--task base --steps 80" if k == 1 else "--task course_base --blocks 3 --steps 240"
            ept = 1500 if k == 1 else 2520
            cmd = [PY, STUB, "--policy", "student", "--ckpt", f"{SWA}/cal_swa_s{s}_k10.bin", "--infer-bin", INFER, "--heading-guard", "2", *ta.split(),
                   "--split", "test", "--seeds", SEEDS, "--corridor-half-width", "3.0", "--action-latency", L, "--episodes", "20", "--sim-timeout", "120.0",
                   "--episode-timeout", str(ept), "--dynamics", "cf_sitl", "--vehicle", "e1_mini", "--hold", "1.0", "--window-from", "observation",
                   "--out", os.path.join(self.r2, f"acc-L{L}-k{k}-s{s}")]
            env = {**self.env, "STUB_SLEEP": "0"}
            subprocess.run(cmd, env=env, capture_output=True, check=False)
        with open(os.path.join(self.r2, "L_STAR"), "w") as f:
            f.write(L + "\n")
        with open(os.path.join(self.r2, "config.env"), "w") as f:
            f.write("DISP_CLAUSE=cruise\nCODE_HEAD=stub\n")
        if inputs:
            with open(os.path.join(self.r2, "inputs.txt"), "w") as f:
                f.write("".join(f"{h}  {p}\n" for h, p in input_lines()))
        code = os.path.join(self.r2, "code")
        shutil.copytree(shared_code(), code, ignore=shutil.ignore_patterns("__pycache__", ".git"), symlinks=True)
        for rel, text in (code_edit or {}).items():
            with open(os.path.join(code, rel), "a") as f:
                f.write(text)
        if done:
            with open(os.path.join(self.r2, "DONE"), "w") as f:
                f.write(done + "\n")

    def popen(self, script=None):
        return subprocess.Popen(["bash", script or os.path.join(self.out, "run.sh")], env=self.env, cwd=self.out, stdout=subprocess.DEVNULL,
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

    def done(self):
        return self.read("DONE").strip()

    def done_rc(self):
        m = re.match(r"rc=(\d+)", self.done())
        return int(m.group(1)) if m else None

    def arm_dir(self, k, s, suffix="", L="0.20"):
        return os.path.join(self.out, f"soc100-L{L}-k{k}-s{s}{suffix}")

    def started_arms(self):
        return sorted(d for d in os.listdir(self.out) if re.fullmatch(r"soc100-L[\d.]+-k\d-s\d(-retry1)?", d) and os.path.isdir(os.path.join(self.out, d)))

    def expect_done(self, rc, *needles):
        self.check(self.done_rc() == rc, f"DONE rc={rc}（实际：{self.done()[:160]!r}）")
        for n in needles:
            self.check(n in self.done(), f"DONE 含「{n}」")

    def expect_log(self, *needles):
        log = self.read("run.log")
        for n in needles:
            self.check(n in log, f"run.log 含「{n}」")

    def expect_no_arms(self):
        self.check(not self.started_arms(), f"没有起任何满电臂（实际：{self.started_arms()}）")

    def verdict(self):
        return self.read("VERDICT").split(" ", 1)[0].strip()


def r2_snapshot(ctx):
    """第二轮桩目录的（路径, 大小, 修改时间）清单：跑完前后不能变（实验不碰对照）。"""
    out = []
    for root, _d, files in os.walk(ctx.r2):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out.append((p, st.st_size, st.st_mtime_ns))
    return sorted(out)


# ---------------------------------------------------------------- 场景
def sc_pass(c):
    """主路径：核对通过 → 探针 → 六个臂 → PASS；命令行逐项核对。第二轮桩代码在允许清单内的文件上与冻结代码不同。"""
    c.make_r2(code_edit={"sim/tools/scene_gen.py": "\n# 演练：允许清单内的差异\n", "sim/cf_sitl/gz/README.md": "\n演练\n"})
    before = r2_snapshot(c)
    rc, err = c.run()
    c.check(rc == 0, f"run.sh 退出码 0（实际 {rc}，stderr {err[-200:]!r}）")
    c.expect_done(0, "PASS", "六个满电臂跑完")
    c.check(c.verdict() == "PASS", f"VERDICT = PASS（实际 {c.read('VERDICT')[:100]!r}）")
    arms = c.started_arms()
    c.check(len(arms) == 6, f"六个满电臂目录（实际 {arms}）")
    for k, s in ARMS:
        args = c.read(os.path.relpath(os.path.join(c.arm_dir(k, s), "stub_args.txt"), c.out))
        ept = 1500 if k == 1 else 2520
        want = [f"--cf-sitl-params {PARAMS}", "--action-latency 0.20", "--heading-guard 2", "--window-from observation", "--dynamics cf_sitl", "--vehicle e1_mini",
                "--hold 1.0", "--sim-timeout 120.0", f"--episode-timeout {ept}", "--episodes 20", f"--ckpt {SWA}/cal_swa_s{s}_k10.bin", f"--seeds {SEEDS}",
                "--policy student", "--split test", "--corridor-half-width 3.0", "--task base --steps 80" if k == 1 else "--task course_base --blocks 3 --steps 240"]
        miss = [w for w in want if w not in args]
        c.check(not miss and "--rtf" not in args, f"k{k}-s{s} 命令行：带副本、L、护栏、协议参数，学生臂不带 --rtf（缺 {miss}）")
    c.check(os.path.exists(os.path.join(c.out, "probe", "PASS")), "probe/PASS 存在")
    log = c.read("run.log")
    order = [log.find(x) for x in ("条件 ① 满足", "满电副本核对通过", "条件 ③ 满足", "条件 ④ 满足", "第二轮核对通过", "输入与代码核对通过", "探针开始", "探针 rc=0", "六个满电学生臂")]
    c.check(all(x >= 0 for x in order) and order == sorted(order), f"run.log 里的顺序：① → 副本 → ③ → ④ → 第二轮核对 → 输入核对 → 探针 → 臂（位置 {order}）")
    cfg = c.read("config.env")
    c.check("CONTROL=" + c.r2 in cfg and "DISP_CLAUSE=cruise" in cfg and "PARAMS_PROFILE=cf_sitl_v1:06e5f017:lib31" in cfg, "config.env 记了 CONTROL / DISP_CLAUSE / PARAMS_PROFILE")
    inputs = c.read("inputs.txt")
    c.check(sha256(PARAMS) in inputs and "libCfSitlSystem.so" in inputs, "inputs.txt 含副本与插件的 sha256")
    diff = c.read("code_diff_vs_control.txt")
    c.check("scene_gen.py" in diff and "README.md" in diff, f"code_diff_vs_control.txt 记了允许清单内的差异（{diff[:160]!r}）")
    summ = c.read("summary.md")
    c.check("判决：PASS" in summ and all(f"| P{i} |" in summ for i in range(1, 7)), "summary.md 有判决与 P1..P6 六行")
    c.check("起飞到就绪的时间（代理）" in summ and "插件读参数的 Gazebo 探针" in summ and "通过" in summ, "summary.md 有起飞代理与探针两节")
    c.check(r2_snapshot(c) == before, "第二轮桩目录里的文件跑完前后没有变化（不碰对照）")
    c.check(os.path.exists(os.path.join(c.out, "code", "experiments", "2026-11-11-soc100-parity", "summarize_parity.py")), "汇总脚本从冻结副本里跑")


def sc_falsified(c):
    """满电臂位移 −12%：F 触发，判决 FALSIFIED；DONE rc 仍是 0。"""
    c.env["STUB_FULL_DISP_SCALE"] = "0.88"
    c.make_r2()
    c.run()
    c.expect_done(0, "FALSIFIED")
    c.check(c.verdict() == "FALSIFIED", f"VERDICT = FALSIFIED（实际 {c.read('VERDICT')[:80]!r}）")
    summ = c.read("summary.md")
    c.check("证伪触发" in summ and "不能互用" in summ and "| P1 |" in summ and "**证伪**" in summ, "summary.md 写证伪与「不能互用」")
    rc = subprocess.run([PY, os.path.join(c.out, "code", "experiments", "2026-11-11-soc100-parity", "summarize_parity.py"), c.out, "--control", c.r2],
                        capture_output=True).returncode
    c.check(rc == 3, f"summarize_parity.py 退出码 3（实际 {rc}）")


def sc_fail_p1(c):
    """位移 −7%：P1 没过但没到证伪线 → FAIL。"""
    c.env["STUB_FULL_DISP_SCALE"] = "0.93"
    c.make_r2()
    c.run()
    c.expect_done(0, "FAIL")
    c.check(c.verdict() == "FAIL", f"VERDICT = FAIL（实际 {c.read('VERDICT')[:80]!r}）")
    summ = c.read("summary.md")
    c.check("P1 未过" in c.read("VERDICT") and "证伪触发" not in summ, "VERDICT 写 P1 未过，没有证伪")


def sc_boundary(c):
    """位移 −4%：过；转角 +6%：P2 没过（5% 与 10% 之间）→ FAIL，不证伪。"""
    c.env["STUB_FULL_DISP_SCALE"] = "0.96"
    c.env["STUB_FULL_YAW_SCALE"] = "1.06"
    c.make_r2()
    c.run()
    c.expect_done(0, "FAIL")
    v = c.read("VERDICT")
    c.check("P1 过" in v and "P2 未过" in v, f"P1 过、P2 未过（{v[:120]!r}）")


def sc_fail_p3(c):
    """K=1 满电臂 20 个 seed 全部取反：60 对里几乎全不一致，McNemar p 很小 → P3 未过。"""
    c.env["STUB_FULL_FLIP_SEEDS"] = SEEDS
    c.make_r2()
    c.run()
    c.expect_done(0, "FAIL")
    v = c.read("VERDICT")
    c.check("P3 未过" in v, f"P3 未过（{v[:140]!r}）")


def sc_k3(c):
    """K=3 满电档：掉高 1 集（P4 未过）+ 通过概率乘 0.5（P5、P6 未过）。"""
    c.env["STUB_FULL_DROP_SEEDS"] = "1"
    c.env["STUB_FULL_K3_SCALE"] = "0.5"
    c.make_r2()
    c.run()
    c.expect_done(0, "FAIL")
    v = c.read("VERDICT")
    c.check("P4 未过" in v and "P5 未过" in v and "P6 未过" in v and "P1 过" in v and "P3 过" in v, f"P4/P5/P6 未过，K=1 三项过（{v[:160]!r}）")


def sc_invalid(c):
    """满电臂 manifest 的指纹是对照的指纹（副本没生效）：INVALID，rc=24，不给 P1..P6。"""
    c.env["STUB_FULL_PROFILE_BAD"] = "1"
    c.make_r2()
    c.run()
    c.expect_done(24, "INVALID")
    c.check(c.verdict() == "INVALID", f"VERDICT = INVALID（实际 {c.read('VERDICT')[:80]!r}）")
    summ = c.read("summary.md")
    c.check("指纹不符" in summ and "| P1 |" not in summ, "summary.md 写指纹不符，没有 P1..P6 的判决表")


def sc_r2_nodone(c):
    """第二轮没有 DONE：拿到机器锁后停下，rc=20，不起探针也不起任何臂。"""
    c.make_r2(done=None)
    c.run()
    c.expect_done(20, "没有 DONE")
    c.expect_no_arms()
    c.check(not os.path.isdir(os.path.join(c.out, "probe")), "没有起探针")


def sc_r2_badrc(c):
    c.make_r2(done="rc=11 2026-11-12 00:30:00 桩：数据不全")
    c.run()
    c.expect_done(20, "rc 不是 0")
    c.expect_no_arms()


def sc_r2_missing_arm(c):
    c.make_r2()
    shutil.rmtree(os.path.join(c.r2, "acc-L0.20-k3-s2"))
    c.run()
    c.expect_done(20, "对照臂没有评完", "acc-k3-s2")
    c.expect_no_arms()


def sc_r2_late(c):
    """进场时第二轮还占着机器锁、DONE 还没写；它写完 DONE 再释放锁后，本实验拿到锁，在锁之后读 DONE，通过。"""
    c.make_r2(done=None)
    lock, done = os.path.join(c.T, "machine.lock"), os.path.join(c.r2, "DONE")
    c.spawn(f"import fcntl,time\nf=open({lock!r},'a')\nfcntl.flock(f,fcntl.LOCK_EX)\ntime.sleep(12)\nopen({done!r},'w').write('rc=0 2026-11-12 00:30:00 桩\\n')\ntime.sleep(0.5)\n")
    time.sleep(1)
    c.run()
    c.expect_done(0, "PASS")
    c.expect_log("被占用，阻塞等待", "第二轮核对通过")


def sc_params_bad(c):
    """副本被改过（多一个空格以外的内容：换成另一个值）：指纹或 sha256 不符，rc=22，没有等锁也没有起臂。"""
    bad = os.path.join(c.T, "bad_params.json")
    with open(PARAMS, encoding="utf-8") as f:
        text = f.read()
    with open(bad, "w", encoding="utf-8") as f:
        f.write(text + "\n")   # 只多一个换行：值指纹不变，文件 sha256 变
    c.env["PARAMS"] = bad
    c.make_r2()
    c.run()
    c.expect_done(22, "满电副本核对没过", "sha256")
    c.expect_no_arms()
    c.check("条件 ③ 满足" not in c.read("run.log"), "副本核对在等锁之前")
    # 值本身被改（电量 1.0 → 0.9）：指纹不同
    bad2 = os.path.join(c.T, "bad_params2.json")
    changed = text.replace('"battery_soc_init": 1.0', '"battery_soc_init": 0.9', 1)
    c.check(changed != text, "演练输入：确实改了 battery_soc_init")
    with open(bad2, "w", encoding="utf-8") as f:
        f.write(changed)
    c.env["PARAMS"] = bad2
    rc, _ = c.run()
    c.expect_done(22, "指纹")
    c.check(rc == 22, f"进程退出码 22（实际 {rc}）")
    c.env["PARAMS"] = os.path.join(c.T, "不存在.json")
    c.run()
    c.expect_done(22, "副本不存在")


def sc_probe_fail(c):
    c.env["PROBE_STUB_RC"] = "1"
    c.make_r2()
    c.run()
    c.expect_done(21, "探针判据没过")
    c.expect_no_arms()
    c.check(not os.path.exists(os.path.join(c.out, "probe", "PASS")), "没有 probe/PASS")
    c.check("G2" in c.done() or "未过" in c.done(), f"DONE 带探针的未过原因（{c.done()[:200]!r}）")
    c.check(os.path.isfile(os.path.join(c.out, "summary.md")), "仍然写了 summary.md（INCOMPLETE）")


def sc_probe_infra(c):
    c.env["PROBE_STUB_RC"] = "2"
    c.make_r2()
    c.run()
    c.expect_done(21, "探针基础设施失败")
    c.expect_no_arms()


def sc_inputs_bad(c):
    """第二轮 inputs.txt 里权重的哈希不是现在这份：配对不成立，rc=23。"""
    c.make_r2()
    p = os.path.join(c.r2, "inputs.txt")
    with open(p) as f:
        text = f.read()
    with open(p, "w") as f:
        f.write(text.replace(input_lines()[0][0], "0" * 64, 1))
    c.run()
    c.expect_done(23, "sha256 不一致", "cal_swa_s0_k10.bin")
    c.expect_no_arms()


def sc_codediff_bad(c):
    """第二轮冻结代码的 teacher.py 与本实验不同（不在允许清单）：rc=23。"""
    c.make_r2(code_edit={"sim/tools/teacher.py": "\n# 演练：协议相关文件的差异\n"})
    c.run()
    c.expect_done(23, "允许清单之外", "teacher.py")
    c.expect_no_arms()


def sc_late_retry(c):
    """学生臂里 seed 1 第 0 步迟到：重跑一次，K=1 s0 的结果以重跑为准，判决照出。"""
    c.env["STUB_STUDENT_LATE_SEEDS"] = "1"
    c.env["STUB_STUDENT_LATE_ARM"] = "soc100-L0.20-k1-s0"
    c.make_r2()
    c.run()
    c.expect_done(0)
    rd = c.arm_dir(1, 0, "-retry1")
    c.check(os.path.isdir(rd), "soc100-L0.20-k1-s0-retry1 存在")
    args = c.read(os.path.relpath(os.path.join(rd, "stub_args.txt"), c.out))
    c.check("--seeds 1 " in args + " " and "--episodes 1" in args and "--cf-sitl-params" in args, f"重跑只含迟到的 seed，仍带副本（{args[:200]!r}）")
    c.expect_log("含迟到步的集 [1]，重跑一次")
    summ = c.read("summary.md")
    c.check("ok" in summ and "soc100-L0.20-k1-s0" in summ, "summary.md 第 1 节记了重跑")
    c.check(c.verdict() in ("PASS", "FAIL"), f"重跑后出了判决（{c.read('VERDICT')[:60]!r}）")


def sc_late_retry_still(c):
    """重跑仍迟到：该集按没成功算，仍然出判决（不卡在 INCOMPLETE）。"""
    c.env.update(STUB_STUDENT_LATE_SEEDS="1", STUB_STUDENT_LATE_ARM="soc100-L0.20-k1-s0", STUB_RETRY_LATE="1")
    c.make_r2()
    c.run()
    c.expect_done(0)
    c.check(c.verdict() in ("PASS", "FAIL"), f"重跑仍迟到也出判决（{c.read('VERDICT')[:80]!r}）")


def sc_missing_arm(c):
    """一个臂没跑完（评测进程写 partial 后 rc=9 退出）：rc=11，不写判决，summary 标未完成。"""
    c.env["STUB_FAIL_ARM"] = "soc100-L0.20-k3-s2"
    c.make_r2()
    c.run()
    c.expect_done(11, "soc100-L0.20-k3-s2")
    c.check(c.verdict() == "INCOMPLETE", f"VERDICT = INCOMPLETE（实际 {c.read('VERDICT')[:80]!r}）")


def sc_gpu(c):
    """nvidia-smi 桩里的 GPU 不是配置的那张（CTX_OPTS 换了桩的 UUID）：rc=12。"""
    c.make_r2()
    c.run()
    c.expect_done(12, "条件 ① 不满足")
    c.expect_no_arms()


def sc_cgroup(c):
    """cgroup 配额 4 核 < 12 核（CTX_OPTS 换了 cpu.max）：rc=13。"""
    c.make_r2()
    c.run()
    c.expect_done(13, "条件 ② 不满足")
    c.expect_no_arms()


def sc_idle(c):
    """拿到锁后 gz 进程数 > 0：每秒复查；约 4 s 后清零，再开始。"""
    c.make_r2()
    c.write("gz_count", "2\n")
    c.spawn(f"import time\ntime.sleep(4)\nopen({os.path.join(c.T, 'gz_count')!r},'w').write('0\\n')\n")
    c.run()
    c.expect_done(0)
    c.expect_log("条件 ④ 不满足", "条件 ④ 满足")


def sc_running_twice(c):
    c.env["STUB_SLEEP"] = "2"
    c.make_r2()
    p1 = c.popen()
    time.sleep(3)
    p2 = c.popen()
    _o, err2 = p2.communicate(timeout=60)
    c.check(p2.returncode == 3 and "已有实例" in err2, f"第二个实例 rc=3 并提示已有实例（rc={p2.returncode}，stderr {err2[:80]!r}）")
    p1.communicate(timeout=300)
    c.expect_done(0)


def sc_resume(c):
    """跑到一半收 SIGTERM（rc=130），同一条命令重起：评完的臂跳过，半成品挪到 .partial-<时刻>，最后 rc=0。"""
    c.env["STUB_SLEEP"] = "3"
    c.make_r2()
    p = c.popen()
    deadline = time.time() + 120
    while time.time() < deadline:
        arms = c.started_arms()
        done = [a for a in arms if os.path.exists(os.path.join(c.out, a, "eval.json"))]
        if done and len(arms) > len(done):
            break
        time.sleep(0.3)
    os.killpg(p.pid, signal.SIGTERM)
    p.communicate(timeout=60)
    c.expect_done(130, "终止信号")
    first = c.started_arms()
    p2 = c.popen()
    p2.communicate(timeout=300)
    c.expect_done(0)
    log = c.read("run.log")
    parts = [d for d in os.listdir(c.out) if ".partial-" in d and os.path.isdir(os.path.join(c.out, d)) and os.path.exists(os.path.join(c.out, d, "stub_args.txt"))]
    c.check("已评完，跳过" in log, "续跑时评完的臂跳过")
    c.check(len(parts) >= 1, f"半成品的目录（带内容）挪到 .partial-<时刻>，不是删掉（{parts}）")
    c.check("探针已通过" in log, "续跑时探针不重跑")
    c.check(len(first) >= 2 and len(c.started_arms()) == 6, f"续跑后六个臂都在（被杀时 {len(first)} 个）")


def sc_selfcheck(c):
    """SELFCHECK=1：改过的 run.sh 与 CODE_HEAD 里的提交版不一致 → rc=4。脚本未提交时也是 rc=4（没有可比的版本）。"""
    c.make_r2()
    c.env["SELFCHECK"] = "1"
    with open(os.path.join(c.out, "run.sh"), "a") as f:
        f.write("\n# 演练：脚本被改过\n")
    c.run()
    c.expect_done(4, "不一致")
    c.expect_no_arms()


CTX_OPTS = {"gpu": {"gpu": "GPU-00000000-0000-0000-0000-000000000000"}, "cgroup": {"cpu_max": "400000 100000"}}   # 场景需要的桩参数

SCENARIOS = {
    "pass": sc_pass, "falsified": sc_falsified, "fail_p1": sc_fail_p1, "boundary": sc_boundary, "fail_p3": sc_fail_p3, "k3": sc_k3, "invalid": sc_invalid,
    "r2_nodone": sc_r2_nodone, "r2_badrc": sc_r2_badrc, "r2_missing_arm": sc_r2_missing_arm, "r2_late": sc_r2_late,
    "params_bad": sc_params_bad, "probe_fail": sc_probe_fail, "probe_infra": sc_probe_infra, "inputs_bad": sc_inputs_bad, "codediff_bad": sc_codediff_bad,
    "late_retry": sc_late_retry, "late_retry_still": sc_late_retry_still, "missing_arm": sc_missing_arm,
    "gpu": sc_gpu, "cgroup": sc_cgroup, "idle": sc_idle, "running_twice": sc_running_twice, "resume": sc_resume, "selfcheck": sc_selfcheck,
}


def main(argv):
    names = [n for n in argv if n != "all"] or list(SCENARIOS)
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"没有这些场景：{unknown}；可用：{list(SCENARIOS)}", file=sys.stderr)
        return 2
    rows, bad = [], 0
    for name in names:
        c = Ctx(name, **CTX_OPTS.get(name, {}))
        t0 = time.time()
        try:
            SCENARIOS[name](c)
        except Exception as e:  # noqa: BLE001 - 演练自身的错误也记成一条不过的断言
            c.check(False, f"场景抛异常 {type(e).__name__}: {e}")
        finally:
            c.cleanup()
        n_ok = sum(1 for ok, _ in c.checks if ok)
        failed = [t for ok, t in c.checks if not ok]
        bad += bool(failed)
        rows.append((name, n_ok, len(c.checks), time.time() - t0, c.T, failed))
        print(f"[{'OK ' if not failed else 'FAIL'}] {name}: {n_ok}/{len(c.checks)} 条断言（{time.time() - t0:.0f}s）{c.T}", flush=True)
        for t in failed:
            print(f"      未过：{t}", flush=True)
    print("\n| 场景 | 断言 | 结果 | 用时 s | 临时目录 |\n|---|---|---|---|---|")
    for name, n_ok, n, dt, T, failed in rows:
        print(f"| {name} | {n_ok}/{n} | {'过' if not failed else '未过'} | {dt:.0f} | {T} |")
    print(f"\n共 {len(rows)} 个场景，{len(rows) - bad} 过，{bad} 未过；断言合计 {sum(r[2] for r in rows)} 条")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
