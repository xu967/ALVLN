#!/usr/bin/env python3
"""流水线（tools/e1_pipeline.py / tools/dagger_pipeline.py / tools/dagger_gz_pipeline.py / tools/compare_datasets.py）共用的小件。

续跑判定（AUDIT_SIMTOOLS 2026-11-09 Must 1/2，审核 E-8 / E-9）：
  train_finished(run_dir)  训练完成 = run/metrics.jsonl 末行是 status "done: …"。trainer 每个 epoch 都写 ckpt.bin，
                           中途被杀的 run 也有 ckpt 文件，所以「ckpt 存在」不是完成。
  eval_state(eval_dir, metric_version=None, ckpt=None) / eval_finished(...)
                           「评完」的唯一判据：eval.json 存在、非 partial、failed_count==0、skipped_count==0、
                           episodes_done>0 且等于 episodes_requested；给了 metric_version / ckpt 时还要求与已有记录一致，
                           不一致判 stale。
  settle_eval(...) / retire_dir(...) / retire_unfinished_eval(...)
                           未完成评测挪 <dir>.failed-<时刻>，口径/权重不一致挪 <dir>.stale-<时刻>，半成品挪 <dir>.partial-<时刻>，
                           记进 <out>/retired_dirs.jsonl，收尾由 read_retired / retired_lines 列进汇总。
实例锁与进程记录（审核 E-3 / E-4）：
  lock_out(out) / unlock_out()  同一个输出目录只许一个流水线实例：起跑先 flock <out>/.pipeline.lock，拿不到
                           打印持锁者并 SystemExit(3)（写任何状态文件之前调）；锁随进程消失，被杀后原命令可直接再起。
  write_run_pid(out)       <out>/run.pid 记 pid / pgid / start（进程启动时刻，/proc/<pid>/stat 第 22 列）/ cmd，四行，
                           与 tools/exp_lib.sh 的 run.pid 逐键一致；停进程前核对 start，PID 被别的进程拿走就对不上。
                           parse_stat_field / proc_stat_field / proc_cmdline 是它用的取数函数。
信号（AUDIT Should 4）：
  install_signal_handlers()  SIGTERM/SIGINT 变成 Interrupted 异常（rc = 128 + signum），流水线 main 据此把
                           status.json 当前步标 interrupted、DONE 写真实 rc（以前 Ctrl-C 直接穿过 except Exception，DONE rc=0）。
  popen()/run()            子进程都起在自己的进程组并登记；run() 被打断时先 SIGTERM 子进程组（eval_closed_loop
                           收到后写部分 eval.json、收掉 gz），宽限期后 SIGKILL；terminate_children() 收全部登记过的。
共享 helper（A4，以前三份各抄一遍）：log / run_cmd / sh / read_json / write_json / mark / need / wilson_ci。
  DRY 是本模块的全局开关：流水线 main 里 `pc.DRY = DRY = a.dry_run` 一行同步；dry-run 下 run_cmd 只打印、
  mark 不落盘、need 只提醒。
gsplat 缓存目录：torch_extensions_dir（dagger_pipeline / demo_3dgs_routes / scene_import_mipnerf 共用；3DGS venv 也
  import 本模块，所以只能用标准库）。
单测：tools/test_pipeline_common.py。
"""
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
import time

DRY = False
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")
LOCK_NAME = ".pipeline.lock"   # 不叫 .lock：shell 入口也用 $OUT/.lock，在同一个 OUT 里起 Python 流水线时两把锁会互相顶
RETIRED_LOG = "retired_dirs.jsonl"
SIM_METRIC_VERSION = "sim09.v2"       # 与 sim/tools/eval_closed_loop.METRIC_VERSION 同步（test_pipeline_common 对账）
COAST_WINDOW_S = {"cf_sitl": 1.0}     # 与 sim/tools/eval_closed_loop.COAST_WINDOW_S 同步


def log(msg):
    print(time.strftime("%F %T"), msg, flush=True)


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def mark(out, step, state, **extra):
    """每步落一次 <out>/status.json：新会话一看就知道跑到哪。dry-run 不写。"""
    if DRY:
        return
    path = os.path.join(out, "status.json")
    status = read_json(path) if os.path.isfile(path) else {}
    status[step] = dict(state=state, at=time.strftime("%F %T"), **extra)
    write_json(path, status)


def need(path, hint):
    """真跑缺就停；dry-run 只提醒，让主会话在起守护进程前知道还缺什么。"""
    if os.path.exists(path):
        return
    if not DRY:
        raise RuntimeError(f"缺 {path}（{hint}）")
    log(f"[dry] 缺 {path}（{hint}）")


_LOCK_FILE = None   # 持锁的文件对象：关掉锁就放，所以进程活着期间要一直引用它
_LOCK_OUT = None    # 当前持锁的输出根目录（retire_dir 据此追加 retired_dirs.jsonl）


def lock_out(out):
    """E-3：对输出目录加实例锁（<out>/.pipeline.lock，flock）。已有实例占着就往 stderr 说明持锁者并 SystemExit(3)，
    退出码与 shell 入口的「已有实例在跑」一致。必须在写任何状态文件（config.json / DONE / status.json / run.pid）之前调：
    第二个实例不能碰第一个实例的状态。锁随进程消失（SIGKILL、容器重启后都不残留）；子进程不继承（Popen 默认 close_fds）。
    shortcut: 只防两个流水线进程同时跑；流水线被 SIGKILL 后留在自己会话里的子进程不在锁的范围内。"""
    global _LOCK_FILE, _LOCK_OUT
    out = os.path.abspath(out)
    path = os.path.join(out, LOCK_NAME)
    f = open(path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        with open(path, encoding="utf-8") as g:
            holder = g.read().strip() or "（持锁者没写记录）"
        f.close()
        print(time.strftime("%F %T"), f"已有实例在跑（{path} 被占：{holder}），本次退出；没有改动 {out} 里的任何文件",
              file=sys.stderr, flush=True)
        raise SystemExit(3)
    f.seek(0)
    f.truncate()
    f.write(f"pid={os.getpid()} pgid={os.getpgid(0)} cmd={' '.join(sys.argv)}\n")
    f.flush()
    unlock_out()    # 一个进程一把锁；换目录再拿时放掉旧的
    _LOCK_FILE = f
    _LOCK_OUT = out


def unlock_out():
    """放锁（关文件）。流水线 main 收尾时调；同一进程里连着跑两次 main（测试）才用得上，正常进程退出时内核自己放。没持锁时什么都不做。"""
    global _LOCK_FILE, _LOCK_OUT
    if _LOCK_FILE is not None:
        _LOCK_FILE.close()
        _LOCK_FILE = None
    _LOCK_OUT = None


def parse_stat_field(text, n):
    """/proc/<pid>/stat 一行文本的第 n 列（n 从 3 数起，返回字符串）；格式不对或列号超出返回 None。
    第 2 列 comm 带括号、里面可以有空格和括号，所以从最后一个「) 」往后数：第 3 列是后面第 0 个。"""
    if ") " not in text:
        return None
    tail = text.rsplit(") ", 1)[1].split()
    return tail[n - 3] if 0 <= n - 3 < len(tail) else None


def proc_stat_field(pid, n):
    """进程 pid 的 /proc/<pid>/stat 第 n 列；进程不存在或读不了返回 None。5 = PGID；22 = 启动时刻：自宿主机开机起的时钟节拍，
    容器重启不归零，同一个 PID 被别的进程拿走后这一列一定不同（E-4：停进程前靠它认人）。"""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            return parse_stat_field(f.read(), n)
    except OSError:
        return None


def proc_cmdline(pid):
    """进程 pid 的命令行，参数间一个空格（与 tools/exp_lib.sh 的 exp_cmdline 同一规则：NUL 换空格、去掉末尾空格）；读不了返回 None。"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            raw = f.read()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").rstrip(" ")


def write_run_pid(out):
    """E-4：<out>/run.pid 记本进程的 PID、PGID、启动时刻与命令行，四行 pid= / pgid= / start= / cmd=，先写临时文件再改名（读的人看不到半截）。
    格式与 tools/exp_lib.sh 的 run.pid 逐键一致，tools/stop_daemon.sh 一个解析器读两种入口。停进程前要核对「这个 PID 现在的启动时刻还是 start」：
    容器重启后 PID 会被别的进程拿走，旧写法 echo $! > run.pid 只有 PID，核对不了，kill -- -<PID> 可能停错。
    要在 lock_out 之后调：第二个实例拿不到锁就退出了，不会把第一个实例的记录换掉。
    shortcut: 按 PGID 停的前提是本进程是进程组长（setsid nohup 起）；不是的话只在日志里提醒，不拒绝起跑。"""
    pid = os.getpid()
    pgid = proc_stat_field(pid, 5)
    path = os.path.join(out, "run.pid")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        f.write(f"pid={pid}\npgid={pgid or ''}\nstart={proc_stat_field(pid, 22) or ''}\ncmd={proc_cmdline(pid) or ''}\n")
    os.replace(path + ".tmp", path)
    if pgid != str(pid):
        log(f"注意：本进程不是进程组长（pid={pid} pgid={pgid}），按 PGID 停会带走同组的其他进程；守护进程请用 setsid nohup 起")


def wilson_ci(successes, n, z=1.96):
    """Wilson 95% 区间 (lo, hi)；n=0 返回 None。与 sim/tools/eval_closed_loop.wilson_ci 同公式（那边不 import tools/）。"""
    if not n:
        return None
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


class Interrupted(Exception):
    """SIGTERM / SIGINT 到达。是 Exception 而不是 BaseException：流水线 main 先接它再接别的。"""

    def __init__(self, signum):
        self.signum = int(signum)
        self.name = signal.Signals(self.signum).name
        self.rc = 128 + self.signum
        super().__init__(f"收到 {self.name}")


def install_signal_handlers():
    def _raise(signum, _frame):
        raise Interrupted(signum)
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, _raise)


_CHILDREN = []


def popen(cmd, **kw):
    """subprocess.Popen + 独立进程组 + 登记（terminate_children 能连 gz 等孙进程一起收）。"""
    kw.setdefault("start_new_session", True)
    proc = subprocess.Popen(cmd, **kw)
    _CHILDREN.append(proc)
    return proc


def children():
    """还活着的登记子进程。"""
    return [p for p in _CHILDREN if p.poll() is None]


def terminate_children(grace_s=20.0):
    """给每个活着的子进程组发 SIGTERM，宽限 grace_s 秒（eval_closed_loop 要写部分 eval.json、停 gz），没退的 SIGKILL。返回收掉的个数。"""
    live = children()
    for p in live:
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + grace_s
    for p in live:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait()
    return len(live)


def run(cmd, env=None, cwd=None, log_path=None, grace_s=20.0):
    """起子进程并等它结束，返回退出码。stdout/stderr 追加进 log_path（没给就继承）。
    等待中被 Interrupted/其它异常打断：先按 terminate_children 收掉子进程组，再把异常抛给调用方。"""
    log_f = open(log_path, "ab") if log_path else None
    try:
        proc = popen(cmd, env=env, cwd=cwd, stdout=log_f, stderr=subprocess.STDOUT if log_f else None)
        try:
            return proc.wait()
        except BaseException:
            terminate_children(grace_s)
            raise
    finally:
        if log_f:
            log_f.close()


def run_cmd(cmd, env_extra=None, log_path=None, cwd=None, env_drop=()):
    """流水线的一条命令：打印、dry-run 直接 0、否则 run()。env_extra 叠在 os.environ 上，env_drop 的键删掉。"""
    log("$ " + " ".join(cmd) + (f"  > {log_path}" if log_path else ""))
    if DRY:
        return 0
    env = dict(os.environ, **(env_extra or {}))
    for key in env_drop:
        env.pop(key, None)
    if log_path:
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    return run(cmd, env=env, cwd=cwd, log_path=log_path)


def sh(cmd, **kw):
    """run_cmd，非零退出就抛 → 流水线停。"""
    rc = run_cmd(cmd, **kw)
    if rc != 0:
        raise RuntimeError(f"退出码 {rc}: {' '.join(os.path.basename(c) for c in cmd[:3])}"
                           + (f"，看 {kw['log_path']}" if kw.get("log_path") else ""))


def _last_nonempty_line(path):
    with open(path, "rb") as f:
        try:
            f.seek(-4096, os.SEEK_END)      # 只读尾部；metrics.jsonl 几百 KB 也不用整个读
        except OSError:
            f.seek(0)
        lines = [ln for ln in f.read().decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def train_finished(run_dir):
    """uavrgb_train 正常跑完：<run_dir>/metrics.jsonl 末行 {"event":"status","message":"done: …"}。坏文件算未完成。"""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        return False
    try:
        row = json.loads(_last_nonempty_line(path))
    except (ValueError, OSError):
        return False
    return bool(isinstance(row, dict) and row.get("event") == "status"
                and str(row.get("message", "")).startswith("done:"))


def metric_version_for(dynamics, corridor_half_width=None):
    """与 sim/tools/eval_closed_loop.report_metric_version 同一规则（tools/ 不 import sim/，由单测逐组对账）。"""
    v = SIM_METRIC_VERSION if dynamics not in COAST_WINDOW_S else f"{SIM_METRIC_VERSION}.coast{COAST_WINDOW_S[dynamics]:g}s"
    if corridor_half_width is not None and corridor_half_width > 0:
        v += f".corridor{corridor_half_width:g}m"
    return v


def eval_state(eval_dir, metric_version=None, ckpt=None):
    """E-8：「评完」的唯一判据。返回 (state, reason)，state ∈ {"missing", "unfinished", "stale", "done"}。
    被跳过（skipped_count>0）或截短（episodes_done != episodes_requested）的评测不算评完；
    已有 eval.json 的 metric_version 或 ckpt 路径与当前配置不同判 stale（旧 eval.json 无该键时不判 stale）。"""
    if not os.path.isdir(eval_dir):
        return "missing", "目录不存在"
    if not os.listdir(eval_dir):
        return "missing", "空目录"
    path = os.path.join(eval_dir, "eval.json")
    if not os.path.isfile(path):
        return "unfinished", "没有 eval.json"
    try:
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
    except (ValueError, OSError):
        return "unfinished", "eval.json 损坏"
    if not isinstance(rep, dict):
        return "unfinished", "eval.json 顶层不是对象"
    if rep.get("partial"):
        return "unfinished", "partial=true"
    failed = rep.get("failed_count", len(rep.get("failed") or []))
    skipped = rep.get("skipped_count", len(rep.get("skipped") or []))
    done = rep.get("episodes_done", len(rep.get("per_episode") or []))
    req = rep.get("episodes_requested")
    if failed != 0:
        return "unfinished", f"failed_count={failed}"
    if skipped != 0:
        return "unfinished", f"skipped_count={skipped}"
    if done <= 0:
        return "unfinished", f"episodes_done={done}"
    if req is not None and done != req:
        return "unfinished", f"episodes_done={done} != episodes_requested={req}"
    if metric_version is not None and "metric_version" in rep and rep["metric_version"] != metric_version:
        return "stale", f"metric_version={rep['metric_version']} != {metric_version}"
    if ckpt is not None and "ckpt" in rep:
        got = rep["ckpt"]
        if not got or os.path.realpath(str(got)) != os.path.realpath(str(ckpt)):
            return "stale", f"ckpt={got} != {ckpt}"
    return "done", ""


def eval_finished(eval_dir, metric_version=None, ckpt=None):
    """评测是否已完成且与给定的 metric_version / ckpt 一致（唯一实现见 eval_state）。"""
    return eval_state(eval_dir, metric_version=metric_version, ckpt=ckpt)[0] == "done"


def retire_dir(path, kind="failed", why=""):
    """把半成品或过期的目录改名为 <path>.<kind>-<YYYYmmdd-HHMMSS>（同一秒冲突加 -2，保留现场不删），返回新路径；
    目录不存在或 DRY 下返回 None。改名失败抛 RuntimeError 停下，不在旧目录上重做。
    本进程持有输出目录锁且 path 位于该目录下时，追加一行进 <out>/retired_dirs.jsonl 供汇总列出。"""
    if DRY or not os.path.isdir(path):
        return None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = f"{path}.{kind}-{stamp}"
    dest, n = base, 1
    while os.path.exists(dest):
        n += 1
        dest = f"{base}-{n}"
    try:
        os.rename(path, dest)
    except OSError as e:
        raise RuntimeError(f"改名让出目录失败（{path} -> {dest}）：{e}；停下，不在旧目录上重做") from e
    if _LOCK_OUT is not None:
        abs_path = os.path.abspath(path)
        if os.path.commonpath([_LOCK_OUT, abs_path]) == _LOCK_OUT and abs_path != _LOCK_OUT:
            row = {"at": time.strftime("%F %T"), "kind": kind,
                   "path": os.path.relpath(abs_path, _LOCK_OUT),
                   "moved_to": os.path.relpath(os.path.abspath(dest), _LOCK_OUT),
                   "why": why}
            with open(os.path.join(_LOCK_OUT, RETIRED_LOG), "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return dest


def read_retired(out):
    """读 <out>/retired_dirs.jsonl；文件不存在返回 []，末行写到一半损坏时跳过。"""
    path = os.path.join(out, RETIRED_LOG)
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def retired_lines(out):
    """把 retired_dirs.jsonl 排成汇总说明行；没有挪走过任何目录时返回 []（默认输出逐字不变）。"""
    rows = read_retired(out)
    if not rows:
        return []
    lines = [f"续跑挪走的目录（共 {len(rows)} 处，旧目录没删）："]
    for r in rows:
        why = f"（{r['why']}）" if r.get("why") else ""
        lines.append(f"  - [{r.get('kind', '?')}] {r.get('path', '?')} -> {r.get('moved_to', '?')}{why}")
    return lines


def settle_eval(eval_dir, metric_version=None, ckpt=None):
    """E-8：判定评测目录状态并让出未完成（.failed-）/ 过期（.stale-）的目录。返回 (state, moved_path, reason)。"""
    state, reason = eval_state(eval_dir, metric_version=metric_version, ckpt=ckpt)
    if state in ("done", "missing") or DRY:
        return state, None, reason
    moved = retire_dir(eval_dir, "stale" if state == "stale" else "failed", reason)
    return state, moved, reason


def retire_unfinished_eval(eval_dir):
    """把没完成的评测目录挪到 <eval_dir>.failed-<YYYYmmdd-HHMMSS>（保留现场），返回新路径；目录不存在返回 None。"""
    _state, reason = eval_state(eval_dir)
    return retire_dir(eval_dir, "failed", reason)


def torch_extensions_dir(capability=None, python=None):
    """gsplat JIT 缓存目录 $DATA_ROOT/uav_rgb/torch-cache/extensions-sm<major><minor>：按 GPU 计算能力分目录，不同卡的 .so 不混用。
    capability 缺省时用 python（3DGS venv）问 torch.cuda.get_device_capability()；问不到（没卡 / 没 venv / 超时）返回 None，
    怎么处理由调用方定。目录名进 torch 扩展的版本哈希：复制或改名都会整目录重编（docs/HANDOFF_3DGS_20261105.md 坑 1）。"""
    if capability is None:
        try:
            r = subprocess.run([python, "-c", "import torch; print(*torch.cuda.get_device_capability())"],
                               capture_output=True, text=True, timeout=120)
            capability = tuple(int(x) for x in r.stdout.split()) if r.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired, ValueError):
            capability = None
    if not capability or len(capability) != 2:
        return None
    return os.path.join(DATA_ROOT, "uav_rgb", "torch-cache", f"extensions-sm{capability[0]}{capability[1]}")
