#!/usr/bin/env python3
"""流水线（tools/e1_pipeline.py / tools/dagger_pipeline.py / tools/dagger_gz_pipeline.py / tools/compare_datasets.py）共用的小件。

续跑判定（AUDIT_SIMTOOLS 2026-11-09 Must 1/2）：
  train_finished(run_dir)  训练完成 = run/metrics.jsonl 末行是 status "done: …"。trainer 每个 epoch 都写 ckpt.bin，
                           中途被杀的 run 也有 ckpt 文件，所以「ckpt 存在」不是完成。
  eval_finished(eval_dir)  评测完成 = eval.json 存在、failed_count == 0、episodes_done > 0、不是 partial
                           （森林那轮 run_xeval.sh 的 done_ok() 就是这个逻辑）。
  retire_unfinished_eval(eval_dir)  没完成的评测目录改名 <dir>.failed-<时间>，让出路径给重跑
                           （eval_closed_loop 要求 --out 为空）。
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
import json
import math
import os
import signal
import subprocess
import time

DRY = False
DATA_ROOT = os.environ.get("DATA_ROOT", "/root/rivermind-data")


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


def eval_finished(eval_dir):
    """eval.json 在、没有失败集、至少完成 1 集、不是中断写出的 partial。旧 eval.json 没有 *_count 时按列表长度算。"""
    path = os.path.join(eval_dir, "eval.json")
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            rep = json.load(f)
    except (ValueError, OSError):
        return False
    if not isinstance(rep, dict) or rep.get("partial"):
        return False
    failed = rep.get("failed_count", len(rep.get("failed") or []))
    done = rep.get("episodes_done", len(rep.get("per_episode") or []))
    return bool(failed == 0 and done > 0)


def retire_unfinished_eval(eval_dir):
    """把没完成的评测目录挪到 <eval_dir>.failed-<YYYYmmdd-HHMMSS>（保留现场），返回新路径；目录不存在返回 None。"""
    if not os.path.isdir(eval_dir):
        return None
    dest = f"{eval_dir}.failed-{time.strftime('%Y%m%d-%H%M%S')}"
    n = 1
    while os.path.exists(dest):
        n += 1
        dest = f"{eval_dir}.failed-{time.strftime('%Y%m%d-%H%M%S')}-{n}"
    os.rename(eval_dir, dest)
    return dest


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
