#!/usr/bin/env python3
"""uav_rgb 训练监控台（本地原生桌面程序，Tkinter 标准库）。

直接读 run-dir 下的 results.csv / metrics.jsonl，不经过 HTTP/浏览器。
metrics.jsonl 按追加偏移增量读取；文件截断/替换时自动重置日志。
用法：python3 monitor.py [--run-dir DIR] [--refresh 秒]
"""
import argparse
import csv
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from collections import deque

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ImportError:  # 无 Tk 的解释器仍可导入解析层做自检
    tk = ttk = filedialog = None

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APP_DIR)
DEFAULT_RUN_DIR = os.path.join(APP_DIR, "demo")

# 曲线配色：train 蓝、dev 橙、accuracy 绿。
C_TRAIN, C_DEV, C_ACC = "#2b6cb0", "#dd6b20", "#2f855a"

EVENT_LIMIT = 200
TAIL_WINDOW = 1 << 20  # 首次/重载时只读末尾 1 MiB，避免长跑日志把 UI 卡住
HEAD_LEN = 64          # 首部指纹，识别同 inode 的原地重写

# 训练控制：默认接本仓库的 DEMO trainer；真实 C++/CUDA trainer 以后实现同一
# `--config` 合同即可。全部子进程用 argv 列表启动，不经过 shell。
TRAIN_SCHEMA = "uavrgb.train.v1"
TRAIN_POLL_MS = 1000   # 训练运行期刷新间隔（空闲仍用 --refresh）
STOP_GRACE_S = 5.0     # SIGINT→SIGTERM→SIGKILL 每一级的等待
TRAIN_LOG_LINES = 2000
TAIL_TEXT_BYTES = 64 << 10


def _float(cell):
    if cell is None or str(cell).strip() == "":
        return None
    try:
        v = float(cell)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def _int(cell):
    if cell is None or str(cell).strip() == "":
        return None
    try:
        return int(cell)
    except ValueError:
        return None


def _safe_text(value):
    """Tk 不接受孤立代理字符（JSON 里的 \\ud800 等）；显示前替换掉。"""
    return str(value).encode("utf-8", "replace").decode("utf-8")


def load_metrics(run_dir):
    """results.csv -> list[dict]：epoch/train_ce/dev_ce/dev_accuracy。

    文件缺失或损坏返回已解析出的行（可能为空）；坏单元格为 None。
    """
    rows = []
    try:
        with open(os.path.join(run_dir, "results.csv"), newline="",
                  encoding="utf-8-sig", errors="replace") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "epoch": _int(row.get("epoch")),
                    "train_ce": _float(row.get("train_ce")),
                    "dev_ce": _float(row.get("dev_ce")),
                    "dev_accuracy": _float(row.get("dev_accuracy")),
                })
    except (OSError, csv.Error):
        return rows
    # 无 epoch 的坏行排到最后；其余按 epoch 升序，避免乱序追加导致曲线折返。
    rows.sort(key=lambda r: (r["epoch"] is None, r["epoch"] or 0))
    return rows


def _reset_tail(state, inode=None, head=None):
    state.clear()
    state.update(inode=inode, offset=0, partial=b"", head=head)


def tail_events(run_dir, state, limit=EVENT_LIMIT):
    """读 metrics.jsonl 自上次调用后新增的完整行；返回 (events, reset)。

    state 由调用方持有并跨刷新复用。文件被截断/替换时 reset=True，调用方
    应清空旧日志。shortcut: 用首 64 字节指纹识别同 inode 的原地重写；若新文件
    前 64 字节与旧文件相同，仍会按旧偏移读，需改成全文件 hash。
    """
    path = os.path.join(run_dir, "metrics.jsonl")
    try:
        st = os.stat(path)
    except OSError:
        _reset_tail(state)
        return [], True      # 文件没了就让调用方清空，别把旧事件当成当前

    reset = False
    if state.get("inode") != st.st_ino or st.st_size < state.get("offset", 0):
        _reset_tail(state, st.st_ino)
        reset = True

    try:
        with open(path, "rb") as f:
            head = f.read(HEAD_LEN)
            old_head = state.get("head")
            if old_head is None:
                state["head"] = head
            elif head[:len(old_head)] != old_head:
                # 同 inode 的原地重写（常写回更大文件）不触发上面的 size 检查；
                # 首部指纹变了就从头重读，避免从旧偏移切进新文件中间。
                _reset_tail(state, st.st_ino, head)
                reset = True
            else:
                # 文件不足 64 字节时追加会让指纹变长；前缀相同就是追加。
                state["head"] = head
            if reset and st.st_size > TAIL_WINDOW:
                f.seek(st.st_size - TAIL_WINDOW)
                f.readline()  # 丢弃窗口首行的半截内容
            else:
                f.seek(state.get("offset", 0))
            chunk = f.read()
            state["offset"] = f.tell()
    except OSError:
        _reset_tail(state)
        return [], reset

    lines = (state.get("partial", b"") + chunk).split(b"\n")
    state["partial"] = lines.pop()  # 末尾无换行的半行留到下次
    events = []
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(event, dict):
            events.append(event)
    return (events[-limit:] if limit > 0 else []), reset


def load_events(run_dir, limit=EVENT_LIMIT):
    """一次性读取（自检/外部调用用）；GUI 用 tail_events 增量读。"""
    events, _ = tail_events(run_dir, {}, limit)
    return events


def fmt_event(e):
    """事件对象 -> 单行可读文本。"""
    if not isinstance(e, dict):
        return _safe_text(e)
    ev = e.get("event") or "?"
    step = e.get("step")
    msg = e.get("message") or ""
    if ev == "train_step":
        text = f"[train] step={step}"
        if e.get("total_steps") is not None:
            text += f"/{e['total_steps']}"
        text += f" loss={e.get('loss')} acc={e.get('action_accuracy')} lr={e.get('lr')}"
        eta = e.get("eta_sec")
        if isinstance(eta, (int, float)) and not isinstance(eta, bool):
            text += f" eta={eta:.0f}s"
        return text
    if ev == "validation":
        return f"[val]   step={step} loss={e.get('loss')} acc={e.get('action_accuracy')}"
    if ev == "error":
        sev = e.get("severity")
        tag = f"[error/{sev}]" if sev else "[error]"
        return f"{tag} {e.get('code')} {msg}"
    if step is None:
        return f"[{ev}] {msg}"
    return f"[{ev}] step={step} {msg}".rstrip()


def _line_chart(canvas, series, title):
    """在 canvas 上画折线图。series=[dict(name,color,points=[(x,y)...])]。

    y 为 None 的点断开线段；只有单点的线段画成小圆点，避免 1 个 epoch 时不可见。
    """
    canvas.delete("all")
    w = canvas.winfo_width()
    h = canvas.winfo_height()
    if w < 60 or h < 40:
        return
    pad_l, pad_r, pad_t, pad_b = 46, 10, 24, 20
    xs = [x for s in series for x, _ in s["points"] if x is not None]
    ys = [y for s in series for _, y in s["points"] if y is not None]
    if not xs or not ys:
        canvas.create_text(w / 2, h / 2, text="(无数据)", fill="#888")
        return
    xmin, xmax = min(xs), max(xs)
    x_single = xmin == xmax
    if x_single:
        xmin, xmax = xmin - 0.5, xmax + 0.5
    ymin, ymax = min(ys), max(ys)
    if ymin == ymax:
        ymin, ymax = ymin - 0.5, ymax + 0.5

    def X(x):
        return pad_l + (x - xmin) / (xmax - xmin) * (w - pad_l - pad_r)

    def Y(y):
        return h - pad_b - (y - ymin) / (ymax - ymin) * (h - pad_t - pad_b)

    canvas.create_line(pad_l, h - pad_b, w - pad_r, h - pad_b, fill="#555")
    canvas.create_line(pad_l, pad_t, pad_l, h - pad_b, fill="#555")
    for i in range(5):
        yv = ymin + (ymax - ymin) * i / 4
        canvas.create_text(pad_l - 5, Y(yv), text=f"{yv:.2f}", anchor="e",
                           font=("TkDefaultFont", 8))
    # x 轴刻度：最多 ~6 个 epoch 标签；单点时只在真实值处画一个
    if x_single:
        canvas.create_text(X(xs[0]), h - pad_b + 8, text=str(int(round(xs[0]))),
                           anchor="n", font=("TkDefaultFont", 8))
    else:
        n_ticks = max(2, min(6, int(xmax - xmin) + 1))
        for i in range(n_ticks):
            xv = xmin + (xmax - xmin) * i / (n_ticks - 1)
            canvas.create_text(X(xv), h - pad_b + 8, text=str(int(round(xv))),
                               anchor="n", font=("TkDefaultFont", 8))
    canvas.create_text(w / 2, 4, text=title, anchor="n", font=("TkDefaultFont", 9, "bold"))
    for s in series:
        segment = []
        for x, y in s["points"]:
            if x is None or y is None:
                _draw_segment(canvas, segment, s["color"])
                segment = []
            else:
                segment.append((X(x), Y(y)))
        _draw_segment(canvas, segment, s["color"])


def _draw_segment(canvas, segment, color):
    if len(segment) == 1:
        x, y = segment[0]
        canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
    elif len(segment) >= 2:
        flat = [c for p in segment for c in p]
        canvas.create_line(*flat, fill=color, width=2)


def _fmt3(v):
    return "—" if v is None else f"{v:.3f}"


def _fmt_age(seconds):
    seconds = max(0, int(seconds))
    if seconds < 120:
        return f"{seconds}s 前"
    if seconds < 7200:
        return f"{seconds // 60}m 前"
    return f"{seconds / 3600:.1f}h 前"


def _short_path(path, limit=40):
    """工具栏只显示末两级目录，避免长绝对路径挤掉按钮；完整路径在窗口标题。"""
    if len(path) <= limit:
        return path
    tail = os.path.basename(path.rstrip(os.sep))
    parent = os.path.basename(os.path.dirname(path.rstrip(os.sep)))
    candidate = f"…/{parent}/{tail}" if parent else f"…/{tail}"
    return candidate if len(candidate) <= limit else "…" + path[-(limit - 1):]


def _fmt_elapsed(seconds):
    s = max(0, int(seconds))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def default_train_command():
    """默认训练命令：本仓库的 DEMO trainer（模拟，不是真实网络）。"""
    return shlex.join([sys.executable, os.path.join(APP_DIR, "trainer_demo.py")])


def build_train_argv(command, config_path):
    """训练命令字符串 -> argv；不经过 shell。"""
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        raise ValueError(f"训练命令无法解析: {exc}") from exc
    if not argv:
        raise ValueError("训练命令不能为空")
    return argv + ["--config", config_path]


def _parse_int_field(name, text, lo, hi):
    try:
        value = int(str(text).strip())
    except (TypeError, ValueError):
        raise ValueError(f"{name} 必须是整数") from None
    if not lo <= value <= hi:
        raise ValueError(f"{name} 必须在 {lo}–{hi} 之间")
    return value


def validate_train_config(run_dir, epochs, steps_per_epoch, batch_size, lr, seed):
    """解析/校验超参数；返回写进 train_config.json 的 dict（含 schema）。"""
    raw = str(run_dir).strip()
    if not raw:
        raise ValueError("运行目录不能为空")
    run_dir = os.path.abspath(os.path.expanduser(raw))
    try:
        if os.path.realpath(run_dir) == os.path.realpath(DEFAULT_RUN_DIR):
            raise ValueError("不能把 ui/demo 示例目录作为训练输出")
    except (OSError, UnicodeError):
        raise ValueError("运行目录无法解析") from None
    epochs = _parse_int_field("epochs", epochs, 1, 100000)
    steps = _parse_int_field("steps_per_epoch", steps_per_epoch, 1, 1000000)
    batch = _parse_int_field("batch_size", batch_size, 1, 65536)
    try:
        lr_f = float(str(lr).strip())
    except (TypeError, ValueError):
        raise ValueError("lr 必须是数字") from None
    if not math.isfinite(lr_f) or not 0.0 < lr_f <= 1.0:
        raise ValueError("lr 必须是 0<lr<=1 的有限数")
    seed_i = _parse_int_field("seed", seed, 0, 2**32 - 1)
    return {"schema": TRAIN_SCHEMA, "run_dir": run_dir, "epochs": epochs,
            "steps_per_epoch": steps, "batch_size": batch, "lr": lr_f,
            "seed": seed_i}


def _write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def write_train_config(run_dir, config, argv):
    """写 train_config.json + train_command.txt；都原子替换，spawn 前完成。"""
    _write_atomic(os.path.join(run_dir, "train_config.json"),
                  json.dumps(config, indent=2, ensure_ascii=False) + "\n")
    _write_atomic(os.path.join(run_dir, "train_command.txt"),
                  shlex.join(argv) + "\n")


def spawn_trainer(argv, log_path, cwd=REPO_ROOT):
    """启动 trainer；stdout/stderr 都进 train.log，独立进程组便于整组停止。"""
    with open(log_path, "ab", buffering=0) as f:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=f,
                                stderr=subprocess.STDOUT, cwd=cwd,
                                start_new_session=True)


def signal_trainer(proc, sig):
    """发给整个进程组；进程已退出/无权限时退回单进程，最后静默。"""
    try:
        os.killpg(os.getpgid(proc.pid), sig)
        return
    except (ProcessLookupError, PermissionError, OSError):
        pass
    try:
        proc.send_signal(sig)
    except (ProcessLookupError, OSError):
        pass


def tail_text(path, state, max_bytes=TAIL_TEXT_BYTES):
    """增量读文本文件；返回 (text, reset)。日志轮转/截断时 reset=True。

    shortcut: 每次只读 max_bytes（64 KiB），日志产出快过它时显示会落后；
    接真实 trainer 后若看到追不上末尾，再改成循环读到 EOF。
    """
    try:
        st = os.stat(path)
    except OSError:
        state.clear()
        return "", False
    reset = False
    if state.get("inode") != st.st_ino or st.st_size < state.get("offset", 0):
        state.clear()
        state.update(inode=st.st_ino, offset=0)
        reset = True
    try:
        with open(path, "rb") as f:
            f.seek(state.get("offset", 0))
            chunk = f.read(max_bytes)
            state["offset"] = f.tell()
    except OSError:
        state.clear()
        return "", reset
    return chunk.decode("utf-8", "replace"), reset


def fmt_exit(rc, user_stop):
    if user_stop:
        return f"已停止 rc={rc}"
    if rc == 0:
        return "训练完成 rc=0"
    return f"异常退出 rc={rc}"


class MonitorApp:
    def __init__(self, root, run_dir, refresh_ms):
        if tk is None:
            raise RuntimeError("monitor.py 需要 tkinter")
        self.root = root
        self.run_dir = run_dir
        self.refresh_ms = max(200, int(refresh_ms))
        self._tail = {}
        self._events = deque(maxlen=EVENT_LIMIT)
        self._rows = []
        self._sig = None
        self._data_mtime = None
        self._paused = False
        self._resize_job = None
        # 训练控制状态
        self.proc = None
        self._train_state = "idle"
        self._train_pid = None
        self._train_started = None
        self._train_deadline = None
        self._train_stop_stage = 0
        self._user_stop = False
        self._last_rc = None
        self._train_log_state = {}
        self._train_log_lines = 0

        self.root.title("uav_rgb 训练监控台")
        self.root.geometry("980x660")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        self._build_toolbar()
        self._build_train_controls()
        self._build_body()
        self._build_log_notebook()
        self._build_statusbar()
        self._update_title()
        self._update_demo_tag()
        self._apply_train_state("idle")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # 首次刷新放到事件循环里；此时窗口尚未 map，曲线由 <Configure> 补画。
        self.root.after(0, self._tick)

    def _build_toolbar(self):
        bar = ttk.Frame(self.root, padding=(8, 6))
        bar.grid(row=0, column=0, sticky="ew")
        # 先排右侧按钮：窗口变窄时按钮优先保留，左侧文字让位。
        ttk.Button(bar, text="打开目录", command=self._open_dir).pack(side="right")
        ttk.Button(bar, text="选择目录", command=self._choose_dir).pack(
            side="right", padx=(0, 6))
        self.pause_btn = ttk.Button(bar, text="暂停", command=self._toggle_pause)
        self.pause_btn.pack(side="right", padx=(0, 6))
        ttk.Button(bar, text="立即刷新",
                   command=lambda: self.refresh(force=True)).pack(side="right", padx=(0, 6))

        ttk.Label(bar, text="运行目录:").pack(side="left")
        self.run_label = ttk.Label(bar, text=_safe_text(_short_path(self.run_dir)),
                                   foreground="#555")
        self.run_label.pack(side="left", padx=(2, 8))
        self.demo_tag = ttk.Label(bar, text="", foreground="#b45309",
                                  font=("TkDefaultFont", 9, "bold"))
        self.demo_tag.pack(side="left", padx=(0, 8))

    def _build_train_controls(self):
        lf = ttk.LabelFrame(self.root, text="训练控制（默认 DEMO 模拟，非真实训练）",
                            padding=(8, 4))
        lf.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        lf.columnconfigure(1, weight=1)
        lf.columnconfigure(4, weight=1)

        ttk.Label(lf, text="运行目录").grid(row=0, column=0, sticky="w")
        self.train_dir_var = tk.StringVar(value=self._default_run_dir())
        self.train_dir_entry = ttk.Entry(lf, textvariable=self.train_dir_var)
        self.train_dir_entry.grid(row=0, column=1, sticky="ew", padx=(2, 4))
        self.choose_btn = ttk.Button(lf, text="浏览…", command=self._choose_train_dir)
        self.choose_btn.grid(row=0, column=2, padx=(0, 8))
        ttk.Label(lf, text="训练命令").grid(row=0, column=3, sticky="w")
        self.train_cmd_var = tk.StringVar(value=default_train_command())
        self.train_cmd_entry = ttk.Entry(lf, textvariable=self.train_cmd_var)
        self.train_cmd_entry.grid(row=0, column=4, sticky="ew", padx=(2, 4))
        self.start_btn = ttk.Button(lf, text="开始训练", command=self._start_training)
        self.start_btn.grid(row=0, column=5, padx=(0, 4))
        self.stop_btn = ttk.Button(lf, text="停止训练", command=self._stop_training)
        self.stop_btn.grid(row=0, column=6)

        row = ttk.Frame(lf)
        row.grid(row=1, column=0, columnspan=7, sticky="ew", pady=(4, 0))
        self.train_vars = {}
        self.train_inputs = [self.train_dir_entry, self.train_cmd_entry]
        fields = (("epochs", "epochs", "10", 6),
                  ("steps_per_epoch", "steps/epoch", "100", 6),
                  ("batch_size", "batch", "32", 6),
                  ("lr", "lr", "0.001", 9),
                  ("seed", "seed", "0", 6))
        for name, label, default, width in fields:
            ttk.Label(row, text=label).pack(side="left")
            var = tk.StringVar(value=default)
            self.train_vars[name] = var
            entry = ttk.Entry(row, textvariable=var, width=width)
            entry.pack(side="left", padx=(2, 8))
            self.train_inputs.append(entry)
        self.train_status = ttk.Label(row, text="", foreground="#444")
        self.train_status.pack(side="right")
        self.train_demo_tag = ttk.Label(row, text="", foreground="#b45309",
                                        font=("TkDefaultFont", 9, "bold"))
        self.train_demo_tag.pack(side="right", padx=(0, 8))
        self.train_cmd_var.trace_add("write", self._update_train_demo_tag)
        self._update_train_demo_tag()

    def _build_log_notebook(self):
        nb = ttk.Notebook(self.root)
        nb.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 4))
        events = ttk.Frame(nb, padding=(4, 2))
        events.columnconfigure(0, weight=1)
        self.log = tk.Listbox(events, height=6, font=("TkFixedFont", 9))
        lgsb = ttk.Scrollbar(events, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=lgsb.set)
        self.log.grid(row=0, column=0, sticky="ew")
        lgsb.grid(row=0, column=1, sticky="ns")
        nb.add(events, text="指标事件")

        out = ttk.Frame(nb, padding=(4, 2))
        out.columnconfigure(0, weight=1)
        self.train_log = tk.Text(out, height=6, wrap="none", state="disabled",
                                 font=("TkFixedFont", 9))
        tsb = ttk.Scrollbar(out, orient="vertical", command=self.train_log.yview)
        self.train_log.configure(yscrollcommand=tsb.set)
        self.train_log.grid(row=0, column=0, sticky="ew")
        tsb.grid(row=0, column=1, sticky="ns")
        nb.add(out, text="训练输出")
        self.notebook = nb
        self.tab_events = events
        self.tab_output = out

    def _build_body(self):
        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))

        # 汇总区
        self.summary = tk.Text(left, width=30, height=6, wrap="none", state="disabled",
                               relief="flat", background=self.root.cget("background"))
        self.summary.grid(row=0, column=0, sticky="ew")

        # 指标表
        cols = ("epoch", "train_ce", "dev_ce", "dev_accuracy")
        heads = ("epoch", "train_ce", "dev_ce", "dev_acc")
        table = ttk.Treeview(left, columns=cols, show="headings", height=10)
        for c, h in zip(cols, heads):
            table.heading(c, text=h)
            table.column(c, width=72, anchor="e")
        self.table = table
        sb = ttk.Scrollbar(left, orient="vertical", command=table.yview)
        table.configure(yscrollcommand=sb.set)
        table.grid(row=1, column=0, sticky="nsew")
        sb.grid(row=1, column=1, sticky="ns")
        left.rowconfigure(1, weight=1)

        # 曲线区（两张图：loss、accuracy）
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        self.canvas_loss = tk.Canvas(right, background="#ffffff", highlightthickness=1,
                                     highlightbackground="#ddd")
        self.canvas_loss.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self.canvas_acc = tk.Canvas(right, background="#ffffff", highlightthickness=1,
                                    highlightbackground="#ddd")
        self.canvas_acc.grid(row=1, column=0, sticky="nsew")
        self.canvas_loss.bind("<Configure>", self._on_resize)
        self.canvas_acc.bind("<Configure>", self._on_resize)

    def _build_statusbar(self):
        self.status = ttk.Label(self.root, text="", padding=(8, 3), foreground="#444")
        self.status.grid(row=4, column=0, sticky="ew")

    def _default_run_dir(self):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return os.path.join(REPO_ROOT, "logs", "runs", stamp + "-demo")

    def _choose_train_dir(self):
        d = filedialog.askdirectory(
            initialdir=os.path.dirname(self.train_dir_var.get()),
            title="选择训练输出目录（必须是空目录或新目录）")
        if d:
            self.train_dir_var.set(os.path.abspath(d))

    def _update_train_demo_tag(self, *_args):
        try:
            argv = shlex.split(self.train_cmd_var.get())
        except ValueError:
            argv = []
        demo = any(os.path.basename(a) == "trainer_demo.py" for a in argv)
        self.train_demo_tag.config(text="DEMO 模拟训练（非真实网络）" if demo else "")

    def _is_demo(self):
        try:
            return os.path.realpath(self.run_dir) == os.path.realpath(DEFAULT_RUN_DIR)
        except (OSError, UnicodeError):
            # 文件名含无法编码的字节时 realpath 会失败；这只是 DEMO 标识判断。
            return False

    def _update_title(self):
        self.root.title(_safe_text(f"uav_rgb 训练监控台 · {self.run_dir}"))

    def _update_demo_tag(self):
        self.demo_tag.config(text="DEMO 示例数据（非真实训练）" if self._is_demo() else "")

    def refresh(self, force=False):
        """读文件并刷新界面；文件签名未变时只更新状态，避免无谓重绘。"""
        paths = [os.path.join(self.run_dir, n) for n in ("results.csv", "metrics.jsonl")]
        stats = []
        for p in paths:
            try:
                stats.append(os.stat(p))
            except OSError:
                stats.append(None)
        sig = tuple((s.st_mtime_ns, s.st_size, s.st_ino) if s else None for s in stats)
        if not force and sig == self._sig:
            self._update_status()
            return

        rows = load_metrics(self.run_dir)
        new_events, reset = tail_events(self.run_dir, self._tail)
        if reset:
            self._events.clear()
        self._events.extend(new_events)
        self._rows = rows
        mtimes = [s.st_mtime for s in stats if s is not None]
        self._data_mtime = max(mtimes) if mtimes else None

        self._set_summary(rows)
        self._set_table(rows)
        self._set_charts(rows)
        self._set_log(self._events)
        self._update_status()
        # 全部绘制成功后才记签名：若某个 setter 抛异常，下一 tick 会重试而不是被跳过。
        self._sig = sig

    def _update_status(self):
        valid = sum(1 for r in self._rows if r["epoch"] is not None)
        parts = [f"{valid} epochs", f"{len(self._events)} 事件"]
        if self._data_mtime is not None:
            parts.append(f"数据 {_fmt_age(time.time() - self._data_mtime)}")
        missing = [n for n in ("results.csv", "metrics.jsonl")
                   if not os.path.isfile(os.path.join(self.run_dir, n))]
        if missing:
            parts.append("等待 " + "/".join(missing))
        if self._paused:
            parts.append("已暂停")
        self.status.config(text=" · ".join(parts))

    def _set_summary(self, rows):
        valid = [r for r in rows if r["epoch"] is not None]
        dev_ce = [(r["epoch"], r["dev_ce"]) for r in valid if r["dev_ce"] is not None]
        acc = [(r["epoch"], r["dev_accuracy"]) for r in valid
               if r["dev_accuracy"] is not None]
        best_ce = min(dev_ce, key=lambda t: t[1]) if dev_ce else None
        best_acc = max(acc, key=lambda t: t[1]) if acc else None
        last = valid[-1] if valid else None
        lines = [
            f"epochs: {len(valid)}",
            f"当前 train_ce: {_fmt3(last['train_ce']) if last else '—'}",
            f"当前 dev_ce:   {_fmt3(last['dev_ce']) if last else '—'}",
            f"当前 dev_acc:  {_fmt3(last['dev_accuracy']) if last else '—'}",
            f"最佳 dev_ce:   {_fmt3(best_ce[1]) if best_ce else '—'}  (epoch {best_ce[0] if best_ce else '-'})",
            f"最佳 dev_acc:  {_fmt3(best_acc[1]) if best_acc else '—'}  (epoch {best_acc[0] if best_acc else '-'})",
        ]
        self.summary.config(state="normal")
        self.summary.delete("1.0", "end")
        self.summary.insert("1.0", "\n".join(lines))
        self.summary.config(state="disabled")

    def _set_table(self, rows):
        selected = self.table.selection()
        keep = None
        if selected:
            vals = self.table.item(selected[0], "values")
            keep = vals[0] if vals else None
        self.table.delete(*self.table.get_children())
        for r in rows:
            self.table.insert("", "end", values=(
                r["epoch"] if r["epoch"] is not None else "—",
                _fmt3(r["train_ce"]), _fmt3(r["dev_ce"]), _fmt3(r["dev_accuracy"])))
        if keep is not None:
            for iid in self.table.get_children():
                if self.table.item(iid, "values")[0] == keep:
                    self.table.selection_set(iid)
                    break

    def _set_charts(self, rows):
        epochs = [r["epoch"] for r in rows]
        _line_chart(self.canvas_loss, [
            {"name": "train_ce", "color": C_TRAIN,
             "points": list(zip(epochs, [r["train_ce"] for r in rows]))},
            {"name": "dev_ce", "color": C_DEV,
             "points": list(zip(epochs, [r["dev_ce"] for r in rows]))},
        ], "loss（蓝=train，橙=dev）")
        _line_chart(self.canvas_acc, [
            {"name": "dev_accuracy", "color": C_ACC,
             "points": list(zip(epochs, [r["dev_accuracy"] for r in rows]))},
        ], "dev_accuracy")

    def _set_log(self, events):
        pos = self.log.yview()
        follow = not pos or pos[1] >= 0.999
        self.log.delete(0, "end")
        for e in events:
            line = _safe_text(fmt_event(e))
            self.log.insert("end", line)
            if line.startswith("[error"):
                self.log.itemconfig("end", foreground="#c53030")
        self.log.yview_moveto(1.0 if follow else pos[0])

    def _on_resize(self, _event=None):
        # 窗口 map/缩放后 Canvas 才有真实尺寸；去抖后补画，避免启动时空白 5 秒。
        if self._resize_job is not None:
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self._redraw_charts)

    def _redraw_charts(self):
        self._resize_job = None
        self._set_charts(self._rows)

    def _choose_dir(self):
        if self.proc is not None:
            self.status.config(text="训练运行中，不能切换监控目录")
            return
        d = filedialog.askdirectory(initialdir=self.run_dir, title="选择训练 run 目录")
        if d:
            self._set_run_dir(os.path.abspath(d))

    def _set_run_dir(self, path):
        # 先复位状态再碰 Tk：路径显示失败也不会留下“新目录+旧事件”的半切换状态。
        self.run_dir = path
        self._tail.clear()
        self._events.clear()
        self._rows = []
        self._sig = None
        self._data_mtime = None
        self._update_title()
        self._update_demo_tag()
        self.run_label.config(text=_safe_text(_short_path(path)))
        if hasattr(self, "train_dir_var"):
            self.train_dir_var.set(path)
        if hasattr(self, "train_log"):
            self._train_log_state.clear()
            self._train_log_lines = 0
            self.train_log.config(state="normal")
            self.train_log.delete("1.0", "end")
            self.train_log.config(state="disabled")
        self.refresh(force=True)

    def _apply_train_state(self, state):
        self._train_state = state
        editable = state in ("idle", "exited")
        for w in self.train_inputs:
            w.config(state="normal" if editable else "disabled")
        self.choose_btn.config(state="normal" if editable else "disabled")
        self.start_btn.config(state="normal" if editable else "disabled")
        self.stop_btn.config(state="normal" if state == "running" else "disabled")
        if state == "idle":
            self.train_status.config(text="空闲", foreground="#444")
        elif state == "starting":
            self.train_status.config(text="启动中…", foreground="#444")
        elif state == "running":
            if self._train_pid is not None:
                self.train_status.config(
                    text=f"运行中 pid={self._train_pid}", foreground="#2b6cb0")
        elif state == "stopping":
            self.train_status.config(text="停止中（SIGINT→SIGTERM）",
                                     foreground="#b45309")
        elif state == "exited":
            self.train_status.config(
                text=fmt_exit(self._last_rc, self._user_stop),
                foreground="#2f855a" if self._last_rc == 0 else "#c53030")

    def _start_training(self):
        if self.proc is not None:
            return
        rollback = False   # 目录确认是空的之后写下的文件都归本次启动，失败可安全删掉
        try:
            config = validate_train_config(
                self.train_dir_var.get(),
                self.train_vars["epochs"].get(),
                self.train_vars["steps_per_epoch"].get(),
                self.train_vars["batch_size"].get(),
                self.train_vars["lr"].get(),
                self.train_vars["seed"].get())
            run_dir = config["run_dir"]
            if os.path.isdir(run_dir) and os.listdir(run_dir):
                raise ValueError(f"运行目录不是空目录: {run_dir}（请用新目录）")
            os.makedirs(run_dir, exist_ok=True)
            if not os.path.isdir(run_dir):
                raise ValueError(f"无法创建运行目录: {run_dir}")
            rollback = True
            argv = build_train_argv(self.train_cmd_var.get(),
                                    os.path.join(run_dir, "train_config.json"))
            write_train_config(run_dir, config, argv)
            log_path = os.path.join(run_dir, "train.log")
            self._apply_train_state("starting")
            proc = spawn_trainer(argv, log_path)
        except (ValueError, OSError) as exc:
            if rollback:   # 否则改好命令后同一目录也报“不是空目录”
                for name in ("train_config.json", "train_command.txt", "train.log"):
                    try:
                        os.remove(os.path.join(run_dir, name))
                    except OSError:
                        pass
            self._apply_train_state("idle")
            self.train_status.config(text=_safe_text(f"启动失败: {exc}"),
                                     foreground="#c53030")
            self._append_train_log(f"[monitor] 启动失败: {exc}")
            return

        self.proc = proc
        self._user_stop = False
        self._last_rc = None
        self._train_pid = proc.pid
        self._train_started = time.monotonic()
        self._train_deadline = None
        self._train_stop_stage = 0
        self._set_run_dir(run_dir)
        self._append_train_log(f"[monitor] 已启动 pid={proc.pid}: {shlex.join(argv)}")
        self.notebook.select(self.tab_events)
        self._apply_train_state("running")

    def _stop_training(self):
        if self.proc is None or self._train_state != "running":
            return
        self._user_stop = True
        self._train_stop_stage = 0
        self._train_deadline = time.monotonic() + STOP_GRACE_S
        self._apply_train_state("stopping")
        signal_trainer(self.proc, signal.SIGINT)

    def _poll_train(self):
        if self.proc is None:
            return
        rc = self.proc.poll()
        if rc is not None:
            self._on_train_exit(rc)
            return
        if self._train_state == "running":
            elapsed = time.monotonic() - self._train_started
            self.train_status.config(
                text=f"运行中 pid={self._train_pid} 已用 {_fmt_elapsed(elapsed)}",
                foreground="#2b6cb0")
        elif self._train_state == "stopping":
            now = time.monotonic()
            if self._train_deadline is not None and now > self._train_deadline:
                if self._train_stop_stage == 0:
                    signal_trainer(self.proc, signal.SIGTERM)
                    self._train_stop_stage = 1
                    self._train_deadline = now + STOP_GRACE_S
                    self.train_status.config(text="停止中（已发 SIGTERM）")
                elif self._train_stop_stage == 1:
                    signal_trainer(self.proc, signal.SIGKILL)
                    self._train_stop_stage = 2
                    self._train_deadline = now + STOP_GRACE_S
                    self.train_status.config(text="停止中（已发 SIGKILL）")

    def _on_train_exit(self, rc):
        self.proc = None
        self._train_pid = None
        self._last_rc = rc
        self._train_deadline = None
        self._train_stop_stage = 0
        try:
            self.refresh(force=True)
            self._refresh_train_log()
        except Exception:
            traceback.print_exc()
        if rc != 0 and not self._user_stop:
            self._append_train_log(f"[monitor] trainer 异常退出 rc={rc}")
            self.notebook.select(self.tab_output)
        else:
            self._append_train_log(f"[monitor] trainer 退出 rc={rc}")
        self._apply_train_state("exited")
        # 刚跑完的目录已被占用，下一次启动默认换新目录，省得用户手改时间戳。
        self.train_dir_var.set(self._default_run_dir())

    def _refresh_train_log(self):
        text, reset = tail_text(os.path.join(self.run_dir, "train.log"),
                                self._train_log_state)
        if reset:
            self._train_log_lines = 0
            self.train_log.config(state="normal")
            self.train_log.delete("1.0", "end")
            self.train_log.config(state="disabled")
        if text:
            self._append_train_log(text.rstrip("\n"))

    def _append_train_log(self, text):
        if not text:
            return
        pos = self.train_log.yview()
        follow = not pos or pos[1] >= 0.999
        self.train_log.config(state="normal")
        self.train_log.insert("end", _safe_text(text) + "\n")
        self._train_log_lines += text.count("\n") + 1
        if self._train_log_lines > TRAIN_LOG_LINES:
            excess = self._train_log_lines - TRAIN_LOG_LINES
            self.train_log.delete("1.0", f"{excess + 1}.0")
            self._train_log_lines = TRAIN_LOG_LINES
        self.train_log.config(state="disabled")
        if follow:
            self.train_log.see("end")

    def _tick_delay(self):
        return TRAIN_POLL_MS if self.proc is not None else self.refresh_ms

    def _on_close(self):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            self._user_stop = True
            signal_trainer(proc, signal.SIGINT)
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                signal_trainer(proc, signal.SIGTERM)
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    signal_trainer(proc, signal.SIGKILL)
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
        self.root.destroy()

    def _toggle_pause(self):
        self._paused = not self._paused
        self.pause_btn.config(text="继续" if self._paused else "暂停")
        if not self._paused:
            self.refresh(force=True)
        else:
            self._update_status()

    def _open_dir(self):
        if not os.path.isdir(self.run_dir):
            self.status.config(text=_safe_text(f"目录不存在: {self.run_dir}"))
            return
        try:
            subprocess.Popen(["xdg-open", self.run_dir], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except OSError as exc:
            self.status.config(text=_safe_text(f"无法打开目录: {exc}"))

    def _tick(self):
        try:
            # 训练进程轮询不受“暂停”影响，否则停止/升级信号会失效。
            self._poll_train()
            if not self._paused:
                self.refresh()
                self._refresh_train_log()
        except Exception as exc:  # 单次坏数据不能停掉自动刷新
            traceback.print_exc()
            self.status.config(text=f"刷新失败: {exc}")
        finally:
            self.root.after(self._tick_delay(), self._tick)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    p.add_argument("--refresh", type=float, default=5.0,
                   help="刷新间隔（秒，0.5–3600）")
    args = p.parse_args()
    if not math.isfinite(args.refresh) or not 0.5 <= args.refresh <= 3600:
        p.error("--refresh 必须是 0.5–3600 的有限秒数")
    if tk is None:
        raise SystemExit("monitor.py 需要 tkinter；请使用带 tkinter 的 Python 3 运行")
    root = tk.Tk()
    MonitorApp(root, os.path.abspath(os.path.expanduser(args.run_dir)),
               int(args.refresh * 1000))
    root.mainloop()


if __name__ == "__main__":
    main()
