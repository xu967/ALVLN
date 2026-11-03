#!/usr/bin/env python3
"""UAV Navigation Studio: local desktop training, simulation and experiment workbench.

The UI reads the real uavrgb_train artifacts (results.csv / metrics.jsonl), launches
trainer_native.py, and subscribes to Gazebo camera/pose topics. No demo metrics are
shown unless the user explicitly selects the legacy demo trainer.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:  # pragma: no cover - headless CI
    tk = ttk = filedialog = messagebox = None

try:  # Optional: resize camera frames to fit smaller desktop windows.
    from PIL import Image, ImageTk
except ImportError:  # Tk's native PNG decoder remains the fallback.
    Image = ImageTk = None

UI_DIR = Path(__file__).resolve().parent
REPO_ROOT = UI_DIR.parent
SIM_TOOLS = REPO_ROOT / "sim" / "tools"
DATA_ROOT = Path(os.path.expanduser(os.environ.get("DATA_ROOT", str(Path.home() / "rivermind-data"))))
RUNS_ROOT = DATA_ROOT / "uav_rgb" / "studio_runs"
TRAIN_NATIVE = UI_DIR / "trainer_native.py"
TRAINER_BIN = Path(os.environ.get("UAV_RGB_TRAIN_BIN") or os.environ.get("UAVRGB_TRAIN_BIN") or
                   str(Path(os.environ.get("UAV_RGB_BUILD_DIR", str(REPO_ROOT / "build" / "check")))
                       / "uavrgb_train"))
INFER_BIN = Path(os.environ.get(
    "UAV_RGB_INFER_BIN",
    str(Path(os.environ.get("UAV_RGB_BUILD_DIR", str(REPO_ROOT / "build" / "check")))
        / "uavrgb_infer")))
PYTHON_FOR_SIM = os.path.expanduser(os.environ.get("UAV_RGB_PYTHON", sys.executable))
CAMERA_TOPIC = "/uav/camera"
CHASE_TOPIC = "/uav/chase_camera"
TOP_TOPIC = "/uav/top_camera"
POSE_TOPIC = "/model/x3/pose"
ACTION_NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")

sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(SIM_TOOLS))
import monitor as monitor_io  # noqa: E402
import gz_io  # noqa: E402
import save_frame  # noqa: E402


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_float(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def write_json_atomic(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path, chunk_size=4 << 20):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path, max_lines=200000):
    # Retain the tail, not the head: train_step can produce a large JSONL and the
    # final test/status events are the most important records for a run summary.
    rows = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except (ValueError, RecursionError):
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except OSError:
        pass
    return list(rows)


def parse_pose_msg(msg):
    """Return (x,y,z,yaw_rad,sim_time_ns) from gz.msgs.Pose_V / Pose JSON."""
    item = msg
    arr = msg.get("pose") if isinstance(msg, dict) else None
    if isinstance(arr, list) and arr:
        # Current X3 publisher normally emits one Pose. Prefer a named X3 entry.
        item = next((p for p in arr if str(p.get("name", "")).lower() in
                     ("x3", "model/x3", "x3::base_link")), arr[0])
    if not isinstance(item, dict):
        return None
    pos = item.get("position") or {}
    q = item.get("orientation") or {}
    try:
        x, y, z = (float(pos[k]) for k in ("x", "y", "z"))
        qx, qy, qz, qw = (float(q.get(k, 0.0)) for k in ("x", "y", "z", "w"))
    except (KeyError, TypeError, ValueError):
        return None
    yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    header = item.get("header") or {}
    stamp = header.get("stamp") or {}
    try:
        stamp_ns = int(stamp.get("sec", 0)) * 1_000_000_000 + int(stamp.get("nsec", 0))
    except (TypeError, ValueError):
        stamp_ns = 0
    return x, y, z, yaw, stamp_ns


def pose_stamp_ns(msg):
    pose = parse_pose_msg(msg)
    return pose[4] if pose else 0


def _pose_values(text):
    values = [safe_float(x) for x in (text or "").split()]
    values += [0.0] * (6 - len(values))
    return tuple(values[:6])


def _compose_pose(a, b):
    """Compose planar part of two SDF poses (x,y,z,roll,pitch,yaw)."""
    ax, ay, az, ar, ap, ayaw = a
    bx, by, bz, br, bp, byaw = b
    c, s = math.cos(ayaw), math.sin(ayaw)
    return (ax + c * bx - s * by, ay + s * bx + c * by, az + bz,
            ar + br, ap + bp, ayaw + byaw)


def parse_sdf_obstacles(world_path):
    """Extract simple top-down collision footprints from an SDF world."""
    if not world_path:
        return []
    try:
        root = ET.parse(str(world_path)).getroot()
    except (OSError, ET.ParseError):
        return []
    world = root.find(".//world")
    if world is None:
        return []
    result = []
    for model in world.findall("model"):
        name = model.get("name", "model")
        if name in ("x3", "ground_plane") or name.startswith("goal_mark"):
            continue
        model_pose = _pose_values(model.findtext("pose"))
        for link in model.findall("link"):
            link_pose = _compose_pose(model_pose, _pose_values(link.findtext("pose")))
            for collision in link.findall("collision"):
                col_pose = _compose_pose(link_pose, _pose_values(collision.findtext("pose")))
                geometry = collision.find("geometry")
                if geometry is None:
                    continue
                box = geometry.find("box/size")
                cyl = geometry.find("cylinder")
                sphere = geometry.find("sphere")
                if box is not None and box.text:
                    vals = [safe_float(v) for v in box.text.split()]
                    if len(vals) >= 2 and vals[0] and vals[1]:
                        result.append({"name": name, "kind": "box", "x": col_pose[0],
                                       "y": col_pose[1], "sx": vals[0], "sy": vals[1],
                                       "yaw": col_pose[5]})
                elif cyl is not None:
                    r = safe_float(cyl.findtext("radius"))
                    if r and r > 0:
                        result.append({"name": name, "kind": "circle", "x": col_pose[0],
                                       "y": col_pose[1], "r": r})
                elif sphere is not None:
                    r = safe_float(sphere.findtext("radius"))
                    if r and r > 0:
                        result.append({"name": name, "kind": "circle", "x": col_pose[0],
                                       "y": col_pose[1], "r": r})
    return result


def load_scene_meta(path):
    if not path:
        return {}
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def scene_split(meta):
    if not meta:
        return "train"
    try:
        import split_manifest
        return split_manifest.split_for_scene(meta)
    except Exception:
        try:
            import campus
            return campus.split_map(meta.get("map_id"))
        except Exception:
            return str(meta.get("split") or "train")


def read_training_rows(run_dir: Path):
    rows = []
    try:
        with (run_dir / "results.csv").open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                def num(k):
                    return safe_float(row.get(k))
                try:
                    epoch = int(row.get("epoch", ""))
                except (TypeError, ValueError):
                    continue
                rows.append({"epoch": epoch, "train_ce": num("train_ce"),
                             "dev_ce": num("dev_ce"), "dev_accuracy": num("dev_accuracy")})
    except (OSError, csv.Error):
        pass
    return rows


def summarize_run(run_dir: Path, run_meta: dict, return_code, state):
    events = read_jsonl(run_dir / "metrics.jsonl")
    rows = read_training_rows(run_dir)
    validations = [e for e in events if e.get("event") == "validation"]
    val_events = [e for e in validations if e.get("split") == "val"]
    test_events = [e for e in validations if e.get("split") == "test"]
    best_val = max(val_events, key=lambda e: safe_float(e.get("macro_f1"))
                   if safe_float(e.get("macro_f1")) is not None else
                   (safe_float(e.get("action_accuracy")) or -1), default=None)
    test = test_events[-1] if test_events else None
    test_confusion = test.get("confusion") if test else None
    test_per_class_recall = None
    if isinstance(test_confusion, list) and len(test_confusion) == 16:
        matrix = [test_confusion[i:i + 4] for i in range(0, 16, 4)]
        test_per_class_recall = []
        for i, row in enumerate(matrix):
            total = sum(safe_float(v) or 0.0 for v in row)
            diagonal = safe_float(row[i])
            test_per_class_recall.append({"class": ACTION_NAMES[i],
                                          "recall": (diagonal / total if total > 0 and diagonal is not None else None)})
    best_status = next((e.get("message", "") for e in reversed(events)
                        if e.get("event") == "status" and "best checkpoint" in
                        str(e.get("message", "")).lower()), "")
    match = re.search(r"epoch\s+(\d+)", best_status)
    best_epoch = int(match.group(1)) if match else None
    ckpt = run_dir / "ckpt_best.bin"
    sidecar = Path(str(run_dir) + ".monitor")
    metrics = {
        "epochs_completed": len(rows),
        "train_steps_reported": sum(1 for e in events if e.get("event") == "train_step"),
        "best_validation_macro_f1": safe_float(best_val.get("macro_f1")) if best_val else None,
        "best_validation_balanced_accuracy": safe_float(best_val.get("balanced_accuracy")) if best_val else None,
        "best_validation_accuracy": safe_float(best_val.get("action_accuracy")) if best_val else None,
        "best_checkpoint_epoch": best_epoch,
        "test_accuracy": safe_float(test.get("action_accuracy")) if test else None,
        "test_macro_f1": safe_float(test.get("macro_f1")) if test else None,
        "test_balanced_accuracy": safe_float(test.get("balanced_accuracy")) if test else None,
        "test_confusion": test_confusion,
        "test_per_class_recall": test_per_class_recall,
        "validation_split": val_events[-1].get("split") if val_events else None,
        "test_evaluated": test is not None,
    }
    started = run_meta.get("started_at")
    ended = now_iso()
    duration = None
    try:
        if run_meta.get("started_monotonic") is not None:
            duration = round(max(0.0, time.monotonic() - float(run_meta["started_monotonic"])), 2)
    except (TypeError, ValueError):
        pass
    summary = {
        "schema": "uavrgb.studio_run.v1",
        "run_id": run_meta.get("run_id", run_dir.name),
        "status": state,
        "return_code": return_code,
        "started_at": started,
        "finished_at": ended,
        "duration_sec": duration,
        "config": run_meta.get("config", {}),
        "command": run_meta.get("command", []),
        "dataset": run_meta.get("dataset", {}),
        "metrics": metrics,
        "closed_loop_evaluations": [],
        "live_demos": [],
        "monitor_dir": str(sidecar),
        "artifacts": {
            "results_csv": (run_dir / "results.csv").is_file(),
            "metrics_jsonl": (run_dir / "metrics.jsonl").is_file(),
            "checkpoint_last": (run_dir / "ckpt.bin").is_file(),
            "checkpoint_best": ckpt.is_file(),
            "checkpoint_best_bytes": ckpt.stat().st_size if ckpt.is_file() else None,
            "config_json": (run_dir / "config.json").is_file(),
            "train_log": (sidecar / "train.log").is_file(),
            "job_metadata": (sidecar / "job.json").is_file(),
            "lifecycle": (sidecar / "lifecycle.json").is_file(),
        },
        "notes": [
            "test_accuracy / test_macro_f1 are offline action-classification metrics, not navigation success rate.",
            "A formal navigation SR requires a separate eval_closed_loop.py run and its eval.json.",
            "Missing metrics remain null; they are not interpreted as zero."
        ],
    }
    return summary


def write_experiment_report(run_dir: Path, summary: dict):
    m = summary.get("metrics", {})
    c = summary.get("config", {})
    lines = [
        f"# UAV RGB experiment: {summary.get('run_id')}", "",
        f"- Status: **{summary.get('status')}** (exit={summary.get('return_code')})",
        f"- Started: {summary.get('started_at')}",
        f"- Finished: {summary.get('finished_at')}",
        f"- Duration: {summary.get('duration_sec')} s", "",
        "## Configuration", "",
        f"- Dataset: `{c.get('data', '—')}`",
        f"- Epochs: {c.get('epochs', '—')}; batch: {c.get('batch_size', '—')}; lr: {c.get('lr', '—')}; seed: {c.get('seed', '—')}",
        f"- text_dim: {c.get('text_dim', '—')}; class_weights: {c.get('class_weights', '—')}; drop_collision: {c.get('drop_collision', '—')}", "",
        "## Training / offline evaluation", "",
        f"- Epochs completed: {m.get('epochs_completed')}",
        f"- Best validation Macro-F1: {m.get('best_validation_macro_f1')}",
        f"- Best validation Balanced Accuracy: {m.get('best_validation_balanced_accuracy')}",
        f"- Best checkpoint epoch: {m.get('best_checkpoint_epoch')}",
        f"- Offline test accuracy: {m.get('test_accuracy')}",
        f"- Offline test Macro-F1: {m.get('test_macro_f1')}",
        f"- Offline test Balanced Accuracy: {m.get('test_balanced_accuracy')}", "",
    ]
    recalls = m.get("test_per_class_recall")
    if recalls:
        lines += ["### Per-class test recall", "", "| Action | Recall |", "| --- | ---: |"]
        for item in recalls:
            value = item.get("recall")
            lines.append(f"| {item.get('class')} | {value if value is not None else '—'} |")
    confusion = m.get("test_confusion")
    if isinstance(confusion, list) and len(confusion) == 16:
        matrix = [confusion[i:i + 4] for i in range(0, 16, 4)]
        lines += ["", "### Test confusion matrix（rows=true, columns=predicted）", "",
                  "| True / Pred | " + " | ".join(ACTION_NAMES) + " |",
                  "| --- | " + " | ".join(["---:"] * 4) + " |"]
        for name, row in zip(ACTION_NAMES, matrix):
            lines.append("| " + name + " | " + " | ".join(str(v) for v in row) + " |")
    lines += ["", "## Interpretation boundary", "",
        "Offline action-classification metrics do not equal closed-loop navigation success rate.",
        "Only a separately executed closed-loop evaluation with a saved `eval.json` may be reported as navigation SR.",
        "Missing metrics are retained as null rather than converted to zero.",
        f"- Monitor logs and lifecycle metadata: `{summary.get('monitor_dir')}`", "",
        "## Closed-loop navigation evaluations", "",
    ]
    evals = summary.get("closed_loop_evaluations", [])
    if evals:
        for ev in evals:
            lines.append(f"- {ev.get('task')} / {ev.get('split')}: SR={ev.get('success_rate')}, collision={ev.get('collision_rate')}, episodes={ev.get('episodes_done')}/{ev.get('episodes_requested')}, unique_maps={ev.get('unique_maps', ev.get('unique_geometry'))} (`{ev.get('eval_json')}`)")
    else:
        lines.append("- No formal closed-loop navigation evaluation has been attached to this run.")
    lines += ["", "## Live simulation demonstrations", ""]
    demos = summary.get("live_demos", [])
    if demos:
        for demo in demos:
            lines.append(f"- {demo.get('steps')} steps, collision_steps={demo.get('collision_steps')}, actions={demo.get('action_counts')}, output=`{demo.get('output_dir')}`")
        lines.append("- Live demos are single-episode demonstrations, not formal navigation SR.")
    else:
        lines.append("- No live student demonstration has been attached to this run.")
    lines += ["", "## Artifacts", ""]
    for name, exists in summary.get("artifacts", {}).items():
        if name.endswith("_bytes"):
            lines.append(f"- `{name}`: {exists if exists is not None else '—'}")
        else:
            lines.append(f"- `{name}`: {'present' if exists else 'not produced'}")
    (run_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _draw_chart(canvas, series, title, percent=False):
    canvas.delete("all")
    w, h = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
    if w < 80 or h < 60:
        return
    pl, pr, pt, pb = 48, 12, 26, 28
    points = [(x, y) for s in series for x, y in s.get("points", [])
              if x is not None and y is not None and math.isfinite(y)]
    canvas.create_text(12, 7, text=title, anchor="nw", fill="#E5E7EB",
                       font=("TkDefaultFont", 10, "bold"))
    if not points:
        canvas.create_text(w / 2, h / 2, text="等待真实训练指标…", fill="#94A3B8")
        return
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmin == xmax:
        xmin, xmax = xmin - 1, xmax + 1
    if ymin == ymax:
        delta = max(0.05, abs(ymin) * 0.08)
        ymin, ymax = ymin - delta, ymax + delta
    if percent:
        ymin, ymax = max(0.0, min(ymin, 0.0)), min(1.0, max(ymax, 1.0))
    def X(x): return pl + (x - xmin) / (xmax - xmin) * (w - pl - pr)
    def Y(y): return h - pb - (y - ymin) / (ymax - ymin) * (h - pt - pb)
    for i in range(5):
        y = ymin + (ymax - ymin) * i / 4
        yy = Y(y)
        canvas.create_line(pl, yy, w - pr, yy, fill="#263449")
        label = f"{y * 100:.0f}%" if percent else f"{y:.3f}"
        canvas.create_text(pl - 6, yy, text=label, anchor="e", fill="#94A3B8",
                           font=("TkDefaultFont", 8))
    canvas.create_line(pl, pt, pl, h - pb, fill="#64748B")
    canvas.create_line(pl, h - pb, w - pr, h - pb, fill="#64748B")
    for s in series:
        seg = []
        for x, y in s.get("points", []):
            if x is None or y is None:
                seg = []
                continue
            seg.append((X(x), Y(y)))
        if len(seg) == 1:
            x, y = seg[0]
            canvas.create_oval(x - 3, y - 3, x + 3, y + 3, fill=s["color"], outline="")
        elif len(seg) > 1:
            canvas.create_line(*[v for p in seg for v in p], fill=s["color"], width=2,
                               smooth=False)
        canvas.create_rectangle(pl + 8 + 130 * series.index(s), 15,
                                pl + 18 + 130 * series.index(s), 25,
                                fill=s["color"], outline="")
        canvas.create_text(pl + 24 + 130 * series.index(s), 20, text=s["name"],
                           anchor="w", fill="#CBD5E1", font=("TkDefaultFont", 8))


class StudioApp:
    def __init__(self, root, run_dir=None, refresh_ms=1000):
        if tk is None:
            raise RuntimeError("UAV Navigation Studio 需要 tkinter")
        self.root = root
        self.refresh_ms = max(300, int(refresh_ms))
        self.run_dir: Path | None = Path(run_dir).expanduser().resolve() if run_dir else None
        self.sidecar: Path | None = None
        self.run_meta = {}
        self.proc = None
        self.proc_log = None
        self._detached_pid = None
        self.eval_proc = None
        self.eval_log = None
        self.eval_out: Path | None = None
        self._train_started_mono = None
        self._metric_sig = None
        self._events = deque(maxlen=300)  # validation/status/error/class-weight events only
        self._latest_train_event = None
        self._event_tail = {}
        self._train_log_state = {}
        self._train_log_lines = 0
        self._rows = []
        self._closing = False
        self._resize_after = None
        self._sim_subs = {}
        self._sim_last_seq = {"fpv": 0, "chase": 0, "top": 0}
        self._sim_photos = {}
        self._sim_pose = None
        self._sim_path = deque(maxlen=3000)
        self._sim_obstacles = []
        self._sim_waypoint = None
        self._sim_proc = None
        self._sim_log_handle = None
        self._sim_world_path = None
        self._sim_world_sha256 = None
        self._scene_proc = None
        self._scene_log_handle = None
        self._scene_out = None
        self._sim_partition_old = os.environ.get("GZ_PARTITION")
        self._sim_partition_active = None
        self._live_proc = None
        self._live_out = None
        self._live_log = None
        self._live_last_action = None
        self._live_config = None
        self._sim_last_pose_wall = None
        self._sim_frames_seen = {"fpv": 0, "chase": 0, "top": 0}
        self._sim_status_var = tk.StringVar(value="仿真未连接")
        self._training_status_var = tk.StringVar(value="空闲")
        self._eval_status_var = tk.StringVar(value="未运行闭环评测")

        self._configure_style()
        self.root.title("UAV Navigation Studio · RGB Navigation Research Workbench")
        self.root.geometry("1500x960")
        self.root.minsize(1180, 740)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self._build_header()
        self._build_tabs()
        self._build_statusbar()
        if self.run_dir:
            self.load_run(self.run_dir)
        self.refresh_history()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._tick)
        self.root.after(300, self._sim_tick)

    def _configure_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        self.root.configure(background="#0B1220")
        style.configure("TFrame", background="#0B1220")
        style.configure("Card.TFrame", background="#111C2E")
        style.configure("TLabel", background="#0B1220", foreground="#E2E8F0")
        style.configure("Muted.TLabel", background="#0B1220", foreground="#94A3B8")
        style.configure("Title.TLabel", background="#0B1220", foreground="#F8FAFC",
                        font=("TkDefaultFont", 16, "bold"))
        style.configure("CardTitle.TLabel", background="#111C2E", foreground="#94A3B8",
                        font=("TkDefaultFont", 9, "bold"))
        style.configure("Metric.TLabel", background="#111C2E", foreground="#F8FAFC",
                        font=("TkDefaultFont", 18, "bold"))
        style.configure("TNotebook", background="#0B1220", borderwidth=0)
        style.configure("TNotebook.Tab", background="#172338", foreground="#CBD5E1",
                        padding=(16, 9))
        style.map("TNotebook.Tab", background=[("selected", "#2563EB")],
                  foreground=[("selected", "#FFFFFF")])
        style.configure("TLabelframe", background="#0B1220", foreground="#CBD5E1")
        style.configure("TLabelframe.Label", background="#0B1220", foreground="#93C5FD")
        style.configure("TButton", padding=(9, 5))
        style.configure("Treeview", background="#111C2E", fieldbackground="#111C2E",
                        foreground="#E2E8F0", rowheight=25)
        style.configure("Treeview.Heading", background="#1E293B", foreground="#E2E8F0")

    def _build_header(self):
        header = ttk.Frame(self.root, padding=(16, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="UAV NAVIGATION STUDIO", style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Label(header, text="单目 RGB · 视觉语言导航 · Gazebo 闭环实验",
                  style="Muted.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        right = ttk.Frame(header)
        right.grid(row=0, column=2, rowspan=2, sticky="e")
        ttk.Button(right, text="加载已有实验", command=self.choose_existing_run).pack(
            side="right", padx=(6, 0))
        ttk.Button(right, text="打开数据目录", command=self.open_data_root).pack(side="right")

    def _build_tabs(self):
        self.tabs = ttk.Notebook(self.root)
        self.tabs.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 8))
        self.dashboard_tab = ttk.Frame(self.tabs, padding=10)
        self.training_tab = ttk.Frame(self.tabs, padding=10)
        self.sim_tab = ttk.Frame(self.tabs, padding=10)
        self.experiments_tab = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(self.dashboard_tab, text="总览 Dashboard")
        self.tabs.add(self.training_tab, text="训练 Training")
        self.tabs.add(self.sim_tab, text="仿真 Simulation")
        self.tabs.add(self.experiments_tab, text="实验 Experiments")
        self._build_dashboard()
        self._build_training()
        self._build_simulation()
        self._build_experiments()

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, padding=(12, 4))
        bar.grid(row=2, column=0, sticky="ew")
        ttk.Label(bar, textvariable=self._training_status_var).pack(side="left")
        ttk.Label(bar, text="  ·  ", style="Muted.TLabel").pack(side="left")
        ttk.Label(bar, textvariable=self._sim_status_var).pack(side="left")
        ttk.Label(bar, text="  ·  ", style="Muted.TLabel").pack(side="left")
        ttk.Label(bar, textvariable=self._eval_status_var).pack(side="left")

    def _metric_card(self, parent, title, var, col):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        frame.grid(row=0, column=col, sticky="nsew", padx=4)
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=var, style="Metric.TLabel").pack(anchor="w", pady=(6, 0))
        return frame

    def _build_dashboard(self):
        tab = self.dashboard_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        self.run_title_var = tk.StringVar(value="尚未选择实验")
        title_bar = ttk.Frame(tab)
        title_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        title_bar.columnconfigure(0, weight=1)
        ttk.Label(title_bar, textvariable=self.run_title_var, style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Button(title_bar, text="打开三视角仿真 →",
                   command=lambda: self.tabs.select(self.sim_tab)).grid(row=0, column=1, sticky="e")
        cards = ttk.Frame(tab)
        cards.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        for i in range(7):
            cards.columnconfigure(i, weight=1)
        self.card_vars = {k: tk.StringVar(value="—") for k in
                          ("epoch", "train_ce", "dev_ce", "dev_macro", "test_macro", "test_bal", "closed_loop_sr")}
        titles = [("训练轮数", "epoch"), ("Train CE", "train_ce"), ("Validation CE", "dev_ce"),
                  ("Best Val Macro-F1", "dev_macro"), ("Test Macro-F1", "test_macro"),
                  ("Test Balanced Acc", "test_bal"), ("闭环导航 SR", "closed_loop_sr")]
        for i, (title, key) in enumerate(titles):
            self._metric_card(cards, title, self.card_vars[key], i)
        body = ttk.Frame(tab)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        charts = ttk.Frame(body)
        charts.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        charts.columnconfigure(0, weight=1)
        charts.rowconfigure(0, weight=1)
        charts.rowconfigure(1, weight=1)
        self.loss_canvas = tk.Canvas(charts, bg="#111C2E", highlightthickness=1,
                                     highlightbackground="#263449")
        self.loss_canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 6))
        self.acc_canvas = tk.Canvas(charts, bg="#111C2E", highlightthickness=1,
                                    highlightbackground="#263449")
        self.acc_canvas.grid(row=1, column=0, sticky="nsew")
        self.loss_canvas.bind("<Configure>", self._redraw_charts)
        self.acc_canvas.bind("<Configure>", self._redraw_charts)
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="训练事件与诊断", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 6))
        self.event_text = tk.Text(right, bg="#111C2E", fg="#CBD5E1", insertbackground="#fff",
                                  relief="flat", wrap="word", font=("TkFixedFont", 9), height=20)
        self.event_text.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(right, command=self.event_text.yview)
        sb.grid(row=1, column=1, sticky="ns")
        self.event_text.configure(yscrollcommand=sb.set)
        self.test_summary_var = tk.StringVar(value="尚无 test split 离线评测记录。")
        ttk.Label(right, textvariable=self.test_summary_var, style="Muted.TLabel",
                  wraplength=420, justify="left").grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.confusion_var = tk.StringVar(value="Test confusion matrix：—")
        ttk.Label(right, textvariable=self.confusion_var, style="Muted.TLabel",
                  wraplength=420, justify="left").grid(row=3, column=0, sticky="ew", pady=(5, 0))
        self.per_class_var = tk.StringVar(value="Per-class Recall：—")
        ttk.Label(right, textvariable=self.per_class_var, style="Muted.TLabel",
                  wraplength=420, justify="left").grid(row=4, column=0, sticky="ew", pady=(4, 0))

    def _build_training(self):
        tab = self.training_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        config = ttk.LabelFrame(tab, text="真实训练配置（调用 C++/CUDA uavrgb_train）", padding=12)
        config.grid(row=0, column=0, sticky="ew")
        config.columnconfigure(1, weight=1)
        self.data_var = tk.StringVar(value="")
        self.epochs_var = tk.StringVar(value="20")
        self.batch_var = tk.StringVar(value="64")
        self.lr_var = tk.StringVar(value="0.001")
        self.seed_var = tk.StringVar(value="1234")
        self.text_dim_var = tk.StringVar(value="16")
        self.class_weights_var = tk.BooleanVar(value=True)
        self.drop_collision_var = tk.BooleanVar(value=True)
        ttk.Label(config, text="训练数据集 (.uavr)").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(config, textvariable=self.data_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(config, text="选择数据集…", command=self.choose_dataset).grid(row=0, column=2)
        fields = [("Epochs", self.epochs_var), ("Batch", self.batch_var),
                  ("Learning rate", self.lr_var), ("Seed", self.seed_var),
                  ("Text dim（0=纯视觉）", self.text_dim_var)]
        for i, (label, var) in enumerate(fields):
            r = 1 + i // 3
            c = (i % 3) * 2
            ttk.Label(config, text=label).grid(row=r, column=c, sticky="w", pady=4)
            ttk.Entry(config, textvariable=var, width=13).grid(row=r, column=c + 1,
                                                               sticky="w", padx=(6, 18))
        ttk.Checkbutton(config, text="类别权重 auto", variable=self.class_weights_var).grid(
            row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(config, text="剔除碰撞样本", variable=self.drop_collision_var).grid(
            row=3, column=1, sticky="w", pady=(6, 0))
        buttons = ttk.Frame(tab)
        buttons.grid(row=1, column=0, sticky="ew", pady=10)
        self.start_train_btn = ttk.Button(buttons, text="▶ 开始真实训练", command=self.start_training)
        self.start_train_btn.pack(side="left")
        self.stop_train_btn = ttk.Button(buttons, text="■ 停止训练", command=self.stop_training,
                                         state="disabled")
        self.stop_train_btn.pack(side="left", padx=8)
        self.stop_detached_btn = ttk.Button(buttons, text="停止后台训练", command=self.stop_detached_training,
                                            state="disabled")
        self.stop_detached_btn.pack(side="left", padx=4)
        ttk.Button(buttons, text="刷新指标", command=lambda: self.refresh_run(force=True)).pack(
            side="left")
        ttk.Label(buttons, textvariable=self._training_status_var, style="Muted.TLabel").pack(
            side="right")
        lower = ttk.Frame(tab)
        lower.grid(row=2, column=0, sticky="nsew")
        lower.columnconfigure(0, weight=1)
        lower.rowconfigure(0, weight=1)
        self.train_log_text = tk.Text(lower, bg="#0F172A", fg="#CBD5E1", insertbackground="#fff",
                                      relief="flat", wrap="none", font=("TkFixedFont", 9))
        self.train_log_text.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(lower, command=self.train_log_text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.train_log_text.configure(yscrollcommand=sb.set)

    def _build_simulation(self):
        tab = self.sim_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        controls = ttk.LabelFrame(tab, text="Gazebo 连接与场景", padding=9)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        self.world_var = tk.StringVar(value=str(REPO_ROOT / "sim" / "worlds" / "uav_rgb.sdf"))
        self.meta_var = tk.StringVar(value="")
        self.partition_var = tk.StringVar(value="")
        ttk.Label(controls, text="World SDF").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(controls, textvariable=self.world_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(controls, text="选择…", command=self.choose_world).grid(row=0, column=2)
        ttk.Label(controls, text="Scene metadata（可选但推荐）").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(controls, textvariable=self.meta_var).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Button(controls, text="选择…", command=self.choose_scene_meta).grid(row=1, column=2)
        ttk.Label(controls, text="GZ_PARTITION（空=default；启动时自动隔离）").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(controls, textvariable=self.partition_var, width=28).grid(
            row=2, column=1, sticky="w", padx=5)
        self.scene_task_var = tk.StringVar(value="campus")
        self.scene_seed_var = tk.StringVar(value="42")
        ttk.Label(controls, text="快速生成场景").grid(row=3, column=0, sticky="w", pady=3)
        gen_row = ttk.Frame(controls)
        gen_row.grid(row=3, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Combobox(gen_row, textvariable=self.scene_task_var,
                     values=("campus", "base", "l1"), width=10, state="readonly").pack(side="left")
        ttk.Label(gen_row, text="Seed").pack(side="left", padx=(10, 4))
        ttk.Entry(gen_row, textvariable=self.scene_seed_var, width=8).pack(side="left")
        ttk.Button(gen_row, text="生成 SDF + metadata", command=self.generate_scene).pack(side="left", padx=8)
        row = ttk.Frame(controls)
        row.grid(row=0, column=3, rowspan=4, sticky="ns", padx=(12, 0))
        ttk.Button(row, text="启动 Gazebo GUI", command=self.start_gazebo).pack(fill="x", pady=2)
        ttk.Button(row, text="连接当前仿真", command=self.connect_sim).pack(fill="x", pady=2)
        ttk.Button(row, text="断开视角", command=self.disconnect_sim).pack(fill="x", pady=2)
        ttk.Button(row, text="停止本程序启动的 Gazebo", command=self.stop_gazebo).pack(fill="x", pady=2)

        live = ttk.LabelFrame(tab, text="学生策略实时演示（使用真实 checkpoint，不是教师动作）", padding=9)
        live.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        live.columnconfigure(1, weight=1)
        self.ckpt_var = tk.StringVar(value="")
        self.live_steps_var = tk.StringVar(value="80")
        self.live_hold_var = tk.StringVar(value="1.0")
        ttk.Label(live, text="Checkpoint").grid(row=0, column=0, sticky="w")
        ttk.Entry(live, textvariable=self.ckpt_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(live, text="选择…", command=self.choose_checkpoint).grid(row=0, column=2)
        ttk.Label(live, text="Steps").grid(row=0, column=3, padx=(12, 0))
        ttk.Entry(live, textvariable=self.live_steps_var, width=7).grid(row=0, column=4, padx=5)
        ttk.Label(live, text="Hold(s)").grid(row=0, column=5)
        ttk.Entry(live, textvariable=self.live_hold_var, width=7).grid(row=0, column=6, padx=5)
        self.live_btn = ttk.Button(live, text="▶ 在当前仿真运行学生策略", command=self.start_live_student)
        self.live_btn.grid(row=0, column=7, padx=(10, 4))
        self.live_stop_btn = ttk.Button(live, text="停止策略", command=self.stop_live_student,
                                        state="disabled")
        self.live_stop_btn.grid(row=0, column=8)
        self.live_action_var = tk.StringVar(value="动作：—")
        ttk.Label(live, textvariable=self.live_action_var, style="Muted.TLabel").grid(
            row=1, column=0, columnspan=9, sticky="w", pady=(5, 0))

        views = ttk.Frame(tab)
        views.grid(row=2, column=0, sticky="nsew")
        views.columnconfigure(0, weight=1)
        views.columnconfigure(1, weight=1)
        views.rowconfigure(0, weight=1)
        views.rowconfigure(1, weight=1)
        self.chase_panel, self.chase_label = self._camera_panel(views, "第三人称跟随视角 · CHASE CAMERA")
        self.chase_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        self.fpv_panel, self.fpv_label = self._camera_panel(views, "第一人称视角 · RGB INPUT")
        self.fpv_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        self.topcam_panel, self.topcam_label = self._camera_panel(views, "顶视相机 · RENDERED TOP CAMERA")
        self.topcam_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(4, 0))
        top = ttk.LabelFrame(views, text="轨迹分析地图 · TOP-DOWN MAP", padding=4)
        top.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(4, 0))
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        self.top_canvas = tk.Canvas(top, bg="#0B1422", highlightthickness=0, height=300)
        self.top_canvas.grid(row=0, column=0, sticky="nsew")
        self.top_canvas.bind("<Configure>", lambda _e: self._draw_topdown())
        self.sim_info_var = tk.StringVar(value="等待 Gazebo 位姿数据")
        ttk.Label(top, textvariable=self.sim_info_var, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0))

    def _camera_panel(self, parent, title):
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        label = tk.Label(frame, text="等待相机数据…", bg="#0B1422", fg="#94A3B8",
                         font=("TkDefaultFont", 12), anchor="center")
        label.grid(row=0, column=0, sticky="nsew")
        return frame, label

    def _build_experiments(self):
        tab = self.experiments_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(tab)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(toolbar, text="刷新实验列表", command=self.refresh_history).pack(side="left")
        ttk.Button(toolbar, text="加载选中实验", command=self.load_selected_history).pack(
            side="left", padx=6)
        ttk.Button(toolbar, text="打开实验目录", command=self.open_selected_history).pack(side="left")
        ttk.Button(toolbar, text="打开实验报告", command=self.open_report).pack(side="left", padx=6)
        table = ttk.Treeview(tab, columns=("status", "epochs", "macro", "test", "sr", "started"),
                             show="tree headings", height=12)
        table.heading("#0", text="Run ID")
        for col, label in (("status", "状态"), ("epochs", "Epochs"),
                           ("macro", "Test Macro-F1"), ("test", "Test Balanced Acc"),
                           ("sr", "Closed-loop SR"), ("started", "开始时间")):
            table.heading(col, text=label)
        table.column("#0", width=330)
        table.column("status", width=90, anchor="center")
        table.column("epochs", width=90, anchor="e")
        table.column("macro", width=120, anchor="e")
        table.column("test", width=140, anchor="e")
        table.column("sr", width=110, anchor="e")
        table.column("started", width=190)
        table.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(tab, command=table.yview)
        sb.grid(row=1, column=1, sticky="ns")
        table.configure(yscrollcommand=sb.set)
        self.history_tree = table
        eval_box = ttk.LabelFrame(tab, text="独立 Gazebo 闭环评测（训练分类指标 ≠ 导航成功率）", padding=10)
        eval_box.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        self.eval_task_var = tk.StringVar(value="base")
        self.eval_split_var = tk.StringVar(value="test")
        self.eval_episodes_var = tk.StringVar(value="20")
        self.eval_appearance_var = tk.StringVar(value="v1")
        ttk.Label(eval_box, text="Task（必须匹配训练数据）").pack(side="left")
        ttk.Combobox(eval_box, textvariable=self.eval_task_var,
                     values=("base", "l1", "campus"), width=10, state="readonly").pack(
            side="left", padx=5)
        ttk.Label(eval_box, text="Split").pack(side="left")
        ttk.Combobox(eval_box, textvariable=self.eval_split_var,
                     values=("train", "val", "test"), width=8, state="readonly").pack(
            side="left", padx=5)
        ttk.Label(eval_box, text="Episodes").pack(side="left")
        ttk.Entry(eval_box, textvariable=self.eval_episodes_var, width=7).pack(side="left", padx=5)
        ttk.Label(eval_box, text="Appearance").pack(side="left")
        ttk.Combobox(eval_box, textvariable=self.eval_appearance_var,
                     values=("legacy", "v1"), width=8, state="readonly").pack(side="left", padx=5)
        self.eval_btn = ttk.Button(eval_box, text="运行正式闭环评测", command=self.start_closed_loop_eval)
        self.eval_btn.pack(side="left", padx=(12, 4))
        self.eval_stop_btn = ttk.Button(eval_box, text="停止评测", command=self.stop_closed_loop_eval,
                                        state="disabled")
        self.eval_stop_btn.pack(side="left", padx=4)
        ttk.Button(eval_box, text="打开评测目录", command=self.open_eval_dir).pack(side="left")
        ttk.Label(tab, textvariable=self._eval_status_var, style="Muted.TLabel").grid(
            row=3, column=0, sticky="w", pady=(8, 0))

    # ------------------------------------------------------------------ training
    def choose_dataset(self):
        path = filedialog.askopenfilename(title="选择 .uavr 数据集",
                                          filetypes=[("UAV RGB dataset", "*.uavr"), ("All", "*.*")])
        if path:
            self.data_var.set(path)
            sidecar = Path(path).parent / "instructions.jsonl"
            self.text_dim_var.set("16" if sidecar.is_file() else "0")

    def _new_run_dir(self):
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return RUNS_ROOT / f"{stamp}-{uuid.uuid4().hex[:8]}"

    def start_training(self):
        if self.proc is not None:
            return
        if self._detached_pid is not None and self.run_dir and self._run_pid_alive(self._detached_pid, self.run_dir):
            messagebox.showwarning("后台训练仍在运行", "请先停止或等待当前后台训练结束，再启动新训练。")
            return
        self._detached_pid = None
        data = Path(self.data_var.get()).expanduser().resolve()
        if not data.is_file():
            messagebox.showerror("训练无法启动", "请先选择存在的 .uavr 数据集文件。")
            return
        if not os.access(TRAINER_BIN, os.X_OK):
            messagebox.showerror("训练器不存在",
                                 f"找不到可执行文件：\n{TRAINER_BIN}\n\n先在 uav_rgb 目录执行 bash tools/check.sh。")
            return
        try:
            epochs = int(self.epochs_var.get())
            batch = int(self.batch_var.get())
            seed = int(self.seed_var.get())
            text_dim = int(self.text_dim_var.get())
            lr = float(self.lr_var.get())
            if epochs < 1 or batch < 1 or seed < 0 or text_dim < 0 or not (math.isfinite(lr) and 0 < lr <= 1):
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "Epochs/Batch/Seed/Text dim 必须为有效整数，lr 必须满足 0<lr<=1。")
            return
        text_sidecar = data.parent / "instructions.jsonl"
        if text_dim > 0 and not text_sidecar.is_file():
            messagebox.showerror("文本侧车缺失",
                                 f"当前 text_dim={text_dim}，但数据集目录没有 instructions.jsonl：\n{text_sidecar}\n\n如果这是纯 RGB 数据，请把 Text dim 设为 0。")
            return
        run_dir = self._new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=False)
        sidecar = Path(str(run_dir) + ".monitor")
        sidecar.mkdir(parents=True, exist_ok=True)
        config = {
            "schema": "uavrgb.train.v1", "run_dir": str(run_dir), "data": str(data),
            "epochs": epochs, "batch_size": batch, "lr": lr, "seed": seed,
            "text_dim": text_dim, "class_weights": "auto" if self.class_weights_var.get() else "none",
            "drop_collision": bool(self.drop_collision_var.get()),
        }
        extra = []
        if self.class_weights_var.get():
            extra += ["--class-weights", "auto"]
        if self.drop_collision_var.get():
            extra.append("--drop-collision")
        extra += ["--text-dim", str(text_dim)] if text_dim > 0 else ["--vision-only"]
        config_path = sidecar / "train_config.json"
        write_json_atomic(config_path, config)
        command = [sys.executable, str(TRAIN_NATIVE), "--config", str(config_path), *extra]
        run_meta = {
            "run_id": run_dir.name, "started_at": now_iso(), "started_monotonic": time.monotonic(),
            "config": config, "command": command,
            "dataset": {"path": str(data), "bytes": data.stat().st_size,
                        "mtime_ns": data.stat().st_mtime_ns},
        }
        self.run_meta = run_meta
        write_json_atomic(sidecar / "job.json", run_meta)
        write_json_atomic(sidecar / "lifecycle.json", {"status": "starting", "started_at": run_meta["started_at"]})
        log_path = sidecar / "train.log"
        self.proc_log = log_path.open("ab", buffering=0)
        try:
            self.proc = subprocess.Popen(command, cwd=str(REPO_ROOT), stdin=subprocess.DEVNULL,
                                         stdout=self.proc_log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        except OSError as exc:
            self.proc_log.close()
            self.proc_log = None
            write_json_atomic(sidecar / "lifecycle.json", {"status": "failed_to_start", "error": str(exc)})
            failed_summary = summarize_run(run_dir, run_meta, 127, "failed_to_start")
            failed_summary["notes"].append(f"Trainer process could not be started: {exc}")
            write_json_atomic(run_dir / "studio_summary.json", failed_summary)
            write_experiment_report(run_dir, failed_summary)
            self.run_dir, self.sidecar = run_dir, sidecar
            self.refresh_history()
            self.run_title_var.set(f"当前实验：{run_dir.name}（启动失败）")
            self._training_status_var.set("训练启动失败；失败记录已保存")
            messagebox.showerror("启动失败", str(exc))
            return
        # pid + running 必须落盘：重开 Studio 时靠它判断后台训练是否还活着，
        # 只写 starting 会让恢复逻辑永远读不到 pid，把在跑的训练判成 unobserved。
        write_json_atomic(sidecar / "lifecycle.json",
                          {"status": "running", "pid": self.proc.pid,
                           "started_at": run_meta["started_at"]})
        self.run_dir, self.sidecar = run_dir, sidecar
        self._train_started_mono = time.monotonic()
        self._metric_sig = None
        self._events.clear()
        self._latest_train_event = None
        self._event_tail = {}
        self._train_log_state = {}
        self._train_log_lines = 0
        self._rows = []
        self._training_status_var.set(f"真实训练运行中 · PID {self.proc.pid}")
        self.start_train_btn.configure(state="disabled")
        self.stop_train_btn.configure(state="normal")
        self.tabs.select(self.dashboard_tab)
        self.run_title_var.set(f"当前实验：{run_dir.name}")
        self._append_train_log(f"[studio] started: {shlex.join(command)}")
        self.refresh_history()

    def stop_detached_training(self):
        pid = self._detached_pid
        if pid is None or not self._run_pid_alive(pid, self.run_dir or Path(".")):
            self._detached_pid = None
            self.stop_detached_btn.configure(state="disabled")
            self.start_train_btn.configure(state="normal")
            return
        try:
            os.killpg(pid, signal.SIGINT)
            self._training_status_var.set(f"已向后台训练进程组 {pid} 发送 SIGINT")
        except (ProcessLookupError, PermissionError, OSError) as exc:
            self._training_status_var.set(f"停止后台训练失败：{exc}")

    def stop_training(self):
        if self.proc is None:
            return
        self._training_status_var.set("正在请求停止训练…")
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.proc.send_signal(signal.SIGINT)
            except OSError:
                pass

    def _finish_training(self, rc):
        proc = self.proc
        self.proc = None
        if self.proc_log:
            self.proc_log.close()
            self.proc_log = None
        state = "finished" if rc == 0 else ("stopped_or_interrupted" if rc in (-2, 130, 143) else "failed")
        if self.run_dir is not None:
            summary = summarize_run(self.run_dir, self.run_meta, rc, state)
            write_json_atomic(self.run_dir / "studio_summary.json", summary)
            write_experiment_report(self.run_dir, summary)
            if self.sidecar:
                write_json_atomic(self.sidecar / "lifecycle.json", {
                    "status": state, "return_code": rc, "finished_at": summary["finished_at"]})
            try:
                with (RUNS_ROOT / "index.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"run_id": summary["run_id"], "run_dir": str(self.run_dir),
                                        "status": state, "finished_at": summary["finished_at"]},
                                       ensure_ascii=False) + "\n")
            except OSError:
                pass
            self._refresh_train_log()
            self._training_status_var.set(f"训练结束：{state} · rc={rc}")
            best_ckpt = self.run_dir / "ckpt_best.bin"
            if best_ckpt.is_file():
                self.ckpt_var.set(str(best_ckpt))
            self._append_train_log(f"[studio] training {state}, rc={rc}")
            self.refresh_run(force=True)
            self.refresh_history()
            self.tabs.select(self.dashboard_tab)
        self.start_train_btn.configure(state="normal")
        self.stop_train_btn.configure(state="disabled")

    def _refresh_train_log(self):
        if not self.sidecar:
            return
        path = self.sidecar / "train.log"
        text, reset = monitor_io.tail_text(str(path), self._train_log_state)
        if reset:
            self._train_log_lines = 0
            self.train_log_text.delete("1.0", "end")
        if text:
            self._append_train_log(text.rstrip("\n"), to_dashboard=False)

    def _append_train_log(self, text, to_dashboard=True):
        if not text:
            return
        self.train_log_text.insert("end", text.rstrip("\n") + "\n")
        self.train_log_text.see("end")
        self._train_log_lines += text.count("\n") + 1
        if self._train_log_lines > 3000:
            excess = self._train_log_lines - 3000
            self.train_log_text.delete("1.0", f"{excess + 1}.0")
            self._train_log_lines = 3000
        if to_dashboard:
            try:
                self.event_text.insert("end", text.rstrip("\n") + "\n")
                self.event_text.see("end")
            except tk.TclError:
                pass

    def refresh_run(self, force=False):
        if not self.run_dir or not self.run_dir.exists():
            return
        paths = [self.run_dir / "results.csv", self.run_dir / "metrics.jsonl"]
        sig = []
        for p in paths:
            try:
                st = p.stat()
                sig.append((st.st_mtime_ns, st.st_size, st.st_ino))
            except OSError:
                sig.append(None)
        sig = tuple(sig)
        if not force and sig == self._metric_sig:
            return
        self._rows = read_training_rows(self.run_dir)
        events, reset = monitor_io.tail_events(str(self.run_dir), self._event_tail)
        if reset:
            self._events.clear()
            self._latest_train_event = None
        for event in events:
            if event.get("event") == "train_step":
                self._latest_train_event = event
            else:
                self._events.append(event)
        self._metric_sig = sig
        self._update_dashboard()
        self._update_event_text()

    def _update_dashboard(self):
        rows = self._rows
        valid = [r for r in rows if r.get("epoch") is not None]
        last = valid[-1] if valid else {}
        evs = list(self._events)
        vals = [e for e in evs if e.get("event") == "validation" and e.get("split") == "val"]
        tests = [e for e in evs if e.get("event") == "validation" and e.get("split") == "test"]
        def metric_key(e):
            v = safe_float(e.get("macro_f1"))
            return v if v is not None else -1.0
        best = max(vals, key=metric_key, default=None)
        test = tests[-1] if tests else None
        self.card_vars["epoch"].set(str(len(valid)))
        self.card_vars["train_ce"].set(f"{last['train_ce']:.4f}" if last.get("train_ce") is not None else "—")
        self.card_vars["dev_ce"].set(f"{last['dev_ce']:.4f}" if last.get("dev_ce") is not None else "—")
        self.card_vars["dev_macro"].set(f"{safe_float(best.get('macro_f1')):.3f}" if best and safe_float(best.get("macro_f1")) is not None else "—")
        self.card_vars["test_macro"].set(f"{safe_float(test.get('macro_f1')):.3f}" if test and safe_float(test.get("macro_f1")) is not None else "—")
        self.card_vars["test_bal"].set(f"{safe_float(test.get('balanced_accuracy')):.3f}" if test and safe_float(test.get("balanced_accuracy")) is not None else "—")
        latest_sr = None
        try:
            run_summary = json.loads((self.run_dir / "studio_summary.json").read_text(encoding="utf-8"))
            evals = run_summary.get("closed_loop_evaluations", [])
            if evals:
                latest_sr = safe_float(evals[-1].get("success_rate"))
        except (OSError, ValueError, TypeError):
            pass
        self.card_vars["closed_loop_sr"].set(f"{latest_sr:.3f}" if latest_sr is not None else "—")
        if test:
            self.test_summary_var.set(
                f"离线 test：Acc={test.get('action_accuracy')} · Macro-F1={test.get('macro_f1')} · "
                f"Balanced Acc={test.get('balanced_accuracy')}。这是动作分类指标，不是闭环导航 SR。")
        else:
            self.test_summary_var.set("当前没有 test split validation 事件；缺失指标保持 —，不会用训练准确率代替。")
        confusion = test.get("confusion") if test else None
        if isinstance(confusion, list) and len(confusion) == 16:
            matrix = [confusion[i:i + 4] for i in range(0, 16, 4)]
            self.confusion_var.set("Test confusion matrix（行=真值，列=预测；SAFE_NEUTRAL/FORWARD_SLOW/YAW_LEFT/YAW_RIGHT）：\n" +
                                   "\n".join("  " + "  ".join(str(v) for v in row) for row in matrix))
            recalls = []
            for i, row in enumerate(matrix):
                total = sum(safe_float(v) or 0.0 for v in row)
                diagonal = safe_float(row[i])
                recalls.append(f"{ACTION_NAMES[i]}={diagonal / total:.3f}" if total > 0 and diagonal is not None else f"{ACTION_NAMES[i]}=—")
            self.per_class_var.set("Per-class Recall：" + " · ".join(recalls))
        else:
            self.confusion_var.set("Test confusion matrix：—（当前 run 未提供有效 test confusion）")
            self.per_class_var.set("Per-class Recall：—")
        self._redraw_charts()

    def _update_event_text(self):
        lines = []
        if self._latest_train_event:
            e = self._latest_train_event
            lines.append(f"[TRAIN LIVE] step {e.get('step')}/{e.get('total_steps')}  loss={e.get('loss')}  acc={e.get('action_accuracy')}  lr={e.get('lr')}  ETA={e.get('eta_sec')}")
        for e in list(self._events)[-120:]:
            typ = e.get("event", "?")
            if typ == "train_step":
                lines.append(f"[TRAIN] step {e.get('step')}/{e.get('total_steps')}  loss={e.get('loss')}  acc={e.get('action_accuracy')}  lr={e.get('lr')}  ETA={e.get('eta_sec')}")
            elif typ == "validation":
                lines.append(f"[VALIDATION/{e.get('split')}] loss={e.get('loss')} acc={e.get('action_accuracy')} macro_f1={e.get('macro_f1')} balanced_acc={e.get('balanced_accuracy')}")
            elif typ == "error":
                lines.append(f"[ERROR/{e.get('severity')}] {e.get('code')}: {e.get('message')}")
            elif typ == "status":
                lines.append(f"[STATUS] {e.get('message')}")
            elif typ == "class_weights":
                lines.append(f"[CLASS WEIGHTS] epoch={e.get('epoch')} weights={e.get('class_weights')} source={e.get('class_weights_source')}")
        self.event_text.delete("1.0", "end")
        self.event_text.insert("1.0", "\n".join(lines) if lines else "等待 C++ trainer 写入 metrics.jsonl…")
        self.event_text.see("end")

    def _redraw_charts(self, _event=None):
        if not hasattr(self, "loss_canvas"):
            return
        epochs = [r["epoch"] for r in self._rows]
        _draw_chart(self.loss_canvas, [
            {"name": "train CE", "color": "#60A5FA", "points": list(zip(epochs, [r["train_ce"] for r in self._rows]))},
            {"name": "dev CE", "color": "#F59E0B", "points": list(zip(epochs, [r["dev_ce"] for r in self._rows]))},
        ], "LOSS CURVE · cross entropy")
        evs = list(self._events)
        vals = [e for e in evs if e.get("event") == "validation" and e.get("split") == "val"]
        _draw_chart(self.acc_canvas, [
            {"name": "accuracy", "color": "#34D399", "points": [(i + 1, safe_float(e.get("action_accuracy"))) for i, e in enumerate(vals)]},
            {"name": "macro F1", "color": "#A78BFA", "points": [(i + 1, safe_float(e.get("macro_f1"))) for i, e in enumerate(vals)]},
            {"name": "balanced acc", "color": "#F472B6", "points": [(i + 1, safe_float(e.get("balanced_accuracy"))) for i, e in enumerate(vals)]},
        ], "VALIDATION METRICS · accuracy / Macro-F1 / Balanced Accuracy", percent=True)

    # ---------------------------------------------------------------- simulation
    def choose_world(self):
        path = filedialog.askopenfilename(title="选择 Gazebo world SDF",
                                          filetypes=[("SDF", "*.sdf"), ("All", "*.*")])
        if path:
            self.world_var.set(path)
            self._load_world_geometry()

    def choose_scene_meta(self):
        path = filedialog.askopenfilename(title="选择 scene metadata JSON",
                                          filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.meta_var.set(path)
            self._load_world_geometry()

    def choose_checkpoint(self):
        path = filedialog.askopenfilename(title="选择学生 checkpoint",
                                          filetypes=[("UAV RGB checkpoint", "*.bin"), ("All", "*.*")])
        if path:
            self.ckpt_var.set(path)

    def _load_world_geometry(self):
        self._sim_obstacles = parse_sdf_obstacles(self.world_var.get())
        meta = load_scene_meta(self.meta_var.get())
        wp = meta.get("waypoint") if isinstance(meta.get("waypoint"), dict) else None
        self._sim_waypoint = (safe_float(wp.get("x")), safe_float(wp.get("y"))) if wp else None
        if not self._sim_obstacles and isinstance(meta.get("pillars"), list):
            for p in meta["pillars"]:
                try:
                    sx, sy = p.get("size", [0.4, 0.4, 1.0])[:2]
                    self._sim_obstacles.append({"name": p.get("name", "obstacle"), "kind": "box",
                                                "x": float(p["x"]), "y": float(p["y"]),
                                                "sx": float(sx), "sy": float(sy), "yaw": 0.0})
                except (KeyError, TypeError, ValueError):
                    continue
        self._draw_topdown()

    def _set_partition(self, partition):
        if self._sim_partition_active is not None:
            return
        self._sim_partition_old = os.environ.get("GZ_PARTITION")
        if partition:
            os.environ["GZ_PARTITION"] = partition
        else:
            os.environ.pop("GZ_PARTITION", None)
        self._sim_partition_active = partition or "default"

    def generate_scene(self):
        if self._scene_proc is not None:
            return
        try:
            seed = int(self.scene_seed_var.get())
            if seed < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Seed 错误", "Seed 必须是非负整数。")
            return
        if not Path(PYTHON_FOR_SIM).is_file() and not shutil.which(PYTHON_FOR_SIM):
            messagebox.showerror("Python 环境不存在",
                                 f"找不到 UAV_RGB_PYTHON：{PYTHON_FOR_SIM}\n请先执行 tools/setup_env.sh。")
            return
        task = self.scene_task_var.get()
        scene_root = DATA_ROOT / "uav_rgb" / "studio_scenes"
        scene_root.mkdir(parents=True, exist_ok=True)
        stem = f"scene_{task}_seed{seed}_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:4]}"
        out = scene_root / (stem + ".sdf")
        log_root = scene_root / "logs"
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / (stem + ".log")
        log = log_path.open("ab", buffering=0)
        cmd = [PYTHON_FOR_SIM, str(SIM_TOOLS / "scene_gen.py"), "--seed", str(seed),
               "--out", str(out), "--task", task]
        try:
            self._scene_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
                                                stdin=subprocess.DEVNULL, stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.close()
            messagebox.showerror("场景生成启动失败", str(exc))
            return
        self._scene_log_handle = log
        self._scene_out = out
        self._sim_status_var.set(f"场景生成中 · task={task} seed={seed}")

    def _finish_scene_generation(self, rc):
        if self._scene_log_handle:
            self._scene_log_handle.close()
            self._scene_log_handle = None
        out = self._scene_out
        self._scene_proc = None
        if rc == 0 and out and out.is_file():
            meta = out.with_suffix(".json")
            self.world_var.set(str(out))
            self.meta_var.set(str(meta) if meta.is_file() else "")
            self._load_world_geometry()
            self._sim_status_var.set(f"场景已生成 · {out.name}；现在可以启动 Gazebo GUI")
        else:
            self._sim_status_var.set(f"场景生成失败 rc={rc} · 查看 {out}")

    def start_gazebo(self):
        if self._scene_proc is not None and self._scene_proc.poll() is None:
            messagebox.showwarning("场景仍在生成", "请等待 SDF 和 metadata 生成完成后再启动 Gazebo。")
            return
        if self._sim_proc is not None and self._sim_proc.poll() is None:
            messagebox.showinfo("Gazebo 已运行", "当前 Studio 已经启动一个 Gazebo 实例。")
            return
        world = Path(self.world_var.get()).expanduser().resolve()
        if not world.is_file():
            messagebox.showerror("World 不存在", f"找不到 SDF：{world}")
            return
        partition = self.partition_var.get().strip() or ("uavrgb_studio_" + uuid.uuid4().hex[:10])
        self.partition_var.set(partition)
        self._set_partition(partition)
        worlds_resource = str(REPO_ROOT / "sim" / "worlds")
        existing_resources = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
        resources = [worlds_resource] + [x for x in existing_resources.split(os.pathsep) if x]
        os.environ["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(dict.fromkeys(resources))
        env = os.environ.copy()
        env["GZ_PARTITION"] = partition
        log_dir = RUNS_ROOT / "simulation_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-gazebo.log")
        self._sim_log_handle = log_path.open("ab", buffering=0)
        self._sim_world_path = world
        self._sim_world_sha256 = sha256_file(world)
        try:
            self._sim_proc = subprocess.Popen(["gz", "sim", "-r", str(world)], cwd=str(REPO_ROOT),
                                              env=env, stdin=subprocess.DEVNULL,
                                              stdout=self._sim_log_handle, stderr=subprocess.STDOUT,
                                              start_new_session=True)
        except OSError as exc:
            self._sim_log_handle.close()
            self._sim_log_handle = None
            self._sim_world_path = None
            self._sim_world_sha256 = None
            self.disconnect_sim()
            self._sim_status_var.set(f"Gazebo 启动失败：{exc}")
            return
        self._sim_status_var.set(f"Gazebo GUI 启动中 · partition={partition}")
        self._load_world_geometry()
        self.root.after(3500, self.connect_sim)

    def connect_sim(self):
        if self._sim_subs:
            self.disconnect_sim()
        partition = self.partition_var.get().strip()
        self._set_partition(partition or None)
        new_subs = {}
        try:
            new_subs["fpv"] = gz_io.TopicSub(CAMERA_TOPIC, buf=3)
            new_subs["chase"] = gz_io.TopicSub(CHASE_TOPIC, buf=3)
            new_subs["top"] = gz_io.TopicSub(TOP_TOPIC, buf=3)
            new_subs["pose"] = gz_io.TopicSub(POSE_TOPIC, buf=8, stamp_fn=pose_stamp_ns)
            self._sim_subs = new_subs
        except Exception as exc:
            for sub in new_subs.values():
                try:
                    sub.close()
                except Exception:
                    pass
            self._sim_subs = {}
            self.disconnect_sim()
            self._sim_status_var.set(f"订阅失败：{exc}")
            return
        self._sim_last_seq = {"fpv": 0, "chase": 0, "top": 0}
        self._sim_path.clear()
        self._sim_status_var.set("已连接 Gazebo Transport · 等待相机帧")
        self._load_world_geometry()

    def disconnect_sim(self):
        for sub in self._sim_subs.values():
            try:
                sub.close()
            except Exception:
                pass
        self._sim_subs = {}
        for key, label in (("fpv", self.fpv_label), ("chase", self.chase_label),
                           ("top", self.topcam_label)):
            try:
                label.configure(image="", text="视角已断开")
            except (tk.TclError, AttributeError):
                pass
            self._sim_photos.pop(key, None)
        if self._sim_partition_active is not None:
            if self._sim_partition_old is None:
                os.environ.pop("GZ_PARTITION", None)
            else:
                os.environ["GZ_PARTITION"] = self._sim_partition_old
            self._sim_partition_active = None
        self._sim_status_var.set("视角已断开")

    def stop_gazebo(self):
        if self._sim_proc is None or self._sim_proc.poll() is not None:
            self._sim_status_var.set("没有由 Studio 启动的 Gazebo 进程")
            return
        try:
            os.killpg(os.getpgid(self._sim_proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._sim_proc.terminate()
            except OSError:
                pass
        self._sim_status_var.set("已请求停止 Studio 启动的 Gazebo")

    def _show_camera_frame(self, key, msg):
        try:
            w, h, nch, png = save_frame.png_bytes(msg)
            label = {"fpv": self.fpv_label, "chase": self.chase_label,
                     "top": self.topcam_label}[key]
            if Image is not None and ImageTk is not None:
                image = Image.open(io.BytesIO(png)).convert("RGB")
                label.update_idletasks()
                max_w = max(320, label.winfo_width() - 12)
                max_h = max(220, label.winfo_height() - 12)
                scale = min(1.0, max_w / max(1, w), max_h / max(1, h))
                if scale < 0.999:
                    resampling = getattr(Image, "Resampling", Image)
                    image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                         getattr(resampling, "LANCZOS", 1))
                photo = ImageTk.PhotoImage(image)
            else:
                photo = tk.PhotoImage(data=base64.b64encode(png).decode("ascii"), format="png")
            label.configure(image=photo, text="")
            self._sim_photos[key] = photo
            self._sim_frames_seen[key] += 1
        except Exception as exc:
            label = {"fpv": self.fpv_label, "chase": self.chase_label,
                     "top": self.topcam_label}[key]
            label.configure(text=f"图像解码失败\n{exc}", image="")
            self._sim_photos.pop(key, None)

    def _sim_tick(self):
        try:
            if self._sim_subs:
                for key in ("fpv", "chase", "top"):
                    sub = self._sim_subs.get(key)
                    if not sub:
                        continue
                    frames = sub.frames()
                    if frames:
                        seq, msg = frames[-1]
                        if seq != self._sim_last_seq[key]:
                            self._sim_last_seq[key] = seq
                            self._show_camera_frame(key, msg)
                pose_msg = self._sim_subs["pose"].latest()
                pose = parse_pose_msg(pose_msg) if pose_msg else None
                if pose:
                    x, y, z, yaw, sim_ns = pose
                    self._sim_pose = pose
                    if not self._sim_path or math.hypot(x - self._sim_path[-1][0], y - self._sim_path[-1][1]) > 0.005:
                        self._sim_path.append((x, y))
                    self.sim_info_var.set(
                        f"Position  x={x:.2f} m  y={y:.2f} m  z={z:.2f} m    "
                        f"Yaw={math.degrees(yaw):.1f}°    Sim time={sim_ns / 1e9:.2f} s    "
                        f"Path points={len(self._sim_path)}")
                    self._draw_topdown()
                statuses = []
                for key, topic in (("fpv", CAMERA_TOPIC), ("chase", CHASE_TOPIC),
                                   ("top", TOP_TOPIC), ("pose", POSE_TOPIC)):
                    sub = self._sim_subs.get(key)
                    statuses.append(f"{key}:{'OK' if sub and sub.healthy() and sub.frames() else 'WAIT'}")
                self._sim_status_var.set("Gazebo " + " · ".join(statuses))
            if self._live_proc is not None:
                rc = self._live_proc.poll()
                self._update_live_action()
                if rc is not None:
                    self._finish_live_demo(rc)
            if self._scene_proc is not None:
                scene_rc = self._scene_proc.poll()
                if scene_rc is not None:
                    self._finish_scene_generation(scene_rc)
            if self._sim_proc is not None and self._sim_proc.poll() is not None:
                self._sim_status_var.set(f"Gazebo 已退出 rc={self._sim_proc.returncode}")
                self._sim_proc = None
                self._sim_world_path = None
                self._sim_world_sha256 = None
                if self._sim_log_handle:
                    self._sim_log_handle.close()
                    self._sim_log_handle = None
        except Exception:
            traceback.print_exc()
        finally:
            if not self._closing:
                self.root.after(80, self._sim_tick)

    def _draw_topdown(self):
        if not hasattr(self, "top_canvas"):
            return
        c = self.top_canvas
        c.delete("all")
        w, h = max(1, c.winfo_width()), max(1, c.winfo_height())
        margin = 34
        xs = [o["x"] for o in self._sim_obstacles]
        ys = [o["y"] for o in self._sim_obstacles]
        if self._sim_path:
            xs += [p[0] for p in self._sim_path]
            ys += [p[1] for p in self._sim_path]
        if self._sim_waypoint:
            xs.append(self._sim_waypoint[0]); ys.append(self._sim_waypoint[1])
        if self._sim_pose:
            xs.append(self._sim_pose[0]); ys.append(self._sim_pose[1])
        if not xs:
            xs, ys = [-1, 8], [-4, 4]
        xmin, xmax = min(xs) - 1.5, max(xs) + 1.5
        ymin, ymax = min(ys) - 1.5, max(ys) + 1.5
        sx = (w - 2 * margin) / max(1.0, xmax - xmin)
        sy = (h - 2 * margin) / max(1.0, ymax - ymin)
        scale = min(sx, sy)
        cx = (w - (xmax - xmin) * scale) / 2
        cy = (h - (ymax - ymin) * scale) / 2
        def screen(x, y):
            return cx + (x - xmin) * scale, h - cy - (y - ymin) * scale
        # meter grid
        gx0, gx1 = math.floor(xmin), math.ceil(xmax)
        gy0, gy1 = math.floor(ymin), math.ceil(ymax)
        for gx in range(gx0, gx1 + 1):
            x0, y0 = screen(gx, ymin)
            x1, y1 = screen(gx, ymax)
            c.create_line(x0, y0, x1, y1, fill="#1E293B")
            c.create_text(x0, h - 12, text=str(gx), fill="#64748B", font=("TkDefaultFont", 8))
        for gy in range(gy0, gy1 + 1):
            x0, y0 = screen(xmin, gy)
            x1, y1 = screen(xmax, gy)
            c.create_line(x0, y0, x1, y1, fill="#1E293B")
            c.create_text(15, y0, text=str(gy), fill="#64748B", font=("TkDefaultFont", 8))
        # obstacles from SDF collision geometry
        for o in self._sim_obstacles:
            if o["kind"] == "box":
                halfx, halfy = o["sx"] / 2, o["sy"] / 2
                yaw = o.get("yaw", 0.0)
                corners = []
                for px, py in ((-halfx, -halfy), (halfx, -halfy),
                               (halfx, halfy), (-halfx, halfy)):
                    wx = o["x"] + math.cos(yaw) * px - math.sin(yaw) * py
                    wy = o["y"] + math.sin(yaw) * px + math.cos(yaw) * py
                    corners.extend(screen(wx, wy))
                color = "#9A6B3F" if "tree" in o["name"].lower() else "#475569"
                if "target" in o["name"].lower() or "landmark" in o["name"].lower():
                    color = "#B45309"
                c.create_polygon(*corners, fill=color, outline="#CBD5E1", width=1)
            else:
                x0, y0 = screen(o["x"] - o["r"], o["y"] - o["r"])
                x1, y1 = screen(o["x"] + o["r"], o["y"] + o["r"])
                c.create_oval(x0, y1, x1, y0, fill="#64748B", outline="#CBD5E1")
        if self._sim_waypoint:
            x, y = screen(*self._sim_waypoint)
            c.create_oval(x - 7, y - 7, x + 7, y + 7, outline="#FBBF24", width=2)
            c.create_text(x + 10, y - 10, text="WAYPOINT", fill="#FBBF24", anchor="w")
        if len(self._sim_path) > 1:
            pts = [v for p in self._sim_path for v in screen(*p)]
            c.create_line(*pts, fill="#38BDF8", width=3, smooth=True)
            sx0, sy0 = screen(*self._sim_path[0])
            c.create_oval(sx0 - 4, sy0 - 4, sx0 + 4, sy0 + 4, fill="#34D399", outline="")
        if self._sim_pose:
            x, y, z, yaw, _ = self._sim_pose
            px, py = screen(x, y)
            length = max(18, min(35, scale * 0.55))
            ex, ey = px + math.cos(yaw) * length, py - math.sin(yaw) * length
            c.create_oval(px - 8, py - 8, px + 8, py + 8, fill="#2563EB", outline="#BFDBFE", width=2)
            c.create_line(px, py, ex, ey, fill="#F8FAFC", width=3, arrow="last", arrowshape=(10, 12, 5))
            c.create_text(px + 12, py + 12, text=f"UAV z={z:.1f}m", fill="#93C5FD", anchor="nw")
        c.create_text(w - 12, 12, text="TOP VIEW · metres", fill="#94A3B8", anchor="ne")

    # --------------------------------------------------------------- live student
    def start_live_student(self):
        if self._live_proc is not None:
            return
        if not self._sim_subs:
            messagebox.showerror("尚未连接 Gazebo", "请先启动 Gazebo 或连接已有仿真。")
            return
        ckpt = Path(self.ckpt_var.get()).expanduser().resolve()
        meta_path = Path(self.meta_var.get()).expanduser().resolve()
        if not ckpt.is_file():
            messagebox.showerror("Checkpoint 不存在", "请选择真实的 ckpt_best.bin。")
            return
        if not meta_path.is_file():
            messagebox.showerror("缺少场景元数据", "学生策略演示需要 scene metadata，以便记录指令、目标和场景来源。")
            return
        try:
            steps = int(self.live_steps_var.get())
            hold = float(self.live_hold_var.get())
            if steps < 1 or not math.isfinite(hold) or hold <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "Steps 必须为正整数，Hold 必须为有限正数。")
            return
        meta = load_scene_meta(meta_path)
        split = scene_split(meta)
        world_path = Path(self.world_var.get()).expanduser().resolve()
        if not world_path.is_file():
            messagebox.showerror("World 不存在", f"找不到当前选择的 world SDF：\n{world_path}")
            return
        if self._sim_proc is not None and self._sim_proc.poll() is None and self._sim_world_path:
            if world_path != self._sim_world_path or sha256_file(world_path) != self._sim_world_sha256:
                messagebox.showerror("当前仿真 World 不匹配",
                                     "Studio 启动的 Gazebo 正在运行另一个 SDF。请停止 Gazebo，再用所选 world 重新启动。")
                return
        elif self._sim_proc is None:
            if not messagebox.askyesno("外部仿真 World 无法核验",
                                       "当前连接的是外部 Gazebo，Studio 无法确认它是否由所选 SDF 启动。\n\n请确认当前运行场景与所选 world / metadata 一致。仍要运行学生策略吗？"):
                return
        expected_world_sha = meta.get("world_sha256")
        if expected_world_sha:
            actual_world_sha = sha256_file(world_path)
            if actual_world_sha != expected_world_sha:
                messagebox.showerror("World / metadata 不匹配",
                                     "所选 world SDF 的 SHA256 与 scene metadata 中的 world_sha256 不一致。\n\n请选同一 seed 生成的 .sdf 和同名 .json，避免在错误场景上运行学生策略。")
                return
        out_root = self.run_dir if self.run_dir else (DATA_ROOT / "uav_rgb" / "live_demos")
        out_root.mkdir(parents=True, exist_ok=True)
        out = out_root / ("live_demo_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4])
        out.mkdir(parents=True, exist_ok=False)
        side = Path(str(out) + ".monitor")
        side.mkdir(parents=True, exist_ok=True)
        log = (side / "live_demo.log").open("ab", buffering=0)
        cmd = [PYTHON_FOR_SIM, str(SIM_TOOLS / "collect.py"), "--policy", "student",
               "--student-ckpt", str(ckpt), "--steps", str(steps), "--hold", str(hold),
               "--out", str(out), "--scene-meta", str(meta_path), "--split", split,
               "--stop-on-collision"]
        if meta.get("scene_id") is not None:
            cmd += ["--scene-id", str(meta["scene_id"])]
        if meta.get("seed") is not None:
            cmd += ["--seed", str(meta["seed"])]
        env = os.environ.copy()
        if self._sim_partition_active and self._sim_partition_active != "default":
            env["GZ_PARTITION"] = self._sim_partition_active
        self._live_out, self._live_log = out, log
        self._live_config = {"checkpoint": str(ckpt), "scene_meta": str(meta_path),
                             "world": str(Path(self.world_var.get()).expanduser().resolve()),
                             "steps": steps, "hold_s": hold, "split": split,
                             "partition": self._sim_partition_active}
        write_json_atomic(side / "job.json", self._live_config)
        self._live_action_offset = 0
        self._live_last_action = None
        self.live_action_var.set("动作：等待 StudentPolicy 初始化…")
        try:
            self._live_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                               stdin=subprocess.DEVNULL, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.close()
            self._live_log = None
            messagebox.showerror("学生策略启动失败", str(exc))
            return
        self.live_btn.configure(state="disabled")
        self.live_stop_btn.configure(state="normal")
        self._sim_status_var.set("学生策略正在驱动 Gazebo · 三视角实时更新")

    def _update_live_action(self):
        if not self._live_out:
            return
        path = self._live_out / "actions.jsonl"
        try:
            with path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 8192))
                tail = f.read().decode("utf-8", "replace").splitlines()
            if tail:
                item = json.loads(tail[-1])
                action = item.get("action_executed", item.get("action"))
                if action is not None:
                    idx = int(action)
                    name = ACTIONS[idx] if 0 <= idx < len(ACTIONS) else str(idx)
                    self._live_last_action = name
                    self.live_action_var.set(f"当前学生动作：{name} · step={item.get('step', '—')} · collision={item.get('collision', False)}")
        except (OSError, ValueError, TypeError, IndexError):
            pass

    def stop_live_student(self):
        if self._live_proc is None:
            return
        try:
            os.killpg(os.getpgid(self._live_proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._live_proc.send_signal(signal.SIGINT)
            except OSError:
                pass
        self.live_action_var.set("正在停止学生策略；采集器会先发布零速指令…")

    def _finish_live_demo(self, rc):
        self._live_proc = None
        if self._live_log:
            self._live_log.close()
            self._live_log = None
        if self._live_out:
            manifest = {}
            try:
                manifest = json.loads((self._live_out / "episode_manifest.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            actions = read_jsonl(self._live_out / "actions.jsonl")
            counts = {name: 0 for name in ACTIONS}
            collisions = 0
            for a in actions:
                idx = a.get("action_executed")
                if isinstance(idx, int) and 0 <= idx < len(ACTIONS):
                    counts[ACTIONS[idx]] += 1
                collisions += int(bool(a.get("collision")))
            summary = {"schema": "uavrgb.live_demo.v1", "return_code": rc,
                       "finished_at": now_iso(), "steps": len(actions),
                       "collision_steps": collisions, "action_counts": counts,
                       "config": self._live_config or {}, "manifest": manifest,
                       "interpretation": "single-episode live demonstration; not a formal navigation success-rate evaluation"}
            write_json_atomic(self._live_out / "live_demo_summary.json", summary)
            if self.run_dir and self._live_out.parent == self.run_dir:
                summary_path = self.run_dir / "studio_summary.json"
                try:
                    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    run_summary = summarize_run(self.run_dir, self.run_meta or {"run_id": self.run_dir.name},
                                                0, "finished")
                run_summary.setdefault("live_demos", []).append({
                    "output_dir": str(self._live_out), "steps": len(actions),
                    "collision_steps": collisions, "action_counts": counts,
                    "return_code": rc, "finished_at": summary["finished_at"]})
                write_json_atomic(summary_path, run_summary)
                write_experiment_report(self.run_dir, run_summary)
                self.refresh_history()
            self.live_action_var.set(f"学生策略演示结束 rc={rc} · steps={len(actions)} · collision_steps={collisions} · 结果：{self._live_out}")
        self.live_btn.configure(state="normal")
        self.live_stop_btn.configure(state="disabled")

    # --------------------------------------------------------------- evaluations
    def start_closed_loop_eval(self):
        if self.eval_proc is not None:
            return
        if (self._sim_proc is not None and self._sim_proc.poll() is None) or self._sim_subs:
            proceed = messagebox.askyesno(
                "闭环评测会启动独立仿真",
                "正式闭环评测会启动独立的 headless Gazebo 实例，并与当前桌面仿真共享 GPU。\n\n仍要继续吗？")
            if not proceed:
                return
        if not self.run_dir:
            messagebox.showerror("没有实验", "请先加载一个训练 run。")
            return
        ckpt = self.run_dir / "ckpt_best.bin"
        if not ckpt.is_file():
            messagebox.showerror("缺少 best checkpoint", f"找不到：{ckpt}")
            return
        try:
            episodes = int(self.eval_episodes_var.get())
            if episodes < 1 or episodes > 500:
                raise ValueError
        except ValueError:
            messagebox.showerror("参数错误", "Episodes 必须在 1–500 之间。")
            return
        task = self.eval_task_var.get()
        split = self.eval_split_var.get()
        out = self.run_dir / ("closed_loop_eval_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        out.mkdir(parents=True, exist_ok=False)
        side = Path(str(out) + ".monitor")
        side.mkdir(parents=True, exist_ok=True)
        log = (side / "eval.log").open("ab", buffering=0)
        cmd = [PYTHON_FOR_SIM, str(SIM_TOOLS / "eval_closed_loop.py"), "--policy", "student",
               "--ckpt", str(ckpt), "--episodes", str(episodes), "--task", task,
               "--split", split, "--out", str(out), "--text-mode", "keep"]
        if task == "campus":
            cmd += ["--appearance", self.eval_appearance_var.get()]
        env = os.environ.copy()
        # The evaluator owns its own isolated GZ_PARTITIONs; do not inherit the GUI demo partition.
        env.pop("GZ_PARTITION", None)
        self.eval_out, self.eval_log = out, log
        try:
            self.eval_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                              stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.close()
            self.eval_log = None
            messagebox.showerror("评测启动失败", str(exc))
            return
        self.eval_btn.configure(state="disabled")
        self.eval_stop_btn.configure(state="normal")
        self._eval_status_var.set(f"闭环评测运行中 · {out.name}")

    def stop_closed_loop_eval(self):
        if self.eval_proc is None:
            return
        try:
            os.killpg(os.getpgid(self.eval_proc.pid), signal.SIGINT)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self.eval_proc.send_signal(signal.SIGINT)
            except OSError:
                pass
        self._eval_status_var.set("正在请求停止闭环评测…")

    def _attach_eval_to_run(self, report, return_code=None):
        if not self.run_dir or not report:
            return
        summary_path = self.run_dir / "studio_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            summary = summarize_run(self.run_dir, self.run_meta or {"run_id": self.run_dir.name},
                                    0, "finished")
        entries = summary.setdefault("closed_loop_evaluations", [])
        entry = {"task": report.get("task"), "split": report.get("split"),
                 "success_rate": report.get("success_rate"),
                 "collision_rate": report.get("collision_rate"),
                 "stuck_rate": report.get("stuck_rate"),
                 "false_forward_rate": report.get("false_forward_rate"),
                 "episodes_done": report.get("episodes_done"),
                 "episodes_requested": report.get("episodes_requested"),
                 "unique_maps": report.get("unique_maps", report.get("unique_geometry")),
                 "success_criterion": report.get("success_criterion"),
                 "eval_json": str(self.eval_out / "eval.json") if self.eval_out else None,
                 "return_code": return_code, "finished_at": now_iso()}
        entries.append(entry)
        write_json_atomic(summary_path, summary)
        write_experiment_report(self.run_dir, summary)
        self.refresh_history()
        self.refresh_run(force=True)

    def open_eval_dir(self):
        if self.eval_out and self.eval_out.exists():
            self._open_path(self.eval_out)
        elif self.run_dir:
            candidates = sorted(self.run_dir.glob("closed_loop_eval_*"), key=lambda p: p.stat().st_mtime, reverse=True)
            if candidates:
                self._open_path(candidates[0])

    # ---------------------------------------------------------------- experiments
    @staticmethod
    def _run_pid_alive(pid, run_dir):
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        # Linux: avoid mistaking a reused PID for this run's trainer process.
        cmdline = Path(f"/proc/{pid}/cmdline")
        try:
            raw = cmdline.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            config_path = str(Path(str(run_dir) + ".monitor") / "train_config.json")
            return (("uavrgb_train" in raw and str(run_dir) in raw) or
                    ("trainer_native.py" in raw and config_path in raw))
        except OSError:
            return True

    def _recover_detached_runs(self):
        """Finalize runs whose desktop window was closed while the child kept running."""
        if not RUNS_ROOT.is_dir():
            return
        for run_dir in RUNS_ROOT.iterdir():
            if not run_dir.is_dir() or run_dir.name == "simulation_logs":
                continue
            summary_path = run_dir / "studio_summary.json"
            if summary_path.exists():
                continue
            sidecar = Path(str(run_dir) + ".monitor")
            life_path = sidecar / "lifecycle.json"
            job_path = sidecar / "job.json"
            try:
                life = json.loads(life_path.read_text(encoding="utf-8"))
                job = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if life.get("status") not in ("running", "starting"):
                continue
            if self._run_pid_alive(life.get("pid") or job.get("pid"), run_dir):
                continue
            events = read_jsonl(run_dir / "metrics.jsonl")
            done = any(e.get("event") == "status" and
                       str(e.get("message", "")).lower().startswith("done:") for e in events)
            state = "finished_unobserved" if done else "incomplete_unobserved"
            summary = summarize_run(run_dir, job, None, state)
            summary["notes"].append("The desktop monitor was closed while the child process ran; process exit code was not observed by Studio.")
            write_json_atomic(summary_path, summary)
            write_experiment_report(run_dir, summary)
            write_json_atomic(life_path, {"status": state, "return_code": None,
                                          "finished_at": summary["finished_at"],
                                          "recovered_by_studio": True})

    def refresh_history(self):
        if not hasattr(self, "history_tree"):
            return
        self.history_tree.delete(*self.history_tree.get_children())
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self._recover_detached_runs()
        run_dirs = sorted((p for p in RUNS_ROOT.iterdir() if p.is_dir() and p.name != "simulation_logs"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        for run_dir in run_dirs[:300]:
            summary_path = run_dir / "studio_summary.json"
            try:
                if summary_path.exists():
                    s = json.loads(summary_path.read_text(encoding="utf-8"))
                else:
                    sidecar = Path(str(run_dir) + ".monitor")
                    life = json.loads((sidecar / "lifecycle.json").read_text(encoding="utf-8"))
                    job = json.loads((sidecar / "job.json").read_text(encoding="utf-8"))
                    s = {"run_id": run_dir.name, "status": "running_detached",
                         "started_at": job.get("started_at"), "metrics": {
                             "epochs_completed": len(read_training_rows(run_dir)),
                             "test_macro_f1": None, "test_balanced_accuracy": None}}
                    if not self._run_pid_alive(life.get("pid") or job.get("pid"), run_dir):
                        continue
            except (OSError, ValueError, KeyError):
                continue
            m = s.get("metrics", {})
            evals = s.get("closed_loop_evaluations", [])
            latest_sr = evals[-1].get("success_rate") if evals else None
            self.history_tree.insert("", "end", iid=str(run_dir),
                                     text=s.get("run_id", run_dir.name),
                                     values=(s.get("status", "—"), m.get("epochs_completed", "—"),
                                             m.get("test_macro_f1") if m.get("test_macro_f1") is not None else "—",
                                             m.get("test_balanced_accuracy") if m.get("test_balanced_accuracy") is not None else "—",
                                             latest_sr if latest_sr is not None else "—",
                                             s.get("started_at", "—")))

    def load_selected_history(self):
        sel = self.history_tree.selection()
        if sel:
            self.load_run(Path(sel[0]))
            self.tabs.select(self.dashboard_tab)

    def open_selected_history(self):
        sel = self.history_tree.selection()
        if sel:
            self._open_path(Path(sel[0]))

    def open_report(self):
        if self.run_dir:
            path = self.run_dir / "experiment_report.md"
            if path.exists():
                self._open_path(path)
            else:
                messagebox.showinfo("报告不存在", "当前实验尚未生成 experiment_report.md。")

    def choose_existing_run(self):
        path = filedialog.askdirectory(title="选择已有训练 run 目录")
        if path:
            self.load_run(Path(path))
            self.tabs.select(self.dashboard_tab)

    def load_run(self, path: Path):
        path = path.expanduser().resolve()
        if self.proc is not None and self.proc.poll() is None and path == self.run_dir:
            self.refresh_run(force=True)
            return
        if self._detached_pid is not None and path != self.run_dir and self._run_pid_alive(self._detached_pid, self.run_dir or Path(".")):
            messagebox.showwarning("后台训练仍在运行", "当前后台训练仍在运行，请先停止或等待完成，再切换到其他实验。")
            return
        if self.proc is not None and self.proc.poll() is None and path != self.run_dir:
            messagebox.showwarning("训练正在运行", "当前训练尚未结束，不能切换到另一个 run 目录。")
            return
        if not path.is_dir():
            messagebox.showerror("目录不存在", str(path))
            return
        self.run_dir = path
        self.sidecar = Path(str(path) + ".monitor")
        self.run_title_var.set(f"当前实验：{path.name}")
        self._metric_sig = None
        self._event_tail = {}
        self._events.clear()
        self._latest_train_event = None
        self._train_log_state = {}
        self._train_log_lines = 0
        self.train_log_text.delete("1.0", "end")
        self.refresh_run(force=True)
        best = path / "ckpt_best.bin"
        if best.is_file():
            self.ckpt_var.set(str(best))
        lifecycle = {}
        try:
            lifecycle = json.loads((self.sidecar / "lifecycle.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            pass
        pid = lifecycle.get("pid")
        if lifecycle.get("status") == "running" and self._run_pid_alive(pid, path):
            self._detached_pid = pid
            self.start_train_btn.configure(state="disabled")
            self.stop_detached_btn.configure(state="normal")
            self._training_status_var.set(f"后台训练仍在运行（只读监控）· PID {pid}")
        else:
            self._detached_pid = None
            self.start_train_btn.configure(state="normal")
            self.stop_detached_btn.configure(state="disabled")
            self._training_status_var.set(f"已加载实验 · {path}")

    def open_data_root(self):
        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        self._open_path(DATA_ROOT / "uav_rgb")

    def _open_path(self, path: Path):
        try:
            if sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, start_new_session=True)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                os.startfile(str(path))  # type: ignore[attr-defined]
        except OSError as exc:
            messagebox.showerror("无法打开", str(exc))

    # ---------------------------------------------------------------- lifecycle
    def _tick(self):
        try:
            if self.proc is not None:
                rc = self.proc.poll()
                if rc is not None:
                    self._finish_training(rc)
                else:
                    self.refresh_run()
                    self._refresh_train_log()
                    elapsed = time.monotonic() - (self._train_started_mono or time.monotonic())
                    step = self._latest_train_event or {}
                    progress = (f" · step {step.get('step')}/{step.get('total_steps')}"
                                if step.get("step") is not None else "")
                    eta = safe_float(step.get("eta_sec"))
                    eta_text = f" · ETA {eta / 60:.1f} min" if eta is not None and eta > 0 else ""
                    self._training_status_var.set(f"真实训练中 · PID {self.proc.pid} · {elapsed / 60:.1f} min{progress}{eta_text}")
            elif self.run_dir:
                self.refresh_run()
                self._refresh_train_log()
                summary_path = self.run_dir / "studio_summary.json"
                if not summary_path.exists():
                    self._recover_detached_runs()
                    if summary_path.exists():
                        self.refresh_history()
                        self._detached_pid = None
                        self.start_train_btn.configure(state="normal")
                        self.stop_detached_btn.configure(state="disabled")
                        try:
                            recovered = json.loads(summary_path.read_text(encoding="utf-8"))
                            self._training_status_var.set(f"后台实验已结束 · {recovered.get('status')}")
                        except (OSError, ValueError):
                            pass
            if self.eval_proc is not None:
                rc = self.eval_proc.poll()
                if rc is not None:
                    self.eval_proc = None
                    if self.eval_log:
                        self.eval_log.close()
                        self.eval_log = None
                    self.eval_btn.configure(state="normal")
                    self.eval_stop_btn.configure(state="disabled")
                    report = None
                    if self.eval_out and (self.eval_out / "eval.json").is_file():
                        try:
                            report = json.loads((self.eval_out / "eval.json").read_text(encoding="utf-8"))
                        except (OSError, ValueError):
                            pass
                    if report:
                        self._attach_eval_to_run(report, rc)
                        self._eval_status_var.set(
                            f"闭环评测结束 rc={rc} · SR={report.get('success_rate')} · collision={report.get('collision_rate')} · "
                            f"episodes={report.get('episodes_done')}/{report.get('episodes_requested')} · "
                            f"completion={report.get('completion_rate')} · unique_maps={report.get('unique_maps', report.get('unique_geometry'))}")
                    else:
                        self._eval_status_var.set(f"闭环评测结束 rc={rc} · 请检查 {self.eval_out}")
            if self._live_proc is not None:
                self._update_live_action()
        except Exception:
            traceback.print_exc()
        finally:
            if not self._closing:
                self.root.after(self.refresh_ms, self._tick)

    def _on_close(self):
        if self._closing:
            return
        active = []
        if self.proc is not None:
            active.append("训练")
        elif self._detached_pid is not None and self.run_dir and self._run_pid_alive(self._detached_pid, self.run_dir):
            active.append("后台训练")
        if self.eval_proc is not None:
            active.append("闭环评测")
        if self._live_proc is not None:
            active.append("学生策略演示")
        if self._scene_proc is not None:
            active.append("场景生成")
        if active:
            stop = messagebox.askyesno("仍有任务运行", "当前仍在运行：" + "、".join(active) +
                                       "。\n\n选择“是”会发送安全停止信号；选择“否”则关闭界面但让任务继续运行并保留日志。")
            if stop:
                for proc in (self.proc, self.eval_proc, self._live_proc, self._scene_proc):
                    if proc is not None and proc.poll() is None:
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                if self._detached_pid is not None:
                    try:
                        os.killpg(self._detached_pid, signal.SIGINT)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
        if self._sim_subs:
            self.disconnect_sim()
        if self._sim_proc is not None and self._sim_proc.poll() is None:
            stop_sim = messagebox.askyesno("Gazebo 仍在运行", "是否停止由 Studio 启动的 Gazebo？")
            if stop_sim:
                self.stop_gazebo()
        self._closing = True
        self.root.destroy()


def main():
    p = argparse.ArgumentParser(description="UAV Navigation Studio desktop app")
    p.add_argument("--run-dir", default="", help="可选：启动时加载已有训练 run")
    p.add_argument("--refresh", type=float, default=1.0, help="监控刷新间隔（0.3–60 秒）")
    args = p.parse_args()
    if not math.isfinite(args.refresh) or not 0.3 <= args.refresh <= 60:
        p.error("--refresh 必须是 0.3–60 的有限秒数")
    if tk is None:
        raise SystemExit("UAV Navigation Studio 需要 tkinter；Ubuntu 可安装 python3-tk")
    root = tk.Tk()
    StudioApp(root, args.run_dir or None, int(args.refresh * 1000))
    root.mainloop()


if __name__ == "__main__":
    main()
