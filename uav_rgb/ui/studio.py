#!/usr/bin/env python3
"""UAV Navigation Studio: local desktop training, simulation and experiment workbench.

The UI reads the real uavrgb_train artifacts (results.csv / metrics.jsonl), launches
trainer_native.py, and subscribes to Gazebo camera/pose topics. No demo metrics are
shown unless the user explicitly selects the legacy demo trainer.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
import xml.etree.ElementTree as ET
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, font as tkfont, messagebox, ttk
except ImportError:  # pragma: no cover - headless CI
    tk = ttk = filedialog = messagebox = tkfont = None

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
# 载具模型名从世界 SDF 推断（含 <sensor name="camera"> 的 model：x3 或 e1_mini）；推断不到退回 x3。
# 相机话题不随载具变，位姿话题是 /model/<载具>/pose。
DEFAULT_VEHICLE = "x3"

# ---- 用户会改的默认值集中在这里（改数值即可，调用链不用动）----
# 刷新周期
DASHBOARD_REFRESH_MS = 1000  # 仪表盘/训练事件轮询周期（--refresh-ms 可覆盖；下限 300，更快只会多读文件）
SIM_TICK_MS = 80             # 仿真页取帧/位姿周期：机载相机 30 Hz 不漏帧够用，再快主线程画不过来
HISTORY_POLL_MS = 50         # 等后台扫描线程结果的轮询间隔（只影响“历史”刷新出现的延迟）
SIM_CONNECT_DELAY_MS = 3500  # 启动 Gazebo 后多久订阅话题：server 起来约 2–3 s，早了订阅不到
SIM_VIEW_PERIOD_S = 0.1      # chase/顶视相机最多 10 Hz 刷新（15 Hz 的 chase 每帧都缩放会拖慢主线程）
SIM_STALE_S = 1.0            # 相机/位姿 header stamp 落后当前仿真时间超过此值即标 STALE
SIM_STOP_GRACE_S = 5.0       # 停止 Gazebo：SIGINT→SIGTERM→SIGKILL 每级宽限
SIM_CLOSE_GRACE_S = 2.0      # 关窗/收到信号时的每级宽限（窗口已没了，不值得多等）
# 评测 / 演示默认值
EVAL_EPISODES_DEFAULT = "20"  # 手动闭环评测集数：20 集 Wilson 区间半宽约 ±0.2，再少没统计意义
AUTO_EVAL_EPISODES = 1        # 训练后自动录像只跑 1 集：是演示录像不是成功率（仪表盘按“单集录像”显示）
LIVE_STEPS_DEFAULT = "80"     # 实时演示步数，与 eval_closed_loop 的 --steps 默认一致
LIVE_HOLD_DEFAULT = "1.0"     # 实时演示 Hold 秒；选中权重后会按其训练数据的 hold_s 覆盖
# Gazebo 渲染后端（启动 Gazebo 前可在仿真页改；进程环境里已有 UAVRGB_RENDER_BACKEND 时以它为准）。
# 本容器 gpu 后端（EGL）段错误，只能 software；换了机器想用 GPU 就把这里改成 "gpu"。
DEFAULT_RENDER_BACKEND = "software"
RENDER_BACKENDS = ("software", "gpu")
# 录像帧率不在这里：由 sim/tools/record_views.py 的 FPS 决定（评测录像），3DGS 录像 fps = 1/hold。
ACTION_NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
ACTIONS = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
# 任务列表与 scene_gen / eval_closed_loop 的 --task 一致；forest_e1 要配 --density（档位同 scene_gen）
# 和 E1 载具（--vehicle e1_mini）。
TASKS = ("base", "l1", "campus", "forest_e1")
DENSITIES = ("sparse", "medium", "dense")
DENSITY_DEFAULT = "sparse"   # 森林密度档：sparse 最窄缝 0.45 m，是 e1_pipeline 的默认档

sys.path.insert(0, str(UI_DIR))
sys.path.insert(0, str(SIM_TOOLS))
sys.path.insert(0, str(REPO_ROOT / "tools"))
import monitor as monitor_io  # noqa: E402
import gz_io  # noqa: E402
import save_frame  # noqa: E402
import collect_parallel  # noqa: E402  （只用 DYNAMICS_VEHICLE / gz_env，顶层只有标准库）
# 复用 CLI 包装的 .uavr meta 推断，避免训练后自动评测与独立 CLI 两处规则漂移。
import train_student_with_video as video_cli  # noqa: E402


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


_CKPT_SHA_CACHE = {}


def sha256_file_cached(path: Path):
    """ckpt 可能几十 MB，Dashboard 每秒刷新不能反复整文件哈希。"""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key not in _CKPT_SHA_CACHE:
        if len(_CKPT_SHA_CACHE) > 16:
            _CKPT_SHA_CACHE.clear()
        _CKPT_SHA_CACHE[key] = sha256_file(path)
    return _CKPT_SHA_CACHE[key]


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


def pose_topic(vehicle):
    return f"/model/{vehicle}/pose"


def _vehicle_from_root(root):
    """载具 = 第一个带 <sensor name="camera"> 的 model（机载相机只装在载具上）；没有返回 None。"""
    for model in root.iter("model"):
        if any(s.get("name") == "camera" for s in model.iter("sensor")):
            return model.get("name") or None
    return None


def sdf_root(world_path):
    """解析 SDF 返回根元素；读不了/不是 XML 返回 None。

    uav_rgb_e1.sdf 的注释里有 "--camera-rate" 这样的 "--"：Gazebo（tinyxml2）照常加载，
    Python expat 按 XML 规范报 not well-formed。严格解析失败就去掉注释再试一次，
    不然 e1 世界会被当成 x3，地图也画不出障碍物。
    """
    try:
        return ET.parse(str(world_path)).getroot()
    except OSError:
        return None
    except ET.ParseError:
        pass
    try:
        text = Path(world_path).read_text(encoding="utf-8", errors="replace")
        return ET.fromstring(re.sub(r"<!--.*?-->", "", text, flags=re.S))
    except (OSError, ET.ParseError):
        return None


def vehicle_from_sdf(world_path):
    """从世界 SDF 推断载具模型名；文件读不了/没有机载相机时退回 x3。"""
    if not world_path:
        return DEFAULT_VEHICLE
    root = sdf_root(world_path)
    if root is None:
        return DEFAULT_VEHICLE
    return _vehicle_from_root(root) or DEFAULT_VEHICLE


def parse_pose_msg(msg, vehicle=DEFAULT_VEHICLE):
    """Return (x,y,z,yaw_rad,sim_time_ns) from gz.msgs.Pose_V / Pose JSON."""
    item = msg
    arr = msg.get("pose") if isinstance(msg, dict) else None
    if isinstance(arr, list) and arr:
        # Pose_V 里优先取载具自己那条（模型名 / model/<名> / <名>::base_link），没有就取第一条。
        names = tuple(n.lower() for n in (vehicle, f"model/{vehicle}", f"{vehicle}::base_link"))
        item = next((p for p in arr if str(p.get("name", "")).lower() in names), arr[0])
    if not isinstance(item, dict):
        return None
    pos = item.get("position") or {}
    q = item.get("orientation") or {}
    try:
        # protobuf 转 JSON 省略值为 0 的字段：出生点 y=0 时消息里就没有 "y"。
        x, y, z = (float(pos.get(k, 0.0)) for k in ("x", "y", "z"))
        qx, qy, qz, qw = (float(q.get(k, 0.0)) for k in ("x", "y", "z", "w"))
    except (AttributeError, TypeError, ValueError):
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
    root = sdf_root(world_path)
    if root is None:
        return []
    world = root.find(".//world")
    if world is None:
        return []
    # 载具自己不是障碍物：以前只排除名为 x3 的模型，换成 e1_mini 后机体会被画成障碍。
    vehicle = _vehicle_from_root(world) or DEFAULT_VEHICLE
    result = []
    for model in world.findall("model"):
        name = model.get("name", "model")
        if name in (vehicle, "ground_plane") or name.startswith("goal_mark"):
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


VIDEO_SUFFIXES = (".mp4", ".webm")  # mp4v 只是兼容旧调用；浏览器能放的是 VP8 .webm
LIVE_STATES = ("running", "starting")


def unobserved_state(run_dir: Path):
    """Studio 没看到退出码的 run：metrics.jsonl 有 done: 就算跑完，否则算未完成。"""
    events = read_jsonl(Path(run_dir) / "metrics.jsonl")
    done = any(e.get("event") == "status" and
               str(e.get("message", "")).lower().startswith("done:") for e in events)
    return "finished_unobserved" if done else "incomplete_unobserved"


def carry_attachments(old_path: Path, summary: dict):
    """把旧摘要里的评测/演示带进新摘要（训练结束、恢复时重写摘要用）。"""
    try:
        old = json.loads(Path(old_path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return summary
    if not isinstance(old, dict):
        return summary
    for key in ("closed_loop_evaluations", "live_demos"):
        if isinstance(old.get(key), list) and old[key]:
            summary[key] = old[key] + summary.get(key, [])
    return summary


def summary_is_final(path: Path):
    """摘要存在且状态不是 running/starting。训练中途挂评测写的摘要不算定稿。"""
    try:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return False
    return isinstance(summary, dict) and summary.get("status") not in LIVE_STATES


def _pid_exists(pid):
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return False  # os.kill(0/-1) 是进程组语义，不能拿来判断单个进程
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


RUN_MARKERS = ("metrics.jsonl", "results.csv", "eval.json", "studio_summary.json")
SKIP_RUN_NAMES = {"simulation_logs"}


def _is_run_dir(path: Path):
    return path.is_dir() and any((path / name).is_file() for name in RUN_MARKERS)


def _skip_name(name):
    """扫描时不进入的目录名（run 发现与 eval.json 遍历共用，规则只写一份）。"""
    return (name in SKIP_RUN_NAMES or name.startswith(".") or name.startswith("ep_")
            or name.endswith(".monitor") or name == "__pycache__")


def _skip_scan_dir(path: Path):
    return _skip_name(path.name)


def discover_runs(roots):
    """两层扫描 $DATA_ROOT/uav_rgb：<group>/<run> 和 <run>。纯函数，不读内容。"""
    found = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        try:
            entries = list(root.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_dir() or _skip_scan_dir(entry):
                continue
            if _is_run_dir(entry):
                found[entry.resolve()] = entry
                continue
            try:
                children = list(entry.iterdir())
            except OSError:
                continue
            for child in children:
                if _is_run_dir(child) and not _skip_scan_dir(child):
                    found[child.resolve()] = child
    return sorted(found.values(), key=lambda p: p.stat().st_mtime, reverse=True)


def _walk_eval_json(root: Path):
    """找 root 下所有 eval.json；遍历时就剪枝，不进 venv / pip 缓存 / episode 目录。

    以前 rglob 先把 3dgs-venv、teacher-venv 和 5000+ 个 episode 目录全走一遍再过滤，
    主线程 1.3 s。episode 目录（含 episode_manifest.json）里从不放 eval.json。
    """
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "eval.json" in filenames:
            found.append(Path(dirpath) / "eval.json")
        if "pyvenv.cfg" in filenames or "episode_manifest.json" in filenames:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not _skip_name(d)]
    return found


def eval_ckpt_state(run_dir: Path, report: dict):
    """eval 的 ckpt_sha256 与它评测的那份权重当前内容是否一致。

    'current' / 'stale' / 'unlabeled'。旧 eval 没有 sha 时不能判定，保留为未标注，
    不拿“路径相同”冒充同一份权重。比对对象是报告里的 ckpt（评末轮 ckpt.bin 就比
    ckpt.bin）；路径失效时找 run 里同名文件（数据盘搬家），报告没写 ckpt 才退回 ckpt_best.bin。
    以前一律比 ckpt_best.bin，评 ckpt.bin 的报告永远被判成旧评测藏起来。
    """
    reported = report.get("ckpt_sha256")
    if not reported:
        return "unlabeled"
    named = report.get("ckpt")
    if named:
        ckpt = Path(named)
        if not ckpt.is_file():
            ckpt = Path(run_dir) / ckpt.name
    else:
        ckpt = Path(run_dir) / "ckpt_best.bin"
    if not ckpt.is_file():
        return "stale"
    try:
        return "current" if sha256_file_cached(ckpt) == str(reported) else "stale"
    except OSError:
        return "stale"


def eval_display_fields(report: dict):
    """实验表展示字段；旧 eval 缺字段时写“未标注”，不从别的字段推断。"""
    def text(key):
        value = report.get(key)
        return str(value) if value not in (None, "") else "未标注"
    dynamics = report.get("dynamics") or report.get("dynamics_profile")
    return {"task": text("task"), "criterion": text("success_criterion"),
            "window_from": text("window_from"),
            "dynamics": str(dynamics) if dynamics not in (None, "") else "未标注"}


def eval_rate_context(report: dict):
    """卡片副标题：判据与三个分母/失败数；缺失字段写“未标注”，不推断。"""
    def shown(key):
        value = safe_float(report.get(key))
        return f"{value:.3f}" if value is not None else "未标注"
    done = report.get("episodes_done")
    if done is None and isinstance(report.get("per_episode"), list):
        done = len(report["per_episode"])
    requested = report.get("episodes_requested")
    failed = report.get("failed_count")
    return (f"criterion={report.get('success_criterion') or '未标注'} · "
            f"requested={shown('requested_success_rate')} · "
            f"completion={shown('completion_rate')} · "
            f"failed={failed if failed is not None else '未标注'} · "
            f"episodes={done if done is not None else '—'}/"
            f"{requested if requested is not None else '未标注'}")


def is_single_episode(report: dict):
    """1 集评测（训练后自动录像，或 CLI --episodes 1）只是录像，成功率 1.0/0.0 不是 SR。"""
    if not isinstance(report, dict):
        return False
    requested = report.get("episodes_requested")
    if isinstance(requested, int) and not isinstance(requested, bool):
        return requested == 1
    episodes = report.get("per_episode")
    return isinstance(episodes, list) and len(episodes) == 1


def split_formal_evals(paths):
    """(正式评测 [(path, report|None)], 单集录像 [(path, report)])；读不出的仍算正式（数量不丢）。"""
    formal, single = [], []
    for path in paths:
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            report = None
        if isinstance(report, dict) and is_single_episode(report):
            single.append((path, report))
        else:
            formal.append((path, report if isinstance(report, dict) else None))
    return formal, single


def sr_text(paths):
    """实验表 Success rate 列：1 条正式评测给数值，多条只报数量，只有单集录像时写“单集录像”。"""
    formal, single = split_formal_evals(paths)
    if not formal:
        return f"单集录像({len(single)})" if single else "—"
    if len(formal) > 1:
        return f"多评测({len(formal)})"
    report = formal[0][1]
    if report is None:
        return "—"
    sr = safe_float(report.get("success_rate"))
    return f"{sr:.3f}" if sr is not None else "未标注"


def eval_fields_for_latest(paths):
    """实验表用：取 mtime 最新的 eval.json 的 task/criterion/window/dynamics。"""
    if not paths:
        return eval_display_fields({})
    try:
        path = max(paths, key=lambda p: Path(p).stat().st_mtime)
        report = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return eval_display_fields({})
    return eval_display_fields(report if isinstance(report, dict) else {})


def eval_index(roots):
    """按 eval.json 的 ckpt 反查 run 目录。

    返回 (run_dir -> [当前 ckpt 的 eval...], 无 ckpt 的独立评测, 旧 checkpoint 的评测)。
    第三类不挂到新 run 上，由实验表单独标成“旧评测”。
    """
    index, standalone, stale = {}, [], []
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        try:
            paths = _walk_eval_json(root)
        except OSError:
            continue
        for path in paths:
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                continue
            if not isinstance(report, dict):
                continue
            ckpt = report.get("ckpt")
            run_dir = Path(ckpt).parent.resolve() if ckpt else None
            if run_dir is not None and run_dir.is_dir():
                if eval_ckpt_state(run_dir, report) == "stale":
                    stale.append(path.resolve())
                else:
                    index.setdefault(run_dir, []).append(path.resolve())
            else:
                standalone.append(path.resolve())
    for paths in index.values():
        paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    standalone.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    stale.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return index, standalone, stale


def classify_episode(ep):
    """逐集失败归类；success 优先，其次假前进/碰撞/卡住，全 false 即未到达。

    false_forward 必须先于 collision：eval 里假前进一定同时带 collision=True，
    顺序反了会让“假前进”标签和计数恒为 0。
    """
    if not isinstance(ep, dict):
        return "未到达"
    if ep.get("success"):
        return "成功"
    if ep.get("false_forward"):
        return "假前进"
    if ep.get("collision"):
        return "碰撞"
    if ep.get("stuck"):
        return "卡住"
    return "未到达"


def _tail_jsonl(path: Path, max_bytes=262144):
    """只读文件尾部（历史 run 的 metrics.jsonl 可能很大），用于取最终 test 事件。"""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - max_bytes)
            f.seek(start)
            data = f.read()
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # 首行可能被截断
    rows = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except (ValueError, RecursionError):
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def run_data_path(run_dir, fallback_config=None):
    """训练数据路径：优先 <run>/config.json（trainer 写），其次 Studio 的 job 配置。"""
    try:
        cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        cfg = {}
    data = cfg.get("data") if isinstance(cfg, dict) else None
    if not data:
        data = (fallback_config or {}).get("data")
    return Path(data).expanduser() if data else None


def run_eval_protocol(run_dir, fallback_config=None):
    """run 训练数据的执行协议（dynamics / window_from / hold_s / task / split），(params, 原因)。

    与训练后自动录像共用 auto_eval_params：手动评测和 live demo 以前不传这些参数，
    cf_stab + hold 0.2 训练的权重会按 x3_velocity + hold 1.0 跑，eval.json 还写 ood=false。
    steps / template 不在 .uavr meta 里，仍由调用方给。
    fallback_config 缺省时读该 run 自己的 <run>.monitor/job.json（不用 Studio 内存里上一次训练的 run_meta）。
    """
    if fallback_config is None:
        try:
            job = json.loads((Path(str(run_dir) + ".monitor") / "job.json").read_text(encoding="utf-8"))
            fallback_config = job.get("config") if isinstance(job, dict) else None
        except (OSError, ValueError, RecursionError):
            fallback_config = None
    data = run_data_path(run_dir, fallback_config)
    if data is None:
        return None, "run 配置里没有训练数据路径"
    try:
        meta = video_cli.read_uavr_meta(data)
    except (OSError, ValueError) as exc:
        return None, f"读取 {data} 的 .uavr meta 失败（{exc}）"
    params, reason = video_cli.auto_eval_params(meta)
    if params is not None:
        params["data"] = str(data)
    return params, reason


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
        "## Closed-loop evaluations", "",
    ]
    evals = summary.get("closed_loop_evaluations", [])
    if evals:
        for ev in evals:
            def shown(key):
                value = ev.get(key)
                return value if value not in (None, "") else "未标注"
            failed = ev.get("failed_count")
            lines.append(
                f"- task={shown('task')} / split={shown('split')} / criterion={shown('success_criterion')} / "
                f"window_from={shown('window_from')} / dynamics={shown('dynamics')}: "
                f"success={shown('success_rate')}, requested_success={shown('requested_success_rate')}, "
                f"completion={shown('completion_rate')}, "
                f"failed={failed if failed is not None else '未标注'}, "
                f"episodes={ev.get('episodes_done') if ev.get('episodes_done') is not None else '—'}"
                f"/{ev.get('episodes_requested') if ev.get('episodes_requested') is not None else '未标注'}, "
                f"unique_maps={ev.get('unique_maps', ev.get('unique_geometry'))} (`{ev.get('eval_json')}`)")
            for video in ev.get("videos") or []:
                try:
                    rel = os.path.relpath(str(video), str(run_dir)).replace(os.sep, "/")
                except (TypeError, ValueError):
                    rel = str(video)
                lines.append(f"  - video: [{Path(str(video)).name}]({rel})")
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
    if w < 80 or h < 80:
        return
    # 标题在 (12,6)，图例在 y=32，绘图区从 y=52 开始：三者不再重叠。
    pl, pr, pt, pb = 48, 14, 52, 32
    points = [(x, y) for s in series for x, y in s.get("points", [])
              if x is not None and y is not None and math.isfinite(y)]
    canvas.create_text(12, 6, text=title, anchor="nw", fill="#E5E7EB",
                       font=("TkDefaultFont", 9, "bold"))
    for i, s in enumerate(series):
        lx = pl + 8 + 118 * i
        canvas.create_rectangle(lx, 32, lx + 9, 41, fill=s["color"], outline="")
        canvas.create_text(lx + 13, 36, text=s["name"], anchor="w", fill="#CBD5E1",
                           font=("TkDefaultFont", 8))
    if not points:
        canvas.create_text(w / 2, (pt + h - pb) / 2, text="等待真实训练指标…",
                           fill="#94A3B8")
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
        ymin, ymax = 0.0, 1.0
    def X(x): return pl + (x - xmin) / (xmax - xmin) * (w - pl - pr)
    def Y(y): return h - pb - (y - ymin) / (ymax - ymin) * (h - pt - pb)
    for i in range(5):
        y = ymin + (ymax - ymin) * i / 4
        yy = Y(y)
        canvas.create_line(pl, yy, w - pr, yy, fill="#263449")
        label = f"{y * 100:.0f}%" if percent else f"{y:.3f}"
        canvas.create_text(pl - 6, yy, text=label, anchor="e", fill="#94A3B8",
                           font=("TkDefaultFont", 9))
    # x 轴 5 个刻度
    for i in range(5):
        x = xmin + (xmax - xmin) * i / 4
        xx = X(x)
        canvas.create_line(xx, h - pb, xx, h - pb + 4, fill="#475569")
        label = f"{x:.0f}" if xmax - xmin >= 4 else f"{x:.1f}"
        canvas.create_text(xx, h - pb + 7, text=label, anchor="n", fill="#94A3B8",
                           font=("TkDefaultFont", 9))
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


class StudioApp:
    def __init__(self, root, run_dir=None, refresh_ms=DASHBOARD_REFRESH_MS):
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
        self._eval_auto = False
        self._eval_owner: Path | None = None  # 启动评测时的 run；归档只写它，不写“结束时界面上的 run”
        self._train_started_mono = None
        self._metric_sig = None
        self._events = deque(maxlen=300)  # validation/status/error/class-weight events only
        self._val_events = []  # val 事件不设上限：Best Val 卡片和验证曲线要全量
        self._latest_train_event = None
        self._event_tail = {}
        self._train_log_state = {}
        self._train_log_lines = 0
        self._rows = []
        self._closing = False
        self._resize_after = None
        self._sim_subs = {}
        self._sim_last_seq = {"fpv": 0, "chase": 0, "top": 0}
        self._sim_frame_stamps = {"fpv": None, "chase": None, "top": None}
        self._cam_status_vars = {}
        self._sim_photos = {}
        self._sim_pose = None
        self._sim_path = deque(maxlen=3000)
        self._sim_obstacles = []
        self._sim_waypoint = None
        self._map_static_key = None
        self._map_world = None
        self._map_world_sha = None
        self._map_source = "SDF"
        self._sim_proc = None
        self._sim_stop = None  # (pgid, stage, deadline)：停止 Gazebo 时的 SIGINT→SIGTERM→SIGKILL 升级
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
        self._live_owner = None  # 启动演示时的 run；演示结束只挂到它
        self._live_log = None
        self._live_last_action = None
        self._live_config = None
        self._sim_last_pose_wall = None
        self._sim_frames_seen = {"fpv": 0, "chase": 0, "top": 0}
        self._sim_last_shown = {}  # key -> 上次显示该路相机帧的墙钟（chase/top 限 10 Hz）
        self._vehicle = DEFAULT_VEHICLE  # connect_sim 时按所选 world 重新推断
        self._sim_status_var = tk.StringVar(value="仿真未连接")
        # 流状态每 80 ms 刷新，单独一个变量；事件消息（Gazebo 已退出等）留在 _sim_status_var 不被覆盖。
        self._sim_stream_var = tk.StringVar(value="")
        self._training_status_var = tk.StringVar(value="空闲")
        self._eval_status_var = tk.StringVar(value="未运行闭环评测")
        self._scan_sig = None
        self._scan_cache = None
        self._history_thread = None
        self._history_again = False
        self._eval_index = {}
        self._standalone_evals = []
        self._eval_report = None
        self._eval_episodes = []

        self._configure_style()
        self.root.title("UAV Navigation Studio · RGB Navigation Research Workbench")
        self.root.geometry("1500x960")
        self.root.minsize(1180, 740)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self._build_header()
        self._build_tabs()
        self._build_statusbar()
        # 启动时扫描一次（后台线程），Dashboard 的闭环 SR 卡片在扫描结果回来后刷新。
        self.refresh_history(force=True)
        if self.run_dir:
            self.load_run(self.run_dir)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(200, self._tick)
        self.root.after(300, self._sim_tick)

    def _configure_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        # 正文 10pt，等宽日志保持 9pt；降低默认控件尺寸。
        if tkfont is not None:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
                try:
                    tkfont.nametofont(name).configure(size=10)
                except tk.TclError:
                    pass
            try:
                tkfont.nametofont("TkFixedFont").configure(size=9)
            except tk.TclError:
                pass
        self.root.configure(background="#0B1220")
        self.root.option_add("*TCombobox*Listbox.background", "#0F1A2B")
        self.root.option_add("*TCombobox*Listbox.foreground", "#E2E8F0")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#2563EB")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")
        style.configure("TFrame", background="#0B1220")
        style.configure("Card.TFrame", background="#111C2E")
        style.configure("TLabel", background="#0B1220", foreground="#E2E8F0")
        style.configure("Muted.TLabel", background="#0B1220", foreground="#94A3B8")
        style.configure("CardMuted.TLabel", background="#111C2E", foreground="#94A3B8",
                        font=("TkDefaultFont", 9))
        style.configure("Title.TLabel", background="#0B1220", foreground="#F8FAFC",
                        font=("TkDefaultFont", 16, "bold"))
        style.configure("CardTitle.TLabel", background="#111C2E", foreground="#94A3B8",
                        font=("TkDefaultFont", 9, "bold"))
        style.configure("Hero.TLabel", background="#111C2E", foreground="#F8FAFC",
                        font=("TkDefaultFont", 28, "bold"))
        style.configure("HeroSub.TLabel", background="#111C2E", foreground="#94A3B8",
                        font=("TkDefaultFont", 9))
        style.configure("Chip.TLabel", background="#111C2E", foreground="#F8FAFC",
                        font=("TkDefaultFont", 15, "bold"))
        style.configure("TNotebook", background="#0B1220", borderwidth=0)
        style.configure("TNotebook.Tab", background="#172338", foreground="#CBD5E1",
                        padding=(14, 6))
        style.map("TNotebook.Tab", background=[("selected", "#2563EB")],
                  foreground=[("selected", "#FFFFFF")])
        style.configure("TLabelframe", background="#0B1220", foreground="#CBD5E1")
        style.configure("TLabelframe.Label", background="#0B1220", foreground="#93C5FD",
                        font=("TkDefaultFont", 9, "bold"))
        style.configure("TButton", background="#1E293B", foreground="#E2E8F0",
                        bordercolor="#334155", lightcolor="#1E293B", darkcolor="#1E293B",
                        padding=(8, 3))
        style.map("TButton", background=[("pressed", "#0F1A2B"), ("active", "#263449")],
                  foreground=[("disabled", "#64748B")])
        style.configure("TEntry", fieldbackground="#0F1A2B", foreground="#E2E8F0",
                        insertcolor="#E2E8F0", bordercolor="#334155",
                        lightcolor="#334155", darkcolor="#334155", padding=(4, 2))
        style.map("TEntry", fieldbackground=[("readonly", "#0F1A2B"), ("disabled", "#0B1220")],
                  foreground=[("disabled", "#64748B")])
        style.configure("TCombobox", fieldbackground="#0F1A2B", background="#1E293B",
                        foreground="#E2E8F0", arrowcolor="#E2E8F0", bordercolor="#334155",
                        padding=(4, 2))
        style.map("TCombobox",
                  fieldbackground=[("readonly", "#0F1A2B"), ("disabled", "#0B1220")],
                  background=[("active", "#263449")],
                  selectbackground=[("readonly", "#1E293B")],
                  selectforeground=[("readonly", "#E2E8F0")])
        style.configure("TCheckbutton", background="#0B1220", foreground="#E2E8F0",
                        indicatorbackground="#0F1A2B", indicatorforeground="#E2E8F0",
                        padding=(2, 2))
        style.map("TCheckbutton", background=[("active", "#0B1220")],
                  indicatorbackground=[("selected", "#2563EB")])
        for orient in ("Vertical", "Horizontal"):
            style.configure(f"{orient}.TScrollbar", background="#334155",
                            troughcolor="#0B1220", bordercolor="#0B1220",
                            arrowcolor="#94A3B8", relief="flat")
        style.configure("Treeview", background="#111C2E", fieldbackground="#111C2E",
                        foreground="#E2E8F0", rowheight=26, bordercolor="#263449")
        style.configure("Treeview.Heading", background="#1E293B", foreground="#E2E8F0",
                        font=("TkDefaultFont", 9, "bold"), padding=(4, 3))
        style.map("Treeview", background=[("selected", "#1E3A5F")],
                  foreground=[("selected", "#F8FAFC")])

    def _build_header(self):
        header = ttk.Frame(self.root, padding=(16, 8))
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
        ttk.Label(bar, textvariable=self._sim_stream_var, style="Muted.TLabel").pack(
            side="left", padx=(8, 0))
        ttk.Label(bar, text="  ·  ", style="Muted.TLabel").pack(side="left")
        ttk.Label(bar, textvariable=self._eval_status_var).pack(side="left")

    def _hero_card(self, parent, title, var, sub_var):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=(10, 6))
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=var, style="Hero.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=sub_var, style="HeroSub.TLabel").pack(anchor="w")
        return frame

    def _chip_card(self, parent, title, var):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=(8, 5))
        ttk.Label(frame, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(frame, textvariable=var, style="Chip.TLabel").pack(anchor="w")
        return frame

    def _build_dashboard(self):
        tab = self.dashboard_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        self.run_title_var = tk.StringVar(value="尚未选择实验")
        title_bar = ttk.Frame(tab)
        title_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        title_bar.columnconfigure(0, weight=1)
        ttk.Label(title_bar, textvariable=self.run_title_var, style="Title.TLabel").grid(
            row=0, column=0, sticky="w")
        ttk.Button(title_bar, text="打开三视角仿真 →",
                   command=lambda: self.tabs.select(self.sim_tab)).grid(row=0, column=1, sticky="e")
        # 指标两级：2 张 hero 卡（各 50%）+ 5 个 chip（各 20%）。
        cards = ttk.Frame(tab)
        cards.grid(row=1, column=0, sticky="ew", pady=(0, 6))
        self.card_vars = {k: tk.StringVar(value="—") for k in
                          ("epoch", "train_ce", "dev_ce", "dev_macro",
                           "test_macro", "test_bal", "closed_loop_sr")}
        self.hero_sub = {"test_macro": tk.StringVar(value="test split · 离线动作分类"),
                         "closed_loop_sr": tk.StringVar(value="尚无闭环评测")}
        hero = ttk.Frame(cards)
        hero.grid(row=0, column=0, sticky="ew")
        hero.columnconfigure(0, weight=1, uniform="hero")
        hero.columnconfigure(1, weight=1, uniform="hero")
        self._hero_card(hero, "Test Macro-F1 · 离线", self.card_vars["test_macro"],
                        self.hero_sub["test_macro"]).grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        self._hero_card(hero, "闭环评测成功率 · 判据见副标题", self.card_vars["closed_loop_sr"],
                        self.hero_sub["closed_loop_sr"]).grid(row=0, column=1, sticky="nsew",
                                                              padx=(4, 0))
        chips = ttk.Frame(cards)
        chips.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        chip_titles = [("Best Val Macro-F1 @ep", "dev_macro"), ("训练轮数", "epoch"),
                       ("Test Balanced Acc", "test_bal"), ("Train CE", "train_ce"),
                       ("Validation CE", "dev_ce")]
        for i, (title, key) in enumerate(chip_titles):
            chips.columnconfigure(i, weight=1, uniform="chip")
            self._chip_card(chips, title, self.card_vars[key]).grid(
                row=0, column=i, sticky="nsew",
                padx=(0 if i == 0 else 3, 0 if i == len(chip_titles) - 1 else 3))
        body = ttk.Frame(tab)
        body.grid(row=2, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        # 右列固定为请求宽度：event_text 不再用 80 字符宽把图表列挤窄。
        body.columnconfigure(1, weight=0)
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
        right.rowconfigure(5, weight=1)
        right.columnconfigure(0, weight=1)
        # 诊断区放在事件日志上面：最该先看的结论不被日志压到下面。
        ttk.Label(right, text="离线诊断 · test split", style="Title.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 4))
        self.test_summary_var = tk.StringVar(value="尚无 test split 离线评测记录。")
        ttk.Label(right, textvariable=self.test_summary_var, style="Muted.TLabel",
                  wraplength=400, justify="left").grid(row=1, column=0, sticky="ew")
        self.confusion_canvas = tk.Canvas(right, bg="#111C2E", highlightthickness=0, height=200)
        self.confusion_canvas.grid(row=2, column=0, sticky="ew", pady=(6, 0))
        self.recall_canvas = tk.Canvas(right, bg="#111C2E", highlightthickness=0, height=104)
        self.recall_canvas.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        self._confusion_matrix = None
        self._recalls = []
        self.confusion_canvas.bind("<Configure>", lambda _e: self._draw_confusion())
        self.recall_canvas.bind("<Configure>", lambda _e: self._draw_recall())
        ttk.Label(right, text="训练事件", style="CardTitle.TLabel").grid(
            row=4, column=0, sticky="w", pady=(8, 4))
        self.event_text = tk.Text(right, bg="#111C2E", fg="#CBD5E1", insertbackground="#fff",
                                  relief="flat", wrap="word", font=("TkFixedFont", 9),
                                  width=1, height=8)
        self.event_text.grid(row=5, column=0, sticky="nsew")
        sb = ttk.Scrollbar(right, command=self.event_text.yview)
        sb.grid(row=5, column=1, sticky="ns")
        self.event_text.configure(yscrollcommand=sb.set)

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
        self.save_video_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(config, text="训练后保存视频（闭环 1 集）",
                        variable=self.save_video_var).grid(
            row=3, column=2, columnspan=2, sticky="w", pady=(6, 0))
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
        # 按钮实际启动 gz sim -s（headless server）；容器内无 NVIDIA EGL，GUI 必失败。
        part_row = ttk.Frame(controls)
        part_row.grid(row=2, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Entry(part_row, textvariable=self.partition_var, width=28).pack(side="left")
        # 渲染后端：进程环境里已有 UAVRGB_RENDER_BACKEND 就沿用，否则用文件顶部的默认值。
        self.render_backend_var = tk.StringVar(
            value=os.environ.get("UAVRGB_RENDER_BACKEND") or DEFAULT_RENDER_BACKEND)
        ttk.Label(part_row, text="渲染后端").pack(side="left", padx=(10, 4))
        ttk.Combobox(part_row, textvariable=self.render_backend_var, values=RENDER_BACKENDS,
                     width=9, state="readonly").pack(side="left")
        self.scene_task_var = tk.StringVar(value="campus")
        self.scene_density_var = tk.StringVar(value=DENSITY_DEFAULT)
        self.scene_seed_var = tk.StringVar(value="42")
        ttk.Label(controls, text="快速生成场景").grid(row=3, column=0, sticky="w", pady=3)
        gen_row = ttk.Frame(controls)
        gen_row.grid(row=3, column=1, columnspan=2, sticky="w", padx=5)
        ttk.Combobox(gen_row, textvariable=self.scene_task_var,
                     values=TASKS, width=10, state="readonly").pack(side="left")
        ttk.Label(gen_row, text="Density(forest_e1)").pack(side="left", padx=(10, 4))
        ttk.Combobox(gen_row, textvariable=self.scene_density_var,
                     values=DENSITIES, width=8, state="readonly").pack(side="left")
        ttk.Label(gen_row, text="Seed").pack(side="left", padx=(10, 4))
        ttk.Entry(gen_row, textvariable=self.scene_seed_var, width=8).pack(side="left")
        ttk.Button(gen_row, text="生成 SDF + metadata", command=self.generate_scene).pack(side="left", padx=8)
        row = ttk.Frame(controls)
        row.grid(row=0, column=3, rowspan=4, sticky="ns", padx=(12, 0))
        ttk.Button(row, text="启动 Gazebo（headless）", command=self.start_gazebo).pack(fill="x", pady=2)
        ttk.Button(row, text="连接当前仿真", command=self.connect_sim).pack(fill="x", pady=2)
        ttk.Button(row, text="断开视角", command=self.disconnect_sim).pack(fill="x", pady=2)
        ttk.Button(row, text="停止本程序启动的 Gazebo", command=self.stop_gazebo).pack(fill="x", pady=2)

        live = ttk.LabelFrame(tab, text="学生策略实时演示（使用真实 checkpoint，不是教师动作）", padding=9)
        live.grid(row=1, column=0, sticky="ew", pady=(8, 8))
        live.columnconfigure(1, weight=1)
        self.ckpt_var = tk.StringVar(value="")
        self.live_steps_var = tk.StringVar(value=LIVE_STEPS_DEFAULT)
        self.live_hold_var = tk.StringVar(value=LIVE_HOLD_DEFAULT)
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
        # uniform：格子按权重分配，不按各自当前图像大小；否则小面板缩图后请求更小，越缩越小。
        views.columnconfigure(0, weight=1, uniform="sim_col")
        views.columnconfigure(1, weight=1, uniform="sim_col")
        # 上排 3:2 下排：Canvas 的 height=300 请求高度会把上排挤成标题条。
        views.rowconfigure(0, weight=3, uniform="sim_row")
        views.rowconfigure(1, weight=2, uniform="sim_row")
        self.chase_panel, self.chase_label = self._camera_panel(
            views, "第三人称跟随视角 · CHASE CAMERA", "chase")
        self.chase_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 4))
        self.fpv_panel, self.fpv_label = self._camera_panel(
            views, "第一人称视角 · RGB INPUT", "fpv")
        self.fpv_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 4))
        self.topcam_panel, self.topcam_label = self._camera_panel(
            views, "顶视相机 · RENDERED TOP CAMERA", "top")
        self.topcam_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 4), pady=(4, 0))
        top = ttk.LabelFrame(views, text="轨迹分析地图 · TOP-DOWN MAP", padding=4)
        top.grid(row=1, column=1, sticky="nsew", padx=(4, 0), pady=(4, 0))
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        self.top_canvas = tk.Canvas(top, bg="#0B1422", highlightthickness=0, height=1)
        self.top_canvas.grid(row=0, column=0, sticky="nsew")
        self.top_canvas.bind("<Configure>", lambda _e: self._draw_topdown())
        self.sim_info_var = tk.StringVar(value="等待 Gazebo 位姿数据")
        ttk.Label(top, textvariable=self.sim_info_var, style="Muted.TLabel").grid(
            row=1, column=0, sticky="w", pady=(4, 0))

    def _camera_panel(self, parent, title, key):
        frame = ttk.LabelFrame(parent, text=title, padding=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        label = tk.Label(frame, text="等待相机数据…", bg="#0B1422", fg="#94A3B8",
                         font=("TkDefaultFont", 10), anchor="center")
        label.grid(row=0, column=0, sticky="nsew")
        self._cam_status_vars[key] = tk.StringVar(value="等待相机数据…")
        ttk.Label(frame, textvariable=self._cam_status_vars[key],
                  style="Muted.TLabel", anchor="w").grid(row=1, column=0, sticky="ew")
        return frame, label

    def _build_experiments(self):
        tab = self.experiments_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(tab)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Button(toolbar, text="刷新实验列表",
                   command=lambda: self.refresh_history(force=True)).pack(side="left")
        ttk.Button(toolbar, text="加载选中实验", command=self.load_selected_history).pack(
            side="left", padx=6)
        ttk.Button(toolbar, text="打开实验目录", command=self.open_selected_history).pack(side="left")
        ttk.Button(toolbar, text="打开实验报告", command=self.open_report).pack(side="left", padx=6)
        cols = ("path", "kind", "status", "task", "criterion", "window_from", "dynamics",
                "epochs", "macro", "test", "sr", "started")
        table = ttk.Treeview(tab, columns=cols, show="headings", height=9)
        for col, label, width, anchor in (
                ("path", "实验路径（相对 DATA_ROOT）", 300, "w"),
                ("kind", "类型", 65, "center"),
                ("status", "状态", 105, "w"),
                ("task", "Task", 60, "center"),
                ("criterion", "Success criterion", 125, "w"),
                ("window_from", "Window from", 80, "center"),
                ("dynamics", "Dynamics", 70, "center"),
                ("epochs", "Epochs", 55, "e"),
                ("macro", "Test Macro-F1", 105, "e"),
                ("test", "Test Balanced Acc", 115, "e"),
                ("sr", "Success rate", 95, "e"),
                ("started", "开始时间", 135, "w")):
            table.heading(col, text=label)
            table.column(col, width=width, anchor=anchor)
        table.grid(row=1, column=0, sticky="nsew")
        sb = ttk.Scrollbar(tab, command=table.yview)
        sb.grid(row=1, column=1, sticky="ns")
        table.configure(yscrollcommand=sb.set)
        table.bind("<Double-1>", lambda _e: self.load_selected_history(show_dashboard=False))
        self.history_tree = table
        eval_box = ttk.LabelFrame(tab, text="独立 Gazebo 闭环评测（训练分类指标 ≠ 导航成功率）", padding=8)
        eval_box.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        self.eval_task_var = tk.StringVar(value="base")
        self.eval_density_var = tk.StringVar(value=DENSITY_DEFAULT)
        self.eval_split_var = tk.StringVar(value="test")
        self.eval_episodes_var = tk.StringVar(value=EVAL_EPISODES_DEFAULT)
        self.eval_appearance_var = tk.StringVar(value="v1")
        ttk.Label(eval_box, text="Task（必须匹配训练数据）").pack(side="left")
        ttk.Combobox(eval_box, textvariable=self.eval_task_var,
                     values=TASKS, width=10, state="readonly").pack(
            side="left", padx=5)
        ttk.Label(eval_box, text="Density").pack(side="left")
        ttk.Combobox(eval_box, textvariable=self.eval_density_var,
                     values=DENSITIES, width=8, state="readonly").pack(side="left", padx=5)
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
            row=3, column=0, sticky="w", pady=(6, 0))
        # 逐集评测明细：失败模式（尤其“未到达”）必须可见，不能只留一行状态栏。
        detail = ttk.LabelFrame(tab, text="评测明细（双击历史行自动加载最新 eval.json）", padding=6)
        detail.grid(row=4, column=0, sticky="nsew", pady=(6, 0))
        self.eval_detail_box = detail
        detail.columnconfigure(0, weight=1)
        chips = ttk.Frame(detail)
        chips.grid(row=0, column=0, sticky="ew")
        self.eval_chip_vars = {k: tk.StringVar(value="—") for k in
                               ("sr", "collision", "stuck", "false_forward",
                                "unreached", "maps", "p95")}
        for i, (label, key) in enumerate((("SR", "sr"), ("碰撞", "collision"),
                                          ("卡住", "stuck"), ("假前进", "false_forward"),
                                          ("未到达", "unreached"), ("maps", "maps"),
                                          ("p95", "p95"))):
            ttk.Label(chips, text=label, style="CardTitle.TLabel").grid(row=0, column=i * 2,
                                                                        sticky="w", padx=(0, 4))
            ttk.Label(chips, textvariable=self.eval_chip_vars[key]).grid(
                row=0, column=i * 2 + 1, sticky="w", padx=(0, 16))
        filter_row = ttk.Frame(detail)
        filter_row.grid(row=1, column=0, sticky="ew", pady=(6, 4))
        ttk.Label(filter_row, text="筛选").pack(side="left")
        self.eval_filter_var = tk.StringVar(value="全部")
        self.eval_filter_combo = ttk.Combobox(
            filter_row, textvariable=self.eval_filter_var, width=10, state="readonly",
            values=("全部", "仅失败", "碰撞", "卡住", "假前进", "未到达", "成功"))
        self.eval_filter_combo.pack(side="left", padx=6)
        self.eval_filter_combo.bind("<<ComboboxSelected>>", lambda _e: self._fill_episode_table())
        self.eval_count_var = tk.StringVar(value="显示 0/0")
        ttk.Label(filter_row, textvariable=self.eval_count_var,
                  style="Muted.TLabel").pack(side="left", padx=8)
        ep_cols = ("idx", "seed", "result", "steps", "max_x", "wp_min", "actions", "instruction")
        self.eval_tree = ttk.Treeview(detail, columns=ep_cols, show="headings", height=8)
        for col, label, width, anchor in (
                ("idx", "#", 40, "e"), ("seed", "Seed", 70, "e"),
                ("result", "结果", 80, "center"), ("steps", "Steps", 60, "e"),
                ("max_x", "Max x", 75, "e"), ("wp_min", "WP min dist", 95, "e"),
                ("actions", "Actions(SN/FS/YL/YR)", 160, "w"),
                ("instruction", "指令", 420, "w")):
            self.eval_tree.heading(col, text=label)
            self.eval_tree.column(col, width=width, anchor=anchor)
        self.eval_tree.grid(row=2, column=0, sticky="nsew")
        ep_sb = ttk.Scrollbar(detail, command=self.eval_tree.yview)
        ep_sb.grid(row=2, column=1, sticky="ns")
        self.eval_tree.configure(yscrollcommand=ep_sb.set)
        for tag, fg, bg in (("成功", "#34D399", "#0F2A1E"), ("碰撞", "#F87171", "#2A1416"),
                            ("卡住", "#FBBF24", "#2A2310"), ("假前进", "#C084FC", "#231530"),
                            ("未到达", "#94A3B8", "#1A2233")):
            self.eval_tree.tag_configure(tag, foreground=fg, background=bg)

    # ------------------------------------------------------------------ training
    def choose_dataset(self):
        path = filedialog.askopenfilename(title="选择 .uavr 数据集",
                                          filetypes=[("UAV RGB dataset", "*.uavr"), ("All", "*.*")])
        if path:
            self.data_var.set(path)
            sidecar = Path(path).parent / "instructions.jsonl"
            self.text_dim_var.set("16" if sidecar.is_file() else "0")

    def _new_run_dir(self):
        try:
            RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("数据目录不可用",
                                 f"无法创建 run 目录：\n{RUNS_ROOT}\n\n{exc}\n\n"
                                 "请检查 DATA_ROOT 环境变量是否指向可写目录。")
            return None
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
        if run_dir is None:
            return
        run_dir.mkdir(parents=True, exist_ok=False)
        sidecar = Path(str(run_dir) + ".monitor")
        sidecar.mkdir(parents=True, exist_ok=True)
        config = {
            "schema": "uavrgb.train.v1", "run_dir": str(run_dir), "data": str(data),
            "epochs": epochs, "batch_size": batch, "lr": lr, "seed": seed,
            "text_dim": text_dim, "class_weights": "auto" if self.class_weights_var.get() else "none",
            "drop_collision": bool(self.drop_collision_var.get()),
            # trainer_native 只读已知键，save_video/video_episodes 供 Studio 训练后自动评测。
            "save_video": bool(self.save_video_var.get()), "video_episodes": 1,
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
        # studio_pid：别的 Studio 窗口据此判断这个 starting 是否有人在管，不把它当失联回收。
        write_json_atomic(sidecar / "lifecycle.json", {"status": "starting", "started_at": run_meta["started_at"],
                                                       "studio_pid": os.getpid()})
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
            self.refresh_history(force=True)
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
        self._val_events = []
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
        self.refresh_history(force=True)

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
            # 训练中途完成的评测/演示已写进旧摘要；整份覆盖会把它们从报告里抹掉。
            carry_attachments(self.run_dir / "studio_summary.json", summary)
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
            self.refresh_history(force=True)
            self.tabs.select(self.dashboard_tab)
            config = (self.run_meta or {}).get("config", {})
            if rc == 0 and best_ckpt.is_file() and config.get("save_video"):
                try:
                    self._start_auto_eval(best_ckpt, config)
                except Exception as exc:  # noqa: BLE001 - 自动录像不能影响训练结果
                    self._eval_auto = False
                    self._auto_eval_note(
                        f"自动录像失败：{type(exc).__name__}: {exc}", self.run_dir)
        self.start_train_btn.configure(state="normal")
        self.stop_train_btn.configure(state="disabled")

    def _start_auto_eval(self, ckpt, config):
        """训练成功后自动跑 1 集闭环录像；失败只写 note，不改训练状态。"""
        run_dir = self.run_dir
        params, reason = run_eval_protocol(run_dir, config)
        if params is None:
            self._auto_eval_note(f"自动录像跳过：{reason}", run_dir)
            return
        params["episodes"] = AUTO_EVAL_EPISODES
        if INFER_BIN.is_file():
            params["infer_bin"] = str(INFER_BIN)
        self._eval_auto = True
        if not self.start_closed_loop_eval(auto=params):
            self._eval_auto = False
            self._auto_eval_note("自动录像未启动（已有评测在运行或参数无效）", run_dir)

    def _auto_eval_note(self, text, run_dir):
        """自动录像的失败只进 note 和训练日志，不把训练改成失败；只写这次录像所属的 run。"""
        self._append_train_log(f"[studio] {text}")
        if not run_dir:
            return
        run_dir = Path(run_dir)
        path = run_dir / "studio_summary.json"
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        summary.setdefault("notes", []).append(text)
        write_json_atomic(path, summary)
        write_experiment_report(run_dir, summary)

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
        # 全量读：只读末尾 1 MiB / 200 条时，长 run 的 val 事件几乎全被 train_step 挤掉。
        events, reset = monitor_io.tail_events(str(self.run_dir), self._event_tail,
                                               limit=None, window=None)
        if reset:
            self._events.clear()
            self._val_events = []
            self._latest_train_event = None
        for event in events:
            if event.get("event") == "train_step":
                self._latest_train_event = event
                continue
            self._events.append(event)
            if event.get("event") == "validation" and event.get("split") == "val":
                self._val_events.append(event)
        self._metric_sig = sig
        self._update_dashboard()
        self._update_event_text()

    def _run_eval_paths(self, run_dir):
        """只读缓存索引，不做目录扫描；1 条数值、多条只报数量，避免混口径。"""
        if not run_dir:
            return []
        run_dir = Path(run_dir).resolve()
        paths = list(self._eval_index.get(run_dir, []))
        own = run_dir / "eval.json"
        if own.is_file() and own not in paths:
            try:
                report = json.loads(own.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                report = None
            # 旧 checkpoint 的 eval.json 不挂到当前 run；它由实验表“旧评测”行展示。
            if not isinstance(report, dict) or eval_ckpt_state(run_dir, report) != "stale":
                paths.append(own)
        paths.sort(key=lambda p: Path(p).stat().st_mtime, reverse=True)
        return paths

    @staticmethod
    def _report_maps(report):
        if not isinstance(report, dict):
            return None
        value = report.get("unique_maps", report.get("unique_geometry"))
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
        gids = {ep.get("geometry_gid") for ep in report.get("per_episode") or []
                if isinstance(ep, dict) and ep.get("geometry_gid")}
        return len(gids) if gids else None

    def _update_sr_card(self):
        formal, single = split_formal_evals(self._run_eval_paths(self.run_dir))
        var, sub = self.card_vars["closed_loop_sr"], self.hero_sub["closed_loop_sr"]
        extra = f" · 另有 {len(single)} 份单集录像未计入" if single else ""
        if not formal:
            if single:
                var.set("单集录像")
                sub.set(f"{len(single)} 份 1 集录像，不是成功率 · 需要多集正式闭环评测")
            else:
                var.set("—")
                sub.set("尚无闭环评测 · 训练分类指标 ≠ 闭环成功率")
            return
        if len(formal) > 1:
            var.set(f"多评测({len(formal)})")
            detail = "多评测未合并，口径可能不同；在 Experiments 页选择具体报告"
            latest = formal[0][1]
            if latest is not None:
                maps = self._report_maps(latest)
                detail = (f"最新：{eval_rate_context(latest)} · "
                          f"{maps if maps is not None else '—'} 张图 · " + detail)
            sub.set(detail + extra)
            return
        report = formal[0][1]
        if report is None:
            var.set("—")
            sub.set("eval.json 无法读取" + extra)
            return
        sr = safe_float(report.get("success_rate"))
        maps = self._report_maps(report)
        var.set(f"{sr:.3f}" if sr is not None else "未标注")
        # 报告必须同时给出判据、请求集成功率、完成率和失败数：三者分母不同，
        # 只报 success_rate 会让“被跳过的集”看起来像成功。
        sub.set(f"{eval_rate_context(report)} · {maps if maps is not None else '—'} 张图" + extra)

    def _update_dashboard(self):
        rows = self._rows
        valid = [r for r in rows if r.get("epoch") is not None]
        last = valid[-1] if valid else {}
        evs = list(self._events)
        vals = self._val_events
        tests = [e for e in evs if e.get("event") == "validation" and e.get("split") == "test"]
        def metric_key(e):
            v = safe_float(e.get("macro_f1"))
            return v if v is not None else -1.0
        best = max(vals, key=metric_key, default=None)
        test = tests[-1] if tests else None
        self.card_vars["epoch"].set(str(len(valid)))
        self.card_vars["train_ce"].set(f"{last['train_ce']:.4f}" if last.get("train_ce") is not None else "—")
        self.card_vars["dev_ce"].set(f"{last['dev_ce']:.4f}" if last.get("dev_ce") is not None else "—")
        best_f1 = safe_float(best.get("macro_f1")) if best else None
        self.card_vars["dev_macro"].set(f"{best_f1:.3f}" if best_f1 is not None else "—")
        self.card_vars["test_macro"].set(f"{safe_float(test.get('macro_f1')):.3f}" if test and safe_float(test.get("macro_f1")) is not None else "—")
        self.card_vars["test_bal"].set(f"{safe_float(test.get('balanced_accuracy')):.3f}" if test and safe_float(test.get("balanced_accuracy")) is not None else "—")
        self._update_sr_card()
        if test:
            self.test_summary_var.set(
                f"test：Acc={test.get('action_accuracy')} · Macro-F1={test.get('macro_f1')} · "
                f"Bal={test.get('balanced_accuracy')}（动作分类，不是导航 SR）")
        else:
            self.test_summary_var.set("当前没有 test split validation 事件；缺失指标保持 —，不会用训练准确率代替。")
        confusion = test.get("confusion") if test else None
        if isinstance(confusion, list) and len(confusion) == 16:
            self._confusion_matrix = [[safe_float(v) or 0.0 for v in confusion[i:i + 4]]
                                      for i in range(0, 16, 4)]
            self._recalls = []
            for i, row in enumerate(self._confusion_matrix):
                total = sum(row)
                self._recalls.append({"class": ACTION_NAMES[i],
                                      "recall": (row[i] / total) if total > 0 else None})
        else:
            self._confusion_matrix = None
            self._recalls = []
        self._draw_confusion()
        self._draw_recall()
        self._redraw_charts()

    def _draw_confusion(self):
        if not hasattr(self, "confusion_canvas"):
            return
        c = self.confusion_canvas
        c.delete("all")
        w, h = max(1, c.winfo_width()), max(1, c.winfo_height())
        matrix = self._confusion_matrix
        if not matrix:
            c.create_text(w / 2, h / 2, text="Test confusion matrix：—（当前 run 无有效 test confusion）",
                          fill="#64748B", font=("TkDefaultFont", 9))
            return
        short = ("SN", "FS", "YL", "YR")
        left, top = 30, 20
        cell = max(8, min((w - left - 6) / 4, (h - top - 22) / 4))
        def mix(t):
            a, b = (0x16, 0x23, 0x3A), (0x25, 0x63, 0xEB)
            return "#%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(a, b))
        for j, name in enumerate(short):
            c.create_text(left + cell * (j + 0.5), top - 8, text=name, anchor="s",
                          fill="#94A3B8", font=("TkDefaultFont", 9, "bold"))
        for i, row in enumerate(matrix):
            total = sum(row) or 1.0
            c.create_text(left - 7, top + cell * (i + 0.5), text=short[i], anchor="e",
                          fill="#94A3B8", font=("TkDefaultFont", 9, "bold"))
            for j, value in enumerate(row):
                x0, y0 = left + cell * j, top + cell * i
                c.create_rectangle(x0, y0, x0 + cell, y0 + cell, fill=mix(value / total),
                                   outline="#0B1220")
                if i == j:
                    c.create_rectangle(x0 + 1, y0 + 1, x0 + cell - 1, y0 + cell - 1,
                                       outline="#34D399", width=2)
                c.create_text(x0 + cell / 2, y0 + cell / 2, text=str(int(value)),
                              fill="#F8FAFC", font=("TkDefaultFont", 9))
        c.create_text(left, top + cell * 4 + 6,
                      text="行=真值，列=预测 · SN/FS/YL/YR · 颜色=行内占比",
                      anchor="nw", fill="#94A3B8", font=("TkDefaultFont", 8))

    def _draw_recall(self):
        if not hasattr(self, "recall_canvas"):
            return
        c = self.recall_canvas
        c.delete("all")
        w, h = max(1, c.winfo_width()), max(1, c.winfo_height())
        if not self._recalls:
            c.create_text(w / 2, h / 2, text="Per-class Recall：—", fill="#64748B",
                          font=("TkDefaultFont", 9))
            return
        row_h = 25
        short = ("SN", "FS", "YL", "YR")
        for i, item in enumerate(self._recalls[:4]):
            y0 = i * row_h
            value = item.get("recall")
            c.create_text(0, y0 + row_h / 2, text=short[i], anchor="w",
                          fill="#94A3B8", font=("TkDefaultFont", 9))
            bar_x, bar_w = 26, max(10, w - 26 - 42)
            c.create_rectangle(bar_x, y0 + 7, bar_x + bar_w, y0 + row_h - 7,
                               fill="#16233A", outline="")
            if value is not None:
                color = "#F87171" if value < 0.4 else ("#F59E0B" if value < 0.7 else "#34D399")
                c.create_rectangle(bar_x, y0 + 7, bar_x + bar_w * min(1.0, value),
                                   y0 + row_h - 7, fill=color, outline="")
                c.create_text(w - 2, y0 + row_h / 2, text=f"{value:.2f}", anchor="e",
                              fill="#CBD5E1", font=("TkDefaultFont", 9))
            else:
                c.create_text(w - 2, y0 + row_h / 2, text="—", anchor="e", fill="#64748B",
                              font=("TkDefaultFont", 9))

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
        vals = self._val_events
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
            self._set_live_ckpt(Path(path))

    def _ckpt_protocol(self, ckpt):
        """权重所在 run（ckpt 的父目录）的执行协议。"""
        return run_eval_protocol(Path(ckpt).expanduser().resolve().parent)

    def _set_live_ckpt(self, ckpt):
        """选中权重时把 Hold 预填成训练数据的 hold_s：演示窗口与训练协议一致，用户仍可改。"""
        self.ckpt_var.set(str(ckpt))
        protocol, _reason = self._ckpt_protocol(ckpt)
        if protocol is not None and protocol.get("hold_s") is not None:
            self.live_hold_var.set(f"{float(protocol['hold_s']):g}")

    def _load_world_geometry(self):
        raw = self.world_var.get().strip()
        world = Path(raw).expanduser().resolve() if raw else None
        self._sim_obstacles = parse_sdf_obstacles(world) if world else []
        self._map_world, self._map_source = world, "SDF"
        try:
            self._map_world_sha = sha256_file(world) if world and world.is_file() else None
        except OSError:
            self._map_world_sha = None
        meta = load_scene_meta(self.meta_var.get())
        wp = meta.get("waypoint") if isinstance(meta.get("waypoint"), dict) else None
        self._sim_waypoint = (safe_float(wp.get("x")), safe_float(wp.get("y"))) if wp else None
        if not self._sim_obstacles and isinstance(meta.get("pillars"), list):
            self._map_source = "metadata"
            for p in meta["pillars"]:
                try:
                    sx, sy = p.get("size", [0.4, 0.4, 1.0])[:2]
                    self._sim_obstacles.append({"name": p.get("name", "obstacle"), "kind": "box",
                                                "x": float(p["x"]), "y": float(p["y"]),
                                                "sx": float(sx), "sy": float(sy), "yaw": 0.0})
                except (KeyError, TypeError, ValueError):
                    continue
        self._map_static_key = None  # 障碍物/目标变了，静态层必须重画
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
        if task == "forest_e1":
            # forest_e1 是 E1 任务：scene_gen 要 --density；不传 --vehicle 会生成 X3 世界，cf_sitl 跑不了。
            cmd += ["--density", self.scene_density_var.get(), "--vehicle", "e1_mini"]
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
            self._sim_status_var.set(f"场景已生成 · {out.name}；现在可以启动 Gazebo")
        else:
            self._sim_status_var.set(f"场景生成失败 rc={rc} · 查看 {out}")

    def start_gazebo(self):
        if self._scene_proc is not None and self._scene_proc.poll() is None:
            messagebox.showwarning("场景仍在生成", "请等待 SDF 和 metadata 生成完成后再启动 Gazebo。")
            return
        if self._sim_proc is not None and self._sim_proc.poll() is None:
            messagebox.showinfo("Gazebo 已运行", "当前 Studio 已经启动一个 Gazebo 实例。")
            return
        if self._sim_stop is not None:
            messagebox.showinfo("Gazebo 仍在停止", "上一个 Gazebo 进程组还没退出，请稍候再启动。")
            return
        world = Path(self.world_var.get()).expanduser().resolve()
        if not world.is_file():
            messagebox.showerror("World 不存在", f"找不到 SDF：{world}")
            return
        partition = self.partition_var.get().strip() or ("uavrgb_studio_" + uuid.uuid4().hex[:10])
        self.partition_var.set(partition)
        owns_partition = self._sim_partition_active is None
        self._set_partition(partition)
        worlds_resource = str(REPO_ROOT / "sim" / "worlds")
        existing_resources = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
        resources = [worlds_resource] + [x for x in existing_resources.split(os.pathsep) if x]
        os.environ["GZ_SIM_RESOURCE_PATH"] = os.pathsep.join(dict.fromkeys(resources))
        backend = self._apply_render_backend()
        try:
            # 复用采集脚本的 EGL/XDG 环境；容器里 GUI 渲染会失败，统一走 headless server。
            import collect_parallel
            env = collect_parallel.gz_env(partition)
            xdg = env.get("XDG_RUNTIME_DIR")
            if xdg:
                os.environ["XDG_RUNTIME_DIR"] = xdg  # 同步回本进程，后续 gz 命令共用
                try:
                    Path(xdg).mkdir(mode=0o700, parents=True, exist_ok=True)
                except OSError:
                    pass
        except (ImportError, ValueError, OSError) as exc:
            if owns_partition:  # 失败要撤销本次 _set_partition，否则之后连接被锁在这个分区
                self._restore_partition()
            self._sim_status_var.set(f"Gazebo 环境准备失败：{exc}")
            messagebox.showerror("Gazebo 环境准备失败", str(exc))
            return
        env["GZ_SIM_RESOURCE_PATH"] = os.environ["GZ_SIM_RESOURCE_PATH"]
        log_dir = RUNS_ROOT / "simulation_logs"
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-gazebo.log")
            self._sim_log_handle = log_path.open("ab", buffering=0)
        except OSError as exc:
            if owns_partition:
                self._restore_partition()
            self._sim_status_var.set(f"Gazebo 日志无法写入：{exc}")
            messagebox.showerror("数据目录不可用", f"无法写入仿真日志：\n{log_dir}\n\n{exc}")
            return
        self._sim_world_path = world
        self._sim_world_sha256 = sha256_file(world)
        cmd = ["gz", "sim", "-s", "-r"]
        if env.get("UAVRGB_RENDER_BACKEND") == "gpu":
            cmd.append("--headless-rendering")
        cmd.append(str(world))
        try:
            self._sim_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT),
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
        self._sim_status_var.set(f"Gazebo headless 启动中 · partition={partition} · 渲染={backend}")
        self._load_world_geometry()
        self.root.after(SIM_CONNECT_DELAY_MS, lambda proc=self._sim_proc: self._connect_after_start(proc))

    def _apply_render_backend(self):
        """把界面选的渲染后端写进本进程环境并返回。

        collect_parallel.gz_env 只认环境变量；评测/演示子进程继承本进程环境再各自起 Gazebo，
        这样三处用同一个后端。空值退回 DEFAULT_RENDER_BACKEND；非法值留给 gz_env 报错。
        """
        backend = self.render_backend_var.get().strip() or DEFAULT_RENDER_BACKEND
        os.environ["UAVRGB_RENDER_BACKEND"] = backend
        return backend

    def _connect_after_start(self, proc):
        # 3.5 s 内 Gazebo 已退出（坏 SDF 等）就不再连接，否则“已连接·等待相机帧”会盖掉退出原因。
        if self._sim_proc is proc and proc.poll() is None:
            self.connect_sim()

    def connect_sim(self):
        if self._sim_subs:
            self.disconnect_sim()
        partition = self.partition_var.get().strip()
        self._set_partition(partition or None)
        # 位姿话题跟着所选 world 里的载具走（x3 → /model/x3/pose，e1_mini → /model/e1_mini/pose）。
        self._vehicle = vehicle_from_sdf(self.world_var.get().strip())
        new_subs = {}
        try:
            new_subs["fpv"] = gz_io.TopicSub(CAMERA_TOPIC, buf=3)
            new_subs["chase"] = gz_io.TopicSub(CHASE_TOPIC, buf=3)
            new_subs["top"] = gz_io.TopicSub(TOP_TOPIC, buf=3)
            new_subs["pose"] = gz_io.TopicSub(pose_topic(self._vehicle), buf=8,
                                              stamp_fn=pose_stamp_ns)
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
        self._sim_frame_stamps = {"fpv": None, "chase": None, "top": None}
        self._sim_path.clear()
        self._sim_pose = None
        self._sim_last_pose_wall = None
        self.sim_info_var.set("等待 Gazebo 位姿数据")
        self._sim_status_var.set("已连接 Gazebo Transport · 等待相机帧")
        self._load_world_geometry()

    def disconnect_sim(self):
        for sub in self._sim_subs.values():
            try:
                sub.close()
            except Exception:
                pass
        self._sim_subs = {}
        self._sim_last_seq = {"fpv": 0, "chase": 0, "top": 0}
        self._sim_frame_stamps = {"fpv": None, "chase": None, "top": None}
        self._sim_pose = None
        self._sim_path.clear()
        self._sim_last_pose_wall = None
        try:
            self.sim_info_var.set("等待 Gazebo 位姿数据")
        except (tk.TclError, AttributeError):
            pass
        for key, label in (("fpv", self.fpv_label), ("chase", self.chase_label),
                           ("top", self.topcam_label)):
            try:
                label.configure(image="", text="视角已断开")
            except (tk.TclError, AttributeError):
                pass
            self._sim_photos.pop(key, None)
            status = self._cam_status_vars.get(key)
            if status is not None:
                status.set("视角已断开")
        if hasattr(self, "top_canvas"):
            self._draw_topdown()
        self._restore_partition()
        self._sim_stream_var.set("")
        self._sim_status_var.set("视角已断开")

    def _restore_partition(self):
        if self._sim_partition_active is not None:
            if self._sim_partition_old is None:
                os.environ.pop("GZ_PARTITION", None)
            else:
                os.environ["GZ_PARTITION"] = self._sim_partition_old
            self._sim_partition_active = None

    def stop_gazebo(self, grace_s=None):
        if self._sim_stop is not None:
            return  # 已在升级流程里，重复点击不重置计时
        if self._sim_proc is None or self._sim_proc.poll() is not None:
            self._sim_status_var.set("没有由 Studio 启动的 Gazebo 进程")
            return
        grace = SIM_STOP_GRACE_S if grace_s is None else grace_s
        # start_new_session=True，pgid == pid；领头进程退出后仍按这个组号收尾。
        pgid = self._sim_proc.pid
        self._signal_sim_group(pgid, signal.SIGINT)
        self._sim_stop = (pgid, 0, time.monotonic() + grace, grace)
        self._sim_status_var.set("已请求停止 Studio 启动的 Gazebo（SIGINT）")

    def _signal_sim_group(self, pgid, sig):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            pass
        except (PermissionError, OSError):
            if self._sim_proc is not None:
                try:
                    self._sim_proc.send_signal(sig)
                except OSError:
                    pass

    def _poll_sim_stop(self):
        """停止升级：SIGINT 宽限后 SIGTERM，再宽限后 SIGKILL。返回进程组是否还活着。"""
        if self._sim_stop is None:
            return False
        pgid, stage, deadline, grace = self._sim_stop
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            self._sim_stop = None
            self._sim_status_var.set("Studio 启动的 Gazebo 已停止")
            return False
        except PermissionError:
            pass
        if time.monotonic() < deadline:
            return True
        if stage == 0:
            self._signal_sim_group(pgid, signal.SIGTERM)
            self._sim_stop = (pgid, 1, time.monotonic() + grace, grace)
            self._sim_status_var.set("Gazebo 未响应 SIGINT，已发 SIGTERM")
        elif stage == 1:
            self._signal_sim_group(pgid, signal.SIGKILL)
            self._sim_stop = (pgid, 2, time.monotonic() + grace, grace)
            self._sim_status_var.set("Gazebo 未响应 SIGTERM，已发 SIGKILL")
        else:
            self._sim_stop = None
            self._sim_status_var.set(f"Gazebo 进程组 {pgid} SIGKILL 后仍在，请手动检查")
            return False
        return True

    def _stop_gazebo_blocking(self, grace_s=SIM_CLOSE_GRACE_S):
        """关窗/信号退出用：同一升级流程，但就地等完，窗口销毁后没有 tick 来收尾。"""
        if self._sim_stop is not None:  # 交互停止已在进行：沿用当前阶段，宽限缩短到关窗用的值
            pgid, stage, deadline, _ = self._sim_stop
            self._sim_stop = (pgid, stage, min(deadline, time.monotonic() + grace_s), grace_s)
        self.stop_gazebo(grace_s)
        while self._sim_stop is not None:
            if self._sim_proc is not None and self._sim_proc.poll() is not None:
                self._sim_proc = None
            if not self._poll_sim_stop():
                break
            time.sleep(0.05)

    def _show_camera_frame(self, key, msg):
        label = {"fpv": self.fpv_label, "chase": self.chase_label,
                 "top": self.topcam_label}[key]
        try:
            w, h, nch, rows = save_frame.decode(msg)
            # 不调 update_idletasks()：上一轮布局后的尺寸就够用，每帧强制布局是卡顿来源之一。
            # 未映射的 label 宽高是 1，此时不缩放，避免把首帧缩成 1 像素。
            lw, lh = label.winfo_width(), label.winfo_height()
            max_w = lw - 12 if lw > 12 else w
            max_h = lh - 12 if lh > 12 else h
            scale = min(1.0, max_w / w, max_h / h)
            if Image is not None and ImageTk is not None:
                image = Image.frombytes("RGB", (w, h), rows, "raw",
                                        "RGBX" if nch == 4 else "RGB")
                if scale < 0.999:
                    image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                                         getattr(Image, "Resampling", Image).BILINEAR)
                photo = ImageTk.PhotoImage(image)
            else:
                # 无 PIL（launch_studio.sh 默认的 /usr/bin/python3）：PPM 不压缩，Tk 直接拷贝像素。
                if nch == 4:
                    rgb = bytearray(w * h * 3)
                    for c in range(3):
                        rgb[c::3] = rows[c::4]
                    rows = bytes(rgb)
                photo = tk.PhotoImage(data=b"P6 %d %d 255\n" % (w, h) + rows, format="ppm")
                factor = max(1, math.ceil(max(w / max_w, h / max_h)))
                if factor > 1:  # 无 PIL 时只能整数降采样，避免超出面板被裁切
                    photo = photo.subsample(factor, factor)
            label.configure(image=photo, text="")
            self._sim_photos[key] = photo
            self._sim_frames_seen[key] += 1
            stamp = gz_io.header_stamp_ns(msg)
            if stamp:
                self._sim_frame_stamps[key] = stamp
        except Exception as exc:
            label.configure(text=f"图像解码失败\n{exc}", image="")
            self._sim_photos.pop(key, None)

    def _sim_tick(self):
        try:
            if self._sim_subs:
                now = time.monotonic()
                for key in ("fpv", "chase", "top"):
                    sub = self._sim_subs.get(key)
                    if not sub:
                        continue
                    frames = sub.frames()
                    if frames:
                        seq, msg = frames[-1]
                        # chase/顶视只是观感画面，限 10 Hz；机载画面是模型输入，每帧都显示。
                        throttled = (key != "fpv" and
                                     now - self._sim_last_shown.get(key, -1.0) < SIM_VIEW_PERIOD_S)
                        if seq != self._sim_last_seq[key] and not throttled:
                            self._sim_last_seq[key] = seq
                            self._sim_last_shown[key] = now
                            self._show_camera_frame(key, msg)
                pose_msg = self._sim_subs["pose"].latest()
                pose = parse_pose_msg(pose_msg, self._vehicle) if pose_msg else None
                if pose:
                    self._ingest_pose(pose)
                now_ns = self._sim_pose[4] if self._sim_pose else None
                pose_stale = (self._sim_pose is not None and
                              self._sim_last_pose_wall is not None and
                              time.monotonic() - self._sim_last_pose_wall > SIM_STALE_S)
                statuses = []
                for key in ("fpv", "chase", "top"):
                    sub = self._sim_subs.get(key)
                    stamp = self._sim_frame_stamps.get(key)
                    if pose_stale and stamp:
                        statuses.append(f"{key}:STALE")
                    elif not sub or not sub.healthy() or not stamp or now_ns is None:
                        statuses.append(f"{key}:WAIT")
                    else:
                        age = (now_ns - stamp) / 1e9
                        statuses.append(f"{key}:{'STALE %.1fs' % age if age > SIM_STALE_S else 'OK'}")
                    self._cam_status_vars[key].set(self._camera_age_text(key, now_ns))
                if self._sim_pose is None:
                    statuses.append("pose:WAIT")
                elif pose_stale:
                    statuses.append("pose:STALE")
                else:
                    statuses.append("pose:OK")
                self._sim_stream_var.set("Gazebo " + " · ".join(statuses))
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
            # 放在 poll() 之后：先回收领头进程，killpg(pgid, 0) 才不会把僵尸当成还活着。
            self._poll_sim_stop()
        except Exception:
            traceback.print_exc()
        finally:
            if not self._closing:
                self.root.after(SIM_TICK_MS, self._sim_tick)

    def _ingest_pose(self, pose):
        x, y, z, yaw, sim_ns = pose
        prev = self._sim_pose
        if prev is not None and pose == prev:
            return  # 同一条位姿消息：不追加轨迹，也不重画地图
        if prev is not None and 0 < sim_ns < prev[4]:
            # 仿真时间倒退 = 世界重置/重启：新一集的轨迹不能接到上一集末尾。
            self._sim_path.clear()
        if prev is None or sim_ns != prev[4]:
            self._sim_last_pose_wall = time.monotonic()
        self._sim_pose = pose
        if not self._sim_path or math.hypot(x - self._sim_path[-1][0], y - self._sim_path[-1][1]) > 0.005:
            self._sim_path.append((x, y))
        self.sim_info_var.set(
            f"Position  x={x:.2f} m  y={y:.2f} m  z={z:.2f} m    "
            f"Yaw={math.degrees(yaw):.1f}°    Sim time={sim_ns / 1e9:.2f} s    "
            f"Path points={len(self._sim_path)}")
        self._draw_topdown()

    def _camera_age_text(self, key, now_ns):
        stamp = self._sim_frame_stamps.get(key)
        if not stamp:
            return "等待相机数据…"
        if now_ns is None:
            return f"last sim {stamp / 1e9:.2f} s · 等待 pose"
        age = max(0.0, (now_ns - stamp) / 1e9)
        stale = age > SIM_STALE_S
        if (self._sim_last_pose_wall is not None and
                time.monotonic() - self._sim_last_pose_wall > SIM_STALE_S):
            stale = True
        return f"last sim {stamp / 1e9:.2f} s · age {age:.2f} s" + (" · STALE" if stale else "")

    def _map_bounds(self):
        """地图范围取整到米：轨迹每帧延伸几毫米时范围不变，静态层就不用重画。"""
        xs = [o["x"] for o in self._sim_obstacles]
        ys = [o["y"] for o in self._sim_obstacles]
        if self._sim_path:
            xs += (min(p[0] for p in self._sim_path), max(p[0] for p in self._sim_path))
            ys += (min(p[1] for p in self._sim_path), max(p[1] for p in self._sim_path))
        if self._sim_waypoint:
            xs.append(self._sim_waypoint[0]); ys.append(self._sim_waypoint[1])
        if self._sim_pose:
            xs.append(self._sim_pose[0]); ys.append(self._sim_pose[1])
        if not xs:
            xs, ys = [-1, 8], [-4, 4]
        return (math.floor(min(xs) - 1.5), math.ceil(max(xs) + 1.5),
                math.floor(min(ys) - 1.5), math.ceil(max(ys) + 1.5))

    def _map_world_note(self):
        """(文字, 颜色)：底图来自哪个 SDF，是否已核对为正在运行的世界。"""
        world = self._map_world
        if world is None:
            return "障碍物：未加载 SDF", "#FBBF24"
        name = f"{self._map_source}: {world.name}"
        running = (self._sim_proc is not None and self._sim_proc.poll() is None
                   and self._sim_world_path is not None)
        if running and world == self._sim_world_path and self._map_world_sha == self._sim_world_sha256:
            return f"{name} · 已核对（Studio 启动的世界）", "#34D399"
        return f"{name} · 未核对：不一定是正在运行的世界", "#FBBF24"

    def _draw_topdown(self):
        if not hasattr(self, "top_canvas"):
            return
        c = self.top_canvas
        w, h = max(1, c.winfo_width()), max(1, c.winfo_height())
        margin = 34
        xmin, xmax, ymin, ymax = self._map_bounds()
        scale = min((w - 2 * margin) / max(1.0, xmax - xmin),
                    (h - 2 * margin) / max(1.0, ymax - ymin))
        cx = (w - (xmax - xmin) * scale) / 2
        cy = (h - (ymax - ymin) * scale) / 2

        def screen(x, y):
            return cx + (x - xmin) * scale, h - cy - (y - ymin) * scale
        key = (w, h, xmin, xmax, ymin, ymax)
        if key != self._map_static_key:
            self._draw_map_static(c, w, h, xmin, xmax, ymin, ymax, screen)
            self._map_static_key = key
        # 动态层只改坐标/状态，不删重建（以前每 80 ms delete("all") 再画 3000 点平滑线）。
        if len(self._sim_path) > 1:
            c.coords("map_path", *[v for p in self._sim_path for v in screen(*p)])
            sx0, sy0 = screen(*self._sim_path[0])
            c.coords("map_start", sx0 - 4, sy0 - 4, sx0 + 4, sy0 + 4)
            c.itemconfigure("map_path", state="normal")
            c.itemconfigure("map_start", state="normal")
        else:
            c.itemconfigure("map_path", state="hidden")
            c.itemconfigure("map_start", state="hidden")
        if self._sim_pose:
            x, y, z, yaw, _ = self._sim_pose
            px, py = screen(x, y)
            length = max(18, min(35, scale * 0.55))
            c.coords("map_uav", px - 8, py - 8, px + 8, py + 8)
            c.coords("map_heading", px, py, px + math.cos(yaw) * length,
                     py - math.sin(yaw) * length)
            c.coords("map_uav_text", px + 12, py + 12)
            c.itemconfigure("map_uav_text", text=f"UAV z={z:.1f}m")
            for tag in ("map_uav", "map_heading", "map_uav_text"):
                c.itemconfigure(tag, state="normal")
        else:
            for tag in ("map_uav", "map_heading", "map_uav_text"):
                c.itemconfigure(tag, state="hidden")
        note, color = self._map_world_note()
        c.itemconfigure("map_world", text=note, fill=color)

    def _draw_map_static(self, c, w, h, xmin, xmax, ymin, ymax, screen):
        c.delete("all")
        for gx in range(int(xmin), int(xmax) + 1):
            x0, y0 = screen(gx, ymin)
            x1, y1 = screen(gx, ymax)
            c.create_line(x0, y0, x1, y1, fill="#1E293B")
            c.create_text(x0, h - 12, text=str(gx), fill="#64748B", font=("TkDefaultFont", 8))
        for gy in range(int(ymin), int(ymax) + 1):
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
        c.create_text(w - 12, 12, text="TOP VIEW · metres", fill="#94A3B8", anchor="ne")
        c.create_text(12, 12, text="", anchor="nw", tags="map_world",
                      font=("TkDefaultFont", 9))
        # 动态项先建好、隐藏，之后只用 coords() 移动；轨迹点间距只有毫米级，不用 smooth。
        c.create_line(0, 0, 0, 0, fill="#38BDF8", width=3, tags="map_path", state="hidden")
        c.create_oval(0, 0, 0, 0, fill="#34D399", outline="", tags="map_start", state="hidden")
        c.create_oval(0, 0, 0, 0, fill="#2563EB", outline="#BFDBFE", width=2, tags="map_uav",
                      state="hidden")
        c.create_line(0, 0, 0, 0, fill="#F8FAFC", width=3, arrow="last", arrowshape=(10, 12, 5),
                      tags="map_heading", state="hidden")
        c.create_text(0, 0, text="", fill="#93C5FD", anchor="nw", tags="map_uav_text",
                      state="hidden")

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
        # dynamics / window_from 按权重所在 run 的训练数据推断（N2）；推断不到就按 collect 默认值。
        protocol, _reason = self._ckpt_protocol(ckpt)
        if protocol is not None and protocol.get("renderer") != "3dgs":
            cmd += ["--dynamics", protocol["dynamics"]]
            if protocol.get("window_from"):
                cmd += ["--window-from", str(protocol["window_from"])]
        # 载具按所选 world 推断（connect_sim 已算出 self._vehicle）；collect.py 缺省只看 scene-meta 的
        # vehicle 字段，旧 metadata 没有它时 E1 世界会被当 x3，cf_sitl 直接报错。
        if self._vehicle != DEFAULT_VEHICLE:
            cmd += ["--vehicle", self._vehicle]
        env = os.environ.copy()
        if self._sim_partition_active and self._sim_partition_active != "default":
            env["GZ_PARTITION"] = self._sim_partition_active
        self._live_out, self._live_log = out, log
        self._live_config = {"checkpoint": str(ckpt), "scene_meta": str(meta_path),
                             "world": str(Path(self.world_var.get()).expanduser().resolve()),
                             "steps": steps, "hold_s": hold, "split": split,
                             "dynamics": protocol.get("dynamics") if protocol else None,
                             "window_from": protocol.get("window_from") if protocol else None,
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
        # 演示目录在 run_dir 下时，结束后只挂到这个 run（期间切换实验也不变）。
        self._live_owner = self.run_dir
        # 新一集从当前位置重新画轨迹，不接上一集（collect.py 不重置位姿，旧轨迹会连成一条）。
        self._sim_path.clear()
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
            owner = self._live_owner
            if owner and self._live_out.parent == owner:
                summary_path = owner / "studio_summary.json"
                run_summary = self._load_run_summary(owner)
                run_summary.setdefault("live_demos", []).append({
                    "output_dir": str(self._live_out), "steps": len(actions),
                    "collision_steps": collisions, "action_counts": counts,
                    "return_code": rc, "finished_at": summary["finished_at"]})
                write_json_atomic(summary_path, run_summary)
                write_experiment_report(owner, run_summary)
                self.refresh_history(force=True)
            self.live_action_var.set(f"学生策略演示结束 rc={rc} · steps={len(actions)} · collision_steps={collisions} · 结果：{self._live_out}")
        self.live_btn.configure(state="normal")
        self.live_stop_btn.configure(state="disabled")

    # --------------------------------------------------------------- evaluations
    def start_closed_loop_eval(self, auto=None):
        """自动录像传 auto=params（跳过确认对话框）；手动调用 auto=None，行为不变。"""
        if self.eval_proc is not None:
            return False
        if auto is None and ((self._sim_proc is not None and self._sim_proc.poll() is None)
                             or self._sim_subs):
            proceed = messagebox.askyesno(
                "闭环评测会启动独立仿真",
                "正式闭环评测会启动独立的 headless Gazebo 实例，并与当前桌面仿真共享 GPU。\n\n仍要继续吗？")
            if not proceed:
                return False
        if not self.run_dir:
            if auto is None:
                messagebox.showerror("没有实验", "请先加载一个训练 run。")
            return False
        ckpt = self.run_dir / "ckpt_best.bin"
        if not ckpt.is_file():
            if auto is None:
                messagebox.showerror("缺少 best checkpoint", f"找不到：{ckpt}")
            return False
        if auto is None:
            try:
                episodes = int(self.eval_episodes_var.get())
                if episodes < 1 or episodes > 500:
                    raise ValueError
            except ValueError:
                messagebox.showerror("参数错误", "Episodes 必须在 1–500 之间。")
                return False
            task = self.eval_task_var.get()
            split = self.eval_split_var.get()
            # 执行协议（dynamics / hold / window_from）从训练数据推断，与自动录像同一规则（N2）。
            # 不用 self.run_meta 兜底：它只在本 Studio 启动训练时赋值，load_run 不更新，可能是上一个 run 的。
            protocol, reason = run_eval_protocol(self.run_dir)
            if protocol is not None and protocol.get("renderer") == "3dgs":
                messagebox.showerror("数据来自 3DGS",
                                     f"该 run 的训练数据是 3DGS（{protocol.get('dynamics')}），"
                                     "Gazebo 闭环评测不适用；请用 sim/3dgs/eval_3dgs.py。")
                return False
            if protocol is None and not messagebox.askyesno(
                    "无法推断执行协议",
                    f"{reason}。\n\n将按 eval_closed_loop 默认值（x3_velocity、hold 1.0、"
                    "observation）评测；训练数据若不是这套协议，结果不可比。仍要继续吗？"):
                return False
            if protocol is not None and protocol.get("task") != task and not messagebox.askyesno(
                    "Task 与训练数据不一致",
                    f"界面选的 Task={task}，训练数据推断为 {protocol.get('task')}。\n\n仍要继续吗？"):
                return False
        out = self.run_dir / ("closed_loop_eval_" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        out.mkdir(parents=True, exist_ok=False)
        side = Path(str(out) + ".monitor")
        side.mkdir(parents=True, exist_ok=True)
        log = (side / "eval.log").open("ab", buffering=0)
        if auto is None:
            cmd = [PYTHON_FOR_SIM, str(SIM_TOOLS / "eval_closed_loop.py"), "--policy", "student",
                   "--ckpt", str(ckpt), "--episodes", str(episodes), "--task", task,
                   "--split", split, "--out", str(out), "--text-mode", "keep"]
            if task == "campus":
                cmd += ["--appearance", self.eval_appearance_var.get()]
            dynamics = protocol["dynamics"] if protocol is not None else None
            if task == "forest_e1":
                # forest_e1 只有 E1 载具 + cf_sitl（eval_closed_loop 要 --density）；训练数据推断不到协议
                # 时也按这套跑，不能退回 x3_velocity。
                cmd += ["--density", self.eval_density_var.get()]
                dynamics = dynamics or "cf_sitl"
            if dynamics:
                cmd += ["--dynamics", dynamics]
            if protocol is not None:
                if protocol.get("window_from"):
                    cmd += ["--window-from", str(protocol["window_from"])]
                if protocol.get("hold_s") is not None:
                    cmd += ["--hold", repr(float(protocol["hold_s"]))]
            # cf_sitl 的 eval_closed_loop 要显式 --vehicle e1_mini（resolve_vehicle 不从 dynamics 反推）
            vehicle = collect_parallel.DYNAMICS_VEHICLE.get(dynamics)
            if vehicle:
                cmd += ["--vehicle", vehicle]
        else:
            cmd = video_cli.build_eval_command(
                PYTHON_FOR_SIM, SIM_TOOLS / "eval_closed_loop.py", ckpt, out, auto)
            if auto.get("renderer") != "3dgs":
                cmd += ["--text-mode", "keep"]  # 自动录像不做文本消融
        if auto is None or auto.get("renderer") != "3dgs":
            self._apply_render_backend()  # 评测自己起的 Gazebo 与仿真页用同一渲染后端
        env = os.environ.copy()
        # The evaluator owns its own isolated GZ_PARTITIONs; do not inherit the GUI demo partition.
        env.pop("GZ_PARTITION", None)
        self.eval_out, self.eval_log = out, log
        owner = self.run_dir
        try:
            self.eval_proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env,
                                              stdin=subprocess.DEVNULL, stdout=log,
                                              stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.close()
            self.eval_log = None
            if auto is not None:
                self._eval_auto = False
                self._auto_eval_note(f"自动录像启动失败：{exc}", owner)
            else:
                messagebox.showerror("评测启动失败", str(exc))
            return False
        self._eval_owner = owner
        self.eval_btn.configure(state="disabled")
        self.eval_stop_btn.configure(state="normal")
        self._eval_status_var.set(
            f"{'闭环录像评测' if auto is not None else '闭环评测'}运行中 · {out.name}")
        return True

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

    def _load_run_summary(self, run_dir):
        """读 run 的 studio_summary.json；没有时按该 run 自己的 job/lifecycle 建一份。

        以前缺摘要时一律写 finished / return_code=0：评测或演示在训练中途结束，
        就把仍在训练的 run 标成已完成。现在只用该 run 自己记录的状态，不编造。
        """
        run_dir = Path(run_dir)
        try:
            summary = json.loads((run_dir / "studio_summary.json").read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                return summary
        except (OSError, ValueError, RecursionError):
            pass
        sidecar = Path(str(run_dir) + ".monitor")
        try:
            job = json.loads((sidecar / "job.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            job = {}
        try:
            life = json.loads((sidecar / "lifecycle.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            life = {}
        if not isinstance(job, dict):
            job = {}
        state = life.get("status") if isinstance(life, dict) else None
        if not state:
            state = unobserved_state(run_dir)
        return summarize_run(run_dir, job or {"run_id": run_dir.name},
                             life.get("return_code") if isinstance(life, dict) else None, state)

    def _attach_eval_to_run(self, run_dir, out, report, return_code=None):
        """评测结果只写进启动它的 run（run_dir），与结束时界面上加载的 run 无关。"""
        if not run_dir or not report:
            return
        run_dir = Path(run_dir)
        summary_path = run_dir / "studio_summary.json"
        summary = self._load_run_summary(run_dir)
        entries = summary.setdefault("closed_loop_evaluations", [])
        video_dir = out / "videos" if out else None
        videos = (sorted(str(p) for p in video_dir.iterdir() if p.suffix in VIDEO_SUFFIXES)
                  if video_dir is not None and video_dir.is_dir() else [])
        entry = {"task": report.get("task"), "split": report.get("split"),
                 "success_rate": report.get("success_rate"),
                 "requested_success_rate": report.get("requested_success_rate"),
                 "completion_rate": report.get("completion_rate"),
                 "failed_count": report.get("failed_count"),
                 "collision_rate": report.get("collision_rate"),
                 "stuck_rate": report.get("stuck_rate"),
                 "false_forward_rate": report.get("false_forward_rate"),
                 "episodes_done": report.get("episodes_done"),
                 "episodes_requested": report.get("episodes_requested"),
                 "unique_maps": report.get("unique_maps", report.get("unique_geometry")),
                 "success_criterion": report.get("success_criterion"),
                 "window_from": report.get("window_from"),
                 "dynamics": report.get("dynamics"),
                 "ckpt_sha256": report.get("ckpt_sha256"),
                 "eval_json": str(out / "eval.json") if out else None,
                 "video_dir": str(video_dir) if video_dir is not None and video_dir.is_dir() else None,
                 "videos": videos,
                 "return_code": return_code, "finished_at": now_iso()}
        entries.append(entry)
        write_json_atomic(summary_path, summary)
        write_experiment_report(run_dir, summary)
        self.refresh_history(force=True)
        self.refresh_run(force=True)

    def _finish_closed_loop_eval(self, rc):
        auto, owner, out = self._eval_auto, self._eval_owner, self.eval_out
        self._eval_auto = False
        self._eval_owner = None
        self.eval_proc = None
        if self.eval_log:
            self.eval_log.close()
            self.eval_log = None
        self.eval_btn.configure(state="normal")
        self.eval_stop_btn.configure(state="disabled")
        report = None
        if out and (out / "eval.json").is_file():
            try:
                report = json.loads((out / "eval.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        run_name = owner.name if owner else "—"
        if report:
            self._attach_eval_to_run(owner, out, report, rc)
            if auto and rc != 0:
                self._auto_eval_note(
                    f"自动录像评测 rc={rc}：部分或全部集失败，已附加可用结果", owner)
            self._eval_status_var.set(
                f"闭环评测结束（{run_name}）rc={rc} · success={report.get('success_rate')} · "
                f"requested={report.get('requested_success_rate')} · "
                f"collision={report.get('collision_rate')} · "
                f"episodes={report.get('episodes_done')}/{report.get('episodes_requested')} · "
                f"completion={report.get('completion_rate')} · "
                f"failed={report.get('failed_count')} · "
                f"unique_maps={report.get('unique_maps', report.get('unique_geometry'))}")
        else:
            if auto:
                self._auto_eval_note(
                    f"自动录像评测失败 rc={rc}：未生成 eval.json（见 {out}）", owner)
            self._eval_status_var.set(f"闭环评测结束（{run_name}）rc={rc} · 请检查 {out}")

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
            if summary_is_final(summary_path):
                continue
            sidecar = Path(str(run_dir) + ".monitor")
            life_path = sidecar / "lifecycle.json"
            job_path = sidecar / "job.json"
            try:
                life = json.loads(life_path.read_text(encoding="utf-8"))
                job = json.loads(job_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if life.get("status") not in LIVE_STATES:
                continue
            if life.get("status") == "starting" and not life.get("pid"):
                # 启动窗口里还没有 trainer PID。另一个 Studio 这时写摘要会让 run 目录非空，
                # RunOwner 随即拒绝训练；只有确认启动它的 Studio 已不在才回收。
                studio_pid = life.get("studio_pid")
                if studio_pid is None or _pid_exists(studio_pid):
                    continue  # 旧格式没有启动者身份，或启动者还活着：不动
            elif self._run_pid_alive(life.get("pid") or job.get("pid"), run_dir):
                continue
            state = unobserved_state(run_dir)
            summary = summarize_run(run_dir, job, None, state)
            summary["notes"].append("The desktop monitor was closed while the child process ran; process exit code was not observed by Studio.")
            carry_attachments(summary_path, summary)
            write_json_atomic(summary_path, summary)
            write_experiment_report(run_dir, summary)
            write_json_atomic(life_path, {"status": state, "return_code": None,
                                          "finished_at": summary["finished_at"],
                                          "recovered_by_studio": True})

    @staticmethod
    def _rel_path(path: Path):
        try:
            return str(Path(path).resolve().relative_to(DATA_ROOT))
        except (ValueError, OSError):
            return str(path)

    def _run_metrics(self, run_dir):
        """历史 run 的轻量指标：优先 studio_summary，否则只读 metrics.jsonl 尾部。"""
        summary_path = run_dir / "studio_summary.json"
        if summary_path.is_file():
            try:
                s = json.loads(summary_path.read_text(encoding="utf-8"))
                m = s.get("metrics", {}) or {}
                return (m.get("epochs_completed"), m.get("test_macro_f1"),
                        m.get("test_balanced_accuracy"), s.get("status", "—"),
                        s.get("started_at"))
            except (OSError, ValueError, RecursionError, TypeError):
                pass
        epochs = len(read_training_rows(run_dir))
        tests = [e for e in _tail_jsonl(run_dir / "metrics.jsonl")
                 if e.get("event") == "validation" and e.get("split") == "test"]
        test = tests[-1] if tests else {}
        return (epochs or None, test.get("macro_f1"), test.get("balanced_accuracy"),
                "—", None)

    def _scan_runs(self, force=False):
        """扫描 DATA_ROOT/uav_rgb（两层）并缓存；_tick 每秒刷新绝不触发这里。"""
        root = DATA_ROOT / "uav_rgb"
        sig = []
        try:
            sig.append((str(root), root.stat().st_mtime_ns))
            for top in sorted(root.iterdir()):
                if top.is_dir():
                    sig.append((top.name, top.stat().st_mtime_ns))
        except OSError:
            sig.append(("missing", 0))
        sig = tuple(sig)
        # shortcut: 深层 eval.json 内容变化不一定改动顶层 mtime；刷新按钮会 force 重扫。
        if not force and self._scan_cache is not None and self._scan_sig == sig:
            return self._scan_cache
        runs = discover_runs([root])
        index, standalone, stale = eval_index([root])
        entries = []
        run_set = {p.resolve() for p in runs}
        for run_dir in runs:
            resolved = run_dir.resolve()
            evals = list(index.get(resolved, []))
            own = run_dir / "eval.json"
            if own.is_file() and own not in evals:
                try:
                    report = json.loads(own.read_text(encoding="utf-8"))
                except (OSError, ValueError, RecursionError):
                    report = None
                if not isinstance(report, dict) or eval_ckpt_state(resolved, report) != "stale":
                    evals.append(own)
            has_training = any((run_dir / name).is_file()
                               for name in ("metrics.jsonl", "results.csv", "studio_summary.json"))
            kind = "训练" if has_training else "独立评测"
            epochs, macro, bal, status, started = self._run_metrics(run_dir)
            if not started:
                try:
                    started = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                except OSError:
                    started = "—"
            entries.append({"path": run_dir, "kind": kind, "evals": evals,
                            "status": status, "epochs": epochs, "macro": macro,
                            "bal": bal, "sr": sr_text(evals), "started": started,
                            **eval_fields_for_latest(evals)})
        for ev in standalone:
            # 已在 run 行里出现过的目录不再重复列。
            if ev.parent.resolve() in run_set:
                continue
            try:
                report = json.loads(ev.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                continue
            if not isinstance(report, dict):
                continue
            entries.append({"path": ev, "kind": "独立评测", "evals": [ev], "status": "—",
                            "epochs": None, "macro": None, "bal": None,
                            "sr": sr_text([ev]),
                            "started": datetime.fromtimestamp(ev.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                            **eval_display_fields(report)})
        for ev in stale:
            # ckpt_sha256 与它评测的那份权重当前内容不一致：单独列出，不挂到新 run。
            try:
                report = json.loads(ev.read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                continue
            if not isinstance(report, dict):
                continue
            entries.append({"path": ev, "kind": "旧评测", "evals": [ev],
                            "status": "旧 checkpoint/旧评测",
                            "epochs": None, "macro": None, "bal": None,
                            "sr": sr_text([ev]),
                            "started": datetime.fromtimestamp(ev.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                            **eval_display_fields(report)})
        def _mtime(entry):
            try:
                return entry["path"].stat().st_mtime
            except OSError:
                return 0.0
        entries.sort(key=_mtime, reverse=True)
        self._scan_sig = sig
        self._scan_cache = entries
        self._eval_index = index
        self._standalone_evals = standalone
        return entries

    def refresh_history(self, force=False):
        """扫描放后台线程（真实数据盘约 1 s）；Tk 控件只在主线程的 _apply_history 里改。

        扫描进行中再次请求时只记一笔，结束后补扫一次，不并发开多个线程。
        """
        if not hasattr(self, "history_tree"):
            return
        # DATA_ROOT 配错时不能让界面启动即崩。
        try:
            RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            if messagebox is not None:
                messagebox.showerror("数据目录不可用",
                                     f"无法创建 {RUNS_ROOT}：\n{exc}\n\n"
                                     "请检查 DATA_ROOT 环境变量是否指向可写目录。")
            return
        self._recover_detached_runs()
        if self._history_thread is not None:
            self._history_again = True
            return
        box = {}

        def work():
            try:
                box["entries"] = self._scan_runs(force)
            except Exception:  # noqa: BLE001 - 线程里的异常要回到主线程打印
                box["error"] = traceback.format_exc()
        self._history_thread = threading.Thread(target=work, name="studio-history-scan",
                                                daemon=True)
        self._history_thread.start()
        self.root.after(HISTORY_POLL_MS, lambda: self._poll_history(box))

    def _poll_history(self, box):
        thread = self._history_thread
        if thread is not None and thread.is_alive():
            if not self._closing:
                self.root.after(HISTORY_POLL_MS, lambda: self._poll_history(box))
            return
        self._history_thread = None
        if self._closing:
            return
        if "error" in box:
            sys.stderr.write(box["error"])
        else:
            self._apply_history(box.get("entries") or [])
        if self._history_again:
            self._history_again = False
            self.refresh_history(force=True)

    def _apply_history(self, entries):
        self.history_tree.delete(*self.history_tree.get_children())
        for item in entries[:300]:
            path = item["path"]
            self.history_tree.insert(
                "", "end", iid=str(path),
                values=(self._rel_path(path if path.name != "eval.json" else path.parent),
                        item["kind"], item["status"] or "—",
                        item["task"], item["criterion"], item["window_from"],
                        item["dynamics"],
                        item["epochs"] if item["epochs"] is not None else "—",
                        f"{item['macro']:.4f}" if isinstance(item["macro"], (int, float)) else "—",
                        f"{item['bal']:.4f}" if isinstance(item["bal"], (int, float)) else "—",
                        item["sr"], item["started"] or "—"))
        # SR 卡片读 _eval_index；扫描在后台完成后这里才有新索引（例如刚归档的评测）。
        if hasattr(self, "card_vars") and self.run_dir:
            self._update_sr_card()

    def _pick_eval(self, run_dir):
        """按任务约定：closed_loop_eval_* 最新 mtime > run_dir/eval.json > 反查索引。

        ckpt_sha256 与当前 ckpt_best.bin 不一致的旧评测不自动选中；它们仍在
        Experiments 表里以“旧评测”单独列出。
        """
        run_dir = Path(run_dir)
        def current(path):
            try:
                report = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError, RecursionError):
                return False
            return isinstance(report, dict) and eval_ckpt_state(run_dir, report) != "stale"
        candidates = sorted(run_dir.glob("closed_loop_eval_*/eval.json"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            if current(path):
                return path
        own = run_dir / "eval.json"
        if own.is_file() and current(own):
            return own
        paths = [p for p in self._run_eval_paths(run_dir) if current(p)]
        return max(paths, key=lambda p: p.stat().st_mtime) if paths else None

    def load_selected_history(self, show_dashboard=True):
        sel = self.history_tree.selection()
        if not sel:
            return
        path = Path(sel[0])
        if path.name == "eval.json":
            self.load_run(path.parent)
            self._load_eval_report(path)
        else:
            self.load_run(path)
            ev = self._pick_eval(path)
            if ev:
                self._load_eval_report(ev)
        if show_dashboard:
            self.tabs.select(self.dashboard_tab)

    @staticmethod
    def _format_actions(actions):
        if not isinstance(actions, dict):
            return "—"
        short = ("SN", "FS", "YL", "YR")
        counts = [0, 0, 0, 0]
        for key, value in actions.items():
            try:
                idx = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(counts):
                counts[idx] = value
        return " ".join(f"{short[i]}={counts[i]}" for i in range(4))

    def _load_eval_report(self, path):
        try:
            report = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError):
            return
        if not isinstance(report, dict):
            return
        self._eval_report = report
        episodes = [ep for ep in report.get("per_episode") or [] if isinstance(ep, dict)]
        self._eval_episodes = episodes
        total = len(episodes)
        results = [classify_episode(ep) for ep in episodes]
        # chip 用 eval.json 的原始布尔字段计数，与报告里的 collision_rate/
        # false_forward_rate 同口径；false_forward 同时满足 collision，互斥标签
        # 会把它从“碰撞”里扣掉，也会让“假前进”恒为 0。
        def rate(field, report_key):
            if total and all(field in ep for ep in episodes):
                return f"{sum(1 for ep in episodes if ep.get(field)) / total * 100:.0f}%"
            value = safe_float(report.get(report_key))
            return f"{value * 100:.0f}%" if value is not None else "未标注"
        has_success = total and all("success" in ep for ep in episodes)
        sr = safe_float(report.get("success_rate"))
        if sr is None and has_success:
            sr = sum(1 for ep in episodes if ep.get("success")) / total
        maps = self._report_maps(report)
        p95 = safe_float((report.get("inference_ms") or {}).get("p95"))
        if is_single_episode(report):
            self.eval_chip_vars["sr"].set("单集录像（非 SR）")
        else:
            self.eval_chip_vars["sr"].set(f"{sr * 100:.0f}%" if sr is not None else "未标注")
        self.eval_chip_vars["collision"].set(rate("collision", "collision_rate"))
        self.eval_chip_vars["stuck"].set(rate("stuck", "stuck_rate"))
        self.eval_chip_vars["false_forward"].set(rate("false_forward", "false_forward_rate"))
        flags = ("success", "collision", "stuck", "false_forward")
        if total and all(any(k in ep for k in flags) for ep in episodes):
            self.eval_chip_vars["unreached"].set(
                f"{results.count('未到达') / total * 100:.0f}%")
        else:
            self.eval_chip_vars["unreached"].set("未标注")
        self.eval_chip_vars["maps"].set(str(maps) if maps is not None else "未标注")
        self.eval_chip_vars["p95"].set(f"{p95:.1f} ms" if p95 is not None else "未标注")
        criterion = report.get("success_criterion") or "口径未标注"
        try:
            rel = self._rel_path(path)
        except (TypeError, ValueError):
            rel = str(path)
        self.eval_detail_box.configure(text=f"评测明细 · {rel} · {criterion}")
        self._fill_episode_table()

    def _fill_episode_table(self):
        if not hasattr(self, "eval_tree"):
            return
        self.eval_tree.delete(*self.eval_tree.get_children())
        choice = self.eval_filter_var.get()
        field_filter = {"碰撞": "collision", "卡住": "stuck", "假前进": "false_forward"}
        episodes = self._eval_episodes
        shown = 0
        for i, ep in enumerate(episodes, 1):
            result = classify_episode(ep)
            # 失败筛选用原始字段，避免“碰撞但假前进”被互斥标签挡在“碰撞”之外；
            # “未到达”仍是互斥分类里没有其它 flag 的集。
            if choice == "仅失败" and ep.get("success"):
                continue
            if choice == "成功" and not ep.get("success"):
                continue
            if choice in field_filter and not ep.get(field_filter[choice]):
                continue
            if choice == "未到达" and result != "未到达":
                continue
            shown += 1
            max_x = safe_float(ep.get("max_x"))
            wp_min = safe_float(ep.get("waypoint_min_dist"))
            self.eval_tree.insert(
                "", "end", tags=(result,),
                values=(i, ep.get("seed", "—"), result, ep.get("steps", "—"),
                        f"{max_x:.2f}" if max_x is not None else "—",
                        f"{wp_min:.2f}" if wp_min is not None else "—",
                        self._format_actions(ep.get("actions")),
                        ep.get("instruction", "—")))
        self.eval_count_var.set(f"显示 {shown}/{len(episodes)}")

    def open_selected_history(self):
        sel = self.history_tree.selection()
        if sel:
            self._open_path(Path(sel[0]).parent if Path(sel[0]).name == "eval.json"
                            else Path(sel[0]))

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
        self._val_events = []
        self._latest_train_event = None
        self._train_log_state = {}
        self._train_log_lines = 0
        self.train_log_text.delete("1.0", "end")
        self.refresh_run(force=True)
        best = path / "ckpt_best.bin"
        if best.is_file():
            self._set_live_ckpt(best)
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
        try:
            RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            messagebox.showerror("数据目录不可用", f"无法创建 {RUNS_ROOT}：\n{exc}")
            return
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
                if not summary_is_final(summary_path):
                    self._recover_detached_runs()
                    if summary_is_final(summary_path):
                        self.refresh_history(force=True)
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
                    self._finish_closed_loop_eval(rc)
            if self._live_proc is not None:
                self._update_live_action()
        except Exception:
            traceback.print_exc()
        finally:
            if not self._closing:
                self.root.after(self.refresh_ms, self._tick)

    def _on_close(self, interactive=True):
        """interactive=False 来自 SIGINT/SIGTERM：不弹框，后台任务照常继续（与选“否”相同），
        只收掉 Studio 自己启动的 Gazebo，避免留下孤儿 gz sim 占 GPU。"""
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
        if active and interactive:
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
        if (self._sim_proc is not None and self._sim_proc.poll() is None) or self._sim_stop:
            if not interactive or messagebox.askyesno("Gazebo 仍在运行", "是否停止由 Studio 启动的 Gazebo？"):
                self._stop_gazebo_blocking()
        self._closing = True
        self.root.destroy()

    def _request_quit(self, *_args):
        """信号处理里不直接碰 Tk：排进事件循环，走与关窗相同的收尾。"""
        if self._closing:
            return
        try:
            self.root.after(0, lambda: self._on_close(interactive=False))
        except tk.TclError:
            pass


def install_quit_signals(app):
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, app._request_quit)


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
    app = StudioApp(root, args.run_dir or None, int(args.refresh * 1000))
    # 终端 Ctrl+C / 注销 SIGTERM 以前直接抛 KeyboardInterrupt 退出，_on_close 不跑，Gazebo 成孤儿。
    install_quit_signals(app)
    root.mainloop()


if __name__ == "__main__":
    main()
