#!/usr/bin/env python3
"""训练学生模型后自动跑 1 集闭环录像；录像失败只告警，不把训练判失败。

用法：
  python3 tools/train_student_with_video.py --data <dataset.uavr> --out <run_dir> \
      [uavrgb_train 参数...]
  python3 tools/train_student_with_video.py --data <dataset.uavr> --out <run_dir> --dry-run

先跑 $UAV_RGB_BUILD_DIR/uavrgb_train（默认 <repo>/src/build/uavrgb_train），成功后
读 .uavr 尾部 meta 推断 task/split/dynamics，再跑
`eval_closed_loop.py --policy student --ckpt <out>/ckpt_best.bin --record-video
--episodes 1`，录像写到 <out>/closed_loop_eval_<UTCstamp>/videos/。
test 与 val 都为空、任务不确定或没有 best ckpt 时只警告跳过。

推断规则（studio.py 复用同一份实现）：
  task      task_version: campus.assets* -> campus_assets，campus* -> campus，
            l1* -> l1；没有 task_version 但 geometry_version 是 uavrgb.geom.* -> base
  split     meta.splits: test>0 用 test，否则 val>0 用 val，否则跳过
  dynamics  dynamics_profile: x3_v1* -> x3_velocity，cf_stab*/cf_sil*/cf_blackbox*
            原名；缺字段按旧基线 x3_velocity
"""
import argparse
import json
import os
import shlex
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL = REPO_ROOT / "sim" / "tools" / "eval_closed_loop.py"
UAVR_HEADER = struct.Struct("<4sHHIHHBBBBQQQI")
UAVR_MAGIC = b"UAVR"
UAVR_VERSION = 1


def read_uavr_meta(path):
    """只读 UAVR 文件尾部的 meta JSON；格式不对抛 ValueError/OSError。"""
    path = Path(path)
    with path.open("rb") as f:
        head = f.read(UAVR_HEADER.size)
        if len(head) != UAVR_HEADER.size:
            raise ValueError("文件太短，不是 UAVR")
        (magic, version, header_size, _n, _h, _w, _c, _idt, _ldt, _reserved,
         _images, _labels, meta_offset, meta_len) = UAVR_HEADER.unpack(head)
        if magic != UAVR_MAGIC or version != UAVR_VERSION or header_size != UAVR_HEADER.size:
            raise ValueError(f"UAVR 头不匹配（magic={magic!r} version={version}）")
        f.seek(0, os.SEEK_END)
        if meta_offset + meta_len > f.tell():
            raise ValueError("meta 偏移超出文件长度（文件被截断？）")
        f.seek(meta_offset)
        raw = f.read(meta_len)
    if len(raw) != meta_len:
        raise ValueError("meta 读取不完整")
    meta = json.loads(raw.decode("utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("UAVR meta 顶层必须是对象")
    return meta


def infer_task(meta):
    """task_version -> eval_closed_loop --task；无法确定返回 None。"""
    version = str(meta.get("task_version") or "")
    if version.startswith("campus.assets"):
        return "campus_assets"
    if version.startswith("campus"):
        return "campus"
    if version.startswith("l1"):
        return "l1"
    if version.startswith("base"):
        return "base"
    if version:
        return None
    # base 任务不写 task_version；只有柱阵几何版本才可确认是 base。
    if str(meta.get("geometry_version") or "").startswith("uavrgb.geom."):
        return "base"
    return None


def infer_split(meta):
    """test>0 用 test，否则 val>0 用 val；都空返回 None。"""
    splits = meta.get("splits") if isinstance(meta.get("splits"), dict) else {}

    def count(key):
        try:
            return int(splits.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    if count("test") > 0:
        return "test"
    if count("val") > 0:
        return "val"
    return None


def _dynamics_profile(meta):
    profile = meta.get("dynamics_profile")
    if not profile:
        for episode in meta.get("episodes") or []:
            if isinstance(episode, dict) and episode.get("dynamics_profile"):
                profile = episode["dynamics_profile"]
                break
    return str(profile) if profile else None


def infer_dynamics(meta):
    """dynamics_profile -> eval_closed_loop --dynamics；未知返回 None。

    旧数据集没有该字段，历史上只有 x3_velocity 基线，按它推断。
    """
    profile = _dynamics_profile(meta)
    if not profile:
        return "x3_velocity"
    if profile.startswith("x3_v1"):
        return "x3_velocity"
    for name in ("cf_stab", "cf_sitl", "cf_sil", "cf_blackbox"):
        if profile.startswith(name):
            return name
    return None


def auto_eval_params(meta):
    """从 .uavr meta 推导自动录像参数；不能确定时返回 (None, 原因)。"""
    dynamics = infer_dynamics(meta)
    if dynamics is None:
        return None, f"未知 dynamics_profile={_dynamics_profile(meta)!r}"
    renderer = ("3dgs" if (meta.get("renderer_backend") == "gsplat"
                           or dynamics == "cf_blackbox") else "gazebo")
    task = infer_task(meta)
    if task is None and renderer == "3dgs":
        task = "3dgs"  # eval_3dgs 不用 task；仅作元数据/UI 标记
    if task is None:
        return None, f"无法从 task_version={meta.get('task_version')!r} 推断任务"
    split = infer_split(meta)
    if split is None:
        return None, "数据集 test/val 都为空，无法闭环评测"
    window_from = meta.get("window_from")
    if window_from not in ("observation", "dispatch"):
        window_from = None
    hold_s = meta.get("hold_s")
    if not isinstance(hold_s, (int, float)) or isinstance(hold_s, bool) or hold_s <= 0:
        hold_s = None
    return {"task": task, "split": split, "dynamics": dynamics, "episodes": 1,
            "window_from": window_from, "hold_s": hold_s,
            "renderer": renderer}, None


def build_train_command(trainer, data, out, extra):
    return [str(trainer), "--data", str(data), "--out", str(out), *extra]


def _3dgs_python():
    return os.environ.get("UAVRGB_3DGS_PYTHON") or os.path.join(
        os.environ.get("DATA_ROOT", "/root/rivermind-data"),
        "uav_rgb", "3dgs-venv", "bin", "python")


def build_eval_command(python, eval_script, ckpt, out_dir, params):
    """自动录像评测命令；studio.py 的训练后自动评测复用同一构造。"""
    if params.get("renderer") == "3dgs":
        data = params.get("data")
        scene = params.get("scene")
        if not scene and data:
            scene = str(Path(data).parent / "scene.json")
        if not scene:
            raise ValueError("3DGS 自动评测需要 scene.json（数据集旁或 --scene）")
        cmd = [_3dgs_python(), "-m", "sim.3dgs.eval_3dgs",
               "--scene", scene, "--out", str(out_dir),
               "--ckpt", str(ckpt), "--episodes", "1", "--record-video"]
        if params.get("infer_bin"):
            cmd += ["--infer-bin", str(params["infer_bin"])]
        return cmd
    cmd = [str(python), str(eval_script), "--policy", "student",
           "--ckpt", str(ckpt), "--task", params["task"], "--split", params["split"],
           "--dynamics", params["dynamics"], "--episodes", str(params.get("episodes", 1)),
           "--record-video", "--out", str(out_dir)]
    if params["dynamics"] == "cf_sitl":
        # eval_closed_loop.resolve_vehicle：cf_sitl 必须显式 --vehicle，缺了直接 argparse 退出 2（REVIEW2_UIDOCS M1）
        cmd += ["--vehicle", "e1_mini"]
    if params.get("window_from"):
        cmd += ["--window-from", str(params["window_from"])]
    if params.get("hold_s") is not None:
        cmd += ["--hold", repr(float(params["hold_s"]))]
    if params["task"] == "campus":
        cmd += ["--appearance", str(params.get("appearance") or "v1")]
    return cmd


LEGACY_CORRIDOR_NOTE = "提示: 柱阵录像评测为旧口径，未带走廊（--corridor-half-width），且评 ckpt_best.bin，仅作诊断参考"


def _eval_out_dir(out):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(out) / f"closed_loop_eval_{stamp}"


def _eval_python():
    return os.environ.get("UAV_RGB_PYTHON") or sys.executable


def main(argv=None):
    p = argparse.ArgumentParser(
        description="uavrgb_train 训练后自动闭环录像 1 集（录像失败只告警）")
    p.add_argument("--data", required=True, help="训练数据集 .uavr")
    p.add_argument("--out", required=True, help="训练 run 目录（传给 uavrgb_train --out）")
    p.add_argument("--scene", default=None,
                   help="3DGS scene.json；缺省用 <data 所在目录>/scene.json")
    p.add_argument("--dry-run", action="store_true", help="只打印训练/评测命令，不执行")
    args, extra = p.parse_known_args(argv)

    data = Path(args.data).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not data.is_file():
        p.error(f"数据集不存在: {data}")
    try:
        meta = read_uavr_meta(data)
    except (OSError, ValueError) as exc:
        p.error(f"读取 .uavr meta 失败: {exc}")
    params, reason = auto_eval_params(meta)
    build_dir = Path(os.environ.get("UAV_RGB_BUILD_DIR") or (REPO_ROOT / "src" / "build"))
    trainer = build_dir.expanduser() / "uavrgb_train"
    if params is not None:
        params["data"] = str(data)
        if args.scene:
            params["scene"] = str(Path(args.scene).expanduser().resolve())
        elif params.get("renderer") == "3dgs":
            params["scene"] = str(data.parent / "scene.json")
        infer_bin = build_dir.expanduser() / "uavrgb_infer"
        if infer_bin.is_file():
            params["infer_bin"] = str(infer_bin)
    train_cmd = build_train_command(trainer, data, out, extra)
    eval_cmd = None
    eval_out = _eval_out_dir(out)
    if params is not None:
        try:
            eval_cmd = build_eval_command(_eval_python(), EVAL,
                                          out / "ckpt_best.bin", eval_out, params)
        except ValueError as exc:
            params, reason = None, str(exc)
    is_base_gz = bool(eval_cmd and params and params.get("renderer") != "3dgs" and params.get("task") == "base")
    if args.dry_run:
        print("训练: " + shlex.join(train_cmd))
        print("录像: " + (shlex.join(eval_cmd) if eval_cmd else f"跳过（{reason}）"))
        if is_base_gz:
            print(LEGACY_CORRIDOR_NOTE)
        return 0

    if not os.access(trainer, os.X_OK):
        print(f"train_student_with_video: 找不到可执行的 trainer: {trainer}\n"
              "先构建 src/build（或设置 UAV_RGB_BUILD_DIR）", file=sys.stderr)
        return 2
    rc = subprocess.run(train_cmd).returncode
    if rc != 0:
        return rc
    ckpt = out / "ckpt_best.bin"
    if not ckpt.is_file():
        print(f"警告: 训练结束但没有 {ckpt}，跳过录像", file=sys.stderr)
        return 0
    if eval_cmd is None:
        print(f"警告: 跳过闭环录像：{reason}", file=sys.stderr)
        return 0
    if params.get("renderer") != "3dgs" and shutil.which("gz") is None:
        print("警告: 跳过闭环录像：找不到 gz（Gazebo CLI 不在 PATH）", file=sys.stderr)
        return 0
    if params.get("renderer") == "3dgs":
        scene = Path(params.get("scene") or "")
        if not scene.is_file():
            print(f"警告: 跳过 3DGS 录像：找不到 scene.json（{scene}）", file=sys.stderr)
            return 0
        if not Path(_3dgs_python()).is_file():
            print(f"警告: 跳过 3DGS 录像：找不到 {_3dgs_python()}",
                  file=sys.stderr)
            return 0
    if is_base_gz:
        print(LEGACY_CORRIDOR_NOTE)
    try:
        rc = subprocess.run(eval_cmd).returncode
    except OSError as exc:
        print(f"警告: 闭环录像启动失败：{exc}", file=sys.stderr)
        return 0
    if rc != 0:
        print(f"警告: 闭环录像失败 rc={rc}；训练结果和 ckpt 已保留", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
