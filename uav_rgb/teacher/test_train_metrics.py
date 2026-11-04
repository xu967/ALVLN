#!/usr/bin/env python3
"""训练指标自测：macro-F1 必须能暴露「多数类把少数类吃掉」的情况。"""
import re

from train import (DATA_ROOT, RUNS_ROOT, _accuracy, _macro_f1, build_parser,
                   closed_loop_video, record_video_best_effort, resolve_out)

import train as train_mod


class _Args:
    video_task = "l1"
    video_split = "test"
    video_episodes = 1


def _check_parser_defaults():
    """--record-video 默认开，--video-episodes 默认 1；仍可用 --no-record-video 关闭。"""
    p = build_parser()
    args = p.parse_args(["--data", "/tmp/d", "--out", "run-x"])
    assert args.record_video is True, args.record_video
    assert args.video_episodes == 1, args.video_episodes
    off = p.parse_args(["--data", "/tmp/d", "--out", "run-x", "--no-record-video"])
    assert off.record_video is False, off.record_video
    custom = p.parse_args(["--data", "/tmp/d", "--out", "run-x",
                           "--video-episodes", "5"])
    assert custom.video_episodes == 5, custom.video_episodes
    print("parser defaults PASS")


def _check_video_command():
    """--record-video 必须把录像参数传给 eval，并落到 <out>/closed_loop_<UTCstamp>。"""
    calls = []
    old_run = train_mod.subprocess.run
    train_mod.subprocess.run = lambda cmd: calls.append(cmd) or type(
        "R", (), {"returncode": 0})()
    try:
        rc = closed_loop_video("/tmp/ckpt.pth", "/tmp/out", _Args())
    finally:
        train_mod.subprocess.run = old_run
    assert rc == 0 and len(calls) == 1
    cmd = calls[0]
    for flag in ("--record-video", "--teacher-ckpt", "--episodes"):
        assert flag in cmd, cmd
    out = cmd[cmd.index("--out") + 1]
    assert re.fullmatch(r"/tmp/out/closed_loop_\d{8}T\d{12}Z", out), out
    assert cmd[cmd.index("--episodes") + 1] == "1", cmd
    print("video command PASS")


def _check_video_failure_only_warns():
    """eval rc!=0 不能 sys.exit，也不能改变训练结果。"""
    old_run = train_mod.subprocess.run
    old_reason = train_mod.video_skip_reason
    train_mod.video_skip_reason = lambda: None
    train_mod.subprocess.run = lambda cmd: type("R", (), {"returncode": 7})()
    try:
        rc = record_video_best_effort("/tmp/ckpt.pth", "/tmp/out", _Args())
    finally:
        train_mod.subprocess.run = old_run
        train_mod.video_skip_reason = old_reason
    assert rc == 7, rc
    print("video failure warns PASS")


def _check_video_dependency_skip():
    """缺 gz / cv2 时告警跳过，不启动评测进程。"""
    old_which = train_mod.shutil.which
    old_video = train_mod.closed_loop_video
    calls = []
    train_mod.shutil.which = lambda name: None if name == "gz" else old_which(name)
    train_mod.closed_loop_video = lambda *a: calls.append(a) or 0
    try:
        rc = record_video_best_effort("/tmp/ckpt.pth", "/tmp/out", _Args())
    finally:
        train_mod.shutil.which = old_which
        train_mod.closed_loop_video = old_video
    assert rc is None, rc
    assert not calls, calls
    print("video dependency skip PASS")


def main():
    gts = [1] * 90 + [2] * 10          # 90% FORWARD，10% YAW_LEFT
    # 全猜 FORWARD：acc 很高，macro_f1 很低
    all_fwd = [1] * 100
    assert abs(_accuracy(gts, all_fwd) - 0.90) < 1e-9
    assert abs(_macro_f1(gts, all_fwd) - 0.4737) < 1e-3, _macro_f1(gts, all_fwd)

    # 完美预测：acc=1，macro_f1=1
    assert _accuracy(gts, gts) == 1.0
    assert abs(_macro_f1(gts, gts) - 1.0) < 1e-9

    # 一半 YAW 被漏判：acc 掉得少，macro_f1 掉得多
    half = [1] * 95 + [2] * 5
    assert abs(_accuracy(gts, half) - 0.95) < 1e-9
    # 少数类召回减半：acc 只掉 5 点，macro-F1 掉约 18 点
    assert abs(_macro_f1(gts, half) - 0.8198) < 1e-3, _macro_f1(gts, half)

    # 训练产物只能落数据盘：相对路径进 runs/，绝对路径越界或 .. 逃逸都拒绝
    assert resolve_out("teacher-x") == RUNS_ROOT / "teacher-x"
    assert resolve_out(str(DATA_ROOT / "uav_rgb" / "a")) == DATA_ROOT / "uav_rgb" / "a"
    for bad in ("/tmp/run", "/root/claude-code-backup-20261028/code/run", "../../../../tmp/x"):
        try:
            resolve_out(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} 应被拒绝")

    print("PASS")
    _check_parser_defaults()
    _check_video_command()
    _check_video_failure_only_warns()
    _check_video_dependency_skip()


if __name__ == "__main__":
    main()
