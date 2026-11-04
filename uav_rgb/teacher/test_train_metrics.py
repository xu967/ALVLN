#!/usr/bin/env python3
"""训练指标自测：macro-F1 必须能暴露「多数类把少数类吃掉」的情况。"""
from train import DATA_ROOT, RUNS_ROOT, _accuracy, _macro_f1, closed_loop_video, resolve_out


def _check_video_command():
    """--record-video 必须真的把录像参数传给 eval，并落到 <out>/closed_loop。"""
    class Args:
        video_task = "l1"
        video_split = "test"
        video_episodes = 1

    import train as train_mod
    from pathlib import Path
    calls = []
    old_run = train_mod.subprocess.run
    train_mod.subprocess.run = lambda cmd: calls.append(cmd) or type(
        "R", (), {"returncode": 0})()
    try:
        rc = closed_loop_video("/tmp/ckpt.pth", Path("/tmp/out"), Args())
    finally:
        train_mod.subprocess.run = old_run
    assert rc == 0 and len(calls) == 1
    cmd = calls[0]
    for flag in ("--record-video", "--teacher-ckpt", "--episodes"):
        assert flag in cmd, cmd
    assert cmd[cmd.index("--out") + 1] == "/tmp/out/closed_loop", cmd
    print("video command PASS")



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
    _check_video_command()


if __name__ == "__main__":
    main()
