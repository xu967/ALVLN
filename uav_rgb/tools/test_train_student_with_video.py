#!/usr/bin/env python3
"""tools/train_student_with_video.py 自检：meta 推断、命令构造、--dry-run 不执行。"""
import contextlib
import io
import json
import os
import struct
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_student_with_video as cli  # noqa: E402


def write_uavr(path, meta):
    """最小 UAVR：48 B 头 + meta JSON（0 个样本，只测尾部 meta 读取）。"""
    raw = json.dumps(meta).encode("utf-8")
    header = struct.pack("<4sHHIHHBBBBQQQI", b"UAVR", 1, 48, 0, 120, 160, 3,
                         1, 1, 0, 48, 48, 48, len(raw))
    Path(path).write_bytes(header + raw)


def check_inference():
    assert cli.infer_task({"task_version": "campus.assets.v1"}) == "campus_assets"
    assert cli.infer_task({"task_version": "campus.v2"}) == "campus"
    assert cli.infer_task({"task_version": "l1.v4"}) == "l1"
    assert cli.infer_task({"geometry_version": "uavrgb.geom.v5"}) == "base"
    assert cli.infer_task({}) is None
    assert cli.infer_task({"task_version": "unknown.v1"}) is None

    assert cli.infer_split({"splits": {"train": 1, "val": 2, "test": 3}}) == "test"
    assert cli.infer_split({"splits": {"train": 1, "val": 2, "test": 0}}) == "val"
    assert cli.infer_split({"splits": {"train": 1, "val": 0, "test": 0}}) is None
    assert cli.infer_split({}) is None

    assert cli.infer_dynamics({"dynamics_profile": "x3_v1"}) == "x3_velocity"
    assert cli.infer_dynamics({"dynamics_profile": "cf_stab_v1:tilt2:k1"}) == "cf_stab"
    assert cli.infer_dynamics({"dynamics_profile": "cf_sil_v1:abc123"}) == "cf_sil"
    assert cli.infer_dynamics({"dynamics_profile": "cf_blackbox_v1"}) == "cf_blackbox"
    # 旧数据集没有 dynamics_profile，历史基线是 x3_velocity。
    assert cli.infer_dynamics({}) == "x3_velocity"
    assert cli.infer_dynamics({"episodes": [{"dynamics_profile": "cf_stab_v1"}]}) == "cf_stab"
    assert cli.infer_dynamics({"dynamics_profile": "mystery"}) is None

    params, reason = cli.auto_eval_params({
        "task_version": "campus.v2", "splits": {"train": 1, "val": 0, "test": 1},
        "dynamics_profile": "x3_v1", "window_from": "dispatch", "hold_s": 1.0})
    assert reason is None
    assert params == {"task": "campus", "split": "test", "dynamics": "x3_velocity",
                      "episodes": 1, "window_from": "dispatch", "hold_s": 1.0,
                      "renderer": "gazebo"}, params

    gs, reason = cli.auto_eval_params({
        "task_version": None, "renderer_backend": "gsplat",
        "splits": {"train": 1, "val": 1, "test": 0},
        "dynamics_profile": "cf_blackbox_v1:abc", "hold_s": 0.5})
    assert reason is None and gs["renderer"] == "3dgs", gs
    assert gs["task"] == "3dgs" and gs["split"] == "val", gs

    skipped, reason = cli.auto_eval_params({
        "task_version": "campus.v2", "splits": {"train": 1, "val": 0, "test": 0}})
    assert skipped is None and "test/val" in reason, reason
    skipped, reason = cli.auto_eval_params({"splits": {"test": 1}})
    assert skipped is None and "task_version" in reason, reason
    print("inference PASS")


def check_commands():
    train = cli.build_train_command("/build/uavrgb_train", "/d.uavr", "/run",
                                    ["--epochs", "2"])
    assert train == ["/build/uavrgb_train", "--data", "/d.uavr", "--out", "/run",
                     "--epochs", "2"], train
    params = {"task": "campus", "split": "test", "dynamics": "cf_stab",
              "episodes": 1, "window_from": "dispatch", "hold_s": 0.2}
    cmd = cli.build_eval_command("/py", "/eval.py", "/run/ckpt_best.bin",
                                 "/run/closed_loop_eval_x", params)
    assert cmd[:4] == ["/py", "/eval.py", "--policy", "student"], cmd
    assert cmd[cmd.index("--ckpt") + 1] == "/run/ckpt_best.bin", cmd
    assert cmd[cmd.index("--task") + 1] == "campus", cmd
    assert cmd[cmd.index("--split") + 1] == "test", cmd
    assert cmd[cmd.index("--dynamics") + 1] == "cf_stab", cmd
    assert cmd[cmd.index("--episodes") + 1] == "1", cmd
    assert cmd[cmd.index("--window-from") + 1] == "dispatch", cmd
    assert cmd[cmd.index("--hold") + 1] == "0.2", cmd
    assert cmd[cmd.index("--appearance") + 1] == "v1", cmd
    assert "--record-video" in cmd and cmd[cmd.index("--out") + 1] == "/run/closed_loop_eval_x"
    assert "--vehicle" not in cmd, cmd   # 非 cf_sitl 不传，旧命令行逐字不变

    # cf_sitl（E1 MINI 固件在环）数据集：eval_closed_loop 要求显式 --vehicle e1_mini，否则 argparse 退出 2
    e1 = {"task": "base", "split": "test", "dynamics": "cf_sitl", "episodes": 1,
          "window_from": "observation", "hold_s": 1.0}
    cmd = cli.build_eval_command("/py", "/eval.py", "/run/ckpt_best.bin",
                                 "/run/closed_loop_eval_x", e1)
    assert cmd[cmd.index("--dynamics") + 1] == "cf_sitl", cmd
    assert cmd[cmd.index("--vehicle") + 1] == "e1_mini", cmd

    gs = {"task": "3dgs", "split": "val", "dynamics": "cf_blackbox",
          "episodes": 1, "renderer": "3dgs", "data": "/d/ds/dataset.uavr",
          "scene": "/d/ds/scene.json", "infer_bin": "/build/uavrgb_infer"}
    cmd = cli.build_eval_command("/py", "/eval.py", "/run/ckpt_best.bin",
                                 "/run/closed_loop_eval_x", gs)
    assert "sim.3dgs.eval_3dgs" in cmd, cmd
    assert cmd[cmd.index("--scene") + 1] == "/d/ds/scene.json", cmd
    assert cmd[cmd.index("--ckpt") + 1] == "/run/ckpt_best.bin", cmd
    assert cmd[cmd.index("--infer-bin") + 1] == "/build/uavrgb_infer", cmd
    assert "--record-video" in cmd and "--task" not in cmd, cmd
    print("commands PASS")


def check_dry_run():
    meta = {"task_version": "campus.v2", "geometry_version": "campus.v2",
            "splits": {"train": 1, "val": 0, "test": 1},
            "dynamics_profile": "x3_v1", "window_from": "dispatch", "hold_s": 1.0}
    with tempfile.TemporaryDirectory() as d:
        data = Path(d) / "ds.uavr"
        write_uavr(data, meta)
        out = Path(d) / "run"
        old_build = os.environ.get("UAV_RGB_BUILD_DIR")
        old_run = cli.subprocess.run
        os.environ["UAV_RGB_BUILD_DIR"] = str(Path(d) / "build")
        cli.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("--dry-run 不能执行命令"))
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["--data", str(data), "--out", str(out), "--dry-run"])
        finally:
            cli.subprocess.run = old_run
            if old_build is None:
                os.environ.pop("UAV_RGB_BUILD_DIR", None)
            else:
                os.environ["UAV_RGB_BUILD_DIR"] = old_build
        text = buf.getvalue()
        assert rc == 0, rc
        assert "训练:" in text and "uavrgb_train" in text and "--out" in text, text
        assert "录像:" in text and "--record-video" in text, text
        assert "--episodes 1" in text and "--dynamics x3_velocity" in text, text
        assert "--window-from dispatch" in text and "--hold 1.0" in text, text
        assert "closed_loop_eval_" in text, text

        # test/val 都空：只打印跳过原因，仍然 rc=0。
        empty = Path(d) / "empty.uavr"
        write_uavr(empty, {"task_version": "l1.v4",
                           "splits": {"train": 1, "val": 0, "test": 0}})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["--data", str(empty), "--out", str(out), "--dry-run"])
        assert rc == 0 and "跳过" in buf.getvalue() and "test/val" in buf.getvalue(), buf.getvalue()
        assert "旧口径，未带走廊" not in text, text
    print("dry-run PASS")


def test_legacy_corridor_note_for_base():
    """W1-15b 第 6 条：柱阵 base 评测打印「旧口径，未带走廊」提示，build_eval_command 与录像命令行本身不改。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        out = Path(d) / "run"
        base_uavr = Path(d) / "base.uavr"
        write_uavr(base_uavr, {"geometry_version": "uavrgb.geom.v5",
                               "splits": {"train": 1, "val": 0, "test": 1},
                               "dynamics_profile": "cf_sitl_v1:abc", "window_from": "observation", "hold_s": 1.0})
        buf = io.StringIO()
        old_run2 = cli.subprocess.run
        cli.subprocess.run = lambda *a, **k: (_ for _ in ()).throw(AssertionError("--dry-run 不能执行命令"))
        try:
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["--data", str(base_uavr), "--out", str(out), "--dry-run"])
        finally:
            cli.subprocess.run = old_run2
        base_dry = buf.getvalue()
        eval_line = [ln for ln in base_dry.splitlines() if ln.startswith("录像:")][0]
        assert rc == 0 and "旧口径，未带走廊" in base_dry and "--corridor-half-width" not in eval_line, base_dry

        # 真跑路径（替身 subprocess.run，不起训练器与 gz）同样输出该提示
        build = Path(d) / "build_exec"
        build.mkdir(parents=True, exist_ok=True)
        trainer = build / "uavrgb_train"
        trainer.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        trainer.chmod(0o755)
        ran_cmds = []

        def fake_subproc(cmd, **_kw):
            ran_cmds.append(list(cmd))
            if Path(cmd[0]).name == "uavrgb_train":
                out.mkdir(parents=True, exist_ok=True)
                (out / "ckpt_best.bin").write_bytes(b"ck")
            return type("R", (), {"returncode": 0})()

        old_env, old_which, old_sub = os.environ.get("UAV_RGB_BUILD_DIR"), cli.shutil.which, cli.subprocess.run
        os.environ["UAV_RGB_BUILD_DIR"] = str(build)
        cli.shutil.which = lambda name: "/fake/gz" if name == "gz" else old_which(name)
        cli.subprocess.run = fake_subproc
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["--data", str(base_uavr), "--out", str(out)])
        finally:
            cli.subprocess.run, cli.shutil.which = old_sub, old_which
            if old_env is None:
                os.environ.pop("UAV_RGB_BUILD_DIR", None)
            else:
                os.environ["UAV_RGB_BUILD_DIR"] = old_env
        assert rc == 0 and len(ran_cmds) == 2, (rc, ran_cmds)
        assert "旧口径，未带走廊" in buf.getvalue(), buf.getvalue()
        assert "--corridor-half-width" not in ran_cmds[1], ran_cmds[1]
    print("legacy-corridor-note PASS")


def main():
    check_inference()
    check_commands()
    check_dry_run()
    test_legacy_corridor_note_for_base()
    print("test_train_student_with_video: PASS")


if __name__ == "__main__":
    main()
