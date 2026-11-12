#!/usr/bin/env python3
"""tools/train_student_with_video.py 自检：meta 推断、命令构造、--dry-run 不执行。"""
import contextlib
import io
import json
import os
import shlex
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
    assert "--record-video" in cmd and "--allow-train-scene" in cmd and "--task" not in cmd, cmd
    # O-1：逐项锁里补上 --data（窗口来源）；原来锁的命令没有它，eval_3dgs 在 resolve_hold 就退出 2
    assert [t for t in cmd if t != "--allow-train-scene"] == [
        cli._3dgs_python(), "-m", "sim.3dgs.eval_3dgs",
        "--scene", "/d/ds/scene.json", "--out", "/run/closed_loop_eval_x",
        "--ckpt", "/run/ckpt_best.bin", "--episodes", "1", "--record-video",
        "--data", "/d/ds/dataset.uavr",
        "--infer-bin", "/build/uavrgb_infer"], cmd
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


def test_d1_3dgs_dry_run_and_gazebo_exact_lock():
    """D1：3DGS 自动录像 dry-run 命令行含 --allow-train-scene（Studio 共用同一 build_eval_command）；
    Gazebo 支路逐字锁死不变，不含 --allow-train-scene。"""
    assert cli.build_eval_command("/py", "/eval.py", "/r/ckpt_best.bin", "/r/ev", {
        "task": "base", "split": "test", "dynamics": "cf_sitl", "episodes": 1,
        "window_from": "observation", "hold_s": 1.0, "renderer": "gazebo"}) == [
        "/py", "/eval.py", "--policy", "student", "--ckpt", "/r/ckpt_best.bin",
        "--task", "base", "--split", "test", "--dynamics", "cf_sitl", "--episodes", "1",
        "--record-video", "--out", "/r/ev", "--vehicle", "e1_mini",
        "--window-from", "observation", "--hold", "1.0"]
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        gs_uavr = Path(d) / "gs.uavr"
        write_uavr(gs_uavr, {"renderer_backend": "gsplat", "splits": {"train": 2, "val": 1, "test": 0},
                             "dynamics_profile": "cf_sitl_v1:abc", "hold_s": 0.5})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["--data", str(gs_uavr), "--out", str(Path(d) / "run"), "--dry-run"])
        eval_line = [ln for ln in buf.getvalue().splitlines() if ln.startswith("录像:")][0]
        assert rc == 0 and "sim.3dgs.eval_3dgs" in eval_line and "--allow-train-scene" in eval_line, eval_line
    print("d1-3dgs-and-gazebo-lock PASS")


def test_o1_3dgs_eval_command_passes_eval_3dgs_argument_parsing():
    """O-1：3DGS 自动录像命令必须带 --data（或 --hold）。eval_3dgs 不再静默用 scene.step_s（AUDIT #12），
    两个都没有就在 resolve_hold 里 p.error 退出 2，训练后的 3DGS 录像每次都失败，只在真跑时才发现。
    这里把 build_eval_command 生成的命令里 `-m sim.3dgs.eval_3dgs` 之后的参数交给真实的 eval_3dgs.main() 解析：
    sys.argv 换成它，CfSitlParams.from_json 换成哨兵（参数解析与 resolve_hold 都过了才走到它），
    不加载高斯、不起渲染与 GPU。命令里没有 --collision-model，不依赖尚未合并的 p12-s1b2。"""
    import importlib
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    eval_mod = importlib.import_module("sim.3dgs.eval_3dgs")
    scene_mod = importlib.import_module("sim.3dgs.scene")

    class Reached(Exception):
        """eval_3dgs.main() 走过了参数解析与 resolve_hold。"""

    class StopHere:
        @classmethod
        def from_json(cls, _path):
            raise Reached()

    def parse_with_real_eval_3dgs(cmd):
        argv = cmd[cmd.index("sim.3dgs.eval_3dgs") + 1:]
        seen, real_resolve = [], eval_mod.resolve_hold

        def spy(hold_arg, data_arg):
            got = real_resolve(hold_arg, data_arg)
            seen.append(got)
            return got
        old_argv, old_resolve, old_params = sys.argv, eval_mod.resolve_hold, eval_mod.CfSitlParams
        sys.argv, eval_mod.resolve_hold, eval_mod.CfSitlParams = ["eval_3dgs.py", *argv], spy, StopHere
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                eval_mod.main()
        except Reached:
            return seen, None
        except SystemExit as exc:
            return seen, f"退出码 {exc.code}：{err.getvalue().strip().splitlines()[-1] if err.getvalue().strip() else ''}"
        finally:
            sys.argv, eval_mod.resolve_hold, eval_mod.CfSitlParams = old_argv, old_resolve, old_params
        return seen, "main() 没有走到 CfSitlParams.from_json"

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        d = Path(d)
        scene_mod.make_corridor(str(d / "scene"), seed=0, length=12.0)
        meta = {"renderer_backend": "gsplat", "splits": {"train": 2, "val": 1, "test": 0},
                "dynamics_profile": "cf_sitl_v1:abc", "hold_s": 0.5}
        data = d / "gs.uavr"
        write_uavr(data, meta)
        ckpt = d / "run" / "ckpt_best.bin"
        ckpt.parent.mkdir()
        ckpt.write_bytes(b"ck")
        params, reason = cli.auto_eval_params(meta)
        assert reason is None and params["renderer"] == "3dgs" and params["hold_s"] == 0.5, (params, reason)
        params.update(data=str(data), scene=str(d / "scene" / "scene.json"))

        # 1) 数据集是 .uavr：带 --data，eval_3dgs 自己从 meta 读 hold_s，来源写进产物
        cmd = cli.build_eval_command("/py", "/eval.py", ckpt, d / "ev", params)
        seen, problem = parse_with_real_eval_3dgs(cmd)
        assert problem is None, f"3DGS 录像命令过不了 eval_3dgs 的参数解析：{problem}\n{' '.join(cmd)}"
        assert cmd[cmd.index("--data") + 1] == str(data) and "--hold" not in cmd, cmd
        assert seen == [(0.5, "data uavr meta")], seen

        # 2) 数据集不是 .uavr（eval_3dgs 读不了）：退回 --hold，值取自 meta 的 hold_s
        odd = dict(params, data=str(d / "dataset.bin"))
        cmd = cli.build_eval_command("/py", "/eval.py", ckpt, d / "ev", odd)
        seen, problem = parse_with_real_eval_3dgs(cmd)
        assert problem is None and "--data" not in cmd and cmd[cmd.index("--hold") + 1] == "0.5", (problem, cmd)
        assert seen == [(0.5, "cli --hold")], seen

        # 3) 窗口来源一个都没有（没有 .uavr 数据集，meta 也没有 hold_s）：命令构造阶段就拒绝（main() 据此跳过并写原因），
        #    不留一条注定在 resolve_hold 退出 2 的命令
        try:
            cli.build_eval_command("/py", "/eval.py", ckpt, d / "ev", dict(params, data=None, hold_s=None))
        except ValueError as exc:
            assert "hold_s" in str(exc), exc
        else:
            raise AssertionError("没有窗口来源时 3DGS 评测命令必须在构造阶段拒绝")

        # 4) dry-run 打印的录像命令里带 --data，且 main() 走的就是同一构造
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["--data", str(data), "--out", str(d / "run2"), "--dry-run"])
        eval_line = [ln for ln in buf.getvalue().splitlines() if ln.startswith("录像:")][0]
        assert rc == 0 and f"--data {data}" in eval_line, eval_line
    print("o1-3dgs-eval-command-parsing PASS")


def test_o1_default_build_dir_is_build_check_like_other_tools():
    """O-1：不设 UAV_RGB_BUILD_DIR 时默认构建目录是 <repo>/build/check（tools/check.sh 的构建目录，其它工具同此），
    不是历史路径 src/build（里面是 11-04 的旧二进制，训练与评测会同流水线对不上）；设了环境变量就用环境变量。"""
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        uavr = Path(d) / "ds.uavr"
        write_uavr(uavr, {"geometry_version": "uavrgb.geom.v5", "splits": {"train": 2, "val": 1, "test": 1},
                          "dynamics_profile": "cf_sitl_v1:abc", "window_from": "observation", "hold_s": 1.0})

        def trainer_in_dry_run():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["--data", str(uavr), "--out", str(Path(d) / "run"), "--dry-run"])
            line = [ln for ln in buf.getvalue().splitlines() if ln.startswith("训练:")][0]
            assert rc == 0, buf.getvalue()
            return shlex.split(line[len("训练:"):])[0]

        old = os.environ.pop("UAV_RGB_BUILD_DIR", None)
        try:
            assert trainer_in_dry_run() == str(cli.REPO_ROOT / "build" / "check" / "uavrgb_train")
            os.environ["UAV_RGB_BUILD_DIR"] = str(Path(d) / "bin")
            assert trainer_in_dry_run() == str(Path(d) / "bin" / "uavrgb_train")
        finally:
            if old is None:
                os.environ.pop("UAV_RGB_BUILD_DIR", None)
            else:
                os.environ["UAV_RGB_BUILD_DIR"] = old
    print("o1-default-build-dir PASS")


def main():
    check_inference()
    check_commands()
    check_dry_run()
    test_legacy_corridor_note_for_base()
    test_d1_3dgs_dry_run_and_gazebo_exact_lock()
    test_o1_3dgs_eval_command_passes_eval_3dgs_argument_parsing()
    test_o1_default_build_dir_is_build_check_like_other_tools()
    print("test_train_student_with_video: PASS")


if __name__ == "__main__":
    main()
