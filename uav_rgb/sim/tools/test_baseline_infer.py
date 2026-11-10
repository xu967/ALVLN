#!/usr/bin/env python3
"""baselines/baseline_infer.py 自检（check.sh 按 sim/tools/test_*.py 收集，所以放在这一层）。

不依赖 torch 与权重：映射规则的纯函数、forward 与探针基线经真实 StudentPolicy 的协议往返、启动器静态检查。
有权重与 teacher-venv 时才跑真实基线的往返（缺什么就 SKIP 并打印原因）。
"""
import base64
import json
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.join(HERE, "baselines")
sys.path.insert(0, HERE)
sys.path.insert(0, BASELINES)
# 配置区的默认值要测的是文件里写的数，不是调用者 shell 里碰巧导出的覆盖值
for _k in ("BASELINE", "BASELINE_PREP", "BASELINE_WEIGHTS", "BASELINE_LOG", "YAW_STEP_DEG", "HOLD_S", "STEER_SIGN",
           "P_CRIT", "YAW_SCALE_PULP", "YAW_SCALE_DRONET", "FWD_STEP_M", "BASELINE_ROOT"):
    os.environ.pop(_k, None)
import baseline_infer as bi  # noqa: E402
from student import StudentPolicy  # noqa: E402

TEACHER_PY = "/root/rivermind-data/uav_rgb/teacher-venv/bin/python"   # 启动器写死的解释器；没有它就跳过真实基线往返
LAUNCHERS = {"launch_forward.sh": "forward", "launch_pulp_v3.sh": "pulp_v3", "launch_pulp_tiny.sh": "pulp_tiny",
             "launch_dronet.sh": "dronet", "launch_dronet_recal.sh": "dronet"}


def gz_msg(w, h, nch, data):
    fmt = {3: "RGB_INT8", 4: "RGBA_INT8"}[nch]
    return {"width": w, "height": h, "step": w * nch, "pixelFormatType": fmt, "data": base64.b64encode(data).decode()}


def const_msg(value, w=320, h=240, nch=3):
    return gz_msg(w, h, nch, bytes([value]) * (w * h * nch))


def write_exec(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    os.chmod(path, 0o755)


def sh_launcher(td, name="launch_test.sh"):
    """用当前解释器跑 baseline_infer.py 的临时启动器：测试不依赖 teacher-venv（forward 不 import torch）。"""
    path = os.path.join(td, name)
    write_exec(path, f'#!/bin/sh\nexec "{sys.executable}" "{os.path.join(BASELINES, "baseline_infer.py")}" "$@"\n')
    return path


def with_env(**kv):
    """临时设置环境变量（StudentPolicy 的子进程继承当前进程环境）；返回还原函数。"""
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update({k: str(v) for k, v in kv.items()})

    def restore():
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return restore


# ---------------------------------------------------------------- 纯函数
def test_map_steer_coll():
    m = bi.map_steer_coll
    thr = 12.5 / 90                                              # S=90 时转向门槛：半个 25° 步长
    assert m(thr - 1e-6, 0.0, 90) == bi.FWD and m(thr + 1e-6, 0.0, 90) == bi.LEFT
    assert m(-(thr - 1e-6), 0.0, 90) == bi.FWD and m(-(thr + 1e-6), 0.0, 90) == bi.RIGHT
    assert m(0.624, 0.0, 20) == bi.FWD and m(0.626, 0.0, 20) == bi.LEFT   # DroNet S=20：门槛 steer = 0.625
    assert m(0.5, 0.0, 90, sign=-1) == bi.RIGHT and m(-0.5, 0.0, 90, sign=-1) == bi.LEFT   # 符号翻转
    assert m(0.5, 1.0, 90) == bi.LEFT and m(-0.5, 1.0, 90) == bi.RIGHT   # 转向优先于刹停
    assert m(0.0, 0.7, 90) == bi.FWD and m(0.0, 0.7001, 90) == bi.SAFE   # 碰撞阈值 0.7：不超过才前进
    assert m(0.0, 0.3, 90, p_crit=0.2) == bi.SAFE and m(0.0, 0.2, 90, p_crit=0.2) == bi.FWD
    for bad in (float("nan"), float("inf"), float("-inf")):                # 输出坏了 = 不动
        assert m(bad, 0.0, 90) == bi.SAFE and m(0.0, bad, 90) == bi.SAFE and m(0.5, bad, 90) == bi.SAFE
    assert m(0.2, 0.0, 90, hold_s=0.5) == bi.FWD and m(0.2, 0.0, 90, hold_s=1.0) == bi.LEFT   # hold 与步长参与门槛
    assert m(0.2, 0.0, 90, yaw_step_deg=50.0) == bi.FWD


def test_map_velocity():
    v = bi.map_velocity
    assert v(1.0, 12.4) == bi.FWD and v(1.0, 12.6) == bi.LEFT and v(1.0, -12.6) == bi.RIGHT   # 转向优先于前进
    assert v(0.124, 0.0) == bi.SAFE and v(0.126, 0.0) == bi.FWD and v(0.0, 0.0) == bi.SAFE    # 0.125 m = 半个 0.25 m 步长
    assert v(0.3, 0.0, hold_s=0.2) == bi.SAFE and v(0.7, 0.0, hold_s=0.2) == bi.FWD           # hold 0.2 s 时位移按 0.2 s 算


def test_defaults_are_preregistered_values():
    """配置区默认值就是预注册的忠实映射参数；改它们必须同时改 README 与实验预注册。"""
    assert (bi.YAW_STEP_DEG, bi.HOLD_S, bi.P_CRIT, bi.YAW_SCALE_PULP, bi.YAW_SCALE_DRONET, bi.STEER_SIGN) == (25.0, 1.0, 0.7, 90.0, 20.0, 1.0)
    assert set(bi.REGISTRY) == {"forward", "pulp_v3", "pulp_tiny", "dronet"}


def export_env(text):
    """只解析启动器里 `export` 开头的行：注释里提到的变量名与数值不算配置。"""
    return dict(re.findall(r"(\w+)=(\S+)", " ".join(l for l in text.splitlines() if l.startswith("export "))))


def test_launchers_static():
    for fn, base in LAUNCHERS.items():
        path = os.path.join(BASELINES, fn)
        assert os.path.isfile(path) and os.access(path, os.X_OK), f"{fn} 缺失或不可执行"
        text = open(path, encoding="utf-8").read()
        env = export_env(text)
        assert env.get("BASELINE") == base and base in bi.REGISTRY, (fn, env)
        assert env.get("PYTHONDONTWRITEBYTECODE") == "1", fn
        assert "teacher-venv/bin/python" in text and "baseline_infer.py" in text, f"{fn} 必须用 teacher-venv 解释器启动 baseline_infer.py"
        assert float(env.get("STEER_SIGN", "1")) in (1.0, -1.0), fn
        if base in ("pulp_v3", "pulp_tiny", "forward"):          # 忠实臂不覆盖映射参数：用文件里的默认（S=90、阈值 0.7）
            assert not ({"P_CRIT", "YAW_SCALE_PULP", "YAW_SCALE_DRONET"} & set(env)), fn
    faithful = export_env(open(os.path.join(BASELINES, "launch_dronet.sh"), encoding="utf-8").read())
    assert "P_CRIT" not in faithful and "YAW_SCALE_DRONET" not in faithful, "dronet 忠实臂不能改 S=20、阈值 0.7"
    recal = export_env(open(os.path.join(BASELINES, "launch_dronet_recal.sh"), encoding="utf-8").read())
    assert 0.0 < float(recal["P_CRIT"]) < 1.0 and recal.get("YAW_SCALE_DRONET") == "90", "recal 臂：阈值取校准帧中位数、S=90"


# ---------------------------------------------------------------- 协议往返（真实 StudentPolicy）
PROBE = '''#!{py}
"""测试用探针适配器：动作 = 第一个像素的 R 值 % 4；raw 带回收到的帧形状、像素与总和，由 BASELINE_LOG 读出。"""
import sys
sys.path.insert(0, {baselines!r})
import numpy as np
import baseline_infer as bi


class Probe(bi.Adapter):
    def step(self, frames, prev_action):
        f = frames[0]
        return int(f[0, 0, 0]) % 4, {{"shape": list(f.shape), "px": f[0, 0].tolist(), "sum": int(f.astype(np.int64).sum()),
                                      "n_frames": len(frames)}}


bi.REGISTRY["probe"] = Probe
sys.exit(bi.main())
'''


def test_forward_roundtrip_via_student_policy():
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "stub.ckpt")
        with open(ckpt, "w", encoding="utf-8") as f:
            f.write("stub ckpt\n")
        restore = with_env(BASELINE="forward")
        try:
            policy = StudentPolicy(ckpt, sh_launcher(td), timeout_s=60.0)
            try:
                assert (policy.h, policy.w, policy.in_c, policy.prev_dim, policy.text_dim, policy.aux_classes) == (240, 320, 3, 0, 0, 0)
                for v in (0, 90, 255):
                    a, ms, probs = policy.act(const_msg(v))
                    assert a == bi.FWD and probs == [0.0, 1.0, 0.0, 0.0] and ms >= 0.0, (a, probs, ms)
                a, _, _ = policy.act(const_msg(10, nch=4))                   # RGBA 帧照常
                assert a == bi.FWD
                a, _, _ = policy.act(const_msg(10, w=640, h=480))           # 其它分辨率由 StudentPolicy 缩到 320x240
                assert a == bi.FWD and policy.frames == 5
                policy.reset()
            finally:
                policy.close()                                               # 退出码非 0 会抛 RuntimeError
        finally:
            restore()


def test_missing_or_bad_baseline_name_fails_loudly():
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "stub.ckpt")
        open(ckpt, "w").close()
        for env, needle in (({}, "BASELINE"), ({"BASELINE": "da2_rule"}, "da2_rule")):
            restore = with_env(**env)
            try:
                StudentPolicy(ckpt, sh_launcher(td), timeout_s=30.0)
            except RuntimeError as exc:
                assert needle in str(exc), exc                               # stderr 原因要带进异常，评测日志里能看出来
            else:
                raise AssertionError(f"{env} 应当在握手阶段失败")
            finally:
                restore()


def test_probe_bytes_and_actions_survive_the_pipe():
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "stub.ckpt")
        open(ckpt, "w").close()
        probe_bin = os.path.join(td, "probe_bin.py")
        write_exec(probe_bin, PROBE.format(py=sys.executable, baselines=BASELINES))
        log = os.path.join(td, "probe.jsonl")
        restore = with_env(BASELINE="probe", BASELINE_LOG=log)
        try:
            policy = StudentPolicy(ckpt, probe_bin, timeout_s=60.0)
            try:
                got = [policy.act(const_msg(v))[0] for v in (1, 2, 3, 4)]
                assert got == [1, 2, 3, 0], got                             # 动作 0..3 原样回到调用方
                policy.act(const_msg(7, nch=4))                             # RGBA → RGB
                policy.act(const_msg(9, w=640, h=480))                      # 缩放到 320x240
            finally:
                policy.close()
        finally:
            restore()
        rows = [json.loads(l) for l in open(log, encoding="utf-8")]
        assert rows[0]["sum"] == 0 and rows[0]["n_frames"] == 1             # 第 0 行是 StudentPolicy 的全零预热帧
        for row, v in zip(rows[1:], (1, 2, 3, 4, 7, 9)):
            assert row["shape"] == [240, 320, 3] and row["px"] == [v, v, v] and row["sum"] == v * 240 * 320 * 3, (row, v)


def test_eof_and_truncated_frame_exit_codes():
    env = dict(os.environ, BASELINE="forward")
    exe = [sys.executable, os.path.join(BASELINES, "baseline_infer.py"), "--ckpt", "x"]
    p = subprocess.Popen(exe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    hello = json.loads(p.stdout.readline())
    assert hello["num_classes"] == 4 and (hello["in_h"], hello["in_w"], hello["in_c"]) == (240, 320, 3)
    p.stdin.close()
    assert p.wait(timeout=30) == 0, "干净 EOF 必须退出码 0（StudentPolicy.close 靠它判成败）"
    p = subprocess.Popen(exe, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    p.stdout.readline()
    p.stdin.write(bytes(100))                                                # 半截帧
    p.stdin.close()
    assert p.wait(timeout=30) == 1 and "截断" in p.stderr.read().decode()
    p = subprocess.run(exe[:-2], env=env, capture_output=True, timeout=30)    # 缺 --ckpt
    assert p.returncode == 2


# ---------------------------------------------------------------- 有权重时才跑
def test_real_baselines_if_weights_present():
    root = bi.BASE
    need = {"pulp_v3": f"{root}/pulp-dronet/src/tiny-pulp-dronet-v3/model/pulp-dronet-v3-resblock-1.0.pth",
            "pulp_tiny": f"{root}/pulp-dronet/src/tiny-pulp-dronet-v3/model/tiny-pulp-dronet-v3-dw-pw-0.125.pth",
            "dronet": f"{root}/dronet/dronet_best_weights.npz"}
    if not os.path.isfile(TEACHER_PY):
        print(f"SKIP test_real_baselines_if_weights_present: 没有 {TEACHER_PY}（启动器用它做解释器）")
        return
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8).tobytes()
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "stub.ckpt")
        open(ckpt, "w").close()
        for launcher, base in LAUNCHERS.items():
            if base in need and not os.path.isfile(need[base]):
                print(f"SKIP {launcher}: 没有权重 {need[base]}")
                continue
            log = os.path.join(td, f"{launcher}.jsonl")
            restore = with_env(BASELINE_LOG=log)
            try:
                policy = StudentPolicy(ckpt, os.path.join(BASELINES, launcher), timeout_s=120.0)
                try:
                    for _ in range(2):
                        a, _, probs = policy.act(gz_msg(320, 240, 3, frame))
                        assert a in (0, 1, 2, 3) and sum(probs) == 1.0
                finally:
                    policy.close()
            finally:
                restore()
            rows = [json.loads(l) for l in open(log, encoding="utf-8")]
            assert len(rows) == 3, (launcher, len(rows))                      # 预热帧 + 2 帧
            if base != "forward":
                for r in rows:
                    assert abs(r["steer"]) < 10 and 0.0 <= r["p_coll"] <= 1.0, (launcher, r)
            print(f"ok {launcher}: 往返通过，{len(rows)} 行日志")


def main():
    for t in (test_map_steer_coll, test_map_velocity, test_defaults_are_preregistered_values, test_launchers_static,
              test_forward_roundtrip_via_student_policy, test_missing_or_bad_baseline_name_fails_loudly,
              test_probe_bytes_and_actions_survive_the_pipe, test_eof_and_truncated_frame_exit_codes,
              test_real_baselines_if_weights_present):
        t()
    print("test_baseline_infer: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
