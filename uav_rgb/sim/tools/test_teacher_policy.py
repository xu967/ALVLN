#!/usr/bin/env python3
"""teacher_policy.py 自检：帧缓冲 + 预处理纯逻辑；可选 torch 冒烟。

不启动 Gazebo、不依赖真实 ckpt：
  python3 sim/tools/test_teacher_policy.py

纯逻辑部分只用 numpy/PIL。当前解释器没有可用 torch 时，会自动找
$DATA_ROOT/uav_rgb/teacher-venv/bin/python 跑「随机权重模型能返回 0..3」的冒烟；
找不到可用的 torch 就打印 SKIP（不判失败）。
"""
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import teacher_policy as tp


def _rgb(channel, size=8, value=200):
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., channel] = value
    return rgb


def _depth(value=2.0, size=8):
    return np.full((size, size), value, dtype=np.float32)


def test_buffer():
    buf = tp._FrameBuffer(seq_len=4)
    buf.append(_rgb(0), _depth(), (0.0, 0.0, 2.0), 0.0)
    buf.append(_rgb(1), _depth(), (1.0, 0.0, 2.0), 90.0)
    rgb, depth, pose, text = buf.build([1, 2, 3])

    assert rgb.shape == (4, 3, tp.IMAGE_SIZE, tp.IMAGE_SIZE), rgb.shape
    assert depth.shape == (4, 1, tp.IMAGE_SIZE, tp.IMAGE_SIZE), depth.shape
    assert pose.shape == (4, tp.POSE_DIM), pose.shape
    assert text.shape == (tp.TEXT_LEN,), text.shape
    assert text.tolist() == [1, 2, 3] + [0] * (tp.TEXT_LEN - 3), text

    # 帧数不足 seq_len：用最早帧填充（前 3 帧都是第 1 帧）。
    assert np.array_equal(rgb[0], rgb[2]), "帧数不足时没有用最早帧填充"
    assert not np.array_equal(rgb[2], rgb[3]), "填充把新帧覆盖掉了"

    # 深度单位是米：2.0 不能被 /65535（也不能被 /1000）。
    assert np.allclose(depth, 2.0, atol=1e-6), depth

    # ImageNet 归一化后的值应该有限且在合理范围内。
    assert np.isfinite(rgb).all(), "RGB 归一化出现 NaN/Inf"
    assert np.abs(rgb).max() < 6.0, np.abs(rgb).max()

    # 相邻帧 x 差 1 m、dt 1 s -> vx=1.0；首帧 vx=0；yaw 转弧度。
    assert abs(float(pose[0, 4])) < 1e-6, pose[0, 4]
    assert abs(float(pose[3, 4]) - 1.0) < 1e-6, pose[3, 4]
    assert abs(float(pose[3, 3]) - math.pi / 2) < 1e-6, pose[3, 3]

    # 缓冲满后只保留最后 seq_len 帧（deque maxlen 生效）。
    for i in range(3):
        buf.append(_rgb(2), _depth(), (2.0 + i, 0.0, 2.0), 0.0)
    rgb2, _, pose2, _ = buf.build([])
    assert rgb2.shape[0] == 4, rgb2.shape
    assert abs(float(pose2[-1, 4]) - 1.0) < 1e-6, pose2[-1, 4]


def torch_smoke():
    """随机权重 ckpt：只验证加载/前向链路能返回合法动作，不验证精度。"""
    try:
        import torch
        if tp.TEACHER_DIR not in sys.path:
            sys.path.insert(0, tp.TEACHER_DIR)
        from model import PrivilegedTeacher
    except Exception as e:  # noqa: BLE001 - torch/torchvision 不在就是 SKIP 条件
        print(f"SKIP: torch 不可用（{type(e).__name__}: {e}）")
        return
    with tempfile.TemporaryDirectory() as td:
        ckpt = os.path.join(td, "random.pth")
        torch.save({"model": PrivilegedTeacher().state_dict()}, ckpt)
        policy = tp.TeacherPolicy(ckpt, seq_len=2, device="cpu")
        for x in (0.0, 1.0):
            action = policy.act(_rgb(int(x) % 3), _depth(), (x, 0.0, 2.0),
                                0.0, [1, 2, 3])
            assert isinstance(action, int), action
            assert 0 <= action < tp.NUM_ACTIONS, action
    print("torch smoke: PASS（随机权重模型返回 0..3）")


def teacher_venv_python():
    data_root = os.environ.get("DATA_ROOT", "/root/rivermind-data")
    return os.path.join(data_root, "uav_rgb", "teacher-venv", "bin", "python")


def run_smoke():
    try:
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001 - 换 teacher-venv 子进程试
        venv = teacher_venv_python()
        if not os.path.isfile(venv):
            print(f"SKIP: 当前解释器没有 torch（{type(e).__name__}），"
                  f"也没找到 {venv}")
            return
        r = subprocess.run([venv, os.path.abspath(__file__), "--torch-smoke"],
                           capture_output=True, text=True, timeout=300)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            raise AssertionError(
                f"teacher-venv torch 冒烟失败: {r.stderr.strip()[-800:]}")
        return
    torch_smoke()


def main():
    if "--torch-smoke" in sys.argv[1:]:
        torch_smoke()
        return 0
    test_buffer()
    run_smoke()
    print("test_teacher_policy: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
