#!/usr/bin/env python3
"""用训练好的 PrivilegedTeacher 在 Gazebo 里闭环飞（collect --policy teacher-nn）。

输入口径与 teacher/dataset.py 完全对齐：
  rgb   [T,3,224,224]  [0,1] + ImageNet 归一化
  depth [T,1,224,224]  单位米（collect 的 R_FLOAT32 原始读数，不再 /65535）
  pose  [T,8]          [x, y, z, yaw(rad), vx, vy, goal_fwd, goal_left]
  text  [9]            instruction.tokenize + PAD

torch 惰性 import：解释器里没有（或装坏）torch 时抛带解释器建议的
RuntimeError，而不是把 ImportError/OSError 堆栈甩给用户。
"""
import collections
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instruction  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEACHER_DIR = os.path.join(REPO_ROOT, "teacher")

TEXT_LEN = 9        # 与 teacher/dataset.py、src/include/uavrgb/text.hpp 一致
TEXT_VOCAB = 22
POSE_DIM = 8
IMAGE_SIZE = 224
NUM_ACTIONS = 4

# 与 teacher/dataset.py 的常量保持一致；改一处必须两处一起改。
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# dataset.py 按相邻 rgb_ts 差分；闭环每步就是一个动作窗口（hold 秒）。
# shortcut: 固定按 1 s 计，hold != 1.0 时 vx/vy 会等比例失真，需要时把时间戳传进来。
_POSE_DT_S = 1.0


def _prep_rgb(rgb_hwc, size=IMAGE_SIZE):
    """HWC uint8(RGB/RGBA) -> [3,size,size] ImageNet 归一化（与 dataset._load_rgb 同口径）。"""
    arr = np.asarray(rgb_hwc)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"RGB 需要 HWC 3/4 通道，实际 {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return ((arr - _MEAN) / _STD).transpose(2, 0, 1)


def _prep_depth(depth_m, size=IMAGE_SIZE):
    """HxW 米 -> [1,size,size]；保持米单位（只有 dataset 的 png16_mm 口径才 /1000）。"""
    arr = np.asarray(depth_m, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"depth 需要 HxW（米），实际 {arr.shape}")
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape != (size, size):
        arr = np.asarray(Image.fromarray(arr).resize((size, size), Image.BILINEAR),
                         dtype=np.float32)
    return arr[None]


def _prep_text(text_ids):
    """token list -> [9] int64；不足补 PAD，超长按 dataset 口径拒绝（不截断）。"""
    ids = [] if text_ids is None else [int(t) for t in text_ids]
    if len(ids) > TEXT_LEN:
        raise ValueError(f"指令 {len(ids)} token > {TEXT_LEN}，与训练口径不一致")
    ids += [instruction.PAD] * (TEXT_LEN - len(ids))
    return np.asarray(ids, dtype=np.int64)


class _FrameBuffer:
    """最后 seq_len 帧的 rgb/depth/pose 缓冲 + 预处理；不依赖 torch，便于单测。"""

    def __init__(self, seq_len=8, image_size=IMAGE_SIZE):
        self.seq_len = int(seq_len)
        if self.seq_len < 1:
            raise ValueError("seq_len 必须 >= 1")
        self.image_size = int(image_size)
        self._frames = collections.deque(maxlen=self.seq_len)
        self._prev_xyz = None

    def append(self, rgb_hwc, depth_m, pose_xyz, yaw_deg, waypoint_xy):
        x, y, z = (float(v) for v in pose_xyz)
        if self._prev_xyz is None:
            vx = vy = 0.0
        else:
            vx = (x - self._prev_xyz[0]) / _POSE_DT_S
            vy = (y - self._prev_xyz[1]) / _POSE_DT_S
        self._prev_xyz = (x, y, z)
        yaw_rad = float(np.radians(yaw_deg))
        # 机身系目标向量（前向/左向），与 dataset._pose_sequence 同口径。
        dx, dy = float(waypoint_xy[0]) - x, float(waypoint_xy[1]) - y
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        pose = np.array([x, y, z, yaw_rad, vx, vy,
                         dx * c + dy * s, -dx * s + dy * c], dtype=np.float32)
        self._frames.append((_prep_rgb(rgb_hwc, self.image_size),
                             _prep_depth(depth_m, self.image_size), pose))

    def build(self, text_ids):
        """-> rgb [T,3,S,S], depth [T,1,S,S], pose [T,6], text [9]。

        帧数不足 seq_len 时用最早帧填充（不报错），与训练时的定长序列对齐。
        """
        if not self._frames:
            raise ValueError("还没有任何帧，无法构建模型输入")
        frames = list(self._frames)
        frames = [frames[0]] * (self.seq_len - len(frames)) + frames
        return (np.stack([f[0] for f in frames]),
                np.stack([f[1] for f in frames]),
                np.stack([f[2] for f in frames]),
                _prep_text(text_ids))


def _import_torch():
    try:
        import torch
    except Exception as e:  # noqa: BLE001 - ImportError 与缺 libcudnn 的 OSError 同属「不可用」
        raise RuntimeError(
            "teacher-nn 需要 torch，但当前解释器无法 import torch："
            f"{type(e).__name__}: {e}\n"
            "请用带 torch 的解释器，例如 "
            "$DATA_ROOT/uav_rgb/teacher-venv/bin/python") from None
    return torch


class TeacherPolicy:
    """加载 PrivilegedTeacher，按最后 seq_len 帧返回动作 0..3。"""

    def __init__(self, ckpt_path, seq_len=8, device=None):
        if not os.path.isfile(ckpt_path):
            raise RuntimeError(f"找不到 teacher checkpoint {ckpt_path}")
        self.torch = _import_torch()
        torch = self.torch
        if TEACHER_DIR not in sys.path:
            sys.path.insert(0, TEACHER_DIR)
        from model import PrivilegedTeacher

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(ckpt_path, map_location=self.device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            for key, want in (("text_len", TEXT_LEN), ("text_vocab", TEXT_VOCAB),
                              ("pose_dim", POSE_DIM), ("image_size", IMAGE_SIZE)):
                got = ckpt.get(key)
                if got is not None and int(got) != want:
                    raise RuntimeError(f"ckpt {key}={got} 与输入口径 {want} 不一致")
            state = ckpt["model"]
        else:
            state = ckpt   # 兼容直接保存 state_dict 的旧 ckpt
        model = PrivilegedTeacher()
        model.load_state_dict(state)
        model.to(self.device).eval()
        self.model = model
        self.buffer = _FrameBuffer(seq_len)

    def act(self, rgb_hwc, depth_m, pose_xyz, yaw_deg, waypoint_xy, text_ids):
        """追加一帧观测并返回动作 int(0..3)。rgb_hwc 是 HWC uint8，depth_m 是 HxW 米。"""
        self.buffer.append(rgb_hwc, depth_m, pose_xyz, yaw_deg, waypoint_xy)
        rgb, depth, pose, text = self.buffer.build(text_ids)
        torch = self.torch
        with torch.no_grad():
            logits = self.model(
                torch.from_numpy(rgb).unsqueeze(0).to(self.device),
                torch.from_numpy(depth).unsqueeze(0).to(self.device),
                torch.from_numpy(pose).unsqueeze(0).to(self.device),
                torch.from_numpy(text).unsqueeze(0).to(self.device))
            return int(logits.argmax(dim=1).item())
