"""把 collect.py 产出的 episode 目录读成教师模型序列样本。

每个 episode 取「最后 seq_len 帧」组成一个样本，标签取最后一帧的
action_expert（0..3）。teacher 只用离线软标签，不参与 student 交付。

容错约定：缺 depth 文件、缺 pose_before/yaw_before、缺 RGB、指令为空、
帧数不足、标签越界 —— 该集跳过并打印警告，不中断整体扫描。
require_depth=False 时只读 RGB+text+pose（当前 collect.py 不落盘深度）。
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 复用仓库唯一词表定义，避免 Python 侧与 text.hpp 漂移。
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim" / "tools"))
import instruction  # noqa: E402

TEXT_LEN = 9        # 与 src/include/uavrgb/text.hpp kTextLen 一致
TEXT_VOCAB = 22     # 与 src/include/uavrgb/text.hpp kTextVocab 一致
NUM_ACTIONS = 4
POSE_DIM = 8        # [x, y, z, yaw(rad), vx, vy, goal_fwd, goal_left]
IMAGE_SIZE = 224    # RGB/深度统一缩放到该边长
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

if len(instruction.VOCAB) != TEXT_VOCAB:
    raise RuntimeError(
        f"词表大小不一致：instruction.VOCAB={len(instruction.VOCAB)}，"
        f"text.hpp kTextVocab={TEXT_VOCAB}")


def _warn(msg):
    print(f"警告: {msg}", file=sys.stderr)


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _depth_path(ep_dir, manifest, row):
    """定位某一步的深度文件；找不到时返回 None。

    优先级：
      1. actions.jsonl 行内 depth_file / depth_frame；
      2. manifest.depth_ref / depth_dir，支持 "{step}"、"%05d" 模板或目录名；
      3. episode 目录下 depth_%05d.png。
    """
    step = row.get("step")
    if step is None:
        return None
    step = int(step)
    name = row.get("depth_file") or row.get("depth_frame")
    if not name:
        ref = manifest.get("depth_ref") or manifest.get("depth_dir")
        if isinstance(ref, str) and ref:
            if "{" in ref:
                name = ref.format(step=step, frame=step, index=step)
            elif "%" in ref:
                name = ref % step
            elif Path(ref).suffix:
                name = ref
            else:
                name = str(Path(ref) / f"depth_{step:05d}.png")
        else:
            name = f"depth_{step:05d}.png"
    p = Path(name)
    return p if p.is_absolute() else ep_dir / p


def _pose_sequence(rows, waypoint):
    """actions.jsonl 行 -> (T,8) 位姿+目标向量；缺字段返回 None。

    假设 pose_before 含 x/y/z，yaw_before 为角度；这里转弧度。
    vx/vy 用相邻帧 rgb_ts 差分，首帧为 0。
    末两维是**机身系**的目标向量（前向/左向）：教师是特权模型，允许直接知道
    终点在哪；否则它要从图像和指令里反推，等于把学生要学的难题也丢给教师。
    """
    wx, wy = waypoint
    poses = []
    prev_xyz = None
    prev_ts = None
    for row in rows:
        pb = row.get("pose_before")
        yaw = row.get("yaw_before")
        if not isinstance(pb, dict) or yaw is None:
            return None
        try:
            x = float(pb["x"])
            y = float(pb["y"])
            z = float(pb["z"])
            yaw_rad = math.radians(float(yaw))
        except (KeyError, TypeError, ValueError):
            return None
        ts = int(row.get("rgb_ts") or row.get("timestamp_ns") or 0)
        vx = vy = 0.0
        if prev_xyz is not None and prev_ts and ts > prev_ts:
            dt = (ts - prev_ts) / 1e9
            vx = (x - prev_xyz[0]) / dt
            vy = (y - prev_xyz[1]) / dt
        dx, dy = wx - x, wy - y
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        poses.append([x, y, z, yaw_rad, vx, vy, dx * c + dy * s, -dx * s + dy * c])
        prev_xyz = (x, y, z)
        prev_ts = ts
    return np.asarray(poses, dtype=np.float32)


def _load_rgb(path, size):
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return (arr - _MEAN) / _STD


def _load_depth(path, size, depth_format=None):
    """返回**米**。depth_format="png16_mm" 时按毫米换算，其余保持旧归一化口径。"""
    if path.suffix.lower() == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        with Image.open(path) as im:
            if im.mode == "F":
                arr = np.asarray(im, dtype=np.float32)
            else:
                arr = np.asarray(im)
                if arr.ndim == 3:  # 单通道图也可能带 alpha
                    arr = arr[..., 0]
                if depth_format == "png16_mm":
                    arr = arr.astype(np.float32) / 1000.0
                else:
                    # 旧口径：8/16 位 PNG 当作归一化深度；浮点 PNG 保持原始值。
                    scale = 65535.0 if arr.dtype == np.uint16 else 255.0
                    arr = arr.astype(np.float32) / scale
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.shape != (size, size):
        im = Image.fromarray(arr)
        arr = np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32)
    return arr


class EpisodeSequenceDataset(Dataset):
    """split 目录（train/val/test 之一）下每个 episode 目录一个序列样本。"""

    def __init__(self, split_dir, seq_len, image_size=IMAGE_SIZE,
                 require_depth=True, stride=1, cache_frames=True):
        self.seq_len = int(seq_len)
        self.stride = max(1, int(stride))
        self.image_size = int(image_size)
        self.require_depth = bool(require_depth)
        # 滑窗会重复读同一帧；本机内存充足，缓存解码后的数组（每帧约 0.6 MB）。
        self._cache = {} if cache_frames else None
        if self.seq_len < 1:
            raise ValueError("seq_len 必须 >= 1")
        root = Path(split_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"数据目录不存在: {root}")
        ep_dirs = sorted(
            p for p in root.iterdir()
            if p.is_dir() and (p / "episode_manifest.json").is_file())
        if not ep_dirs:
            _warn(f"{root} 下没有 episode 目录（需含 episode_manifest.json）")
        self.samples = []
        for ep_dir in ep_dirs:
            samples = self._build_samples(ep_dir)
            if samples:
                self.samples.extend(samples)
        _warn(f"{root}: {len(self.samples)} 个滑窗样本，来自 "
              f"{len({s['episode_id'] for s in self.samples})}/{len(ep_dirs)} 个 episode")

    def _build_samples(self, ep_dir):
        """每集产出多个滑窗样本：窗口 = 连续 seq_len 帧，标签取**窗口末帧**的动作。

        旧实现只取每集最后 seq_len 帧、标签取最后一帧，而教师集都以停在 waypoint
        的 SAFE 收尾，导致所有标签恒为 SAFE_NEUTRAL（训练 acc 假性 1.0）。
        """
        try:
            with open(ep_dir / "episode_manifest.json", encoding="utf-8") as f:
                manifest = json.load(f)
        except (OSError, ValueError) as exc:
            _warn(f"跳过 {ep_dir}: manifest 读取失败 ({exc})")
            return None

        text = manifest.get("instruction")
        if not isinstance(text, str) or not text.strip():
            _warn(f"跳过 {ep_dir}: manifest 缺 instruction")
            return None
        text_ids = instruction.tokenize(text)
        if not text_ids:
            _warn(f"跳过 {ep_dir}: 指令 tokenize 为空")
            return None
        if len(text_ids) > TEXT_LEN:
            # 与 text.hpp 一致：超长不截断，整集跳过。
            _warn(f"跳过 {ep_dir}: 指令 {len(text_ids)} token > {TEXT_LEN}")
            return None
        wp = manifest.get("waypoint")
        if not isinstance(wp, dict) or "x" not in wp or "y" not in wp:
            _warn(f"跳过 {ep_dir}: manifest 缺 waypoint（教师需要它作为特权输入）")
            return None
        text_ids = text_ids + [instruction.PAD] * (TEXT_LEN - len(text_ids))

        actions_path = ep_dir / "actions.jsonl"
        if not actions_path.is_file():
            _warn(f"跳过 {ep_dir}: 缺 actions.jsonl")
            return None
        try:
            rows = _read_jsonl(actions_path)
        except (OSError, ValueError) as exc:
            _warn(f"跳过 {ep_dir}: actions.jsonl 读取失败 ({exc})")
            return None
        rows = [r for r in rows if isinstance(r, dict) and r.get("rgb_ok", True)]
        rows.sort(key=lambda r: int(r.get("step") or 0))
        if any(r.get("step") is None for r in rows):
            _warn(f"跳过 {ep_dir}: step 缺失，无法核对原始帧编号")
            return None
        if len({int(r["step"]) for r in rows}) != len(rows):
            _warn(f"跳过 {ep_dir}: step 编号重复")
            return None
        if len(rows) < self.seq_len:
            _warn(f"跳过 {ep_dir}: 有效帧 {len(rows)} < seq_len {self.seq_len}")
            return None

        rgb_all = []
        depth_all = [] if self.require_depth else None
        for row in rows:
            step = row.get("step")
            frame = row.get("frame")
            if not frame and step is not None:
                frame = f"rgb_{int(step):05d}.png"
            if not frame:
                _warn(f"跳过 {ep_dir}: 行缺 frame/step")
                return None
            rgb_path = ep_dir / frame
            if not rgb_path.is_file():
                _warn(f"跳过 {ep_dir}: 缺 RGB 文件 {frame}")
                return None
            rgb_all.append(rgb_path)
            if self.require_depth:
                depth_path = _depth_path(ep_dir, manifest, row)
                if depth_path is None or not depth_path.is_file():
                    _warn(f"跳过 {ep_dir}: step {step} 缺深度文件"
                          f"（当前 collect.py 不保存深度；"
                          f"仅验证 RGB 用 require_depth=False）")
                    return None
                depth_all.append(depth_path)

        poses = _pose_sequence(rows, (float(wp["x"]), float(wp["y"])))
        if poses is None:
            _warn(f"跳过 {ep_dir}: actions.jsonl 缺 pose_before/yaw_before")
            return None

        labels = [r.get("action_expert") for r in rows]
        for i, lab in enumerate(labels):
            if not isinstance(lab, int) or not 0 <= lab < NUM_ACTIONS:
                _warn(f"跳过 {ep_dir}: 第 {i} 帧 action_expert 非法 ({lab!r})")
                return None

        episode_id = str(manifest.get("episode_id", ep_dir.name))
        text_arr = np.asarray(text_ids, dtype=np.int64)
        depth_format = manifest.get("depth_format")
        # 闭环 warm-up：推理前 seq_len-1 步缓冲还没填满，_FrameBuffer 用**首帧重复**
        # 补齐。训练必须见过同样的拼接，否则闭环第 0/1 步（专家 70%/58% 是转向）
        # 会被判成 FORWARD，一开局就直飞、之后再也回不来（2026-11-02 实测闭环
        # 316 步 0 次转向、SR 0）。不受 stride 影响：闭环开头总会经过。
        out = []
        for i in range(self.seq_len - 1):
            if any(int(rows[j+1]["step"]) != int(rows[j]["step"])+1 for j in range(i)):
                continue  # 断帧后不能假装是连续历史帧
            idx = [0] * (self.seq_len - i - 1) + list(range(i + 1))
            out.append({
                "ep_dir": ep_dir,
                "rgb_paths": [rgb_all[j] for j in idx],
                "depth_paths": ([depth_all[j] for j in idx]
                                if depth_all is not None else None),
                "depth_format": depth_format,
                "pose": poses[idx],
                "text": text_arr,
                "label": labels[i],
                "episode_id": episode_id,
                # Preserve the SOURCE step ID; filtered arrays may have gaps.
                "step": int(rows[i]["step"]),
            })
        for end in range(self.seq_len, len(rows) + 1, self.stride):
            lo = end - self.seq_len
            if any(int(rows[j+1]["step"]) != int(rows[j]["step"])+1
                   for j in range(lo,end-1)):
                continue  # 过滤坏帧导致的间隙不可拼接为相邻时序
            out.append({
                "ep_dir": ep_dir,
                "rgb_paths": rgb_all[lo:end],
                "depth_paths": depth_all[lo:end] if depth_all is not None else None,
                "depth_format": depth_format,
                "pose": poses[lo:end],
                "text": text_arr,
                "label": labels[end - 1],
                "episode_id": episode_id,
                "step": int(rows[end - 1]["step"]),
            })
        return out

    def __len__(self):
        return len(self.samples)

    def _rgb(self, path):
        if self._cache is None:
            return _load_rgb(path, self.image_size)
        arr = self._cache.get(path)
        if arr is None:
            arr = _load_rgb(path, self.image_size)
            self._cache[path] = arr
        return arr

    def _depth(self, path, fmt):
        if self._cache is None:
            return _load_depth(path, self.image_size, fmt)
        key = ("depth", path)
        arr = self._cache.get(key)
        if arr is None:
            arr = _load_depth(path, self.image_size, fmt)
            self._cache[key] = arr
        return arr

    def __getitem__(self, index):
        sample = self.samples[index]
        rgb = np.stack([self._rgb(p) for p in sample["rgb_paths"]])
        item = {
            "rgb": torch.from_numpy(rgb).permute(0, 3, 1, 2).contiguous(),
            "pose": torch.from_numpy(sample["pose"]),
            "text": torch.from_numpy(sample["text"]),
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "episode_id": sample["episode_id"],
            "step": int(sample["step"]),
        }
        if sample["depth_paths"] is not None:
            depth = np.stack([
                self._depth(p, sample.get("depth_format"))
                for p in sample["depth_paths"]
            ])
            item["depth"] = torch.from_numpy(depth).unsqueeze(1).contiguous()
        return item
