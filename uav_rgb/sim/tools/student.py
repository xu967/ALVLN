#!/usr/bin/env python3
"""常驻学生策略：管理 uavrgb_infer 子进程，把 gz RGB 帧缩放成模型输入。

与 collect.py 的 teacher 分支同构：动作来自当前帧，推理在发指令之前完成。
输入缩放口径与 tools/preprocess.py 一致（PIL bilinear，320x240 -> 模型 HxW）。
"""
import json
import os
import subprocess
import sys
import tempfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_frame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INFER = os.path.join(REPO_ROOT, "src", "build", "uavrgb_infer")
RESAMPLE = (Image.Resampling.BILINEAR if hasattr(Image, "Resampling")
            else Image.BILINEAR)


class StudentPolicy:
    """推理进程只启动一次；每帧写 RGB 字节、读一行 JSON。"""

    def __init__(self, ckpt, infer_bin=None):
        binary = infer_bin or DEFAULT_INFER
        if not os.path.isfile(binary):
            raise RuntimeError(
                f"找不到推理程序 {binary}；先构建：cmake --build src/build --target uavrgb_infer")
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"找不到 checkpoint {ckpt}")
        # stderr 用临时文件而不是 PIPE：长采集时 CUDA/驱动警告写满 64 KiB 管道会让
        # 子进程阻塞在 write，父进程的 readline 永远等不到结果（T5 卡死风险）。
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(
            [binary, "--ckpt", ckpt],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err)
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"推理进程启动失败：{self._stderr()}")
        hello = json.loads(line)
        self.h = int(hello["in_h"])
        self.w = int(hello["in_w"])
        self.in_c = int(hello["in_c"])
        if self.in_c != 3:
            raise RuntimeError(f"模型输入通道 {self.in_c} != 3，相机是 RGB")
        self.frames = 0
        self._warmup()

    def _warmup(self):
        """先跑一帧全零：把 CUDA 模块加载/首帧开销移出第 1 步动作。

        否则每集第一帧的推理要多花约 0.4 s，动作基于更旧的画面。
        """
        self.proc.stdin.write(bytes(self.h * self.w * self.in_c))
        self.proc.stdin.flush()
        if not self.proc.stdout.readline():
            raise RuntimeError(f"推理进程 warmup 失败：{self._stderr()}")

    def act(self, msg):
        """gz.msgs.Image dict -> (action, infer_ms, probs)。解码/推理失败抛异常。"""
        w, h, nch, rows = save_frame.decode(msg)
        if nch not in (3, 4):
            raise ValueError(f"RGB 通道数 {nch} 不支持")
        mode = "RGB" if nch == 3 else "RGBA"
        im = Image.frombytes(mode, (w, h), rows).convert("RGB")
        im = im.resize((self.w, self.h), RESAMPLE)
        self.proc.stdin.write(im.tobytes())
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"推理进程无输出：{self._stderr()}")
        obj = json.loads(line)
        self.frames += 1
        return int(obj["action"]), float(obj["infer_ms"]), obj["probs"]

    def close(self):
        if self.proc is None:
            return
        proc = self.proc
        self.proc = None
        try:
            if proc.stdin:
                proc.stdin.close()
            rc = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait()
        err = self._stderr()
        if rc != 0:
            raise RuntimeError(f"推理进程退出码 {rc}：{err or '（空）'}")
        self._err.close()
        self._err = None

    def _stderr(self):
        if self._err is None:
            return "（无 stderr）"
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip() or "（空）"
        except (OSError, ValueError):
            return "（stderr 已关闭）"
