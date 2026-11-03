#!/usr/bin/env python3
"""常驻学生策略：管理 uavrgb_infer 子进程，把 gz RGB 帧缩放成模型输入。

与 collect.py 的 teacher 分支同构：动作来自当前帧，推理在发指令之前完成。
输入缩放口径与 tools/preprocess.py 一致（PIL bilinear，320x240 -> 模型 HxW）。
"""
import json
import os
import select
import subprocess
import sys
import tempfile
import time
import math

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import save_frame

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INFER = os.path.join(os.environ.get("UAV_RGB_BUILD_DIR") or
                             os.path.join(REPO_ROOT, "build", "check"), "uavrgb_infer")
RESAMPLE = (Image.Resampling.BILINEAR if hasattr(Image, "Resampling")
            else Image.BILINEAR)


class StudentPolicy:
    """推理进程只启动一次；每帧写 RGB 字节、读一行 JSON。"""

    def __init__(self, ckpt, infer_bin=None, text_ids=None, timeout_s=30.0):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("推理超时必须为有限正数")
        self.timeout_s = timeout_s
        binary = infer_bin or DEFAULT_INFER
        if not os.path.isfile(binary):
            raise RuntimeError(
                f"找不到推理程序 {binary}；先运行 bash tools/check.sh（支持 UAV_RGB_BUILD_DIR）")
        if not os.path.isfile(ckpt):
            raise RuntimeError(f"找不到 checkpoint {ckpt}")
        # stderr 用临时文件而不是 PIPE：长采集时 CUDA/驱动警告写满 64 KiB 管道会让
        # 子进程阻塞在 write，父进程的 readline 永远等不到结果（T5 卡死风险）。
        self._err = tempfile.TemporaryFile()
        cmd = [binary, "--ckpt", ckpt]
        if text_ids:
            # L0：单集一条指令；L2 才把 token 送进文本编码器。
            cmd += ["--text", ",".join(str(int(t)) for t in text_ids)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, bufsize=0)
        os.set_blocking(self.proc.stdin.fileno(), False)
        os.set_blocking(self.proc.stdout.fileno(), False)
        line = self._exchange(b"", "hello")
        if not line:
            self._abort_start(f"推理进程启动失败：{self._stderr()}")
        try:
            hello = json.loads(line)
        except ValueError as exc:
            self._abort_start(f"推理进程 hello 不是合法 JSON：{exc}")
        self.h = int(hello["in_h"])
        self.w = int(hello["in_w"])
        self.in_c = int(hello["in_c"])
        if self.in_c != 3:
            self._abort_start(f"模型输入通道 {self.in_c} != 3，相机是 RGB")
        # 非四动作模型会让越界动作在 collect 里静默变成悬停，闭环数字看似正常但无意义。
        # 用 get：旧 infer 没有该字段时应走 _abort_start 杀掉子进程，而不是 KeyError 泄漏。
        classes = hello.get("num_classes")
        if classes is None or int(classes) != 4:
            self._abort_start(
                f"模型输出 {classes} 类 != 4（SAFE/FORWARD/LEFT/RIGHT）")
        # 文本分支维度：0 表示纯视觉模型；L/I/S 消融靠它确认改文本真的会改推理。
        self.text_dim = int(hello.get("text_dim", 0))
        self.frames = 0
        self._warmup()

    def _warmup(self):
        """先跑一帧全零：把 CUDA 模块加载/首帧开销移出第 1 步动作。

        否则每集第一帧的推理要多花约 0.4 s，动作基于更旧的画面。
        """
        if not self._exchange(bytes(self.h * self.w * self.in_c), "warmup"):
            self._abort_start(f"推理进程 warmup 失败：{self._stderr()}")

    def act(self, msg):
        """gz.msgs.Image dict -> (action, infer_ms, probs)。解码/推理失败抛异常。"""
        w, h, nch, rows = save_frame.decode(msg)
        if nch not in (3, 4):
            raise ValueError(f"RGB 通道数 {nch} 不支持")
        mode = "RGB" if nch == 3 else "RGBA"
        im = Image.frombytes(mode, (w, h), rows).convert("RGB")
        im = im.resize((self.w, self.h), RESAMPLE)
        line = self._exchange(im.tobytes(), "act")
        if not line:
            raise RuntimeError(f"推理进程无输出：{self._stderr()}")
        obj = json.loads(line)
        self.frames += 1
        return int(obj["action"]), float(obj["infer_ms"]), obj["probs"]

    def _exchange(self, payload, stage):
        # 同时约束写入和读完整行：仅给 readline 加超时挡不住不消费 stdin 的进程。
        deadline = time.monotonic() + self.timeout_s
        pending = memoryview(payload)
        reply = bytearray()
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._abort_start(f"推理 {stage} 超时（{self.timeout_s:g}s）")
                readable, writable, _ = select.select(
                    [] if pending else [self.proc.stdout],
                    [self.proc.stdin] if pending else [], [], remaining)
                if writable:
                    pending = pending[os.write(self.proc.stdin.fileno(), pending):]
                if readable:
                    chunk = os.read(self.proc.stdout.fileno(), 4096)
                    if not chunk:
                        self._abort_start(f"推理 {stage} 输出中断：{self._stderr()}")
                    reply.extend(chunk)
                    if len(reply) > 65536:
                        self._abort_start(f"推理 {stage} 响应超过 64 KiB")
                    if reply.endswith(b"\n"):
                        return bytes(reply)
        except (OSError, ValueError) as exc:
            self._abort_start(f"推理 {stage} I/O 失败：{exc}")

    def _abort_start(self, msg):
        """构造失败时杀掉已启动的推理进程，避免留下阻塞在 stdin 的孤儿。"""
        proc = self.proc
        self.proc = None
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
            for stream in (proc.stdin, proc.stdout):
                if stream:
                    stream.close()
        if self._err is not None:
            self._err.close()
            self._err = None
        raise RuntimeError(msg)

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
        if proc.stdout:
            proc.stdout.close()
        self._err.close()
        self._err = None
        if rc != 0:
            raise RuntimeError(f"推理进程退出码 {rc}：{err or '（空）'}")

    def _stderr(self):
        if self._err is None:
            return "（无 stderr）"
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip() or "（空）"
        except (OSError, ValueError):
            return "（stderr 已关闭）"
