#!/usr/bin/env python3
"""uavrgb_infer 行协议的最小客户端：对拍与回放脚本用，不依赖 sim/tools。

协议见 src/infer.cpp、src/infer_main.cpp 与 sim/tools/student.py：
  启动后先读一行 hello JSON {"in_h","in_w","in_c","num_classes","text_dim","prev_action_dim","aux_classes"}；
  之后每步写一个 payload（in_c/3 张 HWC RGB uint8 帧，当前帧在前、上一帧在后；接 prev-action 的模型末尾再加 1 字节），
  读一行 {"action":a,"probs":[...],"infer_ms":ms}。stdin 关闭即退出。
真正的闭环评测用 sim/tools/student.py 的 StudentPolicy；这里的客户端是为了批量灌帧、拿 probs。
"""
import io
import json
import subprocess
import tempfile

import numpy as np


def chw_to_hwc_bytes(frame_chw):
    """[3,H,W] uint8 -> HWC 字节（协议里一张图的格式）。"""
    return np.ascontiguousarray(np.asarray(frame_chw, dtype=np.uint8).transpose(1, 2, 0)).tobytes()


def make_payload(cur_chw, prev_chw=None, prev_action=None):
    """一步的 payload。prev_chw 不为 None 表示双帧模型；prev_action 不为 None 表示接 prev-action。"""
    out = chw_to_hwc_bytes(cur_chw)
    if prev_chw is not None:
        out += chw_to_hwc_bytes(prev_chw)
    if prev_action is not None:
        out += bytes([int(prev_action)])
    return out


class InferClient:
    """启动一个推理进程（C++ 二进制或 Python 包装器的启动器），每步一问一答。"""

    def __init__(self, binary, ckpt, text_ids=None, env=None):
        cmd = [binary, "--ckpt", ckpt]
        if text_ids:
            cmd += ["--text", ",".join(str(int(t)) for t in text_ids)]
        self._err = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err,
                                     bufsize=0, env=env)
        self._rd = io.BufferedReader(self.proc.stdout, buffer_size=1 << 16)   # 逐字节 readline 太慢
        line = self._rd.readline()
        if not line:
            raise RuntimeError(f"推理进程没有 hello：{self.stderr()}")
        self.hello = json.loads(line)
        self.payload_bytes = (self.hello["in_h"] * self.hello["in_w"] * self.hello["in_c"]
                              + (1 if self.hello.get("prev_action_dim", 0) else 0))

    def step(self, payload):
        if len(payload) != self.payload_bytes:
            raise ValueError(f"payload 长度 {len(payload)} != {self.payload_bytes}")
        view = memoryview(payload)
        while view:   # 管道写可能只写一部分
            view = view[self.proc.stdin.write(view):]
        line = self._rd.readline()
        if not line:
            raise RuntimeError(f"推理进程无输出：{self.stderr()}")
        return json.loads(line)

    def stderr(self):
        self._err.seek(0)
        return self._err.read().decode("utf-8", "replace").strip()

    def close(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        rc = self.proc.wait(timeout=30)
        err = self.stderr()
        self._rd.close()
        self._err.close()
        return rc, err

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.close()
