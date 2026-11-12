#!/usr/bin/env python3
"""uavrgb_infer 的 PyTorch 原型替身：同一条命令行、同一行协议，读 C++ 训练出的 checkpoint（v6）。

用法：infer_wrap.py --ckpt <ckpt.bin> [--text <id,id,...>]      （经启动器 proto/uavrgb_infer_proto 调用）
  sim/tools/student.py 的 StudentPolicy、tools/eval_*.py 的 --infer-bin 不用改：把 --infer-bin 指到启动器即可。

协议（与 src/infer_main.cpp、src/infer.cpp 一致）：
  启动：读 checkpoint，stdout 写一行 hello
        {"in_h":..,"in_w":..,"in_c":..,"num_classes":..,"text_dim":..,"prev_action_dim":..,"aux_classes":..}
  每步：stdin 读 payload = in_c/3 张 HWC RGB uint8 帧（当前帧在前、上一帧在后）+ 接 prev-action 的模型再加 1 字节动作 id；
        stdout 写一行 {"action":a,"probs":[..],"infer_ms":ms}
  stdin 干净 EOF（0 字节）退出 0；半个 payload 报错退出 1；命令行错误退出 2。
与 C++ 的区别（有意的）：
  - probs 写 float64 的完整精度（C++ 的 `out << double` 只有 6 位有效数字）；动作取 float32 logits 的第一个最大值，与 C++ 同。
  - infer_ms 计时范围同 C++（输入上传 + 前向 + logits 取回，不含 HWC→CHW 重排），但这里是 PyTorch 前向。
  - 只读 v6 checkpoint（C++ 读 v2–v6）；辅助头不算（与 C++ 一样，部署路径只算分类头）。
  - stdout 只给协议用：启动时把 fd 1 复制出去，fd 1 与 sys.stdout 改指 stderr，库的打印不会混进协议。
环境变量：UAV_RGB_PROTO_DEVICE=cuda|cpu（不设：有 CUDA 用 CUDA，否则 CPU；显式设 cuda 而 CUDA 不可用则报错）、
          UAV_RGB_PROTO_THREADS（CPU 线程数，默认 1）。
"""
import json
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_io as cio  # noqa: E402

TEXT_LEN = cio.TEXT_LEN
TEXT_VOCAB = cio.TEXT_VOCAB
USAGE = "用法: infer_wrap.py --ckpt <ckpt.bin> [--text <id,id,...>]"
_INT_RE = re.compile(r"^[ \t\n\v\f\r]*[+-]?[0-9]+$")   # C++ strtol(base 10) 且 *end=='\0' 接受的写法


class UsageError(Exception):
    pass


def parse_text(s):
    """同 infer_main.cpp 的 parse_text：空串 = 无 token；每个 token 必须是 [0, 22) 的十进制整数。"""
    if s == "":
        return []
    out = []
    for tok in s.split(","):
        if tok == "" or not _INT_RE.match(tok):
            raise UsageError(f"--text 的 token 非法: {tok!r}")
        v = int(tok)
        if v < 0 or v >= TEXT_VOCAB:
            raise UsageError(f"--text 的 token 越界: {v}")
        out.append(v)
    return out


def parse_args(argv):
    ckpt, text_ids = "", []
    i = 0
    while i < len(argv):
        if argv[i] == "--ckpt" and i + 1 < len(argv):
            ckpt = argv[i + 1]
            i += 2
        elif argv[i] == "--text" and i + 1 < len(argv):
            text_ids = parse_text(argv[i + 1])
            i += 2
        else:
            raise UsageError(f"无法识别的参数: {argv[i]!r}")
    if not ckpt:
        raise UsageError("缺 --ckpt")
    return ckpt, text_ids


def protect_stdout():
    """把协议输出口复制到新 fd 并返回；之后 fd 1 与 sys.stdout 都指向 stderr。"""
    sys.stdout.flush()
    out_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return out_fd


def write_all(fd, data):
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def hello_line(ck):
    return ('{"in_h":%d,"in_w":%d,"in_c":%d,"num_classes":%d,"text_dim":%d,"prev_action_dim":%d,"aux_classes":%d}\n'
            % (ck.in_h, ck.in_w, ck.in_c, ck.num_classes, ck.text_dim, ck.prev_dim, ck.aux_classes)).encode()


def reply_line(action, probs, ms):
    return (json.dumps({"action": int(action), "probs": [float(p) for p in probs], "infer_ms": float(ms)},
                       separators=(",", ":")) + "\n").encode()


class Engine:
    """载入 checkpoint 的 PyTorch 模型 + 单帧推理。step() 的返回值与 C++ run_infer_loop 每步产出的三项一致。"""

    def __init__(self, ck, text_ids, device=None, threads=1):
        import torch
        import model as pm
        self.torch = torch
        self.pm = pm
        want = device or os.environ.get("UAV_RGB_PROTO_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        if want.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("infer: 要求 CUDA 但 torch.cuda 不可用")
        self.device = torch.device(want)
        if self.device.type == "cpu":
            torch.set_num_threads(int(os.environ.get("UAV_RGB_PROTO_THREADS", threads)))
        pm.configure_determinism()
        self.model = pm.model_from_ckpt(ck, self.device)
        self.h, self.w, self.c = ck.in_h, ck.in_w, ck.in_c
        self.images = self.c // 3
        self.one = self.h * self.w * 3
        self.prev_dim = ck.prev_dim
        self.payload = self.one * self.images + (1 if self.prev_dim > 0 else 0)
        self.text = None
        self.text_overflow = ck.text_dim > 0 and len(text_ids) > TEXT_LEN   # C++ 在 hello 之后、第一帧之前才报这个错
        self.text_was_empty = ck.text_dim > 0 and len(text_ids) == 0
        if ck.text_dim > 0:
            ids = (list(text_ids) + [0] * TEXT_LEN)[:TEXT_LEN]
            self.text = torch.tensor([ids], dtype=torch.uint8, device=self.device)
        # 预热：把 CUDA 上下文、cuDNN 与 kernel 加载放到 hello 之前，StudentPolicy 的 30 s 超时只管一次小前向
        self.step(bytes(self.payload))

    def step(self, buf):
        """buf：一步 payload（长度 self.payload）。返回 (action, probs[float64 数组], infer_ms)。"""
        torch = self.torch
        arr = np.frombuffer(buf, dtype=np.uint8)
        chw = np.empty((self.c, self.h, self.w), np.uint8)   # 通道 0..2 当前帧、3..5 上一帧（C 连续、可写）
        for k in range(self.images):
            chw[3 * k:3 * k + 3] = arr[k * self.one:(k + 1) * self.one].reshape(self.h, self.w, 3).transpose(2, 0, 1)
        prev_id = int(arr[-1]) if self.prev_dim > 0 else 0
        if self.prev_dim > 0 and prev_id >= self.prev_dim:
            raise RuntimeError(f"infer: prev-action 字节越界: {prev_id}")
        t0 = time.perf_counter()
        x = torch.from_numpy(chw).unsqueeze(0).to(self.device)
        prev = torch.tensor([prev_id], dtype=torch.long, device=self.device) if self.prev_dim > 0 else None
        with torch.inference_mode():
            logits = self.model(x, self.text, prev)
        z = logits.float().cpu().numpy()[0]          # 取回即同步
        t1 = time.perf_counter()
        if z.shape[0] != self.model.num_classes:
            raise RuntimeError("infer: logits 维度与 num_classes 不符")
        if not np.all(np.isfinite(z)):
            raise RuntimeError("infer: logits 含非有限值")
        probs = self.pm.softmax_probs(z[None])[0]
        return int(np.argmax(z)), probs, (t1 - t0) * 1000.0


def serve(engine, stdin, out_fd):
    if engine.text_overflow:
        raise ValueError("infer: --text token 数超过 kTextLen")
    if engine.text_was_empty:
        sys.stderr.write("infer_wrap: 文本模型没有 --text，按全 <pad> 推理\n")
    frames = 0
    while True:
        buf = stdin.read(engine.payload)
        if not buf:
            return frames
        if len(buf) != engine.payload:
            raise RuntimeError("infer: 帧数据不足（截断）")
        action, probs, ms = engine.step(buf)
        try:
            write_all(out_fd, reply_line(action, probs, ms))
        except OSError as exc:
            raise RuntimeError(f"infer: stdout 写入失败: {exc}")
        frames += 1


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    try:
        ckpt_path, text_ids = parse_args(argv)
    except UsageError as exc:
        sys.stderr.write(f"infer_wrap: {exc}\n{USAGE}\n")
        return 2
    out_fd = protect_stdout()
    try:
        ck = cio.read_checkpoint(ckpt_path)
        engine = Engine(ck, text_ids)
        sys.stderr.write(f"infer_wrap: device={engine.device} params={sum(a.size for a in ck.params)}\n")
        write_all(out_fd, hello_line(ck))
        frames = serve(engine, sys.stdin.buffer, out_fd)
        sys.stderr.write(f"infer_wrap: {frames} 帧\n")
    except Exception as exc:  # noqa: BLE001  与 C++ 的 catch (std::exception) 同：一行错误 + 退出码 1
        sys.stderr.write(f"infer_wrap: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
