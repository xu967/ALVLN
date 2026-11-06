#!/usr/bin/env python3
"""训练 / 闭环两条输入路径的数值对拍（TASKS.md B4）。

同一帧 gz.msgs.Image 走两条路，比较到达模型输入（uint8 NCHW）时的最大绝对差：
  路 A（训练）：帧 -> collect.write_rgb_png 的无损 PNG（save_frame._png）
                 -> tools/preprocess.load_frame（PIL convert("RGB") + resize BILINEAR，HWC->CHW）
  路 B（闭环）：帧 -> sim/tools/student.StudentPolicy.act
                 （save_frame.decode -> PIL frombytes/convert/resize -> tobytes HWC）
                 -> uavrgb_infer 的 HWC->CHW（src/infer.cpp）
两条路之后是同一个设备 kernel x/255.0f（src/augment_dev.cu，训练 DeviceCnn 与推理共用），
所以 uint8 逐位一致 ⇔ 模型输入逐位一致。

用法（仓库根）：
  .venv-tools/bin/python tools/check_input_parity.py                      # 合成帧：RGB / RGBA / BGR 三种像素格式
  .venv-tools/bin/python tools/check_input_parity.py --png <rgb_00000.png>  # 真实采集帧（320x240）
  .venv-tools/bin/python tools/check_input_parity.py --png <png> \\
      --infer-bin <build>/uavrgb_infer --ckpt <run>/ckpt_best.bin         # 再把两路字节喂真推理进程，比较 action/probs
退出码 0 = 全部一致；1 = 有差异或推理输出不同。
"""
import argparse
import base64
import json
import os
import sys
import tempfile

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sim", "tools"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))
import save_frame  # noqa: E402
import student  # noqa: E402
import preprocess  # noqa: E402


def gz_msg(w, h, nch, data, fmt):
    return {"width": w, "height": h, "step": w * nch, "pixelFormatType": fmt,
            "data": base64.b64encode(data).decode()}


def path_train(msg, size, png_path=None):
    """collect.write_rgb_png 的字节级等价（save_frame.decode + _png），再过 preprocess.load_frame。"""
    w, h, nch, rows = save_frame.decode(msg)
    with tempfile.TemporaryDirectory() as td:
        if png_path is None:
            png_path = os.path.join(td, "rgb_00000.png")
            with open(png_path, "wb") as f:
                f.write(save_frame._png(w, h, nch, rows))
        return preprocess.load_frame(png_path, w, h, size)  # CHW uint8


def path_closed_loop(msg, size):
    """StudentPolicy.act 的真实代码路径；只把子进程交换换成捕获 payload。"""
    captured = []
    pol = student.StudentPolicy.__new__(student.StudentPolicy)
    pol.w, pol.h, pol.in_c, pol.frames = size[0], size[1], 3, 0
    pol._exchange = lambda payload, stage: (
        captured.append(bytes(payload)) or b'{"action":0,"probs":[1,0,0,0],"infer_ms":0}\n')
    pol.act(msg)
    hwc = np.frombuffer(captured[0], dtype=np.uint8).reshape(size[1], size[0], 3)
    return np.ascontiguousarray(hwc.transpose(2, 0, 1))  # infer.cpp 的 HWC->CHW


def compare(name, a, b):
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    bad = int((diff > 0).sum())
    print("%-28s max|A-B|=%d  不同像素=%d/%d  %s"
          % (name, int(diff.max()), bad, diff.size, "一致" if bad == 0 else "不一致"))
    return bad == 0


def infer_once(pol, payload):
    line = pol._exchange(payload, "parity")
    obj = json.loads(line)
    return int(obj["action"]), [float(p) for p in obj["probs"]]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--png", help="真实采集帧 rgb_%05d.png（缺省用合成帧）")
    p.add_argument("--size", default="160x120", help="模型输入 WxH（preprocess --size 口径）")
    p.add_argument("--infer-bin", help="uavrgb_infer 路径；与 --ckpt 同给时做端到端推理对拍")
    p.add_argument("--ckpt", help="checkpoint 路径")
    args = p.parse_args()
    size = tuple(int(v) for v in args.size.lower().split("x"))

    cases = []
    if args.png:
        with Image.open(args.png) as im:
            im = im.convert("RGB")
            w, h = im.size
            rgb = im.tobytes()
        cases.append(("真实帧 RGB %dx%d" % (w, h), gz_msg(w, h, 3, rgb, "RGB_INT8"), args.png))
    else:
        rng = np.random.default_rng(0)
        w, h = 320, 240
        rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        cases.append(("合成 RGB_INT8", gz_msg(w, h, 3, rgb.tobytes(), "RGB_INT8"), None))
        rgba = np.concatenate([rgb, np.full((h, w, 1), 255, np.uint8)], axis=2)
        cases.append(("合成 RGBA_INT8", gz_msg(w, h, 4, rgba.tobytes(), "RGBA_INT8"), None))
        cases.append(("合成 BGR_INT8", gz_msg(w, h, 3, rgb[:, :, ::-1].tobytes(), "BGR_INT8"), None))

    ok = True
    first = None
    for name, msg, png_path in cases:
        a = path_train(msg, size, png_path)
        b = path_closed_loop(msg, size)
        ok &= compare(name, a, b)
        if first is None:
            first = (a, b, msg)
    if not args.png:
        # 三种像素格式解码后应是同一幅图：B 路结果互相也要一致。
        ok &= compare("RGB vs RGBA vs BGR（B 路）", path_closed_loop(cases[0][1], size),
                      path_closed_loop(cases[1][1], size))

    if args.infer_bin or args.ckpt:
        if not (args.infer_bin and args.ckpt):
            p.error("--infer-bin 与 --ckpt 要同时给")
        a, _b, msg = first
        pol = student.StudentPolicy(args.ckpt, args.infer_bin)
        try:
            if (pol.w, pol.h) != size:
                p.error("checkpoint 输入 %dx%d 与 --size 不符" % (pol.w, pol.h))
            act_b, probs_b = pol.act(msg)[0], pol.act(msg)[2]
            # A 路字节按 infer 协议（HWC）喂同一个进程。
            act_a, probs_a = infer_once(pol, np.ascontiguousarray(a.transpose(1, 2, 0)).tobytes())
        finally:
            pol.close()
        same = act_a == act_b and probs_a == probs_b
        print("端到端推理  A: action=%d probs=%s\n            B: action=%d probs=%s  %s"
              % (act_a, probs_a, act_b, probs_b, "一致" if same else "不一致"))
        ok &= same
    print("check_input_parity:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
