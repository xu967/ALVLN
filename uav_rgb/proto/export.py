#!/usr/bin/env python3
"""PyTorch 训练产物 -> C++ checkpoint（UAVRCK01 v6），供主线工具（tools/average_ckpt.py、uavrgb_infer、uavrgb_train --eval-only）直接读。

proto/train.py 每个 epoch 在 <run>/pt/ 下写 PyTorch 侧的状态文件（torch.save 的张量字典，格式名 proto-train-v1）：
  epNN.pt     第 NN 轮结束时的权重、Adam 的 m/v/t、epoch、类别权重、best_macro_f1、训练契约
  last.pt     最近一轮（对应 C++ 的 ckpt.bin）
  best.pt     val 宏 F1 最高的一轮（对应 ckpt_best.bin）
本模块把它们逐个写成 C++ 的文件名与布局：<run>/ckpt_epNN.bin、ckpt.bin、ckpt_best.bin。权重与 Adam 状态按 cnn_params 的顺序原样写入，
头部字段与 113 B 训练契约来自 .pt 里记的值，不在这里重新计算。

用法（仓库根）：
  python proto/export.py --run <run 目录>                      把 <run>/pt/*.pt 全部导出到 <run>
  python proto/export.py --pt <文件.pt> --out <文件.bin>       导出单个文件
目标文件已存在：内容一致就跳过，不一致就报错（--overwrite 才覆盖）。
"""
import argparse
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io as cio  # noqa: E402

PT_FORMAT = "proto-train-v1"
STRUCT_KEYS = ("in_h", "in_w", "in_c", "num_classes", "text_dim", "aux_classes", "prev_dim")


def _torch():
    import torch  # 延迟导入：export 的函数在没有 torch 的环境里只能读 .bin，不能读 .pt
    return torch


def save_pt(path, *, structure, params, adam_m, adam_v, adam_t, epoch, class_weights, best_macro_f1, contract, extra=None):
    """写一个 .pt。params/adam_m/adam_v 是 numpy float32 数组列表（cnn_params 顺序）；先写临时文件再 rename。"""
    torch = _torch()
    state = {
        "format": PT_FORMAT,
        "structure": {k: int(structure[k]) for k in STRUCT_KEYS},
        "params": [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in params],
        "adam_m": [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in adam_m],
        "adam_v": [torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) for a in adam_v],
        "adam_t": int(adam_t),
        "epoch": int(epoch),
        "class_weights": torch.from_numpy(np.ascontiguousarray(class_weights, dtype=np.float32)),
        "best_macro_f1": float(best_macro_f1),
        "contract": torch.tensor(list(bytes(contract)), dtype=torch.uint8),
    }
    if extra:
        state["extra"] = dict(extra)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_pt(path):
    torch = _torch()
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or state.get("format") != PT_FORMAT:
        raise cio.CkptError(f"{path}: 不是 {PT_FORMAT} 格式的训练状态")
    return state


def pt_to_ckpt(state):
    """load_pt 的结果 -> ckpt_io.Ckpt（形状、有限值、结构由 ckpt_io 的编码器再校验一遍）。"""
    st = state["structure"]
    arrs = lambda key: [t.numpy().astype(np.float32, copy=True) for t in state[key]]  # noqa: E731
    return cio.Ckpt(**{k: int(st[k]) for k in STRUCT_KEYS}, params=arrs("params"), adam_m=arrs("adam_m"),
                    adam_v=arrs("adam_v"), adam_t=int(state["adam_t"]), epoch=int(state["epoch"]),
                    class_weights=state["class_weights"].numpy().astype(np.float32, copy=True),
                    best_macro_f1=float(state["best_macro_f1"]), contract=bytes(state["contract"].numpy().tobytes()))


def export_pt(pt_path, bin_path, overwrite=False):
    """一个 .pt -> 一个 .bin。返回 'written' | 'same'。"""
    ck = pt_to_ckpt(load_pt(pt_path))
    raw = cio.encode_checkpoint(ck)
    if os.path.exists(bin_path):
        with open(bin_path, "rb") as f:
            same = f.read() == raw
        if same:
            return "same"
        if not overwrite:
            raise cio.CkptError(f"{bin_path} 已存在且内容不同（--overwrite 才覆盖）")
    cio.write_checkpoint(bin_path, ck)
    return "written"


def export_run(run_dir, overwrite=False):
    """<run>/pt/*.pt -> <run>/ckpt_epNN.bin、ckpt.bin、ckpt_best.bin。返回 [(源, 目标, 结果)]。"""
    pt_dir = os.path.join(run_dir, "pt")
    if not os.path.isdir(pt_dir):
        raise cio.CkptError(f"{run_dir} 里没有 pt/ 目录")
    jobs = []
    for name in sorted(os.listdir(pt_dir)):
        m = re.fullmatch(r"ep(\d+)\.pt", name)
        if m:
            jobs.append((name, f"ckpt_ep{int(m.group(1)):02d}.bin"))
        elif name == "last.pt":
            jobs.append((name, "ckpt.bin"))
        elif name == "best.pt":
            jobs.append((name, "ckpt_best.bin"))
    if not jobs:
        raise cio.CkptError(f"{pt_dir} 里没有 epNN.pt / last.pt / best.pt")
    out = []
    for src, dst in jobs:
        out.append((src, dst, export_pt(os.path.join(pt_dir, src), os.path.join(run_dir, dst), overwrite=overwrite)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", help="proto/train.py 的 run 目录（含 pt/）")
    ap.add_argument("--pt", help="单个 .pt 文件")
    ap.add_argument("--out", help="与 --pt 配合：输出的 .bin 路径")
    ap.add_argument("--overwrite", action="store_true", help="目标已存在且内容不同时覆盖")
    a = ap.parse_args(argv)
    if bool(a.run) == bool(a.pt):
        ap.error("--run 与 --pt 二选一")
    if a.pt:
        if not a.out:
            ap.error("--pt 需要 --out")
        print(f"{a.pt} -> {a.out}: {export_pt(a.pt, a.out, overwrite=a.overwrite)}")
        return 0
    res = export_run(a.run, overwrite=a.overwrite)
    for src, dst, r in res:
        print(f"pt/{src} -> {dst}: {r}")
    print(f"共 {len(res)} 个文件，写出 {sum(r == 'written' for _, _, r in res)} 个，已一致 {sum(r == 'same' for _, _, r in res)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
