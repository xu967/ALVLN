#!/usr/bin/env python3
"""把「只校准 FWD 先验」落成可直接闭环评测的 ckpt：fc.b[FORWARD_SLOW] += δ（δ 在 val 上定，目标 = 预测 FWD
占比等于 val 标签 FWD 占比）。只写草稿目录。

  python -B calib.py make     生成 ckpt/cal_*.bin，打印 run_val.py list 用的 tag=path
  python -B calib.py report   读 probs/cal_*.npy：与离线平移的预测逐帧对拍 + 指标
"""
import os
import sys

import numpy as np

import analyze as A
import analyze2 as A2
import ck

RUNS = A.RUNS
SRC = {}
for s in A.SEEDS:
    SRC[f"cal_swa_s{s}_k10"] = (f"ckpt/swa_s{s}_k10.bin", f"swa_s{s}_k10")
    SRC[f"cal_s{s}_ep30"] = (f"{RUNS}/seed{s}/run/ckpt_ep30.bin", f"s{s}_ep30")
    SRC[f"cal_s{s}_best"] = (f"{RUNS}/seed{s}/run/ckpt_ep{A.BEST[s]:02d}.bin", f"s{s}_ep{A.BEST[s]:02d}")
TARGET = (A.Y[A.SL["val"]] == 1).mean()


def make():
    pairs = []
    for tag, (path, ptag) in SRC.items():
        d = A2.calibrate(A.probs(ptag), TARGET)
        c = ck.load(path)
        params = [p.copy() for p in c["params"]]
        params[7][1] += np.float32(d)
        out = f"ckpt/{tag}.bin"
        ck.save_with_params(c, params, out)
        print(f"# {out}: fc.b[FWD] {c['params'][7][1]:+.4f} → {params[7][1]:+.4f}（δ={d:+.3f} nat）", file=sys.stderr)
        pairs.append(f"{tag}={os.path.abspath(out)}")
    print(" ".join(pairs))


def report():
    P = {tag: np.load(f"probs/{tag}.npy") for tag in SRC}
    for tag, (_path, ptag) in SRC.items():
        d = A2.calibrate(A.probs(ptag), TARGET)
        off = A2.shifted_pred(A.probs(ptag), d)
        same = np.mean(off == P[tag].argmax(1))
        print(f"{tag:18s} δ={d:+.2f} 真实推理与离线平移的预测一致率={same:.4f}")
        for split in ("val", "test"):
            print(f"{'':18s} {split}: {A.fmt(A.split_metrics(P[tag], split))}")
    for kind in ("cal_swa_s{}_k10", "cal_s{}_ep30", "cal_s{}_best"):
        for split in ("val", "test"):
            sl = A.SL[split]
            d = [(P[kind.format(a)].argmax(1)[sl] != P[kind.format(b)].argmax(1)[sl]).mean() for a, b in ((0, 1), (0, 2), (1, 2))]
            print(f"跨种子分歧率 {kind.format('*'):16s} {split}: {np.round(d, 3).tolist()}")


if __name__ == "__main__":
    {"make": make, "report": report}[sys.argv[1]]()
