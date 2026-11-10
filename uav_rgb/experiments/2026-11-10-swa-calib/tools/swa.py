#!/usr/bin/env python3
"""权重空间平均（SWA 类）：把同一 run 末 k 轮的 8 个参数张量取算术平均，写成新 ckpt（只写草稿目录）。

用法：
  python -B swa.py make      生成 ckpt/swa_s{seed}_{tag}.bin，并打印给 run_val.py list 用的 tag=path 列表
  python -B swa.py report    读 probs/swa_*.npy 出指标（先跑 run_val.py list）
模型没有归一化层，权重平均后不需要重估统计量；Adam 的 m/v 与头部沿用 ep30 的（推理不读）。
"""
import os
import sys

import numpy as np

import ck

RUNS = "/root/rivermind-data/uav_rgb/exp-student-select-20261109/e1-noise"
OUT = "ckpt"
SEEDS = (0, 1, 2)
WINDOWS = {"k05": range(26, 31), "k10": range(21, 31), "k15": range(16, 31), "k10m1": range(20, 30), "k20": range(11, 31)}


def make():
    os.makedirs(OUT, exist_ok=True)
    pairs = []
    for s in SEEDS:
        src = ck.load(f"{RUNS}/seed{s}/run/ckpt_ep30.bin")
        cache = {}
        for tag, eps in WINDOWS.items():
            acc = None
            for e in eps:
                if e not in cache:
                    cache[e] = ck.load(f"{RUNS}/seed{s}/run/ckpt_ep{e:02d}.bin")["params"]
                p = cache[e]
                acc = [a.astype(np.float64) for a in p] if acc is None else [a + b for a, b in zip(acc, p)]
            avg = [(a / len(eps)).astype(np.float32) for a in acc]
            path = f"{OUT}/swa_s{s}_{tag}.bin"
            ck.save_with_params(src, avg, path)
            norms = " ".join(f"{n}={np.linalg.norm(a):.2f}" for n, a in zip(ck.NAMES, avg) if n.endswith("W"))
            print(f"# {path}: {os.path.getsize(path)} B  {norms}", file=sys.stderr)
            pairs.append(f"swa_s{s}_{tag}={os.path.abspath(path)}")
    print(" ".join(pairs))


def report():
    import analyze as A
    import analyze2 as A2
    target = (A.Y[A.SL["val"]] == 1).mean()
    P = {}
    for s in SEEDS:
        for tag in WINDOWS:
            P[(s, tag)] = np.load(f"probs/swa_s{s}_{tag}.npy")
    print("== SWA（权重平均）指标：真实 uavrgb_infer 逐帧推理 ==")
    for s in SEEDS:
        best = A.probs(f"s{s}_ep{A.BEST[s]:02d}")
        for split in ("val", "test"):
            print(f"seed {s} {split}: ckpt_best(ep{A.BEST[s]})  {A.fmt(A.split_metrics(best, split))}")
            for tag in WINDOWS:
                print(f"seed {s} {split}: SWA {tag:6s}        {A.fmt(A.split_metrics(P[(s, tag)], split))}")
    print("\n== SWA + 只校准 FWD 先验（δ 在 val 上定）==")
    for s in SEEDS:
        for tag in ("k10", "k15"):
            d = A2.calibrate(P[(s, tag)], target)
            pr = A2.shifted_pred(P[(s, tag)], d)
            for split in ("val", "test"):
                sl = A.SL[split]
                print(f"seed {s} {split}: SWA {tag}+校准(δ={d:+.2f}) {A.fmt(A.metrics(A.Y[sl], pr[sl], A.EMPTY[sl]))}")
    print("\n== 稳定性 ==")
    for s in SEEDS:
        a, b = P[(s, "k10m1")].argmax(1), P[(s, "k10")].argmax(1)
        raw29, raw30 = A.probs(f"s{s}_ep29").argmax(1), A.probs(f"s{s}_ep30").argmax(1)
        print(f"seed {s}: SWA 窗口 ep20-29 → ep21-30 翻转率 val={np.mean(a[A.SL['val']] != b[A.SL['val']]):.3f} test={np.mean(a[A.SL['test']] != b[A.SL['test']]):.3f}"
              f" | 单轮 ep29→30 val={np.mean(raw29[A.SL['val']] != raw30[A.SL['val']]):.3f} test={np.mean(raw29[A.SL['test']] != raw30[A.SL['test']]):.3f}"
              f" | SWA k10 与同窗口概率平均的预测一致率 val={np.mean(b[A.SL['val']] == np.mean([A.probs(f's{s}_ep{e:02d}') for e in range(21, 31)], axis=0).argmax(1)[A.SL['val']]):.3f}")
    for split in ("val", "test"):
        sl = A.SL[split]
        for tag in ("k10", "k15"):
            d = [(P[(a, tag)].argmax(1)[sl] != P[(b, tag)].argmax(1)[sl]).mean() for a, b in ((0, 1), (0, 2), (1, 2))]
            print(f"{split} 跨种子分歧率 SWA {tag}: {np.round(d, 3).tolist()}")
    print("\n== SWA 模型的 δ（越接近、越小越好：三个种子的类先验是否一致）==")
    for tag in WINDOWS:
        print(tag, [round(A2.calibrate(P[(s, tag)], target), 2) for s in SEEDS])


if __name__ == "__main__":
    {"make": make, "report": report}[sys.argv[1]]()
