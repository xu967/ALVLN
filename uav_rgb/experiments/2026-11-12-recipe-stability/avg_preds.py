#!/usr/bin/env python3
"""对一个 run 做「末 k 轮权重平均」与「平均 + FORWARD 校准」，再用真实 uavrgb_infer 在 val 上落盘逐帧预测（任务卡 E2E-R1 第一部分）。

用法：avg_preds.py --run <run 目录> --data <dataset.uavr> --out <输出目录> --infer-bin <uavrgb_infer> [--last-k 10]
产物（<out> 必须是新目录）：
  avg_raw.bin / avg_cal.bin        末 k 轮平均、平均加校准的 ckpt（各带 .json：来源 sha256、δ、校准前后 FORWARD 占比），都由 tools/average_ckpt.py 写出，本脚本不改它
  preds_avg_raw.npy / preds_avg_cal.npy   val 逐帧 [idx, y, pred, p0..p3]，布局与 tools/select_ckpt.py 的 preds_epNN.npy 相同
pred 是 uavrgb_infer 输出概率（6 位有效数字）的 argmax，不是推理进程自己报的 action；两者只在概率并列到第 6 位时才会不同，
校准工具对这种差别的实测上限是 1 帧（tools/average_ckpt.py:VERIFY_FRAMES 的注释）。
GPU 由调用方用 CUDA_VISIBLE_DEVICES 钉。
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))   # 工作树与冻结副本里都是 experiments/<名>/ 的上两级放着 tools/
import average_ckpt as ac  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="--save-every-epoch 训出的 run 目录")
    ap.add_argument("--data", required=True, help="训练用的那份 dataset.uavr（校准要核对 sha256）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--infer-bin", required=True)
    ap.add_argument("--last-k", type=int, default=ac.LAST_K)
    args = ap.parse_args(argv)
    out = os.path.abspath(args.out)
    if os.path.exists(out) and os.listdir(out):
        sys.exit(f"avg_preds: {out} 非空，换新目录")
    os.makedirs(out, exist_ok=True)
    ckpts = {"raw": os.path.join(out, "avg_raw.bin"), "cal": os.path.join(out, "avg_cal.bin")}
    common = ["--run", args.run, "--last-k", str(args.last_k), "--infer-bin", args.infer_bin]
    ac.main(common + ["--out", ckpts["raw"], "--no-calibrate"])
    ac.main(common + ["--out", ckpts["cal"], "--data", args.data])
    images, idx, y = ac.read_val(args.data)
    for tag, ck in ckpts.items():
        probs = ac.infer_probs(args.infer_bin, ck, images, idx)
        np.save(os.path.join(out, f"preds_avg_{tag}.npy"), np.c_[idx, y, probs.argmax(1), probs])
        print(f"avg_preds: {tag} val {len(idx)} 帧 → preds_avg_{tag}.npy，FORWARD 预测占比 {(probs.argmax(1) == ac.FORWARD).mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
