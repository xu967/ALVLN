#!/usr/bin/env python3
"""工作点扫描的权重：在现成的 cal_swa_s{0,1,2}_k10.bin（末 10 轮平均 + 前进校准）上再把 FORWARD 的 fc.b 平移 −0.5 与 +0.5 nat（任务卡 E2E-R1 第二部分）。

只做一件事：用 tools/average_ckpt.py 里现成的 read_ckpt / shift_forward_bias / write_ckpt（库代码原样 import，不改），不训练。
0 档不生成文件：直接用原权重（脚本把原文件的 sha256 记下来，并核对「平移 0」得到的字节与原文件逐字节相同，证明这条写出路径不引入别的改动）。
每个写出的文件都核对：除 fc.b 的 FORWARD 那一个 float32 以外，其余字节与原文件逐字节相同；FORWARD 偏置的变化量等于 delta（float32 舍入内，容差 1e-6）。
输出 shifts.json：每个文件的来源 sha256、新 sha256、fc.b 平移前后的四个值、核对结果。

用法：make_shifted_ckpts.py --src-dir <含 cal_swa_s{0,1,2}_k10.bin 的目录> --out-dir <输出目录> [--shifts=-0.5,0.5]
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tools"))
import average_ckpt as ac  # noqa: E402

# ───────────────────────────── 配置区（改这里） ─────────────────────────────
SEEDS = (0, 1, 2)                 # 三个训练种子：任务卡
SHIFTS = (-0.5, 0.5)              # 任务卡：现有校准值之上再加的 nat；0 档用原文件，不在这里
SRC_NAME = "cal_swa_s{seed}_k10.bin"   # 源权重文件名（$D/exp-swa-calib-20261110/ckpt）
TOL = 1e-6                        # FORWARD 偏置变化量与 delta 的容差：float32 在 |b|≈1 附近的最小间隔约 1.2e-7，这里放宽到 1e-6
# ──────────────────────────────────────────────────────────────────────────


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def shift_tag(delta):
    """文件名里的档位记号：-0.5 → fwdm0.5，+0.5 → fwdp0.5，0 → fwd0。"""
    return "fwd0" if delta == 0 else f"fwd{'m' if delta < 0 else 'p'}{abs(delta):g}"


def write_shifted(src, delta, path):
    """src（read_ckpt 的返回）平移 FORWARD 偏置 delta，用库里的 write_ckpt 写到 path（头、契约、Adam 段原样），返回写出的整份文件字节。"""
    ac.write_ckpt(src, ac.shift_forward_bias(src["params"], delta), path)
    with open(path, "rb") as f:
        return f.read()


def check_shift(src, new_bytes, delta, path_for_msg=""):
    """核对 new_bytes 相对 src：长度相同；除 fc.b[FORWARD] 那 4 个字节外逐字节相同；FORWARD 偏置变化量 = delta（容差 TOL）。返回记录 dict，不过抛 ValueError。"""
    raw = src["raw"]
    if len(new_bytes) != len(raw):
        raise ValueError(f"{path_for_msg}: 长度 {len(new_bytes)} 与源 {len(raw)} 不同")
    off_b, _ = src["spans"][ac.FC_B]
    pos = off_b + 4 * ac.FORWARD
    diff = [i for i in range(len(raw)) if raw[i] != new_bytes[i]]
    if delta == 0 and diff:
        raise ValueError(f"{path_for_msg}: 平移 0 却有 {len(diff)} 个字节不同")
    if any(not (pos <= i < pos + 4) for i in diff):
        bad = [i for i in diff if not (pos <= i < pos + 4)]
        raise ValueError(f"{path_for_msg}: fc.b[FORWARD] 之外有 {len(bad)} 个字节不同（首个偏移 {bad[0]}）")
    before = float(np.frombuffer(raw, "<f4", 1, pos)[0])
    after = float(np.frombuffer(new_bytes, "<f4", 1, pos)[0])
    if abs((after - before) - delta) > TOL:
        raise ValueError(f"{path_for_msg}: FORWARD 偏置变化 {after - before:+.9f}，应为 {delta:+g}（容差 {TOL:g}）")
    return {"fc_b_forward_before": before, "fc_b_forward_after": after, "delta_actual": after - before, "bytes_changed": len(diff)}


def make_all(src_dir, out_dir, shifts=SHIFTS, seeds=SEEDS):
    os.makedirs(out_dir, exist_ok=True)
    rec = {"schema": "e2e-r1.shifted_ckpts.v1", "src_dir": os.path.abspath(src_dir), "shifts": list(shifts), "files": []}
    for s in seeds:
        src_path = os.path.join(src_dir, SRC_NAME.format(seed=s))
        src = ac.read_ckpt(src_path)
        src_sha = ac.sha256_file(src_path)
        zero_path = os.path.join(out_dir, f".zero-check-s{s}.bin")      # 自己建的临时文件，核对完删
        zero = write_shifted(src, 0.0, zero_path)
        os.remove(zero_path)
        if zero != src["raw"]:
            raise ValueError(f"{src_path}: 平移 0 得到的字节与原文件不同，写出路径不是恒等")
        rec["files"].append({"seed": s, "delta": 0.0, "path": src_path, "sha256": src_sha, "source": "原文件，不拷贝", "bytes_changed": 0,
                             "fc_b_forward_before": float(src["params"][ac.FC_B][ac.FORWARD]), "fc_b_forward_after": float(src["params"][ac.FC_B][ac.FORWARD])})
        for d in shifts:
            out_path = os.path.join(out_dir, SRC_NAME.format(seed=s).replace(".bin", f"_{shift_tag(d)}.bin"))
            nb = write_shifted(src, d, out_path)
            info = check_shift(src, nb, d, out_path)
            back = ac.read_ckpt(out_path)       # 写出的文件要能被读回，且与写出的字节一致
            if back["raw"] != nb:
                raise ValueError(f"{out_path}: 读回与写出不一致")
            rec["files"].append({"seed": s, "delta": d, "path": out_path, "sha256": sha256_bytes(nb), "src_sha256": src_sha, **info})
    with open(os.path.join(out_dir, "shifts.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, ensure_ascii=False, indent=1)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shifts", default=",".join(f"{x:g}" for x in SHIFTS), help="逗号分隔的 nat，缺省 -0.5,0.5（写成 --shifts=-0.5,0.5）")
    args = ap.parse_args(argv)
    shifts = tuple(float(x) for x in args.shifts.split(",") if x.strip())
    rec = make_all(args.src_dir, args.out_dir, shifts)
    for r in rec["files"]:
        print(f"s{r['seed']} 平移 {r['delta']:+g}：fc.b[FORWARD] {r['fc_b_forward_before']:+.7f} → {r['fc_b_forward_after']:+.7f}（变化 {r['fc_b_forward_after'] - r['fc_b_forward_before']:+.7f}，改动 {r['bytes_changed']} 字节）sha256 {r['sha256'][:12]} {r['path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
