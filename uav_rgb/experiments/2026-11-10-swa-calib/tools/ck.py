#!/usr/bin/env python3
"""UAVRCK01 v6 checkpoint 的只读解析与「换参数后另存」（布局见 src/checkpoint.cpp save_checkpoint）。

只用于审查分析：读实验目录里的 ckpt（只读），新文件只写到草稿目录。
"""
import struct

import numpy as np

NAMES = ["c1.W", "c1.b", "c2.W", "c2.b", "c3.W", "c3.b", "fc.W", "fc.b"]


def load(path):
    b = open(path, "rb").read()
    if b[:8] != b"UAVRCK01":
        raise ValueError(f"{path}: magic 不符")
    ver, pc = struct.unpack_from("<II", b, 8)
    in_h, in_w, in_c, ncls = struct.unpack_from("<4i", b, 16)
    t, epoch, nw = struct.unpack_from("<3I", b, 32)
    off = 44
    cw = np.frombuffer(b, "<f4", nw, off).copy()
    off += 4 * nw
    best = struct.unpack_from("<f", b, off)[0]
    off += 4
    text_dim = struct.unpack_from("<i", b, off)[0]
    off += 4
    contract = b[off:off + 113]
    off += 113
    aux, prev = struct.unpack_from("<2i", b, off)
    off += 8
    header_bytes = off
    params, spans, shape_bytes = [], [], 0
    for _ in range(pc):
        nd = struct.unpack_from("<I", b, off)[0]
        off += 4
        dims = struct.unpack_from("<%dq" % nd, b, off)
        off += 8 * nd
        shape_bytes += 4 + 8 * nd
        n = int(np.prod(dims))
        params.append(np.frombuffer(b, "<f4", n, off).reshape(dims).copy())
        spans.append((off, n))
        off += 4 * n
    m, v = [], []
    for dst in (m, v):
        for p in params:
            dst.append(np.frombuffer(b, "<f4", p.size, off).reshape(p.shape).copy())
            off += 4 * p.size
    if off != len(b):
        raise ValueError(f"{path}: 尾部多 {len(b) - off} 字节")
    return {"raw": b, "version": ver, "param_count": pc, "in_h": in_h, "in_w": in_w, "in_c": in_c,
            "num_classes": ncls, "adam_t": t, "epoch": epoch, "class_weights": cw, "best_macro_f1": best,
            "text_dim": text_dim, "contract": contract, "aux_classes": aux, "prev_dim": prev,
            "header_bytes": header_bytes, "shape_bytes": shape_bytes, "params": params, "m": m, "v": v,
            "spans": spans, "size": len(b)}


def save_with_params(src, params, path):
    """以 src（load 的结果）为模板，只把 8 个参数张量换成 params 后写到 path；Adam m/v、头部原样保留。"""
    buf = bytearray(src["raw"])
    for (off, n), p in zip(src["spans"], params):
        a = np.ascontiguousarray(p, dtype="<f4").reshape(-1)
        if a.size != n or not np.isfinite(a).all():
            raise ValueError("参数形状不符或含非有限值")
        buf[off:off + 4 * n] = a.tobytes()
    with open(path, "wb") as f:
        f.write(bytes(buf))


if __name__ == "__main__":
    import sys
    c = load(sys.argv[1])
    total = sum(p.size for p in c["params"])
    print(f"version={c['version']} param_tensors={c['param_count']} in={c['in_c']}x{c['in_h']}x{c['in_w']} "
          f"classes={c['num_classes']} adam_t={c['adam_t']} epoch={c['epoch']} best_macro_f1={c['best_macro_f1']:.6f} "
          f"text_dim={c['text_dim']} aux={c['aux_classes']} prev={c['prev_dim']}")
    print(f"class_weights={c['class_weights'].tolist()}")
    print(f"header={c['header_bytes']} B, shape_headers={c['shape_bytes']} B, params={total} 个 = {4 * total} B, "
          f"file={c['size']} B = {c['header_bytes']} + {c['shape_bytes']} + 3*{4 * total}")
    for name, p in zip(NAMES, c["params"]):
        print(f"  {name:5s} shape={tuple(p.shape)} n={p.size:6d} L2={np.linalg.norm(p):8.4f} "
              f"max|w|={np.abs(p).max():.4f} mean={p.mean():+.5f} std={p.std():.5f}")
