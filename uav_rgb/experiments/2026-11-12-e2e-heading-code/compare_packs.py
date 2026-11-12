#!/usr/bin/env python3
"""数据集重建核对（卡 E2E-1 第 3 步的前置）：重打一份包，与已有包逐项对账。

用法：python compare_packs.py <原包 dataset.uavr> <新包 dataset.uavr> [--ignore-episode-keys source_dir,...]
        [--cond-pack [--aux-ref <参照 dataset.aux>]]
  默认：新包是同参数重打的包，侧车（两包同目录都有 dataset.aux 时）prev_index、prev_action 都要相同。
  --cond-pack：新包是 preprocess --cond heading-code 打的包。prev_action 列允许与参照不同（航向码取代上一步动作，报告不同的行数），
    但要求：新包 meta 的 cond 标签是 heading_code.v1、原包没有 cond 标签、prev_action 列只含 0 / 2 / 3、prev_index 与参照相同。
    参照侧车缺省取原包同目录的 dataset.aux；原包没有侧车（demo 的现有数据集）时用 --aux-ref 指向同参数重打的默认包的侧车。
对账项（任何一项不符退出码 1）：
  1) 样本数、H、W、C；2) 图像区与标签区的逐字节比较（分块，不整个读进内存）；
  3) 总体与分 split 的样本数、标签直方图（由标签区与 meta.episodes 复算，不读 meta 里的 class_counts）；
  4) meta.episodes 逐集对账（除 --ignore-episode-keys 列出的键）与 collision_flags；
  5) 两个 meta 顶层键的差异（只报告：新增的键、缺的键、值不同的键；created_utc 不算差异）。
整文件 sha256 只作参考：新版 preprocess 写入的 meta 多了 aux_sidecar 等键、created_utc 也不同，整文件不会逐字节相同。
两个包同目录有 dataset.aux 时，再比 prev_index 与 prev_action 两段（头里的 uavr sha256 随整文件变，不比）。
"""
import argparse
import hashlib
import json
import os
import struct
import sys

import numpy as np

HEAD = struct.Struct("<4sHHIHHBBBBQQQI")
AUX_HEAD_SIZE = 128
CHUNK = 64 << 20


def read_head(path):
    with open(path, "rb") as f:
        magic, ver, hs, n, h, w, c, idt, ldt, _r, io, lo, mo, ml = HEAD.unpack(f.read(HEAD.size))
        if magic != b"UAVR" or ver != 1 or hs != HEAD.size:
            raise ValueError(f"{path}: 不是 UAVR v1")
        f.seek(mo)
        meta = json.loads(f.read(ml).decode("utf-8"))
    return {"n": n, "h": h, "w": w, "c": c, "io": io, "lo": lo, "mo": mo, "ml": ml, "meta": meta,
            "size": os.path.getsize(path)}


def region_equal(pa, a_off, pb, b_off, length):
    """两个文件各自从 offset 起 length 字节是否逐字节相同；返回 (相同, 第一处不同的相对偏移或 None)。"""
    with open(pa, "rb") as fa, open(pb, "rb") as fb:
        fa.seek(a_off)
        fb.seek(b_off)
        done = 0
        while done < length:
            k = min(CHUNK, length - done)
            x, y = fa.read(k), fb.read(k)
            if x != y:
                d = np.flatnonzero(np.frombuffer(x, np.uint8) != np.frombuffer(y, np.uint8))
                return False, done + int(d[0]) if d.size else done
            done += k
    return True, None


def labels_of(path, head):
    return np.fromfile(path, dtype=np.uint8, count=head["n"], offset=head["lo"])


def hist(labels):
    return [int((labels == k).sum()) for k in range(4)]


def per_split(head, labels):
    out = {}
    for ep in head["meta"]["episodes"]:
        s, c = ep["sample_start"], ep["sample_count"]
        acc = out.setdefault(ep["split"], [0, [0, 0, 0, 0]])
        acc[0] += c
        for k, v in enumerate(hist(labels[s:s + c])):
            acc[1][k] += v
    return {k: (v[0], v[1]) for k, v in out.items()}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_aux_cols(path, n):
    with open(path, "rb") as f:
        head = f.read(AUX_HEAD_SIZE)
        if head[:8] != b"UAVRAUX1":
            raise ValueError(f"{path}: 不是 UAVRAUX1")
        n_aux = struct.unpack_from("<Q", head, 8)[0]
        if n_aux != n:
            raise ValueError(f"{path}: N={n_aux} 与数据集 {n} 不符")
        prev_index = np.frombuffer(f.read(4 * n), dtype="<u4")
        prev_action = np.frombuffer(f.read(n), dtype=np.uint8)
    return prev_index, prev_action


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("orig")
    ap.add_argument("new")
    ap.add_argument("--ignore-episode-keys", default="source_dir")
    ap.add_argument("--cond-pack", action="store_true",
                    help="新包是 preprocess --cond heading-code 的包：侧车 prev_action 列允许与参照不同（报告行数），"
                         "但 meta 须带 cond=heading_code.v1、列值只含 0/2/3、prev_index 须与参照相同")
    ap.add_argument("--aux-ref", help="侧车参照（dataset.aux 路径）；缺省取原包同目录的 dataset.aux")
    args = ap.parse_args(argv)
    ignore = {k for k in args.ignore_episode_keys.split(",") if k}
    a, b = read_head(args.orig), read_head(args.new)
    ok = True

    def verdict(name, good, detail=""):
        nonlocal ok
        ok &= bool(good)
        print(f"{'相同' if good else '不同'}  {name}{('  ' + detail) if detail else ''}")

    print(f"原包 {args.orig}\n新包 {args.new}")
    verdict("样本数 N、H、W、C", (a["n"], a["h"], a["w"], a["c"]) == (b["n"], b["h"], b["w"], b["c"]),
            f"原 {(a['n'], a['h'], a['w'], a['c'])} 新 {(b['n'], b['h'], b['w'], b['c'])}")
    if (a["n"], a["h"], a["w"], a["c"]) == (b["n"], b["h"], b["w"], b["c"]):
        img_len = a["n"] * a["c"] * a["h"] * a["w"]
        same, at = region_equal(args.orig, a["io"], args.new, b["io"], img_len)
        verdict("图像区逐字节", same, f"{img_len} B" + ("" if same else f"，第一处不同在图像区偏移 {at}"))
        same, at = region_equal(args.orig, a["lo"], args.new, b["lo"], a["n"])
        verdict("标签区逐字节", same, f"{a['n']} B" + ("" if same else f"，第一处不同在样本 {at}"))
        la, lb = labels_of(args.orig, a), labels_of(args.new, b)
        verdict("总体标签直方图", hist(la) == hist(lb), f"原 {hist(la)} 新 {hist(lb)}")
        pa, pb = per_split(a, la), per_split(b, lb)
        verdict("分 split 样本数与标签直方图", pa == pb)
        for s in sorted(set(pa) | set(pb)):
            print(f"      {s:5s} 原 {pa.get(s)}  新 {pb.get(s)}")
    ea, eb = a["meta"]["episodes"], b["meta"]["episodes"]
    verdict("集数", len(ea) == len(eb), f"原 {len(ea)} 新 {len(eb)}")
    diffs, new_only = [], {}
    for i, (x, y) in enumerate(zip(ea, eb)):
        xs = {k: v for k, v in x.items() if k not in ignore}
        ys = {k: v for k, v in y.items() if k not in ignore}
        # 后来版本的 preprocess 往每集条目里加过键（beta、dagger_round 等）：新包独有的键只报告，不算差异；
        # 原包独有的键（新包丢了信息）与公共键的值不同才算差异。
        for k in set(ys) - set(xs):
            new_only.setdefault(k, set()).add(json.dumps(ys[k], sort_keys=True))
        keys = sorted(k for k in set(xs) if xs.get(k) != ys.get(k) or k not in ys)
        if keys:
            diffs.append((i, x.get("episode_id"), keys))
    verdict(f"meta.episodes 逐集（公共键相等、原包独有键不丢；忽略键 {sorted(ignore)}）", not diffs and len(ea) == len(eb),
            "" if not diffs else f"{len(diffs)} 集有差异，前 5 个 {diffs[:5]}")
    print(f"参考  新包条目里多出的键（各取值集合）：{ {k: sorted(v) for k, v in sorted(new_only.items())} }")
    verdict("collision_flags", a["meta"].get("collision_flags") == b["meta"].get("collision_flags"))
    ka, kb = a["meta"], b["meta"]
    only_a = sorted(set(ka) - set(kb))
    only_b = sorted(set(kb) - set(ka))
    changed = sorted(k for k in set(ka) & set(kb) if k not in ("created_utc", "episodes", "collision_flags") and ka[k] != kb[k])
    print(f"参考  meta 顶层键：只在原包 {only_a}；只在新包 {only_b}；值不同 {changed}（created_utc 不算）")
    print(f"参考  整文件 sha256：原 {sha256_file(args.orig)[:16]}… 新 {sha256_file(args.new)[:16]}…")
    aa = args.aux_ref or os.path.join(os.path.dirname(args.orig), "dataset.aux")
    ab = os.path.join(os.path.dirname(args.new), "dataset.aux")
    if args.cond_pack:
        verdict("新包 meta 的 cond 标签", b["meta"].get("cond") == "heading_code.v1", f"得到 {b['meta'].get('cond')!r}")
        verdict("原包没有 cond 标签", a["meta"].get("cond") is None, f"得到 {a['meta'].get('cond')!r}")
    if os.path.isfile(aa) and os.path.isfile(ab):
        (ia, pa_), (ib, pb_) = read_aux_cols(aa, a["n"]), read_aux_cols(ab, b["n"])
        verdict("侧车 prev_index", bool((ia == ib).all()), f"参照 {aa}")
        diff_rows = int((pa_ != pb_).sum())
        if args.cond_pack:
            vals = sorted(int(v) for v in np.unique(pb_))
            counts = {v: int((pb_ == v).sum()) for v in vals}
            verdict("新包 prev_action 列只含码 0/2/3", set(vals) <= {0, 2, 3}, f"取值计数 {counts}")
            print(f"参考  prev_action 列与参照不同的行数 {diff_rows}/{a['n']}（航向码对上一步动作，允许不同）")
        else:
            verdict("侧车 prev_action", diff_rows == 0, f"不同的行数 {diff_rows}")
    else:
        if args.cond_pack:
            verdict("侧车可对（参照与新包都有 dataset.aux）", False, f"参照 {aa} {'有' if os.path.isfile(aa) else '无'}，新包 {'有' if os.path.isfile(ab) else '无'}")
        else:
            print(f"参考  侧车：原包 {'有' if os.path.isfile(aa) else '无'}，新包 {'有' if os.path.isfile(ab) else '无'}（不都有就不比）")
    print("RESULT:", "核对项全部相同" if ok else "有不同，见上")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
