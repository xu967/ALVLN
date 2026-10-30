#!/usr/bin/env python3
"""多数类先验基线：全部预测成训练集里最多的动作，报 EVAL_STANDARD 四件套。

用途：T7 的「多数类先验」对标对象。它不读图像，只读 .uavr 的标签列和 meta 的
split 区间；多数类必须从 train 选，不能在 val/test 上看标签后再选（那是泄漏）。
默认按 meta 的 collision_flags 剔除碰撞步，与 uavrgb_train --drop-collision 同口径
（碰撞步的专家标签本身是错的）；--keep-collision 保留未过滤口径。

用法：
  python3 tools/baseline_priors.py --data run/dataset.uavr
  python3 tools/baseline_priors.py --data run/dataset.uavr --out prior.json
  python3 tools/baseline_priors.py --self-check

口径与 src/include/uavrgb/eval_metrics.hpp 一致（sklearn macro）：
  acc / macro_f1 / balanced_accuracy / 每类 recall + 混淆矩阵。
"""
import argparse
import json
import struct
import sys
from collections import Counter

NAMES = ("SAFE_NEUTRAL", "FORWARD_SLOW", "YAW_LEFT", "YAW_RIGHT")
HEADER = struct.Struct("<4sHHIHHBBBBQQQI")


def die(msg):
    sys.exit("baseline_priors: %s" % msg)


def read_dataset(path):
    try:
        with open(path, "rb") as f:
            head = f.read(HEADER.size)
            if len(head) != HEADER.size:
                die("文件头不足 48 字节")
            (magic, version, hsize, n, h, w, c, img_dtype, lab_dtype, _rsv,
             img_off, lab_off, meta_off, meta_len) = HEADER.unpack(head)
            if magic != b"UAVR" or version != 1 or hsize != HEADER.size:
                die("不是 UAVR v1 文件")
            if c != 3 or img_dtype != 1 or lab_dtype != 1:
                die(f"不支持 c={c} img_dtype={img_dtype} lab_dtype={lab_dtype}")
            if img_off != HEADER.size or lab_off != img_off + n * c * h * w:
                die("images/labels offset 不自洽")
            if meta_off != lab_off + n:
                die("meta offset 不自洽")
            f.seek(lab_off)
            labels = list(f.read(n))
            f.seek(meta_off)
            meta = json.loads(f.read(meta_len).decode("utf-8"))
    except (OSError, ValueError, struct.error) as e:
        die(f"读 {path} 失败：{type(e).__name__}: {e}")
    if len(labels) != n:
        die("标签区被截断")
    if any(v >= len(NAMES) for v in labels):
        die("标签值超出 0..3")
    return labels, meta


def split_indices(meta):
    out = {"train": [], "val": [], "test": []}
    for e in meta.get("episodes", []):
        split = e.get("split")
        if split not in out:
            die(f"episode split 非法：{split!r}")
        start, count = int(e["sample_start"]), int(e["sample_count"])
        out[split].extend(range(start, start + count))
    return out


def collision_flags(meta, n):
    """meta 的 collision_flags 是 N 个 '0'/'1'；缺字段返回 None（老数据/别处产物）。"""
    flags = meta.get("collision_flags")
    if flags is None:
        return None
    if (not isinstance(flags, str) or len(flags) != n
            or any(c not in "01" for c in flags)):
        die(f"collision_flags 长度/字符非法（应为 {n} 个 '0'/'1'）")
    return flags


def drop_collision(idx, flags):
    """与 uavrgb_train --drop-collision 同口径：碰撞步标签是错的，两边都不许算。"""
    return {s: [i for i in ids if flags[i] == "0"] for s, ids in idx.items()}


def self_check():
    meta = {"episodes": [
        {"split": "train", "sample_start": 0, "sample_count": 2},
        {"split": "val", "sample_start": 2, "sample_count": 1},
        {"split": "test", "sample_start": 3, "sample_count": 2}]}
    idx = split_indices(meta)
    assert idx == {"train": [0, 1], "val": [2], "test": [3, 4]}, idx
    assert collision_flags(meta, 5) is None
    kept = drop_collision(idx, "01001")
    assert kept == {"train": [0], "val": [2], "test": [3]}, kept
    print("baseline_priors self-check: PASS")
    return 0


def macro_metrics(pred, truth, ncls=len(NAMES)):
    """pred/truth 等长；返回 sklearn 口径的混淆矩阵与宏指标。"""
    m = [[0] * ncls for _ in range(ncls)]
    for t, p in zip(truth, pred):
        m[t][p] += 1
    total = sum(sum(row) for row in m)
    recall, precision = [], []
    for k in range(ncls):
        tp = m[k][k]
        support = sum(m[k])
        predicted = sum(m[i][k] for i in range(ncls))
        recall.append(tp / support if support else 0.0)
        precision.append(tp / predicted if predicted else 0.0)
    f1 = [2 * p * r / (p + r) if p + r else 0.0 for p, r in zip(precision, recall)]
    return {
        "accuracy": sum(m[i][i] for i in range(ncls)) / total if total else None,
        "macro_f1": sum(f1) / ncls,
        "balanced_accuracy": sum(recall) / ncls,
        "recall": {NAMES[k]: recall[k] for k in range(ncls)},
        "precision": {NAMES[k]: precision[k] for k in range(ncls)},
        "confusion": m,
        "support": {NAMES[k]: sum(m[k]) for k in range(ncls)},
    }


def main():
    p = argparse.ArgumentParser(description="多数类先验基线（不读图像）")
    p.add_argument("--data", default="", help="dataset.uavr")
    p.add_argument("--out", default="", help="写 JSON 报告")
    p.add_argument("--keep-collision", action="store_true",
                   help="保留碰撞步（默认剔除，与学生 --drop-collision 同口径）")
    p.add_argument("--self-check", action="store_true", help="跑内置断言后退出")
    args = p.parse_args()
    if args.self_check:
        return self_check()
    if not args.data:
        p.error("--data 必填")

    labels, meta = read_dataset(args.data)
    idx = split_indices(meta)
    flags = collision_flags(meta, len(labels))
    dropped = None
    if flags is None:
        if not args.keep_collision:
            print("警告: meta 无 collision_flags，无法剔除碰撞步；"
                  "与学生 --drop-collision 口径不同", file=sys.stderr)
    elif not args.keep_collision:
        kept = drop_collision(idx, flags)
        dropped = {s: len(idx[s]) - len(kept[s]) for s in idx}
        idx = kept
    if not idx["train"]:
        die("数据集没有 train 样本（或全被碰撞过滤），无法确定多数类")
    counts = Counter(labels[i] for i in idx["train"])
    majority = max(range(len(NAMES)), key=lambda k: (counts.get(k, 0), -k))
    pred = [majority] * len(labels)
    report = {
        "schema": "uavrgb.baseline_prior.v1",
        "data": args.data,
        "majority_class": NAMES[majority],
        "majority_class_index": majority,
        "train_counts": {NAMES[k]: counts.get(k, 0) for k in range(len(NAMES))},
        "collision_filtered": dropped is not None,
        "collision_dropped": dropped,
        "splits": {},
    }
    print(f"多数类（train 里最多）：{NAMES[majority]}  "
          f"train={dict(sorted(report['train_counts'].items()))}")
    if dropped is not None:
        print(f"  已按 collision_flags 剔除 {sum(dropped.values())} 步"
              f"（train/val/test={dropped['train']}/{dropped['val']}/{dropped['test']}），"
              f"与训练 --drop-collision 同口径")
    for split in ("train", "val", "test"):
        ids = idx[split]
        if not ids:
            report["splits"][split] = {"samples": 0, "note": "empty"}
            print(f"  {split:5} 空")
            continue
        m = macro_metrics([pred[i] for i in ids], [labels[i] for i in ids])
        report["splits"][split] = {"samples": len(ids), **m}
        print(f"  {split:5} n={len(ids):5d} acc={m['accuracy']:.4f} "
              f"macro_f1={m['macro_f1']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
              f"recall={ {k: round(v, 3) for k, v in m['recall'].items()} }")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
