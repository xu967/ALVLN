#!/usr/bin/env python3
"""核对 tools/average_ckpt.py 校准用的逐帧推理（uavrgb_infer + 每帧条件字节）与 uavrgb_train --eval-only 一致。

背景：uavrgb_train --eval-only 只落聚合指标（metrics.jsonl 的 validation 事件：loss、action_accuracy、macro_f1、
balanced_accuracy、confusion），不逐帧导出预测。所以这里分两路算同一个 val 集（剔除碰撞步，与训练配方一致）：
  A. 训练器路：uavrgb_train --data D --eval-only CKPT --split val --out OUT --drop-collision（条件从数据集侧车取）。
  B. 推理路：average_ckpt.read_val / read_aux_cond / infer_probs，逐帧起 uavrgb_infer，条件字节取同一侧车列。
再比较：混淆矩阵逐格相同（整数，不放容差）、dev_ce（无权 CE，trainer_eval.cpp）在 1e-4 内、acc / macro_f1 / balanced_accuracy 在 1e-6 内。
容差的来源：训练器把 acc、macro_f1 等以 float32 落盘（约 7 位有效数字），推理进程把概率以十进制文本回传（精度有限）；
它们不是给实现差异留的余量，实现差异会先反映在混淆矩阵的整数格上。
两条负对照（只对带条件输入的 ckpt 做）：条件全置 0；条件整体错位一帧。它们应当与训练器路不同，否则本核对分不出条件有没有送对。
用法：verify_infer_vs_evalonly.py --ckpt X.bin --data D/dataset.uavr --train-bin uavrgb_train --infer-bin uavrgb_infer --out DIR
退出码：0 = 真实条件一路全部相同；1 = 有不同；2 = 参数/运行错误。
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

NC = 4
CE_TOL = 1e-4
F_TOL = 1e-6


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics_from_probs(probs, y):
    """probs[n,4]、标签 y[n] → 与训练器同定义的指标（缺类按 0 计，见 src/eval_metrics.cpp：macro_f1 与 balanced_accuracy 都除以 4）。"""
    n = len(y)
    pred = probs.argmax(1)
    conf = np.zeros((NC, NC), np.int64)
    np.add.at(conf, (y.astype(np.int64), pred), 1)
    row, col, diag = conf.sum(1), conf.sum(0), np.diag(conf)
    recall = np.where(row > 0, diag / np.maximum(row, 1), 0.0)
    precision = np.where(col > 0, diag / np.maximum(col, 1), 0.0)
    f1 = np.where(recall + precision > 0, 2 * recall * precision / np.maximum(recall + precision, 1e-300), 0.0)
    ce = float(-np.log(np.clip(probs[np.arange(n), y], 1e-12, None)).mean())
    return {"confusion": conf.ravel().tolist(), "acc": float(diag.sum() / n), "ce": ce,
            "macro_f1": float(f1.sum() / NC), "balanced_acc": float(recall.sum() / NC)}


def run_eval_only(train_bin, data, ckpt, out, batch):
    cmd = [train_bin, "--data", data, "--eval-only", ckpt, "--split", "val", "--out", out, "--batch", str(batch), "--drop-collision"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"uavrgb_train --eval-only rc={r.returncode}: {(r.stdout + r.stderr).strip()[-600:]}")
    rows = [json.loads(line) for line in open(os.path.join(out, "metrics.jsonl"), encoding="utf-8") if line.strip()]
    val = [x for x in rows if x.get("event") == "validation" and x.get("split") == "val"]
    if len(val) != 1:
        raise RuntimeError(f"{out}/metrics.jsonl 里 split=val 的 validation 事件有 {len(val)} 条（期望 1 条）")
    v = val[0]
    return {"confusion": [int(c) for c in v["confusion"]], "acc": float(v["action_accuracy"]), "ce": float(v["loss"]),
            "macro_f1": float(v["macro_f1"]), "balanced_acc": float(v["balanced_accuracy"])}


def compare(a, b):
    """a = 训练器路，b = 推理路。返回 (逐项结果 dict, 是否全部相同)。"""
    items = {
        "confusion 逐格": (a["confusion"] == b["confusion"], f"训练器 {a['confusion']} 推理 {b['confusion']}"),
        "acc（容差 %g，训练器以 float32 落盘）" % F_TOL: (abs(a["acc"] - b["acc"]) < F_TOL, f"{a['acc']:.9f} vs {b['acc']:.9f}"),
        "ce（容差 %g）" % CE_TOL: (abs(a["ce"] - b["ce"]) < CE_TOL, f"{a['ce']:.8f} vs {b['ce']:.8f}（差 {abs(a['ce'] - b['ce']):.2e}）"),
        "macro_f1（容差 %g）" % F_TOL: (abs(a["macro_f1"] - b["macro_f1"]) < F_TOL, f"{a['macro_f1']:.8f} vs {b['macro_f1']:.8f}"),
        "balanced_acc（容差 %g）" % F_TOL: (abs(a["balanced_acc"] - b["balanced_acc"]) < F_TOL, f"{a['balanced_acc']:.8f} vs {b['balanced_acc']:.8f}"),
    }
    return items, all(ok for ok, _ in items.values())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--train-bin", required=True)
    ap.add_argument("--infer-bin", required=True)
    ap.add_argument("--out", required=True, help="eval-only 的输出目录（不存在则新建）与本脚本的 json 结果目录")
    ap.add_argument("--tools-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "tools"),
                    help="含 average_ckpt.py 的目录（默认仓库 tools/）")
    ap.add_argument("--batch", type=int, default=128)
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.abspath(args.tools_dir))
    import average_ckpt as ac  # 校准工具本身：同一份 read_val / read_aux_cond / infer_probs，核对的是它

    os.makedirs(args.out, exist_ok=True)
    ck = ac.read_ckpt(args.ckpt)
    prev_dim = ck["prev_dim"]
    if prev_dim not in (0, ac.PREV_DIM):
        print(f"ckpt prev_dim={prev_dim}，本核对只覆盖 0 与 {ac.PREV_DIM}", file=sys.stderr)
        return 2
    data_sha = sha256_file(args.data)
    images, idx, y = ac.read_val(args.data)
    cond = None
    if prev_dim:
        col, aux_path = ac.read_aux_cond(args.data, data_sha)
        cond = col[idx]

    trainer = run_eval_only(os.path.abspath(args.train_bin), os.path.abspath(args.data), os.path.abspath(args.ckpt),
                            os.path.join(os.path.abspath(args.out), "eval_only"), args.batch)
    cases = {}
    probs = ac.infer_probs(args.infer_bin, args.ckpt, images, idx, cond)
    cases["真实条件" if prev_dim else "无条件输入"] = metrics_from_probs(probs, y)
    if prev_dim:
        cases["对照：条件全置 0"] = metrics_from_probs(ac.infer_probs(args.infer_bin, args.ckpt, images, idx, np.zeros_like(cond)), y)
        cases["对照：条件错位一帧"] = metrics_from_probs(ac.infer_probs(args.infer_bin, args.ckpt, images, idx, np.roll(cond, 1)), y)

    print(f"ckpt {args.ckpt}（prev_dim={prev_dim}）sha256={sha256_file(args.ckpt)}")
    print(f"数据 {args.data} sha256={data_sha}；val 剔除碰撞步 {len(idx)} 帧；条件列取值计数 "
          f"{[int((cond == v).sum()) for v in range(ac.PREV_DIM)] if prev_dim else '无'}")
    print(f"训练器路 eval-only：{json.dumps(trainer)}")
    summary = {"ckpt": os.path.abspath(args.ckpt), "ckpt_sha256": sha256_file(args.ckpt), "prev_dim": prev_dim, "data": os.path.abspath(args.data),
               "data_sha256": data_sha, "val_frames": int(len(idx)), "trainer": trainer, "cases": {}}
    real_ok = None
    for name, m in cases.items():
        items, same = compare(trainer, m)
        real = name in ("真实条件", "无条件输入")
        print(f"\n[{name}] 推理路 {json.dumps(m)}")
        for k, (ok, detail) in items.items():
            print(f"  {'相同' if ok else '不同'}  {k}  {detail}")
        if real:
            real_ok = same
            print(f"  → {'与训练器路全部相同' if same else '与训练器路有不同'}")
        else:
            print(f"  → 对照{'与训练器路不同（条件确实进了推理，符合预期）' if not same else '与训练器路相同（这个模型对该扰动不敏感，对照没有分辨力，见报告）'}")
        summary["cases"][name] = {"metrics": m, "equal_to_trainer": same, "items": {k: bool(ok) for k, (ok, _) in items.items()}}
    summary["real_condition_equal"] = bool(real_ok)
    with open(os.path.join(args.out, "verify.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    print("\nRESULT:", "真实条件一路与 eval-only 全部相同" if real_ok else "真实条件一路与 eval-only 有不同")
    return 0 if real_ok else 1


if __name__ == "__main__":
    sys.exit(main())
