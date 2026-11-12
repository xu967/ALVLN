#!/usr/bin/env python3
"""核对 PyTorch 训练产物能被 C++ 读回：同一份 ckpt_best.bin，proto/train.py 在 val/test 上算的混淆矩阵与主线
`uavrgb_train --eval-only <ckpt_best.bin> --split all --drop-collision` 算的逐格相同，CE 差不超过 CE_TOL。

只依赖标准库。用法：
  python proto/crosscheck_evalonly.py --run <proto run 目录> --eval-dir <eval-only 的 --out 目录> [--out <json>]
退出码：0 一致；1 有差异；2 读不到文件。
"""
import argparse
import json
import sys

CE_TOL = 1e-5     # 两边 CE 的差：CPU/GPU 的 exp、log 与累加顺序不同，实测 4e-8 量级；留两个数量级余量


def validation_events(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"validation"' not in line:
                continue
            e = json.loads(line)
            if e.get("event") == "validation":
                out.setdefault(e["split"], []).append(e)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--eval-dir", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    try:
        mine = validation_events(f"{a.run}/metrics.jsonl")
        cpp = validation_events(f"{a.eval_dir}/metrics.jsonl")
    except OSError as exc:
        print(f"读不到文件：{exc}", file=sys.stderr)
        return 2
    best_epoch = None
    with open(f"{a.run}/metrics.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"best checkpoint: epoch ' in line:
                best_epoch = int(json.loads(line)["message"].split("epoch ")[1].split()[0])
    res = {"run": a.run, "eval_dir": a.eval_dir, "best_epoch": best_epoch, "splits": {}}
    ok = True
    for split in ("val", "test"):
        # proto 的 val 事件每个 epoch 一条，交付的是 best_epoch 那条；test 只有一条
        mine_ev = mine[split][best_epoch - 1] if split == "val" else mine[split][-1]
        cpp_ev = cpp[split][-1]
        same_conf = mine_ev["confusion"] == cpp_ev["confusion"]
        dce = abs(mine_ev["loss"] - cpp_ev["loss"])
        res["splits"][split] = {"confusion_identical": same_conf, "ce_proto": mine_ev["loss"], "ce_cpp": cpp_ev["loss"],
                                "ce_abs_diff": dce, "macro_f1_proto": mine_ev["macro_f1"], "macro_f1_cpp": cpp_ev["macro_f1"]}
        ok = ok and same_conf and dce <= CE_TOL
    res["identical"] = ok
    text = json.dumps(res, ensure_ascii=False, indent=1)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
