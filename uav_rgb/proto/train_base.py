#!/usr/bin/env python3
"""训练 ``base_torch_v1``：一个可复现实验用的 Torch-only 基线。

这个入口故意不写 C++ checkpoint。它的作用是让 YAML profile 真正可运行，
并给较大模型一条和 ``proto/train.py`` 分开的实验路径。
"""
import argparse
import fcntl
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import base_model  # noqa: E402
import config as cfgio  # noqa: E402
import data as dataio  # noqa: E402
import metrics as metricio  # noqa: E402


def _evaluate(model, ds, indices, device):
    model.eval()
    truth, predicted = [], []
    with torch.no_grad():
        for start in range(0, len(indices), 256):
            x, y = ds.take(indices[start:start + 256])
            logits = model(torch.from_numpy(x).to(device))
            truth.extend(y.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    conf = metricio.confusion(truth, predicted, 4)
    result = metricio.summarize(conf)
    result["samples"] = len(truth)
    return result


def run(cfg, max_steps=0):
    out = cfg["train"].get("out")
    if not out:
        raise ValueError("base_torch_v1 需要 train.out")
    os.makedirs(out, exist_ok=True)
    lock = open(os.path.join(out, ".run.lock"), "a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        lock.close()
        raise ValueError(f"{out} 已被另一个训练进程占用") from exc
    try:
        return _run_locked(cfg, max_steps=max_steps)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _run_locked(cfg, max_steps=0):
    train_cfg = cfg["train"]
    model_cfg = cfg["model"]
    data_path = train_cfg.get("data")
    out = train_cfg.get("out")
    if not data_path or not out:
        raise ValueError("base_torch_v1 需要 train.data 和 train.out")
    os.makedirs(out, exist_ok=True)
    existing = [name for name in os.listdir(out) if name != ".run.lock"]
    if existing:
        raise ValueError(f"{out} 不是空目录，拒绝混写：{existing[:3]}")
    ds = dataio.Uavr(data_path, drop_collision=bool(train_cfg.get("drop_collision", True)))
    device = torch.device(train_cfg.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求 CUDA，但当前环境没有可用 CUDA")
    seed = int(train_cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    h = int(model_cfg["input"].get("height", ds.h))
    w = int(model_cfg["input"].get("width", ds.w))
    if (h, w) != (ds.h, ds.w):
        raise ValueError(f"配置输入 {h}x{w} 与 UAVR {ds.h}x{ds.w} 不一致")
    model = base_model.BaseCnn(in_h=h, in_w=w).to(device)
    weights_np = ds.auto_class_weights().astype(np.float32)
    weights = torch.from_numpy(weights_np).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["lr"]),
                           betas=(float(train_cfg["beta1"]), float(train_cfg["beta2"])),
                           eps=float(train_cfg["eps"]), weight_decay=float(train_cfg["weight_decay"]))
    rng = np.random.RandomState(seed)
    indices = ds.splits["train"].copy()
    batch = int(train_cfg["batch"])
    epochs = int(train_cfg["epochs"])
    step = 0
    rows = []
    best_state = None
    best_val = -1.0
    best_epoch = 0
    started = time.monotonic()
    cfg_hash = cfgio.write_resolved(out, cfg)
    for epoch in range(epochs):
        rng.shuffle(indices)
        model.train()
        losses = []
        for start in range(0, len(indices), batch):
            x, y = ds.take(indices[start:start + batch])
            opt.zero_grad(set_to_none=True)
            logits = model(torch.from_numpy(x).to(device))
            loss = F.cross_entropy(logits, torch.from_numpy(y).to(device), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["max_grad_norm"]))
            opt.step()
            step += 1
            losses.append(float(loss.detach().cpu()))
            if max_steps and step >= max_steps:
                break
        row = {"epoch": epoch + 1, "step": step,
               "train_loss": float(np.mean(losses)) if losses else None,
               "val": _evaluate(model, ds, ds.splits["val"], device)}
        rows.append(row)
        val_macro_f1 = row["val"]["macro_f1"]
        if val_macro_f1 is not None and val_macro_f1 >= best_val:
            best_val = float(val_macro_f1)
            best_epoch = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if max_steps and step >= max_steps:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"profile": model_cfg["profile"], "config_hash": cfg_hash,
                "model": model.state_dict(), "step": step, "best_epoch": best_epoch},
               os.path.join(out, "model.pt"))
    with open(os.path.join(out, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = {"profile": model_cfg["profile"], "steps": step, "epochs": len(rows),
              "best_epoch": best_epoch,
              "best_val_macro_f1": best_val if best_state is not None else None,
              "elapsed_s": round(time.monotonic() - started, 3), "config_hash": cfg_hash,
              "test": _evaluate(model, ds, ds.splits["test"], device)}
    with open(os.path.join(out, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", required=True, help="base_torch_v1 YAML/JSON 配置")
    parser.add_argument("--data", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)
    cfg = cfgio.resolve(cfgio.load(args.config), cli=vars(args))
    if cfg["model"]["profile"] != "base_torch_v1":
        raise SystemExit("train_base.py 只接受 model.profile=base_torch_v1")
    if cfg["train"].get("data") is None or cfg["train"].get("out") is None:
        raise SystemExit("配置或 CLI 必须提供 train.data 和 train.out")
    result = run(cfg, max_steps=max(0, int(args.max_steps)))
    with open(os.path.join(cfg["train"]["out"], "DONE"), "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
