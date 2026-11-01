#!/usr/bin/env python3
"""离线训练 PrivilegedTeacher，按 val 准确率保存 ckpt_best.pth。

用法：
  python3 teacher/train.py --data /path/dataset --epochs 30 --batch 8 \
      --lr 1e-3 --seq-len 8 --out /path/run --device cuda

数据目录假定为 <data>/train、<data>/val（每集一个目录，见 dataset.py）。
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EpisodeSequenceDataset, IMAGE_SIZE, POSE_DIM
from model import PrivilegedTeacher, TEXT_LEN, TEXT_VOCAB


def _make_dataset(data_root, split, seq_len):
    path = Path(data_root) / split
    if not path.is_dir():
        return None
    return EpisodeSequenceDataset(path, seq_len=seq_len)


def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["rgb"].to(device),
                batch["depth"].to(device),
                batch["pose"].to(device),
                batch["text"].to(device),
            )
            pred = logits.argmax(dim=1)
            correct += int((pred == batch["label"].to(device)).sum())
            total += len(batch["label"])
    return correct / total if total else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="数据集根目录（含 train/ val/）")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", dest="batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=8,
                   help="每个 episode 取最后 T 帧")
    p.add_argument("--out", required=True, help="输出目录（写 ckpt_best.pth）")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrained", action="store_true",
                   help="ResNet18 加载 ImageNet 权重（需联网下载，默认关闭）")
    args = p.parse_args()

    train_ds = _make_dataset(args.data, "train", args.seq_len)
    if train_ds is None or len(train_ds) == 0:
        sys.exit(
            f"train 集没有可用样本（{Path(args.data) / 'train'}）。\n"
            "常见原因：当前 collect.py 不保存深度图，dataset.py 会跳过缺 "
            "depth_*.png 的集；请先补齐深度文件（见 teacher/README.md）。")
    val_ds = _make_dataset(args.data, "val", args.seq_len)
    if val_ds is not None and len(val_ds) == 0:
        print("警告: val 集为空，改用 train 准确率选择 ckpt", file=sys.stderr)
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
                  if val_ds is not None else None)

    device = torch.device(args.device)
    model = PrivilegedTeacher(pretrained=args.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "ckpt_best.pth"
    best_metric = -1.0

    print(f"train={len(train_ds)} val={len(val_ds) if val_ds else 0} "
          f"device={device} seq_len={args.seq_len}", file=sys.stderr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            depth = batch["depth"].to(device)
            pose = batch["pose"].to(device)
            text = batch["text"].to(device)
            label = batch["label"].to(device)

            optimizer.zero_grad()
            loss = criterion(model(rgb, depth, pose, text), label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * len(label)
            seen += len(label)

        train_acc = _accuracy(model, train_loader, device)
        if val_loader is not None:
            metric = _accuracy(model, val_loader, device)
            metric_name = "val_acc"
        else:
            metric = train_acc
            metric_name = "train_acc"
        print(f"epoch {epoch}/{args.epochs} loss={total_loss / max(seen, 1):.4f} "
              f"train_acc={train_acc:.4f} {metric_name}={metric:.4f}",
              file=sys.stderr)

        if metric > best_metric:
            best_metric = metric
            torch.save({
                "model": model.state_dict(),
                "seq_len": args.seq_len,
                "text_len": TEXT_LEN,
                "text_vocab": TEXT_VOCAB,
                "image_size": IMAGE_SIZE,
                "pose_dim": POSE_DIM,
                "best_metric": best_metric,
            }, best_path)
            print(f"保存 {best_path} ({metric_name}={metric:.4f})", file=sys.stderr)

    print(f"完成：best {best_path} ({best_metric:.4f})", file=sys.stderr)


if __name__ == "__main__":
    main()
