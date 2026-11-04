#!/usr/bin/env python3
"""离线训练 PrivilegedTeacher，按 val 准确率保存 ckpt_best.pth。

用法：
  python3 teacher/train.py --data /path/dataset --epochs 30 --batch 8 \
      --lr 1e-3 --seq-len 8 --out teacher-run-x --device cuda [--no-record-video]

数据目录假定为 <data>/train、<data>/val（每集一个目录，见 dataset.py）。
--out 相对路径落在 $DATA_ROOT/uav_rgb/runs/ 下；绝对路径必须在 $DATA_ROOT 内
（根分区只有 30G，训练产物一律放数据盘）。
--record-video：默认开。训练完用 ckpt_best.pth 跑闭环评测，每集存三视图视频到
<out>/closed_loop_<UTCstamp>/videos/seed_<seed>.mp4（需 Gazebo 和 cv2）；
依赖缺失或录像失败只告警，不影响已保存的 ckpt。
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import EpisodeSequenceDataset, IMAGE_SIZE, POSE_DIM
from model import PrivilegedTeacher, TEXT_LEN, TEXT_VOCAB


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/root/rivermind-data")).resolve()
RUNS_ROOT = DATA_ROOT / "uav_rgb" / "runs"
EVAL = Path(__file__).resolve().parent.parent / "sim" / "tools" / "eval_closed_loop.py"


def resolve_out(out):
    """相对路径 -> RUNS_ROOT/out；绝对路径必须落在 DATA_ROOT 内，否则 ValueError。"""
    path = Path(out).expanduser()
    path = (path if path.is_absolute() else RUNS_ROOT / path).resolve()
    if path != DATA_ROOT and DATA_ROOT not in path.parents:
        raise ValueError(f"--out {path} 不在 {DATA_ROOT} 下；训练结果必须放数据盘")
    return path


def video_out_dir(out_dir):
    """每次录像用独立 UTC 目录：二次训练不会撞上 eval_closed_loop 的非空目录检查。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(out_dir) / f"closed_loop_{stamp}"


def video_skip_reason():
    """录像依赖：Gazebo CLI 与当前解释器的 cv2；缺失返回原因，否则 None。"""
    if shutil.which("gz") is None:
        return "找不到 gz（Gazebo CLI 不在 PATH）"
    try:
        import cv2  # noqa: F401
    except ImportError as exc:
        return f"当前解释器缺少 cv2（{exc}）"
    return None


def closed_loop_video(ckpt, out_dir, args):
    """用 ckpt 跑闭环评测并录三视图视频；返回 eval 退出码（失败不影响已存的 ckpt）。"""
    video_out = video_out_dir(out_dir)
    cmd = [sys.executable, str(EVAL), "--policy", "teacher-nn", "--teacher-ckpt", str(ckpt),
           "--task", args.video_task, "--split", args.video_split,
           "--episodes", str(args.video_episodes), "--record-video",
           "--out", str(video_out)]
    print("闭环录像: " + " ".join(cmd), file=sys.stderr)
    rc = subprocess.run(cmd).returncode
    if rc == 0:
        mp4s = sorted((video_out / "videos").glob("*.mp4"))
        print(f"闭环录像完成：{len(mp4s)} 个 mp4 -> {video_out / 'videos'}", file=sys.stderr)
        for path in mp4s:
            print(f"  {path}", file=sys.stderr)
    return rc


def record_video_best_effort(ckpt, out_dir, args):
    """训练后自动录像：依赖缺失或评测失败只告警，绝不改变训练退出码。"""
    reason = video_skip_reason()
    if reason:
        print(f"警告: 跳过闭环录像：{reason}", file=sys.stderr)
        return None
    rc = closed_loop_video(ckpt, out_dir, args)
    if rc != 0:
        print(f"警告: 闭环录像失败 rc={rc}；ckpt 已保存在 {ckpt}", file=sys.stderr)
    return rc


def _make_dataset(data_root, split, seq_len):
    path = Path(data_root) / split
    if not path.is_dir():
        return None
    return EpisodeSequenceDataset(path, seq_len=seq_len)


def _tree_sha256(root):
    """训练数据的身份指纹：只哈希 manifest/actions（标签与样本顺序的来源）。

    不含 RGB/深度像素：几十 GB 全量哈希会让每次训练多耗数分钟；这两类文件一变，
    样本集合/标签/窗口顺序就变了，足以识别数据集换代。
    """
    h = hashlib.sha256()
    root = Path(root)
    files = sorted(list(root.rglob("episode_manifest.json")) + list(root.rglob("actions.jsonl")))
    for f in files:
        h.update(str(f.relative_to(root)).encode())
        h.update(b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()


def _predict(model, loader, device):
    model.eval()
    gts, preds = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["rgb"].to(device),
                batch["depth"].to(device),
                batch["pose"].to(device),
                batch["text"].to(device),
            )
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            gts.extend(batch["label"].cpu().tolist())
    return gts, preds


def _accuracy(gts, preds):
    return sum(g == p for g, p in zip(gts, preds)) / len(gts) if gts else 0.0


def _macro_f1(gts, preds, n=4):
    """宏 F1（只对 GT 中出现过的类求平均）；YAW 召回低时它不会像 acc 那样被掩盖。"""
    f1s = []
    for c in range(n):
        tp = sum(g == c and p == c for g, p in zip(gts, preds))
        fp = sum(g != c and p == c for g, p in zip(gts, preds))
        fn = sum(g == c and p != c for g, p in zip(gts, preds))
        if tp + fn == 0:
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="数据集根目录（含 train/ val/）")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", dest="batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=8,
                   help="每个 episode 取最后 T 帧")
    p.add_argument("--out", required=True,
                   help="输出目录（写 ckpt_best.pth）；相对路径放 $DATA_ROOT/uav_rgb/runs/，"
                        "绝对路径必须在 $DATA_ROOT 内")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--pretrained", action="store_true",
                   help="ResNet18 加载 ImageNet 权重（需联网下载，默认关闭）")
    p.add_argument("--class-weights", choices=("auto", "none"), default="auto",
                   help="auto=训练集逆频率加权（YAW 只占 8.8%%，不加权会被 FORWARD 淹没）")
    p.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=True,
                   help="训练结束后用 best ckpt 闭环评测并录 机载/追尾/顶视 三视图视频"
                        "（默认开；--no-record-video 关闭）")
    p.add_argument("--video-episodes", type=int, default=1, help="录像集数（默认 1）")
    p.add_argument("--video-task", choices=("base", "l1", "campus"), default="l1",
                   help="录像评测的任务（与训练数据一致；teacher 数据是 l1）")
    p.add_argument("--video-split", choices=("train", "val", "test"), default="test")
    return p


def main():
    p = build_parser()
    args = p.parse_args()
    try:
        out_dir = resolve_out(args.out)
    except ValueError as e:
        p.error(str(e))
    if args.video_episodes < 1:
        p.error("--video-episodes 必须 >= 1")

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
    if args.class_weights == "auto":
        counts = [0] * 4
        for s in train_ds.samples:
            counts[s["label"]] += 1
        total = sum(counts)
        weights = [total / (4 * c) if c else 0.0 for c in counts]
        print(f"class counts={counts} weights={[round(w, 3) for w in weights]}",
              file=sys.stderr)
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "ckpt_best.pth"
    best_metric = -1.0
    # 选模 split 与训练数据指纹写进 ckpt：export_soft_targets.py 会带进软标签元信息，
    # 供 student 判断「教师是否用 student val 选过模」。
    select_split = "val" if val_ds is not None else "train"
    train_data_sha256 = _tree_sha256(Path(args.data) / "train")
    print(f"select_split={select_split} train_data_sha256={train_data_sha256}",
          file=sys.stderr)

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

        gts, preds = _predict(model, train_loader, device)
        train_acc = _accuracy(gts, preds)
        if val_loader is not None:
            gts_v, preds_v = _predict(model, val_loader, device)
            metric = _macro_f1(gts_v, preds_v)
            metric_name = "val_macro_f1"
            extra = f"val_acc={_accuracy(gts_v, preds_v):.4f}"
        else:
            metric = _macro_f1(gts, preds)
            metric_name = "train_macro_f1"
            extra = f"train_acc={train_acc:.4f}"
        print(f"epoch {epoch}/{args.epochs} loss={total_loss / max(seen, 1):.4f} "
              f"train_acc={train_acc:.4f} {extra} {metric_name}={metric:.4f}",
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
                # 来源审计：导出软标签时原样传下去。
                "select_split": select_split,
                "train_data_sha256": train_data_sha256,
            }, best_path)
            print(f"保存 {best_path} ({metric_name}={metric:.4f})", file=sys.stderr)

    print(f"完成：best {best_path} ({best_metric:.4f})", file=sys.stderr)
    if args.record_video:
        record_video_best_effort(best_path, out_dir, args)


if __name__ == "__main__":
    main()
