#!/usr/bin/env python3
"""加载教师 ckpt，为给定 split 的每个 episode 导出软标签到 .npz。

输出布局（episode_id 为字符串）：
  episode_ids              (N,)          所有 episode_id
  "<episode_id>/logits"    (4,)          四类 logits
  "<episode_id>/feat"      (512,)        fc 之前的特征

用法：
  python3 teacher/export_soft_targets.py --data /path/dataset/test \
      --ckpt /path/run/ckpt_best.pth --out /path/soft_targets.npz --device cuda

--data 指向 split 目录本身（train/ 或 val/ 或 test/），不是它的上级。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import EpisodeSequenceDataset
from model import PrivilegedTeacher

NUM_ACTIONS = 4
FEAT_DIM = 512


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="split 目录（含若干 episode 子目录）")
    p.add_argument("--ckpt", required=True, help="train.py 产出的 ckpt_best.pth")
    p.add_argument("--out", required=True, help="输出 .npz 路径")
    p.add_argument("--seq-len", dest="seq_len", type=int, default=8)
    p.add_argument("--batch", dest="batch_size", type=int, default=8)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    dataset = EpisodeSequenceDataset(args.data, seq_len=args.seq_len)
    if len(dataset) == 0:
        sys.exit(f"{args.data} 没有可用样本（缺 depth/pose 的集会被跳过）。")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
    model = PrivilegedTeacher(pretrained=False)
    model.load_state_dict(state)

    device = torch.device(args.device)
    model.to(device).eval()

    payload = {}
    episode_ids = []
    seen = set()
    with torch.no_grad():
        for batch in loader:
            logits, feat = model(
                batch["rgb"].to(device),
                batch["depth"].to(device),
                batch["pose"].to(device),
                batch["text"].to(device),
                return_feat=True,
            )
            logits = logits.cpu().numpy().astype(np.float32)
            feat = feat.cpu().numpy().astype(np.float32)
            for eid, lg, ft in zip(batch["episode_id"], logits, feat):
                eid = str(eid)
                if eid in seen:
                    sys.exit(f"episode_id 重复: {eid!r}；请检查 manifest/episode 目录。")
                seen.add(eid)
                episode_ids.append(eid)
                payload[f"{eid}/logits"] = lg
                payload[f"{eid}/feat"] = ft

    payload["episode_ids"] = np.asarray(episode_ids)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"导出 {len(episode_ids)} 个 episode -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
