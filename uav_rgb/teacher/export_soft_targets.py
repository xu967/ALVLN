#!/usr/bin/env python3
"""加载教师 ckpt，为给定 split 的每个 episode 导出软标签到 .npz。

一个 episode 会产出多个滑窗样本（窗口可重叠），所以 key 是 (episode_id, step)：
  episode_ids                    (N,)     每条样本来自哪个 episode
  steps                          (N,)     该窗口末帧的 step（= 标签所在的帧）
  seq_len                        ()       窗口长度
  "<episode_id>/<step>/logits"   (4,)     四类 logits
  "<episode_id>/<step>/feat"     (512,)   fc 之前的特征

用法：
  python3 teacher/export_soft_targets.py --data /path/dataset/test \
      --ckpt /path/run/ckpt_best.pth --out /path/soft_targets.npz --device cuda

--data 指向 split 目录本身（train/ 或 val/ 或 test/），不是它的上级。
"""
import argparse
import hashlib
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
    p.add_argument("--seq-len", dest="seq_len", type=int, default=None,
                   help="滑窗长度；默认读 ckpt 里训练时的 seq_len")
    p.add_argument("--batch", dest="batch_size", type=int, default=8)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model") or ckpt.get("model_state_dict") or ckpt
    seq_len = args.seq_len or int(ckpt.get("seq_len") or 8)
    # 来源审计：教师 ckpt 的哈希 + 教师选模 split + 教师训练数据指纹。
    with open(args.ckpt, "rb") as f:
        teacher_ckpt_sha256 = hashlib.file_digest(f, "sha256").hexdigest()
    select_split = str(ckpt.get("select_split") or "unknown")
    train_data_sha256 = str(ckpt.get("train_data_sha256") or "")
    dataset = EpisodeSequenceDataset(args.data, seq_len=seq_len)
    if len(dataset) == 0:
        sys.exit(f"{args.data} 没有可用样本（缺 depth/pose 的集会被跳过）。")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = PrivilegedTeacher(pretrained=False)
    model.load_state_dict(state)

    device = torch.device(args.device)
    model.to(device).eval()

    payload = {}
    episode_ids = []
    steps = []
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
            for eid, st, lg, ft in zip(batch["episode_id"], batch["step"].tolist(),
                                       logits, feat):
                # 一个 episode 有多个滑窗（每个窗口一个标签），key 必须带 step，
                # 否则同集第二个窗口就会覆盖前一个。
                key = (str(eid), int(st))
                if key in seen:
                    sys.exit(f"(episode_id, step) 重复: {key}；请检查 dataset 滑窗步长。")
                seen.add(key)
                episode_ids.append(key[0])
                steps.append(key[1])
                payload[f"{key[0]}/{key[1]}/logits"] = lg
                payload[f"{key[0]}/{key[1]}/feat"] = ft

    if len(episode_ids) != len(dataset):
        sys.exit(f"导出 {len(episode_ids)} 条 != 数据集 {len(dataset)} 条")
    payload["episode_ids"] = np.asarray(episode_ids)
    payload["steps"] = np.asarray(steps, dtype=np.int64)
    payload["seq_len"] = np.asarray(seq_len)
    # 标量字符串元信息；join_soft_targets.py 会原样写进 <out>.json。
    payload["teacher_ckpt_sha256"] = np.asarray(teacher_ckpt_sha256)
    payload["select_split"] = np.asarray(select_split)
    payload["train_data_sha256"] = np.asarray(train_data_sha256)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    print(f"导出 {len(episode_ids)} 条滑窗样本（{len({*episode_ids})} 个 episode）"
          f" -> {out_path}", file=sys.stderr)
    print(f"teacher ckpt sha256={teacher_ckpt_sha256} select_split={select_split}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
