#!/usr/bin/env python3
"""把 costmap 教师的 Q 值（relabel_costmap.py / collect --teacher-planner costmap 写进 actions.jsonl 的 q_costs）
导成软标签 npz：`{episode_id}/{step}/logits` = log softmax(−Q/α)（α 单位：步）。

下游：`tools/join_soft_targets.py --soft <npz> --temperature 1.0` 还原 p 并按 UAVR 样本序拼成 UAVSOFT1，
再 `uavrgb_train --soft-targets … --distill-alpha 1.0`（训练器禁止软标签 + 类权重）。
没有 Q 的帧（fallback：no_pillars / goal_stop / bad_waypoint，或 all_collide 时 fallback 动作 ≠ argmin）
不能拿 Q 当分布：写成以硬标签为峰的分布（p_label = --fallback-peak），数量记进 npz 的 n_fallback 与 stdout。

用法：
  python3 tools/export_costmap_soft.py --src <relabeled collection> --alpha 0.5 --out <soft.npz>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "sim", "tools"))
from teacher import recomputable_versions  # noqa: E402

NUM_CLASSES = 4


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    m = z.max()
    return z - m - np.log(np.exp(z - m).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="collection 根目录（<src>/<split>/<id>/actions.jsonl 含 q_costs）")
    p.add_argument("--alpha", type=float, required=True, help="softmax 温度（步）：p ∝ exp(−Q/α)")
    p.add_argument("--out", required=True, help="输出 .npz")
    p.add_argument("--fallback-peak", type=float, default=0.97, help="无 Q 帧：硬标签的概率，其余均分")
    p.add_argument("--splits", default="train,val,test")
    args = p.parse_args()
    if not (np.isfinite(args.alpha) and args.alpha > 0):
        p.error("--alpha 必须为正")
    if not 0.5 < args.fallback_peak < 1.0:
        p.error("--fallback-peak 在 (0.5, 1)")

    out, n_q, n_fb, n_mismatch = {}, 0, 0, 0
    accepted = recomputable_versions(costmap=True)
    version = None       # 各集 costmap 教师版本必须一致：v1 点目标、v2 过柱阵后线目标，出口区标签口径不同，不混导
    for split in args.splits.split(","):
        for ep in sorted(glob.glob(os.path.join(args.src, split, "*"))):
            man_path = os.path.join(ep, "episode_manifest.json")
            act_path = os.path.join(ep, "actions.jsonl")
            if not (os.path.isfile(man_path) and os.path.isfile(act_path)):
                continue
            man = json.load(open(man_path, encoding="utf-8"))
            if man.get("planner") != "costmap" or man.get("teacher_version") not in accepted:
                sys.exit(f"错误: {ep} 不是 costmap 集（planner={man.get('planner')}, "
                         f"teacher_version={man.get('teacher_version')}，可接受 {list(accepted)}）")
            if version is None:
                version = man["teacher_version"]
            elif man["teacher_version"] != version:
                sys.exit(f"错误: {ep} 的 teacher_version={man['teacher_version']} 与此前的 {version} 不一致，"
                         f"不同版本的软标签口径不同，不能混导")
            eid = str(man.get("episode_id", os.path.basename(ep)))
            for line in open(act_path, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                label = int(r["action_expert"])
                q = r.get("q_costs")
                use_q = (isinstance(q, list) and len(q) == NUM_CLASSES and all(np.isfinite(q))
                         and r.get("costmap_fallback") is None)
                if use_q and int(np.argmin(q)) != label:
                    n_mismatch += 1          # 平局按可见柱定向时 argmin 与标签可不同；仍用 Q 的分布
                if use_q:
                    logits = log_softmax(-np.asarray(q) / args.alpha)
                    n_q += 1
                else:
                    pr = np.full(NUM_CLASSES, (1.0 - args.fallback_peak) / (NUM_CLASSES - 1))
                    pr[label] = args.fallback_peak
                    logits = np.log(pr)
                    n_fb += 1
                key = f"{eid}/{int(r['step'])}/logits"
                if key in out:
                    sys.exit(f"错误: 重复键 {key}")
                out[key] = logits.astype(np.float32)
    if not out:
        sys.exit("错误: 没有读到任何帧")
    meta = {"teacher_version": np.array(version), "alpha_steps": np.array(str(args.alpha)),
            "aggregate": np.array("costmap_softmax_negQ_over_alpha"),
            "n_frames": np.array(len(out)), "n_from_q": np.array(n_q), "n_fallback": np.array(n_fb),
            "n_argmin_ne_label": np.array(n_mismatch), "src": np.array(os.path.abspath(args.src))}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(args.out, **out, **meta)
    print(f"{args.out}: {len(out)} 帧，Q 分布 {n_q}，fallback 峰分布 {n_fb}，argmin≠标签 {n_mismatch}，α={args.alpha}")


if __name__ == "__main__":
    main()
