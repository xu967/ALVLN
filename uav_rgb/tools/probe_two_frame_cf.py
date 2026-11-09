#!/usr/bin/env python3
"""双帧反事实探针：双帧学生（--two-frame）在原侧车 vs 三个反事实侧车下的预测差异，只回答「双帧是不是捷径」
（DISTILL §2.3 对 prev-action 的同一套方法：固定 / 翻转 / 打乱线索，看离线增益与方向预测怎么变）。

运行（仓库根）：
  .venv-tools/bin/python tools/probe_two_frame_cf.py --ds <dir：dataset.uavr + dataset.aux> --ckpt <双帧 ckpt_best.bin> \
      --out <dir> [--infer-bin <uavrgb_infer>] [--train-bin <uavrgb_train>] [--baseline-ckpt <单帧 ckpt>] [--splits val,test]
三个反事实（prev_action 列原样，只改 prev_index）：
  self     上一帧 = 当前帧（BRIEF §6 要求的那个；首帧口径，运动线索为零）
  next     上一帧 = 同集 step+1 的帧（时间反演：左转的光流变成右转；prev-action 探针「L↔R 翻转」的类比）
  shuffle  上一帧 = 同集内随机帧（seed 0；破坏运动线索但保留画面统计）
每个反事实在 <out>/cf-<mode>/ 建一份数据集（硬链接 dataset.uavr，preprocess.write_aux_sidecar 写新侧车），
对 orig + 三个反事实逐帧推理（真实 uavrgb_infer，两张 HWC 帧一步），输出 <out>/probe.json 与 probe.md：
四件套、预测变化率（vs orig）、YAW_L 帧预测 L:R、YAW_R 帧预测 R:L、预测分布；--baseline-ckpt 给出时单帧模型同帧对照。
--train-bin 给出时再对 orig 与每个反事实目录跑 uavrgb_train --eval-only，核对 C++ 读侧车那条路的 macro_f1 与逐帧推理一致。
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import preprocess as pp  # noqa: E402
import select_ckpt as sc  # noqa: E402
import student  # noqa: E402

MODES = ("self", "next", "shuffle")
NAMES = ("SAFE", "FWD", "YAW_L", "YAW_R")


def cf_prev_index(mode, episodes, step=None, seed=0):
    """反事实的 prev_index（全局样本下标，int64[N]）。episodes = meta["episodes"]（sample_start/sample_count）；
    step = 每个样本的 step（None 时按集内位置当 step，即假设没有被丢掉的步）。"""
    n = sum(ep["sample_count"] for ep in episodes)
    out = np.arange(n, dtype=np.int64)
    if mode == "self":
        return out
    rng = np.random.default_rng(seed)
    for ep in episodes:
        g0, cnt = ep["sample_start"], ep["sample_count"]
        idx = np.arange(g0, g0 + cnt)
        if mode == "next":
            st = step[idx] if step is not None else np.arange(cnt)
            by_step = {int(s): int(g) for s, g in zip(st, idx)}
            for s, g in zip(st, idx):
                out[g] = by_step.get(int(s) + 1, int(g))   # 末帧（或下一步被丢掉）：指向自身，与首帧口径对称
        elif mode == "shuffle":
            out[idx] = idx[rng.permutation(cnt)]
        else:
            raise ValueError(f"mode 须为 {MODES}，得到 {mode!r}")
    return out


def make_cf_dataset(src_dir, dst_dir, aux, prev_index):
    """<dst>/dataset.uavr 硬链接到源，侧车按新 prev_index 重写（prev_action / depth / grid / 包络原样）。"""
    os.makedirs(dst_dir, exist_ok=True)
    dst_uavr = os.path.join(dst_dir, "dataset.uavr")
    if not os.path.exists(dst_uavr):
        os.link(os.path.join(src_dir, "dataset.uavr"), dst_uavr)
    return pp.write_aux_sidecar(dst_dir, dst_uavr, prev_index.astype(np.uint32), aux["prev_action"], aux["depth"],
                                aux["grid"], aux["envelope_radius_m"])


def direction_ratio(y, pred, label):
    """标签为 label（YAW_L=2 / YAW_R=3）的帧里，预测成同向 : 反向 的计数。"""
    sub = pred[y == label]
    same, other = int((sub == label).sum()), int((sub == (5 - label)).sum())
    return [same, other]


def condition_summary(y, pred, ref_pred=None):
    m = sc.metrics_from_confusion(sc.confusion(y, pred))
    yaw_lab = np.isin(y, sc.YAW)
    d = {"n": int(len(y)), **m, "pred_share": [float(v) for v in np.bincount(pred, minlength=4) / max(len(y), 1)],
         "yawl_frames_pred_L_R": direction_ratio(y, pred, 2), "yawr_frames_pred_R_L": direction_ratio(y, pred, 3)}
    if ref_pred is not None:
        changed = pred != ref_pred
        d["changed_rate"] = float(changed.mean())
        d["changed_rate_yaw_label"] = float(changed[yaw_lab].mean()) if yaw_lab.any() else float("nan")
        d["changed_rate_fwd_label"] = float(changed[y == 1].mean()) if (y == 1).any() else float("nan")
    return d


def eval_only_f1(train_bin, ds_dir, ckpt, out_dir):
    """uavrgb_train --eval-only 的 val/test macro_f1（stdout 上的 validation JSON 行）。"""
    cmd = [train_bin, "--data", os.path.join(ds_dir, "dataset.uavr"), "--eval-only", ckpt, "--out", out_dir,
           "--split", "all"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"eval-only 失败 rc={res.returncode}: {res.stderr.strip()[-500:]}")
    out = {}
    for line in res.stdout.splitlines():
        if line.startswith("{") and '"event":"validation"' in line:
            ev = json.loads(line)
            out[ev["split"]] = ev["macro_f1"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds", required=True, help="含 dataset.uavr 与 dataset.aux 的目录（训练双帧模型用的那份）")
    ap.add_argument("--ckpt", required=True, help="双帧模型 ckpt（in_c=6）；eval-only 校验会读同目录 config.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--infer-bin", default=student.DEFAULT_INFER)
    ap.add_argument("--train-bin", default=None, help="给出则用 uavrgb_train --eval-only 核对每个条件的 macro_f1")
    ap.add_argument("--baseline-ckpt", default=None, help="单帧模型 ckpt：同帧对照（四件套 + 与双帧预测的一致率）")
    ap.add_argument("--splits", default="val,test")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ds = sc.Uavr(os.path.join(args.ds, "dataset.uavr"))
    aux = sc.read_aux(os.path.join(args.ds, "dataset.aux"))
    if aux is None:
        sys.exit(f"{args.ds} 没有 dataset.aux（用新版 preprocess.py 重打包）")
    run_cfg = os.path.join(os.path.dirname(args.ckpt), "config.json")
    with open(run_cfg, encoding="utf-8") as f:
        drop = bool(json.load(f).get("drop_collision", False))
    poses = sc.sample_poses(ds.meta, bool((ds.meta.get("collision_filter") or {}).get("drop_collision_steps")))
    step = poses[1] if poses is not None else None
    if step is None:
        print("警告: source_dir 不可读，next 按集内位置当 step", file=sys.stderr)

    cf_index = {"orig": aux["prev_index"]}
    cf_dirs = {"orig": args.ds}
    for mode in MODES:
        cf_index[mode] = cf_prev_index(mode, ds.meta["episodes"], step)
        cf_dirs[mode] = os.path.join(args.out, f"cf-{mode}")
        make_cf_dataset(args.ds, cf_dirs[mode], aux, cf_index[mode])
        print(f"cf-{mode}: prev_index 与原侧车不同的样本 {int((cf_index[mode] != aux['prev_index']).sum())}/{ds.n}")

    result = {"ds": args.ds, "ckpt": args.ckpt, "drop_collision": drop, "splits": {}}
    inf = sc.Infer(args.ckpt, args.infer_bin)
    if inf.pol.in_c != 6:
        inf.close()
        sys.exit(f"{args.ckpt} 不是双帧模型（in_c={inf.pol.in_c}）")
    base = sc.Infer(args.baseline_ckpt, args.infer_bin) if args.baseline_ckpt else None
    for split in args.splits.split(","):
        idx = ds.split_indices(split, drop)
        y = ds.labels[idx].astype(np.int64)
        preds = {}
        for cond in ("orig",) + MODES:
            preds[cond], _ = sc.predict_indices(inf, ds, idx, cf_index[cond], aux["prev_action"])
        summ = {cond: condition_summary(y, preds[cond], None if cond == "orig" else preds["orig"])
                for cond in preds}
        if base is not None:
            bp, _ = sc.predict_indices(base, ds, idx)
            summ["single_frame_baseline"] = condition_summary(y, bp)
            summ["single_frame_baseline"]["agree_with_two_frame_orig"] = float((bp == preds["orig"]).mean())
            summ["single_frame_baseline"]["agree_with_two_frame_self"] = float((bp == preds["self"]).mean())
        result["splits"][split] = summ
        np.save(os.path.join(args.out, f"preds_{split}.npy"),
                np.c_[idx, y, *[preds[c] for c in ("orig",) + MODES]])
        for cond, d in summ.items():
            print(f"[{split}] {cond:22s} n={d['n']} acc={d['acc']:.4f} mF1={d['macro_f1']:.4f} bal={d['balanced_acc']:.4f} "
                  f"recall={[round(v, 3) for v in d['recall']]} changed={d.get('changed_rate', float('nan')):.3f} "
                  f"L:R={d['yawl_frames_pred_L_R']} R:L={d['yawr_frames_pred_R_L']}", flush=True)
    inf.close()
    if base is not None:
        base.close()

    if args.train_bin:
        result["eval_only_check"] = {}
        for cond in ("orig",) + MODES:
            got = eval_only_f1(args.train_bin, cf_dirs[cond], args.ckpt, os.path.join(args.out, f"eval-only-{cond}"))
            diffs = {s: abs(got[s] - result["splits"][s][cond]["macro_f1"]) for s in got if s in result["splits"]}
            result["eval_only_check"][cond] = {"eval_only_macro_f1": got, "abs_diff_vs_per_frame": diffs}
            print(f"eval-only {cond}: {got}  |Δ| vs 逐帧 {diffs}")
            if any(v > 1e-5 for v in diffs.values()):
                print(f"自检失败: {cond} 的 eval-only 与逐帧推理 macro_f1 不一致", file=sys.stderr)
                result["eval_only_check"]["failed"] = True

    with open(os.path.join(args.out, "probe.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, ensure_ascii=False)
    lines = ["| split | 条件 | n | acc | macro_f1 | bal_acc | recall S/F/L/R | 预测变化率(全/YAW 帧) | YAW_L 帧 L:R | YAW_R 帧 R:L |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for split, summ in result["splits"].items():
        for cond, d in summ.items():
            ch = ("—" if "changed_rate" not in d else
                  f"{d['changed_rate']:.3f} / {d['changed_rate_yaw_label']:.3f}")
            lines.append(f"| {split} | {cond} | {d['n']} | {d['acc']:.3f} | {d['macro_f1']:.3f} | {d['balanced_acc']:.3f} | "
                         f"{'/'.join(f'{v:.2f}' for v in d['recall'])} | {ch} | "
                         f"{d['yawl_frames_pred_L_R'][0]}:{d['yawl_frames_pred_L_R'][1]} | "
                         f"{d['yawr_frames_pred_R_L'][0]}:{d['yawr_frames_pred_R_L'][1]} |")
    with open(os.path.join(args.out, "probe.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 1 if result.get("eval_only_check", {}).get("failed") else 0


if __name__ == "__main__":
    sys.exit(main())
