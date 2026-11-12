#!/usr/bin/env python3
"""卡 E2E-P0 第三关的诊断：同种子原型线与 C++ 线的 loss 差，是浮点噪声放大，还是配方里还有差异？

两个探针，都用 train.py 的 Trainer（与训练同一份代码），不改训练配方：

  resume    从 C++ 参照 run 的 ckpt_epNN.bin 续训：权重、Adam 的 m/v/t 都取自该文件，打乱引擎前进 NN 次，
            再训 K 步，逐步对照 C++ 参照 run 的 train_step loss。
            这一步检验：Adam 在 t 很大时的更新式、偏置修正、第 NN+1 轮的批次（打乱引擎持续状态）。
            起点逐位相同，K 步里 loss 差应保持在舍入量级，增长速度与下面的扰动探针相当。
  perturb   原型线自己与自己对比：同一种子，初始权重逐元素乘 (1 ± rel)（rel 默认 1e-7，约 1 ulp），
            训练 N 步，对照未扰动那条。给出「只差 1 ulp 的两条轨迹，|Δloss| 第一次超过 1e-5/1e-4/1e-3/1e-2 是第几步」。
            原型 vs C++ 同种子的对应步数落在这个量级，说明差异是舍入噪声被放大。

用法（teacher-venv，需要 torch；GPU 命令前加 CUDA_VISIBLE_DEVICES）：
  python proto/probe_divergence.py resume --ref-run <C++ run> --epochs 1 10 29 --steps 24
  python proto/probe_divergence.py perturb --seed 0 --rel 1e-7 --draws 3 --steps 130
都可加 --out-json <路径>。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ckpt_io  # noqa: E402
import data as dataio  # noqa: E402
import train as T  # noqa: E402

THRESHOLDS = (1e-5, 1e-4, 1e-3, 1e-2)                 # 与 compare_training.py 相同
SEED = 0                                              # 参照种子 0（e1-noise/seed0）
DEFAULT_STEPS_RESUME = 24                             # 续训后看几步：足够看到偏差从舍入量级起步的增长
DEFAULT_STEPS_PERTURB = 130                           # 两个 epoch
DEFAULT_REL = 1e-7                                    # 扰动的相对量：float32 在 1 附近的 ulp 是 1.19e-7（向下 5.96e-8）
DEFAULT_DRAWS = 3                                     # 扰动的随机抽样次数


def first_exceed(diffs):
    out = {}
    for t in THRESHOLDS:
        idx = np.flatnonzero(np.asarray(diffs) > t)
        out[f"{t:g}"] = int(idx[0]) + 1 if idx.size else None
    return out


def load_ds(path):
    ds = dataio.Uavr(path, drop_collision=T.DROP_COLLISION)
    images, labels = T.load_images(ds)
    return ds, images, labels


def make_trainer(ds, images, labels, seed, device, epochs):
    return T.Trainer(ds, seed, device, epochs=epochs, batch=T.BATCH, run_dir=None, images=images, labels=labels)


def run_steps(tr, steps):
    out = []
    while len(out) < steps:
        for idx in tr.epoch_batches():
            loss, _, _, _ = tr.train_step(idx)
            out.append(loss)
            if len(out) >= steps:
                break
    return out


def read_ref_losses(run):
    out = {}
    with open(os.path.join(run, "metrics.jsonl"), encoding="utf-8") as f:
        for line in f:
            if '"train_step"' in line:
                e = json.loads(line)
                out[e["step"]] = e["loss"]
    return out


@torch.no_grad()
def load_cpp_state(tr, ck):
    dev = tr.flat.device
    cat = lambda arrs: torch.cat([torch.from_numpy(np.asarray(a, np.float32)).reshape(-1) for a in arrs]).to(dev)   # noqa: E731
    tr.flat.copy_(cat(ck.params))
    tr.adam.m.copy_(cat(ck.adam_m))
    tr.adam.v.copy_(cat(ck.adam_v))
    tr.adam.t = int(ck.adam_t)
    tr.step = int(ck.adam_t)


def resume_probe(ref_run, epochs, steps, device, data):
    ds, images, labels = load_ds(data)
    ref = read_ref_losses(ref_run)
    results = {}
    for e in epochs:
        ck = ckpt_io.read_checkpoint(os.path.join(ref_run, f"ckpt_ep{e:02d}.bin"))
        tr = make_trainer(ds, images, labels, SEED, device, e + 1)
        load_cpp_state(tr, ck)
        for _ in range(e):                            # 打乱引擎前进 e 次（每个已训完的 epoch 打乱一次）
            tr.rng.shuffle(tr.order)
        base = tr.step
        mine = run_steps(tr, steps)
        diffs = [abs(mine[i] - ref[base + 1 + i]) for i in range(steps)]
        results[e] = {
            "ckpt_adam_t": int(ck.adam_t),
            "rows": [{"step": base + 1 + i, "proto": mine[i], "ref": ref[base + 1 + i], "diff": mine[i] - ref[base + 1 + i]} for i in range(steps)],
            "first_exceed_after_resume": first_exceed(diffs),
            "max_abs_diff": float(max(diffs)),
        }
        del tr
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return results


@torch.no_grad()
def perturb_weights(tr, rel, draw):
    g = torch.Generator().manual_seed(1000 + draw)
    r = (torch.randint(0, 2, tr.flat.shape, generator=g) * 2 - 1).float().to(tr.flat.device)
    tr.flat.mul_(1.0 + rel * r)


def perturb_probe(seed, rel, draws, steps, device, data):
    ds, images, labels = load_ds(data)
    tr = make_trainer(ds, images, labels, seed, device, (steps + 64) // 65 + 1)
    base = run_steps(tr, steps)
    del tr
    out = {"rel": rel, "steps": steps, "draws": []}
    for k in range(draws):
        tr = make_trainer(ds, images, labels, seed, device, (steps + 64) // 65 + 1)
        perturb_weights(tr, rel, k)
        got = run_steps(tr, steps)
        d = [abs(a - b) for a, b in zip(got, base)]
        out["draws"].append({"draw": k, "first_exceed": first_exceed(d), "max_abs_diff": float(max(d)),
                             "first_rows": [{"step": i + 1, "diff": got[i] - base[i]} for i in range(12)]})
        del tr
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return out


def fmt_first(fe):
    return "；".join(f"> {t} 在第 {v} 步" if v else f"> {t} 未出现" for t, v in fe.items())


def render(kind, res):
    L = []
    if kind == "resume":
        L += ["# 续训探针：从 C++ checkpoint 接着训，逐步对照 C++ loss", ""]
        for e, r in res.items():
            L += [f"## 从第 {e} 轮末（Adam t={r['ckpt_adam_t']}）续训", "",
                  f"- |Δloss| 第一次超过阈值（从续训后第 1 步数起）：{fmt_first(r['first_exceed_after_resume'])}；最大 {r['max_abs_diff']:.3e}。", "",
                  "| 全局步 | 原型 loss | C++ loss | 差 |", "|---|---|---|---|"]
            for row in r["rows"][:12]:
                L.append(f"| {row['step']} | {row['proto']:.9g} | {row['ref']:.9g} | {row['diff']:+.2e} |")
            L.append("")
    else:
        L += [f"# 扰动探针：同种子，初始权重乘 (1 ± {res['rel']:g})，训 {res['steps']} 步，对照未扰动", ""]
        for d in res["draws"]:
            L += [f"- 抽样 {d['draw']}：{fmt_first(d['first_exceed'])}；最大 |Δloss| {d['max_abs_diff']:.3e}。"]
        L.append("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)
    a = sub.add_parser("resume")
    a.add_argument("--ref-run", required=True)
    a.add_argument("--epochs", type=int, nargs="+", default=[1, 10, 29])
    a.add_argument("--steps", type=int, default=DEFAULT_STEPS_RESUME)
    b = sub.add_parser("perturb")
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--rel", type=float, default=DEFAULT_REL)
    b.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    b.add_argument("--steps", type=int, default=DEFAULT_STEPS_PERTURB)
    for s in (a, b):
        s.add_argument("--data", default=T.DATA)
        s.add_argument("--device", default=T.DEVICE)
        s.add_argument("--out-json")
        s.add_argument("--out-md")
    args = ap.parse_args(argv)
    if args.kind == "resume":
        res = resume_probe(args.ref_run, args.epochs, args.steps, args.device, args.data)
    else:
        res = perturb_probe(args.seed, args.rel, args.draws, args.steps, args.device, args.data)
    md = render(args.kind, res)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
    if args.out_md:
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
