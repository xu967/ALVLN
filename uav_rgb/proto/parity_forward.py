#!/usr/bin/env python3
"""卡 E2E-P0 第一关：PyTorch 前向与 uavrgb_infer 逐帧对拍。

两类对拍：
  real    真实 checkpoint + 真实数据集的帧：四类概率与 uavrgb_infer 的输出逐帧相差不超过 TOL（验收标准 1e-5）。
  variant 随机权重的各结构变体（双帧 / prev-action / 文本 / 辅助头）：用 proto/ckpt_io.py 写出 C++ checkpoint，
          C++ 读得进去（读端校验通过）且概率与 PyTorch 前向一致。变体没有现成的 C++ 训练产物可对，用随机权重代替；
          这一步同时验证写端格式。
用法：
  CUDA_VISIBLE_DEVICES=<UUID> $D/teacher-venv/bin/python proto/parity_forward.py real \
      --ckpt <C++ ckpt> --data <dataset.uavr> --infer-bin <uavrgb_infer> --frames 2000 --out <json>
  ... proto/parity_forward.py variants --data <dataset.uavr> --infer-bin <uavrgb_infer> --out <json>
"""
import argparse
import json
import os
import sys
import tempfile
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ckpt_io as cio  # noqa: E402
import data as dsmod  # noqa: E402
import proto_client as pc  # noqa: E402

TOL = 1e-5   # 卡规格：四类概率逐帧相差不超过 1e-5


def select_frames(ds, total=2000, seed=0):
    """取 total 帧：全部 val、全部 test，不足部分从 train 里按 seed 随机补。
    val 分区只有几百帧（trunc 数据集 574 帧），凑不满 2000，所以补 test 与 train。返回 (全局下标, 来源 split 标签)。"""
    idx = [ds.splits["val"], ds.splits["test"]]
    tags = [["val"] * len(idx[0]), ["test"] * len(idx[1])]
    have = sum(len(a) for a in idx)
    if have < total:
        rng = np.random.RandomState(seed)
        pick = np.sort(rng.choice(ds.splits["train"], size=total - have, replace=False))
        idx.append(pick)
        tags.append(["train"] * len(pick))
    elif have > total:
        idx = [np.concatenate(idx)[:total]]
        tags = [(tags[0] + tags[1])[:total]]
    return np.concatenate(idx), [t for tg in tags for t in tg]


def run_cpp(binary, ckpt, payloads, text_ids=None):
    """把 payload 逐个灌给 C++ 推理进程，返回 (probs[N,C] float64, actions[N], infer_ms[N], hello)。"""
    probs, acts, ms = [], [], []
    with pc.InferClient(binary, ckpt, text_ids=text_ids) as cli:
        hello = cli.hello
        for p in payloads:
            r = cli.step(p)
            probs.append(r["probs"])
            acts.append(r["action"])
            ms.append(r["infer_ms"])
        rc, err = cli.close()
    if rc != 0:
        raise RuntimeError(f"C++ 推理进程退出码 {rc}: {err}")
    return np.array(probs, dtype=np.float64), np.array(acts), np.array(ms), hello


def compare(cpp_probs, cpp_actions, torch_logits):
    """逐帧比较。torch_logits 是 float32 [N,C]。返回统计 dict。"""
    import model as pm
    tp = pm.softmax_probs(torch_logits)
    d = np.abs(tp - cpp_probs)
    per_frame = d.max(axis=1)
    ta = pm.first_argmax(torch_logits)
    srt = np.sort(tp, axis=1)
    gap = srt[:, -1] - srt[:, -2]
    mism = np.flatnonzero(ta != cpp_actions)
    return {"frames": int(len(tp)), "max_abs_prob_diff": float(per_frame.max()), "mean_abs_prob_diff": float(d.mean()),
            "p99_frame_max_diff": float(np.percentile(per_frame, 99)),
            "worst_frame": int(per_frame.argmax()), "frames_over_tol": int((per_frame > TOL).sum()),
            "argmax_mismatch": int(len(mism)), "argmax_mismatch_frames": mism[:20].tolist(),
            "frames_top2_gap_below_1e-5": int((gap < 1e-5).sum()),
            "min_top2_gap": float(gap.min()), "tol": TOL, "pass": bool(per_frame.max() <= TOL and len(mism) == 0)}


def torch_logits_for(model, x_u8, text=None, prev=None, batch=1, device="cuda"):
    """x_u8: numpy uint8 [N,in_c,H,W]；按 batch 分块前向，返回 float32 logits [N,C]（numpy）。"""
    import torch
    out = []
    with torch.inference_mode():
        for s in range(0, len(x_u8), batch):
            xb = torch.from_numpy(x_u8[s:s + batch]).to(device)
            tb = None if text is None else torch.from_numpy(text[s:s + batch]).to(device)
            pb = None if prev is None else torch.from_numpy(prev[s:s + batch]).to(device)
            out.append(model(xb, tb, pb).float().cpu().numpy())
    return np.concatenate(out)


def run_real(args):
    import torch
    import model as pm
    pm.configure_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = cio.read_checkpoint(args.ckpt)
    ds = dsmod.Uavr(args.data, drop_collision=True)
    contract = cio.parse_contract(ck.contract)
    sha_ok = (contract["dataset_sha256"] == ds.sha256) if contract["dataset_sha256"] else None
    idx, tags = select_frames(ds, args.frames, seed=0)
    cur, _ = ds.take(idx)
    aux = dsmod.read_aux(args.data) if (ck.prev_dim > 0 or ck.in_c == 6) else None
    if (ck.prev_dim > 0 or ck.in_c == 6) and aux is None:
        raise SystemExit("checkpoint 接 prev-action 或双帧，但数据集没有 dataset.aux 侧车")
    if ck.text_dim > 0:
        raise SystemExit("真实对拍还没接文本侧车（instructions.jsonl）；文本分支由 variants 子命令覆盖")
    prv = ds.take(aux["prev_index"][idx])[0] if ck.in_c == 6 else None   # 双帧：上一帧取侧车的 prev_index
    prev = aux["prev_action"][idx] if ck.prev_dim > 0 else None
    x = np.concatenate([cur, prv], axis=1) if prv is not None else cur
    model = pm.model_from_ckpt(ck, device)
    payloads = [pc.make_payload(cur[i], None if prv is None else prv[i], None if prev is None else prev[i])
                for i in range(len(cur))]
    cpp_p, cpp_a, cpp_ms, hello = run_cpp(args.infer_bin, args.ckpt, payloads)
    res = {"ckpt": args.ckpt, "data": args.data, "infer_bin": args.infer_bin, "device": device,
           "dataset_sha256_matches_contract": sha_ok, "hello": hello,
           "composition": {s: tags.count(s) for s in ("val", "test", "train")},
           "cpp_infer_ms_p50": float(np.percentile(cpp_ms, 50)), "cpp_infer_ms_p95": float(np.percentile(cpp_ms, 95))}
    for bs in (1, 100):
        lg = torch_logits_for(model, x, prev=prev, batch=bs, device=device)
        res[f"batch{bs}"] = compare(cpp_p, cpp_a, lg)
    res["pass"] = all(res[f"batch{b}"]["pass"] for b in (1, 100))
    return res


# ---- 结构变体 ----
VARIANTS = {
    "base": dict(),
    "two_frame": dict(in_c=6),
    "prev_action": dict(prev_dim=4),
    "text": dict(text_dim=8),
    "text_prev_two_frame": dict(in_c=6, text_dim=8, prev_dim=4),
    "aux_head": dict(aux_classes=8),
    "all": dict(in_c=6, text_dim=4, prev_dim=4, aux_classes=8),
}


def random_model(structure, seed):
    """Xavier 权重 + 非零随机偏置（让 FiLM 偏置、文本 embedding 等路径都对输出有贡献）。"""
    import torch
    import model as pm
    g = torch.Generator().manual_seed(seed)
    m = pm.ProtoCnn(**structure)
    with torch.no_grad():
        for w, fan_in, fan_out in m.xavier_plan():
            lim = (6.0 / (fan_in + fan_out)) ** 0.5
            w.copy_((torch.rand(w.shape, generator=g) * 2 - 1) * lim)
        for p in m.cpp_params():
            if p.ndim == 1:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
        if m.text_emb is not None:
            m.text_emb.copy_(torch.randn(m.text_emb.shape, generator=g))
    return m.eval()


def run_variants(args):
    import torch
    import model as pm
    pm.configure_determinism()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = dsmod.Uavr(args.data, drop_collision=True)
    rng = np.random.RandomState(1)
    n = args.frames
    base = np.sort(rng.choice(ds.splits["val"], size=n + 1, replace=False))   # 下标 i 的"上一帧"取 i-1 号样本
    out = {}
    tmp = tempfile.mkdtemp(prefix="proto_variant_")
    for name, st in VARIANTS.items():
        structure = dict(in_h=ds.h, in_w=ds.w, in_c=3, num_classes=4, text_dim=0, aux_classes=0, prev_dim=0)
        structure.update(st)
        model = random_model(structure, seed=zlib.crc32(name.encode()) % 1000)
        path = os.path.join(tmp, f"{name}.bin")
        from model import ckpt_from_model
        cio.write_checkpoint(path, ckpt_from_model(model, epoch=1))
        model = model.to(device)
        cur, _ = ds.take(base[1:])
        prv, _ = ds.take(base[:-1])
        x = np.concatenate([cur, prv], axis=1) if structure["in_c"] == 6 else cur
        # C++ 的 --text 只给一条指令、逐帧复用；取 6 个 token，其余 3 位由 C++ 补 pad（0）
        text_ids, text = None, None
        if structure["text_dim"]:
            text_ids = [int(t) for t in rng.randint(1, cio.TEXT_VOCAB, size=6)]
            text = np.tile(np.array(text_ids + [0] * (cio.TEXT_LEN - 6), dtype=np.uint8), (n, 1))
        prev = rng.randint(0, 4, size=n).astype(np.int64) if structure["prev_dim"] else None
        payloads = [pc.make_payload(cur[i], prv[i] if structure["in_c"] == 6 else None,
                                    None if prev is None else prev[i]) for i in range(n)]
        cpp_p, cpp_a, cpp_ms, hello = run_cpp(args.infer_bin, path, payloads, text_ids)
        lg = torch_logits_for(model, x, text=text, prev=prev, batch=1, device=device)
        r = compare(cpp_p, cpp_a, lg)
        r["structure"] = structure
        r["hello"] = hello
        r["text_ids"] = text_ids
        out[name] = r
    out["pass"] = all(v["pass"] for k, v in out.items() if k != "pass")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("real")
    r.add_argument("--ckpt", required=True)
    v = sub.add_parser("variants")
    for p in (r, v):
        p.add_argument("--data", required=True)
        p.add_argument("--infer-bin", required=True)
        p.add_argument("--frames", type=int, default=2000 if p is r else 200)
        p.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run_real(args) if args.cmd == "real" else run_variants(args)
    text = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
