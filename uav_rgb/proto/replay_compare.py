#!/usr/bin/env python3
"""卡 E2E-P0 第二关（离线部分）：同一份 C++ checkpoint、同一串帧，分别灌给 uavrgb_infer 与 PyTorch 包装器，
逐步比较动作是否一致，并记两边的推理耗时（进程自报的 infer_ms）与一问一答往返耗时。

两种模式：
  dataset  数据集里的 160×120 帧，按 split 的 episode 顺序、时间顺序，经行协议客户端（proto/proto_client.py）灌帧。
  pngs     采集目录里的 320×240 PNG 帧，经 sim/tools/student.py 的 StudentPolicy（PIL 缩放、预热、超时、
           prev-action 与上一帧缓存都走真实路径）灌帧，两边各一个 StudentPolicy，每集开始 reset()。
用法：
  python proto/replay_compare.py dataset --ckpt <ckpt> --data <dataset.uavr> --cpp-bin <uavrgb_infer> \
      --wrap-bin proto/uavrgb_infer_proto --splits val,test --out <json>
  python proto/replay_compare.py pngs --ckpt <ckpt> --episodes-root <col/test> --cpp-bin ... --wrap-bin ... --out <json>
退出码：动作逐步一致且无报错为 0，否则 1。
"""
import argparse
import base64
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import data as dsmod  # noqa: E402
import proto_client as pc  # noqa: E402


def pct(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q)) if len(a) else None


def lat_stats(a):
    return {"n": int(len(a)), "p50": pct(a, 50), "p95": pct(a, 95), "p99": pct(a, 99),
            "max": float(max(a)) if len(a) else None, "mean": float(np.mean(a)) if len(a) else None}


def summarize(tag_steps, cpp, wrap):
    """cpp / wrap：dict(actions, probs, infer_ms, rtt_ms)，长度同 tag_steps。"""
    ca, wa = np.array(cpp["actions"]), np.array(wrap["actions"])
    cp, wp = np.array(cpp["probs"]), np.array(wrap["probs"])
    diff = np.flatnonzero(ca != wa)
    pd = np.abs(cp - wp)
    return {
        "steps": int(len(ca)),
        "actions_identical": bool(len(diff) == 0),
        "action_mismatch_steps": [{"i": int(i), "at": tag_steps[i], "cpp": int(ca[i]), "wrap": int(wa[i]),
                                   "cpp_probs": cp[i].tolist(), "wrap_probs": wp[i].tolist()} for i in diff[:20]],
        "action_mismatch_count": int(len(diff)),
        "action_histogram_cpp": np.bincount(ca, minlength=4).tolist(),
        "max_abs_prob_diff": float(pd.max()), "mean_abs_prob_diff": float(pd.mean()),
        "infer_ms": {"cpp": lat_stats(cpp["infer_ms"]), "wrap": lat_stats(wrap["infer_ms"])},
        "rtt_ms": {"cpp": lat_stats(cpp["rtt_ms"]), "wrap": lat_stats(wrap["rtt_ms"])},
    }


def replay_dataset(args):
    ds = dsmod.Uavr(args.data, drop_collision=True)
    keep = {s: set(ds.splits[s].tolist()) for s in args.splits}
    frames, tags = [], []
    for s in args.splits:
        for eid, start, count, _src in ds.episode_spans(s):
            for g in range(start, start + count):
                if g in keep[s]:
                    frames.append(g)
                    tags.append(f"{s}/{eid}#{g - start}")
    x, _ = ds.take(np.array(frames, dtype=np.int64))
    out = {}
    clients = {"cpp": pc.InferClient(args.cpp_bin, args.ckpt), "wrap": pc.InferClient(args.wrap_bin, args.ckpt)}
    try:
        if clients["cpp"].hello != clients["wrap"].hello:
            raise RuntimeError(f"hello 不一致: {clients['cpp'].hello} vs {clients['wrap'].hello}")
        hello = clients["cpp"].hello
        if hello["in_c"] != 3 or hello.get("prev_action_dim", 0) or hello.get("text_dim", 0):
            raise SystemExit("dataset 模式只回放单帧、无 prev-action、无文本的模型；其余结构用 pngs 模式（StudentPolicy 管缓存）")
        rec = {k: {"actions": [], "probs": [], "infer_ms": [], "rtt_ms": []} for k in clients}
        for i in range(len(frames)):
            payload = pc.make_payload(x[i])
            for k in (("cpp", "wrap") if i % 2 == 0 else ("wrap", "cpp")):   # 交替先后，两边的缓存与调度条件对等
                t = time.perf_counter()
                r = clients[k].step(payload)
                rtt = (time.perf_counter() - t) * 1000.0
                rec[k]["actions"].append(r["action"])
                rec[k]["probs"].append(r["probs"])
                rec[k]["infer_ms"].append(r["infer_ms"])
                rec[k]["rtt_ms"].append(rtt)
        rcs = {k: c.close() for k, c in clients.items()}
        clients = {}
        out.update({"hello": hello, "close": {k: list(v) for k, v in rcs.items()}})
    finally:
        for c in clients.values():
            c.proc.kill()
    res = summarize(tags, rec["cpp"], rec["wrap"])
    res.update(mode="dataset", ckpt=args.ckpt, data=args.data, splits=args.splits, **out)
    return res


def png_msg(path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    raw = im.tobytes()
    return {"width": im.width, "height": im.height, "step": im.width * 3, "pixelFormatType": "RGB_INT8",
            "data": base64.b64encode(raw).decode("ascii")}


def replay_pngs(args):
    sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
    from student import StudentPolicy
    dirs = list(args.episode_dirs or [])
    if args.episodes_root:
        dirs += sorted(d for d in glob.glob(os.path.join(args.episodes_root, "*")) if os.path.isdir(d))
    if args.limit:
        dirs = dirs[:args.limit]
    if not dirs:
        raise SystemExit("没有可回放的 episode 目录")
    pols = {"cpp": StudentPolicy(args.ckpt, infer_bin=args.cpp_bin, text_ids=args.text),
            "wrap": StudentPolicy(args.ckpt, infer_bin=args.wrap_bin, text_ids=args.text)}
    rtt = {k: [] for k in pols}
    for k, p in pols.items():     # 往返耗时：包住 _exchange，只记 act 阶段
        orig = p._exchange

        def timed(payload, stage, _o=orig, _k=k):
            t = time.perf_counter()
            r = _o(payload, stage)
            if stage == "act":
                rtt[_k].append((time.perf_counter() - t) * 1000.0)
            return r
        p._exchange = timed
    rec = {k: {"actions": [], "probs": [], "infer_ms": [], "rtt_ms": rtt[k]} for k in pols}
    tags, per_episode = [], []
    try:
        for d in dirs:
            for p in pols.values():
                p.reset()
            files = sorted(glob.glob(os.path.join(d, "rgb_*.png")))
            n0 = len(tags)
            for j, f in enumerate(files):
                msg = png_msg(f)
                for k in (("cpp", "wrap") if j % 2 == 0 else ("wrap", "cpp")):
                    a, ms, pr = pols[k].act(msg)
                    rec[k]["actions"].append(a)
                    rec[k]["probs"].append(pr)
                    rec[k]["infer_ms"].append(ms)
                tags.append(f"{os.path.basename(d)}#{j}")
            ca = rec["cpp"]["actions"][n0:]
            wa = rec["wrap"]["actions"][n0:]
            per_episode.append({"episode": os.path.basename(d), "steps": len(files), "identical": ca == wa})
        info = {k: {"in_c": p.in_c, "prev_dim": p.prev_dim, "text_dim": p.text_dim, "aux_classes": p.aux_classes}
                for k, p in pols.items()}
    finally:
        for p in pols.values():
            try:
                p.close()
            except RuntimeError as exc:
                sys.stderr.write(f"close: {exc}\n")
    # StudentPolicy 返回的 rtt 列表按调用顺序追加，与 rec 的交替顺序无关，长度都等于总步数
    res = summarize(tags, rec["cpp"], rec["wrap"])
    res.update(mode="pngs", ckpt=args.ckpt, episodes=per_episode, policy_info=info,
               text_ids=args.text)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    for name in ("dataset", "pngs"):
        p = sub.add_parser(name)
        p.add_argument("--ckpt", required=True)
        p.add_argument("--cpp-bin", required=True)
        p.add_argument("--wrap-bin", required=True)
        p.add_argument("--out", default=None)
        if name == "dataset":
            p.add_argument("--data", required=True)
            p.add_argument("--splits", default="val,test", type=lambda s: s.split(","))
        else:
            p.add_argument("--episodes-root", default=None)
            p.add_argument("--episode-dirs", nargs="*", default=None)
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--text", default=None, type=lambda s: [int(t) for t in s.split(",")] if s else None)
    args = ap.parse_args(argv)
    res = replay_dataset(args) if args.mode == "dataset" else replay_pngs(args)
    res["pass"] = bool(res["actions_identical"])
    text = json.dumps(res, indent=1, ensure_ascii=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text if len(text) < 20000 else text[:20000] + "\n...(截断显示，完整结果见 --out)")
    return 0 if res["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
