#!/usr/bin/env python3
"""在“决定性帧”上量文本对动作的影响（L/I/S 离线指标会被非决定性帧稀释）。

决定性帧 = 把关系词反过来（v5 起两根目标柱同色）后 teacher 动作会变的步。只在这些帧上，
一个真的用文本的模型应该：正确指令 → 原动作，反向指令 → 另一动作。

用法：
  python3 sim/tools/probe_text_effect.py --src <采集根> --ckpt <ckpt.bin> \
      --split test --out <json>
"""
import argparse
import glob
import json
import os
import subprocess
import sys

from PIL import Image

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(TOOLS))
sys.path.insert(0, TOOLS)
import instruction as ins  # noqa: E402
import teacher  # noqa: E402
from make_counterfactual import alt_target  # noqa: E402

DEFAULT_INFER = os.path.join(REPO, "src", "build", "uavrgb_infer")


def load_rows(path):
    return {int(r["episode_id"]): r for r in map(json.loads, open(path, encoding="utf-8"))}


def decisive_frames(src, split, max_frames=0):
    """返回 ([(png_path, 原动作, 反向指令下的动作, episode_id)], {eid: text_ids})。"""
    rows, alt_ids = [], {}
    for d in sorted(glob.glob(os.path.join(src, split, "*"))):
        if not os.path.isdir(d):
            continue
        eid = int(os.path.basename(d))
        scene = json.load(open(os.path.join(d, "scene.json"), encoding="utf-8"))
        # 与 make_decisive / make_counterfactual 用同一份定义，避免探针和训练集
        # 对「决定性」各算一套。
        alt_scene = alt_target(scene)
        if alt_scene is None:
            continue
        alt_relation, wp = alt_scene
        alt_ids[eid] = ins.tokenize(
            ins.make_instruction(scene["target_color"], alt_relation))
        stop = scene.get("stop_radius") or 0.5
        for row in map(json.loads, open(os.path.join(d, "actions.jsonl"), encoding="utf-8")):
            pb, yb = row.get("pose_before"), row.get("yaw_before")
            if pb is None or yb is None:
                continue
            alt = teacher.plan_action_global(
                pb["x"], pb["y"], yb, scene["pillars"], waypoint=wp,
                stop_at_waypoint=True, stop_radius=stop)
            if alt == row["action_expert"]:
                continue
            png = os.path.join(d, row["frame"])
            if not os.path.isfile(png):
                continue
            rows.append((png, row["action_expert"], alt, eid))
            if max_frames and len(rows) >= max_frames:
                return rows, alt_ids
    return rows, alt_ids


def infer_actions(infer_bin, ckpt, items, text_by_ep):
    """按 episode 分组起进程（一个进程一份文本），返回每帧 action/probs。"""
    acts, probs = [], []
    groups = {}
    for idx, item in enumerate(items):
        groups.setdefault(item[3], []).append(idx)
    for eid, idxs in groups.items():
        row = text_by_ep.get(eid)
        if row is None or "text_ids" not in row:
            raise SystemExit(f"侧车里没有 episode {eid} 的指令，拒绝用空文本代替")
        ids = row["text_ids"] or []
        cmd = [infer_bin, "--ckpt", ckpt]
        if ids:
            cmd += ["--text", ",".join(str(int(t)) for t in ids)]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.DEVNULL)
        hello = json.loads(p.stdout.readline())
        h, w = hello["in_h"], hello["in_w"]
        for i in idxs:
            im = Image.open(items[i][0]).convert("RGB").resize((w, h), Image.BILINEAR)
            p.stdin.write(im.tobytes())
            p.stdin.flush()
            out = json.loads(p.stdout.readline())
            acts.append((i, out["action"]))
            probs.append((i, out["probs"]))
        p.stdin.close()
        p.wait()
    acts.sort()
    probs.sort()
    return [a for _, a in acts], [p for _, p in probs]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--infer-bin", default=DEFAULT_INFER)
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    items, alt_ids = decisive_frames(args.src, args.split, args.max_frames)
    if not items:
        sys.exit("没有决定性帧（缺 scene 颜色 / relation / waypoint / 位姿）")
    correct = load_rows(os.path.join(args.src, "ds", "instructions.jsonl"))
    # 反向指令就地生成：以前读 <src>/ds-s/instructions.jsonl，那个目录没有
    # 任何工具会产出，纯靠手搓；改成和 alt_target 同源就不会再漂。
    alt = {eid: {"text_ids": ids} for eid, ids in alt_ids.items()}
    ca = infer_actions(args.infer_bin, args.ckpt, items, correct)[0]
    aa, pa = infer_actions(args.infer_bin, args.ckpt, items, alt)
    n = len(items)
    report = {
        "schema": "uavrgb.text_probe.v1",
        "split": args.split, "frames": n, "ckpt": os.path.abspath(args.ckpt),
        "acc_correct": round(sum(x == a for x, (_, a, b, _) in zip(ca, items)) / n, 4),
        "acc_alt": round(sum(x == b for x, (_, a, b, _) in zip(aa, items)) / n, 4),
        "follow": round(sum(x == a and y == b for x, y, (_, a, b, _)
                            in zip(ca, aa, items)) / n, 4),
        "flip": round(sum(x != y for x, y in zip(ca, aa)) / n, 4),
    }
    print(json.dumps(report, ensure_ascii=False))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
