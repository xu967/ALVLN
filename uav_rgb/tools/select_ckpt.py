#!/usr/bin/env python3
"""逐轮 ckpt 选模表：对 run 目录里每个 ckpt_epNN.bin（uavrgb_train --save-every-epoch）用真实 uavrgb_infer
跑 val 帧，输出 EVAL_STANDARD §1.1 四件套 + 两列闭环相关的诊断列，并给出显式选模函数 select_epoch 的选择。

运行（仓库根；--run 必须是 --save-every-epoch 训出的目录）：
  .venv-tools/bin/python tools/select_ckpt.py --run <run_dir> --data <ds>/dataset.uavr --out <dir> \
      [--infer-bin <uavrgb_infer>] [--split val] [--no-check-metrics]
输出：<out>/epochs.csv（每轮一行）、<out>/epochs.json（表 + select_epoch 的选择与理由）、<out>/preds_epNN.npy（逐帧预测）。

列（val 上）：acc、macro_f1、balanced_acc、recall_{safe,fwd,l,r}（四件套）；lr_recall_gap = |recall_L − recall_R|；
empty_yaw_rate = 视锥内无柱的帧里预测 YAW 的比例（闭环「空画面乱转」的离线代理，DISTILL §1.3；柱子真值来自 meta
episodes[].source_dir 的 scene.json，位姿来自 actions.jsonl 的 pose_before/yaw_before，可见性按 teacher.visible_pillar_side 现算）；
flip_rate = 与上一轮预测不同的帧占比（逐轮震荡，DISTILL §2.4）。

帧来源：直接读 dataset.uavr（与训练器同一份 val 样本、同一份已缩放像素；run/config.json 的 drop_collision 决定是否
剔除碰撞步，与训练器 LoadOptions 同义），不重解 PNG。自检：每轮的 macro_f1 必须与 run/metrics.jsonl 同轮 validation
事件一致（输入路不同就会对不上），默认开，不一致 rc=1。
"""
import argparse
import csv
import glob
import json
import math
import os
import re
import struct
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "sim", "tools"))
import student  # noqa: E402
import teacher  # noqa: E402

# ───────────────────────────── 配置区（select_epoch 的阈值，改这里） ─────────────────────────────
EMPTY_YAW_MARGIN = 0.05   # 空画面 YAW 率最多比「空画面帧的标签 YAW 占比」高 5 个百分点：学生不能比教师更爱在空画面里转
                          # （DISTILL §1.3：0417 ep30 84–89% vs 标签 24% 就是闭环 seed 127/395 乱飞的离线形态）
LR_GAP_MAX = 0.15         # L/R 召回差上限：auto 类权重下逐轮均值 0.19（DISTILL §2.4），一侧偏置会让闭环 seed 256 式原地打转
SMOOTH_HALF = 1           # macro_f1 按 ±1 轮取均值再比大小：单轮最大值是噪声序列上的幸运轮（§2.4 末 10 轮 sd 0.014）
NAMES = ("SAFE", "FWD", "YAW_L", "YAW_R")
YAW = (2, 3)
# ──────────────────────────────────────────────────────────────────────────────────────────────

HEADER = struct.Struct("<4sHHIHHBBBBQQQI")   # tools/DATASET_FORMAT.md §1.1
AUX_HEADER = struct.Struct("<8sQIHHBBH")      # §6.1 前 28 B；整头 128 B
AUX_HEADER_SIZE = 128
CKPT_HEAD = struct.Struct("<8sIIiiiiIII")     # src/checkpoint.cpp 定长头 44 B
CKPT_CONTRACT_BYTES = 113                     # v5 训练契约 113 B


class Uavr:
    """dataset.uavr 的只读视图：images 为 (N,C,H,W) 的 memmap，labels (N,)，meta 为 dict。"""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            magic, ver, hs, n, h, w, c, idt, ldt, _res, io, lo, mo, ml = HEADER.unpack(f.read(HEADER.size))
            if magic != b"UAVR" or ver != 1 or hs != HEADER.size:
                raise ValueError(f"{path}: 不是 UAVR v1（magic={magic!r} version={ver}）")
            f.seek(mo)
            self.meta = json.loads(f.read(ml).decode("utf-8"))
        self.n, self.h, self.w, self.c = n, h, w, c
        self.images = np.memmap(path, dtype=np.uint8, mode="r", offset=io, shape=(n, c, h, w))
        self.labels = np.fromfile(path, dtype=np.uint8, count=n, offset=lo)
        flags = self.meta.get("collision_flags")
        self.collision = (np.frombuffer(flags.encode(), dtype=np.uint8) == ord("1")) if flags else np.zeros(n, bool)

    def hwc_bytes(self, i):
        """样本 i 的 HWC RGB 字节：uavrgb_infer 的输入布局（见 src/infer.cpp），与 student.py 发 PIL 图一致。"""
        return np.ascontiguousarray(self.images[i].transpose(1, 2, 0)).tobytes()

    def split_indices(self, split, drop_collision):
        """与 src/dataset.cpp 同义：episodes[].split 选区间，drop_collision 时剔除 collision_flags=='1' 的样本。"""
        idx = []
        for ep in self.meta["episodes"]:
            if ep["split"] == split:
                idx.extend(range(ep["sample_start"], ep["sample_start"] + ep["sample_count"]))
        idx = np.asarray(idx, dtype=np.int64)
        if drop_collision:
            idx = idx[~self.collision[idx]]
        return idx


def read_aux(path):
    """UAVRAUX1 侧车（DATASET_FORMAT.md §6）→ dict(n, prev_index int64[N], prev_action u8[N], depth u8[N,dh,dw] 或 None,
    grid, envelope_radius_m)；文件不存在返回 None。"""
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        head = f.read(AUX_HEADER_SIZE)
        magic, n, flags, dh, dw, _k, _unk, _res = AUX_HEADER.unpack(head[:AUX_HEADER.size])
        if magic != b"UAVRAUX1":
            raise ValueError(f"{path}: 不是 UAVRAUX1")
        env_r = struct.unpack_from("<f", head, 120)[0]
        prev_index = np.frombuffer(f.read(4 * n), dtype="<u4").astype(np.int64)
        prev_action = np.frombuffer(f.read(n), dtype=np.uint8).copy()
        depth = np.frombuffer(f.read(n * dh * dw), dtype=np.uint8).reshape(n, dh, dw).copy() if flags & 1 else None
    return {"n": n, "prev_index": prev_index, "prev_action": prev_action, "depth": depth,
            "grid": (dh, dw), "envelope_radius_m": env_r}


def sample_poses(meta, drop_collision_steps):
    """每个样本的 (x, y, yaw_deg, step) 与所属 episode 的柱子：按 preprocess.load_episode 的规则复算样本→step
    （按 step 排序；打包时 --drop-collision 则跳过 collision 行），并用 sample_count 核对。source_dir 不在了返回 None。"""
    n = meta["num_samples"]
    pose = np.full((n, 3), np.nan)
    step = np.full(n, -1, dtype=np.int64)
    pillars = [None] * n
    for ep in meta["episodes"]:
        d = ep.get("source_dir")
        if not d or not os.path.isfile(os.path.join(d, "actions.jsonl")):
            return None
        with open(os.path.join(d, "actions.jsonl"), encoding="utf-8") as f:
            rows = sorted((json.loads(l) for l in f if l.strip()), key=lambda r: int(r["step"]))
        if drop_collision_steps:
            rows = [r for r in rows if not r.get("collision")]
        if len(rows) != ep["sample_count"]:
            raise ValueError(f"{d}: actions.jsonl 复算得 {len(rows)} 个样本，meta sample_count={ep['sample_count']}")
        with open(os.path.join(d, "scene.json"), encoding="utf-8") as f:
            pil = json.load(f).get("pillars") or []
        for i, r in enumerate(rows):
            g = ep["sample_start"] + i
            pose[g] = (r["pose_before"]["x"], r["pose_before"]["y"], r["yaw_before"])
            step[g] = int(r["step"])
            pillars[g] = pil
    return pose, step, pillars


def empty_frame_mask(pose, pillars, idx):
    """idx 里哪些帧「视锥内一根柱都没有」（teacher.visible_pillar_side 返回 None）。
    传 Python float：teacher 里 (b > 0) - (b < 0) 对 numpy 标量会报 bool 相减错误。"""
    return np.asarray([teacher.visible_pillar_side(float(pose[g, 0]), float(pose[g, 1]), float(pose[g, 2]),
                                                   pillars[g]) is None for g in idx], dtype=bool)


class Infer:
    """uavrgb_infer 常驻进程（经 student.StudentPolicy 起、同一条输入路）；predict 直接喂已缩放的 HWC 字节。"""

    def __init__(self, ckpt, infer_bin):
        self.pol = student.StudentPolicy(ckpt, infer_bin)

    def predict(self, cur, prev=None, prev_action=0):
        pol = self.pol
        payload = cur
        if pol.in_c == 6:
            payload += prev if prev is not None else cur   # 首帧口径：上一帧 = 当前帧（student.py / preprocess 同）
        if pol.prev_dim:
            payload += bytes([int(prev_action)])
        # act() 吃 gz 消息并自己缩放；这里的帧已是数据集里的缩放结果，所以直接走底层交换。
        obj = json.loads(pol._exchange(payload, "act"))
        return int(obj["action"]), obj["probs"]

    def close(self):
        self.pol.close()


def predict_indices(infer, ds, idx, prev_index=None, prev_action=None):
    """对样本下标 idx 逐帧推理；双帧模型按 prev_index 取上一帧。返回 (pred int[k], probs float[k,4])。"""
    pred = np.zeros(len(idx), dtype=np.int64)
    probs = np.zeros((len(idx), 4), dtype=np.float64)
    for k, g in enumerate(idx):
        prev = ds.hwc_bytes(int(prev_index[g])) if (prev_index is not None and infer.pol.in_c == 6) else None
        pa = int(prev_action[g]) if prev_action is not None else 0
        pred[k], probs[k] = infer.predict(ds.hwc_bytes(int(g)), prev, pa)
    return pred, probs


# ---- 指标：与 src/eval_metrics.cpp 同定义（sklearn 口径；无真值样本的类召回记 0，P+R=0 的类 F1 记 0） ----
def confusion(y, pred, c=4):
    m = np.zeros((c, c), dtype=np.int64)
    np.add.at(m, (y, pred), 1)
    return m


def metrics_from_confusion(m):
    tp = np.diag(m).astype(float)
    row, col = m.sum(1).astype(float), m.sum(0).astype(float)
    recall = np.where(row > 0, tp / np.maximum(row, 1), 0.0)
    prec = np.where(col > 0, tp / np.maximum(col, 1), 0.0)
    f1 = np.where(prec + recall > 0, 2 * prec * recall / np.maximum(prec + recall, 1e-300), 0.0)
    return {"acc": float(tp.sum() / max(m.sum(), 1)), "macro_f1": float(f1.mean()),
            "balanced_acc": float(recall.mean()), "recall": [float(r) for r in recall]}


def epoch_row(epoch, y, pred, empty, prev_pred):
    """一轮的表行：四件套 + L/R 召回差 + 空画面 YAW 率 + 与上一轮的翻转率。"""
    m = metrics_from_confusion(confusion(y, pred))
    is_yaw_pred, is_yaw_lab = np.isin(pred, YAW), np.isin(y, YAW)
    row = {"epoch": epoch, "n": int(len(y)), **{k: m[k] for k in ("acc", "macro_f1", "balanced_acc")},
           "recall_safe": m["recall"][0], "recall_fwd": m["recall"][1], "recall_l": m["recall"][2],
           "recall_r": m["recall"][3], "lr_recall_gap": abs(m["recall"][2] - m["recall"][3]),
           "empty_n": int(empty.sum()) if empty is not None else 0,
           "empty_yaw_rate": float(is_yaw_pred[empty].mean()) if empty is not None and empty.any() else float("nan"),
           "empty_label_yaw_rate": float(is_yaw_lab[empty].mean()) if empty is not None and empty.any() else float("nan"),
           "flip_rate": float((pred != prev_pred).mean()) if prev_pred is not None else float("nan"),
           "flip_rate_yaw_label": (float((pred != prev_pred)[is_yaw_lab].mean())
                                   if prev_pred is not None and is_yaw_lab.any() else float("nan"))}
    return row


def select_epoch(rows, empty_yaw_margin=EMPTY_YAW_MARGIN, lr_gap_max=LR_GAP_MAX, smooth_half=SMOOTH_HALF):
    """显式选模规则：先按两条闭环相关的约束过滤，再在平滑后的 macro_f1 上取最大。返回 (epoch 或 None, 说明 dict)。

    依据（DISTILL 2026-11-07）：§2.4 val macro_f1 逐轮在 0.11–0.40 间跳、末 10 轮 sd 0.014，严格 argmax 选到的是
    噪声序列上的幸运轮，所以用 ±smooth_half 轮的均值代替单轮值；§1.3 空画面 YAW 率（ep14 26–32% → ep30 84–89%）
    与 §2.4 L/R 召回差（均值 0.19）是 val macro_f1 完全不度量、却直接对应闭环失败形态（乱飞 / 原地打转）的两个量，
    作为硬约束而不是加权项：加权会把两种错误折成一个分数再滑动工作点。
    约束全空时放宽为「只看平滑 macro_f1」并在说明里标明，不静默。闭环仍是唯一裁判（BRIEF_ALGO §0.1）。
    """
    by_ep = {r["epoch"]: r for r in rows}
    eps = sorted(by_ep)

    def smooth(e):
        vals = [by_ep[k]["macro_f1"] for k in range(e - smooth_half, e + smooth_half + 1) if k in by_ep]
        return sum(vals) / len(vals)

    def ok(r):
        lim = r["empty_label_yaw_rate"] + empty_yaw_margin
        empty_ok = math.isnan(r["empty_yaw_rate"]) or r["empty_yaw_rate"] <= lim
        return empty_ok and r["lr_recall_gap"] <= lr_gap_max

    pool = [e for e in eps if ok(by_ep[e])]
    relaxed = not pool
    if relaxed:
        pool = eps
    if not pool:
        return None, {"reason": "没有任何轮"}
    best = max(pool, key=lambda e: (smooth(e), e))   # 平滑分相同取靠后的轮：更多步数、离初始化更远
    return best, {"candidates": pool, "relaxed_constraints": relaxed, "smoothed_macro_f1": smooth(best),
                  "raw_macro_f1": by_ep[best]["macro_f1"], "empty_yaw_rate": by_ep[best]["empty_yaw_rate"],
                  "lr_recall_gap": by_ep[best]["lr_recall_gap"],
                  "argmax_raw_macro_f1_epoch": max(eps, key=lambda e: by_ep[e]["macro_f1"]),
                  "rule": f"empty_yaw_rate <= label_rate + {empty_yaw_margin}; lr_recall_gap <= {lr_gap_max}; "
                          f"argmax mean(macro_f1[e-{smooth_half}..e+{smooth_half}])"}


def epoch_ckpts(run_dir):
    out = []
    for p in glob.glob(os.path.join(run_dir, "ckpt_ep*.bin")):
        m = re.fullmatch(r"ckpt_ep(\d+)\.bin", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def metrics_val_f1(run_dir):
    """run/metrics.jsonl 里训练期 validation(split=val) 事件的 macro_f1，按轮序；没有文件返回 []。"""
    path = os.path.join(run_dir, "metrics.jsonl")
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if '"event":"validation"' in line and '"split":"val"' in line:
                out.append(json.loads(line)["macro_f1"])
    return out


def read_ckpt_drop_collision(path):
    """按 src/checkpoint.cpp:read_v5_contract 读 ckpt 头的训练契约位 → (has_drop_collision, drop_collision)。
    v2..v4 或全零占位契约返回 (False, False)；损坏或非法头抛 ValueError。"""
    with open(path, "rb") as f:
        raw = f.read(CKPT_HEAD.size)
        if len(raw) != CKPT_HEAD.size:
            raise ValueError(f"{path}: checkpoint 头不足 {CKPT_HEAD.size} B")
        magic, ver, _np, _h, _w, _c, _nc, _at, _ep, nw = CKPT_HEAD.unpack(raw)
        if magic != b"UAVRCK01" or ver < 2 or ver > 6 or nw > 64:
            raise ValueError(f"{path}: 非法 UAVRCK01 头（magic={magic!r} version={ver}）")
        if ver < 5:
            return False, False
        f.seek(4 * nw + 8, os.SEEK_CUR)      # class_weights + v3 best_macro_f1 + v4 text_dim
        contract = f.read(CKPT_CONTRACT_BYTES)
    if len(contract) != CKPT_CONTRACT_BYTES:
        raise ValueError(f"{path}: v5 训练契约被截断（{len(contract)} < {CKPT_CONTRACT_BYTES}）")
    distill_alpha, lr, _seed, batch = struct.unpack_from("<ffIi", contract, 65)
    if not (math.isfinite(distill_alpha) and math.isfinite(lr)):
        raise ValueError(f"{path}: 训练契约含非有限值")
    present = not (lr == 0.0 and batch == 0 and contract[:32] == b"\x00" * 32)
    return present, bool(contract[32] & 1)


def resolve_drop_collision(run_dir, ckpts):
    """对齐 src/trainer_eval.cpp:194-212（C6）：先读 config.resume.json 再读 config.json，并核对 ckpt v5 训练契约；
    两处矛盾或两处均无记录时抛 ValueError 拒绝静默回落。"""
    has_cfg, cfg_drop, cfg_used = False, False, None
    for name in ("config.resume.json", "config.json"):
        p = os.path.join(run_dir, name)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict) and "drop_collision" in cfg:
                if not isinstance(cfg["drop_collision"], bool):
                    raise ValueError(f"{p}: drop_collision 必须是布尔值")
                has_cfg, cfg_drop, cfg_used = True, cfg["drop_collision"], p
            break
    has_ck, ck_drop = False, False
    seen = set()
    for _ep, path in ckpts:
        present, val = read_ckpt_drop_collision(path)
        if present:
            has_ck, ck_drop = True, val
            seen.add(val)
    if len(seen) > 1:
        raise ValueError(f"{run_dir}: 不同轮 ckpt_ep*.bin 的训练契约 drop_collision 不一致")
    if has_cfg and has_ck and cfg_drop != ck_drop:
        raise ValueError(f"{cfg_used} 的 drop_collision({cfg_drop}) 与 checkpoint 训练契约({ck_drop}) 矛盾")
    if not has_cfg and not has_ck:
        raise ValueError(f"{run_dir} 无含 drop_collision 的 config.json，且 checkpoint 无训练契约，拒绝静默回落")
    return cfg_drop if has_cfg else ck_drop


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="--save-every-epoch 训出的 run 目录（含 ckpt_epNN.bin、config.json）")
    ap.add_argument("--data", required=True, help="dataset.uavr（与训练同一份）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--infer-bin", default=student.DEFAULT_INFER)
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--no-check-metrics", action="store_true",
                    help="不与 run/metrics.jsonl 的同轮 macro_f1 对账（数据不是训练那份时用）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ckpts = epoch_ckpts(args.run)
    if not ckpts:
        sys.exit(f"{args.run} 没有 ckpt_ep*.bin（训练要加 --save-every-epoch）")
    try:
        drop = resolve_drop_collision(args.run, ckpts)
    except (ValueError, OSError) as exc:
        sys.exit(str(exc))
    ds = Uavr(args.data)
    idx = ds.split_indices(args.split, drop)
    if not len(idx):
        sys.exit(f"{args.split} split 没有样本")
    y = ds.labels[idx].astype(np.int64)
    poses = sample_poses(ds.meta, bool((ds.meta.get("collision_filter") or {}).get("drop_collision_steps")))
    empty = None
    if poses is None:
        print("警告: meta episodes[].source_dir 不可读，空画面列记 NaN", file=sys.stderr)
    else:
        teacher.set_vehicle("e1_mini", "base")   # 只影响柱子半宽的缺省；可见性本身用 scene.json 的 size
        empty = empty_frame_mask(poses[0], poses[2], idx)
    aux = read_aux(os.path.join(os.path.dirname(args.data), "dataset.aux"))
    prev_index, prev_action = (aux["prev_index"], aux["prev_action"]) if aux else (None, None)

    rows, prev_pred = [], None
    for epoch, path in ckpts:
        inf = Infer(path, args.infer_bin)
        pred, probs = predict_indices(inf, ds, idx, prev_index, prev_action)
        inf.close()
        np.save(os.path.join(args.out, f"preds_ep{epoch:02d}.npy"), np.c_[idx, y, pred, probs])
        rows.append(epoch_row(epoch, y, pred, empty, prev_pred))
        prev_pred = pred
        r = rows[-1]
        print(f"ep{epoch:02d} acc={r['acc']:.4f} mF1={r['macro_f1']:.4f} bal={r['balanced_acc']:.4f} "
              f"recall={[round(v, 3) for v in (r['recall_safe'], r['recall_fwd'], r['recall_l'], r['recall_r'])]} "
              f"lr_gap={r['lr_recall_gap']:.3f} empty_yaw={r['empty_yaw_rate']:.3f} flip={r['flip_rate']:.3f}", flush=True)

    check = {"checked": False}
    if not args.no_check_metrics and args.split == "val":
        ref = metrics_val_f1(args.run)
        diffs = [abs(r["macro_f1"] - ref[r["epoch"] - 1]) for r in rows if r["epoch"] - 1 < len(ref)]
        check = {"checked": bool(diffs), "max_abs_diff_macro_f1_vs_metrics_jsonl": max(diffs) if diffs else None}
        if diffs and max(diffs) > 1e-5:
            print(f"自检失败: 与 metrics.jsonl 的 val macro_f1 最大差 {max(diffs):.3g}（输入路不一致？）", file=sys.stderr)
            rc = 1
        else:
            rc = 0
    else:
        rc = 0

    chosen, why = select_epoch(rows)
    best_lines = [l for l in open(os.path.join(args.run, "metrics.jsonl"), encoding="utf-8")
                  if "best checkpoint: epoch" in l] if os.path.isfile(os.path.join(args.run, "metrics.jsonl")) else []
    best_epoch = int(re.search(r"epoch (\d+)", best_lines[-1]).group(1)) if best_lines else None
    with open(os.path.join(args.out, "epochs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "epochs.json"), "w", encoding="utf-8") as f:
        json.dump({"run": args.run, "data": args.data, "split": args.split, "n": int(len(idx)), "drop_collision": drop,
                   "trainer_best_epoch": best_epoch, "selected_epoch": chosen, "selection": why,
                   "metrics_check": check, "rows": rows}, f, indent=1, ensure_ascii=False)
    print(f"trainer ckpt_best = epoch {best_epoch}; select_epoch → {chosen} ({why.get('rule')})"
          f"{'; 约束全空已放宽' if why.get('relaxed_constraints') else ''}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
